#!/usr/bin/env python3
"""
BI vs Bisimulation: Selection Geometry Ablation
=================================================
Tests whether bisimulation's advantage over BI comes from the metric
or from the fact that bisimulation distributes selections more broadly.

Conditions:
  1. BI-guided (original): selects lowest-BI layers with gap>=2
  2. BI-distributed: selects lowest-BI layers with gap>=4 (forces spread)
  3. Bisim-guided (original): selects lowest-removability layers with gap>=2
    4. Bisim-distributed: selects lowest-removability layers with gap>=4
    5. Bisim-clustered: selects lowest-removability layers from middle 12 layers only
    6. Random-distributed: random selection with gap>=4

If BI-distributed ≈ bisim-guided, the advantage is geometry, not metric.
If BI-distributed < bisim-guided, bisimulation genuinely selects better layers.
"""

import json, os, copy, logging
from itertools import combinations
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from wikitext_ppl import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MAX_WORDS,
    DEFAULT_STRIDE,
    build_wikitext2_eval_input,
    evaluate_perplexity_input_ids,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

REPORT_DIR = os.environ.get("REPORT_DIR", "reports/2026-03-31T20-59-53")
os.makedirs(REPORT_DIR, exist_ok=True)


def greedy_select(scores, n, num_layers=24, min_gap=2, layer_range=None):
    """Select n layers with lowest scores, respecting min_gap and optional layer_range."""
    if layer_range is None:
        layer_range = range(1, num_layers - 1)
    candidates = [(scores[i], i) for i in layer_range if 0 < i < num_layers - 1]
    candidates.sort()
    selected, used_positions = [], set()
    for _, layer in candidates:
        if len(selected) >= n:
            break
        # Check gap constraint
        too_close = False
        for s in selected:
            if abs(layer - s) < min_gap:
                too_close = True
                break
        if too_close:
            continue
        selected.append(layer)
    return sorted(selected)


def optimal_select(scores, n, num_layers=24, min_gap=4, layer_range=None):
    """Find the minimum-score subset of size n under a spacing constraint."""
    if layer_range is None:
        layer_range = range(1, num_layers - 1)
    candidates = [layer for layer in layer_range if 0 < layer < num_layers - 1]

    best_layers = None
    best_score = None
    for combo in combinations(candidates, n):
        if any(abs(combo[i] - combo[i - 1]) < min_gap for i in range(1, len(combo))):
            continue
        score = float(sum(scores[layer] for layer in combo))
        if best_score is None or score < best_score:
            best_score = score
            best_layers = list(combo)

    return best_layers or []


def random_distributed(n, num_layers=24, min_gap=4, seed=42):
    """Random selection with forced spread."""
    rng = np.random.default_rng(seed)
    candidates = list(range(1, num_layers - 1))
    rng.shuffle(candidates)
    selected = []
    for layer in candidates:
        if len(selected) >= n:
            break
        too_close = any(abs(layer - s) < min_gap for s in selected)
        if not too_close:
            selected.append(layer)
    return sorted(selected)


def remove_layers(model, remove_indices):
    model_copy = copy.deepcopy(model)
    keep = [i for i in range(len(model_copy.transformer.h)) if i not in remove_indices]
    model_copy.transformer.h = torch.nn.ModuleList([model_copy.transformer.h[i] for i in keep])
    model_copy.config.n_layer = len(keep)
    for i, layer in enumerate(model_copy.transformer.h):
        layer.attn.layer_idx = i
    return model_copy


def main():
    log.info("Loading GPT-2-Medium...")
    model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    model.eval()

    eval_input_ids, eval_protocol = build_wikitext2_eval_input(
        tokenizer,
        split="test",
        max_words=DEFAULT_MAX_WORDS,
    )
    log.info(
        "Using standardized WikiText-2 eval: words=%d tokens=%d max_length=%d stride=%d",
        eval_protocol["text_words"],
        eval_protocol["token_count"],
        DEFAULT_MAX_LENGTH,
        DEFAULT_STRIDE,
    )

    log.info("Computing baseline PPL...")
    baseline_ppl, baseline_eval = evaluate_perplexity_input_ids(
        model,
        eval_input_ids,
        max_length=DEFAULT_MAX_LENGTH,
        stride=DEFAULT_STRIDE,
    )
    eval_protocol = {**eval_protocol, **baseline_eval}
    log.info(f"Baseline PPL: {baseline_ppl:.2f}")

    # Load scores
    bi_path = "reports/2026-03-31T00-18-24/bi_score_comparison.json"
    with open(bi_path) as f:
        bi_data = json.load(f)
    bi_scores = np.array([bi_data["bi_scores"][str(i)] for i in range(24)])
    removability = np.array([bi_data["removability_scores"][str(i)] for i in range(24)])

    results = {
        "baseline_ppl": float(baseline_ppl),
        "eval_protocol": eval_protocol,
        "conditions": [],
    }

    for n in range(1, 6):
        log.info(f"\n=== Remove {n} layers ===")
        condition_results = {"n_removed": n, "methods": {}}

        methods = {
            "bi_original":       greedy_select(bi_scores, n, min_gap=2),
            "bi_distributed":    optimal_select(bi_scores, n, min_gap=4),
            "bisim_original":    greedy_select(removability, n, min_gap=2),
            "bisim_distributed": optimal_select(removability, n, min_gap=4),
            "bisim_clustered":   greedy_select(removability, n, min_gap=2,
                                               layer_range=range(6, 18)),
            "random_distributed": random_distributed(n, min_gap=4),
        }

        for method_name, layers in methods.items():
            if len(layers) < n:
                log.info(f"  {method_name}: only {len(layers)} layers feasible (need {n}), skipping")
                condition_results["methods"][method_name] = {
                    "layers": layers, "ppl": None, "delta_pct": None,
                    "note": f"only {len(layers)} feasible"
                }
                continue

            model_mod = remove_layers(model, layers)
            ppl, _ = evaluate_perplexity_input_ids(
                model_mod,
                eval_input_ids,
                max_length=DEFAULT_MAX_LENGTH,
                stride=DEFAULT_STRIDE,
            )
            delta = 100 * (ppl - baseline_ppl) / baseline_ppl
            log.info(f"  {method_name:20s}: layers={layers}, PPL={ppl:.2f} (+{delta:.1f}%)")
            del model_mod

            condition_results["methods"][method_name] = {
                "layers": layers,
                "ppl": float(ppl),
                "delta_pct": float(delta),
            }

        results["conditions"].append(condition_results)

    out_path = os.path.join(REPORT_DIR, "bi_geometry_ablation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved: {out_path}")

    # Summary table
    log.info(f"\n{'n':>3} | {'Method':>20} | {'Layers':>25} | {'PPL':>8} | {'Δ%':>7}")
    log.info("-" * 75)
    for cond in results["conditions"]:
        for method, data in cond["methods"].items():
            if data["ppl"] is not None:
                log.info(f"{cond['n_removed']:>3} | {method:>20} | {str(data['layers']):>25} | {data['ppl']:>8.2f} | {data['delta_pct']:>+7.1f}")
            else:
                log.info(f"{cond['n_removed']:>3} | {method:>20} | {str(data['layers']):>25} | {'N/A':>8} | {'N/A':>7}")
        log.info("-" * 75)


if __name__ == "__main__":
    main()
