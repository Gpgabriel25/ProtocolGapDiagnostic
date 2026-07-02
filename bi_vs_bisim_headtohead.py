#!/usr/bin/env python3
"""
Head-to-head: BI-guided vs Bisimulation-guided layer removal on PPL.
Uses the same evaluation protocol as compression_sweep.py.
"""

import argparse
import json
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy
import logging

from wikitext_ppl import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MAX_WORDS,
    DEFAULT_STRIDE,
    build_wikitext2_eval_input,
    evaluate_perplexity_input_ids,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CYCLE_ID = "2026-03-31T00-18-24"
DEFAULT_REPORT_DIR = f"reports/{CYCLE_ID}"
REPORT_DIR = os.environ.get("REPORT_DIR", DEFAULT_REPORT_DIR)
BI_SCORE_JSON = os.environ.get(
    "BI_SCORE_JSON",
    os.path.join(DEFAULT_REPORT_DIR, "bi_score_comparison.json"),
)


def greedy_select_bi(bi_scores, n, num_layers=24):
    """Select n layers with lowest BI scores (most removable), no two adjacent.
    Skip layers 0 and num_layers-1 (boundary layers)."""
    candidates = [(bi_scores[i], i) for i in range(1, num_layers - 1)]
    candidates.sort()  # ascending BI = most removable first
    
    selected = []
    used = set()
    for _, layer in candidates:
        if len(selected) >= n:
            break
        if layer - 1 in used or layer + 1 in used:
            continue
        selected.append(layer)
        used.add(layer)
    return sorted(selected)


def greedy_select_bisim(removability, n, num_layers=24):
    """Select n layers with lowest bisimulation removability."""
    candidates = [(removability[i], i) for i in range(1, num_layers - 1)]
    candidates.sort()
    
    selected = []
    used = set()
    for _, layer in candidates:
        if len(selected) >= n:
            break
        if layer - 1 in used or layer + 1 in used:
            continue
        selected.append(layer)
        used.add(layer)
    return sorted(selected)


def remove_layers(model, remove_indices):
    """Remove specified layers and fix layer_idx."""
    model_copy = copy.deepcopy(model)
    keep = [i for i in range(len(model_copy.transformer.h)) if i not in remove_indices]
    model_copy.transformer.h = torch.nn.ModuleList([model_copy.transformer.h[i] for i in keep])
    model_copy.config.n_layer = len(keep)
    for i, layer in enumerate(model_copy.transformer.h):
        layer.attn.layer_idx = i
    return model_copy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-max-words", type=int, default=DEFAULT_MAX_WORDS)
    parser.add_argument("--eval-max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--eval-stride", type=int, default=DEFAULT_STRIDE)
    args = parser.parse_args()

    log.info("Loading GPT-2-Medium...")
    model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    model.eval()

    eval_input_ids, eval_protocol = build_wikitext2_eval_input(
        tokenizer,
        split="test",
        max_words=args.eval_max_words,
    )
    log.info(
        "Using standardized WikiText-2 eval: words=%d tokens=%d max_length=%d stride=%d",
        eval_protocol["text_words"],
        eval_protocol["token_count"],
        args.eval_max_length,
        args.eval_stride,
    )
    
    # Compute baseline PPL
    log.info("Computing baseline PPL...")
    baseline_ppl, baseline_eval = evaluate_perplexity_input_ids(
        model,
        eval_input_ids,
        max_length=args.eval_max_length,
        stride=args.eval_stride,
    )
    eval_protocol = {**eval_protocol, **baseline_eval}
    log.info(f"Baseline PPL: {baseline_ppl:.2f}")
    
    # Load BI scores
    with open(BI_SCORE_JSON) as f:
        bi_data = json.load(f)
    bi_scores = np.array([bi_data["bi_scores"][str(i)] for i in range(24)])
    
    # Load bisimulation removability
    removability = np.array([bi_data["removability_scores"][str(i)] for i in range(24)])
    
    results = {
        "baseline_ppl": float(baseline_ppl),
        "eval_protocol": eval_protocol,
        "comparisons": [],
    }
    
    for n in range(1, 6):
        bi_layers = greedy_select_bi(bi_scores, n)
        bisim_layers = greedy_select_bisim(removability, n)
        
        log.info(f"\n=== Remove {n} layers ===")
        log.info(f"  BI-guided:     {bi_layers}")
        log.info(f"  Bisim-guided:  {bisim_layers}")
        
        # BI-guided removal
        model_bi = remove_layers(model, bi_layers)
        ppl_bi, _ = evaluate_perplexity_input_ids(
            model_bi,
            eval_input_ids,
            max_length=args.eval_max_length,
            stride=args.eval_stride,
        )
        delta_bi = 100 * (ppl_bi - baseline_ppl) / baseline_ppl
        log.info(f"  BI PPL: {ppl_bi:.2f} (+{delta_bi:.1f}%)")
        del model_bi
        
        # Bisim-guided removal
        model_bisim = remove_layers(model, bisim_layers)
        ppl_bisim, _ = evaluate_perplexity_input_ids(
            model_bisim,
            eval_input_ids,
            max_length=args.eval_max_length,
            stride=args.eval_stride,
        )
        delta_bisim = 100 * (ppl_bisim - baseline_ppl) / baseline_ppl
        log.info(f"  Bisim PPL: {ppl_bisim:.2f} (+{delta_bisim:.1f}%)")
        del model_bisim
        
        results["comparisons"].append({
            "n_removed": n,
            "bi_layers": bi_layers,
            "bisim_layers": bisim_layers,
            "bi_ppl": float(ppl_bi),
            "bi_delta_pct": float(delta_bi),
            "bisim_ppl": float(ppl_bisim),
            "bisim_delta_pct": float(delta_bisim),
        })
    
    out_path = os.path.join(REPORT_DIR, "bi_vs_bisim_ppl.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved: {out_path}")
    
    # Summary table
    log.info(f"\n{'n':>3} | {'BI Layers':>20} | {'BI PPL':>8} | {'BI Δ%':>7} | {'Bisim Layers':>20} | {'Bisim PPL':>8} | {'Bisim Δ%':>7}")
    log.info("-" * 95)
    for r in results["comparisons"]:
        log.info(f"{r['n_removed']:>3} | {str(r['bi_layers']):>20} | {r['bi_ppl']:>8.2f} | {r['bi_delta_pct']:>+7.1f} | {str(r['bisim_layers']):>20} | {r['bisim_ppl']:>8.2f} | {r['bisim_delta_pct']:>+7.1f}")


if __name__ == "__main__":
    main()
