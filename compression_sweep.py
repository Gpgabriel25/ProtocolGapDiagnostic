#!/usr/bin/env python3
"""
Multi-Layer Compression Sweep
==============================
Compare bisimilar-guided vs random vs anti-guided layer removal
for GPT-2-Medium. Measures perplexity degradation as 1-5 layers
are removed.
"""

import os
import sys
import json
import time
import random
import argparse
import logging
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from wikitext_ppl import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MAX_WORDS,
    DEFAULT_STRIDE,
    build_wikitext2_eval_input,
    evaluate_perplexity_input_ids,
)

CYCLE_ID = "2026-03-31T00-18-24"
REPORT_DIR = os.environ.get("REPORT_DIR", f"reports/{CYCLE_ID}")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def remove_layers(model, layer_indices):
    """Remove multiple layers from a GPT-2 model (returns modified model)."""
    layers = model.transformer.h
    keep = [i for i in range(len(layers)) if i not in layer_indices]
    new_layers = nn.ModuleList([layers[i] for i in keep])
    model.transformer.h = new_layers
    model.config.n_layer = len(new_layers)
    # Fix layer_idx for attention KV cache
    for i, layer in enumerate(new_layers):
        if hasattr(layer, "attn") and hasattr(layer.attn, "layer_idx"):
            layer.attn.layer_idx = i
    return model


def get_layer_removability(kl_data):
    """Compute per-layer removability score = min adjacent KL.
    Lower = more safely removable."""
    # Build adjacency map
    adj_kl = {}
    for entry in kl_data["pairs"]:
        a, b = entry["layer_a"], entry["layer_b"]
        if abs(a - b) == 1:
            adj_kl[(a, b)] = entry["mean_kl"]
            adj_kl[(b, a)] = entry["mean_kl"]

    n_layers = max(max(a, b) for a, b in adj_kl.keys()) + 1
    scores = {}
    for L in range(n_layers):
        neighbors = []
        if (L, L - 1) in adj_kl:
            neighbors.append(adj_kl[(L, L - 1)])
        if (L, L + 1) in adj_kl:
            neighbors.append(adj_kl[(L, L + 1)])
        scores[L] = min(neighbors) if neighbors else float("inf")
    return scores


def greedy_select(scores, n, forbidden=None):
    """Greedily select n layers by lowest score, no two adjacent."""
    if forbidden is None:
        forbidden = set()
    selected = []
    candidates = sorted(scores.items(), key=lambda x: x[1])
    used = set(forbidden)
    for layer, score in candidates:
        if layer in used:
            continue
        # Check adjacency with already selected
        if any(abs(layer - s) <= 0 for s in selected):
            pass  # same layer already selected shouldn't happen
        selected.append(layer)
        used.add(layer)
        used.add(layer - 1)  # block adjacent
        used.add(layer + 1)
        if len(selected) == n:
            break
    return selected


def greedy_select_worst(scores, n):
    """Greedily select n layers by HIGHEST score (least removable)."""
    inverted = {k: -v for k, v in scores.items()}
    # Exclude layer 0 and 23 to keep comparison fair (they're always worst)
    # Actually include them - that's the point of anti-guided
    return greedy_select(inverted, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-medium")
    parser.add_argument("--kl-data", default=f"reports/2026-03-30T15-15-07/distance_checkpoint.json")
    parser.add_argument("--max-remove", type=int, default=5)
    parser.add_argument("--random-trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-max-words", type=int, default=DEFAULT_MAX_WORDS)
    parser.add_argument("--eval-max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--eval-stride", type=int, default=DEFAULT_STRIDE)
    args = parser.parse_args()

    os.makedirs(REPORT_DIR, exist_ok=True)
    random.seed(args.seed)

    # Load KL data
    with open(args.kl_data) as f:
        kl_data = json.load(f)

    scores = get_layer_removability(kl_data)
    n_layers = len(scores)
    log.info("Per-layer removability scores:")
    for L in sorted(scores.keys()):
        log.info(f"  Layer {L}: {scores[L]:.4f}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
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

    # --- Baseline ---
    log.info("=" * 60)
    log.info("BASELINE (24 layers)")
    log.info("=" * 60)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    baseline_ppl, baseline_eval = evaluate_perplexity_input_ids(
        model,
        eval_input_ids,
        max_length=args.eval_max_length,
        stride=args.eval_stride,
    )
    eval_protocol = {**eval_protocol, **baseline_eval}
    log.info(f"Baseline PPL: {baseline_ppl:.2f}")
    del model

    results = {
        "baseline_ppl": baseline_ppl,
        "eval_protocol": eval_protocol,
        "guided": {},
        "random": {},
        "anti_guided": {},
    }

    for n_remove in range(1, args.max_remove + 1):
        log.info("\n" + "=" * 60)
        log.info(f"REMOVING {n_remove} LAYERS")
        log.info("=" * 60)

        # --- Guided removal ---
        guided_layers = greedy_select(scores, n_remove)
        guided_layers_sorted = sorted(guided_layers)
        log.info(f"  Guided: removing layers {guided_layers_sorted}")

        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
        model = remove_layers(model, set(guided_layers_sorted))
        model.eval()
        t0 = time.time()
        guided_ppl, _ = evaluate_perplexity_input_ids(
            model,
            eval_input_ids,
            max_length=args.eval_max_length,
            stride=args.eval_stride,
        )
        guided_time = time.time() - t0
        guided_delta = (guided_ppl - baseline_ppl) / baseline_ppl * 100
        log.info(f"  Guided PPL: {guided_ppl:.2f} (Δ={guided_delta:+.1f}%, {guided_time:.1f}s)")
        results["guided"][n_remove] = {
            "layers": guided_layers_sorted, "ppl": guided_ppl,
            "delta_pct": guided_delta
        }
        del model

        # --- Anti-guided removal ---
        anti_layers = greedy_select_worst(scores, n_remove)
        anti_layers_sorted = sorted(anti_layers)
        log.info(f"  Anti-guided: removing layers {anti_layers_sorted}")

        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
        model = remove_layers(model, set(anti_layers_sorted))
        model.eval()
        t0 = time.time()
        anti_ppl, _ = evaluate_perplexity_input_ids(
            model,
            eval_input_ids,
            max_length=args.eval_max_length,
            stride=args.eval_stride,
        )
        anti_time = time.time() - t0
        anti_delta = (anti_ppl - baseline_ppl) / baseline_ppl * 100
        log.info(f"  Anti-guided PPL: {anti_ppl:.2f} (Δ={anti_delta:+.1f}%, {anti_time:.1f}s)")
        results["anti_guided"][n_remove] = {
            "layers": anti_layers_sorted, "ppl": anti_ppl,
            "delta_pct": anti_delta
        }
        del model

        # --- Random removal ---
        random_ppls = []
        # Exclude layers 0, 23 (endpoints) from random pool to be fair
        middle_layers = list(range(1, n_layers - 1))
        for trial in range(args.random_trials):
            rand_layers = sorted(random.sample(middle_layers, n_remove))
            log.info(f"  Random trial {trial+1}: removing layers {rand_layers}")

            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
            model = remove_layers(model, set(rand_layers))
            model.eval()
            t0 = time.time()
            rand_ppl, _ = evaluate_perplexity_input_ids(
                model,
                eval_input_ids,
                max_length=args.eval_max_length,
                stride=args.eval_stride,
            )
            rand_time = time.time() - t0
            rand_delta = (rand_ppl - baseline_ppl) / baseline_ppl * 100
            log.info(f"    PPL: {rand_ppl:.2f} (Δ={rand_delta:+.1f}%, {rand_time:.1f}s)")
            random_ppls.append({"layers": rand_layers, "ppl": rand_ppl, "delta_pct": rand_delta})
            del model

        mean_rand_ppl = sum(r["ppl"] for r in random_ppls) / len(random_ppls)
        std_rand_ppl = (sum((r["ppl"] - mean_rand_ppl)**2 for r in random_ppls) / len(random_ppls))**0.5
        mean_rand_delta = (mean_rand_ppl - baseline_ppl) / baseline_ppl * 100
        results["random"][n_remove] = {
            "trials": random_ppls,
            "mean_ppl": mean_rand_ppl,
            "std_ppl": std_rand_ppl,
            "mean_delta_pct": mean_rand_delta
        }
        log.info(f"  Random mean PPL: {mean_rand_ppl:.2f} ± {std_rand_ppl:.2f} (Δ={mean_rand_delta:+.1f}%)")

    # --- Save ---
    out_path = os.path.join(REPORT_DIR, "compression_sweep.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved: {out_path}")

    # --- Summary table ---
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info(f"{'Removed':>8} | {'Guided':>12} | {'Random':>16} | {'Anti-guided':>12}")
    log.info("-" * 60)
    for n in range(1, args.max_remove + 1):
        g = results["guided"][n]
        r = results["random"][n]
        a = results["anti_guided"][n]
        log.info(f"{n:>8} | {g['ppl']:>6.2f} ({g['delta_pct']:+5.1f}%) | "
                f"{r['mean_ppl']:>6.2f}±{r['std_ppl']:.1f} ({r['mean_delta_pct']:+5.1f}%) | "
                f"{a['ppl']:>6.2f} ({a['delta_pct']:+5.1f}%)")


if __name__ == "__main__":
    main()
