#!/usr/bin/env python3
"""
LaCo and SLEB compression baselines for comparison with bisimulation.
=====================================================================
Implements:
  - LaCo-bisim: adjacent layer collapse ordered by bisimulation distance
  - LaCo-BI: adjacent layer collapse ordered by BI score similarity
  - LaCo-sequential: bottom-up sequential layer collapse
  - SLEB-greedy: greedy single-layer removal by calibration importance
  - Bisimulation-guided removal (existing baseline)
  - BI-guided removal (existing baseline)

Runs on GPT-2-Medium (CPU). Tests budgets 1-5.
"""

import os
import json
import math
import copy
import csv
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from wikitext_ppl import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MAX_WORDS,
    DEFAULT_STRIDE,
    build_wikitext2_eval_input,
    evaluate_perplexity_input_ids,
)

CYCLE_ID = "2026-03-31T12-24-40"
REPORT_DIR = os.environ.get("REPORT_DIR", f"reports/{CYCLE_ID}")
MODEL_NAME = "gpt2-medium"
SORTED_PAIRS_CSV = "reports/2026-03-30T15-15-07/sorted_pairs.csv"
BI_SCORES_JSON = "reports/2026-03-31T00-18-24/bi_score_comparison.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model surgery helpers
# ---------------------------------------------------------------------------

def remove_layers(model, layer_indices):
    """Remove layers from GPT-2 model (in-place). Returns model."""
    layers = model.transformer.h
    keep = [i for i in range(len(layers)) if i not in layer_indices]
    model.transformer.h = nn.ModuleList([layers[i] for i in keep])
    model.config.n_layer = len(model.transformer.h)
    for i, layer in enumerate(model.transformer.h):
        if hasattr(layer, "attn") and hasattr(layer.attn, "layer_idx"):
            layer.attn.layer_idx = i
    return model


def merge_adjacent_layers(model, layer_a, layer_b):
    """Merge two adjacent layers by averaging their weights.
    Replaces the pair with a single merged layer at the position of layer_a.
    layer_a < layer_b must hold. Returns model (modified in-place)."""
    assert layer_a < layer_b
    layers = model.transformer.h
    sd_a = layers[layer_a].state_dict()
    sd_b = layers[layer_b].state_dict()
    merged_sd = {}
    for key in sd_a:
        merged_sd[key] = (sd_a[key] + sd_b[key]) / 2.0
    layers[layer_a].load_state_dict(merged_sd)
    # Remove layer_b
    keep = [i for i in range(len(layers)) if i != layer_b]
    model.transformer.h = nn.ModuleList([layers[i] for i in keep])
    model.config.n_layer = len(model.transformer.h)
    for i, layer in enumerate(model.transformer.h):
        if hasattr(layer, "attn") and hasattr(layer.attn, "layer_idx"):
            layer.attn.layer_idx = i
    return model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_bisim_pairs():
    """Load sorted bisimulation pairs from CSV. Returns list of dicts."""
    pairs = []
    with open(SORTED_PAIRS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append({
                "layer_a": int(row["layer_a"]),
                "layer_b": int(row["layer_b"]),
                "gap": int(row["gap"]),
                "mean_kl": float(row["mean_kl"]),
            })
    return pairs


def load_bi_scores():
    """Load BI scores from JSON. Returns dict layer_idx -> score."""
    with open(BI_SCORES_JSON) as f:
        data = json.load(f)
    return {int(k): v for k, v in data["bi_scores"].items()}


def get_adjacent_bisim_pairs(pairs):
    """Filter to adjacent pairs (gap==1), sorted by mean_kl ascending."""
    adj = [p for p in pairs if p["gap"] == 1]
    adj.sort(key=lambda p: p["mean_kl"])
    return adj


def get_calibration_ids(tokenizer, n_tokens=1000):
    """Get first n_tokens of the standardized WikiText-2 train stream."""
    input_ids, info = build_wikitext2_eval_input(
        tokenizer,
        split="train",
        max_words=DEFAULT_MAX_WORDS,
    )
    return input_ids[0, :n_tokens].clone(), {**info, "eval_tokens": int(min(n_tokens, input_ids.size(1)))}


# ---------------------------------------------------------------------------
# LaCo implementations
# ---------------------------------------------------------------------------

def laco_bisim(model, budget, adj_pairs):
    """LaCo ordered by bisimulation distance (lowest KL first).
    Merge `budget` adjacent pairs. Returns (model, layers_merged)."""
    merged = []
    used_layers = set()
    for pair in adj_pairs:
        if len(merged) >= budget:
            break
        a, b = pair["layer_a"], pair["layer_b"]
        if a in used_layers or b in used_layers:
            continue
        used_layers.add(a)
        used_layers.add(b)
        merged.append((a, b))

    # Merge from highest index first to avoid index shifting
    merged.sort(key=lambda x: x[1], reverse=True)
    layers_merged = []
    for a, b in merged:
        # Find current indices of these layers
        # After prior merges, indices shift. Track by original index.
        # We need to map original layer index to current position.
        current_layers = list(range(model.config.n_layer))
        # Since we merge from high to low, just need to find the layers
        model = merge_adjacent_layers(model, a, b)
        layers_merged.append((a, b))

    return model, layers_merged


def _remap_merge_pairs(pairs_to_merge):
    """Given a list of (a, b) pairs in original indexing, return them
    in the order they should be merged (highest index first) so index
    shifts are handled correctly."""
    # Sort by b descending so we merge from the back
    return sorted(pairs_to_merge, key=lambda x: x[1], reverse=True)


def laco_merge_pairs(model_name, pairs_to_merge):
    """Load a fresh model and merge the given pairs (in original indexing).
    Returns modified model."""
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    # We need to merge from back to front to keep indices valid
    ordered = _remap_merge_pairs(pairs_to_merge)
    # Build a mapping from original layer index to current index
    original_to_current = list(range(24))  # GPT-2-Medium has 24 layers

    for orig_a, orig_b in ordered:
        # Find current positions
        cur_a = original_to_current.index(orig_a)
        cur_b = original_to_current.index(orig_b)
        if cur_a > cur_b:
            cur_a, cur_b = cur_b, cur_a
        model = merge_adjacent_layers(model, cur_a, cur_b)
        # Update mapping: remove orig_b from the list
        original_to_current.pop(cur_b)

    return model


def laco_bi(budget, bi_scores):
    """LaCo ordered by BI score similarity.
    For adjacent pairs, compute |BI[i] - BI[i+1]|. Merge pairs with
    most similar BI scores first (smallest difference)."""
    n_layers = max(bi_scores.keys()) + 1
    adj_bi_diff = []
    for i in range(n_layers - 1):
        diff = abs(bi_scores[i] - bi_scores[i + 1])
        adj_bi_diff.append((i, i + 1, diff))
    adj_bi_diff.sort(key=lambda x: x[2])

    pairs = []
    used = set()
    for a, b, _ in adj_bi_diff:
        if len(pairs) >= budget:
            break
        if a in used or b in used:
            continue
        used.add(a)
        used.add(b)
        pairs.append((a, b))
    return pairs


def laco_sequential(budget, n_layers=24):
    """LaCo bottom-up sequential: merge (0,1), (2,3), ..."""
    pairs = []
    for i in range(0, n_layers - 1, 2):
        if len(pairs) >= budget:
            break
        pairs.append((i, i + 1))
    return pairs


# ---------------------------------------------------------------------------
# SLEB implementation
# ---------------------------------------------------------------------------

def sleb_importance(model_name, tokenizer, calib_ids, n_layers=24):
    """Compute per-layer importance as PPL increase when layer is removed.
    Uses a small calibration set for speed."""
    log.info("SLEB: Computing per-layer importance scores...")
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    base_model.eval()
    base_ppl, _ = evaluate_perplexity_input_ids(
        base_model,
        calib_ids,
        max_length=DEFAULT_MAX_LENGTH,
        stride=DEFAULT_STRIDE,
    )
    log.info(f"  SLEB calibration baseline PPL: {base_ppl:.2f}")
    del base_model

    importance = {}
    for layer_idx in range(n_layers):
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        model.eval()
        model = remove_layers(model, {layer_idx})
        ppl, _ = evaluate_perplexity_input_ids(
            model,
            calib_ids,
            max_length=DEFAULT_MAX_LENGTH,
            stride=DEFAULT_STRIDE,
        )
        importance[layer_idx] = ppl - base_ppl
        log.info(f"  Layer {layer_idx}: ΔPPL = {importance[layer_idx]:+.2f}")
        del model

    return importance


def sleb_greedy_removal(importance, budget):
    """Greedily remove `budget` layers with lowest importance (least PPL impact)."""
    sorted_layers = sorted(importance.items(), key=lambda x: x[1])
    return [layer for layer, _ in sorted_layers[:budget]]


# ---------------------------------------------------------------------------
# Bisimulation-guided removal
# ---------------------------------------------------------------------------

def bisim_guided_removal(pairs, budget):
    """Remove layers that have the lowest bisimulation distance to their
    nearest neighbor. For each adjacent pair sorted by mean_kl, remove
    the layer that appears more redundant (higher-index layer of the pair)."""
    to_remove = []
    used = set()
    for p in pairs:
        if len(to_remove) >= budget:
            break
        a, b = p["layer_a"], p["layer_b"]
        # Remove the higher-index layer (it's more likely redundant in
        # adjacent pairs)
        candidate = b
        if candidate in used:
            candidate = a
        if candidate in used:
            continue
        to_remove.append(candidate)
        used.add(candidate)
    return sorted(to_remove)


def bi_guided_removal(bi_scores, budget):
    """Remove layers with lowest BI scores (most redundant per ShortGPT)."""
    sorted_layers = sorted(bi_scores.items(), key=lambda x: x[1])
    return [layer for layer, _ in sorted_layers[:budget]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    log.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    log.info("Preparing standardized WikiText-2 evaluation stream...")
    eval_input_ids, eval_protocol = build_wikitext2_eval_input(
        tokenizer,
        split="test",
        max_words=DEFAULT_MAX_WORDS,
    )
    log.info(
        "  words=%d tokens=%d max_length=%d stride=%d",
        eval_protocol["text_words"],
        eval_protocol["token_count"],
        DEFAULT_MAX_LENGTH,
        DEFAULT_STRIDE,
    )

    log.info("Loading calibration data...")
    calib_ids, calibration_protocol = get_calibration_ids(tokenizer, n_tokens=1000)
    log.info(f"  {calibration_protocol['eval_tokens']} calibration tokens")

    log.info("Loading bisimulation pairs...")
    all_pairs = load_bisim_pairs()
    adj_pairs = get_adjacent_bisim_pairs(all_pairs)
    log.info(f"  {len(adj_pairs)} adjacent pairs loaded")

    log.info("Loading BI scores...")
    bi_scores = load_bi_scores()

    # Baseline
    log.info("Computing baseline perplexity...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    base_model.eval()
    baseline_ppl, baseline_eval = evaluate_perplexity_input_ids(
        base_model,
        eval_input_ids,
        max_length=DEFAULT_MAX_LENGTH,
        stride=DEFAULT_STRIDE,
    )
    eval_protocol = {**eval_protocol, **baseline_eval}
    log.info(f"  Baseline PPL: {baseline_ppl:.2f}")
    del base_model

    # SLEB importance (compute once)
    sleb_imp = sleb_importance(MODEL_NAME, tokenizer, calib_ids)

    results = []

    for budget in range(1, 6):
        log.info(f"\n{'='*60}")
        log.info(f"BUDGET = {budget} layers")
        log.info(f"{'='*60}")

        # --- 1. Bisimulation-guided removal ---
        bisim_layers = bisim_guided_removal(adj_pairs, budget)
        log.info(f"  Bisim-guided removal: {bisim_layers}")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        model.eval()
        model = remove_layers(model, set(bisim_layers))
        ppl, _ = evaluate_perplexity_input_ids(
            model,
            eval_input_ids,
            max_length=DEFAULT_MAX_LENGTH,
            stride=DEFAULT_STRIDE,
        )
        results.append({
            "method": "bisim-guided",
            "budget": budget,
            "ppl": round(ppl, 2),
            "delta_pct": round((ppl - baseline_ppl) / baseline_ppl * 100, 2),
            "layers_removed": bisim_layers,
        })
        log.info(f"  Bisim-guided PPL: {ppl:.2f} (Δ={results[-1]['delta_pct']:+.1f}%)")
        del model

        # --- 2. BI-guided removal ---
        bi_layers = bi_guided_removal(bi_scores, budget)
        log.info(f"  BI-guided removal: {bi_layers}")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        model.eval()
        model = remove_layers(model, set(bi_layers))
        ppl, _ = evaluate_perplexity_input_ids(
            model,
            eval_input_ids,
            max_length=DEFAULT_MAX_LENGTH,
            stride=DEFAULT_STRIDE,
        )
        results.append({
            "method": "bi-guided",
            "budget": budget,
            "ppl": round(ppl, 2),
            "delta_pct": round((ppl - baseline_ppl) / baseline_ppl * 100, 2),
            "layers_removed": bi_layers,
        })
        log.info(f"  BI-guided PPL: {ppl:.2f} (Δ={results[-1]['delta_pct']:+.1f}%)")
        del model

        # --- 3. LaCo-bisim (merge by bisimulation distance) ---
        laco_bisim_pairs = []
        used = set()
        for p in adj_pairs:
            if len(laco_bisim_pairs) >= budget:
                break
            a, b = p["layer_a"], p["layer_b"]
            if a in used or b in used:
                continue
            used.add(a)
            used.add(b)
            laco_bisim_pairs.append((a, b))
        log.info(f"  LaCo-bisim merge pairs: {laco_bisim_pairs}")
        model = laco_merge_pairs(MODEL_NAME, laco_bisim_pairs)
        ppl, _ = evaluate_perplexity_input_ids(
            model,
            eval_input_ids,
            max_length=DEFAULT_MAX_LENGTH,
            stride=DEFAULT_STRIDE,
        )
        merged_layers_flat = sorted(set(l for p in laco_bisim_pairs for l in p))
        results.append({
            "method": "laco-bisim",
            "budget": budget,
            "ppl": round(ppl, 2),
            "delta_pct": round((ppl - baseline_ppl) / baseline_ppl * 100, 2),
            "layers_merged": [list(p) for p in laco_bisim_pairs],
        })
        log.info(f"  LaCo-bisim PPL: {ppl:.2f} (Δ={results[-1]['delta_pct']:+.1f}%)")
        del model

        # --- 4. LaCo-BI (merge by BI similarity) ---
        laco_bi_pairs = laco_bi(budget, bi_scores)
        log.info(f"  LaCo-BI merge pairs: {laco_bi_pairs}")
        model = laco_merge_pairs(MODEL_NAME, laco_bi_pairs)
        ppl, _ = evaluate_perplexity_input_ids(
            model,
            eval_input_ids,
            max_length=DEFAULT_MAX_LENGTH,
            stride=DEFAULT_STRIDE,
        )
        results.append({
            "method": "laco-bi",
            "budget": budget,
            "ppl": round(ppl, 2),
            "delta_pct": round((ppl - baseline_ppl) / baseline_ppl * 100, 2),
            "layers_merged": [list(p) for p in laco_bi_pairs],
        })
        log.info(f"  LaCo-BI PPL: {ppl:.2f} (Δ={results[-1]['delta_pct']:+.1f}%)")
        del model

        # --- 5. LaCo-sequential ---
        seq_pairs = laco_sequential(budget)
        log.info(f"  LaCo-sequential merge pairs: {seq_pairs}")
        model = laco_merge_pairs(MODEL_NAME, seq_pairs)
        ppl, _ = evaluate_perplexity_input_ids(
            model,
            eval_input_ids,
            max_length=DEFAULT_MAX_LENGTH,
            stride=DEFAULT_STRIDE,
        )
        results.append({
            "method": "laco-sequential",
            "budget": budget,
            "ppl": round(ppl, 2),
            "delta_pct": round((ppl - baseline_ppl) / baseline_ppl * 100, 2),
            "layers_merged": [list(p) for p in seq_pairs],
        })
        log.info(f"  LaCo-sequential PPL: {ppl:.2f} (Δ={results[-1]['delta_pct']:+.1f}%)")
        del model

        # --- 6. SLEB-greedy removal ---
        sleb_layers = sleb_greedy_removal(sleb_imp, budget)
        log.info(f"  SLEB-greedy removal: {sleb_layers}")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        model.eval()
        model = remove_layers(model, set(sleb_layers))
        ppl, _ = evaluate_perplexity_input_ids(
            model,
            eval_input_ids,
            max_length=DEFAULT_MAX_LENGTH,
            stride=DEFAULT_STRIDE,
        )
        results.append({
            "method": "sleb-greedy",
            "budget": budget,
            "ppl": round(ppl, 2),
            "delta_pct": round((ppl - baseline_ppl) / baseline_ppl * 100, 2),
            "layers_removed": sleb_layers,
        })
        log.info(f"  SLEB-greedy PPL: {ppl:.2f} (Δ={results[-1]['delta_pct']:+.1f}%)")
        del model

    # Save results
    output = {
        "model": MODEL_NAME,
        "baseline_ppl": round(baseline_ppl, 2),
        "n_layers": 24,
        "eval_protocol": eval_protocol,
        "calibration_protocol": calibration_protocol,
        "results": results,
    }
    out_path = os.path.join(REPORT_DIR, "laco_sleb_baselines.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nResults saved to {out_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY: Baseline PPL = {baseline_ppl:.2f}")
    print(f"{'='*80}")
    print(f"{'Method':<20} {'Budget':>6} {'PPL':>8} {'ΔPPL%':>8} {'Layers':>30}")
    print(f"{'-'*80}")
    for r in sorted(results, key=lambda x: (x["budget"], x["method"])):
        layers_str = str(r.get("layers_removed", r.get("layers_merged", "")))
        if len(layers_str) > 28:
            layers_str = layers_str[:25] + "..."
        print(f"{r['method']:<20} {r['budget']:>6} {r['ppl']:>8.2f} {r['delta_pct']:>+7.1f}% {layers_str:>30}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
