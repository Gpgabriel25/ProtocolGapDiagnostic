#!/usr/bin/env python3
"""
Taylor Importance Baseline
============================
Adds the missing Taylor importance pruning baseline to match paper Table 4 and 5 formats.
Works for GPT-2-Medium, Pythia-1.4B, and Qwen3-8B (the last via a separate TPU script).

Taylor importance score for layer k:
  I_k = sum_{p in theta_k} |grad_p * p|  (first-order Taylor approximation of loss change)
  averaged over calibration examples.

Layers with lowest I_k are removed first (most redundant).

This uses *calibration data* and *gradients*, confirming Taylor is NOT calibration-free.
The comparison therefore shows: swap-KL is the only output-grounded, calibration-free method.

Usage:
  python taylor_importance_baselines.py --model gpt2-medium --max-remove 5
  python taylor_importance_baselines.py --model pythia-1.4b --max-remove 5
"""

import os
import sys
import json
import time
import math
import random
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from wikitext_ppl import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MAX_WORDS,
    DEFAULT_STRIDE,
    build_wikitext2_eval_input,
    evaluate_perplexity_input_ids,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CYCLE_ID = os.environ.get("CYCLE_ID", "2026-04-12T16-25-41")
LOG_DIR = os.environ.get("LOG_DIR", f"logs/{CYCLE_ID}")

MODEL_CONFIGS = {
    "gpt2-medium": {
        "model_name": "openai-community/gpt2-medium",
        "layers_fn": lambda m: list(m.transformer.h),
        "set_layers_fn": lambda m, ls: setattr(m.transformer, "h", nn.ModuleList(ls)),
        "layer_idx_fn": None,  # GPT-2 doesn't use layer_idx
        # Match tab:extended_baselines evaluator (laco_sleb_baselines.py uses DEFAULT_MAX_WORDS=20000)
        "eval_max_words": 20000,
    },
    "pythia-1.4b": {
        "model_name": "EleutherAI/pythia-1.4b",
        "layers_fn": lambda m: list(m.gpt_neox.layers),
        "set_layers_fn": lambda m, ls: setattr(m.gpt_neox, "layers", nn.ModuleList(ls)),
        "layer_idx_fn": lambda m: [
            setattr(layer.attention, "layer_idx", idx)
            for idx, layer in enumerate(m.gpt_neox.layers)
            if hasattr(layer, "attention") and hasattr(layer.attention, "layer_idx")
        ],
        # Match tab:pythia_baselines evaluator (pythia_full_baselines.py uses max_words=10000)
        "eval_max_words": 10000,
    },
}

N_CALIB = int(os.environ.get("N_CALIB", "32"))
CALIB_MAX_LEN = int(os.environ.get("CALIB_MAX_LEN", "256"))


def load_calib_texts(n=N_CALIB, max_len=CALIB_MAX_LEN, seed=42):
    """Load calibration texts from WikiText-2 train split."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if len(t.strip()) > 50]
    rng = random.Random(seed)
    sampled = rng.sample(texts, min(n, len(texts)))
    return sampled


def get_layer_params(layer):
    """Return all parameters belonging to this layer module."""
    return list(layer.parameters())


def compute_taylor_importance_scores(model_key, device="cpu"):
    """
    Compute Taylor importance scores for each layer.
    I_k = mean over calibration examples of: sum_{p in theta_k} |grad_p * p|

    Higher = more important (lower = more removable).
    """
    cfg = MODEL_CONFIGS[model_key]
    model_name = cfg["model_name"]
    layers_fn = cfg["layers_fn"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.to(device)
    model.train()  # enable gradients

    calib_texts = load_calib_texts()
    layers = layers_fn(model)
    n_layers = len(layers)

    # Accumulate |grad * weight| for each layer
    layer_importance = [0.0] * n_layers
    n_processed = 0

    for text in calib_texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=CALIB_MAX_LEN, padding=False
        ).to(device)
        input_ids = inputs["input_ids"]
        if input_ids.shape[1] < 2:
            continue

        labels = input_ids.clone()
        model.zero_grad()
        out = model(**inputs, labels=labels)
        loss = out.loss
        loss.backward()

        # Collect gradient importance per layer
        for i, layer in enumerate(layers_fn(model)):
            layer_score = 0.0
            for p in layer.parameters():
                if p.grad is not None:
                    layer_score += (p.grad * p.data).abs().sum().item()
            layer_importance[i] += layer_score

        n_processed += 1

    model.eval()
    # Normalize by number of processed examples
    layer_importance = [s / max(n_processed, 1) for s in layer_importance]
    log.info(f"Taylor scores (first 10): {layer_importance[:10]}")
    return layer_importance, tokenizer, model_name, n_layers


def remove_layers_and_eval_gpt2(model_name, tokenizer, eval_input_ids, skip_set,
                                 max_length, stride, device="cpu"):
    """Load fresh GPT-2 model with layers removed and evaluate PPL."""
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.to(device)
    model.eval()
    layers = model.transformer.h
    keep = [i for i in range(len(layers)) if i not in skip_set]
    model.transformer.h = nn.ModuleList([layers[i] for i in keep])
    model.config.n_layer = len(keep)
    # Reset layer_idx for DynamicCache compatibility (new transformers)
    for new_idx, layer in enumerate(model.transformer.h):
        attn = getattr(layer, 'attn', None)
        if attn is not None and hasattr(attn, 'layer_idx'):
            attn.layer_idx = new_idx
    with torch.no_grad():
        ppl, _ = evaluate_perplexity_input_ids(
            model, eval_input_ids.to(device),
            max_length=max_length, stride=stride,
        )
    del model
    return ppl


def remove_layers_and_eval_pythia(model_name, tokenizer, eval_input_ids, skip_set,
                                   max_length, stride, device="cpu"):
    """Load fresh Pythia model with layers removed and evaluate PPL."""
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.to(device)
    model.eval()
    layers = model.gpt_neox.layers
    n_orig = len(layers)
    keep = [i for i in range(n_orig) if i not in skip_set]
    model.gpt_neox.layers = nn.ModuleList([layers[i] for i in keep])
    model.config.num_hidden_layers = len(keep)
    for new_idx, layer in enumerate(model.gpt_neox.layers):
        if hasattr(layer, "attention") and hasattr(layer.attention, "layer_idx"):
            layer.attention.layer_idx = new_idx
    with torch.no_grad():
        ppl, _ = evaluate_perplexity_input_ids(
            model, eval_input_ids.to(device),
            max_length=max_length, stride=stride,
        )
    del model
    return ppl


def greedy_select_lowest(scores, n):
    """Select n layers with lowest importance (most removable)."""
    indexed = sorted(enumerate(scores), key=lambda x: x[1])
    return [idx for idx, _ in indexed[:n]]


def run_taylor_eval(model_key, max_remove=5, device="cpu"):
    """Full pipeline: compute Taylor scores, remove n best, evaluate PPL."""
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    log.info(f"=== Taylor Importance Baseline: {model_key} ===")
    t0 = time.time()

    # Compute scores
    scores, tokenizer, model_name, n_layers = compute_taylor_importance_scores(model_key, device)
    log.info(f"Score computation: {time.time()-t0:.1f}s. n_layers={n_layers}")

    # Build eval input — use model-specific max_words to match paper table evaluators
    eval_max_words = MODEL_CONFIGS[model_key].get("eval_max_words", DEFAULT_MAX_WORDS)
    eval_tokenizer = AutoTokenizer.from_pretrained(model_name)
    eval_tokenizer.pad_token = eval_tokenizer.eos_token
    eval_input_ids, _info = build_wikitext2_eval_input(eval_tokenizer, max_words=eval_max_words)
    log.info(f"Eval input: {eval_max_words} words → {eval_input_ids.shape[1]} tokens")

    # Dispatcher for removal+eval
    if model_key.startswith("gpt2"):
        remove_fn = remove_layers_and_eval_gpt2
    else:
        remove_fn = remove_layers_and_eval_pythia

    # Baseline PPL (no removals)
    baseline_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(device).eval()
    with torch.no_grad():
        baseline_ppl, _ = evaluate_perplexity_input_ids(
            baseline_model, eval_input_ids.to(device),
            max_length=DEFAULT_MAX_LENGTH, stride=DEFAULT_STRIDE,
        )
    del baseline_model
    log.info(f"Baseline PPL: {baseline_ppl:.4f}")

    results = {
        "model": model_key,
        "model_name": model_name,
        "n_layers": n_layers,
        "n_calib": N_CALIB,
        "baseline_ppl": baseline_ppl,
        "baseline_delta_pct": 0.0,
        "scores": scores,
        "removals": {},
    }

    # Evaluate at each n
    for n in range(1, max_remove + 1):
        selected = greedy_select_lowest(scores, n)
        t1 = time.time()
        ppl = remove_fn(
            model_name, tokenizer, eval_input_ids, set(selected),
            max_length=DEFAULT_MAX_LENGTH, stride=DEFAULT_STRIDE, device=device
        )
        delta_pct = 100.0 * (ppl - baseline_ppl) / baseline_ppl
        elapsed = time.time() - t1
        log.info(f"  n={n}: removed={selected}, PPL={ppl:.4f}, Δ={delta_pct:+.2f}%, t={elapsed:.1f}s")
        results["removals"][str(n)] = {
            "n_removed": n,
            "layers_removed": selected,
            "ppl": round(ppl, 4),
            "delta_ppl_pct": round(delta_pct, 2),
            "elapsed_s": round(elapsed, 1),
        }

    total = time.time() - t0
    results["total_elapsed_s"] = round(total, 1)
    log.info(f"Total elapsed: {total:.1f}s")

    out_path = f"{LOG_DIR}/{model_key.replace('/', '_')}_taylor_baselines.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved: {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pythia-1.4b", choices=list(MODEL_CONFIGS.keys()),
                        help="Which model key to evaluate")
    parser.add_argument("--max-remove", type=int, default=5,
                        help="Maximum number of layers to remove")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()

    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    results = run_taylor_eval(args.model, max_remove=args.max_remove, device=args.device)
    # Print summary
    print(f"\nSummary — {args.model}")
    print(f"  baseline PPL: {results['baseline_ppl']:.4f}")
    for n_str, r in results["removals"].items():
        print(f"  n={n_str}: PPL={r['ppl']:.4f}  Δ={r['delta_ppl_pct']:+.2f}%  layers={r['layers_removed']}")
