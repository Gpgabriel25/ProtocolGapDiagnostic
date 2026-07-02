#!/usr/bin/env python3
"""
Pythia-1.4B Full Baseline Comparison
======================================
Fills the baseline coverage gap: currently Pythia-1.4B only has
interchange-guided vs replacement-guided removal. This adds:
  (A) BI-guided (ShortGPT-style Block Influence)
  (B) SLEB-iterative (calibration-based angular distance with iterative removal)
  (C) Random baseline (mean over 5 trials)
  (D) CKA-guided (Centered Kernel Alignment)

All methods select layers for REMOVAL at n=1,2,3 compression points.
Perplexity evaluated on WikiText-2 with the standardized protocol.

Usage:
    python pythia_full_baselines.py [--max-remove 3] [--n-calib 32]
"""

import os
import sys
import json
import time
import math
import random
import argparse
import logging

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

CYCLE_ID = "2026-04-06T17-39-31"
REPORT_DIR = os.environ.get("REPORT_DIR", f"reports/{CYCLE_ID}")
MODEL_NAME = "EleutherAI/pythia-1.4b"
DEVICE = "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def remove_layers_and_eval(model_name, tokenizer, eval_input_ids, skip_set,
                           max_length, stride, device="cpu"):
    """Load fresh model, remove layers, evaluate PPL."""
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.to(device)
    model.eval()

    # Pythia uses gpt_neox.layers
    layers = model.gpt_neox.layers
    n_orig = len(layers)
    keep = [i for i in range(n_orig) if i not in skip_set]
    new_layers = nn.ModuleList([layers[i] for i in keep])
    model.gpt_neox.layers = new_layers
    model.config.num_hidden_layers = len(new_layers)

    # Fix layer_idx in attention modules so KV cache indices match
    for new_idx, layer in enumerate(model.gpt_neox.layers):
        if hasattr(layer, 'attention') and hasattr(layer.attention, 'layer_idx'):
            layer.attention.layer_idx = new_idx

    with torch.no_grad():
        ppl, _ = evaluate_perplexity_input_ids(
            model, eval_input_ids.to(device),
            max_length=max_length, stride=stride,
        )
    del model
    return ppl


def get_position_embeddings(model, hidden_states, position_ids=None, device="cpu"):
    """Compute rotary position embeddings for GPT-NeoX models."""
    if position_ids is None:
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    return model.gpt_neox.rotary_emb(hidden_states, position_ids)


def compute_bi_scores(model, tokenizer, calib_texts, device="cpu"):
    """Compute Block Influence (ShortGPT BI) scores for each layer.
    BI_i = 1 - cos_sim(input_to_layer_i, output_of_layer_i)
    Low = more redundant.
    Uses forward hooks to capture layer inputs/outputs correctly."""
    model.to(device)
    model.eval()

    n_layers = len(model.gpt_neox.layers)
    bi_scores = [0.0] * n_layers
    n_samples = 0

    # Storage for hook captures
    layer_inputs = {}
    layer_outputs = {}

    def make_hook(idx):
        def hook_fn(module, inp, out):
            # inp is a tuple; first element is hidden_states
            layer_inputs[idx] = inp[0].detach()
            # out is a tuple; first element is hidden_states
            layer_outputs[idx] = out[0].detach()
        return hook_fn

    hooks = []
    for i, layer in enumerate(model.gpt_neox.layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    try:
        for text in calib_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
            with torch.no_grad():
                model(**inputs)

            for i in range(n_layers):
                cos = F.cosine_similarity(
                    layer_inputs[i].view(-1, layer_inputs[i].shape[-1]),
                    layer_outputs[i].view(-1, layer_outputs[i].shape[-1]),
                    dim=-1,
                ).mean().item()
                bi_scores[i] += (1.0 - cos)
            layer_inputs.clear()
            layer_outputs.clear()
            n_samples += 1
    finally:
        for h in hooks:
            h.remove()

    bi_scores = [s / n_samples for s in bi_scores]
    return bi_scores


def compute_cka_scores(model, tokenizer, calib_texts, device="cpu"):
    """Compute CKA-based layer importance.
    For each layer, CKA(layer_input, layer_output).
    High CKA = layers are similar = more redundant = lower importance.
    Uses forward hooks for correct position embedding handling."""
    model.to(device)
    model.eval()

    n_layers = len(model.gpt_neox.layers)

    def linear_cka(X, Y):
        """Compute linear CKA between two activation matrices."""
        X = X - X.mean(0)
        Y = Y - Y.mean(0)
        hsic_xy = torch.norm(X.T @ Y, p="fro") ** 2
        hsic_xx = torch.norm(X.T @ X, p="fro") ** 2
        hsic_yy = torch.norm(Y.T @ Y, p="fro") ** 2
        if hsic_xx * hsic_yy == 0:
            return 0.0
        return (hsic_xy / (hsic_xx.sqrt() * hsic_yy.sqrt())).item()

    # Storage for hook captures
    layer_inputs = {}
    layer_outputs = {}

    def make_hook(idx):
        def hook_fn(module, inp, out):
            layer_inputs[idx] = inp[0].detach()
            layer_outputs[idx] = out[0].detach()
        return hook_fn

    hooks = []
    for i, layer in enumerate(model.gpt_neox.layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    cka_scores = [0.0] * n_layers
    n_samples = 0

    try:
        for text in calib_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
            with torch.no_grad():
                model(**inputs)

            for i in range(n_layers):
                X = layer_inputs[i].view(-1, layer_inputs[i].shape[-1])
                Y = layer_outputs[i].view(-1, layer_outputs[i].shape[-1])
                cka = linear_cka(X, Y)
                cka_scores[i] += cka
            layer_inputs.clear()
            layer_outputs.clear()
            n_samples += 1
    finally:
        for h in hooks:
            h.remove()

    cka_scores = [s / n_samples for s in cka_scores]
    return cka_scores


def compute_sleb_scores(model_name, tokenizer, calib_texts, device="cpu"):
    """SLEB: calibration-based angular distance.
    For each layer, compute angular distance between model output
    with vs without the layer (using calibration data).
    Lower angular distance = more safely removable.
    Loads a fresh model for each layer skip to avoid position embedding issues."""
    # First get baseline outputs using full model
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    n_layers = len(model.gpt_neox.layers)

    calib_outputs = []
    for text in calib_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = model(**inputs)
        calib_outputs.append(out.logits[:, -1, :].detach())
    del model

    # For each layer, skip it and measure angular distance of output
    angular_scores = []
    for skip_idx in range(n_layers):
        # Load fresh model with this layer removed
        skip_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        skip_model.to(device)
        skip_model.eval()
        layers = skip_model.gpt_neox.layers
        keep = [i for i in range(n_layers) if i != skip_idx]
        skip_model.gpt_neox.layers = nn.ModuleList([layers[i] for i in keep])
        skip_model.config.num_hidden_layers = len(keep)

        # Fix layer_idx for KV cache
        for new_idx, layer in enumerate(skip_model.gpt_neox.layers):
            if hasattr(layer, 'attention') and hasattr(layer.attention, 'layer_idx'):
                layer.attention.layer_idx = new_idx

        dists = []
        for t_idx, text in enumerate(calib_texts):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
            with torch.no_grad():
                skip_out = skip_model(**inputs).logits[:, -1, :]

            cos = F.cosine_similarity(calib_outputs[t_idx], skip_out, dim=-1).clamp(-1, 1)
            angle = torch.acos(cos).mean().item()
            dists.append(angle)

        angular_scores.append(sum(dists) / len(dists))
        log.info(f"  SLEB layer {skip_idx}: angular_dist={angular_scores[-1]:.4f}")
        del skip_model

    return angular_scores


def greedy_select_by_score(scores, n, ascending=True):
    """Select n layers by score. ascending=True picks lowest first (most redundant)."""
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=not ascending)
    return [idx for idx, _ in indexed[:n]]


def sleb_iterative_select(model_name, tokenizer, calib_texts, n_remove, device="cpu"):
    """SLEB with iterative recalibration: remove one layer, recompute scores, repeat.
    Each iteration loads one fresh model with previously-removed layers excised,
    then tests each remaining layer by temporarily removing it."""
    removed = []
    for step in range(n_remove):
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        model.to(device)
        model.eval()

        # Remove already-selected layers
        all_layers = list(model.gpt_neox.layers)
        n_orig = len(all_layers)
        keep = [i for i in range(n_orig) if i not in removed]
        reduced_layers = nn.ModuleList([all_layers[i] for i in keep])
        model.gpt_neox.layers = reduced_layers
        model.config.num_hidden_layers = len(keep)

        # Fix layer_idx for KV cache
        for new_idx, layer in enumerate(model.gpt_neox.layers):
            if hasattr(layer, 'attention') and hasattr(layer.attention, 'layer_idx'):
                layer.attention.layer_idx = new_idx

        # Store original layer_idx values for restore
        _saved_layer_idx = {id(layer): layer.attention.layer_idx
                           for layer in model.gpt_neox.layers
                           if hasattr(layer, 'attention') and hasattr(layer.attention, 'layer_idx')}

        # Get baseline outputs from this reduced model
        calib_outputs = []
        for text in calib_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
            with torch.no_grad():
                out = model(**inputs)
            calib_outputs.append(out.logits[:, -1, :].detach())

        # For each remaining layer, temporarily remove it and measure angular distance
        scores = []
        for local_idx in range(len(keep)):
            # Temporarily remove this layer
            further_reduced = nn.ModuleList(
                [reduced_layers[j] for j in range(len(keep)) if j != local_idx]
            )
            model.gpt_neox.layers = further_reduced
            model.config.num_hidden_layers = len(further_reduced)

            # Fix layer_idx for the reduced set
            for new_idx2, layer in enumerate(model.gpt_neox.layers):
                if hasattr(layer, 'attention') and hasattr(layer.attention, 'layer_idx'):
                    layer.attention.layer_idx = new_idx2

            dists = []
            for t_idx, text in enumerate(calib_texts):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
                with torch.no_grad():
                    skip_out = model(**inputs).logits[:, -1, :]
                cos = F.cosine_similarity(calib_outputs[t_idx], skip_out, dim=-1).clamp(-1, 1)
                angle = torch.acos(cos).mean().item()
                dists.append(angle)
            scores.append(sum(dists) / len(dists))

            # Restore layers and layer_idx
            model.gpt_neox.layers = reduced_layers
            model.config.num_hidden_layers = len(keep)
            for layer in model.gpt_neox.layers:
                lid = id(layer)
                if lid in _saved_layer_idx:
                    layer.attention.layer_idx = _saved_layer_idx[lid]

        best_local = min(range(len(scores)), key=lambda i: scores[i])
        best_original = keep[best_local]
        removed.append(best_original)
        log.info(f"  SLEB-iter step {step+1}: remove layer {best_original} "
                 f"(local {best_local}, score {scores[best_local]:.4f})")
        del model
    return sorted(removed)


def main():
    parser = argparse.ArgumentParser(description="Pythia-1.4B full baseline comparison")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--max-remove", type=int, default=3)
    parser.add_argument("--n-calib", type=int, default=32, help="Number of calibration texts")
    parser.add_argument("--random-trials", type=int, default=5)
    parser.add_argument("--eval-max-words", type=int, default=10000)
    parser.add_argument("--eval-max-length", type=int, default=1024)
    parser.add_argument("--eval-stride", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(REPORT_DIR, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = DEVICE

    log.info(f"Model: {args.model}, device: {device}")

    # Load tokenizer and eval data
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    eval_input_ids, eval_protocol = build_wikitext2_eval_input(
        tokenizer, split="test", max_words=args.eval_max_words,
    )
    log.info(f"Eval tokens: {eval_input_ids.shape[1]}")

    # Calibration data
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calib_texts = [t for t in ds["text"] if len(t.strip()) > 100][:args.n_calib]
    log.info(f"Calibration texts: {len(calib_texts)}")

    # Baseline
    log.info("Computing baseline PPL...")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    with torch.no_grad():
        baseline_ppl, _ = evaluate_perplexity_input_ids(
            model, eval_input_ids, max_length=args.eval_max_length, stride=args.eval_stride,
        )
    log.info(f"Baseline PPL: {baseline_ppl:.2f}")

    # --- BI scores ---
    log.info("Computing BI scores...")
    bi_scores = compute_bi_scores(model, tokenizer, calib_texts, device)
    log.info(f"BI scores: {['%.4f' % s for s in bi_scores]}")

    # --- CKA scores ---
    log.info("Computing CKA scores...")
    cka_scores = compute_cka_scores(model, tokenizer, calib_texts, device)
    log.info(f"CKA scores: {['%.4f' % s for s in cka_scores]}")

    # --- SLEB single-pass scores ---
    log.info("Computing SLEB scores (single-pass)...")
    sleb_scores = compute_sleb_scores(args.model, tokenizer, calib_texts, device)
    log.info(f"SLEB scores: {['%.4f' % s for s in sleb_scores]}")

    del model

    all_results = {
        "model": args.model,
        "baseline_ppl": baseline_ppl,
        "n_layers": 24,
        "eval_protocol": eval_protocol,
        "scores": {
            "bi": bi_scores,
            "cka": cka_scores,
            "sleb_single": sleb_scores,
        },
        "methods": {},
    }

    # --- Evaluate each method at n=1,2,3 ---
    methods = {
        "bi_guided": lambda n: greedy_select_by_score(bi_scores, n, ascending=True),
        "cka_guided": lambda n: greedy_select_by_score(
            [1.0 - s for s in cka_scores], n, ascending=True  # High CKA = removable
        ),
        "sleb_oneshot": lambda n: greedy_select_by_score(sleb_scores, n, ascending=True),
    }

    for method_name, selector in methods.items():
        log.info(f"\n{'=' * 60}")
        log.info(f"METHOD: {method_name}")
        log.info(f"{'=' * 60}")
        method_results = {}
        for n in range(1, args.max_remove + 1):
            selected = selector(n)
            ppl = remove_layers_and_eval(
                args.model, tokenizer, eval_input_ids, set(selected),
                args.eval_max_length, args.eval_stride, device
            )
            delta = (ppl - baseline_ppl) / baseline_ppl * 100
            method_results[n] = {
                "layers": selected,
                "ppl": ppl,
                "delta_pct": delta,
            }
            log.info(f"  n={n}: skip {selected} -> PPL {ppl:.2f} ({delta:+.1f}%)")
        all_results["methods"][method_name] = method_results

    # --- SLEB iterative ---
    log.info(f"\n{'=' * 60}")
    log.info("METHOD: SLEB iterative")
    log.info(f"{'=' * 60}")
    sleb_iter_results = {}
    for n in range(1, args.max_remove + 1):
        selected = sleb_iterative_select(args.model, tokenizer, calib_texts, n, device)
        ppl = remove_layers_and_eval(
            args.model, tokenizer, eval_input_ids, set(selected),
            args.eval_max_length, args.eval_stride, device
        )
        delta = (ppl - baseline_ppl) / baseline_ppl * 100
        sleb_iter_results[n] = {
            "layers": selected,
            "ppl": ppl,
            "delta_pct": delta,
        }
        log.info(f"  n={n}: skip {selected} -> PPL {ppl:.2f} ({delta:+.1f}%)")
    all_results["methods"]["sleb_iterative"] = sleb_iter_results

    # --- Random baseline ---
    log.info(f"\n{'=' * 60}")
    log.info("METHOD: Random")
    log.info(f"{'=' * 60}")
    random_results = {}
    for n in range(1, args.max_remove + 1):
        trial_ppls = []
        trial_layers = []
        for trial in range(args.random_trials):
            layers = random.sample(range(1, 23), n)  # exclude boundary layers 0, 23
            ppl = remove_layers_and_eval(
                args.model, tokenizer, eval_input_ids, set(layers),
                args.eval_max_length, args.eval_stride, device
            )
            trial_ppls.append(ppl)
            trial_layers.append(layers)
            log.info(f"  Random trial {trial+1}: skip {layers} -> PPL {ppl:.2f}")
        mean_ppl = sum(trial_ppls) / len(trial_ppls)
        delta = (mean_ppl - baseline_ppl) / baseline_ppl * 100
        random_results[n] = {
            "mean_ppl": mean_ppl,
            "delta_pct": delta,
            "trials": [{"layers": l, "ppl": p} for l, p in zip(trial_layers, trial_ppls)],
        }
        log.info(f"  n={n} random mean: PPL {mean_ppl:.2f} ({delta:+.1f}%)")
    all_results["methods"]["random"] = random_results

    # --- Summary ---
    log.info(f"\n{'=' * 60}")
    log.info("SUMMARY")
    log.info(f"{'=' * 60}")
    log.info(f"Baseline PPL: {baseline_ppl:.2f}")
    for method_name, data in all_results["methods"].items():
        log.info(f"\n{method_name}:")
        for n_key in sorted(data.keys(), key=lambda x: int(x)):
            entry = data[n_key]
            ppl = entry.get("ppl", entry.get("mean_ppl", "?"))
            delta = entry.get("delta_pct", 0)
            layers = entry.get("layers", "random")
            log.info(f"  n={n_key}: {layers} -> {ppl:.2f} ({delta:+.1f}%)")

    # Save
    out_path = os.path.join(REPORT_DIR, "pythia_full_baselines.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
