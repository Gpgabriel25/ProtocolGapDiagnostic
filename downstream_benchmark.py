#!/usr/bin/env python3
"""
Downstream Task Benchmarks for Bisimulation Compression
========================================================
Evaluates baseline GPT-2-Medium and skip-layer variants on standard
NLP benchmarks using lm-evaluation-harness.

Tasks: hellaswag, piqa, arc_easy, arc_challenge, winogrande, lambada_openai
"""

import os
import sys
import json
import time
import copy
import argparse
import logging
from datetime import datetime

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel

import lm_eval
from lm_eval.models.huggingface import HFLM

# ---------------------------------------------------------------------------
CYCLE_ID = "2026-03-31T00-18-24"
REPORT_DIR = f"reports/{CYCLE_ID}"
TASKS = ["hellaswag", "piqa", "arc_easy", "arc_challenge", "winogrande", "lambada_openai"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def remove_layer(model: GPT2LMHeadModel, layer_idx: int) -> GPT2LMHeadModel:
    """Remove a single transformer layer from a GPT-2 model (in-place)."""
    layers = model.transformer.h
    new_layers = nn.ModuleList(
        [layers[i] for i in range(len(layers)) if i != layer_idx]
    )
    model.transformer.h = new_layers
    model.config.n_layer = len(new_layers)
    # Fix layer_idx for attention KV cache
    for i, layer in enumerate(new_layers):
        if hasattr(layer, "attn") and hasattr(layer.attn, "layer_idx"):
            layer.attn.layer_idx = i
    return model


def evaluate_model(model, tokenizer, tasks, batch_size=4, limit=None):
    """Evaluate a model on the given tasks using lm-eval."""
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        batch_size=batch_size,
        limit=limit,
    )
    return results


def extract_scores(results):
    """Extract accuracy scores from lm-eval results."""
    scores = {}
    for task_name, task_results in results["results"].items():
        # Try acc_norm first, then acc
        if "acc_norm,none" in task_results:
            scores[task_name] = task_results["acc_norm,none"]
        elif "acc,none" in task_results:
            scores[task_name] = task_results["acc,none"]
        else:
            # Use first metric that looks like accuracy
            for k, v in task_results.items():
                if "acc" in k and isinstance(v, (int, float)):
                    scores[task_name] = v
                    break
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-medium", help="Base model name")
    parser.add_argument("--skip-layers", nargs="*", type=int, default=[5],
                        help="Layer indices to test removal for")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tasks", nargs="*", default=TASKS)
    parser.add_argument("--limit", type=int, default=None,
                        help="Max examples per task (for faster CPU runs)")
    args = parser.parse_args()

    os.makedirs(REPORT_DIR, exist_ok=True)

    log.info(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # --- Baseline evaluation ---
    log.info("=" * 60)
    log.info("BASELINE EVALUATION")
    log.info("=" * 60)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()

    t0 = time.time()
    baseline_results = evaluate_model(model, tokenizer, args.tasks, args.batch_size, args.limit)
    baseline_time = time.time() - t0
    baseline_scores = extract_scores(baseline_results)
    log.info(f"Baseline completed in {baseline_time:.1f}s")
    for task, score in baseline_scores.items():
        log.info(f"  {task}: {score:.4f}")

    all_results = {"baseline": {"scores": baseline_scores, "time": baseline_time}}

    # --- Skip-layer evaluations ---
    for skip_idx in args.skip_layers:
        log.info("=" * 60)
        log.info(f"SKIP-LAYER {skip_idx} EVALUATION")
        log.info("=" * 60)

        # Reload fresh model and remove layer
        skip_model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32
        )
        skip_model = remove_layer(skip_model, skip_idx)
        skip_model.eval()
        n_layers = len(skip_model.transformer.h)
        log.info(f"  Model reduced to {n_layers} layers")

        t0 = time.time()
        skip_results = evaluate_model(skip_model, tokenizer, args.tasks, args.batch_size, args.limit)
        skip_time = time.time() - t0
        skip_scores = extract_scores(skip_results)
        log.info(f"Skip-{skip_idx} completed in {skip_time:.1f}s")

        deltas = {}
        for task in baseline_scores:
            if task in skip_scores:
                delta = skip_scores[task] - baseline_scores[task]
                deltas[task] = delta
                log.info(f"  {task}: {skip_scores[task]:.4f} (Δ={delta:+.4f})")

        all_results[f"skip_{skip_idx}"] = {
            "scores": skip_scores, "time": skip_time, "deltas": deltas
        }

        # Free memory
        del skip_model

    # --- Save results ---
    out_path = os.path.join(REPORT_DIR, "downstream_benchmarks.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Saved: {out_path}")

    # --- Write markdown report ---
    md_path = os.path.join(REPORT_DIR, "downstream_benchmarks.md")
    with open(md_path, "w") as f:
        f.write("# Downstream Task Benchmarks\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Tasks: {', '.join(args.tasks)}\n\n")
        f.write("## Results\n\n")
        f.write("| Task | Baseline | " +
                " | ".join(f"Skip-{s}" for s in args.skip_layers) +
                " | " +
                " | ".join(f"Δ Skip-{s}" for s in args.skip_layers) +
                " |\n")
        f.write("|---" * (2 + 2 * len(args.skip_layers)) + "|\n")
        for task in sorted(baseline_scores.keys()):
            row = f"| {task} | {baseline_scores[task]:.4f}"
            for skip_idx in args.skip_layers:
                key = f"skip_{skip_idx}"
                s = all_results[key]["scores"].get(task, 0)
                d = all_results[key]["deltas"].get(task, 0)
                row += f" | {s:.4f} | {d:+.4f}"
            row += " |"
            f.write(row + "\n")
        f.write("\n")
    log.info(f"Saved: {md_path}")

    # --- Summary ---
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    mean_baseline = sum(baseline_scores.values()) / len(baseline_scores)
    log.info(f"Baseline mean accuracy: {mean_baseline:.4f}")
    for skip_idx in args.skip_layers:
        key = f"skip_{skip_idx}"
        skip_scores = all_results[key]["scores"]
        mean_skip = sum(skip_scores.values()) / len(skip_scores)
        mean_delta = mean_skip - mean_baseline
        log.info(f"Skip-{skip_idx} mean accuracy: {mean_skip:.4f} (Δ={mean_delta:+.4f})")


if __name__ == "__main__":
    main()
