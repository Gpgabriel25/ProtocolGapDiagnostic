#!/usr/bin/env python3
"""
Skip-Layer Compression Test — GPT-2-Medium
=============================================
Remove one or more layers from GPT-2-Medium and evaluate perplexity
on WikiText-2 to test whether bisimilar layers can be safely dropped.

Tests:
  1. Baseline perplexity (all 24 layers)
  2. Skip layer 5 (best bisimilar pair: 4↔5, mean_KL=0.035)
  3. Skip layer 15 (second-best pair: 14↔15, mean_KL=0.037)
  4. Skip layers 5 + 15 (two removals)
  5. Skip middle cluster layer 13 (from 12-17 redundant band)
  6. Skip layers 5, 13, 15 (three removals = 12.5% compression)

Metric: Perplexity on WikiText-2 test set (stride-based sliding window).
"""

import os
import sys
import time
import json
import logging
import argparse
import math

import torch
import torch.nn.functional as F
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ---------------------------------------------------------------------------
CYCLE_ID   = "2026-03-30T16-10-52"
MODEL_NAME = "gpt2-medium"
REPORT_DIR = f"reports/{CYCLE_ID}"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 1024   # context window for perplexity evaluation
STRIDE     = 512    # sliding window stride

os.makedirs(REPORT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Skip-layer model wrapper
# ---------------------------------------------------------------------------
class SkipLayerGPT2:
    """Wraps a GPT-2 model to skip specified layers during forward pass."""

    def __init__(self, model, skip_layers):
        self.model = model
        self.skip_layers = set(skip_layers)
        self._original_blocks = list(model.transformer.h)
        # Build new block list excluding skipped layers
        new_blocks = [b for i, b in enumerate(self._original_blocks)
                      if i not in self.skip_layers]
        model.transformer.h = torch.nn.ModuleList(new_blocks)
        log.info(f"SkipLayerGPT2: Skipping layers {sorted(skip_layers)}, "
                 f"{len(self._original_blocks)} → {len(new_blocks)} layers")

    def restore(self):
        """Restore original layers."""
        self.model.transformer.h = torch.nn.ModuleList(self._original_blocks)

    def __enter__(self):
        return self.model

    def __exit__(self, *args):
        self.restore()


# ---------------------------------------------------------------------------
# Perplexity evaluation (sliding window)
# ---------------------------------------------------------------------------
def evaluate_perplexity(model, tokenizer, text, max_length=MAX_LENGTH, stride=STRIDE):
    """Compute perplexity using sliding-window approach."""
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(DEVICE)
    seq_len = input_ids.size(1)

    nlls = []
    n_tokens = 0
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end  # tokens we score in this window

        input_chunk = input_ids[:, begin:end]
        target_chunk = input_chunk.clone()
        # Mask out tokens before the stride boundary (already scored)
        target_chunk[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_chunk, labels=target_chunk)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood.item())
        n_tokens += trg_len
        prev_end = end

        if end == seq_len:
            break

    ppl = math.exp(sum(nlls) / n_tokens)
    return ppl, n_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit WikiText-2 test samples (for faster debug)")
    args = parser.parse_args()

    log.info(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval().to(DEVICE)

    # Load WikiText-2 test set
    log.info("Loading WikiText-2 test set...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # Concatenate all text (standard perplexity evaluation)
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    # Truncate for CPU feasibility (full set is ~240K words → hours on CPU)
    max_words = args.max_samples if args.max_samples else 20000
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    log.info(f"Test text: {len(text)} chars, ~{len(text.split())} words")

    # Define skip configurations to test
    configs = [
        ("baseline",          []),
        ("skip_5",            [5]),    # best bisimilar pair 4↔5
        ("skip_15",           [15]),   # second-best pair 14↔15
        ("skip_13",           [13]),   # middle cluster
        ("skip_5_15",         [5, 15]),
        ("skip_5_13_15",      [5, 13, 15]),
    ]

    results = {}
    for name, skip in configs:
        log.info(f"\n{'='*60}")
        log.info(f"Config: {name} (skip={skip})")
        log.info(f"{'='*60}")

        t0 = time.time()
        if skip:
            wrapper = SkipLayerGPT2(model, skip)
            ppl, n_tok = evaluate_perplexity(model, tokenizer, text)
            wrapper.restore()
        else:
            ppl, n_tok = evaluate_perplexity(model, tokenizer, text)
        elapsed = time.time() - t0

        results[name] = {
            "skip_layers": skip,
            "n_layers": 24 - len(skip),
            "perplexity": round(ppl, 4),
            "n_tokens": n_tok,
            "time_s": round(elapsed, 1),
        }
        log.info(f"  PPL = {ppl:.4f}  ({n_tok} tokens, {elapsed:.1f}s)")

    # Compute relative changes
    baseline_ppl = results["baseline"]["perplexity"]
    for name, r in results.items():
        r["ppl_delta_pct"] = round(
            100.0 * (r["perplexity"] - baseline_ppl) / baseline_ppl, 2
        )

    # Write results
    report_path = os.path.join(REPORT_DIR, "skip_layer_results.md")
    json_path = os.path.join(REPORT_DIR, "skip_layer_results.json")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved: {json_path}")

    with open(report_path, "w") as f:
        f.write("# Skip-Layer Compression Test Results\n\n")
        f.write(f"**Model:** {MODEL_NAME} (24 layers)\n")
        f.write(f"**Dataset:** WikiText-2 test set\n")
        f.write(f"**Context:** {MAX_LENGTH} tokens, stride {STRIDE}\n\n")
        f.write("## Results\n\n")
        f.write("| Config | Layers Skipped | # Layers | Perplexity | Δ PPL (%) | Time |\n")
        f.write("|--------|---------------|----------|------------|-----------|------|\n")
        for name, r in results.items():
            skip_str = str(r["skip_layers"]) if r["skip_layers"] else "none"
            f.write(f"| {name} | {skip_str} | {r['n_layers']} | "
                    f"{r['perplexity']:.4f} | {r['ppl_delta_pct']:+.2f}% | "
                    f"{r['time_s']:.0f}s |\n")
        f.write(f"\n## Interpretation\n\n")
        f.write(f"Baseline perplexity: {baseline_ppl:.4f}\n\n")
        for name, r in results.items():
            if name == "baseline":
                continue
            verdict = "EXCELLENT" if abs(r["ppl_delta_pct"]) < 1.0 else \
                      "GOOD" if abs(r["ppl_delta_pct"]) < 5.0 else \
                      "MODERATE" if abs(r["ppl_delta_pct"]) < 15.0 else "POOR"
            f.write(f"- **{name}**: {r['perplexity']:.4f} ({r['ppl_delta_pct']:+.2f}%) — {verdict}\n")
    log.info(f"Saved: {report_path}")

    # Summary
    log.info(f"\n{'='*60}")
    log.info("SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"Baseline PPL: {baseline_ppl:.4f}")
    for name, r in results.items():
        if name != "baseline":
            log.info(f"  {name}: PPL={r['perplexity']:.4f} ({r['ppl_delta_pct']:+.2f}%)")


if __name__ == "__main__":
    main()
