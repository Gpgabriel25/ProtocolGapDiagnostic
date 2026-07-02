#!/usr/bin/env python3
"""
Weight-Sharing Bisimulation Quotient — GPT-2-Medium
=====================================================
Tests the "true bisimulation quotient" idea: instead of removing a layer,
share weights across two positions so they are literally the same computation.

Sharing means model.transformer.h[5] = model.transformer.h[4] (same object).
This preserves depth (and skip connections) while forcing the two layers to
be mathematically identical — a strict bisimulation.

Configurations:
  1. baseline            — all 24 layers, no changes
  2. skip_5              — remove layer 5 (for direct comparison)
  3. share_4_at_both     — layer 4's weights used at both positions 4 and 5
  4. share_5_at_both     — layer 5's weights used at both positions 4 and 5
  5. multi_share         — share (4,5) AND (14,15) simultaneously

Metric: Perplexity on WikiText-2 test set (stride-based sliding window).
"""

import os
import sys
import copy
import time
import json
import logging
import argparse
import math

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ---------------------------------------------------------------------------
CYCLE_ID   = "2026-03-31T00-18-24"
MODEL_NAME = "gpt2-medium"
REPORT_DIR = f"reports/{CYCLE_ID}"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 1024   # context window for perplexity evaluation
STRIDE     = 512    # sliding window stride
MAX_WORDS  = 20000  # truncate test set (same as skip_layer_test.py)

os.makedirs(REPORT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Perplexity evaluation (sliding window) — copied from skip_layer_test.py
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
        trg_len = end - prev_end  # tokens scored in this window

        input_chunk = input_ids[:, begin:end]
        target_chunk = input_chunk.clone()
        # Mask out tokens before stride boundary (already scored)
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
# Model modification helpers
# ---------------------------------------------------------------------------
def apply_sharing(model, share_pairs):
    """
    For each (keep_idx, replace_idx) pair, make h[replace_idx] point to
    the same nn.Module object as h[keep_idx].

    Returns a list of (replace_idx, original_module) for restore().
    """
    saved = []
    blocks = model.transformer.h
    for keep_idx, replace_idx in share_pairs:
        saved.append((replace_idx, blocks[replace_idx]))
        blocks[replace_idx] = blocks[keep_idx]
        log.info(f"  Shared: h[{replace_idx}] ← h[{keep_idx}] "
                 f"(id match: {id(blocks[keep_idx]) == id(blocks[replace_idx])})")
    return saved


def restore_sharing(model, saved):
    """Restore original module references."""
    blocks = model.transformer.h
    for replace_idx, original_module in saved:
        blocks[replace_idx] = original_module


def apply_skip(model, skip_layers):
    """Remove layers at given indices. Returns (original_blocks, new_len)."""
    original_blocks = list(model.transformer.h)
    new_blocks = [b for i, b in enumerate(original_blocks) if i not in set(skip_layers)]
    model.transformer.h = nn.ModuleList(new_blocks)
    log.info(f"  Skipped layers {sorted(skip_layers)}: "
             f"{len(original_blocks)} → {len(new_blocks)} layers")
    return original_blocks


def restore_skip(model, original_blocks):
    """Restore full block list after a skip-layer test."""
    model.transformer.h = nn.ModuleList(original_blocks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Weight-sharing bisimulation quotient test for GPT-2-Medium."
    )
    parser.add_argument(
        "--max-words", type=int, default=MAX_WORDS,
        help=f"Word count to truncate WikiText-2 test set (default: {MAX_WORDS})"
    )
    parser.add_argument(
        "--max-length", type=int, default=MAX_LENGTH,
        help=f"Sliding window context length (default: {MAX_LENGTH})"
    )
    parser.add_argument(
        "--stride", type=int, default=STRIDE,
        help=f"Sliding window stride (default: {STRIDE})"
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip the baseline evaluation (use cached value 19.19 from prior run)"
    )
    args = parser.parse_args()

    log.info(f"Device: {DEVICE}")
    log.info(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval().to(DEVICE)

    log.info("Loading WikiText-2 test set...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    words = text.split()
    if len(words) > args.max_words:
        text = " ".join(words[:args.max_words])
    log.info(f"Test text: {len(text)} chars, ~{len(text.split())} words")

    # ------------------------------------------------------------------
    # Configurations to evaluate
    # Each entry: (name, kind, kwargs)
    #   kind = "baseline" | "share" | "skip"
    #   share kwargs: share_pairs = [(keep_idx, replace_idx), ...]
    #   skip  kwargs: skip_layers = [idx, ...]
    # ------------------------------------------------------------------
    configs = [
        # 1. Baseline
        ("baseline",         "baseline", {}),
        # 2. Skip layer 5 (remove it) — reference for comparison
        ("skip_5",           "skip",    {"skip_layers": [5]}),
        # 3. Share: use h[4] at both positions 4 and 5
        ("share_4_at_both",  "share",   {"share_pairs": [(4, 5)]}),
        # 4. Share: use h[5] at both positions 4 and 5
        ("share_5_at_both",  "share",   {"share_pairs": [(5, 4)]}),
        # 5. Multi-pair: share (4,5) and (14,15) simultaneously
        ("multi_share",      "share",   {"share_pairs": [(4, 5), (14, 15)]}),
    ]

    results = {}

    for name, kind, kwargs in configs:
        if name == "baseline" and args.skip_baseline:
            log.info("Skipping baseline (--skip-baseline flag set).")
            results[name] = {
                "kind": "baseline",
                "n_layers": 24,
                "perplexity": 19.19,
                "n_tokens": None,
                "time_s": 0.0,
                "note": "cached from prior run",
            }
            continue

        log.info(f"\n{'='*60}")
        log.info(f"Config: {name}  kind={kind}  kwargs={kwargs}")
        log.info(f"{'='*60}")

        t0 = time.time()

        if kind == "baseline":
            ppl, n_tok = evaluate_perplexity(
                model, tokenizer, text, args.max_length, args.stride
            )
            n_layers = len(model.transformer.h)

        elif kind == "share":
            saved = apply_sharing(model, kwargs["share_pairs"])
            ppl, n_tok = evaluate_perplexity(
                model, tokenizer, text, args.max_length, args.stride
            )
            restore_sharing(model, saved)
            n_layers = len(model.transformer.h)  # unchanged (same depth)

        elif kind == "skip":
            original_blocks = apply_skip(model, kwargs["skip_layers"])
            ppl, n_tok = evaluate_perplexity(
                model, tokenizer, text, args.max_length, args.stride
            )
            restore_skip(model, original_blocks)
            n_layers = 24 - len(kwargs["skip_layers"])

        else:
            raise ValueError(f"Unknown kind: {kind}")

        elapsed = time.time() - t0
        log.info(f"  PPL = {ppl:.4f}  ({n_tok} tokens, {elapsed:.1f}s)")

        results[name] = {
            "kind": kind,
            "n_layers": n_layers,
            "perplexity": round(ppl, 4),
            "n_tokens": n_tok,
            "time_s": round(elapsed, 1),
        }

    # ------------------------------------------------------------------
    # Compute Δ% from baseline
    # ------------------------------------------------------------------
    baseline_ppl = results["baseline"]["perplexity"]
    for name, r in results.items():
        r["ppl_delta_pct"] = round(
            100.0 * (r["perplexity"] - baseline_ppl) / baseline_ppl, 2
        )

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    json_path = os.path.join(REPORT_DIR, "weight_sharing.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved: {json_path}")

    # ------------------------------------------------------------------
    # Save Markdown report
    # ------------------------------------------------------------------
    md_path = os.path.join(REPORT_DIR, "weight_sharing.md")
    with open(md_path, "w") as f:
        f.write("# Weight-Sharing Bisimulation Quotient Results\n\n")
        f.write(f"**Model:** {MODEL_NAME} (24 layers nominal)\n")
        f.write(f"**Dataset:** WikiText-2 test set (~{args.max_words} words)\n")
        f.write(f"**Context:** {args.max_length} tokens, stride {args.stride}\n\n")

        f.write("## Summary Table\n\n")
        f.write("| Config | Kind | # Layers | Perplexity | Δ PPL (%) | Notes |\n")
        f.write("|--------|------|----------|------------|-----------|-------|\n")
        for name, r in results.items():
            note = r.get("note", "")
            f.write(
                f"| {name} | {r['kind']} | {r['n_layers']} | "
                f"{r['perplexity']:.4f} | {r['ppl_delta_pct']:+.2f}% | {note} |\n"
            )

        f.write("\n## Interpretation\n\n")
        f.write(
            "Weight sharing preserves network **depth** while enforcing exact weight "
            "identity between the two positions. If perplexity is close to baseline, "
            "the two layers are functionally redundant — a true bisimulation quotient.\n\n"
        )
        f.write(f"**Baseline PPL:** {baseline_ppl:.4f}\n\n")
        for name, r in results.items():
            if name == "baseline":
                continue
            delta = r["ppl_delta_pct"]
            if abs(delta) < 1.0:
                verdict = "EXCELLENT — functionally equivalent"
            elif abs(delta) < 5.0:
                verdict = "GOOD — minor degradation"
            elif abs(delta) < 15.0:
                verdict = "MODERATE"
            else:
                verdict = "POOR"
            f.write(f"- **{name}**: {r['perplexity']:.4f} ({delta:+.2f}%) — {verdict}\n")

        f.write("\n## Key question\n\n")
        f.write(
            "If `share_4_at_both` and `share_5_at_both` both give lower Δ PPL than "
            "`skip_5`, then bisimulation quotient (sharing) is strictly better than "
            "skip compression. The ideal result is Δ < 1% for sharing vs Δ > 5% for skip.\n"
        )

    log.info(f"Saved: {md_path}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    log.info(f"\n{'='*60}")
    log.info("SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"{'Config':<22} {'PPL':>8}  {'Δ PPL':>8}  {'Layers':>6}")
    log.info(f"{'-'*50}")
    for name, r in results.items():
        sign = "+" if r["ppl_delta_pct"] >= 0 else ""
        log.info(
            f"{name:<22} {r['perplexity']:>8.4f}  "
            f"{sign}{r['ppl_delta_pct']:>7.2f}%  {r['n_layers']:>6}"
        )
    log.info(f"\nReports: {json_path}, {md_path}")


if __name__ == "__main__":
    main()
