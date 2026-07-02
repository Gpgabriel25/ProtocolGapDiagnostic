#!/usr/bin/env python3
"""
Pythia-1.4B Bisimulation — Adjacent Pairs Only
=================================================
Quick comparison: do bigger models have more layer redundancy?
Tests only adjacent pairs (gap=1) for speed.
"""

import os
import sys
import time
import json
import logging
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer

CYCLE_ID   = "2026-03-30T16-10-52"
MODEL_NAME = "EleutherAI/pythia-1.4b"
REPORT_DIR = f"reports/{CYCLE_ID}"
DEVICE     = "cpu"
N_PROMPTS  = 20
MAX_LENGTH = 64

os.makedirs(REPORT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PROMPT_SEEDS = [
    "The French Revolution was a period of radical political and societal change in France that began",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) +",
    "In a quiet village at the edge of the mountains, the old woman told stories of",
    "Breaking news: Scientists have discovered a new species of deep-sea creature that",
    "The theory of general relativity, published by Albert Einstein in 1915, describes",
    "Once upon a time in a land far away, there lived a young princess who",
    "import numpy as np\nimport torch\n\nclass NeuralNetwork(torch.nn.Module):\n    def __init__(self",
    "The stock market experienced significant volatility today as investors reacted to",
    "To make a perfect sourdough bread, you will need flour, water, salt, and a",
    "The quantum mechanical wave function describes the probability amplitude of finding",
    "Dear Sir or Madam, I am writing to express my concern regarding the recent changes",
    "In the year 2050, artificial intelligence had transformed every aspect of human",
    "The mitochondria is the powerhouse of the cell, responsible for producing adenosine",
    "SELECT users.name, orders.total FROM users INNER JOIN orders ON users.id =",
    "The Beatles, formed in Liverpool in 1960, became the most commercially successful",
    "According to the latest census data, the population of the metropolitan area has",
    "Roses are red, violets are blue, the sun is shining and the sky is",
    "The fundamental theorem of calculus establishes the relationship between differentiation and",
    "In a shocking turn of events, the defending champions were eliminated from the tournament",
    "Climate change poses one of the greatest challenges facing humanity in the twenty-first",
]


def get_layer_accessor(model):
    """Get the layer list attribute depending on model architecture."""
    if hasattr(model, 'gpt_neox'):
        return model.gpt_neox.layers  # Pythia / GPT-NeoX
    elif hasattr(model, 'transformer'):
        return model.transformer.h  # GPT-2
    else:
        raise ValueError(f"Unknown model architecture: {type(model)}")


def compute_pair_kl(model, tokenizer, layer_a, layer_b, prompts, baseline_logits, layers_attr):
    """Swap layer_a weights with layer_b's, measure KL divergence."""
    layers = layers_attr

    # Backup layer_a
    orig_state = {k: v.clone() for k, v in layers[layer_a].state_dict().items()}

    # Swap: put layer_b's weights into layer_a's position
    layers[layer_a].load_state_dict(layers[layer_b].state_dict())

    kl_divs = []
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        swapped_logits = outputs.logits[0, -1]

        p = F.softmax(baseline_logits[i], dim=-1)
        q = F.softmax(swapped_logits, dim=-1)
        kl = F.kl_div(q.log(), p, reduction='sum').item()
        kl_divs.append(abs(kl))

    # Restore
    layers[layer_a].load_state_dict(orig_state)

    return {
        'mean_kl': float(np.mean(kl_divs)),
        'max_kl': float(np.max(kl_divs)),
        'median_kl': float(np.median(kl_divs)),
        'p95_kl': float(np.percentile(kl_divs, 95)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-gap", type=int, default=3,
                        help="Max layer gap to test (default: 3 for speed)")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16"])
    args = parser.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.float16

    log.info(f"Loading model: {MODEL_NAME} (dtype={args.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    model.eval().to(DEVICE)

    layers_attr = get_layer_accessor(model)
    n_layers = len(layers_attr)
    log.info(f"Model loaded: {n_layers} layers")

    prompts = PROMPT_SEEDS[:N_PROMPTS]

    # Baseline logits
    log.info("Computing baseline logits...")
    baseline_logits = {}
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        baseline_logits[i] = outputs.logits[0, -1].float().clone()  # float for KL

    # Generate pairs to test
    pairs = []
    for gap in range(1, args.max_gap + 1):
        for a in range(n_layers - gap):
            pairs.append((a, a + gap, gap))
    log.info(f"Testing {len(pairs)} pairs (max gap={args.max_gap})")

    results = []
    t0 = time.time()

    for idx, (a, b, gap) in enumerate(pairs):
        log.info(f"[{idx+1}/{len(pairs)}] Layer {a} <-> {b} (gap={gap})...")
        t1 = time.time()

        # Direction 1: a→b
        r1 = compute_pair_kl(model, tokenizer, a, b, prompts, baseline_logits, layers_attr)
        # Direction 2: b→a
        r2 = compute_pair_kl(model, tokenizer, b, a, prompts, baseline_logits, layers_attr)

        # Take the worse (max) of both directions for conservative estimate
        mean_kl = max(r1['mean_kl'], r2['mean_kl'])
        max_kl = max(r1['max_kl'], r2['max_kl'])

        elapsed = time.time() - t1
        log.info(f"  mean_KL={mean_kl:.4f}  max_KL={max_kl:.4f}  [{elapsed:.1f}s]")

        results.append({
            'layer_a': a,
            'layer_b': b,
            'gap': gap,
            'mean_kl': round(mean_kl, 6),
            'max_kl': round(max_kl, 6),
            'median_kl': round(max(r1['median_kl'], r2['median_kl']), 6),
            'p95_kl': round(max(r1['p95_kl'], r2['p95_kl']), 6),
            'a_to_b_mean': round(r1['mean_kl'], 6),
            'b_to_a_mean': round(r2['mean_kl'], 6),
        })

    total_time = time.time() - t0
    log.info(f"\nAll pairs done in {total_time/60:.1f} min")

    # Sort by mean_kl
    results.sort(key=lambda x: x['mean_kl'])

    # Save JSON
    json_path = os.path.join(REPORT_DIR, "pythia_bisimulation.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved: {json_path}")

    # Count by threshold
    strong = sum(1 for r in results if r['mean_kl'] < 0.05)
    cond = sum(1 for r in results if r['mean_kl'] < 0.10)
    weak_above = sum(1 for r in results if r['mean_kl'] > 0.50)

    # Write report
    report_path = os.path.join(REPORT_DIR, "pythia_bisimulation.md")
    with open(report_path, "w") as f:
        f.write("# Pythia-1.4B Bisimulation Test\n\n")
        f.write(f"**Model:** {MODEL_NAME} ({n_layers} layers)\n")
        f.write(f"**Pairs tested:** {len(pairs)} (max gap={args.max_gap})\n")
        f.write(f"**Prompts:** {N_PROMPTS}\n")
        f.write(f"**Time:** {total_time/60:.1f} min\n\n")
        f.write("## Verdict Counts\n\n")
        f.write(f"- STRONG GO (mean_KL < 0.05): **{strong}**\n")
        f.write(f"- COND GO (mean_KL < 0.10): **{cond}**\n")
        f.write(f"- WEAK (mean_KL > 0.50): **{weak_above}**\n\n")
        f.write("## Top-10 Closest Pairs\n\n")
        f.write("| Rank | Layer A | Layer B | Gap | Mean KL | Max KL | p95 KL |\n")
        f.write("|------|---------|---------|-----|---------|--------|--------|\n")
        for i, r in enumerate(results[:10]):
            f.write(f"| {i+1} | {r['layer_a']} | {r['layer_b']} | {r['gap']} | "
                    f"{r['mean_kl']:.4f} | {r['max_kl']:.4f} | {r['p95_kl']:.4f} |\n")

        f.write("\n## All Pairs (sorted by mean KL)\n\n")
        f.write("| Layer A | Layer B | Gap | Mean KL | Max KL |\n")
        f.write("|---------|---------|-----|---------|--------|\n")
        for r in results:
            f.write(f"| {r['layer_a']} | {r['layer_b']} | {r['gap']} | "
                    f"{r['mean_kl']:.4f} | {r['max_kl']:.4f} |\n")

        # Comparison header
        f.write("\n## Comparison vs GPT-2-Medium\n\n")
        f.write("GPT-2-Medium adjacent pair (gap=1) statistics:\n")
        f.write("- Best adjacent: layers 4↔5, mean_KL=0.035\n")
        f.write("- 16 pairs total with mean_KL < 0.05\n\n")
        f.write(f"Pythia-1.4B (gap≤{args.max_gap}):\n")
        f.write(f"- Best pair: layers {results[0]['layer_a']}↔{results[0]['layer_b']}, "
                f"mean_KL={results[0]['mean_kl']:.4f}\n")
        f.write(f"- {strong} pairs with mean_KL < 0.05\n")

    log.info(f"Saved: {report_path}")
    log.info(f"\n{'='*60}")
    log.info(f"Best pair: {results[0]['layer_a']}↔{results[0]['layer_b']} "
             f"(mean_KL={results[0]['mean_kl']:.4f})")
    log.info(f"STRONG GO pairs: {strong}, COND GO pairs: {cond}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
