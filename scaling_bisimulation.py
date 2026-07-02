#!/usr/bin/env python3
"""
Unified Bisimulation Scaling Experiment
========================================
Tests bisimulation distance across multiple model families (GPT-2, Pythia, OPT).
For each pair of transformer layers (i, j) with |i-j| <= max_gap, swaps layer
weights and measures KL divergence of output logits vs. baseline.

Supported models:
  GPT-2:   gpt2, gpt2-medium, gpt2-large, gpt2-xl
  Pythia:  EleutherAI/pythia-160m, pythia-410m, pythia-1.4b, pythia-2.8b
  OPT:     facebook/opt-350m, facebook/opt-1.3b

Usage:
  python scaling_bisimulation.py --model gpt2-medium --max-gap 1 --n-prompts 20
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

from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CYCLE_ID   = "2026-03-31T00-18-24"
REPORT_DIR = f"reports/{CYCLE_ID}"
LOG_DIR    = f"logs/{CYCLE_ID}"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT_SEEDS = [
    "The French Revolution was a period of radical political and societal change in France that began with the Estates General of 1789.",
    "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles.",
    "The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America.",
    "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics.",
    "The Great Wall of China is a series of fortifications that were built across the historical northern borders of ancient Chinese states.",
    "DNA, or deoxyribonucleic acid, is a molecule composed of two polynucleotide chains that coil around each other to form a double helix.",
    "The Industrial Revolution was the transition to new manufacturing processes in Great Britain, continental Europe, and the United States.",
    "Shakespeare's works have been translated into every major language and are performed more often than those of any other playwright.",
    "Black holes are regions of spacetime where gravity is so strong that nothing, not even light or other electromagnetic waves, has enough speed to escape.",
    "The human immune system is a complex network of cells, tissues, and organs that work together to defend the body against attacks by pathogens.",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):",
    "import numpy as np\nimport torch\nimport torch.nn as nn\n\nclass TransformerBlock(nn.Module):\n    def __init__(self, d_model, n_heads):\n        super().__init__()",
    "SELECT u.name, COUNT(o.id) as order_count, SUM(o.total) as revenue\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id\nWHERE o.created_at > '2024-01-01'\nGROUP BY u.id, u.name",
    "class BinarySearchTree:\n    def __init__(self):\n        self.root = None\n\n    def insert(self, val):\n        if not self.root:\n            self.root = TreeNode(val)",
    "Customer: Hi, I'd like to return this jacket. I bought it last week but the zipper is broken.\nAgent: I'm sorry to hear that! Could you please provide your order number?",
    "Scientists announced today the discovery of a potentially habitable exoplanet orbiting a nearby star system.",
    "Two roads diverged in a yellow wood, / And sorry I could not travel both / And be one traveler, long I stood",
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness.",
    "The Fibonacci sequence is defined by F(0)=0, F(1)=1, and F(n)=F(n-1)+F(n-2). The golden ratio phi = (1+sqrt(5))/2.",
    "Theorem: There are infinitely many prime numbers. Proof: Assume for contradiction that there are finitely many primes p1, p2, ..., pn.",
    "To make sourdough bread: 1) Create a starter from flour and water. 2) Feed it daily for 5-7 days.",
    "Setting up a Python virtual environment: First, ensure Python 3.8+ is installed. Run 'python -m venv venv'.",
    "The stock market experienced significant volatility today as investors reacted to new data showing inflation remained stubbornly high.",
    "In the year 2050, artificial intelligence had transformed every aspect of human civilization in ways no one predicted.",
    "Climate change poses one of the greatest challenges facing humanity in the twenty-first century, requiring urgent global cooperation.",
    "The mitochondria is the powerhouse of the cell, responsible for producing adenosine triphosphate through oxidative phosphorylation.",
    "Once upon a time in a land far away, there lived a young princess who dreamed of exploring the world beyond her kingdom.",
    "Given a 3x3 matrix A = [[1,2,3],[4,5,6],[7,8,9]], compute the determinant using cofactor expansion.",
    "Dear Sir or Madam, I am writing to express my concern regarding the recent changes to the community development plan.",
    "The Beatles, formed in Liverpool in 1960, became the most commercially successful and critically acclaimed band in popular music.",
]


def build_diverse_prompts(n: int) -> list:
    """Generate n diverse prompts by cycling through seeds with suffix variation."""
    prompts = []
    suffixes = [
        "", " Furthermore,", " In addition,", " However,", " As a result,",
        " This means that", " Consequently,", " For example,", " In contrast,",
        " Similarly,", " On the other hand,", " Notably,", " First,", " Finally,",
        " Despite this,", " Given that", " It is worth noting that",
        " Recent research suggests", " According to experts,", " In practice,",
    ]
    seed_i, suf_i = 0, 0
    while len(prompts) < n:
        prompt = PROMPT_SEEDS[seed_i % len(PROMPT_SEEDS)] + suffixes[suf_i % len(suffixes)]
        prompts.append(prompt)
        seed_i += 1
        if seed_i % len(PROMPT_SEEDS) == 0:
            suf_i += 1
    return prompts[:n]


# ---------------------------------------------------------------------------
# Model architecture helpers
# ---------------------------------------------------------------------------
def get_layers(model):
    """Return the nn.ModuleList of transformer layers for any supported model."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h          # GPT-2
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers        # Pythia / GPT-NeoX
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        return model.model.decoder.layers   # OPT
    raise ValueError(f"Unsupported model architecture: {type(model).__name__}")


def get_num_layers(config):
    """Extract layer count from model config (different field names per arch)."""
    for attr in ("n_layer", "num_hidden_layers", "num_layers"):
        if hasattr(config, attr):
            return getattr(config, attr)
    raise ValueError(f"Cannot determine layer count from config: {config}")


def get_num_params(model) -> int:
    """Total parameter count."""
    return sum(p.numel() for p in model.parameters())


def model_short_name(model_name: str) -> str:
    """E.g. 'EleutherAI/pythia-1.4b' -> 'pythia-1.4b'."""
    return model_name.split("/")[-1]


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------
def kl_div_pair(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(P || Q) where P=baseline, Q=swapped. Uses log-space for stability."""
    log_p = F.log_softmax(p_logits, dim=-1).double()
    log_q = F.log_softmax(q_logits, dim=-1).double()
    p = log_p.exp()
    kl = F.kl_div(log_q, p, reduction="sum", log_target=False)
    return float(kl.item())


# ---------------------------------------------------------------------------
# Baseline logits
# ---------------------------------------------------------------------------
def compute_baseline_logits(model, tokenizer, prompts, max_length, log):
    log.info(f"Computing baseline logits for {len(prompts)} prompts on {DEVICE}...")
    model.eval()
    baselines = []
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=max_length, padding=False).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        baselines.append(outputs.logits[0, -1].float().cpu().clone())
        if (i + 1) % 50 == 0:
            log.info(f"  Baseline: {i+1}/{len(prompts)}")
    return baselines


# ---------------------------------------------------------------------------
# Bisimulation distance for one pair
# ---------------------------------------------------------------------------
def compute_pair_distance(model, tokenizer, layer_a, layer_b, prompts,
                          baseline_logits, layers, max_length):
    """
    Swap layer_a weights with layer_b's in both directions, measure KL divergence.
    Returns conservative (max of both directions) stats.
    """
    results = {}
    model.eval()

    for direction, (src, tgt) in enumerate([(layer_b, layer_a), (layer_a, layer_b)]):
        original_state = {k: v.clone() for k, v in layers[tgt].state_dict().items()}
        layers[tgt].load_state_dict(layers[src].state_dict())

        kl_vals = []
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=max_length, padding=False).to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)
            swapped_logits = outputs.logits[0, -1].float().cpu()
            kl = kl_div_pair(baseline_logits[i], swapped_logits)
            kl_vals.append(kl)

        layers[tgt].load_state_dict(original_state)

        results[f"dir{direction}"] = {
            "mean_kl":   float(np.mean(kl_vals)),
            "max_kl":    float(np.max(kl_vals)),
            "median_kl": float(np.median(kl_vals)),
            "p95_kl":    float(np.percentile(kl_vals, 95)),
            "min_kl":    float(np.min(kl_vals)),
        }

    # Conservative: max across both directions
    mean_kl  = max(results["dir0"]["mean_kl"],  results["dir1"]["mean_kl"])
    max_kl   = max(results["dir0"]["max_kl"],   results["dir1"]["max_kl"])
    median_kl = max(results["dir0"]["median_kl"], results["dir1"]["median_kl"])
    p95_kl   = max(results["dir0"]["p95_kl"],   results["dir1"]["p95_kl"])

    return {
        "mean_kl":   round(mean_kl, 6),
        "max_kl":    round(max_kl, 6),
        "median_kl": round(median_kl, 6),
        "p95_kl":    round(p95_kl, 6),
        "dir0":      results["dir0"],
        "dir1":      results["dir1"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Unified bisimulation scaling experiment across GPT-2 / Pythia / OPT families.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scaling_bisimulation.py --model gpt2-medium --max-gap 1 --n-prompts 20
  python scaling_bisimulation.py --model EleutherAI/pythia-1.4b --max-gap 2
  python scaling_bisimulation.py --model facebook/opt-350m
""",
    )
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name (e.g. gpt2, EleutherAI/pythia-1.4b, facebook/opt-350m)")
    parser.add_argument("--max-gap", type=int, default=1,
                        help="Max layer gap to test (default: 1, adjacent only)")
    parser.add_argument("--n-prompts", type=int, default=20,
                        help="Number of diverse prompts (default: 20)")
    parser.add_argument("--max-length", type=int, default=64,
                        help="Max token length per prompt (default: 64)")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16"],
                        help="Model dtype (default: float32)")
    parser.add_argument("--cycle-id", type=str, default=CYCLE_ID,
                        help="Cycle ID for report directory")
    args = parser.parse_args()

    report_dir = f"reports/{args.cycle_id}"
    log_dir = f"logs/{args.cycle_id}"
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    short_name = model_short_name(args.model)
    log_file = os.path.join(log_dir, f"{short_name}_bisim.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    log = logging.getLogger(__name__)

    dtype = torch.float32 if args.dtype == "float32" else torch.float16

    log.info("=" * 70)
    log.info(f"Bisimulation Scaling — {args.model}")
    log.info(f"Device: {DEVICE} | Prompts: {args.n_prompts} | MaxLen: {args.max_length} | MaxGap: {args.max_gap}")
    log.info("=" * 70)

    # Load model
    log.info(f"Loading {args.model} (dtype={args.dtype})...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model = model.to(DEVICE)
    model.eval()

    layers = get_layers(model)
    n_layers = len(layers)
    n_params = get_num_params(model)
    log.info(f"Loaded in {time.time()-t0:.1f}s — {n_layers} layers, {n_params/1e6:.1f}M params")

    # Build prompts
    prompts = build_diverse_prompts(args.n_prompts)
    log.info(f"Built {len(prompts)} diverse prompts.")

    # Baseline logits
    t1 = time.time()
    baseline_logits = compute_baseline_logits(model, tokenizer, prompts, args.max_length, log)
    log.info(f"Baseline computed in {time.time()-t1:.1f}s.")

    # Generate pairs ordered by gap
    pairs = []
    for gap in range(1, args.max_gap + 1):
        for a in range(n_layers - gap):
            pairs.append((a, a + gap, gap))

    log.info(f"Testing {len(pairs)} pairs (max gap={args.max_gap})...")
    results = []
    t2 = time.time()

    for idx, (a, b, gap) in enumerate(pairs):
        t_pair = time.time()
        log.info(f"[{idx+1}/{len(pairs)}] Layer {a} <-> {b} (gap={gap})...")

        stats = compute_pair_distance(
            model, tokenizer, a, b, prompts, baseline_logits, layers, args.max_length
        )

        elapsed = time.time() - t_pair
        log.info(f"  mean_KL={stats['mean_kl']:.4f}  max_KL={stats['max_kl']:.4f}  [{elapsed:.1f}s]")

        results.append({
            "layer_a": a,
            "layer_b": b,
            "gap": gap,
            "mean_kl": stats["mean_kl"],
            "max_kl": stats["max_kl"],
            "median_kl": stats["median_kl"],
            "p95_kl": stats["p95_kl"],
            "a_to_b_mean": round(stats["dir0"]["mean_kl"], 6),
            "b_to_a_mean": round(stats["dir1"]["mean_kl"], 6),
            "elapsed_s": round(elapsed, 2),
        })

    total_time = time.time() - t2
    log.info(f"All pairs done in {total_time/60:.1f} min")

    # Sort by mean_kl
    results.sort(key=lambda x: x["mean_kl"])

    # Counts
    strong_go = sum(1 for r in results if r["mean_kl"] < 0.05)
    cond_go   = sum(1 for r in results if r["mean_kl"] < 0.10)

    # Save JSON
    json_path = os.path.join(report_dir, f"{short_name}_bisimulation.json")
    output_data = {
        "model": args.model,
        "short_name": short_name,
        "n_layers": n_layers,
        "n_params": n_params,
        "n_prompts": args.n_prompts,
        "max_length": args.max_length,
        "max_gap": args.max_gap,
        "device": DEVICE,
        "dtype": args.dtype,
        "total_time_s": round(total_time, 2),
        "strong_go_count": strong_go,
        "cond_go_count": cond_go,
        "pairs": results,
    }
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    log.info(f"Saved: {json_path}")

    # Write markdown report
    md_path = os.path.join(report_dir, f"{short_name}_bisimulation.md")
    best = results[0] if results else None
    with open(md_path, "w") as f:
        f.write(f"# {args.model} Bisimulation Test\n\n")
        f.write(f"**Model:** {args.model} ({n_layers} layers, {n_params/1e6:.1f}M params)\n")
        f.write(f"**Pairs tested:** {len(pairs)} (max gap={args.max_gap})\n")
        f.write(f"**Prompts:** {args.n_prompts} | **Max length:** {args.max_length}\n")
        f.write(f"**Device:** {DEVICE} | **dtype:** {args.dtype}\n")
        f.write(f"**Time:** {total_time/60:.1f} min\n\n")

        f.write("## Summary\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| STRONG GO (mean_KL < 0.05) | **{strong_go}** |\n")
        f.write(f"| COND GO (mean_KL < 0.10) | **{cond_go}** |\n")
        if best:
            f.write(f"| Best pair | {best['layer_a']} <-> {best['layer_b']} |\n")
            f.write(f"| Best mean_KL | {best['mean_kl']:.6f} |\n")
        f.write("\n")

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

        # Summary comparison table header
        f.write("\n## Scaling Summary Row\n\n")
        f.write("```\n")
        f.write(f"Model: {short_name}\n")
        f.write(f"Params: {n_params/1e6:.1f}M\n")
        f.write(f"Layers: {n_layers}\n")
        if best:
            f.write(f"Best Pair: {best['layer_a']}<->{best['layer_b']}\n")
            f.write(f"Best KL: {best['mean_kl']:.6f}\n")
        f.write(f"STRONG GO (<0.05): {strong_go}\n")
        f.write(f"COND GO (<0.10): {cond_go}\n")
        f.write("```\n")

    log.info(f"Saved: {md_path}")

    # Print final summary table row
    log.info("")
    log.info("=" * 90)
    log.info(f"{'Model':<25} {'Params':>10} {'Layers':>7} {'Best Pair':>12} "
             f"{'Best KL':>10} {'STRONG GO':>10} {'COND GO':>10}")
    log.info("-" * 90)
    best_pair_str = f"{best['layer_a']}<->{best['layer_b']}" if best else "N/A"
    best_kl_str = f"{best['mean_kl']:.6f}" if best else "N/A"
    log.info(f"{short_name:<25} {n_params/1e6:>9.1f}M {n_layers:>7} {best_pair_str:>12} "
             f"{best_kl_str:>10} {strong_go:>10} {cond_go:>10}")
    log.info("=" * 90)


if __name__ == "__main__":
    main()
