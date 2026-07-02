#!/usr/bin/env python3
"""
Head-Level Bisimulation — GPT-2-Medium
=========================================
For each attention head within near-bisimilar layer pairs,
swap that single head and measure output KL divergence.

This reveals which specific heads drive the similarity (or difference)
between layers identified as approximately bisimilar.

Targets: Top-5 bisimilar pairs from the full experiment.
"""

import os
import sys
import time
import json
import logging
import argparse
import copy

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoTokenizer

CYCLE_ID   = "2026-03-30T16-10-52"
MODEL_NAME = "gpt2-medium"
REPORT_DIR = f"reports/{CYCLE_ID}"
DEVICE     = "cpu"
N_PROMPTS  = 20
MAX_LENGTH = 64
N_HEADS    = 16  # GPT-2-Medium has 16 heads per layer
HEAD_DIM   = 64  # 1024 / 16

os.makedirs(REPORT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Top bisimilar pairs from full experiment
TARGET_PAIRS = [
    (4, 5),     # mean_KL=0.035
    (14, 15),   # mean_KL=0.037
    (16, 17),   # mean_KL=0.039
    (15, 16),   # mean_KL=0.039
    (12, 14),   # mean_KL=0.039
]

# Diverse prompts
PROMPT_TEMPLATES = [
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


def get_head_params(layer_block, head_idx, n_heads=N_HEADS, head_dim=HEAD_DIM):
    """Extract parameter slices for a specific attention head."""
    attn = layer_block.attn
    # GPT-2 stores Q, K, V concatenated in c_attn.weight [embed_dim, 3*embed_dim]
    # and output projection in c_proj.weight [embed_dim, embed_dim]
    embed_dim = n_heads * head_dim

    # c_attn: [embed_dim, 3*embed_dim] — columns split into Q, K, V each [embed_dim, embed_dim]
    # Within each of Q, K, V, columns [head_idx*head_dim : (head_idx+1)*head_dim] are this head
    h_start = head_idx * head_dim
    h_end = (head_idx + 1) * head_dim

    # Extract slices (indices into c_attn columns)
    q_cols = slice(h_start, h_end)
    k_cols = slice(embed_dim + h_start, embed_dim + h_end)
    v_cols = slice(2 * embed_dim + h_start, 2 * embed_dim + h_end)

    # c_proj: rows [h_start:h_end] are this head's contribution
    proj_rows = slice(h_start, h_end)

    return {
        "q_cols": q_cols,
        "k_cols": k_cols,
        "v_cols": v_cols,
        "proj_rows": proj_rows,
    }


def swap_single_head(model, layer_a, layer_b, head_idx):
    """
    Swap a single attention head from layer_b into layer_a's position.
    Returns backup for restoration.
    """
    block_a = model.transformer.h[layer_a]
    block_b = model.transformer.h[layer_b]

    slices = get_head_params(block_a, head_idx)

    # Backup layer_a's head params
    backup = {
        "c_attn_w_q": block_a.attn.c_attn.weight[:, slices["q_cols"]].clone(),
        "c_attn_w_k": block_a.attn.c_attn.weight[:, slices["k_cols"]].clone(),
        "c_attn_w_v": block_a.attn.c_attn.weight[:, slices["v_cols"]].clone(),
        "c_attn_b_q": block_a.attn.c_attn.bias[slices["q_cols"]].clone(),
        "c_attn_b_k": block_a.attn.c_attn.bias[slices["k_cols"]].clone(),
        "c_attn_b_v": block_a.attn.c_attn.bias[slices["v_cols"]].clone(),
        "c_proj_w": block_a.attn.c_proj.weight[slices["proj_rows"], :].clone(),
        "c_proj_b_share": block_a.attn.c_proj.bias.clone(),  # shared across heads but save for safety
    }

    # Copy layer_b's head params into layer_a
    with torch.no_grad():
        block_a.attn.c_attn.weight[:, slices["q_cols"]].copy_(
            block_b.attn.c_attn.weight[:, slices["q_cols"]]
        )
        block_a.attn.c_attn.weight[:, slices["k_cols"]].copy_(
            block_b.attn.c_attn.weight[:, slices["k_cols"]]
        )
        block_a.attn.c_attn.weight[:, slices["v_cols"]].copy_(
            block_b.attn.c_attn.weight[:, slices["v_cols"]]
        )
        block_a.attn.c_attn.bias[slices["q_cols"]].copy_(
            block_b.attn.c_attn.bias[slices["q_cols"]]
        )
        block_a.attn.c_attn.bias[slices["k_cols"]].copy_(
            block_b.attn.c_attn.bias[slices["k_cols"]]
        )
        block_a.attn.c_attn.bias[slices["v_cols"]].copy_(
            block_b.attn.c_attn.bias[slices["v_cols"]]
        )
        block_a.attn.c_proj.weight[slices["proj_rows"], :].copy_(
            block_b.attn.c_proj.weight[slices["proj_rows"], :]
        )

    return backup


def restore_head(model, layer_a, head_idx, backup):
    """Restore a head from backup."""
    block_a = model.transformer.h[layer_a]
    slices = get_head_params(block_a, head_idx)

    with torch.no_grad():
        block_a.attn.c_attn.weight[:, slices["q_cols"]].copy_(backup["c_attn_w_q"])
        block_a.attn.c_attn.weight[:, slices["k_cols"]].copy_(backup["c_attn_w_k"])
        block_a.attn.c_attn.weight[:, slices["v_cols"]].copy_(backup["c_attn_w_v"])
        block_a.attn.c_attn.bias[slices["q_cols"]].copy_(backup["c_attn_b_q"])
        block_a.attn.c_attn.bias[slices["k_cols"]].copy_(backup["c_attn_b_k"])
        block_a.attn.c_attn.bias[slices["v_cols"]].copy_(backup["c_attn_b_v"])
        block_a.attn.c_proj.weight[slices["proj_rows"], :].copy_(backup["c_proj_w"])


def main():
    log.info(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval().to(DEVICE)

    prompts = PROMPT_TEMPLATES[:N_PROMPTS]

    # Compute baseline logits
    log.info("Computing baseline logits...")
    baseline_logits = {}
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        baseline_logits[i] = outputs.logits[0, -1].clone()

    # For each target pair, swap each head individually
    all_results = {}
    total_tests = len(TARGET_PAIRS) * N_HEADS
    done = 0

    for layer_a, layer_b in TARGET_PAIRS:
        pair_key = f"{layer_a}_{layer_b}"
        head_kl = np.zeros(N_HEADS)

        log.info(f"\nPair ({layer_a}, {layer_b}): testing {N_HEADS} individual head swaps")

        for h in range(N_HEADS):
            # Swap head h from layer_b into layer_a
            backup = swap_single_head(model, layer_a, layer_b, h)

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

            restore_head(model, layer_a, h, backup)

            mean_kl = np.mean(kl_divs)
            head_kl[h] = mean_kl
            done += 1
            log.info(f"  Head {h:2d}: mean_KL={mean_kl:.6f} [{done}/{total_tests}]")

        all_results[pair_key] = {
            "layer_a": layer_a,
            "layer_b": layer_b,
            "head_kl": head_kl.tolist(),
            "min_head": int(np.argmin(head_kl)),
            "max_head": int(np.argmax(head_kl)),
            "min_kl": float(np.min(head_kl)),
            "max_kl": float(np.max(head_kl)),
            "mean_kl": float(np.mean(head_kl)),
        }

    # Save results
    json_path = os.path.join(REPORT_DIR, "head_bisimulation.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nSaved: {json_path}")

    # Generate per-pair head barplots
    fig, axes = plt.subplots(len(TARGET_PAIRS), 1, figsize=(12, 3 * len(TARGET_PAIRS)),
                              sharex=True)
    if len(TARGET_PAIRS) == 1:
        axes = [axes]

    for idx, (pair_key, res) in enumerate(all_results.items()):
        ax = axes[idx]
        kls = res["head_kl"]
        colors = ['#2ecc71' if k < np.mean(kls) else '#e74c3c' for k in kls]
        ax.bar(range(N_HEADS), kls, color=colors)
        ax.set_ylabel("Mean KL")
        ax.set_title(f"Layer {res['layer_a']} ↔ {res['layer_b']}: "
                     f"Head-level bisimulation distance")
        ax.axhline(y=np.mean(kls), color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(range(N_HEADS))

    axes[-1].set_xlabel("Head index")
    plt.tight_layout()
    fig_path = os.path.join(REPORT_DIR, "head_bisimulation.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    log.info(f"Saved: {fig_path}")

    # Write report
    report_path = os.path.join(REPORT_DIR, "head_bisimulation.md")
    with open(report_path, "w") as f:
        f.write("# Head-Level Bisimulation Analysis\n\n")
        f.write(f"**Model:** {MODEL_NAME} ({N_HEADS} heads per layer)\n")
        f.write(f"**Prompts:** {N_PROMPTS}\n\n")
        f.write("## Summary\n\n")
        f.write("| Pair | Min Head KL | Max Head KL | Mean Head KL | Most Similar | Most Different |\n")
        f.write("|------|------------|------------|-------------|-------------|----------------|\n")
        for pair_key, res in all_results.items():
            f.write(f"| ({res['layer_a']},{res['layer_b']}) | "
                    f"{res['min_kl']:.6f} | {res['max_kl']:.6f} | "
                    f"{res['mean_kl']:.6f} | Head {res['min_head']} | "
                    f"Head {res['max_head']} |\n")
        f.write("\n## Per-Head Details\n\n")
        for pair_key, res in all_results.items():
            f.write(f"\n### Layer {res['layer_a']} ↔ {res['layer_b']}\n\n")
            f.write("| Head | Mean KL |\n|------|--------|\n")
            sorted_heads = sorted(range(N_HEADS), key=lambda h: res["head_kl"][h])
            for h in sorted_heads:
                f.write(f"| {h} | {res['head_kl'][h]:.6f} |\n")

    log.info(f"Saved: {report_path}")
    log.info("\nHead-level bisimulation complete.")


if __name__ == "__main__":
    main()
