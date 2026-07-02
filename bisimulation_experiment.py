#!/usr/bin/env python3
"""
Bisimulation Distance Experiment — GPT-2-Medium
=================================================
For each pair of transformer layers (ℓ₁, ℓ₂), swap layer weights and measure
the KL divergence of output logits vs. baseline.  Low max-KL ≈ bisimilar layers.

Outputs:
  reports/<cycle>/distance_matrix.csv
  reports/<cycle>/sorted_pairs.csv
  reports/<cycle>/heatmap.png
  reports/<cycle>/histogram.png
  reports/<cycle>/adjacency_trend.png
  reports/<cycle>/verdict.md
  logs/<cycle>/experiment.log
"""

import os
import sys
import copy
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
import matplotlib.ticker as ticker

from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CYCLE_ID    = "2026-03-30T15-15-07"
MODEL_NAME  = "gpt2-medium"
N_PROMPTS   = 200     # prompts per pair
MAX_LENGTH  = 128     # token length per prompt (shorter = faster on CPU)
REPORT_DIR  = f"reports/{CYCLE_ID}"
LOG_DIR     = f"logs/{CYCLE_ID}"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"{LOG_DIR}/experiment.log"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Diverse prompt set — built-in, no external download required
# ---------------------------------------------------------------------------
PROMPT_SEEDS = [
    # Wikipedia-style
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
    # Code-style
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):",
    "import numpy as np\nimport torch\nimport torch.nn as nn\n\nclass TransformerBlock(nn.Module):\n    def __init__(self, d_model, n_heads):\n        super().__init__()\n        self.attn = nn.MultiheadAttention(d_model, n_heads)\n        self.norm1 = nn.LayerNorm(d_model)\n        self.ff = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))\n        self.norm2 = nn.LayerNorm(d_model)",
    "SELECT u.name, COUNT(o.id) as order_count, SUM(o.total) as revenue\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id\nWHERE o.created_at > '2024-01-01'\nGROUP BY u.id, u.name\nHAVING COUNT(o.id) > 5\nORDER BY revenue DESC;",
    "class BinarySearchTree:\n    def __init__(self):\n        self.root = None\n\n    def insert(self, val):\n        if not self.root:\n            self.root = TreeNode(val)\n        else:\n            self._insert(self.root, val)\n\n    def _insert(self, node, val):",
    "#!/bin/bash\nset -euo pipefail\n\n# Deploy to production\necho \"Starting deployment...\"\ndocker build -t myapp:latest .\ndocker push registry.example.com/myapp:latest\nkubectl set image deployment/myapp myapp=registry.example.com/myapp:latest\nkubectl rollout status deployment/myapp",
    # Dialogue
    "Customer: Hi, I'd like to return this jacket. I bought it last week but the zipper is broken.\nAgent: I'm sorry to hear that! Could you please provide your order number?\nCustomer: Sure, it's ORD-884291.\nAgent: Thank you. I can see the order. Would you prefer a refund or an exchange?",
    "Doctor: Your test results came back. Your cholesterol is slightly elevated at 215 mg/dL.\nPatient: Is that dangerous?\nDoctor: Not immediately, but we should monitor it. I recommend reducing saturated fats and increasing exercise.\nPatient: How much exercise are we talking about?",
    "Alice: Have you read the new paper on attention mechanisms?\nBob: Which one? There are like a dozen this week.\nAlice: The one that replaces softmax with a sparse variant. Claims 3x speedup.\nBob: Oh, that one. I'm skeptical of the benchmark methodology.",
    # News-style
    "Scientists announced today the discovery of a potentially habitable exoplanet orbiting a nearby star system. The planet, designated Kepler-452f, is located approximately 1,400 light-years from Earth.",
    "Global markets fell sharply on Monday as investors reacted to new data showing inflation remained stubbornly high in the eurozone, reinforcing expectations that central banks would maintain high interest rates.",
    "The city council voted 7-2 to approve the new transit expansion plan, which will add three new subway lines and 45 stations over the next decade at a cost of $8.2 billion.",
    "Tech giant announced record quarterly profits of $28.3 billion, driven by strong growth in cloud computing services and advertising revenue, beating analyst expectations by 12%.",
    # Poetry / literary
    "Two roads diverged in a yellow wood, / And sorry I could not travel both / And be one traveler, long I stood / And looked down one as far as I could / To where it bent in the undergrowth;",
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity.",
    "Call me Ishmael. Some years ago — never mind how long precisely — having little money in my pocket and nothing particular to interest me on shore, I thought I would sail about a little.",
    # Instructional
    "To make sourdough bread: 1) Create a starter from flour and water. 2) Feed it daily for 5-7 days. 3) Mix 500g flour, 375g water, 100g starter, 10g salt. 4) Autolyse for 30 minutes.",
    "Setting up a Python virtual environment: First, ensure Python 3.8+ is installed. Run 'python -m venv venv' to create the environment. Activate with 'source venv/bin/activate' on Linux/Mac.",
    "To configure nginx as a reverse proxy: Install nginx, edit /etc/nginx/sites-available/default, add a server block with proxy_pass pointing to your application server.",
    "The recipe requires 2 cups of all-purpose flour, 1 tsp baking powder, 1/2 tsp salt, 3/4 cup sugar, 2 eggs, 1/2 cup butter, and 1 tsp vanilla extract. Preheat the oven to 350°F.",
    # Math / reasoning
    "Theorem: There are infinitely many prime numbers. Proof: Assume for contradiction that there are finitely many primes p₁, p₂, ..., pₙ. Consider N = p₁·p₂·...·pₙ + 1.",
    "The Fibonacci sequence is defined by F(0)=0, F(1)=1, and F(n)=F(n-1)+F(n-2). The golden ratio φ = (1+√5)/2 ≈ 1.618 appears as the limit of consecutive Fibonacci ratios.",
    "Given a 3×3 matrix A = [[1,2,3],[4,5,6],[7,8,9]], compute the determinant using cofactor expansion along the first row: det(A) = 1*det([[5,6],[8,9]]) - 2*det([[4,6],[7,9]]) + 3*det([[4,5],[7,8]])",
]


def build_diverse_prompts(n: int) -> list[str]:
    """Generate n diverse prompts by cycling through seeds with suffix variation."""
    prompts = []
    suffixes = [
        "", " Furthermore,", " In addition,", " However,", " As a result,",
        " This means that", " Consequently,", " For example,", " In contrast,",
        " Similarly,", " On the other hand,", " Notably,", " First,", " Finally,",
        " Despite this,", " Given that", " It is worth noting that",
        " Recent research suggests", " According to experts,", " In practice,",
    ]
    seed_i = 0
    suf_i = 0
    while len(prompts) < n:
        prompt = PROMPT_SEEDS[seed_i % len(PROMPT_SEEDS)] + suffixes[suf_i % len(suffixes)]
        prompts.append(prompt)
        seed_i += 1
        if seed_i % len(PROMPT_SEEDS) == 0:
            suf_i += 1
    return prompts[:n]


# ---------------------------------------------------------------------------
# Baseline logits
# ---------------------------------------------------------------------------
def compute_baseline_logits(model, tokenizer, prompts: list[str]) -> list[torch.Tensor]:
    log.info(f"Computing baseline logits for {len(prompts)} prompts on {DEVICE}...")
    model.eval()
    baseline = []
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=MAX_LENGTH, padding=False).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        # Last position logits (next-token prediction)
        baseline.append(outputs.logits[0, -1].cpu().clone())
        if (i + 1) % 50 == 0:
            log.info(f"  Baseline: {i+1}/{len(prompts)}")
    return baseline


# ---------------------------------------------------------------------------
# Bisimulation distance for one pair
# ---------------------------------------------------------------------------
def kl_div_pair(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(P || Q) where P=baseline, Q=swapped."""
    p = F.softmax(p_logits, dim=-1).double()
    q = F.softmax(q_logits, dim=-1).double()
    # Clamp for numerical stability
    q = q.clamp(min=1e-40)
    kl = (p * (p.log() - q.log())).sum().item()
    return float(kl)


def compute_bisimulation_distance(
    model,
    tokenizer,
    layer_a: int,
    layer_b: int,
    prompts: list[str],
    baseline_logits: list[torch.Tensor],
) -> dict:
    """
    Swap layer_a weights with layer_b's, measure KL divergence from baseline.
    Returns stats dict with mean/max/median/p95 KL for both swap directions.
    """
    results = {}
    model.eval()

    for direction, (src, tgt) in enumerate([(layer_b, layer_a), (layer_a, layer_b)]):
        # Swap: replace layer tgt with weights from layer src
        original_state = {k: v.clone() for k, v in model.transformer.h[tgt].state_dict().items()}
        src_state = model.transformer.h[src].state_dict()
        model.transformer.h[tgt].load_state_dict(src_state)

        kl_vals = []
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=MAX_LENGTH, padding=False).to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)
            swapped_logits = outputs.logits[0, -1].cpu()
            kl = kl_div_pair(baseline_logits[i], swapped_logits)
            kl_vals.append(kl)

        # Restore original weights
        model.transformer.h[tgt].load_state_dict(original_state)

        label = f"dir{direction}"
        results[label] = {
            "mean_kl":   float(np.mean(kl_vals)),
            "max_kl":    float(np.max(kl_vals)),
            "median_kl": float(np.median(kl_vals)),
            "p95_kl":    float(np.percentile(kl_vals, 95)),
            "min_kl":    float(np.min(kl_vals)),
        }

    # Symmetric distance = max KL across both directions and all prompts
    mean_kl  = max(results["dir0"]["mean_kl"],  results["dir1"]["mean_kl"])
    max_kl   = max(results["dir0"]["max_kl"],   results["dir1"]["max_kl"])
    p95_kl   = max(results["dir0"]["p95_kl"],   results["dir1"]["p95_kl"])
    median_kl = max(results["dir0"]["median_kl"], results["dir1"]["median_kl"])

    return {
        "mean_kl":   mean_kl,
        "max_kl":    max_kl,
        "median_kl": median_kl,
        "p95_kl":    p95_kl,
        "dir0":      results["dir0"],
        "dir1":      results["dir1"],
    }


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
def run_experiment(skip_pairs: bool = False):
    log.info("=" * 70)
    log.info(f"Bisimulation Experiment — {MODEL_NAME}")
    log.info(f"Device: {DEVICE}  |  N_PROMPTS: {N_PROMPTS}  |  MAX_LENGTH: {MAX_LENGTH}")
    log.info("=" * 70)

    # Load model
    log.info(f"Loading {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model = model.to(DEVICE)
    model.eval()
    n_layers = model.config.n_layer
    log.info(f"Model loaded in {time.time()-t0:.1f}s. Layers: {n_layers}")

    # Build prompts
    prompts = build_diverse_prompts(N_PROMPTS)
    log.info(f"Built {len(prompts)} diverse prompts.")

    # Baseline
    t1 = time.time()
    baseline_logits = compute_baseline_logits(model, tokenizer, prompts)
    log.info(f"Baseline computed in {time.time()-t1:.1f}s.")

    # Layer pair matrix
    n_pairs = n_layers * (n_layers - 1) // 2
    log.info(f"Computing {n_pairs} layer pairs ({n_layers}×{n_layers-1}/2)...")

    distance_matrix = np.full((n_layers, n_layers), np.nan)
    np.fill_diagonal(distance_matrix, 0.0)

    pair_records = []
    checkpoint_path = f"{REPORT_DIR}/distance_checkpoint.json"

    # Load checkpoint if exists
    completed_pairs = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        for rec in ckpt["pairs"]:
            a, b = rec["layer_a"], rec["layer_b"]
            distance_matrix[a, b] = rec["mean_kl"]
            distance_matrix[b, a] = rec["mean_kl"]
            completed_pairs.add((a, b))
            pair_records.append(rec)
        log.info(f"Resumed from checkpoint: {len(completed_pairs)} pairs already done.")

    # Generate pair order: adjacent first, then by gap
    ordered_pairs = []
    for gap in range(1, n_layers):
        for a in range(n_layers - gap):
            b = a + gap
            if (a, b) not in completed_pairs:
                ordered_pairs.append((a, b))

    t2 = time.time()
    done = len(completed_pairs)

    for pair_idx, (a, b) in enumerate(ordered_pairs):
        t_pair = time.time()
        log.info(f"[{done+1}/{n_pairs}] Layer {a} <-> {b} (gap={b-a})...")

        stats = compute_bisimulation_distance(
            model, tokenizer, a, b, prompts, baseline_logits
        )

        distance_matrix[a, b] = stats["mean_kl"]
        distance_matrix[b, a] = stats["mean_kl"]

        rec = {
            "layer_a": a,
            "layer_b": b,
            "gap": b - a,
            **{k: v for k, v in stats.items() if k not in ("dir0", "dir1")},
            "dir0": stats["dir0"],
            "dir1": stats["dir1"],
            "elapsed_s": time.time() - t_pair,
        }
        pair_records.append(rec)
        done += 1

        log.info(
            f"  mean_KL={stats['mean_kl']:.4f}  max_KL={stats['max_kl']:.4f}  "
            f"p95_KL={stats['p95_kl']:.4f}  [{time.time()-t_pair:.1f}s]"
        )

        # Checkpoint every 10 pairs
        if done % 10 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump({"pairs": pair_records}, f)
            log.info(f"  --> Checkpoint saved ({done}/{n_pairs} pairs).")

        # Early-stop signal: if we've done all adjacent (gap=1) pairs and ALL are > 5.0,
        # the research is killed; still continue for reporting purposes.

    # Final checkpoint
    with open(checkpoint_path, "w") as f:
        json.dump({"pairs": pair_records}, f)

    log.info(f"All pairs done in {(time.time()-t2)/60:.1f} min total.")
    return model, tokenizer, n_layers, distance_matrix, pair_records, baseline_logits, prompts


# ---------------------------------------------------------------------------
# Analysis & Plotting
# ---------------------------------------------------------------------------
def analyze_and_plot(n_layers, distance_matrix, pair_records):
    log.info("Generating analysis artifacts...")

    # ---- CSV: distance matrix ----
    import csv
    matrix_path = f"{REPORT_DIR}/distance_matrix.csv"
    with open(matrix_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + [str(i) for i in range(n_layers)])
        for i in range(n_layers):
            row = [str(i)] + [f"{distance_matrix[i, j]:.6f}" if not np.isnan(distance_matrix[i, j]) else "nan" for j in range(n_layers)]
            w.writerow(row)
    log.info(f"Saved: {matrix_path}")

    # ---- CSV: sorted pairs ----
    sorted_records = sorted(pair_records, key=lambda r: r["mean_kl"])
    sorted_path = f"{REPORT_DIR}/sorted_pairs.csv"
    with open(sorted_path, "w", newline="") as f:
        fieldnames = ["rank", "layer_a", "layer_b", "gap", "mean_kl", "max_kl", "median_kl", "p95_kl"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rank, rec in enumerate(sorted_records, 1):
            w.writerow({
                "rank": rank,
                "layer_a": rec["layer_a"],
                "layer_b": rec["layer_b"],
                "gap": rec["gap"],
                "mean_kl": f"{rec['mean_kl']:.6f}",
                "max_kl": f"{rec['max_kl']:.6f}",
                "median_kl": f"{rec['median_kl']:.6f}",
                "p95_kl": f"{rec['p95_kl']:.6f}",
            })
    log.info(f"Saved: {sorted_path}")

    # ---- Heatmap ----
    fig, ax = plt.subplots(figsize=(10, 8))
    # Fill diagonal with 0 for display
    dm_display = distance_matrix.copy()
    np.fill_diagonal(dm_display, 0.0)
    # Use log scale if range is large
    vmax = np.nanpercentile(dm_display[dm_display > 0], 95)
    im = ax.imshow(dm_display, cmap="viridis_r", aspect="auto",
                   vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Mean KL Divergence (lower = more similar)")
    ax.set_title(f"Layer Bisimulation Distance — {MODEL_NAME}\n(lower = more interchangeable)", fontsize=13)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Layer index")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    heatmap_path = f"{REPORT_DIR}/heatmap.png"
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    log.info(f"Saved: {heatmap_path}")

    # ---- Histogram ----
    mean_kls = [r["mean_kl"] for r in pair_records]
    max_kls  = [r["max_kl"]  for r in pair_records]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(mean_kls, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0].set_title("Distribution of Mean KL (per pair)")
    axes[0].set_xlabel("Mean KL Divergence"); axes[0].set_ylabel("# Pairs")
    axes[0].axvline(0.05, color="green",  linestyle="--", label="0.05 (strong GO)")
    axes[0].axvline(0.10, color="orange", linestyle="--", label="0.10 (cond. GO)")
    axes[0].axvline(0.50, color="red",    linestyle="--", label="0.50 (weak)")
    axes[0].legend(fontsize=8)

    axes[1].hist(max_kls,  bins=40, color="coral",    edgecolor="white", alpha=0.85)
    axes[1].set_title("Distribution of Max KL (per pair)")
    axes[1].set_xlabel("Max KL Divergence"); axes[1].set_ylabel("# Pairs")
    axes[1].axvline(0.05, color="green",  linestyle="--", label="0.05")
    axes[1].axvline(0.10, color="orange", linestyle="--", label="0.10")
    axes[1].legend(fontsize=8)

    hist_path = f"{REPORT_DIR}/histogram.png"
    plt.tight_layout()
    plt.savefig(hist_path, dpi=150)
    plt.close()
    log.info(f"Saved: {hist_path}")

    # ---- Adjacency trend ----
    from collections import defaultdict
    by_gap = defaultdict(list)
    for rec in pair_records:
        by_gap[rec["gap"]].append(rec["mean_kl"])
    gaps = sorted(by_gap.keys())
    gap_means  = [np.mean(by_gap[g])   for g in gaps]
    gap_medians = [np.median(by_gap[g]) for g in gaps]
    gap_mins   = [np.min(by_gap[g])    for g in gaps]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(gaps, gap_means,   "o-", label="Mean KL", color="steelblue")
    ax.plot(gaps, gap_medians, "s--", label="Median KL", color="darkorange")
    ax.plot(gaps, gap_mins,    "^:", label="Min KL (best pair at this gap)", color="green")
    ax.set_title(f"Bisimulation Distance vs Layer Gap — {MODEL_NAME}")
    ax.set_xlabel("|ℓ₁ - ℓ₂| (layer gap)")
    ax.set_ylabel("KL Divergence")
    ax.legend()
    ax.axhline(0.05, color="green",  linestyle="--", alpha=0.4)
    ax.axhline(0.10, color="orange", linestyle="--", alpha=0.4)
    adj_path = f"{REPORT_DIR}/adjacency_trend.png"
    plt.tight_layout()
    plt.savefig(adj_path, dpi=150)
    plt.close()
    log.info(f"Saved: {adj_path}")

    return sorted_records, mean_kls, max_kls, by_gap


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def write_verdict(n_layers, sorted_records, mean_kls, max_kls, by_gap):
    n_pairs = len(sorted_records)
    all_mean = np.array(mean_kls)
    all_max  = np.array(max_kls)

    strong_go   = int((all_mean < 0.05).sum())
    cond_go     = int((all_mean < 0.10).sum())
    weak        = int((all_mean > 0.50).sum())
    kill        = int((all_mean > 2.00).sum())

    top5 = sorted_records[:5]

    # Is adjacency trend monotone?
    gaps = sorted(by_gap.keys())
    gap_means = [np.mean(by_gap[g]) for g in gaps]
    monotone = all(gap_means[i] <= gap_means[i+1] for i in range(len(gap_means)-1))

    # Determine verdict
    if strong_go >= 3:
        verdict_label = "STRONG GO"
        verdict_body  = (
            f"{strong_go} layer pairs have mean_KL < 0.05. Strong evidence of approximate bisimulation. "
            "Multiple layers are approximately interchangeable. Scaling to larger models is strongly warranted."
        )
    elif cond_go >= 1:
        verdict_label = "CONDITIONAL GO"
        best = top5[0]
        verdict_body  = (
            f"{cond_go} layer pair(s) have mean_KL < 0.10. Some approximation bisimulation exists. "
            f"Best pair: layers {best['layer_a']} <-> {best['layer_b']} (gap={best['gap']}, "
            f"mean_KL={best['mean_kl']:.4f}, max_KL={best['max_kl']:.4f}). "
            "Recommend testing on 7-8B models where redundancy likely increases."
        )
    elif weak < n_pairs:  # some pairs < 0.5
        verdict_label = "WEAK"
        verdict_body  = (
            f"No pairs with mean_KL < 0.10. Minimum mean_KL = {all_mean.min():.4f}. "
            "No approximate bisimulation at GPT-2-Medium scale. "
            "Try larger models (7B+) before drawing final conclusions."
        )
    else:
        verdict_label = "KILL"
        verdict_body  = (
            f"All {n_pairs} pairs have mean_KL > 0.50 ({kill} pairs > 2.0). "
            "Every layer is unique. Bisimulation quotient = original model. "
            "The bisimulation compression hypothesis is falsified at this scale."
        )

    # Top-5 table rows
    top5_rows = "\n".join(
        f"| {i+1} | {r['layer_a']} | {r['layer_b']} | {r['gap']} | {r['mean_kl']:.4f} | {r['max_kl']:.4f} | {r['p95_kl']:.4f} |"
        for i, r in enumerate(top5)
    )

    report = f"""# Bisimulation Distance Experiment — Verdict

**Model:** {MODEL_NAME} ({n_layers} layers, {n_pairs} pairs tested)  
**Prompts per pair:** {N_PROMPTS} (diverse: Wikipedia, code, dialogue, news, poetry, instructions)  
**Date:** 2026-03-30  

---

## Verdict: {verdict_label}

{verdict_body}

---

## Key Statistics

| Metric | Value |
|---|---|
| Total pairs | {n_pairs} |
| Pairs with mean_KL < 0.05 (STRONG GO) | {strong_go} |
| Pairs with mean_KL < 0.10 (COND. GO) | {cond_go} |
| Pairs with mean_KL > 0.50 (WEAK) | {weak} |
| Pairs with mean_KL > 2.00 (KILL) | {kill} |
| Global min mean_KL | {all_mean.min():.6f} |
| Global max mean_KL | {all_mean.max():.6f} |
| Global median mean_KL | {np.median(all_mean):.6f} |
| Global min max_KL | {all_max.min():.6f} |
| Adjacency trend monotone? | {'Yes' if monotone else 'No'} |

---

## Top-5 Closest Layer Pairs

| Rank | Layer A | Layer B | Gap | Mean KL | Max KL | p95 KL |
|---|---|---|---|---|---|---|
{top5_rows}

---

## Adjacency Trend Summary

| Gap | Mean KL | Min KL |
|---|---|---|
""" + "\n".join(
        f"| {g} | {np.mean(by_gap[g]):.4f} | {np.min(by_gap[g]):.4f} |"
        for g in sorted(by_gap.keys())
    ) + f"""

---

## Artifacts

- `distance_matrix.csv` — Full 24×24 pairwise KL matrix
- `sorted_pairs.csv` — All {n_pairs} pairs sorted by mean_KL ascending
- `heatmap.png` — Bisimulation distance heatmap
- `histogram.png` — Distribution of mean and max KL across pairs
- `adjacency_trend.png` — KL vs layer gap plot

---

## Interpretation Notes

- **Mean KL is generous.** For true bisimulation, max_KL across all prompts must be small. Low mean but high max indicates average-case similarity but worst-case distinguishability.
- **Two-directional test performed.** Reported distance = max KL across both swap directions (symmetric).
- **GPT-2-Medium ({n_layers} layers)** may show less redundancy than larger models. A WEAK or KILL result here does not rule out bisimulation in 7B+ models.
- **Next step if GO:** Test single-merge compression on WikiText-2 perplexity.
- **Next step if KILL:** Consider finer granularity (head-level, sublayer-level) or directly test on 7B models.
"""

    verdict_path = f"{REPORT_DIR}/verdict.md"
    with open(verdict_path, "w") as f:
        f.write(report)
    log.info(f"Verdict saved: {verdict_path}")
    log.info(f"\n{'='*60}\nVERDICT: {verdict_label}\n{'='*60}")
    log.info(f"Min mean_KL = {all_mean.min():.6f}")
    log.info(f"STRONG GO pairs (mean_KL < 0.05): {strong_go}")
    log.info(f"COND GO pairs (mean_KL < 0.10):   {cond_go}")
    return verdict_label


# ---------------------------------------------------------------------------
# Optional: Single merge perplexity test
# ---------------------------------------------------------------------------
def test_single_merge(model, tokenizer, sorted_records, baseline_logits, prompts, n_layers):
    """Average weights of best pair, evaluate KL degradation vs 23-layer model."""
    best = sorted_records[0]
    a, b = best["layer_a"], best["layer_b"]
    log.info(f"\nTesting single merge: layers {a} <-> {b} (mean_KL={best['mean_kl']:.4f})")

    # Average weights
    state_a = model.transformer.h[a].state_dict()
    state_b = model.transformer.h[b].state_dict()
    merged_state = {k: 0.5 * (state_a[k].float() + state_b[k].float()) for k in state_a}

    # Clone model, apply merged weights to layer a, skip layer b
    # We test: use merged for a, keep b as merged (not original) — i.e., both layers use the same merged state
    model.transformer.h[a].load_state_dict(merged_state)
    model.transformer.h[b].load_state_dict(merged_state)

    kl_vals = []
    for i, prompt in enumerate(prompts[:50]):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=MAX_LENGTH, padding=False).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        merged_logits = outputs.logits[0, -1].cpu()
        kl = kl_div_pair(baseline_logits[i], merged_logits)
        kl_vals.append(kl)

    log.info(f"  Post-merge KL: mean={np.mean(kl_vals):.4f}, max={np.max(kl_vals):.4f}, p95={np.percentile(kl_vals,95):.4f}")

    merge_path = f"{REPORT_DIR}/merge_test.md"
    with open(merge_path, "w") as f:
        f.write(f"# Single Merge Test\n\n")
        f.write(f"- Best pair: Layer {a} <-> Layer {b} (gap={best['gap']})\n")
        f.write(f"- Pre-merge mean_KL: {best['mean_kl']:.6f}\n")
        f.write(f"- Post-merge mean_KL (both layers merged): {np.mean(kl_vals):.6f}\n")
        f.write(f"- Post-merge max_KL: {np.max(kl_vals):.6f}\n")
        f.write(f"- Post-merge p95_KL: {np.percentile(kl_vals,95):.6f}\n")
    log.info(f"Merge test saved: {merge_path}")

    # Restore
    model.transformer.h[a].load_state_dict({k: state_a[k] for k in state_a})
    model.transformer.h[b].load_state_dict({k: state_b[k] for k in state_b})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-prompts", type=int, default=N_PROMPTS)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--no-merge-test", action="store_true")
    args = parser.parse_args()
    N_PROMPTS  = args.n_prompts
    MAX_LENGTH = args.max_length

    model, tokenizer, n_layers, dist_mat, pair_records, baseline_logits, prompts = run_experiment()
    sorted_records, mean_kls, max_kls, by_gap = analyze_and_plot(n_layers, dist_mat, pair_records)
    verdict = write_verdict(n_layers, sorted_records, mean_kls, max_kls, by_gap)

    if verdict in ("STRONG GO", "CONDITIONAL GO") and not args.no_merge_test:
        test_single_merge(model, tokenizer, sorted_records, baseline_logits, prompts, n_layers)

    log.info("Experiment complete.")
