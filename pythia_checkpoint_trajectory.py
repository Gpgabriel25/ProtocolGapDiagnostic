#!/usr/bin/env python3
"""
Pythia Training Checkpoint Trajectory
======================================
Tracks layer redundancy across training checkpoints to reveal the
*mechanism* by which bisimulation-identifiable redundant layers emerge.

For each checkpoint of Pythia-410M (154 available from EleutherAI):
  1. Loads the model at that training step
  2. Computes per-layer skip-KL (replacement distance)
  3. Computes adjacent-pair swap-KL (interchange distance)
  4. Records the training loss estimate on calibration data

Output: JSON with per-checkpoint per-layer metrics + heatmap figure.

Key hypothesis: redundant layers emerge *during* training (not from
architecture alone), and the pattern stabilizes by mid-training.

Usage:
  # Smoke test (3 checkpoints):
  python pythia_checkpoint_trajectory.py --steps 0,1000,143000 --n-prompts 8

  # Full trajectory (20 checkpoints):
  python pythia_checkpoint_trajectory.py --n-prompts 16

  # Extended (all interesting checkpoints):
  python pythia_checkpoint_trajectory.py --steps all --n-prompts 16
"""

import os
import sys
import time
import json
import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

MODEL_NAME = "EleutherAI/pythia-410m"

# Default 20 checkpoints spanning 6 orders of magnitude
DEFAULT_STEPS = [
    0, 1, 2, 4, 8, 16, 64, 256, 512, 1000,
    2000, 4000, 8000, 16000, 32000, 64000,
    100000, 120000, 140000, 143000,
]

# Diverse calibration prompts
PROMPTS = [
    "The French Revolution was a period of radical political and societal change in France that began",
    "Quantum mechanics is a fundamental theory in physics that provides a description of the physical",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) +",
    "The Amazon rainforest, also known as Amazonia, is a moist broadleaf tropical rainforest in the",
    "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity,",
    "DNA, or deoxyribonucleic acid, is a molecule composed of two polynucleotide chains that coil",
    "The stock market experienced significant volatility today as investors reacted to new data showing",
    "Once upon a time in a land far away, there lived a young princess who dreamed of exploring",
    "import numpy as np\nimport torch\n\nclass NeuralNetwork(torch.nn.Module):\n    def __init__(self",
    "Climate change poses one of the greatest challenges facing humanity in the twenty-first century,",
    "SELECT users.name, COUNT(orders.id) as order_count FROM users INNER JOIN orders ON users.id =",
    "The mitochondria is the powerhouse of the cell, responsible for producing adenosine triphosphate",
    "In the year 2050, artificial intelligence had transformed every aspect of human civilization in",
    "Dear Sir or Madam, I am writing to express my concern regarding the recent changes to the",
    "The Beatles, formed in Liverpool in 1960, became the most commercially successful and critically",
    "Two roads diverged in a yellow wood, and sorry I could not travel both and be one traveler,",
]

MAX_LENGTH = 64


def get_layers(model):
    """Return the list of transformer blocks."""
    if hasattr(model, 'gpt_neox'):
        return model.gpt_neox.layers
    raise ValueError(f"Unknown architecture: {type(model)}")


def compute_baseline(model, tokenizer, prompts, device):
    """Compute baseline logits for all prompts."""
    baseline_logits = []
    total_loss = 0.0
    n_tokens = 0
    for prompt in prompts:
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        baseline_logits.append(outputs.logits.detach().cpu())
        total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
        n_tokens += inputs["input_ids"].shape[1]
    avg_loss = total_loss / max(n_tokens, 1)
    return baseline_logits, avg_loss


def compute_skip_kl(model, tokenizer, prompts, baseline_logits, layer_idx, device):
    """Skip layer_idx (identity bypass) and measure KL divergence from baseline."""
    layers = get_layers(model)

    # Register a hook that replaces the layer output with input (skip)
    def skip_hook(module, inp, out):
        hidden = inp[0]
        if isinstance(out, tuple):
            return (hidden,) + out[1:]
        return hidden

    handle = layers[layer_idx].register_forward_hook(skip_hook)

    kl_vals = []
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
        ).to(device)
        with torch.no_grad():
            skip_logits = model(**inputs).logits.detach().cpu()

        # KL(baseline || skip) per token, averaged
        log_p = F.log_softmax(baseline_logits[i], dim=-1)
        log_q = F.log_softmax(skip_logits, dim=-1)
        p = log_p.exp()
        # KL = sum(p * (log_p - log_q))
        kl = (p * (log_p - log_q)).sum(dim=-1).mean().item()
        kl_vals.append(max(kl, 0.0))  # clamp numerical noise

    handle.remove()
    return float(np.mean(kl_vals))


def compute_swap_kl(model, tokenizer, prompts, baseline_logits, layer_a, layer_b, device):
    """Swap layer_a output with layer_b output and measure KL divergence."""
    layers = get_layers(model)
    captured = {}

    # First pass: capture layer_b output
    def capture_hook(module, inp, out):
        if isinstance(out, tuple):
            captured['hidden'] = out[0].clone()
        else:
            captured['hidden'] = out.clone()

    # Second pass: replace layer_a output with captured layer_b output
    def swap_hook(module, inp, out):
        if isinstance(out, tuple):
            return (captured['hidden'],) + out[1:]
        return captured['hidden']

    kl_vals = []
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
        ).to(device)

        # Pass 1: capture layer_b output
        h_capture = layers[layer_b].register_forward_hook(capture_hook)
        with torch.no_grad():
            model(**inputs)
        h_capture.remove()

        # Pass 2: swap layer_a output with captured layer_b output
        h_swap = layers[layer_a].register_forward_hook(swap_hook)
        with torch.no_grad():
            swap_logits = model(**inputs).logits.detach().cpu()
        h_swap.remove()

        log_p = F.log_softmax(baseline_logits[i], dim=-1)
        log_q = F.log_softmax(swap_logits, dim=-1)
        p = log_p.exp()
        kl = (p * (log_p - log_q)).sum(dim=-1).mean().item()
        kl_vals.append(max(kl, 0.0))

    return float(np.mean(kl_vals))


def run_checkpoint(step, tokenizer, prompts, device, dtype, skip_swap=False):
    """Load model at given step and compute all metrics."""
    log.info(f"Loading checkpoint step={step}...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=f"step{step}", dtype=dtype
    ).to(device)
    model.eval()
    load_time = time.time() - t0
    log.info(f"  Loaded in {load_time:.1f}s")

    n_layers = len(get_layers(model))

    # Baseline
    t0 = time.time()
    baseline_logits, avg_loss = compute_baseline(model, tokenizer, prompts, device)
    baseline_time = time.time() - t0

    # Skip-KL for each layer
    skip_kls = []
    t0 = time.time()
    for li in range(n_layers):
        kl = compute_skip_kl(model, tokenizer, prompts, baseline_logits, li, device)
        skip_kls.append(kl)
    skip_time = time.time() - t0

    # Adjacent swap-KL
    swap_kls = []
    if not skip_swap:
        t0 = time.time()
        for li in range(n_layers - 1):
            kl = compute_swap_kl(
                model, tokenizer, prompts, baseline_logits, li, li + 1, device
            )
            swap_kls.append(kl)
        swap_time = time.time() - t0
    else:
        swap_time = 0.0

    total_time = load_time + baseline_time + skip_time + swap_time
    log.info(
        f"  step={step}: loss={avg_loss:.4f}, "
        f"skip_kl_range=[{min(skip_kls):.4f}, {max(skip_kls):.4f}], "
        f"time={total_time:.1f}s"
    )

    # Free memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "step": step,
        "avg_loss": avg_loss,
        "skip_kl": skip_kls,
        "swap_kl": swap_kls,
        "n_layers": n_layers,
        "load_time_s": round(load_time, 1),
        "compute_time_s": round(baseline_time + skip_time + swap_time, 1),
    }


def generate_figure(results, output_path):
    """Generate heatmap of layer importance across training."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    steps = [r["step"] for r in results]
    n_layers = results[0]["n_layers"]

    # Build skip-KL matrix: checkpoints × layers
    skip_matrix = np.array([r["skip_kl"] for r in results])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 3]})

    # Top panel: training loss curve
    ax_loss = axes[0]
    losses = [r["avg_loss"] for r in results]
    ax_loss.plot(steps, losses, "ko-", markersize=4)
    ax_loss.set_xscale("symlog", linthresh=10)
    ax_loss.set_ylabel("Calibration Loss")
    ax_loss.set_title("Pythia-410M: Layer Redundancy Emergence During Training")
    ax_loss.grid(True, alpha=0.3)

    # Bottom panel: skip-KL heatmap
    ax_heat = axes[1]
    # Use log scale for better visibility
    vmin = max(skip_matrix[skip_matrix > 0].min(), 1e-4) if (skip_matrix > 0).any() else 1e-4
    vmax = skip_matrix.max()
    im = ax_heat.imshow(
        skip_matrix.T,
        aspect="auto",
        cmap="viridis_r",  # dark = redundant (low KL), bright = important (high KL)
        norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10)),
        interpolation="nearest",
    )
    ax_heat.set_yticks(range(n_layers))
    ax_heat.set_yticklabels(range(n_layers), fontsize=7)
    ax_heat.set_xticks(range(len(steps)))
    ax_heat.set_xticklabels([str(s) for s in steps], rotation=45, ha="right", fontsize=7)
    ax_heat.set_ylabel("Layer Index")
    ax_heat.set_xlabel("Training Step")
    cbar = plt.colorbar(im, ax=ax_heat, shrink=0.8)
    cbar.set_label("Skip-KL (lower = more redundant)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Figure saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Pythia checkpoint trajectory")
    parser.add_argument(
        "--steps", type=str, default=None,
        help="Comma-separated checkpoint steps, or 'all' for full 20-point sweep"
    )
    parser.add_argument("--n-prompts", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"])
    parser.add_argument("--output", type=str, default="pythia_checkpoint_trajectory.json")
    parser.add_argument("--skip-swap", action="store_true", help="Skip swap-KL computation")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.steps is None or args.steps == "all":
        steps = DEFAULT_STEPS
    else:
        steps = [int(s.strip()) for s in args.steps.split(",")]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    prompts = PROMPTS[: args.n_prompts]

    log.info(f"Model: {MODEL_NAME}")
    log.info(f"Device: {device}, dtype: {args.dtype}")
    log.info(f"Checkpoints: {len(steps)} steps")
    log.info(f"Prompts: {len(prompts)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    results = []
    t_total = time.time()
    for i, step in enumerate(steps):
        log.info(f"[{i+1}/{len(steps)}] Processing step {step}")
        r = run_checkpoint(step, tokenizer, prompts, device, dtype, skip_swap=args.skip_swap)
        results.append(r)

        # Incremental save
        with open(args.output, "w") as f:
            json.dump(
                {
                    "model": MODEL_NAME,
                    "n_prompts": len(prompts),
                    "dtype": args.dtype,
                    "device": device,
                    "checkpoints": results,
                    "completed": len(results),
                    "total": len(steps),
                },
                f,
                indent=2,
            )

    elapsed = time.time() - t_total
    log.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Generate figure
    fig_path = args.output.replace(".json", ".png")
    generate_figure(results, fig_path)

    # Summary statistics
    final = results[-1]
    skip_kls = np.array(final["skip_kl"])
    redundant_layers = np.where(skip_kls < np.median(skip_kls) * 0.1)[0].tolist()
    log.info(f"Final checkpoint redundant layers (skip-KL < 10% median): {redundant_layers}")

    # Phase transition detection: when does the redundancy pattern stabilize?
    if len(results) > 2:
        profiles = np.array([r["skip_kl"] for r in results])
        # Normalize each profile
        norms = np.linalg.norm(profiles, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        profiles_norm = profiles / norms
        # Cosine similarity with final profile
        final_norm = profiles_norm[-1]
        cos_sims = [float(np.dot(profiles_norm[i], final_norm)) for i in range(len(results))]
        stabilization_idx = next(
            (i for i in range(len(cos_sims)) if cos_sims[i] > 0.95), len(cos_sims) - 1
        )
        log.info(
            f"Pattern stabilizes at step {results[stabilization_idx]['step']} "
            f"(cosine sim > 0.95 with final)"
        )

    log.info("Done.")


if __name__ == "__main__":
    main()
