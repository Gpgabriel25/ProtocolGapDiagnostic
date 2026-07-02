#!/usr/bin/env python3
"""Generate paper figures from experimental results."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

# Global readability bump (per visual-audit feedback 2026-04-21):
# enlarge tick labels, axis labels, titles, legends, and colorbar text
# so figures remain legible at print scale and on small screens.
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.titlesize": 15,
})

CYCLE = "2026-03-31T00-18-24"
REPORTS = f"reports/{CYCLE}"
FIGS = f"paper/figures"
STANDARDIZED_CYCLE = "2026-04-01T18-45-48"
STANDARDIZED_REPORTS = f"reports/{STANDARDIZED_CYCLE}"
os.makedirs(FIGS, exist_ok=True)

# ---------- Figure 1: Scaling plot (best KL vs layers, by architecture) ----------
def fig_scaling():
    # Data points: (model, arch, params_M, layers, best_kl)
    data = [
        ("GPT-2\nSmall",  "GPT-2 (Abs PE)", 124,  12, 0.099),
        ("GPT-2\nMedium", "GPT-2 (Abs PE)", 355,  24, 0.035),
        # GPT-2-Large will be added dynamically if available
        ("Pythia\n160M",  "Pythia (RoPE)",   162,  12, 0.778),
        ("Pythia\n410M",  "Pythia (RoPE)",   405,  24, 0.137),
        ("Pythia\n1.4B",  "Pythia (RoPE)",  1400,  24, 0.096),
        ("Pythia\n2.8B",  "Pythia (RoPE)",  2900,  32, 0.055),
        ("Qwen3\n8B",     "Qwen (RoPE+GQA)", 8200, 36, 0.056),
    ]

    # Try to load GPT-2-large results
    gpt2l_path = os.path.join(REPORTS, "gpt2-large_bisimulation.json")
    if os.path.exists(gpt2l_path):
        with open(gpt2l_path) as f:
            d = json.load(f)
        best = min(d["pairs"], key=lambda p: p["mean_kl"])
        data.insert(2, ("GPT-2\nLarge", "GPT-2 (Abs PE)", 774, 36, best["mean_kl"]))

    # Try to load GPT-2-XL results
    gpt2xl_path = os.path.join(REPORTS, "gpt2-xl_bisimulation.json")
    if os.path.exists(gpt2xl_path):
        with open(gpt2xl_path) as f:
            d = json.load(f)
        best = min(d["pairs"], key=lambda p: p["mean_kl"])
        # Insert after GPT-2-Large
        idx = next((i for i, x in enumerate(data) if "Large" in x[0]), len(data))
        data.insert(idx + 1, ("GPT-2\nXL", "GPT-2 (Abs PE)", 1558, 48, best["mean_kl"]))

    # Try to load OPT-350M results
    opt_path = os.path.join(REPORTS, "opt-350m_bisimulation.json")
    if os.path.exists(opt_path):
        with open(opt_path) as f:
            d = json.load(f)
        best = min(d["pairs"], key=lambda p: p["mean_kl"])
        data.append(("OPT\n350M", "OPT (Abs PE)", 331, 24, best["mean_kl"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Plot 1: Best KL vs Layers
    for arch in ["GPT-2 (Abs PE)", "Pythia (RoPE)", "OPT (Abs PE)", "Qwen (RoPE+GQA)"]:
        pts = [(d[3], d[4]) for d in data if d[1] == arch]
        if not pts:
            continue
        xs, ys = zip(*pts)
        marker = "s" if "GPT" in arch else ("D" if "OPT" in arch else ("^" if "Qwen" in arch else "o"))
        ax1.plot(xs, ys, marker=marker, label=arch, linewidth=2, markersize=8)

    ax1.set_xlabel("Number of Layers", fontsize=12)
    ax1.set_ylabel("Best empirical swap-KL (KL)", fontsize=12)
    ax1.set_title("a) Swap-KL vs. depth", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_yscale("log")
    ax1.axhline(y=0.05, color="green", linestyle="--", alpha=0.5, label="Strong GO")
    ax1.axhline(y=0.10, color="orange", linestyle="--", alpha=0.5, label="Cond GO")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Best KL vs Parameters
    for arch in ["GPT-2 (Abs PE)", "Pythia (RoPE)", "OPT (Abs PE)", "Qwen (RoPE+GQA)"]:
        pts = [(d[2], d[4]) for d in data if d[1] == arch]
        if not pts:
            continue
        xs, ys = zip(*pts)
        marker = "s" if "GPT" in arch else ("D" if "OPT" in arch else ("^" if "Qwen" in arch else "o"))
        ax2.plot(xs, ys, marker=marker, label=arch, linewidth=2, markersize=8)

    ax2.set_xlabel("Parameters (M)", fontsize=12)
    ax2.set_ylabel("Best empirical swap-KL (KL)", fontsize=12)
    ax2.set_title("b) Swap-KL vs. parameters", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.axhline(y=0.05, color="green", linestyle="--", alpha=0.5)
    ax2.axhline(y=0.10, color="orange", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGS, "scaling.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# ---------- Figure 2: GPT-2-Medium heatmap ----------
def fig_heatmap():
    # Load from original experiment
    csv_path = "reports/2026-04-02T09-57-25/v14_distance_matrix_100p.csv"
    if not os.path.exists(csv_path):
        print(f"Skipping heatmap — {csv_path} not found")
        return

    # Build 24x24 matrix from pairwise CSV
    import csv as csv_mod
    n_layers = 24
    matrix = np.full((n_layers, n_layers), np.nan)
    with open(csv_path) as f:
        for row in csv_mod.DictReader(f):
            i, j = int(row['layer_a']), int(row['layer_b'])
            v = float(row['mean_kl'])
            matrix[i, j] = v
            matrix[j, i] = v
    np.fill_diagonal(matrix, 0)
    n = n_layers

    from matplotlib.colors import BoundaryNorm
    bounds = [0, 0.05, 0.10, 0.20, 0.50]
    cmap = plt.get_cmap("YlGnBu", len(bounds) - 1)
    norm = BoundaryNorm(bounds, ncolors=cmap.N)

    fig, ax = plt.subplots(1, 1, figsize=(9.2, 7.8))
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="equal")
    ax.set_xlabel("Layer j", fontsize=15, labelpad=10)
    ax.set_ylabel("Layer i", fontsize=15, labelpad=10)
    ax.set_title("GPT-2-Medium empirical swap-KL matrix", fontsize=16, pad=14)
    ax.set_xticks(range(0, n, 2))
    ax.set_yticks(range(0, n, 2))
    ax.tick_params(axis="both", labelsize=13, pad=4)
    cbar = fig.colorbar(im, ax=ax, shrink=0.90, pad=0.08, fraction=0.05, aspect=30, ticks=bounds)
    cbar.set_label("Swap-KL (KL)", fontsize=14, labelpad=10)
    cbar.set_ticklabels([str(b) for b in bounds])
    cbar.ax.tick_params(labelsize=12, pad=3)
    fig.subplots_adjust(left=0.12, right=0.86, bottom=0.11, top=0.90)
    path = os.path.join(FIGS, "heatmap.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# ---------- Figure 3: Fine-tune recovery ----------
def fig_finetune():
    steps = [0, 100, 200, 300, 400, 500]
    ppls = [22.63, 19.13, 18.67, 18.51, 18.28, 18.47]
    baseline = 19.19

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(steps, ppls, "b-o", linewidth=2, markersize=8, label="23-layer (skip 5)")
    ax.axhline(y=baseline, color="r", linestyle="--", linewidth=1.5, label="24-layer baseline")
    ax.fill_between(steps, baseline, ppls, where=[p < baseline for p in ppls],
                     alpha=0.15, color="green", label="Surpasses baseline")
    ax.set_xlabel("Fine-tuning Steps", fontsize=12)
    ax.set_ylabel("WikiText-2 Perplexity (↓)", fontsize=12)
    ax.set_title("Skip-Layer Recovery via Fine-tuning", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(17.5, 23.5)
    plt.tight_layout()
    path = os.path.join(FIGS, "finetune_recovery.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# ---------- Figure 4: Per-layer adjacent-pair swap-KL profile (GPT-2-Medium) ----------
def fig_layer_profile():
    csv_path = "reports/2026-03-30T15-15-07/sorted_pairs.csv"
    if not os.path.exists(csv_path):
        print(f"Skipping layer profile — {csv_path} not found")
        return

    import csv
    pairs = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["gap"]) == 1:
                pairs.append((int(row["layer_a"]), float(row["mean_kl"])))
    pairs.sort()

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))
    ax.bar(xs, ys, color=["green" if y < 0.05 else "orange" if y < 0.10 else "red" for y in ys],
           alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Layer i (pair i↔i+1)", fontsize=12)
    ax.set_ylabel("Empirical swap-KL (KL)", fontsize=12)
    ax.set_title("GPT-2-Medium: adjacent-pair swap-KL profile", fontsize=13)
    ax.axhline(y=0.05, color="green", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(y=0.10, color="orange", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xticks(range(0, 24))
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    path = os.path.join(FIGS, "layer_profile.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# ---------- Figure 5: Standardized GPT-2-Medium compression sweep ----------
def fig_compression_sweep():
    json_path = os.path.join(STANDARDIZED_REPORTS, "compression_sweep.json")
    if not os.path.exists(json_path):
        print(f"Skipping compression sweep figure — {json_path} not found")
        return

    with open(json_path) as f:
        data = json.load(f)

    budgets = sorted(int(k) for k in data["guided"].keys())
    baseline = data["baseline_ppl"]
    guided_ppl = [data["guided"][str(k)]["ppl"] for k in budgets]
    random_mean_ppl = [data["random"][str(k)]["mean_ppl"] for k in budgets]
    random_std_ppl = [data["random"][str(k)]["std_ppl"] for k in budgets]
    anti_ppl = [data["anti_guided"][str(k)]["ppl"] for k in budgets]

    guided_delta = [data["guided"][str(k)]["delta_pct"] for k in budgets]
    random_delta = [data["random"][str(k)]["mean_delta_pct"] for k in budgets]
    random_delta_std = []
    for k in budgets:
        trial_deltas = [trial["delta_pct"] for trial in data["random"][str(k)]["trials"]]
        random_delta_std.append(float(np.std(trial_deltas)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    ax1.plot(budgets, guided_ppl, "o-", linewidth=2, markersize=7, label="Interchange-guided")
    ax1.plot(budgets, random_mean_ppl, "s-", linewidth=2, markersize=6, label="Random mean")
    ax1.fill_between(
        budgets,
        np.array(random_mean_ppl) - np.array(random_std_ppl),
        np.array(random_mean_ppl) + np.array(random_std_ppl),
        alpha=0.2,
        label="Random ±1 std",
    )
    ax1.plot(budgets, anti_ppl, "^-", linewidth=1.8, markersize=6, label="Anti-guided")
    ax1.axhline(y=baseline, color="black", linestyle="--", linewidth=1.2, label="Baseline")
    ax1.set_yscale("log")
    ax1.set_xlabel("Layers Removed", fontsize=12)
    ax1.set_ylabel("WikiText-2 Perplexity (log scale)", fontsize=12)
    ax1.set_title("a) Standardized Compression Sweep", fontsize=13)
    ax1.set_xticks(budgets)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.plot(budgets, guided_delta, "o-", linewidth=2, markersize=7, label="Interchange-guided")
    ax2.plot(budgets, random_delta, "s-", linewidth=2, markersize=6, label="Random mean")
    ax2.fill_between(
        budgets,
        np.array(random_delta) - np.array(random_delta_std),
        np.array(random_delta) + np.array(random_delta_std),
        alpha=0.2,
        label="Random ±1 std",
    )
    ax2.set_xlabel("Layers Removed", fontsize=12)
    ax2.set_ylabel("Relative PPL Increase (%)", fontsize=12)
    ax2.set_title("b) Guided vs. Random", fontsize=13)
    ax2.set_xticks(budgets)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGS, "compression_sweep.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# ---------- Figure 6: Qwen3-8B compute-vs-quality frontier ----------
def fig_qwen_compute_quality_frontier():
    """Plot strict fair-cost timing against matched-evaluator quality for key Qwen methods."""
    ci_path = "qwen3_8b_ci.json"
    fair_cost_path = "reports/2026-04-12T12-10-20/fair_cost_final_summary.json"
    if not os.path.exists(ci_path) or not os.path.exists(fair_cost_path):
        print(f"Skipping Qwen frontier — missing {ci_path} or {fair_cost_path}")
        return

    with open(ci_path) as f:
        ci = json.load(f)
    with open(fair_cost_path) as f:
        fair = json.load(f)

    baseline_ppl = ci["baseline"]["ppl"]
    t_sleb = float(fair["8k"]["sleb_s"])
    t_inter = float(fair["8k"]["inter_s"])

    # Replacement uses the same pairwise swap-KL diagnostic sweep as interchange.
    method_specs = [
        {
            "name": "Interchange-guided",
            "time_s": t_inter,
            "color": "#1b9e77",
            "marker": "o",
            "jitter": -5.0,
            "keys": {
                1: "interchange_n1",
                3: "interchange_clustered_n3",
                5: "interchange_clustered_n5",
            },
        },
        {
            "name": "SLEB-iterative",
            "time_s": t_sleb,
            "color": "#d95f02",
            "marker": "s",
            "jitter": 0.0,
            "keys": {
                1: "sleb_n1",
                3: "sleb_n3",
                5: "sleb_n5",
            },
        },
        {
            "name": "Replacement-guided",
            "time_s": t_inter,
            "color": "#7570b3",
            "marker": "^",
            "jitter": 5.0,
            "keys": {
                1: "replacement_n1",
                3: "replacement_n3",
                5: "replacement_n5",
            },
        },
    ]

    n_offsets = {1: -10.0, 3: 0.0, 5: 10.0}

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.6))

    for spec in method_specs:
        for n_removed, ci_key in spec["keys"].items():
            row = ci[ci_key]
            x = spec["time_s"] + n_offsets[n_removed] + spec["jitter"]
            y = float(row["delta_ppl_pct"])

            low = ((float(row["ci_95_lower"]) / baseline_ppl) - 1.0) * 100.0
            high = ((float(row["ci_95_upper"]) / baseline_ppl) - 1.0) * 100.0
            yerr = np.array([[y - low], [high - y]])

            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt=spec["marker"],
                markersize=7,
                color=spec["color"],
                ecolor=spec["color"],
                elinewidth=1.2,
                capsize=3,
                alpha=0.95,
            )
            ax.annotate(
                f"n={n_removed}",
                (x, y),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                color=spec["color"],
            )

    # Method-only legend.
    method_handles = [
        plt.Line2D([], [], color=s["color"], marker=s["marker"], linestyle="", label=s["name"])
        for s in method_specs
    ]

    ax.legend(handles=method_handles, loc="upper left", fontsize=9, frameon=True)

    ratio_8k = float(fair["8k"]["ratio"])
    ratio_64k = float(fair["64k"]["ratio"])
    ax.text(
        0.02,
        0.02,
        f"Strict fair-cost timing: inter/sleb={ratio_8k:.4f}x (8K), {ratio_64k:.4f}x (64K)",
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    ax.set_xlabel("Wall-clock Scoring Time (seconds, 8K-token strict fair run)", fontsize=14)
    ax.set_ylabel("Matched-Evaluator Degradation (Delta PPL %, 5K words)", fontsize=14)
    ax.set_title("Qwen3-8B Compute-vs-Quality Frontier", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = os.path.join(FIGS, "qwen_compute_quality_frontier.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    fig_scaling()
    fig_heatmap()
    fig_finetune()
    fig_layer_profile()
    fig_compression_sweep()
    fig_qwen_compute_quality_frontier()
    print("Done generating figures.")
