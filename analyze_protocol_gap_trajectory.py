#!/usr/bin/env python3
"""
Analyze protocol-gap trajectory results.

Reads protocol_gap_trajectory.json and emits:
  - reports/<cycle>/protocol_gap_summary.md (table + Pearson scaling tests)
  - figures/protocol_gap_trajectory.{png,pdf} (gap vs step, one line per model size)
  - figures/protocol_gap_scale_sweep.{png,pdf} (gap at final step vs model params)

Usage:
  python3 analyze_protocol_gap_trajectory.py \\
    --in reports/2026-04-18T21-51-24/protocol_gap_trajectory.json \\
    --md-out reports/2026-04-18T21-51-24/protocol_gap_summary.md \\
    --fig-dir figures/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Approximate parameter counts (in millions) for x-axis scaling
PARAM_COUNTS = {
    "EleutherAI/pythia-160m": 160,
    "EleutherAI/pythia-410m": 410,
    "EleutherAI/pythia-1.4b": 1400,
    "EleutherAI/pythia-2.8b": 2800,
    "EleutherAI/pythia-6.9b": 6900,
}


def step_int(checkpoint: str) -> int:
    return int(checkpoint.replace("step", ""))


def _finite(x: object) -> bool:
    return isinstance(x, (int, float)) and not (math.isnan(x) or math.isinf(x))


def pair_gap_values(r: dict) -> List[float]:
    """Per-pair (replacement - interchange) KL gaps; skip non-finite entries."""
    out: List[float] = []
    for p in r["per_pair"]:
        a, b = p["interchange_kl"], p["replacement_kl"]
        if not (_finite(a) and _finite(b)):
            continue
        out.append(float(b) - float(a))
    return out


def mean_gap_from_pairs(r: dict) -> float:
    v = pair_gap_values(r)
    if not v:
        return float("nan")
    return float(sum(v) / len(v))


def mean_inter_from_pairs(r: dict) -> float:
    xs: List[float] = []
    for p in r["per_pair"]:
        a = p["interchange_kl"]
        if _finite(a):
            xs.append(float(a))
    return float(sum(xs) / len(xs)) if xs else float("nan")


def mean_repl_from_pairs(r: dict) -> float:
    xs: List[float] = []
    for p in r["per_pair"]:
        b = p["replacement_kl"]
        if _finite(b):
            xs.append(float(b))
    return float(sum(xs) / len(xs)) if xs else float("nan")


def group_by_model(results: Sequence[dict]) -> Dict[str, List[dict]]:
    by_model: Dict[str, List[dict]] = {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r)
    return by_model


def model_checkpoint_means(rows: List[dict]) -> List[float]:
    rows_sorted = sorted(rows, key=lambda r: step_int(r["checkpoint"]))
    return [mean_gap_from_pairs(r) for r in rows_sorted]


def is_degenerate_trajectory(rows: List[dict]) -> bool:
    """True if mean gap is (numerically) identical at every logged checkpoint."""
    means = model_checkpoint_means(rows)
    if len(means) < 2:
        return True
    rounded = [round(m, 6) for m in means]
    return len(set(rounded)) <= 1


def filter_results_for_figures(
    results: List[dict],
    only_models: Sequence[str] | None,
) -> Tuple[List[dict], List[str]]:
    """
    Drop models with unusable trajectories for plotting.

    Returns (filtered_results, excluded_model_ids).
    """
    by_model = group_by_model(results)
    excluded: List[str] = []
    candidates = list(only_models) if only_models is not None else sorted(by_model.keys())
    keep: set[str] = set()
    for model in candidates:
        rows = by_model.get(model)
        if not rows:
            excluded.append(model)
            continue
        if is_degenerate_trajectory(rows):
            excluded.append(model)
            continue
        keep.add(model)
    filtered = [r for r in results if r["model"] in keep]
    return filtered, excluded



def make_summary_md(results: List[dict], figure_excluded: Sequence[str] | None = None) -> str:
    lines = ["# Protocol Gap Trajectory — Mechanism Experiment", ""]
    lines.append("Tests whether the gap between **replacement** (bisim) and **interchange** "
                 "protocols emerges with **scale × training duration** on Pythia models.")
    lines.append("")
    lines.append("Higher gap = the two protocols disagree more = layers are interchangeable "
                 "but not replaceable. This is the paper's central observation in modern transformers.")
    lines.append("")
    if figure_excluded:
        lines.append(f"_Models excluded from figure export (missing checkpoints or degenerate "
                     f"duplicate trajectory in this JSON): {', '.join(sorted(figure_excluded))}._")
        lines.append("")

    # Group by model
    by_model = group_by_model(results)

    for model in sorted(by_model.keys(), key=lambda m: PARAM_COUNTS.get(m, 0)):
        rows = sorted(by_model[model], key=lambda r: step_int(r["checkpoint"]))
        lines.append(f"## {model} ({PARAM_COUNTS.get(model, '?')}M params)")
        lines.append("")
        lines.append("| step | n_layers | n_pairs | inter_kl | repl_kl | gap_kl | pearson_r | wall_s |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            step = step_int(r["checkpoint"])
            r_val = r.get("pearson_r")
            r_str = f"{r_val:.3f}" if r_val is not None else "n/a"
            g_json = r.get("gap_kl")
            g_show = float(g_json) if isinstance(g_json, (int, float)) and _finite(float(g_json)) else mean_gap_from_pairs(r)
            mi = r.get("mean_interchange_kl")
            mr = r.get("mean_replacement_kl")
            mi_show = float(mi) if isinstance(mi, (int, float)) and _finite(float(mi)) else mean_inter_from_pairs(r)
            mr_show = float(mr) if isinstance(mr, (int, float)) and _finite(float(mr)) else mean_repl_from_pairs(r)
            mi_s = f"{mi_show:.4f}" if _finite(mi_show) else "nan"
            mr_s = f"{mr_show:.4f}" if _finite(mr_show) else "nan"
            g_s = f"{g_show:+.4f}" if _finite(g_show) else "nan"
            lines.append(f"| {step:>6} | {r['n_layers']} | {r['n_pairs']} | "
                         f"{mi_s} | {mr_s} | "
                         f"**{g_s}** | {r_str} | {r['wall_time_s']:.1f} |")
        # Trajectory growth
        if len(rows) >= 2:
            gap_init = mean_gap_from_pairs(rows[0])
            gap_final = mean_gap_from_pairs(rows[-1])
            growth_x = gap_final / gap_init if abs(gap_init) > 1e-6 else float("inf")
            lines.append("")
            lines.append(f"**Gap growth (init→final, pair-recomputed mean): {gap_init:+.4f} → {gap_final:+.4f}** "
                         f"({growth_x:.1f}× | Δ={gap_final - gap_init:+.4f})")
        lines.append("")

    # Cross-model scale comparison at final step
    final_rows = []
    for model, rows in by_model.items():
        final = max(rows, key=lambda r: step_int(r["checkpoint"]))
        final_rows.append((PARAM_COUNTS.get(model, 0), model, final))
    final_rows.sort()

    lines.append("## Scale sweep at final checkpoint (step143000)")
    lines.append("")
    lines.append("| model | params (M) | inter_kl | repl_kl | gap_kl |")
    lines.append("|---|---|---|---|---|")
    for params, model, r in final_rows:
        gap = mean_gap_from_pairs(r)
        mi = mean_inter_from_pairs(r)
        mr = mean_repl_from_pairs(r)
        lines.append(f"| {model.split('/')[-1]} | {params} | "
                     f"{mi:.4f} | {mr:.4f} | "
                     f"**{gap:+.4f}** |")
    lines.append("")

    # Per-pair distribution analysis at final checkpoint (depth-normalized view)
    try:
        import numpy as np
    except Exception:
        np = None
    if np is not None:
        lines.append("## Per-pair gap distribution at final checkpoint")
        lines.append("")
        lines.append("Mean alone obscures heterogeneity across the layer stack. We also report the "
                     "per-pair gap distribution (median, p75, max). Different scaling trends in these "
                     "statistics reveal whether scale produces uniformly larger gaps or "
                     "**bimodal divergence** (most pairs converge to bisimilarity, a few specialize).")
        lines.append("")
        lines.append("| model | params (M) | n_pairs | mean | median | p75 | max |")
        lines.append("|---|---|---|---|---|---|---|")
        rows_for_corr = []
        for params, model, r in final_rows:
            gaps = np.array(pair_gap_values(r), dtype=float)
            if gaps.size == 0:
                continue
            rows_for_corr.append((params, gaps))
            lines.append(f"| {model.split('/')[-1]} | {params} | {len(gaps)} | "
                         f"{gaps.mean():+.4f} | {np.median(gaps):+.4f} | "
                         f"{np.percentile(gaps, 75):+.4f} | {gaps.max():+.4f} |")
        lines.append("")
        # Scaling correlations across distribution statistics
        if len(rows_for_corr) >= 3:
            xs = np.log10([p for p, _ in rows_for_corr])
            lines.append("**Pearson r (log10 params, statistic) at final step:**")
            lines.append("")
            for stat_name, stat_fn in [
                ("mean", lambda g: float(g.mean())),
                ("median", lambda g: float(np.median(g))),
                ("p75", lambda g: float(np.percentile(g, 75))),
                ("max", lambda g: float(g.max())),
            ]:
                ys = np.array([stat_fn(g) for _, g in rows_for_corr])
                rcoef = float(np.corrcoef(xs, ys)[0, 1])
                lines.append(f"- {stat_name}: r = {rcoef:+.3f}")
            lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append("")
    monotonic_growth_models = []
    for model, rows in by_model.items():
        rows_sorted = sorted(rows, key=lambda r: step_int(r["checkpoint"]))
        gaps = [mean_gap_from_pairs(r) for r in rows_sorted]
        if len(gaps) >= 2 and all(gaps[i] <= gaps[i + 1] + 1e-3 for i in range(len(gaps) - 1)):
            monotonic_growth_models.append(model.split("/")[-1])
    lines.append(f"- Monotonic mean-gap growth with training duration: {len(monotonic_growth_models)}/{len(by_model)} models "
                 f"({', '.join(monotonic_growth_models) or 'none'})")
    if len(final_rows) >= 2:
        gaps_by_scale = [mean_gap_from_pairs(r) for _, _, r in final_rows]
        scale_monotonic = all(gaps_by_scale[i] <= gaps_by_scale[i + 1] + 1e-3 for i in range(len(gaps_by_scale) - 1))
        lines.append(f"- Monotonic mean-gap growth with model scale (at final step): {scale_monotonic}")
        # Pearson scale corr
        try:
            import numpy as np
            xs = np.log10([params for params, _, _ in final_rows])
            ys = np.array(gaps_by_scale)
            r_scale = float(np.corrcoef(xs, ys)[0, 1])
            lines.append(f"- Pearson r (log10(params), mean_gap) = {r_scale:.3f}")
        except Exception:
            pass

    return "\n".join(lines) + "\n"


def make_figures(results: List[dict], fig_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams.update(
            {
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "mathtext.fontset": "stix",
                "font.size": 13,
                "axes.titlesize": 15,
                "axes.labelsize": 14,
                "legend.fontsize": 11,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
            }
        )
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available; skipping figures")
        return

    fig_dir.mkdir(parents=True, exist_ok=True)

    by_model = group_by_model(results)
    if not by_model:
        print("No models left after filtering; skipping protocol-gap figures")
        return

    # Figure 1: Gap vs step, one curve per model
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in sorted(by_model.keys(), key=lambda m: PARAM_COUNTS.get(m, 0)):
        rows = sorted(by_model[model], key=lambda r: step_int(r["checkpoint"]))
        steps = [step_int(r["checkpoint"]) for r in rows]
        gaps = [mean_gap_from_pairs(r) for r in rows]
        steps_plot = [max(s, 1) for s in steps]
        ckpt_max = max(steps)
        label = f"{model.split('/')[-1]} ({PARAM_COUNTS.get(model, '?')}M, to step {ckpt_max})"
        ax.plot(steps_plot, gaps, marker="o", label=label, linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"Protocol gap = $\overline{\mathrm{KL}_{\mathrm{repl}}} - \overline{\mathrm{KL}_{\mathrm{inter}}}$")
    ax.set_title("Protocol gap emerges with training duration and model scale")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "protocol_gap_trajectory.png", dpi=150)
    plt.savefig(fig_dir / "protocol_gap_trajectory.pdf")
    plt.close()
    print(f"Wrote {fig_dir / 'protocol_gap_trajectory.png'}")

    # Figure 2: Final-step gap vs params
    final_rows: List[Tuple[int, str, dict]] = []
    for model, rows in by_model.items():
        final = max(rows, key=lambda r: step_int(r["checkpoint"]))
        final_rows.append((PARAM_COUNTS.get(model, 0), model, final))
    final_rows.sort()

    if len(final_rows) >= 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        params = [p for p, _, _ in final_rows]
        gaps = [mean_gap_from_pairs(r) for _, _, r in final_rows]
        inter = [mean_inter_from_pairs(r) for _, _, r in final_rows]
        repl = [mean_repl_from_pairs(r) for _, _, r in final_rows]
        ax.plot(params, repl, marker="s", label="replacement (bisim) KL", linewidth=2)
        ax.plot(params, inter, marker="^", label="interchange KL", linewidth=2)
        ax.plot(params, gaps, marker="o", label="gap (repl - inter)", linewidth=2, color="C3")
        ax.set_xscale("log")
        ax.set_xlabel("Parameters (M)")
        ax.set_ylabel("Mean KL across adjacent pairs")
        ax.set_title("Protocol divergence at final training step")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "protocol_gap_scale_sweep.png", dpi=150)
        plt.savefig(fig_dir / "protocol_gap_scale_sweep.pdf")
        plt.close()
        print(f"Wrote {fig_dir / 'protocol_gap_scale_sweep.png'}")

    # Figure 3: Per-pair gap distribution at final step (bimodal divergence view)
    if len(final_rows) >= 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0))
        params_arr = [p for p, _, _ in final_rows]
        means, medians, maxes, p75s = [], [], [], []
        for _, _, r in final_rows:
            gaps = np.array(pair_gap_values(r), dtype=float)
            means.append(float(gaps.mean()))
            medians.append(float(np.median(gaps)))
            maxes.append(float(gaps.max()))
            p75s.append(float(np.percentile(gaps, 75)))
        ax1.plot(params_arr, means, marker="o", label="mean", linewidth=2)
        ax1.plot(params_arr, medians, marker="s", label="median", linewidth=2)
        ax1.plot(params_arr, p75s, marker="^", label="p75", linewidth=2)
        ax1.plot(params_arr, maxes, marker="D", label="max", linewidth=2, color="C3")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel("Parameters (M)", fontsize=14)
        ax1.set_ylabel("Per-pair gap (KL_repl - KL_inter)", fontsize=14)
        ax1.set_title("Bimodal divergence: median falls, max rises with scale", fontsize=15)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, which="both")
        for _, model, r in final_rows:
            gaps = np.array(pair_gap_values(r), dtype=float)
            xs = np.arange(len(gaps)) / max(len(gaps) - 1, 1)
            step_tag = step_int(r["checkpoint"])
            ax2.plot(
                xs,
                gaps,
                marker="o",
                label=f"{model.split('/')[-1]} (step {step_tag})",
                linewidth=1.5,
                alpha=0.85,
            )
        ax2.set_yscale("log")
        ax2.set_xlabel("Normalized layer depth (pair index / n_pairs)", fontsize=14)
        ax2.set_ylabel("Per-pair gap (KL_repl - KL_inter)", fontsize=14)
        ax2.set_title("Where the gap concentrates across the stack", fontsize=15)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig(fig_dir / "protocol_gap_distribution.png", dpi=150)
        plt.savefig(fig_dir / "protocol_gap_distribution.pdf")
        plt.close()
        print(f"Wrote {fig_dir / 'protocol_gap_distribution.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--md-out", dest="md_out", required=True)
    ap.add_argument("--fig-dir", dest="fig_dir", default="figures/")
    ap.add_argument(
        "--only-models",
        default=None,
        help="Comma-separated model ids (e.g. EleutherAI/pythia-410m,EleutherAI/pythia-1.4b). "
        "Default: all models in JSON except degenerate trajectories.",
    )
    args = ap.parse_args()

    with open(args.in_path) as f:
        data = json.load(f)

    results = data["results"]
    print(f"Loaded {len(results)} (model, checkpoint) results from {args.in_path}")

    only: Tuple[str, ...] | None = None
    if args.only_models:
        only = tuple(m.strip() for m in args.only_models.split(",") if m.strip())
    plot_results, excluded = filter_results_for_figures(results, only)
    if excluded:
        print("Excluded from figure export:", ", ".join(excluded))

    md = make_summary_md(results, figure_excluded=excluded)
    Path(args.md_out).write_text(md)
    print(f"Wrote {args.md_out}")

    make_figures(plot_results, Path(args.fig_dir))


if __name__ == "__main__":
    main()
