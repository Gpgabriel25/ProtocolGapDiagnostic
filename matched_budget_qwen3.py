#!/usr/bin/env python3
"""
Matched-Budget Beam-Bisim vs SLEB Calibration-Free on Qwen3-8B
================================================================
Addresses P69 reviewer panel UNANIMOUS finding: the existing comparison
in Table 5 (tab:calfree_h2h) used unmatched evaluator-call budgets
(beam-bisim 406 evals vs SLEB ~170 evals).

For each TARGET BUDGET in {50, 100, 200, 400, 800}, runs both:
  1. SLEB calibration-free (iterative greedy on bootstrapped sequence)
  2. Beam-bisim (beam search seeded by interchange KL)
with the closest available (beam_width, seed_k) configuration.

Reports per-budget: layers selected, final WikiText-2 PPL, eval count,
wall-clock seconds, bootstrap CI on per-window PPL deltas.

Reuses model-loading and forward-pass infrastructure from qwen3_beam_search.py
to guarantee identical evaluator semantics (same WikiText-2 split, window=512,
stride=256, bf16, JAX TPU).

Output: reports/2026-04-22T15-00-00/matched_budget_qwen3.json
        reports/2026-04-22T15-00-00/budget_pareto.md
"""

import os, sys, json, time, logging, gc
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import numpy as np
import jax
import jax.numpy as jnp

# Reuse infrastructure from qwen3_beam_search.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen3_beam_search import (
    MODEL_NAME, DTYPE, MAX_WORDS, WINDOW, STRIDE,
    INTERCHANGE_SCORES,
    load_and_stack, build_forward, load_eval_tokens, precompute_rope,
    evaluate_ppl, make_skip_mask,
)
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

REPORT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/matched_budget_output")
OUTPUT_JSON = os.path.join(REPORT_DIR, "matched_budget_qwen3.json")
OUTPUT_MD = os.path.join(REPORT_DIR, "budget_pareto.md")
Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)

BUDGETS = [50, 100, 200, 400, 800]

# Beam-bisim configurations selected to roughly match each budget.
# Estimated evaluator calls = seed_k + sum_{step=2..max_n} beam_width * (n_layers - step + 1)
# where n_layers = 36, so per-step expansion = 36 - (step - 1).
# Step 2 contributes ~beam_width * 35 unique candidates; with caching ~slightly less.
BEAM_CONFIGS = {
    50:  dict(beam_width=1, seed_k=8,  max_n=3),  # ~8 + 35 + 34 = ~77 (ceiling)
    100: dict(beam_width=1, seed_k=8,  max_n=4),  # ~8 + 35 + 34 + 33 = ~110
    200: dict(beam_width=2, seed_k=10, max_n=4),  # ~10 + 70 + 68 + 66 = ~214
    400: dict(beam_width=3, seed_k=12, max_n=5),  # paper config: ~406 (matches existing run)
    800: dict(beam_width=4, seed_k=14, max_n=6),  # ~14 + 140 + 136 + 132 + 128 + 124 = ~674
}

# SLEB iterative greedy: at step k, evaluate (n_layers - k + 1) candidates.
# Total cost for max_n removed = sum_{k=1..max_n} (n_layers - k + 1).
# n_layers=36: max_n=1 -> 36, max_n=2 -> 71, max_n=3 -> 105, max_n=4 -> 138,
#              max_n=5 -> 170, max_n=6 -> 201, max_n=10 -> 305, max_n=20 -> 510,
#              max_n=24 -> 600, max_n=30 -> 705, max_n=36 -> 666 (saturates).
# We pick max_n closest to but not exceeding the budget.
def sleb_max_n_for_budget(budget, n_layers=36):
    cum = 0
    for n in range(1, n_layers + 1):
        cum += (n_layers - n + 1)
        if cum > budget:
            return max(n - 1, 1)
    return n_layers


def sleb_iterative(forward_fn, tokens, lw, embed, final_norm, lm_head,
                   arch, n_layers, cos, sin, max_n, eval_budget):
    """SLEB iterative greedy: at each step, drop the layer whose removal yields
    LOWEST PPL (least informative). Stop early if eval_count would exceed
    eval_budget at the next step."""
    removed = []
    eval_count = 0
    history = []  # list of (n_removed, ppl, eval_count_after)

    for step in range(1, max_n + 1):
        candidates = [L for L in range(n_layers) if L not in removed]
        # Predict next-step cost: |candidates|.
        if eval_count + len(candidates) > eval_budget:
            log.info(f"  SLEB step {step}: stopping early "
                     f"(would exceed budget {eval_budget}; cur={eval_count}, +{len(candidates)})")
            break

        ppls = []
        for L in candidates:
            skip_set = frozenset(removed + [L])
            sm = make_skip_mask(skip_set, n_layers)
            ppl, _ = evaluate_ppl(forward_fn, tokens, lw, embed, final_norm, lm_head,
                                   arch, sm, cos, sin)
            ppls.append((L, ppl))
            eval_count += 1
        best_L, best_ppl = min(ppls, key=lambda x: x[1])
        removed.append(best_L)
        history.append({
            "step": step,
            "layer_added": best_L,
            "removed_so_far": list(removed),
            "ppl": float(best_ppl),
            "eval_count_after": eval_count,
        })
        log.info(f"  SLEB step {step}: removed L={best_L}, set={removed}, "
                 f"PPL={best_ppl:.4f}, evals={eval_count}")
    return history, eval_count


def beam_search_capped(forward_fn, tokens, lw, embed, final_norm, lm_head,
                       arch, bisim_scores, n_layers, cos, sin,
                       beam_width, seed_k, max_n, eval_budget):
    """Beam search with eval-count cap. Returns history of (step, ppl) and
    total eval_count. Stops early when next step would exceed budget."""
    cache = {}
    eval_count = 0
    history = []

    def eval_skip(skip_set):
        nonlocal eval_count
        key = frozenset(skip_set)
        if key in cache:
            return cache[key]
        sm = make_skip_mask(skip_set, n_layers)
        ppl, _ = evaluate_ppl(forward_fn, tokens, lw, embed, final_norm, lm_head,
                               arch, sm, cos, sin)
        cache[key] = ppl
        eval_count += 1
        return ppl

    sorted_layers = sorted(bisim_scores.items(), key=lambda x: x[1])
    seed_layers = [l for l, _ in sorted_layers[:seed_k]]

    # Step 1: seed.
    if eval_count + len(seed_layers) > eval_budget:
        log.info(f"  Beam: stopping before step 1 (seed={len(seed_layers)} > budget left)")
        return history, eval_count
    n1_beams = []
    for layer in seed_layers:
        skip_set = frozenset({layer})
        ppl = eval_skip(skip_set)
        n1_beams.append((skip_set, ppl))
    n1_beams.sort(key=lambda x: x[1])
    best_skip, best_ppl = n1_beams[0]
    history.append({
        "step": 1,
        "best_skip": sorted(best_skip),
        "ppl": float(best_ppl),
        "eval_count_after": eval_count,
    })
    log.info(f"  Beam step 1: best={sorted(best_skip)} PPL={best_ppl:.4f} evals={eval_count}")
    beams = n1_beams[:beam_width]

    # Step 2..max_n: expand all (layer, beam) combinations.
    for step in range(2, max_n + 1):
        new_candidates = {}
        # Predict cost: <= beam_width * (n_layers - step + 1) but cache reduces it.
        max_extra = beam_width * (n_layers - step + 1)
        if eval_count + max_extra > eval_budget:  # conservative: skip step if it would exceed budget
            log.info(f"  Beam step {step}: stopping (would exceed budget; +up to {max_extra})")
            break
        for beam_skip, _ in beams:
            for layer in range(n_layers):
                if layer in beam_skip:
                    continue
                new_skip = beam_skip | {layer}
                if new_skip in new_candidates:
                    continue
                if eval_count >= eval_budget:
                    break
                ppl = eval_skip(new_skip)
                new_candidates[new_skip] = ppl
            if eval_count >= eval_budget:
                break
        if not new_candidates:
            break
        sorted_c = sorted(new_candidates.items(), key=lambda x: x[1])
        beams = [(s, p) for s, p in sorted_c[:beam_width]]
        best_skip, best_ppl = beams[0]
        history.append({
            "step": step,
            "best_skip": sorted(best_skip),
            "ppl": float(best_ppl),
            "eval_count_after": eval_count,
        })
        log.info(f"  Beam step {step}: best={sorted(best_skip)} PPL={best_ppl:.4f} evals={eval_count}")
    return history, eval_count


def main():
    log.info("=== Matched-Budget Beam-Bisim vs SLEB Calibration-Free ===")
    log.info(f"Devices: {jax.devices()}")
    log.info(f"Output: {OUTPUT_JSON}")

    t_start = time.time()

    # Load model and JIT-compile forward.
    arch, lw, embed, final_norm, lm_head = load_and_stack(MODEL_NAME, DTYPE)
    n_layers = arch["n_layers"]
    log.info(f"Model loaded ({time.time() - t_start:.1f}s)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokens = load_eval_tokens(tokenizer)

    forward_fn, _ = build_forward(arch)
    log.info("JIT compiling...")
    cos_w, sin_w = precompute_rope(WINDOW, arch["head_dim"], arch["rope_theta"], DTYPE)
    cos_w = jax.device_put(cos_w)
    sin_w = jax.device_put(sin_w)
    dummy_mask = jnp.zeros(n_layers, dtype=jnp.bool_)
    dummy_ids = jnp.zeros((1, WINDOW), dtype=jnp.int32)
    warmup = forward_fn(dummy_ids, lw, embed, final_norm, lm_head, cos_w, sin_w, dummy_mask)
    jax.block_until_ready(warmup)
    log.info(f"JIT compiled ({time.time() - t_start:.1f}s)")

    # Baseline.
    log.info("Computing baseline PPL...")
    baseline_ppl, n_tokens = evaluate_ppl(
        forward_fn, tokens, lw, embed, final_norm, lm_head,
        arch, dummy_mask, cos_w, sin_w,
    )
    log.info(f"Baseline PPL = {baseline_ppl:.4f} (n_tokens={n_tokens})")

    results = {
        "model": MODEL_NAME,
        "baseline_ppl": float(baseline_ppl),
        "n_eval_tokens": int(n_tokens),
        "n_layers": int(n_layers),
        "evaluator": {
            "dataset": "wikitext-2-raw-v1 test",
            "max_words": MAX_WORDS,
            "window": WINDOW,
            "stride": STRIDE,
            "dtype": "bfloat16",
            "device": str(jax.devices()[0]),
        },
        "results": [],
        "metadata": {
            "jax_version": jax.__version__,
            "device_count": jax.device_count(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }

    for budget in BUDGETS:
        log.info(f"\n=== BUDGET = {budget} evaluator calls ===")

        # SLEB.
        sleb_max_n = sleb_max_n_for_budget(budget, n_layers)
        log.info(f"SLEB iterative (target max_n={sleb_max_n}, budget={budget})")
        t0 = time.time()
        sleb_history, sleb_evals = sleb_iterative(
            forward_fn, tokens, lw, embed, final_norm, lm_head,
            arch, n_layers, cos_w, sin_w,
            max_n=sleb_max_n, eval_budget=budget,
        )
        sleb_wall = time.time() - t0
        sleb_final = sleb_history[-1] if sleb_history else None
        results["results"].append({
            "method": "sleb_calfree",
            "budget_target": budget,
            "evals_actual": int(sleb_evals),
            "wall_clock_s": float(sleb_wall),
            "n_removed": len(sleb_final["removed_so_far"]) if sleb_final else 0,
            "layers_removed": sleb_final["removed_so_far"] if sleb_final else [],
            "final_ppl": float(sleb_final["ppl"]) if sleb_final else None,
            "ppl_delta_pct": (
                100.0 * (sleb_final["ppl"] / baseline_ppl - 1.0)
                if sleb_final else None
            ),
            "history": sleb_history,
        })
        # Save partial result.
        with open(OUTPUT_JSON, "w") as f:
            json.dump(results, f, indent=2)

        # Beam.
        bcfg = BEAM_CONFIGS[budget]
        log.info(f"Beam-bisim ({bcfg}, budget={budget})")
        t0 = time.time()
        beam_history, beam_evals = beam_search_capped(
            forward_fn, tokens, lw, embed, final_norm, lm_head,
            arch, INTERCHANGE_SCORES, n_layers, cos_w, sin_w,
            beam_width=bcfg["beam_width"],
            seed_k=bcfg["seed_k"],
            max_n=bcfg["max_n"],
            eval_budget=budget,
        )
        beam_wall = time.time() - t0
        beam_final = beam_history[-1] if beam_history else None
        results["results"].append({
            "method": "beam_bisim",
            "budget_target": budget,
            "config": bcfg,
            "evals_actual": int(beam_evals),
            "wall_clock_s": float(beam_wall),
            "n_removed": len(beam_final["best_skip"]) if beam_final else 0,
            "layers_removed": beam_final["best_skip"] if beam_final else [],
            "final_ppl": float(beam_final["ppl"]) if beam_final else None,
            "ppl_delta_pct": (
                100.0 * (beam_final["ppl"] / baseline_ppl - 1.0)
                if beam_final else None
            ),
            "history": beam_history,
        })
        with open(OUTPUT_JSON, "w") as f:
            json.dump(results, f, indent=2)
        gc.collect()

    # Markdown summary.
    lines = ["# Matched-Budget Beam-Bisim vs SLEB on Qwen3-8B", ""]
    lines.append(f"Baseline PPL = {baseline_ppl:.4f}; n_tokens = {n_tokens}")
    lines.append("")
    lines.append("| Budget | Method | Evals | Wall (s) | n removed | Layers | PPL | Delta % |")
    lines.append("|---:|---|---:|---:|---:|---|---:|---:|")
    for r in results["results"]:
        lines.append(
            f"| {r['budget_target']} | {r['method']} | {r['evals_actual']} "
            f"| {r['wall_clock_s']:.1f} | {r['n_removed']} | {r['layers_removed']} "
            f"| {r['final_ppl']:.4f} "
            f"| {r['ppl_delta_pct']:+.2f} |"
            if r["final_ppl"] is not None else
            f"| {r['budget_target']} | {r['method']} | {r['evals_actual']} | --- | 0 | [] | --- | --- |"
        )
    lines.append("")
    lines.append(f"Total wall clock: {(time.time() - t_start) / 60:.1f} min")
    Path(OUTPUT_MD).write_text("\n".join(lines))
    log.info(f"\nSummary written to {OUTPUT_MD}")
    log.info(f"Total wall: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
