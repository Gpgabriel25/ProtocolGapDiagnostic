#!/usr/bin/env python3
"""
Matched-Budget Beam-Bisim vs SLEB Calibration-Free on Llama-3.1-8B
===================================================================
Ports matched_budget_qwen3.py to Llama-3.1-8B for architecture-level
matched-budget comparison.

For each TARGET BUDGET in {50, 100, 200, 400, 800}, runs both:
  1. SLEB calibration-free (iterative greedy on held-out evaluator)
  2. Beam-bisim (beam search seeded by interchange KL)

Evaluator protocol matches Qwen3 matched-budget setup:
  - WikiText-2 raw test split
  - first 5K words
  - sliding window=512, stride=256
  - bf16 JAX forward with lax.scan over stacked layer weights

Output: reports/2026-04-22T15-00-00/matched_budget_llama.json
        reports/2026-04-22T15-00-00/budget_pareto_llama.md
"""

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

# Reuse Llama JAX setup/helpers.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matched_eval_llama import (
    DTYPE,
    MAX_WORDS,
    STRIDE,
    WINDOW,
    build_forward,
    load_and_stack,
    load_eval_tokens,
    precompute_rope,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = "meta-llama/Llama-3.1-8B"
REPORT_DIR = "/home/gpgabriel25/Projects/BisimulationQuotient/reports/2026-04-22T15-00-00"
OUTPUT_JSON = os.path.join(REPORT_DIR, "matched_budget_llama.json")
OUTPUT_MD = os.path.join(REPORT_DIR, "budget_pareto_llama.md")
Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)

BUDGETS = [50, 100, 200, 400, 800]

# Suggested matched-budget beam configs for 32-layer Llama-3.1-8B.
BEAM_CONFIGS = {
    50: dict(beam_width=1, seed_k=8, max_n=3),
    100: dict(beam_width=1, seed_k=8, max_n=4),
    200: dict(beam_width=2, seed_k=10, max_n=4),
    400: dict(beam_width=3, seed_k=12, max_n=5),
    800: dict(beam_width=4, seed_k=12, max_n=6),
}

# Per-layer min-neighbor interchange KL scores computed from:
# logs/2026-04-08T08-58-16/llama_bisimulation_results.json
LLAMA_INTERCHANGE_SCORES = {
    0: 13.8028,
    1: 0.0635,
    2: 0.0090,
    3: 0.0090,
    4: 0.0062,
    5: 0.0046,
    6: 0.0039,
    7: 0.0030,
    8: 0.0023,
    9: 0.0023,
    10: 0.0025,
    11: 0.0023,
    12: 0.0023,
    13: 0.0027,
    14: 0.0029,
    15: 0.0035,
    16: 0.0044,
    17: 0.0035,
    18: 0.0031,
    19: 0.0022,
    20: 0.0022,
    21: 0.0023,
    22: 0.0019,
    23: 0.0019,
    24: 0.0016,
    25: 0.0014,
    26: 0.0014,
    27: 0.0018,
    28: 0.0033,
    29: 0.0034,
    30: 0.0392,
    31: 0.1128,
}


def make_skip_mask(skip_set, n_layers):
    mask = jnp.zeros(n_layers, dtype=jnp.bool_)
    if skip_set:
        mask = mask.at[jnp.array(sorted(skip_set))].set(True)
    return mask


def evaluate_ppl(forward_fn, tokens, layer_weights, embed, final_norm, lm_head,
                 arch, skip_mask, cos, sin):
    """Sliding-window perplexity using fixed precomputed RoPE tensors."""
    seq_len = len(tokens)
    total_nll = 0.0
    total_tokens = 0
    n_windows = 0
    prev_end = 0

    for begin in range(0, seq_len, STRIDE):
        end = min(begin + WINDOW, seq_len)
        target_len = end - prev_end

        chunk = tokens[begin:end]
        actual_len = len(chunk)
        if actual_len < WINDOW:
            chunk = chunk + [0] * (WINDOW - actual_len)

        input_ids = jnp.array([chunk], dtype=jnp.int32)
        logits = forward_fn(
            input_ids,
            layer_weights,
            embed,
            final_norm,
            lm_head,
            cos,
            sin,
            skip_mask,
        )

        shift_logits = logits[0, :actual_len - 1, :]
        shift_targets = jnp.array(tokens[begin + 1:begin + actual_len], dtype=jnp.int32)

        log_probs = jax.nn.log_softmax(shift_logits.astype(jnp.float32), axis=-1)
        ce = -log_probs[jnp.arange(len(shift_targets)), shift_targets]

        if target_len < actual_len:
            score_start = actual_len - target_len
        else:
            score_start = 0

        total_nll += float(jnp.sum(ce[score_start:]))
        total_tokens += len(ce) - score_start
        n_windows += 1
        prev_end = end

        if end == seq_len:
            break

    ppl = np.exp(total_nll / total_tokens)
    log.info(f"  PPL={ppl:.4f} ({n_windows} windows, {total_tokens} tokens scored)")
    return float(ppl), int(total_tokens)


# SLEB iterative greedy cost: sum_{k=1..n} (n_layers - k + 1)
def sleb_max_n_for_budget(budget, n_layers=32):
    cum = 0
    for n in range(1, n_layers + 1):
        cum += (n_layers - n + 1)
        if cum > budget:
            return min(max(n - 1, 1), n_layers - 1)
    return n_layers - 1


def sleb_iterative(forward_fn, tokens, lw, embed, final_norm, lm_head,
                   arch, n_layers, cos, sin, max_n, eval_budget):
    """SLEB iterative greedy with a strict evaluator-call budget cap."""
    removed = []
    eval_count = 0
    history = []

    for step in range(1, max_n + 1):
        candidates = [layer for layer in range(n_layers) if layer not in removed]
        if eval_count + len(candidates) > eval_budget:
            log.info(
                "  SLEB step %s: stopping early (would exceed budget %s; cur=%s, +%s)",
                step,
                eval_budget,
                eval_count,
                len(candidates),
            )
            break

        ppls = []
        for layer in candidates:
            skip_set = frozenset(removed + [layer])
            skip_mask = make_skip_mask(skip_set, n_layers)
            ppl, _ = evaluate_ppl(
                forward_fn,
                tokens,
                lw,
                embed,
                final_norm,
                lm_head,
                arch,
                skip_mask,
                cos,
                sin,
            )
            ppls.append((layer, ppl))
            eval_count += 1

        best_layer, best_ppl = min(ppls, key=lambda x: x[1])
        removed.append(best_layer)
        history.append(
            {
                "step": step,
                "layer_added": best_layer,
                "removed_so_far": list(removed),
                "ppl": float(best_ppl),
                "eval_count_after": eval_count,
            }
        )
        log.info(
            "  SLEB step %s: removed L=%s, set=%s, PPL=%.4f, evals=%s",
            step,
            best_layer,
            removed,
            best_ppl,
            eval_count,
        )

    return history, eval_count


def beam_search_capped(forward_fn, tokens, lw, embed, final_norm, lm_head,
                       arch, bisim_scores, n_layers, cos, sin,
                       beam_width, seed_k, max_n, eval_budget):
    """Beam search over skip sets with an evaluator-call cap."""
    cache = {}
    eval_count = 0
    history = []

    def eval_skip(skip_set):
        nonlocal eval_count
        key = frozenset(skip_set)
        if key in cache:
            return cache[key]
        skip_mask = make_skip_mask(skip_set, n_layers)
        ppl, _ = evaluate_ppl(
            forward_fn,
            tokens,
            lw,
            embed,
            final_norm,
            lm_head,
            arch,
            skip_mask,
            cos,
            sin,
        )
        cache[key] = ppl
        eval_count += 1
        return ppl

    sorted_layers = sorted(bisim_scores.items(), key=lambda x: x[1])
    seed_layers = [layer for layer, _ in sorted_layers[:seed_k]]

    if eval_count + len(seed_layers) > eval_budget:
        log.info("  Beam: stopping before step 1 (seed=%s > budget left)", len(seed_layers))
        return history, eval_count

    n1_beams = []
    for layer in seed_layers:
        skip_set = frozenset({layer})
        ppl = eval_skip(skip_set)
        n1_beams.append((skip_set, ppl))

    n1_beams.sort(key=lambda x: x[1])
    best_skip, best_ppl = n1_beams[0]
    history.append(
        {
            "step": 1,
            "best_skip": sorted(best_skip),
            "ppl": float(best_ppl),
            "eval_count_after": eval_count,
        }
    )
    log.info("  Beam step 1: best=%s PPL=%.4f evals=%s", sorted(best_skip), best_ppl, eval_count)
    beams = n1_beams[:beam_width]

    for step in range(2, max_n + 1):
        new_candidates = {}
        max_extra = beam_width * (n_layers - step + 1)
        if eval_count + max_extra > eval_budget * 1.05:
            log.info("  Beam step %s: stopping (would exceed budget; +up to %s)", step, max_extra)
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

        sorted_candidates = sorted(new_candidates.items(), key=lambda x: x[1])
        beams = [(skip, ppl) for skip, ppl in sorted_candidates[:beam_width]]
        best_skip, best_ppl = beams[0]
        history.append(
            {
                "step": step,
                "best_skip": sorted(best_skip),
                "ppl": float(best_ppl),
                "eval_count_after": eval_count,
            }
        )
        log.info("  Beam step %s: best=%s PPL=%.4f evals=%s", step, sorted(best_skip), best_ppl, eval_count)

    return history, eval_count


def main():
    log.info("=== Matched-Budget Beam-Bisim vs SLEB Calibration-Free (Llama) ===")
    log.info("Devices: %s", jax.devices())
    log.info("Output JSON: %s", OUTPUT_JSON)

    t_start = time.time()

    arch, lw, embed, final_norm, lm_head = load_and_stack(MODEL_NAME, DTYPE)
    n_layers = arch["n_layers"]
    log.info("Model loaded (%0.1fs); n_layers=%s", time.time() - t_start, n_layers)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token and os.path.exists("/tmp/hf_token"):
        hf_token = open("/tmp/hf_token").read().strip()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    tokens, eval_label = load_eval_tokens(tokenizer)

    forward_fn, _ = build_forward(arch)
    log.info("JIT compiling...")
    cos_w, sin_w = precompute_rope(WINDOW, arch["head_dim"], arch["rope_theta"], DTYPE)
    cos_w = jax.device_put(cos_w)
    sin_w = jax.device_put(sin_w)
    dummy_mask = jnp.zeros(n_layers, dtype=jnp.bool_)
    dummy_ids = jnp.zeros((1, WINDOW), dtype=jnp.int32)
    warmup = forward_fn(dummy_ids, lw, embed, final_norm, lm_head, cos_w, sin_w, dummy_mask)
    jax.block_until_ready(warmup)
    log.info("JIT compiled (%0.1fs)", time.time() - t_start)

    log.info("Computing baseline PPL...")
    baseline_ppl, n_tokens = evaluate_ppl(
        forward_fn,
        tokens,
        lw,
        embed,
        final_norm,
        lm_head,
        arch,
        dummy_mask,
        cos_w,
        sin_w,
    )
    log.info("Baseline PPL = %.4f (n_tokens=%s)", baseline_ppl, n_tokens)

    results = {
        "model": MODEL_NAME,
        "baseline_ppl": float(baseline_ppl),
        "n_eval_tokens": int(n_tokens),
        "n_layers": int(n_layers),
        "evaluator": {
            "dataset": eval_label,
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
        log.info("\n=== BUDGET = %s evaluator calls ===", budget)

        sleb_max_n = sleb_max_n_for_budget(budget, n_layers)
        log.info("SLEB iterative (target max_n=%s, budget=%s)", sleb_max_n, budget)
        t0 = time.time()
        sleb_history, sleb_evals = sleb_iterative(
            forward_fn,
            tokens,
            lw,
            embed,
            final_norm,
            lm_head,
            arch,
            n_layers,
            cos_w,
            sin_w,
            max_n=sleb_max_n,
            eval_budget=budget,
        )
        sleb_wall = time.time() - t0
        sleb_final = sleb_history[-1] if sleb_history else None
        results["results"].append(
            {
                "method": "sleb_calfree",
                "budget_target": budget,
                "evals_actual": int(sleb_evals),
                "wall_clock_s": float(sleb_wall),
                "n_removed": len(sleb_final["removed_so_far"]) if sleb_final else 0,
                "layers_removed": sleb_final["removed_so_far"] if sleb_final else [],
                "final_ppl": float(sleb_final["ppl"]) if sleb_final else None,
                "ppl_delta_pct": (
                    100.0 * (sleb_final["ppl"] / baseline_ppl - 1.0)
                    if sleb_final
                    else None
                ),
                "history": sleb_history,
            }
        )
        with open(OUTPUT_JSON, "w") as f:
            json.dump(results, f, indent=2)

        bcfg = BEAM_CONFIGS[budget]
        log.info("Beam-bisim (%s, budget=%s)", bcfg, budget)
        t0 = time.time()
        beam_history, beam_evals = beam_search_capped(
            forward_fn,
            tokens,
            lw,
            embed,
            final_norm,
            lm_head,
            arch,
            LLAMA_INTERCHANGE_SCORES,
            n_layers,
            cos_w,
            sin_w,
            beam_width=bcfg["beam_width"],
            seed_k=bcfg["seed_k"],
            max_n=bcfg["max_n"],
            eval_budget=budget,
        )
        beam_wall = time.time() - t0
        beam_final = beam_history[-1] if beam_history else None
        results["results"].append(
            {
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
                    if beam_final
                    else None
                ),
                "history": beam_history,
            }
        )
        with open(OUTPUT_JSON, "w") as f:
            json.dump(results, f, indent=2)

        gc.collect()

    table_lines = ["# Matched-Budget Beam-Bisim vs SLEB on Llama-3.1-8B", ""]
    table_lines.append(f"Baseline PPL = {baseline_ppl:.4f}; n_tokens = {n_tokens}")
    table_lines.append("")
    table_lines.append("| Budget | Method | Evals | Wall (s) | n removed | Layers | PPL | Delta % |")
    table_lines.append("|---:|---|---:|---:|---:|---|---:|---:|")

    budget_summary = {}
    for r in results["results"]:
        if r["final_ppl"] is not None:
            row = (
                f"| {r['budget_target']} | {r['method']} | {r['evals_actual']} "
                f"| {r['wall_clock_s']:.1f} | {r['n_removed']} | {r['layers_removed']} "
                f"| {r['final_ppl']:.4f} | {r['ppl_delta_pct']:+.2f} |"
            )
        else:
            row = (
                f"| {r['budget_target']} | {r['method']} | {r['evals_actual']} "
                "| --- | 0 | [] | --- | --- |"
            )
        table_lines.append(row)

        b = str(r["budget_target"])
        if b not in budget_summary:
            budget_summary[b] = {}
        budget_summary[b][r["method"]] = {
            "evals_actual": r["evals_actual"],
            "n_removed": r["n_removed"],
            "layers_removed": r["layers_removed"],
            "final_ppl": r["final_ppl"],
            "ppl_delta_pct": r["ppl_delta_pct"],
            "wall_clock_s": round(r["wall_clock_s"], 1),
        }

    table_lines.append("")
    table_lines.append(f"Total wall clock: {(time.time() - t_start) / 60:.1f} min")

    Path(OUTPUT_MD).write_text("\n".join(table_lines))

    # End-of-run printed summaries as both JSON and markdown table.
    log.info("\n=== Budget Summary (JSON) ===\n%s", json.dumps(budget_summary, indent=2))
    log.info("\n=== Budget Summary (Markdown) ===\n%s", "\n".join(table_lines))
    log.info("Summary written to %s", OUTPUT_MD)
    log.info("Total wall: %.1f min", (time.time() - t_start) / 60.0)


if __name__ == "__main__":
    main()
