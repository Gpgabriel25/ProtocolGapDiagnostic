#!/usr/bin/env python3
"""
Bootstrap CI head-to-head on Qwen3-8B (TPU v6e-8).

Re-evaluates the harmonized wikitext-2 perplexity for a small set of
configurations, this time saving per-window NLLs so we can compute
block-bootstrap confidence intervals over windows.

Configurations evaluated:
  - baseline (no skipping)
  - beam_bisim n=1..5 from reports/2026-04-18T21-51-24/qwen3_8b_beam_search.json
  - sleb_iter n=1..5 (calibrated SLEB-iterative from harmonized table)
  - sleb_calfree n=1..5 from today's run

Output:
  reports/2026-04-20T10-32-11/sleb_calfree/qwen3_bootstrap_headtohead.json
"""

import os, sys, json, time, logging, gc

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matched_eval_qwen3 import (
    load_and_stack,
    build_forward,
    precompute_rope,
    load_eval_tokens,
    WINDOW,
)
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-8B"
DTYPE = jnp.bfloat16
REPORT_DIR = os.environ.get("REPORT_DIR", "/tmp/sleb_calfree")
STRIDE = 256

CONFIGS = {
    "baseline": [],
    # Beam-search bisimulation (from 2026-04-18T21-51-24/qwen3_8b_beam_search.json)
    "beam_bisim_n1": [20],
    "beam_bisim_n2": [15, 20],
    "beam_bisim_n3": [2, 15, 20],
    "beam_bisim_n4": [2, 15, 16, 20],
    "beam_bisim_n5": [2, 15, 16, 20, 26],
    # Calibrated SLEB-iterative (from harmonized/qwen/matched_eval_results.json)
    "sleb_iter_n1": [17],
    "sleb_iter_n2": [17, 18],
    "sleb_iter_n3": [17, 18, 19],
    "sleb_iter_n5": [17, 18, 19, 20, 21],
    # Calibration-free SLEB (from today's run)
    "sleb_calfree_n1": [16],
    "sleb_calfree_n2": [16, 10],
    "sleb_calfree_n3": [16, 10, 15],
    "sleb_calfree_n5": [16, 10, 15, 2, 30],
}


def evaluate_ppl_with_windows(forward_fn, tokens, layer_weights, embed,
                              final_norm, lm_head, arch, skip_mask, dtype):
    """Like matched_eval_qwen3.evaluate_ppl but returns per-window (nll, n_tokens)."""
    seq_len = len(tokens)
    cos, sin = precompute_rope(WINDOW, arch["head_dim"], arch["rope_theta"], dtype)
    cos, sin = jax.device_put(cos), jax.device_put(sin)
    per_window = []  # list of (nll_sum, n_scored)
    prev_end = 0
    for begin in range(0, seq_len, STRIDE):
        end = min(begin + WINDOW, seq_len)
        target_len = end - prev_end
        chunk = tokens[begin:end]
        actual_len = len(chunk)
        if actual_len < WINDOW:
            chunk_padded = chunk + [0] * (WINDOW - actual_len)
        else:
            chunk_padded = chunk
        ids = jnp.array([chunk_padded], dtype=jnp.int32)
        logits = forward_fn(ids, layer_weights, embed, final_norm, lm_head,
                            cos, sin, skip_mask)
        shift_logits = logits[0, :actual_len - 1, :]
        shift_targets = jnp.array(tokens[begin + 1:begin + actual_len], dtype=jnp.int32)
        log_probs = jax.nn.log_softmax(shift_logits.astype(jnp.float32), axis=-1)
        ce = -log_probs[jnp.arange(len(shift_targets)), shift_targets]
        if target_len < actual_len:
            score_start = actual_len - target_len
        else:
            score_start = 0
        nll = float(jnp.sum(ce[score_start:]))
        n_scored = len(ce) - score_start
        per_window.append((nll, n_scored))
        prev_end = end
        if end == seq_len:
            break
    return per_window


def block_bootstrap_ppl(per_window, n_iters=2000, seed=0):
    """Resample windows with replacement; return (mean_ppl, ci_low, ci_high) at 95%."""
    rng = np.random.default_rng(seed)
    nlls = np.array([w[0] for w in per_window])
    ns = np.array([w[1] for w in per_window])
    n_w = len(per_window)
    samples = []
    for _ in range(n_iters):
        idx = rng.integers(0, n_w, size=n_w)
        total_nll = nlls[idx].sum()
        total_n = ns[idx].sum()
        samples.append(np.exp(total_nll / total_n))
    samples = np.array(samples)
    return float(samples.mean()), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    log.info(f"=== Bootstrap CI head-to-head: {MODEL_NAME} ===")
    log.info(f"Devices: {jax.devices()}")

    t0 = time.time()
    arch, lw, embed, final_norm, lm_head = load_and_stack(MODEL_NAME, DTYPE)
    n_layers = arch["n_layers"]
    log.info(f"Model loaded in {time.time() - t0:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokens, eval_label = load_eval_tokens(tokenizer)
    forward_fn, _ = build_forward(arch)

    # JIT warmup
    no_skip = jnp.zeros(n_layers, dtype=jnp.bool_)
    dummy = jnp.zeros((1, WINDOW), dtype=jnp.int32)
    _ = forward_fn(dummy, lw, embed, final_norm, lm_head,
                   *[jax.device_put(x) for x in precompute_rope(
                       WINDOW, arch["head_dim"], arch["rope_theta"], DTYPE)],
                   no_skip)
    jax.block_until_ready(_)
    log.info("JIT compiled.")

    results = {}
    baseline_pwin = None
    for name, layers in CONFIGS.items():
        skip_mask = jnp.zeros(n_layers, dtype=jnp.bool_)
        if layers:
            skip_mask = skip_mask.at[jnp.array(layers)].set(True)
        log.info(f"Eval {name}: skip={layers}")
        t_e = time.time()
        pwin = evaluate_ppl_with_windows(
            forward_fn, tokens, lw, embed, final_norm, lm_head,
            arch, skip_mask, DTYPE,
        )
        elapsed = time.time() - t_e
        total_nll = sum(w[0] for w in pwin)
        total_n = sum(w[1] for w in pwin)
        ppl = float(np.exp(total_nll / total_n))
        boot_mean, boot_lo, boot_hi = block_bootstrap_ppl(pwin, n_iters=2000)
        log.info(f"  PPL={ppl:.4f}  boot_mean={boot_mean:.4f}  CI=[{boot_lo:.4f}, {boot_hi:.4f}]  ({elapsed:.1f}s)")
        if name == "baseline":
            baseline_pwin = pwin
        results[name] = {
            "layers_removed": layers,
            "n_removed": len(layers),
            "ppl": round(ppl, 4),
            "boot_mean_ppl": round(boot_mean, 4),
            "boot_ci_low": round(boot_lo, 4),
            "boot_ci_high": round(boot_hi, 4),
            "n_windows": len(pwin),
            "n_tokens": total_n,
            "per_window_nll": [(round(n, 4), s) for n, s in pwin],
        }

    # Add delta-CI vs baseline (paired bootstrap on same window resamples)
    baseline_nlls = np.array([w[0] for w in baseline_pwin])
    baseline_ns = np.array([w[1] for w in baseline_pwin])
    n_w = len(baseline_pwin)
    rng = np.random.default_rng(42)
    boot_idx = [rng.integers(0, n_w, size=n_w) for _ in range(2000)]
    base_samples = np.array([
        np.exp(baseline_nlls[idx].sum() / baseline_ns[idx].sum()) for idx in boot_idx
    ])
    for name, info in results.items():
        if name == "baseline":
            continue
        nlls = np.array([n for n, _ in info["per_window_nll"]])
        ns = np.array([s for _, s in info["per_window_nll"]])
        cfg_samples = np.array([
            np.exp(nlls[idx].sum() / ns[idx].sum()) for idx in boot_idx
        ])
        delta = (cfg_samples / base_samples - 1.0) * 100.0
        results[name]["delta_ppl_pct"] = round(float(delta.mean()), 2)
        results[name]["delta_ci_low_pct"] = round(float(np.percentile(delta, 2.5)), 2)
        results[name]["delta_ci_high_pct"] = round(float(np.percentile(delta, 97.5)), 2)
        # Strip per-window arrays from final JSON to keep it small
    for name in list(results.keys()):
        results[name].pop("per_window_nll", None)

    results["_meta"] = {
        "model": MODEL_NAME,
        "evaluator": eval_label,
        "max_words": 5000,
        "window": WINDOW,
        "stride": STRIDE,
        "dtype": "bfloat16",
        "n_layers": n_layers,
        "bootstrap_iters": 2000,
        "ci_level": 0.95,
        "ci_method": "block bootstrap over wikitext sliding windows; paired resample for delta CI",
        "timestamp": time.strftime("%Y-%m-%dT%H-%M-%S"),
        "total_time_s": round(time.time() - t0, 1),
    }

    out = os.path.join(REPORT_DIR, "qwen3_bootstrap_headtohead.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved {out}")

    # Print head-to-head verdict
    log.info("\n=== Head-to-head (delta-PPL %, mean and 95% CI) ===")
    for n in [1, 2, 3, 5]:
        log.info(f"--- n={n} ---")
        for prefix in ["beam_bisim", "sleb_iter", "sleb_calfree"]:
            key = f"{prefix}_n{n}"
            if key in results:
                r = results[key]
                log.info(f"  {prefix:<14} delta={r['delta_ppl_pct']:+6.2f}%  "
                         f"CI=[{r['delta_ci_low_pct']:+6.2f}, {r['delta_ci_high_pct']:+6.2f}]")


if __name__ == "__main__":
    main()
