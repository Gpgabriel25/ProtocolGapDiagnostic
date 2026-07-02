#!/usr/bin/env python3
"""
Calibration-free SLEB head-to-head on Qwen3-8B (TPU v6e-8).

This is a calibration-free SLEB variant: instead of using a held-out
calibration corpus (the standard SLEB protocol), we self-bootstrap a
fixed pseudo-text sequence from the model itself by greedy-decoding
256 tokens from the BOS token. The iterative SLEB importance ranking
is then computed on this self-bootstrapped sequence.

Final perplexity for each chosen layer set is measured on the
harmonized wikitext-2 evaluator (max_words=5000, window=512, stride=256)
matching reports/2026-04-18T18-21-54/harmonized/qwen/.

Outputs:
  - reports/<cycle-id>/sleb_calfree/qwen3_8b_sleb_calfree.json
    {sleb_calfree_n1..n5: {layers_removed, ppl, delta_ppl_pct}, ...}
"""

import os, sys, json, time, logging, gc

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

# Reuse weight loading + forward + eval from matched_eval_qwen3
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matched_eval_qwen3 import (
    load_and_stack,
    build_forward,
    precompute_rope,
    load_eval_tokens,
    evaluate_ppl,
    WINDOW,
)
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-8B"
DTYPE = jnp.bfloat16
REPORT_DIR = os.environ.get("REPORT_DIR", "/tmp/sleb_calfree")

# Self-bootstrap config: greedy-decode this many tokens from BOS using the
# FULL (no skipping) model. Length must be <= WINDOW (=512).
BOOTSTRAP_LEN = 256

# SLEB target budgets (matched to harmonized eval table)
N_BUDGETS = [1, 2, 3, 5]


def greedy_generate(forward_fn, lw, embed, final_norm, lm_head, cos, sin,
                    n_layers, bos_id, length):
    """Greedy-decode `length` tokens starting from the BOS token, using the
    FULL model (no skipping). Returns int32 array of length `length`.

    Uses naive O(L^2) prefix forward passes (no KV cache) to stay inside
    the existing JIT-compiled fixed-WINDOW forward function.
    """
    no_skip = jnp.zeros(n_layers, dtype=jnp.bool_)
    tokens = [int(bos_id)]
    for _ in range(length - 1):
        prefix = tokens[:]
        # Pad to WINDOW
        prefix_padded = prefix + [0] * (WINDOW - len(prefix))
        ids = jnp.array([prefix_padded], dtype=jnp.int32)
        logits = forward_fn(ids, lw, embed, final_norm, lm_head, cos, sin, no_skip)
        # logits[0, len(prefix)-1, :] is the next-token distribution
        next_id = int(jnp.argmax(logits[0, len(prefix) - 1, :]))
        tokens.append(next_id)
    return tokens


def nll_on_sequence(forward_fn, lw, embed, final_norm, lm_head, cos, sin,
                    tokens, skip_mask):
    """Compute total NLL of `tokens` (length L <= WINDOW) under the model
    with `skip_mask` applied. Returns float NLL summed over positions."""
    L = len(tokens)
    padded = tokens + [0] * (WINDOW - L)
    ids = jnp.array([padded], dtype=jnp.int32)
    logits = forward_fn(ids, lw, embed, final_norm, lm_head, cos, sin, skip_mask)
    shift_logits = logits[0, :L - 1, :]
    shift_targets = jnp.array(tokens[1:L], dtype=jnp.int32)
    log_probs = jax.nn.log_softmax(shift_logits.astype(jnp.float32), axis=-1)
    ce = -log_probs[jnp.arange(L - 1), shift_targets]
    return float(jnp.sum(ce))


def sleb_iterative_calfree(forward_fn, lw, embed, final_norm, lm_head, cos, sin,
                            n_layers, bootstrap_tokens, max_n):
    """Iterative SLEB on the self-bootstrapped sequence. Returns a dict
    {n: [layers_removed_in_order]} for n in 1..max_n.

    At each step k:
      1. For each surviving layer L, compute NLL with (already_removed ∪ {L}).
      2. Remove the layer that produces the LOWEST NLL (least informative).
    """
    removed = []
    history = {}
    for k in range(1, max_n + 1):
        candidates = [L for L in range(n_layers) if L not in removed]
        nlls = []
        for L in candidates:
            mask_layers = removed + [L]
            skip_mask = jnp.zeros(n_layers, dtype=jnp.bool_)
            skip_mask = skip_mask.at[jnp.array(mask_layers)].set(True)
            nll = nll_on_sequence(
                forward_fn, lw, embed, final_norm, lm_head, cos, sin,
                bootstrap_tokens, skip_mask,
            )
            nlls.append((L, nll))
        # Pick layer with lowest resulting NLL
        nlls.sort(key=lambda x: x[1])
        best_L, best_nll = nlls[0]
        removed.append(best_L)
        history[k] = list(removed)
        log.info(
            f"  SLEB-calfree step {k}: removed layer {best_L} "
            f"(bootstrap NLL={best_nll:.4f}); cumulative removed={removed}"
        )
    return history


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    log.info(f"=== Calibration-Free SLEB on {MODEL_NAME} ===")
    log.info(f"Devices: {jax.devices()}")
    log.info(f"Report dir: {REPORT_DIR}")

    t0 = time.time()
    arch, lw, embed, final_norm, lm_head = load_and_stack(MODEL_NAME, DTYPE)
    n_layers = arch["n_layers"]
    log.info(f"Model loaded in {time.time() - t0:.1f}s ({n_layers} layers)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.eos_token_id  # Qwen3 uses <|endoftext|>; fallback OK
    log.info(f"BOS token id: {bos_id}")

    forward_fn, _ = build_forward(arch)
    cos, sin = precompute_rope(WINDOW, arch["head_dim"], arch["rope_theta"], DTYPE)
    cos, sin = jax.device_put(cos), jax.device_put(sin)

    # JIT warmup
    log.info("JIT warmup...")
    no_skip = jnp.zeros(n_layers, dtype=jnp.bool_)
    dummy = jnp.zeros((1, WINDOW), dtype=jnp.int32)
    _ = forward_fn(dummy, lw, embed, final_norm, lm_head, cos, sin, no_skip)
    jax.block_until_ready(_)
    log.info("JIT compiled.")

    # Step 1: self-bootstrap the calibration-free sequence
    log.info(f"Greedy-decoding {BOOTSTRAP_LEN} tokens from BOS (full model)...")
    t_gen = time.time()
    bootstrap_tokens = greedy_generate(
        forward_fn, lw, embed, final_norm, lm_head, cos, sin,
        n_layers, bos_id, BOOTSTRAP_LEN,
    )
    log.info(f"Bootstrap done in {time.time() - t_gen:.1f}s")
    log.info(f"First 16 bootstrap token ids: {bootstrap_tokens[:16]}")
    sample_text = tokenizer.decode(bootstrap_tokens[:64], skip_special_tokens=False)
    log.info(f"Decoded preview (first 64 tok): {sample_text!r}")

    # Step 2: iterative SLEB ranking on the self-bootstrapped sequence
    max_n = max(N_BUDGETS)
    log.info(f"Running iterative SLEB-calfree up to n={max_n}...")
    t_rank = time.time()
    history = sleb_iterative_calfree(
        forward_fn, lw, embed, final_norm, lm_head, cos, sin,
        n_layers, bootstrap_tokens, max_n,
    )
    log.info(f"SLEB ranking done in {time.time() - t_rank:.1f}s")

    # Step 3: evaluate each chosen layer set on the harmonized wikitext eval
    log.info("Loading harmonized eval tokens (wikitext, max_words=5000)...")
    eval_tokens, eval_label = load_eval_tokens(tokenizer)

    # Baseline (no skipping) on the harmonized eval
    log.info("Baseline PPL on harmonized eval (no skipping)...")
    no_skip = jnp.zeros(n_layers, dtype=jnp.bool_)
    baseline_ppl, n_tok = evaluate_ppl(
        forward_fn, eval_tokens, lw, embed, final_norm, lm_head,
        arch, no_skip, dtype=DTYPE,
    )
    log.info(f"Baseline PPL = {baseline_ppl:.4f} ({n_tok} tokens)")

    results = {
        "baseline": {
            "layers_removed": [],
            "n_removed": 0,
            "ppl": round(float(baseline_ppl), 4),
            "delta_ppl_pct": 0.0,
            "n_tokens": n_tok,
        }
    }

    for n in N_BUDGETS:
        layers = history[n]
        skip_mask = jnp.zeros(n_layers, dtype=jnp.bool_)
        skip_mask = skip_mask.at[jnp.array(layers)].set(True)
        log.info(f"Eval sleb_calfree_n{n}: skip={layers}")
        ppl, n_tok = evaluate_ppl(
            forward_fn, eval_tokens, lw, embed, final_norm, lm_head,
            arch, skip_mask, dtype=DTYPE,
        )
        delta = ((ppl / baseline_ppl) - 1.0) * 100.0
        log.info(f"  PPL={ppl:.4f}  Δ={delta:+.2f}%")
        results[f"sleb_calfree_n{n}"] = {
            "layers_removed": list(layers),
            "n_removed": n,
            "ppl": round(float(ppl), 4),
            "delta_ppl_pct": round(float(delta), 2),
            "n_tokens": n_tok,
        }

    results["_meta"] = {
        "model": MODEL_NAME,
        "bootstrap_len": BOOTSTRAP_LEN,
        "bootstrap_first16": bootstrap_tokens[:16],
        "bootstrap_protocol": "greedy decode from BOS, full model, no held-out calibration corpus",
        "evaluator": eval_label,
        "max_words": int(os.environ.get("EVAL_MAX_WORDS", "5000")),
        "window": WINDOW,
        "stride": 256,
        "dtype": "bfloat16",
        "n_layers": n_layers,
        "timestamp": time.strftime("%Y-%m-%dT%H-%M-%S"),
        "total_time_s": round(time.time() - t0, 1),
    }

    out_path = os.path.join(REPORT_DIR, "qwen3_8b_sleb_calfree.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
