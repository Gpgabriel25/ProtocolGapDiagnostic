#!/usr/bin/env python3
"""
Clean-Oracle Bootstrap CIs for Beam-Bisim vs SLEB-Iterative on Qwen3-8B
=========================================================================
Addresses the primary outstanding reviewer concern (7-8/10 P83-P84 panels):
  "beam-bisim clean-oracle (WikiText-2 train) result is only a point estimate"

Protocol:
  oracle_tokens = WikiText-2 TRAIN (used by beam search + SLEB for layer scoring)
  eval_tokens   = WikiText-2 TEST  (used only for final PPL + bootstrap CIs)

This separates the oracle from the evaluation, removing test-set contamination.
The point estimate for this run (+23.5% beam-bisim vs +46.2% SLEB) is already
in the paper (Table tab:calfree_h2h footnote). This script adds the CIs.

Expected output: /tmp/clean_oracle_ci/qwen3_clean_oracle_ci.json
Expected runtime: ~45-90 min on TPU v6e-8
"""

import os, sys, json, time, logging, gc
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.93")

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from safetensors import safe_open
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-8B"
DTYPE = jnp.bfloat16
REPORT_DIR = os.environ.get("REPORT_DIR", "/tmp/clean_oracle_ci")
Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = os.path.join(REPORT_DIR, "qwen3_clean_oracle_ci.json")

MAX_ORACLE_WORDS = 5000   # TRAIN words for beam search oracle
MAX_EVAL_WORDS = 5000     # TEST words for final bootstrap CI evaluation
WINDOW = 512
STRIDE = 256
BEAM_WIDTH = 3
SEED_K = 12
N_BOOTSTRAP = 2000

# Interchange KL scores from 500-prompt TPU run (from paper Table tab:skip_qwen)
# Top-SEED_K layers by interchange KL min-neighbor score
INTERCHANGE_SCORES = {
    0: 0.0015, 1: 0.0041, 2: 0.0064, 3: 0.0108, 4: 0.0086,
    5: 0.0119, 6: 0.0073, 7: 0.0071, 8: 0.0072, 9: 0.0074,
    10: 0.0046, 11: 0.0040, 12: 0.0038, 13: 0.0042, 14: 0.0046,
    15: 0.0038, 16: 0.0015, 17: 0.0011, 18: 0.0014, 19: 0.0020,
    20: 0.0018, 21: 0.0019, 22: 0.0022, 23: 0.0027, 24: 0.0028,
    25: 0.0042, 26: 0.0058, 27: 0.0063, 28: 0.0074, 29: 0.0069,
    30: 0.0802, 31: 0.0842, 32: 0.0928, 33: 0.0928, 34: 0.1985,
    35: 7.5975,
}


# ── Model components (identical to qwen3_beam_search.py) ──────────────────────
def rms_norm(x, w, eps=1e-6):
    xf = x.astype(jnp.float32)
    norm = xf * lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (norm * w.astype(jnp.float32)).astype(x.dtype)


def apply_rope(q, k, cos, sin):
    def rotate(x, c, s):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).astype(x.dtype)
    return rotate(q, cos, sin), rotate(k, cos, sin)


def precompute_rope(seq_len, head_dim, theta, dtype):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(t, freqs)
    cos = jnp.cos(angles).astype(dtype)
    sin = jnp.sin(angles).astype(dtype)
    return cos, sin


def build_forward(arch):
    n_layers = arch["n_layers"]
    n_heads = arch["n_heads"]
    n_kv = arch["n_kv"]
    head_dim = arch["head_dim"]
    eps = arch["eps"]
    has_qk = arch["has_qk_norm"]
    kv_rep = n_heads // n_kv

    def one_layer(hidden, lw_slice, cos, sin):
        B, S, H = hidden.shape
        residual = hidden
        hidden = rms_norm(hidden, lw_slice["input_ln"], eps)
        q = jnp.dot(hidden, lw_slice["q_proj"].T)
        k = jnp.dot(hidden, lw_slice["k_proj"].T)
        v = jnp.dot(hidden, lw_slice["v_proj"].T)
        q = q.reshape(B, S, n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, n_kv, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, n_kv, head_dim).transpose(0, 2, 1, 3)
        if has_qk:
            q = rms_norm(q, lw_slice["q_norm"], eps)
            k = rms_norm(k, lw_slice["k_norm"], eps)
        cos_s = cos[:S, :]; sin_s = sin[:S, :]
        cos_b = cos_s[None, None, :, :]; sin_b = sin_s[None, None, :, :]
        q, k = apply_rope(q, k, cos_b, sin_b)
        if kv_rep > 1:
            k = jnp.repeat(k, kv_rep, axis=1)
            v = jnp.repeat(v, kv_rep, axis=1)
        scale = head_dim ** -0.5
        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        neg_inf = jnp.finfo(hidden.dtype).min
        attn = jnp.where(mask[None, None], attn, neg_inf)
        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(hidden.dtype)
        out = jnp.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, H)
        out = jnp.dot(out, lw_slice["o_proj"].T)
        hidden = residual + out
        residual2 = hidden
        hidden = rms_norm(hidden, lw_slice["post_ln"], eps)
        gate = jnp.dot(hidden, lw_slice["gate_proj"].T)
        up = jnp.dot(hidden, lw_slice["up_proj"].T)
        hidden = jax.nn.silu(gate.astype(jnp.float32)).astype(hidden.dtype) * up
        hidden = jnp.dot(hidden, lw_slice["down_proj"].T)
        hidden = residual2 + hidden
        return hidden

    def forward(input_ids, layer_weights, embed, final_norm, lm_head, cos, sin, skip_mask):
        B, S = input_ids.shape
        hidden = embed[input_ids]

        def scan_body(hidden, scan_input):
            lw_slice, skip = scan_input
            new_hidden = one_layer(hidden, lw_slice, cos, sin)
            hidden = jnp.where(skip, hidden, new_hidden)
            return hidden, None

        hidden, _ = lax.scan(scan_body, hidden, (layer_weights, skip_mask))
        hidden = rms_norm(hidden, final_norm, 1e-6)
        logits = jnp.dot(hidden, lm_head.T)
        return logits

    return jax.jit(forward), None


# ── Weight loading (identical to qwen3_beam_search.py) ────────────────────────
def load_and_stack(model_name, dtype=jnp.bfloat16, hf_tok=None):
    log.info(f"Downloading {model_name}...")
    path = snapshot_download(model_name, token=hf_tok, ignore_patterns=["*.msgpack", "*.h5"])
    config = AutoConfig.from_pretrained(model_name)
    arch = {
        "n_layers": config.num_hidden_layers,
        "n_heads": config.num_attention_heads,
        "n_kv": config.num_key_value_heads,
        "head_dim": config.hidden_size // config.num_attention_heads,
        "hidden": config.hidden_size,
        "intermediate": config.intermediate_size,
        "eps": config.rms_norm_eps,
        "rope_theta": getattr(config, "rope_theta", 10000.0),
        "vocab_size": config.vocab_size,
        "has_qk_norm": getattr(config, "use_sliding_window", False) or (config.model_type == "qwen3"),
    }
    n_layers = arch["n_layers"]
    log.info(f"Architecture: {arch}")

    shards = sorted(Path(path).glob("model-*.safetensors"))
    if not shards:
        shards = sorted(Path(path).glob("*.safetensors"))
    log.info(f"Loading {len(shards)} shards...")
    raw = {}
    for shard in shards:
        with safe_open(str(shard), framework="numpy") as f:
            for key in f.keys():
                raw[key] = f.get_tensor(key)

    def get(key):
        tensor = raw[key]
        # Convert to float32 first (works for any input dtype including bf16 stored as uint16)
        if tensor.dtype != np.float32:
            tensor = tensor.astype(np.float32)
        return jnp.array(tensor, dtype=dtype)

    def stack_weight(template):
        return jnp.stack([get(template.format(i=i)) for i in range(n_layers)])

    embed = get("model.embed_tokens.weight")
    final_norm = get("model.norm.weight")
    lm_head = get("lm_head.weight") if "lm_head.weight" in raw else embed

    layer_weights = {
        "input_ln":  stack_weight("model.layers.{i}.input_layernorm.weight"),
        "q_proj":    stack_weight("model.layers.{i}.self_attn.q_proj.weight"),
        "k_proj":    stack_weight("model.layers.{i}.self_attn.k_proj.weight"),
        "v_proj":    stack_weight("model.layers.{i}.self_attn.v_proj.weight"),
        "o_proj":    stack_weight("model.layers.{i}.self_attn.o_proj.weight"),
        "post_ln":   stack_weight("model.layers.{i}.post_attention_layernorm.weight"),
        "gate_proj": stack_weight("model.layers.{i}.mlp.gate_proj.weight"),
        "up_proj":   stack_weight("model.layers.{i}.mlp.up_proj.weight"),
        "down_proj": stack_weight("model.layers.{i}.mlp.down_proj.weight"),
        "q_norm": stack_weight("model.layers.{i}.self_attn.q_norm.weight") if arch["has_qk_norm"] else None,
        "k_norm": stack_weight("model.layers.{i}.self_attn.k_norm.weight") if arch["has_qk_norm"] else None,
    }
    del raw
    gc.collect()
    return arch, layer_weights, embed, final_norm, lm_head


# ── Data loading ──────────────────────────────────────────────────────────────
def load_wiki_tokens(tokenizer, split="test", max_words=5000):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join([t for t in ds["text"] if isinstance(t, str) and t.strip()])
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    tokens = tokenizer.encode(text)
    log.info(f"WikiText-2 {split}: {len(tokens)} tokens ({min(len(words), max_words)} words)")
    return tokens


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_ppl_and_windows(forward_fn, tokens, lw, embed, final_norm, lm_head,
                              arch, skip_mask, cos, sin, dtype=jnp.bfloat16):
    """Returns (ppl, n_tokens, window_nlls, window_counts)."""
    seq_len = len(tokens)
    window_nlls = []
    window_counts = []
    prev_end = 0
    for begin in range(0, seq_len, STRIDE):
        end = min(begin + WINDOW, seq_len)
        target_len = end - prev_end
        chunk = tokens[begin:end]
        actual_len = len(chunk)
        if actual_len < WINDOW:
            chunk = chunk + [0] * (WINDOW - actual_len)
        input_ids = jnp.array([chunk], dtype=jnp.int32)
        logits = forward_fn(input_ids, lw, embed, final_norm, lm_head, cos, sin, skip_mask)
        shift_logits = logits[0, :actual_len - 1, :]
        shift_targets = jnp.array(tokens[begin + 1:begin + actual_len], dtype=jnp.int32)
        log_probs = jax.nn.log_softmax(shift_logits.astype(jnp.float32), axis=-1)
        ce = -log_probs[jnp.arange(len(shift_targets)), shift_targets]
        if target_len < actual_len:
            score_start = actual_len - target_len
        else:
            score_start = 0
        scored = ce[score_start:]
        window_nlls.append(float(jnp.mean(scored)))
        window_counts.append(len(scored))
        prev_end = end
        if end >= seq_len:
            break
    nlls = np.array(window_nlls)
    cnts = np.array(window_counts)
    total = np.sum(cnts)
    ppl = float(np.exp(np.sum(nlls * cnts) / total))
    return ppl, int(total), nlls, cnts


def quick_ppl(forward_fn, tokens, lw, embed, final_norm, lm_head, arch, skip_mask, cos, sin):
    """Fast PPL-only for beam search oracle (no window storage)."""
    ppl, _, _, _ = evaluate_ppl_and_windows(
        forward_fn, tokens, lw, embed, final_norm, lm_head, arch, skip_mask, cos, sin)
    return ppl


# ── Beam search with custom oracle tokens ─────────────────────────────────────
def beam_search_with_oracle(forward_fn, oracle_tokens, lw, embed, final_norm, lm_head,
                             arch, scores, n_layers, cos, sin,
                             beam_width=3, max_n=5, seed_k=12):
    """Run beam search scoring against oracle_tokens, not eval_tokens."""
    sorted_layers = sorted(scores, key=lambda k: scores[k])[:seed_k]
    log.info(f"Beam search seed layers (top-{seed_k} by interchange KL): {sorted_layers}")

    beams = [([], None)]  # (layers_removed, cached_ppl)
    best_by_n = {}
    total_evals = 0

    for step in range(max_n):
        new_beams_with_score = []
        for current_layers, _ in beams:
            candidates = [l for l in range(n_layers) if l not in current_layers]
            if step == 0:
                candidates = sorted_layers
            for cand in candidates:
                proposed = sorted(current_layers + [cand])
                skip_mask = jnp.array(
                    [i in proposed for i in range(n_layers)], dtype=jnp.bool_)
                ppl = quick_ppl(forward_fn, oracle_tokens, lw, embed, final_norm, lm_head,
                                arch, skip_mask, cos, sin)
                total_evals += 1
                new_beams_with_score.append((proposed, ppl))

        new_beams_with_score.sort(key=lambda x: x[1])
        beams = [(layers, ppl) for layers, ppl in new_beams_with_score[:beam_width]]
        best_layers, best_ppl = beams[0]
        best_by_n[step + 1] = (best_layers, best_ppl)
        log.info(f"n={step+1}: best={sorted(best_layers)} oracle_ppl={best_ppl:.4f} "
                 f"({total_evals} evals so far)")

    return best_by_n, total_evals


# ── SLEB-iterative with oracle tokens ─────────────────────────────────────────
def sleb_iterative_with_oracle(forward_fn, oracle_tokens, lw, embed, final_norm, lm_head,
                                arch, n_layers, cos, sin, max_n=5):
    """SLEB-iterative using oracle tokens for scoring."""
    removed = []
    results = {}
    for step in range(max_n):
        best_ppl = float("inf")
        best_layer = None
        for l in range(n_layers):
            if l in removed:
                continue
            proposed = sorted(removed + [l])
            skip_mask = jnp.array([i in proposed for i in range(n_layers)], dtype=jnp.bool_)
            ppl = quick_ppl(forward_fn, oracle_tokens, lw, embed, final_norm, lm_head,
                            arch, skip_mask, cos, sin)
            if ppl < best_ppl:
                best_ppl = ppl
                best_layer = l
        removed = sorted(removed + [best_layer])
        results[step + 1] = (removed[:], best_ppl)
        log.info(f"SLEB n={step+1}: removed={removed} oracle_ppl={best_ppl:.4f}")
    return results


# ── Bootstrap CI ──────────────────────────────────────────────────────────────
def bootstrap_paired_ci(baseline_nlls, pruned_nlls, counts, n_bootstrap=2000, seed=42):
    """Block bootstrap for paired delta_ppl_pct CI."""
    rng = np.random.default_rng(seed)
    n_windows = len(baseline_nlls)
    base_ppl = float(np.exp(np.sum(baseline_nlls * counts) / np.sum(counts)))
    pruned_ppl = float(np.exp(np.sum(pruned_nlls * counts) / np.sum(counts)))
    delta_pct = (pruned_ppl / base_ppl - 1.0) * 100.0

    boot_deltas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_windows, size=n_windows)
        b_cnt = counts[idx]
        b_base = np.exp(np.sum(baseline_nlls[idx] * b_cnt) / np.sum(b_cnt))
        b_prun = np.exp(np.sum(pruned_nlls[idx] * b_cnt) / np.sum(b_cnt))
        boot_deltas.append((b_prun / b_base - 1.0) * 100.0)

    boot_deltas = np.array(boot_deltas)
    ci_low = float(np.percentile(boot_deltas, 2.5))
    ci_high = float(np.percentile(boot_deltas, 97.5))
    return base_ppl, pruned_ppl, delta_pct, ci_low, ci_high


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 70)
    log.info("Clean-Oracle Bootstrap CI: Qwen3-8B beam-bisim vs SLEB-iterative")
    log.info("Oracle: WikiText-2 TRAIN  |  Evaluation: WikiText-2 TEST")
    log.info("=" * 70)
    log.info(f"Devices: {jax.devices()}")

    t0 = time.time()

    # Load model
    arch, lw, embed, final_norm, lm_head = load_and_stack(MODEL_NAME, DTYPE)
    n_layers = arch["n_layers"]
    log.info(f"Model loaded in {time.time()-t0:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    oracle_tokens = load_wiki_tokens(tokenizer, split="train",  max_words=MAX_ORACLE_WORDS)
    eval_tokens   = load_wiki_tokens(tokenizer, split="test",   max_words=MAX_EVAL_WORDS)

    # JIT compile
    forward_fn, _ = build_forward(arch)
    cos_w, sin_w = precompute_rope(WINDOW, arch["head_dim"], arch["rope_theta"], DTYPE)
    cos_w = jax.device_put(cos_w)
    sin_w = jax.device_put(sin_w)
    dummy_mask = jnp.zeros(n_layers, dtype=jnp.bool_)
    dummy_ids  = jnp.zeros((1, WINDOW), dtype=jnp.int32)
    warmup = forward_fn(dummy_ids, lw, embed, final_norm, lm_head, cos_w, sin_w, dummy_mask)
    jax.block_until_ready(warmup)
    log.info(f"JIT compiled. ({time.time()-t0:.1f}s)")

    # Baseline evaluation on TEST
    baseline_ppl, n_eval_tokens, baseline_nlls, baseline_counts = evaluate_ppl_and_windows(
        forward_fn, eval_tokens, lw, embed, final_norm, lm_head,
        arch, dummy_mask, cos_w, sin_w)
    log.info(f"Baseline PPL (TEST) = {baseline_ppl:.4f}  n_windows={len(baseline_nlls)}")

    # Run beam-bisim with TRAIN oracle
    log.info("\n--- Beam-bisim (TRAIN oracle) ---")
    t_bs = time.time()
    beam_by_n, beam_evals = beam_search_with_oracle(
        forward_fn, oracle_tokens, lw, embed, final_norm, lm_head,
        arch, INTERCHANGE_SCORES, n_layers, cos_w, sin_w,
        beam_width=BEAM_WIDTH, max_n=5, seed_k=SEED_K)
    log.info(f"Beam search done in {time.time()-t_bs:.1f}s, {beam_evals} oracle evals")

    # Run SLEB-iterative with TRAIN oracle
    log.info("\n--- SLEB-iterative (TRAIN oracle) ---")
    t_sleb = time.time()
    sleb_by_n = sleb_iterative_with_oracle(
        forward_fn, oracle_tokens, lw, embed, final_norm, lm_head,
        arch, n_layers, cos_w, sin_w, max_n=5)
    log.info(f"SLEB done in {time.time()-t_sleb:.1f}s")

    # Evaluate all selections on TEST with bootstrap CIs
    results = {
        "_meta": {
            "model": MODEL_NAME,
            "oracle": "wikitext-2-raw-v1 train",
            "evaluator": "wikitext-2-raw-v1 test",
            "oracle_max_words": MAX_ORACLE_WORDS,
            "eval_max_words": MAX_EVAL_WORDS,
            "window": WINDOW, "stride": STRIDE,
            "dtype": "bfloat16",
            "n_bootstrap": N_BOOTSTRAP,
            "n_eval_windows": len(baseline_nlls),
            "device": str(jax.devices()[0]),
            "jax_version": jax.__version__,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "baseline": {
            "ppl": round(baseline_ppl, 4),
            "n_tokens": n_eval_tokens,
            "n_windows": len(baseline_nlls),
        }
    }

    log.info("\n--- TEST evaluation + bootstrap CIs ---")
    for n_val in sorted(beam_by_n.keys()):
        layers, oracle_ppl = beam_by_n[n_val]
        skip_mask = jnp.array([i in layers for i in range(n_layers)], dtype=jnp.bool_)
        ppl, _, pruned_nlls, pruned_counts = evaluate_ppl_and_windows(
            forward_fn, eval_tokens, lw, embed, final_norm, lm_head,
            arch, skip_mask, cos_w, sin_w)
        _, _, delta_pct, ci_low, ci_high = bootstrap_paired_ci(
            baseline_nlls, pruned_nlls, baseline_counts, N_BOOTSTRAP)
        key = f"beam_bisim_n{n_val}"
        results[key] = {
            "layers_removed": sorted(layers),
            "oracle_ppl": round(oracle_ppl, 4),
            "ppl": round(ppl, 4),
            "delta_ppl_pct": round(delta_pct, 2),
            "ci_95_low": round(ci_low, 2),
            "ci_95_high": round(ci_high, 2),
            "n_windows": len(pruned_nlls),
        }
        log.info(f"beam_bisim n={n_val}: layers={sorted(layers)} "
                 f"ppl={ppl:.4f} delta={delta_pct:.2f}% CI=[{ci_low:.2f},{ci_high:.2f}]")

    for n_val in sorted(sleb_by_n.keys()):
        layers, oracle_ppl = sleb_by_n[n_val]
        skip_mask = jnp.array([i in layers for i in range(n_layers)], dtype=jnp.bool_)
        ppl, _, pruned_nlls, pruned_counts = evaluate_ppl_and_windows(
            forward_fn, eval_tokens, lw, embed, final_norm, lm_head,
            arch, skip_mask, cos_w, sin_w)
        _, _, delta_pct, ci_low, ci_high = bootstrap_paired_ci(
            baseline_nlls, pruned_nlls, baseline_counts, N_BOOTSTRAP)
        key = f"sleb_iter_n{n_val}"
        results[key] = {
            "layers_removed": sorted(layers),
            "oracle_ppl": round(oracle_ppl, 4),
            "ppl": round(ppl, 4),
            "delta_ppl_pct": round(delta_pct, 2),
            "ci_95_low": round(ci_low, 2),
            "ci_95_high": round(ci_high, 2),
            "n_windows": len(pruned_nlls),
        }
        log.info(f"sleb_iter n={n_val}: layers={sorted(layers)} "
                 f"ppl={ppl:.4f} delta={delta_pct:.2f}% CI=[{ci_low:.2f},{ci_high:.2f}]")

    # Summary
    if "beam_bisim_n5" in results and "sleb_iter_n5" in results:
        bm = results["beam_bisim_n5"]
        sl = results["sleb_iter_n5"]
        overlap = bm["ci_95_high"] > sl["ci_95_low"]
        log.info("\n=== PRIMARY RESULT (n=5, clean oracle) ===")
        log.info(f"Beam-bisim:     {bm['delta_ppl_pct']:.2f}%  CI=[{bm['ci_95_low']:.2f},{bm['ci_95_high']:.2f}]")
        log.info(f"SLEB-iterative: {sl['delta_ppl_pct']:.2f}%  CI=[{sl['ci_95_low']:.2f},{sl['ci_95_high']:.2f}]")
        log.info(f"Non-overlapping: {not overlap}")
        results["_primary_result"] = {
            "n": 5,
            "beam_bisim_delta_pct": bm["delta_ppl_pct"],
            "beam_bisim_ci": [bm["ci_95_low"], bm["ci_95_high"]],
            "sleb_iter_delta_pct": sl["delta_ppl_pct"],
            "sleb_iter_ci": [sl["ci_95_low"], sl["ci_95_high"]],
            "non_overlapping": not overlap,
        }

    results["_meta"]["total_time_s"] = round(time.time() - t0, 1)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved to {OUTPUT_JSON}")
    log.info(f"Total time: {results['_meta']['total_time_s']}s")


if __name__ == "__main__":
    main()
