#!/usr/bin/env python3
"""
Bootstrap confidence intervals for Qwen3-8B and Llama-3.1-8B matched-evaluator results.

This script is a direct rebase of matched_eval_qwen3.py + matched_eval_llama.py:
- IDENTICAL weight loading (safetensors, no PyTorch)
- IDENTICAL forward pass (rms_norm, QK-norm, RoPE, skip-mask scan)
- IDENTICAL window/stride/scoring protocol (512/256, 5K-word WikiText-2 test)
- Adds bootstrap CI over per-window NLLs (N_BOOTSTRAP resamples)

Produces:
  /tmp/bootstrap_8b/qwen3_8b_ci.json
  /tmp/bootstrap_8b/llama_8b_ci.json

Target: TPU v6e-8+, JAX bf16
Expected runtime: ~90-120 minutes total (both models, 1000 bootstrap resamples)
"""

import os, sys, json, time, logging, gc, glob
import numpy as np
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import jax
import jax.numpy as jnp
from jax import lax
from safetensors import safe_open
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DTYPE = jnp.bfloat16
REPORT_DIR = os.environ.get("REPORT_DIR", "/tmp/bootstrap_8b")
Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)

MAX_WORDS = 5000
WINDOW = 512
STRIDE = 256
N_BOOTSTRAP = 1000


# ====================================================================
# Model components — identical to matched_eval_qwen3.py / matched_eval_llama.py
# ====================================================================

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
    """Build JAX forward pass with skip-mask scan — identical to matched_eval scripts."""
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

        cos_s = cos[:S, :]
        sin_s = sin[:S, :]
        cos_b = cos_s[None, None, :, :]
        sin_b = sin_s[None, None, :, :]
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

        residual = hidden
        hidden = rms_norm(hidden, lw_slice["post_ln"], eps)
        gate = jnp.dot(hidden, lw_slice["gate_proj"].T)
        up = jnp.dot(hidden, lw_slice["up_proj"].T)
        hidden = jax.nn.silu(gate.astype(jnp.float32)).astype(hidden.dtype) * up
        hidden = jnp.dot(hidden, lw_slice["down_proj"].T)

        return residual + hidden

    @jax.jit
    def forward(input_ids, layer_weights, embed, final_norm, lm_head,
                cos, sin, skip_mask):
        hidden = embed[input_ids]

        def scan_body(hidden, scan_input):
            idx, should_skip = scan_input
            lw_slice = jax.tree.map(lambda w: w[idx], layer_weights)
            new_hidden = one_layer(hidden, lw_slice, cos, sin)
            return jnp.where(should_skip, hidden, new_hidden), None

        indices = jnp.arange(n_layers, dtype=jnp.int32)
        hidden, _ = lax.scan(scan_body, hidden, (indices, skip_mask))

        hidden = rms_norm(hidden, final_norm, eps)
        logits = jnp.dot(hidden, lm_head.T)
        return logits

    return forward


# ====================================================================
# Weight loading — identical to matched_eval scripts (safetensors)
# ====================================================================

def load_and_stack(model_name, dtype=jnp.bfloat16, hf_tok=None):
    """Load safetensors -> stack per-layer weights for lax.scan."""
    config = AutoConfig.from_pretrained(model_name, token=hf_tok)
    n_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv = getattr(config, "num_key_value_heads", n_heads)
    head_dim = getattr(config, "head_dim", hidden_size // n_heads)
    inter = config.intermediate_size

    rp = getattr(config, "rope_parameters", None) or {}
    rope_theta = rp.get("rope_theta", getattr(config, "rope_theta", 10000.0))
    eps = getattr(config, "rms_norm_eps", 1e-6)

    log.info(f"Model: {model_name}")
    log.info(f"  {n_layers}L, h={hidden_size}, heads={n_heads}/{n_kv}, inter={inter}")
    log.info(f"  rope_theta={rope_theta}, eps={eps}")

    repo = snapshot_download(model_name, allow_patterns=["*.safetensors", "*.json"], token=hf_tok)

    shards = sorted(glob.glob(os.path.join(repo, "model*.safetensors")))

    idx_path = os.path.join(repo, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        weight_map = None

    shard_handles = {}
    for sf in shards:
        name = os.path.basename(sf)
        shard_handles[name] = safe_open(sf, framework="numpy")

    all_keys = set()
    for h in shard_handles.values():
        all_keys.update(h.keys())

    def get(key):
        if weight_map:
            shard_name = weight_map[key]
        else:
            shard_name = os.path.basename(shards[0])
        return shard_handles[shard_name].get_tensor(key)

    has_qk_norm = "model.layers.0.self_attn.q_norm.weight" in all_keys
    log.info(f"  has_qk_norm={has_qk_norm}")

    use_bf16 = (dtype == jnp.bfloat16)

    def stack_weight(template):
        arrs = []
        for i in range(n_layers):
            arr = get(template.format(i=i))
            if use_bf16:
                arr = arr.astype(np.float32)
            arrs.append(arr)
        stacked = np.stack(arrs)
        result = jax.device_put(jnp.array(stacked, dtype=dtype))
        del arrs, stacked
        gc.collect()
        return result

    def single_weight(key):
        arr = get(key)
        if use_bf16:
            arr = arr.astype(np.float32)
        return jax.device_put(jnp.array(arr, dtype=dtype))

    log.info("Stacking layer weights...")
    lw = {}
    for name, template in [
        ("q_proj",    "model.layers.{i}.self_attn.q_proj.weight"),
        ("k_proj",    "model.layers.{i}.self_attn.k_proj.weight"),
        ("v_proj",    "model.layers.{i}.self_attn.v_proj.weight"),
        ("o_proj",    "model.layers.{i}.self_attn.o_proj.weight"),
        ("gate_proj", "model.layers.{i}.mlp.gate_proj.weight"),
        ("up_proj",   "model.layers.{i}.mlp.up_proj.weight"),
        ("down_proj", "model.layers.{i}.mlp.down_proj.weight"),
        ("input_ln",  "model.layers.{i}.input_layernorm.weight"),
        ("post_ln",   "model.layers.{i}.post_attention_layernorm.weight"),
    ]:
        log.info(f"    {name}...")
        lw[name] = stack_weight(template)

    if has_qk_norm:
        lw["q_norm"] = stack_weight("model.layers.{i}.self_attn.q_norm.weight")
        lw["k_norm"] = stack_weight("model.layers.{i}.self_attn.k_norm.weight")

    embed = single_weight("model.embed_tokens.weight")
    final_norm = single_weight("model.norm.weight")
    if "lm_head.weight" in all_keys:
        lm_head = single_weight("lm_head.weight")
    else:
        lm_head = embed

    shard_handles.clear()
    gc.collect()
    log.info("Weights loaded.")

    arch = {
        "n_layers": n_layers,
        "n_heads": n_heads, "n_kv": n_kv, "head_dim": head_dim, "inter": inter,
        "rope_theta": float(rope_theta), "eps": eps, "has_qk_norm": has_qk_norm,
    }
    return arch, lw, embed, final_norm, lm_head


# ====================================================================
# WikiText-2 loading
# ====================================================================

def load_wikitext2(tokenizer):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    words = text.split()
    if len(words) > MAX_WORDS:
        text = " ".join(words[:MAX_WORDS])
        log.info(f"WikiText-2 test: truncated to {MAX_WORDS} words")
    else:
        log.info(f"WikiText-2 test: {len(words)} words")
    tokens = tokenizer.encode(text)
    log.info(f"WikiText-2 tokenized: {len(tokens)} tokens")
    return tokens


# ====================================================================
# Per-window NLL collection — exact window/stride/scoring as matched_eval
# ====================================================================

def collect_window_nlls(forward_fn, tokens, layer_weights, embed, final_norm, lm_head,
                        arch, skip_mask, dtype=jnp.bfloat16):
    """
    Per-window NLL collection.
    Protocol IDENTICAL to evaluate_ppl() in matched_eval scripts:
      - window=512, stride=256, score only non-overlapping region
      - pad shorter final chunk to WINDOW to avoid JIT recompilation
    Returns:
      window_nlls: float array of shape (n_windows,)
      window_counts: int array of shape (n_windows,) -- tokens scored per window
    """
    seq_len = len(tokens)
    window_nlls = []
    window_counts = []
    prev_end = 0

    cos, sin = precompute_rope(WINDOW, arch["head_dim"], arch["rope_theta"], dtype)
    cos, sin = jax.device_put(cos), jax.device_put(sin)

    for begin in range(0, seq_len, STRIDE):
        end = min(begin + WINDOW, seq_len)
        target_len = end - prev_end  # non-overlapping tokens to score

        chunk = tokens[begin:end]
        actual_len = len(chunk)

        # Pad to WINDOW if shorter (avoid JIT recompilation -- identical to matched_eval)
        if actual_len < WINDOW:
            chunk = chunk + [0] * (WINDOW - actual_len)

        input_ids = jnp.array([chunk], dtype=jnp.int32)

        logits = forward_fn(input_ids, layer_weights, embed, final_norm, lm_head,
                            cos, sin, skip_mask)

        # Only use logits up to actual_len -- next-token prediction shift
        shift_logits = logits[0, :actual_len - 1, :]
        shift_targets = jnp.array(tokens[begin + 1:begin + actual_len], dtype=jnp.int32)

        # Cross-entropy in float32
        log_probs = jax.nn.log_softmax(shift_logits.astype(jnp.float32), axis=-1)
        ce = -log_probs[jnp.arange(len(shift_targets)), shift_targets]

        # Only score the non-overlapping region (rightmost target_len - 1 tokens)
        if target_len < actual_len:
            score_start = actual_len - target_len
        else:
            score_start = 0  # first window: score everything

        scored_ce = ce[score_start:]
        n_scored = len(scored_ce)

        if n_scored > 0:
            window_nlls.append(float(jnp.sum(scored_ce)))
            window_counts.append(n_scored)

        prev_end = end
        if end == seq_len:
            break

    return np.array(window_nlls, dtype=np.float64), np.array(window_counts, dtype=np.int64)


# ====================================================================
# Bootstrap CI computation
# ====================================================================

def bootstrap_ppl_ci(window_nlls, window_counts, n_bootstrap=1000, seed=42):
    """Compute PPL point estimate and bootstrap 95% CI from per-window NLLs."""
    ppl = float(np.exp(window_nlls.sum() / window_counts.sum()))

    rng = np.random.RandomState(seed)
    n = len(window_nlls)
    bootstrap_ppls = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        bootstrap_ppls.append(float(np.exp(window_nlls[idx].sum() / window_counts[idx].sum())))

    bootstrap_ppls = np.array(bootstrap_ppls)
    return {
        "ppl": round(ppl, 4),
        "ci_95_lower": round(float(np.percentile(bootstrap_ppls, 2.5)), 4),
        "ci_95_upper": round(float(np.percentile(bootstrap_ppls, 97.5)), 4),
        "ci_std": round(float(np.std(bootstrap_ppls)), 4),
        "ci_mean": round(float(np.mean(bootstrap_ppls)), 4),
        "n_windows": n,
        "n_tokens": int(window_counts.sum()),
    }


# ====================================================================
# Configs -- from matched_eval_results.json and matched_eval_llama_results.json
# ====================================================================

QWEN_CONFIGS = {
    "baseline": [],
    # Interchange-guided
    "interchange_n1": [17],
    "interchange_clustered_n3": [15, 17, 20],
    "interchange_distributed_n3": [17, 21, 26],
    "interchange_clustered_n5": [15, 17, 18, 19, 20],
    "interchange_distributed_n5": [17, 21, 26, 28, 30],
    # Replacement-guided
    "replacement_n1": [32],
    "replacement_n3": [28, 31, 32],
    "replacement_n5": [25, 28, 30, 31, 32],
    # BI-guided
    "bi_n1": [17],
    "bi_n3": [7, 11, 17],
    "bi_n5": [7, 8, 11, 15, 17],
    "bi_distributed_n5": [7, 12, 17, 22, 27],
    # SLEB
    "sleb_n1": [17],
    "sleb_n3": [17, 18, 19],
    "sleb_n5": [17, 18, 19, 20, 21],
    # CKA
    "cka_n1": [7],
    "cka_n2": [7, 9],
    "cka_n3": [7, 9, 17],
    # Random
    "random_n1": [10],
    "random_n3": [10, 20, 25],
    "non_bisimilar_n1": [6],
}

LLAMA_CONFIGS = {
    "baseline": [],
    # Interchange-guided
    "interchange_n1": [25],
    "interchange_n3": [24, 25, 26],
    "interchange_n5": [22, 23, 24, 25, 26],
    # Replacement-guided
    "replacement_n1": [11],
    "replacement_n3": [9, 10, 11],
    "replacement_n5": [7, 8, 9, 10, 11],
    # BI-guided
    "bi_n1": [2],
    "bi_n3": [2, 3, 5],
    "bi_n5": [2, 3, 4, 5, 6],
    # SLEB
    "sleb_greedy_n1": [11],
    "sleb_iterative_n1": [11],
    "sleb_greedy_n3": [10, 11, 25],
    "sleb_iterative_n3": [10, 11, 25],
    "sleb_greedy_n5": [8, 9, 10, 11, 25],
    "sleb_iterative_n5": [10, 11, 12, 25, 28],
    # Random
    "random_n1": [3],
    "random_n3": [14, 19, 22],
    "random_n5": [3, 6, 16, 19, 29],
}


# ====================================================================
# Main evaluation loop
# ====================================================================

def run_model_bootstrap(model_name, configs, out_file, hf_tok=None):
    log.info(f"\n{'='*70}")
    log.info(f"Bootstrap CIs: {model_name}")
    log.info(f"{'='*70}")

    arch, layer_weights, embed, final_norm, lm_head = load_and_stack(
        model_name, dtype=DTYPE, hf_tok=hf_tok
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_tok)
    tokens = load_wikitext2(tokenizer)

    forward_fn = build_forward(arch)

    n_layers = arch["n_layers"]
    results = {}

    # Sanity check: run baseline first and validate against threshold
    log.info("Running baseline sanity check...")
    skip_mask_none = jnp.zeros(n_layers, dtype=jnp.bool_)
    nlls, counts = collect_window_nlls(
        forward_fn, tokens, layer_weights, embed, final_norm, lm_head,
        arch, skip_mask_none, DTYPE
    )
    baseline_ci = bootstrap_ppl_ci(nlls, counts, N_BOOTSTRAP)
    baseline_ppl = baseline_ci["ppl"]
    log.info(f"  Baseline PPL: {baseline_ppl:.4f} "
             f"CI=[{baseline_ci['ci_95_lower']:.2f}, {baseline_ci['ci_95_upper']:.2f}]")

    if baseline_ppl > 50.0:
        log.error(f"SANITY FAIL: baseline PPL {baseline_ppl:.2f} >> expected <20. Aborting.")
        results["_sanity_failure"] = {
            "baseline_ppl": baseline_ppl,
            "error": "Baseline PPL too high -- forward pass has a bug",
        }
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        return results

    log.info(f"Sanity check PASSED. Baseline PPL {baseline_ppl:.4f}")

    results["baseline"] = {
        "layers_removed": [],
        "n_removed": 0,
        "ppl": baseline_ci["ppl"],
        "delta_ppl_pct": 0.0,
        "ci_95_lower": baseline_ci["ci_95_lower"],
        "ci_95_upper": baseline_ci["ci_95_upper"],
        "ci_std": baseline_ci["ci_std"],
        "n_windows": baseline_ci["n_windows"],
        "n_tokens": baseline_ci["n_tokens"],
    }

    for config_name, skip_layers in configs.items():
        if config_name == "baseline":
            continue  # already done above

        log.info(f"\n  Config: {config_name}, skip={skip_layers}")

        skip_mask = jnp.zeros(n_layers, dtype=jnp.bool_)
        if skip_layers:
            skip_mask = skip_mask.at[jnp.array(skip_layers)].set(True)

        t0 = time.time()
        nlls, counts = collect_window_nlls(
            forward_fn, tokens, layer_weights, embed, final_norm, lm_head,
            arch, skip_mask, DTYPE
        )
        ci = bootstrap_ppl_ci(nlls, counts, N_BOOTSTRAP)
        elapsed = time.time() - t0

        delta_pct = (ci["ppl"] - baseline_ppl) / baseline_ppl * 100

        result = {
            "layers_removed": skip_layers,
            "n_removed": len(skip_layers),
            "ppl": ci["ppl"],
            "delta_ppl_pct": round(delta_pct, 2),
            "ci_95_lower": ci["ci_95_lower"],
            "ci_95_upper": ci["ci_95_upper"],
            "ci_std": ci["ci_std"],
            "n_windows": ci["n_windows"],
            "n_tokens": ci["n_tokens"],
            "elapsed_s": round(elapsed, 1),
        }
        results[config_name] = result

        log.info(f"    PPL={ci['ppl']:.4f} (delta={delta_pct:+.2f}%) "
                 f"CI=[{ci['ci_95_lower']:.2f}, {ci['ci_95_upper']:.2f}] "
                 f"{elapsed:.1f}s")

        # Checkpoint: save after every config in case of preemption
        results["_meta"] = {
            "model": model_name,
            "evaluator": "wikitext2_test_5Kword_window512_stride256",
            "n_bootstrap": N_BOOTSTRAP,
            "dtype": "bfloat16",
            "device": str(jax.devices()[0]),
            "n_devices": jax.device_count(),
            "jax_version": jax.__version__,
        }
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

    log.info(f"\n  Saved {len([k for k in results if not k.startswith('_')])} configs to {out_file}")
    return results


def main():
    log.info(f"Devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
    log.info(f"JAX version: {jax.__version__}")

    # HF token for gated Llama
    hf_tok = os.environ.get("HF_TOKEN") or (
        open("/tmp/hf_token").read().strip()
        if os.path.exists("/tmp/hf_token") else None
    )
    if hf_tok:
        log.info("HF_TOKEN found.")
    else:
        log.warning("No HF_TOKEN found -- Llama-3.1-8B will be skipped.")

    os.makedirs(REPORT_DIR, exist_ok=True)

    # Run Qwen3-8B bootstrap
    qwen_out = os.path.join(REPORT_DIR, "qwen3_8b_ci.json")
    run_model_bootstrap("Qwen/Qwen3-8B", QWEN_CONFIGS, qwen_out)

    gc.collect()
    jax.clear_caches()

    if hf_tok:
        llama_out = os.path.join(REPORT_DIR, "llama_8b_ci.json")
        run_model_bootstrap("meta-llama/Llama-3.1-8B", LLAMA_CONFIGS, llama_out, hf_tok=hf_tok)
    else:
        log.error("Skipping Llama-3.1-8B: no HF_TOKEN.")
        log.info("To run Llama: set HF_TOKEN env var or write token to /tmp/hf_token, then rerun.")

    log.info("\n=== Bootstrap CI computation complete. ===")


if __name__ == "__main__":
    main()
