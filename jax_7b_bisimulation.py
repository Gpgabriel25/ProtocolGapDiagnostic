#!/usr/bin/env python3
"""
Ultra-fast 7B+ bisimulation via JAX JIT + lax.scan.
Targets Qwen3-8B (36 layers) on AMD ROCm iGPU or CPU.

Speed optimizations:
  1. Stacked weights + lax.scan (XLA loop fusion, single compilation)
  2. Index-based layer swap (no weight copy, no recompilation per pair)
  3. fp16 storage / fp32 accumulation (softmax, KL, RMSNorm)
  4. Batched prompts (single forward pass = all prompts at once)
  5. Adjacent pairs only (35 pairs for 36 layers)
  6. Short sequences (32 tokens)
  7. Precomputed RoPE cos/sin tables (outside JIT)
  8. Progressive weight loading (pop-as-you-go to limit peak RAM)
  9. Single baseline computation, reused for all pairs

Usage:
  python jax_7b_bisimulation.py                                # GPU default
  JAX_PLATFORMS=cpu python jax_7b_bisimulation.py              # CPU fallback
  MODEL=meta-llama/Llama-3.1-8B python jax_7b_bisimulation.py # Alt model
"""

import os, sys, json, time, logging, gc
import numpy as np
from pathlib import Path

# ROCm/gfx1150 stability: disable Triton GEMM to avoid miscompilation
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=false")
# Use 95% of GPU memory (default 75% is too tight for 8B model on 18GB iGPU)
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import jax
import jax.numpy as jnp
from jax import lax
from safetensors import safe_open
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────
MODEL_NAME  = os.environ.get("MODEL", "Qwen/Qwen3-8B")
N_PROMPTS   = int(os.environ.get("N_PROMPTS", "5"))
SEQ_LEN     = int(os.environ.get("SEQ_LEN", "32"))
DTYPE       = jnp.float16
REPORT_DIR  = os.environ.get("REPORT_DIR", "reports/2026-03-31T12-24-40")

PROMPTS = [
    "The quick brown fox jumps over the lazy dog and then proceeds to",
    "In quantum mechanics, the wave function describes the probability of",
    "The stock market rally continued as investors grew confident about future",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci",
    "Climate scientists warn that global temperatures could rise significantly by",
    "Machine learning models can be compressed through various techniques including",
    "The ancient Roman Empire at its peak controlled vast territories spanning",
    "In the kitchen, she carefully measured two cups of flour and added",
][:N_PROMPTS]


# ── Weight loading (memory-efficient, lazy from shards) ──────

def load_and_stack(model_name, dtype=jnp.float16):
    """Load safetensors lazily → stack per-layer weights for lax.scan.
    
    Key optimization: never holds all 399 tensors in RAM simultaneously.
    Reads from shard files on demand, one weight type at a time.
    Peak RAM ≈ 2× largest weight type (stacking overhead) ≈ 7GB for 8B model.
    """
    config = AutoConfig.from_pretrained(model_name)
    n_layers = config.num_hidden_layers
    hidden   = config.hidden_size
    n_heads  = config.num_attention_heads
    n_kv     = getattr(config, "num_key_value_heads", n_heads)
    head_dim = getattr(config, "head_dim", hidden // n_heads)
    inter    = config.intermediate_size

    # RoPE theta
    rp = getattr(config, "rope_parameters", None) or {}
    rope_theta = rp.get("rope_theta", getattr(config, "rope_theta", 10000.0))
    eps = getattr(config, "rms_norm_eps", 1e-6)

    log.info(f"Model: {model_name}")
    log.info(f"  {n_layers}L, h={hidden}, heads={n_heads}/{n_kv}, inter={inter}, head_dim={head_dim}")
    log.info(f"  rope_theta={rope_theta}, eps={eps}")

    # Download shards
    log.info("Ensuring model files are downloaded...")
    repo = snapshot_download(model_name, allow_patterns=["*.safetensors", "*.json"])

    import glob
    shards = sorted(glob.glob(os.path.join(repo, "model*.safetensors")))
    log.info(f"  {len(shards)} shards")

    # Load weight map (tells us which shard has which tensor)
    idx_path = os.path.join(repo, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        # Single shard model
        weight_map = None

    # Open shard handles (lazy — data stays on disk until get_tensor)
    shard_handles = {}
    for sf in shards:
        name = os.path.basename(sf)
        shard_handles[name] = safe_open(sf, framework="numpy")

    # List all available keys
    all_keys = set()
    for h in shard_handles.values():
        all_keys.update(h.keys())
    log.info(f"  {len(all_keys)} tensors available across shards")

    def get(key):
        """Read a single tensor from the appropriate shard."""
        if weight_map:
            shard_name = weight_map[key]
        else:
            shard_name = os.path.basename(shards[0])
        return shard_handles[shard_name].get_tensor(key)

    # Detect QK norm
    has_qk_norm = "model.layers.0.self_attn.q_norm.weight" in all_keys
    log.info(f"  QK norm: {has_qk_norm}")

    # Stack one weight type at a time (lazy read → numpy stack → GPU)
    def stack_weight(template):
        arrs = []
        for i in range(n_layers):
            arrs.append(get(template.format(i=i)))
        stacked = np.stack(arrs)
        del arrs
        # Convert to float16 (handles bf16 → fp16)
        if stacked.dtype != np.float16:
            stacked = stacked.astype(np.float16)
        result = jax.device_put(jnp.array(stacked))
        del stacked
        gc.collect()
        return result

    def single_weight(key):
        arr = get(key)
        if arr.dtype != np.float16:
            arr = arr.astype(np.float16)
        return jax.device_put(jnp.array(arr))

    log.info("Stacking layer weights (lazy, one type at a time)...")
    lw = {}
    for name, template in [
        ("q_proj",   "model.layers.{i}.self_attn.q_proj.weight"),
        ("k_proj",   "model.layers.{i}.self_attn.k_proj.weight"),
        ("v_proj",   "model.layers.{i}.self_attn.v_proj.weight"),
        ("o_proj",   "model.layers.{i}.self_attn.o_proj.weight"),
        ("gate_proj","model.layers.{i}.mlp.gate_proj.weight"),
        ("up_proj",  "model.layers.{i}.mlp.up_proj.weight"),
        ("down_proj","model.layers.{i}.mlp.down_proj.weight"),
        ("input_ln", "model.layers.{i}.input_layernorm.weight"),
        ("post_ln",  "model.layers.{i}.post_attention_layernorm.weight"),
    ]:
        log.info(f"    {name}...")
        lw[name] = stack_weight(template)

    if has_qk_norm:
        log.info("    q_norm, k_norm...")
        lw["q_norm"] = stack_weight("model.layers.{i}.self_attn.q_norm.weight")
        lw["k_norm"] = stack_weight("model.layers.{i}.self_attn.k_norm.weight")

    # Global weights
    log.info("    embed, final_norm, lm_head...")
    embed      = single_weight("model.embed_tokens.weight")
    final_norm = single_weight("model.norm.weight")
    if "lm_head.weight" in all_keys:
        lm_head = single_weight("lm_head.weight")
    else:
        lm_head = embed  # tied

    # Close shard handles
    shard_handles.clear()
    gc.collect()
    log.info("Weights stacked and on device.")

    arch = {
        "n_layers": n_layers, "hidden": hidden, "n_heads": n_heads,
        "n_kv": n_kv, "head_dim": head_dim, "inter": inter,
        "rope_theta": float(rope_theta), "eps": eps, "has_qk_norm": has_qk_norm,
    }
    return arch, lw, embed, final_norm, lm_head


# ── Model components ─────────────────────────────────────────

def rms_norm(x, w, eps=1e-6):
    xf = x.astype(jnp.float32)
    norm = xf * lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (norm * w).astype(x.dtype)


def apply_rope(q, k, cos, sin):
    """Rotary embeddings. q/k: (..., seq, head_dim). cos/sin: (seq, head_dim)."""
    def rotate(x, c, s):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).astype(x.dtype)
    return rotate(q, cos, sin), rotate(k, cos, sin)


def precompute_rope(seq_len, head_dim, theta, dtype):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(t, freqs)
    cos = jnp.cos(angles).astype(dtype)  # (seq, head_dim/2)
    sin = jnp.sin(angles).astype(dtype)
    return cos, sin


# ── Forward pass builder ─────────────────────────────────────

def build_forward(arch):
    n_layers = arch["n_layers"]
    n_heads  = arch["n_heads"]
    n_kv     = arch["n_kv"]
    head_dim = arch["head_dim"]
    eps      = arch["eps"]
    has_qk   = arch["has_qk_norm"]
    kv_rep   = n_heads // n_kv

    def one_layer(hidden, lw_slice, cos, sin):
        """Single transformer block. lw_slice = dict of weight tensors for one layer."""
        B, S, H = hidden.shape
        residual = hidden

        # Pre-attention norm
        hidden = rms_norm(hidden, lw_slice["input_ln"], eps)

        # Q, K, V projections (no bias)
        q = jnp.dot(hidden, lw_slice["q_proj"].T)   # (B, S, n_heads*hd)
        k = jnp.dot(hidden, lw_slice["k_proj"].T)   # (B, S, n_kv*hd)
        v = jnp.dot(hidden, lw_slice["v_proj"].T)

        # Reshape to (B, heads, S, hd)
        q = q.reshape(B, S, n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, n_kv, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, n_kv, head_dim).transpose(0, 2, 1, 3)

        # QK norm (Qwen3)
        if has_qk:
            q = rms_norm(q, lw_slice["q_norm"], eps)
            k = rms_norm(k, lw_slice["k_norm"], eps)

        # RoPE — broadcast cos/sin over batch & heads dims
        cos_b = cos[None, None, :, :]  # (1, 1, S, hd)
        sin_b = sin[None, None, :, :]
        q, k = apply_rope(q, k, cos_b, sin_b)

        # GQA: repeat K, V
        if kv_rep > 1:
            k = jnp.repeat(k, kv_rep, axis=1)
            v = jnp.repeat(v, kv_rep, axis=1)

        # Attention scores
        scale = head_dim ** -0.5
        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale

        # Causal mask
        mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        neg_inf = jnp.finfo(hidden.dtype).min
        attn = jnp.where(mask[None, None], attn, neg_inf)

        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(hidden.dtype)
        out = jnp.matmul(attn, v)

        # Reshape back → output proj
        out = out.transpose(0, 2, 1, 3).reshape(B, S, out.shape[2] * out.shape[3])
        out = jnp.dot(out, lw_slice["o_proj"].T)
        hidden = residual + out

        # Post-attention norm + MLP (SiLU gate)
        residual = hidden
        hidden = rms_norm(hidden, lw_slice["post_ln"], eps)

        gate = jnp.dot(hidden, lw_slice["gate_proj"].T)
        up   = jnp.dot(hidden, lw_slice["up_proj"].T)
        hidden = jax.nn.silu(gate.astype(jnp.float32)).astype(hidden.dtype) * up
        hidden = jnp.dot(hidden, lw_slice["down_proj"].T)

        return residual + hidden

    @jax.jit
    def forward(input_ids, layer_weights, embed, final_norm, lm_head,
                cos, sin, layer_indices):
        """Full forward with layer-index swap. lax.scan over layer_indices.

        layer_indices: (n_layers,) int32 — identity for baseline, swapped for pairs.
        """
        hidden = embed[input_ids]  # (B, S, H)

        def scan_body(hidden, idx):
            # Dynamic-index into stacked weights for this layer
            lw_slice = jax.tree.map(lambda w: w[idx], layer_weights)
            return one_layer(hidden, lw_slice, cos, sin), None

        hidden, _ = lax.scan(scan_body, hidden, layer_indices)

        hidden = rms_norm(hidden, final_norm, eps)
        logits = jnp.dot(hidden, lm_head.T)
        return logits

    return forward


# ── KL divergence ────────────────────────────────────────────

@jax.jit
def kl_div(logits_base, logits_swap):
    """KL(base || swap) averaged over batch × tokens."""
    lp = jax.nn.log_softmax(logits_base.astype(jnp.float32), axis=-1)
    lq = jax.nn.log_softmax(logits_swap.astype(jnp.float32), axis=-1)
    p  = jnp.exp(lp)
    return jnp.mean(jnp.sum(p * (lp - lq), axis=-1))


# ── Main ─────────────────────────────────────────────────────

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    log.info(f"=== 7B Bisimulation: {MODEL_NAME} ===")
    log.info(f"Device: {jax.devices()}")
    log.info(f"Prompts: {N_PROMPTS}, SeqLen: {SEQ_LEN}, dtype: fp16")

    t0 = time.time()

    # ── Load model ──
    arch, lw, embed, final_norm, lm_head = load_and_stack(MODEL_NAME, DTYPE)
    n_layers = arch["n_layers"]
    t_load = time.time() - t0
    log.info(f"Weight loading: {t_load:.1f}s")

    # ── Tokenize ──
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    ids_list = []
    for p in PROMPTS:
        ids = tokenizer.encode(p, max_length=SEQ_LEN, truncation=True)
        ids = (ids + [pad] * SEQ_LEN)[:SEQ_LEN]
        ids_list.append(ids)
    input_ids = jnp.array(ids_list, dtype=jnp.int32)
    log.info(f"Input: {input_ids.shape}")

    # ── Precompute RoPE ──
    cos, sin = precompute_rope(SEQ_LEN, arch["head_dim"], arch["rope_theta"], DTYPE)
    cos, sin = jax.device_put(cos), jax.device_put(sin)

    # ── Build forward ──
    forward = build_forward(arch)

    # ── JIT warmup (first call compiles) ──
    log.info("JIT compile (first forward)...")
    baseline_idx = jnp.arange(n_layers, dtype=jnp.int32)
    t_jit = time.time()
    baseline_logits = forward(input_ids, lw, embed, final_norm, lm_head,
                              cos, sin, baseline_idx)
    baseline_logits.block_until_ready()
    t_jit = time.time() - t_jit
    log.info(f"JIT compilation: {t_jit:.1f}s")

    # ── Baseline (cached) ──
    log.info("Baseline forward...")
    t_base = time.time()
    baseline_logits = forward(input_ids, lw, embed, final_norm, lm_head,
                              cos, sin, baseline_idx)
    baseline_logits.block_until_ready()
    t_base = time.time() - t_base
    log.info(f"Baseline: {t_base:.2f}s")

    # ── Adjacent pairs ──
    n_pairs = n_layers - 1
    log.info(f"Computing {n_pairs} adjacent pairs...")
    results = []
    t_pairs = time.time()

    for i in range(n_pairs):
        j = i + 1
        t_p = time.time()

        # Swap index: swap position i and j
        swap_idx = baseline_idx.at[i].set(j).at[j].set(i)

        swap_logits = forward(input_ids, lw, embed, final_norm, lm_head,
                              cos, sin, swap_idx)
        swap_logits.block_until_ready()

        kl = float(kl_div(baseline_logits, swap_logits))
        dt = time.time() - t_p

        cat = "strong" if kl < 0.05 else "conditional" if kl < 0.10 else "non-bisimilar"
        results.append({"layer_a": i, "layer_b": j, "kl": kl, "category": cat})
        log.info(f"  ({i:2d},{j:2d}): KL={kl:.6e} [{cat:14s}] {dt:.2f}s")

    t_pairs_total = time.time() - t_pairs
    t_total = time.time() - t0

    # ── Summary ──
    strong = sum(1 for r in results if r["category"] == "strong")
    cond   = sum(1 for r in results if r["category"] == "conditional")
    non_b  = len(results) - strong - cond
    best   = min(results, key=lambda r: r["kl"])

    log.info(f"\n{'='*50}")
    log.info(f"Model: {MODEL_NAME} ({n_layers} layers)")
    log.info(f"Strong: {strong}  Conditional: {cond}  Non-bisimilar: {non_b}")
    log.info(f"Best pair: ({best['layer_a']},{best['layer_b']}) KL={best['kl']:.6e}")
    log.info(f"Pairs: {t_pairs_total:.1f}s ({t_pairs_total/n_pairs:.2f}s/pair)")
    log.info(f"Total (incl load+JIT): {t_total:.1f}s")

    # ── Save ──
    model_short = MODEL_NAME.split("/")[-1].lower()
    output = {
        "model": MODEL_NAME,
        "n_layers": n_layers,
        "architecture": arch,
        "n_prompts": N_PROMPTS,
        "seq_len": SEQ_LEN,
        "dtype": "float16",
        "device": str(jax.devices()[0]),
        "timing": {
            "load_s": t_load,
            "jit_compile_s": t_jit,
            "baseline_s": t_base,
            "pairs_s": t_pairs_total,
            "per_pair_s": t_pairs_total / n_pairs,
            "total_s": t_total,
        },
        "results": results,
        "summary": {
            "strong": strong,
            "conditional": cond,
            "non_bisimilar": non_b,
            "best_pair": [best["layer_a"], best["layer_b"]],
            "best_kl": best["kl"],
        },
    }
    out_path = os.path.join(REPORT_DIR, f"{model_short}_bisimulation.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
