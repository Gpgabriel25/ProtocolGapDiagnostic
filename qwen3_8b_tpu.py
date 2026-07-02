#!/usr/bin/env python3
"""
Qwen3-8B Bisimulation Distance on TPU v6e-16 Pod
=================================================
Computes replacement AND interchange bisimulation distances for 102 layer pairs
(gap 1–3) using 500 diverse prompts from WikiText-103 (128 tokens each).

Architecture:
  - JAX SPMD data parallelism across all 16 TPU v6e chips (4 workers × 4 chips)
  - Model weights replicated on each chip (~16 GB BF16, fits in 32 GB HBM)
  - Batch dimension sharded: 512 prompts / 16 chips = 32 prompts/chip
  - Index-based layer swap via lax.scan (single JIT compilation, no recompile per pair)

Distance definitions:
  - Replacement d(i→j): position j uses layer i's weights.  KL(base ‖ swapped).
  - Replacement d(j→i): position i uses layer j's weights.  KL(base ‖ swapped).
  - replacement_max = max(mean(d(i→j)), mean(d(j→i)))
  - Interchange: swap positions i and j.  KL(base ‖ swapped).

Launch on ALL 4 workers simultaneously:
  python3 /tmp/qwen3_8b_tpu.py

Output (worker 0 only): /tmp/qwen3_8b_output/qwen3_8b_results.json
"""

import os, sys, json, time, logging, gc, glob
import numpy as np

# Single-worker mode: restrict to local TPU chips only (no pod coordination)
os.environ.setdefault("TPU_CHIPS_PER_PROCESS_BOUNDS", "2,2,1")
os.environ.setdefault("TPU_PROCESS_BOUNDS", "1,1,1")
# Let JAX manage TPU memory
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from safetensors import safe_open
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoConfig

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s [w{jax.process_index()}] %(message)s",
)
log = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME   = os.environ.get("MODEL", "Qwen/Qwen3-8B")
N_PROMPTS    = int(os.environ.get("N_PROMPTS", "500"))
SEQ_LEN      = int(os.environ.get("SEQ_LEN", "128"))
MAX_GAP      = int(os.environ.get("MAX_GAP", "3"))
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", "0"))   # 0 = auto (all at once)
N_BOOTSTRAP  = int(os.environ.get("N_BOOTSTRAP", "1000"))
OUTPUT_DIR   = os.environ.get("OUTPUT_DIR", "/tmp/qwen3_8b_output")
DTYPE        = jnp.bfloat16

IS_LEADER    = jax.process_index() == 0


# ── Mesh setup ───────────────────────────────────────────────────────────────

def setup_mesh():
    """Create a 1-D data-parallel mesh over all TPU chips."""
    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices, axis_names=("data",))
    log.info(f"Mesh: {jax.device_count()} devices, shape={devices.shape}")
    return mesh


# ── Weight loading (memory-efficient, lazy from shards) ──────────────────────

def load_and_stack(model_name, mesh, dtype=jnp.bfloat16):
    """Load safetensors → stack per-layer weights → replicate across mesh.

    Returns: arch dict, layer_weights dict, embed, final_norm, lm_head, repo_path
    """
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    n_layers = config.num_hidden_layers
    hidden   = config.hidden_size
    n_heads  = config.num_attention_heads
    n_kv     = getattr(config, "num_key_value_heads", n_heads)
    head_dim = getattr(config, "head_dim", hidden // n_heads)
    inter    = config.intermediate_size

    rp = getattr(config, "rope_parameters", None) or {}
    rope_theta = rp.get("rope_theta", getattr(config, "rope_theta", 10000.0))
    eps = getattr(config, "rms_norm_eps", 1e-6)

    log.info(f"Model: {model_name}")
    log.info(f"  {n_layers}L, h={hidden}, heads={n_heads}/{n_kv}, inter={inter}, "
             f"head_dim={head_dim}, rope_theta={rope_theta}, eps={eps}")

    # Download model shards (each worker downloads independently; HF caches)
    log.info("Downloading model shards...")
    repo = snapshot_download(
        model_name,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model"],
    )

    shards = sorted(glob.glob(os.path.join(repo, "model*.safetensors")))
    assert shards, f"No safetensors in {repo}"
    log.info(f"  {len(shards)} shard files")

    # Weight map
    idx_path = os.path.join(repo, "model.safetensors.index.json")
    weight_map = None
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            weight_map = json.load(f)["weight_map"]

    # Open shard handles (lazy — data stays on disk until get_tensor)
    handles = {}
    for sf in shards:
        name = os.path.basename(sf)
        handles[name] = safe_open(sf, framework="numpy")

    all_keys = set()
    for h in handles.values():
        all_keys.update(h.keys())
    log.info(f"  {len(all_keys)} tensors available")

    def get(key):
        if weight_map:
            shard_name = weight_map[key]
        else:
            shard_name = os.path.basename(shards[0])
        return handles[shard_name].get_tensor(key)

    has_qk_norm = "model.layers.0.self_attn.q_norm.weight" in all_keys
    log.info(f"  QK norm: {has_qk_norm}")

    # Sharding specs for replicated placement
    replicated = NamedSharding(mesh, P())  # fully replicated

    def to_bf16(arr):
        """Convert numpy array to bfloat16. Handle bf16 source gracefully."""
        if arr.dtype == np.float32 or arr.dtype == np.float16:
            return arr.astype(jnp.bfloat16)
        # numpy doesn't natively support bfloat16; jnp handles it
        return arr

    def stack_weight(template):
        """Stack one weight type across all layers → (n_layers, ...) replicated."""
        arrs = []
        for i in range(n_layers):
            arrs.append(get(template.format(i=i)))
        stacked = np.stack(arrs)
        del arrs
        stacked = to_bf16(stacked)
        result = jax.device_put(jnp.array(stacked, dtype=dtype), replicated)
        del stacked
        gc.collect()
        return result

    def single_weight(key):
        arr = get(key)
        arr = to_bf16(arr)
        return jax.device_put(jnp.array(arr, dtype=dtype), replicated)

    log.info("Stacking layer weights (bf16, replicated)...")
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
        log.info("    q_norm, k_norm...")
        lw["q_norm"] = stack_weight("model.layers.{i}.self_attn.q_norm.weight")
        lw["k_norm"] = stack_weight("model.layers.{i}.self_attn.k_norm.weight")

    log.info("    embed, final_norm, lm_head...")
    embed      = single_weight("model.embed_tokens.weight")
    final_norm = single_weight("model.norm.weight")
    if "lm_head.weight" in all_keys:
        lm_head = single_weight("lm_head.weight")
    else:
        lm_head = embed  # tied

    handles.clear()
    gc.collect()
    log.info("Weights stacked and replicated across mesh.")

    arch = {
        "n_layers": n_layers, "hidden": hidden, "n_heads": n_heads,
        "n_kv": n_kv, "head_dim": head_dim, "inter": inter,
        "rope_theta": float(rope_theta), "eps": eps,
        "has_qk_norm": has_qk_norm,
        "vocab_size": int(config.vocab_size),
    }
    return arch, lw, embed, final_norm, lm_head, repo


# ── Model components ─────────────────────────────────────────────────────────

def rms_norm(x, w, eps=1e-6):
    xf = x.astype(jnp.float32)
    norm = xf * lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (norm * w.astype(jnp.float32)).astype(x.dtype)


def precompute_rope(seq_len, head_dim, theta, dtype):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(t, freqs)  # (S, hd/2)
    cos = jnp.cos(angles).astype(dtype)
    sin = jnp.sin(angles).astype(dtype)
    return cos, sin


def apply_rope(q, k, cos, sin):
    """q/k: (B, heads, S, hd).  cos/sin: (1, 1, S, hd/2)."""
    def rotate(x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([x1 * cos - x2 * sin,
                                x2 * cos + x1 * sin], axis=-1).astype(x.dtype)
    return rotate(q), rotate(k)


# ── Forward pass builder ─────────────────────────────────────────────────────

def build_forward(arch, seq_len):
    """Returns a JIT-compiled forward function with index-based layer swap."""
    n_layers = arch["n_layers"]
    n_heads  = arch["n_heads"]
    n_kv     = arch["n_kv"]
    head_dim = arch["head_dim"]
    eps      = arch["eps"]
    has_qk   = arch["has_qk_norm"]
    kv_rep   = n_heads // n_kv

    # Static causal mask
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))

    def one_layer(hidden, lw_slice, cos_b, sin_b):
        """Single transformer block."""
        B, S, H = hidden.shape
        residual = hidden

        # Pre-attention norm
        hidden = rms_norm(hidden, lw_slice["input_ln"], eps)

        # QKV projections
        q = jnp.dot(hidden, lw_slice["q_proj"].T)
        k = jnp.dot(hidden, lw_slice["k_proj"].T)
        v = jnp.dot(hidden, lw_slice["v_proj"].T)

        q = q.reshape(B, S, n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, n_kv, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, n_kv, head_dim).transpose(0, 2, 1, 3)

        # QK norm (Qwen3)
        if has_qk:
            q = rms_norm(q, lw_slice["q_norm"], eps)
            k = rms_norm(k, lw_slice["k_norm"], eps)

        # RoPE
        q, k = apply_rope(q, k, cos_b, sin_b)

        # GQA: repeat K,V
        if kv_rep > 1:
            k = jnp.repeat(k, kv_rep, axis=1)
            v = jnp.repeat(v, kv_rep, axis=1)

        # Attention
        scale = head_dim ** -0.5
        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        neg_inf = jnp.finfo(hidden.dtype).min
        attn = jnp.where(causal_mask[None, None], attn, neg_inf)
        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(hidden.dtype)
        out = jnp.matmul(attn, v)

        # Output projection
        out = out.transpose(0, 2, 1, 3).reshape(B, S, H)
        out = jnp.dot(out, lw_slice["o_proj"].T)
        hidden = residual + out

        # MLP (SwiGLU)
        residual = hidden
        hidden = rms_norm(hidden, lw_slice["post_ln"], eps)
        gate = jnp.dot(hidden, lw_slice["gate_proj"].T)
        up   = jnp.dot(hidden, lw_slice["up_proj"].T)
        hidden = jax.nn.silu(gate.astype(jnp.float32)).astype(hidden.dtype) * up
        hidden = jnp.dot(hidden, lw_slice["down_proj"].T)

        return residual + hidden

    @jax.jit
    def forward(input_ids, layer_weights, embed, final_norm_w, lm_head_w,
                cos_b, sin_b, layer_indices):
        """Full forward pass with index-based layer swap via lax.scan.

        layer_indices: (n_layers,) int32 — identity for baseline, modified for swaps.
        Returns: logits (B, S, V)
        """
        hidden = embed[input_ids]  # (B, S, H)

        def scan_body(hidden, idx):
            lw_slice = jax.tree.map(lambda w: w[idx], layer_weights)
            return one_layer(hidden, lw_slice, cos_b, sin_b), None

        hidden, _ = lax.scan(scan_body, hidden, layer_indices)
        hidden = rms_norm(hidden, final_norm_w, eps)
        logits = jnp.dot(hidden, lm_head_w.T)
        return logits

    return forward


# ── KL divergence ────────────────────────────────────────────────────────────

@jax.jit
def kl_per_prompt(logits_base, logits_swap, mask):
    """Per-prompt KL(base || swap), averaged over valid token positions.

    logits_base, logits_swap: (B, S, V) in bfloat16
    mask: (B, S) bool — True for real tokens
    Returns: (B,) float32 KL values
    """
    lp_base = jax.nn.log_softmax(logits_base.astype(jnp.float32), axis=-1)
    lp_swap = jax.nn.log_softmax(logits_swap.astype(jnp.float32), axis=-1)
    p_base  = jnp.exp(lp_base)

    # Per-token KL: sum over vocab
    kl_per_token = jnp.sum(p_base * (lp_base - lp_swap), axis=-1)  # (B, S)
    kl_per_token = jnp.maximum(kl_per_token, 0.0)  # numerical safety

    # Mask and average over valid tokens per prompt
    kl_masked = kl_per_token * mask  # (B, S)
    n_valid = jnp.maximum(mask.sum(axis=-1), 1.0)  # (B,)
    return kl_masked.sum(axis=-1) / n_valid  # (B,)


@jax.jit
def kl_mean(logits_base, logits_swap, mask):
    """Scalar mean KL across all prompts and tokens."""
    return jnp.mean(kl_per_prompt(logits_base, logits_swap, mask))


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(values, n_boot=1000, seed=42):
    """95% bootstrap CI for the mean of `values`."""
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    means = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means[b] = np.mean(sample)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ── Prompt loading from WikiText-103 ─────────────────────────────────────────

def load_wikitext_prompts(tokenizer, n_prompts, seq_len):
    """Load diverse prompts from WikiText-103 test set.

    Returns: input_ids (n_prompts_padded, seq_len) int32,
             attention_mask (n_prompts_padded, seq_len) bool,
             actual_n_prompts int
    """
    from datasets import load_dataset

    log.info("Loading WikiText-103 test set...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

    # Filter to non-empty, non-header paragraphs
    texts = []
    for row in ds:
        text = row["text"].strip()
        if len(text) < 50:
            continue
        if text.startswith("="):
            continue
        texts.append(text)

    log.info(f"  {len(texts)} candidate paragraphs")

    # Deterministic shuffle for reproducibility
    rng = np.random.default_rng(42)
    rng.shuffle(texts)

    # Tokenize and filter to sequences with enough tokens
    all_ids = []
    all_mask = []
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    for text in texts:
        if len(all_ids) >= n_prompts:
            break
        ids = tokenizer.encode(text, max_length=seq_len, truncation=True)
        if len(ids) < 16:  # skip very short sequences
            continue
        # Pad to seq_len
        n_real = len(ids)
        ids = (ids + [pad_id] * seq_len)[:seq_len]
        mask = [True] * n_real + [False] * (seq_len - n_real)
        mask = mask[:seq_len]
        all_ids.append(ids)
        all_mask.append(mask)

    actual_n = len(all_ids)
    assert actual_n >= n_prompts, f"Only got {actual_n} prompts, need {n_prompts}"
    all_ids = all_ids[:n_prompts]
    all_mask = all_mask[:n_prompts]

    # Pad to next multiple of device count for even sharding
    n_devices = jax.device_count()
    remainder = len(all_ids) % n_devices
    if remainder != 0:
        n_pad = n_devices - remainder
        for _ in range(n_pad):
            all_ids.append([pad_id] * seq_len)
            all_mask.append([False] * seq_len)
    else:
        n_pad = 0

    padded_n = len(all_ids)
    log.info(f"  {n_prompts} prompts + {n_pad} padding = {padded_n} total "
             f"({padded_n // n_devices} per device)")

    input_ids = np.array(all_ids, dtype=np.int32)
    attn_mask = np.array(all_mask, dtype=np.bool_)
    return input_ids, attn_mask, n_prompts


# ── Generate layer pairs ─────────────────────────────────────────────────────

def generate_pairs(n_layers, max_gap):
    """Generate all (i, j) pairs with 1 <= gap <= max_gap."""
    pairs = []
    for gap in range(1, max_gap + 1):
        for i in range(n_layers - gap):
            j = i + gap
            pairs.append((i, j, gap))
    return pairs


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    if IS_LEADER:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info(f"=== Qwen3-8B Bisimulation on TPU v6e-16 ===")
    log.info(f"JAX version: {jax.__version__}")
    log.info(f"Process: {jax.process_index()}/{jax.process_count()}")
    log.info(f"Devices: {jax.device_count()} total, {jax.local_device_count()} local")
    log.info(f"Config: {N_PROMPTS} prompts, {SEQ_LEN} tokens, max_gap={MAX_GAP}, "
             f"dtype=bf16, bootstrap={N_BOOTSTRAP}")

    # ── Setup mesh ──
    mesh = setup_mesh()
    data_sharding = NamedSharding(mesh, P("data"))        # shard batch dim
    replicated    = NamedSharding(mesh, P())               # replicate

    # ── Load model ──
    arch, lw, embed, final_norm, lm_head, repo = load_and_stack(
        MODEL_NAME, mesh, dtype=DTYPE
    )
    n_layers = arch["n_layers"]
    t_load = time.time() - t0
    log.info(f"Model loaded in {t_load:.1f}s")

    # ── Tokenize prompts ──
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids_np, attn_mask_np, actual_n = load_wikitext_prompts(
        tokenizer, N_PROMPTS, SEQ_LEN
    )
    padded_n = input_ids_np.shape[0]

    # Place on mesh: shard along batch dimension
    input_ids = jax.device_put(jnp.array(input_ids_np), data_sharding)
    attn_mask = jax.device_put(jnp.array(attn_mask_np, dtype=jnp.bool_), data_sharding)

    log.info(f"Input: {input_ids.shape}, sharded across {jax.device_count()} devices")

    # ── Precompute RoPE ──
    cos, sin = precompute_rope(SEQ_LEN, arch["head_dim"], arch["rope_theta"], DTYPE)
    # Broadcast to (1, 1, S, hd/2) for attention
    cos_b = jax.device_put(cos[None, None, :, :], replicated)
    sin_b = jax.device_put(sin[None, None, :, :], replicated)

    # ── Build forward ──
    forward = build_forward(arch, SEQ_LEN)

    # ── JIT warmup ──
    baseline_idx = jnp.arange(n_layers, dtype=jnp.int32)
    baseline_idx = jax.device_put(baseline_idx, replicated)

    log.info("JIT compilation (first forward)...")
    t_jit = time.time()
    baseline_logits = forward(input_ids, lw, embed, final_norm, lm_head,
                              cos_b, sin_b, baseline_idx)
    baseline_logits.block_until_ready()
    t_jit = time.time() - t_jit
    log.info(f"JIT compilation: {t_jit:.1f}s")

    # ── Baseline logits ──
    log.info("Computing baseline logits...")
    t_base = time.time()
    baseline_logits = forward(input_ids, lw, embed, final_norm, lm_head,
                              cos_b, sin_b, baseline_idx)
    baseline_logits.block_until_ready()
    t_base = time.time() - t_base
    log.info(f"Baseline forward: {t_base:.2f}s, logits shape: {baseline_logits.shape}")

    # ── Generate pairs ──
    pairs = generate_pairs(n_layers, MAX_GAP)
    n_pairs = len(pairs)
    log.info(f"Computing {n_pairs} pairs (gap 1–{MAX_GAP})...")

    # ── Pair computation loop ──
    results = []
    t_pairs_start = time.time()

    for pi, (layer_i, layer_j, gap) in enumerate(pairs):
        t_p = time.time()

        try:
            # --- Replacement i→j: position j uses layer i's weights ---
            rep_ij_idx = jnp.arange(n_layers, dtype=jnp.int32).at[layer_j].set(layer_i)
            rep_ij_idx = jax.device_put(rep_ij_idx, replicated)

            rep_ij_logits = forward(input_ids, lw, embed, final_norm, lm_head,
                                    cos_b, sin_b, rep_ij_idx)
            rep_ij_logits.block_until_ready()

            kl_ij_per = np.array(kl_per_prompt(baseline_logits, rep_ij_logits, attn_mask))
            kl_ij_per = kl_ij_per[:actual_n]  # drop padding prompts
            rep_kl_ij = float(np.mean(kl_ij_per))

            del rep_ij_logits

            # --- Replacement j→i: position i uses layer j's weights ---
            rep_ji_idx = jnp.arange(n_layers, dtype=jnp.int32).at[layer_i].set(layer_j)
            rep_ji_idx = jax.device_put(rep_ji_idx, replicated)

            rep_ji_logits = forward(input_ids, lw, embed, final_norm, lm_head,
                                    cos_b, sin_b, rep_ji_idx)
            rep_ji_logits.block_until_ready()

            kl_ji_per = np.array(kl_per_prompt(baseline_logits, rep_ji_logits, attn_mask))
            kl_ji_per = kl_ji_per[:actual_n]
            rep_kl_ji = float(np.mean(kl_ji_per))

            del rep_ji_logits

            # --- Interchange: swap positions i and j ---
            inter_idx = jnp.arange(n_layers, dtype=jnp.int32)
            inter_idx = inter_idx.at[layer_i].set(layer_j).at[layer_j].set(layer_i)
            inter_idx = jax.device_put(inter_idx, replicated)

            inter_logits = forward(input_ids, lw, embed, final_norm, lm_head,
                                   cos_b, sin_b, inter_idx)
            inter_logits.block_until_ready()

            kl_inter_per = np.array(kl_per_prompt(baseline_logits, inter_logits, attn_mask))
            kl_inter_per = kl_inter_per[:actual_n]
            inter_kl = float(np.mean(kl_inter_per))

            del inter_logits

            # --- Derived metrics ---
            rep_max = max(rep_kl_ij, rep_kl_ji)
            ir_ratio = inter_kl / max(rep_max, 1e-10)

            # --- Bootstrap CIs (worker 0 only) ---
            ci_rep_ij = bootstrap_ci(kl_ij_per, N_BOOTSTRAP)
            ci_rep_ji = bootstrap_ci(kl_ji_per, N_BOOTSTRAP)
            ci_inter  = bootstrap_ci(kl_inter_per, N_BOOTSTRAP)

            # --- Symmetrized mean KL for replacement ---
            sym_per = (kl_ij_per + kl_ji_per) / 2.0
            sym_mean = float(np.mean(sym_per))
            ci_sym = bootstrap_ci(sym_per, N_BOOTSTRAP)

            dt = time.time() - t_p

            pair_result = {
                "layer_i": int(layer_i),
                "layer_j": int(layer_j),
                "gap": int(gap),
                "replacement_kl_ij": rep_kl_ij,
                "replacement_kl_ji": rep_kl_ji,
                "replacement_max": rep_max,
                "replacement_sym_mean": sym_mean,
                "interchange_kl": inter_kl,
                "ir_ratio": ir_ratio,
                "ci_95_rep_ij": list(ci_rep_ij),
                "ci_95_rep_ji": list(ci_rep_ji),
                "ci_95_interchange": list(ci_inter),
                "ci_95_sym": list(ci_sym),
                "time_s": dt,
            }
            results.append(pair_result)

            if IS_LEADER:
                cat_rep = ("strong" if rep_max < 0.05 else
                           "cond" if rep_max < 0.10 else "non")
                cat_int = ("strong" if inter_kl < 0.05 else
                           "cond" if inter_kl < 0.10 else "non")
                log.info(
                    f"  [{pi+1:3d}/{n_pairs}] ({layer_i:2d},{layer_j:2d}) g={gap}  "
                    f"rep_max={rep_max:.4e} [{cat_rep:4s}]  "
                    f"inter={inter_kl:.4e} [{cat_int:4s}]  "
                    f"IR={ir_ratio:.3f}  {dt:.1f}s"
                )

            # Periodic checkpoint save
            if IS_LEADER and (pi + 1) % 10 == 0:
                _save_checkpoint(results, arch, t0, n_pairs)

        except Exception as e:
            log.warning(f"  [{pi+1}/{n_pairs}] ({layer_i},{layer_j}) FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "layer_i": int(layer_i),
                "layer_j": int(layer_j),
                "gap": int(gap),
                "error": str(e),
            })
            continue

    t_pairs_total = time.time() - t_pairs_start
    t_total = time.time() - t0

    # ── Final save (worker 0 only) ──
    if IS_LEADER:
        _save_final(results, arch, t0, t_load, t_jit, t_base,
                    t_pairs_total, t_total, n_pairs, actual_n)

    log.info(f"Done. Total: {t_total:.1f}s")


def _save_checkpoint(results, arch, t0, n_pairs):
    """Save intermediate results every 10 pairs."""
    ckpt = {
        "model": MODEL_NAME,
        "checkpoint": True,
        "pairs_complete": len(results),
        "pairs_total": n_pairs,
        "elapsed_s": time.time() - t0,
        "pairs": results,
    }
    ckpt_path = os.path.join(OUTPUT_DIR, "qwen3_8b_checkpoint.json")
    with open(ckpt_path, "w") as f:
        json.dump(ckpt, f, indent=2)
    log.info(f"  Checkpoint saved ({len(results)}/{n_pairs} pairs)")


def _save_final(results, arch, t0, t_load, t_jit, t_base,
                t_pairs_total, t_total, n_pairs, actual_n):
    """Save final results and summary."""
    # Filter successful results
    ok = [r for r in results if "error" not in r]
    failed = len(results) - len(ok)

    if ok:
        rep_kls = [r["replacement_max"] for r in ok]
        int_kls = [r["interchange_kl"] for r in ok]
        ir_ratios = [r["ir_ratio"] for r in ok]

        # Per-gap statistics
        gap_stats = {}
        for gap in range(1, MAX_GAP + 1):
            gap_pairs = [r for r in ok if r["gap"] == gap]
            if gap_pairs:
                gap_stats[f"gap_{gap}"] = {
                    "n_pairs": len(gap_pairs),
                    "mean_replacement": float(np.mean([r["replacement_max"] for r in gap_pairs])),
                    "mean_interchange": float(np.mean([r["interchange_kl"] for r in gap_pairs])),
                    "mean_ir_ratio": float(np.mean([r["ir_ratio"] for r in gap_pairs])),
                    "min_replacement": float(np.min([r["replacement_max"] for r in gap_pairs])),
                    "min_interchange": float(np.min([r["interchange_kl"] for r in gap_pairs])),
                }

        # Bisimulation categories
        strong_rep = sum(1 for r in ok if r["replacement_max"] < 0.05)
        cond_rep   = sum(1 for r in ok if 0.05 <= r["replacement_max"] < 0.10)
        strong_int = sum(1 for r in ok if r["interchange_kl"] < 0.05)
        cond_int   = sum(1 for r in ok if 0.05 <= r["interchange_kl"] < 0.10)

        summary = {
            "mean_replacement": float(np.mean(rep_kls)),
            "mean_interchange": float(np.mean(int_kls)),
            "mean_ir_ratio": float(np.mean(ir_ratios)),
            "median_replacement": float(np.median(rep_kls)),
            "median_interchange": float(np.median(int_kls)),
            "min_replacement": float(np.min(rep_kls)),
            "min_interchange": float(np.min(int_kls)),
            "max_replacement": float(np.max(rep_kls)),
            "max_interchange": float(np.max(int_kls)),
            "strong_replacement": strong_rep,
            "conditional_replacement": cond_rep,
            "strong_interchange": strong_int,
            "conditional_interchange": cond_int,
            "num_pairs": len(ok),
            "num_failed": failed,
            "gap_stats": gap_stats,
        }

        best_rep = min(ok, key=lambda r: r["replacement_max"])
        best_int = min(ok, key=lambda r: r["interchange_kl"])
        summary["best_replacement_pair"] = [best_rep["layer_i"], best_rep["layer_j"]]
        summary["best_replacement_kl"] = best_rep["replacement_max"]
        summary["best_interchange_pair"] = [best_int["layer_i"], best_int["layer_j"]]
        summary["best_interchange_kl"] = best_int["interchange_kl"]
    else:
        summary = {"num_pairs": 0, "num_failed": failed}

    payload = {
        "model": MODEL_NAME,
        "num_prompts": actual_n,
        "seq_len": SEQ_LEN,
        "max_gap": MAX_GAP,
        "dtype": "bfloat16",
        "n_devices": jax.device_count(),
        "n_workers": jax.process_count(),
        "jax_version": jax.__version__,
        "architecture": arch,
        "n_bootstrap": N_BOOTSTRAP,
        "timing": {
            "load_s": t_load,
            "jit_compile_s": t_jit,
            "baseline_s": t_base,
            "pairs_s": t_pairs_total,
            "per_pair_s": t_pairs_total / max(1, len(ok)),
            "total_s": t_total,
        },
        "summary": summary,
        "pairs": results,
    }

    out_path = os.path.join(OUTPUT_DIR, "qwen3_8b_results.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info(f"Results saved: {out_path}")

    # Also print a summary table
    if ok:
        print(f"\n{'='*90}")
        print(f"BISIMULATION DISTANCES — {MODEL_NAME}  ({arch['n_layers']} layers)")
        print(f"Protocol: {actual_n} prompts × {SEQ_LEN} tokens, bf16, {jax.device_count()} TPU chips")
        print(f"{'='*90}")
        print(f"{'Pair':>8s}  {'Gap':>3s}  {'Rep max':>10s}  {'Inter':>10s}  "
              f"{'IR ratio':>8s}  {'Rep cat':>8s}  {'Int cat':>8s}")
        print("-" * 70)
        for r in ok:
            cat_r = ("strong" if r["replacement_max"] < 0.05 else
                     "cond" if r["replacement_max"] < 0.10 else "non")
            cat_i = ("strong" if r["interchange_kl"] < 0.05 else
                     "cond" if r["interchange_kl"] < 0.10 else "non")
            print(f"({r['layer_i']:2d},{r['layer_j']:2d})  {r['gap']:3d}  "
                  f"{r['replacement_max']:10.4e}  {r['interchange_kl']:10.4e}  "
                  f"{r['ir_ratio']:8.3f}  {cat_r:>8s}  {cat_i:>8s}")
        print(f"{'='*90}")
        print(f"Summary: replacement strong={strong_rep} cond={cond_rep}  "
              f"interchange strong={strong_int} cond={cond_int}")
        print(f"Best replacement: ({best_rep['layer_i']},{best_rep['layer_j']}) "
              f"KL={best_rep['replacement_max']:.4e}")
        print(f"Best interchange: ({best_int['layer_i']},{best_int['layer_j']}) "
              f"KL={best_int['interchange_kl']:.4e}")
        print(f"Timing: {t_total:.0f}s total, {t_pairs_total/max(1,len(ok)):.1f}s/pair")
        print(f"{'='*90}")

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "qwen3_8b_pairs.csv")
    with open(csv_path, "w") as f:
        f.write("layer_i,layer_j,gap,rep_kl_ij,rep_kl_ji,rep_max,rep_sym,"
                "interchange,ir_ratio,ci_sym_lo,ci_sym_hi,ci_inter_lo,ci_inter_hi\n")
        for r in ok:
            f.write(f"{r['layer_i']},{r['layer_j']},{r['gap']},"
                    f"{r['replacement_kl_ij']:.6f},{r['replacement_kl_ji']:.6f},"
                    f"{r['replacement_max']:.6f},{r['replacement_sym_mean']:.6f},"
                    f"{r['interchange_kl']:.6f},{r['ir_ratio']:.4f},"
                    f"{r['ci_95_sym'][0]:.6f},{r['ci_95_sym'][1]:.6f},"
                    f"{r['ci_95_interchange'][0]:.6f},{r['ci_95_interchange'][1]:.6f}\n")
    log.info(f"CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
