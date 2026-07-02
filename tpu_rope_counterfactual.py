#!/usr/bin/env python3
"""
tpu_rope_counterfactual.py — RoPE counterfactual experiment on Qwen3-8B.

Tests whether RoPE rotation is MECHANISTICALLY responsible for the protocol gap
(replacement vs interchange distance divergence) by disabling RoPE at inference
time and measuring how the I/R ratio changes.

For each adjacent layer pair (i, i+1):
  - Normal condition: compute replacement + interchange distance with standard RoPE
  - No-RoPE condition: same distances with cos=1, sin=0 (identity rotation)

If the protocol gap shrinks when RoPE is disabled, RoPE rotation is the mechanism.
If the gap persists, the mechanism is layer weight specialization independent of PE.

Usage on TPU v6e-16 (single-host mode):
  screen -dmS rope bash -c 'source ~/venv311/bin/activate && \\
    python3 tpu_rope_counterfactual.py 2>&1 | tee /tmp/rope_counterfactual/run.log'
"""

import os, sys, json, time, logging, gc, math
import numpy as np
from pathlib import Path

# ── TPU single-host env vars ─────────────────────────────────────────────────
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
os.environ.setdefault("TPU_CHIPS_PER_PROCESS_BOUNDS", "2,2,1")
os.environ.setdefault("TPU_PROCESS_BOUNDS", "1,1,1")
os.environ.setdefault("CLOUD_TPU_TASK_ID", "0")

import jax
import jax.numpy as jnp
from jax import lax
from safetensors import safe_open
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoConfig
import glob as _glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("MODEL", "Qwen/Qwen3-8B")
DTYPE = jnp.bfloat16
OUTPUT_DIR = "/tmp/rope_counterfactual"
SEQ_LEN = 64
N_PROMPTS = 50

# ~15 evenly-spaced adjacent pairs across 36 layers
PAIRS = [(2, 3), (4, 5), (6, 7), (9, 10), (11, 12), (14, 15),
         (17, 18), (19, 20), (22, 23), (24, 25), (27, 28),
         (29, 30), (31, 32), (33, 34), (34, 35)]

_ARCH = {}


# ── Weight loading ───────────────────────────────────────────────────────────

def load_and_stack(model_name, dtype=jnp.bfloat16):
    log.info(f"Downloading/loading {model_name}...")
    repo = snapshot_download(
        model_name,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model"],
    )
    cfg = AutoConfig.from_pretrained(repo, local_files_only=True)

    n_layers = cfg.num_hidden_layers
    hidden = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_heads)
    head_dim = getattr(cfg, "head_dim", hidden // n_heads)
    inter = cfg.intermediate_size
    rp = getattr(cfg, "rope_scaling", None) or {}
    rope_theta = float(rp.get("rope_theta",
                               getattr(cfg, "rope_theta", 10000.0)))
    eps = getattr(cfg, "rms_norm_eps", 1e-6)

    _ARCH.update(
        n_layers=n_layers, hidden=hidden, n_heads=n_heads, n_kv=n_kv,
        head_dim=head_dim, inter=inter, rope_theta=rope_theta, eps=eps,
    )
    log.info(f"{model_name}: {n_layers}L h={hidden} heads={n_heads}/{n_kv} "
             f"head_dim={head_dim} inter={inter}")

    shards = sorted(_glob.glob(os.path.join(repo, "model*.safetensors")))
    assert shards, f"No safetensors found in {repo}"

    idx_path = os.path.join(repo, "model.safetensors.index.json")
    weight_map = None
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            weight_map = json.load(f)["weight_map"]

    handles = {os.path.basename(s): safe_open(s, framework="numpy")
               for s in shards}
    all_keys = {k for h in handles.values() for k in h.keys()}

    def get_tensor(key):
        sf = weight_map[key] if weight_map else os.path.basename(shards[0])
        return handles[sf].get_tensor(key)

    def stack(tmpl):
        arr = np.stack([get_tensor(tmpl.format(i=i)) for i in range(n_layers)])
        out = jax.device_put(jnp.array(arr, dtype=dtype))
        del arr; gc.collect()
        return out

    def one(key):
        arr = get_tensor(key)
        return jax.device_put(jnp.array(arr, dtype=dtype))

    has_qk = "model.layers.0.self_attn.q_norm.weight" in all_keys
    _ARCH["has_qk_norm"] = has_qk

    log.info("Stacking per-layer weights (bf16)...")
    weight_templates = [
        ("q_proj",    "model.layers.{i}.self_attn.q_proj.weight"),
        ("k_proj",    "model.layers.{i}.self_attn.k_proj.weight"),
        ("v_proj",    "model.layers.{i}.self_attn.v_proj.weight"),
        ("o_proj",    "model.layers.{i}.self_attn.o_proj.weight"),
        ("gate_proj", "model.layers.{i}.mlp.gate_proj.weight"),
        ("up_proj",   "model.layers.{i}.mlp.up_proj.weight"),
        ("down_proj", "model.layers.{i}.mlp.down_proj.weight"),
        ("input_ln",  "model.layers.{i}.input_layernorm.weight"),
        ("post_ln",   "model.layers.{i}.post_attention_layernorm.weight"),
    ]
    lw = {}
    for name, tmpl in weight_templates:
        log.info(f"  {name}")
        lw[name] = stack(tmpl)
    if has_qk:
        lw["q_norm"] = stack("model.layers.{i}.self_attn.q_norm.weight")
        lw["k_norm"] = stack("model.layers.{i}.self_attn.k_norm.weight")

    embed = one("model.embed_tokens.weight")
    final_norm = one("model.norm.weight")
    if "lm_head.weight" in all_keys:
        lm_head = one("lm_head.weight")
    else:
        lm_head = embed

    handles.clear(); gc.collect()
    log.info("All weights on device.")
    return lw, embed, final_norm, lm_head, repo


# ── RoPE + norms ─────────────────────────────────────────────────────────────

def precompute_rope(seq_len, head_dim, theta, dtype):
    freqs = 1.0 / (theta ** (
        jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    t = jnp.arange(seq_len, dtype=jnp.float32)
    angles = jnp.outer(t, freqs)
    cos = jnp.cos(angles).astype(dtype)[None, None]
    sin = jnp.sin(angles).astype(dtype)[None, None]
    return cos, sin


def apply_rope(q, k, cos, sin):
    def rot(x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([x1 * cos - x2 * sin,
                                 x2 * cos + x1 * sin], axis=-1).astype(x.dtype)
    return rot(q), rot(k)


def rms_norm(x, w, eps=1e-6):
    xf = x.astype(jnp.float32)
    return (xf * lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
            * w.astype(jnp.float32)).astype(x.dtype)


# ── Forward pass builder ─────────────────────────────────────────────────────

def build_model(arch, seq_len):
    """Build JIT-compiled forward passes.

    Both forward_full and forward_swap accept (cos, sin) as explicit arguments
    so we can switch between normal RoPE and no-RoPE modes.
    """
    n_layers = arch["n_layers"]
    n_heads = arch["n_heads"]
    n_kv = arch["n_kv"]
    head_dim = arch["head_dim"]
    eps = arch["eps"]
    has_qk = arch["has_qk_norm"]
    kv_rep = n_heads // n_kv
    _cmask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))

    def one_layer(h, lw_s, cos, sin):
        B, S, H = h.shape
        res = h
        h = rms_norm(h, lw_s["input_ln"], eps)
        q = jnp.dot(h, lw_s["q_proj"].T)
        k = jnp.dot(h, lw_s["k_proj"].T)
        v = jnp.dot(h, lw_s["v_proj"].T)
        q = q.reshape(B, S, n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, n_kv, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, n_kv, head_dim).transpose(0, 2, 1, 3)
        if has_qk:
            q = rms_norm(q, lw_s["q_norm"], eps)
            k = rms_norm(k, lw_s["k_norm"], eps)
        q, k = apply_rope(q, k, cos, sin)
        if kv_rep > 1:
            k = jnp.repeat(k, kv_rep, axis=1)
            v = jnp.repeat(v, kv_rep, axis=1)
        scale = head_dim ** -0.5
        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        neg_inf = jnp.finfo(h.dtype).min
        attn = jnp.where(_cmask[None, None], attn, neg_inf)
        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(h.dtype)
        out = jnp.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, H)
        out = jnp.dot(out, lw_s["o_proj"].T)
        h = res + out
        res = h
        h = rms_norm(h, lw_s["post_ln"], eps)
        gate = jnp.dot(h, lw_s["gate_proj"].T)
        up = jnp.dot(h, lw_s["up_proj"].T)
        h = jax.nn.silu(gate.astype(jnp.float32)).astype(h.dtype) * up
        h = jnp.dot(h, lw_s["down_proj"].T)
        return res + h

    @jax.jit
    def forward_full(input_ids, lw, embed, final_norm_w, lm_head_w, cos, sin):
        h = embed[input_ids]
        base_idx = jnp.arange(n_layers, dtype=jnp.int32)
        def scan_body(h, idx):
            lw_s = jax.tree.map(lambda w: w[idx], lw)
            return one_layer(h, lw_s, cos, sin), None
        h, _ = lax.scan(scan_body, h, base_idx)
        h = rms_norm(h, final_norm_w, eps)
        return jnp.dot(h, lm_head_w.T)

    @jax.jit
    def forward_swap(input_ids, lw, embed, final_norm_w, lm_head_w, cos, sin, swap_idx):
        h = embed[input_ids]
        def scan_body(h, idx):
            lw_s = jax.tree.map(lambda w: w[idx], lw)
            return one_layer(h, lw_s, cos, sin), None
        h, _ = lax.scan(scan_body, h, swap_idx)
        h = rms_norm(h, final_norm_w, eps)
        return jnp.dot(h, lm_head_w.T)

    return forward_full, forward_swap


# ── Prompt preparation ───────────────────────────────────────────────────────

def prepare_prompts(tokenizer, seq_len, n_prompts):
    """Load diverse prompts from Wikitext-2."""
    log.info(f"Loading {n_prompts} prompts from Wikitext-2...")
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test",
                          trust_remote_code=True)
        texts = [t for t in ds["text"] if len(t.strip()) > 80]
    except Exception as e:
        log.warning(f"Wikitext load failed ({e}), using built-in prompts")
        texts = []

    # Built-in diverse fallback prompts
    fallback = [
        "The quick brown fox jumps over the lazy dog and then",
        "In quantum mechanics, the wave function describes",
        "Machine learning models can be compressed through",
        "def fibonacci(n):\n    if n <= 1:\n        return n",
        "Climate scientists warn that global temperatures",
        "The stock market rally continued as investors",
        "The ancient Roman Empire at its peak controlled",
        "In the kitchen, she carefully measured two cups",
        "The neural network architecture consists of multiple",
        "According to the theory of general relativity",
        "Recent advances in natural language processing",
        "The algorithm iterates through each element in",
        "Political tensions escalated as world leaders",
        "The pharmaceutical company announced a breakthrough",
        "Shakespeare's most famous tragedy explores themes",
        "The database query optimization reduced latency",
        "Astronomers discovered a new exoplanet orbiting",
        "The recursive function computes the factorial of",
        "Economic indicators suggest a potential recession",
        "The convolutional neural network achieved state-of-the-art",
        "During the medieval period, European kingdoms",
        "The open-source library provides efficient tools for",
        "Researchers found that sleep deprivation affects",
        "The transformer architecture revolutionized the field",
        "In a surprising turn of events, the election",
        "The chemical reaction produces a volatile compound",
        "Modern cryptographic protocols rely on the difficulty",
        "The patient presented with symptoms including fever",
        "Genetic mutations in the BRCA1 gene are associated",
        "The spacecraft successfully entered orbit around",
        "Abstract algebra studies algebraic structures such",
        "The central bank lowered interest rates to stimulate",
        "Photosynthesis converts carbon dioxide and water into",
        "The hash table provides O(1) average lookup time",
        "In cognitive psychology, working memory is defined as",
        "The documentary explores the impact of deforestation",
        "Superconductors exhibit zero electrical resistance below",
        "The API endpoint returns a JSON response containing",
        "Archaeological evidence suggests that early humans",
        "The gradient descent algorithm minimizes the loss function",
        "Renewable energy sources including solar and wind",
        "The philosopher argued that consciousness arises from",
        "Data compression algorithms reduce file sizes by",
        "The clinical trial demonstrated significant efficacy",
        "Black holes are regions of spacetime where gravity",
        "The compiler translates high-level source code into",
        "Demographic trends show an aging population in",
        "The enzyme catalyzes the reaction by lowering the",
        "Game theory analyzes strategic interactions between",
        "The satellite imagery revealed changes in the polar",
    ]

    if len(texts) >= n_prompts:
        # Sample evenly from Wikitext
        step = len(texts) // n_prompts
        texts = texts[::step][:n_prompts]
    else:
        texts = fallback[:n_prompts]

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    ids_list = []
    for t in texts:
        ids = tokenizer.encode(t, max_length=seq_len, truncation=True)[:seq_len]
        ids = ids + [pad_id] * (seq_len - len(ids))
        ids_list.append(ids)
    input_ids = jax.device_put(jnp.array(ids_list[:n_prompts], dtype=jnp.int32))
    log.info(f"  Prepared {input_ids.shape[0]} prompts, seq_len={seq_len}")
    return input_ids


# ── Distance computation ─────────────────────────────────────────────────────

def compute_kl(base_logits, swap_logits):
    """Compute mean KL divergence: KL(base || swap) averaged over prompts."""
    base_lp = jax.nn.log_softmax(base_logits.astype(jnp.float32), axis=-1)
    swap_lp = jax.nn.log_softmax(swap_logits.astype(jnp.float32), axis=-1)
    base_p = jnp.exp(base_lp)
    # KL(p || q) = sum p * (log p - log q)
    kl_per_token = jnp.sum(base_p * (base_lp - swap_lp), axis=-1)
    # Mean over prompts and positions
    return float(jnp.mean(kl_per_token))


def compute_max_kl(base_logits, swap_logits):
    """Compute max KL divergence over prompts (bisimulation supremum)."""
    base_lp = jax.nn.log_softmax(base_logits.astype(jnp.float32), axis=-1)
    swap_lp = jax.nn.log_softmax(swap_logits.astype(jnp.float32), axis=-1)
    base_p = jnp.exp(base_lp)
    kl_per_token = jnp.sum(base_p * (base_lp - swap_lp), axis=-1)
    # Mean over positions, max over prompts
    kl_per_prompt = jnp.mean(kl_per_token, axis=-1)
    return float(jnp.max(kl_per_prompt))


# ── Main experiment ──────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_start = time.time()

    log.info(f"JAX version: {jax.__version__}")
    log.info(f"Devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
    log.info(f"Output: {OUTPUT_DIR}")

    # 1. Load model
    lw, embed, final_norm, lm_head, repo = load_and_stack(MODEL_NAME, DTYPE)
    tokenizer = AutoTokenizer.from_pretrained(repo, local_files_only=True)
    n_layers = _ARCH["n_layers"]
    head_dim = _ARCH["head_dim"]
    rope_theta = _ARCH["rope_theta"]

    # Filter pairs to valid range
    pairs = [(a, b) for a, b in PAIRS if b < n_layers]
    log.info(f"Testing {len(pairs)} layer pairs: {pairs}")

    # 2. Prepare prompts
    input_ids = prepare_prompts(tokenizer, SEQ_LEN, N_PROMPTS)

    # 3. Precompute RoPE: normal and no-rope
    cos_normal, sin_normal = precompute_rope(SEQ_LEN, head_dim, rope_theta, DTYPE)
    cos_normal = jax.device_put(cos_normal)
    sin_normal = jax.device_put(sin_normal)

    cos_norope = jnp.ones_like(cos_normal)   # identity: no rotation
    sin_norope = jnp.zeros_like(sin_normal)  # identity: no rotation

    # 4. Build model
    forward_full, forward_swap = build_model(_ARCH, SEQ_LEN)
    fwd_args_base = (lw, embed, final_norm, lm_head)

    # 5. Compute baselines (both conditions)
    log.info("Computing baseline logits (normal RoPE)...")
    baseline_normal = forward_full(input_ids, *fwd_args_base, cos_normal, sin_normal)
    baseline_normal.block_until_ready()
    log.info("  Normal baseline done.")

    log.info("Computing baseline logits (no-RoPE)...")
    baseline_norope = forward_full(input_ids, *fwd_args_base, cos_norope, sin_norope)
    baseline_norope.block_until_ready()
    log.info("  No-RoPE baseline done.")

    # Measure how different the two baselines are
    baseline_kl = compute_kl(baseline_normal, baseline_norope)
    baseline_max_kl = compute_max_kl(baseline_normal, baseline_norope)
    log.info(f"  Baseline divergence (normal vs no-rope): "
             f"mean_kl={baseline_kl:.4f}, max_kl={baseline_max_kl:.4f}")

    base_idx = jnp.arange(n_layers, dtype=jnp.int32)

    # 6. Compute distances for each pair under both conditions
    results = []
    log.info(f"\n{'='*70}")
    log.info("Computing distances for each pair under normal and no-RoPE conditions")
    log.info(f"{'='*70}")

    for pair_idx, (i, j) in enumerate(pairs):
        log.info(f"\n--- Pair ({i},{j}) [{pair_idx+1}/{len(pairs)}] ---")
        t_pair = time.time()

        # Replacement index: use layer i's weights at both positions i and i+1
        # Swap index = [0,1,...,i,...,i,...,n-1]
        replace_idx = base_idx.at[j].set(i)

        # Interchange index: swap layers i and j
        interchange_idx = base_idx.at[i].set(j).at[j].set(i)

        # ── Normal RoPE condition ──
        # Replacement
        rep_logits_normal = forward_swap(
            input_ids, *fwd_args_base, cos_normal, sin_normal, replace_idx)
        rep_logits_normal.block_until_ready()
        rep_kl_normal = compute_kl(baseline_normal, rep_logits_normal)
        rep_max_kl_normal = compute_max_kl(baseline_normal, rep_logits_normal)

        # Interchange
        int_logits_normal = forward_swap(
            input_ids, *fwd_args_base, cos_normal, sin_normal, interchange_idx)
        int_logits_normal.block_until_ready()
        int_kl_normal = compute_kl(baseline_normal, int_logits_normal)
        int_max_kl_normal = compute_max_kl(baseline_normal, int_logits_normal)

        # ── No-RoPE condition ──
        # Replacement
        rep_logits_norope = forward_swap(
            input_ids, *fwd_args_base, cos_norope, sin_norope, replace_idx)
        rep_logits_norope.block_until_ready()
        rep_kl_norope = compute_kl(baseline_norope, rep_logits_norope)
        rep_max_kl_norope = compute_max_kl(baseline_norope, rep_logits_norope)

        # Interchange
        int_logits_norope = forward_swap(
            input_ids, *fwd_args_base, cos_norope, sin_norope, interchange_idx)
        int_logits_norope.block_until_ready()
        int_kl_norope = compute_kl(baseline_norope, int_logits_norope)
        int_max_kl_norope = compute_max_kl(baseline_norope, int_logits_norope)

        # Compute I/R ratios (using mean KL)
        ir_normal = int_kl_normal / max(rep_kl_normal, 1e-10)
        ir_norope = int_kl_norope / max(rep_kl_norope, 1e-10)

        # Same for max KL
        ir_max_normal = int_max_kl_normal / max(rep_max_kl_normal, 1e-10)
        ir_max_norope = int_max_kl_norope / max(rep_max_kl_norope, 1e-10)

        dt = time.time() - t_pair
        log.info(f"  Normal:  rep={rep_kl_normal:.6f}  int={int_kl_normal:.6f}  "
                 f"I/R={ir_normal:.3f}")
        log.info(f"  No-RoPE: rep={rep_kl_norope:.6f}  int={int_kl_norope:.6f}  "
                 f"I/R={ir_norope:.3f}")
        log.info(f"  I/R change: {ir_normal:.3f} -> {ir_norope:.3f} "
                 f"(delta={ir_norope - ir_normal:+.3f})  [{dt:.1f}s]")

        results.append({
            "layer_a": i,
            "layer_b": j,
            # Normal RoPE
            "replacement_kl_normal": rep_kl_normal,
            "interchange_kl_normal": int_kl_normal,
            "replacement_max_kl_normal": rep_max_kl_normal,
            "interchange_max_kl_normal": int_max_kl_normal,
            "ir_ratio_normal": ir_normal,
            "ir_max_ratio_normal": ir_max_normal,
            # No-RoPE
            "replacement_kl_norope": rep_kl_norope,
            "interchange_kl_norope": int_kl_norope,
            "replacement_max_kl_norope": rep_max_kl_norope,
            "interchange_max_kl_norope": int_max_kl_norope,
            "ir_ratio_norope": ir_norope,
            "ir_max_ratio_norope": ir_max_norope,
            # Deltas
            "ir_ratio_delta": ir_norope - ir_normal,
            "ir_max_ratio_delta": ir_max_norope - ir_max_normal,
        })

    total_time = time.time() - t_start

    # 7. Analysis
    log.info(f"\n{'='*70}")
    log.info("RESULTS SUMMARY")
    log.info(f"{'='*70}")

    ir_normals = [r["ir_ratio_normal"] for r in results]
    ir_noropes = [r["ir_ratio_norope"] for r in results]
    ir_deltas = [r["ir_ratio_delta"] for r in results]

    mean_ir_normal = np.mean(ir_normals)
    mean_ir_norope = np.mean(ir_noropes)
    mean_delta = np.mean(ir_deltas)

    log.info(f"\nMean I/R ratio (normal RoPE):  {mean_ir_normal:.4f}")
    log.info(f"Mean I/R ratio (no-RoPE):      {mean_ir_norope:.4f}")
    log.info(f"Mean I/R delta:                {mean_delta:+.4f}")

    # Interpretation
    if mean_delta < -0.1:
        interpretation = ("ROPE_CAUSAL: Protocol gap SHRINKS without RoPE. "
                          "RoPE rotation is mechanistically responsible.")
    elif mean_delta > 0.1:
        interpretation = ("ROPE_AMPLIFYING: Protocol gap GROWS without RoPE. "
                          "RoPE partially masks the gap; weight specialization is the cause.")
    else:
        interpretation = ("ROPE_NEUTRAL: Protocol gap is UNCHANGED without RoPE. "
                          "Mechanism is layer weight specialization, not positional encoding.")

    log.info(f"\nInterpretation: {interpretation}")
    log.info(f"Baseline divergence (normal vs no-rope): "
             f"mean={baseline_kl:.4f}, max={baseline_max_kl:.4f}")

    # Print table
    log.info(f"\n{'Pair':<10} {'R-norm':>8} {'I-norm':>8} {'I/R-n':>7} "
             f"{'R-noRP':>8} {'I-noRP':>8} {'I/R-nr':>7} {'delta':>7}")
    log.info("-" * 73)
    for r in results:
        log.info(f"({r['layer_a']:2d},{r['layer_b']:2d})   "
                 f"{r['replacement_kl_normal']:8.4f} "
                 f"{r['interchange_kl_normal']:8.4f} "
                 f"{r['ir_ratio_normal']:7.3f} "
                 f"{r['replacement_kl_norope']:8.4f} "
                 f"{r['interchange_kl_norope']:8.4f} "
                 f"{r['ir_ratio_norope']:7.3f} "
                 f"{r['ir_ratio_delta']:+7.3f}")

    # 8. Save results
    output = {
        "model": MODEL_NAME,
        "n_prompts": N_PROMPTS,
        "seq_len": SEQ_LEN,
        "n_layers": n_layers,
        "pairs": [list(p) for p in pairs],
        "baseline_divergence_mean_kl": baseline_kl,
        "baseline_divergence_max_kl": baseline_max_kl,
        "mean_ir_ratio_normal": mean_ir_normal,
        "mean_ir_ratio_norope": mean_ir_norope,
        "mean_ir_delta": mean_delta,
        "interpretation": interpretation,
        "results": results,
        "total_time_s": total_time,
        "arch": {k: v for k, v in _ARCH.items()},
    }

    out_path = os.path.join(OUTPUT_DIR, "rope_counterfactual_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nResults saved to {out_path}")
    log.info(f"Total time: {total_time:.0f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()
