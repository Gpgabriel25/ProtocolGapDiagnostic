#!/usr/bin/env python3
"""
tpu_replacement_pruning.py — Replacement-guided layer removal on Qwen3-8B.

Computes replacement distances for all 35 adjacent pairs, selects layers for
removal via replacement-guided vs interchange-guided selection, then measures
Wikitext-2 perplexity for each configuration.

Also computes BI scores (cosine-similarity-based block influence) for comparison.

Usage on TPU v6e-16 (single-host mode):
  screen -dmS rep bash -c 'source ~/venv311/bin/activate && \
    REPORT_DIR=/tmp/replacement_pruning python3 tpu_replacement_pruning.py 2>&1 | \
    tee /tmp/replacement_pruning/run.log'
"""

import os, sys, json, time, logging, gc, math
import numpy as np
from pathlib import Path

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
REPORT_DIR = os.environ.get("REPORT_DIR", "/tmp/replacement_pruning")
DIST_SEQ_LEN = 64        # for distance computation (speed)
PPL_SEQ_LEN = 512        # for Wikitext PPL evaluation
PPL_STRIDE = 256          # sliding window stride
PPL_MAX_WORDS = 20000     # max words from Wikitext-2
N_DISTANCE_PROMPTS = 100  # prompts for distance computation

# Interchange-guided skip configs from prior experiments (Table 2)
INTERCHANGE_SKIP = {
    "interchange_n1": [17],
    "interchange_n3": [15, 17, 20],
    "interchange_n5": [15, 17, 18, 19, 20],
}

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
    log.info(f"{model_name}: {n_layers}L h={hidden} heads={n_heads}/{n_kv} inter={inter}")

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
    def forward_skip(input_ids, lw, embed, final_norm_w, lm_head_w, cos, sin, layer_mask):
        h = embed[input_ids]
        base_idx = jnp.arange(n_layers, dtype=jnp.int32)
        def scan_body(h, idx):
            lw_s = jax.tree.map(lambda w: w[idx], lw)
            h_new = one_layer(h, lw_s, cos, sin)
            keep = layer_mask[idx]
            return jnp.where(keep, h_new, h), None
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

    return forward_full, forward_skip, forward_swap


# ── Distance computation ─────────────────────────────────────────────────────

def compute_all_distances(lw, embed, final_norm, lm_head, tokenizer, n_prompts):
    """Compute replacement and interchange distances for all adjacent pairs."""
    log.info("=== Computing pairwise distances ===")
    n_layers = _ARCH["n_layers"]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    seq_len = DIST_SEQ_LEN

    # Diverse prompts
    test_prompts = [
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
    ][:n_prompts]

    ids_list = []
    for p in test_prompts:
        ids = tokenizer.encode(p, max_length=seq_len, truncation=True)
        ids = ids[:seq_len]
        ids = ids + [pad_id] * (seq_len - len(ids))
        ids_list.append(ids)
    input_ids = jax.device_put(jnp.array(ids_list, dtype=jnp.int32))

    cos_d, sin_d = precompute_rope(seq_len, _ARCH["head_dim"], _ARCH["rope_theta"], DTYPE)
    cos_d = jax.device_put(cos_d)
    sin_d = jax.device_put(sin_d)

    fwd_full_d, _, fwd_swap_d = build_model(_ARCH, seq_len)

    # Baseline logits
    log.info("  Computing baseline logits...")
    fwd_args = (lw, embed, final_norm, lm_head, cos_d, sin_d)
    baseline_logits = fwd_full_d(input_ids, *fwd_args)
    baseline_logits.block_until_ready()
    base_lp = jax.nn.log_softmax(baseline_logits.astype(jnp.float32), axis=-1)
    base_p = jnp.exp(base_lp)

    base_idx = jnp.arange(n_layers, dtype=jnp.int32)
    results = []

    # Replacement distances
    log.info(f"  Computing replacement distances for {n_layers-1} pairs...")
    t0 = time.time()
    for i in range(n_layers - 1):
        replace_idx = base_idx.at[i].set(i + 1)
        swap_logits = fwd_swap_d(input_ids, lw, embed, final_norm, lm_head,
                                  cos_d, sin_d, replace_idx)
        swap_logits.block_until_ready()
        swap_lp = jax.nn.log_softmax(swap_logits.astype(jnp.float32), axis=-1)
        kl = float(jnp.mean(jnp.sum(base_p * (base_lp - swap_lp), axis=-1)))
        results.append({"layer_a": i, "layer_b": i + 1, "replacement_kl": kl})
        if (i + 1) % 5 == 0:
            log.info(f"    Pair ({i},{i+1}): replacement_kl={kl:.6f}")
    t_rep = time.time() - t0
    log.info(f"  Replacement distances: {t_rep:.1f}s")

    # Interchange distances
    log.info(f"  Computing interchange distances for {n_layers-1} pairs...")
    t0 = time.time()
    for i in range(n_layers - 1):
        swap_idx = base_idx.at[i].set(i + 1).at[i + 1].set(i)
        swap_logits = fwd_swap_d(input_ids, lw, embed, final_norm, lm_head,
                                  cos_d, sin_d, swap_idx)
        swap_logits.block_until_ready()
        swap_lp = jax.nn.log_softmax(swap_logits.astype(jnp.float32), axis=-1)
        kl = float(jnp.mean(jnp.sum(base_p * (base_lp - swap_lp), axis=-1)))
        results[i]["interchange_kl"] = kl
    t_int = time.time() - t0
    log.info(f"  Interchange distances: {t_int:.1f}s")

    return results, t_rep, t_int


# ── BI score computation (JAX) ───────────────────────────────────────────────

def compute_bi_scores(lw, embed, tokenizer, n_prompts=50):
    """Compute Block Influence scores: BI(ℓ) = 1 - cos_sim(h_in, h_out) for each layer.

    Uses manual layer-by-layer forward to capture hidden states.
    """
    log.info("=== Computing BI scores ===")
    n_layers = _ARCH["n_layers"]
    n_heads = _ARCH["n_heads"]
    n_kv = _ARCH["n_kv"]
    head_dim = _ARCH["head_dim"]
    eps = _ARCH["eps"]
    has_qk = _ARCH["has_qk_norm"]
    kv_rep = n_heads // n_kv
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    seq_len = DIST_SEQ_LEN

    # Reuse same prompts
    test_prompts = [
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
    ][:n_prompts]

    ids_list = []
    for p in test_prompts:
        ids = tokenizer.encode(p, max_length=seq_len, truncation=True)[:seq_len]
        ids = ids + [pad_id] * (seq_len - len(ids))
        ids_list.append(ids)
    input_ids = jnp.array(ids_list, dtype=jnp.int32)

    cos_d, sin_d = precompute_rope(seq_len, head_dim, _ARCH["rope_theta"], DTYPE)
    cos_d = jax.device_put(cos_d)
    sin_d = jax.device_put(sin_d)
    _cmask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))

    # Manual layer-by-layer forward to capture hidden states
    h = embed[input_ids]  # (B, S, H)
    bi_scores = np.zeros(n_layers)

    for layer_idx in range(n_layers):
        h_in = h
        lw_s = jax.tree.map(lambda w: w[layer_idx], lw)

        B, S, H = h.shape
        res = h
        h_n = rms_norm(h, lw_s["input_ln"], eps)
        q = jnp.dot(h_n, lw_s["q_proj"].T)
        k = jnp.dot(h_n, lw_s["k_proj"].T)
        v = jnp.dot(h_n, lw_s["v_proj"].T)
        q = q.reshape(B, S, n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, n_kv, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, n_kv, head_dim).transpose(0, 2, 1, 3)
        if has_qk:
            q = rms_norm(q, lw_s["q_norm"], eps)
            k = rms_norm(k, lw_s["k_norm"], eps)
        q, k = apply_rope(q, k, cos_d, sin_d)
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
        h_n = rms_norm(h, lw_s["post_ln"], eps)
        gate = jnp.dot(h_n, lw_s["gate_proj"].T)
        up = jnp.dot(h_n, lw_s["up_proj"].T)
        h_n = jax.nn.silu(gate.astype(jnp.float32)).astype(h.dtype) * up
        h_n = jnp.dot(h_n, lw_s["down_proj"].T)
        h = res + h_n

        h_out = h
        # BI = 1 - mean cosine similarity
        h_in_flat = h_in.reshape(-1).astype(jnp.float32)
        h_out_flat = h_out.reshape(-1).astype(jnp.float32)
        cos_sim = float(jnp.sum(h_in_flat * h_out_flat) /
                        (jnp.linalg.norm(h_in_flat) * jnp.linalg.norm(h_out_flat) + 1e-8))
        bi_scores[layer_idx] = 1.0 - cos_sim

        if (layer_idx + 1) % 6 == 0:
            log.info(f"  BI layer {layer_idx}: {bi_scores[layer_idx]:.6f}")

    log.info(f"  BI scores range: [{bi_scores.min():.6f}, {bi_scores.max():.6f}]")
    return bi_scores


# ── Layer selection ──────────────────────────────────────────────────────────

def greedy_select(scores, n, n_layers, mode="lowest"):
    """Select n layers with lowest (or highest if mode='highest') scores.
    Skip boundary layers 0 and n_layers-1. No adjacency constraint."""
    candidates = [(scores[i], i) for i in range(1, n_layers - 1)]
    if mode == "lowest":
        candidates.sort()
    else:
        candidates.sort(reverse=True)
    return sorted([layer for _, layer in candidates[:n]])


def select_replacement_layers(distances, n, mode="lowest_replacement"):
    """Select n layers to remove based on replacement distance.

    mode='lowest_replacement': layers whose replacement causes LEAST distortion
        → replacement says these are redundant (wrong intuition, should be bad picks)
    mode='highest_replacement': layers with HIGHEST replacement distance
        → replacement says these are critical
    """
    # Each distance entry has layer_a, layer_b. For removal, we pick layer_a
    # (the one being replaced by its neighbor).
    scored = [(d["replacement_kl"], d["layer_a"]) for d in distances
              if 0 < d["layer_a"] < _ARCH["n_layers"] - 1]
    if mode == "lowest_replacement":
        scored.sort()
    else:
        scored.sort(reverse=True)
    selected = []
    for _, layer in scored:
        if layer not in selected and len(selected) < n:
            selected.append(layer)
    return sorted(selected)


def select_interchange_from_distances(distances, n):
    """Select n layers to remove: lowest interchange distance (most swappable)."""
    scored = [(d["interchange_kl"], d["layer_a"]) for d in distances
              if 0 < d["layer_a"] < _ARCH["n_layers"] - 1]
    scored.sort()
    selected = []
    for _, layer in scored:
        if layer not in selected and len(selected) < n:
            selected.append(layer)
    return sorted(selected)


# ── Wikitext-2 PPL evaluation ────────────────────────────────────────────────

def load_wikitext2_tokens(tokenizer, max_words=PPL_MAX_WORDS):
    """Load and tokenize Wikitext-2 test set."""
    from datasets import load_dataset
    log.info("Loading Wikitext-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(s for s in ds["text"] if s.strip())
    words = text.split()
    if max_words and len(words) > max_words:
        text = " ".join(words[:max_words])
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    log.info(f"  Wikitext-2: {len(words)} words, {len(token_ids)} tokens")
    return token_ids


def eval_ppl(forward_fn, fwd_args, token_ids, seq_len=PPL_SEQ_LEN,
             stride=PPL_STRIDE, label=""):
    """Evaluate perplexity with sliding window + overlap masking."""
    total_nll = 0.0
    total_tokens = 0
    n_windows = 0
    total_len = len(token_ids)

    for begin in range(0, total_len - 1, stride):
        end = min(begin + seq_len, total_len)
        chunk = token_ids[begin:end]

        # Overlap region: only count loss on new tokens (after stride)
        if begin == 0:
            target_start = 0
        else:
            target_start = seq_len - stride if end - begin == seq_len else 0

        if len(chunk) < 2:
            continue

        # Pad to seq_len
        pad_len = seq_len - len(chunk)
        padded = chunk + [0] * pad_len
        input_j = jnp.array([padded], dtype=jnp.int32)

        logits = forward_fn(input_j, *fwd_args)
        logits = logits[0]  # (S, V)
        log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)

        # Compute NLL for target positions
        actual_len = len(chunk)
        for pos in range(max(target_start, 0), actual_len - 1):
            next_tok = chunk[pos + 1]
            nll = -float(log_probs[pos, next_tok])
            total_nll += nll
            total_tokens += 1

        n_windows += 1
        if n_windows % 20 == 0:
            cur_ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float('inf')
            log.info(f"  {label} window {n_windows}: running_ppl={cur_ppl:.2f} ({total_tokens} tokens)")

        if end >= total_len:
            break

    ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float('inf')
    log.info(f"  {label} final PPL: {ppl:.4f} ({total_tokens} tokens, {n_windows} windows)")
    return ppl, total_tokens, n_windows


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    log.info(f"Devices: {jax.device_count()} × {jax.devices()[0].device_kind}")
    log.info(f"Report dir: {REPORT_DIR}")

    # Load model
    lw, embed, final_norm, lm_head, repo = load_and_stack(MODEL_NAME, dtype=DTYPE)
    n_layers = _ARCH["n_layers"]
    tokenizer = AutoTokenizer.from_pretrained(repo, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Phase 1: Compute pairwise distances ──
    log.info("\n" + "=" * 70)
    log.info("PHASE 1: Pairwise Distance Computation")
    log.info("=" * 70)

    distances, t_rep, t_int = compute_all_distances(
        lw, embed, final_norm, lm_head, tokenizer, N_DISTANCE_PROMPTS)

    # Print distance summary
    rep_sorted = sorted(distances, key=lambda r: r["replacement_kl"])
    int_sorted = sorted(distances, key=lambda r: r["interchange_kl"])

    log.info("\nTop-5 by REPLACEMENT distance (lowest = most replaceable):")
    for r in rep_sorted[:5]:
        log.info(f"  ({r['layer_a']},{r['layer_b']}): rep={r['replacement_kl']:.6f} int={r['interchange_kl']:.6f}")

    log.info("\nTop-5 by INTERCHANGE distance (lowest = most swappable):")
    for r in int_sorted[:5]:
        log.info(f"  ({r['layer_a']},{r['layer_b']}): rep={r['replacement_kl']:.6f} int={r['interchange_kl']:.6f}")

    # ── Phase 1b: BI scores ──
    log.info("\n" + "=" * 70)
    log.info("PHASE 1b: BI Score Computation")
    log.info("=" * 70)

    bi_scores = compute_bi_scores(lw, embed, tokenizer, n_prompts=20)

    # ── Phase 2: Select layers for each method ──
    log.info("\n" + "=" * 70)
    log.info("PHASE 2: Layer Selection")
    log.info("=" * 70)

    configs = {}

    # Interchange-guided (from paper Table 2)
    for name, layers in INTERCHANGE_SKIP.items():
        configs[name] = layers

    # Replacement-guided: lowest replacement distance → "most replaceable"
    for n in [1, 3, 5]:
        layers = select_replacement_layers(distances, n, mode="lowest_replacement")
        configs[f"replacement_n{n}"] = layers

    # BI-guided: lowest BI score → "least influential"
    for n in [1, 3, 5]:
        layers = greedy_select(bi_scores, n, n_layers, mode="lowest")
        configs[f"bi_n{n}"] = layers

    # Random baseline (fixed seed)
    rng = np.random.default_rng(42)
    interior = list(range(1, n_layers - 1))
    for n in [1, 3, 5]:
        layers = sorted(rng.choice(interior, n, replace=False).tolist())
        configs[f"random_n{n}"] = layers

    log.info("\nLayer selections:")
    for name, layers in sorted(configs.items()):
        log.info(f"  {name}: skip {layers}")

    # ── Phase 3: Wikitext-2 PPL evaluation ──
    log.info("\n" + "=" * 70)
    log.info("PHASE 3: Wikitext-2 Perplexity Evaluation")
    log.info("=" * 70)

    token_ids = load_wikitext2_tokens(tokenizer)

    # Build model for PPL seq_len
    cos_p, sin_p = precompute_rope(PPL_SEQ_LEN, _ARCH["head_dim"], _ARCH["rope_theta"], DTYPE)
    cos_p = jax.device_put(cos_p)
    sin_p = jax.device_put(sin_p)
    fwd_full_p, fwd_skip_p, _ = build_model(_ARCH, PPL_SEQ_LEN)

    # JIT warmup
    log.info("JIT warmup (PPL model)...")
    warmup_ids = jnp.zeros((1, PPL_SEQ_LEN), dtype=jnp.int32)
    fwd_args_full = (lw, embed, final_norm, lm_head, cos_p, sin_p)
    out = fwd_full_p(warmup_ids, *fwd_args_full)
    out.block_until_ready()
    warmup_mask = jnp.ones(n_layers, dtype=jnp.bool_)
    fwd_args_skip = (lw, embed, final_norm, lm_head, cos_p, sin_p, warmup_mask)
    out = fwd_skip_p(warmup_ids, *fwd_args_skip)
    out.block_until_ready()
    log.info("JIT warmup complete.")
    del out; gc.collect()

    # Baseline PPL
    log.info("\n--- Baseline (no skip) ---")
    t0 = time.time()
    baseline_ppl, baseline_toks, _ = eval_ppl(
        fwd_full_p, fwd_args_full, token_ids, label="baseline")
    baseline_time = time.time() - t0

    ppl_results = {
        "baseline": {
            "ppl": baseline_ppl,
            "tokens": baseline_toks,
            "skip_layers": [],
            "elapsed_s": baseline_time,
        }
    }

    # Evaluate each config
    for name, skip_layers in sorted(configs.items()):
        log.info(f"\n--- {name}: skip {skip_layers} ---")
        layer_mask = jnp.ones(n_layers, dtype=jnp.bool_)
        for idx in skip_layers:
            layer_mask = layer_mask.at[idx].set(False)
        fwd_args = (lw, embed, final_norm, lm_head, cos_p, sin_p, layer_mask)
        t0 = time.time()
        ppl, toks, _ = eval_ppl(fwd_skip_p, fwd_args, token_ids, label=name)
        elapsed = time.time() - t0
        delta_pct = (ppl - baseline_ppl) / baseline_ppl * 100
        ppl_results[name] = {
            "ppl": ppl,
            "ppl_delta_pct": delta_pct,
            "tokens": toks,
            "skip_layers": skip_layers,
            "n_skip": len(skip_layers),
            "elapsed_s": elapsed,
        }
        log.info(f"  {name}: PPL={ppl:.2f} (Δ={delta_pct:+.1f}%)")

    # ── Summary ──
    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info(f"{'Method':<25} {'Skip Layers':<25} {'PPL':>8} {'ΔPPL%':>8}")
    log.info("-" * 70)
    log.info(f"{'baseline':<25} {'[]':<25} {baseline_ppl:>8.2f} {'0.0':>8}")
    for name in sorted(ppl_results.keys()):
        if name == "baseline":
            continue
        r = ppl_results[name]
        log.info(f"{name:<25} {str(r['skip_layers']):<25} {r['ppl']:>8.2f} {r['ppl_delta_pct']:>+8.1f}%")

    # ── Save results ──
    all_results = {
        "model": MODEL_NAME,
        "n_layers": n_layers,
        "architecture": {k: v for k, v in _ARCH.items()
                         if not isinstance(v, (np.ndarray, jnp.ndarray))},
        "device": str(jax.devices()[0]),
        "device_count": jax.device_count(),
        "dtype": "bfloat16",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pairwise_distances": distances,
        "bi_scores": {str(i): float(bi_scores[i]) for i in range(n_layers)},
        "layer_selections": {k: v for k, v in configs.items()},
        "ppl_results": ppl_results,
        "config": {
            "dist_seq_len": DIST_SEQ_LEN,
            "ppl_seq_len": PPL_SEQ_LEN,
            "ppl_stride": PPL_STRIDE,
            "ppl_max_words": PPL_MAX_WORDS,
            "n_distance_prompts": N_DISTANCE_PROMPTS,
        },
    }

    save_path = os.path.join(REPORT_DIR, "replacement_pruning_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
