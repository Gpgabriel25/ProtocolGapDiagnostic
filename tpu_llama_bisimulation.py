#!/usr/bin/env python3
"""
tpu_llama_bisimulation.py — Full bisimulation protocol on Llama-3.1-8B.

Same protocol as tpu_replacement_pruning.py (Qwen3-8B) but targeting
Llama-3.1-8B for cross-family replication. Computes:
  1. Pairwise replacement + interchange distances (all 31 adjacent pairs)
  2. BI scores
  3. Auto-selects layers via interchange/replacement/BI
  4. Wikitext-2 PPL evaluation for all configurations

Matched evaluation parameters: same prompt set, same evaluator window,
same seq_len, same stride, same Wikitext-2 subset.

Usage on TPU v6e-16 (single-host mode):
  screen -dmS llama bash -c 'source ~/venv311/bin/activate && \
    HF_TOKEN=$(cat /tmp/hf_token) \
    REPORT_DIR=/tmp/llama_bisim python3 /tmp/llama_bisim.py 2>&1 | \
    tee /tmp/llama_bisim/run.log'
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

MODEL_NAME = os.environ.get("MODEL", "meta-llama/Llama-3.1-8B")
DTYPE = jnp.bfloat16
REPORT_DIR = os.environ.get("REPORT_DIR", "/tmp/llama_bisim")
DIST_SEQ_LEN = 64        # matched to Qwen3-8B protocol
PPL_SEQ_LEN = 512        # matched to Qwen3-8B protocol
PPL_STRIDE = 256          # matched
PPL_MAX_WORDS = 20000     # matched
N_DISTANCE_PROMPTS = 100  # matched

_ARCH = {}


# ── Weight loading ───────────────────────────────────────────────────────────

def load_and_stack(model_name, dtype=jnp.bfloat16):
    log.info(f"Downloading/loading {model_name}...")
    token = os.environ.get("HF_TOKEN")
    repo = snapshot_download(
        model_name,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*", "*.model"],
        token=token,
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

    log.info(f"Stacking per-layer weights (bf16), qk_norm={has_qk}...")
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

# Exactly matched prompt set from Qwen3-8B protocol
_PROMPT_POOL = [
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
    "A well-designed user interface should prioritize",
    "The fossil record provides evidence for the evolution",
    "Bayesian inference updates prior beliefs given new",
    "The orchestra performed Beethoven's Fifth Symphony",
    "Reinforcement learning agents learn optimal policies by",
    "The volcanic eruption deposited a thick layer of ash",
    "Object-oriented programming encapsulates data and behavior",
    "The treaty established new trade agreements between",
    "Stem cell research has opened new possibilities for",
    "The operating system manages hardware resources and",
    "Tectonic plate movements cause earthquakes along fault",
    "The encryption algorithm ensures that data remains",
    "Coral reefs support a diverse ecosystem of marine",
    "The peer-reviewed journal article presents findings on",
    "Autonomous vehicles use sensor fusion to navigate",
    "The Renaissance period saw a flourishing of art and science",
    "Functional programming emphasizes immutable data and pure",
    "The archaeological dig uncovered artifacts dating back",
    "Quantum entanglement allows particles to be correlated",
    "The supply chain disruption affected manufacturing across",
    "Natural selection acts on heritable variation within",
    "The microprocessor executes instructions stored in memory",
    "Inflation has eroded purchasing power in many developing",
    "The double helix structure of DNA was discovered by",
    "Cloud computing provides scalable resources on demand",
    "The United Nations Security Council convened to discuss",
    "Antibiotic resistance is an increasing global health",
    "The sorting algorithm has a worst-case time complexity of",
    "Marine biologists observed unusual migration patterns in",
    "The social media platform implemented new content moderation",
    "Thermodynamics describes the relationships between heat",
    "The legislative body passed a controversial new bill",
    "Deep learning models require large amounts of labeled",
    "The ancient Egyptian civilization developed along the",
    "Topological spaces generalize the notion of continuity",
    "The humanitarian crisis displaced millions of civilians",
    "Graph neural networks operate on structured relational",
    "The semiconductor industry faces supply constraints in",
    "Epidemiological studies traced the outbreak to a single",
    "The distributed system ensures fault tolerance through",
    "Impressionist painters captured the play of light and",
    "Nuclear fusion promises virtually unlimited clean energy",
    "The garbage collector automatically frees memory that is",
    "Archaeological carbon dating places the settlement at",
    "The privacy regulation requires explicit user consent for",
    "Fluid dynamics equations describe the motion of liquids",
    "The startup raised a Series B round of funding to expand",
    "CRISPR gene editing technology enables precise modifications",
    "The kernel module handles system calls from user space",
    "International trade agreements reduce tariff barriers",
]


DIST_BATCH_SIZE = 10  # prompts per batch to avoid OOM with large vocab


def _batched_kl(fwd_fn, input_ids_batches, fwd_args, base_lp_batches, base_p_batches):
    """Compute mean KL between baseline and fwd_fn output over batches."""
    total_kl = 0.0
    total_tokens = 0
    for ids_b, blp_b, bp_b in zip(input_ids_batches, base_lp_batches, base_p_batches):
        logits = fwd_fn(ids_b, *fwd_args)
        logits.block_until_ready()
        slp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        kl_batch = jnp.sum(bp_b * (blp_b - slp), axis=-1)  # (B, S)
        total_kl += float(jnp.sum(kl_batch))
        total_tokens += kl_batch.size
    return total_kl / total_tokens


def compute_all_distances(lw, embed, final_norm, lm_head, tokenizer, n_prompts):
    """Compute replacement and interchange distances for all adjacent pairs.
    Uses batched processing to avoid OOM with large vocab models."""
    log.info("=== Computing pairwise distances ===")
    n_layers = _ARCH["n_layers"]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    seq_len = DIST_SEQ_LEN

    test_prompts = _PROMPT_POOL[:n_prompts]

    ids_list = []
    for p in test_prompts:
        ids = tokenizer.encode(p, max_length=seq_len, truncation=True)
        ids = ids[:seq_len]
        ids = ids + [pad_id] * (seq_len - len(ids))
        ids_list.append(ids)

    # Split into batches
    all_ids = jnp.array(ids_list, dtype=jnp.int32)
    n_batches = (len(ids_list) + DIST_BATCH_SIZE - 1) // DIST_BATCH_SIZE
    id_batches = [jax.device_put(all_ids[i*DIST_BATCH_SIZE:(i+1)*DIST_BATCH_SIZE])
                  for i in range(n_batches)]
    log.info(f"  {n_prompts} prompts in {n_batches} batches of ≤{DIST_BATCH_SIZE}")

    cos_d, sin_d = precompute_rope(seq_len, _ARCH["head_dim"], _ARCH["rope_theta"], DTYPE)
    cos_d = jax.device_put(cos_d)
    sin_d = jax.device_put(sin_d)

    fwd_full_d, _, fwd_swap_d = build_model(_ARCH, seq_len)

    # Baseline logits (batched)
    log.info("  Computing baseline logits (batched)...")
    base_lp_batches = []
    base_p_batches = []
    for ids_b in id_batches:
        logits = fwd_full_d(ids_b, lw, embed, final_norm, lm_head, cos_d, sin_d)
        logits.block_until_ready()
        blp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        bp = jnp.exp(blp)
        base_lp_batches.append(blp)
        base_p_batches.append(bp)
    log.info("  Baseline computed.")

    base_idx = jnp.arange(n_layers, dtype=jnp.int32)
    results = []

    # Replacement distances
    log.info(f"  Computing replacement distances for {n_layers-1} pairs...")
    t0 = time.time()
    for i in range(n_layers - 1):
        replace_idx = base_idx.at[i].set(i + 1)
        fwd_args = (lw, embed, final_norm, lm_head, cos_d, sin_d, replace_idx)
        kl = _batched_kl(fwd_swap_d, id_batches, fwd_args,
                         base_lp_batches, base_p_batches)
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
        fwd_args = (lw, embed, final_norm, lm_head, cos_d, sin_d, swap_idx)
        kl = _batched_kl(fwd_swap_d, id_batches, fwd_args,
                         base_lp_batches, base_p_batches)
        results[i]["interchange_kl"] = kl
    t_int = time.time() - t0
    log.info(f"  Interchange distances: {t_int:.1f}s")

    # Free baseline tensors
    del base_lp_batches, base_p_batches
    gc.collect()

    return results, t_rep, t_int


# ── BI score computation ─────────────────────────────────────────────────────

def compute_bi_scores(lw, embed, tokenizer, n_prompts=20):
    """Compute Block Influence scores: BI(ℓ) = 1 - cos_sim(h_in, h_out)."""
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

    test_prompts = _PROMPT_POOL[:n_prompts]
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

    h = embed[input_ids]
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
    candidates = [(scores[i], i) for i in range(1, n_layers - 1)]
    if mode == "lowest":
        candidates.sort()
    else:
        candidates.sort(reverse=True)
    return sorted([layer for _, layer in candidates[:n]])


def select_replacement_layers(distances, n, mode="lowest_replacement"):
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
    total_nll = 0.0
    total_tokens = 0
    n_windows = 0
    total_len = len(token_ids)

    for begin in range(0, total_len - 1, stride):
        end = min(begin + seq_len, total_len)
        chunk = token_ids[begin:end]

        if begin == 0:
            target_start = 0
        else:
            target_start = seq_len - stride if end - begin == seq_len else 0

        if len(chunk) < 2:
            continue

        pad_len = seq_len - len(chunk)
        padded = chunk + [0] * pad_len
        input_j = jnp.array([padded], dtype=jnp.int32)

        logits = forward_fn(input_j, *fwd_args)
        logits = logits[0]
        log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)

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
    log.info(f"Model: {MODEL_NAME}")

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

    # ── Phase 2: Layer Selection (auto-computed from distances) ──
    log.info("\n" + "=" * 70)
    log.info("PHASE 2: Layer Selection (auto-computed)")
    log.info("=" * 70)

    configs = {}

    # Interchange-guided (computed from distance matrix)
    for n in [1, 3, 5]:
        layers = select_interchange_from_distances(distances, n)
        configs[f"interchange_n{n}"] = layers

    # Replacement-guided
    for n in [1, 3, 5]:
        layers = select_replacement_layers(distances, n, mode="lowest_replacement")
        configs[f"replacement_n{n}"] = layers

    # BI-guided
    for n in [1, 3, 5]:
        layers = greedy_select(bi_scores, n, n_layers, mode="lowest")
        configs[f"bi_n{n}"] = layers

    # Random baseline
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

    # ── Protocol gap analysis ──
    log.info("\n" + "=" * 70)
    log.info("PROTOCOL GAP ANALYSIS")
    log.info("=" * 70)

    # Compute I/R ratios from distances
    for d in distances:
        d["i_r_ratio"] = d["interchange_kl"] / max(d["replacement_kl"], 1e-10)

    ir_ratios = [d["i_r_ratio"] for d in distances]
    log.info(f"  Mean I/R ratio: {np.mean(ir_ratios):.3f}")
    log.info(f"  Median I/R ratio: {np.median(ir_ratios):.3f}")
    log.info(f"  Min I/R ratio: {min(ir_ratios):.3f} (pair {distances[np.argmin(ir_ratios)]['layer_a']},{distances[np.argmin(ir_ratios)]['layer_b']})")
    log.info(f"  Max I/R ratio: {max(ir_ratios):.3f} (pair {distances[np.argmax(ir_ratios)]['layer_a']},{distances[np.argmax(ir_ratios)]['layer_b']})")
    log.info(f"  Pairs with I/R > 1 (interchange harder): {sum(1 for r in ir_ratios if r > 1)}/{len(ir_ratios)}")

    # Compare interchange vs replacement PPL advantage at each n
    for n in [1, 3, 5]:
        int_key = f"interchange_n{n}"
        rep_key = f"replacement_n{n}"
        if int_key in ppl_results and rep_key in ppl_results:
            int_ppl = ppl_results[int_key]["ppl"]
            rep_ppl = ppl_results[rep_key]["ppl"]
            advantage = rep_ppl / int_ppl
            log.info(f"  n={n}: interchange PPL={int_ppl:.2f}, replacement PPL={rep_ppl:.2f}, advantage={advantage:.2f}×")

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
        "protocol_gap": {
            "mean_ir_ratio": float(np.mean(ir_ratios)),
            "median_ir_ratio": float(np.median(ir_ratios)),
            "min_ir_ratio": float(min(ir_ratios)),
            "max_ir_ratio": float(max(ir_ratios)),
            "pairs_above_1": int(sum(1 for r in ir_ratios if r > 1)),
            "total_pairs": len(ir_ratios),
        },
        "config": {
            "dist_seq_len": DIST_SEQ_LEN,
            "ppl_seq_len": PPL_SEQ_LEN,
            "ppl_stride": PPL_STRIDE,
            "ppl_max_words": PPL_MAX_WORDS,
            "n_distance_prompts": N_DISTANCE_PROMPTS,
        },
    }

    save_path = os.path.join(REPORT_DIR, "llama_bisimulation_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\nResults saved to {save_path}")
    log.info("DONE")


if __name__ == "__main__":
    main()
