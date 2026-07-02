#!/usr/bin/env python3
"""
Matched-evaluator experiment: Mistral-7B-v0.1 layer removal methods evaluated
under a single unified held-out-corpus evaluator (WikiText-2 test, 5K words, 512/256 window).

Same evaluator protocol as matched_eval_qwen3.py / matched_eval_llama.py for cross-model
consistency.

Usage:
  MODEL_DIR=/path/to/mistral python matched_eval_mistral.py
  REPORT_DIR=/tmp/mistral_eval python matched_eval_mistral.py

NOTE: Layer indices 0-based (Mistral-7B-v0.1 has 32 layers: 0-31)
CONFIGS will be auto-populated from bisimulation results JSON if BISIM_JSON env var is set.
If BISIM_JSON is not set, uses hardcoded placeholder CONFIGS that will be overwritten
after bisimulation results are available.

Target: TPU v6e-8, JAX + bf16
Expected runtime: ~45-70 minutes total (Mistral-7B-v0.1 is slightly smaller than Llama-3.1-8B)
"""

import os, sys, json, time, logging, gc, math
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

MODEL_NAME = os.environ.get("MODEL", "mistralai/Mistral-7B-v0.1")
DTYPE = jnp.bfloat16
REPORT_DIR = os.environ.get("REPORT_DIR", "/tmp/matched_eval_mistral")

# ── Evaluation corpus (same contract as matched_eval_qwen3.py / matched_eval_llama.py) ──
MAX_WORDS = int(os.environ.get("EVAL_MAX_WORDS", "5000"))
WINDOW = 512
STRIDE = 256
EVAL_DATASET_NAME = os.environ.get("EVAL_DATASET_NAME", "wikitext")
EVAL_DATASET_CONFIG = os.environ.get("EVAL_DATASET_CONFIG", "wikitext-2-raw-v1")
EVAL_SPLIT = os.environ.get("EVAL_SPLIT", "test")

# ── Layer configs ─────────────────────────────────────────────
# Mistral-7B-v0.1 has 32 layers (0-31).
# Placeholder CONFIGS below: will be updated from bisimulation results.
# After running bisimulation, set BISIM_JSON to the results file to auto-populate.
# For now, using similar adjacent-cluster selections as Llama-3.1-8B as placeholder.
# INTERCHANGE-guided: layers with lowest adjacent-pair interchange KL → most similar to neighbors
# REPLACEMENT-guided: layers with highest single-layer removal impact → most replaceable by neighbor

BISIM_JSON = os.environ.get("BISIM_JSON", "")

def load_configs_from_bisim(bisim_json_path):
    """Auto-populate CONFIGS from bisimulation results JSON."""
    with open(bisim_json_path) as f:
        d = json.load(f)
    results = d.get("results", [])
    # Sort adjacent pairs by KL (ascending = low interchange distance = safe layers)
    pairs = [(r["layer_a"], r["layer_b"], r["kl"]) for r in results
             if r["kl"] == r["kl"] and r["kl"] is not None]
    pairs_sorted = sorted(pairs, key=lambda x: x[2])

    # Interchange-guided: top n adjacent pairs with lowest KL
    def interchange_layers(n):
        # Pick n layers from the n lowest-KL pairs, prefer distributed
        selected = []
        for la, lb, kl in pairs_sorted:
            # Add the "interior" layer of the pair (the less extreme one)
            candidate = lb if la > 0 else la
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= n:
                break
        return sorted(selected)

    # Replacement-guided: top n pairs with highest KL → most distinct from neighbors
    pairs_desc = sorted(pairs, key=lambda x: x[2], reverse=True)

    def replacement_layers(n):
        selected = []
        for la, lb, kl in pairs_desc:
            candidate = lb if la > 0 else la
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= n:
                break
        return sorted(selected)

    configs = {
        "baseline": [],
        "interchange_n1": interchange_layers(1),
        "interchange_n3": interchange_layers(3),
        "interchange_n5": interchange_layers(5),
        "replacement_n1": replacement_layers(1),
        "replacement_n3": replacement_layers(3),
        "replacement_n5": replacement_layers(5),
    }
    log.info(f"Auto-populated CONFIGS from {bisim_json_path}:")
    for k, v in configs.items():
        log.info(f"  {k}: {v}")
    return configs

# Default CONFIGS (placeholders, replace after bisimulation)
CONFIGS_DEFAULT = {
    "baseline": [],
    # Interchange-guided: low KL pairs → safer to remove (placeholder, similar to Llama pattern)
    "interchange_n1": [23],
    "interchange_n3": [22, 23, 24],
    "interchange_n5": [20, 21, 22, 23, 24],
    # Replacement-guided: high KL pairs → least interchangeable (placeholder)
    "replacement_n1": [0],
    "replacement_n3": [0, 1, 30],
    "replacement_n5": [0, 1, 29, 30, 31],
    # BI-guided (ShortGPT): low block importance
    "bi_n1": [2],
    "bi_n3": [2, 3, 4],
    "bi_n5": [2, 3, 4, 5, 6],
    # Random controls (seed 42) - fixed for reproducibility
    "random_n1": [7],
    "random_n3": [4, 15, 23],
    "random_n5": [2, 9, 15, 22, 28],
}

# Load from bisimulation results if available
if BISIM_JSON and os.path.exists(BISIM_JSON):
    CONFIGS = load_configs_from_bisim(BISIM_JSON)
    # Add BI and random baselines
    CONFIGS.update({
        "bi_n1": CONFIGS_DEFAULT["bi_n1"],
        "bi_n3": CONFIGS_DEFAULT["bi_n3"],
        "bi_n5": CONFIGS_DEFAULT["bi_n5"],
        "random_n1": CONFIGS_DEFAULT["random_n1"],
        "random_n3": CONFIGS_DEFAULT["random_n3"],
        "random_n5": CONFIGS_DEFAULT["random_n5"],
    })
else:
    log.warning("BISIM_JSON not set; using placeholder CONFIGS. "
                "Run bisimulation first and set BISIM_JSON=/path/to/results.json")
    CONFIGS = CONFIGS_DEFAULT


# ── Weight loading (adapted from matched_eval_llama.py) ───────

def load_and_stack(model_name, dtype=jnp.bfloat16):
    """Load safetensors → stack for lax.scan. Memory-efficient."""
    hf_tok = (os.environ.get("HF_TOKEN") or
              (open("/tmp/hf_token").read().strip() if os.path.exists("/tmp/hf_token") else None))
    config = AutoConfig.from_pretrained(model_name, token=hf_tok)
    n_layers = config.num_hidden_layers
    hidden = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv = getattr(config, "num_key_value_heads", n_heads)
    head_dim = getattr(config, "head_dim", hidden // n_heads)
    inter = config.intermediate_size
    eps = getattr(config, "rms_norm_eps", 1e-5)
    rp = getattr(config, "rope_scaling", {}) or {}
    rope_theta = rp.get("rope_theta", getattr(config, "rope_theta", 10000.0))

    log.info(f"Model: {model_name}")
    log.info(f"  {n_layers}L, h={hidden}, heads={n_heads}/{n_kv}, inter={inter}")
    log.info(f"  rope_theta={rope_theta}, eps={eps}")

    repo = snapshot_download(model_name, allow_patterns=["*.safetensors", "*.json"],
                             token=hf_tok)
    log.info(f"Downloaded to: {repo}")

    arch = {
        "n_layers": n_layers, "hidden": hidden, "n_heads": n_heads,
        "n_kv": n_kv, "head_dim": head_dim, "inter": inter,
        "rope_theta": float(rope_theta), "eps": eps, "has_qk_norm": False,
    }

    # Find all safetensor shards
    shard_files = sorted(Path(repo).glob("*.safetensors"))
    log.info(f"Loading from {len(shard_files)} shard(s)")

    # Pre-allocate stacked weight arrays
    lw = {
        "q":   np.zeros((n_layers, n_heads * head_dim, hidden), dtype=np.float16),
        "k":   np.zeros((n_layers, n_kv * head_dim, hidden), dtype=np.float16),
        "v":   np.zeros((n_layers, n_kv * head_dim, hidden), dtype=np.float16),
        "o":   np.zeros((n_layers, hidden, n_heads * head_dim), dtype=np.float16),
        "g1":  np.zeros((n_layers, inter, hidden), dtype=np.float16),
        "g2":  np.zeros((n_layers, hidden, inter), dtype=np.float16),
        "g3":  np.zeros((n_layers, inter, hidden), dtype=np.float16),
        "ln1": np.zeros((n_layers, hidden), dtype=np.float32),
        "ln2": np.zeros((n_layers, hidden), dtype=np.float32),
    }
    embed_np = None
    final_norm_np = None
    lm_head_np = None

    for shard_file in shard_files:
        with safe_open(str(shard_file), framework="numpy") as f:
            keys = list(f.keys())
            for k in keys:
                # Embedding
                if k == "model.embed_tokens.weight":
                    embed_np = f.get_tensor(k).astype(np.float16)
                elif k == "model.norm.weight":
                    final_norm_np = f.get_tensor(k).astype(np.float32)
                elif k == "lm_head.weight":
                    lm_head_np = f.get_tensor(k).astype(np.float16)
                elif k.startswith("model.layers."):
                    parts = k.split(".")
                    li = int(parts[2])
                    rest = ".".join(parts[3:])
                    t = f.get_tensor(k).astype(np.float16 if "norm" not in rest else np.float32)
                    if rest == "self_attn.q_proj.weight": lw["q"][li] = t
                    elif rest == "self_attn.k_proj.weight": lw["k"][li] = t
                    elif rest == "self_attn.v_proj.weight": lw["v"][li] = t
                    elif rest == "self_attn.o_proj.weight": lw["o"][li] = t
                    elif rest == "mlp.gate_proj.weight": lw["g1"][li] = t
                    elif rest == "mlp.down_proj.weight": lw["g2"][li] = t
                    elif rest == "mlp.up_proj.weight": lw["g3"][li] = t
                    elif rest == "input_layernorm.weight": lw["ln1"][li] = t.astype(np.float32)
                    elif rest == "post_attention_layernorm.weight": lw["ln2"][li] = t.astype(np.float32)
        gc.collect()

    # Convert to JAX
    lw_jax = {k: jax.device_put(jnp.array(v)) for k, v in lw.items()}
    embed = jax.device_put(jnp.array(embed_np))
    final_norm = jax.device_put(jnp.array(final_norm_np))
    lm_head = jax.device_put(jnp.array(lm_head_np))

    log.info(f"Loaded {n_layers} layers")
    return arch, lw_jax, embed, final_norm, lm_head


# ── RoPE ─────────────────────────────────────────────────────

def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

def precompute_rope(seq_len, head_dim, theta, dtype):
    freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    angles = positions[:, None] * freqs[None, :]
    cos = jnp.cos(angles).astype(dtype)
    sin = jnp.sin(angles).astype(dtype)
    return cos, sin


# ── Transformer forward (with skip mask) ─────────────────────

def build_forward(arch):
    n_layers = arch["n_layers"]
    hidden = arch["hidden"]
    n_heads = arch["n_heads"]
    n_kv = arch["n_kv"]
    head_dim = arch["head_dim"]
    inter = arch["inter"]
    eps = arch["eps"]
    scale = head_dim ** -0.5

    def rms_norm(x, w):
        x32 = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(x32 ** 2, axis=-1, keepdims=True) + eps)
        return (x32 / rms * w.astype(jnp.float32)).astype(x.dtype)

    def silu(x):
        return x * jax.nn.sigmoid(x)

    def transformer_layer(h, lw_i, cos, sin, skip):
        """Single transformer layer; skip=True → identity."""
        # Attention
        h_norm = rms_norm(h, lw_i["ln1"])
        B, S, _ = h_norm.shape
        q = h_norm @ lw_i["q"].T
        k = h_norm @ lw_i["k"].T
        v = h_norm @ lw_i["v"].T
        q = q.reshape(B, S, n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, n_kv, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, n_kv, head_dim).transpose(0, 2, 1, 3)
        q = apply_rope(q, cos[None, None], sin[None, None])
        k = apply_rope(k, cos[None, None], sin[None, None])
        # GQA: expand k/v
        if n_kv < n_heads:
            rep = n_heads // n_kv
            k = jnp.repeat(k, rep, axis=1)
            v = jnp.repeat(v, rep, axis=1)
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale
        mask = jnp.tril(jnp.ones((S, S), dtype=attn.dtype))
        attn = jnp.where(mask[None, None] == 0, jnp.finfo(attn.dtype).min, attn)
        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(attn.dtype)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, S, n_heads * head_dim)
        h_attn = out @ lw_i["o"].T
        h = h + jnp.where(skip, 0.0, h_attn)
        # FFN
        h_norm2 = rms_norm(h, lw_i["ln2"])
        gate = silu(h_norm2 @ lw_i["g1"].T)
        up = h_norm2 @ lw_i["g3"].T
        h_ffn = (gate * up) @ lw_i["g2"].T
        h = h + jnp.where(skip, 0.0, h_ffn)
        return h

    def forward(input_ids, lw, embed, final_norm, lm_head, cos, sin, skip_mask):
        """Full forward pass; skip_mask[i]=True → skip layer i."""
        h = embed[input_ids]

        def scan_fn(carry, x):
            h = carry
            lw_i = {k: v[0] for k, v in x.items() if k != "skip"}
            skip = x["skip"]
            h = transformer_layer(h, lw_i, cos, sin, skip)
            return h, None

        # Build per-layer dict for scan
        lw_scan = {k: v for k, v in lw.items()}
        lw_scan["skip"] = skip_mask

        def body(h, lw_i):
            skip = lw_i.pop("skip")
            h = transformer_layer(h, lw_i, cos, sin, skip)
            return h, None

        # Manual loop (lax.scan requires uniform structure)
        for i in range(n_layers):
            lw_i = {k: lw[k][i] for k in lw}
            h = transformer_layer(h, lw_i, cos, sin, skip_mask[i])

        h32 = h.astype(jnp.float32)
        w32 = final_norm.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(h32 ** 2, axis=-1, keepdims=True) + eps)
        h_norm = h32 / rms * w32
        logits = h_norm @ lm_head.T.astype(jnp.float32)
        return logits

    return jax.jit(forward), transformer_layer

# ── Evaluation ────────────────────────────────────────────────

def eval_dataset_label():
    if EVAL_DATASET_CONFIG:
        return f"{EVAL_DATASET_NAME}/{EVAL_DATASET_CONFIG}:{EVAL_SPLIT}"
    return f"{EVAL_DATASET_NAME}:{EVAL_SPLIT}"


def get_text_column(ds):
    for column in ("text", "sentence", "content", "review"):
        if column in ds.column_names:
            return column
    raise KeyError(f"No supported text column found in {ds.column_names}")


def load_eval_tokens(tokenizer):
    from datasets import load_dataset
    dataset_args = [EVAL_DATASET_NAME]
    dataset_label = EVAL_DATASET_NAME
    if EVAL_DATASET_CONFIG:
        dataset_args.append(EVAL_DATASET_CONFIG)
        dataset_label = f"{EVAL_DATASET_NAME}/{EVAL_DATASET_CONFIG}"

    log.info(f"Loading {dataset_label} {EVAL_SPLIT}...")
    ds = load_dataset(*dataset_args, split=EVAL_SPLIT)
    text_column = get_text_column(ds)
    text = "\n\n".join([t for t in ds[text_column] if isinstance(t, str) and t.strip()])
    words = text.split()
    if len(words) > MAX_WORDS:
        text = " ".join(words[:MAX_WORDS])
        log.info(f"{dataset_label} {EVAL_SPLIT}: truncated to {MAX_WORDS} words")
    else:
        log.info(f"{dataset_label} {EVAL_SPLIT}: {len(words)} words (no truncation needed)")

    tokens = np.array(tokenizer.encode(text), dtype=np.int64)
    label = f"{eval_dataset_label()}, max_words={MAX_WORDS}, window={WINDOW}, stride={STRIDE}"
    log.info(f"Eval tokens: {len(tokens)}")
    return tokens, label


def evaluate_ppl(forward_fn, tokens, lw, embed, final_norm, lm_head,
                 arch, skip_mask, dtype=jnp.bfloat16):
    cos_w, sin_w = precompute_rope(WINDOW, arch["head_dim"], arch["rope_theta"], dtype)
    cos_w = jax.device_put(cos_w)
    sin_w = jax.device_put(sin_w)

    total_nll = 0.0
    n_tokens = 0

    for start in range(0, len(tokens) - WINDOW, STRIDE):
        chunk = tokens[start: start + WINDOW]
        ids = jnp.array(chunk[None], dtype=jnp.int32)
        logits = forward_fn(ids, lw, embed, final_norm, lm_head, cos_w, sin_w, skip_mask)
        logits = logits[0, :-1, :]
        targets = chunk[1:]
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        nlls = -log_probs[jnp.arange(len(targets)), targets]
        total_nll += float(jnp.sum(nlls))
        n_tokens += len(targets)

    ppl = math.exp(total_nll / n_tokens)
    return ppl, n_tokens


# ── Main ──────────────────────────────────────────────────────

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    log.info(f"=== Matched-Evaluator Experiment: {MODEL_NAME} ===")
    log.info(f"Devices: {jax.devices()}")
    log.info(f"Report dir: {REPORT_DIR}")

    t0 = time.time()

    arch, lw, embed, final_norm, lm_head = load_and_stack(MODEL_NAME, DTYPE)
    n_layers = arch["n_layers"]
    t_load = time.time() - t0
    log.info(f"Model loaded in {t_load:.1f}s")

    hf_tok = (os.environ.get("HF_TOKEN") or
              (open("/tmp/hf_token").read().strip() if os.path.exists("/tmp/hf_token") else None))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_tok)
    tokens, eval_label = load_eval_tokens(tokenizer)

    forward_fn, _ = build_forward(arch)

    log.info("JIT compile (warmup)...")
    dummy_mask = jnp.zeros(n_layers, dtype=jnp.bool_)
    dummy_ids = jnp.zeros((1, WINDOW), dtype=jnp.int32)
    cos_w, sin_w = precompute_rope(WINDOW, arch["head_dim"], arch["rope_theta"], DTYPE)
    cos_w, sin_w = jax.device_put(cos_w), jax.device_put(sin_w)
    _ = forward_fn(dummy_ids, lw, embed, final_norm, lm_head, cos_w, sin_w, dummy_mask)
    jax.block_until_ready(_)
    log.info("JIT compiled.")

    results = {}
    baseline_ppl = None

    unique_configs = {}
    for name, layers in CONFIGS.items():
        key = tuple(sorted(layers))
        if key not in unique_configs:
            unique_configs[key] = []
        unique_configs[key].append(name)

    log.info(f"\n{'='*70}")
    log.info(f"Evaluating {len(unique_configs)} unique configurations ({len(CONFIGS)} named)")
    log.info(f"{'='*70}")

    for i, (layer_key, names) in enumerate(unique_configs.items()):
        skip_layers = list(layer_key)
        primary_name = names[0]

        skip_mask = jnp.zeros(n_layers, dtype=jnp.bool_)
        if skip_layers:
            skip_mask = skip_mask.at[jnp.array(skip_layers)].set(True)

        log.info(f"\n[{i+1}/{len(unique_configs)}] {primary_name} (skip={skip_layers})")
        t_start = time.time()

        ppl, n_tokens = evaluate_ppl(
            forward_fn, tokens, lw, embed, final_norm, lm_head,
            arch, skip_mask, dtype=DTYPE
        )

        elapsed = time.time() - t_start

        if not skip_layers:
            baseline_ppl = ppl

        delta = ((ppl / baseline_ppl) - 1) * 100 if baseline_ppl else 0.0

        for name in names:
            results[name] = {
                "layers_removed": skip_layers,
                "n_removed": len(skip_layers),
                "ppl": round(ppl, 4),
                "delta_ppl_pct": round(delta, 2),
                "n_tokens": n_tokens,
                "elapsed_s": round(elapsed, 1),
            }

        log.info(f"  PPL={ppl:.4f}  Δ={delta:+.2f}%  ({elapsed:.1f}s)")
        for name in names[1:]:
            log.info(f"  (also: {name})")

    results["_meta"] = {
        "model": MODEL_NAME,
        "evaluator": f"{eval_label}, max_words={MAX_WORDS}, window={WINDOW}, stride={STRIDE}",
        "dataset_name": EVAL_DATASET_NAME,
        "dataset_config": EVAL_DATASET_CONFIG,
        "dataset_split": EVAL_SPLIT,
        "baseline_ppl": baseline_ppl,
        "dtype": "bfloat16",
        "n_layers": n_layers,
        "timestamp": time.strftime("%Y-%m-%dT%H-%M-%S"),
        "total_time_s": round(time.time() - t0, 1),
        "device": str(jax.devices()),
    }

    out_path = os.path.join(REPORT_DIR, "matched_eval_mistral_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved to {out_path}")

    log.info(f"\n{'='*70}")
    log.info("SUMMARY TABLE")
    log.info(f"{'='*70}")
    log.info(f"Baseline PPL: {baseline_ppl:.4f}")
    log.info(f"{'Method':<30} {'Layers':<30} {'PPL':>8} {'Δ%':>8}")
    log.info(f"{'-'*80}")
    for name in CONFIGS:
        if name == "baseline":
            continue
        r = results.get(name)
        if r:
            log.info(f"{name:<30} {str(r['layers_removed']):<30} {r['ppl']:>8.4f} {r['delta_ppl_pct']:>+8.2f}%")

    log.info(f"\nTotal time: {time.time() - t0:.1f}s")
    log.info("DONE.")


if __name__ == "__main__":
    main()
