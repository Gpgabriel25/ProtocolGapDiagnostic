#!/usr/bin/env python3
"""
Matched-evaluator experiment: ALL Llama-3.1-8B layer removal methods evaluated
under a single unified held-out-corpus evaluator (test split, 5K words, 512/256 window).

Same evaluator protocol as matched_eval_qwen3.py for cross-model consistency.

Target: TPU v6e-8 (bq-v6e-8, europe-west4-a), JAX + bf16
Expected runtime: ~60-90 minutes total
"""

import os, sys, json, time, logging, gc
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

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DTYPE = jnp.bfloat16
REPORT_DIR = os.environ.get("REPORT_DIR", "/tmp/matched_eval_llama")


# ── All layer removal configurations ──────────────────────────
# Layer indices 0-based (Llama-3.1-8B has 32 layers: 0-31)
# Selections from logs/2026-04-08T08-58-16/llama_bisimulation_results.json
# and reports/llama_sleb_results.json
CONFIGS = {
    # Baseline
    "baseline": [],

    # === Interchange-guided ===
    "interchange_n1": [25],
    "interchange_n3": [24, 25, 26],
    "interchange_n5": [22, 23, 24, 25, 26],

    # === Replacement-guided ===
    "replacement_n1": [11],
    "replacement_n3": [9, 10, 11],
    "replacement_n5": [7, 8, 9, 10, 11],

    # === BI-guided (ShortGPT) ===
    "bi_n1": [2],
    "bi_n3": [2, 3, 5],
    "bi_n5": [2, 3, 4, 5, 6],

    # === SLEB-greedy ===
    "sleb_greedy_n1": [11],
    "sleb_greedy_n3": [10, 11, 25],
    "sleb_greedy_n5": [8, 9, 10, 11, 25],

    # === SLEB-iterative ===
    "sleb_iterative_n1": [11],
    "sleb_iterative_n3": [10, 11, 25],
    "sleb_iterative_n5": [10, 11, 12, 25, 28],

    # === Random controls (seed 42) ===
    "random_n1": [3],
    "random_n3": [14, 19, 22],
    "random_n5": [3, 6, 16, 19, 29],
}


# ── Weight loading ────────────────────────────────────────────

def load_and_stack(model_name, dtype=jnp.bfloat16):
    """Load safetensors → stack for lax.scan. Memory-efficient."""
    hf_tok = os.environ.get("HF_TOKEN") or (open("/tmp/hf_token").read().strip() if os.path.exists("/tmp/hf_token") else None)
    config = AutoConfig.from_pretrained(model_name, token=hf_tok)
    n_layers = config.num_hidden_layers
    hidden = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv = getattr(config, "num_key_value_heads", n_heads)
    head_dim = getattr(config, "head_dim", hidden // n_heads)
    inter = config.intermediate_size

    rp = getattr(config, "rope_parameters", None) or {}
    rope_theta = rp.get("rope_theta", getattr(config, "rope_theta", 10000.0))
    eps = getattr(config, "rms_norm_eps", 1e-6)

    log.info(f"Model: {model_name}")
    log.info(f"  {n_layers}L, h={hidden}, heads={n_heads}/{n_kv}, inter={inter}")
    log.info(f"  rope_theta={rope_theta}, eps={eps}")

    repo = snapshot_download(model_name, allow_patterns=["*.safetensors", "*.json"], token=hf_tok)

    import glob
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
        "n_layers": n_layers, "hidden": hidden, "n_heads": n_heads,
        "n_kv": n_kv, "head_dim": head_dim, "inter": inter,
        "rope_theta": float(rope_theta), "eps": eps, "has_qk_norm": has_qk_norm,
    }
    return arch, lw, embed, final_norm, lm_head


# ── Model components ──────────────────────────────────────────

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


# ── Forward pass with skip mask ───────────────────────────────

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
        """Forward with skip mask. skip_mask: (n_layers,) bool — True = skip."""
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

    return forward, n_layers


# ── Perplexity evaluation ─────────────────────────────────────

MAX_WORDS = int(os.environ.get("EVAL_MAX_WORDS", "5000"))
WINDOW = 512       # Context window
STRIDE = 256       # Sliding-window stride
EVAL_DATASET_NAME = os.environ.get("EVAL_DATASET_NAME", "wikitext")
EVAL_DATASET_CONFIG = os.environ.get("EVAL_DATASET_CONFIG", "wikitext-2-raw-v1")
EVAL_SPLIT = os.environ.get("EVAL_SPLIT", "test")


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
    """Load a held-out text corpus and truncate to MAX_WORDS words."""
    from datasets import load_dataset
    dataset_args = [EVAL_DATASET_NAME]
    dataset_label = EVAL_DATASET_NAME
    if EVAL_DATASET_CONFIG:
        dataset_args.append(EVAL_DATASET_CONFIG)
        dataset_label = f"{EVAL_DATASET_NAME}/{EVAL_DATASET_CONFIG}"

    ds = load_dataset(*dataset_args, split=EVAL_SPLIT)
    text_column = get_text_column(ds)
    text = "\n\n".join([t for t in ds[text_column] if isinstance(t, str) and t.strip()])
    words = text.split()
    if len(words) > MAX_WORDS:
        text = " ".join(words[:MAX_WORDS])
        log.info(f"{dataset_label} {EVAL_SPLIT}: truncated to {MAX_WORDS} words")
    else:
        log.info(f"{dataset_label} {EVAL_SPLIT}: {len(words)} words (no truncation needed)")

    tokens = tokenizer.encode(text)
    log.info(f"{dataset_label} {EVAL_SPLIT} tokenized: {len(tokens)} tokens")
    return tokens, eval_dataset_label()


def evaluate_ppl(forward_fn, tokens, layer_weights, embed, final_norm, lm_head,
                 arch, skip_mask, dtype=jnp.bfloat16):
    """Sliding-window perplexity. Protocol: window=512, stride=256, held-out corpus subset."""
    seq_len = len(tokens)
    total_nll = 0.0
    total_tokens = 0
    n_windows = 0
    prev_end = 0

    cos, sin = precompute_rope(WINDOW, arch["head_dim"], arch["rope_theta"], dtype)
    cos, sin = jax.device_put(cos), jax.device_put(sin)

    for begin in range(0, seq_len, STRIDE):
        end = min(begin + WINDOW, seq_len)
        target_len = end - prev_end

        chunk = tokens[begin:end]
        actual_len = len(chunk)

        # Pad to WINDOW if shorter (avoid JIT recompilation)
        if actual_len < WINDOW:
            chunk = chunk + [0] * (WINDOW - actual_len)

        input_ids = jnp.array([chunk], dtype=jnp.int32)

        logits = forward_fn(input_ids, layer_weights, embed, final_norm, lm_head,
                            cos, sin, skip_mask)

        shift_logits = logits[0, :actual_len - 1, :]
        shift_targets = jnp.array(tokens[begin + 1:begin + actual_len], dtype=jnp.int32)

        log_probs = jax.nn.log_softmax(shift_logits.astype(jnp.float32), axis=-1)
        ce = -log_probs[jnp.arange(len(shift_targets)), shift_targets]

        if target_len < actual_len:
            score_start = actual_len - target_len
        else:
            score_start = 0

        nll = jnp.sum(ce[score_start:])
        n_scored = len(ce) - score_start

        total_nll += float(nll)
        total_tokens += n_scored
        n_windows += 1
        prev_end = end

        if end == seq_len:
            break

    ppl = np.exp(total_nll / total_tokens)
    log.info(f"  PPL={ppl:.4f} ({n_windows} windows, {total_tokens} tokens scored)")
    return ppl, total_tokens


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

    hf_token = os.environ.get("HF_TOKEN") or open("/tmp/hf_token").read().strip() if os.path.exists("/tmp/hf_token") else None
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
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

    # Deduplicate configs
    unique_configs = {}
    for name, layers in CONFIGS.items():
        key = tuple(sorted(layers))
        if key not in unique_configs:
            unique_configs[key] = []
        unique_configs[key].append(name)

    log.info(f"\n{'='*70}")
    log.info(f"Evaluating {len(unique_configs)} unique configurations "
             f"({len(CONFIGS)} named)")
    log.info(f"{'='*70}")

    for i, (layer_key, names) in enumerate(unique_configs.items()):
        skip_layers = list(layer_key)
        primary_name = names[0]

        skip_mask = jnp.zeros(n_layers, dtype=jnp.bool_)
        if skip_layers:
            skip_mask = skip_mask.at[jnp.array(skip_layers)].set(True)

        log.info(f"\n[{i+1}/{len(unique_configs)}] {primary_name} "
                 f"(skip={skip_layers})")
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

    out_path = os.path.join(REPORT_DIR, "matched_eval_llama_results.json")
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
        r = results[name]
        log.info(f"{name:<30} {str(r['layers_removed']):<30} {r['ppl']:>8.4f} {r['delta_ppl_pct']:>+8.2f}%")

    log.info(f"\nTotal time: {time.time() - t0:.1f}s")
    log.info("DONE.")


if __name__ == "__main__":
    main()
