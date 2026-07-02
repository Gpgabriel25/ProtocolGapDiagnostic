#!/usr/bin/env python3
"""
Matched-evaluator experiment: ALL Pythia-1.4B layer removal methods evaluated
under a single unified held-out-corpus evaluator (5K words, window=512, stride=256).

EXACTLY the same evaluator protocol as matched_eval_qwen3.py and matched_eval_llama.py
for strict cross-model comparability.

Target: TPU v6e-8 (bq-v6e-8, europe-west4-a), JAX + fp32
Expected runtime: ~20-30 minutes total

Produces:
  /tmp/pythia_matched_eval/results.json
"""

import os, sys, json, time, logging, gc, math
import numpy as np
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import jax
import jax.numpy as jnp
from jax import lax

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = "EleutherAI/pythia-1.4b"
# Use fp32 for this matched-eval pass to avoid NaNs under aggressive skip masks.
DTYPE = jnp.float32
REPORT_DIR = os.environ.get("REPORT_DIR", "/tmp/pythia_matched_eval")
Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)

# ── All layer removal configurations ──────────────────────────
# Pythia-1.4B has 24 layers (0-23)
# Layer selections from reports/2026-04-02T16-23-40/pythia-1.4b_interchange.json
# and reports/2026-04-06T17-39-31/pythia_full_baselines.json

CONFIGS = {
    # Baseline
    "baseline": [],

    # === Interchange-guided (lowest min-neighbor interchange distance) ===
    # Per-layer min interchange: 20=0.0225, 16=0.0266, 17=0.0266, 15=0.0280, 19=0.0225
    "interchange_n1": [20],          # Best: layer 20 (pair 19-20, d=0.0225)
    "interchange_n3": [16, 17, 20],  # Top-3 layers by min interchange
    "interchange_n5": [15, 16, 17, 19, 20],  # Top-5

    # === Replacement-guided (lowest min-neighbor replacement distance) ===
    # Per-layer min replacement: 10=0.0823, 7=0.1081, 11=0.0972, 13=0.1031
    "replacement_n1": [10],          # Best: layer 10 (pair 9-10, d=0.0823)
    "replacement_n3": [10, 11, 7],   # Top-3 layers by min replacement
    "replacement_n5": [7, 9, 10, 11, 13],  # Top-5

    # === BI-guided (ShortGPT — lowest BI score = most removable) ===
    # BI scores from reports/2026-04-06T17-39-31/pythia_full_baselines.json
    # Lowest BI: 20 (0.0105), 21 (0.0107), 19 (0.0141), 22 (0.0145), 17 (0.0168)
    "bi_n1": [20],
    "bi_n3": [19, 20, 21],
    "bi_n5": [17, 18, 19, 20, 21],

    # === CKA-guided (highest CKA between adjacent = most redundant) ===
    # Highest CKA: 7 (0.9998), 9 (0.9998), 8 (0.9998), 5 (0.9998), 12 (0.9997)
    "cka_n1": [7],
    "cka_n3": [7, 8, 9],
    "cka_n5": [5, 7, 8, 9, 12],

    # === SLEB-greedy (one-shot from pythia_full_baselines.json) ===
    # Lowest SLEB: 17 (0.1824), 9 (0.1868), 13 (0.2019), 10 (0.2029), 7 (0.2104)
    "sleb_greedy_n1": [17],
    "sleb_greedy_n3": [9, 13, 17],
    "sleb_greedy_n5": [7, 9, 10, 13, 17],

    # === SLEB-iterative (from pythia_full_baselines.json) ===
    "sleb_iterative_n1": [17],
    "sleb_iterative_n3": [9, 10, 17],
    "sleb_iterative_n5": [7, 9, 10, 13, 17],

    # === Random controls (seed 42 — deterministic for reproducibility) ===
    "random_n1": [3],
    "random_n3": [3, 12, 18],
    "random_n5": [1, 3, 12, 15, 18],
}


# ── Weight loading ────────────────────────────────────────────

def load_pythia_weights():
    """Load Pythia-1.4B weights from HuggingFace → JAX arrays on TPU."""
    from transformers import AutoModelForCausalLM, AutoConfig
    import torch

    hf_tok = os.environ.get("HF_TOKEN") or (
        open("/tmp/hf_token").read().strip()
        if os.path.exists("/tmp/hf_token") else None
    )

    log.info(f"Loading {MODEL_NAME}...")
    config = AutoConfig.from_pretrained(MODEL_NAME, token=hf_tok)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, token=hf_tok, torch_dtype=torch.bfloat16
    )

    sd = model.state_dict()
    n_layers = config.num_hidden_layers
    d_model = config.hidden_size
    n_heads = config.num_attention_heads
    d_head = d_model // n_heads
    inter = config.intermediate_size

    log.info(f"  {n_layers}L, d={d_model}, heads={n_heads}, d_head={d_head}, inter={inter}")

    def to_jax(t):
        return jax.device_put(jnp.array(t.detach().float().numpy(), dtype=DTYPE))

    # Embeddings
    wte = to_jax(sd["gpt_neox.embed_in.weight"])
    ln_f_w = to_jax(sd["gpt_neox.final_layer_norm.weight"])
    ln_f_b = to_jax(sd["gpt_neox.final_layer_norm.bias"])
    lm_head = to_jax(sd["embed_out.weight"])

    # Stack per-layer weights for scan
    layer_keys = ["qkv_w", "qkv_b", "o_w", "o_b", "ln1_w", "ln1_b",
                  "ff1_w", "ff1_b", "ff2_w", "ff2_b", "ln2_w", "ln2_b"]

    all_layers = []
    for i in range(n_layers):
        p = f"gpt_neox.layers.{i}"
        lw = {
            "qkv_w": to_jax(sd[f"{p}.attention.query_key_value.weight"]),
            "qkv_b": to_jax(sd[f"{p}.attention.query_key_value.bias"]),
            "o_w": to_jax(sd[f"{p}.attention.dense.weight"]),
            "o_b": to_jax(sd[f"{p}.attention.dense.bias"]),
            "ln1_w": to_jax(sd[f"{p}.input_layernorm.weight"]),
            "ln1_b": to_jax(sd[f"{p}.input_layernorm.bias"]),
            "ff1_w": to_jax(sd[f"{p}.mlp.dense_h_to_4h.weight"]),
            "ff1_b": to_jax(sd[f"{p}.mlp.dense_h_to_4h.bias"]),
            "ff2_w": to_jax(sd[f"{p}.mlp.dense_4h_to_h.weight"]),
            "ff2_b": to_jax(sd[f"{p}.mlp.dense_4h_to_h.bias"]),
            "ln2_w": to_jax(sd[f"{p}.post_attention_layernorm.weight"]),
            "ln2_b": to_jax(sd[f"{p}.post_attention_layernorm.bias"]),
        }
        all_layers.append(lw)

    # Stack into pytrees for scan
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *all_layers)

    del model, sd, all_layers
    gc.collect()

    rotary_pct = getattr(config, "rotary_pct", 0.25)
    rotary_ndims = int(d_head * rotary_pct)
    log.info(f"  rotary_pct={rotary_pct}, rotary_ndims={rotary_ndims}")

    arch = {
        "n_layers": n_layers, "d_model": d_model, "n_heads": n_heads,
        "d_head": d_head, "inter": inter, "rotary_ndims": rotary_ndims,
    }
    log.info("Weights loaded and stacked on TPU.")
    return arch, stacked, wte, ln_f_w, ln_f_b, lm_head


# ── Model components ──────────────────────────────────────────

def layer_norm(x, w, b):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return w * (x - mean) / jnp.sqrt(var + 1e-5) + b


def precompute_rope(seq_len, rotary_ndims, dtype):
    half_rot = rotary_ndims // 2
    freqs = 1.0 / (10000.0 ** (jnp.arange(0, half_rot, dtype=jnp.float32) / half_rot))
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    angles = positions[:, None] * freqs[None, :]
    cos = jnp.cos(angles).astype(dtype)
    sin = jnp.sin(angles).astype(dtype)
    return cos, sin


# ── Forward pass with skip mask ───────────────────────────────

def build_forward(arch):
    n_heads = arch["n_heads"]
    d_head = arch["d_head"]
    d_model = arch["d_model"]
    rotary_ndims = arch["rotary_ndims"]
    half_rot = rotary_ndims // 2

    def one_layer(hidden, lw_slice, cos, sin):
        B, S, D = hidden.shape

        # Layer norms (parallel GPT-NeoX: both on the SAME input)
        h_attn = layer_norm(hidden, lw_slice["ln1_w"], lw_slice["ln1_b"])
        h_ff = layer_norm(hidden, lw_slice["ln2_w"], lw_slice["ln2_b"])

        # --- Attention ---
        qkv = h_attn @ lw_slice["qkv_w"].T + lw_slice["qkv_b"]
        qkv = qkv.reshape(B, S, n_heads, 3, d_head)
        q = qkv[:, :, :, 0, :]
        k = qkv[:, :, :, 1, :]
        v = qkv[:, :, :, 2, :]

        # RoPE — Pythia applies rotary only to the first rotary_ndims dims
        cos_b = cos[None, :S, None, :]  # (1, S, 1, half_rot)
        sin_b = sin[None, :S, None, :]

        def apply_rotary(t):
            t_rot = t[..., :rotary_ndims]
            t_pass = t[..., rotary_ndims:]
            t1, t2 = t_rot[..., :half_rot], t_rot[..., half_rot:]
            t_rot_out = jnp.concatenate([t1 * cos_b - t2 * sin_b,
                                         t2 * cos_b + t1 * sin_b], axis=-1)
            return jnp.concatenate([t_rot_out, t_pass], axis=-1)

        q = apply_rotary(q)
        k = apply_rotary(k)

        # Attention scores
        q = q.transpose(0, 2, 1, 3)  # (B, H, S, D)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scale = d_head ** -0.5
        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale

        # Causal mask
        mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        neg_inf = jnp.array(-1e30, dtype=attn.dtype)
        attn = jnp.where(mask[None, None], attn, neg_inf)
        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(hidden.dtype)

        out = jnp.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
        out = out @ lw_slice["o_w"].T + lw_slice["o_b"]

        # --- FFN (parallel: operates on same input, not attention output) ---
        ff = h_ff @ lw_slice["ff1_w"].T + lw_slice["ff1_b"]
        ff = jax.nn.gelu(ff, approximate=False)
        ff = ff @ lw_slice["ff2_w"].T + lw_slice["ff2_b"]

        return hidden + out + ff

    @jax.jit
    def forward(input_ids, layer_weights, wte, ln_f_w, ln_f_b, lm_head,
                cos, sin, skip_mask):
        """Forward with skip mask. skip_mask: (n_layers,) bool — True = skip."""
        hidden = wte[input_ids]

        def scan_body(hidden, scan_input):
            idx, should_skip = scan_input
            lw_slice = jax.tree.map(lambda w: w[idx], layer_weights)
            new_hidden = one_layer(hidden, lw_slice, cos, sin)
            return jnp.where(should_skip, hidden, new_hidden), None

        indices = jnp.arange(arch["n_layers"])
        hidden, _ = lax.scan(scan_body, hidden, (indices, skip_mask))

        hidden = layer_norm(hidden, ln_f_w, ln_f_b)
        logits = hidden @ lm_head.T
        return logits

    return forward


# ── Perplexity evaluation (EXACTLY matching Qwen3-8B / Llama evaluator) ──

MAX_WORDS = int(os.environ.get("EVAL_MAX_WORDS", "5000"))
WINDOW = 512       # Same
STRIDE = 256       # Same
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
    """Load a held-out text corpus and truncate to MAX_WORDS. Same surface as Qwen3/Llama."""
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


def evaluate_ppl(forward_fn, tokens, layer_weights, wte, ln_f_w, ln_f_b, lm_head,
                 arch, skip_mask):
    """Sliding-window PPL. EXACTLY the same protocol as Qwen3-8B/Llama matched eval.
    Protocol: window=512, stride=256, held-out corpus subset.
    """
    seq_len = len(tokens)
    total_nll = 0.0
    total_tokens = 0
    n_windows = 0
    prev_end = 0

    cos, sin = precompute_rope(WINDOW, arch["rotary_ndims"], DTYPE)
    cos, sin = jax.device_put(cos), jax.device_put(sin)

    for begin in range(0, seq_len, STRIDE):
        end = min(begin + WINDOW, seq_len)
        target_len = end - prev_end

        chunk = tokens[begin:end]
        input_ids = jax.device_put(jnp.array([chunk], dtype=jnp.int32))

        logits = forward_fn(input_ids, layer_weights, wte, ln_f_w, ln_f_b, lm_head,
                            cos, sin, skip_mask)

        if not bool(jnp.all(jnp.isfinite(logits))):
            return float("nan"), total_tokens, n_windows

        # Score only the non-overlapping suffix
        shift_logits = logits[0, -(target_len):-1, :]
        shift_labels = jnp.array(chunk[-(target_len)+1:], dtype=jnp.int32) if target_len > 1 else jnp.array([], dtype=jnp.int32)

        if shift_labels.size == 0:
            prev_end = end
            continue

        log_probs = jax.nn.log_softmax(shift_logits.astype(jnp.float32), axis=-1)
        nll = -log_probs[jnp.arange(shift_labels.shape[0]), shift_labels]
        if not bool(jnp.all(jnp.isfinite(nll))):
            return float("nan"), total_tokens, n_windows
        total_nll += float(jnp.sum(nll))
        total_tokens += int(shift_labels.shape[0])
        n_windows += 1
        prev_end = end

        if end >= seq_len:
            break

    ppl = float(np.exp(total_nll / total_tokens)) if total_tokens > 0 else float('inf')
    return ppl, total_tokens, n_windows


# ── Bootstrap CI ──────────────────────────────────────────────

def bootstrap_ppl(forward_fn, tokens, layer_weights, wte, ln_f_w, ln_f_b, lm_head,
                  arch, skip_mask, n_bootstrap=200, seed=42):
    """Bootstrap CI over sliding windows. Resample windows and recompute PPL."""
    cos, sin = precompute_rope(WINDOW, arch["rotary_ndims"], DTYPE)
    cos, sin = jax.device_put(cos), jax.device_put(sin)

    # Compute per-window NLLs
    seq_len = len(tokens)
    window_nlls = []
    window_counts = []
    prev_end = 0

    for begin in range(0, seq_len, STRIDE):
        end = min(begin + WINDOW, seq_len)
        target_len = end - prev_end

        chunk = tokens[begin:end]
        input_ids = jax.device_put(jnp.array([chunk], dtype=jnp.int32))

        logits = forward_fn(input_ids, layer_weights, wte, ln_f_w, ln_f_b, lm_head,
                            cos, sin, skip_mask)

        if not bool(jnp.all(jnp.isfinite(logits))):
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "median": float("nan"),
            }

        shift_logits = logits[0, -(target_len):-1, :]
        shift_labels = jnp.array(chunk[-(target_len)+1:], dtype=jnp.int32) if target_len > 1 else jnp.array([], dtype=jnp.int32)

        if shift_labels.size == 0:
            prev_end = end
            continue

        log_probs = jax.nn.log_softmax(shift_logits.astype(jnp.float32), axis=-1)
        nll = -log_probs[jnp.arange(shift_labels.shape[0]), shift_labels]
        if not bool(jnp.all(jnp.isfinite(nll))):
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "median": float("nan"),
            }
        window_nlls.append(float(jnp.sum(nll)))
        window_counts.append(int(shift_labels.shape[0]))
        prev_end = end
        if end >= seq_len:
            break

    # Bootstrap: resample windows with replacement
    rng = np.random.RandomState(seed)
    n_windows = len(window_nlls)
    nlls = np.array(window_nlls)
    counts = np.array(window_counts)

    ppls = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_windows, size=n_windows, replace=True)
        total_nll = nlls[idx].sum()
        total_count = counts[idx].sum()
        ppls.append(np.exp(total_nll / total_count))

    return {
        "mean": float(np.mean(ppls)),
        "std": float(np.std(ppls)),
        "ci_lower": float(np.percentile(ppls, 2.5)),
        "ci_upper": float(np.percentile(ppls, 97.5)),
        "median": float(np.median(ppls)),
    }


# ── Main ──────────────────────────────────────────────────────

def main():
    log.info(f"Devices: {jax.device_count()} × {jax.devices()[0].device_kind}")
    log.info(f"JAX version: {jax.__version__}")

    # Load weights
    t0 = time.time()
    arch, layer_weights, wte, ln_f_w, ln_f_b, lm_head = load_pythia_weights()
    log.info(f"Model loaded in {time.time()-t0:.1f}s")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load eval data
    tokens, eval_label = load_eval_tokens(tokenizer)

    # Build forward function
    forward_fn = build_forward(arch)

    n_layers = arch["n_layers"]
    results = {}

    for config_name, skip_layers in CONFIGS.items():
        log.info(f"\n{'='*60}")
        log.info(f"Config: {config_name}, skip={skip_layers}")

        skip_mask = jnp.zeros(n_layers, dtype=jnp.bool_)
        if skip_layers:
            skip_mask = skip_mask.at[jnp.array(skip_layers)].set(True)

        t0 = time.time()
        ppl, n_tok, n_win = evaluate_ppl(
            forward_fn, tokens, layer_weights, wte, ln_f_w, ln_f_b, lm_head,
            arch, skip_mask
        )
        elapsed = time.time() - t0

        # Bootstrap CI
        ci = bootstrap_ppl(
            forward_fn, tokens, layer_weights, wte, ln_f_w, ln_f_b, lm_head,
            arch, skip_mask, n_bootstrap=200
        )

        baseline_ppl = results.get("baseline", {}).get("ppl", ppl)
        delta_pct = ((ppl - baseline_ppl) / baseline_ppl * 100) if baseline_ppl > 0 else 0.0

        result = {
            "layers_removed": skip_layers,
            "n_removed": len(skip_layers),
            "ppl": round(ppl, 4),
            "delta_ppl_pct": round(delta_pct, 2),
            "ci_95_lower": round(ci["ci_lower"], 4),
            "ci_95_upper": round(ci["ci_upper"], 4),
            "ci_std": round(ci["std"], 4),
            "n_tokens": n_tok,
            "n_windows": n_win,
            "elapsed_s": round(elapsed, 1),
        }
        results[config_name] = result

        log.info(f"  PPL={ppl:.4f} (delta={delta_pct:+.2f}%)")
        log.info(f"  95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
        log.info(f"  {n_tok} tokens, {n_win} windows, {elapsed:.1f}s")

    # Add metadata
    results["_meta"] = {
        "model": MODEL_NAME,
        "evaluator": f"{eval_label}, max_words={MAX_WORDS}, window={WINDOW}, stride={STRIDE}",
        "dataset_name": EVAL_DATASET_NAME,
        "dataset_config": EVAL_DATASET_CONFIG,
        "dataset_split": EVAL_SPLIT,
        "dtype": "bfloat16",
        "device": str(jax.devices()[0]),
        "n_devices": jax.device_count(),
        "jax_version": jax.__version__,
        "n_bootstrap": 200,
        "max_words": MAX_WORDS,
        "window": WINDOW,
        "stride": STRIDE,
    }

    # Save results
    out_path = os.path.join(REPORT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved to {out_path}")

    # Summary table
    log.info("\n" + "="*80)
    log.info(f"SUMMARY TABLE: Pythia-1.4B Matched Evaluator ({eval_label}, {MAX_WORDS} words, w={WINDOW}, s={STRIDE})")
    log.info("="*80)
    log.info(f"{'Config':<30s} {'PPL':>8s} {'Δ%':>8s} {'CI-lo':>8s} {'CI-hi':>8s} {'Layers'}")
    for name, r in results.items():
        if name.startswith("_"):
            continue
        layers_str = str(r["layers_removed"]) if r["layers_removed"] else "[]"
        log.info(f"{name:<30s} {r['ppl']:8.2f} {r['delta_ppl_pct']:+8.2f} "
                 f"{r.get('ci_95_lower', 0):8.2f} {r.get('ci_95_upper', 0):8.2f} {layers_str}")


if __name__ == "__main__":
    main()
