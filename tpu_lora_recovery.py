"""
tpu_lora_recovery.py — LoRA fine-tuning recovery experiment on TPU.

Shows that skip-layer models can recover most PPL loss via cheap LoRA fine-tuning.
Runs on TPU using JAX/optax. Addresses reviewer concern about missing recovery results.

Protocol:
1. Load Qwen3-8B, evaluate WikiText-2 PPL (full baseline)
2. Remove layers (interchange-guided), evaluate PPL (pruned)
3. Add LoRA (rank 16) to Q/K/V/O projections, fine-tune 200 steps
4. Evaluate PPL (recovered)
"""

import os, sys, json, time, logging, math, gc
from pathlib import Path
from functools import partial

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-8B"
N_LAYERS_FULL = 36

# Skip configs from the paper (interchange-guided)
SKIP_CONFIGS = {
    "skip_n1": [17],
    "skip_n3": [15, 17, 20],
}

# LoRA config
LORA_RANK = 16
LORA_ALPHA = 32  # scaling = alpha / rank = 2.0
LORA_TARGETS = ["q_w", "k_w", "v_w", "o_w"]  # attention projections

# Training config
LR = 2e-4
N_STEPS = 200
BATCH_SIZE = 1  # keep small for memory (8B model + backprop)
SEQ_LEN = 128   # shorter for memory
EVAL_SEQS = 50  # number of sequences for PPL evaluation

HF_TOKEN_PATH = Path("/tmp/hf_token")
OUTDIR = Path("/tmp/lora_recovery")

# ── Setup ───────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp
import numpy as np
import optax

log.info(f"JAX version: {jax.__version__}")
log.info(f"Devices: {jax.device_count()} × {jax.devices()[0].device_kind}")

OUTDIR.mkdir(parents=True, exist_ok=True)

hf_token = None
if HF_TOKEN_PATH.exists():
    hf_token = HF_TOKEN_PATH.read_text().strip()
    os.environ["HF_TOKEN"] = hf_token


# ── Weight loading ──────────────────────────────────────────────────
def load_qwen_weights(skip_layers=None):
    """Load Qwen3-8B weights, optionally skipping some layers."""
    from transformers import AutoModelForCausalLM, AutoConfig
    import torch

    skip_layers = skip_layers or []
    kept_layers = [i for i in range(N_LAYERS_FULL) if i not in skip_layers]
    n_kept = len(kept_layers)

    log.info(f"Loading {MODEL_NAME}, keeping {n_kept}/{N_LAYERS_FULL} layers...")
    config = AutoConfig.from_pretrained(MODEL_NAME, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, token=hf_token, torch_dtype=torch.bfloat16
    )
    sd = model.state_dict()

    d_model = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv = config.num_key_value_heads
    d_head = d_model // n_heads
    inter = config.intermediate_size
    has_qk_norm = getattr(config, "qk_norm", False) or any("q_norm" in k for k in sd)

    def to_jax(t):
        return jnp.array(t.detach().float().numpy(), dtype=jnp.bfloat16)

    wte = to_jax(sd["model.embed_tokens.weight"])
    ln_f_w = to_jax(sd["model.norm.weight"])
    lm_head_w = to_jax(sd["lm_head.weight"])

    layers = []
    for orig_idx in kept_layers:
        p = f"model.layers.{orig_idx}"
        layer = {
            "q_w": to_jax(sd[f"{p}.self_attn.q_proj.weight"]),
            "k_w": to_jax(sd[f"{p}.self_attn.k_proj.weight"]),
            "v_w": to_jax(sd[f"{p}.self_attn.v_proj.weight"]),
            "o_w": to_jax(sd[f"{p}.self_attn.o_proj.weight"]),
            "g_w": to_jax(sd[f"{p}.mlp.gate_proj.weight"]),
            "u_w": to_jax(sd[f"{p}.mlp.up_proj.weight"]),
            "d_w": to_jax(sd[f"{p}.mlp.down_proj.weight"]),
            "ln1_w": to_jax(sd[f"{p}.input_layernorm.weight"]),
            "ln2_w": to_jax(sd[f"{p}.post_attention_layernorm.weight"]),
        }
        if has_qk_norm:
            layer["q_norm_w"] = to_jax(sd[f"{p}.self_attn.q_norm.weight"])
            layer["k_norm_w"] = to_jax(sd[f"{p}.self_attn.k_norm.weight"])

        for bias_key in ["q_proj.bias", "k_proj.bias", "v_proj.bias"]:
            full_key = f"{p}.self_attn.{bias_key}"
            if full_key in sd:
                short = bias_key.replace("_proj.", "_")
                layer[short] = to_jax(sd[full_key])

        layers.append(layer)

    del model, sd
    gc.collect()

    return {
        "wte": wte, "ln_f_w": ln_f_w, "lm_head_w": lm_head_w,
        "layers": layers, "n_layers": n_kept,
        "n_heads": n_heads, "n_kv": n_kv, "d_head": d_head,
        "d_model": d_model, "inter": inter, "has_qk_norm": has_qk_norm,
    }


# ── Forward pass ────────────────────────────────────────────────────
def rms_norm(x, w, eps=1e-6):
    ms = jnp.mean(x * x, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(ms + eps) * w


def qk_norm_fn(x, w, eps=1e-6):
    ms = jnp.mean(x * x, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(ms + eps) * w


def forward_layer(x, layer_w, lora_params, n_heads, n_kv, d_head, has_qk_norm):
    """Forward pass through one layer with optional LoRA."""
    B, S, D = x.shape
    rep = n_heads // n_kv

    h = rms_norm(x, layer_w["ln1_w"])

    # Q/K/V with LoRA: W_new = W + (B @ A) * scaling
    scaling = LORA_ALPHA / LORA_RANK

    def proj_with_lora(h_in, w_name, layer_w, lora_params):
        W = layer_w[w_name]
        out = h_in @ W.T
        if lora_params is not None and w_name in lora_params:
            lp = lora_params[w_name]
            # LoRA: h @ (W + B @ A * scaling)^T = h @ W^T + h @ A^T @ B^T * scaling
            out = out + (h_in @ lp["A"].T @ lp["B"].T) * scaling
        return out

    q = proj_with_lora(h, "q_w", layer_w, lora_params)
    k = proj_with_lora(h, "k_w", layer_w, lora_params)
    v = proj_with_lora(h, "v_w", layer_w, lora_params)

    if "q_b" in layer_w:
        q = q + layer_w["q_b"]
    if "k_b" in layer_w:
        k = k + layer_w["k_b"]
    if "v_b" in layer_w:
        v = v + layer_w["v_b"]

    q = q.reshape(B, S, n_heads, d_head)
    k = k.reshape(B, S, n_kv, d_head)
    v = v.reshape(B, S, n_kv, d_head)

    if has_qk_norm:
        q = qk_norm_fn(q, layer_w["q_norm_w"])
        k = qk_norm_fn(k, layer_w["k_norm_w"])

    # RoPE
    half = d_head // 2
    pos = jnp.arange(S)
    freqs = 1.0 / (1e6 ** (jnp.arange(0, half).astype(jnp.float32) / half))
    angles = pos[:, None] * freqs[None, :]
    cos_a = jnp.cos(angles).astype(x.dtype)[None, :, None, :]
    sin_a = jnp.sin(angles).astype(x.dtype)[None, :, None, :]

    def apply_rope(t):
        t1, t2 = t[..., :half], t[..., half:]
        return jnp.concatenate([t1 * cos_a - t2 * sin_a, t2 * cos_a + t1 * sin_a], axis=-1)

    q = apply_rope(q)
    k = apply_rope(k)

    if rep > 1:
        k = jnp.repeat(k, rep, axis=2)
        v = jnp.repeat(v, rep, axis=2)

    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    scale = math.sqrt(d_head)
    attn = (q @ k.transpose(0, 1, 3, 2)) / scale
    mask = jnp.triu(jnp.full((S, S), -1e9, dtype=x.dtype), k=1)
    attn = attn + mask[None, None, :, :]
    attn = jax.nn.softmax(attn, axis=-1)

    out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, S, D)

    # Output projection with LoRA
    out = proj_with_lora(out, "o_w", layer_w, lora_params)
    x = x + out

    # FFN (no LoRA on FFN)
    h2 = rms_norm(x, layer_w["ln2_w"])
    gate = jax.nn.silu(h2 @ layer_w["g_w"].T)
    up = h2 @ layer_w["u_w"].T
    x = x + (gate * up) @ layer_w["d_w"].T

    return x


def forward_model(tokens, weights, lora_params_list):
    """Full model forward pass with gradient checkpointing. Returns logits."""
    x = weights["wte"][tokens]
    n_heads = weights["n_heads"]
    n_kv = weights["n_kv"]
    d_head = weights["d_head"]
    has_qk_norm = weights["has_qk_norm"]

    # Create a partial with static config args to avoid tracing issues
    layer_fn = partial(forward_layer,
                       n_heads=n_heads, n_kv=n_kv,
                       d_head=d_head, has_qk_norm=has_qk_norm)

    for i, layer_w in enumerate(weights["layers"]):
        lp = lora_params_list[i] if lora_params_list is not None else None
        # Gradient checkpointing: recompute forward during backward
        x = jax.checkpoint(layer_fn)(x, layer_w, lp)

    x = rms_norm(x, weights["ln_f_w"])
    logits = x @ weights["lm_head_w"].T
    return logits


# ── LoRA initialization ────────────────────────────────────────────
def init_lora_params(weights, rng_key):
    """Initialize LoRA A/B matrices for each layer's target projections."""
    lora_params = []
    for i, layer_w in enumerate(weights["layers"]):
        layer_lora = {}
        for target in LORA_TARGETS:
            W = layer_w[target]
            out_dim, in_dim = W.shape
            rng_key, k1, k2 = jax.random.split(rng_key, 3)
            # A: (rank, in_dim), init Kaiming
            A = jax.random.normal(k1, (LORA_RANK, in_dim), dtype=jnp.bfloat16) * (1.0 / math.sqrt(LORA_RANK))
            # B: (out_dim, rank), init zeros
            B = jnp.zeros((out_dim, LORA_RANK), dtype=jnp.bfloat16)
            layer_lora[target] = {"A": A, "B": B}
        lora_params.append(layer_lora)
    return lora_params


# ── Loss function ───────────────────────────────────────────────────
def compute_loss(lora_params_flat, tokens, weights, n_layers):
    """Cross-entropy loss for next-token prediction."""
    # Unflatten lora_params
    lora_params_list = unflatten_lora(lora_params_flat, n_layers)
    logits = forward_model(tokens, weights, lora_params_list)
    # Shift logits and labels
    shift_logits = logits[:, :-1, :]
    shift_labels = tokens[:, 1:]
    # Cross-entropy
    log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, shift_labels[:, :, None], axis=-1).squeeze(-1)
    return jnp.mean(nll)


def flatten_lora(lora_params_list):
    """Flatten LoRA params into a single pytree for optax."""
    flat = {}
    for i, layer_lora in enumerate(lora_params_list):
        for target, params in layer_lora.items():
            flat[f"L{i}_{target}_A"] = params["A"]
            flat[f"L{i}_{target}_B"] = params["B"]
    return flat


def unflatten_lora(flat, n_layers):
    """Reconstruct LoRA params list from flat dict."""
    lora_params = []
    for i in range(n_layers):
        layer_lora = {}
        for target in LORA_TARGETS:
            key_a = f"L{i}_{target}_A"
            key_b = f"L{i}_{target}_B"
            if key_a in flat:
                layer_lora[target] = {"A": flat[key_a], "B": flat[key_b]}
        lora_params.append(layer_lora if layer_lora else None)
    return lora_params


def forward_model_eval(tokens, weights, lora_params_list):
    """Forward pass WITHOUT gradient checkpointing (for eval only)."""
    x = weights["wte"][tokens]
    n_heads = weights["n_heads"]
    n_kv = weights["n_kv"]
    d_head = weights["d_head"]
    has_qk_norm = weights["has_qk_norm"]

    for i, layer_w in enumerate(weights["layers"]):
        lp = lora_params_list[i] if lora_params_list is not None else None
        x = forward_layer(x, layer_w, lp, n_heads, n_kv, d_head, has_qk_norm)

    x = rms_norm(x, weights["ln_f_w"])
    logits = x @ weights["lm_head_w"].T
    return logits


# ── PPL evaluation ──────────────────────────────────────────────────
def evaluate_ppl(weights, lora_params_list, eval_tokens):
    """Compute perplexity on evaluation data."""
    total_nll = 0.0
    total_tokens = 0

    for i in range(0, len(eval_tokens), BATCH_SIZE):
        batch = eval_tokens[i:i+BATCH_SIZE]
        if len(batch) == 0:
            break
        tokens = jnp.array(batch, dtype=jnp.int32)
        logits = forward_model_eval(tokens, weights, lora_params_list)
        shift_logits = logits[:, :-1, :]
        shift_labels = tokens[:, 1:]
        log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
        nll = -jnp.take_along_axis(log_probs, shift_labels[:, :, None], axis=-1).squeeze(-1)
        total_nll += float(jnp.sum(nll))
        total_tokens += nll.size

    avg_nll = total_nll / max(total_tokens, 1)
    return math.exp(avg_nll)


# ── Data loading ────────────────────────────────────────────────────
def load_wikitext2(tokenizer, split="train", max_seqs=None):
    """Load and tokenize WikiText-2."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join([x for x in ds["text"] if x.strip()])
    enc = tokenizer.encode(text)
    # Split into chunks
    chunks = []
    for i in range(0, len(enc) - SEQ_LEN, SEQ_LEN):
        chunks.append(enc[i:i+SEQ_LEN])
        if max_seqs and len(chunks) >= max_seqs:
            break
    return chunks


# ── Main experiment ─────────────────────────────────────────────────
def main():
    t0 = time.time()
    results = {}

    log.info("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)

    log.info("Loading WikiText-2 data...")
    train_chunks = load_wikitext2(tokenizer, "train", max_seqs=2000)
    eval_chunks = load_wikitext2(tokenizer, "test", max_seqs=EVAL_SEQS)
    log.info(f"Train: {len(train_chunks)} seqs, Eval: {len(eval_chunks)} seqs")

    # ── Phase 1: Full model PPL ─────────────────────────────────────
    log.info("=" * 60)
    log.info("Phase 1: Full model baseline PPL")
    weights_full = load_qwen_weights(skip_layers=[])
    ppl_full = evaluate_ppl(weights_full, None, eval_chunks)
    log.info(f"Full model ({N_LAYERS_FULL}L) PPL: {ppl_full:.2f}")
    results["full_model_ppl"] = ppl_full

    # Free full model weights
    del weights_full
    gc.collect()

    # ── Phase 2-3: For each skip config, compute pruned + recovered PPL ─
    for config_name, skip_layers in SKIP_CONFIGS.items():
        log.info("=" * 60)
        log.info(f"Phase 2: Pruned model PPL — {config_name} (skip {skip_layers})")

        weights = load_qwen_weights(skip_layers=skip_layers)
        n_kept = weights["n_layers"]

        # Pruned PPL (before LoRA)
        ppl_pruned = evaluate_ppl(weights, None, eval_chunks)
        log.info(f"{config_name} ({n_kept}L) pruned PPL: {ppl_pruned:.2f}")
        ppl_increase = ((ppl_pruned - ppl_full) / ppl_full) * 100

        # ── Phase 3: LoRA fine-tuning ───────────────────────────────
        log.info(f"Phase 3: LoRA recovery — {config_name}")

        rng = jax.random.PRNGKey(42)
        lora_params_list = init_lora_params(weights, rng)
        lora_flat = flatten_lora(lora_params_list)

        # Count LoRA params
        n_lora = sum(p.size for p in jax.tree.leaves(lora_flat))
        log.info(f"LoRA params: {n_lora:,} (rank={LORA_RANK})")

        # Optimizer
        optimizer = optax.adamw(LR, weight_decay=0.01)
        opt_state = optimizer.init(lora_flat)

        # Training step
        @jax.jit
        def train_step(lora_flat, opt_state, tokens):
            loss, grads = jax.value_and_grad(compute_loss)(
                lora_flat, tokens, weights, n_kept
            )
            updates, opt_state = optimizer.update(grads, opt_state, lora_flat)
            lora_flat = optax.apply_updates(lora_flat, updates)
            return lora_flat, opt_state, loss

        # Training loop
        log.info(f"Training {N_STEPS} steps, batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}")
        rng_data = np.random.RandomState(42)
        losses = []

        for step in range(N_STEPS):
            # Sample random batch from training data
            idxs = rng_data.randint(0, len(train_chunks), size=BATCH_SIZE)
            batch = jnp.array([train_chunks[i] for i in idxs], dtype=jnp.int32)

            lora_flat, opt_state, loss = train_step(lora_flat, opt_state, batch)

            if step == 0 or (step + 1) % 20 == 0:
                loss_val = float(loss)
                losses.append({"step": step + 1, "loss": loss_val})
                elapsed = time.time() - t0
                log.info(f"  step {step+1}/{N_STEPS}  loss={loss_val:.4f}  elapsed={elapsed:.0f}s")

        # Evaluate recovered PPL
        lora_params_recovered = unflatten_lora(lora_flat, n_kept)
        ppl_recovered = evaluate_ppl(weights, lora_params_recovered, eval_chunks)
        log.info(f"{config_name} recovered PPL: {ppl_recovered:.2f}")

        ppl_recovery = ((ppl_pruned - ppl_recovered) / (ppl_pruned - ppl_full)) * 100 if ppl_pruned > ppl_full else 0.0

        results[config_name] = {
            "skip_layers": skip_layers,
            "n_kept": n_kept,
            "ppl_pruned": ppl_pruned,
            "ppl_recovered": ppl_recovered,
            "ppl_increase_pct": ppl_increase,
            "ppl_recovery_pct": ppl_recovery,
            "n_lora_params": n_lora,
            "training_steps": N_STEPS,
            "losses": losses,
        }
        log.info(f"  PPL increase from pruning: {ppl_increase:.1f}%")
        log.info(f"  PPL recovery with LoRA: {ppl_recovery:.1f}%")

        del weights, lora_params_list, lora_flat, opt_state
        gc.collect()

    # ── Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t0
    results["elapsed_sec"] = elapsed
    results["config"] = {
        "model": MODEL_NAME,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lr": LR,
        "n_steps": N_STEPS,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "eval_seqs": EVAL_SEQS,
        "targets": LORA_TARGETS,
    }

    out_file = OUTDIR / "lora_recovery_results.json"
    out_file.write_text(json.dumps(results, indent=2))
    log.info(f"\nResults saved to {out_file}")
    log.info(f"Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Print summary table
    log.info("\n" + "=" * 70)
    log.info(f"{'Config':<12} {'Layers':<8} {'Pruned PPL':<12} {'Recovered':<12} {'Recovery %'}")
    log.info("-" * 70)
    log.info(f"{'full':<12} {N_LAYERS_FULL:<8} {ppl_full:<12.2f} {'—':<12} {'—'}")
    for name in SKIP_CONFIGS:
        r = results[name]
        log.info(f"{name:<12} {r['n_kept']:<8} {r['ppl_pruned']:<12.2f} {r['ppl_recovered']:<12.2f} {r['ppl_recovery_pct']:.1f}%")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
