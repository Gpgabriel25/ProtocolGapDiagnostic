"""
tpu_lora_control.py — LoRA on FULL (unpruned) model as control experiment.

Addresses reviewer concern: the original LoRA recovery experiment shows PPL 
dropping below the unpruned baseline (domain adaptation confound). We need 
to apply the SAME LoRA training to the full model to disentangle domain 
adaptation from pruning recovery.

The key comparison:
  - Full model PPL (no LoRA, no training)     = P_base
  - Full model + LoRA (200 steps)             = P_full_lora  (domain adaptation)
  - Pruned model PPL                          = P_pruned
  - Pruned model + LoRA (200 steps)           = P_pruned_lora

True pruning recovery % = 1 - (P_pruned_lora - P_full_lora) / (P_pruned - P_base)
If P_pruned_lora ≈ P_full_lora, LoRA fully recovers the pruning loss.

Uses exactly the same hyperparameters as tpu_lora_recovery.py:
  - LoRA rank 16, alpha 32
  - 200 steps, AdamW lr=2e-4
  - batch_size=1, seq_len=128
  - 50 eval sequences from WikiText-2 test
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

# ── Config (SAME as tpu_lora_recovery.py) ───────────────────────────
MODEL_NAME = "Qwen/Qwen3-8B"
N_LAYERS_FULL = 36

LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGETS = ["q_w", "k_w", "v_w", "o_w"]

LR = 2e-4
N_STEPS = 200
BATCH_SIZE = 1
SEQ_LEN = 128
EVAL_SEQS = 50

HF_TOKEN_PATH = Path("/tmp/hf_token")
OUTDIR = Path("/tmp/lora_control")

# ── Setup ───────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp
import numpy as np
import optax

log.info(f"JAX version: {jax.__version__}")
log.info(f"Devices: {jax.device_count()} x {jax.devices()[0].device_kind}")

OUTDIR.mkdir(parents=True, exist_ok=True)

hf_token = None
if HF_TOKEN_PATH.exists():
    hf_token = HF_TOKEN_PATH.read_text().strip()
    os.environ["HF_TOKEN"] = hf_token


# ── Weight loading (same as tpu_lora_recovery.py) ──────────────────
def load_qwen_weights():
    """Load full Qwen3-8B weights (no layers skipped)."""
    from transformers import AutoModelForCausalLM, AutoConfig
    import torch

    log.info(f"Loading {MODEL_NAME} (full {N_LAYERS_FULL} layers)...")
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
    for idx in range(N_LAYERS_FULL):
        p = f"model.layers.{idx}"
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
        "layers": layers, "n_layers": N_LAYERS_FULL,
        "n_heads": n_heads, "n_kv": n_kv, "d_head": d_head,
        "d_model": d_model, "inter": inter, "has_qk_norm": has_qk_norm,
    }


# ── Forward pass (same as tpu_lora_recovery.py) ────────────────────
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
    scaling = LORA_ALPHA / LORA_RANK

    h = rms_norm(x, layer_w["ln1_w"])

    def proj_with_lora(h_in, w_name, layer_w, lora_params):
        W = layer_w[w_name]
        out = h_in @ W.T
        if lora_params is not None and w_name in lora_params:
            lp = lora_params[w_name]
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
    out = proj_with_lora(out, "o_w", layer_w, lora_params)
    x = x + out

    h2 = rms_norm(x, layer_w["ln2_w"])
    gate = jax.nn.silu(h2 @ layer_w["g_w"].T)
    up = h2 @ layer_w["u_w"].T
    x = x + (gate * up) @ layer_w["d_w"].T

    return x


def forward_model(tokens, weights, lora_params_list):
    """Full model forward pass with gradient checkpointing."""
    x = weights["wte"][tokens]
    layer_fn = partial(forward_layer,
                       n_heads=weights["n_heads"], n_kv=weights["n_kv"],
                       d_head=weights["d_head"], has_qk_norm=weights["has_qk_norm"])

    for i, layer_w in enumerate(weights["layers"]):
        lp = lora_params_list[i] if lora_params_list is not None else None
        x = jax.checkpoint(layer_fn)(x, layer_w, lp)

    x = rms_norm(x, weights["ln_f_w"])
    logits = x @ weights["lm_head_w"].T
    return logits


def forward_model_eval(tokens, weights, lora_params_list):
    """Forward pass WITHOUT gradient checkpointing (for eval)."""
    x = weights["wte"][tokens]

    for i, layer_w in enumerate(weights["layers"]):
        lp = lora_params_list[i] if lora_params_list is not None else None
        x = forward_layer(x, layer_w, lp,
                          weights["n_heads"], weights["n_kv"],
                          weights["d_head"], weights["has_qk_norm"])

    x = rms_norm(x, weights["ln_f_w"])
    logits = x @ weights["lm_head_w"].T
    return logits


# ── LoRA init ───────────────────────────────────────────────────────
def init_lora_params(weights, rng_key):
    lora_params = []
    for i, layer_w in enumerate(weights["layers"]):
        layer_lora = {}
        for target in LORA_TARGETS:
            W = layer_w[target]
            out_dim, in_dim = W.shape
            rng_key, k1, k2 = jax.random.split(rng_key, 3)
            A = jax.random.normal(k1, (LORA_RANK, in_dim), dtype=jnp.bfloat16) * (1.0 / math.sqrt(LORA_RANK))
            B = jnp.zeros((out_dim, LORA_RANK), dtype=jnp.bfloat16)
            layer_lora[target] = {"A": A, "B": B}
        lora_params.append(layer_lora)
    return lora_params


def flatten_lora(lora_params_list):
    flat = {}
    for i, layer_lora in enumerate(lora_params_list):
        for target, params in layer_lora.items():
            flat[f"L{i}_{target}_A"] = params["A"]
            flat[f"L{i}_{target}_B"] = params["B"]
    return flat


def unflatten_lora(flat, n_layers):
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


# ── Loss ────────────────────────────────────────────────────────────
def compute_loss(lora_params_flat, tokens, weights, n_layers):
    lora_params_list = unflatten_lora(lora_params_flat, n_layers)
    logits = forward_model(tokens, weights, lora_params_list)
    shift_logits = logits[:, :-1, :]
    shift_labels = tokens[:, 1:]
    log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, shift_labels[:, :, None], axis=-1).squeeze(-1)
    return jnp.mean(nll)


# ── PPL eval ────────────────────────────────────────────────────────
def evaluate_ppl(weights, lora_params_list, eval_tokens):
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


# ── Data ────────────────────────────────────────────────────────────
def load_wikitext2(tokenizer, split="train", max_seqs=None):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join([x for x in ds["text"] if x.strip()])
    enc = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(enc) - SEQ_LEN, SEQ_LEN):
        chunks.append(enc[i:i+SEQ_LEN])
        if max_seqs and len(chunks) >= max_seqs:
            break
    return chunks


# ── Main ────────────────────────────────────────────────────────────
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

    # ── Load full model ─────────────────────────────────────────────
    weights = load_qwen_weights()

    # ── Baseline PPL (no LoRA) ──────────────────────────────────────
    log.info("=" * 60)
    log.info("Evaluating full model baseline PPL (no LoRA)...")
    ppl_baseline = evaluate_ppl(weights, None, eval_chunks)
    log.info(f"Full model ({N_LAYERS_FULL}L) baseline PPL: {ppl_baseline:.2f}")
    results["full_baseline_ppl"] = ppl_baseline

    # ── LoRA training on full model ─────────────────────────────────
    log.info("=" * 60)
    log.info("Training LoRA on FULL (unpruned) model — control experiment")
    
    rng = jax.random.PRNGKey(42)
    lora_params_list = init_lora_params(weights, rng)
    lora_flat = flatten_lora(lora_params_list)

    n_lora = sum(p.size for p in jax.tree.leaves(lora_flat))
    log.info(f"LoRA params: {n_lora:,} (rank={LORA_RANK}, {N_LAYERS_FULL} layers)")

    optimizer = optax.adamw(LR, weight_decay=0.01)
    opt_state = optimizer.init(lora_flat)

    @jax.jit
    def train_step(lora_flat, opt_state, tokens):
        loss, grads = jax.value_and_grad(compute_loss)(
            lora_flat, tokens, weights, N_LAYERS_FULL
        )
        updates, opt_state = optimizer.update(grads, opt_state, lora_flat)
        lora_flat = optax.apply_updates(lora_flat, updates)
        return lora_flat, opt_state, loss

    log.info(f"Training {N_STEPS} steps, batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}")
    rng_data = np.random.RandomState(42)
    losses = []

    for step in range(N_STEPS):
        idxs = rng_data.randint(0, len(train_chunks), size=BATCH_SIZE)
        batch = jnp.array([train_chunks[i] for i in idxs], dtype=jnp.int32)

        lora_flat, opt_state, loss = train_step(lora_flat, opt_state, batch)

        if step == 0 or (step + 1) % 20 == 0:
            loss_val = float(loss)
            losses.append({"step": step + 1, "loss": loss_val})
            elapsed = time.time() - t0
            log.info(f"  step {step+1}/{N_STEPS}  loss={loss_val:.4f}  elapsed={elapsed:.0f}s")

    # ── Evaluate full model + LoRA PPL ──────────────────────────────
    lora_params_final = unflatten_lora(lora_flat, N_LAYERS_FULL)
    ppl_full_lora = evaluate_ppl(weights, lora_params_final, eval_chunks)
    log.info(f"Full model + LoRA PPL: {ppl_full_lora:.2f}")

    domain_adaptation_gain = ppl_baseline - ppl_full_lora
    log.info(f"Domain adaptation gain (baseline - full+LoRA): {domain_adaptation_gain:.2f}")

    results["full_lora_ppl"] = ppl_full_lora
    results["domain_adaptation_gain"] = domain_adaptation_gain
    results["n_lora_params"] = n_lora
    results["losses"] = losses
    results["elapsed_sec"] = time.time() - t0
    results["config"] = {
        "model": MODEL_NAME,
        "n_layers": N_LAYERS_FULL,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lr": LR,
        "n_steps": N_STEPS,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "eval_seqs": EVAL_SEQS,
        "targets": LORA_TARGETS,
    }

    out_file = OUTDIR / "lora_control_results.json"
    out_file.write_text(json.dumps(results, indent=2))
    log.info(f"\nResults saved to {out_file}")

    # ── Analysis (combine with prior results) ───────────────────────
    prior_file = Path("/tmp/lora_recovery/lora_recovery_results.json")
    if prior_file.exists():
        prior = json.loads(prior_file.read_text())
        log.info("\n" + "=" * 70)
        log.info("COMBINED ANALYSIS (with prior pruned+LoRA results)")
        log.info("=" * 70)
        log.info(f"{'Config':<15} {'PPL':<12} {'vs Full Base':<14} {'vs Full+LoRA':<14} {'Net Recovery %'}")
        log.info("-" * 70)
        log.info(f"{'Full (36L)':<15} {ppl_baseline:<12.2f} {'baseline':<14} {'—':<14} {'—'}")
        log.info(f"{'Full+LoRA':<15} {ppl_full_lora:<12.2f} {ppl_full_lora - ppl_baseline:<+14.2f} {'control':<14} {'—'}")
        
        for config_name in ["skip_n1", "skip_n3"]:
            if config_name in prior:
                r = prior[config_name]
                ppl_pruned = r["ppl_pruned"]
                ppl_recovered = r["ppl_recovered"]
                # True pruning-specific recovery:
                # How much of the pruning-induced PPL increase does LoRA eliminate,
                # AFTER controlling for domain adaptation?
                pruning_damage = ppl_pruned - ppl_baseline  # PPL increase from pruning
                residual_damage = ppl_recovered - ppl_full_lora  # PPL gap after LoRA (vs adapted baseline)
                if pruning_damage > 0:
                    net_recovery = (1.0 - residual_damage / pruning_damage) * 100
                else:
                    net_recovery = 0.0
                
                log.info(f"{'Pruned '+config_name:<15} {ppl_pruned:<12.2f} {ppl_pruned - ppl_baseline:<+14.2f} {'—':<14} {'—'}")
                log.info(f"{'Recov '+config_name:<15} {ppl_recovered:<12.2f} {ppl_recovered - ppl_baseline:<+14.2f} {ppl_recovered - ppl_full_lora:<+14.2f} {net_recovery:.1f}%")
        
        log.info("=" * 70)
        log.info(f"\nInterpretation:")
        log.info(f"  Full+LoRA PPL ({ppl_full_lora:.2f}) shows domain adaptation accounts for")
        log.info(f"  {domain_adaptation_gain:.2f} PPL of the improvement.")
        log.info(f"  'Net Recovery %' = pruning-specific recovery after removing domain adaptation effect.")
    else:
        log.info("Prior pruned+LoRA results not found at /tmp/lora_recovery/lora_recovery_results.json")
        log.info("Run tpu_lora_recovery.py first, then this script.")

    total_elapsed = time.time() - t0
    log.info(f"\nTotal elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
