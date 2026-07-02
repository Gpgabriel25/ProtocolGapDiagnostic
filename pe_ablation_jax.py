#!/usr/bin/env python3
"""
Controlled PE Ablation: Absolute Positional Encoding vs. RoPE — JAX version
============================================================================
Trains two architecturally IDENTICAL transformers on WikiText-103 (same data,
same init scheme) with only the positional encoding different:
  - Model A: learned absolute positional embedding (GPT-2 style)
  - Model B: rotary positional embedding (RoPE)

After training, computes interchange & replacement bisimulation distances
for layer pairs with gap <= max_gap, plus Jacobian norm estimates.

Requires: jax, flax, optax, transformers, datasets
"""

import math, json, time, random, logging, os, sys, functools
import numpy as np

import jax
import jax.numpy as jnp
from jax import random as jrandom
import flax.linen as nn
import optax

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v is not None else default

def _env_float(name, default):
    v = os.environ.get(name)
    return float(v) if v is not None else default

def _env_int_tuple(name, default):
    v = os.environ.get(name)
    if not v:
        return default
    return tuple(int(p.strip()) for p in v.split(",") if p.strip())


# ─────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────
OUTPUT_DIR = os.environ.get("PE_OUTPUT_DIR", "/tmp/pe_ablation_output")

N_LAYERS   = _env_int("PE_N_LAYERS", 12)
N_HEADS    = _env_int("PE_N_HEADS", 8)
D_MODEL    = _env_int("PE_D_MODEL", 512)
D_FF       = _env_int("PE_D_FF", 2048)
MAX_SEQ    = _env_int("PE_MAX_SEQ_LEN", 256)
VOCAB_SIZE = 50257
DROPOUT    = _env_float("PE_DROPOUT", 0.1)

TOTAL_STEPS   = _env_int("PE_TOTAL_STEPS", 25_000)
BATCH_SIZE    = _env_int("PE_BATCH_SIZE", 16)
SEQ_LEN       = _env_int("PE_SEQ_LEN", 256)
LR            = _env_float("PE_LR", 3e-4)
WARMUP_STEPS  = _env_int("PE_WARMUP_STEPS", 1000)
GRAD_CLIP     = _env_float("PE_GRAD_CLIP", 1.0)
EVAL_STEPS    = _env_int("PE_EVAL_STEPS", 2000)
LOG_STEPS     = _env_int("PE_LOG_STEPS", 500)
DISTANCE_CHECKPOINTS = _env_int_tuple("PE_DISTANCE_CHECKPOINTS", (5000, 15000, 25000))
CHECKPOINT_PROMPTS   = _env_int("PE_CHECKPOINT_PROMPTS", 100)
FINAL_PROMPTS        = _env_int("PE_FINAL_PROMPTS", 100)
JACOBIAN_PROMPTS     = _env_int("PE_JACOBIAN_PROMPTS", 10)
TRAIN_MAX_TOKENS     = _env_int("PE_TRAIN_MAX_TOKENS", 50_000_000)
EVAL_PROMPT_COUNT    = _env_int("PE_EVAL_PROMPTS", 500)
MAX_GAP              = _env_int("PE_MAX_GAP", 4)
USE_PMAP             = _env_int("PE_USE_PMAP", 1) == 1
N_DEVICES            = jax.local_device_count() if USE_PMAP else 1


# ─────────────────────────────────────────────────
#  RoPE
# ─────────────────────────────────────────────────
def build_rope_cache(dim, max_len):
    """Pre-compute sin/cos for RoPE. Returns (max_len, dim)."""
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    t = np.arange(max_len, dtype=np.float32)
    freqs = np.outer(t, inv_freq)               # (max_len, dim//2)
    emb = np.concatenate([freqs, freqs], axis=-1)  # (max_len, dim)
    return jnp.array(np.cos(emb)), jnp.array(np.sin(emb))

def apply_rope(q, k, cos, sin):
    """q, k: (B, heads, T, d_head).  cos, sin: (T, d_head)."""
    T = q.shape[2]
    cos = cos[:T][None, None, :, :]    # (1, 1, T, d_head)
    sin = sin[:T][None, None, :, :]
    def rotate_half(x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


# ─────────────────────────────────────────────────
#  Model (Flax)
# ─────────────────────────────────────────────────
class CausalSelfAttention(nn.Module):
    n_heads: int
    d_model: int
    use_rope: bool
    rope_cos: jnp.ndarray = None
    rope_sin: jnp.ndarray = None

    @nn.compact
    def __call__(self, x, deterministic=True):
        B, T, C = x.shape
        d_head = C // self.n_heads
        qkv = nn.Dense(3 * C, use_bias=False, name="qkv")(x)
        qkv = qkv.reshape(B, T, 3, self.n_heads, d_head)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q = q.squeeze(2).transpose(0, 2, 1, 3)   # (B, heads, T, d_head)
        k = k.squeeze(2).transpose(0, 2, 1, 3)
        v = v.squeeze(2).transpose(0, 2, 1, 3)

        if self.use_rope and self.rope_cos is not None:
            q, k = apply_rope(q, k, self.rope_cos, self.rope_sin)

        scale = 1.0 / jnp.sqrt(jnp.float32(d_head))
        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        mask = jnp.tril(jnp.ones((T, T)))[None, None, :, :]
        attn = jnp.where(mask, attn, jnp.float32(-1e9))
        attn = jax.nn.softmax(attn, axis=-1)
        if not deterministic:
            attn = nn.Dropout(rate=DROPOUT)(attn, deterministic=False)

        out = jnp.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        out = nn.Dense(C, use_bias=False, name="out_proj")(out)
        if not deterministic:
            out = nn.Dropout(rate=DROPOUT)(out, deterministic=False)
        return out


class FFN(nn.Module):
    d_ff: int
    d_model: int

    @nn.compact
    def __call__(self, x, deterministic=True):
        x = nn.Dense(self.d_ff, use_bias=False)(x)
        x = jax.nn.gelu(x, approximate=False)
        x = nn.Dense(self.d_model, use_bias=False)(x)
        if not deterministic:
            x = nn.Dropout(rate=DROPOUT)(x, deterministic=False)
        return x


class TransformerBlock(nn.Module):
    n_heads: int
    d_model: int
    d_ff: int
    use_rope: bool
    rope_cos: jnp.ndarray = None
    rope_sin: jnp.ndarray = None

    @nn.compact
    def __call__(self, x, deterministic=True):
        h = nn.LayerNorm()(x)
        h = CausalSelfAttention(
            n_heads=self.n_heads, d_model=self.d_model,
            use_rope=self.use_rope, rope_cos=self.rope_cos, rope_sin=self.rope_sin
        )(h, deterministic=deterministic)
        x = x + h
        h = nn.LayerNorm()(x)
        h = FFN(d_ff=self.d_ff, d_model=self.d_model)(h, deterministic=deterministic)
        x = x + h
        return x


class MiniGPT(nn.Module):
    n_layers: int
    n_heads: int
    d_model: int
    d_ff: int
    vocab_size: int
    max_seq_len: int
    use_rope: bool

    def setup(self):
        if self.use_rope:
            d_head = self.d_model // self.n_heads
            self.rope_cos, self.rope_sin = build_rope_cache(d_head, self.max_seq_len)
        else:
            self.rope_cos = None
            self.rope_sin = None

    @nn.compact
    def __call__(self, input_ids, deterministic=True):
        B, T = input_ids.shape
        tok_emb = nn.Embed(self.vocab_size, self.d_model, name="tok_emb")(input_ids)

        if not self.use_rope:
            pos = jnp.arange(T)
            pos_emb = nn.Embed(self.max_seq_len, self.d_model, name="pos_emb")(pos)
            tok_emb = tok_emb + pos_emb

        x = tok_emb
        if not deterministic:
            x = nn.Dropout(rate=DROPOUT)(x, deterministic=False)

        for i in range(self.n_layers):
            x = TransformerBlock(
                n_heads=self.n_heads, d_model=self.d_model, d_ff=self.d_ff,
                use_rope=self.use_rope, rope_cos=self.rope_cos, rope_sin=self.rope_sin,
                name=f"block_{i}"
            )(x, deterministic=deterministic)

        x = nn.LayerNorm()(x)
        # Weight-tie: reuse tok_emb
        logits = x @ self.variables["params"]["tok_emb"]["embedding"].T
        return logits


def create_model(use_rope):
    return MiniGPT(
        n_layers=N_LAYERS, n_heads=N_HEADS, d_model=D_MODEL, d_ff=D_FF,
        vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ, use_rope=use_rope,
    )


def init_params(model, rng):
    dummy = jnp.ones((1, SEQ_LEN), dtype=jnp.int32)
    params = model.init(rng, dummy, deterministic=True)["params"]
    return params


def count_params(params):
    return sum(x.size for x in jax.tree.leaves(params))


# ─────────────────────────────────────────────────
#  Forward / loss with layer-swap support
# ─────────────────────────────────────────────────
# Pre-compute RoPE caches for use outside module context
_rope_cache = {}  # Keyed by (d_head, max_len)

def _get_rope_cache(d_model, n_heads, max_len):
    d_head = d_model // n_heads
    key = (d_head, max_len)
    if key not in _rope_cache:
        _rope_cache[key] = build_rope_cache(d_head, max_len)
    return _rope_cache[key]


def forward_with_swap(model, params, input_ids, swap=None):
    """
    Forward pass with optional layer swap.
    swap is None or dict: {"mode": "interchange"|"replacement", "i": int, "j": int}

    For swap support we manually run the model internals.
    """
    B, T = input_ids.shape
    tok_emb_w = params["tok_emb"]["embedding"]
    x = tok_emb_w[input_ids]

    if not model.use_rope:
        pos = jnp.arange(T)
        pos_emb_w = params["pos_emb"]["embedding"]
        x = x + pos_emb_w[pos]

    # Pre-compute RoPE cache outside module context
    if model.use_rope:
        rope_cos, rope_sin = _get_rope_cache(model.d_model, model.n_heads, model.max_seq_len)
    else:
        rope_cos, rope_sin = None, None

    # build layer param mapping with potential swaps
    layer_params = []
    for i in range(model.n_layers):
        layer_params.append(params[f"block_{i}"])

    if swap is not None:
        mode = swap["mode"]
        si, sj = swap["i"], swap["j"]
        if mode == "interchange":
            layer_params[si], layer_params[sj] = layer_params[sj], layer_params[si]
        elif mode == "replacement":
            layer_params[si] = layer_params[sj]

    # manually apply blocks
    for i in range(model.n_layers):
        block = TransformerBlock(
            n_heads=model.n_heads, d_model=model.d_model, d_ff=model.d_ff,
            use_rope=model.use_rope,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            name=f"block_{i}"
        )
        x = block.apply({"params": layer_params[i]}, x, deterministic=True)

    # final LN — params are at top level
    ln_params = {k: v for k, v in params.items()
                 if k.startswith("LayerNorm")}
    # Flax names the final LN as "LayerNorm_0" at the top scope
    ln = nn.LayerNorm(name="LayerNorm_0")
    x = ln.apply({"params": params["LayerNorm_0"]}, x)

    logits = x @ tok_emb_w.T
    return logits


# ─────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────
def load_data_tokens(max_tokens=50_000_000):
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    log.info("Loading GPT-2 tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    log.info("Downloading WikiText-103...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    log.info("Tokenizing...")
    all_tokens = []
    for item in ds:
        text = item["text"].strip()
        if not text:
            continue
        toks = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(toks)
        all_tokens.append(tokenizer.eos_token_id)
        if len(all_tokens) >= max_tokens:
            break

    log.info(f"Total tokens: {len(all_tokens):,}")
    return np.array(all_tokens, dtype=np.int32), tokenizer


def make_batches(tokens, batch_size, seq_len, rng_key, n_batches=None):
    """Yield (x, y) batches — each is (batch_size, seq_len) int32 arrays."""
    total_chunks = len(tokens) // (seq_len + 1)
    if n_batches is None:
        n_batches = total_chunks // batch_size

    indices = np.arange(total_chunks)
    np.random.shuffle(indices)
    idx = 0

    for _ in range(n_batches):
        if idx + batch_size > len(indices):
            np.random.shuffle(indices)
            idx = 0
        batch_idx = indices[idx: idx + batch_size]
        idx += batch_size

        x = np.zeros((batch_size, seq_len), dtype=np.int32)
        y = np.zeros((batch_size, seq_len), dtype=np.int32)
        for bi, ci in enumerate(batch_idx):
            start = ci * seq_len
            chunk = tokens[start: start + seq_len + 1]
            x[bi] = chunk[:seq_len]
            y[bi] = chunk[1: seq_len + 1]

        yield jnp.array(x), jnp.array(y)


def load_eval_prompts(n=500):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    prompts = []
    for item in ds:
        text = item["text"].strip()
        if len(text) > 100:
            prompts.append(text[:512])
        if len(prompts) >= n:
            break
    return prompts


# ─────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────
def cosine_schedule(step, total_steps, warmup_steps, lr_min=1e-5, lr_max=3e-4):
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


@jax.jit
def train_step(params, opt_state, x, y, tx):
    """Single training step. Returns (new_params, new_opt_state, loss)."""
    def loss_fn(p):
        # Standard forward (no swap)
        tok_emb_w = p["tok_emb"]["embedding"]
        h = tok_emb_w[x]
        B, T = x.shape
        if "pos_emb" in p:
            pos = jnp.arange(T)
            h = h + p["pos_emb"]["embedding"][pos]

        for i in range(N_LAYERS):
            block = TransformerBlock(
                n_heads=N_HEADS, d_model=D_MODEL, d_ff=D_FF,
                use_rope=("pos_emb" not in p),
                rope_cos=rope_cos_cache if "pos_emb" not in p else None,
                rope_sin=rope_sin_cache if "pos_emb" not in p else None,
                name=f"block_{i}"
            )
            h = block.apply({"params": p[f"block_{i}"]}, h, deterministic=True)

        ln = nn.LayerNorm(name="LayerNorm_0")
        h = ln.apply({"params": p["LayerNorm_0"]}, h)
        logits = h @ tok_emb_w.T
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    grads = jax.tree.map(lambda g: jnp.clip(g, -GRAD_CLIP, GRAD_CLIP), grads)
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss


# We'll set these globals based on use_rope before JIT
rope_cos_cache = None
rope_sin_cache = None


def train_model(model, params, tokens, label, eval_prompts):
    global rope_cos_cache, rope_sin_cache

    if model.use_rope:
        d_head = D_MODEL // N_HEADS
        rope_cos_cache, rope_sin_cache = build_rope_cache(d_head, MAX_SEQ)
    else:
        rope_cos_cache, rope_sin_cache = None, None

    log.info(f"[{label}] Parameters: {count_params(params):,}")

    # Optimizer with warmup + cosine decay
    schedule_fn = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, LR, WARMUP_STEPS),
            optax.cosine_decay_schedule(LR, TOTAL_STEPS - WARMUP_STEPS, alpha=1e-5 / LR),
        ],
        boundaries=[WARMUP_STEPS],
    )
    tx = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP),
        optax.adamw(schedule_fn, b1=0.9, b2=0.95, weight_decay=0.1),
    )
    opt_state = tx.init(params)

    # Need a JIT-compiled step that captures the right rope caches
    @jax.jit
    def step_fn(params, opt_state, x, y):
        use_rope = model.use_rope

        def loss_fn(p):
            tok_emb_w = p["tok_emb"]["embedding"]
            h = tok_emb_w[x]
            B, T = x.shape
            if not use_rope:
                pos = jnp.arange(T)
                h = h + p["pos_emb"]["embedding"][pos]

            for i in range(N_LAYERS):
                block = TransformerBlock(
                    n_heads=N_HEADS, d_model=D_MODEL, d_ff=D_FF,
                    use_rope=use_rope,
                    rope_cos=rope_cos_cache if use_rope else None,
                    rope_sin=rope_sin_cache if use_rope else None,
                    name=f"block_{i}"
                )
                h = block.apply({"params": p[f"block_{i}"]}, h, deterministic=True)

            h = nn.LayerNorm(name="LayerNorm_0").apply({"params": p["LayerNorm_0"]}, h)
            logits = h @ tok_emb_w.T
            return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # ── Optionally pmap-replicate across all local devices for data-parallel ──
    if USE_PMAP and N_DEVICES > 1:
        if BATCH_SIZE % N_DEVICES != 0:
            raise ValueError(f"BATCH_SIZE={BATCH_SIZE} not divisible by N_DEVICES={N_DEVICES}")
        per_dev_bs = BATCH_SIZE // N_DEVICES
        log.info(f"[{label}] pmap data-parallel: {N_DEVICES} devices x per-device batch {per_dev_bs}")

        @functools.partial(jax.pmap, axis_name="batch")
        def pstep_fn(params, opt_state, x, y):
            use_rope = model.use_rope

            def loss_fn(p):
                tok_emb_w = p["tok_emb"]["embedding"]
                h = tok_emb_w[x]
                B, T = x.shape
                if not use_rope:
                    pos = jnp.arange(T)
                    h = h + p["pos_emb"]["embedding"][pos]
                for i in range(N_LAYERS):
                    block = TransformerBlock(
                        n_heads=N_HEADS, d_model=D_MODEL, d_ff=D_FF,
                        use_rope=use_rope,
                        rope_cos=rope_cos_cache if use_rope else None,
                        rope_sin=rope_sin_cache if use_rope else None,
                        name=f"block_{i}"
                    )
                    h = block.apply({"params": p[f"block_{i}"]}, h, deterministic=True)
                h = nn.LayerNorm(name="LayerNorm_0").apply({"params": p["LayerNorm_0"]}, h)
                logits = h @ tok_emb_w.T
                return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

            loss, grads = jax.value_and_grad(loss_fn)(params)
            grads = jax.lax.pmean(grads, axis_name="batch")
            loss  = jax.lax.pmean(loss,  axis_name="batch")
            updates, new_opt_state = tx.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss

        # Replicate state across devices
        params    = jax.device_put_replicated(params,    jax.local_devices())
        opt_state = jax.device_put_replicated(opt_state, jax.local_devices())

    losses = []
    checkpoint_distances = {}
    checkpoint_set = set(DISTANCE_CHECKPOINTS)
    t0 = time.time()
    step = 0
    rng = jrandom.PRNGKey(42)
    batch_gen = make_batches(tokens, BATCH_SIZE, SEQ_LEN, rng, n_batches=TOTAL_STEPS)

    log.info(f"[{label}] Starting training for {TOTAL_STEPS} steps...")

    use_pmap_local = USE_PMAP and N_DEVICES > 1
    per_dev_bs = (BATCH_SIZE // N_DEVICES) if use_pmap_local else BATCH_SIZE

    for x, y in batch_gen:
        if use_pmap_local:
            # Reshape (B, T) -> (N_DEVICES, B/N_DEVICES, T)
            x = x.reshape(N_DEVICES, per_dev_bs, SEQ_LEN)
            y = y.reshape(N_DEVICES, per_dev_bs, SEQ_LEN)
            params, opt_state, loss = pstep_fn(params, opt_state, x, y)
            loss_val = float(loss[0])  # all devices have the same pmean'd loss
        else:
            params, opt_state, loss = step_fn(params, opt_state, x, y)
            loss_val = float(loss)
        losses.append(loss_val)
        step += 1

        if step % LOG_STEPS == 0:
            avg_loss = sum(losses[-LOG_STEPS:]) / LOG_STEPS
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t0
            log.info(f"[{label}] step={step}/{TOTAL_STEPS} loss={avg_loss:.4f} "
                     f"ppl={ppl:.1f} elapsed={elapsed:.0f}s")

        if step in checkpoint_set and eval_prompts is not None:
            avg_loss_now = sum(losses[-min(500, len(losses)):]) / min(500, len(losses))
            ppl_now = math.exp(min(avg_loss_now, 20))
            log.info(f"[{label}] CHECKPOINT step={step}: computing distances...")
            # Unreplicate params (take device-0 copy) for single-device distance compute
            ckpt_params = jax.tree.map(lambda x: x[0], params) if use_pmap_local else params
            dists = compute_distances(model, ckpt_params, eval_prompts,
                                       n_prompts=CHECKPOINT_PROMPTS)
            all_ratio = [d["ratio"] for d in dists.values()]
            mean_ratio = float(np.mean(all_ratio))
            log.info(f"[{label}] CHECKPOINT step={step}: mean I/R ratio = {mean_ratio:.4f}")
            checkpoint_distances[step] = {
                "step": step, "loss": float(avg_loss_now), "ppl": float(ppl_now),
                "mean_I_R_ratio": mean_ratio, "pair_distances": dists,
            }

        if step >= TOTAL_STEPS:
            break

    final_loss = sum(losses[-500:]) / min(500, len(losses))
    log.info(f"[{label}] Training complete. Final loss={final_loss:.4f}")
    # Unreplicate params before returning so downstream single-device code works
    if use_pmap_local:
        params = jax.tree.map(lambda x: x[0], params)
    return params, checkpoint_distances


# ─────────────────────────────────────────────────
#  Bisimulation distances
# ─────────────────────────────────────────────────
def compute_kl(logits_orig, logits_mod):
    log_p = jax.nn.log_softmax(logits_orig.astype(jnp.float32), axis=-1)
    log_q = jax.nn.log_softmax(logits_mod.astype(jnp.float32), axis=-1)
    p = jnp.exp(log_p)
    kl = (p * (log_p - log_q)).sum(-1)
    return float(kl.mean())


def compute_distances(model, params, prompts, n_prompts=100):
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    random.shuffle(prompts)
    use_prompts = prompts[:n_prompts]

    batch_size = 16
    token_batches = []
    for i in range(0, len(use_prompts), batch_size):
        batch = use_prompts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="np", truncation=True,
                         max_length=MAX_SEQ, padding="max_length")
        token_batches.append(jnp.array(enc["input_ids"]))

    # Base logits
    base_logits_list = []
    for ids in token_batches:
        logits = forward_with_swap(model, params, ids, swap=None)
        base_logits_list.append(logits)

    pairs = [(i, j) for i in range(N_LAYERS)
             for j in range(i+1, N_LAYERS) if j - i <= MAX_GAP]
    results = {}

    for pi, (li, lj) in enumerate(pairs):
        kl_int_fwd, kl_int_bwd = [], []
        kl_rep_ij, kl_rep_ji = [], []

        for b_idx, ids in enumerate(token_batches):
            base = base_logits_list[b_idx]

            # Interchange
            swap_lgt = forward_with_swap(model, params, ids,
                                          swap={"mode": "interchange", "i": li, "j": lj})
            kl_int_fwd.append(compute_kl(base, swap_lgt))
            kl_int_bwd.append(compute_kl(swap_lgt, base))

            # Replacement i→j
            rep_ij = forward_with_swap(model, params, ids,
                                        swap={"mode": "replacement", "i": li, "j": lj})
            kl_rep_ij.append(compute_kl(base, rep_ij))

            # Replacement j→i
            rep_ji = forward_with_swap(model, params, ids,
                                        swap={"mode": "replacement", "i": lj, "j": li})
            kl_rep_ji.append(compute_kl(base, rep_ji))

        mean_int = (np.mean(kl_int_fwd) + np.mean(kl_int_bwd)) / 2
        mean_rep = (np.mean(kl_rep_ij) + np.mean(kl_rep_ji)) / 2
        ratio = mean_int / (mean_rep + 1e-10)

        results[f"{li},{lj}"] = {
            "interchange": float(mean_int),
            "replacement": float(mean_rep),
            "ratio": float(ratio),
            "strongly_bisimilar": bool(mean_int < 0.05),
        }
        if (pi + 1) % 5 == 0:
            log.info(f"  Pairs computed: {pi+1}/{len(pairs)}")

    return results


# ─────────────────────────────────────────────────
#  Jacobian norms (power iteration)
# ─────────────────────────────────────────────────
def compute_jacobian_norms(model, params, prompts, n_prompts=10, n_iters=10, eps=1e-3):
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    use_prompts = prompts[:n_prompts]
    enc = tokenizer(use_prompts, return_tensors="np", truncation=True,
                     max_length=MAX_SEQ, padding="max_length")
    input_ids = jnp.array(enc["input_ids"])

    # Pre-compute RoPE cache outside module context
    if model.use_rope:
        rope_cos, rope_sin = _get_rope_cache(model.d_model, model.n_heads, model.max_seq_len)
    else:
        rope_cos, rope_sin = None, None

    norms = {}
    for layer_idx in range(N_LAYERS):
        # Run forward to get input to this layer
        tok_emb_w = params["tok_emb"]["embedding"]
        x = tok_emb_w[input_ids]
        if not model.use_rope:
            B, T = input_ids.shape
            pos = jnp.arange(T)
            x = x + params["pos_emb"]["embedding"][pos]

        for k in range(layer_idx):
            block = TransformerBlock(
                n_heads=N_HEADS, d_model=D_MODEL, d_ff=D_FF,
                use_rope=model.use_rope,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                name=f"block_{k}"
            )
            x = block.apply({"params": params[f"block_{k}"]}, x, deterministic=True)

        # Power iteration
        rng = jrandom.PRNGKey(layer_idx)
        v = jrandom.normal(rng, x.shape)
        v = v / (jnp.linalg.norm(v) + 1e-10)

        block_params = params[f"block_{layer_idx}"]
        block = TransformerBlock(
            n_heads=N_HEADS, d_model=D_MODEL, d_ff=D_FF,
            use_rope=model.use_rope,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            name=f"block_{layer_idx}"
        )

        for _ in range(n_iters):
            f_plus = block.apply({"params": block_params}, x + eps * v, deterministic=True)
            f_minus = block.apply({"params": block_params}, x - eps * v, deterministic=True)
            Jv = (f_plus - f_minus) / (2 * eps)
            norm_Jv = jnp.linalg.norm(Jv)
            v = Jv / (norm_Jv + 1e-10)

        norms[layer_idx] = float(norm_Jv)

    return norms


# ─────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────
def main():
    t_start = time.time()
    random.seed(42)
    np.random.seed(42)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Log file
    log_path = os.path.join(OUTPUT_DIR, "pe_ablation.log")
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(fh)

    log.info("=" * 60)
    log.info("PE Ablation Experiment (JAX)")
    log.info("=" * 60)
    log.info(f"JAX version: {jax.__version__}")
    log.info(f"Devices: {jax.devices()}")
    log.info(f"Device count: {jax.device_count()}")
    log.info(f"Local device count: {jax.local_device_count()}")

    log.info(f"Model: layers={N_LAYERS} heads={N_HEADS} d_model={D_MODEL} "
             f"d_ff={D_FF} max_seq={MAX_SEQ}")
    log.info(f"Training: steps={TOTAL_STEPS} batch={BATCH_SIZE} seq_len={SEQ_LEN} "
             f"lr={LR} checkpoints={list(DISTANCE_CHECKPOINTS)}")

    # Load data
    log.info("Loading training data...")
    tokens, tokenizer = load_data_tokens(max_tokens=TRAIN_MAX_TOKENS)
    log.info(f"Training tokens: {len(tokens):,}")

    log.info("Loading eval prompts...")
    eval_prompts = load_eval_prompts(n=EVAL_PROMPT_COUNT)
    log.info(f"Eval prompts: {len(eval_prompts)}")

    all_results = {}

    skip_models = set(os.environ.get("PE_SKIP_MODELS", "").split(",")) - {""}
    if skip_models:
        log.info(f"Skipping models: {skip_models}")

    for model_type, use_rope in [("abspe", False), ("rope", True)]:
        if model_type in skip_models:
            log.info(f"Skipping {model_type} per PE_SKIP_MODELS")
            partial_path = os.path.join(OUTPUT_DIR, f"pe_ablation_{model_type}_partial.json")
            if os.path.exists(partial_path):
                with open(partial_path) as f:
                    all_results[model_type] = json.load(f)
                log.info(f"Loaded prior {model_type} results from {partial_path}")
            continue
        log.info(f"\n{'='*60}")
        log.info(f"Training: {model_type.upper()} ({'RoPE' if use_rope else 'Absolute PE'})")
        log.info(f"{'='*60}")

        model = create_model(use_rope)
        rng = jrandom.PRNGKey(1337)
        params = init_params(model, rng)

        params, ckpt_dists = train_model(model, params, tokens, label=model_type.upper(),
                                          eval_prompts=eval_prompts)

        # Save raw checkpoint distances IMMEDIATELY (before any extra computation)
        # so a crash in compute_distances/jacobian doesn't lose training-time data.
        ckpt_save_path = os.path.join(OUTPUT_DIR, f"pe_ablation_{model_type}_ckpt_dists.json")
        with open(ckpt_save_path, "w") as f:
            json.dump({str(k): {kk: vv for kk, vv in v.items()
                                if kk != "pair_distances"}
                       for k, v in ckpt_dists.items()}, f, indent=2, default=str)
        log.info(f"Saved checkpoint distances to {ckpt_save_path}")

        # Free pmap JIT cache before compute_distances/jacobian (replicated opt_state
        # held by the cache otherwise OOMs at FINAL_PROMPTS=200).
        import gc
        gc.collect()
        try:
            jax.clear_caches()
            log.info(f"Cleared JAX caches before final eval for {model_type}")
        except Exception as e:
            log.warning(f"jax.clear_caches failed pre-eval: {e}")

        log.info(f"Computing final pairwise distances for {model_type.upper()}...")
        distances = compute_distances(model, params, eval_prompts, n_prompts=FINAL_PROMPTS)

        log.info(f"Computing Jacobian norms for {model_type.upper()}...")
        jacobian_norms = compute_jacobian_norms(model, params, eval_prompts,
                                                  n_prompts=JACOBIAN_PROMPTS)
        log.info(f"[{model_type.upper()}] Jacobian norms: " +
                 ", ".join(f"L{k}={v:.3f}" for k, v in sorted(jacobian_norms.items())))

        all_int = [d["interchange"] for d in distances.values()]
        all_rep = [d["replacement"] for d in distances.values()]
        all_ratio = [d["ratio"] for d in distances.values()]

        log.info(f"\n[{model_type.upper()}] Final Results:")
        log.info(f"  Mean interchange distance: {np.mean(all_int):.4f}")
        log.info(f"  Mean replacement distance:  {np.mean(all_rep):.4f}")
        log.info(f"  Mean I/R ratio:             {np.mean(all_ratio):.4f}")

        emergence_trajectory = []
        for step, cd in sorted(ckpt_dists.items()):
            emergence_trajectory.append({
                "step": cd["step"], "loss": cd["loss"], "ppl": cd["ppl"],
                "mean_I_R_ratio": cd["mean_I_R_ratio"],
            })

        all_results[model_type] = {
            "config": {"use_rope": use_rope, "n_layers": N_LAYERS,
                        "d_model": D_MODEL, "n_heads": N_HEADS},
            "summary": {
                "mean_interchange": float(np.mean(all_int)),
                "mean_replacement": float(np.mean(all_rep)),
                "mean_I_R_ratio": float(np.mean(all_ratio)),
                "pairs_ratio_lt_0.5": int(sum(1 for r in all_ratio if r < 0.5)),
                "pairs_ratio_gt_0.8": int(sum(1 for r in all_ratio if r > 0.8)),
                "n_pairs": len(distances),
            },
            "pair_distances": distances,
            "jacobian_norms": {str(k): v for k, v in jacobian_norms.items()},
            "emergence_trajectory": emergence_trajectory,
            "checkpoint_distances": {str(k): {kk: vv for kk, vv in v.items()
                                              if kk != "pair_distances"}
                                     for k, v in ckpt_dists.items()},
        }

        # Save intermediate results after each model to avoid data loss on crash
        partial_path = os.path.join(OUTPUT_DIR, f"pe_ablation_{model_type}_partial.json")
        with open(partial_path, "w") as f:
            json.dump(all_results[model_type], f, indent=2, default=str)
        log.info(f"Saved intermediate results to {partial_path}")

        # Free device memory between models (opt_state, replicated params, JIT cache)
        del params, model, distances
        import gc
        gc.collect()
        try:
            jax.clear_caches()
            log.info(f"Cleared JAX caches after {model_type}")
        except Exception as e:
            log.warning(f"jax.clear_caches failed: {e}")

    # ─ Summary ──────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("PE ABLATION SUMMARY")
    log.info("=" * 60)

    abspe = all_results["abspe"]["summary"]
    rope = all_results["rope"]["summary"]

    log.info(f"\n{'Metric':<35} {'AbsPE':>10} {'RoPE':>10}")
    log.info("-" * 57)
    log.info(f"{'Mean interchange distance':<35} {abspe['mean_interchange']:>10.4f} {rope['mean_interchange']:>10.4f}")
    log.info(f"{'Mean replacement distance':<35} {abspe['mean_replacement']:>10.4f} {rope['mean_replacement']:>10.4f}")
    log.info(f"{'Mean I/R ratio':<35} {abspe['mean_I_R_ratio']:>10.4f} {rope['mean_I_R_ratio']:>10.4f}")

    log.info("\nJacobian norms (per layer):")
    for layer in range(N_LAYERS):
        a_n = float(all_results["abspe"]["jacobian_norms"].get(str(layer), 0))
        r_n = float(all_results["rope"]["jacobian_norms"].get(str(layer), 0))
        log.info(f"  L{layer}: AbsPE={a_n:.3f} RoPE={r_n:.3f}")

    log.info("\nTrajectory:")
    at = all_results["abspe"]["emergence_trajectory"]
    rt = all_results["rope"]["emergence_trajectory"]
    for a, r in zip(at, rt):
        log.info(f"  step={a['step']} AbsPE ppl={a['ppl']:.1f} I/R={a['mean_I_R_ratio']:.4f} | "
                 f"RoPE ppl={r['ppl']:.1f} I/R={r['mean_I_R_ratio']:.4f}")

    abspe_ratio = abspe["mean_I_R_ratio"]
    rope_ratio = rope["mean_I_R_ratio"]
    gap_confirmed = bool(rope_ratio < 0.5 and abspe_ratio > 0.7)

    log.info(f"\nProtocol gap confirmed: {gap_confirmed}")
    log.info(f"  AbsPE I/R ratio: {abspe_ratio:.3f}")
    log.info(f"  RoPE I/R ratio:  {rope_ratio:.3f}")

    # Save
    results_path = os.path.join(OUTPUT_DIR, "pe_ablation_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "experiment": "pe_ablation_jax",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "jax_version": jax.__version__,
            "devices": str(jax.devices()),
            "device_count": jax.device_count(),
            "training": {
                "total_steps": TOTAL_STEPS, "batch_size": BATCH_SIZE,
                "seq_len": SEQ_LEN, "n_layers": N_LAYERS, "d_model": D_MODEL,
                "dataset": "wikitext-103-raw-v1",
            },
            "results": all_results,
            "conclusion": {
                "gap_confirmed": gap_confirmed,
                "abspe_I_R_ratio": abspe_ratio,
                "rope_I_R_ratio": rope_ratio,
            }
        }, f, indent=2)

    total_time = time.time() - t_start
    log.info(f"\nTotal time: {total_time/60:.1f} minutes")
    log.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
