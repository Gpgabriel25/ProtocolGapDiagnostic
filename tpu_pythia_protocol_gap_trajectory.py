#!/usr/bin/env python3
"""
TPU JAX protocol-gap trajectory sweep for Pythia checkpoints.

Measures the protocol gap between:
1) INTERCHANGE: swap layer positions i <-> j.
2) REPLACEMENT: run layer j at position i while position j stays layer j.

This script is designed for remote TPU launch and writes incremental JSON output
after each (model, checkpoint) tuple completes.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

# Numerical stabilization: force fp32 matmul accumulation even when weights/activations
# are bf16/fp16. Without this, deep converged models (e.g. pythia-6.9b @ step143000)
# can produce NaN aggregate metrics due to overflow in low-precision matmul accumulators.
jax.config.update("jax_default_matmul_precision", "float32")
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


DEFAULT_MODELS = [
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
]

DEFAULT_CHECKPOINTS = [
    "step0",
    "step1000",
    "step16000",
    "step64000",
    "step143000",
]

DEFAULT_OUTPUT_JSON = (
    "/home/gpgabriel25/Projects/BisimulationQuotient/reports/"
    "2026-04-18T21-51-24/protocol_gap_trajectory.json"
)

# First 32 prompts are used by default; smoke uses first 4.
PROMPTS_100 = [
    "The quick brown fox jumps over the lazy dog and then proceeds to",
    "In the year 2050, artificial intelligence had become so advanced that",
    "The stock market experienced significant volatility today as investors",
    "Scientists at CERN announced a breakthrough discovery in particle physics",
    "The recipe calls for two cups of flour, one egg, and a tablespoon of",
    "According to the latest research published in Nature, the human brain",
    "The president delivered a speech addressing the nation's economic concerns",
    "In a small village nestled between mountains, there lived an old",
    "The programming language Python was created by Guido van Rossum in",
    "Climate change continues to pose significant challenges for coastal cities",
    "The theory of relativity, proposed by Albert Einstein in 1905",
    "During the Renaissance period, art and science flourished across Europe",
    "The company reported quarterly earnings that exceeded analyst expectations",
    "In quantum mechanics, the uncertainty principle states that one cannot",
    "The novel begins with the protagonist waking up in an unfamiliar",
    "Recent advances in natural language processing have enabled machines to",
    "The ancient Romans built an extensive network of roads connecting",
    "Photosynthesis is the process by which plants convert sunlight into",
    "The basketball game went into overtime after a last-second three-pointer",
    "Machine learning algorithms can be broadly categorized into supervised and",
    "The human genome project was completed in 2003 after thirteen years of",
    "In thermodynamics, entropy is a measure of the disorder within a",
    "The Great Wall of China stretches over thousands of miles and was built to",
    "A new study published in Science suggests that dark matter may be",
    "The Fourier transform decomposes a function of time into its constituent",
    "Shakespeare wrote his most famous tragedies during the early seventeenth",
    "Deep learning models have achieved remarkable success in computer vision",
    "The Amazon rainforest produces approximately twenty percent of the world's",
    "In abstract algebra, a group is a set equipped with an operation that",
    "The International Space Station orbits the Earth at an altitude of",
    "Recent developments in quantum computing have raised concerns about the",
    "The philosopher Immanuel Kant argued that moral principles must be",
    "Graphene is a single layer of carbon atoms arranged in a hexagonal lattice",
    "The nervous system transmits signals between the brain and the rest of",
    "In macroeconomics, the Phillips curve describes the inverse relationship",
    "The Hubble Space Telescope has captured images of galaxies billions of",
    "Reinforcement learning agents learn optimal policies through trial and",
    "The French Revolution began in 1789 and fundamentally transformed the",
    "CRISPR-Cas9 gene editing technology has revolutionized molecular biology",
    "The Riemann hypothesis remains one of the most important unsolved problems",
]


def resolve_hf_token() -> Optional[str]:
    """Resolve Hugging Face token from env or common local token file."""
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        return env_token

    token_file = Path("/tmp/hf_token")
    if token_file.exists():
        token = token_file.read_text(encoding="utf-8").strip()
        if token:
            return token
    return None


def _download_if_exists(
    repo_id: str,
    filename: str,
    revision: str,
    token: Optional[str],
) -> Optional[str]:
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, token=token)
    except Exception:
        return None


def _torch_load_state_dict(path: str) -> Dict[str, Any]:
    import torch

    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"Unexpected torch checkpoint object type: {type(obj)}")
    return obj


def _tensor_to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def build_tensor_reader(model_name: str, revision: str, token: Optional[str]) -> Dict[str, Any]:
    """Build a reader that supports safetensors (single/sharded) or bin fallback."""
    single_safe = _download_if_exists(model_name, "model.safetensors", revision, token)
    if single_safe is not None:
        handle = safe_open(single_safe, framework="numpy")
        log.info("  using single safetensors checkpoint")
        return {
            "kind": "safetensors_single",
            "keys": set(handle.keys()),
            "handles": {"model.safetensors": handle},
            "weight_map": None,
        }

    safe_index_path = _download_if_exists(model_name, "model.safetensors.index.json", revision, token)
    if safe_index_path is not None:
        with open(safe_index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        shard_names = sorted(set(weight_map.values()))
        shard_paths: Dict[str, str] = {}
        for shard in shard_names:
            shard_paths[shard] = hf_hub_download(
                repo_id=model_name,
                filename=shard,
                revision=revision,
                token=token,
            )
        handles = {name: safe_open(path, framework="numpy") for name, path in shard_paths.items()}
        log.info("  using sharded safetensors checkpoint (%d shards)", len(shard_names))
        return {
            "kind": "safetensors_sharded",
            "keys": set(weight_map.keys()),
            "handles": handles,
            "weight_map": weight_map,
        }

    single_bin = _download_if_exists(model_name, "pytorch_model.bin", revision, token)
    if single_bin is not None:
        state = _torch_load_state_dict(single_bin)
        log.info("  using single pytorch_model.bin checkpoint")
        return {
            "kind": "bin_single",
            "keys": set(state.keys()),
            "state": state,
        }

    bin_index_path = _download_if_exists(model_name, "pytorch_model.bin.index.json", revision, token)
    if bin_index_path is not None:
        with open(bin_index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        shard_names = sorted(set(weight_map.values()))
        shard_paths: Dict[str, str] = {}
        for shard in shard_names:
            shard_paths[shard] = hf_hub_download(
                repo_id=model_name,
                filename=shard,
                revision=revision,
                token=token,
            )
        log.info("  using sharded pytorch_model.bin checkpoint (%d shards)", len(shard_names))
        return {
            "kind": "bin_sharded",
            "keys": set(weight_map.keys()),
            "weight_map": weight_map,
            "shard_paths": shard_paths,
            "cache": {},
            "cache_order": [],
            "cache_limit": 2,
        }

    raise FileNotFoundError(
        f"No supported weight format found for {model_name} revision={revision}. "
        "Expected safetensors or pytorch_model.bin (single or sharded)."
    )


def reader_has_key(reader: Dict[str, Any], key: str) -> bool:
    return key in reader["keys"]


def reader_get(reader: Dict[str, Any], key: str) -> np.ndarray:
    kind = reader["kind"]
    if kind == "safetensors_single":
        handle = next(iter(reader["handles"].values()))
        return handle.get_tensor(key)

    if kind == "safetensors_sharded":
        shard_name = reader["weight_map"][key]
        return reader["handles"][shard_name].get_tensor(key)

    if kind == "bin_single":
        return _tensor_to_numpy(reader["state"][key])

    if kind == "bin_sharded":
        shard_name = reader["weight_map"][key]
        cache = reader["cache"]
        order = reader["cache_order"]
        if shard_name not in cache:
            cache[shard_name] = _torch_load_state_dict(reader["shard_paths"][shard_name])
            order.append(shard_name)
            if len(order) > reader["cache_limit"]:
                old = order.pop(0)
                del cache[old]
                gc.collect()
        return _tensor_to_numpy(cache[shard_name][key])

    raise ValueError(f"Unknown reader kind: {kind}")


def release_reader(reader: Dict[str, Any]) -> None:
    kind = reader.get("kind")
    if kind in {"safetensors_single", "safetensors_sharded"}:
        reader["handles"].clear()
    if kind == "bin_single":
        reader["state"].clear()
    if kind == "bin_sharded":
        reader["cache"].clear()
        reader["cache_order"].clear()
    gc.collect()


def load_pythia_weights(
    model_name: str,
    revision: str,
    dtype: jnp.dtype,
    token: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, jax.Array], jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Load one Pythia checkpoint and return stacked layer weights + arch metadata."""
    config = AutoConfig.from_pretrained(model_name, revision=revision, token=token)
    reader = build_tensor_reader(model_name, revision, token)

    n_layers = int(config.num_hidden_layers)
    d_model = int(config.hidden_size)
    n_heads = int(config.num_attention_heads)
    d_head = d_model // n_heads

    rope_params = getattr(config, "rope_parameters", None) or {}
    partial_rotary_factor = None
    if isinstance(rope_params, dict):
        partial_rotary_factor = rope_params.get("partial_rotary_factor")
    if partial_rotary_factor is None:
        partial_rotary_factor = getattr(config, "rotary_pct", 0.25)
    rotary_ndims = int(d_head * float(partial_rotary_factor))
    rotary_ndims = max(2, rotary_ndims - (rotary_ndims % 2))

    log.info(
        "  %s @ %s -> %d layers, hidden=%d, heads=%d, d_head=%d, rotary_ndims=%d",
        model_name,
        revision,
        n_layers,
        d_model,
        n_heads,
        d_head,
        rotary_ndims,
    )

    def to_jax(arr: np.ndarray) -> jax.Array:
        arr32 = np.asarray(arr, dtype=np.float32)
        out = jax.device_put(jnp.asarray(arr32, dtype=dtype))
        return out

    def single(key: str) -> jax.Array:
        return to_jax(reader_get(reader, key))

    def stack(template: str) -> jax.Array:
        tensors: List[np.ndarray] = []
        for i in range(n_layers):
            key = template.format(i=i)
            tensors.append(np.asarray(reader_get(reader, key), dtype=np.float32))
        stacked = np.stack(tensors, axis=0)
        out = jax.device_put(jnp.asarray(stacked, dtype=dtype))
        del tensors
        del stacked
        gc.collect()
        return out

    wte = single("gpt_neox.embed_in.weight")
    ln_f_w = single("gpt_neox.final_layer_norm.weight")
    ln_f_b = single("gpt_neox.final_layer_norm.bias")
    lm_head = single("embed_out.weight") if reader_has_key(reader, "embed_out.weight") else wte

    if reader_has_key(reader, "embed_out.bias"):
        lm_head_b = single("embed_out.bias")
    else:
        lm_head_b = jnp.zeros((lm_head.shape[0],), dtype=dtype)

    layer_weights = {
        "qkv_w": stack("gpt_neox.layers.{i}.attention.query_key_value.weight"),
        "qkv_b": stack("gpt_neox.layers.{i}.attention.query_key_value.bias"),
        "o_w": stack("gpt_neox.layers.{i}.attention.dense.weight"),
        "o_b": stack("gpt_neox.layers.{i}.attention.dense.bias"),
        "ln1_w": stack("gpt_neox.layers.{i}.input_layernorm.weight"),
        "ln1_b": stack("gpt_neox.layers.{i}.input_layernorm.bias"),
        "ff1_w": stack("gpt_neox.layers.{i}.mlp.dense_h_to_4h.weight"),
        "ff1_b": stack("gpt_neox.layers.{i}.mlp.dense_h_to_4h.bias"),
        "ff2_w": stack("gpt_neox.layers.{i}.mlp.dense_4h_to_h.weight"),
        "ff2_b": stack("gpt_neox.layers.{i}.mlp.dense_4h_to_h.bias"),
        "ln2_w": stack("gpt_neox.layers.{i}.post_attention_layernorm.weight"),
        "ln2_b": stack("gpt_neox.layers.{i}.post_attention_layernorm.bias"),
    }

    release_reader(reader)

    arch = {
        "n_layers": n_layers,
        "d_model": d_model,
        "n_heads": n_heads,
        "d_head": d_head,
        "rotary_ndims": rotary_ndims,
    }
    return arch, layer_weights, wte, ln_f_w, ln_f_b, lm_head, lm_head_b


def layer_norm(x: jax.Array, w: jax.Array, b: jax.Array) -> jax.Array:
    # Compute in fp32 for numerical stability (variance can overflow/underflow in bf16
    # for converged large models).
    in_dtype = x.dtype
    x32 = x.astype(jnp.float32)
    mean = jnp.mean(x32, axis=-1, keepdims=True)
    var = jnp.var(x32, axis=-1, keepdims=True)
    normed = (x32 - mean) / jnp.sqrt(var + 1e-5)
    out = w.astype(jnp.float32) * normed + b.astype(jnp.float32)
    return out.astype(in_dtype)


def precompute_rope(seq_len: int, rotary_ndims: int, dtype: jnp.dtype) -> Tuple[jax.Array, jax.Array]:
    half_rot = rotary_ndims // 2
    freqs = 1.0 / (10000.0 ** (jnp.arange(0, half_rot, dtype=jnp.float32) / half_rot))
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    angles = positions[:, None] * freqs[None, :]
    cos = jnp.cos(angles).astype(dtype)
    sin = jnp.sin(angles).astype(dtype)
    return cos, sin


def build_forward(arch: Dict[str, Any]):
    n_heads = int(arch["n_heads"])
    d_head = int(arch["d_head"])
    d_model = int(arch["d_model"])
    n_layers = int(arch["n_layers"])
    rotary_ndims = int(arch["rotary_ndims"])
    half_rot = rotary_ndims // 2

    def one_layer(hidden: jax.Array, lw_slice: Dict[str, jax.Array], cos: jax.Array, sin: jax.Array) -> jax.Array:
        batch, seq_len, _ = hidden.shape

        h_attn = layer_norm(hidden, lw_slice["ln1_w"], lw_slice["ln1_b"])
        h_ff = layer_norm(hidden, lw_slice["ln2_w"], lw_slice["ln2_b"])

        qkv = h_attn @ lw_slice["qkv_w"].T + lw_slice["qkv_b"]
        qkv = qkv.reshape(batch, seq_len, n_heads, 3, d_head)
        q = qkv[:, :, :, 0, :]
        k = qkv[:, :, :, 1, :]
        v = qkv[:, :, :, 2, :]

        cos_b = cos[None, :seq_len, None, :]
        sin_b = sin[None, :seq_len, None, :]

        def apply_rotary(t: jax.Array) -> jax.Array:
            t_rot = t[..., :rotary_ndims]
            t_pass = t[..., rotary_ndims:]
            t1 = t_rot[..., :half_rot]
            t2 = t_rot[..., half_rot:]
            t_rot_out = jnp.concatenate([t1 * cos_b - t2 * sin_b, t2 * cos_b + t1 * sin_b], axis=-1)
            return jnp.concatenate([t_rot_out, t_pass], axis=-1)

        q = apply_rotary(q)
        k = apply_rotary(k)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * (d_head ** -0.5)

        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        neg_inf = jnp.array(-1e30, dtype=attn.dtype)
        attn = jnp.where(mask[None, None], attn, neg_inf)
        attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(hidden.dtype)

        out = jnp.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
        out = out @ lw_slice["o_w"].T + lw_slice["o_b"]

        ff = h_ff @ lw_slice["ff1_w"].T + lw_slice["ff1_b"]
        ff = jax.nn.gelu(ff, approximate=False)
        ff = ff @ lw_slice["ff2_w"].T + lw_slice["ff2_b"]

        # Keep residual stream in fp32 to avoid bf16 accumulation overflow across deep stacks.
        return hidden.astype(jnp.float32) + out.astype(jnp.float32) + ff.astype(jnp.float32)

    @jax.jit
    def gpt_neox_forward(
        input_ids: jax.Array,
        layer_weights: Dict[str, jax.Array],
        wte: jax.Array,
        ln_f_w: jax.Array,
        ln_f_b: jax.Array,
        lm_head: jax.Array,
        lm_head_b: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        layer_a: jax.Array,
        layer_b: jax.Array,
        mode: jax.Array,
    ) -> jax.Array:
        """
        mode=0: baseline
        mode=1: interchange (a <-> b)
        mode=2: replacement (a <- b, while b unchanged)
        """
        hidden = wte[input_ids].astype(jnp.float32)

        def scan_body(carry: jax.Array, idx: jax.Array):
            interchange_idx = jnp.where(idx == layer_a, layer_b, jnp.where(idx == layer_b, layer_a, idx))
            replacement_idx = jnp.where(idx == layer_a, layer_b, idx)
            run_idx = jnp.where(mode == 1, interchange_idx, jnp.where(mode == 2, replacement_idx, idx))

            lw_slice = jax.tree.map(lambda w: w[run_idx], layer_weights)
            next_hidden = one_layer(carry, lw_slice, cos, sin)
            return next_hidden, None

        indices = jnp.arange(n_layers, dtype=jnp.int32)
        hidden, _ = lax.scan(scan_body, hidden, indices)
        hidden = layer_norm(hidden, ln_f_w, ln_f_b)
        logits = hidden @ lm_head.T
        logits = logits + lm_head_b[None, None, :]
        return logits

    return gpt_neox_forward


@jax.jit
def kl_per_prompt(logits_p: jax.Array, logits_q: jax.Array) -> jax.Array:
    """KL(P || Q) averaged over tokens, returned per prompt (batch element)."""
    log_p = jax.nn.log_softmax(logits_p.astype(jnp.float32), axis=-1)
    log_q = jax.nn.log_softmax(logits_q.astype(jnp.float32), axis=-1)
    p = jnp.exp(log_p)
    kl_tok = jnp.sum(p * (log_p - log_q), axis=-1)
    return jnp.mean(kl_tok, axis=-1)


def tokenize_prompts(
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    max_length: int,
) -> jax.Array:
    enc = tokenizer(
        list(prompts),
        return_tensors="np",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    return jax.device_put(jnp.asarray(enc["input_ids"], dtype=jnp.int32))


def safe_pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if x.size < 2 or y.size < 2:
        return None
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    r = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(r):
        return None
    return r


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write prevents truncated JSON when remote disk/network hiccups occur.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def is_finite_scalar(x: float) -> bool:
    return bool(np.isfinite(np.asarray(x, dtype=np.float64)))


def run_model_checkpoint(
    model_name: str,
    checkpoint: str,
    dtype: jnp.dtype,
    token: Optional[str],
    input_ids: jax.Array,
    max_length: int,
    smoke_pairs: Optional[int] = None,
    max_gap: Optional[int] = 1,
    all_pairs: bool = False,
) -> Dict[str, Any]:
    t0 = time.time()
    arch, layer_weights, wte, ln_f_w, ln_f_b, lm_head, lm_head_b = load_pythia_weights(
        model_name=model_name,
        revision=checkpoint,
        dtype=dtype,
        token=token,
    )

    forward = build_forward(arch)
    cos, sin = precompute_rope(max_length, int(arch["rotary_ndims"]), dtype)

    # Warm/compile baseline path once.
    logits_base = forward(
        input_ids,
        layer_weights,
        wte,
        ln_f_w,
        ln_f_b,
        lm_head,
        lm_head_b,
        cos,
        sin,
        jnp.int32(0),
        jnp.int32(0),
        jnp.int32(0),
    )
    logits_base.block_until_ready()

    n_layers = int(arch["n_layers"])
    if all_pairs:
        pairs: List[Tuple[int, int]] = [(i, j) for i in range(n_layers - 1) for j in range(i + 1, n_layers)]
    elif max_gap is not None and max_gap > 1:
        pairs = [
            (i, j)
            for i in range(n_layers - 1)
            for j in range(i + 1, min(n_layers, i + max_gap + 1))
        ]
    else:
        pairs = [(i, i + 1) for i in range(n_layers - 1)]
    if smoke_pairs is not None:
        pairs = pairs[:smoke_pairs]

    all_interchange: List[float] = []
    all_replacement: List[float] = []
    pair_interchange: List[float] = []
    pair_replacement: List[float] = []
    per_pair: List[Dict[str, Any]] = []

    for pair_idx, (layer_a, layer_b) in enumerate(pairs, start=1):
        log.info(
            "  [%d/%d] model=%s checkpoint=%s pair=(%d,%d)",
            pair_idx,
            len(pairs),
            model_name,
            checkpoint,
            layer_a,
            layer_b,
        )

        logits_interchange = forward(
            input_ids,
            layer_weights,
            wte,
            ln_f_w,
            ln_f_b,
            lm_head,
            lm_head_b,
            cos,
            sin,
            jnp.int32(layer_a),
            jnp.int32(layer_b),
            jnp.int32(1),
        )

        logits_replacement = forward(
            input_ids,
            layer_weights,
            wte,
            ln_f_w,
            ln_f_b,
            lm_head,
            lm_head_b,
            cos,
            sin,
            jnp.int32(layer_a),
            jnp.int32(layer_b),
            jnp.int32(2),
        )

        kl_interchange_prompt = np.asarray(
            jax.device_get(kl_per_prompt(logits_base, logits_interchange)),
            dtype=np.float64,
        )
        kl_replacement_prompt = np.asarray(
            jax.device_get(kl_per_prompt(logits_base, logits_replacement)),
            dtype=np.float64,
        )

        interchange_kl = float(np.mean(kl_interchange_prompt))
        replacement_kl = float(np.mean(kl_replacement_prompt))

        pair_interchange.append(interchange_kl)
        pair_replacement.append(replacement_kl)
        all_interchange.extend(kl_interchange_prompt.tolist())
        all_replacement.extend(kl_replacement_prompt.tolist())

        per_pair.append(
            {
                "layer_a": int(layer_a),
                "layer_b": int(layer_b),
                "interchange_kl": interchange_kl,
                "replacement_kl": replacement_kl,
            }
        )

    mean_interchange = float(np.mean(np.asarray(all_interchange, dtype=np.float64)))
    mean_replacement = float(np.mean(np.asarray(all_replacement, dtype=np.float64)))
    gap_kl = mean_replacement - mean_interchange

    if not (is_finite_scalar(mean_interchange) and is_finite_scalar(mean_replacement) and is_finite_scalar(gap_kl)):
        raise ValueError(
            f"Non-finite aggregate metrics for {model_name}@{checkpoint}: "
            f"inter={mean_interchange}, repl={mean_replacement}, gap={gap_kl}"
        )

    pearson_r = safe_pearson(
        np.asarray(pair_interchange, dtype=np.float64),
        np.asarray(pair_replacement, dtype=np.float64),
    )

    result = {
        "model": model_name,
        "checkpoint": checkpoint,
        "n_layers": n_layers,
        "n_pairs": len(per_pair),
        "mean_interchange_kl": mean_interchange,
        "mean_replacement_kl": mean_replacement,
        "gap_kl": gap_kl,
        "pearson_r": pearson_r,
        "per_pair": per_pair,
        "wall_time_s": round(time.time() - t0, 2),
    }

    del layer_weights, wte, ln_f_w, ln_f_b, lm_head, lm_head_b, logits_base
    gc.collect()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Protocol gap trajectory on Pythia checkpoints (JAX TPU)")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--checkpoints", nargs="*", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--prompts", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--max-gap",
        type=int,
        default=1,
        help="Maximum layer distance |j-i| included in pair set (default: 1, adjacent-only).",
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Evaluate all i<j layer pairs (overrides --max-gap).",
    )
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--continue-from-existing",
        action="store_true",
        help="If output JSON exists, load it and skip already-completed model/checkpoint tuples.",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dtype_map = {
        "bfloat16": jnp.bfloat16,
        "float16": jnp.float16,
        "float32": jnp.float32,
    }
    dtype = dtype_map[args.dtype]

    token = resolve_hf_token()

    models = list(args.models)
    checkpoints = list(args.checkpoints)
    n_prompts = int(args.prompts)
    smoke_pairs: Optional[int] = None

    if args.smoke:
        models = ["EleutherAI/pythia-410m"]
        checkpoints = ["step143000"]
        n_prompts = 4
        smoke_pairs = 3
        log.info("SMOKE MODE enabled: model=pythia-410m, checkpoint=step143000, prompts=4, pairs=3")

    prompts = PROMPTS_100[:n_prompts]
    if len(prompts) < n_prompts:
        raise ValueError(f"Requested {n_prompts} prompts but only {len(PROMPTS_100)} are available")

    output_path = Path(args.output_json)

    payload: Dict[str, Any]
    done_keys: set[Tuple[str, str]] = set()
    if args.continue_from_existing and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for row in payload.get("results", []):
            model_k = row.get("model")
            ckpt_k = row.get("checkpoint")
            if isinstance(model_k, str) and isinstance(ckpt_k, str):
                done_keys.add((model_k, ckpt_k))
        log.info("Loaded existing JSON with %d completed tuples", len(done_keys))
    else:
        payload = {
            "config": {
                "prompts": n_prompts,
                "max_length": int(args.max_length),
                "dtype": args.dtype,
                "models": models,
                "checkpoints": checkpoints,
                "smoke": bool(args.smoke),
            },
            "results": [],
        }

    log.info("JAX backend: %s", jax.default_backend())
    log.info("JAX devices (%d): %s", len(jax.devices()), jax.devices())

    for model_name in models:
        log.info("Preparing tokenizer for %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        input_ids = tokenize_prompts(tokenizer, prompts, max_length=int(args.max_length))

        for checkpoint in checkpoints:
            if (model_name, checkpoint) in done_keys:
                log.info("Skipping already-complete tuple model=%s checkpoint=%s", model_name, checkpoint)
                continue
            ckpt_start = time.time()
            log.info("Starting model=%s checkpoint=%s", model_name, checkpoint)
            try:
                result = run_model_checkpoint(
                    model_name=model_name,
                    checkpoint=checkpoint,
                    dtype=dtype,
                    token=token,
                    input_ids=input_ids,
                    max_length=int(args.max_length),
                    smoke_pairs=smoke_pairs,
                    max_gap=(None if args.all_pairs else int(args.max_gap)),
                    all_pairs=bool(args.all_pairs),
                )
                payload["results"].append(result)
                done_keys.add((model_name, checkpoint))
                log.info(
                    "Completed model=%s checkpoint=%s in %.1fs | inter=%.6f repl=%.6f gap=%.6f",
                    model_name,
                    checkpoint,
                    time.time() - ckpt_start,
                    result["mean_interchange_kl"],
                    result["mean_replacement_kl"],
                    result["gap_kl"],
                )
            except Exception as exc:
                log.warning(
                    "Skipping model=%s checkpoint=%s due to failure: %s",
                    model_name,
                    checkpoint,
                    exc,
                )

            save_json(output_path, payload)
            log.info("Wrote intermediate results to %s", output_path)
            gc.collect()

    save_json(output_path, payload)
    log.info("Done. Final JSON saved to %s", output_path)


if __name__ == "__main__":
    main()
