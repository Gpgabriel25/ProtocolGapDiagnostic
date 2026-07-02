#!/usr/bin/env python3
"""
Beam Search Layer Selection for Qwen3-8B
==========================================
Width-3 beam search over bisimulation (interchange KL) score candidates.
Uses the same matched evaluator as Table tab:skip_qwen:
  WikiText-2 test, 5K words, window=512, stride=256, JAX bf16 on TPU v6e-8

Addresses P63 reviewer concern: does beam search over bisim scores beat
SLEB-iterative at large scale?

Seed candidates: top-K=12 by interchange KL min-neighbor score
  (from 102-pair gap<=3 analysis: reports/2026-04-07T16-02-23/qwen3_8b_results.json)
Expansion: ALL 36 layers at each beam step
Beam width: 3
Reports best beam at n=1,2,3,4,5

Output JSON: /home/gpgabriel25/BisimulationQuotient/reports/2026-04-18T21-51-24/qwen3_8b_beam_search.json
"""

import os, sys, json, time, logging, gc, glob
from pathlib import Path
from collections import defaultdict

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import jax
import jax.numpy as jnp
from jax import lax
from safetensors import safe_open
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-8B"
DTYPE = jnp.bfloat16
REPORT_DIR = "/home/gpgabriel25/BisimulationQuotient/reports/2026-04-18T21-51-24"
OUTPUT_JSON = os.path.join(REPORT_DIR, "qwen3_8b_beam_search.json")
Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)

MAX_WORDS = 5000
WINDOW = 512
STRIDE = 256
BEAM_WIDTH = 3

# ── Bisim (interchange KL) scores from reports/2026-04-07T16-02-23/qwen3_8b_output/
# Per-layer min-neighbor interchange KL (gap=1 pairs, 500 prompts, 128 tokens, bf16 TPU v6e)
# Source: 102-pair analysis with bootstrap CIs
INTERCHANGE_SCORES = {
    0: 9.0243,  1: 0.5869,  2: 0.2006,  3: 0.2006,  4: 0.3886,
    5: 0.3471,  6: 0.3351,  7: 0.2157,  8: 0.2149,  9: 0.2149,
   10: 0.1836, 11: 0.1675, 12: 0.1442, 13: 0.1316, 14: 0.1150,
   15: 0.0938, 16: 0.0856, 17: 0.0778, 18: 0.0742, 19: 0.0742,
   20: 0.0612, 21: 0.0612, 22: 0.0729, 23: 0.0846, 24: 0.0863,
   25: 0.0555, 26: 0.0555, 27: 0.0582, 28: 0.0581, 29: 0.0581,
   30: 0.0802, 31: 0.0842, 32: 0.0928, 33: 0.0928, 34: 0.1985,
   35: 7.5975,
}

# Paper's known results for comparison (from Table tab:skip_qwen, matched evaluator)
PAPER_RESULTS = {
    "baseline_ppl": 12.09,
    "interchange_n1": {"layers": [17], "ppl": 12.42, "delta": 2.5},
    "interchange_n2": {"layers": [17, 21], "ppl": 12.88, "delta": 6.3},
    "bisim_clustered_n3": {"layers": [15, 17, 20], "ppl": 13.35, "delta": 10.2},
    "bisim_clustered_n5": {"layers": [15, 17, 18, 19, 20], "ppl": 17.61, "delta": 45.4},
    "sleb_iterative_n2": {"layers": [17, 18], "ppl": 13.00, "delta": 7.3},
    "sleb_iterative_n3": {"layers": [17, 18, 19], "ppl": 13.47, "delta": 11.2},
    "sleb_iterative_n5": {"layers": [17, 18, 19, 20, 21], "ppl": 17.71, "delta": 46.2},
    "sleb_greedy_n2": {"layers": [15, 20], "ppl": 12.55, "delta": 3.7},
}


# ── Model components (identical to matched_eval_qwen3.py) ─────

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


def load_and_stack(model_name, dtype=jnp.bfloat16):
    config = AutoConfig.from_pretrained(model_name)
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

    repo = snapshot_download(model_name, allow_patterns=["*.safetensors", "*.json"])
    shards = sorted(glob.glob(os.path.join(repo, "model*.safetensors")))

    idx_path = os.path.join(repo, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]

    shard_handles = {}
    all_keys = set()
    for sp in shards:
        h = safe_open(sp, framework="numpy")
        shard_handles[os.path.basename(sp)] = h
        all_keys.update(h.keys())

    def get(key):
        return shard_handles[weight_map[key]].get_tensor(key)

    has_qk_norm = "model.layers.0.self_attn.q_norm.weight" in all_keys
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
    lm_head = single_weight("lm_head.weight") if "lm_head.weight" in all_keys else embed

    shard_handles.clear()
    gc.collect()
    log.info("Weights loaded.")

    arch = {
        "n_layers": n_layers, "hidden": hidden, "n_heads": n_heads,
        "n_kv": n_kv, "head_dim": head_dim, "inter": inter,
        "rope_theta": float(rope_theta), "eps": eps, "has_qk_norm": has_qk_norm,
    }
    return arch, lw, embed, final_norm, lm_head


def load_eval_tokens(tokenizer):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in ds["text"] if isinstance(t, str) and t.strip()])
    words = text.split()
    if len(words) > MAX_WORDS:
        text = " ".join(words[:MAX_WORDS])
    tokens = tokenizer.encode(text)
    log.info(f"WikiText-2 test: {len(tokens)} tokens")
    return tokens


def evaluate_ppl(forward_fn, tokens, lw, embed, final_norm, lm_head,
                 arch, skip_mask, cos, sin, dtype=jnp.bfloat16):
    """Sliding-window PPL: window=512, stride=256."""
    seq_len = len(tokens)
    total_nll = 0.0
    total_tokens = 0
    prev_end = 0

    for begin in range(0, seq_len, STRIDE):
        end = min(begin + WINDOW, seq_len)
        target_len = end - prev_end

        chunk = tokens[begin:end]
        actual_len = len(chunk)
        if actual_len < WINDOW:
            chunk = chunk + [0] * (WINDOW - actual_len)

        input_ids = jnp.array([chunk], dtype=jnp.int32)
        logits = forward_fn(input_ids, lw, embed, final_norm, lm_head, cos, sin, skip_mask)

        shift_logits = logits[0, :actual_len - 1, :]
        shift_targets = jnp.array(tokens[begin + 1:begin + actual_len], dtype=jnp.int32)
        log_probs = jax.nn.log_softmax(shift_logits.astype(jnp.float32), axis=-1)
        ce = -log_probs[jnp.arange(len(shift_targets)), shift_targets]

        if target_len < actual_len:
            score_start = actual_len - target_len
        else:
            score_start = 0

        total_nll += float(jnp.sum(ce[score_start:]))
        total_tokens += len(ce) - score_start
        prev_end = end
        if end == seq_len:
            break

    return float(np.exp(total_nll / total_tokens)), total_tokens


def make_skip_mask(skip_set, n_layers):
    m = jnp.zeros(n_layers, dtype=jnp.bool_)
    if skip_set:
        m = m.at[jnp.array(sorted(skip_set))].set(True)
    return m


def beam_search(forward_fn, tokens, lw, embed, final_norm, lm_head,
                arch, bisim_scores, n_layers, cos, sin,
                beam_width=3, max_n=5, seed_k=12):
    """
    Beam search layer selection using bisim scores as seeds.

    seed_k: number of initial single-layer candidates (top-K by bisim score = min interchange KL)
    Beam width: keep top beam_width beams at each step.
    Expansion: try ALL n_layers at each step.
    Records best beam at each n=1..max_n.
    """
    # Cache: frozenset(skip_layers) -> (ppl, n_tokens)
    cache = {}
    eval_count = 0

    def eval_skip(skip_set):
        nonlocal eval_count
        key = frozenset(skip_set)
        if key in cache:
            return cache[key]
        sm = make_skip_mask(skip_set, n_layers)
        ppl, ntok = evaluate_ppl(forward_fn, tokens, lw, embed, final_norm, lm_head,
                                  arch, sm, cos, sin)
        cache[key] = (ppl, ntok)
        eval_count += 1
        return ppl, ntok

    # Seed: top-seed_k by bisim score (low interchange KL = more bisimilar)
    sorted_layers = sorted(bisim_scores.items(), key=lambda x: x[1])
    seed_layers = [l for l, _ in sorted_layers[:seed_k]]
    log.info(f"Seed layers (top-{seed_k} by interchange KL): {seed_layers}")

    # Initialize beams at n=1
    log.info("=== Beam search n=1 (initial seeding) ===")
    n1_beams = []
    for layer in seed_layers:
        skip_set = frozenset({layer})
        ppl, _ = eval_skip(skip_set)
        n1_beams.append((skip_set, ppl))
        log.info(f"  n=1 seed skip={sorted(skip_set)} -> PPL={ppl:.4f}")

    n1_beams.sort(key=lambda x: x[1])
    best_by_n = {1: n1_beams[0]}
    log.info(f"n=1 best: skip={sorted(n1_beams[0][0])} PPL={n1_beams[0][1]:.4f}")

    # Keep top beam_width beams
    beams = n1_beams[:beam_width]

    # Expand for n=2,3,4,5
    for step in range(2, max_n + 1):
        log.info(f"=== Beam search n={step} (expanding {len(beams)} beams) ===")
        t_step = time.time()
        new_beam_candidates = {}  # frozenset -> ppl

        for beam_skip, beam_ppl in beams:
            for layer in range(n_layers):
                if layer in beam_skip:
                    continue
                new_skip = beam_skip | {layer}
                if new_skip in new_beam_candidates:
                    continue
                ppl, _ = eval_skip(new_skip)
                new_beam_candidates[new_skip] = ppl

        # Sort and keep top beam_width
        sorted_candidates = sorted(new_beam_candidates.items(), key=lambda x: x[1])
        beams = [(skip, ppl) for skip, ppl in sorted_candidates[:beam_width]]

        best_by_n[step] = beams[0]
        elapsed = time.time() - t_step
        log.info(f"n={step} best: skip={sorted(beams[0][0])} PPL={beams[0][1]:.4f} "
                 f"({elapsed:.1f}s, {eval_count} total evals)")
        log.info(f"n={step} top-{beam_width} beams:")
        for skip, ppl in beams:
            log.info(f"  {sorted(skip)} -> PPL={ppl:.4f}")

        # Checkpoint save after each step
        save_partial(best_by_n, eval_count, cache)

    return best_by_n, eval_count, cache


def save_partial(best_by_n, eval_count, cache):
    """Save intermediate results to allow monitoring."""
    out_path = OUTPUT_JSON + ".partial"
    data = {
        "eval_count": eval_count,
        "best_by_n": {}
    }
    for n, (skip, ppl) in best_by_n.items():
        baseline = PAPER_RESULTS["baseline_ppl"]
        delta = ((ppl / baseline) - 1) * 100
        data["best_by_n"][str(n)] = {
            "layers_removed": sorted(skip),
            "ppl": round(ppl, 4),
            "delta_ppl_pct": round(delta, 2),
        }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    log.info("=== Qwen3-8B Beam Search (width=3) ===")
    log.info(f"Devices: {jax.devices()}")
    log.info(f"Output: {OUTPUT_JSON}")

    t0 = time.time()

    # Load model
    arch, lw, embed, final_norm, lm_head = load_and_stack(MODEL_NAME, DTYPE)
    n_layers = arch["n_layers"]
    t_load = time.time() - t0
    log.info(f"Model loaded in {t_load:.1f}s")

    # Load tokenizer and eval data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokens = load_eval_tokens(tokenizer)

    # Build forward function and JIT compile
    forward_fn, _ = build_forward(arch)
    log.info("JIT compile (warmup)...")
    dummy_mask = jnp.zeros(n_layers, dtype=jnp.bool_)
    dummy_ids = jnp.zeros((1, WINDOW), dtype=jnp.int32)
    cos_w, sin_w = precompute_rope(WINDOW, arch["head_dim"], arch["rope_theta"], DTYPE)
    cos_w = jax.device_put(cos_w)
    sin_w = jax.device_put(sin_w)
    warmup = forward_fn(dummy_ids, lw, embed, final_norm, lm_head, cos_w, sin_w, dummy_mask)
    jax.block_until_ready(warmup)
    log.info(f"JIT compiled. ({time.time()-t0:.1f}s total so far)")

    # Baseline PPL
    log.info("Computing baseline PPL...")
    baseline_mask = jnp.zeros(n_layers, dtype=jnp.bool_)
    baseline_ppl, n_tokens = evaluate_ppl(
        forward_fn, tokens, lw, embed, final_norm, lm_head,
        arch, baseline_mask, cos_w, sin_w
    )
    log.info(f"Baseline PPL = {baseline_ppl:.4f} (n_tokens={n_tokens})")

    # Run beam search
    t_bs = time.time()
    best_by_n, total_evals, cache = beam_search(
        forward_fn, tokens, lw, embed, final_norm, lm_head,
        arch, INTERCHANGE_SCORES, n_layers, cos_w, sin_w,
        beam_width=BEAM_WIDTH,
        max_n=5,
        seed_k=12,
    )
    t_bs_end = time.time()
    log.info(f"Beam search completed in {t_bs_end - t_bs:.1f}s, {total_evals} total evals")

    # Compile results
    results = {
        "model": MODEL_NAME,
        "method": "beam_search_bisim",
        "beam_width": BEAM_WIDTH,
        "seed_k": 12,
        "seed_strategy": "top-K by interchange KL min-neighbor",
        "evaluator": {
            "dataset": "wikitext-2-raw-v1 test",
            "max_words": MAX_WORDS,
            "window": WINDOW,
            "stride": STRIDE,
            "dtype": "bfloat16",
            "device": str(jax.devices()[0]),
        },
        "baseline_ppl": round(baseline_ppl, 4),
        "n_tokens": n_tokens,
        "total_evals": total_evals,
        "total_time_s": round(t_bs_end - t0, 1),
        "beam_search_time_s": round(t_bs_end - t_bs, 1),
        "best_by_n": {},
        "comparison": {},
    }

    log.info("\n=== RESULTS SUMMARY ===")
    log.info(f"{'n':>3} | {'Method':<30} | {'Layers':<25} | {'PPL':>8} | {'Delta%':>8}")
    log.info("-" * 85)

    for n_val in sorted(best_by_n.keys()):
        skip, ppl = best_by_n[n_val]
        delta = ((ppl / baseline_ppl) - 1) * 100
        results["best_by_n"][str(n_val)] = {
            "layers_removed": sorted(skip),
            "ppl": round(ppl, 4),
            "delta_ppl_pct": round(delta, 2),
        }
        log.info(f"{n_val:>3} | {'beam_bisim':<30} | {str(sorted(skip)):<25} | {ppl:>8.4f} | {delta:>7.1f}%")

    # Compare against paper numbers
    log.info("\n=== COMPARISON WITH PAPER METHODS ===")
    for paper_key, paper_data in PAPER_RESULTS.items():
        if paper_key == "baseline_ppl":
            continue
        n_str = paper_key.split("_n")[-1]
        try:
            n_val = int(n_str)
        except ValueError:
            continue
        if n_val not in best_by_n:
            continue
        beam_ppl, _ = best_by_n[n_val]
        beam_delta = ((beam_ppl / baseline_ppl) - 1) * 100
        paper_delta = paper_data["delta"]
        diff = beam_delta - paper_delta
        improvement = "BEAM WINS" if diff < -0.5 else ("TIED" if abs(diff) <= 0.5 else "BEAM LOSES")
        results["comparison"][paper_key] = {
            "n": n_val,
            "paper_layers": paper_data["layers"],
            "paper_delta_pct": paper_delta,
            "beam_layers": sorted(best_by_n[n_val][0]) if n_val in best_by_n else None,
            "beam_delta_pct": round(beam_delta, 2),
            "delta_diff_pct": round(diff, 2),
            "verdict": improvement,
        }
        log.info(f"  {paper_key}: paper={paper_delta:.1f}% beam={beam_delta:.1f}% -> {improvement}")

    # Save final results
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved to {OUTPUT_JSON}")

    # Remove partial file
    partial_path = OUTPUT_JSON + ".partial"
    if os.path.exists(partial_path):
        os.remove(partial_path)

    return results


if __name__ == "__main__":
    main()
