#!/usr/bin/env python3
"""
Qwen3-8B Taylor Importance Score Computation (TPU via PyTorch)
================================================================
Computes Taylor importance scores for each Qwen3-8B layer using PyTorch.
Designed to run on TPU v6e-8.

Outputs layer rankings that can be fed into the matched evaluator.

Usage on TPU:
  export MODEL_NAME=Qwen/Qwen3-8B
  export N_CALIB=32
  export OUT_DIR=/tmp/taylor_qwen
  python tpu_qwen_taylor.py
"""

import os, sys, json, time, logging, gc, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")
N_CALIB = int(os.environ.get("N_CALIB", "32"))
CALIB_MAX_LEN = int(os.environ.get("CALIB_MAX_LEN", "256"))
OUT_DIR = os.environ.get("OUT_DIR", "/tmp/taylor_qwen")
MAX_REMOVE = int(os.environ.get("MAX_REMOVE", "5"))

# Evaluation config - match the existing matched evaluator
EVAL_MAX_WORDS = int(os.environ.get("EVAL_MAX_WORDS", "5000"))
EVAL_WINDOW = 512
EVAL_STRIDE = 256


def load_calib_texts(n=N_CALIB, seed=42):
    """Load calibration texts from WikiText-2 train split."""
    import random
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if len(t.strip()) > 50]
    rng = random.Random(seed)
    return rng.sample(texts, min(n, len(texts)))


def load_eval_tokens(tokenizer, max_words=EVAL_MAX_WORDS):
    """Load WikiText-2 eval tokens."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(t for t in ds["text"] if t.strip())
    words = text.split()[:max_words]
    text = " ".join(words)
    tokens = tokenizer.encode(text, return_tensors="pt")[0]
    return tokens


def compute_taylor_importance(model, tokenizer, device):
    """
    Compute Taylor importance scores for each transformer layer.
    I_k = mean over calib samples of: sum_{p in theta_k} |grad_p * p|
    Higher = more important, Lower = more removable.
    """
    log.info("Computing Taylor importance scores...")
    calib_texts = load_calib_texts()

    # Qwen3 uses model.model.layers
    layers = model.model.layers
    n_layers = len(layers)
    layer_importance = [0.0] * n_layers
    n_processed = 0

    model.train()  # enable grad tracking

    for i, text in enumerate(calib_texts):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=CALIB_MAX_LEN, padding=False
        ).to(device)
        input_ids = inputs["input_ids"]
        if input_ids.shape[1] < 2:
            continue

        labels = input_ids.clone()
        model.zero_grad()

        with torch.enable_grad():
            out = model(**inputs, labels=labels)
            loss = out.loss
            loss.backward()

        # Collect gradient importance for each layer
        for k, layer in enumerate(layers):
            score = 0.0
            for p in layer.parameters():
                if p.grad is not None:
                    score += (p.grad * p.data).abs().sum().item()
            layer_importance[k] += score

        n_processed += 1
        if (i + 1) % 8 == 0:
            log.info(f"  Processed {i+1}/{len(calib_texts)} calib samples...")

        # Free memory
        del out, loss
        gc.collect()

    model.eval()
    layer_importance = [s / max(n_processed, 1) for s in layer_importance]
    log.info(f"Taylor scores computed over {n_processed} samples.")
    log.info(f"Top-3 most removable layers: {sorted(range(n_layers), key=lambda k: layer_importance[k])[:3]}")
    return layer_importance


def compute_ppl_skip(model, tokenizer, tokens, skip_indices, device, window=EVAL_WINDOW, stride=EVAL_STRIDE):
    """
    Evaluate perplexity with given layers skipped.
    Uses a skip mask via forward hooks for memory efficiency.
    """
    layers = model.model.layers
    n_layers = len(layers)
    handles = []

    def make_skip_fn(enabled):
        def hook(module, inp, out):
            if enabled:
                # Pass through residual (hidden_state is first element)
                hs = inp[0]
                # Return same shape as normal output
                return (hs,) + out[1:]
            return out
        return hook

    # Register hooks for layers to skip
    for k in skip_indices:
        h = layers[k].register_forward_hook(make_skip_fn(True))
        handles.append(h)

    tokens = tokens.to(device)
    n_tokens = len(tokens)
    nlls = []

    with torch.no_grad():
        for begin in range(0, n_tokens - 1, stride):
            end = min(begin + window, n_tokens)
            input_ids = tokens[begin:end].unsqueeze(0)
            if input_ids.shape[1] < 2:
                continue
            target_ids = input_ids.clone()
            # Only compute loss on the non-context part minus first token
            context_len = max(0, end - begin - stride) if begin > 0 else 0
            target_ids[0, :context_len] = -100

            out = model(input_ids=input_ids, labels=target_ids)
            nll = out.loss.item() * (end - begin - context_len - 1)
            nlls.append(nll)

    for h in handles:
        h.remove()

    total_tokens = sum(min(window, n_tokens - begin * stride) - max(0, window - stride) - 1
                       for begin in range(0, n_tokens - 1, stride)
                       if min(begin + window, n_tokens) - begin >= 2)

    if not nlls:
        return float("inf")
    nll_per_token = sum(nlls) / max(total_tokens, 1)
    return math.exp(nll_per_token)


def evaluate_ppl_simple(model, tokenizer, tokens, skip_indices, device):
    """Simpler PPL evaluation using sequence NLL sum."""
    layers = model.model.layers
    handles = []

    def make_skip_fn():
        def hook(module, inp, out):
            hs = inp[0]
            return (hs,) + out[1:]
        return hook

    for k in skip_indices:
        h = layers[k].register_forward_hook(make_skip_fn())
        handles.append(h)

    tokens = tokens.to(device)
    n_tokens = len(tokens)
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        begin = 0
        while begin < n_tokens - 1:
            end = min(begin + EVAL_WINDOW, n_tokens)
            input_ids = tokens[begin:end].unsqueeze(0)
            if input_ids.shape[1] < 2:
                break
            target_ids = input_ids.clone()
            # Mask context tokens from first window
            if begin > 0:
                context = EVAL_WINDOW - EVAL_STRIDE
                target_ids[0, :context] = -100
                n_active = end - begin - context
            else:
                n_active = end - begin - 1
            out = model(input_ids=input_ids, labels=target_ids)
            total_nll += out.loss.item() * n_active
            total_tokens += n_active
            begin += EVAL_STRIDE

    for h in handles:
        h.remove()

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            log.info(f"Using TPU: {device}")
        except ImportError:
            device = torch.device("cpu")
            log.info("Using CPU (no GPU/TPU available)")

    t0 = time.time()
    log.info(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=str(device) if "xla" not in str(device) else None,
    )
    if "xla" in str(device):
        model = model.to(device)
    model.eval()

    n_layers = len(model.model.layers)
    log.info(f"Model loaded in {time.time()-t0:.1f}s. n_layers={n_layers}")

    # Step 1: compute Taylor importance scores
    scores = compute_taylor_importance(model, tokenizer, device)

    # Rank: lowest score = most removable
    ranked = sorted(range(n_layers), key=lambda k: scores[k])
    log.info(f"Ranked layers (low=removable): {ranked[:10]}")

    # Step 2: evaluate perplexity with greedy removal
    log.info("Loading evaluation tokens...")
    tokens = load_eval_tokens(tokenizer)
    log.info(f"Eval tokens: {len(tokens)}")

    # Baseline PPL
    t1 = time.time()
    baseline_ppl = evaluate_ppl_simple(model, tokenizer, tokens, [], device)
    log.info(f"Baseline PPL: {baseline_ppl:.4f} ({time.time()-t1:.1f}s)")

    results = {
        "model": MODEL_NAME,
        "method": "taylor_importance",
        "n_calib": N_CALIB,
        "n_layers": n_layers,
        "device": str(device),
        "baseline_ppl": round(baseline_ppl, 4),
        "layer_scores": scores,
        "ranked_layers": ranked,
        "removals": {},
        "timestamp": time.strftime("%Y-%m-%dT%H-%M-%S"),
    }

    for n in range(1, MAX_REMOVE + 1):
        selected = ranked[:n]
        t1 = time.time()
        ppl = evaluate_ppl_simple(model, tokenizer, tokens, selected, device)
        delta = 100.0 * (ppl - baseline_ppl) / baseline_ppl
        elapsed = time.time() - t1
        log.info(f"  n={n}: removed={selected}, PPL={ppl:.4f}, Δ={delta:+.2f}%, t={elapsed:.1f}s")
        results["removals"][str(n)] = {
            "n_removed": n,
            "layers_removed": selected,
            "ppl": round(ppl, 4),
            "delta_ppl_pct": round(delta, 2),
            "elapsed_s": round(elapsed, 1),
        }

    results["total_elapsed_s"] = round(time.time() - t0, 1)
    out_path = f"{OUT_DIR}/qwen3_taylor_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {out_path}")

    # Summary
    print("\n=== Summary ===")
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    for n_str, r in results["removals"].items():
        print(f"  n={n_str}: PPL={r['ppl']:.4f}  Δ={r['delta_ppl_pct']:+.2f}%  layers={r['layers_removed']}")


if __name__ == "__main__":
    main()
