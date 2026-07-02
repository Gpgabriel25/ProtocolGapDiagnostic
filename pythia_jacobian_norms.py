#!/usr/bin/env python3
"""
Compute spectral norms of residual Jacobians for Pythia-410M layers.
Extends the GPT-2-Medium Jacobian analysis to a RoPE model, addressing
reviewer concern that Prop 1 mechanism was only tested on absolute-PE.

Uses finite-difference power iteration on CPU.
"""

import json, os, time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "EleutherAI/pythia-410m"
NUM_PROMPTS = 10
SEQ_LEN = 64
NUM_POWER_ITERS = 20
EPSILON = 1e-3

PROMPTS = [
    "The history of mathematics dates back to",
    "In quantum mechanics, the wave function",
    "Once upon a time in a distant kingdom",
    "The stock market experienced significant",
    "Machine learning algorithms can be broadly",
    "Climate change is affecting biodiversity",
    "The Python programming language was created",
    "In the year 2050, scientists discovered",
    "The human brain contains approximately",
    "Economic policy decisions often involve",
]


def get_position_embeddings(model, seq_len):
    """Compute RoPE (cos, sin) position embeddings for a given sequence length."""
    with torch.no_grad():
        position_ids = torch.arange(seq_len).unsqueeze(0)
        rotary_emb = model.gpt_neox.rotary_emb
        cos, sin = rotary_emb(
            torch.zeros(1, seq_len, model.config.hidden_size),
            position_ids
        )
    return cos, sin


def get_hidden_at_layer(model, input_ids, layer_idx, position_embeddings):
    """Get the hidden state (input to layer_idx) for a Pythia model."""
    with torch.no_grad():
        hidden = model.gpt_neox.embed_in(input_ids)
        for i in range(layer_idx):
            hidden = model.gpt_neox.layers[i](hidden, position_embeddings=position_embeddings)[0]
    return hidden


def estimate_spectral_norm(model, layer_idx, input_ids, position_embeddings, n_iters=20, eps=1e-3):
    """
    Estimate ||J_k||_2 for residual Jacobian J_k = dg_k/dx via finite-diff power iteration.
    g_k(x) = block(x) - x (the residual update).
    """
    model.eval()
    with torch.no_grad():
        hidden = get_hidden_at_layer(model, input_ids, layer_idx, position_embeddings)
        block = model.gpt_neox.layers[layer_idx]

        output_base = block(hidden, position_embeddings=position_embeddings)[0]
        residual_base = (output_base - hidden).view(-1)

        d = hidden.numel()
        v = torch.randn(d)
        v = v / v.norm()

        sigma = 0.0
        for _ in range(n_iters):
            hidden_pert = hidden + eps * v.view(hidden.shape)
            output_pert = block(hidden_pert, position_embeddings=position_embeddings)[0]
            residual_pert = (output_pert - hidden_pert).view(-1)

            Jv = (residual_pert - residual_base) / eps
            sigma = Jv.norm().item()
            if sigma > 0:
                v = Jv / Jv.norm()

    return sigma


def main():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()

    num_layers = model.config.num_hidden_layers
    print(f"Model: {MODEL_NAME}, {num_layers} layers")

    # Tokenize prompts
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = []
    for p in PROMPTS:
        tokens = tokenizer.encode(p, return_tensors="pt")
        if tokens.shape[1] >= SEQ_LEN:
            tokens = tokens[:, :SEQ_LEN]
        else:
            # Repeat tokens to fill SEQ_LEN
            repeats = (SEQ_LEN // tokens.shape[1]) + 1
            tokens = tokens.repeat(1, repeats)[:, :SEQ_LEN]
        inputs.append(tokens)

    # Precompute RoPE position embeddings
    pos_emb = get_position_embeddings(model, SEQ_LEN)
    print(f"RoPE position embeddings computed for seq_len={SEQ_LEN}")

    results = {"model": MODEL_NAME, "num_layers": num_layers, "layers": {}}

    total_t0 = time.time()
    for layer_idx in range(num_layers):
        norms = []
        t0 = time.time()
        for input_ids in inputs:
            sigma = estimate_spectral_norm(model, layer_idx, input_ids, pos_emb,
                                           n_iters=NUM_POWER_ITERS, eps=EPSILON)
            norms.append(sigma)

        mean_n = float(np.mean(norms))
        max_n = float(np.max(norms))
        min_n = float(np.min(norms))
        elapsed = time.time() - t0

        results["layers"][str(layer_idx)] = {
            "mean": round(mean_n, 4),
            "max": round(max_n, 4),
            "min": round(min_n, 4),
            "contractive": max_n < 1.0
        }

        tag = "✓" if max_n < 1.0 else "✗"
        print(f"  Layer {layer_idx:2d}: mean={mean_n:.3f}  max={max_n:.3f}  min={min_n:.3f}  {tag}  ({elapsed:.1f}s)")

    total_elapsed = time.time() - total_t0

    # Summary
    means = [v["mean"] for v in results["layers"].values()]
    maxes = [v["max"] for v in results["layers"].values()]
    n_contr = sum(1 for v in results["layers"].values() if v["contractive"])

    results["summary"] = {
        "overall_mean": round(float(np.mean(means)), 4),
        "overall_max": round(float(np.max(maxes)), 4),
        "contractive_layers": n_contr,
        "total_layers": num_layers,
        "total_seconds": round(total_elapsed, 1)
    }

    print(f"\n=== SUMMARY ===")
    print(f"Contractive: {n_contr}/{num_layers}")
    print(f"Mean ||J||: {np.mean(means):.3f}")
    print(f"Max ||J||: {np.max(maxes):.3f}")
    print(f"Total time: {total_elapsed:.0f}s")

    outf = "pythia_jacobian_norms.json"
    with open(outf, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {outf}")


if __name__ == "__main__":
    main()
