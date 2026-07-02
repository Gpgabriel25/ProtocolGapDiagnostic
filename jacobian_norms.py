#!/usr/bin/env python3
"""
Compute spectral norms of residual Jacobians for GPT-2-Medium layers.
Verifies the contractivity assumption in Proposition 1: ||J_k|| < 1.

For each layer k, the residual block computes f_k(x) = x + g_k(x),
so the Jacobian of the residual update is J_k = dg_k/dx.
We estimate ||J_k||_2 (spectral norm) using the power iteration method
on batches of inputs.
"""

import json, os, time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "openai-community/gpt2-medium"
NUM_PROMPTS = 50  # number of prompts to estimate over
SEQ_LEN = 64
NUM_POWER_ITERS = 10  # for spectral norm estimation
OUTPUT_FILE = "reports/2026-04-03T13-21-03/jacobian_norms.json"

def get_diverse_prompts(tokenizer, n=50, seq_len=64):
    """Generate diverse prompts for Jacobian estimation."""
    prefixes = [
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
        "Shakespeare wrote many famous plays including",
        "The theory of general relativity predicts",
        "Cooking Italian pasta requires fresh",
        "Recent advances in artificial intelligence",
        "The Amazon rainforest is home to",
        "Political philosophy examines questions about",
        "Neural networks learn representations by",
        "The Great Wall of China was built",
        "Advances in renewable energy technology",
        "The principles of thermodynamics state",
        "In modern architecture, sustainable design",
        "The evolution of language is a complex",
        "Cryptography plays a crucial role in",
        "The Mediterranean diet is known for",
        "Space exploration has revealed many",
        "The fundamentals of music theory include",
        "Quantum computing promises to solve",
        "The history of art reflects changes in",
        "Genetics research has uncovered the role",
        "The future of transportation includes",
        "Philosophy of mind explores the nature",
        "Data science combines statistics with",
        "The ocean covers approximately seventy",
        "Medieval European society was organized",
        "Blockchain technology enables decentralized",
        "The human immune system defends against",
        "Modern cosmology suggests the universe",
        "Agricultural practices have evolved from",
        "The principles of game theory apply to",
        "Nanotechnology operates at the scale of",
        "The Renaissance period saw a revival of",
        "Behavioral economics challenges the assumption",
        "Robotics engineering combines mechanical",
        "The study of linguistics reveals patterns",
        "Environmental conservation efforts focus on",
        "The development of vaccines has saved",
        "Urban planning must balance growth with",
        "The mathematics of chaos theory shows",
        "Digital privacy concerns have grown with",
        "The philosophy of science examines how",
    ]
    
    inputs = []
    for i in range(n):
        prefix = prefixes[i % len(prefixes)]
        tokens = tokenizer.encode(prefix, return_tensors="pt")
        if tokens.shape[1] > seq_len:
            tokens = tokens[:, :seq_len]
        elif tokens.shape[1] < seq_len:
            # Pad with repeated tokens
            pad_len = seq_len - tokens.shape[1]
            tokens = torch.cat([tokens, tokens[:, :pad_len]], dim=1)
        inputs.append(tokens)
    return inputs

def estimate_spectral_norm_power_iter(model, layer_idx, input_ids, n_iters=10):
    """
    Estimate spectral norm of residual Jacobian J_k = dg_k/dx
    using power iteration: ||J||_2 = max_v ||Jv|| / ||v||
    
    We use torch.autograd.functional.jvp to compute J @ v efficiently.
    """
    model.eval()
    
    # Get the hidden state input to this layer
    with torch.no_grad():
        # Run through embedding + all layers before layer_idx
        hidden = model.transformer.wte(input_ids) + model.transformer.wpe(
            torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        )
        hidden = model.transformer.drop(hidden)
        for i in range(layer_idx):
            hidden = model.transformer.h[i](hidden)[0]
    
    # Now we need the Jacobian of the residual update g_k(x) = block(x) - x
    # The block is model.transformer.h[layer_idx]
    block = model.transformer.h[layer_idx]
    
    # Power iteration to estimate spectral norm
    hidden_flat = hidden.detach().clone().view(-1)  # flatten for spectral norm
    d = hidden_flat.shape[0]
    
    # Random initial vector
    v = torch.randn(d, device=hidden.device)
    v = v / v.norm()
    
    sigma = 0.0
    for _ in range(n_iters):
        # Compute J @ v using JVP
        hidden_input = hidden.detach().clone().requires_grad_(True)
        v_reshaped = v.view(hidden.shape)
        
        # Forward pass through the block
        output = block(hidden_input)[0]
        residual = output - hidden_input  # g_k(x) = block(x) - x
        
        # Compute J @ v via backward
        residual_flat = residual.view(-1)
        Jv = torch.autograd.grad(
            residual_flat, hidden_input,
            grad_outputs=v.view(residual_flat.shape),
            create_graph=False
        )[0].view(-1)
        
        # Actually we want J^T @ v for power iteration on J^T J
        # Let's use the simpler approach: just estimate ||g_k(x+εv) - g_k(x)|| / ε
        # This is a finite-difference spectral norm estimate
        
        sigma_new = Jv.norm().item()
        if sigma_new > 0:
            v = Jv / Jv.norm()
        sigma = sigma_new
    
    return sigma

def estimate_spectral_norm_finite_diff(model, layer_idx, input_ids, n_iters=20, eps=1e-3):
    """
    Estimate spectral norm using finite differences + power iteration.
    More numerically stable than autograd for large models.
    
    ||J_k||_2 ≈ max_v ||g_k(x + εv) - g_k(x)|| / (ε||v||)
    """
    model.eval()
    
    with torch.no_grad():
        # Get hidden state at layer_idx input
        hidden = model.transformer.wte(input_ids) + model.transformer.wpe(
            torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        )
        hidden = model.transformer.drop(hidden)
        for i in range(layer_idx):
            hidden = model.transformer.h[i](hidden)[0]
        
        block = model.transformer.h[layer_idx]
        
        # Baseline residual
        output_base = block(hidden)[0]
        residual_base = (output_base - hidden).view(-1)
        
        d = hidden.numel()
        v = torch.randn(d, device=hidden.device)
        v = v / v.norm()
        
        sigma = 0.0
        for _ in range(n_iters):
            # Perturbed input
            hidden_pert = hidden + eps * v.view(hidden.shape)
            output_pert = block(hidden_pert)[0]
            residual_pert = (output_pert - hidden_pert).view(-1)
            
            # Jv ≈ (g(x+εv) - g(x)) / ε
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
    
    num_layers = model.config.n_layer
    print(f"Model has {num_layers} layers")
    
    prompts = get_diverse_prompts(tokenizer, n=NUM_PROMPTS, seq_len=SEQ_LEN)
    print(f"Generated {len(prompts)} prompts of length {SEQ_LEN}")
    
    results = {
        "model": MODEL_NAME,
        "num_layers": num_layers,
        "num_prompts": NUM_PROMPTS,
        "seq_len": SEQ_LEN,
        "method": "finite_difference_power_iteration",
        "n_power_iters": 20,
        "layers": {}
    }
    
    # For each layer, estimate spectral norm over multiple prompts
    for layer_idx in range(num_layers):
        norms = []
        t0 = time.time()
        for p_idx, input_ids in enumerate(prompts[:10]):  # Use 10 prompts per layer
            try:
                sigma = estimate_spectral_norm_finite_diff(
                    model, layer_idx, input_ids, n_iters=20, eps=1e-3
                )
                norms.append(sigma)
            except Exception as e:
                print(f"  Layer {layer_idx}, prompt {p_idx}: error {e}")
        
        mean_norm = np.mean(norms) if norms else float('nan')
        max_norm = np.max(norms) if norms else float('nan')
        min_norm = np.min(norms) if norms else float('nan')
        
        elapsed = time.time() - t0
        contractivity = "contractive" if max_norm < 1.0 else "expansive"
        
        results["layers"][str(layer_idx)] = {
            "mean_spectral_norm": float(mean_norm),
            "max_spectral_norm": float(max_norm),
            "min_spectral_norm": float(min_norm),
            "n_prompts": len(norms),
            "contractivity": contractivity,
            "elapsed_seconds": round(elapsed, 1)
        }
        
        status = "✓" if max_norm < 1.0 else "✗"
        print(f"  Layer {layer_idx:2d}: mean ||J|| = {mean_norm:.4f}, max = {max_norm:.4f}, min = {min_norm:.4f} [{contractivity}] {status} ({elapsed:.1f}s)")
    
    # Summary
    all_means = [v["mean_spectral_norm"] for v in results["layers"].values()]
    all_maxes = [v["max_spectral_norm"] for v in results["layers"].values()]
    n_contractive = sum(1 for v in results["layers"].values() if v["contractivity"] == "contractive")
    
    results["summary"] = {
        "overall_mean_norm": float(np.mean(all_means)),
        "overall_max_norm": float(np.max(all_maxes)),
        "n_contractive_layers": n_contractive,
        "n_total_layers": num_layers,
        "fraction_contractive": n_contractive / num_layers,
        "prop1_verified": n_contractive == num_layers
    }
    
    print(f"\n=== SUMMARY ===")
    print(f"Contractive layers: {n_contractive}/{num_layers}")
    print(f"Overall mean ||J||: {np.mean(all_means):.4f}")
    print(f"Overall max ||J||: {np.max(all_maxes):.4f}")
    print(f"Prop 1 verified: {n_contractive == num_layers}")
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
