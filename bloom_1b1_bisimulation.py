#!/usr/bin/env python3
"""BLOOM-1.1B (ALiBi) bisimulation distance computation.
Tests all 23 adjacent pairs with 100 diverse prompts.
Purpose: Second ALiBi data point to strengthen PE hierarchy claim (W5).
Uses same PROMPTS_100 as BLOOM-560M for fair comparison."""

import torch
import numpy as np
import json
import time
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-6s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = "bigscience/bloom-1b1"
N_PROMPTS = 100
MAX_LENGTH = 128
REPORT_DIR = "reports/2026-04-02T23-29-53"

def kl_div_pair(p_logits, q_logits):
    """KL(p || q) from logits."""
    p = torch.softmax(p_logits, dim=-1)
    q = torch.softmax(q_logits, dim=-1)
    p = p.clamp(min=1e-8)
    q = q.clamp(min=1e-8)
    return (p * (p.log() - q.log())).sum().item()

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prompt_set_100 import PROMPTS_100
    
    checkpoint_path = os.path.join(REPORT_DIR, "bloom_1b1_checkpoint.json")
    
    log.info(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    
    n_layers = len(model.transformer.h)
    log.info(f"Model loaded: {n_layers} layers, ALiBi PE")
    
    # Use same prompts as BLOOM-560M for fair comparison
    log.info(f"Tokenizing {N_PROMPTS} prompts from PROMPTS_100...")
    all_inputs = []
    for p in PROMPTS_100[:N_PROMPTS]:
        encoded = tokenizer(p, return_tensors="pt", max_length=MAX_LENGTH, truncation=True, padding="max_length")
        all_inputs.append(encoded)
    
    # Compute baseline logits
    log.info("Computing baseline logits...")
    baseline_logits = []
    with torch.no_grad():
        for inputs in all_inputs:
            outputs = model(**inputs)
            baseline_logits.append(outputs.logits[0, -1].clone())
    log.info("Baseline computed.")
    
    # Compute adjacent pair distances (with checkpoint/resume)
    results = []
    completed_pairs = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        results = ckpt.get("pairs", [])
        completed_pairs = {(p["layer_a"], p["layer_b"]) for p in results}
        log.info(f"Resuming from checkpoint: {len(completed_pairs)} pairs done")
    
    n_pairs = n_layers - 1
    t0 = time.time()
    pair_count = len(completed_pairs)
    
    for idx in range(n_pairs):
        a, b = idx, idx + 1
        if (a, b) in completed_pairs:
            continue
        pair_count += 1
        
        # Direction A→B: put A's weights in position B
        original_b = {k: v.clone() for k, v in model.transformer.h[b].state_dict().items()}
        model.transformer.h[b].load_state_dict(model.transformer.h[a].state_dict())
        
        kl_vals_ab = []
        for i, inputs in enumerate(all_inputs):
            with torch.no_grad():
                outputs = model(**inputs)
            swapped_logits = outputs.logits[0, -1]
            kl = kl_div_pair(baseline_logits[i], swapped_logits)
            kl_vals_ab.append(kl)
        
        model.transformer.h[b].load_state_dict(original_b)
        
        # Direction B→A: put B's weights in position A
        original_a = {k: v.clone() for k, v in model.transformer.h[a].state_dict().items()}
        model.transformer.h[a].load_state_dict(model.transformer.h[b].state_dict())
        
        kl_vals_ba = []
        for i, inputs in enumerate(all_inputs):
            with torch.no_grad():
                outputs = model(**inputs)
            swapped_logits = outputs.logits[0, -1]
            kl = kl_div_pair(baseline_logits[i], swapped_logits)
            kl_vals_ba.append(kl)
        
        model.transformer.h[a].load_state_dict(original_a)
        
        # Bidirectional max of per-direction means (consistent with paper Definition 1)
        mean_ab = np.mean(kl_vals_ab)
        mean_ba = np.mean(kl_vals_ba)
        bisim_dist = max(mean_ab, mean_ba)
        max_kl = max(np.max(kl_vals_ab), np.max(kl_vals_ba))
        median_kl = (np.median(kl_vals_ab) + np.median(kl_vals_ba)) / 2
        p95_kl = max(np.percentile(kl_vals_ab, 95), np.percentile(kl_vals_ba, 95))
        
        pair_result = {
            "layer_a": a, "layer_b": b, "gap": 1,
            "bisim_dist": float(bisim_dist),
            "mean_ab": float(mean_ab), "mean_ba": float(mean_ba),
            "max_kl": float(max_kl),
            "median_kl": float(median_kl), "p95_kl": float(p95_kl),
        }
        results.append(pair_result)
        
        elapsed = time.time() - t0
        rate = pair_count / elapsed if elapsed > 0 else 0
        remaining = (n_pairs - pair_count) / rate if rate > 0 else 0
        log.info(f"  Pair ({a},{b}): bisim_dist={bisim_dist:.6f} max_KL={max_kl:.4f}  "
                 f"[{pair_count}/{n_pairs}, ~{remaining/60:.0f}min left]")
        
        # Checkpoint every 3 pairs
        if pair_count % 3 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump({"pairs": results, "n_prompts": N_PROMPTS, "model": MODEL_NAME}, f)
            log.info(f"  Checkpoint saved ({len(results)} pairs)")
    
    # Save results
    out = {
        "model": MODEL_NAME,
        "pe_type": "ALiBi",
        "n_layers": n_layers,
        "n_prompts": N_PROMPTS,
        "pairs": results,
    }
    out_path = os.path.join(REPORT_DIR, "bloom_1b1_bisimulation.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    
    # Summary
    sorted_pairs = sorted(results, key=lambda p: p['bisim_dist'])
    strong = sum(1 for p in results if p['bisim_dist'] < 0.05)
    cond = sum(1 for p in results if 0.05 <= p['bisim_dist'] < 0.10)
    
    log.info(f"\n=== BLOOM-1.1B (ALiBi) RESULTS ===")
    log.info(f"Layers: {n_layers}")
    log.info(f"Best pair: {sorted_pairs[0]['layer_a']}↔{sorted_pairs[0]['layer_b']} "
             f"(bisim_dist={sorted_pairs[0]['bisim_dist']:.4f})")
    log.info(f"Strongly bisimilar (<0.05): {strong}")
    log.info(f"Conditionally bisimilar (0.05-0.10): {cond}")
    
    log.info("\nAll pairs (sorted):")
    for p in sorted_pairs:
        label = "STRONG" if p['bisim_dist'] < 0.05 else ("COND" if p['bisim_dist'] < 0.10 else "NON")
        log.info(f"  ({p['layer_a']},{p['layer_b']}): {p['bisim_dist']:.4f} [{label}]")
    
    log.info(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
