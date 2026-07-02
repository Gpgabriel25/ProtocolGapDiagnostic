#!/usr/bin/env python3
"""Quick BLOOM-560M (ALiBi) bisimulation distance computation.
Tests all 23 adjacent pairs with 100 diverse prompts.
Purpose: Third PE type for the PE ablation study.
Uses same PROMPTS_100 as v14_gpt2_fixes.py for fair comparison."""

import torch
import numpy as np
import json
import time
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-6s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = "bigscience/bloom-560m"
N_PROMPTS = 100
MAX_LENGTH = 128
REPORT_DIR = "reports/2026-04-02T09-57-25"

def kl_div_pair(p_logits, q_logits):
    """KL(p || q) from logits."""
    p = torch.softmax(p_logits, dim=-1)
    q = torch.softmax(q_logits, dim=-1)
    p = p.clamp(min=1e-8)
    q = q.clamp(min=1e-8)
    return (p * (p.log() - q.log())).sum().item()

def generate_diverse_prompts(tokenizer, n=100, max_length=128):
    """Generate diverse prompts from multiple domains."""
    prompts = []
    templates = [
        "The history of mathematics begins with",
        "In a dark forest, the wizard discovered",
        "def fibonacci(n):\n    if n <= 1:\n        return n",
        "Breaking news: Scientists have announced",
        "The economic implications of artificial intelligence",
        "Once upon a time in a land far away",
        "import torch\nimport numpy as np\n\nclass Model:",
        "The philosophical implications of consciousness",
        "Dear valued customer, we are writing to inform",
        "The quantum mechanical properties of",
        "In the year 2050, humanity had finally",
        "The recipe calls for two cups of flour",
        "According to recent studies, the correlation",
        "The sun set behind the mountains as",
        "function calculateTotal(items) {\n  return items",
        "The political landscape has shifted dramatically",
        "Mozart's Symphony No. 40 in G minor",
        "The molecular structure of DNA consists of",
        "Hello! How can I help you today?",
        "The Industrial Revolution transformed society by",
    ]
    
    for i in range(n):
        template = templates[i % len(templates)]
        suffix = f" [{i}]" if i >= len(templates) else ""
        text = template + suffix
        encoded = tokenizer(text, return_tensors="pt", max_length=max_length,
                          truncation=True, padding="max_length")
        prompts.append(encoded)
    return prompts

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prompt_set_100 import PROMPTS_100
    
    checkpoint_path = os.path.join(REPORT_DIR, "bloom_checkpoint.json")
    
    log.info(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    
    n_layers = len(model.transformer.h)
    log.info(f"Model loaded: {n_layers} layers, ALiBi PE")
    
    # Use same prompts as v14 for fair comparison
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
        
        # Average both directions
        mean_kl = (np.mean(kl_vals_ab) + np.mean(kl_vals_ba)) / 2
        max_kl = max(np.max(kl_vals_ab), np.max(kl_vals_ba))
        median_kl = (np.median(kl_vals_ab) + np.median(kl_vals_ba)) / 2
        p95_kl = max(np.percentile(kl_vals_ab, 95), np.percentile(kl_vals_ba, 95))
        
        pair_result = {
            "layer_a": a, "layer_b": b, "gap": 1,
            "mean_kl": float(mean_kl), "max_kl": float(max_kl),
            "median_kl": float(median_kl), "p95_kl": float(p95_kl),
        }
        results.append(pair_result)
        
        elapsed = time.time() - t0
        rate = pair_count / elapsed if elapsed > 0 else 0
        remaining = (n_pairs - idx - 1) / rate if rate > 0 else 0
        log.info(f"  Pair ({a},{b}): mean_KL={mean_kl:.6f} max_KL={max_kl:.4f}  "
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
    out_path = os.path.join(REPORT_DIR, "bloom_560m_bisimulation.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    
    # Summary
    sorted_pairs = sorted(results, key=lambda p: p['mean_kl'])
    strong = sum(1 for p in results if p['mean_kl'] < 0.05)
    cond = sum(1 for p in results if 0.05 <= p['mean_kl'] < 0.10)
    
    log.info(f"\n=== BLOOM-560M (ALiBi) RESULTS ===")
    log.info(f"Best pair: {sorted_pairs[0]['layer_a']}↔{sorted_pairs[0]['layer_b']} "
             f"(mean_KL={sorted_pairs[0]['mean_kl']:.4f})")
    log.info(f"Strongly bisimilar (<0.05): {strong}")
    log.info(f"Conditionally bisimilar (0.05-0.10): {cond}")
    log.info(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
