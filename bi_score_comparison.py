#!/usr/bin/env python3
"""
Block Influence (BI) Score Comparison
=====================================
Implements ShortGPT's Block Influence metric alongside our bisimulation
removability score. Compares layer rankings and Spearman correlation.

BI score for layer i = 1 - cos_sim(hidden_states[i], hidden_states[i+1])
where hidden_states[i] is the input to layer i (= output of layer i-1).
Higher BI = more influential (harder to remove).
Lower BI = more redundant (easier to remove).
"""

import json
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats

CYCLE_ID = "2026-03-31T00-18-24"
REPORT_DIR = f"reports/{CYCLE_ID}"

def compute_bi_scores(model, tokenizer, texts, max_length=512):
    """Compute Block Influence scores for each layer."""
    model.eval()
    num_layers = len(model.transformer.h)
    
    # Accumulate cosine similarities across texts
    bi_accum = np.zeros(num_layers)
    count = 0
    
    with torch.no_grad():
        for text in texts:
            encodings = tokenizer(text, return_tensors="pt", truncation=True,
                                  max_length=max_length)
            input_ids = encodings.input_ids
            
            # Get hidden states from all layers
            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # (num_layers+1,) tuples
            # hidden_states[0] = embedding output
            # hidden_states[i] = output of layer i-1 (= input to layer i)
            
            for layer_idx in range(num_layers):
                h_in = hidden_states[layer_idx].float()    # input to this layer
                h_out = hidden_states[layer_idx + 1].float()  # output of this layer
                
                # Flatten to (seq_len * hidden_dim)
                h_in_flat = h_in.view(-1)
                h_out_flat = h_out.view(-1)
                
                cos_sim = F.cosine_similarity(h_in_flat.unsqueeze(0), 
                                              h_out_flat.unsqueeze(0)).item()
                bi_accum[layer_idx] += (1.0 - cos_sim)
            count += 1
    
    bi_scores = bi_accum / count
    return bi_scores


def compute_bootstrap_kl(checkpoint_path, n_bootstrap=1000, seed=42):
    """Bootstrap confidence intervals for bisimulation KL divergences."""
    with open(checkpoint_path) as f:
        data = json.load(f)
    
    pairs = data['pairs']
    # Group by pair
    pair_data = {}
    for p in pairs:
        key = (p['layer_a'], p['layer_b'])
        pair_data[key] = p  # each pair has mean_kl computed over 20 prompts
    
    # For adjacent pairs, compute per-layer removability with bootstrap
    # We need the raw per-prompt KL values, but the checkpoint only stores mean_kl
    # So we'll bootstrap over the adjacent pairs themselves
    return pair_data


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)
    
    log.info("Loading GPT-2-Medium...")
    model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    model.eval()
    
    # Load evaluation texts (same diverse prompts as bisimulation)
    prompts = [
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
    ]
    
    log.info("Computing Block Influence (BI) scores...")
    bi_scores = compute_bi_scores(model, tokenizer, prompts)
    
    log.info("BI scores per layer:")
    for i, bi in enumerate(bi_scores):
        log.info(f"  Layer {i}: BI={bi:.6f}")
    
    # Load bisimulation removability scores
    checkpoint_path = os.path.join("reports/2026-03-30T15-15-07", "distance_checkpoint.json")
    with open(checkpoint_path) as f:
        data = json.load(f)
    
    # Compute per-layer removability (min adjacent KL)
    pairs = data['pairs']
    adj_kl = {}
    for p in pairs:
        a, b = p['layer_a'], p['layer_b']
        if abs(a - b) == 1:
            adj_kl[(min(a,b), max(a,b))] = p['mean_kl']
    
    num_layers = 24
    removability = np.zeros(num_layers)
    for layer in range(num_layers):
        neighbors = []
        if layer > 0:
            neighbors.append(adj_kl.get((layer-1, layer), float('inf')))
        if layer < num_layers - 1:
            neighbors.append(adj_kl.get((layer, layer+1), float('inf')))
        removability[layer] = min(neighbors)
    
    # Compare rankings
    # BI: lower = more redundant (easier to remove)
    # Removability: lower = more redundant (easier to remove)
    # So both should correlate positively: low BI ↔ low removability
    
    # Rank correlation (exclude layer 0 and 23 which are boundary)
    inner_layers = list(range(1, 23))  # layers 1-22
    bi_inner = bi_scores[inner_layers]
    rem_inner = removability[inner_layers]
    
    spearman_corr, spearman_p = stats.spearmanr(bi_inner, rem_inner)
    pearson_corr, pearson_p = stats.pearsonr(bi_inner, rem_inner)
    
    log.info(f"\n--- Rank Comparison (layers 1-22) ---")
    log.info(f"Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.6f})")
    log.info(f"Pearson correlation:  {pearson_corr:.4f} (p={pearson_p:.6f})")
    
    # Top-5 most removable by each method
    bi_rank = np.argsort(bi_scores)  # ascending = most removable first
    rem_rank = np.argsort(removability)  # ascending = most removable first
    
    log.info(f"\nTop-5 most removable layers:")
    log.info(f"  BI score:     {list(bi_rank[:5])} (scores: {[f'{bi_scores[i]:.4f}' for i in bi_rank[:5]]})")
    log.info(f"  Bisimulation: {list(rem_rank[:5])} (scores: {[f'{removability[i]:.4f}' for i in rem_rank[:5]]})")
    
    # Overlap in top-k
    for k in [5, 10]:
        overlap = len(set(bi_rank[:k]) & set(rem_rank[:k]))
        log.info(f"  Top-{k} overlap: {overlap}/{k}")
    
    # Save results
    results = {
        "bi_scores": {int(i): float(bi_scores[i]) for i in range(len(bi_scores))},
        "removability_scores": {int(i): float(removability[i]) for i in range(len(removability))},
        "spearman_rho": float(spearman_corr),
        "spearman_p": float(spearman_p),
        "pearson_r": float(pearson_corr),
        "pearson_p": float(pearson_p),
        "bi_rank_ascending": [int(x) for x in bi_rank],
        "removability_rank_ascending": [int(x) for x in rem_rank],
    }
    
    out_path = os.path.join(REPORT_DIR, "bi_score_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved: {out_path}")
    
    # Print comparison table
    log.info(f"\n{'Layer':>6} | {'BI Score':>10} | {'BI Rank':>8} | {'Bisim KL':>10} | {'Bisim Rank':>10}")
    log.info("-" * 55)
    bi_rankings = np.argsort(np.argsort(bi_scores))  # rank of each layer
    rem_rankings = np.argsort(np.argsort(removability))
    for i in range(num_layers):
        log.info(f"{i:>6} | {bi_scores[i]:>10.6f} | {bi_rankings[i]+1:>8} | {removability[i]:>10.4f} | {rem_rankings[i]+1:>10}")


if __name__ == "__main__":
    main()
