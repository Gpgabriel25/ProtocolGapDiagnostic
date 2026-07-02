#!/usr/bin/env python3
"""Generate BLOOM vs GPT-2 adjacent pair comparison figure.
Run after bloom_bisimulation.py completes."""

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "stix",
})
import matplotlib.pyplot as plt
import numpy as np
import csv
import json
import os

def load_gpt2_adjacent():
    """Load GPT-2-Medium 100-prompt adjacent pairs."""
    pairs = []
    with open('reports/2026-04-02T09-57-25/v14_distance_matrix_100p.csv') as f:
        for row in csv.DictReader(f):
            if int(row['gap']) == 1:
                pairs.append((int(row['layer_a']), float(row['mean_kl'])))
    pairs.sort()
    return [p[0] for p in pairs], [p[1] for p in pairs]

def load_bloom_adjacent():
    """Load BLOOM-560M adjacent pairs from checkpoint or results."""
    ckpt_path = 'reports/2026-04-02T09-57-25/bloom_checkpoint.json'
    if os.path.exists(ckpt_path):
        data = json.load(open(ckpt_path))
        pairs = [(p['layer_a'], p['mean_kl']) for p in data['pairs']]
        pairs.sort()
        return [p[0] for p in pairs], [p[1] for p in pairs]
    return [], []

def main():
    gpt2_x, gpt2_y = load_gpt2_adjacent()
    bloom_x, bloom_y = load_bloom_adjacent()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    ax.plot(gpt2_x, gpt2_y, 'b-o', linewidth=2, markersize=6, label='GPT-2-Medium (Abs PE, 355M)', alpha=0.8)
    if bloom_x:
        ax.plot(bloom_x, bloom_y, 'r-s', linewidth=2, markersize=6, label='BLOOM-560M (ALiBi, 560M)', alpha=0.8)
    
    ax.axhline(y=0.05, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Strong threshold (0.05)')
    ax.axhline(y=0.10, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Conditional threshold (0.10)')
    
    ax.set_xlabel('Layer i (adjacent pair i↔i+1)', fontsize=12)
    ax.set_ylabel('Mean KL Divergence', fontsize=12)
    ax.set_title('Adjacent Pair Bisimulation: GPT-2 vs BLOOM (100 prompts)', fontsize=13)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_yscale('log')
    ax.set_ylim(0.01, 5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = 'paper/figures/bloom_comparison.pdf'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close()

if __name__ == '__main__':
    main()
