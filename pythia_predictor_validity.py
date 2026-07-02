#!/usr/bin/env python3
"""Quick predictor-validity check on a RoPE model (Pythia-1.4B).

Computes single-layer removal costs and correlates them with adjacent-pair
interchange/replacement distances from an existing score file.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from scipy import stats
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_pairs(path: str):
    with open(path) as f:
        data = json.load(f)
    return data["pairs"]


def build_eval_texts(n_samples: int):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"].strip() for x in ds if len(x["text"].strip()) > 50]
    rng = np.random.default_rng(42)
    rng.shuffle(texts)
    return texts[:n_samples]


def eval_ppl(model, tokenizer, texts, max_length: int):
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc.input_ids.to(model.device)
            if input_ids.shape[1] < 4:
                continue
            out = model(input_ids, labels=input_ids)
            loss = out.loss
            if not torch.isfinite(loss):
                continue
            n = int(input_ids.shape[1])
            total_nll += float(loss.item()) * n
            total_tokens += n
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)


def layer_remove_costs(model, tokenizer, texts, max_length: int):
    layers = model.gpt_neox.layers
    baseline = eval_ppl(model, tokenizer, texts, max_length)
    costs = {}

    for i in range(len(layers)):
        input_ref = [None]

        def pre_hook(module, args):
            input_ref[0] = args[0]

        def post_hook(module, args, output):
            ih = input_ref[0]
            if isinstance(output, tuple):
                return (ih,) + output[1:]
            return ih

        h_pre = layers[i].register_forward_pre_hook(pre_hook)
        h_post = layers[i].register_forward_hook(post_hook)

        ppl_i = eval_ppl(model, tokenizer, texts, max_length)
        h_pre.remove()
        h_post.remove()

        costs[i] = {
            "ppl": ppl_i,
            "delta_pct": (ppl_i - baseline) / baseline * 100.0,
        }

    return baseline, costs


def correlate(pairs, costs, metric_key):
    xs, ys = [], []
    for p in pairs:
        i, j = int(p["layer_a"]), int(p["layer_b"])
        if i not in costs or j not in costs:
            continue
        x = float(p[metric_key])
        y = min(float(costs[i]["delta_pct"]), float(costs[j]["delta_pct"]))
        xs.append(x)
        ys.append(y)

    rho, pval = stats.spearmanr(xs, ys)
    return {
        "n_pairs": len(xs),
        "spearman_rho": float(rho),
        "spearman_pvalue": float(pval),
    }


def _flatten_hidden(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().reshape(-1, x.shape[-1])


def _linear_cka(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    hsic_ab = torch.norm(a.T @ b, p="fro") ** 2
    hsic_aa = torch.norm(a.T @ a, p="fro") ** 2
    hsic_bb = torch.norm(b.T @ b, p="fro") ** 2
    denom = torch.sqrt(hsic_aa * hsic_bb) + 1e-12
    return float((hsic_ab / denom).item())


def compute_bi_and_cka_scores(model, tokenizer, texts, max_length: int):
    n_layers = len(model.gpt_neox.layers)
    bi_sum = np.zeros(n_layers, dtype=np.float64)
    cka_sum = np.zeros(n_layers, dtype=np.float64)
    count = 0

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc.input_ids.to(model.device)
            if input_ids.shape[1] < 4:
                continue

            out = model(input_ids, output_hidden_states=True)
            hs = out.hidden_states
            if hs is None or len(hs) < n_layers + 1:
                continue

            for i in range(n_layers):
                h_in = _flatten_hidden(hs[i])
                h_out = _flatten_hidden(hs[i + 1])

                cos = torch.nn.functional.cosine_similarity(h_in, h_out, dim=-1).mean().item()
                bi_sum[i] += 1.0 - float(cos)

                cka = _linear_cka(h_in, h_out)
                cka_sum[i] += 1.0 - cka

            count += 1

    if count == 0:
        raise RuntimeError("No valid samples for BI/CKA computation")

    bi_scores = bi_sum / count
    cka_scores = cka_sum / count
    return bi_scores, cka_scores


def per_layer_swap_scores(pairs, n_layers):
    inter_by_layer = [[] for _ in range(n_layers)]
    repl_by_layer = [[] for _ in range(n_layers)]
    for p in pairs:
        i, j = int(p["layer_a"]), int(p["layer_b"])
        if i >= n_layers or j >= n_layers:
            continue
        inter = float(p["d_interchange"])
        repl = float(p["d_bisim"])
        inter_by_layer[i].append(inter)
        inter_by_layer[j].append(inter)
        repl_by_layer[i].append(repl)
        repl_by_layer[j].append(repl)

    inter_layer = np.array([
        min(vals) if len(vals) > 0 else np.nan for vals in inter_by_layer
    ], dtype=np.float64)
    repl_layer = np.array([
        min(vals) if len(vals) > 0 else np.nan for vals in repl_by_layer
    ], dtype=np.float64)
    return inter_layer, repl_layer


def spearman_layer(scores: np.ndarray, costs: dict):
    xs, ys = [], []
    for i, s in enumerate(scores):
        if not np.isfinite(s):
            continue
        if i not in costs:
            continue
        xs.append(float(s))
        ys.append(float(costs[i]["delta_pct"]))
    rho, pval = stats.spearmanr(xs, ys)
    return {"n_layers": len(xs), "spearman_rho": float(rho), "spearman_pvalue": float(pval)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument(
        "--pairs-json",
        default="/home/gpgabriel25/Projects/BisimulationQuotient/reports/2026-04-02T16-23-40/pythia-1.4b_interchange.json",
    )
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument(
        "--out",
        default="/home/gpgabriel25/Projects/BisimulationQuotient/reports/2026-04-05T21-27-05/pythia_predictor_validity.json",
    )
    args = ap.parse_args()

    t0 = time.time()
    # Avoid CUDA probing stalls on CPU-only environments.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    print("Loading model/tokenizer:", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="eager")
    model.eval()

    texts = build_eval_texts(args.n_samples)
    print(f"Using {len(texts)} eval samples, max_length={args.max_length}")

    baseline, costs = layer_remove_costs(model, tokenizer, texts, args.max_length)
    pairs = load_pairs(args.pairs_json)

    corr_inter = correlate(pairs, costs, "d_interchange")
    corr_repl = correlate(pairs, costs, "d_bisim")

    n_layers = len(model.gpt_neox.layers)
    inter_layer_scores, repl_layer_scores = per_layer_swap_scores(pairs, n_layers)
    bi_scores, cka_scores = compute_bi_and_cka_scores(model, tokenizer, texts, args.max_length)

    layer_corr = {
        "interchange_layer_minpair": spearman_layer(inter_layer_scores, costs),
        "replacement_layer_minpair": spearman_layer(repl_layer_scores, costs),
        "bi_layer": spearman_layer(bi_scores, costs),
        "cka_layer": spearman_layer(cka_scores, costs),
    }

    result = {
        "model": args.model,
        "n_samples": args.n_samples,
        "max_length": args.max_length,
        "baseline_ppl": baseline,
        "correlation_interchange": corr_inter,
        "correlation_replacement": corr_repl,
        "layer_ranking_correlations": layer_corr,
        "bi_scores": {str(i): float(bi_scores[i]) for i in range(len(bi_scores))},
        "cka_scores": {str(i): float(cka_scores[i]) for i in range(len(cka_scores))},
        "interchange_layer_scores": {str(i): float(inter_layer_scores[i]) for i in range(len(inter_layer_scores)) if np.isfinite(inter_layer_scores[i])},
        "replacement_layer_scores": {str(i): float(repl_layer_scores[i]) for i in range(len(repl_layer_scores)) if np.isfinite(repl_layer_scores[i])},
        "elapsed_s": time.time() - t0,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print("Interchange:", corr_inter)
    print("Replacement:", corr_repl)
    print("Layer ranking correlations:", layer_corr)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
