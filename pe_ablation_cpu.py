#!/usr/bin/env python3
"""
Controlled PE Ablation: Absolute Positional Encoding vs. RoPE
=============================================================
Trains two architecturally IDENTICAL 6-layer transformers on the same data
with only the positional encoding different:
  - Model A: learned absolute positional embedding (GPT-2 style)
  - Model B: rotary positional embedding (RoPE) — no absolute PE tokens

Both models share the same architecture, parameter count, optimizer, and
training data stream. After training, we compute interchange and replacement
bisimulation distances for all 15 pairwise layer combinations.

Key claim to test:
  PE type, not architecture or training data, causes the protocol gap
  (i.e., the systematic difference between interchange and replacement distance).

Success criterion:
  - RoPE model: mean interchange/replacement ratio < 0.5
    (interchange distances reliably smaller than replacement)
  - AbsPE model: mean interchange/replacement ratio > 0.7
    (no systematic gap between interchange and replacement)
"""

import os, sys, math, json, time, random, logging, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────

OUTPUT_DIR = "."
DEVICE = "cpu"

@dataclass
class ModelConfig:
    vocab_size: int = 50257           # GPT-2 tokenizer
    n_layers: int = 6
    n_heads: int = 4
    d_model: int = 256
    d_ff: int = 1024
    max_seq_len: int = 128
    dropout: float = 0.1
    use_rope: bool = False            # Key toggle: AbsPE vs RoPE

@dataclass
class TrainConfig:
    total_steps: int = 3_000         # Reduced for CPU
    batch_size: int = 4
    seq_len: int = 128
    lr: float = 3e-4
    warmup_steps: int = 500
    grad_clip: float = 1.0
    eval_steps: int = 1000
    save_steps: int = 5_000
    log_steps: int = 250

# ─────────────────────────────────────────────────
#  RoPE implementation
# ─────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._cos_cache = None
        self._sin_cache = None
        self._cache_len = 0
        self.max_seq_len = max_seq_len

    def _build_cache(self, seq_len):
        if seq_len <= self._cache_len:
            return
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos_cache = emb.cos()[None, None, :, :]   # (1, 1, T, dim)
        self._sin_cache = emb.sin()[None, None, :, :]
        self._cache_len = seq_len

    def rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k):
        """Apply RoPE to query and key tensors. Shape: (B, H, T, d_head)"""
        seq_len = q.shape[2]
        self._build_cache(seq_len)
        cos = self._cos_cache[:, :, :seq_len, :q.shape[-1]].to(q.device)
        sin = self._sin_cache[:, :, :seq_len, :q.shape[-1]].to(q.device)
        q_rot = q * cos + self.rotate_half(q) * sin
        k_rot = k * cos + self.rotate_half(k) * sin
        return q_rot, k_rot


# ─────────────────────────────────────────────────
#  Model architecture
# ─────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.d_model = config.d_model
        self.use_rope = config.use_rope

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.drop_attn = nn.Dropout(config.dropout)
        self.drop_res = nn.Dropout(config.dropout)

        if self.use_rope:
            self.rope = RotaryEmbedding(self.d_head)

        # Causal mask
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, config.max_seq_len, config.max_seq_len))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each: (B, T, H, d_head)
        q = q.transpose(1, 2)  # (B, H, T, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_rope:
            q, k = self.rope(q, k)

        scale = 1.0 / math.sqrt(self.d_head)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.drop_attn(attn)

        out = torch.matmul(attn, v)  # (B, H, T, d_head)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.drop_res(self.out(out))


class FFN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FFN(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        # Absolute PE: only used when NOT using RoPE
        if not config.use_rope:
            self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=0.02)

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)
        if not self.config.use_rope:
            pos = torch.arange(T, device=input_ids.device)
            x = x + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)  # (B, T, vocab)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────
#  Data loading: WikiText-103
# ─────────────────────────────────────────────────

def load_data_tokens(seq_len, max_tokens=5_000_000):
    """Load WikiText-2 and tokenize into a flat token array."""
    try:
        from datasets import load_dataset
        from transformers import GPT2TokenizerFast
    except ImportError:
        log.error("datasets/transformers not installed")
        sys.exit(1)

    log.info("Loading GPT-2 tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    log.info("Downloading WikiText-2...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    log.info("Tokenizing...")
    all_tokens = []
    for item in ds:
        text = item["text"].strip()
        if not text:
            continue
        toks = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(toks)
        all_tokens.append(tokenizer.eos_token_id)
        if len(all_tokens) >= max_tokens:
            break

    log.info(f"Total tokens: {len(all_tokens):,}")
    tokens = torch.tensor(all_tokens, dtype=torch.long)
    return tokens, tokenizer


class TokenDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len
        self.n = len(tokens) // (seq_len + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        start = i * self.seq_len
        chunk = self.tokens[start: start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


# ─────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────

def cosine_schedule(step, total_steps, warmup_steps, lr_min=1e-5, lr_max=3e-4):
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def train(model, tokens_arr, config: ModelConfig, tcfg: TrainConfig, label: str):
    model.to(DEVICE)
    log.info(f"[{label}] Parameters: {model.count_params():,}")

    dataset = TokenDataset(tokens_arr, tcfg.seq_len)
    loader = DataLoader(dataset, batch_size=tcfg.batch_size, shuffle=True,
                        num_workers=0, pin_memory=False, drop_last=True)
    data_iter = iter(loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr,
                                  betas=(0.9, 0.95), weight_decay=0.1)
    use_amp = (DEVICE == "cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)

    losses = []
    t0 = time.time()
    step = 0

    while step < tcfg.total_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)

        x, y = x.to(DEVICE), y.to(DEVICE)
        lr = cosine_schedule(step, tcfg.total_steps, tcfg.warmup_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        model.train()
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=DEVICE, enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        step += 1

        if step % tcfg.log_steps == 0:
            avg_loss = sum(losses[-tcfg.log_steps:]) / tcfg.log_steps
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t0
            log.info(f"[{label}] step={step}/{tcfg.total_steps} loss={avg_loss:.4f} ppl={ppl:.1f} lr={lr:.2e} elapsed={elapsed:.0f}s")

    model.eval()
    final_loss = sum(losses[-500:]) / min(500, len(losses))
    log.info(f"[{label}] Training complete. Final loss={final_loss:.4f}")
    return model


# ─────────────────────────────────────────────────
#  Bisimulation distance computation
# ─────────────────────────────────────────────────

def get_logits_with_layer_swap(model, input_ids, swap_i, swap_j, mode="interchange"):
    """
    Get output logits with layers i and j swapped (interchange)
    or with layer i replaced by layer j (replacement).
    
    mode: "interchange" — swap layers i↔j
          "replacement" — replace layer i with layer j's weights
    """
    blocks = model.blocks

    if mode == "interchange":
        orig_i = blocks[swap_i]
        orig_j = blocks[swap_j]
        blocks[swap_i] = orig_j
        blocks[swap_j] = orig_i
    elif mode == "replacement":
        orig_i = blocks[swap_i]
        blocks[swap_i] = blocks[swap_j]  # replace i with j
    else:
        raise ValueError(f"Unknown mode: {mode}")

    with torch.no_grad():
        logits = model(input_ids)

    # Restore
    if mode == "interchange":
        blocks[swap_i] = orig_i
        blocks[swap_j] = orig_j
    elif mode == "replacement":
        blocks[swap_i] = orig_i

    return logits


def compute_kl_divergence(logits_orig, logits_mod):
    """KL(p_orig || p_mod) per sequence position, then mean over positions and batch."""
    log_p = F.log_softmax(logits_orig.float(), dim=-1)
    log_q = F.log_softmax(logits_mod.float(), dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(-1)  # (B, T)
    return kl.mean().item()


def compute_distances(model, prompts, n_layers=6, n_prompts=300):
    """Compute interchange and replacement distances for all layer pairs."""
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    model.to(DEVICE)

    # Sample prompts
    random.shuffle(prompts)
    use_prompts = prompts[:n_prompts]

    # Tokenize in batches
    batch_size = 16
    token_batches = []
    for i in range(0, len(use_prompts), batch_size):
        batch = use_prompts[i: i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True,
                         max_length=model.config.max_seq_len, padding=True)
        token_batches.append(enc.input_ids.to(DEVICE))

    # Compute base logits for each batch
    with torch.no_grad():
        base_logits_list = [model(ids) for ids in token_batches]

    results = {}
    pair_count = 0
    total_pairs = n_layers * (n_layers - 1) // 2

    for i in range(n_layers):
        for j in range(i + 1, n_layers):
            kl_interchange_fwd = []
            kl_interchange_bwd = []
            kl_replace_ij = []
            kl_replace_ji = []

            for b_idx, ids in enumerate(token_batches):
                base_lgt = base_logits_list[b_idx]

                # Interchange i↔j
                swap_lgt = get_logits_with_layer_swap(model, ids, i, j, mode="interchange")
                kl_interchange_fwd.append(compute_kl_divergence(base_lgt, swap_lgt))
                # Symmetric (same result for interchange, but compute both directions)
                kl_interchange_bwd.append(compute_kl_divergence(swap_lgt, base_lgt))

                # Replace i→j
                rep_ij_lgt = get_logits_with_layer_swap(model, ids, i, j, mode="replacement")
                kl_replace_ij.append(compute_kl_divergence(base_lgt, rep_ij_lgt))

                # Replace j→i
                rep_ji_lgt = get_logits_with_layer_swap(model, ids, j, i, mode="replacement")
                kl_replace_ji.append(compute_kl_divergence(base_lgt, rep_ji_lgt))

            mean_interchange = (np.mean(kl_interchange_fwd) + np.mean(kl_interchange_bwd)) / 2
            mean_replacement = (np.mean(kl_replace_ij) + np.mean(kl_replace_ji)) / 2
            ratio = mean_interchange / (mean_replacement + 1e-10)

            results[f"{i},{j}"] = {
                "interchange": float(mean_interchange),
                "replacement": float(mean_replacement),
                "ratio": float(ratio),
                "strongly_bisimilar": bool(mean_interchange < 0.05),
            }
            pair_count += 1
            if pair_count % 5 == 0:
                log.info(f"  Pairs computed: {pair_count}/{total_pairs}")

    return results


# ─────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────

def load_eval_prompts(n=300):
    """Load diverse prompts for distance evaluation."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    prompts = []
    for item in ds:
        text = item["text"].strip()
        if len(text) > 100:
            prompts.append(text[:512])
        if len(prompts) >= n:
            break
    return prompts


def parse_args():
    parser = argparse.ArgumentParser(description="Run controlled PE ablation on CPU")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Directory for checkpoints and result JSON")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed")
    parser.add_argument("--max-train-tokens", type=int, default=2_000_000,
                        help="Max tokens to load from WikiText-2 train split")
    parser.add_argument("--eval-prompts", type=int, default=200,
                        help="Number of validation prompts to sample")
    parser.add_argument("--distance-prompts", type=int, default=200,
                        help="Number of prompts used for pairwise distance computation")
    parser.add_argument("--total-steps", type=int, default=3_000,
                        help="Training steps per model")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Training/eval context length")
    parser.add_argument("--log-steps", type=int, default=250,
                        help="Logging interval in training steps")
    parser.add_argument("--no-save-models", action="store_true",
                        help="Skip writing model checkpoints to disk")
    return parser.parse_args()


def main():
    args = parse_args()
    t_start = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    log.info("=" * 60)
    log.info("PE Ablation Experiment: AbsPE vs RoPE Protocol Gap")
    log.info("=" * 60)
    log.info(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load training data
    log.info("Loading training data...")
    tokens, tokenizer = load_data_tokens(seq_len=args.seq_len, max_tokens=args.max_train_tokens)
    log.info(f"Training tokens available: {len(tokens):,}")

    # Load eval prompts
    log.info("Loading evaluation prompts...")
    eval_prompts = load_eval_prompts(n=args.eval_prompts)
    log.info(f"Eval prompts: {len(eval_prompts)}")

    # Training config — reduced for CPU
    warmup_steps = min(500, max(10, args.total_steps // 6))
    tcfg = TrainConfig(
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        warmup_steps=warmup_steps,
        log_steps=args.log_steps,
    )

    all_results = {}

    for model_type, use_rope in [("abspe", False), ("rope", True)]:
        log.info(f"\n{'='*60}")
        log.info(f"Training model: {model_type.upper()} ({'RoPE' if use_rope else 'Absolute PE'})")
        log.info(f"{'='*60}")

        config = ModelConfig(use_rope=use_rope)
        model = MiniGPT(config)

        # Same seed for both models (different from data seed to avoid artifacts)
        torch.manual_seed(1337)
        model._init_weights()

        # Train
        model = train(model, tokens, config, tcfg, label=model_type.upper())

        # Save model
        if not args.no_save_models:
            save_path = os.path.join(output_dir, f"mini_gpt_{model_type}.pt")
            torch.save(model.state_dict(), save_path)
            log.info(f"Saved model to {save_path}")

        # Compute distances
        log.info(f"Computing pairwise distances for {model_type.upper()}...")
        distances = compute_distances(
            model,
            eval_prompts,
            n_layers=config.n_layers,
            n_prompts=args.distance_prompts,
        )

        # Analyze
        all_interchange = [d["interchange"] for d in distances.values()]
        all_replacement = [d["replacement"] for d in distances.values()]
        all_ratio = [d["ratio"] for d in distances.values()]

        log.info(f"\n[{model_type.upper()}] Results:")
        log.info(f"  Mean interchange distance: {np.mean(all_interchange):.4f}")
        log.info(f"  Mean replacement distance:  {np.mean(all_replacement):.4f}")
        log.info(f"  Mean I/R ratio:             {np.mean(all_ratio):.4f}")
        log.info(f"  Pairs with ratio < 0.5:     {sum(1 for r in all_ratio if r < 0.5)}/{len(all_ratio)}")
        log.info(f"  Pairs with ratio > 0.8:     {sum(1 for r in all_ratio if r > 0.8)}/{len(all_ratio)}")

        all_results[model_type] = {
            "config": {"use_rope": use_rope, "n_layers": config.n_layers,
                        "d_model": config.d_model, "n_heads": config.n_heads},
            "summary": {
                "mean_interchange": float(np.mean(all_interchange)),
                "mean_replacement": float(np.mean(all_replacement)),
                "mean_I_R_ratio": float(np.mean(all_ratio)),
                "pairs_ratio_lt_0.5": int(sum(1 for r in all_ratio if r < 0.5)),
                "pairs_ratio_gt_0.8": int(sum(1 for r in all_ratio if r > 0.8)),
                "n_pairs": len(distances),
            },
            "pair_distances": distances,
        }

        # Cleanup GPU memory before next model
        del model
        torch.cuda.empty_cache() if DEVICE == "cuda" else None

    # ─ Summary ──────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("PE ABLATION SUMMARY")
    log.info("=" * 60)

    abspe = all_results["abspe"]["summary"]
    rope = all_results["rope"]["summary"]

    log.info(f"\n{'Metric':<35} {'AbsPE':>10} {'RoPE':>10}")
    log.info("-" * 57)
    log.info(f"{'Mean interchange distance':<35} {abspe['mean_interchange']:>10.4f} {rope['mean_interchange']:>10.4f}")
    log.info(f"{'Mean replacement distance':<35} {abspe['mean_replacement']:>10.4f} {rope['mean_replacement']:>10.4f}")
    log.info(f"{'Mean I/R ratio':<35} {abspe['mean_I_R_ratio']:>10.4f} {rope['mean_I_R_ratio']:>10.4f}")
    log.info(f"{'Pairs with I/R ratio < 0.5':<35} {abspe['pairs_ratio_lt_0.5']:>10}/{abspe['n_pairs']} "
             f"{rope['pairs_ratio_lt_0.5']:>10}/{rope['n_pairs']}")
    log.info(f"{'Pairs with I/R ratio > 0.8':<35} {abspe['pairs_ratio_gt_0.8']:>10}/{abspe['n_pairs']} "
             f"{rope['pairs_ratio_gt_0.8']:>10}/{rope['n_pairs']}")

    abspe_ratio = abspe["mean_I_R_ratio"]
    rope_ratio = rope["mean_I_R_ratio"]
    gap_confirmed = bool(rope_ratio < 0.5 and abspe_ratio > 0.7)
    log.info(f"\nProtocol gap confirmed by ablation: {gap_confirmed}")
    log.info(f"  RoPE I/R ratio: {rope_ratio:.3f} (target < 0.5)")
    log.info(f"  AbsPE I/R ratio: {abspe_ratio:.3f} (target > 0.7)")

    # Full per-pair results
    log.info("\nPer-pair results (AbsPE):")
    for pair_key, d in sorted(all_results["abspe"]["pair_distances"].items(),
                               key=lambda x: x[1]["interchange"]):
        log.info(f"  Layers {pair_key}: interchange={d['interchange']:.4f} "
                 f"replacement={d['replacement']:.4f} ratio={d['ratio']:.3f}")

    log.info("\nPer-pair results (RoPE):")
    for pair_key, d in sorted(all_results["rope"]["pair_distances"].items(),
                               key=lambda x: x[1]["interchange"]):
        log.info(f"  Layers {pair_key}: interchange={d['interchange']:.4f} "
                 f"replacement={d['replacement']:.4f} ratio={d['ratio']:.3f}")

    # Save results
    results_path = os.path.join(output_dir, "pe_ablation_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "experiment": "pe_ablation",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "training": {
                "total_steps": tcfg.total_steps,
                "batch_size": tcfg.batch_size,
                "seq_len": tcfg.seq_len,
                "max_train_tokens": args.max_train_tokens,
                "eval_prompts": args.eval_prompts,
                "distance_prompts": args.distance_prompts,
                "dataset": "wikitext-2-raw-v1",
                "device": DEVICE,
            },
            "results": all_results,
            "conclusion": {
                "gap_confirmed": gap_confirmed,
                "abspe_I_R_ratio": abspe_ratio,
                "rope_I_R_ratio": rope_ratio,
            }
        }, f, indent=2)

    total_time = time.time() - t_start
    log.info(f"\nTotal experiment time: {total_time/60:.1f} minutes")
    log.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
