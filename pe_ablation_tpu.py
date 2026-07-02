#!/usr/bin/env python3
"""
Controlled PE Ablation: Absolute Positional Encoding vs. RoPE
=============================================================
TPU-adapted version of pe_ablation_kaggle_p100.py.

Trains two architecturally IDENTICAL transformers on the same data
with only the positional encoding different:
  - Model A: learned absolute positional embedding (GPT-2 style)
  - Model B: rotary positional embedding (RoPE)

After training, computes interchange and replacement bisimulation distances
for layer pairs with gap <= max_gap, plus Jacobian norm estimates.

Requires: torch, torch_xla, transformers, datasets
"""

import math, json, time, random, logging, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
from typing import Optional

# ── XLA imports ──
import torch_xla
import torch_xla.core.xla_model as xm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def _env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _env_int_tuple(name, default):
    value = os.environ.get(name)
    if not value:
        return default
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


# ─────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────

OUTPUT_DIR = os.environ.get("PE_OUTPUT_DIR", "/tmp/pe_ablation_output")
DEVICE = xm.xla_device()

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    n_layers: int = _env_int("PE_N_LAYERS", 12)
    n_heads: int = _env_int("PE_N_HEADS", 8)
    d_model: int = _env_int("PE_D_MODEL", 512)
    d_ff: int = _env_int("PE_D_FF", 2048)
    max_seq_len: int = _env_int("PE_MAX_SEQ_LEN", 256)
    dropout: float = float(os.environ.get("PE_DROPOUT", "0.1"))
    use_rope: bool = False

@dataclass
class TrainConfig:
    total_steps: int = _env_int("PE_TOTAL_STEPS", 25_000)
    batch_size: int = _env_int("PE_BATCH_SIZE", 16)
    seq_len: int = _env_int("PE_SEQ_LEN", 256)
    lr: float = float(os.environ.get("PE_LR", "3e-4"))
    warmup_steps: int = _env_int("PE_WARMUP_STEPS", 1000)
    grad_clip: float = float(os.environ.get("PE_GRAD_CLIP", "1.0"))
    eval_steps: int = _env_int("PE_EVAL_STEPS", 2000)
    save_steps: int = _env_int("PE_SAVE_STEPS", 10_000)
    log_steps: int = _env_int("PE_LOG_STEPS", 500)
    distance_checkpoints: tuple = _env_int_tuple("PE_DISTANCE_CHECKPOINTS", (5000, 15000, 25000))


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

    def _build_cache(self, seq_len, device):
        if seq_len <= self._cache_len and self._cos_cache is not None:
            return
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos_cache = emb.cos()[None, None, :, :]
        self._sin_cache = emb.sin()[None, None, :, :]
        self._cache_len = seq_len

    def rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k):
        seq_len = q.shape[2]
        self._build_cache(seq_len, q.device)
        cos = self._cos_cache[:, :, :seq_len, :q.shape[-1]]
        sin = self._sin_cache[:, :, :seq_len, :q.shape[-1]]
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

        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, config.max_seq_len, config.max_seq_len))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_rope:
            q, k = self.rope(q, k)

        scale = 1.0 / math.sqrt(self.d_head)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.drop_attn(attn)

        out = torch.matmul(attn, v)
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
        if not config.use_rope:
            self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
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
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────
#  Data loading: WikiText-103
# ─────────────────────────────────────────────────

def load_data_tokens(seq_len, max_tokens=100_000_000):
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    log.info("Loading GPT-2 tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    log.info("Downloading WikiText-103...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

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
#  Training loop (TPU-adapted)
# ─────────────────────────────────────────────────

def cosine_schedule(step, total_steps, warmup_steps, lr_min=1e-5, lr_max=3e-4):
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def train(model, tokens_arr, config: ModelConfig, tcfg: TrainConfig, label: str,
          eval_prompts=None, distance_checkpoints=None):
    model.to(DEVICE)
    log.info(f"[{label}] Parameters: {model.count_params():,}")

    dataset = TokenDataset(tokens_arr, tcfg.seq_len)
    loader = DataLoader(dataset, batch_size=tcfg.batch_size, shuffle=True,
                        num_workers=2, drop_last=True)
    data_iter = iter(loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr,
                                  betas=(0.9, 0.95), weight_decay=0.1)

    losses = []
    checkpoint_distances = {}
    checkpoint_set = set(distance_checkpoints or [])
    checkpoint_prompt_count = _env_int("PE_CHECKPOINT_PROMPTS", 100)
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

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()
        xm.mark_step()

        losses.append(loss.item())
        step += 1

        if step % tcfg.log_steps == 0:
            avg_loss = sum(losses[-tcfg.log_steps:]) / tcfg.log_steps
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t0
            log.info(f"[{label}] step={step}/{tcfg.total_steps} loss={avg_loss:.4f} ppl={ppl:.1f} lr={lr:.2e} elapsed={elapsed:.0f}s")

        if step in checkpoint_set and eval_prompts is not None:
            avg_loss_now = sum(losses[-min(500, len(losses)):]) / min(500, len(losses))
            ppl_now = math.exp(min(avg_loss_now, 20))
            log.info(f"[{label}] CHECKPOINT step={step}: computing distances (loss={avg_loss_now:.4f}, ppl={ppl_now:.1f})...")
            dists = compute_distances(model, eval_prompts, n_layers=config.n_layers, n_prompts=checkpoint_prompt_count)
            all_ratio = [d["ratio"] for d in dists.values()]
            mean_ratio = float(np.mean(all_ratio))
            log.info(f"[{label}] CHECKPOINT step={step}: mean I/R ratio = {mean_ratio:.4f}")
            checkpoint_distances[step] = {
                "step": step, "loss": float(avg_loss_now), "ppl": float(ppl_now),
                "mean_I_R_ratio": mean_ratio, "pair_distances": dists,
            }
            model.train()

    model.eval()
    final_loss = sum(losses[-500:]) / min(500, len(losses))
    log.info(f"[{label}] Training complete. Final loss={final_loss:.4f}")
    return model, checkpoint_distances


# ─────────────────────────────────────────────────
#  Bisimulation distance computation
# ─────────────────────────────────────────────────

def get_logits_with_layer_swap(model, input_ids, swap_i, swap_j, mode="interchange"):
    blocks = model.blocks

    if mode == "interchange":
        orig_i = blocks[swap_i]
        orig_j = blocks[swap_j]
        blocks[swap_i] = orig_j
        blocks[swap_j] = orig_i
    elif mode == "replacement":
        orig_i = blocks[swap_i]
        blocks[swap_i] = blocks[swap_j]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    with torch.no_grad():
        logits = model(input_ids)
    xm.mark_step()

    if mode == "interchange":
        blocks[swap_i] = orig_i
        blocks[swap_j] = orig_j
    elif mode == "replacement":
        blocks[swap_i] = orig_i

    return logits


def compute_kl_divergence(logits_orig, logits_mod):
    log_p = F.log_softmax(logits_orig.float(), dim=-1)
    log_q = F.log_softmax(logits_mod.float(), dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(-1)
    return kl.mean().item()


def compute_distances(model, prompts, n_layers=12, n_prompts=300, max_gap=4):
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    random.shuffle(prompts)
    use_prompts = prompts[:n_prompts]

    batch_size = 16
    token_batches = []
    for i in range(0, len(use_prompts), batch_size):
        batch = use_prompts[i: i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True,
                         max_length=model.config.max_seq_len, padding=True)
        token_batches.append(enc.input_ids.to(DEVICE))

    with torch.no_grad():
        base_logits_list = []
        for ids in token_batches:
            base_logits_list.append(model(ids))
            xm.mark_step()

    results = {}
    pair_count = 0
    pairs = [(i, j) for i in range(n_layers) for j in range(i + 1, n_layers) if j - i <= max_gap]
    total_pairs = len(pairs)

    for i, j in pairs:
        kl_interchange_fwd = []
        kl_interchange_bwd = []
        kl_replace_ij = []
        kl_replace_ji = []

        for b_idx, ids in enumerate(token_batches):
            base_lgt = base_logits_list[b_idx]

            swap_lgt = get_logits_with_layer_swap(model, ids, i, j, mode="interchange")
            kl_interchange_fwd.append(compute_kl_divergence(base_lgt, swap_lgt))
            kl_interchange_bwd.append(compute_kl_divergence(swap_lgt, base_lgt))

            rep_ij_lgt = get_logits_with_layer_swap(model, ids, i, j, mode="replacement")
            kl_replace_ij.append(compute_kl_divergence(base_lgt, rep_ij_lgt))

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
#  Jacobian norms
# ─────────────────────────────────────────────────

def compute_jacobian_norms(model, prompts, n_layers=12, n_prompts=10, eps=1e-3):
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    use_prompts = prompts[:n_prompts]
    enc = tokenizer(use_prompts, return_tensors="pt", truncation=True,
                     max_length=model.config.max_seq_len, padding=True)
    input_ids = enc.input_ids.to(DEVICE)

    norms = {}
    for layer_idx in range(n_layers):
        with torch.no_grad():
            x = model.tok_emb(input_ids)
            if not model.config.use_rope:
                B, T = input_ids.shape
                pos = torch.arange(T, device=input_ids.device)
                x = x + model.pos_emb(pos)
            x = model.drop(x)
            for k in range(layer_idx):
                x = model.blocks[k](x)

            v = torch.randn_like(x)
            v = v / (v.norm() + 1e-10)

            for _ in range(10):
                x_plus = x + eps * v
                x_minus = x - eps * v
                f_plus = model.blocks[layer_idx](x_plus)
                f_minus = model.blocks[layer_idx](x_minus)
                Jv = (f_plus - f_minus) / (2 * eps)
                norm_Jv = Jv.norm()
                v = Jv / (norm_Jv + 1e-10)
            xm.mark_step()

            norms[layer_idx] = float(norm_Jv.item())

    return norms


# ─────────────────────────────────────────────────
#  Eval prompt loading
# ─────────────────────────────────────────────────

def load_eval_prompts(n=500):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    prompts = []
    for item in ds:
        text = item["text"].strip()
        if len(text) > 100:
            prompts.append(text[:512])
        if len(prompts) >= n:
            break
    return prompts


# ─────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────

def main():
    t_start = time.time()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    experiment_name = os.environ.get("PE_EXPERIMENT_NAME", "pe_ablation_tpu")
    train_max_tokens = _env_int("PE_TRAIN_MAX_TOKENS", 50_000_000)
    eval_prompt_count = _env_int("PE_EVAL_PROMPTS", 500)
    final_prompt_count = _env_int("PE_FINAL_PROMPTS", 100)
    jacobian_prompt_count = _env_int("PE_JACOBIAN_PROMPTS", 10)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("=" * 60)
    log.info(f"PE Ablation Experiment (TPU): {experiment_name}")
    log.info("=" * 60)
    log.info(f"Device: {DEVICE}")

    # Load training data
    log.info("Loading training data...")
    tokens, tokenizer = load_data_tokens(seq_len=256, max_tokens=train_max_tokens)
    log.info(f"Training tokens available: {len(tokens):,}")

    log.info("Loading evaluation prompts...")
    eval_prompts = load_eval_prompts(n=eval_prompt_count)
    log.info(f"Eval prompts: {len(eval_prompts)}")

    tcfg = TrainConfig()
    base_config = ModelConfig()
    log.info(
        "Model config: layers=%d heads=%d d_model=%d d_ff=%d max_seq_len=%d",
        base_config.n_layers, base_config.n_heads, base_config.d_model,
        base_config.d_ff, base_config.max_seq_len,
    )
    log.info(
        "Training config: total_steps=%d batch_size=%d seq_len=%d checkpoints=%s",
        tcfg.total_steps, tcfg.batch_size, tcfg.seq_len, list(tcfg.distance_checkpoints),
    )

    all_results = {}

    for model_type, use_rope in [("abspe", False), ("rope", True)]:
        log.info(f"\n{'='*60}")
        log.info(f"Training model: {model_type.upper()} ({'RoPE' if use_rope else 'Absolute PE'})")
        log.info(f"{'='*60}")

        mcfg = ModelConfig(use_rope=use_rope)
        model = MiniGPT(mcfg)

        torch.manual_seed(1337)
        model._init_weights()

        model, ckpt_dists = train(
            model, tokens, mcfg, tcfg, label=model_type.upper(),
            eval_prompts=eval_prompts,
            distance_checkpoints=tcfg.distance_checkpoints,
        )

        save_path = os.path.join(OUTPUT_DIR, f"mini_gpt_{model_type}.pt")
        xm.save(model.state_dict(), save_path)
        log.info(f"Saved model to {save_path}")

        log.info(f"Computing final pairwise distances for {model_type.upper()}...")
        distances = compute_distances(model, eval_prompts, n_layers=mcfg.n_layers,
                                       n_prompts=final_prompt_count, max_gap=4)

        log.info(f"Computing Jacobian norms for {model_type.upper()}...")
        jacobian_norms = compute_jacobian_norms(model, eval_prompts,
                                                  n_layers=mcfg.n_layers,
                                                  n_prompts=jacobian_prompt_count)
        log.info(f"[{model_type.upper()}] Jacobian norms: " +
                 ", ".join(f"L{k}={v:.3f}" for k, v in sorted(jacobian_norms.items())))

        all_interchange = [d["interchange"] for d in distances.values()]
        all_replacement = [d["replacement"] for d in distances.values()]
        all_ratio = [d["ratio"] for d in distances.values()]

        log.info(f"\n[{model_type.upper()}] Final Results:")
        log.info(f"  Mean interchange distance: {np.mean(all_interchange):.4f}")
        log.info(f"  Mean replacement distance:  {np.mean(all_replacement):.4f}")
        log.info(f"  Mean I/R ratio:             {np.mean(all_ratio):.4f}")
        log.info(f"  Pairs with ratio < 0.5:     {sum(1 for r in all_ratio if r < 0.5)}/{len(all_ratio)}")
        log.info(f"  Pairs with ratio > 0.8:     {sum(1 for r in all_ratio if r > 0.8)}/{len(all_ratio)}")

        emergence_trajectory = []
        for step, cd in sorted(ckpt_dists.items()):
            emergence_trajectory.append({
                "step": cd["step"], "loss": cd["loss"], "ppl": cd["ppl"],
                "mean_I_R_ratio": cd["mean_I_R_ratio"],
            })
            log.info(f"[{model_type.upper()}] Trajectory step={cd['step']}: "
                     f"ppl={cd['ppl']:.1f} I/R={cd['mean_I_R_ratio']:.4f}")

        all_results[model_type] = {
            "config": {"use_rope": use_rope, "n_layers": mcfg.n_layers,
                        "d_model": mcfg.d_model, "n_heads": mcfg.n_heads},
            "summary": {
                "mean_interchange": float(np.mean(all_interchange)),
                "mean_replacement": float(np.mean(all_replacement)),
                "mean_I_R_ratio": float(np.mean(all_ratio)),
                "pairs_ratio_lt_0.5": int(sum(1 for r in all_ratio if r < 0.5)),
                "pairs_ratio_gt_0.8": int(sum(1 for r in all_ratio if r > 0.8)),
                "n_pairs": len(distances),
            },
            "pair_distances": distances,
            "jacobian_norms": {str(k): v for k, v in jacobian_norms.items()},
            "emergence_trajectory": emergence_trajectory,
            "checkpoint_distances": {str(k): {kk: vv for kk, vv in v.items() if kk != "pair_distances"}
                                     for k, v in ckpt_dists.items()},
        }

        del model

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

    log.info("\nJacobian norms (per layer):")
    log.info(f"{'Layer':<8} {'AbsPE':>10} {'RoPE':>10}")
    log.info("-" * 30)
    for layer in range(base_config.n_layers):
        a_norm = float(all_results["abspe"]["jacobian_norms"].get(str(layer), 0))
        r_norm = float(all_results["rope"]["jacobian_norms"].get(str(layer), 0))
        log.info(f"  {layer:<6} {a_norm:>10.3f} {r_norm:>10.3f}")

    log.info("\nGap emergence over training:")
    log.info(f"{'Step':<10} {'AbsPE PPL':>12} {'AbsPE I/R':>12} {'RoPE PPL':>12} {'RoPE I/R':>12}")
    log.info("-" * 60)
    abspe_traj = all_results["abspe"]["emergence_trajectory"]
    rope_traj = all_results["rope"]["emergence_trajectory"]
    for at, rt in zip(abspe_traj, rope_traj):
        log.info(f"  {at['step']:<8} {at['ppl']:>12.1f} {at['mean_I_R_ratio']:>12.4f} "
                 f"{rt['ppl']:>12.1f} {rt['mean_I_R_ratio']:>12.4f}")

    abspe_ratio = abspe["mean_I_R_ratio"]
    rope_ratio = rope["mean_I_R_ratio"]
    gap_confirmed = bool(rope_ratio < 0.5 and abspe_ratio > 0.7)
    gap_trending = False
    if len(abspe_traj) >= 2 and len(rope_traj) >= 2:
        early_diff = abspe_traj[0]["mean_I_R_ratio"] - rope_traj[0]["mean_I_R_ratio"]
        late_diff = abspe_traj[-1]["mean_I_R_ratio"] - rope_traj[-1]["mean_I_R_ratio"]
        gap_trending = late_diff > early_diff + 0.05

    log.info(f"\nProtocol gap confirmed: {gap_confirmed}")
    log.info(f"  RoPE I/R ratio: {rope_ratio:.3f} (target < 0.5)")
    log.info(f"  AbsPE I/R ratio: {abspe_ratio:.3f} (target > 0.7)")
    log.info(f"Gap trending in expected direction: {gap_trending}")

    for mt in ["abspe", "rope"]:
        log.info(f"\nPer-pair results ({mt.upper()}):")
        for pair_key, d in sorted(all_results[mt]["pair_distances"].items(),
                                   key=lambda x: x[1]["interchange"]):
            log.info(f"  Layers {pair_key}: interchange={d['interchange']:.4f} "
                     f"replacement={d['replacement']:.4f} ratio={d['ratio']:.3f}")

    results_path = os.path.join(OUTPUT_DIR, "pe_ablation_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "experiment": experiment_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "training": {
                "total_steps": tcfg.total_steps,
                "batch_size": tcfg.batch_size,
                "seq_len": tcfg.seq_len,
                "n_layers": base_config.n_layers,
                "d_model": base_config.d_model,
                "dataset": "wikitext-103-raw-v1",
                "device": str(DEVICE),
                "train_max_tokens": train_max_tokens,
                "eval_prompts_loaded": eval_prompt_count,
                "final_prompts_used": final_prompt_count,
                "jacobian_prompts_used": jacobian_prompt_count,
                "distance_checkpoints": list(tcfg.distance_checkpoints),
                "output_dir": OUTPUT_DIR,
            },
            "results": all_results,
            "conclusion": {
                "gap_confirmed": gap_confirmed,
                "gap_trending": gap_trending,
                "abspe_I_R_ratio": abspe_ratio,
                "rope_I_R_ratio": rope_ratio,
            }
        }, f, indent=2)

    total_time = time.time() - t_start
    log.info(f"\nTotal experiment time: {total_time/60:.1f} minutes")
    log.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
