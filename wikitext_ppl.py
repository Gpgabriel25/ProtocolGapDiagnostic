#!/usr/bin/env python3
"""Shared WikiText-2 perplexity evaluation with overlap masking."""

import math
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import torch
from datasets import load_dataset

STANDARD_PROTOCOL = "wikitext2_concatenated_overlap_masking_v1"
DEFAULT_MAX_WORDS = 20000
DEFAULT_MAX_LENGTH = 1024
DEFAULT_STRIDE = 512
_DATASET_NAME = "wikitext"
_DATASET_CONFIG = "wikitext-2-raw-v1"


@lru_cache(maxsize=None)
def load_wikitext2_text(
    split: str = "test",
    max_words: Optional[int] = DEFAULT_MAX_WORDS,
) -> str:
    dataset = load_dataset(_DATASET_NAME, _DATASET_CONFIG, split=split)
    text = "\n\n".join(sample for sample in dataset["text"] if sample.strip())
    if max_words is not None:
        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[:max_words])
    return text


def build_wikitext2_eval_input(
    tokenizer,
    split: str = "test",
    max_words: Optional[int] = DEFAULT_MAX_WORDS,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    text = load_wikitext2_text(split=split, max_words=max_words)
    input_ids = tokenizer(text, return_tensors="pt", verbose=False).input_ids
    info: Dict[str, Any] = {
        "protocol": STANDARD_PROTOCOL,
        "dataset": f"{_DATASET_NAME}/{_DATASET_CONFIG}",
        "split": split,
        "max_words": max_words,
        "text_chars": len(text),
        "text_words": len(text.split()),
        "token_count": int(input_ids.size(1)),
    }
    return input_ids, info


def evaluate_perplexity_input_ids(
    model,
    input_ids: torch.Tensor,
    max_length: int = DEFAULT_MAX_LENGTH,
    stride: int = DEFAULT_STRIDE,
) -> Tuple[float, Dict[str, Any]]:
    if max_length <= 0 or stride <= 0:
        raise ValueError("max_length and stride must be positive")

    model.eval()
    device = next(model.parameters()).device
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)

    seq_len = input_ids.size(1)
    nlls = []
    n_tokens = 0
    prev_end = 0

    with torch.no_grad():
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            target_len = end - prev_end

            chunk = input_ids[:, begin:end]
            target_chunk = chunk.clone()
            if target_len < chunk.size(1):
                target_chunk[:, :-target_len] = -100

            outputs = model(chunk, labels=target_chunk)
            nlls.append(outputs.loss.item() * target_len)
            n_tokens += target_len
            prev_end = end

            if end == seq_len:
                break

    perplexity = math.exp(sum(nlls) / n_tokens)
    info: Dict[str, Any] = {
        "max_length": max_length,
        "stride": stride,
        "eval_tokens": n_tokens,
    }
    return perplexity, info