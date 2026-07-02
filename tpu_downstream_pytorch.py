#!/usr/bin/env python3
"""
PyTorch-based downstream evaluation for Qwen3-8B on TPU VM (CPU inference).
============================================================================
Uses HuggingFace's native model implementation to guarantee correctness.
The TPU VM has 96GB RAM — Qwen3-8B in BF16 (~16GB) fits on CPU easily.

Runs: LAMBADA, HellaSwag, ARC-Easy, WinoGrande
Configs: baseline + skip_n1/n2/n3/n5

Output: /tmp/downstream_pytorch_results.json
"""

import os, sys, json, time, logging
import numpy as np

os.environ["HF_HOME"] = "/tmp/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/tmp/downstream_pytorch.log"),
    ],
)
log = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-8B"
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "1000"))
OUTPUT_PATH = "/tmp/downstream_pytorch_results.json"

SKIP_CONFIGS = {
    "baseline": [],
    "skip_n1": [33],
    "skip_n2": [30, 33],
    "skip_n3": [26, 30, 33],
    "skip_n5": [17, 21, 26, 30, 33],
}


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(f"Loading {MODEL_NAME} in BF16 on CPU...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, cache_dir="/tmp/hf_cache"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        cache_dir="/tmp/hf_cache",
    )
    model.eval()

    n_layers = len(model.model.layers)
    log.info(f"Model loaded in {time.time()-t0:.1f}s. Layers: {n_layers}")
    return model, tokenizer


class SkipLayerContext:
    """Temporarily remove specified layer indices from the forward pass."""

    def __init__(self, model, skip_indices):
        self.model = model
        self.skip = set(skip_indices)
        self._saved = None

    def __enter__(self):
        import torch
        layers = self.model.model.layers
        self._saved = list(layers)
        kept = [l for i, l in enumerate(layers) if i not in self.skip]
        self.model.model.layers = torch.nn.ModuleList(kept)
        return self

    def __exit__(self, *_):
        import torch
        self.model.model.layers = torch.nn.ModuleList(self._saved)
        self._saved = None


def score_continuation(model, tokenizer, context, continuation, max_ctx=512):
    import torch
    full = context + continuation
    enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=max_ctx)
    ctx_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    n_ctx = len(ctx_ids)

    input_ids = enc.input_ids
    with torch.no_grad():
        logits = model(input_ids).logits
    log_probs = torch.log_softmax(logits[0].float(), dim=-1)

    cont_ids = enc.input_ids[0, n_ctx:].tolist()
    if not cont_ids:
        return -float("inf")
    score = sum(log_probs[n_ctx - 1 + k, tok].item() for k, tok in enumerate(cont_ids))
    return score


def eval_lambada(model, tokenizer, max_samples):
    import torch
    from datasets import load_dataset

    log.info("Evaluating LAMBADA...")
    ds = load_dataset("lambada", split="test", trust_remote_code=True)
    correct = 0
    n = 0
    for item in ds:
        if n >= max_samples:
            break
        text = item["text"].strip()
        if not text:
            continue
        parts = text.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        context, last_word = parts
        context = context + " "

        enc_ctx = tokenizer(context, return_tensors="pt", truncation=True, max_length=400)
        ctx_ids = enc_ctx.input_ids

        with torch.no_grad():
            lgt = model(ctx_ids).logits[0, -1, :]

        pred_id = lgt.argmax().item()
        pred_tok = tokenizer.decode([pred_id]).strip().lower()
        last_word_clean = last_word.strip().lower().rstrip(".,!?;:")

        if pred_tok == last_word_clean or last_word_clean.startswith(pred_tok):
            correct += 1
        n += 1
        if n % 100 == 0:
            log.info(f"  LAMBADA: {n}/{max_samples}, acc={correct/n:.3f}")

    acc = correct / n if n > 0 else 0.0
    log.info(f"  LAMBADA final: {correct}/{n} = {acc:.4f}")
    return acc, n


def eval_hellaswag(model, tokenizer, max_samples):
    from datasets import load_dataset

    log.info("Evaluating HellaSwag...")
    ds = load_dataset("hellaswag", split="validation", trust_remote_code=True)
    correct = 0
    n = 0
    for item in ds:
        if n >= max_samples:
            break
        ctx = item["ctx"]
        endings = item["endings"]
        label = int(item["label"])

        scores = [score_continuation(model, tokenizer, ctx + " ", e) for e in endings]
        pred = int(np.argmax(scores))
        if pred == label:
            correct += 1
        n += 1
        if n % 100 == 0:
            log.info(f"  HellaSwag: {n}/{max_samples}, acc={correct/n:.3f}")

    acc = correct / n if n > 0 else 0.0
    log.info(f"  HellaSwag final: {correct}/{n} = {acc:.4f}")
    return acc, n


def eval_arc_easy(model, tokenizer, max_samples):
    from datasets import load_dataset

    log.info("Evaluating ARC-Easy...")
    ds = load_dataset("ai2_arc", "ARC-Easy", split="test", trust_remote_code=True)
    correct = 0
    n = 0
    for item in ds:
        if n >= max_samples:
            break
        question = item["question"]
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        answer_key = item["answerKey"]

        scores = [
            score_continuation(
                model, tokenizer, "Question: " + question + "\nAnswer: ", c
            )
            for c in choices
        ]
        best_idx = int(np.argmax(scores))
        pred_label = labels[best_idx]
        if pred_label == answer_key:
            correct += 1
        n += 1
        if n % 100 == 0:
            log.info(f"  ARC-Easy: {n}/{max_samples}, acc={correct/n:.3f}")

    acc = correct / n if n > 0 else 0.0
    log.info(f"  ARC-Easy final: {correct}/{n} = {acc:.4f}")
    return acc, n


def eval_winogrande(model, tokenizer, max_samples):
    from datasets import load_dataset

    log.info("Evaluating Winogrande...")
    ds = load_dataset(
        "winogrande", "winogrande_xl", split="validation", trust_remote_code=True
    )
    correct = 0
    n = 0
    for item in ds:
        if n >= max_samples:
            break
        sentence = item["sentence"]
        opt1, opt2 = item["option1"], item["option2"]
        answer = item["answer"]

        s1 = score_continuation(model, tokenizer, "", sentence.replace("_", opt1))
        s2 = score_continuation(model, tokenizer, "", sentence.replace("_", opt2))

        pred = "1" if s1 > s2 else "2"
        if pred == answer:
            correct += 1
        n += 1
        if n % 100 == 0:
            log.info(f"  Winogrande: {n}/{max_samples}, acc={correct/n:.3f}")

    acc = correct / n if n > 0 else 0.0
    log.info(f"  Winogrande final: {correct}/{n} = {acc:.4f}")
    return acc, n


def run_config(config_name, skip_indices, model, tokenizer, max_samples):
    log.info(f"\n{'='*60}")
    log.info(f"Config: {config_name} (skip layers: {skip_indices})")
    log.info(f"{'='*60}")

    with SkipLayerContext(model, skip_indices):
        results = {}
        t0 = time.time()

        lmbd_acc, lmbd_n = eval_lambada(model, tokenizer, max_samples)
        results["lambada"] = {"accuracy": lmbd_acc, "n": lmbd_n}

        hella_acc, hella_n = eval_hellaswag(model, tokenizer, max_samples)
        results["hellaswag"] = {"accuracy": hella_acc, "n": hella_n}

        arc_acc, arc_n = eval_arc_easy(model, tokenizer, max_samples)
        results["arc_easy"] = {"accuracy": arc_acc, "n": arc_n}

        wino_acc, wino_n = eval_winogrande(model, tokenizer, max_samples)
        results["winogrande"] = {"accuracy": wino_acc, "n": wino_n}

        results["elapsed_s"] = time.time() - t0
        accs = [lmbd_acc, hella_acc, arc_acc, wino_acc]
        results["mean_accuracy"] = float(np.mean(accs))

        log.info(f"  Mean accuracy: {results['mean_accuracy']:.4f}")
        log.info(f"  Elapsed: {results['elapsed_s']:.1f}s")
        return results


def main():
    import torch

    log.info(f"=== PyTorch Downstream Evaluation ===")
    log.info(f"Model: {MODEL_NAME}")
    log.info(f"Max samples per task: {MAX_SAMPLES}")
    log.info(f"PyTorch version: {torch.__version__}")
    log.info(f"Device: CPU (BF16)")

    model, tokenizer = load_model()

    all_results = {
        "model": MODEL_NAME,
        "max_samples": MAX_SAMPLES,
        "dtype": "bfloat16",
        "device": "cpu",
        "n_layers": len(model.model.layers),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configs": {},
    }

    # Select which configs to run
    run_configs = os.environ.get("RUN_CONFIGS", "baseline,skip_n1,skip_n3,skip_n5")
    config_names = [c.strip() for c in run_configs.split(",")]

    for name in config_names:
        if name not in SKIP_CONFIGS:
            log.warning(f"Unknown config: {name}, skipping")
            continue
        try:
            result = run_config(name, SKIP_CONFIGS[name], model, tokenizer, MAX_SAMPLES)
            all_results["configs"][name] = result

            # Save incrementally
            with open(OUTPUT_PATH, "w") as f:
                json.dump(all_results, f, indent=2)
            log.info(f"Results saved to {OUTPUT_PATH}")

        except Exception as e:
            log.error(f"Config {name} failed: {e}", exc_info=True)
            all_results["configs"][name] = {"error": str(e)}

    # Final save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    log.info(f"\n{'='*60}")
    log.info(f"ALL DONE. Results at {OUTPUT_PATH}")
    log.info(f"{'='*60}")

    # Print summary
    for cfg, res in all_results["configs"].items():
        if "error" in res:
            log.info(f"  {cfg}: ERROR - {res['error']}")
        else:
            log.info(
                f"  {cfg}: LAMBADA={res['lambada']['accuracy']:.3f} "
                f"HS={res['hellaswag']['accuracy']:.3f} "
                f"ARC={res['arc_easy']['accuracy']:.3f} "
                f"WG={res['winogrande']['accuracy']:.3f} "
                f"mean={res['mean_accuracy']:.3f}"
            )


if __name__ == "__main__":
    main()
