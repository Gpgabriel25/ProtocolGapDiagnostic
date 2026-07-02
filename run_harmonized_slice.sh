#!/usr/bin/env bash
set -euo pipefail

# Run a strict cross-model matched-evaluator slice with identical evaluator settings.
# Usage:
#   ./run_harmonized_slice.sh [cycle_id]
# Optional env:
#   DATASET_NAME (default: wikitext)
#   DATASET_CONFIG (default: wikitext-2-raw-v1)
#   DATASET_SPLIT (default: test)
#   MAX_WORDS (default: 5000)

CYCLE_ID="${1:-2026-04-18T18-21-54}"
DATASET_NAME="${DATASET_NAME:-wikitext}"
DATASET_CONFIG="${DATASET_CONFIG:-wikitext-2-raw-v1}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
MAX_WORDS="${MAX_WORDS:-5000}"

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$ROOT_DIR/reports/$CYCLE_ID/harmonized"
mkdir -p "$OUT_DIR/qwen" "$OUT_DIR/pythia"

echo "[harmonized] cycle_id=$CYCLE_ID"
echo "[harmonized] dataset=${DATASET_NAME}/${DATASET_CONFIG}:${DATASET_SPLIT}, max_words=${MAX_WORDS}"

cd "$ROOT_DIR"

export EVAL_DATASET_NAME="$DATASET_NAME"
export EVAL_DATASET_CONFIG="$DATASET_CONFIG"
export EVAL_SPLIT="$DATASET_SPLIT"
export EVAL_MAX_WORDS="$MAX_WORDS"

# Qwen3-8B matched evaluator (bf16 TPU path in script)
REPORT_DIR="$OUT_DIR/qwen" python matched_eval_qwen3.py

# Pythia-1.4B matched evaluator (fp32 TPU path in script for stability)
REPORT_DIR="$OUT_DIR/pythia" python tpu_pythia_matched_eval.py

echo "[harmonized] finished. outputs under $OUT_DIR"
