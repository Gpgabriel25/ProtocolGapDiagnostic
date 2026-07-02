#!/bin/bash
# Launch 152M Chinchilla PE ablation (RoPE + AbsPE) — JAX pmap version — on TPU v6e-4
# sic-v6e-4, europe-west4-a
# Builder Council 2026-04-20T11-06-15 minimum-viable configuration
#
# pe_ablation_jax.py uses jax.pmap across all 4 local devices (per-device batch=16, total=64)

set -e

# Model (152M: 16L, d_model=768, 12 heads, d_ff=3072)
export PE_N_LAYERS=16
export PE_N_HEADS=12
export PE_D_MODEL=768
export PE_D_FF=3072
export PE_MAX_SEQ_LEN=512
export PE_SEQ_LEN=512
export PE_DROPOUT=0.1

# Training (Chinchilla: ~2B tokens = 61035 steps @ 32768 tok/step; effective batch=64)
export PE_BATCH_SIZE=64
export PE_TOTAL_STEPS=61035
export PE_LR=3e-4
export PE_WARMUP_STEPS=2000
export PE_GRAD_CLIP=1.0
export PE_LOG_STEPS=500
export PE_EVAL_STEPS=5000

# Data
export PE_TRAIN_MAX_TOKENS=100000000

# Distance measurement (emergence trajectory: 500M, 1B, 2B token marks)
export PE_DISTANCE_CHECKPOINTS=15000,30000,61035
export PE_CHECKPOINT_PROMPTS=100
export PE_FINAL_PROMPTS=100
export PE_EVAL_PROMPTS=500
export PE_JACOBIAN_PROMPTS=10
export PE_MAX_GAP=4

# Pmap data-parallel across all 4 v6e-4 chips
export PE_USE_PMAP=1

# Output
export PE_EXPERIMENT_NAME=pe_ablation_152m_chinchilla_pmap
export PE_OUTPUT_DIR=$HOME/pe_ablation_152m_pmap_output

mkdir -p $PE_OUTPUT_DIR

echo "[$(date)] Launching 152M PE ablation (pmap, 4 chips) on $(hostname)" | tee $PE_OUTPUT_DIR/launch.log
echo "Effective batch=$PE_BATCH_SIZE; per-device batch=$((PE_BATCH_SIZE/4)); total steps=$PE_TOTAL_STEPS" | tee -a $PE_OUTPUT_DIR/launch.log

cd /tmp
nohup python3 -u /tmp/pe_ablation_jax.py > $PE_OUTPUT_DIR/run.log 2>&1 &
PID=$!
echo "PID=$PID" | tee -a $PE_OUTPUT_DIR/launch.log
sleep 3
ps -p $PID -o pid,etime,cmd
