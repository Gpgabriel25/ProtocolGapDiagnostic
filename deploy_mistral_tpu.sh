#!/usr/bin/env bash
# deploy_mistral_tpu.sh - Deploy and run Mistral-7B-v0.1 experiment on bq-v6e-8
# Usage: ./deploy_mistral_tpu.sh
# Prereqs: TPU bq-v6e-8 must be ACTIVE/READY in us-east1-d

set -e
ZONE=us-east1-d
TPU=bq-v6e-8
WORKER=0
QR_RECREATE_ATTEMPTED=0

recreate_queued_resource() {
  echo "=== Attempting automatic queued resource recreate for $TPU ==="
  gcloud compute tpus queued-resources delete "$TPU" --zone="$ZONE" --quiet >/dev/null 2>&1 || true
  gcloud alpha compute tpus queued-resources create "$TPU" \
    --zone="$ZONE" \
    --accelerator-type=v6e-8 \
    --runtime-version=v2-alpha-tpuv6e \
    --node-id="$TPU" \
    --spot \
    --provisioning-model=SPOT
}

wait_for_tpu() {
  local max_polls="${TPU_MAX_POLLS:-80}"
  local sleep_s="${TPU_POLL_SLEEP_S:-30}"
  local i
  local reached_active=0

  echo "=== Waiting for queued resource to become ACTIVE ==="
  for ((i=1; i<=max_polls; i++)); do
    local qr_state
    qr_state=$(gcloud compute tpus queued-resources list --zone="$ZONE" 2>/dev/null | awk -v tpu="$TPU" '$1==tpu {print $NF}')
    echo "[queued poll $i/$max_polls] state=${qr_state:-MISSING}"
    if [[ "$qr_state" == "ACTIVE" ]]; then
      reached_active=1
      break
    fi
    if [[ "$qr_state" == "FAILED" || "$qr_state" == "SUSPENDED" || "$qr_state" == "SUSPENDING" ]]; then
      if [[ "$QR_RECREATE_ATTEMPTED" -eq 0 ]]; then
        QR_RECREATE_ATTEMPTED=1
        recreate_queued_resource
        continue
      fi
      echo "Queued resource entered terminal state: $qr_state (auto-recreate already attempted)"
      return 1
    fi
    sleep "$sleep_s"
  done

  if [[ "$reached_active" -ne 1 ]]; then
    if [[ "$QR_RECREATE_ATTEMPTED" -eq 0 ]]; then
      echo "Queued resource did not become ACTIVE in time. Attempting one-time recreate..."
      QR_RECREATE_ATTEMPTED=1
      recreate_queued_resource
      wait_for_tpu
      return $?
    fi
    echo "Timed out waiting for queued resource to become ACTIVE"
    return 1
  fi

  echo "=== Waiting for TPU VM to become READY ==="
  for ((i=1; i<=max_polls; i++)); do
    local vm_state
    vm_state=$(gcloud compute tpus tpu-vm list --zone="$ZONE" 2>/dev/null | awk -v tpu="$TPU" '$1==tpu {print $NF}')
    echo "[vm poll $i/$max_polls] state=${vm_state:-MISSING}"

    local qr_state
    qr_state=$(gcloud compute tpus queued-resources list --zone="$ZONE" 2>/dev/null | awk -v tpu="$TPU" '$1==tpu {print $NF}')
    if [[ "$qr_state" == "FAILED" || "$qr_state" == "SUSPENDED" || "$qr_state" == "SUSPENDING" ]]; then
      echo "Queued resource entered terminal state during VM wait: $qr_state"
      return 1
    fi

    if [[ "$vm_state" == "READY" ]]; then
      return 0
    fi
    if [[ "$vm_state" == "DELETING" ]]; then
      echo "TPU VM is DELETING. Waiting for next provisioning cycle..."
    fi
    sleep "$sleep_s"
  done

  echo "Timed out waiting for TPU READY"
  return 1
}

echo "=== Checking TPU status ==="
wait_for_tpu
gcloud compute tpus tpu-vm list --zone=$ZONE | grep $TPU

echo ""
echo "=== Step 1: Enable transparent hugepages (all workers) ==="
gcloud compute tpus tpu-vm ssh $TPU --zone=$ZONE --worker=all --command="echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled && cat /sys/kernel/mm/transparent_hugepage/enabled"

echo ""
echo "=== Step 2: Install Python 3.11 + JAX (all workers) ==="
gcloud compute tpus tpu-vm ssh $TPU --zone=$ZONE --worker=all --command="
  set -e
  # Install Python 3.11 if needed
  python3 --version | grep -q '3.11\|3.12' || (
    sudo apt-get install -y software-properties-common &&
    sudo add-apt-repository -y ppa:deadsnakes/ppa &&
    sudo apt-get update &&
    sudo apt-get install -y python3.11 python3.11-pip python3.11-venv &&
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 10
  )
  python3 --version
  # Install JAX TPU
  python3 -m pip install --upgrade 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q
  python3 -m pip install transformers safetensors huggingface_hub datasets scipy -q
  python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu -q 2>/dev/null || true
  python3 -c 'import jax; print(\"JAX devices:\", jax.devices())'
"

echo ""
echo "=== Step 3: Copy experiment scripts to TPU ==="
gcloud compute tpus tpu-vm scp jax_7b_bisimulation.py $TPU:/tmp/ --zone=$ZONE --worker=all
gcloud compute tpus tpu-vm scp matched_eval_mistral.py $TPU:/tmp/ --zone=$ZONE --worker=all

echo ""
echo "=== Step 4: Run bisimulation experiment (worker 0 only) ==="
gcloud compute tpus tpu-vm ssh $TPU --zone=$ZONE --worker=$WORKER --command="
  cd /tmp
  MODEL=mistralai/Mistral-7B-v0.1 \
  N_PROMPTS=500 \
  SEQ_LEN=128 \
  REPORT_DIR=/tmp/mistral_bisim \
  python3 jax_7b_bisimulation.py 2>&1 | tee /tmp/mistral_bisim_run.log
  echo 'Bisimulation complete'
  ls /tmp/mistral_bisim/
"

echo ""
echo "=== Step 5: Fetch bisimulation results ==="
mkdir -p /tmp/mistral_bisim_results
gcloud compute tpus tpu-vm scp "$TPU:/tmp/mistral_bisim/*.json" /tmp/mistral_bisim_results/ --zone=$ZONE --worker=$WORKER 2>/dev/null || \
  gcloud compute tpus tpu-vm scp "$TPU:/tmp/mistral_bisim_run.log" /tmp/mistral_bisim_results/ --zone=$ZONE --worker=$WORKER

echo "Bisimulation results:"
ls /tmp/mistral_bisim_results/
cat /tmp/mistral_bisim_results/*.json 2>/dev/null | python3 -c "
import json,sys,numpy as np
d = json.load(sys.stdin)
results = d.get('results', [])
pairs = [(r['layer_a'], r['layer_b'], r['kl']) for r in results if r.get('kl') == r.get('kl') and r.get('kl') is not None]
pairs_sorted = sorted(pairs, key=lambda x: x[2])
print('Top 10 lowest KL pairs (interchange-guided):')
for la,lb,kl in pairs_sorted[:10]:
    print(f'  ({la},{lb}): {kl:.4f}')
print()
print('Top 5 highest KL pairs (replacement-guided):')
for la,lb,kl in sorted(pairs, key=lambda x: x[2], reverse=True)[:5]:
    print(f'  ({la},{lb}): {kl:.4f}')
" 2>/dev/null || echo "Parse step will be done manually"

echo ""
echo "=== Step 6: Run matched eval (set BISIM_JSON if available) ==="
BISIM_JSON_PATH=$(gcloud compute tpus tpu-vm ssh $TPU --zone=$ZONE --worker=$WORKER --command="ls /tmp/mistral_bisim/*.json 2>/dev/null | head -1" 2>/dev/null || echo "")
echo "BISIM_JSON: $BISIM_JSON_PATH"

gcloud compute tpus tpu-vm ssh $TPU --zone=$ZONE --worker=$WORKER --command="
  cd /tmp
  BISIM_JSON=\$(ls /tmp/mistral_bisim/*.json 2>/dev/null | head -1)
  MODEL=mistralai/Mistral-7B-v0.1 \
  BISIM_JSON=\$BISIM_JSON \
  REPORT_DIR=/tmp/mistral_eval \
  python3 matched_eval_mistral.py 2>&1 | tee /tmp/mistral_eval_run.log
  echo 'Matched eval complete'
"

echo ""
echo "=== Step 7: Fetch results ==="
mkdir -p /tmp/mistral_eval_results
gcloud compute tpus tpu-vm scp "$TPU:/tmp/mistral_eval/*.json" /tmp/mistral_eval_results/ --zone=$ZONE --worker=$WORKER
gcloud compute tpus tpu-vm scp "$TPU:/tmp/mistral_eval_run.log" /tmp/mistral_eval_results/ --zone=$ZONE --worker=$WORKER 2>/dev/null || true
echo "Results:"
ls /tmp/mistral_eval_results/

echo ""
echo "=== DONE ==="
echo "Results at: /tmp/mistral_eval_results/"
