#!/usr/bin/env bash
# Durable GPU queue for the registered Qwen n=250 follow-up gates.
#
# The n=250 evidence must already be registered and its SHA-256 must replace
# the PENDING_REGISTERED_N250_LENS_SHA256 sentinels in both downstream YAMLs.
# Every producer is independently resumable and registry-idempotent.
set -euo pipefail

PHASE4_ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$PHASE4_ROOT/../.." && pwd)"
RUN_ROOT="${JSPACE4_RUN_ROOT:-/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731}"
LOCAL_WORK="${JSPACE4_LOCAL_WORK:-/content/sl4_work}"
HF_CACHE="${HF_HUB_CACHE:-/content/hf_local}"
MPL_CACHE="${MPLCONFIGDIR:-/tmp/matplotlib-phase4}"
QUEUE_LOG="$RUN_ROOT/qwen_postfit_queue_20260801.log"
HEARTBEAT_LOG="$RUN_ROOT/QWEN_POSTFIT_QUEUE_WATCHDOG.log"

if [[ "${1:-}" == "--approval-probe" ]]; then
  printf '%s\n' "qwen post-fit queue entrypoint is installed; no GPU work started"
  exit 0
fi

export JSPACE4_RUN_ROOT="$RUN_ROOT"
export JSPACE4_LOCAL_WORK="$LOCAL_WORK"
export HF_HUB_CACHE="$HF_CACHE"
export MPLCONFIGDIR="$MPL_CACHE"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

mkdir -p "$RUN_ROOT" "$LOCAL_WORK" "$MPL_CACHE"
cd "$REPO_ROOT"

if rg -n "PENDING_REGISTERED_N250_LENS_SHA256" \
    interpretability/jspace_phase4/configs/p4_qwen_lens_convergence_drawA_dev.yaml \
    interpretability/jspace_phase4/configs/p4_qwen_multilens_functional_gate_dev.yaml; then
  printf '%s\n' "post-fit queue refused: registered n=250 lens SHA is not bound" >&2
  exit 2
fi

if [[ -n "$(git status --porcelain)" ]]; then
  printf '%s\n' "post-fit queue refused: repository is not clean" >&2
  git status --short >&2
  exit 3
fi

heartbeat() {
  while true; do
    gpu_status=$(nvidia-smi \
      --query-gpu=utilization.gpu,memory.used,memory.total \
      --format=csv,noheader 2>/dev/null || printf '%s' unavailable)
    printf '%s queue_pid=%s gpu=[%s]\n' \
      "$(date -u +%FT%TZ)" "$$" "$gpu_status" >> "$HEARTBEAT_LOG"
    sleep 300
  done
}

heartbeat &
HEARTBEAT_PID=$!
trap 'kill "$HEARTBEAT_PID" 2>/dev/null || true' EXIT

run_stage() {
  local stage=$1
  shift
  printf '%s START %s commit=%s\n' \
    "$(date -u +%FT%TZ)" "$stage" "$(git rev-parse HEAD)" | tee -a "$QUEUE_LOG"
  "$@" 2>&1 | tee -a "$QUEUE_LOG"
  printf '%s COMPLETE %s\n' "$(date -u +%FT%TZ)" "$stage" | tee -a "$QUEUE_LOG"
}

run_stage qwen_lens_convergence \
  python -m jspace_phase4.experiments.p4_qwen_lens_convergence \
  --config interpretability/jspace_phase4/configs/p4_qwen_lens_convergence_drawA_dev.yaml

run_stage qwen_lens_influence_prompt112 \
  python -m jspace_phase4.experiments.p4_qwen_lens_influence \
  --config interpretability/jspace_phase4/configs/p4_qwen_lens_influence_prompt112_dev.yaml

run_stage qwen_multilens_functional_gate \
  python -m jspace_phase4.experiments.p4_qwen_multilens_functional_gate \
  --config interpretability/jspace_phase4/configs/p4_qwen_multilens_functional_gate_dev.yaml

printf '%s QUEUE_COMPLETE\n' "$(date -u +%FT%TZ)" | tee -a "$QUEUE_LOG"
