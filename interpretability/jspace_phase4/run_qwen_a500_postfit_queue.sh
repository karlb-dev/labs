#!/usr/bin/env bash
# Durable GPU queue for the pre-frozen A250-to-A500, mode-v2, and Qwen
# Bank W baseline-capability gates.
set -euo pipefail

PHASE4_ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$PHASE4_ROOT/../.." && pwd)"
RUN_ROOT="${JSPACE4_RUN_ROOT:-/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731}"
LOCAL_WORK="${JSPACE4_LOCAL_WORK:-/content/sl4_work}"
HF_CACHE="${HF_HUB_CACHE:-/content/hf_local}"
MPL_CACHE="${MPLCONFIGDIR:-/tmp/matplotlib-phase4}"
EXPECTED_BRANCH="${JSPACE4_EXPECTED_BRANCH:-interp_jspace_part2}"
REGISTRY_PATH="interpretability/jspace_phase4/reports/evidence_events.jsonl"
CONVERGENCE_CONFIG="interpretability/jspace_phase4/configs/p4_qwen_lens_convergence_drawA_n250_n500_dev.yaml"
FUNCTIONAL_CONFIG="interpretability/jspace_phase4/configs/p4_qwen_multilens_functional_gate_a250_a500_dev.yaml"
MODE_CONFIG="interpretability/jspace_phase4/configs/p4_qwen_mode_model_gate_v2_dev.yaml"
BANK_W_CAPABILITY_CONFIG="interpretability/jspace_phase4/configs/p4_bank_w_capability_protocol_dev.yaml"
QUEUE_LOG="$RUN_ROOT/qwen_a500_postfit_queue_20260801.log"
HEARTBEAT_LOG="$RUN_ROOT/QWEN_A500_POSTFIT_QUEUE_WATCHDOG.log"

if [[ "${1:-}" == "--approval-probe" ]]; then
  printf '%s\n' \
    "qwen A500 post-fit queue entrypoint is installed; no GPU work started"
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

if rg -n "PENDING_REGISTERED_A500_SHA256" \
    "$CONVERGENCE_CONFIG" "$FUNCTIONAL_CONFIG"; then
  printf '%s\n' \
    "A500 post-fit queue refused: registered A500 lens SHA is not bound" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  printf '%s\n' "A500 post-fit queue refused: repository is not clean" >&2
  git status --short >&2
  exit 3
fi
if [[ "$(git branch --show-current)" != "$EXPECTED_BRANCH" ]]; then
  printf '%s\n' "A500 post-fit queue refused: unexpected Git branch" >&2
  git branch --show-current >&2
  exit 4
fi

exec 9>"$RUN_ROOT/qwen_a500_postfit_queue.lock"
if ! flock -n 9; then
  printf '%s\n' "A500 post-fit queue refused: another owner holds the lock" >&2
  exit 5
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
    "$(date -u +%FT%TZ)" "$stage" "$(git rev-parse HEAD)" \
    | tee -a "$QUEUE_LOG"
  "$@" 2>&1 | tee -a "$QUEUE_LOG"
  printf '%s COMPLETE %s\n' "$(date -u +%FT%TZ)" "$stage" \
    | tee -a "$QUEUE_LOG"
}

bank_registry_event() {
  local stage=$1
  mapfile -t changes < <(git status --porcelain)
  if (( ${#changes[@]} == 0 )); then
    printf '%s REGISTRY_ALREADY_CLEAN %s\n' \
      "$(date -u +%FT%TZ)" "$stage" | tee -a "$QUEUE_LOG"
    return
  fi
  if (( ${#changes[@]} != 1 )) || \
      [[ "${changes[0]:3}" != "$REGISTRY_PATH" ]]; then
    printf '%s\n' \
      "A500 post-fit queue refused to bank unexpected repository changes" >&2
    git status --short >&2
    exit 6
  fi
  git add "$REGISTRY_PATH"
  git commit -m "data: register $stage"
  git push origin "$EXPECTED_BRANCH"
  printf '%s REGISTRY_BANKED %s commit=%s\n' \
    "$(date -u +%FT%TZ)" "$stage" "$(git rev-parse HEAD)" \
    | tee -a "$QUEUE_LOG"
}

run_stage qwen_lens_convergence_a250_a500 \
  python -m jspace_phase4.experiments.p4_qwen_lens_convergence \
  --config "$CONVERGENCE_CONFIG"
bank_registry_event qwen_lens_convergence_a250_a500

run_stage qwen_multilens_functional_gate_a250_a500 \
  python -m jspace_phase4.experiments.p4_qwen_multilens_functional_gate \
  --config "$FUNCTIONAL_CONFIG"
bank_registry_event qwen_multilens_functional_gate_a250_a500

run_stage qwen_mode_model_gate_v2 \
  python -m jspace_phase4.experiments.p4_qwen_mode_model_gate \
  --config "$MODE_CONFIG"
bank_registry_event qwen_mode_model_gate_v2

run_stage bank_w_capability_qwen36_27b \
  python -m jspace_phase4.experiments.p4_bank_w_capability \
  --config "$BANK_W_CAPABILITY_CONFIG" --model-slug qwen36-27b
bank_registry_event bank_w_capability_qwen36_27b

printf '%s QUEUE_COMPLETE\n' "$(date -u +%FT%TZ)" | tee -a "$QUEUE_LOG"
