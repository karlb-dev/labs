#!/usr/bin/env bash
# Durable GPU entrypoint for the frozen post-functional-gate Qwen branch.
set -euo pipefail

PHASE4_ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
RUN_ROOT="${JSPACE4_RUN_ROOT:-/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731}"
LOCAL_WORK="${JSPACE4_LOCAL_WORK:-/content/sl4_work}"
HF_CACHE="${HF_HUB_CACHE:-/content/hf_local}"
MPL_CACHE="${MPLCONFIGDIR:-/tmp/matplotlib-phase4}"
EXPECTED_BRANCH="${JSPACE4_EXPECTED_BRANCH:-interp_jspace_part2}"
REGISTRY_PATH="interpretability/jspaces/phases/phase4/reports/evidence_events.jsonl"
CONFIG_PATH="interpretability/jspaces/phases/phase4/configs/p4_qwen_nested_lens_fit_dev.yaml"

case "${1:-}" in
  draw_b_n120)
    DRAW=draw_b
    STOP_AT=120
    EVIDENCE_ID=p4-qwen-lens-fit-drawB-n120-dev-v1
    ;;
  draw_a_n500)
    DRAW=draw_a
    STOP_AT=500
    EVIDENCE_ID=p4-qwen-lens-fit-drawA-n500-dev-v1
    ;;
  draw_a_n1000)
    DRAW=draw_a
    STOP_AT=1000
    EVIDENCE_ID=p4-qwen-lens-fit-drawA-n1000-dev-v1
    ;;
  --approval-probe)
    printf '%s\n' \
      "qwen continuation entrypoint is installed; no GPU work started"
    exit 0
    ;;
  *)
    printf '%s\n' \
      "usage: $0 {draw_b_n120|draw_a_n500|draw_a_n1000|--approval-probe}" >&2
    exit 2
    ;;
esac

export JSPACE4_RUN_ROOT="$RUN_ROOT"
export JSPACE4_LOCAL_WORK="$LOCAL_WORK"
export HF_HUB_CACHE="$HF_CACHE"
export MPLCONFIGDIR="$MPL_CACHE"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

mkdir -p "$RUN_ROOT" "$LOCAL_WORK" "$MPL_CACHE"
cd "$REPO_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  printf '%s\n' "continuation fit refused: repository is not clean" >&2
  git status --short >&2
  exit 3
fi
if [[ "$(git branch --show-current)" != "$EXPECTED_BRANCH" ]]; then
  printf '%s\n' "continuation fit refused: unexpected Git branch" >&2
  git branch --show-current >&2
  exit 4
fi

exec 9>"$RUN_ROOT/qwen_continuation_fit.lock"
if ! flock -n 9; then
  printf '%s\n' "continuation fit refused: another owner holds the lock" >&2
  exit 5
fi

LOG_PATH="$RUN_ROOT/qwen_continuation_${DRAW}_n${STOP_AT}_20260801.log"
HEARTBEAT_PATH="$RUN_ROOT/QWEN_CONTINUATION_WATCHDOG.log"

heartbeat() {
  while true; do
    gpu_status=$(nvidia-smi \
      --query-gpu=utilization.gpu,memory.used,memory.total \
      --format=csv,noheader 2>/dev/null || printf '%s' unavailable)
    printf '%s fit_pid=%s branch=%s gpu=[%s]\n' \
      "$(date -u +%FT%TZ)" "$$" "$1" "$gpu_status" \
      >> "$HEARTBEAT_PATH"
    sleep 300
  done
}

heartbeat "$1" &
HEARTBEAT_PID=$!
trap 'kill "$HEARTBEAT_PID" 2>/dev/null || true' EXIT

printf '%s START %s commit=%s\n' \
  "$(date -u +%FT%TZ)" "$EVIDENCE_ID" "$(git rev-parse HEAD)" \
  | tee -a "$LOG_PATH"
python -m jspace_phase4.experiments.p4_qwen_nested_lens_fit \
  --config "$CONFIG_PATH" --draw "$DRAW" --stop-at "$STOP_AT" \
  2>&1 | tee -a "$LOG_PATH"
printf '%s COMPLETE %s\n' "$(date -u +%FT%TZ)" "$EVIDENCE_ID" \
  | tee -a "$LOG_PATH"

mapfile -t changes < <(git status --porcelain)
if (( ${#changes[@]} == 0 )); then
  printf '%s REGISTRY_ALREADY_CLEAN %s\n' \
    "$(date -u +%FT%TZ)" "$EVIDENCE_ID" | tee -a "$LOG_PATH"
elif (( ${#changes[@]} == 1 )) \
    && [[ "${changes[0]:3}" == "$REGISTRY_PATH" ]]; then
  git add "$REGISTRY_PATH"
  git commit -m "data: register Qwen continuation fit"
  git push origin "$EXPECTED_BRANCH"
  printf '%s REGISTRY_BANKED %s commit=%s\n' \
    "$(date -u +%FT%TZ)" "$EVIDENCE_ID" "$(git rev-parse HEAD)" \
    | tee -a "$LOG_PATH"
else
  printf '%s\n' \
    "continuation fit refused to auto-bank unexpected repository changes" >&2
  git status --short >&2
  exit 6
fi
