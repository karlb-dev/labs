#!/usr/bin/env bash
# Drive-backed liveness watchdog for the resumable Phase 4 Qwen fit.
#
# Usage:
#   watchdog_phase4.sh CHECKPOINT_STATE_JSON TARGET_N [HEARTBEAT_LOG]
#
# The scientific producer atomically advances CHECKPOINT_STATE_JSON after
# every checkpoint_sync_every prompts. This watchdog treats that advancement
# as the primary liveness signal, records GPU/process status on Drive, and
# raises a durable alarm if the producer disappears or the recovery state
# stalls. It never restarts or mutates the scientific job.
set -u

STATE_PATH=${1:?checkpoint_state.json path is required}
TARGET_N=${2:?target prompt count is required}
HEARTBEAT_LOG=${3:-/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731/WATCHDOG_PHASE4.log}
INTERVAL_SECONDS=${PHASE4_WATCHDOG_INTERVAL_SECONDS:-300}
STALL_SECONDS=${PHASE4_WATCHDOG_STALL_SECONDS:-2100}
STARTUP_GRACE_SECONDS=${PHASE4_WATCHDOG_STARTUP_GRACE_SECONDS:-900}
PROCESS_PATTERN=${PHASE4_WATCHDOG_PROCESS_PATTERN:-[p]4_qwen_nested_lens_fit}

mkdir -p "$(dirname "$HEARTBEAT_LOG")"

say() {
  local message=$1
  printf '%s\n' "$message"
  printf '%s\n' "$message" >> "$HEARTBEAT_LOG"
}

read_n_done() {
  if [[ -s "$STATE_PATH" ]]; then
    jq -r '.n_done // -1' "$STATE_PATH" 2>/dev/null || printf '%s\n' -1
  else
    printf '%s\n' -1
  fi
}

state_signature() {
  if [[ -s "$STATE_PATH" ]]; then
    stat --printf='%Y:%s' "$STATE_PATH" 2>/dev/null || printf '%s\n' missing
  else
    printf '%s\n' missing
  fi
}

started_epoch=$(date -u +%s)
last_change_epoch=$started_epoch
last_signature=$(state_signature)
last_n_done=$(read_n_done)
say "$(date -u +%FT%TZ) watchdog armed target_n=$TARGET_N n_done=$last_n_done state=$STATE_PATH"

while true; do
  now_epoch=$(date -u +%s)
  signature=$(state_signature)
  n_done=$(read_n_done)
  process_count=$(pgrep -fc "$PROCESS_PATTERN" 2>/dev/null || true)
  gpu_status=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || printf '%s' unavailable)

  if [[ "$signature" != "$last_signature" || "$n_done" != "$last_n_done" ]]; then
    last_change_epoch=$now_epoch
    last_signature=$signature
    last_n_done=$n_done
  fi

  idle_seconds=$((now_epoch - last_change_epoch))
  runtime_seconds=$((now_epoch - started_epoch))
  say "$(date -u +%FT%TZ) watchdog ok n_done=$n_done/$TARGET_N idle_s=$idle_seconds processes=$process_count gpu=[$gpu_status]"

  if [[ "$n_done" =~ ^[0-9]+$ ]] && (( n_done >= TARGET_N )); then
    say "$(date -u +%FT%TZ) watchdog complete n_done=$n_done target_n=$TARGET_N"
    exit 0
  fi
  if (( process_count == 0 && runtime_seconds >= STARTUP_GRACE_SECONDS )); then
    say "$(date -u +%FT%TZ) WATCHDOG_ALARM producer_missing n_done=$n_done target_n=$TARGET_N"
    exit 2
  fi
  if (( idle_seconds >= STALL_SECONDS )); then
    say "$(date -u +%FT%TZ) WATCHDOG_ALARM recovery_state_stalled idle_s=$idle_seconds n_done=$n_done target_n=$TARGET_N"
    exit 3
  fi
  sleep "$INTERVAL_SECONDS"
done
