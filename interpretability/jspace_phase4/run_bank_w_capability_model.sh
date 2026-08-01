#!/usr/bin/env bash
# Stage one pinned model to local NVMe and run its frozen Bank W baseline gate.
set -euo pipefail

PHASE4_ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$PHASE4_ROOT/../.." && pwd)"
RUN_ROOT="${JSPACE4_RUN_ROOT:-/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731}"
HF_CACHE="${HF_HUB_CACHE:-/content/hf_local}"
EXPECTED_BRANCH="${JSPACE4_EXPECTED_BRANCH:-interp_jspace_part2}"
CONFIG="interpretability/jspace_phase4/configs/p4_bank_w_capability_protocol_dev.yaml"
REGISTRY_PATH="interpretability/jspace_phase4/reports/evidence_events.jsonl"

if [[ "${1:-}" == "--approval-probe" ]]; then
  printf '%s\n' \
    "Bank W capability model runner is installed; no copy or GPU work started"
  exit 0
fi

MODEL_SLUG="${1:-}"
case "$MODEL_SLUG" in
  olmo31-think)
    CACHE_NAME="models--allenai--Olmo-3.1-32B-Think"
    REVISION="832c3f543499af8fe68b88359501de9cb7840544"
    SOURCE="/content/drive/MyDrive/hf_cache/hub/$CACHE_NAME/snapshots/$REVISION"
    ;;
  olmo31-instruct)
    CACHE_NAME="models--allenai--Olmo-3.1-32B-Instruct"
    REVISION="ac0587e4a7744a551c059d8cd17ba220bc940dae"
    SOURCE="/content/drive/MyDrive/hf_cache/hub/$CACHE_NAME/snapshots/$REVISION"
    ;;
  qwen36-27b)
    CACHE_NAME="models--Qwen--Qwen3.6-27B"
    REVISION="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    SOURCE=""
    ;;
  *)
    printf '%s\n' \
      "usage: $0 {olmo31-think|olmo31-instruct|qwen36-27b}" >&2
    exit 2
    ;;
esac

TARGET="$HF_CACHE/$CACHE_NAME/snapshots/$REVISION"
LOG="$RUN_ROOT/bank_w_capability_${MODEL_SLUG}_20260801.log"
HEARTBEAT="$RUN_ROOT/BANK_W_CAPABILITY_${MODEL_SLUG}_WATCHDOG.log"

export JSPACE4_RUN_ROOT="$RUN_ROOT"
export HF_HUB_CACHE="$HF_CACHE"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

mkdir -p "$RUN_ROOT" "$HF_CACHE"
cd "$REPO_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  printf '%s\n' "Bank W capability runner refused: repository is not clean" >&2
  git status --short >&2
  exit 3
fi
if [[ "$(git branch --show-current)" != "$EXPECTED_BRANCH" ]]; then
  printf '%s\n' "Bank W capability runner refused: unexpected Git branch" >&2
  git branch --show-current >&2
  exit 4
fi

snapshot_complete() {
  [[ -f "$TARGET/model.safetensors.index.json" ]] || return 1
  python - "$TARGET" <<'PY' >/dev/null
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
index = json.loads((root / "model.safetensors.index.json").read_text())
missing = {name for name in index["weight_map"].values()
           if not (root / name).is_file()}
raise SystemExit(1 if missing else 0)
PY
}

if ! snapshot_complete; then
  if [[ -z "$SOURCE" || ! -d "$SOURCE" ]]; then
    printf '%s\n' \
      "Pinned $MODEL_SLUG snapshot is absent from local NVMe and has no " \
      "configured Drive source. Materialize it at $TARGET before retrying." >&2
    exit 5
  fi
  source_bytes=$(du -sbL "$SOURCE" | awk '{print $1}')
  target_bytes=0
  if [[ -d "$TARGET" ]]; then
    target_bytes=$(du -sb "$TARGET" | awk '{print $1}')
  fi
  available_bytes=$(df -B1 --output=avail "$HF_CACHE" | tail -1 | tr -d ' ')
  reserve_bytes=$((5 * 1024 * 1024 * 1024))
  if (( available_bytes + target_bytes < source_bytes + reserve_bytes )); then
    printf '%s\n' \
      "Bank W capability runner refused: local NVMe cannot stage $MODEL_SLUG." \
      "source_bytes=$source_bytes target_bytes=$target_bytes " \
      "available_bytes=$available_bytes reserve_bytes=$reserve_bytes" >&2
    exit 6
  fi
  mkdir -p "$TARGET"
  printf '%s STAGE_START %s source=%s target=%s\n' \
    "$(date -u +%FT%TZ)" "$MODEL_SLUG" "$SOURCE" "$TARGET" \
    | tee -a "$LOG"
  rsync -rL --size-only --partial "$SOURCE/" "$TARGET/" \
    2>&1 | tee -a "$LOG"
  printf '%s STAGE_COMPLETE %s\n' "$(date -u +%FT%TZ)" "$MODEL_SLUG" \
    | tee -a "$LOG"
fi

python - "$TARGET" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
index = json.loads((root / "model.safetensors.index.json").read_text())
missing = sorted({name for name in index["weight_map"].values()
                  if not (root / name).is_file()})
if missing:
    raise SystemExit(f"local model snapshot lacks shards: {missing}")
print(f"verified {len(set(index['weight_map'].values()))} local weight shards")
PY

exec 9>"$RUN_ROOT/bank_w_capability_${MODEL_SLUG}.lock"
if ! flock -n 9; then
  printf '%s\n' "Bank W capability runner refused: lock already held" >&2
  exit 7
fi

heartbeat() {
  while true; do
    gpu_status=$(nvidia-smi \
      --query-gpu=utilization.gpu,memory.used,memory.total \
      --format=csv,noheader 2>/dev/null || printf '%s' unavailable)
    printf '%s runner_pid=%s gpu=[%s]\n' \
      "$(date -u +%FT%TZ)" "$$" "$gpu_status" >> "$HEARTBEAT"
    sleep 300
  done
}

heartbeat &
HEARTBEAT_PID=$!
trap 'kill "$HEARTBEAT_PID" 2>/dev/null || true' EXIT

printf '%s RUN_START %s commit=%s\n' \
  "$(date -u +%FT%TZ)" "$MODEL_SLUG" "$(git rev-parse HEAD)" \
  | tee -a "$LOG"
python -m jspace_phase4.experiments.p4_bank_w_capability \
  --config "$CONFIG" --model-slug "$MODEL_SLUG" 2>&1 | tee -a "$LOG"

mapfile -t changes < <(git status --porcelain)
if (( ${#changes[@]} == 1 )) && \
    [[ "${changes[0]:3}" == "$REGISTRY_PATH" ]]; then
  git add "$REGISTRY_PATH"
  git commit -m "data: register Bank W capability $MODEL_SLUG"
  git push origin "$EXPECTED_BRANCH"
elif (( ${#changes[@]} != 0 )); then
  printf '%s\n' \
    "Bank W capability runner refused to bank unexpected changes" >&2
  git status --short >&2
  exit 8
fi

printf '%s RUN_COMPLETE %s commit=%s\n' \
  "$(date -u +%FT%TZ)" "$MODEL_SLUG" "$(git rev-parse HEAD)" \
  | tee -a "$LOG"
