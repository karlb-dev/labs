#!/usr/bin/env bash
# Stage one pinned OLMo model on NVMe and run its baseline-only Bank-W gate.
set -euo pipefail

OL_ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
RUN_ROOT="${JSPACE_OLMO_RUN_ROOT:-/content/drive/MyDrive/interpret/special-lab-1/olmo_lineage_20260801}"
HF_CACHE="${HF_HUB_CACHE:-/content/hf_local}"
EXPECTED_BRANCH="interp_jspace_olmo_lineage"
CONFIG="interpretability/jspaces/sidelines/olmo/configs/ol_bank_w_capability_v1.yaml"
REGISTRY_PATH="interpretability/jspaces/sidelines/olmo/reports/evidence_events.jsonl"

if [[ "${1:-}" == "--approval-probe" ]]; then
  printf '%s\n' \
    "OLMo Bank-W runner installed; no copy, model load, or GPU work started"
  exit 0
fi

MODEL_SLUG="${1:-}"
case "$MODEL_SLUG" in
  olmo31-think)
    CACHE_NAME="models--allenai--Olmo-3.1-32B-Think"
    REVISION="832c3f543499af8fe68b88359501de9cb7840544"
    EVIDENCE_ID="ol-bank-w-capability-olmo31-think-dev-v1"
    ;;
  olmo31-instruct)
    CACHE_NAME="models--allenai--Olmo-3.1-32B-Instruct"
    REVISION="ac0587e4a7744a551c059d8cd17ba220bc940dae"
    EVIDENCE_ID="ol-bank-w-capability-olmo31-instruct-dev-v1"
    ;;
  *)
    printf '%s\n' "usage: $0 {olmo31-think|olmo31-instruct}" >&2
    exit 2
    ;;
esac

SOURCE="/content/drive/MyDrive/hf_cache/hub/$CACHE_NAME/snapshots/$REVISION"
TARGET="$HF_CACHE/$CACHE_NAME/snapshots/$REVISION"
LOG="$RUN_ROOT/logs/bank_w_capability_${MODEL_SLUG}_20260802.log"
HEARTBEAT="$RUN_ROOT/logs/BANK_W_CAPABILITY_${MODEL_SLUG}_WATCHDOG.log"

export JSPACE_OLMO_RUN_ROOT="$RUN_ROOT"
export HF_HUB_CACHE="$HF_CACHE"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

mkdir -p "$RUN_ROOT/logs" "$HF_CACHE"
cd "$REPO_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  printf '%s\n' "OLMo Bank-W runner refused: repository is not clean" >&2
  git status --short >&2
  exit 3
fi
if [[ "$(git branch --show-current)" != "$EXPECTED_BRANCH" ]]; then
  printf '%s\n' "OLMo Bank-W runner refused: unexpected Git branch" >&2
  git branch --show-current >&2
  exit 4
fi
if [[ ! -d "$SOURCE" ]]; then
  printf '%s\n' "Pinned Drive snapshot is absent: $SOURCE" >&2
  exit 5
fi

snapshot_complete() {
  [[ -f "$TARGET/model.safetensors.index.json" ]] || return 1
  python - "$SOURCE" "$TARGET" <<'PY' >/dev/null
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
index = json.loads((target / "model.safetensors.index.json").read_text())
names = sorted(set(index["weight_map"].values()))
for name in names:
    left = source / name
    right = target / name
    if not left.is_file() or not right.is_file():
        raise SystemExit(1)
    if left.stat().st_size != right.stat().st_size:
        raise SystemExit(1)
PY
}

if ! snapshot_complete; then
  source_bytes=$(du -sbL "$SOURCE" | awk '{print $1}')
  target_bytes=0
  if [[ -d "$TARGET" ]]; then
    target_bytes=$(du -sb "$TARGET" | awk '{print $1}')
  fi
  available_bytes=$(df -B1 --output=avail "$HF_CACHE" | tail -1 | tr -d ' ')
  reserve_bytes=$((10 * 1024 * 1024 * 1024))
  if (( available_bytes + target_bytes < source_bytes + reserve_bytes )); then
    printf '%s\n' \
      "Insufficient NVMe for $MODEL_SLUG: source=$source_bytes " \
      "target=$target_bytes available=$available_bytes reserve=$reserve_bytes" >&2
    exit 6
  fi
  mkdir -p "$TARGET"
  printf '%s STAGE_START %s source=%s target=%s\n' \
    "$(date -u +%FT%TZ)" "$MODEL_SLUG" "$SOURCE" "$TARGET" \
    | tee -a "$LOG"
  rsync -rL --size-only --partial "$SOURCE/" "$TARGET/" \
    2>&1 | tee -a "$LOG"
  printf '%s STAGE_COMPLETE %s\n' \
    "$(date -u +%FT%TZ)" "$MODEL_SLUG" | tee -a "$LOG"
fi

if ! snapshot_complete; then
  printf '%s\n' "Local snapshot failed exact shard-size verification" >&2
  exit 7
fi

python - "$TARGET" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
index = json.loads((root / "model.safetensors.index.json").read_text())
names = sorted(set(index["weight_map"].values()))
print(f"verified {len(names)} local weight shards under {root}")
PY

exec 9>"$RUN_ROOT/logs/bank_w_capability_${MODEL_SLUG}.lock"
if ! flock -n 9; then
  printf '%s\n' "OLMo Bank-W runner refused: model lock already held" >&2
  exit 8
fi

heartbeat() {
  while true; do
    gpu_status=$(nvidia-smi \
      --query-gpu=utilization.gpu,memory.used,memory.total \
      --format=csv,noheader 2>/dev/null || printf '%s' unavailable)
    printf '%s runner_pid=%s evidence=%s gpu=[%s]\n' \
      "$(date -u +%FT%TZ)" "$$" "$EVIDENCE_ID" "$gpu_status" \
      >> "$HEARTBEAT"
    sleep 300
  done
}

heartbeat &
HEARTBEAT_PID=$!
trap 'kill "$HEARTBEAT_PID" 2>/dev/null || true' EXIT

printf '%s RUN_START %s evidence=%s commit=%s\n' \
  "$(date -u +%FT%TZ)" "$MODEL_SLUG" "$EVIDENCE_ID" \
  "$(git rev-parse HEAD)" | tee -a "$LOG"
python -m jspace_olmo_lineage.experiments.bank_w_capability \
  --config "$CONFIG" --model-slug "$MODEL_SLUG" 2>&1 | tee -a "$LOG"

mapfile -t changes < <(git status --porcelain)
if (( ${#changes[@]} == 1 )) && \
    [[ "${changes[0]:3}" == "$REGISTRY_PATH" ]]; then
  git add "$REGISTRY_PATH"
  git commit -m "olmo: register Bank-W capability $MODEL_SLUG"
  git pull --rebase origin "$EXPECTED_BRANCH"
  bash interpretability/jspaces/sidelines/olmo/repro.sh
  git push origin "$EXPECTED_BRANCH"
elif (( ${#changes[@]} != 0 )); then
  printf '%s\n' "Runner refused to bank unexpected Git changes" >&2
  git status --short >&2
  exit 9
fi

printf '%s RUN_COMPLETE %s evidence=%s commit=%s\n' \
  "$(date -u +%FT%TZ)" "$MODEL_SLUG" "$EVIDENCE_ID" \
  "$(git rev-parse HEAD)" | tee -a "$LOG"
