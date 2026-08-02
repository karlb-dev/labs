#!/usr/bin/env bash
# Durable queue for the frozen A500-to-A1000 decision boundary. The exact
# registered A1000 SHA must already be bound in the three prospective configs.
set -euo pipefail

PHASE4_ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$PHASE4_ROOT/../.." && pwd)"
RUN_ROOT="${JSPACE4_RUN_ROOT:-/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731}"
LOCAL_WORK="${JSPACE4_LOCAL_WORK:-/content/sl4_work}"
HF_CACHE="${HF_HUB_CACHE:-/content/hf_local}"
MPL_CACHE="${MPLCONFIGDIR:-/tmp/matplotlib-phase4-a1000}"
EXPECTED_BRANCH="${JSPACE4_EXPECTED_BRANCH:-interp_jspace_part2}"
REGISTRY_PATH="interpretability/jspace_phase4/reports/evidence_events.jsonl"
STRUCTURAL_CONFIG="interpretability/jspace_phase4/configs/p4_qwen_lens_convergence_drawA_n500_n1000_dev.yaml"
FUNCTIONAL_CONFIG="interpretability/jspace_phase4/configs/p4_qwen_multilens_functional_gate_a500_a1000_dev.yaml"
MARGIN_CONFIG="interpretability/jspace_phase4/configs/p4_qwen_selection_margin_a500_a1000_dev.yaml"
INFLUENCE_CONFIG="interpretability/jspace_phase4/configs/p4_qwen_lens_influence_prompt323_dev.yaml"
DECISION_CONFIG="interpretability/jspace_phase4/configs/p4_qwen_canonical_lens_decision_a1000_dev.yaml"
QUEUE_LOG="$RUN_ROOT/qwen_a1000_postfit_queue_20260802.log"
HEARTBEAT_LOG="$RUN_ROOT/QWEN_A1000_POSTFIT_QUEUE_WATCHDOG.log"
STOP_AFTER="${JSPACE4_STOP_AFTER:-canonical}"

if [[ "${1:-}" == "--approval-probe" ]]; then
  printf '%s\n' \
    "qwen A1000 post-fit queue is installed; no file or GPU work started"
  exit 0
fi

case "$STOP_AFTER" in
  structural|functional|margin|influence|canonical) ;;
  *)
    printf '%s\n' \
      "A1000 post-fit queue refused: invalid JSPACE4_STOP_AFTER=$STOP_AFTER" \
      >&2
    exit 2
    ;;
esac

export JSPACE4_RUN_ROOT="$RUN_ROOT"
export JSPACE4_LOCAL_WORK="$LOCAL_WORK"
export HF_HUB_CACHE="$HF_CACHE"
export MPLCONFIGDIR="$MPL_CACHE"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PHASE4_ROOT:$REPO_ROOT/interpretability/jspace_phase3:$REPO_ROOT/interpretability/jspace_part2${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$RUN_ROOT" "$LOCAL_WORK" "$MPL_CACHE"
cd "$REPO_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  printf '%s\n' "A1000 post-fit queue refused: repository is not clean" >&2
  git status --short >&2
  exit 2
fi
if [[ "$(git branch --show-current)" != "$EXPECTED_BRANCH" ]]; then
  printf '%s\n' "A1000 post-fit queue refused: unexpected Git branch" >&2
  git branch --show-current >&2
  exit 3
fi
if pgrep -f '[p]4_qwen_nested_lens_fit' >/dev/null; then
  printf '%s\n' \
    "A1000 post-fit queue refused: the frozen lens fitter still owns the GPU" >&2
  exit 4
fi

git pull --ff-only origin "$EXPECTED_BRANCH"

exec 9>"$RUN_ROOT/qwen_a1000_postfit_queue.lock"
if ! flock -n 9; then
  printf '%s\n' \
    "A1000 post-fit queue refused: another owner holds the queue lock" >&2
  exit 5
fi

A1000_SHA=$(python - \
  "$STRUCTURAL_CONFIG" "$FUNCTIONAL_CONFIG" "$INFLUENCE_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

import yaml

from jspace_phase4.manifests import file_sha256
from jspace_phase4.paths4 import resolve_uri
from jspace_phase4.registry4 import resolve

event_id = "p4-qwen-lens-fit-drawA-n1000-dev-v1"
uri = (
    "artifact://phase4/lens/qwen36-27b/nested_fit/draw_a/"
    "qwen36-27b_jlens_drawA_n1000.pt")
event = resolve(event_id)
if not event["live"]:
    raise SystemExit("registered A1000 fit event is not live")
path = resolve_uri(uri)
matches = [
    row for row in event["outputs"]
    if Path(row["path"]).resolve() == path.resolve()
]
if len(matches) != 1:
    raise SystemExit("A1000 fit event does not contain exactly one final lens")
digest = matches[0]["sha256"]
if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
    raise SystemExit("registered A1000 lens digest is malformed")
if file_sha256(path) != digest:
    raise SystemExit("registered A1000 lens bytes do not match the event")

for config_name in sys.argv[1:]:
    config = yaml.safe_load(Path(config_name).read_text())
    specification = config["lenses"]["a1000"]
    expected = {
        "evidence_id": event_id,
        "lens_uri": uri,
        "lens_sha256": digest,
        "n_prompts": 1000,
    }
    observed = {key: specification.get(key) for key in expected}
    if observed != expected:
        raise SystemExit(
            f"A1000 binding drift in {config_name}: "
            + json.dumps({"expected": expected, "observed": observed},
                         sort_keys=True))
print(digest)
PY
)
printf '%s VERIFIED_A1000_BINDING sha256=%s\n' \
  "$(date -u +%FT%TZ)" "$A1000_SHA" | tee -a "$QUEUE_LOG"

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
  if [[ -n "$(git status --porcelain)" ]]; then
    printf '%s\n' "A1000 queue found a dirty tree before $stage" >&2
    git status --short >&2
    exit 6
  fi
  printf '%s START %s commit=%s a1000=%s\n' \
    "$(date -u +%FT%TZ)" "$stage" "$(git rev-parse HEAD)" "$A1000_SHA" \
    | tee -a "$QUEUE_LOG"
  "$@" 2>&1 | tee -a "$QUEUE_LOG"
  printf '%s COMPLETE %s\n' "$(date -u +%FT%TZ)" "$stage" \
    | tee -a "$QUEUE_LOG"
}

stop_after_stage() {
  local stage=$1
  if [[ "$STOP_AFTER" == "$stage" ]]; then
    printf '%s QUEUE_STOPPED_AFTER stage=%s restart_safe=true\n' \
      "$(date -u +%FT%TZ)" "$stage" | tee -a "$QUEUE_LOG"
    exit 0
  fi
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
      "A1000 queue refused to bank unexpected repository changes" >&2
    git status --short >&2
    exit 7
  fi
  git add "$REGISTRY_PATH"
  git commit -m "data: register $stage"
  git pull --ff-only origin "$EXPECTED_BRANCH"
  git push origin "$EXPECTED_BRANCH"
  printf '%s REGISTRY_BANKED %s commit=%s\n' \
    "$(date -u +%FT%TZ)" "$stage" "$(git rev-parse HEAD)" \
    | tee -a "$QUEUE_LOG"
}

preserve_registered_outputs() {
  local evidence_id=$1
  local backup_root="$LOCAL_WORK/postfit_registered_backups"
  python - "$evidence_id" "$backup_root" <<'PY' | tee -a "$QUEUE_LOG"
import json
import os
import shutil
import sys
from pathlib import Path

from jspace_phase4.manifests import file_sha256
from jspace_phase4.registry4 import resolve

evidence_id = sys.argv[1]
backup_root = Path(sys.argv[2]) / evidence_id
event = resolve(evidence_id)
if not event["live"]:
    raise SystemExit(f"cannot preserve non-live evidence {evidence_id}")
backup_root.mkdir(parents=True, exist_ok=True)
manifest = {"schema_version": 1, "evidence_id": evidence_id, "outputs": []}
for ordinal, output in enumerate(event["outputs"]):
    source = Path(output["path"])
    expected = output["sha256"]
    if file_sha256(source) != expected:
        raise SystemExit(f"registered output hash mismatch: {source}")
    destination = backup_root / f"{ordinal:02d}_{source.name}"
    if destination.exists():
        if file_sha256(destination) != expected:
            raise SystemExit(f"local backup hash mismatch: {destination}")
    else:
        temporary = destination.with_suffix(
            destination.suffix + f".tmp{os.getpid()}")
        shutil.copyfile(source, temporary)
        if file_sha256(temporary) != expected:
            temporary.unlink(missing_ok=True)
            raise SystemExit(f"copied backup hash mismatch: {destination}")
        os.replace(temporary, destination)
    manifest["outputs"].append({
        "registered_path": str(source), "backup_path": str(destination),
        "sha256": expected, "bytes": destination.stat().st_size,
    })
manifest_path = backup_root / "backup_manifest.json"
temporary = manifest_path.with_suffix(
    manifest_path.suffix + f".tmp{os.getpid()}")
temporary.write_text(json.dumps(manifest, sort_keys=True, indent=1) + "\n")
os.replace(temporary, manifest_path)
print(json.dumps({
    "local_backup_complete": evidence_id,
    "outputs": len(manifest["outputs"]),
    "bytes": sum(row["bytes"] for row in manifest["outputs"]),
    "manifest": str(manifest_path),
}, sort_keys=True))
PY
}

preserve_registered_outputs p4-qwen-lens-fit-drawA-n1000-dev-v1

run_stage qwen_lens_convergence_a500_a1000 \
  python -m jspace_phase4.experiments.p4_qwen_lens_convergence \
  --config "$STRUCTURAL_CONFIG"
bank_registry_event qwen_lens_convergence_a500_a1000
preserve_registered_outputs \
  p4-qwen-lens-convergence-drawA-n500-n1000-dev-v1
stop_after_stage structural

run_stage qwen_multilens_functional_gate_a500_a1000 \
  python -m jspace_phase4.experiments.p4_qwen_multilens_functional_gate \
  --config "$FUNCTIONAL_CONFIG"
bank_registry_event qwen_multilens_functional_gate_a500_a1000
preserve_registered_outputs \
  p4-qwen-multilens-functional-gate-a500-a1000-published-dev-v1
stop_after_stage functional

run_stage qwen_selection_margin_a500_a1000 \
  python -m jspace_phase4.experiments.p4_qwen_selection_margin \
  --config "$MARGIN_CONFIG"
bank_registry_event qwen_selection_margin_a500_a1000
preserve_registered_outputs p4-qwen-selection-margin-a500-a1000-dev-v1
stop_after_stage margin

run_stage qwen_lens_influence_prompt323 \
  python -m jspace_phase4.experiments.p4_qwen_lens_influence_paired \
  --config "$INFLUENCE_CONFIG"
bank_registry_event qwen_lens_influence_prompt323
preserve_registered_outputs p4-qwen-lens-influence-prompt323-dev-v1
stop_after_stage influence

run_stage qwen_canonical_lens_decision_a1000 \
  python -m jspace_phase4.experiments.p4_qwen_canonical_lens_decision \
  --config "$DECISION_CONFIG"
bank_registry_event qwen_canonical_lens_decision_a1000
preserve_registered_outputs \
  p4-qwen-canonical-lens-decision-a1000-dev-v1
stop_after_stage canonical

printf '%s QUEUE_COMPLETE canonical_decision_registered=true\n' \
  "$(date -u +%FT%TZ)" | tee -a "$QUEUE_LOG"
