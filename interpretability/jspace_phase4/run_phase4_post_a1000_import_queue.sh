#!/usr/bin/env bash
# Resume-safe, methods-only admission queue for completed Gemma and OLMo side
# tracks. Run only after the frozen Qwen A1000 canonical decision is registered.
set -euo pipefail

PHASE4_ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$PHASE4_ROOT/../.." && pwd)"
RUN_ROOT="${JSPACE4_RUN_ROOT:-/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731}"
LOCAL_WORK="${JSPACE4_LOCAL_WORK:-/content/sl4_work}"
EXPECTED_BRANCH="${JSPACE4_EXPECTED_BRANCH:-interp_jspace_part2}"
REGISTRY_PATH="interpretability/jspace_phase4/reports/evidence_events.jsonl"
EARLY_OLMO_BUNDLE="interpretability/jspace_phase4/reports/OLMO_BANK_W_CAPABILITY_IMPORT_V1.json"
EARLY_OLMO_VALIDATION="interpretability/jspace_phase4/reports/olmo_bank_w_capability_import_validation_v1.json"
JOINT_CONFIG="interpretability/jspace_phase4/configs/p4_bank_w_joint_imported_dev_v1.yaml"
JOINT_REPORT="interpretability/jspace_phase4/reports/bank_w_capability_joint_imported_dev_v1.json"
JOINT_PNG="interpretability/jspace_phase4/reports/figures/p4f30_bank_w_joint_imported.png"
JOINT_PDF="interpretability/jspace_phase4/reports/figures/p4f30_bank_w_joint_imported.pdf"
GEMMA_BUNDLE="interpretability/jspace_phase4/reports/GEMMA_TRANSPORT_IMPORT_V1.json"
GEMMA_VALIDATION="interpretability/jspace_phase4/reports/gemma_transport_import_validation_v1.json"
FINAL_OLMO_BUNDLE="interpretability/jspace_phase4/reports/OLMO_LINEAGE_IMPORT_V1.json"
FINAL_OLMO_VALIDATION="interpretability/jspace_phase4/reports/olmo_lineage_import_validation_v1.json"
QUEUE_LOG="$RUN_ROOT/phase4_post_a1000_import_queue_20260802.log"

if [[ "${1:-}" == "--approval-probe" ]]; then
  printf '%s\n' \
    "phase4 post-A1000 import queue is installed; no file work started"
  exit 0
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PHASE4_ROOT:$REPO_ROOT/interpretability/jspace_phase3:$REPO_ROOT/interpretability/jspace_part2${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$RUN_ROOT" "$LOCAL_WORK"
cd "$REPO_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  printf '%s\n' "post-A1000 import queue refused: repository is not clean" >&2
  git status --short >&2
  exit 2
fi
if [[ "$(git branch --show-current)" != "$EXPECTED_BRANCH" ]]; then
  printf '%s\n' "post-A1000 import queue refused: unexpected Git branch" >&2
  git branch --show-current >&2
  exit 3
fi

git pull --ff-only origin "$EXPECTED_BRANCH"

exec 9>"$RUN_ROOT/phase4_post_a1000_import_queue.lock"
if ! flock -n 9; then
  printf '%s\n' "post-A1000 import queue refused: lock is held" >&2
  exit 4
fi

python - <<'PY'
from jspace_phase4.registry4 import read_events, resolve

required = [
    "p4-qwen-lens-fit-drawA-n1000-dev-v1",
    "p4-qwen-lens-convergence-drawA-n500-n1000-dev-v1",
    "p4-qwen-multilens-functional-gate-a500-a1000-published-dev-v1",
    "p4-qwen-selection-margin-a500-a1000-dev-v1",
    "p4-qwen-lens-influence-prompt323-dev-v1",
    "p4-qwen-canonical-lens-decision-a1000-dev-v1",
]
for evidence_id in required:
    if not resolve(evidence_id)["live"]:
        raise SystemExit(f"required post-A1000 evidence is not live: {evidence_id}")
native_side_ids = sorted({
    row["evidence_id"] for row in read_events()
    if str(row.get("evidence_id", "")).startswith(("ol-", "gm-"))
})
if native_side_ids:
    raise SystemExit(
        "native side-track IDs leaked into Phase 4 registry: "
        + ", ".join(native_side_ids))
print("verified canonical A1000 boundary and native-namespace exclusion")
PY

event_status() {
  python - "$1" <<'PY'
import sys
from jspace_phase4.registry4 import read_events, resolve

evidence_id = sys.argv[1]
origins = [
    row for row in read_events()
    if row.get("evidence_id") == evidence_id
    and row.get("event") in {"evidence_created", "evidence_imported"}
]
if not origins:
    print("absent")
elif len(origins) != 1:
    raise SystemExit(f"ambiguous origin count for {evidence_id}: {len(origins)}")
else:
    print("live" if resolve(evidence_id)["live"] else "dead")
PY
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
field = "source_outputs" if event["event"] == "evidence_imported" else "outputs"
backup_root.mkdir(parents=True, exist_ok=True)
manifest = {"schema_version": 1, "evidence_id": evidence_id, "outputs": []}
for ordinal, output in enumerate(event[field]):
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

bank_registry_event() {
  local stage=$1
  local evidence_id=$2
  mapfile -t changes < <(git status --porcelain)
  if (( ${#changes[@]} != 1 )) || \
      [[ "${changes[0]:3}" != "$REGISTRY_PATH" ]]; then
    printf '%s\n' "import queue refused unexpected changes after $stage" >&2
    git status --short >&2
    exit 5
  fi
  git add "$REGISTRY_PATH"
  git commit -m "data: register $stage"
  git pull --ff-only origin "$EXPECTED_BRANCH"
  git push origin "$EXPECTED_BRANCH"
  preserve_registered_outputs "$evidence_id"
  printf '%s REGISTERED %s commit=%s\n' \
    "$(date -u +%FT%TZ)" "$stage" "$(git rev-parse HEAD)" | tee -a "$QUEUE_LOG"
}

verify_existing_import() {
  local evidence_id=$1
  local bundle=$2
  local validation=$3
  python - "$evidence_id" "$bundle" "$validation" <<'PY'
import json
import sys
from pathlib import Path

from jspace_phase4.import_bundle import validate_import_bundle
from jspace_phase4.registry4 import resolve

evidence_id, bundle, validation = sys.argv[1:]
recorded = json.loads(Path(validation).read_text())
current = validate_import_bundle(bundle, allow_existing_target=True)
if current != recorded:
    raise SystemExit(f"fresh validation drift for {evidence_id}")
event = resolve(evidence_id)
if not event["live"] or event["event"] != "evidence_imported":
    raise SystemExit(f"existing import is not a live import: {evidence_id}")
if current["target_import_evidence_id"] != evidence_id:
    raise SystemExit(f"bundle target drift for {evidence_id}")
print(f"fresh validation matches existing import: {evidence_id}")
PY
}

register_import() {
  local stage=$1
  local evidence_id=$2
  local bundle=$3
  local validation=$4
  local status
  status=$(event_status "$evidence_id")
  if [[ "$status" == "dead" ]]; then
    printf '%s\n' "import queue refused dead target event: $evidence_id" >&2
    exit 6
  fi
  if [[ "$status" == "live" ]]; then
    verify_existing_import "$evidence_id" "$bundle" "$validation"
    preserve_registered_outputs "$evidence_id"
    printf '%s ALREADY_REGISTERED %s\n' \
      "$(date -u +%FT%TZ)" "$stage" | tee -a "$QUEUE_LOG"
    return
  fi
  python -m jspace_phase4.experiments.p4_import_side_bundle \
    --bundle "$bundle" --validation "$validation" 2>&1 | tee -a "$QUEUE_LOG"
  bank_registry_event "$stage" "$evidence_id"
}

bank_joint_outputs() {
  local expected=("$JOINT_REPORT" "$JOINT_PNG" "$JOINT_PDF")
  mapfile -t changes < <(git status --porcelain)
  if (( ${#changes[@]} != ${#expected[@]} )); then
    printf '%s\n' "joint replay produced an unexpected change count" >&2
    git status --short >&2
    exit 7
  fi
  for path in "${expected[@]}"; do
    if ! printf '%s\n' "${changes[@]}" | cut -c4- | grep -Fxq "$path"; then
      printf '%s\n' "joint replay did not exclusively produce $path" >&2
      exit 8
    fi
  done
  git add "${expected[@]}"
  git commit -m "data: materialize imported Bank-W joint replay"
  git pull --ff-only origin "$EXPECTED_BRANCH"
  git push origin "$EXPECTED_BRANCH"
}

register_joint_replay() {
  local evidence_id="p4-bank-w-capability-joint-imported-dev-v1"
  local status
  status=$(event_status "$evidence_id")
  if [[ "$status" == "dead" ]]; then
    printf '%s\n' "import queue refused dead joint event" >&2
    exit 9
  fi
  if [[ "$status" == "live" ]]; then
    preserve_registered_outputs "$evidence_id"
    printf '%s ALREADY_REGISTERED bank_w_joint_replay\n' \
      "$(date -u +%FT%TZ)" | tee -a "$QUEUE_LOG"
    return
  fi

  local present=0
  for path in "$JOINT_REPORT" "$JOINT_PNG" "$JOINT_PDF"; do
    [[ -f "$path" ]] && present=$((present + 1))
  done
  if (( present == 0 )); then
    python -m jspace_phase4.experiments.p4_bank_w_joint_imported \
      --config "$JOINT_CONFIG" --generate 2>&1 | tee -a "$QUEUE_LOG"
    bank_joint_outputs
  elif (( present != 3 )); then
    printf '%s\n' "import queue found a partial joint replay output set" >&2
    exit 10
  elif [[ -n "$(git status --porcelain)" ]]; then
    printf '%s\n' "import queue found unbanked joint replay outputs" >&2
    exit 11
  fi

  python -m jspace_phase4.experiments.p4_bank_w_joint_imported \
    --config "$JOINT_CONFIG" --register 2>&1 | tee -a "$QUEUE_LOG"
  bank_registry_event bank_w_joint_replay "$evidence_id"
}

register_import olmo_bank_w_capability \
  p4-import-olmo-bank-w-capability-v1 \
  "$EARLY_OLMO_BUNDLE" "$EARLY_OLMO_VALIDATION"

register_joint_replay

register_import gemma_transport_terminal \
  p4-import-gemma-transport-v1 \
  "$GEMMA_BUNDLE" "$GEMMA_VALIDATION"

register_import olmo_lineage_terminal \
  p4-import-olmo-lineage-final-v1 \
  "$FINAL_OLMO_BUNDLE" "$FINAL_OLMO_VALIDATION"

python - <<'PY'
from jspace_phase4.registry4 import read_events, resolve

required = [
    "p4-import-olmo-bank-w-capability-v1",
    "p4-bank-w-capability-joint-imported-dev-v1",
    "p4-import-gemma-transport-v1",
    "p4-import-olmo-lineage-final-v1",
]
for evidence_id in required:
    if not resolve(evidence_id)["live"]:
        raise SystemExit(f"queue did not leave live evidence: {evidence_id}")
if any(str(row.get("evidence_id", "")).startswith(("ol-", "gm-"))
       for row in read_events()):
    raise SystemExit("native side-track ID appeared in Phase 4 registry")
print("all methods-only side admissions and joint replay are live")
PY

printf '%s QUEUE_COMPLETE methods_only=true native_side_events=false\n' \
  "$(date -u +%FT%TZ)" | tee -a "$QUEUE_LOG"
