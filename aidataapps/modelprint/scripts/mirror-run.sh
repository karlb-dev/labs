#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lab_dir="$(cd "$script_dir/.." && pwd)"
cd "$lab_dir"
# shellcheck disable=SC1091
source "$script_dir/runtime-env.sh"
run_dir="$(node --import tsx -e 'import {resolveRunDirectory} from "./src/run.ts"; console.log(resolveRunDirectory())')"
run_id="$(basename "$run_dir")"
mirror_root="${MODELPRINT_DRIVE_MIRROR:-/content/drive/MyDrive/aidataapps/lab02/runs}"
target="$mirror_root/$run_id"
if [[ ! -d /content/drive/MyDrive ]]; then
  echo "Drive mirror is not mounted at /content/drive/MyDrive" >&2
  exit 3
fi
node --import tsx scripts/archive-run.ts >/dev/null
mkdir -p "$target"
rsync -a --partial --human-readable "$run_dir/" "$target/"
jq -n --arg runId "$run_id" --arg source "$run_dir" --arg target "$target" --arg mirroredAt "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{schemaVersion:1,runId:$runId,source:$source,target:$target,mirroredAt:$mirroredAt}' >"$run_dir/MIRROR_RECEIPT.json"
node --import tsx scripts/archive-run.ts >/dev/null
rsync -a "$run_dir/ARTIFACT_INVENTORY.json" "$run_dir/MIRROR_RECEIPT.json" "$run_dir/RESUME.md" "$target/"
echo "$target"
