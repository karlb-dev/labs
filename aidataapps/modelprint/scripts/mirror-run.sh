#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lab_dir="$(cd "$script_dir/.." && pwd)"
cd "$lab_dir"
# shellcheck disable=SC1091
source "$script_dir/runtime-env.sh"
# shellcheck disable=SC1091
source "$script_dir/portable-sha256.sh"
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
# A BACPAC is written directly to its final local name while SqlPackage is
# still exporting it. Never let a concurrent mirror publish that growing file.
rsync -a --partial --human-readable --exclude='database/*.bacpac' "$run_dir/" "$target/"
mkdir -p "$target/database"
shopt -s nullglob
for source_bacpac in "$run_dir"/database/*.bacpac; do
  sidecar="$source_bacpac.sha256";metadata="$source_bacpac.json"
  # Absence of either finalization sidecar means the export is still in flight.
  [[ -f "$sidecar" && -f "$metadata" ]] || continue
  expected="$(awk 'NR==1 {print $1}' "$sidecar")"
  [[ "$expected" =~ ^[0-9a-f]{64}$ && "$(sha256_file "$source_bacpac")" == "$expected" ]] || {
    echo "Refusing to mirror unfinished or invalid BACPAC: $source_bacpac" >&2
    exit 2
  }
  base="$(basename "$source_bacpac")";destination="$target/database/$base"
  if [[ ! -f "$destination" || "$(stat -c %s "$destination")" != "$(stat -c %s "$source_bacpac")" || "$(sha256_file "$destination")" != "$expected" ]]; then
    upload="$target/database/$base.uploading-${expected:0:16}"
    rm -f -- "$upload"
    rsync -a --inplace "$source_bacpac" "$upload"
    [[ "$(stat -c %s "$upload")" == "$(stat -c %s "$source_bacpac")" && "$(sha256_file "$upload")" == "$expected" ]] || {
      echo "Drive BACPAC upload readback failed: $upload" >&2
      exit 2
    }
    mv -f -- "$upload" "$destination"
  fi
  rsync -a "$sidecar" "$metadata" "$target/database/"
  [[ "$(stat -c %s "$destination")" == "$(stat -c %s "$source_bacpac")" && "$(sha256_file "$destination")" == "$expected" ]] || {
    echo "Drive BACPAC promotion readback failed: $destination" >&2
    exit 2
  }
done
shopt -u nullglob
jq -n --arg runId "$run_id" --arg source "$run_dir" --arg target "$target" --arg mirroredAt "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  '{schemaVersion:1,runId:$runId,source:$source,target:$target,mirroredAt:$mirroredAt}' >"$run_dir/MIRROR_RECEIPT.json"
node --import tsx scripts/archive-run.ts >/dev/null
rsync -a "$run_dir/ARTIFACT_INVENTORY.json" "$run_dir/MIRROR_RECEIPT.json" "$run_dir/RESUME.md" "$target/"
echo "$target"
