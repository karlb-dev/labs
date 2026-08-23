#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lab_dir="$(cd "$script_dir/.." && pwd)"
repo_dir="$(git -C "$lab_dir" rev-parse --show-toplevel)"
cd "$lab_dir"
# shellcheck disable=SC1091
source "$script_dir/runtime-env.sh"
# shellcheck disable=SC1091
source "$script_dir/portable-sha256.sh"

run_dir="$(node --import tsx -e 'import {resolveRunDirectory} from "./src/run.ts"; console.log(resolveRunDirectory())')"
run_id="$(basename "$run_dir")"
branch="$(git -C "$repo_dir" symbolic-ref --quiet --short HEAD)"
drive_root="${AIDATAAPPS_DRIVE_ROOT:-/content/drive/MyDrive/aidataapps}"
drive_run="${MODELPRINT_DRIVE_MIRROR:-$drive_root/lab02/runs}/$run_id"
drive_target="$drive_root/lab02/cpu-handoff/$run_id"
[[ -d /content/drive/MyDrive ]] || { echo "Drive is not mounted." >&2; exit 3; }

stage="$(mktemp -d /content/modelprint-cpu-handoff.XXXXXX)"
cleanup() { [[ -d "$stage" ]] && rm -rf -- "$stage"; }
trap cleanup EXIT

git -C "$repo_dir" bundle create "$stage/repository.bundle" "$branch"
git -C "$repo_dir" bundle verify "$stage/repository.bundle" >"$stage/repository-bundle-verify.txt" 2>&1
git -C "$repo_dir" status --short --branch >"$stage/git-status.txt"
git -C "$repo_dir" diff --binary HEAD -- aidataapps/modelprint aidataapps/resume.md aidataapps/inprogress_lab2.md aidataapps/handoff_lab02_to_cpu.md >"$stage/working-tree.patch"
untracked="$stage/untracked-source.nul"
git -C "$repo_dir" ls-files --others --exclude-standard -z -- aidataapps/modelprint aidataapps/resume.md aidataapps/inprogress_lab2.md aidataapps/handoff_lab02_to_cpu.md >"$untracked"
tar --null -czf "$stage/untracked-source.tar.gz" -C "$repo_dir" --files-from="$untracked"
rm -f -- "$untracked"

# Database exports and recovery snapshots are transferred separately. Everything
# needed by the remaining analyses, including raw governed output and atomic
# probe checkpoints, is retained in this compressed run-state archive.
tar -czf "$stage/$run_id-run-state.tar.gz" -C "$run_dir" --exclude='./database' --exclude='./checkpoints' .
printf '%s\n' "runs/$run_id" >"$stage/current-run.txt"
cp "$repo_dir/aidataapps/resume.md" "$stage/resume.md"
cp "$repo_dir/aidataapps/inprogress_lab2.md" "$stage/inprogress_lab2.md"
cp "$repo_dir/aidataapps/handoff_lab02_to_cpu.md" "$stage/handoff_lab02_to_cpu.md"
[[ ! -f /content/handoff.md ]] || cp /content/handoff.md "$stage/handoff-colab.md"

latest_backup="$(ls -1t "$run_dir"/database/*.bak 2>/dev/null | head -1 || true)"
latest_bacpac="$(ls -1t "$run_dir"/database/*.bacpac 2>/dev/null | head -1 || true)"
[[ -n "$latest_backup" && -f "$latest_backup" ]] || { echo "No native backup is available." >&2; exit 2; }
backup_sha="$(sha256_file "$latest_backup")"
if [[ -f "$latest_backup.sha256" ]]; then [[ "$backup_sha" == "$(awk 'NR==1 {print $1}' "$latest_backup.sha256")" ]] || { echo "Native backup sidecar mismatch." >&2; exit 2; }; fi
bacpac_name="";bacpac_sha="";bacpac_bytes=0
if [[ -n "$latest_bacpac" && -f "$latest_bacpac" ]]; then
  bacpac_name="$(basename "$latest_bacpac")";bacpac_sha="$(sha256_file "$latest_bacpac")";bacpac_bytes="$(stat -c %s "$latest_bacpac")"
  if [[ -f "$latest_bacpac.sha256" ]]; then [[ "$bacpac_sha" == "$(awk 'NR==1 {print $1}' "$latest_bacpac.sha256")" ]] || { echo "BACPAC sidecar mismatch." >&2; exit 2; }; fi
fi
jq -n \
  --arg runId "$run_id" --arg driveRun "$drive_run" \
  --arg backupFile "$(basename "$latest_backup")" --arg backupSha256 "$backup_sha" --argjson backupBytes "$(stat -c %s "$latest_backup")" \
  --arg bacpacFile "$bacpac_name" --arg bacpacSha256 "$bacpac_sha" --argjson bacpacBytes "$bacpac_bytes" \
  '{schemaVersion:1,runId:$runId,driveRun:$driveRun,nativeBackup:{file:$backupFile,sha256:$backupSha256,bytes:$backupBytes,required:true},bacpac:(if $bacpacFile=="" then null else {file:$bacpacFile,sha256:$bacpacSha256,bytes:$bacpacBytes,required:false} end)}' \
  >"$stage/DATABASE_FILES.json"

for file in "$stage"/*; do [[ -f "$file" && "$(basename "$file")" != "TRANSFER_SHA256.txt" ]] && printf '%s  %s\n' "$(sha256_file "$file")" "$(basename "$file")"; done | sort >"$stage/TRANSFER_SHA256.txt"
jq -n \
  --arg createdAt "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg runId "$run_id" --arg branch "$branch" --arg gitHead "$(git -C "$repo_dir" rev-parse HEAD)" \
  --arg driveTarget "$drive_target" --arg runArchive "$run_id-run-state.tar.gz" --arg runArchiveSha256 "$(sha256_file "$stage/$run_id-run-state.tar.gz")" \
  '{schemaVersion:1,createdAt:$createdAt,runId:$runId,branch:$branch,gitHead:$gitHead,driveTarget:$driveTarget,runState:{file:$runArchive,sha256:$runArchiveSha256,excludes:["database/","checkpoints/"]},excludedMachineState:[".env","node_modules/",".venv/","tools/","Docker volumes","Hugging Face model caches","vLLM caches"]}' \
  >"$stage/TRANSFER_MANIFEST.json"
# Refresh the checksum list after writing the manifest itself.
for file in "$stage"/*; do [[ -f "$file" && "$(basename "$file")" != "TRANSFER_SHA256.txt" ]] && printf '%s  %s\n' "$(sha256_file "$file")" "$(basename "$file")"; done | sort >"$stage/TRANSFER_SHA256.txt"

mkdir -p "$drive_target"
rsync -a --partial "$stage/" "$drive_target/"
while IFS= read -r line; do
  expected="${line%%  *}";name="${line#*  }";actual="$(sha256_file "$drive_target/$name")"
  [[ "$actual" == "$expected" ]] || { echo "Drive handoff mismatch: $name" >&2; exit 2; }
done <"$stage/TRANSFER_SHA256.txt"
echo "$drive_target"
