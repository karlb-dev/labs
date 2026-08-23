#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lab_dir="$(cd "$script_dir/.." && pwd)"
cd "$lab_dir"
# shellcheck disable=SC1091
source "$script_dir/runtime-env.sh"

retention="${MODELPRINT_LOCAL_BACKUP_RETENTION:-3}"
if [[ ! "$retention" =~ ^[1-9][0-9]*$ ]]; then
  echo "MODELPRINT_LOCAL_BACKUP_RETENTION must be a positive integer" >&2
  exit 2
fi

run_dir="$(node --import tsx -e 'import {resolveRunDirectory} from "./src/run.ts"; console.log(resolveRunDirectory())')"
run_dir="$(realpath "$run_dir")"
case "$run_dir" in
  "$lab_dir"/runs/*) ;;
  *) echo "Refusing retention outside $lab_dir/runs: $run_dir" >&2; exit 3 ;;
esac
database_dir="$run_dir/database"
mirror_root="${MODELPRINT_DRIVE_MIRROR:-/content/drive/MyDrive/aidataapps/lab02/runs}"
drive_database="$mirror_root/$(basename "$run_dir")/database"
[[ -d "$database_dir" && -d "$drive_database" ]] || { echo "Local or Drive database directory is absent" >&2; exit 3; }

mapfile -t backups < <(find "$database_dir" -maxdepth 1 -type f -name '*.bak' -printf '%T@ %p\n' | sort -rn | cut -d' ' -f2-)
if ((${#backups[@]} <= retention)); then
  echo "No local backups eligible for pruning; retained=${#backups[@]}"
  exit 0
fi

removed=0
for ((index=retention; index<${#backups[@]}; index+=1)); do
  local_file="${backups[$index]}"
  case "$local_file" in
    "$database_dir"/*.bak) ;;
    *) echo "Skipping unexpected path: $local_file" >&2; continue ;;
  esac
  base="$(basename "$local_file")"
  [[ "$base" =~ ^modelprint-full-[0-9TZ-]+\.bak$ ]] || { echo "Skipping unexpected filename: $base" >&2; continue; }
  sidecar="$local_file.sha256"
  metadata="$local_file.json"
  drive_file="$drive_database/$base"
  [[ -f "$sidecar" && -f "$drive_file" ]] || { echo "Skipping unmirrored backup: $base" >&2; continue; }
  expected="$(awk 'NR==1 {print $1}' "$sidecar")"
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || { echo "Skipping invalid checksum sidecar: $base" >&2; continue; }
  drive_actual="$(sha256sum "$drive_file" | awk '{print $1}')"
  if [[ "$drive_actual" != "$expected" ]]; then
    echo "Skipping Drive checksum mismatch: $base" >&2
    continue
  fi
  rm -f -- "$local_file" "$sidecar" "$metadata"
  echo "Removed verified redundant local backup: $base (Drive SHA-256 $drive_actual)"
  removed=$((removed + 1))
done
echo "Local backup retention complete; removed=$removed retained_target=$retention"
