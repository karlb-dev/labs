#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
# shellcheck disable=SC1091
source scripts/runtime-env.sh
# shellcheck disable=SC1091
source scripts/container-storage.sh
# shellcheck disable=SC1091
source scripts/portable-sha256.sh
run_dir="$(node --import tsx -e 'import {resolveRunDirectory} from "./src/run.ts"; console.log(resolveRunDirectory())')"
run_id="$(basename "$run_dir")"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
inventory="$run_dir/ARTIFACT_INVENTORY.json"
[[ -f "$inventory" ]] || { echo "Missing $inventory; run npm run run:archive" >&2; exit 2; }
while IFS=$'\t' read -r relative expected; do
  [[ "$relative" != /* && "$relative" != *".."* ]] || { echo "Unsafe inventory path: $relative" >&2; exit 2; }
  actual="$(sha256_file "$run_dir/$relative")"
  [[ "$actual" == "$expected" ]] || { echo "Digest mismatch: $relative" >&2; exit 2; }
done < <(jq -r '.artifacts[] | [.path,.sha256] | @tsv' "$inventory")
./scripts/python.sh analysis/verify_predictions.py --run "$run_dir"
local_backup="$(ls -1t "$run_dir"/database/*.bak 2>/dev/null | head -1 || true)"
mirror_root="${MODELPRINT_DRIVE_MIRROR:-/content/drive/MyDrive/aidataapps/lab02/runs}"
mirror_run="$mirror_root/$run_id"
mirror_backup=""
if [[ -d "$mirror_run/database" ]]; then
  mirror_backup="$(ls -1t "$mirror_run"/database/*.bak 2>/dev/null | head -1 || true)"
fi
backup="$local_backup"; backup_source="local-run"; backup_sha=""
if [[ -n "$mirror_backup" && -f "$mirror_backup.sha256" ]]; then
  mirror_expected="$(awk 'NR==1 {print $1}' "$mirror_backup.sha256")"
  mirror_actual="$(sha256_file "$mirror_backup")"
  if [[ "$mirror_actual" == "$mirror_expected" ]]; then
    backup="$mirror_backup"; backup_source="drive-mirror"; backup_sha="$mirror_actual"
  fi
fi
[[ -n "$backup" && -f "$backup" ]] || { echo "No retained SQL backup" >&2; exit 2; }
[[ -n "$backup_sha" ]] || backup_sha="$(sha256_file "$backup")"
if [[ -f "$backup.sha256" ]]; then
  expected_sha="$(awk 'NR==1 {print $1}' "$backup.sha256")"
  [[ "$backup_sha" == "$expected_sha" ]] || { echo "Backup digest mismatch: $backup" >&2; exit 2; }
fi
sql_container="${SQL_CONTAINER_NAME:-aidataapps-rag-sqlserver-1}";token="${backup_sha:0:12}";target="ModelPrintRepro_${token}_$$";container_backup="/var/opt/mssql/data/${target}.bak"
copy_to_container_file "$backup" "$sql_container" "$container_backup"
cleanup() { MSSQL_DATABASE=master node --import tsx scripts/restore-database.ts --target "$target" --drop >/dev/null 2>&1 || true; remove_container_file "$sql_container" "$container_backup" >/dev/null 2>&1 || true; }
trap cleanup EXIT
node --import tsx scripts/restore-database.ts --target "$target" --backup "$container_backup"
expected_headline="$(mktemp)";cp "$run_dir/reports/HEADLINE.md" "$expected_headline"
MSSQL_DATABASE="$target" ./scripts/python.sh analysis/build_reports.py --run "$run_dir"
diff -u "$expected_headline" "$run_dir/reports/HEADLINE.md"
rm -f "$expected_headline"
jq -n --arg runId "$run_id" --arg startedAt "$started_at" --arg completedAt "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg gitHead "$(git rev-parse HEAD)" --arg inventorySha256 "$(sha256_file "$inventory")" \
  --arg backupSource "$backup_source" --arg backupFile "$(basename "$backup")" --arg backupSha256 "$backup_sha" \
  --arg headlineSha256 "$(sha256_file "$run_dir/reports/HEADLINE.md")" \
  '{schemaVersion:1,runId:$runId,status:"PASS",startedAt:$startedAt,completedAt:$completedAt,gitHead:$gitHead,inventorySha256:$inventorySha256,backupSource:$backupSource,backupFile:$backupFile,backupSha256:$backupSha256,predictionReconstruction:"PASS",databaseRestore:"PASS",headlineByteDiff:"PASS",headlineSha256:$headlineSha256}' \
  >"$run_dir/metrics/reproduction.json"
echo "Reproduction PASS: manifests, prediction metrics, restored SQL report, and headline diff."
