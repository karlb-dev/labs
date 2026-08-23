#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
# shellcheck disable=SC1091
source scripts/runtime-env.sh
# shellcheck disable=SC1091
source scripts/container-storage.sh
run_dir="$(node --import tsx -e 'import {resolveRunDirectory} from "./src/run.ts"; console.log(resolveRunDirectory())')"
inventory="$run_dir/ARTIFACT_INVENTORY.json"
[[ -f "$inventory" ]] || { echo "Missing $inventory; run npm run run:archive" >&2; exit 2; }
while IFS=$'\t' read -r relative expected; do
  [[ "$relative" != /* && "$relative" != *".."* ]] || { echo "Unsafe inventory path: $relative" >&2; exit 2; }
  actual="$(sha256sum "$run_dir/$relative" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || { echo "Digest mismatch: $relative" >&2; exit 2; }
done < <(jq -r '.artifacts[] | [.path,.sha256] | @tsv' "$inventory")
./scripts/python.sh analysis/verify_predictions.py --run "$run_dir"
backup="$(find "$run_dir/database" -maxdepth 1 -type f -name '*.bak' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
[[ -n "$backup" && -f "$backup" ]] || { echo "No retained SQL backup" >&2; exit 2; }
[[ -f "$backup.sha256" ]] && (cd "$(dirname "$backup")" && sha256sum -c "$(basename "$backup").sha256")
sql_container="${SQL_CONTAINER_NAME:-aidataapps-rag-sqlserver-1}";token="$(sha256sum "$backup" | cut -c1-12)";target="ModelPrintRepro_${token}_$$";container_backup="/var/opt/mssql/data/${target}.bak"
copy_to_container_file "$backup" "$sql_container" "$container_backup"
cleanup() { MSSQL_DATABASE=master node --import tsx scripts/restore-database.ts --target "$target" --drop >/dev/null 2>&1 || true; remove_container_file "$sql_container" "$container_backup" >/dev/null 2>&1 || true; }
trap cleanup EXIT
node --import tsx scripts/restore-database.ts --target "$target" --backup "$container_backup"
expected_headline="$(mktemp)";cp "$run_dir/reports/HEADLINE.md" "$expected_headline"
MSSQL_DATABASE="$target" ./scripts/python.sh analysis/build_reports.py --run "$run_dir"
diff -u "$expected_headline" "$run_dir/reports/HEADLINE.md"
rm -f "$expected_headline"
echo "Reproduction PASS: manifests, prediction metrics, restored SQL report, and headline diff."
