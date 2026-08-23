#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lab_dir="$(cd "$script_dir/.." && pwd)"
cd "$lab_dir"
# shellcheck disable=SC1091
source "$script_dir/runtime-env.sh"
# shellcheck disable=SC1091
source "$script_dir/container-storage.sh"
# shellcheck disable=SC1091
source "$script_dir/portable-sha256.sh"

backup=""
expected_sha=""
target="${MSSQL_DATABASE:-ModelPrint}"
while (($#)); do
  case "$1" in
    --backup) [[ $# -ge 2 ]] || exit 2; backup="$2"; shift 2 ;;
    --sha256) [[ $# -ge 2 ]] || exit 2; expected_sha="$2"; shift 2 ;;
    --target) [[ $# -ge 2 ]] || exit 2; target="$2"; shift 2 ;;
    *) echo "Usage: $0 --backup FILE [--sha256 HASH] [--target ModelPrint]" >&2; exit 2 ;;
  esac
done
[[ -n "$backup" && -f "$backup" ]] || { echo "A readable --backup FILE is required." >&2; exit 2; }
[[ "$target" == "ModelPrint" || "$target" =~ ^ModelPrintRepro_[A-Za-z0-9_]+$ ]] || { echo "Unsafe target database: $target" >&2; exit 2; }
backup="$(cd "$(dirname "$backup")" && pwd)/$(basename "$backup")"
actual_sha="$(sha256_file "$backup")"
if [[ -z "$expected_sha" && -f "$backup.sha256" ]]; then expected_sha="$(awk 'NR==1 {print $1}' "$backup.sha256")"; fi
[[ -n "$expected_sha" ]] || { echo "Provide --sha256 or place the verified sidecar at $backup.sha256" >&2; exit 2; }
[[ "$actual_sha" == "$expected_sha" ]] || { echo "Backup digest mismatch: expected $expected_sha, got $actual_sha" >&2; exit 2; }

sql_container="${SQL_CONTAINER_NAME:-}"
if [[ -z "$sql_container" ]]; then sql_container="$(docker compose ps -q sqlserver)"; fi
[[ -n "$sql_container" ]] || { echo "SQL Server container is not running; start it with docker compose up -d sqlserver." >&2; exit 3; }
docker inspect "$sql_container" >/dev/null
container_backup="/var/opt/mssql/data/modelprint-handoff-${actual_sha:0:16}.bak"
copy_to_container_file "$backup" "$sql_container" "$container_backup"
cleanup() { remove_container_file "$sql_container" "$container_backup" >/dev/null 2>&1 || true; }
trap cleanup EXIT
node --import tsx scripts/restore-database.ts --target "$target" --backup "$container_backup"
MSSQL_DATABASE="$target" MODELPRINT_GIT_HEAD="$(git rev-parse HEAD)" npm run db:verify-handoff
