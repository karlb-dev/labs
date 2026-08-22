#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lab_dir="$(cd "$script_dir/.." && pwd)"
cd "$lab_dir"
# shellcheck disable=SC1091
source "$script_dir/runtime-env.sh"

mode="${1:-backup}"
run_dir="$(node --import tsx -e 'import {resolveRunDirectory} from "./src/run.ts"; console.log(resolveRunDirectory())')"
database_dir="$run_dir/database"
mkdir -p "$database_dir"
sql_container="${SQL_CONTAINER_NAME:-aidataapps-rag-sqlserver-1}"

if [[ "$mode" == "backup" ]]; then
  result="$(node --import tsx scripts/database-backup.ts)"
  container_path="$(jq -r '.containerPath' <<<"$result")"
  file_name="$(jq -r '.fileName' <<<"$result")"
  docker cp "$sql_container:$container_path" "$database_dir/$file_name"
  sha256sum "$database_dir/$file_name" >"$database_dir/$file_name.sha256"
  jq -n --arg mode backup --arg file "$file_name" --arg sha "$(sha256sum "$database_dir/$file_name" | awk '{print $1}')" \
    --arg createdAt "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '{schemaVersion:1,mode:$mode,file:$file,sha256:$sha,createdAt:$createdAt,verified:true}' \
    >"$database_dir/$file_name.json"
  echo "$database_dir/$file_name"
elif [[ "$mode" == "bacpac" ]]; then
  "$script_dir/install-sqlpackage.sh" >/dev/null
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  target="$database_dir/ModelPrint-$stamp.bacpac"
  connection="Server=${SQLSERVER_HOST:-127.0.0.1},${SQLSERVER_PORT:-1433};Initial Catalog=${MSSQL_DATABASE:-ModelPrint};User ID=sa;Password=${MSSQL_SA_PASSWORD};Encrypt=False;TrustServerCertificate=True;Connection Timeout=60"
  "$lab_dir/tools/sqlpackage/sqlpackage" /Action:Export /SourceConnectionString:"$connection" /TargetFile:"$target" /p:CommandTimeout=3600 /p:LongRunningCommandTimeout=0
  sha256sum "$target" >"$target.sha256"
  jq -n --arg mode bacpac --arg file "$(basename "$target")" --arg sha "$(sha256sum "$target" | awk '{print $1}')" \
    --arg version "$("$lab_dir/tools/sqlpackage/sqlpackage" /Version | tail -n1)" --arg createdAt "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{schemaVersion:1,mode:$mode,file:$file,sha256:$sha,sqlpackageVersion:$version,createdAt:$createdAt}' >"$target.json"
  echo "$target"
else
  echo "Usage: $0 backup|bacpac" >&2
  exit 2
fi
