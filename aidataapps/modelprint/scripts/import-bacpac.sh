#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lab_dir="$(cd "$script_dir/.." && pwd)"
cd "$lab_dir"
# shellcheck disable=SC1091
source "$script_dir/runtime-env.sh"
# shellcheck disable=SC1091
source "$script_dir/portable-sha256.sh"

bacpac=""
expected_sha=""
target="${MSSQL_DATABASE:-ModelPrint}"
while (($#)); do
  case "$1" in
    --bacpac) [[ $# -ge 2 ]] || exit 2; bacpac="$2"; shift 2 ;;
    --sha256) [[ $# -ge 2 ]] || exit 2; expected_sha="$2"; shift 2 ;;
    --target) [[ $# -ge 2 ]] || exit 2; target="$2"; shift 2 ;;
    *) echo "Usage: $0 --bacpac FILE [--sha256 HASH] [--target ModelPrint]" >&2; exit 2 ;;
  esac
done
[[ -n "$bacpac" && -f "$bacpac" ]] || { echo "A readable --bacpac FILE is required." >&2; exit 2; }
[[ "$target" =~ ^[A-Za-z][A-Za-z0-9_]{0,63}$ ]] || { echo "Unsafe target database: $target" >&2; exit 2; }
bacpac="$(cd "$(dirname "$bacpac")" && pwd)/$(basename "$bacpac")"
actual_sha="$(sha256_file "$bacpac")"
if [[ -z "$expected_sha" && -f "$bacpac.sha256" ]]; then expected_sha="$(awk 'NR==1 {print $1}' "$bacpac.sha256")"; fi
[[ -n "$expected_sha" ]] || { echo "Provide --sha256 or place the verified sidecar at $bacpac.sha256" >&2; exit 2; }
[[ "$actual_sha" == "$expected_sha" ]] || { echo "BACPAC digest mismatch: expected $expected_sha, got $actual_sha" >&2; exit 2; }

"$script_dir/install-sqlpackage.sh" >/dev/null
"$lab_dir/tools/sqlpackage/sqlpackage" /Action:Import /SourceFile:"$bacpac" \
  /TargetServerName:"${SQLSERVER_HOST:-127.0.0.1},${SQLSERVER_PORT:-1433}" /TargetDatabaseName:"$target" \
  /TargetUser:sa /TargetPassword:"$MSSQL_SA_PASSWORD" /TargetEncryptConnection:False /TargetTrustServerCertificate:True \
  /p:CommandTimeout=3600 /p:LongRunningCommandTimeout=0
MSSQL_DATABASE="$target" npm run db:prepare-handoff
MSSQL_DATABASE="$target" MODELPRINT_GIT_HEAD="$(git rev-parse HEAD)" npm run db:verify-handoff
