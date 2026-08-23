#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lab_dir="$(cd "$script_dir/.." && pwd)"
cd "$lab_dir"
# shellcheck disable=SC1091
source "$script_dir/runtime-env.sh"
source_database="${MSSQL_DATABASE:-ModelPrint}"
scratch="ModelPrintRepro_Schema_$$_$(date -u +%H%M%S)"
cleanup() { MSSQL_DATABASE=master node --import tsx scripts/restore-database.ts --target "$scratch" --drop >/dev/null 2>&1 || true; }
trap cleanup EXIT
MSSQL_DATABASE="$scratch" npm run db:setup
node --import tsx scripts/compare-database-schema.ts --left "$source_database" --right "$scratch"
