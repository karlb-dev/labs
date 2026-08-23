#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lab_dir="$(cd "$script_dir/.." && pwd)"
cd "$lab_dir"

backup=""
external_sql=0
skip_dependencies=0
while (($#)); do
  case "$1" in
    --backup) [[ $# -ge 2 ]] || exit 2; backup="$2"; shift 2 ;;
    --external-sql) external_sql=1; shift ;;
    --skip-dependencies) skip_dependencies=1; shift ;;
    *) echo "Usage: $0 [--backup FILE] [--external-sql] [--skip-dependencies]" >&2; exit 2 ;;
  esac
done

for command_name in node npm python3 openssl; do command -v "$command_name" >/dev/null || { echo "Missing prerequisite: $command_name" >&2; exit 1; }; done
if [[ ! -f .env ]]; then
  generated_password="ModelPrint!$(openssl rand -hex 18)"
  sed "s/replace-with-a-generated-strong-password/$generated_password/" .env.example >.env
  chmod 0600 .env
  echo "Created .env with a generated SQL password."
fi
# shellcheck disable=SC1091
source "$script_dir/runtime-env.sh"

if ((skip_dependencies == 0)); then
  npm ci
  python3 -m venv --clear .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/pip install -r requirements.lock
fi

if ((external_sql == 0)); then
  command -v docker >/dev/null || { echo "Docker is required unless --external-sql is used." >&2; exit 1; }
  docker compose version >/dev/null
  architecture="$(uname -m)"
  if [[ "$architecture" != "x86_64" && "$architecture" != "amd64" ]]; then
    if [[ "${MODELPRINT_ALLOW_EMULATED_SQL:-0}" == "1" ]] \
      && [[ "$(docker run --rm --platform linux/amd64 alpine uname -m 2>/dev/null)" == "x86_64" ]]; then
      echo "WARNING: running the x86-64 SQL Server image under emulation on $architecture." >&2
      echo "Microsoft does not support this host; continuing per MODELPRINT_ALLOW_EMULATED_SQL=1" >&2
      echo "(deviation recorded in EXPERIMENT_LOG.md; Rosetta viability evidenced by Lab 03)." >&2
      export DOCKER_DEFAULT_PLATFORM=linux/amd64
    else
      echo "SQL Server 2025 containers are x86-64 only and Microsoft does not support emulation on $architecture." >&2
      echo "Use --external-sql with an x86-64 Linux SQL Server host; see handoff_lab02_to_cpu.md," >&2
      echo "or set MODELPRINT_ALLOW_EMULATED_SQL=1 to proceed under a working amd64 emulator." >&2
      exit 4
    fi
  fi
  docker compose up --detach sqlserver
  sql_container="$(docker compose ps -q sqlserver)"
  [[ -n "$sql_container" ]] || { echo "SQL Server container did not start." >&2; exit 3; }
  for _ in $(seq 1 120); do
    if docker exec "$sql_container" /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C -Q "SELECT 1" -b -o /dev/null 2>/dev/null; then break; fi
    sleep 2
  done
  docker exec "$sql_container" /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C -Q "SELECT @@VERSION" -W
fi

npm run check
if [[ -n "$backup" ]]; then
  ((external_sql == 0)) || { echo "Native backup restore requires local Docker; import the BACPAC or restore on the external x86-64 host." >&2; exit 4; }
  npm run db:restore-handoff -- --backup "$backup"
else
  echo "Dependencies are ready. Restore a native backup with npm run db:restore-handoff -- --backup FILE."
fi
