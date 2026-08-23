#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
mode="rows"
requested_run=""
while (($#)); do
  case "$1" in
    --mode) mode="${2:?--mode requires rows or restore}"; shift 2 ;;
    --run) requested_run="${2:?--run requires a run directory}"; shift 2 ;;
    *) echo "Usage: $0 --mode rows|restore [--run runs/<run-id>]" >&2; exit 2 ;;
  esac
done
if [[ "$mode" != "rows" && "$mode" != "restore" ]]; then
  echo "Unsupported reproduction mode: $mode" >&2
  exit 2
fi

if [[ -z "$requested_run" ]]; then
  if [[ ! -s .current-run ]]; then
    echo "Tier 1 is not frozen: no current run is recorded." >&2
    exit 3
  fi
  requested_run="$(<.current-run)"
fi
if [[ "$requested_run" == /* ]]; then
  run_dir="$(realpath "$requested_run")"
else
  run_dir="$(realpath "$script_dir/$requested_run")"
fi
case "$run_dir" in
  "$script_dir"/runs/*) ;;
  *) echo "Refusing run outside $script_dir/runs: $run_dir" >&2; exit 4 ;;
esac
if [[ ! -f "$run_dir/manifests/freeze.json" || ! -f "$run_dir/repro/rows/manifest.json" || ! -f "$run_dir/repro/expected-report-outputs.json" ]]; then
  echo "Tier 1 row/freeze inputs are incomplete under $run_dir." >&2
  exit 3
fi

npm ci
npm run check
node --import tsx scripts/verify-report-row-bundles.ts --run "$run_dir" --expected "$run_dir/repro/rows"

if [[ "$mode" == "rows" ]]; then
  ./scripts/python.sh analysis/build_reports.py --run "$run_dir" --rows-dir "$run_dir/repro/rows"
  node --import tsx scripts/record-repro-pass.ts --run "$run_dir" --mode rows
  echo "LogWarden row-only reconstruction: PASS"
  exit 0
fi

if [[ ! -f "$run_dir/database/backup-receipt.json" ]]; then
  echo "Restore mode requires $run_dir/database/backup-receipt.json." >&2
  exit 3
fi

# shellcheck disable=SC1091
source scripts/runtime-env.sh
export COMPOSE_FILE="${COMPOSE_FILE:-compose.yaml}:compose.repro.yaml"
docker info >/dev/null
repro_port="$(node --input-type=module -e 'import net from "node:net"; const s=net.createServer(); s.listen(0,"127.0.0.1",()=>{console.log(s.address().port);s.close();});')"
repro_stamp="$(date -u +%H%M%S)-$$"
export COMPOSE_PROJECT_NAME="lwrepro-$repro_stamp"
export SQLSERVER_PORT="$repro_port"
export SQLSERVER_INTERNAL_PORT="$repro_port"
export CONTROL_DATABASE="LWReproControl$$"
export WORKLOAD_DATABASE="LWReproWorkload$$"
export LOGWARDEN_REPRO_SQL_CONTAINER="${COMPOSE_PROJECT_NAME}-sqlserver-1"
export LOGWARDEN_REPRO_BACKUPS_DIR="$run_dir/repro/restore-backups"
if ! [[ "$COMPOSE_PROJECT_NAME" =~ ^lwrepro-[0-9-]+$ && "$LOGWARDEN_REPRO_SQL_CONTAINER" == "$COMPOSE_PROJECT_NAME-sqlserver-1" ]]; then
  echo "Unsafe isolated Compose identity: $COMPOSE_PROJECT_NAME" >&2
  exit 5
fi

cleanup_restore() {
  local status=$?
  local container_pid=""
  local candidate_pid=""
  local candidate_ppid=""
  local removal_pid=""
  trap - EXIT INT TERM
  set +e
  if ((status != 0)); then timeout 10 docker compose logs --no-color --tail=160 sqlserver >&2; fi

  is_repro_sql_pid() {
    local pid="$1"
    local command_line=""
    [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/environ" && -r "/proc/$pid/cmdline" ]] || return 1
    grep -zFxq "MSSQL_TCP_PORT=$SQLSERVER_INTERNAL_PORT" "/proc/$pid/environ" || return 1
    command_line="$(tr '\0' ' ' <"/proc/$pid/cmdline")"
    [[ "$command_line" == /opt/mssql/bin/sqlservr* ||
       "$command_line" == /bin/bash\ /opt/mssql/bin/launch_sqlservr.sh* ]]
  }

  kill_repro_sql_pids() {
    local environ_path=""
    local pid=""
    for environ_path in /proc/[0-9]*/environ; do
      pid="${environ_path#/proc/}"
      pid="${pid%/environ}"
      if is_repro_sql_pid "$pid"; then kill -KILL "$pid" 2>/dev/null; fi
    done
  }

  container_pid="$(timeout 5 docker inspect "$LOGWARDEN_REPRO_SQL_CONTAINER" --format '{{.State.Pid}}' 2>/dev/null)"
  if ! is_repro_sql_pid "$container_pid"; then
    container_pid=""
    for candidate_environ in /proc/[0-9]*/environ; do
      candidate_pid="${candidate_environ#/proc/}"
      candidate_pid="${candidate_pid%/environ}"
      is_repro_sql_pid "$candidate_pid" || continue
      candidate_ppid="$(awk '{print $4}' "/proc/$candidate_pid/stat" 2>/dev/null)"
      if ! is_repro_sql_pid "$candidate_ppid"; then
        container_pid="$candidate_pid"
        break
      fi
    done
  fi
  if is_repro_sql_pid "$container_pid"; then
    echo "Stopping isolated SQL PID $container_pid on port $SQLSERVER_INTERNAL_PORT" >&2
    kill -TERM "$container_pid" 2>/dev/null
    for _attempt in $(seq 1 10); do
      kill -0 "$container_pid" 2>/dev/null || break
      sleep 0.2
    done
    if kill -0 "$container_pid" 2>/dev/null; then kill -KILL "$container_pid" 2>/dev/null; fi
  fi
  timeout 20 docker rm -f "$LOGWARDEN_REPRO_SQL_CONTAINER" >/dev/null 2>&1 &
  removal_pid=$!
  for _attempt in $(seq 1 80); do
    kill_repro_sql_pids
    kill -0 "$removal_pid" 2>/dev/null || break
    sleep 0.25
  done
  wait "$removal_pid" 2>/dev/null
  timeout 20 docker compose down --volumes --remove-orphans >/dev/null 2>&1
  timeout 10 docker volume rm \
    "${COMPOSE_PROJECT_NAME}-run-exports" \
    "${COMPOSE_PROJECT_NAME}-sqlserver-data" >/dev/null 2>&1
  timeout 10 docker network rm "${COMPOSE_PROJECT_NAME}_default" >/dev/null 2>&1
  node --import tsx scripts/stage-repro-backups.ts --run "$run_dir" --cleanup >/dev/null 2>&1
  exit "$status"
}
trap cleanup_restore EXIT INT TERM

node --import tsx scripts/stage-repro-backups.ts --run "$run_dir" --receipt "$run_dir/database/backup-receipt.json"
docker compose up -d --wait --wait-timeout 240 sqlserver
node --import tsx scripts/restore-report-backups.ts --run "$run_dir" --receipt "$run_dir/database/backup-receipt.json"
node --import tsx scripts/export-report-rows.ts --run "$run_dir" --control-database "$CONTROL_DATABASE" --out "$run_dir/repro/restore-rows" --clean
node --import tsx scripts/verify-report-row-bundles.ts --run "$run_dir" --expected "$run_dir/repro/rows" --actual "$run_dir/repro/restore-rows"
./scripts/python.sh analysis/build_reports.py --run "$run_dir" --rows-dir "$run_dir/repro/restore-rows"
node --import tsx scripts/record-repro-pass.ts --run "$run_dir" --mode restore
echo "LogWarden fresh-SQL restore reconstruction: PASS"
