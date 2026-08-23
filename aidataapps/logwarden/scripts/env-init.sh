#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lab_dir="$(cd "$script_dir/.." && pwd)"
cd "$lab_dir"

skip_build=0
with_embedding=0
while (($#)); do
  case "$1" in
    --skip-build) skip_build=1; shift ;;
    --with-embedding) with_embedding=1; shift ;;
    *)
      echo "Usage: $0 [--skip-build] [--with-embedding]" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f .env ]]; then
  echo "Missing .env; run ./scripts/colab-host-init.sh (Colab) or ./scripts/mac-host-init.sh (macOS) first." >&2
  exit 2
fi

# shellcheck disable=SC1091
source "$script_dir/runtime-env.sh"

required_commands=(docker curl npm)
if [[ "${CONTAINER_RUNTIME_PROFILE:-}" != "mac-docker-desktop" ]]; then
  required_commands+=(nvidia-smi)
fi
for command_name in "${required_commands[@]}"; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing prerequisite: $command_name" >&2
    exit 1
  fi
done
docker compose version >/dev/null
docker info >/dev/null
if [[ "${CONTAINER_RUNTIME_PROFILE:-}" != "mac-docker-desktop" ]]; then
  nvidia-smi >/dev/null
fi

# macOS has no GNU timeout; degrade to an unbounded call rather than failing.
run_with_deadline() {
  local seconds="$1"
  shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "${seconds}s" "$@"
  elif command -v gtimeout >/dev/null 2>&1; then
    gtimeout "${seconds}s" "$@"
  else
    "$@"
  fi
}

npm ci
docker compose config --quiet
if ((skip_build == 0)); then
  if [[ "${CONTAINER_RUNTIME_PROFILE:-}" == "colab-rootless" ]]; then
    "$script_dir/build-mssql-fts-colab.sh"
  else
    docker compose build --pull sqlserver
  fi
fi
docker compose up --detach sqlserver

echo "Waiting for isolated SQL Server 2025 on port ${SQLSERVER_PORT}..."
ready=0
for _ in $(seq 1 120); do
  if run_with_deadline 15 docker compose exec -T sqlserver /bin/bash -lc \
    'SQLCMDPASSWORD="$MSSQL_SA_PASSWORD" /opt/mssql-tools18/bin/sqlcmd -S "localhost,$MSSQL_TCP_PORT" -U sa -C -Q "SELECT 1" -b -o /dev/null' \
    2>/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
if ((ready == 0)); then
  echo "SQL Server did not become ready; inspect docker compose logs sqlserver." >&2
  exit 3
fi
run_with_deadline 30 docker compose exec -T sqlserver /bin/bash -lc \
  'SQLCMDPASSWORD="$MSSQL_SA_PASSWORD" /opt/mssql-tools18/bin/sqlcmd -S "localhost,$MSSQL_TCP_PORT" -U sa -C -Q "SELECT @@VERSION AS version, FULLTEXTSERVICEPROPERTY('"'"'IsFullTextInstalled'"'"') AS fulltext_installed" -W -b'

if ((with_embedding == 1)); then
  if [[ "${CONTAINER_RUNTIME_PROFILE:-}" == "mac-docker-desktop" ]]; then
    echo "--with-embedding uses the CUDA vLLM image and cannot run on the mac profile." >&2
    echo "Serve embeddings with Foundry Local instead: npm run mac:model -- up --profile qwen3-embedding-0.6b-foundry" >&2
    exit 2
  fi
  docker compose --profile embedding pull embedding-qwen
  docker compose --profile embedding up --detach embedding-qwen
  echo "Waiting for the Qwen embedding endpoint..."
  for _ in $(seq 1 600); do
    if curl --fail --silent "${EMBEDDING_BASE_URL%/v1}/health" >/dev/null; then
      break
    fi
    sleep 2
  done
  curl --fail --silent "${EMBEDDING_BASE_URL%/v1}/health" >/dev/null
fi

echo "Foundation environment is ready. Run run:init, db:setup, doctor, then check."
