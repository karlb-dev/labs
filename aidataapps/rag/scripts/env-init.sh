#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lab_dir="$(cd "$script_dir/.." && pwd)"
cd "$lab_dir"

model_profile="qwen-smoke"
skip_pull=0
while (($#)); do
  case "$1" in
    --model)
      model_profile="${2:?--model requires a profile key}"
      shift 2
      ;;
    --skip-pull)
      skip_pull=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: ./scripts/env-init.sh [--model PROFILE] [--skip-pull]" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f .env ]]; then
  generated_password="RagLab!$(openssl rand -hex 18)"
  sed "s/replace-with-a-generated-strong-password/$generated_password/" .env.example >.env
  chmod 600 .env
  echo "Created .env with a generated SQL Server password."
fi

# shellcheck disable=SC1091
source "$script_dir/runtime-env.sh"
export MODEL_PROFILE="$model_profile"

for command_name in docker curl npm openssl nvidia-smi; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing prerequisite: $command_name" >&2
    exit 1
  fi
done
docker compose version >/dev/null
docker info >/dev/null
if ! nvidia-smi >/dev/null; then
  echo "NVIDIA driver is not ready; local vLLM cannot start." >&2
  exit 1
fi

npm ci
docker compose config --quiet
if ((skip_pull == 0)); then
  docker compose pull sqlserver vllm-embedding
fi
docker compose up --detach sqlserver vllm-embedding

echo "Waiting for SQL Server 2025..."
for _ in $(seq 1 120); do
  if docker compose exec -T sqlserver /opt/mssql-tools18/bin/sqlcmd \
    -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C -Q "SELECT 1" -b -o /dev/null 2>/dev/null; then
    break
  fi
  sleep 2
done
docker compose exec -T sqlserver /opt/mssql-tools18/bin/sqlcmd \
  -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C \
  -Q "SELECT @@VERSION AS version" -W

echo "Waiting for the embedding model (the first download can take several minutes)..."
for _ in $(seq 1 600); do
  if curl --fail --silent "${EMBEDDING_BASE_URL%/v1}/health" >/dev/null; then
    break
  fi
  sleep 2
done
curl --fail --silent "${EMBEDDING_BASE_URL%/v1}/health" >/dev/null

npm run model -- start --profile "$MODEL_PROFILE" --replace
echo "Waiting for $MODEL_PROFILE (the first model download can take several minutes)..."
for _ in $(seq 1 900); do
  if curl --fail --silent "${CHAT_BASE_URL%/v1}/health" >/dev/null; then
    break
  fi
  sleep 2
done
curl --fail --silent "${CHAT_BASE_URL%/v1}/health" >/dev/null

npm run db:setup
npm run check

npm run start &
app_pid=$!
cleanup() {
  kill "$app_pid" 2>/dev/null || true
  wait "$app_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 90); do
  if curl --fail --silent "http://${HOST:-127.0.0.1}:${PORT:-3000}/ready" >/dev/null; then
    break
  fi
  sleep 1
done
curl --fail --silent "http://${HOST:-127.0.0.1}:${PORT:-3000}/ready" >/dev/null
npm run smoke

echo "Environment is ready. Dependencies remain running; use ./scripts/env-down.sh when finished."
