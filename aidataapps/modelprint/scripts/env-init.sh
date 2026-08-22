#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lab_dir="$(cd "$script_dir/.." && pwd)"
cd "$lab_dir"

skip_pull=0
skip_python=0
primary_only=0
while (($#)); do
  case "$1" in
    --skip-pull) skip_pull=1; shift ;;
    --skip-python) skip_python=1; shift ;;
    --primary-embedding-only) primary_only=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -f .env ]]; then
  generated_password="ModelPrint!$(openssl rand -hex 18)"
  sed "s/replace-with-a-generated-strong-password/$generated_password/" .env.example >.env
  if [[ -f ../rag/.env ]]; then
    inherited_token="$(sed -n 's/^HF_TOKEN=//p' ../rag/.env | head -n1)"
    if [[ -n "$inherited_token" ]]; then
      token_tmp="$(mktemp .env.token.XXXXXX)"
      awk -v token="$inherited_token" 'BEGIN{done=0} /^HF_TOKEN=/{print "HF_TOKEN=" token; done=1; next} {print} END{if(!done) print "HF_TOKEN=" token}' .env >"$token_tmp"
      mv "$token_tmp" .env
    fi
  fi
  chmod 0600 .env
  echo "Created .env with a generated SQL password."
fi
if [[ -S /run/user/1000/docker.sock ]] && ! DOCKER_HOST=unix:///var/run/docker.sock docker info >/dev/null 2>&1; then
  profile_tmp="$(mktemp .env.runtime.XXXXXX)"
  awk 'BEGIN{done=0} /^CONTAINER_RUNTIME_PROFILE=/{print "CONTAINER_RUNTIME_PROFILE=colab-rootless"; done=1; next} {print} END{if(!done) print "CONTAINER_RUNTIME_PROFILE=colab-rootless"}' .env >"$profile_tmp"
  chmod 0600 "$profile_tmp"
  mv "$profile_tmp" .env
fi
# shellcheck disable=SC1091
source "$script_dir/runtime-env.sh"

reuse_lab1_sql=0
if [[ "${CONTAINER_RUNTIME_PROFILE:-}" == "colab-rootless" ]] && docker ps --format '{{.Names}}' | grep -qx 'aidataapps-rag-sqlserver-1'; then
  reuse_lab1_sql=1
  inherited_sql_password="$(sed -n 's/^MSSQL_SA_PASSWORD=//p' ../rag/.env | head -n1)"
  password_tmp="$(mktemp .env.sql.XXXXXX)"
  awk -v password="$inherited_sql_password" 'BEGIN{done=0} /^MSSQL_SA_PASSWORD=/{print "MSSQL_SA_PASSWORD=" password; done=1; next} {print} END{if(!done) print "MSSQL_SA_PASSWORD=" password}' .env >"$password_tmp"
  chmod 0600 "$password_tmp"
  mv "$password_tmp" .env
  export MSSQL_SA_PASSWORD="$inherited_sql_password"
  echo "Reusing the healthy Lab 1 SQL Server instance with an isolated ModelPrint database."
fi

for command_name in docker curl npm openssl nvidia-smi python3; do command -v "$command_name" >/dev/null || { echo "Missing prerequisite: $command_name" >&2; exit 1; }; done
docker compose version >/dev/null
docker info >/dev/null
nvidia-smi >/dev/null

# These caches are intentionally shared with Lab 1 so model downloads remain
# available across labs. Compose treats them as externally managed volumes.
for volume_name in "${SHARED_HF_VOLUME:-aidataapps-rag-huggingface-cache}" "${SHARED_VLLM_VOLUME:-aidataapps-rag-vllm-cache}"; do
  docker volume inspect "$volume_name" >/dev/null 2>&1 || docker volume create "$volume_name" >/dev/null
done

# The two labs use the same fixed Colab ports. Stop only the known Lab 1
# containers; volumes and its database remain intact and recoverable.
for container in aidataapps-rag-chat aidataapps-rag-vllm-embedding-1; do
  if docker container inspect "$container" >/dev/null 2>&1; then
    if [[ "${CONTAINER_RUNTIME_PROFILE:-}" == "colab-rootless" ]]; then docker kill "$container" >/dev/null || true;
    else docker stop --time 30 "$container" >/dev/null || docker kill "$container" >/dev/null; fi
  fi
done

npm ci
if ((skip_python == 0)); then
  if ! python3 -m ensurepip --version >/dev/null 2>&1; then
    python_minor="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    apt-get update
    apt-get install -y "python${python_minor}-venv"
  fi
  python3 -m venv --clear .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/pip install -r requirements.lock
fi
docker compose config --quiet
services=(embedding-qwen)
if ((reuse_lab1_sql == 0)); then services=(sqlserver "${services[@]}"); fi
if ((primary_only == 0)); then services+=(embedding-bge); fi
if ((skip_pull == 0)); then docker compose pull "${services[@]}"; fi
docker compose up --detach "${services[@]}"

echo "Waiting for SQL Server..."
if ((reuse_lab1_sql == 0)); then
  for _ in $(seq 1 120); do
    if docker compose exec -T sqlserver /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C -Q "SELECT 1" -b -o /dev/null 2>/dev/null; then break; fi
    sleep 2
  done
  docker compose exec -T sqlserver /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P "$MSSQL_SA_PASSWORD" -C -Q "SELECT @@VERSION" -W
else
  for _ in $(seq 1 120); do (echo >/dev/tcp/127.0.0.1/1433) >/dev/null 2>&1 && break; sleep 2; done
fi

for port in "$EMBEDDING_PORT"; do
  echo "Waiting for embedding service on $port..."
  for _ in $(seq 1 600); do curl --fail --silent "http://127.0.0.1:${port}/health" >/dev/null && break; sleep 2; done
  curl --fail --silent "http://127.0.0.1:${port}/health" >/dev/null
done
if ((primary_only == 0)); then
  for _ in $(seq 1 600); do curl --fail --silent "http://127.0.0.1:${SECOND_EMBEDDING_PORT}/health" >/dev/null && break; sleep 2; done
  curl --fail --silent "http://127.0.0.1:${SECOND_EMBEDDING_PORT}/health" >/dev/null
fi

npm run db:setup
npm run models:sync
npm run prompts:build
npm run check
echo "ModelPrint foundation services are ready. No target chat model has been loaded."
