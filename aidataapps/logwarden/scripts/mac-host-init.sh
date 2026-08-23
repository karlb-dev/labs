#!/usr/bin/env bash
set -Eeuo pipefail

# Mac host bootstrap for the LogWarden mac profile (see docs/MAC_PROFILE.md).
# Verifies Docker Desktop with Rosetta amd64 emulation, generates .env with
# local secrets, and selects CONTAINER_RUNTIME_PROFILE=mac-docker-desktop.
# Azure Foundry Local is checked but optional: the CPU-only foundation stage
# (env-init, run:init, db:setup, doctor, check) needs no model service.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lab_dir="$(cd "$script_dir/.." && pwd)"
cd "$lab_dir"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This bootstrap is for macOS; on Colab run ./scripts/colab-host-init.sh." >&2
  exit 1
fi

for command_name in docker curl npm openssl node; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing prerequisite: $command_name" >&2
    exit 1
  fi
done
docker info >/dev/null || { echo "Docker Desktop is not running." >&2; exit 1; }

emulated_arch="$(docker run --rm --platform linux/amd64 alpine uname -m 2>/dev/null || true)"
if [[ "$emulated_arch" != "x86_64" ]]; then
  echo "amd64 emulation is unavailable; enable Rosetta in Docker Desktop:" >&2
  echo "Settings > General > 'Use Rosetta for x86_64/amd64 emulation'." >&2
  exit 1
fi

if command -v foundry >/dev/null 2>&1; then
  echo "Foundry Local CLI: $(foundry --version)"
else
  echo "Foundry Local CLI not found (optional for the foundation stage)."
  echo "Install later with: brew tap microsoft/foundrylocal && brew install foundrylocal"
fi

runs_mirror="${LOGWARDEN_RUNS_MIRROR:-$HOME/aidataapps/lab03/runs}"
mkdir -p "$runs_mirror"

if [[ ! -f .env ]]; then
  umask 077
  sa_password="LogWarden!$(openssl rand -hex 18)"
  lab_password="LwLab!$(openssl rand -hex 18)"
  agent_password="LwAgent!$(openssl rand -hex 18)"
  sed \
    -e "s/replace-with-a-generated-strong-password/$sa_password/" \
    -e "s/replace-with-a-generated-lab-password/$lab_password/" \
    -e "s/replace-with-a-generated-agent-password/$agent_password/" \
    -e "s#^RUNS_MIRROR=.*#RUNS_MIRROR=$runs_mirror#" \
    -e "s#^EMBEDDING_BASE_URL=.*#EMBEDDING_BASE_URL=http://127.0.0.1:8010/v1#" \
    .env.example >.env
  chmod 0600 .env
fi

profile_tmp="$(mktemp .env.tmp.XXXXXX)"
awk '
  BEGIN { replaced = 0 }
  /^CONTAINER_RUNTIME_PROFILE=/ {
    print "CONTAINER_RUNTIME_PROFILE=mac-docker-desktop"
    replaced = 1
    next
  }
  { print }
  END {
    if (!replaced) print "CONTAINER_RUNTIME_PROFILE=mac-docker-desktop"
  }
' .env >"$profile_tmp"
chmod 0600 "$profile_tmp"
mv "$profile_tmp" .env

echo "Mac host is ready (runs mirror: $runs_mirror)"
echo "Next: ./scripts/env-init.sh"
