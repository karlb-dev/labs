#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
# shellcheck disable=SC1091
source "$script_dir/runtime-env.sh"
docker compose --profile embedding --profile embedding-secondary down
