#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lab_dir="$(cd "$script_dir/.." && pwd)"
cd "$lab_dir"

# shellcheck disable=SC1091
source "$script_dir/runtime-env.sh"

npm run model -- stop
docker compose down
echo "Stopped containers. Named data/model cache volumes were preserved."
