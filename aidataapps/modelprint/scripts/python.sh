#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lab_dir="$(cd "$script_dir/.." && pwd)"
if [[ ! -x "$lab_dir/.venv/bin/python" ]]; then
  echo "Missing .venv. Run ./scripts/env-init.sh first." >&2
  exit 1
fi
exec "$lab_dir/.venv/bin/python" "$@"
