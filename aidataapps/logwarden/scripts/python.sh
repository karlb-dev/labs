#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lab_dir="$(cd "$script_dir/.." && pwd)"
venv="$lab_dir/.venv"
if [[ ! -x "$venv/bin/python" ]]; then
  python3 -m venv "$venv"
  "$venv/bin/pip" install --upgrade pip
  "$venv/bin/pip" install -e "$lab_dir[dev]"
fi
exec "$venv/bin/python" "$@"
