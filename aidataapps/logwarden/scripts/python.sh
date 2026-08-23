#!/usr/bin/env bash
set -Eeuo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lab_dir="$(cd "$script_dir/.." && pwd)"
venv="$lab_dir/.venv"
required_imports='import matplotlib, numpy, pandas, sklearn'
if [[ -x "$venv/bin/python" ]] && "$venv/bin/python" -c "$required_imports" >/dev/null 2>&1; then
  exec "$venv/bin/python" "$@"
fi

if [[ ! -x "$venv/bin/python" ]] && python3 -m venv "$venv"; then
  "$venv/bin/pip" install --upgrade pip
  "$venv/bin/pip" install -e "$lab_dir[dev]"
  exec "$venv/bin/python" "$@"
fi

# Colab's Python image can omit ensurepip while preinstalling the complete
# scientific stack. Keep the report path usable there without mutating the VM.
if python3 -c "$required_imports" >/dev/null 2>&1; then
  echo "python.sh: isolated venv unavailable; using the verified host scientific stack" >&2
  exec python3 "$@"
fi

echo "python.sh: neither a complete virtualenv nor the required host packages are available" >&2
exit 1
