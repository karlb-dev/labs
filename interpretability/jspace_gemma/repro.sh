#!/usr/bin/env bash
set -euo pipefail

GEMMA_ROOT="$(cd "$(dirname "$0")" && pwd)"

python -m pip install -q -e "$GEMMA_ROOT"
python -m pytest "$GEMMA_ROOT/tests" -q
python -m jspace_gemma verify

echo "jspace_gemma installed; conformance and live-evidence gates passed."
