#!/bin/bash
# Single-command reproduction entry (protocol/REPRO_CONTRACT.md).
#   bash interpretability/jspace_part2/repro.sh                 # install + selftest + env audit
#   bash interpretability/jspace_part2/repro.sh <evidence-id>   # + verify/re-run that evidence
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
pip install -q -e "$HERE"
jspace-part2 selftest
jspace-part2 audit-env >/dev/null
echo "package installed; conformance tests green; environment audited."
if [ $# -ge 1 ]; then
  jspace-part2 repro "$@"
fi
