#!/usr/bin/env bash
# CPU reproduction gate for the official-repro study namespace.
# --verify additionally rehashes every registered output (needs Drive for
# Drive-resident rows).
set -euo pipefail
cd "$(dirname "$0")/../../../.."

pip install -e interpretability/jspaces/sidelines/official_repro

if [ ! -d "${JLENS_ROOT:-/content/or1_work/jacobian-lens}" ]; then
  echo "jacobian-lens clone missing; clone at pinned 581d3986 first" >&2
  exit 1
fi

python -m pytest interpretability/jspaces/sidelines/official_repro/tests -q

if [ "${1:-}" = "--verify" ]; then
  jspace-or1 verify
fi
