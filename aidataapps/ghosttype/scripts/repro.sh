#!/usr/bin/env bash
# GhostType reproduction gate.
#   ./scripts/repro.sh --mode rows      rebuild all row-level state from the
#                                       vendored package into the running SQL
#                                       container and re-run every gate.
#   ./scripts/repro.sh --mode restore   restore the latest .bak (Tier-2 path;
#                                       re-applies PREVIEW_FEATURES + compat
#                                       per the ModelPrint restore scar).
# Row mode is the Tier-1 closeout requirement: it must pass on a fresh
# checkout with only the SQL container and node toolchain present.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="rows"
[[ "${1:-}" == "--mode" ]] && MODE="${2:-rows}"

export PATH="$HOME/.nvm/versions/node/v22.22.1/bin:$PATH"

echo "[repro] mode=$MODE"
if [[ "$MODE" == "rows" ]]; then
  npm run -s db:setup
  npm run -s dataset:import
  npm run -s dataset:validate
  npm run -s catalog:snapshot
  npm run -s oracles:validate
  npm run -s baselines:evaluate > /dev/null
  npm run -s metrics:compute > /dev/null
  npm test
  echo "[repro] rows mode PASS: gates, oracles, baselines, and unit suite reproduced"
elif [[ "$MODE" == "restore" ]]; then
  echo "[repro] restore mode is Tier-2; use scripts/export-database.sh restore <bak>" >&2
  exit 2
else
  echo "[repro] unknown mode $MODE" >&2
  exit 2
fi
