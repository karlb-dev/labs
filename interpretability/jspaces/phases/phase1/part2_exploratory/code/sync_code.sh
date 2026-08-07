#!/bin/bash
# Mirror the Part-2 live lab code to the Drive run dir. Run after every edit
# session (VM1 died with code/ empty once; never again).
set -euo pipefail
RUN="${SL2_RUN_DIR:-/content/drive/MyDrive/interpret/special-lab-1/part2_20260727}"
SRC="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$RUN/code" "$RUN/logs"
rsync -a --exclude='__pycache__' --exclude='*.pyc' --exclude='logs_local' \
  "$SRC/" "$RUN/code/special_lab2/"
rsync -a "$SRC/logs_local/" "$RUN/logs/" 2>/dev/null || true
# keep the session-reset instructions copy at the special-lab root fresh
cp -f "$SRC/experiment_reset_instructions.md" "$RUN/../experiment_reset_instructions.md"
echo "synced $(date -u +%H:%M:%S) -> $RUN/code/special_lab2/"
