#!/bin/bash
# Mirror the lab code to the Drive run dir. Run after every edit session —
# VM1 died with code/ empty; this is the rule that prevents a repeat.
set -euo pipefail
RUN_DIR="${SL1_RUN_DIR:-/content/drive/MyDrive/interpret/special-lab-1/2026-07-25_1726}"
SRC="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$RUN_DIR/code" "$RUN_DIR/logs"
rsync -a --exclude='__pycache__' --exclude='*.pyc' --exclude='logs_local' \
  "$SRC/" "$RUN_DIR/code/special_lab1/"
# jlens is a third-party clone; keep a copy of the exact revision too (small).
rsync -a --exclude='.git' --exclude='__pycache__' --exclude='assets' \
  /content/jacobian-lens/jlens "$RUN_DIR/code/jacobian-lens-pkg/" 2>/dev/null || true
git -C /content/jacobian-lens rev-parse HEAD > "$RUN_DIR/code/jacobian-lens-pkg/GIT_REVISION.txt" 2>/dev/null || true
# Local logs -> Drive (append-only copies).
rsync -a "$SRC/logs_local/" "$RUN_DIR/logs/" 2>/dev/null || true
# v2 delta run gets its own full mirror (same rules).
RUN_DIR_V2="${SL1_RUN_DIR_V2:-/content/drive/MyDrive/interpret/special-lab-1/2026-07-26_v2}"
if [ -d "$RUN_DIR_V2" ]; then
  mkdir -p "$RUN_DIR_V2/code" "$RUN_DIR_V2/logs"
  rsync -a --exclude='__pycache__' --exclude='*.pyc' --exclude='logs_local' \
    --exclude='handout' "$SRC/" "$RUN_DIR_V2/code/special_lab1/"
  rsync -a "$SRC/logs_local/" "$RUN_DIR_V2/logs/" 2>/dev/null || true
fi
echo "synced $(date -u +%H:%M:%S) -> $RUN_DIR/code/ (+v2)"
