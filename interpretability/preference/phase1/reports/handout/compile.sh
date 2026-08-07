#!/usr/bin/env bash
# Reproducible handout build (jspaces compile.sh pattern): pinned PDF
# metadata clock so committed PDFs rebuild byte-stably; two passes; build
# in /tmp so aux files never enter git.
set -euo pipefail
cd "$(dirname "$0")"
command -v pdflatex >/dev/null || { echo "pdflatex missing"; exit 1; }
export SOURCE_DATE_EPOCH=1786212000   # 2026-08-07 preference phase1 part1
export FORCE_SOURCE_DATE=1
BUILD=$(mktemp -d)
trap 'rm -rf "$BUILD"' EXIT
for tex in *.tex; do
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD" "$tex" >/dev/null
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD" "$tex" >/dev/null
  cp "$BUILD/${tex%.tex}.pdf" .
  echo "built ${tex%.tex}.pdf"
done
