#!/usr/bin/env bash
set -euo pipefail

OLMO_PAPER_ROOT="$(cd "$(dirname "$0")" && pwd)"
OLMO_PAPER_BUILD="${JSPACE_OLMO_PAPER_BUILD:-/tmp/olmo_lineage_paper_build}"

command -v pdflatex >/dev/null
command -v pdfinfo >/dev/null
mkdir -p "$OLMO_PAPER_BUILD"

# Freeze the PDF metadata clock to the independent-reconstruction event so the
# committed run-specific PDF can be rebuilt byte-for-byte under the same TeX
# toolchain.
export SOURCE_DATE_EPOCH=1785648410
export FORCE_SOURCE_DATE=1

cd "$OLMO_PAPER_ROOT"
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory="$OLMO_PAPER_BUILD" \
  olmo_lineage_parallel_phase.tex
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory="$OLMO_PAPER_BUILD" \
  olmo_lineage_parallel_phase.tex
cp "$OLMO_PAPER_BUILD/olmo_lineage_parallel_phase.pdf" \
  "$OLMO_PAPER_ROOT/olmo_lineage_parallel_phase.pdf"
pdfinfo "$OLMO_PAPER_ROOT/olmo_lineage_parallel_phase.pdf"
