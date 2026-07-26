#!/bin/bash
# One-command handout refresh — run after every result-bearing v2 phase.
# Regenerates v2 figures from metrics, recompiles the living handout tex,
# ships tex+pdf+figures to the v2 run dir, and mirrors code. The v1 run
# dir's report/handout/ stays frozen as the v1 archive.
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
V2="${SL1_RUN_DIR_V2:-/content/drive/MyDrive/interpret/special-lab-1/2026-07-26_v2}"
cd "$SRC"
python scripts/s19_figures_v2.py
cp -f "$V2"/figures/f*.png handout/figures/ 2>/dev/null || true
cd handout
pdflatex -interaction=nonstopmode olmo32b_jspace_handout.tex >/dev/null
pdflatex -interaction=nonstopmode olmo32b_jspace_handout.tex > .compile.log 2>&1
grep -E "^!" .compile.log && { echo "LATEX ERROR"; exit 1; }
grep -E "Output written" .compile.log
rm -f *.aux *.out .compile.log olmo32b_jspace_handout.log
mkdir -p "$V2/report/handout"
rsync -a olmo32b_jspace_handout.tex olmo32b_jspace_handout.pdf figures \
  "$V2/report/handout/"
cd "$SRC" && bash sync_code.sh
echo "handout refreshed -> $V2/report/handout/"
