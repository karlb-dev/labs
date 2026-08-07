#!/bin/bash
# One-command Part-2 reporting refresh — run after every result-bearing phase.
# Regenerates figures from metrics, recompiles the living handout (when the
# tex exists), ships tex+pdf+figures to the run dir, and mirrors code.
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
RUN="${SL2_RUN_DIR:-/content/drive/MyDrive/interpret/special-lab-1/part2_20260727}"
cd "$SRC"
[ -f scripts/p2fig.py ] && python scripts/p2fig.py
TEX=handout/jspace_part2_handout.tex
if [ -f "$TEX" ]; then
  cp -f "$RUN"/figures/p2f*.png handout/figures/ 2>/dev/null || true
  cd handout
  pdflatex -interaction=nonstopmode jspace_part2_handout.tex >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode jspace_part2_handout.tex > .compile.log 2>&1 || true
  if ! grep -q "Output written" .compile.log; then
    echo "LATEX ERROR:"; grep -B1 -A4 "^!" .compile.log | head -20; exit 1
  fi
  rm -f *.aux *.out .compile.log jspace_part2_handout.log
  mkdir -p "$RUN/report/handout"
  rsync -a jspace_part2_handout.tex jspace_part2_handout.pdf figures "$RUN/report/handout/"
  cd "$SRC"
else
  echo "(no handout tex yet — skipping compile)"
fi
bash sync_code.sh
echo "reporting refreshed -> $RUN/report/"
