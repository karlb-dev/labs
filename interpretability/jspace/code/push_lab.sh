#!/bin/bash
# Mirror the live lab into the published jspace/ dir and push to the draft
# branch. Run at phase boundaries, right after refresh_handout.sh:
#   bash push_lab.sh "P1: energy-matched grid landed — <one-line result>"
# Small metrics only (<5MB) go to git; heavy artifacts stay on Drive.
set -euo pipefail
MSG="${1:?usage: push_lab.sh \"commit message\"}"
SRC="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SRC/.." && pwd)"
V1="${SL1_RUN_DIR:-/content/drive/MyDrive/interpret/special-lab-1/2026-07-25_1726}"
V2="${SL1_RUN_DIR_V2:-/content/drive/MyDrive/interpret/special-lab-1/2026-07-26_v2}"
J="$REPO/jspace"
BRANCH=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)
[ "$BRANCH" = "interp_jspace" ] || { echo "refusing: on branch $BRANCH"; exit 1; }

mkdir -p "$J"/{code/scripts,report,handout,figures,results}
cp "$SRC"/sl1_common.py "$SRC"/PLAN.md "$SRC"/PLAN_v2.md "$SRC"/LOG.md \
   "$SRC"/sync_code.sh "$SRC"/refresh_handout.sh "$SRC"/push_lab.sh "$J/code/"
cp "$SRC"/scripts/*.py "$J/code/scripts/"
cp "$SRC"/handout/olmo32b_jspace_handout.tex \
   "$SRC"/handout/olmo32b_jspace_handout.pdf "$J/handout/"
rsync -a "$SRC/handout/figures/" "$J/handout/figures/"
cp "$V1"/figures/*.png "$J/figures/" 2>/dev/null || true
cp "$V2"/figures/*.png "$J/figures/" 2>/dev/null || true
cp "$V1"/report/REPORT.md "$V1"/report/summary.json "$J/report/" 2>/dev/null || true
cp "$V2"/report/summary_v2.json "$V2"/report/REPORT_v2.md "$J/report/" 2>/dev/null || true
for f in ablation_results broadcast lens_sanity_32b fit_32b smoke_7b \
         smoke_verify_vm2 cot_lead evalaware cot_results; do
  cp "$V1/metrics/$f.json" "$J/results/v1_$f.json" 2>/dev/null || true
done
for p in "$V2"/metrics/*.json; do
  [ -f "$p" ] || continue
  [ "$(stat -c%s "$p")" -lt 5000000 ] || { echo "skip (>5MB): $p"; continue; }
  cp "$p" "$J/results/v2_$(basename "$p")"
done
cd "$REPO"
git add jspace labs/lab37_jspace_workspace.md
if git diff --cached --quiet; then echo "nothing to commit"; exit 0; fi
git commit -q -m "$MSG" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push -q origin interp_jspace
echo "pushed: $(git log --format='%h %s' -1)"
