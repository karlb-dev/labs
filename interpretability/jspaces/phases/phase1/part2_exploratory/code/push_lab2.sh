#!/bin/bash
# Mirror the live Part-2 lab into interpretability/jspace/part2/ and push.
# Run at phase boundaries, right after refresh_handout.sh:
#   bash push_lab2.sh "A0: <one-line result>"
# Small metrics only (<5MB) go to git; heavy artifacts stay on Drive.
set -euo pipefail
MSG="${1:?usage: push_lab2.sh \"commit message\"}"
SRC="$(cd "$(dirname "$0")" && pwd)"                  # .../interpretability/special_lab2
REPO="$(cd "$SRC/../.." && pwd)"                      # .../labs
RUN="${SL2_RUN_DIR:-/content/drive/MyDrive/interpret/special-lab-1/part2_20260727}"
J="$REPO/interpretability/jspace/part2"
BRANCH=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)
[ "$BRANCH" = "interp_jspace_part2" ] || { echo "refusing: on branch $BRANCH"; exit 1; }

mkdir -p "$J"/{code/scripts,report,handout,figures,results}
cp "$SRC"/sl2_common.py "$SRC"/p2lib.py "$SRC"/PLAN_PART2.md \
   "$SRC"/preregistration.md "$SRC"/REPAIR_PREREGISTRATION.md \
   "$SRC"/jspace_part2_plan1_addendum.md \
   "$SRC"/experiment_reset_instructions.md "$J/code/"
cp "$SRC"/sync_code.sh "$SRC"/push_lab2.sh "$SRC"/refresh_handout.sh "$J/code/" 2>/dev/null || true
cp "$SRC"/scripts/*.py "$J/code/scripts/" 2>/dev/null || true
# metrics: per-model subdirs preserved, <5MB each
if [ -d "$RUN/metrics" ]; then
  (cd "$RUN/metrics" && find . -name '*.json' -size -5M | while read -r p; do
     mkdir -p "$J/results/$(dirname "$p")"; cp "$p" "$J/results/$p"; done)
fi
cp "$RUN"/figures/*.png "$J/figures/" 2>/dev/null || true
cp "$RUN"/report/REPORT_PART2.md "$RUN"/report/summary_part2.json \
   "$RUN"/report/matrix_master.csv "$RUN"/report/matrix_master.json "$J/report/" 2>/dev/null || true
cp "$RUN"/report/handout/*.tex "$RUN"/report/handout/*.pdf "$J/handout/" 2>/dev/null || true

cd "$REPO"
git add interpretability/jspace/part2 interpretability/jspace/REPORT_v2_ERRATA.md
if git diff --cached --quiet; then echo "nothing to commit"; exit 0; fi
git commit -q -m "$MSG" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LZJU4xBSWgBeqsnD2Wfsh5"
git push -q origin interp_jspace_part2
echo "pushed: $(git log --format='%h %s' -1)"
