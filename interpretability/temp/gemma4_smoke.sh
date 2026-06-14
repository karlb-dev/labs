#!/bin/bash
# Gemma-4-E4B smoke suite across a diverse set of labs (bounded for ~30 min).
set -u
cd /content/labs/interpretability
export HF_HOME=/content/hf_local HF_HUB_DISABLE_TELEMETRY=1
MODEL=google/gemma-4-E4B-it
DEST=/content/drive/MyDrive/interpret/validate_part2_gemma/e4b_smoke
mkdir -p "$DEST"
RESULTS="$DEST/SMOKE_RESULTS.md"
echo "# Gemma-4-E4B smoke results ($(date -u +%H:%M)Z)" > "$RESULTS"
echo "" >> "$RESULTS"
echo "Model: \`$MODEL\` | bounded --max-examples 8 | tier b" >> "$RESULTS"
echo "" >> "$RESULTS"
echo "| lab | exit | secs | plots | headline |" >> "$RESULTS"
echo "|---|---|---|---|---|" >> "$RESULTS"

LABS="lab1 lab5 lab7 lab13 lab14 lab16 lab22"
for L in $LABS; do
  RN=${L}_gemma4_smoke
  t0=$SECONDS
  timeout 500 python -u interp_bench.py --lab $L --tier b --model $MODEL \
    --max-examples 8 --run-name $RN > /tmp/${RN}.log 2>&1
  ec=$?
  dt=$((SECONDS - t0))
  np=$(ls runs/$RN/plots/*.png 2>/dev/null | wc -l)
  # headline: first self-check + a key metric line
  hp=$(grep -oE "hook parity[^,]*OK[^,)]*" /tmp/${RN}.log | head -1)
  hl=$(grep -iE "verdict|selected|AUC|Best|finished" runs/$RN/run_summary.md 2>/dev/null | head -1 | cut -c1-80)
  [ -z "$hl" ] && hl=$(grep -iE "Error|Traceback|Could not|RuntimeError|ValueError" /tmp/${RN}.log | head -1 | cut -c1-80)
  echo "| $L | $ec | $dt | $np | ${hp:+hookOK; }$hl |" >> "$RESULTS"
  # sync the run (light: no state/matplotlib)
  rsync -a --exclude matplotlib_config --exclude state runs/$RN/ "$DEST/$L/" 2>/dev/null
  echo "DONE $L ec=$ec ${dt}s plots=$np"
done
echo "" >> "$RESULTS"
echo "_Completed $(date -u +%H:%M)Z_" >> "$RESULTS"
echo "ALL_DONE"
