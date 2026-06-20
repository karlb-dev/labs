#!/bin/bash
# Detached chained runner: Olmo (already running) -> commit -> Gemma -> commit.
# Keeps the Colab GPU busy across models with no operator in the loop.
# Launched via nohup; logs to /tmp/remaining_runner.log.
set -uo pipefail
export PYTHONUNBUFFERED=1   # line-flush child output so progress is observable
cd /content/labs/interpretability
REPO=/content/labs
DATE=20260620
say() { echo "[runner $(date +%H:%M:%S)] $*"; }

commit_push() {
  cd "$REPO"
  git add interpretability/lab06_matrix/ interpretability/labs/lab06_circuit_discovery.py interpretability/interp_bench.py 2>/dev/null
  git commit -q -m "$1" 2>&1 | tail -1 || say "nothing to commit"
  git push origin interp2 2>&1 | tail -1 || say "push failed (will retry next stage)"
  cd /content/labs/interpretability
}

# ---- Stage 1: wait for the already-running Olmo matrix -------------------------
say "waiting for Olmo-32B matrix to finish (PID watch + log DONE marker)"
while true; do
  if grep -q "^\[matrix\] DONE\." /tmp/olmo_matrix.log 2>/dev/null; then
    say "Olmo matrix reported DONE"; break
  fi
  if ! pgrep -f "lab06_matrix.py --model allenai/Olmo-3-1125-32B" >/dev/null; then
    say "Olmo matrix process gone (check /tmp/olmo_matrix.log for errors)"; break
  fi
  sleep 30
done
commit_push "Lab 6 matrix: Olmo-3-1125-32B Base FINDINGS (auto)"

# ---- Stage 2: disk safeguard before Gemma download ----------------------------
FREE=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
say "disk free before Gemma: ${FREE}G"
THINK=/root/.cache/huggingface/hub/models--allenai--Olmo-3-32B-Think
if [ "${FREE:-0}" -lt 15 ] && [ -d "$THINK" ]; then
  say "disk < 15G: removing user-authorized Think cache to make room"
  rm -rf "$THINK"
  say "disk free now: $(df --output=avail -BG / | tail -1 | tr -dc '0-9')G"
fi

# ---- Stage 3: Gemma-4-E4B-it matrix -------------------------------------------
say "starting Gemma-4-E4B-it matrix (downloads weights on first load)"
python lab06_matrix.py --model google/gemma-4-E4B-it --dtype bfloat16 --tier b \
  --date "$DATE" --resample-draws 5 --trust-remote-code \
  > /tmp/gemma_matrix.log 2>&1
GEMMA_RC=$?
say "Gemma matrix exited rc=$GEMMA_RC (see /tmp/gemma_matrix.log)"
commit_push "Lab 6 matrix: gemma-4-E4B-it FINDINGS (auto)"

say "ALL STAGES DONE"
