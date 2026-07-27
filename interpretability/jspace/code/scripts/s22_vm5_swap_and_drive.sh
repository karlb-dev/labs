#!/bin/bash
# VM5 disk swap + OLMo driver (final form — see LOG.md for the incident
# history). Root cause of the earlier crashes: Colab's DriveFS keeps its
# streamed-read cache in a single SQLite chunks.db on local disk, so
# (a) Qwen-local (55 GB) + OLMo-from-Drive (64 GB stream) cannot coexist
# on the 113 GB disk, and (b) file-level cache eviction is IMPOSSIBLE —
# deleting chunks.db under the daemon wedges the mount into serving
# empty content until the worker is restarted (kill -9 the
# --single_process drive worker; its supervisor respawns it and reads
# recover). Strategy: delete the Qwen blobs after phase Q, purge unused
# fat packages (RAPIDS/pyspark/cupy: ~7 GB), restart the worker for a
# fresh cache, and run with headroom > total OLMo-phase Drive reads
# (~67 GB). The watchdog only LOGS if free disk dips — it never deletes.
set -u
cd "$(dirname "$0")/.."
SCRATCH="${SCRATCH:-/tmp/claude-0/-content-labs/a6ed18f3-9bf8-4f6a-aed0-b73ed71939ad/scratchpad}"

rm -rf /content/hf_local
sync
echo "post-swap disk: $(df -h / | tail -1)"

( while true; do
    avail=$(df --output=avail / | tail -1 | tr -d ' ')
    if [ "$avail" -lt 4194304 ]; then
      echo "[watchdog $(date -u +%H:%M:%S)] LOW DISK: ${avail}KB free" \
        >> logs_local/s21_driver_vm5.log
    fi
    sleep 20
  done ) &
echo $! > "$SCRATCH/watchdog.pid"

export HF_HUB_OFFLINE=1
set -o pipefail
python scripts/s21_vm5_driver.py 2>&1 | tee logs_local/s21_driver_vm5.log
rc=$?
kill "$(cat "$SCRATCH/watchdog.pid")" 2>/dev/null
echo "driver exit: $rc; disk: $(df -h / | tail -1)"
exit $rc
