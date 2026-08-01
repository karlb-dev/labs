#!/usr/bin/env bash
# Apply the prospectively frozen post-A500 A/B/C continuation map.
set -euo pipefail

PHASE4_ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$PHASE4_ROOT/../.." && pwd)"
RUN_ROOT="${JSPACE4_RUN_ROOT:-/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731}"
EXPECTED_BRANCH="${JSPACE4_EXPECTED_BRANCH:-interp_jspace_part2}"
CONFIG_PATH="interpretability/jspace_phase4/configs/p4_qwen_multilens_functional_gate_a250_a500_dev.yaml"

if [[ "${1:-}" == "--approval-probe" ]]; then
  printf '%s\n' "frozen A500 branch router is installed; no GPU work started"
  exit 0
fi
if (( $# != 0 )); then
  printf '%s\n' "usage: $0 [--approval-probe]" >&2
  exit 2
fi

cd "$REPO_ROOT"
if [[ -n "$(git status --porcelain)" ]]; then
  printf '%s\n' "branch router refused: repository is not clean" >&2
  git status --short >&2
  exit 3
fi
if [[ "$(git branch --show-current)" != "$EXPECTED_BRANCH" ]]; then
  printf '%s\n' "branch router refused: unexpected Git branch" >&2
  exit 4
fi

exec 9>"$RUN_ROOT/qwen_frozen_branch_followup.lock"
if ! flock -n 9; then
  printf '%s\n' "branch router refused: another owner holds the lock" >&2
  exit 5
fi

ROUTE_JSON=$(python -m jspace_phase4.experiments.p4_qwen_branch_router \
  --config "$CONFIG_PATH")
CONTINUATION=$(jq -r '.continuation' <<<"$ROUTE_JSON")
BRANCH=$(jq -r '.branch' <<<"$ROUTE_JSON")
case "$CONTINUATION" in
  draw_b_n120|draw_a_n1000) ;;
  *)
    printf '%s\n' "branch router produced an invalid continuation" >&2
    exit 6
    ;;
esac

LOG_PATH="$RUN_ROOT/qwen_frozen_branch_followup_20260801.log"
printf '%s ROUTE branch=%s continuation=%s commit=%s\n' \
  "$(date -u +%FT%TZ)" "$BRANCH" "$CONTINUATION" "$(git rev-parse HEAD)" \
  | tee -a "$LOG_PATH"
printf '%s\n' "$ROUTE_JSON" | tee -a "$LOG_PATH"

exec bash "$PHASE4_ROOT/run_qwen_continuation_fit.sh" "$CONTINUATION"
