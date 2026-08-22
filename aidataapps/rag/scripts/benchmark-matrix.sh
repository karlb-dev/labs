#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lab_dir="$(cd "$script_dir/.." && pwd)"
cd "$lab_dir"

set -a
# shellcheck disable=SC1091
source .env
set +a

if (($#)); then
  profiles=("$@")
else
  mapfile -t profiles < <(npm run --silent model -- benchmark-list)
fi

failures=0
for profile in "${profiles[@]}"; do
  echo "Starting benchmark profile: $profile"
  npm run model -- start --profile "$profile" --replace
  for _ in $(seq 1 900); do
    if curl --fail --silent "${CHAT_BASE_URL%/v1}/health" >/dev/null; then
      break
    fi
    sleep 2
  done
  if ! curl --fail --silent "${CHAT_BASE_URL%/v1}/health" >/dev/null; then
    echo "Model failed to become ready: $profile" >&2
    failures=$((failures + 1))
    continue
  fi
  if ! MODEL_PROFILE="$profile" npm run benchmark -- --profile "$profile"; then
    failures=$((failures + 1))
  fi
done

if ((failures)); then
  echo "$failures profile(s) failed." >&2
  exit 1
fi
echo "Benchmark matrix completed. See runs/benchmark-*/."
