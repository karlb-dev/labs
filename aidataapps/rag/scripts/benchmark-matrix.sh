#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lab_dir="$(cd "$script_dir/.." && pwd)"
cd "$lab_dir"

# shellcheck disable=SC1091
source "$script_dir/runtime-env.sh"

keep_model_cache=0
profiles=()
for argument in "$@"; do
  case "$argument" in
    --keep-model-cache)
      keep_model_cache=1
      ;;
    --*)
      echo "Unknown option: $argument" >&2
      echo "Usage: ./scripts/benchmark-matrix.sh [--keep-model-cache] [PROFILE ...]" >&2
      exit 2
      ;;
    *)
      profiles+=("$argument")
      ;;
  esac
done
if ((${#profiles[@]} == 0)); then
  mapfile -t profiles < <(npm run --silent model -- benchmark-list)
fi

failures=0
for profile_index in "${!profiles[@]}"; do
  profile="${profiles[$profile_index]}"
  echo "Starting benchmark profile: $profile"
  if ! npm run model -- start --profile "$profile" --replace; then
    echo "Model container failed to start: $profile" >&2
    failures=$((failures + 1))
    if [[ "${CONTAINER_RUNTIME_PROFILE:-}" == "colab-rootless" ]] && ((keep_model_cache == 0)); then
      npm run model -- evict --profile "$profile" || true
    fi
    continue
  fi
  for _ in $(seq 1 900); do
    if curl --fail --silent "${CHAT_BASE_URL%/v1}/health" >/dev/null; then
      break
    fi
    sleep 2
  done
  if ! curl --fail --silent "${CHAT_BASE_URL%/v1}/health" >/dev/null; then
    echo "Model failed to become ready: $profile" >&2
    failures=$((failures + 1))
    if [[ "${CONTAINER_RUNTIME_PROFILE:-}" == "colab-rootless" ]] && ((keep_model_cache == 0)); then
      npm run model -- evict --profile "$profile" || true
    fi
    continue
  fi
  if ! MODEL_PROFILE="$profile" npm run benchmark -- --profile "$profile"; then
    failures=$((failures + 1))
  fi
  if [[ "${CONTAINER_RUNTIME_PROFILE:-}" == "colab-rootless" ]] \
    && ((keep_model_cache == 0)) \
    && ((profile_index + 1 < ${#profiles[@]})); then
    echo "Evicting $profile weights to make room for the next Colab matrix cell."
    npm run model -- evict --profile "$profile"
  fi
done

if ((failures)); then
  echo "$failures profile(s) failed." >&2
  exit 1
fi
echo "Benchmark matrix completed. See runs/benchmark-*/."
