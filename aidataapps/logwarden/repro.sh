#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
mode="rows"
while (($#)); do
  case "$1" in
    --mode) mode="${2:?--mode requires rows or restore}"; shift 2 ;;
    *) echo "Usage: $0 --mode rows|restore" >&2; exit 2 ;;
  esac
done
if [[ "$mode" != "rows" && "$mode" != "restore" ]]; then
  echo "Unsupported reproduction mode: $mode" >&2
  exit 2
fi

npm ci
npm run check

if [[ ! -s .current-run ]]; then
  echo "Tier 1 is not frozen: no current run is recorded." >&2
  exit 3
fi
run_dir="$(realpath "$(cat .current-run)")"
case "$run_dir" in
  "$script_dir"/runs/*) ;;
  *) echo "Refusing run outside $script_dir/runs: $run_dir" >&2; exit 4 ;;
esac

if [[ ! -f "$run_dir/manifests/freeze.json" ]]; then
  echo "Tier 1 is not frozen: $run_dir/manifests/freeze.json is absent." >&2
  exit 3
fi

echo "The $mode reconstruction driver is scaffolded but the Tier 1 row exports are not complete." >&2
exit 3
