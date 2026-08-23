#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lab_dir="$(cd "$script_dir/.." && pwd)"
repo_root="$(git -C "$lab_dir" rev-parse --show-toplevel)"
interval_seconds=1200
mode=watch

while [[ $# -gt 0 ]]; do
  case "$1" in
    --once) mode=once; shift ;;
    --interval-seconds) interval_seconds="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if ! [[ "$interval_seconds" =~ ^[0-9]+$ ]] || (( interval_seconds < 60 )); then
  echo "checkpoint interval must be an integer >= 60 seconds" >&2
  exit 2
fi

cd "$lab_dir"
# shellcheck disable=SC1091
source "$script_dir/runtime-env.sh"
run_dir="$(node --import tsx -e 'import {resolveRunDirectory} from "./src/run.ts"; console.log(resolveRunDirectory())')"
mkdir -p "$run_dir/watchdog" "$run_dir/recovery"
lock_file="$run_dir/watchdog/checkpoint.lock"

checkpoint_once() {
  local stamp branch head clean_at_start bundle patch untracked_list untracked_archive
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  branch="$(git -C "$repo_root" branch --show-current)"
  if [[ "$branch" != "aidataapps-logwarden" ]]; then
    echo "Refusing watchdog on unexpected branch: $branch" >&2
    return 4
  fi
  head="$(git -C "$repo_root" rev-parse HEAD)"
  clean_at_start=0
  if [[ -z "$(git -C "$repo_root" status --porcelain)" ]]; then clean_at_start=1; fi

  npm run db:backup
  patch="$run_dir/recovery/worktree-$stamp.patch"
  git -C "$repo_root" diff --binary HEAD >"$patch"
  untracked_list="$run_dir/recovery/untracked-$stamp.list"
  git -C "$repo_root" ls-files --others --exclude-standard >"$untracked_list"
  if [[ -s "$untracked_list" ]]; then
    untracked_archive="$run_dir/recovery/untracked-$stamp.tar.gz"
    tar -C "$repo_root" -czf "$untracked_archive" -T "$untracked_list"
  fi

  if (( clean_at_start == 1 )); then
    node --import tsx scripts/update-watchdog-status.ts
    git -C "$repo_root" add aidataapps/inprogress_lab3.md
    git -C "$repo_root" commit -m "Checkpoint Lab 3 watchdog $stamp"
  else
    node --import tsx scripts/update-watchdog-status.ts --drive-only
  fi

  git -C "$repo_root" push origin "HEAD:refs/heads/$branch"
  head="$(git -C "$repo_root" rev-parse HEAD)"
  bundle="$run_dir/recovery/$branch-$stamp.bundle"
  git -C "$repo_root" bundle create "$bundle" "$branch"
  git -C "$repo_root" bundle verify "$bundle" >"$run_dir/recovery/bundle-verify-$stamp.txt" 2>&1

  npm run checkpoints:prune -- --apply --keep-latest 2 >/dev/null
  npm run run:archive >/dev/null
  npm run run:mirror >/dev/null
  jq -cn \
    --arg atUtc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg branch "$branch" --arg head "$head" --arg bundle "$bundle" \
    --argjson cleanAtStart "$clean_at_start" \
    '{schemaVersion:1,atUtc:$atUtc,branch:$branch,head:$head,bundle:$bundle,cleanAtStart:($cleanAtStart==1),disposition:"complete"}' \
    >>"$run_dir/watchdog/checkpoints.jsonl"
}

if [[ "$mode" == "once" ]]; then
  (
    flock -n 9
    checkpoint_once
  ) 9>"$lock_file"
  exit
fi

while true; do
  if ! (
    flock -n 9
    checkpoint_once
  ) 9>"$lock_file"; then
    jq -cn --arg atUtc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      '{schemaVersion:1,atUtc:$atUtc,disposition:"failed_or_lock_busy"}' \
      >>"$run_dir/watchdog/checkpoints.jsonl"
  fi
  sleep "$interval_seconds"
done
