#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lab_dir="$(cd "$script_dir/.." && pwd)"
repo_dir="$(git -C "$lab_dir" rev-parse --show-toplevel)"
mode="loop"
interval_seconds="${MODELPRINT_CHECKPOINT_INTERVAL_SECONDS:-1200}"

usage() {
  echo "Usage: $0 [--once] [--interval-seconds N]" >&2
}

while (($#)); do
  case "$1" in
    --once) mode="once"; shift ;;
    --interval-seconds)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      interval_seconds="$2"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

if [[ ! "$interval_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "Checkpoint interval must be a positive integer" >&2
  exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required for singleton checkpoint ownership" >&2
  exit 3
fi

cd "$lab_dir"
# shellcheck disable=SC1091
source "$script_dir/runtime-env.sh"

run_dir="$(node --import tsx -e 'import {resolveRunDirectory} from "./src/run.ts"; console.log(resolveRunDirectory())')"
run_dir="$(realpath "$run_dir")"
case "$run_dir" in
  "$lab_dir"/runs/*) ;;
  *) echo "Refusing checkpoint outside $lab_dir/runs: $run_dir" >&2; exit 4 ;;
esac

branch="$(git -C "$repo_dir" symbolic-ref --quiet --short HEAD || true)"
expected_branch="${MODELPRINT_CHECKPOINT_BRANCH:-aidataapps-modelprint}"
if [[ "$branch" != "$expected_branch" ]]; then
  echo "Refusing Lab 2 checkpoint on branch '$branch'; expected '$expected_branch'" >&2
  exit 4
fi

checkpoint_dir="$run_dir/checkpoints"
latest_dir="$checkpoint_dir/latest"
log_file="$checkpoint_dir/watchdog.log"
pid_file="$checkpoint_dir/watchdog.pid"
lock_file="$checkpoint_dir/watchdog.lock"
mkdir -p "$latest_dir"

exec 9>"$lock_file"
if ! flock -n 9; then
  echo "A checkpoint watchdog already owns $lock_file"
  exit 0
fi
printf '%s\n' "$$" >"$pid_file"

stage_dir=""
cleanup() {
  rm -f -- "$pid_file"
  if [[ -n "$stage_dir" && "$stage_dir" == /content/modelprint-checkpoint-* && -d "$stage_dir" ]]; then
    rm -rf -- "$stage_dir"
  fi
}
trap cleanup EXIT INT TERM

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >>"$log_file"
}

run_logged() {
  local label="$1"
  shift
  local step_log rc
  step_log="$(mktemp /content/modelprint-step.XXXXXX)"
  log "START $label"
  if "$@" >"$step_log" 2>&1; then
    rc=0
    log "PASS $label"
  else
    rc=$?
    log "FAIL $label rc=$rc"
  fi
  if [[ -s "$step_log" ]]; then
    sed "s/^/[$label] /" "$step_log" >>"$log_file"
  fi
  rm -f -- "$step_log"
  return "$rc"
}

write_recovery_snapshot() {
  local stamp head source_scope_file file metadata_tmp
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  head="$(git -C "$repo_dir" rev-parse HEAD)"
  stage_dir="$(mktemp -d "/content/modelprint-checkpoint-${stamp}.XXXXXX")"
  source_scope_file="$stage_dir/untracked-files.nul"

  git -C "$repo_dir" status --short --branch >"$stage_dir/git-status.txt"
  git -C "$repo_dir" diff --binary HEAD -- \
    aidataapps/modelprint aidataapps/resume.md aidataapps/inprogress_lab2.md \
    >"$stage_dir/working-tree.patch"
  git -C "$repo_dir" ls-files --others --exclude-standard -z -- \
    aidataapps/modelprint aidataapps/resume.md aidataapps/inprogress_lab2.md \
    >"$source_scope_file"
  tr '\0' '\n' <"$source_scope_file" >"$stage_dir/untracked-files.txt"
  tar --null -czf "$stage_dir/untracked-source.tar.gz" -C "$repo_dir" \
    --files-from="$source_scope_file"
  rm -f -- "$source_scope_file"

  git -C "$repo_dir" bundle create "$stage_dir/repository.bundle" "$branch"
  git -C "$repo_dir" bundle verify "$stage_dir/repository.bundle" \
    >"$stage_dir/bundle-verify.txt" 2>&1

  cp "$repo_dir/aidataapps/resume.md" "$stage_dir/resume.md"
  cp "$repo_dir/aidataapps/inprogress_lab2.md" "$stage_dir/inprogress_lab2.md"
  if [[ -f /content/handoff.md ]]; then
    cp /content/handoff.md "$stage_dir/handoff.md"
  fi

  (
    cd "$stage_dir"
    sha256sum repository.bundle working-tree.patch untracked-source.tar.gz \
      >recovery-sha256.txt
    sha256sum -c recovery-sha256.txt >recovery-verify.txt
  )
  metadata_tmp="$stage_dir/metadata.json"
  jq -n \
    --arg createdAt "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg runId "$(basename "$run_dir")" \
    --arg runDirectory "$run_dir" \
    --arg repository "$repo_dir" \
    --arg branch "$branch" \
    --arg head "$head" \
    --arg driveTarget "${MODELPRINT_DRIVE_MIRROR:-/content/drive/MyDrive/aidataapps/lab02/runs}/$(basename "$run_dir")" \
    '{schemaVersion:1,createdAt:$createdAt,runId:$runId,runDirectory:$runDirectory,repository:$repository,branch:$branch,head:$head,driveTarget:$driveTarget,autoCommit:false}' \
    >"$metadata_tmp"

  for file in "$stage_dir"/*; do
    mv -f -- "$file" "$latest_dir/$(basename "$file")"
  done
  rmdir "$stage_dir"
  stage_dir=""
}

sync_coordination_docs() {
  local drive_root lab_drive_root stamp sync_stage file
  drive_root="${AIDATAAPPS_DRIVE_ROOT:-/content/drive/MyDrive/aidataapps}"
  lab_drive_root="$drive_root/lab02"
  if [[ ! -d /content/drive/MyDrive ]]; then
    echo "Drive is not mounted at /content/drive/MyDrive" >&2
    return 3
  fi
  mkdir -p "$drive_root" "$lab_drive_root"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  sync_stage="$(mktemp -d "$drive_root/.coordination-${stamp}.XXXXXX")"
  cp "$repo_dir/aidataapps/resume.md" "$sync_stage/resume.md"
  cp "$repo_dir/aidataapps/inprogress_lab2.md" "$sync_stage/inprogress_lab2.md"
  if [[ -f /content/handoff.md ]]; then
    cp /content/handoff.md "$sync_stage/handoff_lab2.md"
  fi
  jq -n \
    --arg syncedAt "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg branch "$branch" \
    --arg head "$(git -C "$repo_dir" rev-parse HEAD)" \
    '{schemaVersion:1,syncedAt:$syncedAt,branch:$branch,head:$head}' \
    >"$sync_stage/coordination-sync-lab2.json"
  chmod 0644 "$sync_stage"/*

  for file in "$sync_stage"/*; do
    case "$(basename "$file")" in
      resume.md|inprogress_lab2.md|coordination-sync-lab2.json)
        mv -f -- "$file" "$drive_root/$(basename "$file")"
        ;;
      handoff_lab2.md)
        mv -f -- "$file" "$lab_drive_root/handoff.md"
        ;;
    esac
  done
  rmdir "$sync_stage"
}

checkpoint_cycle() {
  local failures=0
  log "BEGIN checkpoint pid=$$ branch=$branch head=$(git -C "$repo_dir" rev-parse --short HEAD)"

  run_logged "git-push" git -C "$repo_dir" push origin "$branch" || failures=$((failures + 1))
  run_logged "coordination-docs" sync_coordination_docs || failures=$((failures + 1))
  run_logged "database-backup" npm run db:backup || failures=$((failures + 1))
  run_logged "recovery-snapshot" write_recovery_snapshot || failures=$((failures + 1))
  run_logged "drive-mirror" npm run run:mirror || failures=$((failures + 1))

  log "END checkpoint failures=$failures next_seconds=$interval_seconds"
  return "$failures"
}

if [[ "$mode" == "once" ]]; then
  checkpoint_cycle
  exit $?
fi

log "WATCHDOG started interval_seconds=$interval_seconds pid=$$"
while [[ ! -f "$checkpoint_dir/STOP" ]]; do
  checkpoint_cycle || true
  sleep "$interval_seconds" &
  wait $!
done
log "WATCHDOG stopped by $checkpoint_dir/STOP"
