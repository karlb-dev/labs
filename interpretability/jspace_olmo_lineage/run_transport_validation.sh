#!/usr/bin/env bash
set -euo pipefail

model="${1:?usage: run_transport_validation.sh base|olmo31_think stage|preflight|run|finalize|register}"
phase="${2:?usage: run_transport_validation.sh base|olmo31_think stage|preflight|run|finalize|register}"
case "${model}" in
  base|olmo31_think) ;;
  *) echo "invalid H6 model: ${model}" >&2; exit 2 ;;
esac
case "${phase}" in
  stage|preflight|run|finalize|register) ;;
  *) echo "invalid H6 phase: ${phase}" >&2; exit 2 ;;
esac

repo_root="$(git rev-parse --show-toplevel)"
run_root="/content/drive/MyDrive/interpret/special-lab-1/olmo_lineage_2_20260803"
log_dir="${run_root}/logs"
mkdir -p "${log_dir}"
export JSPACE_OLMO_RUN_ROOT="${run_root}"
export JSPACE_OLMO_LOCAL_WORK="/content/olmo_lineage_work"
export HF_HUB_CACHE="/content/hf_local"
export TOKENIZERS_PARALLELISM=false

producer_log="${log_dir}/${model}_transport_${phase}_producer.log"
watchdog_log="${log_dir}/${model}_transport_${phase}_watchdog.log"
producer_pid_file="${log_dir}/${model}_transport_${phase}.pid"

watchdog_pid=""
cleanup() {
  if [[ -n "${watchdog_pid}" ]]; then
    kill "${watchdog_pid}" 2>/dev/null || true
    wait "${watchdog_pid}" 2>/dev/null || true
  fi
  rm -f "${producer_pid_file}"
}
trap cleanup EXIT INT TERM

(
  while true; do
    timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    gpu="$(nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>&1 || true)"
    disk="$(df -P /content | awk 'NR==2 {print $4}')"
    printf '%s pid=%s gpu=%s disk_kib_free=%s\n' "${timestamp}" "$$" "${gpu}" "${disk}" >> "${watchdog_log}"
    sleep 30
  done
) &
watchdog_pid="$!"
printf '%s\n' "$$" > "${producer_pid_file}"

cd "${repo_root}"
python -m jspace_olmo_lineage.experiments.transport_validation \
  --phase "${phase}" --model "${model}" 2>&1 | tee -a "${producer_log}"
