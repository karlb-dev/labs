#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
run_root="/content/drive/MyDrive/interpret/special-lab-1/olmo_lineage_2_20260803"
log_dir="${run_root}/logs"
mkdir -p "${log_dir}"
export JSPACE_OLMO_RUN_ROOT="${run_root}"
export JSPACE_OLMO_LOCAL_WORK="/content/olmo_lineage_work"

cd "${repo_root}"
python -m jspace_olmo_lineage.experiments.transport_validation_analysis \
  --config interpretability/jspaces/sidelines/olmo/configs/ol2_transport_validation.yaml \
  2>&1 | tee -a "${log_dir}/transport_joint_analysis_producer.log"
