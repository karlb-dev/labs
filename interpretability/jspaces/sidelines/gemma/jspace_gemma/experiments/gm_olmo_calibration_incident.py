"""Register the complete-cell/post-summary OLMo serialization incident."""
from __future__ import annotations

import json
import time
from pathlib import Path

from jspace_gemma.calibration_audit import audit_completed_checkpoint
from jspace_gemma.manifests import atomic_json, file_sha256, require_clean_tree
from jspace_gemma.paths import directory
from jspace_gemma.registry import create

EVIDENCE_ID = "gm-olmo-calibration-finalize-diagnostic-v1"
CALIBRATION_ID = "gm-jvp-olmo-calibration-v1"
COMPUTE_COMMIT = "06b2a3d2fbe42fd5f70abb121573b1e7a62b45ec"
STATE_SHA256 = "f696f28cecc44d3a3d925308dd10226f1f7fa84e09e6e63ff37913ea3960278c"
FULL_LOG_SHA256 = "28a6aecdff750821603e5355bf0776ff38bc069181ad83d6edf4677249225dfe"
CHECKPOINT_ROOT = (
    directory("metrics") / "olmo_control" / CALIBRATION_ID
)
FULL_LOG = directory("logs") / "gm_exact_transport_olmo_full_20260802.log"


def main() -> None:
    git = require_clean_tree()
    if file_sha256(FULL_LOG) != FULL_LOG_SHA256:
        raise RuntimeError("failed-run log hash drifted")
    log_text = FULL_LOG.read_text(errors="replace")
    required_markers = [
        "completed gm-p004-L60-uniform_valid",
        "total=56",
        "TypeError: Object of type int64 is not JSON serializable",
    ]
    if any(marker not in log_text for marker in required_markers):
        raise RuntimeError("failed-run log lacks the frozen incident markers")

    audit = audit_completed_checkpoint(
        CHECKPOINT_ROOT,
        expected_evidence_id=CALIBRATION_ID,
        expected_compute_commit=COMPUTE_COMMIT,
        expected_cells=56,
        expected_rows_per_cell=28,
        expected_parity_rows=28,
        inspect_raw_tensors=True,
    )
    if audit["state_sha256"] != STATE_SHA256:
        raise RuntimeError("complete calibration state hash drifted")
    forbidden = [
        CHECKPOINT_ROOT / "olmo_calibration_summary.json",
        CHECKPOINT_ROOT / "olmo_calibration_rows.parquet",
        CHECKPOINT_ROOT / "raw_inventory.json",
    ]
    if any(path.exists() for path in forbidden):
        raise RuntimeError("a final calibration output unexpectedly exists")

    manifest = {
        "schema_version": 1,
        "evidence_id": EVIDENCE_ID,
        "tier": "methods",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "diagnostic_code_commit": git["code_commit"],
        "compute_code_commit": COMPUTE_COMMIT,
        "calibration_evidence_id": CALIBRATION_ID,
        "failure_stage": "post_compute_summary_json_serialization",
        "exception": {
            "type": "TypeError",
            "message": "Object of type int64 is not JSON serializable",
            "offending_field": "floor_curvature_fits[*].source_layer",
            "offending_runtime_type": "numpy.int64",
        },
        "audit": {
            key: value
            for key, value in audit.items()
            if key not in {"all_rows", "inventory"}
        },
        "checkpoint_inventory": audit["inventory"],
        "failed_run_log": {
            "path": str(FULL_LOG),
            "sha256": FULL_LOG_SHA256,
        },
        "summary_created": False,
        "parquet_created": False,
        "inventory_created": False,
        "calibration_evidence_registered": False,
        "model_response_data_created": True,
        "interpreted_scientific_result": False,
        "control_model_opened": True,
        "target_model_opened": False,
        "recovery_contract": (
            "never rewrite or rerun the 56 complete cells; finalize only after "
            "verifying the frozen state, every cell/raw hash, and separate "
            "compute/finalization commits"
        ),
    }
    output = directory("manifests") / "gm_olmo_calibration_finalize_diagnostic_v1.json"
    atomic_json(output, manifest)
    create(
        EVIDENCE_ID,
        tier="methods",
        what=(
            "immutable audit of 56 complete OLMo calibration cells followed "
            "by a post-compute numpy.int64 summary-serialization failure"
        ),
        command="python -m jspace_gemma.experiments.gm_olmo_calibration_incident",
        outputs=[output],
        inputs={
            "compute_code_commit": COMPUTE_COMMIT,
            "state_sha256": STATE_SHA256,
            "checkpoint_inventory_sha256": audit["inventory_sha256"],
            "failed_run_log_sha256": FULL_LOG_SHA256,
        },
        model_response_data_created=True,
        interpreted_scientific_result=False,
        control_model_opened=True,
        target_model_opened=False,
    )
    print(json.dumps({"output": str(output), "sha256": file_sha256(output)}, indent=1))


if __name__ == "__main__":
    main()
