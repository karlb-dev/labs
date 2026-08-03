"""Register the one-field G2.1 architecture correction before any data row."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from jspace_gemma.manifests import (
    atomic_json,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from jspace_gemma.paths import PACKAGE_ROOT, run_root
from jspace_gemma.registry import append_event, read_events, resolve


BRANCH = "interp_jspace_gemma_transport_2"
FOUNDATION_ID = "gm2-foundation-v1"
CONFIG = PACKAGE_ROOT / "configs/gm2_backend_parity_calibration.yaml"
STUDY1_EXECUTION = PACKAGE_ROOT / "configs/gm_g1_stage1_execution.yaml"
PROTOCOL = PACKAGE_ROOT / "protocol/G2_PRE_DATA_ARCHITECTURE_CORRECTION.md"
MODEL_KEY = "gemma4_31b"
EXPECTED_TEXT_TYPE = "gemma4_text"
EXPECTED_CONFIG_SHA = "e967dd38bc5cfd38bd09a995a7bf4a754075df2b46aba68f7fbb5a791e6d8dd1"


def main() -> None:
    git = require_clean_tree(branch=BRANCH)
    root = run_root()
    config = yaml.safe_load(CONFIG.read_text())
    if root.resolve() != Path(config["run_root"]).resolve():
        raise RuntimeError("study-2 run root differs from corrected config")
    if config["models"][MODEL_KEY]["expected_text_model_type"] != EXPECTED_TEXT_TYPE:
        raise RuntimeError("corrected Gemma text-model label is absent")
    study1 = yaml.safe_load(STUDY1_EXECUTION.read_text())
    if study1["model"]["expected_text_model_type"] != EXPECTED_TEXT_TYPE:
        raise RuntimeError("study-1 architecture contract does not support correction")

    raw_state = root / "raw/gm2-backend-parity-calibration-v1/raw_rows_state.json"
    if raw_state.exists():
        raise RuntimeError("pre-data correction refused because a G2.1 raw state exists")
    threshold = root / "derived/gm2-backend-parity-calibration-v1/backend_ceiling_frozen.json"
    if threshold.exists():
        raise RuntimeError("pre-data correction refused after a G2.1 ceiling")

    backup_root = root / "backups/g2_predata_arch_correction"
    staging_backup = backup_root / "gm2_snapshot_gemma4_31b_pre_correction.json"
    failed_log = backup_root / "gm2_run_gemma4_31b_failed_arch_lock.log"
    for path in (staging_backup, failed_log):
        if not path.is_file():
            raise RuntimeError(f"required forensic backup is absent: {path}")
    staged = json.loads(staging_backup.read_text())
    snapshot_config = Path(staged["snapshot"]) / "config.json"
    if file_sha256(snapshot_config) != EXPECTED_CONFIG_SHA:
        raise RuntimeError("exact staged Gemma config hash differs")
    checkpoint = json.loads(snapshot_config.read_text())
    if (
        checkpoint.get("model_type") != "gemma4"
        or checkpoint.get("text_config", {}).get("model_type") != EXPECTED_TEXT_TYPE
        or int(checkpoint.get("text_config", {}).get("num_hidden_layers", -1)) != 60
    ):
        raise RuntimeError("exact checkpoint architecture does not support correction")
    log_text = failed_log.read_text(errors="replace")
    if "loaded model violates frozen architecture lock" not in log_text:
        raise RuntimeError("failed-load forensic log does not contain the static gate stop")

    foundation = resolve(FOUNDATION_ID)
    old_config_sha = foundation.get("inputs", {}).get("calibration_config_sha256")
    if not foundation["live"] or old_config_sha != staged.get("config_sha256"):
        raise RuntimeError("foundation/original staging config binding is incompatible")
    if foundation.get("model_outcome_opened") is not False:
        raise RuntimeError("foundation did not preserve the pre-outcome boundary")
    prior_corrections = [
        row
        for row in read_events()
        if row["evidence_id"] == FOUNDATION_ID
        and row["event"] == "evidence_corrected"
        and row.get("correction_kind") == "predata_architecture_label"
    ]
    if prior_corrections:
        raise RuntimeError("pre-data architecture correction is already registered")

    artifact = {
        "schema_version": 1,
        "evidence_id": FOUNDATION_ID,
        "correction_kind": "predata_architecture_label",
        "tier": "methods",
        "code_commit": git["code_commit"],
        "old_config_sha256": old_config_sha,
        "corrected_config": {"path": str(CONFIG), "sha256": file_sha256(CONFIG)},
        "field": "models.gemma4_31b.expected_text_model_type",
        "old_value": "gemma3_text",
        "corrected_value": EXPECTED_TEXT_TYPE,
        "exact_snapshot_config": {
            "path": str(snapshot_config),
            "sha256": file_sha256(snapshot_config),
            "outer_model_type": checkpoint["model_type"],
            "text_model_type": checkpoint["text_config"]["model_type"],
            "decoder_layers": checkpoint["text_config"]["num_hidden_layers"],
        },
        "study1_execution": {
            "path": str(STUDY1_EXECUTION),
            "sha256": file_sha256(STUDY1_EXECUTION),
            "expected_text_model_type": study1["model"]["expected_text_model_type"],
        },
        "protocol": {"path": str(PROTOCOL), "sha256": file_sha256(PROTOCOL)},
        "forensic_backups": [
            {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in (staging_backup, failed_log)
        ],
        "predata_assertions": {
            "raw_row_state_exists": False,
            "exact_jvp_pair_executed": False,
            "model_outcome_created": False,
            "scientific_grid_changed": False,
            "threshold_or_router_changed": False,
            "stage1_target_opened": False,
        },
    }
    artifact["payload_sha256"] = object_sha256(artifact)
    output = root / "manifests/gm2_predata_architecture_correction.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    atomic_json(output, artifact)
    event = append_event(
        {
            "event": "evidence_corrected",
            "evidence_id": FOUNDATION_ID,
            "reason": (
                "Pre-data repair of one static Gemma text model-type label after "
                "the exact checkpoint stopped at the architecture gate; no JVP "
                "or model outcome existed."
            ),
            "corrected_fields": {
                "calibration_config_sha256": file_sha256(CONFIG),
                "architecture_contract_correction_artifact": {
                    "path": str(output),
                    "sha256": file_sha256(output),
                },
                "model_outcome_opened": False,
            },
            "correction_kind": "predata_architecture_label",
            "correction_code_commit": git["code_commit"],
            "correction_artifact": {
                "path": str(output),
                "sha256": file_sha256(output),
            },
            "old_config_sha256": old_config_sha,
            "corrected_config_sha256": file_sha256(CONFIG),
            "model_outcome_opened": False,
            "stage1_target_opened": False,
        }
    )
    print(
        json.dumps(
            {
                "artifact": str(output),
                "artifact_sha256": file_sha256(output),
                "event": event,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
