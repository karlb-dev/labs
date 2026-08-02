import json
from pathlib import Path

import pytest
import yaml

from jspace_gemma.experiments.gm_stage_gemma import (
    EVIDENCE_ID,
    REPO_ID,
    REVISION,
    require_gemma_staging_gate,
)
from jspace_gemma.manifests import file_sha256


def _gate_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    thresholds = tmp_path / "gm_g1_thresholds_frozen.yaml"
    thresholds.write_text("schema_version: 1\nprimary: frozen\n")
    artifact = tmp_path / "positive_control.json"
    artifact.write_text('{"positive_control_pass":true}\n')
    threshold_sha = file_sha256(thresholds)
    artifact_sha = file_sha256(artifact)
    config = tmp_path / "gm_g1_design.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "gemma_target": {"model_id": REPO_ID, "revision": REVISION}
                },
                "threshold_calibration": {
                    "status": "FROZEN_PRE_GEMMA_REGISTERED",
                    "gemma_execution_allowed": True,
                    "threshold_evidence_id": EVIDENCE_ID,
                    "frozen_threshold_file": (
                        "repo://interpretability/jspace_gemma/configs/"
                        "gm_g1_thresholds_frozen.yaml"
                    ),
                    "positive_control_artifact_sha256": artifact_sha,
                },
            }
        )
    )
    registry = tmp_path / "evidence_events.jsonl"
    registry.write_text(
        json.dumps(
            {
                "event": "evidence_created",
                "evidence_id": EVIDENCE_ID,
                "tier": "methods",
                "code_commit": "7f6a36e" + "0" * 33,
                "positive_control_pass": True,
                "thresholds_frozen_before_target": True,
                "target_model_opened": False,
                "inputs": {"threshold_config_sha256": threshold_sha},
                "outputs": [
                    {"path": str(thresholds), "sha256": threshold_sha},
                    {"path": str(artifact), "sha256": artifact_sha},
                ],
            }
        )
        + "\n"
    )
    return config, thresholds, registry


def test_gemma_staging_gate_binds_control_thresholds_and_unopened_target(tmp_path):
    config, thresholds, registry = _gate_fixture(tmp_path)
    result = require_gemma_staging_gate(
        config_path=config,
        thresholds_path=thresholds,
        registry_path=registry,
    )
    assert result["gemma_execution_allowed"] is True
    assert result["target_model_opened_by_threshold_event"] is False
    assert result["threshold_config_sha256"] == file_sha256(thresholds)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("positive_control_pass", False, "absent, non-live, or failing"),
        ("target_model_opened", True, "does not preserve the target firewall"),
    ],
)
def test_gemma_staging_gate_rejects_failed_or_target_open_event(
    tmp_path, field, value, message
):
    config, thresholds, registry = _gate_fixture(tmp_path)
    event = json.loads(registry.read_text())
    event[field] = value
    registry.write_text(json.dumps(event) + "\n")
    with pytest.raises(RuntimeError, match=message):
        require_gemma_staging_gate(
            config_path=config,
            thresholds_path=thresholds,
            registry_path=registry,
        )
