import hashlib
from pathlib import Path

import pytest
import yaml
from jspace_gemma.experiments.gm2_study2_release import (
    _flatten_admissions,
    registry_prefix_record,
    source_artifact_records,
    verify_registry_prefix,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/gm2_sidelines2_release.yaml"


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def test_registry_prefix_survives_append_and_rejects_mutation(tmp_path):
    source = ROOT / "reports/evidence_events.jsonl"
    copy = tmp_path / "events.jsonl"
    copy.write_bytes(source.read_bytes())
    record = registry_prefix_record(copy)
    with copy.open("ab") as handle:
        handle.write(b'{"future":"event"}\n')
    assert verify_registry_prefix(record, copy)["ok"] is True

    changed = bytearray(copy.read_bytes())
    changed[0] = ord("[")
    copy.write_bytes(changed)
    with pytest.raises(ValueError, match="prefix hash drift"):
        verify_registry_prefix(record, copy)


def test_release_config_pins_every_source_artifact():
    config = _config()
    records = source_artifact_records(config)
    assert len(records) == 13
    assert {row["role"] for row in records} == {
        row["role"] for row in config["source_artifacts"]
    }
    for row in records:
        path = ROOT.parents[1] / row["repo_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]


def test_release_admission_and_partial_contract_is_exact():
    config = _config()
    identifiers = _flatten_admissions(config["admitted_evidence"])
    assert len(identifiers) == 8
    assert identifiers[-1] == "gm2-stage1-relicense-v1"
    assert config["result_summary"]["calibration_route"] == "benign_scheduling_floor"
    assert config["result_summary"]["pooled_ceiling"] == 0.07870368901355948
    assert config["result_summary"]["license_branch"] == (
        "branch_1_relicense_without_recompute"
    )
    assert config["result_summary"]["g2_2_model_compute_performed"] is False
    assert config["partial_statuses"]["intervention"] == "not-opened"
    assert config["partial_statuses"]["study1_blocker_record"].startswith(
        "preserved-immutable"
    )
