import hashlib
import json
from pathlib import Path

import pytest
import yaml
from jspace_gemma.experiments.gm2_study2_release import (
    _flatten_admissions,
    registry_prefix_record,
    source_artifact_records,
    verify_bundle_source,
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
    assert config["result_summary"]["historical_all_slot_relative_error_exact"] == (
        "0.0024581113830208778"
    )
    assert config["result_summary"]["license_branch"] == (
        "branch_1_relicense_without_recompute"
    )
    assert config["result_summary"]["g2_2_model_compute_performed"] is False
    assert config["partial_statuses"]["intervention"] == "not-opened"
    assert config["partial_statuses"]["study1_blocker_record"].startswith(
        "preserved-immutable"
    )


def test_rendered_bundle_and_registry_snapshot_are_exact():
    release = ROOT / "release"
    expected = {
        "IMPORT_BUNDLE_SIDELINES2.json": (
            "9ef48b8ab1d99d52a756ddea1e285a9d61e781fd054cbf92702bfe81be56f5b0"
        ),
        "IMPORT_BUNDLE_SIDELINES2.md": (
            "547da552fee0057fa304b1dffe657eb269315007073b0a5c3cbb29c96577b315"
        ),
        "evidence_events_prefix_sidelines2.jsonl": (
            "2a144bcf0e7be0ac4307f7e2a2984c1879340b9a7e9278d10143d122a14fd30a"
        ),
    }
    for name, digest in expected.items():
        assert hashlib.sha256((release / name).read_bytes()).hexdigest() == digest
    verification = verify_bundle_source(CONFIG)
    assert verification["ok"] is True
    assert verification["admitted_events"] == 8
    assert verification["release_artifacts"] == 13


def test_terminal_study2_release_event_is_methods_only_and_partial_safe():
    rows = [
        json.loads(line)
        for line in (ROOT / "reports/evidence_events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    events = [
        row
        for row in rows
        if row["event"] == "evidence_created"
        and row["evidence_id"] == "gm2-sidelines2-import-bundle-v1"
    ]
    assert len(events) == 1
    event = events[0]
    assert event["code_commit"] == "d7bc87e47480513a88bfbf64c6ec683c79f5932f"
    assert event["tier"] == "methods"
    assert event["verdict"] == "complete-mandatory-partial-conditional"
    assert event["mandatory_stages_complete"] is True
    assert event["partial_bundle"] is True
    assert event["interventions_opened"] is False
    assert event["mechanism_claim_opened"] is False
    assert event["workspace_claim_opened"] is False
    assert event["confirmatory_cell_opened"] is False
    assert event["selected_branch"] == "branch_1_relicense_without_recompute"
    assert len(event["outputs"]) == 19
    output_hashes = {row["sha256"] for row in event["outputs"]}
    assert {
        "9ef48b8ab1d99d52a756ddea1e285a9d61e781fd054cbf92702bfe81be56f5b0",
        "547da552fee0057fa304b1dffe657eb269315007073b0a5c3cbb29c96577b315",
        "2a144bcf0e7be0ac4307f7e2a2984c1879340b9a7e9278d10143d122a14fd30a",
    } <= output_hashes
