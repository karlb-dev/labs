import hashlib
import json
from pathlib import Path

import yaml


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_blocked_release_freezes_the_registered_failure_prefix():
    root = _root()
    config = yaml.safe_load((root / "configs/gm_blocked_release.yaml").read_text())
    assert config["status"] == "FROZEN_METHODS_BLOCKER_RELEASE"
    assert config["evidence_id"] == "gm-state-of-record-v1"
    assert config["terminal_contract"] == {
        "status": "COMPLETE_METHODS_BLOCKER",
        "scientific_expansion_allowed": False,
        "mechanism_interpretation_allowed": False,
        "phase4_model_cell_opened": False,
        "independent_review_or_pi_signoff": False,
        "future_repair_requires_new_evidence_id": True,
        "preserve_failed_artifact": True,
        "merge_after_release_registration_and_verification": True,
        "unrun_by_hard_stop": [
            "G1_stage2", "G2", "G3", "G4", "G5", "G6", "G7", "G8"
        ],
    }
    prefix = config["source_registry_prefix"]
    registry = (root / "reports/evidence_events.jsonl").read_bytes()
    frozen = registry[: prefix["expected_bytes"]]
    assert len(frozen) == prefix["expected_bytes"]
    assert hashlib.sha256(frozen).hexdigest() == prefix["expected_sha256"]
    rows = [json.loads(line) for line in frozen.decode().splitlines() if line]
    assert len(rows) == prefix["expected_live_events"]
    assert rows[-1]["evidence_id"] == "gm-jvp-gemma-backend-parity-v1"
    assert rows[-1]["backend_parity_pass"] is False
    assert rows[-1]["stage1_mismatch_reproduced"] is True


def test_blocked_release_documents_enforce_the_methods_claim_boundary():
    root = _root()
    config = yaml.safe_load((root / "configs/gm_blocked_release.yaml").read_text())
    documents = {
        label: (root / relative).read_text()
        for label, relative in config["release_documents"].items()
    }
    assert "terminal methods blocker" in documents["state_of_record"]
    assert "G2 layer/sublayer localization" in documents["state_of_record"]
    assert "not licensed" in documents["claim_ledger"]
    assert "Finite differences are secants" in documents["transport_gate_protocol"]
    assert "methods-only" in documents["import_markdown"]
    output_names = set(config["drive_outputs"].values())
    assert output_names == {
        "GEMMA_TRANSPORT_STATE_OF_RECORD.md",
        "IMPORT_BUNDLE_PHASE4.json",
        "IMPORT_BUNDLE_PHASE4.md",
        "gemma_transport_inventory.json",
        "gemma_transport_environment_lock.json",
        "gemma_transport_claim_ledger.md",
        "TRANSPORT_GATE_PROTOCOL.md",
        "gemma_transport_release_manifest.json",
    }


def test_blocked_release_producer_is_model_free_and_idempotence_guarded():
    root = _root()
    source = (
        root
        / "jspace_gemma/experiments/gm_blocked_release.py"
    ).read_text()
    assert "from transformers" not in source
    assert "AutoModel" not in source
    assert "require_cuda" not in source
    assert "refusing to overwrite unregistered release outputs" in source
    assert "Gemma state-of-record evidence already exists" in source


def test_terminal_state_of_record_is_registered_methods_only():
    root = _root()
    rows = [
        json.loads(line)
        for line in (root / "reports/evidence_events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    events = [
        row
        for row in rows
        if row["event"] == "evidence_created"
        and row["evidence_id"] == "gm-state-of-record-v1"
    ]
    assert len(events) == 1
    event = events[0]
    assert event["code_commit"] == "b80004843a5bbe57536e4da18297f7c52cf201a3"
    assert event["terminal_status"] == "COMPLETE_METHODS_BLOCKER"
    assert event["methods_blocker"] is True
    assert event["scientific_expansion"] is False
    assert event["mechanism_interpretation_allowed"] is False
    assert event["phase4_import_tier"] == "methods_only"
    assert event["target_model_opened"] is False
    assert event["independent_review_or_pi_signoff"] is False
    outputs = {Path(row["path"]).name: row["sha256"] for row in event["outputs"]}
    assert outputs == {
        "GEMMA_TRANSPORT_STATE_OF_RECORD.md": (
            "0eeab02938c32839a8ca19e2446bfc3b69bfcb18769428afb6904133363fa344"
        ),
        "IMPORT_BUNDLE_PHASE4.json": (
            "005532754166644e42a369358565b9ce72235151e64559a9f12254d987ff7729"
        ),
        "IMPORT_BUNDLE_PHASE4.md": (
            "91bc1f1528a03c0c5c842c95bd5ab597d1bd5e6a577eec1b0662c0ed10daf74c"
        ),
        "gemma_transport_inventory.json": (
            "0744e32348f7c859475b4482fd9b38f83ef13925710dca8b0c0316cb9b372449"
        ),
        "gemma_transport_environment_lock.json": (
            "7eceeaa500001aa78cba9cac5ce181233414d2529a76267bf6ac94de74a0203d"
        ),
        "gemma_transport_claim_ledger.md": (
            "812a45a5044069baf41d8db276b1d59f0c89dd76acb790e9e0e6506c5cccd779"
        ),
        "TRANSPORT_GATE_PROTOCOL.md": (
            "f1d83baa36a41623ba4a5990cd83f09bf8e9e09690c1b95c5d5ce7e716e8b768"
        ),
        "gemma_transport_release_manifest.json": (
            "1f896c7029a2f4cee10378a95c71463d1346e2a500529101af80de7063c1e483"
        ),
    }
