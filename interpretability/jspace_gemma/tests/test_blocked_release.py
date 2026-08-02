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
