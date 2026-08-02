import json
from pathlib import Path

import pytest
import torch

from jspace_gemma.calibration_audit import audit_completed_checkpoint
from jspace_gemma.manifests import file_sha256, object_sha256


def _checkpoint(tmp_path: Path) -> Path:
    root = tmp_path / "checkpoint"
    cells = root / "cells"
    raw_root = root / "raw"
    cells.mkdir(parents=True)
    raw_root.mkdir()
    commit = "a" * 40
    header = {
        "schema_version": 1,
        "evidence_id": "gm-test",
        "config_sha256": "config",
        "code_commit": commit,
        "model_id": "control",
        "model_revision": "b" * 40,
        "environment_sha256": "environment",
    }
    input_manifest = {
        "evidence_id": header["evidence_id"],
        "code_commit": commit,
        "model_id": header["model_id"],
        "model_revision": header["model_revision"],
        "environment_sha256": header["environment_sha256"],
        "config": {"sha256": header["config_sha256"]},
    }
    (root / "input_manifest.json").write_text(json.dumps(input_manifest))
    cell_id = "cell-0"
    provenance = {
        "cell_id": cell_id,
        "code_commit": commit,
        "config_sha256": header["config_sha256"],
        "environment_sha256": header["environment_sha256"],
        "model_id": header["model_id"],
        "model_revision": header["model_revision"],
    }
    metrics_path = cells / f"{cell_id}.json"
    metrics_path.write_text(json.dumps({"schema_version": 2, "rows": [provenance]}))
    raw_path = raw_root / f"{cell_id}.pt"
    torch.save(
        {
            "schema_version": 2,
            "cell_id": cell_id,
            "metadata": provenance,
            "records": [{"value": torch.ones(2)}],
        },
        raw_path,
    )
    payload = {
        "completed_cells": {
            cell_id: {
                "metrics": {"path": str(metrics_path), "sha256": file_sha256(metrics_path)},
                "raw": {"path": str(raw_path), "sha256": file_sha256(raw_path)},
                "runtime_seconds": 1.25,
            }
        },
        "parity": {"p:L0": {"ok": True}},
        "wrong_hook": {"relative_l2_error": 0.5},
    }
    state = {
        "schema_version": 1,
        "header": header,
        "payload": payload,
        "payload_sha256": object_sha256(payload),
    }
    (root / "state.json").write_text(json.dumps(state))
    return root


def test_complete_checkpoint_audit_binds_rows_raw_and_provenance(tmp_path):
    root = _checkpoint(tmp_path)
    audit = audit_completed_checkpoint(
        root,
        expected_evidence_id="gm-test",
        expected_compute_commit="a" * 40,
        expected_cells=1,
        expected_rows_per_cell=1,
        expected_parity_rows=1,
    )
    assert audit["completed_cells"] == 1
    assert audit["rows"] == 1
    assert len(audit["inventory"]) == 2


def test_complete_checkpoint_audit_rejects_hash_drift(tmp_path):
    root = _checkpoint(tmp_path)
    (root / "cells/cell-0.json").write_text("{}")
    with pytest.raises(RuntimeError, match="checkpoint hash drift"):
        audit_completed_checkpoint(
            root,
            expected_evidence_id="gm-test",
            expected_compute_commit="a" * 40,
            expected_cells=1,
            expected_rows_per_cell=1,
            expected_parity_rows=1,
        )
