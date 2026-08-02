import json
from pathlib import Path

import pytest
import torch

from jspace_gemma.calibration_audit import audit_completed_checkpoint
from jspace_gemma.experiments.gm_exact_transport_gate import _aggregate
from jspace_gemma.experiments.gm_finalize_olmo_calibration import _parquet_frame
from jspace_gemma.experiments.gm_freeze_g1_thresholds import (
    _smallest_measurable,
    _tangent_pass,
)
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
        "implementation_sha256": "c" * 64,
        "transport_implementation_sha256": "d" * 64,
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


def test_aggregate_casts_pandas_group_keys_to_json_native_types():
    rows = []
    for epsilon, error in ((0.01, 0.2), (0.02, 0.25), (0.05, 0.3)):
        rows.append(
            {
                "faithful_delivery": True,
                "prompt_id": "p",
                "source_layer": 4,
                "perturbation_mode": "single_position",
                "direction_id": "d",
                "desired_relative_epsilon": epsilon,
                "tangent_cosine": 0.9,
                "tangent_relative_error": error,
                "central_tangent_relative_error": error,
                "homogeneity_defect": error,
                "homogeneity_nonlinear_remainder_defect": error,
                "odd_symmetry_defect": error,
                "odd_nonlinear_remainder_defect": error,
                "response_snr": 5.0,
                "backend_parity_relative_error": 0.0,
            }
        )
    aggregate = _aggregate(
        rows,
        [{"relative_l2_error": 0.0}],
        {"relative_l2_error": 1.0},
    )
    json.dumps(aggregate)
    fit = aggregate["floor_curvature_fits_unfiltered_pre_snr_threshold"][0]
    assert type(fit["source_layer"]) is int


def test_parquet_frame_preserves_mixed_source_position_type_explicitly():
    frame = _parquet_frame(
        [
            {"source_position": -1, "value": 1.0},
            {"source_position": "all_valid", "value": 2.0},
        ]
    )
    assert frame["source_position"].tolist() == ["-1", "all_valid"]
    assert frame["source_position_runtime_type"].tolist() == ["int", "str"]


def test_threshold_selector_uses_smallest_delivered_high_snr_row():
    import pandas as pd

    frame = pd.DataFrame(
        [
            {
                "prompt_id": "p",
                "source_layer": 60,
                "direction_id": "d",
                "perturbation_mode": "single_position",
                "desired_relative_epsilon": 0.05,
                "faithful_delivery": True,
                "response_snr": 10.0,
                "tangent_cosine": 0.99,
                "tangent_relative_error": 0.1,
                "central_tangent_relative_error": 0.05,
            },
            {
                "prompt_id": "p",
                "source_layer": 60,
                "direction_id": "d",
                "perturbation_mode": "single_position",
                "desired_relative_epsilon": 0.10,
                "faithful_delivery": True,
                "response_snr": 25.0,
                "tangent_cosine": 0.99,
                "tangent_relative_error": 0.1,
                "central_tangent_relative_error": 0.05,
            },
        ]
    )
    selected = _smallest_measurable(
        frame, layers=[60], mode="single_position", snr_floor=20.0
    )
    assert selected["desired_relative_epsilon"].tolist() == [0.10]
    passed = _tangent_pass(
        selected,
        {
            "tangent_cosine_floor": 0.98,
            "tangent_relative_error_ceiling": 0.20,
            "central_tangent_relative_error_ceiling": 0.10,
        },
    )
    assert passed.tolist() == [True]
