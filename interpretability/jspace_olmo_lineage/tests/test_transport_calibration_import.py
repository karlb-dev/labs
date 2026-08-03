import json

import pytest

from jspace_olmo_lineage.experiments.transport_calibration_import import (
    ImportBoundaryError,
    MODEL_SCOPE_KEY,
    extract_olmo_license,
)


def _fixture(tmp_path):
    threshold = tmp_path / "backend_ceiling_frozen.json"
    threshold.write_text(json.dumps({
        "licensed_ceilings": {
            "by_model": {MODEL_SCOPE_KEY: 0.125},
            "pooled": 999.0,
        }
    }))
    from jspace_olmo_lineage.manifests import file_sha256

    output = {"path": str(threshold), "sha256": file_sha256(threshold)}
    event = {
        "route": "benign_scheduling_floor",
        "no_target_read_assertion": True,
        "licensed_ceilings": {
            "by_model": {MODEL_SCOPE_KEY: 0.125},
            "pooled": 999.0,
        },
        "inputs": {"threshold_sha256_pre_registry": output["sha256"]},
    }
    return event, [output]


def test_extract_license_uses_only_olmo_specific_ceiling(tmp_path):
    event, outputs = _fixture(tmp_path)
    result = extract_olmo_license(event, outputs)
    assert result["backend_relative_error_ceiling"] == 0.125
    assert result["model_scope_key"] == MODEL_SCOPE_KEY
    assert result["pooled_ceiling_imported"] is False
    assert result["pooled_ceiling_use_forbidden"] is True


def test_extract_license_rejects_nonlicensing_route(tmp_path):
    event, outputs = _fixture(tmp_path)
    event["route"] = "ambiguous_backend_path"
    with pytest.raises(ImportBoundaryError):
        extract_olmo_license(event, outputs)


def test_extract_license_rejects_artifact_registry_disagreement(tmp_path):
    event, outputs = _fixture(tmp_path)
    event["licensed_ceilings"]["by_model"][MODEL_SCOPE_KEY] = 0.25
    with pytest.raises(ImportBoundaryError):
        extract_olmo_license(event, outputs)
