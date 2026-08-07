import json
from pathlib import Path

import pandas as pd
import pytest

from jspace_phase4.manifests import file_sha256, object_sha256


ALIASES = [
    " amber", " blue", " coral", " green",
    " ivory", " lilac", " ochre", " silver",
]


def _rows():
    scores = json.dumps(
        {alias: -float(index) for index, alias in enumerate(ALIASES)},
        sort_keys=True,
    )
    rows = []
    for family_index in range(24):
        family = f"family-{family_index:02d}"
        for seed in range(8):
            for load in ("low", "high"):
                rows.append({
                    "item_id": f"{family}:{seed}:{load}",
                    "canonical_family": family,
                    "item_seed": seed,
                    "load": load,
                    "correct": True,
                    "baseline_answer_margin": 2.0,
                    "true_answer_sequence_lp": -1.0,
                    "candidate_scores_json": scores,
                    "prompt_token_count": 100 + family_index,
                    "answer_token_count": 1,
                })
    return rows


def _selection():
    return {
        "loads": ["low", "high"],
        "expected_families": 24,
        "expected_seeds_per_family": 8,
        "expected_rows_per_model": 384,
    }


def _guard():
    return {
        "baseline_accuracy_floor": 0.70,
        "low_high_accuracy_difference_sesoi": 0.08,
        "equivalence_interval_level": 0.90,
        "family_bootstrap_draws": 200,
        "family_bootstrap_seed": 20260801,
        "family_capability_accuracy_floor_by_load": 0.70,
        "minimum_joint_common_families": 20,
    }


def _write_envelope(path: Path, payload: dict, **extra):
    path.write_text(json.dumps({
        "schema_version": 1,
        "payload": payload,
        "payload_sha256": object_sha256(payload),
        **extra,
    }))


def test_imported_model_table_is_replayed_exactly(tmp_path):
    from jspace_phase4.experiments.p4_bank_w_capability import (
        analyze_model_rows,
    )
    from jspace_phase4.experiments.p4_bank_w_joint_imported import (
        _finite_row_audit,
        verify_model_artifacts,
    )

    rows = _rows()
    selection = _selection()
    guard = _guard()
    source_hash = "f" * 64
    analysis = analyze_model_rows(
        rows, selection=selection, guard=guard)
    analysis["side_track_finite_gate"] = _finite_row_audit(
        rows, aliases=ALIASES)
    analysis["phase4_function_source_sha256"] = source_hash
    evidence_id = "ol-test-capability-v1"
    manifest_path = tmp_path / "input_manifest.json"
    rows_path = tmp_path / "rows.parquet"
    result_path = tmp_path / "result.json"
    manifest = {
        "experiment_id": evidence_id,
        "model_id": "org/model",
        "model_revision": "a" * 40,
        "tokenizer_manifest_sha256": "b" * 64,
        "bank_sha256": "c" * 64,
        "partition_sha256": "d" * 64,
        "scoring_spec_sha256": "e" * 64,
    }
    _write_envelope(manifest_path, manifest)
    pd.DataFrame(rows).to_parquet(rows_path, index=False)
    result = {
        "model": {"model_id": "org/model", "revision": "a" * 40},
        "model_slug": "model",
        "tokenizer_manifest_sha256": "b" * 64,
        "source_phase4_protocol_sha256": "1" * 64,
        "interventions_opened": False,
        "answer_token_ids": {
            alias: [index + 1] for index, alias in enumerate(ALIASES)},
        "analysis": analysis,
    }
    _write_envelope(result_path, result, provenance={
        "dirty_tree": False,
        "evidence_id": evidence_id,
        "input_manifest_sha256": object_sha256(manifest),
    })
    specification = {
        "slug": "model",
        "source": "imported",
        "evidence_id": evidence_id,
        "model_id": "org/model",
        "model_revision": "a" * 40,
        "tokenizer_manifest_sha256": "b" * 64,
        "input_manifest_path": str(manifest_path),
        "input_manifest_sha256": file_sha256(manifest_path),
        "rows_path": str(rows_path),
        "rows_sha256": file_sha256(rows_path),
        "result_path": str(result_path),
        "result_sha256": file_sha256(result_path),
    }
    inventory = {
        specification[f"{name}_path"]: specification[f"{name}_sha256"]
        for name in ("input_manifest", "rows", "result")
    }
    verified = verify_model_artifacts(
        specification, inventory=inventory, selection=selection,
        guard=guard, aliases=ALIASES,
        bank_contract={
            "development_rows_sha256": "c" * 64,
            "source_partition_payload_sha256": "d" * 64,
            "scoring_spec_sha256": "e" * 64,
            "phase4_function_source_sha256": source_hash,
            "source_protocol_sha256": "1" * 64,
        },
    )
    assert verified["analysis"]["independently_capability_eligible"]
    assert verified["finite_audit"]["n_numeric_values_checked"] == 4608
    assert verified["intervention_columns_absent"] is True


def test_finite_audit_rejects_incomplete_candidate_vector():
    from jspace_phase4.experiments.p4_bank_w_joint_imported import (
        JointImportError,
        _finite_row_audit,
    )

    row = _rows()[0]
    row["candidate_scores_json"] = json.dumps({" amber": -1.0})
    with pytest.raises(JointImportError, match="incomplete candidate"):
        _finite_row_audit([row], aliases=ALIASES)


def test_joint_replay_keeps_every_passing_model_when_support_fails(
        monkeypatch):
    import jspace_phase4.experiments.p4_bank_w_joint_imported as module

    monkeypatch.setattr(module, "validate_import_contract", lambda *a, **k: {
        "audit": {}, "output_inventory": {}})
    monkeypatch.setattr(module, "verify_bank_v3_compatibility", lambda *a: {})
    monkeypatch.setattr(module, "resolve", lambda evidence_id: {
        "live": True, "evidence_id": evidence_id, "outputs": []})
    capable = {
        "a": range(24),
        "b": range(20),
        "c": range(16),
    }

    def verify(specification, **kwargs):
        ids = [f"family-{index:02d}" for index in capable[
            specification["slug"]]]
        return {
            "analysis": {
                "independently_capability_eligible": True,
                "capable_family_ids": ids,
                "n_capable_families": len(ids),
                "load_summaries": {
                    "low": {"accuracy": 0.8},
                    "high": {"accuracy": 0.8},
                },
            }
        }

    monkeypatch.setattr(module, "verify_model_artifacts", verify)
    config = {
        "evidence_id": "p4-joint-test-v1",
        "import_contract": {},
        "bank_contract": {},
        "selection": {},
        "capability_guard": {"minimum_joint_common_families": 20},
        "answer_aliases": ALIASES,
        "models": [
            {"slug": "a", "source": "imported"},
            {"slug": "b", "source": "imported"},
            {"slug": "c", "source": "native", "evidence_id": "p4-c"},
        ],
        "claim_boundary": "baseline only",
    }
    result = module.build_joint_result(
        config, require_registered_import=False)
    assert result["joint"]["would_be_primary_model_set"] == ["a", "b", "c"]
    assert result["joint"]["primary_model_set"] == []
    assert result["joint"]["n_joint_common_capable_families"] == 16
    assert result["p4p3_disposition"].startswith("BLOCKED")
