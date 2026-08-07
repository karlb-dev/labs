import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from jspace_olmo_lineage.compat import (
    CompatibilityError,
    analyze_model_rows,
    candidate_scores,
    validate_finite_rows,
    verify_sources,
)
from jspace_olmo_lineage.experiments.bank_w_capability import (
    _critical_source_conformance,
    _verify_bank_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/ol_bank_w_capability_v1.yaml"


def config():
    return yaml.safe_load(CONFIG_PATH.read_text())


def selection():
    return {
        "loads": ["low", "high"],
        "expected_families": 24,
        "expected_seeds_per_family": 8,
        "expected_rows_per_model": 384,
    }


def guard():
    return {
        "baseline_accuracy_floor": 0.70,
        "low_high_accuracy_difference_sesoi": 0.08,
        "equivalence_interval_level": 0.90,
        "family_bootstrap_draws": 2000,
        "family_bootstrap_seed": 20260801,
        "family_capability_accuracy_floor_by_load": 0.70,
        "minimum_joint_common_families": 20,
    }


def rows():
    result = []
    aliases = [" a", " b", " c", " d", " e", " f", " g", " h"]
    scores = json.dumps(
        {alias: -float(index) for index, alias in enumerate(aliases)},
        sort_keys=True,
    )
    for family_index in range(24):
        family = f"family-{family_index:02d}"
        for seed in range(8):
            for load in ("low", "high"):
                result.append({
                    "item_id": f"{family}:{seed}:{load}",
                    "canonical_family": family,
                    "item_seed": seed,
                    "load": load,
                    "correct": True,
                    "baseline_answer_margin": 1.0,
                    "true_answer_sequence_lp": -1.0,
                    "candidate_scores_json": scores,
                    "prompt_token_count": 80,
                    "answer_token_count": 1,
                })
    return result, aliases


def test_frozen_phase4_sources_and_protocol_are_exact():
    value = config()
    result = verify_sources(value["compatibility_sources"])
    assert result["scoring_spec"]["version"] == "p4-scoring-v1"
    conformance = _critical_source_conformance(value)
    assert conformance["all_pass"]
    assert all(row["exact"] for row in conformance["comparisons"].values())
    _, bank = _verify_bank_inputs(value)
    assert all(bank["checks"].values())


def test_frozen_registry_accepts_only_exact_append_only_extension(
        tmp_path, monkeypatch):
    from jspace_olmo_lineage.experiments import bank_w_capability as producer

    frozen = (
        b'{"event":"evidence_created","evidence_id":"frozen",'
        b'"schema_version":1}\n')
    appended = (
        b'{"event":"evidence_created","evidence_id":"later",'
        b'"schema_version":1}\n')
    registry = tmp_path / "events.jsonl"
    registry.write_bytes(frozen + appended)
    monkeypatch.setattr(producer, "resolve_uri", lambda _uri: registry)
    value = {"source_phase4": {
        "registry_uri": "repo://events.jsonl",
        "registry_sha256": hashlib.sha256(frozen).hexdigest(),
    }}

    events, record = producer._source_registry(value)
    assert [row["evidence_id"] for row in events] == ["frozen"]
    assert record["sha256"] == hashlib.sha256(frozen).hexdigest()
    assert record["bytes"] == len(frozen)
    assert record["append_only_extension"] is True
    assert record["appended_events"] == 1

    registry.write_bytes(frozen.replace(b"frozen", b"edited") + appended)
    with pytest.raises(RuntimeError, match="frozen Phase 4 source registry"):
        producer._source_registry(value)


def test_side_analysis_matches_phase4_and_adds_finite_gate():
    from jspace_phase4.experiments.p4_bank_w_capability import (
        analyze_model_rows as source_analyze,
    )

    values, aliases = rows()
    expected = source_analyze(
        copy.deepcopy(values), selection=selection(), guard=guard())
    observed = analyze_model_rows(
        copy.deepcopy(values), selection=selection(), guard=guard(),
        aliases=aliases)
    finite = observed.pop("side_track_finite_gate")
    observed.pop("phase4_function_source_sha256")
    assert observed == expected
    assert finite["all_rows_finite"]
    assert finite["n_rows"] == 384


def test_nonfinite_or_incomplete_candidate_vector_is_rejected():
    values, aliases = rows()
    values[0]["baseline_answer_margin"] = float("nan")
    with pytest.raises(CompatibilityError):
        validate_finite_rows(values, aliases=aliases, expected_rows=384)
    values, aliases = rows()
    values[0]["candidate_scores_json"] = json.dumps({" a": 0.0})
    with pytest.raises(CompatibilityError):
        validate_finite_rows(values, aliases=aliases, expected_rows=384)


def test_candidate_scores_use_all_answer_tokens_and_padding():
    class Session:
        def prompt_ids(self, prompt):
            assert prompt == "prompt"
            return torch.tensor([[1, 2]])

        def answer_ids(self, alias):
            return torch.tensor([{" a": [3], " b": [4, 5]}[alias]])

    class Model:
        def __call__(self, *, input_ids, attention_mask, use_cache):
            assert not use_cache
            logits = torch.zeros((*input_ids.shape, 8), dtype=torch.float32)
            for row in range(input_ids.shape[0]):
                for position in range(1, int(attention_mask[row].sum()) - 1):
                    target = int(input_ids[row, position + 1])
                    logits[row, position, target] = float(target)
            return SimpleNamespace(logits=logits)

    scores, prompt_length, manifest = candidate_scores(
        Model(), Session(), "prompt", [" a", " b"],
        batch_size=2, pad_token_id=0)
    assert prompt_length == 2
    assert manifest == {" a": [3], " b": [4, 5]}
    assert scores[" b"] != scores[" a"]


def test_o1_config_is_baseline_only_and_side_namespaced():
    value = config()
    assert all(row["evidence_id"].startswith("ol-")
               for row in value["models"])
    assert value["protocol_evidence_id"].startswith("ol-")
    assert value["joint_evidence_id"].startswith("ol-")
    assert value["selection"]["expected_rows_per_model"] == 384
    assert value["answer_contract"]["runtime_candidate_batch_size"] == 8
    assert "intervention" not in value["outputs"]
    assert value["claim_boundary"].startswith(
        "Baseline capability development gate only")
