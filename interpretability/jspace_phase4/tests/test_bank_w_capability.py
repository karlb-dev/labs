import copy

import torch
import yaml


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
        "family_bootstrap_draws": 2000,
        "family_bootstrap_seed": 20260801,
        "family_capability_accuracy_floor_by_load": 0.70,
        "minimum_joint_common_families": 20,
    }


def _rows(*, high_correct_seeds=range(8), low_correct_seeds=range(8)):
    rows = []
    high_correct_seeds = set(high_correct_seeds)
    low_correct_seeds = set(low_correct_seeds)
    for family_index in range(24):
        family = f"family-{family_index:02d}"
        for seed in range(8):
            for load, correct_seeds in (
                    ("low", low_correct_seeds),
                    ("high", high_correct_seeds)):
                correct = seed in correct_seeds
                rows.append({
                    "item_id": f"{family}:{seed}:{load}",
                    "canonical_family": family,
                    "item_seed": seed,
                    "load": load,
                    "correct": correct,
                    "baseline_answer_margin": 2.0 if correct else -1.0,
                    "prompt_token_count": 100 + family_index,
                    "answer_token_count": 1 + int(seed % 4 == 0),
                })
    return rows


def test_bank_w_capability_balanced_perfect_grid_passes():
    from jspace_phase4.experiments.p4_bank_w_capability import (
        analyze_model_rows,
    )

    result = analyze_model_rows(
        _rows(), selection=_selection(), guard=_guard())
    assert result["independently_capability_eligible"]
    assert result["n_capable_families"] == 24
    assert result["load_summaries"]["low"]["accuracy"] == 1.0
    assert result["load_summaries"]["high"]["accuracy"] == 1.0
    assert result["paired_high_minus_low_accuracy"][
        "family_bootstrap_ci90"] == [0.0, 0.0]


def test_bank_w_capability_equivalence_cannot_be_rescued_by_accuracy():
    from jspace_phase4.experiments.p4_bank_w_capability import (
        analyze_model_rows,
    )

    result = analyze_model_rows(
        _rows(high_correct_seeds=range(7)),
        selection=_selection(), guard=_guard())
    assert result["load_summaries"]["high"]["accuracy"] == 0.875
    assert result["independent_gate_checks"]["accuracy_floor_both_loads"]
    assert not result["independent_gate_checks"][
        "load_difference_equivalent"]
    assert not result["independently_capability_eligible"]


def _analysis(*, eligible=True, capable=range(24)):
    return {
        "independently_capability_eligible": eligible,
        "capable_family_ids": [f"family-{index:02d}" for index in capable],
        "n_capable_families": len(list(capable)),
        "load_summaries": {
            "low": {"accuracy": 0.9},
            "high": {"accuracy": 0.9},
        },
    }


def _joint_config():
    return {
        "models": [
            {"slug": "a"}, {"slug": "b"}, {"slug": "c"}],
        "capability_guard": {"minimum_joint_common_families": 20},
        "claim_boundary": "baseline only",
    }


def test_bank_w_joint_gate_excludes_only_independent_failures():
    from jspace_phase4.experiments.p4_bank_w_capability import (
        aggregate_model_payloads,
    )

    payloads = {
        "a": {"analysis": _analysis()},
        "b": {"analysis": _analysis(capable=range(21))},
        "c": {"analysis": _analysis(eligible=False)},
    }
    result = aggregate_model_payloads(payloads, config=_joint_config())
    assert result["p4p3_baseline_capability_ready"]
    assert result["primary_model_set"] == ["a", "b"]
    assert result["n_joint_common_capable_families"] == 21


def test_bank_w_joint_gate_blocks_instead_of_dropping_eligible_model():
    from jspace_phase4.experiments.p4_bank_w_capability import (
        aggregate_model_payloads,
    )

    payloads = {
        "a": {"analysis": _analysis()},
        "b": {"analysis": _analysis(capable=range(19))},
        "c": {"analysis": _analysis(eligible=False)},
    }
    result = aggregate_model_payloads(payloads, config=_joint_config())
    assert not result["p4p3_baseline_capability_ready"]
    assert result["would_be_primary_model_set"] == ["a", "b"]
    assert result["primary_model_set"] == []
    assert result["n_joint_common_capable_families"] == 19


def test_bank_w_capability_protocol_matches_registered_bank_contract():
    from jspace_phase4.experiments.p4_bank_w_capability import author_protocol

    path = (
        "interpretability/jspace_phase4/configs/"
        "p4_bank_w_capability_protocol_dev.yaml")
    with open(path) as handle:
        config = yaml.safe_load(handle)
    result = author_protocol(copy.deepcopy(config))
    assert result["all_protocol_gates_pass"]
    assert result["selection"]["expected_rows_per_model"] == 384
    assert all(row["passes"] for row in result[
        "tokenizer_answer_audits"].values())
    assert result["outcome_blinding"].startswith("Bank structure")


def test_bank_w_candidate_scores_use_complete_sequences_and_padding():
    from types import SimpleNamespace

    from jspace_phase4.experiments.p4_bank_w_capability import (
        _candidate_scores,
    )

    class Session:
        def prompt_ids(self, prompt):
            assert prompt == "prompt"
            return torch.tensor([[1, 2]])

        def answer_ids(self, alias):
            values = {" a": [3], " b": [4, 5]}[alias]
            return torch.tensor([values])

    class Model:
        def __call__(self, *, input_ids, attention_mask, use_cache):
            assert not use_cache
            logits = torch.zeros((*input_ids.shape, 8), dtype=torch.float32)
            for row in range(input_ids.shape[0]):
                # Reward each observed answer token at the exact position that
                # predicts it. Padding is deliberately left unscored.
                for position in range(1, int(attention_mask[row].sum()) - 1):
                    target = int(input_ids[row, position + 1])
                    logits[row, position, target] = float(target)
            return SimpleNamespace(logits=logits)

    scores, prompt_length, token_manifest = _candidate_scores(
        Model(), Session(), "prompt", [" a", " b"],
        batch_size=2, pad_token_id=0)
    assert prompt_length == 2
    assert token_manifest == {" a": [3], " b": [4, 5]}
    expected_a = torch.log_softmax(
        torch.tensor([0.0, 0, 0, 3, 0, 0, 0, 0]), dim=0)[3]
    expected_b_first = torch.log_softmax(
        torch.tensor([0.0, 0, 0, 0, 4, 0, 0, 0]), dim=0)[4]
    expected_b_second = torch.log_softmax(
        torch.tensor([0.0, 0, 0, 0, 0, 5, 0, 0]), dim=0)[5]
    assert abs(scores[" a"] - float(expected_a)) < 1e-6
    assert abs(scores[" b"] - float(
        expected_b_first + expected_b_second)) < 1e-6
