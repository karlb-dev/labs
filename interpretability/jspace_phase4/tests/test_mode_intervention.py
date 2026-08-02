import torch
import pytest


class TinyTokenizer:
    pieces = {
        " alpha": [4],
        " the beta": [5, 6],
    }

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return {"input_ids": self.pieces[text]}


def test_alias_protection_unions_clean_topk_and_complete_aliases():
    from jspace_phase4.mode_intervention import (
        accepted_alias_token_ids,
        combined_protection_sets,
    )

    aliases = accepted_alias_token_ids(
        TinyTokenizer(), [" alpha", " the beta"])
    assert aliases == [4, 5, 6]
    logits = torch.tensor([
        [0.0, 10.0, 9.0, 8.0, -1.0, -2.0, -3.0],
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ])
    protection = combined_protection_sets(
        logits, alias_token_ids=aliases, top_k=2)
    assert protection.shape == (2, 5)
    assert set(protection[0].tolist()) == {1, 2, 4, 5, 6}
    assert set(protection[1].tolist()) == {4, 5, 6}


def _configured(arm, hidden, dictionary, protection, active=None):
    from jspace_phase4.mode_intervention import ExactProfileModeAblator

    active = torch.ones(hidden.shape[1], dtype=torch.bool) \
        if active is None else active
    ablator = ExactProfileModeAblator([], [])
    ablator.configure(
        arm=arm, dictionaries={0: dictionary},
        protection_sets=protection,
        active_position_mask=active,
        target_phase="prefill", current_phase="prefill",
        forward_index=0, k=4, evidence_id="p4-test",
        item_id="item", condition="condition", base_seed=17,
        energy_relative_floor=1e-6,
    )
    return ablator, ablator._apply(hidden, 0)


def test_matched_arm_consumes_exact_j_rank_energy_and_protection_profile():
    generator = torch.Generator().manual_seed(20260802)
    hidden = torch.randn(1, 3, 16, generator=generator)
    dictionary = torch.nn.functional.normalize(
        torch.randn(32, 16, generator=generator), dim=1)
    protection = torch.tensor([
        [0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=torch.long)
    j_ablator, j_hidden = _configured(
        "span_safe_j", hidden, dictionary, protection)
    matched_ablator, matched_hidden = _configured(
        "matched_control", hidden, dictionary, protection)
    assert not torch.equal(j_hidden, hidden)
    assert not torch.equal(matched_hidden, hidden)
    assert len(j_ablator.log.records) == len(matched_ablator.log.records) == 3
    for j_record, matched_record in zip(
            j_ablator.log.records, matched_ablator.log.records, strict=True):
        assert matched_record.requested_rank == j_record.requested_rank
        assert matched_record.delivered_rank == j_record.delivered_rank
        assert matched_record.target_energy_frac == pytest.approx(
            j_record.target_energy_frac, abs=1e-7)
        assert matched_record.delivered_energy_frac == pytest.approx(
            j_record.delivered_energy_frac, rel=1e-4, abs=1e-7)
        assert matched_record.energy_relative_error < 1e-3
        assert matched_record.selected_protected_overlap == 0
        assert matched_record.maximum_protected_cosine < 1e-4
        assert not matched_record.control_clamped
    summary = matched_ablator.log.summary()
    assert summary["rank_match_exact"]
    assert summary["maximum_selected_protected_overlap"] == 0


def test_position_mask_restores_every_inactive_hidden_state_exactly():
    generator = torch.Generator().manual_seed(9)
    hidden = torch.randn(1, 4, 12, generator=generator)
    dictionary = torch.nn.functional.normalize(
        torch.randn(20, 12, generator=generator), dim=1)
    protection = torch.tensor([[0, 1]] * 4)
    active = torch.tensor([False, True, False, True])
    ablator, changed = _configured(
        "span_safe_j", hidden, dictionary, protection, active=active)
    assert torch.equal(changed[:, ~active], hidden[:, ~active])
    assert len(ablator.log.records) == 2
    assert {record.position for record in ablator.log.records} == {1, 3}


def test_wrong_phase_configuration_is_a_hard_failure():
    from jspace_phase4.mode_intervention import ExactProfileModeAblator

    ablator = ExactProfileModeAblator([], [])
    with pytest.raises(RuntimeError, match="wrong phase"):
        ablator.configure(
            arm="span_safe_j", dictionaries={},
            protection_sets=torch.tensor([1]),
            active_position_mask=torch.tensor([True]),
            target_phase="final_answer", current_phase="reasoning",
            forward_index=1, k=1, evidence_id="p4-test",
            item_id="item", condition="condition", base_seed=0,
            energy_relative_floor=1e-6,
        )
    assert ablator.log.wrong_phase_hook_fires == 1


def test_prediction_phase_tracks_multitoken_reasoning_boundary():
    from jspace_phase4.mode_intervention import prediction_phase
    from jspace_phase4.phase_hooks import DelimiterSpec

    delimiters = DelimiterSpec(
        reasoning_start_ids=(10, 11), reasoning_end_ids=(12, 13),
        eos_token_ids=(99,), require_closed_reasoning=True)
    prompt = [1, 10, 11]
    assert prediction_phase(
        prompt, prompt_length=len(prompt),
        delimiters=delimiters) == "reasoning"
    assert prediction_phase(
        prompt + [7, 12], prompt_length=len(prompt),
        delimiters=delimiters) == "reasoning"
    assert prediction_phase(
        prompt + [7, 12, 13], prompt_length=len(prompt),
        delimiters=delimiters) == "final_answer"
    thinking_off = [1, 10, 11, 12, 13]
    assert prediction_phase(
        thinking_off, prompt_length=len(thinking_off),
        delimiters=delimiters) == "final_answer"


def test_answer_prediction_mask_includes_context_boundary_not_last_answer():
    from jspace_phase4.mode_intervention import answer_prediction_mask

    mask = answer_prediction_mask(sequence_length=8, context_length=5)
    assert mask.tolist() == [False, False, False, False,
                             True, True, True, False]
