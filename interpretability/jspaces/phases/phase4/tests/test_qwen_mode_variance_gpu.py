from types import SimpleNamespace

import pytest
import torch


class FakeCache:
    def __init__(self, stream_id, history):
        self.stream_id = int(stream_id)
        self.history = tuple(int(value) for value in history)

    def get_seq_length(self):
        return len(self.history)


class FakeLog:
    def rows(self):
        return []

    def summary(self):
        return {
            "n_positions": 0,
            "hook_fires": {
                "prefill": 0, "reasoning": 0, "final_answer": 0},
            "wrong_phase_hook_fires": 0,
        }


class RecordingAblator:
    ALLOWED_ARMS = {"span_safe_j", "matched_control"}

    def __init__(self):
        self.mode = None
        self.log = FakeLog()
        self.configurations = []

    def configure(self, **values):
        copied = dict(values)
        copied["active_position_mask"] = values[
            "active_position_mask"].detach().cpu().tolist()
        copied["position_phases"] = list(values["position_phases"])
        self.configurations.append(copied)
        self.mode = dict(values)

    def reset(self):
        self.mode = None
        self.log = FakeLog()


class FakeHF(torch.nn.Module):
    def __init__(self, ablator, transitions, vocab_size=128):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.ablator = ablator
        self.transitions = {
            tuple(key): int(value) for key, value in transitions.items()}
        self.vocab_size = int(vocab_size)
        self.next_stream = 0
        self.calls = []

    def forward(self, input_ids, past_key_values=None, use_cache=True):
        incoming = [int(value) for value in input_ids[0].tolist()]
        if past_key_values is None:
            self.next_stream += 1
            stream = self.next_stream
            prefix = []
            past_stream = None
        else:
            stream = past_key_values.stream_id
            prefix = list(past_key_values.history)
            past_stream = stream
        history = [*prefix, *incoming]
        logits = torch.full(
            (1, len(incoming), self.vocab_size), -20.0,
            device=input_ids.device)
        for position in range(len(incoming)):
            local_history = tuple([*prefix, *incoming[:position + 1]])
            token = self.transitions.get(local_history, 99)
            logits[0, position, token] = 20.0
        self.calls.append({
            "past_stream": past_stream,
            "out_stream": stream,
            "input": incoming,
            "intervened": self.ablator.mode is not None,
        })
        return SimpleNamespace(
            logits=logits,
            past_key_values=FakeCache(stream, history) if use_cache else None,
        )


def _generation_config(max_new_tokens=16):
    return {
        "evidence_id": "p4-test-mode-pilot",
        "generation": {"max_new_tokens": max_new_tokens},
        "intervention": {
            "clean_protect_top_k": 10,
            "k": 4,
            "matched_seed": 17,
            "energy_relative_floor": 1e-6,
        },
    }


def _delimiters():
    from jspace_phase4.phase_hooks import DelimiterSpec

    return DelimiterSpec(
        reasoning_start_ids=(10, 11), reasoning_end_ids=(12, 13),
        eos_token_ids=(99,), require_closed_reasoning=True)


def _transitions():
    thinking_on = (1, 10, 11)
    thinking_off = (1, 10, 11, 12, 13)
    return {
        thinking_on: 7,
        (*thinking_on, 7): 12,
        (*thinking_on, 7, 12): 13,
        (*thinking_on, 7, 12, 13): 5,
        (*thinking_on, 7, 12, 13, 5): 99,
        thinking_off: 5,
        (*thinking_off, 5): 99,
    }


@pytest.mark.parametrize(
    "prompt,mode,phase,expected_forward_indices,expected_width",
    [
        ([1, 10, 11], "thinking_on", "final_answer", [3, 4], 1),
        ([1, 10, 11, 12, 13], "thinking_off", "final_answer", [0, 1], 1),
        ([1, 10, 11], "thinking_on", "prefill", [0], 3),
        ([1, 10, 11, 12, 13], "thinking_off", "prefill", [0], 5),
    ],
)
def test_generation_owns_exact_phase_boundaries_and_isolates_clean_cache(
        prompt, mode, phase, expected_forward_indices, expected_width):
    from jspace_phase4.experiments.p4_qwen_mode_variance_gpu import (
        generate_intervened_tokens,
    )

    ablator = RecordingAblator()
    model = FakeHF(ablator, _transitions())
    result = generate_intervened_tokens(
        model, ablator, prompt_ids=prompt, alias_ids=[5],
        delimiters=_delimiters(), mode=mode, phase=phase,
        arm="span_safe_j", dictionaries={}, config=_generation_config(),
        item_id="fact")
    assert result["generated_token_ids"][-1] == 99
    assert [row["forward_index"] for row in ablator.configurations] == \
        expected_forward_indices
    assert len(ablator.configurations[0]["active_position_mask"]) == \
        expected_width
    assert all(
        set(row["position_phases"]) == {phase}
        and all(row["active_position_mask"])
        for row in ablator.configurations)

    clean_stream = model.calls[0]["out_stream"]
    intervention_streams = {
        (row["past_stream"] if row["past_stream"] is not None
         else row["out_stream"])
        for row in model.calls if row["intervened"]
    }
    assert clean_stream not in intervention_streams


@pytest.mark.parametrize(
    "phase,expected_active",
    [
        ("prefill", [True, True, True, False, False, False, False]),
        ("final_answer", [False, False, False, False, True, True, False]),
    ],
)
def test_secondary_lp_uses_exact_predictor_position_mask(
        phase, expected_active):
    from jspace_phase4.experiments.p4_qwen_mode_variance_gpu import (
        secondary_answer_lp,
    )

    ablator = RecordingAblator()
    model = FakeHF(ablator, {})
    value, _rows = secondary_answer_lp(
        model, ablator, context_ids=[1, 2, 3, 4, 5],
        answer_ids=[6, 7], prompt_length=3, alias_ids=[6, 7],
        phase=phase, arm="matched_control", dictionaries={},
        config=_generation_config(), item_id="fact", condition="cell")
    assert value < 0
    configured = ablator.configurations[-1]
    assert configured["active_position_mask"] == expected_active
    assert all(
        configured["position_phases"][position] == phase
        for position, active in enumerate(expected_active) if active)


def _valid_summary(phase="prefill"):
    return {
        "n_positions": 1,
        "hook_fires": {
            "prefill": int(phase == "prefill"),
            "reasoning": 0,
            "final_answer": int(phase == "final_answer"),
        },
        "wrong_phase_hook_fires": 0,
        "requested_rank_total": 2,
        "delivered_rank_total": 2,
        "selected_effective_rank_total": 3,
        "span_safe_effective_rank_total": 2,
        "lost_rank_total": 1,
        "rank_match_exact": True,
        "maximum_energy_relative_error": 0.001,
        "maximum_selected_protected_overlap": 0,
        "maximum_protected_cosine": 1e-6,
        "control_clamped_positions": 0,
    }


def _mechanical_config():
    return {
        "intervention": {
            "maximum_wrong_phase_hook_fires": 0,
            "minimum_expected_phase_hook_fires_per_row": 1,
            "require_zero_selected_protected_overlap": True,
            "require_exact_rank_match": True,
            "maximum_energy_relative_error": 0.01,
            "maximum_protected_cosine": 1e-4,
        }
    }


def test_missing_hook_and_protected_overlap_are_hard_failures():
    from jspace_phase4.experiments.p4_qwen_mode_variance_gpu import (
        _enforce_intervention_summary,
    )

    good = _valid_summary()
    _enforce_intervention_summary(good, _mechanical_config(), endpoint="test")
    missing = dict(good)
    missing["hook_fires"] = {
        "prefill": 0, "reasoning": 0, "final_answer": 0}
    with pytest.raises(RuntimeError, match="expected_phase_hook_fires"):
        _enforce_intervention_summary(
            missing, _mechanical_config(), endpoint="test")
    overlap = dict(good)
    overlap["maximum_protected_cosine"] = 0.1
    with pytest.raises(RuntimeError, match="maximum_protected_cosine"):
        _enforce_intervention_summary(
            overlap, _mechanical_config(), endpoint="test")


def _profile_row():
    return {
        "endpoint": "generation", "layer": 20, "forward_index": 0,
        "position": 0, "requested_rank": 2,
        "selected_effective_rank": 3, "span_safe_effective_rank": 2,
        "lost_rank": 1,
    }


def test_profile_duplicate_and_rank_arithmetic_are_refused():
    from jspace_phase4.experiments.p4_qwen_mode_variance_gpu import (
        _validate_profile_rows,
    )

    row = _profile_row()
    _validate_profile_rows([row])
    with pytest.raises(RuntimeError, match="duplicate"):
        _validate_profile_rows([row, dict(row)])
    bad = dict(row, lost_rank=0)
    with pytest.raises(RuntimeError, match="lost-rank"):
        _validate_profile_rows([bad])


def test_resume_state_refuses_header_mismatch(tmp_path):
    from jspace_phase4.experiments.p4_qwen_mode_variance_gpu import _state

    path = tmp_path / "state.json"
    _state(path, {"config_sha256": "a" * 64})
    with pytest.raises(RuntimeError, match="incompatible"):
        _state(path, {"config_sha256": "b" * 64})


def test_clean_sentinel_hash_drift_is_refused():
    from jspace_phase4.experiments.p4_qwen_mode_variance_gpu import (
        _validate_clean_sentinel,
    )
    from jspace_phase4.manifests import object_sha256

    payload = {
        "prompt_token_ids_sha256": "a" * 64,
        "top32_ids": list(range(32)),
        "top32_logits_rounded_1e4": [0.0] * 32,
        "cache_signature": {"type": "FakeCache", "sequence_length": 3},
    }
    sentinel = {**payload, "sentinel_sha256": object_sha256(payload)}
    _validate_clean_sentinel(sentinel)
    sentinel["top32_ids"][0] = 99
    with pytest.raises(RuntimeError, match="hash drift"):
        _validate_clean_sentinel(sentinel)


class TinyTokenizer:
    def apply_chat_template(
            self, _messages, *, tokenize, add_generation_prompt,
            enable_thinking):
        assert add_generation_prompt
        if tokenize:
            return [1, 10, 11] if enable_thinking \
                else [1, 10, 11, 12, 13]
        return "ON" if enable_thinking else "OFF"

    def decode(self, ids, skip_special_tokens=False):
        values = [int(value) for value in ids]
        if values == [1, 10, 11]:
            return "ON"
        if values == [1, 10, 11, 12, 13]:
            return "OFF"
        pieces = {5: "alpha", 7: "reason", 12: "</", 13: "think>"}
        return "".join(pieces.get(value, "") for value in values)

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        assert text == " alpha"
        return SimpleNamespace(input_ids=[5])


class OneRecordLog:
    def __init__(self, phase, arm):
        self.phase = phase
        self.arm = arm

    def summary(self):
        return _valid_summary(self.phase)

    def rows(self):
        return [{
            **_profile_row(),
            "phase": self.phase, "arm": self.arm,
            "selected_ids": [1, 2, 3], "protected_ids": [5, 6],
            "selected_protected_overlap": 0,
            "delivered_rank": 2,
            "target_energy_frac": 0.1,
            "delivered_energy_frac": 0.1,
            "energy_relative_error": 0.0,
            "maximum_protected_cosine": 0.0,
            "protected_effective_rank": 2,
            "control_clamped": False,
        }]


def test_parser_failure_with_eos_is_recorded_incorrect(monkeypatch):
    from jspace_phase3.bank import FactBundle
    from jspace_phase4.experiments import p4_qwen_mode_variance_gpu as module

    phase = "prefill"
    arm = "span_safe_j"
    log = OneRecordLog(phase, arm)
    monkeypatch.setattr(module, "generate_intervened_tokens", lambda *a, **k: {
        "generated_token_ids": [7, 99],
        "clean_sentinel": {
            "prompt_token_ids_sha256": "a" * 64,
            "top32_ids": list(range(32)),
            "top32_logits_rounded_1e4": [0.0] * 32,
            "cache_signature": {"type": "FakeCache"},
            "sentinel_sha256": "b" * 64,
        },
        "cache_identity_sha256": "c" * 64,
        "intervention_log": log,
    })
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    bundle = FactBundle(
        fact_id="fact", canonical_family="family", relation_group="group",
        bank="F", source="s", bridge="b", answer="alpha",
        accepted_answers=[" alpha"], prompts={"composed": "Question"})
    config = {
        "generation": {
            "prompt_variant": "composed", "prompt_instruction": "Answer.",
            "max_new_tokens": 16,
        },
        "intervention": _mechanical_config()["intervention"],
        "secondary_answer_lp": {"enabled": False},
    }
    row, profile, _sentinel = module.run_pilot_cell(
        object(), TinyTokenizer(), RecordingAblator(), bundle=bundle,
        mode="thinking_on", phase=phase, arm=arm,
        methods={"parser": {"version": "p4-phase-parser-v2"}},
        delimiters=_delimiters(), dictionaries={}, config=config,
        baseline_sentinel=None)
    assert row["stop_reason"] == "eos"
    assert row["parse_valid"] is False
    assert row["correct"] is False
    assert row["generated_token_phases_json"]
    assert len(profile) == 1
