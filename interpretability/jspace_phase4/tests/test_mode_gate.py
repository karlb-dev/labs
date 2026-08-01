import re


class _Tokens:
    def __init__(self, values):
        self.input_ids = values


class _WordTokenizer:
    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        pieces = re.findall(r"[A-Za-z0-9_-]+|[^\w\s]", text)
        return _Tokens([sum(map(ord, piece)) % 997 for piece in pieces])


def test_rationale_controls_are_exactly_token_matched():
    from jspace_phase4.experiments.p4_mode_gate import token_match_controls

    audit = token_match_controls(
        _WordTokenizer(),
        {"correct": "one two three", "wrong": "one two",
         "shuffled": "three one", "filler": "neutral"},
        neutral_fragments=[" Neutral.", " N.", " ."], maximum_span=0)
    assert audit["passes"]
    assert audit["token_span"] == 0
    assert len({row["n_tokens"] for row in audit["controls"].values()}) == 1


def test_answer_before_reasoning_close_is_not_final_answer():
    from jspace_phase4.experiments.p4_mode_gate import answer_boundary_status
    from jspace_phase4.phase_hooks import DelimiterSpec, classify_token_phases

    delimiters = DelimiterSpec(
        reasoning_start_ids=(10,), reasoning_end_ids=(11,), eos_token_ids=(12,))
    prompt = [1, 10]
    tokens = prompt + [77, 11, 77, 12]
    parsed = classify_token_phases(
        tokens, prompt_length=len(prompt), delimiters=delimiters)
    status = answer_boundary_status(
        tokens, prompt_length=len(prompt), parsed=parsed,
        accepted_token_sequences=[(77,)])
    assert status["status"] == "accepted_answer_in_final"
    assert status["reasoning_match_indices"]
    assert status["final_match_indices"]

    only_early = prompt + [77, 11, 12]
    parsed_early = classify_token_phases(
        only_early, prompt_length=len(prompt), delimiters=delimiters)
    early = answer_boundary_status(
        only_early, prompt_length=len(prompt), parsed=parsed_early,
        accepted_token_sequences=[(77,)])
    assert early["status"] == "answer_before_reasoning_close"


def test_parse_failure_and_truncation_are_not_coerced_to_incorrect():
    from jspace_phase4.experiments.p4_mode_gate import (
        answer_boundary_status, completion_outcome,
    )
    from jspace_phase4.phase_hooks import DelimiterSpec, classify_token_phases

    prompt = [1, 10]
    parsed = classify_token_phases(
        prompt + [88], prompt_length=len(prompt),
        delimiters=DelimiterSpec(
            reasoning_start_ids=(10,), reasoning_end_ids=(11,)))
    status = answer_boundary_status(
        prompt + [88], prompt_length=len(prompt), parsed=parsed,
        accepted_token_sequences=[(77,)])
    outcome = completion_outcome(
        parsed, stop_reason="length", answer_status=status)
    assert outcome["parse_failure"]
    assert outcome["truncated"]
    assert outcome["answer_omission"]
    assert "correct" not in outcome
    assert not outcome["eligible_for_final_answer_grading"]


def test_mode_protocol_freezes_only_phases_common_to_both_modes():
    import yaml

    with open("interpretability/jspace_phase4/configs/"
              "p4_qwen_mode_parser_gate_dev.yaml") as handle:
        config = yaml.safe_load(handle)
    support = config["structural_common_support"]
    assert support["primary_phases"] == ["prefill", "final_answer"]
    assert support["missing_cell"] == "thinking_off_x_reasoning"
    assert config["primary"]["reasoning_only_status"] == \
        "thinking-on-secondary"
    assert config["primary"]["alternative"] == "greater"
    assert config["parser"]["version"] == "p4-phase-parser-v2"
