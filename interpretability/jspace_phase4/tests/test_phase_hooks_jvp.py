import pytest
import torch

from jspace_phase4.phase_hooks import (
    DelimiterSpec,
    Phase,
    PhaseHookSentinel,
    classify_token_phases,
)


def _delimiters():
    return DelimiterSpec(
        reasoning_start_ids=(10, 11),
        reasoning_end_ids=(12,),
    )


def test_phase_parser_happy_path_and_thinking_off():
    parsed = classify_token_phases(
        [1, 2, 10, 11, 7, 12, 8],
        prompt_length=2,
        delimiters=_delimiters(),
    )
    assert parsed.valid
    assert parsed.phases == (
        "prefill", "prefill", "reasoning", "reasoning",
        "reasoning", "reasoning", "final_answer")
    off = classify_token_phases(
        [1, 2, 8, 9],
        prompt_length=2,
        delimiters=_delimiters(),
    )
    assert off.valid
    assert off.phases[-2:] == ("final_answer", "final_answer")


def test_phase_parser_rejects_malformed_delimiters():
    unclosed = classify_token_phases(
        [1, 10, 11, 7],
        prompt_length=1,
        delimiters=_delimiters(),
    )
    assert not unclosed.valid
    assert "unclosed_reasoning" in unclosed.errors
    end_first = classify_token_phases(
        [1, 12, 8],
        prompt_length=1,
        delimiters=_delimiters(),
    )
    assert not end_first.valid
    assert "reasoning_end_without_start" in end_first.errors


def test_phase_hook_sentinel_rejects_cross_phase_fire():
    sentinel = PhaseHookSentinel([Phase.REASONING.value])
    sentinel.record("reasoning")
    assert sentinel.require_fired()["hook_fires"]["reasoning"] == 1
    with pytest.raises(RuntimeError, match="forbidden phase"):
        sentinel.record("prefill")


def test_tiny_nonlinear_jvp_matches_smallest_secant_best():
    source = torch.tensor([0.4, -0.7], dtype=torch.float64,
                          requires_grad=True)
    direction = torch.tensor([0.6, 0.8], dtype=torch.float64)

    def downstream(value):
        return torch.stack([
            value[0] ** 3 + value[1],
            torch.sin(value[1]) + value[0] * value[1],
        ])

    _, jvp = torch.autograd.functional.jvp(
        downstream, source, direction, strict=True)
    errors = []
    for epsilon in (0.1, 0.01, 0.001):
        secant = (
            downstream(source + epsilon * direction)
            - downstream(source - epsilon * direction)
        ) / (2 * epsilon)
        errors.append(float((secant - jvp).norm().detach()))
    assert errors[-1] < errors[0] * 1e-3


def test_olmo_like_linear_positive_control_is_exact():
    generator = torch.Generator().manual_seed(8)
    matrix = torch.randn(5, 7, generator=generator, dtype=torch.float64)
    source = torch.randn(7, generator=generator, dtype=torch.float64,
                         requires_grad=True)
    direction = torch.randn(7, generator=generator, dtype=torch.float64)

    def downstream(value):
        return matrix @ value

    _, jvp = torch.autograd.functional.jvp(
        downstream, source, direction, strict=True)
    epsilon = 1e-4
    secant = (
        downstream(source + epsilon * direction)
        - downstream(source - epsilon * direction)
    ) / (2 * epsilon)
    assert torch.allclose(jvp, matrix @ direction, atol=1e-10, rtol=1e-10)
    assert torch.allclose(jvp, secant, atol=1e-10, rtol=1e-10)
