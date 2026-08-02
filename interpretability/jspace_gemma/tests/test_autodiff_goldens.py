import pytest
import torch

from jspace_gemma.autodiff import ExactJVPError, exact_jvp, exact_linearize


def analytic_function(value):
    return torch.stack(
        [
            value[0] ** 3 + value[1],
            torch.sin(value[1]) + value[0] * value[1],
            torch.exp(value[0] - value[1]),
        ]
    )


def analytic_jvp(value, direction):
    return torch.stack(
        [
            3 * value[0] ** 2 * direction[0] + direction[1],
            direction[1] * torch.cos(value[1])
            + direction[0] * value[1]
            + value[0] * direction[1],
            torch.exp(value[0] - value[1]) * (direction[0] - direction[1]),
        ]
    )


@pytest.mark.parametrize(
    "backend", ["torch.func.jvp", "torch.autograd.functional.jvp"]
)
def test_analytic_exact_jvp(backend):
    source = torch.tensor([0.4, -0.7], dtype=torch.float64)
    direction = torch.tensor([0.6, 0.8], dtype=torch.float64)
    result = exact_jvp(analytic_function, source, direction, backend=backend)
    assert result.backend == backend
    assert torch.allclose(result.tangent, analytic_jvp(source, direction), atol=1e-12, rtol=1e-12)


def test_central_secant_converges_to_but_is_not_labeled_exact():
    source = torch.tensor([0.4, -0.7], dtype=torch.float64)
    direction = torch.tensor([0.6, 0.8], dtype=torch.float64)
    exact = analytic_jvp(source, direction)
    errors = []
    for epsilon in (1e-1, 1e-2, 1e-3):
        numerical = (
            analytic_function(source + epsilon * direction)
            - analytic_function(source - epsilon * direction)
        ) / (2 * epsilon)
        errors.append(float((numerical - exact).norm()))
    assert errors[2] < errors[1] < errors[0]


def test_detached_derivative_path_is_a_methods_blocker():
    source = torch.tensor([1.0, 2.0], dtype=torch.float64)
    direction = torch.ones_like(source)

    def detached(value):
        return value.detach().clone()

    with pytest.raises(ExactJVPError, match="finite differences are forbidden"):
        exact_jvp(detached, source, direction)


def test_linear_positive_control_is_exact():
    generator = torch.Generator().manual_seed(8)
    matrix = torch.randn(7, 5, generator=generator, dtype=torch.float64)
    source = torch.randn(5, generator=generator, dtype=torch.float64)
    direction = torch.randn(5, generator=generator, dtype=torch.float64)
    result = exact_jvp(lambda value: matrix @ value, source, direction)
    assert torch.allclose(result.tangent, matrix @ direction, atol=1e-12, rtol=1e-12)


def test_cached_linearization_matches_fresh_jvp_for_multiple_tangents():
    source = torch.tensor([0.4, -0.7], dtype=torch.float64)
    cached = exact_linearize(analytic_function, source)
    for direction in (
        torch.tensor([0.6, 0.8], dtype=torch.float64),
        torch.tensor([-0.2, 1.3], dtype=torch.float64),
    ):
        fresh = exact_jvp(analytic_function, source, direction)
        assert torch.allclose(cached.apply(direction), fresh.tangent, atol=1e-12, rtol=1e-12)
