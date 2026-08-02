import pytest
import torch

from jspace_gemma.hooks import delivery_audit, patterned_direction, source_mask
from jspace_gemma.transport_metrics import (
    additivity_metrics,
    homogeneity_metrics,
    odd_symmetry_defect,
    tangent_metrics,
)


def test_source_modes_and_delivery_contract():
    attention = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
    single = source_mask(attention, mode="single_position", position=-1)
    uniform = source_mask(attention, mode="uniform_valid")
    assert single.tolist() == [[False, False, True, False]]
    assert uniform.tolist() == [[True, True, True, False]]
    clean = torch.randn(1, 4, 8)
    direction = torch.arange(1, 9, dtype=torch.float32)
    desired = patterned_direction(clean, direction, single, 0.2)
    realized, audit = delivery_audit(
        clean, desired, model_dtype=torch.float32, selected_mask=single
    )
    assert audit.faithful
    assert audit.cosine == pytest.approx(1.0)
    assert torch.allclose(realized, desired, atol=2e-7, rtol=2e-7)


def test_linear_response_has_zero_vector_defects():
    left = torch.tensor([1.0, 2.0, -3.0])
    right = torch.tensor([-2.0, 1.0, 4.0])
    tangent = tangent_metrics(left, left)
    homogeneous = homogeneity_metrics(left, 2 * left)
    additive = additivity_metrics(left + right, left, right)
    odd = odd_symmetry_defect(left, -left)
    assert tangent["tangent_cosine"] == pytest.approx(1.0)
    assert tangent["gain"] == pytest.approx(1.0)
    assert tangent["tangent_relative_error"] == pytest.approx(0.0)
    assert homogeneous["homogeneity_defect"] == pytest.approx(0.0)
    assert homogeneous["homogeneity_scale_ratio"] == pytest.approx(2.0)
    assert additive["additivity_defect"] == pytest.approx(0.0)
    assert odd == pytest.approx(0.0)
