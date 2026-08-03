from pathlib import Path

import pandas as pd
import pytest
import torch
from jspace_olmo_lineage.experiments.transport_validation import (
    classify_rows,
    evaluate_dual_backend_transport_cell,
)


class LinearSuffix:
    def __init__(self):
        self.clean_source = torch.tensor([[[1.0, -2.0, 0.5, 3.0]]])

    def __call__(self, source):
        result = 3.0 * source[:, -1, :]
        return result[0] if result.shape[0] == 1 else result


def _config():
    return {
        "delivery": {"cosine_floor": 0.999, "relative_norm_error_ceiling": 0.01},
        "response_snr": {"measurement_floor": 12.0, "decision_floor": 20.0},
        "transport_gate": {
            "tangent_cosine_floor": 0.98,
            "forward_relative_error_ceiling": 0.20,
            "central_relative_error_ceiling": 0.10,
        },
    }


def test_dual_backend_wrapper_restores_shared_binding_and_records_parity():
    from jspace_gemma import transport

    before = transport.exact_jvp
    rows, raw = evaluate_dual_backend_transport_cell(
        LinearSuffix(),
        attention_mask=torch.ones((1, 1), dtype=torch.long),
        perturbation_mode="single_position",
        direction_specs=[
            {"type": "rademacher", "id": "rademacher-0"},
            {"type": "gaussian", "id": "gaussian-0"},
            {"type": "sphere_tangent", "id": "sphere_tangent-0"},
        ],
        epsilon_ladder=[0.05],
        seed=123,
        cell_id="tiny",
        metadata={"source_layer": 0},
        delivery_cosine_floor=0.999,
        delivery_norm_error_ceiling=0.01,
        batch_size=4,
    )
    assert transport.exact_jvp is before
    assert len(rows) == 3
    assert raw["exact_backends"] == [
        "torch.func.jvp", "torch.autograd.functional.jvp"]
    assert all(row["backend_tangent_relative_error"] < 1e-8 for row in rows)
    assert all(row["backend_tangent_cosine"] == pytest.approx(1.0) for row in rows)


def test_classification_uses_backend_snr_forward_and_central_gates():
    row = {
        "faithful_delivery": True,
        "backend_tangent_relative_error": 0.01,
        "response_snr": 30.0,
        "tangent_cosine": 0.99,
        "tangent_relative_error": 0.10,
        "central_tangent_cosine": 0.99,
        "central_tangent_relative_error": 0.05,
    }
    result = classify_rows(pd.DataFrame([row]), _config(), 0.02)
    assert bool(result.loc[0, "transport_row_passed"])
    failed = dict(row, central_tangent_relative_error=0.11)
    result = classify_rows(pd.DataFrame([failed]), _config(), 0.02)
    assert not bool(result.loc[0, "transport_row_passed"])


def test_transport_module_does_not_import_phase4_registry():
    source = Path(__file__).resolve().parents[1] / (
        "jspace_olmo_lineage/experiments/transport_validation.py")
    text = source.read_text()
    assert "jspace_phase4.registry" not in text
    assert "finite_difference" not in text.lower()
