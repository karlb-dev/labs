import math

import torch

from jspace_gemma.hooks import ExplicitDecoderSuffix, TargetSpec
from jspace_gemma.transport import evaluate_transport_cell


class _BatchShapeAndSlotSensitiveSuffix:
    """Identity map plus an artificial batch-shape/slot-dependent offset."""

    def __init__(self):
        self.clean_source = torch.tensor(
            [[[1.0, -2.0, 0.5, 3.0], [0.25, 1.5, -1.0, 2.0]]]
        )

    def __call__(self, source):
        count = source.shape[0]
        slot = torch.arange(count, device=source.device, dtype=source.dtype)
        output = source[:, -1, :] + float(count) + slot[:, None]
        return output[0] if count == 1 else output


def test_batch_aligned_baseline_cancels_shape_and_slot_artifact():
    suffix = _BatchShapeAndSlotSensitiveSuffix()
    attention = torch.ones((1, 2), dtype=torch.long)
    rows, raw = evaluate_transport_cell(
        suffix,
        attention_mask=attention,
        perturbation_mode="single_position",
        direction_specs=[
            {"type": "rademacher", "id": "random-rademacher-0"},
            {"type": "gaussian", "id": "random-gaussian-0"},
        ],
        epsilon_ladder=[0.05],
        seed=123,
        cell_id="batch-sensitive",
        metadata={"source_layer": 0},
        delivery_cosine_floor=0.999,
        delivery_norm_error_ceiling=0.01,
        batch_size=4,
    )
    assert all(row["tangent_relative_error"] < 1e-6 for row in rows)
    assert all(row["backend_parity_relative_error"] < 1e-6 for row in rows)
    assert all(
        diagnostic["within_batch_clean_spread_norm"] > 0
        for diagnostic in raw["exact_batch_diagnostics"]
    )


def test_complete_tiny_transport_cell_has_exact_and_vector_metrics():
    from transformers.models.olmo3.configuration_olmo3 import Olmo3Config
    from transformers.models.olmo3.modeling_olmo3 import Olmo3ForCausalLM

    torch.manual_seed(41)
    config = Olmo3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        eos_token_id=1,
        use_cache=False,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention", "sliding_attention"],
    )
    config._attn_implementation = "eager"
    model = Olmo3ForCausalLM(config).eval()
    ids = torch.tensor([[2, 3, 4, 5, 6]], dtype=torch.long)
    attention = torch.ones_like(ids)
    suffix = ExplicitDecoderSuffix(
        model,
        input_ids=ids,
        attention_mask=attention,
        source_layer=0,
        target=TargetSpec("final_residual"),
    )
    rows, raw = evaluate_transport_cell(
        suffix,
        attention_mask=attention,
        perturbation_mode="single_position",
        direction_specs=[
            {"type": "rademacher", "id": "random-rademacher-0"},
            {"type": "gaussian", "id": "random-gaussian-0"},
            {"type": "radial", "id": "activation-radial"},
            {"type": "sphere_tangent", "id": "activation-tangent-0"},
        ],
        epsilon_ladder=[0.001, 0.01],
        seed=123,
        cell_id="tiny-L0-single",
        metadata={"source_layer": 0},
        delivery_cosine_floor=0.999,
        delivery_norm_error_ceiling=0.01,
        batch_size=4,
    )
    assert len(rows) == 8
    assert len(raw["records"]) == 8
    assert raw["schema_version"] == 2
    assert raw["max_backend_parity_relative_error"] < 1e-5
    assert raw["exact_batch_diagnostics"]
    assert all(row["faithful_delivery"] for row in rows)
    assert all(row["tangent_cosine"] > 0.99 for row in rows)
    assert all(row["response_snr"] > 0 for row in rows)
    assert all(
        row["homogeneity_nonlinear_remainder_defect"] is not None
        and row["odd_nonlinear_remainder_defect"] is not None
        for row in rows
    )
    for row in rows:
        for key, value in row.items():
            if isinstance(value, float):
                assert math.isfinite(value), key
    pair_rows = [
        row for row in rows
        if row["direction_id"] in {"random-rademacher-0", "random-gaussian-0"}
    ]
    assert all(row["additivity_defect"] is not None for row in pair_rows)
    assert all(
        row["additivity_nonlinear_remainder_defect"] is not None
        and row["additivity_faithful_delivery"]
        for row in pair_rows
    )
