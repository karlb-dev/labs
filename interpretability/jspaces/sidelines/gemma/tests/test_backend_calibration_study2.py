import math
from pathlib import Path

import torch
import yaml

from jspace_gemma.backend_calibration import (
    bfloat16_quantum_metrics,
    derive_calibration,
    direction_tensor,
)
from jspace_gemma.experiments.gm2_backend_parity_calibration import (
    _evaluate_pair,
    _validate_pair_summaries,
)


ROOT = Path(__file__).resolve().parents[1]


def test_bfloat16_quantum_conversion_at_one():
    spacing = 1.0 / 128.0
    result = bfloat16_quantum_metrics(
        torch.tensor([spacing]), torch.tensor([1.0])
    )
    assert math.isclose(result["max_difference_dtype_quanta"], 1.0)
    assert math.isclose(
        result["one_dtype_quantum_relative_equivalent"], spacing
    )
    assert math.isclose(
        result["ten_dtype_quanta_relative_equivalent"], 10.0 * spacing
    )


def test_all_frozen_direction_families_are_unit_and_deterministic():
    reference = torch.arange(1, 65, dtype=torch.float32)
    radial = reference / reference.norm()
    for family in ("rademacher", "gaussian", "sphere_tangent"):
        first = direction_tensor(reference, family=family, seed=17)
        second = direction_tensor(reference, family=family, seed=17)
        assert torch.equal(first, second)
        assert torch.isclose(first.norm(), torch.tensor(1.0), atol=1e-6)
        if family == "sphere_tangent":
            assert abs(float(torch.dot(first, radial))) < 1e-5


def _synthetic_rows(config):
    rows = []
    for model_key, spec in config["models"].items():
        for layer in spec["layers_zero_indexed"]:
            for prompt_id in config["prompt_ids"]:
                for batch_size in config["batch_sizes"]:
                    for draw in config["directions"]["draws"]:
                        pair = "|".join(
                            [
                                model_key,
                                str(layer),
                                prompt_id,
                                str(batch_size),
                                draw["draw_id"],
                                "full",
                            ]
                        )
                        for slot in range(batch_size):
                            rows.append(
                                {
                                    "row_id": f"{pair}|slot={slot}",
                                    "pair_id": pair,
                                    "model_key": model_key,
                                    "model_id": spec["model_id"],
                                    "model_revision": spec["revision"],
                                    "layer": layer,
                                    "prompt_id": prompt_id,
                                    "batch_size": batch_size,
                                    "slot": slot,
                                    "suffix_variant": "full",
                                    "finite_or_exception_state": "finite",
                                    "primal_relative_error": 0.0,
                                    "tangent_relative_error": 1.0e-4,
                                    "tangent_cosine": 0.999999,
                                    "ten_dtype_quanta_relative_equivalent": 1.0e-3,
                                    "deterministic_replay": True,
                                }
                            )
    return rows


def test_calibration_reconstructs_216_pairs_without_target_data():
    config = yaml.safe_load(
        (ROOT / "configs/gm2_backend_parity_calibration.yaml").read_text()
    )
    result = derive_calibration(_synthetic_rows(config), config)
    assert result["row_counts"]["full_backend_pairs"] == 216
    assert result["row_counts"]["full"] == 936
    assert result["router"]["route"] == "benign_scheduling_floor"
    assert math.isclose(result["licensed_ceilings"]["pooled"], 1.0e-3)


def test_runtime_and_derivation_sources_exclude_stage1_target_artifact():
    runtime = (
        ROOT / "jspace_gemma/experiments/gm2_backend_parity_calibration.py"
    ).read_text()
    derivation = (ROOT / "jspace_gemma/backend_calibration.py").read_text()
    source = runtime + derivation
    assert "gm2_stage1_relicense" not in source
    assert "0.002458" not in source


def test_pair_runner_preserves_singleton_batch_axis_across_both_backends():
    class TinySuffix:
        def __init__(self):
            self.clean_source = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
            self.weight = torch.tensor(
                [[1.0, 0.5], [0.25, -0.5], [0.75, 0.125], [-0.25, 1.0]]
            )

        def __call__(self, value):
            output = value[:, -1].matmul(self.weight)
            return output[0] if output.shape[0] == 1 else output

    config = yaml.safe_load(
        (ROOT / "configs/gm2_backend_parity_calibration.yaml").read_text()
    )
    spec = config["models"]["gemma4_31b"]
    rows, pair = _evaluate_pair(
        TinySuffix(),
        config=config,
        model_key="gemma4_31b",
        model_spec=spec,
        layer=22,
        prompt={"prompt_id": "gm-p001", "text": "sentinel"},
        prompt_sha="a" * 64,
        token_sha="b" * 64,
        batch_size=1,
        draw=config["directions"]["draws"][0],
        suffix_variant="full",
    )
    assert len(rows) == 1
    assert rows[0]["finite_or_exception_state"] == "finite"
    assert rows[0]["deterministic_replay"] is True
    assert rows[0]["tangent_relative_error"] == 0.0
    assert pair["all_slots"]["elements"] == 2


def test_all_slot_cosine_audit_allows_only_dimension_bounded_fp32_reduction():
    row = {
        "pair_id": "sentinel",
        "slot": 0,
        "primary_tangent_norm": 1.0,
        "independent_tangent_norm": 1.0,
        "tangent_difference_norm": 0.01,
        "tangent_dot_product": 0.99997,
        "max_absolute_difference": 0.001,
        "tangent_relative_error": 0.01,
        "tangent_cosine": 0.99997,
    }
    pair = {
        "pair_id": "sentinel",
        "finite_or_exception_state": "finite",
        "all_slots": {
            "relative_error": 0.01,
            "cosine": 0.99998,
            "max_absolute_difference": 0.001,
            "elements": 40960,
        },
        "selected_slot": {"relative_error": 0.01, "cosine": 0.99997},
    }
    audit = _validate_pair_summaries(
        {"rows": [row], "pair_summaries": [pair]}
    )
    assert audit["passed"] is True
    pair["all_slots"]["cosine"] = 0.999
    assert _validate_pair_summaries(
        {"rows": [row], "pair_summaries": [pair]}
    )["passed"] is False
