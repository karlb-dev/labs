from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _minimal() -> dict:
    digest = "a" * 64
    return {
        "tier": "phase4-development",
        "canonical_lens_unchanged": True,
        "prompt": {
            "one_based_index": 323,
            "zero_based_index": 322,
            "retained_unconditionally": True,
        },
        "lenses": {
            "a500": {"n_prompts": 500, "lens_sha256": digest},
            "a1000": {"n_prompts": 1000, "lens_sha256": digest},
        },
        "equal_weight_contract": {
            "tiny_model_direct_refit_required": True,
            "adjacent_atomic_checkpoint_required": True,
            "earlier_checkpoint": {"n": 195},
            "later_checkpoint": {"n": 198},
            "prompt_indices_one_based": [196, 197, 198],
        },
        "analysis": {
            "leave_one_out_formulas": {
                "a500": "(500 * J_a500 - J_prompt323) / 499",
                "a1000": "(1000 * J_a1000 - J_prompt323) / 999",
            },
            "decision_load_bearing_metrics": [
                "assay_task_token_median_disagreement",
                "assay_task_token_q05_disagreement",
                "assay_identity_adjusted_matrix_disagreement",
            ],
        },
    }


def test_paired_influence_contract_cannot_trim_or_change_fit_pair():
    import pytest

    from jspace_phase4.experiments.p4_qwen_lens_influence_paired import (
        validate_paired_config,
    )

    config = _minimal()
    validate_paired_config(config)
    config["prompt"]["retained_unconditionally"] = False
    with pytest.raises(RuntimeError, match="retained unconditionally"):
        validate_paired_config(config)
    config = _minimal()
    config["lenses"]["a1000"]["n_prompts"] = 999
    with pytest.raises(RuntimeError, match="prompt count drift"):
        validate_paired_config(config)


def test_tiny_direct_refit_assertion_is_runtime_reproducible():
    from jspace_phase4.experiments.p4_qwen_lens_influence_paired import (
        tiny_direct_refit_contract,
    )

    first = tiny_direct_refit_contract()
    second = tiny_direct_refit_contract()
    assert first == second
    assert first["pass"]
    assert first["maximum_absolute_error"] <= 1e-6


def test_materiality_decision_has_the_frozen_three_way_wording():
    from jspace_phase4.experiments.p4_qwen_lens_influence_paired import (
        classify_materiality,
    )

    thresholds = {
        "assay_task_token_median_disagreement": 0.02,
        "assay_task_token_q05_disagreement": 0.05,
        "assay_identity_adjusted_matrix_disagreement": 0.03,
    }
    base = {key: 0.0 for key in thresholds}
    assert classify_materiality(
        {"a500": base, "a1000": base}, thresholds) == "negligible"
    small_only = {**base, "assay_task_token_median_disagreement": 0.021}
    assert classify_materiality(
        {"a500": small_only, "a1000": base}, thresholds
    ) == "material_small_fit_only"
    large = {**base, "assay_identity_adjusted_matrix_disagreement": 0.031}
    assert classify_materiality(
        {"a500": base, "a1000": large}, thresholds
    ) == "material_at_a1000"


def test_prompt323_config_reuses_registered_equal_weight_assertion():
    config = yaml.safe_load((
        ROOT / "configs" / "p4_qwen_lens_influence_prompt323_dev.yaml"
    ).read_text())
    source = config["equal_weight_contract"]["registered_assertion_source"]
    assert source["evidence_id"] == "p4-qwen-lens-influence-prompt112-dev-v1"
    assert len(source["result_sha256"]) == 64
    assert len(source["table_sha256"]) == 64
    assert config["analysis"]["decision_load_bearing_metrics"] == [
        "assay_task_token_median_disagreement",
        "assay_task_token_q05_disagreement",
        "assay_identity_adjusted_matrix_disagreement",
    ]
    assert config["figure"]["stem"] == "p4f29_qwen_prompt323_influence"
