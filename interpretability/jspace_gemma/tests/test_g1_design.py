import json
from pathlib import Path

import yaml

from jspace_gemma.experiments.gm_band_convention import mapped_band


def test_g1_design_is_frozen_but_target_thresholds_are_firewalled():
    root = Path(__file__).resolve().parents[1]
    design = yaml.safe_load((root / "configs/gm_g1_design.yaml").read_text())
    assert len(design["stage1_prompt_ids"]) == 4
    assert len(design["stage2_prompt_ids"]) == 16
    assert design["models"]["gemma_target"]["layers_zero_indexed"] == [22, 30, 37, 44, 52]
    assert design["models"]["olmo_positive_control"]["layers_zero_indexed"] == [4, 24, 32, 40, 47, 56, 60]
    assert design["relative_epsilon_ladder"] == [0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    assert design["delivery_gate"] == {
        "cosine_floor": 0.999,
        "relative_norm_error_ceiling": 0.01,
        "below_gate": "unmeasurable",
    }
    calibration = design["threshold_calibration"]
    assert calibration["status"] == "REQUIRED_BEFORE_GEMMA"
    assert calibration["gemma_execution_allowed"] is False
    assert calibration["tangent_vs_secant"] is None
    assert design["forbidden_exact_backend"] == "finite_difference"
    assert design["batch_alignment"]["finite_baseline_same_batch_shape_and_slot"]
    assert design["batch_alignment"]["exact_jvp_same_batch_shape_and_slot"]
    assert design["response_snr_contract"] == {
        "signal_norm": "min_finite_response_and_exact_jvp_norm",
        "noise_norm": "max_in_batch_clean_repeat_and_target_dtype_half_step_norm",
        "numeric_floor_is_calibrated_on_olmo": True,
    }
    assert all(design["realized_delivery_adjustment"].values())


def test_prompt_bank_membership_and_strata_are_exact():
    root = Path(__file__).resolve().parents[1]
    rows = [
        json.loads(line)
        for line in (root / "data/g1_prompts_v1.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [row["prompt_id"] for row in rows] == [f"gm-p{i:03d}" for i in range(1, 17)]
    stage1 = [row for row in rows if row["stage"] == 1]
    assert {row["stratum"] for row in stage1} == {
        "factual", "multi-hop", "neutral-prose", "code-sql"
    }
    assert len({row["text"] for row in rows}) == 16


def test_primary_paper_band_maps_to_the_addendum_gemma_range():
    root = Path(__file__).resolve().parents[1]
    convention = yaml.safe_load(
        (root / "configs/gm_g6_band_convention.yaml").read_text()
    )
    resolution = convention["resolution"]
    mapped = mapped_band(
        resolution["transferable_workspace_depth_fraction"],
        resolution["gemma_decoder_blocks"],
    )
    assert mapped == [23, 55]
    assert all(
        mapped[0] <= layer <= mapped[1]
        for layer in resolution["g6_candidate_layers_zero_indexed"]
    )
