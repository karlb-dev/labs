import json
from pathlib import Path

import yaml


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
