from pathlib import Path

import numpy as np
import pandas as pd

from jspace_phase3.experiments.p3_n8_phase3_analysis import exact_signflip
from jspace_phase3.experiments.p3_n8_phase3_cells import (
    deviation_block, family_tail_effect, select_sentinel_from_frame)
from jspace_phase3.experiments.p3_n8_phase3_compare import get


def test_n8_exact_signflip_independent_result():
    values = np.array([-1.0, -0.5, 0.2, -0.7])
    result = exact_signflip(values)
    brute = []
    for bits in range(16):
        signs = 1 - 2 * ((bits >> np.arange(4)) & 1)
        brute.append(float((signs * values).mean()))
    expected = np.mean(np.abs(brute) >= abs(values.mean()) - 1e-15)
    assert result["p"] == expected


def test_n8_runner_source_has_no_campaign_outcome_access():
    source = (
        Path(__file__).resolve().parents[1]
        / "jspace_phase3" / "experiments"
        / "p3_n8_phase3_analysis.py"
    ).read_text()
    forbidden = (
        "REPORT_PHASE3", "evidence_events", "phase3_locked_analysis.json",
        "p3_inference_audit.json", "p3_protected_answer_audit.json",
        "from ..stats", "phase3_locked_analysis import",
    )
    assert not [needle for needle in forbidden if needle in source]


def test_n8_comparison_paths_preserve_decimal_json_keys():
    value = {"curve": {"-1.0": {"estimate": 0.25}}}
    assert get(value, ("curve", "-1.0", "estimate")) == 0.25


def test_n8_cell_deviation_ignores_paired_null_bridge_rows():
    merged = pd.DataFrame({
        "lp_baseline_new": [1.0, 2.0],
        "lp_baseline_frozen": [1.0, 2.001],
        "lp_true_bridge_new": [np.nan, 3.0],
        "lp_true_bridge_frozen": [np.nan, 3.0],
    })
    result = deviation_block(
        merged, ["lp_baseline", "lp_true_bridge"])
    assert result["lp_baseline"]["max_abs_error"] < 0.0011
    assert result["lp_true_bridge"]["n"] == 1


def test_n8_cell_family_tail_effect_is_equal_family_weighted():
    frame = pd.DataFrame({
        "canonical_family": ["a", "a", "b"],
        "lp_baseline": [0.0, 0.0, 0.0],
        "lp_meanJ_span_safe": [-2.0, -2.0, 0.0],
        "lp_ss_matched": [0.0, -2.0, 0.0],
    })
    # family a has item differences [1, 0] -> .5; family b -> 0.
    assert family_tail_effect(frame, "lp_ss_matched") == 0.25


def test_n8_cell_sentinel_is_paired_and_family_stratified():
    rows = []
    for family in range(12):
        for fact in range(2):
            for variant in ("direct", "composed"):
                fact_id = f"f{family}:x{fact}"
                rows.append({
                    "item_id": f"{fact_id}#{variant}",
                    "fact_id": fact_id,
                    "variant": variant,
                    "canonical_family": f"f{family}",
                })
    frame = pd.DataFrame(rows)
    selected = select_sentinel_from_frame(frame, 20)
    chosen = frame[frame["item_id"].isin(selected)]
    assert len(selected) == 20
    assert chosen["fact_id"].nunique() == 10
    assert chosen["canonical_family"].nunique() == 10
    assert all(
        set(group["variant"]) == {"direct", "composed"}
        for _, group in chosen.groupby("fact_id")
    )
