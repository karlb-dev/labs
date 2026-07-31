from pathlib import Path

import numpy as np

from jspace_phase3.experiments.p3_n8_phase3_analysis import exact_signflip
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
