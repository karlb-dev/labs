import pandas as pd
import pytest

from jspace_phase4.experiments.p4_lineage_trajectory_analysis import (
    independently_validate_rows,
    summarize_checkpoint,
    validate_shared_contract,
)


def _trajectory_frame(*, common: bool = False) -> pd.DataFrame:
    base = {
        ("F", "direct"): -0.30,
        ("F", "composed"): 0.10,
        ("S", "direct"): -0.20,
        ("S", "composed"): -0.05,
    }
    common_delta = {
        ("F", "direct"): 0.08,
        ("F", "composed"): -0.03,
        ("S", "direct"): -0.02,
        ("S", "composed"): 0.04,
    }
    rows = []
    for bank in ("F", "S"):
        for family_index, family_offset in enumerate((-0.10, 0.0, 0.10)):
            fact_id = f"{bank.lower()}{family_index}"
            for variant in ("direct", "composed"):
                specific = base[(bank, variant)] + family_offset
                if common:
                    specific += common_delta[(bank, variant)]
                rows.append({
                    "item_id": f"{fact_id}#{variant}",
                    "bank": bank,
                    "fact_id": fact_id,
                    "canonical_family": f"{bank}-family-{family_index}",
                    "variant": variant,
                    "lp_baseline": -1.0,
                    "lp_meanJ_span_safe": -1.0 + specific,
                    "lp_ss_matched": -1.0,
                    "lp_meanJ_label_protected": -1.0,
                    "lp_prot_energy_matched": -1.0,
                    "lp_mechanics_random": -1.0,
                    "lp_logit_label_protected": -1.0,
                })
    return pd.DataFrame(rows)


def _checkpoint(*, common_is_own: bool) -> dict:
    return {
        "key": "test-checkpoint",
        "label": "Test",
        "lineage_role": "think_path",
        "common_is_own": common_is_own,
        "own_grid_evidence_id": "own-grid-v1",
        "common_grid_evidence_id": "common-grid-v1",
    }


def _by_metric(rows, frame):
    return {
        row["metric_key"]: row
        for row in rows
        if row["frame"] == frame
    }


def test_trajectory_summary_effects_and_composition():
    rows, validation = summarize_checkpoint(
        _trajectory_frame(),
        _trajectory_frame(common=True),
        checkpoint=_checkpoint(common_is_own=False),
        checkpoint_index=1,
        draws=2000,
        seed=1234,
    )
    assert len(rows) == 12
    own = _by_metric(rows, "own")
    common = _by_metric(rows, "common")
    assert own["F:direct"]["estimate"] == pytest.approx(-0.30)
    assert own["F:composed"]["estimate"] == pytest.approx(0.10)
    assert own["F:composition"]["estimate"] == pytest.approx(0.40)
    assert own["S:composition"]["estimate"] == pytest.approx(0.15)
    assert common["F:direct"]["estimate"] == pytest.approx(-0.22)
    assert common["F:composition"]["estimate"] == pytest.approx(0.29)
    audit = independently_validate_rows(rows, validation)
    assert audit["all_exact"]
    assert audit["n_summaries_reconstructed"] == 12


def test_base_common_frame_is_exact_own_frame_copy():
    rows, validation = summarize_checkpoint(
        _trajectory_frame(),
        _trajectory_frame(common=True),
        checkpoint=_checkpoint(common_is_own=True),
        checkpoint_index=0,
        draws=2000,
        seed=1234,
    )
    own = _by_metric(rows, "own")
    common = _by_metric(rows, "common")
    for metric_key in own:
        for field in (
                "estimate",
                "ci95_low",
                "ci95_high",
                "distribution_sha256"):
            assert own[metric_key][field] == common[metric_key][field]
    audit = independently_validate_rows(rows, validation)
    assert audit["n_summaries_reconstructed"] == 12
    assert audit["n_unique_distributions"] == 6


def test_trajectory_requires_shared_bank_and_scoring_contract():
    first = {"bank_sha256": "a", "scoring_spec_sha256": "b"}
    second = {"bank_sha256": "a", "scoring_spec_sha256": "b"}
    assert validate_shared_contract([first, second]) == first
    second["bank_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="bank_sha256 mismatch"):
        validate_shared_contract([first, second])
