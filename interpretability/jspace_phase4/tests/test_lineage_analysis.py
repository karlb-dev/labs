import pandas as pd
import pytest

from jspace_phase4.experiments.p4_lineage_analysis import (
    add_effects,
    composition_rows,
)
from jspace_phase4.experiments.p4_lens_frame_analysis import (
    composition_frame,
    pair_effects,
)


def test_lineage_effect_and_composition_definitions():
    frame = pd.DataFrame([
        {
            "bank": "S",
            "fact_id": "f1",
            "canonical_family": "family",
            "variant": "direct",
            "lp_baseline": -1.0,
            "lp_meanJ_span_safe": -1.6,
            "lp_ss_matched": -1.2,
            "lp_meanJ_label_protected": -1.3,
            "lp_prot_energy_matched": -1.1,
            "lp_mechanics_random": -1.05,
            "lp_logit_label_protected": -1.4,
        },
        {
            "bank": "S",
            "fact_id": "f1",
            "canonical_family": "family",
            "variant": "composed",
            "lp_baseline": -2.0,
            "lp_meanJ_span_safe": -2.3,
            "lp_ss_matched": -2.1,
            "lp_meanJ_label_protected": -2.2,
            "lp_prot_energy_matched": -2.1,
            "lp_mechanics_random": -2.05,
            "lp_logit_label_protected": -2.4,
        },
    ])
    effects = add_effects(frame)
    assert effects.loc[0, "specific"] == pytest.approx(-0.4)
    assert effects.loc[1, "specific"] == pytest.approx(-0.2)
    composition = composition_rows(effects)
    assert composition.iloc[0].composition == pytest.approx(0.2)


def _lens_frame_rows(*, common: bool = False) -> pd.DataFrame:
    baseline = -1.0
    if common:
        direct = (-1.3, -1.2)
        composed = (-1.5, -1.1)
    else:
        direct = (-1.4, -1.1)
        composed = (-1.2, -1.1)
    rows = []
    for variant, (j_value, control_value) in (
            ("direct", direct), ("composed", composed)):
        rows.append({
            "item_id": f"f1#{variant}",
            "bank": "S",
            "fact_id": "f1",
            "canonical_family": "family",
            "variant": variant,
            "lp_baseline": baseline,
            "lp_meanJ_span_safe": j_value,
            "lp_ss_matched": control_value,
            "lp_meanJ_label_protected": -1.3,
            "lp_prot_energy_matched": -1.1,
            "lp_mechanics_random": -1.05,
            "lp_logit_label_protected": -1.4,
        })
    return pd.DataFrame(rows)


def test_paired_lens_frame_effect_and_composition_definitions():
    paired = pair_effects(
        _lens_frame_rows(),
        _lens_frame_rows(common=True),
        baseline_tolerance=1e-8,
    )
    direct = paired[paired.variant == "direct"].iloc[0]
    assert direct.specific_own == pytest.approx(-0.3)
    assert direct.specific_common == pytest.approx(-0.1)
    assert direct.specific_delta == pytest.approx(0.2)
    composition = composition_frame(paired).iloc[0]
    assert composition.composition_own == pytest.approx(0.2)
    assert composition.composition_common == pytest.approx(-0.3)
    assert composition.composition_delta == pytest.approx(-0.5)


def test_paired_lens_frame_refuses_baseline_drift():
    common = _lens_frame_rows(common=True)
    common.loc[0, "lp_baseline"] -= 0.01
    with pytest.raises(ValueError, match="baseline drift"):
        pair_effects(
            _lens_frame_rows(),
            common,
            baseline_tolerance=1e-6,
        )
