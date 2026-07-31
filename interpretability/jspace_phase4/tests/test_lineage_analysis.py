import pandas as pd
import pytest

from jspace_phase4.experiments.p4_lineage_analysis import (
    add_effects,
    composition_rows,
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
