from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from jspace_phase4.experiments import p4_lineage_common_cohort_analysis as cc


def _frame(facts: list[str], *, offset: float = 0.0) -> pd.DataFrame:
    rows = []
    for index, fact_id in enumerate(facts):
        for variant_index, variant in enumerate(("direct", "composed")):
            base = float(index + variant_index + offset)
            rows.append({
                "fact_id": fact_id,
                "item_id": f"{fact_id}#{variant}",
                "canonical_family": fact_id.split(":", 1)[0],
                "bank": "F" if fact_id.startswith("f") else "S",
                "variant": variant,
                "lp_baseline": base,
                "lp_meanJ_span_safe": base - 0.4,
                "lp_ss_matched": base - 0.1,
                "lp_meanJ_label_protected": base - 0.3,
                "lp_prot_energy_matched": base - 0.05,
                "lp_mechanics_random": base - 0.02,
                "lp_logit_label_protected": base - 0.01,
                "capability_margin": 2.0 + base,
            })
    return pd.DataFrame(rows)


def _checkpoint_frames() -> dict[str, pd.DataFrame]:
    return {
        "olmo3-base": _frame(["famf:f1", "fams:s1", "fams:s2"]),
        "olmo3-think": _frame([
            "famf:f1", "fams:s1", "fams:s2", "fams:s3"
        ], offset=0.1),
        "olmo31-think": _frame([
            "famf:f1", "fams:s1", "fams:s2", "famf:f2"
        ], offset=0.2),
        "olmo31-instruct": _frame([
            "famf:f1", "fams:s1", "fams:s2", "fams:s4"
        ], offset=0.3),
    }


def test_population_membership_is_outcome_blind():
    frames = _checkpoint_frames()
    populations, manifest = cc.construct_populations(frames)
    assert populations["all_four"]["fact_ids"] == [
        "famf:f1", "fams:s1", "fams:s2"
    ]
    assert populations["base_vs_30think"]["n_facts"] == 3
    assert populations["30think_vs_31think"]["n_facts"] == 3
    assert populations["31think_vs_31instruct"]["n_facts"] == 3
    assert all("fact_ids" not in row for row in manifest)

    changed = {
        key: frame.assign(
            lp_meanJ_span_safe=np.arange(len(frame), dtype=float) * 1000,
            lp_ss_matched=-999.0,
        )
        for key, frame in frames.items()
    }
    changed_populations, changed_manifest = cc.construct_populations(changed)
    assert changed_manifest == manifest
    assert changed_populations["all_four"]["fact_ids"] == (
        populations["all_four"]["fact_ids"])


def test_population_rejects_missing_variant_and_metadata_drift():
    frames = _checkpoint_frames()
    frames["olmo3-think"] = frames["olmo3-think"].iloc[1:].copy()
    with pytest.raises(ValueError, match="direct and composed"):
        cc.construct_populations(frames)

    frames = _checkpoint_frames()
    mask = frames["olmo31-think"].fact_id == "fams:s1"
    frames["olmo31-think"].loc[mask, "canonical_family"] = "changed"
    with pytest.raises(ValueError, match="metadata changes"):
        cc.construct_populations(frames)


def test_metric_rows_and_paired_contrast_are_exact():
    left = cc.metric_fact_rows(
        cc.add_effects(_frame([
            "famf:f1", "famf:f2", "famf:f3"
        ])),
        bank="F",
        metric="composition",
    )
    right_grid = _frame([
        "famf:f1", "famf:f2", "famf:f3"
    ], offset=1.0)
    # Change only the composed J arm at the right checkpoint.
    right_grid.loc[
        right_grid.variant == "composed", "lp_meanJ_span_safe"
    ] -= 0.2
    right = cc.metric_fact_rows(
        cc.add_effects(right_grid), bank="F", metric="composition")
    assert np.allclose(left.value, 0.0)
    assert np.allclose(right.value, -0.2)
    paired = cc.paired_contrast(left, right)
    assert np.allclose(paired.specific_delta, -0.2)
    assert np.allclose(paired.baseline_lp_delta, 0.0)


def test_bootstrap_and_adjusted_sensitivity_are_deterministic():
    first = cc.bootstrap_vector(
        np.asarray([1.0, 2.0, 4.0]), draws=1000, seed=7)
    second = cc.bootstrap_vector(
        np.asarray([1.0, 2.0, 4.0]), draws=1000, seed=7)
    assert first == second
    assert first["estimate"] == pytest.approx(7 / 3)

    x = np.asarray([-1.0, 0.0, 1.0, 2.0])
    y = 2.5 + 3.0 * x
    intercept, slope = cc._ols_intercept(x, y)
    assert intercept == pytest.approx(2.5)
    assert slope == pytest.approx(3.0)
    summary = cc.regression_bootstrap(
        x, y, draws=1000, seed=9, chunk_size=100)
    assert summary["estimate"] == pytest.approx(2.5)
    assert summary["slope"] == pytest.approx(3.0)


def test_common_cohort_config_freezes_development_evidence():
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/p4_lineage_common_cohort_olmo_dev.yaml"
    )
    config = yaml.safe_load(path.read_text())
    assert config["evidence_id"] == (
        "p4-lineage-common-cohort-analysis-olmo-dev-v1")
    assert config["tier"] == "phase4-development"
    assert tuple(row["key"] for row in config["checkpoints"]) == (
        cc.CHECKPOINT_ORDER)
    assert config["bootstrap_draws"] == 100_000
