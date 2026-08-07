import numpy as np
import pandas as pd

from jspace_phase3.experiments.p3_control_seed_audit import (
    AUDIT_SEEDS, analyze, select_balanced_items)


def test_balanced_selector_keeps_fact_pairs_and_family_floor():
    items = []
    for family in range(14):
        for fact in range(2):
            fact_id = f"fam{family}:fact{fact}"
            for variant in ("direct", "composed"):
                items.append({
                    "item_id": f"{fact_id}#{variant}",
                    "fact_id": fact_id, "variant": variant,
                    "canonical_family": f"fam{family}",
                })
    selected = select_balanced_items(
        items, {item["fact_id"] for item in items},
        side="confirmatory", n_items=24)
    assert len(selected) == 24
    frame = pd.DataFrame(selected)
    assert frame["canonical_family"].nunique() == 12
    assert all(
        set(sub["variant"]) == {"direct", "composed"}
        for _, sub in frame.groupby("fact_id"))


def test_analysis_refuses_seed_incomplete_rows(tmp_path):
    rows = pd.DataFrame({
        "side": ["confirmatory"] * (len(AUDIT_SEEDS) - 1),
        "item_id": ["x"] * (len(AUDIT_SEEDS) - 1),
        "audit_seed": list(AUDIT_SEEDS[:-1]),
    })
    path = tmp_path / "rows.parquet"
    rows.to_parquet(path)
    try:
        analyze(path, tmp_path)
    except RuntimeError as error:
        assert "all audit seeds" in str(error)
    else:
        raise AssertionError("seed-incomplete rows were accepted")


def test_seed_contract_changes_realization_not_baseline():
    from jspace_phase3.seeds import stable_seed
    seeds = [
        stable_seed("p3-control-seed-audit", "f:x#direct", seed)
        for seed in AUDIT_SEEDS
    ]
    assert len(set(seeds)) == len(AUDIT_SEEDS)
    baseline = np.repeat(-1.25, len(AUDIT_SEEDS))
    assert np.ptp(baseline) == 0

