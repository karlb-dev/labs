# family_split_v2 suites (§5.6): determinism, ACTIVE seed, floors,
# disjointness, imbalance bound. The Phase 2 splitter recorded a seed a
# dead branch never used — the seed-activity test here is the direct
# regression against that defect class.
import numpy as np
import pandas as pd
import pytest

from jspace_phase3.family_split import (Partition, SplitConstraints,
                                        split_families_v2)


def synth_table(n=80, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append({
            "canonical_family": f"fam{i:03d}",
            "bank": "F" if i < n * 3 // 4 else "S",
            "relation_group": f"g{i % 5}",
            "n_items": int(rng.integers(9, 19)),
            "n_counterfactual": int(rng.integers(3, 6)),
            "intersection_capable": bool(i % 3 != 0),
            "median_lp_think": float(rng.normal(-1.5, 0.4)),
            "median_lp_instruct": float(rng.normal(-1.4, 0.4)),
            "median_lp_qwen": float(rng.normal(-1.2, 0.4)),
            "answer_len_mean": float(rng.normal(2.2, 0.5)),
        })
    return pd.DataFrame(rows)


CONS = SplitConstraints(min_twohop_families_per_side=25,
                        min_intersection_families_per_side=20,
                        max_standardized_imbalance=0.5,
                        seed=4242, restarts=8, sweeps=120)


def test_deterministic_given_seed():
    t = synth_table()
    p1 = split_families_v2(t, CONS)
    p2 = split_families_v2(t, CONS)
    assert p1.confirmatory == p2.confirmatory
    assert p1.replication == p2.replication


def test_seed_is_active():
    """Different seeds must be able to produce different assignments on
    a symmetric table — the direct regression on the Phase 2 defect."""
    t = synth_table()
    outs = {split_families_v2(
        t, SplitConstraints(**{**CONS.__dict__, "seed": s})).confirmatory
        for s in (1, 2, 3)}
    assert len(outs) > 1


def test_floors_enforced():
    t = synth_table(n=30)          # too few F families for the floor
    with pytest.raises(RuntimeError, match="no feasible split"):
        split_families_v2(t, SplitConstraints(
            min_twohop_families_per_side=40,
            min_intersection_families_per_side=1,
            max_standardized_imbalance=2.0, seed=1,
            restarts=4, sweeps=40))


def test_balance_and_report():
    t = synth_table()
    p = split_families_v2(t, CONS)
    rep = p.balance_report
    assert max(rep["per_dimension"].values()) <= 0.5
    assert min(rep["twohop_F"]) >= 25
    assert min(rep["intersection"]) >= 20
    assert rep["n_confirmatory"] + rep["n_replication"] == len(t)


def test_assert_disjoint_catches_cross_partition_hash():
    t = synth_table(n=60)
    p = split_families_v2(t, SplitConstraints(
        min_twohop_families_per_side=18,
        min_intersection_families_per_side=15,
        max_standardized_imbalance=0.6, seed=9, restarts=6, sweeps=80))
    items = pd.DataFrame({
        "canonical_family": list(t.canonical_family) * 2,
        "fact_id": [f"{f}:x{k}" for k in (0, 1)
                    for f in t.canonical_family],
        "template_hash": [f"h{i}" for i in range(len(t))] * 2})
    p.assert_disjoint(items, "canonical_family", "fact_id")
    # template h0 belongs to two different families -> may cross sides
    items.loc[len(t):, "template_hash"] = "h0"
    fam0_side = p.side_of(t.canonical_family[0])
    crossers = [f for f in t.canonical_family if p.side_of(f) != fam0_side]
    items.loc[items.index[-1], "canonical_family"] = crossers[0]
    with pytest.raises(AssertionError, match="template_hash"):
        p.assert_disjoint(items, "template_hash")
