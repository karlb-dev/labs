import json

import pandas as pd


def _grid() -> pd.DataFrame:
    rows = []
    for bank in ("F", "S"):
        for index in range(12):
            fact_id = f"{bank.lower()}-{index:02d}"
            family = f"{bank}-family-{index % 6}"
            for variant in ("direct", "composed"):
                rows.append({
                    "item_id": f"{fact_id}#{variant}",
                    "fact_id": fact_id,
                    "variant": variant,
                    "bank": bank,
                    "canonical_family": family,
                    "relation_group": f"relation-{index % 3}",
                    # Deliberately present but never admitted to the selector.
                    "outcome_score": float(index),
                })
    return pd.DataFrame(rows)


def _guard_rows() -> list[dict]:
    return [
        {
            "item_id": f"guard-{index:03d}",
            "domain": ("grammar_pairs" if index < 4 else
                       f"domain-{index % 4}"),
            "text": f"guard text {index}",
        }
        for index in range(64)
    ]


def _g4_rows() -> list[dict]:
    return [{"name": f"g4-{index:03d}"} for index in range(20)]


def _specification() -> dict:
    return {
        "namespace": "unit-test-functional-subset",
        "primary_facts_per_bank": 8,
        "bridge_facts_per_bank": 4,
        "prose_items_n": 16,
        "capacity_items_n": 8,
        "capacity_positions": [8, 16],
        "max_seq_len": 32,
        "required_primary_families": 10,
        "required_bridge_families": 6,
        "consumed_phase3_only": True,
        "outcome_columns_allowed": False,
        "g4_calibration_n": 3,
        "g4_total_n": 12,
    }


def test_functional_subset_is_deterministic_paired_and_json_safe():
    from jspace_phase4.experiments.p4_qwen_functional_subset import (
        select_functional_subset,
    )
    first = select_functional_subset(
        _grid(), _guard_rows(), _g4_rows(),
        specification=_specification())
    second = select_functional_subset(
        _grid().sample(frac=1, random_state=17),
        list(reversed(_guard_rows())), _g4_rows(),
        specification=_specification())
    assert first == second
    assert first["primary"]["n_facts"] == 16
    assert first["primary"]["n_items"] == 32
    assert first["primary"]["banks"] == {"F": 8, "S": 8}
    assert first["bridge"]["banks"] == {"F": 4, "S": 4}
    assert first["primary"]["n_families"] >= 10
    assert first["bridge"]["n_families"] >= 6
    assert first["g4"]["item_names"] == [
        f"g4-{index:03d}" for index in range(12)]
    json.dumps(first, sort_keys=True)


def test_functional_subset_does_not_use_grid_outcomes():
    from jspace_phase4.experiments.p4_qwen_functional_subset import (
        select_functional_subset,
    )
    grid = _grid()
    first = select_functional_subset(
        grid, _guard_rows(), _g4_rows(),
        specification=_specification())
    grid["outcome_score"] = list(reversed(grid["outcome_score"].tolist()))
    second = select_functional_subset(
        grid, _guard_rows(), _g4_rows(),
        specification=_specification())
    assert first == second


def test_functional_subset_refuses_outcome_or_unconsumed_contracts():
    from jspace_phase4.experiments.p4_qwen_functional_subset import (
        select_functional_subset,
    )
    for field, value, message in (
        ("outcome_columns_allowed", True, "outcome"),
        ("consumed_phase3_only", False, "consumed Phase 3"),
    ):
        specification = _specification()
        specification[field] = value
        try:
            select_functional_subset(
                _grid(), _guard_rows(), _g4_rows(),
                specification=specification)
        except RuntimeError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"unsafe contract {field} was accepted")
