import numpy as np
import pandas as pd

from jspace_phase3.experiments.p3_alias_endpoint_audit import (
    ENDPOINTS, aggregate_alias_rows, analyze_cross_model,
    prefix_disjoint_aliases)


def test_prefix_disjoint_exact_selection_prefers_canonical_then_first():
    aliases = [" a", " a long", " b", " c"]
    token_ids = {
        " a": [1],
        " a long": [1, 2],
        " b": [3],
        " c": [4],
    }
    # Both {a,b,c} and {a long,b,c} have maximum cardinality.  Canonical
    # preference wins before historical-first preference.
    assert prefix_disjoint_aliases(
        aliases, token_ids, " a long") == [" a long", " b", " c"]
    assert prefix_disjoint_aliases(
        aliases, token_ids, " a") == [" a", " b", " c"]


def test_prefix_disjoint_keeps_full_safe_set():
    aliases = [" the naira", " naira", " the Nigerian naira"]
    ids = {
        " the naira": [1, 2],
        " naira": [3],
        " the Nigerian naira": [1, 4, 2],
    }
    assert prefix_disjoint_aliases(
        aliases, ids, " naira") == aliases


def _raw_alias_rows():
    rows = []
    for ordinal, alias in enumerate([" first", " canonical"]):
        rows.append({
            "slug": "qwen36-27b",
            "item_id": "f#direct",
            "fact_id": "f",
            "variant": "direct",
            "bank": "F",
            "canonical_family": "fam",
            "relation_group": "rel",
            "alias": alias,
            "alias_ordinal": ordinal,
            "is_canonical_alias": ordinal == 1,
            "in_prefix_disjoint_set": True,
            "lp_baseline": [-2.0, -3.0][ordinal],
            "lp_meanJ_span_safe": [-3.0, -5.0][ordinal],
            "lp_ss_matched": [-2.5, -4.0][ordinal],
            "historical_lp_baseline": -2.1,
            "historical_lp_meanJ_span_safe": -3.2,
            "historical_lp_ss_matched": -2.4,
        })
    return pd.DataFrame(rows)


def test_alias_aggregation_has_all_endpoints_and_exact_lse():
    result = aggregate_alias_rows(_raw_alias_rows())
    assert set(result.endpoint) == set(ENDPOINTS)
    first = result.set_index("endpoint").loc["stable_first_alias"]
    assert first.specific == -0.5
    canonical = result.set_index("endpoint").loc["canonical_alias"]
    assert canonical.specific == -1.0
    lse = result.set_index("endpoint").loc[
        "prefix_disjoint_logsumexp"]
    expected_j = np.logaddexp(-3.0, -5.0)
    expected_control = np.logaddexp(-2.5, -4.0)
    assert abs(lse.specific - (expected_j - expected_control)) < 1e-12
    historical = result.set_index("endpoint").loc[
        "historical_first_alias"]
    assert historical.control_realization.startswith("historical")


def test_cross_analysis_preserves_aliases_as_views_not_rows():
    rows = []
    models = ("olmo31-think", "olmo31-instruct", "qwen36-27b")
    for family_index in range(3):
        for model_index, model in enumerate(models):
            for variant in ("direct", "composed"):
                for endpoint in ENDPOINTS:
                    specific = (
                        model_index
                        + (variant == "composed")
                        * (family_index + 1)
                        * (1 if model == "qwen36-27b" else 0.2)
                    )
                    if endpoint == "canonical_alias":
                        specific += (
                            0.1 if variant == "composed" else 0.0)
                    rows.append({
                        "slug": model,
                        "item_id": f"f{family_index}#{variant}",
                        "fact_id": f"f{family_index}",
                        "variant": variant,
                        "bank": "F",
                        "canonical_family": f"fam{family_index}",
                        "relation_group": "rel",
                        "endpoint": endpoint,
                        "specific": specific,
                        "J_effect": specific - 0.1,
                        "C_effect": -0.1,
                        "n_accepted_aliases": (
                            2 if family_index == 0 else 1),
                    })
    report, paired = analyze_cross_model(pd.DataFrame(rows))
    assert report["P3-P1_subset"]["stable_first_alias"]["n_facts"] == 3
    assert len(paired) == 3 * len(ENDPOINTS)
    assert report["multi_alias_facts_descriptive"][
        "stable_first_alias"]["n_facts"] == 1
