import pandas as pd

from jspace_phase3.experiments.p3_alias_and_cohort_sensitivity import (
    boundary_grade, contains_alias, review_sample)


def test_boundary_matcher_rejects_partial_words():
    assert contains_alias("India.", " India")
    assert not contains_alias("Indian state", " India")
    assert not contains_alias("Dutchman", " Dutch")
    assert contains_alias("Answer: Cu.", " Cu")
    assert not contains_alias("curious", " Cu")
    assert contains_alias("locker 2867.", " 2867")


def test_boundary_grade_records_exact_alias():
    result = boundary_grade(
        "The Thai baht.", [" baht", " the Thai baht"])
    assert result["capable_generation_boundary_safe"]
    assert "the Thai baht" in result["matched_aliases_json"]


def test_manual_sample_is_exactly_100_100_and_model_balanced():
    rows = []
    for model in ("olmo31-think", "olmo31-instruct", "qwen36-27b"):
        for label in (True, False):
            for index in range(120):
                rows.append({
                    "model": model, "item_id": f"{model}:{label}:{index}",
                    "canonical_family": f"family{index % 20}",
                    "variant": "direct", "generation": "answer",
                    "aliases_json": '[" answer"]',
                    "alias_min_words": 1 + index % 3,
                    "alias_max_words": 1 + index % 3,
                    "capable_generation": label,
                    "capable_prefix": label,
                    "capable_generation_boundary_safe": label,
                    "matched_aliases_json": (
                        '[" answer"]' if label else "[]"),
                })
    sample = review_sample(pd.DataFrame(rows))
    assert sample.capable_generation_boundary_safe.value_counts().to_dict() \
        == {True: 100, False: 100}
    assert sample.model.value_counts().to_dict() == {
        "olmo31-think": 68,
        "olmo31-instruct": 66,
        "qwen36-27b": 66,
    }
