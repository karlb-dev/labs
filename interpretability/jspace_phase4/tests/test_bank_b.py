import json


class _Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return type("Tokens", (), {
            "input_ids": [ord(value) % 97 for value in text.strip()]})()


def _record(index: int) -> dict:
    return {
        "name": f"Country {index}",
        "ISO": {"alpha2": f"X{index}", "alpha3": f"Q{index:02d}"},
        "capital": f"Capital {index}",
        "demonym": f"Demonym {index}",
        "tld": [f".x{index}"],
        "callingCodes": [f"7{index:02d}"],
        "currencies": [f"C{index:02d}"],
        "languages": [f"l{index}"],
        "nativeName": f"Native {index}",
        "translations": {
            "fr": f"French {index}", "es": f"Spanish {index}"},
        "wiki": f"https://example.test/country-{index}",
        "_source_record": f"countryinfo/data/country_{index}.json",
        "_source_sha256": f"{index:064x}"[-64:],
    }


def test_bank_b_family_and_assignment_are_deterministic_and_unique():
    from jspace_phase4.experiments.p4_author_bank_b import (
        assign_unique_records,
        choose_family_specs,
    )
    records = [_record(index) for index in range(20)]
    kwargs = dict(
        source_types=["capital", "demonym"],
        answer_types=["alpha2", "alpha3", "currency"],
        prior=set(), excluded=set(), minimum_eligible=10,
        target_families=4, maximum_characters=50,
        namespace="bank-b-unit",
    )
    first = choose_family_specs(records, **kwargs)
    second = choose_family_specs(list(reversed(records)), **kwargs)
    assert [row["family_id"] for row in first] == [
        row["family_id"] for row in second]
    assignment = assign_unique_records(
        first, facts_per_family=4, namespace="bank-b-unit")
    names = [record["name"] for rows in assignment.values()
             for record in rows]
    assert len(names) == len(set(names)) == 16


def test_bank_b_prior_overlap_is_excluded_before_assignment():
    from jspace_phase4.experiments.p4_author_bank_b import (
        choose_family_specs,
        normalize,
    )
    records = [_record(index) for index in range(12)]
    prior = {normalize("Country 0"), normalize("Capital 1")}
    families = choose_family_specs(
        records, source_types=["capital"],
        answer_types=["alpha2", "currency"], prior=prior,
        excluded=set(), minimum_eligible=5, target_families=2,
        maximum_characters=50, namespace="prior-test")
    for family in families:
        assert all(record["name"] != "Country 0"
                   for record in family["eligible"])
        assert all(record["capital"] != "Capital 1"
                   for record in family["eligible"])


def test_bank_b_source_schema_marks_independent_verification_explicit():
    schema = json.loads(open(
        "interpretability/jspace_phase4/data/"
        "bank_b_source_schema_v1.json").read())
    status = schema["properties"]["verification_status"]["enum"]
    assert "candidate-single-source-pending-independent-verification" in status
    assert "independently-verified" in status
    assert "independent_sources" in schema["required"]
