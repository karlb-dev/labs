def _country(name, cca2, cca3, capital):
    return {
        "name": {
            "common": name,
            "official": f"Republic of {name}",
            "nativeName": {"eng": {
                "common": name, "official": f"Republic of {name}"}},
        },
        "altSpellings": [cca2, cca3],
        "cca2": cca2,
        "cca3": cca3,
        "capital": [capital],
        "demonyms": {"eng": {"m": f"{name}ian", "f": f"{name}ian"}},
        "translations": {
            "fra": {"common": name, "official": f"République de {name}"},
            "spa": {"common": name, "official": f"República de {name}"},
        },
        "tld": [f".{cca2.lower()}"],
    }


def _config():
    return {
        "independent_source": {
            "name": "REST Countries",
            "repository": "https://example.test/source",
            "revision": "a" * 40,
            "source_path": "countries.json",
            "snapshot_sha256": "b" * 64,
        },
        "verification": {
            "require_unique_country_resolution": True,
            "require_true_source_match": True,
            "require_true_answer_match": True,
            "require_alternate_answer_match": True,
            "require_every_counterfactual_answer_match": True,
            "require_every_counterfactual_alternate_answer_match": True,
            "require_unrelated_bridge_resolution": True,
            "manual_review_multi_valued_answer_relations": True,
            "geopolitical_manual_review_names": [],
        },
    }


def _row():
    return {
        "fact_id": "fact:x",
        "canonical_family": "alpha2__to__capital",
        "partition": "development",
        "source": "XX",
        "source_type": "alpha2",
        "answer": "X City",
        "answer_type": "capital",
        "alternate_answer": "XXX",
        "alternate_relation": "alpha3",
        "bridge": "Xland",
        "counterfactuals": [{
            "bridge": "Yland", "answer": "Y City",
            "alternate_answer": "YYY"}],
        "unrelated_bridge": "Zland",
    }


def _source_row():
    return {
        "source_record": "countryinfo/data/xland.json",
        "source_record_sha256": "c" * 64,
        "verification_status": (
            "candidate-single-source-pending-independent-verification"),
    }


def test_bank_b_verification_normalizes_accents_and_resolves_aliases():
    from jspace_phase4.experiments.p4_bank_b_restcountries_verification import (
        CountryIndex,
        normalized_forms,
    )

    assert "iles malouines" in normalized_forms("Îles Malouines")
    index = CountryIndex([_country("Xland", "XX", "XXX", "X City")])
    result = index.resolve("Republic of Xland")
    assert result["n_candidates"] == 1
    assert result["country"]["cca3"] == "XXX"

    index = CountryIndex(
        [_country("North Xland", "XX", "XXX", "X City")],
        aliases={"Republic of Xland": {
            "cca3": "XXX", "manual_review": True,
            "reason": "outdated name"}})
    alias = index.resolve("Republic of Xland")
    assert alias["n_candidates"] == 1
    assert alias["resolution_method"] == "configured_name_alias"
    assert alias["alias_manual_review"]


def test_bank_b_verification_passes_complete_independent_match():
    from jspace_phase4.experiments.p4_bank_b_restcountries_verification import (
        CountryIndex,
        verify_fact,
    )

    countries = [
        _country("Xland", "XX", "XXX", "X City"),
        _country("Yland", "YY", "YYY", "Y City"),
        _country("Zland", "ZZ", "ZZZ", "Z City"),
    ]
    result = verify_fact(
        _row(), _source_row(), index=CountryIndex(countries),
        config=_config())
    assert result["verification_status"] == "verified-exact-unambiguous"
    assert result["independent_match"]
    assert not result["manual_review_required"]
    assert all(result["required_checks"].values())


def test_bank_b_verification_keeps_match_and_ambiguity_separate():
    from jspace_phase4.experiments.p4_bank_b_restcountries_verification import (
        CountryIndex,
        verify_fact,
    )

    countries = [
        _country("Xland", "XX", "XXX", "X City"),
        _country("Yland", "YY", "YYY", "Y City"),
        _country("Zland", "ZZ", "ZZZ", "Z City"),
    ]
    countries[0]["capital"].append("X Administrative City")
    result = verify_fact(
        _row(), _source_row(), index=CountryIndex(countries),
        config=_config())
    assert result["independent_match"]
    assert result["manual_review_required"]
    assert result["verification_status"] == \
        "independent-match-manual-ambiguity-review"


def test_bank_b_verification_mismatch_is_not_certified():
    from jspace_phase4.experiments.p4_bank_b_restcountries_verification import (
        CountryIndex,
        verify_fact,
    )

    countries = [
        _country("Xland", "XX", "XXX", "Different City"),
        _country("Yland", "YY", "YYY", "Y City"),
        _country("Zland", "ZZ", "ZZZ", "Z City"),
    ]
    result = verify_fact(
        _row(), _source_row(), index=CountryIndex(countries),
        config=_config())
    assert not result["independent_match"]
    assert result["verification_status"] == "independent-source-mismatch"
    assert not result["required_checks"]["true_answer_matches"]


def test_bank_b_verification_resolves_complete_multivalue_alias_coverage():
    from jspace_phase4.experiments.p4_bank_b_restcountries_verification import (
        CountryIndex,
        verify_fact,
    )

    countries = [
        _country("Xland", "XX", "XXX", "X City"),
        _country("Yland", "YY", "YYY", "Y City"),
        _country("Zland", "ZZ", "ZZZ", "Z City"),
    ]
    countries[0]["name"]["nativeName"]["xla"] = {
        "common": "Xish", "official": "Republic of Xish"}
    row = _row()
    row.update({
        "answer": "Xland",
        "answer_type": "nativeName",
        "accepted_answers": [" Xland", " Xish"],
    })
    row["counterfactuals"][0]["answer"] = "Yland"
    row["counterfactuals"][0]["accepted_answers"] = [" Yland"]
    config = _config()
    config["verification"][
        "resolve_multi_valued_by_complete_alias_coverage"] = True
    result = verify_fact(
        row, _source_row(), index=CountryIndex(countries), config=config)
    assert result["independent_match"]
    assert not result["manual_review_required"]
    assert result["reviewed_ambiguity"]
    assert result["verification_status"] == \
        "verified-exact-reviewed-ambiguity"
    assert result["ambiguity_resolutions"][0]["coverage"]["passes"]

    row["accepted_answers"] = [" Xland"]
    unresolved = verify_fact(
        row, _source_row(), index=CountryIndex(countries), config=config)
    assert unresolved["manual_review_required"]
    assert unresolved["verification_status"] == \
        "independent-match-manual-ambiguity-review"


def test_bank_b_verification_preserves_configured_geopolitical_review():
    from jspace_phase4.experiments.p4_bank_b_restcountries_verification import (
        CountryIndex,
        verify_fact,
    )

    countries = [
        _country("Xland", "XX", "XXX", "X City"),
        _country("Yland", "YY", "YYY", "Y City"),
        _country("Zland", "ZZ", "ZZZ", "Z City"),
    ]
    config = _config()
    config["verification"]["geopolitical_manual_review_names"] = ["Xland"]
    config["verification"]["geopolitical_review_resolutions"] = {
        "Xland": "metadata-only inclusion; no sovereignty inference"}
    result = verify_fact(
        _row(), _source_row(), index=CountryIndex(countries), config=config)
    assert result["verification_status"] == \
        "verified-exact-reviewed-ambiguity"
    assert result["reviewed_ambiguity"]
    assert not result["manual_review_required"]
