# bank.py validator suites (nextsteps §5.4 + the S-bank final-query
# semantics: in-context composition necessarily states the answer in a
# definition, so leakage binds the final query sentence for bank S).
from jspace_phase3.bank import FactBundle, validate_bank, validate_bundle


def f_bundle(**kw):
    base = dict(
        fact_id="capital_to_currency:testville",
        canonical_family="capital_to_currency",
        relation_group="geo_political", bank="F",
        source="Testville", bridge="Testland", answer="quill",
        accepted_answers=[" the quill", " quill"],
        prompts=dict(
            direct="The official currency of Testland is",
            composed="The official currency of the country whose capital "
                     "city is Testville is",
            bridge_supplied="Testville is the capital of Testland. The "
                            "official currency of Testland is"),
        counterfactual_bridge="Otherland", counterfactual_answer="peck",
        counterfactual_accepted=[" peck"],
        provenance={"source": "wikipedia-20231101.en"})
    base.update(kw)
    return FactBundle(**base)


def s_bundle(**kw):
    base = dict(
        fact_id="s_two_codes:varnel", canonical_family="s_two_codes",
        relation_group="synthetic_lookup", bank="S",
        source="Varnel", bridge="Toskin", answer="Merrid",
        accepted_answers=[" Merrid"],
        prompts=dict(
            direct="In code B, the word Toskin stands for Merrid. "
                   "According to code B, Toskin translates to",
            composed="In code A, the word Varnel stands for Toskin. In "
                     "code B, the word Toskin stands for Merrid. "
                     "Translating Varnel through code A and then code B "
                     "gives",
            bridge_supplied="In code A, the word Varnel stands for "
                            "Toskin. In code B, the word Toskin stands "
                            "for Merrid. Code A turns Varnel into Toskin, "
                            "and applying code B to Toskin gives"),
        counterfactual_bridge="Zarnex", counterfactual_answer="Fenlow",
        counterfactual_accepted=[" Fenlow"],
        provenance={"source": "synthetic"})
    base.update(kw)
    return FactBundle(**base)


def test_clean_f_bundle_validates():
    assert validate_bundle(f_bundle()) == []


def test_f_answer_leak_flagged():
    b = f_bundle(prompts=dict(
        direct="The official currency of Testland, the quill, is",
        composed="The official currency of the country whose capital "
                 "city is Testville is",
        bridge_supplied="Testville is the capital of Testland. The "
                        "official currency of Testland is"))
    assert any("ANSWER leaks" in v for v in validate_bundle(b))


def test_f_bridge_in_composed_flagged():
    b = f_bundle(prompts=dict(
        direct="The official currency of Testland is",
        composed="The official currency of Testland, whose capital "
                 "city is Testville, is",
        bridge_supplied="Testville is the capital of Testland. The "
                        "official currency of Testland is"))
    assert any("BRIDGE leaks" in v for v in validate_bundle(b))


def test_clean_s_bundle_validates():
    assert validate_bundle(s_bundle()) == []


def test_s_answer_in_definitions_is_legal_but_query_leak_is_not():
    b = s_bundle()
    assert validate_bundle(b) == []          # answer in definitions: fine
    bad = s_bundle(prompts=dict(
        direct=b.prompts["direct"],
        composed="In code A, the word Varnel stands for Toskin. In code "
                 "B, the word Toskin stands for Merrid. Translating "
                 "Varnel gives Merrid, which written out is",
        bridge_supplied=b.prompts["bridge_supplied"]))
    assert any("ANSWER leaks into the query" in v
               for v in validate_bundle(bad))


def test_s_bridge_in_composed_query_flagged():
    b = s_bundle()
    bad = s_bundle(prompts=dict(
        direct=b.prompts["direct"],
        composed="In code A, the word Varnel stands for Toskin. In code "
                 "B, the word Toskin stands for Merrid. Translating "
                 "Varnel, that is Toskin under code A, through code B "
                 "gives",
        bridge_supplied=b.prompts["bridge_supplied"]))
    assert any("BRIDGE leaks into the primary query" in v
               for v in validate_bundle(bad))


def test_s_undefined_answer_flagged():
    b = s_bundle()
    bad = s_bundle(prompts=dict(
        direct=b.prompts["direct"],
        composed="In code A, the word Varnel stands for Toskin. "
                 "Translating Varnel through code A and then code B "
                 "gives",
        bridge_supplied=b.prompts["bridge_supplied"]))
    assert any("never defined in-context" in v for v in validate_bundle(bad))


def test_bank_level_dedup_and_phase2_collision():
    b1, b2 = f_bundle(), f_bundle(fact_id="capital_to_currency:other")
    rep = validate_bank([b1, b2])
    assert rep["duplicate_triples"]          # same (bridge, answer)
    rep2 = validate_bank([b1], phase2_triples={b1.triple_key})
    assert rep2["phase2_triple_collisions"] == [b1.fact_id]
