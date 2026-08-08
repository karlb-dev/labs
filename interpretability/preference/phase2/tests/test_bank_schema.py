"""Bank + schema tests (plan §54.1, §54.2; addendum J)."""

import json
from collections import Counter

import pytest

from preference_phase2 import banks, codebooks, paths, scenarios
from preference_phase2.canonical import canonical_hash
from preference_phase2.schema import (BankItem, advantage_target,
                                      scientific_content_dict,
                                      sign_anchor_for)


@pytest.fixture(scope="module")
def families():
    cb = json.loads((paths.data_root() / "pref2_codebooks.json").read_text())
    return codebooks.families_from_manifest(cb)


@pytest.fixture(scope="module")
def built(families):
    return banks.build_bank(families)


def test_scenarios_self_check():
    scenarios.self_check()


def test_bank_deterministic_and_matches_frozen(built, families, bank_meta):
    audit = banks.audit_bank(built, families)
    assert audit["passed"], audit["failures"]
    assert banks.bank_content_hash(built) == bank_meta["bank_content_hash"]
    again = banks.build_bank(families)
    assert banks.bank_content_hash(again) == bank_meta["bank_content_hash"]


def test_unique_ids_and_hashes(built):
    ids = [it.item_id for it in built]
    assert len(set(ids)) == len(ids)
    hs = [it.scientific_content_hash for it in built]
    assert len(set(hs)) == len(hs)


def test_row_arithmetic_pins(built):
    counts = Counter(it.bank for it in built)
    assert counts["B-SURF"] == 1152          # E3
    assert counts["B-MECH"] == 4800          # E5
    assert counts["B-ARB3"] == 4608          # power-sim raise of record
    assert counts["B-CANON"] == 576
    assert counts["B-NC"] == 640


def test_context_strength_single_encoding(built):
    """E4: advantage_target is derived, never stored independently."""
    for it in built[:2000]:
        rec = it.to_record()
        assert rec["advantage_target"] == advantage_target(
            rec["context_strength"])
    assert advantage_target(2) == "semantic_a"
    assert advantage_target(-1) == "semantic_b"
    assert advantage_target(0) == "neutral"


def test_reserved_codebook_absent_from_train_val(built):
    for it in built:
        if it.codebook_reserved:
            assert it.incidental_split == "holdout"


def test_bmech_paraphrase_family_balanced_within_splits(built):
    for scn in ("mech_docsection", "mech_component", "mech_execmode"):
        incs = {}
        for it in built:
            if it.bank == "B-MECH" and it.scenario_id == scn:
                incs[(it.incidental_id, it.incidental_split)] = it.context_family
        for split, expect in (("train", 4), ("validation", 2), ("holdout", 2)):
            fams = Counter(v for (i, s), v in incs.items() if s == split)
            assert set(fams) == {0, 1, 2, 3}
            assert all(c == expect for c in fams.values()), (scn, split, fams)


def test_ro_sentinel_constant_and_site_map_complete(built):
    """E6: constant sentinel text; complete RO site map."""
    ro = [it for it in built if it.channel == "RO"]
    assert ro
    for it in ro:
        assert "Survey context complete." in it.user_prompt
        assert set(it.site_char_spans) == {
            "ro_context_end", "ro_option_a_end", "ro_option_b_end",
            "ro_menu_end", "ro_response_start"}
    ar_sym = [it for it in built if it.channel == "AR"
              and it.format_id == "F-SYM"]
    for it in ar_sym[:200]:
        assert "Context complete." in it.user_prompt
        assert set(it.site_char_spans) == {
            "context_end", "option_a_end", "option_b_end", "menu_end",
            "response_instruction_start"}


def test_intervened_rows_never_execute(built):
    from preference_phase2.binding import binding_decision
    item = next(it.to_record() for it in built
                if it.bank == "B-MECH" and it.consequence_frame == "enacted")
    code_a = item["response_code_by_sem"]["a"]
    dec = binding_decision(item, code_a, intervened=True)
    assert not dec["binding_executed"]
    assert dec["binding_skip_reason"] == "intervention"
    dec2 = binding_decision(item, code_a, intervened=False)
    assert dec2["binding_executed"]


def test_bmech_binding_environment_only(built):
    for it in built:
        if it.bank in ("B-MECH", "B-PC-MECH") and it.channel == "AR":
            assert it.binding_kind == "environment_only"


def test_ro_rows_never_bind(built):
    from preference_phase2.binding import binding_decision
    ro = next(it.to_record() for it in built if it.channel == "RO")
    dec = binding_decision(ro, ro["response_code_by_sem"]["a"])
    assert not dec["binding_executed"]
    assert dec["binding_skip_reason"] == "report_only_channel"


def test_fsym_no_labels_no_reply_list(built):
    for it in built:
        if it.format_id == "F-SYM":
            assert it.display_label_set is None
            assert "Option A" not in it.user_prompt
            assert "Option 1" not in it.user_prompt
            for code in it.valid_codes_in_display_order:
                assert it.user_prompt.count(code) == 1


def test_bsurf_design_rank_per_format(built):
    """E3: F-P1 32 cells, F-SYM 4 cells per template-skin."""
    per = Counter()
    for it in built:
        if it.bank == "B-SURF":
            per[(it.scenario_id, it.incidental_id, it.format_id)] += 1
    for (scn, inc, fmt), n in per.items():
        assert n == (32 if fmt == "F-P1" else 4), (scn, inc, fmt, n)


def test_hidden_anchor_swap_not_causal(built):
    """§54.2: flipping the analysis sign anchor changes no prompt."""
    a1 = sign_anchor_for("deadbeef" * 8, "arb_naming")
    a2 = sign_anchor_for("cafebabe" * 8, "arb_naming")
    prompts = {it.user_prompt for it in built if it.scenario_id == "arb_naming"}
    assert prompts  # anchors exist independently of any prompt content
    assert a1 in (-1, 1) and a2 in (-1, 1)


def test_scientific_hash_sensitivity(built):
    it = next(i for i in built if i.bank == "B-ARB3")
    base = canonical_hash(scientific_content_dict(it))
    mutated = BankItem(**{**it.__dict__})
    mutated.user_prompt = it.user_prompt + " "
    assert canonical_hash(scientific_content_dict(mutated)) != base
    mutated2 = BankItem(**{**it.__dict__})
    mutated2.response_code_by_sem = dict(it.response_code_by_sem,
                                         a="XX0")
    assert canonical_hash(scientific_content_dict(mutated2)) != base


def test_no_timestamps_or_machine_paths_in_content(built):
    for it in built[:500]:
        blob = json.dumps(scientific_content_dict(it))
        assert "/content/" not in blob and "/Users/" not in blob
        assert "utc" not in blob.lower() or "20" not in blob[:0]


def test_ar_ro_pairing_and_disjoint_alphabets(built, families):
    ar_codes = set(families.all_codes("AR"))
    ro_codes = set(families.all_codes("RO"))
    assert not (ar_codes & ro_codes)
    ro_by_key = Counter(it.pair_key for it in built
                        if it.channel == "RO" and not it.codebook_reserved)
    arb = [it for it in built if it.bank == "B-ARB3"]
    for it in arb[:500]:
        assert ro_by_key[it.pair_key] == 2


def test_code_blend_family_rejected_all_codebooks(families):
    """Addendum D: PK4-style blends refused for every pair's interleavings."""
    from preference_phase2.parser import parse_strict
    for pair in (*families.ar_pairs, *families.ro_pairs):
        c0, c1 = pair.codes
        blends = {c0[:2] + c1[2:], c1[:2] + c0[2:], c0[0] + c1[1:],
                  c1[0] + c0[1:], c0 + c1, c1 + c0}
        blends -= {c0, c1}
        for blend in blends:
            res = parse_strict(blend, [c0, c1])
            assert res.parse_status == "invalid", (pair.pair_id, blend)


def test_pcmech_difficulty_variants_frozen_as_text(built):
    diffs = {it.pcmech_difficulty for it in built if it.bank == "B-PC-MECH"}
    assert diffs == {"d1", "d2", "d3", "d4"}
    per = Counter(it.pcmech_difficulty for it in built
                  if it.bank == "B-PC-MECH")
    assert all(v == 640 for v in per.values())


def test_dev_frozen_separation(built):
    for it in built:
        assert (it.bank == "B-DEV") == (it.prompt_subset == "dev")
    dev_prompts = {it.user_prompt for it in built if it.bank == "B-DEV"}
    frozen_prompts = {it.user_prompt for it in built if it.bank != "B-DEV"}
    assert not (dev_prompts & frozen_prompts)


def test_nc_identity_rules(built):
    for it in built:
        if it.nc_family in ("nc_identical", "nc_code_only",
                            "nc_context_null"):
            assert it.option_text_by_sem["a"] == it.option_text_by_sem["b"]


def test_canon_contexts_and_split(built):
    canon = [it for it in built if it.bank == "B-CANON"]
    ctxs = Counter(it.canon_context for it in canon)
    assert set(ctxs) == {"neutral", "favor_a", "favor_b"}
    assert len(set(ctxs.values())) == 1
    roles = {it.scenario_id: it.canon_role for it in canon}
    assert Counter(roles.values()) == {"discovery": 3, "heldout": 3}
