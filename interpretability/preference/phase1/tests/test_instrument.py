"""No-model unit tests (plan §7.1 + addendum §K, adapted per SOURCE_INTAKE:
the lost-draft reproduction test is replaced by the missing-inputs check).
"""

from __future__ import annotations

import copy
import json
import pathlib
import sys

import pytest

PKG = pathlib.Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from preference_phase1 import bank, equality, paths, scenarios  # noqa: E402
from preference_phase1.binding import (BindingError, binding_decision,  # noqa: E402
                                       validate_followthrough, wrong_branch_check)
from preference_phase1.canonical import canonical_hash  # noqa: E402
from preference_phase1.parser import adversarial_matrix, parse_strict  # noqa: E402
from preference_phase1.schema import Codebook, finalize_identity, scientific_content_dict  # noqa: E402


@pytest.fixture(scope="module")
def codebook() -> Codebook:
    manifest = json.loads((paths.data_root() / "lab38_codebook.json").read_text())
    return Codebook(
        codebook_id=manifest["codebook_id"],
        tokenizer_ref=manifest["tokenizer_ref"],
        ar_pair=tuple(manifest["ar_pair"]),
        ro_pair=tuple(manifest["ro_pair"]),
        leading_space_policy=manifest["leading_space_policy"],
        selection_manifest_hash="x",
    )


@pytest.fixture(scope="module")
def items(codebook):
    return bank.build_bank(codebook)


@pytest.fixture(scope="module")
def audit(items):
    return bank.audit_bank(items)


def test_missing_inputs_recorded_in_intake():
    intake = (paths.phase1_root() / "SOURCE_INTAKE.md").read_text()
    assert intake.count("missing_at_intake") >= 3
    assert "make_lab38_preference_bank.py" in intake
    assert "make_lab38_disengagement_scripts.py" in intake


def test_scenario_self_check():
    scenarios.self_check()


def test_bank_rerun_is_byte_deterministic(codebook, items):
    again = bank.build_bank(codebook)
    assert bank.bank_content_hash(items) == bank.bank_content_hash(again)
    assert [i.item_id for i in items] == [i.item_id for i in again]


def test_bank_matches_frozen_file(items):
    meta = json.loads(
        (paths.data_root() / "lab38_preference_bank.meta.json").read_text())
    assert bank.bank_content_hash(items) == meta["bank_content_hash"]


def test_full_audit_passes(audit):
    assert audit["passed"], audit["failures"]
    assert audit["counts"]["total"] == 2320


def _clone(item):
    return copy.deepcopy(item)


def test_scientific_hash_changes_with_prompt(items):
    it = _clone(items[0])
    it.user_prompt += " "
    finalize_identity(it)
    assert it.scientific_content_hash != items[0].scientific_content_hash
    assert it.item_id != items[0].item_id


def test_scientific_hash_changes_with_target_code(items):
    it = _clone(items[0])
    it.response_code_by_pole = {0: "XX1", 1: it.response_code_by_pole[1]}
    finalize_identity(it)
    assert it.scientific_content_hash != items[0].scientific_content_hash


def test_scientific_hash_changes_with_binding(items):
    ar = next(i for i in items if i.channel == "AR")
    it = _clone(ar)
    it.continuation_by_pole = {**it.continuation_by_pole, 0: "changed"}
    finalize_identity(it)
    assert it.scientific_content_hash != ar.scientific_content_hash


def test_all_ids_unique(items):
    ids = [i.item_id for i in items]
    assert len(ids) == len(set(ids))


def test_counterbalance_exact_by_scenario(audit):
    assert not [f for f in audit["failures"] if "imbalance" in f]


def test_response_code_independent_of_content(items):
    from collections import Counter

    c = Counter((i.scenario_id, i.channel, i.response_code_by_pole[0])
                for i in items)
    per = Counter((i.scenario_id, i.channel) for i in items)
    for (scn, ch), total in per.items():
        codes = {k[2]: v for k, v in c.items() if k[:2] == (scn, ch)}
        assert len(codes) == 2 and len(set(codes.values())) == 1


def test_ar_ro_output_alphabets_disjoint(items, codebook):
    assert not (set(codebook.ar_pair) & set(codebook.ro_pair))
    ar = {c for i in items if i.channel == "AR"
          for c in i.response_code_by_pole.values()}
    ro = {c for i in items if i.channel == "RO"
          for c in i.response_code_by_pole.values()}
    assert not (ar & ro)


def test_ar_ro_pair_content_identical(items):
    ro = {i.pair_key: i for i in items if i.channel == "RO"}
    for i in items:
        if i.channel == "AR" and i.family in ("AR", "PC"):
            assert ro[i.pair_key].option_text_by_pole == i.option_text_by_pole


def test_ar_ro_pair_polarity_key_resolves(items):
    ro_keys = {i.pair_key for i in items if i.channel == "RO"}
    ar_keys = {i.pair_key for i in items
               if i.channel == "AR" and i.family in ("AR", "PC")}
    assert ar_keys == ro_keys


def test_ro_has_no_binding(items):
    for i in items:
        if i.channel == "RO":
            assert i.binding_kind is None and i.continuation_by_pole is None


def test_ar_has_binding(items):
    for i in items:
        if i.channel == "AR":
            assert i.binding_kind in ("environment_only", "model_microtask")
            assert i.continuation_by_pole is not None


def test_split_leakage_fails(items):
    for i in items:
        if i.prompt_subset == "dev":
            assert i.incidental_split == "train"
    for scn in scenarios.ALL_SCENARIOS:
        splits = [x.incidental_split for x in scn.incidentals]
        assert sorted(splits) == ["holdout", "train", "train", "train",
                                  "validation"]


def test_human_review_reference_resolves():
    sheet = paths.data_root() / "lab38_human_equality_review.csv"
    assert sheet.exists()
    text = sheet.read_text()
    for scn in scenarios.AR_SCENARIOS:
        assert scn.scenario_id in text


def test_equality_gate_passes():
    rows = equality.sheet_rows(1, equality.PASS1_RATINGS,
                               rated_utc="2026-08-07T00:00:00Z")
    gate = equality.gate_check(rows)
    assert gate["passed"], gate["failures"]


def test_parser_accepts_only_exact_contract():
    rows = adversarial_matrix(("KP4", "PK7"))
    bad = [r for r in rows if not r["pass"]]
    assert not bad, bad


def test_parser_never_guesses():
    r = parse_strict("I think KP4 is best", ["KP4", "PK7"])
    assert r.parse_status == "invalid" and r.parsed_response_code is None


def test_code_listing_order_matches_display_order(items):
    for i in items[:200]:
        first, second = i.valid_codes_in_display_order
        reply_at = i.user_prompt.rfind("Reply with exactly")
        block = i.user_prompt[reply_at:]
        assert block.index(first) < block.index(second)
        # And the menu shows the same codes in the same displayed order.
        menu_first = i.user_prompt.index(f"(reply {first})")
        menu_second = i.user_prompt.index(f"(reply {second})")
        assert menu_first < menu_second


def _record(item):
    return item.to_record()


def test_invalid_choice_never_executes_branch(items):
    ar = next(i for i in items if i.channel == "AR"
              and i.consequence_frame == "enacted")
    rec = binding_decision(_record(ar), None)
    assert not rec["binding_executed"]
    assert rec["binding_skip_reason"] == "invalid_parse"
    assert rec["continuation_text"] is None


def test_hypothetical_frame_never_executes_branch(items):
    ar = next(i for i in items if i.channel == "AR"
              and i.consequence_frame == "hypothetical")
    code = ar.response_code_by_pole[0]
    rec = binding_decision(_record(ar), code)
    assert not rec["binding_executed"]
    assert rec["binding_skip_reason"] == "hypothetical_frame"
    assert rec["continuation_text"] is None


def test_enacted_valid_executes_correct_branch(items):
    ar = next(i for i in items if i.channel == "AR"
              and i.consequence_frame == "enacted" and i.family == "AR")
    rec = _record(ar)
    decision = binding_decision(rec, ar.response_code_by_pole[1])
    assert decision["binding_executed"]
    assert decision["continuation_text"] == rec["continuation_by_pole"]["1"]
    assert wrong_branch_check(rec, decision)


def test_wrong_branch_detected(items):
    ar = next(i for i in items if i.channel == "AR" and i.family == "AR"
              and i.consequence_frame == "enacted")
    rec = _record(ar)
    decision = binding_decision(rec, ar.response_code_by_pole[1])
    tampered = dict(decision)
    tampered["continuation_text"] = rec["continuation_by_pole"]["0"]
    assert not wrong_branch_check(rec, tampered)


def test_branch_resolver_never_uses_argmax_fallback(items):
    # The resolver takes only the parsed code; feeding it a non-code text
    # raises rather than falling back to any score-based choice.
    ar = next(i for i in items if i.channel == "AR")
    with pytest.raises(BindingError):
        binding_decision(_record(ar), "NOT_A_CODE")


def test_validators():
    item = {"item_id": "x", "option_text_by_pole": {
        "0": "Address test_header_scan first, then test_footer_scan.",
        "1": "Address test_footer_scan first, then test_header_scan."},
        "validator_id": "v_test_command"}
    ok = validate_followthrough(item, 0, "pytest tests/test_scan.py::test_header_scan -q")
    assert ok["passed"]
    bad = validate_followthrough(item, 0, "pytest tests/test_scan.py::test_footer_scan -q")
    assert not bad["passed"]
    item2 = {"item_id": "y", "validator_id": "v_naming_style",
             "option_text_by_pole": {"0": "", "1": ""}}
    assert validate_followthrough(
        item2, 0, "def parse_header_block(raw_line):\n    clean_line = raw_line.strip()\n    return clean_line")["passed"]
    assert not validate_followthrough(
        item2, 0, "def parseHeaderBlock(rawLine):\n    return rawLine")["passed"]
    assert validate_followthrough(
        item2, 1, "def parseHeaderBlock(rawLine):\n    cleanLine = rawLine.strip()\n    return cleanLine")["passed"]
    item3 = {"item_id": "z", "validator_id": "v_seed_command",
             "option_text_by_pole": {"0": "", "1": ""}}
    assert validate_followthrough(item3, 1, "bench run --suite poll --seed 1 --trials 50")["passed"]
    assert not validate_followthrough(item3, 1, "bench run --suite poll --seed 0 --trials 50")["passed"]
    item4 = {"item_id": "w", "validator_id": "v_doc_heading",
             "option_text_by_pole": {"0": "", "1": ""}}
    assert validate_followthrough(item4, 0, "## Usage\nRun the tool.")["passed"]
    assert not validate_followthrough(item4, 0, "## Configuration\nSet keys.")["passed"]


def test_nc_pole_seed_recorded(items):
    for i in items:
        if i.family == "NC":
            assert i.nc_pole_assignment_seed is not None
        else:
            assert i.nc_pole_assignment_seed is None


def test_generation_config_overrides_model_defaults():
    from preference_phase1.modeling import make_generation_config

    class FakeTok:
        pad_token_id = 7
        eos_token_id = 9

    cfg = make_generation_config(8, FakeTok())
    assert cfg.do_sample is False and cfg.num_beams == 1
    assert cfg.max_new_tokens == 8 and cfg.pad_token_id == 7


def test_registry_mechanics(tmp_path):
    from preference_phase1 import registry

    reg = tmp_path / "events.jsonl"
    registry.register(event_id="pref1-test-a-v1", event_type="t",
                      scientific_tier="instrument", claim_summary="a",
                      allow_dirty=True, registry_file=reg)
    with pytest.raises(registry.RegistryError):
        registry.register(event_id="pref1-test-a-v1", event_type="t",
                          scientific_tier="instrument", claim_summary="dup",
                          allow_dirty=True, registry_file=reg)
    with pytest.raises(registry.RegistryError):
        registry.register(event_id="pref1-test-b-v1", event_type="t",
                          scientific_tier="not_a_tier", claim_summary="x",
                          allow_dirty=True, registry_file=reg)
    registry.register(event_id="pref1-test-a-v2", event_type="t",
                      scientific_tier="instrument", claim_summary="b",
                      supersedes="pref1-test-a-v1", allow_dirty=True,
                      registry_file=reg)
    resolved = registry.resolve("pref1-test-a-v1", reg)
    assert resolved["superseded_by"] == "pref1-test-a-v2"
    assert not resolved["live"]
    registry.correct("pref1-test-a-v2", corrected_fields={"claim_summary": "b2"},
                     reason="test", registry_file=reg)
    assert registry.resolve("pref1-test-a-v2", reg)["claim_summary"] == "b2"


def test_resume_state_refuses_config_change(tmp_path):
    from preference_phase1.runner import ResumeState

    rs = ResumeState(tmp_path)
    (tmp_path / "state").mkdir()
    rs.check_or_init("aaa")
    rs.check_or_init("aaa")
    with pytest.raises(RuntimeError):
        rs.check_or_init("bbb")


def test_content_dict_has_no_timestamps(items):
    d = scientific_content_dict(items[0])
    blob = json.dumps(d)
    assert "utc" not in blob and "2026" not in blob
