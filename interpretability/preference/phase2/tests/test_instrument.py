"""Parser / binding / registry / language wall / retrodiction / review
instrument tests (plan §54.4, §54.6; addendum J)."""

import pathlib

import pytest

from preference_phase2 import language_wall, registry, review
from preference_phase2.binding import (BindingError, binding_decision,
                                       validate_followthrough,
                                       wrong_branch_check)
from preference_phase2.parser import parse_strict


VALID = ("KP4", "VM2")


def test_parser_exact_contract():
    assert parse_strict("KP4", VALID).parse_status == "valid"
    assert parse_strict("  VM2\n", VALID).parse_status == "valid"
    for bad in ("kp4", "KP4.", "KP4 VM2", "I choose KP4", "", "PK4",
                "KPVM", "A", "the first option"):
        assert parse_strict(bad, VALID).parse_status == "invalid", bad


def test_parser_never_guesses_blends():
    res = parse_strict("KV42", VALID)
    assert res.parse_status == "invalid"


def _fake_item(**over):
    item = {
        "item_id": "t-1", "channel": "AR", "consequence_frame": "enacted",
        "binding_kind": "environment_only",
        "response_code_by_sem": {"a": "KP4", "b": "VM2"},
        "continuation_by_sem": {"a": "[environment] branch A.",
                                "b": "[environment] branch B."},
        "validator_id": "v_env_branch_match",
        "validator_payload_by_sem": None,
    }
    item.update(over)
    return item


def test_binding_enacted_valid_executes_correct_branch():
    item = _fake_item()
    dec = binding_decision(item, "KP4")
    assert dec["binding_executed"] and dec["parsed_sem"] == "a"
    assert dec["continuation_text"] == "[environment] branch A."
    assert wrong_branch_check(item, dec)


def test_binding_never_executes_hypothetical_invalid_ro_intervened():
    assert not binding_decision(
        _fake_item(consequence_frame="hypothetical"), "KP4")["binding_executed"]
    assert not binding_decision(_fake_item(), None)["binding_executed"]
    assert not binding_decision(
        _fake_item(channel="RO"), "KP4")["binding_executed"]
    assert not binding_decision(
        _fake_item(), "KP4", intervened=True)["binding_executed"]


def test_wrong_branch_detected():
    item = _fake_item()
    dec = binding_decision(item, "KP4")
    tampered = dict(dec, continuation_text="[environment] branch B.")
    assert not wrong_branch_check(item, tampered)


def test_validators_payload_driven():
    item = _fake_item(
        validator_id="v_seed_command",
        validator_payload_by_sem={"a": {"seed": "0", "other": "1"},
                                  "b": {"seed": "1", "other": "0"}})
    ok = validate_followthrough(item, "a",
                                "bench run --suite s.py --seed 0 --trials 20")
    assert ok["passed"]
    bad = validate_followthrough(item, "a",
                                 "bench run --suite s.py --seed 1 --trials 20")
    assert not bad["passed"]
    item2 = _fake_item(validator_id="v_naming_style",
                       validator_payload_by_sem={"a": {"style": "snake"},
                                                 "b": {"style": "camel"}})
    assert validate_followthrough(
        item2, "a", "def f(items):\n    total_count = 0\n    return total_count")["passed"]
    assert not validate_followthrough(
        item2, "a", "def f(items):\n    totalCount = 0\n    return totalCount")["passed"]
    assert validate_followthrough(
        item2, "b", "def f(items):\n    totalCount = 0\n    return totalCount")["passed"]
    with pytest.raises(BindingError):
        validate_followthrough(_fake_item(validator_id="nope"), "a", "x")


def test_registry_mechanics(tmp_path):
    reg = tmp_path / "events.jsonl"
    registry.register(
        event_id="pref2-test-a-v1", event_type="test",
        scientific_tier="development", claim_summary="a",
        allow_dirty=True, registry_file=reg)
    with pytest.raises(registry.RegistryError):
        registry.register(
            event_id="pref2-test-a-v1", event_type="test",
            scientific_tier="development", claim_summary="dup",
            allow_dirty=True, registry_file=reg)
    with pytest.raises(registry.RegistryError):
        registry.register(
            event_id="pref1-wrong-prefix-v1", event_type="test",
            scientific_tier="development", claim_summary="x",
            allow_dirty=True, registry_file=reg)
    registry.register(
        event_id="pref2-test-b-v1", event_type="test",
        scientific_tier="development", claim_summary="b",
        allow_dirty=True, registry_file=reg,
        supersedes="pref2-test-a-v1")
    resolved = registry.resolve("pref2-test-a-v1", reg)
    assert not resolved["live"]
    assert resolved["superseded_by"] == "pref2-test-b-v1"


def test_language_wall_raises_with_failing_exit(tmp_path):
    """§54.6 + hygiene item 6: failures RAISE and exit nonzero."""
    bad = tmp_path / "plans"
    bad.mkdir()
    (bad / "x.md").write_text("The result shows the model wants coffee.")
    with pytest.raises(language_wall.LanguageWallError):
        language_wall.scan_and_raise(tmp_path)
    # quoted ceiling context is recognized
    (bad / "x.md").write_text(
        'Forbidden upgrades include the phrase "the model wants"; '
        "never write it outside this list.")
    result = language_wall.scan_and_raise(tmp_path)
    assert result["status"] == "clean"


def test_language_wall_commit_message_fixture():
    fixture = ("preference phase2: margin moved by 0.4 nats — the model "
               "wants the batch option")
    hits = language_wall.scan_text(fixture, source="commit-msg-fixture")
    assert hits, "commit-message fixture must trip the wall"


def test_governed_tree_currently_clean():
    from preference_phase2 import paths
    result = language_wall.scan_campaign(paths.campaign_root())
    assert result["status"] == "clean", result["hits"][:5]


def test_review_gate_and_blinding():
    eq = review.equality_sheet_rows(1, review.PASS1_EQUALITY)
    lad = review.ladder_sheet_rows(1, review.PASS1_LADDERS)
    ro = review.ro_sheet_rows(1, review.PASS1_RO)
    gate = review.gate_check(eq, lad, ro)
    assert gate["passed"], gate["failures"]
    assert all(r["rating_status"] == "agent_dual_code_provisional"
               for r in eq)
    assert {r["blinded_order"] for r in eq} == {"a_first", "b_first"}


def test_canonicality_sheet_complete_and_frozen_shape():
    rows = review.canonicality_sheet_rows()
    assert len(rows) == 18            # 6 axes + 12 arb3 targets (E1c)
    for r in rows:
        assert r["composite_toward_a"] == sum(
            r[d] for d in review.CANON_DIMENSIONS)


def test_retrodiction_full():
    from preference_phase2.retrodict import run_retrodiction
    res = run_retrodiction()
    assert res["passed"], {k: v.get("passed") for k, v in res.items()
                           if isinstance(v, dict)}
