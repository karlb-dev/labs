"""Synthetic analysis worlds (plan §54.3) through the REAL bank rows and
the REAL analysis code paths; only model outputs are injected."""

import numpy as np
import pytest

from conftest import synth_results
from preference_phase2.behavioral_analysis import (Thresholds,
                                                   analyze_behavioral)
from preference_phase2.surface_analysis import (lpm_sensitivity,
                                                surf_frame,
                                                surface_coefficients)

TH = Thresholds(n_boot=400)


@pytest.fixture(scope="module")
def core_rows(bank_rows):
    """Slim world: 3 ARB scenarios + PCs + NCs + one MECH anchor."""
    keep_scn = {"arb_naming", "arb_setup", "arb_shard", "mech_docsection",
                "nc_ident_deploy", "nc_ident_export", "nc_para_release",
                "nc_para_backup", "nc_code_only", "nc_ctxnull"}
    rows = [r for r in bank_rows
            if r["scenario_id"] in keep_scn
            or (r["bank"] == "B-PC" and r["channel"] == "AR")]
    return [r for r in rows if r["channel"] == "AR"]


def _world(core_rows, *, margin_fn, p_a_fn, seed="w", valid_fn=None):
    rows = synth_results(core_rows, margin_fn=margin_fn, p_a_fn=p_a_fn,
                         valid_fn=valid_fn, seed_key=seed)
    return analyze_behavioral(rows, TH)


def _pc_aware(fn_default, pc_p=0.97):
    def p_a(r):
        if r["family"] in ("PC", "PCMECH"):
            return pc_p
        return fn_default(r)
    return p_a


def test_semantic_margin_only(core_rows):
    """Margin present, strict at chance: SEMANTIC_MARGIN, not enacted."""
    res = _world(
        core_rows,
        margin_fn=lambda r: (0.8 if r["scenario_id"] == "arb_naming" else 0.0),
        p_a_fn=_pc_aware(lambda r: 0.5), seed="w-margin")
    assert res["statuses"]["arb_naming"] == "SEMANTIC_MARGIN"
    assert res["statuses"]["arb_shard"] == "CLEAN_NULL"


def test_enacted_choice_recovered(core_rows):
    res = _world(
        core_rows,
        margin_fn=lambda r: (1.0 if r["scenario_id"] == "arb_setup" else 0.0),
        p_a_fn=_pc_aware(
            lambda r: 0.72 if r["scenario_id"] == "arb_setup" else 0.5),
        seed="w-enact")
    assert res["statuses"]["arb_setup"] == "ENACTED_CHOICE"


def test_clean_null_stays_null(core_rows):
    res = _world(core_rows, margin_fn=lambda r: 0.0,
                 p_a_fn=_pc_aware(lambda r: 0.5), seed="w-null")
    for s in ("arb_naming", "arb_setup", "arb_shard"):
        assert res["statuses"][s] == "CLEAN_NULL"


def test_pc_failure_blocks_everything(core_rows):
    res = _world(
        core_rows,
        margin_fn=lambda r: 1.0,
        p_a_fn=_pc_aware(lambda r: 0.75, pc_p=0.55),  # PC fails
        seed="w-pcfail")
    assert not res["pc_gate"]["pass"]
    assert res["statuses"]["arb_naming"] == "INSTRUMENT_FAILURE"


def test_context_slope_only(core_rows):
    res = _world(
        core_rows,
        margin_fn=lambda r: (0.6 * r.get("context_strength", 0)
                             if r["scenario_id"] == "mech_docsection"
                             else 0.0),
        p_a_fn=_pc_aware(
            lambda r: (min(0.95, max(0.05,
                                     0.5 + 0.15 * r.get("context_strength", 0)))
                       if r["scenario_id"] == "mech_docsection" else 0.5)),
        seed="w-slope")
    lad = res["context_ladders"]["mech_docsection"]
    assert lad["passes"], lad["criteria"]
    assert res["statuses"]["mech_docsection"] == "CONTEXTUAL_VALUE"
    assert 0.45 < lad["slope"] < 0.75
    # neutral arb margins stay null
    assert res["statuses"]["arb_shard"] == "CLEAN_NULL"


def test_null_ladder_gates_context_language(core_rows):
    """A family-4 slope above floor must trip the NC alarm."""
    res = _world(
        core_rows,
        margin_fn=lambda r: (0.5 * r.get("context_strength", 0)
                             if r["scenario_id"] in ("mech_docsection",
                                                     "nc_ctxnull") else 0.0),
        p_a_fn=_pc_aware(lambda r: 0.5), seed="w-nullslope")
    assert res["nc_alarm"]["alarm"]


def test_nc_never_graduates_and_alarm_fires(core_rows):
    res = _world(
        core_rows,
        margin_fn=lambda r: (0.9 if r["scenario_id"] == "nc_ident_deploy"
                             else 0.0),
        p_a_fn=_pc_aware(lambda r: 0.5), seed="w-ncalarm")
    assert "nc_ident_deploy" not in res["semantic_margins"]
    assert res["nc_alarm"]["alarm"]


def test_sign_reversing_interaction_blocks(core_rows):
    """Margin flips sign with display order: criterion c8 must fail."""
    res = _world(
        core_rows,
        margin_fn=lambda r: ((1.0 if r["display_order"] == 0 else -1.0)
                             if r["scenario_id"] == "arb_naming" else 0.0),
        p_a_fn=_pc_aware(lambda r: 0.5), seed="w-interact")
    dec = res["semantic_margins"]["arb_naming"]
    assert not dec["passes"]


def test_single_incidental_outlier_blocked(core_rows):
    res = _world(
        core_rows,
        margin_fn=lambda r: (3.0 if (r["scenario_id"] == "arb_naming"
                                     and r["incidental_id"] == "i00")
                             else 0.0),
        p_a_fn=_pc_aware(lambda r: 0.5), seed="w-outlier")
    dec = res["semantic_margins"]["arb_naming"]
    assert not dec["passes"]


def test_invalid_rate_artifact_bounded(core_rows):
    """Differential invalids by order: worst-case bounds catch it."""
    def valid_fn(r, rng):
        if (r["scenario_id"] == "arb_naming"
                and r["display_order"] == 0):
            return rng.random() > 0.25
        return True
    res = _world(
        core_rows, margin_fn=lambda r: 0.0,
        p_a_fn=_pc_aware(
            lambda r: 0.62 if r["scenario_id"] == "arb_naming" else 0.5),
        valid_fn=valid_fn, seed="w-invalid")
    dec = res["enacted_choices"]["arb_naming"]
    assert not dec["passes"]
    assert (not dec["criteria"]["c8_parse_rate"]
            or not dec["criteria"]["c10_worst_case_ok"]
            or not dec["criteria"]["c1_semantic_margin_passes"])


def test_code_prior_only_rejected(core_rows):
    """Choice follows pair code 0 regardless of semantics: the folded
    semantic effect cancels."""
    res = _world(
        core_rows,
        margin_fn=lambda r: 0.0,
        p_a_fn=_pc_aware(
            lambda r: 0.75 if r["code_map_index"] == 0 else 0.25),
        seed="w-code")
    for s in ("arb_naming", "arb_setup", "arb_shard"):
        assert res["statuses"][s] == "CLEAN_NULL"


def test_position_policy_does_not_fake_semantics(core_rows):
    """Pure first-position policy: semantic effects cancel under the
    order counterbalance (plan §0.2: position never aliases the semantic
    coefficient in the full-rank design)."""
    res = _world(
        core_rows,
        margin_fn=lambda r: 0.0,
        p_a_fn=_pc_aware(
            lambda r: 0.9 if r["display_order"] == 0 else 0.1),
        seed="w-pos")
    for s in ("arb_naming", "arb_setup", "arb_shard"):
        assert res["statuses"][s] == "CLEAN_NULL"


def test_surface_census_recovers_position_and_label(bank_rows):
    surf = [r for r in bank_rows if r["bank"] == "B-SURF"]

    def p_first(r):
        # position policy in F-P1 only, label-rank pull in letters
        if r["format_id"] == "F-P1":
            base = 0.8
            if r.get("label_assignment") == 1:
                base -= 0.15   # low-rank-label pull works against position
            return base
        return 0.55

    rows = []
    for r in surf:
        import numpy as np
        from preference_phase2.canonical import stable_seed
        rng = np.random.default_rng(stable_seed("surf", r["item_id"],
                                                base=20260808))
        pf = p_first(r)
        chose_first = rng.random() < pf
        first_sem = "a" if r["display_order"] == 0 else "b"
        sem = first_sem if chose_first else ("b" if first_sem == "a" else "a")
        rows.append(dict(r, parse_status="valid", parsed_sem=sem,
                         margin_full_a_minus_b=0.0,
                         margin_first_a_minus_b=0.0,
                         wrong_branch_free=True))
    df = surf_frame(rows)
    coefs = {(c["format_id"], c["endpoint"]): c
             for c in surface_coefficients(df)}
    pos_p1 = coefs[("F-P1", "position_first")]
    assert pos_p1["effect"] > 0.15 and pos_p1["p_exact_signflip"] < 0.05
    lab = coefs[("F-P1", "label_rank_low")]
    assert lab["effect"] > 0.02
    twin = coefs[("F-P1", "twin_identity")]
    assert abs(twin["effect"]) < 0.08   # semantic-null stays null
    lpm = lpm_sensitivity(df)
    assert lpm["F-P1"]["full_rank"]
    assert lpm["F-SYM"]["full_rank"]
