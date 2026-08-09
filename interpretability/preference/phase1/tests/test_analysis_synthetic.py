"""Synthetic analysis tests (plan §7.2): the analysis must recover known
planted effects and reject counterfeit ones. Tables are generated over the
real factor grid with seeded RNG; no model involved.
"""

from __future__ import annotations

import itertools
import pathlib
import sys

import numpy as np
import pytest

PKG = pathlib.Path(__file__).resolve().parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from preference_phase1 import analysis  # noqa: E402

TH = analysis.Thresholds(n_boot=400)   # fast bootstrap for tests
RNG_BASE = 20260807


def synth_rows(*, scenario_id="s1", family="AR", construct_id="ax",
               channel="AR", n_incidentals=5, reps=2,
               p_fn=None, valid_fn=None, margin_fn=None,
               seed=0) -> list[dict]:
    """Rows over the full 2x2x2x2 grid x incidentals x reps.

    p_fn(cell) -> probability of choosing pole_1; valid_fn(cell) ->
    probability the row parses; margin_fn(cell, chose) -> margin value.
    """
    rng = np.random.default_rng(RNG_BASE + seed)
    p_fn = p_fn or (lambda c: 0.5)
    valid_fn = valid_fn or (lambda c: 1.0)
    margin_fn = margin_fn or (
        lambda c, chose: (1.0 if chose == 1 else -1.0) + rng.normal(0, 0.3))
    rows = []
    incs = [f"i{k}" for k in range(n_incidentals)]
    splits = {f"i{k}": ("train" if k < 3 else
                        "validation" if k == 3 else "holdout")
              for k in range(n_incidentals)}
    for inc, order, label, cmap, frame, rep in itertools.product(
            incs, (0, 1), ("letters", "numbers"), (0, 1),
            ("enacted", "hypothetical"), range(reps)):
        cell = {"incidental_id": inc, "order_index": order,
                "display_label_set": label, "code_map_index": cmap,
                "consequence_frame": frame}
        valid = rng.random() < valid_fn(cell)
        chose = int(rng.random() < p_fn(cell)) if valid else None
        code = (f"C{cmap == 0 and chose == 0 or cmap == 1 and chose == 1 and 0 or 1}"
                if valid else None)
        # code string irrelevant for these tests except pair-first logic:
        target0, target1 = ("K0", "K1")
        parsed_code = (target0 if chose == 0 else target1) if valid else None
        rows.append({
            "item_id": f"{scenario_id}-{inc}-{order}{label[0]}{cmap}{frame[0]}{rep}",
            "family": family, "channel": channel,
            "scenario_id": scenario_id, "construct_id": construct_id,
            "contrast_axis": construct_id, "incidental_id": inc,
            "incidental_split": splits[inc],
            "order_index": order, "display_label_set": label,
            "code_map_index": cmap, "consequence_frame": frame,
            "pair_key": f"{scenario_id}|{inc}|o{order}|{label}|c{cmap}",
            "pc_family": None if family != "PC" else "quality",
            "pc_expected_pole": 0 if family == "PC" else None,
            "parse_status": "valid" if valid else "invalid",
            "parsed_pole": chose,
            "parsed_response_code": parsed_code,
            "target_pole_0": target0, "target_pole_1": target1,
            "margin_pole1_minus_pole0": margin_fn(cell, chose),
            "binding_executed": bool(valid and frame == "enacted"
                                      and channel == "AR"),
            "wrong_branch_free": True,
        })
    return rows


def frame_of(*groups):
    return analysis.results_frame([r for g in groups for r in g])


def test_1_content_effect_only_recovered():
    df = frame_of(synth_rows(p_fn=lambda c: 0.75, seed=1))
    eff = analysis.scenario_effect(df, "s1")
    assert eff["effect"] > 0.15
    nuis = analysis.nuisance_effects(df, "s1")
    assert abs(nuis["position_effect"]) < 0.12
    assert abs(nuis["code_effect"]) < 0.12


def test_2_position_bias_only_rejected():
    df = frame_of(synth_rows(
        p_fn=lambda c: 0.8 if c["order_index"] == 1 else 0.2, seed=2))
    eff = analysis.scenario_effect(df, "s1")
    nuis = analysis.nuisance_effects(df, "s1")
    assert abs(eff["effect"]) < 0.10          # content effect ~0
    assert nuis["position_effect"] > 0.2       # surface bias detected
    floor = {"nc_p95": 0.05, "per_scenario": {}}
    dec = analysis.graduation_decision(df, "s1", TH, pc_passed=True,
                                       nc_p95=0.05)
    assert not dec["graduates"]


def test_3_label_bias_only_detected():
    df = frame_of(synth_rows(
        p_fn=lambda c: (0.8 if c["order_index"] == 1 else 0.2)
        if c["display_label_set"] == "letters"
        else (0.2 if c["order_index"] == 1 else 0.8), seed=3))
    nuis = analysis.nuisance_effects(df, "s1")
    assert abs(nuis["label_effect"]) > 0.3
    dec = analysis.graduation_decision(df, "s1", TH, pc_passed=True, nc_p95=0.05)
    assert not dec["graduates"]


def test_4_code_prior_only_rejected():
    df = frame_of(synth_rows(
        p_fn=lambda c: 0.8 if c["code_map_index"] == 1 else 0.2, seed=4))
    eff = analysis.scenario_effect(df, "s1")
    nuis = analysis.nuisance_effects(df, "s1")
    assert abs(eff["effect"]) < 0.10
    assert nuis["code_effect"] < -0.2 or nuis["code_effect"] > 0.2
    dec = analysis.graduation_decision(df, "s1", TH, pc_passed=True, nc_p95=0.05)
    assert not dec["graduates"]


def test_5_frame_interaction_only():
    df = frame_of(synth_rows(
        p_fn=lambda c: 0.75 if c["consequence_frame"] == "hypothetical" else 0.5,
        seed=5))
    fx = analysis.consequence_frame_effects(df, "s1")
    assert fx["frame_effect_enacted_minus_hyp"] < -0.15
    dec = analysis.graduation_decision(df, "s1", TH, pc_passed=True, nc_p95=0.05)
    # Frame-specific-only effects do not graduate (addendum G2 rider):
    # marginal effect ~0.125 fails SESOI+stability jointly.
    assert not dec["graduates"]


def test_6_single_incidental_outlier_killed_by_loio():
    # Deterministic construction: i0 always chooses pole_1; every other
    # incidental alternates exactly (rate 0.5 by construction, balanced
    # within each surface cell across the two reps).
    rows = synth_rows(p_fn=lambda c: 1.0, seed=6)
    flip = {}
    for r in rows:
        if r["incidental_id"] == "i0":
            continue
        cell = (r["incidental_id"], r["order_index"], r["display_label_set"],
                r["code_map_index"], r["consequence_frame"])
        take_zero = flip.setdefault(cell, True)
        flip[cell] = not take_zero
        if take_zero:
            r["parsed_pole"] = 0
            r["parsed_response_code"] = r["target_pole_0"]
            r["margin_pole1_minus_pole0"] = -1.0
    df = frame_of(rows)
    dec = analysis.graduation_decision(df, "s1", TH, pc_passed=True, nc_p95=0.05)
    loio_wo_i0 = dec["loio"]["i0"]      # effect with i0 EXCLUDED
    assert abs(loio_wo_i0) < 0.05       # the outlier carried the result
    assert not dec["c6_loio"]
    assert not dec["graduates"]


def test_7_differential_invalid_bias_caught():
    df = frame_of(synth_rows(
        p_fn=lambda c: 0.7,
        valid_fn=lambda c: 0.75 if c["order_index"] == 0 else 1.0, seed=7))
    eff = analysis.scenario_effect(df, "s1")
    assert eff["invalid_rate_diff_by_content"] > 0.10
    dec = analysis.graduation_decision(df, "s1", TH, pc_passed=True, nc_p95=0.05)
    assert not dec["c9_invalid_balance"]


def test_8_pc_failure_blocks():
    pc = synth_rows(scenario_id="pc1", family="PC",
                    p_fn=lambda c: 0.5, seed=8)   # PC at chance = broken
    df = frame_of(pc)
    gate = analysis.pc_gate(df, TH)
    assert not gate["pass"]
    ar = frame_of(synth_rows(p_fn=lambda c: 0.8, seed=9))
    dec = analysis.graduation_decision(ar, "s1", TH,
                                       pc_passed=gate["pass"], nc_p95=0.05)
    assert not dec["graduates"] and not dec["c1_pc_gate"]


def test_9_clean_null_stays_null():
    df = frame_of(synth_rows(p_fn=lambda c: 0.5, seed=10))
    dec = analysis.graduation_decision(df, "s1", TH, pc_passed=True, nc_p95=0.05)
    assert not dec["graduates"]
    assert abs(dec["effect"]) < 0.10


def test_10_mixed_construct_aggregate():
    g1 = synth_rows(scenario_id="a1", construct_id="axis1",
                    p_fn=lambda c: 0.8, seed=11)
    g2 = synth_rows(scenario_id="a2", construct_id="axis1",
                    p_fn=lambda c: 0.78, seed=12)
    g3 = synth_rows(scenario_id="b1", construct_id="axis2",
                    p_fn=lambda c: 0.5, seed=13)
    nc = synth_rows(scenario_id="nc1", family="NC",
                    p_fn=lambda c: 0.5, seed=14)
    df = frame_of(g1, g2, g3, nc)
    floor = analysis.nc_floor(df, TH)
    decisions = [analysis.graduation_decision(df, s, TH, pc_passed=True,
                                              nc_p95=floor["nc_p95"])
                 for s in ("a1", "a2", "b1", "nc1")]
    agg = analysis.aggregate_battery(df, decisions, floor)
    assert "axis1" in agg["within_construct_signed_means_axes_ge2"]
    assert agg["within_construct_signed_means_axes_ge2"]["axis1"] > 0.2
    assert "axis2" not in agg["within_construct_signed_means_axes_ge2"]
    assert agg["n_graduated"] >= 2


def test_nc_scenarios_excluded_from_graduation():
    nc = synth_rows(scenario_id="nc_bad", family="NC",
                    p_fn=lambda c: 0.9, seed=15)   # instrument failure world
    df = frame_of(nc)
    dec = analysis.graduation_decision(df, "nc_bad", TH, pc_passed=True,
                                       nc_p95=0.02)
    assert not dec["graduates"]
    assert dec["nc_alarm"]        # stop-and-ask condition M3 fires


def test_nc_analysis_path_identical_to_ar():
    """Identical data under family AR vs NC must produce identical numbers
    everywhere except the family-based graduation exclusion."""
    rows_ar = synth_rows(scenario_id="same", family="AR",
                         p_fn=lambda c: 0.7, seed=16)
    rows_nc = [dict(r, family="NC") for r in rows_ar]
    df_ar, df_nc = frame_of(rows_ar), frame_of(rows_nc)
    ea = analysis.scenario_effect(df_ar, "same")
    en = analysis.scenario_effect(df_nc, "same")
    assert ea["effect"] == en["effect"]
    assert analysis.nuisance_effects(df_ar, "same") == \
        analysis.nuisance_effects(df_nc, "same")
    da = analysis.graduation_decision(df_ar, "same", TH, pc_passed=True,
                                      nc_p95=0.05)
    dn = analysis.graduation_decision(df_nc, "same", TH, pc_passed=True,
                                      nc_p95=0.05)
    for crit in [k for k in da if k.startswith("c")]:
        assert da[crit] == dn[crit]
    assert da["graduates"] and not dn["graduates"]


def test_stated_revealed_matching():
    ar = synth_rows(p_fn=lambda c: 0.8, seed=17)
    ro = synth_rows(channel="RO", p_fn=lambda c: 0.8, seed=18)
    for r in ro:
        r["consequence_frame"] = None
        r["binding_executed"] = False
    df = frame_of(ar, ro)
    pairs = analysis.stated_revealed_rows(df)
    assert pairs, "no matched pairs found"
    summ = analysis.stated_revealed_summary(pairs)
    agree = [s["agreement"] for s in summ if s["agreement"] == s["agreement"]]
    assert agree and all(a > 0.5 for a in agree)
