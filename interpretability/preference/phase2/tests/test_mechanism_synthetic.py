"""Mechanism synthetic worlds (plan §54.5) through the real estimator,
gates, and selection paths, with planted state geometry."""

import numpy as np
import pytest

from preference_phase2.canonical import stable_seed
from preference_phase2.mechanism import (decoder_eval, fit_direction,
                                         identifiability_precheck,
                                         select_site_depth)
from preference_phase2.retrodict import synthetic_intercept_blindness

DIM = 96


def _mech_world(*, signal=1.2, noise=0.8, dim=DIM, n_inc=32, seed=1,
                margin_slope=0.8, neutral_align=True):
    """Planted context direction world over a B-MECH-shaped grid."""
    rng = np.random.default_rng(stable_seed("mechworld", seed, base=2262))
    d_true = rng.standard_normal(dim)
    d_true /= np.linalg.norm(d_true)
    rows, states = [], {}
    idx = 0
    for i in range(n_inc):
        split = ("train" if i < 16 else
                 "validation" if i < 24 else "holdout")
        base = rng.standard_normal(dim) * 0.3   # incidental offset
        neutral_default = rng.normal(0.4, 1.0)  # per-incidental default
        for s in (-2, -1, 0, 1, 2):
            for order in (0, 1):
                for cmap in (0, 1):
                    item_id = f"mw-{seed}-{idx}"
                    idx += 1
                    h = (signal * s * d_true + base
                         + noise * rng.standard_normal(dim))
                    if neutral_align:
                        h = h + neutral_default * d_true
                    m = (neutral_default + margin_slope * s
                         + 0.1 * rng.standard_normal())
                    states[item_id] = h.astype(np.float32)
                    rows.append({
                        "item_id": item_id, "incidental_id": f"i{i:02d}",
                        "incidental_split": split,
                        "display_order": order, "code_map_index": cmap,
                        "paraphrase_id": 0,
                        "codebook_pair_id": f"ar{i % 3}",
                        "codebook_reserved": False,
                        "context_strength": s,
                        "margin_full_a_minus_b": m,
                    })
    return rows, states, d_true


def test_planted_direction_recovered_and_gates_pass():
    rows, states, d_true = _mech_world(seed=11)
    get = lambda i: states[i]
    fit = fit_direction(rows, get)
    assert fit["ok"]
    assert abs(float(fit["direction"] @ d_true)) > 0.85
    pre = identifiability_precheck(
        rows, get, scenario_id="mw", site="context_end", depth=10,
        behavioral_slope_passes=True)
    assert pre["ready"], pre["checks"]


def test_label_permutation_rejected():
    """Shuffled strengths must not decode (permutation band logic)."""
    rows, states, _ = _mech_world(seed=12)
    rng = np.random.default_rng(0)
    shuffled = [dict(r, context_strength=int(s)) for r, s in
                zip(rows, rng.permutation(
                    [r["context_strength"] for r in rows]))]
    get = lambda i: states[i]
    fit = fit_direction(shuffled, get)
    if fit["ok"]:
        ev = decoder_eval(fit["direction"], shuffled, get,
                          split="validation")
        assert not np.isfinite(ev["corr"]) or abs(ev["corr"]) < 0.25


def test_no_signal_world_not_ready():
    rows, states, _ = _mech_world(signal=0.0, margin_slope=0.0, seed=13)
    pre = identifiability_precheck(
        rows, lambda i: states[i], scenario_id="mw0", site="context_end",
        depth=10, behavioral_slope_passes=True)
    assert not pre["ready"]


def test_behavioral_gate_blocks_mechanism():
    rows, states, _ = _mech_world(seed=14)
    pre = identifiability_precheck(
        rows, lambda i: states[i], scenario_id="mwb", site="context_end",
        depth=10, behavioral_slope_passes=False)
    assert not pre["ready"]
    assert not pre["checks"]["behavioral_slope"]


def test_site_selection_prefers_upstream_on_tie():
    rows, states, _ = _mech_world(seed=15)
    get = lambda i: states[i]
    cells = []
    for site, depth in (("context_end", 10), ("menu_end", 10),
                        ("final_prompt_token", 10)):
        pre = identifiability_precheck(
            rows, get, scenario_id="mws", site=site, depth=depth,
            behavioral_slope_passes=True)
        cells.append(pre)
    best = select_site_depth(cells)
    assert best is not None
    assert best["site"] == "context_end"   # identical stats -> upstream wins


def test_shallower_depth_on_site_tie():
    rows, states, _ = _mech_world(seed=16)
    get = lambda i: states[i]
    cells = [identifiability_precheck(rows, get, scenario_id="mwd",
                                      site="context_end", depth=d,
                                      behavioral_slope_passes=True)
             for d in (30, 12)]
    best = select_site_depth(cells)
    assert best["depth"] == 12


def test_holdout_never_selected_from():
    """Selection consumes validation only; holdout rows influence nothing
    in the precheck decoder gate."""
    rows, states, d_true = _mech_world(seed=17)
    # corrupt holdout states: if selection touched holdout, gates change
    corrupted = dict(states)
    rng = np.random.default_rng(3)
    for r in rows:
        if r["incidental_split"] == "holdout":
            corrupted[r["item_id"]] = rng.standard_normal(DIM).astype(
                np.float32)
    pre_a = identifiability_precheck(
        rows, lambda i: states[i], scenario_id="mwh", site="context_end",
        depth=10, behavioral_slope_passes=True)
    pre_b = identifiability_precheck(
        rows, lambda i: corrupted[i], scenario_id="mwh",
        site="context_end", depth=10, behavioral_slope_passes=True)
    assert pre_a["ready"] == pre_b["ready"]
    assert (pre_a["validation_eval"]["corr"]
            == pytest.approx(pre_b["validation_eval"]["corr"], abs=1e-9))


def test_intercept_blindness_retrodiction():
    res = synthetic_intercept_blindness()
    assert res["passed"], res


def test_direct_output_geometry_flagged():
    """A world where only the final-token site carries signal: upstream
    cells are not ready, the final-token cell is — the selection would
    surface only the direct-output positive-control site, which the
    assay layer treats as DIRECT_OUTPUT, never a semantic handle."""
    rows_up, states_up, _ = _mech_world(signal=0.0, margin_slope=0.8,
                                        seed=18)
    rows_ft, states_ft, _ = _mech_world(signal=1.2, margin_slope=0.8,
                                        seed=18)
    get_up = lambda i: states_up[i]
    get_ft = lambda i: states_ft[i]
    pre_up = identifiability_precheck(
        rows_up, get_up, scenario_id="mwft", site="context_end", depth=10,
        behavioral_slope_passes=True)
    pre_ft = identifiability_precheck(
        rows_ft, get_ft, scenario_id="mwft", site="final_prompt_token",
        depth=10, behavioral_slope_passes=True)
    assert not pre_up["ready"]
    assert pre_ft["ready"]
    best = select_site_depth([pre_up, pre_ft])
    assert best["site"] == "final_prompt_token"


def test_wrong_scenario_direction_distinct():
    rows_a, states_a, d_a = _mech_world(seed=19)
    rows_b, states_b, d_b = _mech_world(seed=20)
    fa = fit_direction(rows_a, lambda i: states_a[i])
    fb = fit_direction(rows_b, lambda i: states_b[i])
    assert abs(float(fa["direction"] @ fb["direction"])) < 0.3
