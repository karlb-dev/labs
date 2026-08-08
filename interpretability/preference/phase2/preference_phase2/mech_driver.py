"""S6-S9 orchestration: prechecks -> mechanistic PC -> AR mechanism ->
coupling (plan §65-§70; addendum F/G/E12). One scenario block runs the
complete assay stack against the frozen S4 captures and rows.

Cost containment (declared): the eight random-direction controls and the
context-text/semantic-identity/format/position controls run on the first
16 holdout receivers in canonical order; all seven Holm primaries (M1-M7)
run on the full holdout receiver set. Holdout opens exactly once per
scenario, in the assay block.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np

from . import paths
from .artifacts import atomic_write_json, read_jsonl
from .capture import CaptureReader
from .chat import render_item_prompt, target_ids
from .mechanism import (fit_direction, identifiability_precheck,
                        select_site_depth)
from .mechanism_run import (DOSE_GRID, N_RANDOM_CONTROLS, PRIMARY_SITES,
                            code_gradient_direction, dose_guardrail,
                            fit_factor_direction, intervened_generation,
                            intervened_margin, random_directions,
                            recapture_downstream, site_hook)
from .parser import parse_strict
from .stats import exact_sign_flip_p, holm

SCAN_SITES = ("context_end", "option_a_end", "option_b_end", "menu_end",
              "response_instruction_start")
N_CONTROL_RECEIVERS = 16


def _rows_for(results: list[dict], scenario_id: str, *, bank: str,
              channel: str = "AR") -> list[dict]:
    rows = [r for r in results
            if r["scenario_id"] == scenario_id and r["bank"] == bank
            and r["channel"] == channel]
    rows.sort(key=lambda r: r["item_id"])
    return rows


def precheck_scenario(results, reader: CaptureReader, scenario_id: str, *,
                      bank: str, depths: list[int],
                      behavioral_pass: bool) -> dict[str, Any]:
    rows = [r for r in _rows_for(results, scenario_id, bank=bank)
            if not r["codebook_reserved"]]
    cells = []
    for site in SCAN_SITES:
        for depth in depths:
            get = reader.state_fn(site, depth)
            pre = identifiability_precheck(
                rows, get, scenario_id=scenario_id, site=site, depth=depth,
                behavioral_slope_passes=behavioral_pass)
            cells.append(pre)
    best = select_site_depth(cells)
    return {"scenario_id": scenario_id, "cells": cells, "selected": best}


def _receivers(results, scenario_id, *, bank, reserved: bool) -> list[dict]:
    return [r for r in _rows_for(results, scenario_id, bank=bank)
            if r["incidental_split"] == "holdout"
            and int(r["context_strength"]) == 0
            and bool(r["codebook_reserved"]) == reserved]


def _donors(results, scenario_id, receiver, *, bank,
            strength: int) -> tuple[dict | None, dict | None]:
    """E12 matching: same incidental/order/cmap/paraphrase/codebook."""
    cands = [r for r in _rows_for(results, scenario_id, bank=bank)
             if r["incidental_id"] == receiver["incidental_id"]
             and r["display_order"] == receiver["display_order"]
             and r["code_map_index"] == receiver["code_map_index"]
             and r["paraphrase_id"] == receiver["paraphrase_id"]
             and r["codebook_pair_id"] == receiver["codebook_pair_id"]]
    plus = next((r for r in cands
                 if int(r["context_strength"]) == strength), None)
    minus = next((r for r in cands
                  if int(r["context_strength"]) == -strength), None)
    return plus, minus


def _prep(bundle, pin, row) -> dict:
    rp = render_item_prompt(bundle.tokenizer, pin, row)
    return {
        "ids": rp.input_ids,
        "sites": dict(rp.site_token_index),
        "ids_a": list(target_ids(bundle.tokenizer,
                                 row["response_code_by_sem"]["a"])),
        "ids_b": list(target_ids(bundle.tokenizer,
                                 row["response_code_by_sem"]["b"])),
        "codes": list(row["valid_codes_in_display_order"]),
        "baseline": float(row["margin_full_a_minus_b"]),
    }


def run_mech_scenario(bundle, pin, results, bank_rows_by_id, reader,
                      *, scenario_id: str, bank: str,
                      precheck: dict[str, Any],
                      wrong_direction: np.ndarray | None,
                      out_dir: pathlib.Path,
                      is_pc: bool = False) -> dict[str, Any]:
    sel = precheck["selected"]
    if sel is None:
        result = {"scenario_id": scenario_id,
                  "status": "DIRECTION_NOT_IDENTIFIABLE",
                  "n_ready_cells": 0}
        atomic_write_json(out_dir / f"mech_{scenario_id}.json", result)
        return result
    site, depth = sel["site"], sel["depth"]
    block_index = depth - 1
    rows = [r for r in _rows_for(results, scenario_id, bank=bank)
            if not r["codebook_reserved"]]
    get = reader.state_fn(site, depth)
    fit = fit_direction(rows, get)
    d = fit["direction"].astype(np.float32)
    train = [r for r in rows if r["incidental_split"] == "train"]
    proj_sd = float(np.std([get(r["item_id"]) @ d for r in train]))

    # ---- controls (same estimator / same site) ----------------------------
    controls: dict[str, np.ndarray | None] = {
        "d_position": fit_factor_direction(rows, get,
                                           factor="display_order",
                                           level_pos=0, level_neg=1),
        "d_code": fit_factor_direction(rows, get, factor="code_map_index",
                                       level_pos=0, level_neg=1),
        "d_wrong_scenario": wrong_direction,
    }
    # semantic-identity control: option_a_end minus option_b_end states on
    # neutral train rows (the "which option text was read" direction)
    ga = reader.state_fn("option_a_end", depth)
    gb = reader.state_fn("option_b_end", depth)
    neu_train = [r for r in train if int(r["context_strength"]) == 0]
    if neu_train:
        raw = np.mean([ga(r["item_id"]) - gb(r["item_id"])
                       for r in neu_train], axis=0)
        n = np.linalg.norm(raw)
        controls["d_semantic_identity"] = (
            (raw / n).astype(np.float32) if n > 0 else None)
    g_code = code_gradient_direction(
        bundle,
        codes_a=[r["response_code_by_sem"]["a"] for r in train[:8]],
        codes_b=[r["response_code_by_sem"]["b"] for r in train[:8]])
    randoms = random_directions(len(d), N_RANDOM_CONTROLS,
                                f"{scenario_id}|{site}|{depth}")

    # ---- dose selection on validation (frozen before holdout) ------------
    val_neutral = [r for r in rows if r["incidental_split"] == "validation"
                   and int(r["context_strength"]) == 0][:12]
    preps_val = {r["item_id"]: _prep(bundle, pin, bank_rows_by_id[r["item_id"]])
                 for r in val_neutral}
    dose_rows = []
    for beta in DOSE_GRID:
        deltas = []
        for r in val_neutral:
            p = preps_val[r["item_id"]]
            pos = p["sites"][site]
            m_plus = intervened_margin(
                bundle, p["ids"], p["ids_a"], p["ids_b"],
                block_index=block_index, position=pos, vector=d,
                mode="add", scale=beta * proj_sd)
            m_minus = intervened_margin(
                bundle, p["ids"], p["ids_a"], p["ids_b"],
                block_index=block_index, position=pos, vector=d,
                mode="add", scale=-beta * proj_sd)
            deltas.append(m_plus - m_minus)
        guard = dose_guardrail(bundle, pin, block_index=block_index,
                               position_frac=0.5, vector=d, mode="add",
                               scale=beta * proj_sd)
        dose_rows.append({"beta": beta, "pm_contrast_mean":
                          float(np.mean(deltas)), "guard": guard})
    admissible = [dr for dr in dose_rows if dr["guard"]["passes"]]
    monotone = all(
        dose_rows[i]["pm_contrast_mean"] <= dose_rows[i + 1]["pm_contrast_mean"]
        or abs(dose_rows[i + 1]["pm_contrast_mean"]) >=
        abs(dose_rows[i]["pm_contrast_mean"]) * 0.8
        for i in range(len(dose_rows) - 1))
    beta = (max(admissible, key=lambda dr: dr["beta"])["beta"]
            if admissible else None)
    if beta is None:
        result = {"scenario_id": scenario_id, "status": "NO_GUARD_SAFE_DOSE",
                  "dose_rows": dose_rows, "selected": {"site": site,
                                                       "depth": depth}}
        atomic_write_json(out_dir / f"mech_{scenario_id}.json", result)
        return result
    dose = beta * proj_sd

    # ---- holdout assays (holdout opens HERE, once) ------------------------
    receivers = _receivers(results, scenario_id, bank=bank, reserved=False)
    reserved_receivers = _receivers(results, scenario_id, bank=bank,
                                    reserved=True)
    preps = {r["item_id"]: _prep(bundle, pin, bank_rows_by_id[r["item_id"]])
             for r in receivers + reserved_receivers}

    def margin_delta(r, vector, mode, scale, *, site_key=site) -> float:
        p = preps[r["item_id"]]
        m = intervened_margin(bundle, p["ids"], p["ids_a"], p["ids_b"],
                              block_index=block_index,
                              position=p["sites"][site_key],
                              vector=vector, mode=mode, scale=scale)
        return m - p["baseline"]

    m1_patch, m1_mono = [], []
    for r in receivers:
        plus2, minus2 = _donors(results, scenario_id, r, bank=bank,
                                strength=2)
        if not plus2 or not minus2:
            continue
        va = reader.get(plus2["item_id"], site, depth)
        vb = reader.get(minus2["item_id"], site, depth)
        da = margin_delta(r, va, "patch", 1.0)
        db = margin_delta(r, vb, "patch", 1.0)
        m1_patch.append(da - db)
        plus1, minus1 = _donors(results, scenario_id, r, bank=bank,
                                strength=1)
        if plus1 and minus1:
            va1 = reader.get(plus1["item_id"], site, depth)
            vb1 = reader.get(minus1["item_id"], site, depth)
            m1_mono.append((da - db) - (margin_delta(r, va1, "patch", 1.0)
                                        - margin_delta(r, vb1, "patch", 1.0)))
    self_noop = []
    for r in receivers[:N_CONTROL_RECEIVERS]:
        v_self = reader.get(r["item_id"], site, depth)
        self_noop.append(margin_delta(r, v_self, "patch", 1.0))

    m2_pm = [margin_delta(r, d, "add", dose)
             - margin_delta(r, d, "add", -dose) for r in receivers]
    m3_rem = [margin_delta(r, d, "remove", 1.0) for r in receivers]
    m3_sens = {a: float(np.mean([margin_delta(r, d, "remove", a)
                                 for r in receivers[:N_CONTROL_RECEIVERS]]))
               for a in (0.5, 1.5)}
    m4_pm = [margin_delta(r, d, "add", dose)
             - margin_delta(r, d, "add", -dose)
             for r in reserved_receivers]
    ctl_pm: dict[str, list[float]] = {}
    for name, vec in controls.items():
        if vec is None:
            continue
        ctl_pm[name] = [margin_delta(r, vec, "add", dose)
                        - margin_delta(r, vec, "add", -dose)
                        for r in (receivers
                                  if name in ("d_wrong_scenario", "d_code")
                                  else receivers[:N_CONTROL_RECEIVERS])]
    rand_pm = []
    for i, rv in enumerate(randoms):
        rand_pm.append(float(np.mean(
            [margin_delta(r, rv, "add", dose)
             - margin_delta(r, rv, "add", -dose)
             for r in receivers[:N_CONTROL_RECEIVERS]])))
    wrong_site_pm = [margin_delta(r, d, "add", dose,
                                  site_key="response_instruction_start")
                     - margin_delta(r, d, "add", -dose,
                                    site_key="response_instruction_start")
                     for r in receivers[:N_CONTROL_RECEIVERS]]
    final_tok_pm = [margin_delta(r, d, "add", dose,
                                 site_key="final_prompt_token")
                    - margin_delta(r, d, "add", -dose,
                                   site_key="final_prompt_token")
                    for r in receivers[:N_CONTROL_RECEIVERS]]

    # M7 propagation: intended downstream projection movement
    prop = []
    for r in receivers[:N_CONTROL_RECEIVERS]:
        p = preps[r["item_id"]]
        down_sites = {"menu_end": p["sites"]["menu_end"],
                      "response_instruction_start":
                      p["sites"]["response_instruction_start"],
                      "final_prompt_token": p["sites"]["final_prompt_token"]}
        plus = recapture_downstream(
            bundle, p["ids"], block_index=block_index,
            position=p["sites"][site], vector=d, mode="add", scale=dose,
            capture_positions=down_sites, depth=depth)
        minus = recapture_downstream(
            bundle, p["ids"], block_index=block_index,
            position=p["sites"][site], vector=d, mode="add", scale=-dose,
            capture_positions=down_sites, depth=depth)
        prop.append({k: float((plus[k] - minus[k]) @ d)
                     for k in down_sites})
    injected_effect = 2 * dose
    prop_menu = [p_["menu_end"] for p_ in prop]
    prop_ok = (bool(prop_menu)
               and float(np.mean(prop_menu)) >= 0.5 * injected_effect
               if site == "context_end" else True)

    # strict outputs on holdout receivers
    flips = {"clean": [], "plus": [], "minus": [], "removed": []}
    for r in receivers:
        p = preps[r["item_id"]]
        conds = {
            "clean": (None, "noop", 0.0),
            "plus": (d, "add", dose),
            "minus": (d, "add", -dose),
            "removed": (d, "remove", 1.0),
        }
        for name, (vec, mode, scale) in conds.items():
            raw = intervened_generation(
                bundle, p["ids"], block_index=block_index,
                position=p["sites"][site], vector=vec, mode=mode,
                scale=scale)
            parsed = parse_strict(raw, p["codes"])
            sem = None
            if parsed.parse_status == "valid":
                sem = ("a" if parsed.parsed_response_code
                       == bank_rows_by_id[r["item_id"]]
                       ["response_code_by_sem"]["a"] else "b")
            flips[name].append(sem)
    n_flips = sum(1 for a, b in zip(flips["clean"], flips["plus"])
                  if a is not None and b is not None and a != b)
    n_flips += sum(1 for a, b in zip(flips["clean"], flips["minus"])
                   if a is not None and b is not None and a != b)
    plus_rate = float(np.mean([s == "a" for s in flips["plus"]
                               if s is not None])) if flips["plus"] else float("nan")
    minus_rate = float(np.mean([s == "a" for s in flips["minus"]
                                if s is not None])) if flips["minus"] else float("nan")

    # ---- Holm primary family M1-M7 ---------------------------------------
    primary = float(np.mean(m2_pm))
    wrong_mean = float(np.mean(ctl_pm.get("d_wrong_scenario", [np.nan])))
    code_mean = float(np.mean(ctl_pm.get("d_code", [np.nan])))
    ps = {
        "M1_patch": exact_sign_flip_p(np.array(m1_patch),
                                      seed_key=f"m1-{scenario_id}"),
        "M2_addition_pm": exact_sign_flip_p(np.array(m2_pm),
                                            seed_key=f"m2-{scenario_id}"),
        "M3_removal": exact_sign_flip_p(np.array(m3_rem),
                                        seed_key=f"m3-{scenario_id}"),
        "M4_heldout_codebook": exact_sign_flip_p(
            np.array(m4_pm), seed_key=f"m4-{scenario_id}"),
        "M5_wrong_scenario_smaller": exact_sign_flip_p(
            np.array(m2_pm) - np.array(
                ctl_pm.get("d_wrong_scenario", [0.0] * len(m2_pm))[:len(m2_pm)]),
            seed_key=f"m5-{scenario_id}"),
        "M6_code_smaller": exact_sign_flip_p(
            np.array(m2_pm) - np.array(
                ctl_pm.get("d_code", [0.0] * len(m2_pm))[:len(m2_pm)]),
            seed_key=f"m6-{scenario_id}"),
        "M7_propagation": exact_sign_flip_p(np.array(prop_menu),
                                            seed_key=f"m7-{scenario_id}"),
    }
    hh = holm(ps)
    m1_pass = hh["M1_patch"]["reject_at_05"] and np.mean(m1_patch) > 0
    m2_pass = hh["M2_addition_pm"]["reject_at_05"] and primary > 0
    m4_pass = (hh["M4_heldout_codebook"]["reject_at_05"]
               and np.mean(m4_pm) > 0)
    spec_wrong = (np.isfinite(wrong_mean)
                  and abs(wrong_mean) < 0.5 * abs(primary))
    spec_code = (np.isfinite(code_mean)
                 and abs(code_mean) < 0.5 * abs(primary))
    m7_pass = prop_ok and hh["M7_propagation"]["reject_at_05"]
    margin_handle = bool((m1_pass or m2_pass) and m4_pass and spec_wrong
                         and spec_code and m7_pass)
    cos_gcode = (float(abs(d @ g_code)) if g_code is not None
                 else float("nan"))

    if is_pc:
        pc_pass = bool(margin_handle and (n_flips >= 1))
        status = ("PC_MECH_PASS" if pc_pass else "PC_MECH_FAIL")
    else:
        pc_pass = None
        if margin_handle and n_flips >= 1:
            status = "ENACTED_HANDLE"
        elif margin_handle:
            status = "MARGIN_HANDLE"
        elif (float(np.mean(final_tok_pm)) > 0
              and not (m1_pass or m2_pass)):
            status = "DIRECT_OUTPUT"
        else:
            status = "NO_HANDLE"

    result = {
        "scenario_id": scenario_id, "status": status,
        "selected": {"site": site, "depth": depth,
                     "block_index": block_index, "score": sel["score"]},
        "direction_fit": {"n_pairs": fit["n_pairs"],
                          "norm_raw": fit["norm_raw"],
                          "proj_train_sd": proj_sd},
        "dose": {"beta": beta, "scale": dose, "rows": dose_rows,
                 "monotone_on_validation": monotone},
        "holdout": {
            "n_receivers": len(receivers),
            "n_reserved_receivers": len(reserved_receivers),
            "m1_patch_contrast_mean": float(np.mean(m1_patch)) if m1_patch else float("nan"),
            "m1_donor_strength_monotonicity_mean": float(np.mean(m1_mono)) if m1_mono else float("nan"),
            "self_patch_noop_mean_abs": float(np.mean(np.abs(self_noop))) if self_noop else float("nan"),
            "m2_addition_pm_mean": primary,
            "m3_removal_mean": float(np.mean(m3_rem)),
            "m3_removal_sensitivity": m3_sens,
            "m4_reserved_pm_mean": float(np.mean(m4_pm)) if m4_pm else float("nan"),
            "controls_pm_mean": {k: float(np.mean(v))
                                 for k, v in ctl_pm.items()},
            "randoms_pm_mean": rand_pm,
            "randoms_pm_max_abs": float(np.max(np.abs(rand_pm))) if rand_pm else float("nan"),
            "wrong_site_pm_mean": float(np.mean(wrong_site_pm)),
            "final_token_pm_mean": float(np.mean(final_tok_pm)),
            "propagation": {"menu_end_mean": float(np.mean(prop_menu)) if prop_menu else float("nan"),
                            "injected_effect": injected_effect,
                            "ok": bool(prop_ok)},
            "strict": {"n_flips": n_flips,
                       "plus_a_rate": plus_rate,
                       "minus_a_rate": minus_rate,
                       "clean_a_rate": float(np.mean(
                           [s == "a" for s in flips["clean"]
                            if s is not None])) if flips["clean"] else float("nan")},
        },
        "output_adjacency": {"cos_to_code_gradient": cos_gcode},
        "primaries_holm": hh,
        "gates": {"m1_pass": bool(m1_pass), "m2_pass": bool(m2_pass),
                  "m4_pass": bool(m4_pass), "spec_wrong": bool(spec_wrong),
                  "spec_code": bool(spec_code), "m7_pass": bool(m7_pass),
                  "margin_handle": margin_handle,
                  "strict_flips": int(n_flips)},
        "pc_mech_pass": pc_pass,
        "holdout_opened": True,
    }
    atomic_write_json(out_dir / f"mech_{scenario_id}.json", result)
    return result
