"""Report coupling (plan Part VII; addendum G; deviation D3: the coupling
primary is the RO full-target margin contrast).

Order per plan §45/§70: natural-state readout (DECODE evidence only),
then upstream AR-fitted interventions at ro_context_end / ro_menu_end,
heldout-codebook transfer, controls, propagation; the final-token
direct-steering positive control runs AFTER the upstream primaries.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np

from .artifacts import atomic_write_json
from .canonical import stable_seed
from .capture import CaptureReader
from .chat import render_item_prompt, target_ids
from .mechanism_run import (intervened_generation, intervened_margin,
                            random_directions, recapture_downstream)
from .parser import parse_strict
from .stats import SEED_BASE, exact_sign_flip_p, holm

RO_PRIMARY_SITE = "ro_context_end"
RO_SECONDARY_SITE = "ro_menu_end"
N_CONTROL_RECEIVERS = 16


def _ro_rows(results, scenario_id, *, reserved: bool) -> list[dict]:
    rows = [r for r in results
            if r["scenario_id"] == scenario_id and r["channel"] == "RO"
            and bool(r["codebook_reserved"]) == reserved]
    rows.sort(key=lambda r: r["item_id"])
    return rows


def natural_readout(reader: CaptureReader, results, *, scenario_id: str,
                    direction: np.ndarray, depth: int,
                    site: str = RO_PRIMARY_SITE) -> dict[str, Any]:
    rows = _ro_rows(results, scenario_id, reserved=False)
    if len(rows) < 6:
        return {"error": "too few RO rows", "n": len(rows)}
    get = reader.state_fn(site, depth)
    proj = np.array([get(r["item_id"]) @ direction for r in rows])
    marg = np.array([float(r["margin_full_a_minus_b"]) for r in rows])
    corr = (float(np.corrcoef(proj, marg)[0, 1])
            if np.std(marg) > 0 else float("nan"))
    # strict report prediction AUC
    sems = [r["parsed_sem"] for r in rows]
    mask = np.array([s in ("a", "b") for s in sems])
    auc = float("nan")
    if mask.sum() >= 6:
        p = proj[mask]
        lab = np.array([s == "a" for s in np.array(sems)[mask]])
        pos, neg = p[lab], p[~lab]
        if len(pos) and len(neg):
            auc = float(((pos[:, None] > neg[None, :]).sum()
                         + 0.5 * (pos[:, None] == neg[None, :]).sum())
                        / (len(pos) * len(neg)))
    rng_dirs = random_directions(len(direction), 200,
                                 f"ro-nat-{scenario_id}")
    rand = []
    for v in rng_dirs:
        pv = np.array([get(r["item_id"]) @ v for r in rows])
        if np.std(marg) > 0 and np.std(pv) > 0:
            rand.append(abs(float(np.corrcoef(pv, marg)[0, 1])))
    band = float(np.quantile(rand, 0.95)) if rand else float("nan")
    # heldout paraphrase + reserved codebook survival
    strata = {}
    for para in (0, 1):
        sub = [i for i, r in enumerate(rows) if r["paraphrase_id"] == para]
        if len(sub) >= 4 and np.std(marg[sub]) > 0:
            strata[f"para{para}"] = float(
                np.corrcoef(proj[sub], marg[sub])[0, 1])
    res_rows = _ro_rows(results, scenario_id, reserved=True)
    corr_res = float("nan")
    if len(res_rows) >= 6:
        pr = np.array([get(r["item_id"]) @ direction for r in res_rows])
        mr = np.array([float(r["margin_full_a_minus_b"]) for r in res_rows])
        if np.std(mr) > 0:
            corr_res = float(np.corrcoef(pr, mr)[0, 1])
    constant_code_rate = float(np.mean(
        [r["parsed_sem"] == max(set(s for s in sems if s),
                                key=[s for s in sems if s].count)
         for r in rows if r["parsed_sem"]])) if any(sems) else float("nan")
    return {
        "site": site, "depth": depth, "n_rows": len(rows),
        "corr_margin": corr, "auc_strict": auc, "rand_band_p95": band,
        "beats_band": bool(np.isfinite(corr) and np.isfinite(band)
                           and abs(corr) > band),
        "strata_corr": strata, "corr_reserved_codebook": corr_res,
        "ro_constant_code_rate": constant_code_rate,
    }


def _prep(bundle, pin, row) -> dict:
    rp = render_item_prompt(bundle.tokenizer, pin, row)
    return {"ids": rp.input_ids, "sites": dict(rp.site_token_index),
            "ids_a": list(target_ids(bundle.tokenizer,
                                     row["response_code_by_sem"]["a"])),
            "ids_b": list(target_ids(bundle.tokenizer,
                                     row["response_code_by_sem"]["b"])),
            "codes": list(row["valid_codes_in_display_order"]),
            "baseline": float(row["margin_full_a_minus_b"])}


def ro_intervention(bundle, pin, results, bank_rows_by_id, reader,
                    *, scenario_id: str, direction: np.ndarray,
                    dose: float, depth: int,
                    controls: dict[str, np.ndarray | None],
                    out_dir: pathlib.Path) -> dict[str, Any]:
    block_index = depth - 1
    receivers = [r for r in _ro_rows(results, scenario_id, reserved=False)
                 if r["incidental_split"] == "holdout"]
    reserved = [r for r in _ro_rows(results, scenario_id, reserved=True)
                if r["incidental_split"] == "holdout"]
    preps = {r["item_id"]: _prep(bundle, pin, bank_rows_by_id[r["item_id"]])
             for r in receivers + reserved}

    def pm(r, vector, site_key) -> float:
        p = preps[r["item_id"]]
        args = dict(block_index=block_index, position=p["sites"][site_key],
                    vector=vector, mode="add")
        plus = intervened_margin(bundle, p["ids"], p["ids_a"], p["ids_b"],
                                 scale=dose, **args)
        minus = intervened_margin(bundle, p["ids"], p["ids_a"], p["ids_b"],
                                  scale=-dose, **args)
        return plus - minus

    primary_pm = [pm(r, direction, RO_PRIMARY_SITE) for r in receivers]
    menu_pm = [pm(r, direction, RO_SECONDARY_SITE)
               for r in receivers[:N_CONTROL_RECEIVERS]]
    reserved_pm = [pm(r, direction, RO_PRIMARY_SITE) for r in reserved]
    ctl_pm = {}
    for name, vec in controls.items():
        if vec is None:
            continue
        n_recv = (len(receivers) if name in ("d_wrong_scenario", "d_code")
                  else N_CONTROL_RECEIVERS)
        ctl_pm[name] = [pm(r, vec, RO_PRIMARY_SITE)
                        for r in receivers[:n_recv]]
    randoms = random_directions(len(direction), 4, f"ro-int-{scenario_id}")
    rand_pm = [float(np.mean([pm(r, v, RO_PRIMARY_SITE)
                              for r in receivers[:8]])) for v in randoms]
    final_pm = [pm(r, direction, "ro_final_prompt_token")
                for r in receivers[:N_CONTROL_RECEIVERS]]

    # strict comparative-report shift (secondary per D3)
    shift = {"clean": [], "plus": [], "minus": []}
    for r in receivers:
        p = preps[r["item_id"]]
        for name, scale in (("clean", 0.0), ("plus", dose),
                            ("minus", -dose)):
            raw = intervened_generation(
                bundle, p["ids"], block_index=block_index,
                position=p["sites"][RO_PRIMARY_SITE],
                vector=None if name == "clean" else direction,
                mode="noop" if name == "clean" else "add", scale=scale)
            parsed = parse_strict(raw, p["codes"])
            sem = None
            if parsed.parse_status == "valid":
                sem = ("a" if parsed.parsed_response_code
                       == bank_rows_by_id[r["item_id"]]
                       ["response_code_by_sem"]["a"] else "b")
            shift[name].append(sem)
    def a_rate(key):
        vals = [s == "a" for s in shift[key] if s is not None]
        return float(np.mean(vals)) if vals else float("nan")
    report_shift = a_rate("plus") - a_rate("minus")

    # propagation downstream of ro_context_end
    prop = []
    for r in receivers[:8]:
        p = preps[r["item_id"]]
        down = {"ro_menu_end": p["sites"]["ro_menu_end"],
                "ro_response_start": p["sites"]["ro_response_start"],
                "ro_final_prompt_token": p["sites"]["ro_final_prompt_token"]}
        plus = recapture_downstream(
            bundle, p["ids"], block_index=block_index,
            position=p["sites"][RO_PRIMARY_SITE], vector=direction,
            mode="add", scale=dose, capture_positions=down, depth=depth)
        minus = recapture_downstream(
            bundle, p["ids"], block_index=block_index,
            position=p["sites"][RO_PRIMARY_SITE], vector=direction,
            mode="add", scale=-dose, capture_positions=down, depth=depth)
        prop.append({k: float((plus[k] - minus[k]) @ direction)
                     for k in down})
    prop_menu = [x["ro_menu_end"] for x in prop]

    primary = float(np.mean(primary_pm))
    wrong_mean = float(np.mean(ctl_pm.get("d_wrong_scenario", [np.nan])))
    code_mean = float(np.mean(ctl_pm.get("d_code", [np.nan])))
    ps = {
        "C1_upstream_ro_margin": exact_sign_flip_p(
            np.array(primary_pm), seed_key=f"c1-{scenario_id}"),
        "C2_reserved_codebook": exact_sign_flip_p(
            np.array(reserved_pm), seed_key=f"c2-{scenario_id}"),
        "C3_wrong_scenario_smaller": exact_sign_flip_p(
            np.array(primary_pm)
            - np.array(ctl_pm.get("d_wrong_scenario",
                                  [0.0] * len(primary_pm))[:len(primary_pm)]),
            seed_key=f"c3-{scenario_id}"),
        "C4_code_smaller": exact_sign_flip_p(
            np.array(primary_pm)
            - np.array(ctl_pm.get("d_code",
                                  [0.0] * len(primary_pm))[:len(primary_pm)]),
            seed_key=f"c4-{scenario_id}"),
        "C5_propagation": exact_sign_flip_p(
            np.array(prop_menu), seed_key=f"c5-{scenario_id}"),
    }
    hh = holm(ps)
    upstream_pass = hh["C1_upstream_ro_margin"]["reject_at_05"] and primary > 0
    reserved_pass = (hh["C2_reserved_codebook"]["reject_at_05"]
                     and np.mean(reserved_pm) > 0)
    spec = (np.isfinite(wrong_mean) and abs(wrong_mean) < 0.5 * abs(primary)
            and np.isfinite(code_mean)
            and abs(code_mean) < 0.5 * abs(primary))
    prop_pass = hh["C5_propagation"]["reject_at_05"]
    final_mean = float(np.mean(final_pm))
    non_margin_moves = (np.isfinite(report_shift)
                        and abs(report_shift) >= 0.15
                        and np.sign(report_shift) == np.sign(primary))

    result = {
        "scenario_id": scenario_id,
        "n_receivers": len(receivers),
        "n_reserved": len(reserved),
        "upstream_ro_margin_pm_mean": primary,
        "menu_site_pm_mean": float(np.mean(menu_pm)),
        "reserved_pm_mean": float(np.mean(reserved_pm)) if reserved_pm else float("nan"),
        "controls_pm_mean": {k: float(np.mean(v)) for k, v in ctl_pm.items()},
        "randoms_pm_mean": rand_pm,
        "final_token_pm_mean": final_mean,
        "strict_report": {"clean_a": a_rate("clean"),
                          "plus_a": a_rate("plus"),
                          "minus_a": a_rate("minus"),
                          "shift": report_shift},
        "propagation_menu_mean": float(np.mean(prop_menu)) if prop_menu else float("nan"),
        "primaries_holm": hh,
        "gates": {"upstream_pass": bool(upstream_pass),
                  "reserved_pass": bool(reserved_pass),
                  "specificity": bool(spec),
                  "propagation_pass": bool(prop_pass),
                  "non_margin_endpoint_moves": bool(non_margin_moves),
                  "final_token_only": bool(not upstream_pass
                                           and abs(final_mean) > 0.2)},
    }
    return result


def route(ar_result: dict[str, Any], readout: dict[str, Any],
          intervention: dict[str, Any]) -> str:
    """Coupling router (plan §48 + D3)."""
    g = intervention["gates"]
    ar_ok = ar_result.get("gates", {}).get("margin_handle", False)
    if not ar_ok:
        return "NO_AR_HANDLE"
    if g["final_token_only"]:
        return "DIRECT_OUTPUT"
    coupled = (g["upstream_pass"] and g["reserved_pass"] and g["specificity"]
               and g["propagation_pass"] and readout.get("beats_band", False)
               and g["non_margin_endpoint_moves"])
    if coupled:
        return "CHOICE_REPORT_COUPLED"
    if g["upstream_pass"] and g["specificity"]:
        return "COUPLED_MARGIN_ONLY" if not g["non_margin_endpoint_moves"] \
            else "BEHAVIOR_SPECIFIC"
    return "BEHAVIOR_SPECIFIC"
