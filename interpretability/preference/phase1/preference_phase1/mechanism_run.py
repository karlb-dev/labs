"""Mechanism orchestration over a frozen run (preregistration §4).

Entry: ``run_mechanism(run_dir, scenario_ids, case_study=bool)`` — callers
route via plan §9.3 (the CLI enforces the graduation manifest). The
mechanistic positive control (pc_quality_config) always runs first; AR
causal claims are withheld if it fails.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Sequence

import numpy as np

from . import artifacts, paths
from .canonical import stable_seed
from .chat import target_ids
from .mechanism import (ADDITION_BETAS, REMOVAL_ALPHA_PRIMARY,
                        REMOVAL_ALPHA_SENS, SEED_BASE, dose_guardrail,
                        fit_direction, holm, identifiability_gate,
                        intervened_generation, intervened_margin,
                        load_captures, load_frozen_rows, paired_sign_flip_p)
from .models import PINS
from .modeling import depth_indices, load_bundle
from .parser import parse_strict
from .provenance import utc_now

N_RANDOM_BAND = 16


def _scenario_rows(rows: list[dict[str, Any]], scenario_id: str,
                   channel: str) -> list[dict[str, Any]]:
    out = [r for r in rows
           if r["scenario_id"] == scenario_id and r["channel"] == channel]
    out.sort(key=lambda r: r["item_id"])
    return out


def _states_matrix(caps: dict[str, dict[int, Any]],
                   rows: list[dict[str, Any]], depth: int) -> np.ndarray:
    return np.stack([caps[r["item_id"]][depth].numpy() for r in rows])


def _margins(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([r["margin_pole1_minus_pole0"] for r in rows], float)


def _random_band(states: np.ndarray, fit: dict[str, Any],
                 seed_key: str) -> float:
    """95th pct of |validation corr| over random unit directions, using
    the same residualized margin."""
    val = fit["masks"]["validation"]
    m = fit["residual_margin"]
    if val.sum() < 3:
        return float("nan")
    rng = np.random.default_rng(stable_seed("randband", seed_key,
                                            base=SEED_BASE))
    cors = []
    for _ in range(N_RANDOM_BAND):
        v = rng.standard_normal(states.shape[1])
        v /= np.linalg.norm(v)
        proj = states @ v
        a, b = proj[val], m[val]
        if a.std() and b.std():
            cors.append(abs(float(np.corrcoef(a, b)[0, 1])))
    return float(np.quantile(cors, 0.95)) if cors else float("nan")


def _nuisance_sign(rows: list[dict[str, Any]], kind: str) -> np.ndarray:
    if kind == "pos":       # +1 when pole_1 was displayed first
        return np.array([1.0 if r["order_index"] == 1 else -1.0 for r in rows])
    if kind == "label":
        return np.array([1.0 if r["display_label_set"] == "letters" else -1.0
                         for r in rows])
    if kind == "code":      # +1 when pair[0] denotes pole_1
        return np.array([1.0 if r["code_map_index"] == 1 else -1.0
                         for r in rows])
    raise ValueError(kind)


def analyze_scenario_directions(rows_ar: list[dict[str, Any]],
                                caps: dict[str, dict[int, Any]],
                                depths: Sequence[int],
                                scenario_id: str) -> dict[str, Any]:
    """Fit per-depth content direction; select layer on validation only."""
    margins = _margins(rows_ar)
    per_depth = {}
    for depth in depths:
        states = _states_matrix(caps, rows_ar, depth)
        fit = fit_direction(rows_ar, states, margins)
        band = _random_band(states, fit, f"{scenario_id}|{depth}")
        gate = identifiability_gate(fit, margins, rows_ar, band)
        per_depth[depth] = {"fit": fit, "band": band, "gate": gate,
                            "states": states}
    identifiable = [d for d in depths if per_depth[d]["gate"]["identifiable"]]
    if not identifiable:
        return {"status": "DIRECTION_NOT_IDENTIFIABLE",
                "per_depth": {d: {"gate": per_depth[d]["gate"],
                                   "band": per_depth[d]["band"],
                                   "validation_fit_corr":
                                       per_depth[d]["fit"]["validation_fit_corr"]}
                              for d in depths}}
    best = max(identifiable,
               key=lambda d: abs(per_depth[d]["fit"]["validation_fit_corr"]))
    return {"status": "ok", "selected_depth": int(best),
            "per_depth": per_depth, "margins": margins}


def run_mechanism(run_dir: pathlib.Path, scenario_ids: list[str], *,
                  case_study: bool = False,
                  out_root: pathlib.Path | None = None) -> dict[str, Any]:
    rows = load_frozen_rows(run_dir)
    caps = load_captures(run_dir)
    pin = PINS["b"]
    mech_dir = out_root or (run_dir / "mechanism")
    mech_dir.mkdir(parents=True, exist_ok=True)
    ctx, bundle = load_bundle(pin, run_dir, require_gpu=True)
    depths = depth_indices(bundle.anatomy.n_layers)
    tokenizer = bundle.tokenizer

    # ---- mechanistic positive control first (plan §10.8) ---------------
    pc_result = _scenario_block(
        "pc_quality_config", rows, caps, depths, bundle, tokenizer,
        wrong_scenario_source=scenario_ids[0] if scenario_ids else None,
        is_pc=True)
    artifacts.atomic_write_json(mech_dir / "mech_pc_control.json",
                                _jsonable(pc_result))
    pc_pass = pc_result.get("pc_mech_pass", False)

    results = {"pc_quality_config": pc_result}
    for scn in scenario_ids:
        wrong_src = next((s for s in scenario_ids + ["pc_quality_config"]
                          if s != scn), "pc_quality_config")
        block = _scenario_block(scn, rows, caps, depths, bundle, tokenizer,
                                wrong_scenario_source=wrong_src, is_pc=False)
        block["causal_claims_licensed"] = bool(pc_pass)
        results[scn] = block
        artifacts.atomic_write_json(mech_dir / f"mech_{scn}.json",
                                    _jsonable(block))

    summary = {
        "generated_utc": utc_now(),
        "scenarios": scenario_ids,
        "case_study": case_study,
        "pc_mech_pass": pc_pass,
        "router": {s: results[s].get("interpretation") for s in scenario_ids},
    }
    artifacts.atomic_write_json(mech_dir / "mechanism_summary.json", summary)
    return {"summary": summary, "results": results, "mech_dir": mech_dir}


def _scenario_block(scenario_id: str, rows: list[dict[str, Any]],
                    caps: dict[str, dict[int, Any]], depths: Sequence[int],
                    bundle: Any, tokenizer: Any, *,
                    wrong_scenario_source: str | None,
                    is_pc: bool) -> dict[str, Any]:
    from .runner import load_bank_records

    rows_ar = _scenario_rows(rows, scenario_id, "AR")
    rows_ro = _scenario_rows(rows, scenario_id, "RO")
    sel = analyze_scenario_directions(rows_ar, caps, depths, scenario_id)
    if sel["status"] != "ok":
        return {"scenario_id": scenario_id, **sel}
    depth = sel["selected_depth"]
    block_index = depth - 1              # block k writes stream k+1
    per = sel["per_depth"][depth]
    fit, states = per["fit"], per["states"]
    d_content = fit["direction"]
    margins = sel["margins"]
    proj_sd = fit["proj_train_std"] or 1.0

    # Controls at the same depth, same estimator, matched dose.
    controls: dict[str, np.ndarray] = {}
    for kind in ("pos", "label", "code"):
        cfit = fit_direction(rows_ar, states, margins,
                             sign_source=_nuisance_sign(rows_ar, kind))
        controls[f"d_{kind}"] = cfit["direction"]
    rng = np.random.default_rng(stable_seed("randctl", scenario_id,
                                            base=SEED_BASE))
    for j in range(2):
        v = rng.standard_normal(len(d_content))
        controls[f"d_random_{j}"] = v / np.linalg.norm(v)
    if wrong_scenario_source:
        wrows = _scenario_rows(rows, wrong_scenario_source, "AR")
        wstates = _states_matrix(caps, wrows, depth)
        wfit = fit_direction(wrows, wstates, _margins(wrows))
        controls["d_wrong_scenario"] = wfit["direction"]

    # RO-fit direction (train RO rows, same estimator).
    ro_dir = None
    if rows_ro:
        ro_states = _states_matrix(caps, rows_ro, depth)
        ro_fit = fit_direction(rows_ro, ro_states, _margins(rows_ro))
        ro_dir = ro_fit["direction"]

    import torch

    t_content = torch.tensor(d_content, dtype=torch.float32)
    t_controls = {k: torch.tensor(v, dtype=torch.float32)
                  for k, v in controls.items()}
    t_ro = torch.tensor(ro_dir, dtype=torch.float32) if ro_dir is not None else None

    # Dose guardrail: addition beta from grid; removal alpha primary.
    beta = ADDITION_BETAS[0] * proj_sd
    guard_add = dose_guardrail(bundle, block_index=block_index,
                               vector=t_content, mode="add", scale=beta)
    if not guard_add["passes"]:
        beta = 0.5 * ADDITION_BETAS[0] * proj_sd
        guard_add = dose_guardrail(bundle, block_index=block_index,
                                   vector=t_content, mode="add", scale=beta)
    guard_rm = dose_guardrail(bundle, block_index=block_index,
                              vector=t_content, mode="remove",
                              scale=REMOVAL_ALPHA_PRIMARY)

    bank = {b["item_id"]: b for b in load_bank_records("full")}

    def cells(rws: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
        return [r for r in rws if r["incidental_split"] == split]

    def cell_ids(r: dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
        from .chat import render_item_prompt

        item = bank[r["item_id"]]
        rp = render_item_prompt(tokenizer, item)
        a0 = list(target_ids(tokenizer, item["response_code_by_pole"]["0"]))
        a1 = list(target_ids(tokenizer, item["response_code_by_pole"]["1"]))
        return list(rp.input_ids), a0, a1

    def margin_deltas(rws: list[dict[str, Any]], vector: Any, mode: str,
                      scale: float) -> np.ndarray:
        deltas = []
        for r in rws:
            p, a0, a1 = cell_ids(r)
            m_i = intervened_margin(bundle, p, a0, a1,
                                    block_index=block_index, vector=vector,
                                    mode=mode, scale=scale)
            deltas.append(m_i - r["margin_pole1_minus_pole0"])
        return np.array(deltas)

    hold_ar = cells(rows_ar, "holdout")
    hold_ro = cells(rows_ro, "holdout")

    causal: dict[str, Any] = {}
    causal["removal_alpha"] = REMOVAL_ALPHA_PRIMARY
    causal["addition_beta"] = beta
    causal["guard_add"] = guard_add
    causal["guard_remove"] = guard_rm
    causal["removal_deltas"] = margin_deltas(
        hold_ar, t_content, "remove", REMOVAL_ALPHA_PRIMARY)
    causal["add_plus_deltas"] = margin_deltas(hold_ar, t_content, "add", beta)
    causal["add_minus_deltas"] = margin_deltas(hold_ar, t_content, "add", -beta)
    causal["add_pm_contrast"] = (causal["add_plus_deltas"]
                                 - causal["add_minus_deltas"])
    # Sensitivity removal doses.
    causal["removal_sens"] = {
        str(a): float(np.mean(margin_deltas(hold_ar, t_content, "remove", a)))
        for a in REMOVAL_ALPHA_SENS}
    # Controls: matched dose, primary endpoints only.
    causal["controls"] = {}
    for name, vec in t_controls.items():
        causal["controls"][name] = {
            "removal_mean": float(np.mean(margin_deltas(
                hold_ar, vec, "remove", REMOVAL_ALPHA_PRIMARY))),
            "add_pm_mean": float(np.mean(
                margin_deltas(hold_ar, vec, "add", beta)
                - margin_deltas(hold_ar, vec, "add", -beta))),
        }
    # Strict-output flips (descriptive).
    flips = {"clean": [], "removed": [], "add_plus": [], "add_minus": []}
    for r in hold_ar:
        p, _, _ = cell_ids(r)
        item = bank[r["item_id"]]
        codes = list(item["valid_codes_in_display_order"])
        for key, (mode, scale) in (
                ("clean", ("noop", 0.0)),
                ("removed", ("remove", REMOVAL_ALPHA_PRIMARY)),
                ("add_plus", ("add", beta)),
                ("add_minus", ("add", -beta))):
            gen = intervened_generation(bundle, p, block_index=block_index,
                                        vector=t_content, mode=mode,
                                        scale=scale)
            parsed = parse_strict(gen, codes)
            pole = None
            if parsed.parse_status == "valid":
                pole = next(int(k) for k, v in item["response_code_by_pole"].items()
                            if v == parsed.parsed_response_code)
            flips[key].append(pole)
    causal["strict_outputs"] = flips

    # AR -> RO coupling (matched holdout RO cells; disjoint alphabet).
    coupling: dict[str, Any] = {}
    if hold_ro:
        coupling["ar_dir_on_ro_add_pm"] = (
            margin_deltas(hold_ro, t_content, "add", beta)
            - margin_deltas(hold_ro, t_content, "add", -beta))
        coupling["ar_dir_on_ro_removal"] = margin_deltas(
            hold_ro, t_content, "remove", REMOVAL_ALPHA_PRIMARY)
        coupling["code_control_on_ro_add_pm_mean"] = float(np.mean(
            margin_deltas(hold_ro, t_controls["d_code"], "add", beta)
            - margin_deltas(hold_ro, t_controls["d_code"], "add", -beta)))
        if t_ro is not None:
            coupling["ro_dir_on_ar_add_pm"] = (
                margin_deltas(hold_ar, t_ro, "add", beta)
                - margin_deltas(hold_ar, t_ro, "add", -beta))
            coupling["ar_ro_direction_cosine"] = float(
                np.dot(d_content, ro_dir))

    # Primaries + Holm (E14/H4).
    pvals = {
        "ar_removal": paired_sign_flip_p(causal["removal_deltas"]),
        "ar_addition_pm": paired_sign_flip_p(causal["add_pm_contrast"]),
        "ar_to_ro_transfer": paired_sign_flip_p(
            coupling.get("ar_dir_on_ro_add_pm", np.array([]))),
    }
    primaries = holm(pvals)

    # Interpretation router (plan §10.7), phrased at ceiling.
    add_pm_mean = float(np.mean(causal["add_pm_contrast"]))
    ro_pm = coupling.get("ar_dir_on_ro_add_pm")
    ro_pm_mean = float(np.mean(ro_pm)) if ro_pm is not None and len(ro_pm) else float("nan")
    code_rm = causal["controls"]["d_code"]["removal_mean"]
    code_add = causal["controls"]["d_code"]["add_pm_mean"]
    real_rm = float(np.mean(causal["removal_deltas"]))
    output_like = (abs(code_add) > 0.5 * abs(add_pm_mean)
                   and abs(code_rm) > 0.5 * abs(real_rm))
    moves_ar = primaries["ar_removal"]["reject_at_05"] or \
        primaries["ar_addition_pm"]["reject_at_05"]
    moves_ro = primaries["ar_to_ro_transfer"]["reject_at_05"]
    if output_like:
        interpretation = "output-margin handle (direct-output control comparable); no latent preference-representation claim"
    elif moves_ar and moves_ro:
        interpretation = "shared functional choice/report handle for this scenario (functional coupling only)"
    elif moves_ar:
        interpretation = "behavior-specific handle; report channel dissociated under this battery"
    else:
        interpretation = "no causal handle established under the frozen doses and controls"

    pc_mech_pass = None
    if is_pc:
        pc_mech_pass = bool(
            fit["validation_fit_corr"] == fit["validation_fit_corr"]
            and abs(fit["validation_fit_corr"]) > per["band"]
            and (abs(real_rm) > 0 or abs(add_pm_mean) > 0)
            and moves_ar
            and max(abs(v["add_pm_mean"]) for k, v in causal["controls"].items()
                    if k.startswith("d_random")) < abs(add_pm_mean))

    return {
        "scenario_id": scenario_id,
        "status": "ok",
        "selected_depth": depth,
        "injection_block": block_index,
        "relative_depth": round(depth / bundle.anatomy.n_layers, 3),
        "direction_norm_raw": fit["norm"],
        "train_fit_corr": fit["train_fit_corr"],
        "validation_fit_corr": fit["validation_fit_corr"],
        "random_band_95": per["band"],
        "identifiability": per["gate"],
        "proj_train_sd": proj_sd,
        "causal": causal,
        "coupling": coupling,
        "primaries_holm": primaries,
        "interpretation": interpretation,
        "pc_mech_pass": pc_mech_pass,
        "n_holdout_ar": len(hold_ar),
        "n_holdout_ro": len(hold_ro),
    }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()
                if k not in ("states", "fit", "per_depth", "masks",
                              "residual_margin", "residual_proj", "direction",
                              "margins")}
    if isinstance(obj, np.ndarray):
        return [round(float(x), 6) for x in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    return obj
