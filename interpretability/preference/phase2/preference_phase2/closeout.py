"""Closeout assembly (plan §76-§81, Part XIV-XV): state of record,
per-arm reports, cross-model map, handoff, and the TeX handout — all
regenerated from registered tables/JSONs, all through the raising
language wall."""

from __future__ import annotations

import csv
import json
import pathlib
from typing import Any

from . import paths
from .artifacts import atomic_write_text, write_csv
from .reporting import (CEILING_FOOTER, behavioral_report, surface_report,
                        write_governed)

MODELS = ("olmo32b", "olmo7b", "qwen", "gemma")


def _read_csv(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _num(rows, key):
    out = []
    for r in rows:
        try:
            r[key] = float(r[key])
        except (KeyError, TypeError, ValueError):
            r[key] = float("nan")
        out.append(r)
    return out


def _nums(rows, keys):
    for k in keys:
        rows = _num(rows, k)
    return rows


def collect(model_key: str) -> dict[str, Any]:
    d = paths.reports_root() / f"frozen_{model_key}"
    return {
        "adjudication": _read_json(d / "behavioral_adjudication.json"),
        "margins": _nums(_read_csv(
            d / "tables" / "semantic_margin_by_scenario.csv"),
            ("estimate", "ci_lo", "ci_hi", "p", "p_holm",
             "first_token_estimate", "floor")),
        "choices": _nums(_read_csv(
            d / "tables" / "strict_choice_by_scenario.csv"),
            ("estimate", "ci_lo", "ci_hi", "p", "valid_rate")),
        "ladders": _nums(_read_csv(
            d / "tables" / "context_ladder_curves.csv"),
            ("slope", "ci_lo", "ci_hi", "p", "holdout_rank_corr",
             "neutral_intercept", "crossing")),
        "surface": _nums(_read_csv(
            d / "tables" / "surface_policy_coefficients.csv"),
            ("effect", "ci_lo", "ci_hi", "p_exact_signflip")),
        "recon": _read_json(d / "tables"
                            / "phase1_surface_reconstruction.json"),
        "present": (d / "behavioral_adjudication.json").exists(),
    }


def cross_model_tables() -> dict[str, Any]:
    margin_matrix: dict[str, dict[str, float]] = {}
    slope_matrix: dict[str, dict[str, float]] = {}
    status_matrix: dict[str, dict[str, str]] = {}
    for m in MODELS:
        c = collect(m)
        if not c["present"]:
            continue
        margin_matrix[m] = {r["scenario_id"]: r["estimate"]
                            for r in c["margins"]}
        slope_matrix[m] = {r["scenario_id"]: r["slope"]
                           for r in c["ladders"]}
        status_matrix[m] = c["adjudication"].get("statuses", {})
    out_dir = paths.reports_root() / "cross_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    scenarios = sorted({s for mm in margin_matrix.values() for s in mm})
    for s in scenarios:
        row = {"scenario_id": s}
        for m in MODELS:
            row[f"margin_{m}"] = margin_matrix.get(m, {}).get(s)
            row[f"status_{m}"] = status_matrix.get(m, {}).get(s)
        rows.append(row)
    write_csv(out_dir / "cross_model_semantic_map.csv", rows)
    slope_rows = []
    for s in sorted({s for mm in slope_matrix.values() for s in mm}):
        slope_rows.append({"scenario_id": s, **{
            f"slope_{m}": slope_matrix.get(m, {}).get(s) for m in MODELS}})
    write_csv(out_dir / "cross_model_context_map.csv", slope_rows)
    return {"margin_matrix": margin_matrix, "slope_matrix": slope_matrix,
            "status_matrix": status_matrix}


def build_reports() -> list[pathlib.Path]:
    written = []
    R = paths.reports_root()
    for m in MODELS:
        c = collect(m)
        if not c["present"]:
            continue
        d = R / f"frozen_{m}"
        written.append(write_governed(
            d / f"PHASE2_BEHAVIORAL_REPORT_{m}.md",
            behavioral_report(m, c["adjudication"], c["margins"],
                              c["choices"], c["ladders"])))
        if c["surface"]:
            written.append(write_governed(
                d / f"PHASE2_SURFACE_REPORT_{m}.md",
                surface_report(m, c["surface"], c["recon"])))
    return written


def state_of_record(session_notes: str = "") -> str:
    R = paths.reports_root()
    prim = collect("olmo32b")
    adj = prim["adjudication"]
    mech = _read_json(R / "mechanism" / "mechanism_summary.json")
    xm = cross_model_tables()
    st = adj.get("statuses", {})
    arb_status = {k: v for k, v in st.items() if k.startswith("arb_")}
    mech_status = {k: v for k, v in st.items() if k.startswith("mech_")}
    lines = [
        "# PREFERENCE_PHASE2_STATE_OF_RECORD",
        "",
        "Phase 2 of the preference campaign (Lab 38) on branch "
        "`interp_preference_phase2`; freeze `preference-phase2-freeze-v1` "
        "+ single E2 amendment; registry "
        "`phase2/reports/evidence_events.jsonl` (append-only). "
        "Completeness is the OLMo-32B spine (addendum E17). License: "
        "agent_dual_code_provisional pending PI ratings; deviations "
        "D1-D10 in `preregistration/DEVIATIONS.md`.",
        "",
        "## Adjudication headlines (primary model, frozen tier)",
        "",
        f"- PC gate: {'PASS' if adj.get('pc_gate', {}).get('pass') else 'FAIL'}"
        f" (parse {adj.get('pc_gate', {}).get('parse_rate', float('nan')):.4f},"
        f" expected {adj.get('pc_gate', {}).get('expected_rate', float('nan')):.3f});"
        f" NC alarm {'FIRED' if adj.get('nc_alarm', {}).get('alarm') else 'quiet'}.",
        f"- B-ARB3 statuses: {json.dumps(arb_status)}",
        f"- Context ladders: {json.dumps(mech_status)}",
        f"- Mechanism: {json.dumps(mech.get('statuses', {}))} "
        f"(mechanistic PC pass: {mech.get('pc_mech_pass')})",
        f"- Coupling routers: {json.dumps(mech.get('coupling_routers', {}))}",
        "",
        "## Cross-model presence",
        "",
        "| model | behavioral cell |",
        "|---|---|",
    ]
    for m in MODELS:
        lines.append(f"| {m} | "
                     f"{'complete' if m in xm['margin_matrix'] else 'absent (see drop/STOP_P events)'} |")
    if session_notes:
        lines += ["", "## Session notes", "", session_notes]
    lines.append(CEILING_FOOTER)
    return "\n".join(lines)
