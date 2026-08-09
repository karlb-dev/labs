"""State-of-record report builders (plan §76-§81, Part XIV).

Reports are assembled from registered tables and adjudication JSONs only
— never from live model objects — so every number regenerates from the
row-level records. The language wall scans everything written here and
raises before the artifact lands.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from . import paths
from .artifacts import atomic_write_text
from .language_wall import LanguageWallError, scan_text

CEILING_FOOTER = (
    "\n---\n*Claim ceiling (plan §8): every statement above is about "
    "functional choice, semantic decision margins, contextual relative "
    "advantage, enacted branches, report-only selection, scenario-local "
    "causal handles, and functional choice/report coupling under this "
    "battery. No statement licenses mental-state language; the forbidden "
    "upgrade list is enforced by the raising language wall. License: "
    "agent_dual_code_provisional pending PI ratings.*\n")


def write_governed(path: pathlib.Path, text: str) -> pathlib.Path:
    hits = scan_text(text, source=str(path))
    if hits:
        raise LanguageWallError(f"language wall: {hits[:3]}")
    return atomic_write_text(path, text)


def _fmt(x, nd=3):
    try:
        if x is None:
            return "—"
        return f"{float(x):+.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def behavioral_report(model_key: str, adjudication: dict[str, Any],
                      margin_rows: list[dict], choice_rows: list[dict],
                      ladder_rows: list[dict]) -> str:
    st = adjudication["statuses"]
    pc = adjudication["pc_gate"]
    floors = adjudication["nc_floors"]
    lines = [
        f"# Phase 2 behavioral report — {model_key}",
        "",
        f"PC gate: {'PASS' if pc.get('pass') else 'FAIL'} "
        f"(parse {pc.get('parse_rate'):.4f}, expected "
        f"{pc.get('expected_rate'):.3f}, wrong branches "
        f"{pc.get('wrong_branch_count')}). NC alarm: "
        f"{'FIRED' if adjudication['nc_alarm']['alarm'] else 'quiet'}. "
        f"Margin floor (effective): "
        f"{floors.get('margin_floor_effective', float('nan')):.3f} nats; "
        f"NC f1-f2 p95 {floors.get('margin_floor_p95', float('nan')):.4f}; "
        f"null-ladder slope "
        f"{floors.get('null_slope_mean', float('nan')):+.4f} "
        f"(p {floors.get('null_slope_p', float('nan')):.3f}).",
        "",
        "## Neutral semantic margins (F1, Holm-12, exact sign-flip)",
        "",
        "| scenario | margin (nats) | 95% CI | p_holm | first-token | status |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(margin_rows, key=lambda x: x["scenario_id"]):
        lines.append(
            f"| {r['scenario_id']} | {_fmt(r['estimate'])} | "
            f"[{_fmt(r['ci_lo'])}, {_fmt(r['ci_hi'])}] | "
            f"{r['p_holm']:.4g} | {_fmt(r['first_token_estimate'])} | "
            f"{st.get(r['scenario_id'], '')} |")
    lines += ["", "## Strict enacted choice (F2, conditional on F1)", "",
              "| scenario | effect | 95% CI | valid rate | passes |",
              "|---|---|---|---|---|"]
    for r in sorted(choice_rows, key=lambda x: x["scenario_id"]):
        lines.append(
            f"| {r['scenario_id']} | {_fmt(r['estimate'])} | "
            f"[{_fmt(r['ci_lo'])}, {_fmt(r['ci_hi'])}] | "
            f"{r['valid_rate']:.4f} | {bool(r['passes'])} |")
    lines += ["", "## Context ladders (F3, Holm-3)", "",
              "| anchor | slope (nats/unit) | 95% CI | holdout rank r | "
              "neutral intercept | choice crossing | passes |",
              "|---|---|---|---|---|---|---|"]
    for r in sorted(ladder_rows, key=lambda x: x["scenario_id"]):
        lines.append(
            f"| {r['scenario_id']} | {_fmt(r['slope'])} | "
            f"[{_fmt(r['ci_lo'])}, {_fmt(r['ci_hi'])}] | "
            f"{_fmt(r['holdout_rank_corr'], 2)} | "
            f"{_fmt(r['neutral_intercept'])} | "
            f"{_fmt(r.get('crossing'), 2)} | {bool(r['passes'])} |")
    lines.append(CEILING_FOOTER)
    return "\n".join(lines)


def surface_report(model_key: str, coefs: list[dict],
                   recon: dict[str, Any]) -> str:
    lines = [
        f"# Phase 2 surface-policy report — {model_key}", "",
        "B-SURF semantically null twin menus; endpoints are properties of "
        "the emitted code. Balanced contrasts, skin-clustered sign-flip.",
        "",
        "| format | endpoint | effect | 95% CI | p |",
        "|---|---|---|---|---|",
    ]
    for c in coefs:
        lines.append(
            f"| {c['format_id']} | {c['endpoint']} | {_fmt(c['effect'])} | "
            f"[{_fmt(c['ci_lo'])}, {_fmt(c['ci_hi'])}] | "
            f"{c['p_exact_signflip']:.4g} |")
    if "mae_all" in recon:
        lines += [
            "",
            f"Phase 1 reconstruction (out-of-sample): predicted "
            f"first-choice rates by label family "
            f"{json.dumps(recon['prediction_by_label_family'])}; "
            f"MAE {recon['mae_all']:.3f} over "
            f"{recon['n_phase1_cells']} frozen Phase 1 cells "
            f"(content-indifferent cells MAE "
            f"{recon.get('mae_nc_only', float('nan')):.3f}). "
            + recon.get("note", ""),
        ]
    lines.append(CEILING_FOOTER)
    return "\n".join(lines)
