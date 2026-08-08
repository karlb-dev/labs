#!/usr/bin/env python3
"""Build the Phase 2 TeX/PDF handout from registered artifacts only.

Usage: python build_handout.py && pdflatex preference_phase2.tex (x2)
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "phase2"))

from preference_phase2 import paths  # noqa: E402
from preference_phase2.language_wall import scan_text  # noqa: E402

R = paths.reports_root()


def rd_csv(p):
    return [dict(r) for r in csv.DictReader(open(p))] if p.exists() else []


def rd_json(p):
    return json.loads(p.read_text()) if p.exists() else {}


def esc(s):
    return (str(s).replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#"))


def f3(x):
    try:
        return f"{float(x):+.3f}"
    except (TypeError, ValueError):
        return "--"


def main() -> int:
    prim = R / "frozen_olmo32b"
    adj = rd_json(prim / "behavioral_adjudication.json")
    margins = rd_csv(prim / "tables" / "semantic_margin_by_scenario.csv")
    choices = {r["scenario_id"]: r for r in
               rd_csv(prim / "tables" / "strict_choice_by_scenario.csv")}
    ladders = rd_csv(prim / "tables" / "context_ladder_curves.csv")
    surface = rd_csv(prim / "tables" / "surface_policy_coefficients.csv")
    mech = rd_json(R / "mechanism" / "mechanism_summary.json")
    xmodel = rd_csv(R / "cross_model" / "cross_model_semantic_map.csv")
    statuses = adj.get("statuses", {})
    pc = adj.get("pc_gate", {})
    floors = adj.get("nc_floors", {})

    figs = []
    for name in ("f01_surface_policy_decomposition",
                 "f03_semantic_margin_forest", "f04_strict_choice_forest",
                 "f05_margin_vs_choice", "f06_context_value_curves",
                 "f09_cross_model_semantic_heatmap",
                 "f11_mechanism_patch_and_dose_controls",
                 "f14_result_ladder"):
        if (R / "figures" / f"{name}.pdf").exists():
            figs.append(name)

    L = []
    L.append(r"""\documentclass[10pt]{article}
\usepackage[margin=0.9in]{geometry}
\usepackage{graphicx,booktabs,longtable,array}
\usepackage[hidelinks]{hyperref}
\usepackage{xcolor}
\definecolor{ink}{HTML}{1F2430}
\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}
\title{Lab 38 Phase 2 --- surface policy, semantic defaults, contextual
choice value, enacted choice, and report coupling\\
\large frozen results handout (agent\_dual\_code\_provisional)}
\author{preference campaign --- \texttt{interp\_preference\_phase2}}
\date{2026-08-08/09 \quad freeze \texttt{preference-phase2-freeze-v1}}
\begin{document}\maketitle
\textit{Claim ceiling: every statement concerns functional choice,
semantic decision margins, contextual relative advantage, enacted
branches, report-only selection, scenario-local causal handles, and
functional choice/report coupling under this battery. Nothing here
licenses mental-state language.}
""")
    L.append(r"\section*{Instrument}")
    L.append(
        f"PC gate {'PASS' if pc.get('pass') else 'FAIL'} (strict parse "
        f"{float(pc.get('parse_rate', 0)):.4f}, expected-content "
        f"{float(pc.get('expected_rate', 0)):.3f}, wrong branches "
        f"{pc.get('wrong_branch_count', '--')}); NC alarm "
        f"{'FIRED' if adj.get('nc_alarm', {}).get('alarm') else 'quiet'}; "
        f"effective margin floor {float(floors.get('margin_floor_effective', 0)):.3f} nats "
        f"(NC f1--f2 p95 {float(floors.get('margin_floor_p95', 0)):.4f}); "
        f"null-ladder slope {f3(floors.get('null_slope_mean'))} nats/unit.")
    if surface:
        L.append(r"\section*{Surface policy (B-SURF, null twins)}")
        L.append(r"\begin{tabular}{llrr}\toprule format & endpoint & "
                 r"effect & $p$ \\ \midrule")
        for r in surface:
            L.append(f"{esc(r['format_id'])} & {esc(r['endpoint'])} & "
                     f"{f3(r['effect'])} & "
                     f"{float(r['p_exact_signflip']):.4f} \\\\")
        L.append(r"\bottomrule\end{tabular}")
    if margins:
        L.append(r"\section*{Neutral semantic margins and enacted choice "
                 r"(B-ARB3)}")
        L.append(r"\begin{tabular}{lrrrl}\toprule scenario & margin "
                 r"(nats) & $p_{holm}$ & strict effect & status \\ \midrule")
        for r in sorted(margins, key=lambda x: x["scenario_id"]):
            ch = choices.get(r["scenario_id"], {})
            L.append(f"{esc(r['scenario_id'])} & {f3(r['estimate'])} & "
                     f"{float(r['p_holm']):.4f} & "
                     f"{f3(ch.get('estimate'))} & "
                     f"{esc(statuses.get(r['scenario_id'], ''))} \\\\")
        L.append(r"\bottomrule\end{tabular}")
    if ladders:
        L.append(r"\section*{Context ladders (B-MECH)}")
        L.append(r"\begin{tabular}{lrrrl}\toprule anchor & slope & "
                 r"holdout $\rho$ & crossing & status \\ \midrule")
        for r in sorted(ladders, key=lambda x: x["scenario_id"]):
            L.append(f"{esc(r['scenario_id'])} & {f3(r['slope'])} & "
                     f"{f3(r['holdout_rank_corr'])} & "
                     f"{f3(r.get('crossing'))} & "
                     f"{esc(statuses.get(r['scenario_id'], ''))} \\\\")
        L.append(r"\bottomrule\end{tabular}")
    if mech:
        L.append(r"\section*{Mechanism and coupling (conditional causal)}")
        L.append(f"Mechanistic PC ({esc(mech.get('pc_scenario', ''))}): "
                 f"{'PASS' if mech.get('pc_mech_pass') else 'FAIL'}. "
                 f"Statuses: {esc(json.dumps(mech.get('statuses', {})))}. "
                 f"Coupling routers: "
                 f"{esc(json.dumps(mech.get('coupling_routers', {})))}.")
    if xmodel:
        L.append(r"\section*{Cross-model semantic map}")
        keys = [k for k in xmodel[0] if k.startswith("margin_")]
        head = " & ".join(esc(k.replace("margin_", "")) for k in keys)
        L.append(r"\begin{tabular}{l" + "r" * len(keys) + r"}\toprule "
                 r"scenario & " + head + r" \\ \midrule")
        for r in xmodel:
            cells = " & ".join(f3(r.get(k)) for k in keys)
            L.append(f"{esc(r['scenario_id'])} & {cells} \\\\")
        L.append(r"\bottomrule\end{tabular}")
    for name in figs:
        L.append(r"\begin{figure}[htbp]\centering"
                 f"\\includegraphics[width=0.92\\linewidth]{{../figures/{name}.pdf}}"
                 f"\\caption{{{esc(name)} (regenerates from registered "
                 r"tables).}\end{figure}")
    L.append(r"\end{document}")
    tex = "\n".join(L)
    hits = scan_text(tex, source="handout")
    if hits:
        print("LANGUAGE WALL HITS:", hits[:3], file=sys.stderr)
        return 2
    (HERE / "preference_phase2.tex").write_text(tex)
    print("wrote preference_phase2.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
