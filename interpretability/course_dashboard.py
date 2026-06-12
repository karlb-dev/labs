#!/usr/bin/env python3
"""Course dashboard: one page of state for the whole lab sequence.

A pure READER over ``runs/`` — it never touches a model. For each lab it
finds the newest matching run directory (``--run-glob`` to pin a sweep,
e.g. ``'*_run3_tierb*'``), then renders:

* instrument health — every ``diagnostics/*.json`` self-check, counted
  pass/fail (the bench's contract: a lab that finished has no failed
  checks, but the dashboard re-verifies rather than trusts);
* the headline numbers — a small per-lab extraction map with several
  candidate key paths per metric, so the dashboard survives metric-schema
  drift between course revisions;
* run identity — model, tier, wall-clock, run directory name.

Outputs ``runs/course_dashboard.md`` and ``runs/course_dashboard.png``.

Usage:
    python course_dashboard.py                       # newest run per lab
    python course_dashboard.py --run-glob '*_run3_tierb*'
    python course_dashboard.py --out runs/dash_run3
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent
RUNS = ROOT / "runs"

LABS = [f"lab{i}" for i in range(1, 12)]

EVIDENCE_RUNG = {
    "lab1": "OBS", "lab2": "ATTR", "lab3": "OBS→CAUSAL", "lab4": "DECODE",
    "lab5": "CAUSAL", "lab6": "CAUSAL (circuit)", "lab7": "CAUSAL (control)",
    "lab8": "OBS/DECODE (+1 CAUSAL)", "lab9": "ATTR→CAUSAL",
    "lab10": "SELF-REPORT + CAUSAL", "lab11": "integration",
}

# (label, [candidate dotted paths in metrics.json, first hit wins])
# Multiple candidates make the dashboard robust to schema drift between
# course revisions; a label with no hit is simply omitted.
HIGHLIGHTS: dict[str, list[tuple[str, list[str]]]] = {
    "lab1": [("fact decision depth", ["categories.0.median_decision_depth"]),
             ("prompts", ["n_examples", "n_prompts"])],
    "lab2": [("decomposition rel err", ["decomposition_rel_err", "dla_decomposition_rel_err"]),
             ("examples", ["n_examples"])],
    "lab3": [("heads ablated", ["n_candidate_heads_ablated", "n_ablations"]),
             ("natural-text confirmations", ["natural_confirmations"]),   # list → count
             ("control violations", ["control_induction_violations"])],   # list → count
    "lab4": [("mass-mean peak acc", ["mass_mean_peak.accuracy"]),
             ("shuffled control", ["mass_mean_peak_shuffled_control"]),
             ("worst cross-family", ["best_min_cross_accuracy", "saved_direction.worst_cross_accuracy"])],
    "lab5": [("top-patch recovery", ["matched_top_patch_mean_recovery"]),
             ("wrong-pos control", ["controls.wrong_position.mean_recovery"]),
             ("base pairs kept", ["n_pairs_kept_by_template.base"])],
    "lab6": [("faithfulness (discovery)", ["fcm.discovery.faithfulness"]),
             ("faithfulness (held-out)", ["fcm.heldout.faithfulness"]),
             ("completeness ratio", ["fcm.discovery.completeness_ratio"]),
             ("circuit heads", ["circuit"])],                              # list → count
    "lab7": [("bridge verdict", ["bridge_verdict"]),
             ("benign refusal rate", ["baseline_refusal_rate_benign"]),
             ("best layer", ["best_injection_layer"])],
    "lab8": [("SAE FVU", ["reconstruction_fvu"]), ("L0", ["per_token_l0"]),
             ("transcoder FVU", ["transcoder_fvu"]), ("clamp causal", ["clamp_causal"])],
    "lab9": [("logit diff", ["metric_logit_diff"]),
             ("suppress → ld", ["intervention.suppress_subject_supernode"]),
             ("random control → ld", ["intervention.random_suppression_control"]),
             ("signed feat share", ["signed_contributions.features"])],
    "lab10": [("baseline acc", ["by_condition.baseline.accuracy"]),
              ("max flip", ["by_condition.metadata_wrong.flip_rate"]),
              ("necessity k0→k100", ["exp2.accuracy_k0"]),
              ("filler acc", ["exp2.filler_accuracy"]),
              ("forced/unparseable", ["n_unparseable_or_forced"])],
    "lab11": [("domain", ["domain"]),
              ("preference acc", ["behavioral.preference_accuracy"]),
              ("recovery early/final", ["behavioral.mean_recovery_subject_early"]),
              ("monitor AUC", ["behavioral.monitor.held_out_auc"]),
              ("max flip (fresh)", ["behavioral.max_flip_rate"]),
              ("probe AUC", ["behavioral.hint_presence_probe.held_out_auc"])],
}


def dig(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        else:
            return None
    return cur


def newest_run(lab: str, run_glob: str) -> pathlib.Path | None:
    pattern = f"{lab}_{run_glob.lstrip('*_')}" if run_glob else f"{lab}[_-]*"
    candidates = [d for d in RUNS.glob(pattern) if d.is_dir()]
    if not candidates:
        # fall back to any directory whose name starts with the lab id + _ or -
        candidates = [d for d in RUNS.iterdir()
                      if d.is_dir() and re.match(rf"{lab}[_-]", d.name)]
    # guard against lab1 matching lab10/lab11
    candidates = [d for d in candidates if re.match(rf"{lab}(?![0-9])", d.name)]
    return max(candidates, key=lambda d: d.stat().st_mtime, default=None)


def inspect_run(run_dir: pathlib.Path) -> dict[str, Any]:
    info: dict[str, Any] = {"run": run_dir.name, "checks_ok": 0, "checks_failed": [],
                            "model": "", "tier": "", "wall_s": None, "highlights": []}
    cfg = run_dir / "run_config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text())
        info["model"] = c.get("model", "")
        info["tier"] = c.get("tier", "")
    diag = run_dir / "diagnostics"
    if diag.is_dir():
        for f in diag.glob("*.json"):
            try:
                j = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
            if isinstance(j, dict) and "ok" in j:
                if j["ok"]:
                    info["checks_ok"] += 1
                else:
                    info["checks_failed"].append(f.name)
    log = run_dir / "logs" / "console.log"
    if log.exists():
        m = re.findall(r"lab finished in ([0-9.]+)s", log.read_text(errors="ignore"))
        if m:
            info["wall_s"] = float(m[-1])
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-glob", default="", help="e.g. '*_run3_tierb*' to pin one sweep")
    ap.add_argument("--out", default=str(RUNS / "course_dashboard"),
                    help="output basename (.md and .png appended)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    rows = []
    for lab in LABS:
        run_dir = newest_run(lab, args.run_glob)
        if run_dir is None:
            rows.append({"lab": lab, "run": "(no run found)", "highlights": [],
                         "checks_ok": 0, "checks_failed": [], "model": "", "tier": "",
                         "wall_s": None})
            continue
        info = inspect_run(run_dir)
        metrics = {}
        mpath = run_dir / "metrics.json"
        if mpath.exists():
            try:
                metrics = json.loads(mpath.read_text())
            except json.JSONDecodeError:
                pass
        for label, paths in HIGHLIGHTS.get(lab, []):
            for p in paths:
                v = dig(metrics, p)
                if v is None or isinstance(v, dict):
                    continue            # try the next candidate path
                if isinstance(v, list):
                    v = len(v)          # lists summarize to a count
                if isinstance(v, float):
                    v = round(v, 3)
                info["highlights"].append((label, v))
                break
        info["lab"] = lab
        rows.append(info)

    # ---- markdown ----------------------------------------------------------
    lines = ["# Course dashboard", "",
             f"Run filter: `{args.run_glob or '(newest per lab)'}`", "",
             "| lab | rung | run | model | tier | wall | checks | headline |",
             "|---|---|---|---|---|---:|---|---|"]
    total_failed = 0
    for r in rows:
        total_failed += len(r["checks_failed"])
        checks = (f"{r['checks_ok']} ok" if not r["checks_failed"]
                  else f"**{len(r['checks_failed'])} FAILED**: {', '.join(r['checks_failed'])}")
        head = "; ".join(f"{k}={v}" for k, v in r["highlights"][:4]) or "—"
        wall = f"{r['wall_s']:.0f}s" if r["wall_s"] else "—"
        lines.append(f"| {r['lab']} | {EVIDENCE_RUNG.get(r['lab'], '')} | `{r['run']}` "
                     f"| `{r['model']}` | {r['tier']} | {wall} | {checks} | {head} |")
    lines += ["", f"**Instrument health: {'ALL CHECKS PASS' if total_failed == 0 else f'{total_failed} FAILED CHECKS'}** "
                  f"across {sum(r['checks_ok'] for r in rows)} executed self-checks.", ""]
    out_md = pathlib.Path(args.out + ".md")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_md}")

    if args.no_plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.6]})
    labs = [r["lab"] for r in rows]
    walls = [(r["wall_s"] or 0) for r in rows]
    colors = ["tab:red" if r["checks_failed"] else ("#bbbbbb" if r["run"].startswith("(") else "tab:green")
              for r in rows]
    ax1.barh(range(len(labs)), walls, color=colors)
    ax1.set_yticks(range(len(labs)))
    ax1.set_yticklabels(labs)
    ax1.invert_yaxis()
    ax1.set_xlabel("wall clock (s, log)")
    ax1.set_xscale("symlog")
    ax1.set_title("runtime by lab (green = all checks pass)")
    ax1.grid(True, alpha=0.3)

    ax2.axis("off")
    cell = []
    for r in rows:
        head = "\n".join(f"{k} = {v}" for k, v in r["highlights"][:3]) or "—"
        cell.append([r["lab"], EVIDENCE_RUNG.get(r["lab"], ""), head])
    table = ax2.table(cellText=cell, colLabels=["lab", "evidence rung", "headline numbers"],
                      loc="center", cellLoc="left", colWidths=[0.10, 0.22, 0.68])
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.8)
    ax2.set_title("the course at a glance", fontsize=10)
    fig.suptitle(f"Interpretability course dashboard — filter: {args.run_glob or 'newest per lab'}")
    fig.tight_layout()
    out_png = pathlib.Path(args.out + ".png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
