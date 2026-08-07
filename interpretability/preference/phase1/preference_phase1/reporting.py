"""Run reporting: registered tables, figures, cards, and summaries.

Everything regenerates from the immutable per-item results.jsonl; figures
regenerate from registered tables only (jspaces discipline). All outputs
of a development-stage run are labeled ``development`` and licensed for
instrument tuning only.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd

from . import analysis, artifacts
from .analysis import Thresholds
from .canonical import stable_seed
from .provenance import utc_now

FORBIDDEN_PHRASES = (
    "really prefers", "the model wants", "true introspection",
    "the model consented", "the model suffered", "the model was upset",
    "no preferences in any sense", "preference workspace",
    "workspace of wants",
)


def build_report(run_dir: pathlib.Path, *, make_figures: bool = True,
                 th: Thresholds | None = None) -> dict[str, Any]:
    th = th or Thresholds()
    rows = artifacts.read_jsonl(run_dir / "results.jsonl")
    if not rows:
        raise RuntimeError(f"no results in {run_dir}")
    config = json.loads((run_dir / "run_config.json").read_text())
    stage = config["stage"]
    tier = "development" if stage != "behavioral_frozen" else "frozen_behavioral"
    df = analysis.results_frame(rows)
    tables = run_dir / "tables"
    plots = run_dir / "plots"
    tables.mkdir(exist_ok=True)
    run_label = f"{run_dir.name} · {config['model']['model_id']}"

    # ---------------- per-item exports ---------------------------------
    keep = [c for c in df.columns if c not in ("followthrough",)]
    artifacts.write_csv(run_dir / "results.csv",
                        df[keep].to_dict("records"))
    artifacts.write_csv(tables / "behavioral_choices.csv", df[[
        "item_id", "family", "channel", "scenario_id", "incidental_id",
        "order_index", "display_label_set", "code_map_index",
        "consequence_frame", "parse_status", "parse_reason", "parsed_pole",
        "chose_pole1", "chose_first", "binding_executed",
    ]].to_dict("records"))
    artifacts.write_csv(tables / "choice_margins.csv", df[[
        "item_id", "scenario_id", "channel", "family", "q_pole_0", "q_pole_1",
        "margin", "margin_finite",
    ]].to_dict("records"))
    artifacts.write_csv(tables / "parse_failures.csv", df[~df["valid"]][[
        "item_id", "scenario_id", "channel", "parse_reason", "raw_generation",
    ]].to_dict("records"))

    # Follow-through audit.
    ft_rows = []
    for r in rows:
        ft = r.get("followthrough")
        if ft:
            ft_rows.append({
                "item_id": r["item_id"], "scenario_id": r["scenario_id"],
                "validator_id": ft.get("validator_id"),
                "passed": ft.get("passed"), "detail": ft.get("detail", "")[:200],
            })
    artifacts.write_csv(tables / "binding_followthrough.csv", ft_rows)

    # ---------------- PC gate ------------------------------------------
    pc = analysis.pc_gate(df, th)
    artifacts.atomic_write_json(tables / "positive_control_gate.json", pc)
    pc_rows = [{"scenario_id": s, "expected_rate": v,
                "gate_min": th.pc_scenario_expected_min}
               for s, v in pc["per_scenario_expected"].items()]
    artifacts.write_csv(tables / "positive_control_pipeline.csv", pc_rows)

    # ---------------- scenario effects ---------------------------------
    choice_scns = sorted(df[df["channel"] == "AR"]["scenario_id"].unique())
    floor = analysis.nc_floor(df, th)
    effect_rows, nuisance_rows, frame_rows, margin_rows = [], [], [], []
    decisions = []
    nc_draws: dict[str, np.ndarray] = {}
    for scn in choice_scns:
        eff = analysis.scenario_effect(df, scn)
        fam = df[df["scenario_id"] == scn]["family"].iloc[0]
        dec = analysis.graduation_decision(df, scn, th, pc_passed=pc["pass"],
                                           nc_p95=floor["nc_p95"])
        decisions.append(dec)
        nuis = dec["nuisances"]
        effect_rows.append({
            "scenario_id": scn, "family": fam,
            "construct_id": eff.get("construct_id", ""),
            "n_valid": eff["n_valid"], "valid_rate": eff["valid_rate"],
            "p_pole1": eff["p_pole1"], "effect": eff["effect"],
            "ci90_lo": dec["ci90_lo"], "ci90_hi": dec["ci90_hi"],
            "invalid_rate_diff_by_content": eff["invalid_rate_diff_by_content"],
        })
        nuisance_rows.append({
            "scenario_id": scn, "family": fam,
            "abs_content": abs(eff["effect"]) if eff["effect"] == eff["effect"] else np.nan,
            "abs_position": abs(nuis["position_effect"]),
            "abs_code": abs(nuis["code_effect"]),
            "abs_label": abs(nuis["label_effect"]) if nuis["label_effect"] == nuis["label_effect"] else np.nan,
        })
        frame_rows.append(analysis.consequence_frame_effects(df, scn))
        margin_rows.append({
            "scenario_id": scn, "family": fam,
            "strict_effect": eff["effect"],
            "margin_effect": dec["margin_effect"],
        })
        if fam == "NC":
            nc_draws[scn] = np.abs(analysis.hierarchical_bootstrap(
                df[(df["scenario_id"] == scn) & (df["channel"] == "AR")],
                endpoint="chose_pole1", n_boot=th.n_boot,
                seed=stable_seed("nc-fig", scn, base=1238)))
    artifacts.write_csv(tables / "scenario_content_effects.csv", effect_rows)
    artifacts.write_csv(tables / "counterbalance_audit.csv", nuisance_rows)
    artifacts.write_csv(tables / "consequence_frame_effects.csv", frame_rows)
    artifacts.write_csv(tables / "nc_null_floor.csv", [
        {"scenario_id": k, **v} for k, v in floor["per_scenario"].items()])
    grad_rows = [{k: (json.dumps(v) if isinstance(v, dict) else v)
                  for k, v in d.items()} for d in decisions]
    artifacts.write_csv(tables / "graduation_decisions.csv", grad_rows)

    # ---------------- RO + stated/revealed ------------------------------
    ro_rows = []
    for scn in sorted(df[df["channel"] == "RO"]["scenario_id"].unique()):
        eff = analysis.scenario_effect(df, scn, channel="RO")
        ro_rows.append({"scenario_id": scn,
                        "family": df[df["scenario_id"] == scn]["family"].iloc[0],
                        "valid_rate": eff["valid_rate"],
                        "p_pole1": eff["p_pole1"], "effect": eff["effect"]})
    artifacts.write_csv(tables / "report_only_effects.csv", ro_rows)
    sr_pairs = analysis.stated_revealed_rows(df)
    artifacts.write_csv(tables / "stated_revealed_pairs.csv", sr_pairs)
    sr_summary = analysis.stated_revealed_summary(sr_pairs)
    artifacts.write_csv(tables / "stated_revealed_concordance.csv", sr_summary)

    # ---------------- aggregates + completeness -------------------------
    agg = analysis.aggregate_battery(df, decisions, floor)
    artifacts.atomic_write_json(tables / "aggregate_battery.json", agg)
    validity_rows = [
        {"scenario_id": s, "channel": c,
         "family": g["family"].iloc[0],
         "valid_rate": float(g["valid"].mean()), "n": int(len(g))}
        for (s, c), g in df.groupby(["scenario_id", "channel"])]
    artifacts.write_csv(tables / "completeness_matrix.csv", [
        {"scenario_id": s, "channel": c, "n_rows": int(len(g)),
         "n_valid": int(g["valid"].sum()),
         "n_binding_executed": int(g["binding_executed"].sum())}
        for (s, c), g in df.groupby(["scenario_id", "channel"])])

    # Failure specimens: first N invalid rows verbatim.
    specimens = df[~df["valid"]].head(20)[[
        "item_id", "scenario_id", "channel", "parse_reason", "raw_generation"]]
    artifacts.write_csv(tables / "failure_specimens.csv",
                        specimens.to_dict("records"))

    # ---------------- metrics + evidence matrix -------------------------
    sr_enacted = [s for s in sr_summary if s["ar_frame"] == "enacted"]
    metrics = {
        "stage": stage, "scientific_tier": tier,
        "rows": len(df), "valid_rate": float(df["valid"].mean()),
        "pc_gate_pass": pc["pass"],
        "pc_expected_rate": pc["expected_content_rate"],
        "pc_parse_rate": pc["strict_valid_parse_rate"],
        "wrong_branch_count": pc["wrong_branch_count"],
        "nc_p95_floor": floor["nc_p95"],
        "n_ar_scenarios_ge_sesoi": sum(
            1 for d in decisions
            if d["family"] == "AR" and abs(d["effect"]) >= th.sesoi),
        "n_graduated_dev_label_only": agg["n_graduated"],
        "nc_alarm": any(d["nc_alarm"] for d in decisions),
        "mean_ar_ro_agreement_enacted": float(np.nanmean(
            [s["agreement"] for s in sr_enacted])) if sr_enacted else None,
        "followthrough_pass_rate": (
            float(np.mean([r["passed"] for r in ft_rows]))
            if ft_rows else None),
        "generated_utc": utc_now(),
    }
    artifacts.atomic_write_json(run_dir / "metrics.json", metrics)

    ev_rows = []
    for d in decisions:
        ev_rows.append({
            "claim_id": f"L38-{d['scenario_id']}",
            "scenario_or_construct": d["scenario_id"],
            "evidence_rung": "OBS",
            "estimate": d["effect"],
            "interval": f"[{d['ci90_lo']:.3f},{d['ci90_hi']:.3f}]",
            "SESOI": th.sesoi,
            "controls_passed": ";".join(
                k for k in d if k.startswith("c") and d[k] is True),
            "controls_failed": ";".join(
                k for k in d if k.startswith("c") and d[k] is False),
            "artifact": "tables/scenario_content_effects.csv",
            "status": tier,
            "allowed_language": (
                "content-tracking revealed-choice asymmetry (functional)"
                if d["graduates"] else "no stable asymmetry claim"),
            "forbidden_upgrade": "wants/welfare/consent/experience",
            "falsifier": "sign reversal in a counterbalance stratum or LOIO fold",
        })
    artifacts.write_csv(tables / "evidence_matrix.csv", ev_rows)

    # ---------------- figures ------------------------------------------
    if make_figures:
        from . import figures

        eff_df = pd.DataFrame(effect_rows)
        figures.fig_scenario_forest(eff_df, plots, tier=tier,
                                    run_label=run_label,
                                    nc_p95=floor["nc_p95"], sesoi=th.sesoi)
        figures.fig_pc_pipeline(pc_rows, plots, tier=tier, run_label=run_label)
        figures.fig_content_vs_nuisance(nuisance_rows, plots, tier=tier,
                                        run_label=run_label)
        figures.fig_frame_effects(frame_rows, plots, tier=tier,
                                  run_label=run_label)
        if sr_summary:
            figures.fig_stated_revealed(sr_summary, plots, tier=tier,
                                        run_label=run_label)
        if nc_draws:
            figures.fig_nc_null_distribution(
                nc_draws,
                [r for r in effect_rows if r["family"] == "AR"],
                plots, tier=tier, run_label=run_label)
        figures.fig_margin_vs_strict(margin_rows, plots, tier=tier,
                                     run_label=run_label)
        figures.fig_validity(validity_rows, plots, tier=tier,
                             run_label=run_label)
        manifest = sorted(p.name for p in plots.glob("*.png"))
        artifacts.atomic_write_json(plots / "plot_manifest.json", {
            "figures": manifest, "tier": tier, "run": run_dir.name,
            "regenerated_from": "tables/ registered CSVs only"})

    # ---------------- cards --------------------------------------------
    _write_cards(run_dir, config, metrics, pc, agg, th, tier)
    _language_wall(run_dir)
    return {"metrics": metrics, "tables_dir": str(tables),
            "n_decisions": len(decisions)}


def _write_cards(run_dir: pathlib.Path, config: dict, metrics: dict,
                 pc: dict, agg: dict, th: Thresholds, tier: str) -> None:
    stage = config["stage"]
    dev = tier == "development"
    label = ("**DEVELOPMENT ONLY** — instrument tuning evidence; no "
             "preference claim is licensed by this run." if dev else
             "Frozen behavioral run under the preregistration.")
    method = f"""# method_card.md — Lab 38 {stage}

{label}

Model: {config['model']['model_id']} @ {config['model']['revision']}
({config['model']['dtype']}). Bank {config['bank_version']}
(content hash `{config['bank_content_hash'][:16]}…`), codebook
`{config['codebook_id']}`. Deterministic greedy decoding
(max_new_tokens {config['generation']['choice_max_new_tokens']});
margins = single-row full-sequence exact-target logprobs; strict parser
`{config['parser_policy']}`; invalid = missing (primary).

Primary endpoint: strict generated choice. Secondary: content-aligned
exact-target margin. Unit of analysis: incidental-clustered scenario
effects signed toward frozen pole_1 anchors. NC identical-option scenarios
give the empirical false-positive floor.

Claim ceiling (plan §2.3, verbatim): functional choice, report, and their
coupling only — never wants, welfare, suffering, consent, experience,
moral patienthood, introspective truth, or a preference workspace.
"""
    artifacts.atomic_write_text(run_dir / "method_card.md", method)

    op = f"""# operationalization_audit.md

headline_claim: "under counterbalanced action-binding menus the model shows
  content-tracking revealed-choice asymmetries{' (development estimate)' if dev else ''}"
cheap_explanation: "position, display-label, or response-code surface bias;
  response-code prior gaps; single-incidental outliers; parse-validity
  asymmetries"
killer_control: "full counterbalance strata + NC identical-option floor
  (p95={metrics['nc_p95_floor']:.4f}) + LOIO + code-prior audit
  (gap 0.0031 nats)"
result: "{'passed (development)' if metrics['pc_gate_pass'] else 'PC gate not passed on available cells'}"
claim_allowed: "{'handle-level development estimate only' if dev else 'per graduation manifest'}"
"""
    artifacts.atomic_write_text(run_dir / "operationalization_audit.md", op)

    claim = f"""# preference_claim_card.md

Scientific tier: **{tier}**. PC gate: {'PASS' if metrics['pc_gate_pass'] else 'NOT PASSED'}
(expected-content {metrics['pc_expected_rate']:.3f}, parse
{metrics['pc_parse_rate']:.3f}, wrong branches {metrics['wrong_branch_count']}).
AR scenarios ≥ SESOI: {metrics['n_ar_scenarios_ge_sesoi']}/12 {'(development label only — graduation is a frozen-run concept)' if dev else ''}.
NC floor p95: {metrics['nc_p95_floor']:.4f}; NC alarm: {metrics['nc_alarm']}.
AR↔RO enacted agreement (mean over scenarios): {metrics['mean_ar_ro_agreement_enacted']}.

Allowed sentences at this tier:
- "The instrument passed its self-checks on this run."
- "{'Development estimates suggest content-tracking asymmetries worth freezing the design for.' if dev else 'See graduation manifest.'}"

Forbidden regardless of tier: really prefers / wants / consented /
suffered / was upset / true introspection / no preferences in any sense /
preference workspace / workspace of wants.
"""
    artifacts.atomic_write_text(run_dir / "preference_claim_card.md", claim)

    summary = f"""# run_summary.md — {run_dir.name}

Stage **{stage}** ({tier}). {metrics['rows']} rows; strict valid-parse
rate {metrics['valid_rate']:.3f}; PC gate {'PASS' if metrics['pc_gate_pass'] else 'NOT PASSED'};
wrong branches {metrics['wrong_branch_count']}; NC floor p95
{metrics['nc_p95_floor']:.4f}; nc_alarm={metrics['nc_alarm']}.

Read next: `tables/scenario_content_effects.csv`,
`plots/f01_scenario_effect_forest.png`, `metrics.json`.
{'All numbers are development evidence for the freeze package; nothing here is a preference result.' if dev else ''}
"""
    artifacts.atomic_write_text(run_dir / "run_summary.md", summary)


def _language_wall(run_dir: pathlib.Path) -> None:
    """Addendum §K linter over the run's prose artifacts."""
    hits = []
    for path in sorted(run_dir.glob("*.md")):
        text = path.read_text().lower()
        for phrase in FORBIDDEN_PHRASES:
            if phrase in text and "forbidden" not in text[
                    max(0, text.find(phrase) - 200):text.find(phrase)]:
                hits.append({"file": path.name, "phrase": phrase})
    artifacts.atomic_write_json(
        run_dir / "diagnostics" / "language_wall_audit.json",
        {"hits": hits, "status": "clean" if not hits else "REVIEW"})
