"""Lab 35: Reproducible interpretability paper capstone.

This lab is a package generator and validator, not a new mechanistic method.
It turns a frozen seed track into a preregistration, evidence matrix,
adversarial review, repair log, claim card, paper draft, reproduction guide,
plots, and package-validation diagnostics. Students replace the seed evidence
with their own frozen run, but the structure and review rubric are fixed.

Evidence level: AUDIT + FORMAL.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pathlib
import statistics
from collections import Counter
from typing import Any, Mapping, Sequence

import interp_bench as bench

LAB_ID = "L35"
DATA_FILE = "capstone_seed_tracks.jsonl"
PROMPT_SET_CAPS = {"small": 4, "medium": 4, "full": 0}

RUBRIC = [
    ("instrument_validity", 20, "Hook points, tokenization, dtype, cache parity, and artifact schemas are validated."),
    ("control_design", 20, "Primary controls attack the easiest alternative explanations."),
    ("evidence_rung_discipline", 20, "Allowed claims match OBS/ATTR/DECODE/CAUSAL/AUDIT/FORMAL evidence actually earned."),
    ("reproducibility", 15, "Frozen data, commands, seeds, environment, and artifact index are sufficient for rerun."),
    ("negative_result_handling", 10, "Counterexamples and failed controls are reported without being patched out."),
    ("writing_clarity", 10, "The paper distinguishes measurement, result, caveat, and claim."),
    ("safety_scope", 5, "Safety boundaries and blocked uses are explicit."),
]


@dataclasses.dataclass
class CapstoneTrack:
    track_id: str
    track_type: str
    title: str
    research_question: str
    source_lab: str
    dataset: str
    model: str
    measurement_sites: str
    primary_metric: str
    controls: list[str]
    stopping_rule: str
    allowed_claim: str
    forbidden_claim: str
    expected_failure_modes: list[str]
    safety_scope: str


def file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rounded(value: Any, digits: int = 4) -> Any:
    try:
        f = float(value)
    except Exception:
        return value
    return round(f, digits) if math.isfinite(f) else ""


def safe_mean(values: Sequence[Any], default: float = float("nan")) -> float:
    vals: list[float] = []
    for value in values:
        try:
            f = float(value)
        except Exception:
            continue
        if math.isfinite(f):
            vals.append(f)
    return float(statistics.fmean(vals)) if vals else default


def data_path(args: Any) -> pathlib.Path:
    prompt_set = str(getattr(args, "prompt_set", "") or "")
    candidate = pathlib.Path(prompt_set)
    if prompt_set not in PROMPT_SET_CAPS and candidate.suffix.lower() in {".jsonl", ".json"}:
        return candidate if candidate.is_absolute() else (bench.COURSE_ROOT / candidate).resolve()
    return bench.COURSE_ROOT / "data" / DATA_FILE


def load_tracks(ctx: bench.RunContext) -> tuple[list[CapstoneTrack], dict[str, Any]]:
    path = data_path(ctx.args)
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    tracks = [CapstoneTrack(**row) for row in rows]
    cap = PROMPT_SET_CAPS.get(str(ctx.args.prompt_set), 0)
    if cap:
        tracks = tracks[:cap]
    if int(ctx.args.max_examples or 0) > 0:
        tracks = tracks[: int(ctx.args.max_examples)]
    info = {
        "data_path": str(path),
        "sha256": file_sha256(path),
        "n_rows_file": len(rows),
        "n_rows_selected": len(tracks),
        "track_types": dict(Counter(t.track_type for t in tracks)),
        "source_labs": dict(Counter(t.source_lab for t in tracks)),
        "science_ready": False,
        "science_scope": "capstone package scaffold; student must bind a real frozen run before paper claims",
    }
    return tracks, info


def selected_track(tracks: Sequence[CapstoneTrack]) -> CapstoneTrack:
    for track in tracks:
        if track.track_id == "new_scoped_tool_surface_cues":
            return track
    return tracks[0]


def table_tracks(tracks: Sequence[CapstoneTrack]) -> list[dict[str, Any]]:
    return [
        {
            "track_id": t.track_id,
            "track_type": t.track_type,
            "title": t.title,
            "source_lab": t.source_lab,
            "dataset": t.dataset,
            "primary_metric": t.primary_metric,
            "n_controls": len(t.controls),
            "n_expected_failure_modes": len(t.expected_failure_modes),
            "safety_scope": t.safety_scope,
        }
        for t in tracks
    ]


def artifact_checklist(track: CapstoneTrack) -> list[dict[str, Any]]:
    required = [
        ("preregistration.md", "Preregistered question, controls, metric, stopping rule, and safety scope."),
        ("paper.md", "Short paper draft with result, controls, caveats, and claim boundary."),
        ("claim_card.md", "One allowed claim, forbidden claim, evidence rung, artifact, and falsifier."),
        ("adversarial_review.md", "Fixed rubric attack on instrumentation, leakage, confounds, language, power, and safety."),
        ("review_response.md", "Response to review with one repair run plan."),
        ("reproduction_guide.md", "Commands, environment, data, seed, and expected artifacts."),
        ("tables/evidence_matrix.csv", "Claim-by-artifact evidence matrix."),
        ("tables/review_rubric.csv", "Weighted capstone review rubric."),
        ("tables/repair_log.csv", "Frozen-run and repair-run accounting."),
        ("tables/failure_modes_contribution.csv", "Contribution to the living failure-mode atlas."),
    ]
    return [{"artifact": name, "description": desc, "status": "generated_from_seed_track", "track_id": track.track_id} for name, desc in required]


def review_rubric_rows() -> list[dict[str, Any]]:
    rows = []
    for key, weight, desc in RUBRIC:
        rows.append({
            "rubric_area": key,
            "weight_percent": weight,
            "review_question": desc,
            "seed_score": 0.75 if key in {"instrument_validity", "reproducibility", "safety_scope"} else 0.65,
            "student_score": "",
            "reviewer_notes": "",
        })
    return rows


def evidence_rows(track: CapstoneTrack) -> list[dict[str, Any]]:
    controls = "; ".join(track.controls)
    failures = "; ".join(track.expected_failure_modes)
    return [
        {
            "claim_component": "research_question_preregistered",
            "evidence_rung": "FORMAL",
            "artifact": "preregistration.md",
            "primary_check": track.research_question,
            "status": "present",
            "falsifier": "Question changed after seeing results without repair log.",
        },
        {
            "claim_component": "data_and_model_frozen",
            "evidence_rung": "AUDIT",
            "artifact": "reproduction_guide.md",
            "primary_check": f"dataset={track.dataset}; model={track.model}",
            "status": "seeded_not_final",
            "falsifier": "Data, model, or seed cannot be recovered.",
        },
        {
            "claim_component": "primary_metric_declared",
            "evidence_rung": "FORMAL",
            "artifact": "preregistration.md",
            "primary_check": track.primary_metric,
            "status": "present",
            "falsifier": "Paper switches to a secondary metric without disclosure.",
        },
        {
            "claim_component": "controls_attack_shortcuts",
            "evidence_rung": "AUDIT",
            "artifact": "tables/evidence_matrix.csv",
            "primary_check": controls,
            "status": "present",
            "falsifier": "A cheaper explanation is untested.",
        },
        {
            "claim_component": "expected_failure_modes_reported",
            "evidence_rung": "AUDIT",
            "artifact": "tables/failure_modes_contribution.csv",
            "primary_check": failures,
            "status": "present",
            "falsifier": "Counterexamples or negative controls are omitted.",
        },
        {
            "claim_component": "claim_language_bounded",
            "evidence_rung": "AUDIT",
            "artifact": "claim_card.md",
            "primary_check": f"allowed={track.allowed_claim}; forbidden={track.forbidden_claim}",
            "status": "present",
            "falsifier": "Final prose implies the forbidden claim.",
        },
    ]


def repair_log(track: CapstoneTrack) -> list[dict[str, Any]]:
    return [
        {
            "phase": "frozen_run",
            "allowed_action": "run preregistered command once",
            "status": "template_pending_student_run",
            "notes": "Do not overwrite the original frozen run.",
        },
        {
            "phase": "adversarial_review",
            "allowed_action": "review attacks instrumentation, leakage, confounds, language, power, and safety",
            "status": "generated_seed_review",
            "notes": "Replace or extend with human/AI review using the fixed rubric.",
        },
        {
            "phase": "repair_run",
            "allowed_action": "one repair run for an instrumentation bug or missing control",
            "status": "not_run",
            "notes": track.stopping_rule,
        },
    ]


def failure_mode_rows(track: CapstoneTrack) -> list[dict[str, Any]]:
    rows = []
    for i, mode in enumerate(track.expected_failure_modes, start=1):
        rows.append({
            "failure_mode_id": f"{track.track_id}-FM{i}",
            "source_lab": track.source_lab,
            "failure_mode": mode,
            "how_to_trigger": "Run the named control or inspect the listed counterexample family.",
            "how_to_report": "State whether it killed the claim, narrowed the scope, or required repair.",
            "claim_boundary": track.forbidden_claim,
        })
    return rows


def package_validation(track: CapstoneTrack, rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    required_tables = ["tracks", "artifact_checklist", "review_rubric", "evidence", "repair_log", "failure_modes"]
    counts = {name: len(rows[name]) for name in required_tables}
    return {
        "track_id": track.track_id,
        "required_tables_present": all(counts[name] > 0 for name in required_tables),
        "rubric_weight_total": sum(weight for _, weight, _ in RUBRIC),
        "has_forbidden_claim": bool(track.forbidden_claim),
        "has_stopping_rule": bool(track.stopping_rule),
        "has_failure_modes": bool(track.expected_failure_modes),
        "science_ready": False,
        "package_ready_for_student_replacement": True,
        "counts": counts,
    }


def write_preregistration(ctx: bench.RunContext, track: CapstoneTrack) -> None:
    lines = [
        "# Capstone preregistration",
        "",
        f"Track: `{track.track_id}`",
        f"Title: {track.title}",
        "",
        "## Research question",
        "",
        track.research_question,
        "",
        "## Allowed claim",
        "",
        track.allowed_claim,
        "",
        "## Forbidden claim",
        "",
        track.forbidden_claim,
        "",
        "## Dataset and model",
        "",
        f"- dataset: `{track.dataset}`",
        f"- model: {track.model}",
        f"- measurement sites: {track.measurement_sites}",
        "",
        "## Primary metric",
        "",
        track.primary_metric,
        "",
        "## Controls",
        "",
    ]
    lines += [f"- {c}" for c in track.controls]
    lines += [
        "",
        "## Stopping rule",
        "",
        track.stopping_rule,
        "",
        "## Expected failure modes",
        "",
    ]
    lines += [f"- {m}" for m in track.expected_failure_modes]
    lines += ["", "## Safety statement", "", track.safety_scope, ""]
    path = ctx.path("preregistration.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Capstone preregistration template.")


def write_paper(ctx: bench.RunContext, track: CapstoneTrack) -> None:
    lines = [
        "# Capstone paper draft",
        "",
        "## Abstract",
        "",
        f"This package studies: {track.research_question} The seed package is a scaffold; replace its placeholders with a frozen run before making a final claim.",
        "",
        "## Methods",
        "",
        f"We use `{track.dataset}` with `{track.model}` and measure {track.measurement_sites}.",
        "",
        "## Primary metric",
        "",
        track.primary_metric,
        "",
        "## Controls",
        "",
    ]
    lines += [f"- {c}" for c in track.controls]
    lines += [
        "",
        "## Results",
        "",
        "Insert the frozen-run result table here. Do not delete failed controls.",
        "",
        "## Counterexamples and negative results",
        "",
    ]
    lines += [f"- {m}" for m in track.expected_failure_modes]
    lines += [
        "",
        "## Claim",
        "",
        track.allowed_claim,
        "",
        "## Caveat",
        "",
        f"The paper must not imply: {track.forbidden_claim}",
        "",
    ]
    path = ctx.path("paper.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Capstone paper draft.")


def write_claim_card(ctx: bench.RunContext, track: CapstoneTrack) -> None:
    text = "\n".join(
        [
            "# Claim card",
            "",
            f"- evidence rung: `AUDIT + FORMAL` plus the student's chosen method rung",
            f"- allowed claim: {track.allowed_claim}",
            f"- forbidden claim: {track.forbidden_claim}",
            f"- primary artifact: `tables/evidence_matrix.csv`",
            f"- falsifier: any listed control or failure mode explains the result better than the proposed mechanism",
            "",
        ]
    )
    path = ctx.path("claim_card.md")
    bench.write_text(path, text)
    ctx.register_artifact(path, "summary", "Capstone claim card.")


def write_adversarial_review(ctx: bench.RunContext, track: CapstoneTrack) -> None:
    lines = [
        "# Adversarial review",
        "",
        "Use this fixed review structure so review quality is reproducible.",
        "",
        "## Instrumentation",
        "",
        "Which hook, cache, tokenization, dtype, or artifact-schema failure would invalidate the result?",
        "",
        "## Data leakage",
        "",
        f"Could `{track.dataset}` leak the label, target, or answer through prompt wording or ordering?",
        "",
        "## Confounds",
        "",
    ]
    lines += [f"- Attack control: {c}" for c in track.controls]
    lines += [
        "",
        "## Interpretation language",
        "",
        f"Does the draft imply the forbidden claim: {track.forbidden_claim}",
        "",
        "## Statistical power",
        "",
        "Is the sample size large enough for the claimed heterogeneity and control comparisons?",
        "",
        "## Safety",
        "",
        track.safety_scope,
        "",
        "## Required response",
        "",
        "The author must either narrow the claim, add one repair run, or mark the result unsupported.",
        "",
    ]
    path = ctx.path("adversarial_review.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Fixed adversarial review template.")


def write_review_response(ctx: bench.RunContext, track: CapstoneTrack) -> None:
    text = "\n".join(
        [
            "# Review response",
            "",
            "## Summary of changes",
            "",
            "Replace this section after review. The original frozen run must stay in the package.",
            "",
            "## Repair run decision",
            "",
            f"Stopping rule: {track.stopping_rule}",
            "",
            "## Claim revision",
            "",
            f"Allowed claim remains: {track.allowed_claim}",
            "",
            "## Unresolved risks",
            "",
        ]
        + [f"- {m}" for m in track.expected_failure_modes]
        + [""]
    )
    path = ctx.path("review_response.md")
    bench.write_text(path, text)
    ctx.register_artifact(path, "summary", "Capstone review response template.")


def write_reproduction_guide(ctx: bench.RunContext, track: CapstoneTrack) -> None:
    text = "\n".join(
        [
            "# Reproduction guide",
            "",
            "## Environment",
            "",
            "```bash",
            "cd interpretability",
            "pip install -r requirements.txt",
            "```",
            "",
            "## Frozen run",
            "",
            "Replace `labXX` with the source lab and keep the run directory immutable.",
            "",
            "```bash",
            f"python interp_bench.py --lab {track.source_lab} --tier a --prompt-set small",
            "```",
            "",
            "## Data",
            "",
            f"- dataset: `{track.dataset}`",
            f"- seed track: `{track.track_id}`",
            "",
            "## Expected package artifacts",
            "",
            "- `paper.md`",
            "- `claim_card.md`",
            "- `tables/evidence_matrix.csv`",
            "- `adversarial_review.md`",
            "- `review_response.md`",
            "- `plots/` dashboard, heterogeneity, controls, and failure-case plots",
            "",
        ]
    )
    path = ctx.path("reproduction_guide.md")
    bench.write_text(path, text)
    ctx.register_artifact(path, "summary", "Capstone reproduction guide.")


def write_tables(ctx: bench.RunContext, rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    specs = [
        ("tables/track_options.csv", rows["tracks"], "Available capstone seed tracks."),
        ("tables/artifact_checklist.csv", rows["artifact_checklist"], "Required package artifact checklist."),
        ("tables/review_rubric.csv", rows["review_rubric"], "Weighted adversarial review rubric."),
        ("tables/evidence_matrix.csv", rows["evidence"], "Capstone evidence matrix."),
        ("tables/repair_log.csv", rows["repair_log"], "Frozen-run and repair-run accounting."),
        ("tables/failure_modes_contribution.csv", rows["failure_modes"], "Failure-mode atlas contribution."),
    ]
    for rel, table_rows, desc in specs:
        path = ctx.path(*rel.split("/"))
        bench.write_csv_with_context(ctx, path, table_rows)
        ctx.register_artifact(path, "table", desc)


def write_method_card(ctx: bench.RunContext, track: CapstoneTrack, validation: Mapping[str, Any]) -> None:
    lines = [
        "# Lab 35 method card",
        "",
        "Lab 35 is a reproducible package generator and validator.",
        "",
        f"- selected track: `{track.track_id}`",
        f"- track type: `{track.track_type}`",
        f"- source lab: `{track.source_lab}`",
        f"- science_ready: `{validation['science_ready']}`",
        f"- package_ready_for_student_replacement: `{validation['package_ready_for_student_replacement']}`",
        "- forbidden claim discipline is part of the graded output",
        "",
    ]
    path = ctx.path("method_card.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Lab 35 method card.")


def write_operationalization_audit(ctx: bench.RunContext, track: CapstoneTrack) -> None:
    lines = [
        "# Lab 35 operationalization audit",
        "",
        "Favorite interpretation under attack: a clean-looking paper is reproducible and well-controlled.",
        "",
        "## What the package can say",
        "",
        "The preregistration, artifact checklist, review rubric, repair log, evidence matrix, and claim card exist and can be audited.",
        "",
        "## What it cannot say",
        "",
        "It cannot certify a scientific result until the student binds a real frozen run and survives review.",
        "",
        "## Main failure modes",
        "",
    ]
    lines += [f"- {m}" for m in track.expected_failure_modes]
    lines += ["", "## Forbidden claim", "", track.forbidden_claim, ""]
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Lab 35 operationalization audit.")


def write_run_summary(ctx: bench.RunContext, data_info: Mapping[str, Any], track: CapstoneTrack, validation: Mapping[str, Any]) -> None:
    lines = [
        "# Lab 35 run summary: reproducible interpretability paper capstone",
        "",
        f"- seed tracks selected: {data_info['n_rows_selected']} from `{pathlib.Path(str(data_info['data_path'])).name}`",
        f"- selected track: `{track.track_id}`",
        f"- source lab: `{track.source_lab}`",
        f"- dataset: `{track.dataset}`",
        f"- rubric weight total: `{validation['rubric_weight_total']}`",
        f"- package ready for student replacement: `{validation['package_ready_for_student_replacement']}`",
        f"- science_ready: `{validation['science_ready']}`",
        "",
        "## Required reading order",
        "",
        "1. `preregistration.md`",
        "2. `tables/evidence_matrix.csv`",
        "3. `adversarial_review.md`",
        "4. `review_response.md`",
        "5. `paper.md`",
        "6. `reproduction_guide.md`",
        "",
        "## Smallest surviving claim",
        "",
        "The capstone package structure is reproducible and reviewable. It is not yet a scientific result until a real frozen run is attached.",
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Lab 35 run summary and reading order.")


def write_plot_guide(ctx: bench.RunContext) -> None:
    rows = [
        {"plot": "plots/capstone_dashboard.png", "read_for": "Package readiness, rubric weights, and table counts.", "non_claim": "Readiness is not result validity."},
        {"plot": "plots/evidence_rung_matrix.png", "read_for": "Evidence components by rung.", "non_claim": "Rows are scaffold checks until real run is bound."},
        {"plot": "plots/review_score_radar.png", "read_for": "Rubric seed scores by area.", "non_claim": "Seed scores are placeholders."},
        {"plot": "plots/control_power_curve.png", "read_for": "Control count and expected failure-mode pressure.", "non_claim": "Counts are not statistical power."},
        {"plot": "plots/failure_mode_atlas.png", "read_for": "Failure modes contributed by the track.", "non_claim": "Atlas rows require concrete examples in final paper."},
    ]
    path = ctx.path("plots", "plot_reading_guide.csv")
    bench.write_csv(path, rows)
    ctx.register_artifact(path, "table", "Plot reading guide for Lab 35.")


def write_plots(ctx: bench.RunContext, track: CapstoneTrack, rows: Mapping[str, Sequence[Mapping[str, Any]]], validation: Mapping[str, Any]) -> None:
    write_plot_guide(ctx)
    if ctx.args.no_plots:
        return
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Lab 35 capstone package dashboard", fontsize=14, fontweight="bold")
    counts = validation["counts"]
    axes[0, 0].bar(list(counts), list(counts.values()), color="#0072B2")
    axes[0, 0].set_xticks(range(len(counts)), list(counts), rotation=35, ha="right")
    axes[0, 0].set_title("Generated table rows")
    weights = [r["weight_percent"] for r in rows["review_rubric"]]
    areas = [r["rubric_area"] for r in rows["review_rubric"]]
    axes[0, 1].bar(areas, weights, color="#009E73")
    axes[0, 1].set_xticks(range(len(areas)), areas, rotation=35, ha="right")
    axes[0, 1].set_title("Rubric weights")
    axes[1, 0].bar(["controls", "failure_modes"], [len(track.controls), len(track.expected_failure_modes)], color="#D55E00")
    axes[1, 0].set_title("Control pressure")
    flags = ["has_stopping_rule", "has_forbidden_claim", "has_failure_modes", "science_ready"]
    axes[1, 1].bar(flags, [1.0 if validation[f] else 0.0 for f in flags], color="#CC79A7")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_xticks(range(len(flags)), flags, rotation=35, ha="right")
    axes[1, 1].set_title("Package validation flags")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    bench.save_figure(ctx, fig, "capstone_dashboard.png", "Lab 35 capstone package dashboard.")

    rungs = sorted({r["evidence_rung"] for r in rows["evidence"]})
    comps = [r["claim_component"] for r in rows["evidence"]]
    mat = np.zeros((len(rungs), len(comps)))
    for j, row in enumerate(rows["evidence"]):
        mat[rungs.index(row["evidence_rung"]), j] = 1.0
    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(rungs)), rungs)
    ax.set_xticks(range(len(comps)), comps, rotation=35, ha="right")
    ax.set_title("Evidence rung matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "evidence_rung_matrix.png", "Capstone evidence-rung matrix.")

    scores = [float(r["seed_score"]) for r in rows["review_rubric"]]
    angles = np.linspace(0, 2 * np.pi, len(scores), endpoint=False).tolist()
    scores_closed = scores + scores[:1]
    angles_closed = angles + angles[:1]
    fig = plt.figure(figsize=(6.2, 6.2))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles_closed, scores_closed, color="#0072B2", linewidth=2)
    ax.fill(angles_closed, scores_closed, color="#0072B2", alpha=0.2)
    ax.set_xticks(angles, [r["rubric_area"] for r in rows["review_rubric"]], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Review score radar")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "review_score_radar.png", "Seed review-score radar.")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    xs = np.arange(1, len(track.controls) + 1)
    ys = [min(1.0, x / max(1, len(track.controls))) for x in xs]
    ax.plot(xs, ys, marker="o", color="#009E73")
    ax.set_xlabel("controls included")
    ax.set_ylabel("control coverage proxy")
    ax.set_title("Control power curve")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "control_power_curve.png", "Control coverage proxy curve.")

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.45 * len(track.expected_failure_modes) + 1.5)))
    ax.barh(track.expected_failure_modes, list(range(1, len(track.expected_failure_modes) + 1)), color="#D55E00")
    ax.set_xlabel("atlas row index")
    ax.set_title("Failure-mode atlas contribution")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "failure_mode_atlas.png", "Failure mode atlas contribution.")


def write_claims(ctx: bench.RunContext, track: CapstoneTrack, validation: Mapping[str, Any]) -> None:
    run_name = ctx.run_dir.name
    claims = [
        {
            "id": f"{LAB_ID}-C1",
            "tag": "AUDIT,FORMAL",
            "text": (
                f"Capstone seed track `{track.track_id}` generated preregistration, review, repair, evidence, "
                f"claim-card, reproduction, and failure-mode artifacts; package_ready={validation['package_ready_for_student_replacement']}."
            ),
            "artifact": f"runs/{run_name}/tables/evidence_matrix.csv",
            "falsifier": "Required artifacts are missing, rubric weights do not sum to 100, or the final paper implies the forbidden claim.",
        }
    ]
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    tracks, data_info = load_tracks(ctx)
    track = selected_track(tracks)
    manifest_path = ctx.path("diagnostics", "data_manifest.json")
    bench.write_json(manifest_path, data_info)
    ctx.register_artifact(manifest_path, "diagnostic", "Lab 35 seed-track data manifest.")
    bench.run_hook_parity_check(ctx, bundle, track.research_question)
    first = bench.run_with_residual_cache(bundle, track.research_question)
    bench.run_lens_self_check(ctx, bundle, first)
    bench.run_patch_noop_check(ctx, bundle, track.research_question)

    rows = {
        "tracks": table_tracks(tracks),
        "artifact_checklist": artifact_checklist(track),
        "review_rubric": review_rubric_rows(),
        "evidence": evidence_rows(track),
        "repair_log": repair_log(track),
        "failure_modes": failure_mode_rows(track),
    }
    validation = package_validation(track, rows)
    safety = {
        "lab": LAB_ID,
        "selected_track": track.track_id,
        "safety_scope": track.safety_scope,
        "blocked_claim": track.forbidden_claim,
        "science_ready": validation["science_ready"],
    }
    safety_path = ctx.path("diagnostics", "safety_status.json")
    bench.write_json(safety_path, safety)
    ctx.register_artifact(safety_path, "diagnostic", "Capstone safety and claim-boundary status.")
    validation_path = ctx.path("diagnostics", "package_validation.json")
    bench.write_json(validation_path, validation)
    ctx.register_artifact(validation_path, "diagnostic", "Capstone package validation checks.")

    write_preregistration(ctx, track)
    write_paper(ctx, track)
    write_claim_card(ctx, track)
    write_adversarial_review(ctx, track)
    write_review_response(ctx, track)
    write_reproduction_guide(ctx, track)
    write_tables(ctx, rows)
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, {"selected_track": dataclasses.asdict(track), "validation": validation, "data": data_info})
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 35 package metrics.")
    write_method_card(ctx, track, validation)
    write_operationalization_audit(ctx, track)
    write_run_summary(ctx, data_info, track, validation)
    write_claims(ctx, track, validation)
    write_plots(ctx, track, rows, validation)
    print(f"[lab35] generated capstone package for {track.track_id} with {len(rows['evidence'])} evidence rows")
