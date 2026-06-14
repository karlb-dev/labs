"""Lab 35: Reproducible interpretability paper capstone.

This lab is a package generator and validator, not a new mechanistic method.
It turns a frozen seed track into a preregistration, evidence matrix,
adversarial review, repair log, claim card, paper draft, reproduction guide,
plots, and package-validation diagnostics. Students bind a real frozen source
run before making source-lab scientific claims.

Evidence level: AUDIT + FORMAL for the capstone package. The scientific rung is
inherited from the chosen source lab and is never upgraded by neat packaging.
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
MANIFEST_FILE = "capstone_MANIFEST.json"
PROMPT_SET_CAPS = {"small": 4, "medium": 6, "full": 0}

REQUIRED_FIELDS = {
    "track_id", "track_type", "route", "title", "source_lab", "evidence_rung_ceiling",
    "research_question", "dataset", "model", "measurement_sites", "primary_metric",
    "secondary_metrics", "controls", "falsifiers", "expected_failure_modes",
    "planned_artifacts", "planned_plots", "tier_a_command", "frozen_run_command",
    "stopping_rule", "allowed_claim", "forbidden_claim", "safety_scope",
    "human_review_required", "claim_ledger_template", "paper_outline",
}
LIST_FIELDS = {
    "secondary_metrics", "controls", "falsifiers", "expected_failure_modes",
    "planned_artifacts", "planned_plots", "paper_outline",
}
PACKAGE_ARTIFACTS = [
    "run_summary.md", "method_card.md", "preregistration.md", "paper.md",
    "claim_card.md", "adversarial_review.md", "review_response.md",
    "reproduction_guide.md", "operationalization_audit.md",
    "package_readiness_report.md", "negative_result_appendix.md",
    "public_release_checklist.md", "tables/evidence_matrix.csv",
    "tables/result_binding_template.csv", "tables/review_rubric.csv",
    "diagnostics/self_check_status.json", "diagnostics/safety_status.json",
    "plots/capstone_dashboard.png",
]
RUBRIC = [
    ("instrument_validity", 20, "Hook points, tokenization, dtype, cache parity, and artifact schemas are validated."),
    ("control_design", 20, "Controls attack the easiest alternative explanations before the preferred claim is stated."),
    ("evidence_rung_discipline", 20, "Allowed claims match the selected source lab's evidence rung."),
    ("reproducibility", 15, "Frozen data, command, seed, model, environment, and artifact index are sufficient for rerun."),
    ("negative_result_handling", 10, "Failed controls and counterexamples remain visible in the main package."),
    ("writing_clarity", 10, "The paper separates question, method, result, caveat, and claim."),
    ("safety_scope", 5, "Safety boundaries, blocked uses, and review requirements are explicit."),
]


@dataclasses.dataclass(frozen=True)
class CapstoneTrack:
    track_id: str
    track_type: str
    route: str
    title: str
    source_lab: str
    evidence_rung_ceiling: str
    research_question: str
    dataset: str
    model: str
    measurement_sites: str
    primary_metric: str
    secondary_metrics: list[str]
    controls: list[str]
    falsifiers: list[str]
    expected_failure_modes: list[str]
    planned_artifacts: list[str]
    planned_plots: list[str]
    tier_a_command: str
    frozen_run_command: str
    stopping_rule: str
    allowed_claim: str
    forbidden_claim: str
    safety_scope: str
    human_review_required: bool
    claim_ledger_template: str
    paper_outline: list[str]
    negative_result_sentence: str = "If a control explains the result, report the failed favorite hypothesis and the narrower supported audit claim."
    source_lab_title: str = ""
    notes: str = ""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CapstoneTrack":
        missing = sorted(REQUIRED_FIELDS - set(payload))
        if missing:
            raise ValueError(f"seed track {payload.get('track_id', '<unknown>')} missing fields: {missing}")
        data = dict(payload)
        for key in LIST_FIELDS:
            value = data.get(key, [])
            if isinstance(value, str):
                value = [part.strip() for part in value.split(";") if part.strip()]
            if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)) or not value:
                raise ValueError(f"seed track {data.get('track_id', '<unknown>')} field {key!r} must be a non-empty list")
            data[key] = [str(v) for v in value]
        data["human_review_required"] = bool(data["human_review_required"])
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


def file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def rounded(value: Any, digits: int = 4) -> Any:
    val = fnum(value)
    return round(val, digits) if math.isfinite(val) else ""


def safe_mean(values: Sequence[Any], default: float = float("nan")) -> float:
    vals = [fnum(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(statistics.fmean(vals)) if vals else default


def bullets(items: Sequence[str]) -> list[str]:
    return [f"- {item}" for item in items] if items else ["- none declared"]


def data_path(args: Any) -> pathlib.Path:
    prompt_set = str(getattr(args, "prompt_set", "") or "")
    candidate = pathlib.Path(prompt_set)
    if prompt_set not in PROMPT_SET_CAPS and candidate.suffix.lower() in {".jsonl", ".json"}:
        return candidate if candidate.is_absolute() else (bench.COURSE_ROOT / candidate).resolve()
    return bench.COURSE_ROOT / "data" / DATA_FILE


def manifest_expected_hash(path: pathlib.Path) -> tuple[str | None, str]:
    for mpath in (path.parent / MANIFEST_FILE, path.parent / "MANIFEST.json"):
        if not mpath.exists():
            continue
        try:
            manifest = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception as exc:
            return None, f"{mpath.name} unreadable: {exc}"
        entries: list[Any] = []
        if isinstance(manifest, dict):
            entries.extend([manifest.get(path.name), manifest.get(str(path))])
            if isinstance(manifest.get("files"), dict):
                entries.append(manifest["files"].get(path.name))
        for entry in entries:
            if isinstance(entry, str):
                return entry, f"found string entry in {mpath.name}"
            if isinstance(entry, dict):
                for key in ("sha256", "hash", "sha256_hex"):
                    if isinstance(entry.get(key), str):
                        return entry[key], f"found {key} in {mpath.name}"
    return None, f"no usable sha256 entry for {path.name}"


def load_tracks(ctx: bench.RunContext) -> tuple[list[CapstoneTrack], list[dict[str, Any]], dict[str, Any]]:
    path = data_path(ctx.args)
    if not path.exists():
        raise FileNotFoundError(f"Lab 35 seed-track file not found: {path}. Generate it with data/make_capstone_seed_tracks.py.")
    payloads: list[dict[str, Any]] = []
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        payloads = list(raw if isinstance(raw, list) else raw.get("tracks", []))
    else:
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if line.strip():
                    try:
                        payloads.append(json.loads(line))
                    except Exception as exc:
                        raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    if not payloads:
        raise ValueError(f"{path} contains no seed tracks")
    schema_rows: list[dict[str, Any]] = []
    all_tracks: list[CapstoneTrack] = []
    for i, payload in enumerate(payloads, start=1):
        missing = sorted(REQUIRED_FIELDS - set(payload))
        bad_lists = sorted(k for k in LIST_FIELDS if not isinstance(payload.get(k), list) or not payload.get(k))
        schema_rows.append({
            "row_index": i,
            "track_id": payload.get("track_id", ""),
            "schema_ok": not missing and not bad_lists,
            "missing_fields": ";".join(missing),
            "bad_list_fields": ";".join(bad_lists),
            "source_lab": payload.get("source_lab", ""),
            "evidence_rung_ceiling": payload.get("evidence_rung_ceiling", ""),
            "n_controls": len(payload.get("controls", [])) if isinstance(payload.get("controls"), list) else 0,
            "n_falsifiers": len(payload.get("falsifiers", [])) if isinstance(payload.get("falsifiers"), list) else 0,
            "n_failure_modes": len(payload.get("expected_failure_modes", [])) if isinstance(payload.get("expected_failure_modes"), list) else 0,
        })
        all_tracks.append(CapstoneTrack.from_payload(payload))
    cap = PROMPT_SET_CAPS.get(str(ctx.args.prompt_set), 0)
    selected = all_tracks[:cap] if cap else list(all_tracks)
    if int(ctx.args.max_examples or 0) > 0:
        selected = selected[: int(ctx.args.max_examples)]
    showcase = str(getattr(ctx.args, "showcase", "") or "")
    if showcase and not any(t.track_id == showcase for t in selected):
        for track in all_tracks:
            if track.track_id == showcase:
                selected.append(track)
                break
    actual = file_sha256(path)
    expected, note = manifest_expected_hash(path)
    info = {
        "data_file": DATA_FILE,
        "data_path": str(path),
        "data_sha256": actual,
        "manifest_expected_sha256": expected,
        "manifest_note": note,
        "manifest_ok": (actual == expected) if expected else None,
        "n_rows_file": len(all_tracks),
        "n_rows_selected": len(selected),
        "track_types": dict(Counter(t.track_type for t in selected)),
        "source_labs": dict(Counter(t.source_lab for t in selected)),
        "routes": dict(Counter(t.route for t in selected)),
        "science_ready": False,
        "science_scope": "capstone scaffold only; bind an immutable source run before scientific claims",
    }
    return selected, schema_rows, info


def select_track(args: Any, tracks: Sequence[CapstoneTrack]) -> tuple[CapstoneTrack, str]:
    # Lab 35 reuses the bench's generic --showcase flag as a friendly track picker.
    # The older --mode path remains as a fallback for consistency with other special-topic labs.
    showcase = str(getattr(args, "showcase", "") or "")
    if showcase:
        for track in tracks:
            if track.track_id == showcase:
                return track, f"selected by --showcase track_id {showcase}"
        for track in tracks:
            if track.track_type == showcase or track.route == showcase or track.source_lab == showcase:
                return track, f"selected first track matching --showcase {showcase}"
    mode = str(getattr(args, "mode", "") or "")
    if mode and mode != "lora":
        for track in tracks:
            if track.track_id == mode:
                return track, f"selected by --mode track_id {mode}"
        for track in tracks:
            if track.track_type == mode or track.route == mode:
                return track, f"selected first track matching --mode {mode}"
    for track in tracks:
        if track.route == "recommended":
            return track, "selected first recommended seed track"
    return tracks[0], "selected first seed track"


def track_options(tracks: Sequence[CapstoneTrack], selected: CapstoneTrack) -> list[dict[str, Any]]:
    return [{
        "track_id": t.track_id, "selected": t.track_id == selected.track_id,
        "track_type": t.track_type, "route": t.route, "title": t.title,
        "source_lab": t.source_lab, "evidence_rung_ceiling": t.evidence_rung_ceiling,
        "dataset": t.dataset, "primary_metric": t.primary_metric,
        "n_controls": len(t.controls), "n_falsifiers": len(t.falsifiers),
        "n_failure_modes": len(t.expected_failure_modes), "human_review_required": t.human_review_required,
    } for t in tracks]


def claim_language_audit(track: CapstoneTrack) -> list[dict[str, Any]]:
    return [
        {"sentence_type": "allowed_claim", "claim_language": track.allowed_claim, "status": "template_allowed_after_numbers_bound", "evidence_ceiling": track.evidence_rung_ceiling},
        {"sentence_type": "forbidden_claim", "claim_language": track.forbidden_claim, "status": "blocked", "evidence_ceiling": track.evidence_rung_ceiling},
        {"sentence_type": "negative_result_sentence", "claim_language": track.negative_result_sentence, "status": "allowed_if_controls_fail", "evidence_ceiling": "AUDIT"},
        {"sentence_type": "ledger_template", "claim_language": track.claim_ledger_template, "status": "edit_before_append_ledger", "evidence_ceiling": track.evidence_rung_ceiling},
    ]


def slug(text: Any) -> str:
    raw = str(text or "").strip().lower()
    out = []
    for ch in raw:
        out.append(ch if ch.isalnum() else "_")
    compact = "_".join(part for part in "".join(out).split("_") if part)
    return compact[:80] or "item"


def stable_row_id(prefix: str, *parts: Any) -> str:
    joined = "|".join(str(p) for p in parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def status_score(status: str) -> float:
    s = str(status or "").lower()
    if s in {"generated_by_lab35", "generated_seed_scaffold", "generated_seed_scaffold_pending_results", "not_required_but_available", "seeded"}:
        return 0.75
    if s in {"generated_unless_no_plots", "generated_review_questions_pending_answers"}:
        return 0.65
    if "pending" in s or "not_run" in s or "not_bound" in s:
        return 0.15
    if "blocked" in s:
        return 0.05
    if "ok" in s or "pass" in s or "valid" in s:
        return 1.0
    return 0.35


def bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def plot_guide() -> list[dict[str, Any]]:
    return [
        {"plot_id": "fig01", "plot": "capstone_dashboard.png", "source_table": "tables/figure_sources/capstone_dashboard_source.csv", "first_question": "Is the package complete enough to review, and what is still blocked?", "non_claim": "Completeness is not scientific validity.", "claim_supported": "scaffold readiness only"},
        {"plot_id": "fig02", "plot": "target_vs_control.png", "source_table": "tables/figure_sources/target_vs_control_source.csv", "first_question": "Is the target claim shown beside the controls and falsifiers that can kill it?", "non_claim": "More controls do not imply the source result passed them.", "claim_supported": "claim-pressure visibility"},
        {"plot_id": "fig03", "plot": "artifact_contract_status.png", "source_table": "tables/figure_sources/artifact_contract_status_source.csv", "first_question": "Which artifacts are generated by the scaffold and which remain pending source-run binding?", "non_claim": "Pending source-run artifacts cannot support final claims.", "claim_supported": "artifact coverage audit"},
        {"plot_id": "fig04", "plot": "evidence_rung_matrix.png", "source_table": "tables/figure_sources/evidence_rung_matrix_source.csv", "first_question": "Which claim components are FORMAL, AUDIT, or pending source evidence?", "non_claim": "A bright cell does not raise the source evidence ceiling.", "claim_supported": "evidence ceiling discipline"},
        {"plot_id": "fig05", "plot": "review_score_radar.png", "source_table": "tables/figure_sources/review_score_radar_source.csv", "first_question": "Where should the reviewer spend scrutiny first?", "non_claim": "Seed risk scores are triage, not review outcomes.", "claim_supported": "review triage"},
        {"plot_id": "fig06", "plot": "control_falsifier_map.png", "source_table": "tables/figure_sources/control_falsifier_map_source.csv", "first_question": "Which controls, falsifiers, and failure modes must stay visible?", "non_claim": "Control count is not control quality.", "claim_supported": "control visibility"},
        {"plot_id": "fig07", "plot": "failure_mode_atlas.png", "source_table": "tables/figure_sources/failure_mode_atlas_source.csv", "first_question": "Which negative-result paths must be reported if observed?", "non_claim": "The atlas needs concrete source-run examples before it supports science.", "claim_supported": "negative result readiness"},
        {"plot_id": "fig08", "plot": "reproduction_readiness_ladder.png", "source_table": "tables/figure_sources/reproduction_readiness_ladder_source.csv", "first_question": "Which reproducibility fields remain unbound?", "non_claim": "A scaffold cannot reproduce a source run by itself.", "claim_supported": "reproducibility gap audit"},
        {"plot_id": "fig09", "plot": "claim_risk_register.png", "source_table": "tables/figure_sources/claim_risk_register_source.csv", "first_question": "Which overclaim risks are explicitly caught by artifacts?", "non_claim": "A named risk is not a resolved risk.", "claim_supported": "risk visibility"},
        {"plot_id": "fig10", "plot": "binding_gap_matrix.png", "source_table": "tables/figure_sources/binding_gap_matrix_source.csv", "first_question": "Which exact source-run values are still missing before the paper can make a result claim?", "non_claim": "Empty binding cells mean no final source claim yet.", "claim_supported": "binding gap visibility"},
        {"plot_id": "fig11", "plot": "paired_examples.png", "source_table": "tables/figure_sources/paired_examples_source.csv", "first_question": "How do allowed, forbidden, and negative-result sentences differ?", "non_claim": "Sentence templates must be filled from source artifacts before publication.", "claim_supported": "claim-language discipline"},
    ]


def artifact_checklist(track: CapstoneTrack) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for a in PACKAGE_ARTIFACTS:
        artifact_id = stable_row_id("artifact", track.track_id, a)
        rows.append({
            "artifact_id": artifact_id,
            "artifact": a,
            "category": "capstone_package",
            "status": "generated_by_lab35" if not a.startswith("plots/") else "generated_unless_no_plots",
            "status_score": status_score("generated_by_lab35" if not a.startswith("plots/") else "generated_unless_no_plots"),
            "required_before_submission": True,
            "required_before_science_claim": True,
            "source_run_bound_required": False,
            "failure_if_missing": "package_incomplete",
            "student_action": "inspect and edit" if a.endswith(".md") else "inspect and bind into paper where cited",
        })
    for a in track.planned_artifacts:
        artifact_id = stable_row_id("artifact", track.track_id, "source", a)
        rows.append({
            "artifact_id": artifact_id,
            "artifact": a,
            "category": "source_run_artifact",
            "status": "pending_student_frozen_run_binding",
            "status_score": status_score("pending_student_frozen_run_binding"),
            "required_before_submission": True,
            "required_before_science_claim": True,
            "source_run_bound_required": True,
            "failure_if_missing": "source_evidence_gap",
            "student_action": "bind immutable source-run artifact path and hash",
        })
    return rows


def evidence_matrix(track: CapstoneTrack) -> list[dict[str, Any]]:
    base = [
        ("research_question_preregistered", "FORMAL", "preregistration.md", "generated_seed_scaffold", track.research_question, "paper question changes after results without repair log"),
        ("source_run_immutable", "AUDIT", "tables/result_binding_template.csv", "pending_student_frozen_run_binding", track.frozen_run_command, "original run is missing, overwritten, or replaced by repair run"),
        ("metric_declared", "FORMAL", "preregistration.md", "generated_seed_scaffold", track.primary_metric, "paper switches to a secondary metric without disclosure"),
        ("controls_attack_shortcuts", "AUDIT", "tables/control_falsifier_matrix.csv", "generated_seed_scaffold_pending_results", "; ".join(track.controls), "a cheaper explanation is untested or hidden"),
        ("failure_modes_visible", "AUDIT", "tables/failure_modes_contribution.csv", "generated_seed_scaffold_pending_results", "; ".join(track.expected_failure_modes), "counterexamples or negative controls are omitted"),
        ("claim_ceiling_respected", "AUDIT + FORMAL", "claim_card.md", "generated_seed_scaffold", track.evidence_rung_ceiling, track.forbidden_claim),
        ("human_review_accounted", "AUDIT", "tables/human_review_queue.csv", "pending_human_review" if track.human_review_required else "not_required_but_available", str(track.human_review_required), "publication claim made before required human review"),
        ("repair_run_limited", "FORMAL + AUDIT", "tables/repair_log.csv", "generated_seed_scaffold", track.stopping_rule, "repair run replaces frozen run instead of being compared to it"),
        ("paper_numbers_traceable", "AUDIT", "tables/claim_to_artifact_map.csv", "pending_student_frozen_run_binding", "every number in paper.md maps to a bound artifact row", "paper contains unbound numbers"),
        ("negative_result_path_preregistered", "AUDIT", "negative_result_appendix.md", "generated_seed_scaffold_pending_results", track.negative_result_sentence, "failed controls are moved to a footnote or deleted"),
    ]
    rows = []
    for i, (component, rung, artifact, status, check, falsifier) in enumerate(base, 1):
        rows.append({
            "claim_component_id": f"{track.track_id}-E{i:02d}",
            "claim_component": component,
            "evidence_rung": rung,
            "artifact": artifact,
            "status": status,
            "status_score": status_score(status),
            "source_run_bound_required": "source" in component or "paper_numbers" in component,
            "primary_check": check,
            "falsifier": falsifier,
            "claim_language_effect": "blocks final source claim" if "pending" in status else "supports scaffold claim",
        })
    return rows


def control_falsifier_matrix(track: CapstoneTrack) -> list[dict[str, Any]]:
    n = max(len(track.controls), len(track.falsifiers), len(track.expected_failure_modes))
    rows = []
    for i in range(n):
        control = track.controls[i] if i < len(track.controls) else ""
        falsifier = track.falsifiers[i] if i < len(track.falsifiers) else ""
        failure = track.expected_failure_modes[i] if i < len(track.expected_failure_modes) else ""
        rows.append({
            "control_falsifier_id": f"{track.track_id}-CF{i+1:02d}",
            "control": control,
            "falsifier": falsifier,
            "expected_failure_mode": failure,
            "source_artifact": track.planned_artifacts[i % len(track.planned_artifacts)] if track.planned_artifacts else "",
            "must_appear_in_paper": bool(control or falsifier or failure),
            "source_result_status": "pending_student_binding",
            "student_result": "",
            "claim_effect": "pending: supports | narrows | kills | unrelated",
            "reviewer_notes": "",
        })
    return rows


def review_rubric_rows() -> list[dict[str, Any]]:
    rows = []
    for i, (area, weight, question) in enumerate(RUBRIC, 1):
        # This is a triage prior based on how likely the area is to hide claim inflation.
        risk_prior = {
            "instrument_validity": 0.85,
            "control_design": 0.90,
            "evidence_rung_discipline": 0.95,
            "reproducibility": 0.80,
            "negative_result_handling": 0.90,
            "writing_clarity": 0.65,
            "safety_scope": 0.70,
        }.get(area, 0.60)
        rows.append({
            "rubric_id": f"R{i:02d}",
            "rubric_area": area,
            "weight_percent": weight,
            "risk_prior": risk_prior,
            "review_question": question,
            "student_score": "",
            "reviewer_notes": "",
        })
    return rows


def human_review_queue(track: CapstoneTrack) -> list[dict[str, Any]]:
    base = [
        ("claim_card", "Does the final claim stay below the evidence ceiling?", True),
        ("abstract", "Does the abstract imply the forbidden claim?", True),
        ("control_results", "Are failed controls reported in the main text?", True),
        ("negative_results", "Is the negative-result interpretation explicit?", True),
        ("repair_accounting", "Does any repair run preserve the original frozen run?", True),
        ("figure_captions", "Do captions separate visual patterns from supported claims?", True),
        ("human_label_fields", "Are required human review or labeling fields filled before citation?", track.human_review_required),
    ]
    rows = []
    for i, (item, question, always_required) in enumerate(base, 1):
        required = bool(always_required or track.human_review_required or item in {"claim_card", "repair_accounting"})
        rows.append({
            "review_item_id": f"{track.track_id}-HR{i:02d}",
            "review_item": item,
            "required": required,
            "question": question,
            "source_or_package_artifact": "claim_card.md" if item == "claim_card" else "paper.md",
            "student_answer": "",
            "reviewer_decision": "pending: approve | revise | reject",
            "reviewer_notes": "",
        })
    return rows


def repair_log(track: CapstoneTrack) -> list[dict[str, Any]]:
    phases = [
        ("preregistration", "generated_seed_scaffold", "write question, metric, controls, falsifiers, stopping rule, and claim ceiling before reading source-run results", track.research_question),
        ("frozen_run", "pending_student_run", "run the source-lab command once and keep the original directory immutable", track.frozen_run_command),
        ("adversarial_review", "generated_review_questions_pending_answers", "attack instrumentation, leakage, controls, language, power, and safety", "Review is evidence, not a formality."),
        ("repair_run", "not_run", "at most one repair run for named instrumentation or missing-control defect", track.stopping_rule),
        ("final_claim", "pending_student_binding", "write the smallest claim that survives review", track.allowed_claim),
    ]
    return [{
        "repair_phase_id": f"{track.track_id}-RP{i:02d}",
        "phase": phase,
        "status": status,
        "status_score": status_score(status),
        "allowed_action": action,
        "original_run_visible_required": phase in {"frozen_run", "repair_run", "final_claim"},
        "notes": notes,
    } for i, (phase, status, action, notes) in enumerate(phases, 1)]


def failure_mode_rows(track: CapstoneTrack) -> list[dict[str, Any]]:
    rows = []
    for i, mode in enumerate(track.expected_failure_modes, 1):
        rows.append({
            "failure_mode_id": f"{track.track_id}-FM{i:02d}",
            "source_lab": track.source_lab,
            "failure_mode": mode,
            "how_to_trigger": track.controls[(i - 1) % len(track.controls)] if track.controls else "bind source run and inspect controls",
            "how_to_report": "state whether it killed, narrowed, or required repair",
            "status": "pending_source_run_example",
            "student_observed_example": "",
            "paper_section": "negative_result_appendix.md",
        })
    return rows


def reproduction_checklist(track: CapstoneTrack, data_info: Mapping[str, Any]) -> list[dict[str, Any]]:
    items = [
        ("course_commit_or_archive", "Record git SHA, dirty state, or zip hash.", "pending_student_binding"),
        ("source_run_command", track.frozen_run_command, "pending_student_binding"),
        ("tier_a_smoke_command", track.tier_a_command, "seeded"),
        ("seed_track_data_hash", str(data_info.get("data_sha256", "")), "seeded"),
        ("source_dataset", track.dataset, "seeded"),
        ("model_and_revision", track.model, "pending_student_binding"),
        ("random_seed", "Record seed from source run_config.json.", "pending_student_binding"),
        ("artifact_index", "Attach artifact_index.json from source and capstone runs.", "pending_student_binding"),
        ("source_run_hash", "Hash immutable source run directory or release zip.", "pending_student_binding"),
        ("human_review", "Fill review queue if required." if track.human_review_required else "Document why human review is not required.", "pending_student_binding"),
    ]
    return [{
        "reproduction_check_id": f"{track.track_id}-RC{i:02d}",
        "check_id": key,
        "required": True,
        "status": status,
        "status_score": status_score(status),
        "evidence_needed": detail,
        "student_value": "",
        "reviewer_notes": "",
    } for i, (key, detail, status) in enumerate(items, 1)]


def result_binding_template(track: CapstoneTrack) -> list[dict[str, Any]]:
    fields = [
        ("frozen_run_dir", "Path to immutable source-lab run directory.", "path"),
        ("frozen_run_artifact_index", "Path to artifact_index.json.", "path"),
        ("frozen_run_command", track.frozen_run_command, "command"),
        ("source_lab", track.source_lab, "id"),
        ("model_id", track.model, "id"),
        ("model_revision", "Pinned model revision or run_config default.", "id"),
        ("data_path", track.dataset, "path"),
        ("data_sha256", "Dataset digest from source run manifest.", "hash"),
        ("seed", "Seed from source run_config.json.", "integer"),
        ("primary_metric_value", track.primary_metric, "number"),
        ("strongest_control_value", "Strongest control result backing specificity.", "number"),
        ("n_examples", "Number of source-run examples supporting the claim.", "integer"),
        ("failed_controls", "Controls that failed, matched, or narrowed the claim.", "text"),
        ("repair_run_dir", "Only if repair run was used.", "path"),
        ("release_package_hash", "Hash of final zipped package.", "hash"),
    ]
    rows = []
    for i, (field, desc, kind) in enumerate(fields, 1):
        seeded = field in {"source_lab", "model_id", "data_path", "frozen_run_command"}
        rows.append({
            "binding_field_id": f"{track.track_id}-BF{i:02d}",
            "field": field,
            "value_kind": kind,
            "description": desc,
            "student_value": "",
            "status": "seeded_from_track" if seeded else "pending_student_binding",
            "status_score": status_score("seeded" if seeded else "pending_student_binding"),
            "required_before_final_claim": True,
            "source_artifact_expected": "run_config.json" if field in {"model_revision", "seed", "frozen_run_command"} else "artifact_index.json",
        })
    return rows


def preregistration_drift_audit(track: CapstoneTrack) -> list[dict[str, Any]]:
    fields = [
        ("research_question", track.research_question, "narrowing only or repair-log entry required"),
        ("primary_metric", track.primary_metric, "do not switch primary metric after results"),
        ("controls", "; ".join(track.controls), "add through repair; do not delete failed controls"),
        ("falsifiers", "; ".join(track.falsifiers), "do not remove after source run"),
        ("allowed_claim", track.allowed_claim, "narrowing allowed"),
        ("forbidden_claim", track.forbidden_claim, "never remove boundary"),
    ]
    return [{
        "drift_audit_id": f"{track.track_id}-DA{i:02d}",
        "field": field,
        "preregistered_value": value,
        "paper_value": "",
        "drift_status": "pending_student_binding",
        "allowed_change": allowed,
    } for i, (field, value, allowed) in enumerate(fields, 1)]


def package_stage_status(track: CapstoneTrack) -> list[dict[str, Any]]:
    stages = [
        ("01_preregistration", "generated", "preregistration.md", "Question, metric, controls, falsifiers, claim ceiling."),
        ("02_source_run", "pending_student_binding", "tables/result_binding_template.csv", "Immutable source run and exact measurements."),
        ("03_adversarial_review", "pending_human_review", "adversarial_review.md", "Reviewer attacks instrumentation, controls, leakage, claim language."),
        ("04_repair_accounting", "not_run", "tables/repair_log.csv", "At most one repair, with original run preserved."),
        ("05_final_paper", "draft_generated", "paper.md", "Result prose must be filled from bound artifacts."),
        ("06_release", "pending_student_binding", "public_release_checklist.md", "Release hash, review signoff, and safety scope."),
    ]
    return [{
        "stage_id": f"{track.track_id}-{stage}",
        "stage": stage,
        "status": status,
        "status_score": status_score(status),
        "primary_artifact": artifact,
        "what_to_inspect": inspect,
        "science_claim_allowed": stage == "06_release" and False,
    } for stage, status, artifact, inspect in stages]


def source_claim_binding_matrix(track: CapstoneTrack) -> list[dict[str, Any]]:
    rows = []
    for i, component in enumerate(evidence_matrix(track), 1):
        requires_source = bool(component.get("source_run_bound_required")) or "source" in str(component.get("status", ""))
        rows.append({
            "binding_row_id": f"{track.track_id}-CB{i:02d}",
            "claim_component": component["claim_component"],
            "capstone_artifact": component["artifact"],
            "source_run_required": requires_source,
            "bound_status": "pending" if requires_source else "not_required_for_scaffold",
            "evidence_rung_ceiling": track.evidence_rung_ceiling if requires_source else "AUDIT + FORMAL scaffold",
            "student_bound_artifact": "",
            "student_bound_row_or_metric": "",
        })
    return rows


def artifact_dependency_edges(track: CapstoneTrack) -> list[dict[str, Any]]:
    edges = [
        ("preregistration.md", "paper.md", "paper must answer preregistered question"),
        ("tables/result_binding_template.csv", "paper.md", "paper numbers must come from frozen run"),
        ("tables/control_falsifier_matrix.csv", "adversarial_review.md", "review must attack registered controls"),
        ("tables/review_rubric.csv", "review_response.md", "response must answer weighted review"),
        ("tables/repair_log.csv", "review_response.md", "repair cannot replace original run"),
        ("claim_card.md", "ledger_suggestions.md", "ledger claim must match claim card"),
        ("negative_result_appendix.md", "paper.md", "failed controls stay visible"),
        ("reproduction_guide.md", "public_release_checklist.md", "release requires rerun recipe"),
    ]
    rows = []
    for i, (src, dst, reason) in enumerate(edges, 1):
        rows.append({
            "edge_id": f"{track.track_id}-AD{i:02d}",
            "source_artifact": src,
            "target_artifact": dst,
            "dependency_reason": reason,
            "breakage_if_missing": "claim drift or unreproducible result",
        })
    return rows


def claim_to_artifact_map(track: CapstoneTrack) -> list[dict[str, Any]]:
    claims = [
        ("package_scaffold_ready", "AUDIT + FORMAL", "package_readiness_report.md", "tables/package_validation.csv", "supported after scaffold run"),
        ("source_claim_true", track.evidence_rung_ceiling, "paper.md", "tables/result_binding_template.csv", "blocked until frozen source run is bound"),
        ("controls_visible", "AUDIT", "tables/control_falsifier_matrix.csv", "adversarial_review.md", "scaffolded; source outcomes pending"),
        ("negative_result_reported", "AUDIT", "negative_result_appendix.md", "tables/failure_modes_contribution.csv", "scaffolded; concrete examples pending"),
        ("repair_accounted", "AUDIT + FORMAL", "tables/repair_log.csv", "review_response.md", "scaffolded; repair status pending"),
        ("reproducible_package", "AUDIT", "reproduction_guide.md", "tables/reproduction_checklist.csv", "blocked until source command, seed, hashes filled"),
    ]
    return [{
        "claim_map_id": f"{track.track_id}-CM{i:02d}",
        "claim_key": key,
        "evidence_rung": rung,
        "primary_artifact": primary,
        "supporting_artifact": support,
        "current_status": status,
        "student_result_value": "",
        "must_cite_in_paper": key != "source_claim_true" or True,
    } for i, (key, rung, primary, support, status) in enumerate(claims, 1)]


def claim_risk_register(track: CapstoneTrack) -> list[dict[str, Any]]:
    risks = [
        ("preregistration_drift", 5, "paper answers a nicer question than preregistration", "tables/preregistration_drift_audit.csv"),
        ("frozen_run_replacement", 5, "repair run silently replaces original run", "tables/repair_log.csv"),
        ("control_evasion", 5, "failed control disappears from main paper", "tables/control_falsifier_matrix.csv"),
        ("claim_inflation", 5, "source rung or capstone scaffold is overstated", "claim_card.md"),
        ("reproduction_gap", 4, "command, seed, hash, model, or artifact index missing", "tables/reproduction_checklist.csv"),
        ("review_formality", 4, "review exists but no decision changed anything", "adversarial_review.md"),
        ("human_label_omission", 4, "human-label or review fields are cited while blank", "tables/human_review_queue.csv"),
        ("figure_overclaim", 4, "figure title or caption implies source claim without bound results", "tables/plot_manifest.csv"),
    ]
    return [{
        "risk_id": f"{track.track_id}-RK{i:02d}",
        "risk_key": key,
        "severity_1_to_5": severity,
        "risk_description": desc,
        "catch_artifact": artifact,
        "visible_in_main_package": True,
        "current_status": "pending_review" if severity >= 4 else "watch",
    } for i, (key, severity, desc, artifact) in enumerate(risks, 1)]


def review_action_items(track: CapstoneTrack, rubric: Sequence[Mapping[str, Any]], queue: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    i = 0
    for item in rubric:
        i += 1
        rows.append({
            "action_id": f"{track.track_id}-RA{i:02d}",
            "action_type": "rubric_area",
            "priority": float(item.get("risk_prior", 0.5)) * float(item.get("weight_percent", 0)) / 20.0,
            "question": item.get("review_question", ""),
            "target_artifact": "adversarial_review.md",
            "status": "pending_reviewer_score",
        })
    for item in queue:
        i += 1
        rows.append({
            "action_id": f"{track.track_id}-RA{i:02d}",
            "action_type": "human_review_item",
            "priority": 1.0 if item.get("required") else 0.4,
            "question": item.get("question", ""),
            "target_artifact": item.get("source_or_package_artifact", "paper.md"),
            "status": "pending_reviewer_decision",
        })
    return rows


def failure_specimens(track: CapstoneTrack) -> list[dict[str, Any]]:
    specimens = []
    for i, row in enumerate(failure_mode_rows(track), 1):
        specimens.append({
            "specimen_id": f"{track.track_id}-FS{i:02d}",
            "specimen_type": "expected_failure_mode",
            "failure_key": row["failure_mode"],
            "trigger_or_control": row["how_to_trigger"],
            "source_example_id": "",
            "observed_value": "",
            "claim_effect": "pending: narrows | kills | repair_needed",
            "reporting_destination": "negative_result_appendix.md",
        })
    extra = [
        ("forbidden_claim_detected", track.forbidden_claim, "claim_card.md"),
        ("repair_run_replaced_original", "repair run overwrote or hid original frozen run", "tables/repair_log.csv"),
        ("source_metric_unbound", "paper number has no source-row binding", "tables/result_binding_template.csv"),
    ]
    offset = len(specimens)
    for j, (key, trigger, dest) in enumerate(extra, 1):
        specimens.append({
            "specimen_id": f"{track.track_id}-FS{offset + j:02d}",
            "specimen_type": "package_failure",
            "failure_key": key,
            "trigger_or_control": trigger,
            "source_example_id": "",
            "observed_value": "",
            "claim_effect": "pending_review",
            "reporting_destination": dest,
        })
    return specimens


def build_rows(track: CapstoneTrack, tracks: Sequence[CapstoneTrack], schema_rows: Sequence[Mapping[str, Any]], data_info: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {
        "track_options": track_options(tracks, track),
        "seed_track_schema_audit": [dict(r) for r in schema_rows],
        "artifact_checklist": artifact_checklist(track),
        "evidence_matrix": evidence_matrix(track),
        "control_falsifier_matrix": control_falsifier_matrix(track),
        "claim_language_audit": claim_language_audit(track),
        "review_rubric": review_rubric_rows(),
        "human_review_queue": human_review_queue(track),
        "repair_log": repair_log(track),
        "failure_modes_contribution": failure_mode_rows(track),
        "reproduction_checklist": reproduction_checklist(track, data_info),
        "result_binding_template": result_binding_template(track),
        "preregistration_drift_audit": preregistration_drift_audit(track),
        "package_stage_status": package_stage_status(track),
        "source_claim_binding_matrix": source_claim_binding_matrix(track),
        "artifact_dependency_edges": artifact_dependency_edges(track),
        "claim_to_artifact_map": claim_to_artifact_map(track),
        "claim_risk_register": claim_risk_register(track),
        "failure_specimens": failure_specimens(track),
        "plot_reading_guide": plot_guide(),
    }
    rows["review_action_items"] = review_action_items(track, rows["review_rubric"], rows["human_review_queue"])
    return rows


def package_validation(track: CapstoneTrack, rows: Mapping[str, Sequence[Mapping[str, Any]]], data_info: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checks = [
        ("rubric_weights_sum_to_100", sum(w for _, w, _ in RUBRIC) == 100, "Review weights must sum to 100.", "formal_package"),
        ("seed_data_hash_available", bool(data_info.get("data_sha256")), "Seed data must have a hash.", "data"),
        ("seed_schema_ok", all(r.get("schema_ok") for r in rows["seed_track_schema_audit"]), "Seed-track rows must pass schema checks.", "data"),
        ("track_has_controls", len(track.controls) >= 3, "Track needs multiple controls.", "controls"),
        ("track_has_falsifiers", len(track.falsifiers) >= 2, "Track needs falsifiers.", "controls"),
        ("failure_modes_present", len(track.expected_failure_modes) >= 2, "Track should name expected failure modes.", "negative_results"),
        ("forbidden_claim_present", bool(track.forbidden_claim), "Claim boundary must be explicit.", "claim_language"),
        ("stopping_rule_present", bool(track.stopping_rule), "Stopping rule must be explicit.", "repair"),
        ("evidence_matrix_nonempty", len(rows["evidence_matrix"]) >= 8, "Evidence matrix must cover formal/audit/binding components.", "evidence"),
        ("human_review_queue_present", bool(rows["human_review_queue"]), "Review queue must exist.", "review"),
        ("binding_template_has_primary_metric", any(r.get("field") == "primary_metric_value" for r in rows["result_binding_template"]), "Binding template must ask for the primary metric.", "binding"),
        ("plot_manifest_planned", len(rows["plot_reading_guide"]) >= 8, "Plot guide must enumerate figures and non-claims.", "plots"),
        ("source_run_binding_pending", True, "Source run still must be bound before science claims.", "binding"),
    ]
    validation_rows = []
    for i, (key, ok, desc, family) in enumerate(checks, 1):
        validation_rows.append({
            "validation_check_id": f"{track.track_id}-PV{i:02d}",
            "check": key,
            "family": family,
            "ok": bool(ok),
            "status": "pass" if ok else "fail",
            "description": desc,
            "blocks_package_readiness": key != "source_run_binding_pending" and not bool(ok),
            "blocks_science_claim": key == "source_run_binding_pending" or not bool(ok),
        })
    package_ready = all(r["ok"] for r in validation_rows if r["check"] != "source_run_binding_pending")
    validation = {
        "track_id": track.track_id,
        "package_ready_for_student_replacement": package_ready,
        "science_ready": False,
        "why_not_science_ready": "Lab 35 generated a scaffold; bind a frozen source-lab run before final claims.",
        "rubric_weight_total": sum(w for _, w, _ in RUBRIC),
        "table_counts": {k: len(v) for k, v in rows.items()},
        "n_controls": len(track.controls),
        "n_falsifiers": len(track.falsifiers),
        "n_failure_modes": len(track.expected_failure_modes),
        "human_review_required": track.human_review_required,
        "source_run_bound": False,
        "checks_passed": sum(1 for r in validation_rows if r["ok"]),
        "checks_total": len(validation_rows),
    }
    return validation, validation_rows


def warning_rows(track: CapstoneTrack, data_info: Mapping[str, Any], validation: Mapping[str, Any]) -> list[dict[str, Any]]:
    warnings = [
        ("science_ready_false", "info", "Default Lab 35 run is a scaffold only; no source-lab science claim is supported yet.", "Bind an immutable source run and fill result_binding_template.csv."),
        ("source_run_unbound", "warning", "Frozen source-run path, metrics, controls, and hashes are blank.", "Fill source binding fields before paper numbers are cited."),
        ("review_pending", "warning" if track.human_review_required else "info", "Human review fields remain blank in the scaffold.", "Complete human_review_queue.csv before final publication claim."),
        ("negative_results_pending", "info", "Failure specimens are expected paths, not observed examples yet.", "Populate failure_specimens with concrete source-run rows."),
    ]
    if data_info.get("manifest_ok") is not True:
        warnings.append(("manifest_not_verified", "warning", "Seed data hash was unavailable or did not match a manifest entry.", "Regenerate or update the data manifest before release."))
    if not validation.get("package_ready_for_student_replacement"):
        warnings.append(("package_validation_failed", "error", "One or more package-readiness checks failed.", "Inspect tables/package_validation.csv."))
    rows = []
    for i, (key, level, message, action) in enumerate(warnings, 1):
        rows.append({
            "warning_id": f"{track.track_id}-W{i:02d}",
            "warning_key": key,
            "level": level,
            "message": message,
            "recommended_action": action,
        })
    return rows


def write_csv_table(ctx: bench.RunContext, rel: str, rows: Sequence[Mapping[str, Any]], desc: str) -> None:
    path = ctx.path(*rel.split("/"))
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", desc)


def write_tables(ctx: bench.RunContext, rows: Mapping[str, Sequence[Mapping[str, Any]]], validation_rows: Sequence[Mapping[str, Any]]) -> None:
    specs = [
        ("tables/track_options.csv", "track_options", "Available seed tracks and selected-track marker."),
        ("tables/seed_track_schema_audit.csv", "seed_track_schema_audit", "Seed-track schema audit."),
        ("tables/artifact_checklist.csv", "artifact_checklist", "Generated and pending artifacts with stable IDs."),
        ("tables/evidence_matrix.csv", "evidence_matrix", "Claim-component evidence matrix with status scores."),
        ("tables/control_falsifier_matrix.csv", "control_falsifier_matrix", "Controls, falsifiers, expected failure modes, and source artifact slots."),
        ("tables/claim_language_audit.csv", "claim_language_audit", "Allowed, forbidden, negative, and ledger claim language."),
        ("tables/review_rubric.csv", "review_rubric", "Weighted review rubric with risk priors."),
        ("tables/human_review_queue.csv", "human_review_queue", "Human review queue."),
        ("tables/review_action_items.csv", "review_action_items", "Unified review action queue from rubric and review items."),
        ("tables/repair_log.csv", "repair_log", "Frozen-run and repair-run accounting."),
        ("tables/failure_modes_contribution.csv", "failure_modes_contribution", "Failure-mode atlas contribution."),
        ("tables/failure_specimens.csv", "failure_specimens", "Concrete slots for failed controls, contradicted claims, and source-run counterexamples."),
        ("tables/reproduction_checklist.csv", "reproduction_checklist", "Reproducibility checklist."),
        ("tables/result_binding_template.csv", "result_binding_template", "Fields students fill from the frozen source run."),
        ("tables/preregistration_drift_audit.csv", "preregistration_drift_audit", "Preregistration-vs-paper drift audit."),
        ("tables/package_stage_status.csv", "package_stage_status", "Capstone phase status table."),
        ("tables/source_claim_binding_matrix.csv", "source_claim_binding_matrix", "Which claim components require source-run binding."),
        ("tables/artifact_dependency_edges.csv", "artifact_dependency_edges", "Artifact dependency edges for package review."),
        ("tables/claim_to_artifact_map.csv", "claim_to_artifact_map", "Claim-to-artifact binding map."),
        ("tables/claim_risk_register.csv", "claim_risk_register", "Claim-risk register."),
        ("tables/plot_reading_guide.csv", "plot_reading_guide", "Plot reading guide."),
    ]
    for rel, key, desc in specs:
        write_csv_table(ctx, rel, rows[key], desc)
    write_csv_table(ctx, "tables/package_validation.csv", validation_rows, "Package validation checks.")
    results = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results, rows["evidence_matrix"])
    ctx.register_artifact(results, "table", "Alias of tables/evidence_matrix.csv.")


def write_jsonl(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(r), sort_keys=True, default=str) + "\n" for r in rows), encoding="utf-8")


def write_failure_cards(ctx: bench.RunContext, specimens: Sequence[Mapping[str, Any]]) -> None:
    jsonl_path = ctx.path("tables", "failure_specimens.jsonl")
    write_jsonl(jsonl_path, specimens)
    ctx.register_artifact(jsonl_path, "table", "JSONL copy of failure and counterexample specimen slots.")
    lines = ["# Failure specimens", "", "These are not observed source-run failures yet. They are the slots that keep negative evidence visible once a source run is bound.", ""]
    for row in specimens:
        lines += [
            f"## {row.get('specimen_id', '')}: {row.get('failure_key', '')}",
            "",
            f"- type: `{row.get('specimen_type', '')}`",
            f"- trigger or control: {row.get('trigger_or_control', '')}",
            f"- claim effect: `{row.get('claim_effect', '')}`",
            f"- reporting destination: `{row.get('reporting_destination', '')}`",
            "",
        ]
    path = ctx.path("cards", "failure_specimens.md")
    bench.write_text(path, "\n".join(lines).rstrip() + "\n")
    ctx.register_artifact(path, "summary", "Human-readable failure specimen cards.")


def write_text_file(ctx: bench.RunContext, rel: str, lines: Sequence[str], kind: str, desc: str) -> None:
    path = ctx.path(rel)
    bench.write_text(path, "\n".join(lines).rstrip() + "\n")
    ctx.register_artifact(path, kind, desc)


def write_markdown_files(ctx: bench.RunContext, track: CapstoneTrack, validation: Mapping[str, Any], data_info: Mapping[str, Any], select_note: str) -> None:
    write_text_file(ctx, "preregistration.md", [
        "# Capstone preregistration", "", f"Track: `{track.track_id}`", f"Title: {track.title}", f"Source lab: `{track.source_lab}`", f"Evidence ceiling: `{track.evidence_rung_ceiling}`", "", "## Research question", "", track.research_question, "", "## Dataset, model, and measurement sites", "", f"- dataset: `{track.dataset}`", f"- model: {track.model}", f"- measurement sites: {track.measurement_sites}", "", "## Primary metric", "", track.primary_metric, "", "## Secondary metrics", "", *bullets(track.secondary_metrics), "", "## Controls", "", *bullets(track.controls), "", "## Falsifiers", "", *bullets(track.falsifiers), "", "## Stopping rule", "", track.stopping_rule, "", "## Allowed claim", "", track.allowed_claim, "", "## Forbidden claim", "", track.forbidden_claim, "", "## Expected failure modes", "", *bullets(track.expected_failure_modes), "", "## Safety scope", "", track.safety_scope, ""], "summary", "Capstone preregistration.")
    write_text_file(ctx, "paper.md", [
        "# Capstone paper draft", "", "## Abstract", "", "This draft is not a result yet. Bind the frozen source run, report controls, and update the claim only after review.", "", "## Question", "", track.research_question, "", "## Methods", "", f"The package uses `{track.dataset}` and the model/run specified in `tables/result_binding_template.csv`.", "", "## Primary metric", "", track.primary_metric, "", "## Results from frozen run", "", "Binding required: paste measured source-run values and exact artifact paths here. Do not delete failed controls.", "", "## Figure-reading rule", "", "Every figure caption must name the table it was built from and must say whether it supports a scaffold claim, a source-lab claim, or only a pending review question.", "", "## Controls and counterexamples", "", *bullets(track.controls), "", "## Negative-result path", "", track.negative_result_sentence, "", "## Claim", "", f"Allowed only after source-run binding: {track.allowed_claim}", "", "## Boundary", "", f"This paper must not imply: {track.forbidden_claim}", "", "## Outline checkpoints", "", *bullets(track.paper_outline), ""], "summary", "Capstone paper draft.")
    write_text_file(ctx, "claim_card.md", [
        "# Claim card", "", f"- selected track: `{track.track_id}`", f"- source lab: `{track.source_lab}`", f"- evidence ceiling: `{track.evidence_rung_ceiling}`", "- capstone evidence: `AUDIT + FORMAL` until the source run is bound", f"- human review required: `{str(track.human_review_required).lower()}`", "", "## Allowed claim", "", track.allowed_claim, "", "## Forbidden claim", "", track.forbidden_claim, "", "## Negative-result fallback", "", track.negative_result_sentence, "", "## Ledger draft", "", "```text", track.claim_ledger_template, "```", ""], "summary", "Claim card.")
    write_text_file(ctx, "adversarial_review.md", [
        "# Adversarial review", "", "Answer each section before finalizing the paper.", "", "## Instrumentation", "", "Which hook, tokenization, cache, dtype, split, score, or schema bug would invalidate the result?", "", "## Data leakage and shortcuts", "", f"Could `{track.dataset}` leak the answer or label through ordering, wording, token overlap, template cues, or split contamination?", "", "## Controls to attack", "", *bullets(track.controls), "", "## Falsifiers to make true", "", *bullets(track.falsifiers), "", "## Claim language", "", f"Does any sentence imply the forbidden claim: {track.forbidden_claim}", "", "## Figure audit", "", "Open `plot_manifest.json`. For every figure, verify that the source table row count is nonzero, the claim supported is scoped, and any pending-source result is labeled pending.", "", "## Safety", "", track.safety_scope, ""], "summary", "Adversarial review template.")
    write_text_file(ctx, "review_response.md", [
        "# Review response", "", "## Reviewer decision", "", "Pending: approve as written | narrow claim | run one repair | unsupported.", "", "## Repair run decision", "", track.stopping_rule, "", "## Claim revision", "", f"Starting allowed claim: {track.allowed_claim}", "", "## Figure or table changes required by review", "", "List any revised figures here and keep the original frozen-run artifacts visible.", "", "## Unresolved risks", "", *bullets(track.expected_failure_modes), ""], "summary", "Review response template.")
    write_text_file(ctx, "reproduction_guide.md", [
        "# Reproduction guide", "", "```bash", "cd interpretability", "pip install -r requirements.txt", "python data/make_capstone_seed_tracks.py", track.tier_a_command, track.frozen_run_command, "python interp_bench.py --lab lab35 --tier a", "```", "", f"Seed data hash: `{data_info.get('data_sha256', '')}`", "", "## Source artifacts required", "", *[f"- `{a}`" for a in track.planned_artifacts], "", "## Minimum binding fields", "", "Fill `tables/result_binding_template.csv` before final claims. At minimum: frozen run directory, artifact index, command, model revision, data hash, seed, primary metric value, strongest control value, failed controls, and release package hash.", "", "## Release rule", "", "Do not publish until the review queue is filled and the claim card matches the evidence matrix.", ""], "summary", "Reproduction guide.")
    write_text_file(ctx, "method_card.md", [
        "# Lab 35 method card", "", "Lab 35 is a capstone package generator and validator, not a new mechanistic measurement.", "", f"- selected track: `{track.track_id}`", f"- selection note: {select_note}", f"- source lab: `{track.source_lab}`", f"- source evidence ceiling: `{track.evidence_rung_ceiling}`", "- capstone evidence: `AUDIT + FORMAL`", f"- package ready for student replacement: `{str(validation['package_ready_for_student_replacement']).lower()}`", f"- science ready: `{str(validation['science_ready']).lower()}`", "", "## What this run validated", "", "Seed schema, package artifact coverage, rubric weights, human-review fields, claim-boundary scaffolding, source-binding slots, failure-specimen slots, and shared bench self-checks.", "", "## What remains pending", "", "A real frozen source-lab run must be attached and reviewed before scientific claims are publishable.", ""], "summary", "Method card.")
    write_text_file(ctx, "operationalization_audit.md", [
        "# Operationalization audit", "", "```yaml", "headline_claim: \"a reproducible interpretability paper package is ready to defend\"", "cheap_explanation: \"the package looks complete but hides drift, failed controls, blank source bindings, or claim inflation\"", "killer_control: \"artifact checklist, evidence matrix, binding template, repair log, review rubric, failure specimens, plot manifest, and claim-language audit all remain visible\"", "result: \"scaffold_generated_source_run_pending\"", "claim_allowed: \"package scaffold only\"", "```", "", "## Controls", "", *bullets(track.controls), "", "## Falsifiers", "", *bullets(track.falsifiers), "", "## Claim boundary", "", track.forbidden_claim, ""], "summary", "Operationalization audit.")
    write_text_file(ctx, "package_readiness_report.md", ["# Package readiness report", "", f"- package_ready_for_student_replacement: `{str(validation['package_ready_for_student_replacement']).lower()}`", f"- science_ready: `{str(validation['science_ready']).lower()}`", f"- why: {validation['why_not_science_ready']}", f"- checks: {validation['checks_passed']} / {validation['checks_total']}", "", "## Remaining steps", "", "1. Run and freeze the source lab.", "2. Fill `tables/result_binding_template.csv`.", "3. Answer the adversarial review.", "4. Log any repair run separately.", "5. Ensure every figure has a source table and bounded caption.", "6. Ensure the claim card does not exceed the evidence ceiling.", ""], "summary", "Package readiness report.")
    write_text_file(ctx, "negative_result_appendix.md", ["# Negative-result appendix", "", track.negative_result_sentence, "", "## Failure modes to report", "", *bullets(track.expected_failure_modes), "", "## Specimen rule", "", "Concrete negative examples go in `tables/failure_specimens.csv` and `cards/failure_specimens.md`, not only in prose.", ""], "summary", "Negative-result appendix.")
    write_text_file(ctx, "public_release_checklist.md", ["# Public release checklist", "", "- Claim card evidence rung does not exceed source lab.", "- Original frozen run remains visible.", "- Failed controls remain visible.", "- Required human-review fields are filled.", "- Plot manifest exists and every figure names its source table.", "- Safety scope is explicit.", "- Release zip hash is recorded.", ""], "summary", "Public release checklist.")


def write_run_summary(ctx: bench.RunContext, track: CapstoneTrack, validation: Mapping[str, Any], data_info: Mapping[str, Any], select_note: str) -> None:
    write_text_file(ctx, "run_summary.md", [
        "# Lab 35 run summary: reproducible interpretability paper capstone", "", f"- selected track: `{track.track_id}` ({select_note})", f"- source lab: `{track.source_lab}`", f"- evidence ceiling: `{track.evidence_rung_ceiling}`", f"- seed tracks selected: {data_info.get('n_rows_selected')} from `{pathlib.Path(str(data_info.get('data_path', ''))).name}`", f"- package_ready_for_student_replacement: `{str(validation['package_ready_for_student_replacement']).lower()}`", f"- science_ready: `{str(validation['science_ready']).lower()}`", "- smallest surviving claim: the package scaffold is reproducible and reviewable; no source-lab science claim is made yet", "", "## Why science_ready is false", "", validation["why_not_science_ready"], "", "## What to inspect first", "", "1. `diagnostics/warning_summary.csv`", "2. `plot_manifest.json`", "3. `tables/result_binding_template.csv`", "4. `tables/source_claim_binding_matrix.csv`", "5. `tables/control_falsifier_matrix.csv`", "6. `cards/failure_specimens.md`", "", "## Reading order", "", "1. `method_card.md`", "2. `preregistration.md`", "3. `tables/result_binding_template.csv`", "4. `tables/evidence_matrix.csv`", "5. `adversarial_review.md`", "6. `review_response.md`", "7. `paper.md`", "8. `reproduction_guide.md`", "9. `claim_card.md`", ""], "summary", "Run summary.")


def write_warning_artifacts(ctx: bench.RunContext, warnings: Sequence[Mapping[str, Any]]) -> None:
    csv_path = ctx.path("diagnostics", "warning_summary.csv")
    bench.write_csv_with_context(ctx, csv_path, warnings)
    ctx.register_artifact(csv_path, "diagnostic", "Run warnings and caveats.")
    json_path = ctx.path("diagnostics", "warning_summary.json")
    bench.write_json(json_path, {"warnings": list(warnings), "n_warnings": len(warnings)})
    ctx.register_artifact(json_path, "diagnostic", "JSON copy of run warnings and caveats.")


def write_status_and_state(ctx: bench.RunContext, track: CapstoneTrack, data_info: Mapping[str, Any], validation: Mapping[str, Any], validation_rows: Sequence[Mapping[str, Any]], hook_check: Mapping[str, Any], lens_check: Mapping[str, Any], patch_noop: Mapping[str, Any], warnings: Sequence[Mapping[str, Any]]) -> None:
    run_snapshot = {
        "lab": LAB_ID,
        "run_name": ctx.run_dir.name,
        "selected_track": track.track_id,
        "source_lab": track.source_lab,
        "tier": getattr(ctx.args, "tier", ""),
        "prompt_set": getattr(ctx.args, "prompt_set", ""),
        "max_examples": getattr(ctx.args, "max_examples", ""),
        "seed": getattr(ctx.args, "seed", ""),
        "model_id": getattr(ctx, "model_id", "") or getattr(ctx.args, "model", ""),
        "dtype": getattr(ctx.args, "dtype", ""),
        "science_ready": False,
        "source_run_bound": False,
    }
    path = ctx.path("diagnostics", "run_config_snapshot.json"); bench.write_json(path, run_snapshot); ctx.register_artifact(path, "diagnostic", "Lab 35 run configuration snapshot for exported tables and figures.")
    safety = {"lab": LAB_ID, "selected_track": track.track_id, "source_lab": track.source_lab, "safety_scope": track.safety_scope, "human_review_required": track.human_review_required, "unsafe_generation": False, "model_editing_performed": False, "external_tool_side_effects": False, "science_ready": False, "blocked_claim": track.forbidden_claim}
    path = ctx.path("diagnostics", "safety_status.json"); bench.write_json(path, safety); ctx.register_artifact(path, "diagnostic", "Safety and claim-boundary status.")
    self_check = {"hook_parity_ok": bool(hook_check.get("ok")), "lens_self_check_ok": bool(lens_check.get("ok")), "patch_noop_ok": bool(patch_noop.get("ok")), "seed_manifest_ok": data_info.get("manifest_ok"), "package_validation_checks_passed": validation.get("checks_passed"), "package_validation_checks_total": validation.get("checks_total"), "package_ready_for_student_replacement": validation.get("package_ready_for_student_replacement"), "science_ready": False, "source_run_binding_required": True, "warnings": len(warnings)}
    path = ctx.path("diagnostics", "self_check_status.json"); bench.write_json(path, self_check); ctx.register_artifact(path, "diagnostic", "Aggregated self-check status.")
    path = ctx.path("diagnostics", "package_validation.json"); bench.write_json(path, {"summary": validation, "checks": list(validation_rows)}); ctx.register_artifact(path, "diagnostic", "Package validation summary.")
    path = ctx.path("diagnostics", "frozen_run_binding_status.json"); bench.write_json(path, {"source_run_bound": False, "binding_table": "tables/result_binding_template.csv", "source_lab": track.source_lab, "frozen_run_command": track.frozen_run_command}); ctx.register_artifact(path, "diagnostic", "Explicit status that source run is not bound yet.")
    path = ctx.path("state", "selected_track.json"); bench.write_json(path, dataclasses.asdict(track)); ctx.register_artifact(path, "state", "Selected seed track.")
    path = ctx.path("state", "capstone_package_manifest.json"); bench.write_json(path, {"lab": LAB_ID, "selected_track": track.track_id, "validation": validation, "data": dict(data_info), "required_package_artifacts": PACKAGE_ARTIFACTS, "pending_source_artifacts": track.planned_artifacts, "warnings": list(warnings)}); ctx.register_artifact(path, "state", "Capstone scaffold manifest.")


def write_claims(ctx: bench.RunContext, track: CapstoneTrack, validation: Mapping[str, Any]) -> None:
    claims = [
        {"id": f"{LAB_ID}-C1", "tag": "AUDIT,FORMAL", "text": f"Lab 35 generated a reproducible capstone scaffold for `{track.track_id}` with {validation['checks_passed']}/{validation['checks_total']} package checks passing. This is a scaffold-readiness claim only; source-lab scientific claims require frozen-run binding.", "artifact": f"runs/{ctx.run_dir.name}/package_readiness_report.md", "falsifier": "Required package artifacts are missing, rubric weights do not sum to 100, plot source tables are absent, or the final claim exceeds the source lab evidence ceiling."},
        {"id": f"{LAB_ID}-C2", "tag": "AUDIT", "text": f"Lab 35 made source-run binding, review action items, failure specimens, and claim-risk rows explicit for `{track.track_id}`; all source-lab result claims remain blocked until those rows are filled.", "artifact": f"runs/{ctx.run_dir.name}/tables/source_claim_binding_matrix.csv", "falsifier": "The paper cites source-run numbers that do not map to result_binding_template.csv or claim_to_artifact_map.csv."},
    ]
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)


def write_figure_source(ctx: bench.RunContext, plot_name: str, rows: Sequence[Mapping[str, Any]], description: str) -> tuple[str, int]:
    stem = pathlib.Path(plot_name).stem
    path = ctx.path("tables", "figure_sources", f"{stem}_source.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", description)
    return str(path.relative_to(ctx.run_dir)), len(rows)


def manifest_entry(ctx: bench.RunContext, plot_name: str, source_rel: str, row_count: int, metric: str, claim: str, caption: str, status: str = "generated") -> dict[str, Any]:
    return {
        "figure": f"plots/{plot_name}",
        "status": status,
        "source_table": source_rel,
        "row_count": row_count,
        "metric": metric,
        "control_or_comparison": "see source table",
        "claim_supported": claim,
        "caption": caption,
    }


def write_plot_manifest(ctx: bench.RunContext, manifest_rows: Sequence[Mapping[str, Any]]) -> None:
    csv_path = ctx.path("tables", "plot_manifest.csv")
    bench.write_csv_with_context(ctx, csv_path, manifest_rows)
    ctx.register_artifact(csv_path, "table", "Plot manifest table with source tables, row counts, metrics, and bounded claims.")
    json_path = ctx.path("plot_manifest.json")
    bench.write_json(json_path, {"figures": list(manifest_rows), "n_figures": len(manifest_rows), "note": "Every figure is scaffold/package evidence unless a future source run binds concrete results."})
    ctx.register_artifact(json_path, "diagnostic", "Machine-readable plot manifest.")


def annotate_caption(fig: Any, caption: str) -> None:
    fig.text(0.01, 0.01, caption, ha="left", va="bottom", fontsize=7.2, color="#333333", wrap=True)


def write_plots(ctx: bench.RunContext, track: CapstoneTrack, rows: Mapping[str, Sequence[Mapping[str, Any]]], validation: Mapping[str, Any]) -> None:
    guide = list(rows["plot_reading_guide"])
    if ctx.args.no_plots:
        manifest = [manifest_entry(ctx, g["plot"], g["source_table"], 0, "not generated", g["claim_supported"], g["non_claim"], status="not_generated_no_plots") for g in guide]
        write_plot_manifest(ctx, manifest)
        return
    import matplotlib.pyplot as plt
    from matplotlib import patches

    manifest: list[dict[str, Any]] = []

    # 1. Dashboard
    dashboard_source = []
    dashboard_source.extend({"panel": "validation", "label": r["check"], "value": 1 if r["ok"] else 0, "status": r["status"], "family": r["family"]} for r in package_validation(track, rows, {"data_sha256": "placeholder"})[1])
    dashboard_source.extend({"panel": "stage", "label": r["stage"], "value": r["status_score"], "status": r["status"], "family": "package_stage"} for r in rows["package_stage_status"])
    source_rel, n = write_figure_source(ctx, "capstone_dashboard.png", dashboard_source, "Source table for capstone dashboard.")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.8))
    fig.suptitle("Lab 35 capstone package dashboard", fontsize=14, fontweight="bold")
    validation_rows = [r for r in dashboard_source if r["panel"] == "validation"]
    axes[0, 0].bar(range(len(validation_rows)), [r["value"] for r in validation_rows])
    axes[0, 0].set_ylim(0, 1.05); axes[0, 0].set_ylabel("pass = 1")
    axes[0, 0].set_xticks(range(len(validation_rows)), [r["label"].replace("_", "\n") for r in validation_rows], rotation=0, fontsize=7)
    axes[0, 0].set_title("Package validation checks")
    stage_rows = [r for r in dashboard_source if r["panel"] == "stage"]
    axes[0, 1].barh(range(len(stage_rows)), [r["value"] for r in stage_rows])
    axes[0, 1].set_yticks(range(len(stage_rows)), [r["label"].replace("_", " ") for r in stage_rows], fontsize=8)
    axes[0, 1].set_xlim(0, 1); axes[0, 1].set_xlabel("scaffold status score")
    axes[0, 1].set_title("Package phase status")
    artifact_counts = Counter(r["category"] for r in rows["artifact_checklist"])
    axes[1, 0].bar(list(artifact_counts), list(artifact_counts.values()))
    axes[1, 0].set_ylabel("artifact count"); axes[1, 0].set_title("Artifact categories")
    flags = ["package_ready", "science_ready", "source_bound", "human_review_required"]
    vals = [1 if validation["package_ready_for_student_replacement"] else 0, 0, 0, 1 if track.human_review_required else 0]
    axes[1, 1].bar(range(len(flags)), vals)
    axes[1, 1].set_xticks(range(len(flags)), flags, rotation=20, ha="right")
    axes[1, 1].set_ylim(0, 1.05); axes[1, 1].set_title("Readiness flags")
    caption = "Dashboard reads the package scaffold only: source-run science remains blocked until result bindings and review fields are filled."
    annotate_caption(fig, caption); fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    bench.save_figure(ctx, fig, "capstone_dashboard.png", "Lab 35 dashboard.")
    manifest.append(manifest_entry(ctx, "capstone_dashboard.png", source_rel, n, "validation pass flags and stage scores", "scaffold readiness only", caption))

    # 2. Target versus controls
    tvc_rows = [
        {"comparison_id": "target_claim", "group": "target claim", "count": 1, "bound_status": "blocked_until_source_run", "interpretation": track.allowed_claim},
        {"comparison_id": "controls", "group": "controls", "count": len(track.controls), "bound_status": "pending_source_results", "interpretation": "controls must be reported beside target"},
        {"comparison_id": "falsifiers", "group": "falsifiers", "count": len(track.falsifiers), "bound_status": "pending_source_results", "interpretation": "falsifiers can kill or narrow claim"},
        {"comparison_id": "failure_modes", "group": "failure modes", "count": len(track.expected_failure_modes), "bound_status": "pending_source_examples", "interpretation": "negative outcomes have reporting slots"},
        {"comparison_id": "human_review", "group": "review gates", "count": sum(1 for r in rows["human_review_queue"] if r.get("required")), "bound_status": "pending_review", "interpretation": "review decisions required before publication"},
    ]
    source_rel, n = write_figure_source(ctx, "target_vs_control.png", tvc_rows, "Source table for target versus controls plot.")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar([r["group"] for r in tvc_rows], [r["count"] for r in tvc_rows])
    ax.set_ylabel("registered rows / obligations")
    ax.set_title("Target claim is shown beside controls, falsifiers, and review gates")
    ax.tick_params(axis="x", rotation=25)
    caption = "The target claim has one slot, but its controls, falsifiers, failure modes, and review obligations are visible next to it. Counts are obligations, not successes."
    annotate_caption(fig, caption); fig.tight_layout(rect=(0, 0.08, 1, 1))
    bench.save_figure(ctx, fig, "target_vs_control.png", "Target claim obligations beside controls and falsifiers.")
    manifest.append(manifest_entry(ctx, "target_vs_control.png", source_rel, n, "registered obligation count", "claim-pressure visibility", caption))

    # 3. Artifact status
    art_rows = list(rows["artifact_checklist"])
    source_rel, n = write_figure_source(ctx, "artifact_contract_status.png", art_rows, "Source table for artifact contract status.")
    status_counts = Counter((r["category"], r["status"]) for r in art_rows)
    labels = [f"{cat}\n{status.replace('_', ' ')}" for cat, status in status_counts]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(range(len(labels)), list(status_counts.values()))
    ax.set_xticks(range(len(labels)), labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("artifact count")
    ax.set_title("Artifact contract status")
    caption = "Capstone artifacts are generated by the scaffold; source-run artifacts remain pending and cannot support final source claims yet."
    annotate_caption(fig, caption); fig.tight_layout(rect=(0, 0.08, 1, 1))
    bench.save_figure(ctx, fig, "artifact_contract_status.png", "Artifact readiness matrix.")
    manifest.append(manifest_entry(ctx, "artifact_contract_status.png", source_rel, n, "artifact status count", "artifact coverage audit", caption))

    # 4. Evidence matrix
    ev_rows = list(rows["evidence_matrix"])
    source_rel, n = write_figure_source(ctx, "evidence_rung_matrix.png", ev_rows, "Source table for evidence rung matrix.")
    rungs = sorted({r["evidence_rung"] for r in ev_rows})
    comps = [r["claim_component"] for r in ev_rows]
    mat = [[0.0 for _ in comps] for _ in rungs]
    for j, row in enumerate(ev_rows):
        mat[rungs.index(row["evidence_rung"])] [j] = float(row.get("status_score", status_score(row.get("status", ""))))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(rungs)), rungs)
    ax.set_xticks(range(len(comps)), [c.replace("_", "\n") for c in comps], rotation=0, fontsize=7)
    ax.set_title("Evidence rung matrix with pending-source status visible")
    fig.colorbar(im, ax=ax, shrink=.82, label="scaffold status score")
    caption = "Rows show evidence-rung categories; cell intensity is scaffold or binding status, not proof strength. Pending source cells stay dim."
    annotate_caption(fig, caption); fig.tight_layout(rect=(0, 0.09, 1, 1))
    bench.save_figure(ctx, fig, "evidence_rung_matrix.png", "Evidence rung matrix.")
    manifest.append(manifest_entry(ctx, "evidence_rung_matrix.png", source_rel, n, "status score by evidence rung", "evidence ceiling discipline", caption))

    # 5. Review radar
    review_rows = list(rows["review_rubric"])
    source_rel, n = write_figure_source(ctx, "review_score_radar.png", review_rows, "Source table for review radar.")
    areas = [r["rubric_area"] for r in review_rows]
    scores = [float(r["risk_prior"]) for r in review_rows]
    angles = [math.tau * i / len(scores) for i in range(len(scores))]
    fig = plt.figure(figsize=(6.8, 6.8))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles + angles[:1], scores + scores[:1])
    ax.fill(angles + angles[:1], scores + scores[:1], alpha=.15)
    ax.set_xticks(angles, [a.replace("_", "\n") for a in areas], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Review triage pressure")
    caption = "Radar values are risk priors for review triage. They are not reviewer scores and should be overwritten by human review."
    annotate_caption(fig, caption); fig.tight_layout(rect=(0, 0.04, 1, 1))
    bench.save_figure(ctx, fig, "review_score_radar.png", "Review rubric triage radar.")
    manifest.append(manifest_entry(ctx, "review_score_radar.png", source_rel, n, "risk prior", "review triage", caption))

    # 6. Control falsifier map
    cf_rows = list(rows["control_falsifier_matrix"])
    source_rel, n = write_figure_source(ctx, "control_falsifier_map.png", cf_rows, "Source table for control falsifier map.")
    y = list(range(len(cf_rows)))
    fig, ax = plt.subplots(figsize=(10, max(4.8, 0.48 * len(cf_rows) + 1.5)))
    ax.barh([i - .2 for i in y], [1 if r.get("control") else 0 for r in cf_rows], height=.25, label="control")
    ax.barh(y, [1 if r.get("falsifier") else 0 for r in cf_rows], height=.25, label="falsifier")
    ax.barh([i + .2 for i in y], [1 if r.get("expected_failure_mode") else 0 for r in cf_rows], height=.25, label="failure mode")
    ax.set_yticks(y, [r["control_falsifier_id"].split("-")[-1] for r in cf_rows])
    ax.set_xlim(0, 1.25); ax.set_xlabel("registered = 1")
    ax.set_title("Controls, falsifiers, and failure modes share the same review lane")
    ax.legend(loc="lower right")
    caption = "Every row should later receive source-run evidence and a claim effect: supports, narrows, kills, or unrelated."
    annotate_caption(fig, caption); fig.tight_layout(rect=(0, 0.06, 1, 1))
    bench.save_figure(ctx, fig, "control_falsifier_map.png", "Control falsifier dashboard.")
    manifest.append(manifest_entry(ctx, "control_falsifier_map.png", source_rel, n, "registered control/falsifier flags", "control visibility", caption))

    # 7. Failure mode atlas
    fm_rows = list(rows["failure_modes_contribution"])
    source_rel, n = write_figure_source(ctx, "failure_mode_atlas.png", fm_rows, "Source table for failure mode atlas.")
    fig, ax = plt.subplots(figsize=(10, max(4.8, .55 * len(fm_rows) + 1.5)))
    ax.barh(range(len(fm_rows)), [1 for _ in fm_rows])
    ax.set_yticks(range(len(fm_rows)), [r["failure_mode_id"].split("-")[-1] for r in fm_rows])
    ax.set_xlim(0, 1.2); ax.set_xlabel("reporting slot present")
    ax.set_title("Failure-mode atlas, source examples pending")
    caption = "A failure slot is useful only after it is populated with concrete source-run examples or marked not observed."
    annotate_caption(fig, caption); fig.tight_layout(rect=(0, 0.07, 1, 1))
    bench.save_figure(ctx, fig, "failure_mode_atlas.png", "Failure-mode atlas.")
    manifest.append(manifest_entry(ctx, "failure_mode_atlas.png", source_rel, n, "failure slot present", "negative result readiness", caption))

    # 8. Reproduction readiness
    rep_rows = list(rows["reproduction_checklist"])
    source_rel, n = write_figure_source(ctx, "reproduction_readiness_ladder.png", rep_rows, "Source table for reproduction readiness ladder.")
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.barh(range(len(rep_rows)), [float(r["status_score"]) for r in rep_rows])
    ax.set_yticks(range(len(rep_rows)), [r["check_id"].replace("_", " ") for r in rep_rows], fontsize=8)
    ax.set_xlim(0, 1); ax.set_xlabel("readiness score")
    ax.set_title("Reproduction readiness ladder")
    caption = "Seeded fields are partly filled by the scaffold; source-run command, seed, hashes, model revision, and artifact index remain binding gaps."
    annotate_caption(fig, caption); fig.tight_layout(rect=(0, 0.07, 1, 1))
    bench.save_figure(ctx, fig, "reproduction_readiness_ladder.png", "Reproduction readiness.")
    manifest.append(manifest_entry(ctx, "reproduction_readiness_ladder.png", source_rel, n, "readiness score", "reproducibility gap audit", caption))

    # 9. Claim risk register
    risk_rows = list(rows["claim_risk_register"])
    source_rel, n = write_figure_source(ctx, "claim_risk_register.png", risk_rows, "Source table for claim risk register.")
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(range(len(risk_rows)), [float(r["severity_1_to_5"]) for r in risk_rows])
    ax.set_yticks(range(len(risk_rows)), [r["risk_key"].replace("_", " ") for r in risk_rows], fontsize=8)
    ax.set_xlim(0, 5.2); ax.set_xlabel("severity 1 to 5")
    ax.set_title("Claim-risk register")
    caption = "Named risks are review targets, not resolved issues. The catch artifact column says where each risk should be inspected."
    annotate_caption(fig, caption); fig.tight_layout(rect=(0, 0.07, 1, 1))
    bench.save_figure(ctx, fig, "claim_risk_register.png", "Claim-risk register.")
    manifest.append(manifest_entry(ctx, "claim_risk_register.png", source_rel, n, "risk severity", "risk visibility", caption))

    # 10. Binding gap matrix
    bind_rows = list(rows["result_binding_template"])
    source_rel, n = write_figure_source(ctx, "binding_gap_matrix.png", bind_rows, "Source table for binding gap matrix.")
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.barh(range(len(bind_rows)), [float(r["status_score"]) for r in bind_rows])
    ax.set_yticks(range(len(bind_rows)), [r["field"].replace("_", " ") for r in bind_rows], fontsize=8)
    ax.set_xlim(0, 1); ax.set_xlabel("binding completeness score")
    ax.set_title("Source-run binding gap matrix")
    caption = "The final paper should not cite a source result until the corresponding binding field has a value and artifact path."
    annotate_caption(fig, caption); fig.tight_layout(rect=(0, 0.07, 1, 1))
    bench.save_figure(ctx, fig, "binding_gap_matrix.png", "Source binding gap matrix.")
    manifest.append(manifest_entry(ctx, "binding_gap_matrix.png", source_rel, n, "binding completeness score", "binding gap visibility", caption))

    # 11. Paired examples
    pair_rows = [
        {"sentence_type": "allowed_after_binding", "text": track.allowed_claim, "status": "template_waiting_for_source_values"},
        {"sentence_type": "forbidden", "text": track.forbidden_claim, "status": "blocked"},
        {"sentence_type": "negative_result", "text": track.negative_result_sentence, "status": "allowed_if_controls_fail"},
        {"sentence_type": "scaffold_claim", "text": "This package generated a reviewable scaffold; source-run evidence remains pending.", "status": "supported_now"},
    ]
    source_rel, n = write_figure_source(ctx, "paired_examples.png", pair_rows, "Source table for paired example claim language.")
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.axis("off")
    ax.set_title("Claim-language pairs: allowed, forbidden, negative, scaffold", loc="left", fontsize=13, fontweight="bold")
    y = 0.9
    for row in pair_rows:
        ax.add_patch(patches.Rectangle((0.02, y - 0.105), 0.96, 0.10, fill=False, linewidth=0.8))
        ax.text(0.04, y - 0.025, row["sentence_type"].replace("_", " "), fontsize=10, fontweight="bold", va="top")
        text = row["text"]
        wrapped = "\n".join([text[i:i+120] for i in range(0, len(text), 120)])
        ax.text(0.28, y - 0.025, wrapped, fontsize=8.5, va="top")
        ax.text(0.86, y - 0.025, row["status"].replace("_", " "), fontsize=8.5, va="top")
        y -= 0.18
    caption = "The scaffold supports only the scaffold sentence now. Source-result and negative-result sentences need bound source artifacts."
    annotate_caption(fig, caption); fig.tight_layout(rect=(0, 0.05, 1, 1))
    bench.save_figure(ctx, fig, "paired_examples.png", "Claim-language paired examples.")
    manifest.append(manifest_entry(ctx, "paired_examples.png", source_rel, n, "sentence status", "claim-language discipline", caption))

    write_plot_manifest(ctx, manifest)


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    tracks, schema_rows, data_info = load_tracks(ctx)
    track, select_note = select_track(ctx.args, tracks)
    path = ctx.path("diagnostics", "data_manifest.json"); bench.write_json(path, data_info); ctx.register_artifact(path, "diagnostic", "Lab 35 seed-track data manifest.")
    schema_diag = ctx.path("diagnostics", "seed_track_schema_audit.csv"); bench.write_csv_with_context(ctx, schema_diag, schema_rows); ctx.register_artifact(schema_diag, "diagnostic", "Seed-track schema audit.")
    prompt = f"Capstone package audit: {track.research_question}"
    hook_check = bench.run_hook_parity_check(ctx, bundle, prompt)
    first = bench.run_with_residual_cache(bundle, prompt)
    lens_check = bench.run_lens_self_check(ctx, bundle, first)
    patch_noop = bench.run_patch_noop_check(ctx, bundle, prompt)
    rows = build_rows(track, tracks, schema_rows, data_info)
    validation, validation_rows = package_validation(track, rows, data_info)
    warnings = warning_rows(track, data_info, validation)
    write_tables(ctx, rows, validation_rows)
    write_failure_cards(ctx, rows["failure_specimens"])
    write_warning_artifacts(ctx, warnings)
    write_markdown_files(ctx, track, validation, data_info, select_note)
    write_run_summary(ctx, track, validation, data_info, select_note)
    write_status_and_state(ctx, track, data_info, validation, validation_rows, hook_check, lens_check, patch_noop, warnings)
    metrics = {"selected_track": dataclasses.asdict(track), "selection_note": select_note, "validation": validation, "data": data_info, "warnings": warnings, "science_ready": False, "evidence_rung": "AUDIT + FORMAL scaffold; source rung inherited after frozen run binding"}
    metrics_path = ctx.path("metrics.json"); bench.write_json(metrics_path, metrics); ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 35 metrics.")
    write_claims(ctx, track, validation)
    write_plots(ctx, track, rows, validation)
    print(f"[lab35] generated capstone scaffold for {track.track_id}; science_ready=false until source run is bound")
