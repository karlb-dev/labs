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


def artifact_checklist(track: CapstoneTrack) -> list[dict[str, Any]]:
    rows = [{
        "artifact": a, "category": "capstone_package", "status": "generated_by_lab35" if not a.startswith("plots/") else "generated_unless_no_plots",
        "required_before_submission": True, "failure_if_missing": "package_incomplete",
    } for a in PACKAGE_ARTIFACTS]
    rows += [{
        "artifact": a, "category": "source_run_artifact", "status": "pending_student_frozen_run_binding",
        "required_before_submission": True, "failure_if_missing": "source_evidence_gap",
    } for a in track.planned_artifacts]
    return rows


def evidence_matrix(track: CapstoneTrack) -> list[dict[str, Any]]:
    return [
        {"claim_component": "research_question_preregistered", "evidence_rung": "FORMAL", "artifact": "preregistration.md", "status": "generated_seed_scaffold", "primary_check": track.research_question, "falsifier": "paper question changes after results without repair log"},
        {"claim_component": "source_run_immutable", "evidence_rung": "AUDIT", "artifact": "tables/result_binding_template.csv", "status": "pending_student_frozen_run_binding", "primary_check": track.frozen_run_command, "falsifier": "original run is missing, overwritten, or replaced by repair run"},
        {"claim_component": "metric_declared", "evidence_rung": "FORMAL", "artifact": "preregistration.md", "status": "generated_seed_scaffold", "primary_check": track.primary_metric, "falsifier": "paper switches to a secondary metric without disclosure"},
        {"claim_component": "controls_attack_shortcuts", "evidence_rung": "AUDIT", "artifact": "tables/control_falsifier_matrix.csv", "status": "generated_seed_scaffold_pending_results", "primary_check": "; ".join(track.controls), "falsifier": "a cheaper explanation is untested or hidden"},
        {"claim_component": "failure_modes_visible", "evidence_rung": "AUDIT", "artifact": "tables/failure_modes_contribution.csv", "status": "generated_seed_scaffold_pending_results", "primary_check": "; ".join(track.expected_failure_modes), "falsifier": "counterexamples or negative controls are omitted"},
        {"claim_component": "claim_ceiling_respected", "evidence_rung": "AUDIT + FORMAL", "artifact": "claim_card.md", "status": "generated_seed_scaffold", "primary_check": track.evidence_rung_ceiling, "falsifier": track.forbidden_claim},
        {"claim_component": "human_review_accounted", "evidence_rung": "AUDIT", "artifact": "tables/human_review_queue.csv", "status": "pending_human_review" if track.human_review_required else "not_required_but_available", "primary_check": str(track.human_review_required), "falsifier": "publication claim made before required human review"},
        {"claim_component": "repair_run_limited", "evidence_rung": "FORMAL + AUDIT", "artifact": "tables/repair_log.csv", "status": "generated_seed_scaffold", "primary_check": track.stopping_rule, "falsifier": "repair run replaces frozen run instead of being compared to it"},
    ]


def control_falsifier_matrix(track: CapstoneTrack) -> list[dict[str, Any]]:
    n = max(len(track.controls), len(track.falsifiers), len(track.expected_failure_modes))
    rows = []
    for i in range(n):
        rows.append({
            "row_id": f"{track.track_id}-CF{i+1}",
            "control": track.controls[i] if i < len(track.controls) else "",
            "falsifier": track.falsifiers[i] if i < len(track.falsifiers) else "",
            "expected_failure_mode": track.expected_failure_modes[i] if i < len(track.expected_failure_modes) else "",
            "source_artifact": track.planned_artifacts[i % len(track.planned_artifacts)] if track.planned_artifacts else "",
            "student_result": "", "claim_effect": "pending: supports | narrows | kills | unrelated", "reviewer_notes": "",
        })
    return rows


def claim_language_audit(track: CapstoneTrack) -> list[dict[str, Any]]:
    return [
        {"sentence_type": "allowed_claim", "claim_language": track.allowed_claim, "status": "template_allowed_after_numbers_bound", "evidence_ceiling": track.evidence_rung_ceiling},
        {"sentence_type": "forbidden_claim", "claim_language": track.forbidden_claim, "status": "blocked", "evidence_ceiling": track.evidence_rung_ceiling},
        {"sentence_type": "negative_result_sentence", "claim_language": track.negative_result_sentence, "status": "allowed_if_controls_fail", "evidence_ceiling": "AUDIT"},
        {"sentence_type": "ledger_template", "claim_language": track.claim_ledger_template, "status": "edit_before_append_ledger", "evidence_ceiling": track.evidence_rung_ceiling},
    ]


def review_rubric_rows() -> list[dict[str, Any]]:
    return [{"rubric_area": k, "weight_percent": w, "review_question": q, "student_score": "", "reviewer_notes": ""} for k, w, q in RUBRIC]


def human_review_queue(track: CapstoneTrack) -> list[dict[str, Any]]:
    base = [
        ("claim_card", "Does the final claim stay below the evidence ceiling?"),
        ("abstract", "Does the abstract imply the forbidden claim?"),
        ("control_results", "Are failed controls reported in the main text?"),
        ("negative_results", "Is the negative-result interpretation explicit?"),
        ("repair_accounting", "Does any repair run preserve the original frozen run?"),
    ]
    return [{"review_item": item, "required": track.human_review_required or item in {"claim_card", "repair_accounting"}, "question": q, "student_answer": "", "reviewer_decision": "pending: approve | revise | reject", "reviewer_notes": ""} for item, q in base]


def repair_log(track: CapstoneTrack) -> list[dict[str, Any]]:
    return [
        {"phase": "preregistration", "status": "generated_seed_scaffold", "allowed_action": "write question, metric, controls, falsifiers, stopping rule, and claim ceiling before reading source-run results", "notes": track.research_question},
        {"phase": "frozen_run", "status": "pending_student_run", "allowed_action": "run the source-lab command once and keep the original directory immutable", "notes": track.frozen_run_command},
        {"phase": "adversarial_review", "status": "generated_review_questions_pending_answers", "allowed_action": "attack instrumentation, leakage, controls, language, power, and safety", "notes": "Review is evidence, not theater."},
        {"phase": "repair_run", "status": "not_run", "allowed_action": "at most one repair run for named instrumentation or missing-control defect", "notes": track.stopping_rule},
        {"phase": "final_claim", "status": "pending_student_binding", "allowed_action": "write the smallest claim that survives review", "notes": track.allowed_claim},
    ]


def failure_mode_rows(track: CapstoneTrack) -> list[dict[str, Any]]:
    return [{"failure_mode_id": f"{track.track_id}-FM{i}", "source_lab": track.source_lab, "failure_mode": m, "how_to_trigger": track.controls[(i - 1) % len(track.controls)], "how_to_report": "state whether it killed, narrowed, or required repair", "student_observed_example": ""} for i, m in enumerate(track.expected_failure_modes, 1)]


def reproduction_checklist(track: CapstoneTrack, data_info: Mapping[str, Any]) -> list[dict[str, Any]]:
    items = [
        ("course_commit_or_archive", "Record git SHA, dirty state, or zip hash."),
        ("source_run_command", track.frozen_run_command),
        ("tier_a_smoke_command", track.tier_a_command),
        ("seed_track_data_hash", str(data_info.get("data_sha256", ""))),
        ("source_dataset", track.dataset),
        ("model_and_revision", track.model),
        ("random_seed", "Record seed from source run_config.json."),
        ("artifact_index", "Attach artifact_index.json from source and capstone runs."),
        ("human_review", "Fill review queue if required." if track.human_review_required else "Document why human review is not required."),
    ]
    return [{"check_id": k, "required": True, "status": "seeded" if k in {"seed_track_data_hash", "source_dataset"} else "pending_student_binding", "evidence_needed": v, "student_value": "", "reviewer_notes": ""} for k, v in items]


def result_binding_template(track: CapstoneTrack) -> list[dict[str, Any]]:
    fields = [
        ("frozen_run_dir", "Path to immutable source-lab run directory."),
        ("frozen_run_artifact_index", "Path to artifact_index.json."),
        ("frozen_run_command", track.frozen_run_command),
        ("source_lab", track.source_lab),
        ("model_id", track.model),
        ("model_revision", "Pinned model revision or run_config default."),
        ("data_path", track.dataset),
        ("data_sha256", "Dataset digest from source run manifest."),
        ("seed", "Seed from source run_config.json."),
        ("primary_metric_value", track.primary_metric),
        ("strongest_control_value", "Strongest control result backing specificity."),
        ("repair_run_dir", "Only if repair run was used."),
        ("release_package_hash", "Hash of final zipped package."),
    ]
    return [{"field": k, "description": v, "student_value": "", "required_before_final_claim": True} for k, v in fields]


def preregistration_drift_audit(track: CapstoneTrack) -> list[dict[str, Any]]:
    return [
        {"field": "research_question", "preregistered_value": track.research_question, "paper_value": "", "drift_status": "pending", "allowed_change": "narrowing only or repair-log entry required"},
        {"field": "primary_metric", "preregistered_value": track.primary_metric, "paper_value": "", "drift_status": "pending", "allowed_change": "do not switch primary metric after results"},
        {"field": "controls", "preregistered_value": "; ".join(track.controls), "paper_value": "", "drift_status": "pending", "allowed_change": "add through repair; do not delete failed controls"},
        {"field": "allowed_claim", "preregistered_value": track.allowed_claim, "paper_value": "", "drift_status": "pending", "allowed_change": "narrowing allowed"},
        {"field": "forbidden_claim", "preregistered_value": track.forbidden_claim, "paper_value": "", "drift_status": "pending", "allowed_change": "never remove boundary"},
    ]


def plot_guide() -> list[dict[str, Any]]:
    return [
        {"plot": "capstone_dashboard.png", "first_question": "Is the package complete enough to review?", "non_claim": "Completeness is not scientific validity."},
        {"plot": "artifact_contract_status.png", "first_question": "Which artifacts are generated versus pending source-run binding?", "non_claim": "Pending source-run artifacts cannot support final claims."},
        {"plot": "evidence_rung_matrix.png", "first_question": "Which components are FORMAL, AUDIT, or inherited?", "non_claim": "A bright cell does not raise the evidence ceiling."},
        {"plot": "review_score_radar.png", "first_question": "Which rubric areas are weakest before review?", "non_claim": "Seed scores are placeholders."},
        {"plot": "control_falsifier_map.png", "first_question": "What controls can kill the favorite claim?", "non_claim": "Control count is not control quality."},
        {"plot": "failure_mode_atlas.png", "first_question": "Which failure modes must be reported?", "non_claim": "Atlas rows need concrete examples."},
        {"plot": "reproduction_readiness_ladder.png", "first_question": "Which reproducibility fields remain unbound?", "non_claim": "A scaffold cannot reproduce a source run by itself."},
    ]


def build_rows(track: CapstoneTrack, tracks: Sequence[CapstoneTrack], schema_rows: Sequence[Mapping[str, Any]], data_info: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
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
        "plot_reading_guide": plot_guide(),
    }


def package_validation(track: CapstoneTrack, rows: Mapping[str, Sequence[Mapping[str, Any]]], data_info: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checks = [
        ("rubric_weights_sum_to_100", sum(w for _, w, _ in RUBRIC) == 100, "Review weights must sum to 100."),
        ("seed_data_hash_available", bool(data_info.get("data_sha256")), "Seed data must have a hash."),
        ("seed_schema_ok", all(r.get("schema_ok") for r in rows["seed_track_schema_audit"]), "Seed-track rows must pass schema checks."),
        ("track_has_controls", len(track.controls) >= 3, "Track needs multiple controls."),
        ("track_has_falsifiers", len(track.falsifiers) >= 2, "Track needs falsifiers."),
        ("forbidden_claim_present", bool(track.forbidden_claim), "Claim boundary must be explicit."),
        ("stopping_rule_present", bool(track.stopping_rule), "Stopping rule must be explicit."),
        ("evidence_matrix_nonempty", len(rows["evidence_matrix"]) >= 6, "Evidence matrix must cover formal/audit/binding components."),
        ("human_review_queue_present", bool(rows["human_review_queue"]), "Review queue must exist."),
        ("source_run_binding_pending", True, "Source run still must be bound before science claims."),
    ]
    validation_rows = [{"check": k, "ok": bool(ok), "description": d} for k, ok, d in checks]
    package_ready = all(ok for k, ok, _ in checks if k != "source_run_binding_pending")
    validation = {
        "track_id": track.track_id,
        "package_ready_for_student_replacement": package_ready,
        "science_ready": False,
        "why_not_science_ready": "Lab 35 generated a scaffold; bind a frozen source-lab run before final claims.",
        "rubric_weight_total": sum(w for _, w, _ in RUBRIC),
        "table_counts": {k: len(v) for k, v in rows.items()},
        "n_controls": len(track.controls), "n_falsifiers": len(track.falsifiers),
        "n_failure_modes": len(track.expected_failure_modes),
        "human_review_required": track.human_review_required,
        "checks_passed": sum(1 for _, ok, _ in checks if ok), "checks_total": len(checks),
    }
    return validation, validation_rows


def write_csv_table(ctx: bench.RunContext, rel: str, rows: Sequence[Mapping[str, Any]], desc: str) -> None:
    path = ctx.path(*rel.split("/"))
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", desc)


def write_tables(ctx: bench.RunContext, rows: Mapping[str, Sequence[Mapping[str, Any]]], validation_rows: Sequence[Mapping[str, Any]]) -> None:
    specs = [
        ("tables/track_options.csv", "track_options", "Available seed tracks and selected-track marker."),
        ("tables/seed_track_schema_audit.csv", "seed_track_schema_audit", "Seed-track schema audit."),
        ("tables/artifact_checklist.csv", "artifact_checklist", "Generated and pending artifacts."),
        ("tables/evidence_matrix.csv", "evidence_matrix", "Claim-component evidence matrix."),
        ("tables/control_falsifier_matrix.csv", "control_falsifier_matrix", "Controls, falsifiers, and expected failure modes."),
        ("tables/claim_language_audit.csv", "claim_language_audit", "Allowed, forbidden, negative, and ledger claim language."),
        ("tables/review_rubric.csv", "review_rubric", "Weighted review rubric."),
        ("tables/human_review_queue.csv", "human_review_queue", "Human review queue."),
        ("tables/repair_log.csv", "repair_log", "Frozen-run and repair-run accounting."),
        ("tables/failure_modes_contribution.csv", "failure_modes_contribution", "Failure-mode atlas contribution."),
        ("tables/reproduction_checklist.csv", "reproduction_checklist", "Reproducibility checklist."),
        ("tables/result_binding_template.csv", "result_binding_template", "Fields students fill from the frozen source run."),
        ("tables/preregistration_drift_audit.csv", "preregistration_drift_audit", "Preregistration-vs-paper drift audit."),
        ("tables/plot_reading_guide.csv", "plot_reading_guide", "Plot reading guide."),
    ]
    for rel, key, desc in specs:
        write_csv_table(ctx, rel, rows[key], desc)
    write_csv_table(ctx, "tables/package_validation.csv", validation_rows, "Package validation checks.")
    results = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results, rows["evidence_matrix"])
    ctx.register_artifact(results, "table", "Alias of tables/evidence_matrix.csv.")


def write_text_file(ctx: bench.RunContext, rel: str, lines: Sequence[str], kind: str, desc: str) -> None:
    path = ctx.path(rel)
    bench.write_text(path, "\n".join(lines).rstrip() + "\n")
    ctx.register_artifact(path, kind, desc)


def write_markdown_files(ctx: bench.RunContext, track: CapstoneTrack, validation: Mapping[str, Any], data_info: Mapping[str, Any], select_note: str) -> None:
    write_text_file(ctx, "preregistration.md", [
        "# Capstone preregistration", "", f"Track: `{track.track_id}`", f"Title: {track.title}", f"Source lab: `{track.source_lab}`", f"Evidence ceiling: `{track.evidence_rung_ceiling}`", "", "## Research question", "", track.research_question, "", "## Dataset, model, and measurement sites", "", f"- dataset: `{track.dataset}`", f"- model: {track.model}", f"- measurement sites: {track.measurement_sites}", "", "## Primary metric", "", track.primary_metric, "", "## Secondary metrics", "", *bullets(track.secondary_metrics), "", "## Controls", "", *bullets(track.controls), "", "## Falsifiers", "", *bullets(track.falsifiers), "", "## Stopping rule", "", track.stopping_rule, "", "## Allowed claim", "", track.allowed_claim, "", "## Forbidden claim", "", track.forbidden_claim, "", "## Expected failure modes", "", *bullets(track.expected_failure_modes), "", "## Safety scope", "", track.safety_scope, ""], "summary", "Capstone preregistration.")
    write_text_file(ctx, "paper.md", [
        "# Capstone paper draft", "", "## Abstract", "", "This draft is not a result yet. Bind the frozen source run, report controls, and update the claim only after review.", "", "## Question", "", track.research_question, "", "## Methods", "", f"The package uses `{track.dataset}` and the model/run specified in `tables/result_binding_template.csv`.", "", "## Primary metric", "", track.primary_metric, "", "## Results from frozen run", "", "Binding required: paste measured source-run values and exact artifact paths here. Do not delete failed controls.", "", "## Controls and counterexamples", "", *bullets(track.controls), "", "## Negative-result path", "", track.negative_result_sentence, "", "## Claim", "", f"Allowed only after source-run binding: {track.allowed_claim}", "", "## Boundary", "", f"This paper must not imply: {track.forbidden_claim}", "", "## Outline checkpoints", "", *bullets(track.paper_outline), ""], "summary", "Capstone paper draft.")
    write_text_file(ctx, "claim_card.md", [
        "# Claim card", "", f"- selected track: `{track.track_id}`", f"- source lab: `{track.source_lab}`", f"- evidence ceiling: `{track.evidence_rung_ceiling}`", "- capstone evidence: `AUDIT + FORMAL` until the source run is bound", f"- human review required: `{str(track.human_review_required).lower()}`", "", "## Allowed claim", "", track.allowed_claim, "", "## Forbidden claim", "", track.forbidden_claim, "", "## Ledger draft", "", "```text", track.claim_ledger_template, "```", ""], "summary", "Claim card.")
    write_text_file(ctx, "adversarial_review.md", [
        "# Adversarial review", "", "Answer each section before finalizing the paper.", "", "## Instrumentation", "", "Which hook, tokenization, cache, dtype, split, score, or schema bug would invalidate the result?", "", "## Data leakage and shortcuts", "", f"Could `{track.dataset}` leak the answer or label through ordering, wording, token overlap, template cues, or split contamination?", "", "## Controls to attack", "", *bullets(track.controls), "", "## Falsifiers to make true", "", *bullets(track.falsifiers), "", "## Claim language", "", f"Does any sentence imply the forbidden claim: {track.forbidden_claim}", "", "## Safety", "", track.safety_scope, ""], "summary", "Adversarial review template.")
    write_text_file(ctx, "review_response.md", [
        "# Review response", "", "## Reviewer decision", "", "Pending: approve as written | narrow claim | run one repair | unsupported.", "", "## Repair run decision", "", track.stopping_rule, "", "## Claim revision", "", f"Starting allowed claim: {track.allowed_claim}", "", "## Unresolved risks", "", *bullets(track.expected_failure_modes), ""], "summary", "Review response template.")
    write_text_file(ctx, "reproduction_guide.md", [
        "# Reproduction guide", "", "```bash", "cd interpretability", "pip install -r requirements.txt", "python data/make_capstone_seed_tracks.py", track.tier_a_command, track.frozen_run_command, "python interp_bench.py --lab lab35 --tier a", "```", "", f"Seed data hash: `{data_info.get('data_sha256', '')}`", "", "## Source artifacts required", "", *[f"- `{a}`" for a in track.planned_artifacts], "", "## Release rule", "", "Do not publish until the review queue is filled and the claim card matches the evidence matrix.", ""], "summary", "Reproduction guide.")
    write_text_file(ctx, "method_card.md", [
        "# Lab 35 method card", "", "Lab 35 is a capstone package generator and validator, not a new mechanistic measurement.", "", f"- selected track: `{track.track_id}`", f"- selection note: {select_note}", f"- source lab: `{track.source_lab}`", f"- source evidence ceiling: `{track.evidence_rung_ceiling}`", "- capstone evidence: `AUDIT + FORMAL`", f"- package ready for student replacement: `{str(validation['package_ready_for_student_replacement']).lower()}`", f"- science ready: `{str(validation['science_ready']).lower()}`", "", "## What this run validated", "", "Seed schema, package artifact coverage, rubric weights, human-review fields, claim-boundary scaffolding, and shared bench self-checks.", "", "## What remains pending", "", "A real frozen source-lab run must be attached and reviewed before scientific claims are publishable.", ""], "summary", "Method card.")
    write_text_file(ctx, "operationalization_audit.md", [
        "# Operationalization audit", "", "```yaml", "headline_claim: \"a reproducible interpretability paper package is ready to defend\"", "cheap_explanation: \"the package looks complete but hides drift, failed controls, or claim inflation\"", "killer_control: \"artifact checklist, evidence matrix, repair log, review rubric, and claim-language audit all remain visible\"", "result: \"scaffold_generated_source_run_pending\"", "claim_allowed: \"package scaffold only\"", "```", "", "## Controls", "", *bullets(track.controls), "", "## Falsifiers", "", *bullets(track.falsifiers), "", "## Claim boundary", "", track.forbidden_claim, ""], "summary", "Operationalization audit.")
    write_text_file(ctx, "package_readiness_report.md", ["# Package readiness report", "", f"- package_ready_for_student_replacement: `{str(validation['package_ready_for_student_replacement']).lower()}`", f"- science_ready: `{str(validation['science_ready']).lower()}`", f"- why: {validation['why_not_science_ready']}", f"- checks: {validation['checks_passed']} / {validation['checks_total']}", "", "## Remaining steps", "", "1. Run and freeze the source lab.", "2. Fill `tables/result_binding_template.csv`.", "3. Answer the adversarial review.", "4. Log any repair run separately.", "5. Ensure the claim card does not exceed the evidence ceiling.", ""], "summary", "Package readiness report.")
    write_text_file(ctx, "negative_result_appendix.md", ["# Negative-result appendix", "", track.negative_result_sentence, "", "## Failure modes to report", "", *bullets(track.expected_failure_modes), ""], "summary", "Negative-result appendix.")
    write_text_file(ctx, "public_release_checklist.md", ["# Public release checklist", "", "- Claim card evidence rung does not exceed source lab.", "- Original frozen run remains visible.", "- Failed controls remain visible.", "- Required human-review fields are filled.", "- Safety scope is explicit.", "- Release zip hash is recorded.", ""], "summary", "Public release checklist.")


def write_run_summary(ctx: bench.RunContext, track: CapstoneTrack, validation: Mapping[str, Any], data_info: Mapping[str, Any], select_note: str) -> None:
    write_text_file(ctx, "run_summary.md", [
        "# Lab 35 run summary: reproducible interpretability paper capstone", "", f"- selected track: `{track.track_id}` ({select_note})", f"- source lab: `{track.source_lab}`", f"- evidence ceiling: `{track.evidence_rung_ceiling}`", f"- seed tracks selected: {data_info.get('n_rows_selected')} from `{pathlib.Path(str(data_info.get('data_path', ''))).name}`", f"- package_ready_for_student_replacement: `{str(validation['package_ready_for_student_replacement']).lower()}`", f"- science_ready: `{str(validation['science_ready']).lower()}`", "- smallest surviving claim: the package scaffold is reproducible and reviewable; no source-lab science claim is made yet", "", "## Why science_ready is false", "", validation["why_not_science_ready"], "", "## Reading order", "", "1. `method_card.md`", "2. `preregistration.md`", "3. `tables/result_binding_template.csv`", "4. `tables/evidence_matrix.csv`", "5. `adversarial_review.md`", "6. `review_response.md`", "7. `paper.md`", "8. `reproduction_guide.md`", "9. `claim_card.md`", ""], "summary", "Run summary.")


def write_status_and_state(ctx: bench.RunContext, track: CapstoneTrack, data_info: Mapping[str, Any], validation: Mapping[str, Any], validation_rows: Sequence[Mapping[str, Any]], hook_check: Mapping[str, Any], lens_check: Mapping[str, Any], patch_noop: Mapping[str, Any]) -> None:
    safety = {"lab": LAB_ID, "selected_track": track.track_id, "source_lab": track.source_lab, "safety_scope": track.safety_scope, "human_review_required": track.human_review_required, "unsafe_generation": False, "model_editing_performed": False, "external_tool_side_effects": False, "science_ready": False, "blocked_claim": track.forbidden_claim}
    path = ctx.path("diagnostics", "safety_status.json"); bench.write_json(path, safety); ctx.register_artifact(path, "diagnostic", "Safety and claim-boundary status.")
    self_check = {"hook_parity_ok": bool(hook_check.get("ok")), "lens_self_check_ok": bool(lens_check.get("ok")), "patch_noop_ok": bool(patch_noop.get("ok")), "seed_manifest_ok": data_info.get("manifest_ok"), "package_validation_checks_passed": validation.get("checks_passed"), "package_validation_checks_total": validation.get("checks_total"), "package_ready_for_student_replacement": validation.get("package_ready_for_student_replacement"), "science_ready": False, "source_run_binding_required": True}
    path = ctx.path("diagnostics", "self_check_status.json"); bench.write_json(path, self_check); ctx.register_artifact(path, "diagnostic", "Aggregated self-check status.")
    path = ctx.path("diagnostics", "package_validation.json"); bench.write_json(path, {"summary": validation, "checks": list(validation_rows)}); ctx.register_artifact(path, "diagnostic", "Package validation summary.")
    path = ctx.path("diagnostics", "frozen_run_binding_status.json"); bench.write_json(path, {"source_run_bound": False, "binding_table": "tables/result_binding_template.csv", "source_lab": track.source_lab, "frozen_run_command": track.frozen_run_command}); ctx.register_artifact(path, "diagnostic", "Explicit status that source run is not bound yet.")
    path = ctx.path("state", "selected_track.json"); bench.write_json(path, dataclasses.asdict(track)); ctx.register_artifact(path, "state", "Selected seed track.")
    path = ctx.path("state", "capstone_package_manifest.json"); bench.write_json(path, {"lab": LAB_ID, "selected_track": track.track_id, "validation": validation, "data": dict(data_info), "required_package_artifacts": PACKAGE_ARTIFACTS, "pending_source_artifacts": track.planned_artifacts}); ctx.register_artifact(path, "state", "Capstone scaffold manifest.")


def write_claims(ctx: bench.RunContext, track: CapstoneTrack, validation: Mapping[str, Any]) -> None:
    claims = [{"id": f"{LAB_ID}-C1", "tag": "AUDIT,FORMAL", "text": f"Lab 35 generated a reproducible capstone scaffold for `{track.track_id}` with {validation['checks_passed']}/{validation['checks_total']} package checks passing. This is a scaffold-readiness claim only; source-lab scientific claims require frozen-run binding.", "artifact": f"runs/{ctx.run_dir.name}/package_readiness_report.md", "falsifier": "Required package artifacts are missing, rubric weights do not sum to 100, or the final claim exceeds the source lab evidence ceiling."}]
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)


def write_plots(ctx: bench.RunContext, track: CapstoneTrack, rows: Mapping[str, Sequence[Mapping[str, Any]]], validation: Mapping[str, Any]) -> None:
    if ctx.args.no_plots:
        return
    import matplotlib.pyplot as plt
    import numpy as np
    counts = validation["table_counts"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5)); fig.suptitle("Lab 35 capstone package dashboard", fontsize=14, fontweight="bold")
    keys = list(counts); axes[0,0].bar(range(len(keys)), [counts[k] for k in keys]); axes[0,0].set_xticks(range(len(keys)), keys, rotation=35, ha="right"); axes[0,0].set_title("Generated table rows")
    areas = [r["rubric_area"] for r in rows["review_rubric"]]; weights = [float(r["weight_percent"]) for r in rows["review_rubric"]]; axes[0,1].bar(range(len(areas)), weights); axes[0,1].set_xticks(range(len(areas)), areas, rotation=35, ha="right"); axes[0,1].set_title("Rubric weights")
    axes[1,0].bar(["controls", "falsifiers", "failures"], [len(track.controls), len(track.falsifiers), len(track.expected_failure_modes)]); axes[1,0].set_title("Pressure against favorite claim")
    flags = ["package_ready", "science_ready", "human_review", "source_bound"]; vals = [1 if validation["package_ready_for_student_replacement"] else 0, 0, 1 if track.human_review_required else 0, 0]; axes[1,1].bar(range(len(flags)), vals); axes[1,1].set_xticks(range(len(flags)), flags, rotation=25, ha="right"); axes[1,1].set_ylim(0,1.05); axes[1,1].set_title("Readiness flags"); fig.tight_layout(rect=(0,0,1,0.95)); bench.save_figure(ctx, fig, "capstone_dashboard.png", "Lab 35 dashboard.")
    cats = Counter(r["category"] for r in rows["artifact_checklist"]); stats = Counter(r["status"] for r in rows["artifact_checklist"]); labels = list(cats) + list(stats); vals = [cats[k] for k in cats] + [stats[k] for k in stats]; fig, ax = plt.subplots(figsize=(9,4.8)); ax.bar(range(len(labels)), vals); ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right"); ax.set_ylabel("count"); ax.set_title("Artifact readiness matrix"); fig.tight_layout(); bench.save_figure(ctx, fig, "artifact_contract_status.png", "Artifact readiness matrix.")
    ev = rows["evidence_matrix"]; rungs = sorted({r["evidence_rung"] for r in ev}); comps = [r["claim_component"] for r in ev]; mat = np.zeros((len(rungs), len(comps)))
    for j, row in enumerate(ev): mat[rungs.index(row["evidence_rung"]), j] = 1
    fig, ax = plt.subplots(figsize=(10,4.8)); im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1); ax.set_yticks(range(len(rungs)), rungs); ax.set_xticks(range(len(comps)), comps, rotation=35, ha="right"); ax.set_title("Evidence rung matrix"); fig.colorbar(im, ax=ax, shrink=.8); fig.tight_layout(); bench.save_figure(ctx, fig, "evidence_rung_matrix.png", "Evidence rung matrix.")
    scores = [0.75 if r["rubric_area"] in {"instrument_validity", "reproducibility", "safety_scope"} else 0.65 for r in rows["review_rubric"]]; angles = np.linspace(0, 2*np.pi, len(scores), endpoint=False).tolist(); fig = plt.figure(figsize=(6.2,6.2)); ax = fig.add_subplot(111, polar=True); ax.plot(angles + angles[:1], scores + scores[:1]); ax.fill(angles + angles[:1], scores + scores[:1], alpha=.18); ax.set_xticks(angles, areas, fontsize=8); ax.set_ylim(0,1); ax.set_title("Seed review score radar"); fig.tight_layout(); bench.save_figure(ctx, fig, "review_score_radar.png", "Review rubric weights.")
    cf = rows["control_falsifier_matrix"]; y = np.arange(len(cf)); fig, ax = plt.subplots(figsize=(9.5, max(4.5, .45*len(cf)+1.5))); ax.barh(y-.18, [1 if r.get("control") else 0 for r in cf], height=.3, label="control"); ax.barh(y+.18, [1 if r.get("falsifier") else 0 for r in cf], height=.3, label="falsifier"); ax.set_yticks(y, [r["row_id"].replace(track.track_id+"-", "") for r in cf]); ax.set_xlim(0,1.2); ax.set_title("Control falsifier dashboard"); ax.legend(); fig.tight_layout(); bench.save_figure(ctx, fig, "control_falsifier_map.png", "Control falsifier dashboard.")
    fm = rows["failure_modes_contribution"]; fig, ax = plt.subplots(figsize=(9.5, max(4.5, .5*len(fm)+1.5))); ax.barh([r["failure_mode_id"].replace(track.track_id+"-", "") for r in fm], list(range(1,len(fm)+1))); ax.set_xlabel("atlas row index"); ax.set_title("Failure-mode atlas"); fig.tight_layout(); bench.save_figure(ctx, fig, "failure_mode_atlas.png", "Failure-mode atlas.")
    statuses = Counter(r["status"] for r in rows["reproduction_checklist"]); fig, ax = plt.subplots(figsize=(8.5,4.8)); ax.bar(list(statuses), list(statuses.values())); ax.set_ylabel("check count"); ax.set_title("Reproduction readiness"); fig.tight_layout(); bench.save_figure(ctx, fig, "reproduction_readiness_ladder.png", "Reproduction readiness.")


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
    write_tables(ctx, rows, validation_rows)
    write_markdown_files(ctx, track, validation, data_info, select_note)
    write_run_summary(ctx, track, validation, data_info, select_note)
    write_status_and_state(ctx, track, data_info, validation, validation_rows, hook_check, lens_check, patch_noop)
    metrics = {"selected_track": dataclasses.asdict(track), "selection_note": select_note, "validation": validation, "data": data_info, "science_ready": False, "evidence_rung": "AUDIT + FORMAL scaffold; source rung inherited after frozen run binding"}
    metrics_path = ctx.path("metrics.json"); bench.write_json(metrics_path, metrics); ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 35 metrics.")
    write_claims(ctx, track, validation)
    write_plots(ctx, track, rows, validation)
    print(f"[lab35] generated capstone scaffold for {track.track_id}; science_ready=false until source run is bound")
