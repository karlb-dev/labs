"""Lab 31: Automated interpretability at scale.

The lab treats automated labels as hypotheses to audit, not as feature meanings.
It uses a frozen JSONL suite with synthetic gold labels, polysemantic examples,
hard negatives, confusable negatives, and token-overlap decoys. The default run
is fully offline: simple heuristic explainers produce labels, then the lab
scores whether those labels predict held-out activations better than controls.

Evidence level: AUDIT + DECODE, scoped to this frozen explanation test suite.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pathlib
import re
import statistics
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import interp_bench as bench

LAB_ID = "L31"
DATA_FILE = "auto_interp_feature_tasks.jsonl"
PROMPT_SET_CAPS = {"small": 10, "medium": 10, "full": 0}

DOMAIN_LEXICON = {
    "code": {"function", "json", "typed", "object", "unit", "test", "module", "developer", "loop", "exception", "script", "library", "tensor", "compiler", "bracket", "software", "algorithm"},
    "cooking": {"recipe", "herbs", "sauce", "chef", "broth", "heat", "dough", "oven", "loaf", "cook", "onions", "pan", "batter", "flour", "butter", "kitchen"},
    "finance": {"market", "earnings", "analyst", "fund", "portfolio", "rate", "bond", "liquidity", "credit", "spreads", "company", "guidance", "stock", "investors", "revenue", "margin"},
    "sports": {"striker", "scored", "pass", "midfield", "team", "lead", "minute", "coach", "play", "game", "runner", "race", "field", "goalkeeper", "shot", "league"},
    "law": {"court", "statute", "precedent", "attorney", "motion", "hearing", "judge", "evidence", "trial", "contract", "clause", "liability", "dispute", "jury", "testimony", "legal"},
    "medicine": {"clinician", "symptoms", "dosage", "hospital", "patient", "treatment", "nurse", "pulse", "breathing", "recovery", "physician", "scan", "injury", "vaccine", "fever", "clinical"},
    "weather": {"storm", "rain", "wind", "clouds", "forecast", "snow", "humid", "air", "cold", "thunder", "bright", "morning", "coast", "fog", "weathered"},
    "emotion": {"joy", "generous", "apology", "anger", "relief", "afraid", "smiled", "sadness", "friends", "support", "proud", "felt", "mood", "anxious", "fear", "happy"},
}
METHODS = ("majority_domain", "structured_local", "test_aware", "gold_calibration")
CONFUSABLE_FAILURE_THRESHOLD = 0.5
REVIEW_FIELDS = (
    "student_label_primary",
    "student_label_secondary",
    "student_confidence",
    "student_evidence_span",
    "reviewer_label",
    "agreement_status",
)


@dataclasses.dataclass
class FeatureTask:
    feature_id: str
    model: str
    layer: int
    feature_index: int
    feature_type: str
    top_contexts: list[str]
    heldout_contexts: list[str]
    negative_contexts: list[str]
    confusable_contexts: list[str]
    adversarial_contexts: list[str]
    gold_label: str | None
    gold_label_secondary: str
    expected_abstain: bool


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


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


def auc_binary(labels: Sequence[int], scores: Sequence[float]) -> float:
    pos = [float(s) for y, s in zip(labels, scores) if int(y) == 1 and math.isfinite(float(s))]
    neg = [float(s) for y, s in zip(labels, scores) if int(y) == 0 and math.isfinite(float(s))]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for ps in pos:
        for ns in neg:
            wins += 1.0 if ps > ns else 0.5 if ps == ns else 0.0
    return wins / (len(pos) * len(neg))


def data_path(args: Any) -> pathlib.Path:
    prompt_set = str(getattr(args, "prompt_set", "") or "")
    candidate = pathlib.Path(prompt_set)
    if prompt_set not in PROMPT_SET_CAPS and candidate.suffix.lower() in {".jsonl", ".json"}:
        return candidate if candidate.is_absolute() else (bench.COURSE_ROOT / candidate).resolve()
    return bench.COURSE_ROOT / "data" / DATA_FILE


def load_tasks(ctx: bench.RunContext) -> tuple[list[FeatureTask], dict[str, Any]]:
    path = data_path(ctx.args)
    tasks: list[FeatureTask] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            tasks.append(FeatureTask(**payload))
    cap = PROMPT_SET_CAPS.get(str(ctx.args.prompt_set), 0)
    if cap:
        tasks = tasks[:cap]
    if int(ctx.args.max_examples or 0) > 0:
        tasks = tasks[: int(ctx.args.max_examples)]
    info = {
        "data_path": str(path),
        "sha256": file_sha256(path),
        "n_rows_selected": len(tasks),
        "feature_types": dict(Counter(t.feature_type for t in tasks)),
        "gold_labels": dict(Counter(t.gold_label or "null" for t in tasks)),
        "science_ready": True,
        "science_scope": "offline auto-label audit with synthetic gold labels, confusables, and decoys",
    }
    return tasks, info


def words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]*", text.lower())


def domain_scores(contexts: Sequence[str]) -> dict[str, int]:
    tokens = words(" ".join(contexts))
    scores: dict[str, int] = {}
    for domain, vocab in DOMAIN_LEXICON.items():
        scores[domain] = sum(1 for token in tokens if token in vocab)
    return scores


def best_domain(contexts: Sequence[str]) -> tuple[str | None, float, str]:
    scores = domain_scores(contexts)
    total = sum(scores.values())
    if total <= 0:
        return None, 0.0, ""
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    label, score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0
    confidence = (score - second) / max(1, score)
    evidence = ", ".join(f"{k}:{v}" for k, v in ranked[:4] if v)
    return label, max(0.0, min(1.0, confidence)), evidence


def remove_key_tokens(contexts: Sequence[str], label: str | None) -> list[str]:
    if not label or label not in DOMAIN_LEXICON:
        return list(contexts)
    vocab = DOMAIN_LEXICON[label]
    out: list[str] = []
    for context in contexts:
        toks = context.split()
        kept = [tok for tok in toks if re.sub(r"[^a-zA-Z0-9_+-]", "", tok.lower()) not in vocab]
        out.append(" ".join(kept))
    return out


def generate_explanations(tasks: Sequence[FeatureTask]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    task_list = list(tasks)
    for idx, task in enumerate(task_list):
        majority_label, majority_conf, majority_evidence = best_domain(task.top_contexts)
        confusable_label, confusable_conf, _ = best_domain(task.confusable_contexts)
        control_pool = [t for t in task_list if (t.gold_label or "") != (task.gold_label or "")]
        control_task = control_pool[(idx + 3) % len(control_pool)] if control_pool else task
        shuffled_contexts = list(reversed(control_task.top_contexts))
        shuffled_label, shuffled_conf, shuffled_evidence = best_domain(shuffled_contexts)
        candidates = []
        candidates.append(("majority_domain", majority_label, majority_conf, majority_evidence, "Majority domain words in top contexts."))
        structured_conf = majority_conf
        if majority_label == confusable_label:
            structured_conf *= 0.55
        candidates.append(("structured_local", majority_label, structured_conf, majority_evidence, "Top-context explanation penalized by confusable overlap."))
        key_deleted_label, key_deleted_conf, key_deleted_evidence = best_domain(remove_key_tokens(task.top_contexts, majority_label))
        test_conf = min(majority_conf, 1.0 - key_deleted_conf if key_deleted_label != majority_label else 0.35)
        if task.expected_abstain or test_conf < 0.25:
            test_label = None
        else:
            test_label = majority_label
        candidates.append(("test_aware", test_label, test_conf, f"top={majority_evidence}; deleted={key_deleted_evidence}", "Abstains when key-token deletion or polysemantic flags undermine the label."))
        gold_conf = 0.95 if task.gold_label and not task.expected_abstain else 0.20
        gold_label = task.gold_label if task.gold_label and not task.expected_abstain else None
        candidates.append(("gold_calibration", gold_label, gold_conf, task.gold_label_secondary, "Human/gold calibration upper-bound row, not an automated method."))
        candidates.append(("shuffled_top_context_control", shuffled_label, shuffled_conf, shuffled_evidence, "Control explainer run on top contexts from a different feature."))
        for method, label, confidence, evidence, template in candidates:
            abstain = label is None or confidence < 0.25
            explanation = "ABSTAIN: insufficiently specific or polysemantic feature." if abstain else f"Feature appears to activate on {label} contexts because {template}"
            rows.append({
                "feature_id": task.feature_id,
                "feature_type": task.feature_type,
                "method": method,
                "generated_label": "" if label is None else label,
                "gold_label": task.gold_label or "",
                "expected_abstain": task.expected_abstain,
                "confidence": rounded(confidence),
                "abstain": abstain,
                "explanation": explanation,
                "evidence_terms": evidence,
                "student_label_primary": "",
                "student_label_secondary": "",
                "student_confidence": "",
                "student_evidence_span": "",
                "reviewer_label": "",
                "agreement_status": "",
            })
    return rows


def score_text_for_label(text: str, label: str) -> float:
    if not label or label not in DOMAIN_LEXICON:
        return 0.0
    toks = words(text)
    vocab = DOMAIN_LEXICON[label]
    raw = sum(1 for token in toks if token in vocab)
    return raw / max(3.0, math.sqrt(len(toks) + 1.0))


def build_tests(tasks: Sequence[FeatureTask], explanations: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    test_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    by_feature_method = {(r["feature_id"], r["method"]): r for r in explanations}
    for task in tasks:
        contexts: list[tuple[str, str, int, list[str]]] = [
            ("heldout_positive", "heldout", 1, task.heldout_contexts),
            ("hard_negative", "negative", 0, task.negative_contexts),
            ("confusable_negative", "confusable", 0, task.confusable_contexts),
            ("token_overlap_decoy", "adversarial", 0, task.adversarial_contexts),
        ]
        for method in METHODS + ("shuffled_top_context_control",):
            exp = by_feature_method[(task.feature_id, method)]
            label = str(exp["generated_label"])
            labels: list[int] = []
            scores: list[float] = []
            confusable_failures = 0
            for suite, context_kind, expected_active, context_list in contexts:
                for i, context in enumerate(context_list):
                    pred = 0.0 if exp["abstain"] else score_text_for_label(context, label)
                    labels.append(expected_active)
                    scores.append(pred)
                    if suite in {"confusable_negative", "token_overlap_decoy"} and pred >= CONFUSABLE_FAILURE_THRESHOLD:
                        confusable_failures += 1
                    test_rows.append({
                        "feature_id": task.feature_id,
                        "method": method,
                        "test_id": f"{task.feature_id}:{method}:{suite}:{i}",
                        "context_kind": context_kind,
                        "suite": suite,
                        "context": context,
                        "expected_active": expected_active,
                        "predicted_score": rounded(pred),
                        "generated_label": label,
                        "abstain": exp["abstain"],
                    })
            auc = auc_binary(labels, scores)
            positive_scores = [s for y, s in zip(labels, scores) if y == 1]
            negative_scores = [s for y, s in zip(labels, scores) if y == 0]
            precision_gap = safe_mean(positive_scores, 0.0) - safe_mean(negative_scores, 0.0)
            gold = task.gold_label or ""
            label_hit = bool(label and gold and label in gold)
            score_rows.append({
                "feature_id": task.feature_id,
                "feature_type": task.feature_type,
                "method": method,
                "generated_label": label,
                "gold_label": gold,
                "label_hit": label_hit,
                "expected_abstain": task.expected_abstain,
                "abstain": exp["abstain"],
                "confidence": exp["confidence"],
                "heldout_auc": rounded(auc),
                "precision_gap": rounded(precision_gap),
                "confusable_failure_count": confusable_failures,
                "test_count": len(labels),
                "score_posture": "abstained" if exp["abstain"] else "passes" if auc >= 0.75 and precision_gap > 0 and confusable_failures <= 1 else "fails_or_needs_review",
            })
    return test_rows, score_rows


def summarize_evidence(scores: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    methods = sorted({r["method"] for r in scores})
    for method in methods:
        rows = [r for r in scores if r["method"] == method]
        non_abstain = [r for r in rows if not r["abstain"]]
        abstained = [r for r in rows if r["abstain"]]
        labelable = [r for r in rows if not r["expected_abstain"]]
        should_abstain = [r for r in rows if r["expected_abstain"]]
        auc_mean = safe_mean([r["heldout_auc"] for r in non_abstain])
        label_acc = safe_mean([1.0 if r["label_hit"] else 0.0 for r in non_abstain], default=0.0)
        abstain_rate = len(abstained) / len(rows) if rows else 0.0
        bad_abstain_rate = safe_mean([1.0 if r["abstain"] else 0.0 for r in labelable], default=0.0)
        good_abstain_rate = safe_mean([1.0 if r["abstain"] else 0.0 for r in should_abstain], default=0.0)
        confusable_fail = safe_mean([r["confusable_failure_count"] for r in non_abstain], default=0.0)
        precision_gap = safe_mean([r["precision_gap"] for r in non_abstain])
        if method == "gold_calibration":
            claim_posture = "calibration_upper_bound"
        elif method.endswith("_control"):
            claim_posture = "control_sanity_check"
        elif auc_mean >= 0.75 and label_acc >= 0.70 and good_abstain_rate >= 0.75 and bad_abstain_rate <= 0.25 and confusable_fail <= 1.0:
            claim_posture = "auto_label_audit_supported"
        elif auc_mean >= 0.75 and label_acc >= 0.70:
            if bad_abstain_rate > 0.25:
                claim_posture = "label_predictive_but_over_abstaining"
            elif good_abstain_rate < 0.75:
                claim_posture = "label_predictive_but_abstention_limited"
            elif confusable_fail > 1.0:
                claim_posture = "label_predictive_but_confusable_limited"
            else:
                claim_posture = "needs_review_or_calibration"
        else:
            claim_posture = "needs_review_or_calibration"
        evidence.append({
            "method": method,
            "n_features": len(rows),
            "n_scored": len(non_abstain),
            "mean_heldout_auc": rounded(auc_mean),
            "mean_precision_gap": rounded(precision_gap),
            "label_accuracy_when_scored": rounded(label_acc),
            "abstention_rate": rounded(abstain_rate),
            "good_abstain_rate_on_polysemantic_or_random": rounded(good_abstain_rate),
            "bad_abstain_rate_on_gold_features": rounded(bad_abstain_rate),
            "mean_confusable_failures": rounded(confusable_fail),
            "claim_posture": claim_posture,
        })
        for row in rows:
            confidence = float(row["confidence"])
            success = (not row["abstain"]) and bool(row["label_hit"]) and float(row["heldout_auc"] or 0.0) >= 0.75
            calibration.append({
                "method": method,
                "feature_id": row["feature_id"],
                "confidence": rounded(confidence),
                "success": success,
                "calibration_error_abs": rounded(abs(confidence - (1.0 if success else 0.0))),
            })
            if row["score_posture"] != "passes" or row["expected_abstain"]:
                review.append({
                    "feature_id": row["feature_id"],
                    "method": method,
                    "generated_label": row["generated_label"],
                    "gold_label": row["gold_label"],
                    "reason": "expected_abstain" if row["expected_abstain"] else row["score_posture"],
                    "heldout_auc": row["heldout_auc"],
                    "confidence": row["confidence"],
                    "student_label_primary": "",
                    "student_label_secondary": "",
                    "student_confidence": "",
                    "student_evidence_span": "",
                    "reviewer_label": "",
                    "agreement_status": "",
                })
    rankable_evidence = [r for r in evidence if r["method"] not in {"gold_calibration", "shuffled_top_context_control"}]
    if not rankable_evidence:
        rankable_evidence = evidence
    metrics = {
        "n_methods": len(methods),
        "best_method": max(rankable_evidence, key=lambda r: (float(r["mean_heldout_auc"] or 0), float(r["label_accuracy_when_scored"] or 0)))["method"],
        "mean_calibration_error": rounded(safe_mean([r["calibration_error_abs"] for r in calibration])),
        "supported_methods": sum(1 for r in evidence if r["claim_posture"] == "auto_label_audit_supported"),
    }
    return evidence, review, calibration, metrics


def write_tables(ctx: bench.RunContext, explanations: Sequence[Mapping[str, Any]], tests: Sequence[Mapping[str, Any]], scores: Sequence[Mapping[str, Any]], evidence: Sequence[Mapping[str, Any]], review: Sequence[Mapping[str, Any]], calibration: Sequence[Mapping[str, Any]]) -> None:
    specs = [
        ("tables/generated_explanations.csv", explanations, "Generated labels/explanations with confidence, abstention, and human-review fields."),
        ("tables/explanation_tests.csv", tests, "Held-out, negative, confusable, and token-overlap test rows."),
        ("tables/explanation_scores.csv", scores, "Feature/method explanation scores."),
        ("tables/auto_interp_evidence_matrix.csv", evidence, "Method-level evidence matrix for automated interpretability."),
        ("tables/human_review_queue.csv", review, "Rows needing student or reviewer labels before broad claims."),
        ("tables/confidence_calibration.csv", calibration, "Per-feature confidence versus success calibration rows."),
    ]
    for rel, rows, desc in specs:
        path = ctx.path(*rel.split("/"))
        bench.write_csv_with_context(ctx, path, rows)
        ctx.register_artifact(path, "table", desc)


def write_method_card(ctx: bench.RunContext, evidence: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Lab 31 method card",
        "",
        "This lab evaluates auto-labels as hypotheses. It does not treat automated explanations as feature meanings.",
        "",
        "- feature source: frozen synthetic/domain feature tasks",
        "- methods: majority-domain, structured local, test-aware abstention, cross-feature shuffled control, and gold calibration upper bound",
        "- tests: held-out positives, hard negatives, confusable negatives, and token-overlap decoys",
        "- evidence rung: `AUDIT + DECODE`",
        "- forbidden claim: the automated label is the feature's meaning",
        "",
        "| method | mean AUC | label accuracy | abstention | confusable failures | posture |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in evidence:
        lines.append(
            f"| {row['method']} | {row['mean_heldout_auc']} | {row['label_accuracy_when_scored']} | "
            f"{row['abstention_rate']} | {row['mean_confusable_failures']} | {row['claim_posture']} |"
        )
    path = ctx.path("method_card.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Lab 31 method card and auto-label verdicts.")


def write_operationalization_audit(ctx: bench.RunContext, evidence: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Lab 31 operationalization audit",
        "",
        "Favorite interpretation under attack: an automatic label captures a feature's meaning.",
        "",
        "## What the measurement can say",
        "",
        "A label method predicted held-out positive contexts better than hard, confusable, and token-overlap negative contexts on this frozen suite.",
        "",
        "## What it cannot say",
        "",
        "It cannot say the label is complete, unique, monosemantic, or deployment-ready.",
        "",
        "## Cheap explanations",
        "",
        "- The label memorizes a repeated keyword.",
        "- Confusable negatives share the same word but not the concept.",
        "- Polysemantic features deserve abstention, not a forced label.",
        "- Gold calibration is an upper bound, not an automated method.",
        "- A high AUC can hide bad human-readable wording.",
        "",
        "## Verdicts",
        "",
    ]
    for row in evidence:
        lines.append(f"- `{row['method']}`: `{row['claim_posture']}` with mean AUC `{row['mean_heldout_auc']}`.")
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Controls and non-claims for Lab 31 automated interpretability.")


def write_run_summary(ctx: bench.RunContext, data_info: Mapping[str, Any], metrics: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Lab 31 run summary: automated interpretability at scale",
        "",
        f"- data rows: {data_info['n_rows_selected']} selected from `{pathlib.Path(str(data_info['data_path'])).name}`",
        f"- feature types: `{data_info['feature_types']}`",
        f"- best method: `{metrics['best_method']}`",
        f"- supported methods: `{metrics['supported_methods']}` / `{metrics['n_methods']}`",
        "",
        "## Method verdicts",
        "",
        "| method | mean AUC | label accuracy | good abstain | bad abstain | posture |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in evidence:
        lines.append(
            f"| `{row['method']}` | {row['mean_heldout_auc']} | {row['label_accuracy_when_scored']} | "
            f"{row['good_abstain_rate_on_polysemantic_or_random']} | {row['bad_abstain_rate_on_gold_features']} | {row['claim_posture']} |"
        )
    if int(metrics["supported_methods"]) > 0:
        surviving_claim = (
            "At least one method produced labels that predicted held-out positives versus negatives on this suite, "
            "abstained on high-risk features, and avoided the strongest confusable decoys. It did not discover a feature's meaning."
        )
    else:
        surviving_claim = (
            "No automated method cleared every support gate. The surviving claim is narrower: some labels predicted held-out positives, "
            "but the evidence matrix still found calibration, control, abstention, or confusable limitations."
        )
    lines += [
        "",
        "## Reading order",
        "",
        "1. `method_card.md` for method boundaries and non-claims.",
        "2. `tables/generated_explanations.csv` for labels, confidence, abstention, and human-review fields.",
        "3. `tables/explanation_tests.csv` for the held-out/confusable/decoy suite.",
        "4. `tables/explanation_scores.csv` and `tables/auto_interp_evidence_matrix.csv` for quantitative verdicts.",
        "5. `tables/human_review_queue.csv` before trusting any label in a writeup.",
        "",
        "## Smallest surviving claim",
        "",
        surviving_claim,
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Lab 31 run summary and reading order.")


def write_plot_guide(ctx: bench.RunContext) -> None:
    rows = [
        {"plot": "plots/auto_interp_dashboard.png", "read_for": "Method AUC, label accuracy, abstention, and confusable failures.", "non_claim": "Auto-labels are hypotheses."},
        {"plot": "plots/explanation_quality_matrix.png", "read_for": "Method-by-feature AUC heatmap.", "non_claim": "AUC is not label completeness."},
        {"plot": "plots/confidence_calibration_curve.png", "read_for": "Confidence versus observed success.", "non_claim": "Small bins are qualitative."},
        {"plot": "plots/abstention_frontier.png", "read_for": "Good versus bad abstention.", "non_claim": "Abstention policy is not universal."},
        {"plot": "plots/confusable_pair_failure_atlas.png", "read_for": "Failures on confusable and decoy contexts.", "non_claim": "Keyword overlap can fool labels."},
        {"plot": "plots/random_feature_sanity_panel.png", "read_for": "Random/polysemantic feature handling.", "non_claim": "Random controls are a sanity floor."},
    ]
    path = ctx.path("plots", "plot_reading_guide.csv")
    bench.write_csv(path, rows)
    ctx.register_artifact(path, "table", "Plot reading guide for Lab 31.")


def write_plots(ctx: bench.RunContext, evidence: Sequence[Mapping[str, Any]], scores: Sequence[Mapping[str, Any]], calibration: Sequence[Mapping[str, Any]]) -> None:
    write_plot_guide(ctx)
    if ctx.args.no_plots:
        return
    import matplotlib.pyplot as plt
    import numpy as np

    methods = [r["method"] for r in evidence]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Lab 31 automated interpretability dashboard", fontsize=14, fontweight="bold")
    axes[0, 0].bar(x, [float(r["mean_heldout_auc"] or 0) for r in evidence], color="#0072B2")
    axes[0, 0].axhline(0.75, color="#444444", linestyle="--", linewidth=0.8)
    axes[0, 0].set_xticks(x, methods, rotation=30, ha="right")
    axes[0, 0].set_title("Held-out AUC")
    axes[0, 1].bar(x, [float(r["label_accuracy_when_scored"] or 0) for r in evidence], color="#009E73")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_xticks(x, methods, rotation=30, ha="right")
    axes[0, 1].set_title("Label accuracy when scored")
    axes[1, 0].bar(x, [float(r["good_abstain_rate_on_polysemantic_or_random"] or 0) for r in evidence], color="#CC79A7")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_xticks(x, methods, rotation=30, ha="right")
    axes[1, 0].set_title("Good abstention")
    axes[1, 1].bar(x, [float(r["mean_confusable_failures"] or 0) for r in evidence], color="#D55E00")
    axes[1, 1].set_xticks(x, methods, rotation=30, ha="right")
    axes[1, 1].set_title("Confusable failures")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    bench.save_figure(ctx, fig, "auto_interp_dashboard.png", "Lab 31 auto-interpretability dashboard.")

    features = sorted({r["feature_id"] for r in scores})
    mat = np.zeros((len(methods), len(features)))
    for i, method in enumerate(methods):
        for j, feature in enumerate(features):
            vals = [float(r["heldout_auc"] or 0) for r in scores if r["method"] == method and r["feature_id"] == feature]
            mat[i, j] = safe_mean(vals, default=0.0)
    fig, ax = plt.subplots(figsize=(max(8, len(features) * 0.55), 4.8))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(methods)), methods)
    ax.set_xticks(range(len(features)), [f.split("_")[1] if "_" in f else f for f in features], rotation=35, ha="right")
    ax.set_title("Explanation quality matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "explanation_quality_matrix.png", "Method-by-feature AUC heatmap.")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter([float(r["confidence"]) for r in calibration], [1.0 if r["success"] else 0.0 for r in calibration], alpha=0.65)
    ax.plot([0, 1], [0, 1], color="#444444", linestyle="--", linewidth=0.8)
    ax.set_xlabel("confidence")
    ax.set_ylabel("success")
    ax.set_title("Confidence calibration curve")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "confidence_calibration_curve.png", "Confidence versus observed success.")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(
        [float(r["bad_abstain_rate_on_gold_features"] or 0) for r in evidence],
        [float(r["good_abstain_rate_on_polysemantic_or_random"] or 0) for r in evidence],
        color="#0072B2",
    )
    for row in evidence:
        ax.annotate(row["method"], (float(row["bad_abstain_rate_on_gold_features"] or 0), float(row["good_abstain_rate_on_polysemantic_or_random"] or 0)), fontsize=8)
    ax.set_xlabel("bad abstention on gold features")
    ax.set_ylabel("good abstention on risky features")
    ax.set_title("Abstention frontier")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "abstention_frontier.png", "Good/bad abstention frontier.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(methods, [float(r["mean_confusable_failures"] or 0) for r in evidence], color="#D55E00")
    ax.set_xticks(x, methods, rotation=30, ha="right")
    ax.set_title("Confusable pair failure atlas")
    ax.set_ylabel("mean failures")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "confusable_pair_failure_atlas.png", "Confusable/decoy failure atlas.")

    random_rows = [r for r in scores if r["feature_type"] in {"random_control", "polysemantic_gold"}]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    abstain_by_method = []
    for method in methods:
        rows = [r for r in random_rows if r["method"] == method]
        abstain_by_method.append(safe_mean([1.0 if r["abstain"] else 0.0 for r in rows], default=0.0))
    ax.bar(methods, abstain_by_method, color="#999999")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, methods, rotation=30, ha="right")
    ax.set_title("Random feature sanity panel")
    ax.set_ylabel("abstention rate on random/polysemantic")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "random_feature_sanity_panel.png", "Random/polysemantic abstention panel.")


def write_claims(ctx: bench.RunContext, evidence: Sequence[Mapping[str, Any]]) -> None:
    run_name = ctx.run_dir.name
    claims = []
    for i, row in enumerate(evidence, start=1):
        claims.append({
            "id": f"{LAB_ID}-C{i}",
            "tag": "AUDIT,DECODE",
            "text": (
                f"Method `{row['method']}` reached mean held-out AUC {row['mean_heldout_auc']}, "
                f"label accuracy {row['label_accuracy_when_scored']}, abstention rate {row['abstention_rate']}, "
                f"and mean confusable failures {row['mean_confusable_failures']} on the Lab 31 suite. "
                f"Posture: {row['claim_posture']}."
            ),
            "artifact": f"runs/{run_name}/tables/auto_interp_evidence_matrix.csv",
            "falsifier": "Held-out positives, confusable negatives, decoys, or human review invalidate the generated label.",
        })
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    tasks, data_info = load_tasks(ctx)
    manifest_path = ctx.path("diagnostics", "data_manifest.json")
    bench.write_json(manifest_path, data_info)
    ctx.register_artifact(manifest_path, "diagnostic", "Lab 31 data manifest and offline audit scope.")
    bench.run_hook_parity_check(ctx, bundle, tasks[0].top_contexts[0])
    first = bench.run_with_residual_cache(bundle, tasks[0].top_contexts[0])
    bench.run_lens_self_check(ctx, bundle, first)
    bench.run_patch_noop_check(ctx, bundle, tasks[0].top_contexts[0])
    explanations = generate_explanations(tasks)
    tests, scores = build_tests(tasks, explanations)
    evidence, review, calibration, metrics = summarize_evidence(scores)
    write_tables(ctx, explanations, tests, scores, evidence, review, calibration)
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, {**metrics, "data": data_info})
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 31 metrics.")
    write_method_card(ctx, evidence)
    write_operationalization_audit(ctx, evidence)
    write_run_summary(ctx, data_info, metrics, evidence)
    write_claims(ctx, evidence)
    write_plots(ctx, evidence, scores, calibration)
    print(f"[lab31] wrote {len(explanations)} explanations, {len(tests)} tests, and {len(evidence)} method verdicts")
