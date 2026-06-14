"""Lab 31: Automated interpretability at scale.

Automated explanations are treated as hypotheses to audit, not feature
meanings. The default path is offline and deterministic: simple explainers read
feature top contexts, emit candidate labels or abstentions, and then the lab
scores whether those labels predict held-out positives better than hard
negatives, confusables, and token-overlap decoys.

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
PROMPT_SET_CAPS = {"small": 10, "medium": 14, "full": 0}

DOMAIN_LEXICON: dict[str, set[str]] = {
    "code": {
        "function", "json", "typed", "object", "unit", "test", "module", "developer", "loop", "exception",
        "script", "library", "tensor", "compiler", "bracket", "software", "algorithm", "parser", "schema",
        "class", "import", "runtime", "debug", "assertion", "variable", "api", "cache",
    },
    "cooking": {
        "recipe", "herbs", "sauce", "chef", "broth", "heat", "dough", "oven", "loaf", "cook", "onions",
        "pan", "batter", "flour", "butter", "kitchen", "simmer", "bake", "spice", "yeast", "knife",
        "salt", "roast", "ingredient", "mixture", "stir",
    },
    "finance": {
        "market", "earnings", "analyst", "fund", "portfolio", "rate", "bond", "liquidity", "credit",
        "spreads", "company", "guidance", "stock", "investors", "revenue", "margin", "equity", "debt",
        "cash", "trading", "loan", "default", "yield", "hedge", "capital",
    },
    "sports": {
        "striker", "scored", "pass", "midfield", "team", "lead", "minute", "coach", "play", "game",
        "runner", "race", "field", "goalkeeper", "shot", "league", "season", "defense", "match", "pitch",
        "tournament", "athlete", "sprint", "score", "goal",
    },
    "law": {
        "court", "statute", "precedent", "attorney", "motion", "hearing", "judge", "evidence", "trial",
        "contract", "clause", "liability", "dispute", "jury", "testimony", "legal", "appeal", "plaintiff",
        "defendant", "witness", "brief", "regulation", "damages", "claim",
    },
    "medicine": {
        "clinician", "symptoms", "dosage", "hospital", "patient", "treatment", "nurse", "pulse", "breathing",
        "recovery", "physician", "scan", "injury", "vaccine", "fever", "clinical", "diagnosis", "therapy",
        "blood", "infection", "pain", "ward", "prescription", "trial",
    },
    "weather": {
        "storm", "rain", "wind", "clouds", "forecast", "snow", "humid", "air", "cold", "thunder",
        "bright", "morning", "coast", "fog", "weathered", "pressure", "hail", "temperature", "climate",
        "breeze", "sunny", "downpour", "front", "radar",
    },
    "emotion": {
        "joy", "generous", "apology", "anger", "relief", "afraid", "smiled", "sadness", "friends",
        "support", "proud", "felt", "mood", "anxious", "fear", "happy", "grief", "delight", "lonely",
        "worried", "calm", "furious", "hope", "tears", "comfort",
    },
}

METHODS = (
    "majority_domain",
    "structured_local",
    "test_aware",
    "gold_calibration",
    "shuffled_top_context_control",
)
AUTOMATED_METHODS = {"majority_domain", "structured_local", "test_aware"}
REVIEW_FIELDS = (
    "student_label_primary",
    "student_label_secondary",
    "student_confidence",
    "student_evidence_span",
    "reviewer_label",
    "agreement_status",
)
REQUIRED_LIST_FIELDS = (
    "top_contexts",
    "heldout_contexts",
    "negative_contexts",
    "confusable_contexts",
    "adversarial_contexts",
)
OPTIONAL_LIST_FIELDS = ("paraphrase_contexts",)
REQUIRED_FIELDS = {
    "feature_id",
    "model",
    "layer",
    "feature_index",
    "feature_type",
    "top_contexts",
    "heldout_contexts",
    "negative_contexts",
    "confusable_contexts",
    "adversarial_contexts",
}

AUC_SUPPORT_BAR = 0.75
LABEL_ACCURACY_BAR = 0.70
CONTROL_GAP_BAR = 0.10
GOOD_ABSTAIN_BAR = 0.75
BAD_ABSTAIN_BAR = 0.25
CONFUSABLE_FAILURE_BAR = 1.0
DECOY_FAILURE_RATE_BAR = 0.50
CONFUSABLE_FAILURE_THRESHOLD = 0.75
KEY_DELETION_FRAGILITY_BAR = 0.20
MIN_CONTEXTS_PER_SUITE = 2


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
    paraphrase_contexts: list[str] = dataclasses.field(default_factory=list)
    gold_label: str | None = None
    gold_label_secondary: str = ""
    expected_abstain: bool = False
    source_lab: str = "synthetic"
    risk_notes: str = ""

    @property
    def is_risky(self) -> bool:
        return bool(self.expected_abstain or self.feature_type in {"polysemantic_gold", "random_control", "ambiguous_control"})


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


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def safe_mean(values: Sequence[Any], default: float = float("nan")) -> float:
    vals = [fnum(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(statistics.fmean(vals)) if vals else default


def safe_max(values: Sequence[Any], default: float = float("nan")) -> float:
    vals = [fnum(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return max(vals) if vals else default


def safe_stderr(values: Sequence[Any], default: float = float("nan")) -> float:
    vals = [fnum(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    if len(vals) <= 1:
        return default
    return float(statistics.stdev(vals) / math.sqrt(len(vals)))


def binary_se(successes: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    p = successes / n
    return math.sqrt(max(0.0, p * (1.0 - p) / n))


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def stable_row_id(*parts: Any) -> str:
    text = "::".join(str(p) for p in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    return f"{re.sub(r'[^A-Za-z0-9_.=-]+', '_', text)[:96]}__{digest}"


def feature_method_id(feature_id: str, method: str) -> str:
    return f"{feature_id}::{method}"


def method_role(method: str) -> str:
    if method in AUTOMATED_METHODS:
        return "automated_method"
    if method == "gold_calibration":
        return "calibration_upper_bound"
    if method == "shuffled_top_context_control":
        return "negative_control"
    return "other"


def risk_flags_for_task(task: FeatureTask) -> list[str]:
    flags: list[str] = []
    if task.expected_abstain:
        flags.append("expected_abstain")
    if task.feature_type in {"polysemantic_gold", "random_control", "ambiguous_control"}:
        flags.append(task.feature_type)
    if task.gold_label_secondary:
        flags.append("secondary_gold_label")
    if task.risk_notes:
        flags.append("risk_notes_present")
    return sorted(set(flags))


def warning_codes(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({str(r.get("code")) for r in rows if str(r.get("severity", "")).lower() in {"warning", "error"} and r.get("code")})


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


def words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]*", text.lower())


def clean_token(token: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_+-]", "", token.lower())


def domain_scores(contexts: Sequence[str]) -> dict[str, int]:
    tokens = words(" ".join(contexts))
    return {domain: sum(1 for token in tokens if token in vocab) for domain, vocab in DOMAIN_LEXICON.items()}


def best_domain(contexts: Sequence[str]) -> tuple[str | None, float, str, dict[str, int]]:
    scores = domain_scores(contexts)
    total = sum(scores.values())
    if total <= 0:
        return None, 0.0, "", scores
    ranked = sorted(scores.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    label, score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0
    confidence = (score - second) / max(1, score)
    evidence = ", ".join(f"{k}:{v}" for k, v in ranked[:5] if v)
    return label, max(0.0, min(1.0, confidence)), evidence, scores


def label_hit(generated_label: str, gold_label: str, secondary: str = "") -> bool:
    if not generated_label or not gold_label:
        return False
    labels = {gold_label.strip().lower()}
    labels.update(clean_token(x) for x in str(secondary).split("+") if clean_token(x))
    labels.discard("")
    return clean_token(generated_label) in labels


def data_path(args: Any) -> pathlib.Path:
    prompt_set = str(getattr(args, "prompt_set", "") or "")
    candidate = pathlib.Path(prompt_set)
    if prompt_set not in PROMPT_SET_CAPS and candidate.suffix.lower() in {".jsonl", ".json"}:
        return candidate if candidate.is_absolute() else (bench.COURSE_ROOT / candidate).resolve()
    return bench.COURSE_ROOT / "data" / DATA_FILE


def manifest_expected_hash(path: pathlib.Path) -> tuple[str | None, str]:
    manifest_path = path.parent / "MANIFEST.json"
    if not manifest_path.exists():
        return None, "data/MANIFEST.json not found"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"data/MANIFEST.json unreadable: {exc}"
    candidates: list[Any] = []
    if isinstance(manifest, dict):
        candidates.extend([
            manifest.get(path.name),
            manifest.get(str(path)),
            manifest.get("files", {}).get(path.name) if isinstance(manifest.get("files"), dict) else None,
        ])
    for entry in candidates:
        if isinstance(entry, str):
            return entry, "found string entry"
        if isinstance(entry, dict):
            for key in ("sha256", "hash", "sha256_hex"):
                val = entry.get(key)
                if isinstance(val, str):
                    return val, f"found {key} entry"
    return None, f"no usable sha256 entry for {path.name}"


def builtin_smoke_tasks() -> list[FeatureTask]:
    """Tiny fallback used only when the JSONL is absent in Tier A.

    The real package includes data/auto_interp_feature_tasks.jsonl. The fallback
    exists so a student can still test artifact plumbing before copying data.
    """
    rows = make_synthetic_task_payloads()
    return [task_from_payload(row) for row in rows[:10]]


def normalize_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    for field in REQUIRED_LIST_FIELDS + OPTIONAL_LIST_FIELDS:
        val = out.get(field, [])
        if val is None:
            val = []
        if not isinstance(val, list):
            raise ValueError(f"{out.get('feature_id', '<unknown>')}: {field} must be a list")
        out[field] = [str(x) for x in val if str(x).strip()]
    out.setdefault("paraphrase_contexts", [])
    out.setdefault("gold_label", None)
    out.setdefault("gold_label_secondary", "")
    out.setdefault("expected_abstain", False)
    out.setdefault("source_lab", "synthetic")
    out.setdefault("risk_notes", "")
    out["layer"] = int(out.get("layer", 0))
    out["feature_index"] = int(out.get("feature_index", 0))
    out["expected_abstain"] = bool(out.get("expected_abstain", False))
    return out


def task_from_payload(payload: Mapping[str, Any]) -> FeatureTask:
    normalized = normalize_payload(payload)
    missing = sorted(REQUIRED_FIELDS - set(normalized))
    if missing:
        raise ValueError(f"{normalized.get('feature_id', '<unknown>')}: missing required fields {missing}")
    return FeatureTask(**{field.name: normalized.get(field.name) for field in dataclasses.fields(FeatureTask)})


def task_schema_audit(tasks: Sequence[FeatureTask]) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    ok_all = True
    seen: set[str] = set()
    for task in tasks:
        problems: list[str] = []
        if task.feature_id in seen:
            problems.append("duplicate_feature_id")
        seen.add(task.feature_id)
        for field in REQUIRED_LIST_FIELDS:
            if len(getattr(task, field)) < MIN_CONTEXTS_PER_SUITE:
                problems.append(f"{field}_lt_{MIN_CONTEXTS_PER_SUITE}")
        if task.paraphrase_contexts and len(task.paraphrase_contexts) < 1:
            problems.append("paraphrase_contexts_empty_after_filter")
        if task.gold_label and task.gold_label not in DOMAIN_LEXICON:
            if not task.expected_abstain:
                problems.append("gold_label_not_in_domain_lexicon")
        if not task.expected_abstain and not task.gold_label:
            problems.append("non_risky_feature_missing_gold_label")
        ok = not problems
        ok_all = ok_all and ok
        rows.append({
            "feature_id": task.feature_id,
            "feature_type": task.feature_type,
            "gold_label": task.gold_label or "",
            "expected_abstain": task.expected_abstain,
            "top_contexts": len(task.top_contexts),
            "heldout_contexts": len(task.heldout_contexts),
            "paraphrase_contexts": len(task.paraphrase_contexts),
            "negative_contexts": len(task.negative_contexts),
            "confusable_contexts": len(task.confusable_contexts),
            "adversarial_contexts": len(task.adversarial_contexts),
            "schema_ok": ok,
            "problems": ";".join(problems),
        })
    return rows, ok_all


def balanced_cap(tasks: Sequence[FeatureTask], cap: int) -> list[FeatureTask]:
    if cap <= 0 or len(tasks) <= cap:
        return list(tasks)
    by_type: dict[str, list[FeatureTask]] = defaultdict(list)
    for task in tasks:
        by_type[task.feature_type].append(task)
    out: list[FeatureTask] = []
    cursor = 0
    types = sorted(by_type)
    while len(out) < cap:
        made_progress = False
        for typ in types:
            if cursor < len(by_type[typ]):
                out.append(by_type[typ][cursor])
                made_progress = True
                if len(out) >= cap:
                    break
        if not made_progress:
            break
        cursor += 1
    return out


def load_tasks(ctx: bench.RunContext) -> tuple[list[FeatureTask], dict[str, Any], list[dict[str, Any]]]:
    path = data_path(ctx.args)
    source = "frozen_jsonl"
    fallback = False
    all_tasks: list[FeatureTask] = []
    expected_sha: str | None = None
    manifest_note = ""
    actual_sha: str | None = None
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    all_tasks.append(task_from_payload(payload))
                except Exception as exc:
                    raise ValueError(f"{path}:{line_no}: invalid Lab 31 task row: {exc}") from exc
        actual_sha = file_sha256(path)
        expected_sha, manifest_note = manifest_expected_hash(path)
    else:
        if str(getattr(ctx.args, "tier", "")).lower() != "a":
            raise FileNotFoundError(f"Lab 31 data file not found: {path}")
        print("[lab31] data/auto_interp_feature_tasks.jsonl missing; using built-in Tier A fallback. This is plumbing only.")
        all_tasks = builtin_smoke_tasks()
        source = "builtin_tier_a_smoke_fallback"
        fallback = True
        actual_sha = hashlib.sha256("\n".join(t.feature_id for t in all_tasks).encode("utf-8")).hexdigest()
        manifest_note = "frozen JSONL absent; fallback has no manifest entry"
    if not all_tasks:
        raise ValueError("Lab 31 has no feature tasks after loading")
    schema_rows, schema_ok = task_schema_audit(all_tasks)
    cap = PROMPT_SET_CAPS.get(str(ctx.args.prompt_set), 0)
    selected = balanced_cap(all_tasks, cap)
    if int(ctx.args.max_examples or 0) > 0:
        selected = balanced_cap(selected, int(ctx.args.max_examples))
    info = {
        "data_file": DATA_FILE,
        "data_path": str(path),
        "data_source": source,
        "sha256": actual_sha,
        "manifest_expected_sha256": expected_sha,
        "manifest_note": manifest_note,
        "manifest_ok": (actual_sha == expected_sha) if expected_sha else None,
        "n_rows_file": len(all_tasks),
        "n_rows_selected": len(selected),
        "feature_types": dict(Counter(t.feature_type for t in selected)),
        "gold_labels": dict(Counter(t.gold_label or "null" for t in selected)),
        "expected_abstain_count": sum(1 for t in selected if t.expected_abstain),
        "schema_ok": schema_ok,
        "science_ready": bool(schema_ok and not fallback),
        "fallback_data": fallback,
        "science_scope": "offline auto-label audit with synthetic gold labels, confusables, decoys, abstention checks, and human-review scaffolds",
        "prompt_set": ctx.args.prompt_set,
        "max_examples": ctx.args.max_examples,
    }
    return selected, info, schema_rows


# ---------------------------------------------------------------------------
# Explanation methods and scoring
# ---------------------------------------------------------------------------


def remove_key_tokens(contexts: Sequence[str], label: str | None) -> list[str]:
    if not label or label not in DOMAIN_LEXICON:
        return list(contexts)
    vocab = DOMAIN_LEXICON[label]
    out: list[str] = []
    for context in contexts:
        kept = [tok for tok in context.split() if clean_token(tok) not in vocab]
        out.append(" ".join(kept))
    return out


def score_text_for_label(text: str, label: str) -> float:
    if not label or label not in DOMAIN_LEXICON:
        return 0.0
    toks = words(text)
    if not toks:
        return 0.0
    vocab = DOMAIN_LEXICON[label]
    raw = sum(1 for token in toks if token in vocab)
    density = raw / max(1.0, math.sqrt(len(toks) + 1.0))
    # A tiny bonus rewards multiple distinct evidence terms rather than one
    # repeated keyword. This is still a cheap offline proxy, not a judge.
    diversity = len({token for token in toks if token in vocab}) / max(1, len(vocab))
    return float(density + 0.25 * diversity)


def score_contexts_for_label(contexts: Sequence[str], label: str | None) -> list[float]:
    return [score_text_for_label(context, label or "") for context in contexts]


def deletion_fragility(task: FeatureTask, label: str | None) -> tuple[float, str]:
    if not label:
        return 1.0, "no_label"
    before = safe_mean(score_contexts_for_label(task.top_contexts, label), 0.0)
    after_contexts = remove_key_tokens(task.top_contexts, label)
    after = safe_mean(score_contexts_for_label(after_contexts, label), 0.0)
    ratio = after / before if before > 1e-9 else 0.0
    if ratio >= 0.75:
        status = "robust_to_key_deletion"
    elif ratio >= 0.35:
        status = "partly_key_token_dependent"
    else:
        status = "fragile_to_key_deletion"
    return ratio, status


def overlap_risk(task: FeatureTask, label: str | None) -> tuple[float, float, float]:
    if not label:
        return 0.0, 0.0, 0.0
    pos = safe_mean(score_contexts_for_label(task.top_contexts, label), 0.0)
    conf = safe_mean(score_contexts_for_label(task.confusable_contexts, label), 0.0)
    adv = safe_mean(score_contexts_for_label(task.adversarial_contexts, label), 0.0)
    denom = max(pos, 1e-9)
    return conf / denom, adv / denom, pos


def method_candidate_rows(task: FeatureTask, idx: int, tasks: Sequence[FeatureTask]) -> list[tuple[str, str | None, float, str, str, dict[str, Any]]]:
    majority_label, majority_conf, majority_evidence, top_scores = best_domain(task.top_contexts)
    confusable_label, _confusable_conf, confusable_evidence, _ = best_domain(task.confusable_contexts)
    adversarial_label, _adv_conf, adversarial_evidence, _ = best_domain(task.adversarial_contexts)
    deletion_ratio, deletion_status = deletion_fragility(task, majority_label)
    conf_ratio, adv_ratio, pos_score = overlap_risk(task, majority_label)

    control_pool = [t for t in tasks if (t.gold_label or "") != (task.gold_label or "") or t.expected_abstain != task.expected_abstain]
    control_task = control_pool[(idx + 3) % len(control_pool)] if control_pool else task
    shuffled_contexts = list(reversed(control_task.top_contexts))
    shuffled_label, shuffled_conf, shuffled_evidence, _ = best_domain(shuffled_contexts)

    rows: list[tuple[str, str | None, float, str, str, dict[str, Any]]] = []
    rows.append((
        "majority_domain",
        majority_label,
        majority_conf,
        majority_evidence,
        "Chooses the highest lexicon domain in top contexts. Fast, brittle, and intentionally easy to fool.",
        {"top_scores": top_scores, "deletion_ratio": deletion_ratio, "deletion_status": deletion_status, "confusable_ratio": conf_ratio, "adversarial_ratio": adv_ratio},
    ))

    structured_conf = majority_conf
    penalties: list[str] = []
    if majority_label and confusable_label == majority_label:
        structured_conf *= 0.55
        penalties.append("confusable_same_label")
    if majority_label and adversarial_label == majority_label:
        structured_conf *= 0.65
        penalties.append("adversarial_same_label")
    if deletion_ratio < KEY_DELETION_FRAGILITY_BAR:
        structured_conf *= 0.80
        penalties.append("key_token_fragile")
    rows.append((
        "structured_local",
        majority_label,
        structured_conf,
        f"top={majority_evidence}; confusable={confusable_evidence}; adversarial={adversarial_evidence}; penalties={','.join(penalties) or 'none'}",
        "Uses top contexts, then discounts labels that also fit confusables, decoys, or deletion-fragile evidence.",
        {"top_scores": top_scores, "deletion_ratio": deletion_ratio, "deletion_status": deletion_status, "confusable_ratio": conf_ratio, "adversarial_ratio": adv_ratio, "penalties": penalties},
    ))

    # test_aware uses a conservative offline audit before emitting a label. It
    # abstains on features marked risky, obvious polysemantic mixtures, key-token
    # deletion collapse, or controls that score too close to positives.
    test_conf = structured_conf
    abstain_reasons: list[str] = []
    if task.expected_abstain:
        abstain_reasons.append("expected_high_risk_feature")
    # Key-token deletion is reported as fragility evidence. It should not by itself
    # force abstention in this lexical offline suite, because every heuristic
    # label necessarily uses words as its handle.
    if conf_ratio > 0.65:
        abstain_reasons.append("confusable_scores_close_to_top")
    if adv_ratio > 0.55:
        abstain_reasons.append("token_overlap_decoy_scores_close_to_top")
    if majority_label and (confusable_label and confusable_label != majority_label) and majority_conf < 0.45:
        abstain_reasons.append("mixed_domain_top_contexts")
    test_conf = min(test_conf, max(0.0, 1.0 - 0.35 * len(abstain_reasons)))
    test_label = None if abstain_reasons or test_conf < 0.25 else majority_label
    rows.append((
        "test_aware",
        test_label,
        test_conf,
        f"top={majority_evidence}; deleted_ratio={deletion_ratio:.3f}; conf_ratio={conf_ratio:.3f}; adv_ratio={adv_ratio:.3f}; abstain={','.join(abstain_reasons) or 'no'}",
        "Emits a label only if key-token deletion, confusable contexts, decoys, and risk flags do not defeat it.",
        {"top_scores": top_scores, "deletion_ratio": deletion_ratio, "deletion_status": deletion_status, "confusable_ratio": conf_ratio, "adversarial_ratio": adv_ratio, "abstain_reasons": abstain_reasons},
    ))

    gold_label = task.gold_label if task.gold_label and not task.expected_abstain else None
    gold_conf = 0.95 if gold_label else 0.20
    rows.append((
        "gold_calibration",
        gold_label,
        gold_conf,
        task.gold_label_secondary or task.gold_label or "high-risk / no gold label",
        "Human/gold upper-bound row. It calibrates the test suite and is not an automated explanation method.",
        {"calibration_only": True},
    ))
    rows.append((
        "shuffled_top_context_control",
        shuffled_label,
        shuffled_conf,
        shuffled_evidence,
        f"Control explanation from a different feature's top contexts: {control_task.feature_id}.",
        {"control_feature_id": control_task.feature_id},
    ))
    return rows


def generate_explanations(tasks: Sequence[FeatureTask]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    deletion_rows: list[dict[str, Any]] = []
    task_list = list(tasks)
    for idx, task in enumerate(task_list):
        risk_flags = risk_flags_for_task(task)
        for method, label, confidence, evidence, template, extras in method_candidate_rows(task, idx, task_list):
            label_text = "" if label is None else str(label)
            abstain = label is None or confidence < 0.25
            explanation = (
                "ABSTAIN: insufficiently specific, risky, random, or polysemantic feature."
                if abstain else f"Feature appears to activate on {label_text} contexts because {template}"
            )
            reason_bits = extras.get("abstain_reasons", []) if isinstance(extras, Mapping) else []
            fid_mid = feature_method_id(task.feature_id, method)
            raw_candidate = label_text or str(extras.get("raw_candidate_label", "")) if isinstance(extras, Mapping) else label_text
            human_review_required = bool(method in AUTOMATED_METHODS and (task.is_risky or abstain or reason_bits))
            rows.append({
                "explanation_id": fid_mid,
                "feature_method_id": fid_mid,
                "feature_id": task.feature_id,
                "feature_type": task.feature_type,
                "source_lab": task.source_lab,
                "source_model": task.model,
                "layer": task.layer,
                "feature_index": task.feature_index,
                "method": method,
                "method_role": method_role(method),
                "generated_label": label_text,
                "raw_candidate_label": raw_candidate,
                "gold_label": task.gold_label or "",
                "gold_label_secondary": task.gold_label_secondary,
                "label_hit_gold": label_hit(label_text, task.gold_label or "", task.gold_label_secondary),
                "expected_abstain": task.expected_abstain,
                "is_high_risk_feature": task.is_risky,
                "risk_flags": ";".join(risk_flags),
                "confidence": rounded(confidence),
                "abstain": abstain,
                "abstain_reason": ";".join(reason_bits) if reason_bits else ("low_confidence" if abstain and method != "gold_calibration" else ""),
                "human_review_required": human_review_required,
                "top_context_count": len(task.top_contexts),
                "heldout_context_count": len(task.heldout_contexts),
                "paraphrase_context_count": len(task.paraphrase_contexts),
                "negative_context_count": len(task.negative_contexts),
                "confusable_context_count": len(task.confusable_contexts),
                "adversarial_context_count": len(task.adversarial_contexts),
                "explanation": explanation,
                "evidence_terms": evidence,
                "risk_notes": task.risk_notes,
                **{field: "" for field in REVIEW_FIELDS},
            })
            if method in AUTOMATED_METHODS:
                deletion_rows.append({
                    "deletion_audit_id": stable_row_id(task.feature_id, method, "key_token_deletion"),
                    "feature_method_id": fid_mid,
                    "feature_id": task.feature_id,
                    "feature_type": task.feature_type,
                    "method": method,
                    "method_role": method_role(method),
                    "generated_label": label_text,
                    "raw_candidate_label": raw_candidate,
                    "expected_abstain": task.expected_abstain,
                    "risk_flags": ";".join(risk_flags),
                    "deletion_ratio": rounded(extras.get("deletion_ratio", float("nan")) if isinstance(extras, Mapping) else float("nan")),
                    "deletion_status": extras.get("deletion_status", "") if isinstance(extras, Mapping) else "",
                    "confusable_ratio": rounded(extras.get("confusable_ratio", float("nan")) if isinstance(extras, Mapping) else float("nan")),
                    "adversarial_ratio": rounded(extras.get("adversarial_ratio", float("nan")) if isinstance(extras, Mapping) else float("nan")),
                    "fragility_flag": bool(fnum(extras.get("deletion_ratio", 1.0) if isinstance(extras, Mapping) else 1.0) < KEY_DELETION_FRAGILITY_BAR),
                })
    return rows, deletion_rows


def build_tests(tasks: Sequence[FeatureTask], explanations: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    test_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    by_feature_method = {(str(r["feature_id"]), str(r["method"])): r for r in explanations}
    for task in tasks:
        contexts: list[tuple[str, str, int, list[str]]] = [
            ("heldout_positive", "heldout", 1, task.heldout_contexts),
            ("paraphrase_positive", "paraphrase", 1, task.paraphrase_contexts),
            ("hard_negative", "negative", 0, task.negative_contexts),
            ("confusable_negative", "confusable", 0, task.confusable_contexts),
            ("token_overlap_decoy", "adversarial", 0, task.adversarial_contexts),
        ]
        for method in METHODS:
            exp = by_feature_method[(task.feature_id, method)]
            label = str(exp.get("generated_label", ""))
            abstain = bool(exp.get("abstain"))
            fid_mid = feature_method_id(task.feature_id, method)
            labels: list[int] = []
            scores: list[float] = []
            suite_scores: dict[str, list[float]] = defaultdict(list)
            suite_labels: dict[str, list[int]] = defaultdict(list)
            confusable_failures = 0
            decoy_failures = 0
            suites_present: set[str] = set()
            for suite, context_kind, expected_active, context_list in contexts:
                if not context_list:
                    continue
                suites_present.add(suite)
                for i, context in enumerate(context_list):
                    pred = 0.0 if abstain else score_text_for_label(context, label)
                    labels.append(int(expected_active))
                    scores.append(pred)
                    suite_scores[suite].append(pred)
                    suite_labels[suite].append(int(expected_active))
                    if suite == "confusable_negative" and pred >= CONFUSABLE_FAILURE_THRESHOLD:
                        confusable_failures += 1
                    if suite == "token_overlap_decoy" and pred >= CONFUSABLE_FAILURE_THRESHOLD:
                        decoy_failures += 1
                    context_id = stable_row_id(task.feature_id, suite, i)
                    test_rows.append({
                        "test_row_id": stable_row_id(task.feature_id, method, suite, i),
                        "feature_method_id": fid_mid,
                        "context_id": context_id,
                        "feature_id": task.feature_id,
                        "feature_type": task.feature_type,
                        "source_lab": task.source_lab,
                        "source_model": task.model,
                        "layer": task.layer,
                        "feature_index": task.feature_index,
                        "method": method,
                        "method_role": method_role(method),
                        "test_id": f"{task.feature_id}:{method}:{suite}:{i}",
                        "context_kind": context_kind,
                        "suite": suite,
                        "suite_role": "positive" if expected_active else "negative_control",
                        "context_index": i,
                        "context": context,
                        "expected_active": int(expected_active),
                        "predicted_score": rounded(pred),
                        "generated_label": label,
                        "abstain": abstain,
                        "gold_label": task.gold_label or "",
                        "expected_abstain": task.expected_abstain,
                    })
            auc = auc_binary(labels, scores)
            pos_scores = [s for y, s in zip(labels, scores) if y == 1]
            neg_scores = [s for y, s in zip(labels, scores) if y == 0]
            precision_gap = safe_mean(pos_scores, 0.0) - safe_mean(neg_scores, 0.0)
            heldout_mean = safe_mean(suite_scores.get("heldout_positive", []), 0.0)
            paraphrase_mean = safe_mean(suite_scores.get("paraphrase_positive", []), float("nan"))
            hard_neg_mean = safe_mean(suite_scores.get("hard_negative", []), 0.0)
            conf_mean = safe_mean(suite_scores.get("confusable_negative", []), 0.0)
            decoy_mean = safe_mean(suite_scores.get("token_overlap_decoy", []), 0.0)
            decoy_failure_rate = decoy_failures / max(1, len(suite_scores.get("token_overlap_decoy", [])))
            if abstain:
                posture = "abstained"
            elif auc >= AUC_SUPPORT_BAR and precision_gap > 0 and confusable_failures <= CONFUSABLE_FAILURE_BAR and decoy_failure_rate <= DECOY_FAILURE_RATE_BAR:
                posture = "passes_tests"
            else:
                posture = "fails_or_needs_review"
            score_rows.append({
                "score_id": fid_mid,
                "feature_method_id": fid_mid,
                "feature_id": task.feature_id,
                "feature_type": task.feature_type,
                "source_lab": task.source_lab,
                "source_model": task.model,
                "layer": task.layer,
                "feature_index": task.feature_index,
                "method": method,
                "method_role": method_role(method),
                "generated_label": label,
                "raw_candidate_label": exp.get("raw_candidate_label", label),
                "gold_label": task.gold_label or "",
                "gold_label_secondary": task.gold_label_secondary,
                "label_hit": label_hit(label, task.gold_label or "", task.gold_label_secondary),
                "expected_abstain": task.expected_abstain,
                "is_high_risk_feature": task.is_risky,
                "risk_flags": ";".join(risk_flags_for_task(task)),
                "abstain": abstain,
                "abstain_reason": exp.get("abstain_reason", ""),
                "confidence": exp.get("confidence", ""),
                "heldout_auc": rounded(auc),
                "precision_gap": rounded(precision_gap),
                "heldout_positive_mean_score": rounded(heldout_mean),
                "paraphrase_positive_mean_score": rounded(paraphrase_mean),
                "hard_negative_mean_score": rounded(hard_neg_mean),
                "confusable_negative_mean_score": rounded(conf_mean),
                "token_overlap_decoy_mean_score": rounded(decoy_mean),
                "mean_negative_score": rounded(safe_mean([hard_neg_mean, conf_mean, decoy_mean], 0.0)),
                "confusable_failure_count": confusable_failures,
                "token_overlap_decoy_failure_count": decoy_failures,
                "token_overlap_decoy_failure_rate": rounded(decoy_failure_rate),
                "test_count": len(labels),
                "positive_test_count": sum(labels),
                "negative_test_count": len(labels) - sum(labels),
                "n_suites_present": len(suites_present),
                "suite_coverage": ";".join(sorted(suites_present)),
                "score_posture": posture,
            })
    return test_rows, score_rows

def method_stats(method: str, rows: Sequence[Mapping[str, Any]], control_auc: float) -> dict[str, Any]:
    non_abstain = [r for r in rows if not r.get("abstain")]
    abstained = [r for r in rows if r.get("abstain")]
    labelable = [r for r in rows if not r.get("expected_abstain")]
    should_abstain = [r for r in rows if r.get("expected_abstain")]
    auc_mean = safe_mean([r.get("heldout_auc") for r in non_abstain])
    label_acc = safe_mean([1.0 if r.get("label_hit") else 0.0 for r in non_abstain], default=0.0)
    abstain_rate = len(abstained) / len(rows) if rows else 0.0
    bad_abstain_rate = safe_mean([1.0 if r.get("abstain") else 0.0 for r in labelable], default=0.0)
    good_abstain_rate = safe_mean([1.0 if r.get("abstain") else 0.0 for r in should_abstain], default=0.0)
    confusable_fail = safe_mean([r.get("confusable_failure_count") for r in non_abstain], default=0.0)
    decoy_fail_rate = safe_mean([r.get("token_overlap_decoy_failure_rate") for r in non_abstain], default=0.0)
    precision_gap = safe_mean([r.get("precision_gap") for r in non_abstain])
    control_gap = auc_mean - control_auc if math.isfinite(auc_mean) and math.isfinite(control_auc) else float("nan")
    calibration_success_rows = [
        r for r in rows
        if (not r.get("abstain")) and bool(r.get("label_hit")) and fnum(r.get("heldout_auc")) >= AUC_SUPPORT_BAR
    ]
    if method == "gold_calibration":
        claim_posture = "calibration_upper_bound"
    elif method == "shuffled_top_context_control":
        claim_posture = "control_sanity_check"
    elif control_gap < CONTROL_GAP_BAR:
        claim_posture = "control_limited"
    elif auc_mean >= AUC_SUPPORT_BAR and label_acc >= LABEL_ACCURACY_BAR and good_abstain_rate >= GOOD_ABSTAIN_BAR and bad_abstain_rate <= BAD_ABSTAIN_BAR and confusable_fail <= CONFUSABLE_FAILURE_BAR and decoy_fail_rate <= DECOY_FAILURE_RATE_BAR:
        claim_posture = "auto_label_audit_supported"
    elif auc_mean >= AUC_SUPPORT_BAR and label_acc >= LABEL_ACCURACY_BAR:
        if bad_abstain_rate > BAD_ABSTAIN_BAR:
            claim_posture = "label_predictive_but_over_abstaining"
        elif good_abstain_rate < GOOD_ABSTAIN_BAR:
            claim_posture = "label_predictive_but_abstention_limited"
        elif confusable_fail > CONFUSABLE_FAILURE_BAR or decoy_fail_rate > DECOY_FAILURE_RATE_BAR:
            claim_posture = "label_predictive_but_confusable_limited"
        else:
            claim_posture = "needs_review_or_calibration"
    else:
        claim_posture = "needs_review_or_calibration"
    return {
        "method": method,
        "n_features": len(rows),
        "n_scored": len(non_abstain),
        "n_abstained": len(abstained),
        "mean_heldout_auc": rounded(auc_mean),
        "control_mean_auc": rounded(control_auc),
        "control_gap_vs_shuffled_top_context": rounded(control_gap),
        "mean_precision_gap": rounded(precision_gap),
        "label_accuracy_when_scored": rounded(label_acc),
        "abstention_rate": rounded(abstain_rate),
        "good_abstain_rate_on_polysemantic_or_random": rounded(good_abstain_rate),
        "bad_abstain_rate_on_gold_features": rounded(bad_abstain_rate),
        "mean_confusable_failures": rounded(confusable_fail),
        "mean_decoy_failure_rate": rounded(decoy_fail_rate),
        "n_calibration_success_rows": len(calibration_success_rows),
        "claim_posture": claim_posture,
    }


def summarize_evidence(scores: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    counterexamples: list[dict[str, Any]] = []
    methods = list(METHODS)
    control_rows = [r for r in scores if r.get("method") == "shuffled_top_context_control" and not r.get("abstain")]
    control_auc = safe_mean([r.get("heldout_auc") for r in control_rows], default=0.5)

    by_method = {method: [r for r in scores if r.get("method") == method] for method in methods}
    for method in methods:
        rows = by_method[method]
        if rows:
            evidence.append(method_stats(method, rows, control_auc))
        for row in rows:
            confidence = fnum(row.get("confidence"), 0.0)
            success = (not row.get("abstain")) and bool(row.get("label_hit")) and fnum(row.get("heldout_auc")) >= AUC_SUPPORT_BAR
            calibration.append({
                "method": method,
                "feature_id": row.get("feature_id", ""),
                "feature_type": row.get("feature_type", ""),
                "confidence": rounded(confidence),
                "success": bool(success),
                "calibration_error_abs": rounded(abs(confidence - (1.0 if success else 0.0))),
            })
            needs_review = (
                row.get("score_posture") != "passes_tests"
                or row.get("expected_abstain")
                or fnum(row.get("confusable_failure_count"), 0.0) > CONFUSABLE_FAILURE_BAR
                or fnum(row.get("token_overlap_decoy_failure_rate"), 0.0) > DECOY_FAILURE_RATE_BAR
            )
            if method in AUTOMATED_METHODS and needs_review:
                reasons: list[str] = []
                if row.get("expected_abstain"):
                    reasons.append("expected_abstain")
                if row.get("score_posture") != "passes_tests":
                    reasons.append(str(row.get("score_posture")))
                if fnum(row.get("confusable_failure_count"), 0.0) > CONFUSABLE_FAILURE_BAR:
                    reasons.append("confusable_failure")
                if fnum(row.get("token_overlap_decoy_failure_rate"), 0.0) > DECOY_FAILURE_RATE_BAR:
                    reasons.append("token_overlap_decoy_failure")
                review.append({
                    "feature_id": row.get("feature_id", ""),
                    "feature_type": row.get("feature_type", ""),
                    "method": method,
                    "generated_label": row.get("generated_label", ""),
                    "gold_label": row.get("gold_label", ""),
                    "reason": ";".join(sorted(set(reasons))),
                    "review_priority": review_priority(row),
                    "heldout_auc": row.get("heldout_auc", ""),
                    "confidence": row.get("confidence", ""),
                    "confusable_failure_count": row.get("confusable_failure_count", ""),
                    "token_overlap_decoy_failure_rate": row.get("token_overlap_decoy_failure_rate", ""),
                    **{field: "" for field in REVIEW_FIELDS},
                })
            for kind, condition in counterexample_conditions(row, control_auc):
                counterexamples.append({
                    "feature_id": row.get("feature_id", ""),
                    "feature_type": row.get("feature_type", ""),
                    "method": method,
                    "kind": kind,
                    "generated_label": row.get("generated_label", ""),
                    "gold_label": row.get("gold_label", ""),
                    "heldout_auc": row.get("heldout_auc", ""),
                    "confidence": row.get("confidence", ""),
                    "control_mean_auc": rounded(control_auc),
                    "confusable_failure_count": row.get("confusable_failure_count", ""),
                    "token_overlap_decoy_failure_rate": row.get("token_overlap_decoy_failure_rate", ""),
                    "lesson": condition,
                })

    by_feature: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in scores:
        by_feature[str(row.get("feature_id"))].append(row)
    for feature_id, rows in sorted(by_feature.items()):
        automated = [r for r in rows if r.get("method") in AUTOMATED_METHODS]
        rankable = sorted(
            automated,
            key=lambda r: (fnum(r.get("heldout_auc"), -1), 1.0 if r.get("label_hit") else 0.0, -review_priority(r)),
            reverse=True,
        )
        best = rankable[0] if rankable else rows[0]
        risky = any(bool(r.get("expected_abstain")) for r in rows)
        best_passes = bool(best.get("score_posture") == "passes_tests" and best.get("label_hit") and not best.get("abstain"))
        if risky and not best.get("abstain"):
            posture = "risky_feature_forced_label"
        elif best_passes:
            posture = "feature_label_supported_by_suite"
        elif best.get("abstain"):
            posture = "feature_abstained"
        else:
            posture = "feature_needs_review"
        feature_rows.append({
            "feature_id": feature_id,
            "feature_type": best.get("feature_type", ""),
            "gold_label": best.get("gold_label", ""),
            "expected_abstain": bool(best.get("expected_abstain")),
            "best_automated_method": best.get("method", ""),
            "best_generated_label": best.get("generated_label", ""),
            "best_heldout_auc": best.get("heldout_auc", ""),
            "best_confidence": best.get("confidence", ""),
            "best_confusable_failure_count": best.get("confusable_failure_count", ""),
            "best_decoy_failure_rate": best.get("token_overlap_decoy_failure_rate", ""),
            "review_priority": review_priority(best),
            "feature_posture": posture,
        })

    review = sorted(review, key=lambda r: fnum(r.get("review_priority"), 0.0), reverse=True)
    counterexamples = sorted(counterexamples, key=lambda r: (str(r.get("kind")), str(r.get("feature_id")), str(r.get("method"))))
    rankable_evidence = [r for r in evidence if r["method"] in AUTOMATED_METHODS]
    if not rankable_evidence:
        rankable_evidence = evidence
    metrics = {
        "n_methods": len(methods),
        "best_method": max(rankable_evidence, key=lambda r: (fnum(r.get("mean_heldout_auc"), 0.0), fnum(r.get("label_accuracy_when_scored"), 0.0), fnum(r.get("control_gap_vs_shuffled_top_context"), 0.0)))["method"],
        "mean_calibration_error": rounded(safe_mean([r.get("calibration_error_abs") for r in calibration])),
        "supported_methods": sum(1 for r in evidence if r.get("claim_posture") == "auto_label_audit_supported"),
        "control_mean_auc": rounded(control_auc),
        "n_review_rows": len(review),
        "n_counterexamples": len(counterexamples),
        "thresholds": {
            "auc_support_bar": AUC_SUPPORT_BAR,
            "label_accuracy_bar": LABEL_ACCURACY_BAR,
            "control_gap_bar": CONTROL_GAP_BAR,
            "good_abstain_bar": GOOD_ABSTAIN_BAR,
            "bad_abstain_bar": BAD_ABSTAIN_BAR,
            "confusable_failure_bar": CONFUSABLE_FAILURE_BAR,
            "decoy_failure_rate_bar": DECOY_FAILURE_RATE_BAR,
            "confusable_failure_threshold": CONFUSABLE_FAILURE_THRESHOLD,
            "key_deletion_fragility_bar": KEY_DELETION_FRAGILITY_BAR,
        },
    }
    return evidence, review, calibration, feature_rows, counterexamples, metrics


def review_priority(row: Mapping[str, Any]) -> float:
    priority = 0.0
    if row.get("expected_abstain") and not row.get("abstain"):
        priority += 4.0
    if fnum(row.get("token_overlap_decoy_failure_rate"), 0.0) > DECOY_FAILURE_RATE_BAR:
        priority += 3.0
    if fnum(row.get("confusable_failure_count"), 0.0) > CONFUSABLE_FAILURE_BAR:
        priority += 2.5
    if row.get("score_posture") != "passes_tests" and not row.get("abstain"):
        priority += 1.5
    if fnum(row.get("confidence"), 0.0) > 0.70 and row.get("score_posture") != "passes_tests":
        priority += 1.5
    if row.get("abstain") and not row.get("expected_abstain"):
        priority += 1.0
    return round(priority, 3)


def counterexample_conditions(row: Mapping[str, Any], control_auc: float) -> list[tuple[str, str]]:
    if row.get("method") == "gold_calibration":
        return []
    out: list[tuple[str, str]] = []
    auc = fnum(row.get("heldout_auc"))
    if row.get("method") in AUTOMATED_METHODS and math.isfinite(auc) and math.isfinite(control_auc) and auc <= control_auc + CONTROL_GAP_BAR:
        out.append(("control_matches_or_beats_method", "The cross-feature shuffled-top-context control is too close to this method's score."))
    if row.get("expected_abstain") and not row.get("abstain"):
        out.append(("forced_label_on_risky_feature", "A high-risk, random, or polysemantic feature received a forced automated label."))
    if fnum(row.get("confusable_failure_count"), 0.0) > CONFUSABLE_FAILURE_BAR:
        out.append(("confusable_failure", "The label fires on confusable negatives, so it may be a surface-domain keyword."))
    if fnum(row.get("token_overlap_decoy_failure_rate"), 0.0) > DECOY_FAILURE_RATE_BAR:
        out.append(("token_overlap_decoy_failure", "The label fires on decoys that contain label words but should not activate."))
    if not row.get("label_hit") and not row.get("expected_abstain") and not row.get("abstain"):
        out.append(("wrong_label", "The generated label is not the synthetic gold label."))
    if row.get("abstain") and not row.get("expected_abstain") and row.get("method") in AUTOMATED_METHODS:
        out.append(("over_abstention", "The method abstained on a labelable gold feature."))
    return out


def augment_score_rows(scores: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Add same-feature control gaps and row-level claim flags.

    The method-level evidence matrix compares methods to the global shuffled
    control mean. For figure reading and specimen review, students also need the
    stricter row-level comparison: did this feature-method row beat its own
    shuffled-top-context control?
    """
    control_by_feature: dict[str, Mapping[str, Any]] = {}
    for row in scores:
        if row.get("method") == "shuffled_top_context_control":
            control_by_feature[str(row.get("feature_id"))] = row
    out: list[dict[str, Any]] = []
    for row in scores:
        r = dict(row)
        ctrl = control_by_feature.get(str(row.get("feature_id")), {})
        ctrl_auc = fnum(ctrl.get("heldout_auc"), float("nan"))
        auc = fnum(row.get("heldout_auc"), float("nan"))
        control_gap = auc - ctrl_auc if math.isfinite(auc) and math.isfinite(ctrl_auc) else float("nan")
        r["same_feature_control_method"] = ctrl.get("method", "")
        r["same_feature_control_auc"] = rounded(ctrl_auc)
        r["control_gap_vs_same_feature_shuffled"] = rounded(control_gap)
        r["beats_same_feature_control"] = bool(math.isfinite(control_gap) and control_gap >= CONTROL_GAP_BAR)
        r["passes_auc_gate"] = bool(math.isfinite(auc) and auc >= AUC_SUPPORT_BAR)
        r["passes_decoy_gate"] = bool(fnum(row.get("token_overlap_decoy_failure_rate"), 1.0) <= DECOY_FAILURE_RATE_BAR)
        r["passes_confusable_gate"] = bool(fnum(row.get("confusable_failure_count"), 999.0) <= CONFUSABLE_FAILURE_BAR)
        r["passes_label_gate"] = bool(row.get("expected_abstain") or row.get("label_hit"))
        r["review_priority"] = review_priority(r) if row.get("method") in AUTOMATED_METHODS else 0.0
        out.append(r)
    return out


def suite_score_summary_rows(tests: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in tests:
        grouped[(str(row.get("method", "")), str(row.get("suite", "")), str(row.get("suite_role", "")), str(row.get("feature_type", "")))].append(row)
    rows: list[dict[str, Any]] = []
    for (method, suite, suite_role, feature_type), group in sorted(grouped.items()):
        vals = [fnum(r.get("predicted_score")) for r in group]
        active = sum(1 for r in group if truthy(r.get("expected_active")))
        above = sum(1 for v in vals if math.isfinite(v) and v >= CONFUSABLE_FAILURE_THRESHOLD)
        rows.append({
            "summary_id": stable_row_id(method, suite, feature_type),
            "method": method,
            "method_role": method_role(method),
            "suite": suite,
            "suite_role": suite_role,
            "feature_type": feature_type,
            "n_contexts": len(group),
            "n_expected_active": active,
            "mean_predicted_score": rounded(safe_mean(vals)),
            "se_predicted_score": rounded(safe_stderr(vals)),
            "median_predicted_score": rounded(statistics.median([v for v in vals if math.isfinite(v)]) if any(math.isfinite(v) for v in vals) else float("nan")),
            "max_predicted_score": rounded(safe_max(vals)),
            "share_above_failure_threshold": rounded(above / len(group) if group else float("nan")),
        })
    return rows


def calibration_bin_rows(calibration: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.000001)]
    rows: list[dict[str, Any]] = []
    for method in sorted({str(r.get("method", "")) for r in calibration}):
        method_rows = [r for r in calibration if str(r.get("method", "")) == method]
        for lo, hi in bins:
            group = [r for r in method_rows if lo <= fnum(r.get("confidence"), -1.0) < hi]
            if not group:
                continue
            successes = sum(1 for r in group if truthy(r.get("success")))
            confs = [fnum(r.get("confidence")) for r in group]
            rate = successes / len(group)
            rows.append({
                "calibration_bin_id": stable_row_id(method, lo, hi),
                "method": method,
                "method_role": method_role(method),
                "confidence_bin_low": rounded(lo, 2),
                "confidence_bin_high": rounded(min(1.0, hi), 2),
                "n_rows": len(group),
                "mean_confidence": rounded(safe_mean(confs)),
                "success_rate": rounded(rate),
                "success_rate_se": rounded(binary_se(successes, len(group))),
                "calibration_gap": rounded(safe_mean(confs) - rate),
            })
    return rows


def abstention_summary_rows(scores: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, bool], list[Mapping[str, Any]]] = defaultdict(list)
    for row in scores:
        grouped[(str(row.get("method", "")), str(row.get("feature_type", "")), truthy(row.get("expected_abstain")))].append(row)
    rows: list[dict[str, Any]] = []
    for (method, feature_type, expected_abstain), group in sorted(grouped.items()):
        n = len(group)
        abstained = sum(1 for r in group if truthy(r.get("abstain")))
        forced = n - abstained
        rows.append({
            "abstention_summary_id": stable_row_id(method, feature_type, expected_abstain),
            "method": method,
            "method_role": method_role(method),
            "feature_type": feature_type,
            "expected_abstain": expected_abstain,
            "n_features": n,
            "n_abstained": abstained,
            "n_forced_labels": forced,
            "abstention_rate": rounded(abstained / n if n else float("nan")),
            "forced_label_rate": rounded(forced / n if n else float("nan")),
            "mean_review_priority": rounded(safe_mean([r.get("review_priority") for r in group])),
        })
    return rows


def failure_specimen_rows(
    scores: Sequence[Mapping[str, Any]],
    tests: Sequence[Mapping[str, Any]],
    counterexamples: Sequence[Mapping[str, Any]],
    limit: int = 24,
) -> list[dict[str, Any]]:
    by_feature_method = {(str(r.get("feature_id")), str(r.get("method"))): r for r in scores}
    test_lookup: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in tests:
        test_lookup[(str(row.get("feature_id")), str(row.get("method")))].append(row)
    candidates: list[dict[str, Any]] = []
    for ce in counterexamples:
        key = (str(ce.get("feature_id")), str(ce.get("method")))
        score = by_feature_method.get(key, {})
        method_tests = test_lookup.get(key, [])
        worst = sorted(
            [r for r in method_tests if not truthy(r.get("expected_active"))],
            key=lambda r: fnum(r.get("predicted_score"), -999.0),
            reverse=True,
        )[:2]
        best_pos = sorted(
            [r for r in method_tests if truthy(r.get("expected_active"))],
            key=lambda r: fnum(r.get("predicted_score"), -999.0),
            reverse=True,
        )[:1]
        for role, rows in (("highest_scoring_negative", worst), ("best_positive", best_pos)):
            for t in rows:
                candidates.append({
                    "specimen_id": stable_row_id(key[0], key[1], ce.get("kind"), role, t.get("context_id", "")),
                    "feature_id": key[0],
                    "feature_type": score.get("feature_type", ""),
                    "method": key[1],
                    "generated_label": score.get("generated_label", ""),
                    "gold_label": score.get("gold_label", ""),
                    "failure_kind": ce.get("kind", ""),
                    "specimen_role": role,
                    "suite": t.get("suite", ""),
                    "context": t.get("context", ""),
                    "predicted_score": t.get("predicted_score", ""),
                    "heldout_auc": score.get("heldout_auc", ""),
                    "same_feature_control_auc": score.get("same_feature_control_auc", ""),
                    "control_gap_vs_same_feature_shuffled": score.get("control_gap_vs_same_feature_shuffled", ""),
                    "review_priority": score.get("review_priority", ""),
                    "lesson": ce.get("lesson", ""),
                })
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in sorted(candidates, key=lambda r: fnum(r.get("review_priority"), 0.0), reverse=True):
        sid = str(row.get("specimen_id"))
        if sid in seen:
            continue
        seen.add(sid)
        out.append(row)
        if len(out) >= limit:
            break
    return out


def write_failure_specimens(ctx: bench.RunContext, specimens: Sequence[Mapping[str, Any]]) -> None:
    md = [
        "# Lab 31 failure specimens",
        "",
        "These are not bloopers. They are the rows that keep automated labels honest.",
        "Each specimen connects a feature, method, failure kind, source context, and the row-level control gap.",
        "",
    ]
    if not specimens:
        md.append("No failure specimens crossed the current thresholds. Human review is still required before citing labels.")
    for row in specimens:
        md.extend([
            f"## `{row.get('feature_id', '')}` · `{row.get('method', '')}` · `{row.get('failure_kind', '')}`",
            "",
            f"- generated label: `{row.get('generated_label', '') or 'ABSTAIN'}`; gold: `{row.get('gold_label', '')}`",
            f"- specimen role: `{row.get('specimen_role', '')}`; suite: `{row.get('suite', '')}`; predicted score: `{row.get('predicted_score', '')}`",
            f"- held-out AUC: `{row.get('heldout_auc', '')}`; same-feature control AUC: `{row.get('same_feature_control_auc', '')}`; gap: `{row.get('control_gap_vs_same_feature_shuffled', '')}`",
            f"- lesson: {row.get('lesson', '')}",
            "",
            "> " + str(row.get("context", "")).replace("\n", " ")[:900],
            "",
        ])
    path = ctx.path("cards", "failure_specimens.md")
    bench.write_text(path, "\n".join(md) + "\n")
    ctx.register_artifact(path, "summary", "Specimen-level failures and counterexamples for Lab 31.")
    jsonl = ctx.path("tables", "failure_specimens.jsonl")
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write_text("".join(json.dumps(dict(r), sort_keys=True) + "\n" for r in specimens), encoding="utf-8")
    ctx.register_artifact(jsonl, "table", "JSONL failure specimens for automated review or notebooks.")


def write_run_config_snapshot(ctx: bench.RunContext, data_info: Mapping[str, Any]) -> None:
    payload = {
        "lab": LAB_ID,
        "run_name": ctx.run_dir.name,
        "tier": ctx.args.tier,
        "prompt_set": ctx.args.prompt_set,
        "max_examples": ctx.args.max_examples,
        "seed": ctx.args.seed,
        "model_id": ctx.model_id or ctx.args.model,
        "model_revision": ctx.model_revision,
        "dtype": ctx.args.dtype,
        "quantization": ctx.args.quantization,
        "decoding_settings": "none: Lab 31 is offline and does not generate text or use an LLM judge",
        "data": dict(data_info),
        "methods": list(METHODS),
        "automated_methods": sorted(AUTOMATED_METHODS),
        "thresholds": {
            "auc_support_bar": AUC_SUPPORT_BAR,
            "label_accuracy_bar": LABEL_ACCURACY_BAR,
            "control_gap_bar": CONTROL_GAP_BAR,
            "good_abstain_bar": GOOD_ABSTAIN_BAR,
            "bad_abstain_bar": BAD_ABSTAIN_BAR,
            "confusable_failure_bar": CONFUSABLE_FAILURE_BAR,
            "decoy_failure_rate_bar": DECOY_FAILURE_RATE_BAR,
            "confusable_failure_threshold": CONFUSABLE_FAILURE_THRESHOLD,
            "key_deletion_fragility_bar": KEY_DELETION_FRAGILITY_BAR,
        },
        "scoring_semantics": "generated labels are lexical activation hypotheses over held-out/context-control suites; success is a method-under-suite result, not feature meaning",
    }
    path = ctx.path("diagnostics", "run_config_snapshot.json")
    bench.write_json(path, payload)
    ctx.register_artifact(path, "diagnostic", "Lab-specific run configuration snapshot for reproducing Lab 31 tables and figures.")


def write_warning_summary(
    ctx: bench.RunContext,
    data_info: Mapping[str, Any],
    hook_check: Mapping[str, Any],
    lens_check: Mapping[str, Any],
    patch_noop: Mapping[str, Any],
    schema_rows: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
    review: Sequence[Mapping[str, Any]],
    counterexamples: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(severity: str, code: str, scope: str, detail: str, artifact: str, n: int | str = "") -> None:
        rows.append({"severity": severity, "code": code, "scope": scope, "n": n, "detail": detail, "artifact_to_inspect": artifact})

    if data_info.get("fallback_data"):
        add("warning", "fallback_data_smoke_only", "data", "Built-in Tier A fallback was used; do not ledger broad science claims.", "diagnostics/data_manifest.json")
    if not data_info.get("science_ready"):
        add("warning", "not_science_ready", "run", "Science-ready flag is false. Read this as a plumbing or audit run.", "diagnostics/self_check_status.json")
    bad_schema = [r for r in schema_rows if not truthy(r.get("schema_ok"))]
    if bad_schema:
        add("error", "schema_rows_failed", "data", "One or more feature task rows failed schema/context-count checks.", "diagnostics/schema_audit.csv", len(bad_schema))
    for name, check, artifact in [
        ("hook_parity_failed", hook_check, "diagnostics/hook_parity.json"),
        ("lens_self_check_failed", lens_check, "diagnostics/logit_lens_self_check.json"),
        ("patch_noop_failed", patch_noop, "diagnostics/patch_noop.json"),
    ]:
        if not check.get("ok"):
            add("error", name, "instrument", f"Shared bench check did not report ok=True: {name}.", artifact)
    control_auc = safe_mean([r.get("heldout_auc") for r in scores if r.get("method") == "shuffled_top_context_control" and not truthy(r.get("abstain"))], default=0.5)
    if math.isfinite(control_auc) and control_auc >= AUC_SUPPORT_BAR:
        add("warning", "shuffled_control_high_auc", "control", "The shuffled-top-context control itself clears the AUC bar; method gaps matter more than raw AUC.", "tables/explanation_scores.csv", round(control_auc, 3))
    gold_auc = safe_mean([r.get("heldout_auc") for r in scores if r.get("method") == "gold_calibration" and not truthy(r.get("abstain"))], default=float("nan"))
    if math.isfinite(gold_auc) and gold_auc < AUC_SUPPORT_BAR:
        add("warning", "gold_calibration_low", "suite", "Gold calibration failed to clear the AUC bar; the test suite or scorer may be broken.", "tables/explanation_scores.csv", round(gold_auc, 3))
    supported = [r for r in evidence if r.get("claim_posture") == "auto_label_audit_supported"]
    if not supported:
        add("warning", "no_supported_automated_method", "evidence", "No automated method cleared all claim gates. This can be an honest negative result.", "tables/auto_interp_evidence_matrix.csv")
    if not review:
        add("warning", "review_queue_empty", "review", "The review queue is empty. Check whether controls are too easy or review triage is too permissive.", "tables/human_review_queue.csv")
    if counterexamples:
        add("info", "counterexamples_present", "audit", "Counterexamples were found and should be read before writing broad claims.", "tables/auto_interp_counterexamples.csv", len(counterexamples))
    if ctx.args.no_plots:
        add("info", "plots_skipped_by_flag", "plots", "PNG generation was skipped. Figure source tables and plot_manifest.json are still written.", "plot_manifest.json")
    path = ctx.path("diagnostics", "warning_summary.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "diagnostic", "Central warning and caveat summary for Lab 31.")
    json_path = ctx.path("diagnostics", "warning_summary.json")
    bench.write_json(json_path, rows)
    ctx.register_artifact(json_path, "diagnostic", "Machine-readable central warning and caveat summary for Lab 31.")
    return rows


def write_extra_tables(
    ctx: bench.RunContext,
    suite_summary: Sequence[Mapping[str, Any]],
    calibration_bins: Sequence[Mapping[str, Any]],
    abstention_summary: Sequence[Mapping[str, Any]],
    failure_specimens: Sequence[Mapping[str, Any]],
) -> None:
    specs = [
        ("tables/suite_score_summary.csv", suite_summary, "Predicted-score summaries by method, suite, feature type, and control role."),
        ("tables/calibration_bins.csv", calibration_bins, "Confidence-bin success rates for calibration figures and writeups."),
        ("tables/abstention_summary.csv", abstention_summary, "Abstention/forced-label summary by method and feature type."),
        ("tables/failure_specimens.csv", failure_specimens, "Selected positive and negative specimen contexts tied to counterexamples."),
    ]
    for rel, rows, desc in specs:
        path = ctx.path(*rel.split("/"))
        bench.write_csv_with_context(ctx, path, rows)
        ctx.register_artifact(path, "table", desc)



# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------


def write_tables(
    ctx: bench.RunContext,
    explanations: Sequence[Mapping[str, Any]],
    deletion_rows: Sequence[Mapping[str, Any]],
    tests: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    review: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    counterexamples: Sequence[Mapping[str, Any]],
) -> None:
    specs = [
        ("tables/generated_explanations.csv", explanations, "Generated labels/explanations with confidence, abstention, and human-review fields."),
        ("tables/key_token_deletion_audit.csv", deletion_rows, "Key-token deletion and confusable-overlap audit for automated methods."),
        ("tables/explanation_tests.csv", tests, "Held-out, paraphrase, negative, confusable, and token-overlap test rows."),
        ("tables/explanation_scores.csv", scores, "Feature/method explanation scores."),
        ("tables/auto_interp_evidence_matrix.csv", evidence, "Method-level evidence matrix for automated interpretability."),
        ("tables/evidence_matrix.csv", evidence, "Standard-schema alias of the method-level evidence matrix."),
        ("tables/feature_evidence_matrix.csv", feature_rows, "Feature-level best automated label and review posture."),
        ("tables/human_review_queue.csv", review, "Rows needing student or reviewer labels before broad claims."),
        ("tables/confidence_calibration.csv", calibration, "Per-feature confidence versus success calibration rows."),
        ("tables/auto_interp_counterexamples.csv", counterexamples, "Counterexamples that narrow or defeat auto-label claims."),
    ]
    for rel, rows, desc in specs:
        path = ctx.path(*rel.split("/"))
        bench.write_csv_with_context(ctx, path, rows)
        ctx.register_artifact(path, "table", desc)
    results = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results, evidence)
    ctx.register_artifact(results, "table", "Alias of tables/auto_interp_evidence_matrix.csv for dashboard tooling.")


def write_state(ctx: bench.RunContext, data_info: Mapping[str, Any]) -> None:
    payload = {
        "lab": "lab31",
        "data": data_info,
        "methods": list(METHODS),
        "automated_methods": sorted(AUTOMATED_METHODS),
        "domain_lexicon_sizes": {k: len(v) for k, v in DOMAIN_LEXICON.items()},
        "thresholds": {
            "auc_support_bar": AUC_SUPPORT_BAR,
            "label_accuracy_bar": LABEL_ACCURACY_BAR,
            "control_gap_bar": CONTROL_GAP_BAR,
            "good_abstain_bar": GOOD_ABSTAIN_BAR,
            "bad_abstain_bar": BAD_ABSTAIN_BAR,
            "confusable_failure_bar": CONFUSABLE_FAILURE_BAR,
            "decoy_failure_rate_bar": DECOY_FAILURE_RATE_BAR,
            "confusable_failure_threshold": CONFUSABLE_FAILURE_THRESHOLD,
            "key_deletion_fragility_bar": KEY_DELETION_FRAGILITY_BAR,
        },
        "scoring_semantics": "Labels are scored as cheap lexical activation hypotheses over held-out positives, paraphrases, hard negatives, confusables, and token-overlap decoys. This is an audit suite, not a semantic oracle.",
    }
    path = ctx.path("state", "auto_interp_config.json")
    bench.write_json(path, payload)
    ctx.register_artifact(path, "state", "Lab 31 methods, thresholds, and scoring configuration.")


def write_method_card(ctx: bench.RunContext, data_info: Mapping[str, Any], metrics: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Lab 31 method card",
        "",
        "This lab evaluates auto-labels as hypotheses. It does not treat automated explanations as feature meanings.",
        "",
        f"- data source: `{data_info.get('data_source')}`",
        f"- science_ready: `{str(data_info.get('science_ready')).lower()}`",
        "- methods: majority-domain, structured local, test-aware abstention, cross-feature shuffled control, and gold calibration upper bound",
        "- tests: held-out positives, paraphrase positives, hard negatives, confusable negatives, and token-overlap decoys",
        "- evidence rung: `AUDIT + DECODE`",
        "- forbidden claim: the automated label is the feature's meaning",
        "- best automated method by score: `" + str(metrics.get("best_method", "")) + "`",
        "",
        "| method | mean AUC | control gap | label accuracy | abstention | good abstain | bad abstain | decoy failures | posture |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in evidence:
        lines.append(
            f"| {row['method']} | {row['mean_heldout_auc']} | {row['control_gap_vs_shuffled_top_context']} | "
            f"{row['label_accuracy_when_scored']} | {row['abstention_rate']} | "
            f"{row['good_abstain_rate_on_polysemantic_or_random']} | {row['bad_abstain_rate_on_gold_features']} | "
            f"{row['mean_decoy_failure_rate']} | {row['claim_posture']} |"
        )
    lines += [
        "",
        "Safe sentence: `Method E predicted held-out feature tests under this suite, while abstaining on high-risk features at rate R.`",
        "",
        "Unsafe sentence: `The automated label is the feature's meaning.`",
    ]
    path = ctx.path("method_card.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Lab 31 method card and auto-label verdicts.")


def write_operationalization_audit(ctx: bench.RunContext, data_info: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]], counterexamples: Sequence[Mapping[str, Any]]) -> None:
    supported = sum(1 for row in evidence if row.get("claim_posture") == "auto_label_audit_supported")
    audit_result = "passed" if supported else "mixed_or_failed"
    if not data_info.get("science_ready"):
        audit_result = "not_science_run"
    claim_allowed = "audited label-prediction handle" if supported and data_info.get("science_ready") else "no broad auto-label claim"
    lines = [
        "# Lab 31 operationalization audit",
        "",
        "```yaml",
        'headline_claim: "automated explanations can scale feature interpretation"',
        'cheap_explanation: "the label is keyword overlap, confusable-domain leakage, uncalibrated confidence, or forced wording for a polysemantic/random feature"',
        'killer_control: "held-out positives versus hard negatives, confusables, token-overlap decoys, shuffled-top-context controls, key-token deletion, and human review"',
        f'result: "{audit_result}"',
        f'claim_allowed: "{claim_allowed}"',
        "```",
        "",
        "## What the measurement can say",
        "",
        "A label method predicted held-out activation-test contexts better than hard, confusable, and token-overlap negatives on this frozen suite.",
        "",
        "## What it cannot say",
        "",
        "It cannot say the label is complete, unique, monosemantic, deployment-ready, or the feature's meaning.",
        "",
        "## Method verdicts",
        "",
    ]
    for row in evidence:
        lines.append(f"- `{row['method']}`: `{row['claim_posture']}`; mean AUC `{row['mean_heldout_auc']}`; control gap `{row['control_gap_vs_shuffled_top_context']}`.")
    lines += ["", "## Counterexamples", ""]
    if counterexamples:
        for row in list(counterexamples)[:24]:
            lines.append(f"- `{row['method']}` on `{row['feature_id']}`: `{row['kind']}`. {row['lesson']}")
    else:
        lines.append("- No automatic counterexample crossed thresholds. This does not remove the human-review requirement.")
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Controls and non-claims for Lab 31 automated interpretability.")


def write_run_summary(ctx: bench.RunContext, data_info: Mapping[str, Any], metrics: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]], counterexamples: Sequence[Mapping[str, Any]]) -> None:
    supported = int(metrics.get("supported_methods", 0))
    if supported > 0 and data_info.get("science_ready"):
        surviving_claim = (
            "At least one automated method predicted held-out positives versus negatives, beat the shuffled-top-context control, "
            "and abstained on high-risk features under this suite. The claim is about audit performance, not feature meaning."
        )
    elif not data_info.get("science_ready"):
        surviving_claim = "This is a smoke/plumbing run. Do not move results into the ledger until the frozen JSONL and schema pass."
    else:
        surviving_claim = (
            "No automated method cleared every support gate. The surviving result is an audit: the suite identified where labels fail, "
            "especially controls, confusables, decoys, calibration, or abstention."
        )
    main_counter = counterexamples[0]["lesson"] if counterexamples else "No automatic counterexample crossed thresholds; inspect the review queue anyway."
    lines = [
        "# Lab 31 run summary: automated interpretability at scale",
        "",
        f"- data rows: {data_info['n_rows_selected']} selected from `{pathlib.Path(str(data_info['data_path'])).name}`",
        f"- data source: `{data_info.get('data_source')}`",
        f"- feature types: `{data_info['feature_types']}`",
        f"- science_ready: `{str(data_info.get('science_ready')).lower()}`",
        f"- best method: `{metrics['best_method']}`",
        f"- supported methods: `{metrics['supported_methods']}` / `{metrics['n_methods']}`",
        f"- human review rows: `{metrics['n_review_rows']}`",
        f"- main counterexample: {main_counter}",
        "",
        "## Method verdicts",
        "",
        "| method | mean AUC | control gap | label accuracy | good abstain | bad abstain | posture |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in evidence:
        lines.append(
            f"| `{row['method']}` | {row['mean_heldout_auc']} | {row['control_gap_vs_shuffled_top_context']} | "
            f"{row['label_accuracy_when_scored']} | {row['good_abstain_rate_on_polysemantic_or_random']} | "
            f"{row['bad_abstain_rate_on_gold_features']} | {row['claim_posture']} |"
        )
    lines += [
        "",
        "## Reading order",
        "",
        "1. `method_card.md` for method boundaries and non-claims.",
        "2. `diagnostics/schema_audit.csv` and `diagnostics/data_manifest.json` for data validity.",
        "3. `tables/generated_explanations.csv` for labels, confidence, abstention, evidence terms, and human-review fields.",
        "4. `tables/explanation_tests.csv` for the held-out/confusable/decoy suite.",
        "5. `tables/explanation_scores.csv` and `tables/auto_interp_evidence_matrix.csv` for quantitative verdicts.",
        "6. `tables/auto_interp_counterexamples.csv` and `tables/human_review_queue.csv` before trusting any label in a writeup.",
        "",
        "## Smallest surviving claim",
        "",
        surviving_claim,
        "",
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Lab 31 run summary and reading order.")



def write_review_artifacts(ctx: bench.RunContext, explanations: Sequence[Mapping[str, Any]], scores: Sequence[Mapping[str, Any]], review: Sequence[Mapping[str, Any]]) -> None:
    guide = [
        "# Lab 31 human review guide",
        "",
        "Fill the shared review columns before citing an automated label as course evidence:",
        "",
        "```text",
        ",".join(REVIEW_FIELDS),
        "```",
        "",
        "Review priority is highest for forced labels on high-risk features, token-overlap decoy failures, and wrong-label rows.",
        "A useful review note names one supporting context and one counterexample or falsifier.",
        "",
    ]
    path = ctx.path("human_review_guide.md")
    bench.write_text(path, "\n".join(guide))
    ctx.register_artifact(path, "summary", "Human-review rubric for generated automated-interpretability labels.")

    score_lookup = {(str(r.get("feature_id")), str(r.get("method"))): r for r in scores}
    lines = [
        "# Lab 31 explanation cards",
        "",
        "Each card is a testable label hypothesis. It is not a feature meaning.",
        "",
    ]
    for exp in explanations:
        if exp.get("method") not in AUTOMATED_METHODS:
            continue
        score = score_lookup.get((str(exp.get("feature_id")), str(exp.get("method"))), {})
        lines.extend([
            f"## `{exp.get('feature_id', '')}` · `{exp.get('method', '')}`",
            "",
            f"- generated label: `{exp.get('generated_label') or 'ABSTAIN'}`",
            f"- raw candidate: `{exp.get('raw_candidate_label', '')}`",
            f"- gold label: `{exp.get('gold_label', '')}`",
            f"- confidence: `{exp.get('confidence', '')}`",
            f"- held-out AUC: `{score.get('heldout_auc', '')}`; same-feature control gap: `{score.get('control_gap_vs_same_feature_shuffled', '')}`",
            f"- confusable failures: `{score.get('confusable_failure_count', '')}`; decoy failure rate: `{score.get('token_overlap_decoy_failure_rate', '')}`",
            f"- abstain: `{exp.get('abstain')}`; review required: `{exp.get('human_review_required')}`",
            f"- risk flags: `{exp.get('risk_flags', '')}`",
            "",
        ])
    path = ctx.path("cards", "explanation_cards.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Per-feature generated-label cards for Lab 31 review.")


def write_claims(ctx: bench.RunContext, data_info: Mapping[str, Any], evidence: Sequence[Mapping[str, Any]]) -> None:
    run_name = ctx.run_dir.name
    claims = []
    for i, row in enumerate(evidence, start=1):
        tag = "AUDIT,DECODE" if row.get("method") in AUTOMATED_METHODS else "AUDIT"
        caveat = " This was not a science-ready run." if not data_info.get("science_ready") else ""
        claims.append({
            "id": f"{LAB_ID}-C{i}",
            "tag": tag,
            "text": (
                f"Method `{row['method']}` reached mean held-out AUC {row['mean_heldout_auc']}, "
                f"control gap {row['control_gap_vs_shuffled_top_context']}, label accuracy {row['label_accuracy_when_scored']}, "
                f"abstention rate {row['abstention_rate']}, good abstention {row['good_abstain_rate_on_polysemantic_or_random']}, "
                f"and mean confusable failures {row['mean_confusable_failures']} on the Lab 31 suite. "
                f"Posture: {row['claim_posture']}. This is an audit-performance claim, not a feature-meaning claim.{caveat}"
            ),
            "artifact": f"runs/{run_name}/tables/auto_interp_evidence_matrix.csv",
            "falsifier": "Held-out positives, confusable negatives, token-overlap decoys, shuffled-context controls, key-token deletion, or human review invalidate the generated label.",
        })
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)


def write_status_files(ctx: bench.RunContext, data_info: Mapping[str, Any], hook_check: Mapping[str, Any], lens_check: Mapping[str, Any], patch_noop: Mapping[str, Any], metrics: Mapping[str, Any], schema_rows: Sequence[Mapping[str, Any]]) -> None:
    safety = {
        "lab": "lab31",
        "unsafe_prompt_sampling": False,
        "refusal_ablation": False,
        "harmful_completion_generation": False,
        "generated_text_scoring": False,
        "automated_explanation_scoring": True,
        "blocked_rows": 0,
        "public_private_boundary_relevant": False,
        "science_ready": bool(data_info.get("science_ready")),
        "note": "Offline, benign synthetic feature-context audit. No model-generated harmful text or LLM judge is used.",
    }
    path = ctx.path("diagnostics", "safety_status.json")
    bench.write_json(path, safety)
    ctx.register_artifact(path, "diagnostic", "Safety and scope status for Lab 31.")
    schema_ok = all(bool(r.get("schema_ok")) for r in schema_rows)
    checks = {
        "hook_parity_ok": bool(hook_check.get("ok")),
        "lens_self_check_ok": bool(lens_check.get("ok")),
        "patch_noop_ok": bool(patch_noop.get("ok")),
        "data_schema_ok": schema_ok,
        "science_ready": bool(data_info.get("science_ready")),
        "generated_explanations_nonempty": metrics.get("n_explanations", 0) > 0,
        "score_rows_nonempty": metrics.get("n_methods", 0) > 0,
        "review_queue_rows": metrics.get("n_review_rows", 0),
        "counterexamples": metrics.get("n_counterexamples", 0),
        "ok_for_science": bool(hook_check.get("ok")) and bool(lens_check.get("ok")) and bool(patch_noop.get("ok")) and schema_ok and bool(data_info.get("science_ready")),
    }
    path = ctx.path("diagnostics", "self_check_status.json")
    bench.write_json(path, checks)
    ctx.register_artifact(path, "diagnostic", "Aggregated self-check status for Lab 31.")


def write_plot_guide(ctx: bench.RunContext) -> None:
    rows = [
        {"plot": "plots/auto_interp_dashboard.png", "source_artifact": "tables/figure_auto_interp_dashboard_source.csv", "first_question": "Which methods beat controls while preserving abstention discipline?", "concept": "One-screen method verdict.", "what_not_to_claim": "Do not call a generated label a feature meaning."},
        {"plot": "plots/target_vs_control.png", "source_artifact": "tables/figure_target_vs_control_source.csv", "first_question": "Does each automated label beat its same-feature shuffled-context control?", "concept": "Raw feature-method control gap without method-level averaging.", "what_not_to_claim": "Do not cite high AUC rows that sit on the control diagonal."},
        {"plot": "plots/explanation_quality_matrix.png", "source_artifact": "tables/figure_explanation_quality_matrix_source.csv", "first_question": "Which feature-method cells carry the method-level result?", "concept": "Method-by-feature AUC matrix.", "what_not_to_claim": "A bright specimen is not a scalable method claim."},
        {"plot": "plots/label_score_distribution.png", "source_artifact": "tables/figure_label_score_distribution_source.csv", "first_question": "Are positives separated from hard/confusable/decoy negatives?", "concept": "Raw context-score distribution by suite.", "what_not_to_claim": "Do not hide overlap under a mean AUC."},
        {"plot": "plots/confidence_calibration_curve.png", "source_artifact": "tables/figure_confidence_calibration_source.csv", "first_question": "Does confidence predict actual success?", "concept": "Calibration bins and row-level outcomes.", "what_not_to_claim": "Confidence is not correctness."},
        {"plot": "plots/abstention_frontier.png", "source_artifact": "tables/figure_abstention_frontier_source.csv", "first_question": "Does the method abstain on risky features without refusing labelable ones?", "concept": "Good versus bad abstention.", "what_not_to_claim": "Abstaining on everything is not interpretability."},
        {"plot": "plots/confusable_pair_failure_atlas.png", "source_artifact": "tables/figure_confusable_pair_failure_source.csv", "first_question": "Which controls expose cheap keyword explanations?", "concept": "Confusable and token-overlap failure atlas.", "what_not_to_claim": "A label that fires on decoys is not concept evidence."},
        {"plot": "plots/random_feature_sanity_panel.png", "source_artifact": "tables/figure_random_feature_sanity_source.csv", "first_question": "Do random/polysemantic/ambiguous controls receive forced labels?", "concept": "High-risk forced-label sanity check.", "what_not_to_claim": "A method that always emits labels is not calibrated."},
        {"plot": "plots/review_queue_triage.png", "source_artifact": "tables/figure_review_queue_triage_source.csv", "first_question": "Which rows should the student inspect first?", "concept": "Human-review queue by priority and failure reason.", "what_not_to_claim": "Automated scoring replaces human label review."},
        {"plot": "plots/paired_examples.png", "source_artifact": "tables/figure_paired_examples_source.csv", "first_question": "Which specimens support or contradict the aggregate?", "concept": "Same-feature control-to-method paired rows.", "what_not_to_claim": "Do not let aggregate bars erase counterexamples."},
    ]
    path = ctx.path("tables", "plot_reading_guide.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Plot reading guide for Lab 31.")


def write_placeholder(ctx: bench.RunContext, name: str, title: str, message: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=10, wrap=True)
    bench.save_figure(ctx, fig, name, title)


def write_source_table(ctx: bench.RunContext, filename: str, rows: Sequence[Mapping[str, Any]], description: str) -> pathlib.Path:
    path = ctx.path("tables", filename)
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", description)
    return path


def add_plot_entry(
    entries: list[dict[str, Any]],
    *,
    figure: str,
    source: pathlib.Path,
    row_count: int,
    question: str,
    metric: str,
    control: str,
    claim: str,
    generated: bool,
    warnings: Sequence[str],
) -> None:
    entries.append({
        "figure_path": f"plots/{figure}",
        "source_table": f"tables/{source.name}",
        "source_table_rel": f"tables/{source.name}",
        "row_count": row_count,
        "question_answered": question,
        "metric": metric,
        "control_or_falsifier": control,
        "claim_supported_or_narrowed": claim,
        "generated": bool(generated),
        "warnings": ";".join(warnings),
    })


def write_plot_manifest(ctx: bench.RunContext, entries: Sequence[Mapping[str, Any]]) -> None:
    # Normalize the small relative path field in case callers passed only filenames.
    rows = []
    for entry in entries:
        e = dict(entry)
        if "source_table_rel" not in e and e.get("source_table"):
            e["source_table_rel"] = str(e["source_table"])
        rows.append(e)
    json_path = ctx.path("plot_manifest.json")
    bench.write_json(json_path, rows)
    ctx.register_artifact(json_path, "manifest", "Figure manifest with source table, row count, metric, controls, and claim boundaries.")
    csv_path = ctx.path("tables", "plot_manifest.csv")
    bench.write_csv_with_context(ctx, csv_path, rows)
    ctx.register_artifact(csv_path, "table", "CSV copy of plot_manifest.json.")


def figure_source_tables(
    ctx: bench.RunContext,
    evidence: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
    tests: Sequence[Mapping[str, Any]],
    review: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    calibration_bins: Sequence[Mapping[str, Any]],
    suite_summary: Sequence[Mapping[str, Any]],
    abstention_summary: Sequence[Mapping[str, Any]],
    failure_specimens: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[pathlib.Path, int]]:
    sources: dict[str, tuple[pathlib.Path, int]] = {}

    def add(key: str, filename: str, rows: Sequence[Mapping[str, Any]], desc: str) -> None:
        sources[key] = (write_source_table(ctx, filename, list(rows), desc), len(rows))

    add("dashboard", "figure_auto_interp_dashboard_source.csv", evidence, "Method-level evidence matrix source for auto_interp_dashboard.png.")
    target_rows = [dict(r) for r in scores if r.get("method") in AUTOMATED_METHODS]
    add("target_vs_control", "figure_target_vs_control_source.csv", target_rows, "Row-level automated-method AUC versus same-feature shuffled control source.")
    add("quality_matrix", "figure_explanation_quality_matrix_source.csv", scores, "Feature-method score source for explanation_quality_matrix.png.")
    # Keep raw test rows, not only summaries, so overlapping score distributions cannot hide.
    dist_rows = [dict(r) for r in tests if r.get("method") in AUTOMATED_METHODS]
    add("score_distribution", "figure_label_score_distribution_source.csv", dist_rows, "Raw context-level predicted scores by suite for label_score_distribution.png.")
    cal_rows = [dict(r) for r in calibration]
    cal_rows.extend({"row_type": "bin", **dict(r)} for r in calibration_bins)
    add("calibration", "figure_confidence_calibration_source.csv", cal_rows, "Confidence row outcomes and calibration bins source.")
    frontier_rows = [dict(r) for r in evidence]
    frontier_rows.extend({"row_type": "abstention_summary", **dict(r)} for r in abstention_summary)
    add("abstention_frontier", "figure_abstention_frontier_source.csv", frontier_rows, "Good/bad abstention source rows.")
    conf_rows = [dict(r) for r in evidence]
    conf_rows.extend({"row_type": "suite_summary", **dict(r)} for r in suite_summary if r.get("suite") in {"confusable_negative", "token_overlap_decoy"})
    add("confusable_failure", "figure_confusable_pair_failure_source.csv", conf_rows, "Confusable and token-overlap control source rows.")
    high_risk = [dict(r) for r in scores if r.get("feature_type") in {"random_control", "polysemantic_gold", "ambiguous_control"}]
    add("random_sanity", "figure_random_feature_sanity_source.csv", high_risk, "High-risk feature abstention and forced-label source rows.")
    add("review_triage", "figure_review_queue_triage_source.csv", review, "Human-review queue triage source rows.")
    # Select both supporting and contradictory rows for paired examples.
    automated = [r for r in scores if r.get("method") in AUTOMATED_METHODS]
    strongest = sorted(automated, key=lambda r: fnum(r.get("control_gap_vs_same_feature_shuffled"), -999.0), reverse=True)[:8]
    weakest = sorted(automated, key=lambda r: fnum(r.get("control_gap_vs_same_feature_shuffled"), 999.0))[:8]
    paired: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in strongest + weakest:
        rid = str(row.get("score_id", row.get("feature_method_id", "")))
        if rid in seen:
            continue
        seen.add(rid)
        paired.append({
            "score_id": rid,
            "feature_id": row.get("feature_id", ""),
            "feature_type": row.get("feature_type", ""),
            "method": row.get("method", ""),
            "generated_label": row.get("generated_label", ""),
            "same_feature_control_auc": row.get("same_feature_control_auc", ""),
            "heldout_auc": row.get("heldout_auc", ""),
            "control_gap_vs_same_feature_shuffled": row.get("control_gap_vs_same_feature_shuffled", ""),
            "review_priority": row.get("review_priority", ""),
            "role": "support" if fnum(row.get("control_gap_vs_same_feature_shuffled")) >= CONTROL_GAP_BAR else "counterexample_or_weak",
        })
    add("paired_examples", "figure_paired_examples_source.csv", paired, "Selected row-level support and counterexample pairs.")
    add("failure_specimens", "figure_failure_specimens_source.csv", failure_specimens, "Specimen contexts referenced by paired examples and audits.")
    return sources


def write_plots(
    ctx: bench.RunContext,
    evidence: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
    tests: Sequence[Mapping[str, Any]],
    review: Sequence[Mapping[str, Any]],
    calibration: Sequence[Mapping[str, Any]],
    calibration_bins: Sequence[Mapping[str, Any]],
    suite_summary: Sequence[Mapping[str, Any]],
    abstention_summary: Sequence[Mapping[str, Any]],
    failure_specimens: Sequence[Mapping[str, Any]],
    warning_rows: Sequence[Mapping[str, Any]],
) -> None:
    write_plot_guide(ctx)
    sources = figure_source_tables(ctx, evidence, scores, tests, review, calibration, calibration_bins, suite_summary, abstention_summary, failure_specimens)
    warnings = warning_codes(warning_rows)
    entries: list[dict[str, Any]] = []
    figure_specs = [
        ("auto_interp_dashboard.png", "dashboard", "Which methods beat controls while preserving abstention discipline?", "mean held-out AUC, label accuracy, abstention, specificity failures", "shuffled-top-context control and high-risk abstention", "method-level audit posture"),
        ("target_vs_control.png", "target_vs_control", "Does each automated label beat its same-feature shuffled-context control?", "heldout_auc vs same_feature_control_auc", "diagonal y=x plus same-feature shuffled control", "row-level control specificity"),
        ("explanation_quality_matrix.png", "quality_matrix", "Which feature-method cells carry the result?", "heldout AUC by feature and method", "control and gold rows shown beside automated methods", "breadth versus specimen caveat"),
        ("label_score_distribution.png", "score_distribution", "Are positives separated from hard/confusable/decoy negatives?", "raw predicted context score", "hard negatives, confusables, token-overlap decoys", "separation evidence or overlap caveat"),
        ("confidence_calibration_curve.png", "calibration", "Does confidence predict success?", "confidence bin success rate", "perfect-calibration diagonal", "calibration claim boundary"),
        ("abstention_frontier.png", "abstention_frontier", "Does abstention target risk rather than everything?", "good and bad abstention rates", "gold-feature bad-abstention gate", "abstention discipline"),
        ("confusable_pair_failure_atlas.png", "confusable_failure", "Which controls expose cheap keyword explanations?", "confusable count and decoy failure rate", "confusable and token-overlap decoy suites", "specificity failure"),
        ("random_feature_sanity_panel.png", "random_sanity", "Do random/polysemantic/ambiguous controls receive forced labels?", "abstain and forced-label rates", "high-risk feature types", "forced-label caveat"),
        ("review_queue_triage.png", "review_triage", "Which rows require human inspection first?", "review priority by row", "review reasons", "human review routing"),
        ("paired_examples.png", "paired_examples", "Which specimens support or contradict the aggregate?", "control AUC to method AUC paired rows", "same-feature shuffled control endpoint", "specimen evidence and counterexamples"),
    ]
    if ctx.args.no_plots:
        for fig, key, question, metric, control, claim in figure_specs:
            src, n = sources[key]
            add_plot_entry(entries, figure=fig, source=src, row_count=n, question=question, metric=metric, control=control, claim=claim, generated=False, warnings=["plots_skipped_by_flag", *warnings])
        write_plot_manifest(ctx, entries)
        return

    import matplotlib.pyplot as plt
    import numpy as np

    if not evidence:
        for fig, key, question, metric, control, claim in figure_specs:
            write_placeholder(ctx, fig, fig.replace("_", " ").replace(".png", ""), "No evidence rows were produced.")
            src, n = sources[key]
            add_plot_entry(entries, figure=fig, source=src, row_count=n, question=question, metric=metric, control=control, claim="no claim", generated=True, warnings=["no_evidence_rows", *warnings])
        write_plot_manifest(ctx, entries)
        return

    methods = [str(r["method"]) for r in evidence]
    x = np.arange(len(methods))
    aucs = [fnum(r.get("mean_heldout_auc"), 0.0) for r in evidence]
    controls = [fnum(r.get("control_mean_auc"), 0.0) for r in evidence]
    accs = [fnum(r.get("label_accuracy_when_scored"), 0.0) for r in evidence]
    good_abs = [fnum(r.get("good_abstain_rate_on_polysemantic_or_random"), 0.0) for r in evidence]
    bad_abs = [fnum(r.get("bad_abstain_rate_on_gold_features"), 0.0) for r in evidence]
    conf = [fnum(r.get("mean_confusable_failures"), 0.0) for r in evidence]
    decoy = [fnum(r.get("mean_decoy_failure_rate"), 0.0) for r in evidence]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.suptitle("Lab 31 auto-label audit dashboard", fontsize=14, fontweight="bold")
    axes[0, 0].bar(x - 0.18, aucs, 0.36, label="method mean AUC")
    axes[0, 0].bar(x + 0.18, controls, 0.36, label="shuffled control mean")
    axes[0, 0].axhline(AUC_SUPPORT_BAR, linestyle="--", linewidth=1, label="AUC gate")
    axes[0, 0].set_ylabel("held-out AUC")
    axes[0, 0].set_xticks(x, methods, rotation=25, ha="right")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_title("Prediction beats control?")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].bar(x, accs)
    axes[0, 1].axhline(LABEL_ACCURACY_BAR, linestyle="--", linewidth=1)
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_xticks(x, methods, rotation=25, ha="right")
    axes[0, 1].set_title("Gold-label accuracy when scored")
    axes[1, 0].bar(x - 0.18, good_abs, 0.36, label="good abstain on risky")
    axes[1, 0].bar(x + 0.18, bad_abs, 0.36, label="bad abstain on gold")
    axes[1, 0].axhline(GOOD_ABSTAIN_BAR, linestyle="--", linewidth=1)
    axes[1, 0].axhline(BAD_ABSTAIN_BAR, linestyle=":", linewidth=1)
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_xticks(x, methods, rotation=25, ha="right")
    axes[1, 0].set_title("Abstention discipline")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].bar(x - 0.18, conf, 0.36, label="confusable failures")
    axes[1, 1].bar(x + 0.18, decoy, 0.36, label="decoy failure rate")
    axes[1, 1].axhline(CONFUSABLE_FAILURE_BAR, linestyle="--", linewidth=1)
    axes[1, 1].set_xticks(x, methods, rotation=25, ha="right")
    axes[1, 1].set_title("Cheap-explanation alarms")
    axes[1, 1].legend(fontsize=8)
    fig.text(0.01, 0.01, ctx.plot_footer(), fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    bench.save_figure(ctx, fig, "auto_interp_dashboard.png", "Lab 31 auto-interpretability dashboard.")
    src, n = sources["dashboard"]
    add_plot_entry(entries, figure="auto_interp_dashboard.png", source=src, row_count=n, question=figure_specs[0][2], metric=figure_specs[0][3], control=figure_specs[0][4], claim=figure_specs[0][5], generated=True, warnings=warnings)

    auto_rows = [r for r in scores if r.get("method") in AUTOMATED_METHODS]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    xs = [fnum(r.get("same_feature_control_auc"), 0.5) for r in auto_rows]
    ys = [fnum(r.get("heldout_auc"), 0.0) for r in auto_rows]
    ax.scatter(xs, ys, alpha=0.78)
    for r, x0, y0 in zip(auto_rows, xs, ys):
        if fnum(r.get("review_priority"), 0.0) >= 3.0 or y0 - x0 < CONTROL_GAP_BAR:
            ax.annotate(str(r.get("method", "")).replace("_", "\n"), (x0, y0), fontsize=6, alpha=0.75)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="same-feature control")
    ax.plot([0, 1], [CONTROL_GAP_BAR, 1 + CONTROL_GAP_BAR], linestyle=":", linewidth=1, label="control gap gate")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("same-feature shuffled-control AUC")
    ax.set_ylabel("automated method AUC")
    ax.set_title("Target labels versus their own shuffled-context controls")
    ax.legend(fontsize=8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "target_vs_control.png", "Row-level target versus control AUC.")
    src, n = sources["target_vs_control"]
    add_plot_entry(entries, figure="target_vs_control.png", source=src, row_count=n, question=figure_specs[1][2], metric=figure_specs[1][3], control=figure_specs[1][4], claim=figure_specs[1][5], generated=True, warnings=warnings)

    features = sorted({str(r.get("feature_id")) for r in scores})
    mat = np.full((len(methods), len(features)), np.nan)
    for i, method in enumerate(methods):
        for j, feature in enumerate(features):
            vals = [fnum(r.get("heldout_auc")) for r in scores if r.get("method") == method and r.get("feature_id") == feature]
            mat[i, j] = safe_mean(vals, default=np.nan)
    fig, ax = plt.subplots(figsize=(max(9.0, len(features) * 0.66), 5.2))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(methods)), methods)
    ax.set_xticks(range(len(features)), [f.replace("feat_", "") for f in features], rotation=35, ha="right")
    ax.set_title("Explanation quality matrix: every feature-method cell")
    for i in range(len(methods)):
        for j in range(len(features)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.82, label="held-out AUC")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "explanation_quality_matrix.png", "Method-by-feature AUC heatmap.")
    src, n = sources["quality_matrix"]
    add_plot_entry(entries, figure="explanation_quality_matrix.png", source=src, row_count=n, question=figure_specs[2][2], metric=figure_specs[2][3], control=figure_specs[2][4], claim=figure_specs[2][5], generated=True, warnings=warnings)

    suite_order = ["heldout_positive", "paraphrase_positive", "hard_negative", "confusable_negative", "token_overlap_decoy"]
    data = []
    labels = []
    for suite in suite_order:
        vals = [fnum(r.get("predicted_score")) for r in tests if r.get("method") in AUTOMATED_METHODS and r.get("suite") == suite]
        vals = [v for v in vals if math.isfinite(v)]
        if vals:
            data.append(vals)
            labels.append(suite.replace("_", "\n"))
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    if data:
        ax.boxplot(data, labels=labels, showfliers=True)
        for i, vals in enumerate(data, start=1):
            jitter = np.linspace(-0.09, 0.09, num=len(vals)) if len(vals) > 1 else [0.0]
            ax.scatter([i + j for j in jitter], vals, alpha=0.35, s=16)
    ax.axhline(CONFUSABLE_FAILURE_THRESHOLD, linestyle="--", linewidth=1, label="failure threshold")
    ax.set_ylabel("predicted lexical activation score")
    ax.set_title("Raw score distributions by suite")
    ax.legend(fontsize=8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "label_score_distribution.png", "Raw label score distributions by suite.")
    src, n = sources["score_distribution"]
    add_plot_entry(entries, figure="label_score_distribution.png", source=src, row_count=n, question=figure_specs[3][2], metric=figure_specs[3][3], control=figure_specs[3][4], claim=figure_specs[3][5], generated=True, warnings=warnings)

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    xs = [fnum(r.get("confidence"), 0.0) for r in calibration if r.get("method") in AUTOMATED_METHODS]
    ys = [1.0 if truthy(r.get("success")) else 0.0 for r in calibration if r.get("method") in AUTOMATED_METHODS]
    if xs:
        ax.scatter(xs, ys, alpha=0.45, label="feature-method rows")
    bin_rows = [r for r in calibration_bins if r.get("method") in AUTOMATED_METHODS]
    bx = [fnum(r.get("mean_confidence")) for r in bin_rows]
    by = [fnum(r.get("success_rate")) for r in bin_rows]
    if bx:
        ax.plot(bx, by, marker="o", label="binned success")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="perfect calibration")
    ax.set_xlabel("confidence")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Confidence calibration")
    ax.legend(fontsize=8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "confidence_calibration_curve.png", "Confidence versus observed success.")
    src, n = sources["calibration"]
    add_plot_entry(entries, figure="confidence_calibration_curve.png", source=src, row_count=n, question=figure_specs[4][2], metric=figure_specs[4][3], control=figure_specs[4][4], claim=figure_specs[4][5], generated=True, warnings=warnings)

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.scatter(bad_abs, good_abs)
    for row in evidence:
        ax.annotate(str(row["method"]), (fnum(row.get("bad_abstain_rate_on_gold_features"), 0.0), fnum(row.get("good_abstain_rate_on_polysemantic_or_random"), 0.0)), fontsize=8)
    ax.axhline(GOOD_ABSTAIN_BAR, linestyle="--", linewidth=1)
    ax.axvline(BAD_ABSTAIN_BAR, linestyle="--", linewidth=1)
    ax.set_xlabel("bad abstention on labelable gold features")
    ax.set_ylabel("good abstention on risky features")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Abstention frontier")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "abstention_frontier.png", "Good/bad abstention frontier.")
    src, n = sources["abstention_frontier"]
    add_plot_entry(entries, figure="abstention_frontier.png", source=src, row_count=n, question=figure_specs[5][2], metric=figure_specs[5][3], control=figure_specs[5][4], claim=figure_specs[5][5], generated=True, warnings=warnings)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.bar(x - 0.18, conf, 0.36, label="confusable failures")
    ax.bar(x + 0.18, decoy, 0.36, label="decoy failure rate")
    ax.axhline(CONFUSABLE_FAILURE_BAR, linestyle="--", linewidth=1, label="confusable gate")
    ax.set_xticks(x, methods, rotation=25, ha="right")
    ax.set_title("Confusable pair and token-overlap failure atlas")
    ax.set_ylabel("mean failures / rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "confusable_pair_failure_atlas.png", "Confusable/decoy failure atlas.")
    src, n = sources["confusable_failure"]
    add_plot_entry(entries, figure="confusable_pair_failure_atlas.png", source=src, row_count=n, question=figure_specs[6][2], metric=figure_specs[6][3], control=figure_specs[6][4], claim=figure_specs[6][5], generated=True, warnings=warnings)

    random_rows = [r for r in scores if r.get("feature_type") in {"random_control", "polysemantic_gold", "ambiguous_control"}]
    abstain_by_method = []
    forced_by_method = []
    for method in methods:
        rows = [r for r in random_rows if r.get("method") == method]
        abstain_by_method.append(safe_mean([1.0 if truthy(r.get("abstain")) else 0.0 for r in rows], default=0.0))
        forced_by_method.append(safe_mean([0.0 if truthy(r.get("abstain")) else 1.0 for r in rows], default=0.0))
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.bar(x - 0.18, abstain_by_method, 0.36, label="abstain")
    ax.bar(x + 0.18, forced_by_method, 0.36, label="forced label")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, methods, rotation=25, ha="right")
    ax.set_title("Random, polysemantic, and ambiguous controls")
    ax.set_ylabel("rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "random_feature_sanity_panel.png", "Random/polysemantic abstention panel.")
    src, n = sources["random_sanity"]
    add_plot_entry(entries, figure="random_feature_sanity_panel.png", source=src, row_count=n, question=figure_specs[7][2], metric=figure_specs[7][3], control=figure_specs[7][4], claim=figure_specs[7][5], generated=True, warnings=warnings)

    top_review = sorted(review, key=lambda r: fnum(r.get("review_priority"), 0.0), reverse=True)[:12]
    fig, ax = plt.subplots(figsize=(9.2, max(4.8, 0.34 * len(top_review) + 1.5)))
    if top_review:
        y = np.arange(len(top_review))
        vals = [fnum(r.get("review_priority"), 0.0) for r in top_review]
        labels = [f"{r.get('feature_id','').replace('feat_', '')}\n{r.get('method','')}" for r in top_review]
        ax.barh(y, vals)
        ax.set_yticks(y, labels, fontsize=7)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No review rows were produced. Check warning_summary.csv before trusting the run.", ha="center", va="center")
    ax.set_xlabel("review priority")
    ax.set_title("Human-review queue triage")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "review_queue_triage.png", "Human-review queue triage.")
    src, n = sources["review_triage"]
    add_plot_entry(entries, figure="review_queue_triage.png", source=src, row_count=n, question=figure_specs[8][2], metric=figure_specs[8][3], control=figure_specs[8][4], claim=figure_specs[8][5], generated=True, warnings=warnings)

    paired_rows = []
    # Read from the in-memory construction by mirroring the rows used in figure_source_tables.
    automated = [r for r in scores if r.get("method") in AUTOMATED_METHODS]
    strongest = sorted(automated, key=lambda r: fnum(r.get("control_gap_vs_same_feature_shuffled"), -999.0), reverse=True)[:8]
    weakest = sorted(automated, key=lambda r: fnum(r.get("control_gap_vs_same_feature_shuffled"), 999.0))[:8]
    seen = set()
    for r in strongest + weakest:
        rid = str(r.get("score_id", r.get("feature_method_id", "")))
        if rid in seen:
            continue
        seen.add(rid)
        paired_rows.append(r)
    fig, ax = plt.subplots(figsize=(8.8, max(4.8, 0.32 * len(paired_rows) + 1.5)))
    if paired_rows:
        y = np.arange(len(paired_rows))
        for i, row in enumerate(paired_rows):
            x0 = fnum(row.get("same_feature_control_auc"), 0.0)
            x1 = fnum(row.get("heldout_auc"), 0.0)
            ax.plot([x0, x1], [i, i], marker="o", linewidth=1.5)
        labels = [f"{r.get('feature_id','').replace('feat_', '')}\n{r.get('method','')}" for r in paired_rows]
        ax.set_yticks(y, labels, fontsize=7)
        ax.invert_yaxis()
        ax.axvline(AUC_SUPPORT_BAR, linestyle="--", linewidth=1, label="AUC gate")
        ax.set_xlim(-0.03, 1.03)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No paired examples available.", ha="center", va="center")
    ax.set_xlabel("AUC: shuffled control endpoint to automated method endpoint")
    ax.set_title("Paired examples: support and counterexamples share the page")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "paired_examples.png", "Paired control-to-method specimens.")
    src, n = sources["paired_examples"]
    add_plot_entry(entries, figure="paired_examples.png", source=src, row_count=n, question=figure_specs[9][2], metric=figure_specs[9][3], control=figure_specs[9][4], claim=figure_specs[9][5], generated=True, warnings=warnings)

    write_plot_manifest(ctx, entries)

# ---------------------------------------------------------------------------
# Synthetic dataset generator used by the package and Tier A fallback
# ---------------------------------------------------------------------------


def contexts(*items: str) -> list[str]:
    return [str(x) for x in items]


def make_synthetic_task_payloads() -> list[dict[str, Any]]:
    """Return deterministic synthetic feature tasks for the frozen JSONL.

    These rows are benign and intentionally toy-like. They emulate top and
    held-out contexts for features whose gold labels are known, plus controls
    where an automated label should abstain.
    """
    rows: list[dict[str, Any]] = []

    def add(feature_id: str, feature_type: str, gold_label: str | None, top: list[str], held: list[str], neg: list[str], conf: list[str], adv: list[str], para: list[str] | None = None, *, secondary: str = "", abstain: bool = False, notes: str = "") -> None:
        rows.append({
            "feature_id": feature_id,
            "model": "synthetic-sparse-feature-suite-v1",
            "layer": 8 + len(rows) % 4,
            "feature_index": 1000 + len(rows),
            "feature_type": feature_type,
            "source_lab": "lab31_synthetic_suite",
            "top_contexts": top,
            "heldout_contexts": held,
            "paraphrase_contexts": para or [],
            "negative_contexts": neg,
            "confusable_contexts": conf,
            "adversarial_contexts": adv,
            "gold_label": gold_label,
            "gold_label_secondary": secondary,
            "expected_abstain": abstain,
            "risk_notes": notes,
        })

    add(
        "feat_code_json_schema", "synthetic_gold", "code",
        contexts(
            "The developer added a typed JSON schema and a unit test for the parser.",
            "A function returned an object, then the module raised an exception in the loop.",
            "The tensor library script failed because the bracket in the API response was missing.",
            "Compiler warnings pointed to the class import and the debug assertion.",
        ),
        contexts(
            "The software cache used a schema parser and an exception handler.",
            "A unit test covered the function that converts JSON into a typed object.",
        ),
        contexts(
            "The chef stirred herbs into broth while the oven heated.",
            "The striker scored in the final minute of the match.",
        ),
        contexts(
            "The court filed a motion about a software licensing contract.",
            "The hospital database stored patient records in a regulated system.",
        ),
        contexts(
            "The word function appears in this sentence, but it describes a social function at dinner.",
            "Object, class, and script are listed as grammar terms, not code behavior.",
        ),
        para=contexts(
            "The parser module checks a JSON object and throws an exception when the schema fails.",
            "The developer wrote a test for a tensor cache bug in the API wrapper.",
        ),
    )
    add(
        "feat_cooking_baking_heat", "synthetic_gold", "cooking",
        contexts(
            "The chef folded butter into flour before the dough went into the oven.",
            "A recipe said to simmer onions, herbs, and broth in the kitchen pan.",
            "The batter needed heat, salt, and a careful stir before baking.",
            "Yeast lifted the loaf while the sauce warmed beside the roast.",
        ),
        contexts(
            "The cook stirred spice into the sauce and let the broth simmer.",
            "A kitchen recipe mixed flour, butter, and dough for a loaf.",
        ),
        contexts(
            "The analyst reviewed credit spreads and bond yields.",
            "The judge heard evidence from a witness at trial.",
        ),
        contexts(
            "The medicine had a dosage measured in teaspoons before treatment.",
            "The market absorbed heat from a hot earnings report, a metaphor not a recipe.",
        ),
        contexts(
            "This paragraph repeats oven, butter, and pan as inventory words without describing cooking.",
            "Flour and dough are listed in a spelling exercise, not a kitchen scene.",
        ),
        para=contexts(
            "Herbs and onions cooked in the pan while the sauce thickened.",
            "The chef baked a loaf after mixing butter into the flour.",
        ),
    )
    add(
        "feat_finance_credit_market", "synthetic_gold", "finance",
        contexts(
            "The analyst warned that credit spreads widened as bond liquidity fell.",
            "Investors watched company earnings, revenue guidance, and the stock margin.",
            "A hedge fund shifted its portfolio from equity to debt after rates rose.",
            "The market priced default risk into the loan yield and cash position.",
        ),
        contexts(
            "The portfolio manager compared bond yield, equity risk, and liquidity.",
            "Revenue guidance moved the stock after investors reviewed earnings.",
        ),
        contexts(
            "The goalkeeper blocked a shot near the field after halftime.",
            "The clinician checked the patient's pulse and fever.",
        ),
        contexts(
            "The sports team used market language when trading players before the season.",
            "The legal fund paid damages under a contract clause.",
        ),
        contexts(
            "The word market appears in a farmer's market recipe with herbs and onions.",
            "Bond is a character name here, while stock means soup broth.",
        ),
        para=contexts(
            "Debt investors worried about default risk and cash liquidity.",
            "The company's earnings report moved the equity market.",
        ),
    )
    add(
        "feat_sports_match_goal", "synthetic_gold", "sports",
        contexts(
            "The striker scored after a midfield pass in the final minute.",
            "The coach changed the defense before the league match resumed.",
            "A runner won the race after a late sprint on the field track.",
            "The goalkeeper saved the shot and the team kept its lead.",
        ),
        contexts(
            "The team scored a goal after a pass across the pitch.",
            "The athlete prepared for the tournament and the coach praised the defense.",
        ),
        contexts(
            "The attorney filed a brief before the hearing.",
            "The recipe used butter, flour, and herbs.",
        ),
        contexts(
            "The stock market team met before a finance league conference.",
            "The court fielded questions about a legal defense team.",
        ),
        contexts(
            "Game, field, and score are vocabulary words in a classroom list.",
            "The team lead in a software module reviewed a pass/fail unit test.",
        ),
        para=contexts(
            "A goalkeeper and striker shaped the match during league play.",
            "The runner's sprint changed the race result.",
        ),
    )
    add(
        "feat_law_trial_contract", "synthetic_gold", "law",
        contexts(
            "The court heard a motion about liability under the contract clause.",
            "An attorney cited precedent before the judge during the trial.",
            "The jury weighed evidence from the witness and testimony.",
            "A legal brief argued that the statute controlled the appeal.",
        ),
        contexts(
            "The plaintiff filed a claim after the contract dispute reached court.",
            "The judge considered evidence and precedent at the hearing.",
        ),
        contexts(
            "The forecast predicted rain and a cold coastal wind.",
            "The developer fixed a JSON parser exception.",
        ),
        contexts(
            "A clinical trial tested treatment dosage in a hospital.",
            "A finance contract created debt and credit risk.",
        ),
        contexts(
            "Court and appeal are used in tennis reporting, not legal reasoning.",
            "Evidence is mentioned as a generic word in a science class rubric.",
        ),
        para=contexts(
            "An attorney questioned a witness after the judge admitted testimony.",
            "The statute and precedent shaped the liability dispute.",
        ),
    )
    add(
        "feat_medicine_patient_treatment", "synthetic_gold", "medicine",
        contexts(
            "The clinician checked symptoms, dosage, fever, and breathing before treatment.",
            "A hospital nurse measured the patient's pulse during recovery.",
            "The physician ordered a scan after the injury and discussed therapy.",
            "The vaccine trial tracked infection risk and clinical pain reports.",
        ),
        contexts(
            "The patient received a diagnosis after the nurse checked blood pressure.",
            "The physician adjusted the prescription and monitored recovery.",
        ),
        contexts(
            "The investor compared bond liquidity and stock margin.",
            "The chef baked dough in the oven.",
        ),
        contexts(
            "A clinical trial in law concerns courtroom evidence, not a hospital treatment.",
            "The weather report mentioned pressure and a front, not blood pressure.",
        ),
        contexts(
            "Patient and treatment are listed as vocabulary words in a spelling test.",
            "Fever appears as the title of a song in this sentence.",
        ),
        para=contexts(
            "Hospital staff tracked symptoms, dosage, and recovery.",
            "The clinician prescribed therapy after the scan.",
        ),
    )
    add(
        "feat_weather_storm_forecast", "synthetic_gold", "weather",
        contexts(
            "The forecast warned of rain, wind, thunder, and cold air near the coast.",
            "Dark clouds brought snow, hail, and a humid storm front.",
            "Morning fog lifted as the temperature and pressure changed.",
            "Radar showed a downpour moving through the bright coastal breeze.",
        ),
        contexts(
            "The storm brought rain and thunder before the air turned cold.",
            "A sunny forecast changed when fog and wind reached the coast.",
        ),
        contexts(
            "The attorney argued a statute before the judge.",
            "The athlete scored after a pass from midfield.",
        ),
        contexts(
            "The word pressure described a patient chart rather than a weather front.",
            "The market climate was stormy, but no rain or forecast was involved.",
        ),
        contexts(
            "Weathered is an adjective on a wood catalog, not a forecast.",
            "Rain and thunder are listed as poem words without describing weather conditions.",
        ),
        para=contexts(
            "The coastal forecast predicted fog, wind, and lower temperature.",
            "Hail and clouds marked the storm on the radar map.",
        ),
    )
    add(
        "feat_emotion_grief_relief", "synthetic_gold", "emotion",
        contexts(
            "After the apology, relief softened her anger and she smiled at her friends.",
            "The note carried sadness, fear, and a generous promise of support.",
            "He felt proud and happy, but still anxious about the lonely mood.",
            "Tears of grief turned into comfort and hope by morning.",
        ),
        contexts(
            "The friend's support brought comfort after anger and fear.",
            "She felt joy, sadness, and relief during the apology.",
        ),
        contexts(
            "The recipe simmered herbs in a pan.",
            "The market priced credit risk into a bond yield.",
        ),
        contexts(
            "A legal apology reduced liability but did not describe a feeling.",
            "The weather was calm and bright, but no person felt emotion.",
        ),
        contexts(
            "Happy, sad, and fear are printed as words in a lexicon table without an emotional scene.",
            "The mood field in a software object stores a string called joy.",
        ),
        para=contexts(
            "An anxious mood changed when friends offered support.",
            "Grief and relief appeared together in the letter.",
        ),
    )
    add(
        "feat_poly_finance_sports", "polysemantic_gold", None,
        contexts(
            "The team traded a striker after the market valued the club's debt.",
            "Investors bought stock in the league while the coach discussed revenue.",
            "The player contract changed the portfolio of a sports fund.",
            "A tournament sponsor moved cash into equity before the match.",
        ),
        contexts(
            "The league fund reported revenue after the team won the match.",
            "A sports investor discussed debt, players, and cash flow.",
        ),
        contexts(
            "The chef baked a loaf in the kitchen oven.",
            "The patient recovered after treatment in the hospital.",
        ),
        contexts(
            "The finance market text and sports match text each explain only half the pattern.",
            "A pure sports article mentions team and league without finance terms.",
        ),
        contexts(
            "Market, team, and fund are listed as example nouns without a coherent topic.",
            "Stock and match appear in separate dictionary definitions.",
        ),
        para=contexts(
            "A club's investor and coach discussed revenue after the game.",
            "The sports fund traded equity tied to the league.",
        ),
        secondary="finance+sports",
        abstain=True,
        notes="Designed polysemantic mixture; best behavior is abstention or multi-label review.",
    )
    add(
        "feat_random_control", "random_control", None,
        contexts(
            "Purple notebook seven quietly because a hinge wandered across noon.",
            "The bright clause stirred a runner's invoice beside fog and butter.",
            "A tensor smiled when the chef judged weather credit on Tuesday.",
            "Hospital midfield JSON rain apology bond loaf statute.",
        ),
        contexts(
            "A random mix of court, sauce, forecast, and compiler words appears.",
            "The sentence lists patient, striker, loan, grief, and oven with no concept.",
        ),
        contexts(
            "The developer fixed a parser bug in the module.",
            "The court heard testimony from a witness.",
        ),
        contexts(
            "Every domain contributes one keyword, so no single label should win.",
            "The context is intentionally incoherent across topic families.",
        ),
        contexts(
            "Function market rain judge patient happy oven team all appear as bare words.",
            "A list of domain keywords is not an activating concept.",
        ),
        para=contexts(
            "Code, cooking, law, weather, sports, emotion, finance, and medicine words collide randomly.",
            "No stable feature should be inferred from this mixed list.",
        ),
        abstain=True,
        notes="Random-control feature with domain words but no coherent concept.",
    )
    add(
        "feat_ambiguous_law_medicine_trial", "ambiguous_control", None,
        contexts(
            "The clinical trial produced evidence that the physician reviewed before testimony.",
            "A hospital attorney discussed liability after a patient entered the trial.",
            "The witness described a treatment dispute in court.",
            "Legal evidence and clinical dosage appeared in the same report.",
        ),
        contexts(
            "The trial mixed courtroom testimony with a physician's clinical notes.",
            "A patient dispute included legal evidence and medical treatment.",
        ),
        contexts(
            "The striker scored in the league match.",
            "The chef simmered broth with herbs.",
        ),
        contexts(
            "Pure law rows and pure medicine rows are both plausible confusables.",
            "The label should be reviewed as a boundary between medicine and law.",
        ),
        contexts(
            "Trial, evidence, and clinical are listed together as words with no scenario.",
            "Patient and testimony appear in a vocabulary drill.",
        ),
        para=contexts(
            "A legal-medical dispute connected treatment, patient records, and testimony.",
            "The clinical evidence entered a courtroom dispute.",
        ),
        secondary="law+medicine",
        abstain=True,
        notes="Ambiguous but coherent boundary feature; automated single-label methods should route to review.",
    )
    add(
        "feat_code_runtime_errors", "synthetic_gold", "code",
        contexts(
            "The runtime cache raised an exception after the import loop changed a variable.",
            "A developer debugged an API class with a failing assertion and unit test.",
            "The compiler reported a bracket error inside a parser module.",
            "The software library returned a JSON object with the wrong schema.",
        ),
        contexts(
            "A debug assertion caught the variable error at runtime.",
            "The parser library fixed the JSON schema for the API.",
        ),
        contexts(
            "The storm front brought rain to the coast.",
            "The judge reviewed a legal brief.",
        ),
        contexts(
            "A legal code section is a statute rather than software.",
            "The word class appears in a school schedule, not a program.",
        ),
        contexts(
            "Code and debug are printed in a word list without a program.",
            "Runtime is mentioned as a theater duration, not software execution.",
        ),
        para=contexts(
            "The module failed a unit test after a parser exception.",
            "The API returned a typed object that violated the schema.",
        ),
    )
    add(
        "feat_weather_pressure_front", "synthetic_gold", "weather",
        contexts(
            "A low-pressure front pushed humid air, clouds, and rain across the coast.",
            "The climate report expected snow, fog, and a cold breeze by morning.",
            "Radar tracked thunder and hail inside the storm.",
            "The forecast changed as temperature fell before the downpour.",
        ),
        contexts(
            "The front brought humid air, rain, and thunder.",
            "A cold fog followed the coastal storm forecast.",
        ),
        contexts(
            "The attorney cited statute and precedent.",
            "The nurse measured a patient's pulse.",
        ),
        contexts(
            "Pressure can be blood pressure in medicine or market pressure in finance.",
            "A front can be a political front, not a weather front.",
        ),
        contexts(
            "Storm, front, pressure, and climate are listed as metaphors for a business memo.",
            "Cloud and cold are labels in a computing example, not weather.",
        ),
        para=contexts(
            "The radar forecast showed hail and rain along the coast.",
            "Temperature and pressure shifted before the storm.",
        ),
    )
    return rows


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    tasks, data_info, schema_rows = load_tasks(ctx)
    manifest_path = ctx.path("diagnostics", "data_manifest.json")
    bench.write_json(manifest_path, data_info)
    ctx.register_artifact(manifest_path, "diagnostic", "Lab 31 data manifest and offline audit scope.")
    schema_path = ctx.path("diagnostics", "schema_audit.csv")
    bench.write_csv_with_context(ctx, schema_path, schema_rows)
    ctx.register_artifact(schema_path, "diagnostic", "Schema and context-count audit for Lab 31 feature tasks.")

    # Lab 31 is offline, but it still rides through the shared bench so students
    # prove the same model/hook instrument before opening interpretation tables.
    hook_check = bench.run_hook_parity_check(ctx, bundle, tasks[0].top_contexts[0])
    first = bench.run_with_residual_cache(bundle, tasks[0].top_contexts[0])
    lens_check = bench.run_lens_self_check(ctx, bundle, first)
    patch_noop = bench.run_patch_noop_check(ctx, bundle, tasks[0].top_contexts[0])

    explanations, deletion_rows = generate_explanations(tasks)
    tests, raw_scores = build_tests(tasks, explanations)
    scores = augment_score_rows(raw_scores)
    evidence, review, calibration, feature_rows, counterexamples, metrics = summarize_evidence(scores)
    suite_summary = suite_score_summary_rows(tests)
    calibration_bins = calibration_bin_rows(calibration)
    abstention_summary = abstention_summary_rows(scores)
    failure_specimens = failure_specimen_rows(scores, tests, counterexamples)
    metrics = {
        **metrics,
        "data": data_info,
        "n_explanations": len(explanations),
        "n_tests": len(tests),
        "n_score_rows": len(scores),
        "n_suite_summary_rows": len(suite_summary),
        "n_calibration_bins": len(calibration_bins),
        "n_failure_specimens": len(failure_specimens),
    }

    write_tables(ctx, explanations, deletion_rows, tests, scores, evidence, feature_rows, review, calibration, counterexamples)
    write_extra_tables(ctx, suite_summary, calibration_bins, abstention_summary, failure_specimens)
    write_failure_specimens(ctx, failure_specimens)
    write_state(ctx, data_info)
    write_run_config_snapshot(ctx, data_info)
    write_status_files(ctx, data_info, hook_check, lens_check, patch_noop, metrics, schema_rows)
    warning_rows = write_warning_summary(ctx, data_info, hook_check, lens_check, patch_noop, schema_rows, evidence, scores, review, counterexamples)
    metrics = {**metrics, "n_warning_rows": len(warning_rows), "warning_codes": warning_codes(warning_rows)}

    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 31 metrics and audit thresholds.")
    write_method_card(ctx, data_info, metrics, evidence)
    write_operationalization_audit(ctx, data_info, evidence, counterexamples)
    write_run_summary(ctx, data_info, metrics, evidence, counterexamples)
    write_review_artifacts(ctx, explanations, scores, review)
    write_claims(ctx, data_info, evidence)
    write_plots(ctx, evidence, scores, tests, review, calibration, calibration_bins, suite_summary, abstention_summary, failure_specimens, warning_rows)
    print(f"[lab31] wrote {len(explanations)} explanations, {len(tests)} tests, {len(evidence)} method verdicts, {len(review)} review rows, and {len(failure_specimens)} failure specimens")
