"""Lab 36: Severance report-channel verification.

This lab turns the Severance guide into a bench-registered experiment. It is
stricter than Lab 25: concept injection into a report is treated as a screen,
while matched-output source attribution and content-blind insertion detection
are the headline tests.

The ceiling is functional report-channel coupling. The lab does not establish
experience, phenomenal self-knowledge, or absence of experience.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import re
import statistics
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import interp_bench as bench

LAB_ID = "L36"
DATA_DIR = bench.COURSE_ROOT / "severance" / "data"
SYSTEM_PROMPT = (
    "You are a careful assistant. Keep answers concise. Do not invent hidden "
    "causes, private feelings, or mechanistic explanations you were not given."
)
SOURCE_SYSTEM_PROMPT = (
    "You are a careful assistant. When asked about the source of a response, "
    "choose the literal source from the listed labels."
)

PROMPT_SET_CAPS = {"small": 4, "medium": 8, "full": 0}
B2_DOSES = (0.0, 0.5, 1.0, 2.0)
HEADLINE_DOSE = 2.0
REPORT_TOKENS = 16
BEHAVIOR_TOKENS = 48
DETECTION_TOKENS = 4
CONFIDENCE_TOKENS = 8
RANDOM_CONTROLS = 8
SHUFFLED_CONTROLS = 8
SOURCE_CONDITIONS = ("default_mode", "matched_default", "system_prompt", "user_instruction", "activation_injection")
BLOCKED_SAFETY_PATTERNS = ("jailbreak", "bypass safety", "malware", "explosive", "poison", "weapon", "self-harm")


@dataclasses.dataclass(frozen=True)
class SeveranceItem:
    item_id: str
    concept_family: str
    split: str
    target_concept: str
    wrong_concept: str
    positive_prompt: str
    negative_prompt: str
    report_prompt: str
    behavior_prompt: str
    target_markers: tuple[str, ...]
    wrong_markers: tuple[str, ...]
    semantic_target_gloss: str
    semantic_wrong_gloss: str
    source_note: str = ""


@dataclasses.dataclass(frozen=True)
class SourceItem:
    item_id: str
    split: str
    style_family: str
    target_style: str
    canonical_answer: str
    task_prompt: str
    system_style_instruction: str
    user_style_instruction: str
    activation_direction_id: str
    answer_options: str
    correct_source_by_condition: dict[str, str]


@dataclasses.dataclass(frozen=True)
class DetectionItem:
    item_id: str
    split: str
    distractor_task: str
    report_prompt: str
    behavior_prompt: str
    injection_position_policy: str
    positive_conditions: tuple[str, ...]
    negative_conditions: tuple[str, ...]
    target_direction_id: str
    wrong_direction_id: str


@dataclasses.dataclass(frozen=True)
class UncertaintyItem:
    item_id: str
    split: str
    question: str
    answer: str
    known_status: str
    expected_confidence: str
    target_markers: tuple[str, ...]
    wrong_markers: tuple[str, ...]
    difficulty_bucket: str


@dataclasses.dataclass
class DirectionBundle:
    concept: str
    family: str
    stream_depth: int
    injection_layer: int
    b4_injection_layer: int
    vector: Any
    random_vector: Any
    shuffled_vector: Any
    residual_rms: float
    train_gap: float
    validation_gap: float
    heldout_gap: float
    train_auc: float
    validation_auc: float
    heldout_auc: float
    markers: tuple[str, ...]
    wrong_markers: tuple[str, ...]
    target_gloss: str
    wrong_gloss: str


@dataclasses.dataclass
class LabelResolver:
    name: str
    ids_by_label: dict[str, list[int]]
    variants_by_label: dict[str, list[str]]


class nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def rounded(value: Any, digits: int = 4) -> Any:
    out = fnum(value)
    return round(out, digits) if math.isfinite(out) else ""


def safe_mean(values: Sequence[Any], default: float = float("nan")) -> float:
    vals = [fnum(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(statistics.fmean(vals)) if vals else default


def short_hash(text: Any, n: int = 12) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:n]


def sha256_ids(ids: Sequence[int]) -> str:
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode("utf-8")).hexdigest()


def stable_seed(text: str, base: int = 0) -> int:
    h = hashlib.sha256((str(base) + "|" + str(text)).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def normalize_label(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def human_label(text: Any) -> str:
    return str(text).replace("_", " ").strip()


def split_markers(text: Any) -> tuple[str, ...]:
    parts = re.split(r"[;|,]", str(text or ""))
    return tuple(dict.fromkeys(p.strip().lower() for p in parts if p.strip()))


def flatten_markers(markers: Sequence[str]) -> str:
    return ";".join(str(m) for m in markers if str(m))


def marker_hit(text: Any, markers: Sequence[str]) -> bool:
    low = str(text or "").lower()
    for marker in markers:
        m = str(marker or "").strip().lower()
        if not m:
            continue
        if re.search(r"\b" + re.escape(m) + r"\b", low):
            return True
        if " " in m and m in low:
            return True
    return False


def concept_markers(concept: str, markers: Sequence[str]) -> tuple[str, ...]:
    base = [concept, human_label(concept), normalize_label(concept).replace("_", " ")]
    return tuple(dict.fromkeys([m.lower() for m in list(markers) + base if str(m).strip()]))


def unit(vec: Any) -> Any:
    norm = vec.detach().float().norm()
    return vec.detach().float() if float(norm) <= 1e-9 else vec.detach().float() / norm


def random_unit(dim: int, seed: int) -> Any:
    import torch

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) % (2**31 - 1))
    return unit(torch.randn(dim, generator=gen))


def cosine(a: Any, b: Any) -> float:
    import torch

    aa = a.detach().float().flatten()
    bb = b.detach().float().flatten()
    denom = float(aa.norm() * bb.norm())
    return float("nan") if denom <= 1e-9 else float(torch.dot(aa, bb) / denom)


def auc_from_scores(pos: Sequence[float], neg: Sequence[float]) -> float:
    pairs = [(p, n) for p in pos for n in neg]
    if not pairs:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p, n in pairs)
    return wins / len(pairs)


def normal_ppf(p: float) -> float:
    p = min(1.0 - 1e-12, max(1e-12, float(p)))
    try:
        from scipy.stats import norm

        return float(norm.ppf(p))
    except Exception:
        return statistics.NormalDist().inv_cdf(p)


def clipped_rate(k: int, n: int) -> float:
    return (int(k) + 0.5) / (int(n) + 1.0)


def d_prime(hits: int, n_signal: int, false_alarms: int, n_noise: int) -> float:
    return normal_ppf(clipped_rate(hits, n_signal)) - normal_ppf(clipped_rate(false_alarms, n_noise))


def parse_mode(args: Any) -> set[str]:
    raw = str(getattr(args, "mode", "") or os.environ.get("LAB36_MODE", "all") or "all").strip().lower()
    aliases = {
        "all": {"instrument", "cartography", "directions", "b2", "b3", "b4", "b5", "patch"},
        # The shared bench CLI defaults --mode to "lora" for older labs.
        "lora": {"instrument", "cartography", "directions", "b2", "b3", "b4", "b5", "patch"},
        "smoke": {"instrument", "cartography", "directions", "b2", "b4", "b5"},
        "both": {"instrument", "directions", "b2", "b4", "b5"},
        "instrument_proof": {"instrument"},
        "build_directions": {"instrument", "directions"},
        "inject_screen": {"instrument", "directions", "b2"},
        "source_attribution": {"instrument", "directions", "b4"},
        "injection_detection": {"instrument", "directions", "b5"},
        "certainty": {"instrument", "b3"},
        "patch_recovery": {"instrument", "directions", "patch"},
    }
    if raw in aliases:
        return set(aliases[raw])
    out: set[str] = set()
    for part in re.split(r"[,; ]+", raw):
        if part:
            out |= aliases.get(part, {part})
    return out or set(aliases["all"])


def read_csv_rows(path: pathlib.Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))


def row_to_item(row: Mapping[str, Any], idx: int) -> SeveranceItem:
    return SeveranceItem(
        item_id=str(row.get("item_id") or f"item_{idx:03d}"),
        concept_family=normalize_label(row.get("concept_family") or "general"),
        split=normalize_label(row.get("split") or "train"),
        target_concept=normalize_label(row.get("target_concept") or ""),
        wrong_concept=normalize_label(row.get("wrong_concept") or ""),
        positive_prompt=str(row.get("positive_prompt") or "").strip(),
        negative_prompt=str(row.get("negative_prompt") or "").strip(),
        report_prompt=str(row.get("report_prompt") or "").strip(),
        behavior_prompt=str(row.get("behavior_prompt") or "").strip(),
        target_markers=split_markers(row.get("target_markers")),
        wrong_markers=split_markers(row.get("wrong_markers")),
        semantic_target_gloss=str(row.get("semantic_target_gloss") or "").strip(),
        semantic_wrong_gloss=str(row.get("semantic_wrong_gloss") or "").strip(),
        source_note=str(row.get("source_note") or "").strip(),
    )


def parse_source_map(text: str) -> dict[str, str]:
    out = {}
    for part in str(text or "").split(";"):
        if ":" in part:
            key, val = part.split(":", 1)
            out[normalize_label(key)] = val.strip().upper()[:1]
    return out


def row_to_source(row: Mapping[str, Any], idx: int) -> SourceItem:
    return SourceItem(
        item_id=str(row.get("item_id") or f"src_{idx:03d}"),
        split=normalize_label(row.get("split") or "heldout"),
        style_family=normalize_label(row.get("style_family") or "register"),
        target_style=normalize_label(row.get("target_style") or ""),
        canonical_answer=str(row.get("canonical_answer") or "").strip(),
        task_prompt=str(row.get("task_prompt") or "").strip(),
        system_style_instruction=str(row.get("system_style_instruction") or "").strip(),
        user_style_instruction=str(row.get("user_style_instruction") or "").strip(),
        activation_direction_id=normalize_label(row.get("activation_direction_id") or row.get("target_style") or ""),
        answer_options=str(row.get("answer_options") or "A=default;B=system;C=user;D=hidden;E=unclear"),
        correct_source_by_condition=parse_source_map(str(row.get("correct_source_by_condition") or "")),
    )


def row_to_detection(row: Mapping[str, Any], idx: int) -> DetectionItem:
    return DetectionItem(
        item_id=str(row.get("item_id") or f"inj_{idx:03d}"),
        split=normalize_label(row.get("split") or "heldout"),
        distractor_task=str(row.get("distractor_task") or "").strip(),
        report_prompt=str(row.get("report_prompt") or "").strip(),
        behavior_prompt=str(row.get("behavior_prompt") or "").strip(),
        injection_position_policy=normalize_label(row.get("injection_position_policy") or "report_query"),
        positive_conditions=split_markers(row.get("positive_conditions")),
        negative_conditions=split_markers(row.get("negative_conditions")),
        target_direction_id=normalize_label(row.get("target_direction_id") or ""),
        wrong_direction_id=normalize_label(row.get("wrong_direction_id") or ""),
    )


def row_to_uncertainty(row: Mapping[str, Any], idx: int) -> UncertaintyItem:
    return UncertaintyItem(
        item_id=str(row.get("item_id") or f"q_{idx:03d}"),
        split=normalize_label(row.get("split") or "heldout"),
        question=str(row.get("question") or "").strip(),
        answer=str(row.get("answer") or "").strip(),
        known_status=normalize_label(row.get("known_status") or ""),
        expected_confidence=normalize_label(row.get("expected_confidence") or ""),
        target_markers=split_markers(row.get("target_markers")),
        wrong_markers=split_markers(row.get("wrong_markers")),
        difficulty_bucket=normalize_label(row.get("difficulty_bucket") or ""),
    )


def round_robin(items: Sequence[Any], key_fn: Any, cap: int) -> list[Any]:
    if cap <= 0 or len(items) <= cap:
        return list(items)
    buckets: dict[str, list[Any]] = defaultdict(list)
    for item in items:
        buckets[str(key_fn(item))].append(item)
    keys = sorted(buckets)
    selected = []
    cursor = 0
    while len(selected) < cap and any(buckets.values()):
        key = keys[cursor % len(keys)]
        if buckets[key]:
            selected.append(buckets[key].pop(0))
        cursor += 1
    return selected


def selected_cap(args: Any) -> int:
    prompt_set = str(getattr(args, "prompt_set", "small") or "small")
    cap = PROMPT_SET_CAPS.get(prompt_set, 0)
    max_examples = int(getattr(args, "max_examples", 0) or 0)
    return min(cap, max_examples) if cap > 0 and max_examples > 0 else (max_examples if max_examples > 0 else cap)


def load_all_data(args: Any) -> tuple[list[SeveranceItem], list[SourceItem], list[DetectionItem], list[UncertaintyItem], list[dict[str, Any]]]:
    paths = {
        "introspection": DATA_DIR / "introspection_queries.csv",
        "source": DATA_DIR / "source_attribution_prompts.csv",
        "detection": DATA_DIR / "injection_detection_prompts.csv",
        "uncertainty": DATA_DIR / "uncertainty_questions.csv",
        "patchscope": DATA_DIR / "patchscope_prompts.csv",
        "rubric": DATA_DIR / "semantic_judge_rubric.md",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Lab 36 data file missing: {path}")
    all_items = [row_to_item(r, i) for i, r in enumerate(read_csv_rows(paths["introspection"]))]
    all_sources = [row_to_source(r, i) for i, r in enumerate(read_csv_rows(paths["source"]))]
    all_detection = [row_to_detection(r, i) for i, r in enumerate(read_csv_rows(paths["detection"]))]
    all_uncertainty = [row_to_uncertainty(r, i) for i, r in enumerate(read_csv_rows(paths["uncertainty"]))]
    cap = selected_cap(args)
    items = round_robin(sorted(all_items, key=lambda x: (x.concept_family, x.target_concept, x.split, x.item_id)), lambda x: x.concept_family, cap)
    concepts = {i.target_concept for i in items}
    sources = [s for s in all_sources if not concepts or s.activation_direction_id in concepts or s.target_style in concepts]
    detections = [d for d in all_detection if not concepts or d.target_direction_id in concepts or d.wrong_direction_id in concepts]
    if cap > 0:
        sources = sources[: max(2, min(len(sources), cap))]
        detections = detections[: max(2, min(len(detections), cap))]
        all_uncertainty = all_uncertainty[: max(2, min(len(all_uncertainty), cap))]
    manifest = [
        {
            "file": str(path.relative_to(bench.COURSE_ROOT)),
            "exists": path.exists(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for path in paths.values()
    ]
    return items, sources, detections, all_uncertainty, manifest


def split_is_train(split: str) -> bool:
    return normalize_label(split) == "train"


def split_is_validation(split: str) -> bool:
    return normalize_label(split) in {"validation", "val"}


def split_is_eval(split: str) -> bool:
    return normalize_label(split) in {"heldout", "holdout", "eval", "test"}


def render_user(bundle: bench.ModelBundle, user: str, *, system: str = SYSTEM_PROMPT) -> tuple[str, str]:
    if bench.supports_chat_template(bundle):
        return bench.apply_chat_template(bundle, user, system=system, add_generation_prompt=True), "chat_template"
    return f"System: {system}\nUser: {user}\nAssistant:", "raw_fallback_no_chat_template"


def make_report_user(item: SeveranceItem) -> str:
    return f"{item.report_prompt}\n\nReply with one short phrase only. If nothing stands out, say none."


def make_behavior_user(item: SeveranceItem) -> str:
    return f"{item.behavior_prompt}\n\nAnswer normally. Do not discuss hidden mechanisms."


def make_confidence_report_user(item: UncertaintyItem) -> str:
    return "Before answering, report your current confidence as a number from 0 to 1. Reply with only the number.\n\nQuestion: " + item.question


def make_confidence_behavior_user(item: UncertaintyItem) -> str:
    return "Answer the question in one short sentence.\n\nQuestion: " + item.question


def resolve_label_ids(bundle: bench.ModelBundle, labels: Mapping[str, Sequence[str]], name: str) -> tuple[LabelResolver, list[dict[str, Any]]]:
    tok = bundle.tokenizer
    ids_by_label: dict[str, list[int]] = {}
    variants_by_label: dict[str, list[str]] = {}
    rows = []
    for label, variants in labels.items():
        ids = []
        kept = []
        for variant in variants:
            enc = tok(str(variant), add_special_tokens=False)["input_ids"]
            rows.append({"label_set": name, "label": label, "variant": variant, "token_ids": " ".join(str(i) for i in enc), "n_tokens": len(enc), "single_token": int(len(enc) == 1), "decoded": tok.decode(enc) if enc else ""})
            if len(enc) == 1 and int(enc[0]) not in ids:
                ids.append(int(enc[0]))
                kept.append(str(variant))
        if not ids:
            for variant in variants:
                enc = tok(str(variant), add_special_tokens=False)["input_ids"]
                if enc:
                    ids.append(int(enc[0]))
                    kept.append(str(variant) + " [first-token]")
                    break
        ids_by_label[label] = ids
        variants_by_label[label] = kept
    return LabelResolver(name=name, ids_by_label=ids_by_label, variants_by_label=variants_by_label), rows


def label_scores_from_logits(logits: Any, resolver: LabelResolver) -> dict[str, float]:
    return {label: max([float(logits[int(i)]) for i in ids], default=float("-inf")) for label, ids in resolver.ids_by_label.items()}


def choose_label(logits: Any, resolver: LabelResolver) -> tuple[str, dict[str, float]]:
    scores = label_scores_from_logits(logits, resolver)
    return (max(scores, key=lambda k: scores[k]), scores) if scores else ("unknown", {})


def write_bench_note(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    payload = {
        "lab": LAB_ID,
        "actual_tokenizer_has_chat_template": bool(bench.supports_chat_template(bundle)),
        "lab_listed_in_CHAT_TEMPLATE_LABS": str(ctx.args.lab) in set(getattr(bench, "CHAT_TEMPLATE_LABS", set())),
        "claim_ceiling": "functional report-channel coupling only",
    }
    path = ctx.path("diagnostics", "bench_integration_note.json")
    bench.write_json(path, payload)
    ctx.register_artifact(path, "diagnostic", "Bench integration note for Lab 36.")


def run_exact_hook_parity(ctx: bench.RunContext, bundle: bench.ModelBundle, rendered_prompt: str) -> dict[str, Any]:
    block_outputs: dict[int, Any] = {}

    def make_hook(idx: int):
        def hook(_module: Any, _hook_args: tuple, output: Any) -> None:
            out = output[0] if isinstance(output, tuple) else output
            block_outputs[idx] = bench.tensor_cpu_float(out)
        return hook

    handles = [block.register_forward_hook(make_hook(i)) for i, block in enumerate(bundle.blocks)]
    try:
        capture = bench.run_with_residual_cache(bundle, rendered_prompt, add_special_tokens=False)
    finally:
        for handle in handles:
            handle.remove()
    rows = []
    max_diff = 0.0
    missing = []
    for layer in range(bundle.anatomy.n_layers):
        if layer not in block_outputs:
            missing.append(layer)
            continue
        hook_out = block_outputs[layer][0]
        expected = capture.streams[layer + 1]
        diff = (hook_out - expected).abs()
        layer_max = float(diff.max())
        max_diff = max(max_diff, layer_max)
        rows.append({"layer": layer, "stream_depth_expected": layer + 1, "max_abs_diff": layer_max, "mean_abs_diff": float(diff.mean()), "ok_at_tolerance": int(layer_max <= ctx.args.hook_tolerance)})
    by_layer = ctx.path("diagnostics", "hook_parity_by_layer.csv")
    bench.write_csv_with_context(ctx, by_layer, rows)
    ctx.register_artifact(by_layer, "diagnostic", "Exact rendered-prompt hook parity by layer.")
    ok = not missing and max_diff <= ctx.args.hook_tolerance and len(rows) == bundle.anatomy.n_layers
    payload = {"prompt_hash": short_hash(rendered_prompt), "max_abs_diff": max_diff, "tolerance": ctx.args.hook_tolerance, "missing_layers": missing, "ok": bool(ok)}
    path = ctx.path("diagnostics", "hook_parity.json")
    bench.write_json(path, payload)
    ctx.register_artifact(path, "diagnostic", "Exact rendered-prompt hook parity summary.")
    print(f"[lab36] hook parity: {'OK' if ok else 'MISMATCH'} max_abs={max_diff:g}")
    if not ok and not ctx.args.allow_hook_mismatch:
        raise RuntimeError("Lab 36 hook parity failed.")
    return payload


def no_op_generation_parity(ctx: bench.RunContext, bundle: bench.ModelBundle, rendered_prompt: str) -> dict[str, Any]:
    import torch

    zero = torch.zeros(int(bundle.anatomy.d_model))
    try:
        baseline = bench.generate_continuous(bundle, [rendered_prompt], 12, max_concurrent=1, skip_special_tokens=False)[0]
        no_op = bench.generate_continuous(bundle, [rendered_prompt], 12, max_concurrent=1, skip_special_tokens=False, steer=(0, zero, 0.0))[0]
        payload = {"prompt_hash": short_hash(rendered_prompt), "baseline_generation": baseline, "noop_hook_generation": no_op, "generated_text_identical": bool(baseline == no_op), "ok": bool(baseline == no_op)}
    except Exception as exc:
        payload = {"prompt_hash": short_hash(rendered_prompt), "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    path = ctx.path("diagnostics", "noop_generation_parity.json")
    bench.write_json(path, payload)
    ctx.register_artifact(path, "diagnostic", "No-op generation hook parity.")
    return payload


def batch_invariance_audit(ctx: bench.RunContext, bundle: bench.ModelBundle, prompts: Sequence[str]) -> None:
    probe = list(prompts[:2]) if len(prompts) >= 2 else [prompts[0], prompts[0]]
    try:
        single = [bench.generate_continuous(bundle, [p], 8, max_concurrent=1, skip_special_tokens=False)[0] for p in probe]
        paired = bench.generate_continuous(bundle, probe, 8, max_concurrent=2, skip_special_tokens=False)
        payload = {"headline_policy": "headline comparisons use max_concurrent=1", "single_generations": single, "paired_generations": paired, "single_vs_paired_identical": [a == b for a, b in zip(single, paired)], "ok_for_headline": True}
    except Exception as exc:
        payload = {"ok_for_headline": True, "audit_failed": f"{type(exc).__name__}: {exc}"}
    path = ctx.path("diagnostics", "batch_invariance.json")
    bench.write_json(path, payload)
    ctx.register_artifact(path, "diagnostic", "Batch-invariance audit.")


def prompt_leakage_rows(bundle: bench.ModelBundle, items: Sequence[SeveranceItem], sources: Sequence[SourceItem], detections: Sequence[DetectionItem]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        for role, text in (("positive_prompt", item.positive_prompt), ("negative_prompt", item.negative_prompt), ("report_prompt", item.report_prompt), ("behavior_prompt", item.behavior_prompt)):
            rendered, mode = render_user(bundle, text)
            target_leak = marker_hit(text, concept_markers(item.target_concept, item.target_markers))
            wrong_leak = marker_hit(text, concept_markers(item.wrong_concept, item.wrong_markers))
            rows.append({"row_id": item.item_id, "row_type": "introspection", "role": role, "split": item.split, "render_mode": mode, "rendered_hash": short_hash(rendered), "token_count": len(bundle.tokenizer(rendered, add_special_tokens=False)["input_ids"]), "target_or_marker_present": int(target_leak), "wrong_or_marker_present": int(wrong_leak), "science_leak_failure": int(role == "report_prompt" and (target_leak or wrong_leak))})
    for src in sources:
        answer_ids = bundle.tokenizer(src.canonical_answer, add_special_tokens=False)["input_ids"]
        rows.append({"row_id": src.item_id, "row_type": "source_attribution", "role": "canonical_answer", "split": src.split, "render_mode": "teacher_forced_answer", "rendered_hash": short_hash(src.canonical_answer), "token_count": len(answer_ids), "science_leak_failure": 0, "canonical_answer_ids_sha256": sha256_ids(answer_ids)})
    for det in detections:
        rows.append({"row_id": det.item_id, "row_type": "injection_detection", "role": "report_prompt", "split": det.split, "render_mode": "chat_template_or_raw", "rendered_hash": short_hash(det.report_prompt), "token_count": len(bundle.tokenizer(det.report_prompt, add_special_tokens=False)["input_ids"]), "science_leak_failure": 0})
    return rows


def safety_rows(items: Sequence[SeveranceItem], sources: Sequence[SourceItem], detections: Sequence[DetectionItem]) -> list[dict[str, Any]]:
    payloads = []
    for item in items:
        payloads.extend([(item.item_id, "positive_prompt", item.positive_prompt), (item.item_id, "negative_prompt", item.negative_prompt), (item.item_id, "report_prompt", item.report_prompt), (item.item_id, "behavior_prompt", item.behavior_prompt)])
    for src in sources:
        payloads.extend([(src.item_id, "task_prompt", src.task_prompt), (src.item_id, "canonical_answer", src.canonical_answer)])
    for det in detections:
        payloads.extend([(det.item_id, "distractor_task", det.distractor_task), (det.item_id, "report_prompt", det.report_prompt)])
    rows = []
    for row_id, field, text in payloads:
        hits = [p for p in BLOCKED_SAFETY_PATTERNS if p in text.lower()]
        rows.append({"row_id": row_id, "field": field, "status": "blocked" if hits else "ok", "blocked_pattern_hits": ";".join(hits), "text_hash": short_hash(text)})
    return rows


def random_seed_manifest(ctx: bench.RunContext) -> None:
    payload = {"seed": int(ctx.args.seed), "python_hash_seed": os.environ.get("PYTHONHASHSEED", ""), "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM", ""), "determinism_note": "greedy decoding; seed affects random/shuffled directions"}
    path = ctx.path("diagnostics", "random_seed_manifest.json")
    bench.write_json(path, payload)
    ctx.register_artifact(path, "diagnostic", "Random seed manifest.")


def gpu_memory_rows() -> list[dict[str, Any]]:
    rows = []
    try:
        import torch

        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                rows.append({"device_index": idx, "device_name": torch.cuda.get_device_name(idx), "allocated_bytes": int(torch.cuda.memory_allocated(idx)), "reserved_bytes": int(torch.cuda.memory_reserved(idx)), "max_allocated_bytes": int(torch.cuda.max_memory_allocated(idx))})
    except Exception as exc:
        rows.append({"device_index": "", "device_name": "", "error": f"{type(exc).__name__}: {exc}"})
    return rows


def candidate_depths(bundle: bench.ModelBundle) -> list[int]:
    n = int(bundle.anatomy.n_layers)
    raw = {1, max(1, n // 4), max(1, n // 2), max(1, (3 * n) // 4), n}
    for part in re.split(r"[,; ]+", os.environ.get("LAB36_DEPTHS", "")):
        if part.strip().isdigit():
            raw.add(max(1, min(n, int(part))))
    return sorted(raw)


def capture_features(bundle: bench.ModelBundle, items: Sequence[SeveranceItem], depths: Sequence[int]) -> tuple[dict[tuple[str, str, int], Any], list[dict[str, Any]]]:
    features = {}
    rows = []
    for item in items:
        for side, prompt in (("positive", item.positive_prompt), ("negative", item.negative_prompt)):
            rendered, mode = render_user(bundle, prompt)
            cap = bench.run_with_residual_cache(bundle, rendered, add_special_tokens=False)
            for depth in depths:
                if 0 <= int(depth) < cap.streams.shape[0]:
                    features[(item.item_id, side, int(depth))] = cap.streams[int(depth), -1, :].detach().float().cpu()
            rows.append({"item_id": item.item_id, "target_concept": item.target_concept, "split": item.split, "side": side, "render_mode": mode, "rendered_hash": short_hash(rendered), "seq_len": len(cap.input_ids), "last_token_text": cap.tokens_text[-1] if cap.tokens_text else "", "depths_captured": ";".join(str(d) for d in depths), "read_position_policy": "final rendered prompt token"})
    return features, rows


def direction_from_items(items: Sequence[SeveranceItem], features: Mapping[tuple[str, str, int], Any], depth: int) -> tuple[Any | None, float]:
    import torch

    diffs = []
    for item in items:
        pos = features.get((item.item_id, "positive", depth))
        neg = features.get((item.item_id, "negative", depth))
        if pos is not None and neg is not None:
            diffs.append(pos - neg)
    if not diffs:
        return None, float("nan")
    raw = torch.stack(diffs).mean(dim=0)
    return unit(raw), float(raw.norm())


def shuffled_direction(items: Sequence[SeveranceItem], features: Mapping[tuple[str, str, int], Any], depth: int, seed: int) -> Any | None:
    import torch

    diffs = []
    for item in items:
        pos = features.get((item.item_id, "positive", depth))
        neg = features.get((item.item_id, "negative", depth))
        if pos is not None and neg is not None:
            diffs.append((1.0 if stable_seed(item.item_id, seed) % 2 == 0 else -1.0) * (pos - neg))
    return unit(torch.stack(diffs).mean(dim=0)) if diffs else None


def projection_stats(items: Sequence[SeveranceItem], features: Mapping[tuple[str, str, int], Any], direction: Any, depth: int) -> tuple[float, float]:
    pos_vals = []
    neg_vals = []
    for item in items:
        pos = features.get((item.item_id, "positive", depth))
        neg = features.get((item.item_id, "negative", depth))
        if pos is not None and neg is not None:
            pos_vals.append(float(pos @ direction))
            neg_vals.append(float(neg @ direction))
    return safe_mean([p - n for p, n in zip(pos_vals, neg_vals)]), auc_from_scores(pos_vals, neg_vals)


def build_directions(ctx: bench.RunContext, bundle: bench.ModelBundle, items: Sequence[SeveranceItem]) -> tuple[dict[str, DirectionBundle], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    depths = candidate_depths(bundle)
    features, capture_rows = capture_features(bundle, items, depths)
    directions = {}
    sweep = []
    selected = []
    d_model = int(bundle.anatomy.d_model)
    n_layers = int(bundle.anatomy.n_layers)
    b4_layer = max(0, min(n_layers - 1, n_layers // 2))
    for concept in sorted({i.target_concept for i in items}):
        citems = [i for i in items if i.target_concept == concept]
        train = [i for i in citems if split_is_train(i.split)] or citems
        val = [i for i in citems if split_is_validation(i.split)] or citems
        held = [i for i in citems if split_is_eval(i.split)] or val
        best_row = None
        best_vec = None
        best_shuf = None
        best_rms = 1.0
        for depth in depths:
            vec, norm = direction_from_items(train, features, depth)
            if vec is None:
                continue
            train_gap, train_auc = projection_stats(train, features, vec, depth)
            val_gap, val_auc = projection_stats(val, features, vec, depth)
            held_gap, held_auc = projection_stats(held, features, vec, depth)
            random_gaps = []
            for j in range(RANDOM_CONTROLS):
                rv = random_unit(d_model, stable_seed(f"{concept}|{depth}|random|{j}", int(ctx.args.seed) + 3600))
                rgap, _ = projection_stats(train, features, rv, depth)
                random_gaps.append(abs(rgap))
            shuffled_gaps = []
            shuffled_vecs = []
            for j in range(SHUFFLED_CONTROLS):
                sv = shuffled_direction(train, features, depth, stable_seed(f"{concept}|{depth}|shuffled|{j}", int(ctx.args.seed) + 3700))
                if sv is not None:
                    sgap, _ = projection_stats(train, features, sv, depth)
                    shuffled_gaps.append(abs(sgap))
                    shuffled_vecs.append(sv)
            adjusted = train_gap - max(safe_mean(random_gaps, 0.0), safe_mean(shuffled_gaps, 0.0))
            norms = [float(features[(i.item_id, side, depth)].norm()) for i in train for side in ("positive", "negative") if (i.item_id, side, depth) in features]
            rms = safe_mean(norms, 1.0)
            row = {"target_concept": concept, "concept_family": citems[0].concept_family, "stream_depth": depth, "injection_layer": max(0, min(n_layers - 1, depth - 1)), "b4_injection_layer": b4_layer, "depth_fraction": rounded(depth / max(1, n_layers)), "direction_norm_before_unit": rounded(norm), "residual_rms": rounded(rms), "train_gap": rounded(train_gap), "train_auc": rounded(train_auc), "validation_gap": rounded(val_gap), "validation_auc": rounded(val_auc), "heldout_gap": rounded(held_gap), "heldout_auc": rounded(held_auc), "random_gap_mean": rounded(safe_mean(random_gaps, 0.0)), "shuffled_gap_mean": rounded(safe_mean(shuffled_gaps, 0.0)), "control_adjusted_gap": rounded(adjusted), "selected": 0}
            sweep.append(row)
            if best_row is None or adjusted > fnum(best_row.get("control_adjusted_gap")):
                best_row = dict(row)
                best_row["control_adjusted_gap"] = adjusted
                best_vec = vec
                best_shuf = shuffled_vecs[0] if shuffled_vecs else -vec
                best_rms = rms
        if best_row is None or best_vec is None:
            continue
        markers = tuple(dict.fromkeys(m for i in citems for m in concept_markers(i.target_concept, i.target_markers)))
        wrong_markers = tuple(dict.fromkeys(m for i in citems for m in concept_markers(i.wrong_concept, i.wrong_markers)))
        directions[concept] = DirectionBundle(concept=concept, family=citems[0].concept_family, stream_depth=int(best_row["stream_depth"]), injection_layer=int(best_row["injection_layer"]), b4_injection_layer=b4_layer, vector=best_vec.detach().float().cpu(), random_vector=random_unit(d_model, stable_seed(f"{concept}|selected|random", int(ctx.args.seed) + 3900)), shuffled_vector=best_shuf.detach().float().cpu(), residual_rms=float(best_rms), train_gap=fnum(best_row.get("train_gap")), validation_gap=fnum(best_row.get("validation_gap")), heldout_gap=fnum(best_row.get("heldout_gap")), train_auc=fnum(best_row.get("train_auc")), validation_auc=fnum(best_row.get("validation_auc")), heldout_auc=fnum(best_row.get("heldout_auc")), markers=markers, wrong_markers=wrong_markers, target_gloss=citems[0].semantic_target_gloss, wrong_gloss=citems[0].semantic_wrong_gloss)
        selected_row = dict(best_row)
        selected_row["selected"] = 1
        selected_row["selection_rule"] = "max train control-adjusted gap; validation/heldout reported once"
        selected.append(selected_row)
        for row in sweep:
            if row["target_concept"] == concept and int(row["stream_depth"]) == int(best_row["stream_depth"]):
                row["selected"] = 1
    if not directions:
        raise RuntimeError("Lab 36 built zero usable directions.")
    return directions, sweep, selected, capture_rows


def save_directions(ctx: bench.RunContext, bundle: bench.ModelBundle, directions: Mapping[str, DirectionBundle], selected_rows: Sequence[Mapping[str, Any]]) -> None:
    import torch

    state = {"lab_id": LAB_ID, "model_id": bundle.anatomy.model_id, "d_model": bundle.anatomy.d_model, "n_layers": bundle.anatomy.n_layers, "dose_convention": "unit direction multiplied by dose * residual_rms", "directions": {n: d.vector.detach().cpu() for n, d in directions.items()}, "random_directions": {n: d.random_vector.detach().cpu() for n, d in directions.items()}, "shuffled_directions": {n: d.shuffled_vector.detach().cpu() for n, d in directions.items()}, "metadata": list(selected_rows)}
    path = ctx.path("state", "directions.pt")
    torch.save(state, path)
    ctx.register_artifact(path, "tensor", "Lab 36 contrast directions.")
    meta = {k: v for k, v in state.items() if k not in {"directions", "random_directions", "shuffled_directions"}}
    meta["direction_names"] = sorted(directions)
    meta_path = ctx.path("state", "direction_manifest.json")
    bench.write_json(meta_path, meta)
    ctx.register_artifact(meta_path, "metadata", "Human-readable Lab 36 direction manifest.")


def wrong_direction(item: SeveranceItem, directions: Mapping[str, DirectionBundle]) -> Any:
    if item.wrong_concept and item.wrong_concept in directions:
        return directions[item.wrong_concept].vector
    for name, d in directions.items():
        if name != item.target_concept:
            return d.vector
    return directions[item.target_concept].random_vector


def steer_tuple(direction: DirectionBundle, vector: Any | None, dose: float, *, layer: int | None = None) -> tuple[int, Any, float] | None:
    if vector is None or abs(float(dose)) <= 1e-12:
        return None
    return (direction.injection_layer if layer is None else int(layer), vector, float(dose) * float(direction.residual_rms))


def generate_one(bundle: bench.ModelBundle, rendered: str, *, direction: DirectionBundle | None = None, vector: Any | None = None, dose: float = 0.0, layer: int | None = None, max_new_tokens: int = REPORT_TOKENS, label: str = "lab36 generation") -> str:
    steer = None if direction is None else steer_tuple(direction, vector, dose, layer=layer)
    return bench.generate_continuous(bundle, [rendered], max_new_tokens, max_concurrent=1, skip_special_tokens=True, progress_label=label, steer=steer)[0]


def next_token_logits(bundle: bench.ModelBundle, rendered: str, *, direction: DirectionBundle | None = None, vector: Any | None = None, dose: float = 0.0, layer: int | None = None) -> Any:
    import torch

    enc = bundle.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"].to(bundle.input_device)
    mask = enc.get("attention_mask")
    if mask is not None:
        mask = mask.to(bundle.input_device)
    steer = None if direction is None else steer_tuple(direction, vector, dose, layer=layer)
    cm = bench.steering_hooks(bundle, steer[0], steer[1], steer[2]) if steer is not None else nullcontext()
    with cm, torch.inference_mode():
        out = bundle.model(input_ids=ids, attention_mask=mask, use_cache=False)
    return out.logits[0, -1, :].detach().float().cpu()


def parse_yes_no(text: str, scores: Mapping[str, float] | None = None) -> str:
    low = str(text or "").strip().lower()
    if re.match(r"^yes\b", low):
        return "yes"
    if re.match(r"^no\b", low):
        return "no"
    if scores:
        return "yes" if fnum(scores.get("yes"), -1e9) >= fnum(scores.get("no"), -1e9) else "no"
    return "unknown"


def entropy_from_logits(logits: Any) -> float:
    import torch

    p = torch.softmax(logits.float(), dim=-1).clamp_min(1e-12)
    return float(-(p * torch.log(p)).sum())


def parse_confidence(text: str) -> tuple[str, float | None]:
    low = str(text or "").lower()
    nums = re.findall(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)", low)
    if nums:
        return "numeric", max(0.0, min(1.0, float(nums[0])))
    if any(w in low for w in ("high", "confident", "certain")):
        return "high", 0.9
    if any(w in low for w in ("medium", "moderate", "somewhat")):
        return "medium", 0.5
    if any(w in low for w in ("low", "uncertain", "unsure", "not confident")):
        return "low", 0.1
    return "unparsed", None


def find_token_index(tokens_text: Sequence[str], marker: str) -> int:
    marker_low = marker.lower()
    for idx, tok in enumerate(tokens_text):
        if marker_low in tok.lower().strip():
            return idx
    return min(len(tokens_text) - 1, 0)


def run_cartography(ctx: bench.RunContext, bundle: bench.ModelBundle) -> list[dict[str, Any]]:
    path = DATA_DIR / "patchscope_prompts.csv"
    rows = []
    for row in read_csv_rows(path):
        rendered, mode = render_user(bundle, row["source_text"])
        cap = bench.run_with_residual_cache(bundle, rendered, add_special_tokens=False)
        pos = find_token_index(cap.tokens_text, str(row.get("source_marker", "")))
        for depth in candidate_depths(bundle):
            if depth >= cap.streams.shape[0]:
                continue
            logits = bench.logit_lens_all_depths(bundle, cap.streams[int(depth), pos, :].unsqueeze(0))[0]
            top = logits.topk(k=min(5, logits.numel()))
            rows.append({"source_id": row.get("source_id"), "source_role": row.get("source_role"), "source_marker": row.get("source_marker"), "render_mode": mode, "source_position": pos, "source_token_text": cap.tokens_text[pos] if cap.tokens_text else "", "depth": depth, "top1_token": bundle.tokenizer.decode([int(top.indices[0])]), "top1_logit": rounded(float(top.values[0])), "top5_tokens": ";".join(bundle.tokenizer.decode([int(i)]) for i in top.indices), "cartography_scope": "patchscope-lite logit-lens decode of marked-token residual; OBS only"})
    return rows


def b2_trial_specs(item: SeveranceItem, direction: DirectionBundle, directions: Mapping[str, DirectionBundle]) -> list[dict[str, Any]]:
    specs = [{"condition": "target_direction", "control_family": "target", "dose": d, "vector": direction.vector} for d in B2_DOSES]
    specs.extend([
        {"condition": "opposite_direction", "control_family": "opposite", "dose": -HEADLINE_DOSE, "vector": direction.vector},
        {"condition": "random_direction", "control_family": "random", "dose": HEADLINE_DOSE, "vector": direction.random_vector},
        {"condition": "shuffled_direction", "control_family": "shuffled", "dose": HEADLINE_DOSE, "vector": direction.shuffled_vector},
        {"condition": "wrong_concept_direction", "control_family": "wrong_concept", "dose": HEADLINE_DOSE, "vector": wrong_direction(item, directions)},
        {"condition": "wrong_layer_supporting", "control_family": "wrong_layer", "dose": HEADLINE_DOSE, "vector": direction.vector, "layer": max(0, min(direction.injection_layer + max(1, int(0.2 * direction.injection_layer + 1)), direction.injection_layer + 1))},
    ])
    return specs


def score_b2(item: SeveranceItem, report: str, behavior: str) -> dict[str, Any]:
    target = concept_markers(item.target_concept, item.target_markers)
    wrong = concept_markers(item.wrong_concept, item.wrong_markers)
    report_hit = marker_hit(report, target)
    wrong_hit = marker_hit(report, wrong)
    behavior_hit = marker_hit(behavior, target)
    return {"target_hit_lexical": int(report_hit), "wrong_hit_lexical": int(wrong_hit), "none_hit_lexical": int("none" in report.lower() or "no " in report.lower()), "behavior_target_visible_lexical": int(behavior_hit), "strong_grounded": int(report_hit and not behavior_hit), "rationalization_risk": int(report_hit and behavior_hit)}


def run_b2_screen(bundle: bench.ModelBundle, items: Sequence[SeveranceItem], directions: Mapping[str, DirectionBundle]) -> list[dict[str, Any]]:
    rows = []
    eval_items = [i for i in items if not split_is_train(i.split)] or list(items)
    for item in eval_items:
        if item.target_concept not in directions:
            continue
        direction = directions[item.target_concept]
        report_rendered, report_mode = render_user(bundle, make_report_user(item))
        behavior_rendered, behavior_mode = render_user(bundle, make_behavior_user(item))
        for spec in b2_trial_specs(item, direction, directions):
            layer = int(spec.get("layer", direction.injection_layer))
            dose = float(spec["dose"])
            vector = spec["vector"] if abs(dose) > 1e-12 else None
            report = generate_one(bundle, report_rendered, direction=direction, vector=vector, dose=dose, layer=layer, max_new_tokens=REPORT_TOKENS, label="lab36 B2 report")
            behavior = generate_one(bundle, behavior_rendered, direction=direction, vector=vector, dose=dose, layer=layer, max_new_tokens=BEHAVIOR_TOKENS, label="lab36 B2 behavior")
            rows.append({"track": "B2_SCREEN", "item_id": item.item_id, "split": item.split, "concept_family": item.concept_family, "target_concept": item.target_concept, "wrong_concept": item.wrong_concept, "condition": spec["condition"], "control_family": spec["control_family"], "dose": dose, "alpha_effective_residual_units": rounded(dose * direction.residual_rms), "stream_depth": direction.stream_depth, "injection_layer": layer, "report_render_mode": report_mode, "behavior_render_mode": behavior_mode, "report_text": report, "behavior_text": behavior, **score_b2(item, report, behavior), "hand_label_report": "", "hand_label_behavior": ""})
    return rows


def b2_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["target_concept"]), str(row["condition"]), float(row["dose"]))].append(row)
    out = []
    for (concept, condition, dose), sub in sorted(grouped.items()):
        out.append({"target_concept": concept, "condition": condition, "dose": dose, "n": len(sub), "target_detection_rate": rounded(safe_mean([r["target_hit_lexical"] for r in sub], 0.0)), "wrong_detection_rate": rounded(safe_mean([r["wrong_hit_lexical"] for r in sub], 0.0)), "behavior_visible_rate": rounded(safe_mean([r["behavior_target_visible_lexical"] for r in sub], 0.0)), "grounded_rate": rounded(safe_mean([r["strong_grounded"] for r in sub], 0.0)), "rationalization_risk_rate": rounded(safe_mean([r["rationalization_risk"] for r in sub], 0.0))})
    return out


def false_positive_floor_rows(summary: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for concept in sorted({str(r["target_concept"]) for r in summary}):
        sub = [r for r in summary if str(r["target_concept"]) == concept]
        target = safe_mean([r["target_detection_rate"] for r in sub if str(r["condition"]) == "target_direction" and abs(float(r["dose"]) - HEADLINE_DOSE) < 1e-9], 0.0)
        controls = [fnum(r["target_detection_rate"], 0.0) for r in sub if str(r["condition"]) in {"target_direction", "opposite_direction", "random_direction", "shuffled_direction", "wrong_concept_direction"} and not (str(r["condition"]) == "target_direction" and abs(float(r["dose"]) - HEADLINE_DOSE) < 1e-9)]
        floor = max(controls) if controls else 0.0
        wrong_layer = safe_mean([r["target_detection_rate"] for r in sub if str(r["condition"]) == "wrong_layer_supporting"], 0.0)
        out.append({"target_concept": concept, "target_detection_rate_at_headline_dose": rounded(target), "core_false_positive_floor": rounded(floor), "target_minus_core_floor": rounded(target - floor), "wrong_layer_rate_supporting": rounded(wrong_layer), "b2_screen_pass_gap_0p20": int((target - floor) >= 0.20)})
    return out


def source_prompt(src: SourceItem, condition: str) -> tuple[str, str, str]:
    base_user = f"Task: {src.task_prompt}\nAnswer the task."
    if condition == "system_prompt":
        return base_user, f"{SOURCE_SYSTEM_PROMPT} {src.system_style_instruction}", "B"
    if condition == "user_instruction":
        return f"{src.user_style_instruction}\n{base_user}", SOURCE_SYSTEM_PROMPT, "C"
    if condition == "activation_injection":
        return base_user, SOURCE_SYSTEM_PROMPT, "D"
    return base_user, SOURCE_SYSTEM_PROMPT, src.correct_source_by_condition.get(normalize_label(condition), "A")


def source_attribution_question(src: SourceItem) -> str:
    return "\n\nQuestion: Which source best explains the response style?\nA. default behavior\nB. system instruction\nC. user instruction\nD. hidden internal factor\nE. unclear\nReply with one letter only.\nAnswer:"


def forward_cache_call(model: Any, **kwargs: Any) -> Any:
    try:
        return model(**kwargs)
    except TypeError:
        kwargs.pop("position_ids", None)
        kwargs.pop("cache_position", None)
        return model(**kwargs)


def softmax_list(logits: Any) -> list[float]:
    import torch

    return [float(x) for x in torch.softmax(logits.float(), dim=-1)]


def kv_replay_label_logits(bundle: bench.ModelBundle, prompt_ids: Sequence[int], answer_ids: Sequence[int], attribution_ids: Sequence[int], *, direction: DirectionBundle | None = None, vector: Any | None = None, dose: float = 0.0, layer: int = 0) -> tuple[Any, dict[str, Any]]:
    import torch
    from transformers import DynamicCache

    device = bundle.input_device
    model = bundle.model
    ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        out = forward_cache_call(model, input_ids=ids, attention_mask=mask, past_key_values=DynamicCache(), use_cache=True)
    past = out.past_key_values
    past_len = int(ids.shape[1])
    mean_answer_logprob = []
    handle = None
    if direction is not None and vector is not None and abs(dose) > 1e-12:
        scale = float(dose) * float(direction.residual_rms)

        def hook(_module: Any, _hook_args: tuple, output: Any) -> Any:
            if isinstance(output, tuple):
                h = output[0]
                return (h + (scale * vector).to(h.device, h.dtype),) + tuple(output[1:])
            return output + (scale * vector).to(output.device, output.dtype)

        handle = bundle.blocks[int(layer)].register_forward_hook(hook)
    try:
        prev_logits = out.logits[0, -1, :].detach().float().cpu()
        for tok in answer_ids:
            probs = softmax_list(prev_logits)
            if 0 <= int(tok) < len(probs):
                mean_answer_logprob.append(math.log(max(1e-12, probs[int(tok)])))
            step = torch.tensor([[int(tok)]], dtype=torch.long, device=device)
            mask = torch.ones((1, past_len + 1), dtype=torch.long, device=device)
            pos = torch.tensor([[past_len]], dtype=torch.long, device=device)
            with torch.inference_mode():
                out = forward_cache_call(model, input_ids=step, attention_mask=mask, position_ids=pos, past_key_values=past, use_cache=True)
            past = out.past_key_values
            past_len += 1
            prev_logits = out.logits[0, -1, :].detach().float().cpu()
    finally:
        if handle is not None:
            handle.remove()
    attr = torch.tensor([list(attribution_ids)], dtype=torch.long, device=device)
    mask = torch.ones((1, past_len + len(attribution_ids)), dtype=torch.long, device=device)
    pos = torch.arange(past_len, past_len + len(attribution_ids), dtype=torch.long, device=device).unsqueeze(0)
    with torch.inference_mode():
        out = forward_cache_call(model, input_ids=attr, attention_mask=mask, position_ids=pos, past_key_values=past, use_cache=True)
    return out.logits[0, -1, :].detach().float().cpu(), {"mean_canonical_answer_logprob": safe_mean(mean_answer_logprob) if mean_answer_logprob else "", "final_past_len": past_len + len(attribution_ids)}


def full_forward_label_logits(bundle: bench.ModelBundle, ids: Sequence[int]) -> Any:
    import torch

    tensor = torch.tensor([list(ids)], dtype=torch.long, device=bundle.input_device)
    mask = torch.ones_like(tensor)
    with torch.inference_mode():
        out = bundle.model(input_ids=tensor, attention_mask=mask, use_cache=False)
    return out.logits[0, -1, :].detach().float().cpu()


def run_kv_replay_parity(ctx: bench.RunContext, bundle: bench.ModelBundle, src: SourceItem, resolver: LabelResolver) -> dict[str, Any]:
    user, system, _ = source_prompt(src, "matched_default")
    rendered, _ = render_user(bundle, user, system=system)
    tok = bundle.tokenizer
    prompt_ids = tok(rendered, add_special_tokens=False)["input_ids"]
    answer_ids = tok(src.canonical_answer, add_special_tokens=False)["input_ids"]
    attr_ids = tok(source_attribution_question(src), add_special_tokens=False)["input_ids"]
    full_ids = prompt_ids + answer_ids + attr_ids
    try:
        inc_logits, meta = kv_replay_label_logits(bundle, prompt_ids, answer_ids, attr_ids)
        full_logits = full_forward_label_logits(bundle, full_ids)
        diff = (inc_logits - full_logits).abs()
        label_inc, _ = choose_label(inc_logits, resolver)
        label_full, _ = choose_label(full_logits, resolver)
        payload = {"item_id": src.item_id, "full_input_ids_sha256": sha256_ids(full_ids), "prompt_len": len(prompt_ids), "answer_len": len(answer_ids), "attribution_len": len(attr_ids), "max_abs_logit_diff": float(diff.max()), "mean_abs_logit_diff": float(diff.mean()), "incremental_label": label_inc, "full_forward_label": label_full, "label_match": bool(label_inc == label_full), "ok": bool(float(diff.max()) <= 2e-2 or label_inc == label_full), **meta}
    except Exception as exc:
        payload = {"item_id": src.item_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    path = ctx.path("diagnostics", "kv_replay_parity.json")
    bench.write_json(path, payload)
    ctx.register_artifact(path, "diagnostic", "KV replay parity for matched-output source attribution.")
    return payload


def run_b4_source_attribution(bundle: bench.ModelBundle, sources: Sequence[SourceItem], directions: Mapping[str, DirectionBundle], resolver: LabelResolver) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    replay_rows = []
    tok = bundle.tokenizer
    for src in sources:
        direction = directions.get(src.activation_direction_id) or directions.get(src.target_style)
        if direction is None:
            continue
        answer_ids = tok(src.canonical_answer, add_special_tokens=False)["input_ids"]
        attr_ids = tok(source_attribution_question(src), add_special_tokens=False)["input_ids"]
        answer_hash = sha256_ids(answer_ids)
        for condition in SOURCE_CONDITIONS:
            user, system, fallback_expected = source_prompt(src, condition)
            expected = src.correct_source_by_condition.get(normalize_label(condition), fallback_expected)
            rendered, mode = render_user(bundle, user, system=system)
            prompt_ids = tok(rendered, add_special_tokens=False)["input_ids"]
            inject = condition == "activation_injection"
            try:
                logits, meta = kv_replay_label_logits(bundle, prompt_ids, answer_ids, attr_ids, direction=direction if inject else None, vector=direction.vector if inject else None, dose=HEADLINE_DOSE if inject else 0.0, layer=direction.b4_injection_layer)
                parsed, scores = choose_label(logits, resolver)
                fresh_logits = full_forward_label_logits(bundle, prompt_ids + answer_ids + attr_ids)
                fresh_label, fresh_scores = choose_label(fresh_logits, resolver)
                error = ""
            except Exception as exc:
                parsed, scores, fresh_label, fresh_scores, meta, error = "error", {}, "error", {}, {}, f"{type(exc).__name__}: {exc}"
            rows.append({"track": "B4_MATCHED_SOURCE", "item_id": src.item_id, "split": src.split, "target_style": src.target_style, "condition": condition, "expected_label": expected, "kv_preserved": 1, "fresh_transcript_control": 0, "injection_during_replay": int(inject), "injection_during_attribution": 0, "render_mode": mode, "prompt_ids_sha256": sha256_ids(prompt_ids), "canonical_answer_ids_sha256": answer_hash, "visible_answer_sha256": short_hash(src.canonical_answer), "parsed_label": parsed, "correct": int(parsed == expected), "hidden_label_false_alarm": int(condition != "activation_injection" and parsed == "D"), "score_A": rounded(scores.get("A")), "score_B": rounded(scores.get("B")), "score_C": rounded(scores.get("C")), "score_D": rounded(scores.get("D")), "score_E": rounded(scores.get("E")), "fresh_parsed_label": fresh_label, "fresh_correct": int(fresh_label == expected), "fresh_score_D": rounded(fresh_scores.get("D")), "b4_injection_layer": direction.b4_injection_layer, "dose": HEADLINE_DOSE if inject else 0.0, "alpha_effective_residual_units": rounded(HEADLINE_DOSE * direction.residual_rms) if inject else 0.0, "mean_canonical_answer_logprob": rounded(meta.get("mean_canonical_answer_logprob")), "error": error, "hand_label_source": ""})
            replay_rows.append({"item_id": src.item_id, "condition": condition, "canonical_answer_text": src.canonical_answer, "canonical_answer_ids_sha256": answer_hash, "visible_transcript_sha256": short_hash(src.canonical_answer + source_attribution_question(src)), "kv_preserved": 1, "injection_during_replay": int(inject), "injection_during_attribution": 0, "visible_answer_text_matched": 1, "activation_injection_layer": direction.b4_injection_layer if inject else ""})
    return rows, replay_rows


def b4_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)
    out = []
    for cond, sub in sorted(grouped.items()):
        out.append({"condition": cond, "n": len(sub), "accuracy": rounded(safe_mean([r["correct"] for r in sub], 0.0)), "fresh_transcript_accuracy": rounded(safe_mean([r["fresh_correct"] for r in sub], 0.0)), "hidden_label_false_alarm_rate": rounded(safe_mean([r["hidden_label_false_alarm"] for r in sub], 0.0)), "parsed_labels": json.dumps(dict(Counter(str(r.get("parsed_label")) for r in sub)), sort_keys=True)})
    return out


def detection_vectors(det: DetectionItem, directions: Mapping[str, DirectionBundle]) -> list[tuple[str, str, DirectionBundle | None, Any | None, float]]:
    target = directions.get(det.target_direction_id)
    wrong = directions.get(det.wrong_direction_id)
    rows = [("zero", "clean", target, None, 0.0), ("noop", "clean", target, None, 0.0)]
    if target is not None:
        rows.extend([("target_direction", "injected", target, target.vector, HEADLINE_DOSE), ("random_direction", "injected", target, target.random_vector, HEADLINE_DOSE)])
    if wrong is not None:
        rows.append(("wrong_direction", "injected", wrong, wrong.vector, HEADLINE_DOSE))
    return rows


def run_b5_detection(bundle: bench.ModelBundle, detections: Sequence[DetectionItem], directions: Mapping[str, DirectionBundle], resolver: LabelResolver) -> list[dict[str, Any]]:
    rows = []
    for det in detections:
        report_rendered, report_mode = render_user(bundle, det.report_prompt)
        behavior_rendered, behavior_mode = render_user(bundle, det.behavior_prompt)
        for condition, signal_class, direction, vector, dose in detection_vectors(det, directions):
            if direction is None:
                continue
            logits = next_token_logits(bundle, report_rendered, direction=direction, vector=vector, dose=dose, layer=direction.injection_layer)
            label, scores = choose_label(logits, resolver)
            report_text = generate_one(bundle, report_rendered, direction=direction, vector=vector, dose=dose, max_new_tokens=DETECTION_TOKENS, label="lab36 B5 report")
            behavior_text = generate_one(bundle, behavior_rendered, direction=direction, vector=vector, dose=dose, max_new_tokens=BEHAVIOR_TOKENS, label="lab36 B5 behavior")
            all_markers = tuple(dict.fromkeys(direction.markers + direction.wrong_markers))
            concept_leak = int(marker_hit(report_text, all_markers) or marker_hit(behavior_text, all_markers))
            yes_no = parse_yes_no(report_text, scores)
            rows.append({"track": "B5_ANOMALY_DETECTION", "item_id": det.item_id, "split": det.split, "condition": condition, "signal_class": signal_class, "direction_id": direction.concept, "dose": dose, "alpha_effective_residual_units": rounded(dose * direction.residual_rms), "layer": direction.injection_layer, "position_policy": det.injection_position_policy, "report_render_mode": report_mode, "behavior_render_mode": behavior_mode, "report_text": report_text, "yes_no_label": yes_no, "yes_logit": rounded(scores.get("yes")), "no_logit": rounded(scores.get("no")), "yes_no_logit_margin": rounded(fnum(scores.get("yes"), 0.0) - fnum(scores.get("no"), 0.0)), "concept_marker_hit": concept_leak, "content_leak": concept_leak, "behavior_text": behavior_text, "behavior_task_success": int(bool(str(behavior_text).strip()) and "hidden" not in str(behavior_text).lower()), "format_error": int(yes_no not in {"yes", "no"}), "hand_label_yes_no": "", "hand_label_content_leak": ""})
    return rows


def b5_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    clean = [r for r in rows if str(r.get("signal_class")) == "clean"]
    injected = [r for r in rows if str(r.get("signal_class")) == "injected"]
    hits = sum(1 for r in injected if str(r.get("yes_no_label")) == "yes")
    false_alarms = sum(1 for r in clean if str(r.get("yes_no_label")) == "yes")
    summary = [{"comparison": "all_insertions_vs_clean", "n_injected": len(injected), "n_clean": len(clean), "hit_rate": rounded(clipped_rate(hits, len(injected))) if injected else "", "false_alarm_rate": rounded(clipped_rate(false_alarms, len(clean))) if clean else "", "d_prime": rounded(d_prime(hits, len(injected), false_alarms, len(clean))) if injected and clean else "", "mean_logit_margin_injected": rounded(safe_mean([r["yes_no_logit_margin"] for r in injected], 0.0)), "mean_logit_margin_clean": rounded(safe_mean([r["yes_no_logit_margin"] for r in clean], 0.0)), "content_leak_rate": rounded(safe_mean([r["content_leak"] for r in injected], 0.0)), "behavior_task_success_rate": rounded(safe_mean([r["behavior_task_success"] for r in rows], 0.0)), "pass_gate": int(bool(injected and clean) and d_prime(hits, len(injected), false_alarms, len(clean)) >= 0.75 and clipped_rate(false_alarms, len(clean)) <= 0.25 and safe_mean([r["content_leak"] for r in injected], 1.0) <= 0.10)}]
    for cond in sorted({str(r.get("condition")) for r in rows}):
        sub = [r for r in rows if str(r.get("condition")) == cond]
        summary.append({"comparison": cond, "yes_rate": rounded(safe_mean([1 if str(r.get("yes_no_label")) == "yes" else 0 for r in sub], 0.0)), "mean_logit_margin": rounded(safe_mean([r["yes_no_logit_margin"] for r in sub], 0.0)), "content_leak_rate": rounded(safe_mean([r["content_leak"] for r in sub], 0.0))})
    return summary


def build_certainty_direction(bundle: bench.ModelBundle, qitems: Sequence[UncertaintyItem]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    import torch

    depths = candidate_depths(bundle)
    train_hi = [q for q in qitems if split_is_train(q.split) and q.expected_confidence == "high"]
    train_lo = [q for q in qitems if split_is_train(q.split) and q.expected_confidence == "low"]
    if not train_hi or not train_lo:
        return None, []
    features = {}
    rows = []
    for item in train_hi + train_lo:
        rendered, mode = render_user(bundle, make_confidence_behavior_user(item))
        cap = bench.run_with_residual_cache(bundle, rendered, add_special_tokens=False)
        for depth in depths:
            features[(item.item_id, int(depth))] = cap.streams[int(depth), -1, :].detach().float().cpu()
        rows.append({"item_id": item.item_id, "split": item.split, "expected_confidence": item.expected_confidence, "render_mode": mode, "rendered_hash": short_hash(rendered)})
    best = None
    for depth in depths:
        hi = [features[(i.item_id, depth)] for i in train_hi if (i.item_id, depth) in features]
        lo = [features[(i.item_id, depth)] for i in train_lo if (i.item_id, depth) in features]
        if not hi or not lo:
            continue
        raw = torch.stack(hi).mean(dim=0) - torch.stack(lo).mean(dim=0)
        vec = unit(raw)
        hi_scores = [float(h @ vec) for h in hi]
        lo_scores = [float(l @ vec) for l in lo]
        rec = {"depth": depth, "gap": safe_mean(hi_scores) - safe_mean(lo_scores), "auc": auc_from_scores(hi_scores, lo_scores), "vector": vec, "residual_rms": safe_mean([float(x.norm()) for x in hi + lo], 1.0)}
        if best is None or rec["gap"] > best["gap"]:
            best = rec
    return best, rows


def run_b3_certainty(bundle: bench.ModelBundle, qitems: Sequence[UncertaintyItem]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    best, capture_rows = build_certainty_direction(bundle, qitems)
    if best is None:
        return [], capture_rows, [{"condition": "skipped", "reason": "not enough high/low train rows"}]
    direction = DirectionBundle("certainty_direction", "certainty", int(best["depth"]), max(0, min(bundle.anatomy.n_layers - 1, int(best["depth"]) - 1)), max(0, min(bundle.anatomy.n_layers - 1, bundle.anatomy.n_layers // 2)), best["vector"].detach().float().cpu(), random_unit(int(bundle.anatomy.d_model), 4500), -best["vector"].detach().float().cpu(), float(best["residual_rms"]), float(best["gap"]), float("nan"), float("nan"), float(best["auc"]), float("nan"), float("nan"), ("high", "confident", "certain"), ("low", "uncertain", "unsure"), "high confidence/certainty", "low confidence/uncertainty")
    rows = []
    trials = [("zero", None, 0.0), ("certainty_plus", direction.vector, 1.0), ("certainty_minus", direction.vector, -1.0), ("random_plus", direction.random_vector, 1.0)]
    for item in [q for q in qitems if not split_is_train(q.split)] or qitems:
        report_rendered, _ = render_user(bundle, make_confidence_report_user(item))
        behavior_rendered, _ = render_user(bundle, make_confidence_behavior_user(item))
        for condition, vec, dose in trials:
            report = generate_one(bundle, report_rendered, direction=direction, vector=vec, dose=dose, max_new_tokens=CONFIDENCE_TOKENS, label="lab36 B3 confidence")
            label, numeric = parse_confidence(report)
            logits = next_token_logits(bundle, behavior_rendered, direction=direction, vector=vec, dose=dose, layer=direction.injection_layer)
            answer = generate_one(bundle, behavior_rendered, direction=direction, vector=vec, dose=dose, max_new_tokens=BEHAVIOR_TOKENS, label="lab36 B3 answer")
            rows.append({"track": "B3_DISSOCIATED_CONFIDENCE", "item_id": item.item_id, "split": item.split, "condition": condition, "dose": dose, "stream_depth": direction.stream_depth, "injection_layer": direction.injection_layer, "reported_confidence_text": report, "parsed_confidence_label": label, "parsed_confidence": "" if numeric is None else numeric, "first_token_entropy_nats": rounded(entropy_from_logits(logits)), "answer_text": answer, "answer_correct_marker": int(item.answer.lower() != "unknown" and item.answer.lower() in answer.lower()), "hand_label_confidence": "", "hand_label_correct": ""})
    return rows, capture_rows, entropy_dissociation_rows(rows)


def entropy_dissociation_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)
    out = []
    for cond, sub in sorted(grouped.items()):
        conf = [fnum(r.get("parsed_confidence")) for r in sub if str(r.get("parsed_confidence")) != ""]
        out.append({"condition": cond, "n": len(sub), "n_parsed": len(conf), "mean_parsed_confidence": rounded(safe_mean(conf)) if conf else "", "mean_first_token_entropy_nats": rounded(safe_mean([r["first_token_entropy_nats"] for r in sub], 0.0)), "accuracy": rounded(safe_mean([r["answer_correct_marker"] for r in sub], 0.0))})
    by_cond = {r["condition"]: r for r in out}
    if "certainty_plus" in by_cond and "certainty_minus" in by_cond:
        plus = fnum(by_cond["certainty_plus"].get("mean_parsed_confidence"))
        minus = fnum(by_cond["certainty_minus"].get("mean_parsed_confidence"))
        eplus = fnum(by_cond["certainty_plus"].get("mean_first_token_entropy_nats"))
        eminus = fnum(by_cond["certainty_minus"].get("mean_first_token_entropy_nats"))
        out.append({"condition": "dissociation_test", "reported_confidence_delta_plus_minus": rounded(plus - minus), "entropy_delta_plus_minus": rounded(eplus - eminus), "pass_report_delta_0p15_entropy_stable_0p05": int(math.isfinite(plus - minus) and abs(plus - minus) >= 0.15 and math.isfinite(eplus - eminus) and abs(eplus - eminus) <= 0.05)})
    return out


def yes_no_margin_from_logits(logits: Any, resolver: LabelResolver) -> float:
    scores = label_scores_from_logits(logits, resolver)
    return fnum(scores.get("yes"), 0.0) - fnum(scores.get("no"), 0.0)


def project_out(vec: Any, direction: Any) -> Any:
    d = unit(direction)
    return vec.detach().float() - float(vec.detach().float() @ d) * d


def run_patch_recovery(bundle: bench.ModelBundle, detections: Sequence[DetectionItem], directions: Mapping[str, DirectionBundle], resolver: LabelResolver) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    ablations = []
    for det in detections[: min(4, len(detections))]:
        direction = directions.get(det.target_direction_id)
        if direction is None:
            continue
        rendered, _ = render_user(bundle, det.report_prompt)
        cap = bench.run_with_residual_cache(bundle, rendered, add_special_tokens=False)
        layer = direction.injection_layer
        pos = len(cap.input_ids) - 1
        clean_metric = yes_no_margin_from_logits(next_token_logits(bundle, rendered), resolver)
        injected_metric = yes_no_margin_from_logits(next_token_logits(bundle, rendered, direction=direction, vector=direction.vector, dose=HEADLINE_DOSE, layer=layer), resolver)
        patch_vec = cap.streams[layer, pos, :].detach().float().cpu() + HEADLINE_DOSE * direction.residual_rms * direction.vector
        patched_metric = yes_no_margin_from_logits(bench.run_with_residual_patch(bundle, rendered, layer, pos, patch_vec, add_special_tokens=False), resolver)
        denom = injected_metric - clean_metric
        recovery = (patched_metric - clean_metric) / denom if abs(denom) > 1e-9 else float("nan")
        rows.append({"track": "C_LOCALIZATION", "item_id": det.item_id, "metric": "yes_no_logit_margin", "layer": layer, "position": pos, "baseline_metric": rounded(clean_metric), "intervention_metric": rounded(injected_metric), "patched_metric": rounded(patched_metric), "recovery": rounded(recovery), "valid_denominator": int(abs(denom) > 1e-9)})
        ablated_metric = yes_no_margin_from_logits(bench.run_with_residual_patch(bundle, rendered, layer, pos, project_out(patch_vec, direction.vector), add_special_tokens=False), resolver)
        ablations.append({"track": "C_LOCALIZATION", "item_id": det.item_id, "metric": "yes_no_logit_margin", "ablation": "project_out_target_direction_from_patched_residual", "layer": layer, "position": pos, "patched_metric": rounded(patched_metric), "ablated_metric": rounded(ablated_metric), "drop_from_project_out": rounded(patched_metric - ablated_metric)})
    return rows, ablations


def direction_cosine_rows(directions: Mapping[str, DirectionBundle]) -> list[dict[str, Any]]:
    rows = []
    for a in sorted(directions):
        for b in sorted(directions):
            rows.append({"direction_a": a, "direction_b": b, "family_a": directions[a].family, "family_b": directions[b].family, "cosine": rounded(cosine(directions[a].vector, directions[b].vector)), "abs_cosine": rounded(abs(cosine(directions[a].vector, directions[b].vector)))})
    return rows


def evidence_matrix_rows(direction_rows: Sequence[Mapping[str, Any]], b2_floor: Sequence[Mapping[str, Any]], b3_summary: Sequence[Mapping[str, Any]], b4_summary: Sequence[Mapping[str, Any]], b5_summary: Sequence[Mapping[str, Any]], patch_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    dir_by = {str(r["target_concept"]): r for r in direction_rows}
    b2_by = {str(r["target_concept"]): r for r in b2_floor}
    out = []
    for concept, drow in sorted(dir_by.items()):
        frow = b2_by.get(concept, {})
        out.append({"state_family": concept, "direction_heldout_auc": drow.get("heldout_auc", ""), "direction_validation_auc": drow.get("validation_auc", ""), "b2_target_minus_floor": frow.get("target_minus_core_floor", ""), "b2_pass": frow.get("b2_screen_pass_gap_0p20", ""), "allowed_claim": "B2 screen only unless B4/B5 rows below pass"})
    b4_act = next((r for r in b4_summary if str(r.get("condition")) == "activation_injection"), {})
    b5_all = next((r for r in b5_summary if str(r.get("comparison")) == "all_insertions_vs_clean"), {})
    diss = next((r for r in b3_summary if str(r.get("condition")) == "dissociation_test"), {})
    out.append({"state_family": "matched_output_source", "b4_activation_accuracy": b4_act.get("accuracy", ""), "b4_fresh_accuracy": b4_act.get("fresh_transcript_accuracy", ""), "allowed_claim": "B4_MATCHED_SOURCE only if activation beats chance and fresh/control false alarms are low"})
    out.append({"state_family": "insertion_presence", "b5_d_prime": b5_all.get("d_prime", ""), "b5_false_alarm_rate": b5_all.get("false_alarm_rate", ""), "b5_content_leak_rate": b5_all.get("content_leak_rate", ""), "b5_pass": b5_all.get("pass_gate", ""), "patch_recovery_max": rounded(max([fnum(r.get("recovery")) for r in patch_rows], default=float("nan"))) if patch_rows else "", "allowed_claim": "B5_ANOMALY_DETECTION only if d-prime passes and content leak stays low"})
    out.append({"state_family": "certainty_bridge", "b3_confidence_delta": diss.get("reported_confidence_delta_plus_minus", ""), "b3_entropy_delta": diss.get("entropy_delta_plus_minus", ""), "b3_pass": diss.get("pass_report_delta_0p15_entropy_stable_0p05", ""), "allowed_claim": "B3_DISSOCIATED_CONFIDENCE only if confidence moves while entropy/correctness stay stable"})
    return out


def aggregate_metrics(items: Sequence[SeveranceItem], directions: Mapping[str, DirectionBundle], b2_floor: Sequence[Mapping[str, Any]], b4_summary: Sequence[Mapping[str, Any]], b5_summary: Sequence[Mapping[str, Any]], b3_summary: Sequence[Mapping[str, Any]], patch_rows: Sequence[Mapping[str, Any]], mode: set[str]) -> dict[str, Any]:
    b2_gaps = [fnum(r.get("target_minus_core_floor")) for r in b2_floor if math.isfinite(fnum(r.get("target_minus_core_floor")))]
    b4_act = next((r for r in b4_summary if str(r.get("condition")) == "activation_injection"), {})
    b5_all = next((r for r in b5_summary if str(r.get("comparison")) == "all_insertions_vs_clean"), {})
    b3_diss = next((r for r in b3_summary if str(r.get("condition")) == "dissociation_test"), {})
    metrics = {"lab": LAB_ID, "mode": ",".join(sorted(mode)), "n_items": len(items), "n_directions": len(directions), "mean_direction_heldout_auc": rounded(safe_mean([d.heldout_auc for d in directions.values()])), "mean_b2_target_minus_floor": rounded(safe_mean(b2_gaps)) if b2_gaps else "", "b4_activation_source_accuracy": b4_act.get("accuracy", ""), "b4_activation_fresh_accuracy": b4_act.get("fresh_transcript_accuracy", ""), "b5_d_prime_all_insertions": b5_all.get("d_prime", ""), "b5_false_alarm_rate": b5_all.get("false_alarm_rate", ""), "b5_content_leak_rate": b5_all.get("content_leak_rate", ""), "b3_confidence_delta": b3_diss.get("reported_confidence_delta_plus_minus", ""), "b3_entropy_delta": b3_diss.get("entropy_delta_plus_minus", ""), "max_patch_recovery": rounded(max([fnum(r.get("recovery")) for r in patch_rows], default=float("nan"))) if patch_rows else ""}
    b4_acc = fnum(metrics.get("b4_activation_source_accuracy"), 0.0)
    b4_fresh = fnum(metrics.get("b4_activation_fresh_accuracy"), 0.0)
    b5_dp = fnum(metrics.get("b5_d_prime_all_insertions"), 0.0)
    b5_leak = fnum(metrics.get("b5_content_leak_rate"), 1.0)
    if b4_acc >= 0.35 and b4_acc - b4_fresh > 0.10:
        verdict = "b4_matched_source_candidate"
    elif b5_dp >= 0.75 and b5_leak <= 0.10:
        verdict = "b5_anomaly_detection_candidate"
    elif b2_gaps and safe_mean(b2_gaps) >= 0.20:
        verdict = "b2_screen_only_propagation_explicable"
    else:
        verdict = "no_report_channel_coupling_validated"
    metrics["verdict"] = verdict
    return metrics


def write_labeling_guide(ctx: bench.RunContext) -> None:
    lines = ["# Lab 36 Hand-Labeling Guide", "", "Auto labels are lexical heuristics. Fill human labels before strong claims.", "", "- `tables/b2_injection_generations.csv`: label whether report text genuinely names the target state and whether behavior visibly expresses it.", "- `tables/source_attribution_results.csv`: label source as A/B/C/D/E and mark visible-style-driven rationalizations.", "- `tables/injection_detection_results.csv`: label yes/no and content leak.", "- `tables/uncertainty_bridge_results.csv`: label confidence and answer correctness.", "", "A B2 hit without B4/B5 support is propagation-explicable. A B5 yes with concept words is content leakage, not anomaly detection."]
    path = ctx.path("tables", "hand_labeling_guide.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "guide", "Hand-labeling guide for Severance report-channel artifacts.")


def write_operationalization_audit(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    lines = ["# Lab 36 Operationalization Audit", "", "```yaml", 'headline_claim: "A model report channel may be functionally coupled to hidden state/source/anomaly variables."', 'cheap_explanation: "Report text is steered propagation, prompt prior, visible-output rationalization, option bias, or scorer bias."', 'killer_control: "B4 matched-output KV replay plus fresh transcript, and B5 content-blind insertion detection."', f'result: "{metrics.get("verdict")}"', 'claim_allowed: "handle | correlation | no claim; never phenomenal evidence"', "```", "", "This run can support or fail to support functional report-channel coupling. It does not establish consciousness, experience, or phenomenal self-knowledge.", "", "| Risk | Artifact |", "|---|---|", "| Concept steering propagates into report logits | `tables/false_positive_floor.csv`, B4/B5 outcomes |", "| Visible answer style explains source reports | `tables/matched_output_replay_results.csv`, `tables/source_attribution_results.csv` |", "| Cache/KV replay bug creates source labels | `diagnostics/kv_replay_parity.json` |", "| Yes/no labels use wrong token IDs | `diagnostics/label_token_resolution.csv` |", "| Insertion report leaks concept content | `tables/injection_detection_results.csv` |", "| Confidence follows output entropy | `tables/entropy_dissociation.csv` |"]
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "audit", "Lab 36 operationalization audit.")


def write_report(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    lines = ["# Lab 36 Find the Wire Report", "", "## Verdict", "", f"`{metrics.get('verdict')}`", "", "## Headline Numbers", "", f"- Mean direction heldout AUC: {metrics.get('mean_direction_heldout_auc')}", f"- Mean B2 target-minus-floor: {metrics.get('mean_b2_target_minus_floor')}", f"- B4 activation-source accuracy: {metrics.get('b4_activation_source_accuracy')}", f"- B4 fresh-transcript accuracy: {metrics.get('b4_activation_fresh_accuracy')}", f"- B5 d-prime all insertions: {metrics.get('b5_d_prime_all_insertions')}", f"- B5 false-alarm rate: {metrics.get('b5_false_alarm_rate')}", f"- B5 content-leak rate: {metrics.get('b5_content_leak_rate')}", f"- B3 confidence delta: {metrics.get('b3_confidence_delta')}", f"- B3 entropy delta: {metrics.get('b3_entropy_delta')}", f"- Max patch recovery: {metrics.get('max_patch_recovery')}", "", "## Read Next", "", "1. `tables/evidence_matrix.csv` for the claim boundary.", "2. `diagnostics/kv_replay_parity.json` before trusting B4.", "3. `tables/source_attribution_results.csv` and `tables/matched_output_replay_results.csv` for B4.", "4. `tables/injection_detection_results.csv` for B5 content leakage.", "5. `operationalization_audit.md` before writing ledger claims."]
    path = ctx.path("find_the_wire_report.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "report", "Read-first Lab 36 Severance report.")


def write_run_summary(ctx: bench.RunContext, metrics: Mapping[str, Any], data_manifest: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# Lab 36 Run Summary", "", f"- Model: `{ctx.model_id}`", f"- Mode: `{metrics.get('mode')}`", f"- Verdict: `{metrics.get('verdict')}`", f"- Data files: {len(data_manifest)} hash-locked files under `severance/data/`", "", "This lab treats B2 concept-report steering as a screen only. Headline functional coupling requires B4 matched-output source attribution or B5 content-blind insertion detection."]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Human-readable Lab 36 summary.")


def write_ledger(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    run_name = ctx.run_dir.name
    claims = [
        {"id": f"{LAB_ID}-C1", "tag": "B2_SCREEN", "text": f"Concept-injection report screen mean target-minus-floor was {metrics.get('mean_b2_target_minus_floor')}; this is propagation-explicable unless B4/B5 pass.", "artifact": f"runs/{run_name}/tables/false_positive_floor.csv", "falsifier": "Random, shuffled, wrong-concept, or zero controls match target detection; hand labels remove the effect."},
        {"id": f"{LAB_ID}-C2", "tag": "B4_MATCHED_SOURCE", "text": f"Matched-output source attribution activation accuracy was {metrics.get('b4_activation_source_accuracy')} versus fresh-transcript {metrics.get('b4_activation_fresh_accuracy')}.", "artifact": f"runs/{run_name}/tables/source_attribution_results.csv", "falsifier": "KV replay parity fails, canonical answer tokens differ, or fresh transcript/source priors explain the activation label."},
        {"id": f"{LAB_ID}-C3", "tag": "B5_ANOMALY_DETECTION", "text": f"Injection-presence detection d-prime was {metrics.get('b5_d_prime_all_insertions')} with false-alarm {metrics.get('b5_false_alarm_rate')} and content leak {metrics.get('b5_content_leak_rate')}.", "artifact": f"runs/{run_name}/tables/injection_detection_results.csv", "falsifier": "Detection only appears with concept leakage, high false alarms, or degraded behavior."},
    ]
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)


def plot_dashboard(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    if ctx.args.no_plots:
        return
    import numpy as np

    rows = [("B2 gap", fnum(metrics.get("mean_b2_target_minus_floor"), 0.0)), ("B4 act acc", fnum(metrics.get("b4_activation_source_accuracy"), 0.0)), ("B4 fresh acc", fnum(metrics.get("b4_activation_fresh_accuracy"), 0.0)), ("B5 dprime/2", fnum(metrics.get("b5_d_prime_all_insertions"), 0.0) / 2.0), ("B5 no leak", 1.0 - fnum(metrics.get("b5_content_leak_rate"), 1.0)), ("B3 abs delta", abs(fnum(metrics.get("b3_confidence_delta"), 0.0)))]
    labels = [r[0] for r in rows]
    vals = [max(0.0, min(1.0, r[1])) for r in rows]
    fig, ax = bench.new_figure(figsize=(9.0, 5.4))
    x = np.arange(len(labels))
    ax.bar(x, vals, color=["#0072B2", "#009E73", "#E69F00", "#009E73", "#56B4E9", "#CC79A7"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    bench.style_ax(ax, title="Lab 36 Severance Evidence Dashboard", xlabel="track", ylabel="normalized support")
    bench.save_figure(ctx, fig, "severance_dashboard.png", "One-screen evidence dashboard for Lab 36.")


def plot_b5(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    if ctx.args.no_plots or not rows:
        return
    import numpy as np

    conds = [str(r.get("comparison")) for r in rows]
    margins = [fnum(r.get("mean_logit_margin", r.get("mean_logit_margin_injected")), 0.0) for r in rows]
    fig, ax = bench.new_figure(figsize=(9.5, 5.2))
    x = np.arange(len(conds))
    ax.bar(x, margins, color="#0072B2")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in conds], fontsize=8)
    bench.style_ax(ax, title="B5 yes/no logit margins", xlabel="condition", ylabel="yes minus no logit")
    bench.save_figure(ctx, fig, "b5_detection_margins.png", "Injection-presence yes/no margin summary.")


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    mode = parse_mode(ctx.args)
    items, sources, detections, qitems, data_manifest = load_all_data(ctx.args)
    if not items:
        raise RuntimeError("Lab 36 selected no introspection rows.")

    write_bench_note(ctx, bundle)
    random_seed_manifest(ctx)

    manifest_path = ctx.path("diagnostics", "data_manifest.json")
    bench.write_json(manifest_path, {"files": data_manifest, "selected_items": len(items), "selected_sources": len(sources), "selected_detections": len(detections), "selected_uncertainty": len(qitems)})
    ctx.register_artifact(manifest_path, "diagnostic", "Lab 36 data hashes and selection.")

    safety = safety_rows(items, sources, detections)
    safety_path = ctx.path("diagnostics", "safety_wall.csv")
    bench.write_csv_with_context(ctx, safety_path, safety)
    ctx.register_artifact(safety_path, "diagnostic", "Benign safety wall audit.")
    if any(str(r.get("status")) == "blocked" for r in safety) and os.environ.get("LAB36_ALLOW_SAFETY_AUDIT_FAIL") != "1":
        raise RuntimeError("Lab 36 safety audit blocked a prompt.")

    label_rows: list[dict[str, Any]] = []
    source_resolver, rows = resolve_label_ids(bundle, {"A": ("A", " A", "\nA"), "B": ("B", " B", "\nB"), "C": ("C", " C", "\nC"), "D": ("D", " D", "\nD"), "E": ("E", " E", "\nE")}, "source_A_to_E")
    label_rows.extend(rows)
    yesno_resolver, rows = resolve_label_ids(bundle, {"yes": ("yes", " yes", "Yes", " Yes"), "no": ("no", " no", "No", " No")}, "yes_no")
    label_rows.extend(rows)
    label_path = ctx.path("diagnostics", "label_token_resolution.csv")
    bench.write_csv_with_context(ctx, label_path, label_rows)
    ctx.register_artifact(label_path, "diagnostic", "Runtime label-token resolution.")

    leakage = prompt_leakage_rows(bundle, items, sources, detections)
    leakage_path = ctx.path("diagnostics", "prompt_leakage_audit.csv")
    bench.write_csv_with_context(ctx, leakage_path, leakage)
    ctx.register_artifact(leakage_path, "diagnostic", "Prompt leakage audit.")
    if any(int(r.get("science_leak_failure", 0)) for r in leakage) and os.environ.get("LAB36_ALLOW_LEAKAGE_FAIL") != "1":
        raise RuntimeError("Lab 36 prompt leakage audit failed for a science row.")

    rendered0, _ = render_user(bundle, make_report_user(items[0]))
    if "instrument" in mode:
        run_exact_hook_parity(ctx, bundle, rendered0)
        cap0 = bench.run_with_residual_cache(bundle, rendered0, add_special_tokens=False)
        lens_result = bench.run_lens_self_check(ctx, bundle, cap0)
        lens_path = ctx.path("diagnostics", "lens_parity.json")
        bench.write_json(lens_path, {**lens_result, "prompt_hash": short_hash(rendered0)})
        ctx.register_artifact(lens_path, "diagnostic", "Final-depth lens parity.")
        no_op_generation_parity(ctx, bundle, rendered0)
        batch_prompts = [rendered0]
        if len(items) > 1:
            batch_prompts.append(render_user(bundle, make_report_user(items[1]))[0])
        batch_invariance_audit(ctx, bundle, batch_prompts)

    rendered_hash_rows = []
    for item in items:
        for role, user in (("positive", item.positive_prompt), ("negative", item.negative_prompt), ("report", make_report_user(item)), ("behavior", make_behavior_user(item))):
            rendered, mode_name = render_user(bundle, user)
            ids = bundle.tokenizer(rendered, add_special_tokens=False)["input_ids"]
            rendered_hash_rows.append({"item_id": item.item_id, "role": role, "render_mode": mode_name, "input_ids_sha256": sha256_ids(ids), "token_count": len(ids), "rendered_text_hash": short_hash(rendered)})
    rendered_hash_path = ctx.path("diagnostics", "rendered_prompt_hashes.csv")
    bench.write_csv_with_context(ctx, rendered_hash_path, rendered_hash_rows)
    ctx.register_artifact(rendered_hash_path, "diagnostic", "Exact rendered prompt hashes.")

    inventory_path = ctx.path("tables", "introspection_queries.csv")
    inv_rows = []
    for item in items:
        row = dataclasses.asdict(item)
        row["target_markers"] = flatten_markers(item.target_markers)
        row["wrong_markers"] = flatten_markers(item.wrong_markers)
        inv_rows.append(row)
    bench.write_csv_with_context(ctx, inventory_path, inv_rows)
    ctx.register_artifact(inventory_path, "table", "Selected Lab 36 introspection rows.")

    directions: dict[str, DirectionBundle] = {}
    selected_rows: list[dict[str, Any]] = []
    if mode & {"directions", "b2", "b4", "b5", "patch"}:
        directions, sweep_rows, selected_rows, capture_rows = build_directions(ctx, bundle, items)
        capture_path = ctx.path("diagnostics", "direction_activation_capture.csv")
        bench.write_csv_with_context(ctx, capture_path, capture_rows)
        ctx.register_artifact(capture_path, "diagnostic", "Contrast-prompt activation captures.")
        sweep_path = ctx.path("tables", "direction_depth_sweep.csv")
        bench.write_csv_with_context(ctx, sweep_path, sweep_rows)
        ctx.register_artifact(sweep_path, "table", "Direction layer sweep with controls.")
        selected_path = ctx.path("tables", "direction_eval.csv")
        bench.write_csv_with_context(ctx, selected_path, selected_rows)
        ctx.register_artifact(selected_path, "table", "Selected directions and scores.")
        save_directions(ctx, bundle, directions, selected_rows)
        cos_path = ctx.path("tables", "direction_cosines.csv")
        bench.write_csv_with_context(ctx, cos_path, direction_cosine_rows(directions))
        ctx.register_artifact(cos_path, "table", "Cosine audit among Lab 36 directions.")

    if "cartography" in mode:
        cartography_rows = run_cartography(ctx, bundle)
        path = ctx.path("tables", "patchscope_decodes.csv")
        bench.write_csv_with_context(ctx, path, cartography_rows)
        ctx.register_artifact(path, "table", "Patchscope-lite cartography; OBS only.")

    b2_floor: list[dict[str, Any]] = []
    if "b2" in mode and directions:
        b2_rows = run_b2_screen(bundle, items, directions)
        path = ctx.path("tables", "b2_injection_generations.csv")
        bench.write_csv_with_context(ctx, path, b2_rows)
        ctx.register_artifact(path, "table", "B2 concept-injection report and behavior generations.")
        b2_summary = b2_summary_rows(b2_rows)
        path = ctx.path("tables", "self_report_detection_dose_response.csv")
        bench.write_csv_with_context(ctx, path, b2_summary)
        ctx.register_artifact(path, "table", "B2 dose-response summary.")
        b2_floor = false_positive_floor_rows(b2_summary)
        path = ctx.path("tables", "false_positive_floor.csv")
        bench.write_csv_with_context(ctx, path, b2_floor)
        ctx.register_artifact(path, "table", "B2 core false-positive floor.")

    b3_summary: list[dict[str, Any]] = []
    if "b3" in mode:
        b3_rows, b3_capture, b3_summary = run_b3_certainty(bundle, qitems)
        path = ctx.path("diagnostics", "certainty_direction_capture.csv")
        bench.write_csv_with_context(ctx, path, b3_capture)
        ctx.register_artifact(path, "diagnostic", "B3 certainty-direction capture rows.")
        path = ctx.path("tables", "uncertainty_bridge_results.csv")
        bench.write_csv_with_context(ctx, path, b3_rows)
        ctx.register_artifact(path, "table", "B3 certainty report/entropy rows.")
        path = ctx.path("tables", "entropy_dissociation.csv")
        bench.write_csv_with_context(ctx, path, b3_summary)
        ctx.register_artifact(path, "table", "B3 entropy-dissociation summary.")

    b4_summary: list[dict[str, Any]] = []
    if "b4" in mode and directions:
        if sources:
            run_kv_replay_parity(ctx, bundle, sources[0], source_resolver)
        b4_rows, replay_rows = run_b4_source_attribution(bundle, sources, directions, source_resolver)
        path = ctx.path("tables", "source_attribution_results.csv")
        bench.write_csv_with_context(ctx, path, b4_rows)
        ctx.register_artifact(path, "table", "B4 matched-output source attribution rows.")
        path = ctx.path("tables", "matched_output_replay_results.csv")
        bench.write_csv_with_context(ctx, path, replay_rows)
        ctx.register_artifact(path, "table", "B4 matched-output replay matching diagnostics.")
        b4_summary = b4_summary_rows(b4_rows)
        path = ctx.path("tables", "source_attribution_summary.csv")
        bench.write_csv_with_context(ctx, path, b4_summary)
        ctx.register_artifact(path, "table", "B4 source-attribution summary.")

    b5_summary: list[dict[str, Any]] = []
    if "b5" in mode and directions:
        b5_rows = run_b5_detection(bundle, detections, directions, yesno_resolver)
        path = ctx.path("tables", "injection_detection_results.csv")
        bench.write_csv_with_context(ctx, path, b5_rows)
        ctx.register_artifact(path, "table", "B5 insertion detection rows.")
        b5_summary = b5_summary_rows(b5_rows)
        path = ctx.path("tables", "injection_detection_summary.csv")
        bench.write_csv_with_context(ctx, path, b5_summary)
        ctx.register_artifact(path, "table", "B5 signal-detection summary.")

    patch_rows: list[dict[str, Any]] = []
    if "patch" in mode and directions:
        patch_rows, ablation_rows = run_patch_recovery(bundle, detections, directions, yesno_resolver)
        path = ctx.path("tables", "patch_recovery_heatmap.csv")
        bench.write_csv_with_context(ctx, path, patch_rows)
        ctx.register_artifact(path, "table", "Minimal C-track residual patch recovery rows.")
        path = ctx.path("tables", "ablation_results.csv")
        bench.write_csv_with_context(ctx, path, ablation_rows)
        ctx.register_artifact(path, "table", "Minimal project-out ablation rows.")

    evidence = evidence_matrix_rows(selected_rows, b2_floor, b3_summary, b4_summary, b5_summary, patch_rows)
    path = ctx.path("tables", "evidence_matrix.csv")
    bench.write_csv_with_context(ctx, path, evidence)
    ctx.register_artifact(path, "table", "Lab 36 final evidence matrix.")

    tuning_manifest = {"num_direction_layers_considered": len(candidate_depths(bundle)), "num_injection_layers_considered": 1, "num_doses_considered": len(B2_DOSES), "num_positions_considered": 1, "num_controls_considered": 5, "selection_metric": "train control-adjusted direction gap; fixed headline dose for smoke/pilot", "selected_config_sha256": short_hash(json.dumps(selected_rows, sort_keys=True, default=bench.json_default)), "heldout_once_note": "full science should freeze configs before expanding heldout"}
    path = ctx.path("diagnostics", "tuning_manifest.json")
    bench.write_json(path, tuning_manifest)
    ctx.register_artifact(path, "diagnostic", "Configuration search and tuning manifest.")
    path = ctx.path("state", "frozen_eval_configs.json")
    bench.write_json(path, tuning_manifest)
    ctx.register_artifact(path, "state", "Frozen evaluation config scaffold.")

    gmem_path = ctx.path("diagnostics", "gpu_memory.csv")
    bench.write_csv_with_context(ctx, gmem_path, gpu_memory_rows())
    ctx.register_artifact(gmem_path, "diagnostic", "GPU memory snapshot from torch.")

    metrics = aggregate_metrics(items, directions, b2_floor, b4_summary, b5_summary, b3_summary, patch_rows, mode)
    metrics["model_id"] = bundle.anatomy.model_id
    metrics["model_revision"] = bundle.anatomy.revision or ""
    metrics["gpt_oss_120b_skipped"] = True
    path = ctx.path("metrics.json")
    bench.write_json(path, metrics)
    ctx.register_artifact(path, "metrics", "Aggregate Lab 36 metrics.")

    result_rows = list(evidence) + list(b4_summary) + list(b5_summary) + list(b3_summary)
    if not result_rows:
        result_rows = selected_rows
    path = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, path, result_rows)
    ctx.register_artifact(path, "results", "Standard results alias for Lab 36.")

    write_labeling_guide(ctx)
    write_operationalization_audit(ctx, metrics)
    write_report(ctx, metrics)
    write_run_summary(ctx, metrics, data_manifest)
    write_ledger(ctx, metrics)
    plot_dashboard(ctx, metrics)
    plot_b5(ctx, b5_summary)
