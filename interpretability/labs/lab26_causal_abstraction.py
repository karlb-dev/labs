"""Lab 26: Causal abstraction by residual-stream resampling.

This lab turns an explanation into a test object. A hypothesis must name the
high-level variables it cares about, the residual-stream sites that are meant to
carry them, the donor rules that preserve or break those variables, and the
behavior that should survive. The experiment then tries to falsify that mapping
with interchange interventions.

Scope discipline:
  * FORMAL: the JSON spec names variables, sites, rules, and gates before the run.
  * CAUSAL: residual-stream interchange interventions change the target-vs-
    distractor logit margin under controls.
  * AUDIT: no-op checks, donor coverage, split summaries, counterexamples, and
    v2 refinement rows are written even when they are inconvenient.

This is not full path-specific causal scrubbing. It is the Lab 26 residual
resampling battery. Lab 27 owns path mediation.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import interp_bench as bench

LAB_ID = "L26"
LAB_NAME = "lab26_causal_abstraction"
DATA_FILE = "causal_abstraction_tasks.csv"
SPEC_FILES = (
    "lab26_induction_hypothesis.json",
    "lab26_relation_hypothesis.json",
)

PROMPT_SET_DOMAIN_CAPS = {"small": 6, "medium": 12, "full": 0}
MIN_BASELINE_MARGIN = 0.20
MIN_DONORS_PER_CONDITION = 1
MAX_COUNTEREXAMPLES = 32
NOOP_SCORE_ATOL = 0.05
RESIDUAL_PATCH_BATCH_SIZE = 48

CONDITION_ORDER = (
    "no_op_same_example",
    "preserve_variable",
    "break_variable",
    "random_matched",
    "wrong_site_preserve",
)
CONTROL_CONDITIONS = ("random_matched", "wrong_site_preserve")
PLOT_SOURCE_SUBDIR = "figure_sources"
CI_Z = 1.96


def residual_patch_batch_size(bundle: bench.ModelBundle) -> int:
    """Architecture-aware batch size for residual patching.

    The default batched path is verified on GPT-2 and OLMo. Gemma 4's bf16
    language-model wrapper shows large batch-shape drift for cached residual
    vectors, so Lab 26 uses single-prompt patch forwards there. The environment
    override is intentionally narrow and lab-local for future validation work.
    """
    override = os.environ.get("LAB26_RESIDUAL_PATCH_BATCH_SIZE")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            print(f"[lab26] ignoring invalid LAB26_RESIDUAL_PATCH_BATCH_SIZE={override!r}")
    model_id = str(getattr(bundle.anatomy, "model_id", "")).lower()
    arch = str(getattr(bundle.model.config, "architectures", [""])[0]).lower()
    if "gemma" in model_id or "gemma" in arch:
        return 1
    return RESIDUAL_PATCH_BATCH_SIZE

CONDITION_DISPLAY = {
    "no_op_same_example": "no-op self",
    "preserve_variable": "preserve variable",
    "break_variable": "break variable",
    "random_matched": "random matched",
    "wrong_site_preserve": "wrong-site preserve",
}

REQUIRED_DATA_COLUMNS = {
    "item_id",
    "domain",
    "split",
    "high_level_task",
    "template_family",
    "prompt",
    "target",
    "distractor",
    "source_token",
    "source_position",
    "target_position",
    "relation_family",
    "subject",
    "answer",
    "high_level_variables_json",
}

REQUIRED_SPEC_FIELDS = {
    "hypothesis_id",
    "domain",
    "behavior_metric",
    "high_level_variables",
    "low_level_sites",
    "resampling_rules",
    "predicted_preservation_min",
    "predicted_damage_when_broken_min",
    "predicted_specificity_gap_min",
}

BUILTIN_SPECS: dict[str, dict[str, Any]] = {
    "lab26_induction_hypothesis.json": {
        "hypothesis_id": "induction_copy_v1",
        "domain": "induction",
        "behavior_metric": "logit(target) - logit(distractor)",
        "high_level_variables": ["COPY_SOURCE", "QUERY_TOKEN"],
        "low_level_sites": [
            {
                "kind": "residual",
                "positions": ["target_position"],
                "stream_depths": "coarse",
                "rationale": "The final query-token stream should carry the instruction to copy the token after the previous query occurrence.",
            },
            {
                "kind": "residual",
                "positions": ["source_position"],
                "stream_depths": "coarse",
                "rationale": "The previous answer-token stream is a locality control for the copy-source variable.",
            },
        ],
        "resampling_rules": [
            {"preserve": ["COPY_SOURCE", "QUERY_TOKEN"], "vary": ["SURFACE_FRAME"]},
            {"preserve": ["QUERY_TOKEN"], "break": ["COPY_SOURCE"]},
        ],
        "predicted_preservation_min": 0.65,
        "predicted_damage_when_broken_min": 0.25,
        "predicted_specificity_gap_min": 0.10,
    },
    "lab26_relation_hypothesis.json": {
        "hypothesis_id": "relation_identity_v1",
        "domain": "relation",
        "behavior_metric": "logit(target) - logit(distractor)",
        "high_level_variables": ["RELATION", "SUBJECT"],
        "low_level_sites": [
            {
                "kind": "residual",
                "positions": ["source_position"],
                "stream_depths": "coarse",
                "rationale": "The relation-word stream should carry relation identity while the subject remains in the recipient prompt.",
            },
            {
                "kind": "residual",
                "positions": ["target_position"],
                "stream_depths": "coarse",
                "rationale": "The final-position stream is a deliberately broader site that may mix relation, subject, and answer evidence.",
            },
        ],
        "resampling_rules": [
            {"preserve": ["RELATION"], "vary": ["SUBJECT"]},
            {"preserve": ["SUBJECT"], "break": ["RELATION"]},
        ],
        "predicted_preservation_min": 0.55,
        "predicted_damage_when_broken_min": 0.20,
        "predicted_specificity_gap_min": 0.10,
    },
}


@dataclasses.dataclass
class CausalItem:
    item_id: str
    domain: str
    split: str
    high_level_task: str
    template_family: str
    prompt: str
    target: str
    distractor: str
    source_token: str
    source_position_raw: int
    target_position_raw: int
    relation_family: str
    subject: str
    answer: str
    variables: dict[str, Any]
    raw_input_ids: list[int] = dataclasses.field(default_factory=list)
    input_ids: list[int] = dataclasses.field(default_factory=list)
    token_text: list[str] = dataclasses.field(default_factory=list)
    source_position: int = -1
    target_position: int = -1
    special_token_offset: int = 0
    target_id: int = -1
    distractor_id: int = -1
    clean_diff: float = float("nan")
    clean_rank_target: int = -1
    clean_rank_distractor: int = -1
    top_token_text: str = ""


@dataclasses.dataclass(frozen=True)
class HypothesisSpec:
    hypothesis_id: str
    domain: str
    behavior_metric: str
    high_level_variables: tuple[str, ...]
    low_level_sites: tuple[dict[str, Any], ...]
    resampling_rules: tuple[dict[str, Any], ...]
    predicted_preservation_min: float
    predicted_damage_when_broken_min: float
    predicted_specificity_gap_min: float
    path: pathlib.Path
    source: str


@dataclasses.dataclass(frozen=True)
class ResidualSite:
    hypothesis_id: str
    domain: str
    site_label: str
    position_name: str
    depths: tuple[int, ...]
    order: int
    rationale: str


@dataclasses.dataclass(frozen=True)
class DonorPlan:
    condition: str
    donor_id: str
    rule_index: int
    preserves_variables: tuple[str, ...]
    breaks_variables: tuple[str, ...]
    varies_variables: tuple[str, ...]
    same_variables: tuple[str, ...]
    different_variables: tuple[str, ...]
    expected: str
    note: str


@dataclasses.dataclass(frozen=True)
class PatchJob:
    row: dict[str, Any]
    layer: int
    position: int
    vector: Any


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json_sha(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def rounded(value: Any, digits: int = 4) -> Any:
    try:
        f = float(value)
    except Exception:
        return value
    if not math.isfinite(f):
        return ""
    return round(f, digits)


def as_float(value: Any, default: float = float("nan")) -> float:
    try:
        f = float(value)
    except Exception:
        return default
    return f if math.isfinite(f) else default


def safe_mean(values: Sequence[Any], default: float = float("nan")) -> float:
    vals = [as_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return float(statistics.fmean(vals)) if vals else default


def safe_stdev(values: Sequence[Any], default: float = float("nan")) -> float:
    vals = [as_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    if len(vals) < 2:
        return default
    return float(statistics.stdev(vals))


def item_var(item: CausalItem, key: str, default: str = "") -> str:
    return str(item.variables.get(key, default))


def split_group_for(item: CausalItem) -> str:
    s = (item.split or "").strip().lower()
    return s if s in {"train", "eval", "test", "heldout"} else "unspecified"


def write_jsonl(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), sort_keys=True, default=bench.json_default) + "\n")


def token_rank(logits: Any, token_id: int) -> int:
    try:
        values = logits.tolist()
        score = values[token_id]
        return 1 + sum(1 for v in values if v > score)
    except Exception:
        return -1


def logit_diff(logits: Any, item: CausalItem) -> float:
    return float(logits[item.target_id] - logits[item.distractor_id])


def condition_values(rows: Sequence[Mapping[str, Any]], condition: str, key: str) -> list[float]:
    vals = []
    for row in rows:
        if row.get("condition") != condition or row.get("error"):
            continue
        val = as_float(row.get(key))
        if math.isfinite(val):
            vals.append(val)
    return vals


def stable_id(*parts: Any, n: int = 16) -> str:
    """Stable short identifier for prompts, interventions, and plot rows."""
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:n]


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def finite_or_blank(value: Any, digits: int = 4) -> Any:
    val = as_float(value)
    return round(val, digits) if math.isfinite(val) else ""


def mean_ci(values: Sequence[Any]) -> dict[str, Any]:
    vals = [as_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": "", "stdev": "", "stderr": "", "ci_low": "", "ci_high": "", "min": "", "max": ""}
    mean = float(statistics.fmean(vals))
    stdev = float(statistics.stdev(vals)) if n > 1 else float("nan")
    stderr = stdev / math.sqrt(n) if n > 1 else float("nan")
    ci_low = mean - CI_Z * stderr if math.isfinite(stderr) else float("nan")
    ci_high = mean + CI_Z * stderr if math.isfinite(stderr) else float("nan")
    return {
        "n": n,
        "mean": rounded(mean),
        "stdev": rounded(stdev),
        "stderr": rounded(stderr),
        "ci_low": rounded(ci_low),
        "ci_high": rounded(ci_high),
        "min": rounded(min(vals)),
        "max": rounded(max(vals)),
    }


def condition_label(condition: str) -> str:
    return CONDITION_DISPLAY.get(condition, condition.replace("_", " "))


def plot_source_path(ctx: bench.RunContext, filename: str) -> pathlib.Path:
    return ctx.path("tables", PLOT_SOURCE_SUBDIR, filename)


def write_plot_source(
    ctx: bench.RunContext,
    filename: str,
    rows: Sequence[Mapping[str, Any]],
    description: str,
) -> pathlib.Path:
    path = plot_source_path(ctx, filename)
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", description)
    return path


# ---------------------------------------------------------------------------
# Data and spec loading
# ---------------------------------------------------------------------------


def data_path_from_args(args: Any) -> pathlib.Path:
    prompt_set = str(getattr(args, "prompt_set", "") or "")
    candidate = pathlib.Path(prompt_set)
    if prompt_set not in PROMPT_SET_DOMAIN_CAPS and candidate.suffix.lower() in {".csv", ".tsv"}:
        return candidate if candidate.is_absolute() else (bench.COURSE_ROOT / candidate).resolve()
    return bench.COURSE_ROOT / "data" / DATA_FILE


def parse_variables(raw: str, item_id: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{item_id}: invalid high_level_variables_json: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{item_id}: high_level_variables_json must decode to an object")
    return value


def builtin_smoke_rows() -> list[dict[str, str]]:
    """Tiny Tier A fallback for plumbing when the CSV is absent.

    The repo science path should use the committed CSV. This fallback exists so
    the lab file remains runnable in isolation on a laptop, with metrics clearly
    marked as smoke-only.
    """
    rows: list[dict[str, str]] = []

    def add(
        item_id: str,
        domain: str,
        split: str,
        prompt: str,
        target: str,
        distractor: str,
        source_token: str,
        source_pos: int,
        target_pos: int,
        relation_family: str,
        subject: str,
        answer: str,
        variables: Mapping[str, str],
        task: str,
        template: str,
    ) -> None:
        rows.append({
            "item_id": item_id,
            "domain": domain,
            "split": split,
            "high_level_task": task,
            "template_family": template,
            "prompt": prompt,
            "target": target,
            "distractor": distractor,
            "source_token": source_token,
            "source_position": str(source_pos),
            "target_position": str(target_pos),
            "relation_family": relation_family,
            "subject": subject,
            "answer": answer,
            "high_level_variables_json": json.dumps(dict(variables), sort_keys=True),
        })

    frames = ["red", "black", "white"]
    colors = [("green", " yellow"), ("yellow", " green"), ("white", " black"), ("black", " white")]
    for ci, (copy, distractor) in enumerate(colors):
        for fi, frame in enumerate(frames):
            split = "train" if fi == 0 else "eval"
            add(
                f"smoke_ind_{copy}_{frame}",
                "induction",
                split,
                f"{frame} blue {copy} {frame} blue {copy} {frame} blue",
                f" {copy}",
                distractor,
                copy,
                5,
                7,
                "",
                "blue",
                copy,
                {
                    "COPY_SOURCE": copy,
                    "QUERY_TOKEN": "blue",
                    "SURFACE_FRAME": f"{frame}_frame",
                    "ANSWER_CLASS": "color",
                    "SEQUENCE_RULE": "repeat_triplet",
                },
                "copy_after_query",
                "repeat_triplet",
            )

    countries = [
        ("France", " Paris", " French", " Europe", "train"),
        ("Germany", " Berlin", " German", " Europe", "eval"),
        ("Italy", " Rome", " Italian", " Europe", "train"),
        ("Japan", " Tokyo", " Japanese", " Asia", "eval"),
    ]
    for country, capital, language, continent, split in countries:
        for rel, source_token, target, distractor, answer_class in [
            ("capital_of", "capital", capital, language, "city"),
            ("language_of", "language", language, capital, "language"),
            ("continent_of", "continent", continent, capital, "continent"),
        ]:
            add(
                f"smoke_rel_{rel}_{country.lower()}",
                "relation",
                split,
                f"The {source_token} of {country} is",
                target,
                distractor,
                source_token,
                1,
                4,
                rel,
                country,
                target.strip(),
                {"RELATION": rel, "SUBJECT": country, "ANSWER_CLASS": answer_class, "SWAP_GROUP": "country_sem"},
                "relation_answer",
                "country_sem",
            )
    return rows


def rows_to_items(rows: Sequence[Mapping[str, str]]) -> list[CausalItem]:
    items: list[CausalItem] = []
    for row in rows:
        item_id = str(row["item_id"]).strip()
        items.append(CausalItem(
            item_id=item_id,
            domain=str(row["domain"]).strip(),
            split=str(row["split"]).strip(),
            high_level_task=str(row["high_level_task"]).strip(),
            template_family=str(row["template_family"]).strip(),
            prompt=str(row["prompt"]),
            target=str(row["target"]),
            distractor=str(row["distractor"]),
            source_token=str(row["source_token"]).strip(),
            source_position_raw=int(row["source_position"]),
            target_position_raw=int(row["target_position"]),
            relation_family=str(row["relation_family"]).strip(),
            subject=str(row["subject"]).strip(),
            answer=str(row["answer"]).strip(),
            variables=parse_variables(str(row["high_level_variables_json"]), item_id),
        ))
    return items


def apply_item_caps(items: list[CausalItem], args: Any) -> list[CausalItem]:
    grouped: dict[str, list[CausalItem]] = defaultdict(list)
    for item in items:
        grouped[item.domain].append(item)

    prompt_set = str(getattr(args, "prompt_set", "") or "")
    per_domain_cap = PROMPT_SET_DOMAIN_CAPS.get(prompt_set, 0)
    selected: list[CausalItem] = []
    for domain in sorted(grouped):
        domain_items = grouped[domain]
        selected.extend(domain_items[:per_domain_cap] if per_domain_cap else domain_items)

    max_examples = int(getattr(args, "max_examples", 0) or 0)
    if max_examples > 0 and len(selected) > max_examples:
        by_domain: dict[str, list[CausalItem]] = defaultdict(list)
        for item in selected:
            by_domain[item.domain].append(item)
        balanced: list[CausalItem] = []
        cursor = 0
        while len(balanced) < max_examples:
            progressed = False
            for domain in sorted(by_domain):
                if cursor < len(by_domain[domain]):
                    balanced.append(by_domain[domain][cursor])
                    progressed = True
                    if len(balanced) >= max_examples:
                        break
            if not progressed:
                break
            cursor += 1
        selected = balanced
    return selected


def load_items(ctx: bench.RunContext) -> tuple[list[CausalItem], dict[str, Any]]:
    path = data_path_from_args(ctx.args)
    source = "frozen_csv"
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            rows = [dict(row) for row in csv.DictReader(f)]
        data_sha = file_sha256(path)
        data_path = str(path)
    else:
        if str(getattr(ctx.args, "tier", "")).lower() != "a":
            raise FileNotFoundError(
                f"Lab 26 data file not found: {path}. Tier B/C science runs need the committed CSV."
            )
        print("[lab26] data CSV missing; using builtin Tier A smoke fallback. Do not ledger science claims from this run.")
        rows = builtin_smoke_rows()
        source = "builtin_tier_a_smoke_fallback"
        data_sha = hashlib.sha256("\n".join(row["item_id"] for row in rows).encode("utf-8")).hexdigest()
        data_path = str(path)

    if rows:
        missing = sorted(REQUIRED_DATA_COLUMNS - set(rows[0]))
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
    items = rows_to_items(rows)
    selected = apply_item_caps(items, ctx.args)
    domains = sorted({item.domain for item in selected})
    splits = sorted({split_group_for(item) for item in selected})
    info = {
        "data_source": source,
        "science_ready_data": source == "frozen_csv",
        "data_path": data_path,
        "data_sha256": data_sha,
        "n_rows_file": len(items),
        "n_rows_selected": len(selected),
        "domains_selected": {d: sum(1 for it in selected if it.domain == d) for d in domains},
        "splits_selected": {s: sum(1 for it in selected if split_group_for(it) == s) for s in splits},
        "prompt_set": ctx.args.prompt_set,
        "max_examples": ctx.args.max_examples,
    }
    return selected, info


def load_spec_payload(path: pathlib.Path, name: str, tier: str) -> tuple[dict[str, Any], str, str]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload, "json_file", file_sha256(path)
    if tier.lower() == "a" and name in BUILTIN_SPECS:
        payload = BUILTIN_SPECS[name]
        return payload, "builtin_tier_a_smoke_fallback", stable_json_sha(payload)
    raise FileNotFoundError(f"Lab 26 spec file not found: {path}")


def normalize_spec(path: pathlib.Path, payload: Mapping[str, Any], source: str) -> HypothesisSpec:
    return HypothesisSpec(
        hypothesis_id=str(payload.get("hypothesis_id", path.stem)),
        domain=str(payload.get("domain", "")),
        behavior_metric=str(payload.get("behavior_metric", "")),
        high_level_variables=tuple(str(x) for x in payload.get("high_level_variables", [])),
        low_level_sites=tuple(dict(x) for x in payload.get("low_level_sites", [])),
        resampling_rules=tuple(dict(x) for x in payload.get("resampling_rules", [])),
        predicted_preservation_min=float(payload.get("predicted_preservation_min", 0.0)),
        predicted_damage_when_broken_min=float(payload.get("predicted_damage_when_broken_min", 0.0)),
        predicted_specificity_gap_min=float(payload.get("predicted_specificity_gap_min", 0.0)),
        path=path,
        source=source,
    )


def validate_spec(spec: HypothesisSpec, payload: Mapping[str, Any]) -> tuple[bool, list[str]]:
    problems: list[str] = []
    missing = sorted(REQUIRED_SPEC_FIELDS - set(payload))
    if missing:
        problems.append("missing_fields=" + ";".join(missing))
    if not spec.domain:
        problems.append("empty_domain")
    if not spec.high_level_variables:
        problems.append("no_high_level_variables")
    for i, site in enumerate(spec.low_level_sites):
        if site.get("kind") != "residual":
            problems.append(f"site_{i}_non_residual_kind={site.get('kind')}")
        if not site.get("positions"):
            problems.append(f"site_{i}_no_positions")
    if not any(rule.get("preserve") for rule in spec.resampling_rules):
        problems.append("no_preserve_rule")
    if not any(rule.get("break") for rule in spec.resampling_rules):
        problems.append("no_break_rule")
    for field_name in (
        "predicted_preservation_min",
        "predicted_damage_when_broken_min",
        "predicted_specificity_gap_min",
    ):
        value = getattr(spec, field_name)
        if not math.isfinite(value) or value < 0:
            problems.append(f"bad_threshold_{field_name}={value}")
    return not problems, problems


def load_specs(ctx: bench.RunContext) -> tuple[list[HypothesisSpec], list[dict[str, Any]], list[dict[str, Any]]]:
    specs: list[HypothesisSpec] = []
    audit_rows: list[dict[str, Any]] = []
    spec_payloads: list[dict[str, Any]] = []
    spec_root = bench.COURSE_ROOT / "specs"
    for name in SPEC_FILES:
        path = spec_root / name
        payload, source, digest = load_spec_payload(path, name, str(getattr(ctx.args, "tier", "")))
        spec = normalize_spec(path, payload, source)
        ok, problems = validate_spec(spec, payload)
        if not ok:
            raise ValueError(f"{name} failed Lab 26 spec validation: {problems}")
        specs.append(spec)
        spec_payloads.append({"spec_file": str(path.relative_to(bench.COURSE_ROOT)), "source": source, "payload": payload})
        audit_rows.append({
            "spec_file": str(path.relative_to(bench.COURSE_ROOT)),
            "hypothesis_id": spec.hypothesis_id,
            "domain": spec.domain,
            "source": source,
            "ok": ok,
            "problems": ";".join(problems),
            "n_high_level_variables": len(spec.high_level_variables),
            "n_low_level_sites": len(spec.low_level_sites),
            "n_resampling_rules": len(spec.resampling_rules),
            "predicted_preservation_min": spec.predicted_preservation_min,
            "predicted_damage_when_broken_min": spec.predicted_damage_when_broken_min,
            "predicted_specificity_gap_min": spec.predicted_specificity_gap_min,
            "sha256": digest,
        })
    path = ctx.path("tables", "hypothesis_spec_audit.csv")
    bench.write_csv_with_context(ctx, path, audit_rows)
    ctx.register_artifact(path, "table", "Schema, provenance, and threshold audit for the Lab 26 hypothesis specs.")
    state_path = ctx.path("state", "hypothesis_specs_used.json")
    bench.write_json(state_path, spec_payloads)
    ctx.register_artifact(state_path, "state", "Exact hypothesis spec payloads used by this run.")
    return specs, audit_rows, spec_payloads


# ---------------------------------------------------------------------------
# Tokenization, baseline behavior, and donor planning
# ---------------------------------------------------------------------------


def find_raw_span(full_ids: Sequence[int], raw_ids: Sequence[int]) -> int | None:
    if not raw_ids:
        return None
    for start in range(0, len(full_ids) - len(raw_ids) + 1):
        if list(full_ids[start:start + len(raw_ids)]) == list(raw_ids):
            return start
    return None


def find_token_position_by_text(tokenizer: Any, input_ids: Sequence[int], expected: str) -> int | None:
    expected_norm = expected.strip().lower()
    if not expected_norm:
        return None
    hits = []
    for i, tok_id in enumerate(input_ids):
        text = tokenizer.decode([tok_id]).strip().lower()
        if text == expected_norm or expected_norm in text:
            hits.append(i)
    return hits[0] if len(hits) == 1 else None


def tokenization_gate(
    ctx: bench.RunContext, bundle: bench.ModelBundle, items: list[CausalItem]
) -> tuple[list[CausalItem], list[dict[str, Any]]]:
    tokenizer = bundle.tokenizer
    kept: list[CausalItem] = []
    rows: list[dict[str, Any]] = []
    for item in items:
        raw_ids = tokenizer.encode(item.prompt, add_special_tokens=False)
        encoded = tokenizer(item.prompt, add_special_tokens=True)
        full_ids = list(encoded["input_ids"])
        span_start = find_raw_span(full_ids, raw_ids)
        offset = int(span_start or 0)
        source_pos = item.source_position_raw + offset
        target_pos = item.target_position_raw + offset
        problems: list[str] = []
        warnings: list[str] = []

        target_ids = tokenizer.encode(item.target, add_special_tokens=False)
        distractor_ids = tokenizer.encode(item.distractor, add_special_tokens=False)
        if len(target_ids) != 1:
            problems.append(f"target_token_count={len(target_ids)}")
        if len(distractor_ids) != 1:
            problems.append(f"distractor_token_count={len(distractor_ids)}")
        if target_ids and distractor_ids and target_ids == distractor_ids:
            problems.append("target_equals_distractor_token")
        if span_start is None and full_ids != raw_ids:
            warnings.append("raw_prompt_not_contiguous_inside_special_token_encoding")
        if not (0 <= source_pos < len(full_ids)):
            problems.append("source_position_out_of_range")
        if not (0 <= target_pos < len(full_ids)):
            problems.append("target_position_out_of_range")

        source_piece = ""
        if 0 <= source_pos < len(full_ids):
            source_piece = tokenizer.decode([full_ids[source_pos]]).strip()
            if item.source_token and item.source_token.lower() not in source_piece.lower():
                found = find_token_position_by_text(tokenizer, full_ids, item.source_token)
                if found is None:
                    problems.append(f"source_token_mismatch:{source_piece}")
                else:
                    warnings.append(f"source_position_corrected_from_{source_pos}_to_{found}")
                    source_pos = found
                    source_piece = tokenizer.decode([full_ids[source_pos]]).strip()

        item.raw_input_ids = list(raw_ids)
        item.input_ids = list(full_ids)
        item.source_position = source_pos
        item.target_position = target_pos
        item.special_token_offset = offset
        item.token_text = [tokenizer.decode([tid]) for tid in full_ids]
        if not problems:
            item.target_id = int(target_ids[0])
            item.distractor_id = int(distractor_ids[0])
            kept.append(item)
        rows.append({
            "item_id": item.item_id,
            "domain": item.domain,
            "split": split_group_for(item),
            "prompt": item.prompt,
            "n_raw_tokens": len(raw_ids),
            "n_forward_tokens": len(full_ids),
            "special_token_offset": offset,
            "source_position_raw": item.source_position_raw,
            "source_position_forward": source_pos,
            "source_token_expected": item.source_token,
            "source_token_observed": source_piece,
            "target_position_raw": item.target_position_raw,
            "target_position_forward": target_pos,
            "target": item.target,
            "target_token_count": len(target_ids),
            "distractor": item.distractor,
            "distractor_token_count": len(distractor_ids),
            "kept": not problems,
            "problems": ";".join(problems),
            "warnings": ";".join(warnings),
            "tokens": " | ".join(f"{i}:{tok!r}" for i, tok in enumerate(item.token_text)),
        })
    path = ctx.path("diagnostics", "tokenization_gate.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "diagnostic", "Single-token answer and token-position validation for Lab 26 data.")
    if not kept:
        raise RuntimeError("Lab 26 tokenization gate dropped every item.")
    print(f"[lab26] tokenization gate kept {len(kept)}/{len(items)} items")
    return kept, rows


def cache_items(
    ctx: bench.RunContext, bundle: bench.ModelBundle, items: list[CausalItem]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    captures: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    report_every = max(1, len(items) // 4)
    for i, item in enumerate(items):
        cap = bench.run_with_residual_cache(bundle, item.prompt)
        if cap.input_ids != item.input_ids:
            raise RuntimeError(
                f"{item.item_id}: forward tokenization differs from tokenization gate. "
                "Check diagnostics/tokenization_gate.csv before patching positions."
            )
        captures[item.item_id] = cap
        item.clean_diff = logit_diff(cap.final_logits_last, item)
        item.clean_rank_target = token_rank(cap.final_logits_last, item.target_id)
        item.clean_rank_distractor = token_rank(cap.final_logits_last, item.distractor_id)
        top_id = int(cap.final_logits_last.argmax())
        item.top_token_text = bundle.tokenizer.decode([top_id])
        rows.append({
            "item_id": item.item_id,
            "domain": item.domain,
            "split": split_group_for(item),
            "high_level_task": item.high_level_task,
            "template_family": item.template_family,
            "relation_family": item.relation_family,
            "subject": item.subject,
            "target": item.target,
            "distractor": item.distractor,
            "clean_diff": rounded(item.clean_diff),
            "baseline_pass": item.clean_diff > MIN_BASELINE_MARGIN,
            "target_rank": item.clean_rank_target,
            "distractor_rank": item.clean_rank_distractor,
            "top_token": item.top_token_text,
            "copy_source": item_var(item, "COPY_SOURCE"),
            "query_token": item_var(item, "QUERY_TOKEN"),
            "relation": item_var(item, "RELATION", item.relation_family),
            "subject_var": item_var(item, "SUBJECT", item.subject),
            "answer_class": item_var(item, "ANSWER_CLASS"),
        })
        if (i + 1) % report_every == 0 or i + 1 == len(items):
            print(f"[lab26] cached {i + 1}/{len(items)} prompts")
    path = ctx.path("tables", "baseline_behavior.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Clean target-vs-distractor margins and baseline gate status.")
    return captures, rows


def compare_variables(target: CausalItem, donor: CausalItem) -> tuple[tuple[str, ...], tuple[str, ...]]:
    keys = sorted(set(target.variables) | set(donor.variables))
    same = []
    different = []
    for key in keys:
        if str(target.variables.get(key, "")) == str(donor.variables.get(key, "")):
            same.append(key)
        else:
            different.append(key)
    return tuple(same), tuple(different)


def choose_donor(
    item: CausalItem,
    items: Sequence[CausalItem],
    predicate: Any,
    *,
    label: str,
    avoid: set[str] | None = None,
) -> CausalItem | None:
    avoid = avoid or set()
    candidates = [cand for cand in items if cand.item_id != item.item_id and cand.item_id not in avoid and predicate(cand)]
    if not candidates:
        return None
    candidates.sort(key=lambda cand: stable_int(f"{label}|{item.item_id}|{cand.item_id}"))
    return candidates[0]


def rule_preserve(rule: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(x) for x in rule.get("preserve", []) or [])


def rule_break(rule: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(x) for x in rule.get("break", []) or [])


def rule_vary(rule: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(x) for x in rule.get("vary", []) or [])


def donor_matches_rule(target: CausalItem, donor: CausalItem, rule: Mapping[str, Any], *, require_vary: bool) -> bool:
    for key in rule_preserve(rule):
        if item_var(target, key) != item_var(donor, key):
            return False
    for key in rule_break(rule):
        if item_var(target, key) == item_var(donor, key):
            return False
    if require_vary:
        for key in rule_vary(rule):
            if item_var(target, key) == item_var(donor, key):
                return False
    return True


def first_rule(spec: HypothesisSpec, *, wants_break: bool) -> tuple[int, Mapping[str, Any]] | tuple[int, dict[str, Any]]:
    for i, rule in enumerate(spec.resampling_rules):
        has_break = bool(rule_break(rule))
        if wants_break == has_break:
            return i, rule
    return -1, {}


def build_donor_plans(
    items: list[CausalItem], specs: Sequence[HypothesisSpec]
) -> tuple[dict[str, list[DonorPlan]], list[dict[str, Any]], list[dict[str, Any]]]:
    specs_by_domain = {spec.domain: spec for spec in specs}
    by_id = {item.item_id: item for item in items}
    plans: dict[str, list[DonorPlan]] = {}
    audit_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for item in items:
        item_plans: list[DonorPlan] = [DonorPlan(
            condition="no_op_same_example",
            donor_id=item.item_id,
            rule_index=-1,
            preserves_variables=tuple(),
            breaks_variables=tuple(),
            varies_variables=tuple(),
            same_variables=tuple(sorted(item.variables)),
            different_variables=tuple(),
            expected="identity",
            note="self-patching identity control",
        )]
        spec = specs_by_domain.get(item.domain)
        preserve: CausalItem | None = None
        broken: CausalItem | None = None
        preserve_rule_index = break_rule_index = -1
        preserve_rule: Mapping[str, Any] = {}
        break_rule: Mapping[str, Any] = {}
        if spec is not None:
            preserve_rule_index, preserve_rule = first_rule(spec, wants_break=False)
            break_rule_index, break_rule = first_rule(spec, wants_break=True)
            if preserve_rule:
                preserve = choose_donor(
                    item,
                    items,
                    lambda cand: cand.domain == item.domain and donor_matches_rule(item, cand, preserve_rule, require_vary=True),
                    label="preserve_strict",
                )
                if preserve is None:
                    preserve = choose_donor(
                        item,
                        items,
                        lambda cand: cand.domain == item.domain and donor_matches_rule(item, cand, preserve_rule, require_vary=False),
                        label="preserve_relaxed",
                    )
            if break_rule:
                broken = choose_donor(
                    item,
                    items,
                    lambda cand: cand.domain == item.domain and donor_matches_rule(item, cand, break_rule, require_vary=False),
                    label="break_variable",
                )

        avoid = {p.item_id for p in (preserve, broken) if p is not None}
        if preserve is not None:
            same, diff = compare_variables(item, preserve)
            item_plans.append(DonorPlan(
                condition="preserve_variable",
                donor_id=preserve.item_id,
                rule_index=preserve_rule_index,
                preserves_variables=rule_preserve(preserve_rule),
                breaks_variables=tuple(),
                varies_variables=rule_vary(preserve_rule),
                same_variables=same,
                different_variables=diff,
                expected="preserve",
                note="donor selected by the spec preserve/vary rule",
            ))
            item_plans.append(DonorPlan(
                condition="wrong_site_preserve",
                donor_id=preserve.item_id,
                rule_index=preserve_rule_index,
                preserves_variables=rule_preserve(preserve_rule),
                breaks_variables=tuple(),
                varies_variables=rule_vary(preserve_rule),
                same_variables=same,
                different_variables=diff,
                expected="control",
                note="same preserving donor patched at a token position outside the named site",
            ))
        if broken is not None:
            same, diff = compare_variables(item, broken)
            item_plans.append(DonorPlan(
                condition="break_variable",
                donor_id=broken.item_id,
                rule_index=break_rule_index,
                preserves_variables=rule_preserve(break_rule),
                breaks_variables=rule_break(break_rule),
                varies_variables=rule_vary(break_rule),
                same_variables=same,
                different_variables=diff,
                expected="damage",
                note="donor selected by the spec break-variable rule",
            ))

        random_donor = choose_donor(
            item,
            items,
            lambda cand: cand.domain == item.domain and cand.input_ids and len(cand.input_ids) == len(item.input_ids),
            label="random_matched",
            avoid=avoid,
        )
        if random_donor is None:
            random_donor = choose_donor(
                item,
                items,
                lambda cand: cand.domain == item.domain and cand.input_ids,
                label="random_unmatched_fallback",
                avoid=avoid,
            )
        if random_donor is not None:
            same, diff = compare_variables(item, random_donor)
            item_plans.append(DonorPlan(
                condition="random_matched",
                donor_id=random_donor.item_id,
                rule_index=-1,
                preserves_variables=tuple(),
                breaks_variables=tuple(),
                varies_variables=tuple(),
                same_variables=same,
                different_variables=diff,
                expected="control",
                note="deterministic same-domain donor, length-matched when available",
            ))

        plans[item.item_id] = item_plans
        present = {plan.condition for plan in item_plans}
        coverage_rows.append({
            "item_id": item.item_id,
            "domain": item.domain,
            "split": split_group_for(item),
            "baseline_pass": item.clean_diff > MIN_BASELINE_MARGIN if math.isfinite(item.clean_diff) else "",
            "has_noop": "no_op_same_example" in present,
            "has_preserve": "preserve_variable" in present,
            "has_break": "break_variable" in present,
            "has_random": "random_matched" in present,
            "has_wrong_site": "wrong_site_preserve" in present,
            "n_conditions": len(present),
            "missing_conditions": ";".join(cond for cond in CONDITION_ORDER if cond not in present),
        })
        for plan in item_plans:
            donor = by_id[plan.donor_id]
            audit_rows.append({
                "item_id": item.item_id,
                "domain": item.domain,
                "split": split_group_for(item),
                "condition": plan.condition,
                "donor_id": plan.donor_id,
                "donor_split": split_group_for(donor),
                "same_length": len(item.input_ids) == len(donor.input_ids),
                "expected": plan.expected,
                "rule_index": plan.rule_index,
                "preserves_variables": ";".join(plan.preserves_variables),
                "breaks_variables": ";".join(plan.breaks_variables),
                "varies_variables": ";".join(plan.varies_variables),
                "same_variables": ";".join(plan.same_variables),
                "different_variables": ";".join(plan.different_variables),
                "donor_prompt": donor.prompt,
                "note": plan.note,
            })
    return plans, audit_rows, coverage_rows


# ---------------------------------------------------------------------------
# Residual sites and resampling
# ---------------------------------------------------------------------------


def position_value(item: CausalItem, name: str) -> int:
    if name == "source_position":
        return item.source_position
    if name == "target_position":
        return item.target_position
    if name == "final_position":
        return len(item.input_ids) - 1
    raise ValueError(f"Unknown Lab 26 position spec {name!r}")


def wrong_position(item: CausalItem, donor: CausalItem) -> int:
    banned = {item.source_position, item.target_position, len(item.input_ids) - 1}
    max_len = min(len(item.input_ids), len(donor.input_ids))
    for pos in range(max_len):
        if pos not in banned and item.input_ids[pos] == donor.input_ids[pos]:
            return pos
    for pos in range(max_len):
        if pos not in banned:
            return pos
    return 0


def stream_depths(bundle: bench.ModelBundle, args: Any, depth_spec: Any) -> tuple[int, ...]:
    n_layers = bundle.anatomy.n_layers
    if isinstance(depth_spec, list):
        return tuple(sorted({max(0, min(n_layers, int(d))) for d in depth_spec}))
    label = str(depth_spec).lower()
    if label == "all":
        return tuple(range(n_layers + 1))
    if label == "coarse" and str(getattr(args, "prompt_set", "")) == "full":
        return tuple(range(n_layers + 1))
    fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
    return tuple(sorted({max(0, min(n_layers, int(round(n_layers * f)))) for f in fractions}))


def iter_residual_sites(spec: HypothesisSpec, bundle: bench.ModelBundle, args: Any) -> list[ResidualSite]:
    sites: list[ResidualSite] = []
    order = 0
    for site in spec.low_level_sites:
        if site.get("kind") != "residual":
            continue
        for pos_name in site.get("positions", []):
            sites.append(ResidualSite(
                hypothesis_id=spec.hypothesis_id,
                domain=spec.domain,
                site_label=f"residual:{pos_name}",
                position_name=str(pos_name),
                depths=stream_depths(bundle, args, site.get("stream_depths", "coarse")),
                order=order,
                rationale=str(site.get("rationale", "")),
            ))
            order += 1
    return sites


def depth_claimable(bundle: bench.ModelBundle, depth: int) -> bool:
    # Depth 0 is token embedding substitution. Depth L is the final norm input,
    # which is often a readout bottleneck rather than an abstraction site.
    return 0 < int(depth) < bundle.anatomy.n_layers


def run_resampling(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    items: list[CausalItem],
    specs: list[HypothesisSpec],
    captures: dict[str, Any],
    donor_plans: dict[str, list[DonorPlan]],
) -> list[dict[str, Any]]:
    by_id = {item.item_id: item for item in items}
    specs_by_id = {spec.hypothesis_id: spec for spec in specs}
    sites_by_domain: dict[str, list[ResidualSite]] = defaultdict(list)
    for spec in specs:
        sites_by_domain[spec.domain].extend(iter_residual_sites(spec, bundle, ctx.args))

    total = 0
    for item in items:
        for site in sites_by_domain[item.domain]:
            total += len(donor_plans[item.item_id]) * len(site.depths)
    patch_batch_size = residual_patch_batch_size(bundle)
    print(f"[lab26] running {total} residual-resampling cells, batched by target prompt")
    print(f"[lab26] residual patch batch size: {patch_batch_size}")

    rows: list[dict[str, Any]] = []
    done = 0
    report_every = max(1, total // 5)
    for item in items:
        item_jobs: list[PatchJob] = []
        immediate_rows: list[dict[str, Any]] = []
        for site in sites_by_domain[item.domain]:
            spec = specs_by_id[site.hypothesis_id]
            for plan in donor_plans[item.item_id]:
                donor = by_id[plan.donor_id]
                donor_cap = captures[donor.item_id]
                same_length = len(item.input_ids) == len(donor.input_ids)
                if plan.condition == "wrong_site_preserve":
                    target_pos = wrong_position(item, donor)
                    donor_pos = target_pos
                    position_name = "wrong_position"
                    patched_site_label = "residual:wrong_position"
                else:
                    target_pos = position_value(item, site.position_name)
                    donor_pos = position_value(donor, site.position_name)
                    position_name = site.position_name
                    patched_site_label = site.site_label
                for depth in site.depths:
                    error = ""
                    if not same_length:
                        error = "donor_length_mismatch"
                    elif donor_pos >= donor_cap.streams.shape[1] or target_pos >= len(item.input_ids):
                        error = "position_out_of_range"
                    base_row = {
                        "hypothesis_id": spec.hypothesis_id,
                        "domain": item.domain,
                        "split": split_group_for(item),
                        "item_id": item.item_id,
                        "donor_id": donor.item_id,
                        "donor_split": split_group_for(donor),
                        "condition": plan.condition,
                        "expected": plan.expected,
                        "site": site.site_label,
                        "patched_site": patched_site_label,
                        "position_name": position_name,
                        "target_position_index": target_pos,
                        "donor_position_index": donor_pos,
                        "depth": depth,
                        "claimable_depth": depth_claimable(bundle, depth),
                        "site_order": site.order,
                        "clean_diff": rounded(item.clean_diff),
                        "baseline_pass": item.clean_diff > MIN_BASELINE_MARGIN,
                        "target_prompt": item.prompt,
                        "donor_prompt": donor.prompt,
                        "target": item.target,
                        "distractor": item.distractor,
                        "preserves_variables": ";".join(plan.preserves_variables),
                        "breaks_variables": ";".join(plan.breaks_variables),
                        "varies_variables": ";".join(plan.varies_variables),
                        "same_variables": ";".join(plan.same_variables),
                        "different_variables": ";".join(plan.different_variables),
                        "copy_source": item_var(item, "COPY_SOURCE"),
                        "donor_copy_source": item_var(donor, "COPY_SOURCE"),
                        "query_token": item_var(item, "QUERY_TOKEN"),
                        "donor_query_token": item_var(donor, "QUERY_TOKEN"),
                        "relation": item_var(item, "RELATION", item.relation_family),
                        "donor_relation": item_var(donor, "RELATION", donor.relation_family),
                        "subject": item_var(item, "SUBJECT", item.subject),
                        "donor_subject": item_var(donor, "SUBJECT", donor.subject),
                        "intervention_id": stable_id(
                            "lab26", spec.hypothesis_id, item.item_id, donor.item_id,
                            plan.condition, site.site_label, patched_site_label, depth,
                            target_pos, donor_pos,
                        ),
                        "recipient_id": item.item_id,
                        "prompt_id": item.item_id,
                        "target_token_id": item.target_id,
                        "distractor_token_id": item.distractor_id,
                        "control_family": "target" if plan.condition == "preserve_variable" else (
                            "identity" if plan.condition == "no_op_same_example" else "control"
                        ),
                        "error": error,
                    }
                    if error:
                        base_row.update({
                            "patched_diff": "",
                            "scrub_score": "",
                            "delta_from_clean": "",
                            "noop_abs_delta": "",
                            "noop_score_error": "",
                        })
                        immediate_rows.append(base_row)
                    else:
                        vector = donor_cap.streams[depth, donor_pos]
                        item_jobs.append(PatchJob(base_row, int(depth), int(target_pos), vector))
        if item_jobs:
            if patch_batch_size <= 1:
                for job in item_jobs:
                    logits = bench.run_with_residual_patch(bundle, item.prompt, job.layer, job.position, job.vector)
                    patched = logit_diff(logits, item)
                    clean = item.clean_diff
                    score = patched / clean if abs(clean) > 1e-9 else float("nan")
                    row = dict(job.row)
                    row.update({
                        "cached_clean_diff": rounded(item.clean_diff),
                        "batch_clean_diff": rounded(clean),
                        "batch_clean_delta_from_cached": rounded(0.0),
                        "clean_diff": rounded(clean),
                        "baseline_pass": clean > MIN_BASELINE_MARGIN,
                        "patched_diff": rounded(patched),
                        "scrub_score": rounded(score),
                        "delta_from_clean": rounded(patched - clean),
                        "noop_abs_delta": rounded(abs(patched - clean)) if row["condition"] == "no_op_same_example" else "",
                        "noop_score_error": rounded(abs(score - 1.0)) if row["condition"] == "no_op_same_example" else "",
                    })
                    rows.append(row)
                    done += 1
                    if done % report_every == 0 or done == total:
                        print(f"[lab26] resampling {done}/{total}")
            else:
                ref_layer = 0
                ref_pos = 0
                ref_vector = captures[item.item_id].streams[ref_layer, ref_pos]
                # GPU bf16 kernels can produce small batch-shape-dependent logit
                # drift. Score every intervention against a same-batch identity
                # reference so the no-op gate tests the patch, not batch shape.
                chunk_size = max(1, patch_batch_size - 1)
                for start in range(0, len(item_jobs), chunk_size):
                    chunk_jobs = item_jobs[start:start + chunk_size]
                    cells = [(ref_layer, ref_pos, ref_vector)] + [
                        (job.layer, job.position, job.vector) for job in chunk_jobs
                    ]
                    logits_list = bench.run_with_residual_patch_batched(
                        bundle,
                        item.prompt,
                        cells,
                        max_batch=len(cells),
                    )
                    clean_ref = logit_diff(logits_list[0], item)
                    for job, logits in zip(chunk_jobs, logits_list[1:]):
                        patched = logit_diff(logits, item)
                        clean = clean_ref
                        score = patched / clean if abs(clean) > 1e-9 else float("nan")
                        row = dict(job.row)
                        row.update({
                            "cached_clean_diff": rounded(item.clean_diff),
                            "batch_clean_diff": rounded(clean_ref),
                            "batch_clean_delta_from_cached": rounded(clean_ref - item.clean_diff),
                            "clean_diff": rounded(clean),
                            "baseline_pass": clean > MIN_BASELINE_MARGIN,
                            "patched_diff": rounded(patched),
                            "scrub_score": rounded(score),
                            "delta_from_clean": rounded(patched - clean),
                            "noop_abs_delta": rounded(abs(patched - clean)) if row["condition"] == "no_op_same_example" else "",
                            "noop_score_error": rounded(abs(score - 1.0)) if row["condition"] == "no_op_same_example" else "",
                        })
                        rows.append(row)
                        done += 1
                        if done % report_every == 0 or done == total:
                            print(f"[lab26] resampling {done}/{total}")
        for row in immediate_rows:
            rows.append(row)
            done += 1
            if done % report_every == 0 or done == total:
                print(f"[lab26] resampling {done}/{total}")
    return rows


# ---------------------------------------------------------------------------
# Aggregation, evidence, and counterexamples
# ---------------------------------------------------------------------------


def control_floor(random_ctl: float, wrong_site: float) -> float:
    vals = [v for v in (random_ctl, wrong_site) if math.isfinite(v)]
    if not vals:
        return float("nan")
    return max(vals)


def summarize_resampling_group(
    group: Sequence[Mapping[str, Any]], spec: HypothesisSpec, *, split_group: str
) -> dict[str, Any]:
    cond_vals = {cond: condition_values(group, cond, "scrub_score") for cond in CONDITION_ORDER}
    means = {cond: safe_mean(vals) for cond, vals in cond_vals.items()}
    stdevs = {cond: safe_stdev(vals) for cond, vals in cond_vals.items()}
    ci = {cond: mean_ci(vals) for cond, vals in cond_vals.items()}
    counts = {cond: len(cond_vals[cond]) for cond in CONDITION_ORDER}
    preserve = means["preserve_variable"]
    broken = means["break_variable"]
    random_ctl = means["random_matched"]
    wrong_site = means["wrong_site_preserve"]
    floor = control_floor(random_ctl, wrong_site)
    damage_gap = preserve - broken if math.isfinite(preserve) and math.isfinite(broken) else float("nan")
    specificity_gap = preserve - floor if math.isfinite(preserve) and math.isfinite(floor) else float("nan")
    noop = means["no_op_same_example"]
    noop_error = abs(noop - 1.0) if math.isfinite(noop) else float("nan")
    enough = counts["preserve_variable"] >= MIN_DONORS_PER_CONDITION and counts["break_variable"] >= MIN_DONORS_PER_CONDITION
    pass_preserve = math.isfinite(preserve) and preserve >= spec.predicted_preservation_min
    pass_damage = math.isfinite(damage_gap) and damage_gap >= spec.predicted_damage_when_broken_min
    pass_specificity = math.isfinite(specificity_gap) and specificity_gap >= spec.predicted_specificity_gap_min
    formal_pass = bool(enough and pass_preserve and pass_damage and pass_specificity)
    sample = group[0]
    return {
        "split_group": split_group,
        "domain": sample["domain"],
        "hypothesis_id": sample["hypothesis_id"],
        "site": sample["site"],
        "depth": int(sample["depth"]),
        "claimable_depth": bool(sample.get("claimable_depth")),
        "site_order": int(sample.get("site_order", 999)),
        "mean_noop": rounded(noop),
        "mean_preserve_variable": rounded(preserve),
        "mean_break_variable": rounded(broken),
        "mean_random_matched": rounded(random_ctl),
        "mean_wrong_site_preserve": rounded(wrong_site),
        "stdev_preserve_variable": rounded(stdevs["preserve_variable"]),
        "stdev_break_variable": rounded(stdevs["break_variable"]),
        "stderr_preserve_variable": ci["preserve_variable"]["stderr"],
        "stderr_break_variable": ci["break_variable"]["stderr"],
        "stderr_random_matched": ci["random_matched"]["stderr"],
        "stderr_wrong_site_preserve": ci["wrong_site_preserve"]["stderr"],
        "ci_low_preserve_variable": ci["preserve_variable"]["ci_low"],
        "ci_high_preserve_variable": ci["preserve_variable"]["ci_high"],
        "ci_low_break_variable": ci["break_variable"]["ci_low"],
        "ci_high_break_variable": ci["break_variable"]["ci_high"],
        "ci_low_random_matched": ci["random_matched"]["ci_low"],
        "ci_high_random_matched": ci["random_matched"]["ci_high"],
        "ci_low_wrong_site_preserve": ci["wrong_site_preserve"]["ci_low"],
        "ci_high_wrong_site_preserve": ci["wrong_site_preserve"]["ci_high"],
        "control_floor": rounded(floor),
        "damage_gap": rounded(damage_gap),
        "specificity_gap": rounded(specificity_gap),
        "noop_score_error": rounded(noop_error),
        "n_noop": counts["no_op_same_example"],
        "n_preserve": counts["preserve_variable"],
        "n_break": counts["break_variable"],
        "n_random": counts["random_matched"],
        "n_wrong_site": counts["wrong_site_preserve"],
        "enough_donors": enough,
        "pass_preservation": pass_preserve,
        "pass_damage": pass_damage,
        "pass_specificity": pass_specificity,
        "formal_pass": formal_pass,
        "predicted_preservation_min": spec.predicted_preservation_min,
        "predicted_damage_when_broken_min": spec.predicted_damage_when_broken_min,
        "predicted_specificity_gap_min": spec.predicted_specificity_gap_min,
    }


def aggregate_resampling(
    rows: list[dict[str, Any]], specs: Sequence[HypothesisSpec]
) -> list[dict[str, Any]]:
    spec_by_id = {spec.hypothesis_id: spec for spec in specs}
    out: list[dict[str, Any]] = []
    split_labels = ["all"] + sorted({str(row.get("split", "unspecified")) for row in rows})
    for split_label in split_labels:
        grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row.get("error"):
                continue
            if not row.get("claimable_depth"):
                continue
            if not row.get("baseline_pass") and str(row.get("baseline_pass")).lower() != "true":
                continue
            if split_label != "all" and row.get("split") != split_label:
                continue
            key = (str(row["domain"]), str(row["hypothesis_id"]), str(row["site"]), int(row["depth"]))
            grouped[key].append(row)
        for (_domain, hyp, _site, _depth), group in sorted(grouped.items()):
            out.append(summarize_resampling_group(group, spec_by_id[hyp], split_group=split_label))
    return out


def best_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        bool(row.get("formal_pass")),
        as_float(row.get("damage_gap"), -999.0),
        as_float(row.get("specificity_gap"), -999.0),
        as_float(row.get("mean_preserve_variable"), -999.0),
        -int(row.get("site_order", 999)),
        -int(row.get("depth", 999)),
    )


def select_best_cells(summary_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    hypotheses = sorted({str(row["hypothesis_id"]) for row in summary_rows})
    for hyp in hypotheses:
        train_candidates = [row for row in summary_rows if row["hypothesis_id"] == hyp and row["split_group"] == "train"]
        source_split = "train"
        if not train_candidates:
            train_candidates = [row for row in summary_rows if row["hypothesis_id"] == hyp and row["split_group"] == "all"]
            source_split = "all"
        if not train_candidates:
            continue
        chosen = dict(max(train_candidates, key=best_key))
        chosen["selection_split"] = source_split
        chosen["selection_rule"] = "train_split_if_available_then_formal_pass_then_gaps_then_spec_order_then_earlier_depth"
        out.append(chosen)
    return out


def matching_summary(
    summary_rows: Sequence[Mapping[str, Any]], hyp: str, site: str, depth: int, split: str
) -> Mapping[str, Any] | None:
    for row in summary_rows:
        if row["hypothesis_id"] == hyp and row["site"] == site and int(row["depth"]) == int(depth) and row["split_group"] == split:
            return row
    return None


def baseline_pass_rate(baseline_rows: Sequence[Mapping[str, Any]], domain: str, split: str = "all") -> float:
    rows = [row for row in baseline_rows if row["domain"] == domain and (split == "all" or row.get("split") == split)]
    return safe_mean([1.0 if row.get("baseline_pass") else 0.0 for row in rows], 0.0)


def build_evidence_matrix(
    best_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    counterexamples: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    counter_by_hyp = defaultdict(int)
    for row in counterexamples:
        counter_by_hyp[row["hypothesis_id"]] += 1
    evidence_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {"verdicts": {}}
    for best in best_rows:
        hyp = str(best["hypothesis_id"])
        site = str(best["site"])
        depth = int(best["depth"])
        domain = str(best["domain"])
        eval_row = matching_summary(summary_rows, hyp, site, depth, "eval")
        all_row = matching_summary(summary_rows, hyp, site, depth, "all")
        selected_row = best
        eval_pass = bool(eval_row and eval_row.get("formal_pass"))
        selected_pass = bool(selected_row.get("formal_pass"))
        has_eval = eval_row is not None
        has_counters = counter_by_hyp[hyp] > 0
        if selected_pass and eval_pass and not has_counters:
            posture = "supported_on_train_and_eval"
            evidence_tag = "FORMAL+CAUSAL"
        elif selected_pass and eval_pass:
            posture = "supported_best_cell_with_counterexamples_needing_v2_scope"
            evidence_tag = "FORMAL+CAUSAL,AUDIT"
        elif selected_pass and not has_eval:
            posture = "train_supported_but_no_eval_cell"
            evidence_tag = "FORMAL+CAUSAL,AUDIT"
        elif selected_pass:
            posture = "train_supported_but_eval_failed_or_controls_leaked"
            evidence_tag = "FORMAL+CAUSAL,AUDIT"
        elif all_row and all_row.get("formal_pass"):
            posture = "aggregate_only_candidate_requires_split_replication"
            evidence_tag = "FORMAL+CAUSAL,AUDIT"
        else:
            posture = "needs_refinement_or_negative_result"
            evidence_tag = "FORMAL+CAUSAL_FAILED,AUDIT"
        row = {
            "hypothesis_id": hyp,
            "domain": domain,
            "evidence_tag": evidence_tag,
            "selection_split": best.get("selection_split"),
            "best_site": site,
            "best_depth": depth,
            "baseline_pass_rate_all": rounded(baseline_pass_rate(baseline_rows, domain, "all")),
            "baseline_pass_rate_train": rounded(baseline_pass_rate(baseline_rows, domain, "train")),
            "baseline_pass_rate_eval": rounded(baseline_pass_rate(baseline_rows, domain, "eval")),
            "train_preservation": selected_row.get("mean_preserve_variable"),
            "train_broken_variable": selected_row.get("mean_break_variable"),
            "train_damage_gap": selected_row.get("damage_gap"),
            "train_specificity_gap": selected_row.get("specificity_gap"),
            "train_random_matched": selected_row.get("mean_random_matched"),
            "train_wrong_site_preserve": selected_row.get("mean_wrong_site_preserve"),
            "train_control_floor": selected_row.get("control_floor"),
            "train_formal_pass": selected_pass,
            "eval_preservation": eval_row.get("mean_preserve_variable", "") if eval_row else "",
            "eval_broken_variable": eval_row.get("mean_break_variable", "") if eval_row else "",
            "eval_damage_gap": eval_row.get("damage_gap", "") if eval_row else "",
            "eval_specificity_gap": eval_row.get("specificity_gap", "") if eval_row else "",
            "eval_random_matched": eval_row.get("mean_random_matched", "") if eval_row else "",
            "eval_wrong_site_preserve": eval_row.get("mean_wrong_site_preserve", "") if eval_row else "",
            "eval_control_floor": eval_row.get("control_floor", "") if eval_row else "",
            "eval_formal_pass": eval_pass if has_eval else "",
            "all_preservation": all_row.get("mean_preserve_variable", "") if all_row else "",
            "all_broken_variable": all_row.get("mean_break_variable", "") if all_row else "",
            "all_damage_gap": all_row.get("damage_gap", "") if all_row else "",
            "all_specificity_gap": all_row.get("specificity_gap", "") if all_row else "",
            "all_random_matched": all_row.get("mean_random_matched", "") if all_row else "",
            "all_wrong_site_preserve": all_row.get("mean_wrong_site_preserve", "") if all_row else "",
            "all_control_floor": all_row.get("control_floor", "") if all_row else "",
            "all_formal_pass": bool(all_row and all_row.get("formal_pass")),
            "counterexamples": counter_by_hyp[hyp],
            "claim_posture": posture,
        }
        evidence_rows.append(row)
        metrics["verdicts"][hyp] = posture
        for split in ("train", "eval", "all"):
            sr = matching_summary(summary_rows, hyp, site, depth, split)
            split_rows.append({
                "hypothesis_id": hyp,
                "domain": domain,
                "site": site,
                "depth": depth,
                "split_group": split,
                "present": sr is not None,
                "formal_pass": bool(sr and sr.get("formal_pass")),
                "preservation": sr.get("mean_preserve_variable", "") if sr else "",
                "broken_variable": sr.get("mean_break_variable", "") if sr else "",
                "random_matched": sr.get("mean_random_matched", "") if sr else "",
                "wrong_site_preserve": sr.get("mean_wrong_site_preserve", "") if sr else "",
                "damage_gap": sr.get("damage_gap", "") if sr else "",
                "specificity_gap": sr.get("specificity_gap", "") if sr else "",
            })
    return evidence_rows, split_rows, metrics


def noop_identity_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    score_grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("condition") != "no_op_same_example" or row.get("error"):
            continue
        key = (str(row["hypothesis_id"]), str(row["site"]), str(row["patched_site"]), int(row["depth"]))
        val = as_float(row.get("noop_abs_delta"))
        if math.isfinite(val):
            grouped[key].append(val)
        score_error = as_float(row.get("noop_score_error"))
        if not math.isfinite(score_error):
            score = as_float(row.get("scrub_score"))
            score_error = abs(score - 1.0) if math.isfinite(score) else float("nan")
        if math.isfinite(score_error):
            score_grouped[key].append(score_error)
    out: list[dict[str, Any]] = []
    for (hyp, site, patched_site, depth), vals in sorted(grouped.items()):
        score_vals = score_grouped.get((hyp, site, patched_site, depth), [])
        max_score_error = max(score_vals) if score_vals else float("inf")
        out.append({
            "hypothesis_id": hyp,
            "site": site,
            "patched_site": patched_site,
            "depth": depth,
            "mean_abs_delta_from_clean": rounded(safe_mean(vals)),
            "max_abs_delta_from_clean": rounded(max(vals) if vals else float("nan")),
            "mean_noop_score_error": rounded(safe_mean(score_vals)),
            "max_noop_score_error": rounded(max_score_error),
            "n": len(vals),
            "ok": max_score_error <= NOOP_SCORE_ATOL,
            "score_atol": NOOP_SCORE_ATOL,
        })
    return out


def assert_noop_identity(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = noop_identity_rows(rows)
    path = ctx.path("tables", "noop_identity_check.csv")
    bench.write_csv_with_context(ctx, path, out)
    ctx.register_artifact(path, "table", "Lab-local proof that self-resampling at every named site is numerically close to identity.")
    worst = max([as_float(row.get("max_noop_score_error"), 0.0) for row in out] or [0.0])
    if worst > NOOP_SCORE_ATOL:
        raise RuntimeError(
            f"Lab 26 no-op resampling check failed: max no-op score error {worst:.4g} exceeds {NOOP_SCORE_ATOL}. "
            "Do not interpret the resampling plots until token positions and patch hooks are fixed."
        )
    return out


def build_counterexamples(
    rows: list[dict[str, Any]], specs: Sequence[HypothesisSpec], best_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    spec_by_id = {spec.hypothesis_id: spec for spec in specs}
    best_lookup = {row["hypothesis_id"]: row for row in best_rows}
    out: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("baseline_pass") or row.get("error") or not row.get("claimable_depth"):
            continue
        spec = spec_by_id[str(row["hypothesis_id"])]
        score = as_float(row.get("scrub_score"))
        if not math.isfinite(score):
            continue
        kind = ""
        severity = 0.0
        if row["condition"] == "no_op_same_example" and abs(score - 1.0) > NOOP_SCORE_ATOL:
            kind = "noop_not_identity"
            severity = abs(score - 1.0)
        elif row["condition"] == "preserve_variable" and score < spec.predicted_preservation_min:
            kind = "preservation_failure"
            severity = spec.predicted_preservation_min - score
        elif row["condition"] == "break_variable":
            allowed_broken = spec.predicted_preservation_min - spec.predicted_damage_when_broken_min
            if score > allowed_broken:
                kind = "broken_variable_leak"
                severity = score - allowed_broken
        elif row["condition"] in CONTROL_CONDITIONS and score > spec.predicted_preservation_min:
            kind = "control_leak"
            severity = score - spec.predicted_preservation_min
        if kind:
            best = best_lookup.get(row["hypothesis_id"], {})
            out.append({
                "kind": kind,
                "severity": rounded(severity),
                "hypothesis_id": row["hypothesis_id"],
                "domain": row["domain"],
                "split": row.get("split", ""),
                "item_id": row["item_id"],
                "donor_id": row["donor_id"],
                "condition": row["condition"],
                "site": row["site"],
                "patched_site": row.get("patched_site", ""),
                "depth": row["depth"],
                "scrub_score": row["scrub_score"],
                "clean_diff": row["clean_diff"],
                "patched_diff": row["patched_diff"],
                "target_prompt": row["target_prompt"],
                "donor_prompt": row["donor_prompt"],
                "target_variables": f"COPY_SOURCE={row.get('copy_source','')}; RELATION={row.get('relation','')}; SUBJECT={row.get('subject','')}",
                "donor_variables": f"COPY_SOURCE={row.get('donor_copy_source','')}; RELATION={row.get('donor_relation','')}; SUBJECT={row.get('donor_subject','')}",
                "best_site_for_hypothesis": best.get("site", ""),
                "best_depth_for_hypothesis": best.get("depth", ""),
            })
    out.sort(key=lambda r: as_float(r["severity"], 0.0), reverse=True)
    return out[:MAX_COUNTEREXAMPLES]


def build_refinement_log(
    evidence_rows: Sequence[Mapping[str, Any]], counterexamples: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    counter_by_hyp = defaultdict(list)
    for row in counterexamples:
        counter_by_hyp[row["hypothesis_id"]].append(row)
    rows: list[dict[str, Any]] = []
    for ev in evidence_rows:
        hyp = str(ev["hypothesis_id"])
        failed: list[str] = []
        if not ev.get("train_formal_pass"):
            failed.append("train_selection_cell_failed_formal_gates")
        if ev.get("eval_formal_pass") is False:
            failed.append("eval_split_failed_same_cell")
        if ev.get("eval_formal_pass") == "":
            failed.append("no_eval_cell_for_selected_site_depth")
        if int(ev.get("counterexamples", 0)) > 0:
            failed.append("counterexamples_crossed_thresholds")
        if not failed:
            rows.append({
                "hypothesis_id": hyp,
                "version": "v1",
                "failed_rule": "",
                "evidence_path": "tables/evidence_matrix.csv",
                "revision": "No automatic revision proposed. Replicate on a larger prompt set before broadening the claim.",
                "student_notes": "",
            })
            continue
        revisions = [
            f"narrow to {ev.get('best_site')} at stream depth {ev.get('best_depth')}",
            "state the split-specific result before the aggregate result",
        ]
        if "eval_split_failed_same_cell" in failed:
            revisions.append("treat the original hypothesis as train-only until a held-out rerun passes")
        if any(row["kind"] == "broken_variable_leak" for row in counter_by_hyp[hyp]):
            revisions.append("split the high-level variable into a smaller variable or add a downstream-variable caveat")
        if any(row["kind"] == "control_leak" for row in counter_by_hyp[hyp]):
            revisions.append("do not use specificity language until random and wrong-site controls fall below preserve")
        if any(row["kind"] == "preservation_failure" for row in counter_by_hyp[hyp]):
            revisions.append("identify which prompt family fails preservation before claiming a pattern")
        rows.append({
            "hypothesis_id": hyp,
            "version": "v1",
            "failed_rule": ";".join(failed),
            "evidence_path": "tables/evidence_matrix.csv;tables/counterexamples.csv",
            "revision": "; ".join(revisions),
            "student_notes": "",
        })
        rows.append({
            "hypothesis_id": hyp,
            "version": "v2_proposed",
            "failed_rule": "student_to_test",
            "evidence_path": "tables/hypothesis_refinement_log.csv",
            "revision": "Write a new spec or claim that includes only the surviving variable, site, depth band, and split. Rerun before moving it to the ledger.",
            "student_notes": "",
        })
    return rows


# ---------------------------------------------------------------------------
# Cards and markdown artifacts
# ---------------------------------------------------------------------------


def write_plot_reading_guide(ctx: bench.RunContext) -> None:
    rows = [
        {
            "plot": "plots/causal_abstraction_dashboard.png",
            "source_table": "tables/figure_sources/dashboard_evidence.csv",
            "read_for": "One-screen posture: baseline health, train/eval support, control gaps, and counterexample load.",
            "do_not_claim": "A dashboard pass is not proof of a whole algorithm.",
        },
        {
            "plot": "plots/target_vs_control.png",
            "source_table": "tables/figure_sources/selected_cell_condition_points.csv",
            "read_for": "Raw examples, means, and uncertainty for preserving donors versus break/random/wrong-site controls at the selected cell.",
            "do_not_claim": "A mean gap is stable when the raw points or tiny-n warning disagree.",
        },
        {
            "plot": "plots/resampling_preservation_matrix.png",
            "source_table": "tables/figure_sources/resampling_matrix_source.csv",
            "read_for": "Which site/depth/condition combinations preserve the clean target margin across all rows.",
            "do_not_claim": "A hot cell by itself identifies a complete circuit or survives eval.",
        },
        {
            "plot": "plots/hypothesis_pass_fail_atlas.png",
            "source_table": "tables/figure_sources/pass_fail_atlas_source.csv",
            "read_for": "Which formal gates passed on train, eval, and all rows.",
            "do_not_claim": "A failed gate means the lab broke.",
        },
        {
            "plot": "plots/variable_specificity_ladder.png",
            "source_table": "tables/evidence_matrix.csv",
            "read_for": "How large the preserve-minus-break and preserve-minus-control gaps are at the selected cell.",
            "do_not_claim": "Preservation above random alone is enough if broken-variable donors also preserve.",
        },
        {
            "plot": "plots/split_generalization_ladder.png",
            "source_table": "tables/figure_sources/split_generalization_source.csv",
            "read_for": "Whether the train-selected best cell survives the eval split without reselection.",
            "do_not_claim": "Aggregate-only support is held-out support.",
        },
        {
            "plot": "plots/counterexample_gallery.png",
            "source_table": "tables/failure_specimens.md",
            "read_for": "The rows most likely to shrink or kill the favorite claim.",
            "do_not_claim": "Counterexamples can be ignored because aggregates look pretty.",
        },
    ]
    path = ctx.path("plots", "plot_reading_guide.csv")
    bench.write_csv(path, rows)
    ctx.register_artifact(path, "table", "Reading guide for the Lab 26 plot suite.")


def write_method_card(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    data_info: Mapping[str, Any],
    evidence_rows: Sequence[Mapping[str, Any]],
) -> None:
    science_ready = bool(data_info.get("science_ready_data")) and all(
        row.get("claim_posture") != "train_supported_but_no_eval_cell" for row in evidence_rows
    )
    lines = [
        "# Lab 26 method card",
        "",
        "Question: can a formal high-level explanation survive residual-stream resampling tests?",
        "",
        "## Scope",
        "",
        f"- model: `{bundle.anatomy.model_id}` ({bundle.anatomy.n_layers} blocks, d_model {bundle.anatomy.d_model})",
        f"- data source: `{data_info.get('data_source')}`",
        f"- science-ready data: `{bool(data_info.get('science_ready_data'))}`",
        f"- science-ready verdict posture: `{science_ready}`",
        "- intervention: residual-stream interchange at hypothesis-named token positions",
        "- metric: next-token `logit(target) - logit(distractor)` and `patched_diff / clean_diff`",
        "- claimable depths: interior stream depths only, excluding depth 0 and final-norm input",
        "- evidence tags: `FORMAL`, `CAUSAL`, and `AUDIT`",
        "",
        "## Verdict table",
        "",
        "| hypothesis | domain | site | depth | train preserve | train break | eval preserve | eval break | posture |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in evidence_rows:
        lines.append(
            f"| `{row['hypothesis_id']}` | {row['domain']} | `{row['best_site']}` | {row['best_depth']} | "
            f"{row['train_preservation']} | {row['train_broken_variable']} | {row['eval_preservation']} | "
            f"{row['eval_broken_variable']} | {row['claim_posture']} |"
        )
    lines += [
        "",
        "## Method contract",
        "",
        "Positive language requires preserving donors to beat broken-variable donors and the stronger of random/wrong-site controls. It also requires the same train-selected site/depth to survive the eval split when eval rows exist.",
        "",
        "This run can support a narrow sentence about this dataset, this model, this residual-resampling battery, and the selected site/depth. It cannot support an algorithm-identity claim, a whole-circuit claim, or a claim that the v2 refinement has already been tested.",
        "",
    ]
    path = ctx.path("method_card.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Compact method contract and Lab 26 verdict table.")


def write_spec_card(ctx: bench.RunContext, specs: Sequence[HypothesisSpec]) -> None:
    lines = [
        "# Lab 26 causal abstraction specs",
        "",
        "A Lab 26 hypothesis is a little contract:",
        "",
        "```text",
        "high-level variables -> low-level residual sites -> donor rule -> expected behavior",
        "```",
        "",
        "The run used these specs.",
        "",
    ]
    for spec in specs:
        lines += [
            f"## `{spec.hypothesis_id}`",
            "",
            f"- file: `{spec.path.relative_to(bench.COURSE_ROOT)}`",
            f"- source: `{spec.source}`",
            f"- domain: `{spec.domain}`",
            f"- behavior metric: `{spec.behavior_metric}`",
            f"- high-level variables: `{', '.join(spec.high_level_variables)}`",
            f"- gates: preserve >= {spec.predicted_preservation_min}, preserve minus break >= {spec.predicted_damage_when_broken_min}, preserve minus control >= {spec.predicted_specificity_gap_min}",
            "",
            "Low-level sites:",
            "",
        ]
        for site in spec.low_level_sites:
            lines.append(
                f"- `{site.get('kind')}` at `{site.get('positions')}` with depths `{site.get('stream_depths')}`: {site.get('rationale', '')}"
            )
        lines += ["", "Resampling rules:", ""]
        for i, rule in enumerate(spec.resampling_rules):
            lines.append(
                f"- rule {i}: preserve `{rule.get('preserve', [])}`, vary `{rule.get('vary', [])}`, break `{rule.get('break', [])}`"
            )
        lines.append("")
    lines += [
        "Machine-readable provenance is in `tables/hypothesis_spec_audit.csv` and `state/hypothesis_specs_used.json`.",
        "",
    ]
    path = ctx.path("causal_abstraction_spec.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Human-readable copy of the formal Lab 26 hypothesis specs.")


def write_operationalization_audit(
    ctx: bench.RunContext,
    evidence_rows: Sequence[Mapping[str, Any]],
    counterexamples: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Lab 26 operationalization audit",
        "",
        "```yaml",
        "headline_claim: \"a named high-level variable mapping survives residual-stream resampling\"",
        "cheap_explanation: \"the patch helps for reasons unrelated to the named variable\"",
        "killer_control: \"break-variable, random matched, wrong-site, no-op, and held-out split checks\"",
        "result: \"filled by the verdict table below\"",
        "claim_allowed: \"narrow residual-resampling handle, not algorithm identity\"",
        "```",
        "",
        "## Cheap explanations and controls",
        "",
        "| Cheap explanation | Control | What would make the cheap explanation win? |",
        "|---|---|---|",
        "| Any donor works because patching perturbs the stream | `random_matched` | random donors preserve about as well as preserving donors |",
        "| The site is not specific | `wrong_site_preserve` | the wrong-site preserving donor preserves about as well as the named site |",
        "| The variable is too broad | `break_variable` | donors that break the variable still preserve the behavior |",
        "| The result is a token or readout artifact | claimable-depth filter | only depth 0 or final-norm input passes |",
        "| The result is a specimen story | eval split and counterexamples | the train-selected site fails on eval rows or one row carries the effect |",
        "| The instrumentation is broken | no-op resampling | self-resampling changes the clean logits |",
        "",
        "## Verdicts",
        "",
    ]
    for row in evidence_rows:
        lines.append(
            f"- `{row['hypothesis_id']}`: `{row['claim_posture']}`. Train gap {row['train_damage_gap']} and specificity {row['train_specificity_gap']}; eval gap {row['eval_damage_gap']} and specificity {row['eval_specificity_gap']}; counterexamples {row['counterexamples']}."
        )
    lines += ["", "## Counterexamples", ""]
    if counterexamples:
        for row in counterexamples[:10]:
            lines.append(
                f"- `{row['kind']}`: `{row['item_id']}` with donor `{row['donor_id']}` at `{row['site']}` depth {row['depth']} scored {row['scrub_score']}."
            )
    else:
        lines.append("- No automatic counterexamples crossed the configured thresholds. Replicate before generalizing.")
    lines += [
        "",
        "## Allowed language",
        "",
        "- `The formal hypothesis survived or failed this residual-resampling battery on this model and dataset.`",
        "- `At site S and stream depth k, preserving variable X preserved more target margin than breaking X or using controls.`",
        "",
        "## Forbidden language",
        "",
        "- `The model implements exactly this algorithm.`",
        "- `This site is the whole circuit.`",
        "- `The abstraction works generally.`",
        "- `The proposed v2 is validated before rerunning it.`",
        "",
    ]
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Cheap explanations, controls, counterexamples, and allowed claim grammar.")


def write_run_summary(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    data_info: Mapping[str, Any],
    evidence_rows: Sequence[Mapping[str, Any]],
    counterexamples: Sequence[Mapping[str, Any]],
    refinement_rows: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Lab 26 run summary: causal abstraction and residual resampling",
        "",
        "## Run identity",
        "",
        f"- model: `{bundle.anatomy.model_id}` ({bundle.anatomy.n_layers} blocks, d_model {bundle.anatomy.d_model})",
        f"- data: `{pathlib.Path(str(data_info['data_path'])).name}` sha256 `{str(data_info['data_sha256'])[:16]}`",
        f"- selected rows: {data_info['n_rows_selected']} from {data_info['n_rows_file']}",
        f"- selected domains: `{data_info['domains_selected']}`",
        f"- selected splits: `{data_info['splits_selected']}`",
        "- intervention: residual-stream resampling at hypothesis-named positions",
        "- evidence: `FORMAL + CAUSAL + AUDIT` when gates pass",
        "",
        "## 1. What behavior was measured?",
        "",
        "The behavior is the next-token margin `logit(target) - logit(distractor)` on induction-copying and relation-answer prompts.",
        "",
        "## 2. What abstraction was proposed?",
        "",
        "The JSON specs name high-level variables, residual-stream sites, donor rules, and thresholds. Read `causal_abstraction_spec.md` before reading any plot.",
        "",
        "## 3. What intervention tested it?",
        "",
        "The run patched the recipient prompt with donor residual vectors. Donors either preserved the named variable, broke it, were random same-domain controls, or used a preserving donor at a wrong site. The no-op condition patched each prompt with its own vector.",
        "",
        "## 4. Headline verdicts",
        "",
        "| hypothesis | domain | site | depth | train pass | eval pass | posture |",
        "|---|---|---|---:|---|---|---|",
    ]
    for row in evidence_rows:
        lines.append(
            f"| `{row['hypothesis_id']}` | {row['domain']} | `{row['best_site']}` | {row['best_depth']} | {row['train_formal_pass']} | {row['eval_formal_pass']} | {row['claim_posture']} |"
        )
    lines += [
        "",
        "## 5. What should students read first?",
        "",
        "1. `method_card.md` for the verdict table.",
        "2. `causal_abstraction_spec.md` for the formal mapping.",
        "3. `tables/evidence_matrix.csv` and `tables/split_generalization_summary.csv` for claim readiness.",
        "4. `tables/counterexamples.csv` before writing positive language.",
        "5. `operationalization_audit.md` for the cheap explanations.",
        "6. `plots/causal_abstraction_dashboard.png`, then `plots/plot_reading_guide.csv`.",
        "",
        "## 6. Counterexamples and refinement",
        "",
        f"- automatic counterexamples written: {len(counterexamples)}",
        f"- refinement rows written: {len(refinement_rows)}",
        "",
        "The correct response to a failed gate is to shrink the claim, not to hide the row. The refinement log proposes smaller v2 claims, but those proposals are not validated until rerun.",
        "",
        "## 7. Caveats",
        "",
        "- This lab tests residual-stream resampling only. It is not path-specific scrubbing.",
        "- Depth 0 and the final-norm input are recorded in raw tables but excluded from formal gates.",
        "- A preserved margin is evidence for this formal mapping under this intervention, not evidence for a universal algorithm.",
        "- Low clean margins make ratios unstable. The baseline table records which items passed the gate.",
        "",
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "The standard run-summary questions answered for Lab 26.")


def write_ledger_claims(ctx: bench.RunContext, evidence_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    run_name = ctx.run_dir.name
    claims: list[dict[str, str]] = []
    for i, row in enumerate(evidence_rows, start=1):
        if row["claim_posture"] == "supported_on_train_and_eval":
            tag = "FORMAL+CAUSAL"
            text = (
                f"For `{row['domain']}` prompts in Lab 26, formal hypothesis `{row['hypothesis_id']}` survived residual resampling at `{row['best_site']}` depth {row['best_depth']}: "
                f"train preserving donors scored {row['train_preservation']} vs broken-variable {row['train_broken_variable']} "
                f"and eval preserving donors scored {row['eval_preservation']} vs broken-variable {row['eval_broken_variable']}."
            )
        elif "supported" in str(row["claim_posture"]):
            tag = "FORMAL+CAUSAL,AUDIT"
            text = (
                f"For `{row['domain']}` prompts in Lab 26, formal hypothesis `{row['hypothesis_id']}` has a supported best cell at `{row['best_site']}` depth {row['best_depth']}, "
                f"but `{row['claim_posture']}` means the claim must be narrowed before broad use."
            )
        else:
            tag = "FORMAL+CAUSAL,AUDIT"
            text = (
                f"For `{row['domain']}` prompts in Lab 26, formal hypothesis `{row['hypothesis_id']}` did not earn a positive abstraction claim under the current residual-resampling gates "
                f"(train damage gap {row['train_damage_gap']}, train specificity gap {row['train_specificity_gap']})."
            )
        claims.append({
            "id": f"{LAB_ID}-C{i}",
            "tag": tag,
            "text": text,
            "artifact": f"runs/{run_name}/tables/evidence_matrix.csv",
            "falsifier": "A held-out run where preserving donors no longer beat broken-variable and wrong-site/random controls, or a row where the same preservation appears with the named variable broken.",
        })
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)
    return claims


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------



def plot_blank(ctx: bench.RunContext, filename: str, title: str, message: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.axis("off")
    ax.text(0.5, 0.55, message, ha="center", va="center", fontsize=11)
    ax.text(0.5, 0.18, ctx.plot_footer(), ha="center", va="center", fontsize=7, color="#555555")
    fig.tight_layout()
    bench.save_figure(ctx, fig, filename, title)


def selected_cell_raw_rows(
    resampling_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Raw scrub-score rows at each hypothesis' selected site/depth.

    These rows are the source table for the control-comparison plots. Keeping
    them separate from the long intervention table makes the figure auditable
    after the PNG leaves the run directory.
    """
    selected = {
        str(row["hypothesis_id"]): {
            "site": str(row["best_site"]),
            "depth": int(row["best_depth"]),
            "selection_split": str(row.get("selection_split") or "all"),
            "domain": str(row.get("domain", "")),
            "claim_posture": str(row.get("claim_posture", "")),
        }
        for row in evidence_rows
    }
    out: list[dict[str, Any]] = []
    for row in resampling_rows:
        hyp = str(row.get("hypothesis_id", ""))
        sel = selected.get(hyp)
        if not sel:
            continue
        if row.get("error") or not boolish(row.get("claimable_depth")) or not boolish(row.get("baseline_pass")):
            continue
        if str(row.get("site")) != sel["site"] or int(row.get("depth", -1)) != sel["depth"]:
            continue
        score = as_float(row.get("scrub_score"))
        if not math.isfinite(score):
            continue
        out.append({
            "selected_point_id": stable_id("selected", row.get("intervention_id", "")),
            "intervention_id": row.get("intervention_id", ""),
            "hypothesis_id": hyp,
            "domain": sel["domain"] or row.get("domain", ""),
            "site": sel["site"],
            "depth": sel["depth"],
            "selection_split": sel["selection_split"],
            "claim_posture": sel["claim_posture"],
            "split_group": row.get("split", ""),
            "condition": row.get("condition", ""),
            "condition_label": condition_label(str(row.get("condition", ""))),
            "item_id": row.get("item_id", ""),
            "donor_id": row.get("donor_id", ""),
            "scrub_score": score,
            "clean_diff": as_float(row.get("clean_diff")),
            "patched_diff": as_float(row.get("patched_diff")),
            "delta_from_clean": as_float(row.get("delta_from_clean")),
            "target_prompt": row.get("target_prompt", ""),
            "donor_prompt": row.get("donor_prompt", ""),
            "preserves_variables": row.get("preserves_variables", ""),
            "breaks_variables": row.get("breaks_variables", ""),
            "same_variables": row.get("same_variables", ""),
            "different_variables": row.get("different_variables", ""),
        })
    return out


def selected_condition_summary_rows(
    selected_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected_meta = {str(row["hypothesis_id"]): row for row in evidence_rows}
    summary_lookup = {
        (str(row["hypothesis_id"]), str(row["site"]), int(row["depth"]), str(row["split_group"])): row
        for row in summary_rows
    }
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in selected_rows:
        # Add both the actual split and an all-row rollup. This lets the split
        # plot and the selected-control plot cite the same raw table.
        for split in (str(row.get("split_group", "")), "all"):
            groups[(str(row["hypothesis_id"]), split, str(row["condition"]))].append(as_float(row.get("scrub_score")))

    out: list[dict[str, Any]] = []
    for hyp in sorted(selected_meta):
        ev = selected_meta[hyp]
        site = str(ev["best_site"])
        depth = int(ev["best_depth"])
        selection_split = str(ev.get("selection_split") or "all")
        for split in [selection_split, "train", "eval", "all"]:
            for cond in CONDITION_ORDER:
                stats = mean_ci(groups.get((hyp, split, cond), []))
                if stats["n"] == 0 and split == selection_split:
                    # Keep an explicit empty row for selected split so a missing
                    # control is visible in the source table and manifest.
                    pass
                elif stats["n"] == 0:
                    continue
                sr = summary_lookup.get((hyp, site, depth, split), {})
                out.append({
                    "hypothesis_id": hyp,
                    "domain": ev.get("domain", ""),
                    "site": site,
                    "depth": depth,
                    "selection_split": selection_split,
                    "split_group": split,
                    "condition": cond,
                    "condition_label": condition_label(cond),
                    "n": stats["n"],
                    "mean_scrub_score": stats["mean"],
                    "stdev_scrub_score": stats["stdev"],
                    "stderr_scrub_score": stats["stderr"],
                    "ci_low_scrub_score": stats["ci_low"],
                    "ci_high_scrub_score": stats["ci_high"],
                    "min_scrub_score": stats["min"],
                    "max_scrub_score": stats["max"],
                    "formal_pass": sr.get("formal_pass", ""),
                    "pass_preservation": sr.get("pass_preservation", ""),
                    "pass_damage": sr.get("pass_damage", ""),
                    "pass_specificity": sr.get("pass_specificity", ""),
                    "predicted_preservation_min": sr.get("predicted_preservation_min", ""),
                    "predicted_damage_when_broken_min": sr.get("predicted_damage_when_broken_min", ""),
                    "predicted_specificity_gap_min": sr.get("predicted_specificity_gap_min", ""),
                })
    return out


def baseline_plot_rows(baseline_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in baseline_rows:
        clean = as_float(row.get("clean_diff"))
        out.append({
            "item_id": row.get("item_id", ""),
            "domain": row.get("domain", ""),
            "split_group": row.get("split", ""),
            "clean_diff": clean,
            "baseline_pass": boolish(row.get("baseline_pass")),
            "target_rank": row.get("target_rank", ""),
            "distractor_rank": row.get("distractor_rank", ""),
            "top_token": row.get("top_token", ""),
            "target": row.get("target", ""),
            "distractor": row.get("distractor", ""),
            "low_margin_warning": math.isfinite(clean) and clean <= MIN_BASELINE_MARGIN,
        })
    return out


def matrix_source_rows(summary_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in summary_rows:
        if str(row.get("split_group")) != "all":
            continue
        label = f"{row.get('hypothesis_id')} | {row.get('site')} | d{row.get('depth')}"
        for cond, col in [
            ("no_op_same_example", "mean_noop"),
            ("preserve_variable", "mean_preserve_variable"),
            ("break_variable", "mean_break_variable"),
            ("random_matched", "mean_random_matched"),
            ("wrong_site_preserve", "mean_wrong_site_preserve"),
        ]:
            out.append({
                "matrix_cell_id": stable_id("matrix", row.get("hypothesis_id"), row.get("site"), row.get("depth"), cond),
                "row_label": label,
                "hypothesis_id": row.get("hypothesis_id", ""),
                "domain": row.get("domain", ""),
                "site": row.get("site", ""),
                "depth": row.get("depth", ""),
                "condition": cond,
                "condition_label": condition_label(cond),
                "mean_scrub_score": row.get(col, ""),
                "formal_pass": row.get("formal_pass", ""),
                "pass_preservation": row.get("pass_preservation", ""),
                "pass_damage": row.get("pass_damage", ""),
                "pass_specificity": row.get("pass_specificity", ""),
                "n_preserve": row.get("n_preserve", ""),
                "n_break": row.get("n_break", ""),
                "n_random": row.get("n_random", ""),
                "n_wrong_site": row.get("n_wrong_site", ""),
            })
    return out


def pass_fail_source_rows(summary_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary_rows:
        label = f"{row.get('hypothesis_id')} | {row.get('site')} | d{row.get('depth')} | {row.get('split_group')}"
        for gate in ("pass_preservation", "pass_damage", "pass_specificity", "formal_pass"):
            rows.append({
                "gate_cell_id": stable_id("gate", row.get("hypothesis_id"), row.get("site"), row.get("depth"), row.get("split_group"), gate),
                "row_label": label,
                "hypothesis_id": row.get("hypothesis_id", ""),
                "domain": row.get("domain", ""),
                "site": row.get("site", ""),
                "depth": row.get("depth", ""),
                "split_group": row.get("split_group", ""),
                "gate": gate,
                "passed": boolish(row.get(gate)),
                "preserve_mean": row.get("mean_preserve_variable", ""),
                "break_mean": row.get("mean_break_variable", ""),
                "control_floor": row.get("control_floor", ""),
                "damage_gap": row.get("damage_gap", ""),
                "specificity_gap": row.get("specificity_gap", ""),
            })
    return rows


def counterexample_kind_rows(counterexamples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in counterexamples:
        grouped[(str(row.get("hypothesis_id", "")), str(row.get("kind", "")))].append(as_float(row.get("severity"), 0.0))
    out: list[dict[str, Any]] = []
    for (hyp, kind), vals in sorted(grouped.items()):
        stats = mean_ci(vals)
        out.append({
            "hypothesis_id": hyp,
            "kind": kind,
            "n": stats["n"],
            "mean_severity": stats["mean"],
            "max_severity": stats["max"],
        })
    if not out:
        out.append({"hypothesis_id": "all", "kind": "none", "n": 0, "mean_severity": "", "max_severity": ""})
    return out


def write_failure_specimens(
    ctx: bench.RunContext, counterexamples: Sequence[Mapping[str, Any]]
) -> tuple[pathlib.Path, pathlib.Path]:
    jsonl_path = ctx.path("tables", "failure_specimens.jsonl")
    write_jsonl(jsonl_path, [{"specimen_rank": i + 1, **dict(row)} for i, row in enumerate(counterexamples)])
    ctx.register_artifact(jsonl_path, "table", "JSONL specimens for failed, leaky, or contradictory Lab 26 examples.")

    lines = [
        "# Lab 26 failure specimens",
        "",
        "These are not bloopers. They are the rows that keep the claim honest.",
        "",
    ]
    if not counterexamples:
        lines.append("No automatic failure specimens crossed the configured thresholds. This is not a proof; rerun on Tier B/full before broadening the claim.")
    else:
        for i, row in enumerate(counterexamples[:12], start=1):
            lines += [
                f"## {i}. `{row.get('kind')}` for `{row.get('hypothesis_id')}`",
                "",
                f"- item: `{row.get('item_id')}`; donor: `{row.get('donor_id')}`; split: `{row.get('split')}`",
                f"- site/depth: `{row.get('site')}` depth `{row.get('depth')}`; condition: `{row.get('condition')}`",
                f"- score: `{row.get('scrub_score')}`; clean diff: `{row.get('clean_diff')}`; patched diff: `{row.get('patched_diff')}`; severity: `{row.get('severity')}`",
                f"- target variables: `{row.get('target_variables')}`",
                f"- donor variables: `{row.get('donor_variables')}`",
                "",
                "Target prompt:",
                "",
                "```text",
                str(row.get("target_prompt", "")),
                "```",
                "",
                "Donor prompt:",
                "",
                "```text",
                str(row.get("donor_prompt", "")),
                "```",
                "",
            ]
    md_path = ctx.path("tables", "failure_specimens.md")
    bench.write_text(md_path, "\n".join(lines))
    ctx.register_artifact(md_path, "summary", "Human-readable failure specimens for Lab 26 plots and writeups.")
    return jsonl_path, md_path


def write_figure_source_tables(
    ctx: bench.RunContext,
    evidence_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
    counterexamples: Sequence[Mapping[str, Any]],
    resampling_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    selected_rows = selected_cell_raw_rows(resampling_rows, evidence_rows)
    selected_summary = selected_condition_summary_rows(selected_rows, evidence_rows, summary_rows)
    baseline_source = baseline_plot_rows(baseline_rows)
    matrix_source = matrix_source_rows(summary_rows)
    passfail_source = pass_fail_source_rows(summary_rows)
    counter_kind_source = counterexample_kind_rows(counterexamples)
    dashboard_source = [
        {
            "hypothesis_id": row.get("hypothesis_id", ""),
            "domain": row.get("domain", ""),
            "best_site": row.get("best_site", ""),
            "best_depth": row.get("best_depth", ""),
            "selection_split": row.get("selection_split", ""),
            "claim_posture": row.get("claim_posture", ""),
            "train_preservation": row.get("train_preservation", ""),
            "train_broken_variable": row.get("train_broken_variable", ""),
            "train_damage_gap": row.get("train_damage_gap", ""),
            "train_specificity_gap": row.get("train_specificity_gap", ""),
            "eval_preservation": row.get("eval_preservation", ""),
            "eval_broken_variable": row.get("eval_broken_variable", ""),
            "eval_damage_gap": row.get("eval_damage_gap", ""),
            "eval_specificity_gap": row.get("eval_specificity_gap", ""),
            "counterexamples": row.get("counterexamples", ""),
            "baseline_pass_rate_all": row.get("baseline_pass_rate_all", ""),
            "baseline_pass_rate_train": row.get("baseline_pass_rate_train", ""),
            "baseline_pass_rate_eval": row.get("baseline_pass_rate_eval", ""),
        }
        for row in evidence_rows
    ]

    sources: dict[str, dict[str, Any]] = {}
    specs = [
        ("dashboard_evidence.csv", dashboard_source, "Source rows for causal_abstraction_dashboard.png."),
        ("selected_cell_condition_points.csv", selected_rows, "Raw per-example selected-cell scrub scores for target/control plots."),
        ("selected_cell_condition_summary.csv", selected_summary, "Mean, uncertainty, and n for selected-cell condition comparisons."),
        ("baseline_margins.csv", baseline_source, "Baseline clean margins used by dashboard and baseline-health panels."),
        ("resampling_matrix_source.csv", matrix_source, "Heatmap source table for resampling_preservation_matrix.png."),
        ("pass_fail_atlas_source.csv", passfail_source, "Gate pass/fail source table for hypothesis_pass_fail_atlas.png."),
        ("split_generalization_source.csv", list(split_rows), "Train/eval/all source table for split_generalization_ladder.png."),
        ("counterexample_kind_summary.csv", counter_kind_source, "Counterexample kind counts for dashboard and gallery."),
    ]
    for filename, rows, description in specs:
        path = write_plot_source(ctx, filename, rows, description)
        sources[filename] = {"path": str(path.relative_to(ctx.run_dir)), "row_count": len(rows)}
    failure_jsonl, failure_md = write_failure_specimens(ctx, counterexamples)
    sources["failure_specimens.jsonl"] = {"path": str(failure_jsonl.relative_to(ctx.run_dir)), "row_count": len(counterexamples)}
    sources["failure_specimens.md"] = {"path": str(failure_md.relative_to(ctx.run_dir)), "row_count": len(counterexamples)}
    return sources


def yerr_from_ci(row: Mapping[str, Any], mean_key: str = "mean_scrub_score") -> tuple[float, float]:
    mean = as_float(row.get(mean_key))
    lo = as_float(row.get("ci_low_scrub_score"))
    hi = as_float(row.get("ci_high_scrub_score"))
    if math.isfinite(mean) and math.isfinite(lo) and math.isfinite(hi):
        return max(0.0, mean - lo), max(0.0, hi - mean)
    return 0.0, 0.0


def plot_dashboard(
    ctx: bench.RunContext,
    evidence_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    condition_summary: Sequence[Mapping[str, Any]],
    counter_kind: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if not evidence_rows:
        plot_blank(ctx, "causal_abstraction_dashboard.png", "Lab 26 dashboard", "No evidence rows were available after gates and aggregation.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.8))
    fig.suptitle("Lab 26 causal abstraction dashboard: preservation must beat break-variable and controls", fontsize=13, fontweight="bold")

    # Panel 1: selected-cell condition means on the selection split.
    labels = [f"{row['domain']}\n{row['hypothesis_id']}" for row in evidence_rows]
    x = np.arange(len(labels), dtype=float)
    conds = ["preserve_variable", "break_variable", "random_matched", "wrong_site_preserve"]
    width = 0.18
    selected_split = {str(row["hypothesis_id"]): str(row.get("selection_split") or "all") for row in evidence_rows}
    summary_lookup = {
        (str(row.get("hypothesis_id")), str(row.get("split_group")), str(row.get("condition"))): row
        for row in condition_summary
    }
    for j, cond in enumerate(conds):
        means, err_low, err_high = [], [], []
        for ev in evidence_rows:
            row = summary_lookup.get((str(ev["hypothesis_id"]), selected_split[str(ev["hypothesis_id"])], cond), {})
            mean = as_float(row.get("mean_scrub_score"), float("nan"))
            lo, hi = yerr_from_ci(row)
            means.append(mean if math.isfinite(mean) else 0.0)
            err_low.append(lo)
            err_high.append(hi)
        axes[0, 0].bar(x + (j - 1.5) * width, means, width, yerr=[err_low, err_high], capsize=2, label=condition_label(cond))
    axes[0, 0].axhline(1.0, linestyle=":", linewidth=1)
    axes[0, 0].axhline(0.0, linewidth=0.8)
    axes[0, 0].set_xticks(x, labels, fontsize=8)
    axes[0, 0].set_ylabel("scrub score = patched / clean margin")
    axes[0, 0].set_title("Selected cell, selected split (95% CI when n>1)")
    axes[0, 0].legend(fontsize=7, frameon=False)

    # Panel 2: train/eval/all preservation for the same selected cell.
    splits = ["train", "eval", "all"]
    width2 = 0.22
    for j, split in enumerate(splits):
        vals = []
        for ev in evidence_rows:
            row = summary_lookup.get((str(ev["hypothesis_id"]), split, "preserve_variable"), {})
            val = as_float(row.get("mean_scrub_score"), float("nan"))
            vals.append(val if math.isfinite(val) else 0.0)
        axes[0, 1].bar(x + (j - 1) * width2, vals, width2, label=split)
    axes[0, 1].axhline(1.0, linestyle=":", linewidth=1)
    axes[0, 1].set_xticks(x, labels, fontsize=8)
    axes[0, 1].set_title("Does the train-selected cell survive eval?")
    axes[0, 1].set_ylabel("preserve-variable mean score")
    axes[0, 1].legend(fontsize=8, frameon=False)

    # Panel 3: baseline health, with all raw clean margins visible.
    domains = sorted({str(row.get("domain")) for row in baseline_rows}) or sorted({str(row.get("domain")) for row in evidence_rows})
    by_domain: dict[str, list[float]] = defaultdict(list)
    for row in baseline_rows:
        val = as_float(row.get("clean_diff"))
        if math.isfinite(val):
            by_domain[str(row.get("domain"))].append(val)
    for i, domain in enumerate(domains):
        vals = by_domain.get(domain, [])
        if vals:
            axes[1, 0].boxplot([vals], positions=[i], widths=0.45, showfliers=False)
            jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.array([0.0])
            axes[1, 0].scatter(np.full(len(vals), i) + jitter, vals, s=18, alpha=0.75)
    axes[1, 0].axhline(MIN_BASELINE_MARGIN, linestyle="--", linewidth=1, label=f"baseline gate {MIN_BASELINE_MARGIN}")
    axes[1, 0].set_xticks(range(len(domains)), domains, fontsize=8)
    axes[1, 0].set_title("Baseline target-vs-distractor margins")
    axes[1, 0].set_ylabel("clean logit diff")
    axes[1, 0].legend(fontsize=8, frameon=False)

    # Panel 4: posture and counterexample load.
    axes[1, 1].axis("off")
    posture_lines = []
    for row in evidence_rows:
        posture_lines.append(
            f"{row['hypothesis_id']}: {row['claim_posture']}\n"
            f"  train gap={row.get('train_damage_gap')} spec={row.get('train_specificity_gap')} | "
            f"eval gap={row.get('eval_damage_gap')} spec={row.get('eval_specificity_gap')} | "
            f"counterexamples={row.get('counterexamples')}"
        )
    kind_lines = [f"{row.get('kind')}: n={row.get('n')}" for row in counter_kind if row.get("kind") != "none"]
    if not kind_lines:
        kind_lines = ["no automatic counterexamples crossed thresholds"]
    axes[1, 1].text(0.0, 1.0, "Claim posture\n" + "\n".join(posture_lines), va="top", ha="left", fontsize=8)
    axes[1, 1].text(0.0, 0.28, "Counterexample types\n" + "\n".join(kind_lines), va="top", ha="left", fontsize=8)
    axes[1, 1].text(0.0, 0.03, ctx.plot_footer(), va="bottom", ha="left", fontsize=7, color="#555555")

    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    bench.save_figure(ctx, fig, "causal_abstraction_dashboard.png", "One-screen Lab 26 evidence posture with controls, split survival, baseline health, and counterexamples.")


def plot_target_vs_control(
    ctx: bench.RunContext,
    selected_rows: Sequence[Mapping[str, Any]],
    condition_summary: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    hyps = sorted({str(row["hypothesis_id"]) for row in condition_summary})
    if not hyps:
        plot_blank(ctx, "target_vs_control.png", "Target vs control", "No selected-cell condition rows were available.")
        return
    conds = ["preserve_variable", "break_variable", "random_matched", "wrong_site_preserve"]
    selection_split = {}
    for row in condition_summary:
        selection_split.setdefault(str(row["hypothesis_id"]), str(row.get("selection_split") or "all"))
    summary_lookup = {
        (str(row.get("hypothesis_id")), str(row.get("split_group")), str(row.get("condition"))): row
        for row in condition_summary
    }

    fig, ax = plt.subplots(figsize=(max(9.5, 2.4 * len(hyps)), 5.6))
    x = np.arange(len(hyps), dtype=float)
    width = 0.18
    rng_offsets = {cond: (i - 1.5) * width for i, cond in enumerate(conds)}
    for i, cond in enumerate(conds):
        means, err_low, err_high = [], [], []
        for hyp in hyps:
            row = summary_lookup.get((hyp, selection_split[hyp], cond), {})
            mean = as_float(row.get("mean_scrub_score"), float("nan"))
            lo, hi = yerr_from_ci(row)
            means.append(mean if math.isfinite(mean) else 0.0)
            err_low.append(lo)
            err_high.append(hi)
        ax.bar(x + rng_offsets[cond], means, width, yerr=[err_low, err_high], capsize=2.5, label=condition_label(cond))

    # Raw points: deterministic tiny jitter based on row id, only on selected split.
    for row in selected_rows:
        hyp = str(row.get("hypothesis_id"))
        if hyp not in hyps or str(row.get("split_group")) != selection_split[hyp]:
            continue
        cond = str(row.get("condition"))
        if cond not in rng_offsets:
            continue
        val = as_float(row.get("scrub_score"))
        if not math.isfinite(val):
            continue
        jitter = ((stable_int(str(row.get("selected_point_id"))) % 1000) / 1000.0 - 0.5) * width * 0.45
        ax.scatter(hyps.index(hyp) + rng_offsets[cond] + jitter, val, s=18, alpha=0.68, linewidths=0.25, edgecolors="white")

    ax.axhline(1.0, linestyle=":", linewidth=1, label="clean preserved")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xticks(x, [f"{hyp}\n({selection_split[hyp]})" for hyp in hyps], fontsize=8)
    ax.set_ylabel("scrub score = patched_diff / clean_diff")
    ax.set_title("Selected-cell target versus controls: bars are means, whiskers are 95% CI, dots are examples")
    ax.legend(fontsize=8, frameon=False, ncol=3)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "target_vs_control.png", "Direct selected-cell comparison of preserving donors against break-variable and control donors.")


def plot_specificity_ladder(ctx: bench.RunContext, evidence_rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if not evidence_rows:
        plot_blank(ctx, "variable_specificity_ladder.png", "Variable specificity ladder", "No evidence rows were available.")
        return
    labels = [f"{row['hypothesis_id']}\n{row['best_site']} d{row['best_depth']}" for row in evidence_rows]
    x = np.arange(len(labels), dtype=float)
    cols = [
        ("preserve - break", "train_damage_gap"),
        ("preserve - strongest control", "train_specificity_gap"),
    ]
    width = 0.28
    fig, ax = plt.subplots(figsize=(max(8.5, 2.5 * len(labels)), 4.8))
    for i, (label, key) in enumerate(cols):
        vals = [as_float(row.get(key), 0.0) for row in evidence_rows]
        ax.bar(x + (i - 0.5) * width, vals, width, label=label)
    ax.axhline(0.0, linewidth=0.9)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("gap at selected cell")
    ax.set_title("Specificity gaps: the hypothesis pays rent only when preserve beats break and controls")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "variable_specificity_ladder.png", "Preserve-minus-break and preserve-minus-control gaps at the selected cell.")


def plot_resampling_matrix(ctx: bench.RunContext, matrix_rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if not matrix_rows:
        plot_blank(ctx, "resampling_preservation_matrix.png", "Resampling matrix", "No baseline-passing claimable resampling cells were available.")
        return
    row_labels = []
    for row in matrix_rows:
        label = str(row["row_label"])
        if label not in row_labels:
            row_labels.append(label)
    conds = list(CONDITION_ORDER)
    mat = np.full((len(row_labels), len(conds)), np.nan)
    formal = {label: False for label in row_labels}
    for row in matrix_rows:
        i = row_labels.index(str(row["row_label"]))
        j = conds.index(str(row["condition"]))
        mat[i, j] = as_float(row.get("mean_scrub_score"), float("nan"))
        formal[str(row["row_label"])] = formal[str(row["row_label"])] or boolish(row.get("formal_pass"))
    ylabels = [("★ " if formal[label] else "  ") + label for label in row_labels]
    fig, ax = plt.subplots(figsize=(11.5, max(4.0, 0.32 * len(row_labels) + 1.4)))
    im = ax.imshow(mat, aspect="auto", vmin=-0.5, vmax=1.5)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=7)
    ax.set_xticks(range(len(conds)), [condition_label(c) for c in conds], rotation=28, ha="right")
    ax.set_title("Resampling preservation matrix (all split): ★ marks cells passing all formal gates")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if math.isfinite(mat[i, j]) and len(row_labels) <= 42:
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, shrink=0.85, label="mean scrub score")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "resampling_preservation_matrix.png", "Heatmap of all-split resampling preservation by site, depth, and condition.")


def plot_pass_fail_atlas(ctx: bench.RunContext, passfail_rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if not passfail_rows:
        plot_blank(ctx, "hypothesis_pass_fail_atlas.png", "Pass/fail atlas", "No formal gate rows were available.")
        return
    row_labels = []
    gates = ["pass_preservation", "pass_damage", "pass_specificity", "formal_pass"]
    for row in passfail_rows:
        label = str(row["row_label"])
        if label not in row_labels:
            row_labels.append(label)
    mat = np.zeros((len(row_labels), len(gates)))
    for row in passfail_rows:
        i = row_labels.index(str(row["row_label"]))
        j = gates.index(str(row["gate"]))
        mat[i, j] = 1.0 if boolish(row.get("passed")) else 0.0
    fig, ax = plt.subplots(figsize=(10.8, max(4.0, 0.22 * len(row_labels) + 1.4)))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=6.2)
    ax.set_xticks(range(len(gates)), [g.replace("pass_", "").replace("_", " ") for g in gates], rotation=25, ha="right")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, "✓" if mat[i, j] else "·", ha="center", va="center", fontsize=8)
    ax.set_title("Formal gate atlas by site/depth and split")
    fig.colorbar(im, ax=ax, shrink=0.85, ticks=[0, 1], label="gate passed")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "hypothesis_pass_fail_atlas.png", "Pass/fail atlas for formal hypothesis gates.")


def plot_split_generalization(ctx: bench.RunContext, split_rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    hyps = []
    for row in split_rows:
        label = f"{row.get('hypothesis_id')}\n{row.get('site')} d{row.get('depth')}"
        if label not in hyps:
            hyps.append(label)
    if not hyps:
        plot_blank(ctx, "split_generalization_ladder.png", "Split generalization", "No train/eval/all split rows were available.")
        return
    splits = ["train", "eval", "all"]
    mat = np.full((len(hyps), len(splits)), np.nan)
    pass_mat = np.zeros((len(hyps), len(splits)), dtype=bool)
    for i, label in enumerate(hyps):
        for j, split in enumerate(splits):
            vals = [
                row for row in split_rows
                if f"{row.get('hypothesis_id')}\n{row.get('site')} d{row.get('depth')}" == label and row.get("split_group") == split
            ]
            if vals:
                mat[i, j] = as_float(vals[0].get("preservation"), float("nan"))
                pass_mat[i, j] = boolish(vals[0].get("formal_pass"))
    fig, ax = plt.subplots(figsize=(7.8, max(3.4, 0.65 * len(hyps) + 1.0)))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1.5)
    ax.set_xticks(range(len(splits)), splits)
    ax.set_yticks(range(len(hyps)), hyps, fontsize=8)
    ax.set_title("Train-selected cell preservation by split (✓ = formal gates pass)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if math.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}" + (" ✓" if pass_mat[i, j] else ""), ha="center", va="center", fontsize=8)
            else:
                ax.text(j, i, "missing", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.85, label="mean preserve score")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "split_generalization_ladder.png", "Train-selected best-cell preservation on train, eval, and all rows.")


def plot_counterexamples(ctx: bench.RunContext, counterexamples: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, max(4.2, 0.55 * max(1, min(10, len(counterexamples))))))
    ax.axis("off")
    if not counterexamples:
        ax.text(0.5, 0.58, "No automatic counterexamples crossed thresholds", ha="center", va="center", fontsize=11)
        ax.text(0.5, 0.36, "Read this as a prompt to rerun wider, not as proof of the abstraction.", ha="center", va="center", fontsize=9)
    else:
        shown = list(counterexamples[:10])
        table_data = [
            [
                row.get("kind", ""),
                row.get("hypothesis_id", ""),
                row.get("split", ""),
                row.get("item_id", ""),
                row.get("donor_id", ""),
                f"{row.get('site')} d{row.get('depth')}",
                row.get("condition", ""),
                row.get("scrub_score", ""),
                row.get("severity", ""),
            ]
            for row in shown
        ]
        table = ax.table(
            cellText=table_data,
            colLabels=["kind", "hyp", "split", "item", "donor", "site/depth", "condition", "score", "severity"],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1, 1.35)
    ax.set_title("Counterexample gallery: the rows that shrink or kill the favorite claim", pad=12)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "counterexample_gallery.png", "Top automatic counterexamples for Lab 26 hypotheses.")


def write_plot_manifest(ctx: bench.RunContext, sources: Mapping[str, Mapping[str, Any]], no_plots: bool) -> None:
    entries = [
        {
            "figure_path": "plots/causal_abstraction_dashboard.png",
            "question": "Do the selected cells, baseline margins, split survival, and counterexamples tell a coherent story?",
            "source_table": sources.get("dashboard_evidence.csv", {}).get("path", ""),
            "raw_source_table": sources.get("selected_cell_condition_points.csv", {}).get("path", ""),
            "row_count": sources.get("dashboard_evidence.csv", {}).get("row_count", 0),
            "metric": "scrub_score, clean_diff, pass/fail gates",
            "controls": "break_variable, random_matched, wrong_site_preserve, no_op_same_example, eval split",
            "claim_supported": "Overall evidence posture and where to read next; not a verdict machine.",
        },
        {
            "figure_path": "plots/target_vs_control.png",
            "question": "At the selected site/depth, do preserving donors beat broken-variable and matched controls?",
            "source_table": sources.get("selected_cell_condition_summary.csv", {}).get("path", ""),
            "raw_source_table": sources.get("selected_cell_condition_points.csv", {}).get("path", ""),
            "row_count": sources.get("selected_cell_condition_points.csv", {}).get("row_count", 0),
            "metric": "scrub_score",
            "controls": "break_variable, random_matched, wrong_site_preserve",
            "claim_supported": "Variable specificity at the selected cell, including raw variation and tiny-n caveats.",
        },
        {
            "figure_path": "plots/resampling_preservation_matrix.png",
            "question": "Which site/depth/condition cells preserve the target margin across all baseline-passing examples?",
            "source_table": sources.get("resampling_matrix_source.csv", {}).get("path", ""),
            "raw_source_table": "tables/resampling_interventions.csv",
            "row_count": sources.get("resampling_matrix_source.csv", {}).get("row_count", 0),
            "metric": "mean scrub_score by condition",
            "controls": "all donor conditions shown side by side",
            "claim_supported": "Search map for candidate abstraction cells, not a split-survival claim.",
        },
        {
            "figure_path": "plots/hypothesis_pass_fail_atlas.png",
            "question": "Which formal gates pass by split, site, and depth?",
            "source_table": sources.get("pass_fail_atlas_source.csv", {}).get("path", ""),
            "raw_source_table": "tables/variable_preservation_summary.csv",
            "row_count": sources.get("pass_fail_atlas_source.csv", {}).get("row_count", 0),
            "metric": "preservation gate, damage gate, specificity gate, formal pass",
            "controls": "thresholded gates include break-variable and strongest control comparisons",
            "claim_supported": "Gate-level audit, including negative cells.",
        },
        {
            "figure_path": "plots/variable_specificity_ladder.png",
            "question": "How much room is there between preserve and the strongest alternative explanation?",
            "source_table": sources.get("dashboard_evidence.csv", {}).get("path", ""),
            "raw_source_table": "tables/evidence_matrix.csv",
            "row_count": sources.get("dashboard_evidence.csv", {}).get("row_count", 0),
            "metric": "damage_gap and specificity_gap",
            "controls": "break_variable and max(random_matched, wrong_site_preserve)",
            "claim_supported": "Compact gap reading for selected cells.",
        },
        {
            "figure_path": "plots/split_generalization_ladder.png",
            "question": "Does the train-selected best cell survive held-out eval rows?",
            "source_table": sources.get("split_generalization_source.csv", {}).get("path", ""),
            "raw_source_table": "tables/split_generalization_summary.csv",
            "row_count": sources.get("split_generalization_source.csv", {}).get("row_count", 0),
            "metric": "preserve-variable mean and formal pass by split",
            "controls": "same site/depth is reused on eval rather than reselected",
            "claim_supported": "Split-survival or train-only posture.",
        },
        {
            "figure_path": "plots/counterexample_gallery.png",
            "question": "Which individual rows most embarrass the aggregate claim?",
            "source_table": sources.get("failure_specimens.jsonl", {}).get("path", ""),
            "raw_source_table": "tables/counterexamples.csv",
            "row_count": sources.get("failure_specimens.jsonl", {}).get("row_count", 0),
            "metric": "severity and scrub_score",
            "controls": "negative results and leaky controls are surfaced, not hidden",
            "claim_supported": "Counterexample-driven caveats and v2 refinement needs.",
        },
    ]
    for entry in entries:
        entry["created"] = not no_plots
        entry["model"] = ctx.model_id
        entry["tier"] = ctx.args.tier
        entry["prompt_set"] = ctx.args.prompt_set
        entry["seed"] = ctx.args.seed
        entry["run_name"] = ctx.run_dir.name
    json_path = ctx.path("plots", "plot_manifest.json")
    bench.write_json(json_path, entries)
    ctx.register_artifact(json_path, "manifest", "Figure manifest with source tables, row counts, metrics, controls, and claim boundaries.")
    csv_path = ctx.path("plots", "plot_manifest.csv")
    bench.write_csv(csv_path, entries)
    ctx.register_artifact(csv_path, "manifest", "CSV copy of the Lab 26 figure manifest.")


def write_plots(
    ctx: bench.RunContext,
    evidence_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
    counterexamples: Sequence[Mapping[str, Any]],
    resampling_rows: Sequence[Mapping[str, Any]],
) -> None:
    write_plot_reading_guide(ctx)
    sources = write_figure_source_tables(ctx, evidence_rows, baseline_rows, summary_rows, split_rows, counterexamples, resampling_rows)
    selected_rows = selected_cell_raw_rows(resampling_rows, evidence_rows)
    condition_summary = selected_condition_summary_rows(selected_rows, evidence_rows, summary_rows)
    matrix_rows = matrix_source_rows(summary_rows)
    passfail_rows = pass_fail_source_rows(summary_rows)
    counter_kind = counterexample_kind_rows(counterexamples)
    write_plot_manifest(ctx, sources, ctx.args.no_plots)
    if ctx.args.no_plots:
        return
    plot_dashboard(ctx, evidence_rows, baseline_rows, condition_summary, counter_kind)
    plot_target_vs_control(ctx, selected_rows, condition_summary)
    plot_resampling_matrix(ctx, matrix_rows)
    plot_pass_fail_atlas(ctx, passfail_rows)
    plot_specificity_ladder(ctx, evidence_rows)
    plot_split_generalization(ctx, split_rows)
    plot_counterexamples(ctx, counterexamples)


def write_warning_summary(
    ctx: bench.RunContext,
    token_rows: Sequence[Mapping[str, Any]],
    donor_coverage: Sequence[Mapping[str, Any]],
    resampling_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    data_info: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Write explicit warnings for skipped rows, weak baselines, and plot caveats."""
    rows: list[dict[str, Any]] = []

    def add(category: str, severity: str, count: int, detail: str, inspect: str) -> None:
        rows.append({
            "category": category,
            "severity": severity,
            "count": count,
            "detail": detail,
            "inspect": inspect,
        })

    token_dropped = [row for row in token_rows if row.get("kept") is False]
    token_warned = [row for row in token_rows if row.get("warnings")]
    donor_missing_preserve = [row for row in donor_coverage if not boolish(row.get("has_preserve"))]
    donor_missing_break = [row for row in donor_coverage if not boolish(row.get("has_break"))]
    low_baseline = [row for row in baseline_rows if not boolish(row.get("baseline_pass"))]
    error_rows = [row for row in resampling_rows if row.get("error")]
    tiny_conditions = []
    grouped: dict[tuple[str, str, int, str], int] = defaultdict(int)
    for row in resampling_rows:
        if row.get("error") or not boolish(row.get("baseline_pass")) or not boolish(row.get("claimable_depth")):
            continue
        grouped[(str(row.get("hypothesis_id")), str(row.get("site")), int(row.get("depth", -1)), str(row.get("condition")))] += 1
    for key, n in grouped.items():
        if key[3] in {"preserve_variable", "break_variable"} and n < 3:
            tiny_conditions.append((key, n))

    add(
        "data_source",
        "warning" if not bool(data_info.get("science_ready_data")) else "info",
        0 if data_info.get("science_ready_data") else 1,
        "Tier A fallback data is smoke-only." if not data_info.get("science_ready_data") else "Frozen CSV data was used.",
        "diagnostics/data_manifest.json",
    )
    add("tokenization_dropped", "error" if token_dropped else "info", len(token_dropped), "Rows dropped by single-token or position gates.", "diagnostics/tokenization_gate.csv")
    add("tokenization_warnings", "warning" if token_warned else "info", len(token_warned), "Rows with corrected or suspicious token positions.", "diagnostics/tokenization_gate.csv")
    add("donor_missing_preserve", "warning" if donor_missing_preserve else "info", len(donor_missing_preserve), "Items without preserving donors.", "diagnostics/donor_coverage.csv")
    add("donor_missing_break", "warning" if donor_missing_break else "info", len(donor_missing_break), "Items without break-variable donors.", "diagnostics/donor_coverage.csv")
    add("low_baseline_margin", "warning" if low_baseline else "info", len(low_baseline), f"Rows at or below clean-margin gate {MIN_BASELINE_MARGIN}.", "tables/baseline_behavior.csv")
    add("intervention_errors", "warning" if error_rows else "info", len(error_rows), "Intervention cells skipped or errored, usually donor length or position mismatch.", "tables/resampling_interventions.csv")
    add("tiny_condition_cells", "warning" if tiny_conditions else "info", len(tiny_conditions), "Selected/claimable condition cells with n<3; read CI and raw points before claiming stability.", "tables/figure_sources/selected_cell_condition_summary.csv")

    csv_path = ctx.path("diagnostics", "warning_summary.csv")
    bench.write_csv_with_context(ctx, csv_path, rows)
    ctx.register_artifact(csv_path, "diagnostic", "Warnings for skipped rows, weak baselines, donor coverage, and tiny plot cells.")
    json_path = ctx.path("diagnostics", "warning_summary.json")
    bench.write_json(json_path, rows)
    ctx.register_artifact(json_path, "diagnostic", "JSON copy of Lab 26 warning summary.")
    return rows


def write_lab26_run_config_snapshot(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    data_info: Mapping[str, Any],
    specs: Sequence[HypothesisSpec],
    items: Sequence[CausalItem],
) -> dict[str, Any]:
    sites: list[dict[str, Any]] = []
    for spec in specs:
        for site in iter_residual_sites(spec, bundle, ctx.args):
            sites.append({
                "hypothesis_id": site.hypothesis_id,
                "domain": site.domain,
                "site_label": site.site_label,
                "position_name": site.position_name,
                "depths": list(site.depths),
                "claimable_depths": [d for d in site.depths if depth_claimable(bundle, d)],
                "rationale": site.rationale,
            })
    snapshot = {
        "lab_id": LAB_ID,
        "lab_name": LAB_NAME,
        "run_name": ctx.run_dir.name,
        "model_id": bundle.anatomy.model_id,
        "model_revision": bundle.anatomy.revision,
        "n_layers": bundle.anatomy.n_layers,
        "d_model": bundle.anatomy.d_model,
        "tier": ctx.args.tier,
        "seed": ctx.args.seed,
        "prompt_set": ctx.args.prompt_set,
        "max_examples": ctx.args.max_examples,
        "dtype": ctx.args.dtype,
        "quantization": ctx.args.quantization,
        "decoding": "none; forward-pass-only residual resampling",
        "data": dict(data_info),
        "thresholds": {
            "min_baseline_margin": MIN_BASELINE_MARGIN,
            "min_donors_per_condition": MIN_DONORS_PER_CONDITION,
            "noop_score_atol": NOOP_SCORE_ATOL,
            "max_counterexamples": MAX_COUNTEREXAMPLES,
        },
        "conditions": list(CONDITION_ORDER),
        "control_conditions": list(CONTROL_CONDITIONS),
        "specs": [
            {
                "hypothesis_id": spec.hypothesis_id,
                "domain": spec.domain,
                "source": spec.source,
                "path": str(spec.path),
                "high_level_variables": list(spec.high_level_variables),
                "predicted_preservation_min": spec.predicted_preservation_min,
                "predicted_damage_when_broken_min": spec.predicted_damage_when_broken_min,
                "predicted_specificity_gap_min": spec.predicted_specificity_gap_min,
            }
            for spec in specs
        ],
        "residual_sites": sites,
        "selected_item_ids": [item.item_id for item in items],
        "selected_splits": {split: sum(1 for item in items if split_group_for(item) == split) for split in sorted({split_group_for(item) for item in items})},
    }
    path = ctx.path("diagnostics", "lab26_run_config_snapshot.json")
    bench.write_json(path, snapshot)
    ctx.register_artifact(path, "diagnostic", "Lab-specific config snapshot for data, specs, sites, depths, thresholds, and controls.")
    return snapshot


# ---------------------------------------------------------------------------
# Run entry point
# ---------------------------------------------------------------------------


def write_self_check_status(
    ctx: bench.RunContext,
    token_rows: Sequence[Mapping[str, Any]],
    donor_coverage: Sequence[Mapping[str, Any]],
    noop_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    status = {
        "tokenization_kept": sum(1 for row in token_rows if row.get("kept") is True),
        "tokenization_dropped": sum(1 for row in token_rows if row.get("kept") is False),
        "donor_rows_missing_preserve": sum(1 for row in donor_coverage if not row.get("has_preserve")),
        "donor_rows_missing_break": sum(1 for row in donor_coverage if not row.get("has_break")),
        "noop_max_abs_delta": max([as_float(row.get("max_abs_delta_from_clean"), 0.0) for row in noop_rows] or [0.0]),
        "noop_max_score_error": max([as_float(row.get("max_noop_score_error"), 0.0) for row in noop_rows] or [0.0]),
        "noop_score_atol": NOOP_SCORE_ATOL,
        "ok": True,
    }
    status["ok"] = bool(status["tokenization_kept"] > 0 and status["noop_max_score_error"] <= NOOP_SCORE_ATOL)
    path = ctx.path("diagnostics", "self_check_status.json")
    bench.write_json(path, status)
    ctx.register_artifact(path, "diagnostic", "Lab 26 tokenization, donor coverage, and no-op self-check summary.")
    return status


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    items, data_info = load_items(ctx)
    specs, spec_audit_rows, _spec_payloads = load_specs(ctx)
    print(
        f"[lab26] loaded {data_info['n_rows_selected']}/{data_info['n_rows_file']} items "
        f"from {pathlib.Path(data_info['data_path']).name}"
    )

    manifest_path = ctx.path("diagnostics", "data_manifest.json")
    bench.write_json(manifest_path, data_info)
    ctx.register_artifact(manifest_path, "diagnostic", "Lab 26 data file hash, selected counts, and prompt-set caps.")

    if not items:
        raise RuntimeError("Lab 26 selected zero items.")
    bench.run_hook_parity_check(ctx, bundle, items[0].prompt)
    first_capture = bench.run_with_residual_cache(bundle, items[0].prompt)
    bench.run_lens_self_check(ctx, bundle, first_capture)
    bench.run_patch_noop_check(ctx, bundle, items[0].prompt)

    items, token_rows = tokenization_gate(ctx, bundle, items)
    write_lab26_run_config_snapshot(ctx, bundle, data_info, specs, items)
    captures, baseline_rows = cache_items(ctx, bundle, items)
    donor_plans, donor_rows, donor_coverage = build_donor_plans(items, specs)

    donor_path = ctx.path("tables", "donor_plan.csv")
    bench.write_csv_with_context(ctx, donor_path, donor_rows)
    ctx.register_artifact(donor_path, "table", "Selected preserving, broken-variable, random, wrong-site, and no-op donors.")

    coverage_path = ctx.path("diagnostics", "donor_coverage.csv")
    bench.write_csv_with_context(ctx, coverage_path, donor_coverage)
    ctx.register_artifact(coverage_path, "diagnostic", "Per-item donor availability for every Lab 26 condition.")

    resampling_rows = run_resampling(ctx, bundle, items, specs, captures, donor_plans)
    noop_rows = assert_noop_identity(ctx, resampling_rows)
    self_check_status = write_self_check_status(ctx, token_rows, donor_coverage, noop_rows)
    warning_rows = write_warning_summary(ctx, token_rows, donor_coverage, resampling_rows, baseline_rows, data_info)

    results_path = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results_path, resampling_rows)
    ctx.register_artifact(results_path, "table", "Long-form Lab 26 residual-resampling interventions.")

    jsonl_path = ctx.path("results.jsonl")
    write_jsonl(jsonl_path, [{**ctx.table_context(), **row} for row in resampling_rows])
    ctx.register_artifact(jsonl_path, "table", "JSONL copy of every Lab 26 residual-resampling intervention.")

    interventions_path = ctx.path("tables", "resampling_interventions.csv")
    bench.write_csv_with_context(ctx, interventions_path, resampling_rows)
    ctx.register_artifact(interventions_path, "table", "Copy of long-form interventions under tables/ for notebooks and reports.")

    summary_rows = aggregate_resampling(resampling_rows, specs)
    summary_path = ctx.path("tables", "variable_preservation_summary.csv")
    bench.write_csv_with_context(ctx, summary_path, summary_rows)
    ctx.register_artifact(summary_path, "table", "Mean scrub scores and pass/fail gates by hypothesis, split, site, depth, and condition.")

    best_rows = select_best_cells(summary_rows)
    best_path = ctx.path("tables", "best_hypothesis_cells.csv")
    bench.write_csv_with_context(ctx, best_path, best_rows)
    ctx.register_artifact(best_path, "table", "Best resampling cell per hypothesis, selected on train when available.")

    counterexamples = build_counterexamples(resampling_rows, specs, best_rows)
    counter_path = ctx.path("tables", "counterexamples.csv")
    bench.write_csv_with_context(ctx, counter_path, counterexamples)
    ctx.register_artifact(counter_path, "table", "Automatic counterexamples that shrink or kill positive abstraction claims.")

    evidence_rows, split_rows, verdict_metrics = build_evidence_matrix(best_rows, summary_rows, baseline_rows, counterexamples)
    evidence_path = ctx.path("tables", "evidence_matrix.csv")
    bench.write_csv_with_context(ctx, evidence_path, evidence_rows)
    ctx.register_artifact(evidence_path, "table", "Compact Lab 26 evidence matrix for claim writing.")

    split_path = ctx.path("tables", "split_generalization_summary.csv")
    bench.write_csv_with_context(ctx, split_path, split_rows)
    ctx.register_artifact(split_path, "table", "Train-selected best-cell metrics on train, eval, and all rows.")

    refinement_rows = build_refinement_log(evidence_rows, counterexamples)
    refinement_path = ctx.path("tables", "hypothesis_refinement_log.csv")
    bench.write_csv_with_context(ctx, refinement_path, refinement_rows)
    ctx.register_artifact(refinement_path, "table", "Suggested v2 hypothesis refinements driven by failed gates and counterexamples.")

    metrics = {
        "lab_id": LAB_ID,
        "lab_name": LAB_NAME,
        "data": data_info,
        "self_check_status": self_check_status,
        "warning_summary": warning_rows,
        "n_interventions": len(resampling_rows),
        "n_summary_cells": len(summary_rows),
        "n_best_cells": len(best_rows),
        "n_counterexamples": len(counterexamples),
        "baseline_pass_rate_by_domain": {
            domain: baseline_pass_rate(baseline_rows, domain, "all")
            for domain in sorted({row["domain"] for row in baseline_rows})
        },
        "spec_audit": spec_audit_rows,
        **verdict_metrics,
    }
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 26 metrics and dynamic verdicts.")

    write_method_card(ctx, bundle, data_info, evidence_rows)
    write_spec_card(ctx, specs)
    write_operationalization_audit(ctx, evidence_rows, counterexamples)
    write_run_summary(ctx, bundle, data_info, evidence_rows, counterexamples, refinement_rows)
    write_ledger_claims(ctx, evidence_rows)
    write_plots(ctx, evidence_rows, baseline_rows, summary_rows, split_rows, counterexamples, resampling_rows)

    pass_count = sum(1 for row in evidence_rows if row.get("claim_posture") == "supported_on_train_and_eval")
    print(
        f"[lab26] wrote run_summary.md, method_card.md, causal_abstraction_spec.md, operationalization_audit.md, "
        f"and {len(evidence_rows)} evidence rows ({pass_count} supported on train and eval)"
    )
