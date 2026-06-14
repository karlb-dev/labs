"""Lab 26: Causal abstraction and causal scrubbing.

Labs 1-25 taught students to build and audit instruments. This lab makes the
claim itself into an instrument: write a high-level hypothesis, name the
low-level sites it maps to, and test whether behavior-preserving resampling
actually preserves behavior.

The implementation is deliberately narrower than a full causal-scrubbing
library. It runs residual-stream interchange interventions at hypothesis-named
sites, comparing donors that preserve a high-level variable against donors that
break it, random matched donors, wrong-site controls, and self-patching no-ops.

Evidence levels:
  * FORMAL for the explicit high-level variable mapping and resampling rules;
  * CAUSAL for residual-stream interchange interventions;
  * AUDIT for counterexample/refinement artifacts.

The output is allowed to say "this hypothesis survived these resampling tests."
It is not allowed to say "the model implements exactly this algorithm."
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import pathlib
import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

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
MAX_COUNTEREXAMPLES = 24

CONDITION_ORDER = (
    "no_op_same_example",
    "preserve_variable",
    "break_variable",
    "random_matched",
    "wrong_site_preserve",
)


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
    source_position: int
    target_position: int
    relation_family: str
    subject: str
    answer: str
    variables: dict[str, Any]
    input_ids: list[int] = dataclasses.field(default_factory=list)
    token_text: list[str] = dataclasses.field(default_factory=list)
    target_id: int = -1
    distractor_id: int = -1
    clean_diff: float = float("nan")
    top_token_text: str = ""


@dataclasses.dataclass
class HypothesisSpec:
    hypothesis_id: str
    domain: str
    behavior_metric: str
    high_level_variables: list[str]
    low_level_sites: list[dict[str, Any]]
    resampling_rules: list[dict[str, Any]]
    predicted_preservation_min: float
    predicted_damage_when_broken_min: float
    predicted_specificity_gap_min: float
    path: pathlib.Path


@dataclasses.dataclass(frozen=True)
class DonorPlan:
    condition: str
    donor_id: str
    preserves_variables: tuple[str, ...]
    breaks_variables: tuple[str, ...]
    expected: str
    note: str


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
    if not math.isfinite(f):
        return ""
    return round(f, digits)


def safe_mean(values: Sequence[Any], default: float = float("nan")) -> float:
    vals = []
    for value in values:
        try:
            f = float(value)
        except Exception:
            continue
        if math.isfinite(f):
            vals.append(f)
    return float(statistics.fmean(vals)) if vals else default


def safe_stdev(values: Sequence[Any], default: float = float("nan")) -> float:
    vals = []
    for value in values:
        try:
            f = float(value)
        except Exception:
            continue
        if math.isfinite(f):
            vals.append(f)
    if len(vals) < 2:
        return default
    return float(statistics.stdev(vals))


def as_float(value: Any, default: float = float("nan")) -> float:
    try:
        f = float(value)
    except Exception:
        return default
    return f if math.isfinite(f) else default


def item_var(item: CausalItem, key: str, default: str = "") -> str:
    value = item.variables.get(key, default)
    return str(value)


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


def load_items(ctx: bench.RunContext) -> tuple[list[CausalItem], dict[str, Any]]:
    path = data_path_from_args(ctx.args)
    if not path.exists():
        raise FileNotFoundError(f"Lab 26 data file not found: {path}")

    rows: list[dict[str, str]]
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    required = {
        "item_id", "domain", "split", "high_level_task", "template_family",
        "prompt", "target", "distractor", "source_token", "source_position",
        "target_position", "relation_family", "subject", "answer",
        "high_level_variables_json",
    }
    missing = sorted(required - set(rows[0] if rows else {}))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    items: list[CausalItem] = []
    for row in rows:
        item_id = row["item_id"].strip()
        items.append(CausalItem(
            item_id=item_id,
            domain=row["domain"].strip(),
            split=row["split"].strip(),
            high_level_task=row["high_level_task"].strip(),
            template_family=row["template_family"].strip(),
            prompt=row["prompt"],
            target=row["target"],
            distractor=row["distractor"],
            source_token=row["source_token"].strip(),
            source_position=int(row["source_position"]),
            target_position=int(row["target_position"]),
            relation_family=row["relation_family"].strip(),
            subject=row["subject"].strip(),
            answer=row["answer"].strip(),
            variables=parse_variables(row["high_level_variables_json"], item_id),
        ))

    selected = apply_item_caps(items, ctx.args)
    info = {
        "data_path": str(path),
        "data_sha256": file_sha256(path),
        "n_rows_file": len(items),
        "n_rows_selected": len(selected),
        "domains_selected": {d: sum(1 for it in selected if it.domain == d) for d in sorted({it.domain for it in selected})},
        "prompt_set": ctx.args.prompt_set,
        "max_examples": ctx.args.max_examples,
    }
    return selected, info


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
        domain_order = sorted(by_domain)
        cursor = 0
        while len(balanced) < max_examples:
            made_progress = False
            for domain in domain_order:
                if cursor < len(by_domain[domain]):
                    balanced.append(by_domain[domain][cursor])
                    made_progress = True
                    if len(balanced) >= max_examples:
                        break
            if not made_progress:
                break
            cursor += 1
        selected = balanced
    return selected


def load_specs(ctx: bench.RunContext) -> tuple[list[HypothesisSpec], list[dict[str, Any]]]:
    specs: list[HypothesisSpec] = []
    audit_rows: list[dict[str, Any]] = []
    spec_root = bench.COURSE_ROOT / "specs"
    for name in SPEC_FILES:
        path = spec_root / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {
            "hypothesis_id", "domain", "behavior_metric", "high_level_variables",
            "low_level_sites", "resampling_rules", "predicted_preservation_min",
            "predicted_damage_when_broken_min", "predicted_specificity_gap_min",
        }
        missing = sorted(required - set(payload))
        ok = not missing
        spec = HypothesisSpec(
            hypothesis_id=str(payload.get("hypothesis_id", name)),
            domain=str(payload.get("domain", "")),
            behavior_metric=str(payload.get("behavior_metric", "")),
            high_level_variables=list(payload.get("high_level_variables", [])),
            low_level_sites=list(payload.get("low_level_sites", [])),
            resampling_rules=list(payload.get("resampling_rules", [])),
            predicted_preservation_min=float(payload.get("predicted_preservation_min", 0.0)),
            predicted_damage_when_broken_min=float(payload.get("predicted_damage_when_broken_min", 0.0)),
            predicted_specificity_gap_min=float(payload.get("predicted_specificity_gap_min", 0.0)),
            path=path,
        )
        specs.append(spec)
        audit_rows.append({
            "spec_file": str(path.relative_to(bench.COURSE_ROOT)),
            "hypothesis_id": spec.hypothesis_id,
            "domain": spec.domain,
            "ok": ok,
            "missing_fields": ";".join(missing),
            "n_high_level_variables": len(spec.high_level_variables),
            "n_low_level_sites": len(spec.low_level_sites),
            "n_resampling_rules": len(spec.resampling_rules),
            "predicted_preservation_min": spec.predicted_preservation_min,
            "predicted_damage_when_broken_min": spec.predicted_damage_when_broken_min,
            "predicted_specificity_gap_min": spec.predicted_specificity_gap_min,
            "sha256": file_sha256(path),
        })
    path = ctx.path("tables", "hypothesis_spec_audit.csv")
    bench.write_csv_with_context(ctx, path, audit_rows)
    ctx.register_artifact(path, "table", "Schema and threshold audit for the selected Lab 26 hypothesis specs.")
    return specs, audit_rows


def tokenization_gate(
    ctx: bench.RunContext, bundle: bench.ModelBundle, items: list[CausalItem]
) -> tuple[list[CausalItem], list[dict[str, Any]]]:
    tokenizer = bundle.tokenizer
    kept: list[CausalItem] = []
    rows: list[dict[str, Any]] = []
    for item in items:
        prompt_ids = tokenizer.encode(item.prompt, add_special_tokens=False)
        target_ids = tokenizer.encode(item.target, add_special_tokens=False)
        distractor_ids = tokenizer.encode(item.distractor, add_special_tokens=False)
        problems: list[str] = []
        if len(target_ids) != 1:
            problems.append(f"target_tokens={len(target_ids)}")
        if len(distractor_ids) != 1:
            problems.append(f"distractor_tokens={len(distractor_ids)}")
        if not (0 <= item.source_position < len(prompt_ids)):
            problems.append("source_position_out_of_range")
        if not (0 <= item.target_position < len(prompt_ids)):
            problems.append("target_position_out_of_range")
        source_piece = ""
        if 0 <= item.source_position < len(prompt_ids):
            source_piece = tokenizer.decode([prompt_ids[item.source_position]]).strip().lower()
            if item.source_token.lower() not in source_piece:
                problems.append(f"source_token_mismatch:{source_piece}")
        item.input_ids = prompt_ids
        item.token_text = [tokenizer.decode([tid]) for tid in prompt_ids]
        if not problems:
            item.target_id = target_ids[0]
            item.distractor_id = distractor_ids[0]
            kept.append(item)
        rows.append({
            "item_id": item.item_id,
            "domain": item.domain,
            "prompt": item.prompt,
            "n_prompt_tokens": len(prompt_ids),
            "source_position": item.source_position,
            "source_token_expected": item.source_token,
            "source_token_observed": source_piece,
            "target_position": item.target_position,
            "target": item.target,
            "target_token_count": len(target_ids),
            "distractor": item.distractor,
            "distractor_token_count": len(distractor_ids),
            "kept": not problems,
            "problems": ";".join(problems),
            "tokens": " | ".join(f"{i}:{tok}" for i, tok in enumerate(item.token_text)),
        })
    path = ctx.path("diagnostics", "tokenization_gate.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "diagnostic", "Single-token answer and source-position validation for Lab 26 data.")
    if not kept:
        raise RuntimeError("Lab 26 tokenization gate dropped every item.")
    return kept, rows


def logit_diff(logits: Any, item: CausalItem) -> float:
    return float(logits[item.target_id] - logits[item.distractor_id])


def cache_items(
    ctx: bench.RunContext, bundle: bench.ModelBundle, items: list[CausalItem]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    captures: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    report_every = max(1, len(items) // 4)
    for i, item in enumerate(items):
        cap = bench.run_with_residual_cache(bundle, item.prompt)
        if cap.input_ids != item.input_ids:
            # This catches tokenizers that add special tokens in the actual run.
            # It is better to fail than to patch the wrong position silently.
            raise RuntimeError(
                f"{item.item_id}: capture tokenization differs from tokenization gate. "
                "This lab currently requires raw prompt token positions to match the forward pass."
            )
        captures[item.item_id] = cap
        item.clean_diff = logit_diff(cap.final_logits_last, item)
        top_id = int(cap.final_logits_last.argmax())
        item.top_token_text = bundle.tokenizer.decode([top_id])
        rows.append({
            "item_id": item.item_id,
            "domain": item.domain,
            "split": item.split,
            "high_level_task": item.high_level_task,
            "relation_family": item.relation_family,
            "subject": item.subject,
            "target": item.target,
            "distractor": item.distractor,
            "clean_diff": rounded(item.clean_diff),
            "baseline_pass": item.clean_diff > MIN_BASELINE_MARGIN,
            "top_token": item.top_token_text,
            "copy_source": item_var(item, "COPY_SOURCE"),
            "query_token": item_var(item, "QUERY_TOKEN"),
            "relation": item_var(item, "RELATION", item.relation_family),
        })
        if (i + 1) % report_every == 0 or i + 1 == len(items):
            print(f"[lab26] cached {i + 1}/{len(items)} prompts")
    path = ctx.path("tables", "baseline_behavior.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Clean prompt logit margins and baseline gate status.")
    return captures, rows


def choose_donor(
    item: CausalItem,
    items: Sequence[CausalItem],
    predicate: Any,
    *,
    label: str,
) -> CausalItem | None:
    candidates = [cand for cand in items if cand.item_id != item.item_id and predicate(cand)]
    if not candidates:
        return None
    candidates.sort(key=lambda c: stable_int(f"{label}|{item.item_id}|{c.item_id}"))
    return candidates[0]


def build_donor_plans(items: list[CausalItem]) -> tuple[dict[str, list[DonorPlan]], list[dict[str, Any]]]:
    by_id = {item.item_id: item for item in items}
    plans: dict[str, list[DonorPlan]] = {}
    audit_rows: list[dict[str, Any]] = []
    for item in items:
        item_plans = [
            DonorPlan(
                condition="no_op_same_example",
                donor_id=item.item_id,
                preserves_variables=tuple(),
                breaks_variables=tuple(),
                expected="preserve",
                note="self-patching identity control",
            )
        ]
        if item.domain == "induction":
            preserve = choose_donor(
                item, items,
                lambda c: c.domain == item.domain
                and item_var(c, "COPY_SOURCE") == item_var(item, "COPY_SOURCE")
                and item_var(c, "QUERY_TOKEN") == item_var(item, "QUERY_TOKEN"),
                label="preserve_induction",
            )
            broken = choose_donor(
                item, items,
                lambda c: c.domain == item.domain
                and item_var(c, "QUERY_TOKEN") == item_var(item, "QUERY_TOKEN")
                and item_var(c, "COPY_SOURCE") != item_var(item, "COPY_SOURCE"),
                label="break_induction",
            )
            preserved_vars = ("COPY_SOURCE", "QUERY_TOKEN")
            broken_vars = ("COPY_SOURCE",)
        elif item.domain == "relation":
            preserve = choose_donor(
                item, items,
                lambda c: c.domain == item.domain
                and c.relation_family == item.relation_family
                and c.subject != item.subject,
                label="preserve_relation",
            )
            broken = choose_donor(
                item, items,
                lambda c: c.domain == item.domain
                and c.subject == item.subject
                and c.relation_family != item.relation_family,
                label="break_relation",
            )
            preserved_vars = ("RELATION",)
            broken_vars = ("RELATION",)
        else:
            preserve = None
            broken = None
            preserved_vars = tuple()
            broken_vars = tuple()

        if preserve is not None:
            item_plans.append(DonorPlan(
                condition="preserve_variable",
                donor_id=preserve.item_id,
                preserves_variables=preserved_vars,
                breaks_variables=tuple(),
                expected="preserve",
                note="donor preserves the hypothesis variable while changing surface/context variables",
            ))
            item_plans.append(DonorPlan(
                condition="wrong_site_preserve",
                donor_id=preserve.item_id,
                preserves_variables=preserved_vars,
                breaks_variables=tuple(),
                expected="control",
                note="same preserving donor patched at a site outside the hypothesis variable site",
            ))
        if broken is not None:
            item_plans.append(DonorPlan(
                condition="break_variable",
                donor_id=broken.item_id,
                preserves_variables=tuple(),
                breaks_variables=broken_vars,
                expected="damage",
                note="donor breaks the named high-level variable",
            ))
        random_donor = choose_donor(
            item, items,
            lambda c: c.domain == item.domain and c.input_ids and len(c.input_ids) == len(item.input_ids),
            label="random_matched",
        )
        if random_donor is not None:
            item_plans.append(DonorPlan(
                condition="random_matched",
                donor_id=random_donor.item_id,
                preserves_variables=tuple(),
                breaks_variables=tuple(),
                expected="control",
                note="deterministic random same-domain, same-length donor",
            ))
        plans[item.item_id] = item_plans
        for plan in item_plans:
            donor = by_id[plan.donor_id]
            audit_rows.append({
                "item_id": item.item_id,
                "domain": item.domain,
                "condition": plan.condition,
                "donor_id": plan.donor_id,
                "donor_domain": donor.domain,
                "donor_prompt": donor.prompt,
                "expected": plan.expected,
                "preserves_variables": ";".join(plan.preserves_variables),
                "breaks_variables": ";".join(plan.breaks_variables),
                "note": plan.note,
            })
    return plans, audit_rows


def position_value(item: CausalItem, name: str) -> int:
    if name == "source_position":
        return item.source_position
    if name == "target_position":
        return item.target_position
    if name == "final_position":
        return len(item.input_ids) - 1
    raise ValueError(f"Unknown Lab 26 position spec {name!r}")


def wrong_position(item: CausalItem, donor: CausalItem) -> int:
    banned = {item.source_position, item.target_position}
    max_len = min(len(item.input_ids), len(donor.input_ids))
    for pos in range(max_len):
        if pos not in banned and item.input_ids[pos] == donor.input_ids[pos]:
            return pos
    for pos in range(max_len):
        if pos not in banned:
            return pos
    return 0


def stream_depths(bundle: bench.ModelBundle, args: Any, depth_spec: Any) -> list[int]:
    n_layers = bundle.anatomy.n_layers
    if isinstance(depth_spec, list):
        return sorted({max(0, min(n_layers, int(d))) for d in depth_spec})
    if str(depth_spec).lower() == "all":
        return list(range(n_layers + 1))
    if str(depth_spec).lower() == "coarse" and str(args.prompt_set) == "full":
        return list(range(n_layers + 1))
    fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
    return sorted({max(0, min(n_layers, int(round(n_layers * f)))) for f in fractions})


def iter_residual_sites(
    spec: HypothesisSpec, bundle: bench.ModelBundle, args: Any
) -> list[dict[str, Any]]:
    sites: list[dict[str, Any]] = []
    for site in spec.low_level_sites:
        if site.get("kind") != "residual":
            continue
        for pos_name in site.get("positions", []):
            sites.append({
                "kind": "residual",
                "position_name": str(pos_name),
                "site_label": f"residual:{pos_name}",
                "depths": stream_depths(bundle, args, site.get("stream_depths", "coarse")),
                "rationale": site.get("rationale", ""),
            })
    return sites


def run_resampling(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    items: list[CausalItem],
    specs: list[HypothesisSpec],
    captures: dict[str, Any],
    donor_plans: dict[str, list[DonorPlan]],
) -> list[dict[str, Any]]:
    by_id = {item.item_id: item for item in items}
    rows: list[dict[str, Any]] = []
    total = 0
    for spec in specs:
        spec_items = [item for item in items if item.domain == spec.domain]
        for site in iter_residual_sites(spec, bundle, ctx.args):
            for _item in spec_items:
                for _plan in donor_plans[_item.item_id]:
                    total += len(site["depths"])
    print(f"[lab26] running {total} residual-resampling forwards")
    done = 0
    report_every = max(1, total // 5)

    for spec in specs:
        spec_items = [item for item in items if item.domain == spec.domain]
        for site in iter_residual_sites(spec, bundle, ctx.args):
            for item in spec_items:
                item_cap = captures[item.item_id]
                for plan in donor_plans[item.item_id]:
                    donor = by_id[plan.donor_id]
                    donor_cap = captures[donor.item_id]
                    same_length = len(item.input_ids) == len(donor.input_ids)
                    site_label = site["site_label"]
                    if plan.condition == "wrong_site_preserve":
                        target_pos = wrong_position(item, donor)
                        donor_pos = target_pos
                        position_name = "wrong_position"
                        patched_site_label = "residual:wrong_position"
                    else:
                        target_pos = position_value(item, site["position_name"])
                        donor_pos = position_value(donor, site["position_name"])
                        position_name = site["position_name"]
                        patched_site_label = site_label

                    for depth in site["depths"]:
                        if not same_length or donor_pos >= donor_cap.streams.shape[1] or target_pos >= item_cap.streams.shape[1]:
                            patched_diff = float("nan")
                            scrub_score = float("nan")
                            error = "position_or_length_mismatch"
                        else:
                            logits = bench.run_with_residual_patch(
                                bundle,
                                item.prompt,
                                depth,
                                target_pos,
                                donor_cap.streams[depth, donor_pos],
                            )
                            patched_diff = logit_diff(logits, item)
                            scrub_score = (
                                patched_diff / item.clean_diff
                                if abs(item.clean_diff) > 1e-9
                                else float("nan")
                            )
                            error = ""
                        rows.append({
                            "hypothesis_id": spec.hypothesis_id,
                            "domain": item.domain,
                            "item_id": item.item_id,
                            "donor_id": donor.item_id,
                            "condition": plan.condition,
                            "expected": plan.expected,
                            "site": site_label,
                            "patched_site": patched_site_label,
                            "position_name": position_name,
                            "target_position_index": target_pos,
                            "donor_position_index": donor_pos,
                            "depth": depth,
                            "clean_diff": rounded(item.clean_diff),
                            "patched_diff": rounded(patched_diff),
                            "scrub_score": rounded(scrub_score),
                            "delta_from_clean": rounded(patched_diff - item.clean_diff),
                            "baseline_pass": item.clean_diff > MIN_BASELINE_MARGIN,
                            "target_prompt": item.prompt,
                            "donor_prompt": donor.prompt,
                            "target": item.target,
                            "distractor": item.distractor,
                            "preserves_variables": ";".join(plan.preserves_variables),
                            "breaks_variables": ";".join(plan.breaks_variables),
                            "copy_source": item_var(item, "COPY_SOURCE"),
                            "donor_copy_source": item_var(donor, "COPY_SOURCE"),
                            "relation": item_var(item, "RELATION", item.relation_family),
                            "donor_relation": item_var(donor, "RELATION", donor.relation_family),
                            "subject": item.subject,
                            "donor_subject": donor.subject,
                            "error": error,
                        })
                        done += 1
                        if done % report_every == 0 or done == total:
                            print(f"[lab26] resampling {done}/{total}")
    return rows


def aggregate_resampling(
    rows: list[dict[str, Any]], specs: Sequence[HypothesisSpec]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    spec_by_id = {spec.hypothesis_id: spec for spec in specs}
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("baseline_pass") is True or str(row.get("baseline_pass")).lower() == "true":
            grouped[(row["domain"], row["hypothesis_id"], row["site"], int(row["depth"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (domain, hypothesis_id, site, depth), group in sorted(grouped.items()):
        cond_values = {
            cond: [as_float(r.get("scrub_score")) for r in group if r["condition"] == cond and not r.get("error")]
            for cond in CONDITION_ORDER
        }
        means = {cond: safe_mean(vals) for cond, vals in cond_values.items()}
        counts = {cond: sum(1 for v in vals if math.isfinite(v)) for cond, vals in cond_values.items()}
        spec = spec_by_id[hypothesis_id]
        preserve = means["preserve_variable"]
        broken = means["break_variable"]
        random_ctl = means["random_matched"]
        wrong_site = means["wrong_site_preserve"]
        control_floor = safe_mean([v for v in (random_ctl, wrong_site) if math.isfinite(v)])
        if math.isfinite(random_ctl) and math.isfinite(wrong_site):
            control_floor = max(random_ctl, wrong_site)
        damage_gap = preserve - broken if math.isfinite(preserve) and math.isfinite(broken) else float("nan")
        specificity_gap = preserve - control_floor if math.isfinite(preserve) and math.isfinite(control_floor) else float("nan")
        pass_preserve = math.isfinite(preserve) and preserve >= spec.predicted_preservation_min
        pass_damage = math.isfinite(damage_gap) and damage_gap >= spec.predicted_damage_when_broken_min
        pass_specificity = math.isfinite(specificity_gap) and specificity_gap >= spec.predicted_specificity_gap_min
        formal_pass = pass_preserve and pass_damage and pass_specificity
        summary_rows.append({
            "domain": domain,
            "hypothesis_id": hypothesis_id,
            "site": site,
            "depth": depth,
            "mean_noop": rounded(means["no_op_same_example"]),
            "mean_preserve_variable": rounded(preserve),
            "mean_break_variable": rounded(broken),
            "mean_random_matched": rounded(random_ctl),
            "mean_wrong_site_preserve": rounded(wrong_site),
            "control_floor": rounded(control_floor),
            "damage_gap": rounded(damage_gap),
            "specificity_gap": rounded(specificity_gap),
            "n_preserve": counts["preserve_variable"],
            "n_break": counts["break_variable"],
            "pass_preservation": pass_preserve,
            "pass_damage": pass_damage,
            "pass_specificity": pass_specificity,
            "formal_pass": formal_pass,
            "predicted_preservation_min": spec.predicted_preservation_min,
            "predicted_damage_when_broken_min": spec.predicted_damage_when_broken_min,
            "predicted_specificity_gap_min": spec.predicted_specificity_gap_min,
        })

    best_rows: list[dict[str, Any]] = []
    for hypothesis_id in sorted({row["hypothesis_id"] for row in summary_rows}):
        candidates = [row for row in summary_rows if row["hypothesis_id"] == hypothesis_id]
        def best_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
            depth = int(row["depth"])
            site = str(row["site"])
            site_priority = 2 if "source_position" in site else 1 if "target_position" in site else 0
            return (
                bool(row["formal_pass"]),
                site_priority,
                depth > 0,
                -depth,  # once a cell passes, prefer the earliest non-embedding depth
                as_float(row["damage_gap"], -999.0),
                as_float(row["specificity_gap"], -999.0),
                as_float(row["mean_preserve_variable"], -999.0),
            )
        candidates.sort(
            key=best_key,
            reverse=True,
        )
        if candidates:
            best = dict(candidates[0])
            best["selection_rule"] = "formal_pass_then_source_site_then_earliest_non_embedding_depth_then_gaps"
            best_rows.append(best)

    metrics = {
        "n_interventions": len(rows),
        "n_summary_cells": len(summary_rows),
        "best_by_hypothesis": best_rows,
        "verdicts": {
            row["hypothesis_id"]: (
                "formal_causal_supported" if row["formal_pass"]
                else "needs_refinement_or_failed_controls"
            )
            for row in best_rows
        },
    }
    return summary_rows, best_rows, metrics


def build_counterexamples(
    rows: list[dict[str, Any]], specs: Sequence[HypothesisSpec], best_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    spec_by_id = {spec.hypothesis_id: spec for spec in specs}
    best_lookup = {row["hypothesis_id"]: row for row in best_rows}
    out: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("baseline_pass") or row.get("error"):
            continue
        spec = spec_by_id[row["hypothesis_id"]]
        score = as_float(row.get("scrub_score"))
        if not math.isfinite(score):
            continue
        kind = ""
        severity = 0.0
        if row["condition"] == "preserve_variable" and score < spec.predicted_preservation_min:
            kind = "preservation_failure"
            severity = spec.predicted_preservation_min - score
        elif row["condition"] == "break_variable" and score > spec.predicted_preservation_min - spec.predicted_damage_when_broken_min:
            kind = "broken_variable_leak"
            severity = score - (spec.predicted_preservation_min - spec.predicted_damage_when_broken_min)
        elif row["condition"] in {"random_matched", "wrong_site_preserve"} and score > spec.predicted_preservation_min:
            kind = "control_leak"
            severity = score - spec.predicted_preservation_min
        if kind:
            best = best_lookup.get(row["hypothesis_id"], {})
            out.append({
                "kind": kind,
                "severity": rounded(severity),
                "hypothesis_id": row["hypothesis_id"],
                "domain": row["domain"],
                "item_id": row["item_id"],
                "donor_id": row["donor_id"],
                "condition": row["condition"],
                "site": row["site"],
                "depth": row["depth"],
                "scrub_score": row["scrub_score"],
                "clean_diff": row["clean_diff"],
                "patched_diff": row["patched_diff"],
                "target_prompt": row["target_prompt"],
                "donor_prompt": row["donor_prompt"],
                "best_site_for_hypothesis": best.get("site", ""),
                "best_depth_for_hypothesis": best.get("depth", ""),
            })
    out.sort(key=lambda r: as_float(r["severity"], 0.0), reverse=True)
    return out[:MAX_COUNTEREXAMPLES]


def build_refinement_log(
    best_rows: Sequence[Mapping[str, Any]],
    specs: Sequence[HypothesisSpec],
    counterexamples: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    spec_by_id = {spec.hypothesis_id: spec for spec in specs}
    counter_by_hyp = defaultdict(int)
    for row in counterexamples:
        counter_by_hyp[row["hypothesis_id"]] += 1
    rows: list[dict[str, Any]] = []
    for best in best_rows:
        spec = spec_by_id[best["hypothesis_id"]]
        failed: list[str] = []
        if not best.get("pass_preservation"):
            failed.append("preservation_below_prediction")
        if not best.get("pass_damage"):
            failed.append("broken_variable_not_damaging_enough")
        if not best.get("pass_specificity"):
            failed.append("controls_too_close_to_preserving_donor")
        if not failed and counter_by_hyp[spec.hypothesis_id] == 0:
            rows.append({
                "hypothesis_id": spec.hypothesis_id,
                "version": "v1",
                "failed_rule": "",
                "evidence_path": "tables/variable_preservation_summary.csv",
                "revision": "No automatic revision proposed. The v1 hypothesis passed the current residual-resampling gates; replicate on a larger prompt set before broadening the claim.",
                "student_notes": "",
            })
            continue
        if not failed:
            rows.append({
                "hypothesis_id": spec.hypothesis_id,
                "version": "v1",
                "failed_rule": "counterexamples_outside_best_cell",
                "evidence_path": "tables/counterexamples.csv",
                "revision": (
                    f"The best cell passed, but {counter_by_hyp[spec.hypothesis_id]} other nominated cells "
                    f"crossed counterexample thresholds. Narrow v2 to {best.get('site')} at depth "
                    f"{best.get('depth')} and treat other sites/depths as outside the current claim."
                ),
                "student_notes": "",
            })
            rows.append({
                "hypothesis_id": spec.hypothesis_id,
                "version": "v2_proposed",
                "failed_rule": "student_to_test",
                "evidence_path": "tables/best_hypothesis_cells.csv",
                "revision": "Rerun a v2 spec that names only the surviving site/depth band, then test held-out items.",
                "student_notes": "",
            })
            continue
        revision_bits = [
            f"narrow low_level_sites to {best.get('site')} at depth {best.get('depth')}",
            "add a held-out surface slice before using general language",
        ]
        if "controls_too_close_to_preserving_donor" in failed:
            revision_bits.append("name the control leak as an allowed failure mode or add a more specific variable")
        if spec.domain == "relation":
            revision_bits.append("separate relation-token evidence from final-position answer evidence")
        if spec.domain == "induction":
            revision_bits.append("separate COPY_SOURCE preservation from generic repeated-sequence facilitation")
        rows.append({
            "hypothesis_id": spec.hypothesis_id,
            "version": "v1",
            "failed_rule": ";".join(failed),
            "evidence_path": "tables/variable_preservation_summary.csv",
            "revision": "; ".join(revision_bits),
            "student_notes": "",
        })
        rows.append({
            "hypothesis_id": spec.hypothesis_id,
            "version": "v2_proposed",
            "failed_rule": "student_to_test",
            "evidence_path": "tables/counterexamples.csv",
            "revision": "Write a smaller formal claim that excludes the listed counterexamples, then rerun with --prompt-set medium or full.",
            "student_notes": "",
        })
    return rows


def build_evidence_matrix(
    best_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    counterexamples: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    baseline_by_domain: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in baseline_rows:
        baseline_by_domain[str(row["domain"])].append(row)
    counter_by_hyp = defaultdict(int)
    for row in counterexamples:
        counter_by_hyp[row["hypothesis_id"]] += 1
    rows = []
    for best in best_rows:
        domain = str(best["domain"])
        base = baseline_by_domain[domain]
        pass_rate = safe_mean([1.0 if row["baseline_pass"] else 0.0 for row in base], 0.0)
        has_counterexamples = counter_by_hyp[best["hypothesis_id"]] > 0
        if best["formal_pass"] and has_counterexamples:
            evidence_tag = "FORMAL+CAUSAL,AUDIT"
            claim_posture = "best-cell supported; refine v2 scope"
        elif best["formal_pass"]:
            evidence_tag = "FORMAL+CAUSAL"
            claim_posture = "narrow supported hypothesis"
        else:
            evidence_tag = "FORMAL+CAUSAL_FAILED_OR_REFINED"
            claim_posture = "counterexample-driven refinement required"
        rows.append({
            "hypothesis_id": best["hypothesis_id"],
            "domain": domain,
            "evidence_tag": evidence_tag,
            "best_site": best["site"],
            "best_depth": best["depth"],
            "baseline_pass_rate": rounded(pass_rate),
            "preservation": best["mean_preserve_variable"],
            "broken_variable": best["mean_break_variable"],
            "damage_gap": best["damage_gap"],
            "specificity_gap": best["specificity_gap"],
            "formal_pass": best["formal_pass"],
            "counterexamples": counter_by_hyp[best["hypothesis_id"]],
            "claim_posture": claim_posture,
        })
    return rows


def write_plot_reading_guide(ctx: bench.RunContext) -> None:
    rows = [
        {
            "plot": "plots/causal_abstraction_dashboard.png",
            "read_for": "One-screen pass/fail posture: baseline health, preserve-vs-break gaps, and counterexample load.",
            "do_not_claim": "A dashboard pass is not a proof of the whole algorithm.",
        },
        {
            "plot": "plots/resampling_preservation_matrix.png",
            "read_for": "Which depth/site/condition combinations preserve the clean behavior.",
            "do_not_claim": "A hot cell by itself does not identify a complete circuit.",
        },
        {
            "plot": "plots/hypothesis_pass_fail_atlas.png",
            "read_for": "Which formal gates passed: preservation, damage, specificity, and all-gates.",
            "do_not_claim": "A failed gate is not a broken lab; it is a counterexample.",
        },
        {
            "plot": "plots/variable_specificity_ladder.png",
            "read_for": "Whether preserving donors beat broken-variable and random/wrong-site controls.",
            "do_not_claim": "Preservation above random is not enough if broken-variable donors also preserve.",
        },
        {
            "plot": "plots/refinement_trajectory.png",
            "read_for": "What the automatic v2 proposal would narrow after v1 failures.",
            "do_not_claim": "The proposed v2 is validated before rerunning it.",
        },
        {
            "plot": "plots/counterexample_gallery.png",
            "read_for": "The rows most likely to kill or shrink the favorite explanation.",
            "do_not_claim": "Counterexamples can be ignored because aggregates look good.",
        },
    ]
    path = ctx.path("plots", "plot_reading_guide.csv")
    bench.write_csv(path, rows)
    ctx.register_artifact(path, "table", "Reading guide for the Lab 26 plot suite.")


def write_method_card(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    metrics: Mapping[str, Any],
    evidence_rows: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Lab 26 method card",
        "",
        "Question: can a high-level explanation survive residual-stream resampling tests?",
        "",
        "## Scope",
        "",
        f"- model: `{bundle.anatomy.model_id}` ({bundle.anatomy.n_layers} blocks, d_model {bundle.anatomy.d_model})",
        "- intervention: residual-stream interchange resampling at hypothesis-named token positions",
        "- metric: next-token `logit(target) - logit(distractor)`",
        "- evidence tags: `FORMAL` for the spec, `CAUSAL` for interchange interventions, `AUDIT` for counterexamples",
        "",
        "## Verdicts",
        "",
        "| hypothesis | domain | best site | depth | preservation | break | damage gap | specificity gap | verdict |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in evidence_rows:
        lines.append(
            f"| `{row['hypothesis_id']}` | {row['domain']} | `{row['best_site']}` | {row['best_depth']} | "
            f"{row['preservation']} | {row['broken_variable']} | {row['damage_gap']} | "
            f"{row['specificity_gap']} | {row['claim_posture']} |"
        )
    lines += [
        "",
        "## Contract",
        "",
        "A positive result means the named abstraction survived this resampling battery on this dataset.",
        "It does not mean the model implements exactly this algorithm, that the same mapping holds in every context,",
        "or that the selected site is the whole mechanism.",
        "",
        "Start with `tables/evidence_matrix.csv`, then inspect `tables/counterexamples.csv` before writing any claim.",
        "",
    ]
    path = ctx.path("method_card.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Compact method contract and Lab 26 verdict table.")


def write_spec_card(
    ctx: bench.RunContext,
    specs: Sequence[HypothesisSpec],
    spec_audit_rows: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Lab 26 causal abstraction specs",
        "",
        "A Lab 26 hypothesis has four moving parts:",
        "",
        "1. a behavior metric;",
        "2. high-level variables;",
        "3. low-level sites and token-position semantics;",
        "4. resampling rules that say which variables are preserved or broken.",
        "",
        "The run used these specs:",
        "",
    ]
    for spec in specs:
        lines += [
            f"## `{spec.hypothesis_id}`",
            "",
            f"- file: `{spec.path.relative_to(bench.COURSE_ROOT)}`",
            f"- domain: `{spec.domain}`",
            f"- metric: `{spec.behavior_metric}`",
            f"- high-level variables: `{', '.join(spec.high_level_variables)}`",
            f"- thresholds: preserve >= {spec.predicted_preservation_min}, "
            f"preserve-minus-break >= {spec.predicted_damage_when_broken_min}, "
            f"preserve-minus-control >= {spec.predicted_specificity_gap_min}",
            "",
            "Low-level sites:",
            "",
        ]
        for site in spec.low_level_sites:
            lines.append(
                f"- `{site.get('kind')}` at `{site.get('positions')}` with depths `{site.get('stream_depths')}`: "
                f"{site.get('rationale', '')}"
            )
        lines += ["", "Resampling rules:", ""]
        for rule in spec.resampling_rules:
            lines.append(f"- preserve `{rule.get('preserve', [])}`, vary `{rule.get('vary', [])}`, break `{rule.get('break', [])}`")
        lines.append("")
    lines += [
        "The machine-readable schema audit is in `tables/hypothesis_spec_audit.csv`.",
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
        "Favorite interpretation under attack: a compact high-level variable mapping explains the model behavior.",
        "",
        "## Cheap explanations and controls",
        "",
        "| Cheap explanation | Control in this lab | What would kill the favorite story? |",
        "|---|---|---|",
        "| Any donor works because patching perturbs the stream | `random_matched` donor | random donor preserves about as well as variable-preserving donor |",
        "| The site is not specific | `wrong_site_preserve` | wrong-site patches preserve about as well as named-site patches |",
        "| The variable name is too broad | `break_variable` donor | broken-variable donors preserve behavior |",
        "| The dataset is too easy | baseline gate and counterexamples | only a few high-margin items carry the aggregate |",
        "| The hypothesis is post hoc | committed JSON spec and spec audit | thresholds are changed after seeing failures |",
        "",
        "## Run verdicts",
        "",
    ]
    for row in evidence_rows:
        lines.append(
            f"- `{row['hypothesis_id']}`: {row['claim_posture']} "
            f"(damage gap {row['damage_gap']}, specificity gap {row['specificity_gap']}, "
            f"{row['counterexamples']} counterexamples)."
        )
    lines += [
        "",
        "## Counterexample discipline",
        "",
    ]
    if counterexamples:
        for row in counterexamples[:8]:
            lines.append(
                f"- `{row['kind']}`: `{row['item_id']}` with donor `{row['donor_id']}` "
                f"at `{row['site']}` depth {row['depth']} scored {row['scrub_score']}."
            )
    else:
        lines.append("- No automatic counterexamples crossed the configured thresholds. Replicate before generalizing.")
    lines += [
        "",
        "## Allowed language",
        "",
        "- `The formal hypothesis survived/did not survive this residual-resampling battery.`",
        "- `Preserving variable X preserved the target margin more than breaking X under these sites.`",
        "",
        "## Forbidden language",
        "",
        "- `The model implements exactly this algorithm.`",
        "- `This site is the whole circuit.`",
        "- `This abstraction holds outside this prompt family without replication.`",
        "",
    ]
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Cheap explanations, killer controls, counterexamples, and allowed claim grammar.")


def write_run_summary(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    data_info: Mapping[str, Any],
    evidence_rows: Sequence[Mapping[str, Any]],
    counterexamples: Sequence[Mapping[str, Any]],
    refinement_rows: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Lab 26 run summary: causal abstraction and causal scrubbing",
        "",
        "## Run identity",
        "",
        f"- model: `{bundle.anatomy.model_id}` ({bundle.anatomy.n_layers} blocks, d_model {bundle.anatomy.d_model})",
        f"- data: `{pathlib.Path(str(data_info['data_path'])).name}` sha256 `{str(data_info['data_sha256'])[:16]}`; "
        f"{data_info['n_rows_selected']} selected rows from {data_info['n_rows_file']}",
        f"- selected domains: `{data_info['domains_selected']}`",
        "- intervention: residual-stream resampling at hypothesis-named token positions",
        "- evidence: `FORMAL + CAUSAL`, with counterexample/refinement audit",
        "",
        "## 1. What behavior was measured?",
        "",
        "Next-token target-vs-distractor logit margins on two domains: induction copying and relation-answer prompts.",
        "",
        "## 2. What abstraction was proposed?",
        "",
        "`specs/lab26_induction_hypothesis.json` maps COPY_SOURCE/QUERY_TOKEN to induction sites; "
        "`specs/lab26_relation_hypothesis.json` maps RELATION/SUBJECT to relation-token and final-position sites.",
        "",
        "## 3. What intervention tested it?",
        "",
        "For each item, the lab patched residual streams from donors that preserved the named variable, donors that broke it, "
        "random matched donors, and wrong-site preserving donors. The score is `patched_diff / clean_diff`.",
        "",
        "## 4. Headline verdicts",
        "",
        "| hypothesis | domain | best site | depth | preserve | break | damage gap | specificity gap | posture |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in evidence_rows:
        lines.append(
            f"| `{row['hypothesis_id']}` | {row['domain']} | `{row['best_site']}` | {row['best_depth']} | "
            f"{row['preservation']} | {row['broken_variable']} | {row['damage_gap']} | "
            f"{row['specificity_gap']} | {row['claim_posture']} |"
        )
    lines += [
        "",
        "## 5. What should students read first?",
        "",
        "1. `method_card.md` for the verdict table.",
        "2. `causal_abstraction_spec.md` for the formal mapping.",
        "3. `tables/evidence_matrix.csv` and `tables/variable_preservation_summary.csv` for the numbers.",
        "4. `tables/counterexamples.csv` before writing any positive claim.",
        "5. `operationalization_audit.md` for the allowed and forbidden language.",
        "6. `plots/causal_abstraction_dashboard.png`, then the plot guide in `plots/plot_reading_guide.csv`.",
        "",
        "## 6. Counterexamples and refinement",
        "",
        f"- automatic counterexamples written: {len(counterexamples)}",
        f"- refinement rows written: {len(refinement_rows)}",
        "",
        "If v1 failed a gate or produced counterexamples outside the best cell, the correct student move is not to delete the failed row. It is to write the smaller v2 claim and rerun.",
        "",
        "## 7. Caveats",
        "",
        "- This first Lab 26 pass tests residual-stream resampling only. It is not path-specific scrubbing; Lab 27 owns paths.",
        "- A preserved margin is evidence for this formal mapping under this intervention, not for a universal algorithm.",
        "- Low or negative margins make scrub ratios unstable; the baseline table records which items passed the gate.",
        "- Relation donors preserve RELATION while the recipient prompt keeps SUBJECT. This is a relation-identity test, not an answer-copy test.",
        "",
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "The seven standard questions answered for Lab 26.")


def write_ledger_claims(
    ctx: bench.RunContext,
    evidence_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    run_name = ctx.run_dir.name
    claims: list[dict[str, str]] = []
    for i, row in enumerate(evidence_rows, start=1):
        evidence_tag = str(row.get("evidence_tag", "FORMAL+CAUSAL"))
        if row["formal_pass"] and "AUDIT" in evidence_tag:
            text = (
                f"For `{row['domain']}` prompts in Lab 26, formal hypothesis `{row['hypothesis_id']}` "
                f"has a supported best cell at `{row['best_site']}` depth {row['best_depth']}: "
                f"preserving donors scored {row['preservation']} vs broken-variable {row['broken_variable']} "
                f"(damage gap {row['damage_gap']}, specificity gap {row['specificity_gap']}). "
                f"However, {row['counterexamples']} counterexamples outside that best cell require a narrower v2 scope."
            )
            tag = "FORMAL+CAUSAL,AUDIT"
        elif row["formal_pass"]:
            text = (
                f"For `{row['domain']}` prompts in Lab 26, formal hypothesis `{row['hypothesis_id']}` "
                f"survived residual resampling at `{row['best_site']}` depth {row['best_depth']}: "
                f"preserving donors scored {row['preservation']} vs broken-variable {row['broken_variable']} "
                f"(damage gap {row['damage_gap']}, specificity gap {row['specificity_gap']})."
            )
            tag = "FORMAL+CAUSAL"
        else:
            text = (
                f"For `{row['domain']}` prompts in Lab 26, formal hypothesis `{row['hypothesis_id']}` "
                f"did not earn the positive abstraction claim under the current residual-resampling gates "
                f"(best damage gap {row['damage_gap']}, specificity gap {row['specificity_gap']})."
            )
            tag = "FORMAL+CAUSAL,AUDIT"
        claims.append({
            "id": f"{LAB_ID}-C{i}",
            "tag": tag,
            "text": text,
            "artifact": f"runs/{run_name}/tables/evidence_matrix.csv",
            "falsifier": (
                "A held-out run where preserving donors no longer beat broken-variable and wrong-site/random controls, "
                "or a counterexample showing the same score with the named variable broken."
            ),
        })
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)
    return claims


def plot_dashboard(
    ctx: bench.RunContext,
    evidence_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    counterexamples: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Lab 26 causal abstraction dashboard", fontsize=14, fontweight="bold")

    labels = [str(r["domain"]) for r in evidence_rows]
    x = np.arange(len(labels))
    width = 0.22
    axes[0, 0].bar(x - width, [as_float(r["preservation"], 0.0) for r in evidence_rows], width, label="preserve", color="#009E73")
    axes[0, 0].bar(x, [as_float(r["broken_variable"], 0.0) for r in evidence_rows], width, label="break", color="#D55E00")
    axes[0, 0].bar(x + width, [max(0.0, as_float(r["specificity_gap"], 0.0)) for r in evidence_rows], width, label="specificity gap", color="#0072B2")
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylabel("scrub score / gap")
    axes[0, 0].set_title("Best resampling cells")
    axes[0, 0].axhline(0, color="#444444", linewidth=0.8)
    axes[0, 0].legend(frameon=False, fontsize=8)

    by_domain = defaultdict(list)
    for row in baseline_rows:
        by_domain[str(row["domain"])].append(as_float(row["clean_diff"]))
    axes[0, 1].boxplot([[v for v in by_domain[d] if math.isfinite(v)] for d in labels], labels=labels, patch_artist=True)
    axes[0, 1].axhline(MIN_BASELINE_MARGIN, color="#D55E00", linestyle="--", linewidth=1, label="baseline gate")
    axes[0, 1].set_ylabel("clean logit diff")
    axes[0, 1].set_title("Baseline behavior health")
    axes[0, 1].legend(frameon=False, fontsize=8)

    gates = ["formal_pass", "damage_gap", "specificity_gap"]
    mat = []
    for row in evidence_rows:
        mat.append([
            1.0 if row["formal_pass"] else 0.0,
            as_float(row["damage_gap"], 0.0),
            as_float(row["specificity_gap"], 0.0),
        ])
    im = axes[1, 0].imshow(mat, aspect="auto", cmap="viridis", vmin=0)
    axes[1, 0].set_xticks(range(len(gates)), gates, rotation=20, ha="right")
    axes[1, 0].set_yticks(range(len(labels)), labels)
    axes[1, 0].set_title("Pass/fail and gap atlas")
    fig.colorbar(im, ax=axes[1, 0], shrink=0.8)

    counts = defaultdict(int)
    for row in counterexamples:
        counts[str(row["kind"])] += 1
    c_labels = sorted(counts) or ["none"]
    c_vals = [counts[k] for k in c_labels] if counts else [0]
    axes[1, 1].bar(c_labels, c_vals, color="#CC79A7")
    axes[1, 1].set_title("Automatic counterexamples")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].tick_params(axis="x", rotation=20)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    bench.save_figure(ctx, fig, "causal_abstraction_dashboard.png", "One-screen Lab 26 evidence posture.")


def plot_resampling_matrix(ctx: bench.RunContext, summary_rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = list(summary_rows)
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No baseline-passing resampling cells", ha="center", va="center")
        ax.axis("off")
        bench.save_figure(ctx, fig, "resampling_preservation_matrix.png", "Empty resampling matrix placeholder.")
        return
    rows.sort(key=lambda r: (r["domain"], r["site"], int(r["depth"])))
    labels = [f"{r['domain']} {r['site']} d{r['depth']}" for r in rows]
    cols = ["mean_noop", "mean_preserve_variable", "mean_break_variable", "mean_random_matched", "mean_wrong_site_preserve"]
    mat = np.array([[as_float(r[c], np.nan) for c in cols] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.26 * len(rows))))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=1.5)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xticks(range(len(cols)), [c.replace("mean_", "") for c in cols], rotation=30, ha="right")
    ax.set_title("Mean scrub score by site, depth, and condition")
    fig.colorbar(im, ax=ax, shrink=0.8, label="patched_diff / clean_diff")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "resampling_preservation_matrix.png", "Heatmap of resampling preservation by condition.")


def plot_pass_fail_atlas(ctx: bench.RunContext, summary_rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    rows = list(summary_rows)
    rows.sort(key=lambda r: (r["hypothesis_id"], r["site"], int(r["depth"])))
    labels = [f"{r['hypothesis_id']} {r['site']} d{r['depth']}" for r in rows]
    cols = ["pass_preservation", "pass_damage", "pass_specificity", "formal_pass"]
    mat = np.array([[1.0 if r[c] else 0.0 for c in cols] for r in rows], dtype=float) if rows else np.zeros((1, len(cols)))
    if not labels:
        labels = ["no baseline-passing cells"]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.24 * len(labels))))
    im = ax.imshow(mat, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xticks(range(len(cols)), [c.replace("pass_", "") for c in cols], rotation=25, ha="right")
    ax.set_title("Hypothesis gate pass/fail atlas")
    fig.colorbar(im, ax=ax, shrink=0.8, ticks=[0, 1])
    fig.tight_layout()
    bench.save_figure(ctx, fig, "hypothesis_pass_fail_atlas.png", "Pass/fail atlas for formal hypothesis gates.")


def plot_specificity_ladder(ctx: bench.RunContext, best_rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [str(r["domain"]) for r in best_rows]
    cols = [
        ("preserve", "mean_preserve_variable", "#009E73"),
        ("break", "mean_break_variable", "#D55E00"),
        ("random", "mean_random_matched", "#999999"),
        ("wrong site", "mean_wrong_site_preserve", "#CC79A7"),
    ]
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, key, color) in enumerate(cols):
        ax.bar(x + (i - 1.5) * width, [as_float(r.get(key), 0.0) for r in best_rows], width, label=label, color=color)
    ax.set_xticks(x, labels)
    ax.set_ylabel("mean scrub score at best cell")
    ax.set_title("Variable specificity ladder")
    ax.axhline(1.0, color="#444444", linestyle=":", linewidth=1)
    ax.axhline(0.0, color="#444444", linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "variable_specificity_ladder.png", "Preserve-vs-break-vs-control ladder at each hypothesis best cell.")


def plot_refinement(ctx: bench.RunContext, refinement_rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    labels = [str(r["hypothesis_id"]) + ":" + str(r["version"]) for r in refinement_rows]
    fail_counts = [0 if not r.get("failed_rule") else len(str(r["failed_rule"]).split(";")) for r in refinement_rows]
    fig, ax = plt.subplots(figsize=(10, max(3, 0.4 * len(labels))))
    ax.barh(labels, fail_counts, color="#E69F00")
    ax.set_xlabel("failed or pending rules")
    ax.set_title("Hypothesis refinement trajectory")
    for y, row in enumerate(refinement_rows):
        ax.text(fail_counts[y] + 0.03, y, str(row.get("failed_rule") or "passed"), va="center", fontsize=8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "refinement_trajectory.png", "Automatic refinement log summarized by failed rules.")


def plot_counterexamples(ctx: bench.RunContext, counterexamples: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, max(4, 0.45 * max(1, min(10, len(counterexamples))))))
    ax.axis("off")
    if not counterexamples:
        ax.text(0.5, 0.5, "No automatic counterexamples crossed thresholds", ha="center", va="center")
    else:
        shown = list(counterexamples[:10])
        table_data = [
            [r["kind"], r["hypothesis_id"], r["item_id"], r["donor_id"], r["site"], r["depth"], r["scrub_score"]]
            for r in shown
        ]
        table = ax.table(
            cellText=table_data,
            colLabels=["kind", "hypothesis", "item", "donor", "site", "depth", "score"],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.35)
    ax.set_title("Counterexample gallery", pad=12)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "counterexample_gallery.png", "Top automatic counterexamples for Lab 26 hypotheses.")


def write_plots(
    ctx: bench.RunContext,
    evidence_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    best_rows: Sequence[Mapping[str, Any]],
    counterexamples: Sequence[Mapping[str, Any]],
    refinement_rows: Sequence[Mapping[str, Any]],
) -> None:
    write_plot_reading_guide(ctx)
    if ctx.args.no_plots:
        return
    plot_dashboard(ctx, evidence_rows, baseline_rows, counterexamples)
    plot_resampling_matrix(ctx, summary_rows)
    plot_pass_fail_atlas(ctx, summary_rows)
    plot_specificity_ladder(ctx, best_rows)
    plot_refinement(ctx, refinement_rows)
    plot_counterexamples(ctx, counterexamples)


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    items, data_info = load_items(ctx)
    specs, spec_audit_rows = load_specs(ctx)
    print(
        f"[lab26] loaded {data_info['n_rows_selected']}/{data_info['n_rows_file']} items "
        f"from {pathlib.Path(data_info['data_path']).name}"
    )

    manifest_path = ctx.path("diagnostics", "data_manifest.json")
    bench.write_json(manifest_path, data_info)
    ctx.register_artifact(manifest_path, "diagnostic", "Lab 26 data file hash, selected counts, and prompt-set caps.")

    bench.run_hook_parity_check(ctx, bundle, items[0].prompt)
    first_capture = bench.run_with_residual_cache(bundle, items[0].prompt)
    bench.run_lens_self_check(ctx, bundle, first_capture)
    bench.run_patch_noop_check(ctx, bundle, items[0].prompt)

    items, token_rows = tokenization_gate(ctx, bundle, items)
    captures, baseline_rows = cache_items(ctx, bundle, items)
    donor_plans, donor_rows = build_donor_plans(items)
    donor_path = ctx.path("tables", "donor_plan.csv")
    bench.write_csv_with_context(ctx, donor_path, donor_rows)
    ctx.register_artifact(donor_path, "table", "Selected preserving, broken-variable, random, wrong-site, and no-op donors.")

    resampling_rows = run_resampling(ctx, bundle, items, specs, captures, donor_plans)
    results_path = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results_path, resampling_rows)
    ctx.register_artifact(results_path, "table", "Long-form Lab 26 residual-resampling interventions.")
    interventions_path = ctx.path("tables", "resampling_interventions.csv")
    bench.write_csv_with_context(ctx, interventions_path, resampling_rows)
    ctx.register_artifact(interventions_path, "table", "Copy of long-form interventions under tables/ for notebooks and reports.")

    summary_rows, best_rows, metrics = aggregate_resampling(resampling_rows, specs)
    summary_path = ctx.path("tables", "variable_preservation_summary.csv")
    bench.write_csv_with_context(ctx, summary_path, summary_rows)
    ctx.register_artifact(summary_path, "table", "Mean scrub scores and pass/fail gates by hypothesis, site, depth, and condition.")

    best_path = ctx.path("tables", "best_hypothesis_cells.csv")
    bench.write_csv_with_context(ctx, best_path, best_rows)
    ctx.register_artifact(best_path, "table", "Best resampling cell per hypothesis according to the preregistered gate ordering.")

    counterexamples = build_counterexamples(resampling_rows, specs, best_rows)
    counter_path = ctx.path("tables", "counterexamples.csv")
    bench.write_csv_with_context(ctx, counter_path, counterexamples)
    ctx.register_artifact(counter_path, "table", "Automatic counterexamples that shrink or kill positive abstraction claims.")

    refinement_rows = build_refinement_log(best_rows, specs, counterexamples)
    refinement_path = ctx.path("tables", "hypothesis_refinement_log.csv")
    bench.write_csv_with_context(ctx, refinement_path, refinement_rows)
    ctx.register_artifact(refinement_path, "table", "Suggested v2 hypothesis refinements driven by failed gates and counterexamples.")

    evidence_rows = build_evidence_matrix(best_rows, baseline_rows, counterexamples)
    evidence_path = ctx.path("tables", "evidence_matrix.csv")
    bench.write_csv_with_context(ctx, evidence_path, evidence_rows)
    ctx.register_artifact(evidence_path, "table", "Compact Lab 26 evidence matrix for claim writing.")

    metrics.update({
        "data": data_info,
        "baseline_pass_rate_by_domain": {
            domain: safe_mean([1.0 if row["baseline_pass"] else 0.0 for row in baseline_rows if row["domain"] == domain], 0.0)
            for domain in sorted({row["domain"] for row in baseline_rows})
        },
        "n_counterexamples": len(counterexamples),
        "spec_audit": spec_audit_rows,
    })
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 26 metrics and dynamic verdicts.")

    write_method_card(ctx, bundle, metrics, evidence_rows)
    write_spec_card(ctx, specs, spec_audit_rows)
    write_operationalization_audit(ctx, evidence_rows, counterexamples)
    write_run_summary(ctx, bundle, data_info, evidence_rows, counterexamples, refinement_rows)
    write_ledger_claims(ctx, evidence_rows)
    write_plots(ctx, evidence_rows, baseline_rows, summary_rows, best_rows, counterexamples, refinement_rows)

    pass_count = sum(1 for row in evidence_rows if row["formal_pass"])
    print(
        f"[lab26] wrote run_summary.md, method_card.md, causal_abstraction_spec.md, "
        f"operationalization_audit.md, and {len(evidence_rows)} evidence rows "
        f"({pass_count} passed current formal gates)"
    )
