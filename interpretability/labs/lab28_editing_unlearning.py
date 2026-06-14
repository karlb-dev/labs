"""Lab 28: Mechanistic editing and unlearning.

This lab implements a deliberately conservative editing battery: reversible
inference-time residual additions, selected from a localization screen and
then audited on target, paraphrase, neighbor, retain, no-op, and control rows.

The word "unlearning" is treated as a scoped behavior measurement. The code
never persists weight changes, never removes facts from parameters, and never
uses unsafe refusal-ablation or private-data examples. The strongest supported
claim is a narrow CAUSAL + AUDIT sentence about a named activation edit under
these controls.
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
from collections.abc import Mapping, Sequence
from typing import Any

import interp_bench as bench

LAB_ID = "L28"
LAB_NAME = "lab28_editing_unlearning"
DATA_FILE = "editing_unlearning_targets.csv"

PROMPT_SET_CAPS = {"small": 5, "medium": 8, "full": 0}
EDIT_SCALES = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
SCIENCE_READY_MIN_ROWS = 5

MIN_BASELINE_BEFORE_MARGIN = 0.05
LOCALITY_GAP_MIN = 0.10
TARGET_GAIN_MIN = 0.10
TARGET_CONTROL_GAP_MIN = 0.10
PARAPHRASE_GAIN_MIN = 0.02
RETAIN_DAMAGE_SOFT_LIMIT = 1.50
RETAIN_MAX_DAMAGE_LIMIT = 3.00
NEIGHBOR_DAMAGE_CAVEAT = 2.50
NOOP_DELTA_ATOL = 1e-4
REVERSIBILITY_ATOL = 1e-4
RESIDUAL_PATCH_BATCH_SIZE = 48

LOCALIZATION_METHODS = (
    "self_patch_noop",
    "localized_patch",
    "wrong_position_patch",
    "random_direction_patch",
)
EDIT_METHODS = (
    "no_edit",
    "localized_addition",
    "wrong_position_addition",
    "random_direction_addition",
    "opposite_direction_addition",
)
CONTROL_EDIT_METHODS = (
    "wrong_position_addition",
    "random_direction_addition",
    "opposite_direction_addition",
)

PLOT_SOURCE_SUBDIR = "figure_sources"
CI_Z = 1.96

EDIT_METHOD_DISPLAY = {
    "no_edit": "baseline",
    "localized_addition": "localized add",
    "wrong_position_addition": "wrong position",
    "random_direction_addition": "random direction",
    "opposite_direction_addition": "opposite sign",
}

LOCALIZATION_METHOD_DISPLAY = {
    "self_patch_noop": "self patch",
    "localized_patch": "localized patch",
    "wrong_position_patch": "wrong position",
    "random_direction_patch": "random direction",
}

REQUIRED_COLUMNS = {
    "target_id",
    "family",
    "edit_type",
    "prompt",
    "target_before",
    "target_after",
    "retain_prompts_json",
    "paraphrase_prompts_json",
    "neighbor_prompts_json",
    "safety_notes",
}

SAFETY_BLOCKLIST = (
    "password",
    "credential",
    "secret key",
    "api key",
    "private key",
    "ssn",
    "social security",
    "jailbreak",
    "refusal ablation",
    "bypass",
    "explosive",
    "weapon",
    "malware",
    "evade detection",
    "personal data",
)


@dataclasses.dataclass
class EvalPrompt:
    prompt: str
    target: str
    distractor: str
    role: str = ""


@dataclasses.dataclass
class EditTarget:
    target_id: str
    family: str
    edit_type: str
    prompt: str
    target_before: str
    target_after: str
    retain_prompts: list[EvalPrompt]
    paraphrase_prompts: list[EvalPrompt]
    neighbor_prompts: list[EvalPrompt]
    safety_notes: str
    donor_prompt: str
    row_index: int
    before_id: int = -1
    after_id: int = -1
    raw_input_ids: list[int] = dataclasses.field(default_factory=list)
    input_ids: list[int] = dataclasses.field(default_factory=list)
    donor_input_ids: list[int] = dataclasses.field(default_factory=list)
    token_text: list[str] = dataclasses.field(default_factory=list)
    donor_token_text: list[str] = dataclasses.field(default_factory=list)
    final_pos: int = -1
    donor_final_pos: int = -1
    base_after_margin: float = float("nan")
    donor_after_margin: float = float("nan")


@dataclasses.dataclass(frozen=True)
class EditVector:
    target_id: str
    depth: int
    final_pos: int
    wrong_pos: int
    direction: Any
    random_direction: Any
    direction_norm: float


@dataclasses.dataclass(frozen=True)
class AdditionJob:
    row: dict[str, Any]
    layer: int
    position: int
    vector: Any
    scale: float


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


def rounded(value: Any, digits: int = 4) -> Any:
    try:
        f = float(value)
    except Exception:
        return value
    return round(f, digits) if math.isfinite(f) else ""


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


def safe_max(values: Sequence[Any], default: float = float("nan")) -> float:
    vals = [as_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return max(vals) if vals else default


def safe_corr(xs: Sequence[Any], ys: Sequence[Any]) -> float:
    pairs: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        xf, yf = as_float(x), as_float(y)
        if math.isfinite(xf) and math.isfinite(yf):
            pairs.append((xf, yf))
    if len(pairs) < 2:
        return float("nan")
    xbar = statistics.fmean(x for x, _ in pairs)
    ybar = statistics.fmean(y for _, y in pairs)
    num = sum((x - xbar) * (y - ybar) for x, y in pairs)
    denx = math.sqrt(sum((x - xbar) ** 2 for x, _ in pairs))
    deny = math.sqrt(sum((y - ybar) ** 2 for _, y in pairs))
    return num / (denx * deny) if denx > 1e-12 and deny > 1e-12 else float("nan")


def stable_id(*parts: Any, prefix: str = "") -> str:
    payload = json.dumps([str(part) for part in parts], sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:14]
    return f"{prefix}{digest}" if prefix else digest


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "ok"}


def finite_values(values: Sequence[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        f = as_float(value)
        if math.isfinite(f):
            out.append(f)
    return out


def safe_stderr(values: Sequence[Any], default: float = float("nan")) -> float:
    vals = finite_values(values)
    if len(vals) < 2:
        return default
    return float(statistics.stdev(vals) / math.sqrt(len(vals)))


def safe_ci_halfwidth(values: Sequence[Any], default: float = float("nan")) -> float:
    err = safe_stderr(values, default=default)
    return CI_Z * err if math.isfinite(err) else default


def method_label(method: Any) -> str:
    return EDIT_METHOD_DISPLAY.get(str(method), str(method).replace("_", " "))


def localization_method_label(method: Any) -> str:
    return LOCALIZATION_METHOD_DISPLAY.get(str(method), str(method).replace("_", " "))


def short_target_label(target_id: Any, max_len: int = 24) -> str:
    label = str(target_id).replace("edit_", "").replace("smoke_", "")
    label = label.replace("_", " ")
    return label[: max_len - 1] + "…" if len(label) > max_len else label


def figure_source_path(ctx: bench.RunContext, name: str) -> pathlib.Path:
    return ctx.path("tables", PLOT_SOURCE_SUBDIR, name)


def write_figure_source(
    ctx: bench.RunContext,
    name: str,
    rows: Sequence[Mapping[str, Any]],
    description: str,
) -> dict[str, Any]:
    path = figure_source_path(ctx, name)
    out_rows = [dict(row) for row in rows]
    if not out_rows:
        out_rows = [{"warning": "no_rows_available_for_this_figure"}]
    bench.write_csv_with_context(ctx, path, out_rows)
    ctx.register_artifact(path, "table", description)
    return {"path": str(path.relative_to(ctx.run_dir)), "row_count": len(rows), "description": description}


def write_jsonl(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), sort_keys=True, default=bench.json_default) + "\n")


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def logit_margin(logits: Any, target_id: int, distractor_id: int) -> float:
    return float(logits[target_id] - logits[distractor_id])


def contains_blocklisted_text(text: str) -> list[str]:
    lo = text.lower()
    return [term for term in SAFETY_BLOCKLIST if term in lo]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def data_path(args: Any) -> pathlib.Path:
    prompt_set = str(getattr(args, "prompt_set", "") or "")
    candidate = pathlib.Path(prompt_set)
    if prompt_set not in PROMPT_SET_CAPS and candidate.suffix.lower() in {".csv", ".tsv"}:
        return candidate if candidate.is_absolute() else (bench.COURSE_ROOT / candidate).resolve()
    return bench.COURSE_ROOT / "data" / DATA_FILE


def parse_eval_prompts(raw: str, *, default_target: str, default_distractor: str) -> list[EvalPrompt]:
    try:
        payload = json.loads(raw or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Bad eval prompt JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise ValueError("Eval prompt JSON must decode to a list.")
    prompts: list[EvalPrompt] = []
    for i, item in enumerate(payload):
        if isinstance(item, str):
            prompts.append(EvalPrompt(prompt=item, target=default_target, distractor=default_distractor, role=f"prompt_{i}"))
            continue
        if not isinstance(item, Mapping):
            raise ValueError(f"Eval prompt entry {i} must be a string or object.")
        prompts.append(EvalPrompt(
            prompt=str(item["prompt"]),
            target=str(item.get("target", default_target)),
            distractor=str(item.get("distractor", default_distractor)),
            role=str(item.get("role", f"prompt_{i}")),
        ))
    return prompts


def builtin_smoke_rows() -> list[dict[str, str]]:
    """Tiny Tier A fallback for plumbing when the frozen CSV is absent."""
    rows: list[dict[str, str]] = []

    def add(
        target_id: str,
        family: str,
        edit_type: str,
        prompt: str,
        before: str,
        after: str,
        donor: str,
        retain: list[dict[str, str]],
        paraphrase: list[dict[str, str]],
        neighbor: list[dict[str, str]],
        notes: str,
    ) -> None:
        rows.append({
            "target_id": target_id,
            "family": family,
            "edit_type": edit_type,
            "prompt": prompt,
            "target_before": before,
            "target_after": after,
            "retain_prompts_json": json.dumps(retain, sort_keys=True),
            "paraphrase_prompts_json": json.dumps(paraphrase, sort_keys=True),
            "neighbor_prompts_json": json.dumps(neighbor, sort_keys=True),
            "safety_notes": notes,
            "donor_prompt": donor,
        })

    add(
        "smoke_capital_france_to_rome",
        "capital_counterfactual",
        "counterfactual_activation_edit",
        "The capital of France is",
        " Paris",
        " Rome",
        "The capital of Italy is",
        [
            {"prompt": "The capital of Germany is", "target": " Berlin", "distractor": " Rome", "role": "other_capital"},
            {"prompt": "The capital of Spain is", "target": " Madrid", "distractor": " Rome", "role": "other_capital"},
        ],
        [
            {"prompt": "In France, the capital city is", "role": "paraphrase"},
            {"prompt": "France has its capital in", "role": "paraphrase"},
        ],
        [
            {"prompt": "The language of France is", "target": " French", "distractor": " Rome", "role": "same_subject_other_relation"},
            {"prompt": "The capital of Italy is", "target": " Rome", "distractor": " Paris", "role": "donor_fact"},
        ],
        "Benign public country-capital association; inference-time activation edit only.",
    )
    add(
        "smoke_color_alice_blue_to_green",
        "synthetic_codeword",
        "synthetic_counterfactual_activation_edit",
        "The code word for Alice is",
        " blue",
        " green",
        "The code word for Bob is",
        [
            {"prompt": "The code word for Carol is", "target": " red", "distractor": " green", "role": "other_synthetic"},
            {"prompt": "The code word for Dave is", "target": " yellow", "distractor": " green", "role": "other_synthetic"},
        ],
        [
            {"prompt": "Alice's code word is", "role": "paraphrase"},
            {"prompt": "For Alice, the code word is", "role": "paraphrase"},
        ],
        [
            {"prompt": "The code word for Bob is", "target": " green", "distractor": " blue", "role": "donor_synthetic"},
        ],
        "Synthetic toy association; not real private data; inference-time activation edit only.",
    )
    return rows


def row_to_target(row: Mapping[str, str], row_index: int) -> EditTarget:
    before = str(row["target_before"])
    after = str(row["target_after"])
    return EditTarget(
        target_id=str(row["target_id"]).strip(),
        family=str(row["family"]).strip(),
        edit_type=str(row["edit_type"]).strip(),
        prompt=str(row["prompt"]),
        target_before=before,
        target_after=after,
        retain_prompts=parse_eval_prompts(str(row.get("retain_prompts_json", "[]")), default_target=before, default_distractor=after),
        paraphrase_prompts=parse_eval_prompts(str(row.get("paraphrase_prompts_json", "[]")), default_target=after, default_distractor=before),
        neighbor_prompts=parse_eval_prompts(str(row.get("neighbor_prompts_json", "[]")), default_target=before, default_distractor=after),
        safety_notes=str(row.get("safety_notes", "")),
        donor_prompt=str(row.get("donor_prompt") or row["prompt"]),
        row_index=row_index,
    )


def load_targets(ctx: bench.RunContext) -> tuple[list[EditTarget], dict[str, Any]]:
    path = data_path(ctx.args)
    source = "frozen_csv"
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            rows = [dict(row) for row in csv.DictReader(f)]
        digest = file_sha256(path)
    else:
        if str(getattr(ctx.args, "tier", "")).lower() != "a":
            raise FileNotFoundError(f"Lab 28 data file not found: {path}. Tier B/C science runs require the frozen CSV.")
        print("[lab28] data CSV missing; using builtin Tier A smoke fallback. Do not ledger science claims from this run.")
        rows = builtin_smoke_rows()
        source = "builtin_tier_a_smoke_fallback"
        digest = hashlib.sha256("\n".join(row["target_id"] for row in rows).encode("utf-8")).hexdigest()

    if rows:
        missing = sorted(REQUIRED_COLUMNS - set(rows[0]))
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
    targets = [row_to_target(row, i) for i, row in enumerate(rows)]
    if len({t.target_id for t in targets}) != len(targets):
        raise ValueError("Lab 28 target_id values must be unique.")

    prompt_set = str(getattr(ctx.args, "prompt_set", "") or "")
    cap = PROMPT_SET_CAPS.get(prompt_set, 0)
    if cap:
        targets = targets[:cap]
    max_examples = int(getattr(ctx.args, "max_examples", 0) or 0)
    if max_examples > 0:
        targets = targets[:max_examples]

    info = {
        "data_source": source,
        "science_ready_data": source == "frozen_csv" and len(targets) >= SCIENCE_READY_MIN_ROWS,
        "data_path": str(path),
        "sha256": digest,
        "n_rows_file": len(rows),
        "n_rows_selected": len(targets),
        "families": {f: sum(1 for t in targets if t.family == f) for f in sorted({t.family for t in targets})},
        "edit_types": {e: sum(1 for t in targets if t.edit_type == e) for e in sorted({t.edit_type for t in targets})},
        "prompt_set": ctx.args.prompt_set,
        "max_examples": ctx.args.max_examples,
        "safe_scope": "benign_public_or_synthetic_associations; inference_time_only; no_persistent_weight_edits",
        "tier_a_fallback_warning": source.startswith("builtin"),
    }
    if not targets:
        raise RuntimeError("Lab 28 selected zero targets.")
    return targets, info


# ---------------------------------------------------------------------------
# Validation and safety gates
# ---------------------------------------------------------------------------


def write_safety_status(ctx: bench.RunContext, targets: Sequence[EditTarget], data_info: Mapping[str, Any]) -> dict[str, Any]:
    blocked_rows = []
    for target in targets:
        blobs = [target.prompt, target.donor_prompt, target.safety_notes]
        blobs += [ep.prompt for ep in target.retain_prompts + target.paraphrase_prompts + target.neighbor_prompts]
        hits = sorted({hit for text in blobs for hit in contains_blocklisted_text(text)})
        if hits:
            blocked_rows.append({"target_id": target.target_id, "hits": hits})
    status = {
        "lab": LAB_NAME,
        "unsafe_prompt_sampling": False,
        "refusal_ablation": False,
        "harmful_completion_generation": False,
        "persistent_weight_edit": False,
        "private_data_unlearning": False,
        "public_private_boundary_relevant": False,
        "blocked_rows": len(blocked_rows),
        "blocked_row_details": blocked_rows,
        "science_ready": bool(data_info.get("science_ready_data")) and not blocked_rows,
        "scope": "benign public facts, toy relations, and synthetic associations only",
    }
    path = ctx.path("diagnostics", "safety_status.json")
    bench.write_json(path, status)
    ctx.register_artifact(path, "diagnostic", "Safety wall status for Lab 28 targets and methods.")
    if blocked_rows:
        raise RuntimeError(f"Lab 28 safety gate blocked {len(blocked_rows)} rows; see diagnostics/safety_status.json")
    return status


def tokenization_gate(ctx: bench.RunContext, bundle: bench.ModelBundle, targets: list[EditTarget]) -> tuple[list[EditTarget], list[dict[str, Any]]]:
    tokenizer = bundle.tokenizer
    kept: list[EditTarget] = []
    rows: list[dict[str, Any]] = []
    for target in targets:
        problems: list[str] = []
        warnings: list[str] = []
        before_ids = token_ids(tokenizer, target.target_before)
        after_ids = token_ids(tokenizer, target.target_after)
        raw_prompt_ids = tokenizer.encode(target.prompt, add_special_tokens=False)
        full_prompt_ids = list(tokenizer(target.prompt, add_special_tokens=True)["input_ids"])
        full_donor_ids = list(tokenizer(target.donor_prompt, add_special_tokens=True)["input_ids"])
        if len(before_ids) != 1:
            problems.append(f"target_before_token_count={len(before_ids)}")
        if len(after_ids) != 1:
            problems.append(f"target_after_token_count={len(after_ids)}")
        if before_ids and after_ids and before_ids == after_ids:
            problems.append("before_equals_after_token")
        if not full_prompt_ids:
            problems.append("empty_prompt")
        if not full_donor_ids:
            problems.append("empty_donor_prompt")
        if len(full_prompt_ids) != len(raw_prompt_ids):
            warnings.append(f"special_token_count_delta={len(full_prompt_ids) - len(raw_prompt_ids)}")
        eval_bad = 0
        eval_total = 0
        for collection_name, collection in (
            ("retain", target.retain_prompts),
            ("paraphrase", target.paraphrase_prompts),
            ("neighbor", target.neighbor_prompts),
        ):
            for ep in collection:
                eval_total += 1
                t_ids = token_ids(tokenizer, ep.target)
                d_ids = token_ids(tokenizer, ep.distractor)
                if len(t_ids) != 1 or len(d_ids) != 1 or t_ids == d_ids:
                    eval_bad += 1
                    warnings.append(f"{collection_name}:{ep.role}:target_or_distractor_token_bad")
        if eval_bad:
            problems.append(f"eval_prompt_token_failures={eval_bad}")
        if not target.paraphrase_prompts:
            warnings.append("no_paraphrase_prompts")
        if not target.retain_prompts:
            warnings.append("no_retain_prompts")
        if not problems:
            target.before_id = int(before_ids[0])
            target.after_id = int(after_ids[0])
            target.raw_input_ids = list(raw_prompt_ids)
            target.input_ids = list(full_prompt_ids)
            target.donor_input_ids = list(full_donor_ids)
            target.token_text = [tokenizer.decode([tid]) for tid in full_prompt_ids]
            target.donor_token_text = [tokenizer.decode([tid]) for tid in full_donor_ids]
            target.final_pos = len(full_prompt_ids) - 1
            target.donor_final_pos = len(full_donor_ids) - 1
            kept.append(target)
        rows.append({
            "target_id": target.target_id,
            "family": target.family,
            "edit_type": target.edit_type,
            "prompt": target.prompt,
            "donor_prompt": target.donor_prompt,
            "target_before": target.target_before,
            "target_before_token_count": len(before_ids),
            "target_after": target.target_after,
            "target_after_token_count": len(after_ids),
            "prompt_raw_tokens": len(raw_prompt_ids),
            "prompt_forward_tokens": len(full_prompt_ids),
            "donor_forward_tokens": len(full_donor_ids),
            "final_pos": len(full_prompt_ids) - 1 if full_prompt_ids else "",
            "donor_final_pos": len(full_donor_ids) - 1 if full_donor_ids else "",
            "eval_prompts": eval_total,
            "retain_prompts": len(target.retain_prompts),
            "paraphrase_prompts": len(target.paraphrase_prompts),
            "neighbor_prompts": len(target.neighbor_prompts),
            "kept": not problems,
            "problems": ";".join(problems),
            "warnings": ";".join(warnings),
            "prompt_tokenization": " | ".join(f"{i}:{tokenizer.decode([tid])!r}" for i, tid in enumerate(full_prompt_ids)),
            "donor_tokenization": " | ".join(f"{i}:{tokenizer.decode([tid])!r}" for i, tid in enumerate(full_donor_ids)),
            "before_id": before_ids[0] if len(before_ids) == 1 else "",
            "after_id": after_ids[0] if len(after_ids) == 1 else "",
        })
    path = ctx.path("diagnostics", "tokenization_gate.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "diagnostic", "Single-token answer and prompt-tokenization audit for Lab 28 targets.")
    if not kept:
        raise RuntimeError("Lab 28 tokenization gate dropped every target.")
    print(f"[lab28] tokenization gate kept {len(kept)}/{len(targets)} targets")
    return kept, rows


# ---------------------------------------------------------------------------
# Residual interventions
# ---------------------------------------------------------------------------


def coarse_depths(n_layers: int, prompt_set: str) -> list[int]:
    if prompt_set == "full":
        return list(range(n_layers + 1))
    return sorted({0, max(1, n_layers // 4), max(1, n_layers // 2), max(1, (3 * n_layers) // 4), n_layers})


def claimable_depth(bundle: bench.ModelBundle, depth: int) -> bool:
    return 0 < int(depth) < bundle.anatomy.n_layers


def wrong_position_for_length(length: int, final_pos: int) -> int:
    if length <= 1:
        return 0
    return 0 if final_pos != 0 else 1


def deterministic_random_like(vector: Any, key: str) -> Any:
    import torch

    gen = torch.Generator(device="cpu")
    gen.manual_seed(stable_int(key) % (2**31 - 1))
    rand = torch.randn(vector.shape, generator=gen, dtype=vector.dtype)
    norm = vector.float().norm().clamp_min(1e-8)
    return rand / rand.float().norm().clamp_min(1e-8) * norm


def run_with_residual_addition(
    bundle: bench.ModelBundle,
    prompt: str,
    layer: int,
    position: int,
    vector: Any,
    scale: float,
) -> Any:
    """Add a vector to streams[layer][position] for one forward pass.

    This matches the bench residual-patching convention. For layer L, the hook
    runs at the final norm input. For layer k < L, it runs at block k input.
    """
    n_layers = bundle.anatomy.n_layers
    if not 0 <= layer <= n_layers:
        raise ValueError(f"stream layer must be in [0, {n_layers}], got {layer}")
    module = bundle.final_norm if layer == n_layers else bundle.blocks[layer]

    def add_hook(mod: Any, hook_args: tuple) -> Any:
        hidden = hook_args[0].clone()
        if not -hidden.shape[1] <= position < hidden.shape[1]:
            raise ValueError(f"edit position {position} out of range for sequence length {hidden.shape[1]}")
        hidden[0, position] = hidden[0, position] + float(scale) * vector.to(hidden.device, hidden.dtype)
        return (hidden,) + tuple(hook_args[1:])

    return bench._forward_logits(bundle, prompt, [(module, add_hook)])


def run_addition_jobs(bundle: bench.ModelBundle, prompt: str, jobs: Sequence[AdditionJob]) -> list[Any]:
    """Run addition jobs. Kept simple and explicit for pedagogy.

    The bench has batched replacement patching, but addition hooks carry a
    scale and vector per row; individual forwards keep the intervention easy to
    audit and the Lab 28 target set is intentionally small.
    """
    out = []
    for job in jobs:
        out.append(run_with_residual_addition(bundle, prompt, job.layer, job.position, job.vector, job.scale))
    return out


# ---------------------------------------------------------------------------
# Baselines and localization
# ---------------------------------------------------------------------------


def cache_baselines(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    targets: list[EditTarget],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    target_caps: dict[str, Any] = {}
    donor_caps: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for i, target in enumerate(targets):
        cap = bench.run_with_residual_cache(bundle, target.prompt)
        donor = bench.run_with_residual_cache(bundle, target.donor_prompt)
        if cap.input_ids != target.input_ids:
            raise RuntimeError(f"{target.target_id}: forward tokenization drifted from tokenization gate.")
        if donor.input_ids != target.donor_input_ids:
            raise RuntimeError(f"{target.target_id}: donor forward tokenization drifted from tokenization gate.")
        target_caps[target.target_id] = cap
        donor_caps[target.target_id] = donor
        target.final_pos = len(cap.input_ids) - 1
        target.donor_final_pos = len(donor.input_ids) - 1
        target.base_after_margin = logit_margin(cap.final_logits_last, target.after_id, target.before_id)
        target.donor_after_margin = logit_margin(donor.final_logits_last, target.after_id, target.before_id)
        rows.append({
            "target_id": target.target_id,
            "family": target.family,
            "edit_type": target.edit_type,
            "prompt": target.prompt,
            "donor_prompt": target.donor_prompt,
            "target_before": target.target_before,
            "target_after": target.target_after,
            "base_after_minus_before": rounded(target.base_after_margin),
            "base_before_minus_after": rounded(-target.base_after_margin),
            "donor_after_minus_before": rounded(target.donor_after_margin),
            "baseline_prefers_before": target.base_after_margin < -MIN_BASELINE_BEFORE_MARGIN,
            "baseline_already_after": target.base_after_margin > 0.0,
            "donor_supports_after": target.donor_after_margin > 0.0,
            "base_top": bundle.tokenizer.decode([int(cap.final_logits_last.argmax())]),
            "donor_top": bundle.tokenizer.decode([int(donor.final_logits_last.argmax())]),
            "target_prompt_tokens": len(cap.input_ids),
            "donor_prompt_tokens": len(donor.input_ids),
        })
        if (i + 1) % max(1, len(targets) // 3) == 0 or i + 1 == len(targets):
            print(f"[lab28] cached baselines {i + 1}/{len(targets)}")
    path = ctx.path("tables", "baseline_behavior.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Target and donor baseline margins before any edit.")
    return target_caps, donor_caps, rows


def localize_sites(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    targets: list[EditTarget],
    target_caps: Mapping[str, Any],
    donor_caps: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    depths = coarse_depths(bundle.anatomy.n_layers, str(ctx.args.prompt_set))
    rows: list[dict[str, Any]] = []
    by_target_depth: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    total = len(targets) * len(depths) * len(LOCALIZATION_METHODS)
    done = 0
    report_every = max(1, total // 5)
    for target in targets:
        cap = target_caps[target.target_id]
        donor = donor_caps[target.target_id]
        wrong_pos = wrong_position_for_length(len(target.input_ids), target.final_pos)
        for depth in depths:
            donor_vec = donor.streams[depth, target.donor_final_pos]
            target_vec = cap.streams[depth, target.final_pos]
            direction = donor_vec - target_vec
            random_vec = target_vec + deterministic_random_like(direction, f"{target.target_id}|{depth}|localize")
            cells = [
                (int(depth), int(target.final_pos), target_vec),
                (int(depth), int(target.final_pos), donor_vec),
                (int(depth), int(wrong_pos), donor_vec),
                (int(depth), int(target.final_pos), random_vec),
            ]
            logits_list = bench.run_with_residual_patch_batched(
                bundle, target.prompt, cells, max_batch=RESIDUAL_PATCH_BATCH_SIZE
            )
            for method, logits, position in zip(LOCALIZATION_METHODS, logits_list, (target.final_pos, target.final_pos, wrong_pos, target.final_pos)):
                margin = logit_margin(logits, target.after_id, target.before_id)
                gain = margin - target.base_after_margin
                noop_abs_delta = abs(gain) if method == "self_patch_noop" else ""
                row = {
                    "target_id": target.target_id,
                    "family": target.family,
                    "method": method,
                    "depth": depth,
                    "claimable_depth": claimable_depth(bundle, int(depth)),
                    "position": position,
                    "base_after_minus_before": rounded(target.base_after_margin),
                    "patched_after_minus_before": rounded(margin),
                    "patch_gain": rounded(gain),
                    "noop_abs_delta": rounded(noop_abs_delta) if noop_abs_delta != "" else "",
                    "localized_candidate": method == "localized_patch",
                }
                rows.append(row)
                by_target_depth[(target.target_id, int(depth))][method] = gain
                done += 1
                if done % report_every == 0 or done == total:
                    print(f"[lab28] localization interventions {done}/{total}")
    # Add control gaps and select best claimable depth per target.
    for row in rows:
        vals = by_target_depth[(str(row["target_id"]), int(row["depth"]))]
        control_floor = max(vals.get("wrong_position_patch", float("nan")), vals.get("random_direction_patch", float("nan")))
        loc = vals.get("localized_patch", float("nan"))
        row["localization_control_floor"] = rounded(control_floor)
        row["localization_gap"] = rounded(loc - control_floor) if math.isfinite(loc) and math.isfinite(control_floor) else ""
    best_by_target: dict[str, dict[str, Any]] = {}
    for target in targets:
        candidates = [
            row for row in rows
            if row["target_id"] == target.target_id and row["method"] == "localized_patch" and row["claimable_depth"]
        ]
        if not candidates:
            candidates = [row for row in rows if row["target_id"] == target.target_id and row["method"] == "localized_patch"]
        if not candidates:
            continue

        def key(row: Mapping[str, Any]) -> tuple[float, float, float]:
            gap = as_float(row.get("localization_gap"), -999.0)
            gain = as_float(row.get("patch_gain"), -999.0)
            depth = abs(int(row["depth"]) - bundle.anatomy.n_layers / 2)
            return (gap, gain, -depth)

        best = dict(max(candidates, key=key))
        best["selection_rule"] = "claimable_depth_then_max_localization_gap_then_gain_then_mid_depth"
        best_by_target[target.target_id] = best
    path = ctx.path("tables", "localization_candidates.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Residual donor-patch localization candidates, controls, and no-op rows.")
    return rows, best_by_target


def write_noop_identity_checks(ctx: bench.RunContext, localization_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in localization_rows:
        if row.get("method") == "self_patch_noop":
            val = as_float(row.get("noop_abs_delta"))
            if math.isfinite(val):
                grouped[f"{row['target_id']}|d{row['depth']}"] .append(val)
    out: list[dict[str, Any]] = []
    for key, vals in sorted(grouped.items()):
        target_id, depth = key.split("|d")
        out.append({
            "target_id": target_id,
            "depth": int(depth),
            "n": len(vals),
            "mean_abs_delta": rounded(safe_mean(vals)),
            "max_abs_delta": rounded(max(vals) if vals else float("nan")),
            "atol": NOOP_DELTA_ATOL,
            "ok": (max(vals) if vals else float("inf")) <= NOOP_DELTA_ATOL,
        })
    path = ctx.path("tables", "edit_noop_identity_check.csv")
    bench.write_csv_with_context(ctx, path, out)
    ctx.register_artifact(path, "table", "Self-patching identity check at every localization depth.")
    worst = max([as_float(row.get("max_abs_delta"), 0.0) for row in out] or [0.0])
    if worst > NOOP_DELTA_ATOL:
        raise RuntimeError(f"Lab 28 self-patching no-op check failed: {worst:.3g} > {NOOP_DELTA_ATOL}")
    return out


# ---------------------------------------------------------------------------
# Editing and side-set evaluation
# ---------------------------------------------------------------------------


def score_prompt_with_method(
    bundle: bench.ModelBundle,
    prompt: str,
    target_id: int,
    distractor_id: int,
    *,
    method: str,
    depth: int,
    position: int,
    direction: Any,
    random_direction: Any,
    wrong_position: int,
    scale: float,
) -> float:
    if method == "no_edit" or abs(float(scale)) < 1e-12:
        logits = bench.run_with_residual_cache(bundle, prompt).final_logits_last
    elif method == "localized_addition":
        logits = run_with_residual_addition(bundle, prompt, depth, position, direction, scale)
    elif method == "wrong_position_addition":
        logits = run_with_residual_addition(bundle, prompt, depth, wrong_position, direction, scale)
    elif method == "random_direction_addition":
        logits = run_with_residual_addition(bundle, prompt, depth, position, random_direction, scale)
    elif method == "opposite_direction_addition":
        logits = run_with_residual_addition(bundle, prompt, depth, position, direction, -scale)
    else:
        raise ValueError(f"unknown edit method {method}")
    return logit_margin(logits, target_id, distractor_id)


def run_edits(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    targets: list[EditTarget],
    target_caps: Mapping[str, Any],
    donor_caps: Mapping[str, Any],
    best_by_target: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, EditVector], dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    vectors: dict[str, EditVector] = {}
    metadata: dict[str, Any] = {
        "lab": LAB_ID,
        "method": "reversible inference-time residual activation addition",
        "scale_semantics": "scale * (donor_final_stream - target_final_stream) at selected stream depth",
        "non_methods": {
            "persistent_weight_edit": "not run; requires apply/restore/hash/rollback plumbing",
            "refusal_ablation": "out of scope and forbidden",
            "private_data_unlearning": "out of scope and forbidden",
            "feature_clamp_suppression": "future extension requiring a validated feature dictionary",
        },
        "targets": {},
    }
    total = len(targets) * (1 + (len(EDIT_METHODS) - 1) * (len(EDIT_SCALES) - 1))
    done = 0
    report_every = max(1, total // 5)
    for target in targets:
        best = best_by_target[target.target_id]
        depth = int(best["depth"])
        cap = target_caps[target.target_id]
        donor = donor_caps[target.target_id]
        direction = (donor.streams[depth, target.donor_final_pos] - cap.streams[depth, target.final_pos]).float().cpu()
        random_direction = deterministic_random_like(direction, f"{target.target_id}|{depth}|edit").float().cpu()
        wrong_pos = wrong_position_for_length(len(target.input_ids), target.final_pos)
        direction_norm = float(direction.norm())
        vectors[target.target_id] = EditVector(
            target_id=target.target_id,
            depth=depth,
            final_pos=target.final_pos,
            wrong_pos=wrong_pos,
            direction=direction,
            random_direction=random_direction,
            direction_norm=direction_norm,
        )
        metadata["targets"][target.target_id] = {
            "family": target.family,
            "prompt": target.prompt,
            "donor_prompt": target.donor_prompt,
            "target_before": target.target_before,
            "target_after": target.target_after,
            "depth": depth,
            "claimable_depth": claimable_depth(bundle, depth),
            "target_position": target.final_pos,
            "wrong_position": wrong_pos,
            "direction_norm": direction_norm,
            "base_after_minus_before": float(target.base_after_margin),
            "donor_after_minus_before": float(target.donor_after_margin),
        }
        for method in EDIT_METHODS:
            scales = (0.0,) if method == "no_edit" else tuple(s for s in EDIT_SCALES if s > 0.0)
            for scale in scales:
                margin = score_prompt_with_method(
                    bundle,
                    target.prompt,
                    target.after_id,
                    target.before_id,
                    method=method,
                    depth=depth,
                    position=target.final_pos,
                    direction=direction,
                    random_direction=random_direction,
                    wrong_position=wrong_pos,
                    scale=scale,
                )
                rows.append({
                    "target_id": target.target_id,
                    "family": target.family,
                    "edit_type": target.edit_type,
                    "method": method,
                    "depth": depth,
                    "claimable_depth": claimable_depth(bundle, depth),
                    "position": target.final_pos if method != "wrong_position_addition" else wrong_pos,
                    "scale": scale,
                    "direction_norm": rounded(direction_norm),
                    "base_after_minus_before": rounded(target.base_after_margin),
                    "edited_after_minus_before": rounded(margin),
                    "target_gain": rounded(margin - target.base_after_margin),
                    "changed_to_after": margin > 0.0,
                    "safe_reversible": True,
                    "weight_edit": False,
                })
                done += 1
                if done % report_every == 0 or done == total:
                    print(f"[lab28] edit interventions {done}/{total}")
    path = ctx.path("tables", "editing_results.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Main target edit results across localized and control methods.")
    state_path = ctx.path("state", "edit_vectors.pt")
    torch.save({
        target_id: {
            "direction": vec.direction,
            "random_direction": vec.random_direction,
            "depth": vec.depth,
            "target_position": vec.final_pos,
            "wrong_position": vec.wrong_pos,
            "direction_norm": vec.direction_norm,
        }
        for target_id, vec in vectors.items()
    }, state_path)
    ctx.register_artifact(state_path, "state", "Residual edit directions and deterministic random controls.")
    meta_path = ctx.path("state", "edit_metadata.json")
    bench.write_json(meta_path, metadata)
    ctx.register_artifact(meta_path, "state", "Human-readable metadata for Lab 28 edit vectors.")
    return rows, vectors, metadata


def control_floor_for_scale(edit_rows: Sequence[Mapping[str, Any]], target_id: str, scale: float) -> float:
    vals = [
        as_float(row.get("target_gain"))
        for row in edit_rows
        if row.get("target_id") == target_id
        and row.get("method") in CONTROL_EDIT_METHODS
        and abs(as_float(row.get("scale")) - scale) < 1e-9
    ]
    return safe_max(vals, default=float("nan"))


def choose_scales(
    ctx: bench.RunContext,
    targets: Sequence[EditTarget],
    edit_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    scale_rows: list[dict[str, Any]] = []
    for target in targets:
        candidates = [
            row for row in edit_rows
            if row["target_id"] == target.target_id and row["method"] == "localized_addition" and as_float(row.get("scale")) > 0.0
        ]
        if not candidates:
            chosen[target.target_id] = {"scale": 0.0, "reason": "no_localized_addition_rows"}
            continue
        annotated = []
        for row in candidates:
            scale = as_float(row["scale"])
            gain = as_float(row.get("target_gain"))
            floor = control_floor_for_scale(edit_rows, target.target_id, scale)
            gap = gain - floor if math.isfinite(gain) and math.isfinite(floor) else float("nan")
            ok = math.isfinite(gap) and gain >= TARGET_GAIN_MIN and gap >= TARGET_CONTROL_GAP_MIN
            ann = {**dict(row), "target_control_floor": floor, "target_control_gap": gap, "scale_passes_target_gate": ok}
            annotated.append(ann)
            scale_rows.append({
                "target_id": target.target_id,
                "family": target.family,
                "scale": scale,
                "localized_target_gain": rounded(gain),
                "target_control_floor": rounded(floor),
                "target_control_gap": rounded(gap),
                "passes_target_gate": ok,
                "changed_to_after": row.get("changed_to_after"),
            })
        passing = [row for row in annotated if row["scale_passes_target_gate"]]
        if passing:
            selected = min(passing, key=lambda r: (as_float(r["scale"]), -as_float(r.get("target_control_gap"))))
            reason = "smallest_scale_passing_target_gain_and_control_gap"
        else:
            selected = max(annotated, key=lambda r: (as_float(r.get("target_control_gap"), -999.0), as_float(r.get("target_gain"), -999.0)))
            reason = "no_scale_passed_target_gate_chose_best_control_gap"
        chosen[target.target_id] = {
            "scale": as_float(selected["scale"]),
            "reason": reason,
            "localized_target_gain": as_float(selected.get("target_gain")),
            "target_control_floor": as_float(selected.get("target_control_floor")),
            "target_control_gap": as_float(selected.get("target_control_gap")),
            "changed_to_after": bool(selected.get("changed_to_after")),
        }
    path = ctx.path("tables", "scale_selection.csv")
    bench.write_csv_with_context(ctx, path, scale_rows)
    ctx.register_artifact(path, "table", "Predeclared target-gate scale selection before side-set audits.")
    return chosen


def eval_ids(bundle: bench.ModelBundle, ep: EvalPrompt) -> tuple[int, int]:
    tokenizer = bundle.tokenizer
    target_ids = token_ids(tokenizer, ep.target)
    distractor_ids = token_ids(tokenizer, ep.distractor)
    if len(target_ids) != 1 or len(distractor_ids) != 1 or target_ids == distractor_ids:
        raise ValueError(f"Eval prompt `{ep.prompt}` has bad target/distractor tokenization.")
    return int(target_ids[0]), int(distractor_ids[0])


def evaluate_side_sets(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    targets: list[EditTarget],
    scale_choice: Mapping[str, Mapping[str, Any]],
    vectors: Mapping[str, EditVector],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retain_rows: list[dict[str, Any]] = []
    paraphrase_rows: list[dict[str, Any]] = []
    methods = ("localized_addition", "wrong_position_addition", "random_direction_addition", "opposite_direction_addition")
    for target in targets:
        vec = vectors[target.target_id]
        scale = as_float(scale_choice[target.target_id]["scale"], 0.0)
        for eval_family, collection in (("retain", target.retain_prompts), ("neighbor", target.neighbor_prompts)):
            for ep in collection:
                tid, did = eval_ids(bundle, ep)
                base_cap = bench.run_with_residual_cache(bundle, ep.prompt)
                base_margin = logit_margin(base_cap.final_logits_last, tid, did)
                pos = len(base_cap.input_ids) - 1
                wrong_pos = wrong_position_for_length(len(base_cap.input_ids), pos)
                for method in methods:
                    edited_margin = score_prompt_with_method(
                        bundle,
                        ep.prompt,
                        tid,
                        did,
                        method=method,
                        depth=vec.depth,
                        position=pos,
                        direction=vec.direction,
                        random_direction=vec.random_direction,
                        wrong_position=wrong_pos,
                        scale=scale,
                    )
                    retain_rows.append({
                        "target_id": target.target_id,
                        "family": target.family,
                        "eval_family": eval_family,
                        "eval_role": ep.role,
                        "prompt": ep.prompt,
                        "method": method,
                        "scale": scale,
                        "target": ep.target,
                        "distractor": ep.distractor,
                        "base_margin": rounded(base_margin),
                        "edited_margin": rounded(edited_margin),
                        "margin_delta": rounded(edited_margin - base_margin),
                        "preserved_sign": (base_margin >= 0.0 and edited_margin >= 0.0) or (base_margin < 0.0 and edited_margin < 0.0),
                        "damage": rounded(max(0.0, base_margin - edited_margin)),
                    })
        for ep in target.paraphrase_prompts:
            tid = target.after_id
            did = target.before_id
            base_cap = bench.run_with_residual_cache(bundle, ep.prompt)
            base_margin = logit_margin(base_cap.final_logits_last, tid, did)
            pos = len(base_cap.input_ids) - 1
            wrong_pos = wrong_position_for_length(len(base_cap.input_ids), pos)
            for method in methods:
                edited_margin = score_prompt_with_method(
                    bundle,
                    ep.prompt,
                    tid,
                    did,
                    method=method,
                    depth=vec.depth,
                    position=pos,
                    direction=vec.direction,
                    random_direction=vec.random_direction,
                    wrong_position=wrong_pos,
                    scale=scale,
                )
                paraphrase_rows.append({
                    "target_id": target.target_id,
                    "family": target.family,
                    "eval_role": ep.role,
                    "prompt": ep.prompt,
                    "method": method,
                    "scale": scale,
                    "base_after_minus_before": rounded(base_margin),
                    "edited_after_minus_before": rounded(edited_margin),
                    "transfer_gain": rounded(edited_margin - base_margin),
                    "transferred_to_after": edited_margin > 0.0,
                })
    retain_path = ctx.path("tables", "retain_forget_matrix.csv")
    bench.write_csv_with_context(ctx, retain_path, retain_rows)
    ctx.register_artifact(retain_path, "table", "Retain and neighbor preservation matrix at chosen scale.")
    paraphrase_path = ctx.path("tables", "paraphrase_robustness.csv")
    bench.write_csv_with_context(ctx, paraphrase_path, paraphrase_rows)
    ctx.register_artifact(paraphrase_path, "table", "Paraphrase transfer results at chosen scale.")
    return retain_rows, paraphrase_rows


def run_reversibility_check(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    targets: Sequence[EditTarget],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in targets:
        cap = bench.run_with_residual_cache(bundle, target.prompt)
        margin = logit_margin(cap.final_logits_last, target.after_id, target.before_id)
        delta = margin - target.base_after_margin
        rows.append({
            "target_id": target.target_id,
            "baseline_after_minus_before": rounded(target.base_after_margin),
            "post_intervention_after_minus_before": rounded(margin),
            "abs_delta": rounded(abs(delta)),
            "atol": REVERSIBILITY_ATOL,
            "ok": abs(delta) <= REVERSIBILITY_ATOL,
        })
    path = ctx.path("tables", "reversibility_check.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Proof that inference-time edits left no persistent model state change.")
    worst = max([as_float(row.get("abs_delta"), 0.0) for row in rows] or [0.0])
    if worst > REVERSIBILITY_ATOL:
        raise RuntimeError(f"Lab 28 reversibility check failed: {worst:.3g} > {REVERSIBILITY_ATOL}")
    return rows


# ---------------------------------------------------------------------------
# Evidence synthesis
# ---------------------------------------------------------------------------


def localization_row_for(best_by_target: Mapping[str, Mapping[str, Any]], target_id: str) -> Mapping[str, Any]:
    return best_by_target.get(target_id, {})


def rows_for(rows: Sequence[Mapping[str, Any]], **filters: Any) -> list[Mapping[str, Any]]:
    return [row for row in rows if all(row.get(k) == v for k, v in filters.items())]


def summarize_evidence(
    targets: Sequence[EditTarget],
    best_by_target: Mapping[str, Mapping[str, Any]],
    edit_rows: Sequence[Mapping[str, Any]],
    scale_choice: Mapping[str, Mapping[str, Any]],
    retain_rows: Sequence[Mapping[str, Any]],
    paraphrase_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    counterexamples: list[dict[str, Any]] = []
    refinement: list[dict[str, Any]] = []
    for target in targets:
        best = localization_row_for(best_by_target, target.target_id)
        selected_scale = as_float(scale_choice[target.target_id]["scale"], 0.0)
        chosen = [
            row for row in edit_rows
            if row["target_id"] == target.target_id
            and row["method"] == "localized_addition"
            and abs(as_float(row["scale"]) - selected_scale) < 1e-9
        ]
        chosen_row = chosen[0] if chosen else {}
        target_gain = as_float(chosen_row.get("target_gain"))
        target_control_floor = as_float(scale_choice[target.target_id].get("target_control_floor"))
        target_control_gap = as_float(scale_choice[target.target_id].get("target_control_gap"))
        target_changed_to_after = bool(scale_choice[target.target_id].get("changed_to_after"))
        para_loc = rows_for(paraphrase_rows, target_id=target.target_id, method="localized_addition")
        para_ctl = [row for row in paraphrase_rows if row["target_id"] == target.target_id and row["method"] in CONTROL_EDIT_METHODS]
        retain_loc = [row for row in retain_rows if row["target_id"] == target.target_id and row["method"] == "localized_addition" and row["eval_family"] == "retain"]
        neighbor_loc = [row for row in retain_rows if row["target_id"] == target.target_id and row["method"] == "localized_addition" and row["eval_family"] == "neighbor"]
        para_gain = safe_mean([row.get("transfer_gain") for row in para_loc], default=0.0)
        para_control_gain = safe_mean([row.get("transfer_gain") for row in para_ctl], default=0.0)
        para_rate = safe_mean([1.0 if row.get("transferred_to_after") else 0.0 for row in para_loc], default=0.0)
        retain_damage = safe_mean([row.get("damage") for row in retain_loc], default=0.0)
        retain_max_damage = safe_max([row.get("damage") for row in retain_loc], default=0.0)
        retain_preserve_rate = safe_mean([1.0 if row.get("preserved_sign") else 0.0 for row in retain_loc], default=0.0)
        neighbor_damage = safe_mean([row.get("damage") for row in neighbor_loc], default=0.0)
        neighbor_max_damage = safe_max([row.get("damage") for row in neighbor_loc], default=0.0)
        localization_gain = as_float(best.get("patch_gain"))
        localization_control_floor = as_float(best.get("localization_control_floor"))
        localization_gap = as_float(best.get("localization_gap"))
        gates = {
            "baseline_prefers_before": target.base_after_margin < -MIN_BASELINE_BEFORE_MARGIN,
            "donor_supports_after": target.donor_after_margin > 0.0,
            "claimable_depth": bool(best.get("claimable_depth")),
            "localization_gap_pass": math.isfinite(localization_gap) and localization_gap >= LOCALITY_GAP_MIN,
            "target_gain_pass": math.isfinite(target_gain) and target_gain >= TARGET_GAIN_MIN,
            "target_control_gap_pass": math.isfinite(target_control_gap) and target_control_gap >= TARGET_CONTROL_GAP_MIN,
            "paraphrase_gain_pass": bool(para_loc) and para_gain >= PARAPHRASE_GAIN_MIN,
            "retain_damage_pass": retain_damage <= RETAIN_DAMAGE_SOFT_LIMIT and retain_max_damage <= RETAIN_MAX_DAMAGE_LIMIT,
            "retain_sign_pass": retain_preserve_rate >= 0.80 if retain_loc else False,
            "neighbor_not_severe": neighbor_max_damage <= NEIGHBOR_DAMAGE_CAVEAT,
        }
        failed = [name for name, ok in gates.items() if not ok]
        if all(gates.values()):
            posture = "localized_edit_supported"
        elif not gates["baseline_prefers_before"] or not gates["donor_supports_after"]:
            posture = "baseline_or_donor_not_eligible"
        elif not gates["target_control_gap_pass"]:
            posture = "control_limited"
        elif not gates["paraphrase_gain_pass"]:
            posture = "prompt_local_or_no_paraphrase_transfer"
        elif not gates["retain_damage_pass"] or not gates["retain_sign_pass"] or not gates["neighbor_not_severe"]:
            posture = "side_effect_limited"
        else:
            posture = "needs_refinement_or_negative_result"
        ev = {
            "target_id": target.target_id,
            "family": target.family,
            "edit_type": target.edit_type,
            "best_depth": best.get("depth", ""),
            "claimable_depth": best.get("claimable_depth", ""),
            "chosen_scale": rounded(selected_scale),
            "scale_selection_reason": scale_choice[target.target_id].get("reason", ""),
            "baseline_after_minus_before": rounded(target.base_after_margin),
            "donor_after_minus_before": rounded(target.donor_after_margin),
            "localization_patch_gain": rounded(localization_gain),
            "localization_control_floor": rounded(localization_control_floor),
            "localization_gap": rounded(localization_gap),
            "localized_target_gain": rounded(target_gain),
            "target_control_floor": rounded(target_control_floor),
            "target_control_gap": rounded(target_control_gap),
            "target_changed_to_after": target_changed_to_after,
            "mean_paraphrase_gain": rounded(para_gain),
            "mean_paraphrase_control_gain": rounded(para_control_gain),
            "paraphrase_transfer_rate": rounded(para_rate),
            "mean_retain_damage": rounded(retain_damage),
            "max_retain_damage": rounded(retain_max_damage),
            "retain_preserve_rate": rounded(retain_preserve_rate),
            "mean_neighbor_damage": rounded(neighbor_damage),
            "max_neighbor_damage": rounded(neighbor_max_damage),
            "n_paraphrases": len(para_loc),
            "n_retain": len(retain_loc),
            "n_neighbor": len(neighbor_loc),
            **{f"gate_{k}": v for k, v in gates.items()},
            "failed_gates": ";".join(failed),
            "claim_posture": posture,
        }
        evidence.append(ev)
        # Counterexamples.
        if not gates["target_control_gap_pass"]:
            counterexamples.append({
                "target_id": target.target_id,
                "kind": "control_matches_or_exceeds_localized_edit",
                "severity": rounded(TARGET_CONTROL_GAP_MIN - target_control_gap if math.isfinite(target_control_gap) else TARGET_CONTROL_GAP_MIN),
                "selected_scale": selected_scale,
                "localized_target_gain": rounded(target_gain),
                "target_control_floor": rounded(target_control_floor),
                "note": "Random, wrong-position, or opposite-direction control explains too much target movement.",
            })
        if para_loc and para_gain < PARAPHRASE_GAIN_MIN:
            worst_para = min(para_loc, key=lambda row: as_float(row.get("transfer_gain"), 0.0))
            counterexamples.append({
                "target_id": target.target_id,
                "kind": "paraphrase_transfer_failure",
                "severity": rounded(PARAPHRASE_GAIN_MIN - para_gain),
                "prompt": worst_para.get("prompt", ""),
                "transfer_gain": worst_para.get("transfer_gain", ""),
                "note": "Exact target prompt moved more than paraphrases.",
            })
        if retain_loc and (retain_damage > RETAIN_DAMAGE_SOFT_LIMIT or retain_max_damage > RETAIN_MAX_DAMAGE_LIMIT):
            worst_retain = max(retain_loc, key=lambda row: as_float(row.get("damage"), 0.0))
            counterexamples.append({
                "target_id": target.target_id,
                "kind": "retain_damage",
                "severity": worst_retain.get("damage", ""),
                "prompt": worst_retain.get("prompt", ""),
                "eval_role": worst_retain.get("eval_role", ""),
                "damage": worst_retain.get("damage", ""),
                "note": "The edit damaged an unrelated retain prompt.",
            })
        if neighbor_loc and neighbor_max_damage > NEIGHBOR_DAMAGE_CAVEAT:
            worst_neighbor = max(neighbor_loc, key=lambda row: as_float(row.get("damage"), 0.0))
            counterexamples.append({
                "target_id": target.target_id,
                "kind": "neighbor_damage",
                "severity": worst_neighbor.get("damage", ""),
                "prompt": worst_neighbor.get("prompt", ""),
                "eval_role": worst_neighbor.get("eval_role", ""),
                "damage": worst_neighbor.get("damage", ""),
                "note": "Nearby fact damaged enough to require a caveat.",
            })
        refinement.append({
            "target_id": target.target_id,
            "version": "v1",
            "claim_posture": posture,
            "failed_gates": ";".join(failed),
            "revision": (
                "No automatic revision proposed; replicate before broadening." if posture == "localized_edit_supported"
                else "Narrow to exact prompt and selected depth/scale, or redesign donor/target/control set before claiming transfer."
            ),
            "evidence_path": "tables/edit_evidence_matrix.csv;tables/edit_counterexamples.csv",
            "student_notes": "",
        })
    metrics = {
        "n_targets": len(targets),
        "n_supported_targets": sum(1 for row in evidence if row["claim_posture"] == "localized_edit_supported"),
        "n_counterexamples": len(counterexamples),
        "mean_target_gain": safe_mean([row.get("localized_target_gain") for row in evidence]),
        "mean_target_control_gap": safe_mean([row.get("target_control_gap") for row in evidence]),
        "mean_paraphrase_gain": safe_mean([row.get("mean_paraphrase_gain") for row in evidence]),
        "mean_retain_damage": safe_mean([row.get("mean_retain_damage") for row in evidence]),
        "localization_editability_corr": safe_corr(
            [row.get("localization_patch_gain") for row in evidence],
            [row.get("localized_target_gain") for row in evidence],
        ),
        "verdicts": {row["target_id"]: row["claim_posture"] for row in evidence},
    }
    counterexamples.sort(key=lambda row: as_float(row.get("severity"), 0.0), reverse=True)
    return evidence, counterexamples, refinement, metrics



# ---------------------------------------------------------------------------
# Plot source tables, manifests, and diagnostics
# ---------------------------------------------------------------------------


def dashboard_source_rows(evidence: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "figure_row_id": stable_id("dashboard", row.get("target_id"), prefix="l28dash_"),
            "target_label": short_target_label(row.get("target_id")),
            **dict(row),
        }
        for row in evidence
    ]


def target_vs_control_source_rows(
    evidence: Sequence[Mapping[str, Any]],
    edit_rows: Sequence[Mapping[str, Any]],
    scale_choice: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    evidence_by_target = {str(row.get("target_id")): row for row in evidence}
    rows: list[dict[str, Any]] = []
    for row in edit_rows:
        tid = str(row.get("target_id"))
        if row.get("method") not in ("localized_addition",) + CONTROL_EDIT_METHODS:
            continue
        selected_scale = as_float(scale_choice.get(tid, {}).get("scale"))
        scale = as_float(row.get("scale"))
        if not math.isfinite(selected_scale) or abs(scale - selected_scale) > 1e-9:
            continue
        ev = evidence_by_target.get(tid, {})
        role = "target" if row.get("method") == "localized_addition" else "control"
        rows.append({
            "source_row_id": stable_id("target_vs_control", tid, row.get("method"), scale, prefix="l28tvc_"),
            "target_id": tid,
            "target_label": short_target_label(tid),
            "family": row.get("family", ev.get("family", "")),
            "method": row.get("method", ""),
            "method_label": method_label(row.get("method")),
            "method_role": role,
            "depth": row.get("depth", ev.get("best_depth", "")),
            "scale": rounded(scale),
            "target_gain": rounded(row.get("target_gain")),
            "edited_after_minus_before": row.get("edited_after_minus_before", ""),
            "base_after_minus_before": row.get("base_after_minus_before", ""),
            "changed_to_after": row.get("changed_to_after", ""),
            "selected_scale": rounded(selected_scale),
            "target_control_floor": rounded(scale_choice.get(tid, {}).get("target_control_floor")),
            "target_control_gap": rounded(scale_choice.get(tid, {}).get("target_control_gap")),
            "claim_posture": ev.get("claim_posture", ""),
        })
    return rows


def dose_response_source_rows(
    edit_rows: Sequence[Mapping[str, Any]],
    scale_choice: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in edit_rows:
        tid = str(row.get("target_id"))
        method = str(row.get("method"))
        scale = as_float(row.get("scale"))
        selected_scale = as_float(scale_choice.get(tid, {}).get("scale"))
        floor = control_floor_for_scale(edit_rows, tid, scale) if math.isfinite(scale) else float("nan")
        gain = as_float(row.get("target_gain"))
        rows.append({
            "source_row_id": stable_id("dose", tid, method, scale, prefix="l28dose_"),
            "target_id": tid,
            "target_label": short_target_label(tid),
            "family": row.get("family", ""),
            "method": method,
            "method_label": method_label(method),
            "method_role": "target" if method == "localized_addition" else ("baseline" if method == "no_edit" else "control"),
            "depth": row.get("depth", ""),
            "scale": rounded(scale),
            "target_gain": rounded(gain),
            "base_after_minus_before": row.get("base_after_minus_before", ""),
            "edited_after_minus_before": row.get("edited_after_minus_before", ""),
            "changed_to_after": row.get("changed_to_after", ""),
            "selected_scale": rounded(selected_scale),
            "is_selected_scale": math.isfinite(scale) and math.isfinite(selected_scale) and abs(scale - selected_scale) < 1e-9,
            "target_control_floor_at_scale": rounded(floor),
            "target_control_gap_at_scale": rounded(gain - floor) if method == "localized_addition" and math.isfinite(gain) and math.isfinite(floor) else "",
        })
    return rows


def localization_editability_source_rows(evidence: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in evidence:
        rows.append({
            "source_row_id": stable_id("loc_edit", row.get("target_id"), prefix="l28le_"),
            "target_id": row.get("target_id", ""),
            "target_label": short_target_label(row.get("target_id", "")),
            "family": row.get("family", ""),
            "best_depth": row.get("best_depth", ""),
            "localization_patch_gain": row.get("localization_patch_gain", ""),
            "localization_control_floor": row.get("localization_control_floor", ""),
            "localization_gap": row.get("localization_gap", ""),
            "localized_target_gain": row.get("localized_target_gain", ""),
            "target_control_gap": row.get("target_control_gap", ""),
            "mean_retain_damage": row.get("mean_retain_damage", ""),
            "claim_posture": row.get("claim_posture", ""),
        })
    return rows


def locality_ladder_source_rows(
    evidence: Sequence[Mapping[str, Any]],
    localization_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ev in evidence:
        tid = str(ev.get("target_id"))
        depth = str(ev.get("best_depth"))
        for method in ("localized_patch", "wrong_position_patch", "random_direction_patch"):
            vals = [
                as_float(row.get("patch_gain"))
                for row in localization_rows
                if str(row.get("target_id")) == tid and str(row.get("depth")) == depth and row.get("method") == method
            ]
            rows.append({
                "source_row_id": stable_id("locality", tid, depth, method, prefix="l28loc_"),
                "target_id": tid,
                "target_label": short_target_label(tid),
                "depth": depth,
                "method": method,
                "method_label": localization_method_label(method),
                "patch_gain": rounded(safe_mean(vals, 0.0)),
                "n_rows": len(finite_values(vals)),
                "claim_posture": ev.get("claim_posture", ""),
            })
    return rows


def layer_sweep_heatmap_source_rows(localization_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per target/depth for the localization gap heatmap.

    This keeps the selected-depth ladder honest by showing the full sweep that
    produced it, including depth 0 and final-depth caveats.
    """
    grouped: dict[tuple[str, int], dict[str, Any]] = defaultdict(dict)
    meta: dict[tuple[str, int], dict[str, Any]] = {}
    for row in localization_rows:
        method = str(row.get("method"))
        if method not in {"localized_patch", "wrong_position_patch", "random_direction_patch"}:
            continue
        tid = str(row.get("target_id"))
        depth = int(as_float(row.get("depth"), 0.0))
        key = (tid, depth)
        grouped[key][method] = as_float(row.get("patch_gain"))
        meta[key] = row
    rows: list[dict[str, Any]] = []
    for (tid, depth), vals in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        loc = vals.get("localized_patch", float("nan"))
        wrong = vals.get("wrong_position_patch", float("nan"))
        random = vals.get("random_direction_patch", float("nan"))
        controls = finite_values([wrong, random])
        floor = max(controls) if controls else float("nan")
        gap = loc - floor if math.isfinite(loc) and math.isfinite(floor) else float("nan")
        m = meta.get((tid, depth), {})
        rows.append({
            "source_row_id": stable_id("layer_sweep", tid, depth, prefix="l28sweep_"),
            "target_id": tid,
            "target_label": short_target_label(tid),
            "family": m.get("family", ""),
            "depth": depth,
            "claimable_depth": m.get("claimable_depth", ""),
            "localized_patch_gain": rounded(loc),
            "wrong_position_patch_gain": rounded(wrong),
            "random_direction_patch_gain": rounded(random),
            "strongest_control_patch_gain": rounded(floor),
            "localization_gap": rounded(gap),
            "n_methods_observed": len(vals),
        })
    return rows


def paraphrase_matrix_source_rows(paraphrase_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in paraphrase_rows:
        if row.get("method") != "localized_addition":
            continue
        tid = str(row.get("target_id"))
        role = str(row.get("eval_role") or "paraphrase")
        rows.append({
            "source_row_id": stable_id("para", tid, role, row.get("prompt"), prefix="l28para_"),
            "target_id": tid,
            "target_label": short_target_label(tid),
            "family": row.get("family", ""),
            "eval_role": role,
            "prompt": row.get("prompt", ""),
            "method": row.get("method", ""),
            "scale": row.get("scale", ""),
            "base_after_minus_before": row.get("base_after_minus_before", ""),
            "edited_after_minus_before": row.get("edited_after_minus_before", ""),
            "transfer_gain": row.get("transfer_gain", ""),
            "transferred_to_after": row.get("transferred_to_after", ""),
        })
    return rows


def retain_neighbor_atlas_source_rows(retain_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in retain_rows:
        if row.get("method") != "localized_addition":
            continue
        tid = str(row.get("target_id"))
        role = f"{row.get('eval_family')}:{row.get('eval_role') or 'prompt'}"
        rows.append({
            "source_row_id": stable_id("retain", tid, role, row.get("prompt"), prefix="l28ret_"),
            "target_id": tid,
            "target_label": short_target_label(tid),
            "family": row.get("family", ""),
            "eval_family": row.get("eval_family", ""),
            "eval_role": row.get("eval_role", ""),
            "display_role": role,
            "prompt": row.get("prompt", ""),
            "target": row.get("target", ""),
            "distractor": row.get("distractor", ""),
            "base_margin": row.get("base_margin", ""),
            "edited_margin": row.get("edited_margin", ""),
            "margin_delta": row.get("margin_delta", ""),
            "preserved_sign": row.get("preserved_sign", ""),
            "damage": row.get("damage", ""),
        })
    return rows


def frontier_source_rows(evidence: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in evidence:
        rows.append({
            "source_row_id": stable_id("frontier", row.get("target_id"), prefix="l28front_"),
            "target_id": row.get("target_id", ""),
            "target_label": short_target_label(row.get("target_id", "")),
            "family": row.get("family", ""),
            "target_control_gap": row.get("target_control_gap", ""),
            "localized_target_gain": row.get("localized_target_gain", ""),
            "mean_retain_damage": row.get("mean_retain_damage", ""),
            "max_retain_damage": row.get("max_retain_damage", ""),
            "mean_neighbor_damage": row.get("mean_neighbor_damage", ""),
            "claim_posture": row.get("claim_posture", ""),
        })
    return rows


def paired_examples_source_rows(
    evidence: Sequence[Mapping[str, Any]],
    edit_rows: Sequence[Mapping[str, Any]],
    retain_rows: Sequence[Mapping[str, Any]],
    paraphrase_rows: Sequence[Mapping[str, Any]],
    scale_choice: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ev_by_target = {str(row.get("target_id")): row for row in evidence}
    rows: list[dict[str, Any]] = []
    for row in edit_rows:
        tid = str(row.get("target_id"))
        if row.get("method") != "localized_addition":
            continue
        selected_scale = as_float(scale_choice.get(tid, {}).get("scale"))
        scale = as_float(row.get("scale"))
        if not math.isfinite(selected_scale) or abs(scale - selected_scale) > 1e-9:
            continue
        rows.append({
            "source_row_id": stable_id("paired", tid, "target_exact", prefix="l28pair_"),
            "target_id": tid,
            "target_label": short_target_label(tid),
            "specimen_family": "target_exact",
            "eval_role": "exact_target_prompt",
            "prompt": ev_by_target.get(tid, {}).get("prompt", ""),
            "method": "localized_addition",
            "scale": rounded(scale),
            "base_margin": row.get("base_after_minus_before", ""),
            "edited_margin": row.get("edited_after_minus_before", ""),
            "margin_delta": row.get("target_gain", ""),
            "damage": "",
            "claim_posture": ev_by_target.get(tid, {}).get("claim_posture", ""),
        })
    for row in paraphrase_rows:
        if row.get("method") != "localized_addition":
            continue
        tid = str(row.get("target_id"))
        rows.append({
            "source_row_id": stable_id("paired", tid, "paraphrase", row.get("eval_role"), row.get("prompt"), prefix="l28pair_"),
            "target_id": tid,
            "target_label": short_target_label(tid),
            "specimen_family": "paraphrase",
            "eval_role": row.get("eval_role", ""),
            "prompt": row.get("prompt", ""),
            "method": "localized_addition",
            "scale": row.get("scale", ""),
            "base_margin": row.get("base_after_minus_before", ""),
            "edited_margin": row.get("edited_after_minus_before", ""),
            "margin_delta": row.get("transfer_gain", ""),
            "damage": "",
            "claim_posture": ev_by_target.get(tid, {}).get("claim_posture", ""),
        })
    for row in retain_rows:
        if row.get("method") != "localized_addition":
            continue
        tid = str(row.get("target_id"))
        rows.append({
            "source_row_id": stable_id("paired", tid, row.get("eval_family"), row.get("eval_role"), row.get("prompt"), prefix="l28pair_"),
            "target_id": tid,
            "target_label": short_target_label(tid),
            "specimen_family": row.get("eval_family", ""),
            "eval_role": row.get("eval_role", ""),
            "prompt": row.get("prompt", ""),
            "method": "localized_addition",
            "scale": row.get("scale", ""),
            "base_margin": row.get("base_margin", ""),
            "edited_margin": row.get("edited_margin", ""),
            "margin_delta": row.get("margin_delta", ""),
            "damage": row.get("damage", ""),
            "claim_posture": ev_by_target.get(tid, {}).get("claim_posture", ""),
        })
    return rows


def write_plot_source_tables(
    ctx: bench.RunContext,
    evidence: Sequence[Mapping[str, Any]],
    localization_rows: Sequence[Mapping[str, Any]],
    edit_rows: Sequence[Mapping[str, Any]],
    retain_rows: Sequence[Mapping[str, Any]],
    paraphrase_rows: Sequence[Mapping[str, Any]],
    scale_choice: Mapping[str, Mapping[str, Any]],
    counterexamples: Sequence[Mapping[str, Any]] = (),
) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    source_rows = {
        "dashboard_evidence.csv": dashboard_source_rows(evidence),
        "target_vs_control_source.csv": target_vs_control_source_rows(evidence, edit_rows, scale_choice),
        "dose_response_source.csv": dose_response_source_rows(edit_rows, scale_choice),
        "localization_editability_source.csv": localization_editability_source_rows(evidence),
        "locality_ladder_source.csv": locality_ladder_source_rows(evidence, localization_rows),
        "layer_sweep_heatmap_source.csv": layer_sweep_heatmap_source_rows(localization_rows),
        "paraphrase_matrix_source.csv": paraphrase_matrix_source_rows(paraphrase_rows),
        "retain_neighbor_atlas_source.csv": retain_neighbor_atlas_source_rows(retain_rows),
        "frontier_source.csv": frontier_source_rows(evidence),
        "paired_examples_source.csv": paired_examples_source_rows(evidence, edit_rows, retain_rows, paraphrase_rows, scale_choice),
    }
    descriptions = {
        "dashboard_evidence.csv": "Source rows for the Lab 28 dashboard.",
        "target_vs_control_source.csv": "Selected-scale target versus edit-control source rows.",
        "dose_response_source.csv": "Target and control dose-response source rows.",
        "localization_editability_source.csv": "Localization patch gain versus additive edit source rows.",
        "locality_ladder_source.csv": "Selected-depth localization-control source rows.",
        "layer_sweep_heatmap_source.csv": "Full-depth localization gap heatmap source rows.",
        "paraphrase_matrix_source.csv": "Paraphrase robustness heatmap source rows.",
        "retain_neighbor_atlas_source.csv": "Retain and neighbor preservation atlas source rows.",
        "frontier_source.csv": "Retain-forget and method-frontier source rows.",
        "paired_examples_source.csv": "Raw paired before/after specimen rows.",
    }
    for name, rows in source_rows.items():
        sources[name] = write_figure_source(ctx, name, rows, descriptions[name])
    sources["failure_specimens.jsonl"] = {
        "path": "tables/failure_specimens.jsonl",
        "row_count": len(counterexamples),
        "description": "Counterexample specimens used by failure-specimen artifacts.",
    }
    return sources


def write_failure_specimens(
    ctx: bench.RunContext,
    counterexamples: Sequence[Mapping[str, Any]],
) -> tuple[pathlib.Path, pathlib.Path]:
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(counterexamples, start=1):
        rows.append({
            "failure_id": stable_id("failure", i, row.get("target_id"), row.get("kind"), row.get("prompt"), prefix="l28fail_"),
            "rank": i,
            "target_id": row.get("target_id", ""),
            "kind": row.get("kind", ""),
            "severity": row.get("severity", ""),
            "prompt": row.get("prompt", ""),
            "eval_role": row.get("eval_role", ""),
            "selected_scale": row.get("selected_scale", ""),
            "localized_target_gain": row.get("localized_target_gain", ""),
            "target_control_floor": row.get("target_control_floor", ""),
            "transfer_gain": row.get("transfer_gain", ""),
            "damage": row.get("damage", ""),
            "note": row.get("note", ""),
            "inspect_first": "tables/edit_counterexamples.csv;tables/edit_evidence_matrix.csv;tables/figure_sources/paired_examples_source.csv",
        })
    jsonl_path = ctx.path("tables", "failure_specimens.jsonl")
    write_jsonl(jsonl_path, rows)
    ctx.register_artifact(jsonl_path, "table", "JSONL counterexample specimens for Lab 28 figures and writeups.")

    md_lines = [
        "# Lab 28 failure specimens",
        "",
        "These specimens are the little gravel in the shoe: rows that should shrink the claim before the student writes anything grand.",
        "",
    ]
    if rows:
        for row in rows[:20]:
            md_lines += [
                f"## {row['rank']}. `{row['kind']}` for `{row['target_id']}`",
                "",
                f"- severity: `{row['severity']}`",
                f"- prompt: `{row['prompt']}`" if row.get("prompt") else "- prompt: see source table",
                f"- note: {row['note']}",
                f"- inspect: `{row['inspect_first']}`",
                "",
            ]
    else:
        md_lines += [
            "No automatic counterexample crossed the configured thresholds.",
            "",
            "That does not certify the edit. It means the configured tripwires did not fire on this run. Replicate on the science tier before broadening the claim.",
            "",
        ]
    md_path = ctx.path("tables", "failure_specimens.md")
    bench.write_text(md_path, "\n".join(md_lines))
    ctx.register_artifact(md_path, "summary", "Readable failure-specimen gallery for Lab 28.")
    return jsonl_path, md_path


def write_plot_manifest(ctx: bench.RunContext, sources: Mapping[str, Mapping[str, Any]], no_plots: bool) -> None:
    def src(name: str) -> str:
        return str(sources.get(name, {}).get("path", ""))

    def nrows(name: str) -> int:
        return int(sources.get(name, {}).get("row_count", 0) or 0)

    manifest_rows = [
        {
            "figure_path": "plots/editing_unlearning_dashboard.png",
            "source_table": src("dashboard_evidence.csv"),
            "row_count": nrows("dashboard_evidence.csv"),
            "metric": "localized_target_gain,target_control_gap,mean_paraphrase_gain,mean_retain_damage,localization_gap",
            "control": "wrong-position, random-direction, opposite-sign, retain, neighbor, paraphrase",
            "question": "Do all gates point in the same direction?",
            "claim_supported": "Only a narrow reversible activation-edit claim when all evidence gates pass.",
            "caveat": "No persistent unlearning or fact erasure.",
            "written_when_no_plots": False,
        },
        {
            "figure_path": "plots/target_vs_control.png",
            "source_table": src("target_vs_control_source.csv"),
            "row_count": nrows("target_vs_control_source.csv"),
            "metric": "target_gain at selected scale",
            "control": "wrong-position, random-direction, opposite-sign additions at the same scale",
            "question": "Does the localized addition beat matched controls at the chosen dose?",
            "claim_supported": "Direction/site-specific target movement if localized gain exceeds the strongest control.",
            "caveat": "One target row is not transfer or side-effect evidence.",
            "written_when_no_plots": False,
        },
        {
            "figure_path": "plots/dose_response.png",
            "source_table": src("dose_response_source.csv"),
            "row_count": nrows("dose_response_source.csv"),
            "metric": "target_gain and localized-minus-control gap over scale",
            "control": "same-scale edit controls",
            "question": "Was the chosen scale earned by a dose-response curve?",
            "claim_supported": "Scale selection transparency, not success by itself.",
            "caveat": "Large doses can create broad perturbation artifacts.",
            "written_when_no_plots": False,
        },
        {
            "figure_path": "plots/layer_sweep_heatmap.png",
            "source_table": src("layer_sweep_heatmap_source.csv"),
            "row_count": nrows("layer_sweep_heatmap_source.csv"),
            "metric": "localization_gap by target and depth",
            "control": "wrong-position and random-direction patch controls at each depth",
            "question": "Where did localization appear across the residual stream, and was it interior?",
            "claim_supported": "Depth-selection context for the selected locality ladder.",
            "caveat": "Depth 0 and final-depth rows are diagnostics, not main claim sites.",
            "written_when_no_plots": False,
        },
        {
            "figure_path": "plots/paired_examples.png",
            "source_table": src("paired_examples_source.csv"),
            "row_count": nrows("paired_examples_source.csv"),
            "metric": "before and after margins per specimen",
            "control": "raw exact/paraphrase/retain/neighbor specimens",
            "question": "Which individual prompts moved, failed, or were damaged?",
            "claim_supported": "Specimen-level caveats and negative-result visibility.",
            "caveat": "Do not infer semantic transfer from the exact target row alone.",
            "written_when_no_plots": False,
        },
        {
            "figure_path": "plots/localization_vs_editability.png",
            "source_table": src("localization_editability_source.csv"),
            "row_count": nrows("localization_editability_source.csv"),
            "metric": "localization_patch_gain vs localized_target_gain",
            "control": "selected-depth localization controls are in locality_ladder_source.csv",
            "question": "Does a patchable site produce an additive edit?",
            "claim_supported": "Whether localization predicts this edit method.",
            "caveat": "Replacement and addition are different interventions.",
            "written_when_no_plots": False,
        },
        {
            "figure_path": "plots/edit_method_frontier.png",
            "source_table": src("frontier_source.csv"),
            "row_count": nrows("frontier_source.csv"),
            "metric": "target_control_gap vs mean_retain_damage",
            "control": "retain audit plus edit controls",
            "question": "Did target specificity come with a side-effect bill?",
            "claim_supported": "Operating-point caution.",
            "caveat": "Upper-right is not victory; it is damage with target movement.",
            "written_when_no_plots": False,
        },
        {
            "figure_path": "plots/mechanistic_locality_ladder.png",
            "source_table": src("locality_ladder_source.csv"),
            "row_count": nrows("locality_ladder_source.csv"),
            "metric": "patch_gain",
            "control": "wrong-position and random-direction donor patches",
            "question": "Was the site localized before editing?",
            "claim_supported": "Locality evidence for selected depth when localized patch beats controls.",
            "caveat": "Locality is a prerequisite, not the edit claim.",
            "written_when_no_plots": False,
        },
        {
            "figure_path": "plots/scale_selection_ladder.png",
            "source_table": src("dose_response_source.csv"),
            "row_count": nrows("dose_response_source.csv"),
            "metric": "localized target gain minus strongest control",
            "control": "same-scale strongest control",
            "question": "Why this dose?",
            "claim_supported": "Pre-side-set dose-selection audit.",
            "caveat": "The side-set audits can still veto the selected dose.",
            "written_when_no_plots": False,
        },
        {
            "figure_path": "plots/paraphrase_robustness_matrix.png",
            "source_table": src("paraphrase_matrix_source.csv"),
            "row_count": nrows("paraphrase_matrix_source.csv"),
            "metric": "transfer_gain",
            "control": "paraphrases are held out from scale selection",
            "question": "Does the edit transfer beyond the exact prompt?",
            "claim_supported": "Transfer caveat when paraphrase gains are weak or uneven.",
            "caveat": "The automatic next-token metric is not a semantic unlearning label.",
            "written_when_no_plots": False,
        },
        {
            "figure_path": "plots/neighbor_preservation_atlas.png",
            "source_table": src("retain_neighbor_atlas_source.csv"),
            "row_count": nrows("retain_neighbor_atlas_source.csv"),
            "metric": "damage=max(0, base_margin-edited_margin)",
            "control": "retain and neighbor prompts with their own target/distractor labels",
            "question": "What did the edit break?",
            "claim_supported": "Side-effect limits and retain caveats.",
            "caveat": "Low mean damage can hide one severe damaged specimen.",
            "written_when_no_plots": False,
        },
        {
            "figure_path": "plots/unlearning_retain_forget_frontier.png",
            "source_table": src("frontier_source.csv"),
            "row_count": nrows("frontier_source.csv"),
            "metric": "localized_target_gain vs mean_retain_damage",
            "control": "retain audit",
            "question": "How much retain damage accompanies target movement?",
            "claim_supported": "Reversible edit operating-point summary.",
            "caveat": "The plot name is historical; weights are not unlearned.",
            "written_when_no_plots": False,
        },
        {
            "figure_path": "tables/failure_specimens.md",
            "source_table": src("failure_specimens.jsonl"),
            "row_count": nrows("failure_specimens.jsonl"),
            "metric": "counterexample severity",
            "control": "all failed gates and side-effect tripwires",
            "question": "Which rows shrink the claim?",
            "claim_supported": "Negative and caveated result reporting.",
            "caveat": "No counterexample threshold crossed is not proof of safety.",
            "written_when_no_plots": True,
        },
    ]
    for row in manifest_rows:
        row["plots_disabled"] = bool(no_plots)
    json_path = ctx.path("plots", "plot_manifest.json")
    bench.write_json(json_path, manifest_rows)
    ctx.register_artifact(json_path, "plot_manifest", "Manifest of Lab 28 figures, source tables, metrics, controls, and claim boundaries.")
    csv_path = ctx.path("plots", "plot_manifest.csv")
    bench.write_csv_with_context(ctx, csv_path, manifest_rows)
    ctx.register_artifact(csv_path, "plot_manifest", "CSV copy of the Lab 28 plot manifest.")


def write_warning_summary(
    ctx: bench.RunContext,
    data_info: Mapping[str, Any],
    token_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    localization_rows: Sequence[Mapping[str, Any]],
    edit_rows: Sequence[Mapping[str, Any]],
    retain_rows: Sequence[Mapping[str, Any]],
    paraphrase_rows: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    scale_choice: Mapping[str, Mapping[str, Any]],
    self_check_status: Mapping[str, Any],
) -> list[dict[str, Any]]:
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
    baseline_already_after = [row for row in baseline_rows if boolish(row.get("baseline_already_after"))]
    donor_not_after = [row for row in baseline_rows if not boolish(row.get("donor_supports_after"))]
    nonclaimable_selected = [row for row in evidence if not boolish(row.get("claimable_depth"))]
    no_paraphrases = [row for row in evidence if int(as_float(row.get("n_paraphrases"), 0)) == 0]
    no_retain = [row for row in evidence if int(as_float(row.get("n_retain"), 0)) == 0]
    no_scale_pass = [tid for tid, choice in scale_choice.items() if str(choice.get("reason", "")).startswith("no_scale_passed")]
    high_retain = [row for row in evidence if as_float(row.get("max_retain_damage"), 0.0) > RETAIN_MAX_DAMAGE_LIMIT]
    high_neighbor = [row for row in evidence if as_float(row.get("max_neighbor_damage"), 0.0) > NEIGHBOR_DAMAGE_CAVEAT]
    no_target_control_rows = [row for row in edit_rows if row.get("method") in CONTROL_EDIT_METHODS]
    small_side_sets = [row for row in evidence if as_float(row.get("n_paraphrases"), 0) < 2 or as_float(row.get("n_retain"), 0) < 2]

    add(
        "data_source",
        "warning" if not bool(data_info.get("science_ready_data")) else "info",
        0 if data_info.get("science_ready_data") else 1,
        "Tier A fallback or tiny data is smoke-only." if not data_info.get("science_ready_data") else "Frozen CSV data was used with enough selected rows for the configured science-ready gate.",
        "diagnostics/data_manifest.json",
    )
    add("tokenization_dropped", "error" if token_dropped else "info", len(token_dropped), "Rows dropped by answer/eval-token checks.", "diagnostics/tokenization_gate.csv")
    add("tokenization_warnings", "warning" if token_warned else "info", len(token_warned), "Rows with tokenization warnings.", "diagnostics/tokenization_gate.csv")
    add("baseline_already_after", "warning" if baseline_already_after else "info", len(baseline_already_after), "Targets whose baseline already favors target_after are weak edit targets.", "tables/baseline_behavior.csv")
    add("donor_not_after_supporting", "warning" if donor_not_after else "info", len(donor_not_after), "Targets whose donor prompt does not support target_after.", "tables/baseline_behavior.csv")
    add("nonclaimable_selected_depth", "warning" if nonclaimable_selected else "info", len(nonclaimable_selected), "Selected localization depth is depth 0 or final depth; positive claim language should be blocked.", "tables/edit_evidence_matrix.csv")
    add("scale_gate_not_passed", "warning" if no_scale_pass else "info", len(no_scale_pass), "No dose passed both target-gain and control-gap gates; code chose the best available control gap.", "tables/scale_selection.csv")
    add("missing_paraphrase_rows", "warning" if no_paraphrases else "info", len(no_paraphrases), "Targets without paraphrase rows cannot claim transfer.", "tables/paraphrase_robustness.csv")
    add("missing_retain_rows", "warning" if no_retain else "info", len(no_retain), "Targets without retain rows cannot claim preservation.", "tables/retain_forget_matrix.csv")
    add("small_side_sets", "warning" if small_side_sets else "info", len(small_side_sets), "Side-set sample count is tiny; inspect raw paired specimens before claiming transfer or preservation.", "tables/figure_sources/paired_examples_source.csv")
    add("retain_damage_high", "warning" if high_retain else "info", len(high_retain), f"Targets with max retain damage above {RETAIN_MAX_DAMAGE_LIMIT}.", "tables/retain_forget_matrix.csv")
    add("neighbor_damage_high", "warning" if high_neighbor else "info", len(high_neighbor), f"Targets with max neighbor damage above {NEIGHBOR_DAMAGE_CAVEAT}.", "tables/retain_forget_matrix.csv")
    add("edit_control_rows_present", "info" if no_target_control_rows else "warning", len(no_target_control_rows), "Rows available for wrong-position/random/opposite edit controls.", "tables/editing_results.csv")
    add("self_checks", "info" if bool(self_check_status.get("ok")) else "error", 0 if self_check_status.get("ok") else 1, "Lab-local tokenization, safety, no-op, and reversibility checks.", "diagnostics/self_check_status.json")

    csv_path = ctx.path("diagnostics", "warning_summary.csv")
    bench.write_csv_with_context(ctx, csv_path, rows)
    ctx.register_artifact(csv_path, "diagnostic", "Warnings for dropped rows, weak targets, control gaps, tiny side sets, and side-effect caveats.")
    json_path = ctx.path("diagnostics", "warning_summary.json")
    bench.write_json(json_path, rows)
    ctx.register_artifact(json_path, "diagnostic", "JSON copy of the Lab 28 warning summary.")
    return rows


def write_lab28_run_config_snapshot(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    data_info: Mapping[str, Any],
    targets: Sequence[EditTarget],
    scale_choice: Mapping[str, Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
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
        "decoding": "none; forward-pass-only residual additions and next-token margins",
        "data": dict(data_info),
        "edit_scales": list(EDIT_SCALES),
        "localization_methods": list(LOCALIZATION_METHODS),
        "edit_methods": list(EDIT_METHODS),
        "control_edit_methods": list(CONTROL_EDIT_METHODS),
        "thresholds": {
            "min_baseline_before_margin": MIN_BASELINE_BEFORE_MARGIN,
            "locality_gap_min": LOCALITY_GAP_MIN,
            "target_gain_min": TARGET_GAIN_MIN,
            "target_control_gap_min": TARGET_CONTROL_GAP_MIN,
            "paraphrase_gain_min": PARAPHRASE_GAIN_MIN,
            "retain_damage_soft_limit": RETAIN_DAMAGE_SOFT_LIMIT,
            "retain_max_damage_limit": RETAIN_MAX_DAMAGE_LIMIT,
            "neighbor_damage_caveat": NEIGHBOR_DAMAGE_CAVEAT,
            "noop_delta_atol": NOOP_DELTA_ATOL,
            "reversibility_atol": REVERSIBILITY_ATOL,
        },
        "selected_targets": [
            {
                "target_id": target.target_id,
                "family": target.family,
                "edit_type": target.edit_type,
                "prompt_sha256": hashlib.sha256(target.prompt.encode("utf-8")).hexdigest(),
                "donor_prompt_sha256": hashlib.sha256(target.donor_prompt.encode("utf-8")).hexdigest(),
                "target_before": target.target_before,
                "target_after": target.target_after,
                "n_retain_prompts": len(target.retain_prompts),
                "n_paraphrase_prompts": len(target.paraphrase_prompts),
                "n_neighbor_prompts": len(target.neighbor_prompts),
                "final_pos": target.final_pos,
                "donor_final_pos": target.donor_final_pos,
                "selected_scale": scale_choice.get(target.target_id, {}).get("scale", ""),
            }
            for target in targets
        ],
        "verdicts": {str(row.get("target_id")): str(row.get("claim_posture")) for row in evidence},
    }
    path = ctx.path("diagnostics", "lab28_run_config_snapshot.json")
    bench.write_json(path, snapshot)
    ctx.register_artifact(path, "diagnostic", "Lab-specific config snapshot for data, targets, scales, methods, thresholds, and verdicts.")
    return snapshot


# ---------------------------------------------------------------------------
# Cards and summaries
# ---------------------------------------------------------------------------


def write_plot_guide(ctx: bench.RunContext) -> None:
    rows = [
        {
            "open_order": 1,
            "plot": "plots/editing_unlearning_dashboard.png",
            "source_table": "tables/figure_sources/dashboard_evidence.csv",
            "question": "Does the chosen edit clear target, control, paraphrase, retain, and locality gates together?",
            "read_for": "One-screen target gain, control gap, transfer, damage, and claim posture.",
            "do_not_claim": "Dashboard positivity is not persistent unlearning or fact erasure.",
        },
        {
            "open_order": 2,
            "plot": "plots/target_vs_control.png",
            "source_table": "tables/figure_sources/target_vs_control_source.csv",
            "question": "Did the localized edit move the target more than wrong-position, random-direction, and opposite-sign controls at the selected dose?",
            "read_for": "Direct target/control comparison at the predeclared selected scale, with raw per-target rows visible.",
            "do_not_claim": "A target gain is specific when the strongest control is similar.",
        },
        {
            "open_order": 3,
            "plot": "plots/dose_response.png",
            "source_table": "tables/figure_sources/dose_response_source.csv",
            "question": "Was the selected dose the smallest credible dose rather than the biggest hammer?",
            "read_for": "Dose-response curves for localized and control additions, with selected scales marked in the source table.",
            "do_not_claim": "Large-scale success is clean evidence when low scales fail and controls rise too.",
        },
        {
            "open_order": 4,
            "plot": "plots/paired_examples.png",
            "source_table": "tables/figure_sources/paired_examples_source.csv",
            "question": "Which exact prompts moved before versus after editing, and which side-set examples moved the wrong way?",
            "read_for": "Raw paired before/after margins for exact targets, paraphrases, retain prompts, and neighbor prompts.",
            "do_not_claim": "An aggregate mean represents every specimen.",
        },
        {
            "open_order": 5,
            "plot": "plots/localization_vs_editability.png",
            "source_table": "tables/figure_sources/localization_editability_source.csv",
            "question": "Does donor-patch localization predict additive edit strength?",
            "read_for": "Patch gain versus additive edit gain, target by target.",
            "do_not_claim": "A good patch site is automatically a good linear edit direction.",
        },
        {
            "open_order": 6,
            "plot": "plots/mechanistic_locality_ladder.png",
            "source_table": "tables/figure_sources/locality_ladder_source.csv",
            "question": "Was localization site-specific before editing?",
            "read_for": "Localized patch gain beside wrong-position and random-direction patch controls at the selected depth.",
            "do_not_claim": "Locality passed if the control floor is high.",
        },
        {
            "open_order": 7,
            "plot": "plots/paraphrase_robustness_matrix.png",
            "source_table": "tables/figure_sources/paraphrase_matrix_source.csv",
            "question": "Does the chosen edit transfer beyond the exact prompt string?",
            "read_for": "Per-target paraphrase transfer gains at the selected scale.",
            "do_not_claim": "Exact-prompt movement is semantic transfer.",
        },
        {
            "open_order": 8,
            "plot": "plots/neighbor_preservation_atlas.png",
            "source_table": "tables/figure_sources/retain_neighbor_atlas_source.csv",
            "question": "What did the edit damage?",
            "read_for": "Retain and neighbor prompt damage cells, not just target success.",
            "do_not_claim": "Side effects are irrelevant because the target moved.",
        },
        {
            "open_order": 9,
            "plot": "plots/edit_method_frontier.png",
            "source_table": "tables/figure_sources/frontier_source.csv",
            "question": "How much target-control advantage was bought per unit of retain damage?",
            "read_for": "Control gap versus retain damage frontier.",
            "do_not_claim": "A point is good when it is high-damage or control-limited.",
        },
        {
            "open_order": 10,
            "plot": "plots/unlearning_retain_forget_frontier.png",
            "source_table": "tables/figure_sources/frontier_source.csv",
            "question": "How does target movement trade off against retain damage?",
            "read_for": "Target gain versus retain damage, with negative and side-effect-limited results visible.",
            "do_not_claim": "The fact was erased from weights.",
        },
        {
            "open_order": 11,
            "plot": "plots/scale_selection_ladder.png",
            "source_table": "tables/figure_sources/dose_response_source.csv",
            "question": "Why was the selected scale chosen?",
            "read_for": "Localized gain minus strongest control across scale.",
            "do_not_claim": "The largest scale is the best evidence.",
        },
        {
            "open_order": 12,
            "plot": "tables/failure_specimens.md",
            "source_table": "tables/failure_specimens.jsonl",
            "question": "Which specimens shrink or kill the claim?",
            "read_for": "Control matches, paraphrase failures, and retain/neighbor damage examples.",
            "do_not_claim": "Counterexamples can be hidden behind aggregate plots.",
        },
    ]
    table_path = ctx.path("tables", "plot_reading_guide.csv")
    bench.write_csv_with_context(ctx, table_path, rows)
    ctx.register_artifact(table_path, "table", "Reading guide for the Lab 28 plot suite.")
    plot_path = ctx.path("plots", "plot_reading_guide.csv")
    bench.write_csv_with_context(ctx, plot_path, rows)
    ctx.register_artifact(plot_path, "table", "Plot-catalog copy stored beside the Lab 28 figures.")


def write_method_card(ctx: bench.RunContext, bundle: bench.ModelBundle, evidence: Sequence[Mapping[str, Any]], data_info: Mapping[str, Any]) -> None:
    lines = [
        "# Lab 28 method card",
        "",
        "Question: does a reversible localized residual addition produce a specific, auditable edit?",
        "",
        "## Scope",
        "",
        f"- model: `{bundle.anatomy.model_id}` ({bundle.anatomy.n_layers} blocks, d_model {bundle.anatomy.d_model})",
        f"- data source: `{data_info.get('data_source')}`",
        f"- science-ready data: `{bool(data_info.get('science_ready_data'))}`",
        "- method run: reversible inference-time residual activation addition",
        "- method not run: persistent weight edit, ROME/MEMIT, feature clamp, refusal ablation, private-data unlearning",
        "- metric: next-token `logit(target_after) - logit(target_before)` plus retain/paraphrase audits",
        "- figure provenance: every major figure has a source CSV under `tables/figure_sources/` plus `plots/plot_manifest.json`",
        "- claimable depths: interior stream depths only for positive posture",
        "",
        "## Verdict table",
        "",
        "| target | family | depth | scale | target gain | control gap | paraphrase gain | retain damage | posture |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in evidence:
        lines.append(
            f"| `{row['target_id']}` | {row['family']} | {row['best_depth']} | {row['chosen_scale']} | "
            f"{row['localized_target_gain']} | {row['target_control_gap']} | {row['mean_paraphrase_gain']} | "
            f"{row['mean_retain_damage']} | `{row['claim_posture']}` |"
        )
    lines += [
        "",
        "## Method contract",
        "",
        "Positive language requires target movement, a control gap, paraphrase transfer, retain preservation, locality evidence, a clean no-op check, and a reversibility check. The result is a reversible activation-edit claim only.",
        "",
        "## Figure evidence contract",
        "",
        "Open `plots/plot_manifest.json` before exporting figures. It names each figure, source table, row count, metric, control, supported claim, and caveat. Open `diagnostics/warning_summary.csv` before trusting smooth-looking plots.",
        "",
    ]
    path = ctx.path("method_card.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Compact method contract and Lab 28 verdict table.")


def write_spec_card(ctx: bench.RunContext, data_info: Mapping[str, Any]) -> None:
    lines = [
        "# Lab 28 editing/unlearning spec",
        "",
        "The lab operationalizes editing as a reversible residual-stream addition, not as a persistent parameter update.",
        "",
        "## Data contract",
        "",
        f"- data path: `{data_info.get('data_path')}`",
        f"- data source: `{data_info.get('data_source')}`",
        f"- sha256: `{data_info.get('sha256')}`",
        f"- selected rows: {data_info.get('n_rows_selected')} of {data_info.get('n_rows_file')}",
        "",
        "Required columns:",
        "",
        "```text",
        "target_id,family,edit_type,prompt,target_before,target_after,",
        "retain_prompts_json,paraphrase_prompts_json,neighbor_prompts_json,",
        "safety_notes,donor_prompt",
        "```",
        "",
        "## Intervention",
        "",
        "```text",
        "direction = donor_stream[depth, donor_final_position] - target_stream[depth, target_final_position]",
        "target_stream[depth, final_position] += scale * direction",
        "```",
        "",
        "The selected depth is chosen from the localization screen using claimable interior depths when available. The selected scale is chosen from target prompt data only before reading paraphrase or retain rows.",
        "",
        "Every plot is built from saved source tables under `tables/figure_sources/`, not from hidden transient state. If a source table has only a warning row, the corresponding figure should be read as a missing-data diagnostic rather than a result.",
        "",
        "## Controls",
        "",
        "- self-patch no-op",
        "- wrong-position donor patch",
        "- random-direction patch",
        "- wrong-position activation addition",
        "- random-direction activation addition",
        "- opposite signed activation addition",
        "- paraphrase transfer",
        "- retain and neighbor damage",
        "- reversibility after all interventions",
        "",
    ]
    path = ctx.path("editing_unlearning_spec.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Human-readable Lab 28 data, intervention, and control contract.")


def write_operationalization_audit(
    ctx: bench.RunContext,
    evidence: Sequence[Mapping[str, Any]],
    counterexamples: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Lab 28 operationalization audit",
        "",
        "```yaml",
        "headline_claim: \"a localized reversible residual addition edits a benign target behavior specifically\"",
        "cheap_explanation: \"the target moved because of broad perturbation, answer-token bias, prompt-local string effects, or side-effect damage\"",
        "killer_control: \"random-direction, wrong-position, opposite-direction, paraphrase, retain, neighbor, no-op, and reversibility checks\"",
        "result: \"filled by edit_evidence_matrix.csv\"",
        "claim_allowed: \"narrow activation-edit handle; not persistent unlearning\"",
        "```",
        "",
        "## Cheap explanations",
        "",
        "| Cheap explanation | Control | Failure pattern |",
        "|---|---|---|",
        "| Any large direction moves the answer token | random-direction addition | random control matches localized gain |",
        "| The site is not specific | wrong-position addition and patching | wrong-position control matches localized gain |",
        "| Sign does not matter | opposite-direction addition | opposite direction also helps |",
        "| The edit is exact-string local | paraphrase audit | target prompt moves but paraphrases do not |",
        "| The edit damages unrelated facts | retain and neighbor audits | retain/neighbor margin damage is high |",
        "| The model changed persistently | reversibility check | baseline margin changed after interventions |",
        "| The hook is mis-targeted | self-patch no-op | self-patching moves logits |",
        "",
        "## Verdicts",
        "",
    ]
    for row in evidence:
        lines.append(
            f"- `{row['target_id']}`: `{row['claim_posture']}`; target gain {row['localized_target_gain']}, "
            f"control gap {row['target_control_gap']}, paraphrase gain {row['mean_paraphrase_gain']}, retain damage {row['mean_retain_damage']}."
        )
    lines += ["", "## Counterexamples", ""]
    if counterexamples:
        for row in counterexamples[:12]:
            lines.append(f"- `{row['kind']}` for `{row['target_id']}`: {row.get('note', '')}")
    else:
        lines.append("- No automatic counterexamples crossed configured thresholds. Replicate before generalizing.")
    lines += [
        "",
        "## Allowed language",
        "",
        "- `This reversible activation addition changed this measured next-token margin under these controls.`",
        "- `The result transferred or failed to transfer to the recorded paraphrases.`",
        "- `The retain audit bounded or failed to bound side effects.`",
        "",
        "## Forbidden language",
        "",
        "- `The fact was erased from the model.`",
        "- `The model now believes the counterfactual.`",
        "- `This edit is safe in deployment.`",
        "- `This site is the whole mechanism.`",
        "",
    ]
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Cheap explanations, controls, counterexamples, and allowed claim grammar.")


def write_run_summary(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    data_info: Mapping[str, Any],
    metrics: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    counterexamples: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Lab 28 run summary: mechanistic editing and unlearning",
        "",
        "## Run identity",
        "",
        f"- model: `{bundle.anatomy.model_id}` ({bundle.anatomy.n_layers} blocks, d_model {bundle.anatomy.d_model})",
        f"- data source: `{data_info.get('data_source')}`",
        f"- data sha256: `{str(data_info.get('sha256', ''))[:16]}`",
        f"- selected rows: {data_info.get('n_rows_selected')} of {data_info.get('n_rows_file')}",
        f"- science-ready data: `{bool(data_info.get('science_ready_data'))}`",
        "- method: reversible inference-time residual activation addition",
        "",
        "## Headline verdicts",
        "",
        "| target | depth | scale | target gain | control gap | retain damage | posture |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in evidence:
        lines.append(
            f"| `{row['target_id']}` | {row['best_depth']} | {row['chosen_scale']} | {row['localized_target_gain']} | "
            f"{row['target_control_gap']} | {row['mean_retain_damage']} | `{row['claim_posture']}` |"
        )
    lines += [
        "",
        "## Read-first artifacts",
        "",
        "1. `method_card.md` for the method contract and verdict table.",
        "2. `editing_unlearning_spec.md` for data and intervention details.",
        "3. `diagnostics/safety_status.json` and `diagnostics/self_check_status.json` before plots.",
        "4. `tables/scale_selection.csv`, `tables/editing_results.csv`, and `tables/edit_evidence_matrix.csv` for claim readiness.",
        "5. `plots/plot_manifest.json` and `diagnostics/warning_summary.csv` for figure provenance and caveats.",
        "6. `tables/failure_specimens.md` and `tables/edit_counterexamples.csv` before writing positive language.",
        "",
        "## Main counterexample",
        "",
    ]
    if counterexamples:
        top = counterexamples[0]
        lines.append(f"- `{top['kind']}` for `{top['target_id']}`: {top.get('note', '')}")
    else:
        lines.append("- No automatic counterexample crossed the configured thresholds.")
    lines += [
        "",
        "## Smallest claim",
        "",
        "A supported row can claim only an inference-time activation edit on this target set, model, tokenizer, selected depth, selected scale, and audit battery. It cannot claim persistent unlearning or fact erasure.",
        "",
        "## Aggregate metrics",
        "",
        f"- supported targets: {metrics.get('n_supported_targets')} / {metrics.get('n_targets')}",
        f"- mean target gain: {rounded(metrics.get('mean_target_gain'))}",
        f"- mean target control gap: {rounded(metrics.get('mean_target_control_gap'))}",
        f"- mean paraphrase gain: {rounded(metrics.get('mean_paraphrase_gain'))}",
        f"- mean retain damage: {rounded(metrics.get('mean_retain_damage'))}",
        f"- localization/editability correlation: {rounded(metrics.get('localization_editability_corr'))}",
        "",
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Standard run summary for Lab 28.")


def write_claims(ctx: bench.RunContext, evidence: Sequence[Mapping[str, Any]]) -> None:
    run_name = ctx.run_dir.name
    claims = []
    for i, row in enumerate(evidence, start=1):
        if row["claim_posture"] == "localized_edit_supported":
            text = (
                f"For `{row['target_id']}`, a reversible localized residual addition at depth {row['best_depth']} "
                f"and scale {row['chosen_scale']} changed the target after-minus-before margin by "
                f"{row['localized_target_gain']}, beating controls by {row['target_control_gap']}, "
                f"with mean paraphrase gain {row['mean_paraphrase_gain']} and mean retain damage {row['mean_retain_damage']}. "
                "This is an inference-time activation-edit claim, not persistent unlearning."
            )
        else:
            text = (
                f"For `{row['target_id']}`, the reversible activation-edit audit did not earn a positive localized edit claim "
                f"because `{row['failed_gates']}`. Posture: `{row['claim_posture']}`."
            )
        claims.append({
            "id": f"{LAB_ID}-C{i}",
            "tag": "CAUSAL+AUDIT",
            "text": text,
            "artifact": f"runs/{run_name}/tables/edit_evidence_matrix.csv",
            "falsifier": "Random/wrong-position/opposite controls match the edit, paraphrases fail, retain damage grows, or the effect vanishes on a held-out target set.",
        })
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_empty(ctx: bench.RunContext, name: str, title: str, message: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    bench.save_figure(ctx, fig, name, f"{title}: empty or insufficient data warning.")


def _annotate_indices(ax: Any, xs: Sequence[Any], ys: Sequence[Any]) -> None:
    for i, (x, y) in enumerate(zip(xs, ys), start=1):
        xf, yf = as_float(x), as_float(y)
        if math.isfinite(xf) and math.isfinite(yf):
            ax.annotate(str(i), (xf, yf), fontsize=8, xytext=(3, 3), textcoords="offset points")


def plot_dashboard(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if not rows:
        _plot_empty(ctx, "editing_unlearning_dashboard.png", "Lab 28 editing/unlearning dashboard", "No evidence rows. Inspect tokenization and safety diagnostics.")
        return
    labels = [short_target_label(row.get("target_id"), 20) for row in rows]
    x = np.arange(len(labels))
    width = 0.22
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    fig.suptitle("Lab 28 editing/unlearning dashboard", fontsize=14, fontweight="bold")

    axes[0, 0].bar(x - width, [as_float(r.get("localized_target_gain"), 0.0) for r in rows], width, label="localized gain")
    axes[0, 0].bar(x, [as_float(r.get("target_control_floor"), 0.0) for r in rows], width, label="strongest control")
    axes[0, 0].bar(x + width, [as_float(r.get("target_control_gap"), 0.0) for r in rows], width, label="localized-control gap")
    axes[0, 0].axhline(0.0, linewidth=0.8)
    axes[0, 0].axhline(TARGET_CONTROL_GAP_MIN, linestyle="--", linewidth=0.9)
    axes[0, 0].set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
    axes[0, 0].set_ylabel("after-minus-before margin movement")
    axes[0, 0].set_title("Target effect must beat controls")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].bar(x - width, [as_float(r.get("mean_paraphrase_gain"), 0.0) for r in rows], width, label="paraphrase gain")
    axes[0, 1].bar(x, [as_float(r.get("mean_retain_damage"), 0.0) for r in rows], width, label="retain damage")
    axes[0, 1].bar(x + width, [as_float(r.get("mean_neighbor_damage"), 0.0) for r in rows], width, label="neighbor damage")
    axes[0, 1].axhline(0.0, linewidth=0.8)
    axes[0, 1].axhline(RETAIN_DAMAGE_SOFT_LIMIT, linestyle="--", linewidth=0.9)
    axes[0, 1].set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
    axes[0, 1].set_title("Transfer and side-effect bill")
    axes[0, 1].legend(fontsize=8)

    xs = [as_float(r.get("localization_gap"), 0.0) for r in rows]
    ys = [as_float(r.get("target_control_gap"), 0.0) for r in rows]
    axes[1, 0].scatter(xs, ys)
    _annotate_indices(axes[1, 0], xs, ys)
    axes[1, 0].axhline(TARGET_CONTROL_GAP_MIN, linestyle="--", linewidth=0.9)
    axes[1, 0].axvline(LOCALITY_GAP_MIN, linestyle="--", linewidth=0.9)
    axes[1, 0].set_xlabel("localization gap")
    axes[1, 0].set_ylabel("target control gap")
    axes[1, 0].set_title("Locality and edit specificity must both pass")

    posture_score = [1 if row.get("claim_posture") == "localized_edit_supported" else 0 for row in rows]
    axes[1, 1].bar(x, posture_score)
    axes[1, 1].set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
    axes[1, 1].set_yticks([0, 1], ["needs caveat", "supported"])
    axes[1, 1].set_title("Claim posture")
    axes[1, 1].text(
        0.02,
        0.95,
        "A supported row still means reversible activation editing only.",
        transform=axes[1, 1].transAxes,
        va="top",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    bench.save_figure(ctx, fig, "editing_unlearning_dashboard.png", "Lab 28 dashboard: target gain, controls, transfer, side effects, locality, and posture.")


def plot_target_vs_control(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if not rows:
        _plot_empty(ctx, "target_vs_control.png", "Target vs controls", "No selected-scale target/control rows were available.")
        return
    targets = sorted({str(row.get("target_id")) for row in rows})
    methods = ["localized_addition", "wrong_position_addition", "random_direction_addition", "opposite_direction_addition"]
    labels = [short_target_label(t, 18) for t in targets]
    x = np.arange(len(targets))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(9.5, len(targets) * 1.2), 5.5))
    for m_i, method in enumerate(methods):
        vals = []
        for tid in targets:
            candidates = [as_float(row.get("target_gain")) for row in rows if row.get("target_id") == tid and row.get("method") == method]
            vals.append(safe_mean(candidates, 0.0))
        ax.bar(x + (m_i - 1.5) * width, vals, width, label=method_label(method))
    ax.axhline(0.0, linewidth=0.8)
    ax.axhline(TARGET_GAIN_MIN, linestyle="--", linewidth=0.9, label="target gain gate")
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("target_gain at selected scale")
    ax.set_title("Target movement beside same-scale controls")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "target_vs_control.png", "Selected-scale localized target movement versus edit controls.")


def plot_dose_response(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        _plot_empty(ctx, "dose_response.png", "Dose response", "No dose-response rows were available.")
        return
    targets = sorted({str(row.get("target_id")) for row in rows if row.get("method") == "localized_addition"})
    if not targets:
        _plot_empty(ctx, "dose_response.png", "Dose response", "No localized-addition rows were available.")
        return
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for tid in targets:
        loc = sorted(
            [row for row in rows if row.get("target_id") == tid and row.get("method") == "localized_addition" and as_float(row.get("scale")) > 0.0],
            key=lambda row: as_float(row.get("scale")),
        )
        if not loc:
            continue
        ax.plot(
            [as_float(row.get("scale")) for row in loc],
            [as_float(row.get("target_gain")) for row in loc],
            marker="o",
            label=f"{short_target_label(tid, 18)} localized",
        )
        selected = [row for row in loc if boolish(row.get("is_selected_scale"))]
        for row in selected:
            ax.scatter([as_float(row.get("scale"))], [as_float(row.get("target_gain"))], marker="s", s=60)
    ax.axhline(0.0, linewidth=0.8)
    ax.axhline(TARGET_GAIN_MIN, linestyle="--", linewidth=0.9, label="target gain gate")
    ax.set_xlabel("scale × donor-minus-target residual direction")
    ax.set_ylabel("target_gain")
    ax.set_title("Dose-response for localized residual addition")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "dose_response.png", "Dose-response curves for localized residual addition.")


def plot_layer_sweep_heatmap(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if not rows:
        _plot_empty(ctx, "layer_sweep_heatmap.png", "Layer/depth localization heatmap", "No localization sweep rows were available.")
        return
    targets = sorted({str(row.get("target_id")) for row in rows})
    depths = sorted({int(as_float(row.get("depth"), 0.0)) for row in rows})
    mat = np.full((len(targets), len(depths)), np.nan)
    claimable = np.zeros((len(targets), len(depths)), dtype=bool)
    for i, tid in enumerate(targets):
        for j, depth in enumerate(depths):
            vals = [
                as_float(row.get("localization_gap"))
                for row in rows
                if str(row.get("target_id")) == tid and int(as_float(row.get("depth"), -1)) == depth
            ]
            vals = finite_values(vals)
            if vals:
                mat[i, j] = safe_mean(vals)
            claimable[i, j] = any(
                boolish(row.get("claimable_depth"))
                for row in rows
                if str(row.get("target_id")) == tid and int(as_float(row.get("depth"), -1)) == depth
            )
    fig, ax = plt.subplots(figsize=(max(8.0, len(depths) * 0.55), max(3.8, len(targets) * 0.5)))
    im = ax.imshow(np.nan_to_num(mat, nan=0.0), aspect="auto")
    ax.set_xticks(range(len(depths)), [str(d) for d in depths], fontsize=7)
    ax.set_yticks(range(len(targets)), [short_target_label(t, 24) for t in targets], fontsize=7)
    ax.set_xlabel("stream depth")
    ax.set_title("Localization gap by target and depth")
    for i in range(len(targets)):
        for j in range(len(depths)):
            if math.isfinite(float(mat[i, j])):
                mark = "" if claimable[i, j] else "*"
                ax.text(j, i, f"{mat[i, j]:.2f}{mark}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.82, label="localized patch gain - strongest control")
    ax.text(0.0, -0.20, "* = non-claimable boundary depth; inspect source table before citing.", transform=ax.transAxes, fontsize=7)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "layer_sweep_heatmap.png", "Localization gap heatmap across stream depths and targets.")


def plot_paired_examples(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if not rows:
        _plot_empty(ctx, "paired_examples.png", "Paired before/after specimens", "No paired specimen rows were available.")
        return
    # Sort so exact targets are first, then the largest damage or negative movement specimens.
    priority = {"target_exact": 0, "paraphrase": 1, "neighbor": 2, "retain": 3}
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            priority.get(str(row.get("specimen_family")), 9),
            -abs(as_float(row.get("damage"), 0.0)),
            as_float(row.get("margin_delta"), 0.0),
        ),
    )[:36]
    fig, ax = plt.subplots(figsize=(9.5, max(5, len(sorted_rows) * 0.24)))
    y = np.arange(len(sorted_rows))
    for i, row in enumerate(sorted_rows):
        base = as_float(row.get("base_margin"))
        edited = as_float(row.get("edited_margin"))
        if not (math.isfinite(base) and math.isfinite(edited)):
            continue
        ax.plot([base, edited], [i, i], marker="o", linewidth=1.2, alpha=0.8)
    labels = [f"{short_target_label(r.get('target_id'), 14)} · {r.get('specimen_family')}:{r.get('eval_role')}"[:52] for r in sorted_rows]
    ax.set_yticks(y, labels, fontsize=7)
    ax.axvline(0.0, linewidth=0.8)
    ax.set_xlabel("margin before and after localized edit")
    ax.set_title("Paired before/after specimens, including side sets")
    ax.invert_yaxis()
    fig.tight_layout()
    bench.save_figure(ctx, fig, "paired_examples.png", "Raw paired before/after margins for target, paraphrase, retain, and neighbor specimens.")


def plot_localization_vs_editability(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        _plot_empty(ctx, "localization_vs_editability.png", "Localization vs editability", "No localization/editability rows were available.")
        return
    xs = [as_float(row.get("localization_patch_gain"), 0.0) for row in rows]
    ys = [as_float(row.get("localized_target_gain"), 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.scatter(xs, ys)
    _annotate_indices(ax, xs, ys)
    ax.axhline(0.0, linestyle=":", linewidth=0.8)
    ax.axvline(0.0, linestyle=":", linewidth=0.8)
    ax.set_xlabel("localized donor-patch gain")
    ax.set_ylabel("additive edit target_gain")
    ax.set_title("Localization strength is not automatically editability")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "localization_vs_editability.png", "Localization patch gain versus additive edit gain.")


def plot_frontiers(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        _plot_empty(ctx, "edit_method_frontier.png", "Edit method frontier", "No frontier rows were available.")
        _plot_empty(ctx, "unlearning_retain_forget_frontier.png", "Retain-forget frontier", "No frontier rows were available.")
        return
    specs = [
        ("edit_method_frontier.png", "Edit method frontier", "target_control_gap", "mean_retain_damage", "localized-control target gap", "mean retain damage"),
        ("unlearning_retain_forget_frontier.png", "Retain-forget frontier", "localized_target_gain", "mean_retain_damage", "localized target gain", "mean retain damage"),
    ]
    for name, title, xkey, ykey, xlabel, ylabel in specs:
        xs = [as_float(row.get(xkey), 0.0) for row in rows]
        ys = [as_float(row.get(ykey), 0.0) for row in rows]
        fig, ax = plt.subplots(figsize=(7.8, 5.2))
        ax.scatter(xs, ys)
        _annotate_indices(ax, xs, ys)
        ax.axhline(RETAIN_DAMAGE_SOFT_LIMIT, linestyle="--", linewidth=0.9)
        ax.axvline(0.0, linestyle=":", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        bench.save_figure(ctx, fig, name, f"{title}: target movement versus retain damage.")


def plot_locality_ladder(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if not rows:
        _plot_empty(ctx, "mechanistic_locality_ladder.png", "Mechanistic locality ladder", "No selected-depth localization rows were available.")
        return
    targets = sorted({str(row.get("target_id")) for row in rows})
    methods = ["localized_patch", "wrong_position_patch", "random_direction_patch"]
    x = np.arange(len(targets))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(9, len(targets) * 1.1), 5.0))
    for m_i, method in enumerate(methods):
        vals = []
        for tid in targets:
            vals.append(safe_mean([row.get("patch_gain") for row in rows if row.get("target_id") == tid and row.get("method") == method], 0.0))
        ax.bar(x + (m_i - 1) * width, vals, width, label=localization_method_label(method))
    ax.axhline(0.0, linewidth=0.8)
    ax.axhline(LOCALITY_GAP_MIN, linestyle="--", linewidth=0.9, label="locality gap gate")
    ax.set_xticks(x, [short_target_label(t, 18) for t in targets], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("patch_gain at selected depth")
    ax.set_title("Localized donor patch beside localization controls")
    ax.legend(fontsize=8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "mechanistic_locality_ladder.png", "Localized patch gain versus wrong-position and random-direction localization controls.")


def plot_scale_selection_ladder(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    loc = [row for row in rows if row.get("method") == "localized_addition" and as_float(row.get("scale")) > 0.0]
    if not loc:
        _plot_empty(ctx, "scale_selection_ladder.png", "Scale selection ladder", "No localized dose-response rows were available.")
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for tid in sorted({str(row.get("target_id")) for row in loc}):
        rs = sorted([row for row in loc if row.get("target_id") == tid], key=lambda row: as_float(row.get("scale")))
        ax.plot(
            [as_float(row.get("scale")) for row in rs],
            [as_float(row.get("target_control_gap_at_scale"), 0.0) for row in rs],
            marker="o",
            label=short_target_label(tid, 16),
        )
        for row in rs:
            if boolish(row.get("is_selected_scale")):
                ax.scatter([as_float(row.get("scale"))], [as_float(row.get("target_control_gap_at_scale"), 0.0)], marker="s", s=60)
    ax.axhline(0.0, linewidth=0.8)
    ax.axhline(TARGET_CONTROL_GAP_MIN, linestyle="--", linewidth=0.9, label="control-gap gate")
    ax.set_xlabel("scale")
    ax.set_ylabel("localized gain minus strongest control")
    ax.set_title("Scale selection ladder")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "scale_selection_ladder.png", "Target-control gap by scale, with selected scales marked.")


def plot_paraphrase_matrix(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if not rows:
        _plot_empty(ctx, "paraphrase_robustness_matrix.png", "Paraphrase robustness matrix", "No localized paraphrase rows were available.")
        return
    targets = sorted({str(row.get("target_id")) for row in rows})
    roles = sorted({str(row.get("eval_role") or "paraphrase") for row in rows})
    mat = np.full((len(targets), len(roles)), np.nan)
    for i, tid in enumerate(targets):
        for j, role in enumerate(roles):
            vals = [as_float(row.get("transfer_gain")) for row in rows if row.get("target_id") == tid and str(row.get("eval_role") or "paraphrase") == role]
            vals = finite_values(vals)
            if vals:
                mat[i, j] = safe_mean(vals)
    fig, ax = plt.subplots(figsize=(max(7.5, len(roles) * 0.8), max(3.8, len(targets) * 0.5)))
    im = ax.imshow(np.nan_to_num(mat, nan=0.0), aspect="auto")
    ax.set_yticks(range(len(targets)), [short_target_label(t, 24) for t in targets], fontsize=7)
    ax.set_xticks(range(len(roles)), roles, rotation=35, ha="right", fontsize=7)
    ax.set_title("Paraphrase transfer at chosen scale")
    fig.colorbar(im, ax=ax, shrink=0.82, label="transfer_gain")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "paraphrase_robustness_matrix.png", "Paraphrase transfer heatmap for localized additions.")


def plot_retain_neighbor_atlas(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if not rows:
        _plot_empty(ctx, "neighbor_preservation_atlas.png", "Neighbor and retain preservation atlas", "No localized retain/neighbor rows were available.")
        return
    targets = sorted({str(row.get("target_id")) for row in rows})
    roles = sorted({str(row.get("display_role")) for row in rows})
    mat = np.full((len(targets), len(roles)), np.nan)
    for i, tid in enumerate(targets):
        for j, role in enumerate(roles):
            vals = [as_float(row.get("damage")) for row in rows if row.get("target_id") == tid and str(row.get("display_role")) == role]
            vals = finite_values(vals)
            if vals:
                mat[i, j] = safe_mean(vals)
    fig, ax = plt.subplots(figsize=(max(8.5, len(roles) * 0.62), max(3.8, len(targets) * 0.5)))
    im = ax.imshow(np.nan_to_num(mat, nan=0.0), aspect="auto")
    ax.set_yticks(range(len(targets)), [short_target_label(t, 24) for t in targets], fontsize=7)
    ax.set_xticks(range(len(roles)), [str(i + 1) for i in range(len(roles))], fontsize=7)
    ax.set_xlabel("retain/neighbor prompt index; inspect source table for text")
    ax.set_title("Retain and neighbor damage atlas")
    fig.colorbar(im, ax=ax, shrink=0.82, label="damage")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "neighbor_preservation_atlas.png", "Retain and neighbor damage heatmap for localized additions.")


def write_plots(
    ctx: bench.RunContext,
    evidence: Sequence[Mapping[str, Any]],
    localization_rows: Sequence[Mapping[str, Any]],
    edit_rows: Sequence[Mapping[str, Any]],
    retain_rows: Sequence[Mapping[str, Any]],
    paraphrase_rows: Sequence[Mapping[str, Any]],
    scale_choice: Mapping[str, Mapping[str, Any]],
    counterexamples: Sequence[Mapping[str, Any]] = (),
) -> None:
    write_plot_guide(ctx)
    sources = write_plot_source_tables(
        ctx,
        evidence,
        localization_rows,
        edit_rows,
        retain_rows,
        paraphrase_rows,
        scale_choice,
        counterexamples,
    )
    write_plot_manifest(ctx, sources, ctx.args.no_plots)
    if ctx.args.no_plots:
        return

    dashboard_rows = dashboard_source_rows(evidence)
    target_control_rows = target_vs_control_source_rows(evidence, edit_rows, scale_choice)
    dose_rows = dose_response_source_rows(edit_rows, scale_choice)
    paired_rows = paired_examples_source_rows(evidence, edit_rows, retain_rows, paraphrase_rows, scale_choice)
    loc_edit_rows = localization_editability_source_rows(evidence)
    locality_rows = locality_ladder_source_rows(evidence, localization_rows)
    layer_sweep_rows = layer_sweep_heatmap_source_rows(localization_rows)
    para_rows = paraphrase_matrix_source_rows(paraphrase_rows)
    atlas_rows = retain_neighbor_atlas_source_rows(retain_rows)
    frontier_rows = frontier_source_rows(evidence)

    plot_dashboard(ctx, dashboard_rows)
    plot_target_vs_control(ctx, target_control_rows)
    plot_dose_response(ctx, dose_rows)
    plot_layer_sweep_heatmap(ctx, layer_sweep_rows)
    plot_paired_examples(ctx, paired_rows)
    plot_localization_vs_editability(ctx, loc_edit_rows)
    plot_frontiers(ctx, frontier_rows)
    plot_locality_ladder(ctx, locality_rows)
    plot_scale_selection_ladder(ctx, dose_rows)
    plot_paraphrase_matrix(ctx, para_rows)
    plot_retain_neighbor_atlas(ctx, atlas_rows)


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


def write_self_check_status(
    ctx: bench.RunContext,
    token_rows: Sequence[Mapping[str, Any]],
    safety_status: Mapping[str, Any],
    noop_rows: Sequence[Mapping[str, Any]],
    reversibility_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    status = {
        "tokenization_kept": sum(1 for row in token_rows if row.get("kept") is True),
        "tokenization_dropped": sum(1 for row in token_rows if row.get("kept") is False),
        "safety_blocked_rows": safety_status.get("blocked_rows", 0),
        "noop_max_abs_delta": max([as_float(row.get("max_abs_delta"), 0.0) for row in noop_rows] or [0.0]),
        "reversibility_max_abs_delta": max([as_float(row.get("abs_delta"), 0.0) for row in reversibility_rows] or [0.0]),
        "noop_atol": NOOP_DELTA_ATOL,
        "reversibility_atol": REVERSIBILITY_ATOL,
    }
    status["ok"] = bool(
        status["tokenization_kept"] > 0
        and status["safety_blocked_rows"] == 0
        and status["noop_max_abs_delta"] <= NOOP_DELTA_ATOL
        and status["reversibility_max_abs_delta"] <= REVERSIBILITY_ATOL
    )
    path = ctx.path("diagnostics", "self_check_status.json")
    bench.write_json(path, status)
    ctx.register_artifact(path, "diagnostic", "Lab 28 local tokenization, safety, no-op, and reversibility self-check status.")
    return status


def write_method_capability_audit(ctx: bench.RunContext) -> None:
    rows = [
        {"method": "localized_residual_addition", "status": "run", "why": "safe reversible inference-time intervention"},
        {"method": "wrong_position_addition", "status": "run", "why": "site specificity control"},
        {"method": "random_direction_addition", "status": "run", "why": "direction specificity control"},
        {"method": "opposite_direction_addition", "status": "run", "why": "signedness control"},
        {"method": "feature_clamp_suppression", "status": "not_run", "why": "requires aligned feature dictionary; future extension"},
        {"method": "persistent_rank_one_weight_edit", "status": "not_run", "why": "requires apply/restore hash and rollback tests; intentionally deferred"},
        {"method": "refusal_ablation", "status": "forbidden", "why": "outside Lab 28 safety scope"},
        {"method": "private_data_unlearning", "status": "forbidden", "why": "outside Lab 28 safety scope"},
    ]
    path = ctx.path("tables", "method_capability_audit.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Methods run, not run, or forbidden in Lab 28.")


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    targets, data_info = load_targets(ctx)
    manifest_path = ctx.path("diagnostics", "data_manifest.json")
    bench.write_json(manifest_path, data_info)
    ctx.register_artifact(manifest_path, "diagnostic", "Lab 28 data manifest and science-ready status.")
    safety_status = write_safety_status(ctx, targets, data_info)
    targets, token_rows = tokenization_gate(ctx, bundle, targets)

    bench.run_hook_parity_check(ctx, bundle, targets[0].prompt)
    first = bench.run_with_residual_cache(bundle, targets[0].prompt)
    bench.run_lens_self_check(ctx, bundle, first)
    bench.run_patch_noop_check(ctx, bundle, targets[0].prompt)

    target_caps, donor_caps, baseline_rows = cache_baselines(ctx, bundle, targets)
    localization_rows, best_by_target = localize_sites(ctx, bundle, targets, target_caps, donor_caps)
    noop_rows = write_noop_identity_checks(ctx, localization_rows)
    edit_rows, vectors, _metadata = run_edits(ctx, bundle, targets, target_caps, donor_caps, best_by_target)
    scale_choice = choose_scales(ctx, targets, edit_rows)
    retain_rows, paraphrase_rows = evaluate_side_sets(ctx, bundle, targets, scale_choice, vectors)
    reversibility_rows = run_reversibility_check(ctx, bundle, targets)
    self_check_status = write_self_check_status(ctx, token_rows, safety_status, noop_rows, reversibility_rows)

    evidence, counterexamples, refinement_rows, metrics = summarize_evidence(
        targets, best_by_target, edit_rows, scale_choice, retain_rows, paraphrase_rows
    )
    evidence_path = ctx.path("tables", "edit_evidence_matrix.csv")
    bench.write_csv_with_context(ctx, evidence_path, evidence)
    ctx.register_artifact(evidence_path, "table", "Target-level Lab 28 evidence matrix.")
    generic_evidence_path = ctx.path("tables", "evidence_matrix.csv")
    bench.write_csv_with_context(ctx, generic_evidence_path, evidence)
    ctx.register_artifact(generic_evidence_path, "table", "Standard-named copy of the Lab 28 evidence matrix.")

    counter_path = ctx.path("tables", "edit_counterexamples.csv")
    bench.write_csv_with_context(ctx, counter_path, counterexamples)
    ctx.register_artifact(counter_path, "table", "Counterexamples where controls, retain damage, or paraphrase failures limit the edit claim.")
    failure_jsonl, failure_md = write_failure_specimens(ctx, counterexamples)
    refinement_path = ctx.path("tables", "edit_refinement_log.csv")
    bench.write_csv_with_context(ctx, refinement_path, refinement_rows)
    ctx.register_artifact(refinement_path, "table", "Suggested v2 target/donor/edit refinements driven by failed gates.")

    results_path = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results_path, edit_rows)
    ctx.register_artifact(results_path, "table", "CSV alias of Lab 28 target edit dose-response rows.")
    jsonl_path = ctx.path("results.jsonl")
    write_jsonl(jsonl_path, [{**ctx.table_context(), **row} for row in edit_rows])
    ctx.register_artifact(jsonl_path, "table", "JSONL copy of Lab 28 target edit dose-response rows.")

    write_method_capability_audit(ctx)
    warning_rows = write_warning_summary(
        ctx,
        data_info,
        token_rows,
        baseline_rows,
        localization_rows,
        edit_rows,
        retain_rows,
        paraphrase_rows,
        evidence,
        scale_choice,
        self_check_status,
    )
    run_config_snapshot = write_lab28_run_config_snapshot(ctx, bundle, data_info, targets, scale_choice, evidence)
    metrics_full = {
        "lab_id": LAB_ID,
        "lab_name": LAB_NAME,
        "data": data_info,
        "safety_status": safety_status,
        "self_check_status": self_check_status,
        "warning_summary": warning_rows,
        "run_config_snapshot_path": "diagnostics/lab28_run_config_snapshot.json",
        "n_localization_rows": len(localization_rows),
        "n_edit_rows": len(edit_rows),
        "n_retain_rows": len(retain_rows),
        "n_paraphrase_rows": len(paraphrase_rows),
        **metrics,
    }
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics_full)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 28 metrics and verdicts.")

    write_method_card(ctx, bundle, evidence, data_info)
    write_spec_card(ctx, data_info)
    write_operationalization_audit(ctx, evidence, counterexamples)
    write_run_summary(ctx, bundle, data_info, metrics_full, evidence, counterexamples)
    write_claims(ctx, evidence)
    write_plots(ctx, evidence, localization_rows, edit_rows, retain_rows, paraphrase_rows, scale_choice, counterexamples)
    print(f"[lab28] wrote {len(evidence)} evidence rows and {len(counterexamples)} counterexamples")
