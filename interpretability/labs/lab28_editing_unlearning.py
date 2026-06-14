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
# Cards and summaries
# ---------------------------------------------------------------------------


def write_plot_guide(ctx: bench.RunContext) -> None:
    rows = [
        {"plot": "plots/editing_unlearning_dashboard.png", "read_for": "Target gain, control gap, paraphrase transfer, retain damage, and posture in one place.", "do_not_claim": "Dashboard positivity is not persistent unlearning."},
        {"plot": "plots/localization_vs_editability.png", "read_for": "Whether replacement-patch localization predicts additive edit strength.", "do_not_claim": "A good patch site is automatically the best editing site."},
        {"plot": "plots/edit_method_frontier.png", "read_for": "Target gain versus side-effect damage.", "do_not_claim": "High target movement is useful if retain damage is high or controls match it."},
        {"plot": "plots/mechanistic_locality_ladder.png", "read_for": "Localized patch gain compared with wrong-position and random-direction patch controls.", "do_not_claim": "Locality passed if the control floor is high."},
        {"plot": "plots/scale_selection_ladder.png", "read_for": "Why the chosen scale was selected before side-set audits.", "do_not_claim": "The largest scale is the best evidence."},
        {"plot": "plots/paraphrase_robustness_matrix.png", "read_for": "Transfer beyond exact target strings.", "do_not_claim": "Exact-prompt movement is semantic transfer."},
        {"plot": "plots/neighbor_preservation_atlas.png", "read_for": "Retain and neighbor damage hot spots.", "do_not_claim": "Side effects are irrelevant because the target moved."},
        {"plot": "plots/unlearning_retain_forget_frontier.png", "read_for": "Target movement versus retain damage.", "do_not_claim": "The fact was erased from weights."},
    ]
    path = ctx.path("tables", "plot_reading_guide.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Reading guide for Lab 28 plot suite.")


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
        "5. `tables/edit_counterexamples.csv` before writing positive language.",
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


def write_plots(
    ctx: bench.RunContext,
    evidence: Sequence[Mapping[str, Any]],
    localization_rows: Sequence[Mapping[str, Any]],
    edit_rows: Sequence[Mapping[str, Any]],
    retain_rows: Sequence[Mapping[str, Any]],
    paraphrase_rows: Sequence[Mapping[str, Any]],
    scale_choice: Mapping[str, Mapping[str, Any]],
) -> None:
    write_plot_guide(ctx)
    if ctx.args.no_plots:
        return
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [str(row["target_id"]).replace("edit_", "")[:20] for row in evidence]
    x = np.arange(len(labels))
    if not evidence:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No evidence rows", ha="center", va="center")
        ax.axis("off")
        bench.save_figure(ctx, fig, "editing_unlearning_dashboard.png", "Empty Lab 28 dashboard.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Lab 28 editing/unlearning dashboard", fontsize=14, fontweight="bold")
    width = 0.22
    axes[0, 0].bar(x - width, [as_float(r.get("localized_target_gain"), 0.0) for r in evidence], width, label="target gain")
    axes[0, 0].bar(x, [as_float(r.get("target_control_floor"), 0.0) for r in evidence], width, label="control floor")
    axes[0, 0].bar(x + width, [as_float(r.get("target_control_gap"), 0.0) for r in evidence], width, label="control gap")
    axes[0, 0].axhline(0.0, linewidth=0.8)
    axes[0, 0].set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
    axes[0, 0].set_title("Target dose-response at selected scale")
    axes[0, 0].set_ylabel("margin movement")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].bar(x - width / 2, [as_float(r.get("mean_paraphrase_gain"), 0.0) for r in evidence], width, label="paraphrase gain")
    axes[0, 1].bar(x + width / 2, [as_float(r.get("mean_retain_damage"), 0.0) for r in evidence], width, label="retain damage")
    axes[0, 1].axhline(0.0, linewidth=0.8)
    axes[0, 1].set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
    axes[0, 1].set_title("Transfer and side effects")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].scatter(
        [as_float(r.get("localization_patch_gain"), 0.0) for r in evidence],
        [as_float(r.get("localized_target_gain"), 0.0) for r in evidence],
    )
    for i, row in enumerate(evidence):
        axes[1, 0].annotate(str(i + 1), (as_float(row.get("localization_patch_gain"), 0.0), as_float(row.get("localized_target_gain"), 0.0)), fontsize=8)
    axes[1, 0].axhline(0.0, linewidth=0.8)
    axes[1, 0].axvline(0.0, linewidth=0.8)
    axes[1, 0].set_xlabel("localization patch gain")
    axes[1, 0].set_ylabel("additive edit gain")
    axes[1, 0].set_title("Localization vs editability")

    posture = [1 if row.get("claim_posture") == "localized_edit_supported" else 0 for row in evidence]
    axes[1, 1].bar(x, posture)
    axes[1, 1].set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
    axes[1, 1].set_yticks([0, 1], ["needs review", "supported"])
    axes[1, 1].set_title("Claim posture")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    bench.save_figure(ctx, fig, "editing_unlearning_dashboard.png", "Lab 28 dashboard: target gain, controls, transfer, side effects, posture.")

    plot_specs = [
        ("localization_vs_editability.png", "Localization vs editability", [as_float(r.get("localization_patch_gain"), 0.0) for r in evidence], [as_float(r.get("localized_target_gain"), 0.0) for r in evidence], "localization patch gain", "edit gain"),
        ("edit_method_frontier.png", "Edit method frontier", [as_float(r.get("target_control_gap"), 0.0) for r in evidence], [as_float(r.get("mean_retain_damage"), 0.0) for r in evidence], "target control gap", "retain damage"),
        ("unlearning_retain_forget_frontier.png", "Retain-forget frontier", [as_float(r.get("localized_target_gain"), 0.0) for r in evidence], [as_float(r.get("mean_retain_damage"), 0.0) for r in evidence], "target gain", "retain damage"),
    ]
    for name, title, xs, ys, xlabel, ylabel in plot_specs:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.scatter(xs, ys)
        for i, lab in enumerate(labels):
            ax.annotate(str(i + 1), (xs[i], ys[i]), fontsize=8)
        ax.axhline(0.0, linestyle=":", linewidth=0.8)
        ax.axvline(0.0, linestyle=":", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        bench.save_figure(ctx, fig, name, title + ".")

    # Locality ladder at selected depth.
    loc_vals = []
    wrong_vals = []
    rand_vals = []
    for row in evidence:
        tid = row["target_id"]
        depth = row["best_depth"]
        loc_vals.append(safe_mean([r.get("patch_gain") for r in localization_rows if r.get("target_id") == tid and r.get("method") == "localized_patch" and str(r.get("depth")) == str(depth)], 0.0))
        wrong_vals.append(safe_mean([r.get("patch_gain") for r in localization_rows if r.get("target_id") == tid and r.get("method") == "wrong_position_patch" and str(r.get("depth")) == str(depth)], 0.0))
        rand_vals.append(safe_mean([r.get("patch_gain") for r in localization_rows if r.get("target_id") == tid and r.get("method") == "random_direction_patch" and str(r.get("depth")) == str(depth)], 0.0))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - width, loc_vals, width, label="localized patch")
    ax.bar(x, wrong_vals, width, label="wrong position")
    ax.bar(x + width, rand_vals, width, label="random direction")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
    ax.set_title("Mechanistic locality ladder")
    ax.set_ylabel("patch gain")
    ax.legend(fontsize=8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "mechanistic_locality_ladder.png", "Localized patch gain versus localization controls.")

    # Scale selection ladder.
    scale_rows = []
    for row in edit_rows:
        if row.get("method") == "localized_addition":
            tid = str(row["target_id"])
            scale = as_float(row.get("scale"))
            scale_rows.append((tid, scale, as_float(row.get("target_gain")), control_floor_for_scale(edit_rows, tid, scale)))
    fig, ax = plt.subplots(figsize=(9, 5))
    for tid in sorted({r[0] for r in scale_rows}):
        rs = sorted([r for r in scale_rows if r[0] == tid], key=lambda r: r[1])
        ax.plot([r[1] for r in rs], [r[2] - r[3] for r in rs], marker="o", label=tid.replace("edit_", "")[:16])
        chosen = as_float(scale_choice[tid]["scale"], 0.0)
        ax.axvline(chosen, linestyle=":", linewidth=0.7)
    ax.axhline(TARGET_CONTROL_GAP_MIN, linestyle="--", linewidth=1, label="target gate")
    ax.set_xlabel("scale")
    ax.set_ylabel("localized gain minus control floor")
    ax.set_title("Scale selection ladder")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "scale_selection_ladder.png", "Scale selection control-gap ladder.")

    # Paraphrase matrix.
    para_targets = sorted({str(r["target_id"]) for r in paraphrase_rows if r.get("method") == "localized_addition"})
    para_roles = sorted({str(r.get("eval_role") or i) for i, r in enumerate(paraphrase_rows) if r.get("method") == "localized_addition"})
    if para_targets and para_roles:
        mat = np.zeros((len(para_targets), len(para_roles)))
        for i, tid in enumerate(para_targets):
            for j, role in enumerate(para_roles):
                vals = [as_float(r.get("transfer_gain")) for r in paraphrase_rows if r.get("target_id") == tid and r.get("method") == "localized_addition" and str(r.get("eval_role")) == role]
                mat[i, j] = safe_mean(vals, 0.0)
        fig, ax = plt.subplots(figsize=(max(7, len(para_roles) * 0.75), max(3.5, len(para_targets) * 0.45)))
        im = ax.imshow(mat, aspect="auto")
        ax.set_yticks(range(len(para_targets)), [t.replace("edit_", "")[:24] for t in para_targets], fontsize=7)
        ax.set_xticks(range(len(para_roles)), para_roles, rotation=35, ha="right", fontsize=7)
        ax.set_title("Paraphrase robustness matrix")
        fig.colorbar(im, ax=ax, shrink=0.8, label="transfer gain")
        fig.tight_layout()
        bench.save_figure(ctx, fig, "paraphrase_robustness_matrix.png", "Paraphrase transfer heatmap.")

    # Retain/neighbor atlas.
    atlas_targets = sorted({str(r["target_id"]) for r in retain_rows if r.get("method") == "localized_addition"})
    atlas_roles = sorted({f"{r['eval_family']}:{r['eval_role']}" for r in retain_rows if r.get("method") == "localized_addition"})
    if atlas_targets and atlas_roles:
        atlas = np.zeros((len(atlas_targets), len(atlas_roles)))
        for i, tid in enumerate(atlas_targets):
            for j, role in enumerate(atlas_roles):
                vals = [as_float(r.get("damage")) for r in retain_rows if r.get("target_id") == tid and r.get("method") == "localized_addition" and f"{r['eval_family']}:{r['eval_role']}" == role]
                atlas[i, j] = safe_mean(vals, 0.0)
        fig, ax = plt.subplots(figsize=(max(8, len(atlas_roles) * 0.65), max(3.5, len(atlas_targets) * 0.45)))
        im = ax.imshow(atlas, aspect="auto")
        ax.set_yticks(range(len(atlas_targets)), [t.replace("edit_", "")[:24] for t in atlas_targets], fontsize=7)
        ax.set_xticks(range(len(atlas_roles)), [str(i + 1) for i in range(len(atlas_roles))], fontsize=7)
        ax.set_title("Neighbor and retain preservation atlas")
        ax.set_xlabel("retain/neighbor prompt index")
        fig.colorbar(im, ax=ax, shrink=0.8, label="damage")
        fig.tight_layout()
        bench.save_figure(ctx, fig, "neighbor_preservation_atlas.png", "Retain and neighbor damage heatmap.")


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
    metrics_full = {
        "lab_id": LAB_ID,
        "lab_name": LAB_NAME,
        "data": data_info,
        "safety_status": safety_status,
        "self_check_status": self_check_status,
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
    write_plots(ctx, evidence, localization_rows, edit_rows, retain_rows, paraphrase_rows, scale_choice)
    print(f"[lab28] wrote {len(evidence)} evidence rows and {len(counterexamples)} counterexamples")
