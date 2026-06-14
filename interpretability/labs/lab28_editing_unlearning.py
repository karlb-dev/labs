"""Lab 28: Mechanistic editing and unlearning.

This first implementation deliberately stays on the safe side of the editing
literature: it performs reversible inference-time activation additions, never
persists weight changes, and audits target, paraphrase, neighbor, and retain
sets before suggesting a claim. The word "unlearning" is treated as a behavior
measurement problem, not as evidence that a fact has been erased from weights.

Evidence level: CAUSAL + AUDIT, scoped to localized residual additions on
benign public or synthetic associations.
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

LAB_ID = "L28"
DATA_FILE = "editing_unlearning_targets.csv"
PROMPT_SET_CAPS = {"small": 5, "medium": 5, "full": 0}
EDIT_SCALES = (0.0, 0.5, 1.0, 1.5, 2.0)
SCIENCE_READY_MIN_ROWS = 5
RETAIN_DAMAGE_SOFT_LIMIT = 2.0
NEIGHBOR_DAMAGE_CAVEAT = 3.0


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
    before_id: int = -1
    after_id: int = -1
    prompt_ids: list[int] = dataclasses.field(default_factory=list)
    donor_ids: list[int] = dataclasses.field(default_factory=list)
    final_pos: int = -1
    donor_final_pos: int = -1
    base_after_margin: float = float("nan")


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


def safe_max(values: Sequence[Any], default: float = float("nan")) -> float:
    vals: list[float] = []
    for value in values:
        try:
            f = float(value)
        except Exception:
            continue
        if math.isfinite(f):
            vals.append(f)
    return max(vals) if vals else default


def safe_corr(xs: Sequence[Any], ys: Sequence[Any]) -> float:
    pairs: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        try:
            xf = float(x)
            yf = float(y)
        except Exception:
            continue
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


def data_path(args: Any) -> pathlib.Path:
    prompt_set = str(getattr(args, "prompt_set", "") or "")
    candidate = pathlib.Path(prompt_set)
    if prompt_set not in PROMPT_SET_CAPS and candidate.suffix.lower() in {".csv", ".tsv"}:
        return candidate if candidate.is_absolute() else (bench.COURSE_ROOT / candidate).resolve()
    return bench.COURSE_ROOT / "data" / DATA_FILE


def parse_eval_prompts(raw: str, *, default_target: str, default_distractor: str) -> list[EvalPrompt]:
    payload = json.loads(raw or "[]")
    prompts: list[EvalPrompt] = []
    for i, item in enumerate(payload):
        if isinstance(item, str):
            prompts.append(EvalPrompt(prompt=item, target=default_target, distractor=default_distractor, role=f"prompt_{i}"))
            continue
        prompts.append(
            EvalPrompt(
                prompt=str(item["prompt"]),
                target=str(item.get("target", default_target)),
                distractor=str(item.get("distractor", default_distractor)),
                role=str(item.get("role", f"prompt_{i}")),
            )
        )
    return prompts


def load_targets(ctx: bench.RunContext) -> tuple[list[EditTarget], dict[str, Any]]:
    path = data_path(ctx.args)
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    targets: list[EditTarget] = []
    for row in rows:
        target = EditTarget(
            target_id=row["target_id"],
            family=row["family"],
            edit_type=row["edit_type"],
            prompt=row["prompt"],
            target_before=row["target_before"],
            target_after=row["target_after"],
            retain_prompts=parse_eval_prompts(
                row["retain_prompts_json"],
                default_target=row["target_before"],
                default_distractor=row["target_after"],
            ),
            paraphrase_prompts=parse_eval_prompts(
                row["paraphrase_prompts_json"],
                default_target=row["target_after"],
                default_distractor=row["target_before"],
            ),
            neighbor_prompts=parse_eval_prompts(
                row["neighbor_prompts_json"],
                default_target=row["target_before"],
                default_distractor=row["target_after"],
            ),
            safety_notes=row["safety_notes"],
            donor_prompt=row.get("donor_prompt") or row["prompt"],
        )
        targets.append(target)

    cap = PROMPT_SET_CAPS.get(str(ctx.args.prompt_set), 0)
    if cap:
        targets = targets[:cap]
    if int(ctx.args.max_examples or 0) > 0:
        targets = targets[: int(ctx.args.max_examples)]
    info = {
        "data_path": str(path),
        "sha256": file_sha256(path),
        "n_rows_file": len(rows),
        "n_rows_selected": len(targets),
        "families": {f: sum(1 for t in targets if t.family == f) for f in sorted({t.family for t in targets})},
        "edit_types": {e: sum(1 for t in targets if t.edit_type == e) for e in sorted({t.edit_type for t in targets})},
        "science_ready": len(targets) >= SCIENCE_READY_MIN_ROWS,
        "safe_scope": "benign_public_or_synthetic_associations; inference_time_only; no persistent_weight_edits",
    }
    return targets, info


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def tokenization_gate(ctx: bench.RunContext, bundle: bench.ModelBundle, targets: list[EditTarget]) -> list[EditTarget]:
    tok = bundle.tokenizer
    kept: list[EditTarget] = []
    rows: list[dict[str, Any]] = []
    for target in targets:
        problems: list[str] = []
        before_ids = token_ids(tok, target.target_before)
        after_ids = token_ids(tok, target.target_after)
        prompt_ids = token_ids(tok, target.prompt)
        donor_ids = token_ids(tok, target.donor_prompt)
        if len(before_ids) != 1:
            problems.append(f"target_before_tokens={len(before_ids)}")
        if len(after_ids) != 1:
            problems.append(f"target_after_tokens={len(after_ids)}")
        if not prompt_ids:
            problems.append("empty_prompt")
        if not donor_ids:
            problems.append("empty_donor_prompt")
        eval_bad = 0
        for collection in (target.retain_prompts, target.paraphrase_prompts, target.neighbor_prompts):
            for ep in collection:
                if len(token_ids(tok, ep.target)) != 1 or len(token_ids(tok, ep.distractor)) != 1:
                    eval_bad += 1
        if eval_bad:
            problems.append(f"eval_prompt_token_failures={eval_bad}")
        if not problems:
            target.before_id = before_ids[0]
            target.after_id = after_ids[0]
            target.prompt_ids = prompt_ids
            target.donor_ids = donor_ids
            target.final_pos = len(prompt_ids) - 1
            target.donor_final_pos = len(donor_ids) - 1
            kept.append(target)
        rows.append({
            "target_id": target.target_id,
            "family": target.family,
            "before_token_count": len(before_ids),
            "after_token_count": len(after_ids),
            "prompt_tokens": len(prompt_ids),
            "donor_prompt_tokens": len(donor_ids),
            "retain_prompts": len(target.retain_prompts),
            "paraphrase_prompts": len(target.paraphrase_prompts),
            "neighbor_prompts": len(target.neighbor_prompts),
            "kept": not problems,
            "problems": ";".join(problems),
            "prompt_tokenization": " | ".join(f"{i}:{tok.decode([tid])}" for i, tid in enumerate(prompt_ids)),
            "before_id": before_ids[0] if len(before_ids) == 1 else "",
            "after_id": after_ids[0] if len(after_ids) == 1 else "",
        })
    path = ctx.path("diagnostics", "tokenization_gate.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "diagnostic", "Single-token and prompt-tokenization audit for Lab 28 targets.")
    if not kept:
        raise RuntimeError("Lab 28 tokenization gate dropped every target.")
    return kept


def coarse_depths(n_layers: int, prompt_set: str) -> list[int]:
    if prompt_set == "full":
        return list(range(n_layers + 1))
    return sorted({0, max(1, n_layers // 4), max(1, n_layers // 2), max(1, (3 * n_layers) // 4), n_layers})


def logit_margin(logits: Any, target_id: int, distractor_id: int) -> float:
    return float(logits[target_id] - logits[distractor_id])


def run_with_residual_addition(
    bundle: bench.ModelBundle,
    prompt: str,
    layer: int,
    position: int,
    vector: Any,
    scale: float,
) -> Any:
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


def deterministic_random_like(vector: Any, key: str) -> Any:
    import torch

    gen = torch.Generator(device="cpu")
    gen.manual_seed(stable_int(key) % (2**31 - 1))
    rand = torch.randn(vector.shape, generator=gen, dtype=vector.dtype)
    norm = vector.float().norm().clamp_min(1e-8)
    return rand / rand.float().norm().clamp_min(1e-8) * norm


def eval_ids(bundle: bench.ModelBundle, ep: EvalPrompt) -> tuple[int, int]:
    tok = bundle.tokenizer
    target_ids = token_ids(tok, ep.target)
    distractor_ids = token_ids(tok, ep.distractor)
    if len(target_ids) != 1 or len(distractor_ids) != 1:
        raise ValueError(f"Eval prompt `{ep.prompt}` has non-single-token target/distractor.")
    return target_ids[0], distractor_ids[0]


def cache_baselines(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    targets: list[EditTarget],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    target_caps: dict[str, Any] = {}
    donor_caps: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for target in targets:
        cap = bench.run_with_residual_cache(bundle, target.prompt)
        donor = bench.run_with_residual_cache(bundle, target.donor_prompt)
        target_caps[target.target_id] = cap
        donor_caps[target.target_id] = donor
        target.final_pos = len(cap.input_ids) - 1
        target.donor_final_pos = len(donor.input_ids) - 1
        target.base_after_margin = logit_margin(cap.final_logits_last, target.after_id, target.before_id)
        donor_after_margin = logit_margin(donor.final_logits_last, target.after_id, target.before_id)
        rows.append({
            "target_id": target.target_id,
            "family": target.family,
            "prompt": target.prompt,
            "donor_prompt": target.donor_prompt,
            "target_before": target.target_before,
            "target_after": target.target_after,
            "base_after_minus_before": rounded(target.base_after_margin),
            "base_before_minus_after": rounded(-target.base_after_margin),
            "donor_after_minus_before": rounded(donor_after_margin),
            "baseline_prefers_before": target.base_after_margin < 0.0,
            "donor_supports_after": donor_after_margin > 0.0,
            "base_top": bundle.tokenizer.decode([int(cap.final_logits_last.argmax())]),
            "donor_top": bundle.tokenizer.decode([int(donor.final_logits_last.argmax())]),
        })
    path = ctx.path("tables", "baseline_behavior.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Target and donor baseline logit margins before any edit.")
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
    best_by_target: dict[str, dict[str, Any]] = {}
    total = len(targets) * len(depths) * 3
    done = 0
    report_every = max(1, total // 5)
    for target in targets:
        cap = target_caps[target.target_id]
        donor = donor_caps[target.target_id]
        wrong_pos = 0 if target.final_pos != 0 else -1
        for depth in depths:
            donor_vec = donor.streams[depth, target.donor_final_pos]
            localized_logits = bench.run_with_residual_patch(bundle, target.prompt, depth, target.final_pos, donor_vec)
            wrong_logits = bench.run_with_residual_patch(bundle, target.prompt, depth, wrong_pos, donor_vec)
            direction = donor_vec - cap.streams[depth, target.final_pos]
            rand_vec = cap.streams[depth, target.final_pos] + deterministic_random_like(direction, f"{target.target_id}|{depth}|localize")
            random_logits = bench.run_with_residual_patch(bundle, target.prompt, depth, target.final_pos, rand_vec)
            for method, logits, position in (
                ("localized_patch", localized_logits, target.final_pos),
                ("wrong_position_patch", wrong_logits, wrong_pos),
                ("random_direction_patch", random_logits, target.final_pos),
            ):
                margin = logit_margin(logits, target.after_id, target.before_id)
                row = {
                    "target_id": target.target_id,
                    "family": target.family,
                    "method": method,
                    "depth": depth,
                    "position": position,
                    "base_after_minus_before": rounded(target.base_after_margin),
                    "patched_after_minus_before": rounded(margin),
                    "patch_gain": rounded(margin - target.base_after_margin),
                    "localized_candidate": method == "localized_patch",
                }
                rows.append(row)
                if method == "localized_patch":
                    current = best_by_target.get(target.target_id)
                    if current is None or float(row["patch_gain"]) > float(current["patch_gain"]):
                        best_by_target[target.target_id] = dict(row)
                done += 1
                if done % report_every == 0:
                    print(f"[lab28] localization interventions {done}/{total}")
    print(f"[lab28] localization interventions {done}/{total}")
    path = ctx.path("tables", "localization_candidates.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Residual donor-patch localization candidates and controls.")
    return rows, best_by_target


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
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    vectors: dict[str, Any] = {}
    metadata: dict[str, Any] = {
        "lab": LAB_ID,
        "method": "reversible inference-time residual activation addition",
        "non_methods": {
            "persistent_weight_edit": "not run in this safe first pass",
            "refusal_ablation": "out of scope and forbidden",
            "private_data_unlearning": "out of scope and forbidden",
            "feature_clamp_suppression": "requires an aligned feature dictionary; logged as future work",
        },
        "targets": {},
    }
    methods = ("no_edit", "localized_addition", "wrong_position_addition", "random_direction_addition")
    total = len(targets) * (1 + (len(methods) - 1) * len(EDIT_SCALES))
    done = 0
    report_every = max(1, total // 5)
    for target in targets:
        best = best_by_target[target.target_id]
        depth = int(best["depth"])
        cap = target_caps[target.target_id]
        donor = donor_caps[target.target_id]
        direction = (donor.streams[depth, target.donor_final_pos] - cap.streams[depth, target.final_pos]).float().cpu()
        random_direction = deterministic_random_like(direction, f"{target.target_id}|{depth}|edit").float().cpu()
        wrong_pos = 0 if target.final_pos != 0 else -1
        vectors[target.target_id] = {
            "direction": direction,
            "random_direction": random_direction,
            "depth": depth,
            "target_position": target.final_pos,
            "wrong_position": wrong_pos,
        }
        metadata["targets"][target.target_id] = {
            "family": target.family,
            "prompt": target.prompt,
            "donor_prompt": target.donor_prompt,
            "target_before": target.target_before,
            "target_after": target.target_after,
            "depth": depth,
            "target_position": target.final_pos,
            "wrong_position": wrong_pos,
            "direction_norm": float(direction.norm()),
            "base_after_minus_before": float(target.base_after_margin),
        }
        for method in methods:
            for scale in EDIT_SCALES:
                if method == "no_edit" and scale != 0.0:
                    continue
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
                    "method": method,
                    "depth": depth,
                    "position": target.final_pos if method != "wrong_position_addition" else wrong_pos,
                    "scale": scale,
                    "direction_norm": rounded(float(direction.norm())),
                    "base_after_minus_before": rounded(target.base_after_margin),
                    "edited_after_minus_before": rounded(margin),
                    "target_gain": rounded(margin - target.base_after_margin),
                    "changed_to_after": margin > 0.0,
                    "safe_reversible": True,
                    "weight_edit": False,
                })
                done += 1
                if done % report_every == 0:
                    print(f"[lab28] edit interventions {done}/{total}")
    print(f"[lab28] edit interventions {done}/{total}")
    path = ctx.path("tables", "editing_results.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Main target edit results across localized and control methods.")
    state_path = ctx.path("state", "edit_vectors.pt")
    torch.save(vectors, state_path)
    ctx.register_artifact(state_path, "state", "Residual edit directions and deterministic random controls.")
    meta_path = ctx.path("state", "edit_metadata.json")
    bench.write_json(meta_path, metadata)
    ctx.register_artifact(meta_path, "state", "Human-readable metadata for Lab 28 edit vectors.")
    return rows, vectors, metadata


def choose_scale(edit_rows: Sequence[Mapping[str, Any]], target_id: str) -> float:
    rows = [r for r in edit_rows if r["target_id"] == target_id and r["method"] == "localized_addition"]
    if not rows:
        return 0.0
    crossing = [r for r in rows if float(r["scale"]) > 0.0 and float(r["edited_after_minus_before"]) > 0.0]
    if crossing:
        return float(min(crossing, key=lambda r: (float(r["scale"]), -float(r["target_gain"])))["scale"])
    return float(max(rows, key=lambda r: float(r["target_gain"]))["scale"])


def evaluate_side_sets(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    targets: list[EditTarget],
    edit_rows: Sequence[Mapping[str, Any]],
    vectors: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retain_rows: list[dict[str, Any]] = []
    paraphrase_rows: list[dict[str, Any]] = []
    methods = ("localized_addition", "wrong_position_addition", "random_direction_addition")
    for target in targets:
        vec_info = vectors[target.target_id]
        depth = int(vec_info["depth"])
        direction = vec_info["direction"]
        random_direction = vec_info["random_direction"]
        wrong_pos = int(vec_info["wrong_position"])
        scale = choose_scale(edit_rows, target.target_id)
        for eval_family, collection in (
            ("retain", target.retain_prompts),
            ("neighbor", target.neighbor_prompts),
        ):
            for ep in collection:
                tid, did = eval_ids(bundle, ep)
                base_margin = score_prompt_with_method(
                    bundle,
                    ep.prompt,
                    tid,
                    did,
                    method="no_edit",
                    depth=depth,
                    position=-1,
                    direction=direction,
                    random_direction=random_direction,
                    wrong_position=wrong_pos,
                    scale=0.0,
                )
                prompt_len = len(bench.run_with_residual_cache(bundle, ep.prompt).input_ids)
                pos = prompt_len - 1
                for method in methods:
                    edited_margin = score_prompt_with_method(
                        bundle,
                        ep.prompt,
                        tid,
                        did,
                        method=method,
                        depth=depth,
                        position=pos,
                        direction=direction,
                        random_direction=random_direction,
                        wrong_position=0 if pos != 0 else -1,
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
            base_margin = score_prompt_with_method(
                bundle,
                ep.prompt,
                tid,
                did,
                method="no_edit",
                depth=depth,
                position=-1,
                direction=direction,
                random_direction=random_direction,
                wrong_position=wrong_pos,
                scale=0.0,
            )
            prompt_len = len(bench.run_with_residual_cache(bundle, ep.prompt).input_ids)
            pos = prompt_len - 1
            for method in methods:
                edited_margin = score_prompt_with_method(
                    bundle,
                    ep.prompt,
                    tid,
                    did,
                    method=method,
                    depth=depth,
                    position=pos,
                    direction=direction,
                    random_direction=random_direction,
                    wrong_position=0 if pos != 0 else -1,
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
    ctx.register_artifact(retain_path, "table", "Retain and neighbor preservation matrix at chosen edit scale.")
    paraphrase_path = ctx.path("tables", "paraphrase_robustness.csv")
    bench.write_csv_with_context(ctx, paraphrase_path, paraphrase_rows)
    ctx.register_artifact(paraphrase_path, "table", "Paraphrase transfer results at chosen edit scale.")
    return retain_rows, paraphrase_rows


def summarize_evidence(
    targets: Sequence[EditTarget],
    localization_rows: Sequence[Mapping[str, Any]],
    edit_rows: Sequence[Mapping[str, Any]],
    retain_rows: Sequence[Mapping[str, Any]],
    paraphrase_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    counterexamples: list[dict[str, Any]] = []
    for target in targets:
        chosen_scale = choose_scale(edit_rows, target.target_id)
        localized = [r for r in edit_rows if r["target_id"] == target.target_id and r["method"] == "localized_addition"]
        random = [r for r in edit_rows if r["target_id"] == target.target_id and r["method"] == "random_direction_addition"]
        wrong = [r for r in edit_rows if r["target_id"] == target.target_id and r["method"] == "wrong_position_addition"]
        chosen = [r for r in localized if abs(float(r["scale"]) - chosen_scale) < 1e-9]
        chosen_row = chosen[0] if chosen else max(localized, key=lambda r: float(r["target_gain"]))
        rand_best = safe_max([r["target_gain"] for r in random])
        wrong_best = safe_max([r["target_gain"] for r in wrong])
        loc_best = max([r for r in localization_rows if r["target_id"] == target.target_id and r["method"] == "localized_patch"], key=lambda r: float(r["patch_gain"]))
        loc_controls = [
            r for r in localization_rows
            if r["target_id"] == target.target_id and r["depth"] == loc_best["depth"] and r["method"] != "localized_patch"
        ]
        loc_control_floor = safe_max([r["patch_gain"] for r in loc_controls])
        para = [r for r in paraphrase_rows if r["target_id"] == target.target_id and r["method"] == "localized_addition"]
        retain = [r for r in retain_rows if r["target_id"] == target.target_id and r["method"] == "localized_addition" and r["eval_family"] == "retain"]
        neighbor = [r for r in retain_rows if r["target_id"] == target.target_id and r["method"] == "localized_addition" and r["eval_family"] == "neighbor"]
        paraphrase_gain = safe_mean([r["transfer_gain"] for r in para])
        paraphrase_transfer_rate = safe_mean([1.0 if r["transferred_to_after"] else 0.0 for r in para], default=0.0)
        retain_damage = safe_mean([r["damage"] for r in retain], default=0.0)
        retain_preserve_rate = safe_mean([1.0 if r["preserved_sign"] else 0.0 for r in retain], default=0.0)
        neighbor_damage = safe_mean([r["damage"] for r in neighbor], default=0.0)
        neighbor_preserve_rate = safe_mean([1.0 if r["preserved_sign"] else 0.0 for r in neighbor], default=0.0)
        control_gap = float(chosen_row["target_gain"]) - max(rand_best, wrong_best)
        locality_gap = float(loc_best["patch_gain"]) - loc_control_floor if math.isfinite(loc_control_floor) else float("nan")
        baseline_prefers_before = target.base_after_margin < 0.0
        claim_ready = (
            baseline_prefers_before
            and float(chosen_row["target_gain"]) > 0.25
            and control_gap > 0.05
            and paraphrase_gain > 0.0
            and retain_damage < RETAIN_DAMAGE_SOFT_LIMIT
            and retain_preserve_rate >= 0.999
        )
        side_effect_caveat = neighbor_damage >= NEIGHBOR_DAMAGE_CAVEAT or neighbor_preserve_rate < 0.999
        if claim_ready and side_effect_caveat:
            posture = "localized_edit_supported_with_neighbor_caveat"
        elif claim_ready:
            posture = "localized_edit_supported"
        else:
            posture = "needs_refinement_or_control_limited"
        evidence.append({
            "target_id": target.target_id,
            "family": target.family,
            "best_depth": chosen_row["depth"],
            "chosen_scale": chosen_scale,
            "baseline_prefers_before": baseline_prefers_before,
            "base_after_minus_before": rounded(target.base_after_margin),
            "localized_after_minus_before": chosen_row["edited_after_minus_before"],
            "localized_target_gain": chosen_row["target_gain"],
            "best_random_gain": rounded(rand_best),
            "best_wrong_position_gain": rounded(wrong_best),
            "target_control_gap": rounded(control_gap),
            "localization_patch_gain": loc_best["patch_gain"],
            "localization_control_floor": rounded(loc_control_floor),
            "locality_gap": rounded(locality_gap),
            "mean_paraphrase_gain": rounded(paraphrase_gain),
            "paraphrase_transfer_rate": rounded(paraphrase_transfer_rate),
            "mean_retain_damage": rounded(retain_damage),
            "retain_preserve_rate": rounded(retain_preserve_rate),
            "mean_neighbor_damage": rounded(neighbor_damage),
            "neighbor_preserve_rate": rounded(neighbor_preserve_rate),
            "claim_posture": posture,
        })
        if not baseline_prefers_before:
            counterexamples.append({
                "target_id": target.target_id,
                "kind": "baseline_already_prefers_after",
                "localized_gain": chosen_row["target_gain"],
                "best_random_gain": rounded(rand_best),
                "best_wrong_position_gain": rounded(wrong_best),
                "lesson": "The smoke model already prefers the counterfactual token; this row cannot support a before-to-after edit claim.",
            })
        if control_gap <= 0.05:
            counterexamples.append({
                "target_id": target.target_id,
                "kind": "control_matches_or_beats_localized_edit",
                "localized_gain": chosen_row["target_gain"],
                "best_random_gain": rounded(rand_best),
                "best_wrong_position_gain": rounded(wrong_best),
                "lesson": "Localization is not enough here; a control edit matched the target movement.",
            })
        if retain_damage >= RETAIN_DAMAGE_SOFT_LIMIT or retain_preserve_rate < 0.999:
            counterexamples.append({
                "target_id": target.target_id,
                "kind": "retain_damage",
                "localized_gain": chosen_row["target_gain"],
                "best_random_gain": rounded(rand_best),
                "best_wrong_position_gain": rounded(wrong_best),
                "lesson": "The edit moved the target but damaged retained facts too much for a specificity claim.",
            })
        if side_effect_caveat:
            counterexamples.append({
                "target_id": target.target_id,
                "kind": "neighbor_side_effect_caveat",
                "localized_gain": chosen_row["target_gain"],
                "best_random_gain": rounded(rand_best),
                "best_wrong_position_gain": rounded(wrong_best),
                "lesson": "The target/retain result may survive, but nearby facts moved enough that the claim needs a neighbor caveat.",
            })
        if paraphrase_gain <= 0.0:
            counterexamples.append({
                "target_id": target.target_id,
                "kind": "paraphrase_failed_to_transfer",
                "localized_gain": chosen_row["target_gain"],
                "best_random_gain": rounded(rand_best),
                "best_wrong_position_gain": rounded(wrong_best),
                "lesson": "The edit appears prompt-local rather than robust across paraphrases.",
            })
    metrics = {
        "n_targets": len(targets),
        "n_localization_rows": len(localization_rows),
        "n_edit_rows": len(edit_rows),
        "n_retain_neighbor_rows": len(retain_rows),
        "n_paraphrase_rows": len(paraphrase_rows),
        "claim_ready_targets": sum(1 for r in evidence if str(r["claim_posture"]).startswith("localized_edit_supported")),
        "mean_localized_target_gain": rounded(safe_mean([r["localized_target_gain"] for r in evidence])),
        "mean_retain_damage": rounded(safe_mean([r["mean_retain_damage"] for r in evidence])),
        "localization_editability_corr": rounded(safe_corr([r["localization_patch_gain"] for r in evidence], [r["localized_target_gain"] for r in evidence])),
    }
    return evidence, counterexamples, metrics


def write_method_card(ctx: bench.RunContext, bundle: bench.ModelBundle, evidence: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Lab 28 method card",
        "",
        "This lab tests reversible activation edits. It does not write weights and does not claim factual erasure.",
        "",
        f"- model: `{bundle.anatomy.model_id}` ({bundle.anatomy.n_layers} blocks)",
        "- edit primitive: add a donor-minus-target residual direction at one localized stream site",
        "- localization primitive: donor residual patch at the final prompt token",
        "- controls: wrong position, deterministic random direction, and no-edit baseline",
        "- evidence rung: `CAUSAL + AUDIT`, scoped to benign logit measurements",
        "- forbidden claim: the fact was erased from the model",
        "",
        "| target | depth | scale | target gain | control gap | paraphrase gain | retain damage | posture |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in evidence:
        lines.append(
            f"| `{row['target_id']}` | {row['best_depth']} | {row['chosen_scale']} | "
            f"{row['localized_target_gain']} | {row['target_control_gap']} | {row['mean_paraphrase_gain']} | "
            f"{row['mean_retain_damage']} | {row['claim_posture']} |"
        )
    path = ctx.path("method_card.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Lab 28 method card and scoped edit verdict.")


def write_operationalization_audit(
    ctx: bench.RunContext,
    evidence: Sequence[Mapping[str, Any]],
    counterexamples: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Lab 28 operationalization audit",
        "",
        "Favorite interpretation under attack: mechanistic localization makes edits specific and robust.",
        "",
        "## What the measurement can say",
        "",
        "A localized residual addition increased the logit margin for a benign counterfactual target, and the audit checks whether that change transfers to paraphrases while retaining nearby facts.",
        "",
        "## What it cannot say",
        "",
        "It cannot say that a fact disappeared from weights, that the model has unlearned a topic, or that a stronger persistent edit would be safe.",
        "",
        "## Cheap explanations",
        "",
        "- A random direction or wrong-position direction moves the answer just as much.",
        "- The edit only affects the exact prompt string.",
        "- The edit damages unrelated retain facts.",
        "- The donor prompt supplies a broad answer-token bias rather than a localized mechanism.",
        "",
        "## Verdicts",
        "",
    ]
    for row in evidence:
        lines.append(f"- `{row['target_id']}`: `{row['claim_posture']}` with target-control gap `{row['target_control_gap']}`.")
    lines += ["", "## Counterexamples", ""]
    if counterexamples:
        for row in counterexamples[:14]:
            lines.append(f"- `{row['target_id']}` `{row['kind']}`: {row['lesson']}")
    else:
        lines.append("- No automatic counterexamples crossed the current thresholds. Replicate before broadening the claim.")
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Controls and non-claims for Lab 28 mechanistic editing.")


def write_run_summary(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    data_info: Mapping[str, Any],
    metrics: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Lab 28 run summary: mechanistic editing and unlearning",
        "",
        f"- model: `{bundle.anatomy.model_id}`",
        f"- data rows: {data_info['n_rows_selected']} selected from `{pathlib.Path(str(data_info['data_path'])).name}`",
        f"- families: `{data_info['families']}`",
        f"- science_ready: `{data_info['science_ready']}`",
        "- safety scope: benign associations, inference-time edits only, no persistent weight edits",
        "",
        "## Headline verdicts",
        "",
        "| target | depth | scale | target gain | paraphrase gain | retain damage | posture |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in evidence:
        lines.append(
            f"| `{row['target_id']}` | {row['best_depth']} | {row['chosen_scale']} | "
            f"{row['localized_target_gain']} | {row['mean_paraphrase_gain']} | "
            f"{row['mean_retain_damage']} | {row['claim_posture']} |"
        )
    lines += [
        "",
        "## Reading order",
        "",
        "1. `method_card.md` for the intervention and claim boundary.",
        "2. `tables/edit_evidence_matrix.csv` for target-level verdicts.",
        "3. `tables/editing_results.csv` for main target dose-response and controls.",
        "4. `tables/paraphrase_robustness.csv` and `tables/retain_forget_matrix.csv` before claiming specificity.",
        "5. `operationalization_audit.md` for counterexamples and non-claims.",
        "",
        "## Smallest surviving claim",
        "",
        "A supported row means an inference-time localized activation addition moved the measured logit margin more than controls on this audit set. It does not mean the model forgot the original fact.",
        "",
        f"Intervention rows: localization={metrics['n_localization_rows']}, edits={metrics['n_edit_rows']}, retain/neighbor={metrics['n_retain_neighbor_rows']}, paraphrase={metrics['n_paraphrase_rows']}.",
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines) + "\n")
    ctx.register_artifact(path, "summary", "Lab 28 run summary and reading order.")


def write_safety_status(ctx: bench.RunContext, targets: Sequence[EditTarget]) -> None:
    payload = {
        "lab": LAB_ID,
        "status": "safe_first_pass",
        "scope": "benign public associations and harmless sequence facts",
        "edit_mode": "inference_time_activation_addition_only",
        "persistent_weight_edits": False,
        "private_data": False,
        "harmful_capability_targets": False,
        "refusal_ablation": False,
        "generation_from_sensitive_prompts": False,
        "targets": {
            t.target_id: {
                "family": t.family,
                "edit_type": t.edit_type,
                "safety_notes": t.safety_notes,
            }
            for t in targets
        },
    }
    path = ctx.path("diagnostics", "safety_status.json")
    bench.write_json(path, payload)
    ctx.register_artifact(path, "diagnostic", "Safety status for Lab 28 editing/unlearning targets.")


def write_plot_guide(ctx: bench.RunContext) -> None:
    rows = [
        {"plot": "plots/editing_unlearning_dashboard.png", "read_for": "Target movement, paraphrase transfer, and retain damage.", "non_claim": "Does not show erasure."},
        {"plot": "plots/localization_vs_editability.png", "read_for": "Whether localization patch gains predict edit gains.", "non_claim": "Correlation is not a persistent edit guarantee."},
        {"plot": "plots/edit_method_frontier.png", "read_for": "Target gain versus retain damage by method.", "non_claim": "A frontier point still needs side-effect review."},
        {"plot": "plots/paraphrase_robustness_matrix.png", "read_for": "Paraphrase transfer by target.", "non_claim": "Small prompt sets are qualitative."},
        {"plot": "plots/neighbor_preservation_atlas.png", "read_for": "Neighbor and retain damage patterns.", "non_claim": "No claim about all nearby knowledge."},
        {"plot": "plots/mechanistic_locality_ladder.png", "read_for": "Localized patch gains versus controls.", "non_claim": "Patch localization is not uniqueness."},
        {"plot": "plots/unlearning_retain_forget_frontier.png", "read_for": "Forget/retain tradeoff at chosen scales.", "non_claim": "Forget means measured target-margin movement only."},
    ]
    path = ctx.path("plots", "plot_reading_guide.csv")
    bench.write_csv(path, rows)
    ctx.register_artifact(path, "table", "Plot reading guide for Lab 28.")


def write_plots(
    ctx: bench.RunContext,
    evidence: Sequence[Mapping[str, Any]],
    localization_rows: Sequence[Mapping[str, Any]],
    edit_rows: Sequence[Mapping[str, Any]],
    retain_rows: Sequence[Mapping[str, Any]],
    paraphrase_rows: Sequence[Mapping[str, Any]],
) -> None:
    write_plot_guide(ctx)
    if ctx.args.no_plots:
        return
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [r["target_id"].replace("edit_", "").replace("_", "\n") for r in evidence]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.suptitle("Lab 28 mechanistic editing dashboard", fontsize=14, fontweight="bold")
    axes[0, 0].bar(x - 0.22, [float(r["localized_target_gain"]) for r in evidence], 0.22, label="target gain", color="#0072B2")
    axes[0, 0].bar(x, [float(r["mean_paraphrase_gain"]) for r in evidence], 0.22, label="paraphrase gain", color="#009E73")
    axes[0, 0].bar(x + 0.22, [float(r["mean_retain_damage"]) for r in evidence], 0.22, label="retain damage", color="#D55E00")
    axes[0, 0].set_xticks(x, labels, fontsize=7)
    axes[0, 0].set_title("Change versus side effects")
    axes[0, 0].legend(frameon=False, fontsize=8)

    axes[0, 1].scatter(
        [float(r["localization_patch_gain"]) for r in evidence],
        [float(r["localized_target_gain"]) for r in evidence],
        c="#CC79A7",
    )
    for i, lab in enumerate(labels):
        axes[0, 1].annotate(str(i + 1), (float(evidence[i]["localization_patch_gain"]), float(evidence[i]["localized_target_gain"])), fontsize=8)
    axes[0, 1].set_xlabel("localization patch gain")
    axes[0, 1].set_ylabel("chosen edit gain")
    axes[0, 1].set_title("Localization vs editability")

    method_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in edit_rows:
        method_groups[str(row["method"])].append(row)
    for method, rows in method_groups.items():
        if method == "no_edit":
            continue
        axes[1, 0].scatter(
            [float(r["target_gain"]) for r in rows],
            [float(r["edited_after_minus_before"]) for r in rows],
            label=method.replace("_addition", ""),
            alpha=0.75,
        )
    axes[1, 0].axhline(0.0, color="#444444", linestyle="--", linewidth=1)
    axes[1, 0].set_xlabel("target gain")
    axes[1, 0].set_ylabel("after-before margin")
    axes[1, 0].set_title("Method dose frontier")
    axes[1, 0].legend(frameon=False, fontsize=8)

    postures = [1 if r["claim_posture"] == "localized_edit_supported" else 0 for r in evidence]
    axes[1, 1].bar(x, postures, color=["#009E73" if p else "#999999" for p in postures])
    axes[1, 1].set_xticks(x, labels, fontsize=7)
    axes[1, 1].set_yticks([0, 1], ["needs review", "supported"])
    axes[1, 1].set_title("Claim posture")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    bench.save_figure(ctx, fig, "editing_unlearning_dashboard.png", "Lab 28 editing/unlearning dashboard.")

    plot_specs = [
        ("localization_vs_editability.png", "Localization vs editability", [float(r["localization_patch_gain"]) for r in evidence], [float(r["localized_target_gain"]) for r in evidence], "localization patch gain", "edit gain"),
        ("edit_method_frontier.png", "Edit method frontier", [float(r["target_control_gap"]) for r in evidence], [float(r["mean_retain_damage"]) for r in evidence], "target control gap", "retain damage"),
        ("unlearning_retain_forget_frontier.png", "Retain-forget frontier", [float(r["localized_target_gain"]) for r in evidence], [float(r["mean_retain_damage"]) for r in evidence], "target gain", "retain damage"),
    ]
    for name, title, xs, ys, xlabel, ylabel in plot_specs:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.scatter(xs, ys, color="#0072B2")
        for i, lab in enumerate(labels):
            ax.annotate(str(i + 1), (xs[i], ys[i]), fontsize=8)
        ax.axhline(0.0, color="#444444", linestyle=":", linewidth=0.8)
        ax.axvline(0.0, color="#444444", linestyle=":", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        bench.save_figure(ctx, fig, name, title + " summary.")

    para_targets = sorted({r["target_id"] for r in paraphrase_rows})
    para_prompts = sorted({r["prompt"] for r in paraphrase_rows})
    mat = np.zeros((len(para_targets), max(1, len(para_prompts))))
    for i, tid in enumerate(para_targets):
        for j, prompt in enumerate(para_prompts):
            vals = [
                float(r["transfer_gain"]) for r in paraphrase_rows
                if r["target_id"] == tid and r["prompt"] == prompt and r["method"] == "localized_addition"
            ]
            mat[i, j] = safe_mean(vals, default=0.0)
    fig, ax = plt.subplots(figsize=(max(7, len(para_prompts) * 0.75), 4.5))
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm")
    ax.set_yticks(range(len(para_targets)), [t.replace("edit_", "") for t in para_targets], fontsize=7)
    ax.set_xticks(range(len(para_prompts)), [str(i + 1) for i in range(len(para_prompts))])
    ax.set_title("Paraphrase robustness matrix")
    ax.set_xlabel("paraphrase prompt index")
    fig.colorbar(im, ax=ax, shrink=0.8, label="transfer gain")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "paraphrase_robustness_matrix.png", "Paraphrase transfer heatmap.")

    atlas_targets = sorted({r["target_id"] for r in retain_rows})
    atlas_roles = sorted({f"{r['eval_family']}:{r['eval_role']}" for r in retain_rows})
    atlas = np.zeros((len(atlas_targets), max(1, len(atlas_roles))))
    for i, tid in enumerate(atlas_targets):
        for j, role in enumerate(atlas_roles):
            vals = [
                float(r["damage"]) for r in retain_rows
                if r["target_id"] == tid and f"{r['eval_family']}:{r['eval_role']}" == role and r["method"] == "localized_addition"
            ]
            atlas[i, j] = safe_mean(vals, default=0.0)
    fig, ax = plt.subplots(figsize=(max(8, len(atlas_roles) * 0.7), 4.5))
    im = ax.imshow(atlas, aspect="auto", cmap="magma_r")
    ax.set_yticks(range(len(atlas_targets)), [t.replace("edit_", "") for t in atlas_targets], fontsize=7)
    ax.set_xticks(range(len(atlas_roles)), [str(i + 1) for i in range(len(atlas_roles))])
    ax.set_title("Neighbor preservation atlas")
    ax.set_xlabel("retain/neighbor probe index")
    fig.colorbar(im, ax=ax, shrink=0.8, label="damage")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "neighbor_preservation_atlas.png", "Retain and neighbor damage heatmap.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.18, [float(r["localization_patch_gain"]) for r in evidence], 0.36, label="localized patch", color="#009E73")
    ax.bar(x + 0.18, [float(r["localization_control_floor"]) for r in evidence], 0.36, label="best control", color="#D55E00")
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_title("Mechanistic locality ladder")
    ax.set_ylabel("patch gain")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "mechanistic_locality_ladder.png", "Localized patch gain versus controls.")


def write_claims(ctx: bench.RunContext, evidence: Sequence[Mapping[str, Any]]) -> None:
    run_name = ctx.run_dir.name
    claims = []
    for i, row in enumerate(evidence, start=1):
        tag = "CAUSAL,AUDIT" if row["claim_posture"] == "localized_edit_supported" else "CAUSAL,AUDIT"
        claims.append({
            "id": f"{LAB_ID}-C{i}",
            "tag": tag,
            "text": (
                f"For `{row['target_id']}`, a reversible localized residual addition at depth {row['best_depth']} "
                f"and scale {row['chosen_scale']} changed the target after-minus-before margin by "
                f"{row['localized_target_gain']}, with target-control gap {row['target_control_gap']}, "
                f"mean paraphrase gain {row['mean_paraphrase_gain']}, and mean retain damage {row['mean_retain_damage']}. "
                f"Posture: {row['claim_posture']}."
            ),
            "artifact": f"runs/{run_name}/tables/edit_evidence_matrix.csv",
            "falsifier": "Random-direction or wrong-position controls match the edit, paraphrases fail, or retain/neighbor damage grows.",
        })
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    targets, data_info = load_targets(ctx)
    manifest_path = ctx.path("diagnostics", "data_manifest.json")
    bench.write_json(manifest_path, data_info)
    ctx.register_artifact(manifest_path, "diagnostic", "Lab 28 data manifest and science-ready status.")
    targets = tokenization_gate(ctx, bundle, targets)
    write_safety_status(ctx, targets)
    bench.run_hook_parity_check(ctx, bundle, targets[0].prompt)
    first = bench.run_with_residual_cache(bundle, targets[0].prompt)
    bench.run_lens_self_check(ctx, bundle, first)
    bench.run_patch_noop_check(ctx, bundle, targets[0].prompt)
    target_caps, donor_caps, baseline_rows = cache_baselines(ctx, bundle, targets)
    localization_rows, best_by_target = localize_sites(ctx, bundle, targets, target_caps, donor_caps)
    edit_rows, vectors, metadata = run_edits(ctx, bundle, targets, target_caps, donor_caps, best_by_target)
    retain_rows, paraphrase_rows = evaluate_side_sets(ctx, bundle, targets, edit_rows, vectors)
    evidence, counterexamples, metrics = summarize_evidence(targets, localization_rows, edit_rows, retain_rows, paraphrase_rows)
    evidence_path = ctx.path("tables", "edit_evidence_matrix.csv")
    bench.write_csv_with_context(ctx, evidence_path, evidence)
    ctx.register_artifact(evidence_path, "table", "Target-level Lab 28 evidence matrix.")
    counter_path = ctx.path("tables", "edit_counterexamples.csv")
    bench.write_csv_with_context(ctx, counter_path, counterexamples)
    ctx.register_artifact(counter_path, "table", "Counterexamples where controls, retain damage, or paraphrase failures limit the edit claim.")
    method_path = ctx.path("tables", "method_capability_audit.csv")
    bench.write_csv_with_context(ctx, method_path, [
        {"method": "localized_addition", "status": "run", "why": "safe reversible inference-time intervention"},
        {"method": "wrong_position_addition", "status": "run", "why": "specificity control"},
        {"method": "random_direction_addition", "status": "run", "why": "direction control"},
        {"method": "feature_clamp_suppression", "status": "not_run", "why": "requires aligned feature dictionary; future extension"},
        {"method": "persistent_rank_one_weight_edit", "status": "not_run", "why": "safe apply/restore plumbing intentionally deferred"},
    ])
    ctx.register_artifact(method_path, "table", "Method capability audit for planned and implemented edit mechanisms.")
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, {**metrics, "data": data_info})
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 28 metrics.")
    write_method_card(ctx, bundle, evidence)
    write_operationalization_audit(ctx, evidence, counterexamples)
    write_run_summary(ctx, bundle, data_info, metrics, evidence)
    write_claims(ctx, evidence)
    write_plots(ctx, evidence, localization_rows, edit_rows, retain_rows, paraphrase_rows)
    print(f"[lab28] wrote {len(evidence)} evidence rows and {len(counterexamples)} counterexamples")
