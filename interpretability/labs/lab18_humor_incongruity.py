"""Lab 18: Humor as incongruity, resolution, and cheap correlates.

This lab treats "humor" as a deliberately narrow operational object:
setup-dependent incongruity that resolves into a joke-shaped ending. It asks
whether a frozen model exposes a handle that separates joke completions from
literal, surprising, silly, and positive controls.

Evidence labels:
  * OBS for target surprisal, entropy, and attention-to-setup measurements;
  * DECODE for held-out joke-vs-control probe selectivity;
  * CAUSAL, narrowly, for activation-addition steering that changes
    joke-register scores more than surprise/silliness/positive/random controls.

The lab does not claim that the model experiences funniness. Its required
audit is precisely the gap between a joke-structure handle and anything richer.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import math
import re
import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

import interp_bench as bench

LAB_ID = "L18"
DATA_FILE = "humor_incongruity_pairs.csv"
CONDITIONS = ("joke", "literal", "surprise", "silly", "positive")
CONTROL_CONDITIONS = ("literal", "surprise", "silly", "positive")
AUDIT_DIRECTIONS = ("humor", "surprise", "silly", "positive")

PROMPT_SET_FAMILY_CAPS = {"small": 2, "medium": 3, "full": 0}
TRAIN_FRACTION = 0.65
MAX_NEW_TOKENS = 42
ENGINE_MAX_CONCURRENT = 16
MAX_STEERING_ITEMS = 10
MAX_ATTENTION_ITEMS = 10
STEERING_DOSE = 0.50

SYSTEM_PROMPT = (
    "You are a careful assistant. Analyze short text without adding personal "
    "experience claims. Keep responses concise."
)

GENERIC_JOKE_MARKERS = (
    "because|turns out|said|only|needed|wanted|pun|joke|joking|wordplay|"
    "forecast|signal|key|date|cells|rolls|verse|breakpoints|deduction"
)
GENERIC_SILLY_MARKERS = "tiny|soup|dance|triangle|hat|socks|spoon|glitter|midnight"
GENERIC_SURPRISE_MARKERS = "suddenly|unexpected|instead|future|hidden|weather|ticket|door|map"
GENERIC_POSITIVE_MARKERS = "good|great|friendly|happy|smiled|calm|helpful|relieved|hopeful|comforting"


@dataclasses.dataclass
class HumorItem:
    item_id: str
    family: str
    setup: str
    joke_completion: str
    literal_completion: str
    surprise_completion: str
    silly_completion: str
    positive_completion: str
    setup_anchor: str
    resolution_keyword: str
    joke_markers: str
    silly_markers: str
    surprise_markers: str
    positive_markers: str
    note: str


def stable_hash_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def rounded(x: Any, ndigits: int = 4) -> Any:
    try:
        if isinstance(x, (int, float)) and math.isfinite(float(x)):
            return round(float(x), ndigits)
    except Exception:
        pass
    return x


def none_if_nan(x: Any, ndigits: int = 4) -> Any:
    try:
        val = float(x)
    except Exception:
        return x
    if not math.isfinite(val):
        return None
    return round(val, ndigits)


def safe_fmean(vals: Sequence[float], default: float = float("nan")) -> float:
    finite = []
    for value in vals:
        try:
            f = float(value)
        except Exception:
            continue
        if math.isfinite(f):
            finite.append(f)
    return float(statistics.fmean(finite)) if finite else default


def auc_from_scores(pos: Sequence[float], neg: Sequence[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def unit(v: Any) -> Any:
    norm = v.norm().clamp_min(1e-9)
    if not bool(norm.isfinite()):
        raise RuntimeError("Direction norm was not finite.")
    return v / norm


def random_unit(d_model: int, seed: int) -> Any:
    import torch

    gen = torch.Generator().manual_seed(int(seed))
    return unit(torch.randn(d_model, generator=gen))


def cosine(a: Any, b: Any) -> float:
    denom = (a.norm() * b.norm()).clamp_min(1e-9)
    return float((a @ b) / denom)


def data_path(name: str) -> Any:
    path = bench.COURSE_ROOT / "data" / name
    if not path.exists():
        raise RuntimeError(f"Frozen dataset missing: {path}")
    return path


def render_chat(bundle: bench.ModelBundle, user_message: str, *, system: str | None = SYSTEM_PROMPT) -> str:
    return bench.apply_chat_template(
        bundle,
        user_message,
        system=system,
        add_generation_prompt=True,
    )


def completion_for(item: HumorItem, condition: str) -> str:
    key = f"{condition}_completion"
    return getattr(item, key)


def contrast_message(item: HumorItem, condition: str) -> str:
    return (
        "Read this setup and ending as a compact text-analysis example.\n"
        f"Setup: {item.setup}\n"
        f"Ending: {completion_for(item, condition)}\n"
        "Reply with exactly one word: noted."
    )


def setup_only_message(item: HumorItem) -> str:
    return (
        "Read this setup before any ending is supplied.\n"
        f"Setup: {item.setup}\n"
        "Reply with exactly one word: noted."
    )


def generation_message(item: HumorItem) -> str:
    return (
        "Write one short, original ending for this setup. Keep it under 18 words.\n"
        f"Setup: {item.setup}\n"
        "Ending:"
    )


def load_items(args: Any) -> tuple[list[HumorItem], dict[str, Any]]:
    path = data_path(DATA_FILE)
    raw: list[HumorItem] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw.append(HumorItem(**row))

    set_name = args.prompt_set
    if set_name not in PROMPT_SET_FAMILY_CAPS:
        raise ValueError("Lab 18 uses --prompt-set small|medium|full.")
    cap = PROMPT_SET_FAMILY_CAPS[set_name]
    if getattr(args, "max_examples", 0) and args.max_examples > 0:
        cap = int(args.max_examples)

    by_family: dict[str, list[HumorItem]] = defaultdict(list)
    for item in raw:
        by_family[item.family].append(item)

    selected: list[HumorItem] = []
    for family, rows in sorted(by_family.items()):
        ranked = sorted(rows, key=lambda r: stable_hash_int(f"{family}:{r.item_id}"))
        selected.extend(ranked[:cap] if cap > 0 else ranked)
    selected = sorted(selected, key=lambda r: (r.family, r.item_id))

    for family in sorted(by_family):
        n = sum(1 for r in selected if r.family == family)
        if n < 2:
            raise RuntimeError(f"Lab 18 needs at least two rows per family; {family} has {n}.")

    info = {
        "data_file": DATA_FILE,
        "data_sha256": bench.sha256_file(path),
        "prompt_set": set_name,
        "family_cap": cap,
        "n_rows": len(selected),
        "families": sorted(by_family),
        "counts_by_family": {
            family: sum(1 for row in selected if row.family == family)
            for family in sorted(by_family)
        },
        "conditions": CONDITIONS,
        "selection_rule": "deterministic per-family cap by stable hash; full keeps all rows",
    }
    return selected, info


def make_split(items: Sequence[HumorItem], seed: int) -> dict[str, bool]:
    split: dict[str, bool] = {}
    by_family: dict[str, list[HumorItem]] = defaultdict(list)
    for item in items:
        by_family[item.family].append(item)
    for family, rows in by_family.items():
        ranked = sorted(rows, key=lambda r: stable_hash_int(f"{seed}:{family}:{r.item_id}"))
        n_train = int(round(TRAIN_FRACTION * len(ranked)))
        if len(ranked) > 1:
            n_train = max(1, min(len(ranked) - 1, n_train))
        else:
            n_train = 1
        train_ids = {row.item_id for row in ranked[:n_train]}
        for row in rows:
            split[row.item_id] = row.item_id in train_ids
    return split


def split_rows(items: Sequence[HumorItem], split: Mapping[str, bool]) -> list[dict[str, Any]]:
    return [
        {
            "item_id": item.item_id,
            "family": item.family,
            "split": "train" if split[item.item_id] else "eval",
            "setup_excerpt": item.setup[:90],
        }
        for item in items
    ]


def cache_features(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    items: Sequence[HumorItem],
) -> tuple[Any, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    import torch

    rows = []
    stacked = []
    features: dict[str, dict[str, Any]] = {}
    phase_features: dict[str, dict[str, Any]] = {}
    report_every = max(1, len(items) // 4)
    for i, item in enumerate(items):
        features[item.item_id] = {}
        for condition in CONDITIONS:
            prompt = render_chat(bundle, contrast_message(item, condition))
            cap = bench.run_with_residual_cache(bundle, prompt, add_special_tokens=False)
            streams = cap.streams[:, -1, :]
            features[item.item_id][condition] = streams
            stacked.append(streams)
            rows.append({
                "item_id": item.item_id,
                "family": item.family,
                "condition": condition,
                "prompt_tokens": len(cap.input_ids),
            })
        setup_prompt = render_chat(bundle, setup_only_message(item))
        setup_cap = bench.run_with_residual_cache(bundle, setup_prompt, add_special_tokens=False)
        phase_features[item.item_id] = {
            "setup": setup_cap.streams[:, -1, :],
            "joke": features[item.item_id]["joke"],
        }
        stacked.append(setup_cap.streams[:, -1, :])
        rows.append({
            "item_id": item.item_id,
            "family": item.family,
            "condition": "setup_only",
            "prompt_tokens": len(setup_cap.input_ids),
        })
        if (i + 1) % report_every == 0:
            print(f"[lab18] cached humor/control features for {i + 1}/{len(items)} rows")

    path = ctx.path("diagnostics", "prompt_token_counts.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "diagnostic", "Rendered chat-template token counts for Lab 18 prompts.")
    return torch.stack(stacked), features, phase_features


def direction_for_depth(
    items: Sequence[HumorItem],
    features: Mapping[str, Mapping[str, Any]],
    split: Mapping[str, bool],
    depth: int,
    *,
    train: bool = True,
    sign_seed: int | None = None,
) -> Any | None:
    import torch

    diffs = []
    rows = [item for item in items if split[item.item_id] == train]
    if not rows:
        return None
    for item in rows:
        joke = features[item.item_id]["joke"][depth]
        control_mean = torch.stack([features[item.item_id][c][depth] for c in CONTROL_CONDITIONS]).mean(dim=0)
        diff = joke - control_mean
        if sign_seed is not None and stable_hash_int(f"{sign_seed}:{item.item_id}") % 2:
            diff = -diff
        diffs.append(diff)
    if not diffs:
        return None
    return unit(torch.stack(diffs).mean(dim=0))


def named_direction_for_depth(
    items: Sequence[HumorItem],
    features: Mapping[str, Mapping[str, Any]],
    split: Mapping[str, bool],
    depth: int,
    name: str,
) -> Any:
    import torch

    diffs = []
    for item in items:
        if not split[item.item_id]:
            continue
        literal = features[item.item_id]["literal"][depth]
        if name == "humor":
            joke = features[item.item_id]["joke"][depth]
            control_mean = torch.stack([features[item.item_id][c][depth] for c in CONTROL_CONDITIONS]).mean(dim=0)
            diffs.append(joke - control_mean)
        elif name == "surprise":
            diffs.append(features[item.item_id]["surprise"][depth] - literal)
        elif name == "silly":
            diffs.append(features[item.item_id]["silly"][depth] - literal)
        elif name == "positive":
            diffs.append(features[item.item_id]["positive"][depth] - literal)
        else:
            raise ValueError(name)
    if not diffs:
        raise RuntimeError(f"Could not build {name} direction.")
    return unit(torch.stack(diffs).mean(dim=0))


def projection_scores(
    rows: Sequence[HumorItem],
    features: Mapping[str, Mapping[str, Any]],
    direction: Any,
    depth: int,
) -> tuple[list[float], list[float]]:
    pos = [float(features[item.item_id]["joke"][depth] @ direction) for item in rows]
    neg = [
        float(features[item.item_id][condition][depth] @ direction)
        for item in rows
        for condition in CONTROL_CONDITIONS
    ]
    return pos, neg


def run_probe_sweep(
    items: Sequence[HumorItem],
    features: Mapping[str, Mapping[str, Any]],
    split: Mapping[str, bool],
    seed: int,
    d_model: int,
) -> tuple[list[dict[str, Any]], int]:
    n_depths = next(iter(next(iter(features.values())).values())).shape[0]
    eval_rows = [item for item in items if not split[item.item_id]]
    report: list[dict[str, Any]] = []
    for depth in range(1, n_depths):
        real = direction_for_depth(items, features, split, depth, train=True)
        shuffled = direction_for_depth(items, features, split, depth, train=True, sign_seed=seed + 1009 * depth)
        random = random_unit(d_model, seed + depth * 7919)
        if real is not None:
            train_pos, train_neg = projection_scores([item for item in items if split[item.item_id]], features, random, depth)
            if train_pos and train_neg and safe_fmean(train_pos) < safe_fmean(train_neg):
                random = -random
        for kind, direction in (("real", real), ("shuffled_sign", shuffled), ("random_oriented", random)):
            if direction is None:
                continue
            pos, neg = projection_scores(eval_rows, features, direction, depth)
            auc = auc_from_scores(pos, neg)
            report.append({
                "probe": "joke_vs_literal_surprise_silly_positive",
                "depth": depth,
                "direction_kind": kind,
                "auc": rounded(auc),
                "selectivity_vs_chance": rounded(auc - 0.5),
                "mean_joke_projection": rounded(safe_fmean(pos)),
                "mean_control_projection": rounded(safe_fmean(neg)),
                "n_eval_jokes": len(pos),
                "n_eval_controls": len(neg),
            })

    def score(depth: int) -> float:
        real = [
            float(r["auc"]) for r in report
            if r["depth"] == depth and r["direction_kind"] == "real" and isinstance(r.get("auc"), (int, float))
        ]
        shuffled = [
            float(r["auc"]) for r in report
            if r["depth"] == depth and r["direction_kind"] == "shuffled_sign" and isinstance(r.get("auc"), (int, float))
        ]
        random = [
            float(r["auc"]) for r in report
            if r["depth"] == depth and r["direction_kind"] == "random_oriented" and isinstance(r.get("auc"), (int, float))
        ]
        return safe_fmean(real, -1.0) - max(safe_fmean(shuffled, 0.5), safe_fmean(random, 0.5))

    best_depth = max(range(1, n_depths), key=score)
    return report, best_depth


def run_phase_probe(
    items: Sequence[HumorItem],
    phase_features: Mapping[str, Mapping[str, Any]],
    split: Mapping[str, bool],
    seed: int,
    d_model: int,
    depth: int,
) -> list[dict[str, Any]]:
    import torch

    train_rows = [item for item in items if split[item.item_id]]
    eval_rows = [item for item in items if not split[item.item_id]]
    diffs = [phase_features[item.item_id]["joke"][depth] - phase_features[item.item_id]["setup"][depth] for item in train_rows]
    if not diffs:
        return []
    real = unit(torch.stack(diffs).mean(dim=0))
    shuffled_diffs = [
        (-diff if stable_hash_int(f"{seed}:phase:{item.item_id}") % 2 else diff)
        for item, diff in zip(train_rows, diffs)
    ]
    shuffled = unit(torch.stack(shuffled_diffs).mean(dim=0))
    random = random_unit(d_model, seed + 4049)
    rows = []
    for kind, direction in (("real", real), ("shuffled_sign", shuffled), ("random", random)):
        pos = [float(phase_features[item.item_id]["joke"][depth] @ direction) for item in eval_rows]
        neg = [float(phase_features[item.item_id]["setup"][depth] @ direction) for item in eval_rows]
        auc = auc_from_scores(pos, neg)
        rows.append({
            "probe": "punchline_phase_joke_full_vs_setup_only",
            "depth": depth,
            "direction_kind": kind,
            "auc": rounded(auc),
            "selectivity_vs_chance": rounded(auc - 0.5),
            "mean_full_joke_projection": rounded(safe_fmean(pos)),
            "mean_setup_projection": rounded(safe_fmean(neg)),
            "n_eval_pairs": len(eval_rows),
        })
    return rows


def target_surprisal_bits(bundle: bench.ModelBundle, prompt: str, target: str) -> dict[str, Any]:
    import torch

    tokenizer = bundle.tokenizer
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(" " + target, add_special_tokens=False)["input_ids"]
    if not prompt_ids or not target_ids:
        return {"target_tokens": len(target_ids), "mean_surprisal_bits": float("nan"), "total_surprisal_bits": float("nan")}
    ids = prompt_ids + target_ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=bundle.input_device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = bundle.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    log_probs = torch.log_softmax(out.logits[0].float(), dim=-1)
    losses = []
    start = len(prompt_ids)
    for i, tok_id in enumerate(target_ids):
        pos = start + i
        if pos == 0:
            continue
        losses.append(float(-log_probs[pos - 1, tok_id] / math.log(2.0)))
    return {
        "target_tokens": len(target_ids),
        "mean_surprisal_bits": safe_fmean(losses),
        "total_surprisal_bits": sum(losses) if losses else float("nan"),
        "tokenization_note": "target encoded separately and appended to the rendered chat prompt",
    }


def next_token_entropy_bits(bundle: bench.ModelBundle, prompt: str) -> float:
    import torch

    logits = bench.next_token_logits(bundle, prompt)
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log2(probs.clamp_min(1e-45))
    return float(-(probs * log_probs).sum())


def run_surprisal_measurements(bundle: bench.ModelBundle, items: Sequence[HumorItem]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        setup_prompt = render_chat(
            bundle,
            "Read this setup and prepare for a short ending.\n"
            f"Setup: {item.setup}\n"
            "Ending:",
        )
        entropy = next_token_entropy_bits(bundle, setup_prompt)
        for condition in CONDITIONS:
            target_prompt = render_chat(
                bundle,
                "Complete this setup with the supplied ending.\n"
                f"Setup: {item.setup}\n"
                "Ending:",
            )
            stats = target_surprisal_bits(bundle, target_prompt, completion_for(item, condition))
            rows.append({
                "item_id": item.item_id,
                "family": item.family,
                "condition": condition,
                "setup_next_token_entropy_bits": rounded(entropy),
                "target_tokens": stats["target_tokens"],
                "mean_surprisal_bits": rounded(stats["mean_surprisal_bits"]),
                "total_surprisal_bits": rounded(stats["total_surprisal_bits"]),
                "tokenization_note": stats["tokenization_note"],
            })
    return rows


def keyword_patterns(spec: str) -> list[str]:
    return [p.strip().lower() for p in spec.split("|") if p.strip()]


def marker_count(text: str, spec: str) -> int:
    low = text.lower()
    count = 0
    for pat in keyword_patterns(spec):
        count += len(re.findall(rf"(?<![a-z0-9]){re.escape(pat)}(?![a-z0-9])", low))
    return count


def score_generation(item: HumorItem, text: str) -> dict[str, Any]:
    joke = marker_count(text, item.joke_markers) + marker_count(text, GENERIC_JOKE_MARKERS)
    silly = marker_count(text, item.silly_markers) + marker_count(text, GENERIC_SILLY_MARKERS)
    surprise = marker_count(text, item.surprise_markers) + marker_count(text, GENERIC_SURPRISE_MARKERS)
    positive = marker_count(text, item.positive_markers) + marker_count(text, GENERIC_POSITIVE_MARKERS)
    return {
        "joke_marker_count": joke,
        "silly_marker_count": silly,
        "surprise_marker_count": surprise,
        "positive_marker_count": positive,
        "joke_vs_cheap_margin": joke - max(silly, surprise, positive),
        "hand_label_funny": "",
        "hand_label_silly": "",
        "hand_label_surprising": "",
        "hand_label_positive": "",
        "hand_label_joke_shaped": "",
    }


def selected_eval_rows(items: Sequence[HumorItem], split: Mapping[str, bool]) -> list[HumorItem]:
    rows = [item for item in items if not split[item.item_id]]
    if not rows:
        rows = list(items)
    return rows[:MAX_STEERING_ITEMS]


def run_steering(
    bundle: bench.ModelBundle,
    items: Sequence[HumorItem],
    directions: Mapping[str, Any],
    depth: int,
    d_model: int,
    seed: int,
    ref_norm: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    injection_layer = max(0, depth - 1)
    scale = STEERING_DOSE * ref_norm
    prompts = [render_chat(bundle, generation_message(item)) for item in items]
    baseline_outs = bench.generate_continuous(
        bundle,
        prompts,
        MAX_NEW_TOKENS,
        max_concurrent=ENGINE_MAX_CONCURRENT,
        progress_label="lab18 steering baseline",
    )
    rows: list[dict[str, Any]] = []
    for item, text in zip(items, baseline_outs):
        rows.append({
            "item_id": item.item_id,
            "family": item.family,
            "steering_condition": "baseline",
            "steering_scale": 0.0,
            "generation": text,
            **score_generation(item, text),
        })

    random = random_unit(d_model, seed + 8803)
    conditions = [
        ("humor_direction", directions["humor"], scale),
        ("opposite_humor_direction", directions["humor"], -scale),
        ("surprise_direction", directions["surprise"], scale),
        ("silly_direction", directions["silly"], scale),
        ("positive_direction", directions["positive"], scale),
        ("random_direction", random, scale),
    ]
    for condition, vec, abs_scale in conditions:
        outs = bench.generate_continuous(
            bundle,
            prompts,
            MAX_NEW_TOKENS,
            max_concurrent=ENGINE_MAX_CONCURRENT,
            progress_label=f"lab18 steering {condition}",
            steer=(injection_layer, vec, abs_scale),
        )
        for item, text in zip(items, outs):
            rows.append({
                "item_id": item.item_id,
                "family": item.family,
                "steering_condition": condition,
                "steering_scale": rounded(abs_scale),
                "generation": text,
                **score_generation(item, text),
            })

    baseline = [row for row in rows if row["steering_condition"] == "baseline"]
    base_metrics = {
        key: safe_fmean([float(row[key]) for row in baseline])
        for key in ("joke_marker_count", "silly_marker_count", "surprise_marker_count", "positive_marker_count", "joke_vs_cheap_margin")
    }
    effect_rows: list[dict[str, Any]] = []
    for condition in sorted({row["steering_condition"] for row in rows}):
        sub = [row for row in rows if row["steering_condition"] == condition]
        out = {"steering_condition": condition, "n": len(sub)}
        for key in base_metrics:
            mean_val = safe_fmean([float(row[key]) for row in sub])
            out[f"mean_{key}"] = rounded(mean_val)
            out[f"{key}_delta_vs_baseline"] = rounded(mean_val - base_metrics[key])
        effect_rows.append(out)
    return rows, effect_rows


def token_span_indices(bundle: bench.ModelBundle, text: str, substring: str) -> tuple[list[int], str]:
    start = text.find(substring)
    if start < 0:
        return [], "substring_not_found"
    end = start + len(substring)
    try:
        enc = bundle.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc.get("offset_mapping") or []
        ids = enc.get("input_ids") or []
        if len(offsets) != len(ids):
            return [], "offset_length_mismatch"
        idxs = [
            i for i, (a, b) in enumerate(offsets)
            if b > start and a < end
        ]
        return idxs, "offset_mapping"
    except Exception as exc:
        return [], f"offset_mapping_failed:{type(exc).__name__}"


def attention_to_setup_rows(bundle: bench.ModelBundle, items: Sequence[HumorItem]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items[:MAX_ATTENTION_ITEMS]:
        for condition in ("joke", "literal", "surprise"):
            message = contrast_message(item, condition)
            prompt = render_chat(bundle, message)
            att = bench.run_with_attention_cache(bundle, prompt, all_positions=False, add_special_tokens=False)
            setup_idxs, setup_method = token_span_indices(bundle, prompt, item.setup)
            ending = completion_for(item, condition)
            ending_idxs, ending_method = token_span_indices(bundle, prompt, ending)
            if not setup_idxs or not ending_idxs:
                rows.append({
                    "item_id": item.item_id,
                    "family": item.family,
                    "condition": condition,
                    "layer": "",
                    "mean_attention_to_setup": "",
                    "max_head_attention_to_setup": "",
                    "n_setup_tokens": len(setup_idxs),
                    "query_token_index": "",
                    "query_token_text": "",
                    "span_method": f"setup={setup_method};ending={ending_method}",
                    "note": "span lookup failed; rerun with a fast tokenizer or inspect prompt text",
                })
                continue
            query_idx = max(ending_idxs)
            token_text = att.capture.tokens_text[query_idx] if query_idx < len(att.capture.tokens_text) else ""
            for layer in range(att.attentions.shape[0]):
                head_scores = att.attentions[layer, :, query_idx, setup_idxs].sum(dim=-1)
                rows.append({
                    "item_id": item.item_id,
                    "family": item.family,
                    "condition": condition,
                    "layer": layer,
                    "mean_attention_to_setup": rounded(float(head_scores.mean())),
                    "max_head_attention_to_setup": rounded(float(head_scores.max())),
                    "n_setup_tokens": len(setup_idxs),
                    "query_token_index": query_idx,
                    "query_token_text": token_text,
                    "span_method": f"setup={setup_method};ending={ending_method}",
                    "note": "attention from final completion token back to setup span",
                })
    return rows


def direction_cosine_rows(directions: Mapping[str, Any]) -> list[dict[str, Any]]:
    names = sorted(directions)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rows.append({
                "direction_a": a,
                "direction_b": b,
                "cosine": rounded(cosine(directions[a], directions[b])),
            })
    return rows


def metric_at(rows: Sequence[Mapping[str, Any]], kind: str, depth: int, key: str = "auc") -> float:
    vals = [
        float(row[key]) for row in rows
        if row.get("direction_kind") == kind
        and row.get("depth") == depth
        and isinstance(row.get(key), (int, float))
    ]
    return safe_fmean(vals)


def effect_delta(rows: Sequence[Mapping[str, Any]], condition: str, key: str) -> float:
    vals = [
        float(row[key]) for row in rows
        if row.get("steering_condition") == condition and isinstance(row.get(key), (int, float))
    ]
    return safe_fmean(vals)


def plot_surprisal(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(8.8, 5.1))
    xs = list(range(len(CONDITIONS)))
    means = [
        safe_fmean([
            float(row["mean_surprisal_bits"]) for row in rows
            if row.get("condition") == condition and isinstance(row.get("mean_surprisal_bits"), (int, float))
        ])
        for condition in CONDITIONS
    ]
    ax.bar(xs, means, color=["#2f6f8f", "#7f7f7f", "#b55a30", "#8c6bb1", "#4c9a62"])
    ax.set_xticks(xs)
    ax.set_xticklabels(CONDITIONS, rotation=20, ha="right")
    ax.set_ylabel("mean target-token surprisal (bits)")
    ax.set_title("Ending surprisal by condition")
    bench.style_ax(ax, legend=False)
    bench.save_figure(ctx, fig, "humor_surprisal_trajectories.png", "Mean target-token surprisal for joke and control endings.")


def plot_probe(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(8.8, 5.1))
    kinds = ("real", "shuffled_sign", "random_oriented")
    colors = {"real": "#2f6f8f", "shuffled_sign": "#b55a30", "random_oriented": "#7f7f7f"}
    for kind in kinds:
        sub = [row for row in rows if row.get("direction_kind") == kind]
        if not sub:
            continue
        xs = [int(row["depth"]) for row in sub]
        ys = [float(row["auc"]) for row in sub]
        ax.plot(xs, ys, marker="o", color=colors[kind], label=kind)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1.1)
    ax.set_xlabel("residual stream depth")
    ax.set_ylabel("held-out AUC")
    ax.set_title("Joke-vs-control probe by depth")
    bench.style_ax(ax)
    bench.save_figure(ctx, fig, "joke_probe_by_layer.png", "Held-out joke-vs-control probe AUC with controls.")


def write_operationalization_audit(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    lines = [
        "# Lab 18 Operationalization Audit",
        "",
        "## What Was Measured",
        "",
        "The lab measures a joke-structure handle: residual-stream differences between joke endings and matched literal, surprising, silly, and positive endings for the same setup.",
        "",
        "It does not measure subjective funniness, enjoyment, social uptake, or a human-like sense of humor.",
        "",
        "## Cheap Explanations",
        "",
        "- Surprise: the dataset includes surprising-but-not-funny endings, and `direction_cosines.csv` compares humor and surprise directions.",
        "- Silliness: silly-not-joke endings get their own direction and steering condition.",
        "- Positive sentiment: positive-not-joke endings get their own direction and steering condition.",
        "- Joke-register markers: generation scoring is a scaffold; hand-label columns must be filled before any writeup leans on steering.",
        "- Setup dependence: `attention_to_setup.csv` asks whether the completion token routes back to setup tokens, but attention is descriptive, not causal proof.",
        "- Probe leakage: train/eval split is by item within each family, and shuffled/random controls are reported beside the real direction.",
        "",
        "## Current Run",
        "",
        f"- Best depth: {metrics.get('best_depth')}",
        f"- Real joke-vs-control AUC at best depth: {metrics.get('real_auc_best_depth')}",
        f"- Shuffled/random AUC at best depth: {metrics.get('shuffled_auc_best_depth')} / {metrics.get('random_auc_best_depth')}",
        f"- Mean joke surprisal bits: {metrics.get('mean_joke_surprisal_bits')}",
        f"- Mean literal-control surprisal bits: {metrics.get('mean_literal_surprisal_bits')}",
        f"- Humor/surprise direction cosine: {metrics.get('humor_surprise_cosine')}",
        f"- Humor-direction joke-margin steering delta: {metrics.get('humor_steering_joke_margin_delta')}",
        f"- Surprise-direction joke-margin steering delta: {metrics.get('surprise_steering_joke_margin_delta')}",
        "",
        "## Allowed Claim",
        "",
        "A Lab 18 claim is allowed only as a handle claim: this model exposes a direction that separates and may steer joke-shaped endings under these controls. If it collapses into surprise, silliness, positivity, or generic joke-register markers, that is the result.",
        "",
    ]
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "audit", "Operationalization limits and cheap-explanation audit for Lab 18.")


def write_run_summary(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    lines = [
        "# Lab 18 Run Summary: Humor as Incongruity",
        "",
        f"- Model: `{metrics.get('model_id')}`",
        f"- Rows: {metrics.get('n_rows')}",
        f"- Best depth: {metrics.get('best_depth')}",
        f"- Joke-vs-control AUC: {metrics.get('real_auc_best_depth')}",
        f"- Shuffled/random AUC: {metrics.get('shuffled_auc_best_depth')} / {metrics.get('random_auc_best_depth')}",
        f"- Mean joke surprisal bits: {metrics.get('mean_joke_surprisal_bits')}",
        f"- Humor/surprise direction cosine: {metrics.get('humor_surprise_cosine')}",
        f"- Humor-direction joke-margin steering delta: {metrics.get('humor_steering_joke_margin_delta')}",
        "",
        "Read `operationalization_audit.md` before translating any result into a claim about humor.",
        "",
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Human-readable summary of headline Lab 18 metrics.")


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    import torch

    args = ctx.args
    if not bench.supports_chat_template(bundle):
        raise RuntimeError("Lab 18 requires an instruct model with a chat template.")

    items, data_info = load_items(args)
    print(f"[lab18] {data_info['n_rows']} rows; prompt_set={args.prompt_set}")
    manifest_path = ctx.path("diagnostics", "frozen_data_manifest.json")
    bench.write_json(manifest_path, data_info)
    ctx.register_artifact(manifest_path, "diagnostic", "Frozen Lab 18 data hash, filters, and counts.")

    first_prompt = render_chat(bundle, contrast_message(items[0], "joke"))
    bench.run_hook_parity_check(ctx, bundle, first_prompt)
    bench.run_lens_self_check(ctx, bundle, bench.run_with_residual_cache(bundle, first_prompt, add_special_tokens=False))

    split = make_split(items, args.seed)
    split_path = ctx.path("diagnostics", "split_audit.csv")
    bench.write_csv_with_context(ctx, split_path, split_rows(items, split))
    ctx.register_artifact(split_path, "diagnostic", "Family-stratified train/eval split for Lab 18.")

    feat_tensor, features, phase_features = cache_features(ctx, bundle, items)
    row_norms = feat_tensor.norm(dim=-1)
    norm_rows = [
        {
            "depth": depth,
            "mean_norm": rounded(safe_fmean(row_norms[:, depth].tolist())),
            "min_norm": rounded(float(row_norms[:, depth].min())),
            "max_norm": rounded(float(row_norms[:, depth].max())),
        }
        for depth in range(row_norms.shape[1])
    ]
    norm_path = ctx.path("diagnostics", "activation_norms_by_depth.csv")
    bench.write_csv_with_context(ctx, norm_path, norm_rows)
    ctx.register_artifact(norm_path, "diagnostic", "Humor/control prompt residual norm audit.")

    probe_rows, best_depth = run_probe_sweep(items, features, split, args.seed, bundle.anatomy.d_model)
    phase_rows = run_phase_probe(items, phase_features, split, args.seed, bundle.anatomy.d_model, best_depth)
    probe_path = ctx.path("tables", "joke_probe_by_layer.csv")
    bench.write_csv_with_context(ctx, probe_path, probe_rows)
    ctx.register_artifact(probe_path, "table", "Joke-vs-control held-out probe sweep with shuffled/random controls.")
    phase_path = ctx.path("tables", "punchline_phase_probe.csv")
    bench.write_csv_with_context(ctx, phase_path, phase_rows)
    ctx.register_artifact(phase_path, "table", "Setup-only versus full joke punchline-phase probe at the selected depth.")
    results_path = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results_path, probe_rows)
    ctx.register_artifact(results_path, "results", "Alias of joke_probe_by_layer.csv for the standard run contract.")
    print(f"[lab18] selected humor depth {best_depth}")

    surprisal_rows = run_surprisal_measurements(bundle, items)
    surprisal_path = ctx.path("tables", "humor_surprisal_trajectories.csv")
    bench.write_csv_with_context(ctx, surprisal_path, surprisal_rows)
    ctx.register_artifact(surprisal_path, "table", "Teacher-forced target surprisal and setup entropy for joke/control endings.")

    directions = {
        name: named_direction_for_depth(items, features, split, best_depth, name)
        for name in AUDIT_DIRECTIONS
    }
    cos_rows = direction_cosine_rows(directions)
    cos_path = ctx.path("tables", "direction_cosines.csv")
    bench.write_csv_with_context(ctx, cos_path, cos_rows)
    ctx.register_artifact(cos_path, "table", "Pairwise cosines among humor, surprise, silliness, and positivity directions.")

    attn_rows = attention_to_setup_rows(bundle, selected_eval_rows(items, split))
    attn_path = ctx.path("tables", "attention_to_setup.csv")
    bench.write_csv_with_context(ctx, attn_path, attn_rows)
    ctx.register_artifact(attn_path, "table", "Attention from completion token to setup span for joke/literal/surprise prompts.")

    ref_norm = safe_fmean(row_norms[:, best_depth].tolist(), default=1.0)
    eval_items = selected_eval_rows(items, split)
    steering_generations, steering_effects = run_steering(
        bundle,
        eval_items,
        directions,
        best_depth,
        bundle.anatomy.d_model,
        args.seed,
        ref_norm,
    )
    generation_path = ctx.path("tables", "humor_steering_generations.csv")
    bench.write_csv_with_context(ctx, generation_path, steering_generations)
    ctx.register_artifact(generation_path, "table", "Baseline and steered endings with marker and hand-label scaffold.")
    effects_path = ctx.path("tables", "humor_direction_audit.csv")
    bench.write_csv_with_context(ctx, effects_path, steering_effects)
    ctx.register_artifact(effects_path, "table", "Humor steering effect compared with surprise, silly, positive, and random controls.")

    state_common = {
        "depth": best_depth,
        "depth_convention": "bench streams[k]: 0 = embeddings, k = residual after block k",
        "read_site": "chat-templated final prompt token before assistant generation",
        "model_id": bundle.anatomy.model_id,
        "d_model": bundle.anatomy.d_model,
        "n_layers": bundle.anatomy.n_layers,
        "method": "train-split mass-mean directions over matched joke/control endings",
    }
    state_path = ctx.path("state", "humor_directions.pt")
    torch.save({**state_common, "directions": directions}, state_path)
    ctx.register_artifact(state_path, "tensor", "Humor, surprise, silly, and positive directions.")
    humor_path = ctx.path("state", "humor_direction.pt")
    torch.save({**state_common, "direction": directions["humor"]}, humor_path)
    ctx.register_artifact(humor_path, "tensor", "Selected humor/joke-structure direction.")
    meta_path = ctx.path("state", "humor_direction_metadata.json")
    bench.write_json(meta_path, {**state_common, "directions": sorted(directions)})
    ctx.register_artifact(meta_path, "state", "Human-readable metadata for Lab 18 saved directions.")

    if not args.no_plots:
        plot_surprisal(ctx, surprisal_rows)
        plot_probe(ctx, probe_rows)

    real_auc = metric_at(probe_rows, "real", best_depth)
    shuffled_auc = metric_at(probe_rows, "shuffled_sign", best_depth)
    random_auc = metric_at(probe_rows, "random_oriented", best_depth)
    joke_surprisal = safe_fmean([
        float(row["mean_surprisal_bits"]) for row in surprisal_rows
        if row.get("condition") == "joke" and isinstance(row.get("mean_surprisal_bits"), (int, float))
    ])
    literal_surprisal = safe_fmean([
        float(row["mean_surprisal_bits"]) for row in surprisal_rows
        if row.get("condition") == "literal" and isinstance(row.get("mean_surprisal_bits"), (int, float))
    ])
    surprise_surprisal = safe_fmean([
        float(row["mean_surprisal_bits"]) for row in surprisal_rows
        if row.get("condition") == "surprise" and isinstance(row.get("mean_surprisal_bits"), (int, float))
    ])
    humor_surprise_cos = cosine(directions["humor"], directions["surprise"])
    metrics = {
        "model_id": bundle.anatomy.model_id,
        "n_rows": len(items),
        "n_eval_rows": len(eval_items),
        "best_depth": best_depth,
        "injection_layer": max(0, best_depth - 1),
        "real_auc_best_depth": none_if_nan(real_auc),
        "shuffled_auc_best_depth": none_if_nan(shuffled_auc),
        "random_auc_best_depth": none_if_nan(random_auc),
        "real_selectivity_vs_shuffled": none_if_nan(real_auc - shuffled_auc),
        "mean_joke_surprisal_bits": none_if_nan(joke_surprisal),
        "mean_literal_surprisal_bits": none_if_nan(literal_surprisal),
        "mean_surprise_surprisal_bits": none_if_nan(surprise_surprisal),
        "joke_surprisal_minus_literal": none_if_nan(joke_surprisal - literal_surprisal),
        "humor_surprise_cosine": none_if_nan(humor_surprise_cos),
        "humor_silly_cosine": none_if_nan(cosine(directions["humor"], directions["silly"])),
        "humor_positive_cosine": none_if_nan(cosine(directions["humor"], directions["positive"])),
        "humor_steering_joke_margin_delta": none_if_nan(effect_delta(
            steering_effects, "humor_direction", "joke_vs_cheap_margin_delta_vs_baseline"
        )),
        "surprise_steering_joke_margin_delta": none_if_nan(effect_delta(
            steering_effects, "surprise_direction", "joke_vs_cheap_margin_delta_vs_baseline"
        )),
        "random_steering_joke_margin_delta": none_if_nan(effect_delta(
            steering_effects, "random_direction", "joke_vs_cheap_margin_delta_vs_baseline"
        )),
        "steering_dose_fraction": STEERING_DOSE,
        "data": data_info,
    }
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 18 metrics.")

    write_operationalization_audit(ctx, metrics)
    write_run_summary(ctx, metrics)

    run_name = ctx.run_dir.name
    claims = [
        {
            "id": f"{LAB_ID}-C1",
            "tag": "DECODE",
            "text": (
                f"At depth {best_depth}, the joke-vs-control direction separates held-out "
                f"joke endings from literal/surprise/silly/positive controls with AUC "
                f"{metrics['real_auc_best_depth']} versus shuffled/random "
                f"{metrics['shuffled_auc_best_depth']} / {metrics['random_auc_best_depth']}. "
                "This is a joke-structure handle claim, not a claim about subjective funniness."
            ),
            "artifact": f"runs/{run_name}/tables/joke_probe_by_layer.csv",
            "falsifier": (
                "Shuffled or random controls match the AUC, or direction cosines show the handle "
                "is just surprise, silliness, or positivity."
            ),
        },
        {
            "id": f"{LAB_ID}-C2",
            "tag": "CAUSAL",
            "text": (
                f"Humor-direction steering changed the joke-vs-cheap marker margin by "
                f"{metrics['humor_steering_joke_margin_delta']} versus surprise-direction "
                f"{metrics['surprise_steering_joke_margin_delta']} and random-direction "
                f"{metrics['random_steering_joke_margin_delta']}. Hand labels are required "
                "before treating this as more than marker movement."
            ),
            "artifact": f"runs/{run_name}/tables/humor_direction_audit.csv",
            "falsifier": (
                "Surprise/silly/positive/random steering matches the effect, or hand labels show "
                "the rubric confuses joke shape with generic weirdness or sentiment."
            ),
        },
    ]
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)
