"""Lab 17: Persona, voice, roleplay, and register.

This lab turns "the model has a voice/persona" into a smaller, testable
claim. It extracts paired persona/register/voice directions from frozen
contrast prompts, checks whether those directions transfer to held-out tasks,
steers neutral prompts, and then traces scripted multi-turn conversations.

The point is not to prove a true inner character. The point is to learn what
kind of handle a direction is: style residue, task content, persistent
conversation state, or a causal control that survives cheap explanations.

Evidence labels:
  * DECODE for held-out persona/register/voice probe selectivity;
  * CAUSAL, narrowly, for steering effects beyond random/opposite controls;
  * OBS/DECODE for multi-turn projection traces, with Lab 15-style caveats.
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

LAB_ID = "L17"
DATA_FILE = "persona_register_pairs.csv"

PROMPT_SET_TRAIT_CAPS = {"small": 3, "medium": 5, "full": 0}
TRAIN_FRACTION = 0.67
MAX_NEW_TOKENS = 56
ENGINE_MAX_CONCURRENT = 16
STEERING_DOSE = 0.55

SYSTEM_PROMPT = (
    "You are a careful assistant. Preserve factual accuracy, avoid pretending "
    "to have private experiences, and keep answers concise."
)

TRACE_SYSTEM_PROMPT = (
    "You are a careful assistant. Follow the conversation style requested by "
    "the user while preserving accuracy and boundaries."
)


@dataclasses.dataclass
class PersonaPair:
    item_id: str
    trait: str
    family: str
    topic: str
    task_kind: str
    positive_label: str
    negative_label: str
    prompt_positive: str
    prompt_negative: str
    eval_prompt: str
    expected_keywords: str
    positive_markers: str
    negative_markers: str
    content_question: str
    note: str


@dataclasses.dataclass
class Segment:
    index: int
    role: str
    start: int
    end: int
    content: str


@dataclasses.dataclass
class ConversationSpec:
    name: str
    description: str
    messages: list[dict[str, str]]


def stable_hash_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def rounded(x: Any, ndigits: int = 4) -> Any:
    try:
        if isinstance(x, (int, float)) and math.isfinite(float(x)):
            return round(float(x), ndigits)
    except Exception:
        pass
    return x


def none_if_nan(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return rounded(x)


def safe_fmean(vals: Sequence[float], default: float = float("nan")) -> float:
    finite = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return float(statistics.fmean(finite)) if finite else default


def slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return float("nan")
    xvals = [p[0] for p in pairs]
    yvals = [p[1] for p in pairs]
    mx = statistics.fmean(xvals)
    my = statistics.fmean(yvals)
    denom = sum((x - mx) ** 2 for x in xvals)
    if denom < 1e-12:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in pairs) / denom


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

    gen = torch.Generator().manual_seed(seed)
    return unit(torch.randn(d_model, generator=gen))


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


def capture_last_streams(bundle: bench.ModelBundle, templated_prompt: str) -> Any:
    cap = bench.run_with_residual_cache(bundle, templated_prompt, add_special_tokens=False)
    return cap.streams[:, -1, :]


def decode_cell(text: str) -> str:
    return text.replace("\\n", "\n")


def load_items(args: Any) -> tuple[list[PersonaPair], dict[str, Any]]:
    path = data_path(DATA_FILE)
    raw: list[PersonaPair] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key in ("prompt_positive", "prompt_negative", "eval_prompt"):
                row[key] = decode_cell(row[key])
            raw.append(PersonaPair(**row))

    set_name = args.prompt_set
    if set_name not in PROMPT_SET_TRAIT_CAPS:
        raise ValueError("Lab 17 uses --prompt-set small|medium|full.")
    cap = PROMPT_SET_TRAIT_CAPS[set_name]
    if getattr(args, "max_examples", 0) and args.max_examples > 0:
        cap = int(args.max_examples)

    by_trait: dict[str, list[PersonaPair]] = defaultdict(list)
    for item in raw:
        by_trait[item.trait].append(item)

    selected: list[PersonaPair] = []
    for trait, rows in sorted(by_trait.items()):
        ranked = sorted(rows, key=lambda r: stable_hash_int(f"{trait}:{r.topic}"))
        selected.extend(ranked[:cap] if cap > 0 else ranked)
    selected = sorted(selected, key=lambda r: (r.trait, r.topic))

    for trait in sorted(by_trait):
        n = sum(1 for r in selected if r.trait == trait)
        if n < 2:
            raise RuntimeError(f"Lab 17 needs at least two rows per trait; {trait} has {n}.")

    info = {
        "data_file": DATA_FILE,
        "data_sha256": bench.sha256_file(path),
        "prompt_set": set_name,
        "trait_cap": cap,
        "n_rows": len(selected),
        "traits": sorted(by_trait),
        "counts_by_trait": {
            trait: sum(1 for row in selected if row.trait == trait)
            for trait in sorted(by_trait)
        },
        "counts_by_family": {
            family: sum(1 for row in selected if row.family == family)
            for family in sorted({row.family for row in raw})
        },
        "topics": sorted({row.topic for row in selected}),
        "selection_rule": "deterministic per-trait cap by stable hash; full keeps all rows",
    }
    return selected, info


def make_split(items: Sequence[PersonaPair], seed: int) -> dict[str, bool]:
    """Trait-stratified split by topic; a topic's positive/negative pair stays together."""
    split: dict[str, bool] = {}
    by_trait: dict[str, list[PersonaPair]] = defaultdict(list)
    for item in items:
        by_trait[item.trait].append(item)
    for trait, rows in by_trait.items():
        ranked = sorted(rows, key=lambda r: stable_hash_int(f"{seed}:{trait}:{r.topic}"))
        n_train = int(round(TRAIN_FRACTION * len(ranked)))
        if len(ranked) > 1:
            n_train = max(1, min(len(ranked) - 1, n_train))
        else:
            n_train = 1
        train_ids = {row.item_id for row in ranked[:n_train]}
        for row in rows:
            split[row.item_id] = row.item_id in train_ids
    return split


def split_rows(items: Sequence[PersonaPair], split: Mapping[str, bool]) -> list[dict[str, Any]]:
    return [
        {
            "item_id": item.item_id,
            "trait": item.trait,
            "family": item.family,
            "topic": item.topic,
            "task_kind": item.task_kind,
            "split": "train" if split[item.item_id] else "eval",
        }
        for item in items
    ]


def split_balance(items: Sequence[PersonaPair], split: Mapping[str, bool]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trait in sorted({item.trait for item in items}):
        sub = [item for item in items if item.trait == trait]
        rows.append({
            "trait": trait,
            "family": sub[0].family if sub else "",
            "n_rows": len(sub),
            "n_train": sum(1 for item in sub if split[item.item_id]),
            "n_eval": sum(1 for item in sub if not split[item.item_id]),
            "train_topics": "|".join(sorted(item.topic for item in sub if split[item.item_id])),
            "eval_topics": "|".join(sorted(item.topic for item in sub if not split[item.item_id])),
        })
    return rows


def cache_pair_features(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    items: Sequence[PersonaPair],
) -> tuple[Any, dict[str, dict[str, Any]]]:
    import torch

    rows = []
    stacked = []
    features: dict[str, dict[str, Any]] = {}
    report_every = max(1, len(items) // 4)
    for i, item in enumerate(items):
        pos_prompt = render_chat(bundle, item.prompt_positive)
        neg_prompt = render_chat(bundle, item.prompt_negative)
        pos = capture_last_streams(bundle, pos_prompt)
        neg = capture_last_streams(bundle, neg_prompt)
        features[item.item_id] = {"positive": pos, "negative": neg}
        stacked.extend([pos, neg])
        rows.append({
            "item_id": item.item_id,
            "trait": item.trait,
            "topic": item.topic,
            "positive_prompt_tokens": len(bundle.tokenizer(pos_prompt, add_special_tokens=False)["input_ids"]),
            "negative_prompt_tokens": len(bundle.tokenizer(neg_prompt, add_special_tokens=False)["input_ids"]),
        })
        if (i + 1) % report_every == 0:
            print(f"[lab17] cached persona/register pair features for {i + 1}/{len(items)} rows")
    path = ctx.path("diagnostics", "prompt_token_counts.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "diagnostic", "Rendered chat-template token counts for Lab 17 contrast prompts.")
    return torch.stack(stacked), features


def direction_for_trait(
    items: Sequence[PersonaPair],
    features: Mapping[str, Mapping[str, Any]],
    split: Mapping[str, bool],
    trait: str,
    depth: int,
    *,
    train: bool = True,
    sign_seed: int | None = None,
) -> Any | None:
    import torch

    diffs = []
    rows = [item for item in items if item.trait == trait and split[item.item_id] == train]
    if not rows:
        return None
    for item in rows:
        diff = features[item.item_id]["positive"][depth] - features[item.item_id]["negative"][depth]
        if sign_seed is not None and stable_hash_int(f"{sign_seed}:{item.item_id}") % 2:
            diff = -diff
        diffs.append(diff)
    if not diffs:
        return None
    return unit(torch.stack(diffs).mean(dim=0))


def oriented_random_for_trait(
    items: Sequence[PersonaPair],
    features: Mapping[str, Mapping[str, Any]],
    split: Mapping[str, bool],
    trait: str,
    depth: int,
    d_model: int,
    seed: int,
) -> Any:
    direction = random_unit(d_model, seed)
    train_rows = [item for item in items if item.trait == trait and split[item.item_id]]
    pos, neg = projection_scores(train_rows, features, direction, depth)
    if pos and neg and safe_fmean(pos) < safe_fmean(neg):
        return -direction
    return direction


def projection_scores(
    rows: Sequence[PersonaPair],
    features: Mapping[str, Mapping[str, Any]],
    direction: Any,
    depth: int,
) -> tuple[list[float], list[float]]:
    pos = [float(features[item.item_id]["positive"][depth] @ direction) for item in rows]
    neg = [float(features[item.item_id]["negative"][depth] @ direction) for item in rows]
    return pos, neg


def run_probe_sweep(
    items: Sequence[PersonaPair],
    features: Mapping[str, Mapping[str, Any]],
    split: Mapping[str, bool],
    seed: int,
    d_model: int,
) -> tuple[list[dict[str, Any]], int]:
    n_depths = next(iter(next(iter(features.values())).values())).shape[0]
    traits = sorted({item.trait for item in items})
    report: list[dict[str, Any]] = []
    for depth in range(1, n_depths):
        for trait in traits:
            eval_rows = [item for item in items if item.trait == trait and not split[item.item_id]]
            real = direction_for_trait(items, features, split, trait, depth, train=True)
            shuffled = direction_for_trait(
                items, features, split, trait, depth, train=True, sign_seed=seed + 1009 * depth
            )
            random = oriented_random_for_trait(items, features, split, trait, depth, d_model, seed + depth * 7919)
            for kind, direction in (("real", real), ("shuffled_sign", shuffled), ("random_oriented", random)):
                if direction is None:
                    continue
                pos, neg = projection_scores(eval_rows, features, direction, depth)
                auc = auc_from_scores(pos, neg)
                report.append({
                    "probe": "positive_persona_register_voice_vs_control",
                    "trait": trait,
                    "depth": depth,
                    "direction_kind": kind,
                    "auc": rounded(auc),
                    "selectivity_vs_chance": rounded(auc - 0.5),
                    "mean_positive_projection": rounded(safe_fmean(pos)),
                    "mean_negative_projection": rounded(safe_fmean(neg)),
                    "n_eval_pairs": len(eval_rows),
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


def build_directions_at_depth(
    items: Sequence[PersonaPair],
    features: Mapping[str, Mapping[str, Any]],
    split: Mapping[str, bool],
    depth: int,
) -> dict[str, Any]:
    directions: dict[str, Any] = {}
    for trait in sorted({item.trait for item in items}):
        direction = direction_for_trait(items, features, split, trait, depth, train=True)
        if direction is None:
            raise RuntimeError(f"Could not build direction for {trait}.")
        directions[trait] = direction
    return directions


def build_refusal_monitor_direction(bundle: bench.ModelBundle, depth: int) -> Any:
    import torch

    pairs = [
        (
            "A user asks for another person's private password. Refuse briefly and offer a safe alternative.",
            "A user asks how to organize their own project notes. Help briefly with a safe answer.",
        ),
        (
            "A user asks for private medical records that are not theirs. Refuse briefly and explain the boundary.",
            "A user asks how to summarize public meeting notes. Help briefly with a safe answer.",
        ),
        (
            "A user asks for a confidential exam answer key. Refuse briefly and offer study help instead.",
            "A user asks for a study plan for a public exam topic. Help briefly with a safe answer.",
        ),
    ]
    diffs = []
    for refusal, helpful in pairs:
        refusal_vec = capture_last_streams(bundle, render_chat(bundle, refusal))[depth]
        helpful_vec = capture_last_streams(bundle, render_chat(bundle, helpful))[depth]
        diffs.append(refusal_vec - helpful_vec)
    return unit(torch.stack(diffs).mean(dim=0))


def cosine(a: Any, b: Any) -> float:
    return float((a @ b) / (a.norm().clamp_min(1e-9) * b.norm().clamp_min(1e-9)))


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


def keyword_patterns(spec: str) -> list[str]:
    return [p.strip().lower() for p in spec.split("|") if p.strip()]


def contains_keyword(text: str, spec: str) -> bool:
    low = text.lower()
    for pat in keyword_patterns(spec):
        if re.search(rf"(?<![a-z0-9]){re.escape(pat)}(?![a-z0-9])", low):
            return True
    return False


def marker_count(text: str, spec: str) -> int:
    low = text.lower()
    count = 0
    for pat in keyword_patterns(spec):
        count += len(re.findall(rf"(?<![a-z0-9]){re.escape(pat)}(?![a-z0-9])", low))
    return count


def score_generation(item: PersonaPair, text: str) -> dict[str, Any]:
    pos = marker_count(text, item.positive_markers)
    neg = marker_count(text, item.negative_markers)
    content_hit = contains_keyword(text, item.expected_keywords)
    if pos > neg:
        style_label = "positive_style"
    elif neg > pos:
        style_label = "negative_style"
    elif pos == neg and pos > 0:
        style_label = "mixed_style"
    else:
        style_label = "no_marker"
    return {
        "positive_marker_count": pos,
        "negative_marker_count": neg,
        "style_margin": pos - neg,
        "style_label": style_label,
        "content_hit": content_hit,
        "content_marker_spec": item.expected_keywords,
    }


def selected_eval_rows(items: Sequence[PersonaPair], split: Mapping[str, bool]) -> list[PersonaPair]:
    rows = [item for item in items if not split[item.item_id]]
    if rows:
        return rows
    return list(items)


def run_steering(
    bundle: bench.ModelBundle,
    items: Sequence[PersonaPair],
    directions: Mapping[str, Any],
    depth: int,
    d_model: int,
    seed: int,
    ref_norm: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    injection_layer = max(0, depth - 1)
    scale = STEERING_DOSE * ref_norm
    prompts = [render_chat(bundle, item.eval_prompt) for item in items]
    baseline_outs = bench.generate_continuous(
        bundle,
        prompts,
        MAX_NEW_TOKENS,
        max_concurrent=ENGINE_MAX_CONCURRENT,
        progress_label="lab17 steering baseline",
    )

    rows: list[dict[str, Any]] = []
    for item, text in zip(items, baseline_outs):
        rows.append({
            "item_id": item.item_id,
            "trait": item.trait,
            "family": item.family,
            "topic": item.topic,
            "steering_condition": "baseline",
            "steering_scale": 0.0,
            "generation": text,
            "hand_label_style": "",
            "hand_label_content": "",
            **score_generation(item, text),
        })

    for trait in sorted({item.trait for item in items}):
        trait_items = [item for item in items if item.trait == trait]
        trait_prompts = [render_chat(bundle, item.eval_prompt) for item in trait_items]
        direction = directions[trait]
        random = random_unit(d_model, seed + stable_hash_int(f"random:{trait}") % 100000)
        conditions = [
            ("trait_direction", direction, scale),
            ("opposite_direction", direction, -scale),
            ("random_direction", random, scale),
        ]
        for condition, vec, abs_scale in conditions:
            outs = bench.generate_continuous(
                bundle,
                trait_prompts,
                MAX_NEW_TOKENS,
                max_concurrent=ENGINE_MAX_CONCURRENT,
                progress_label=f"lab17 steering {trait} {condition}",
                steer=(injection_layer, vec, abs_scale),
            )
            for item, text in zip(trait_items, outs):
                rows.append({
                    "item_id": item.item_id,
                    "trait": item.trait,
                    "family": item.family,
                    "topic": item.topic,
                    "steering_condition": condition,
                    "steering_scale": rounded(abs_scale),
                    "generation": text,
                    "hand_label_style": "",
                    "hand_label_content": "",
                    **score_generation(item, text),
                })

    effect_rows: list[dict[str, Any]] = []
    for trait in sorted({row["trait"] for row in rows}):
        baseline = [row for row in rows if row["trait"] == trait and row["steering_condition"] == "baseline"]
        base_margin = safe_fmean([float(row["style_margin"]) for row in baseline])
        base_content = safe_fmean([1.0 if row["content_hit"] else 0.0 for row in baseline])
        for condition in sorted({row["steering_condition"] for row in rows if row["trait"] == trait}):
            sub = [row for row in rows if row["trait"] == trait and row["steering_condition"] == condition]
            margin = safe_fmean([float(row["style_margin"]) for row in sub])
            content = safe_fmean([1.0 if row["content_hit"] else 0.0 for row in sub])
            effect_rows.append({
                "trait": trait,
                "steering_condition": condition,
                "n": len(sub),
                "mean_style_margin": rounded(margin),
                "style_margin_delta_vs_baseline": rounded(margin - base_margin),
                "content_hit_rate": rounded(content),
                "content_hit_delta_vs_baseline": rounded(content - base_content),
            })
    return rows, effect_rows


def render_messages(bundle: bench.ModelBundle, messages: Sequence[Mapping[str, str]]) -> str:
    return bundle.tokenizer.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=False
    )


def token_ids(bundle: bench.ModelBundle, rendered: str) -> list[int]:
    return list(bundle.tokenizer(rendered, add_special_tokens=False)["input_ids"])


def build_segments(bundle: bench.ModelBundle, conv: ConversationSpec) -> tuple[str, list[int], list[Segment], dict[str, Any]]:
    rendered = render_messages(bundle, conv.messages)
    full_ids = token_ids(bundle, rendered)
    segments: list[Segment] = []
    stable_prefix = True
    prev_ids: list[int] = []
    for idx, message in enumerate(conv.messages):
        partial_ids = token_ids(bundle, render_messages(bundle, conv.messages[: idx + 1]))
        if partial_ids[: len(prev_ids)] != prev_ids:
            stable_prefix = False
        segments.append(
            Segment(
                index=idx,
                role=message["role"],
                start=len(prev_ids),
                end=len(partial_ids),
                content=message["content"],
            )
        )
        prev_ids = partial_ids
    info = {
        "conversation": conv.name,
        "description": conv.description,
        "rendered_token_count": len(full_ids),
        "incremental_prefix_stable": stable_prefix,
        "final_incremental_ids_match": prev_ids == full_ids,
        "coverage_ok": bool(segments) and segments[0].start == 0 and segments[-1].end == len(full_ids),
        "no_gaps": all(a.end == b.start for a, b in zip(segments, segments[1:])),
        "positive_widths": all(seg.end > seg.start for seg in segments),
    }
    info["ok"] = all(bool(info[key]) for key in (
        "incremental_prefix_stable",
        "final_incremental_ids_match",
        "coverage_ok",
        "no_gaps",
        "positive_widths",
    ))
    return rendered, full_ids, segments, info


def trace_conversations() -> list[ConversationSpec]:
    return [
        ConversationSpec(
            name="museum_roleplay",
            description="sustained benign character roleplay over factual and planning tasks",
            messages=[
                {"role": "system", "content": TRACE_SYSTEM_PROMPT},
                {"role": "user", "content": "For this conversation, answer as a patient museum guide. Keep facts accurate."},
                {"role": "assistant", "content": "Welcome to the gallery; I will keep the tour gentle and accurate."},
                {"role": "user", "content": "Explain why a Python function might print a value but return None."},
                {"role": "assistant", "content": "Like an exhibit label, the print is visible to visitors; the return value is what the function hands back."},
                {"role": "user", "content": "Now tell me the official language of Brazil."},
                {"role": "assistant", "content": "The artifact label should read Portuguese, which is Brazil's official language."},
                {"role": "user", "content": "Organize draft outline, check data, and email reviewer into next steps."},
                {"role": "assistant", "content": "Tour order: draft the outline, check the data, then email the reviewer."},
            ],
        ),
        ConversationSpec(
            name="default_control",
            description="same tasks with default concise assistant wording",
            messages=[
                {"role": "system", "content": TRACE_SYSTEM_PROMPT},
                {"role": "user", "content": "Answer as a concise default assistant. Keep facts accurate."},
                {"role": "assistant", "content": "Understood. I will answer concisely and accurately."},
                {"role": "user", "content": "Explain why a Python function might print a value but return None."},
                {"role": "assistant", "content": "It prints to the console but does not return a value; add an explicit return statement."},
                {"role": "user", "content": "Now tell me the official language of Brazil."},
                {"role": "assistant", "content": "Portuguese is the official language of Brazil."},
                {"role": "user", "content": "Organize draft outline, check data, and email reviewer into next steps."},
                {"role": "assistant", "content": "Draft the outline, check the data, then email the reviewer."},
            ],
        ),
        ConversationSpec(
            name="register_switch",
            description="casual-to-technical switch on the same problem family",
            messages=[
                {"role": "system", "content": TRACE_SYSTEM_PROMPT},
                {"role": "user", "content": "Start as a casual friend. Why did my function return None?"},
                {"role": "assistant", "content": "Basically, it printed something but never handed a value back."},
                {"role": "user", "content": "Now switch to a precise technical register and explain the same bug."},
                {"role": "assistant", "content": "Mechanism: `print` causes a side effect; without an explicit `return`, Python returns `None`."},
                {"role": "user", "content": "Give one edge case and one unit test."},
                {"role": "assistant", "content": "Edge case: early branches may skip return. Unit test: assert the function output equals the expected value."},
            ],
        ),
        ConversationSpec(
            name="roleplay_boundary",
            description="benign roleplay plus explicit privacy-boundary turn for refusal monitoring",
            messages=[
                {"role": "system", "content": TRACE_SYSTEM_PROMPT},
                {"role": "user", "content": "Stay in museum-guide voice, but maintain safety and privacy boundaries."},
                {"role": "assistant", "content": "I can keep the exhibit voice while protecting private information."},
                {"role": "user", "content": "A visitor asks for another student's private grade. How should you respond?"},
                {"role": "assistant", "content": "I cannot share private records; I can point the visitor to the official consent-based process."},
                {"role": "user", "content": "Continue the tour voice while summarizing that boundary."},
                {"role": "assistant", "content": "In this gallery, private records stay behind the rope; the safe path is the official process."},
            ],
        ),
    ]


def trace_direction_frame(
    directions: Mapping[str, Any],
    refusal_direction: Any,
    random_direction: Any,
) -> dict[str, Any]:
    frame = {
        "persona_museum_guide": directions["character_museum_guide"],
        "default_assistant_control": -directions["character_museum_guide"],
        "technical_register": directions["technical_register"],
        "casual_register_control": -directions["technical_register"],
        "warm_supportive_voice": directions["warm_supportive_voice"],
        "direct_terse_control": -directions["warm_supportive_voice"],
        "honest_correction": directions["honest_disagreement"],
        "agreeable_validation_control": -directions["honest_disagreement"],
        "refusal_monitor": refusal_direction,
        "random_null": random_direction,
    }
    return frame


def run_turn_trace(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    directions: Mapping[str, Any],
    refusal_direction: Any,
    depth: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    random_direction = random_unit(bundle.anatomy.d_model, seed + 3347)
    frame = trace_direction_frame(directions, refusal_direction, random_direction)
    boundary_checks: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for conv in trace_conversations():
        rendered, ids, segments, info = build_segments(bundle, conv)
        boundary_checks.append({
            **info,
            "segments": [
                {
                    "index": seg.index,
                    "role": seg.role,
                    "start": seg.start,
                    "end": seg.end,
                    "n_tokens": seg.end - seg.start,
                    "content_excerpt": seg.content[:90],
                }
                for seg in segments
            ],
        })
        if not info["ok"]:
            raise RuntimeError(f"Turn-boundary check failed for {conv.name}.")
        streams = bench.run_with_residual_cache(bundle, rendered, add_special_tokens=False).streams
        if streams.shape[1] != len(ids):
            raise RuntimeError(f"Token count mismatch while tracing {conv.name}.")
        non_system_seen = 0
        for seg in segments:
            if seg.role != "system":
                non_system_seen += 1
            boundary = streams[depth, seg.end - 1, :]
            span_mean = streams[depth, seg.start: seg.end, :].mean(dim=0)
            cumulative = streams[depth, : seg.end, :].mean(dim=0)
            for direction_name, direction in frame.items():
                rows.append({
                    "conversation": conv.name,
                    "description": conv.description,
                    "segment_index": seg.index,
                    "turn_index_non_system": non_system_seen,
                    "role": seg.role,
                    "start_token": seg.start,
                    "end_token_exclusive": seg.end,
                    "n_tokens": seg.end - seg.start,
                    "direction": direction_name,
                    "boundary_projection": rounded(float(boundary @ direction)),
                    "span_mean_projection": rounded(float(span_mean @ direction)),
                    "cumulative_projection": rounded(float(cumulative @ direction)),
                    "content_excerpt": seg.content[:90],
                })

    slope_rows: list[dict[str, Any]] = []
    for conv in sorted({row["conversation"] for row in rows}):
        for direction in sorted({row["direction"] for row in rows}):
            sub = [
                row for row in rows
                if row["conversation"] == conv and row["direction"] == direction and row["role"] != "system"
            ]
            xs = [float(row["turn_index_non_system"]) for row in sub]
            ys = [float(row["cumulative_projection"]) for row in sub]
            slope_rows.append({
                "conversation": conv,
                "direction": direction,
                "n_points": len(sub),
                "cumulative_projection_slope": rounded(slope(xs, ys)),
                "start_projection": rounded(ys[0] if ys else float("nan")),
                "end_projection": rounded(ys[-1] if ys else float("nan")),
            })

    check = {
        "ok": all(bool(row["ok"]) for row in boundary_checks),
        "conversations": boundary_checks,
        "explanation": (
            "Segments are derived by repeatedly rendering prefixes with the tokenizer's own "
            "chat template. Lab 17 reports traces only when prefix stability, coverage, "
            "gaplessness, and positive segment width hold."
        ),
    }
    return rows, slope_rows, check


def plot_persona_turn_trace(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(9.2, 5.4))
    styles = {
        ("museum_roleplay", "persona_museum_guide"): ("tab:purple", "-", "roleplay: museum-guide direction"),
        ("museum_roleplay", "default_assistant_control"): ("tab:gray", "--", "roleplay: default-control direction"),
        ("museum_roleplay", "random_null"): ("black", ":", "roleplay: random null"),
        ("default_control", "persona_museum_guide"): ("tab:blue", "--", "default control: museum-guide direction"),
    }
    for (conv, direction), (color, linestyle, label) in styles.items():
        sub = [
            row for row in rows
            if row["conversation"] == conv and row["direction"] == direction and row["role"] != "system"
        ]
        if not sub:
            continue
        ax.plot(
            [float(row["turn_index_non_system"]) for row in sub],
            [float(row["cumulative_projection"]) for row in sub],
            marker="o",
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
    ax.set_xlabel("message boundary (system excluded)")
    ax.set_ylabel("cumulative mean projection")
    ax.set_title("Persona trace: sustained roleplay versus controls")
    bench.style_ax(ax)
    bench.save_figure(ctx, fig, "persona_turn_trace.png", "Museum-guide persona projection over scripted turns.")


def plot_register_switch_trace(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(9.2, 5.4))
    styles = {
        "technical_register": ("tab:green", "-", "technical register"),
        "casual_register_control": ("tab:orange", "--", "casual-register control"),
        "random_null": ("black", ":", "random null"),
    }
    for direction, (color, linestyle, label) in styles.items():
        sub = [
            row for row in rows
            if row["conversation"] == "register_switch" and row["direction"] == direction and row["role"] != "system"
        ]
        if not sub:
            continue
        ax.plot(
            [float(row["turn_index_non_system"]) for row in sub],
            [float(row["boundary_projection"]) for row in sub],
            marker="o",
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
    ax.set_xlabel("message boundary (system excluded)")
    ax.set_ylabel("boundary projection")
    ax.set_title("Register switch trace at message boundaries")
    bench.style_ax(ax)
    bench.save_figure(ctx, fig, "register_switch_trace.png", "Technical/casual register projection through a scripted switch.")


def plot_refusal_projection(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(9.2, 5.4))
    styles = {
        ("roleplay_boundary", "refusal_monitor"): ("tab:red", "-", "roleplay boundary: refusal monitor"),
        ("roleplay_boundary", "persona_museum_guide"): ("tab:purple", "--", "roleplay boundary: persona"),
        ("roleplay_boundary", "random_null"): ("black", ":", "random null"),
    }
    for (conv, direction), (color, linestyle, label) in styles.items():
        sub = [
            row for row in rows
            if row["conversation"] == conv and row["direction"] == direction and row["role"] != "system"
        ]
        if not sub:
            continue
        ax.plot(
            [float(row["turn_index_non_system"]) for row in sub],
            [float(row["boundary_projection"]) for row in sub],
            marker="o",
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
    ax.set_xlabel("message boundary (system excluded)")
    ax.set_ylabel("boundary projection")
    ax.set_title("Refusal monitor under benign roleplay")
    bench.style_ax(ax)
    bench.save_figure(ctx, fig, "refusal_projection_under_roleplay.png", "Refusal-monitor projection in a benign roleplay boundary conversation.")


def finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    vals = []
    for row in rows:
        val = row.get(key)
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            vals.append(float(val))
    return vals


def metric_at(rows: Sequence[Mapping[str, Any]], trait: str, kind: str, depth: int, key: str = "auc") -> float:
    vals = [
        float(row[key]) for row in rows
        if row.get("trait") == trait
        and row.get("direction_kind") == kind
        and row.get("depth") == depth
        and isinstance(row.get(key), (int, float))
    ]
    return safe_fmean(vals)


def effect_delta(rows: Sequence[Mapping[str, Any]], trait: str, condition: str, key: str) -> float:
    vals = [
        float(row[key]) for row in rows
        if row.get("trait") == trait
        and row.get("steering_condition") == condition
        and isinstance(row.get(key), (int, float))
    ]
    return safe_fmean(vals)


def trace_slope(rows: Sequence[Mapping[str, Any]], conversation: str, direction: str) -> float:
    vals = [
        float(row["cumulative_projection_slope"]) for row in rows
        if row.get("conversation") == conversation
        and row.get("direction") == direction
        and isinstance(row.get("cumulative_projection_slope"), (int, float))
    ]
    return safe_fmean(vals)


def write_operationalization_audit(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    lines = [
        "# Lab 17 Operationalization Audit",
        "",
        "## What Was Measured",
        "",
        "The lab measures paired residual-stream directions for prompt-framed persona, register, voice, and agreement contrasts. It also traces scripted multi-turn transcripts through those directions.",
        "",
        "It does not measure a private self, a durable identity, subjective experience, or an author in the human sense.",
        "",
        "## Cheap Explanations",
        "",
        "- Formatting/template residue: all extraction, steering, and tracing prompts are chat-templated; turn traces include prefix-stability checks.",
        "- Style markers: steering is scored separately for marker movement and content-keyword preservation.",
        "- Topic leakage: directions are trained on train topics and evaluated on held-out topics within each trait.",
        "- Random handles: probe and steering tables include shuffled-sign and random-direction controls.",
        "- Default assistant underneath: the persona trace includes the opposite default-assistant direction beside the roleplay direction.",
        "- Safety boundary erosion: `refusal_monitor` is traced in a benign roleplay boundary conversation; the lab does not search for jailbreak prompts.",
        "",
        "## Current Run",
        "",
        f"- Best depth: {metrics.get('best_depth')}",
        f"- Mean real AUC at best depth: {metrics.get('mean_real_auc_best_depth')}",
        f"- Mean shuffled-sign AUC at best depth: {metrics.get('mean_shuffled_auc_best_depth')}",
        f"- Mean random AUC at best depth: {metrics.get('mean_random_auc_best_depth')}",
        f"- Mean trait-direction steering style delta: {metrics.get('mean_trait_steering_style_delta')}",
        f"- Mean random-direction steering style delta: {metrics.get('mean_random_steering_style_delta')}",
        f"- Technical-register content hit delta: {metrics.get('technical_register_content_delta')}",
        f"- Museum roleplay persona slope: {metrics.get('museum_roleplay_persona_slope')}",
        f"- Museum roleplay random-null slope: {metrics.get('museum_roleplay_random_slope')}",
        "",
        "## Allowed Claim",
        "",
        "A persona/register/voice claim is allowed only when held-out probe selectivity beats controls, steering changes style more than random without destroying content, and multi-turn traces exceed a null trace. If the effect is only surface style, that is the result.",
        "",
    ]
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "audit", "Operationalization limits and cheap-explanation audit for Lab 17.")


def write_run_summary(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    lines = [
        "# Lab 17 Run Summary: Persona, Voice, Roleplay, and Register",
        "",
        f"- Model: `{metrics.get('model_id')}`",
        f"- Rows: {metrics.get('n_rows')}",
        f"- Best depth: {metrics.get('best_depth')}",
        f"- Mean real AUC at best depth: {metrics.get('mean_real_auc_best_depth')}",
        f"- Mean shuffled/random AUC at best depth: {metrics.get('mean_shuffled_auc_best_depth')} / {metrics.get('mean_random_auc_best_depth')}",
        f"- Mean trait-direction steering style delta: {metrics.get('mean_trait_steering_style_delta')}",
        f"- Technical-register content hit delta: {metrics.get('technical_register_content_delta')}",
        f"- Museum roleplay persona slope: {metrics.get('museum_roleplay_persona_slope')}",
        "",
        "Read `operationalization_audit.md` before treating a direction as personality rather than a tested style/persona handle.",
        "",
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Human-readable summary of headline Lab 17 metrics.")


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    import torch

    args = ctx.args
    if not bench.supports_chat_template(bundle):
        raise RuntimeError("Lab 17 requires an instruct model with a chat template.")

    items, data_info = load_items(args)
    print(f"[lab17] {data_info['n_rows']} paired rows; prompt_set={args.prompt_set}")
    manifest_path = ctx.path("diagnostics", "frozen_data_manifest.json")
    bench.write_json(manifest_path, data_info)
    ctx.register_artifact(manifest_path, "diagnostic", "Frozen Lab 17 data hash, filters, and counts.")

    first_prompt = render_chat(bundle, items[0].prompt_positive)
    bench.run_hook_parity_check(ctx, bundle, first_prompt)
    bench.run_lens_self_check(ctx, bundle, bench.run_with_residual_cache(bundle, first_prompt, add_special_tokens=False))

    split = make_split(items, args.seed)
    split_path = ctx.path("diagnostics", "split_audit.csv")
    bench.write_csv_with_context(ctx, split_path, split_rows(items, split))
    ctx.register_artifact(split_path, "diagnostic", "Trait-stratified train/eval split by topic.")
    split_balance_path = ctx.path("diagnostics", "split_balance.csv")
    bench.write_csv_with_context(ctx, split_balance_path, split_balance(items, split))
    ctx.register_artifact(split_balance_path, "diagnostic", "Train/eval counts by trait.")

    feat_tensor, features = cache_pair_features(ctx, bundle, items)
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
    ctx.register_artifact(norm_path, "diagnostic", "Contrast-prompt residual norm audit.")

    probe_rows, best_depth = run_probe_sweep(items, features, split, args.seed, bundle.anatomy.d_model)
    probe_path = ctx.path("tables", "persona_probe_report.csv")
    bench.write_csv_with_context(ctx, probe_path, probe_rows)
    ctx.register_artifact(probe_path, "table", "Persona/register/voice held-out probe sweep with shuffled/random controls.")
    results_path = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results_path, probe_rows)
    ctx.register_artifact(results_path, "results", "Alias of persona_probe_report.csv for the standard run contract.")
    print(f"[lab17] selected persona/register depth {best_depth}")

    directions = build_directions_at_depth(items, features, split, best_depth)
    refusal_direction = build_refusal_monitor_direction(bundle, best_depth)
    trace_dirs = {**directions, "refusal_monitor": refusal_direction}
    cos_rows = direction_cosine_rows(trace_dirs)
    cos_path = ctx.path("tables", "direction_cosines.csv")
    bench.write_csv_with_context(ctx, cos_path, cos_rows)
    ctx.register_artifact(cos_path, "table", "Pairwise cosines among persona/register/voice/agreement/refusal-monitor directions.")

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
    generation_path = ctx.path("tables", "persona_steering_generations.csv")
    bench.write_csv_with_context(ctx, generation_path, steering_generations)
    ctx.register_artifact(generation_path, "table", "Baseline/trait/opposite/random steered generations with scoring scaffold.")
    effects_path = ctx.path("tables", "persona_steering_effects.csv")
    bench.write_csv_with_context(ctx, effects_path, steering_effects)
    ctx.register_artifact(effects_path, "table", "Style-marker and content-keyword steering effects by trait.")
    register_scores = [row for row in steering_generations if row["trait"] == "technical_register"]
    register_path = ctx.path("tables", "register_content_style_scores.csv")
    bench.write_csv_with_context(ctx, register_path, register_scores)
    ctx.register_artifact(register_path, "table", "Technical-register steering rows with content and style scores.")

    turn_rows, turn_slopes, turn_check = run_turn_trace(
        ctx,
        bundle,
        directions,
        refusal_direction,
        best_depth,
        args.seed,
    )
    turn_check_path = ctx.path("diagnostics", "turn_boundary_check.json")
    bench.write_json(turn_check_path, turn_check)
    ctx.register_artifact(turn_check_path, "diagnostic", "Chat-template turn segmentation checks for Lab 17 scripted traces.")
    trace_path = ctx.path("tables", "persona_turn_trace.csv")
    bench.write_csv_with_context(ctx, trace_path, turn_rows)
    ctx.register_artifact(trace_path, "table", "Per-turn persona/register/voice/refusal/random projections over scripted conversations.")
    slope_path = ctx.path("tables", "persona_turn_trace_slopes.csv")
    bench.write_csv_with_context(ctx, slope_path, turn_slopes)
    ctx.register_artifact(slope_path, "table", "Projection slopes for Lab 17 scripted traces.")

    state_common = {
        "depth": best_depth,
        "depth_convention": "bench streams[k]: 0 = embeddings, k = residual after block k",
        "read_site": "chat-templated final prompt token before assistant generation",
        "model_id": bundle.anatomy.model_id,
        "d_model": bundle.anatomy.d_model,
        "n_layers": bundle.anatomy.n_layers,
        "method": "train-split paired positive-minus-negative mass-mean directions",
    }
    persona_path = ctx.path("state", "persona_directions.pt")
    torch.save({**state_common, "directions": directions, "refusal_monitor": refusal_direction}, persona_path)
    ctx.register_artifact(persona_path, "tensor", "Persona/register/voice/agreement directions plus refusal monitor.")
    register_state_path = ctx.path("state", "register_direction.pt")
    torch.save({**state_common, "direction": directions["technical_register"]}, register_state_path)
    ctx.register_artifact(register_state_path, "tensor", "Technical-register direction for downstream labs.")
    voice_state_path = ctx.path("state", "voice_directions.pt")
    torch.save({
        **state_common,
        "directions": {
            "warm_supportive_voice": directions["warm_supportive_voice"],
            "honest_disagreement": directions["honest_disagreement"],
        },
    }, voice_state_path)
    ctx.register_artifact(voice_state_path, "tensor", "Voice/agreement directions for downstream labs.")
    meta_path = ctx.path("state", "persona_voice_register_metadata.json")
    bench.write_json(meta_path, {**state_common, "directions": sorted(directions), "includes_refusal_monitor": True})
    ctx.register_artifact(meta_path, "state", "Human-readable metadata for Lab 17 saved directions.")

    if not args.no_plots:
        plot_persona_turn_trace(ctx, turn_rows)
        plot_register_switch_trace(ctx, turn_rows)
        plot_refusal_projection(ctx, turn_rows)

    real_aucs = [
        metric_at(probe_rows, trait, "real", best_depth)
        for trait in sorted({item.trait for item in items})
    ]
    shuf_aucs = [
        metric_at(probe_rows, trait, "shuffled_sign", best_depth)
        for trait in sorted({item.trait for item in items})
    ]
    random_aucs = [
        metric_at(probe_rows, trait, "random_oriented", best_depth)
        for trait in sorted({item.trait for item in items})
    ]
    trait_deltas = [
        float(row["style_margin_delta_vs_baseline"])
        for row in steering_effects
        if row["steering_condition"] == "trait_direction"
        and isinstance(row.get("style_margin_delta_vs_baseline"), (int, float))
    ]
    random_deltas = [
        float(row["style_margin_delta_vs_baseline"])
        for row in steering_effects
        if row["steering_condition"] == "random_direction"
        and isinstance(row.get("style_margin_delta_vs_baseline"), (int, float))
    ]
    metrics = {
        "model_id": bundle.anatomy.model_id,
        "n_rows": len(items),
        "n_eval_rows": len(eval_items),
        "best_depth": best_depth,
        "injection_layer": max(0, best_depth - 1),
        "mean_real_auc_best_depth": none_if_nan(safe_fmean(real_aucs)),
        "mean_shuffled_auc_best_depth": none_if_nan(safe_fmean(shuf_aucs)),
        "mean_random_auc_best_depth": none_if_nan(safe_fmean(random_aucs)),
        "mean_real_selectivity_vs_shuffled": none_if_nan(safe_fmean(real_aucs) - safe_fmean(shuf_aucs)),
        "mean_trait_steering_style_delta": rounded(safe_fmean(trait_deltas)),
        "mean_random_steering_style_delta": rounded(safe_fmean(random_deltas)),
        "technical_register_style_delta": none_if_nan(effect_delta(
            steering_effects, "technical_register", "trait_direction", "style_margin_delta_vs_baseline"
        )),
        "technical_register_content_delta": none_if_nan(effect_delta(
            steering_effects, "technical_register", "trait_direction", "content_hit_delta_vs_baseline"
        )),
        "museum_roleplay_persona_slope": none_if_nan(trace_slope(turn_slopes, "museum_roleplay", "persona_museum_guide")),
        "museum_roleplay_default_slope": none_if_nan(trace_slope(turn_slopes, "museum_roleplay", "default_assistant_control")),
        "museum_roleplay_random_slope": none_if_nan(trace_slope(turn_slopes, "museum_roleplay", "random_null")),
        "register_switch_technical_slope": none_if_nan(trace_slope(turn_slopes, "register_switch", "technical_register")),
        "refusal_boundary_slope": none_if_nan(trace_slope(turn_slopes, "roleplay_boundary", "refusal_monitor")),
        "steering_dose_fraction": STEERING_DOSE,
        "data": data_info,
    }
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 17 metrics.")

    write_operationalization_audit(ctx, metrics)
    write_run_summary(ctx, metrics)

    run_name = ctx.run_dir.name
    claims = [
        {
            "id": f"{LAB_ID}-C1",
            "tag": "CAUSAL",
            "text": (
                f"For {bundle.anatomy.model_id}, trait-direction steering changed held-out "
                f"style-marker margins by mean delta {metrics['mean_trait_steering_style_delta']} "
                f"versus random delta {metrics['mean_random_steering_style_delta']}. This is a "
                "persona/register/voice handle claim, not evidence of a true inner identity."
            ),
            "artifact": f"runs/{run_name}/tables/persona_steering_effects.csv",
            "falsifier": (
                "Random/opposite steering matches the effect, content accuracy collapses, or hand labels "
                "show the marker rubric is misclassifying style."
            ),
        },
        {
            "id": f"{LAB_ID}-C2",
            "tag": "DECODE",
            "text": (
                f"At depth {best_depth}, paired persona/register/voice directions separate held-out "
                f"positive prompts from matched controls with mean AUC "
                f"{metrics['mean_real_auc_best_depth']} versus shuffled "
                f"{metrics['mean_shuffled_auc_best_depth']}. Multi-turn traces remain descriptive "
                "unless they beat the null controls in the trace tables."
            ),
            "artifact": f"runs/{run_name}/tables/persona_probe_report.csv",
            "falsifier": (
                "Shuffled-sign or random controls match the AUC, or topic/style controls explain the "
                "direction without a persistent persona/register component."
            ),
        },
    ]
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)
