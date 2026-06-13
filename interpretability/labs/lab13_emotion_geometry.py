"""Lab 13: Emotion geometry, input/output transfer, and affect steering.

This lab is the first advanced "voice" lab. It asks a narrow, mechanistic
version of a question that otherwise becomes philosophical too fast:

    When an instruct model READS emotional text and when it WRITES emotional
    text, are we touching the same residual-stream geometry?

The object of study is not feeling. It is an operational contrast:
emotion-laden text/prompt minus a neutral paraphrase/prompt about the same
cause. The lab extracts directions two ways:

  * comprehension direction: the model reads emotional text;
  * generation direction: the model is asked to write in that emotion.

Then it cross-tests them. A direction read from comprehension should decode
generation prompts if there is shared input/output geometry; the reverse
tests the same bridge from the write side. Finally, a small generation sweep
injects the input-derived direction and checks whether output affect shifts
more than a random direction.

Evidence labels:
  * DECODE for read/write transfer and direction cosines;
  * CAUSAL, scoped, for activation-addition effects on generated text.

Cheap explanations are the point of the lab: topic/cause, valence, arousal,
and generic sentiment are all audited. Positive results are only "emotion
specific" if they generalize across causes and do not collapse into the
Lab-7-style sentiment direction.
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

LAB_ID = "L13"
DATA_FILE = "affect_emotion_pairs.csv"
VALENCE_FILE = "affect_valence.csv"

EMOTIONS = ("joy", "sadness", "anger", "fear")
SOURCES = ("comprehension", "generation")
FEATURE_SUFFIXES = ("emotion", "neutral")

PROMPT_SET_LIMITS: dict[str, dict[str, int]] = {
    "small": {"per_emotion": 3, "confounds": 4, "sentiment_pairs": 6, "steering_per_emotion": 1},
    "medium": {"per_emotion": 5, "confounds": 6, "sentiment_pairs": 12, "steering_per_emotion": 2},
    "full": {"per_emotion": 0, "confounds": 0, "sentiment_pairs": 24, "steering_per_emotion": 3},
}

TRAIN_FRACTION = 0.67
MAX_NEW_TOKENS = 40
ENGINE_MAX_CONCURRENT = 16
STEERING_DOSE = 0.8

COMPREHENSION_SYSTEM = "You are a careful reader. Answer exactly as requested."
GENERATION_SYSTEM = "You write one sentence in the requested style."

EMOTION_WORDS: dict[str, set[str]] = {
    "joy": set(
        """
        joy joyful joyous happy happiness delighted delight delightedly cheerful cheer
        cheers cheered grin grinned smile smiled smiling laughter laugh laughed
        thrilled thrill wonderful bright warm glad grateful proud celebrate celebrated
        relief relieved pleasant pleased fun excited
        """.split()
    ),
    "sadness": set(
        """
        sad sadness sorrow sorrowful grief grieving grieved mourn mourning tears tear
        lonely loneliness alone loss lost empty heavy disappointed disappointment
        painful quietly quiet sorrowfully unhappy miserable melancholy
        """.split()
    ),
    "anger": set(
        """
        anger angry angrily furious fury rage outraged unfair injustice insulted
        insulting bitter bristle bristled irritation irritated frustrating frustrated
        frustration clenched clench sharp resentful resentment annoyed annoyedly
        """.split()
    ),
    "fear": set(
        """
        fear fearful afraid anxiety anxious panic panicked terrified terror dread
        dreadful nervous danger dangerous worried worry alarm alarmed froze frozen
        trembling tense uneasy scared fright frightened
        """.split()
    ),
}

POSITIVE_WORDS = EMOTION_WORDS["joy"] | {
    "calm", "peaceful", "pleasant", "orderly", "safe", "comfortable", "gentle"
}
NEGATIVE_WORDS = EMOTION_WORDS["sadness"] | EMOTION_WORDS["anger"] | EMOTION_WORDS["fear"]
AROUSAL_WORDS = {
    "rapidly", "quickly", "alarm", "flared", "rush", "rushed", "froze", "panic",
    "urgent", "shook", "thunder", "cheered", "furious", "terror", "excited",
}


@dataclasses.dataclass
class EmotionItem:
    item_id: str
    emotion: str
    cause: str
    arousal: str
    valence: str
    content_text: str
    neutral_text: str
    generation_prompt: str
    neutral_generation_prompt: str
    confound: str
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


def none_if_nan(x: Any) -> Any:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def safe_fmean(vals: Sequence[float], default: float = float("nan")) -> float:
    finite = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
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


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def emotion_counts(text: str) -> dict[str, int]:
    toks = words(text)
    return {emo: sum(1 for w in toks if w in lex) for emo, lex in EMOTION_WORDS.items()}


def target_margin(text: str, emotion: str) -> tuple[float, int, int]:
    counts = emotion_counts(text)
    target = counts.get(emotion, 0)
    other = max((v for k, v in counts.items() if k != emotion), default=0)
    length_norm = math.sqrt(max(1, len(words(text))))
    return (target - other) / length_norm, target, other


def valence_score(text: str) -> float:
    toks = words(text)
    pos = sum(1 for w in toks if w in POSITIVE_WORDS)
    neg = sum(1 for w in toks if w in NEGATIVE_WORDS)
    return (pos - neg) / math.sqrt(max(1, len(toks)))


def arousal_score(text: str) -> float:
    toks = words(text)
    return sum(1 for w in toks if w in AROUSAL_WORDS) / math.sqrt(max(1, len(toks)))


def unit(v: Any) -> Any:
    norm = v.norm().clamp_min(1e-9)
    if not bool(norm.isfinite()):
        raise RuntimeError("Direction norm was not finite.")
    return v / norm


def cosine(a: Any, b: Any) -> float:
    denom = (a.norm() * b.norm()).clamp_min(1e-9)
    return float((a @ b) / denom)


def random_unit(d_model: int, seed: int) -> Any:
    import torch

    gen = torch.Generator().manual_seed(seed)
    return unit(torch.randn(d_model, generator=gen))


def data_path(name: str) -> Any:
    path = bench.COURSE_ROOT / "data" / name
    if not path.exists():
        raise RuntimeError(f"Frozen dataset missing: {path}")
    return path


def comprehension_user_message(text: str) -> str:
    return (
        "Read this sentence and keep its emotional content in mind.\n"
        f"Sentence: {text}\n"
        "Reply with exactly one word: done"
    )


def render_comprehension(bundle: bench.ModelBundle, text: str) -> str:
    return bench.apply_chat_template(
        bundle,
        comprehension_user_message(text),
        system=COMPREHENSION_SYSTEM,
        add_generation_prompt=True,
    )


def render_generation(bundle: bench.ModelBundle, prompt: str) -> str:
    return bench.apply_chat_template(
        bundle,
        prompt,
        system=GENERATION_SYSTEM,
        add_generation_prompt=True,
    )


def last_token_streams(bundle: bench.ModelBundle, templated_prompt: str) -> Any:
    cap = bench.run_with_residual_cache(bundle, templated_prompt, add_special_tokens=False)
    return cap.streams[:, -1, :]


def load_items(args: Any) -> tuple[list[EmotionItem], list[EmotionItem], dict[str, Any]]:
    path = data_path(DATA_FILE)
    all_rows: list[EmotionItem] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            all_rows.append(EmotionItem(**row))

    selected = tuple(
        e.strip() for e in str(getattr(args, "emotions", "")).split(",") if e.strip()
    ) or EMOTIONS
    unknown = set(selected) - {r.emotion for r in all_rows}
    if unknown:
        raise ValueError(f"--emotions included unknown labels: {sorted(unknown)}")

    limits = PROMPT_SET_LIMITS.get(str(args.prompt_set))
    if limits is None:
        raise ValueError("Lab 13 uses --prompt-set small|medium|full, or pass --max-examples.")
    per_emotion = limits["per_emotion"]
    confound_cap = limits["confounds"]
    if getattr(args, "max_examples", 0) and args.max_examples > 0:
        per_emotion = args.max_examples
        confound_cap = min(confound_cap or args.max_examples, args.max_examples)

    targets: list[EmotionItem] = []
    by_emotion: dict[str, list[EmotionItem]] = defaultdict(list)
    for item in all_rows:
        if item.emotion in selected and not item.confound:
            by_emotion[item.emotion].append(item)
    for emotion in selected:
        rows = by_emotion[emotion]
        if per_emotion > 0:
            rows = rows[:per_emotion]
        if len(rows) < 2:
            raise RuntimeError(
                f"Lab 13 needs at least two items for {emotion!r}; got {len(rows)}. "
                "Use a larger --max-examples or prompt set."
            )
        targets.extend(rows)

    confounds = [item for item in all_rows if item.confound]
    if confound_cap > 0:
        confounds = confounds[:confound_cap]

    info = {
        "data_file": DATA_FILE,
        "data_sha256": bench.sha256_file(path),
        "selected_emotions": list(selected),
        "per_emotion_cap": per_emotion,
        "confound_cap": confound_cap,
        "n_target_items": len(targets),
        "n_confound_items": len(confounds),
        "counts_by_emotion": {e: sum(1 for item in targets if item.emotion == e) for e in selected},
        "confounds": sorted({item.confound for item in confounds}),
    }
    return targets, confounds, info


def make_split(items: Sequence[EmotionItem], seed: int) -> dict[str, bool]:
    """Deterministic train/eval split inside each emotion, grouped by item/cause."""
    split: dict[str, bool] = {}
    by_emotion: dict[str, list[EmotionItem]] = defaultdict(list)
    for item in items:
        by_emotion[item.emotion].append(item)
    for emotion, rows in by_emotion.items():
        ranked = sorted(rows, key=lambda it: stable_hash_int(f"{seed}:{emotion}:{it.cause}:{it.item_id}"))
        n_train = int(round(TRAIN_FRACTION * len(ranked)))
        n_train = max(1, min(len(ranked) - 1, n_train))
        train_ids = {it.item_id for it in ranked[:n_train]}
        for item in rows:
            split[item.item_id] = item.item_id in train_ids
    return split


def cache_features(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    targets: Sequence[EmotionItem],
    confounds: Sequence[EmotionItem],
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    all_items = list(targets) + list(confounds)
    report_every = max(1, len(all_items) // 5)
    for i, item in enumerate(all_items):
        templated = {
            "comprehension_emotion": render_comprehension(bundle, item.content_text),
            "comprehension_neutral": render_comprehension(bundle, item.neutral_text),
            "generation_emotion": render_generation(bundle, item.generation_prompt),
            "generation_neutral": render_generation(bundle, item.neutral_generation_prompt),
        }
        features[item.item_id] = {
            key: last_token_streams(bundle, prompt) for key, prompt in templated.items()
        }
        for key, prompt in templated.items():
            rows.append({
                "item_id": item.item_id,
                "emotion": item.emotion,
                "cause": item.cause,
                "confound": item.confound,
                "feature_key": key,
                "n_tokens": len(bundle.tokenizer(prompt, add_special_tokens=False)["input_ids"]),
            })
        if (i + 1) % report_every == 0:
            print(f"[lab13] cached {i + 1}/{len(all_items)} items")

    path = ctx.path("diagnostics", "prompt_token_counts.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "diagnostic", "Chat-rendered prompt token counts for cached read/write contrasts.")
    return features


def feature_key(source: str, suffix: str) -> str:
    return f"{source}_{suffix}"


def paired_delta(features: Mapping[str, dict[str, Any]], item: EmotionItem, source: str, depth: int) -> Any:
    return features[item.item_id][feature_key(source, "emotion")][depth] - features[item.item_id][feature_key(source, "neutral")][depth]


def direction_for(
    items: Sequence[EmotionItem],
    features: Mapping[str, dict[str, Any]],
    split: Mapping[str, bool],
    source: str,
    emotion: str,
    depth: int,
    *,
    train_only: bool = True,
) -> Any | None:
    import torch

    deltas = [
        paired_delta(features, item, source, depth)
        for item in items
        if item.emotion == emotion and ((not train_only) or split[item.item_id])
    ]
    if not deltas:
        return None
    return unit(torch.stack(deltas).mean(dim=0))


def shuffled_direction_for(
    items: Sequence[EmotionItem],
    features: Mapping[str, dict[str, Any]],
    split: Mapping[str, bool],
    source: str,
    emotion: str,
    depth: int,
    seed: int,
) -> Any | None:
    import torch

    deltas = [
        paired_delta(features, item, source, depth)
        for item in items
        if item.emotion == emotion and split[item.item_id]
    ]
    if len(deltas) < 2:
        return direction_for(items, features, split, source, emotion, depth)
    order = sorted(range(len(deltas)), key=lambda i: stable_hash_int(f"{seed}:shuffle:{emotion}:{source}:{i}"))
    flip = set(order[: len(order) // 2])
    signed = [(-d if i in flip else d) for i, d in enumerate(deltas)]
    return unit(torch.stack(signed).mean(dim=0))


def orient_on_train(
    direction: Any,
    items: Sequence[EmotionItem],
    features: Mapping[str, dict[str, Any]],
    split: Mapping[str, bool],
    source: str,
    emotion: str,
    depth: int,
) -> Any:
    pos, neg = projection_scores(items, features, split, direction, source, emotion, depth, train=True)
    if pos and neg and safe_fmean(pos, 0.0) < safe_fmean(neg, 0.0):
        return -direction
    return direction


def projection_scores(
    items: Sequence[EmotionItem],
    features: Mapping[str, dict[str, Any]],
    split: Mapping[str, bool],
    direction: Any,
    target_source: str,
    emotion: str,
    depth: int,
    *,
    train: bool,
    cause: str | None = None,
) -> tuple[list[float], list[float]]:
    pos: list[float] = []
    neg: list[float] = []
    for item in items:
        if item.emotion != emotion:
            continue
        if cause is not None and item.cause != cause:
            continue
        if split[item.item_id] != train:
            continue
        pos.append(float(features[item.item_id][feature_key(target_source, "emotion")][depth] @ direction))
        neg.append(float(features[item.item_id][feature_key(target_source, "neutral")][depth] @ direction))
    return pos, neg


def evaluate_direction(
    items: Sequence[EmotionItem],
    features: Mapping[str, dict[str, Any]],
    split: Mapping[str, bool],
    direction: Any,
    target_source: str,
    emotion: str,
    depth: int,
    *,
    train: bool = False,
    cause: str | None = None,
) -> dict[str, Any]:
    pos, neg = projection_scores(
        items, features, split, direction, target_source, emotion, depth, train=train, cause=cause
    )
    return {
        "auc": auc_from_scores(pos, neg),
        "mean_pos": safe_fmean(pos),
        "mean_neg": safe_fmean(neg),
        "margin": safe_fmean(pos) - safe_fmean(neg),
        "n_pos": len(pos),
        "n_neg": len(neg),
    }


def build_directions_at_depth(
    items: Sequence[EmotionItem],
    features: Mapping[str, dict[str, Any]],
    split: Mapping[str, bool],
    depth: int,
) -> dict[tuple[str, str], Any]:
    dirs: dict[tuple[str, str], Any] = {}
    for source in SOURCES:
        for emotion in sorted({item.emotion for item in items}):
            d = direction_for(items, features, split, source, emotion, depth)
            if d is not None:
                dirs[(source, emotion)] = d
    return dirs


def scan_transfer_depths(
    items: Sequence[EmotionItem],
    features: Mapping[str, dict[str, Any]],
    split: Mapping[str, bool],
    n_depths: int,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    for depth in range(1, n_depths):
        dirs = build_directions_at_depth(items, features, split, depth)
        for (source, emotion), d in dirs.items():
            for target_source in SOURCES:
                ev = evaluate_direction(items, features, split, d, target_source, emotion, depth, train=False)
                rows.append({
                    "depth": depth,
                    "direction_kind": "real",
                    "direction_source": source,
                    "eval_target": target_source,
                    "emotion": emotion,
                    "auc": rounded(ev["auc"]),
                    "selectivity_vs_chance": rounded(ev["auc"] - 0.5) if math.isfinite(ev["auc"]) else "",
                    "mean_pos": rounded(ev["mean_pos"]),
                    "mean_neg": rounded(ev["mean_neg"]),
                    "margin": rounded(ev["margin"]),
                    "n_pos": ev["n_pos"],
                    "n_neg": ev["n_neg"],
                })

    def depth_score(depth: int) -> float:
        vals = [
            float(r["auc"]) - 0.5
            for r in rows
            if r["depth"] == depth
            and r["direction_kind"] == "real"
            and r["direction_source"] != r["eval_target"]
            and isinstance(r["auc"], (int, float))
            and math.isfinite(float(r["auc"]))
        ]
        return safe_fmean(vals, default=-1.0)

    best_depth = max(range(1, n_depths), key=depth_score)
    return rows, best_depth


def add_controls_at_depth(
    rows: list[dict[str, Any]],
    items: Sequence[EmotionItem],
    features: Mapping[str, dict[str, Any]],
    split: Mapping[str, bool],
    depth: int,
    seed: int,
    sentiment_direction: Any | None,
    d_model: int,
) -> None:
    for source in SOURCES:
        for emotion in sorted({item.emotion for item in items}):
            shuffled = shuffled_direction_for(items, features, split, source, emotion, depth, seed)
            random = orient_on_train(
                random_unit(d_model, seed + stable_hash_int(f"{source}:{emotion}:{depth}") % 10_000_000),
                items,
                features,
                split,
                source,
                emotion,
                depth,
            )
            controls = [("shuffled", source, shuffled), ("random", source, random)]
            if sentiment_direction is not None:
                controls.append(("sentiment_control", "lab7_style_sentiment", sentiment_direction))
            for kind, direction_source, d in controls:
                if d is None:
                    continue
                for target_source in SOURCES:
                    ev = evaluate_direction(items, features, split, d, target_source, emotion, depth, train=False)
                    rows.append({
                        "depth": depth,
                        "direction_kind": kind,
                        "direction_source": direction_source,
                        "eval_target": target_source,
                        "emotion": emotion,
                        "auc": rounded(ev["auc"]),
                        "selectivity_vs_chance": rounded(ev["auc"] - 0.5) if math.isfinite(ev["auc"]) else "",
                        "mean_pos": rounded(ev["mean_pos"]),
                        "mean_neg": rounded(ev["mean_neg"]),
                        "margin": rounded(ev["margin"]),
                        "n_pos": ev["n_pos"],
                        "n_neg": ev["n_neg"],
                    })


def build_sentiment_direction(
    bundle: bench.ModelBundle,
    depth: int,
    cap_pairs: int,
) -> tuple[Any | None, int]:
    import torch

    positives: list[str] = []
    negatives: list[str] = []
    with data_path(VALENCE_FILE).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"] == "1":
                positives.append(row["statement"])
            else:
                negatives.append(row["statement"])
    n = min(cap_pairs, len(positives), len(negatives))
    if n < 2:
        return None, 0
    diffs = []
    for pos, neg in zip(positives[:n], negatives[:n]):
        pos_vec = last_token_streams(bundle, render_comprehension(bundle, pos))[depth]
        neg_vec = last_token_streams(bundle, render_comprehension(bundle, neg))[depth]
        diffs.append(pos_vec - neg_vec)
    return unit(torch.stack(diffs).mean(dim=0)), n


def cross_cause_generalization(
    items: Sequence[EmotionItem],
    features: Mapping[str, dict[str, Any]],
    split: Mapping[str, bool],
    depth: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for emotion in sorted({item.emotion for item in items}):
        causes = sorted({item.cause for item in items if item.emotion == emotion})
        for held_cause in causes:
            pseudo_split = {
                item.item_id: item.emotion == emotion and item.cause != held_cause
                for item in items
            }
            for source in SOURCES:
                d = direction_for(items, features, pseudo_split, source, emotion, depth)
                if d is None:
                    continue
                for target_source in SOURCES:
                    ev = evaluate_direction(
                        items,
                        features,
                        {item.item_id: False for item in items},
                        d,
                        target_source,
                        emotion,
                        depth,
                        train=False,
                        cause=held_cause,
                    )
                    rows.append({
                        "emotion": emotion,
                        "held_out_cause": held_cause,
                        "direction_source": source,
                        "eval_target": target_source,
                        "depth": depth,
                        "auc": rounded(ev["auc"]),
                        "margin": rounded(ev["margin"]),
                        "n_pos": ev["n_pos"],
                        "n_neg": ev["n_neg"],
                        "n_train_causes": max(0, len(causes) - 1),
                    })
    return rows


def cosine_rows(
    dirs: Mapping[tuple[str, str], Any],
    sentiment_direction: Any | None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    vectors: dict[str, Any] = {f"{source}_{emotion}": d for (source, emotion), d in dirs.items()}
    if sentiment_direction is not None:
        vectors["sentiment_lab7_style"] = sentiment_direction
    labels = sorted(vectors)
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if i < j:
                rows.append({"direction_a": a, "direction_b": b, "cosine": rounded(cosine(vectors[a], vectors[b]))})
    return rows, labels, vectors


def confound_projection_audit(
    confounds: Sequence[EmotionItem],
    features: Mapping[str, dict[str, Any]],
    dirs: Mapping[tuple[str, str], Any],
    depth: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in confounds:
        for (source, emotion), d in dirs.items():
            pos = float(features[item.item_id][feature_key(source, "emotion")][depth] @ d)
            neg = float(features[item.item_id][feature_key(source, "neutral")][depth] @ d)
            rows.append({
                "item_id": item.item_id,
                "confound": item.confound,
                "label": item.emotion,
                "cause": item.cause,
                "direction": f"{source}_{emotion}",
                "projection_delta": rounded(pos - neg),
                "content_projection": rounded(pos),
                "neutral_projection": rounded(neg),
                "valence_score_content": rounded(valence_score(item.content_text)),
                "arousal_score_content": rounded(arousal_score(item.content_text)),
            })
    return rows


def selected_steering_items(
    items: Sequence[EmotionItem],
    split: Mapping[str, bool],
    per_emotion: int,
) -> list[EmotionItem]:
    selected: list[EmotionItem] = []
    for emotion in sorted({item.emotion for item in items}):
        eval_rows = [item for item in items if item.emotion == emotion and not split[item.item_id]]
        train_rows = [item for item in items if item.emotion == emotion and split[item.item_id]]
        selected.extend((eval_rows or train_rows)[:per_emotion])
    return selected


def run_steering(
    bundle: bench.ModelBundle,
    items: Sequence[EmotionItem],
    dirs: Mapping[tuple[str, str], Any],
    split: Mapping[str, bool],
    depth: int,
    d_model: int,
    seed: int,
    per_emotion: int,
    ref_norm: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    generation_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    injection_layer = depth - 1
    chosen = selected_steering_items(items, split, per_emotion)
    by_emotion: dict[str, list[EmotionItem]] = defaultdict(list)
    for item in chosen:
        by_emotion[item.emotion].append(item)

    for emotion, rows in sorted(by_emotion.items()):
        prompts = [render_generation(bundle, item.generation_prompt) for item in rows]
        conditions: list[tuple[str, Any | None, float]] = [
            ("baseline", None, 0.0),
            ("input_direction", dirs.get(("comprehension", emotion)), STEERING_DOSE * ref_norm),
            ("random", random_unit(d_model, seed + stable_hash_int(f"steer:{emotion}") % 10_000_000), STEERING_DOSE * ref_norm),
        ]
        for condition, direction, abs_scale in conditions:
            if direction is None and condition != "baseline":
                continue
            if condition == "baseline":
                outs = bench.generate_continuous(
                    bundle,
                    prompts,
                    MAX_NEW_TOKENS,
                    max_concurrent=ENGINE_MAX_CONCURRENT,
                    skip_special_tokens=True,
                    progress_label=f"lab13-{emotion}-{condition}",
                )
            else:
                outs = bench.generate_continuous(
                    bundle,
                    prompts,
                    MAX_NEW_TOKENS,
                    max_concurrent=ENGINE_MAX_CONCURRENT,
                    skip_special_tokens=True,
                    progress_label=f"lab13-{emotion}-{condition}",
                    steer=(injection_layer, direction, abs_scale),
                )
            for item, text in zip(rows, outs):
                margin, target_hits, other_hits = target_margin(text, emotion)
                generation_rows.append({
                    "item_id": item.item_id,
                    "emotion": emotion,
                    "cause": item.cause,
                    "condition": condition,
                    "dose_fraction_of_median_norm": 0.0 if condition == "baseline" else STEERING_DOSE,
                    "absolute_scale": rounded(abs_scale),
                    "target_margin": rounded(margin),
                    "target_hits": target_hits,
                    "other_emotion_hits": other_hits,
                    "valence_score": rounded(valence_score(text)),
                    "arousal_score": rounded(arousal_score(text)),
                    "hand_label": "",
                    "prompt": item.generation_prompt,
                    "generation": text,
                })

    for emotion in sorted(by_emotion):
        base = [
            float(r["target_margin"]) for r in generation_rows
            if r["emotion"] == emotion and r["condition"] == "baseline"
        ]
        real = [
            float(r["target_margin"]) for r in generation_rows
            if r["emotion"] == emotion and r["condition"] == "input_direction"
        ]
        rnd = [
            float(r["target_margin"]) for r in generation_rows
            if r["emotion"] == emotion and r["condition"] == "random"
        ]
        effect_rows.append({
            "emotion": emotion,
            "baseline_mean": rounded(safe_fmean(base)),
            "input_direction_mean": rounded(safe_fmean(real)),
            "random_mean": rounded(safe_fmean(rnd)),
            "input_delta_vs_baseline": rounded(safe_fmean(real) - safe_fmean(base)),
            "random_delta_vs_baseline": rounded(safe_fmean(rnd) - safe_fmean(base)),
            "input_over_random_delta": rounded((safe_fmean(real) - safe_fmean(base)) - (safe_fmean(rnd) - safe_fmean(base))),
            "n_prompts": len(base),
            "injection_layer": injection_layer,
            "stream_depth": depth,
        })
    return generation_rows, effect_rows


def plot_transfer_matrix(
    ctx: bench.RunContext,
    rows: Sequence[Mapping[str, Any]],
    emotions: Sequence[str],
    depth: int,
) -> None:
    import numpy as np

    labels = [f"{a}->{b}" for a in SOURCES for b in SOURCES]
    grid = np.full((len(emotions), len(labels)), np.nan)
    for i, emotion in enumerate(emotions):
        for j, label in enumerate(labels):
            source, target = label.split("->")
            vals = [
                float(r["auc"]) for r in rows
                if r["depth"] == depth
                and r["direction_kind"] == "real"
                and r["emotion"] == emotion
                and r["direction_source"] == source
                and r["eval_target"] == target
                and isinstance(r["auc"], (int, float))
            ]
            if vals:
                grid[i, j] = vals[0]
    fig, ax = bench.new_figure(figsize=(8.6, 4.8))
    im = ax.imshow(grid, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(emotions)))
    ax.set_yticklabels(emotions)
    ax.set_title(f"Emotion read/write transfer at depth {depth}")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.035, label="AUC (emotion prompt/text vs neutral match)")
    fig.tight_layout()
    bench.save_figure(
        ctx,
        fig,
        "emotion_transfer_matrix.png",
        "Comprehension/generation direction transfer matrix by emotion at the selected depth.",
    )


def plot_cosines(
    ctx: bench.RunContext,
    labels: Sequence[str],
    vectors: Mapping[str, Any],
    depth: int,
) -> None:
    import numpy as np

    grid = np.full((len(labels), len(labels)), np.nan)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            grid[i, j] = cosine(vectors[a], vectors[b])
    fig, ax = bench.new_figure(figsize=(0.48 * len(labels) + 5.0, 0.48 * len(labels) + 4.6))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"Emotion direction cosines at depth {depth}")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
    fig.colorbar(im, ax=ax, fraction=0.035, label="cosine")
    fig.tight_layout()
    bench.save_figure(
        ctx,
        fig,
        "emotion_direction_cosines.png",
        "Pairwise cosines among comprehension/generation emotion directions and the sentiment control.",
    )


def write_operationalization_audit(
    ctx: bench.RunContext,
    metrics: Mapping[str, Any],
    confound_rows: Sequence[Mapping[str, Any]],
) -> None:
    top_confound = sorted(
        [r for r in confound_rows if isinstance(r.get("projection_delta"), (int, float))],
        key=lambda r: abs(float(r["projection_delta"])),
        reverse=True,
    )[:5]
    lines = [
        "# Lab 13 Operationalization Audit",
        "",
        "## What Was Measured",
        "",
        "The lab measures residual-stream directions for paired contrasts: emotional content/prompt minus a neutral paraphrase/prompt about the same cause. A positive result is a property of this operationalization, not evidence that the model feels the emotion.",
        "",
        "## Cheap Explanations Under Audit",
        "",
        "- Topic/cause: every target item is paired with a neutral paraphrase about the same cause; cross-cause rows are held out by cause.",
        "- Valence/sentiment: the run recomputes a Lab-7-style positive-vs-negative sentiment direction from `affect_valence.csv` and reports its cosine with every emotion direction.",
        "- Arousal: target items carry arousal labels, and confound rows include high-arousal neutral examples.",
        "- Surprise and calm positivity: confound rows include surprising-neutral and positive-calm examples so students can inspect false positives.",
        "",
        "## Current Run Headline",
        "",
        f"- Best depth: {metrics.get('best_depth')}",
        f"- Mean cross input/output transfer AUC: {metrics.get('mean_cross_transfer_auc')}",
        f"- Mean same-emotion comprehension/generation cosine: {metrics.get('mean_comp_gen_cosine')}",
        f"- Max absolute sentiment-control cosine: {metrics.get('max_abs_sentiment_cosine')}",
        f"- Steering input-over-random delta: {metrics.get('steering_input_over_random_delta')}",
        "",
        "## Largest Confound Projection Deltas",
        "",
    ]
    if not top_confound:
        lines.append("No confound projection rows were available.")
    else:
        for row in top_confound:
            lines.append(
                f"- {row['item_id']} ({row['confound']}) on {row['direction']}: "
                f"delta {row['projection_delta']}"
            )
    lines += [
        "",
        "## Allowed Claim",
        "",
        "An emotion-specific direction claim is only defensible if it generalizes across causes, beats shuffled/random controls, and does not become a synonym for sentiment or arousal. Otherwise the honest claim is about a broader affect, valence, or prompt-style direction.",
        "",
    ]
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "audit", "Operationalization limits and cheap-explanation audit for Lab 13.")


def write_run_summary(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    lines = [
        f"# Lab 13 Run Summary: {ctx.run_dir.name}",
        "",
        f"- Model: `{metrics.get('model_id')}`",
        f"- Target items: {metrics.get('n_target_items')}",
        f"- Best stream depth: {metrics.get('best_depth')}",
        f"- Injection layer for steering: {metrics.get('injection_layer')}",
        f"- Mean cross input/output transfer AUC: {metrics.get('mean_cross_transfer_auc')}",
        f"- Mean comprehension/generation cosine: {metrics.get('mean_comp_gen_cosine')}",
        f"- Max abs cosine with sentiment control: {metrics.get('max_abs_sentiment_cosine')}",
        f"- Input-over-random steering delta: {metrics.get('steering_input_over_random_delta')}",
        "",
        "Read `operationalization_audit.md` before moving any claim into the ledger.",
        "",
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Human-readable summary of headline Lab 13 metrics.")


def finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    vals = []
    for row in rows:
        val = row.get(key)
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            vals.append(float(val))
    return vals


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    import torch

    args = ctx.args
    if not bench.supports_chat_template(bundle):
        raise RuntimeError("Lab 13 requires an instruct model with a chat template.")

    targets, confounds, data_info = load_items(args)
    emotions = sorted({item.emotion for item in targets})
    print(
        f"[lab13] {len(targets)} target items across {len(emotions)} emotions; "
        f"{len(confounds)} confounds; prompt_set={args.prompt_set}"
    )
    manifest_path = ctx.path("diagnostics", "frozen_data_manifest.json")
    bench.write_json(manifest_path, data_info)
    ctx.register_artifact(manifest_path, "diagnostic", "Frozen Lab 13 data hash, filters, and counts.")

    first_prompt = render_comprehension(bundle, targets[0].content_text)
    bench.run_hook_parity_check(ctx, bundle, first_prompt)
    bench.run_lens_self_check(ctx, bundle, bench.run_with_residual_cache(bundle, first_prompt, add_special_tokens=False))

    split = make_split(targets, args.seed)
    split_rows = [
        {
            "item_id": item.item_id,
            "emotion": item.emotion,
            "cause": item.cause,
            "split": "train" if split[item.item_id] else "eval",
        }
        for item in targets
    ]
    split_path = ctx.path("diagnostics", "split_audit.csv")
    bench.write_csv_with_context(ctx, split_path, split_rows)
    ctx.register_artifact(split_path, "diagnostic", "Deterministic within-emotion train/eval split.")

    features = cache_features(ctx, bundle, targets, confounds)
    n_depths = bundle.anatomy.n_layers + 1
    transfer_rows, best_depth = scan_transfer_depths(targets, features, split, n_depths)
    print(f"[lab13] selected stream depth {best_depth} by cross input/output transfer")

    limits = PROMPT_SET_LIMITS[str(args.prompt_set)]
    sentiment_cap = limits["sentiment_pairs"]
    if args.max_examples > 0:
        sentiment_cap = max(2, min(sentiment_cap, args.max_examples * 2))
    sentiment_direction, n_sentiment_pairs = build_sentiment_direction(bundle, best_depth, sentiment_cap)
    add_controls_at_depth(
        transfer_rows,
        targets,
        features,
        split,
        best_depth,
        args.seed,
        sentiment_direction,
        bundle.anatomy.d_model,
    )

    transfer_path = ctx.path("tables", "emotion_probe_transfer.csv")
    bench.write_csv_with_context(ctx, transfer_path, transfer_rows)
    ctx.register_artifact(
        transfer_path,
        "table",
        "Emotion/neutral decoding and comprehension-generation transfer by depth, with controls at the selected depth.",
    )
    results_path = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results_path, transfer_rows)
    ctx.register_artifact(results_path, "results", "Alias of emotion_probe_transfer.csv for the standard run contract.")

    best_dirs = build_directions_at_depth(targets, features, split, best_depth)
    cross_cause_rows = cross_cause_generalization(targets, features, split, best_depth)
    cross_path = ctx.path("tables", "cross_cause_generalization.csv")
    bench.write_csv_with_context(ctx, cross_path, cross_cause_rows)
    ctx.register_artifact(cross_path, "table", "Leave-one-cause-out transfer for each emotion and source/target pair.")

    cos_rows, labels, vectors = cosine_rows(best_dirs, sentiment_direction)
    cos_path = ctx.path("tables", "emotion_direction_cosines.csv")
    bench.write_csv_with_context(ctx, cos_path, cos_rows)
    ctx.register_artifact(cos_path, "table", "Pairwise cosines among emotion directions and the sentiment control.")

    confound_rows = confound_projection_audit(confounds, features, best_dirs, best_depth)
    confound_path = ctx.path("tables", "confound_projection_audit.csv")
    bench.write_csv_with_context(ctx, confound_path, confound_rows)
    ctx.register_artifact(confound_path, "table", "Projection of surprise/calm/arousal confounds onto target emotion directions.")

    ref_norm_vals = [
        float(features[item.item_id]["comprehension_emotion"][best_depth].norm())
        for item in targets
        if split[item.item_id]
    ]
    ref_norm = safe_fmean(ref_norm_vals, default=1.0)
    steering_per_emotion = limits["steering_per_emotion"]
    generation_rows, steering_rows = run_steering(
        bundle,
        targets,
        best_dirs,
        split,
        best_depth,
        bundle.anatomy.d_model,
        args.seed,
        steering_per_emotion,
        ref_norm,
    )
    generations_path = ctx.path("tables", "steering_generations.csv")
    bench.write_csv_with_context(ctx, generations_path, generation_rows)
    ctx.register_artifact(
        generations_path,
        "table",
        "Generated outputs for baseline/input-direction/random steering with lexicon scores and blank hand-label column.",
    )
    steering_path = ctx.path("tables", "steering_effects.csv")
    bench.write_csv_with_context(ctx, steering_path, steering_rows)
    ctx.register_artifact(steering_path, "table", "Per-emotion steering effect over baseline and random controls.")

    state_payload: dict[str, Any] = {
        "depth": best_depth,
        "injection_layer": best_depth - 1,
        "depth_convention": "bench streams[k]: 0 = embeddings, k = residual after block k",
        "read_site": "chat-templated final prompt token before assistant generation",
        "directions": {
            f"{source}_{emotion}": direction
            for (source, emotion), direction in best_dirs.items()
        },
        "sentiment_lab7_style": sentiment_direction,
        "n_sentiment_pairs": n_sentiment_pairs,
        "method": "paired difference-in-means, train split only, unit normalized",
        "model_id": bundle.anatomy.model_id,
        "d_model": bundle.anatomy.d_model,
        "n_layers": bundle.anatomy.n_layers,
        "evidence": "DECODE artifact unless injected in the steering table.",
    }
    state_path = ctx.path("state", "emotion_directions.pt")
    torch.save(state_payload, state_path)
    ctx.register_artifact(state_path, "tensor", "Comprehension/generation emotion directions and sentiment control at selected depth.")

    if not args.no_plots:
        plot_transfer_matrix(ctx, transfer_rows, emotions, best_depth)
        plot_cosines(ctx, labels, vectors, best_depth)

    real_best = [
        float(r["auc"]) for r in transfer_rows
        if r["depth"] == best_depth
        and r["direction_kind"] == "real"
        and isinstance(r["auc"], (int, float))
        and math.isfinite(float(r["auc"]))
    ]
    cross_best = [
        float(r["auc"]) for r in transfer_rows
        if r["depth"] == best_depth
        and r["direction_kind"] == "real"
        and r["direction_source"] != r["eval_target"]
        and isinstance(r["auc"], (int, float))
        and math.isfinite(float(r["auc"]))
    ]
    comp_gen_cos = [
        cosine(best_dirs[("comprehension", emotion)], best_dirs[("generation", emotion)])
        for emotion in emotions
        if ("comprehension", emotion) in best_dirs and ("generation", emotion) in best_dirs
    ]
    sentiment_cos = [
        abs(float(r["cosine"])) for r in cos_rows
        if "sentiment_lab7_style" in (r["direction_a"], r["direction_b"])
    ]
    steering_over_random = finite_values(steering_rows, "input_over_random_delta")
    metrics = {
        "model_id": bundle.anatomy.model_id,
        "n_target_items": len(targets),
        "n_confound_items": len(confounds),
        "emotions": emotions,
        "best_depth": best_depth,
        "injection_layer": best_depth - 1,
        "n_depths": n_depths,
        "n_sentiment_pairs": n_sentiment_pairs,
        "mean_real_auc_at_best_depth": none_if_nan(rounded(safe_fmean(real_best))),
        "mean_cross_transfer_auc": none_if_nan(rounded(safe_fmean(cross_best))),
        "mean_comp_gen_cosine": none_if_nan(rounded(safe_fmean(comp_gen_cos))),
        "max_abs_sentiment_cosine": none_if_nan(rounded(max(sentiment_cos, default=float("nan")))),
        "steering_input_over_random_delta": none_if_nan(rounded(safe_fmean(steering_over_random))),
        "reference_activation_norm": rounded(ref_norm),
        "steering_dose_fraction": STEERING_DOSE,
        "data": data_info,
    }
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 13 metrics.")

    write_operationalization_audit(ctx, metrics, confound_rows)
    write_run_summary(ctx, metrics)

    run_name = ctx.run_dir.name
    claims = [
        {
            "id": f"{LAB_ID}-C1",
            "tag": "CAUSAL",
            "text": (
                f"Input-derived emotion directions on {bundle.anatomy.model_id} changed generated "
                f"target-affect scores by {metrics['steering_input_over_random_delta']} over a random "
                f"direction baseline at stream depth {best_depth} (dose {STEERING_DOSE} of mean train "
                f"activation norm). This is a scoped activation-addition effect on this prompt family, "
                f"not evidence of felt emotion."
            ),
            "artifact": f"runs/{run_name}/tables/steering_effects.csv",
            "falsifier": (
                "A random or shuffled direction matches the effect, hand labels disagree with the "
                "lexicon scores, or the effect vanishes on held-out causes/prompts."
            ),
        },
        {
            "id": f"{LAB_ID}-C2",
            "tag": "DECODE",
            "text": (
                f"Comprehension and generation emotion directions share geometry on "
                f"{bundle.anatomy.model_id}: mean cross read/write AUC is "
                f"{metrics['mean_cross_transfer_auc']} and mean same-emotion cosine is "
                f"{metrics['mean_comp_gen_cosine']} at depth {best_depth}. This supports an "
                f"input/output affect-geometry bridge only to the extent the sentiment-control "
                f"cosine ({metrics['max_abs_sentiment_cosine']}) and cross-cause audit stay clean."
            ),
            "artifact": f"runs/{run_name}/tables/emotion_probe_transfer.csv",
            "falsifier": (
                "Cross-source AUC falls to chance on new causes, or the directions are collinear "
                "with the Lab-7-style sentiment direction."
            ),
        },
    ]
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)
