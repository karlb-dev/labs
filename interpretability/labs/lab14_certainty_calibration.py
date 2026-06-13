"""Lab 14: Certainty, hedging, and calibration.

Core question:

    Does the model carry an internal answerability/certainty signal that is
    separable from (a) answer-distribution confidence and (b) verbal hedging?

The lab uses a frozen A/B/C/D dataset with answerable and known-unanswerable
items across three families: MCQ, factual QA, and free-form answerability.
Every row can be scored by next-token option logits, so entropy and margins
are measured without free-form grading. Separately, the model is asked for a
verbal confidence word; that output is a SELF-REPORT artifact, not a truth
source.

Evidence labels:
  * DECODE for residual-stream certainty and hedging probes;
  * SELF-REPORT for generated verbal confidence;
  * OBS for entropy/margin behavior.

Optional steering is intentionally left out of the default lab. Downstream
labs need a trustworthy instrument first; a causal certainty edit is a later
extension only after the disagreement matrix and style controls look clean.
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

LAB_ID = "L14"
DATA_FILE = "certainty_calibration_items.csv"
LETTERS = ("A", "B", "C", "D")

PROMPT_SET_FAMILY_CAPS = {"small": 4, "medium": 8, "full": 0}
TRAIN_FRACTION = 0.6
MAX_NEW_TOKENS_CONFIDENCE = 8
ENGINE_MAX_CONCURRENT = 16

SYSTEM_PROMPT = "You are a careful assistant. Follow the requested answer format exactly."

CONFIDENCE_VALUES = {
    "certain": 0.9,
    "likely": 0.7,
    "unsure": 0.4,
    "guess": 0.2,
    "unparsed": 0.5,
}

HEDGE_WORDS = {
    "maybe", "might", "possibly", "probably", "uncertain", "unsure",
    "guess", "could", "perhaps", "cannot", "determine", "unknown",
}


@dataclasses.dataclass
class CalibrationItem:
    item_id: str
    family: str
    topic: str
    answerable: int
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    answer_key: str
    expected_answer: str
    answer_format: str
    confident_statement: str
    hedged_statement: str
    note: str

    @property
    def options(self) -> dict[str, str]:
        return {
            "A": self.option_a,
            "B": self.option_b,
            "C": self.option_c,
            "D": self.option_d,
        }


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


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 3:
        return float("nan")
    ax = [p[0] for p in pairs]
    ay = [p[1] for p in pairs]
    mx, my = statistics.fmean(ax), statistics.fmean(ay)
    num = sum((x - mx) * (y - my) for x, y in pairs)
    dx = math.sqrt(sum((x - mx) ** 2 for x in ax))
    dy = math.sqrt(sum((y - my) ** 2 for y in ay))
    if dx < 1e-12 or dy < 1e-12:
        return float("nan")
    return num / (dx * dy)


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


def item_question_block(item: CalibrationItem) -> str:
    return (
        f"Question: {item.question}\n"
        f"A. {item.option_a}\n"
        f"B. {item.option_b}\n"
        f"C. {item.option_c}\n"
        f"D. {item.option_d}"
    )


def choice_user_message(item: CalibrationItem) -> str:
    return item_question_block(item) + "\nAnswer with only the letter A, B, C, or D."


def confidence_user_message(item: CalibrationItem) -> str:
    return (
        item_question_block(item)
        + "\nHow confident are you that you can answer correctly? "
        + "Reply with exactly one word: certain, likely, unsure, or guess."
    )


def style_user_message(item: CalibrationItem, style: str) -> str:
    if style == "confident":
        instruction = "Answer in one short sentence with a confident, unhedged tone."
    elif style == "hedged":
        instruction = "Answer in one short sentence with a cautious, hedged tone."
    else:
        raise ValueError(f"unknown style {style!r}")
    return item_question_block(item) + "\n" + instruction


def render_chat(bundle: bench.ModelBundle, user_message: str) -> str:
    return bench.apply_chat_template(
        bundle,
        user_message,
        system=SYSTEM_PROMPT,
        add_generation_prompt=True,
    )


def capture_last_streams(bundle: bench.ModelBundle, templated_prompt: str) -> Any:
    cap = bench.run_with_residual_cache(bundle, templated_prompt, add_special_tokens=False)
    return cap.streams[:, -1, :], cap.final_logits_last


def load_items(args: Any) -> tuple[list[CalibrationItem], dict[str, Any]]:
    path = data_path(DATA_FILE)
    raw: list[CalibrationItem] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["answerable"] = int(row["answerable"])
            raw.append(CalibrationItem(**row))

    set_name = args.prompt_set
    if set_name not in PROMPT_SET_FAMILY_CAPS:
        raise ValueError("Lab 14 uses --prompt-set small|medium|full.")
    cap = PROMPT_SET_FAMILY_CAPS[set_name]
    if getattr(args, "max_examples", 0) and args.max_examples > 0:
        cap = args.max_examples

    items: list[CalibrationItem] = []
    by_family: dict[str, list[CalibrationItem]] = defaultdict(list)
    for item in raw:
        by_family[item.family].append(item)
    for family in sorted(by_family):
        rows = by_family[family]
        if cap > 0:
            rows = rows[:cap]
        labels = {item.answerable for item in rows}
        if labels != {0, 1}:
            raise RuntimeError(f"Family {family!r} lost label balance under cap {cap}; use a larger cap.")
        items.extend(rows)

    info = {
        "data_file": DATA_FILE,
        "data_sha256": bench.sha256_file(path),
        "prompt_set": set_name,
        "per_family_cap": cap,
        "n_items": len(items),
        "families": sorted(by_family),
        "counts_by_family": {family: sum(1 for item in items if item.family == family) for family in sorted(by_family)},
        "answerable_counts": {
            "answerable": sum(1 for item in items if item.answerable == 1),
            "unanswerable": sum(1 for item in items if item.answerable == 0),
        },
    }
    return items, info


def make_split(items: Sequence[CalibrationItem], seed: int) -> dict[str, bool]:
    """Family- and label-stratified split."""
    split: dict[str, bool] = {}
    groups: dict[tuple[str, int], list[CalibrationItem]] = defaultdict(list)
    for item in items:
        groups[(item.family, item.answerable)].append(item)
    for (family, label), rows in groups.items():
        ranked = sorted(rows, key=lambda item: stable_hash_int(f"{seed}:{family}:{label}:{item.item_id}"))
        n_train = int(round(TRAIN_FRACTION * len(ranked)))
        if len(ranked) > 1:
            n_train = max(1, min(len(ranked) - 1, n_train))
        else:
            n_train = 1
        train_ids = {item.item_id for item in ranked[:n_train]}
        for item in rows:
            split[item.item_id] = item.item_id in train_ids
    return split


def option_token_ids(bundle: bench.ModelBundle, letter: str) -> list[int]:
    ids: set[int] = set()
    for text in (letter, " " + letter, letter + ".", " " + letter + "."):
        enc = bundle.tokenizer.encode(text, add_special_tokens=False)
        if enc:
            ids.add(int(enc[0]))
    return sorted(ids)


def option_distribution(bundle: bench.ModelBundle, logits: Any) -> dict[str, Any]:
    import torch

    scores = []
    token_map: dict[str, list[int]] = {}
    for letter in LETTERS:
        ids = option_token_ids(bundle, letter)
        token_map[letter] = ids
        if not ids:
            scores.append(torch.tensor(float("-inf")))
        else:
            scores.append(torch.logsumexp(logits[ids], dim=0))
    score_tensor = torch.stack(scores)
    probs = torch.softmax(score_tensor.float(), dim=0)
    entropy = float(-(probs * torch.log2(probs.clamp_min(1e-12))).sum())
    top = torch.topk(score_tensor.float(), k=2)
    chosen = LETTERS[int(top.indices[0])]
    return {
        "scores": {letter: float(score_tensor[i]) for i, letter in enumerate(LETTERS)},
        "probs": {letter: float(probs[i]) for i, letter in enumerate(LETTERS)},
        "entropy_bits": entropy,
        "top_margin": float(top.values[0] - top.values[1]),
        "chosen": chosen,
        "token_map": token_map,
    }


def cache_choice_and_style(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    items: Sequence[CalibrationItem],
) -> tuple[Any, dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    import torch

    choice_vectors = []
    style_vectors: dict[str, dict[str, Any]] = {}
    behavior_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    report_every = max(1, len(items) // 5)
    for i, item in enumerate(items):
        choice_prompt = render_chat(bundle, choice_user_message(item))
        streams, logits = capture_last_streams(bundle, choice_prompt)
        choice_vectors.append(streams)
        dist = option_distribution(bundle, logits)
        correct_prob = dist["probs"][item.answer_key]
        wrong_probs = [p for letter, p in dist["probs"].items() if letter != item.answer_key]
        wrong_scores = [s for letter, s in dist["scores"].items() if letter != item.answer_key]
        correct_score = dist["scores"][item.answer_key]
        correct_margin = correct_score - max(wrong_scores)
        behavior_rows.append({
            "item_id": item.item_id,
            "family": item.family,
            "topic": item.topic,
            "answerable": item.answerable,
            "answer_key": item.answer_key,
            "chosen": dist["chosen"],
            "correct": dist["chosen"] == item.answer_key,
            "correct_prob": rounded(correct_prob),
            "best_wrong_prob": rounded(max(wrong_probs)),
            "correct_margin": rounded(correct_margin),
            "top_margin": rounded(dist["top_margin"]),
            "entropy_bits": rounded(dist["entropy_bits"]),
            "distribution_confidence": rounded(1.0 - dist["entropy_bits"] / math.log2(len(LETTERS))),
            "question": item.question,
        })
        for letter, ids in dist["token_map"].items():
            token_rows.append({
                "item_id": item.item_id,
                "letter": letter,
                "token_ids": " ".join(str(x) for x in ids),
                "n_token_variants": len(ids),
            })

        style_vectors[item.item_id] = {}
        for style in ("confident", "hedged"):
            style_prompt = render_chat(bundle, style_user_message(item, style))
            style_streams, _ = capture_last_streams(bundle, style_prompt)
            style_vectors[item.item_id][style] = style_streams
        if (i + 1) % report_every == 0:
            print(f"[lab14] cached choice/style features for {i + 1}/{len(items)} items")

    token_path = ctx.path("diagnostics", "option_token_audit.csv")
    bench.write_csv_with_context(ctx, token_path, token_rows)
    ctx.register_artifact(token_path, "diagnostic", "Token IDs used to score A/B/C/D next-token options.")
    return torch.stack(choice_vectors), style_vectors, behavior_rows, token_rows


def masks(items: Sequence[CalibrationItem], split: Mapping[str, bool], train: bool) -> list[bool]:
    return [split[item.item_id] == train for item in items]


def labels(items: Sequence[CalibrationItem]) -> list[int]:
    return [int(item.answerable) for item in items]


def mass_mean_direction(X: Any, y: Sequence[int]) -> Any | None:
    import torch

    y_t = torch.tensor([bool(v) for v in y])
    if not bool(y_t.any()) or not bool((~y_t).any()):
        return None
    return unit(X[y_t].mean(dim=0) - X[~y_t].mean(dim=0))


def scores_by_label(X: Any, direction: Any, y: Sequence[int]) -> tuple[list[float], list[float]]:
    scores = (X @ direction).tolist()
    pos = [float(s) for s, label in zip(scores, y) if label == 1]
    neg = [float(s) for s, label in zip(scores, y) if label == 0]
    return pos, neg


def orient_direction(direction: Any, X: Any, y: Sequence[int]) -> Any:
    pos, neg = scores_by_label(X, direction, y)
    if pos and neg and safe_fmean(pos) < safe_fmean(neg):
        return -direction
    return direction


def shuffled_labels(y: Sequence[int], seed: int) -> list[int]:
    order = sorted(range(len(y)), key=lambda i: stable_hash_int(f"{seed}:shuffle:{i}"))
    out = list(y)
    vals = [out[i] for i in order]
    vals = vals[1:] + vals[:1]
    for i, val in zip(order, vals):
        out[i] = val
    return out


def run_certainty_probe_sweep(
    items: Sequence[CalibrationItem],
    choice_feats: Any,
    split: Mapping[str, bool],
    seed: int,
    d_model: int,
) -> tuple[list[dict[str, Any]], int]:
    import torch

    y_all = labels(items)
    train_idx = torch.tensor(masks(items, split, True))
    eval_idx = torch.tensor(masks(items, split, False))
    y_train = [label for label, keep in zip(y_all, train_idx.tolist()) if keep]
    y_eval = [label for label, keep in zip(y_all, eval_idx.tolist()) if keep]
    rows: list[dict[str, Any]] = []
    n_depths = choice_feats.shape[1]
    for depth in range(1, n_depths):
        Xtr = choice_feats[train_idx, depth, :]
        Xev = choice_feats[eval_idx, depth, :]
        real = mass_mean_direction(Xtr, y_train)
        controls: list[tuple[str, Any | None]] = [("real", real)]
        shuf = mass_mean_direction(Xtr, shuffled_labels(y_train, seed + depth))
        controls.append(("shuffled", shuf))
        rnd = orient_direction(random_unit(d_model, seed + 1009 * depth), Xtr, y_train)
        controls.append(("random", rnd))
        for kind, direction in controls:
            if direction is None:
                continue
            pos, neg = scores_by_label(Xev, direction, y_eval)
            rows.append({
                "probe": "certainty_answerability",
                "depth": depth,
                "direction_kind": kind,
                "auc": rounded(auc_from_scores(pos, neg)),
                "selectivity_vs_chance": rounded(auc_from_scores(pos, neg) - 0.5),
                "mean_answerable_projection": rounded(safe_fmean(pos)),
                "mean_unanswerable_projection": rounded(safe_fmean(neg)),
                "n_eval_pos": len(pos),
                "n_eval_neg": len(neg),
            })

    def score_depth(depth: int) -> float:
        vals = [
            float(r["auc"]) - 0.5
            for r in rows
            if r["probe"] == "certainty_answerability" and r["direction_kind"] == "real" and r["depth"] == depth
            and isinstance(r["auc"], (int, float)) and math.isfinite(float(r["auc"]))
        ]
        return safe_fmean(vals, default=-1.0)

    best_depth = max(range(1, n_depths), key=score_depth)
    return rows, best_depth


def run_hedging_probe_sweep(
    items: Sequence[CalibrationItem],
    style_feats: Mapping[str, dict[str, Any]],
    split: Mapping[str, bool],
    seed: int,
    d_model: int,
    n_depths: int,
) -> list[dict[str, Any]]:
    import torch

    train_items = [item for item in items if split[item.item_id]]
    eval_items = [item for item in items if not split[item.item_id]]
    rows: list[dict[str, Any]] = []
    for depth in range(1, n_depths):
        diffs = [
            style_feats[item.item_id]["confident"][depth] - style_feats[item.item_id]["hedged"][depth]
            for item in train_items
        ]
        if not diffs:
            continue
        real = unit(torch.stack(diffs).mean(dim=0))
        rnd = random_unit(d_model, seed + 7919 * depth)
        for kind, direction in (("real", real), ("random", rnd)):
            pos = [float(style_feats[item.item_id]["confident"][depth] @ direction) for item in eval_items]
            neg = [float(style_feats[item.item_id]["hedged"][depth] @ direction) for item in eval_items]
            rows.append({
                "probe": "hedging_style",
                "depth": depth,
                "direction_kind": kind,
                "auc": rounded(auc_from_scores(pos, neg)),
                "selectivity_vs_chance": rounded(auc_from_scores(pos, neg) - 0.5),
                "mean_confident_projection": rounded(safe_fmean(pos)),
                "mean_hedged_projection": rounded(safe_fmean(neg)),
                "n_eval_pos": len(pos),
                "n_eval_neg": len(neg),
            })
    return rows


def direction_at_depth(
    items: Sequence[CalibrationItem],
    choice_feats: Any,
    split: Mapping[str, bool],
    depth: int,
) -> Any:
    import torch

    train_idx = torch.tensor(masks(items, split, True))
    y_train = [item.answerable for item in items if split[item.item_id]]
    direction = mass_mean_direction(choice_feats[train_idx, depth, :], y_train)
    if direction is None:
        raise RuntimeError("Could not build certainty direction; train split lost a label.")
    return direction


def hedging_direction_at_depth(
    items: Sequence[CalibrationItem],
    style_feats: Mapping[str, dict[str, Any]],
    split: Mapping[str, bool],
    depth: int,
) -> Any:
    import torch

    diffs = [
        style_feats[item.item_id]["confident"][depth] - style_feats[item.item_id]["hedged"][depth]
        for item in items
        if split[item.item_id]
    ]
    if not diffs:
        raise RuntimeError("Could not build hedging direction; empty train split.")
    return unit(torch.stack(diffs).mean(dim=0))


def family_heldout_rows(
    items: Sequence[CalibrationItem],
    choice_feats: Any,
    depth: int,
) -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    families = sorted({item.family for item in items})
    for family in families:
        train_idx = torch.tensor([item.family != family for item in items])
        eval_idx = torch.tensor([item.family == family for item in items])
        y_train = [item.answerable for item in items if item.family != family]
        y_eval = [item.answerable for item in items if item.family == family]
        direction = mass_mean_direction(choice_feats[train_idx, depth, :], y_train)
        if direction is None:
            continue
        pos, neg = scores_by_label(choice_feats[eval_idx, depth, :], direction, y_eval)
        rows.append({
            "held_out_family": family,
            "depth": depth,
            "auc": rounded(auc_from_scores(pos, neg)),
            "selectivity_vs_chance": rounded(auc_from_scores(pos, neg) - 0.5),
            "n_eval_pos": len(pos),
            "n_eval_neg": len(neg),
            "n_train": int(train_idx.sum()),
        })
    return rows


def parse_confidence(text: str) -> tuple[str, float]:
    low = text.lower()
    for marker in ("certain", "likely", "unsure", "guess"):
        if re.search(rf"\b{marker}\b", low):
            return marker, CONFIDENCE_VALUES[marker]
    return "unparsed", CONFIDENCE_VALUES["unparsed"]


def hedge_count(text: str) -> int:
    toks = re.findall(r"[A-Za-z']+", text.lower())
    return sum(1 for tok in toks if tok in HEDGE_WORDS)


def generate_confidence_reports(
    bundle: bench.ModelBundle,
    items: Sequence[CalibrationItem],
) -> list[dict[str, Any]]:
    prompts = [render_chat(bundle, confidence_user_message(item)) for item in items]
    outs = bench.generate_continuous(
        bundle,
        prompts,
        MAX_NEW_TOKENS_CONFIDENCE,
        max_concurrent=ENGINE_MAX_CONCURRENT,
        skip_special_tokens=True,
        progress_label="lab14-confidence",
    )
    rows: list[dict[str, Any]] = []
    for item, text in zip(items, outs):
        label, score = parse_confidence(text)
        rows.append({
            "item_id": item.item_id,
            "family": item.family,
            "answerable": item.answerable,
            "verbal_confidence_label": label,
            "verbal_confidence_score": score,
            "hedge_marker_count": hedge_count(text),
            "confidence_text": text,
        })
    return rows


def median(vals: Sequence[float], default: float = 0.0) -> float:
    finite = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not finite:
        return default
    mid = len(finite) // 2
    if len(finite) % 2:
        return finite[mid]
    return 0.5 * (finite[mid - 1] + finite[mid])


def build_signal_rows(
    items: Sequence[CalibrationItem],
    choice_feats: Any,
    behavior_rows: Sequence[Mapping[str, Any]],
    confidence_rows: Sequence[Mapping[str, Any]],
    certainty_direction: Any,
    hedging_direction: Any,
    split: Mapping[str, bool],
    depth: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    behavior = {r["item_id"]: r for r in behavior_rows}
    confidence = {r["item_id"]: r for r in confidence_rows}
    raw_rows: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        internal = float(choice_feats[i, depth, :] @ certainty_direction)
        hedging_proj = float(choice_feats[i, depth, :] @ hedging_direction)
        b = behavior[item.item_id]
        c = confidence[item.item_id]
        raw_rows.append({
            "item_id": item.item_id,
            "family": item.family,
            "answerable": item.answerable,
            "split": "train" if split[item.item_id] else "eval",
            "correct": bool(b["correct"]),
            "internal_certainty_projection": internal,
            "hedging_style_projection": hedging_proj,
            "distribution_confidence": float(b["distribution_confidence"]),
            "correct_margin": float(b["correct_margin"]),
            "entropy_bits": float(b["entropy_bits"]),
            "verbal_confidence_score": float(c["verbal_confidence_score"]),
            "verbal_confidence_label": c["verbal_confidence_label"],
            "hedge_marker_count": int(c["hedge_marker_count"]),
        })

    train_rows = [r for r in raw_rows if r["split"] == "train"]
    thresholds = {
        "internal": median([r["internal_certainty_projection"] for r in train_rows]),
        "distribution": median([r["distribution_confidence"] for r in train_rows]),
        "verbal": 0.6,
    }
    for row in raw_rows:
        row["internal_signal"] = "high" if row["internal_certainty_projection"] >= thresholds["internal"] else "low"
        row["distribution_signal"] = "high" if row["distribution_confidence"] >= thresholds["distribution"] else "low"
        row["verbal_signal"] = "high" if row["verbal_confidence_score"] >= thresholds["verbal"] else "low"
        row["three_way_agreement"] = (
            row["internal_signal"] == row["distribution_signal"] == row["verbal_signal"]
        )

    matrix_rows: list[dict[str, Any]] = []
    for internal in ("low", "high"):
        for distribution in ("low", "high"):
            for verbal in ("low", "high"):
                sub = [
                    r for r in raw_rows
                    if r["internal_signal"] == internal
                    and r["distribution_signal"] == distribution
                    and r["verbal_signal"] == verbal
                ]
                matrix_rows.append({
                    "internal_signal": internal,
                    "distribution_signal": distribution,
                    "verbal_signal": verbal,
                    "n": len(sub),
                    "accuracy": rounded(safe_fmean([1.0 if r["correct"] else 0.0 for r in sub])),
                    "answerable_rate": rounded(safe_fmean([float(r["answerable"]) for r in sub])),
                    "mean_internal_projection": rounded(safe_fmean([r["internal_certainty_projection"] for r in sub])),
                    "mean_distribution_confidence": rounded(safe_fmean([r["distribution_confidence"] for r in sub])),
                    "mean_verbal_confidence": rounded(safe_fmean([r["verbal_confidence_score"] for r in sub])),
                })
    return raw_rows, matrix_rows, thresholds


def reliability_bins(signal_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    total = len(signal_rows)
    ece = 0.0
    for label in ("guess", "unsure", "likely", "certain", "unparsed"):
        sub = [r for r in signal_rows if r["verbal_confidence_label"] == label]
        if not sub:
            continue
        conf = safe_fmean([float(r["verbal_confidence_score"]) for r in sub])
        acc = safe_fmean([1.0 if r["correct"] else 0.0 for r in sub])
        ece += (len(sub) / max(1, total)) * abs(acc - conf)
        rows.append({
            "verbal_confidence_label": label,
            "n": len(sub),
            "mean_reported_confidence": rounded(conf),
            "accuracy": rounded(acc),
            "abs_gap": rounded(abs(acc - conf)),
        })
    return rows, ece


def plot_probe_by_layer(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]], best_depth: int) -> None:
    fig, ax = bench.new_figure(figsize=(8.8, 5.2))
    series = [
        ("certainty_answerability", "real", "tab:blue", "certainty"),
        ("certainty_answerability", "shuffled", "tab:blue", "certainty shuffled"),
        ("certainty_answerability", "random", "tab:gray", "certainty random"),
        ("hedging_style", "real", "tab:orange", "hedging style"),
    ]
    for probe, kind, color, label in series:
        pts = sorted(
            (int(r["depth"]), float(r["auc"]))
            for r in rows
            if r["probe"] == probe
            and r["direction_kind"] == kind
            and isinstance(r.get("auc"), (int, float))
        )
        if not pts:
            continue
        linestyle = ":" if kind != "real" else "-"
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linestyle=linestyle, marker="o", label=label)
    ax.axhline(0.5, color="black", linewidth=0.8, alpha=0.5)
    ax.axvline(best_depth, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("residual-stream depth")
    ax.set_ylabel("AUC")
    ax.set_title("Certainty and hedging probes by depth")
    bench.style_ax(ax)
    bench.save_figure(ctx, fig, "certainty_probe_by_layer.png", "Certainty/answerability and hedging-style AUC by residual depth.")


def plot_reliability(ctx: bench.RunContext, bins: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(6.4, 5.4))
    xs = [float(r["mean_reported_confidence"]) for r in bins]
    ys = [float(r["accuracy"]) for r in bins]
    labels = [str(r["verbal_confidence_label"]) for r in bins]
    ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1.0, label="perfect calibration")
    ax.scatter(xs, ys, s=[35 + 10 * int(r["n"]) for r in bins], color="tab:green")
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("reported verbal confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title("Verbal confidence reliability")
    bench.style_ax(ax)
    bench.save_figure(ctx, fig, "reliability_diagram.png", "Reliability diagram for generated verbal confidence labels.")


def plot_disagreement_matrix(ctx: bench.RunContext, matrix_rows: Sequence[Mapping[str, Any]]) -> None:
    import numpy as np

    row_labels = [f"I:{i} / D:{d}" for i in ("low", "high") for d in ("low", "high")]
    col_labels = ["V:low", "V:high"]
    grid = np.zeros((len(row_labels), len(col_labels)))
    acc = np.full_like(grid, np.nan, dtype=float)
    for r in matrix_rows:
        ri = row_labels.index(f"I:{r['internal_signal']} / D:{r['distribution_signal']}")
        ci = col_labels.index(f"V:{r['verbal_signal']}")
        grid[ri, ci] = int(r["n"])
        if int(r["n"]):
            acc[ri, ci] = float(r["accuracy"])
    fig, ax = bench.new_figure(figsize=(6.8, 5.4))
    im = ax.imshow(grid, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Three-way confidence disagreement counts")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            text = f"n={int(grid[i, j])}"
            if math.isfinite(float(acc[i, j])):
                text += f"\nacc={acc[i, j]:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.035, label="count")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "confidence_disagreement_matrix.png", "Internal/distribution/verbal confidence disagreement matrix.")


def write_operationalization_audit(
    ctx: bench.RunContext,
    metrics: Mapping[str, Any],
) -> None:
    lines = [
        "# Lab 14 Operationalization Audit",
        "",
        "## What Was Measured",
        "",
        "The lab measures answerability/certainty as a controlled A/B/C/D task: answerable questions have a checkable option, and known-unanswerable questions use `D` as the correct response. The internal direction is a residual-stream probe for that label, not a direct meter of subjective confidence.",
        "",
        "## Cheap Explanations",
        "",
        "- Answer format: every item uses the same A/B/C/D frame.",
        "- Family/topic: the family-held-out table asks whether the direction transfers across MCQ, factual QA, and free-form answerability.",
        "- Hedging style: a separate confident-vs-hedged style direction is trained and saved; a certainty claim weakens if this explains the same projections.",
        "- Distribution confidence: entropy and margin are logged separately so low entropy cannot masquerade as internal certainty without being named.",
        "- Self-report: generated confidence is tagged SELF-REPORT and checked against correctness and internal projection; it is not treated as ground truth.",
        "",
        "## Headline Numbers",
        "",
        f"- Best certainty depth: {metrics.get('best_depth')}",
        f"- Certainty AUC at best depth: {metrics.get('certainty_auc_best_depth')}",
        f"- Mean family-held-out AUC: {metrics.get('mean_family_heldout_auc')}",
        f"- Internal vs distribution correlation: {metrics.get('internal_distribution_correlation')}",
        f"- Internal vs verbal confidence correlation: {metrics.get('internal_verbal_correlation')}",
        f"- Verbal confidence ECE: {metrics.get('verbal_confidence_ece')}",
        "",
        "## Allowed Claim",
        "",
        "An internal uncertainty claim is allowed only when the direction predicts answerability beyond family/topic, does not collapse into the hedging-style direction, and improves on merely reading entropy or verbal confidence. Otherwise the honest claim is about answer format, difficulty, or style.",
        "",
    ]
    path = ctx.path("operationalization_audit.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "audit", "Operationalization limits and cheap-explanation audit for Lab 14.")


def write_run_summary(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    lines = [
        f"# Lab 14 Run Summary: {ctx.run_dir.name}",
        "",
        f"- Model: `{metrics.get('model_id')}`",
        f"- Items: {metrics.get('n_items')}",
        f"- Best certainty depth: {metrics.get('best_depth')}",
        f"- Certainty AUC at best depth: {metrics.get('certainty_auc_best_depth')}",
        f"- Hedging AUC at best depth: {metrics.get('hedging_auc_best_depth')}",
        f"- Mean family-held-out AUC: {metrics.get('mean_family_heldout_auc')}",
        f"- Internal/distribution correlation: {metrics.get('internal_distribution_correlation')}",
        f"- Internal/verbal correlation: {metrics.get('internal_verbal_correlation')}",
        f"- Verbal confidence ECE: {metrics.get('verbal_confidence_ece')}",
        "",
        "Use `confidence_disagreement_matrix.csv` to find one concrete case where signals disagree before writing the SELF-REPORT claim.",
        "",
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Human-readable summary of headline Lab 14 metrics.")


def finite_vals(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    out = []
    for row in rows:
        val = row.get(key)
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            out.append(float(val))
    return out


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    import torch

    args = ctx.args
    if not bench.supports_chat_template(bundle):
        raise RuntimeError("Lab 14 requires an instruct model with a chat template.")

    items, data_info = load_items(args)
    print(f"[lab14] {len(items)} items across {len(data_info['families'])} families; prompt_set={args.prompt_set}")
    manifest_path = ctx.path("diagnostics", "frozen_data_manifest.json")
    bench.write_json(manifest_path, data_info)
    ctx.register_artifact(manifest_path, "diagnostic", "Frozen Lab 14 data hash, filters, and counts.")

    first_prompt = render_chat(bundle, choice_user_message(items[0]))
    bench.run_hook_parity_check(ctx, bundle, first_prompt)
    bench.run_lens_self_check(ctx, bundle, bench.run_with_residual_cache(bundle, first_prompt, add_special_tokens=False))

    split = make_split(items, args.seed)
    split_path = ctx.path("diagnostics", "split_audit.csv")
    bench.write_csv_with_context(ctx, split_path, [
        {
            "item_id": item.item_id,
            "family": item.family,
            "answerable": item.answerable,
            "split": "train" if split[item.item_id] else "eval",
        }
        for item in items
    ])
    ctx.register_artifact(split_path, "diagnostic", "Family- and label-stratified train/eval split.")

    choice_feats, style_feats, behavior_rows, _ = cache_choice_and_style(ctx, bundle, items)
    behavior_path = ctx.path("tables", "answer_distribution_readout.csv")
    bench.write_csv_with_context(ctx, behavior_path, behavior_rows)
    ctx.register_artifact(behavior_path, "table", "A/B/C/D option probabilities, entropy, margins, chosen answer, and correctness.")

    certainty_rows, best_depth = run_certainty_probe_sweep(
        items, choice_feats, split, args.seed, bundle.anatomy.d_model
    )
    hedging_rows = run_hedging_probe_sweep(
        items, style_feats, split, args.seed, bundle.anatomy.d_model, choice_feats.shape[1]
    )
    probe_rows = certainty_rows + hedging_rows
    probe_path = ctx.path("tables", "probe_report.csv")
    bench.write_csv_with_context(ctx, probe_path, probe_rows)
    ctx.register_artifact(probe_path, "table", "Certainty/answerability and hedging-style probe sweep with controls.")
    results_path = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results_path, probe_rows)
    ctx.register_artifact(results_path, "results", "Alias of probe_report.csv for the standard run contract.")
    print(f"[lab14] selected certainty depth {best_depth}")

    certainty_direction = direction_at_depth(items, choice_feats, split, best_depth)
    hedging_direction = hedging_direction_at_depth(items, style_feats, split, best_depth)
    family_rows = family_heldout_rows(items, choice_feats, best_depth)
    family_path = ctx.path("tables", "family_heldout_generalization.csv")
    bench.write_csv_with_context(ctx, family_path, family_rows)
    ctx.register_artifact(family_path, "table", "Certainty direction trained with one family held out and evaluated on that family.")

    confidence_rows = generate_confidence_reports(bundle, items)
    confidence_path = ctx.path("tables", "verbal_confidence_reports.csv")
    bench.write_csv_with_context(ctx, confidence_path, confidence_rows)
    ctx.register_artifact(confidence_path, "table", "Generated verbal confidence self-reports and parsed confidence labels.")

    signal_rows, matrix_rows, thresholds = build_signal_rows(
        items,
        choice_feats,
        behavior_rows,
        confidence_rows,
        certainty_direction,
        hedging_direction,
        split,
        best_depth,
    )
    signal_path = ctx.path("tables", "confidence_signal_table.csv")
    bench.write_csv_with_context(ctx, signal_path, signal_rows)
    ctx.register_artifact(signal_path, "table", "Per-item internal, distributional, and verbal confidence signals.")
    matrix_path = ctx.path("tables", "confidence_disagreement_matrix.csv")
    bench.write_csv_with_context(ctx, matrix_path, matrix_rows)
    ctx.register_artifact(matrix_path, "table", "Three-way internal/distribution/verbal confidence disagreement matrix.")

    reliability_rows, ece = reliability_bins(signal_rows)
    reliability_path = ctx.path("tables", "reliability_curve.csv")
    bench.write_csv_with_context(ctx, reliability_path, reliability_rows)
    ctx.register_artifact(reliability_path, "table", "Reliability bins for generated verbal confidence labels.")

    state_common = {
        "depth": best_depth,
        "depth_convention": "bench streams[k]: 0 = embeddings, k = residual after block k",
        "read_site": "chat-templated final prompt token before assistant generation",
        "model_id": bundle.anatomy.model_id,
        "d_model": bundle.anatomy.d_model,
        "n_layers": bundle.anatomy.n_layers,
    }
    certainty_state = {
        **state_common,
        "direction": certainty_direction,
        "label": "answerable (1) minus known-unanswerable (0)",
        "method": "mass-mean direction on train split only",
    }
    certainty_path = ctx.path("state", "certainty_direction.pt")
    torch.save(certainty_state, certainty_path)
    ctx.register_artifact(certainty_path, "tensor", "Train-split mass-mean answerability/certainty direction.")

    hedging_state = {
        **state_common,
        "direction": hedging_direction,
        "label": "confident style minus hedged style",
        "method": "paired style-instruction direction on train split only",
    }
    hedging_path = ctx.path("state", "hedging_direction.pt")
    torch.save(hedging_state, hedging_path)
    ctx.register_artifact(hedging_path, "tensor", "Train-split confident-vs-hedged style direction.")

    if not args.no_plots:
        plot_probe_by_layer(ctx, probe_rows, best_depth)
        plot_reliability(ctx, reliability_rows)
        plot_disagreement_matrix(ctx, matrix_rows)

    certainty_best = [
        float(r["auc"]) for r in probe_rows
        if r["probe"] == "certainty_answerability"
        and r["direction_kind"] == "real"
        and r["depth"] == best_depth
        and isinstance(r["auc"], (int, float))
    ]
    shuffled_best = [
        float(r["auc"]) for r in probe_rows
        if r["probe"] == "certainty_answerability"
        and r["direction_kind"] == "shuffled"
        and r["depth"] == best_depth
        and isinstance(r["auc"], (int, float))
    ]
    hedging_best = [
        float(r["auc"]) for r in probe_rows
        if r["probe"] == "hedging_style"
        and r["direction_kind"] == "real"
        and r["depth"] == best_depth
        and isinstance(r["auc"], (int, float))
    ]
    internal_vals = finite_vals(signal_rows, "internal_certainty_projection")
    dist_vals = finite_vals(signal_rows, "distribution_confidence")
    verbal_vals = finite_vals(signal_rows, "verbal_confidence_score")
    family_aucs = finite_vals(family_rows, "auc")
    metrics = {
        "model_id": bundle.anatomy.model_id,
        "n_items": len(items),
        "families": data_info["families"],
        "best_depth": best_depth,
        "certainty_auc_best_depth": none_if_nan(rounded(safe_fmean(certainty_best))),
        "certainty_shuffled_auc_best_depth": none_if_nan(rounded(safe_fmean(shuffled_best))),
        "certainty_selectivity_best_depth": none_if_nan(rounded(safe_fmean(certainty_best) - safe_fmean(shuffled_best))),
        "hedging_auc_best_depth": none_if_nan(rounded(safe_fmean(hedging_best))),
        "mean_family_heldout_auc": none_if_nan(rounded(safe_fmean(family_aucs))),
        "internal_distribution_correlation": none_if_nan(rounded(pearson(internal_vals, dist_vals))),
        "internal_verbal_correlation": none_if_nan(rounded(pearson(internal_vals, verbal_vals))),
        "verbal_confidence_ece": rounded(ece),
        "signal_thresholds": {k: rounded(v) for k, v in thresholds.items()},
        "data": data_info,
    }
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 14 metrics.")

    write_operationalization_audit(ctx, metrics)
    write_run_summary(ctx, metrics)

    run_name = ctx.run_dir.name
    claims = [
        {
            "id": f"{LAB_ID}-C1",
            "tag": "DECODE",
            "text": (
                f"A residual-stream certainty direction on {bundle.anatomy.model_id} predicts "
                f"answerability at depth {best_depth} with AUC {metrics['certainty_auc_best_depth']} "
                f"and selectivity {metrics['certainty_selectivity_best_depth']} over shuffled labels; "
                f"mean family-held-out AUC is {metrics['mean_family_heldout_auc']}. This is a "
                f"controlled answerability signal, not a general proof of calibrated confidence."
            ),
            "artifact": f"runs/{run_name}/tables/family_heldout_generalization.csv",
            "falsifier": (
                "Family-held-out AUC collapses, or the hedging-style direction explains the same "
                "separation under matched answer-format controls."
            ),
        },
        {
            "id": f"{LAB_ID}-C2",
            "tag": "SELF-REPORT",
            "text": (
                f"Generated verbal confidence on {bundle.anatomy.model_id} has ECE "
                f"{metrics['verbal_confidence_ece']} on this fixed-choice set; its correlation with "
                f"the internal certainty projection is {metrics['internal_verbal_correlation']} "
                f"(internal-vs-distribution correlation {metrics['internal_distribution_correlation']}). "
                f"Use the disagreement matrix before treating self-report as calibrated."
            ),
            "artifact": f"runs/{run_name}/tables/confidence_disagreement_matrix.csv",
            "falsifier": (
                "A held-out family shows verbal confidence tracking hedging words or option entropy "
                "while diverging from correctness and the internal projection."
            ),
        },
    ]
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)
