"""Lab 14: Certainty, hedging, and calibration.

Core question
=============

Does an instruct model carry an internal answerability / certainty signal that
is separable from (a) its next-token answer distribution and (b) the hedging
or confidence words it emits?

This lab deliberately keeps three instruments in separate jars:

* ``DECODE``: residual-stream directions trained from controlled
  answerable-vs-known-unanswerable items and from paired confident-vs-hedged
  wording variants;
* ``OBS``: the model's A/B/C/D option distribution, entropy, margins, and
  correctness under a fixed next-token scoring frame;
* ``SELF-REPORT``: generated verbal confidence words under frozen decoding.

The product is a reusable certainty instrument for later multi-turn and
self-report labs. The instrument is only usable downstream if it survives the
confound audit: family-held-out transfer, shuffled/random controls, length and
answer-key baselines, and hedging-style leakage checks. Optional certainty
steering is intentionally left out of the default lab; first make the gauge,
then decide whether to poke the tank.
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import hashlib
import json
import math
import pathlib
import re
import statistics
from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

import interp_bench as bench

LAB_ID = "L14"
DATA_FILE = "certainty_calibration_items.csv"
MANIFEST_FILE = "MANIFEST.json"
LETTERS = ("A", "B", "C", "D")

# ``--max-examples`` is interpreted as a PER-FAMILY cap for this lab. Tier A's
# registry default is 4, so the smoke run keeps roughly two answerable and two
# unanswerable items per family.
PROMPT_SET_FAMILY_CAPS = {"small": 4, "medium": 8, "full": 0}
TRAIN_FRACTION = 0.6
N_SHUFFLES = 5
N_RANDOM_DIRECTIONS = 5
N_RELIABILITY_BINS = 4
MAX_NEW_TOKENS_CONFIDENCE = 8
ENGINE_MAX_CONCURRENT = 16

SYSTEM_PROMPT = "You are a careful assistant. Follow the requested answer format exactly."

# These are deliberately coarse because verbal confidence is a self-report
# instrument, not an oracle. The parse table records the raw text so students
# can hand-audit ambiguous cases.
CONFIDENCE_VALUES = {
    "certain": 0.90,
    "likely": 0.70,
    "unsure": 0.40,
    "guess": 0.20,
    "unparsed": 0.50,
}
CONFIDENCE_ORDER = ("guess", "unsure", "likely", "certain", "unparsed")

HEDGE_WORDS = {
    "maybe", "might", "possibly", "probably", "uncertain", "unsure",
    "guess", "could", "perhaps", "cannot", "can't", "determine",
    "unknown", "unclear", "likely", "seems", "appears",
}

UNANSWERABLE_MARKERS = {
    "cannot", "can't", "unknown", "not enough", "insufficient", "determine",
    "private", "future", "hidden", "unknowable", "not possible",
}


@dataclasses.dataclass(frozen=True)
class CalibrationItem:
    """One fixed-choice answerability/calibration item."""

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
    expected_answer: str = ""
    answer_format: str = "abcd"
    confident_statement: str = ""
    hedged_statement: str = ""
    note: str = ""

    @property
    def options(self) -> dict[str, str]:
        return {
            "A": self.option_a,
            "B": self.option_b,
            "C": self.option_c,
            "D": self.option_d,
        }

    def option_text(self, letter: str) -> str:
        return self.options[letter]


# ---------------------------------------------------------------------------
# Small math / data utilities
# ---------------------------------------------------------------------------


def stable_hash_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        out = float(x)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def safe_fmean(vals: Sequence[float], default: float = float("nan")) -> float:
    finite = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return float(statistics.fmean(finite)) if finite else default


def safe_stdev(vals: Sequence[float], default: float = 0.0) -> float:
    finite = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if len(finite) < 2:
        return default
    return float(statistics.stdev(finite))


def median(vals: Sequence[float], default: float = 0.0) -> float:
    finite = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not finite:
        return default
    mid = len(finite) // 2
    if len(finite) % 2:
        return finite[mid]
    return 0.5 * (finite[mid - 1] + finite[mid])


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
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
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


def finite_vals(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        val = row.get(key)
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            out.append(float(val))
    return out


def orientation_from_train(scores: Sequence[float], labels: Sequence[int]) -> float:
    pos = [float(s) for s, y in zip(scores, labels) if int(y) == 1]
    neg = [float(s) for s, y in zip(scores, labels) if int(y) == 0]
    if pos and neg and safe_fmean(pos) < safe_fmean(neg):
        return -1.0
    return 1.0


def empirical_percentile(value: float, reference: Sequence[float]) -> float:
    finite = sorted(float(v) for v in reference if math.isfinite(float(v)))
    if not finite:
        return 0.5
    less = sum(1 for v in finite if v < value)
    equal = sum(1 for v in finite if v == value)
    return (less + 0.5 * equal) / len(finite)


# ---------------------------------------------------------------------------
# Dataset loading and selection
# ---------------------------------------------------------------------------


def fallback_items() -> list[CalibrationItem]:
    """Tiny deterministic smoke dataset used only when the frozen CSV is absent.

    The fallback is a plumbing harness, not science data. It has the same schema
    and label/family shape as the frozen dataset so Tier A can test the whole
    artifact path before the real CSV lands in a checkout.
    """

    rows = [
        # MCQ
        ("mcq_math_known", "mcq", "math", 1, "What is 2 + 2?", "3", "4", "5", "Cannot be determined", "B", "4"),
        ("mcq_geo_known", "mcq", "geography", 1, "Which city is the capital of Japan?", "Tokyo", "Kyoto", "Osaka", "Cannot be determined", "A", "Tokyo"),
        ("mcq_private_unans", "mcq", "private", 0, "What number am I privately thinking of right now?", "7", "13", "42", "Cannot be determined", "D", "Cannot be determined"),
        ("mcq_future_unans", "mcq", "future", 0, "Which exact word will a random user type next Tuesday at 9:03 AM?", "apple", "chair", "river", "Cannot be determined", "D", "Cannot be determined"),
        # Factual QA
        ("fact_planet_known", "factual_qa", "science", 1, "Which planet is known as the Red Planet?", "Venus", "Mars", "Jupiter", "Cannot be determined", "B", "Mars"),
        ("fact_author_known", "factual_qa", "literature", 1, "Who wrote Pride and Prejudice?", "Jane Austen", "Mary Shelley", "George Eliot", "Cannot be determined", "A", "Jane Austen"),
        ("fact_hidden_unans", "factual_qa", "hidden", 0, "What password did the course instructor choose for a private account?", "atlas", "lantern", "comet", "Cannot be determined", "D", "Cannot be determined"),
        ("fact_future_unans", "factual_qa", "future", 0, "What will the exact closing price of a stock be on a future date?", "$101.10", "$104.25", "$99.80", "Cannot be determined", "D", "Cannot be determined"),
        # Freeform answerability rendered as A/B/C/D
        ("freeform_color_known", "freeform_answerability", "commonsense", 1, "What color is a typical clear daytime sky?", "Blue", "Green", "Black", "Cannot be determined", "A", "Blue"),
        ("freeform_boiling_known", "freeform_answerability", "science", 1, "At sea level, what happens to water near 100 degrees Celsius?", "It freezes", "It boils", "It becomes metal", "Cannot be determined", "B", "It boils"),
        ("freeform_secret_unans", "freeform_answerability", "private", 0, "What did an unnamed person write in their locked diary last night?", "A poem", "A shopping list", "A phone number", "Cannot be determined", "D", "Cannot be determined"),
        ("freeform_random_unans", "freeform_answerability", "random", 0, "What number will a fair die show on a future roll?", "1", "3", "6", "Cannot be determined", "D", "Cannot be determined"),
    ]
    out: list[CalibrationItem] = []
    for item_id, family, topic, answerable, question, a, b, c, d, key, expected in rows:
        if answerable:
            confident = f"The answer is {expected}."
            hedged = f"The answer is probably {expected}, but I am not completely sure."
        else:
            confident = "The answer cannot be determined from the information given."
            hedged = "It may not be possible to determine the answer from the information given."
        out.append(CalibrationItem(
            item_id=item_id,
            family=family,
            topic=topic,
            answerable=answerable,
            question=question,
            option_a=a,
            option_b=b,
            option_c=c,
            option_d=d,
            answer_key=key,
            expected_answer=expected,
            answer_format="fallback_abcd",
            confident_statement=confident,
            hedged_statement=hedged,
            note="built-in Tier A smoke fallback; not science data",
        ))
    return out


def data_path(name: str) -> pathlib.Path:
    return bench.COURSE_ROOT / "data" / name


def read_items_csv(path: pathlib.Path) -> list[CalibrationItem]:
    required = {
        "item_id", "family", "topic", "answerable", "question", "option_a",
        "option_b", "option_c", "option_d", "answer_key",
    }
    rows: list[CalibrationItem] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"{path} is missing required columns: {sorted(missing)}")
        for raw in reader:
            answerable = int(raw.get("answerable", 0))
            key = str(raw.get("answer_key", "")).strip().upper()
            if key not in LETTERS:
                raise RuntimeError(f"Item {raw.get('item_id')} has invalid answer_key {key!r}")
            rows.append(CalibrationItem(
                item_id=str(raw.get("item_id", "")).strip(),
                family=str(raw.get("family", "")).strip(),
                topic=str(raw.get("topic", "")).strip(),
                answerable=answerable,
                question=str(raw.get("question", "")).strip(),
                option_a=str(raw.get("option_a", "")).strip(),
                option_b=str(raw.get("option_b", "")).strip(),
                option_c=str(raw.get("option_c", "")).strip(),
                option_d=str(raw.get("option_d", "")).strip(),
                answer_key=key,
                expected_answer=str(raw.get("expected_answer", "")).strip(),
                answer_format=str(raw.get("answer_format", "")).strip(),
                confident_statement=str(raw.get("confident_statement", "")).strip(),
                hedged_statement=str(raw.get("hedged_statement", "")).strip(),
                note=str(raw.get("note", "")).strip(),
            ))
    return rows


def read_items_json(path: pathlib.Path) -> list[CalibrationItem]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("items", [])
    if not isinstance(data, list):
        raise RuntimeError(f"{path} must contain a list of item objects or {{'items': [...]}}")
    rows: list[CalibrationItem] = []
    for raw in data:
        if not isinstance(raw, dict):
            raise RuntimeError(f"Bad item in {path}: expected object, got {type(raw).__name__}")
        rows.append(CalibrationItem(
            item_id=str(raw.get("item_id", "")).strip(),
            family=str(raw.get("family", "")).strip(),
            topic=str(raw.get("topic", "")).strip(),
            answerable=int(raw.get("answerable", 0)),
            question=str(raw.get("question", "")).strip(),
            option_a=str(raw.get("option_a", "")).strip(),
            option_b=str(raw.get("option_b", "")).strip(),
            option_c=str(raw.get("option_c", "")).strip(),
            option_d=str(raw.get("option_d", "")).strip(),
            answer_key=str(raw.get("answer_key", "")).strip().upper(),
            expected_answer=str(raw.get("expected_answer", "")).strip(),
            answer_format=str(raw.get("answer_format", "")).strip(),
            confident_statement=str(raw.get("confident_statement", "")).strip(),
            hedged_statement=str(raw.get("hedged_statement", "")).strip(),
            note=str(raw.get("note", "")).strip(),
        ))
    return rows


def expected_manifest_hash(filename: str) -> tuple[str | None, str]:
    manifest_path = data_path(MANIFEST_FILE)
    if not manifest_path.exists():
        return None, "manifest_missing"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"manifest_unreadable:{exc}"
    candidates = [
        manifest.get(filename) if isinstance(manifest, dict) else None,
        manifest.get("files", {}).get(filename) if isinstance(manifest.get("files"), dict) else None,
    ]
    for entry in candidates:
        if isinstance(entry, str):
            return entry, "manifest_found_string"
        if isinstance(entry, dict):
            for key in ("sha256", "hash", "digest"):
                if entry.get(key):
                    return str(entry[key]), f"manifest_found_{key}"
    return None, "manifest_entry_missing"


def select_items(raw: Sequence[CalibrationItem], cap_per_family: int, seed: int) -> list[CalibrationItem]:
    by_family: dict[str, list[CalibrationItem]] = defaultdict(list)
    for item in raw:
        if not item.item_id or not item.family or not item.question:
            raise RuntimeError(f"Bad Lab 14 item with missing id/family/question: {item}")
        if int(item.answerable) not in (0, 1):
            raise RuntimeError(f"Item {item.item_id} has answerable={item.answerable}; expected 0 or 1.")
        if item.answer_key not in LETTERS:
            raise RuntimeError(f"Item {item.item_id} has invalid answer_key={item.answer_key!r}")
        by_family[item.family].append(item)

    selected: list[CalibrationItem] = []
    for family in sorted(by_family):
        rows = sorted(by_family[family], key=lambda x: stable_hash_int(f"{seed}:item:{x.item_id}"))
        labels = {int(item.answerable) for item in rows}
        if labels != {0, 1}:
            raise RuntimeError(f"Family {family!r} does not contain both answerable and unanswerable rows.")
        if cap_per_family <= 0:
            selected.extend(rows)
            continue
        if cap_per_family < 2:
            raise RuntimeError("Lab 14 needs at least two rows per family to keep both labels alive.")
        grouped = {label: [item for item in rows if int(item.answerable) == label] for label in (0, 1)}
        # Balance first; if cap is odd and there is room, fill deterministically.
        target_each = max(1, cap_per_family // 2)
        taken: list[CalibrationItem] = []
        for label in (0, 1):
            taken.extend(grouped[label][:min(target_each, len(grouped[label]))])
        leftovers = [item for item in rows if item.item_id not in {x.item_id for x in taken}]
        for item in leftovers:
            if len(taken) >= cap_per_family:
                break
            taken.append(item)
        if {int(item.answerable) for item in taken} != {0, 1}:
            raise RuntimeError(f"Family {family!r} lost label balance under cap {cap_per_family}.")
        selected.extend(sorted(taken, key=lambda x: (x.family, x.answerable, x.item_id)))
    return selected


def load_items(args: Any) -> tuple[list[CalibrationItem], dict[str, Any]]:
    prompt_set = str(args.prompt_set)
    custom_path = pathlib.Path(prompt_set).expanduser() if prompt_set not in PROMPT_SET_FAMILY_CAPS else None
    if custom_path is not None and not custom_path.is_absolute() and not custom_path.exists():
        root_relative = bench.COURSE_ROOT / custom_path
        if root_relative.exists():
            custom_path = root_relative

    data_source = "frozen_csv"
    fallback_used = False
    source_path: pathlib.Path | None = None
    raw: list[CalibrationItem]
    if custom_path is not None:
        if not custom_path.exists():
            raise ValueError("Lab 14 uses --prompt-set small|medium|full or a path to a custom CSV/JSON item file.")
        source_path = custom_path.resolve()
        data_source = "custom_json" if source_path.suffix.lower() == ".json" else "custom_csv"
        raw = read_items_json(source_path) if source_path.suffix.lower() == ".json" else read_items_csv(source_path)
        cap = 0 if int(getattr(args, "max_examples", 0)) == 0 else max(0, int(getattr(args, "max_examples", 0)))
    else:
        source_path = data_path(DATA_FILE)
        cap = PROMPT_SET_FAMILY_CAPS[prompt_set]
        if getattr(args, "max_examples", 0) and int(args.max_examples) > 0:
            cap = int(args.max_examples)
        if source_path.exists():
            raw = read_items_csv(source_path)
        else:
            tier = str(getattr(args, "tier", ""))
            if tier == "a" or prompt_set == "small":
                raw = fallback_items()
                data_source = "built_in_tier_a_fallback"
                fallback_used = True
                source_path = None
            else:
                raise RuntimeError(
                    f"Frozen Lab 14 dataset missing: {source_path}. Use --tier a for the built-in "
                    "plumbing fallback, or add data/certainty_calibration_items.csv."
                )

    items = select_items(raw, cap, int(getattr(args, "seed", 0)))
    families = sorted({item.family for item in items})
    counts_by_family = {family: sum(1 for item in items if item.family == family) for family in families}
    label_counts_by_family = {
        family: {
            "answerable": sum(1 for item in items if item.family == family and int(item.answerable) == 1),
            "unanswerable": sum(1 for item in items if item.family == family and int(item.answerable) == 0),
        }
        for family in families
    }
    answer_key_counts_by_label = {
        str(label): {letter: sum(1 for item in items if int(item.answerable) == label and item.answer_key == letter) for letter in LETTERS}
        for label in (0, 1)
    }

    actual_hash = bench.sha256_file(source_path) if source_path and source_path.exists() else "fallback_no_file"
    expected_hash, manifest_status = expected_manifest_hash(DATA_FILE)
    manifest_match = None
    if expected_hash is not None and actual_hash != "fallback_no_file":
        manifest_match = actual_hash == expected_hash

    info = {
        "data_file": str(source_path) if source_path else "built_in_fallback_items",
        "data_source": data_source,
        "fallback_used": fallback_used,
        "data_sha256": actual_hash,
        "expected_sha256": expected_hash,
        "manifest_status": manifest_status,
        "manifest_match": manifest_match,
        "prompt_set": prompt_set,
        "per_family_cap": cap,
        "n_raw_items": len(raw),
        "n_items": len(items),
        "families": families,
        "counts_by_family": counts_by_family,
        "label_counts_by_family": label_counts_by_family,
        "answerable_counts": {
            "answerable": sum(1 for item in items if item.answerable == 1),
            "unanswerable": sum(1 for item in items if item.answerable == 0),
        },
        "answer_key_counts_by_label": answer_key_counts_by_label,
        "unanswerable_always_D": answer_key_counts_by_label["0"].get("D", 0) == sum(1 for item in items if item.answerable == 0),
        "all_answerable_non_D": answer_key_counts_by_label["1"].get("D", 0) == 0,
    }
    return items, info


# ---------------------------------------------------------------------------
# Prompt rendering and activation capture
# ---------------------------------------------------------------------------


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


def style_statement(item: CalibrationItem, style: str) -> str:
    if style == "confident":
        if item.confident_statement:
            return item.confident_statement
        return f"The answer is {item.expected_answer or item.option_text(item.answer_key)}."
    if style == "hedged":
        if item.hedged_statement:
            return item.hedged_statement
        return f"The answer might be {item.expected_answer or item.option_text(item.answer_key)}, but I am not completely sure."
    raise ValueError(f"unknown style {style!r}")


def style_probe_message(item: CalibrationItem, style: str) -> str:
    # Same wrapper for both styles. The contrast lives in the proposed answer
    # sentence, not in an instruction saying "be hedged" or "be confident".
    return (
        "Read this question and proposed answer. Do not solve the question; "
        "evaluate only the wording of the proposed answer.\n"
        f"{item_question_block(item)}\n"
        f"Proposed answer wording: {style_statement(item, style)}\n"
        "Is the proposed answer worded confidently or cautiously? Reply with one word."
    )


def render_chat(bundle: bench.ModelBundle, user_message: str) -> str:
    return bench.apply_chat_template(
        bundle,
        user_message,
        system=SYSTEM_PROMPT,
        add_generation_prompt=True,
    )


@contextlib.contextmanager
def temporary_padding_side(tokenizer: Any, side: str):
    """Temporarily set tokenizer padding side for batched generation.

    The bench continuous generator reads the prefill logits at the last padded
    column, so left padding is the safe convention for variable-length chat
    prompts. Restore the tokenizer immediately afterward so the lab does not
    leak state into later diagnostics.
    """
    old = getattr(tokenizer, "padding_side", None)
    if old is not None:
        tokenizer.padding_side = side
    try:
        yield
    finally:
        if old is not None:
            tokenizer.padding_side = old


def prompt_token_info(bundle: bench.ModelBundle, rendered: str) -> dict[str, Any]:
    ids = bundle.tokenizer.encode(rendered, add_special_tokens=False)
    last_id = ids[-1] if ids else None
    return {
        "n_tokens": len(ids),
        "prompt_sha256": sha256_text(rendered),
        "last_token_id": last_id,
        "last_token_text": bundle.tokenizer.decode([last_id]) if last_id is not None else "",
        "rendered_tail": rendered[-360:].replace("\n", "\\n"),
    }


def write_prompt_render_audit(ctx: bench.RunContext, bundle: bench.ModelBundle, items: Sequence[CalibrationItem]) -> dict[str, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    token_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for item in items:
        prompts = {
            "choice": render_chat(bundle, choice_user_message(item)),
            "confidence_report": render_chat(bundle, confidence_user_message(item)),
            "style_confident": render_chat(bundle, style_probe_message(item, "confident")),
            "style_hedged": render_chat(bundle, style_probe_message(item, "hedged")),
        }
        for kind, rendered in prompts.items():
            info = prompt_token_info(bundle, rendered)
            token_counts[item.item_id][kind] = int(info["n_tokens"])
            rows.append({
                "item_id": item.item_id,
                "family": item.family,
                "answerable": item.answerable,
                "prompt_kind": kind,
                **info,
            })
    path = ctx.path("diagnostics", "prompt_render_audit.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "diagnostic", "Chat-rendered prompt hashes, tails, and token counts for Lab 14 prompt types.")
    return token_counts


def capture_last_streams(bundle: bench.ModelBundle, templated_prompt: str) -> Any:
    cap = bench.run_with_residual_cache(bundle, templated_prompt, add_special_tokens=False)
    return cap.streams[:, -1, :], cap.final_logits_last


# ---------------------------------------------------------------------------
# Option distribution readout
# ---------------------------------------------------------------------------


def option_token_variants(bundle: bench.ModelBundle, letter: str) -> list[dict[str, Any]]:
    variants = (letter, " " + letter, letter + ".", " " + letter + ".", "\n" + letter, "\n" + letter + ".")
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for text in variants:
        ids = tuple(int(i) for i in bundle.tokenizer.encode(text, add_special_tokens=False))
        if not ids or ids in seen:
            continue
        seen.add(ids)
        out.append({
            "variant_text": text.replace("\n", "\\n"),
            "token_ids": " ".join(str(x) for x in ids),
            "first_token_id": int(ids[0]),
            "n_tokens": len(ids),
            "first_token_text": bundle.tokenizer.decode([ids[0]]),
        })
    return out


def option_token_ids(bundle: bench.ModelBundle, letter: str) -> list[int]:
    ids = {int(v["first_token_id"]) for v in option_token_variants(bundle, letter)}
    return sorted(ids)


def option_distribution(bundle: bench.ModelBundle, logits: Any) -> dict[str, Any]:
    import torch

    scores = []
    token_map: dict[str, list[int]] = {}
    for letter in LETTERS:
        ids = option_token_ids(bundle, letter)
        token_map[letter] = ids
        if not ids:
            scores.append(logits.new_tensor(float("-inf")))
        else:
            scores.append(torch.logsumexp(logits[ids], dim=0))
    score_tensor = torch.stack(scores).float()
    probs = torch.softmax(score_tensor, dim=0)
    entropy = float(-(probs * torch.log2(probs.clamp_min(1e-12))).sum())
    top = torch.topk(score_tensor, k=2)
    chosen = LETTERS[int(top.indices[0])]
    return {
        "scores": {letter: float(score_tensor[i]) for i, letter in enumerate(LETTERS)},
        "probs": {letter: float(probs[i]) for i, letter in enumerate(LETTERS)},
        "entropy_bits": entropy,
        "top_margin": float(top.values[0] - top.values[1]),
        "chosen": chosen,
        "chosen_prob": float(probs[int(top.indices[0])]),
        "token_map": token_map,
    }


def count_unanswerable_markers(text: str) -> int:
    low = text.lower()
    return sum(1 for marker in UNANSWERABLE_MARKERS if marker in low)


def item_length_features(item: CalibrationItem, token_counts: Mapping[str, Mapping[str, int]] | None = None) -> dict[str, Any]:
    choice_n_tokens = None
    if token_counts and item.item_id in token_counts and "choice" in token_counts[item.item_id]:
        choice_n_tokens = token_counts[item.item_id]["choice"]
    option_lengths = {letter: len(item.option_text(letter)) for letter in LETTERS}
    return {
        "question_n_chars": len(item.question),
        "option_total_n_chars": sum(option_lengths.values()),
        "option_d_n_chars": len(item.option_d),
        "option_d_unanswerable_markers": count_unanswerable_markers(item.option_d),
        "answer_key_is_d": 1 if item.answer_key == "D" else 0,
        "choice_prompt_n_tokens": choice_n_tokens if choice_n_tokens is not None else "",
    }


def write_option_token_audit(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    rows: list[dict[str, Any]] = []
    for letter in LETTERS:
        for variant in option_token_variants(bundle, letter):
            rows.append({"letter": letter, **variant})
    path = ctx.path("diagnostics", "option_token_audit.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "diagnostic", "Token IDs used to score A/B/C/D next-token options.")


def cache_choice_and_style(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    items: Sequence[CalibrationItem],
    token_counts: Mapping[str, Mapping[str, int]],
) -> tuple[Any, dict[str, dict[str, Any]], list[dict[str, Any]]]:
    import torch

    choice_vectors = []
    style_vectors: dict[str, dict[str, Any]] = {}
    behavior_rows: list[dict[str, Any]] = []
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
        length = item_length_features(item, token_counts)
        behavior_rows.append({
            "item_id": item.item_id,
            "family": item.family,
            "topic": item.topic,
            "answerable": item.answerable,
            "answer_key": item.answer_key,
            "chosen": dist["chosen"],
            "chosen_prob": rounded(dist["chosen_prob"]),
            "correct": dist["chosen"] == item.answer_key,
            "correct_prob": rounded(correct_prob),
            "best_wrong_prob": rounded(max(wrong_probs)),
            "correct_margin": rounded(correct_margin),
            "top_margin": rounded(dist["top_margin"]),
            "entropy_bits": rounded(dist["entropy_bits"]),
            "distribution_confidence": rounded(1.0 - dist["entropy_bits"] / math.log2(len(LETTERS))),
            "question": item.question,
            **{f"prob_{letter}": rounded(dist["probs"][letter]) for letter in LETTERS},
            **{f"score_{letter}": rounded(dist["scores"][letter]) for letter in LETTERS},
            **length,
        })

        style_vectors[item.item_id] = {}
        for style in ("confident", "hedged"):
            style_prompt = render_chat(bundle, style_probe_message(item, style))
            style_streams, _ = capture_last_streams(bundle, style_prompt)
            style_vectors[item.item_id][style] = style_streams
        if (i + 1) % report_every == 0:
            print(f"[lab14] cached choice/style features for {i + 1}/{len(items)} items")

    return torch.stack(choice_vectors), style_vectors, behavior_rows


# ---------------------------------------------------------------------------
# Splits, directions, and probes
# ---------------------------------------------------------------------------


def split_group_key(item: CalibrationItem) -> str:
    """Group matched variants so topic twins do not cross train/eval when possible."""

    topic = item.topic or "no_topic"
    fmt = item.answer_format or "fmt"
    return f"{item.family}::{topic}::{fmt}"


def make_split(items: Sequence[CalibrationItem], seed: int) -> dict[str, bool]:
    """Family- and label-stratified split, grouped by topic when possible."""

    split: dict[str, bool] = {}
    family_label_rows: dict[tuple[str, int], list[CalibrationItem]] = defaultdict(list)
    for item in items:
        family_label_rows[(item.family, int(item.answerable))].append(item)

    for (family, label), rows in family_label_rows.items():
        grouped: dict[str, list[CalibrationItem]] = defaultdict(list)
        for item in rows:
            grouped[split_group_key(item)].append(item)
        # If a tiny prompt set leaves only one group for a family/label, fall
        # back to item-level splitting rather than making the eval set empty.
        if len(grouped) < 2 and len(rows) > 1:
            grouped = {item.item_id: [item] for item in rows}
        ranked_keys = sorted(grouped, key=lambda key: stable_hash_int(f"{seed}:split:{family}:{label}:{key}"))
        n_train = int(round(TRAIN_FRACTION * len(ranked_keys)))
        if len(ranked_keys) > 1:
            n_train = max(1, min(len(ranked_keys) - 1, n_train))
        else:
            n_train = 1
        train_keys = set(ranked_keys[:n_train])
        for key, key_rows in grouped.items():
            for item in key_rows:
                split[item.item_id] = key in train_keys
    return split


def masks(items: Sequence[CalibrationItem], split: Mapping[str, bool], train: bool) -> list[bool]:
    return [bool(split[item.item_id]) == train for item in items]


def labels(items: Sequence[CalibrationItem]) -> list[int]:
    return [int(item.answerable) for item in items]


def mass_mean_direction(X: Any, y: Sequence[int]) -> Any | None:
    import torch

    y_t = torch.tensor([bool(v) for v in y])
    if not bool(y_t.any()) or not bool((~y_t).any()):
        return None
    return unit(X[y_t].mean(dim=0) - X[~y_t].mean(dim=0))


def scores_by_label(X: Any, direction: Any, y: Sequence[int]) -> tuple[list[float], list[float], list[float]]:
    scores = (X @ direction).tolist()
    pos = [float(s) for s, label in zip(scores, y) if int(label) == 1]
    neg = [float(s) for s, label in zip(scores, y) if int(label) == 0]
    return pos, neg, [float(s) for s in scores]


def direction_auc(X: Any, direction: Any, y: Sequence[int]) -> dict[str, float]:
    pos, neg, _ = scores_by_label(X, direction, y)
    return {
        "auc": auc_from_scores(pos, neg),
        "mean_pos_projection": safe_fmean(pos),
        "mean_neg_projection": safe_fmean(neg),
        "n_pos": len(pos),
        "n_neg": len(neg),
    }


def shuffled_labels(y: Sequence[int], seed: int) -> list[int]:
    order = sorted(range(len(y)), key=lambda i: stable_hash_int(f"{seed}:shuffle:{i}"))
    out = list(y)
    vals = [out[i] for i in order]
    if len(vals) > 1:
        vals = vals[1:] + vals[:1]
    for i, val in zip(order, vals):
        out[i] = val
    return out


def summarize_eval_dicts(vals: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    aucs = [float(v["auc"]) for v in vals if math.isfinite(float(v["auc"]))]
    return {
        "auc": safe_fmean(aucs),
        "auc_std": safe_stdev(aucs),
        "mean_pos_projection": safe_fmean([float(v["mean_pos_projection"]) for v in vals]),
        "mean_neg_projection": safe_fmean([float(v["mean_neg_projection"]) for v in vals]),
        "n_pos": int(max([int(v["n_pos"]) for v in vals] or [0])),
        "n_neg": int(max([int(v["n_neg"]) for v in vals] or [0])),
    }


def add_probe_row(
    rows: list[dict[str, Any]],
    *,
    probe: str,
    depth: int,
    direction_kind: str,
    split_name: str,
    summary: Mapping[str, Any],
    n_control_samples: int,
) -> None:
    rows.append({
        "probe": probe,
        "depth": depth,
        "direction_kind": direction_kind,
        "split": split_name,
        "auc": rounded(summary.get("auc")),
        "auc_std": rounded(summary.get("auc_std", 0.0)),
        "selectivity_vs_chance": rounded(safe_float(summary.get("auc")) - 0.5),
        "mean_pos_projection": rounded(summary.get("mean_pos_projection")),
        "mean_neg_projection": rounded(summary.get("mean_neg_projection")),
        "n_pos": summary.get("n_pos"),
        "n_neg": summary.get("n_neg"),
        "n_control_samples": n_control_samples,
    })


def run_certainty_probe_sweep(
    items: Sequence[CalibrationItem],
    choice_feats: Any,
    split: Mapping[str, bool],
    seed: int,
    d_model: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    import torch

    y_all = labels(items)
    train_idx = torch.tensor(masks(items, split, True), dtype=torch.bool)
    eval_idx = torch.tensor(masks(items, split, False), dtype=torch.bool)
    y_train = [label for label, keep in zip(y_all, train_idx.tolist()) if keep]
    y_eval = [label for label, keep in zip(y_all, eval_idx.tolist()) if keep]
    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    n_depths = int(choice_feats.shape[1])

    for depth in range(1, n_depths):
        Xtr = choice_feats[train_idx, depth, :]
        Xev = choice_feats[eval_idx, depth, :]
        real = mass_mean_direction(Xtr, y_train)
        if real is None:
            continue
        real_train = direction_auc(Xtr, real, y_train)
        real_eval = direction_auc(Xev, real, y_eval)
        add_probe_row(rows, probe="certainty_answerability", depth=depth, direction_kind="real", split_name="train", summary=real_train, n_control_samples=1)
        add_probe_row(rows, probe="certainty_answerability", depth=depth, direction_kind="real", split_name="eval", summary=real_eval, n_control_samples=1)

        shuffled_train: list[dict[str, float]] = []
        shuffled_eval: list[dict[str, float]] = []
        for j in range(N_SHUFFLES):
            y_shuf = shuffled_labels(y_train, seed + 10000 * depth + j)
            direction = mass_mean_direction(Xtr, y_shuf)
            if direction is None:
                continue
            shuffled_train.append(direction_auc(Xtr, direction, y_train))
            shuffled_eval.append(direction_auc(Xev, direction, y_eval))
        shuf_train_sum = summarize_eval_dicts(shuffled_train)
        shuf_eval_sum = summarize_eval_dicts(shuffled_eval)
        add_probe_row(rows, probe="certainty_answerability", depth=depth, direction_kind="shuffled", split_name="train", summary=shuf_train_sum, n_control_samples=len(shuffled_train))
        add_probe_row(rows, probe="certainty_answerability", depth=depth, direction_kind="shuffled", split_name="eval", summary=shuf_eval_sum, n_control_samples=len(shuffled_eval))

        random_train: list[dict[str, float]] = []
        random_eval: list[dict[str, float]] = []
        for j in range(N_RANDOM_DIRECTIONS):
            direction = random_unit(d_model, seed + 20000 * depth + j)
            # Orient random controls on the train split to make them a harder
            # baseline than a coin-flip arbitrary sign.
            _, _, train_scores = scores_by_label(Xtr, direction, y_train)
            if orientation_from_train(train_scores, y_train) < 0:
                direction = -direction
            random_train.append(direction_auc(Xtr, direction, y_train))
            random_eval.append(direction_auc(Xev, direction, y_eval))
        rand_train_sum = summarize_eval_dicts(random_train)
        rand_eval_sum = summarize_eval_dicts(random_eval)
        add_probe_row(rows, probe="certainty_answerability", depth=depth, direction_kind="random", split_name="train", summary=rand_train_sum, n_control_samples=len(random_train))
        add_probe_row(rows, probe="certainty_answerability", depth=depth, direction_kind="random", split_name="eval", summary=rand_eval_sum, n_control_samples=len(random_eval))

        train_control = max(safe_float(shuf_train_sum["auc"], 0.5), safe_float(rand_train_sum["auc"], 0.5), 0.5)
        eval_control = max(safe_float(shuf_eval_sum["auc"], 0.5), safe_float(rand_eval_sum["auc"], 0.5), 0.5)
        selection_rows.append({
            "probe": "certainty_answerability",
            "depth": depth,
            "train_real_auc": rounded(real_train["auc"]),
            "train_shuffled_auc": rounded(shuf_train_sum["auc"]),
            "train_random_auc": rounded(rand_train_sum["auc"]),
            "train_control_adjusted_auc": rounded(safe_float(real_train["auc"]) - train_control),
            "eval_real_auc": rounded(real_eval["auc"]),
            "eval_shuffled_auc": rounded(shuf_eval_sum["auc"]),
            "eval_random_auc": rounded(rand_eval_sum["auc"]),
            "eval_control_adjusted_auc": rounded(safe_float(real_eval["auc"]) - eval_control),
            "selection_split": "train",
        })

    def selection_score(row: Mapping[str, Any]) -> tuple[float, float, int]:
        return (
            safe_float(row.get("train_control_adjusted_auc"), -1.0),
            safe_float(row.get("train_real_auc"), -1.0),
            int(row.get("depth", 0)),
        )

    if not selection_rows:
        raise RuntimeError("No certainty probe rows were produced; check split labels.")
    best_row = max(
        selection_rows,
        key=lambda r: (
            safe_float(r.get("train_control_adjusted_auc"), -1.0),
            safe_float(r.get("train_real_auc"), -1.0),
            int(r.get("depth", 0)),
        ),
    )
    best_depth = int(best_row["depth"])
    return rows, selection_rows, best_depth


def signed_style_diffs(
    items: Sequence[CalibrationItem],
    style_feats: Mapping[str, dict[str, Any]],
    depth: int,
    item_filter: Callable[[CalibrationItem], bool],
) -> Any:
    import torch

    diffs = [
        style_feats[item.item_id]["confident"][depth] - style_feats[item.item_id]["hedged"][depth]
        for item in items
        if item_filter(item)
    ]
    if not diffs:
        return None
    return torch.stack(diffs)


def eval_style_direction(
    items: Sequence[CalibrationItem],
    style_feats: Mapping[str, dict[str, Any]],
    direction: Any,
    depth: int,
    item_filter: Callable[[CalibrationItem], bool],
) -> dict[str, float]:
    pos = [float(style_feats[item.item_id]["confident"][depth] @ direction) for item in items if item_filter(item)]
    neg = [float(style_feats[item.item_id]["hedged"][depth] @ direction) for item in items if item_filter(item)]
    return {
        "auc": auc_from_scores(pos, neg),
        "mean_pos_projection": safe_fmean(pos),
        "mean_neg_projection": safe_fmean(neg),
        "n_pos": len(pos),
        "n_neg": len(neg),
    }


def run_hedging_probe_sweep(
    items: Sequence[CalibrationItem],
    style_feats: Mapping[str, dict[str, Any]],
    split: Mapping[str, bool],
    seed: int,
    d_model: int,
    n_depths: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    import torch

    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    train_filter = lambda item: bool(split[item.item_id])
    eval_filter = lambda item: not bool(split[item.item_id])
    for depth in range(1, n_depths):
        diffs = signed_style_diffs(items, style_feats, depth, train_filter)
        if diffs is None:
            continue
        real = unit(diffs.mean(dim=0))
        real_train = eval_style_direction(items, style_feats, real, depth, train_filter)
        real_eval = eval_style_direction(items, style_feats, real, depth, eval_filter)
        add_probe_row(rows, probe="hedging_style", depth=depth, direction_kind="real", split_name="train", summary=real_train, n_control_samples=1)
        add_probe_row(rows, probe="hedging_style", depth=depth, direction_kind="real", split_name="eval", summary=real_eval, n_control_samples=1)

        shuffled_train: list[dict[str, float]] = []
        shuffled_eval: list[dict[str, float]] = []
        for j in range(N_SHUFFLES):
            signs = []
            for idx in range(diffs.shape[0]):
                signs.append(1.0 if stable_hash_int(f"{seed}:styleflip:{depth}:{j}:{idx}") % 2 else -1.0)
            sign_t = torch.tensor(signs, dtype=diffs.dtype).view(-1, 1)
            direction = unit((diffs * sign_t).mean(dim=0))
            shuffled_train.append(eval_style_direction(items, style_feats, direction, depth, train_filter))
            shuffled_eval.append(eval_style_direction(items, style_feats, direction, depth, eval_filter))
        shuf_train_sum = summarize_eval_dicts(shuffled_train)
        shuf_eval_sum = summarize_eval_dicts(shuffled_eval)
        add_probe_row(rows, probe="hedging_style", depth=depth, direction_kind="shuffled", split_name="train", summary=shuf_train_sum, n_control_samples=len(shuffled_train))
        add_probe_row(rows, probe="hedging_style", depth=depth, direction_kind="shuffled", split_name="eval", summary=shuf_eval_sum, n_control_samples=len(shuffled_eval))

        random_train: list[dict[str, float]] = []
        random_eval: list[dict[str, float]] = []
        for j in range(N_RANDOM_DIRECTIONS):
            direction = random_unit(d_model, seed + 30000 * depth + j)
            # Hard random control: orient to confident style on the train split.
            train_pos = [float(style_feats[item.item_id]["confident"][depth] @ direction) for item in items if train_filter(item)]
            train_neg = [float(style_feats[item.item_id]["hedged"][depth] @ direction) for item in items if train_filter(item)]
            if safe_fmean(train_pos) < safe_fmean(train_neg):
                direction = -direction
            random_train.append(eval_style_direction(items, style_feats, direction, depth, train_filter))
            random_eval.append(eval_style_direction(items, style_feats, direction, depth, eval_filter))
        rand_train_sum = summarize_eval_dicts(random_train)
        rand_eval_sum = summarize_eval_dicts(random_eval)
        add_probe_row(rows, probe="hedging_style", depth=depth, direction_kind="random", split_name="train", summary=rand_train_sum, n_control_samples=len(random_train))
        add_probe_row(rows, probe="hedging_style", depth=depth, direction_kind="random", split_name="eval", summary=rand_eval_sum, n_control_samples=len(random_eval))

        train_control = max(safe_float(shuf_train_sum["auc"], 0.5), safe_float(rand_train_sum["auc"], 0.5), 0.5)
        eval_control = max(safe_float(shuf_eval_sum["auc"], 0.5), safe_float(rand_eval_sum["auc"], 0.5), 0.5)
        selection_rows.append({
            "probe": "hedging_style",
            "depth": depth,
            "train_real_auc": rounded(real_train["auc"]),
            "train_shuffled_auc": rounded(shuf_train_sum["auc"]),
            "train_random_auc": rounded(rand_train_sum["auc"]),
            "train_control_adjusted_auc": rounded(safe_float(real_train["auc"]) - train_control),
            "eval_real_auc": rounded(real_eval["auc"]),
            "eval_shuffled_auc": rounded(shuf_eval_sum["auc"]),
            "eval_random_auc": rounded(rand_eval_sum["auc"]),
            "eval_control_adjusted_auc": rounded(safe_float(real_eval["auc"]) - eval_control),
            "selection_split": "train",
        })

    if not selection_rows:
        raise RuntimeError("No hedging-style probe rows were produced; check style variants.")
    best_row = max(
        selection_rows,
        key=lambda r: (
            safe_float(r.get("train_control_adjusted_auc"), -1.0),
            safe_float(r.get("train_real_auc"), -1.0),
            int(r.get("depth", 0)),
        ),
    )
    best_depth = int(best_row["depth"])
    return rows, selection_rows, best_depth


def direction_at_depth(
    items: Sequence[CalibrationItem],
    choice_feats: Any,
    split: Mapping[str, bool],
    depth: int,
) -> Any:
    import torch

    train_idx = torch.tensor(masks(items, split, True), dtype=torch.bool)
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
    seed: int,
    d_model: int,
) -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    families = sorted({item.family for item in items})
    for family in families:
        train_idx = torch.tensor([item.family != family for item in items], dtype=torch.bool)
        eval_idx = torch.tensor([item.family == family for item in items], dtype=torch.bool)
        y_train = [item.answerable for item in items if item.family != family]
        y_eval = [item.answerable for item in items if item.family == family]
        Xtr = choice_feats[train_idx, depth, :]
        Xev = choice_feats[eval_idx, depth, :]
        real = mass_mean_direction(Xtr, y_train)
        if real is None:
            continue
        real_summary = direction_auc(Xev, real, y_eval)
        rows.append({
            "held_out_family": family,
            "depth": depth,
            "direction_kind": "real",
            "auc": rounded(real_summary["auc"]),
            "auc_std": 0.0,
            "selectivity_vs_chance": rounded(safe_float(real_summary["auc"]) - 0.5),
            "n_eval_pos": real_summary["n_pos"],
            "n_eval_neg": real_summary["n_neg"],
            "n_train": int(train_idx.sum()),
            "n_control_samples": 1,
        })

        shuf_summaries: list[dict[str, float]] = []
        for j in range(N_SHUFFLES):
            direction = mass_mean_direction(Xtr, shuffled_labels(y_train, seed + 41000 + 101 * j + stable_hash_int(family) % 997))
            if direction is not None:
                shuf_summaries.append(direction_auc(Xev, direction, y_eval))
        shuf = summarize_eval_dicts(shuf_summaries)
        rows.append({
            "held_out_family": family,
            "depth": depth,
            "direction_kind": "shuffled",
            "auc": rounded(shuf["auc"]),
            "auc_std": rounded(shuf["auc_std"]),
            "selectivity_vs_chance": rounded(safe_float(shuf["auc"]) - 0.5),
            "n_eval_pos": shuf["n_pos"],
            "n_eval_neg": shuf["n_neg"],
            "n_train": int(train_idx.sum()),
            "n_control_samples": len(shuf_summaries),
        })

        rand_summaries: list[dict[str, float]] = []
        for j in range(N_RANDOM_DIRECTIONS):
            direction = random_unit(d_model, seed + 51000 + 101 * j + stable_hash_int(family) % 997)
            _, _, train_scores = scores_by_label(Xtr, direction, y_train)
            if orientation_from_train(train_scores, y_train) < 0:
                direction = -direction
            rand_summaries.append(direction_auc(Xev, direction, y_eval))
        rand = summarize_eval_dicts(rand_summaries)
        rows.append({
            "held_out_family": family,
            "depth": depth,
            "direction_kind": "random",
            "auc": rounded(rand["auc"]),
            "auc_std": rounded(rand["auc_std"]),
            "selectivity_vs_chance": rounded(safe_float(rand["auc"]) - 0.5),
            "n_eval_pos": rand["n_pos"],
            "n_eval_neg": rand["n_neg"],
            "n_train": int(train_idx.sum()),
            "n_control_samples": len(rand_summaries),
        })
    return rows


# ---------------------------------------------------------------------------
# Verbal confidence generation and parsing
# ---------------------------------------------------------------------------


def parse_confidence(text: str) -> tuple[str, float, str]:
    low = text.lower().strip()
    tokens = re.findall(r"[a-z']+", low)
    if tokens:
        first = tokens[0]
        if first in ("certain", "likely", "unsure", "guess"):
            return first, CONFIDENCE_VALUES[first], "first_token_exact"
    if re.search(r"\bnot\s+(certain|sure|confident)\b", low) or "don't know" in low or "do not know" in low:
        return "unsure", CONFIDENCE_VALUES["unsure"], "negated_confidence"
    for marker in ("certain", "likely", "unsure", "guess"):
        if re.search(rf"\b{marker}\b", low):
            return marker, CONFIDENCE_VALUES[marker], "contains_marker"
    if re.search(r"\b(impossible|unknown|cannot|can't|insufficient)\b", low):
        return "guess", CONFIDENCE_VALUES["guess"], "unknown_marker"
    return "unparsed", CONFIDENCE_VALUES["unparsed"], "unparsed_default"


def hedge_count(text: str) -> int:
    toks = re.findall(r"[A-Za-z']+", text.lower())
    return sum(1 for tok in toks if tok in HEDGE_WORDS)


def generate_confidence_reports(
    bundle: bench.ModelBundle,
    items: Sequence[CalibrationItem],
) -> list[dict[str, Any]]:
    prompts = [render_chat(bundle, confidence_user_message(item)) for item in items]
    with temporary_padding_side(bundle.tokenizer, "left"):
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
        label, score, parse_source = parse_confidence(text)
        rows.append({
            "item_id": item.item_id,
            "family": item.family,
            "answerable": item.answerable,
            "verbal_confidence_label": label,
            "verbal_confidence_score": rounded(score),
            "parse_source": parse_source,
            "hedge_marker_count": hedge_count(text),
            "confidence_text": text,
        })
    return rows


def write_confidence_parse_guide(ctx: bench.RunContext) -> None:
    lines = [
        "# Lab 14 verbal-confidence parse guide",
        "",
        "The parser is intentionally small. It supports the requested one-word labels "
        "`certain`, `likely`, `unsure`, and `guess`, plus a few negated-confidence "
        "fallbacks. Treat this as a scaffold for hand audit, not a judge.",
        "",
        "Student columns to add if needed:",
        "",
        "```csv",
        "student_label,student_notes",
        "```",
        "",
        "A good writeup quotes at least one raw `confidence_text` row where the parser "
        "or the model's self-report is questionable.",
    ]
    path = ctx.path("tables", "verbal_confidence_parse_guide.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "guide", "Hand-audit guide for generated verbal confidence labels.")


# ---------------------------------------------------------------------------
# Signal tables and controls
# ---------------------------------------------------------------------------


def build_signal_rows(
    items: Sequence[CalibrationItem],
    choice_feats: Any,
    behavior_rows: Sequence[Mapping[str, Any]],
    confidence_rows: Sequence[Mapping[str, Any]],
    certainty_direction: Any,
    hedging_at_certainty_depth: Any,
    split: Mapping[str, bool],
    depth: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    behavior = {r["item_id"]: r for r in behavior_rows}
    confidence = {r["item_id"]: r for r in confidence_rows}
    raw_rows: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        internal = float(choice_feats[i, depth, :] @ certainty_direction)
        hedging_proj = float(choice_feats[i, depth, :] @ hedging_at_certainty_depth)
        b = behavior[item.item_id]
        c = confidence[item.item_id]
        raw_rows.append({
            "item_id": item.item_id,
            "family": item.family,
            "topic": item.topic,
            "answerable": int(item.answerable),
            "answer_key": item.answer_key,
            "split_key": split_group_key(item),
            "split": "train" if split[item.item_id] else "eval",
            "correct": bool(b["correct"]),
            "chosen": b["chosen"],
            "internal_certainty_projection": internal,
            "hedging_style_projection": hedging_proj,
            "distribution_confidence": float(b["distribution_confidence"]),
            "correct_margin": float(b["correct_margin"]),
            "top_margin": float(b["top_margin"]),
            "entropy_bits": float(b["entropy_bits"]),
            "verbal_confidence_score": float(c["verbal_confidence_score"]),
            "verbal_confidence_label": c["verbal_confidence_label"],
            "parse_source": c.get("parse_source", ""),
            "hedge_marker_count": int(c["hedge_marker_count"]),
            "choice_prompt_n_tokens": safe_float(b.get("choice_prompt_n_tokens")),
            "question_n_chars": int(b.get("question_n_chars", 0)),
            "option_total_n_chars": int(b.get("option_total_n_chars", 0)),
            "option_d_unanswerable_markers": int(b.get("option_d_unanswerable_markers", 0)),
            "answer_key_is_d": int(b.get("answer_key_is_d", 0)),
        })

    train_rows = [r for r in raw_rows if r["split"] == "train"]
    internal_train = [r["internal_certainty_projection"] for r in train_rows]
    thresholds = {
        "internal": median(internal_train),
        "distribution": median([r["distribution_confidence"] for r in train_rows]),
        "verbal": 0.6,
    }
    for row in raw_rows:
        row["internal_rank_confidence"] = empirical_percentile(row["internal_certainty_projection"], internal_train)
        row["internal_signal"] = "high" if row["internal_certainty_projection"] >= thresholds["internal"] else "low"
        row["distribution_signal"] = "high" if row["distribution_confidence"] >= thresholds["distribution"] else "low"
        row["verbal_signal"] = "high" if row["verbal_confidence_score"] >= thresholds["verbal"] else "low"
        row["three_way_agreement"] = (
            row["internal_signal"] == row["distribution_signal"] == row["verbal_signal"]
        )
        row["signal_pattern"] = f"I:{row['internal_signal']}|D:{row['distribution_signal']}|V:{row['verbal_signal']}"
        for key in ("internal_certainty_projection", "hedging_style_projection", "distribution_confidence", "correct_margin", "top_margin", "entropy_bits", "verbal_confidence_score", "internal_rank_confidence"):
            row[key] = rounded(row[key])

    matrix_rows: list[dict[str, Any]] = []
    eval_rows = [r for r in raw_rows if r["split"] == "eval"] or raw_rows
    for internal in ("low", "high"):
        for distribution in ("low", "high"):
            for verbal in ("low", "high"):
                sub = [
                    r for r in eval_rows
                    if r["internal_signal"] == internal
                    and r["distribution_signal"] == distribution
                    and r["verbal_signal"] == verbal
                ]
                matrix_rows.append({
                    "split": "eval" if any(r["split"] == "eval" for r in raw_rows) else "all",
                    "internal_signal": internal,
                    "distribution_signal": distribution,
                    "verbal_signal": verbal,
                    "n": len(sub),
                    "accuracy": rounded(safe_fmean([1.0 if r["correct"] else 0.0 for r in sub])),
                    "answerable_rate": rounded(safe_fmean([float(r["answerable"]) for r in sub])),
                    "mean_internal_projection": rounded(safe_fmean([safe_float(r["internal_certainty_projection"]) for r in sub])),
                    "mean_distribution_confidence": rounded(safe_fmean([safe_float(r["distribution_confidence"]) for r in sub])),
                    "mean_verbal_confidence": rounded(safe_fmean([safe_float(r["verbal_confidence_score"]) for r in sub])),
                    "example_item_ids": " ".join(str(r["item_id"]) for r in sub[:5]),
                })
    return raw_rows, matrix_rows, thresholds


def signal_auc_from_rows(rows: Sequence[Mapping[str, Any]], signal_key: str, label_key: str, train_rows: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    if train_rows is None:
        train_rows = rows
    train_scores = [safe_float(r.get(signal_key)) for r in train_rows]
    train_labels = [int(bool(r.get(label_key))) for r in train_rows]
    sign = orientation_from_train(train_scores, train_labels)
    pos = [sign * safe_float(r.get(signal_key)) for r in rows if int(bool(r.get(label_key))) == 1]
    neg = [sign * safe_float(r.get(signal_key)) for r in rows if int(bool(r.get(label_key))) == 0]
    return {
        "auc": auc_from_scores(pos, neg),
        "orientation": sign,
        "n_pos": len(pos),
        "n_neg": len(neg),
    }


def signal_predictiveness_rows(signal_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    train_rows = [r for r in signal_rows if r["split"] == "train"]
    eval_rows = [r for r in signal_rows if r["split"] == "eval"] or list(signal_rows)
    signals = [
        ("internal_projection", "internal_certainty_projection", "DECODE"),
        ("internal_rank_confidence", "internal_rank_confidence", "DECODE_scaled"),
        ("distribution_confidence", "distribution_confidence", "OBS"),
        ("verbal_confidence", "verbal_confidence_score", "SELF_REPORT"),
        ("hedging_style_projection", "hedging_style_projection", "STYLE_CONTROL"),
        ("prompt_token_length", "choice_prompt_n_tokens", "LENGTH_CONTROL"),
        ("question_char_length", "question_n_chars", "LENGTH_CONTROL"),
        ("answer_key_is_D", "answer_key_is_d", "LETTER_CONTROL"),
        ("option_D_unanswerable_markers", "option_d_unanswerable_markers", "TEXT_CONTROL"),
    ]
    rows: list[dict[str, Any]] = []
    for signal_name, key, evidence_kind in signals:
        for label_name, label_key in (("answerability", "answerable"), ("correctness", "correct")):
            result = signal_auc_from_rows(eval_rows, key, label_key, train_rows)
            rows.append({
                "signal": signal_name,
                "signal_column": key,
                "evidence_kind": evidence_kind,
                "predicts": label_name,
                "split": "eval" if any(r["split"] == "eval" for r in signal_rows) else "all",
                "auc": rounded(result["auc"]),
                "orientation_from_train": result["orientation"],
                "n_pos": result["n_pos"],
                "n_neg": result["n_neg"],
            })
    return rows


def length_baseline_rows(items: Sequence[CalibrationItem], split: Mapping[str, bool], behavior_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    behavior = {r["item_id"]: r for r in behavior_rows}
    base_rows = []
    for item in items:
        b = behavior[item.item_id]
        base_rows.append({
            "item_id": item.item_id,
            "split": "train" if split[item.item_id] else "eval",
            "answerable": item.answerable,
            "correct": bool(b["correct"]),
            "choice_prompt_n_tokens": safe_float(b.get("choice_prompt_n_tokens")),
            "question_n_chars": safe_float(b.get("question_n_chars")),
            "option_total_n_chars": safe_float(b.get("option_total_n_chars")),
            "option_d_n_chars": safe_float(b.get("option_d_n_chars")),
            "option_d_unanswerable_markers": safe_float(b.get("option_d_unanswerable_markers")),
            "answer_key_is_d": safe_float(b.get("answer_key_is_d")),
        })
    train_rows = [r for r in base_rows if r["split"] == "train"]
    eval_rows = [r for r in base_rows if r["split"] == "eval"] or base_rows
    out: list[dict[str, Any]] = []
    for key in ("choice_prompt_n_tokens", "question_n_chars", "option_total_n_chars", "option_d_n_chars", "option_d_unanswerable_markers", "answer_key_is_d"):
        for label_name, label_key in (("answerability", "answerable"), ("correctness", "correct")):
            result = signal_auc_from_rows(eval_rows, key, label_key, train_rows)
            out.append({
                "baseline": key,
                "predicts": label_name,
                "split": "eval" if any(r["split"] == "eval" for r in base_rows) else "all",
                "auc": rounded(result["auc"]),
                "orientation_from_train": result["orientation"],
                "n_pos": result["n_pos"],
                "n_neg": result["n_neg"],
            })
    return out


def quantile_edges(vals: Sequence[float], n_bins: int) -> list[float]:
    finite = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not finite:
        return []
    edges = []
    for k in range(1, n_bins):
        idx = min(len(finite) - 1, max(0, int(round(k * (len(finite) - 1) / n_bins))))
        edges.append(finite[idx])
    return edges


def bin_index(value: float, edges: Sequence[float]) -> int:
    idx = 0
    for edge in edges:
        if value > edge:
            idx += 1
    return idx


def reliability_rows(signal_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    eval_rows = [r for r in signal_rows if r["split"] == "eval"] or list(signal_rows)
    train_rows = [r for r in signal_rows if r["split"] == "train"] or list(signal_rows)
    rows: list[dict[str, Any]] = []
    eces: dict[str, float] = {}

    # Verbal labels use fixed bins because the labels themselves are the report.
    total = len(eval_rows)
    verbal_ece = 0.0
    for label in CONFIDENCE_ORDER:
        sub = [r for r in eval_rows if r["verbal_confidence_label"] == label]
        if not sub:
            continue
        conf = safe_fmean([safe_float(r["verbal_confidence_score"]) for r in sub])
        acc = safe_fmean([1.0 if r["correct"] else 0.0 for r in sub])
        verbal_ece += (len(sub) / max(1, total)) * abs(acc - conf)
        rows.append({
            "signal": "verbal_confidence",
            "bin": label,
            "n": len(sub),
            "mean_signal_confidence": rounded(conf),
            "accuracy": rounded(acc),
            "answerable_rate": rounded(safe_fmean([float(r["answerable"]) for r in sub])),
            "abs_gap": rounded(abs(acc - conf)),
        })
    eces["verbal_confidence"] = verbal_ece

    for signal, key in (("distribution_confidence", "distribution_confidence"), ("internal_rank_confidence", "internal_rank_confidence")):
        edges = quantile_edges([safe_float(r[key]) for r in train_rows], N_RELIABILITY_BINS)
        ece = 0.0
        for b in range(N_RELIABILITY_BINS):
            sub = [r for r in eval_rows if bin_index(safe_float(r[key]), edges) == b]
            if not sub:
                continue
            conf = safe_fmean([safe_float(r[key]) for r in sub])
            acc = safe_fmean([1.0 if r["correct"] else 0.0 for r in sub])
            ece += (len(sub) / max(1, total)) * abs(acc - conf)
            rows.append({
                "signal": signal,
                "bin": f"q{b + 1}",
                "n": len(sub),
                "mean_signal_confidence": rounded(conf),
                "accuracy": rounded(acc),
                "answerable_rate": rounded(safe_fmean([float(r["answerable"]) for r in sub])),
                "abs_gap": rounded(abs(acc - conf)),
            })
        eces[signal] = ece
    return rows, eces


def disagreement_examples(signal_rows: Sequence[Mapping[str, Any]], max_rows: int = 24) -> list[dict[str, Any]]:
    eval_rows = [r for r in signal_rows if r["split"] == "eval"] or list(signal_rows)
    disagree = [r for r in eval_rows if not r["three_way_agreement"]]
    def severity(row: Mapping[str, Any]) -> tuple[int, float]:
        signals = [row["internal_signal"], row["distribution_signal"], row["verbal_signal"]]
        n_high = sum(1 for s in signals if s == "high")
        # 1-vs-2 disagreements first, then larger projection disagreements.
        return (1 if n_high in (1, 2) else 0, abs(safe_float(row["internal_rank_confidence"]) - safe_float(row["verbal_confidence_score"])))
    out = []
    for r in sorted(disagree, key=severity, reverse=True)[:max_rows]:
        out.append({
            "item_id": r["item_id"],
            "family": r["family"],
            "answerable": r["answerable"],
            "correct": r["correct"],
            "chosen": r["chosen"],
            "signal_pattern": r["signal_pattern"],
            "internal_rank_confidence": r["internal_rank_confidence"],
            "distribution_confidence": r["distribution_confidence"],
            "verbal_confidence_score": r["verbal_confidence_score"],
            "verbal_confidence_label": r["verbal_confidence_label"],
            "hedging_style_projection": r["hedging_style_projection"],
            "parse_source": r["parse_source"],
        })
    return out


def paired_finite_values(rows: Sequence[Mapping[str, Any]], x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x = safe_float(row.get(x_key))
        y = safe_float(row.get(y_key))
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return xs, ys


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_probe_by_layer(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]], best_certainty_depth: int, best_hedging_depth: int) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10.6, 7.2), sharex=True)
    fig.patch.set_facecolor("white")
    ax1, ax2 = axes[0], axes[1]
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.25)
        ax.set_facecolor("white")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    configs = [
        (ax1, "certainty_answerability", best_certainty_depth, "Answerability / certainty direction"),
        (ax2, "hedging_style", best_hedging_depth, "Confident-vs-hedged wording direction"),
    ]
    for ax, probe, marker_depth, title in configs:
        for kind, linestyle, label in (("real", "-", "real"), ("shuffled", ":", "shuffled"), ("random", "--", "random")):
            pts = sorted(
                (int(r["depth"]), safe_float(r["auc"]))
                for r in rows
                if r["probe"] == probe and r["direction_kind"] == kind and r.get("split") == "eval"
                and math.isfinite(safe_float(r.get("auc")))
            )
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], linestyle=linestyle, marker="o", label=label)
        ax.axhline(0.5, color="black", linewidth=0.8, alpha=0.5)
        ax.axvline(marker_depth, color="black", linewidth=0.8, alpha=0.5)
        ax.set_ylabel("eval AUC")
        ax.set_title(title)
        ax.legend(fontsize=8)
        bench.style_ax(ax)
    ax2.set_xlabel("residual-stream depth")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "certainty_probe_by_layer.png", "Certainty/answerability and hedging-style AUC by residual depth, with controls.")


def plot_reliability(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(7.2, 5.8))
    ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1.0, label="perfect calibration")
    for signal, marker in (("verbal_confidence", "o"), ("distribution_confidence", "s"), ("internal_rank_confidence", "^")):
        pts = [r for r in rows if r["signal"] == signal]
        if not pts:
            continue
        xs = [safe_float(r["mean_signal_confidence"]) for r in pts]
        ys = [safe_float(r["accuracy"]) for r in pts]
        sizes = [35 + 8 * int(r["n"]) for r in pts]
        ax.scatter(xs, ys, s=sizes, marker=marker, label=signal.replace("_", " "))
        for x, y, r in zip(xs, ys, pts):
            ax.annotate(str(r["bin"]), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("signal value treated as confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title("Reliability of verbal, distributional, and internal confidence proxies")
    ax.legend(fontsize=8)
    bench.style_ax(ax)
    bench.save_figure(ctx, fig, "reliability_diagram.png", "Reliability diagram for verbal, distributional, and internal-rank confidence signals.")


def plot_disagreement_matrix(ctx: bench.RunContext, matrix_rows: Sequence[Mapping[str, Any]]) -> None:
    import numpy as np

    row_labels = [f"I:{i} / D:{d}" for i in ("low", "high") for d in ("low", "high")]
    col_labels = ["V:low", "V:high"]
    grid = np.zeros((len(row_labels), len(col_labels)))
    acc = np.full_like(grid, np.nan, dtype=float)
    answerable = np.full_like(grid, np.nan, dtype=float)
    for r in matrix_rows:
        ri = row_labels.index(f"I:{r['internal_signal']} / D:{r['distribution_signal']}")
        ci = col_labels.index(f"V:{r['verbal_signal']}")
        grid[ri, ci] = int(r["n"])
        if int(r["n"]):
            acc[ri, ci] = safe_float(r["accuracy"])
            answerable[ri, ci] = safe_float(r["answerable_rate"])
    fig, ax = bench.new_figure(figsize=(7.0, 5.6))
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
                text += f"\nacc={acc[i, j]:.2f}\nans={answerable[i, j]:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.035, label="count")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "confidence_disagreement_matrix.png", "Internal/distribution/verbal confidence disagreement matrix with accuracy and answerability rates.")


def plot_signal_correlation(ctx: bench.RunContext, signal_rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    rows = [r for r in signal_rows if r["split"] == "eval"] or list(signal_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    fig.patch.set_facecolor("white")
    ax1, ax2 = axes[0], axes[1]
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.25)
        ax.set_facecolor("white")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    xs = [safe_float(r["internal_rank_confidence"]) for r in rows]
    ys_dist = [safe_float(r["distribution_confidence"]) for r in rows]
    ys_verbal = [safe_float(r["verbal_confidence_score"]) for r in rows]
    labels = [int(r["answerable"]) for r in rows]
    ax1.scatter(xs, ys_dist, c=labels, alpha=0.8)
    ax1.set_xlabel("internal rank confidence")
    ax1.set_ylabel("distribution confidence")
    ax1.set_title("Internal vs option-distribution confidence")
    bench.style_ax(ax1)
    ax2.scatter(xs, ys_verbal, c=labels, alpha=0.8)
    ax2.set_xlabel("internal rank confidence")
    ax2.set_ylabel("verbal confidence score")
    ax2.set_title("Internal vs verbal self-report")
    bench.style_ax(ax2)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "confidence_signal_correlations.png", "Eval-set scatter plots comparing internal confidence to distributional and verbal signals.")


def plot_family_heldout(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    families = sorted({str(r["held_out_family"]) for r in rows})
    if not families:
        return
    x = list(range(len(families)))
    fig, ax = bench.new_figure(figsize=(8.0, 5.0))
    width = 0.24
    kinds = ["real", "shuffled", "random"]
    offsets = {"real": -width, "shuffled": 0.0, "random": width}
    for kind in kinds:
        vals = []
        for family in families:
            sub = [r for r in rows if r["held_out_family"] == family and r["direction_kind"] == kind]
            vals.append(safe_float(sub[0]["auc"]) if sub else float("nan"))
        ax.bar([i + offsets[kind] for i in x], vals, width=width, label=kind)
    ax.axhline(0.5, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=20, ha="right")
    ax.set_ylabel("held-out AUC")
    ax.set_title("Family-held-out answerability generalization")
    ax.legend(fontsize=8)
    bench.style_ax(ax)
    fig.tight_layout()
    bench.save_figure(ctx, fig, "family_heldout_generalization.png", "Family-held-out AUC for the answerability direction and controls.")


# ---------------------------------------------------------------------------
# Cards, audits, and claims
# ---------------------------------------------------------------------------


def metric_value(rows: Sequence[Mapping[str, Any]], *, probe: str, depth: int, kind: str, split: str, key: str = "auc") -> float:
    vals = [
        safe_float(r.get(key))
        for r in rows
        if r.get("probe") == probe and int(r.get("depth", -1)) == depth and r.get("direction_kind") == kind and r.get("split") == split
    ]
    return safe_fmean(vals)


def control_gap(rows: Sequence[Mapping[str, Any]], *, probe: str, depth: int, split: str) -> float:
    real = metric_value(rows, probe=probe, depth=depth, kind="real", split=split)
    shuffled = metric_value(rows, probe=probe, depth=depth, kind="shuffled", split=split)
    random = metric_value(rows, probe=probe, depth=depth, kind="random", split=split)
    return real - max(0.5, shuffled, random)


def infer_verdict(metrics: Mapping[str, Any]) -> str:
    auc = safe_float(metrics.get("certainty_auc_eval_best_depth"), 0.0)
    gap = safe_float(metrics.get("certainty_control_gap_eval_best_depth"), -1.0)
    family_auc = safe_float(metrics.get("mean_family_heldout_auc_real"), 0.0)
    family_gap = safe_float(metrics.get("mean_family_heldout_control_gap"), -1.0)
    length_auc = safe_float(metrics.get("max_length_or_letter_baseline_answerability_auc"), 1.0)
    hedging_auc = safe_float(metrics.get("hedging_projection_answerability_auc"), 1.0)
    distribution_auc = safe_float(metrics.get("distribution_answerability_auc"), 1.0)
    if auc >= 0.68 and gap >= 0.08 and family_auc >= 0.60 and family_gap >= 0.04:
        best_trivial = max(length_auc, hedging_auc, distribution_auc)
        # A "usable" instrument must clearly beat the trivial baselines (margin)
        # AND those baselines must not separate answerable/unanswerable well on
        # their own (absolute level). A near-tie -- e.g. probe 1.0 vs a 0.92
        # length/D-key baseline -- means the split is itself trivially separable
        # by surface features (the D-option/length trap the writeup warns about),
        # so the probe cannot be certified as reading certainty rather than the
        # answer frame. The old 0.03 margin let that case through; require a 0.10
        # margin and a sub-0.80 trivial baseline. A de-confounded dataset (keys
        # spread across A-D, matched lengths) drops the baseline well under 0.80
        # and passes easily.
        if best_trivial <= auc - 0.10 and best_trivial < 0.80:
            return "usable_certainty_instrument"
        return "answerability_decodes_but_confounds_compete"
    if auc >= 0.60 and gap > 0.03:
        return "weak_or_family_limited_certainty_signal"
    return "not_validated_as_certainty_instrument"


def write_certainty_instrument_card(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    verdict = metrics.get("verdict", "unknown")
    if verdict == "usable_certainty_instrument":
        posture = "The run validates a downstream-usable answerability direction under the current controls."
    elif verdict == "answerability_decodes_but_confounds_compete":
        posture = "Answerability decodes, but distribution, style, length, prompt-text, or style controls are close enough to keep downstream claims cautious."
    elif verdict == "weak_or_family_limited_certainty_signal":
        posture = "There is a weak or family-limited signal. Treat it as an instrument under development, not a reusable certainty gauge."
    else:
        posture = "This run does not validate a clean certainty instrument. That is a finding, not a failed lab."

    lines = [
        "# Lab 14 Certainty Instrument Card",
        "",
        f"**Verdict:** `{verdict}`",
        "",
        posture,
        "",
        "## Headline metrics",
        "",
        f"- model: `{metrics.get('model_id')}`",
        f"- items: {metrics.get('n_items')}",
        f"- certainty depth selected on train split: {metrics.get('best_certainty_depth')}",
        f"- certainty eval AUC: {metrics.get('certainty_auc_eval_best_depth')}",
        f"- certainty eval control gap: {metrics.get('certainty_control_gap_eval_best_depth')}",
        f"- mean family-held-out real AUC: {metrics.get('mean_family_heldout_auc_real')}",
        f"- mean family-held-out control gap: {metrics.get('mean_family_heldout_control_gap')}",
        f"- hedging projection answerability AUC: {metrics.get('hedging_projection_answerability_auc')}",
        f"- distribution-confidence answerability AUC: {metrics.get('distribution_answerability_auc')}",
        f"- max length/prompt-text baseline answerability AUC: {metrics.get('max_length_or_letter_baseline_answerability_auc')}",
        f"- verbal confidence ECE: {metrics.get('verbal_confidence_ece')}",
        "",
        "## Read before reuse",
        "",
        "The saved `state/certainty_direction.pt` is an answerability direction in a fixed A/B/C/D frame. "
        "It is not a direct measurement of subjective confidence, knowledge, belief, or honesty. "
        "Downstream labs should project it only with its metadata and should carry the verdict above into their own ledger entries.",
        "",
        "## First artifacts to inspect",
        "",
        "1. `tables/depth_selection.csv` - make sure depth selection was not a pretty-curve pick.",
        "2. `tables/family_heldout_generalization.csv` - check whether the direction transfers across families.",
        "3. `tables/signal_predictiveness.csv` - compare the internal direction against entropy, verbal confidence, hedging, length, and answer-key baselines.",
        "4. `tables/disagreement_examples.csv` - choose a concrete case before writing any SELF-REPORT claim.",
    ]
    path = ctx.path("certainty_instrument_card.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "card", "Read-first verdict card for the Lab 14 certainty instrument.")


def write_operationalization_audit(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    lines = [
        "# Lab 14 Operationalization Audit",
        "",
        "## What was measured",
        "",
        "The lab measures answerability/certainty in a controlled fixed-choice frame. "
        "Answerable questions have a checkable option. Known-unanswerable questions are scored with the designated unanswerable option, usually `D`. "
        "The internal direction is a residual-stream probe for this operational label, not a direct meter of subjective confidence.",
        "",
        "## Cheap explanations and where they are tested",
        "",
        "| Cheap explanation | Audit artifact | How it can kill the claim |",
        "|---|---|---|",
        "| Family or topic | `tables/family_heldout_generalization.csv` | Held-out AUC collapses or controls match the real direction. |",
        "| Hedging or politeness style | `tables/probe_report.csv`, `tables/signal_predictiveness.csv` | The confident-vs-hedged direction predicts answerability as well as the certainty direction. |",
        "| Entropy or option sharpness | `tables/answer_distribution_readout.csv`, `tables/signal_predictiveness.csv` | Distribution confidence explains the labels without the internal projection adding anything. |",
        "| Length or formatting | `tables/length_and_letter_baselines.csv` | Prompt length, option length, or D-option wording predicts answerability competitively. |",
        "| Self-report fluency | `tables/verbal_confidence_reports.csv`, `tables/reliability_curve.csv` | The model emits confidence words that are formatted but not calibrated. |",
        "| Answer-letter metadata audit | `diagnostics/frozen_data_manifest.json`, `tables/length_and_letter_baselines.csv` | `answer_key_is_D` should be treated as label metadata, not a model-visible signal; if students rely on it, the claim is a frame artifact. |",
        "",
        "## Headline numbers",
        "",
        f"- Verdict: `{metrics.get('verdict')}`",
        f"- Best certainty depth: {metrics.get('best_certainty_depth')}",
        f"- Best hedging depth: {metrics.get('best_hedging_depth')}",
        f"- Certainty eval AUC: {metrics.get('certainty_auc_eval_best_depth')}",
        f"- Certainty eval control gap: {metrics.get('certainty_control_gap_eval_best_depth')}",
        f"- Mean family-held-out real AUC: {metrics.get('mean_family_heldout_auc_real')}",
        f"- Internal/distribution correlation: {metrics.get('internal_distribution_correlation_eval')}",
        f"- Internal/verbal correlation: {metrics.get('internal_verbal_correlation_eval')}",
        f"- Internal/hedging-style correlation: {metrics.get('internal_hedging_correlation_eval')}",
        f"- Verbal confidence ECE: {metrics.get('verbal_confidence_ece')}",
        "",
        "## Allowed claim",
        "",
        "An internal uncertainty-adjacent claim is allowed only when the direction predicts answerability beyond family/topic, shuffled/random controls, hedging style, length and prompt-text baselines, and entropy alone. Otherwise the honest claim is narrower: this model exposes a feature of the answer frame, style, prompt difficulty, or its own self-report behavior.",
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
        f"- Verdict: `{metrics.get('verdict')}`",
        f"- Best certainty depth: {metrics.get('best_certainty_depth')}",
        f"- Certainty eval AUC: {metrics.get('certainty_auc_eval_best_depth')}",
        f"- Certainty eval control gap: {metrics.get('certainty_control_gap_eval_best_depth')}",
        f"- Best hedging depth: {metrics.get('best_hedging_depth')}",
        f"- Hedging-style eval AUC: {metrics.get('hedging_auc_eval_best_depth')}",
        f"- Mean family-held-out real AUC: {metrics.get('mean_family_heldout_auc_real')}",
        f"- Internal/distribution correlation on eval: {metrics.get('internal_distribution_correlation_eval')}",
        f"- Internal/verbal correlation on eval: {metrics.get('internal_verbal_correlation_eval')}",
        f"- Verbal confidence ECE: {metrics.get('verbal_confidence_ece')}",
        "",
        "Start with `certainty_instrument_card.md`, then inspect `tables/disagreement_examples.csv` before writing the SELF-REPORT claim. The disagreement case is the tiny trapdoor where the lab becomes useful.",
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Human-readable summary of headline Lab 14 metrics.")


def build_metrics(
    *,
    bundle: bench.ModelBundle,
    items: Sequence[CalibrationItem],
    data_info: Mapping[str, Any],
    probe_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    signal_rows: Sequence[Mapping[str, Any]],
    signal_pred_rows: Sequence[Mapping[str, Any]],
    length_rows: Sequence[Mapping[str, Any]],
    reliability_eces: Mapping[str, float],
    best_certainty_depth: int,
    best_hedging_depth: int,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    eval_rows = [r for r in signal_rows if r["split"] == "eval"] or list(signal_rows)
    family_real = [safe_float(r["auc"]) for r in family_rows if r["direction_kind"] == "real"]
    family_shuf = [safe_float(r["auc"]) for r in family_rows if r["direction_kind"] == "shuffled"]
    family_rand = [safe_float(r["auc"]) for r in family_rows if r["direction_kind"] == "random"]
    family_control = max(safe_fmean(family_shuf, 0.5), safe_fmean(family_rand, 0.5), 0.5)

    def pred_auc(signal: str, predicts: str = "answerability") -> float:
        vals = [safe_float(r["auc"]) for r in signal_pred_rows if r["signal"] == signal and r["predicts"] == predicts]
        return safe_fmean(vals)

    internal_dist_x, internal_dist_y = paired_finite_values(
        eval_rows, "internal_rank_confidence", "distribution_confidence"
    )
    internal_verbal_x, internal_verbal_y = paired_finite_values(
        eval_rows, "internal_rank_confidence", "verbal_confidence_score"
    )
    internal_hedging_x, internal_hedging_y = paired_finite_values(
        eval_rows, "internal_rank_confidence", "hedging_style_projection"
    )
    distribution_verbal_x, distribution_verbal_y = paired_finite_values(
        eval_rows, "distribution_confidence", "verbal_confidence_score"
    )
    length_answerability = [
        safe_float(r["auc"]) for r in length_rows
        if r["predicts"] == "answerability" and r.get("baseline") != "answer_key_is_d"
    ]

    metrics = {
        "model_id": bundle.anatomy.model_id,
        "n_items": len(items),
        "families": data_info["families"],
        "best_certainty_depth": best_certainty_depth,
        "best_hedging_depth": best_hedging_depth,
        "certainty_auc_train_best_depth": none_if_nan(rounded(metric_value(probe_rows, probe="certainty_answerability", depth=best_certainty_depth, kind="real", split="train"))),
        "certainty_auc_eval_best_depth": none_if_nan(rounded(metric_value(probe_rows, probe="certainty_answerability", depth=best_certainty_depth, kind="real", split="eval"))),
        "certainty_shuffled_auc_eval_best_depth": none_if_nan(rounded(metric_value(probe_rows, probe="certainty_answerability", depth=best_certainty_depth, kind="shuffled", split="eval"))),
        "certainty_random_auc_eval_best_depth": none_if_nan(rounded(metric_value(probe_rows, probe="certainty_answerability", depth=best_certainty_depth, kind="random", split="eval"))),
        "certainty_control_gap_eval_best_depth": none_if_nan(rounded(control_gap(probe_rows, probe="certainty_answerability", depth=best_certainty_depth, split="eval"))),
        "certainty_control_gap_train_best_depth": none_if_nan(rounded(control_gap(probe_rows, probe="certainty_answerability", depth=best_certainty_depth, split="train"))),
        "hedging_auc_train_best_depth": none_if_nan(rounded(metric_value(probe_rows, probe="hedging_style", depth=best_hedging_depth, kind="real", split="train"))),
        "hedging_auc_eval_best_depth": none_if_nan(rounded(metric_value(probe_rows, probe="hedging_style", depth=best_hedging_depth, kind="real", split="eval"))),
        "hedging_control_gap_eval_best_depth": none_if_nan(rounded(control_gap(probe_rows, probe="hedging_style", depth=best_hedging_depth, split="eval"))),
        "mean_family_heldout_auc_real": none_if_nan(rounded(safe_fmean(family_real))),
        "mean_family_heldout_auc_shuffled": none_if_nan(rounded(safe_fmean(family_shuf))),
        "mean_family_heldout_auc_random": none_if_nan(rounded(safe_fmean(family_rand))),
        "mean_family_heldout_control_gap": none_if_nan(rounded(safe_fmean(family_real) - family_control)),
        "internal_distribution_correlation_eval": none_if_nan(rounded(pearson(internal_dist_x, internal_dist_y))),
        "internal_verbal_correlation_eval": none_if_nan(rounded(pearson(internal_verbal_x, internal_verbal_y))),
        "internal_hedging_correlation_eval": none_if_nan(rounded(pearson(internal_hedging_x, internal_hedging_y))),
        "distribution_verbal_correlation_eval": none_if_nan(rounded(pearson(distribution_verbal_x, distribution_verbal_y))),
        "internal_answerability_auc": none_if_nan(rounded(pred_auc("internal_projection"))),
        "distribution_answerability_auc": none_if_nan(rounded(pred_auc("distribution_confidence"))),
        "verbal_answerability_auc": none_if_nan(rounded(pred_auc("verbal_confidence"))),
        "hedging_projection_answerability_auc": none_if_nan(rounded(pred_auc("hedging_style_projection"))),
        "max_length_or_letter_baseline_answerability_auc": none_if_nan(rounded(max(length_answerability) if length_answerability else float("nan"))),
        "verbal_confidence_ece": none_if_nan(rounded(reliability_eces.get("verbal_confidence", float("nan")))),
        "distribution_confidence_ece": none_if_nan(rounded(reliability_eces.get("distribution_confidence", float("nan")))),
        "internal_rank_confidence_ece": none_if_nan(rounded(reliability_eces.get("internal_rank_confidence", float("nan")))),
        "signal_thresholds": {k: rounded(v) for k, v in thresholds.items()},
        "data": data_info,
    }
    metrics["verdict"] = infer_verdict(metrics)
    return metrics


def save_directions(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    certainty_direction: Any,
    hedging_direction: Any,
    *,
    best_certainty_depth: int,
    best_hedging_depth: int,
    metrics: Mapping[str, Any],
) -> None:
    import torch

    common = {
        "depth_convention": "bench streams[k]: 0 = embeddings, k = pre-norm residual after k blocks",
        "read_site": "chat-templated final prompt token before assistant generation",
        "model_id": bundle.anatomy.model_id,
        "d_model": bundle.anatomy.d_model,
        "n_layers": bundle.anatomy.n_layers,
        "lab_id": LAB_ID,
        "verdict": metrics.get("verdict"),
    }
    certainty_state = {
        **common,
        "depth": best_certainty_depth,
        "direction": certainty_direction,
        "label": "answerable (1) minus known-unanswerable (0)",
        "method": "mass-mean direction on train split only; depth selected by train control-adjusted AUC",
    }
    certainty_path = ctx.path("state", "certainty_direction.pt")
    torch.save(certainty_state, certainty_path)
    ctx.register_artifact(certainty_path, "tensor", "Train-split mass-mean answerability/certainty direction.")

    certainty_meta = {k: v for k, v in certainty_state.items() if k != "direction"}
    certainty_meta.update({
        "certainty_auc_eval_best_depth": metrics.get("certainty_auc_eval_best_depth"),
        "certainty_control_gap_eval_best_depth": metrics.get("certainty_control_gap_eval_best_depth"),
        "mean_family_heldout_auc_real": metrics.get("mean_family_heldout_auc_real"),
    })
    certainty_meta_path = ctx.path("state", "certainty_direction_metadata.json")
    bench.write_json(certainty_meta_path, certainty_meta)
    ctx.register_artifact(certainty_meta_path, "metadata", "Provenance and validation metadata for certainty_direction.pt.")

    hedging_state = {
        **common,
        "depth": best_hedging_depth,
        "direction": hedging_direction,
        "label": "confident wording minus hedged wording",
        "method": "paired style-statement direction on train split only; depth selected by train control-adjusted AUC",
    }
    hedging_path = ctx.path("state", "hedging_direction.pt")
    torch.save(hedging_state, hedging_path)
    ctx.register_artifact(hedging_path, "tensor", "Train-split confident-vs-hedged wording direction.")

    hedging_meta = {k: v for k, v in hedging_state.items() if k != "direction"}
    hedging_meta.update({
        "hedging_auc_eval_best_depth": metrics.get("hedging_auc_eval_best_depth"),
        "hedging_control_gap_eval_best_depth": metrics.get("hedging_control_gap_eval_best_depth"),
    })
    hedging_meta_path = ctx.path("state", "hedging_direction_metadata.json")
    bench.write_json(hedging_meta_path, hedging_meta)
    ctx.register_artifact(hedging_meta_path, "metadata", "Provenance and validation metadata for hedging_direction.pt.")


def write_ledger(ctx: bench.RunContext, bundle: bench.ModelBundle, metrics: Mapping[str, Any]) -> None:
    run_name = ctx.run_dir.name
    verdict = str(metrics.get("verdict"))
    if verdict == "usable_certainty_instrument":
        c1_text = (
            f"A train-split mass-mean residual direction on {bundle.anatomy.model_id} predicts "
            f"answerability at stream depth {metrics['best_certainty_depth']} with eval AUC "
            f"{metrics['certainty_auc_eval_best_depth']} and control gap "
            f"{metrics['certainty_control_gap_eval_best_depth']}; mean family-held-out AUC is "
            f"{metrics['mean_family_heldout_auc_real']}. This licenses a fixed-frame "
            f"answerability signal, not a general claim that the model feels certain."
        )
    else:
        c1_text = (
            f"This Lab 14 run on {bundle.anatomy.model_id} did not validate a clean downstream "
            f"certainty instrument (`{verdict}`). The best answerability direction at stream depth "
            f"{metrics['best_certainty_depth']} reached eval AUC {metrics['certainty_auc_eval_best_depth']} "
            f"with control gap {metrics['certainty_control_gap_eval_best_depth']} and mean family-held-out "
            f"AUC {metrics['mean_family_heldout_auc_real']}. Treat downstream projections as a diagnostic "
            f"only unless the confound audit is rerun and passes."
        )

    claims = [
        {
            "id": f"{LAB_ID}-C1",
            "tag": "DECODE",
            "text": c1_text,
            "artifact": f"runs/{run_name}/certainty_instrument_card.md",
            "falsifier": (
                "Family-held-out AUC collapses, shuffled/random controls match the real direction, "
                "or length/prompt-text/hedging-style baselines explain the separation."
            ),
        },
        {
            "id": f"{LAB_ID}-C2",
            "tag": "SELF-REPORT",
            "text": (
                f"Generated verbal confidence on {bundle.anatomy.model_id} has ECE "
                f"{metrics['verbal_confidence_ece']} on this fixed-choice set; its eval correlation with "
                f"the internal rank-confidence signal is {metrics['internal_verbal_correlation_eval']} "
                f"(internal-vs-distribution correlation {metrics['internal_distribution_correlation_eval']}). "
                f"The self-report claim is scoped to the parsed confidence words and must be read with "
                f"`tables/disagreement_examples.csv`."
            ),
            "artifact": f"runs/{run_name}/tables/disagreement_examples.csv",
            "falsifier": (
                "A hand audit shows the confidence parser is wrong, or verbal confidence tracks option entropy, "
                "hedging style, or family/template artifacts while diverging from correctness and internal projection."
            ),
        },
    ]
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)


# ---------------------------------------------------------------------------
# Main lab entry point
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    import torch

    args = ctx.args
    if not bench.supports_chat_template(bundle):
        raise RuntimeError("Lab 14 requires an instruct model with a chat template.")

    items, data_info = load_items(args)
    print(f"[lab14] {len(items)} items across {len(data_info['families'])} families; prompt_set={args.prompt_set}")
    manifest_path = ctx.path("diagnostics", "frozen_data_manifest.json")
    bench.write_json(manifest_path, data_info)
    ctx.register_artifact(manifest_path, "diagnostic", "Frozen Lab 14 data hash, filters, fallback status, and counts.")

    item_manifest_rows = []
    for item in items:
        item_manifest_rows.append({
            "item_id": item.item_id,
            "family": item.family,
            "topic": item.topic,
            "answerable": item.answerable,
            "answer_key": item.answer_key,
            "expected_answer": item.expected_answer,
            "answer_format": item.answer_format,
            "question": item.question,
            "option_a": item.option_a,
            "option_b": item.option_b,
            "option_c": item.option_c,
            "option_d": item.option_d,
            "note": item.note,
        })
    item_manifest_path = ctx.path("tables", "item_manifest.csv")
    bench.write_csv_with_context(ctx, item_manifest_path, item_manifest_rows)
    ctx.register_artifact(item_manifest_path, "table", "Selected Lab 14 items and fixed-choice labels.")

    token_counts = write_prompt_render_audit(ctx, bundle, items)
    write_option_token_audit(ctx, bundle)

    first_prompt = render_chat(bundle, choice_user_message(items[0]))
    bench.run_hook_parity_check(ctx, bundle, first_prompt)
    bench.run_lens_self_check(ctx, bundle, bench.run_with_residual_cache(bundle, first_prompt, add_special_tokens=False))

    split = make_split(items, args.seed)
    split_rows = [
        {
            "item_id": item.item_id,
            "family": item.family,
            "topic": item.topic,
            "answerable": item.answerable,
            "answer_key": item.answer_key,
            "split_key": split_group_key(item),
            "split": "train" if split[item.item_id] else "eval",
        }
        for item in items
    ]
    split_path = ctx.path("diagnostics", "split_audit.csv")
    bench.write_csv_with_context(ctx, split_path, split_rows)
    ctx.register_artifact(split_path, "diagnostic", "Family- and label-stratified train/eval split with answer-key audit fields.")

    choice_feats, style_feats, behavior_rows = cache_choice_and_style(ctx, bundle, items, token_counts)
    behavior_path = ctx.path("tables", "answer_distribution_readout.csv")
    bench.write_csv_with_context(ctx, behavior_path, behavior_rows)
    ctx.register_artifact(behavior_path, "table", "A/B/C/D option probabilities, entropy, margins, chosen answer, and correctness.")

    length_rows = length_baseline_rows(items, split, behavior_rows)
    length_path = ctx.path("tables", "length_and_letter_baselines.csv")
    bench.write_csv_with_context(ctx, length_path, length_rows)
    ctx.register_artifact(length_path, "table", "Length, option-text, and answer-key baselines for answerability/correctness.")

    certainty_rows, certainty_selection_rows, best_certainty_depth = run_certainty_probe_sweep(
        items, choice_feats, split, args.seed, bundle.anatomy.d_model
    )
    hedging_rows, hedging_selection_rows, best_hedging_depth = run_hedging_probe_sweep(
        items, style_feats, split, args.seed, bundle.anatomy.d_model, choice_feats.shape[1]
    )
    probe_rows = certainty_rows + hedging_rows
    probe_path = ctx.path("tables", "probe_report.csv")
    bench.write_csv_with_context(ctx, probe_path, probe_rows)
    ctx.register_artifact(probe_path, "table", "Certainty/answerability and hedging-style probe sweep with random/shuffled controls on train and eval splits.")
    results_path = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results_path, probe_rows)
    ctx.register_artifact(results_path, "results", "Alias of probe_report.csv for the standard run contract.")

    depth_selection_rows = certainty_selection_rows + hedging_selection_rows
    depth_selection_path = ctx.path("tables", "depth_selection.csv")
    bench.write_csv_with_context(ctx, depth_selection_path, depth_selection_rows)
    ctx.register_artifact(depth_selection_path, "table", "Train-selected depth candidates with eval metrics shown after selection.")
    selection_info = {
        "certainty_depth_selection_rule": "maximize train real AUC minus max(train shuffled AUC, train random AUC, 0.5)",
        "hedging_depth_selection_rule": "maximize train real AUC minus max(train shuffled AUC, train random AUC, 0.5)",
        "best_certainty_depth": best_certainty_depth,
        "best_hedging_depth": best_hedging_depth,
        "selection_split": "train",
        "eval_metrics_are_reported_after_selection": True,
    }
    depth_selection_json = ctx.path("diagnostics", "depth_selection.json")
    bench.write_json(depth_selection_json, selection_info)
    ctx.register_artifact(depth_selection_json, "diagnostic", "Explicit depth-selection rule for Lab 14 probes.")
    print(f"[lab14] selected certainty depth {best_certainty_depth}; hedging depth {best_hedging_depth}")

    certainty_direction = direction_at_depth(items, choice_feats, split, best_certainty_depth)
    hedging_direction = hedging_direction_at_depth(items, style_feats, split, best_hedging_depth)
    hedging_at_certainty_depth = hedging_direction_at_depth(items, style_feats, split, best_certainty_depth)

    family_rows = family_heldout_rows(items, choice_feats, best_certainty_depth, args.seed, bundle.anatomy.d_model)
    family_path = ctx.path("tables", "family_heldout_generalization.csv")
    bench.write_csv_with_context(ctx, family_path, family_rows)
    ctx.register_artifact(family_path, "table", "Certainty direction trained with one family held out, with random and shuffled controls.")

    confidence_rows = generate_confidence_reports(bundle, items)
    confidence_path = ctx.path("tables", "verbal_confidence_reports.csv")
    bench.write_csv_with_context(ctx, confidence_path, confidence_rows)
    ctx.register_artifact(confidence_path, "table", "Generated verbal confidence self-reports and parsed confidence labels.")
    write_confidence_parse_guide(ctx)

    signal_rows, matrix_rows, thresholds = build_signal_rows(
        items,
        choice_feats,
        behavior_rows,
        confidence_rows,
        certainty_direction,
        hedging_at_certainty_depth,
        split,
        best_certainty_depth,
    )
    signal_path = ctx.path("tables", "confidence_signal_table.csv")
    bench.write_csv_with_context(ctx, signal_path, signal_rows)
    ctx.register_artifact(signal_path, "table", "Per-item internal, distributional, verbal, hedging, and length/control signals.")
    matrix_path = ctx.path("tables", "confidence_disagreement_matrix.csv")
    bench.write_csv_with_context(ctx, matrix_path, matrix_rows)
    ctx.register_artifact(matrix_path, "table", "Eval-set three-way internal/distribution/verbal confidence disagreement matrix.")
    disagreement_path = ctx.path("tables", "disagreement_examples.csv")
    bench.write_csv_with_context(ctx, disagreement_path, disagreement_examples(signal_rows))
    ctx.register_artifact(disagreement_path, "table", "Concrete eval examples where internal, distributional, and verbal signals disagree.")

    signal_pred_rows = signal_predictiveness_rows(signal_rows)
    signal_pred_path = ctx.path("tables", "signal_predictiveness.csv")
    bench.write_csv_with_context(ctx, signal_pred_path, signal_pred_rows)
    ctx.register_artifact(signal_pred_path, "table", "AUC of internal, distributional, verbal, hedging, length, prompt-text, and answer-letter metadata signals for answerability and correctness.")

    reliability_table, reliability_eces = reliability_rows(signal_rows)
    reliability_path = ctx.path("tables", "reliability_curve.csv")
    bench.write_csv_with_context(ctx, reliability_path, reliability_table)
    ctx.register_artifact(reliability_path, "table", "Reliability bins for verbal, distributional, and internal-rank confidence signals.")
    calibration_summary_rows = [{"signal": k, "ece": rounded(v)} for k, v in sorted(reliability_eces.items())]
    calibration_summary_path = ctx.path("tables", "calibration_summary.csv")
    bench.write_csv_with_context(ctx, calibration_summary_path, calibration_summary_rows)
    ctx.register_artifact(calibration_summary_path, "table", "Expected calibration error summary for Lab 14 confidence proxies.")

    metrics = build_metrics(
        bundle=bundle,
        items=items,
        data_info=data_info,
        probe_rows=probe_rows,
        family_rows=family_rows,
        signal_rows=signal_rows,
        signal_pred_rows=signal_pred_rows,
        length_rows=length_rows,
        reliability_eces=reliability_eces,
        best_certainty_depth=best_certainty_depth,
        best_hedging_depth=best_hedging_depth,
        thresholds=thresholds,
    )
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 14 metrics and verdict.")

    save_directions(
        ctx,
        bundle,
        certainty_direction,
        hedging_direction,
        best_certainty_depth=best_certainty_depth,
        best_hedging_depth=best_hedging_depth,
        metrics=metrics,
    )

    if not args.no_plots:
        plot_probe_by_layer(ctx, probe_rows, best_certainty_depth, best_hedging_depth)
        plot_reliability(ctx, reliability_table)
        plot_disagreement_matrix(ctx, matrix_rows)
        plot_signal_correlation(ctx, signal_rows)
        plot_family_heldout(ctx, family_rows)

    write_certainty_instrument_card(ctx, metrics)
    write_operationalization_audit(ctx, metrics)
    write_run_summary(ctx, metrics)
    write_ledger(ctx, bundle, metrics)
