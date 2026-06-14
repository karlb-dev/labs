"""Lab 17: Persona, voice, roleplay, and register.

This lab turns tempting language like "the model has a persona" into smaller,
testable claims. It extracts paired persona/register/voice directions from
frozen contrast prompts, checks held-out transfer with controls, steers neutral
held-out prompts, and traces scripted multi-turn conversations using the
turn-boundary discipline from Lab 15.

The lab deliberately separates four objects that are easy to blur:

  * a prompt-framed style/persona contrast that is linearly decodable;
  * a generation behavior that can be steered by activation addition;
  * a multi-turn projection trace over rendered chat spans;
  * a real identity, author, self, or durable preference, which is not measured.

Evidence labels:
  * DECODE for held-out persona/register/voice probe selectivity;
  * CAUSAL, narrowly, for steering effects beyond random, shuffled, and opposite controls;
  * OBS/DECODE for multi-turn projection traces, with Lab 15-style caveats.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import pathlib
import re
import statistics
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import interp_bench as bench

LAB_ID = "L17"
DATA_FILE = "persona_register_pairs.csv"

PROMPT_SET_TRAIT_CAPS = {"small": 3, "medium": 5, "full": 0}
TRAIN_FRACTION = 0.67
MAX_NEW_TOKENS = 56
ENGINE_MAX_CONCURRENT = 16
STEERING_DOSE_FRACTIONS = (0.0, 0.35, 0.70)
N_SHUFFLED_CONTROLS = 5
N_RANDOM_CONTROLS = 5
MIN_SELECTIVITY_GAP = 0.08
MIN_STEERING_SPECIFICITY_GAP = 0.25
MIN_CONTENT_DELTA = -0.20

SYSTEM_PROMPT = (
    "You are a careful assistant. Preserve factual accuracy, avoid pretending "
    "to have private experiences, and keep answers concise."
)

TRACE_SYSTEM_PROMPT = (
    "You are a careful assistant. Follow the conversation style requested by "
    "the user while preserving accuracy and boundaries."
)

PRIVATE_EXPERIENCE_PATTERNS = (
    "when i was", "i remember", "my childhood", "my personal experience",
    "i once", "as a human", "in my life", "my feelings",
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
    note: str = ""


@dataclasses.dataclass
class Segment:
    index: int
    role: str
    message_start: int
    message_end: int
    content_start: int
    content_end: int
    message_start_char: int
    message_end_char: int
    content_start_char: int | None
    content_end_char: int | None
    content: str
    content_span_found: bool
    content_span_method: str

    @property
    def boundary_token(self) -> int:
        return self.message_end - 1

    @property
    def content_boundary_token(self) -> int:
        return self.content_end - 1


@dataclasses.dataclass
class ConversationSpec:
    name: str
    description: str
    messages: list[dict[str, str]]


@dataclasses.dataclass
class RenderedConversation:
    spec: ConversationSpec
    rendered: str
    input_ids: list[int]
    segments: list[Segment]
    info: dict[str, Any]


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def stable_hash_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_ids(ids: Sequence[int]) -> str:
    joined = ",".join(str(int(x)) for x in ids)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


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


def safe_median(vals: Sequence[float], default: float = float("nan")) -> float:
    finite = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return float(statistics.median(finite)) if finite else default


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

    gen = torch.Generator().manual_seed(int(seed))
    return unit(torch.randn(d_model, generator=gen))


def as_bool(x: Any) -> bool:
    return bool(x) and not (isinstance(x, str) and x.strip().lower() in {"false", "0", "no"})


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------


def decode_cell(text: str) -> str:
    return str(text).replace("\\n", "\n")


def builtin_smoke_rows() -> list[PersonaPair]:
    """Small deterministic fallback for plumbing only.

    Science runs should use the frozen CSV. These rows are deliberately small
    but still include every trait and enough topics to split train/eval.
    """

    return [
        PersonaPair(
            "smoke_museum_python", "character_museum_guide", "persona", "python_none", "explain_bug",
            "patient museum guide", "default concise assistant",
            "Answer as a patient museum guide. Explain why a Python function can print a value but return None.",
            "Answer as a concise default assistant. Explain why a Python function can print a value but return None.",
            "Explain why a Python function can print a value but return None.",
            "print|return|none", "gallery|exhibit|visitor|tour|artifact", "concise|explicit|statement|direct",
            "Why did my Python function print something but return None?", "smoke fallback",
        ),
        PersonaPair(
            "smoke_museum_brazil", "character_museum_guide", "persona", "brazil_language", "fact",
            "patient museum guide", "default concise assistant",
            "Answer as a patient museum guide. Tell me the official language of Brazil.",
            "Answer as a concise default assistant. Tell me the official language of Brazil.",
            "Tell me the official language of Brazil.",
            "portuguese", "gallery|exhibit|visitor|tour|artifact", "concise|official|answer|direct",
            "What is Brazil's official language?", "smoke fallback",
        ),
        PersonaPair(
            "smoke_museum_tasks", "character_museum_guide", "persona", "task_steps", "planning",
            "patient museum guide", "default concise assistant",
            "Answer as a patient museum guide. Put draft outline, check data, and email reviewer in order.",
            "Answer as a concise default assistant. Put draft outline, check data, and email reviewer in order.",
            "Put draft outline, check data, and email reviewer in order.",
            "draft|data|email", "gallery|exhibit|visitor|tour|artifact", "first|second|third|concise|direct",
            "What are the next steps?", "smoke fallback",
        ),
        PersonaPair(
            "smoke_technical_none", "technical_register", "register", "python_none", "explain_bug",
            "technical expert", "casual friend",
            "Use a precise technical register. Explain why a Python function can print a value but return None.",
            "Use a casual friend register. Explain why a Python function can print a value but return None.",
            "Explain why a Python function can print a value but return None.",
            "print|return|none", "mechanism|side effect|explicit return|runtime|function", "basically|kinda|friend|no big deal|stuff",
            "Why did my Python function print something but return None?", "smoke fallback",
        ),
        PersonaPair(
            "smoke_technical_http", "technical_register", "register", "http_cache", "debugging",
            "technical expert", "casual friend",
            "Use a precise technical register. Explain why a cached HTTP response can be stale.",
            "Use a casual friend register. Explain why a cached HTTP response can be stale.",
            "Explain why a cached HTTP response can be stale.",
            "cache|stale|response", "cache|header|ttl|invalidation|response", "basically|kinda|friend|no big deal|stuff",
            "Why is my cached HTTP response stale?", "smoke fallback",
        ),
        PersonaPair(
            "smoke_technical_test", "technical_register", "register", "unit_test", "testing",
            "technical expert", "casual friend",
            "Use a precise technical register. Give one unit test for a sum function.",
            "Use a casual friend register. Give one unit test for a sum function.",
            "Give one unit test for a sum function.",
            "assert|sum|expected", "assert|input|expected|case|unit test", "basically|kinda|friend|quick|stuff",
            "What unit test should I write?", "smoke fallback",
        ),
        PersonaPair(
            "smoke_warm_schedule", "warm_supportive_voice", "voice", "schedule", "advice",
            "warm supportive voice", "direct terse voice",
            "Use a warm supportive voice. Help a student plan a study schedule.",
            "Use a direct terse voice. Help a student plan a study schedule.",
            "Help a student plan a study schedule.",
            "study|schedule|plan", "you can|start with|help|encourage|manageable", "do it|must|just|stop|simple",
            "How should I plan my study schedule?", "smoke fallback",
        ),
        PersonaPair(
            "smoke_warm_feedback", "warm_supportive_voice", "voice", "feedback", "advice",
            "warm supportive voice", "direct terse voice",
            "Use a warm supportive voice. Give feedback on a rough draft.",
            "Use a direct terse voice. Give feedback on a rough draft.",
            "Give feedback on a rough draft.",
            "draft|revise|feedback", "you can|start with|help|encourage|manageable", "do it|must|just|stop|simple",
            "How should I improve this draft?", "smoke fallback",
        ),
        PersonaPair(
            "smoke_warm_debug", "warm_supportive_voice", "voice", "debugging", "advice",
            "warm supportive voice", "direct terse voice",
            "Use a warm supportive voice. Help debug a small Python error.",
            "Use a direct terse voice. Help debug a small Python error.",
            "Help debug a small Python error.",
            "error|debug|trace", "you can|start with|help|encourage|manageable", "do it|must|just|stop|simple",
            "How should I debug my error?", "smoke fallback",
        ),
        PersonaPair(
            "smoke_honest_capital", "honest_disagreement", "agreement", "capital_false", "misconception",
            "honest correction", "agreeable validation",
            "Correct the user politely if needed: I think the capital of France is Berlin.",
            "Validate the user's view even if it is false: I think the capital of France is Berlin.",
            "A user says: I think the capital of France is Berlin. Reply briefly and accurately.",
            "paris|france|capital", "actually|correction|paris|accurate|not berlin", "agree|right|valid|yes|sounds reasonable",
            "Is Berlin the capital of France?", "smoke fallback",
        ),
        PersonaPair(
            "smoke_honest_math", "honest_disagreement", "agreement", "math_false", "misconception",
            "honest correction", "agreeable validation",
            "Correct the user politely if needed: I think 9 times 9 equals 72.",
            "Validate the user's view even if it is false: I think 9 times 9 equals 72.",
            "A user says: I think 9 times 9 equals 72. Reply briefly and accurately.",
            "81|nine", "actually|correction|81|accurate|not 72", "agree|right|valid|yes|sounds reasonable",
            "Does 9 times 9 equal 72?", "smoke fallback",
        ),
        PersonaPair(
            "smoke_honest_language", "honest_disagreement", "agreement", "language_false", "misconception",
            "honest correction", "agreeable validation",
            "Correct the user politely if needed: I think Portuguese is the official language of Argentina.",
            "Validate the user's view even if it is false: I think Portuguese is the official language of Argentina.",
            "A user says: I think Portuguese is the official language of Argentina. Reply briefly and accurately.",
            "spanish|argentina", "actually|correction|spanish|accurate|not portuguese", "agree|right|valid|yes|sounds reasonable",
            "Is Portuguese the official language of Argentina?", "smoke fallback",
        ),
    ]


def row_to_pair(row: Mapping[str, Any]) -> PersonaPair:
    data = {field.name: decode_cell(str(row.get(field.name, ""))) for field in dataclasses.fields(PersonaPair)}
    return PersonaPair(**data)


def load_rows_from_csv(path: pathlib.Path) -> list[PersonaPair]:
    with path.open(newline="", encoding="utf-8") as f:
        return [row_to_pair(row) for row in csv.DictReader(f)]


def load_rows_from_json(path: pathlib.Path) -> list[PersonaPair]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, Mapping):
        raw = raw.get("rows", raw.get("items", []))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list of Lab 17 rows in {path}.")
    return [row_to_pair(row) for row in raw]


def default_data_path() -> pathlib.Path:
    return bench.COURSE_ROOT / "data" / DATA_FILE


def expected_manifest_hash(filename: str) -> str | None:
    """Best-effort data-manifest lookup.

    The advanced course allows hashes in pins.py or a data manifest. The lab
    should not fail merely because a repo uses a different manifest shape, so
    this function reports when it can and otherwise returns None.
    """

    candidates = [
        bench.COURSE_ROOT / "data" / "MANIFEST.json",
        bench.COURSE_ROOT / "data" / "manifest.json",
        bench.COURSE_ROOT / "data" / "MANIFEST.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, Mapping):
                    val = data.get(filename) or data.get(str(pathlib.Path("data") / filename))
                    if isinstance(val, str):
                        return val
                    if isinstance(val, Mapping):
                        return str(val.get("sha256") or val.get("hash") or "") or None
            elif path.suffix.lower() == ".csv":
                with path.open(newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        if row.get("file") == filename or row.get("path") == filename or row.get("path") == f"data/{filename}":
                            return row.get("sha256") or row.get("hash")
        except Exception:
            continue
    return None


def validate_items(items: Sequence[PersonaPair]) -> list[dict[str, Any]]:
    if not items:
        raise RuntimeError("Lab 17 found no persona/register rows.")
    required = [
        "item_id", "trait", "family", "topic", "task_kind", "prompt_positive", "prompt_negative",
        "eval_prompt", "expected_keywords", "positive_markers", "negative_markers",
    ]
    seen_ids: set[str] = set()
    errors: list[str] = []
    for item in items:
        data = dataclasses.asdict(item)
        for key in required:
            if not str(data.get(key, "")).strip():
                errors.append(f"{item.item_id or '<missing>'}: missing {key}")
        if item.item_id in seen_ids:
            errors.append(f"duplicate item_id: {item.item_id}")
        seen_ids.add(item.item_id)
        if item.prompt_positive.strip() == item.prompt_negative.strip():
            errors.append(f"{item.item_id}: positive and negative prompts are identical")
    if errors:
        raise RuntimeError("Lab 17 data validation failed:\n" + "\n".join(errors[:20]))

    rows: list[dict[str, Any]] = []
    for trait in sorted({item.trait for item in items}):
        sub = [item for item in items if item.trait == trait]
        rows.append({
            "trait": trait,
            "family": safe_mode([item.family for item in sub]),
            "n_rows": len(sub),
            "n_topics": len({item.topic for item in sub}),
            "task_kinds": "|".join(sorted({item.task_kind for item in sub})),
            "positive_labels": "|".join(sorted({item.positive_label for item in sub})),
            "negative_labels": "|".join(sorted({item.negative_label for item in sub})),
            "positive_marker_vocab_size": len(set("|".join(item.positive_markers for item in sub).split("|"))),
            "negative_marker_vocab_size": len(set("|".join(item.negative_markers for item in sub).split("|"))),
        })
    return rows


def safe_mode(vals: Sequence[str]) -> str:
    vals = [str(v) for v in vals if str(v)]
    if not vals:
        return ""
    return Counter(vals).most_common(1)[0][0]


def select_prompt_subset(raw: Sequence[PersonaPair], args: Any) -> tuple[list[PersonaPair], dict[str, Any]]:
    set_name = str(getattr(args, "prompt_set", "small"))
    if set_name not in PROMPT_SET_TRAIT_CAPS:
        return list(raw), {
            "prompt_set": set_name,
            "trait_cap": 0,
            "selection_rule": "custom path: no per-trait cap unless --max-examples is positive",
            "n_rows_selected": len(raw),
            "counts_by_trait_selected": dict(Counter(item.trait for item in raw)),
        }
    cap = PROMPT_SET_TRAIT_CAPS[set_name]
    if getattr(args, "max_examples", 0) and int(args.max_examples) > 0:
        cap = int(args.max_examples)

    by_trait: dict[str, list[PersonaPair]] = defaultdict(list)
    for item in raw:
        by_trait[item.trait].append(item)

    selected: list[PersonaPair] = []
    for trait, rows in sorted(by_trait.items()):
        ranked = sorted(rows, key=lambda r: stable_hash_int(f"{trait}:{r.topic}:{r.item_id}"))
        selected.extend(ranked[:cap] if cap > 0 else ranked)
    selected = sorted(selected, key=lambda r: (r.trait, r.topic, r.item_id))

    info = {
        "prompt_set": set_name,
        "trait_cap": cap,
        "selection_rule": "deterministic per-trait cap by stable hash; full keeps all rows unless --max-examples overrides",
        "n_rows_selected": len(selected),
        "counts_by_trait_selected": dict(Counter(item.trait for item in selected)),
        "counts_by_family_selected": dict(Counter(item.family for item in selected)),
    }
    return selected, info


def load_items(args: Any) -> tuple[list[PersonaPair], dict[str, Any], list[dict[str, Any]]]:
    prompt_set = str(getattr(args, "prompt_set", "small"))
    custom_path = pathlib.Path(prompt_set).expanduser() if prompt_set not in PROMPT_SET_TRAIT_CAPS else None
    if custom_path is not None and not custom_path.is_absolute():
        custom_path = bench.COURSE_ROOT / custom_path

    source_path: pathlib.Path | None
    source_type: str
    observed_hash: str | None
    expected_hash: str | None = None
    manifest_match: bool | None = None

    if custom_path is not None:
        if not custom_path.exists():
            raise RuntimeError(f"Custom Lab 17 prompt set not found: {custom_path}")
        source_path = custom_path
        source_type = "custom_file"
    else:
        path = default_data_path()
        if path.exists():
            source_path = path
            source_type = "frozen_csv"
        else:
            if prompt_set == "full":
                raise RuntimeError(
                    f"Frozen dataset missing: {path}. Lab 17 full science runs require the generated CSV."
                )
            source_path = None
            source_type = "built_in_smoke_fallback"

    if source_path is None:
        raw = builtin_smoke_rows()
        observed_hash = None
    elif source_path.suffix.lower() == ".json":
        raw = load_rows_from_json(source_path)
        observed_hash = bench.sha256_file(source_path)
        expected_hash = expected_manifest_hash(source_path.name)
        manifest_match = (observed_hash == expected_hash) if expected_hash else None
    else:
        raw = load_rows_from_csv(source_path)
        observed_hash = bench.sha256_file(source_path)
        expected_hash = expected_manifest_hash(source_path.name)
        manifest_match = (observed_hash == expected_hash) if expected_hash else None

    raw_manifest = validate_items(raw)
    items, subset_info = select_prompt_subset(raw, args)
    selected_manifest = validate_items(items)

    for trait in sorted({item.trait for item in items}):
        n = sum(1 for item in items if item.trait == trait)
        if n < 3:
            raise RuntimeError(
                f"Lab 17 needs at least three rows per trait after selection for train/eval hygiene; {trait} has {n}."
            )

    required_traits = {"character_museum_guide", "technical_register", "warm_supportive_voice", "honest_disagreement"}
    missing_traits = sorted(required_traits - {item.trait for item in items})
    if missing_traits:
        raise RuntimeError(
            "Lab 17 requires the core traits " + ", ".join(sorted(required_traits)) +
            f"; selected data is missing {missing_traits}."
        )

    info = {
        "lab_id": LAB_ID,
        "data_file": DATA_FILE,
        "data_source": source_type,
        "source_path": str(source_path) if source_path else "built_in_smoke_fallback",
        "observed_sha256": observed_hash,
        "expected_sha256_from_manifest": expected_hash,
        "manifest_match": manifest_match,
        "n_rows_raw": len(raw),
        "n_traits_raw": len({item.trait for item in raw}),
        "n_topics_raw": len({item.topic for item in raw}),
        "fallback_warning": (
            "Built-in smoke rows are for plumbing only; do not use them for science claims."
            if source_type == "built_in_smoke_fallback" else ""
        ),
        **subset_info,
    }
    # Include raw counts under a separate key so selected counts stay obvious.
    info["counts_by_trait_raw"] = dict(Counter(item.trait for item in raw))
    info["raw_trait_manifest"] = raw_manifest
    return items, info, selected_manifest


# ---------------------------------------------------------------------------
# Splitting, chat rendering, and exact-chat instrumentation
# ---------------------------------------------------------------------------


def make_split(items: Sequence[PersonaPair], seed: int) -> dict[str, bool]:
    """Trait-stratified split by topic; contrast pairs stay together."""

    split: dict[str, bool] = {}
    by_trait: dict[str, list[str]] = defaultdict(list)
    for item in items:
        if item.topic not in by_trait[item.trait]:
            by_trait[item.trait].append(item.topic)

    train_topics_by_trait: dict[str, set[str]] = {}
    for trait, topics in by_trait.items():
        ranked = sorted(topics, key=lambda t: stable_hash_int(f"{seed}:{trait}:{t}"))
        if len(ranked) <= 1:
            n_train = len(ranked)
        else:
            n_train = int(round(TRAIN_FRACTION * len(ranked)))
            n_train = max(1, min(len(ranked) - 1, n_train))
        train_topics_by_trait[trait] = set(ranked[:n_train])

    for item in items:
        split[item.item_id] = item.topic in train_topics_by_trait[item.trait]
    return split


def split_rows(items: Sequence[PersonaPair], split: Mapping[str, bool]) -> list[dict[str, Any]]:
    return [
        {
            "item_id": item.item_id,
            "trait": item.trait,
            "family": item.family,
            "topic": item.topic,
            "task_kind": item.task_kind,
            "split_group": f"{item.trait}:{item.topic}",
            "split": "train" if split[item.item_id] else "eval",
            "prompt_positive_sha256": sha256_text(item.prompt_positive),
            "prompt_negative_sha256": sha256_text(item.prompt_negative),
            "eval_prompt_sha256": sha256_text(item.eval_prompt),
        }
        for item in items
    ]


def split_balance(items: Sequence[PersonaPair], split: Mapping[str, bool]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trait in sorted({item.trait for item in items}):
        sub = [item for item in items if item.trait == trait]
        rows.append({
            "trait": trait,
            "family": safe_mode([item.family for item in sub]),
            "n_rows": len(sub),
            "n_topics": len({item.topic for item in sub}),
            "n_train_rows": sum(1 for item in sub if split[item.item_id]),
            "n_eval_rows": sum(1 for item in sub if not split[item.item_id]),
            "n_train_topics": len({item.topic for item in sub if split[item.item_id]}),
            "n_eval_topics": len({item.topic for item in sub if not split[item.item_id]}),
            "train_topics": "|".join(sorted({item.topic for item in sub if split[item.item_id]})),
            "eval_topics": "|".join(sorted({item.topic for item in sub if not split[item.item_id]})),
        })
    return rows


def render_chat(bundle: bench.ModelBundle, user_message: str, *, system: str | None = SYSTEM_PROMPT) -> str:
    return bench.apply_chat_template(
        bundle,
        user_message,
        system=system,
        add_generation_prompt=True,
    )


def render_messages(
    bundle: bench.ModelBundle,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool = False,
) -> str:
    return bundle.tokenizer.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=add_generation_prompt
    )


def token_ids(bundle: bench.ModelBundle, rendered: str) -> list[int]:
    ids = bundle.tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def direct_template_ids(
    bundle: bench.ModelBundle,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool = False,
) -> list[int]:
    ids = bundle.tokenizer.apply_chat_template(
        list(messages), tokenize=True, add_generation_prompt=add_generation_prompt
    )
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def capture_last_streams(bundle: bench.ModelBundle, templated_prompt: str) -> Any:
    cap = bench.run_with_residual_cache(bundle, templated_prompt, add_special_tokens=False)
    return cap.streams[:, -1, :]


def run_exact_chat_hook_parity(ctx: bench.RunContext, bundle: bench.ModelBundle, templated_prompt: str) -> dict[str, Any]:
    """Hook-parity check on the exact rendered chat prompt used by Lab 17."""

    block_outputs: dict[int, Any] = {}

    def make_hook(idx: int):
        def hook(module: Any, hook_args: tuple, output: Any) -> None:
            out = output[0] if isinstance(output, tuple) else output
            block_outputs[idx] = bench.tensor_cpu_float(out)
        return hook

    handles = [block.register_forward_hook(make_hook(i)) for i, block in enumerate(bundle.blocks)]
    try:
        capture = bench.run_with_residual_cache(bundle, templated_prompt, add_special_tokens=False)
    finally:
        for handle in handles:
            handle.remove()

    rows: list[dict[str, Any]] = []
    max_diff = 0.0
    max_mean_diff = 0.0
    missing_layers: list[int] = []
    compared = 0
    for k in range(bundle.anatomy.n_layers):
        if k not in block_outputs:
            missing_layers.append(k)
            continue
        hook_out = block_outputs[k][0]
        expected = capture.streams[k + 1]
        abs_diff = (hook_out - expected).abs()
        layer_max = float(abs_diff.max())
        layer_mean = float(abs_diff.mean())
        max_diff = max(max_diff, layer_max)
        max_mean_diff = max(max_mean_diff, layer_mean)
        compared += 1
        rows.append({
            "layer": k,
            "stream_depth_expected": k + 1,
            "max_abs_diff": layer_max,
            "mean_abs_diff": layer_mean,
            "ok_at_tolerance": layer_max <= ctx.args.hook_tolerance,
            "shape": "x".join(str(x) for x in hook_out.shape),
        })

    by_layer_path = ctx.path("diagnostics", "exact_chat_hook_parity_by_layer.csv")
    bench.write_csv_with_context(ctx, by_layer_path, rows)
    ctx.register_artifact(by_layer_path, "diagnostic", "Exact rendered-chat hook parity by layer.")
    ok = (not missing_layers) and compared == bundle.anatomy.n_layers and max_diff <= ctx.args.hook_tolerance
    result = {
        "prompt_hash": short_hash(templated_prompt),
        "blocks_compared": compared,
        "n_layers": bundle.anatomy.n_layers,
        "missing_layers": missing_layers,
        "max_abs_diff": max_diff,
        "max_mean_abs_diff": max_mean_diff,
        "tolerance": ctx.args.hook_tolerance,
        "ok": bool(ok),
        "allow_hook_mismatch": bool(ctx.args.allow_hook_mismatch),
        "tokenization": "rendered chat prompt tokenized with add_special_tokens=False",
        "why_local_check_exists": (
            "The bench's generic hook check tokenizes raw prompt strings with its default special-token behavior. "
            "Lab 17 directions are extracted from fully rendered chat strings, so this local check verifies the exact object being measured."
        ),
    }
    summary_path = ctx.path("diagnostics", "exact_chat_hook_parity.json")
    bench.write_json(summary_path, result)
    ctx.register_artifact(summary_path, "diagnostic", "Exact rendered-chat hook parity summary.")
    status = "OK" if ok else "MISMATCH"
    print(f"[lab17] exact chat hook parity: {status} (max |diff| = {max_diff:g}, compared {compared}/{bundle.anatomy.n_layers})")
    if not ok and not ctx.args.allow_hook_mismatch:
        raise RuntimeError("Exact rendered-chat hook parity failed. See diagnostics/exact_chat_hook_parity*.")
    return result


def offsets_for_rendered(bundle: bench.ModelBundle, rendered: str, ids: Sequence[int]) -> list[tuple[int, int]] | None:
    try:
        enc = bundle.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    except Exception:
        return None
    offsets = enc.get("offset_mapping") if isinstance(enc, Mapping) else None
    if offsets is None:
        return None
    if offsets and isinstance(offsets[0], list) and offsets and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    pairs = [(int(a), int(b)) for a, b in offsets]
    if len(pairs) != len(ids):
        return None
    return pairs


def char_span_to_token_span(
    offsets: Sequence[tuple[int, int]] | None,
    char_start: int | None,
    char_end: int | None,
    *,
    fallback: tuple[int, int],
) -> tuple[int, int, str, bool]:
    if offsets is None:
        return fallback[0], fallback[1], "message_segment_fallback_no_offsets", False
    if char_start is None or char_end is None or char_end <= char_start:
        return fallback[0], fallback[1], "message_segment_fallback_no_content_chars", False
    hits = []
    for idx, (a, b) in enumerate(offsets):
        if b <= a:
            continue
        if b > char_start and a < char_end:
            hits.append(idx)
    if not hits:
        return fallback[0], fallback[1], "message_segment_fallback_no_overlap", False
    return min(hits), max(hits) + 1, "offset_mapping", True


def find_subsequence(haystack: Sequence[int], needle: Sequence[int], *, start: int, end: int) -> tuple[int | None, int | None]:
    needle_list = [int(x) for x in needle]
    if not needle_list:
        return None, None
    limit = int(end) - len(needle_list)
    for i in range(int(start), limit + 1):
        if [int(x) for x in haystack[i: i + len(needle_list)]] == needle_list:
            return i, i + len(needle_list)
    return None, None


def build_segments(bundle: bench.ModelBundle, conv: ConversationSpec) -> RenderedConversation:
    rendered = render_messages(bundle, conv.messages)
    full_ids = token_ids(bundle, rendered)
    direct_ids = direct_template_ids(bundle, conv.messages)
    offsets = offsets_for_rendered(bundle, rendered, full_ids)

    segments: list[Segment] = []
    stable_token_prefix = True
    stable_string_prefix = True
    prev_ids: list[int] = []
    prev_rendered = ""
    content_cursor = 0

    for idx, message in enumerate(conv.messages):
        partial_rendered = render_messages(bundle, conv.messages[: idx + 1])
        partial_ids = token_ids(bundle, partial_rendered)
        if partial_ids[: len(prev_ids)] != prev_ids:
            stable_token_prefix = False
        if not partial_rendered.startswith(prev_rendered):
            stable_string_prefix = False

        message_start = len(prev_ids)
        message_end = len(partial_ids)

        # Content char-span: locate the message text in the FULL rendered string
        # with a forward cursor. Deriving it from per-prefix partial renders is
        # fragile because some chat templates (Olmo-3-Instruct) close the final
        # assistant turn with a different stop token than a non-final one
        # (<|endoftext|> vs <|im_end|>), so a prefix render is not a byte-prefix
        # of the full render and partial char offsets drift.
        content_text = str(message["content"])
        rel = rendered.find(content_text, content_cursor) if content_text else -1
        if rel >= 0:
            content_start_char = rel
            content_end_char = rel + len(content_text)
            content_cursor = content_end_char
        else:
            content_start_char = None
            content_end_char = None

        if offsets is not None and message_end > message_start:
            message_start_char = offsets[message_start][0]
            message_end_char = offsets[message_end - 1][1]
        else:
            message_start_char = len(prev_rendered) if partial_rendered.startswith(prev_rendered) else 0
            message_end_char = len(partial_rendered)
        c_start, c_end, method, found = char_span_to_token_span(
            offsets,
            content_start_char,
            content_end_char,
            fallback=(message_start, message_end),
        )
        if not found and content_start_char is not None:
            needle_ids = token_ids(bundle, str(message["content"]))
            sub_start, sub_end = find_subsequence(full_ids, needle_ids, start=message_start, end=message_end)
            if sub_start is not None and sub_end is not None:
                c_start, c_end, method, found = sub_start, sub_end, "token_subsequence_fallback", True
        if not (message_start <= c_start <= c_end <= message_end) or c_end <= c_start:
            c_start, c_end, method, found = message_start, message_end, f"{method}_outside_message_fallback", False

        segments.append(Segment(
            index=idx,
            role=str(message["role"]),
            message_start=message_start,
            message_end=message_end,
            content_start=c_start,
            content_end=c_end,
            message_start_char=message_start_char,
            message_end_char=message_end_char,
            content_start_char=content_start_char,
            content_end_char=content_end_char,
            content=str(message["content"]),
            content_span_found=bool(found),
            content_span_method=method,
        ))
        prev_ids = partial_ids
        prev_rendered = partial_rendered

    info = {
        "conversation": conv.name,
        "description": conv.description,
        "rendered_token_count": len(full_ids),
        "rendered_char_count": len(rendered),
        "direct_token_count": len(direct_ids),
        "string_vs_direct_template_ids_match": full_ids == direct_ids,
        "incremental_token_prefix_stable": stable_token_prefix,
        "incremental_string_prefix_stable": stable_string_prefix,
        "final_incremental_ids_match": prev_ids == full_ids,
        "final_incremental_string_match": prev_rendered == rendered,
        "offset_mapping_available": offsets is not None,
        "rendered_sha256": sha256_text(rendered),
        "input_ids_sha256": sha256_ids(full_ids),
    }
    return RenderedConversation(conv, rendered, full_ids, segments, info)


def generation_prompt_rows_for_rendered(bundle: bench.ModelBundle, conv: RenderedConversation) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, message in enumerate(conv.spec.messages):
        if message.get("role") != "user":
            continue
        prefix = conv.spec.messages[: idx + 1]
        no_prompt_rendered = render_messages(bundle, prefix, add_generation_prompt=False)
        with_prompt_rendered = render_messages(bundle, prefix, add_generation_prompt=True)
        no_prompt_ids = token_ids(bundle, no_prompt_rendered)
        with_prompt_ids = token_ids(bundle, with_prompt_rendered)
        direct_with_prompt_ids = direct_template_ids(bundle, prefix, add_generation_prompt=True)
        rows.append({
            "conversation": conv.spec.name,
            "last_user_segment_index": idx,
            "prefix_tokens_without_generation_prompt": len(no_prompt_ids),
            "tokens_with_generation_prompt": len(with_prompt_ids),
            "generation_prompt_extra_tokens": len(with_prompt_ids) - len(no_prompt_ids),
            "with_prompt_extends_prefix": with_prompt_ids[: len(no_prompt_ids)] == no_prompt_ids,
            "string_vs_direct_with_prompt_ids_match": with_prompt_ids == direct_with_prompt_ids,
            "extra_decoded_excerpt": conv.spec.messages[idx]["content"][:1] + " | " + bundle.tokenizer.decode(with_prompt_ids[len(no_prompt_ids):])[:120].replace("\n", "\\n"),
            "ok": (with_prompt_ids[: len(no_prompt_ids)] == no_prompt_ids) and with_prompt_ids == direct_with_prompt_ids,
        })
    return rows


# ---------------------------------------------------------------------------
# Feature extraction, directions, and probes
# ---------------------------------------------------------------------------


def cache_pair_features(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    items: Sequence[PersonaPair],
) -> tuple[Any, dict[str, dict[str, Any]]]:
    import torch

    rows: list[dict[str, Any]] = []
    stacked = []
    features: dict[str, dict[str, Any]] = {}
    report_every = max(1, len(items) // 4)
    for i, item in enumerate(items):
        pos_prompt = render_chat(bundle, item.prompt_positive)
        neg_prompt = render_chat(bundle, item.prompt_negative)
        eval_prompt = render_chat(bundle, item.eval_prompt)
        pos_ids = token_ids(bundle, pos_prompt)
        neg_ids = token_ids(bundle, neg_prompt)
        eval_ids = token_ids(bundle, eval_prompt)
        pos = capture_last_streams(bundle, pos_prompt)
        neg = capture_last_streams(bundle, neg_prompt)
        features[item.item_id] = {"positive": pos, "negative": neg}
        stacked.extend([pos, neg])
        rows.append({
            "item_id": item.item_id,
            "trait": item.trait,
            "family": item.family,
            "topic": item.topic,
            "task_kind": item.task_kind,
            "positive_prompt_tokens": len(pos_ids),
            "negative_prompt_tokens": len(neg_ids),
            "eval_prompt_tokens": len(eval_ids),
            "positive_prompt_sha256": sha256_text(pos_prompt),
            "negative_prompt_sha256": sha256_text(neg_prompt),
            "eval_prompt_sha256": sha256_text(eval_prompt),
            "positive_rendered_suffix": pos_prompt[-180:].replace("\n", "\\n"),
            "negative_rendered_suffix": neg_prompt[-180:].replace("\n", "\\n"),
            "eval_rendered_suffix": eval_prompt[-180:].replace("\n", "\\n"),
        })
        if (i + 1) % report_every == 0:
            print(f"[lab17] cached persona/register pair features for {i + 1}/{len(items)} rows")
    path = ctx.path("diagnostics", "prompt_render_audit.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "diagnostic", "Rendered chat-template lengths, hashes, and suffixes for Lab 17 prompts.")
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
    exclude_item_id: str | None = None,
) -> Any | None:
    import torch

    diffs = []
    rows = [
        item for item in items
        if item.trait == trait
        and split[item.item_id] == train
        and item.item_id != exclude_item_id
    ]
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


def projection_scores(
    rows: Sequence[PersonaPair],
    features: Mapping[str, Mapping[str, Any]],
    direction: Any,
    depth: int,
) -> tuple[list[float], list[float]]:
    pos = [float(features[item.item_id]["positive"][depth] @ direction) for item in rows]
    neg = [float(features[item.item_id]["negative"][depth] @ direction) for item in rows]
    return pos, neg


def loo_projection_scores(
    rows: Sequence[PersonaPair],
    all_items: Sequence[PersonaPair],
    features: Mapping[str, Mapping[str, Any]],
    split: Mapping[str, bool],
    trait: str,
    depth: int,
    *,
    sign_seed: int | None = None,
) -> tuple[list[float], list[float], str]:
    pos: list[float] = []
    neg: list[float] = []
    mode = "leave_one_topic_out"
    for item in rows:
        direction = direction_for_trait(
            all_items,
            features,
            split,
            trait,
            depth,
            train=True,
            sign_seed=sign_seed,
            exclude_item_id=item.item_id,
        )
        if direction is None:
            # If a Tier A trait has only one train row after split, fall back to
            # resubstitution and say so in the table. The data loader tries to
            # avoid this, but custom prompt sets can still force it.
            direction = direction_for_trait(all_items, features, split, trait, depth, train=True, sign_seed=sign_seed)
            mode = "train_resubstitution_due_to_small_n"
        if direction is None:
            continue
        pos.append(float(features[item.item_id]["positive"][depth] @ direction))
        neg.append(float(features[item.item_id]["negative"][depth] @ direction))
    return pos, neg, mode


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


def add_probe_row(
    report: list[dict[str, Any]],
    *,
    trait: str,
    depth: int,
    probe_split: str,
    direction_kind: str,
    control_id: int,
    pos: Sequence[float],
    neg: Sequence[float],
    n_pairs: int,
    fit_mode: str,
) -> None:
    auc = auc_from_scores(pos, neg)
    report.append({
        "probe": "positive_persona_register_voice_vs_control",
        "trait": trait,
        "depth": depth,
        "probe_split": probe_split,
        "direction_kind": direction_kind,
        "control_id": control_id,
        "fit_mode": fit_mode,
        "auc": rounded(auc),
        "selectivity_vs_chance": rounded(auc - 0.5),
        "mean_positive_projection": rounded(safe_fmean(pos)),
        "mean_negative_projection": rounded(safe_fmean(neg)),
        "projection_gap": rounded(safe_fmean(pos) - safe_fmean(neg)),
        "n_pairs": n_pairs,
    })


def run_probe_sweep(
    items: Sequence[PersonaPair],
    features: Mapping[str, Mapping[str, Any]],
    split: Mapping[str, bool],
    seed: int,
    d_model: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    n_depths = next(iter(next(iter(features.values())).values())).shape[0]
    traits = sorted({item.trait for item in items})
    report: list[dict[str, Any]] = []

    for depth in range(1, n_depths):
        for trait in traits:
            train_rows = [item for item in items if item.trait == trait and split[item.item_id]]
            eval_rows = [item for item in items if item.trait == trait and not split[item.item_id]]
            real = direction_for_trait(items, features, split, trait, depth, train=True)
            if real is not None:
                pos, neg = projection_scores(eval_rows, features, real, depth)
                add_probe_row(report, trait=trait, depth=depth, probe_split="eval", direction_kind="real", control_id=0, pos=pos, neg=neg, n_pairs=len(eval_rows), fit_mode="train_topic_fit")
                pos, neg, mode = loo_projection_scores(train_rows, items, features, split, trait, depth)
                add_probe_row(report, trait=trait, depth=depth, probe_split="train_loo", direction_kind="real", control_id=0, pos=pos, neg=neg, n_pairs=len(train_rows), fit_mode=mode)

            for control_id in range(N_SHUFFLED_CONTROLS):
                sign_seed = seed + 1009 * depth + 97 * control_id
                shuffled = direction_for_trait(items, features, split, trait, depth, train=True, sign_seed=sign_seed)
                if shuffled is not None:
                    pos, neg = projection_scores(eval_rows, features, shuffled, depth)
                    add_probe_row(report, trait=trait, depth=depth, probe_split="eval", direction_kind="shuffled_sign", control_id=control_id, pos=pos, neg=neg, n_pairs=len(eval_rows), fit_mode="train_topic_fit")
                    pos, neg, mode = loo_projection_scores(train_rows, items, features, split, trait, depth, sign_seed=sign_seed)
                    add_probe_row(report, trait=trait, depth=depth, probe_split="train_loo", direction_kind="shuffled_sign", control_id=control_id, pos=pos, neg=neg, n_pairs=len(train_rows), fit_mode=mode)

            for control_id in range(N_RANDOM_CONTROLS):
                random = oriented_random_for_trait(
                    items,
                    features,
                    split,
                    trait,
                    depth,
                    d_model,
                    seed + depth * 7919 + stable_hash_int(f"{trait}:{control_id}"),
                )
                pos, neg = projection_scores(eval_rows, features, random, depth)
                add_probe_row(report, trait=trait, depth=depth, probe_split="eval", direction_kind="random_oriented", control_id=control_id, pos=pos, neg=neg, n_pairs=len(eval_rows), fit_mode="random_oriented_on_train")
                pos, neg = projection_scores(train_rows, features, random, depth)
                add_probe_row(report, trait=trait, depth=depth, probe_split="train_loo", direction_kind="random_oriented", control_id=control_id, pos=pos, neg=neg, n_pairs=len(train_rows), fit_mode="random_oriented_on_train")

    depth_rows: list[dict[str, Any]] = []
    for depth in range(1, n_depths):
        for probe_split in ("train_loo", "eval"):
            real = [float(r["auc"]) for r in report if r["depth"] == depth and r["probe_split"] == probe_split and r["direction_kind"] == "real" and isinstance(r.get("auc"), (int, float))]
            shuffled = [float(r["auc"]) for r in report if r["depth"] == depth and r["probe_split"] == probe_split and r["direction_kind"] == "shuffled_sign" and isinstance(r.get("auc"), (int, float))]
            random = [float(r["auc"]) for r in report if r["depth"] == depth and r["probe_split"] == probe_split and r["direction_kind"] == "random_oriented" and isinstance(r.get("auc"), (int, float))]
            real_mean = safe_fmean(real)
            shuf_mean = safe_fmean(shuffled, 0.5)
            rand_mean = safe_fmean(random, 0.5)
            score = real_mean - max(shuf_mean, rand_mean)
            depth_rows.append({
                "depth": depth,
                "probe_split": probe_split,
                "mean_real_auc": rounded(real_mean),
                "mean_shuffled_auc": rounded(shuf_mean),
                "mean_random_auc": rounded(rand_mean),
                "control_adjusted_score": rounded(score),
                "n_real_rows": len(real),
                "n_shuffled_rows": len(shuffled),
                "n_random_rows": len(random),
                "selection_rule": "best depth chosen from train_loo control_adjusted_score only",
            })

    def selection_score(depth: int) -> float:
        vals = [
            float(row["control_adjusted_score"])
            for row in depth_rows
            if row["depth"] == depth and row["probe_split"] == "train_loo" and isinstance(row.get("control_adjusted_score"), (int, float))
        ]
        return safe_fmean(vals, -1.0)

    best_depth = max(range(1, n_depths), key=selection_score)
    for row in depth_rows:
        row["selected_depth"] = bool(row["depth"] == best_depth)
    return report, depth_rows, best_depth


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


def build_shuffled_directions_at_depth(
    items: Sequence[PersonaPair],
    features: Mapping[str, Mapping[str, Any]],
    split: Mapping[str, bool],
    depth: int,
    seed: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for trait in sorted({item.trait for item in items}):
        direction = direction_for_trait(items, features, split, trait, depth, train=True, sign_seed=seed + stable_hash_int(trait))
        if direction is not None:
            out[trait] = direction
    return out


def build_refusal_monitor_direction(bundle: bench.ModelBundle, depth: int) -> Any:
    import torch

    # Forward passes only. No refusal-eliciting prompt is ever used for generation.
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


def build_sentiment_style_direction(bundle: bench.ModelBundle, depth: int) -> Any:
    import torch

    pairs = [
        ("Write a cheerful, appreciative one-sentence note about finishing a checklist.", "Write a neutral one-sentence note about finishing a checklist."),
        ("Write a warm positive one-sentence comment about a clean experiment log.", "Write a factual neutral one-sentence comment about a clean experiment log."),
        ("Write an encouraging one-sentence reply about debugging progress.", "Write a plain neutral one-sentence reply about debugging progress."),
    ]
    diffs = []
    for positive, neutral in pairs:
        pos_vec = capture_last_streams(bundle, render_chat(bundle, positive))[depth]
        neg_vec = capture_last_streams(bundle, render_chat(bundle, neutral))[depth]
        diffs.append(pos_vec - neg_vec)
    return unit(torch.stack(diffs).mean(dim=0))


def cosine(a: Any, b: Any) -> float:
    return float((a @ b) / (a.norm().clamp_min(1e-9) * b.norm().clamp_min(1e-9)))


def direction_cosine_rows(directions: Mapping[str, Any]) -> list[dict[str, Any]]:
    names = sorted(directions)
    rows = []
    for a in names:
        for b in names:
            rows.append({
                "direction_a": a,
                "direction_b": b,
                "cosine": rounded(cosine(directions[a], directions[b])),
                "same_direction": a == b,
            })
    return rows


# ---------------------------------------------------------------------------
# Generation scoring and steering
# ---------------------------------------------------------------------------


def keyword_patterns(spec: str) -> list[str]:
    return [p.strip().lower() for p in str(spec).split("|") if p.strip()]


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


def repetition_rate(text: str) -> float:
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    if len(words) < 2:
        return 0.0
    return 1.0 - (len(set(words)) / len(words))


def score_generation(item: PersonaPair, text: str) -> dict[str, Any]:
    pos = marker_count(text, item.positive_markers)
    neg = marker_count(text, item.negative_markers)
    content_hit = contains_keyword(text, item.expected_keywords)
    words = re.findall(r"[a-zA-Z0-9']+", text)
    if pos > neg:
        style_label = "positive_style"
    elif neg > pos:
        style_label = "negative_style"
    elif pos == neg and pos > 0:
        style_label = "mixed_style"
    else:
        style_label = "no_marker"
    private_flag = any(pat in text.lower() for pat in PRIVATE_EXPERIENCE_PATTERNS)
    return {
        "positive_marker_count": pos,
        "negative_marker_count": neg,
        "style_margin": pos - neg,
        "style_marker_total": pos + neg,
        "style_margin_per_100_words": rounded((pos - neg) * 100.0 / max(1, len(words))),
        "style_label": style_label,
        "content_hit": bool(content_hit),
        "content_marker_spec": item.expected_keywords,
        "n_words": len(words),
        "repetition_rate": rounded(repetition_rate(text)),
        "private_experience_flag": bool(private_flag),
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
    shuffled_directions: Mapping[str, Any],
    depth: int,
    d_model: int,
    seed: int,
    ref_norm: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    injection_layer = max(0, depth - 1)
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
            "dose_fraction": 0.0,
            "injection_layer": injection_layer,
            "stream_depth_direction_fit": depth,
            "steering_scale": 0.0,
            "generation": text,
            "hand_label_style": "",
            "hand_label_content": "",
            "hand_label_boundary": "",
            **score_generation(item, text),
        })

    for trait in sorted({item.trait for item in items}):
        trait_items = [item for item in items if item.trait == trait]
        trait_prompts = [render_chat(bundle, item.eval_prompt) for item in trait_items]
        direction = directions[trait]
        shuffled = shuffled_directions.get(trait, direction)
        random = random_unit(d_model, seed + stable_hash_int(f"random-steer:{trait}") % 100000)
        conditions = [
            ("trait_direction", direction, +1.0),
            ("opposite_direction", direction, -1.0),
            ("shuffled_sign_direction", shuffled, +1.0),
            ("random_direction", random, +1.0),
        ]
        for dose_fraction in STEERING_DOSE_FRACTIONS:
            if dose_fraction == 0.0:
                continue
            abs_scale = float(dose_fraction) * float(ref_norm)
            for condition, vec, sign in conditions:
                signed_scale = sign * abs_scale
                outs = bench.generate_continuous(
                    bundle,
                    trait_prompts,
                    MAX_NEW_TOKENS,
                    max_concurrent=ENGINE_MAX_CONCURRENT,
                    progress_label=f"lab17 steering {trait} {condition} dose={dose_fraction}",
                    steer=(injection_layer, vec, signed_scale),
                )
                for item, text in zip(trait_items, outs):
                    rows.append({
                        "item_id": item.item_id,
                        "trait": item.trait,
                        "family": item.family,
                        "topic": item.topic,
                        "steering_condition": condition,
                        "dose_fraction": dose_fraction,
                        "injection_layer": injection_layer,
                        "stream_depth_direction_fit": depth,
                        "steering_scale": rounded(signed_scale),
                        "generation": text,
                        "hand_label_style": "",
                        "hand_label_content": "",
                        "hand_label_boundary": "",
                        **score_generation(item, text),
                    })

    effect_rows: list[dict[str, Any]] = []
    for trait in sorted({row["trait"] for row in rows}):
        baseline = [row for row in rows if row["trait"] == trait and row["steering_condition"] == "baseline"]
        base_margin = safe_fmean([float(row["style_margin"]) for row in baseline])
        base_content = safe_fmean([1.0 if row["content_hit"] else 0.0 for row in baseline])
        base_private = safe_fmean([1.0 if row["private_experience_flag"] else 0.0 for row in baseline])
        for condition in sorted({row["steering_condition"] for row in rows if row["trait"] == trait}):
            dose_values = sorted({float(row["dose_fraction"]) for row in rows if row["trait"] == trait and row["steering_condition"] == condition})
            for dose in dose_values:
                sub = [row for row in rows if row["trait"] == trait and row["steering_condition"] == condition and float(row["dose_fraction"]) == dose]
                margin = safe_fmean([float(row["style_margin"]) for row in sub])
                content = safe_fmean([1.0 if row["content_hit"] else 0.0 for row in sub])
                private = safe_fmean([1.0 if row["private_experience_flag"] else 0.0 for row in sub])
                rep = safe_fmean([float(row["repetition_rate"]) for row in sub])
                effect_rows.append({
                    "trait": trait,
                    "steering_condition": condition,
                    "dose_fraction": dose,
                    "n": len(sub),
                    "mean_style_margin": rounded(margin),
                    "style_margin_delta_vs_baseline": rounded(margin - base_margin),
                    "content_hit_rate": rounded(content),
                    "content_hit_delta_vs_baseline": rounded(content - base_content),
                    "private_experience_rate": rounded(private),
                    "private_experience_delta_vs_baseline": rounded(private - base_private),
                    "mean_repetition_rate": rounded(rep),
                })
    return rows, effect_rows


# ---------------------------------------------------------------------------
# Scripted multi-turn traces
# ---------------------------------------------------------------------------


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
    sentiment_direction: Any,
    random_direction: Any,
) -> dict[str, Any]:
    frame = {
        "persona_museum_guide": directions.get("character_museum_guide"),
        "default_assistant_control": -directions.get("character_museum_guide"),
        "technical_register": directions.get("technical_register"),
        "casual_register_control": -directions.get("technical_register"),
        "warm_supportive_voice": directions.get("warm_supportive_voice"),
        "direct_terse_control": -directions.get("warm_supportive_voice"),
        "honest_correction": directions.get("honest_disagreement"),
        "agreeable_validation_control": -directions.get("honest_disagreement"),
        "sentiment_style_control": sentiment_direction,
        "refusal_monitor": refusal_direction,
        "random_null": random_direction,
    }
    return {k: v for k, v in frame.items() if v is not None}


def segment_rows_for_conversation(bundle: bench.ModelBundle, conv: RenderedConversation) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ids = conv.input_ids
    for seg in conv.segments:
        rows.append({
            "conversation": conv.spec.name,
            "segment_index": seg.index,
            "role": seg.role,
            "message_start": seg.message_start,
            "message_end_exclusive": seg.message_end,
            "message_boundary_token": seg.boundary_token,
            "message_n_tokens": seg.message_end - seg.message_start,
            "content_start": seg.content_start,
            "content_end_exclusive": seg.content_end,
            "content_boundary_token": seg.content_boundary_token,
            "content_n_tokens": seg.content_end - seg.content_start,
            "content_span_found": seg.content_span_found,
            "content_span_method": seg.content_span_method,
            "message_start_char": seg.message_start_char,
            "message_end_char": seg.message_end_char,
            "content_start_char": seg.content_start_char if seg.content_start_char is not None else "",
            "content_end_char": seg.content_end_char if seg.content_end_char is not None else "",
            "message_decoded_excerpt": bundle.tokenizer.decode(ids[seg.message_start: seg.message_end])[:140].replace("\n", "\\n"),
            "content_decoded_excerpt": bundle.tokenizer.decode(ids[seg.content_start: seg.content_end])[:140].replace("\n", "\\n"),
            "content_text_excerpt": seg.content[:140].replace("\n", "\\n"),
        })
    return rows


def run_turn_trace(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    directions: Mapping[str, Any],
    refusal_direction: Any,
    sentiment_direction: Any,
    depth: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    random_direction = random_unit(bundle.anatomy.d_model, seed + 3347)
    frame = trace_direction_frame(directions, refusal_direction, sentiment_direction, random_direction)
    boundary_checks: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    depth_sweep_rows: list[dict[str, Any]] = []

    for conv_spec in trace_conversations():
        conv = build_segments(bundle, conv_spec)
        ids = conv.input_ids
        segments = conv.segments
        coverage_ok = bool(segments) and segments[0].message_start == 0 and segments[-1].message_end == len(ids)
        no_gaps = all(a.message_end == b.message_start for a, b in zip(segments, segments[1:]))
        positive_widths = all(seg.message_end > seg.message_start and seg.content_end > seg.content_start for seg in segments)
        content_inside = all(seg.message_start <= seg.content_start <= seg.content_end <= seg.message_end for seg in segments)
        generation = generation_prompt_rows_for_rendered(bundle, conv)
        generation_rows.extend(generation)
        method_counts = Counter(seg.content_span_method for seg in segments)
        # Trust gate: every content span must decode back to its message text
        # (whitespace normalized). This validates spans against the full render
        # directly. Incremental prefix byte-identity is recorded but NOT gated:
        # templates legitimately re-render the prior turn's stop token when it
        # is no longer final (Olmo-3-Instruct <|endoftext|> -> <|im_end|>).
        content_spans_found = all(seg.content_span_found for seg in segments)
        content_spans_decode_match = all(
            "".join(bundle.tokenizer.decode(ids[seg.content_start: seg.content_end]).split())
            == "".join(seg.content.split())
            for seg in segments
        )
        content_spans_ok = content_spans_found and content_spans_decode_match
        ok = (
            coverage_ok and no_gaps and positive_widths and content_inside
            and content_spans_ok
            and conv.info["string_vs_direct_template_ids_match"]
            and conv.info["final_incremental_ids_match"]
            and all(bool(row["ok"]) for row in generation)
        )
        boundary_checks.append({
            **conv.info,
            "coverage_ok": coverage_ok,
            "no_gaps": no_gaps,
            "positive_widths": positive_widths,
            "content_inside_message": content_inside,
            "generation_prompt_ok": all(bool(row["ok"]) for row in generation),
            "content_spans_found": content_spans_found,
            "content_spans_decode_match": content_spans_decode_match,
            "content_spans_ok": content_spans_ok,
            "content_span_method_counts": dict(method_counts),
            "all_content_spans_offset_mapped": all(seg.content_span_found for seg in segments),
            "ok": ok,
        })
        segment_rows.extend(segment_rows_for_conversation(bundle, conv))
        if not ok:
            raise RuntimeError(f"Turn-boundary check failed for {conv.spec.name}.")

        capture = bench.run_with_residual_cache(bundle, conv.rendered, add_special_tokens=False)
        streams = capture.streams
        if streams.shape[1] != len(ids):
            raise RuntimeError(f"Token count mismatch while tracing {conv.spec.name}.")

        non_system_seen = 0
        for seg in segments:
            if seg.role != "system":
                non_system_seen += 1
            message_boundary = streams[depth, seg.message_end - 1, :]
            content_boundary = streams[depth, seg.content_end - 1, :]
            message_span = streams[depth, seg.message_start: seg.message_end, :]
            content_span = streams[depth, seg.content_start: seg.content_end, :]
            cumulative = streams[depth, : seg.message_end, :].mean(dim=0)
            for direction_name, direction in frame.items():
                rows.append({
                    "conversation": conv.spec.name,
                    "description": conv.spec.description,
                    "segment_index": seg.index,
                    "turn_index_non_system": non_system_seen,
                    "role": seg.role,
                    "message_start_token": seg.message_start,
                    "message_end_token_exclusive": seg.message_end,
                    "content_start_token": seg.content_start,
                    "content_end_token_exclusive": seg.content_end,
                    "n_message_tokens": seg.message_end - seg.message_start,
                    "n_content_tokens": seg.content_end - seg.content_start,
                    "content_span_method": seg.content_span_method,
                    "direction": direction_name,
                    "stream_depth": depth,
                    "boundary_projection": rounded(float(message_boundary @ direction)),
                    "content_boundary_projection": rounded(float(content_boundary @ direction)),
                    "message_span_mean_projection": rounded(float(message_span.mean(dim=0) @ direction)),
                    "content_span_mean_projection": rounded(float(content_span.mean(dim=0) @ direction)),
                    "cumulative_projection": rounded(float(cumulative @ direction)),
                    "content_excerpt": seg.content[:90],
                })

        for sweep_depth in range(streams.shape[0]):
            for direction_name in ("persona_museum_guide", "technical_register", "refusal_monitor", "random_null"):
                if direction_name not in frame:
                    continue
                sub_x: list[float] = []
                sub_y: list[float] = []
                direction = frame[direction_name]
                non_system_seen = 0
                for seg in segments:
                    if seg.role == "system":
                        continue
                    non_system_seen += 1
                    cumulative = streams[sweep_depth, : seg.message_end, :].mean(dim=0)
                    sub_x.append(float(non_system_seen))
                    sub_y.append(float(cumulative @ direction))
                depth_sweep_rows.append({
                    "conversation": conv.spec.name,
                    "direction": direction_name,
                    "direction_fit_depth": depth,
                    "projection_stream_depth": sweep_depth,
                    "n_points": len(sub_x),
                    "cumulative_projection_slope": rounded(slope(sub_x, sub_y)),
                    "start_projection": rounded(sub_y[0] if sub_y else float("nan")),
                    "end_projection": rounded(sub_y[-1] if sub_y else float("nan")),
                    "note": "best-depth direction projected across all stream depths; descriptive depth-sweep only",
                })

    slope_rows: list[dict[str, Any]] = []
    for conv in sorted({row["conversation"] for row in rows}):
        for direction in sorted({row["direction"] for row in rows}):
            sub = [row for row in rows if row["conversation"] == conv and row["direction"] == direction and row["role"] != "system"]
            xs = [float(row["turn_index_non_system"]) for row in sub]
            for measure in ("boundary_projection", "content_boundary_projection", "cumulative_projection"):
                ys = [float(row[measure]) for row in sub]
                slope_rows.append({
                    "conversation": conv,
                    "direction": direction,
                    "projection_measure": measure,
                    "n_points": len(sub),
                    "cumulative_projection_slope": rounded(slope(xs, ys)) if measure == "cumulative_projection" else "",
                    "projection_slope": rounded(slope(xs, ys)),
                    "start_projection": rounded(ys[0] if ys else float("nan")),
                    "end_projection": rounded(ys[-1] if ys else float("nan")),
                    "projection_delta": rounded((ys[-1] - ys[0]) if len(ys) >= 2 else float("nan")),
                })

    check = {
        "ok": all(bool(row["ok"]) for row in boundary_checks),
        "conversations": boundary_checks,
        "explanation": (
            "Message spans are derived by repeatedly rendering prefixes with the tokenizer's own chat template. "
            "Content spans are mapped with offset mapping when available, with token-subsequence fallback. "
            "The trace is reported only when template parity, prefix stability, complete coverage, positive widths, and generation-prompt boundary checks pass."
        ),
    }
    return rows, slope_rows, depth_sweep_rows, segment_rows, generation_rows, check


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def mean_probe_by_depth(rows: Sequence[Mapping[str, Any]], probe_split: str, kind: str) -> list[tuple[int, float]]:
    depths = sorted({int(row["depth"]) for row in rows if row.get("probe_split") == probe_split})
    out = []
    for depth in depths:
        vals = [
            float(row["auc"])
            for row in rows
            if row.get("probe_split") == probe_split
            and row.get("direction_kind") == kind
            and int(row.get("depth", -1)) == depth
            and isinstance(row.get("auc"), (int, float))
        ]
        if vals:
            out.append((depth, safe_fmean(vals)))
    return out


def plot_probe_selectivity(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]], best_depth: int) -> None:
    fig, ax = bench.new_figure(figsize=(9.2, 5.2))
    for split_name, linestyle in (("train_loo", "--"), ("eval", "-")):
        for kind, label in (("real", "real"), ("shuffled_sign", "shuffled"), ("random_oriented", "random")):
            pts = mean_probe_by_depth(rows, split_name, kind)
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", linestyle=linestyle, linewidth=1.8, label=f"{label} {split_name}")
    ax.axhline(0.5, linewidth=1.0, alpha=0.5)
    ax.axvline(best_depth, linewidth=1.0, alpha=0.6, label=f"chosen depth {best_depth}")
    ax.set_xlabel("stream depth")
    ax.set_ylabel("mean AUC across traits")
    ax.set_title("Persona/register/voice probe selectivity with controls")
    bench.style_ax(ax)
    bench.save_figure(ctx, fig, "persona_probe_selectivity.png", "Probe AUC by stream depth, split, and control family.")


def plot_steering_dose_response(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(9.2, 5.2))
    for condition, label in (
        ("trait_direction", "trait direction"),
        ("opposite_direction", "opposite"),
        ("shuffled_sign_direction", "shuffled"),
        ("random_direction", "random"),
    ):
        pts = []
        for dose in sorted({float(row["dose_fraction"]) for row in rows if row.get("steering_condition") == condition}):
            vals = [
                float(row["style_margin_delta_vs_baseline"])
                for row in rows
                if row.get("steering_condition") == condition
                and float(row.get("dose_fraction", -1)) == dose
                and isinstance(row.get("style_margin_delta_vs_baseline"), (int, float))
            ]
            if vals:
                pts.append((dose, safe_fmean(vals)))
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", linewidth=2.0, label=label)
    ax.axhline(0.0, linewidth=1.0, alpha=0.5)
    ax.set_xlabel("dose fraction of mean residual norm")
    ax.set_ylabel("style-marker margin delta vs baseline")
    ax.set_title("Steering dose response: style movement beyond controls")
    bench.style_ax(ax)
    bench.save_figure(ctx, fig, "persona_steering_dose_response.png", "Style-marker dose response for trait, opposite, shuffled, and random steering.")


def plot_style_content_tradeoff(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(8.4, 5.2))
    for condition in sorted({row.get("steering_condition") for row in rows}):
        if condition == "baseline":
            continue
        sub = [row for row in rows if row.get("steering_condition") == condition and row.get("dose_fraction") == max(STEERING_DOSE_FRACTIONS)]
        if not sub:
            continue
        xs = [float(row["style_margin_delta_vs_baseline"]) for row in sub if isinstance(row.get("style_margin_delta_vs_baseline"), (int, float))]
        ys = [float(row["content_hit_delta_vs_baseline"]) for row in sub if isinstance(row.get("content_hit_delta_vs_baseline"), (int, float))]
        if xs and ys:
            ax.scatter([safe_fmean(xs)], [safe_fmean(ys)], s=80, label=condition)
    ax.axhline(0.0, linewidth=1.0, alpha=0.5)
    ax.axvline(0.0, linewidth=1.0, alpha=0.5)
    ax.set_xlabel("style-marker delta at max dose")
    ax.set_ylabel("content-hit delta at max dose")
    ax.set_title("Content vs style: a style handle should not break the task")
    bench.style_ax(ax)
    bench.save_figure(ctx, fig, "style_content_tradeoff.png", "Style movement versus content preservation under steering controls.")


def plot_direction_cosines(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    names = sorted({str(row["direction_a"]) for row in rows})
    idx = {name: i for i, name in enumerate(names)}
    matrix = [[0.0 for _ in names] for _ in names]
    for row in rows:
        a = str(row["direction_a"])
        b = str(row["direction_b"])
        matrix[idx[a]][idx[b]] = float(row["cosine"])
    fig, ax = plt.subplots(figsize=(max(7.5, 0.55 * len(names)), max(6.0, 0.55 * len(names))))
    im = ax.imshow(matrix, vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Direction cosine audit")
    fig.tight_layout()
    bench.save_figure(ctx, fig, "direction_cosine_heatmap.png", "Cosine map among persona, register, voice, agreement, sentiment, and refusal directions.")


def plot_persona_turn_trace(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(9.2, 5.4))
    styles = {
        ("museum_roleplay", "persona_museum_guide"): ("tab:purple", "-", "roleplay: museum-guide direction"),
        ("museum_roleplay", "default_assistant_control"): ("tab:gray", "--", "roleplay: default-control direction"),
        ("museum_roleplay", "random_null"): ("black", ":", "roleplay: random null"),
        ("default_control", "persona_museum_guide"): ("tab:blue", "--", "default control: museum-guide direction"),
    }
    for (conv, direction), (color, linestyle, label) in styles.items():
        sub = [row for row in rows if row["conversation"] == conv and row["direction"] == direction and row["role"] != "system"]
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
        sub = [row for row in rows if row["conversation"] == "register_switch" and row["direction"] == direction and row["role"] != "system"]
        if not sub:
            continue
        ax.plot(
            [float(row["turn_index_non_system"]) for row in sub],
            [float(row["content_boundary_projection"]) for row in sub],
            marker="o",
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
    ax.set_xlabel("message boundary (system excluded)")
    ax.set_ylabel("content-boundary projection")
    ax.set_title("Register switch trace at content boundaries")
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
        sub = [row for row in rows if row["conversation"] == conv and row["direction"] == direction and row["role"] != "system"]
        if not sub:
            continue
        ax.plot(
            [float(row["turn_index_non_system"]) for row in sub],
            [float(row["content_boundary_projection"]) for row in sub],
            marker="o",
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=label,
        )
    ax.set_xlabel("message boundary (system excluded)")
    ax.set_ylabel("content-boundary projection")
    ax.set_title("Refusal monitor under benign roleplay")
    bench.style_ax(ax)
    bench.save_figure(ctx, fig, "refusal_projection_under_roleplay.png", "Refusal-monitor projection in a benign roleplay boundary conversation.")


def plot_trace_depth_sweep(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]], best_depth: int) -> None:
    fig, ax = bench.new_figure(figsize=(9.2, 5.2))
    for conv, direction, label in (
        ("museum_roleplay", "persona_museum_guide", "roleplay persona"),
        ("museum_roleplay", "random_null", "roleplay random"),
        ("register_switch", "technical_register", "register switch"),
        ("roleplay_boundary", "refusal_monitor", "refusal boundary"),
    ):
        sub = [row for row in rows if row["conversation"] == conv and row["direction"] == direction]
        if not sub:
            continue
        sub = sorted(sub, key=lambda r: int(r["projection_stream_depth"]))
        ax.plot(
            [int(row["projection_stream_depth"]) for row in sub],
            [float(row["cumulative_projection_slope"]) for row in sub],
            marker="o",
            linewidth=1.8,
            label=label,
        )
    ax.axvline(best_depth, linewidth=1.0, alpha=0.6, label=f"fit depth {best_depth}")
    ax.axhline(0.0, linewidth=1.0, alpha=0.5)
    ax.set_xlabel("projection stream depth")
    ax.set_ylabel("cumulative projection slope")
    ax.set_title("Trace depth sweep using fixed best-depth directions")
    bench.style_ax(ax)
    bench.save_figure(ctx, fig, "trace_depth_sweep.png", "Descriptive sweep of turn-trace slopes across stream depths.")




# ---------------------------------------------------------------------------
# Visualization upgrade: evidence dashboards, synthesis tables, and richer plots
# ---------------------------------------------------------------------------


def _lab17_color(key: str, default: str = "#555555") -> str:
    fn = getattr(bench, "plot_persona_color", None)
    if callable(fn):
        return fn(key, default)
    palette = {
        "persona": "#7E57C2",
        "character_museum_guide": "#7E57C2",
        "technical_register": "#0072B2",
        "register": "#0072B2",
        "warm_supportive_voice": "#E69F00",
        "voice": "#E69F00",
        "honest_disagreement": "#009E73",
        "agreement": "#009E73",
        "trait_direction": "#D55E00",
        "opposite_direction": "#0072B2",
        "shuffled_sign_direction": "#999999",
        "random_direction": "#777777",
        "baseline": "#444444",
        "real": "#0072B2",
        "shuffled_sign": "#E69F00",
        "random_oriented": "#777777",
        "positive": "#009E73",
        "warning": "#E69F00",
        "failed": "#D55E00",
        "control": "#777777",
        "refusal_monitor": "#D55E00",
        "sentiment_style_control": "#CC79A7",
        "random_null": "#777777",
        "persona_museum_guide": "#7E57C2",
        "default_assistant_control": "#777777",
        "technical_register_direction": "#0072B2",
    }
    return palette.get(str(key), default)


def _lab17_marker(key: str, default: str = "o") -> str:
    fn = getattr(bench, "plot_persona_marker", None)
    if callable(fn):
        return fn(key, default)
    markers = {
        "trait_direction": "o",
        "opposite_direction": "v",
        "shuffled_sign_direction": "s",
        "random_direction": "^",
        "real": "o",
        "shuffled_sign": "s",
        "random_oriented": "^",
        "character_museum_guide": "o",
        "technical_register": "s",
        "warm_supportive_voice": "^",
        "honest_disagreement": "D",
    }
    return markers.get(str(key), default)


def _f(row_or_value: Any, key: str | None = None, default: float = float("nan")) -> float:
    try:
        value = row_or_value.get(key) if key is not None and isinstance(row_or_value, Mapping) else row_or_value
        if value == "" or value is None:
            return default
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def _finite(vals: Sequence[Any]) -> list[float]:
    out = []
    for v in vals:
        fv = _f(v)
        if math.isfinite(fv):
            out.append(fv)
    return out


def _quantile(vals: Sequence[Any], q: float, default: float = float("nan")) -> float:
    xs = sorted(_finite(vals))
    if not xs:
        return default
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def _mean_where(rows: Sequence[Mapping[str, Any]], key: str, **filters: Any) -> float:
    vals = []
    for row in rows:
        ok = True
        for fkey, fval in filters.items():
            if row.get(fkey) != fval:
                ok = False
                break
        if ok:
            v = _f(row, key)
            if math.isfinite(v):
                vals.append(v)
    return safe_fmean(vals)


def _max_dose_from_effects(rows: Sequence[Mapping[str, Any]]) -> float:
    vals = sorted({_f(row, "dose_fraction") for row in rows if math.isfinite(_f(row, "dose_fraction"))})
    vals = [v for v in vals if v > 0]
    return vals[-1] if vals else max(STEERING_DOSE_FRACTIONS)


def _all_traits_from_probe(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    traits = sorted({str(row.get("trait")) for row in rows if row.get("trait") not in {None, ""}})
    return traits


def build_depth_control_gap_rows(probe_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    traits = _all_traits_from_probe(probe_rows)
    depths = sorted({int(row["depth"]) for row in probe_rows if isinstance(row.get("depth"), int) or str(row.get("depth", "")).isdigit()})
    for trait in traits:
        for depth in depths:
            for split_name in ("train_loo", "eval"):
                real = _mean_where(probe_rows, "auc", trait=trait, depth=depth, probe_split=split_name, direction_kind="real")
                shuf = _mean_where(probe_rows, "auc", trait=trait, depth=depth, probe_split=split_name, direction_kind="shuffled_sign")
                rand = _mean_where(probe_rows, "auc", trait=trait, depth=depth, probe_split=split_name, direction_kind="random_oriented")
                best_control = max(shuf if math.isfinite(shuf) else 0.5, rand if math.isfinite(rand) else 0.5)
                rows.append({
                    "trait": trait,
                    "depth": depth,
                    "probe_split": split_name,
                    "real_auc": none_if_nan(real),
                    "shuffled_auc": none_if_nan(shuf),
                    "random_auc": none_if_nan(rand),
                    "best_control_auc": rounded(best_control),
                    "control_adjusted_auc_gap": none_if_nan(real - best_control),
                    "claim_hint": "candidate_depth" if split_name == "train_loo" and math.isfinite(real - best_control) and real - best_control >= MIN_SELECTIVITY_GAP else "report_only",
                })
    return rows


def build_steering_operating_points(steering_effects: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    traits = sorted({str(row.get("trait")) for row in steering_effects if row.get("trait")})
    for trait in traits:
        doses = sorted({_f(row, "dose_fraction") for row in steering_effects if row.get("trait") == trait and math.isfinite(_f(row, "dose_fraction"))})
        for dose in doses:
            control_vals = []
            for ctl in ("random_direction", "shuffled_sign_direction", "opposite_direction"):
                val = _mean_where(steering_effects, "style_margin_delta_vs_baseline", trait=trait, steering_condition=ctl, dose_fraction=dose)
                if math.isfinite(val):
                    control_vals.append(val)
            best_control = max(control_vals) if control_vals else 0.0
            for condition in sorted({str(row.get("steering_condition")) for row in steering_effects if row.get("trait") == trait and _f(row, "dose_fraction") == dose}):
                sub = [row for row in steering_effects if row.get("trait") == trait and row.get("steering_condition") == condition and _f(row, "dose_fraction") == dose]
                style = safe_fmean([_f(row, "style_margin_delta_vs_baseline") for row in sub])
                content = safe_fmean([_f(row, "content_hit_delta_vs_baseline") for row in sub])
                private = safe_fmean([_f(row, "private_experience_rate") for row in sub])
                repetition = safe_fmean([_f(row, "mean_repetition_rate") for row in sub])
                rows.append({
                    "trait": trait,
                    "steering_condition": condition,
                    "dose_fraction": rounded(dose),
                    "style_margin_delta": none_if_nan(style),
                    "best_control_style_delta_at_same_dose": rounded(best_control),
                    "specificity_gap_vs_best_control": none_if_nan(style - best_control),
                    "content_hit_delta": none_if_nan(content),
                    "content_damage": none_if_nan(max(0.0, -content) if math.isfinite(content) else float("nan")),
                    "private_experience_rate": none_if_nan(private),
                    "mean_repetition_rate": none_if_nan(repetition),
                    "claimable_operating_point": bool(condition == "trait_direction" and math.isfinite(style - best_control) and (style - best_control) >= MIN_STEERING_SPECIFICITY_GAP and (not math.isfinite(content) or content >= MIN_CONTENT_DELTA)),
                })
    return rows


def build_trait_evidence_matrix(
    probe_rows: Sequence[Mapping[str, Any]],
    steering_effects: Sequence[Mapping[str, Any]],
    turn_slopes: Sequence[Mapping[str, Any]],
    best_depth: int,
) -> list[dict[str, Any]]:
    max_dose = _max_dose_from_effects(steering_effects)
    trace_map = {
        "character_museum_guide": ("museum_roleplay", "persona_museum_guide", "cumulative_projection"),
        "technical_register": ("register_switch", "technical_register", "content_boundary_projection"),
        "warm_supportive_voice": ("museum_roleplay", "warm_supportive_voice", "cumulative_projection"),
        "honest_disagreement": ("roleplay_boundary", "honest_correction", "content_boundary_projection"),
    }
    rows: list[dict[str, Any]] = []
    for trait in _all_traits_from_probe(probe_rows):
        real = _mean_where(probe_rows, "auc", trait=trait, depth=best_depth, probe_split="eval", direction_kind="real")
        shuf = _mean_where(probe_rows, "auc", trait=trait, depth=best_depth, probe_split="eval", direction_kind="shuffled_sign")
        rand = _mean_where(probe_rows, "auc", trait=trait, depth=best_depth, probe_split="eval", direction_kind="random_oriented")
        best_control = max(shuf if math.isfinite(shuf) else 0.5, rand if math.isfinite(rand) else 0.5)
        style = _mean_where(steering_effects, "style_margin_delta_vs_baseline", trait=trait, steering_condition="trait_direction", dose_fraction=max_dose)
        shuf_s = _mean_where(steering_effects, "style_margin_delta_vs_baseline", trait=trait, steering_condition="shuffled_sign_direction", dose_fraction=max_dose)
        rand_s = _mean_where(steering_effects, "style_margin_delta_vs_baseline", trait=trait, steering_condition="random_direction", dose_fraction=max_dose)
        opp_s = _mean_where(steering_effects, "style_margin_delta_vs_baseline", trait=trait, steering_condition="opposite_direction", dose_fraction=max_dose)
        best_steer_control = max([v for v in (shuf_s, rand_s, opp_s) if math.isfinite(v)] or [0.0])
        content = _mean_where(steering_effects, "content_hit_delta_vs_baseline", trait=trait, steering_condition="trait_direction", dose_fraction=max_dose)
        private = _mean_where(steering_effects, "private_experience_rate", trait=trait, steering_condition="trait_direction", dose_fraction=max_dose)
        repetition = _mean_where(steering_effects, "mean_repetition_rate", trait=trait, steering_condition="trait_direction", dose_fraction=max_dose)
        trace_conv, trace_dir, trace_measure = trace_map.get(trait, ("", "", "cumulative_projection"))
        trace = trace_slope(turn_slopes, trace_conv, trace_dir, projection_measure=trace_measure) if trace_conv else float("nan")
        trace_null = trace_slope(turn_slopes, trace_conv, "random_null", projection_measure=trace_measure) if trace_conv else float("nan")
        selectivity_gap = real - best_control
        steering_gap = style - best_steer_control
        if math.isfinite(selectivity_gap) and selectivity_gap >= MIN_SELECTIVITY_GAP and math.isfinite(steering_gap) and steering_gap >= MIN_STEERING_SPECIFICITY_GAP and (not math.isfinite(content) or content >= MIN_CONTENT_DELTA):
            posture = "controlled_style_handle"
        elif math.isfinite(selectivity_gap) and selectivity_gap >= MIN_SELECTIVITY_GAP:
            posture = "decodable_not_yet_causal"
        elif math.isfinite(steering_gap) and steering_gap >= MIN_STEERING_SPECIFICITY_GAP:
            posture = "steering_effect_needs_decode_controls"
        else:
            posture = "not_validated"
        rows.append({
            "trait": trait,
            "best_depth": best_depth,
            "eval_real_auc": none_if_nan(real),
            "eval_best_control_auc": rounded(best_control),
            "decode_gap_vs_best_control": none_if_nan(selectivity_gap),
            "max_dose": rounded(max_dose),
            "trait_style_delta": none_if_nan(style),
            "best_control_style_delta": none_if_nan(best_steer_control),
            "steering_gap_vs_best_control": none_if_nan(steering_gap),
            "content_hit_delta": none_if_nan(content),
            "private_experience_rate": none_if_nan(private),
            "mean_repetition_rate": none_if_nan(repetition),
            "trace_slope": none_if_nan(trace),
            "trace_random_slope": none_if_nan(trace_null),
            "trace_gap_vs_random": none_if_nan(trace - trace_null if math.isfinite(trace) and math.isfinite(trace_null) else float("nan")),
            "claim_posture": posture,
        })
    return rows


def build_trace_evidence_rows(turn_slopes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for conv in sorted({str(row.get("conversation")) for row in turn_slopes if row.get("conversation")}):
        for measure in sorted({str(row.get("projection_measure")) for row in turn_slopes if row.get("conversation") == conv and row.get("projection_measure")}):
            random_slope = _mean_where(turn_slopes, "projection_slope", conversation=conv, direction="random_null", projection_measure=measure)
            for direction in sorted({str(row.get("direction")) for row in turn_slopes if row.get("conversation") == conv and row.get("projection_measure") == measure and row.get("direction")}):
                slope_val = _mean_where(turn_slopes, "projection_slope", conversation=conv, direction=direction, projection_measure=measure)
                delta = _mean_where(turn_slopes, "projection_delta", conversation=conv, direction=direction, projection_measure=measure)
                rows.append({
                    "conversation": conv,
                    "direction": direction,
                    "projection_measure": measure,
                    "projection_slope": none_if_nan(slope_val),
                    "projection_delta": none_if_nan(delta),
                    "random_null_slope": none_if_nan(random_slope),
                    "gap_vs_random_null": none_if_nan(slope_val - random_slope if math.isfinite(slope_val) and math.isfinite(random_slope) else float("nan")),
                    "trace_claim_posture": "candidate_trace" if direction != "random_null" and math.isfinite(slope_val - random_slope) and abs(slope_val - random_slope) > 0.03 else "weak_or_null",
                })
    return rows


def build_direction_confound_risks(cos_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    control_words = ("sentiment", "refusal", "random", "control", "default", "casual", "direct", "agreeable")
    directions = sorted({str(row.get("direction_a")) for row in cos_rows if row.get("direction_a")})
    rows: list[dict[str, Any]] = []
    for name in directions:
        others = [row for row in cos_rows if row.get("direction_a") == name and row.get("direction_b") != name]
        nearest = max(others, key=lambda r: abs(_f(r, "cosine", 0.0)), default=None)
        confs = [row for row in others if any(w in str(row.get("direction_b", "")) for w in control_words)]
        nearest_conf = max(confs, key=lambda r: abs(_f(r, "cosine", 0.0)), default=None)
        conf_val = abs(_f(nearest_conf, "cosine", float("nan"))) if nearest_conf else float("nan")
        rows.append({
            "direction": name,
            "nearest_direction": nearest.get("direction_b") if nearest else "",
            "nearest_abs_cosine": none_if_nan(abs(_f(nearest, "cosine", float("nan"))) if nearest else float("nan")),
            "nearest_control_or_style_direction": nearest_conf.get("direction_b") if nearest_conf else "",
            "nearest_control_abs_cosine": none_if_nan(conf_val),
            "confound_risk": "high" if math.isfinite(conf_val) and conf_val >= 0.70 else ("medium" if math.isfinite(conf_val) and conf_val >= 0.45 else "low"),
        })
    return rows


def write_plot_reading_guide(ctx: bench.RunContext) -> None:
    rows = [
        {"plot": "persona_evidence_dashboard.png", "read_for": "one-screen verdict: decode, steering, trace, and confound risk", "claim_boundary": "do not call a style handle a real identity"},
        {"plot": "trait_evidence_matrix.png", "read_for": "per-trait evidence posture and which handles survive controls", "claim_boundary": "one strong trait does not validate every persona/register axis"},
        {"plot": "depth_control_gap_atlas.png", "read_for": "where real probe AUC beats shuffled/random controls", "claim_boundary": "depth selection must be train-side and control-adjusted"},
        {"plot": "persona_probe_selectivity.png", "read_for": "depth curves with train/eval and controls", "claim_boundary": "AUC above chance is not enough if controls travel with it"},
        {"plot": "persona_steering_dose_response.png", "read_for": "dose response for trait/opposite/random/shuffled steering", "claim_boundary": "activation addition earns only a scoped behavior handle"},
        {"plot": "steering_operating_frontier.png", "read_for": "style movement versus content, repetition, and private-experience costs", "claim_boundary": "largest dose is not automatically best"},
        {"plot": "generation_style_atlas.png", "read_for": "trait-by-dose style and content deltas", "claim_boundary": "aggregate steering can hide one fragile trait"},
        {"plot": "direction_cosine_heatmap.png", "read_for": "which directions collapse into style, sentiment, refusal, or agreement controls", "claim_boundary": "cosine structure is an audit, not mechanism"},
        {"plot": "direction_confound_risk.png", "read_for": "nearest style/control neighbors for every saved direction", "claim_boundary": "high cosine to a confound narrows the claim"},
        {"plot": "persona_trace_projection_atlas.png", "read_for": "turn-by-turn projection patterns across scripted conversations", "claim_boundary": "multi-turn traces are descriptive unless nulls and boundary checks pass"},
        {"plot": "trace_evidence_atlas.png", "read_for": "slope gaps versus random-null by conversation and direction", "claim_boundary": "a rising trace in every conversation may be template/length residue"},
        {"plot": "refusal_boundary_safety_dashboard.png", "read_for": "refusal-monitor stability under benign roleplay", "claim_boundary": "monitor only; no refusal-eliciting generation or ablation"},
    ]
    path = ctx.path("tables", "plot_reading_guide.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "guide", "Reading guide for upgraded Lab 17 visual artifacts.")


def write_visual_synthesis_tables(
    ctx: bench.RunContext,
    items: Sequence[PersonaPair],
    probe_rows: Sequence[Mapping[str, Any]],
    depth_rows: Sequence[Mapping[str, Any]],
    best_depth: int,
    steering_effects: Sequence[Mapping[str, Any]],
    steering_generations: Sequence[Mapping[str, Any]],
    cos_rows: Sequence[Mapping[str, Any]],
    turn_slopes: Sequence[Mapping[str, Any]],
    turn_rows: Sequence[Mapping[str, Any]],
    trace_depth_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    depth_gap_rows = build_depth_control_gap_rows(probe_rows)
    trait_rows = build_trait_evidence_matrix(probe_rows, steering_effects, turn_slopes, best_depth)
    operating_rows = build_steering_operating_points(steering_effects)
    trace_rows = build_trace_evidence_rows(turn_slopes)
    confound_rows = build_direction_confound_risks(cos_rows)

    outputs = {
        "depth_control_gap_rows": depth_gap_rows,
        "trait_evidence_rows": trait_rows,
        "steering_operating_rows": operating_rows,
        "trace_evidence_rows": trace_rows,
        "direction_confound_rows": confound_rows,
    }
    specs = [
        ("probe_depth_control_gaps.csv", depth_gap_rows, "Real-minus-best-control probe gaps by trait, split, and depth."),
        ("persona_trait_evidence_matrix.csv", trait_rows, "Per-trait decode/steering/trace/control posture for Lab 17."),
        ("persona_steering_operating_points.csv", operating_rows, "Dose-level operating points with specificity, content, repetition, and private-experience costs."),
        ("persona_trace_evidence.csv", trace_rows, "Trace slopes and gaps versus random-null by conversation, direction, and projection measure."),
        ("persona_direction_confound_risks.csv", confound_rows, "Nearest direction/control cosine risks for persona/register/voice directions."),
    ]
    for filename, table_rows, desc in specs:
        path = ctx.path("tables", filename)
        bench.write_csv_with_context(ctx, path, table_rows)
        ctx.register_artifact(path, "table", desc)
    write_plot_reading_guide(ctx)
    return outputs


def _matrix_from_rows(rows: Sequence[Mapping[str, Any]], row_key: str, col_key: str, value_key: str) -> tuple[list[str], list[str], list[list[float]]]:
    row_names = sorted({str(row.get(row_key)) for row in rows if row.get(row_key) not in {None, ""}})
    col_names = sorted({str(row.get(col_key)) for row in rows if row.get(col_key) not in {None, ""}}, key=lambda x: (float(x) if re.fullmatch(r"-?\d+(\.\d+)?", x) else x))
    mat = [[float("nan") for _ in col_names] for _ in row_names]
    rix = {r: i for i, r in enumerate(row_names)}
    cix = {c: i for i, c in enumerate(col_names)}
    for row in rows:
        r = str(row.get(row_key))
        c = str(row.get(col_key))
        if r in rix and c in cix:
            v = _f(row, value_key)
            if math.isfinite(v):
                mat[rix[r]][cix[c]] = v
    return row_names, col_names, mat


def _imshow_with_numbers(ax: Any, mat: Sequence[Sequence[float]], *, fmt: str = ".2f", color_threshold: float | None = None) -> None:
    import numpy as np
    arr = np.array(mat, dtype=float)
    finite = arr[np.isfinite(arr)]
    if color_threshold is None:
        color_threshold = float(np.nanmean(finite)) if finite.size else 0.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if math.isfinite(float(v)):
                ax.text(j, i, format(float(v), fmt), ha="center", va="center", fontsize=7.0,
                        color="white" if abs(float(v)) > abs(color_threshold) else "#222222")


def plot_probe_selectivity(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]], best_depth: int) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    for ax, split_name, title in zip(axes, ("train_loo", "eval"), ("train-side selection rail", "held-out report rail")):
        for kind, label in (("real", "real direction"), ("shuffled_sign", "shuffled-label control"), ("random_oriented", "random-direction control")):
            pts = mean_probe_by_depth(rows, split_name, kind)
            if not pts:
                continue
            color = _lab17_color(kind)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=_lab17_marker(kind), linewidth=2.0, label=label, color=color)
            if hasattr(bench, "label_line_end"):
                bench.label_line_end(ax, [p[0] for p in pts], [p[1] for p in pts], label.replace(" control", ""), color=color)
        ax.axhline(0.5, linewidth=1.0, alpha=0.6, color="#333333", linestyle=":")
        ax.axvline(best_depth, linewidth=1.0, alpha=0.7, color=_lab17_color("persona"), linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("stream depth")
        bench.style_ax(ax, legend=False)
    axes[0].set_ylabel("mean AUC across traits")
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Persona/register/voice probe selectivity: selection rail separated from report rail")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    bench.save_figure(ctx, fig, "persona_probe_selectivity.png", "Probe AUC by stream depth, split, and control family, with depth selection separated from held-out reporting.")


def plot_depth_control_gap_atlas(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]], best_depth: int) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    eval_rows = [row for row in rows if row.get("probe_split") == "eval"]
    traits, depths, mat = _matrix_from_rows(eval_rows, "trait", "depth", "control_adjusted_auc_gap")
    if not traits or not depths:
        return
    arr = np.array(mat, dtype=float)
    fig, ax = plt.subplots(figsize=(max(9.5, 0.32 * len(depths) + 4.0), max(4.6, 0.45 * len(traits) + 1.8)))
    im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", vmin=-0.35, vmax=0.35)
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels(depths, rotation=0, fontsize=7)
    ax.set_yticks(range(len(traits)))
    ax.set_yticklabels(traits)
    if str(best_depth) in depths:
        ax.axvline(depths.index(str(best_depth)), color="#111111", linestyle="--", linewidth=1.2, alpha=0.8)
    _imshow_with_numbers(ax, arr, fmt=".2f", color_threshold=0.18)
    fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02, label="real AUC - best control AUC")
    ax.set_xlabel("stream depth")
    ax.set_title("Held-out decode gap atlas: real direction must beat random/shuffled controls")
    bench.style_ax(ax, legend=False)
    bench.save_figure(ctx, fig, "depth_control_gap_atlas.png", "Trait-by-depth atlas of held-out probe AUC gap over the strongest control.")


def plot_persona_evidence_dashboard(
    ctx: bench.RunContext,
    depth_rows: Sequence[Mapping[str, Any]],
    trait_rows: Sequence[Mapping[str, Any]],
    operating_rows: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    confound_rows: Sequence[Mapping[str, Any]],
    best_depth: int,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.5))
    ax = axes[0, 0]
    train = [row for row in depth_rows if row.get("probe_split") == "train_loo"]
    eval_ = [row for row in depth_rows if row.get("probe_split") == "eval"]
    for rows_, label, ls in ((train, "train-control gap", "--"), (eval_, "held-out control gap", "-")):
        depths = sorted({_f(row, "depth") for row in rows_ if math.isfinite(_f(row, "depth"))})
        pts = []
        for d in depths:
            vals = [_f(row, "control_adjusted_auc_gap") for row in rows_ if _f(row, "depth") == d]
            pts.append((d, safe_fmean(vals)))
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", linewidth=2.0, linestyle=ls, label=label)
    ax.axhline(MIN_SELECTIVITY_GAP, color="#222222", linestyle=":", linewidth=1.0, label="decode bar")
    ax.axvline(best_depth, color=_lab17_color("persona"), linestyle="--", linewidth=1.2, label=f"chosen depth {best_depth}")
    bench.style_ax(ax, title="Decode gap over depth", xlabel="stream depth", ylabel="mean real - best control AUC")

    ax = axes[0, 1]
    for row in trait_rows:
        x = _f(row, "steering_gap_vs_best_control")
        y = _f(row, "content_hit_delta")
        if math.isfinite(x) and math.isfinite(y):
            ax.scatter(x, y, s=120, color=_lab17_color(str(row.get("trait"))), marker=_lab17_marker(str(row.get("trait"))), edgecolor="#222222", linewidth=0.6)
            ax.text(x, y, str(row.get("trait", "")).replace("_", "\n"), fontsize=7, ha="center", va="bottom")
    ax.axvline(MIN_STEERING_SPECIFICITY_GAP, color="#222222", linestyle=":", linewidth=1.0, label="steering bar")
    ax.axhline(MIN_CONTENT_DELTA, color="#777777", linestyle="--", linewidth=1.0, label="content floor")
    bench.style_ax(ax, title="Steering specificity vs task preservation", xlabel="style delta beyond best control", ylabel="content-hit delta")

    ax = axes[1, 0]
    keep = [row for row in trace_rows if row.get("projection_measure") in {"cumulative_projection", "content_boundary_projection"} and row.get("direction") != "random_null"]
    keep = sorted(keep, key=lambda r: abs(_f(r, "gap_vs_random_null", 0.0)), reverse=True)[:10]
    labels = [f"{row.get('conversation')}\n{row.get('direction')}" for row in keep][::-1]
    vals = [_f(row, "gap_vs_random_null", 0.0) for row in keep][::-1]
    colors = [_lab17_color(str(row.get("direction"))) for row in keep][::-1]
    ax.barh(range(len(vals)), vals, color=colors, alpha=0.9)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="#222222", linewidth=0.8)
    bench.style_ax(ax, title="Largest trace gaps vs random null", xlabel="projection slope gap", ylabel="")

    ax = axes[1, 1]
    dirs = [str(row.get("direction")) for row in confound_rows]
    vals = [_f(row, "nearest_control_abs_cosine", 0.0) for row in confound_rows]
    order = sorted(range(len(dirs)), key=lambda i: vals[i])
    ax.barh(range(len(order)), [vals[i] for i in order], color=[_lab17_color("warning" if vals[i] >= 0.45 else "positive") for i in order])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([dirs[i] for i in order], fontsize=7)
    ax.axvline(0.45, color="#E69F00", linestyle=":", linewidth=1.0, label="audit bar")
    ax.axvline(0.70, color="#D55E00", linestyle="--", linewidth=1.0, label="high-risk bar")
    bench.style_ax(ax, title="Nearest style/control cosine risk", xlabel="abs cosine", ylabel="")
    fig.suptitle("Lab 17 persona/register/voice evidence dashboard")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    bench.save_figure(ctx, fig, "persona_evidence_dashboard.png", "Dashboard combining decode controls, steering specificity, multi-turn trace gaps, and confound cosine risk.")


def plot_trait_evidence_matrix(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    if not rows:
        return
    cols = [
        ("decode_gap_vs_best_control", "decode\ngap"),
        ("steering_gap_vs_best_control", "steering\ngap"),
        ("content_hit_delta", "content\ndelta"),
        ("private_experience_rate", "private\nrate"),
        ("trace_gap_vs_random", "trace\ngap"),
    ]
    traits = [str(row.get("trait")) for row in rows]
    mat = [[_f(row, key) for key, _ in cols] for row in rows]
    arr = np.array(mat, dtype=float)
    fig, ax = plt.subplots(figsize=(8.8, max(4.4, 0.55 * len(rows) + 1.8)))
    im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", vmin=-0.35, vmax=0.35)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([label for _, label in cols])
    ax.set_yticks(range(len(traits)))
    ax.set_yticklabels(traits)
    _imshow_with_numbers(ax, arr, fmt=".2f", color_threshold=0.22)
    for i, row in enumerate(rows):
        posture = str(row.get("claim_posture", ""))
        color = _lab17_color("positive" if posture == "controlled_style_handle" else ("warning" if "decodable" in posture or "steering" in posture else "failed"))
        ax.text(len(cols) - 0.02, i, "  " + posture.replace("_", " "), va="center", ha="left", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="signed evidence metric")
    ax.set_title("Trait evidence matrix: decode, steering, task cost, and trace")
    bench.style_ax(ax, legend=False)
    bench.save_figure(ctx, fig, "trait_evidence_matrix.png", "Per-trait evidence matrix for persona/register/voice handles and controls.")


def plot_steering_dose_response(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    fig, ax = bench.new_figure(figsize=(9.8, 5.4))
    order = ("trait_direction", "opposite_direction", "shuffled_sign_direction", "random_direction")
    for condition in order:
        pts = []
        for dose in sorted({_f(row, "dose_fraction") for row in rows if row.get("steering_condition") == condition and math.isfinite(_f(row, "dose_fraction"))}):
            vals = [_f(row, "style_margin_delta_vs_baseline") for row in rows if row.get("steering_condition") == condition and _f(row, "dose_fraction") == dose]
            pts.append((dose, safe_fmean(vals)))
        if pts:
            color = _lab17_color(condition)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=_lab17_marker(condition), linewidth=2.2, color=color, label=condition.replace("_", " "))
            if hasattr(bench, "label_line_end"):
                bench.label_line_end(ax, [p[0] for p in pts], [p[1] for p in pts], condition.replace("_direction", ""), color=color)
    ax.axhline(0.0, linewidth=1.0, alpha=0.55, color="#222222")
    ax.set_xlabel("dose fraction of mean residual norm")
    ax.set_ylabel("style-marker margin delta vs baseline")
    ax.set_title("Steering dose response: the trait vector must separate from controls")
    bench.style_ax(ax)
    bench.save_figure(ctx, fig, "persona_steering_dose_response.png", "Style-marker dose response for trait, opposite, shuffled, and random steering.")


def plot_steering_operating_frontier(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(8.8, 5.6))
    for row in rows:
        if row.get("steering_condition") == "baseline":
            continue
        x = _f(row, "specificity_gap_vs_best_control")
        y = _f(row, "content_damage")
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        cond = str(row.get("steering_condition"))
        trait = str(row.get("trait"))
        size = 70 + 250 * max(0.0, _f(row, "private_experience_rate", 0.0))
        ax.scatter(x, y, s=size, color=_lab17_color(cond), marker=_lab17_marker(trait), alpha=0.82, edgecolor="#222222", linewidth=0.5)
        if cond == "trait_direction":
            ax.text(x, y, trait.replace("_", "\n"), fontsize=7, ha="center", va="bottom")
    ax.axvline(MIN_STEERING_SPECIFICITY_GAP, color="#222222", linestyle=":", linewidth=1.0, label="specificity bar")
    ax.axhline(max(0.0, -MIN_CONTENT_DELTA), color="#777777", linestyle="--", linewidth=1.0, label="content-cost ceiling")
    ax.set_xlabel("style specificity over strongest control")
    ax.set_ylabel("content damage = max(0, -content delta)")
    ax.set_title("Steering operating frontier: move style without shredding the answer")
    bench.style_ax(ax)
    bench.save_figure(ctx, fig, "steering_operating_frontier.png", "Dose operating frontier for persona/register steering specificity versus content and boundary costs.")


def plot_generation_style_atlas(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    trait_rows = [row for row in rows if row.get("steering_condition") == "trait_direction"]
    if not trait_rows:
        return
    traits = sorted({str(row.get("trait")) for row in trait_rows})
    doses = sorted({_f(row, "dose_fraction") for row in trait_rows if math.isfinite(_f(row, "dose_fraction"))})
    fig, axes = plt.subplots(1, 2, figsize=(12.0, max(4.5, 0.55 * len(traits) + 1.7)), sharey=True)
    for ax, key, title, vmin, vmax, cmap in (
        (axes[0], "style_margin_delta", "style delta", -0.5, 1.0, "RdBu_r"),
        (axes[1], "content_hit_delta", "content delta", -0.5, 0.5, "RdBu_r"),
    ):
        mat = []
        for trait in traits:
            rowvals = []
            for dose in doses:
                val = safe_fmean([_f(row, key) for row in trait_rows if row.get("trait") == trait and _f(row, "dose_fraction") == dose])
                rowvals.append(val)
            mat.append(rowvals)
        arr = np.array(mat, dtype=float)
        im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(doses)))
        ax.set_xticklabels([str(rounded(d)) for d in doses])
        ax.set_yticks(range(len(traits)))
        ax.set_yticklabels(traits)
        _imshow_with_numbers(ax, arr, fmt=".2f", color_threshold=0.35)
        ax.set_title(title)
        ax.set_xlabel("dose")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        bench.style_ax(ax, legend=False)
    fig.suptitle("Trait-direction generation atlas: style movement and content preservation")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    bench.save_figure(ctx, fig, "generation_style_atlas.png", "Trait-by-dose atlas of style-marker and content-keyword deltas under trait-direction steering.")


def plot_style_content_tradeoff(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(8.8, 5.8))
    max_dose = _max_dose_from_effects(rows)
    for row in rows:
        if _f(row, "dose_fraction") != max_dose or row.get("steering_condition") == "baseline":
            continue
        x = _f(row, "style_margin_delta_vs_baseline")
        y = _f(row, "content_hit_delta_vs_baseline")
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        cond = str(row.get("steering_condition"))
        trait = str(row.get("trait"))
        ax.scatter(x, y, s=105 if cond == "trait_direction" else 65, color=_lab17_color(cond), marker=_lab17_marker(trait), alpha=0.85, edgecolor="#222222", linewidth=0.55, label=cond.replace("_", " ") if cond not in ax.get_legend_handles_labels()[1] else None)
        if cond == "trait_direction":
            ax.text(x, y, trait.replace("_", "\n"), fontsize=7, ha="center", va="bottom")
    ax.axhline(MIN_CONTENT_DELTA, linewidth=1.0, alpha=0.65, color="#777777", linestyle="--", label="content floor")
    ax.axvline(0.0, linewidth=1.0, alpha=0.55, color="#222222")
    ax.set_xlabel(f"style-marker delta at max dose ({rounded(max_dose)})")
    ax.set_ylabel("content-hit delta at max dose")
    ax.set_title("Content vs style: handles must preserve the task")
    bench.style_ax(ax)
    bench.save_figure(ctx, fig, "style_content_tradeoff.png", "Style movement versus content preservation by trait and steering control.")


def plot_direction_cosines(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    names = sorted({str(row["direction_a"]) for row in rows})
    if not names:
        return
    idx = {name: i for i, name in enumerate(names)}
    matrix = [[float("nan") for _ in names] for _ in names]
    for row in rows:
        a = str(row["direction_a"]); b = str(row["direction_b"])
        matrix[idx[a]][idx[b]] = _f(row, "cosine")
    arr = np.array(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(max(7.5, 0.58 * len(names)), max(6.0, 0.55 * len(names))))
    im = ax.imshow(arr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    _imshow_with_numbers(ax, arr, fmt=".2f", color_threshold=0.55)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine")
    ax.set_title("Direction cosine audit: persona/register handles vs style and safety controls")
    bench.style_ax(ax, legend=False)
    bench.save_figure(ctx, fig, "direction_cosine_heatmap.png", "Cosine map among persona, register, voice, agreement, sentiment, and refusal directions.")


def plot_direction_confound_risk(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(8.8, max(4.6, 0.42 * len(rows) + 1.5)))
    rows = sorted(rows, key=lambda r: _f(r, "nearest_control_abs_cosine", 0.0))
    vals = [_f(row, "nearest_control_abs_cosine", 0.0) for row in rows]
    labels = [str(row.get("direction")) + ("\n→ " + str(row.get("nearest_control_or_style_direction")) if row.get("nearest_control_or_style_direction") else "") for row in rows]
    colors = [_lab17_color("failed" if v >= 0.70 else ("warning" if v >= 0.45 else "positive")) for v in vals]
    ax.barh(range(len(rows)), vals, color=colors, alpha=0.9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0.45, color="#E69F00", linestyle=":", linewidth=1.0, label="medium risk")
    ax.axvline(0.70, color="#D55E00", linestyle="--", linewidth=1.0, label="high risk")
    ax.set_xlabel("nearest abs cosine to style/safety/control direction")
    ax.set_title("Direction confound risk: when persona geometry collapses into a cheaper axis")
    bench.style_ax(ax)
    bench.save_figure(ctx, fig, "direction_confound_risk.png", "Nearest-control cosine risk for each persona/register/voice direction.")


def plot_persona_turn_trace(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(9.4, 5.6))
    styles = {
        ("museum_roleplay", "persona_museum_guide"): (_lab17_color("persona_museum_guide"), "-", "roleplay: museum guide"),
        ("museum_roleplay", "default_assistant_control"): (_lab17_color("default_assistant_control"), "--", "roleplay: default-control"),
        ("museum_roleplay", "sentiment_style_control"): (_lab17_color("sentiment_style_control"), ":", "roleplay: sentiment control"),
        ("museum_roleplay", "random_null"): (_lab17_color("random_null"), ":", "roleplay: random null"),
        ("default_control", "persona_museum_guide"): (_lab17_color("real"), "--", "default transcript: museum direction"),
    }
    for (conv, direction), (color, linestyle, label) in styles.items():
        sub = sorted([row for row in rows if row["conversation"] == conv and row["direction"] == direction and row["role"] != "system"], key=lambda r: _f(r, "turn_index_non_system", 0.0))
        if not sub:
            continue
        xs = [_f(row, "turn_index_non_system") for row in sub]
        ys = [_f(row, "cumulative_projection") for row in sub]
        ax.plot(xs, ys, marker="o", color=color, linestyle=linestyle, linewidth=2.0, label=label)
    ax.set_xlabel("message boundary (system excluded)")
    ax.set_ylabel("cumulative mean projection")
    ax.set_title("Persona trace: sustained roleplay versus default and null rails")
    bench.style_ax(ax)
    bench.save_figure(ctx, fig, "persona_turn_trace.png", "Museum-guide persona projection over scripted turns with default, sentiment, and random controls.")


def plot_register_switch_trace(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(9.4, 5.6))
    styles = {
        "technical_register": (_lab17_color("technical_register"), "-", "technical register"),
        "casual_register_control": (_lab17_color("warm_supportive_voice"), "--", "casual-register control"),
        "sentiment_style_control": (_lab17_color("sentiment_style_control"), ":", "sentiment control"),
        "random_null": (_lab17_color("random_null"), ":", "random null"),
    }
    for direction, (color, linestyle, label) in styles.items():
        sub = sorted([row for row in rows if row["conversation"] == "register_switch" and row["direction"] == direction and row["role"] != "system"], key=lambda r: _f(r, "turn_index_non_system", 0.0))
        if not sub:
            continue
        ax.plot([_f(row, "turn_index_non_system") for row in sub], [_f(row, "content_boundary_projection") for row in sub], marker="o", color=color, linestyle=linestyle, linewidth=2.0, label=label)
    ax.set_xlabel("message boundary (system excluded)")
    ax.set_ylabel("content-boundary projection")
    ax.set_title("Register switch trace: projection should move after the explicit switch")
    bench.style_ax(ax)
    bench.save_figure(ctx, fig, "register_switch_trace.png", "Technical/casual register projection through a scripted switch.")


def plot_refusal_projection(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = bench.new_figure(figsize=(9.4, 5.6))
    styles = {
        ("roleplay_boundary", "refusal_monitor"): (_lab17_color("refusal_monitor"), "-", "refusal monitor"),
        ("roleplay_boundary", "persona_museum_guide"): (_lab17_color("persona_museum_guide"), "--", "persona"),
        ("roleplay_boundary", "sentiment_style_control"): (_lab17_color("sentiment_style_control"), ":", "sentiment control"),
        ("roleplay_boundary", "random_null"): (_lab17_color("random_null"), ":", "random null"),
    }
    for (conv, direction), (color, linestyle, label) in styles.items():
        sub = sorted([row for row in rows if row["conversation"] == conv and row["direction"] == direction and row["role"] != "system"], key=lambda r: _f(r, "turn_index_non_system", 0.0))
        if not sub:
            continue
        ax.plot([_f(row, "turn_index_non_system") for row in sub], [_f(row, "content_boundary_projection") for row in sub], marker="o", color=color, linestyle=linestyle, linewidth=2.0, label=label)
    ax.set_xlabel("message boundary (system excluded)")
    ax.set_ylabel("content-boundary projection")
    ax.set_title("Refusal monitor under benign roleplay: diagnostic, not jailbreak search")
    bench.style_ax(ax)
    bench.save_figure(ctx, fig, "refusal_projection_under_roleplay.png", "Refusal-monitor projection in a benign roleplay boundary conversation.")


def plot_trace_depth_sweep(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]], best_depth: int) -> None:
    fig, ax = bench.new_figure(figsize=(9.6, 5.4))
    for conv, direction, label in (
        ("museum_roleplay", "persona_museum_guide", "roleplay persona"),
        ("museum_roleplay", "random_null", "roleplay random"),
        ("register_switch", "technical_register", "register switch"),
        ("roleplay_boundary", "refusal_monitor", "refusal boundary"),
    ):
        sub = sorted([row for row in rows if row.get("conversation") == conv and row.get("direction") == direction], key=lambda r: _f(r, "projection_stream_depth", 0.0))
        if not sub:
            continue
        color = _lab17_color(direction)
        ax.plot([_f(row, "projection_stream_depth") for row in sub], [_f(row, "cumulative_projection_slope") for row in sub], marker="o", linewidth=2.0, label=label, color=color)
    ax.axvline(best_depth, linewidth=1.0, alpha=0.7, color=_lab17_color("persona"), linestyle="--", label=f"fit depth {best_depth}")
    ax.axhline(0.0, linewidth=1.0, alpha=0.5, color="#222222")
    ax.set_xlabel("projection stream depth")
    ax.set_ylabel("cumulative projection slope")
    ax.set_title("Trace depth sweep: descriptive after the direction has already been chosen")
    bench.style_ax(ax)
    bench.save_figure(ctx, fig, "trace_depth_sweep.png", "Descriptive sweep of turn-trace slopes across stream depths.")


def plot_trace_evidence_atlas(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    rows = [row for row in rows if row.get("projection_measure") in {"cumulative_projection", "content_boundary_projection"}]
    if not rows:
        return
    labels = sorted({f"{row.get('conversation')}\n{row.get('direction')}" for row in rows})
    measures = sorted({str(row.get("projection_measure")) for row in rows})
    mat = []
    for label in labels:
        conv, direction = label.split("\n", 1)
        mat.append([safe_fmean([_f(row, "gap_vs_random_null") for row in rows if row.get("conversation") == conv and row.get("direction") == direction and row.get("projection_measure") == measure]) for measure in measures])
    arr = np.array(mat, dtype=float)
    fig, ax = plt.subplots(figsize=(8.6, max(6.0, 0.22 * len(labels) + 2.0)))
    im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", vmin=-0.12, vmax=0.12)
    ax.set_xticks(range(len(measures)))
    ax.set_xticklabels([m.replace("_", "\n") for m in measures])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    _imshow_with_numbers(ax, arr, fmt=".2f", color_threshold=0.07)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="slope gap vs random-null")
    ax.set_title("Multi-turn trace evidence atlas")
    bench.style_ax(ax, legend=False)
    bench.save_figure(ctx, fig, "trace_evidence_atlas.png", "Conversation-by-direction trace slope gaps versus random-null controls.")


def plot_persona_trace_projection_atlas(ctx: bench.RunContext, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    keep_dirs = ["persona_museum_guide", "technical_register", "warm_supportive_voice", "honest_correction", "refusal_monitor", "sentiment_style_control", "random_null"]
    convs = sorted({str(row.get("conversation")) for row in rows if row.get("conversation")})
    if not convs:
        return
    fig, axes = plt.subplots(len(convs), 1, figsize=(11.0, max(4.0, 2.2 * len(convs))), squeeze=False)
    for ax, conv in zip(axes[:, 0], convs):
        sub = [row for row in rows if row.get("conversation") == conv and row.get("direction") in keep_dirs and row.get("role") != "system"]
        dirs = [d for d in keep_dirs if any(row.get("direction") == d for row in sub)]
        turns = sorted({_f(row, "turn_index_non_system") for row in sub if math.isfinite(_f(row, "turn_index_non_system"))})
        mat = []
        for direction in dirs:
            mat.append([safe_fmean([_f(row, "content_boundary_projection") for row in sub if row.get("direction") == direction and _f(row, "turn_index_non_system") == turn]) for turn in turns])
        arr = np.array(mat, dtype=float)
        if arr.size == 0:
            continue
        lim = max(0.1, float(np.nanpercentile(np.abs(arr[np.isfinite(arr)]), 90)) if np.isfinite(arr).any() else 0.1)
        im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_yticks(range(len(dirs)))
        ax.set_yticklabels(dirs, fontsize=7)
        ax.set_xticks(range(len(turns)))
        ax.set_xticklabels([str(int(t)) if float(t).is_integer() else str(rounded(t)) for t in turns])
        ax.set_title(conv)
        ax.set_ylabel("direction")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.012)
        bench.style_ax(ax, legend=False)
    axes[-1, 0].set_xlabel("message boundary (system excluded)")
    fig.suptitle("Turn-by-turn projection atlas at content boundaries")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    bench.save_figure(ctx, fig, "persona_trace_projection_atlas.png", "Turn-by-turn content-boundary projection atlas across scripted conversations.")


def plot_refusal_boundary_safety_dashboard(ctx: bench.RunContext, turn_rows: Sequence[Mapping[str, Any]], trace_rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9))
    ax = axes[0]
    for direction in ("refusal_monitor", "persona_museum_guide", "sentiment_style_control", "random_null"):
        sub = sorted([row for row in turn_rows if row.get("conversation") == "roleplay_boundary" and row.get("direction") == direction and row.get("role") != "system"], key=lambda r: _f(r, "turn_index_non_system", 0.0))
        if sub:
            ax.plot([_f(row, "turn_index_non_system") for row in sub], [_f(row, "content_boundary_projection") for row in sub], marker="o", linewidth=2.0, color=_lab17_color(direction), label=direction)
    bench.style_ax(ax, title="Benign boundary trace", xlabel="turn", ylabel="content-boundary projection")
    ax = axes[1]
    sub = [row for row in trace_rows if row.get("conversation") == "roleplay_boundary" and row.get("projection_measure") == "content_boundary_projection" and row.get("direction") in {"refusal_monitor", "persona_museum_guide", "sentiment_style_control"}]
    labels = [str(row.get("direction")) for row in sub]
    vals = [_f(row, "gap_vs_random_null", 0.0) for row in sub]
    ax.barh(range(len(labels)), vals, color=[_lab17_color(label) for label in labels])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.axvline(0, color="#222222", linewidth=0.8)
    bench.style_ax(ax, title="Gap vs random-null", xlabel="slope gap", ylabel="")
    fig.suptitle("Refusal safety dashboard: monitor-only under benign roleplay")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    bench.save_figure(ctx, fig, "refusal_boundary_safety_dashboard.png", "Monitor-only refusal projection dashboard for the benign roleplay-boundary conversation.")


# ---------------------------------------------------------------------------
# Metrics, cards, and summaries
# ---------------------------------------------------------------------------


def metric_at(rows: Sequence[Mapping[str, Any]], trait: str, kind: str, depth: int, key: str = "auc", probe_split: str = "eval") -> float:
    vals = [
        float(row[key]) for row in rows
        if row.get("trait") == trait
        and row.get("direction_kind") == kind
        and row.get("probe_split") == probe_split
        and row.get("depth") == depth
        and isinstance(row.get(key), (int, float))
    ]
    return safe_fmean(vals)


def effect_delta(
    rows: Sequence[Mapping[str, Any]],
    trait: str,
    condition: str,
    key: str,
    *,
    dose_fraction: float | None = None,
) -> float:
    vals = []
    for row in rows:
        if row.get("trait") != trait or row.get("steering_condition") != condition:
            continue
        if dose_fraction is not None and float(row.get("dose_fraction", -999)) != float(dose_fraction):
            continue
        if isinstance(row.get(key), (int, float)):
            vals.append(float(row[key]))
    return safe_fmean(vals)


def trace_slope(
    rows: Sequence[Mapping[str, Any]],
    conversation: str,
    direction: str,
    *,
    projection_measure: str = "cumulative_projection",
) -> float:
    vals = [
        float(row["projection_slope"])
        for row in rows
        if row.get("conversation") == conversation
        and row.get("direction") == direction
        and row.get("projection_measure") == projection_measure
        and isinstance(row.get("projection_slope"), (int, float))
    ]
    return safe_fmean(vals)


def aggregate_best_probe_metrics(
    items: Sequence[PersonaPair],
    probe_rows: Sequence[Mapping[str, Any]],
    best_depth: int,
) -> dict[str, Any]:
    traits = sorted({item.trait for item in items})
    real_aucs = [metric_at(probe_rows, trait, "real", best_depth) for trait in traits]
    shuf_aucs = [metric_at(probe_rows, trait, "shuffled_sign", best_depth) for trait in traits]
    random_aucs = [metric_at(probe_rows, trait, "random_oriented", best_depth) for trait in traits]
    rows = []
    for trait in traits:
        rows.append({
            "trait": trait,
            "eval_real_auc": none_if_nan(metric_at(probe_rows, trait, "real", best_depth)),
            "eval_shuffled_auc": none_if_nan(metric_at(probe_rows, trait, "shuffled_sign", best_depth)),
            "eval_random_auc": none_if_nan(metric_at(probe_rows, trait, "random_oriented", best_depth)),
            "eval_selectivity_vs_shuffled": none_if_nan(metric_at(probe_rows, trait, "real", best_depth) - metric_at(probe_rows, trait, "shuffled_sign", best_depth)),
            "eval_selectivity_vs_random": none_if_nan(metric_at(probe_rows, trait, "real", best_depth) - metric_at(probe_rows, trait, "random_oriented", best_depth)),
        })
    return {
        "traits": traits,
        "per_trait_rows": rows,
        "mean_real_auc_best_depth": none_if_nan(safe_fmean(real_aucs)),
        "mean_shuffled_auc_best_depth": none_if_nan(safe_fmean(shuf_aucs)),
        "mean_random_auc_best_depth": none_if_nan(safe_fmean(random_aucs)),
        "mean_real_selectivity_vs_shuffled": none_if_nan(safe_fmean(real_aucs) - safe_fmean(shuf_aucs)),
        "mean_real_selectivity_vs_random": none_if_nan(safe_fmean(real_aucs) - safe_fmean(random_aucs)),
    }


def write_safety_scope(ctx: bench.RunContext, depth: int) -> None:
    audit = {
        "lab_id": LAB_ID,
        "refusal_direction_use": "monitor-only direction for benign roleplay trace",
        "refusal_direction_extraction": "forward passes only over short refusal/helpful contrast prompts",
        "no_refusal_eliciting_generation": True,
        "no_refusal_ablation_or_jailbreak_search": True,
        "trace_scope": "benign scripted roleplay boundary conversation",
        "stream_depth": depth,
        "allowed_use_in_this_lab": "detect whether refusal-monitor projection erodes or remains stable under benign roleplay wording",
    }
    path = ctx.path("diagnostics", "persona_safety_scope.json")
    bench.write_json(path, audit)
    ctx.register_artifact(path, "diagnostic", "Safety scope for refusal-monitor use in Lab 17.")


def write_generation_labeling_guide(ctx: bench.RunContext) -> None:
    lines = [
        "# Lab 17 Generation Labeling Guide",
        "",
        "The automatic marker rubric is a sorting tray, not a judge. For any result you use in a writeup, fill the hand-label columns in `tables/persona_steering_generations.csv`.",
        "",
        "## Suggested labels",
        "",
        "`hand_label_style`: `positive`, `negative`, `mixed`, `none`, or `bad_parse`.",
        "",
        "`hand_label_content`: `preserved`, `partly_preserved`, `changed`, `wrong`, or `ungraded`.",
        "",
        "`hand_label_boundary`: `ok`, `private_experience_claim`, `unsafe_boundary_eroded`, `refusal_overtriggered`, or `ungraded`.",
        "",
        "## What to watch for",
        "",
        "A persona handle is not impressive if it only injects a catchphrase. A register handle is not impressive if it ruins the answer. A warm voice handle is not impressive if it fabricates private experience. Treat the hand labels as the little tribunal before the ledger claim enters town.",
        "",
    ]
    path = ctx.path("tables", "generation_labeling_guide.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "guide", "Manual labeling guide for Lab 17 generations.")


def write_persona_state_card(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    verdict = str(metrics.get("verdict", "unknown"))
    lines = [
        "# Lab 17 Persona-State Frame Card",
        "",
        "Read this before reading the plots. The lab is looking for a handle on prompt-framed persona, voice, register, and agreement style. It is not looking for a private identity.",
        "",
        "## Run verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Model: `{metrics.get('model_id')}`",
        f"- Selected stream depth: {metrics.get('best_depth')}",
        f"- Injection layer for steering: {metrics.get('injection_layer')}",
        f"- Eval real AUC: {metrics.get('mean_real_auc_best_depth')}",
        f"- Eval shuffled/random AUC: {metrics.get('mean_shuffled_auc_best_depth')} / {metrics.get('mean_random_auc_best_depth')}",
        f"- Selectivity vs shuffled/random: {metrics.get('mean_real_selectivity_vs_shuffled')} / {metrics.get('mean_real_selectivity_vs_random')}",
        f"- Max-dose trait steering style delta: {metrics.get('mean_trait_steering_style_delta_max_dose')}",
        f"- Max-dose random steering style delta: {metrics.get('mean_random_steering_style_delta_max_dose')}",
        f"- Max-dose shuffled steering style delta: {metrics.get('mean_shuffled_steering_style_delta_max_dose')}",
        f"- Mean content delta under trait steering: {metrics.get('mean_trait_steering_content_delta_max_dose')}",
        f"- Museum roleplay persona slope: {metrics.get('museum_roleplay_persona_slope')}",
        f"- Museum roleplay random-null slope: {metrics.get('museum_roleplay_random_slope')}",
        "",
        "## Decision rule",
        "",
        "A strong handle requires all three: held-out decodability beats controls, steering changes style beyond random/shuffled/opposite controls, and content is not badly damaged. Multi-turn traces are descriptive unless they exceed the null trace and survive turn-boundary checks.",
        "",
        "## Non-claims",
        "",
        "This card does not claim the model has a real self, a true character, human-like voice ownership, or a jailbreak. It reports a measured coordinate frame and the controls that tried to kill it.",
        "",
    ]
    path = ctx.path("persona_state_frame_card.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "card", "Read-first verdict card for Lab 17.")


def write_operationalization_audit(ctx: bench.RunContext, metrics: Mapping[str, Any]) -> None:
    lines = [
        "# Lab 17 Operationalization Audit",
        "",
        "## What was measured",
        "",
        "The lab measures paired residual-stream directions for prompt-framed persona, register, voice, and agreement contrasts. It measures whether those directions separate matched prompts, steer generated wording, and leave traces in scripted chat transcripts.",
        "",
        "It does not measure a private self, a durable identity, subjective experience, or an author in the human sense.",
        "",
        "## Cheap explanations and their controls",
        "",
        "| Cheap explanation | Control artifact | What would kill the strong claim? |",
        "|---|---|---|",
        "| Formatting or chat-template residue | `diagnostics/exact_chat_hook_parity.json`, `tables/turn_segments.csv`, `diagnostics/turn_boundary_check.json` | Segment checks fail, or traces are driven by role tokens rather than content spans. |",
        "| Marker-only style | `tables/persona_steering_generations.csv`, `tables/generation_labeling_guide.md` | Hand labels show the model only adds catchphrases. |",
        "| Topic leakage | `diagnostics/split_audit.csv`, `diagnostics/split_balance.csv` | Train and eval topics leak or a trait vanishes on held-out topics. |",
        "| Random linear handles | `tables/persona_probe_report.csv`, `plots/persona_probe_selectivity.png` | Random or shuffled controls match the real direction. |",
        "| Politeness/sentiment rather than persona | `tables/direction_cosines.csv`, `plots/direction_cosine_heatmap.png` | Persona/register axes collapse onto warm/supportive or sentiment controls. |",
        "| Style at the cost of task behavior | `tables/persona_steering_effects.csv`, `plots/style_content_tradeoff.png` | Content-hit rate falls while style markers rise. |",
        "| Refusal erosion under roleplay | `diagnostics/persona_safety_scope.json`, `plots/refusal_projection_under_roleplay.png` | The lab only monitors benign traces and must not become prompt optimization. |",
        "",
        "## Current run",
        "",
        f"- Best depth: {metrics.get('best_depth')}",
        f"- Mean real AUC at best depth: {metrics.get('mean_real_auc_best_depth')}",
        f"- Mean shuffled/random AUC at best depth: {metrics.get('mean_shuffled_auc_best_depth')} / {metrics.get('mean_random_auc_best_depth')}",
        f"- Mean trait-direction steering style delta: {metrics.get('mean_trait_steering_style_delta_max_dose')}",
        f"- Mean random-direction steering style delta: {metrics.get('mean_random_steering_style_delta_max_dose')}",
        f"- Mean shuffled-direction steering style delta: {metrics.get('mean_shuffled_steering_style_delta_max_dose')}",
        f"- Mean trait-direction content delta: {metrics.get('mean_trait_steering_content_delta_max_dose')}",
        f"- Museum roleplay persona slope: {metrics.get('museum_roleplay_persona_slope')}",
        f"- Museum roleplay random-null slope: {metrics.get('museum_roleplay_random_slope')}",
        f"- Refusal-monitor slope in boundary conversation: {metrics.get('refusal_boundary_slope')}",
        "",
        "## Allowed claim",
        "",
        "A persona/register/voice claim is allowed only when held-out probe selectivity beats controls, steering changes style more than random/shuffled controls without destroying content, and multi-turn traces exceed a null trace. If the effect is only surface style, that is the result.",
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
        f"- Rows: {metrics.get('n_rows')} selected from `{metrics.get('data', {}).get('data_source')}`",
        f"- Best depth: {metrics.get('best_depth')} selected by train-only control-adjusted score",
        f"- Injection layer: {metrics.get('injection_layer')}",
        f"- Mean held-out real AUC: {metrics.get('mean_real_auc_best_depth')}",
        f"- Mean held-out shuffled/random AUC: {metrics.get('mean_shuffled_auc_best_depth')} / {metrics.get('mean_random_auc_best_depth')}",
        f"- Max-dose trait steering style delta: {metrics.get('mean_trait_steering_style_delta_max_dose')}",
        f"- Max-dose trait steering content delta: {metrics.get('mean_trait_steering_content_delta_max_dose')}",
        f"- Museum roleplay persona/random slopes: {metrics.get('museum_roleplay_persona_slope')} / {metrics.get('museum_roleplay_random_slope')}",
        f"- Verdict: `{metrics.get('verdict')}`",
        "",
        "Start with `persona_state_frame_card.md` and `operationalization_audit.md`. The plots are evidence exhibits, not a personality test.",
        "",
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Human-readable summary of headline Lab 17 metrics.")


def make_metrics(
    *,
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    items: Sequence[PersonaPair],
    eval_items: Sequence[PersonaPair],
    data_info: Mapping[str, Any],
    probe_rows: Sequence[Mapping[str, Any]],
    depth_rows: Sequence[Mapping[str, Any]],
    best_depth: int,
    steering_effects: Sequence[Mapping[str, Any]],
    turn_slopes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    probe = aggregate_best_probe_metrics(items, probe_rows, best_depth)
    max_dose = max(STEERING_DOSE_FRACTIONS)
    trait_deltas = [
        float(row["style_margin_delta_vs_baseline"])
        for row in steering_effects
        if row["steering_condition"] == "trait_direction"
        and float(row.get("dose_fraction", -1)) == max_dose
        and isinstance(row.get("style_margin_delta_vs_baseline"), (int, float))
    ]
    random_deltas = [
        float(row["style_margin_delta_vs_baseline"])
        for row in steering_effects
        if row["steering_condition"] == "random_direction"
        and float(row.get("dose_fraction", -1)) == max_dose
        and isinstance(row.get("style_margin_delta_vs_baseline"), (int, float))
    ]
    shuffled_deltas = [
        float(row["style_margin_delta_vs_baseline"])
        for row in steering_effects
        if row["steering_condition"] == "shuffled_sign_direction"
        and float(row.get("dose_fraction", -1)) == max_dose
        and isinstance(row.get("style_margin_delta_vs_baseline"), (int, float))
    ]
    opposite_deltas = [
        float(row["style_margin_delta_vs_baseline"])
        for row in steering_effects
        if row["steering_condition"] == "opposite_direction"
        and float(row.get("dose_fraction", -1)) == max_dose
        and isinstance(row.get("style_margin_delta_vs_baseline"), (int, float))
    ]
    content_deltas = [
        float(row["content_hit_delta_vs_baseline"])
        for row in steering_effects
        if row["steering_condition"] == "trait_direction"
        and float(row.get("dose_fraction", -1)) == max_dose
        and isinstance(row.get("content_hit_delta_vs_baseline"), (int, float))
    ]
    train_selection_score = safe_fmean([
        float(row["control_adjusted_score"])
        for row in depth_rows
        if row.get("probe_split") == "train_loo"
        and row.get("depth") == best_depth
        and isinstance(row.get("control_adjusted_score"), (int, float))
    ])
    steering_specificity_gap = safe_fmean(trait_deltas) - max(safe_fmean(random_deltas, 0.0), safe_fmean(shuffled_deltas, 0.0))
    trace_gap = trace_slope(turn_slopes, "museum_roleplay", "persona_museum_guide") - trace_slope(turn_slopes, "museum_roleplay", "random_null")

    selectivity_ok = (
        isinstance(probe["mean_real_selectivity_vs_shuffled"], (int, float))
        and float(probe["mean_real_selectivity_vs_shuffled"]) >= MIN_SELECTIVITY_GAP
        and isinstance(probe["mean_real_selectivity_vs_random"], (int, float))
        and float(probe["mean_real_selectivity_vs_random"]) >= MIN_SELECTIVITY_GAP
    )
    steering_ok = math.isfinite(steering_specificity_gap) and steering_specificity_gap >= MIN_STEERING_SPECIFICITY_GAP
    content_ok = safe_fmean(content_deltas, -1.0) >= MIN_CONTENT_DELTA
    if selectivity_ok and steering_ok and content_ok:
        verdict = "validated_style_persona_handle"
    elif selectivity_ok and not steering_ok:
        verdict = "decodable_but_not_steerable_by_this_run"
    elif steering_ok and not selectivity_ok:
        verdict = "steers_behavior_but_probe_controls_weak"
    else:
        verdict = "not_validated_by_controls"

    metrics = {
        "model_id": bundle.anatomy.model_id,
        "n_rows": len(items),
        "n_eval_rows": len(eval_items),
        "best_depth": best_depth,
        "injection_layer": max(0, best_depth - 1),
        "stream_depth_convention": "bench streams[k]: 0 = embeddings, k = residual after k blocks; steering into block layer uses layer = depth - 1",
        "train_selection_score_best_depth": none_if_nan(train_selection_score),
        **{k: v for k, v in probe.items() if k != "per_trait_rows"},
        "mean_trait_steering_style_delta_max_dose": rounded(safe_fmean(trait_deltas)),
        "mean_random_steering_style_delta_max_dose": rounded(safe_fmean(random_deltas)),
        "mean_shuffled_steering_style_delta_max_dose": rounded(safe_fmean(shuffled_deltas)),
        "mean_opposite_steering_style_delta_max_dose": rounded(safe_fmean(opposite_deltas)),
        "mean_trait_steering_content_delta_max_dose": rounded(safe_fmean(content_deltas)),
        "steering_specificity_gap_vs_best_control": rounded(steering_specificity_gap),
        "technical_register_style_delta": none_if_nan(effect_delta(steering_effects, "technical_register", "trait_direction", "style_margin_delta_vs_baseline", dose_fraction=max_dose)),
        "technical_register_content_delta": none_if_nan(effect_delta(steering_effects, "technical_register", "trait_direction", "content_hit_delta_vs_baseline", dose_fraction=max_dose)),
        "museum_roleplay_persona_slope": none_if_nan(trace_slope(turn_slopes, "museum_roleplay", "persona_museum_guide")),
        "museum_roleplay_default_slope": none_if_nan(trace_slope(turn_slopes, "museum_roleplay", "default_assistant_control")),
        "museum_roleplay_random_slope": none_if_nan(trace_slope(turn_slopes, "museum_roleplay", "random_null")),
        "museum_roleplay_trace_gap_vs_random": none_if_nan(trace_gap),
        "register_switch_technical_slope": none_if_nan(trace_slope(turn_slopes, "register_switch", "technical_register", projection_measure="content_boundary_projection")),
        "refusal_boundary_slope": none_if_nan(trace_slope(turn_slopes, "roleplay_boundary", "refusal_monitor", projection_measure="content_boundary_projection")),
        "steering_dose_fractions": list(STEERING_DOSE_FRACTIONS),
        "selectivity_ok": bool(selectivity_ok),
        "steering_ok": bool(steering_ok),
        "content_ok": bool(content_ok),
        "verdict": verdict,
        "data": dict(data_info),
    }
    return metrics


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    import torch

    args = ctx.args
    if not bench.supports_chat_template(bundle):
        raise RuntimeError("Lab 17 requires an instruct model with a chat template.")

    items, data_info, selected_manifest = load_items(args)
    print(f"[lab17] {data_info['n_rows_selected']} paired rows; prompt_set={args.prompt_set}; source={data_info['data_source']}")
    manifest_path = ctx.path("diagnostics", "frozen_data_manifest.json")
    bench.write_json(manifest_path, data_info)
    ctx.register_artifact(manifest_path, "diagnostic", "Frozen Lab 17 data hash, filters, fallback status, and counts.")
    family_path = ctx.path("tables", "persona_family_manifest.csv")
    bench.write_csv_with_context(ctx, family_path, selected_manifest)
    ctx.register_artifact(family_path, "table", "Trait and marker manifest for selected Lab 17 rows.")

    first_prompt = render_chat(bundle, items[0].prompt_positive)
    run_exact_chat_hook_parity(ctx, bundle, first_prompt)
    bench.run_lens_self_check(ctx, bundle, bench.run_with_residual_cache(bundle, first_prompt, add_special_tokens=False))

    split = make_split(items, int(args.seed))
    split_path = ctx.path("diagnostics", "split_audit.csv")
    bench.write_csv_with_context(ctx, split_path, split_rows(items, split))
    ctx.register_artifact(split_path, "diagnostic", "Trait-stratified train/eval split by topic, with prompt hashes.")
    split_balance_path = ctx.path("diagnostics", "split_balance.csv")
    bench.write_csv_with_context(ctx, split_balance_path, split_balance(items, split))
    ctx.register_artifact(split_balance_path, "diagnostic", "Train/eval counts by trait and topic.")

    feat_tensor, features = cache_pair_features(ctx, bundle, items)
    row_norms = feat_tensor.norm(dim=-1)
    norm_rows = [
        {
            "depth": depth,
            "mean_norm": rounded(safe_fmean(row_norms[:, depth].tolist())),
            "median_norm": rounded(safe_median(row_norms[:, depth].tolist())),
            "min_norm": rounded(float(row_norms[:, depth].min())),
            "max_norm": rounded(float(row_norms[:, depth].max())),
        }
        for depth in range(row_norms.shape[1])
    ]
    norm_path = ctx.path("diagnostics", "activation_norms_by_depth.csv")
    bench.write_csv_with_context(ctx, norm_path, norm_rows)
    ctx.register_artifact(norm_path, "diagnostic", "Contrast-prompt residual norm audit.")

    probe_rows, depth_rows, best_depth = run_probe_sweep(items, features, split, int(args.seed), bundle.anatomy.d_model)
    probe_path = ctx.path("tables", "persona_probe_report.csv")
    bench.write_csv_with_context(ctx, probe_path, probe_rows)
    ctx.register_artifact(probe_path, "table", "Persona/register/voice held-out probe sweep with train-only depth selection controls.")
    depth_path = ctx.path("tables", "persona_depth_selection.csv")
    bench.write_csv_with_context(ctx, depth_path, depth_rows)
    ctx.register_artifact(depth_path, "table", "Depth-selection table. Selected depth uses train_loo control-adjusted score only.")
    depth_json_path = ctx.path("diagnostics", "persona_depth_selection.json")
    bench.write_json(depth_json_path, {
        "selected_depth": best_depth,
        "selection_rule": "max train_loo mean(real_auc) - max(mean(shuffled_auc), mean(random_auc)); eval rows are report-only",
        "stream_depth_convention": "streams[k] = residual after k blocks; depth 0 embeddings; depth L final pre-norm stream",
        "injection_layer_for_steering": max(0, best_depth - 1),
    })
    ctx.register_artifact(depth_json_path, "diagnostic", "Machine-readable depth selection rule and chosen depth.")
    results_path = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results_path, probe_rows)
    ctx.register_artifact(results_path, "results", "Alias of persona_probe_report.csv for the standard run contract.")
    print(f"[lab17] selected persona/register stream depth {best_depth}")

    directions = build_directions_at_depth(items, features, split, best_depth)
    shuffled_directions = build_shuffled_directions_at_depth(items, features, split, best_depth, int(args.seed) + 71)
    refusal_direction = build_refusal_monitor_direction(bundle, best_depth)
    sentiment_direction = build_sentiment_style_direction(bundle, best_depth)
    write_safety_scope(ctx, best_depth)

    trace_dirs = {
        **directions,
        "refusal_monitor": refusal_direction,
        "sentiment_style_control": sentiment_direction,
    }
    cos_rows = direction_cosine_rows(trace_dirs)
    cos_path = ctx.path("tables", "direction_cosines.csv")
    bench.write_csv_with_context(ctx, cos_path, cos_rows)
    ctx.register_artifact(cos_path, "table", "Pairwise cosines among persona/register/voice/agreement/sentiment/refusal directions.")
    provenance_rows = [
        {
            "direction": name,
            "source": "paired_train_rows" if name in directions else "local_forward_pass_control",
            "stream_depth": best_depth,
            "fit_split": "train" if name in directions else "fixed internal contrast",
            "norm": rounded(float(vec.norm())),
            "downstream_use": "state save, cosine audit, steering if paired trait" if name in directions else "trace/cosine control only",
        }
        for name, vec in sorted(trace_dirs.items())
    ]
    prov_path = ctx.path("tables", "direction_provenance.csv")
    bench.write_csv_with_context(ctx, prov_path, provenance_rows)
    ctx.register_artifact(prov_path, "table", "Where each Lab 17 direction came from and how it may be used.")

    ref_norm = safe_fmean(row_norms[:, best_depth].tolist(), default=1.0)
    eval_items = selected_eval_rows(items, split)
    steering_generations, steering_effects = run_steering(
        bundle,
        eval_items,
        directions,
        shuffled_directions,
        best_depth,
        bundle.anatomy.d_model,
        int(args.seed),
        ref_norm,
    )
    generation_path = ctx.path("tables", "persona_steering_generations.csv")
    bench.write_csv_with_context(ctx, generation_path, steering_generations)
    ctx.register_artifact(generation_path, "table", "Baseline/trait/opposite/shuffled/random steered generations with hand-label scaffold.")
    effects_path = ctx.path("tables", "persona_steering_effects.csv")
    bench.write_csv_with_context(ctx, effects_path, steering_effects)
    ctx.register_artifact(effects_path, "table", "Style-marker, content-keyword, repetition, and boundary effects by trait, condition, and dose.")
    write_generation_labeling_guide(ctx)
    register_scores = [row for row in steering_generations if row["trait"] == "technical_register"]
    register_path = ctx.path("tables", "register_content_style_scores.csv")
    bench.write_csv_with_context(ctx, register_path, register_scores)
    ctx.register_artifact(register_path, "table", "Technical-register steering rows with content and style scores.")

    turn_rows, turn_slopes, trace_depth_rows, segment_rows, generation_boundary_rows, turn_check = run_turn_trace(
        ctx,
        bundle,
        directions,
        refusal_direction,
        sentiment_direction,
        best_depth,
        int(args.seed),
    )
    turn_check_path = ctx.path("diagnostics", "turn_boundary_check.json")
    bench.write_json(turn_check_path, turn_check)
    ctx.register_artifact(turn_check_path, "diagnostic", "Chat-template turn segmentation, content span, and generation-boundary checks for Lab 17 traces.")
    segment_path = ctx.path("tables", "turn_segments.csv")
    bench.write_csv_with_context(ctx, segment_path, segment_rows)
    ctx.register_artifact(segment_path, "table", "Message/content spans for scripted conversations.")
    gen_boundary_path = ctx.path("diagnostics", "generation_prompt_boundary_check.csv")
    bench.write_csv_with_context(ctx, gen_boundary_path, generation_boundary_rows)
    ctx.register_artifact(gen_boundary_path, "diagnostic", "Checks that add_generation_prompt=True cleanly extends user-turn prefixes.")
    trace_path = ctx.path("tables", "persona_turn_trace.csv")
    bench.write_csv_with_context(ctx, trace_path, turn_rows)
    ctx.register_artifact(trace_path, "table", "Per-turn persona/register/voice/refusal/sentiment/random projections over scripted conversations.")
    slope_path = ctx.path("tables", "persona_turn_trace_slopes.csv")
    bench.write_csv_with_context(ctx, slope_path, turn_slopes)
    ctx.register_artifact(slope_path, "table", "Projection slopes for Lab 17 scripted traces, by projection measure.")
    trace_depth_path = ctx.path("tables", "trace_depth_sweep.csv")
    bench.write_csv_with_context(ctx, trace_depth_path, trace_depth_rows)
    ctx.register_artifact(trace_depth_path, "table", "Descriptive trace slope sweep across stream depths using fixed best-depth directions.")

    state_common = {
        "depth": best_depth,
        "depth_convention": "bench streams[k]: 0 = embeddings, k = residual after block k",
        "injection_layer": max(0, best_depth - 1),
        "injection_layer_convention": "activation addition hook acts on block layer output; layer = stream_depth - 1",
        "read_site": "chat-templated final prompt token before assistant generation",
        "model_id": bundle.anatomy.model_id,
        "d_model": bundle.anatomy.d_model,
        "n_layers": bundle.anatomy.n_layers,
        "method": "train-split paired positive-minus-negative mass-mean directions; depth selected by train_loo control-adjusted score",
        "source_data": data_info,
    }
    persona_path = ctx.path("state", "persona_directions.pt")
    torch.save({**state_common, "directions": directions, "refusal_monitor": refusal_direction, "sentiment_style_control": sentiment_direction}, persona_path)
    ctx.register_artifact(persona_path, "tensor", "Persona/register/voice/agreement directions plus refusal and sentiment controls.")
    register_state_path = ctx.path("state", "register_direction.pt")
    if "technical_register" in directions:
        torch.save({**state_common, "direction": directions["technical_register"]}, register_state_path)
        ctx.register_artifact(register_state_path, "tensor", "Technical-register direction for downstream labs.")
    voice_state_path = ctx.path("state", "voice_directions.pt")
    torch.save({
        **state_common,
        "directions": {
            key: directions[key]
            for key in ("warm_supportive_voice", "honest_disagreement")
            if key in directions
        },
    }, voice_state_path)
    ctx.register_artifact(voice_state_path, "tensor", "Voice/agreement directions for downstream labs.")
    meta_path = ctx.path("state", "persona_voice_register_metadata.json")
    bench.write_json(meta_path, {**state_common, "directions": sorted(directions), "includes_refusal_monitor": True, "includes_sentiment_control": True})
    ctx.register_artifact(meta_path, "state", "Human-readable metadata for Lab 17 saved directions.")

    metrics = make_metrics(
        ctx=ctx,
        bundle=bundle,
        items=items,
        eval_items=eval_items,
        data_info=data_info,
        probe_rows=probe_rows,
        depth_rows=depth_rows,
        best_depth=best_depth,
        steering_effects=steering_effects,
        turn_slopes=turn_slopes,
    )
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 17 metrics and verdict.")
    per_trait_path = ctx.path("tables", "probe_best_depth_by_trait.csv")
    bench.write_csv_with_context(ctx, per_trait_path, aggregate_best_probe_metrics(items, probe_rows, best_depth)["per_trait_rows"])
    ctx.register_artifact(per_trait_path, "table", "Per-trait held-out AUC and control selectivity at the selected depth.")

    visual_tables = write_visual_synthesis_tables(
        ctx,
        items,
        probe_rows,
        depth_rows,
        best_depth,
        steering_effects,
        steering_generations,
        cos_rows,
        turn_slopes,
        turn_rows,
        trace_depth_rows,
    )

    if not args.no_plots:
        plot_persona_evidence_dashboard(
            ctx,
            depth_rows,
            visual_tables["trait_evidence_rows"],
            visual_tables["steering_operating_rows"],
            visual_tables["trace_evidence_rows"],
            visual_tables["direction_confound_rows"],
            best_depth,
        )
        plot_trait_evidence_matrix(ctx, visual_tables["trait_evidence_rows"])
        plot_depth_control_gap_atlas(ctx, visual_tables["depth_control_gap_rows"], best_depth)
        plot_probe_selectivity(ctx, probe_rows, best_depth)
        plot_steering_dose_response(ctx, steering_effects)
        plot_steering_operating_frontier(ctx, visual_tables["steering_operating_rows"])
        plot_generation_style_atlas(ctx, visual_tables["steering_operating_rows"])
        plot_style_content_tradeoff(ctx, steering_effects)
        plot_direction_cosines(ctx, cos_rows)
        plot_direction_confound_risk(ctx, visual_tables["direction_confound_rows"])
        plot_persona_turn_trace(ctx, turn_rows)
        plot_register_switch_trace(ctx, turn_rows)
        plot_refusal_projection(ctx, turn_rows)
        plot_trace_depth_sweep(ctx, trace_depth_rows, best_depth)
        plot_trace_evidence_atlas(ctx, visual_tables["trace_evidence_rows"])
        plot_persona_trace_projection_atlas(ctx, turn_rows)
        plot_refusal_boundary_safety_dashboard(ctx, turn_rows, visual_tables["trace_evidence_rows"])

    write_persona_state_card(ctx, metrics)
    write_operationalization_audit(ctx, metrics)
    write_run_summary(ctx, metrics)

    run_name = ctx.run_dir.name
    if metrics["verdict"] == "validated_style_persona_handle":
        causal_text = (
            f"For {bundle.anatomy.model_id}, max-dose trait-direction steering at layer "
            f"{metrics['injection_layer']} changed held-out style-marker margins by mean delta "
            f"{metrics['mean_trait_steering_style_delta_max_dose']} versus random/shuffled deltas "
            f"{metrics['mean_random_steering_style_delta_max_dose']} / "
            f"{metrics['mean_shuffled_steering_style_delta_max_dose']}, with mean content delta "
            f"{metrics['mean_trait_steering_content_delta_max_dose']}. This is a scoped style/persona handle claim, not evidence of a real identity."
        )
    else:
        causal_text = (
            f"For {bundle.anatomy.model_id}, Lab 17 did not validate a strong causal persona/register/voice handle under its controls "
            f"(verdict `{metrics['verdict']}`). Treat steering effects as exploratory until random/shuffled controls and content preservation clear the audit."
        )
    claims = [
        {
            "id": f"{LAB_ID}-C1",
            "tag": "DECODE",
            "text": (
                f"At stream depth {best_depth}, paired persona/register/voice directions separate held-out positive prompts from matched controls with mean AUC "
                f"{metrics['mean_real_auc_best_depth']} versus shuffled/random {metrics['mean_shuffled_auc_best_depth']} / {metrics['mean_random_auc_best_depth']}. "
                "This is decodability of the operationalized prompt frame, not personality."
            ),
            "artifact": f"runs/{run_name}/tables/persona_probe_report.csv",
            "falsifier": "Shuffled-sign or random controls match the AUC, or topic/style controls explain the direction without a reusable persona/register component.",
        },
        {
            "id": f"{LAB_ID}-C2",
            "tag": "CAUSAL",
            "text": causal_text,
            "artifact": f"runs/{run_name}/tables/persona_steering_effects.csv",
            "falsifier": "Random, shuffled, or opposite steering matches the effect; content accuracy collapses; or hand labels show the marker rubric is misclassifying style.",
        },
        {
            "id": f"{LAB_ID}-C3",
            "tag": "OBS/DECODE",
            "text": (
                f"In the scripted museum roleplay trace, the persona-direction slope was {metrics['museum_roleplay_persona_slope']} versus random-null "
                f"{metrics['museum_roleplay_random_slope']}. This is a turn-trace observation conditioned on Lab 15-style segmentation checks, not evidence of a durable inner character."
            ),
            "artifact": f"runs/{run_name}/tables/persona_turn_trace_slopes.csv",
            "falsifier": "The null trace matches the slope, turn segmentation fails, or content-span projections diverge from message-span projections in a way that points to template residue.",
        },
    ]
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)
