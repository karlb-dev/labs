"""Lab 1: residual stream and logit lens.

Core question
=============

How does a model's next-token prediction emerge across depth?

This file owns the experiment: prompt families, target validation, event-depth
metrics, category aggregates, plots, run summary, and suggested claim-ledger
entries. ``interp_bench.py`` owns the shared instrument: model loading,
anatomy resolution, residual capture, logit-lens math, self-checks, state
dumps, and artifact writing.

Evidence level: OBSERVATION. A raw logit lens can show that a token is
readable from an intermediate residual stream. It does not show that the model
"knows" the token there, nor that later layers causally use the information.

Design notes
============

The lab uses three main prompt families plus optional controls:

* ``fact``: high-certainty completions. These are expected to become sharp and
  target-like under the lens. We deliberately include both plain declarative
  forms ("The capital of France is") and answer-shaped forms ("... is the city
  of") plus some that trigger strong discourse continuations ("... is well
  known as") so students see the model’s actual next-token objective compete
  with the “fact” task.
* ``ambiguous``: prompts with no privileged single-token continuation. These
  are the negative control against overclaiming early commitment. Several are
  length- and syntax-matched to facts to make the entropy and stability
  contrast clean.
* ``counterfactual``: a local context overwrites a memorized fact. Target is
  the in-context answer; distractor is the memorized answer. Strong overrides
  (e.g., an explicit “the document states … is Berlin”) are included so the
  in-context win is visually dramatic.
* ``control``: optional weak or scrambled prompts, enabled with
  ``--include-controls``. These are tripwires for metrics that look confident
  on nonsense.

Targets and distractors must be single tokens for the active tokenizer. The
validation report logs every decision, including token IDs and decoded pieces.
Dropped examples are data about the tokenizer, not an inconvenience to hide.
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
import textwrap
from typing import Any, Iterable

import interp_bench as bench

LAB_ID = "L01"
CATEGORIES = ("fact", "ambiguous", "counterfactual", "control")
LOGIT_DIFF_MEANINGFUL_MARGIN = 1.0
EVENT_DEPTH_KEYS = (
    "decision_depth",
    "target_first_top1",
    "target_stable_top1_depth",
    "target_first_beats_distractor",
    "target_first_beats_distractor_raw",
    "target_stable_beats_distractor",
    "target_rank_first_le_5",
    "kl_to_final_first_le_0.5_bits",
    "cosine_to_final_first_ge_0.95",
)

EVENT_PLOT_ORDER = (
    "target_rank_first_le_5",
    "target_first_beats_distractor_raw",
    "target_first_beats_distractor",
    "target_first_top1",
    "target_stable_top1_depth",
    "decision_depth",
    "cosine_to_final_first_ge_0.95",
    "kl_to_final_first_le_0.5_bits",
)

EVENT_DISPLAY = {
    "decision_depth": "final top-1 stable",
    "target_first_top1": "target first top-1",
    "target_stable_top1_depth": "target stable top-1",
    "target_first_beats_distractor": "target > distractor +1",
    "target_first_beats_distractor_raw": "target > distractor",
    "target_stable_beats_distractor": "target stably > distractor",
    "target_rank_first_le_5": "target rank <= 5",
    "kl_to_final_first_le_0.5_bits": "KL to final <= 0.5 bits",
    "cosine_to_final_first_ge_0.95": "cosine to final >= 0.95",
}

EVENT_MARKERS = {
    "target_rank_first_le_5": "o",
    "target_first_beats_distractor_raw": "v",
    "target_first_beats_distractor": "^",
    "target_first_top1": "P",
    "target_stable_top1_depth": "X",
    "decision_depth": "D",
    "cosine_to_final_first_ge_0.95": "s",
    "kl_to_final_first_le_0.5_bits": "*",
}

PHASE_ORDER = ("embedding", "early_blocks", "middle_blocks", "late_blocks", "final")
PHASE_DISPLAY = {
    "embedding": "embed",
    "early_blocks": "early",
    "middle_blocks": "middle",
    "late_blocks": "late",
    "final": "final",
}


# ---------------------------------------------------------------------------
# Prompt families
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PromptExample:
    """One next-token prompt for the raw logit-lens experiment."""

    example_id: str
    category: str
    prompt: str
    target: str | None = None
    distractor: str | None = None
    note: str = ""


FACT_EXAMPLES = (
    PromptExample("fact_capital_france", "fact", "The capital of France is", " Paris", " London"),
    PromptExample("fact_capital_japan", "fact", "The capital of Japan is", " Tokyo", " Kyoto"),
    PromptExample("fact_capital_italy", "fact", "The capital of Italy is", " Rome", " Milan"),
    PromptExample("fact_capital_germany", "fact", "The capital of Germany is", " Berlin", " Munich"),
    PromptExample("fact_eiffel_city", "fact", "The Eiffel Tower is in the city of", " Paris", " Rome"),
    PromptExample("fact_opposite_hot", "fact", "The opposite of hot is", " cold", " warm"),
    PromptExample("fact_opposite_up", "fact", "The opposite of up is", " down", " left"),
    PromptExample("fact_water_h2o", "fact", "Water is made of hydrogen and", " oxygen", " carbon"),
    PromptExample("fact_two_plus_two", "fact", "Two plus two equals", " four", " five"),
    PromptExample("fact_sky_color", "fact", "On a clear day, the color of the sky is", " blue", " green"),
    PromptExample("fact_week_days", "fact", "The day after Monday is", " Tuesday", " Sunday"),
    PromptExample("fact_primary_red", "fact", "One primary color is", " red", " table"),
    # Stronger "answer-shaped" and discourse-bias contrasts to make model objective vs task pop
    PromptExample("fact_capital_france_city", "fact", "The capital of France is the city of", " Paris", " London"),
    PromptExample("fact_eiffel_tower", "fact", "The Eiffel Tower is located in", " Paris", " Rome"),
    PromptExample("fact_python_creator", "fact", "The creator of the Python language is", " Guido", " Rossum"),  # single-token-ish for many tokenizers
    PromptExample("fact_capital_australia", "fact", "The capital of Australia is", " Canberra", " Sydney"),
    PromptExample("fact_capital_canada", "fact", "The capital of Canada is", " Ottawa", " Toronto"),
    PromptExample("fact_moon_planet", "fact", "The largest moon of Jupiter is", " Ganymede", " Europa"),
    PromptExample("fact_heart_organ", "fact", "The organ that pumps blood is the", " heart", " liver"),
    PromptExample("fact_oxygen_gas", "fact", "The gas we breathe to live is called", " oxygen", " nitrogen"),
    # Discourse-framed fact: fluent set-up, but the continuation is still a labeled fact.
    PromptExample("fact_well_known_capital", "fact", "It is well known that the capital of France is", " Paris", " London"),
)

AMBIGUOUS_EXAMPLES = (
    PromptExample("ambig_solve_problem", "ambiguous", "The best way to solve the problem is"),
    PromptExample("ambig_went_to_the", "ambiguous", "Yesterday afternoon I went to the"),
    PromptExample("ambig_important_thing", "ambiguous", "The most important thing in life is"),
    PromptExample("ambig_opened_door", "ambiguous", "She opened the door and saw"),
    PromptExample("ambig_meeting_about", "ambiguous", "The meeting tomorrow will be about"),
    PromptExample("ambig_favorite", "ambiguous", "My favorite thing about this city is the"),
    PromptExample("ambig_after_the_rain", "ambiguous", "After the rain stopped, we decided to"),
    PromptExample("ambig_scientist_found", "ambiguous", "The scientist looked at the data and found"),
    # Discourse-heavy to highlight that low entropy / high confidence often means fluent continuation, not "the fact"
    PromptExample("ambig_the_capital_of", "ambiguous", "The capital of France is a city that is"),
    PromptExample("ambig_the_best_part", "ambiguous", "The best part of the whole experience was"),
    PromptExample("ambig_later_that_day", "ambiguous", "Later that day we realized we had"),
    PromptExample("ambig_in_the_end", "ambiguous", "In the end it all came down to"),
    PromptExample("ambig_she_said_that", "ambiguous", "She said that the only thing that mattered was"),
)

COUNTERFACTUAL_EXAMPLES = (
    PromptExample(
        "cf_capital_france_london",
        "counterfactual",
        "In this story, the capital of France is London. According to the story, the capital of France is",
        " London",
        " Paris",
        note="context answer vs memorized fact",
    ),
    PromptExample(
        "cf_capital_japan_osaka",
        "counterfactual",
        "In this story, the capital of Japan is Osaka. According to the story, the capital of Japan is",
        " Osaka",
        " Tokyo",
    ),
    PromptExample(
        "cf_sky_green",
        "counterfactual",
        "Professor Hale insists that the sky is green. According to Professor Hale, the sky is",
        " green",
        " blue",
    ),
    PromptExample(
        "cf_opposite_hot_wet",
        "counterfactual",
        "In this puzzle, the opposite of hot is wet. In this puzzle, the opposite of hot is",
        " wet",
        " cold",
    ),
    PromptExample(
        "cf_two_plus_two_five",
        "counterfactual",
        "In this game, two plus two equals five. In this game, two plus two equals",
        " five",
        " four",
    ),
    PromptExample(
        "cf_paris_person",
        "counterfactual",
        "In this story, Paris is a person and London is a dog. In this story, Paris is a",
        " person",
        " city",
    ),
    PromptExample(
        "cf_water_sugar",
        "counterfactual",
        "In this fictional chemistry class, water is made of sugar. In that class, water is made of",
        " sugar",
        " oxygen",
    ),
    PromptExample(
        "cf_monday_friday",
        "counterfactual",
        "In this calendar game, the day after Monday is Friday. In this calendar game, the day after Monday is",
        " Friday",
        " Tuesday",
    ),
    # Strong override to make "in-context beats memorized" pop clearly
    PromptExample(
        "cf_france_berlin",
        "counterfactual",
        "The document clearly states that the capital of France is Berlin. The capital of France is therefore",
        " Berlin",
        " Paris",
    ),
    PromptExample(
        "cf_germany_madrid",
        "counterfactual",
        "According to the map in the story, the capital of Germany is Madrid. So the capital of Germany is",
        " Madrid",
        " Berlin",
    ),
    PromptExample(
        "cf_heart_brain",
        "counterfactual",
        "In the medical textbook we are using today, the organ that pumps blood is the brain. The organ that pumps blood is the",
        " brain",
        " heart",
    ),
    PromptExample(
        "cf_oxygen_argon",
        "counterfactual",
        "In this sealed lab experiment the gas we need is argon. The gas the experiment uses is",
        " argon",
        " oxygen",
    ),
)

CONTROL_EXAMPLES = (
    PromptExample("ctrl_word_salad", "control", "Blue cabinet therefore seven because the"),
    PromptExample("ctrl_scrambled_fact", "control", "France capital the is of"),
    PromptExample("ctrl_repeated_marker", "control", "zq zq zq zq the answer is"),
    PromptExample("ctrl_empty_frame", "control", "In the previous sentence, the correct answer was"),
)

ALL_EXAMPLES = FACT_EXAMPLES + AMBIGUOUS_EXAMPLES + COUNTERFACTUAL_EXAMPLES

SMALL_SET_IDS = {
    "fact_capital_france",
    "fact_capital_japan",
    "fact_opposite_hot",
    "fact_two_plus_two",
    "ambig_solve_problem",
    "ambig_went_to_the",
    "ambig_opened_door",
    "cf_capital_france_london",
    "cf_sky_green",
    "cf_two_plus_two_five",
}

MEDIUM_SET_IDS = SMALL_SET_IDS | {
    "fact_capital_italy",
    "fact_water_h2o",
    "fact_sky_color",
    "ambig_meeting_about",
    "ambig_after_the_rain",
    "cf_capital_japan_osaka",
    "cf_opposite_hot_wet",
    "cf_paris_person",
}


# ---------------------------------------------------------------------------
# Prompt loading and validation
# ---------------------------------------------------------------------------


def validate_prompt_schema(examples: list[PromptExample]) -> None:
    """Fail early on malformed or duplicate prompt examples."""
    seen: set[str] = set()
    problems: list[str] = []
    for ex in examples:
        if not ex.example_id:
            problems.append("example with empty example_id")
        if ex.example_id in seen:
            problems.append(f"duplicate example_id: {ex.example_id}")
        seen.add(ex.example_id)
        if ex.category not in CATEGORIES:
            problems.append(f"{ex.example_id}: unknown category {ex.category!r}")
        if not ex.prompt:
            problems.append(f"{ex.example_id}: empty prompt")
        if ex.category in {"fact", "counterfactual"} and not ex.target:
            problems.append(f"{ex.example_id}: {ex.category} example should have a target")
    if problems:
        rendered = "\n".join(f"  - {p}" for p in problems)
        raise ValueError(f"Prompt-set validation failed:\n{rendered}")


def interleave_by_category(examples: list[PromptExample]) -> list[PromptExample]:
    """Round-robin examples so a max_examples cap still covers categories."""
    queues: dict[str, list[PromptExample]] = {cat: [] for cat in CATEGORIES}
    for ex in examples:
        queues.setdefault(ex.category, []).append(ex)
    out: list[PromptExample] = []
    while any(queues.values()):
        # Drain every queue, not just known CATEGORIES, so an unexpected
        # category can never make this loop spin forever.
        for cat in queues:
            if queues[cat]:
                out.append(queues[cat].pop(0))
    return out


def prompt_set_sha256(examples: list[PromptExample]) -> str:
    """Stable digest of the exact prompt objects selected for this run."""
    payload = [dataclasses.asdict(ex) for ex in examples]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def load_custom_prompt_set(path: pathlib.Path) -> list[PromptExample]:
    """Load custom prompts from JSON or CSV with helpful errors.

    JSON is a list of objects. CSV uses the dataclass field names as columns;
    missing optional columns default to empty strings. Supporting CSV matters
    for students because prompt-set audits often begin in a spreadsheet, but
    the resulting prompts are still frozen into the run manifest.
    """
    allowed = {f.name for f in dataclasses.fields(PromptExample)}
    suffix = path.suffix.lower()
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"Could not read prompt set {str(path)!r}: {exc}. "
            "--prompt-set must be one of small | medium | full, or a path to a prompts .json/.csv file."
        ) from exc

    examples: list[PromptExample] = []
    if suffix == ".json":
        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse prompt JSON at {path}: {exc}") from exc
        if not isinstance(raw, list):
            raise ValueError("Custom prompt JSON must be a list of objects.")
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"Prompt item {i} is not an object: {item!r}")
            extra = set(item) - allowed
            if extra:
                raise ValueError(f"Prompt item {i} has unknown keys: {sorted(extra)}")
            examples.append(PromptExample(**item))
    elif suffix == ".csv":
        reader = csv.DictReader(raw_text.splitlines())
        if reader.fieldnames is None:
            raise ValueError(f"Custom prompt CSV at {path} has no header row.")
        extra = set(reader.fieldnames) - allowed
        if extra:
            raise ValueError(f"Prompt CSV has unknown columns: {sorted(extra)}")
        required = {"example_id", "category", "prompt"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Prompt CSV is missing required columns: {sorted(missing)}")
        for i, row in enumerate(reader, start=2):
            clean = {k: (v if v is not None else "") for k, v in row.items() if k in allowed}
            try:
                examples.append(PromptExample(**clean))
            except TypeError as exc:
                raise ValueError(f"Prompt CSV row {i} could not be parsed as a PromptExample: {exc}") from exc
    else:
        raise ValueError(
            f"Unsupported prompt-set file extension {suffix!r}. Use .json or .csv, or one of small | medium | full."
        )
    return examples


def build_prompt_set(args: Any) -> list[PromptExample]:
    """Resolve --prompt-set into a concrete, interleaved prompt list."""
    if args.prompt_set == "full":
        examples = list(ALL_EXAMPLES)
    elif args.prompt_set == "medium":
        examples = [ex for ex in ALL_EXAMPLES if ex.example_id in MEDIUM_SET_IDS]
    elif args.prompt_set == "small":
        examples = [ex for ex in ALL_EXAMPLES if ex.example_id in SMALL_SET_IDS]
    else:
        examples = load_custom_prompt_set(pathlib.Path(args.prompt_set))

    if getattr(args, "include_controls", False):
        existing = {ex.example_id for ex in examples}
        examples.extend(ex for ex in CONTROL_EXAMPLES if ex.example_id not in existing)

    validate_prompt_schema(examples)
    examples = interleave_by_category(examples)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    return examples


def token_ids(tokenizer: Any, text: str | None) -> list[int]:
    if text is None:
        return []
    return list(tokenizer.encode(text, add_special_tokens=False))


def single_token_id(tokenizer: Any, text: str | None) -> int | None:
    ids = token_ids(tokenizer, text)
    return ids[0] if len(ids) == 1 else None


def decoded_pieces(tokenizer: Any, ids: Iterable[int]) -> str:
    return " ".join(bench.visible_token(tokenizer.decode([int(i)])) for i in ids)


_NOTE_TAG_RE = re.compile(r"(?:^|[;,|\s])([A-Za-z_][A-Za-z0-9_-]*)=([^;,|\s]+)")


def note_tag(note: str, key: str) -> str | None:
    """Extract ``key=value`` tags from prompt notes.

    Custom prompt sets can carry lightweight metadata in the note column, e.g.
    ``relation=capital`` or ``near_tie=1``. The lab keeps the parser tiny and
    forgiving so spreadsheet-authored CSVs work without a new schema.
    """
    for match in _NOTE_TAG_RE.finditer(note or ""):
        if match.group(1).lower() == key.lower():
            return match.group(2)
    return None


def infer_relation_family(ex: PromptExample) -> str:
    """Best-effort family label used for cross-prompt aggregate plots.

    If a custom prompt provides ``relation=...`` or ``family=...`` in ``note``,
    that wins. Otherwise the curated Lab 1 examples get a readable fallback
    based on their ids and text. This lets the same plotting code scale from
    the small smoke set to the 105-row relation diversity CSV mentioned in the
    handout, without turning Lab 1 into a full data model.
    """
    tagged = (
        note_tag(ex.note, "relation_family")
        or note_tag(ex.note, "relation")
        or note_tag(ex.note, "family")
        or note_tag(ex.note, "task")
    )
    if tagged:
        return tagged
    text = f"{ex.example_id} {ex.prompt}".lower()
    if ex.category == "ambiguous":
        if "capital of france" in text:
            return "ambiguous_capital_frame"
        return "open_ended"
    if ex.category == "control":
        return "weak_control"
    if any(token in text for token in ("capital", "eiffel", "paris", "london", "japan", "germany")):
        return "capital_or_place" if ex.category == "fact" else "cf_place"
    if "opposite" in text:
        return "antonym" if ex.category == "fact" else "cf_antonym"
    if "two_plus_two" in text or "two plus two" in text:
        return "arithmetic" if ex.category == "fact" else "cf_arithmetic"
    if "sky" in text or "color" in text:
        return "color" if ex.category == "fact" else "cf_color"
    if any(token in text for token in ("water", "oxygen", "hydrogen", "gas")):
        return "science" if ex.category == "fact" else "cf_science"
    if any(token in text for token in ("heart", "organ", "blood")):
        return "body" if ex.category == "fact" else "cf_body"
    if "monday" in text or "day after" in text:
        return "sequence" if ex.category == "fact" else "cf_sequence"
    return ex.category


def validate_examples(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    examples: list[PromptExample],
) -> tuple[list[tuple[PromptExample, int | None, int | None]], int]:
    """Validate single-token labels and write prompt diagnostics."""
    kept: list[tuple[PromptExample, int | None, int | None]] = []
    report_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    dropped = 0

    tok = bundle.tokenizer
    for ex in examples:
        prompt_ids = token_ids(tok, ex.prompt)
        target_ids = token_ids(tok, ex.target)
        distractor_ids = token_ids(tok, ex.distractor)
        target_id = target_ids[0] if len(target_ids) == 1 else None
        distractor_id = distractor_ids[0] if len(distractor_ids) == 1 else None
        has_target = ex.target not in (None, "")
        has_distractor = ex.distractor not in (None, "")
        target_ok = not has_target or target_id is not None
        distractor_ok = not has_distractor or distractor_id is not None
        same_label_id = target_id is not None and distractor_id is not None and target_id == distractor_id

        reasons: list[str] = []
        if not prompt_ids:
            reasons.append("prompt tokenized to zero tokens")
        if not target_ok:
            reasons.append(f"target tokenized into {len(target_ids)} tokens")
        if not distractor_ok:
            reasons.append(f"distractor tokenized into {len(distractor_ids)} tokens")
        if same_label_id:
            reasons.append("target and distractor are the same token id")

        status = "kept" if not reasons else "dropped"
        reason = "; ".join(reasons)
        if status == "dropped":
            dropped += 1
        else:
            kept.append((ex, target_id, distractor_id))
            manifest_rows.append(
                {
                    "example_id": ex.example_id,
                    "category": ex.category,
                    "relation_family": infer_relation_family(ex),
                    "n_prompt_tokens": len(prompt_ids),
                    "prompt": ex.prompt,
                    "prompt_sha256": hashlib.sha256(ex.prompt.encode("utf-8")).hexdigest(),
                    "prompt_token_ids": " ".join(map(str, prompt_ids)),
                    "prompt_decoded_pieces": decoded_pieces(tok, prompt_ids),
                    "target": bench.visible_token(ex.target) if ex.target else "",
                    "target_id": target_id if target_id is not None else "",
                    "distractor": bench.visible_token(ex.distractor) if ex.distractor else "",
                    "distractor_id": distractor_id if distractor_id is not None else "",
                    "note": ex.note,
                }
            )

        report_rows.append(
            {
                "example_id": ex.example_id,
                "category": ex.category,
                "relation_family": infer_relation_family(ex),
                "prompt": ex.prompt,
                "n_prompt_tokens": len(prompt_ids),
                "prompt_token_ids": " ".join(map(str, prompt_ids)),
                "prompt_decoded_pieces": decoded_pieces(tok, prompt_ids),
                "target": bench.visible_token(ex.target) if ex.target else "",
                "target_token_ids": " ".join(map(str, target_ids)),
                "target_decoded_pieces": decoded_pieces(tok, target_ids),
                "target_n_tokens": len(target_ids) if ex.target else "",
                "distractor": bench.visible_token(ex.distractor) if ex.distractor else "",
                "distractor_token_ids": " ".join(map(str, distractor_ids)),
                "distractor_decoded_pieces": decoded_pieces(tok, distractor_ids),
                "distractor_n_tokens": len(distractor_ids) if ex.distractor else "",
                "target_and_distractor_same_id": same_label_id,
                "status": status,
                "reason": reason,
            }
        )

    tok_path = ctx.path("diagnostics", "tokenization_report.csv")
    bench.write_csv(tok_path, report_rows)
    ctx.register_artifact(tok_path, "diagnostic", "Single-token validation for prompt labels, with tokenized prompts.")

    manifest_path = ctx.path("tables", "prompt_manifest.csv")
    bench.write_csv_with_context(ctx, manifest_path, manifest_rows)
    ctx.register_artifact(manifest_path, "table", "Prompt set that survived tokenization validation.")

    selected_counts = {cat: sum(1 for ex in examples if ex.category == cat) for cat in CATEGORIES}
    kept_counts = {cat: sum(1 for ex, _, _ in kept if ex.category == cat) for cat in CATEGORIES}
    prompt_manifest = {
        "lab": LAB_ID,
        "prompt_set_arg": ctx.args.prompt_set,
        "include_controls": bool(getattr(ctx.args, "include_controls", False)),
        "max_examples_after_tier_defaults": ctx.args.max_examples,
        "n_selected_before_tokenization": len(examples),
        "n_kept_after_tokenization": len(kept),
        "n_dropped_tokenization": dropped,
        "selected_category_counts": selected_counts,
        "kept_category_counts": kept_counts,
        "selected_prompt_set_sha256": prompt_set_sha256(examples),
        "kept_prompt_set_sha256": prompt_set_sha256([ex for ex, _, _ in kept]),
        "custom_prompt_formats": ["json", "csv"],
        "single_token_rule": "Targets and distractors, when present, must each encode to exactly one token and must not share the same token id.",
    }
    manifest_json = ctx.path("diagnostics", "prompt_set_manifest.json")
    bench.write_json(manifest_json, prompt_manifest)
    ctx.register_artifact(manifest_json, "diagnostic", "Exact selected prompt-set counts and stable hashes.")

    if dropped:
        print(f"[lab1] dropped {dropped} example(s) at validation; see diagnostics/tokenization_report.csv")
    return kept, dropped


# ---------------------------------------------------------------------------
# Event metrics
# ---------------------------------------------------------------------------


def first_depth(predicate_values: Iterable[bool]) -> int | None:
    """First index whose predicate is true."""
    for i, ok in enumerate(predicate_values):
        if ok:
            return i
    return None


def stable_depth(predicate_values: list[bool]) -> int | None:
    """First index after which the predicate remains true through final depth."""
    if not predicate_values or not predicate_values[-1]:
        return None
    for i in range(len(predicate_values)):
        if all(predicate_values[i:]):
            return i
    return None


def first_value_le(values: list[float], threshold: float) -> int | None:
    return first_depth(v <= threshold for v in values)


def first_value_ge(values: list[float], threshold: float) -> int | None:
    return first_depth(v >= threshold for v in values)


def stable_top1_depth(traj: bench.LensTrajectory, token_id: int) -> int | None:
    return stable_depth([top1 == token_id for top1 in traj.top1_ids])


def decision_depth(traj: bench.LensTrajectory) -> int:
    """Smallest depth where the final top-1 token remains top-1 thereafter."""
    depth = stable_top1_depth(traj, traj.top1_ids[-1])
    return traj.n_depths - 1 if depth is None else depth


def target_first_top1(traj: bench.LensTrajectory, target_id: int) -> int | None:
    return first_depth(top1 == target_id for top1 in traj.top1_ids)


def top1_flip_count(traj: bench.LensTrajectory) -> int:
    return sum(1 for a, b in zip(traj.top1_ids, traj.top1_ids[1:]) if a != b)


def stable_logit_diff_positive(traj: bench.LensTrajectory) -> int | None:
    if traj.logit_target is None or traj.logit_distractor is None:
        return None
    diffs = [t - d for t, d in zip(traj.logit_target, traj.logit_distractor)]
    return stable_depth([d > 0 for d in diffs])


def first_logit_diff_positive_raw(traj: bench.LensTrajectory) -> int | None:
    if traj.logit_target is None or traj.logit_distractor is None:
        return None
    return first_depth((t - d) > 0 for t, d in zip(traj.logit_target, traj.logit_distractor))


def first_meaningful_logit_diff_positive(traj: bench.LensTrajectory) -> int | None:
    """First target>distractor crossing that is large enough to cite.

    The raw first positive crossing is deliberately too twitchy for claims:
    two rank-80k tokens can swap order in the junk-readout regime. This metric
    requires a one-logit margin at the crossing and requires the target to keep
    the lead through the final depth.
    """
    if traj.logit_target is None or traj.logit_distractor is None:
        return None
    diffs = [t - d for t, d in zip(traj.logit_target, traj.logit_distractor)]
    return first_depth(
        diff > LOGIT_DIFF_MEANINGFUL_MARGIN and all(later > 0 for later in diffs[i:])
        for i, diff in enumerate(diffs)
    )


def first_rank_le(ranks: list[int] | None, threshold: int) -> int | None:
    if ranks is None:
        return None
    return first_depth(r <= threshold for r in ranks)


def list_mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def none_if_nan(x: float | None) -> float | None:
    if x is None:
        return None
    return x if x == x else None


def round_or_blank(value: Any, digits: int = 4) -> Any:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    try:
        return round(float(value), digits)
    except Exception:
        return value


def trajectory_event_row(
    ex: PromptExample,
    capture: bench.ForwardCapture,
    traj: bench.LensTrajectory,
    target_id: int | None,
    distractor_id: int | None,
) -> dict[str, Any]:
    """One per-example event row for tables/trajectory_events.csv."""
    row: dict[str, Any] = {
        "example_id": ex.example_id,
        "category": ex.category,
        "relation_family": infer_relation_family(ex),
        "prompt": ex.prompt,
        "note": ex.note,
        "n_prompt_tokens": len(capture.input_ids),
        "n_depths": traj.n_depths,
        "decision_depth": decision_depth(traj),
        "top1_flip_count": top1_flip_count(traj),
        "final_top1": traj.top1_texts[-1],
        "final_top1_id": traj.top1_ids[-1],
        "final_top1_prob": round(traj.top1_probs[-1], 6),
        "final_top1_margin": round(traj.top1_margin[-1], 6),
        "final_entropy_bits": round(traj.entropy_bits[-1], 4),
        "mean_entropy_bits": round(list_mean(traj.entropy_bits), 4),
        "kl_to_final_first_le_0.5_bits": first_value_le(traj.kl_to_final_bits, 0.5),
        "cosine_to_final_first_ge_0.95": first_value_ge(traj.cosine_to_final, 0.95),
    }
    if target_id is not None:
        row.update(
            {
                "target": bench.visible_token(ex.target or ""),
                "target_id": target_id,
                "target_first_top1": target_first_top1(traj, target_id),
                "target_stable_top1_depth": stable_top1_depth(traj, target_id),
                "target_rank_first_le_5": first_rank_le(traj.target_rank, 5),
                "final_target_rank": traj.target_rank[-1] if traj.target_rank is not None else "",
                "final_p_target": round(traj.p_target[-1], 6) if traj.p_target is not None else "",
                "final_top1_is_target": traj.top1_ids[-1] == target_id,
            }
        )
    if distractor_id is not None:
        row.update(
            {
                "distractor": bench.visible_token(ex.distractor or ""),
                "distractor_id": distractor_id,
                "final_distractor_rank": traj.distractor_rank[-1] if traj.distractor_rank is not None else "",
                "final_p_distractor": round(traj.p_distractor[-1], 6) if traj.p_distractor is not None else "",
            }
        )
    if traj.logit_target is not None and traj.logit_distractor is not None:
        final_diff = traj.logit_target[-1] - traj.logit_distractor[-1]
        row.update(
            {
                "target_first_beats_distractor_raw": first_logit_diff_positive_raw(traj),
                "target_first_beats_distractor": first_meaningful_logit_diff_positive(traj),
                "target_stable_beats_distractor": stable_logit_diff_positive(traj),
                "final_logit_diff": round(final_diff, 6),
                "final_target_beats_distractor": final_diff > 0,
                "final_target_beats_distractor_by_margin": final_diff > LOGIT_DIFF_MEANINGFUL_MARGIN,
                "mean_logit_diff": round(
                    list_mean([t - d for t, d in zip(traj.logit_target, traj.logit_distractor)]),
                    6,
                ),
            }
        )

    # A compact label for the final readout. This keeps three notions separate:
    # target top-1, target merely beating the matched distractor, and confidence
    # on an unlabeled ambiguous continuation.
    #
    # This is the heart of the “readout is an instrument” lesson. On many factual
    # prompts the model’s actual top-1 is a discourse word (“known”, “the city of”)
    # even while the labeled target has a solid rank and beats its distractor.
    # final_readout_audit.csv and the per-example state cards make this visible
    # immediately. Students must not equate “target rank improved” or “target beats
    # distractor” with “the model knows the answer at this depth.”
    cos_event = row.get("cosine_to_final_first_ge_0.95")
    kl_event = row.get("kl_to_final_first_le_0.5_bits")
    if cos_event is not None and kl_event is not None:
        row["convergence_lag_depths"] = int(kl_event) - int(cos_event)
        row["convergence_lag_frac"] = round((int(kl_event) - int(cos_event)) / max(1, traj.n_depths - 1), 4)

    if target_id is None:
        row["final_outcome"] = "unlabeled"
    elif bool(row.get("final_top1_is_target")):
        row["final_outcome"] = "target_top1"
    elif "final_logit_diff" in row and float(row["final_logit_diff"]) > LOGIT_DIFF_MEANINGFUL_MARGIN:
        row["final_outcome"] = "target_beats_distractor_not_top1"
    elif "final_logit_diff" in row and float(row["final_logit_diff"]) > 0:
        row["final_outcome"] = "target_slightly_beats_distractor"
    elif distractor_id is not None:
        row["final_outcome"] = "distractor_or_other_beats_target"
    else:
        row["final_outcome"] = "target_not_top1"
    return row


def add_event_depth_fractions(event_rows: list[dict[str, Any]], n_layers: int) -> None:
    """Add *_frac columns so event depths compare across model sizes."""
    denom = max(1, n_layers)
    for row in event_rows:
        for key in EVENT_DEPTH_KEYS:
            value = row.get(key)
            if value in (None, ""):
                row[f"{key}_frac"] = ""
            else:
                row[f"{key}_frac"] = round(float(value) / denom, 4)


def final_readout_audit_rows(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A compact per-example final-readout table for correctness-vs-confidence audits."""
    keys = [
        "example_id", "category", "relation_family", "final_outcome", "final_top1", "final_top1_prob",
        "final_top1_margin", "final_entropy_bits", "target", "final_p_target",
        "final_target_rank", "final_top1_is_target", "distractor", "final_p_distractor",
        "final_distractor_rank", "final_logit_diff", "final_target_beats_distractor_by_margin",
        "decision_depth", "top1_flip_count",
    ]
    return [{key: row.get(key, "") for key in keys} for row in event_rows]


def top1_transition_rows(
    ex: PromptExample,
    traj: bench.LensTrajectory,
    target_id: int | None,
    distractor_id: int | None,
) -> list[dict[str, Any]]:
    """Compress each top-1 biography into stable token segments."""
    rows: list[dict[str, Any]] = []
    start = 0
    final_id = traj.top1_ids[-1]
    while start < traj.n_depths:
        token_id = traj.top1_ids[start]
        end = start
        while end + 1 < traj.n_depths and traj.top1_ids[end + 1] == token_id:
            end += 1
        rows.append(
            {
                "example_id": ex.example_id,
                "category": ex.category,
                "relation_family": infer_relation_family(ex),
                "start_depth": start,
                "end_depth": end,
                "duration_depths": end - start + 1,
                "token_id": token_id,
                "token": bench.visible_token(traj.top1_texts[start]),
                "is_final_top1_token": token_id == final_id,
                "is_target": target_id is not None and token_id == target_id,
                "is_distractor": distractor_id is not None and token_id == distractor_id,
                "start_prob": round(traj.top1_probs[start], 6),
                "end_prob": round(traj.top1_probs[end], 6),
                "max_prob_in_segment": round(max(traj.top1_probs[start : end + 1]), 6),
            }
        )
        start = end + 1
    return rows


def phase_for_depth(depth: int, n_layers: int) -> str:
    if depth == 0:
        return "embedding"
    if depth == n_layers:
        return "final"
    frac = depth / max(1, n_layers)
    if frac <= 1 / 3:
        return "early_blocks"
    if frac <= 2 / 3:
        return "middle_blocks"
    return "late_blocks"


def readout_phase_rows(ex: PromptExample, traj: bench.LensTrajectory, n_layers: int) -> list[dict[str, Any]]:
    """Summarize trajectories in coarse depth bands (embedding / early / middle / late / final)
    for quick cross-model reading. The phase labels are defined in phase_for_depth and
    are intentionally coarse so students can compare stabilization timing across models
    of different depths without getting lost in layer numbers.
    """
    phases = ["embedding", "early_blocks", "middle_blocks", "late_blocks", "final"]
    out: list[dict[str, Any]] = []
    for phase in phases:
        depths = [d for d in range(traj.n_depths) if phase_for_depth(d, n_layers) == phase]
        if not depths:
            continue
        tokens = [traj.top1_texts[d] for d in depths]
        dominant = max(set(tokens), key=tokens.count)
        row: dict[str, Any] = {
            "example_id": ex.example_id,
            "category": ex.category,
            "relation_family": infer_relation_family(ex),
            "phase": phase,
            "start_depth": depths[0],
            "end_depth": depths[-1],
            "n_depths": len(depths),
            "dominant_top1_token": bench.visible_token(dominant),
            "n_unique_top1_tokens": len(set(tokens)),
            "mean_entropy_bits": round(statistics.fmean(traj.entropy_bits[d] for d in depths), 4),
            "mean_kl_to_final_bits": round(statistics.fmean(traj.kl_to_final_bits[d] for d in depths), 4),
            "mean_top1_margin": round(statistics.fmean(traj.top1_margin[d] for d in depths), 6),
            "mean_cosine_to_final": round(statistics.fmean(traj.cosine_to_final[d] for d in depths), 5),
        }
        if traj.p_target is not None:
            row["max_p_target"] = round(max(traj.p_target[d] for d in depths), 6)
        if traj.target_rank is not None:
            row["best_target_rank"] = min(traj.target_rank[d] for d in depths)
        if traj.logit_target is not None and traj.logit_distractor is not None:
            row["mean_logit_diff"] = round(
                statistics.fmean(traj.logit_target[d] - traj.logit_distractor[d] for d in depths),
                5,
            )
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def category_color(category: str) -> str:
    return getattr(bench, "CATEGORY_COLORS", {}).get(category, "#333333")


def category_label(category: str) -> str:
    return category.replace("_", " ")


def event_label(event: str, *, multiline: bool = False) -> str:
    label = EVENT_DISPLAY.get(event, event.replace("_", " "))
    return label.replace(" ", "\n") if multiline else label


def short_example_label(example_id: str, max_len: int = 34) -> str:
    label = re.sub(r"^(fact|cf|ambig|ctrl)_", "", example_id)
    label = label.replace("_", " ")
    return textwrap.shorten(label, width=max_len, placeholder="…")


def number_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolated percentile for tiny prompt sets."""
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - k) + vals[hi] * (k - lo)


def summarize_curves(rows: list[dict[str, Any]], metric: str) -> dict[str, list[float]] | None:
    """Return median/IQR/mean curves with consistent depth truncation."""
    rows = [r for r in rows if metric in r and r[metric]]
    if not rows:
        return None
    depth_count = min(len(r[metric]) for r in rows)
    if depth_count <= 0:
        return None
    out = {"x": list(range(depth_count)), "median": [], "q25": [], "q75": [], "mean": []}
    for depth in range(depth_count):
        vals: list[float] = []
        for r in rows:
            v = number_or_none(r[metric][depth])
            if v is not None:
                vals.append(v)
        if not vals:
            out["median"].append(float("nan"))
            out["q25"].append(float("nan"))
            out["q75"].append(float("nan"))
            out["mean"].append(float("nan"))
        else:
            out["median"].append(float(percentile(vals, 0.50)))
            out["q25"].append(float(percentile(vals, 0.25)))
            out["q75"].append(float(percentile(vals, 0.75)))
            out["mean"].append(statistics.fmean(vals))
    return out


def apply_depth_guides(ax: Any, n_layers: int, *, final_label: bool = True) -> None:
    """Add light depth guides without stealing the plot's attention."""
    if n_layers <= 0:
        return
    for frac, label in ((1 / 3, "early/mid"), (2 / 3, "mid/late")):
        x = n_layers * frac
        ax.axvline(x, color="#888888", linestyle=":", linewidth=0.7, alpha=0.25)
        ymin, ymax = ax.get_ylim()
        if math.isfinite(ymin) and math.isfinite(ymax) and ymax != ymin:
            ax.text(x, ymax, label, ha="center", va="top", fontsize=6.5, color="#777777", alpha=0.7)
    ax.axvline(n_layers, color="#444444", linestyle=":", linewidth=1.0, alpha=0.55)
    if final_label:
        ax.text(n_layers, 0.98, "final", transform=ax.get_xaxis_transform(), rotation=90,
                va="top", ha="right", fontsize=7, color="#444444", alpha=0.75)


def add_metric_reference(ax: Any, metric: str) -> None:
    """Reference thresholds used by the event definitions."""
    if metric == "kl_to_final_bits":
        ax.axhline(0.5, color="#555555", linestyle="--", linewidth=0.9, alpha=0.55)
        ax.text(0.01, 0.5, "KL ≤ 0.5", transform=ax.get_yaxis_transform(), va="bottom", fontsize=7, color="#555555")
    elif metric == "cosine_to_final":
        ax.axhline(0.95, color="#555555", linestyle="--", linewidth=0.9, alpha=0.55)
        ax.text(0.01, 0.95, "cos ≥ 0.95", transform=ax.get_yaxis_transform(), va="bottom", fontsize=7, color="#555555")
    elif metric == "logit_diff":
        ax.axhline(0.0, color="#333333", linewidth=0.9, alpha=0.55)
        ax.axhline(LOGIT_DIFF_MEANINGFUL_MARGIN, color="#555555", linestyle="--", linewidth=0.9, alpha=0.55)
        ax.text(0.01, LOGIT_DIFF_MEANINGFUL_MARGIN, "+1 logit", transform=ax.get_yaxis_transform(), va="bottom", fontsize=7, color="#555555")
    elif metric == "target_rank":
        ax.axhline(5, color="#555555", linestyle="--", linewidth=0.9, alpha=0.55)
        ax.text(0.01, 5, "rank ≤ 5", transform=ax.get_yaxis_transform(), va="bottom", fontsize=7, color="#555555")


def draw_curve_summary(
    ax: Any,
    per_example: list[dict[str, Any]],
    metric: str,
    *,
    categories: tuple[str, ...] = CATEGORIES,
    show_examples: bool = True,
    label_prefix: str = "",
) -> tuple[bool, int]:
    """Draw thin examples, an IQR band, and a bold median curve."""
    plotted = False
    max_depth_count = 0
    for category in categories:
        rows = [r for r in per_example if r.get("category") == category and metric in r and r[metric]]
        if not rows:
            continue
        summary = summarize_curves(rows, metric)
        if summary is None:
            continue
        plotted = True
        max_depth_count = max(max_depth_count, len(summary["x"]))
        color = category_color(category)
        if show_examples:
            for r in rows:
                ax.plot(range(len(r[metric])), r[metric], color=color, alpha=0.13, linewidth=0.65)
        ax.fill_between(summary["x"], summary["q25"], summary["q75"], color=color, alpha=0.12, linewidth=0)
        ax.plot(
            summary["x"],
            summary["median"],
            color=color,
            linewidth=2.8,
            label=f"{label_prefix}{category_label(category)} median (n={len(rows)})",
        )
        # A faint mean line helps students notice skewed prompt families without
        # making the figure visually busy.
        ax.plot(summary["x"], summary["mean"], color=color, linewidth=1.0, alpha=0.45, linestyle="--")
    return plotted, max_depth_count


def plot_metric_by_depth(
    ctx: bench.RunContext,
    per_example: list[dict[str, Any]],
    metric: str,
    *,
    name: str,
    title: str,
    ylabel: str,
    categories: tuple[str, ...] = CATEGORIES,
    logy: bool = False,
    invert_y: bool = False,
) -> None:
    """Per-example + IQR + median curves for one depth metric."""
    fig, ax = bench.new_figure(figsize=(9.8, 5.6))
    plotted, max_depth_count = draw_curve_summary(ax, per_example, metric, categories=categories, show_examples=True)
    if not plotted:
        bench.close_figure(fig)
        return
    add_metric_reference(ax, metric)
    if logy:
        ax.set_yscale("log")
    if invert_y:
        ax.invert_yaxis()
    apply_depth_guides(ax, max_depth_count - 1)
    bench.style_ax(
        ax,
        title=f"{title}  •  thin=examples, band=IQR, bold=median",
        xlabel="depth (0 = embeddings, k = after k blocks)",
        ylabel=ylabel,
        legend=True,
    )
    bench.save_figure(ctx, fig, name, title)


def plot_readout_dashboard(
    ctx: bench.RunContext,
    per_example: list[dict[str, Any]],
    *,
    categories: tuple[str, ...] = CATEGORIES,
) -> None:
    """One dashboard for the four unlabeled curves students compare most."""
    if not per_example:
        return
    import matplotlib.pyplot as plt

    specs = [
        ("entropy_bits", "sharpness: entropy", "bits", False),
        ("kl_to_final_bits", "decoded convergence: KL(final || depth)", "bits", True),
        ("top1_margin", "commitment: top-1 minus top-2", "probability margin", False),
        ("cosine_to_final", "geometry: cosine to final residual", "cosine", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.8))
    plotted_any = False
    max_depth = 0
    for ax, (metric, title, ylabel, logy) in zip(axes.flat, specs):
        ax.grid(True, alpha=0.25)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        plotted, depth_count = draw_curve_summary(ax, per_example, metric, categories=categories, show_examples=False)
        plotted_any = plotted_any or plotted
        max_depth = max(max_depth, depth_count)
        add_metric_reference(ax, metric)
        if logy:
            ax.set_yscale("log")
        apply_depth_guides(ax, depth_count - 1, final_label=False)
        ax.set_title(title)
        ax.set_xlabel("depth")
        ax.set_ylabel(ylabel)
    if not plotted_any:
        bench.close_figure(fig)
        return
    axes[0, 0].legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle(
        "Raw logit-lens dashboard: sharpness, decoded convergence, commitment, and geometry move on different clocks",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.012,
        "Solid lines are category medians; shaded bands are interquartile ranges. Divergent clocks are the readout-is-an-instrument lesson.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    bench.save_figure(
        ctx,
        fig,
        "readout_dashboard.png",
        "Category-median entropy, KL-to-final, top-1 margin, and residual cosine curves with IQR bands.",
    )


def plot_event_depths(
    ctx: bench.RunContext,
    event_rows: list[dict[str, Any]],
    n_layers: int,
) -> None:
    """Scatter event depths by category for quick outlier spotting."""
    fig, ax = bench.new_figure(figsize=(10.4, 5.8))
    events = [
        "decision_depth",
        "target_first_top1",
        "target_first_beats_distractor",
        "target_first_beats_distractor_raw",
        "target_rank_first_le_5",
        "kl_to_final_first_le_0.5_bits",
        "cosine_to_final_first_ge_0.95",
    ]
    x_positions = {event: i for i, event in enumerate(events)}
    jitter_offsets = {cat: (i - (len(CATEGORIES) - 1) / 2) * 0.065 for i, cat in enumerate(CATEGORIES)}
    any_points = False
    for row in event_rows:
        cat = row["category"]
        for event in events:
            value = number_or_none(row.get(event))
            if value is None:
                continue
            any_points = True
            ax.scatter(
                x_positions[event] + jitter_offsets.get(cat, 0.0),
                value,
                s=30,
                alpha=0.78,
                color=category_color(cat),
                edgecolors="white",
                linewidths=0.35,
                label=cat,
            )
    if not any_points:
        bench.close_figure(fig)
        return
    for y in (n_layers / 3, 2 * n_layers / 3, n_layers):
        ax.axhline(y, color="#888888", linestyle=":" if y < n_layers else "--", linewidth=0.8, alpha=0.35)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, frameon=False, loc="best")
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels([event_label(e) for e in events], rotation=28, ha="right")
    ax.set_ylim(-0.5, n_layers + 0.8)
    bench.style_ax(ax, title="Event depths by example", xlabel="event", ylabel="depth")
    bench.save_figure(ctx, fig, "event_depths.png", "Per-example event depths by metric and category.")


def plot_event_heatmap(
    ctx: bench.RunContext,
    event_rows: list[dict[str, Any]],
    n_layers: int,
) -> None:
    """Example x event grid, with missing events visible instead of silent."""
    if not event_rows:
        return
    import matplotlib.pyplot as plt

    events = list(EVENT_PLOT_ORDER)
    rows = sorted(
        event_rows,
        key=lambda r: (
            CATEGORIES.index(r["category"]) if r["category"] in CATEGORIES else len(CATEGORIES),
            str(r.get("relation_family", "")),
            number_or_none(r.get("decision_depth")) if number_or_none(r.get("decision_depth")) is not None else float(n_layers + 1),
            r["example_id"],
        ),
    )
    data = [[float("nan") for _ in events] for _ in rows]
    for i, row in enumerate(rows):
        for j, event in enumerate(events):
            value = number_or_none(row.get(event))
            if value is not None:
                data[i][j] = value

    fig_height = max(6.2, min(15.0, 0.30 * len(rows) + 2.2))
    fig, ax = bench.new_figure(figsize=(12.2, fig_height))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d8d8d8")
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=n_layers)
    ax.set_xticks(range(len(events)))
    ax.set_xticklabels([event_label(e, multiline=True) for e in events], rotation=0, ha="center", fontsize=7)
    ax.set_yticks(range(len(rows)))
    ylabels = [f"{r['category'][:2]}:{short_example_label(r['example_id'], 26)}" for r in rows]
    ax.set_yticklabels(ylabels, fontsize=6.5)
    ax.set_title("Event-depth heatmap: gray means the event never occurred")
    ax.set_xlabel("event metric")
    ax.set_ylabel("example, grouped by category and relation family")
    ax.set_xticks([x - 0.5 for x in range(1, len(events))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(rows))], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.55, alpha=0.55)
    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.01)
    cbar.set_label("depth (earlier = more early commitment under raw lens)")

    previous = rows[0]["category"]
    cat_start = 0
    for i, row in enumerate(rows[1:] + [{"category": "__END__"}], start=1):
        if row["category"] != previous:
            ax.axhline(i - 0.5, color="white", linewidth=2.3)
            mid = (cat_start + i - 1) / 2.0
            ax.text(-0.65, mid, previous, va="center", ha="right", fontsize=7, color="#444444")
            cat_start = i
            previous = row["category"]
    fig.tight_layout()
    bench.save_figure(
        ctx,
        fig,
        "event_depth_heatmap.png",
        "Example-by-event depth grid; gray cells make non-occurring events explicit.",
    )


def plot_event_timeline(
    ctx: bench.RunContext,
    event_rows: list[dict[str, Any]],
    n_layers: int,
) -> None:
    """Timeline view: which event happens when, per example."""
    if not event_rows:
        return
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    events = [
        "target_rank_first_le_5",
        "target_first_beats_distractor",
        "target_first_top1",
        "decision_depth",
        "cosine_to_final_first_ge_0.95",
        "kl_to_final_first_le_0.5_bits",
    ]
    rows = sorted(
        event_rows,
        key=lambda r: (
            CATEGORIES.index(r["category"]) if r["category"] in CATEGORIES else len(CATEGORIES),
            str(r.get("relation_family", "")),
            number_or_none(r.get("decision_depth")) if number_or_none(r.get("decision_depth")) is not None else float(n_layers + 1),
            r["example_id"],
        ),
    )
    fig_height = max(5.5, min(16.0, 0.28 * len(rows) + 2.2))
    fig, ax = bench.new_figure(figsize=(12.4, fig_height))
    for y, row in enumerate(rows):
        cat = row["category"]
        color = category_color(cat)
        ax.hlines(y, 0, n_layers, color=color, alpha=0.12, linewidth=3.8)
        for event in events:
            value = number_or_none(row.get(event))
            if value is None:
                continue
            ax.scatter(value, y, marker=EVENT_MARKERS.get(event, "o"), s=50, color=color,
                       edgecolors="white", linewidths=0.45, zorder=3)
        # A tiny final-outcome tick at the right helps connect timing to correctness.
        outcome = str(row.get("final_outcome", ""))
        if outcome:
            ax.text(n_layers + 0.35, y, outcome.replace("_", " "), va="center", fontsize=6.4, color="#555555")
    ax.axvline(n_layers, color="#444444", linestyle=":", linewidth=1.0, alpha=0.6)
    for frac in (1 / 3, 2 / 3):
        ax.axvline(n_layers * frac, color="#888888", linestyle=":", linewidth=0.8, alpha=0.25)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['category'][:2]}:{short_example_label(r['example_id'], 30)}" for r in rows], fontsize=6.5)
    ax.set_xlim(-0.5, n_layers + 4.0)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.invert_yaxis()
    ax.set_xlabel("depth")
    ax.set_title("Event timeline: the same example can rank, beat, decide, and converge at different depths")
    event_handles = [
        Line2D([0], [0], marker=EVENT_MARKERS.get(e, "o"), color="none", markerfacecolor="#555555",
               markeredgecolor="white", markeredgewidth=0.4, markersize=7, label=event_label(e))
        for e in events
    ]
    cat_handles = [Line2D([0], [0], color=category_color(c), linewidth=4, label=c) for c in CATEGORIES if any(r["category"] == c for r in rows)]
    leg1 = ax.legend(handles=event_handles, title="event", fontsize=7, title_fontsize=8, frameon=False,
                     loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(leg1)
    ax.legend(handles=cat_handles, title="category", fontsize=7, title_fontsize=8, frameon=False,
              loc="lower left", bbox_to_anchor=(1.01, 0.0))
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    bench.save_figure(
        ctx,
        fig,
        "event_timeline.png",
        "Per-example event timeline connecting rank, target-vs-distractor, decision, cosine, and KL events.",
    )


def plot_top1_transition_ribbons(
    ctx: bench.RunContext,
    transition_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    n_layers: int,
) -> None:
    """Compressed top-1 token biographies as ribbons."""
    if not transition_rows or not event_rows:
        return
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    order_rows = sorted(
        event_rows,
        key=lambda r: (
            CATEGORIES.index(r["category"]) if r["category"] in CATEGORIES else len(CATEGORIES),
            str(r.get("relation_family", "")),
            number_or_none(r.get("decision_depth")) if number_or_none(r.get("decision_depth")) is not None else float(n_layers + 1),
            r["example_id"],
        ),
    )
    order = [r["example_id"] for r in order_rows]
    y_pos = {ex_id: i for i, ex_id in enumerate(order)}
    by_example: dict[str, list[dict[str, Any]]] = {}
    for row in transition_rows:
        by_example.setdefault(row["example_id"], []).append(row)

    fig_height = max(5.5, min(16.0, 0.30 * len(order) + 2.4))
    fig, ax = bench.new_figure(figsize=(12.0, fig_height))
    role_colors = {
        "target": "#009E73",
        "distractor": "#D55E00",
        "final": "#0072B2",
        "other": "#b8b8b8",
    }
    for ex_id in order:
        y = y_pos[ex_id]
        rows = by_example.get(ex_id, [])
        cat = rows[0]["category"] if rows else ""
        ax.hlines(y, 0, n_layers, color=category_color(cat), alpha=0.10, linewidth=5.0)
        for seg in rows:
            if bool(seg.get("is_target")):
                role = "target"
            elif bool(seg.get("is_distractor")):
                role = "distractor"
            elif bool(seg.get("is_final_top1_token")):
                role = "final"
            else:
                role = "other"
            start = number_or_none(seg.get("start_depth")) or 0.0
            end = number_or_none(seg.get("end_depth")) or start
            ax.plot([start, end], [y, y], color=role_colors[role], linewidth=4.0, solid_capstyle="butt", alpha=0.92)
            duration = number_or_none(seg.get("duration_depths")) or 0.0
            if duration >= max(4, n_layers * 0.12):
                token = str(seg.get("token", ""))
                ax.text((start + end) / 2, y - 0.18, textwrap.shorten(token, width=10, placeholder="…"),
                        fontsize=6.3, ha="center", va="top", color="#333333")
    ax.axvline(n_layers, color="#444444", linestyle=":", linewidth=1.0, alpha=0.65)
    ax.set_yticks(range(len(order)))
    labels = []
    row_by_id = {r["example_id"]: r for r in event_rows}
    for ex_id in order:
        r = row_by_id[ex_id]
        labels.append(f"{r['category'][:2]}:{short_example_label(ex_id, 28)}")
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlim(-0.5, n_layers + 0.8)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.invert_yaxis()
    ax.set_xlabel("depth")
    ax.set_title("Top-1 token ribbons: long middle-layer winners are often not the final answer")
    handles = [Line2D([0], [0], color=color, lw=4, label=label) for label, color in [
        ("target token", role_colors["target"]),
        ("distractor token", role_colors["distractor"]),
        ("final top-1 token", role_colors["final"]),
        ("other top-1 token", role_colors["other"]),
    ]]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    bench.save_figure(
        ctx,
        fig,
        "top1_transition_ribbons.png",
        "Compressed top-1 token biographies; exposes long-lived intermediate winners and target/distractor segments.",
    )


def plot_final_readout_scatter(
    ctx: bench.RunContext,
    event_rows: list[dict[str, Any]],
) -> None:
    """Final confidence vs entropy, separating target success from confidence."""
    if not event_rows:
        return
    fig, ax = bench.new_figure(figsize=(8.4, 6.1))
    seen: set[str] = set()
    for row in event_rows:
        entropy = number_or_none(row.get("final_entropy_bits"))
        prob = number_or_none(row.get("final_top1_prob"))
        if entropy is None or prob is None:
            continue
        cat = row["category"]
        target_status = row.get("final_top1_is_target")
        outcome = str(row.get("final_outcome", ""))
        if target_status in (None, ""):
            marker = "s"
            label = f"{cat}: unlabeled"
        elif bool(target_status):
            marker = "o"
            label = f"{cat}: target top-1"
        elif outcome == "target_beats_distractor_not_top1":
            marker = "^"
            label = f"{cat}: target beats distractor"
        else:
            marker = "X"
            label = f"{cat}: target not top-1"
        ax.scatter(
            entropy,
            prob,
            color=category_color(cat),
            marker=marker,
            s=78,
            alpha=0.84,
            edgecolors="black" if target_status not in (None, "") else "none",
            linewidths=0.65,
            label=label if label not in seen else None,
        )
        seen.add(label)
        should_annotate = (
            (target_status is False and cat in {"fact", "counterfactual"})
            or (cat == "ambiguous" and prob > 0.5)
        )
        if should_annotate:
            ax.annotate(short_example_label(row["example_id"], 22), (entropy, prob), textcoords="offset points",
                        xytext=(5, 4), fontsize=7)
    ax.set_xlabel("final entropy (bits)")
    ax.set_ylabel("final top-1 probability")
    ax.set_title("Final readout: confidence, target success, and matched-distractor success are different")
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="best")
    bench.save_figure(
        ctx,
        fig,
        "final_readout_scatter.png",
        "Final entropy vs top-1 probability, with target success and target-vs-distractor outcome marked separately.",
    )


def add_biography_event_lines(axes: Iterable[Any], events: dict[str, int | None]) -> None:
    colors = {
        "rank≤5": "#555555",
        "target>distr": "#D55E00",
        "target top-1": "#009E73",
        "decision": "#0072B2",
        "KL≤0.5": "#666666",
        "cos≥0.95": "#8a8a8a",
    }
    for ax in axes:
        for label, depth in events.items():
            if depth is None:
                continue
            ax.axvline(depth, color=colors.get(label, "#666666"), linestyle=":" if "≤" in label or "≥" in label else "--",
                       linewidth=0.9, alpha=0.45)


def plot_biography(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    example: PromptExample,
    traj: bench.LensTrajectory,
) -> None:
    """Showcase prediction biography for one labeled example."""
    if traj.p_target is None:
        return
    import matplotlib.pyplot as plt

    depths = list(range(traj.n_depths))
    events = {
        "rank≤5": first_rank_le(traj.target_rank, 5),
        "target>distr": first_meaningful_logit_diff_positive(traj),
        "target top-1": target_first_top1(traj, single_token_id(bundle.tokenizer, example.target)),
        "decision": decision_depth(traj),
        "KL≤0.5": first_value_le(traj.kl_to_final_bits, 0.5),
        "cos≥0.95": first_value_ge(traj.cosine_to_final, 0.95),
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6))
    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    add_biography_event_lines(axes.flat, events)

    ax = axes[0, 0]
    ax.plot(depths, traj.p_target, linewidth=2.4, label=f"p(target = {bench.visible_token(example.target or '')})")
    if traj.p_distractor is not None:
        ax.plot(depths, traj.p_distractor, linewidth=2.2, label=f"p(distractor = {bench.visible_token(example.distractor or '')})")
    ax.plot(depths, traj.top1_probs, linewidth=1.1, linestyle="--", color="#555555", label="p(top-1 at depth)")
    ax.set_ylabel("probability")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Target, distractor, and current winner")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[0, 1]
    if traj.logit_target is not None and traj.logit_distractor is not None:
        diffs = [t - d for t, d in zip(traj.logit_target, traj.logit_distractor)]
        ax.axhline(0.0, color="#333333", linewidth=0.9, alpha=0.6)
        ax.axhline(LOGIT_DIFF_MEANINGFUL_MARGIN, color="#555555", linewidth=0.9, linestyle="--", alpha=0.6)
        ax.plot(depths, diffs, linewidth=2.4)
        ax.set_ylabel("logit(target) - logit(distractor)")
        ax.set_title("Matched target-vs-distractor readout")
    else:
        ax.text(0.5, 0.5, "No target/distractor pair", ha="center", va="center", transform=ax.transAxes)

    ax = axes[1, 0]
    if traj.target_rank is not None:
        ax.plot(depths, traj.target_rank, linewidth=2.4, label="target rank")
        ax.axhline(5, color="#555555", linewidth=0.9, linestyle="--", alpha=0.55)
        ax.set_yscale("log")
        ax.invert_yaxis()
        ax.set_ylabel("rank, lower is better")
    else:
        ax.text(0.5, 0.5, "No labeled target", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("depth")
    ax.set_title("Rank can improve before probability looks large")

    ax = axes[1, 1]
    ax.plot(depths, traj.entropy_bits, linewidth=2.0, label="entropy")
    ax.plot(depths, traj.kl_to_final_bits, linewidth=2.0, label="KL(final || depth)")
    ax.axhline(0.5, color="#555555", linewidth=0.9, linestyle="--", alpha=0.55)
    ax.set_yscale("log")
    ax.set_xlabel("depth")
    ax.set_ylabel("bits, log scale")
    ax.set_title("Sharpness, decoded convergence, and geometry")
    ax2 = ax.twinx()
    ax2.plot(depths, traj.cosine_to_final, linewidth=1.6, linestyle=":", color="#333333", label="cosine to final")
    ax2.set_ylabel("cosine", color="#333333")
    ax2.tick_params(axis="y", labelcolor="#333333")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8, frameon=False, loc="best")

    # Sparse token annotations: first, event depths, and final, not every nth point.
    annotated = {0, traj.n_depths - 1}
    annotated.update(d for d in events.values() if d is not None)
    for depth in sorted(annotated):
        if 0 <= depth < traj.n_depths:
            axes[0, 0].annotate(
                bench.visible_token(traj.top1_texts[depth]),
                (depth, traj.top1_probs[depth]),
                textcoords="offset points",
                xytext=(0, 8),
                fontsize=7,
                rotation=42,
                ha="left",
            )
    event_caption = ", ".join(f"{k}:{v}" for k, v in events.items() if v is not None) or "no labeled events occurred"
    prompt = textwrap.fill(example.prompt, width=110)
    fig.suptitle(f"Prediction biography: {example.example_id}\n{prompt}\nEvents: {event_caption}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    bench.save_figure(
        ctx,
        fig,
        f"biography_{bench.sanitize_tag(example.example_id)}.png",
        "Showcase example with event lines, target/distractor probability, logit difference, rank, entropy, KL, and cosine.",
    )


def plot_convergence_lag(
    ctx: bench.RunContext,
    event_rows: list[dict[str, Any]],
    *,
    categories: tuple[str, ...] = CATEGORIES,
) -> None:
    """Visualize the lag between geometric and decoded convergence."""
    if not event_rows:
        return
    fig, ax = bench.new_figure(figsize=(9.8, 5.8))
    plotted = False
    for cat_i, cat in enumerate(categories):
        rows = [r for r in event_rows if r.get("category") == cat]
        if not rows:
            continue
        lags: list[float] = []
        for r in rows:
            cos_d = number_or_none(r.get("cosine_to_final_first_ge_0.95"))
            kl_d = number_or_none(r.get("kl_to_final_first_le_0.5_bits"))
            if cos_d is not None and kl_d is not None:
                lags.append(kl_d - cos_d)
        if not lags:
            continue
        plotted = True
        color = category_color(cat)
        offsets = [((i - (len(lags) - 1) / 2) * 0.075) for i in range(len(lags))]
        ax.scatter([cat_i + o for o in offsets], lags, color=color, alpha=0.70, s=36, label=f"{cat} (n={len(lags)})",
                   edgecolors="white", linewidths=0.35)
        q25 = percentile(lags, 0.25)
        q50 = percentile(lags, 0.50)
        q75 = percentile(lags, 0.75)
        ax.plot([cat_i - 0.25, cat_i + 0.25], [q50, q50], color=color, linewidth=3.2)
        ax.vlines(cat_i, q25, q75, color=color, linewidth=5, alpha=0.18)
    if not plotted:
        bench.close_figure(fig)
        return
    ax.axhline(0, color="#333333", ls=":", lw=1, alpha=0.7)
    ax.text(0.01, 0.02, "positive: residual geometry stabilizes first\nnegative: decoded distribution stabilizes first",
            transform=ax.transAxes, fontsize=8, color="#555555", va="bottom")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_ylabel("KL-stable depth minus cosine-stable depth")
    ax.set_title("Geometric vs decoded convergence lag")
    ax.legend(fontsize=8, frameon=False, loc="best")
    bench.style_ax(ax, legend=False)
    bench.save_figure(
        ctx,
        fig,
        "convergence_lag.png",
        "Key Lab 1 concept: geometric closeness and decoded-distribution closeness stabilize at different depths.",
    )


def plot_event_ordering(
    ctx: bench.RunContext,
    event_rows: list[dict[str, Any]],
    n_layers: int,
) -> None:
    """Median event-depth fractions by category, with event absence visible by count labels."""
    events = [
        "target_rank_first_le_5",
        "target_first_beats_distractor",
        "target_first_top1",
        "decision_depth",
        "cosine_to_final_first_ge_0.95",
        "kl_to_final_first_le_0.5_bits",
    ]
    fig, ax = bench.new_figure(figsize=(11.8, 6.0))
    offsets = {cat: (i - (len(CATEGORIES) - 1) / 2) * 0.13 for i, cat in enumerate(CATEGORIES)}
    plotted = False
    for category in CATEGORIES:
        rows = [r for r in event_rows if r["category"] == category]
        if not rows:
            continue
        for j, event in enumerate(events):
            vals = numeric_values(rows, event)
            if not vals:
                continue
            plotted = True
            y = statistics.median(vals) / max(1, n_layers)
            ax.scatter(j + offsets.get(category, 0.0), y, s=82, color=category_color(category), label=category,
                       edgecolors="white", linewidths=0.5)
            ax.annotate(f"n={len(vals)}/{len(rows)}", (j + offsets.get(category, 0.0), y),
                        textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
    if not plotted:
        bench.close_figure(fig)
        return
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, frameon=False)
    ax.set_xticks(range(len(events)))
    ax.set_xticklabels([event_label(e) for e in events], rotation=28, ha="right")
    ax.set_ylim(-0.04, 1.04)
    ax.set_ylabel("median event depth / number of blocks")
    ax.set_title("Event ordering by category, conditional on event occurring")
    bench.save_figure(ctx, fig, "event_ordering.png", "Median event-depth fraction by category, annotated with occurrence counts.")


def plot_relation_event_matrix(
    ctx: bench.RunContext,
    event_rows: list[dict[str, Any]],
    n_layers: int,
) -> None:
    """If relation-family metadata exists, compare event timing across families."""
    if not event_rows:
        return
    import matplotlib.pyplot as plt

    events = [
        "target_rank_first_le_5",
        "target_first_beats_distractor",
        "target_first_top1",
        "decision_depth",
        "kl_to_final_first_le_0.5_bits",
    ]
    families = sorted({str(r.get("relation_family") or r.get("category")) for r in event_rows})
    if len(families) < 2:
        return
    # Keep dense default runs readable, but still make large custom sets useful.
    family_rows: list[tuple[str, str, list[dict[str, Any]]]] = []
    for fam in families:
        rows = [r for r in event_rows if str(r.get("relation_family") or r.get("category")) == fam]
        cats = sorted({r["category"] for r in rows}, key=lambda c: CATEGORIES.index(c) if c in CATEGORIES else 999)
        family_rows.append((cats[0] if cats else "", fam, rows))
    family_rows.sort(key=lambda t: (CATEGORIES.index(t[0]) if t[0] in CATEGORIES else 999, t[1]))

    data: list[list[float]] = []
    labels: list[str] = []
    for cat, fam, rows in family_rows:
        labels.append(f"{cat[:2]}:{fam} (n={len(rows)})")
        row_vals: list[float] = []
        for event in events:
            vals = numeric_values(rows, event)
            row_vals.append(float("nan") if not vals else statistics.median(vals) / max(1, n_layers))
        data.append(row_vals)
    fig_height = max(4.8, min(12.0, 0.34 * len(labels) + 1.8))
    fig, ax = bench.new_figure(figsize=(10.8, fig_height))
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("#d8d8d8")
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(events)))
    ax.set_xticklabels([event_label(e, multiline=True) for e in events], fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    for i, row in enumerate(data):
        for j, value in enumerate(row):
            if math.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color="#111111")
    ax.set_title("Relation-family event matrix: do different relation types emerge on different clocks?")
    ax.set_xlabel("event metric, value = median event depth / layers")
    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.01)
    cbar.set_label("normalized depth")
    fig.tight_layout()
    bench.save_figure(
        ctx,
        fig,
        "relation_event_matrix.png",
        "Relation-family matrix of median event depths; uses relation= tags from custom prompt notes when present.",
    )


def plot_phase_heatmap(
    ctx: bench.RunContext,
    phase_rows: list[dict[str, Any]],
) -> None:
    """Coarse phase atlas: where entropy and KL fall by category."""
    if not phase_rows:
        return
    import matplotlib.pyplot as plt

    metrics = [("mean_entropy_bits", "entropy bits"), ("mean_kl_to_final_bits", "KL-to-final bits")]
    categories = [cat for cat in CATEGORIES if any(r.get("category") == cat for r in phase_rows)]
    if not categories:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(11.8, max(3.4, 0.45 * len(categories) + 2.0)))
    if len(metrics) == 1:
        axes = [axes]
    for ax, (metric, title) in zip(axes, metrics):
        data: list[list[float]] = []
        for cat in categories:
            row_vals: list[float] = []
            for phase in PHASE_ORDER:
                vals = [number_or_none(r.get(metric)) for r in phase_rows if r.get("category") == cat and r.get("phase") == phase]
                vals = [v for v in vals if v is not None]
                row_vals.append(float("nan") if not vals else statistics.fmean(vals))
            data.append(row_vals)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#d8d8d8")
        im = ax.imshow(data, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(PHASE_ORDER)))
        ax.set_xticklabels([PHASE_DISPLAY[p] for p in PHASE_ORDER], rotation=25, ha="right")
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories)
        ax.set_title(title)
        for i, row in enumerate(data):
            for j, value in enumerate(row):
                if math.isfinite(value):
                    ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=7, color="#111111")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("Readout phase atlas: coarse depth bands make cross-model comparisons less brittle", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    bench.save_figure(
        ctx,
        fig,
        "readout_phase_heatmap.png",
        "Coarse phase heatmaps for entropy and KL-to-final by category.",
    )


# ---------------------------------------------------------------------------
# Aggregation and summary
# ---------------------------------------------------------------------------


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, bool):
            out.append(float(value))
        else:
            try:
                out.append(float(value))
            except (TypeError, ValueError):
                pass
    return out


def median_or_blank(rows: list[dict[str, Any]], key: str, digits: int = 3) -> Any:
    values = numeric_values(rows, key)
    if not values:
        return ""
    return round(statistics.median(values), digits)


def occurrence_count(rows: list[dict[str, Any]], key: str) -> int:
    """How many rows have a numeric value for `key` (event actually occurred)."""
    return len(numeric_values(rows, key))


def mean_or_blank(rows: list[dict[str, Any]], key: str, digits: int = 3) -> Any:
    values = numeric_values(rows, key)
    if not values:
        return ""
    return round(statistics.fmean(values), digits)


def category_stats(event_rows: list[dict[str, Any]], n_layers: int) -> list[dict[str, Any]]:
    """Aggregate per-example events into a category headline table."""
    out: list[dict[str, Any]] = []
    for category in CATEGORIES:
        rows = [r for r in event_rows if r["category"] == category]
        if not rows:
            continue
        decision = median_or_blank(rows, "decision_depth")
        row = {
            "category": category,
            "n_examples": len(rows),
            "mean_prompt_tokens": mean_or_blank(rows, "n_prompt_tokens", 2),
            "median_decision_depth": decision,
            "median_decision_depth_frac": "" if decision == "" else round(float(decision) / n_layers, 3),
            # Event medians are conditional on the event occurring; the paired
            # n_* columns say how many of n_examples that actually was.
            "median_target_first_top1": median_or_blank(rows, "target_first_top1"),
            "n_target_first_top1": occurrence_count(rows, "target_first_top1"),
            "median_target_stable_top1_depth": median_or_blank(rows, "target_stable_top1_depth"),
            "n_target_stable_top1_depth": occurrence_count(rows, "target_stable_top1_depth"),
            "median_target_first_beats_distractor": median_or_blank(rows, "target_first_beats_distractor"),
            "n_target_first_beats_distractor": occurrence_count(rows, "target_first_beats_distractor"),
            "median_target_first_beats_distractor_raw": median_or_blank(rows, "target_first_beats_distractor_raw"),
            "n_target_first_beats_distractor_raw": occurrence_count(rows, "target_first_beats_distractor_raw"),
            "median_target_stable_beats_distractor": median_or_blank(rows, "target_stable_beats_distractor"),
            "n_target_stable_beats_distractor": occurrence_count(rows, "target_stable_beats_distractor"),
            "median_target_rank_first_le_5": median_or_blank(rows, "target_rank_first_le_5"),
            "n_target_rank_first_le_5": occurrence_count(rows, "target_rank_first_le_5"),
            "mean_final_entropy_bits": mean_or_blank(rows, "final_entropy_bits"),
            "mean_final_top1_prob": mean_or_blank(rows, "final_top1_prob"),
            "mean_final_top1_margin": mean_or_blank(rows, "final_top1_margin"),
            "mean_final_p_target": mean_or_blank(rows, "final_p_target"),
            "mean_final_target_rank": mean_or_blank(rows, "final_target_rank", 2),
            "target_final_top1_rate": mean_or_blank(rows, "final_top1_is_target", 3),
            "target_final_beats_distractor_margin_rate": mean_or_blank(rows, "final_target_beats_distractor_by_margin", 3),
            "median_kl_to_final_first_le_0.5_bits": median_or_blank(rows, "kl_to_final_first_le_0.5_bits"),
            "n_kl_to_final_first_le_0.5_bits": occurrence_count(rows, "kl_to_final_first_le_0.5_bits"),
            "median_cosine_to_final_first_ge_0.95": median_or_blank(rows, "cosine_to_final_first_ge_0.95"),
            "n_cosine_to_final_first_ge_0.95": occurrence_count(rows, "cosine_to_final_first_ge_0.95"),
            "mean_convergence_lag_depths": mean_or_blank(rows, "convergence_lag_depths", 2),
            "mean_top1_flip_count": mean_or_blank(rows, "top1_flip_count", 2),
        }
        out.append(row)
    return out


def relation_stats(event_rows: list[dict[str, Any]], n_layers: int) -> list[dict[str, Any]]:
    """Aggregate event timings by relation_family metadata.

    This table is especially useful with ``data/relation_probes_lab1.csv`` or
    custom prompt sets whose ``note`` column contains ``relation=...`` tags. It
    also gives the built-in examples a lightweight relation lens rather than
    collapsing everything into fact vs ambiguous vs counterfactual.
    """
    groups = sorted({str(r.get("relation_family") or r.get("category") or "unknown") for r in event_rows})
    out: list[dict[str, Any]] = []
    for family in groups:
        rows = [r for r in event_rows if str(r.get("relation_family") or r.get("category") or "unknown") == family]
        if not rows:
            continue
        categories = ",".join(sorted({str(r.get("category", "")) for r in rows}))
        decision = median_or_blank(rows, "decision_depth")
        row = {
            "relation_family": family,
            "categories": categories,
            "n_examples": len(rows),
            "median_decision_depth": decision,
            "median_decision_depth_frac": "" if decision == "" else round(float(decision) / max(1, n_layers), 4),
            "median_target_rank_first_le_5": median_or_blank(rows, "target_rank_first_le_5"),
            "n_target_rank_first_le_5": occurrence_count(rows, "target_rank_first_le_5"),
            "median_target_first_beats_distractor": median_or_blank(rows, "target_first_beats_distractor"),
            "n_target_first_beats_distractor": occurrence_count(rows, "target_first_beats_distractor"),
            "median_target_first_top1": median_or_blank(rows, "target_first_top1"),
            "n_target_first_top1": occurrence_count(rows, "target_first_top1"),
            "median_kl_to_final_first_le_0.5_bits": median_or_blank(rows, "kl_to_final_first_le_0.5_bits"),
            "n_kl_to_final_first_le_0.5_bits": occurrence_count(rows, "kl_to_final_first_le_0.5_bits"),
            "median_cosine_to_final_first_ge_0.95": median_or_blank(rows, "cosine_to_final_first_ge_0.95"),
            "n_cosine_to_final_first_ge_0.95": occurrence_count(rows, "cosine_to_final_first_ge_0.95"),
            "mean_final_entropy_bits": mean_or_blank(rows, "final_entropy_bits"),
            "mean_final_p_target": mean_or_blank(rows, "final_p_target"),
            "target_final_top1_rate": mean_or_blank(rows, "final_top1_is_target", 3),
            "target_final_beats_distractor_margin_rate": mean_or_blank(rows, "final_target_beats_distractor_by_margin", 3),
            "mean_convergence_lag_depths": mean_or_blank(rows, "convergence_lag_depths", 2),
        }
        out.append(row)
    return out


def write_plot_reading_guide(ctx: bench.RunContext) -> None:
    """Write a small map from each plot to the question it answers."""
    rows = [
        {"open_order": 1, "plot": "readout_dashboard.png", "question": "Do sharpness, decoded convergence, commitment, and geometry move together?", "use_when": "first overview"},
        {"open_order": 2, "plot": "event_timeline.png", "question": "For each example, which event happened first and which never happened?", "use_when": "debugging overclaims"},
        {"open_order": 3, "plot": "event_depth_heatmap.png", "question": "Where are the missing events, and are they category-specific?", "use_when": "seeing gray as data"},
        {"open_order": 4, "plot": "top1_transition_ribbons.png", "question": "Which top-1 tokens occupy long depth intervals before the final readout?", "use_when": "reading biographies at scale"},
        {"open_order": 5, "plot": "convergence_lag.png", "question": "Does residual geometry stabilize before the decoded distribution?", "use_when": "readout-is-an-instrument lesson"},
        {"open_order": 6, "plot": "final_readout_scatter.png", "question": "Is high final confidence the same as labeled target success?", "use_when": "confidence/correctness audit"},
        {"open_order": 7, "plot": "relation_event_matrix.png", "question": "Do relation families emerge at different depths?", "use_when": "custom relation prompt sets"},
        {"open_order": 8, "plot": "readout_phase_heatmap.png", "question": "Which coarse phase does most of the distributional sharpening occupy?", "use_when": "cross-model comparison"},
    ]
    path = ctx.path("tables", "plot_reading_guide.csv")
    bench.write_csv_with_context(ctx, path, rows)
    ctx.register_artifact(path, "table", "Reading order and conceptual purpose for the Lab 1 plots.")


def render_category_table(cat_rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| category | n | mean prompt tokens | median decision depth | frac of L | target first top-1 | target beats distractor (>1, stable lead) | KL stable | cosine stable | lag | final entropy | final p(target) | target final top-1 rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in cat_rows:
        lines.append(
            f"| {r['category']} | {r['n_examples']} | {r['mean_prompt_tokens']} | "
            f"{r['median_decision_depth']} | {r['median_decision_depth_frac']} | "
            f"{r['median_target_first_top1']} (n={r['n_target_first_top1']}) | "
            f"{r['median_target_first_beats_distractor']} (n={r['n_target_first_beats_distractor']}) | "
            f"{r.get('median_kl_to_final_first_le_0.5_bits', '')} (n={r.get('n_kl_to_final_first_le_0.5_bits', '')}) | "
            f"{r.get('median_cosine_to_final_first_ge_0.95', '')} (n={r.get('n_cosine_to_final_first_ge_0.95', '')}) | "
            f"{r.get('mean_convergence_lag_depths', '')} | "
            f"{r['mean_final_entropy_bits']} | {r['mean_final_p_target']} | "
            f"{r['target_final_top1_rate']} |"
        )
    return lines


def render_summary(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    event_rows: list[dict[str, Any]],
    cat_rows: list[dict[str, Any]],
    dropped: int,
    claims: list[dict[str, str]],
) -> str:
    """Write the run summary as the lab's small paper."""
    a = bundle.anatomy
    L = a.n_layers
    lines = [
        "# Lab 1 run summary: residual stream and logit lens",
        "",
        "## Run identity",
        "",
        f"- model: `{a.model_id}` ({L} blocks, d_model {a.d_model})",
        f"- primary device: `{bundle.device}` | input device: `{bundle.input_device}` | lens device: `{bundle.lens_device}`",
        f"- dtype: `{ctx.args.dtype}` | quantization: `{ctx.args.quantization}` | top-k: {ctx.args.topk}",
        f"- examples: {len(event_rows)} kept, {dropped} dropped at tokenization",
        "- evidence level: `OBS` only",
        "- self-checks: hook parity and lens-at-final-depth diagnostics passed before the experiment loop",
        "",
        "## 1. What behavior was studied?",
        "",
        "Next-token prediction on controlled prompt families: high-certainty facts, ambiguous continuations, counterfactual contexts, and optional weak controls.",
        "",
        "## 2. What internal object was measured?",
        "",
        "The pre-final-norm residual stream at the final token position after every block. Each stream was decoded with the model's own final norm and unembedding (the *raw* logit lens). The run recorded top-k tokens, target and distractor metrics, entropy, KL-to-final, top-1 margin, residual norm, update norm, and residual cosine-to-final. Multiple convergence metrics are reported because geometric closeness of the residual (cosine) and sharpness of the decoded distribution (entropy, KL, rank) can (and often do) occur at different depths.",
        "",
        "## 3. What intervention or control was used?",
        "",
        "No intervention was used. This lab is observational. Ambiguous prompts and optional controls are negative controls for over-reading early top-1 stability as model knowledge.",
        "",
        "## 4. Headline numbers",
        "",
    ]
    lines.extend(render_category_table(cat_rows))
    lines += [
        "",
        f"Definitions: `decision_depth` is the first depth after which the final top-1 token remains top-1. "
        f"`target beats distractor` is the first depth where target logit exceeds distractor logit by "
        f">{LOGIT_DIFF_MEANINGFUL_MARGIN:g} and the target keeps the lead thereafter. "
        "`target_first_beats_distractor_raw` records the ungated first positive crossing as a diagnostic only. "
        "Blank cells mean the metric is not defined for that category or never occurred. "
        "On many factual prompts the model's actual top-1 remains a discourse continuation (e.g. 'known') even while the labeled target improves in rank and beats its distractor. This is the central observation the lab is designed to surface.",
        "",
        "## 5. What claim is supported, and at what evidence level?",
        "",
        "Only observational claims are supported. The strongest claims are about what the raw final readout can decode from intermediate streams under this prompt distribution. The added answer-shaped and discourse-heavy fact prompts plus the explicit convergence_lag plot are designed to make the distinction between geometric convergence and decoded top-1 commitment unmistakable.",
        "",
    ]
    if claims:
        lines.append("Drafted claims, still requiring student editing:")
        lines.append("")
        for claim in claims:
            lines.append(f"- `{claim['id']}` {claim['tag']}: {claim['text']}")
        lines.append("")
    lines += [
        "## 6. What claim is not supported?",
        "",
        "- The run does not show that the model knows the answer at the first readable depth.",
        "- The run does not show that later layers causally use a decoded intermediate token.",
        "- The run does not show that lower entropy means correctness.",
        "- The run does not show that the final readout basis is appropriate for every middle layer.",
        "",
        "## 7. What would falsify or weaken the interpretation?",
        "",
        "- A tuned lens moves the event depth materially earlier or later, or changes which examples look early.",
        "- Length-matched ambiguous or control prompts produce the same event-depth pattern as facts.",
        "- Activation patching fails to change behavior when patching the supposedly informative stream position.",
        "- A held-out prompt family reverses the category pattern.",
        "",
        "## Per-example event table",
        "",
        "| example | category | decision | flips | target first top-1 | target beats distractor (>1, stable lead) | target rank <= 5 | final p(target) | final target rank | final entropy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def cell(row: dict[str, Any], key: str) -> Any:
        value = row.get(key, "")
        return "" if value is None else value

    for r in event_rows:
        lines.append(
            f"| {r['example_id']} | {r['category']} | {cell(r, 'decision_depth')} | "
            f"{cell(r, 'top1_flip_count')} | {cell(r, 'target_first_top1')} | "
            f"{cell(r, 'target_first_beats_distractor')} | {cell(r, 'target_rank_first_le_5')} | "
            f"{cell(r, 'final_p_target')} | {cell(r, 'final_target_rank')} | {cell(r, 'final_entropy_bits')} |"
        )
    lines += [
        "",
        "## Reading path",
        "",
        "Microscope validation (smoke ritual):",
        "1. The three instrument locks (`diagnostics/hook_parity*`, `logit_lens_self_check.json`, `tokenization_report.csv`)",
        "2. `claim_ledger.md` at the course root (`interpretability/`) + one `state_card.md` + basic residual norm plot",
        "",
        "Lab 1 science:",
        "3. `logit_lens_card.md` for scope, headline numbers, claims, and non-claims",
        "4. `tables/final_readout_audit.csv` (the key place to see discourse bias: target rank improves but actual top-1 is often 'known')",
        "5. `state/<example_id>/state_card.md` for one fact and one counterfactual prompt",
        "6. `plots/readout_dashboard.png` + `convergence_lag.png` (explicit lag between geometric closeness of the residual and decoded top-1 commitment — the central 'instrument' lesson), `final_readout_scatter.png`, and `event_depth_heatmap.png`",
        "7. `plots/target_rank_by_depth.png`, `plots/logit_diff_by_depth.png`, and `plots/kl_to_final_by_depth.png`",
        "8. `tables/top1_transition_segments.csv` and `tables/trajectory_events.csv` for outliers and blank cells (n counts show when an event never occurred)",
        "",
        "## Student tooling note",
        "",
        "_What was AI-drafted, what was hand-verified, and one overclaim or bug you caught._",
        "",
    ]
    return "\n".join(lines)


def draft_claims(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    cat_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Draft OBS-tagged ledger claims with measured numbers filled in."""
    by_cat = {r["category"]: r for r in cat_rows}
    run_name = ctx.run_dir.name
    L = bundle.anatomy.n_layers
    claims: list[dict[str, str]] = []

    fact = by_cat.get("fact")
    ambig = by_cat.get("ambiguous")
    cf = by_cat.get("counterfactual")
    control = by_cat.get("control")

    if fact:
        if fact["median_target_first_top1"] in (None, ""):
            target_clause = (
                f"the labeled target never became top-1 at any depth "
                f"(0/{fact['n_examples']} examples)."
            )
        else:
            target_clause = (
                f"the labeled target first became top-1 at median depth {fact['median_target_first_top1']} "
                f"(median over the {fact['n_target_first_top1']}/{fact['n_examples']} examples where it occurred)."
            )
        claims.append(
            {
                "id": f"{LAB_ID}-C1",
                "tag": "OBS",
                "text": (
                    f"On {fact['n_examples']} factual prompts, {bundle.anatomy.model_id}'s final top-1 token "
                    f"stabilized under the raw logit lens at median depth {fact['median_decision_depth']}/{L}; "
                    + target_clause
                ),
                "artifact": f"runs/{run_name}/tables/category_summary.csv",
                "falsifier": "A tuned lens or held-out fact family places stabilization materially earlier/later or changes which token stabilizes.",
            }
        )
    if fact and ambig and fact.get("mean_final_entropy_bits") not in (None, "") and ambig.get("mean_final_entropy_bits") not in (None, ""):
        direction = (
            "higher"
            if float(ambig["mean_final_entropy_bits"]) > float(fact["mean_final_entropy_bits"])
            else "lower"
        )
        claims.append(
            {
                "id": f"{LAB_ID}-C2",
                "tag": "OBS",
                "text": (
                    f"Ambiguous prompts ended with {direction} final raw-lens entropy than facts: mean final entropy "
                    f"{ambig['mean_final_entropy_bits']} bits for ambiguous prompts versus "
                    f"{fact['mean_final_entropy_bits']} bits for facts."
                ),
                "artifact": f"runs/{run_name}/tables/category_summary.csv",
                "falsifier": "Length-matched ambiguous prompts show the same entropy and event-depth trajectory as facts.",
            }
        )
    if cf:
        cf_rows = [r for r in event_rows if r["category"] == "counterfactual" and r.get("final_top1_is_target") not in (None, "")]
        wins = sum(1 for r in cf_rows if bool(r.get("final_top1_is_target")))
        denom = len(cf_rows)
        claims.append(
            {
                "id": f"{LAB_ID}-C3",
                "tag": "OBS",
                "text": (
                    f"In {wins}/{denom} counterfactual prompts, the in-context target was the final top-1 token. "
                    f"Median first meaningful target-over-distractor depth was "
                    f"{cf['median_target_first_beats_distractor']} "
                    f"(median over the {cf['n_target_first_beats_distractor']}/{cf['n_examples']} examples "
                    f"where the target exceeded the distractor by >{LOGIT_DIFF_MEANINGFUL_MARGIN:g} logit "
                    "and kept the lead)."
                ),
                "artifact": f"runs/{run_name}/tables/trajectory_events.csv",
                "falsifier": "Counterfactual and factual trajectories become indistinguishable after prompt-length and syntax matching.",
            }
        )
    if control:
        claims.append(
            {
                "id": f"{LAB_ID}-C4",
                "tag": "OBS",
                "text": (
                    f"Optional control prompts had median decision depth {control['median_decision_depth']}/{L} "
                    f"and mean final entropy {control['mean_final_entropy_bits']} bits, showing how the stability metric behaves on weak prompts."
                ),
                "artifact": f"runs/{run_name}/tables/category_summary.csv",
                "falsifier": "A larger control family shows the same event-depth and entropy profile as high-certainty facts.",
            }
        )
    return claims



def write_event_definition_manifest(ctx: bench.RunContext, n_layers: int) -> None:
    """Write the semantic contract for every event depth used in this lab."""
    path = ctx.path("diagnostics", "event_definitions.json")
    bench.write_json(
        path,
        {
            "evidence_level": "OBS",
            "n_layers": n_layers,
            "stream_semantics": "streams[k] is the pre-final-norm residual stream after k blocks; depth 0 is embeddings and depth L is after all blocks.",
            "readout_semantics": "lens(k) = lm_head(final_norm(streams[k])); middle-depth readouts borrow the final readout basis.",
            "thresholds": {
                "meaningful_logit_margin": LOGIT_DIFF_MEANINGFUL_MARGIN,
                "kl_to_final_bits": 0.5,
                "cosine_to_final": 0.95,
                "target_rank": 5,
            },
            "events": {
                "decision_depth": "first depth after which the final top-1 token remains top-1",
                "target_first_top1": "first depth where the labeled target is top-1",
                "target_stable_top1_depth": "first depth after which the labeled target remains top-1",
                "target_first_beats_distractor_raw": "first depth where target logit exceeds distractor; diagnostic only",
                "target_first_beats_distractor": "first depth where target logit exceeds distractor by the meaningful margin and keeps a positive lead thereafter",
                "target_stable_beats_distractor": "first depth after which target logit keeps exceeding distractor",
                "target_rank_first_le_5": "first depth where target rank is 5 or better",
                "kl_to_final_first_le_0.5_bits": "first depth where KL(final || depth) is at most 0.5 bits",
                "cosine_to_final_first_ge_0.95": "first depth where residual cosine-to-final is at least 0.95",
            },
            "non_claim": "None of these events proves that the model 'knows', 'believes', or 'stores' the target at that depth. They are properties of the *raw final unembedding applied to intermediate pre-final-norm residuals*. Later layers may or may not use the information; only causal interventions (later labs) can address use. The lens borrows the final readout basis at every depth.",
        },
    )
    ctx.register_artifact(path, "diagnostic", "Definitions and thresholds for Lab 1 event-depth metrics.")


def write_logit_lens_card(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    cat_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    dropped: int,
    claims: list[dict[str, str]],
) -> None:
    """A compact deliverable students can read before diving into CSVs."""
    by_cat = {r["category"]: r for r in cat_rows}

    def cat_line(cat: str) -> str:
        r = by_cat.get(cat)
        if not r:
            return f"- `{cat}`: not present in this run."
        return (
            f"- `{cat}`: n={r['n_examples']}, median decision depth {r['median_decision_depth']}, "
            f"mean final entropy {r['mean_final_entropy_bits']}, target top-1 rate {r['target_final_top1_rate']}."
        )

    n_labeled = sum(1 for r in event_rows if r.get("target") not in (None, ""))
    n_target_top1 = sum(1 for r in event_rows if bool(r.get("final_top1_is_target")))
    lines = [
        "# Lab 1 logit-lens card",
        "",
        "## Scope",
        "",
        f"Model: `{bundle.anatomy.model_id}` with {bundle.anatomy.n_layers} blocks. Evidence level: `OBS`.",
        f"Run: `{ctx.run_dir.name}`. Examples kept: {len(event_rows)}. Dropped at tokenization: {dropped}.",
        "",
        "## What was measured",
        "",
        "For each prompt, the lab decoded the final-position residual stream at every depth using the model's own final norm and unembedding.",
        "The readout answers this question: what would the final vocabulary head say if it were attached here?",
        "",
        "## Headline by family",
        "",
        cat_line("fact"),
        cat_line("ambiguous"),
        cat_line("counterfactual"),
        cat_line("control"),
        "",
        "## Correctness and confidence are separate",
        "",
        f"Among labeled examples, the target was final top-1 in {n_target_top1}/{n_labeled} cases.",
        "Use `tables/final_readout_audit.csv` before calling any trajectory a success or failure.",
        "",
        "## Draft claims",
        "",
    ]
    if claims:
        for claim in claims:
            lines.append(f"- `{claim['id']}` {claim['tag']}: {claim['text']}")
            lines.append(f"  - falsifier: {claim['falsifier']}")
    else:
        lines.append("No claims were drafted because required categories were absent.")
    lines += [
        "",
        "## Non-claims",
        "",
        "- This card does not show that the model knows a fact at the first readable depth.",
        "- It does not show that later blocks causally use a decoded token.",
        "- It does not show that the raw lens is the right readout basis for middle layers.",
        "",
        "## First files to open",
        "",
        "1. `diagnostics/logit_lens_self_check.json`",
        "2. `diagnostics/hook_parity_by_layer.csv`",
        "3. `tables/final_readout_audit.csv`",
        "4. `plots/readout_dashboard.png`",
        "5. `plots/event_timeline.png`",
        "6. `plots/top1_transition_ribbons.png`",
        "7. `plots/event_ordering.png`",
        "8. `state/<example_id>/state_card.md`",
        "",
    ]
    path = ctx.path("logit_lens_card.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "Compact Lab 1 deliverable: scope, headline, claims, and non-claims.")


# ---------------------------------------------------------------------------
# Main experiment entry point called by the bench
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    examples = build_prompt_set(ctx.args)
    kept, dropped = validate_examples(ctx, bundle, examples)
    if not kept:
        raise RuntimeError("No examples survived tokenization validation.")
    print(f"[lab1] running {len(kept)} examples ({dropped} dropped)")

    # Instrument self-checks run before the science loop. These are not garnish;
    # they are the lock on the laboratory door.
    first_prompt = kept[0][0].prompt
    bench.run_hook_parity_check(ctx, bundle, first_prompt)
    first_capture = bench.run_with_residual_cache(bundle, first_prompt)
    bench.run_lens_self_check(ctx, bundle, first_capture)
    write_event_definition_manifest(ctx, bundle.anatomy.n_layers)

    results_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    per_example_curves: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    showcase: tuple[PromptExample, bench.LensTrajectory] | None = None

    for i, (ex, target_id, distractor_id) in enumerate(kept, start=1):
        capture = bench.run_with_residual_cache(bundle, ex.prompt)
        traj = bench.compute_lens_trajectory(
            bundle,
            capture,
            target_id=target_id,
            distractor_id=distractor_id,
            topk=ctx.args.topk,
        )
        bench.dump_example_state(
            ctx,
            bundle,
            ex.example_id,
            capture,
            traj,
            target=ex.target,
            distractor=ex.distractor,
        )

        event_row = trajectory_event_row(ex, capture, traj, target_id, distractor_id)
        event_rows.append(event_row)
        transition_rows.extend(top1_transition_rows(ex, traj, target_id, distractor_id))
        phase_rows.extend(readout_phase_rows(ex, traj, bundle.anatomy.n_layers))

        for depth in range(traj.n_depths):
            row: dict[str, Any] = {
                "example_id": ex.example_id,
                "category": ex.category,
                "relation_family": infer_relation_family(ex),
                "note": ex.note,
                "depth": depth,
                "top1_token_id": traj.top1_ids[depth],
                "top1_token": traj.top1_texts[depth],
                "top1_prob": round(traj.top1_probs[depth], 6),
                "top2_token_id": traj.top2_ids[depth],
                "top2_token": traj.top2_texts[depth],
                "top2_prob": round(traj.top2_probs[depth], 6),
                "top1_margin": round(traj.top1_margin[depth], 6),
                "entropy_bits": round(traj.entropy_bits[depth], 4),
                "kl_to_final_bits": round(traj.kl_to_final_bits[depth], 4),
                "cosine_to_final": round(traj.cosine_to_final[depth], 5),
                "cosine_to_prev": round_or_blank(traj.cosine_to_prev[depth], 5),
                "resid_l2": round(traj.resid_l2[depth], 3),
                "stream_delta_l2": round(traj.stream_delta_l2[depth], 3),
            }
            if traj.p_target is not None:
                row["p_target"] = round(traj.p_target[depth], 6)
                row["logit_target"] = round(traj.logit_target[depth], 4)
                row["target_rank"] = traj.target_rank[depth] if traj.target_rank is not None else ""
            if traj.p_distractor is not None:
                row["p_distractor"] = round(traj.p_distractor[depth], 6)
                row["logit_distractor"] = round(traj.logit_distractor[depth], 4)
                row["distractor_rank"] = traj.distractor_rank[depth] if traj.distractor_rank is not None else ""
            if traj.logit_target is not None and traj.logit_distractor is not None:
                row["logit_diff"] = round(traj.logit_target[depth] - traj.logit_distractor[depth], 4)
            results_rows.append(row)

        curve: dict[str, Any] = {
            "example_id": ex.example_id,
            "category": ex.category,
            "relation_family": infer_relation_family(ex),
            "entropy_bits": traj.entropy_bits,
            "kl_to_final_bits": traj.kl_to_final_bits,
            "top1_margin": traj.top1_margin,
            "cosine_to_final": traj.cosine_to_final,
            "resid_l2": traj.resid_l2,
            "stream_delta_l2": traj.stream_delta_l2,
        }
        if traj.p_target is not None:
            curve["p_target"] = traj.p_target
        if traj.target_rank is not None:
            curve["target_rank"] = traj.target_rank
        if traj.logit_target is not None and traj.logit_distractor is not None:
            curve["logit_diff"] = [t - d for t, d in zip(traj.logit_target, traj.logit_distractor)]
        per_example_curves.append(curve)

        if showcase is None and (
            ex.example_id == ctx.args.showcase
            or (ctx.args.showcase is None and ex.category == "counterfactual" and traj.p_target is not None)
        ):
            showcase = (ex, traj)

        detail = ""
        if "final_p_target" in event_row:
            detail = f" p_target(final)={event_row['final_p_target']:.3f} rank={event_row.get('final_target_rank', '')}"
        print(
            f"[lab1] [{i}/{len(kept)}] {ex.example_id} "
            f"decision_depth={event_row['decision_depth']}/{bundle.anatomy.n_layers} "
            f"flips={event_row['top1_flip_count']}{detail}"
        )

    # Tables and metrics.
    add_event_depth_fractions(event_rows, bundle.anatomy.n_layers)

    results_path = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results_path, results_rows)
    ctx.register_artifact(results_path, "results", "Every example-depth raw logit-lens measurement.")

    event_path = ctx.path("tables", "trajectory_events.csv")
    bench.write_csv_with_context(ctx, event_path, event_rows)
    ctx.register_artifact(event_path, "table", "Per-example event depths and final metrics.")

    audit_path = ctx.path("tables", "final_readout_audit.csv")
    bench.write_csv_with_context(ctx, audit_path, final_readout_audit_rows(event_rows))
    ctx.register_artifact(audit_path, "table", "Final-readout correctness, confidence, and target-vs-distractor audit.")

    transition_path = ctx.path("tables", "top1_transition_segments.csv")
    bench.write_csv_with_context(ctx, transition_path, transition_rows)
    ctx.register_artifact(transition_path, "table", "Compressed top-1 token segments over depth for each example.")

    phase_path = ctx.path("tables", "readout_phase_summary.csv")
    bench.write_csv_with_context(ctx, phase_path, phase_rows)
    ctx.register_artifact(phase_path, "table", "Coarse embedding/early/middle/late/final readout summaries per example.")

    # Backward-compatible alias for the current README and starter docs.
    example_summary_path = ctx.path("tables", "example_summary.csv")
    bench.write_csv_with_context(ctx, example_summary_path, event_rows)
    ctx.register_artifact(example_summary_path, "table", "Alias of trajectory_events.csv for Lab 1 summary use.")

    cat_rows = category_stats(event_rows, bundle.anatomy.n_layers)
    cat_path = ctx.path("tables", "category_summary.csv")
    bench.write_csv_with_context(ctx, cat_path, cat_rows)
    ctx.register_artifact(cat_path, "table", "Per-category aggregate metrics.")

    relation_rows = relation_stats(event_rows, bundle.anatomy.n_layers)
    relation_path = ctx.path("tables", "relation_summary.csv")
    bench.write_csv_with_context(ctx, relation_path, relation_rows)
    ctx.register_artifact(relation_path, "table", "Per-relation-family aggregate metrics; uses relation= note tags when present.")

    write_plot_reading_guide(ctx)

    metrics_path = ctx.path("metrics.json")
    bench.write_json(
        metrics_path,
        {
            "lab": LAB_ID,
            "evidence_level": "OBS",
            "n_examples": len(event_rows),
            "n_dropped_tokenization": dropped,
            "n_layers": bundle.anatomy.n_layers,
            "selected_prompt_set_sha256": prompt_set_sha256(examples),
            "kept_prompt_set_sha256": prompt_set_sha256([ex for ex, _, _ in kept]),
            "categories": cat_rows,
            "relation_families": relation_rows,
            "event_metric_definitions": {
                "decision_depth": "first depth after which final top-1 remains top-1",
                "target_first_top1": "first depth where labeled target is top-1",
                "target_first_beats_distractor": (
                    "first depth where target logit exceeds distractor by "
                    f">{LOGIT_DIFF_MEANINGFUL_MARGIN:g} and remains ahead thereafter"
                ),
                "target_first_beats_distractor_raw": (
                    "ungated first depth where target logit exceeds distractor; diagnostic only"
                ),
                "target_stable_beats_distractor": (
                    "first depth after which target logit keeps exceeding distractor"
                ),
                "target_rank_first_le_5": "first depth where target rank is 5 or better",
                "kl_to_final_first_le_0.5_bits": "first depth where KL(final || depth) <= 0.5 bits",
            },
        },
    )
    ctx.register_artifact(metrics_path, "metrics", "Machine-readable aggregate metrics and definitions.")

    # Plots.
    if not ctx.args.no_plots:
        plot_readout_dashboard(ctx, per_example_curves)
        plot_event_timeline(ctx, event_rows, bundle.anatomy.n_layers)
        plot_top1_transition_ribbons(ctx, transition_rows, event_rows, bundle.anatomy.n_layers)
        plot_phase_heatmap(ctx, phase_rows)
        plot_relation_event_matrix(ctx, event_rows, bundle.anatomy.n_layers)
        plot_event_ordering(ctx, event_rows, bundle.anatomy.n_layers)
        plot_convergence_lag(ctx, event_rows)
        plot_metric_by_depth(
            ctx,
            per_example_curves,
            "p_target",
            name="p_target_by_depth.png",
            title="p(target) under the raw logit lens",
            ylabel="p(target)",
            categories=("fact", "counterfactual"),
        )
        plot_metric_by_depth(
            ctx,
            per_example_curves,
            "target_rank",
            name="target_rank_by_depth.png",
            title="Target rank under the raw logit lens",
            ylabel="target rank (lower is better)",
            categories=("fact", "counterfactual"),
            logy=True,
            invert_y=True,
        )
        plot_metric_by_depth(
            ctx,
            per_example_curves,
            "logit_diff",
            name="logit_diff_by_depth.png",
            title="logit(target) minus logit(distractor)",
            ylabel="logit difference",
            categories=("fact", "counterfactual"),
        )
        plot_metric_by_depth(
            ctx,
            per_example_curves,
            "entropy_bits",
            name="entropy_by_depth.png",
            title="Readout entropy by depth",
            ylabel="entropy (bits)",
        )
        plot_metric_by_depth(
            ctx,
            per_example_curves,
            "kl_to_final_bits",
            name="kl_to_final_by_depth.png",
            title="Distributional convergence to the final readout",
            ylabel="KL(final || depth), bits",
            logy=True,
        )
        plot_metric_by_depth(
            ctx,
            per_example_curves,
            "top1_margin",
            name="top1_margin_by_depth.png",
            title="Top-1 minus top-2 probability margin",
            ylabel="probability margin",
        )
        plot_metric_by_depth(
            ctx,
            per_example_curves,
            "cosine_to_final",
            name="cosine_to_final_by_depth.png",
            title="Residual cosine similarity to final stream",
            ylabel="cosine to final stream",
        )
        plot_metric_by_depth(
            ctx,
            per_example_curves,
            "resid_l2",
            name="residual_norm_by_depth.png",
            title="Residual stream L2 norm at the readout position",
            ylabel="L2 norm",
            logy=True,
        )
        plot_metric_by_depth(
            ctx,
            per_example_curves,
            "stream_delta_l2",
            name="residual_delta_norm_by_depth.png",
            title="Residual update norm from previous depth",
            ylabel="delta L2 norm",
            logy=True,
        )
        plot_event_depths(ctx, event_rows, bundle.anatomy.n_layers)
        plot_event_heatmap(ctx, event_rows, bundle.anatomy.n_layers)
        plot_final_readout_scatter(ctx, event_rows)
        if showcase is not None:
            if showcase[1].p_target is None:
                print(
                    f"[lab1] WARNING: showcase example {showcase[0].example_id!r} has no "
                    "single-token target, so no biography plot was produced."
                )
            plot_biography(ctx, bundle, showcase[0], showcase[1])
        elif ctx.args.showcase is not None:
            print(
                f"[lab1] WARNING: --showcase {ctx.args.showcase!r} did not match any kept "
                "example id; no biography plot was produced. See tables/prompt_manifest.csv."
            )

    # Summary and drafted claims.
    claims = draft_claims(ctx, bundle, cat_rows, event_rows)
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)
    write_logit_lens_card(ctx, bundle, cat_rows, event_rows, dropped, claims)
    summary_md = render_summary(ctx, bundle, event_rows, cat_rows, dropped, claims)
    summary_path = ctx.path("run_summary.md")
    bench.write_text(summary_path, summary_md)
    ctx.register_artifact(summary_path, "summary", "The seven standard questions answered with this run's numbers.")
    print(f"[lab1] wrote run_summary.md and {len(claims)} drafted ledger claims")
