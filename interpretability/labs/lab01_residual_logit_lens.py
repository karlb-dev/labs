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
  target-like under the lens.
* ``ambiguous``: prompts with no privileged single-token continuation. These
  are the negative control against overclaiming early commitment.
* ``counterfactual``: a local context overwrites a memorized fact. Target is
  the in-context answer; distractor is the memorized answer.
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
import pathlib
import statistics
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
        "example_id", "category", "final_outcome", "final_top1", "final_top1_prob",
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
    """Summarize trajectories in coarse depth bands for quick cross-model reading."""
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
    """Thin line per example, heavier mean line per category."""
    fig, ax = bench.new_figure()
    plotted = False
    for category in categories:
        rows = [r for r in per_example if r["category"] == category and metric in r and r[metric]]
        if not rows:
            continue
        plotted = True
        color = category_color(category)
        for r in rows:
            ax.plot(range(len(r[metric])), r[metric], color=color, alpha=0.25, linewidth=0.8)
        depth_count = min(len(r[metric]) for r in rows)
        mean = [statistics.fmean(float(r[metric][d]) for r in rows) for d in range(depth_count)]
        ax.plot(range(depth_count), mean, color=color, linewidth=2.5, label=f"{category} (n={len(rows)})")
    if not plotted:
        bench.close_figure(fig)
        return
    if logy:
        ax.set_yscale("log")
    if invert_y:
        ax.invert_yaxis()
    ax.set_xlabel("depth (0 = embeddings, k = after k blocks)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, name, title)


def plot_event_depths(
    ctx: bench.RunContext,
    event_rows: list[dict[str, Any]],
    n_layers: int,
) -> None:
    """Scatter event depths by category for quick outlier spotting."""
    fig, ax = bench.new_figure(figsize=(9.0, 5.5))
    events = [
        "decision_depth",
        "target_first_top1",
        "target_first_beats_distractor",
        "target_first_beats_distractor_raw",
        "target_rank_first_le_5",
        "kl_to_final_first_le_0.5_bits",
    ]
    x_positions = {event: i for i, event in enumerate(events)}
    jitter_offsets = {cat: (i - 1.5) * 0.06 for i, cat in enumerate(CATEGORIES)}
    any_points = False
    for row in event_rows:
        cat = row["category"]
        for event in events:
            value = row.get(event)
            if value in (None, ""):
                continue
            any_points = True
            ax.scatter(
                x_positions[event] + jitter_offsets.get(cat, 0.0),
                float(value),
                s=24,
                alpha=0.75,
                color=category_color(cat),
                label=cat,
            )
    if not any_points:
        bench.close_figure(fig)
        return
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8)
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(events, rotation=25, ha="right")
    ax.set_ylim(-0.5, n_layers + 0.5)
    ax.set_ylabel("depth")
    ax.set_title("Event depths by example")
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

    events = [
        "decision_depth",
        "target_first_top1",
        "target_stable_top1_depth",
        "target_first_beats_distractor",
        "target_first_beats_distractor_raw",
        "target_rank_first_le_5",
        "kl_to_final_first_le_0.5_bits",
    ]
    rows = sorted(
        event_rows,
        key=lambda r: (
            CATEGORIES.index(r["category"]) if r["category"] in CATEGORIES else len(CATEGORIES),
            float(r["decision_depth"]) if r.get("decision_depth") not in (None, "") else float(n_layers + 1),
            r["example_id"],
        ),
    )
    data = [[float("nan") for _ in events] for _ in rows]
    for i, row in enumerate(rows):
        for j, event in enumerate(events):
            value = row.get(event)
            if value not in (None, ""):
                data[i][j] = float(value)

    fig_height = max(6.0, min(12.0, 0.28 * len(rows) + 1.8))
    fig, ax = bench.new_figure(figsize=(10.0, fig_height))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e6e6e6")
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=n_layers)
    ax.set_xticks(range(len(events)))
    ax.set_xticklabels(events, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['category'][:2]}:{r['example_id']}" for r in rows], fontsize=7)
    ax.set_title("Event-depth heatmap (gray = event absent or undefined)")
    ax.set_xlabel("event metric")
    ax.set_ylabel("example")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("depth")

    previous = rows[0]["category"]
    for i, row in enumerate(rows[1:], start=1):
        if row["category"] != previous:
            ax.axhline(i - 0.5, color="white", linewidth=1.5)
            previous = row["category"]
    fig.tight_layout()
    bench.save_figure(
        ctx,
        fig,
        "event_depth_heatmap.png",
        "Example-by-event depth grid; gray cells make non-occurring events explicit.",
    )


def plot_final_readout_scatter(
    ctx: bench.RunContext,
    event_rows: list[dict[str, Any]],
) -> None:
    """Final confidence vs entropy, separating target success from confidence."""
    if not event_rows:
        return
    fig, ax = bench.new_figure(figsize=(8.0, 5.8))
    seen: set[str] = set()
    for row in event_rows:
        entropy = row.get("final_entropy_bits")
        prob = row.get("final_top1_prob")
        if entropy in (None, "") or prob in (None, ""):
            continue
        cat = row["category"]
        target_status = row.get("final_top1_is_target")
        if target_status in (None, ""):
            marker = "s"
            label = f"{cat}: unlabeled"
        elif bool(target_status):
            marker = "o"
            label = f"{cat}: target top-1"
        else:
            marker = "X"
            label = f"{cat}: target not top-1"
        ax.scatter(
            float(entropy),
            float(prob),
            color=category_color(cat),
            marker=marker,
            s=70,
            alpha=0.82,
            edgecolors="black" if target_status not in (None, "") else "none",
            linewidths=0.7,
            label=label if label not in seen else None,
        )
        seen.add(label)
        if target_status is False and cat in {"fact", "counterfactual"}:
            ax.annotate(
                row["example_id"].replace("fact_", "").replace("cf_", ""),
                (float(entropy), float(prob)),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=7,
            )
    ax.set_xlabel("final entropy (bits)")
    ax.set_ylabel("final top-1 probability")
    ax.set_title("Final readout: confidence is not the same as target success")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    bench.save_figure(
        ctx,
        fig,
        "final_readout_scatter.png",
        "Final entropy vs top-1 probability, with labeled target success marked separately.",
    )


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
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    ax = axes[0, 0]
    ax.plot(depths, traj.p_target, linewidth=2.2, label=f"p(target = {bench.visible_token(example.target or '')})")
    if traj.p_distractor is not None:
        ax.plot(depths, traj.p_distractor, linewidth=2.2, label=f"p(distractor = {bench.visible_token(example.distractor or '')})")
    ax.plot(depths, traj.top1_probs, linewidth=1.0, linestyle="--", label="p(top-1 at depth)")
    ax.set_ylabel("probability")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    if traj.logit_target is not None and traj.logit_distractor is not None:
        diffs = [t - d for t, d in zip(traj.logit_target, traj.logit_distractor)]
        ax.axhline(0.0, linewidth=0.8)
        ax.axhline(LOGIT_DIFF_MEANINGFUL_MARGIN, linewidth=0.8, linestyle=":")
        ax.plot(depths, diffs, linewidth=2.2)
        ax.set_ylabel("logit(target) - logit(distractor)")
    else:
        ax.text(0.5, 0.5, "No target/distractor pair", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Matched target-vs-distractor readout")

    ax = axes[1, 0]
    if traj.target_rank is not None:
        ax.plot(depths, traj.target_rank, linewidth=2.2, label="target rank")
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
    ax.set_yscale("log")
    ax.set_xlabel("depth")
    ax.set_ylabel("bits, log scale")
    ax.set_title("Sharpness and convergence are different")
    ax.legend(fontsize=8)

    step = max(1, traj.n_depths // 8)
    annotated = sorted(set(list(range(0, traj.n_depths, step)) + [traj.n_depths - 1]))
    for depth in annotated:
        axes[0, 0].annotate(
            bench.visible_token(traj.top1_texts[depth]),
            (depth, traj.top1_probs[depth]),
            textcoords="offset points",
            xytext=(0, 8),
            fontsize=7,
            rotation=45,
        )
    fig.suptitle(f"Prediction biography: {example.example_id}\n{example.prompt}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    bench.save_figure(
        ctx,
        fig,
        f"biography_{bench.sanitize_tag(example.example_id)}.png",
        "Showcase example: target/distractor probability, logit difference, rank, entropy, and KL over depth.",
    )


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
        ("entropy_bits", "entropy", "bits", False),
        ("kl_to_final_bits", "KL(final || depth)", "bits", True),
        ("top1_margin", "top-1 minus top-2 margin", "probability", False),
        ("cosine_to_final", "cosine to final residual", "cosine", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    plotted = False
    for ax, (metric, title, ylabel, logy) in zip(axes.flat, specs):
        ax.grid(True, alpha=0.3)
        for category in categories:
            rows = [r for r in per_example if r["category"] == category and metric in r and r[metric]]
            if not rows:
                continue
            plotted = True
            depth_count = min(len(r[metric]) for r in rows)
            mean = [statistics.fmean(float(r[metric][d]) for r in rows) for d in range(depth_count)]
            ax.plot(range(depth_count), mean, linewidth=2.2, label=f"{category} (n={len(rows)})", color=category_color(category))
        if logy:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("depth")
        ax.set_ylabel(ylabel)
    if not plotted:
        bench.close_figure(fig)
        return
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Raw logit-lens readout dashboard", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    bench.save_figure(
        ctx,
        fig,
        "readout_dashboard.png",
        "Category-mean entropy, KL-to-final, top-1 margin, and residual cosine curves in one dashboard.",
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
        "kl_to_final_first_le_0.5_bits",
    ]
    fig, ax = bench.new_figure(figsize=(10.5, 5.8))
    offsets = {cat: (i - 1.5) * 0.12 for i, cat in enumerate(CATEGORIES)}
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
            ax.scatter(j + offsets.get(category, 0.0), y, s=70, color=category_color(category), label=category)
            ax.annotate(f"n={len(vals)}", (j + offsets.get(category, 0.0), y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
    if not plotted:
        bench.close_figure(fig)
        return
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8)
    ax.set_xticks(range(len(events)))
    ax.set_xticklabels(events, rotation=25, ha="right")
    ax.set_ylim(-0.04, 1.04)
    ax.set_ylabel("median event depth / number of blocks")
    ax.set_title("Event ordering by category (conditional on event occurring)")
    bench.save_figure(ctx, fig, "event_ordering.png", "Median event-depth fraction by category, annotated with occurrence counts.")


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
        }
        out.append(row)
    return out


def render_category_table(cat_rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| category | n | mean prompt tokens | median decision depth | frac of L | target first top-1 | target beats distractor (>1, stable lead) | final entropy | final p(target) | target final top-1 rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in cat_rows:
        lines.append(
            f"| {r['category']} | {r['n_examples']} | {r['mean_prompt_tokens']} | "
            f"{r['median_decision_depth']} | {r['median_decision_depth_frac']} | "
            f"{r['median_target_first_top1']} (n={r['n_target_first_top1']}) | "
            f"{r['median_target_first_beats_distractor']} (n={r['n_target_first_beats_distractor']}) | "
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
        "The pre-final-norm residual stream at the final token position after every block. Each stream was decoded with the model's own final norm and unembedding. The run recorded top-k tokens, target and distractor metrics, entropy, KL-to-final, top-1 margin, residual norm, update norm, and residual cosine-to-final.",
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
        "Blank cells mean the metric is not defined for that category or never occurred.",
        "",
        "## 5. What claim is supported, and at what evidence level?",
        "",
        "Only observational claims are supported. The strongest claims are about what the raw final readout can decode from intermediate streams under this prompt distribution.",
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
        "1. `logit_lens_card.md` for scope, headline numbers, claims, and non-claims",
        "2. `diagnostics/logit_lens_self_check.json` and `diagnostics/hook_parity_by_layer.csv`",
        "3. `tables/final_readout_audit.csv` to separate correctness, confidence, and matched-distractor wins",
        "4. `state/<example_id>/state_card.md` for one fact and one counterfactual prompt",
        "5. `plots/readout_dashboard.png`, `plots/final_readout_scatter.png`, and `plots/event_depth_heatmap.png`",
        "6. `plots/target_rank_by_depth.png`, `plots/logit_diff_by_depth.png`, and `plots/kl_to_final_by_depth.png`",
        "7. `tables/top1_transition_segments.csv` and `tables/trajectory_events.csv` for outliers and blank cells",
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
            "non_claim": "None of these events proves knowledge, belief, storage, or causal use. They are readout events.",
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
        "5. `plots/event_ordering.png`",
        "6. `state/<example_id>/state_card.md`",
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
        plot_event_ordering(ctx, event_rows, bundle.anatomy.n_layers)
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
