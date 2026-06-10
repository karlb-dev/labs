"""Lab 1: Residual stream and logit lens.

Core question: how does a model's running prediction emerge across layers?

This module owns the *experiment*: the prompt families, target validation,
the per-example measurement loop, aggregation, plots, and the run summary.
The instrument -- model loading, residual capture, the lens math, state
dumps, self-checks -- lives in ``interp_bench.py`` and is shared by every
lab. If you want to know what a number in results.csv means, the chain is:

    interp_bench.run_with_residual_cache   (what was captured, exactly)
    interp_bench.compute_lens_trajectory   (how each metric is computed)
    this file                               (which prompts, what's aggregated)

Evidence level targeted: OBSERVATION. Nothing here is causal. The lens shows
what the unembedding would decode from an intermediate stream; it does not
show that the model "knows" or "uses" anything at that depth. The handout
(lab01_residual_logit_lens.md) carries the full set of caveats.

Design notes
============

Three prompt families create the contrast the lab is about:

* ``fact``: high-certainty completions ("The capital of France is"). The
  interesting measurement is *when* the answer becomes top-1 and stays.
* ``ambiguous``: prompts with no privileged continuation. The expectation is
  late, weak commitment and high entropy throughout -- the control that keeps
  "the model decides early" claims honest.
* ``counterfactual``: a context that overrides a memorized fact ("In this
  story, the capital of France is London. ... The capital of France is").
  Target = the in-context answer, distractor = the memorized fact, so the
  logit-difference trajectory shows context and memory competing over depth.

Targets and distractors must be single tokens for the run's tokenizer.
Examples that fail validation are *dropped with a count* and logged to
diagnostics/tokenization_report.csv -- never silently patched, because
half the bugs in published lens plots are tokenization bugs.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import statistics
from typing import Any

import interp_bench as bench

LAB_ID = "L01"


# ---------------------------------------------------------------------------
# Prompt families
# ---------------------------------------------------------------------------
#
# Targets/distractors are written with their leading space because that is
# how the token actually appears after a word boundary. Validation below
# confirms single-token status per tokenizer and drops failures with a count.

@dataclasses.dataclass(frozen=True)
class PromptExample:
    example_id: str
    category: str  # fact | ambiguous | counterfactual
    prompt: str
    target: str | None = None      # expected continuation (single token)
    distractor: str | None = None  # plausible-but-wrong contrast token
    note: str = ""


FACT_EXAMPLES = (
    PromptExample("fact_capital_france", "fact", "The capital of France is", " Paris", " London"),
    PromptExample("fact_capital_japan", "fact", "The capital of Japan is", " Tokyo", " Kyoto"),
    PromptExample("fact_capital_italy", "fact", "The capital of Italy is", " Rome", " Milan"),
    PromptExample("fact_eiffel_city", "fact", "The Eiffel Tower is in the city of", " Paris", " Rome"),
    PromptExample("fact_opposite_hot", "fact", "The opposite of hot is", " cold", " warm"),
    PromptExample("fact_opposite_up", "fact", "The opposite of up is", " down", " left"),
    PromptExample("fact_water_h2o", "fact", "Water is made of hydrogen and", " oxygen", " carbon"),
    PromptExample("fact_two_plus_two", "fact", "Two plus two equals", " four", " five"),
    PromptExample("fact_sky_color", "fact", "On a clear day, the color of the sky is", " blue", " green"),
    PromptExample("fact_week_days", "fact", "The day after Monday is", " Tuesday", " Sunday"),
)

AMBIGUOUS_EXAMPLES = (
    PromptExample("ambig_solve_problem", "ambiguous", "The best way to solve the problem is"),
    PromptExample("ambig_went_to_the", "ambiguous", "Yesterday afternoon I went to the"),
    PromptExample("ambig_important_thing", "ambiguous", "The most important thing in life is"),
    PromptExample("ambig_opened_door", "ambiguous", "She opened the door and saw"),
    PromptExample("ambig_meeting_about", "ambiguous", "The meeting tomorrow will be about"),
    PromptExample("ambig_favorite", "ambiguous", "My favorite thing about this city is the"),
)

COUNTERFACTUAL_EXAMPLES = (
    PromptExample(
        "cf_capital_france_london",
        "counterfactual",
        "In this story, the capital of France is London. "
        "According to the story, the capital of France is",
        " London",
        " Paris",
        note="context answer vs memorized fact",
    ),
    PromptExample(
        "cf_capital_japan_osaka",
        "counterfactual",
        "In this story, the capital of Japan is Osaka. "
        "According to the story, the capital of Japan is",
        " Osaka",
        " Tokyo",
    ),
    PromptExample(
        "cf_sky_green",
        "counterfactual",
        "Professor Hale insists that the sky is green. "
        "According to Professor Hale, the sky is",
        " green",
        " blue",
    ),
    PromptExample(
        "cf_opposite_hot_wet",
        "counterfactual",
        "In this puzzle, the opposite of hot is wet. "
        "In this puzzle, the opposite of hot is",
        " wet",
        " cold",
    ),
    PromptExample(
        "cf_two_plus_two_five",
        "counterfactual",
        "In this game, two plus two equals five. "
        "In this game, two plus two equals",
        " five",
        " four",
    ),
    PromptExample(
        "cf_paris_person",
        "counterfactual",
        "In this story, Paris is a person and London is a dog. "
        "In this story, Paris is a",
        " person",
        " city",
    ),
)

# The small set is the default: enough examples per family to see a pattern,
# small enough that a full run with state dumps stays under a few minutes on
# Tier B and under ten on CPU with gpt2.
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

ALL_EXAMPLES = FACT_EXAMPLES + AMBIGUOUS_EXAMPLES + COUNTERFACTUAL_EXAMPLES


def build_prompt_set(args: Any) -> list[PromptExample]:
    """Resolve --prompt-set into a list of examples.

    ``small``/``full`` are the built-in sets; anything else is treated as a
    path to a JSON file: a list of objects with the PromptExample fields.
    """
    if args.prompt_set == "full":
        examples = list(ALL_EXAMPLES)
    elif args.prompt_set == "small":
        examples = [e for e in ALL_EXAMPLES if e.example_id in SMALL_SET_IDS]
    else:
        path = pathlib.Path(args.prompt_set)
        raw = json.loads(path.read_text(encoding="utf-8"))
        examples = [PromptExample(**item) for item in raw]
    # Interleave categories round-robin so a --max-examples cap (the Tier A
    # smoke path caps at 4) still exercises every prompt family.
    by_category: dict[str, list[PromptExample]] = {}
    for ex in examples:
        by_category.setdefault(ex.category, []).append(ex)
    interleaved: list[PromptExample] = []
    queues = list(by_category.values())
    while any(queues):
        for queue in queues:
            if queue:
                interleaved.append(queue.pop(0))
    examples = interleaved
    if args.max_examples:
        examples = examples[: args.max_examples]
    return examples


# ---------------------------------------------------------------------------
# Target validation
# ---------------------------------------------------------------------------


def single_token_id(tokenizer: Any, text: str) -> int | None:
    """Return the token id if ``text`` encodes to exactly one token."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def validate_examples(
    ctx: bench.RunContext, bundle: bench.ModelBundle, examples: list[PromptExample]
) -> tuple[list[tuple[PromptExample, int | None, int | None]], int]:
    """Check that targets/distractors are single tokens for this tokenizer.

    Returns (kept examples with resolved token ids, dropped count). Every
    decision is written to diagnostics/tokenization_report.csv so a dropped
    example is a visible fact about the tokenizer, not a silent edit to the
    experiment.
    """
    kept: list[tuple[PromptExample, int | None, int | None]] = []
    report_rows: list[dict[str, Any]] = []
    dropped = 0
    for ex in examples:
        target_id = single_token_id(bundle.tokenizer, ex.target) if ex.target else None
        distractor_id = single_token_id(bundle.tokenizer, ex.distractor) if ex.distractor else None
        target_ok = ex.target is None or target_id is not None
        distractor_ok = ex.distractor is None or distractor_id is not None
        status = "kept" if (target_ok and distractor_ok) else "dropped"
        if status == "dropped":
            dropped += 1
        else:
            kept.append((ex, target_id, distractor_id))
        report_rows.append(
            {
                "example_id": ex.example_id,
                "category": ex.category,
                "target": bench.visible_token(ex.target) if ex.target else "",
                "target_n_tokens": (
                    len(bundle.tokenizer.encode(ex.target, add_special_tokens=False)) if ex.target else ""
                ),
                "distractor": bench.visible_token(ex.distractor) if ex.distractor else "",
                "distractor_n_tokens": (
                    len(bundle.tokenizer.encode(ex.distractor, add_special_tokens=False))
                    if ex.distractor
                    else ""
                ),
                "status": status,
            }
        )
    path = ctx.path("diagnostics", "tokenization_report.csv")
    bench.write_csv(path, report_rows)
    ctx.register_artifact(path, "diagnostic", "Single-token validation for every target/distractor.")
    if dropped:
        print(f"[lab1] dropped {dropped} example(s) with multi-token answers (see tokenization_report.csv)")
    return kept, dropped


# ---------------------------------------------------------------------------
# Per-example metrics
# ---------------------------------------------------------------------------


def decision_depth(traj: bench.LensTrajectory) -> int:
    """Smallest depth k where the top-1 token equals the FINAL top-1 token at
    every depth >= k. This is 'when the model's eventual answer locks in',
    defined for every example whether or not it has a labeled target."""
    final_id = traj.top1_ids[-1]
    k = traj.n_depths - 1
    for depth in range(traj.n_depths - 2, -1, -1):
        if traj.top1_ids[depth] == final_id:
            k = depth
        else:
            break
    return k


def target_first_top1(traj: bench.LensTrajectory, target_id: int) -> int | None:
    """First depth at which the labeled target is the top-1 readout."""
    for depth, top1 in enumerate(traj.top1_ids):
        if top1 == target_id:
            return depth
    return None


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_metric_by_depth(
    ctx: bench.RunContext,
    per_example: list[dict[str, Any]],
    metric: str,
    *,
    name: str,
    title: str,
    ylabel: str,
    categories: tuple[str, ...] = ("fact", "ambiguous", "counterfactual"),
    logy: bool = False,
) -> None:
    """One line per example (thin), one bold mean line per category."""
    fig, ax = bench.new_figure()
    for category in categories:
        rows = [r for r in per_example if r["category"] == category and r.get(metric)]
        if not rows:
            continue
        color = bench.CATEGORY_COLORS[category]
        for r in rows:
            ax.plot(range(len(r[metric])), r[metric], color=color, alpha=0.25, linewidth=0.8)
        depth_count = min(len(r[metric]) for r in rows)
        mean = [
            statistics.fmean(r[metric][d] for r in rows) for d in range(depth_count)
        ]
        ax.plot(range(depth_count), mean, color=color, linewidth=2.5, label=f"{category} (n={len(rows)})")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("depth (0 = embeddings, k = after k blocks)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    bench.save_figure(ctx, fig, name, title)


def plot_biography(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    example: PromptExample,
    traj: bench.LensTrajectory,
) -> None:
    """The showcase 'prediction biography': p(target) and p(distractor) over
    depth with the top-1 token annotated at intervals -- the model appearing
    to make up its mind, as one readable picture."""
    fig, ax = bench.new_figure(figsize=(10.0, 5.5))
    depths = list(range(traj.n_depths))
    ax.plot(depths, traj.p_target, color="#2ca02c", linewidth=2.5,
            label=f"p(target = {bench.visible_token(example.target)})")
    if traj.p_distractor is not None:
        ax.plot(depths, traj.p_distractor, color="#d62728", linewidth=2.5,
                label=f"p(distractor = {bench.visible_token(example.distractor)})")
    ax.plot(depths, traj.top1_probs, color="#7f7f7f", linewidth=1.0, linestyle="--",
            label="p(top-1 at that depth)")

    # Annotate the top-1 token at evenly spaced depths so the picture reads
    # as a story, not just curves.
    step = max(1, traj.n_depths // 8)
    for depth in list(range(0, traj.n_depths, step)) + [traj.n_depths - 1]:
        ax.annotate(
            bench.visible_token(traj.top1_texts[depth]),
            (depth, traj.top1_probs[depth]),
            textcoords="offset points",
            xytext=(0, 8),
            fontsize=7,
            rotation=45,
            color="#444444",
        )
    ax.set_xlabel("depth (0 = embeddings, k = after k blocks)")
    ax.set_ylabel("probability under logit lens")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"Prediction biography: {example.example_id}\n\"{example.prompt}\"")
    ax.legend(loc="upper left", fontsize=8)
    bench.save_figure(
        ctx, fig, f"biography_{bench.sanitize_tag(example.example_id)}.png",
        "Showcase example: target vs distractor probability over depth.",
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def category_stats(summary_rows: list[dict[str, Any]], n_layers: int) -> list[dict[str, Any]]:
    """Aggregate per-example summaries into the per-category table."""
    out = []
    for category in ("fact", "ambiguous", "counterfactual"):
        rows = [r for r in summary_rows if r["category"] == category]
        if not rows:
            continue
        decisions = [r["decision_depth"] for r in rows]
        out.append(
            {
                "category": category,
                "n_examples": len(rows),
                "median_decision_depth": statistics.median(decisions),
                "median_decision_depth_frac": round(statistics.median(decisions) / n_layers, 3),
                "mean_final_entropy_bits": round(
                    statistics.fmean(r["final_entropy_bits"] for r in rows), 3
                ),
                "mean_final_top1_prob": round(
                    statistics.fmean(r["final_top1_prob"] for r in rows), 3
                ),
            }
        )
    return out


def render_summary(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    summary_rows: list[dict[str, Any]],
    cat_rows: list[dict[str, Any]],
    dropped: int,
    claims: list[dict[str, str]],
) -> str:
    """run_summary.md: the standard seven questions, answered with numbers."""
    a = bundle.anatomy
    L = a.n_layers

    def cat(name: str) -> dict[str, Any] | None:
        return next((r for r in cat_rows if r["category"] == name), None)

    fact, ambig, cf = cat("fact"), cat("ambiguous"), cat("counterfactual")
    lines = [
        "# Lab 1 run summary: residual stream and logit lens",
        "",
        f"- model: `{a.model_id}` ({L} blocks) | device: {bundle.device} | dtype: {ctx.args.dtype}"
        f" | quantization: {ctx.args.quantization}",
        f"- examples: {len(summary_rows)} kept, {dropped} dropped at tokenization (see diagnostics/)",
        f"- self-checks: hook parity and lens-at-final-depth both verified this run (see diagnostics/)",
        "",
        "## 1. What behavior was studied?",
        "",
        "Next-token prediction on three prompt families: high-certainty facts,",
        "ambiguous continuations, and counterfactual contexts that override a",
        "memorized fact.",
        "",
        "## 2. What internal object was measured?",
        "",
        "The pre-norm residual stream at the final token position after each",
        "block, decoded through the model's own final norm + unembedding",
        "(logit lens). Per depth: top-k tokens, p(target), p(distractor),",
        "entropy, cosine to the final stream, and residual norm.",
        "",
        "## 3. What intervention or control was used?",
        "",
        "No intervention (this lab is observational). The control is the",
        "ambiguous prompt family: any claim of the form 'the model commits at",
        "depth k' must look different there, or it is an artifact of the",
        "readout rather than a fact about certainty.",
        "",
        "## 4. What metric changed (the headline numbers)?",
        "",
        "| category | n | median decision depth | as fraction of L | mean final entropy (bits) | mean final p(top1) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in cat_rows:
        lines.append(
            f"| {r['category']} | {r['n_examples']} | {r['median_decision_depth']} "
            f"| {r['median_decision_depth_frac']} | {r['mean_final_entropy_bits']} "
            f"| {r['mean_final_top1_prob']} |"
        )
    lines += [
        "",
        "Decision depth = smallest depth k at which the final top-1 token is",
        "top-1 at every depth >= k.",
        "",
        "## 5. What claim is supported, at what evidence level?",
        "",
        "OBSERVATION only. See ledger_suggestions.md for drafted claims with",
        "these numbers filled in; edit them before they enter your ledger.",
        "",
        "## 6. What claim is NOT supported?",
        "",
        "- That the model 'knows the answer' at the decision depth: the lens",
        "  is a readout through the final-layer basis, not a measurement of",
        "  what later layers use.",
        "- That early stabilization implies the late layers are inert: they",
        "  may be doing work invisible to this projection.",
        "- Any causal claim. Patching (Lab 5) is where those become possible.",
        "",
        "## 7. What would falsify the interpretation?",
        "",
        "- A tuned lens (Lab 1 ambitious extension) disagreeing sharply at",
        "  the depths where this run claims stabilization.",
        "- The same analysis on a permuted/random prompt family showing",
        "  identical 'decision depths' (would indicate a readout artifact).",
        "",
        "## Per-example summaries",
        "",
        "| example | category | decision depth | target first top-1 | final p(target) | final entropy |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        first_top1 = r.get("target_first_top1", "")
        if first_top1 is None:
            first_top1 = "never"  # target validated but never top-1 at any depth
        lines.append(
            f"| {r['example_id']} | {r['category']} | {r['decision_depth']} "
            f"| {first_top1} | {r.get('final_p_target', '')} "
            f"| {r['final_entropy_bits']} |"
        )
    lines += [
        "",
        "## Where to look next",
        "",
        "- `state/<example_id>/state_card.md` -- the per-example narrative dump",
        "- `plots/` -- trajectories by category",
        "- `results.csv` -- every (example, depth) measurement",
        "- `diagnostics/` -- proof the instrument worked before you believe any of the above",
        "",
        "## Tooling note (student fills in)",
        "",
        "_What was AI-drafted, what was hand-verified, and one thing the_",
        "_assistant got wrong or overclaimed that you caught._",
        "",
    ]
    return "\n".join(lines)


def draft_claims(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    cat_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Draft OBS-tagged ledger claims with measured numbers filled in."""
    L = bundle.anatomy.n_layers
    run_name = ctx.run_dir.name
    claims: list[dict[str, str]] = []

    fact = next((r for r in cat_rows if r["category"] == "fact"), None)
    ambig = next((r for r in cat_rows if r["category"] == "ambiguous"), None)
    if fact:
        claims.append(
            {
                "id": f"{LAB_ID}-C1",
                "tag": "OBS",
                "text": (
                    f"On {fact['n_examples']} high-certainty factual prompts, "
                    f"{bundle.anatomy.model_id}'s final answer becomes and stays "
                    f"top-1 under the logit lens at median depth "
                    f"{fact['median_decision_depth']}/{L} "
                    f"({fact['median_decision_depth_frac']:.0%} of depth)."
                ),
                "artifact": f"runs/{run_name}/tables/category_summary.csv",
                "falsifier": (
                    "A tuned lens placing stabilization materially later, or the "
                    "pattern failing on a fresh fact family."
                ),
            }
        )
    if fact and ambig:
        claims.append(
            {
                "id": f"{LAB_ID}-C2",
                "tag": "OBS",
                "text": (
                    f"Ambiguous prompts behave differently from facts on the same "
                    f"model: mean final entropy {ambig['mean_final_entropy_bits']} "
                    f"bits vs {fact['mean_final_entropy_bits']} bits, and median "
                    f"decision depth {ambig['median_decision_depth']}/{L} vs "
                    f"{fact['median_decision_depth']}/{L} -- the lens trajectory "
                    f"tracks task certainty, not just depth."
                ),
                "artifact": f"runs/{run_name}/plots/entropy_by_depth.png",
                "falsifier": (
                    "Matched-length ambiguous prompts showing the same trajectory "
                    "as facts (would mean the metric reflects prompt surface, not "
                    "certainty)."
                ),
            }
        )
    cf_rows = [r for r in summary_rows if r["category"] == "counterfactual" and r.get("final_p_target")]
    if cf_rows:
        won = sum(1 for r in cf_rows if r["final_top1_is_target"])
        claims.append(
            {
                "id": f"{LAB_ID}-C3",
                "tag": "OBS",
                "text": (
                    f"In {won}/{len(cf_rows)} counterfactual-context prompts the "
                    f"in-context answer beats the memorized fact at the final "
                    f"depth; the lens trajectories show the two competing over "
                    f"depth rather than the context winning from depth 0."
                ),
                "artifact": f"runs/{run_name}/plots/logit_diff_by_depth.png",
                "falsifier": (
                    "Counterfactual trajectories indistinguishable from fact "
                    "trajectories once prompts are length-matched."
                ),
            }
        )
    return claims


# ---------------------------------------------------------------------------
# Entry point called by the bench
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    examples = build_prompt_set(ctx.args)
    kept, dropped = validate_examples(ctx, bundle, examples)
    if not kept:
        raise RuntimeError("No examples survived tokenization validation.")
    print(f"[lab1] running {len(kept)} examples ({dropped} dropped)")

    # Instrument self-checks run once, on the first kept prompt, before any
    # science: hooks vs hidden_states, then lens(L) vs the model's logits.
    first_prompt = kept[0][0].prompt
    bench.run_hook_parity_check(ctx, bundle, first_prompt)
    first_capture = bench.run_with_residual_cache(bundle, first_prompt)
    bench.run_lens_self_check(ctx, bundle, first_capture)

    results_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    per_example_curves: list[dict[str, Any]] = []
    showcase: tuple[PromptExample, bench.LensTrajectory] | None = None

    for i, (ex, target_id, distractor_id) in enumerate(kept, start=1):
        capture = bench.run_with_residual_cache(bundle, ex.prompt)
        traj = bench.compute_lens_trajectory(
            bundle, capture, target_id=target_id, distractor_id=distractor_id, topk=ctx.args.topk
        )
        bench.dump_example_state(
            ctx, bundle, ex.example_id, capture, traj, target=ex.target, distractor=ex.distractor
        )

        # Long-form rows: one per (example, depth) -- the lab's raw output.
        for depth in range(traj.n_depths):
            row: dict[str, Any] = {
                "example_id": ex.example_id,
                "category": ex.category,
                "depth": depth,
                "top1_token": traj.top1_texts[depth],
                "top1_prob": round(traj.top1_probs[depth], 6),
                "entropy_bits": round(traj.entropy_bits[depth], 4),
                "cosine_to_final": round(traj.cosine_to_final[depth], 5),
                "resid_l2": round(traj.resid_l2[depth], 3),
            }
            if traj.p_target is not None:
                row["p_target"] = round(traj.p_target[depth], 6)
            if traj.p_distractor is not None:
                row["p_distractor"] = round(traj.p_distractor[depth], 6)
            if traj.logit_target is not None and traj.logit_distractor is not None:
                row["logit_diff"] = round(
                    traj.logit_target[depth] - traj.logit_distractor[depth], 4
                )
            results_rows.append(row)

        # Per-example summary metrics.
        summary: dict[str, Any] = {
            "example_id": ex.example_id,
            "category": ex.category,
            "n_prompt_tokens": len(capture.input_ids),
            "decision_depth": decision_depth(traj),
            "final_top1": traj.top1_texts[-1],
            "final_top1_prob": round(traj.top1_probs[-1], 4),
            "final_entropy_bits": round(traj.entropy_bits[-1], 3),
        }
        if target_id is not None:
            summary["target_first_top1"] = target_first_top1(traj, target_id)
            summary["final_p_target"] = round(traj.p_target[-1], 4)
            summary["final_top1_is_target"] = traj.top1_ids[-1] == target_id
        summary_rows.append(summary)

        # Curves for the category plots.
        curve: dict[str, Any] = {
            "category": ex.category,
            "entropy_bits": traj.entropy_bits,
            "cosine_to_final": traj.cosine_to_final,
            "resid_l2": traj.resid_l2,
        }
        if traj.p_target is not None:
            curve["p_target"] = traj.p_target
        if traj.logit_target is not None and traj.logit_distractor is not None:
            curve["logit_diff"] = [
                t - d for t, d in zip(traj.logit_target, traj.logit_distractor)
            ]
        per_example_curves.append(curve)

        if showcase is None and (
            ex.example_id == ctx.args.showcase
            or (ctx.args.showcase is None and ex.category == "fact" and traj.p_target is not None)
        ):
            showcase = (ex, traj)

        detail = ""
        if "final_p_target" in summary:
            detail = f" p_target(final)={summary['final_p_target']:.2f}"
        print(
            f"[lab1] [{i}/{len(kept)}] {ex.example_id} "
            f"decision_depth={summary['decision_depth']}/{bundle.anatomy.n_layers}{detail}"
        )

    # ---- aggregate artifacts -------------------------------------------------
    results_path = ctx.path("results.csv")
    bench.write_csv(results_path, results_rows)
    ctx.register_artifact(results_path, "results", "Every (example, depth) lens measurement.")

    summary_path = ctx.path("tables", "example_summary.csv")
    bench.write_csv(summary_path, summary_rows)
    ctx.register_artifact(summary_path, "table", "Per-example decision depths and final readouts.")

    cat_rows = category_stats(summary_rows, bundle.anatomy.n_layers)
    cat_path = ctx.path("tables", "category_summary.csv")
    bench.write_csv(cat_path, cat_rows)
    ctx.register_artifact(cat_path, "table", "Per-category aggregates (the headline table).")

    bench.write_json(
        ctx.path("metrics.json"),
        {
            "lab": LAB_ID,
            "n_examples": len(summary_rows),
            "n_dropped_tokenization": dropped,
            "n_layers": bundle.anatomy.n_layers,
            "categories": cat_rows,
        },
    )

    # ---- plots ---------------------------------------------------------------
    if not ctx.args.no_plots:
        plot_metric_by_depth(
            ctx, per_example_curves, "p_target",
            name="p_target_by_depth.png",
            title="p(target) under the logit lens",
            ylabel="p(target)",
            categories=("fact", "counterfactual"),
        )
        plot_metric_by_depth(
            ctx, per_example_curves, "logit_diff",
            name="logit_diff_by_depth.png",
            title="logit(target) - logit(distractor) over depth",
            ylabel="logit difference",
            categories=("fact", "counterfactual"),
        )
        plot_metric_by_depth(
            ctx, per_example_curves, "entropy_bits",
            name="entropy_by_depth.png",
            title="Readout entropy over depth (the certainty trajectory)",
            ylabel="entropy (bits)",
        )
        plot_metric_by_depth(
            ctx, per_example_curves, "cosine_to_final",
            name="cosine_to_final_by_depth.png",
            title="Cosine similarity of each depth's stream to the final stream",
            ylabel="cosine similarity",
        )
        plot_metric_by_depth(
            ctx, per_example_curves, "resid_l2",
            name="residual_norm_by_depth.png",
            title="Residual stream L2 norm over depth (final position)",
            ylabel="L2 norm",
            logy=True,
        )
        if showcase is not None:
            plot_biography(ctx, bundle, showcase[0], showcase[1])

    # ---- summary + drafted claims ---------------------------------------------
    claims = draft_claims(ctx, bundle, cat_rows, summary_rows)
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)
    summary_md = render_summary(ctx, bundle, summary_rows, cat_rows, dropped, claims)
    summary_md_path = ctx.path("run_summary.md")
    bench.write_text(summary_md_path, summary_md)
    ctx.register_artifact(summary_md_path, "summary", "The seven standard questions, answered with this run's numbers.")
    print(f"[lab1] wrote run_summary.md and {len(claims)} drafted ledger claims")
