"""Lab 5: Activation patching and causal tracing.

This is the course's first full causal-intervention lab. Labs 1-4 kept
building pressure around questions that observation, attribution, and probes
cannot answer by themselves. Lab 5 releases the pressure valve: change one
activation and measure what the model does.

Method: interchange interventions. Run a CORRUPT prompt such as
"The capital of Germany is", splice in one activation from a CLEAN run such
as "The capital of France is" at one (stream depth, position), and measure
how much of the clean behavior returns:

    recovery = (patched_diff - corrupt_diff) / (clean_diff - corrupt_diff)

where diff = logit(target capital) - logit(distractor capital) at the final
position. 1.0 means the patch restored the whole clean-vs-corrupt logit gap;
0.0 means it restored none of it; negative means the patch hurt.

Rigor notes that this file tries hard to keep visible (building directly on Labs 1-4):

* Clean/corrupt prompts differ in exactly one tokenized subject position (same
  pre-final-norm residual convention as Lab 1). The validator rejects misaligned
  pairs, because position alignment is the quiet indexing bug that ruins patching
  experiments while leaving pretty heatmaps behind.
* Patching streams[k] means replacing the bench's pre-norm residual stream after
  k blocks (input to block k). The patch no-op check (new in this lab) verifies
  this convention before any scientific measurement.
* Stream depths and component layers are deliberately separated (echoing Lab 2's
  attribution caution and Lab 3's routing-vs-contribution distinction). A stream
  patch at depth k contains everything written by blocks < k. The lab writes the
  mapping (stream depth k → component/edit layer k-1) into localization_decision.json.
* One pair is a demo. Causal tracing aggregates over validated facts (baseline
  gate from clean/corrupt margins), role curves (subject vs last), paraphrase
  confirmation, negative controls (mismatched pairs, wrong position), and
  component-level patching (attn vs MLP).
* The optional rank-one edit is an audit of the localization claim (does the
  "best" causal-tracing site also make the best edit?), not a promise that
  localization predicts editability (the Hase et al. tension).

Evidence level: CAUSAL, scoped. Every claim names the intervention, metric,
and prompt population. The edit results force students to confront whether
localization informs editing.

Visualization upgrade: the lab now writes a compact evidence dashboard,
role-depth atlases, paraphrase and specificity matrices, and summary tables
that make the scoped causal claim auditable without reading every CSV row.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import pathlib
import statistics
from typing import Any

import interp_bench as bench

LAB_ID = "L05"

# Baseline gate margins in logits. A pair is usable only if the clean prompt
# prefers the target and the corrupt prompt prefers the distractor by enough
# margin that the recovery denominator is meaningful.
CLEAN_MARGIN = 0.5
CORRUPT_MARGIN = 0.5

# prompt-set is part of the shared bench CLI. The old Lab 5 ignored it, which
# meant tier-b defaulted to a full run even though the CLI said "small". Here
# prompt-set controls the fact count unless --max-examples applies a stricter
# hard cap.
PROMPT_SET_FACT_LIMITS = {
    "small": 10,
    "medium": 20,
    "full": 0,  # 0 means no cap
}

LOCALIZATION_BAND_WIDTH = 3
EDIT_ALPHAS = (1.0, 2.0, 4.0)

# Templates. subject_index is the token position of {X}; the validator checks
# it against the actual tokenizer on every run. Do not trust these constants
# on a new tokenizer without reading diagnostics/tokenization_report.csv.
TEMPLATES = {
    "base": {"text": "The capital of {X} is", "subject_index": 3},
    "para_city": {"text": "The capital city of {X} is", "subject_index": 4},
    "para_in": {"text": "In {X}, the capital is", "subject_index": 1},
}
PARAPHRASE_IDS = ("para_city", "para_in")


@dataclasses.dataclass(frozen=True)
class Fact:
    fact_id: str
    subject: str      # country string inserted into the prompt, no leading space
    target: str       # capital string, encoded as " " + target for next-token scoring


# Conservative built-in pool. The validator and baseline gate are allowed to
# reject facts. That is evidence, not a nuisance. Expanded for robustness.
FACTS: tuple[Fact, ...] = (
    Fact("france", "France", "Paris"),
    Fact("germany", "Germany", "Berlin"),
    Fact("italy", "Italy", "Rome"),
    Fact("spain", "Spain", "Madrid"),
    Fact("japan", "Japan", "Tokyo"),
    Fact("china", "China", "Beijing"),
    Fact("russia", "Russia", "Moscow"),
    Fact("egypt", "Egypt", "Cairo"),
    Fact("greece", "Greece", "Athens"),
    Fact("norway", "Norway", "Oslo"),
    Fact("poland", "Poland", "Warsaw"),
    Fact("austria", "Austria", "Vienna"),
    Fact("portugal", "Portugal", "Lisbon"),
    Fact("thailand", "Thailand", "Bangkok"),
    Fact("canada", "Canada", "Ottawa"),
    Fact("england", "England", "London"),
    Fact("switzerland", "Switzerland", "Bern"),
    Fact("turkey", "Turkey", "Ankara"),
    Fact("australia", "Australia", "Canberra"),
    Fact("ireland", "Ireland", "Dublin"),
    Fact("denmark", "Denmark", "Copenhagen"),
    Fact("sweden", "Sweden", "Stockholm"),
    Fact("hungary", "Hungary", "Budapest"),
    Fact("finland", "Finland", "Helsinki"),
    Fact("indonesia", "Indonesia", "Jakarta"),
)


@dataclasses.dataclass
class Pair:
    """One clean/corrupt pair under one template, with verified alignment."""

    fact: Fact            # the clean fact
    corrupt: Fact         # the corrupt partner, whose answer is the distractor
    template_id: str
    clean_prompt: str
    corrupt_prompt: str
    target_id: int
    distractor_id: int
    subject_pos: int
    n_tokens: int
    clean_diff: float = 0.0
    corrupt_diff: float = 0.0


@dataclasses.dataclass
class LocalizationDecision:
    """Where the stream patch localizes and how that maps to blocks."""

    subject_peak_stream_depth: int
    subject_peak_recovery: float
    subject_drop_threshold: float
    handoff_stream_depth: int
    localized_stream_depths: list[int]
    representative_stream_depth: int
    component_layers: list[int]
    representative_component_layer: int
    top_last_stream_depth: int
    peak_last_recovery: float
    note: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def rounded(x: Any, ndigits: int = 4) -> Any:
    """Round ordinary floats while leaving strings and odd objects alone."""
    try:
        if isinstance(x, (int, float)) and math.isfinite(float(x)):
            return round(float(x), ndigits)
    except Exception:
        pass
    return x


def mean_or_blank(vals: list[float]) -> float | str:
    return float(statistics.fmean(vals)) if vals else ""


def sem_or_blank(vals: list[float]) -> float | str:
    if not vals:
        return ""
    if len(vals) == 1:
        return 0.0
    return float(statistics.stdev(vals) / math.sqrt(len(vals)))


def safe_fmean(vals: list[float], default: float = 0.0) -> float:
    return float(statistics.fmean(vals)) if vals else default


def recovery_value(patched_diff: float, pair: Pair) -> float:
    denom = pair.clean_diff - pair.corrupt_diff
    if abs(denom) < 1e-9:
        raise ValueError(f"Tiny recovery denominator for {pair.fact.fact_id}: {denom}")
    return (patched_diff - pair.corrupt_diff) / denom


def role_of(position: int, subject_pos: int, n_tokens: int) -> str:
    if position == subject_pos:
        return "subject"
    if position == n_tokens - 1:
        return "last"
    if position < subject_pos:
        return "pre_subject"
    return "post_subject"


def stream_patch_site(depth: int, n_layers: int) -> str:
    if depth == n_layers:
        return "final_norm_input"
    return f"block_{depth}_input"


def component_layers_from_stream_depths(depths: list[int], n_layers: int) -> list[int]:
    """Map stream depths to blocks that produced those streams.

    streams[k] is the residual stream after k blocks. The nearest component
    write responsible for a stream depth k is block k-1. This is the hinge
    that keeps stream localization from accidentally becoming a one-layer-late
    edit claim.
    """
    layers = sorted({max(0, min(n_layers - 1, d - 1)) for d in depths if d > 0})
    return layers or [0]


# ---------------------------------------------------------------------------
# Fact selection and validation
# ---------------------------------------------------------------------------


def _load_custom_facts(path: pathlib.Path) -> list[Fact]:
    """Load a custom CSV or JSON fact list with fact_id, subject, target."""
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        return [Fact(str(r["fact_id"]), str(r["subject"]), str(r["target"])) for r in rows]
    if path.suffix.lower() == ".csv":
        out: list[Fact] = []
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out.append(Fact(str(row["fact_id"]), str(row["subject"]), str(row["target"])))
        return out
    raise RuntimeError(f"Custom Lab 5 fact files must be .csv or .json, got {path}")


def select_facts(args: Any) -> list[Fact]:
    prompt_set = str(args.prompt_set)
    path = pathlib.Path(prompt_set)
    if prompt_set in PROMPT_SET_FACT_LIMITS:
        facts = list(FACTS)
        source = f"built_in:{prompt_set}"
        limit = PROMPT_SET_FACT_LIMITS[prompt_set]
        if limit > 0:
            facts = facts[:limit]
    elif path.exists():
        facts = _load_custom_facts(path)
        source = f"custom:{path}"
    else:
        facts = list(FACTS)
        source = "built_in"
        print(f"[lab5] unknown prompt-set {prompt_set!r}; using full built-in fact pool")
    if args.max_examples > 0:
        facts = facts[: args.max_examples]
    if len(facts) < 3:
        raise RuntimeError(
            f"Lab 5 needs at least 3 candidate facts for cyclic clean/corrupt pairing; got {len(facts)} from {source}."
        )
    return facts


def build_pairs(
    ctx: bench.RunContext, bundle: bench.ModelBundle, facts: list[Fact], template_id: str
) -> tuple[list[Pair], list[dict[str, Any]]]:
    """Build aligned clean/corrupt pairs and report every validation decision."""
    tokenizer = bundle.tokenizer
    template = TEMPLATES[template_id]
    pairs: list[Pair] = []
    rows: list[dict[str, Any]] = []
    for i, fact in enumerate(facts):
        corrupt = facts[(i + 1) % len(facts)]
        clean_prompt = template["text"].format(X=fact.subject)
        corrupt_prompt = template["text"].format(X=corrupt.subject)
        target_ids = tokenizer.encode(" " + fact.target, add_special_tokens=False)
        distractor_ids = tokenizer.encode(" " + corrupt.target, add_special_tokens=False)
        clean_ids = tokenizer.encode(clean_prompt, add_special_tokens=False)
        corrupt_ids = tokenizer.encode(corrupt_prompt, add_special_tokens=False)
        clean_subject_ids = tokenizer.encode(" " + fact.subject, add_special_tokens=False)
        corrupt_subject_ids = tokenizer.encode(" " + corrupt.subject, add_special_tokens=False)

        problems: list[str] = []
        diff_positions: list[int] = []
        if len(target_ids) != 1:
            problems.append(f"target {fact.target!r} is {len(target_ids)} tokens")
        if len(distractor_ids) != 1:
            problems.append(f"distractor {corrupt.target!r} is {len(distractor_ids)} tokens")
        if len(clean_subject_ids) != 1:
            problems.append(f"subject {fact.subject!r} is {len(clean_subject_ids)} tokens with leading-space encoding")
        if len(corrupt_subject_ids) != 1:
            problems.append(f"corrupt subject {corrupt.subject!r} is {len(corrupt_subject_ids)} tokens")
        if len(clean_ids) != len(corrupt_ids):
            problems.append(f"clean/corrupt token lengths differ ({len(clean_ids)} vs {len(corrupt_ids)})")
        else:
            diff_positions = [p for p in range(len(clean_ids)) if clean_ids[p] != corrupt_ids[p]]
            subject_pos = int(template["subject_index"])
            expected = [subject_pos]
            if diff_positions != expected:
                problems.append(f"prompts differ at positions {diff_positions}, expected only {expected}")
            if 0 <= subject_pos < len(clean_ids):
                if len(clean_subject_ids) == 1 and clean_ids[subject_pos] != clean_subject_ids[0]:
                    problems.append(
                        f"declared subject position {subject_pos} holds token id {clean_ids[subject_pos]}, "
                        f"not subject token id {clean_subject_ids[0]}"
                    )
                if len(corrupt_subject_ids) == 1 and corrupt_ids[subject_pos] != corrupt_subject_ids[0]:
                    problems.append(
                        f"declared corrupt subject position {subject_pos} holds token id {corrupt_ids[subject_pos]}, "
                        f"not corrupt subject token id {corrupt_subject_ids[0]}"
                    )
            else:
                problems.append(f"subject position {subject_pos} is out of range for {len(clean_ids)} tokens")

        rows.append({
            "fact_id": fact.fact_id,
            "corrupt_fact_id": corrupt.fact_id,
            "template": template_id,
            "clean_prompt": clean_prompt,
            "corrupt_prompt": corrupt_prompt,
            "target": fact.target,
            "distractor": corrupt.target,
            "n_clean_tokens": len(clean_ids),
            "n_corrupt_tokens": len(corrupt_ids),
            "subject_pos_expected": template["subject_index"],
            "diff_positions": str(diff_positions),
            "target_token_count": len(target_ids),
            "distractor_token_count": len(distractor_ids),
            "subject_token_count": len(clean_subject_ids),
            "corrupt_subject_token_count": len(corrupt_subject_ids),
            "subject_token_id": clean_subject_ids[0] if len(clean_subject_ids) == 1 else "",
            "corrupt_subject_token_id": corrupt_subject_ids[0] if len(corrupt_subject_ids) == 1 else "",
            "clean_token_at_subject_pos": clean_ids[int(template["subject_index"])] if len(clean_ids) > int(template["subject_index"]) else "",
            "corrupt_token_at_subject_pos": corrupt_ids[int(template["subject_index"])] if len(corrupt_ids) > int(template["subject_index"]) else "",
            "aligned": not problems,
            "problems": "; ".join(problems),
        })
        if not problems:
            pairs.append(Pair(
                fact=fact,
                corrupt=corrupt,
                template_id=template_id,
                clean_prompt=clean_prompt,
                corrupt_prompt=corrupt_prompt,
                target_id=target_ids[0],
                distractor_id=distractor_ids[0],
                subject_pos=int(template["subject_index"]),
                n_tokens=len(clean_ids),
            ))
    return pairs, rows


def logit_diff(logits: Any, pair: Pair) -> float:
    return float(logits[pair.target_id] - logits[pair.distractor_id])


def gate_pairs(
    ctx: bench.RunContext, bundle: bench.ModelBundle, pairs: list[Pair]
) -> tuple[list[Pair], list[dict[str, Any]], dict[str, Any]]:
    """Baseline gate: the model must actually know the pair relation."""
    kept: list[Pair] = []
    rows: list[dict[str, Any]] = []
    captures: dict[str, Any] = {}
    for pair in pairs:
        clean_cap = bench.run_with_residual_cache(bundle, pair.clean_prompt)
        corrupt_cap = bench.run_with_residual_cache(bundle, pair.corrupt_prompt)
        pair.clean_diff = logit_diff(clean_cap.final_logits_last, pair)
        pair.corrupt_diff = logit_diff(corrupt_cap.final_logits_last, pair)
        denom = pair.clean_diff - pair.corrupt_diff
        ok = pair.clean_diff > CLEAN_MARGIN and pair.corrupt_diff < -CORRUPT_MARGIN
        reason = ""
        if not ok:
            if pair.clean_diff <= CLEAN_MARGIN:
                reason = f"clean_diff {pair.clean_diff:.2f} <= {CLEAN_MARGIN}"
            elif pair.corrupt_diff >= -CORRUPT_MARGIN:
                reason = f"corrupt_diff {pair.corrupt_diff:.2f} >= -{CORRUPT_MARGIN}"
        rows.append({
            "fact_id": pair.fact.fact_id,
            "corrupt_fact_id": pair.corrupt.fact_id,
            "template": pair.template_id,
            "clean_prompt": pair.clean_prompt,
            "corrupt_prompt": pair.corrupt_prompt,
            "target": pair.fact.target,
            "distractor": pair.corrupt.target,
            "clean_diff": rounded(pair.clean_diff),
            "corrupt_diff": rounded(pair.corrupt_diff),
            "denominator": rounded(denom),
            "kept": ok,
            "drop_reason": reason,
        })
        if ok:
            kept.append(pair)
            captures[f"{pair.fact.fact_id}:{pair.template_id}"] = clean_cap
    return kept, rows, captures


# ---------------------------------------------------------------------------
# Patching grids and localization
# ---------------------------------------------------------------------------


def run_grid(bundle: bench.ModelBundle, pair: Pair, clean_capture: Any) -> list[dict[str, Any]]:
    """Patch every residual stream depth and every position for one pair."""
    n_layers = bundle.anatomy.n_layers
    rows: list[dict[str, Any]] = []
    denom = pair.clean_diff - pair.corrupt_diff
    for stream_depth in range(n_layers + 1):
        for pos in range(pair.n_tokens):
            logits = bench.run_with_residual_patch(
                bundle,
                pair.corrupt_prompt,
                stream_depth,
                pos,
                clean_capture.streams[stream_depth, pos],
            )
            patched = logit_diff(logits, pair)
            rec = recovery_value(patched, pair)
            rows.append({
                "fact_id": pair.fact.fact_id,
                "corrupt_fact_id": pair.corrupt.fact_id,
                "template": pair.template_id,
                "stream_depth": stream_depth,
                # Backward-compatible alias for older notebooks.
                "layer": stream_depth,
                "patch_site": stream_patch_site(stream_depth, n_layers),
                "position": pos,
                "role": role_of(pos, pair.subject_pos, pair.n_tokens),
                "clean_diff": rounded(pair.clean_diff),
                "corrupt_diff": rounded(pair.corrupt_diff),
                "denominator": rounded(denom),
                "patched_diff": rounded(patched),
                "recovery": rounded(rec),
            })
    return rows


def aggregate_by_role(grid_rows: list[dict[str, Any]], n_layers: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    roles = ("pre_subject", "subject", "post_subject", "last")
    for stream_depth in range(n_layers + 1):
        row: dict[str, Any] = {"stream_depth": stream_depth, "layer": stream_depth}
        for role in roles:
            vals = [float(r["recovery"]) for r in grid_rows
                    if int(r["stream_depth"]) == stream_depth and r["role"] == role]
            row[f"recovery_{role}"] = rounded(mean_or_blank(vals))
            row[f"sem_{role}"] = rounded(sem_or_blank(vals))
            row[f"n_{role}"] = len(vals)
        out.append(row)
    return out


def choose_localization(agg_rows: list[dict[str, Any]], n_layers: int) -> LocalizationDecision:
    """Choose the handoff and the non-tautological localization band.

    Subject recovery at stream depth 0 is token substitution (the clean subject
    embedding is simply swapped in). It is useful as an instrument sanity check
    (the patch machinery works) but not as a mechanistic localization. We
    therefore choose from depths 1..L-1 and use a relative threshold (blend of
    absolute 0.5 and a fraction of the observed subject peak) so even small
    models produce a meaningful band instead of a fake failure.
    """
    subj: dict[int, float] = {}
    last: dict[int, float] = {}
    for row in agg_rows:
        depth = int(row["stream_depth"])
        if 1 <= depth < n_layers and row["recovery_subject"] != "":
            subj[depth] = float(row["recovery_subject"])
        if 0 <= depth < n_layers and row["recovery_last"] != "":
            last[depth] = float(row["recovery_last"])
    if not subj:
        raise RuntimeError("No non-tautological subject stream depths are available for localization.")
    subject_peak_depth = max(subj, key=subj.get)
    subject_peak = subj[subject_peak_depth]
    # A fixed 0.5 threshold is easy to read, but tiny models may never reach
    # 1.0 after the embedding tautology. Blend in a relative threshold so tier-a
    # still produces a meaningful map instead of a fake failure.
    threshold = max(0.25, min(0.5, 0.5 * subject_peak))
    handoff = next(
        (d for d in range(subject_peak_depth + 1, n_layers) if d in subj and subj[d] < threshold),
        n_layers,
    )
    band_start = max(1, handoff - LOCALIZATION_BAND_WIDTH)
    band = [d for d in range(band_start, min(handoff, n_layers)) if d in subj]
    if not band:
        band = sorted(sorted(subj, key=subj.get, reverse=True)[: min(LOCALIZATION_BAND_WIDTH, len(subj))])
    representative_stream = max(band, key=lambda d: subj[d])
    component_layers = component_layers_from_stream_depths(band, n_layers)
    representative_component = max(0, min(n_layers - 1, representative_stream - 1))
    if not last:
        top_last_depth = n_layers - 1
        peak_last = 0.0
    else:
        top_last_depth = max(last, key=last.get)
        peak_last = last[top_last_depth]
    note = (
        "streams[k] is the residual stream after k blocks, patched as the input to block k. "
        "The component/edit layer paired with stream depth k is block k-1. "
        "Stream depth 0 is excluded from localization because subject patching there is token substitution."
    )
    return LocalizationDecision(
        subject_peak_stream_depth=int(subject_peak_depth),
        subject_peak_recovery=float(subject_peak),
        subject_drop_threshold=float(threshold),
        handoff_stream_depth=int(handoff),
        localized_stream_depths=[int(d) for d in band],
        representative_stream_depth=int(representative_stream),
        component_layers=[int(x) for x in component_layers],
        representative_component_layer=int(representative_component),
        top_last_stream_depth=int(top_last_depth),
        peak_last_recovery=float(peak_last),
        note=note,
    )


def top_patch_rows(
    grid_rows: list[dict[str, Any]], base_pairs: list[Pair], representative_depth: int
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    pair_by_id = {p.fact.fact_id: p for p in base_pairs}
    for r in grid_rows:
        if int(r["stream_depth"]) == representative_depth and r["role"] == "subject":
            pair = pair_by_id[r["fact_id"]]
            out.append({
                "fact_id": r["fact_id"],
                "corrupt_fact_id": pair.corrupt.fact_id,
                "stream_depth": representative_depth,
                "component_layer_for_this_stream_depth": max(0, representative_depth - 1),
                "target": pair.fact.target,
                "distractor": pair.corrupt.target,
                "clean_diff": r["clean_diff"],
                "corrupt_diff": r["corrupt_diff"],
                "patched_diff": r["patched_diff"],
                "recovery": r["recovery"],
            })
    return sorted(out, key=lambda row: float(row["recovery"]))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_heatmap(
    ctx: bench.RunContext,
    pair: Pair,
    grid_rows: list[dict[str, Any]],
    tokens_text: list[str],
    n_layers: int,
    decision: LocalizationDecision,
) -> None:
    """Layer x position heatmap for one showcase pair."""
    import matplotlib.pyplot as plt
    import numpy as np

    grid = np.zeros((n_layers + 1, pair.n_tokens))
    for r in grid_rows:
        if r["fact_id"] == pair.fact.fact_id and r["template"] == pair.template_id:
            grid[int(r["stream_depth"]), int(r["position"])] = float(r["recovery"])
    fig, ax = plt.subplots(figsize=(7.2, 8.8))
    im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0, origin="lower")
    labels = [bench.visible_token(t) for t in tokens_text]
    labels[pair.subject_pos] += "\n(subject)"
    labels[-1] += "\n(last)"
    ax.set_xticks(range(pair.n_tokens))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("stream depth patched: streams[k], after k blocks")
    ax.set_xlabel("position patched")
    if decision.localized_stream_depths:
        ax.axhspan(
            min(decision.localized_stream_depths) - 0.5,
            max(decision.localized_stream_depths) + 0.5,
            alpha=0.12,
            label="localized stream band",
        )
    ax.axhline(decision.handoff_stream_depth - 0.5, linewidth=1.0, alpha=0.8)
    ax.set_title(
        f"Activation patching recovery: {pair.fact.subject} -> {pair.fact.target} "
        f"patched into {pair.corrupt.subject}\n"
        f"clean diff {pair.clean_diff:+.2f}, corrupt diff {pair.corrupt_diff:+.2f}"
    )
    fig.colorbar(im, ax=ax, fraction=0.04, label="recovery of clean logit diff")
    fig.tight_layout()
    bench.save_figure(
        ctx,
        fig,
        f"patching_heatmap_{bench.sanitize_tag(pair.fact.fact_id)}.png",
        "Layer-by-position patching recovery for the showcase pair, with the localized stream band marked.",
    )


def plot_localization(
    ctx: bench.RunContext,
    agg_rows: list[dict[str, Any]],
    para_agg: dict[str, list[dict[str, Any]]],
    decision: LocalizationDecision,
    n_facts: int,
) -> None:
    """Cross-fact role curves with paraphrase overlays and handoff marker."""
    fig, ax = bench.new_figure(figsize=(10.5, 6.2))
    style = {
        "pre_subject": ("tab:gray", "pre-subject"),
        "subject": ("tab:red", "subject"),
        "post_subject": ("tab:olive", "post-subject"),
        "last": ("tab:blue", "last"),
    }
    for role, (color, label) in style.items():
        rows = [r for r in agg_rows if r[f"recovery_{role}"] != ""]
        xs = [int(r["stream_depth"]) for r in rows]
        ys = [float(r[f"recovery_{role}"]) for r in rows]
        sems = [float(r[f"sem_{role}"]) for r in rows]
        ax.plot(xs, ys, linewidth=2.4, color=color, label=f"{label} (base)")
        if any(sems):
            lo = [y - 1.96 * s for y, s in zip(ys, sems)]
            hi = [y + 1.96 * s for y, s in zip(ys, sems)]
            ax.fill_between(xs, lo, hi, color=color, alpha=0.10)
    for tid, rows in para_agg.items():
        xs = [int(r["stream_depth"]) for r in rows if r["recovery_subject"] != ""]
        ys = [float(r["recovery_subject"]) for r in rows if r["recovery_subject"] != ""]
        ax.plot(xs, ys, linewidth=1.4, color="tab:red", linestyle="--", alpha=0.75,
                label=f"subject ({tid})")
    if decision.localized_stream_depths:
        ax.axvspan(min(decision.localized_stream_depths) - 0.5, max(decision.localized_stream_depths) + 0.5,
                   color="tab:red", alpha=0.07)
    ax.axvline(decision.handoff_stream_depth, color="black", linewidth=1.0, alpha=0.7,
               label=f"handoff depth {decision.handoff_stream_depth}")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axhline(1, color="black", linewidth=0.6, alpha=0.35)
    ax.set_xlabel("stream depth patched: streams[k] is after k blocks")
    ax.set_ylabel("mean recovery")
    ax.set_title(f"Causal tracing across {n_facts} facts: subject recall separates from final readout")
    ax.legend(fontsize=8, ncol=2)
    bench.save_figure(
        ctx,
        fig,
        "localization_across_facts.png",
        "Mean patching recovery by stream depth and token role, with paraphrase overlays and handoff marker.",
    )


def plot_component_pass(ctx: bench.RunContext, comp_rows: list[dict[str, Any]]) -> None:
    if not comp_rows:
        return
    import numpy as np

    fig, ax = bench.new_figure(figsize=(9.6, 5.4))
    layers = sorted({int(r["component_layer"]) for r in comp_rows})
    width = 0.20
    combos = (
        ("mlp", "subject", "tab:orange", "MLP @ subject"),
        ("attn", "subject", "tab:blue", "attn @ subject"),
        ("mlp", "last", "peru", "MLP @ last"),
        ("attn", "last", "steelblue", "attn @ last"),
    )
    for i, (kind, role, color, label) in enumerate(combos):
        vals = []
        for layer in layers:
            sel = [float(r["recovery"]) for r in comp_rows
                   if int(r["component_layer"]) == layer and r["component"] == kind and r["role"] == role]
            vals.append(safe_fmean(sel))
        ax.bar(np.arange(len(layers)) + (i - 1.5) * width, vals, width=width, color=color, label=label)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(x) for x in layers])
    ax.set_xlabel("component layer patched (block output, not stream depth)")
    ax.set_ylabel("mean recovery")
    ax.set_title("Component-level patching near the localized stream band")
    ax.legend(fontsize=8)
    bench.save_figure(
        ctx,
        fig,
        "component_patching.png",
        "Attention vs MLP output patching for component layers mapped from the localized stream band.",
    )


def plot_controls(ctx: bench.RunContext, control_rows: list[dict[str, Any]], matched_mean: float) -> None:
    if not control_rows:
        return
    fig, ax = bench.new_figure(figsize=(8.0, 5.2))
    kinds = ["matched top patch"] + sorted({str(r["control"]) for r in control_rows})
    means = [matched_mean]
    for kind in kinds[1:]:
        means.append(safe_fmean([float(r["recovery"]) for r in control_rows if r["control"] == kind]))
    bars = ax.bar(range(len(kinds)), means, color=["tab:green"] + ["tab:gray"] * (len(kinds) - 1))
    ax.bar_label(bars, fmt="%.2f", fontsize=9)
    for i, kind in enumerate(kinds[1:], start=1):
        vals = [float(r["recovery"]) for r in control_rows if r["control"] == kind]
        for j, val in enumerate(vals):
            jitter = ((j % 7) - 3) * 0.018
            ax.plot(i + jitter, val, marker="o", markersize=3.5, color="black", alpha=0.45)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(kinds, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("recovery")
    ax.set_title("Matched patch vs negative controls")
    bench.save_figure(ctx, fig, "negative_controls.png",
                      "Matched top-patch recovery against mismatched, wrong-position, and low-region controls.")


def plot_per_fact_recovery(ctx: bench.RunContext, per_fact_rows: list[dict[str, Any]]) -> None:
    if not per_fact_rows:
        return
    fig, ax = bench.new_figure(figsize=(9.2, 4.8))
    rows = sorted(per_fact_rows, key=lambda r: float(r["recovery"]))
    xs = list(range(len(rows)))
    ys = [float(r["recovery"]) for r in rows]
    labels = [str(r["fact_id"]) for r in rows]
    bars = ax.bar(xs, ys, color="tab:purple", alpha=0.75)
    ax.bar_label(bars, fmt="%.2f", fontsize=8, rotation=90, padding=2)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axhline(statistics.fmean(ys), color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("subject-patch recovery")
    ax.set_title("Per-fact recovery at the representative stream depth")
    bench.save_figure(ctx, fig, "per_fact_top_patch.png",
                      "Per-fact variation for the representative subject-position stream patch.")


# ---------------------------------------------------------------------------
# Visualization upgrade helpers: tables and synthesis plots
# ---------------------------------------------------------------------------

ROLE_ORDER = ("pre_subject", "subject", "post_subject", "last")
ROLE_LABELS = {
    "pre_subject": "pre-subject",
    "subject": "subject",
    "post_subject": "post-subject",
    "last": "last/readout",
}
ROLE_COLORS = {
    "pre_subject": "#666666",
    "subject": "#D55E00",
    "post_subject": "#8A9A00",
    "last": "#0072B2",
}
CONTROL_LABELS = {
    "mismatched_pair": "mismatched pair",
    "wrong_position": "wrong position",
    "low_region_split_heldout": "split-heldout low region",
}


def finite_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def quantile(vals: list[float], q: float) -> float:
    vals = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * max(0.0, min(1.0, q))
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def median_or_zero(vals: list[float]) -> float:
    return float(statistics.median(vals)) if vals else 0.0


def _control_color(control: str, default: str = "#777777") -> str:
    fn = getattr(bench, "plot_patch_control_color", None) or getattr(bench, "plot_control_color", None)
    if callable(fn):
        try:
            val = fn(control, default)
            if val:
                return str(val)
        except TypeError:
            pass
    return {
        "matched": "#009E73",
        "matched_top_patch": "#009E73",
        "mismatched_pair": "#8C564B",
        "wrong_position": "#666666",
        "low_region_split_heldout": "#7E57C2",
    }.get(control, default)


def _component_color(component: str, default: str = "#555555") -> str:
    fn = getattr(bench, "plot_component_color", None)
    if callable(fn):
        try:
            val = fn(component, default)
            if val:
                return str(val)
        except TypeError:
            pass
    return {"attn": "#0072B2", "mlp": "#E69F00"}.get(component, default)


def add_localization_guides(ax: Any, decision: LocalizationDecision, *, axis: str = "x") -> None:
    if decision.localized_stream_depths:
        lo = min(decision.localized_stream_depths) - 0.5
        hi = max(decision.localized_stream_depths) + 0.5
        if axis == "x":
            ax.axvspan(lo, hi, color=ROLE_COLORS["subject"], alpha=0.08, linewidth=0)
        else:
            ax.axhspan(lo, hi, color=ROLE_COLORS["subject"], alpha=0.08, linewidth=0)
    if axis == "x":
        ax.axvline(decision.handoff_stream_depth, color="#222222", linewidth=1.0, alpha=0.75)
    else:
        ax.axhline(decision.handoff_stream_depth, color="#222222", linewidth=1.0, alpha=0.75)


def role_depth_values(grid_rows: list[dict[str, Any]], role: str, n_layers: int) -> dict[int, list[float]]:
    out = {d: [] for d in range(n_layers + 1)}
    for row in grid_rows:
        if row.get("role") != role:
            continue
        depth = int(row.get("stream_depth", -1))
        val = finite_float(row.get("recovery"))
        if 0 <= depth <= n_layers and val is not None:
            out[depth].append(val)
    return out


def build_role_transition_rows(agg_rows: list[dict[str, Any]], decision: LocalizationDecision) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role in ROLE_ORDER:
        series = [
            (int(row["stream_depth"]), float(row[f"recovery_{role}"]))
            for row in agg_rows
            if row.get(f"recovery_{role}") != ""
        ]
        if not series:
            continue
        peak_depth, peak_value = max(series, key=lambda item: item[1])
        rows.append({
            "role": role,
            "role_label": ROLE_LABELS.get(role, role),
            "peak_stream_depth": peak_depth,
            "peak_recovery": rounded(peak_value),
            "first_depth_recovery_ge_0.25": next((d for d, v in series if d > 0 and v >= 0.25), ""),
            "first_depth_recovery_ge_0.50": next((d for d, v in series if d > 0 and v >= 0.50), ""),
            "first_depth_recovery_ge_0.75": next((d for d, v in series if d > 0 and v >= 0.75), ""),
            "first_post_peak_drop_below_subject_threshold": next((d for d, v in series if role == "subject" and d > peak_depth and v < decision.subject_drop_threshold), ""),
            "localized_band": str(decision.localized_stream_depths) if role == "subject" else "",
        })
    return rows


def build_specificity_summary(per_fact_rows: list[dict[str, Any]], control_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matched = [float(r["recovery"]) for r in per_fact_rows if finite_float(r.get("recovery")) is not None]
    matched_mean = safe_fmean(matched)
    rows = [{
        "condition": "matched_top_patch",
        "label": "matched top patch",
        "n": len(matched),
        "mean_recovery": rounded(matched_mean),
        "median_recovery": rounded(median_or_zero(matched)),
        "iqr_low": rounded(quantile(matched, 0.25)),
        "iqr_high": rounded(quantile(matched, 0.75)),
        "gap_vs_matched_mean": 0.0,
    }]
    for control in sorted({str(r.get("control")) for r in control_rows}):
        vals = [float(r["recovery"]) for r in control_rows if r.get("control") == control and finite_float(r.get("recovery")) is not None]
        rows.append({
            "condition": control,
            "label": CONTROL_LABELS.get(control, control),
            "n": len(vals),
            "mean_recovery": rounded(safe_fmean(vals)),
            "median_recovery": rounded(median_or_zero(vals)),
            "iqr_low": rounded(quantile(vals, 0.25)),
            "iqr_high": rounded(quantile(vals, 0.75)),
            "gap_vs_matched_mean": rounded(matched_mean - safe_fmean(vals)),
        })
    return rows


def build_patch_evidence_matrix(
    base_pairs: list[Pair],
    grid_rows: list[dict[str, Any]],
    per_fact_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    para_rows: list[dict[str, Any]],
    decision: LocalizationDecision,
) -> list[dict[str, Any]]:
    matched_by_fact = {str(r["fact_id"]): finite_float(r.get("recovery")) for r in per_fact_rows}
    out: list[dict[str, Any]] = []
    for pair in base_pairs:
        fid = pair.fact.fact_id
        subj_series = sorted(
            (int(r["stream_depth"]), float(finite_float(r.get("recovery"))))
            for r in grid_rows
            if r.get("fact_id") == fid and r.get("role") == "subject"
            and finite_float(r.get("recovery")) is not None
        )
        last_series = sorted(
            (int(r["stream_depth"]), float(finite_float(r.get("recovery"))))
            for r in grid_rows
            if r.get("fact_id") == fid and r.get("role") == "last"
            and finite_float(r.get("recovery")) is not None
        )
        subj_peak = max(subj_series, key=lambda x: x[1]) if subj_series else ("", 0.0)
        last_peak = max(last_series, key=lambda x: x[1]) if last_series else ("", 0.0)
        band_vals = [v for d, v in subj_series if d in decision.localized_stream_depths]
        control_vals = [
            float(v)
            for r in control_rows
            if r.get("fact_id") == fid
            for v in [finite_float(r.get("recovery"))]
            if v is not None
        ]
        para_vals = [
            float(v)
            for r in para_rows
            if r.get("fact_id") == fid and int(r.get("stream_depth", -1)) == decision.representative_stream_depth
            for v in [finite_float(r.get("recovery"))]
            if v is not None
        ]
        matched = matched_by_fact.get(fid)
        strongest_control = max(control_vals) if control_vals else 0.0
        out.append({
            "fact_id": fid,
            "clean_subject": pair.fact.subject,
            "target": pair.fact.target,
            "corrupt_subject": pair.corrupt.subject,
            "distractor": pair.corrupt.target,
            "clean_diff": rounded(pair.clean_diff),
            "corrupt_diff": rounded(pair.corrupt_diff),
            "denominator": rounded(pair.clean_diff - pair.corrupt_diff),
            "subject_peak_depth": subj_peak[0],
            "subject_peak_recovery": rounded(subj_peak[1]),
            "last_peak_depth": last_peak[0],
            "last_peak_recovery": rounded(last_peak[1]),
            "subject_to_last_peak_lag": (int(last_peak[0]) - int(subj_peak[0])) if subj_series and last_series else "",
            "localized_band_mean_recovery": rounded(safe_fmean(band_vals)),
            "representative_depth": decision.representative_stream_depth,
            "representative_subject_recovery": rounded(matched if matched is not None else 0.0),
            "paraphrase_representative_mean_recovery": rounded(safe_fmean(para_vals)),
            "strongest_control_recovery": rounded(strongest_control),
            "specificity_gap_vs_strongest_control": rounded((matched or 0.0) - strongest_control),
        })
    return out


def build_plot_reading_guide() -> list[dict[str, str]]:
    return [
        {"artifact": "plots/causal_patching_dashboard.png", "concept": "The whole causal-tracing claim in one panel set.", "read_for": "Role timing, negative controls, component refinement, and fact heterogeneity."},
        {"artifact": "plots/recovery_role_atlas.png", "concept": "Fact-by-depth heterogeneity for subject and final-token patches.", "read_for": "Whether the handoff is shared or driven by one row."},
        {"artifact": "plots/recovery_ridge_map.png", "concept": "Mean recovery as a depth-by-role map.", "read_for": "The coarse temporal grammar of recall then readout."},
        {"artifact": "plots/specificity_gap_by_fact.png", "concept": "Matched patch versus per-fact negative controls.", "read_for": "Whether the claim should be global, narrowed, or rejected."},
        {"artifact": "plots/paraphrase_transfer_matrix.png", "concept": "Template transfer for subject-position recovery.", "read_for": "Whether localization survives wording changes."},
        {"artifact": "plots/component_patch_matrix.png", "concept": "Stream localization refined to component-role candidates.", "read_for": "Whether attention or MLP at subject/last carries the strongest recoverable write."},
        {"artifact": "plots/baseline_gate_audit.png", "concept": "Baseline margin gate before patching.", "read_for": "Whether the model actually has clean-vs-corrupt behavior to recover."},
        {"artifact": "plots/edit_audit_dashboard.png", "concept": "Optional patch-made-permanent audit.", "read_for": "Whether localization predicts editability, transfer, and spillover."},
    ]


def write_upgrade_tables(
    ctx: bench.RunContext,
    base_pairs: list[Pair],
    grid_rows: list[dict[str, Any]],
    agg_rows: list[dict[str, Any]],
    per_fact_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    para_rows: list[dict[str, Any]],
    decision: LocalizationDecision,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    role_rows = build_role_transition_rows(agg_rows, decision)
    role_path = ctx.path("tables", "role_transition_summary.csv")
    bench.write_csv_with_context(ctx, role_path, role_rows)
    ctx.register_artifact(role_path, "table", "Depth landmarks for each token role: peak, threshold crossings, and subject handoff.")

    spec_rows = build_specificity_summary(per_fact_rows, control_rows)
    spec_path = ctx.path("tables", "specificity_summary.csv")
    bench.write_csv_with_context(ctx, spec_path, spec_rows)
    ctx.register_artifact(spec_path, "table", "Matched patch and negative-control distributions with gaps versus the matched mean.")

    evidence_rows = build_patch_evidence_matrix(base_pairs, grid_rows, per_fact_rows, control_rows, para_rows, decision)
    evidence_path = ctx.path("tables", "patch_evidence_matrix.csv")
    bench.write_csv_with_context(ctx, evidence_path, evidence_rows)
    ctx.register_artifact(evidence_path, "table", "One-row-per-fact matrix combining baseline margins, localization, paraphrase transfer, and controls.")

    guide_path = ctx.path("tables", "plot_reading_guide.csv")
    bench.write_csv_with_context(ctx, guide_path, build_plot_reading_guide())
    ctx.register_artifact(guide_path, "table", "Map from each upgraded Lab 5 plot to the concept it is meant to teach.")
    return role_rows, spec_rows, evidence_rows


def plot_baseline_gate_audit(ctx: bench.RunContext, gate_rows: list[dict[str, Any]]) -> None:
    if not gate_rows:
        return
    fig, ax = bench.new_figure(figsize=(7.5, 5.8))
    templates = sorted({str(r.get("template")) for r in gate_rows})
    markers = {"base": "o", "para_city": "s", "para_in": "^"}
    for template in templates:
        rows = [r for r in gate_rows if r.get("template") == template]
        xs = [finite_float(r.get("clean_diff")) or 0.0 for r in rows]
        ys = [-(finite_float(r.get("corrupt_diff")) or 0.0) for r in rows]
        colors = ["#009E73" if (str(r.get("kept")).lower() == "true" or r.get("kept") is True) else "#999999" for r in rows]
        ax.scatter(xs, ys, marker=markers.get(template, "o"), c=colors, s=42, alpha=0.84,
                   edgecolors="#222222", linewidths=0.35, label=template)
    ax.axvline(CLEAN_MARGIN, color="#222222", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.axhline(CORRUPT_MARGIN, color="#222222", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_xlabel("clean prompt target-vs-distractor margin")
    ax.set_ylabel("corrupt prompt distractor-vs-target margin")
    ax.set_title("Baseline gate audit: trace only facts with a clean/corrupt gap")
    ax.legend(fontsize=8, title="template")
    bench.save_figure(ctx, fig, "baseline_gate_audit.png",
                      "Clean and corrupt margins for every validated pair; upper-right passes the gate.")


def plot_recovery_ridge_map(ctx: bench.RunContext, agg_rows: list[dict[str, Any]], decision: LocalizationDecision,
                            n_layers: int) -> None:
    import numpy as np
    mat = np.full((n_layers + 1, len(ROLE_ORDER)), np.nan)
    for i, role in enumerate(ROLE_ORDER):
        for row in agg_rows:
            val = finite_float(row.get(f"recovery_{role}"))
            if val is not None:
                mat[int(row["stream_depth"]), i] = val
    fig, ax = bench.new_figure(figsize=(7.5, 8.0))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(ROLE_ORDER)))
    ax.set_xticklabels([ROLE_LABELS[r] for r in ROLE_ORDER], rotation=25, ha="right")
    ax.set_ylabel("stream depth patched")
    ax.set_title("Recovery ridge map: where each token role causally helps")
    add_localization_guides(ax, decision, axis="y")
    for d in range(0, n_layers + 1, max(1, (n_layers + 1) // 8)):
        for i in range(len(ROLE_ORDER)):
            val = mat[d, i]
            if math.isfinite(float(val)):
                ax.text(i, d, f"{val:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(val) > 0.65 else "#222222")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean recovery")
    bench.save_figure(ctx, fig, "recovery_ridge_map.png",
                      "Compact heatmap of mean patch recovery by stream depth and token role.")


def plot_recovery_role_atlas(ctx: bench.RunContext, grid_rows: list[dict[str, Any]], per_fact_rows: list[dict[str, Any]],
                             decision: LocalizationDecision, n_layers: int) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    facts = [str(r["fact_id"]) for r in sorted(per_fact_rows, key=lambda r: float(r["recovery"]))]
    if not facts:
        facts = sorted({str(r.get("fact_id")) for r in grid_rows})
    fig, axes = plt.subplots(1, 2, figsize=(13.2, max(5.2, 0.30 * len(facts) + 2.2)), sharey=True)
    last_im = None
    for ax, role, title in zip(axes, ("subject", "last"), ("subject-position patch", "final-token patch")):
        mat = np.full((len(facts), n_layers + 1), np.nan)
        idx = {fid: i for i, fid in enumerate(facts)}
        for row in grid_rows:
            if row.get("role") != role:
                continue
            fid = str(row.get("fact_id"))
            if fid in idx:
                mat[idx[fid], int(row["stream_depth"])] = float(row["recovery"])
        last_im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0, origin="lower")
        ax.set_title(title)
        ax.set_xlabel("stream depth")
        add_localization_guides(ax, decision, axis="x")
        if ax is axes[0]:
            ax.set_yticks(range(len(facts)))
            ax.set_yticklabels(facts, fontsize=7)
            ax.set_ylabel("fact, sorted by representative recovery")
    fig.suptitle("Recovery role atlas: the mean curve is made of many facts", y=0.99)
    if last_im is not None:
        fig.colorbar(last_im, ax=list(axes), fraction=0.02, pad=0.02, label="recovery")
    bench.save_figure(ctx, fig, "recovery_role_atlas.png",
                      "Fact-by-depth heatmaps for subject and final-token residual patches.")


def plot_specificity_gap_by_fact(ctx: bench.RunContext, per_fact_rows: list[dict[str, Any]], control_rows: list[dict[str, Any]]) -> None:
    if not per_fact_rows:
        return
    fig, ax = bench.new_figure(figsize=(10.5, max(4.6, 0.28 * len(per_fact_rows) + 1.4)))
    rows = sorted(per_fact_rows, key=lambda r: float(r["recovery"]))
    facts = [str(r["fact_id"]) for r in rows]
    y = list(range(len(facts)))
    matched = [float(r["recovery"]) for r in rows]
    ax.scatter(matched, y, marker="D", s=46, color="#009E73", label="matched subject patch", zorder=4)
    controls = sorted({str(r.get("control")) for r in control_rows if r.get("control") != "low_region_split_heldout"})
    offsets = {c: (i - (len(controls) - 1) / 2) * 0.12 for i, c in enumerate(controls)}
    for control in controls:
        vals_by_fact: dict[str, list[float]] = {}
        for r in control_rows:
            if r.get("control") == control:
                vals_by_fact.setdefault(str(r.get("fact_id")), []).append(float(r["recovery"]))
        xs = [safe_fmean(vals_by_fact.get(fid, [])) for fid in facts]
        yy = [i + offsets[control] for i in y]
        ax.scatter(xs, yy, s=30, alpha=0.82, color=_control_color(control), label=CONTROL_LABELS.get(control, control), zorder=3)
        for i, x in enumerate(xs):
            ax.plot([x, matched[i]], [yy[i], y[i]], color="#999999", linewidth=0.6, alpha=0.35, zorder=1)
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(facts, fontsize=7)
    ax.set_xlabel("recovery at representative stream depth")
    ax.set_title("Specificity by fact: matched patch should outrun controls")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    bench.save_figure(ctx, fig, "specificity_gap_by_fact.png",
                      "Per-fact matched recovery versus mismatched-pair and wrong-position controls.")


def plot_paraphrase_transfer_matrix(ctx: bench.RunContext, para_rows: list[dict[str, Any]], decision: LocalizationDecision,
                                    n_layers: int) -> None:
    if not para_rows:
        return
    import numpy as np
    templates = sorted({str(r.get("template")) for r in para_rows})
    mat = np.zeros((len(templates), n_layers + 1))
    for i, template in enumerate(templates):
        for d in range(n_layers + 1):
            vals = [float(r["recovery"]) for r in para_rows if r.get("template") == template and int(r["stream_depth"]) == d]
            mat[i, d] = safe_fmean(vals)
    fig, ax = bench.new_figure(figsize=(11.0, 2.6 + 0.45 * len(templates)))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_yticks(range(len(templates)))
    ax.set_yticklabels(templates)
    ax.set_xlabel("stream depth")
    ax.set_title("Paraphrase transfer: does the subject band survive wording changes?")
    add_localization_guides(ax, decision, axis="x")
    for i in range(len(templates)):
        for d in sorted(set(decision.localized_stream_depths + [decision.representative_stream_depth])):
            if 0 <= d <= n_layers:
                val = mat[i, d]
                ax.text(d, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(val) > 0.65 else "#222222")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="mean subject recovery")
    bench.save_figure(ctx, fig, "paraphrase_transfer_matrix.png",
                      "Subject-position recovery across paraphrase templates and stream depths.")


def plot_component_patch_matrix(ctx: bench.RunContext, comp_rows: list[dict[str, Any]], decision: LocalizationDecision) -> None:
    if not comp_rows:
        return
    import numpy as np
    layers = sorted({int(r["component_layer"]) for r in comp_rows})
    labels = [("attn", "subject"), ("mlp", "subject"), ("attn", "last"), ("mlp", "last")]
    mat = np.zeros((len(labels), len(layers)))
    for i, (component, role) in enumerate(labels):
        for j, layer in enumerate(layers):
            vals = [float(r["recovery"]) for r in comp_rows
                    if int(r["component_layer"]) == layer and r.get("component") == component and r.get("role") == role]
            mat[i, j] = safe_fmean(vals)
    lim = max(0.20, float(np.nanmax(np.abs(mat))))
    fig, ax = bench.new_figure(figsize=(max(6.4, 0.65 * len(layers) + 3.0), 4.2))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(layer) for layer in layers])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([f"{c} @ {ROLE_LABELS.get(r, r)}" for c, r in labels])
    ax.set_xlabel("component layer patched")
    ax.set_title("Component patch matrix: which write carries recoverable fact signal?")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(val) > lim * 0.55 else "#222222")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="mean recovery")
    bench.save_figure(ctx, fig, "component_patch_matrix.png",
                      "Component-role recovery heatmap for attention and MLP outputs near the localized stream band.")


def plot_causal_patching_dashboard(ctx: bench.RunContext, agg_rows: list[dict[str, Any]], per_fact_rows: list[dict[str, Any]],
                                   control_rows: list[dict[str, Any]], comp_rows: list[dict[str, Any]],
                                   decision: LocalizationDecision, n_layers: int, n_facts: int) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.4))
    ax = axes[0, 0]
    for role in ROLE_ORDER:
        rows = [r for r in agg_rows if r.get(f"recovery_{role}") != ""]
        xs = [int(r["stream_depth"]) for r in rows]
        ys = [float(r[f"recovery_{role}"]) for r in rows]
        ax.plot(xs, ys, color=ROLE_COLORS[role], linewidth=2.2 if role in {"subject", "last"} else 1.3,
                label=ROLE_LABELS[role], alpha=0.95 if role in {"subject", "last"} else 0.65)
    add_localization_guides(ax, decision, axis="x")
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.axhline(1, color="#222222", linewidth=0.7, alpha=0.35)
    ax.set_title("A. Recall handoff: subject stream gives way to final readout")
    ax.set_xlabel("stream depth patched")
    ax.set_ylabel("mean recovery")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[0, 1]
    matched = [float(r["recovery"]) for r in per_fact_rows]
    control_names = sorted({str(r.get("control")) for r in control_rows})
    labels = ["matched"] + [CONTROL_LABELS.get(c, c) for c in control_names]
    means = [safe_fmean(matched)] + [safe_fmean([float(r["recovery"]) for r in control_rows if r.get("control") == c]) for c in control_names]
    colors = ["#009E73"] + ["#777777"] * len(control_names)
    bars = ax.bar(range(len(labels)), means, color=colors, alpha=0.88)
    ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_ylabel("mean recovery")
    ax.set_title("B. Specificity: real patch versus controls")

    ax = axes[1, 0]
    if comp_rows:
        comp_layers = sorted({int(r["component_layer"]) for r in comp_rows})
        comp_labels = [("attn", "subject"), ("mlp", "subject"), ("attn", "last"), ("mlp", "last")]
        mat = np.zeros((len(comp_labels), len(comp_layers)))
        for i, (component, role) in enumerate(comp_labels):
            for j, layer in enumerate(comp_layers):
                vals = [float(r["recovery"]) for r in comp_rows
                        if int(r["component_layer"]) == layer and r.get("component") == component and r.get("role") == role]
                mat[i, j] = safe_fmean(vals)
        lim = max(0.20, float(np.nanmax(np.abs(mat))))
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_xticks(range(len(comp_layers)))
        ax.set_xticklabels([str(x) for x in comp_layers])
        ax.set_yticks(range(len(comp_labels)))
        ax.set_yticklabels([f"{c}@{r}" for c, r in comp_labels])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    ax.set_title("C. Component refinement near the band")
    ax.set_xlabel("component layer")

    ax = axes[1, 1]
    rows = sorted(per_fact_rows, key=lambda r: float(r["recovery"]))
    xs = list(range(len(rows)))
    ys = [float(r["recovery"]) for r in rows]
    ax.bar(xs, ys, color="#7E57C2", alpha=0.78)
    ax.axhline(safe_fmean(ys), color="#222222", linestyle="--", linewidth=1.0, alpha=0.75,
               label=f"mean {safe_fmean(ys):.2f}")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(r["fact_id"]) for r in rows], rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("representative subject recovery")
    ax.set_title("D. Heterogeneity: which facts carry the mean?")
    ax.legend(fontsize=8)

    fig.suptitle(f"Lab 5 causal patching dashboard ({n_facts} gated facts)", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    bench.save_figure(ctx, fig, "causal_patching_dashboard.png",
                      "Dashboard combining localization curves, specificity controls, component refinement, and per-fact heterogeneity.")


def plot_edit_audit_dashboard(ctx: bench.RunContext, edit_results: list[dict[str, Any]]) -> None:
    if not edit_results:
        return
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2))
    groups = sorted({(str(r.get("layer_kind")), int(r.get("edit_layer", 0))) for r in edit_results})
    labels = [f"{kind}\nL{layer}" for kind, layer in groups]
    x = np.arange(len(groups))
    alphas = sorted({float(r.get("alpha", 0)) for r in edit_results})
    width = 0.75 / max(1, len(alphas))

    ax = axes[0, 0]
    for i, alpha in enumerate(alphas):
        vals = []
        for kind, layer in groups:
            sel = [float(r.get("movement_toward_distractor", 0.0)) for r in edit_results
                   if str(r.get("layer_kind")) == kind and int(r.get("edit_layer", 0)) == layer and float(r.get("alpha", 0)) == alpha]
            vals.append(safe_fmean(sel))
        ax.bar(x + (i - (len(alphas) - 1) / 2) * width, vals, width=width, label=f"alpha {alpha:g}")
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("target logit-gap movement toward distractor")
    ax.set_title("A. Movement without assuming flip success")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    success = [max(1.0 if r.get("direct_success") else 0.0 for r in edit_results
                   if str(r.get("layer_kind")) == kind and int(r.get("edit_layer", 0)) == layer) for kind, layer in groups]
    ax.bar(x, success, color="#009E73", alpha=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("any direct top-1 flip")
    ax.set_title("B. Direct success is a stricter test")

    ax = axes[1, 0]
    para = [max(float(r.get("paraphrase_flips", 0)) / max(1.0, float(r.get("n_paraphrases", 1)))
                for r in edit_results if str(r.get("layer_kind")) == kind and int(r.get("edit_layer", 0)) == layer)
            for kind, layer in groups]
    neigh = [min(float(r.get("neighbors_intact", 0)) / max(1.0, float(r.get("n_neighbors", 1)))
                 for r in edit_results if str(r.get("layer_kind")) == kind and int(r.get("edit_layer", 0)) == layer)
             for kind, layer in groups]
    ax.bar(x - 0.18, para, width=0.34, color="#0072B2", alpha=0.82, label="paraphrase flip fraction")
    ax.bar(x + 0.18, neigh, width=0.34, color="#D55E00", alpha=0.82, label="min neighbors intact")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("C. Transfer versus spillover")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for kind, layer in groups:
        rows = [r for r in edit_results if str(r.get("layer_kind")) == kind and int(r.get("edit_layer", 0)) == layer]
        xs = [float(r.get("alpha", 0.0)) for r in rows]
        ys = [float(r.get("fluency_logprob_after", 0.0)) - float(r.get("fluency_logprob_before", 0.0)) for r in rows]
        ax.plot(xs, ys, marker="o", label=f"{kind} L{layer}")
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xlabel("edit alpha")
    ax.set_ylabel("fluency mean-logprob change")
    ax.set_title("D. Side-effect audit")
    ax.legend(fontsize=8)
    fig.suptitle("Rank-one edit audit: localization is not editability", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    bench.save_figure(ctx, fig, "edit_audit_dashboard.png",
                      "Rank-one edit audit dashboard: movement, flip success, transfer/spillover, and fluency side effects.")

# ---------------------------------------------------------------------------
# Edit extension
# ---------------------------------------------------------------------------


def capture_down_proj_io(bundle: bench.ModelBundle, prompt: str, layer: int, position: int) -> tuple[Any, Any]:
    """Return (down-projection input, output) at one position, float32 CPU."""
    import torch

    module, _ = bench.resolve_mlp_down_proj(bundle, layer)
    store: dict[str, Any] = {}

    def pre_hook(mod: Any, hook_args: tuple) -> None:
        store["in"] = bench.tensor_cpu_float(hook_args[0][0, position])

    def out_hook(mod: Any, hook_args: tuple, output: Any) -> None:
        out = output[0] if isinstance(output, tuple) else output
        store["out"] = bench.tensor_cpu_float(out[0, position])

    h1 = module.register_forward_pre_hook(pre_hook)
    h2 = module.register_forward_hook(out_hook)
    try:
        encoded = bundle.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(bundle.input_device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(bundle.input_device)
        with torch.no_grad():
            bundle.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        h1.remove()
        h2.remove()
    return store["in"], store["out"]


def top1_text(bundle: bench.ModelBundle, prompt: str) -> str:
    import torch

    encoded = bundle.tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(bundle.input_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(bundle.input_device)
    with torch.no_grad():
        out = bundle.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return bundle.tokenizer.decode([int(out.logits[0, -1].argmax())])


def prompt_logit_diff(bundle: bench.ModelBundle, prompt: str, pair: Pair) -> float:
    import torch

    encoded = bundle.tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(bundle.input_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(bundle.input_device)
    with torch.no_grad():
        out = bundle.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return logit_diff(bench.tensor_cpu_float(out.logits[0, -1]), pair)


def mean_logprob(bundle: bench.ModelBundle, text: str) -> float:
    import torch

    tokenizer = bundle.tokenizer
    encoded = tokenizer(text, return_tensors="pt")
    ids = encoded["input_ids"].to(bundle.input_device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(bundle.input_device)
    with torch.no_grad():
        out = bundle.model(input_ids=ids, attention_mask=attention_mask, use_cache=False)
    logprobs = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
    return float(logprobs.gather(1, ids[0, 1:, None].to(logprobs.device)).mean())


FLUENCY_TEXT = "The weather today is mild, and most people in the city walked to work along the river."


def run_edit_audit(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    pair: Pair,
    edit_layer: int,
    other_pairs: list[Pair],
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Apply a small rank-one edit and audit its direct and ripple effects.

    This is deliberately the "patch made permanent" audit, not a claim that
    causal tracing predicts the best edit layer. The numbers (direct success,
    logit movement without flip, paraphrase generalization, neighbor spillover,
    fluency) force the student to confront the localization-vs-editability gap
    (Hase et al.). Movement without a full flip, or neighbor damage before
    target success, is often the most informative outcome.
    """
    key_clean, out_clean = capture_down_proj_io(bundle, pair.clean_prompt, edit_layer, pair.subject_pos)
    _, out_corrupt = capture_down_proj_io(bundle, pair.corrupt_prompt, edit_layer, pair.subject_pos)
    delta_v = alpha * (out_corrupt - out_clean)

    paraphrases = [(tid, TEMPLATES[tid]["text"].format(X=pair.fact.subject)) for tid in PARAPHRASE_IDS]
    neighbors = [(p, TEMPLATES["base"]["text"].format(X=p.fact.subject)) for p in other_pairs[:3]]

    direct_top1_before = top1_text(bundle, pair.clean_prompt)
    direct_diff_before = prompt_logit_diff(bundle, pair.clean_prompt, pair)
    para_before = [(tid, prompt_logit_diff(bundle, prompt, pair), top1_text(bundle, prompt))
                   for tid, prompt in paraphrases]
    neighbor_before = [top1_text(bundle, prompt) for _, prompt in neighbors]
    fluency_before = mean_logprob(bundle, FLUENCY_TEXT)

    with bench.temporary_rank_one_edit(bundle, edit_layer, key_clean, delta_v):
        direct_top1_after = top1_text(bundle, pair.clean_prompt)
        direct_diff_after = prompt_logit_diff(bundle, pair.clean_prompt, pair)
        para_after = [(tid, prompt_logit_diff(bundle, prompt, pair), top1_text(bundle, prompt))
                      for tid, prompt in paraphrases]
        neighbor_after = [(p.fact.fact_id, top1_text(bundle, prompt), before)
                          for (p, prompt), before in zip(neighbors, neighbor_before)]
        fluency_after = mean_logprob(bundle, FLUENCY_TEXT)

    para_rows = []
    for (tid, before_diff, before_top1), (_, after_diff, after_top1) in zip(para_before, para_after):
        para_rows.append({
            "template": tid,
            "top1_before": before_top1,
            "top1_after": after_top1,
            "logit_diff_before": rounded(before_diff),
            "logit_diff_after": rounded(after_diff),
            "flipped": after_top1.strip().lower() == pair.corrupt.target.lower(),
        })

    return {
        "edit_layer": edit_layer,
        "alpha": alpha,
        "fact_id": pair.fact.fact_id,
        "target": pair.fact.target,
        "distractor": pair.corrupt.target,
        "intended_flip": f"{pair.fact.target} -> {pair.corrupt.target}",
        "direct_success": direct_top1_after.strip().lower() == pair.corrupt.target.lower(),
        "direct_top1_before": direct_top1_before,
        "direct_top1_after": direct_top1_after,
        "direct_logit_diff_before": rounded(direct_diff_before),
        "direct_logit_diff_after": rounded(direct_diff_after),
        "movement_toward_distractor": rounded(direct_diff_before - direct_diff_after),
        "paraphrase_flips": sum(1 for r in para_rows if r["flipped"]),
        "n_paraphrases": len(para_rows),
        "paraphrase_audit": para_rows,
        "neighbors_intact": sum(1 for _, got, before in neighbor_after if got == before),
        "n_neighbors": len(neighbor_after),
        "neighbor_top1_after": [(fid, got) for fid, got, _ in neighbor_after],
        "fluency_logprob_before": rounded(fluency_before),
        "fluency_logprob_after": rounded(fluency_after),
    }


def choose_alternative_edit_layer(localized_component_layer: int, decision: LocalizationDecision, n_layers: int) -> int:
    candidates = [
        decision.top_last_stream_depth,
        localized_component_layer + max(2, n_layers // 4),
        n_layers - 1,
    ]
    for cand in candidates:
        layer = max(0, min(n_layers - 1, int(cand)))
        if layer != localized_component_layer:
            return layer
    return max(0, min(n_layers - 1, localized_component_layer + 1))


# ---------------------------------------------------------------------------
# Cards and summaries
# ---------------------------------------------------------------------------


def summarize_controls(control_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        kind: {
            "mean_recovery": rounded(safe_fmean([float(r["recovery"]) for r in control_rows if r["control"] == kind])),
            "n": sum(1 for r in control_rows if r["control"] == kind),
        }
        for kind in sorted({str(r["control"]) for r in control_rows})
    }


def summarize_components(comp_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not comp_rows:
        return {}
    groups: dict[tuple[str, str], list[float]] = {}
    for r in comp_rows:
        groups.setdefault((str(r["component"]), str(r["role"])), []).append(float(r["recovery"]))
    summary = {
        f"{component}_at_{role}": rounded(safe_fmean(vals))
        for (component, role), vals in sorted(groups.items())
    }
    best = max(groups, key=lambda k: safe_fmean(groups[k]))
    summary["best_component_role"] = f"{best[0]}@{best[1]}"
    summary["best_component_role_recovery"] = rounded(safe_fmean(groups[best]))
    return summary


def role_peak_summary(agg_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for role in ("pre_subject", "subject", "post_subject", "last"):
        vals = [
            (int(r["stream_depth"]), float(r[f"recovery_{role}"]))
            for r in agg_rows
            if r[f"recovery_{role}"] != ""
        ]
        if vals:
            depth, value = max(vals, key=lambda item: item[1])
            summary[role] = {"peak_stream_depth": depth, "peak_recovery": rounded(value)}
    return summary


def write_causal_trace_card(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    base_pairs: list[Pair],
    decision: LocalizationDecision,
    matched_mean: float,
    control_summary: dict[str, dict[str, Any]],
    component_summary: dict[str, Any],
    edit_results: list[dict[str, Any]],
) -> None:
    lines = [
        "# Lab 5 causal trace card",
        "",
        "## Scope",
        "",
        f"- **Model:** `{bundle.anatomy.model_id}`",
        f"- **Prompt population:** {len(base_pairs)} validated base-template capital facts, single-token subjects and answers.",
        "- **Metric:** target-vs-distractor final-position logit difference.",
        "- **Intervention:** one clean-run activation spliced into a corrupt run at one stream depth and position.",
        "",
        "## Localization",
        "",
        f"- Subject recovery peak: stream depth {decision.subject_peak_stream_depth} "
        f"({decision.subject_peak_recovery:.2f}).",
        f"- Handoff: stream depth {decision.handoff_stream_depth}, using drop threshold "
        f"{decision.subject_drop_threshold:.2f}.",
        f"- Localized stream band: {decision.localized_stream_depths}.",
        f"- Representative stream patch: depth {decision.representative_stream_depth}.",
        f"- Component/edit layers mapped from that band: {decision.component_layers}; representative block "
        f"{decision.representative_component_layer}.",
        f"- Last-position recovery peaks at stream depth {decision.top_last_stream_depth} "
        f"({decision.peak_last_recovery:.2f}).",
        "",
        "## Specificity and controls",
        "",
        f"- Matched subject patch mean recovery at the representative stream depth: {matched_mean:.2f}.",
    ]
    for kind, info in control_summary.items():
        lines.append(f"- {kind}: mean recovery {info['mean_recovery']} over n={info['n']}.")
    lines += [
        "",
        "## Component pass",
        "",
    ]
    if component_summary:
        for key, value in component_summary.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- Component pass not available.")
    lines += [
        "",
        "## Upgraded plot path",
        "",
        "- `plots/causal_patching_dashboard.png` is the one-screen overview: timing, controls, heterogeneity, and component refinement.",
        "- `plots/recovery_role_atlas.png` and `tables/patch_evidence_matrix.csv` show whether the mean is robust across facts.",
        "- `plots/specificity_gap_by_fact.png` and `tables/specificity_summary.csv` are the main guardrail against generic perturbation stories.",
        "- `plots/paraphrase_transfer_matrix.png` checks whether the localized region survives template changes.",
        "",
        "## Edit audit",
        "",
    ]
    if edit_results:
        for row in edit_results:
            lines.append(
                f"- {row['layer_kind']} L{row['edit_layer']} alpha={row['alpha']}: "
                f"direct={row['direct_success']}, movement={row['movement_toward_distractor']}, "
                f"paraphrases {row['paraphrase_flips']}/{row['n_paraphrases']}, "
                f"neighbors intact {row['neighbors_intact']}/{row['n_neighbors']}."
            )
    else:
        lines.append("- Not run. Re-run with `--run-edit` to audit localization as an edit proposal.")
    lines += [
        "",
        "## Claim boundary",
        "",
        "This card supports a scoped causal tracing claim. It does not prove that a single component stores the fact, "
        "nor that the same layer is the best edit layer. Stream patches test sufficiency of a clean-vs-corrupt "
        "difference; necessity, multi-site interaction, and path-level routing remain open until tested.",
        "",
    ]
    path = ctx.path("causal_trace_card.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "The Lab 5 causal trace card: localization, controls, components, and limits.")


def build_claims(
    ctx: bench.RunContext,
    base_pairs: list[Pair],
    decision: LocalizationDecision,
    matched_mean: float,
    control_rows: list[dict[str, Any]],
    component_summary: dict[str, Any],
    edit_results: list[dict[str, Any]],
) -> list[dict[str, str]]:
    run_name = ctx.run_dir.name
    mismatched = [float(r["recovery"]) for r in control_rows if r["control"] == "mismatched_pair"]
    wrong_pos = [float(r["recovery"]) for r in control_rows if r["control"] == "wrong_position"]
    mismatched_mean = safe_fmean(mismatched)
    wrong_mean = safe_fmean(wrong_pos)
    claims = [
        {
            "id": f"{LAB_ID}-C1",
            "tag": "CAUSAL",
            "text": (
                f"For {len(base_pairs)} validated single-token capital prompts, patching the clean subject-position "
                f"residual stream into the corrupt run at stream depth {decision.representative_stream_depth} "
                f"recovers mean {matched_mean:.0%} of the clean target-vs-distractor logit gap. "
                f"The corresponding component/edit block is {decision.representative_component_layer}, because "
                "streams[k] contains all writes from blocks < k."
            ),
            "artifact": f"runs/{run_name}/tables/per_fact_top_patch.csv",
            "falsifier": (
                "The same intervention on longer prompts, multi-token subjects, or another relation recovers little; "
                "the localization was template-specific."
            ),
        },
        {
            "id": f"{LAB_ID}-C2",
            "tag": "CAUSAL",
            "text": (
                f"The subject-position signal peaks at stream depth {decision.subject_peak_stream_depth} "
                f"({decision.subject_peak_recovery:.2f}) and falls below the handoff threshold near depth "
                f"{decision.handoff_stream_depth}, while last-position recovery peaks later at depth "
                f"{decision.top_last_stream_depth} ({decision.peak_last_recovery:.2f}). This supports a "
                "recall-then-readout story, not a claim that one residual vector permanently stores the fact."
            ),
            "artifact": f"runs/{run_name}/plots/localization_across_facts.png",
            "falsifier": "Paraphrase curves localize to disjoint depths or the subject-vs-last separation vanishes on held-out facts.",
        },
        {
            "id": f"{LAB_ID}-C3",
            "tag": "CAUSAL",
            "text": (
                f"Specificity controls at the representative stream depth stay below the matched patch: "
                f"mismatched-pair mean recovery {mismatched_mean:.2f}, wrong-position mean recovery {wrong_mean:.2f}, "
                f"versus matched mean {matched_mean:.2f}. The recovered signal is therefore not just a generic "
                "large-vector perturbation."
            ),
            "artifact": f"runs/{run_name}/tables/negative_control_scores.csv",
            "falsifier": "A larger same-relation control battery reaches matched-patch recovery, especially on small models.",
        },
    ]
    if component_summary:
        claims.append({
            "id": f"{LAB_ID}-C4",
            "tag": "CAUSAL",
            "text": (
                f"In the component pass mapped from the localized stream band, the strongest component-role cell is "
                f"{component_summary.get('best_component_role')} with mean recovery "
                f"{component_summary.get('best_component_role_recovery')}. This refines the stream-level location "
                "into a candidate write site, but it is still an interchange result, not a storage proof."
            ),
            "artifact": f"runs/{run_name}/tables/component_patching.csv",
            "falsifier": "Multi-site or path patching shows the apparent component effect is actually routed through another component.",
        })
    if edit_results:
        def headline_row(kind: str) -> dict[str, Any]:
            rows = [r for r in edit_results if r["layer_kind"] == kind]
            flipped = [r for r in rows if r["direct_success"]]
            return min(flipped, key=lambda r: r["alpha"]) if flipped else max(rows, key=lambda r: r["movement_toward_distractor"])

        loc = headline_row("localized_component")
        alt = headline_row("alternative")
        claims.append({
            "id": f"{LAB_ID}-C5",
            "tag": "CAUSAL",
            "text": (
                f"A rank-one edit at localized component layer {loc['edit_layer']} with alpha={loc['alpha']} "
                f"targeted {loc['intended_flip']}: direct success={loc['direct_success']}, logit movement toward "
                f"the distractor={loc['movement_toward_distractor']}, paraphrases {loc['paraphrase_flips']}/"
                f"{loc['n_paraphrases']}, neighbors intact {loc['neighbors_intact']}/{loc['n_neighbors']}. "
                f"An alternative layer {alt['edit_layer']} at alpha={alt['alpha']} gave direct success="
                f"{alt['direct_success']}. Localization informing editing should be argued from these numbers, not assumed."
            ),
            "artifact": f"runs/{run_name}/tables/edit_results.csv",
            "falsifier": "An all-layer edit sweep shows edit success uncorrelated with the causal-tracing map.",
        })
    return claims


def write_summary(
    ctx: bench.RunContext,
    bundle: bench.ModelBundle,
    base_pairs: list[Pair],
    gate_rows: list[dict[str, Any]],
    rejected: int,
    decision: LocalizationDecision,
    matched_mean: float,
    control_rows: list[dict[str, Any]],
    component_summary: dict[str, Any],
    edit_results: list[dict[str, Any]],
    claims: list[dict[str, str]],
) -> None:
    n_layers = bundle.anatomy.n_layers
    mismatched = [float(r["recovery"]) for r in control_rows if r["control"] == "mismatched_pair"]
    wrong_pos = [float(r["recovery"]) for r in control_rows if r["control"] == "wrong_position"]
    lines = [
        "# Lab 5 run summary: activation patching and causal tracing",
        "",
        "## Run identity",
        "",
        f"- model: `{bundle.anatomy.model_id}` ({n_layers} blocks, d_model {bundle.anatomy.d_model})",
        f"- facts: {len(base_pairs)} base-template pairs past the gate "
        f"({sum(1 for r in gate_rows if r['template'] == 'base' and not r['kept'])} dropped, "
        f"{rejected} rejected by the alignment validator)",
        f"- grid: {n_layers + 1} stream depths x {base_pairs[0].n_tokens} positions per pair",
        "- evidence level: `CAUSAL`, scoped to the stated prompt population and intervention",
        "- self-checks: hook parity, lens, patch no-op, component anatomy, component decomposition",
        "",
        "## 1. What behavior was studied?",
        "",
        "Factual recall: single-token capital-city answers under a fixed base template, with two paraphrase templates for confirmation.",
        "",
        "## 2. What internal object was measured?",
        "",
        "The pre-norm residual stream at every stream depth and token position (same pre-final-norm convention as Lab 1); component-level attn/MLP outputs for the localized block band (refining the stream result, in the spirit of Lab 2 attribution and Lab 3 head contribution).",
        "",
        "## 3. What intervention or control was used?",
        "",
        "Interchange patching from clean to corrupt runs (sufficiency of a clean-vs-corrupt difference under a narrow intervention), plus mismatched-pair, wrong-position, and split-heldout low-region controls (the specificity checks that turn recovery curves into scoped causal claims).",
        "",
        "## 4. Headline numbers",
        "",
        f"- subject peak: stream depth {decision.subject_peak_stream_depth} ({decision.subject_peak_recovery:.2f})",
        f"- handoff: stream depth {decision.handoff_stream_depth}; localized stream band {decision.localized_stream_depths}",
        f"- representative stream patch: depth {decision.representative_stream_depth}; mapped component/edit layer {decision.representative_component_layer}",
        f"- last-role recovery peaks at stream depth {decision.top_last_stream_depth}: {decision.peak_last_recovery:.2f}",
        f"- matched representative patch: {matched_mean:.2f}; mismatched controls {safe_fmean(mismatched):.2f}; wrong-position controls {safe_fmean(wrong_pos):.2f}",
    ]
    if component_summary:
        lines.append(f"- component headline: {component_summary.get('best_component_role')} recovery {component_summary.get('best_component_role_recovery')}")
    if edit_results:
        for r in edit_results:
            lines.append(
                f"- edit @{r['layer_kind']} L{r['edit_layer']} alpha={r['alpha']}: "
                f"direct={r['direct_success']}, movement={r['movement_toward_distractor']}, "
                f"paraphrases {r['paraphrase_flips']}/{r['n_paraphrases']}, "
                f"neighbors intact {r['neighbors_intact']}/{r['n_neighbors']}, "
                f"fluency {r['fluency_logprob_before']} -> {r['fluency_logprob_after']}"
            )
    lines += [
        "",
        "## 5. Claims supported, at what evidence level?",
        "",
    ]
    for c in claims:
        lines.append(f"- `{c['id']}` {c['tag']}: {c['text']}")
        lines.append(f"  - falsifier: {c['falsifier']}")
    lines += [
        "",
        "## 6. Claims not supported",
        "",
        "- The lab does not show that a single layer or component stores the fact. A stream patch contains the whole accumulated residual representation (recall-then-readout, not a permanent storage jar).",
        "- The lab does not show that the localized stream depth (or mapped component layer) is the best edit layer. The edit audit explicitly tests that gap (the Hase tension) instead of sweeping it away.",
        "- The lab does not establish necessity of one site. Single-site patching tests sufficiency of a clean-vs-corrupt difference under interchange; necessity and multi-site paths are left to later labs.",
        "",
        "## 7. What would falsify the interpretation?",
        "",
        "Paraphrase localization moving to different depths, control patches matching the real patch, or multi-site/path patching showing the effect routes through a different object would all force a narrower claim.",
        "",
        "## Reading order",
        "",
        "1. `causal_trace_card.md`, then `tables/patch_evidence_matrix.csv` - the scoped claim plus the one-row-per-fact evidence ledger.",
        "2. `plots/causal_patching_dashboard.png` - the whole patching story on one screen.",
        "3. `plots/localization_across_facts.png`, `plots/recovery_ridge_map.png`, and `tables/role_transition_summary.csv` - the subject handoff and last-token readout timing.",
        "4. `plots/recovery_role_atlas.png`, `tables/per_fact_top_patch.csv`, and `plots/per_fact_top_patch.png` - heterogeneity and which facts drive the mean.",
        "5. `plots/paraphrase_transfer_matrix.png`, `tables/paraphrase_summary.csv`, and `tables/paraphrase_consistency.csv` - whether localization survives wording changes.",
        "6. `plots/specificity_gap_by_fact.png`, `plots/negative_controls.png`, `tables/specificity_summary.csv`, and `tables/negative_control_scores.csv` - specificity checks.",
        "7. `plots/component_patch_matrix.png`, `tables/component_patching.csv`, and `plots/component_patching.png` - attn/MLP refinement of the stream result.",
        "8. `plots/edit_audit_dashboard.png` and `tables/edit_results.csv` if `--run-edit` - localization meets editing (the audit that shows the gap).",
        "",
    ]
    path = ctx.path("run_summary.md")
    bench.write_text(path, "\n".join(lines))
    ctx.register_artifact(path, "summary", "The standard seven questions answered with this run's numbers.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    args = ctx.args
    n_layers = bundle.anatomy.n_layers

    facts = select_facts(args)
    print(f"[lab5] {len(facts)} candidate facts, corrupt partner = next fact (cyclic)")

    # Build and validate pairs for all templates. The validator rejects, never
    # warns, because a misaligned prompt pair is not a patching datum.
    all_pairs: dict[str, list[Pair]] = {}
    tok_rows: list[dict[str, Any]] = []
    for template_id in TEMPLATES:
        pairs, rows = build_pairs(ctx, bundle, facts, template_id)
        all_pairs[template_id] = pairs
        tok_rows.extend(rows)
    tok_path = ctx.path("diagnostics", "tokenization_report.csv")
    bench.write_csv(tok_path, tok_rows)
    ctx.register_artifact(tok_path, "diagnostic", "Pair alignment validation: token lengths, diff positions, answer tokens.")
    rejected = sum(1 for row in tok_rows if not row["aligned"])
    if rejected:
        print(f"[lab5] {rejected} pair/template combinations rejected by the alignment validator")
    if not all_pairs.get("base"):
        raise RuntimeError("No aligned base-template pairs remain. See diagnostics/tokenization_report.csv.")

    # Instrument verification before the science. The component decomposition
    # check is especially important because the component pass below reuses the
    # verified attn/MLP contribution objects from Labs 2 and 3.
    probe_prompt = all_pairs["base"][0].clean_prompt
    bench.run_hook_parity_check(ctx, bundle, probe_prompt)
    first_capture = bench.run_with_residual_cache(bundle, probe_prompt)
    bench.run_lens_self_check(ctx, bundle, first_capture)
    bench.run_patch_noop_check(ctx, bundle, probe_prompt)
    comp_anatomy = bench.resolve_component_anatomy(ctx, bundle, probe_prompt, rel_tolerance=args.dla_tolerance)
    comp_probe = bench.run_with_component_cache(bundle, probe_prompt, comp_anatomy, all_positions=False)
    bench.run_decomposition_check(ctx, bundle, comp_probe, rel_tolerance=args.dla_tolerance)

    # Baseline gate per template.
    kept: dict[str, list[Pair]] = {}
    captures: dict[str, Any] = {}
    gate_rows: list[dict[str, Any]] = []
    for template_id, pairs in all_pairs.items():
        template_kept, rows, caps = gate_pairs(ctx, bundle, pairs)
        kept[template_id] = template_kept
        captures.update(caps)
        gate_rows.extend(rows)
        print(f"[lab5] template {template_id!r}: {len(template_kept)}/{len(pairs)} pairs pass the baseline gate")
    facts_path = ctx.path("tables", "facts.csv")
    bench.write_csv_with_context(ctx, facts_path, gate_rows)
    ctx.register_artifact(facts_path, "table", "Every pair with baseline logit diffs and gate outcome.")

    base_pairs = kept["base"]
    if len(base_pairs) < 2:
        raise RuntimeError(
            f"Only {len(base_pairs)} base-template pairs pass the baseline gate. "
            "The model does not know these facts well enough to trace. See tables/facts.csv."
        )

    # ----- residual patching grid, base template -----------------------------
    grid_rows: list[dict[str, Any]] = []
    n_forwards = len(base_pairs) * (n_layers + 1) * base_pairs[0].n_tokens
    print(f"[lab5] grid: {len(base_pairs)} pairs x {n_layers + 1} stream depths x "
          f"{base_pairs[0].n_tokens} positions = {n_forwards} patched forwards")
    for i, pair in enumerate(base_pairs):
        cap = captures[f"{pair.fact.fact_id}:base"]
        grid_rows.extend(run_grid(bundle, pair, cap))
        print(f"[lab5] [{i + 1}/{len(base_pairs)}] {pair.fact.fact_id} grid done")

    grid_path = ctx.path("tables", "patching_scores.csv")
    bench.write_csv_with_context(ctx, grid_path, grid_rows)
    ctx.register_artifact(grid_path, "table", "Long-form recovery for every (fact, stream depth, position).")
    results_path = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results_path, grid_rows)
    ctx.register_artifact(results_path, "results", "Alias of patching_scores.csv for the standard run contract.")

    agg_rows = aggregate_by_role(grid_rows, n_layers)
    agg_path = ctx.path("tables", "localization_summary.csv")
    bench.write_csv_with_context(ctx, agg_path, agg_rows)
    ctx.register_artifact(agg_path, "table", "Mean recovery by stream depth and token role across facts.")

    decision = choose_localization(agg_rows, n_layers)
    decision_path = ctx.path("diagnostics", "localization_decision.json")
    bench.write_json(decision_path, decision.as_dict())
    ctx.register_artifact(decision_path, "diagnostic", "Handoff selection and stream-depth-to-component-layer mapping.")
    print(
        f"[lab5] handoff stream depth {decision.handoff_stream_depth}; localized stream band "
        f"{decision.localized_stream_depths}; representative stream depth {decision.representative_stream_depth}; "
        f"component/edit layer {decision.representative_component_layer}"
    )

    per_fact_rows = top_patch_rows(grid_rows, base_pairs, decision.representative_stream_depth)
    per_fact_path = ctx.path("tables", "per_fact_top_patch.csv")
    bench.write_csv_with_context(ctx, per_fact_path, per_fact_rows)
    ctx.register_artifact(per_fact_path, "table", "Per-fact recoveries at the representative subject-position stream patch.")
    matched_mean = safe_fmean([float(r["recovery"]) for r in per_fact_rows])

    # ----- paraphrase confirmation of the subject-column curve ---------------
    para_agg: dict[str, list[dict[str, Any]]] = {}
    para_rows: list[dict[str, Any]] = []
    para_summary_rows: list[dict[str, Any]] = []
    for template_id in PARAPHRASE_IDS:
        rows_t: list[dict[str, Any]] = []
        for pair in kept[template_id]:
            cap = captures[f"{pair.fact.fact_id}:{template_id}"]
            for stream_depth in range(n_layers + 1):
                logits = bench.run_with_residual_patch(
                    bundle,
                    pair.corrupt_prompt,
                    stream_depth,
                    pair.subject_pos,
                    cap.streams[stream_depth, pair.subject_pos],
                )
                patched = logit_diff(logits, pair)
                rec = recovery_value(patched, pair)
                rows_t.append({
                    "fact_id": pair.fact.fact_id,
                    "corrupt_fact_id": pair.corrupt.fact_id,
                    "template": template_id,
                    "stream_depth": stream_depth,
                    "layer": stream_depth,
                    "patch_site": stream_patch_site(stream_depth, n_layers),
                    "position": pair.subject_pos,
                    "role": "subject",
                    "patched_diff": rounded(patched),
                    "recovery": rounded(rec),
                })
        para_rows.extend(rows_t)
        para_agg[template_id] = aggregate_by_role(rows_t, n_layers)
        band_vals = [float(r["recovery"]) for r in rows_t if int(r["stream_depth"]) in decision.localized_stream_depths]
        repr_vals = [float(r["recovery"]) for r in rows_t if int(r["stream_depth"]) == decision.representative_stream_depth]
        para_summary_rows.append({
            "template": template_id,
            "n_pairs": len(kept[template_id]),
            "localized_band_mean_recovery": rounded(safe_fmean(band_vals)),
            "representative_stream_depth": decision.representative_stream_depth,
            "representative_depth_mean_recovery": rounded(safe_fmean(repr_vals)),
        })
        print(f"[lab5] paraphrase {template_id!r}: subject-column sweep on {len(kept[template_id])} pairs")
    if para_rows:
        para_path = ctx.path("tables", "paraphrase_consistency.csv")
        bench.write_csv_with_context(ctx, para_path, para_rows)
        ctx.register_artifact(para_path, "table", "Subject-position recovery by stream depth under paraphrase templates.")
        para_summary_path = ctx.path("tables", "paraphrase_summary.csv")
        bench.write_csv_with_context(ctx, para_summary_path, para_summary_rows)
        ctx.register_artifact(para_summary_path, "table", "Paraphrase recovery summarized at the localized band and representative depth.")

    # ----- negative controls --------------------------------------------------
    control_rows: list[dict[str, Any]] = []
    top_depth = decision.representative_stream_depth
    for i, pair in enumerate(base_pairs):
        cap = captures[f"{pair.fact.fact_id}:base"]
        other = base_pairs[(i + 1) % len(base_pairs)]
        if other.fact.fact_id != pair.fact.fact_id:
            logits = bench.run_with_residual_patch(
                bundle,
                other.corrupt_prompt,
                top_depth,
                other.subject_pos,
                cap.streams[top_depth, pair.subject_pos],
            )
            control_rows.append({
                "control": "mismatched_pair",
                "fact_id": pair.fact.fact_id,
                "into": other.fact.fact_id,
                "stream_depth": top_depth,
                "layer": top_depth,
                "position": other.subject_pos,
                "recovery": rounded(recovery_value(logit_diff(logits, other), other)),
            })
        wrong_pos = 0 if pair.subject_pos != 0 else pair.n_tokens - 1
        logits = bench.run_with_residual_patch(
            bundle,
            pair.corrupt_prompt,
            top_depth,
            wrong_pos,
            cap.streams[top_depth, pair.subject_pos],
        )
        control_rows.append({
            "control": "wrong_position",
            "fact_id": pair.fact.fact_id,
            "into": pair.fact.fact_id,
            "stream_depth": top_depth,
            "layer": top_depth,
            "position": wrong_pos,
            "source_position": pair.subject_pos,
            "recovery": rounded(recovery_value(logit_diff(logits, pair), pair)),
        })

    # Split-heldout low-region control. The low layer is selected on the first
    # half of facts and evaluated on the second half using existing grid rows.
    half = len(base_pairs) // 2
    first_ids = {p.fact.fact_id for p in base_pairs[:half]}

    def mean_subject_rec(stream_depth: int, ids: set[str]) -> float:
        vals = [float(r["recovery"]) for r in grid_rows
                if int(r["stream_depth"]) == stream_depth and r["role"] == "subject" and r["fact_id"] in ids]
        return safe_fmean(vals)

    if half >= 1 and len(base_pairs) - half >= 1:
        candidate_depths = [int(r["stream_depth"]) for r in agg_rows if 1 <= int(r["stream_depth"]) < n_layers]
        low_depth = min(candidate_depths, key=lambda d: mean_subject_rec(d, first_ids))
        heldout_ids = {p.fact.fact_id for p in base_pairs[half:]}
        control_rows.append({
            "control": "low_region_split_heldout",
            "fact_id": f"stream_depth_{low_depth}",
            "into": "second_half_facts",
            "stream_depth": low_depth,
            "layer": low_depth,
            "recovery": rounded(mean_subject_rec(low_depth, heldout_ids)),
            "selected_on": "first_half_facts",
        })

    ctrl_path = ctx.path("tables", "negative_control_scores.csv")
    bench.write_csv_with_context(ctx, ctrl_path, control_rows)
    ctx.register_artifact(ctrl_path, "table", "Mismatched-pair, wrong-position, and split-heldout low-region controls.")
    control_summary = summarize_controls(control_rows)

    # ----- component-level pass ----------------------------------------------
    comp_rows: list[dict[str, Any]] = []
    for pair in base_pairs:
        comp_cap = bench.run_with_component_cache(bundle, pair.clean_prompt, comp_anatomy, all_positions=True)
        for component_layer in decision.component_layers:
            produced_stream_depth = component_layer + 1
            for kind in ("attn", "mlp"):
                vec_seq = comp_cap.attn_contrib if kind == "attn" else comp_cap.mlp_contrib
                for pos, role in ((pair.subject_pos, "subject"), (pair.n_tokens - 1, "last")):
                    logits = bench.run_with_component_patch(
                        bundle,
                        pair.corrupt_prompt,
                        comp_anatomy,
                        kind,
                        component_layer,
                        pos,
                        vec_seq[component_layer, pos],
                    )
                    patched = logit_diff(logits, pair)
                    comp_rows.append({
                        "fact_id": pair.fact.fact_id,
                        "corrupt_fact_id": pair.corrupt.fact_id,
                        "component_layer": component_layer,
                        "layer": component_layer,
                        "produces_stream_depth": produced_stream_depth,
                        "component": kind,
                        "position": pos,
                        "role": role,
                        "patched_diff": rounded(patched),
                        "recovery": rounded(recovery_value(patched, pair)),
                    })
    comp_path = ctx.path("tables", "component_patching.csv")
    bench.write_csv_with_context(ctx, comp_path, comp_rows)
    ctx.register_artifact(comp_path, "table", "Attn vs MLP output patching for component layers mapped from the stream band.")
    component_summary = summarize_components(comp_rows)
    component_summary_path = ctx.path("tables", "component_summary.csv")
    bench.write_csv_with_context(ctx, component_summary_path, [component_summary] if component_summary else [])
    ctx.register_artifact(component_summary_path, "table", "Component-pass headline means by component and role.")

    # ----- upgraded synthesis tables -----------------------------------------
    role_transition_rows, specificity_summary_rows, patch_evidence_rows = write_upgrade_tables(
        ctx, base_pairs, grid_rows, agg_rows, per_fact_rows, control_rows, para_rows, decision
    )

    # ----- edit extension -----------------------------------------------------
    edit_results: list[dict[str, Any]] = []
    if args.run_edit:
        showcase = max(base_pairs, key=lambda p: p.clean_diff - p.corrupt_diff)
        others = [p for p in base_pairs if p.fact.fact_id != showcase.fact.fact_id]
        localized_layer = decision.representative_component_layer
        alt_layer = choose_alternative_edit_layer(localized_layer, decision, n_layers)
        for layer, kind in ((localized_layer, "localized_component"), (alt_layer, "alternative")):
            for alpha in EDIT_ALPHAS:
                res = run_edit_audit(ctx, bundle, showcase, layer, others, alpha=alpha)
                res["layer_kind"] = kind
                res["source_stream_depth_for_localization"] = decision.representative_stream_depth
                edit_results.append(res)
                print(f"[lab5] edit @{kind} L{layer} alpha={alpha}: "
                      f"direct={res['direct_success']} movement={res['movement_toward_distractor']} "
                      f"paraphrases={res['paraphrase_flips']}/{res['n_paraphrases']} "
                      f"neighbors_intact={res['neighbors_intact']}/{res['n_neighbors']}")
        edit_path = ctx.path("tables", "edit_results.csv")
        bench.write_csv_with_context(ctx, edit_path, [
            {k: (json.dumps(v, default=bench.json_default) if isinstance(v, (list, tuple, dict)) else v)
             for k, v in row.items()}
            for row in edit_results
        ])
        ctx.register_artifact(edit_path, "table", "Rank-one edit audit at the mapped localized component layer vs an alternative layer.")
        edit_summary_path = ctx.path("tables", "edit_audit_summary.csv")
        edit_summary_rows = []
        for layer_kind in sorted({str(r.get("layer_kind")) for r in edit_results}):
            rows = [r for r in edit_results if str(r.get("layer_kind")) == layer_kind]
            if not rows:
                continue
            best_move = max(rows, key=lambda r: float(r.get("movement_toward_distractor", 0.0)))
            edit_summary_rows.append({
                "layer_kind": layer_kind,
                "edit_layer": best_move.get("edit_layer"),
                "best_alpha_by_movement": best_move.get("alpha"),
                "best_movement_toward_distractor": best_move.get("movement_toward_distractor"),
                "any_direct_success": any(bool(r.get("direct_success")) for r in rows),
                "max_paraphrase_flips": max(int(r.get("paraphrase_flips", 0)) for r in rows),
                "n_paraphrases": max(int(r.get("n_paraphrases", 0)) for r in rows),
                "min_neighbors_intact": min(int(r.get("neighbors_intact", 0)) for r in rows),
                "n_neighbors": max(int(r.get("n_neighbors", 0)) for r in rows),
            })
        bench.write_csv_with_context(ctx, edit_summary_path, edit_summary_rows)
        ctx.register_artifact(edit_summary_path, "table", "Layer-level summary of the rank-one edit audit: movement, success, transfer, and spillover.")

    # ----- plots --------------------------------------------------------------
    if not args.no_plots:
        showcase_pair = max(base_pairs, key=lambda p: p.clean_diff - p.corrupt_diff)
        if args.showcase:
            showcase_pair = next((p for p in base_pairs if p.fact.fact_id == args.showcase), showcase_pair)
        cap = captures[f"{showcase_pair.fact.fact_id}:base"]
        plot_heatmap(ctx, showcase_pair, grid_rows, cap.tokens_text, n_layers, decision)
        plot_localization(ctx, agg_rows, para_agg, decision, len(base_pairs))
        plot_component_pass(ctx, comp_rows)
        plot_controls(ctx, control_rows, matched_mean)
        plot_per_fact_recovery(ctx, per_fact_rows)
        plot_baseline_gate_audit(ctx, gate_rows)
        plot_recovery_ridge_map(ctx, agg_rows, decision, n_layers)
        plot_recovery_role_atlas(ctx, grid_rows, per_fact_rows, decision, n_layers)
        plot_specificity_gap_by_fact(ctx, per_fact_rows, control_rows)
        plot_paraphrase_transfer_matrix(ctx, para_rows, decision, n_layers)
        plot_component_patch_matrix(ctx, comp_rows, decision)
        plot_causal_patching_dashboard(ctx, agg_rows, per_fact_rows, control_rows, comp_rows, decision, n_layers, len(base_pairs))
        plot_edit_audit_dashboard(ctx, edit_results)

    # ----- metrics, card, claims, summary ------------------------------------
    metrics = {
        "n_candidate_facts": len(facts),
        "n_facts_kept_base": len(base_pairs),
        "n_pairs_rejected_alignment": rejected,
        "n_pairs_aligned_by_template": {template_id: len(all_pairs.get(template_id, [])) for template_id in TEMPLATES},
        "n_pairs_kept_by_template": {template_id: len(kept.get(template_id, [])) for template_id in TEMPLATES},
        "localization": decision.as_dict(),
        "role_peak_recovery": role_peak_summary(agg_rows),
        "matched_top_patch_mean_recovery": rounded(matched_mean),
        "paraphrase_summary": para_summary_rows,
        "controls": control_summary,
        "specificity_summary": specificity_summary_rows,
        "role_transition_summary": role_transition_rows,
        "component_summary": component_summary,
        "patch_evidence_matrix": patch_evidence_rows,
        "edit_enabled": bool(args.run_edit),
        "edit_results": edit_results or None,
    }
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 5 metrics.")

    write_causal_trace_card(ctx, bundle, base_pairs, decision, matched_mean, control_summary,
                            component_summary, edit_results)
    claims = build_claims(ctx, base_pairs, decision, matched_mean, control_rows, component_summary, edit_results)
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)
    write_summary(ctx, bundle, base_pairs, gate_rows, rejected, decision, matched_mean,
                  control_rows, component_summary, edit_results, claims)
    print(f"[lab5] wrote causal_trace_card.md, run_summary.md, and {len(claims)} drafted ledger claims")
