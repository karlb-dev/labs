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

Rigor notes that this file tries hard to keep visible:

* Clean/corrupt prompts differ in exactly one tokenized subject position. The
  validator rejects misaligned pairs, because position alignment is the
  quiet indexing bug that ruins patching experiments while leaving pretty
  heatmaps behind.
* Patching streams[k] means replacing the bench's pre-norm residual stream
  after k blocks, implemented as the input to block k for k < L and as the
  input to the final norm for k = L. The patch no-op check verifies this
  convention before any scientific measurement.
* Stream depths and component layers are deliberately separated. A stream
  patch at depth k contains everything written by blocks < k. If a later
  component or edit is tested, it should usually target block k-1, not block
  k. The lab writes this mapping into localization_decision.json.
* One pair is a demo. Causal tracing aggregates over validated facts, role
  curves, paraphrase confirmation, negative controls, and component-level
  patching.
* The optional rank-one edit is an audit of the localization claim, not a
  promise that localization predicts editability.

Evidence level: CAUSAL, scoped. Every claim names the intervention, metric,
and prompt population.
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
    "small": 6,
    "medium": 10,
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
# reject facts. That is evidence, not a nuisance.
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

    Subject recovery at stream depth 0 is token substitution. It is useful as
    an instrument sanity check but not as a mechanistic localization. We
    therefore choose from depths 1..L-1.
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
    """Apply a small rank-one edit and audit its direct and ripple effects."""
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
        "The pre-norm residual stream at every stream depth and token position; component-level attn/MLP outputs for the localized block band.",
        "",
        "## 3. What intervention or control was used?",
        "",
        "Interchange patching from clean to corrupt runs, plus mismatched-pair, wrong-position, and split-heldout low-region controls.",
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
        "- The lab does not show that a single layer or component stores the fact. A stream patch contains the whole accumulated residual representation.",
        "- The lab does not show that the localized stream depth is the best edit layer. The edit audit tests that gap instead of sweeping it away.",
        "- The lab does not establish necessity of one site. Single-site patching tests sufficiency of a clean-vs-corrupt difference.",
        "",
        "## 7. What would falsify the interpretation?",
        "",
        "Paraphrase localization moving to different depths, control patches matching the real patch, or multi-site/path patching showing the effect routes through a different object would all force a narrower claim.",
        "",
        "## Reading order",
        "",
        "1. `causal_trace_card.md` - the deliverable card.",
        "2. `plots/localization_across_facts.png` - the subject-vs-last handoff story.",
        "3. `plots/patching_heatmap_<fact>.png` - one pair, token-labeled.",
        "4. `plots/negative_controls.png` and `tables/negative_control_scores.csv` - specificity checks.",
        "5. `tables/per_fact_top_patch.csv` and `plots/per_fact_top_patch.png` - which facts drove the average.",
        "6. `tables/component_patching.csv` and `plots/component_patching.png` - attn/MLP refinement.",
        "7. `tables/edit_results.csv` if `--run-edit` - localization meets editing.",
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
        "component_summary": component_summary,
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
