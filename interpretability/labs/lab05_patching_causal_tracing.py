"""Lab 5: Activation patching and causal tracing.

The course's first full causal-intervention lab. Labs 1-4 kept deferring to
this one: Lab 2's "attribution is not causation", Lab 3's indirect-path gap,
Lab 4's "decodable is not used" — patching is the tool that cashes those
checks.

Method: interchange interventions. Run a CORRUPT prompt ("The capital of
Germany is"), splice in one activation from the CLEAN run ("The capital of
France is") at one (layer, position), and measure how much of the clean
behavior returns:

    recovery = (patched_diff - corrupt_diff) / (clean_diff - corrupt_diff)

where diff = logit(Paris) - logit(Berlin) at the final position. recovery 1.0
means that single activation carried everything the corrupt run was missing;
0.0 means it carried nothing the readout could use.

Design decisions that carry the rigor:

* Clean/corrupt pairs differ in EXACTLY ONE single-token subject, so token
  positions align by construction — misaligned positions are the field's #1
  silent patching bug, and this lab refuses to let students meet it unarmed.
* The bench's patch no-op self-check runs before any science: self-patching
  must be bit-exact identity, or the convention is broken and every heatmap
  after it would be a well-rendered lie.
* One pair is a demo; CAUSAL TRACING is the aggregation: recovery by
  (layer, token role) across a dataset of facts, paraphrase confirmation of
  the top region, and negative controls (mismatched pairs, wrong positions,
  low-region stability on held-out facts).
* Facts the model does not know are dropped LOUDLY (baseline gate with
  margins, counts reported).

Extension (--run-edit): the localization claim, made permanent. A rank-one
edit writes the corrupt fact's MLP output for the clean fact's key at the
student's localized layer — then the audit asks the questions editing papers
ask: direct success, paraphrase generalization, neighbor spillover, unrelated
damage, fluency. Run at the localized layer AND an alternative layer, because
Hase et al. found localization often fails to predict the best editing layer;
explaining that tension (not resolving it) is the assignment.

Evidence level: CAUSAL, scoped. Every claim names its intervention and its
prompt population.
"""

from __future__ import annotations

import dataclasses
import statistics
from typing import Any

import interp_bench as bench

LAB_ID = "L05"

# Baseline gate margins (logits). A fact is usable only if the clean run
# prefers the target and the corrupt run prefers the distractor, both by a
# margin that makes the recovery denominator meaningful.
CLEAN_MARGIN = 0.5
CORRUPT_MARGIN = 0.5

# Templates. subject_index is the token position of {X}; verified at runtime
# against the tokenizer (constant prompt length across all single-token
# subjects was checked for both course tokenizers at authoring time).
TEMPLATES = {
    "base": {"text": "The capital of {X} is", "subject_index": 3},
    "para_city": {"text": "The capital city of {X} is", "subject_index": 4},
    "para_in": {"text": "In {X}, the capital is", "subject_index": 1},
}
PARAPHRASE_IDS = ("para_city", "para_in")


@dataclasses.dataclass(frozen=True)
class Fact:
    fact_id: str
    subject: str      # country, single token with leading space
    target: str       # capital, single token with leading space


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

    fact: Fact            # the clean fact (its target is the patching target)
    corrupt: Fact         # the corrupt fact (its target is the distractor)
    template_id: str
    clean_prompt: str
    corrupt_prompt: str
    target_id: int
    distractor_id: int
    subject_pos: int
    n_tokens: int
    clean_diff: float = 0.0
    corrupt_diff: float = 0.0


def role_of(position: int, subject_pos: int, n_tokens: int) -> str:
    if position == subject_pos:
        return "subject"
    if position == n_tokens - 1:
        return "last"
    if position < subject_pos:
        return "pre_subject"
    return "post_subject"


def build_pairs(
    ctx: bench.RunContext, bundle: bench.ModelBundle, facts: list[Fact], template_id: str
) -> tuple[list[Pair], list[dict[str, Any]]]:
    """Build aligned clean/corrupt pairs (corrupt partner = next fact,
    cyclically) and run the tokenization + alignment validator. The validator
    REJECTS rather than warns: a misaligned pair never reaches the grid."""
    tokenizer = bundle.tokenizer
    template = TEMPLATES[template_id]
    pairs: list[Pair] = []
    rows: list[dict[str, Any]] = []
    for i, fact in enumerate(facts):
        corrupt = facts[(i + 1) % len(facts)]
        clean_prompt = template["text"].format(X=fact.subject)
        corrupt_prompt = template["text"].format(X=corrupt.subject)
        t_ids = tokenizer.encode(" " + fact.target, add_special_tokens=False)
        d_ids = tokenizer.encode(" " + corrupt.target, add_special_tokens=False)
        c_tok = tokenizer.encode(clean_prompt, add_special_tokens=False)
        k_tok = tokenizer.encode(corrupt_prompt, add_special_tokens=False)
        problems = []
        if len(t_ids) != 1:
            problems.append(f"target {fact.target!r} is {len(t_ids)} tokens")
        if len(d_ids) != 1:
            problems.append(f"distractor {corrupt.target!r} is {len(d_ids)} tokens")
        if len(c_tok) != len(k_tok):
            problems.append(f"clean/corrupt token lengths differ ({len(c_tok)} vs {len(k_tok)})")
        else:
            diff_positions = [p for p in range(len(c_tok)) if c_tok[p] != k_tok[p]]
            if diff_positions != [template["subject_index"]]:
                problems.append(f"prompts differ at positions {diff_positions}, expected only "
                                f"[{template['subject_index']}]")
        rows.append({
            "fact_id": fact.fact_id, "template": template_id,
            "clean_prompt": clean_prompt, "corrupt_prompt": corrupt_prompt,
            "target": fact.target, "distractor": corrupt.target,
            "aligned": not problems, "problems": "; ".join(problems),
        })
        if not problems:
            pairs.append(Pair(fact, corrupt, template_id, clean_prompt, corrupt_prompt,
                              t_ids[0], d_ids[0], template["subject_index"], len(c_tok)))
    return pairs, rows


def logit_diff(logits: Any, pair: Pair) -> float:
    return float(logits[pair.target_id] - logits[pair.distractor_id])


def gate_pairs(
    ctx: bench.RunContext, bundle: bench.ModelBundle, pairs: list[Pair]
) -> tuple[list[Pair], list[dict[str, Any]], dict[str, Any]]:
    """Baseline gate: the model must actually know both facts in the pair."""
    kept: list[Pair] = []
    rows: list[dict[str, Any]] = []
    captures: dict[str, Any] = {}
    for pair in pairs:
        clean_cap = bench.run_with_residual_cache(bundle, pair.clean_prompt)
        corrupt_logits = bench.run_with_residual_cache(bundle, pair.corrupt_prompt).final_logits_last
        pair.clean_diff = logit_diff(clean_cap.final_logits_last, pair)
        pair.corrupt_diff = logit_diff(corrupt_logits, pair)
        ok = pair.clean_diff > CLEAN_MARGIN and pair.corrupt_diff < -CORRUPT_MARGIN
        rows.append({
            "fact_id": pair.fact.fact_id, "template": pair.template_id,
            "clean_diff": round(pair.clean_diff, 4), "corrupt_diff": round(pair.corrupt_diff, 4),
            "kept": ok,
            "drop_reason": "" if ok else
            (f"clean_diff {pair.clean_diff:.2f} <= {CLEAN_MARGIN}" if pair.clean_diff <= CLEAN_MARGIN
             else f"corrupt_diff {pair.corrupt_diff:.2f} >= -{CORRUPT_MARGIN}"),
        })
        if ok:
            kept.append(pair)
            captures[f"{pair.fact.fact_id}:{pair.template_id}"] = clean_cap
    return kept, rows, captures


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


def run_grid(
    bundle: bench.ModelBundle, pair: Pair, clean_capture: Any
) -> list[dict[str, Any]]:
    """Patch every (stream layer, position) of the clean run into the corrupt
    run. Returns long-form rows with recovery scores."""
    n_layers = bundle.anatomy.n_layers
    denom = pair.clean_diff - pair.corrupt_diff
    rows = []
    for layer in range(n_layers + 1):
        for pos in range(pair.n_tokens):
            logits = bench.run_with_residual_patch(
                bundle, pair.corrupt_prompt, layer, pos, clean_capture.streams[layer, pos]
            )
            patched = logit_diff(logits, pair)
            rows.append({
                "fact_id": pair.fact.fact_id,
                "template": pair.template_id,
                "layer": layer,
                "position": pos,
                "role": role_of(pos, pair.subject_pos, pair.n_tokens),
                "patched_diff": round(patched, 4),
                "recovery": round((patched - pair.corrupt_diff) / denom, 4),
            })
    return rows


def aggregate_by_role(grid_rows: list[dict[str, Any]], n_layers: int) -> list[dict[str, Any]]:
    out = []
    roles = ("pre_subject", "subject", "post_subject", "last")
    for layer in range(n_layers + 1):
        row: dict[str, Any] = {"layer": layer}
        for role in roles:
            vals = [r["recovery"] for r in grid_rows if r["layer"] == layer and r["role"] == role]
            row[f"recovery_{role}"] = round(statistics.fmean(vals), 4) if vals else ""
            row[f"n_{role}"] = len(vals)
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_heatmap(ctx: bench.RunContext, pair: Pair, grid_rows: list[dict[str, Any]],
                 tokens_text: list[str], n_layers: int) -> None:
    """The canonical causal-tracing visual: one pair, layer x position."""
    import matplotlib.pyplot as plt
    import numpy as np

    grid = np.zeros((n_layers + 1, pair.n_tokens))
    for r in grid_rows:
        if r["fact_id"] == pair.fact.fact_id and r["template"] == pair.template_id:
            grid[r["layer"], r["position"]] = r["recovery"]
    fig, ax = plt.subplots(figsize=(6.4, 8.5))
    im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0, origin="lower")
    labels = [bench.visible_token(t) for t in tokens_text]
    labels[pair.subject_pos] += "\n(subject)"
    labels[-1] += "\n(last)"
    ax.set_xticks(range(pair.n_tokens))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("stream layer patched (streams[k] = input to block k)")
    ax.set_xlabel("position patched")
    ax.set_title(f"Recovery: clean {pair.fact.subject!r} patched into corrupt {pair.corrupt.subject!r}\n"
                 f"(clean diff {pair.clean_diff:+.2f}, corrupt {pair.corrupt_diff:+.2f})")
    fig.colorbar(im, ax=ax, fraction=0.04, label="recovery of clean logit diff")
    fig.tight_layout()
    bench.save_figure(ctx, fig, f"patching_heatmap_{bench.sanitize_tag(pair.fact.fact_id)}.png",
                      "Layer-by-position patching recovery for the showcase pair.")


def plot_localization(ctx: bench.RunContext, agg_rows: list[dict[str, Any]],
                      para_agg: dict[str, list[dict[str, Any]]], n_facts: int) -> None:
    """Recovery by layer per token role, aggregated across facts — the
    causal-tracing summary. Paraphrase curves (subject role) overlay dashed."""
    fig, ax = bench.new_figure(figsize=(10.0, 6.0))
    colors = {"pre_subject": "tab:gray", "subject": "tab:red",
              "post_subject": "tab:olive", "last": "tab:blue"}
    for role, color in colors.items():
        xs = [r["layer"] for r in agg_rows if r[f"recovery_{role}"] != ""]
        ys = [r[f"recovery_{role}"] for r in agg_rows if r[f"recovery_{role}"] != ""]
        ax.plot(xs, ys, linewidth=2.4, color=color, label=f"{role} (base template)")
    for tid, rows in para_agg.items():
        xs = [r["layer"] for r in rows if r["recovery_subject"] != ""]
        ys = [r["recovery_subject"] for r in rows if r["recovery_subject"] != ""]
        ax.plot(xs, ys, linewidth=1.4, color="tab:red", linestyle="--", alpha=0.7,
                label=f"subject ({tid})")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axhline(1, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("stream layer patched")
    ax.set_ylabel("mean recovery")
    ax.set_title(f"Causal tracing across {n_facts} facts: where patching recovers the answer")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "localization_across_facts.png",
                      "Mean patching recovery by layer and token role, with paraphrase overlays.")


def plot_component_pass(ctx: bench.RunContext, comp_rows: list[dict[str, Any]]) -> None:
    if not comp_rows:
        return
    import numpy as np

    fig, ax = bench.new_figure(figsize=(9.0, 5.2))
    layers = sorted({r["layer"] for r in comp_rows})
    width = 0.2
    for i, (kind, role, color) in enumerate((
        ("mlp", "subject", "tab:orange"), ("attn", "subject", "tab:blue"),
        ("mlp", "last", "peru"), ("attn", "last", "steelblue"),
    )):
        vals = []
        for layer in layers:
            sel = [r["recovery"] for r in comp_rows
                   if r["layer"] == layer and r["component"] == kind and r["role"] == role]
            vals.append(statistics.fmean(sel) if sel else 0.0)
        ax.bar(np.arange(len(layers)) + (i - 1.5) * width, vals, width=width,
               color=color, label=f"{kind} @ {role}")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("layer (top band by subject-role recovery)")
    ax.set_ylabel("mean recovery")
    ax.set_title("Component-level patching in the top band: who carries the fact?")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "component_patching.png",
                      "Attention vs MLP output patching at subject/last positions in the top band.")


def plot_controls(ctx: bench.RunContext, control_rows: list[dict[str, Any]],
                  matched_mean: float) -> None:
    if not control_rows:
        return
    fig, ax = bench.new_figure(figsize=(7.5, 5.0))
    kinds = ["matched (top patch)"] + sorted({r["control"] for r in control_rows})
    means = [matched_mean]
    for kind in kinds[1:]:
        means.append(statistics.fmean(r["recovery"] for r in control_rows if r["control"] == kind))
    bars = ax.bar(range(len(kinds)), means,
                  color=["tab:green"] + ["tab:gray"] * (len(kinds) - 1))
    ax.bar_label(bars, fmt="%.2f", fontsize=9)
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(kinds, rotation=12, ha="right", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("mean recovery")
    ax.set_title("The top patch vs its negative controls")
    bench.save_figure(ctx, fig, "negative_controls.png",
                      "Matched top-patch recovery against mismatched-pair and wrong-position controls.")


# ---------------------------------------------------------------------------
# Edit extension
# ---------------------------------------------------------------------------


def capture_down_proj_io(bundle: bench.ModelBundle, prompt: str, layer: int, position: int) -> tuple[Any, Any]:
    """(down_proj input, down_proj output) at one position, float32 cpu."""
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
        tokenizer = bundle.tokenizer
        encoded = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            bundle.model(input_ids=encoded["input_ids"].to(bundle.input_device), use_cache=False)
    finally:
        h1.remove()
        h2.remove()
    return store["in"], store["out"]


def top1_text(bundle: bench.ModelBundle, prompt: str) -> str:
    import torch

    tokenizer = bundle.tokenizer
    encoded = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = bundle.model(input_ids=encoded["input_ids"].to(bundle.input_device), use_cache=False)
    return tokenizer.decode([int(out.logits[0, -1].argmax())])


def prompt_logit_diff(bundle: bench.ModelBundle, prompt: str, pair: Pair) -> float:
    import torch

    tokenizer = bundle.tokenizer
    encoded = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = bundle.model(input_ids=encoded["input_ids"].to(bundle.input_device), use_cache=False)
    return logit_diff(bench.tensor_cpu_float(out.logits[0, -1]), pair)


def mean_logprob(bundle: bench.ModelBundle, text: str) -> float:
    import torch

    tokenizer = bundle.tokenizer
    encoded = tokenizer(text, return_tensors="pt")
    ids = encoded["input_ids"].to(bundle.input_device)
    with torch.no_grad():
        out = bundle.model(input_ids=ids, use_cache=False)
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
    """Apply 'the patch made permanent' at edit_layer and audit it.

    The edit: for the clean subject's down-projection key at edit_layer,
    output what the corrupt subject's MLP outputs (scaled by ``alpha``) — a
    rank-one weight change that should flip France->Berlin IFF the localized
    MLP actually carries the recoverable fact. alpha=1 is the literal
    patch-made-permanent; the dose sweep above 1 measures how much harder a
    weight edit must push than an activation patch, which is itself data
    about how distributed the fact is. Everything else is the audit."""
    key_clean, out_clean = capture_down_proj_io(bundle, pair.clean_prompt, edit_layer, pair.subject_pos)
    _, out_corrupt = capture_down_proj_io(bundle, pair.corrupt_prompt, edit_layer, pair.subject_pos)
    delta_v = alpha * (out_corrupt - out_clean)

    paraphrases = [TEMPLATES[tid]["text"].format(X=pair.fact.subject) for tid in PARAPHRASE_IDS]
    neighbors = [(p, TEMPLATES["base"]["text"].format(X=p.fact.subject)) for p in other_pairs[:3]]
    fluency_before = mean_logprob(bundle, FLUENCY_TEXT)
    # "Intact" means unchanged from the model's own pre-edit behavior, not
    # from the gold answer — the audit measures what the edit broke, not what
    # the model never knew.
    neigh_before = [top1_text(bundle, prompt) for _, prompt in neighbors]

    with bench.temporary_rank_one_edit(bundle, edit_layer, key_clean, delta_v):
        direct_after = top1_text(bundle, pair.clean_prompt)
        direct_diff_after = prompt_logit_diff(bundle, pair.clean_prompt, pair)
        para_after = [top1_text(bundle, p) for p in paraphrases]
        neigh_after = [(p.fact.fact_id, top1_text(bundle, prompt), before)
                       for (p, prompt), before in zip(neighbors, neigh_before)]
        fluency_after = mean_logprob(bundle, FLUENCY_TEXT)

    return {
        "edit_layer": edit_layer,
        "alpha": alpha,
        "fact_id": pair.fact.fact_id,
        "intended_flip": f"{pair.fact.target} -> {pair.corrupt.target}",
        "direct_success": direct_after.strip() == pair.corrupt.target,
        "direct_top1_after": direct_after,
        # Movement matters even without a flip: the edit can shrink the
        # target-vs-distractor gap substantially and still not cross zero.
        "direct_logit_diff_before": round(pair.clean_diff, 4),
        "direct_logit_diff_after": round(direct_diff_after, 4),
        "paraphrase_flips": sum(1 for t in para_after if t.strip() == pair.corrupt.target),
        "n_paraphrases": len(para_after),
        "paraphrase_top1_after": para_after,
        "neighbors_intact": sum(1 for _, got, before in neigh_after if got == before),
        "n_neighbors": len(neigh_after),
        "neighbor_top1_after": [(fid, got) for fid, got, _ in neigh_after],
        "fluency_logprob_before": round(fluency_before, 4),
        "fluency_logprob_after": round(fluency_after, 4),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    args = ctx.args
    n_layers = bundle.anatomy.n_layers

    facts = list(FACTS)
    if args.max_examples > 0:
        facts = facts[: args.max_examples]
    print(f"[lab5] {len(facts)} candidate facts, corrupt partner = next fact (cyclic)")

    # Build + validate pairs for all templates (validator rejects, never warns).
    all_pairs: dict[str, list[Pair]] = {}
    tok_rows: list[dict[str, Any]] = []
    for tid in TEMPLATES:
        pairs, rows = build_pairs(ctx, bundle, facts, tid)
        all_pairs[tid] = pairs
        tok_rows.extend(rows)
    tok_path = ctx.path("diagnostics", "tokenization_report.csv")
    bench.write_csv(tok_path, tok_rows)
    ctx.register_artifact(tok_path, "diagnostic", "Pair alignment validation: token lengths, diff positions, answers.")
    rejected = sum(1 for r in tok_rows if not r["aligned"])
    if rejected:
        print(f"[lab5] {rejected} pair/template combinations REJECTED by the alignment validator")

    # Instrument verification before any science.
    probe_prompt = all_pairs["base"][0].clean_prompt
    bench.run_hook_parity_check(ctx, bundle, probe_prompt)
    first_capture = bench.run_with_residual_cache(bundle, probe_prompt)
    bench.run_lens_self_check(ctx, bundle, first_capture)
    bench.run_patch_noop_check(ctx, bundle, probe_prompt)
    comp_anatomy = bench.resolve_component_anatomy(ctx, bundle, probe_prompt, rel_tolerance=args.dla_tolerance)

    # Baseline gate per template.
    kept: dict[str, list[Pair]] = {}
    captures: dict[str, Any] = {}
    gate_rows: list[dict[str, Any]] = []
    for tid, pairs in all_pairs.items():
        k, rows, caps = gate_pairs(ctx, bundle, pairs)
        kept[tid] = k
        captures.update(caps)
        gate_rows.extend(rows)
        print(f"[lab5] template {tid!r}: {len(k)}/{len(pairs)} pairs pass the baseline gate")
    facts_path = ctx.path("tables", "facts.csv")
    bench.write_csv(facts_path, gate_rows)
    ctx.register_artifact(facts_path, "table", "Every pair with baseline logit diffs and gate outcome.")
    base_pairs = kept["base"]
    if len(base_pairs) < 2:
        raise RuntimeError(
            f"Only {len(base_pairs)} base-template pairs pass the baseline gate — the model "
            "does not know these facts well enough to trace. See tables/facts.csv."
        )

    # ----- the patching grid (base template) --------------------------------
    grid_rows: list[dict[str, Any]] = []
    n_forwards = len(base_pairs) * (n_layers + 1) * base_pairs[0].n_tokens
    print(f"[lab5] grid: {len(base_pairs)} pairs x {n_layers + 1} layers x "
          f"{base_pairs[0].n_tokens} positions = {n_forwards} patched forwards")
    for i, pair in enumerate(base_pairs):
        cap = captures[f"{pair.fact.fact_id}:base"]
        grid_rows.extend(run_grid(bundle, pair, cap))
        print(f"[lab5] [{i + 1}/{len(base_pairs)}] {pair.fact.fact_id} grid done")
    grid_path = ctx.path("tables", "patching_scores.csv")
    bench.write_csv(grid_path, grid_rows)
    ctx.register_artifact(grid_path, "table", "Long-form recovery for every (fact, layer, position).")

    agg_rows = aggregate_by_role(grid_rows, n_layers)
    agg_path = ctx.path("tables", "localization_summary.csv")
    bench.write_csv(agg_path, agg_rows)
    ctx.register_artifact(agg_path, "table", "Mean recovery by (layer, token role) across facts.")

    # The informative object with substitution corruption is NOT the peak of
    # subject-position recovery — at layer 0 that patch literally swaps the
    # subject token embedding, so recovery 1.0 is a tautology. The science is
    # the HANDOFF: the layer where subject-position recovery collapses
    # because the fact has been read out of the subject position and moved
    # toward the readout (cf. ROME's causal tracing). The localized band is
    # the last few layers BEFORE the handoff — where the subject stream still
    # causally carries the fact and an MLP edit could still matter.
    subj_by_layer = {r["layer"]: r["recovery_subject"] for r in agg_rows
                     if r["recovery_subject"] != "" and r["layer"] < n_layers}
    handoff_layer = next(
        (k for k in sorted(subj_by_layer) if subj_by_layer[k] < 0.5),
        n_layers,
    )
    top_layers = [k for k in range(max(0, handoff_layer - 3), handoff_layer)] or [0]
    last_by_layer = {r["layer"]: r["recovery_last"] for r in agg_rows
                     if r["recovery_last"] != "" and r["layer"] < n_layers}
    top_last_layer = max(last_by_layer, key=last_by_layer.get)
    print(f"[lab5] subject-recovery handoff at layer {handoff_layer}; localized band {top_layers} "
          f"(recovery {[round(subj_by_layer[k], 2) for k in top_layers]}); "
          f"top last-role layer: {top_last_layer} ({last_by_layer[top_last_layer]:.2f})")

    # ----- paraphrase confirmation of the top region -------------------------
    para_agg: dict[str, list[dict[str, Any]]] = {}
    para_rows: list[dict[str, Any]] = []
    for tid in PARAPHRASE_IDS:
        rows_t: list[dict[str, Any]] = []
        for pair in kept[tid]:
            cap = captures[f"{pair.fact.fact_id}:{tid}"]
            denom = pair.clean_diff - pair.corrupt_diff
            for layer in range(n_layers + 1):
                logits = bench.run_with_residual_patch(
                    bundle, pair.corrupt_prompt, layer, pair.subject_pos,
                    cap.streams[layer, pair.subject_pos])
                rec = (logit_diff(logits, pair) - pair.corrupt_diff) / denom
                rows_t.append({"fact_id": pair.fact.fact_id, "template": tid, "layer": layer,
                               "position": pair.subject_pos, "role": "subject",
                               "recovery": round(rec, 4)})
        para_rows.extend(rows_t)
        para_agg[tid] = aggregate_by_role(rows_t, n_layers)
        print(f"[lab5] paraphrase {tid!r}: subject-column sweep on {len(kept[tid])} pairs")
    if para_rows:
        para_path = ctx.path("tables", "paraphrase_consistency.csv")
        bench.write_csv(para_path, para_rows)
        ctx.register_artifact(para_path, "table", "Subject-position recovery by layer under paraphrase templates.")

    # ----- negative controls --------------------------------------------------
    control_rows: list[dict[str, Any]] = []
    top_layer = top_layers[0]
    for i, pair in enumerate(base_pairs):
        cap = captures[f"{pair.fact.fact_id}:base"]
        # Control 1: mismatched pair — patch THIS fact's clean subject vector
        # into a DIFFERENT pair's corrupt run at the same (layer, position).
        other = base_pairs[(i + 1) % len(base_pairs)]
        if other.fact.fact_id != pair.fact.fact_id:
            logits = bench.run_with_residual_patch(
                bundle, other.corrupt_prompt, top_layer, other.subject_pos,
                cap.streams[top_layer, pair.subject_pos])
            denom = other.clean_diff - other.corrupt_diff
            control_rows.append({
                "control": "mismatched_pair", "fact_id": pair.fact.fact_id,
                "into": other.fact.fact_id, "layer": top_layer,
                "recovery": round((logit_diff(logits, other) - other.corrupt_diff) / denom, 4),
            })
        # Control 2: wrong position — the clean SUBJECT vector patched at
        # position 0 of the matched corrupt run.
        logits = bench.run_with_residual_patch(
            bundle, pair.corrupt_prompt, top_layer, 0, cap.streams[top_layer, pair.subject_pos])
        denom = pair.clean_diff - pair.corrupt_diff
        control_rows.append({
            "control": "wrong_position", "fact_id": pair.fact.fact_id, "into": pair.fact.fact_id,
            "layer": top_layer,
            "recovery": round((logit_diff(logits, pair) - pair.corrupt_diff) / denom, 4),
        })
    # Control 3 (no new forwards): the lowest-recovery subject band found on
    # the first half of facts must stay low on the second half.
    half = len(base_pairs) // 2
    first_ids = {p.fact.fact_id for p in base_pairs[:half]}
    def mean_subject_rec(layer: int, ids: set) -> float:
        vals = [r["recovery"] for r in grid_rows
                if r["layer"] == layer and r["role"] == "subject" and
                (r["fact_id"] in ids)]
        return statistics.fmean(vals) if vals else 0.0
    if half >= 1 and len(base_pairs) - half >= 1:
        block_layers = [r["layer"] for r in agg_rows if r["layer"] < n_layers]
        low_layer = min(block_layers, key=lambda k: mean_subject_rec(k, first_ids))
        held_out = {p.fact.fact_id for p in base_pairs[half:]}
        control_rows.append({
            "control": "low_region_held_out", "fact_id": f"layer_{low_layer}",
            "into": "second_half_facts", "layer": low_layer,
            "recovery": round(mean_subject_rec(low_layer, held_out), 4),
        })
    ctrl_path = ctx.path("tables", "negative_control_scores.csv")
    bench.write_csv(ctrl_path, control_rows)
    ctx.register_artifact(ctrl_path, "table", "Mismatched-pair, wrong-position, and held-out-low-region controls.")
    matched_mean = statistics.fmean(
        r["recovery"] for r in grid_rows if r["layer"] == top_layer and r["role"] == "subject")

    # ----- component-level pass in the top band -------------------------------
    comp_rows: list[dict[str, Any]] = []
    for pair in base_pairs:
        comp_cap = bench.run_with_component_cache(bundle, pair.clean_prompt, comp_anatomy,
                                                  all_positions=True)
        denom = pair.clean_diff - pair.corrupt_diff
        for layer in top_layers:
            for kind in ("attn", "mlp"):
                vec_seq = comp_cap.attn_contrib if kind == "attn" else comp_cap.mlp_contrib
                for pos, role in ((pair.subject_pos, "subject"), (pair.n_tokens - 1, "last")):
                    logits = bench.run_with_component_patch(
                        bundle, pair.corrupt_prompt, comp_anatomy, kind, layer, pos,
                        vec_seq[layer, pos])
                    comp_rows.append({
                        "fact_id": pair.fact.fact_id, "layer": layer, "component": kind,
                        "position": pos, "role": role,
                        "recovery": round((logit_diff(logits, pair) - pair.corrupt_diff) / denom, 4),
                    })
    comp_path = ctx.path("tables", "component_patching.csv")
    bench.write_csv(comp_path, comp_rows)
    ctx.register_artifact(comp_path, "table", "Attn vs MLP output patching in the top layer band.")

    # ----- edit extension ------------------------------------------------------
    edit_results: list[dict[str, Any]] = []
    if args.run_edit:
        showcase = max(base_pairs, key=lambda p: p.clean_diff - p.corrupt_diff)
        others = [p for p in base_pairs if p.fact.fact_id != showcase.fact.fact_id]
        alt_layer = min(n_layers - 1, top_layer + max(2, n_layers // 4))
        for layer in (top_layer, alt_layer):
            for alpha in (1.0, 2.0, 4.0):
                res = run_edit_audit(ctx, bundle, showcase, layer, others, alpha=alpha)
                res["layer_kind"] = "localized" if layer == top_layer else "alternative"
                edit_results.append(res)
                print(f"[lab5] edit @{res['layer_kind']} L{layer} alpha={alpha}: "
                      f"direct={res['direct_success']} "
                      f"paraphrases={res['paraphrase_flips']}/{res['n_paraphrases']} "
                      f"neighbors_intact={res['neighbors_intact']}/{res['n_neighbors']}")
        edit_path = ctx.path("tables", "edit_results.csv")
        bench.write_csv(edit_path, [
            {k: (str(v) if isinstance(v, (list, tuple)) else v) for k, v in r.items()}
            for r in edit_results
        ])
        ctx.register_artifact(edit_path, "table", "Rank-one edit audit at the localized vs alternative layer.")

    # ----- plots ---------------------------------------------------------------
    if not args.no_plots:
        showcase_pair = max(base_pairs, key=lambda p: p.clean_diff - p.corrupt_diff)
        if args.showcase:
            showcase_pair = next((p for p in base_pairs if p.fact.fact_id == args.showcase), showcase_pair)
        cap = captures[f"{showcase_pair.fact.fact_id}:base"]
        plot_heatmap(ctx, showcase_pair, grid_rows, cap.tokens_text, n_layers)
        plot_localization(ctx, agg_rows, para_agg, len(base_pairs))
        plot_component_pass(ctx, comp_rows)
        plot_controls(ctx, control_rows, matched_mean)

    # ----- metrics, claims, summary --------------------------------------------
    peak_subject = max((r["recovery_subject"] for r in agg_rows if r["recovery_subject"] != ""))
    peak_last = max((r["recovery_last"] for r in agg_rows if r["recovery_last"] != ""))
    metrics = {
        "n_facts_kept_base": len(base_pairs),
        "n_pairs_rejected_alignment": rejected,
        "subject_handoff_layer": handoff_layer,
        "top_subject_layers": top_layers,
        "peak_subject_recovery": peak_subject,
        "top_last_layer": top_last_layer,
        "peak_last_recovery": peak_last,
        "matched_top_patch_mean_recovery": round(matched_mean, 4),
        "controls": {r["control"]: r["recovery"] for r in control_rows[-3:]},
        "edit_results": edit_results or None,
    }
    metrics_path = ctx.path("metrics.json")
    bench.write_json(metrics_path, metrics)
    ctx.register_artifact(metrics_path, "metrics", "Aggregate Lab 5 metrics.")

    run_name = ctx.run_dir.name
    mismatched = [r["recovery"] for r in control_rows if r["control"] == "mismatched_pair"]
    claims = [
        {
            "id": f"{LAB_ID}-C1",
            "tag": "CAUSAL",
            "text": (
                f"Patching the clean subject-position residual stream into the corrupt run at layers "
                f"{min(top_layers)}-{max(top_layers)} recovers a mean {matched_mean:.0%} of the clean "
                f"logit difference across {len(base_pairs)} capital facts. Population: 5-token "
                f"'The capital of X is' prompts with single-token subjects; intervention: single-"
                f"(layer,position) interchange."
            ),
            "artifact": f"runs/{run_name}/tables/localization_summary.csv",
            "falsifier": (
                "The same patch on a syntactically different fact family (or longer prompts with "
                "multi-token subjects) recovers nothing — the region was template-specific."
            ),
        },
        {
            "id": f"{LAB_ID}-C2",
            "tag": "CAUSAL",
            "text": (
                f"Subject-position recovery stays high until a handoff at layer {handoff_layer} "
                f"(early-layer recovery near 1.0 is a tautology: patching the subject stream at layer 0 "
                f"just substitutes the token), while last-position recovery peaks late at layer "
                f"{top_last_layer} ({peak_last:.2f}). Between the handoff and the readout, the fact is "
                "in transit: recall and readout are different places and different times."
            ),
            "artifact": f"runs/{run_name}/plots/localization_across_facts.png",
            "falsifier": "Component patching shows the late last-position effect is MLP-driven, not attention moving information.",
        },
        {
            "id": f"{LAB_ID}-C3",
            "tag": "CAUSAL",
            "text": (
                f"Specificity: mismatched-pair patches at the top (layer, position) recover a mean "
                f"{statistics.fmean(mismatched):.2f} (vs {matched_mean:.2f} matched) across "
                f"{len(mismatched)} controls — the recovered signal is fact-specific, not generic."
            ),
            "artifact": f"runs/{run_name}/tables/negative_control_scores.csv",
            "falsifier": "A larger control battery with same-template same-relation pairs shows high cross-recovery.",
        },
    ]
    if edit_results:
        def headline_row(kind: str) -> dict[str, Any]:
            rows = [r for r in edit_results if r["layer_kind"] == kind]
            flipped = [r for r in rows if r["direct_success"]]
            return min(flipped, key=lambda r: r["alpha"]) if flipped else max(rows, key=lambda r: r["alpha"])

        loc = headline_row("localized")
        alt = headline_row("alternative")
        claims.append({
            "id": f"{LAB_ID}-C4",
            "tag": "CAUSAL",
            "text": (
                f"A rank-one edit at the localized layer {loc['edit_layer']} "
                f"(dose alpha={loc['alpha']}) flipped {loc['intended_flip']} "
                f"directly={loc['direct_success']}, paraphrases "
                f"{loc['paraphrase_flips']}/{loc['n_paraphrases']}, neighbors intact "
                f"{loc['neighbors_intact']}/{loc['n_neighbors']}; the alternative layer "
                f"{alt['edit_layer']} (alpha={alt['alpha']}) gave directly={alt['direct_success']}, "
                f"paraphrases {alt['paraphrase_flips']}/{alt['n_paraphrases']}. The dose required "
                "relative to the activation patch (alpha=1) measures how distributed the fact is. "
                "Whether localization predicts editability (Hase et al.) is to be argued from "
                "these numbers, not assumed."
            ),
            "artifact": f"runs/{run_name}/tables/edit_results.csv",
            "falsifier": "Sweeping the edit across all layers shows success uncorrelated with the localization map.",
        })
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)

    lines = [
        "# Lab 5 run summary: activation patching and causal tracing",
        "",
        "## Run identity",
        "",
        f"- model: `{bundle.anatomy.model_id}` ({n_layers} blocks, d_model {bundle.anatomy.d_model})",
        f"- facts: {len(base_pairs)} base pairs past the gate "
        f"({sum(1 for r in gate_rows if r['template'] == 'base' and not r['kept'])} dropped, "
        f"{rejected} rejected by the alignment validator)",
        f"- grid: {n_layers + 1} stream layers x {base_pairs[0].n_tokens} positions per pair",
        "- evidence level: `CAUSAL`, scoped to the stated prompt population and intervention",
        "- self-checks: hook parity, lens, patch no-op (NEW), component anatomy",
        "",
        "## 1. What behavior was studied?",
        "",
        "Factual recall: single-token capital-city answers under a fixed 5-token template,",
        "with two paraphrase templates for confirmation.",
        "",
        "## 2. What intervention was used?",
        "",
        "Interchange: one clean-run residual vector spliced into the corrupt run at one",
        "(layer, position); recovery measured as the fraction of the clean logit difference",
        "restored. Component-level interchange (attn/MLP outputs) refines the top band.",
        "",
        "## 3. What controls were used?",
        "",
        "Mismatched-pair patches, wrong-position patches, and a held-out low-region check;",
        "the baseline gate drops unknown facts loudly; the patch no-op self-check guards the",
        "convention itself.",
        "",
        "## 4. Headline numbers",
        "",
        f"- subject-recovery handoff at layer {handoff_layer} (localized band {top_layers}; "
        f"early-layer recovery ~1.0 is the token-substitution tautology, not a finding)",
        f"- last-role recovery peaks at layer {top_last_layer}: {peak_last:.2f}",
        f"- matched top patch: {matched_mean:.2f} vs mismatched controls "
        f"{statistics.fmean(mismatched):.2f}",
    ]
    if edit_results:
        for r in edit_results:
            lines.append(
                f"- edit @{r['layer_kind']} L{r['edit_layer']} alpha={r['alpha']}: "
                f"direct={r['direct_success']}, paraphrases {r['paraphrase_flips']}/{r['n_paraphrases']}, "
                f"neighbors intact {r['neighbors_intact']}/{r['n_neighbors']}, "
                f"fluency {r['fluency_logprob_before']} -> {r['fluency_logprob_after']}"
            )
    lines += [
        "",
        "## 5. Claims (drafted; edit before the ledger)",
        "",
    ]
    for c in claims:
        lines.append(f"- `{c['id']}` {c['tag']}: {c['text']}")
        lines.append(f"  - falsifier: {c['falsifier']}")
    lines += [
        "",
        "## 6. The reading order",
        "",
        "1. `plots/patching_heatmap_<fact>.png` — one pair, the canonical picture.",
        "2. `plots/localization_across_facts.png` — the aggregation that makes it causal",
        "   tracing; subject vs last curves are the recall-vs-readout story.",
        "3. `plots/negative_controls.png` — what the top patch beats.",
        "4. `plots/component_patching.png` — who carries the fact in the top band.",
        "5. `tables/edit_results.csv` (if --run-edit) — localization meets editing.",
        "",
        "## 7. Caveats students must carry forward",
        "",
        "- Recovery is defined for THIS population (5-token template, single-token",
        "  subjects). State it in every claim; the falsifiers name the escape routes.",
        "- A single (layer,position) patch tests sufficiency-of-difference, not necessity;",
        "  multi-site and path patching are Lab 6's tools.",
        "- Localization informing editing is an open question you now have data on —",
        "  read Hase et al. before writing the reconciliation paragraph.",
        "",
    ]
    summary_path = ctx.path("run_summary.md")
    bench.write_text(summary_path, "\n".join(lines))
    ctx.register_artifact(summary_path, "summary", "The seven standard questions answered with this run's numbers.")
    print(f"[lab5] wrote run_summary.md and {len(claims)} drafted ledger claims")
