"""Lab 6: Circuit discovery and validation, the manual way.

The composition lab: Lab 2's attribution screens candidates, Lab 3's motif
scores name them, Lab 5's intervention logic stress-tests them — and the
deliverable is a CIRCUIT CARD making a subgraph claim with three earned
numbers attached:

* **faithfulness** — the circuit alone (every other head mean-ablated)
  preserves the behavior;
* **completeness** — ablating the circuit (everything else intact) destroys
  the behavior;
* **minimality** — every kept node's removal costs measurable faithfulness.

Task: induction completion (continuity with Lab 3 — students hold their motif
maps next to this circuit). All prompts are fixed-length 8-token repeating
patterns so that DATASET-MEAN ablation is well-defined per (layer, head,
position); held-out vocabulary families test whether the circuit is about
induction or about these particular tokens.

Method notes that carry the rigor:

* Mean-ablation, not zero-ablation: replacing a head's output with its
  dataset mean removes its prompt-specific computation while keeping the
  model in-distribution. Zero-ablation of hundreds of heads tests a model
  that never exists.
* The circuit's node set is attention heads. MLP layers are causally ranked
  and reported as supporting infrastructure, but the faithfulness complement
  never ablates them — a subgraph claim should say what it is a subgraph OF,
  and ours is the routing graph.
* The screen-vs-causal comparison is built in: candidates are screened by
  attribution + motif scores (cheap), then ranked by mean-ablation effect
  (causal). The scatter of one against the other IS the lab's version of
  "attribution patching vs automated discovery" (Syed et al.).
* One edge claim, earned by an ablation interaction: if the previous-token
  head's total effect vanishes when the induction head is already ablated,
  its influence routes THROUGH the induction head. Cheaper than path
  patching, honest about what it shows.

Evidence level: CAUSAL at circuit scope, on a stated prompt population.
"""

from __future__ import annotations

import dataclasses
import statistics
from typing import Any

import interp_bench as bench
from labs.lab02_direct_logit_attribution import compute_direct_logit_attribution
from labs.lab03_attention_routing import (
    head_attribution_scores,
    induction_score,
    prev_token_score,
)

LAB_ID = "L06"

FAITHFULNESS_FLOOR = 0.7   # greedy pruning stops before dropping below this
# Screen breadth scales with model width: 14+6+6 candidates out of 1024 heads
# was measured to start at 0.43 faithfulness on Olmo-3-7B (too thin to prune),
# while the same absolute counts are ample for gpt2's 144 heads.
SCREEN_TOP_ATTRIBUTION = 20
SCREEN_TOP_INDUCTION = 8
SCREEN_TOP_PREV = 8
N_MLP_CANDIDATES = 4


@dataclasses.dataclass(frozen=True)
class CircuitPrompt:
    example_id: str
    family: str          # 'discovery' or 'heldout'
    prompt: str
    target: str
    distractor: str


# All prompts are exactly 8 tokens (validated at runtime): 3-cycles repeat
# the cycle 2.67 times; 2-cycles 4 times. Target = induction continuation,
# distractor = cycle restart (the strongest wrong-but-plausible token).
ALL_PROMPTS: tuple[CircuitPrompt, ...] = (
    CircuitPrompt("d_colors", "discovery", "red blue green red blue green red blue", " green", " red"),
    CircuitPrompt("d_animals", "discovery", "dog cat bird dog cat bird dog cat", " bird", " dog"),
    CircuitPrompt("d_letters", "discovery", "B F Q B F Q B F", " Q", " B"),
    CircuitPrompt("d_moon", "discovery", "moon star moon star moon star moon star", " moon", " star"),
    CircuitPrompt("d_sun", "discovery", "sun rain sun rain sun rain sun rain", " sun", " rain"),
    CircuitPrompt("d_numbers", "discovery", "seven three nine seven three nine seven three", " nine", " seven"),
    CircuitPrompt("h_metals", "heldout", "gold silver gold silver gold silver gold silver", " gold", " silver"),
    CircuitPrompt("h_compass", "heldout", "north south east north south east north south", " east", " north"),
    CircuitPrompt("h_beasts", "heldout", "wolf bear fox wolf bear fox wolf bear", " fox", " wolf"),
    CircuitPrompt("h_matter", "heldout", "glass stone iron glass stone iron glass stone", " iron", " glass"),
)


@dataclasses.dataclass
class TaskExample:
    prompt: CircuitPrompt
    target_id: int
    distractor_id: int
    base_diff: float = 0.0


def node_name(kind: str, layer: int, head: int | None = None) -> str:
    return f"L{layer}H{head}" if kind == "head" else f"MLP{layer}"


# ---------------------------------------------------------------------------
# Dataset validation and baseline
# ---------------------------------------------------------------------------


def build_dataset(
    ctx: bench.RunContext, bundle: bench.ModelBundle, max_examples: int
) -> tuple[list[TaskExample], list[TaskExample], int, int]:
    tokenizer = bundle.tokenizer
    rows = []
    discovery: list[TaskExample] = []
    heldout: list[TaskExample] = []
    lengths = set()
    prompts = list(ALL_PROMPTS)
    if max_examples > 0:
        disc = [p for p in prompts if p.family == "discovery"][:max_examples]
        prompts = disc + [p for p in prompts if p.family == "heldout"]
    for cp in prompts:
        t_ids = tokenizer.encode(cp.target, add_special_tokens=False)
        d_ids = tokenizer.encode(cp.distractor, add_special_tokens=False)
        p_ids = tokenizer.encode(cp.prompt, add_special_tokens=False)
        lengths.add(len(p_ids))
        problems = []
        if len(t_ids) != 1 or len(d_ids) != 1:
            problems.append("multi-token answer")
        if len(p_ids) != 8:
            problems.append(f"prompt is {len(p_ids)} tokens, dataset contract is 8")
        ok = not problems
        rows.append({"example_id": cp.example_id, "family": cp.family, "prompt": cp.prompt,
                     "n_tokens": len(p_ids), "kept": ok, "problems": "; ".join(problems)})
        if ok:
            ex = TaskExample(cp, t_ids[0], d_ids[0])
            (discovery if cp.family == "discovery" else heldout).append(ex)
    path = ctx.path("diagnostics", "tokenization_report.csv")
    bench.write_csv(path, rows)
    ctx.register_artifact(path, "diagnostic", "Fixed-length and single-token validation for every prompt.")

    # Baseline gate: the model must do induction on each discovery prompt.
    kept_d, dropped = [], 0
    for ex in discovery:
        logits = bench.run_with_residual_cache(bundle, ex.prompt.prompt).final_logits_last
        ex.base_diff = float(logits[ex.target_id] - logits[ex.distractor_id])
        if ex.base_diff > 0:
            kept_d.append(ex)
        else:
            dropped += 1
            print(f"[lab6] dropping {ex.prompt.example_id}: base diff {ex.base_diff:+.2f} <= 0")
    for ex in heldout:
        logits = bench.run_with_residual_cache(bundle, ex.prompt.prompt).final_logits_last
        ex.base_diff = float(logits[ex.target_id] - logits[ex.distractor_id])
    if len(kept_d) < 3:
        raise RuntimeError(f"Only {len(kept_d)} discovery prompts pass the baseline gate; "
                           "the model does not do this task reliably enough to trace a circuit.")
    return kept_d, heldout, dropped, len(lengths)


# ---------------------------------------------------------------------------
# Metric under ablation
# ---------------------------------------------------------------------------


def metric_under_ablation(
    bundle: bench.ModelBundle,
    examples: list[TaskExample],
    head_anatomy: bench.HeadAnatomy,
    comp_anatomy: bench.ComponentAnatomy,
    heads: list[tuple[int, int]],
    mlps: list[int],
    head_means: Any,
    mlp_means: Any,
) -> float:
    """Mean logit diff over examples with the given node set mean-ablated."""
    diffs = []
    for ex in examples:
        logits = bench.run_with_node_set_ablation(
            bundle, ex.prompt.prompt, head_anatomy, comp_anatomy,
            heads=heads, mlps=mlps, head_means=head_means, mlp_means=mlp_means,
        )
        diffs.append(float(logits[ex.target_id] - logits[ex.distractor_id]))
    return statistics.fmean(diffs)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_screen_vs_causal(ctx: bench.RunContext, cand_rows: list[dict[str, Any]]) -> None:
    """The Syed et al. lesson in one scatter: cheap ranking vs causal ranking."""
    heads = [r for r in cand_rows if r["kind"] == "head"]
    if not heads:
        return
    fig, ax = bench.new_figure(figsize=(8.0, 6.0))
    colors = {"attribution": "tab:purple", "induction": "tab:red", "prev_token": "tab:blue", "multiple": "tab:green"}
    for r in heads:
        ax.scatter(abs(r["screen_score"]), r["causal_drop"], s=46, alpha=0.85,
                   color=colors.get(r["screen_reason"], "tab:gray"))
        ax.annotate(r["node"], (abs(r["screen_score"]), r["causal_drop"]),
                    textcoords="offset points", xytext=(4, 3), fontsize=7)
    for reason, color in colors.items():
        if any(r["screen_reason"] == reason for r in heads):
            ax.scatter([], [], color=color, label=reason)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("screen score (|attribution| or motif score — cheap)")
    ax.set_ylabel("causal drop (mean-ablation effect on metric — earned)")
    ax.set_title("Cheap screening vs causal ranking of candidate heads")
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "screen_vs_causal.png",
                      "Candidate heads: screening score against single-node mean-ablation effect.")


def plot_prune_trajectory(ctx: bench.RunContext, trajectory: list[dict[str, Any]]) -> None:
    if not trajectory:
        return
    fig, ax = bench.new_figure(figsize=(8.5, 5.2))
    xs = [t["n_nodes"] for t in trajectory]
    ys = [t["faithfulness"] for t in trajectory]
    ax.plot(xs, ys, marker="o", linewidth=2.0, color="tab:green")
    for t in trajectory:
        if t.get("removed"):
            ax.annotate(f"-{t['removed']}", (t["n_nodes"], t["faithfulness"]),
                        textcoords="offset points", xytext=(2, 8), fontsize=7, rotation=30)
    ax.axhline(FAITHFULNESS_FLOOR, color="tab:red", linewidth=1.0, linestyle="--",
               label=f"floor = {FAITHFULNESS_FLOOR}")
    ax.axhline(1.0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("circuit size (heads kept)")
    ax.set_ylabel("faithfulness (complement mean-ablated)")
    ax.set_title("Greedy pruning: what the behavior costs, node by node")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    bench.save_figure(ctx, fig, "prune_trajectory.png",
                      "Faithfulness as the circuit is pruned one head at a time.")


def plot_circuit_graph(
    ctx: bench.RunContext,
    circuit: list[tuple[int, int]],
    head_labels: dict[tuple[int, int], str],
    mlp_support: list[dict[str, Any]],
    edge: dict[str, Any] | None,
    n_layers: int,
    n_heads: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = bench.new_figure(figsize=(10.0, 6.0))
    colors = {"induction": "tab:red", "previous_token": "tab:blue",
              "first_token_sink": "tab:green", "other": "tab:gray", "diffuse": "tab:olive"}
    for (l, h) in circuit:
        label = head_labels.get((l, h), "other")
        ax.scatter(l, h, s=170, color=colors.get(label, "tab:gray"), zorder=3, edgecolors="black")
        ax.annotate(f"L{l}H{h}\n{label}", (l, h), textcoords="offset points", xytext=(6, 6), fontsize=7)
    for r in mlp_support:
        ax.scatter(r["layer"], n_heads + 1, s=120, marker="s", color="tab:orange", zorder=3)
        ax.annotate(f"MLP{r['layer']}\n(drop {r['causal_drop']:+.2f})", (r["layer"], n_heads + 1),
                    textcoords="offset points", xytext=(4, 6), fontsize=7)
    if edge is not None:
        (l1, h1), (l2, h2) = edge["from"], edge["to"]
        if (l1, h1) not in circuit:
            # The edge's source can be a screened head the pruner rejected
            # (e.g. a redundant previous-token head): draw it hollow so the
            # arrow has a visible, honest origin.
            ax.scatter(l1, h1, s=170, facecolors="none", edgecolors="tab:blue",
                       linewidths=1.6, zorder=3)
            ax.annotate(f"L{l1}H{h1}\n(screened, pruned)", (l1, h1),
                        textcoords="offset points", xytext=(6, -16), fontsize=7, color="tab:blue")
        ax.annotate(
            "", xy=(l2, h2), xytext=(l1, h1),
            arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.8, "shrinkA": 12, "shrinkB": 12},
        )
        mid_x, mid_y = (l1 + l2) / 2, (h1 + h2) / 2
        ax.annotate(f"effect routed: {edge['routed_fraction']:.0%}", (mid_x, mid_y),
                    textcoords="offset points", xytext=(0, 10), fontsize=8, color="black")
    ax.set_xlim(-1, n_layers)
    ax.set_ylim(-1.5, n_heads + 3)
    ax.set_xlabel("layer")
    ax.set_ylabel("head index (squares above = supporting MLPs)")
    ax.set_title("The validated circuit (arrow = the one edge claim, earned by ablation interaction)")
    bench.save_figure(ctx, fig, "circuit_graph.png",
                      "Kept heads by (layer, head) with motif labels, supporting MLPs, and the tested edge.")


def plot_fcm(ctx: bench.RunContext, fcm: dict[str, Any]) -> None:
    fig, ax = bench.new_figure(figsize=(7.5, 5.0))
    xs, heights, colors_, alphas, labels = [], [], [], [], []
    for i, key in enumerate(("faithfulness", "completeness_inverted")):
        for j, fam in enumerate(("discovery", "heldout")):
            val = fcm[fam]["faithfulness"] if key == "faithfulness" else 1.0 - fcm[fam]["completeness_ratio"]
            xs.append(i * 2.6 + j)
            heights.append(val)
            colors_.append("tab:green" if key == "faithfulness" else "tab:blue")
            alphas.append(1.0 if fam == "discovery" else 0.55)
            labels.append(f"{key.split('_')[0]}\n{fam}")
    bars = ax.bar(xs, heights, color=colors_)
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)
    ax.bar_label(bars, fmt="%.2f", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(1.0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_ylabel("fraction of base behavior")
    ax.set_title("Circuit scorecard: faithfulness and (1 − completeness ratio), discovery vs held-out")
    bench.save_figure(ctx, fig, "circuit_scorecard.png",
                      "Faithfulness and inverted completeness on discovery and held-out families.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(ctx: bench.RunContext, bundle: bench.ModelBundle) -> None:
    import torch

    args = ctx.args
    n_layers = bundle.anatomy.n_layers

    discovery, heldout, dropped, n_lengths = build_dataset(ctx, bundle, args.max_examples)
    base_metric = statistics.fmean(ex.base_diff for ex in discovery)
    print(f"[lab6] discovery: {len(discovery)} prompts (dropped {dropped}), held-out: {len(heldout)}; "
          f"base metric {base_metric:+.3f}")
    if n_lengths != 1:
        raise RuntimeError("Dataset contract violated: prompts have differing token lengths.")

    # Instrument verification.
    probe = discovery[0].prompt.prompt
    bench.run_hook_parity_check(ctx, bundle, probe)
    comp_anatomy = bench.resolve_component_anatomy(ctx, bundle, probe, rel_tolerance=args.dla_tolerance)
    head_anatomy = bench.resolve_head_anatomy(ctx, bundle)
    first_att = bench.run_with_attention_cache(bundle, probe)
    bench.run_lens_self_check(ctx, bundle, first_att.capture)
    bench.run_head_decomposition_check(ctx, bundle, head_anatomy, first_att, rel_tolerance=args.dla_tolerance)
    n_heads = head_anatomy.n_heads

    # ----- captures and dataset means ---------------------------------------
    att_caps, comp_caps = {}, {}
    for ex in discovery:
        att_caps[ex.prompt.example_id] = bench.run_with_attention_cache(
            bundle, ex.prompt.prompt, all_positions=True)
        comp_caps[ex.prompt.example_id] = bench.run_with_component_cache(
            bundle, ex.prompt.prompt, comp_anatomy, all_positions=True)
    head_means = torch.stack([att_caps[ex.prompt.example_id].o_in_last for ex in discovery]).mean(dim=0)
    mlp_means = torch.stack([comp_caps[ex.prompt.example_id].mlp_contrib for ex in discovery]).mean(dim=0)
    print(f"[lab6] dataset means computed over {len(discovery)} prompts "
          f"(heads {tuple(head_means.shape)}, mlps {tuple(mlp_means.shape)})")

    # ----- screening ----------------------------------------------------------
    head_attr: dict[tuple[int, int], list[float]] = {}
    head_induct: dict[tuple[int, int], list[float]] = {}
    head_prev: dict[tuple[int, int], list[float]] = {}
    mlp_attr: dict[int, list[float]] = {}
    for ex in discovery:
        att = att_caps[ex.prompt.example_id]
        att_final = bench.AttentionCapture(
            capture=att.capture, attentions=att.attentions,
            o_in_last=att.o_in_last[:, -1], attn_out_last=att.attn_out_last[:, -1])
        attr = head_attribution_scores(bundle, comp_anatomy, head_anatomy, att_final,
                                       ex.target_id, ex.distractor_id)
        cc = comp_caps[ex.prompt.example_id]
        cc_final = bench.ComponentCapture(capture=cc.capture,
                                          attn_contrib=cc.attn_contrib[:, -1],
                                          mlp_contrib=cc.mlp_contrib[:, -1])
        dla = compute_direct_logit_attribution(bundle, cc_final, ex.target_id, ex.distractor_id)
        for l in range(n_layers):
            mlp_attr.setdefault(l, []).append(dla["mlp_scores"][l])
            for h in range(n_heads):
                head_attr.setdefault((l, h), []).append(attr["scores"][l][h])
                pattern = att.attentions[l, h]
                head_prev.setdefault((l, h), []).append(prev_token_score(pattern))
                ind = induction_score(pattern, att.capture.input_ids)
                if ind is not None:
                    head_induct.setdefault((l, h), []).append(ind)

    mean_attr = {k: statistics.fmean(v) for k, v in head_attr.items()}
    mean_induct = {k: statistics.fmean(v) for k, v in head_induct.items()}
    mean_prev = {k: statistics.fmean(v) for k, v in head_prev.items()}
    mean_mlp = {k: statistics.fmean(v) for k, v in mlp_attr.items()}

    screen: dict[tuple[int, int], tuple[float, str]] = {}
    for k in sorted(mean_attr, key=lambda k: -abs(mean_attr[k]))[:SCREEN_TOP_ATTRIBUTION]:
        screen[k] = (mean_attr[k], "attribution")
    for k in sorted(mean_induct, key=lambda k: -mean_induct[k])[:SCREEN_TOP_INDUCTION]:
        screen[k] = (mean_induct[k], "multiple") if k in screen else (mean_induct[k], "induction")
    for k in sorted(mean_prev, key=lambda k: -mean_prev[k])[:SCREEN_TOP_PREV]:
        screen[k] = (mean_prev[k], "multiple") if k in screen else (mean_prev[k], "prev_token")
    mlp_candidates = sorted(mean_mlp, key=lambda k: -abs(mean_mlp[k]))[:N_MLP_CANDIDATES]
    print(f"[lab6] screened {len(screen)} candidate heads + {len(mlp_candidates)} candidate MLPs")

    def motif_label(k: tuple[int, int]) -> str:
        if mean_induct.get(k, 0) >= 0.35:
            return "induction"
        if mean_prev.get(k, 0) >= 0.35:
            return "previous_token"
        return "other"

    head_labels = {k: motif_label(k) for k in screen}

    # ----- causal ranking ------------------------------------------------------
    cand_rows: list[dict[str, Any]] = []
    for (l, h), (score, reason) in sorted(screen.items()):
        ablated = metric_under_ablation(bundle, discovery, head_anatomy, comp_anatomy,
                                        heads=[(l, h)], mlps=[], head_means=head_means, mlp_means=mlp_means)
        cand_rows.append({
            "node": node_name("head", l, h), "kind": "head", "layer": l, "head": h,
            "screen_score": round(score, 4), "screen_reason": reason,
            "motif_label": head_labels[(l, h)],
            "causal_drop": round(base_metric - ablated, 4),
        })
    for l in mlp_candidates:
        ablated = metric_under_ablation(bundle, discovery, head_anatomy, comp_anatomy,
                                        heads=[], mlps=[l], head_means=head_means, mlp_means=mlp_means)
        cand_rows.append({
            "node": node_name("mlp", l), "kind": "mlp", "layer": l, "head": "",
            "screen_score": round(mean_mlp[l], 4), "screen_reason": "attribution",
            "motif_label": "", "causal_drop": round(base_metric - ablated, 4),
        })
    cand_path = ctx.path("tables", "candidate_components.csv")
    bench.write_csv_with_context(ctx, cand_path, cand_rows)
    ctx.register_artifact(cand_path, "table", "Screened candidates with screening scores and causal drops.")
    results_path = ctx.path("results.csv")
    bench.write_csv_with_context(ctx, results_path, cand_rows)
    ctx.register_artifact(results_path, "results", "Alias of candidate_components.csv for the standard run contract.")

    # ----- greedy pruning -------------------------------------------------------
    all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]

    def faithfulness_of(circuit_heads: list[tuple[int, int]], examples: list[TaskExample],
                        base: float) -> float:
        complement = [k for k in all_heads if k not in set(circuit_heads)]
        m = metric_under_ablation(bundle, examples, head_anatomy, comp_anatomy,
                                  heads=complement, mlps=[], head_means=head_means, mlp_means=mlp_means)
        return m / base

    circuit = [(r["layer"], r["head"]) for r in cand_rows
               if r["kind"] == "head" and r["causal_drop"] > 0]
    circuit.sort(key=lambda k: -next(r["causal_drop"] for r in cand_rows
                                     if r["kind"] == "head" and (r["layer"], r["head"]) == k))
    trajectory = [{"n_nodes": len(circuit), "faithfulness": round(faithfulness_of(circuit, discovery, base_metric), 4),
                   "removed": ""}]
    print(f"[lab6] starting circuit: {len(circuit)} heads, faithfulness {trajectory[0]['faithfulness']:.3f}")
    while len(circuit) > 1:
        options = []
        for k in circuit:
            reduced = [c for c in circuit if c != k]
            options.append((faithfulness_of(reduced, discovery, base_metric), k))
        best_f, best_k = max(options)
        if best_f < FAITHFULNESS_FLOOR:
            break
        circuit = [c for c in circuit if c != best_k]
        trajectory.append({"n_nodes": len(circuit), "faithfulness": round(best_f, 4),
                           "removed": node_name("head", *best_k)})
        print(f"[lab6] pruned {node_name('head', *best_k)} -> {len(circuit)} heads, "
              f"faithfulness {best_f:.3f}")
    traj_path = ctx.path("tables", "prune_trajectory.csv")
    bench.write_csv_with_context(ctx, traj_path, trajectory)
    ctx.register_artifact(traj_path, "table", "Faithfulness at each greedy pruning step.")

    # ----- F / C / M evaluation ---------------------------------------------------
    heldout_base = statistics.fmean(ex.base_diff for ex in heldout) if heldout else None
    fcm: dict[str, Any] = {}
    minimality_rows = []
    for fam, examples, base in (("discovery", discovery, base_metric),
                                ("heldout", heldout, heldout_base)):
        if not examples or not base or base <= 0:
            continue
        faith = faithfulness_of(circuit, examples, base)
        circuit_ablated = metric_under_ablation(bundle, examples, head_anatomy, comp_anatomy,
                                                heads=circuit, mlps=[], head_means=head_means,
                                                mlp_means=mlp_means)
        fcm[fam] = {
            "base_metric": round(base, 4),
            "faithfulness": round(faith, 4),
            "completeness_ratio": round(circuit_ablated / base, 4),
        }
    for k in circuit:
        reduced = [c for c in circuit if c != k]
        f_without = faithfulness_of(reduced, discovery, base_metric)
        minimality_rows.append({
            "node": node_name("head", *k), "motif_label": head_labels.get(k, "other"),
            "faithfulness_without": round(f_without, 4),
            "marginal_value": round(fcm["discovery"]["faithfulness"] - f_without, 4),
        })
    fcm["minimality_worst_marginal"] = min((r["marginal_value"] for r in minimality_rows), default=None)
    fcm_path = ctx.path("faithfulness_completeness_minimality.json")
    bench.write_json(fcm_path, {"circuit": [node_name("head", *k) for k in circuit], **fcm,
                                "minimality": minimality_rows})
    ctx.register_artifact(fcm_path, "metrics", "The three earned circuit numbers, discovery and held-out.")
    min_path = ctx.path("tables", "pruned_circuit.csv")
    bench.write_csv_with_context(ctx, min_path, minimality_rows)
    ctx.register_artifact(min_path, "table", "Every kept node with its marginal faithfulness value.")
    print(f"[lab6] circuit {[node_name('head', *k) for k in circuit]}: "
          + ", ".join(f"{fam} F={v['faithfulness']:.2f} C-ratio={v['completeness_ratio']:.2f}"
                      for fam, v in fcm.items() if isinstance(v, dict)))

    # ----- the one edge claim: ablation interaction --------------------------------
    edge = None
    inducts = [k for k in circuit if head_labels.get(k) == "induction"]
    prevs = sorted((k for k in screen if head_labels.get(k) == "previous_token"),
                   key=lambda k: -mean_prev[k])
    if inducts and prevs:
        h_i = max(inducts, key=lambda k: mean_induct.get(k, 0))
        h_p = prevs[0]
        m_i = metric_under_ablation(bundle, discovery, head_anatomy, comp_anatomy,
                                    heads=[h_i], mlps=[], head_means=head_means, mlp_means=mlp_means)
        m_p = metric_under_ablation(bundle, discovery, head_anatomy, comp_anatomy,
                                    heads=[h_p], mlps=[], head_means=head_means, mlp_means=mlp_means)
        m_ip = metric_under_ablation(bundle, discovery, head_anatomy, comp_anatomy,
                                     heads=[h_i, h_p], mlps=[], head_means=head_means, mlp_means=mlp_means)
        effect_p = base_metric - m_p
        effect_p_given_i = m_i - m_ip
        routed = 1.0 - (effect_p_given_i / effect_p) if abs(effect_p) > 1e-6 else None
        edge = {
            "from": h_p, "to": h_i,
            "effect_p_alone": round(effect_p, 4),
            "effect_p_given_i_ablated": round(effect_p_given_i, 4),
            "routed_fraction": round(routed, 4) if routed is not None else None,
        }
        edge_path = ctx.path("tables", "edge_claim.json")
        bench.write_json(edge_path, {
            "edge": f"{node_name('head', *h_p)} -> {node_name('head', *h_i)}",
            **{k: v for k, v in edge.items() if k not in ("from", "to")},
            "explanation": (
                "Ablation interaction: if the previous-token head's effect "
                "shrinks when the induction head is already ablated, the "
                "shrunk fraction routed through the induction head. This is "
                "an edge CLAIM at interaction granularity; path patching "
                "(future work) would localize it to keys vs values."
            ),
        })
        ctx.register_artifact(edge_path, "metrics", "The one edge claim, earned by ablation interaction.")
        print(f"[lab6] edge {node_name('head', *h_p)} -> {node_name('head', *h_i)}: "
              f"effect {effect_p:+.3f} alone, {effect_p_given_i:+.3f} given induction ablated "
              f"({(routed or 0):.0%} routed)")

    # ----- failure cases ------------------------------------------------------------
    per_prompt_faith = []
    for ex in discovery + heldout:
        if ex.base_diff <= 0:
            continue
        complement = [k for k in all_heads if k not in set(circuit)]
        logits = bench.run_with_node_set_ablation(
            bundle, ex.prompt.prompt, head_anatomy, comp_anatomy,
            heads=complement, mlps=[], head_means=head_means, mlp_means=mlp_means)
        f = float(logits[ex.target_id] - logits[ex.distractor_id]) / ex.base_diff
        per_prompt_faith.append({"example_id": ex.prompt.example_id, "family": ex.prompt.family,
                                 "base_diff": round(ex.base_diff, 3), "faithfulness": round(f, 3)})
    per_prompt_faith.sort(key=lambda r: r["faithfulness"])
    failures = per_prompt_faith[:2]

    # ----- plots -----------------------------------------------------------------------
    if not args.no_plots:
        plot_screen_vs_causal(ctx, cand_rows)
        plot_prune_trajectory(ctx, trajectory)
        mlp_support = [r for r in cand_rows if r["kind"] == "mlp" and r["causal_drop"] > 0]
        plot_circuit_graph(ctx, circuit, head_labels, mlp_support, edge, n_layers, n_heads)
        if "heldout" in fcm and "discovery" in fcm:
            plot_fcm(ctx, fcm)

    # ----- circuit card ------------------------------------------------------------------
    card = [
        "# Circuit card: induction completion",
        "",
        f"- **Task:** induction completion on 8-token repeating patterns "
        f"(target = induction continuation, distractor = cycle restart)",
        f"- **Model:** `{bundle.anatomy.model_id}` | run `{ctx.run_dir.name}`",
        f"- **Dataset:** {len(discovery)} discovery prompts ({dropped} dropped at the gate), "
        f"{len(heldout)} held-out prompts in fresh vocabularies",
        f"- **Metric:** mean logit(target) − logit(distractor); base = {base_metric:+.3f}",
        f"- **Candidate components:** {len(cand_rows)} screened "
        f"({len([r for r in cand_rows if r['kind'] == 'head'])} heads, "
        f"{len([r for r in cand_rows if r['kind'] == 'mlp'])} MLPs)",
        f"- **Validated circuit (heads):** " + ", ".join(
            f"{node_name('head', *k)} ({head_labels.get(k, 'other')})" for k in circuit),
        f"- **Supporting MLPs (ranked, not in the routing claim):** " + (", ".join(
            f"MLP{r['layer']} (drop {r['causal_drop']:+.2f})"
            for r in cand_rows if r["kind"] == "mlp" and r["causal_drop"] > 0) or "none"),
        "",
        "## Scores",
        "",
        "| family | faithfulness | completeness ratio |",
        "|---|---:|---:|",
    ]
    for fam, v in fcm.items():
        if isinstance(v, dict):
            card.append(f"| {fam} | {v['faithfulness']} | {v['completeness_ratio']} |")
    card += [
        "",
        f"Minimality: worst marginal value {fcm['minimality_worst_marginal']} "
        "(see `tables/pruned_circuit.csv` — every kept node must earn its place).",
        "",
        "## The edge claim",
        "",
    ]
    if edge is not None:
        card.append(
            f"{node_name('head', *edge['from'])} → {node_name('head', *edge['to'])}: "
            f"{edge['routed_fraction']:.0%} of the previous-token head's effect routes through "
            f"the induction head (ablation interaction; see `tables/edge_claim.json`)."
        )
    else:
        card.append("No induction/previous-token pair survived into the circuit; no edge claimed.")
    card += [
        "",
        "## Failure cases the circuit does not explain",
        "",
    ]
    for f in failures:
        card.append(f"- `{f['example_id']}` ({f['family']}): per-prompt faithfulness {f['faithfulness']} "
                    f"(base diff {f['base_diff']})")
    card += [
        "",
        "## Scope and filler terms (MDC honesty section)",
        "",
        "- Population: 8-token repeating patterns from the listed vocabularies. Nothing here",
        "  claims the circuit handles natural-text induction (Lab 3's natural confirmation",
        "  was pattern-level, not circuit-level).",
        "- 'The previous-token head writes positional annotations' is a FILLER TERM at this",
        "  evidence level: the edge claim shows routing, not content.",
        "- Mean-ablation defines 'off'. A different off-distribution defines a different circuit.",
        "",
    ]
    card_path = ctx.path("circuit_card.md")
    bench.write_text(card_path, "\n".join(card))
    ctx.register_artifact(card_path, "summary", "The circuit card: the lab's deliverable.")

    # ----- metrics, claims, summary --------------------------------------------------------
    metrics = {
        "base_metric": base_metric,
        "n_discovery": len(discovery),
        "n_heldout": len(heldout),
        "circuit": [node_name("head", *k) for k in circuit],
        "fcm": {k: v for k, v in fcm.items() if isinstance(v, dict)},
        "minimality_worst_marginal": fcm["minimality_worst_marginal"],
        "edge": edge and {**{k: v for k, v in edge.items() if k not in ("from", "to")},
                          "edge": f"{node_name('head', *edge['from'])}->{node_name('head', *edge['to'])}"},
    }
    bench.write_json(ctx.path("metrics.json"), metrics)
    ctx.register_artifact(ctx.path("metrics.json"), "metrics", "Aggregate Lab 6 metrics.")

    run_name = ctx.run_dir.name
    claims = [
        {
            "id": f"{LAB_ID}-C1",
            "tag": "CAUSAL",
            "text": (
                f"A {len(circuit)}-head circuit ({', '.join(node_name('head', *k) for k in circuit)}) in "
                f"{bundle.anatomy.model_id} is faithful ({fcm['discovery']['faithfulness']:.2f} of base "
                f"behavior with all other heads mean-ablated) and complete-ish (circuit ablation leaves "
                f"{fcm['discovery']['completeness_ratio']:.2f} of base) for induction completion on "
                f"{len(discovery)} 8-token pattern prompts. Intervention: dataset-mean ablation; "
                "MLPs intact throughout."
            ),
            "artifact": f"runs/{run_name}/faithfulness_completeness_minimality.json",
            "falsifier": (
                "A different off-distribution (zero or resample ablation) or longer prompts collapse "
                "faithfulness — the circuit was an artifact of the mean-ablation choice."
            ),
        },
    ]
    if "heldout" in fcm:
        claims.append({
            "id": f"{LAB_ID}-C2",
            "tag": "CAUSAL",
            "text": (
                f"The circuit transfers to {len(heldout)} held-out vocabulary families: faithfulness "
                f"{fcm['heldout']['faithfulness']:.2f} vs {fcm['discovery']['faithfulness']:.2f} on "
                "discovery — the subgraph is about the induction computation, not these tokens."
            ),
            "artifact": f"runs/{run_name}/plots/circuit_scorecard.png",
            "falsifier": "Longer cycles, natural text, or >8-token prompts drop held-out faithfulness to chance.",
        })
    if edge is not None and edge["routed_fraction"] is not None:
        claims.append({
            "id": f"{LAB_ID}-C3",
            "tag": "CAUSAL",
            "text": (
                f"Edge claim at interaction granularity: {edge['routed_fraction']:.0%} of "
                f"{node_name('head', *edge['from'])}'s effect on the metric routes through "
                f"{node_name('head', *edge['to'])} (effect {edge['effect_p_alone']:+.2f} alone vs "
                f"{edge['effect_p_given_i_ablated']:+.2f} with the induction head already ablated)."
            ),
            "artifact": f"runs/{run_name}/tables/edge_claim.json",
            "falsifier": "Path patching shows the interaction is via the residual stream, not k-composition.",
        })
    bench.write_ledger_suggestions(ctx, LAB_ID, claims)

    lines = [
        "# Lab 6 run summary: circuit discovery, the manual way",
        "",
        "## Run identity",
        "",
        f"- model: `{bundle.anatomy.model_id}` ({n_layers} blocks x {n_heads} heads)",
        f"- task: induction completion, {len(discovery)} discovery + {len(heldout)} held-out prompts",
        f"- evidence level: `CAUSAL` at circuit scope; mean-ablation defines 'off'",
        "- self-checks: hook parity, lens, component anatomy, head decomposition",
        "",
        "## 1-4. Behavior, measurement, intervention, headline",
        "",
        f"- base metric {base_metric:+.3f}; circuit of {len(circuit)} heads "
        f"({', '.join(node_name('head', *k) for k in circuit)})",
    ]
    for fam, v in fcm.items():
        if isinstance(v, dict):
            lines.append(f"- {fam}: faithfulness {v['faithfulness']}, completeness ratio {v['completeness_ratio']}")
    lines += [
        f"- minimality: worst marginal value {fcm['minimality_worst_marginal']}",
        f"- screening vs causal ranking: see `plots/screen_vs_causal.png` — the Syed et al. lesson",
        "",
        "## 5. Claims",
        "",
    ]
    for c in claims:
        lines.append(f"- `{c['id']}` {c['tag']}: {c['text']}")
        lines.append(f"  - falsifier: {c['falsifier']}")
    lines += [
        "",
        "## 6. The reading order",
        "",
        "1. `circuit_card.md` — the deliverable; everything else is its evidence.",
        "2. `plots/circuit_graph.png` — the subgraph with its one earned edge.",
        "3. `plots/prune_trajectory.png` — what each node costs.",
        "4. `plots/screen_vs_causal.png` — where cheap ranking lied.",
        "5. `plots/circuit_scorecard.png` — discovery vs held-out.",
        "",
        "## 7. Caveats students must carry forward",
        "",
        "- Keep this circuit card. Lab 9 will hold it next to an attribution graph.",
        "- The card's 'filler terms' section is graded prose, not boilerplate.",
        "- The circuit is heads-only by design; the MLP support table is where the",
        "  'recall' part of the computation hides (Lab 5 said where).",
        "",
    ]
    bench.write_text(ctx.path("run_summary.md"), "\n".join(lines))
    ctx.register_artifact(ctx.path("run_summary.md"), "summary", "The seven standard questions answered.")
    print(f"[lab6] wrote circuit_card.md, run_summary.md, and {len(claims)} drafted ledger claims")
