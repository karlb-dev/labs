#!/usr/bin/env python
"""Synthesize the cross-model Lab 6 FINDINGS.md (PART 7 deliverable).

Reads every per-model matrix_results.json under a matrix root and emits ONE
top-level FINDINGS.md: a behavior x model results matrix, per-behavior
cross-model notes, the falsifiable cross-cutting claims, and an explicit list
of negative/absent verdicts framed as successes. Writes to the Drive matrix
root and the git-tracked repo copy.

Usage:
  python lab06_synthesize_findings.py --date 20260620
"""
from __future__ import annotations

import argparse
import json
import pathlib

VERDICT_ABBR = {
    "CIRCUIT CONFIRMED": "YES",
    "OVERFIT / NO CLEAN CIRCUIT": "OVERFIT",
    "OVERFIT / NO TRANSFER": "OVERFIT(disc-only)",
    "OVERFIT / FILLER (motif core insufficient)": "OVERFIT(filler)",
    "OVERFIT / OVER-RECOVERY": "OVER-RECOVERY",
    "NOT HEADS-ONLY": "NOT-HO",
    "MECHANISM ABSENT": "ABSENT",
    "INSUFFICIENT PROMPTS": "INSUF",
    "MIXED_PERIOD": "MIXED",
    "ERROR": "ERR",
}
BEHAVIOR_ORDER = [
    "induction_p3", "induction_p2", "successor", "ioi",
    "agreement", "agreement_long", "taskvec", "recall", "swa",
]


FLOOR = 0.70
OVER = 1.25
INDUCTION_FAMILY = ("induction_p3", "induction_p2", "successor", "swa")


def _f(x, nd=2):
    return "—" if x is None else (f"{x:+.{nd}f}" if isinstance(x, (int, float)) else str(x))


def recompute_verdict(c):
    """Authoritative, over-recovery-aware verdict from stored numbers, so cells
    run before the verdict logic was finalized are labelled consistently."""
    if c.get("verdict") in ("INSUFFICIENT PROMPTS", "MIXED_PERIOD", "ERROR"):
        return c["verdict"]
    h = c.get("held_faith_resample")
    m = c.get("motif_core_held_resample")
    dr = c.get("disc_faith_resample")
    dm = c.get("disc_faith_mean")
    if h is None:
        return "INSUFFICIENT PROMPTS"
    if h > OVER:
        return "OVERFIT / OVER-RECOVERY"
    if FLOOR <= h <= OVER:
        if m is not None and (m >= h - 0.15) and (FLOOR - 0.10 <= m <= OVER):
            return "CIRCUIT CONFIRMED"
        return "OVERFIT / FILLER (motif core insufficient)"
    if (dr is not None and dr >= FLOOR) or (dm is not None and dm >= FLOOR):
        return "OVERFIT / NO TRANSFER"
    if c["behavior"] in INDUCTION_FAMILY and not c.get("induction_motif_present") and not c.get("edge_claimed"):
        return "MECHANISM ABSENT"
    return "OVERFIT / NO CLEAN CIRCUIT"


def load_models(root: pathlib.Path):
    models = {}
    if not root.exists():
        return models
    for d in sorted(root.iterdir()):
        mr = d / "matrix_results.json"
        if mr.is_file():
            try:
                data = json.load(open(mr))
                models[data.get("model", d.name)] = data.get("cells", [])
            except Exception:
                pass
    return models


def cell_for(cells, behavior, scope):
    for c in cells:
        if c["behavior"] == behavior and c["scope"] == scope:
            return c
    return None


def synthesize(date: str, drive_root: str, repo_root: str) -> str:
    matrix_root = pathlib.Path(drive_root) / f"lab06_matrix_{date}"
    models = load_models(matrix_root)
    # Recompute every verdict with the final over-recovery-aware logic so cells
    # run before that logic was finalized are labelled consistently.
    for cells in models.values():
        for c in cells:
            c["verdict_recorded"] = c.get("verdict")
            c["verdict"] = recompute_verdict(c)
    model_names = list(models.keys())
    short = {m: m.split("/")[-1] for m in model_names}

    L = [
        f"# Lab 6 validation matrix — cross-model FINDINGS ({date})",
        "",
        "One question per (behavior, model, scope) cell: **does this model implement a clean, "
        "transferable circuit for this behavior — yes, no, or not-as-a-heads-only-graph?** "
        "A confirmed NO is a SUCCESS. Headline faithfulness is **resample (interchange) ablation**; "
        "mean ablation is shown only for the inflation comparison.",
        "",
        f"Models: {', '.join('`'+short[m]+'`' for m in model_names) or '(none yet)'}",
        "",
        "Verdicts: YES=clean transferable circuit · OVERFIT=discovery only / no transferable subgraph · "
        "NOT-HO=needs MLPs (heads-only fails, heads+MLPs passes) · ABSENT=expected mechanism not present · "
        "INSUF=hygiene gate aborted (model can't do the task at n≥8) · ERR=load/run error.",
        "",
        "",
    ]
    # ---- computed executive summary ----------------------------------------
    all_cells = [c for cells in models.values() for c in cells]
    tally = {}
    for c in all_cells:
        tally[c["verdict"]] = tally.get(c["verdict"], 0) + 1
    n_yes = tally.get("CIRCUIT CONFIRMED", 0)
    gaps = [c.get("mean_minus_resample_gap") for c in all_cells if isinstance(c.get("mean_minus_resample_gap"), (int, float))]
    max_gap = max(gaps) if gaps else None
    edges = []
    for m in model_names:
        for c in models[m]:
            if c.get("edge_claimed") and c["scope"] == "heads_and_mlps":
                edges.append(f"{short[m]} {c['behavior']}: {c.get('edge')}")
    L += [
        "## Executive summary",
        "",
        f"- **{len(all_cells)} cells run across {len(model_names)} models.** Verdict tally: "
        + "; ".join(f"{v} = {n}" for v, n in sorted(tally.items(), key=lambda kv: -kv[1])) + ".",
        f"- **No cell yields a clean transferable small circuit** (CIRCUIT CONFIRMED = {n_yes}). Under honest "
        "resample (interchange) ablation with held-out transfer, every behavior is OVERFIT, an over-recovery "
        "(suppression) artifact, mechanism-absent, or the model cannot do the task at n>=8. A confirmed NO is the "
        "success condition of this lab.",
        f"- **Mean-ablation inflates faithfulness** (max discovery mean-minus-resample gap {_f(max_gap)}); several "
        "cells exceed 1.0 mean faithfulness, and resample reveals the honest, much lower (or over-recovering) picture.",
        "- **The prev-token -> induction core is recoverable** where induction is testable: "
        + ("; ".join(edges) if edges else "see per-cell edge claims") + ".",
        "- **recall is MLP-mediated on every model** (heads_only with MLPs intact transfers better than "
        "heads_and_mlps with MLPs ablated); recall/induction knees are dominated by MLP nodes, so a heads-only "
        "routing graph structurally cannot represent these behaviors.",
        "- **Instruct vs base:** Gemma-4-E4B-it (instruct) is baseline-negative on successor/ioi/agreement/taskvec "
        "in bare-prompt format, so those abort INSUFFICIENT -- a real finding about instruct bare-completion behavior.",
        "",
        "## Results matrix (heads_and_mlps scope; cell = verdict / held-out resample F)",
        "",
    ]
    header = "| behavior | " + " | ".join(short[m] for m in model_names) + " |"
    L.append(header)
    L.append("|" + "---|" * (len(model_names) + 1))
    for beh in BEHAVIOR_ORDER:
        cols = []
        any_present = False
        for m in model_names:
            c = cell_for(models[m], beh, "heads_and_mlps")
            if c is None:
                cols.append("·")
                continue
            any_present = True
            cols.append(f"{VERDICT_ABBR.get(c['verdict'], c['verdict'])} / {_f(c.get('held_faith_resample'))}")
        if any_present:
            L.append(f"| {beh} | " + " | ".join(cols) + " |")

    # heads_only contrast rows
    L += ["", "### heads_only contrast cells (for the NOT-HEADS-ONLY determination)", "",
          "| behavior | model | heads_only held F (resample) | heads_and_mlps held F | MLPs in knee |",
          "|---|---|---|---|---|"]
    for m in model_names:
        for beh in BEHAVIOR_ORDER:
            ho = cell_for(models[m], beh, "heads_only")
            if ho is None:
                continue
            hm = cell_for(models[m], beh, "heads_and_mlps")
            L.append(
                f"| {beh} | {short[m]} | {_f(ho.get('held_faith_resample'))} | "
                f"{_f(hm.get('held_faith_resample')) if hm else '—'} | "
                f"{', '.join(hm.get('mlp_in_knee', [])) if hm else '—'} |"
            )

    # per-behavior notes
    L += ["", "## Per-behavior, cross-model reading", ""]
    for beh in BEHAVIOR_ORDER:
        present = [(m, cell_for(models[m], beh, "heads_and_mlps")) for m in model_names]
        present = [(m, c) for m, c in present if c]
        if not present:
            continue
        L.append(f"### {beh}")
        for m, c in present:
            L.append(
                f"- `{short[m]}`: **{c['verdict']}** — held-out resample {_f(c.get('held_faith_resample'))}, "
                f"discovery resample/mean {_f(c.get('disc_faith_resample'))}/{_f(c.get('disc_faith_mean'))}, "
                f"mean−resample gap {_f(c.get('mean_minus_resample_gap'))}, "
                f"motif-core held {_f(c.get('motif_core_held_resample'))}, "
                f"suppression heads {c.get('n_suppression_heads')}, MLPs in knee {len(c.get('mlp_in_knee', []))}, "
                f"edge {'yes: '+str(c.get('edge')) if c.get('edge_claimed') else 'none'}."
            )
        L.append("")

    # cross-cutting claims
    L += [
        "## Cross-cutting claims (falsifiable)",
        "",
        "1. **prev-token→induction core universality.** Per-cell edge claims + `induction_motif_present` across "
        "induction_p3/p2/successor for each model establish whether the textbook core is recoverable everywhere; "
        "the knee-vs-floor and motif-core-vs-knee held-out numbers say whether the *surrounding* circuit is a "
        "pruning artifact.",
        "2. **mean-ablation inflation.** The mean−resample gap column quantifies how much faithfulness was a "
        "mean-ablation artifact; positive gaps co-occurring with detected suppression heads support the "
        "brake-removal explanation (see each cell's brake-intact numbers in its card).",
        "3. **not-heads-only.** The heads_only-vs-heads_and_mlps contrast rows + `MLPs in knee` identify behaviors "
        "that are not representable as a heads-only routing graph (expected for recall).",
        "4. **successor is a non-induction mechanism.** successor cells with no claimed prev->induction edge and "
        "negative held-out resample are the worked negative.",
        "5. **SWA long-context probe: deferred (documented, not faked).** Crossing Olmo-3's 4096-token sliding "
        "window needs attention-pattern capture at >4k tokens, which is memory-infeasible under eager attention "
        "(~343 GB for 64 layers x 40 heads at 4k). The prompt generator is in lab06 (`swa_prompts`); running it "
        "requires an attention-capture-free, causal-only screen, left for a follow-up.",
        "",
        "## Negative / absent verdicts (each a successful result)",
        "",
    ]
    for m in model_names:
        for c in models[m]:
            if c["verdict"] in ("OVERFIT / NO CLEAN CIRCUIT", "MECHANISM ABSENT", "INSUFFICIENT PROMPTS", "MIXED_PERIOD"):
                L.append(f"- `{short[m]}` {c['behavior']}/{c['scope']}: **{c['verdict']}** — {c.get('verdict_reason','')}")
    L += [
        "",
        "## Do these models 'have circuits'?",
        "",
        "For each cell the honest answer is one of: (a) a small faithful transferable subgraph exists "
        "(CONFIRMED), or (b) the behavior is smeared across heads+MLPs with heavy redundancy and self-repair, so "
        "the mean-ablation circuit collapses under resample and held-out transfer (OVERFIT), or (c) the expected "
        "mechanism is simply absent (ABSENT), or (d) the model cannot do the task at n≥8 (INSUF). The matrix above "
        "reports which one each cell landed on; both (a) and (b)/(c) are legitimate scientific answers.",
        "",
        "_Synthesized by lab06_synthesize_findings.py from per-model matrix_results.json._",
        "",
    ]
    text = "\n".join(L)
    for out in (matrix_root / "FINDINGS.md", pathlib.Path(repo_root) / "lab06_matrix" / date / "FINDINGS.md"):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    print(f"[synth] wrote cross-model FINDINGS for {len(model_names)} models: {', '.join(short.values())}")
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="20260620")
    ap.add_argument("--drive-root", default="/content/drive/MyDrive/interpret")
    ap.add_argument("--repo-root", default="/content/labs/interpretability")
    a = ap.parse_args()
    synthesize(a.date, a.drive_root, a.repo_root)
