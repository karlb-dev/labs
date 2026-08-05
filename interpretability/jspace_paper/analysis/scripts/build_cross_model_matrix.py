#!/usr/bin/env python3
"""A6: cross-model evidence matrix.

Rows: the six campaign model cells. Columns: evidence dimensions. Every
cell is a (status, tier, annotation, evidence_id) tuple from the frozen
records — no scalar "workspace score" exists or may be derived. Missing/
gated/not-applicable use the frozen state vocabulary.

Outputs: tables/cross_model_evidence_matrix.csv,
figures/cross_model_evidence_matrix.{png,pdf},
reports/CROSS_MODEL_SYNTHESIS.md
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

A = Path("/content/labs/interpretability/jspace_paper/analysis")

MODELS = ["Qwen 3.6 27B", "OLMo-3 Base", "OLMo-3.0 Think", "OLMo-3.1 Think",
          "OLMo-3.1 Instruct", "Gemma-4 31B"]
DIMS = ["Lens / readout validity", "Paper-defined capacity",
        "Span-safe causal specificity", "Held-out replication",
        "Bridge mechanism", "Externalization / load",
        "Fit invariance", "Finite-dose transport gate"]

# status codes: C confirmatory, R replicated, D development, M methods,
# G gated/blocked, X not-applicable (premise/branch), U untested, S split
CELLS = {
 ("Qwen 3.6 27B", "Lens / readout validity"): ("D", "G4 flip 0.78 vs 0.06 rand; published lens; Q-L4: no canonical sparse refit", "r5-swap-positive-control-qwen36-27b-v2;p4-qwen-canonical-lens-decision-a1000-dev-v1"),
 ("Qwen 3.6 27B", "Paper-defined capacity"): ("D", "occ_med 3-4; centered excess 0.050-0.060 (L24-L40)", "r2-occupancy-qwen36-v2"),
 ("Qwen 3.6 27B", "Span-safe causal specificity"): ("C", "P-HP3 +0.2788; P3-P2 +0.0958 vs exact matched control", "n6-confirmatory-analysis-v2;p3-inference-audit-v1"),
 ("Qwen 3.6 27B", "Held-out replication"): ("R", "P-HP3 +0.2966; P3-P2 +0.1021 on disjoint families", "n6-replication-analysis-v2;p3-inference-audit-v1"),
 ("Qwen 3.6 27B", "Bridge mechanism"): ("C", "rescue +0.431 (conf, unreplicated); substitution dev: swap -4.05 vs deletion -0.89; answer-direction not separated", "p3-bridge-geometry-qwen36-27b-v2;p3-bridge-swap-endpoint-qwen36-27b-v1"),
 ("Qwen 3.6 27B", "Externalization / load"): ("G", "Bank-W cross-model blocked 16/20 capable families", "p4-bank-w-capability-joint-imported-dev-v1"),
 ("Qwen 3.6 27B", "Fit invariance"): ("S", "Q-L4 split: aggregates fit-stable; selected IDs (J 0.538), projector (0.710), bridge endpoint (-0.294) not", "p4-qwen-canonical-lens-decision-a1000-dev-v1"),
 ("Qwen 3.6 27B", "Finite-dose transport gate"): ("U", "never run on Qwen; no licensed or unlicensed verdict exists", ""),
 ("OLMo-3 Base", "Lens / readout validity"): ("D", "own-frame lens + readout (lineage grids)", "p4-lineage-grid-olmo3-base-dev-v1;ol-geometry-readout-olmo3-base-v1"),
 ("OLMo-3 Base", "Paper-defined capacity"): ("D", "conserved with lineage (ol-capacity joint)", "ol-capacity-olmo3-base-dev-v1;ol-capacity-joint-dev-v1"),
 ("OLMo-3 Base", "Span-safe causal specificity"): ("D", "Bank-S causal use near zero pre-Think (development trajectory)", "p4-lineage-trajectory-analysis-olmo-dev-v1"),
 ("OLMo-3 Base", "Held-out replication"): ("U", "no replication cell exists", ""),
 ("OLMo-3 Base", "Bridge mechanism"): ("U", "not tested", ""),
 ("OLMo-3 Base", "Externalization / load"): ("G", "no capable cohort; thread gated at lineage level", "ol2-bank-w-olmo-pair-power-v1"),
 ("OLMo-3 Base", "Fit invariance"): ("U", "single fit per frame; no ladder", ""),
 ("OLMo-3 Base", "Finite-dose transport gate"): ("M", "H6 fail in-band; L56@0.10 reaches 9/12 < 0.90 floor - no licensed cell", "ol2-transport-validation-base-v1"),
 ("OLMo-3.0 Think", "Lens / readout validity"): ("D", "own + common-Base lens frames (v3/v4 grids)", "p4-lineage-grid-olmo3-think-common-base-lens-dev-v4"),
 ("OLMo-3.0 Think", "Paper-defined capacity"): ("D", "no material sparse-capacity growth at first Think transition", "ol-capacity-olmo3-think-dev-v1;ol-capacity-joint-dev-v1"),
 ("OLMo-3.0 Think", "Span-safe causal specificity"): ("D", "Bank-S causal use appears on observed Think path (development)", "p4-lineage-trajectory-analysis-olmo-dev-v1"),
 ("OLMo-3.0 Think", "Held-out replication"): ("U", "no replication cell exists", ""),
 ("OLMo-3.0 Think", "Bridge mechanism"): ("U", "not tested", ""),
 ("OLMo-3.0 Think", "Externalization / load"): ("G", "stage wedge capability-gated (SFT/DPO cohorts empty)", "ol2-stage-wedge-joint-analysis-v1"),
 ("OLMo-3.0 Think", "Fit invariance"): ("U", "single fit per frame", ""),
 ("OLMo-3.0 Think", "Finite-dose transport gate"): ("U", "not on the mandatory H6 grid", ""),
 ("OLMo-3.1 Think", "Lens / readout validity"): ("D", "G4 flip 0.76 vs 0.18 rand; independent-lens clause r=0.988 tail-J 0.878", "r5-swap-positive-control-olmo31-think-v2;n6-repl-lens-independence-v2"),
 ("OLMo-3.1 Think", "Paper-defined capacity"): ("D", "occ_med 2; centered excess ~0.004-0.012 (L24-L40)", "r2-occupancy-think-v2"),
 ("OLMo-3.1 Think", "Span-safe causal specificity"): ("D", "prespecified estimates; span-safe tail 0.20 vs control 0.017; HP1 interaction conf-not-replicated", "p3-span-audit-cross-model-v1;n6-confirmatory-analysis-v2"),
 ("OLMo-3.1 Think", "Held-out replication"): ("S", "HP3 estimate replicates; HP1 interaction does not (+0.1036, p=0.71)", "n6-replication-analysis-v2"),
 ("OLMo-3.1 Think", "Bridge mechanism"): ("D", "rescue contrast mildly negative (-0.202); swap inert (-0.089)", "p3-bridge-mediation-olmo31-think-v1"),
 ("OLMo-3.1 Think", "Externalization / load"): ("G", "pair design unpowered: 0.7788 at 16 shared capable families (first pass 18)", "ol2-bank-w-olmo-pair-power-v1"),
 ("OLMo-3.1 Think", "Fit invariance"): ("U", "no fit ladder on OLMo", ""),
 ("OLMo-3.1 Think", "Finite-dose transport gate"): ("M", "H6 fail in-band; late-anchor-only L56@0.10 12/12", "ol2-transport-validation-olmo31-think-v1"),
 ("OLMo-3.1 Instruct", "Lens / readout validity"): ("D", "G4 flip 0.76 vs 0.24 rand; independent-lens r=0.99 tail-J 0.883", "r5-swap-positive-control-olmo31-instruct-v2;n6-repl-lens-independence-v2"),
 ("OLMo-3.1 Instruct", "Paper-defined capacity"): ("D", "occ_med 2; centered excess ~0.004-0.012 - pair-flat", "r2-occupancy-olmo31instruct-v2"),
 ("OLMo-3.1 Instruct", "Span-safe causal specificity"): ("D", "span-safe tail 0.32 vs control 0.0; Bank-S use absent/near-zero at sibling (dev assay)", "p3-span-audit-cross-model-v1;p4-lineage-trajectory-analysis-olmo-dev-v1"),
 ("OLMo-3.1 Instruct", "Held-out replication"): ("S", "HP3 estimate replicates; HP1 interaction does not", "n6-replication-analysis-v2"),
 ("OLMo-3.1 Instruct", "Bridge mechanism"): ("U", "not tested (no mediation cohort)", ""),
 ("OLMo-3.1 Instruct", "Externalization / load"): ("G", "pair design unpowered (see Think row)", "ol2-bank-w-olmo-pair-power-v1"),
 ("OLMo-3.1 Instruct", "Fit invariance"): ("U", "no fit ladder on OLMo", ""),
 ("OLMo-3.1 Instruct", "Finite-dose transport gate"): ("U", "not on the mandatory H6 grid (sibling, not successor)", ""),
 ("Gemma-4 31B", "Lens / readout validity"): ("X", "excluded by J-lens validity premise failure (PI amendment) - not a below-boundary data point", "gm-state-of-record-v1"),
 ("Gemma-4 31B", "Paper-defined capacity"): ("X", "HP4 cell not-applicable after premise exclusion", "gm-state-of-record-v1"),
 ("Gemma-4 31B", "Span-safe causal specificity"): ("X", "no causal assay licensed", ""),
 ("Gemma-4 31B", "Held-out replication"): ("X", "n/a", ""),
 ("Gemma-4 31B", "Bridge mechanism"): ("X", "n/a", ""),
 ("Gemma-4 31B", "Externalization / load"): ("X", "n/a", ""),
 ("Gemma-4 31B", "Fit invariance"): ("U", "no fit ladder", ""),
 ("Gemma-4 31B", "Finite-dose transport gate"): ("M", "closed five-layer finite-scale mismatch under calibrated envelope (L22-L52); cause not localized", "gm-jvp-gemma-stage1-v1;gm2-stage1-relicense-v1"),
}

STATE = {
    "C": ("#0ca30c", "Confirmatory"),
    "R": ("#008300", "Held-out replicated"),
    "S": ("#2a78d6", "Split result (see note)"),
    "D": ("#86b6ef", "Development tier"),
    "M": ("#4a3aa7", "Methods tier"),
    "G": ("#fab219", "Gated / blocked"),
    "U": ("#e1e0d9", "Untested"),
    "X": ("#c3c2b7", "Not applicable (premise/branch)"),
}
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"


def luminance(hexc):
    r, g, b = (int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def main():
    rows = [dict(model=m, dimension=d, status=CELLS[(m, d)][0],
                 status_label=STATE[CELLS[(m, d)][0]][1],
                 annotation=CELLS[(m, d)][1],
                 evidence_ids=CELLS[(m, d)][2])
            for m in MODELS for d in DIMS]
    df = pd.DataFrame(rows)
    (A / "tables").mkdir(exist_ok=True)
    df.to_csv(A / "tables/cross_model_evidence_matrix.csv", index=False)

    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.family": "sans-serif",
        "font.size": 9.5, "text.color": INK})
    n_r, n_c = len(MODELS), len(DIMS)
    fig, ax = plt.subplots(figsize=(11.4, 4.9))
    for i, m in enumerate(MODELS):
        y = n_r - 1 - i
        for j, d in enumerate(DIMS):
            st = CELLS[(m, d)][0]
            color = STATE[st][0]
            ax.add_patch(mpl.patches.FancyBboxPatch(
                (j + 0.06, y + 0.10), 0.88, 0.80,
                boxstyle="round,pad=0,rounding_size=0.06",
                facecolor=color, edgecolor="none"))
            ink = "#ffffff" if luminance(color) < 0.45 else INK
            ax.text(j + 0.5, y + 0.5, st, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=ink)
        ax.text(-0.15, y + 0.5, m, ha="right", va="center", fontsize=9.3,
                color=INK2)
    import textwrap
    for j, d in enumerate(DIMS):
        ax.text(j + 0.5, n_r + 0.15,
                "\n".join(textwrap.wrap(d, 14, break_long_words=False)),
                ha="center", va="bottom", fontsize=7.8, color=MUTED)
    handles = [mpl.patches.Patch(facecolor=c, edgecolor="none",
                                 label=f"{k}  {lab}")
               for k, (c, lab) in STATE.items()]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False,
              fontsize=8.2, handlelength=1.2, columnspacing=1.2)
    ax.set_xlim(-2.6, n_c)
    ax.set_ylim(-0.35, n_r + 1.35)
    ax.axis("off")
    ax.set_title("Cross-model evidence matrix — status and tier per "
                 "dimension (no scalar model ranking exists)",
                 fontsize=11, color=INK, pad=16, x=0.42)
    fig.tight_layout()
    (A / "figures").mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(A / f"figures/cross_model_evidence_matrix.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)

    L = ["# CROSS_MODEL_SYNTHESIS.md — A6 output", "",
         "Source: `tables/cross_model_evidence_matrix.csv` "
         "(`scripts/build_cross_model_matrix.py`); every cell carries "
         "status, tier, annotation, and evidence ids. Figure: "
         "`figures/cross_model_evidence_matrix.{png,pdf}`. There is no "
         "scalar workspace score; the matrix exists to make the "
         "multidimensionality visible.", "",
         "## What the matrix shows", "",
         "1. **Exactly one cell in the campaign is confirmatory + "
         "replicated:** Qwen span-safe causal specificity (with its "
         "replication column). Qwen additionally holds the only "
         "confirmatory mechanism cell (bridge rescue, unreplicated by "
         "protocol).",
         "2. **The OLMo story is development-tier organization, not a "
         "weaker copy of Qwen's:** the 3.1 pair carries prespecified "
         "estimates whose HP3 side replicates while the Think-vs-Instruct "
         "interaction does not; Base -> Think -> Instruct carries the "
         "training-associated trajectory (C2) with the stage wedge "
         "capability-gated.",
         "3. **The transport column is orthogonal to the causal column:** "
         "Gemma's only populated cell is a closed methods result; OLMo "
         "fails H6 in-band while carrying its development causal cells; "
         "Qwen — the causal anchor — was **never transport-gated at "
         "all**. Papers must not read column 8 as bounding column 3 "
         "(see `A5_H6_PHASE3_RECONCILIATION.md`).",
         "4. **The gated column is uniform:** externalization is "
         "gated/blocked everywhere it is defined — the campaign's "
         "largest open question is a capability/power boundary, not a "
         "null.",
         "5. **Untested cells stay visibly untested** (Qwen transport, "
         "OLMo fit ladders, Base/3.0-Think replication): absence of a "
         "verdict is a data state, never an implied negative.", ""]
    (A / "reports/CROSS_MODEL_SYNTHESIS.md").write_text("\n".join(L))
    print(f"{len(df)} cells; figure + synthesis written")


if __name__ == "__main__":
    main()
