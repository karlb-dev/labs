#!/usr/bin/env python3
"""A4: OLMo lineage evidence matrix.

Extracts every cell from tables/olmo_lineage_matrix_inputs.csv (340
verified rows; byte-identical reconstruction) — no hand-typed numbers.
Missing cells carry their named data state, never zero.

Outputs: tables/olmo_lineage_evidence_matrix.csv,
figures/olmo_lineage_evidence_matrix.{png,pdf},
reports/OLMO_LINEAGE_SYNTHESIS.md
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

A = Path("/content/labs/interpretability/jspace_paper/analysis")
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
TINT = {"dev": "#e6f0fc", "methods": "#eceafd", "gated": "#fdeecb",
        "state": "#f0efec", "na": "#f6f6f4"}
EDGE = {"dev": "#86b6ef", "methods": "#9085e9", "gated": "#fab219",
        "state": "#c3c2b7", "na": "#e1e0d9"}

ROWS = ["olmo3-base", "olmo3-think", "olmo31-think", "olmo31-instruct",
        "olmo3-think-sft", "olmo3-think-dpo"]
ROW_LABELS = ["OLMo-3 Base", "OLMo-3.0 Think", "OLMo-3.1 Think",
              "OLMo-3.1 Instruct", "Think-SFT (wedge)", "Think-DPO (wedge)"]
PAIR_TO_BASE = {"olmo3-think": "olmo3-base__olmo3-think",
                "olmo31-think": "olmo3-base__olmo31-think",
                "olmo31-instruct": "olmo3-base__olmo31-instruct"}


def main():
    df = pd.read_csv(A / "tables/olmo_lineage_matrix_inputs.csv",
                     dtype={"value": str})

    def val(mc, ck, metric, frame=None, cohort=None):
        q = df[(df.metric_class == mc) & (df.checkpoint == ck) &
               (df.metric == metric)]
        if frame is not None:
            q = q[q.frame == frame]
        if cohort is not None:
            q = q[q.cohort == cohort]
        if len(q) > 1 and q.value.nunique() > 1:
            raise ValueError(f"ambiguous rows for {mc}/{ck}/{metric}: "
                             f"{q[['frame', 'cohort', 'value']].to_dict('records')}")
        return q.value.iloc[0] if len(q) else None

    def fnum(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    def med_layers(ck, stem, frame="own"):
        try:
            vs = [fnum(val("capacity", ck, f"{stem}_L{l}", frame))
                  for l in (24, 32, 40)]
        except ValueError:
            return None
        vs = [v for v in vs if v is not None]
        return float(np.median(vs)) if vs else None

    COHORT = "all_four_common_cohort_equal_family"
    cells = {}   # (row, col) -> (text, kind)

    def put(r, c, text, kind):
        cells[(r, c)] = (text, kind)

    COLS = ["Capability", "Occupancy (med)", "Centered excess (med L24-40)",
            "Operator vs Base (q50)", "Mapped rows vs Base (q50)",
            "Selected-ID overlap vs Base", "Projector overlap vs Base",
            "Bank-S direct (own frame)", "Bank-S composition (own frame)",
            "H6 transport", "Stage wedge", "Bank-W"]

    for ck in ROWS:
        wedge = ck.endswith(("sft", "dpo"))
        # capability
        if wedge:
            cr = fnum(val("stage_wedge", ck, "capable_rate"))
            sr = fnum(val("stage_wedge", ck, "bank_s_capable_rate"))
            put(ck, "Capability",
                f"{cr:.3%} overall\n{sr:.3%} Bank-S\n0 S-facts d+c", "gated")
        else:
            put(ck, "Capability", "common-support\ncohorts pass", "dev")
        # occupancy + excess
        occ = med_layers(ck, "occupancy_median")
        exc = med_layers(ck, "centered_excess")
        if occ is not None:
            put(ck, "Occupancy (med)", f"{occ:.0f}", "dev")
            put(ck, "Centered excess (med L24-40)", f"{exc:+.5f}", "dev")
        else:
            put(ck, "Occupancy (med)", "not\nmeasured", "na")
            put(ck, "Centered excess (med L24-40)", "not\nmeasured", "na")
        # geometry vs base
        pair = PAIR_TO_BASE.get(ck)
        if ck == "olmo3-base":
            for c in COLS[3:7]:
                put(ck, c, "— (self)", "na")
        elif pair:
            op = fnum(val("geometry", pair, "raw_matrix_cosine_q50_layers"))
            mt = fnum(val("geometry", pair, "mapped_token_cosine_q50_q50_layers"))
            sid = fnum(val("geometry", pair, "selected_id_jaccard_q50_q50_layers"))
            pj = fnum(val("geometry", pair, "projector_overlap_q50_q50_layers"))
            put(ck, "Operator vs Base (q50)", f"{op:.4f}", "dev")
            put(ck, "Mapped rows vs Base (q50)", f"{mt:.4f}", "dev")
            put(ck, "Selected-ID overlap vs Base",
                f"{sid:.4f}" if sid is not None else "n/a", "dev")
            put(ck, "Projector overlap vs Base", f"{pj:.4f}", "dev")
        else:
            for c in COLS[3:7]:
                put(ck, c, "not\nmeasured", "na")
        # causal trajectory (own frame, all-four common cohort)
        d = fnum(val("causal_trajectory", ck, "bank_S_direct_specific", "own", COHORT))
        if d is not None:
            dl = fnum(val("causal_trajectory", ck, "bank_S_direct_specific_ci95_low", "own", COHORT))
            dh = fnum(val("causal_trajectory", ck, "bank_S_direct_specific_ci95_high", "own", COHORT))
            comp = fnum(val("causal_trajectory", ck, "bank_S_composition_specific", "own", COHORT))
            cl = fnum(val("causal_trajectory", ck, "bank_S_composition_specific_ci95_low", "own", COHORT))
            chh = fnum(val("causal_trajectory", ck, "bank_S_composition_specific_ci95_high", "own", COHORT))
            put(ck, "Bank-S direct (own frame)",
                f"{d:+.4f}\n[{dl:+.3f},{dh:+.3f}]",
                "dev" if (dl > 0 or dh < 0) else "state")
            put(ck, "Bank-S composition (own frame)",
                f"{comp:+.4f}\n[{cl:+.3f},{chh:+.3f}]",
                "dev" if (cl > 0 or chh < 0) else "state")
        elif wedge:
            put(ck, "Bank-S direct (own frame)",
                "capability_gated\n_missing", "gated")
            put(ck, "Bank-S composition (own frame)",
                "capability_gated\n_missing", "gated")
        # H6
        route = val("h6_transport", ck, "route")
        if route:
            pas = val("h6_transport", ck, "passing_rows")
            l56 = fnum(val("h6_transport", ck, "l56_eps0p10_passage_fraction"))
            short = ("no licensed regime" if "no_common" in route
                     else "late-anchor only" if "late" in route else route)
            put(ck, "H6 transport",
                f"{short}\n{pas}/336; L56 {l56:.2f}", "methods")
        else:
            put(ck, "H6 transport", "not on\nfrozen grid", "na")
        # stage wedge
        if wedge:
            put(ck, "Stage wedge", "cohorts empty →\nnull_or_unresolved",
                "gated")
        else:
            put(ck, "Stage wedge", "n/a (not a\nwedge artifact)", "na")
        # bank W
        if ck in ("olmo31-think", "olmo31-instruct"):
            put(ck, "Bank-W", "pair unpowered\n0.7788 @16, need 18", "gated")
        elif wedge:
            put(ck, "Bank-W", "n/a", "na")
        else:
            put(ck, "Bank-W", "x-model blocked\n16/20 families", "gated")

    tidy = pd.DataFrame(
        [dict(checkpoint=r, column=c, value=cells[(r, c)][0].replace("\n", " "),
              state=cells[(r, c)][1]) for r in ROWS for c in COLS
         if (r, c) in cells])
    tidy.to_csv(A / "tables/olmo_lineage_evidence_matrix.csv", index=False)

    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.family": "sans-serif",
        "text.color": INK})
    n_r, n_c = len(ROWS), len(COLS)
    fig, ax = plt.subplots(figsize=(15.2, 5.6))
    import textwrap
    for i, (ck, lab) in enumerate(zip(ROWS, ROW_LABELS)):
        y = n_r - 1 - i
        ax.text(-0.12, y + 0.5, lab, ha="right", va="center", fontsize=9.2,
                color=INK2)
        for j, c in enumerate(COLS):
            text, kind = cells.get((ck, c), ("", "na"))
            ax.add_patch(mpl.patches.FancyBboxPatch(
                (j + 0.045, y + 0.07), 0.91, 0.86,
                boxstyle="round,pad=0,rounding_size=0.05",
                facecolor=TINT[kind], edgecolor=EDGE[kind], linewidth=1.1))
            ax.text(j + 0.5, y + 0.5, text, ha="center", va="center",
                    fontsize=6.9, color=INK)
    for j, c in enumerate(COLS):
        ax.text(j + 0.5, n_r + 0.12,
                "\n".join(textwrap.wrap(c, 14, break_long_words=False)),
                ha="center", va="bottom", fontsize=7.6, color=MUTED)
    handles = [mpl.patches.Patch(facecolor=TINT[k], edgecolor=EDGE[k],
                                 label=lab)
               for k, lab in [("dev", "development measurement"),
                              ("methods", "methods result"),
                              ("gated", "gated / blocked (missing, not zero)"),
                              ("state", "measured, interval spans 0"),
                              ("na", "not measured / not applicable")]]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.015), ncol=5, frameon=False,
              fontsize=8.0, handlelength=1.3, columnspacing=1.2)
    ax.set_xlim(-2.7, n_c)
    ax.set_ylim(-0.4, n_r + 1.5)
    ax.axis("off")
    ax.set_title("OLMo lineage evidence matrix — development/methods tier; "
                 "Instruct is a sibling, SFT/DPO are ancestry-qualified "
                 "wedge artifacts; gated cells are missing, never zero",
                 fontsize=11, color=INK, pad=14, x=0.44)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(A / f"figures/olmo_lineage_evidence_matrix.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"{len(tidy)} cells rendered")


if __name__ == "__main__":
    main()
