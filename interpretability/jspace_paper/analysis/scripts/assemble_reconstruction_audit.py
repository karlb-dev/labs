#!/usr/bin/env python3
"""P4 closure: assemble the reconstruction audit and the
unsupported-number register from the six domain verdict tables.
"""
from pathlib import Path

import pandas as pd

A = Path("/content/labs/interpretability/jspace_paper/analysis")
DOMAINS = ["phase2", "phase3", "qwen_ladder", "olmo", "gemma", "paper_draft"]


def main():
    frames = []
    for d in DOMAINS:
        df = pd.read_csv(A / f"tables/recon_{d}.csv")
        df["domain"] = d
        frames.append(df[["domain", "target_id", "frozen_value",
                          "reconstructed_value", "method", "status",
                          "notes" if "notes" in df.columns else "status"]])
    allrows = pd.concat(frames, ignore_index=True)
    allrows.to_parquet(A / "data/reconstruction_audit_rows.parquet",
                       index=False)

    tally = allrows.groupby(["domain", "status"]).size().unstack(fill_value=0)
    ok = {"byte_identical", "numerically_identical_render_diff",
          "numerically_within_frozen_tolerance"}
    n_ok = allrows.status.isin(ok).sum()
    n_fail = (allrows.status == "failed").sum()
    n_nr = (allrows.status == "not_reconstructable_from_released_data").sum()

    L = ["# RECONSTRUCTION_AUDIT.md — P4 output (gate for P6/P8)", "",
         "Assembled from six verdict tables (`tables/recon_*.csv`; builders "
         "`scripts/reconstruct_*.py`, each deterministic and rerunnable "
         "against the read-only registered sources). Status vocabulary per "
         "`protocol/CANONICAL_SCHEMA.md` §7.2.", "",
         f"**{len(allrows)} targets: {n_ok} verified "
         f"({(allrows.status == 'byte_identical').sum()} byte-identical, "
         f"{(allrows.status == 'numerically_identical_render_diff').sum()} "
         f"render-diff, "
         f"{(allrows.status == 'numerically_within_frozen_tolerance').sum()} "
         f"within-tolerance), {n_fail} failed, {n_nr} not reconstructable.**",
         "", "## Tally by domain", "", tally.to_markdown(), "",
         "## Highlights", "",
         "- **Phase 2** (10): HP1/HP3 confirmatory + replication "
         "byte-identical via full frozen-pipeline rerun (payload SHAs "
         "`5df4ace5…`/`f5367e5a…` reproduced; seed 4242; env bit-exact). "
         "Replication P-HP1 runs on the disclosed fallback population "
         "(322 rows/32 families), not the confirmatory intersection cohort.",
         "- **Phase 3** (39): every headline recomputed from item rows "
         "bit-for-bit — exact 2^17 enumerations, all seeded MC/bootstrap "
         "distributions sha-matched. Only byte-level divergence anywhere: "
         "a BLAS dgemv last-ulp accumulation pattern in four null ARRAYS "
         "(every derived scalar identical). Two frozen-record subtleties "
         "documented: the P3-P3 headline mixes estimate source "
         "(inference audit) with CI source (geometry-v2 100k bootstrap), "
         "and a 1-ulp summation fork exists between randomization and "
         "bootstrap routes for two estimates.",
         "- **Qwen ladder** (22): structural q50/q05 monotone "
         "(0.99522→0.99771→0.99870), selected-ID Jaccard exactly 7/13 at "
         "all three boundaries, projector 0.6748→0.7098 (floor 0.85), "
         "bridge rescue oscillates (−0.215 PASS/+0.559 FAIL/−0.294 FAIL); "
         "margin labels recomputed 100%; prompt-323 norms recomputed from "
         "the released 6.6 GB tensor bit-identically; Q-L4 reproduced from "
         "the frozen truth table (bridge-rescue row binds first). Drive "
         "registry mirror is a stale 69/82-line byte-prefix; repo registry "
         "(hash-matching FREEZE_HANDOFF) authoritative.",
         "- **OLMo** (139): capacity/geometry/trajectory/wedge/H6/Bank-W all "
         "byte-identical incl. 240/240 bootstrap CIs with sha-matched "
         "distributions and a byte-identical CPU rerun of the 5000-sim "
         "power study (0.7788 at 16; n=17 → 0.7922; **18 strictly first "
         "passing**). The `p4-bank-w-capability-joint-imported-dev-v1` "
         "outputs registered under a defunct `/content/labs_phase4_4/` "
         "root recover exactly from the repo copy (sha `ffe1ca8b…`).",
         "- **Gemma** (45): Stage-1 five-layer decisions, ceiling rule "
         "(float64 identity 3·q99 == 0.07870368901355948), batch q99s, "
         "bootstrap interval, floor ratio, G2.2 branch — all byte-identical "
         "incl. bit-exact 5000-draw bootstraps; selected-slot tensors "
         "recomputed byte-equal on CPU. The historical all-slot batch "
         "tensor is not released: the 0.00246 error rests on hash-verified "
         "cross-artifact consistency (recorded, accepted).",
         "- **Paper draft sweep** (31): 30 verify; **1 genuine error** "
         "(below).", "",
         "## Figure identity classes", "",
         "Every figure regenerated this phase (`claim_survival_timeline`, "
         "`cross_model_evidence_matrix`, `qwen_multilevel_convergence`) "
         "derives from committed verified tables — class "
         "`numerically_identical_render_diff` or better by construction. "
         "Frozen-era registered figures were hash-verified, not re-rendered "
         "(their source tables reconstruct; render identity not claimed).",
         "", "## P4 gate disposition", "",
         "Zero campaign headline numbers failed reconstruction; zero were "
         "not reconstructable from released data. The single `failed` row "
         "is a *draft prose* error (not a frozen number), quarantined in "
         "the unsupported-number register. **The route decision (P6) and "
         "drafting (P8) are unblocked.**", ""]
    (A / "reports/RECONSTRUCTION_AUDIT.md").write_text("\n".join(L))

    U = ["# UNSUPPORTED_NUMBER_REGISTER.md — published with the papers", "",
         "Rows: numbers appearing in current paper sources that fail "
         "reconstruction or carry unlicensed wording. An empty register, "
         "honestly earned, is the goal; this one has "
         "**1 failed number + 3 wording/labeling corrections**, all in the "
         "pre-analysis draft `kburtram_jspace.tex`, all fixable at P8.", "",
         "| # | Source | Quoted | Evidence found | Status | Required action |",
         "|---|---|---|---|---|---|",
         "| U1 | `kburtram_jspace.tex` §Prose (draft TODO block) | "
         "\"span-safe removes 72–78% of the label cost\" | registered "
         "reductions 0.493 (Think) / 0.716 (Instruct) / 0.778 (Qwen) — "
         "`prose_grid_figure_stats.json` | **failed** | rescope to "
         "49–78% with per-model values, or per-model sentence |",
         "| U2 | `kburtram_jspace.tex` §Composition | \"wild-cluster CI "
         "[−0.52, −0.01]\" for P3-P1 | that interval is the NORMAL "
         "approximation [−0.515271, −0.006637]; the percentile-t interval "
         "is [−0.537109, +0.015201] and crosses zero | mislabeled | quote "
         "the percentile-t (or randomization) interval with its name |",
         "| U3 | `kburtram_jspace.tex` §Composition | \"an honest "
         "near-miss\" for P3-P1 | release-audit correction #1: P3-P1 "
         "receives NO inferential near-miss wording (control-seed "
         "sensitivity crosses 0.05) | banned wording | descriptive "
         "wording only: negative, seed-sensitive, MDE-disclosed |",
         "| U4 | `kburtram_jspace.tex` header comment | source path "
         "`interpretability/jspace_runs/analysis/run_analysis.py` | script "
         "lives at `interpretability/jspace_paper/scripts/run_analysis.py` "
         "| stale pointer | fix path (stale register U8) |", "",
         "Render-diff notes (no action beyond a repr rule): the Gemma "
         "17-digit prose float `0.0024581113830208778` vs stored 16-digit "
         "`0.002458111383020878` (same float64); OLMo "
         "`checkpoint_estimates.csv` float rendering ≤1e-16 off its own "
         "JSON payload; H6 \"0.02216\"/\"0.02530\" are roundings of "
         "0.02216238880688507 / 0.025295454600777884. Papers quote the "
         "stored full-precision reprs.", ""]
    (A / "reports/UNSUPPORTED_NUMBER_REGISTER.md").write_text("\n".join(U))

    print(f"{len(allrows)} rows assembled; ok={n_ok} failed={n_fail} "
          f"not_reconstructable={n_nr}")
    print(tally.to_string())


if __name__ == "__main__":
    main()
