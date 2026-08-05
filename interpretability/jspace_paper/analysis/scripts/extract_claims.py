#!/usr/bin/env python3
"""P2: master claim ledger, claim-evidence edges, survival timeline.

Campaign claims C1-C7 are seeded from the six frozen skeleton sentences
(addendum §2.3; C7 = the runtime-identity methods object joining Paper B)
with maximum-licensed wording taken from the frozen states of record.
Lab-37 rows W1-W10 trace the original REPORT_v2 claims (per
`jspace/REPORT_v2_ERRATA.md`) through the campaign stages.

Outputs:
  data/master_claim_ledger.parquet
  data/master_claim_evidence_edges.parquet
  reports/CLAIM_SURVIVAL_LEDGER.md
  figures/claim_survival_timeline.{png,pdf}

Every evidence id cited here is checked against the live evidence table;
a missing or non-live id is a hard error.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path("/content/labs")
A = REPO / "interpretability/jspace_paper/analysis"

# ---------------------------------------------------------------- claims
C = [
    dict(
        claim_id="C1",
        claim_text_short="Verbalizable causal channels exist; direction content, not dose, carries the effect",
        claim_text_maximum_licensed=(
            "On Qwen3.6-27B, span-safe output-protected J-ablation produces a "
            "content-specific heavy tail beyond an exact per-site rank-and-energy-"
            "matched control (Phase 2 P-HP3 rate diff +0.2788 [0.2048, 0.3608], "
            "Holm p=0.0005; replication +0.2966 [0.2071, 0.3824]; Phase 3 P3-P2 "
            "tail excess +0.095833, plus-one p=1/100001; held-out replication "
            "+0.102083, p=1/100001), surviving clean answer protection, control "
            "seeds, boundary-safe grading, and accepted-alias scoring. Because the "
            "matched control equates rank and removed energy by construction, the "
            "difference isolates direction content. On the OLMo 3.1 pair the "
            "protected effects are prespecified estimates; the Phase 2 "
            "Think-vs-Instruct interaction (P-HP1 -0.5045 [-0.7195, -0.2949], "
            "Holm-rejected) did not replicate (+0.1036, p=0.7075) and licenses no "
            "replicated cross-model contrast."),
        scope_models="Qwen3.6-27B (confirmatory+replicated); OLMo-3.1 Think/Instruct (estimates; interaction unreplicated)",
        scope_tasks="frozen Phase-2/3 fact banks; family-partitioned",
        scope_layers="J band L20-L44 (frozen)",
        scope_doses="span-projection at logged per-site rank/energy",
        scope_estimator="full-sequence logsumexp-alias lp; family-clustered",
        primary_tier="confirmatory",
        status="survived_replicated",
        evidence_ids="n6-confirmatory-analysis-v2;n6-replication-analysis-v2;p3-inference-audit-v1;p3-protocol-audit-protected-answer-qwen-v1;n6-repl-lens-independence-v2",
        replication_ids="n6-replication-analysis-v2;p3-inference-audit-v1",
        falsifiers="matched-control tail equal to J tail; protection failure driving tail; seed/boundary/alias sensitivity flipping sign",
        forbidden_upgrades="global workspace; selective workspace; cross-model universality; OLMo replicated interaction",
        paper_route="A",
    ),
    dict(
        claim_id="C2",
        claim_text_short="Training, not architecture alone, shapes channel occupancy/organization (stage unlocalized)",
        claim_text_maximum_licensed=(
            "Across the tested OLMo lineage, measured sparse capacity is broadly "
            "conserved while J-mapped token geometry, selected spans, and Bank-S "
            "causal use reorganize at the first released Think transition, in own- "
            "and common-lens frames and on common cohorts (development tier). The "
            "official Think-SFT/DPO wedge does not localize the transition: its "
            "prospective Bank-S capability cohorts are empty (972-row batteries; "
            "capable rates 0.00617/0.00309; zero Bank-S facts capable on direct + "
            "composed), so stage effects are missing, not zero, and the wedge is "
            "ancestry-qualified, not objective-attributed."),
        scope_models="OLMo-3 Base, 3.0 Think, 3.1 Think, 3.1 Instruct (+SFT/DPO capability-gated)",
        scope_tasks="G5 development banks; Bank-S development assay",
        scope_layers="J band L20-L44",
        scope_doses="as registered per study",
        scope_estimator="registered capacity/geometry estimators; frames kept separate",
        primary_tier="prespecified_development",
        status="survived_development_stage_open",
        evidence_ids="p4-lineage-trajectory-analysis-olmo-dev-v1;p4-lineage-common-cohort-analysis-olmo-dev-v1;ol-capacity-joint-dev-v1;ol-geometry-joint-dev-v1;ol2-stage-wedge-joint-analysis-v1;ol2-checkpoint-ancestry-v1",
        replication_ids="",
        falsifiers="capacity growth accounting for causal change; common-cohort trajectory vanishing under common frame; capable SFT/DPO cohort localizing the transition",
        forbidden_upgrades="causal training-stage attribution; SFT-installs / DPO-removes wording; cross-model causal primary",
        paper_route="A",
    ),
    dict(
        claim_id="C3",
        claim_text_short="Qwen carries composed knowledge through a bridge-consumable route",
        claim_text_maximum_licensed=(
            "On Qwen3.6-27B, protecting the true bridge rescues composed answers "
            "more than protecting the frozen chosen distractor (+0.431367 nats "
            "[+0.132018, +0.763437], plus-one p=0.009180, 94 items / 26 families), "
            "and measured rank/energy/geometry covariates do not explain the "
            "contrast away (residual +0.403816, p=0.01854). Counterfactual-bridge "
            "injection moves preference and generation toward the intended "
            "counterfactual (+8.582031 nats, exact p=0.000488) but is not "
            "separable from a direct answer-direction route (+1.342254 "
            "[-1.593275, +4.482051], p=0.419): substitution semantics remain "
            "development tier. No held-out-family P3-P3 replication exists; the "
            "distractor was chosen, not randomized."),
        scope_models="Qwen3.6-27B",
        scope_tasks="Phase-3 paired direct/composed bank; mediation cohort 40 facts / 13 families",
        scope_layers="J band L20-L44",
        scope_doses="protection/lesion/injection at registered alphas",
        scope_estimator="teacher-forced lp contrasts + greedy generation checks",
        primary_tier="confirmatory",
        status="survived_confirmatory_unreplicated_mechanism_partial",
        evidence_ids="p3-bridge-geometry-qwen36-27b-v2;p3-n8-p3-level3-qwen36-27b-v1;p3-bridge-swap-endpoint-qwen36-27b-v1",
        replication_ids="",
        falsifiers="distractor-protection matching bridge-protection under randomized distractors; geometry covariates absorbing the rescue; answer-direction route fully explaining substitution",
        forbidden_upgrades="abstract bridge channel distinct from answer direction; replicated mechanism; workspace routing",
        paper_route="A",
    ),
    dict(
        claim_id="C4",
        claim_text_short="Externalization (external-state substitution) remains unresolved: gated, not null",
        claim_text_maximum_licensed=(
            "Bank-S development evidence motivates external-state substitution, "
            "but no externalization intervention is licensed: the cross-model "
            "Bank-W primary is blocked at 16/20 common-capable families (strict "
            "import + fresh mainline replay), and the OLMo Think/Instruct pair "
            "redesign is outcome-blind-powered at only 0.7788 with 16 shared "
            "capable families against the 0.80 target (first passing count 18). "
            "These are capability/planning boundaries, not negative results."),
        scope_models="OLMo-3.1 Think/Instruct; Qwen3.6-27B (capability rows)",
        scope_tasks="Bank-W candidate families under frozen capability protocol",
        scope_layers="n/a (no intervention opened)",
        scope_doses="n/a",
        scope_estimator="frozen max-T pair simulation; SESOI 0.10 nat/doubling",
        primary_tier="capability_gated",
        status="gated_open",
        evidence_ids="ol-bank-w-capability-joint-dev-v1;p4-bank-w-capability-joint-imported-dev-v1;p4-import-olmo-bank-w-capability-v1;ol2-bank-w-olmo-pair-power-v1",
        replication_ids="",
        falsifiers="(future) powered >=18-family outcome in either direction",
        forbidden_upgrades="Bank W is negative; externalization confirmed; pair ruled out",
        paper_route="A_discussion+B_design",
    ),
    dict(
        claim_id="C5",
        claim_text_short="Linear-transport premise is model-, checkpoint-, layer-, and dose-specific and must be gated",
        claim_text_maximum_licensed=(
            "Gemma: under the prospectively calibrated pooled exact-backend "
            "envelope (ceiling 0.07870368901355948; historical all-slot error "
            "0.0024581113830208778 within it; selected slot bit-identical), the "
            "unchanged five-layer local_tangent_mismatch classifier "
            "(L22/L30/L37/L44/L52) is a closed exact-JVP finite-scale methods "
            "result over the tested prompts, layers, directions, target map, and "
            "doses - licensing no nondifferentiability, missing-information, "
            "workspace, or mechanism claim. OLMo: the tested H6 ladder licenses "
            "no in-band finite-dose regime at L24/L32/L40 on either mandatory "
            "checkpoint; OLMo-3.1 Think passes only the L56 late anchor at "
            "epsilon 0.10 (12/12), Base does not (9/12). Neither result bounds "
            "the registered paired projection-ablation effects, whose in-situ "
            "validity evidence is their matched-control and positive-control "
            "behavior. Registered-dose coverage is unavailable (archive schema), "
            "not 0%."),
        scope_models="Gemma-4-31B; OLMo-3 Base; OLMo-3.1 Think",
        scope_tasks="frozen transport prompts/directions",
        scope_layers="Gemma L22/L30/L37/L44/L52; OLMo L24/L32/L40/L56",
        scope_doses="relative epsilon 0.001-0.10 (frozen ladder)",
        scope_estimator="exact JVP dual-backend under calibrated envelope",
        primary_tier="methods",
        status="closed_methods",
        evidence_ids="gm-jvp-gemma-stage1-v1;gm-jvp-gemma-backend-parity-v1;gm2-backend-parity-calibration-v1;gm2-stage1-relicense-v1;gm-jvp-olmo-positive-control-v1;ol2-transport-validation-base-v1;ol2-transport-validation-olmo31-think-v1;ol2-transport-validation-joint-v1",
        replication_ids="",
        falsifiers="calibrated envelope exceeding the scientific mismatch; late-anchor pass generalizing in-band",
        forbidden_upgrades="Gemma is nonlinear/nondifferentiable; OLMo transport fails (unqualified); H6 invalidates causal effects; dose coverage = 0%",
        paper_route="B",
    ),
    dict(
        claim_id="C6",
        claim_text_short="Averaged-operator convergence does not imply sparse-selection or causal-endpoint invariance (Q-L4)",
        claim_text_maximum_licensed=(
            "Across the nested same-corpus Qwen fit ladder A120-A1000, the "
            "averaged transport operator converges strongly (A500-A1000 "
            "structural task q50 0.998702, q05 0.998122, both passing frozen "
            "gates) and occupancy, centered excess, span-safe specificity, tail "
            "rate, G4, and bridge preference are fit-stable - while selected-ID "
            "Jaccard (0.538462), normalized projector overlap (0.709818), and "
            "the bridge-rescue difference (-0.294028 nat) fail their frozen "
            "invariance gates. The selection-margin audit (17,381 retained; "
            "15,536 near-tie, 1,845 stable-core, 0 rank-deficient) and the "
            "prompt-323 influence audit (all frozen materiality metrics "
            "negligible, closest >3,800x below threshold; current-runtime scope "
            "only) do not rescue the instrument. The mechanical route is Q-L4: "
            "no canonical sparse Qwen lens is nominated and no Phase 4 "
            "confirmatory primary was opened."),
        scope_models="Qwen3.6-27B",
        scope_tasks="registered fit corpora + frozen functional battery",
        scope_layers="J band L20-L44",
        scope_doses="registered assay doses",
        scope_estimator="frozen structural/functional gate suite (draw-A nested fits vs published lens)",
        primary_tier="methods",
        status="closed_methods",
        evidence_ids="p4-qwen-lens-convergence-drawA-n500-n1000-dev-v1;p4-qwen-multilens-functional-gate-a500-a1000-published-dev-v1;p4-qwen-selection-margin-a500-a1000-dev-v1;p4-qwen-lens-influence-prompt323-dev-v1;p4-qwen-canonical-lens-decision-a1000-dev-v1",
        replication_ids="",
        falsifiers="ties/margins explaining selection instability (ruled out); single-prompt influence explaining it (ruled out at current runtime)",
        forbidden_upgrades="'A1000 converged' without object naming; 'Qwen has no J-space'; fitting a convergence rate; reopening Q-L4",
        paper_route="B",
    ),
    dict(
        claim_id="C7",
        claim_text_short="Version-level pinning does not pin backward semantics (runtime-identity incident)",
        claim_text_maximum_licensed=(
            "Nominally identical GPU/driver/CUDA/Torch/Transformers/Triton/FLA "
            "runtimes produced materially different Jacobian norms across eras "
            "(prompt 323: 181.826310 vs frozen fit-log 173.345; prompt 112 "
            "control: 55.544060/55.587600 vs registered recompute 160.070954), "
            "each era internally repeatable to <1e-4 relative, caught before any "
            "contribution was written by a prospective 0.5-tolerance runtime "
            "control and resolved by a prospective contract amendment. "
            "Consequence: gradient-based pipelines must pin distribution "
            "contents and compiled-kernel caches by hash, carry a prospective "
            "runtime control with a frozen tolerance, and scope cross-era "
            "reproduction claims accordingly. The Phase 4 influence result is a "
            "current-runtime sensitivity shape; historical-runtime "
            "reproducibility is not claimed."),
        scope_models="Qwen3.6-27B (incident); lesson portable",
        scope_tasks="fit-corpus prompts 112/323",
        scope_layers="all source layers (max at L0)",
        scope_doses="n/a",
        scope_estimator="max ||J||/sqrt(d_model) runtime control",
        primary_tier="methods",
        status="closed_methods",
        evidence_ids="p4-runtime-identity-synthesis-v1;p4-qwen-lens-influence-prompt323-dev-v1",
        replication_ids="",
        falsifiers="historical semantics recovered under content-identical rebuild (not attempted; identities unpreserved)",
        forbidden_upgrades="root-cause attribution (build vs kernel-cache vs other); historical-runtime identity claim",
        paper_route="B",
    ),
]

# ------------------------------------------------- Lab-37 survival rows
STAGES = ["part1", "repair", "phase2", "phase3", "phase4", "side_study1", "side_study2"]
STAGE_LABELS = ["Part 1\n(Lab 37)", "Forensic\nerrata", "Phase 2\nconf+rep",
                "Phase 3\nmech+rep", "Phase 4\ninstrument", "Side\nStudy 1",
                "Side\nStudy 2"]

W = [
    ("W1", "“Disagreement with the paper was instruments all along”",
     ["E", "N", "O", "O", "-", "-", "-"], "overturned",
     "Errata: not identified. The repaired paper-faithful protected assay then produced real specific effects (n6-confirmatory-analysis-v2; p3-inference-audit-v1): the claim did not survive."),
    ("W2", "Static J-span causal dissociation is null on both models",
     ["E", "N", "P", "-", "-", "-", "-"], "still_open",
     "Errata: provisional auxiliary null. The static arm never re-entered a primary family; HP5's load outcome was not run (G5 bank built at dev tier only)."),
    ("W3", "Frozen per-item J ablation is control-clean",
     ["E", "N", "S", "S", "-", "-", "-"], "survived_as_corrected",
     "Errata: provisional (controls unmatched). Phase 2 replaced it with the exact per-site rank/energy-matched control under MC1-MC4 gates; the effect stands against the exact control and replicates (n6-*-analysis-v2; p3-inference-audit-v1)."),
    ("W4", "OLMo capacity is ten times thinner than the paper",
     ["E", "O", "N", "-", "N", "N", "-"], "narrowed_development",
     "Errata: not identified (wrong estimand). Repaired paper-defined occupancy is small (occ_med 2, r2-occupancy-*-v2, pilot) and capacity is fit-stable (Phase 4) and broadly conserved across the lineage (ol-capacity-joint-dev-v1) - a development statement, not the original ratio claim."),
    ("W5", "Qwen is paper-range under the same harness",
     ["E", "O", "N", "-", "N", "-", "-"], "narrowed_development",
     "Errata: not identified ('same harness' was not same). Under the corrected estimator Qwen occupancy exceeds OLMo's (r2-occupancy-qwen36-v2); the paper comparison itself remains open."),
    ("W6", "Live per-token deletion measures computation deletion",
     ["E", "O", "N", "S", "-", "-", "-"], "survived_as_protected_effect",
     "Errata: exploratory - it measured output deletion. Output protection removed the confound and a specific effect remained (P-HP3), then survived span-safe correction and replication (P3-P2). The campaign's flagship self-correction."),
    ("W7", "Workspace leads CoT by 46 steps",
     ["E", "N", "P", "-", "-", "-", "-"], "still_open",
     "Errata: exploratory mid-band with outcome-selected trace saving. Never retested under repaired instruments."),
    ("W8", "Externalization rescues frozen deletion",
     ["E", "N", "P", "N", "G", "G", "G"], "gated_open",
     "Errata: provisional mention-recovery. Became the Bank-S/Bank-W thread: development pattern on the Think path, cross-model primary blocked at 16/20 capable families, pair redesign unpowered (0.7788 at 16; first pass 18). Gated, not null (C4)."),
    ("W9", "Second-seed exact replication",
     ["E", "N", "S", "S", "-", "-", "-"], "survived_as_true_replication",
     "Errata: bundled robustness, not replication. Superseded by frozen held-out family partitions that replicated P-HP3 and P3-P2."),
    ("W10", "Broadcast non-dissociation stands",
     ["E", "O", "-", "-", "-", "-", "-"], "overturned",
     "Errata: not paper-comparable (linear fan-out metric did not distinguish J from structured controls). Retired; the paper's MLP-gain and attention OV assays were never run."),
]

# ------------------------------------------------------- edges (T8)
EDGES = [
    ("C1", "n6-confirmatory-analysis-v2", "supports"),
    ("C1", "n6-replication-analysis-v2", "replicates"),
    ("C1", "p3-inference-audit-v1", "supports"),
    ("C1", "p3-protocol-audit-protected-answer-qwen-v1", "supports"),
    ("C1", "n6-repl-lens-independence-v2", "supports"),
    ("C1", "p3-control-seed-contract-audit-v2", "bounds"),
    ("C1", "p3-boundary-cohort-sensitivity-v2", "bounds"),
    ("C2", "p4-lineage-trajectory-analysis-olmo-dev-v1", "supports"),
    ("C2", "p4-lineage-common-cohort-analysis-olmo-dev-v1", "supports"),
    ("C2", "ol-capacity-joint-dev-v1", "supports"),
    ("C2", "ol-geometry-joint-dev-v1", "supports"),
    ("C2", "ol2-stage-wedge-joint-analysis-v1", "cannot_adjudicate"),
    ("C2", "ol2-checkpoint-ancestry-v1", "bounds"),
    ("C3", "p3-bridge-geometry-qwen36-27b-v2", "supports"),
    ("C3", "p3-n8-p3-level3-qwen36-27b-v1", "supports"),
    ("C3", "p3-bridge-swap-endpoint-qwen36-27b-v1", "narrows"),
    ("C3", "p4-bank-b-power-dev-v1", "blocks"),
    ("C4", "ol-bank-w-capability-joint-dev-v1", "blocks"),
    ("C4", "p4-bank-w-capability-joint-imported-dev-v1", "blocks"),
    ("C4", "p4-import-olmo-bank-w-capability-v1", "imports"),
    ("C4", "ol2-bank-w-olmo-pair-power-v1", "blocks"),
    ("C4", "ol2-stage-wedge-joint-analysis-v1", "cannot_adjudicate"),
    ("C5", "gm-jvp-gemma-stage1-v1", "supports"),
    ("C5", "gm-jvp-gemma-backend-parity-v1", "bounds"),
    ("C5", "gm2-backend-parity-calibration-v1", "supports"),
    ("C5", "gm2-stage1-relicense-v1", "supports"),
    ("C5", "gm-jvp-olmo-positive-control-v1", "supports"),
    ("C5", "ol2-transport-validation-base-v1", "supports"),
    ("C5", "ol2-transport-validation-olmo31-think-v1", "supports"),
    ("C5", "ol2-transport-validation-joint-v1", "supports"),
    ("C6", "p4-qwen-lens-convergence-drawA-n500-n1000-dev-v1", "supports"),
    ("C6", "p4-qwen-multilens-functional-gate-a500-a1000-published-dev-v1", "supports"),
    ("C6", "p4-qwen-selection-margin-a500-a1000-dev-v1", "supports"),
    ("C6", "p4-qwen-lens-influence-prompt323-dev-v1", "supports"),
    ("C6", "p4-qwen-canonical-lens-decision-a1000-dev-v1", "supports"),
    ("C7", "p4-runtime-identity-synthesis-v1", "supports"),
    ("C7", "p4-qwen-lens-influence-prompt323-dev-v1", "bounds"),
]

# ------------------------------------------------------------ figure
# dataviz reference palette (light mode), letters in every cell so state
# is never color-alone.
STATE = {
    "E": ("#c3c2b7", "Exploratory claim (Part 1)"),
    "S": ("#0ca30c", "Survived / confirmed"),
    "N": ("#2a78d6", "Narrowed / corrected"),
    "O": ("#d03b3b", "Overturned / retired"),
    "M": ("#4a3aa7", "Methods-only"),
    "G": ("#fab219", "Capability/power gated"),
    "P": ("#e1e0d9", "Still open / not retested"),
}
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def luminance(hexc):
    r, g, b = (int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def render_timeline(outdir: Path):
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.family": "sans-serif",
        "font.size": 9.5, "axes.edgecolor": "#c3c2b7",
        "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": INK2,
    })
    n_rows, n_cols = len(W), len(STAGES)
    fig, ax = plt.subplots(figsize=(10.6, 6.4))
    for i, (wid, label, states, _term, _note) in enumerate(W):
        y = n_rows - 1 - i
        for j, st in enumerate(states):
            if st == "-":
                ax.add_patch(mpl.patches.FancyBboxPatch(
                    (j + 0.06, y + 0.10), 0.88, 0.80,
                    boxstyle="round,pad=0,rounding_size=0.06",
                    facecolor=SURFACE, edgecolor=GRID, linewidth=1.0))
                continue
            color = STATE[st][0]
            ax.add_patch(mpl.patches.FancyBboxPatch(
                (j + 0.06, y + 0.10), 0.88, 0.80,
                boxstyle="round,pad=0,rounding_size=0.06",
                facecolor=color, edgecolor="none"))
            ink = "#ffffff" if luminance(color) < 0.45 else INK
            ax.text(j + 0.5, y + 0.5, st, ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color=ink)
        short = label if len(label) <= 44 else label[:41] + "…"
        ax.text(-0.15, y + 0.5, f"{wid}  {short}", ha="right", va="center",
                fontsize=9, color=INK2)
    for j, lab in enumerate(STAGE_LABELS):
        ax.text(j + 0.5, n_rows + 0.18, lab, ha="center", va="bottom",
                fontsize=8.6, color=MUTED)
    used = {st for _w, _l, states, _t, _n in W for st in states if st != "-"}
    handles = [mpl.patches.Patch(facecolor=c, edgecolor="none",
                                 label=f"{k}  {desc}")
               for k, (c, desc) in STATE.items() if k in used]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.015), ncol=4, frameon=False,
              fontsize=8.4, handlelength=1.2, handleheight=1.05,
              columnspacing=1.3)
    ax.set_xlim(-6.4, n_cols)
    ax.set_ylim(-0.4, n_rows + 1.1)
    ax.axis("off")
    ax.set_title(
        "Lab 37 claim survival through the repair, confirmation, mechanism, "
        "and instrument campaign", fontsize=11.5, color=INK, pad=18, x=0.27)
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"claim_survival_timeline.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)


def main():
    live = pd.read_parquet(A / "data/master_evidence_live.parquet")
    live_ids = set(live[live.status == "live"].evidence_id)

    ledger = pd.DataFrame(C)
    for _, row in ledger.iterrows():
        for eid in filter(None, row.evidence_ids.split(";")):
            assert eid in live_ids, f"{row.claim_id} cites non-live id {eid}"
    wrows = pd.DataFrame([
        dict(claim_id=wid, claim_text_short=label,
             claim_text_maximum_licensed=note,
             scope_models="", scope_tasks="", scope_layers="", scope_doses="",
             scope_estimator="", primary_tier="exploratory",
             status=term, evidence_ids="", replication_ids="",
             falsifiers="", forbidden_upgrades="revival without new registered evidence",
             paper_route="A_selfcorrection_narrative",
             **{f"stage_{s}": st for s, st in zip(STAGES, states)})
        for wid, label, states, term, note in W])
    full = pd.concat([ledger, wrows], ignore_index=True)
    (A / "data").mkdir(exist_ok=True)
    full.to_parquet(A / "data/master_claim_ledger.parquet", index=False)

    edges = pd.DataFrame(EDGES, columns=["claim_id", "evidence_id", "edge_type"])
    for eid in edges.evidence_id:
        assert eid in live_ids, f"edge cites non-live id {eid}"
    edges.to_parquet(A / "data/master_claim_evidence_edges.parquet", index=False)

    render_timeline(A / "figures")

    # ---------------- CLAIM_SURVIVAL_LEDGER.md
    L = ["# CLAIM_SURVIVAL_LEDGER.md — P2 output", "",
         "Source of truth: `data/master_claim_ledger.parquet` + "
         "`data/master_claim_evidence_edges.parquet` (builder "
         "`scripts/extract_claims.py`; every cited id verified live in the "
         "frozen registries). Campaign claims C1–C7 seed from the six frozen "
         "skeleton sentences (addendum §2.3) plus the runtime-identity "
         "methods object; maximum-licensed wording is copied from the frozen "
         "states of record, never strengthened.", "",
         "## Campaign claim ledger (C1–C7)", ""]
    for c in C:
        L += [f"### {c['claim_id']} — {c['claim_text_short']}", "",
              f"- **Tier:** {c['primary_tier']} · **Status:** {c['status']} · "
              f"**Paper route:** {c['paper_route']}",
              f"- **Scope:** {c['scope_models']}; {c['scope_tasks']}; "
              f"layers {c['scope_layers']}; doses {c['scope_doses']}; "
              f"estimator {c['scope_estimator']}",
              f"- **Maximum licensed wording:** {c['claim_text_maximum_licensed']}",
              f"- **Evidence:** " + ", ".join(
                  f"`{e}`" for e in c["evidence_ids"].split(";") if e),
              f"- **Falsifiers:** {c['falsifiers']}",
              f"- **Forbidden upgrades:** {c['forbidden_upgrades']}", ""]
    L += ["## Skeleton-sentence regeneration map (P2 gate)", "",
          "| Skeleton sentence | Ledger row | Terminal tier |", "|---|---|---|"]
    for i, cid in enumerate(["C1", "C2", "C3", "C4", "C5", "C6"], 1):
        c = next(x for x in C if x["claim_id"] == cid)
        L.append(f"| {i} | {cid} | {c['primary_tier']} / {c['status']} |")
    L += ["| (addendum §2.1 addition) | C7 | methods / closed_methods |", "",
          "The conclusion skeleton regenerates from rows C1–C6 alone; C7 is "
          "the Paper-B runtime-identity section required by the addendum.", "",
          "## Lab 37 claim survival (W1–W10)", "",
          "Figure: `figures/claim_survival_timeline.{png,pdf}`. Stage codes: "
          "E exploratory claim · S survived/confirmed · N narrowed/corrected "
          "· O overturned/retired · G gated · P still open · — not tracked.", "",
          "| ID | Original REPORT_v2 claim | " + " | ".join(
              s.replace("\n", " ") for s in STAGE_LABELS) + " | Terminal |",
          "|---|---|" + "---|" * len(STAGES) + "---|"]
    for wid, label, states, term, _ in W:
        L.append(f"| {wid} | {label} | " + " | ".join(states) + f" | {term} |")
    L += ["", "### Terminal notes with evidence", ""]
    for wid, label, _states, _term, note in W:
        L.append(f"- **{wid}** — {note}")
    L += ["", "## Survival summary", "",
          "Of ten Lab 37 headline claims: **3 survived in corrected/narrowed "
          "form** (W3, W6, W9 — all through instrument repair, not in their "
          "original wording), **2 were overturned outright** (W1, W10), **2 "
          "narrowed to development-tier statements** (W4, W5), **1 is "
          "capability/power gated** (W8), and **2 were never retested** (W2, "
          "W7). No original claim survived verbatim: every surviving result "
          "is the product of at least one registered instrument correction — "
          "which is the A1 self-correction narrative in one sentence.", ""]
    (A / "reports/CLAIM_SURVIVAL_LEDGER.md").write_text("\n".join(L))

    print(f"ledger rows: {len(full)} ({len(C)} campaign + {len(W)} lab37); "
          f"edges: {len(edges)}; figure rendered")


if __name__ == "__main__":
    main()
