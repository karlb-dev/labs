# CLAIM_SURVIVAL_LEDGER.md — P2 output

Source of truth: `data/master_claim_ledger.parquet` + `data/master_claim_evidence_edges.parquet` (builder `scripts/extract_claims.py`; every cited id verified live in the frozen registries). Campaign claims C1–C7 seed from the six frozen skeleton sentences (addendum §2.3) plus the runtime-identity methods object; maximum-licensed wording is copied from the frozen states of record, never strengthened.

## Campaign claim ledger (C1–C7)

### C1 — Verbalizable causal channels exist; direction content, not dose, carries the effect

- **Tier:** confirmatory · **Status:** survived_replicated · **Paper route:** A
- **Scope:** Qwen3.6-27B (confirmatory+replicated); OLMo-3.1 Think/Instruct (estimates; interaction unreplicated); frozen Phase-2/3 fact banks; family-partitioned; layers J band L20-L44 (frozen); doses span-projection at logged per-site rank/energy; estimator full-sequence logsumexp-alias lp; family-clustered
- **Maximum licensed wording:** On Qwen3.6-27B, span-safe output-protected J-ablation produces a content-specific heavy tail beyond an exact per-site rank-and-energy-matched control (Phase 2 P-HP3 rate diff +0.2788 [0.2048, 0.3608], Holm p=0.0005; replication +0.2966 [0.2071, 0.3824]; Phase 3 P3-P2 tail excess +0.095833, plus-one p=1/100001; held-out replication +0.102083, p=1/100001), surviving clean answer protection, control seeds, boundary-safe grading, and accepted-alias scoring. Because the matched control equates rank and removed energy by construction, the difference isolates direction content. On the OLMo 3.1 pair the protected effects are prespecified estimates; the Phase 2 Think-vs-Instruct interaction (P-HP1 -0.5045 [-0.7195, -0.2949], Holm-rejected) did not replicate (+0.1036, p=0.7075) and licenses no replicated cross-model contrast.
- **Evidence:** `n6-confirmatory-analysis-v2`, `n6-replication-analysis-v2`, `p3-inference-audit-v1`, `p3-protocol-audit-protected-answer-qwen-v1`, `n6-repl-lens-independence-v2`
- **Falsifiers:** matched-control tail equal to J tail; protection failure driving tail; seed/boundary/alias sensitivity flipping sign
- **Forbidden upgrades:** global workspace; selective workspace; cross-model universality; OLMo replicated interaction

### C2 — Training, not architecture alone, shapes channel occupancy/organization (stage unlocalized)

- **Tier:** prespecified_development · **Status:** survived_development_stage_open · **Paper route:** A
- **Scope:** OLMo-3 Base, 3.0 Think, 3.1 Think, 3.1 Instruct (+SFT/DPO capability-gated); G5 development banks; Bank-S development assay; layers J band L20-L44; doses as registered per study; estimator registered capacity/geometry estimators; frames kept separate
- **Maximum licensed wording:** Across the tested OLMo lineage, measured sparse capacity is broadly conserved while J-mapped token geometry, selected spans, and Bank-S causal use reorganize at the first released Think transition, in own- and common-lens frames and on common cohorts (development tier). The official Think-SFT/DPO wedge does not localize the transition: its prospective Bank-S capability cohorts are empty (972-row batteries; capable rates 0.00617/0.00309; zero Bank-S facts capable on direct + composed), so stage effects are missing, not zero, and the wedge is ancestry-qualified, not objective-attributed.
- **Evidence:** `p4-lineage-trajectory-analysis-olmo-dev-v1`, `p4-lineage-common-cohort-analysis-olmo-dev-v1`, `ol-capacity-joint-dev-v1`, `ol-geometry-joint-dev-v1`, `ol2-stage-wedge-joint-analysis-v1`, `ol2-checkpoint-ancestry-v1`
- **Falsifiers:** capacity growth accounting for causal change; common-cohort trajectory vanishing under common frame; capable SFT/DPO cohort localizing the transition
- **Forbidden upgrades:** causal training-stage attribution; SFT-installs / DPO-removes wording; cross-model causal primary

### C3 — Qwen carries composed knowledge through a bridge-consumable route

- **Tier:** confirmatory · **Status:** survived_confirmatory_unreplicated_mechanism_partial · **Paper route:** A
- **Scope:** Qwen3.6-27B; Phase-3 paired direct/composed bank; mediation cohort 40 facts / 13 families; layers J band L20-L44; doses protection/lesion/injection at registered alphas; estimator teacher-forced lp contrasts + greedy generation checks
- **Maximum licensed wording:** On Qwen3.6-27B, protecting the true bridge rescues composed answers more than protecting the frozen chosen distractor (+0.431367 nats [+0.132018, +0.763437], plus-one p=0.009180, 94 items / 26 families), and measured rank/energy/geometry covariates do not explain the contrast away (residual +0.403816, p=0.01854). Counterfactual-bridge injection moves preference and generation toward the intended counterfactual (+8.582031 nats, exact p=0.000488) but is not separable from a direct answer-direction route (+1.342254 [-1.593275, +4.482051], p=0.419): substitution semantics remain development tier. No held-out-family P3-P3 replication exists; the distractor was chosen, not randomized.
- **Evidence:** `p3-bridge-geometry-qwen36-27b-v2`, `p3-n8-p3-level3-qwen36-27b-v1`, `p3-bridge-swap-endpoint-qwen36-27b-v1`
- **Falsifiers:** distractor-protection matching bridge-protection under randomized distractors; geometry covariates absorbing the rescue; answer-direction route fully explaining substitution
- **Forbidden upgrades:** abstract bridge channel distinct from answer direction; replicated mechanism; workspace routing

### C4 — Externalization (external-state substitution) remains unresolved: gated, not null

- **Tier:** capability_gated · **Status:** gated_open · **Paper route:** A_discussion+B_design
- **Scope:** OLMo-3.1 Think/Instruct; Qwen3.6-27B (capability rows); Bank-W candidate families under frozen capability protocol; layers n/a (no intervention opened); doses n/a; estimator frozen max-T pair simulation; SESOI 0.10 nat/doubling
- **Maximum licensed wording:** Bank-S development evidence motivates external-state substitution, but no externalization intervention is licensed: the cross-model Bank-W primary is blocked at 16/20 common-capable families (strict import + fresh mainline replay), and the OLMo Think/Instruct pair redesign is outcome-blind-powered at only 0.7788 with 16 shared capable families against the 0.80 target (first passing count 18). These are capability/planning boundaries, not negative results.
- **Evidence:** `ol-bank-w-capability-joint-dev-v1`, `p4-bank-w-capability-joint-imported-dev-v1`, `p4-import-olmo-bank-w-capability-v1`, `ol2-bank-w-olmo-pair-power-v1`
- **Falsifiers:** (future) powered >=18-family outcome in either direction
- **Forbidden upgrades:** Bank W is negative; externalization confirmed; pair ruled out

### C5 — Linear-transport premise is model-, checkpoint-, layer-, and dose-specific and must be gated

- **Tier:** methods · **Status:** closed_methods · **Paper route:** B
- **Scope:** Gemma-4-31B; OLMo-3 Base; OLMo-3.1 Think; frozen transport prompts/directions; layers Gemma L22/L30/L37/L44/L52; OLMo L24/L32/L40/L56; doses relative epsilon 0.001-0.10 (frozen ladder); estimator exact JVP dual-backend under calibrated envelope
- **Maximum licensed wording:** Gemma: under the prospectively calibrated pooled exact-backend envelope (ceiling 0.07870368901355948; historical all-slot error 0.0024581113830208778 within it; selected slot bit-identical), the unchanged five-layer local_tangent_mismatch classifier (L22/L30/L37/L44/L52) is a closed exact-JVP finite-scale methods result over the tested prompts, layers, directions, target map, and doses - licensing no nondifferentiability, missing-information, workspace, or mechanism claim. OLMo: the tested H6 ladder licenses no in-band finite-dose regime at L24/L32/L40 on either mandatory checkpoint; OLMo-3.1 Think passes only the L56 late anchor at epsilon 0.10 (12/12), Base does not (9/12). Neither result bounds the registered paired projection-ablation effects, whose in-situ validity evidence is their matched-control and positive-control behavior. Registered-dose coverage is unavailable (archive schema), not 0%.
- **Evidence:** `gm-jvp-gemma-stage1-v1`, `gm-jvp-gemma-backend-parity-v1`, `gm2-backend-parity-calibration-v1`, `gm2-stage1-relicense-v1`, `gm-jvp-olmo-positive-control-v1`, `ol2-transport-validation-base-v1`, `ol2-transport-validation-olmo31-think-v1`, `ol2-transport-validation-joint-v1`
- **Falsifiers:** calibrated envelope exceeding the scientific mismatch; late-anchor pass generalizing in-band
- **Forbidden upgrades:** Gemma is nonlinear/nondifferentiable; OLMo transport fails (unqualified); H6 invalidates causal effects; dose coverage = 0%

### C6 — Averaged-operator convergence does not imply sparse-selection or causal-endpoint invariance (Q-L4)

- **Tier:** methods · **Status:** closed_methods · **Paper route:** B
- **Scope:** Qwen3.6-27B; registered fit corpora + frozen functional battery; layers J band L20-L44; doses registered assay doses; estimator frozen structural/functional gate suite (draw-A nested fits vs published lens)
- **Maximum licensed wording:** Across the nested same-corpus Qwen fit ladder A120-A1000, the averaged transport operator converges strongly (A500-A1000 structural task q50 0.998702, q05 0.998122, both passing frozen gates) and occupancy, centered excess, span-safe specificity, tail rate, G4, and bridge preference are fit-stable - while selected-ID Jaccard (0.538462), normalized projector overlap (0.709818), and the bridge-rescue difference (-0.294028 nat) fail their frozen invariance gates. The selection-margin audit (17,381 retained; 15,536 near-tie, 1,845 stable-core, 0 rank-deficient) and the prompt-323 influence audit (all frozen materiality metrics negligible, closest >3,800x below threshold; current-runtime scope only) do not rescue the instrument. The mechanical route is Q-L4: no canonical sparse Qwen lens is nominated and no Phase 4 confirmatory primary was opened.
- **Evidence:** `p4-qwen-lens-convergence-drawA-n500-n1000-dev-v1`, `p4-qwen-multilens-functional-gate-a500-a1000-published-dev-v1`, `p4-qwen-selection-margin-a500-a1000-dev-v1`, `p4-qwen-lens-influence-prompt323-dev-v1`, `p4-qwen-canonical-lens-decision-a1000-dev-v1`
- **Falsifiers:** ties/margins explaining selection instability (ruled out); single-prompt influence explaining it (ruled out at current runtime)
- **Forbidden upgrades:** 'A1000 converged' without object naming; 'Qwen has no J-space'; fitting a convergence rate; reopening Q-L4

### C7 — Version-level pinning does not pin backward semantics (runtime-identity incident)

- **Tier:** methods · **Status:** closed_methods · **Paper route:** B
- **Scope:** Qwen3.6-27B (incident); lesson portable; fit-corpus prompts 112/323; layers all source layers (max at L0); doses n/a; estimator max ||J||/sqrt(d_model) runtime control
- **Maximum licensed wording:** Nominally identical GPU/driver/CUDA/Torch/Transformers/Triton/FLA runtimes produced materially different Jacobian norms across eras (prompt 323: 181.826310 vs frozen fit-log 173.345; prompt 112 control: 55.544060/55.587600 vs registered recompute 160.070954), each era internally repeatable to <1e-4 relative, caught before any contribution was written by a prospective 0.5-tolerance runtime control and resolved by a prospective contract amendment. Consequence: gradient-based pipelines must pin distribution contents and compiled-kernel caches by hash, carry a prospective runtime control with a frozen tolerance, and scope cross-era reproduction claims accordingly. The Phase 4 influence result is a current-runtime sensitivity shape; historical-runtime reproducibility is not claimed.
- **Evidence:** `p4-runtime-identity-synthesis-v1`, `p4-qwen-lens-influence-prompt323-dev-v1`
- **Falsifiers:** historical semantics recovered under content-identical rebuild (not attempted; identities unpreserved)
- **Forbidden upgrades:** root-cause attribution (build vs kernel-cache vs other); historical-runtime identity claim

## Skeleton-sentence regeneration map (P2 gate)

| Skeleton sentence | Ledger row | Terminal tier |
|---|---|---|
| 1 | C1 | confirmatory / survived_replicated |
| 2 | C2 | prespecified_development / survived_development_stage_open |
| 3 | C3 | confirmatory / survived_confirmatory_unreplicated_mechanism_partial |
| 4 | C4 | capability_gated / gated_open |
| 5 | C5 | methods / closed_methods |
| 6 | C6 | methods / closed_methods |
| (addendum §2.1 addition) | C7 | methods / closed_methods |

The conclusion skeleton regenerates from rows C1–C6 alone; C7 is the Paper-B runtime-identity section required by the addendum.

## Lab 37 claim survival (W1–W10)

Figure: `figures/claim_survival_timeline.{png,pdf}`. Stage codes: E exploratory claim · S survived/confirmed · N narrowed/corrected · O overturned/retired · G gated · P still open · — not tracked.

| ID | Original REPORT_v2 claim | Part 1 (Lab 37) | Forensic errata | Phase 2 conf+rep | Phase 3 mech+rep | Phase 4 instrument | Side Study 1 | Side Study 2 | Terminal |
|---|---|---|---|---|---|---|---|---|---|
| W1 | “Disagreement with the paper was instruments all along” | E | N | O | O | - | - | - | overturned |
| W2 | Static J-span causal dissociation is null on both models | E | N | P | - | - | - | - | still_open |
| W3 | Frozen per-item J ablation is control-clean | E | N | S | S | - | - | - | survived_as_corrected |
| W4 | OLMo capacity is ten times thinner than the paper | E | O | N | - | N | N | - | narrowed_development |
| W5 | Qwen is paper-range under the same harness | E | O | N | - | N | - | - | narrowed_development |
| W6 | Live per-token deletion measures computation deletion | E | O | N | S | - | - | - | survived_as_protected_effect |
| W7 | Workspace leads CoT by 46 steps | E | N | P | - | - | - | - | still_open |
| W8 | Externalization rescues frozen deletion | E | N | P | N | G | G | G | gated_open |
| W9 | Second-seed exact replication | E | N | S | S | - | - | - | survived_as_true_replication |
| W10 | Broadcast non-dissociation stands | E | O | - | - | - | - | - | overturned |

### Terminal notes with evidence

- **W1** — Errata: not identified. The repaired paper-faithful protected assay then produced real specific effects (n6-confirmatory-analysis-v2; p3-inference-audit-v1): the claim did not survive.
- **W2** — Errata: provisional auxiliary null. The static arm never re-entered a primary family; HP5's load outcome was not run (G5 bank built at dev tier only).
- **W3** — Errata: provisional (controls unmatched). Phase 2 replaced it with the exact per-site rank/energy-matched control under MC1-MC4 gates; the effect stands against the exact control and replicates (n6-*-analysis-v2; p3-inference-audit-v1).
- **W4** — Errata: not identified (wrong estimand). Repaired paper-defined occupancy is small (occ_med 2, r2-occupancy-*-v2, pilot) and capacity is fit-stable (Phase 4) and broadly conserved across the lineage (ol-capacity-joint-dev-v1) - a development statement, not the original ratio claim.
- **W5** — Errata: not identified ('same harness' was not same). Under the corrected estimator Qwen occupancy exceeds OLMo's (r2-occupancy-qwen36-v2); the paper comparison itself remains open.
- **W6** — Errata: exploratory - it measured output deletion. Output protection removed the confound and a specific effect remained (P-HP3), then survived span-safe correction and replication (P3-P2). The campaign's flagship self-correction.
- **W7** — Errata: exploratory mid-band with outcome-selected trace saving. Never retested under repaired instruments.
- **W8** — Errata: provisional mention-recovery. Became the Bank-S/Bank-W thread: development pattern on the Think path, cross-model primary blocked at 16/20 capable families, pair redesign unpowered (0.7788 at 16; first pass 18). Gated, not null (C4).
- **W9** — Errata: bundled robustness, not replication. Superseded by frozen held-out family partitions that replicated P-HP3 and P3-P2.
- **W10** — Errata: not paper-comparable (linear fan-out metric did not distinguish J from structured controls). Retired; the paper's MLP-gain and attention OV assays were never run.

## Survival summary

Of ten Lab 37 headline claims: **3 survived in corrected/narrowed form** (W3, W6, W9 — all through instrument repair, not in their original wording), **2 were overturned outright** (W1, W10), **2 narrowed to development-tier statements** (W4, W5), **1 is capability/power gated** (W8), and **2 were never retested** (W2, W7). No original claim survived verbatim: every surviving result is the product of at least one registered instrument correction — which is the A1 self-correction narrative in one sentence.
