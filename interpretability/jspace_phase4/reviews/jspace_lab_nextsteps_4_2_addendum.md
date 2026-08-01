# jspace_lab_nextsteps_4_2_addendum.md — Verification, Amendments, the Conclusion Skeleton, and the Phase 5 Horizon

**Reviews:** `jspace_lab_nextsteps_4_2.md`, `PHASE4_DEVELOPMENT_REPORT.md`, the Phase 4 development handout, `inprogress.md` (24 h boundary, Qwen fit checkpointed), and branch `interp_jspace_part2` @ `ab9ee3e`.
**Role:** independent verification, the PI's resolutions, amendments to the 4.2 schedule, and — per the PI's direction — the convergence scaffolding: a falsifiable conclusion skeleton for the paper and the Phase 5 horizon document. Precedence: `nextsteps_4_2` governs the next development block; §3–§5 here govern on conflict; `nextsteps_4_1` + its addendum remain the Phase 4 frame.

> **Paste-line for the coding/research agent**
> Read `jspace_lab_nextsteps_4_2.md` in full, then this file. Execute the 4.2 block with the §4 amendments — above all the functional-gate protection rule: the fixed multi-lens functional gate is the block's never-drop item and runs against whatever fit milestones exist if the n=250 fit stalls. Registry discipline unchanged: phase4-development tier only, no confirmatory cells before freeze, append-only, supersede-never-edit. Produce `PAPER_CONCLUSION_SKELETON.md` and `PHASE5_HORIZON.md` on the CPU track per §6–§7. Stop at preregistration candidate v0.2 for PI sign-off.

---

## 1. Verification

Registry-checked at `ab9ee3e`: the four-checkpoint trajectory synthesis event with full input-manifest hashes; the Bank-S trajectory numbers and CIs exactly as 4.2 §1.2 tabulates (base +0.0016 ≈ 0; 3.0 Think +0.0724 [−0.011, +0.159]; 3.1 Think +0.1183 [+0.052, +0.188] CI-clean; 3.1 Instruct +0.0047 ≈ 0), frame-robust across own and common-base lenses; the leakage-safe nested corpora (draw-A n=120 exact historical prefix extended to 1000 minus the 80 legacy evaluation spares; draw-B disjoint to 500); the draw-A n=120 lens at hash `82af4cc7…`; and the structural comparison registered — correctly — as a *recipe/corpus-transfer diagnostic*, anticipating 4.2 §2.8. The §2.4 merge-shift redundancy is confirmed by algebra: `merged − J_r = n_c/(n_c+n_r)·(J_c − J_r)`, so the panel-D line is an exact scalar multiple of the relative Frobenius delta and carries no independent information. The 4.2 review's factual layer is sound; adopt its Part 2 corrections wholesale.

## 2. Assessment of the development block

Two results changed the project's shape. First, **the Bank-S anomaly now has a training trajectory**: absent at the pretrained base, present by 3.0 Think, strongest and CI-clean at 3.1 Think, absent at the Instruct sibling — and invariant to the coordinate-frame choice that could most easily have manufactured it. This is the externalization signature anticipated in the 4_1 addendum (§3.2: an external, redundant prompt state substituting for the internal channel on Think models) now visible as something *reasoning post-training installs by degree*, at development tier. It is the most theoretically loaded curve the campaign owns, and the honest chain of custody matters for the paper: anticipated in a governing document, observed in the trajectory, to be formally predicted-and-tested by Bank W's redundancy/derivation axes. Keep those three timestamps distinct; they are worth more separated than blurred into "we predicted it."

Second, **the scientific center moved to instrument invariance**, and the review's reframe is exactly right: the question is not whether two matrices are close but whether defensible lens fits induce the same selections, capacity, controls, and causal effects. The depth-dependent n=120-vs-published gap is currently uninterpretable (fit size × corpus × recipe × one heavy prompt × identity-dominance all confounded), and the nested design plus identity-adjusted metrics is the correct decomposition. The Qwen fitter's hardening (hash-pinned everything, atomic 3-prompt checkpoints, fused-kernel verification, incompatible-state refusal) is the best engineering in the campaign to date.

## 3. PI resolutions

R1 — The canonical-lens decision rule of 4.2 §3.3 is adopted verbatim and frozen *now*, before any n=250 outcome is inspected. R2 — P4-P1 is revised to the gated intersection-union form (P1a semantic vs. geometry-matched unrelated; P1b bridge vs. counterfactual answer-direction; `p = max(p_a, p_b)`), with the licensed-claim branch table exactly as §3.4 writes it. R3 — P4-P3 adopts the max-T statistic across the three model-specific load slopes; no model is privileged after Bank W development is seen. R4 — No Phase 5 namespace before the five §3.2 events; the horizon document (§7 below) is the sanctioned groundwork. R5 — Gemma stays out of this block. R6 — Bank B's two-relation construction and geometry-blind distractor selection are mandatory, not aspirational; a bank without the second-hop relations does not freeze.

## 4. Amendments to the 4.2 schedule

**4.1 The functional gate is the block's never-drop item.** The 4.2 sequence puts the fixed multi-lens functional gate (Q4) after n=250 completion, convergence-v2, and prompt-112. That ordering risks the exact failure mode §2.7 warns against: a block that ends with beautiful structural plots and no causal-invariance decision. Rule: at **T+6 h**, if the n=250 fit has not completed, freeze the fit at its highest atomic milestone (n=174+), register it as a named partial, and proceed to Q3/Q4 — a three-lens functional gate at unequal n (banked-milestone vs. n=120 vs. published) still answers the decision question, and the fit can resume in leftover time. The functional gate's development subset must be sized to ≤2 h.

**4.2 Task-token strata lead the report.** Adopt §2.5, and make the task-relevant strata (Phase 3 answer/bridge IDs, future Bank B/W IDs) the *primary* panel of convergence-v2, with the uniform 4,096 sample as the global diagnostic beneath it. The paper's exposure is entirely in the task rows.

**4.3 Prompt-112 contract check, twice.** The leave-one-out identity `(120·J₁₂₀ − J₁₁₂)/119` is valid only under the equal-weight running-mean contract. Assert it with the tiny-model reconstruction as specified — and also verify it for free against the atomic checkpoint series (any two adjacent milestones give `J_{n+3}` vs. `J_n` deltas that must match three per-prompt contributions). Two independent contract checks, one costing nothing.

**4.4 Common-cohort lineage analysis is CPU-first and gates the trajectory sentence.** No paper or handout sentence about the training trajectory graduates past development wording until the §6.1 intersection-cohort recompute lands. If the pattern survives fact-pairing, the trajectory becomes the paper's second figure; if it attenuates, that is a cohort-composition finding and gets reported as such.

**4.5 Provenance wording.** Until the same-corpus chain (A120/A250/A500/A1000) exists, every figure comparing to the published lens carries the §2.8 classification — `external published reference, partially specified recipe` — in its caption, not only in the registry.

## 5. The conclusion skeleton (write it now, as a falsifiable target)

Per the PI's direction to start converging: the CPU track produces `paper/PAPER_CONCLUSION_SKELETON.md` containing exactly five candidate conclusion sentences, each tagged with (a) the evidence that currently licenses it and its tier, and (b) the pending experiment that could still kill or upgrade it. Drafting conclusions as falsifiable targets is the disciplined version of "reaching toward conclusions" — the skeleton is a commitment device, not a press release.

1. **Verbalizable causal channels exist in open ~30B models, and direction content — not dose — carries the effect.** Licensed: Phase 3 confirmatory + replication (span-safe tail; exact matched controls at zero). Killable by: nothing currently identified; the seed-ensemble audit already bounded the control realization.
2. **What occupies the channel is set by training, not architecture alone.** Licensed at development tier: the four-checkpoint trajectory, frame-robust. Upgraded by: common-cohort recompute (§4.4), optionally intermediate checkpoints. Killable by: cohort-drift explaining the pattern.
3. **On Qwen the channel carries composed parametric knowledge through a bridge-consumable route.** Licensed: P3-P3 confirmatory (+0.431). Upgraded to "abstract bridge state" only by P4-P1b; killable to "answer-direction shortcut" by the same test. Held-out replication owed either way.
4. **Reasoning post-training installs external-state substitution: composed in-context state reduces reliance on the internal channel, and only on Think models.** Licensed at development tier: Bank-S trajectory. Formally tested by: Bank W's load × derivation × redundancy factorial (the experiment that also decides the noun). Killable by: redundancy/dose confounds Bank W is designed to separate.
5. **The lens's linear-transport premise is itself model-dependent and must be gated per checkpoint.** Licensed at pilot/methods tier: the Gemma battery with the OLMo positive control. Closed by: the exact-JVP ladder with the a+bε fit, in the later Gemma block.

Sentence-level rule: any sentence whose upgrading experiment fails gets *replaced by its downgrade*, verbatim from the branch tables already frozen in 4.1/4.2 — never silently reworded.

## 6. Groundwork for Phase 5 (`PHASE5_HORIZON.md`, CPU track)

Phase 5's namespace waits on the five §3.2 events, but its shape can be committed now so Phase 4's confirmatory outcomes route cleanly. The horizon document records four branches with entry criteria:

- **5A — Circuit completion** (entered if P4-P1b rejects): receiver-level mechanism work — patching, path localization, cross-relation transfer — turning "the model consumes a bridge state" into a traced route. The §7.6 receiver evidence is the pilot for this branch.
- **5B — Working-memory characterization** (entered if P4-P3's max-T rejects): load curves, capacity-vs-behavior coupling, and the earned upgrade of the noun.
- **5C — Cross-family prediction test** (entered regardless, if budget allows): one untouched open lineage with a base/post-trained pair (a Llama-class or Mistral-class family), run through the *frozen* Phase 4 assay stack with the training-installs-organization thesis stated as a prediction before any cell runs. This converts the campaign's central claim from postdiction to prediction — the single highest-credibility upgrade available.
- **5D — Methods release** (entered on Phase 4 completion): transport gate + convergence protocol + span-safe assay as a standalone artifact/note, decoupled from the workspace question.

The horizon document also carries the standing publication decision: with the lineage trajectory and the convergence study in hand, the single-paper option has strengthened — freeze Phase 4, run the confirmatory family once, and write, with 5A/5B as the paper's future-work section unless P4-P1b lands clean, in which case the mechanism section carries the paper and 5C becomes the sequel's spine.

## 7. Risks

(i) The functional gate slipping out of the block is the top risk; §4.1's T+6 h rule exists for it. (ii) Bank B authoring is the long pole for the freeze — start the two-relation construction immediately on CPU; a bank that arrives without second-hop relations forces a weaker P4-P1 and wastes untouched families. (iii) The Bank-S story is one CI-clean point and one near-zero sibling; resist narrating it as settled anywhere outside development-tier documents until Bank W and the common-cohort recompute land. (iv) The n=250 milestone is not a magic number — §5.5's point stands; the branch rule, not the counter, decides. (v) Keep the contradiction heuristic; it has not missed yet.

## 8. Bottom line

The development block did what development is for: it localized the campaign's strangest number onto a training trajectory, moved the biggest validity risk (lens fit-dependence) from anxiety to protocol, and hardened the instruments another notch. The 4.2 plan is the right next block with one reordering (causal invariance before structural completeness), and the two new artifacts — the conclusion skeleton and the Phase 5 horizon — are how this stops being an ever-expanding investigation and starts being a paper with a sequel. Freeze small, run once, write.
