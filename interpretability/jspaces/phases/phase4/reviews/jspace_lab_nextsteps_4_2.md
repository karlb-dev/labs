# jspace_lab_nextsteps_4_2.md

## J-space Phase 4, development block 2: lens convergence, causal invariance, and freeze-ready mechanism banks

**Review target:** `karlb-dev/labs`, branch `interp_jspace_part2`, reviewed through branch head `ab9ee3e65c9fcd2d1c15d6cf5959152dfd17f923` (`docs: checkpoint Qwen fit at 24h boundary`), plus the attached Phase 4 development report, handout, live progress ledger, and `p4f09_qwen_lens_structural_stability.png`.

**Phase decision:** remain in **Phase 4**. Name this document `jspace_lab_nextsteps_4_2.md`, not `5_1`. Phase 4 has a real development program in flight, its preregistration is still a candidate, and none of P4-P1, P4-P2, or P4-P3 is freeze-ready. A Phase 5 boundary would currently be bookkeeping theater: the Qwen nested fit is mid-run, the bridge endpoint still has an answer-direction confound, Bank W does not yet exist, and the official mode-by-phase assay is not gated.

**Review boundary:** the review covers committed code, registered evidence metadata, reports, figures, tests, and the current checkpoint contract. Heavy Drive parquets and tensors were not independently materialized in this review environment. The next-step plan therefore includes explicit row-level and tensor-level revalidation rather than treating prose summaries as substitutes for data.

> **Paste-line for the coding/research agent**
>
> Read `jspace_lab_nextsteps_4_1.md`, its accepted addendum, `PHASE3_STATE_OF_RECORD.md`, `PHASE4_DEVELOPMENT_REPORT.md`, `SCIENTIFIC_PREREGISTRATION_PHASE4_CANDIDATE.md`, and `INPROGRESS_VM11_20260731.md` before changing code. Phase 3 is closed and immutable. Phase 4 remains development-only and must not run confirmatory or replication outcomes before a tagged freeze. First recover the highest valid Qwen draw-A checkpoint and complete n=250 without modifying the pinned fitter or fit config. Then run the same-corpus structural and model-backed functional convergence gates in this document. Use their outcome-blind decision rules to choose draw-A n=500 versus draw-B n=120 as the next GPU fit. In parallel, finish the common-cohort OLMo analysis and author Bank B and Bank W. Revise P4-P1 so it distinguishes an abstract bridge route from direct answer-direction steering, and revise P4-P3 so “at least one model” is a multiplicity-controlled statistic rather than a post-outcome model choice. Register new evidence IDs, supersede rather than overwrite, commit at every boundary, and keep the Qwen recovery interval below roughly thirty minutes of GPU work.

---

# 0. Executive verdict

## 0.1 What Phase 4 has already accomplished

The last block did considerably more than “start Phase 4.” It established a proper new package and provenance boundary, closed the Phase 3 release audit, built a four-checkpoint OLMo trajectory in two coordinate frames, and started the first serious fit-size study on Qwen.

The strongest banked Phase 4 development results are:

1. **OLMo base is near zero under span-safe specificity.** The base checkpoint is near zero in all four known-bank cells. This makes the negative Bank-S effects at the Think checkpoints a real training-trajectory localization target rather than a universal property of the pretrained model.
2. **The Think path develops a reproducible Bank-S pattern.** Bank-S direct specificity is negative by OLMo-3 32B Think and more negative at OLMo-3.1 32B Think. The composed-minus-direct contrast rises from approximately zero at base to approximately `+0.12` at 3.1 Think. The pattern appears under both the checkpoint’s own lens and the frozen base-lens coordinate.
3. **The Instruct sibling does not share the 3.1 Think Bank-S pattern.** Its four primary known-bank intervals include zero, and its Bank-S composition contrast is near zero in both frames.
4. **Coordinate choice matters in some Bank-F cells but does not explain the Think Bank-S trajectory.** The seed-paired own/common analyses are a major improvement over informal transfer comparisons. The 3.1 Think Bank-S composition contrast is nearly unchanged across frames, while selected Bank-F cells are coordinate-sensitive.
5. **The Qwen n=120 lens is not uniformly the published n=1000 lens.** Agreement is strong late and materially weaker at lower and middle layers. That is exactly where a convergence study earns its keep.
6. **The Qwen fitter is unusually well hardened.** Exact model and corpus hashes, an immutable `jlens` source contract, a same-process CUDA gate, fused-kernel verification, atomic 3-prompt checkpoints, incompatible-state refusal, and milestone registration are all in place.

This is a strong development foundation. It is not yet a frozen Phase 4 study.

## 0.2 The scientific center has shifted

At the beginning of Phase 4, the largest open question was whether the OLMo lineage could localize a training transition. It now can, at development tier. The immediate paper risk has shifted to a more basic measurement question:

> Is the Qwen J-space result stable under the number and source of lens-fitting prompts at the exact layers and interventions used by the paper?

The displayed `p4f09` figure answers only a first part. It shows that a new draw-A n=120 map and the published n=1000 map are similar late but not uniformly similar in the assay band. It does **not** yet separate:

- fit size;
- corpus draw;
- recipe or runtime differences;
- one influential prompt;
- trivial convergence toward the identity near the target layer;
- structural coordinate drift that leaves causal conclusions unchanged;
- structural drift that changes selected rows, capacity, or causal effects.

The next block should decompose those possibilities instead of merely extending the fit counter.

## 0.3 Do not freeze Phase 4 after the n=250 milestone

The Phase 4 candidate preregistration still has unresolved fields for every primary family. More importantly, two of its current primary formulations need scientific repair before sign-off:

- **P4-P1 currently tests counterfactual bridge injection against unrelated injection.** Phase 3 development already showed that this moves answer probability and generation toward the counterfactual, but direct counterfactual-answer-direction injection explains most of the same movement. Repeating bridge-versus-unrelated on untouched families would replicate semantic disruption/substitution, but it would not isolate an abstract bridge channel.
- **P4-P3 says the load effect may occur on “at least one preregistered model.”** Without a frozen model selector or joint max statistic, this is a three-model multiple-testing choice waiting to happen.

The official thinking-mode endpoint and parser gates are also unset. Phase 4 should freeze only after the Qwen lens decision, revised bridge endpoint, Bank W design, mode gate, power simulations, and untouched partitions are complete.

## 0.4 The next 24 hours have one main job

The next GPU block should turn the Qwen fit-size question from a pretty structural plot into a **causal invariance decision**.

The high-value sequence is:

```text
recover highest draw-A checkpoint
-> finish and register n=250
-> run same-corpus structural comparison with identity-adjusted metrics
-> run one prompt-112 influence test
-> run one fixed multi-lens functional gate under n=120, n=250, published n=1000
-> choose n=500 or draw-B n=120 by frozen branch rule
-> use remaining GPU time on that branch
```

CPU work runs concurrently:

```text
all-four-checkpoint common-cohort OLMo analysis
Bank B crossed bridge/answer design
Bank W load x redundancy x derivation generator and audits
Phase 4 preregistration candidate v0.2
paper figures and result ledger
```

---

# 1. State of record at the review boundary

## 1.1 Governance

- Phase 3 is closed at `jspace-phase3-complete-v1`.
- Phase 4 imports Phase 3 immutably through `p4-import-phase3-release-v1`.
- The Phase 4 evidence registry is append-only.
- Phase 4 confirmatory and replication tiers remain forbidden before freeze.
- The current Phase 4 candidate is version 0.1 and explicitly unsigned.

## 1.2 OLMo trajectory evidence

The registered trajectory contains:

| checkpoint | role | own lens | common base lens | status |
|---|---|---:|---:|---|
| OLMo-3-1125-32B Base | pretrained anchor | base lens | same lens | complete |
| OLMo-3-32B-Think | first Think endpoint | own 120-prompt lens | base lens | complete, seed-paired repair complete |
| OLMo-3.1-32B-Think | later Think endpoint | own lens | base lens | complete, seed-paired |
| OLMo-3.1-32B-Instruct | sibling endpoint | own lens | base lens | complete, seed-paired |

The Bank-S own-frame summaries are:

| checkpoint | direct specificity | composed specificity | composed minus direct |
|---|---:|---:|---:|
| Base | `+0.0005` | `+0.0021` | `+0.0016` |
| 3.0 Think | `-0.1277` | `-0.0553` | `+0.0724` |
| 3.1 Think | `-0.1674` | `-0.0491` | `+0.1183` |
| 3.1 Instruct | `-0.0224` | `-0.0177` | `+0.0047` |

The common-base-lens view preserves the broad pattern. The strongest current reading is:

- the Bank-S negative direct effect appears between base and 3.0 Think;
- it strengthens at 3.1 Think;
- the composed prompt is less dependent on the internal J channel than the direct prompt, producing the positive composition contrast;
- that pattern collapses at the 3.1 Instruct sibling;
- the pattern is not explained by switching between the tested own and common coordinate frames.

The limitation remains important: each checkpoint fixed its own capability cohort before interventions. The current trajectory is not a fact-paired training contrast.

## 1.3 Qwen fit state

At the reviewed branch head:

- draw-A n=120 is registered as `p4-qwen-lens-fit-drawA-n120-dev-v1`;
- the n=120 lens hash is `82af4cc7f637af33e166606b15993bd6c67d2ea764c9788b96aa5a2120c32b1b`;
- the all-layer float32 recovery checkpoint is approximately 6.6 GB;
- fitting costs approximately 180 seconds per prompt under the fused FLA runtime;
- the last documented atomic recovery boundary is n=174;
- the active target is n=250;
- the fitter and config hashes are part of the recovery contract and must not change before the current cumulative series is complete.

The n=120 versus published n=1000 comparison reports, at representative layers:

| layer | matrix cosine | token median cosine | token q05 | CKA | relative Frobenius delta |
|---|---:|---:|---:|---:|---:|
| L0 | 0.541 | 0.530 | 0.276 | 0.452 | 1.109 |
| L24 | 0.886 | 0.895 | 0.815 | 0.867 | 0.471 |
| L32 | 0.904 | 0.922 | 0.826 | 0.841 | 0.427 |
| L40 | 0.961 | 0.968 | 0.931 | 0.943 | 0.280 |
| L62 | 0.9997 | 0.9997 | 0.9994 | 0.9997 | 0.023 |

This is a depth-dependent transfer result, not a fit-size result.

## 1.4 Phase 3 bridge state that Phase 4 must inherit correctly

The final Phase 3 state of record is sharper than the initial Phase 4 plan assumed:

- the protected span-safe Qwen tail is confirmatory and replicated;
- true-bridge protection versus the frozen chosen distractor is confirmatory but not held-out replicated;
- counterfactual bridge injection on the development cohort moves both log-probability preference and greedy generation toward the counterfactual;
- direct counterfactual-answer-direction injection produces a similarly large preference movement;
- bridge injection minus direct answer-direction injection is unresolved.

Therefore Phase 4 does not merely need “a better swap.” It needs a design that distinguishes:

1. a bridge-level state consumed through the prompted relation;
2. a direct answer-direction route;
3. generic disruption or vector injection;
4. a semantic but non-abstract shortcut from bridge tokens to answer tokens.

---

# 2. Full review of the newest Phase 4 results

## 2.1 The OLMo trajectory is useful, but the next analysis should remove cohort drift

The current trajectory is methodologically stronger than the earlier Part 2 ladder because it has:

- prospective boundary-safe capability scoring;
- exact pinned checkpoints;
- full alias scoring;
- span-safe J and an exact rank/energy control;
- shared scientific seed namespaces for own/common comparisons;
- identical condition order across frames;
- independently reconstructed bootstrap distributions;
- explicit separation of the 3.1 Instruct sibling from the temporal Think path.

The remaining weakness is now concentrated and cheap to address. The report states that no cross-checkpoint paired deltas are computed because the capability cohorts differ. That is correct for the headline development table, but the existing rows should still support a **post hoc development common-support analysis** on facts capable at all four checkpoints.

This analysis cannot become confirmatory. It can answer whether the trajectory is primarily:

- a changing item population;
- a changing baseline difficulty distribution;
- a within-fact change on shared items;
- a mixture of all three.

The common-support result should be produced before the OLMo section is considered mature enough for the paper.

## 2.2 The Qwen n=120 result is informative and appropriately cautious

The new structural producer has several strong properties:

- all 63 source-layer matrices are compared;
- all 5,120 matrix rows enter the operator metrics;
- token directions use a stable, hash-pinned sample;
- CKA and random transport probes complement raw matrix cosine;
- exact model tensors and final-norm convention are verified;
- the figure explicitly says it is not a same-corpus convergence test;
- the report refuses to infer whole-lens validation from L62.

This is good research engineering. The figure does what it says.

## 2.3 Raw late-layer agreement is partly inevitable

The current figure compares the full transport matrices `J_l`. For a residual network, transport from a late source layer to the final layer contains a large identity-like component. As the source approaches the target, two independently fitted maps can have near-unit raw cosine because both are dominated by the same residual path even if their non-identity updates differ.

This does not make the raw metric wrong. It changes what it answers:

- raw `J` agreement measures agreement of the total transported state;
- `J - I` agreement measures agreement of the downstream update around the residual path;
- `J - alpha I`, with `alpha = trace(J)/d`, measures agreement after removing the best scalar identity component.

All three should be reported. Otherwise the visual march toward 1.0 can look more like estimator convergence than it is.

A minimal implementation is:

```python
from __future__ import annotations

import torch


def split_identity_component(J: torch.Tensor) -> tuple[torch.Tensor, float]:
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError("J must be square")
    d = J.shape[0]
    alpha = torch.trace(J.float()) / d
    I = torch.eye(d, device=J.device, dtype=torch.float32)
    residual = J.float() - alpha * I
    return residual, float(alpha.item())
```

For each layer, add:

- `identity_scale_alpha`;
- `identity_fraction_frobenius`;
- cosine and symmetric relative delta for `J - I`;
- cosine and symmetric relative delta for `J - alpha I`;
- random-probe agreement on the residualized operators.

The paper should use raw `J` for total transport and identity-adjusted `J` for the claim that the learned coordinate transform has converged.

## 2.4 The current “n-weighted merge shift” is algebraically redundant

The structural code defines:

```python
merged = (n_c * J_c + n_r * J_r) / (n_c + n_r)
merge_shift = ||merged - J_r|| / ||J_r||
```

Therefore:

```text
merge_shift
= n_c / (n_c + n_r) * ||J_c - J_r|| / ||J_r||
```

It is an exact scalar multiple of the relative Frobenius delta. It supplies no independent sensitivity information. It also assumes that candidate and reference prompts are exchangeable draws from a common estimator population, which has not been established for the published reference.

Replace panel D’s orange line in the next figure with one of these genuinely new quantities:

1. same-corpus incremental-block shift;
2. identity-adjusted symmetric delta;
3. task-selected span disagreement;
4. leave-one-out influence of prompt 112;
5. estimated standard error from independent draw A/B fits.

Keep the historical p4f09 artifact immutable. The successor figure should explain the correction rather than silently changing the old panel.

## 2.5 Uniform vocabulary sampling needs task-relevant strata

The fixed 4,096-token uniform sample is a good global diagnostic, but it gives equal weight to high-frequency content tokens, rare byte fragments, formatting tokens, and special or nearly unused vocabulary rows. Add named strata without replacing the uniform sample:

- all Phase 3 Qwen answer token IDs;
- all Phase 3 bridge token IDs;
- future Bank B bridge and answer IDs;
- future Bank W state-symbol and answer IDs;
- frequency deciles measured on a held-out corpus;
- special/control token IDs reported separately.

A lens can have strong global row cosine while changing exactly the task-relevant directions that determine selection. Conversely, low uniform-vocabulary cosine can coexist with stable task selections because the dictionary is overcomplete. The task-relevant views adjudicate that ambiguity.

## 2.6 Prompt 112 is a required influence analysis, not a trimming invitation

Prompt 112 has a per-record Jacobian norm of approximately `159.952`. Retaining it in the canonical frozen estimator is correct. Ignoring its influence is not.

Run the exact per-prompt estimator on prompt 112 under the same source layers, target, valid-position mask, dtype, and runtime. Then form the exact leave-one-prompt sensitivity:

```python
J_without_112 = (120 * J_120 - J_112) / 119
```

This is valid only if `J_112` is the same equal-prompt estimator contribution used by the fit. Assert that contract with a tiny-model reconstruction test.

Compare full n=120 versus leave-112-out on:

- raw and identity-adjusted operator metrics;
- task-token row cosines;
- selected IDs and selected spans on the fixed development subset;
- capacity at L24/L32/L40;
- the small functional causal gate.

Decision wording:

- if influence is negligible, report a retained but non-load-bearing outlier;
- if influence is material, retain the canonical fit, report the sensitivity, and require n=250 or larger for the canonical Phase 4 Qwen lens;
- never trim and refit the canonical lens after inspecting the outcome.

## 2.7 Structural similarity is not the paper’s decision variable

The paper’s conclusions depend on what the lens causes the experiment to select and remove. The convergence study is incomplete until it measures:

- readout ranks and pass@k;
- per-position selected row IDs;
- selected-span principal angles or projector overlap;
- selected rank and removed-energy profiles;
- protected-span overlap and lost-rank behavior;
- occupancy and centered excess capacity;
- G4 swap positive-control performance;
- span-safe J versus exact matched-control effects;
- bridge protection, rescue, and semantic preference.

The core question is not “are the matrices close?” It is:

> Do different defensible lens fits induce the same scientific conclusion on the same model, items, intervention dose, controls, and scoring contract?

## 2.8 The published n=1000 lens needs a provenance classification

The current code verifies its hash, shape, layer set, and prompt count. It does not prove that every fit recipe field matches the new draw-A fitter. Before using it as a sample-size endpoint, pin whatever can be recovered from the official artifact or model card:

- fitting corpus identity and split;
- exact prompt ordering or construction if available;
- target layer;
- sequence length;
- skipped positions;
- BOS/native convention;
- `jlens` revision;
- merge semantics;
- dtype and save format.

If some fields are unavailable, classify it as:

```text
external published reference, partially specified recipe
```

and never call A120 versus published1000 a pure n comparison. The new same-corpus A120/A250/A500/A1000 chain becomes the fit-size evidence of record.

---

# 3. Phase decision and governance changes

## 3.1 This document governs as Phase 4.2

Phase 4.1 created the release audit, Phase 4 package, primary family, and initial schedule. Phase 4.2 is a delta plan. It does not discard 4.1. It updates priorities in light of:

- the completed Phase 3 state-of-record audit;
- the completed four-checkpoint OLMo trajectory;
- the Phase 3 semantic-swap result and answer-direction confound;
- the active Qwen nested fit;
- the first depth-dependent structural comparison.

On conflict, this document governs the next development block and the candidate-preregistration revisions described here.

## 3.2 Do not cut a Phase 5 namespace yet

A Phase 5 namespace becomes appropriate only after:

1. Phase 4 preregistration freezes;
2. P4-P1/P2/P3 confirmatory outcomes run once;
3. the replication partition runs;
4. Phase 4 independent reproduction closes;
5. the Phase 4 report reaches a state-of-record tag.

Before those events, `5_1` would separate implementation from its own scientific design.

## 3.3 Canonical Qwen lens decision

Freeze the following decision rule before inspecting n=250 functional outcomes:

- The published n=1000 lens remains the canonical Phase 3 lens.
- Phase 4 may nominate a smaller canonical lens only if it passes both a **structural gate** and a **functional gate** against the next larger same-corpus milestone and an independent-corpus fit.
- Use the smallest n that passes. This is an efficiency result, not a reason to prefer small n.
- If n=500 does not pass, use draw-A n=1000 for Phase 4 Qwen confirmatory work.
- If structure changes but all functional endpoints are equivalent, keep the larger lens as canonical and report functional robustness to smaller fits.
- If functional endpoints change beyond the frozen SESOI, fit size is load-bearing and all affected capacity or causal comparisons must use the canonical larger lens.

## 3.4 Revise P4-P1 before freeze

The current candidate’s bridge-versus-unrelated primary is no longer enough. Phase 3 already establishes that counterfactual bridge injection moves behavior relative to unrelated injection on a development cohort. The unresolved question is whether the bridge route does more than inject a downstream answer direction.

Use a gate-kept P4-P1 with one family-wise p-value:

- **P4-P1a, semantic movement gate:** counterfactual bridge injection versus geometry-matched unrelated injection.
- **P4-P1b, bridge-specific gate:** counterfactual bridge injection versus counterfactual answer-direction injection.

The endpoint rejects only if both one-sided tests reject. An intersection-union p-value can be reported as:

```text
p_P4P1 = max(p_semantic, p_bridge_specific)
```

This controls the endpoint at alpha without spending a fourth Holm slot. The licensed claim depends on the branch:

- P1a and P1b reject: bridge-specific semantic route beyond direct answer steering;
- P1a rejects, P1b does not: semantic substitution, but bridge abstraction unresolved;
- P1a does not reject: no replicated semantic substitution;
- generation moves mainly to other-invalid: disruption, not substitution.

## 3.5 Revise P4-P3 before freeze

Replace “positive on at least one model” with either:

1. a jointly calibrated max-T test across the three model-specific load slopes; or
2. a single model selected before outcomes using capability and intervention-identifiability gates only.

Recommendation: use the max-T test so no model is privileged after seeing the Bank W development pattern.

For model `m`, define positive load dependence as:

```text
D_m = -[specific_high(m) - specific_low(m)]
```

where larger `D_m` means more J-specific damage at high load. The primary statistic is:

```text
T = max_m D_m / SE_m
```

Calibrate the null by joint family-level sign flips that preserve the cross-model pairing. Named model estimates and the OLMo-pair/Qwen interaction remain secondary.

## 3.6 Gemma remains out of the next 24-hour block

No Gemma GPU work should run while the Qwen canonical-lens decision, bridge bank, mode gate, and Bank W are open. The Gemma exact-JVP autopsy remains a later Phase 4 methods block with the two-block cap from 4.1.

---

# 4. Next 24-hour execution schedule

The schedule is written for one 96 GB-class GPU and concurrent CPU authoring. It is outcome-conditional by design.

## 4.1 Boundary 0: bootstrap and recovery, 0 to 30 minutes

1. Pull branch `interp_jspace_part2` and verify the reviewed head or a descendant.
2. Run `bash interpretability/jspace_phase4/repro.sh`.
3. Verify all Phase 4 tests are green.
4. Verify the exact Qwen model snapshot, published lens, nested corpora, `jlens` revision, runtime packages, and FLA bindings.
5. Read both local and Drive recovery headers and choose the highest valid boundary.
6. Require `next_idx >= 174` if resuming the reviewed state. If a later durable boundary exists, use it.
7. Do not edit:
   - `p4_qwen_nested_lens_fit.py`;
   - `p4_qwen_nested_lens_fit_dev.yaml`;
   - the nested corpus files;
   until the desired cumulative milestones are complete. The recovery contract pins their hashes.

Resume:

```bash
cd /content/labs/interpretability/jspace_phase4
python -m jspace_phase4.experiments.p4_qwen_nested_lens_fit \
  --config configs/p4_qwen_nested_lens_fit_dev.yaml \
  --draw draw_a \
  --stop-at 250
```

## 4.2 GPU Q1: finish draw-A n=250, approximately 4 hours from n=174

Acceptance conditions:

- exact recovery contract matches;
- all 48 linear-attention blocks bind to the pinned FLA kernels;
- no CPU fallback;
- cumulative prompt count reaches exactly 250;
- checkpoint and fp16 lens hashes verify after Drive copy;
- evidence ID `p4-qwen-lens-fit-drawA-n250-dev-v1` registers once;
- no output path used by n=120 is overwritten;
- progress ledger records per-prompt rate and peak VRAM.

Immediately commit and push the registry boundary. Do not start another fit while the tree is dirty.

## 4.3 CPU Q2: build the convergence-v2 producer while the fit runs

Create:

```text
jspace_phase4/experiments/p4_qwen_lens_convergence.py
configs/p4_qwen_lens_convergence_drawA_dev.yaml
tests/test_qwen_lens_convergence.py
```

It must reuse the exact token-ID and Rademacher-probe sample hashes from p4f09. It must support arbitrary registered lens pairs without deriving new samples from the evidence ID.

Required comparison views:

1. A120 versus A250;
2. A250 versus published1000;
3. A120 versus published1000, historical continuity;
4. incremental block mean:

```python
J_increment_121_250 = (250 * J_250 - 120 * J_120) / 130
```

5. raw `J`, `J-I`, and `J-alpha I` metrics;
6. task-token strata and uniform-vocabulary strata;
7. layer-type annotations;
8. per-layer norm ratio and token-row norm-ratio quantiles;
9. explicit assay-band aggregate over L20-L44;
10. no algebraically redundant merge-shift line.

Register:

```text
p4-qwen-lens-convergence-drawA-n120-n250-dev-v1
```

Suggested figure `p4f10`:

- panel A: raw and identity-adjusted matrix cosine;
- panel B: token cosine by uniform, answer, and bridge strata;
- panel C: CKA and principal-subspace similarity;
- panel D: A120 versus A250 incremental-block disagreement;
- shade L20-L44 and mark L24/L32/L40;
- annotate attention-layer type without cherry-picking spikes.

## 4.4 GPU Q3: prompt-112 influence, approximately 5 to 20 minutes of fitting plus analysis

Create a separate producer that computes the exact per-prompt J contribution for draw-A row 112 under the frozen fit contract. Do not modify the cumulative fitter.

Register:

```text
p4-qwen-lens-influence-prompt112-dev-v1
```

Compare full A120 and A120-minus-112 structurally first. Do not run a full causal grid unless the structural movement is material.

Predeclare “material” as any of:

- assay-band token median cosine changes by more than 0.02;
- assay-band token q05 changes by more than 0.05;
- identity-adjusted matrix cosine changes by more than 0.03;
- selected-ID Jaccard on the fixed functional subset changes by more than 0.05;
- centered excess capacity changes by more than 0.5 percentage points at L24/L32/L40.

The canonical n=120 lens remains unchanged regardless of this sensitivity.

## 4.5 GPU Q4: fixed multi-lens functional gate, approximately 2 to 4 hours

Create one runner that loads Qwen once and evaluates registered lenses in a frozen order:

```text
published n=1000
new draw-A n=120
new draw-A n=250
optional A120-minus-112 sensitivity
```

Use one predeclared development subset drawn only from previously consumed Phase 3 families. Freeze the item IDs before viewing any new lens-specific outcomes. Recommended minimum:

- 30 direct/composed fact pairs across at least 12 canonical families;
- 20 bridge-mediation facts across at least 8 families;
- the exact G4 positive-control set;
- 40 held-out prose items for an NLL guard;
- a fixed capacity activation set disjoint from every fit corpus.

Run all lenses under:

- identical tokenizer and scoring contract;
- identical condition order;
- identical stable seed namespace;
- identical answer aliases;
- lens-specific span-safe J profiles;
- lens-specific exact matched controls consuming those profiles;
- no generated model-selection decisions during the run.

Record:

### Readout and selection

- answer and bridge pass@1/5/20;
- per-position selected-ID Jaccard;
- normalized projector overlap;
- largest and median principal angle;
- selected score rank correlation;
- selected effective rank;
- removed energy;
- protected span overlap and lost rank.

### Capacity

- paper-defined occupancy;
- centered excess variance at median occupancy;
- raw share as sensitivity;
- random-control crossing diagnostics;
- paired prompt bootstrap across lens fits.

### Causal behavior

- G4 flip rate;
- span-safe J and exact-control answer-sequence deltas;
- tail rate at the frozen Phase 3 threshold curve;
- true-bridge protection rescue;
- counterfactual preference margin;
- original/counterfactual/other-invalid generation trichotomy;
- prose NLL per token.

Register:

```text
p4-qwen-multilens-functional-gate-a120-a250-published-dev-v1
```

## 4.6 Branch rule after the functional gate

The next fit is selected without ad hoc debate.

### Branch A: functionally stable, structurally improving

Take this branch if all are true:

- A120 versus A250 assay-band token median cosine is at least 0.95;
- task-token q05 is at least 0.90;
- normalized selected-span overlap is at least 0.85;
- selected-ID Jaccard is at least 0.75;
- occupancy differs by at most one unit;
- centered excess differs by at most one percentage point;
- span-safe specific mean differs by at most 0.15 nats;
- tail rate differs by at most five percentage points;
- G4 differs by at most ten percentage points and passes under both;
- bridge rescue/preference differs by at most 0.25 nats and keeps sign.

Then fit **draw-B n=120** next. The main remaining uncertainty is corpus draw, not obvious same-corpus sample insufficiency.

### Branch B: functional instability or poor A120-A250 convergence

If any load-bearing functional criterion fails, continue draw A to **n=500** before spending GPU time on draw B. The sample-size slope is unresolved.

### Branch C: structural instability but functional equivalence

If structural gates fail while every causal/capacity endpoint is within SESOI, record:

```text
coordinate structure remains fit-sensitive; tested scientific endpoints are functionally stable
```

Fit draw-B n=120 next, keep published n=1000 as canonical, and do not force matrix equivalence as a requirement for publishing functional robustness.

## 4.7 Remaining GPU time

If Branch A:

```bash
python -m jspace_phase4.experiments.p4_qwen_nested_lens_fit \
  --config configs/p4_qwen_nested_lens_fit_dev.yaml \
  --draw draw_b \
  --stop-at 120
```

Then run A120 versus B120 structural and functional comparisons. Use any remaining time to resume draw A toward n=500.

If Branch B:

```bash
python -m jspace_phase4.experiments.p4_qwen_nested_lens_fit \
  --config configs/p4_qwen_nested_lens_fit_dev.yaml \
  --draw draw_a \
  --stop-at 500
```

If the VM ends before n=500, bank the highest atomic 3-prompt boundary. Do not create a pseudo-milestone evidence row for a partial fit.

## 4.8 CPU work that must run in parallel

While GPU fitting proceeds:

1. implement the all-four-checkpoint OLMo common-cohort analysis;
2. author Bank B candidate and its source-verification schema;
3. author Bank W generator and shortcut audits;
4. write the P4-P1 and P4-P3 preregistration amendments;
5. implement joint max-T simulation and type-I calibration;
6. scaffold the Qwen official mode parser and phase-hook goldens;
7. refresh the report, TeX, PDF, and evidence registry after n=250 and after the functional gate.

---

# 5. Qwen lens convergence protocol in full

## 5.1 Comparison matrix

The complete development matrix should eventually be:

| comparison | holds fixed | changes | identifies |
|---|---|---|---|
| A120 vs A250 | corpus prefix, recipe, runtime | n | early sample-size convergence |
| A250 vs A500 | corpus prefix, recipe, runtime | n | continued convergence |
| A500 vs A1000 | corpus prefix, recipe, runtime | n | final same-corpus convergence |
| A120 vs B120 | n, recipe, runtime | corpus draw | corpus sensitivity at small n |
| A500 vs B500 | n, recipe, runtime | corpus draw | corpus sensitivity at larger n |
| A1000 vs published1000 | n, model, broad dataset family | exact corpus/possibly recipe | external reference transfer |
| A120 full vs minus112 | everything but one prompt | influence | outlier sensitivity |

Do not collapse these into one “fit size” plot.

## 5.2 Structural metrics

For every layer and every comparison, report:

### Total operator

- raw matrix cosine;
- Frobenius norm ratio;
- symmetric relative delta;
- random transport-probe cosine and relative error.

### Identity-adjusted operator

- `alpha = trace(J)/d`;
- `||alpha I|| / ||J||`;
- cosine and delta of `J-I`;
- cosine and delta of `J-alpha I`;
- top singular-value and stable-rank profiles of the residualized map;
- principal angles among leading residualized singular subspaces.

### Token dictionary

- token-direction cosine quantiles;
- token-row norm-ratio quantiles;
- centered linear CKA;
- rank correlation of activation scores on actual residual states;
- frequency-stratified and task-stratified views.

### Selection geometry

- top-k selected ID Jaccard;
- normalized projector overlap;
- principal-angle distribution;
- pursuit coefficient correlation;
- effective-rank agreement;
- removed-energy agreement;
- protection-blocked row agreement.

## 5.3 Functional equivalence is paired, not visual

For a fixed item `i` and lens `a`, define:

```text
specific_i(a)
= [LP_J(a) - LP_base]
- [LP_control(a) - LP_base]
```

For lenses `a` and `b`:

```text
frame_delta_i(a,b) = specific_i(a) - specific_i(b)
```

Report equal-family means, paired family bootstrap intervals, exact family sign flips when feasible, and TOST against the frozen 0.15-nat SESOI. Correlation is a sensitivity, not an equivalence test.

For tail membership, report:

- rate difference;
- Jaccard;
- positive and negative agreement separately;
- threshold curve;
- family leave-one-out.

## 5.4 Capacity invariance

Capacity must use:

- the same activation evaluation records for every lens;
- records disjoint from draw A, draw B, and the published fit corpus to the extent known;
- the same random-control dictionaries and seeds;
- the same pursuit implementation and crossing rule;
- globally centered excess variance as the main estimand;
- occupancy, excess, and raw share all reported;
- uncertainty over activation prompts and random controls.

Do not infer a capacity-versus-causal regression from three models. The fit-size result is a within-model instrument sensitivity study.

## 5.5 Why n=250 is not a magic threshold

The purpose of n=250 is to estimate a convergence slope, not to bless a traditional round number. The canonical decision should be based on successive differences:

```text
D_120_250
D_250_500
D_500_1000
```

and independent draw differences:

```text
D_A120_B120
D_A500_B500
```

A defensible result can be:

- convergence by n=250;
- convergence only by n=500;
- no structural convergence but functional convergence;
- persistent corpus dependence;
- one influential prompt at small n;
- published-reference mismatch despite same-corpus convergence.

Every branch is publishable if measured cleanly.

## 5.6 Layer-type localization

Annotate Qwen layer type and test whether disagreement differs across:

- linear-attention blocks;
- any non-linear-attention/full-attention blocks;
- local depth before and after architecture transitions;
- layers with high versus low `||J||`;
- layers where task selections concentrate.

Use a preregistered mixed or permutation analysis over layers only as a descriptive methods result. Do not select “interesting” spikes after viewing the curve.

---

# 6. OLMo lineage closure before freeze

## 6.1 Common-cohort analysis

Create:

```text
jspace_phase4/experiments/p4_lineage_common_cohort_analysis.py
configs/p4_lineage_common_cohort_olmo_dev.yaml
tests/test_lineage_common_cohort.py
```

Construct four populations:

1. all-four-checkpoint intersection;
2. Base versus 3.0 Think intersection;
3. 3.0 versus 3.1 Think intersection;
4. 3.1 Think versus 3.1 Instruct sibling intersection.

Require each fact to have direct and composed rows at both endpoints. Report counts before outcomes.

For each population and both coordinate frames, compute:

- direct specificity;
- composed specificity;
- composed-minus-direct;
- adjacent checkpoint difference;
- baseline answer LP difference;
- capability margin difference;
- equal-family and item-weighted views;
- paired fact and family bootstrap;
- family leave-one-out;
- baseline-LP-adjusted sensitivity.

Register:

```text
p4-lineage-common-cohort-analysis-olmo-dev-v1
```

Interpretation rules:

- if the Think path persists on shared facts, cohort drift is not the main explanation;
- if it collapses, the trajectory is largely population selection;
- if only baseline-adjusted estimates persist, accessibility is a major mediator;
- no branch licenses a causal effect of training.

## 6.2 Crossed activation-frame decomposition

The own/common-lens analyses vary the lens while holding the recipient checkpoint fixed. Add a smaller decomposition on fixed prompts:

```text
checkpoint activation x lens frame
```

At minimum, compare base and 3.1 Think under:

- base activation, base lens;
- Think activation, base lens;
- Think activation, Think lens;
- optional base activation, Think lens as a readout-only diagnostic.

Record selected IDs, scores, spans, and causal effects. This separates:

- activation movement into an existing frame;
- frame drift;
- downstream consumption changes.

Keep this development-only and small unless it cleanly resolves the trajectory.

## 6.3 Intermediate checkpoints

Before downloading another 32B checkpoint, verify that it is a true adjacent or documented post-training stage with the same architecture and tokenizer. A random older release is not an intermediate training point.

Add an intermediate only if:

- its relation to base/3.0/3.1 is documented;
- exact revision and objective stage are known;
- a development cohort can be fixed before interventions;
- it adds temporal resolution rather than another endpoint.

Drop this before Bank W, P4-P1, or the Qwen mode study under compute pressure.

---

# 7. P4-P1 bridge mechanism redesign

## 7.1 What is already established

Phase 3 already shows:

- true bridge protection rescues more than a frozen chosen distractor;
- measured rank, energy, piece count, overlap, and answer cosine do not explain the rescue;
- counterfactual bridge injection moves preference and generation toward the counterfactual;
- direct counterfactual-answer-direction injection is nearly as strong;
- the bridge-minus-answer contrast is unresolved;
- the chosen distractor was not randomized;
- no held-out-family P3-P3 replication exists.

Phase 4 must not spend an untouched bank merely reproducing the easiest part of that ladder.

## 7.2 Bank B design

Author at least 40 untouched canonical families, but aim for 60 so confirmatory and replication sides remain useful after capability filtering.

Each fact bundle should include:

- direct prompt;
- composed prompt;
- true bridge supplied;
- two plausible counterfactual bridges;
- each counterfactual supplied;
- original answer;
- relation-appropriate counterfactual answers;
- unrelated bridge from a different family;
- two second-hop relations for a bridge wherever feasible;
- source-pinned facts and ambiguity notes;
- exact token IDs under the Qwen tokenizer;
- no prior bridge, answer, fact, or template overlap.

The two-relation construction is important. If the same bridge representation supports different relation-conditioned answers, a direct answer-direction explanation becomes less plausible.

## 7.3 Geometry-blind counterfactual selection

Select the confirmatory distractor before behavior outcomes using:

- token piece count;
- dictionary-row norm;
- span rank;
- angle to the true bridge span;
- angle to the answer span;
- achievable injection energy;
- protected-span overlap;
- baseline counterfactual-answer plausibility.

The selection objective and tie-break must be frozen. Never select the distractor that produces the largest swap.

## 7.4 Intervention arms

A full development factorial may include all candidate arms, but the confirmatory core should stay small:

1. baseline;
2. span-safe J reference;
3. exact matched control;
4. true bridge protection;
5. geometry-matched distractor protection;
6. remove true bridge, inject counterfactual bridge;
7. remove true bridge, inject counterfactual answer direction;
8. remove true bridge, inject geometry-matched unrelated bridge;
9. remove true bridge, inject orthogonal control.

Named development-only arms may include true reinjection, answer-only lesion, relation-swap, and receiver patching.

## 7.5 Endpoints

For each arm, score:

```text
M = LP(counterfactual answer) - LP(original answer)
```

and deterministic generation:

```text
original / intended counterfactual / other-valid / other-invalid / parse failure
```

Primary gate-kept contrasts:

```text
semantic = M_cf_bridge - M_unrelated
bridge_specific = M_cf_bridge - M_cf_answer_direction
p_P4P1 = max(p_semantic, p_bridge_specific)
```

Named secondaries:

- true versus distractor protection rescue;
- bridge-only lesion damage;
- true reinjection self-rescue;
- cross-relation transfer;
- receiver-state movement;
- phase and layer surfaces;
- generated counterfactual hit rate;
- other-invalid rate.

## 7.6 Receiver evidence

A stronger bridge claim needs at least one downstream receiver signature. On a development subset:

1. identify layers/positions where counterfactual bridge injection changes a bridge readout before the final answer readout;
2. show direct answer-direction injection does not reproduce the same intermediate bridge state;
3. patch the clean true-bridge receiver activation after a bridge lesion and test rescue;
4. use an unrelated receiver patch and wrong-layer patch as controls.

This can remain secondary, but it is the cleanest route from “semantic vector works” to “the model consumes a bridge state.”

---

# 8. Bank W: the experiment that decides the noun

## 8.1 Purpose

Phase 3 licensed **knowledge-access channel** and explicitly did not license a broad working-memory workspace. Bank W is not optional decoration. It is the assay that decides whether any stronger working-set language is earned.

It must also adjudicate the OLMo Think Bank-S anomaly. The key hypothesis is not simply “more items means more damage.” It is:

> An external, redundant prompt state can substitute for the internal J channel on Think models, while internally derived, low-redundancy state should require the channel more strongly.

## 8.2 Frozen axes

Use three crossed axes:

1. **load:** low versus high number of simultaneously relevant state elements;
2. **derivation:** state explicitly supplied versus internally derived;
3. **redundancy:** state stated once versus repeated or externally summarized.

Add delay/interference as controlled secondary axes, not additional primary degrees of freedom.

## 8.3 Six task superfamilies

1. **Key-value binding:** multiple entities, colors, keys, or locations; query one binding after distractors.
2. **State updates:** variables or registers updated over a sequence; query final state.
3. **Graph/path:** follow edges or relation chains with matched branching.
4. **Stack/queue:** push, pop, enqueue, dequeue, and delayed retrieval.
5. **Deferred recall with interference:** store items, process unrelated content, retrieve a designated item.
6. **Relational table/SQL-like state:** multi-row or multi-table bindings with controlled join or filter operations.

Each family must generate many item seeds without changing its inferential family ID.

## 8.4 Authoring target

Recommended minimum before capability filtering:

- 72 canonical template families, 12 per superfamily;
- 8 to 12 item seeds per family;
- balanced answer alphabet and token lengths;
- low/high prompts length-matched with semantically inert filler;
- direct answer leakage audit;
- order and position shuffle controls;
- shortcut models or heuristics that must fail;
- disjoint development, confirmatory, and replication families.

## 8.5 Capability gate

For each model and primary endpoint:

- baseline accuracy or answer margin must remain above the frozen floor at both low and high load;
- the low/high capability difference must fall inside a frozen equivalence margin, or a baseline-capability covariate model must be locked;
- enough common support must remain for family-level inference;
- prompt length and answer token count must not predict the endpoint after the matched design.

If these fail for a model, its load result is descriptive only.

## 8.6 Primary statistic

For model `m`:

```text
specific(load)
= [J(load) - baseline(load)]
- [matched(load) - baseline(load)]

D_m
= -[specific(high, derived, low-redundancy)
    - specific(low, derived, low-redundancy)]
```

Positive `D_m` means high internal load increases dependence on the J channel.

Use the joint max-T primary described in Section 3.5. Estimate the full load slope and all redundancy/derivation interactions as secondaries.

## 8.7 Interpretive decision rule

“Working-set channel under load” requires all of:

1. positive multiplicity-controlled load effect;
2. exact matched control near zero;
3. baseline capability guard passes;
4. effect is present on internally derived state;
5. effect is not explained solely by prompt length or difficulty;
6. effect replicates on held-out families.

If Bank W is null or capability-confounded, the project should retain **knowledge-access channel** and treat the Think Bank-S composition term as an external-state substitution pattern rather than global working memory.

---

# 9. P4-P2 official thinking-mode and phase study

## 9.1 Keep it same-weights and template-faithful

Use the exact Qwen checkpoint and its official chat-template control for thinking on/off. Do not approximate non-thinking mode by deleting visible tokens from a thinking prompt.

Pin:

- tokenizer revision;
- chat template hash;
- mode flags;
- reasoning delimiters;
- generation budget;
- stop tokens;
- parser version.

## 9.2 Phase conditions

Intervene separately during:

1. prefill;
2. generated reasoning tokens;
3. final-answer tokens;
4. all phases.

Every hook must record phase and position. Golden tests must fail if a hook fires in the wrong phase.

## 9.3 Quality controls

Use:

- correct rationale;
- wrong rationale;
- shuffled rationale;
- length-matched filler;
- no-rationale mode;
- clean mode.

Match token budgets. Report truncation and parse failures rather than forcing them into incorrect answers.

## 9.4 Primary metric freeze

Before untouched outcomes, choose one primary quality metric. Recommendation:

- deterministic normalized final-answer correctness as the primary generation endpoint;
- full accepted-alias answer margin under teacher-forced scoring on the generated context as a named secondary;
- parse failure and answer omission as explicit outcomes.

Freeze a single factorial interaction contrasting J-specific versus matched-control effects across mode and intervention phase. Do not search phases after the run.

---

# 10. Phase 4 preregistration freeze gate

The Phase 4 candidate may be signed only when every item below is complete.

## 10.1 Qwen lens and instrument

- [ ] draw-A n=250 registered;
- [ ] same-corpus A120/A250 structural comparison registered;
- [ ] prompt-112 influence registered;
- [ ] multi-lens functional gate registered;
- [ ] branch decision to A500 or B120 executed;
- [ ] canonical Qwen lens hash selected by the frozen rule;
- [ ] fit-size/corpus sensitivity plan locked;
- [ ] published lens provenance classified.

## 10.2 P4-P1

- [ ] Bank B family count and source audits pass;
- [ ] no overlap with prior facts, bridges, answers, or templates;
- [ ] counterfactual selector is outcome-blind;
- [ ] semantic and bridge-specific endpoint definitions frozen;
- [ ] answer-direction comparator included;
- [ ] geometry and dose tolerances frozen;
- [ ] generation trichotomy frozen;
- [ ] power and SESOI filled;
- [ ] confirmatory/replication split hash frozen.

## 10.3 P4-P2

- [ ] official mode template pinned;
- [ ] parser goldens pass;
- [ ] phase hooks pass;
- [ ] token budgets and truncation rules frozen;
- [ ] primary metric and interaction frozen;
- [ ] rationale controls frozen;
- [ ] families, power, SESOI, and split frozen.

## 10.4 P4-P3

- [ ] Bank W has at least 72 authored families;
- [ ] shortcut audits pass;
- [ ] load, derivation, and redundancy axes frozen;
- [ ] baseline capability guard frozen;
- [ ] max-T primary and joint permutation implementation calibrated;
- [ ] power and SESOI filled;
- [ ] development, confirmatory, and replication splits frozen.

## 10.5 OLMo and governance

- [ ] common-cohort trajectory analysis registered;
- [ ] lineage remains estimation-first in the prereg wording;
- [ ] no known-bank result enters a binary primary;
- [ ] environment and model/lens/tokenizer manifests complete;
- [ ] stable seed AST audit green;
- [ ] no unresolved `PENDING` markers;
- [ ] independent reviewer verifies no untouched outcome exposure;
- [ ] PI signs;
- [ ] freeze commit and tag created.

---

# 11. Implementation queue

## 11.1 New files for the immediate block

```text
interpretability/jspace_phase4/
  configs/
    p4_qwen_lens_convergence_drawA_dev.yaml
    p4_qwen_multilens_functional_gate_dev.yaml
    p4_lineage_common_cohort_olmo_dev.yaml
  jspace_phase4/experiments/
    p4_qwen_lens_convergence.py
    p4_qwen_lens_prompt_influence.py
    p4_qwen_multilens_functional_gate.py
    p4_lineage_common_cohort_analysis.py
    p4_author_bank_b.py
    p4_author_bank_w.py
    p4_bank_w_power.py
    p4_mode_gate.py
  tests/
    test_qwen_lens_convergence.py
    test_qwen_lens_prompt_influence.py
    test_qwen_multilens_functional_gate.py
    test_lineage_common_cohort.py
    test_bank_b.py
    test_bank_w.py
    test_bank_w_max_t.py
    test_mode_gate.py
```

## 11.2 Evidence IDs

Use stable names:

```text
p4-qwen-lens-fit-drawA-n250-dev-v1
p4-qwen-lens-convergence-drawA-n120-n250-dev-v1
p4-qwen-lens-influence-prompt112-dev-v1
p4-qwen-multilens-functional-gate-a120-a250-published-dev-v1
p4-qwen-lens-fit-drawB-n120-dev-v1             # branch A/C
p4-qwen-lens-fit-drawA-n500-dev-v1             # branch B or later
p4-lineage-common-cohort-analysis-olmo-dev-v1
p4-bank-b-candidate-v1
p4-bank-w-candidate-v1
p4-bank-w-power-dev-v1
p4-qwen-mode-gate-dev-v1
```

Never reuse an evidence ID merely because an old output was incomplete.

---

# 12. Required tests and falsifiers

## 12.1 Convergence tests

- raw identical matrices produce exact unit cosine and zero delta;
- scalar-identity changes are detected by raw metrics and separated by identity-adjusted metrics;
- a shared identity plus different low-rank update gives high raw cosine but low residualized cosine;
- fixed sample hashes are invariant to evidence ID;
- layer order and source-layer set must match;
- task-token strata are disjointly and deterministically constructed;
- a missing or partially specified published recipe is labeled, not guessed;
- the old merge-shift formula is asserted to be algebraically redundant and prohibited in the successor main figure.

## 12.2 Prompt influence tests

- tiny-model prompt contributions reconstruct the full mean exactly;
- leave-one-out subtraction matches a direct refit;
- wrong valid-position weighting fails;
- prompt 112 can never be silently dropped from the canonical lens;
- influence evidence is sensitivity tier only.

## 12.3 Functional gate tests

- lens order cannot change item order or random seeds;
- baseline logits are identical across lens conditions;
- exact matched controls consume each lens’s own achieved profile;
- selected-ID and span metrics recompute from raw rows;
- alias aggregation is prefix-disjoint logsumexp;
- TOST and tail decisions use frozen SESOI and threshold;
- a correlation alone cannot emit an equivalence verdict.

## 12.4 Bank B tests

- no prior fact/bridge/answer overlap;
- counterfactual answer follows from the counterfactual bridge and relation;
- unrelated bridge is from a different family;
- geometry selector never reads outcome columns;
- original and counterfactual aliases are prefix-disjoint;
- generated “other” outcomes are not counted as counterfactual hits;
- direct answer-direction arm is mandatory for primary eligibility.

## 12.5 Bank W tests

- hidden state is programmatically solved;
- low/high prompts have matched answer type and bounded length difference;
- positional and lexical shortcut solvers fail;
- state redundancy and derivation are independently manipulated;
- family ID corresponds to generator template, not item seed;
- confirmatory and replication families are disjoint;
- max-T null simulation controls type I error;
- no model is selected from intervention outcomes.

## 12.6 Mode tests

- official thinking on/off renders are pinned and distinct in the intended way;
- parser recovers phase boundaries on goldens;
- hooks fire only in requested phases;
- truncation is explicit;
- filler and shuffled rationale lengths match;
- parse failure stays a separate outcome.

---

# 13. Paper convergence map

The project is now close enough that every additional block should be justified by the sentence it may license.

## 13.1 Stable core already licensed

The Phase 2/3 paper can already support:

1. a span-safe, J-specific causal tail on Qwen beyond exact geometry matching;
2. held-out replication of that tail;
3. confirmatory true-bridge versus chosen-distractor protection rescue;
4. development evidence that counterfactual bridge injection moves semantic preference and generation;
5. a model split between Qwen bridge routing and OLMo accessibility/output-adjacent organization;
6. a negative boundary against selective global-workspace language;
7. a methods contribution on label versus span protection and exact controls.

## 13.2 What the Qwen convergence block adds

Possible outcomes and paper wording:

### Structural and functional convergence by n=250

> The Qwen causal and capacity conclusions are stable under a same-corpus 120-to-250 fit-size increase and an independent 120-prompt draw; the published 1000-prompt lens is not uniquely responsible for the result.

### Structural drift, functional stability

> J-space coordinates are fit-sensitive, especially early, but the selected spans, capacity estimate, and causal bridge/tail endpoints are stable. Functional conclusions are more robust than token-by-token coordinates.

### Functional instability

> Lens fit size is a load-bearing instrument degree of freedom. The canonical result requires the converged larger lens, and smaller-lens comparisons must be labeled pilot or sensitivity results.

### Persistent corpus dependence

> A fixed mean Jacobian is corpus-conditioned on Qwen. The paper must report an ensemble or distribution of lenses rather than one universal map.

Every branch is more informative than simply declaring the n=120 lens “good enough.”

## 13.3 What Bank W adds

### Positive replicated load effect

The strongest licensed noun can become:

```text
load-dependent working-set channel on the tested tasks and models
```

not an unrestricted global workspace.

### Null or capability-confounded load effect

Retain:

```text
knowledge-access channel
```

and present the Think Bank-S composition contrast as an external-state redundancy pattern.

## 13.4 What revised P4-P1 adds

### Bridge-specific gate passes

> Qwen carries a relation-usable intermediate bridge representation whose causal substitution cannot be reduced to direct steering of the final answer direction.

### Semantic gate passes, bridge-specific gate fails

> Counterfactual J-space injection steers Qwen toward the intended answer, but the current assay does not separate an abstract bridge state from a downstream answer-direction route.

This distinction should be decided by the preregistered branch, not by discussion prose after the run.

## 13.5 What P4-P2 adds

A mode-by-phase interaction can localize whether the bridge channel is most important during prefill, visible reasoning, or final answer formation. A null would be equally informative: the Qwen channel would not be reducible to the official thinking toggle.

---

# 14. Drop order under compute pressure

Do not spread the remaining GPU budget into thin, half-finished workstreams.

Drop in this order:

1. Gemma autopsy;
2. extra OLMo intermediate checkpoints;
3. receiver localization beyond the first positive slice;
4. draw-B n=500;
5. draw-A n=1000 if n=500 plus functional gates already close the decision;
6. broad mode-factorial secondaries.

Never drop:

1. completion of the current n=250 milestone;
2. same-corpus structural comparison;
3. multi-lens functional gate;
4. Bank W design and capability guard;
5. revised answer-direction-controlled P4-P1;
6. untouched partitions and preregistration freeze;
7. Phase 4 independent reproduction.

---

# 15. Coding-agent completion checklist for the next block

## Before GPU work

- [ ] clean tree;
- [ ] exact branch and current handoff read;
- [ ] Phase 4 test suite green;
- [ ] CUDA hard gate passes in the same process;
- [ ] exact runtime and FLA bindings pass;
- [ ] highest valid recovery header chosen;
- [ ] fitter and config left unchanged.

## At n=250

- [ ] lens and checkpoint hashes verified;
- [ ] event registered once;
- [ ] commit and push;
- [ ] p4f10 convergence analysis runs;
- [ ] identity-adjusted metrics included;
- [ ] task-token strata included;
- [ ] old merge-shift line not reused as independent evidence;
- [ ] prompt112 influence runs;
- [ ] report and handout refreshed.

## Functional gate

- [ ] fixed item manifest committed before lens outcomes;
- [ ] model loaded once;
- [ ] all lenses evaluated in frozen order;
- [ ] stable seed namespace shared;
- [ ] exact matched controls consume lens-specific profiles;
- [ ] capacity, selection, G4, tail, bridge, generation, and prose metrics banked;
- [ ] branch A/B/C selected mechanically;
- [ ] next fit launched only after the decision event is committed.

## CPU parallel work

- [ ] OLMo all-four common cohort analyzed;
- [ ] Bank B candidate reaches source and overlap audit;
- [ ] Bank W generator reaches shortcut-audit stage;
- [ ] P4-P1 candidate wording revised;
- [ ] P4-P3 max-T implementation calibrated;
- [ ] mode parser and hook goldens scaffolded;
- [ ] prereg candidate stays clearly unfrozen.

## Boundary

- [ ] all registered artifacts hash-verify;
- [ ] no evidence path overwritten;
- [ ] inprogress ledger gives exact resume command and highest checkpoint;
- [ ] Markdown, TeX, PDF, and figures refreshed;
- [ ] branch pushed;
- [ ] no Phase 4 confirmatory or replication outcome has run.

---

# 16. Bottom line

Stay in Phase 4.

The OLMo trajectory is now a substantive development result: a Bank-S J-dependence pattern appears on the Think path, persists across own/common lens frames, strengthens at 3.1 Think, and collapses at the Instruct sibling. The next cheap analysis is a common-fact cohort, not another broad checkpoint sweep.

The Qwen fit-size result is the current hinge. The n=120 lens agrees impressively late and only moderately in the lower/middle assay band, but the present comparison mixes fit size with an external reference corpus and possibly recipe metadata. Raw late-layer agreement also contains an identity-path contribution, and the plotted merge shift is algebraically determined by the Frobenius difference. These are reasons to improve the diagnostic, not reasons to discard it.

The next 24 hours should finish n=250, add identity-adjusted and influence diagnostics, and then test whether selected spans, capacity, the protected tail, G4, and bridge semantics are actually stable across lenses. That functional gate decides whether the next expensive fit is A500 or B120.

In parallel, move the Phase 4 primaries from good intentions to freezeable protocols. P4-P1 must beat direct answer-direction steering, not merely unrelated injection. P4-P3 must use a joint multiplicity-controlled load statistic. Bank W must cross load with redundancy and internal derivation so it can explain the Think anomaly and decide whether “working set” is deserved. Only after those pieces, the mode gate, power, untouched partitions, and reviewer sign-off are complete should Phase 4 freeze.

That path converges the project toward a paper instead of accumulating another shelf of interesting but mutually incomparable plots.
