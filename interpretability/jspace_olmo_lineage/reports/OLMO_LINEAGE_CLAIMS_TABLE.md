# OLMo lineage claims table

State date: 2026-08-02T06:09:33Z

Scope: OLMo-only parallel development/methods workstream on branch
`interp_jspace_olmo_lineage`, with immutable scientific import boundary
`3b041735d8b842de46a9c0a474fccd0c44e0841a`. This ledger does not alter the
Phase 4 or Gemma claim records. Every OLMo-native claim remains development or
methods tier.

## Conclusion-skeleton resolution

| Skeleton target | Original candidate | OLMo resolution | Exact licensed wording for this release |
|---|---|---|---|
| Sentence 2: training dependence | “What occupies the channel is set by training, not architecture alone.” | **Narrowed, development-tier.** O2 rules out a large increase in the measured sparse-capacity estimand, while O3 finds large Base-to-3.0 changes in J-mapped token and selected-span geometry. The checkpoints form a matched-lineage natural experiment, not a randomized training comparison, and finite-dose per-checkpoint transport remains queued. | “Across the tested, architecture-matched OLMo lineage, the first released Think transition is associated with substantial reorganization of J-mapped token and selected-span geometry without a material increase in measured sparse capacity; the design does not identify which training ingredient caused that change.” |
| Sentence 4: externalization | “Reasoning post-training installs external-state substitution: composed in-context state reduces reliance on the internal channel, and only on the tested Think checkpoints.” | **Downgraded to explicitly pending.** The predeclared O1 service gate failed at 16 jointly capable families versus 20 required, so no O4 Bank-W intervention cell was opened. Existing Bank-S development evidence can motivate the hypothesis but cannot choose among external-state substitution, generic difficulty, capacity, or output-adjacent accounts. | “Existing Bank-S development evidence motivates an external-state-substitution hypothesis on the tested Think checkpoints, but the planned fact-paired Bank-W factorial was gated out at 16/20 common families; this release does not resolve an externalization mechanism.” |

These replacements are the only paper-facing sentence-2 and sentence-4 forms
licensed by this OLMo release. The original stronger candidates must not be
silently retained.

## Claim ledger

| ID | Claim | Evidence and decisive result | Tier | Skeleton link | Status and allowed use |
|---|---|---|---|---|---|
| OL-C01 | Both OLMo-3.1 endpoints independently meet the frozen Bank-W aggregate capability/equivalence gate, but the OLMo/Qwen service cohort does not meet the prospective common-family minimum. | `ol-bank-w-capability-olmo31-think-dev-v1`: low/high accuracy 0.7135/0.7188, high-low 0.0052, 90% CI [-0.0313, 0.0417], 17/24 capable families. `ol-bank-w-capability-olmo31-instruct-dev-v1`: 0.7396/0.7188, -0.0208, CI [-0.0573, 0.0208], 17/24. `ol-bank-w-capability-joint-dev-v1`: exact intersection 16/20, service-ready false. | development | Sentence 4 gate | Licensed as a baseline-capability and service-gate result. It is not an intervention or externalization result. |
| OL-C02 | Measured sparse capacity/recruitment is broadly conserved across the four tested checkpoints under the symmetric estimator. | `ol-capacity-joint-dev-v1`: Base-to-3.0 Think own-frame equal-layer centered-excess difference 0.0001538 (+0.0154 percentage points), paired 90% CI [-0.0001211, 0.0004345], occupancy difference 0; all 12 equal-layer rows stable, 46/48 classified rows stable, two Base-common L40 rows unresolved. Frozen verdict `broadly_conserved_capacity_recruitment_consistent`. | development | Sentence 2 | Licensed only for the measured 120-prompt, layers-24/32/40 sparse-capacity estimand and frozen equivalence margins. Do not translate “stable” into literal equality or global coordinate conservation. |
| OL-C03 | J-space geometry shows a dictionary-formation pattern concentrated at the Base-to-3.0 transition. | `ol-geometry-joint-dev-v1`: Base-to-3.0 median raw-operator cosine 0.9614 but mapped-row cosine 0.6744; mapped movement 0.32556 versus 0.00604 for 3.0-to-3.1; Base-to-3.0 selected-ID Jaccard 0.3333 at layers 24/32/40; router `dictionary-formation-pattern`. | development | Sentence 2 | Licensed as descriptive matched-lineage geometry. It supports reorganization, not causal attribution to a Think objective and not a claim that the dictionaries are globally identical or globally different. |
| OL-C04 | The first released Think transition is associated with coordinate/selection reorganization without material measured capacity growth. | Joint reading of `ol-capacity-joint-dev-v1`, `ol-lens-provenance-audit-v1`, and `ol-geometry-joint-dev-v1`; all six lenses are exact same recipe/corpus, so no refit artifact is needed to explain the comparison. | development | Sentence 2 resolution | This is the release’s preferred sentence-2 claim. “Associated with” and “measured” are mandatory. Per-checkpoint finite-dose transport and controlled stage attribution remain open. |
| OL-C05 | OLMo-3.1 Instruct is a sibling endpoint, not a fourth temporal Think checkpoint, and it does not meet the prospective late-shift geometry rule. | `ol-geometry-joint-dev-v1`: Think/Instruct raw-operator median 0.9902, mapped-row median 0.9363, raw-unembedding median 0.9984; early mapped movement 0.2860, late movement 0.0341; `instruct_late_shift=false`. | development | Sentences 2 and 4 qualification | Licensed only as a sibling comparison. Never draw a temporal arrow from 3.1 Think to 3.1 Instruct or infer that Instruct lacks a verbalizable channel. |
| OL-C06 | O4 Bank-W externalization is unresolved because the predeclared service gate failed; no intervention result exists. | `ol-bank-w-capability-joint-dev-v1` and `ol-phase4-early-import-bundle-v1`: 16 common families < 20, `olmo_phase4_service_ready=false`, `interventions_opened=false`. | development + methods handoff | Sentence 4 resolution | Licensed as a gated-out result and explicit pending decision. It does not favor any of the four frozen O4 accounts. A redesigned Bank-W service set requires a new prospective protocol. |
| OL-C07 | Genuine official 32B Think SFT and DPO intermediate artifacts are available and semantically tokenizer-compatible for a bounded H5 wedge. | `ol-checkpoint-inventory-v2`: verdict `genuine-32b-intermediates-available`; H5 `testable-with-bounded-stage-wedge`; SFT and DPO queued-not-started. Version 2 explicitly supersedes the overly conservative byte-identity route in `ol-checkpoint-inventory-v1`. | methods | Sentence 2 upgrade path | Licensed as an availability and queue result only. No intermediate weights were opened by the event and no stage outcome or training-stage attribution is licensed. BOS/chat-template differences and repository-only parent declarations remain qualifications. |
| OL-C08 | The registered evidence cannot identify an activation-model × transport-lens × readout causal decomposition. | `ol-o5-feasibility-decision-v1`: `defer-no-identifiable-crossed-intervention-estimand`, `not-executed-no-proxy-substitution`; crossed intervention rows and required transport, protected-span, rank/energy, logit-lens, and row-order controls are absent. | methods | Sentences 2 and 4 limitation | Licensed as an identifiability decision, not a factor estimate or null. O2/O3 structural metrics must not be relabeled as O5 causal factors. |
| OL-C09 | The headline O1/O2/O3 computations and figures are independently reconstructable, and one exact model row replays without drift. | `ol-independent-reconstruction-v1`: 768 O1 rows; 48 O2 summaries, 96 intervals, 144 arrays, 72 table rows; 84 O3 aggregates/router; five byte-identical PNGs and five regenerated PDFs; 14 exact weight shards; all 8 candidate scores and the -0.25 margin reproduced with maximum drift 0. | methods | Reliability for sentences 2 and 4 | Licensed as reproducibility support. It does not raise the scientific tier or create an independent biological/statistical replication cohort. |
| OL-C10 | Exact kth/k+1 candidate-score gaps, protected-span overlap under alternate readouts, causal core/fringe dose, and per-checkpoint finite-dose transport are not measured in this release. | Explicit null fields in `ol-geometry-joint-dev-v1`; O1 gate and `ol-o5-feasibility-decision-v1`; H6 queue. | methods boundary | Sentence 2 limitation | These quantities must remain null/queued. The available marginal-gain threshold margin, projector overlap, or raw operator cosine is not a substitute. |

## Upgrade and downgrade router

### Sentence 2

Current disposition: **narrowed but resolved for this release**.

An upgrade from association to training-stage attribution requires the queued
official SFT/DPO wedge or another controlled continuation with frozen prompts,
tokenization, scoring, and transport checks. A claim of conserved usable
coordinates additionally requires per-checkpoint finite-dose transport, not
only operator or mapped-row cosine. Until then:

- use “architecture-matched,” “associated with,” and “measured sparse
  capacity”;
- retain the Base-to-3.0 coordinate/selection movement;
- do not write “Think training creates a workspace,” “training causes the
  dictionary,” or “the coordinates are conserved.”

### Sentence 4

Current disposition: **explicitly pending after a failed service gate**.

No version-1 O4 account survived or failed because the factorial never opened.
An upgrade requires a new, prospectively frozen Bank-W capability/service
protocol, a controlled load × derivation × redundancy grid, the frozen §6.6
router, and later untouched confirmation before any binary publication claim.
Until then:

- cite the 16/20 block whenever discussing externalization;
- describe Bank-S only as motivation or prior development evidence;
- do not write that reasoning post-training “installs” substitution or that
  supplied state causally replaces the internal channel.

## Phase 5 handoff claims

1. **H5 queue:** exact official Think-SFT and Think-DPO cells, bounded to the
   minimal two-stage wedge with Base/3.0/3.1 anchors; queued, not started.
2. **H6 transport queue:** validate finite-dose transport separately at every
   checkpoint before using “conserved coordinate system.”
3. **O5 entry:** Bank-S-first Base/3.1-Think/3.1-Instruct pilot crossing
   Base-versus-recipient transport and readout, expanded only after per-
   dictionary transport, protected-span, delivered-rank, and delivered-energy
   checks pass.
4. **Bank W:** closed under version 1. Any new service set is a new prospective
   study, not a repair or subset of the failed 16/20 cohort.

## Prohibited formulations

- “Think training creates a global workspace.”
- “The reasoning objective is the causal variable.”
- “Reasoning post-training installs external-state substitution.”
- “Capacity is unchanged” without the words “measured,” the frozen estimator,
  and its equivalence margin.
- “The dictionaries are identical” or “transport is valid” from structural
  geometry alone.
- “Instruct is the next Think checkpoint” or “Instruct lacks a verbalizable
  channel.”
- Any O4, O5, receiver, intermediate-stage, confirmatory, or replication
  outcome not present in the append-only OLMo registry.

## Citation rule

Paper and handoff prose should cite the claim ID above together with the named
evidence IDs. This ledger is copied into the hash-pinned final release bundle;
the append-only evidence registry and registered Drive outputs remain
authoritative.

## Final release attestation

The registered release copy is an output of
`ol-phase4-final-import-bundle-v1`, created from clean producer commit
`7148d0138471cd529144767b7cdb26adda5a6eae`. Its SHA-256 is
`682bf60cce10810af57258ebd672a0bcde8ce45a44bd851f0a5f289cedbf2951`.
Sentence 2 remains **narrowed-release-resolved** and sentence 4 remains
**explicitly-pending-gate-blocked**. Queued H5/H6/O5 work is not an outcome.

The machine bundle's YAML-folded sentence-4 value contains a nonsemantic space
after the line-ending hyphen in `external-state- substitution`. The registered
claims-ledger copy above retains the exact licensed
`external-state-substitution` wording. The verifier canonicalizes only this
source-wrap whitespace; no registered file was changed.
