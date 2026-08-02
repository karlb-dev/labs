# OLMo 32B J-space lineage development report

Last updated: 2026-08-02T00:27:29Z

Status: active, pre-foundation. No native OLMo-lineage model result has yet
been registered. This report will be updated after every material gate or
analysis and is intentionally separate from the Phase 4 and Gemma reports.

## Executive status

This workstream asks which aspects of the thin verbalizable J-space are
conserved across the OLMo 32B lineage and which aspects are changed by
post-training. It treats six axes separately: coordinate availability, sparse
capacity, causal utilization, downstream consumption, external-state
substitution, and temporal organization. It does not estimate a single
"workspace score."

The current priority is the O1 service obligation: exact Phase 4 Bank-W
baseline capability for OLMo-3.1 Think and OLMo-3.1 Instruct. Those runs are
baseline-only and precede all side-track Bank-W interventions. O2 then adds
the missing Base capacity measurement under a symmetric four-model estimator;
O4 tests the frozen lineage predictions on the development partition.

Phase 4, Gemma, and OLMo are running as concurrent but isolated branches of
work. They will be integrated only in Phase 5 or later after their own state
of record is complete.

## Governance and provenance

- Branch: `interp_jspace_olmo_lineage`.
- Branch parent: `4ea7a9ba7a534daa61e0d8c9960763b921a1b80b`.
- Scientific import boundary:
  `3b041735d8b842de46a9c0a474fccd0c44e0841a`.
- Repository namespace: `interpretability/jspace_olmo_lineage/`.
- Drive root:
  `/content/drive/MyDrive/interpret/special-lab-1/olmo_lineage_20260801`.
- Native evidence prefix: `ol-`.
- Native tiers: `development` and `methods` only.

The foundation manifest verifies the main resume, OLMo plan, accepted OLMo
addendum, Phase 4 parallel contract, imported evidence registries, every
direct artifact, and the side preregistration by exact SHA-256. Imported
evidence remains read-only and retains its original tier.

The branch separation is operational, not a license to expand the scientific
boundary. Documentation ancestry after the accepted Phase 3 state is recorded
but later Phase 4 evidence is imported only through named, live, hash-pinned
development/methods events. No confirmatory or replication intervention
outcome is imported or opened.

## Model graph

The temporal Think path is:

```text
OLMo-3 Base -> OLMo-3 32B Think -> OLMo-3.1 32B Think
```

OLMo-3.1 32B Instruct is a sibling endpoint under a distinct, incompletely
observed post-training recipe. It is never plotted or interpreted as a fourth
time point. Exact model revisions and lens hashes are frozen in the foundation
config.

## Imported context known before this study

The accepted prior work established a thin, measurable J-space dictionary and
development evidence that OLMo checkpoints have broadly related geometry but
differ in downstream causal organization. Prior capacity files used 60
prompts, so they are context rather than the final symmetric O2 comparison.
Base already has a registered lens, but its symmetric capacity cell is
missing.

Phase 4 has a frozen Bank-W development capability protocol. The already-run
Qwen reference contains 384 finite rows, low/high accuracy of 0.8333/0.8333,
a high-minus-low estimate of 0, a family-bootstrap 90% interval of roughly
[-0.0208, 0.0208], and 20 of 24 families capable at both loads. These numbers
remain Phase 4 development evidence and are not native OLMo results.

## Prospective hypotheses

- H1: a broadly conserved thin dictionary is recruited or routed differently
  after post-training.
- H2: supplied or redundant external state substitutes for the internal
  channel on Think checkpoints.
- H3: Instruct routes state through output-adjacent or alternative receivers.
- H4: continuous answer commitment moderates causal cost.
- H5: one unobserved post-training stage installs the Think-path effect; if no
  genuine intermediate checkpoint exists, this is stated-unresolvable.
- H6: finite-dose transport validity may moderate the apparent trajectory.

Predictions and classification margins were frozen prospectively in
`preregistration/OLMO_LINEAGE_DEVELOPMENT_PREREGISTRATION.md` before this
track opened any Bank-W baseline or intervention output.

## O1: Bank-W baseline capability

### Protocol

For each OLMo-3.1 Think and Instruct model: 24 families x 8 seeds x two load
levels for the derived/once cell, totaling 384 rows. Score all eight answer
sequences by summed conditional log probability; do not substitute a
first-token score. The run contains no intervention column.

A model passes only if low and high load accuracy are each at least 0.70 and
the family-bootstrap 90% interval for high-minus-low accuracy lies wholly in
[-0.08, +0.08]. Report all finite-row checks, capable-family counts, exact
tokenizer candidate encodings, and joint support with Qwen. The target joint
support is at least 20 families.

### Results

Pending. Think runs first, then Instruct. An early hash-pinned import bundle
will be emitted immediately after both gates and the joint-support analysis.

## O2: symmetric capacity

### Frozen design

Use the shared ordered 120-prompt corpus: 30 factual, 30 arithmetic, 30
SQL/code, and 30 neutral prose prompts. Measure layers 24/32/40 for all four
models, globally center activations, and use the paper-defined marginal-gain
crossing against three matched random dictionaries. Report occupancy, raw
share, random share, centered excess, prompt-bootstrap intervals, strata, and
solver diagnostics.

Pairwise classifications are stable below 0.25 percentage points with
unchanged occupancy and an equivalence interval inside the margin; small shift
at 0.25--1.0 points or one occupancy unit; material above 1 point or more than
one occupancy unit; unresolved when uncertainty is too wide.

### Results

Pending. The prior three-model 60-prompt files are imported only as historical
context. Base capability is not required for this capacity measurement.

## O3: lens provenance and coordinate comparability

Audit corpus hashes, extraction positions, layer alignment, seeds, split
logic, fitting hyperparameters, convergence, and held-out diagnostics for all
four lenses before considering a refit. Own-frame and common-frame questions
remain distinct. Common-frame stability is not evidence that dictionaries are
identical.

Results: pending.

## O4: development mechanism grid

The primary arm contrast is
`specific = (LP_J - LP_0) - (LP_C - LP_0)`, where C is an exact instantaneous
rank-and-energy matched control. Negative values mean greater J-specific
damage. Frozen contrasts cover load engagement, supplied-state substitution,
redundancy substitution, and load-by-externalization. Report equal-family
estimates, family bootstrap intervals, exact sign flips, leave-one-family-out
sensitivity, accuracy, and log-probability endpoints.

Base may be explicitly `gated_out` if Bank-W capability fails; it is never
silently omitted and never imputed as a zero effect. No O4 outcome will be
opened before O1, O2, and the O3 provenance audit required for its coordinates.

Results: pending.

## Intermediate-stage inventory and bounded follow-ups

The first intermediate-checkpoint action is inventory only: enumerate released
checkpoints, revision hashes, training-stage labels, architecture/tokenizer
compatibility, and whether each is a genuine temporal stage. Controlled
training is outside this release block. Receiver/temporal work is capped at
one clean within-model rescue demonstration and half a GPU day unless the
operative plan authorizes expansion.

## Execution ledger

| UTC | Evidence or action | Outcome | Durable locations |
|---|---|---|---|
| 2026-08-01 | Branch isolation | Created and pushed `interp_jspace_olmo_lineage` at `4ea7a9b`; scientific boundary remains `3b04173` | GitHub |
| 2026-08-02 | Pre-foundation scaffold | Package, prospective predictions, import pins, recovery plan, and reports committed at `e01959e`; 13 tests passed | GitHub |
| 2026-08-02 | Foundation refusal | Import gate detected stale preregistration hash after whitespace normalization; no evidence/output file was created; hash corrected prospectively | Git worktree; empty OLMo Drive directories |

## Current limitations and claim boundary

No native model-backed result exists yet, so the report makes no new claim
about OLMo capability, capacity, utilization, receivers, substitution, or
temporal organization. Existing evidence is development context. The track
will not claim that Think training creates a global workspace, that the
reasoning objective is the causal variable, that Instruct lacks a verbalizable
channel, or that capacity is unchanged from pretraining before Base is
measured.

## Recovery

Use `reports/INPROGRESS_OLMO_LINEAGE.md` for the exact current action and
`reports/OLMO_LINEAGE_RESUME.md` after a VM reclaim. Both are committed to Git
and mirrored into the Drive run root after every material checkpoint.
