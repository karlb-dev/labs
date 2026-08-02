# OLMo 32B J-space lineage development report

Last updated: 2026-08-02T02:29:26Z

Status: active; O1 and the O3 provenance audit are complete, the early Phase 4
bundle is emitted, and O2 is next. Both native OLMo-lineage baseline capability
results, the joint decision, the hash-pinned transfer bundle, and the four-lens
audit are registered. This report will be updated after every material gate or
analysis and is intentionally separate from the Phase 4 and Gemma reports.

## Executive status

This workstream asks which aspects of the thin verbalizable J-space are
conserved across the OLMo 32B lineage and which aspects are changed by
post-training. It treats six axes separately: coordinate availability, sparse
capacity, causal utilization, downstream consumption, external-state
substitution, and temporal organization. It does not estimate a single
"workspace score."

The O1 Phase 4 service obligation is complete: exact Bank-W baselines were run
for OLMo-3.1 Think and OLMo-3.1 Instruct, then aggregated with the imported
Qwen reference and transferred without intervention outcomes. The prospective
20-family common-support gate failed. O3 subsequently established that all six
lens pairs share the exact fitting recipe and ordered corpus, so no refit is
needed. The current priority is O2, which adds the missing Base capacity
measurement under a symmetric four-model estimator. O4 under the original
Bank-W protocol is gated out rather than silently narrowed.

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

`ol-foundation-v1` was created from clean commit `dc5f62336ac83481f09d`
at 2026-08-02T00:30:35Z. Its four immutable Drive outputs verify by SHA-256:

- environment lock: `728996e6760d...e79e9f54`;
- import manifest: `0988f5fd8db4...ce1a4dec`;
- conformance manifest: `58e4ce7828ff...e7bf9285`;
- foundation manifest: `0bd46f0c915e...d770a064e`.

The import gate resolved 33 live named source events, 13 direct artifacts,
three code dependencies, and six governance documents. Fourteen package tests
passed and same-process CUDA validation succeeded on the RTX PRO 6000
Blackwell GPU. The mutable recovery mirrors are indexed separately and are not
immutable evidence outputs.

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

`ol-bank-w-capability-protocol-v1` was frozen before
either OLMo baseline was opened, from clean commit `5e719c66321428a08d54`.
Its Drive manifest SHA-256 is `95db44f86f7d...c7dbda8dedc`. The isolated O1
producer and config pass 19 package tests, including
exact equality to the registered Phase 4 selection, answer/scoring contract,
capability guard, model order, tokenizer revisions, and answer token IDs. The
compatibility boundary directly imports the three hash-pinned Phase 4 source
files and verifies that the installed modules resolve to those repository
paths. It adds an explicit 384-row, no-drop, all-finite, complete-eight-answer
gate without changing the source analysis.

| model | rows | low acc. | high acc. | high-low | family-bootstrap 90% CI | capable families | independent gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| OLMo-3.1 Think | 384 | 0.7135 | 0.7188 | 0.0052 | [-0.0313, 0.0417] | 17/24 | pass |
| OLMo-3.1 Instruct | 384 | 0.7396 | 0.7188 | -0.0208 | [-0.0573, 0.0208] | 17/24 | pass |
| Qwen 3.6 27B, imported | 384 | 0.8333 | 0.8333 | 0.0000 | [-0.0208, 0.0208] | 20/24 | pass |

Think has 384 unique rows, all 4,608 checked numeric values are finite, every
row contains all eight candidate-sequence scores, and no row was dropped. The
low and high aggregate accuracy floors pass, as does the prospective load
equivalence interval. Think is therefore independently capability-eligible.
However, only 17 families meet the 0.70 within-family floor at both loads. Four
deferred-recall templates are near zero, while graph-path, key-value,
relational-table, and stack/queue families are mostly strong. The locked
prompt-length correlation with correctness is -0.188 and remains a secondary
sensitivity that cannot rescue or overturn the primary gate.

The 17-family Think support places a hard upper bound of 17 on the strict
Think/Instruct/Qwen intersection, below the prospectively required 20. The
complete Instruct result likewise has 17 capable families. Its four
deferred-recall templates are all at zero, while most graph-path,
relational-table, and stack/queue cells are strong. Its prompt-length
correlation with correctness is -0.211, again secondary only.

The registered exact three-model intersection contains 16 families. All three
models pass independently, but both the source Phase 4 aggregation and the
strict side service rule return BLOCKED because 16 < 20. The one-family losses
relative to each OLMo support set are asymmetric: Think alone retains
`key_value:template-09`, whereas Instruct alone retains
`state_updates:template-03`; neither belongs to the common set. No post hoc
model or family dropping is allowed. The complete baseline and joint analysis
were transferred in the early hash-pinned bundle, and no Bank-W intervention
outcome may be opened under this failed protocol.

### Early Phase 4 transfer

`ol-phase4-early-import-bundle-v1` was emitted at 2026-08-02T01:22:48Z from
clean source commit `dc20c90c2054d3c24b24505ccf4db8a5161ca88d`. The canonical
JSON bundle is `release/IMPORT_BUNDLE_PHASE4_EARLY.json` under the OLMo Drive
root (20,672 bytes; SHA-256
`debb29ef67ffa8741a4971ec2b0b21340bd5b48dc5729ac12f74f78839bf4f2b`).
Its Markdown companion is 863 bytes with SHA-256
`a7e6faf9ad412bd965cfbc9f7b1e9e98c9194cc5019e669d0695b5852a55159d`.
The JSON object-hash envelope verifies. Its embedded registry prefix covers
8,651 bytes through `ol-bank-w-capability-joint-dev-v1` and verifies to
`dcaca5a819a070f006a8534b820bfd476e0ebe63cd0b583412bfbfd050a79f10`.
The bundle explicitly records `olmo_phase4_service_ready=false` and
`interventions_opened=false`; it is an import handoff, not permission to alter
the active Phase 4 run root.

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

Pending model outcomes. `ol-capacity-protocol-v1` was registered at
2026-08-02T02:29:02Z from clean source commit `0e5800b`. It freezes the first
30 prompts in each of the four existing corpus blocks, for
120 prompts and 7,481 retained content positions. The selected canonical JSONL
hash is `695d29f9...a7948`; all four tokenizers produce the same ordered
content-token manifest `03edeb51...52e`. BOS is prepended explicitly when the
tokenizer exposes it, while selection positions are content-relative.

The primary pursuit target is `h` minus the one global model/layer population
mean. Raw `h` is pursued separately and labeled only as an uncentered energy
sensitivity. Each layer checkpoint retains full J/random error curves,
selected supports, prompt ownership, solver diagnostics, and deterministic
bootstrap distributions, allowing independent reconstruction and paired
cross-model resampling. Thirty package tests and the corpus/tokenizer/lens-
audit/source-pin preflights and 31 package tests pass. The registered corpus is
SHA-256 `695d29f9...a7948` and the protocol is SHA-256
`909c07d3...c9a0`; both verify in the eight-event registry. No O2 model outcome
has been opened. The
prior three-model 60-prompt files remain historical context only; Base
capability is not required for this capacity measurement.

## O3: lens provenance and coordinate comparability

Audit corpus hashes, extraction positions, layer alignment, seeds, split
logic, fitting hyperparameters, convergence, and held-out diagnostics for all
four lenses before considering a refit. Own-frame and common-frame questions
remain distinct. Common-frame stability is not evidence that dictionaries are
identical.

Results: `ol-lens-provenance-audit-v1` was registered at
2026-08-02T01:58:06Z from clean source commit `fb1fc73`. It validated every
final lens hash and internal container, all sixteen 30-prompt slice lenses,
and the four sampled weighted merges. Each merge differs from the mean of its
four fp16 slices by at most 0.0002442 at the audited coordinates, below the
frozen 0.002 serialization tolerance. All lenses attest 120 prompts, d=5120,
the same 21 source layers, target layer 63, dim batch 8, maximum sequence
length 128, skip-first 16, fp32 Jacobian accumulation, and fp16 storage.

The exact ordered raw-text corpus is shared. The three post-trained tokenizers
produce identical token sequences for all 120 texts. Base uses the same raw
texts/order but has no BOS token; the common model-aware `jlens.from_hf` policy
adds BOS only when exposed by the tokenizer. Historical evidence strength is
not uniform: the Instruct fit record names the deterministic source path but
does not embed its hash, and the original 3.0 Think fit loaded an unpinned Hub
ID. For 3.0 Think, the surviving historical and pinned revisions point to the
same 14 weight blobs and have identical semantic weight maps. These are
explicit qualifications, not hidden equivalence assumptions. All six pairwise
classifications are `EXACT_SAME_RECIPE_CORPUS`. All integrity checks pass, and
the formal decision is `no_refit_run_geometry_analysis`; geometry and capacity
analyses are authorized, while intervention outcomes remain closed. The
immutable JSON is SHA-256 `0912d223...81105` and the Markdown companion is
SHA-256 `9f6c8478...5f89`.

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

Result under Bank-W protocol version 1: gated out by O1 common-support failure;
no intervention outcome opened. O2 and O3 remain authorized. Any revised
Bank-W family service set would require a new prospective protocol and cannot
reinterpret the version-1 failure.

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
| 2026-08-02 | `ol-foundation-v1` | Frozen import boundary, environment, isolation, preregistration, and recovery contract; 14 tests and CUDA gate passed | Git commit `dc5f623`; four immutable Drive manifests |
| 2026-08-02 | O1 pre-outcome implementation | Exact Phase 4 compatibility wrapper, two-model config, finite/no-drop gates, resumable runner, and pull-before-push staging workflow; 19 tests passed; no model outcome opened | Git commit `5e719c6`, pushed |
| 2026-08-02 | `ol-bank-w-capability-protocol-v1` | Outcome-blind side protocol frozen; source Phase 4 fields and three compatibility module paths/hashes exact; 384 rows/model; no model or intervention outcome opened | Git commit `5e719c6`; Drive manifest `95db44f...` |
| 2026-08-02 | `ol-bank-w-capability-olmo31-think-dev-v1` | 384/384 finite rows; low/high 0.7135/0.7188; high-low 0.0052, 90% CI [-0.0313, 0.0417]; independent gate passes; 17 capable families makes the ≥20 strict joint service target impossible | Source commit `663e49f`; registry commit `b35adae`; result `944c739...`; rows `badde2c...` |
| 2026-08-02 | `ol-bank-w-capability-olmo31-instruct-dev-v1` | 384/384 finite rows; low/high 0.7396/0.7188; high-low -0.0208, 90% CI [-0.0573, 0.0208]; independent gate passes; 17 capable families | Source commit `b5dc5c3`; registry commit `241be17`; result `44f6399...`; rows `7322306...` |
| 2026-08-02 | `ol-bank-w-capability-joint-dev-v1` | All three models independently eligible; exact common support 16/20; source and side decisions BLOCKED; no intervention authorized | Source commit `241be17`; joint result `1f93c53...`; table `e09032a...` |
| 2026-08-02 | `ol-phase4-early-import-bundle-v1` | Emitted and verified the complete early capability handoff; service-ready false and no interventions opened | Source commit `dc20c90`; JSON `debb29e...`; Markdown `a7e6faf...`; registry prefix `dcaca5a...` |
| 2026-08-02 | O3 pre-evidence implementation | Hash-pinned four-lens provenance audit, tokenizer/BOS comparison, slice/merge integrity checks, and no-refit router implemented; 23 tests and all large-file preflights pass | Git commit `fb1fc73`, pushed; OLMo namespace only |
| 2026-08-02 | `ol-lens-provenance-audit-v1` | All six pairs are exact same recipe/corpus; all lens/slice/merge checks pass; no refit required; geometry and capacity authorized; no intervention opened | Source commit `fb1fc73`; JSON `0912d223...`; Markdown `9f6c8478...` |
| 2026-08-02 | O2 pre-evidence implementation | Symmetric 120-prompt corpus, centered-before-pursuit estimator, raw sensitivity, resumable layer checkpoints, common-frame analysis, paired prompt bootstrap, frozen classification router, and tests implemented; no capacity outcome opened | Git commit `0e5800b`, pushed; 31 tests pass |
| 2026-08-02 | `ol-capacity-protocol-v1` | Outcome-blind 120-prompt/7,481-position corpus, exact four-tokenizer agreement, centered and raw estimands, three random controls, paired bootstrap, and shift router frozen; no model/intervention outcome opened | Source commit `0e5800b`; corpus `695d29f9...`; protocol `909c07d3...` |

## Current limitations and claim boundary

The OLMo results support only independent baseline candidate-set capability on
this development partition, while the prospective joint-support service gate
fails. They do not establish capacity, causal
utilization, receivers, substitution, or temporal organization. The failed
family-support target blocks the planned service set rather than licensing a
post hoc family or model subset. The track will not claim that Think training
creates a global workspace, that the reasoning objective is the causal
variable, that Instruct lacks a verbalizable channel, or that capacity is
unchanged from pretraining before Base is measured.

## Recovery

Use `reports/INPROGRESS_OLMO_LINEAGE.md` for the exact current action and
`reports/OLMO_LINEAGE_RESUME.md` after a VM reclaim. Both are committed to Git
and mirrored into the Drive run root after every material checkpoint.
