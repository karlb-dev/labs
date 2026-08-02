# OLMo 32B J-space lineage development report

Last updated: 2026-08-02T03:28:59Z

Status: active; O1, O2, and the O3 provenance audit are complete, the early
Phase 4 bundle is emitted, and O3 geometry is next. Both native OLMo-lineage
baseline capability results, both joint decisions, the hash-pinned transfer
bundle, four capacity cells, paired capacity verdict, and four-lens audit are
registered. This report will be updated after every material gate or analysis
and is intentionally separate from the Phase 4 and Gemma reports.

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
needed. O2 has measured Base, both Think checkpoints, and the Instruct sibling
under the symmetric four-model estimator and completed the prospectively
paired router. It returns a broadly conserved capacity/recruitment-consistent
verdict. O3 geometry is now needed to test the coordinate-system portion of
that account. O4 under the original Bank-W protocol remains gated out rather
than silently narrowed.

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
Base already had a registered lens, but its symmetric capacity cell was
missing at study start; it is now measured under the prospective O2 protocol.

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

All four model outcomes are complete. `ol-capacity-protocol-v1` was registered at
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
cross-model resampling. Thirty-one package tests and the corpus/tokenizer/lens-
audit/source-pin preflights pass. The registered corpus is
SHA-256 `695d29f9...a7948` and the protocol is SHA-256
`909c07d3...c9a0`; both verify in the thirteen-event registry. The protocol itself
opened no model outcome. The prior three-model 60-prompt files remain
historical context only; Base capability is not required for this capacity
measurement.

`ol-capacity-olmo3-base-dev-v1` completed from clean source commit `f67efcd`
with all registered hashes verified. The primary own-frame results are:

| Layer | Occupancy | Centered excess | 90% prompt-bootstrap interval | Raw sensitivity (occupancy) |
|---:|---:|---:|---:|---:|
| 24 | 2 | -0.0137 pp | [-0.0702, +0.0437] pp | -0.1528 pp (1) |
| 32 | 2 | +0.3151 pp | [+0.2656, +0.3641] pp | -0.0008 pp (1) |
| 40 | 2 | +0.5896 pp | [+0.5342, +0.6497] pp | -0.0033 pp (2) |

All centered occupancies have zero censoring; every J error curve has zero
solver-error increases; and own/common are exact duplicates for Base by
construction. The result SHA-256 is `3708447c...069c0`. These are within-Base
capacity estimates and are not, alone, evidence of a checkpoint shift. The
completed paired classification is reported below.

`ol-capacity-olmo3-think-dev-v1` completed from clean source commit `04870ec`
with all six registered outputs verified. The primary own-frame results are:

| Layer | Occupancy | Centered excess | 90% prompt-bootstrap interval | Raw sensitivity (occupancy) |
|---:|---:|---:|---:|---:|
| 24 | 2 | +0.0650 pp | [+0.0405, +0.0922] pp | +0.0006 pp (1) |
| 32 | 2 | +0.3391 pp | [+0.3041, +0.3787] pp | +0.1854 pp (2) |
| 40 | 2 | +0.5331 pp | [+0.4909, +0.5791] pp | +0.5049 pp (2) |

All centered occupancies again have zero censoring and every J error curve has
zero solver-error increases. In the frozen Base-common lens frame, centered
excess is -0.0770, +0.1759, and +0.3685 percentage points at layers 24/32/40,
with occupancy 2 at each layer and respective 90% intervals
[-0.1239, -0.0300], [+0.1374, +0.2143], and [+0.3253, +0.4133] percentage
points. The own/common difference is a coordinate-frame sensitivity, not a
cross-checkpoint contrast. The result SHA-256 is `2a872e4a...58abc`; its three
independently reconstructable layer checkpoints are `e444dc6b...f8e6d`,
`60d4989c...75d5`, and `9d65e188...0e55`. The frozen router still requires
paired resampling from all four registered events, so no lineage shift label is
assigned from these independent point estimates or intervals; the completed
paired result is reported below.

`ol-capacity-olmo31-think-dev-v1` completed from clean source commit `7baaf64`
with all six registered outputs verified. The primary own-frame results are:

| Layer | Occupancy | Centered excess | 90% prompt-bootstrap interval | Raw sensitivity (occupancy) |
|---:|---:|---:|---:|---:|
| 24 | 2 | +0.0625 pp | [+0.0367, +0.0902] pp | -0.0268 pp (1) |
| 32 | 2 | +0.3275 pp | [+0.2928, +0.3665] pp | +0.1500 pp (2) |
| 40 | 2 | +0.5192 pp | [+0.4773, +0.5635] pp | +0.4861 pp (2) |

All centered occupancies have zero censoring and every J error curve has zero
solver-error increases. In the Base-common frame, centered excess is -0.0779,
+0.1693, and +0.3581 percentage points at layers 24/32/40, with occupancy 2
throughout and 90% intervals [-0.1249, -0.0305], [+0.1301, +0.2074], and
[+0.3144, +0.4029] percentage points. The result SHA-256 is
`f6af4791...908eb`; the layer checkpoints are `1ac1e445...c89fc`,
`ceb2c9d2...025ff`, and `9d69d40b...b29e6d`. This is another within-checkpoint
measurement; the completed paired classification is reported below.

`ol-capacity-olmo31-instruct-dev-v1` completed from clean source commit
`9f213f6` with all six registered outputs verified. The primary own-frame
results are:

| Layer | Occupancy | Centered excess | 90% prompt-bootstrap interval | Raw sensitivity (occupancy) |
|---:|---:|---:|---:|---:|
| 24 | 2 | -0.0157 pp | [-0.0412, +0.0106] pp | +0.0600 pp (1) |
| 32 | 2 | +0.2615 pp | [+0.2247, +0.3023] pp | +0.2407 pp (2) |
| 40 | 2 | +0.4716 pp | [+0.4286, +0.5207] pp | +0.6005 pp (2) |

All centered occupancies have zero censoring and every J error curve has zero
solver-error increases. In the Base-common frame, centered excess is -0.0827,
+0.1673, and +0.3526 percentage points at layers 24/32/40, with occupancy 2
throughout and 90% intervals [-0.1276, -0.0375], [+0.1305, +0.2028], and
[+0.3109, +0.3956] percentage points. The result SHA-256 is
`712ccd18...ebd2c`; the layer checkpoints are `31af395a...b35de`,
`1bbd2ec0...b5009`, and `eaa08327...a42a7`. Instruct is analyzed only as a
sibling endpoint.

### Paired capacity verdict

`ol-capacity-joint-dev-v1` completed from clean source commit `cbc7ab3` using
2,000 shared, domain-stratified prompt-bootstrap draws. The preregistered
headline is the Base-to-3.0 Think own-lens equal-layer mean. Its centered-excess
difference is +0.0154 percentage points with paired 90% interval [-0.0121,
+0.0434] points; occupancy difference and its interval are exactly zero. It is
`stable` under the frozen ±0.25-point equivalence margin. The layer-specific
own-frame differences are:

| Layer | Base to 3.0 Think difference | Paired 90% interval | Occupancy difference | Class |
|---:|---:|---:|---:|---:|
| 24 | +0.0787 pp | [+0.0306, +0.1263] pp | 0 | stable |
| 32 | +0.0240 pp | [-0.0234, +0.0729] pp | 0 | stable |
| 40 | -0.0565 pp | [-0.1077, -0.0075] pp | 0 | stable |

There is no supported positive material individual layer. The Base-common
equal-layer sensitivity is also `stable`: -0.1412 percentage points, paired
90% interval [-0.1512, -0.1316], occupancy difference zero. All twelve
equal-layer model-pair/frame rows are stable. Across the 48 classified rows,
46 are stable and two are unresolved: Base-common layer-40 Base-to-3.1 Think
(-0.2316 points, interval [-0.2523, -0.2114]) and Base-to-Instruct (-0.2371
points, interval [-0.2580, -0.2169]) narrowly cross the -0.25-point
equivalence edge. These do not change the frozen primary router, but they are
retained as coordinate-sensitivity qualifications.

The registered lineage verdict is
`broadly_conserved_capacity_recruitment_consistent`. This supports a bounded
statement that large sparse-capacity growth is not needed to explain the
released Think endpoint; it does not prove literal equality, causal
recruitment, or an effect of the Think objective. The own-frame Base-to-3.0
selected-support mean Jaccard rises from 0.291 at layer 24 to 0.412 at layer 32
and 0.468 at layer 40, illustrating why O3 geometry and the selection-margin
audit remain necessary even when the capacity estimand is stable.

The joint JSON is 51,661 bytes with SHA-256 `cb35a22a...d1492`; the registered
Parquet table is 15,047 bytes with SHA-256 `b3ad02e9...e247e`; and the paired
bootstrap NPZ is 1,047,850 bytes with SHA-256 `c13cebed...10fa`.

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

The prospective geometry implementation is now prepared. Before any new O3
operator outcome, `ol-geometry-protocol-v1` will freeze a deterministic 1,024
row stable common-vocabulary sample, complete Bank F/S/W task-token strata,
and the exhaustive union of every token selected by the registered O2
own-centered prefix at its per-position crossing. The operator series covers
all 21 fitted layers and all six unordered checkpoint pairs. It reports raw,
identity-separated, and trace-projection-separated operator comparisons;
random transport probes; a randomized leading singular spectrum with an
explicit stable-rank estimand; mapped-row CKA/cosines/neighborhood overlap;
unembedding and final-norm movement; and the complete selected-ID/RBO/span
geometry at layers 24/32/40. Figures are a separate evidence boundary and are
rendered only from registered tables.

The selection-margin audit has an explicit data boundary. The O2 sufficient
statistics recover the full chosen prefixes, crossing occupancy, and the
J-marginal-gain minus random-median threshold margin, but not the candidate
correlation scores for the kth and k+1 atoms. The latter therefore remains
`not-estimable` until a compatible replay; it is not silently replaced by the
threshold margin. Protected-span overlap is also unavailable from O2, and
core-versus-fringe causal dose remains blocked by the failed O1 service gate.
This preserves compatibility with the concurrent Phase 4.3 machinery without
reading its in-flight outcomes or coupling the namespaces.

The protocol was frozen as `ol-geometry-protocol-v1` from source commit
`04a27a4`, before any new operator comparison. It identified 100,255 common
nonspecial IDs; fixed 1,024 stable sampled rows, 1,123 task-token IDs, and the
complete 11,517-ID O2 selected-prefix union; and produced a 13,319-row model
tensor extraction list. Four selected IDs are special/noncommon and remain in
the extraction for complete span accounting while being flagged separately.
The protocol JSON is SHA-256 `91410471...9697d` and the row manifest is
`86222e65...ed92c`. This methods event contains no geometry outcome.

The first staged input, `ol-geometry-readout-olmo31-instruct-v1`, extracts all
13,319 frozen unembedding rows and the 5,120-dimensional final norm from exact
revision `ac0587e4...`. Every tensor-contract check passes. The 136,513,840
byte safetensors output is SHA-256 `638d2603...40b48`; its manifest is
`58952cf9...9f687`. This is a methods-only model input, not a geometry result.

The second staged input, `ol-geometry-readout-olmo3-base-v1`, extracts the same
13,319 frozen rows and final norm from exact Base revision `c2b61dae...`. All
six tensor-contract and finiteness checks pass. Its 136,513,840-byte tensor is
SHA-256 `56ef4f98...cfb1a`; its manifest is `07f98b2b...cccf`. This is also a
methods-only input and does not compare checkpoints.

The third staged input, `ol-geometry-readout-olmo3-think-v1`, extracts the
frozen rows and final norm from exact OLMo-3 Think revision `ebd033e4...`; all
checks pass. Its tensor is SHA-256 `ff580ed4...5b418` and its manifest is
`ecbd3c60...9e661`. Like the other extracts, it contains no comparison result.

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
| 2026-08-02 | `ol-capacity-olmo3-base-dev-v1` | Base completed at layers 24/32/40; centered excess -0.0137/+0.3151/+0.5896 pp, occupancy 2/2/2; no censoring or solver increases; no intervention opened | Source commit `f67efcd`; result `3708447c...`; three independently reconstructable layer checkpoints |
| 2026-08-02 | `ol-capacity-olmo3-think-dev-v1` | 3.0 Think completed at layers 24/32/40; own-frame centered excess +0.0650/+0.3391/+0.5331 pp, occupancy 2/2/2; Base-common sensitivity also recorded; no censoring, solver increases, or intervention opened | Source commit `04870ec`; result `2a872e4a...`; layer checkpoints `e444dc6b...`, `60d4989c...`, `9d65e188...` |
| 2026-08-02 | `ol-capacity-olmo31-think-dev-v1` | 3.1 Think completed at layers 24/32/40; own-frame centered excess +0.0625/+0.3275/+0.5192 pp, occupancy 2/2/2; Base-common sensitivity also recorded; no censoring, solver increases, or intervention opened | Source commit `7baaf64`; result `f6af4791...`; layer checkpoints `1ac1e445...`, `ceb2c9d2...`, `9d69d40b...` |
| 2026-08-02 | `ol-capacity-olmo31-instruct-dev-v1` | Instruct sibling completed at layers 24/32/40; own-frame centered excess -0.0157/+0.2615/+0.4716 pp, occupancy 2/2/2; Base-common sensitivity also recorded; no censoring, solver increases, or intervention opened | Source commit `9f213f6`; result `712ccd18...`; layer checkpoints `31af395a...`, `1bbd2ec0...`, `eaa08327...` |
| 2026-08-02 | `ol-capacity-joint-dev-v1` | Paired Base-to-3.0 Think own-frame equal-layer difference +0.0154 pp, 90% CI [-0.0121, +0.0434], occupancy difference 0; all primary layers stable, no positive material layer; verdict broadly conserved capacity/recruitment-consistent; Instruct retained as sibling | Source commit `cbc7ab3`; JSON `cb35a22a...`; Parquet `b3ad02e9...`; paired bootstrap `c13cebed...` |
| 2026-08-02 | O3 geometry pre-evidence implementation | Outcome-blind geometry protocol, exact-revision two-shard readout extraction, all-layer operator/token aggregate, exhaustive O2 selection postprocessing, explicit unavailable-margin fields, and registered-table-only five-figure producer; no new O3 outcome opened | Git commit `04a27a4`, pushed; 39 tests pass |
| 2026-08-02 | `ol-geometry-protocol-v1` | Froze 1,024 stable rows, 1,123 task IDs, the complete 11,517 selected-ID union, and 13,319 total extracted rows; exact score-gap and blocked causal fields remain explicit nulls; no geometry outcome opened | Source commit `04a27a4`; protocol `91410471...`; row manifest `86222e65...` |
| 2026-08-02 | `ol-geometry-readout-olmo31-instruct-v1` | Extracted 13,319 frozen unembedding rows plus final norm from exact Instruct revision; all tensor-contract and finiteness checks pass; no comparison outcome opened | Source commit `8c86909`; tensor `638d2603...`; manifest `58952cf9...` |
| 2026-08-02 | `ol-geometry-readout-olmo3-base-v1` | Extracted the same 13,319 frozen unembedding rows plus final norm from exact Base revision; all tensor-contract and finiteness checks pass; no comparison outcome opened | Source commit `64f377f`; tensor `56ef4f98...`; manifest `07f98b2b...` |
| 2026-08-02 | `ol-geometry-readout-olmo3-think-v1` | Extracted the frozen rows plus final norm from exact OLMo-3 Think revision; all tensor-contract and finiteness checks pass; no comparison outcome opened | Source commit `49a1dfd`; tensor `ff580ed4...`; manifest `ecbd3c60...` |

## Current limitations and claim boundary

The OLMo results establish independent baseline candidate-set capability and a
development-tier sparse-capacity estimand on this partition; the prospective
Bank-W joint-support service gate nevertheless fails. Stable capacity within a
frozen equivalence margin does not establish literal equality, causal
utilization, receivers, substitution, temporal organization, or coordinate
identity. The failed family-support target blocks the planned intervention
service set rather than licensing a post hoc family or model subset. The track
will not claim that Think training creates a global workspace, that the
reasoning objective is the causal variable, that Instruct lacks a verbalizable
channel, or that the registered recruitment-consistent verdict proves a
mechanism before O3 geometry/transport evidence and an authorized causal test.

## Recovery

Use `reports/INPROGRESS_OLMO_LINEAGE.md` for the exact current action and
`reports/OLMO_LINEAGE_RESUME.md` after a VM reclaim. Both are committed to Git
and mirrored into the Drive run root after every material checkpoint.
