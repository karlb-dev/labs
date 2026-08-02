# OLMo lineage state of record

State date: 2026-08-02T05:47:48Z

Status: OLMo parallel-phase scientific execution and the isolated run-specific
paper are complete at the first release boundary. The final import/restart
bundle is being assembled.
This state remains separate from the concurrently running main Phase 4 and
Gemma workstreams and may be integrated only in Phase 5 or later.

## 1. Authority and isolation

- Branch: `interp_jspace_olmo_lineage`.
- Scientific import boundary:
  `3b041735d8b842de46a9c0a474fccd0c44e0841a`.
- Repository namespace: `interpretability/jspace_olmo_lineage/`.
- Drive root:
  `/content/drive/MyDrive/interpret/special-lab-1/olmo_lineage_20260801`.
- Native evidence prefix: `ol-`.
- Native tiers: `development` and `methods` only.
- Instruct role: sibling endpoint, never a fourth Think trajectory point.
- Forbidden writes: Phase 3, Phase 4, Gemma, and shared main-paper registries,
  run roots, figures, or preregistrations.
- Forbidden reads: untouched Phase 4 confirmatory/replication intervention
  outcomes. The only Bank-W contact was the authorized baseline capability
  service obligation and its handoff.

The governing plan, accepted addendum, main resume, parallel namespace
contract, source registries, and imported artifacts are pinned by SHA-256 in
`ol-foundation-v1`. Imported evidence remains read-only and retains its source
tier.

## 2. Model graph and inference boundary

The temporal path is:

```text
OLMo-3 Base -> OLMo-3 32B Think -> OLMo-3.1 32B Think
```

OLMo-3.1 32B Instruct is a sibling endpoint under a distinct and incompletely
observed post-training recipe. Shared width, architecture, base lineage, task
battery, and assay code make this a matched-lineage natural experiment. They do
not randomize data, objective, formatting, stage duration, or checkpoint
selection. All changes below are associations among released checkpoints
unless an imported intervention result is explicitly named.

## 3. Registry state

At this state date the append-only registry contains 24 origin events, of which
23 are live. `ol-checkpoint-inventory-v1` remains immutable but is explicitly
superseded by version 2. The 23 live events have 88 immutable outputs, all of
which pass byte/hash verification. Fifty-four package tests and the exact dependency
lock pass.

The latest event is `ol-independent-reconstruction-v1`, created at
2026-08-02T05:26:50Z from clean source commit
`12f21ad5badeac980c11f0817906ad18c6c1d52d`.

## 4. Objective-by-objective disposition

### O0 — foundation and prospective boundary: complete

`ol-foundation-v1` verifies the isolated branch/run root, governance sources,
scientific import boundary, environment lock, direct imported artifacts, and
development preregistration. No OLMo outcome was opened before the prospective
O1/O2/O3 rules were frozen.

### O1 — Bank-W capability service: complete; intervention service blocked

The frozen protocol used 24 families × 8 seeds × two loads for the
derived/once cell: 384 rows per OLMo endpoint, with all eight answer sequences
scored by summed conditional log probability.

| Model | Rows | Low accuracy | High accuracy | High-low | Family-bootstrap 90% CI | Capable families |
|---|---:|---:|---:|---:|---:|---:|
| OLMo-3.1 Think | 384 | 0.7135 | 0.7188 | +0.0052 | [-0.0313, +0.0417] | 17/24 |
| OLMo-3.1 Instruct | 384 | 0.7396 | 0.7188 | -0.0208 | [-0.0573, +0.0208] | 17/24 |
| Qwen reference, imported | 384 | 0.8333 | 0.8333 | 0 | about [-0.0208, +0.0208] | 20/24 |

Both OLMo models independently pass. The exact three-model intersection is 16
families, below the prospective minimum of 20. Therefore
`olmo_phase4_service_ready=false`; no Bank-W intervention, confirmatory, or
replication outcome was opened. The early handoff is
`ol-phase4-early-import-bundle-v1`, with JSON SHA-256
`debb29ef67ffa8741a4971ec2b0b21340bd5b48dc5729ac12f74f78839bf4f2b`
and Markdown SHA-256
`a7e6faf9ad412bd965cfbc9f7b1e9e98c9194cc5019e669d0695b5852a55159d`.

### O2 — symmetric sparse capacity: complete

The prospective estimator uses the same ordered 120-prompt corpus and 7,481
content positions at layers 24/32/40 for Base, 3.0 Think, 3.1 Think, and the
3.1 Instruct sibling. The primary target is globally centered activation; raw
activation is a separately labeled sensitivity. Three matched random
dictionaries and 2,000 paired, domain-stratified prompt-bootstrap draws support
the joint contrasts.

The preregistered Base-to-3.0 Think own-frame equal-layer centered-excess
difference is 0.0001538, or +0.0154 percentage points, with paired 90% interval
[-0.0001211, 0.0004345]. Occupancy difference and interval are exactly zero.
All three primary layers and all twelve equal-layer pair/frame summaries are
stable under the frozen ±0.25-point margin. Forty-six of 48 classified rows are
stable; two Base-common layer-40 sensitivities are unresolved at the
equivalence edge. The registered verdict is
`broadly_conserved_capacity_recruitment_consistent`.

Key artifacts for `ol-capacity-joint-dev-v1`:

- JSON SHA-256 `cb35a22a0176f568fc7368285af65b148f342d834d3ca9457d78f3bb1dcd1492`;
- 72-row table SHA-256 `b3ad02e9d66340b33748a2b74d3e69cc9b8f6a366ba6ce4d256c641d1dae247e`;
- paired bootstrap SHA-256 `c13cebed02b985b93fa1220bd1a811bdad9a334d2f52f4579929df33cf7810fa`.

This licenses a bounded measured-capacity statement, not literal equality,
coordinate conservation, downstream causal recruitment, or training-stage
causality.

### O3 — provenance, geometry, and figure set: complete

`ol-lens-provenance-audit-v1` classifies all six lens pairs as
`EXACT_SAME_RECIPE_CORPUS`; all fitting corpus/order, 4 × 30 slice, merge, and
tokenization checks pass. The decision was to use the existing lenses rather
than manufacture a refit.

`ol-geometry-joint-dev-v1` uses all 21 source layers and all six checkpoint
pairs, plus exhaustive selected-span geometry for the 7,481 aligned positions
at layers 24/32/40. The frozen router returns
`dictionary-formation-pattern`:

- Base-to-3.0 raw-operator cosine median 0.9614, minimum 0.7343 at layer 4;
- Base-to-3.0 mapped-row cosine median 0.6744, so token continuity fails;
- Base-to-3.0 mapped movement 0.32556 versus 0.00604 for 3.0-to-3.1;
- Base-to-3.0 selected-ID Jaccard 0.3333 at layers 24/32/40, with projector
  overlap 0.2657/0.3336/0.4549;
- 3.0-to-3.1 selected-ID Jaccard 1.0 at the assay layers;
- 3.1 Think/Instruct sibling mapped-row median 0.9363 and
  `instruct_late_shift=false`.

Joint artifact hashes are JSON `c09d8e73...d0f4c`, layer table
`8692836d...9097`, selection table `c2826b14...73bd`, and readout table
`93fd2108...0bd9`. `ol-geometry-figures-dev-v1` registers five PNG/PDF pairs:
operator similarity, token-row similarity, selected-span trajectory,
capacity/known-Bank-S-causal state space, and transport/readout movement.
Figure 4 explicitly identifies Bank S and the 16/20 Bank-W block.

Exact kth/k+1 candidate-score gaps, protected-span overlap under crossed
readouts, and causal core/fringe dose remain null. The available threshold
margin and projector metrics are not substitutes.

### O4 — Bank-W mechanism grid: resolved as gated out

The four §6.6 accounts were frozen before any O4 cell: external-state
substitution, generic difficulty, capacity, and output-adjacent routing. None
was selected because O1 failed the prospective service gate. Version 1 has no
O4 intervention result. Any redesign must be a new prospective protocol and
cannot silently drop models or families from the failed cohort.

Conclusion-skeleton sentence 4 is therefore explicitly pending, not upgraded.

### Checkpoint inventory / H5: complete inventory; wedge queued

Version 1 conservatively required byte-identical tokenizer files and returned
no provenance-complete intermediate pair. A versioned semantic audit showed
that Think-SFT's tokenizer differs in serialization only: the complete
100,278-entry ID map, normalized 100,000 BPE merges, processing components, and
all 207 frozen encodings agree where required. `ol-checkpoint-inventory-v2`
supersedes version 1 and returns:

- official 32B SFT and DPO artifacts eligible;
- verdict `genuine-32b-intermediates-available`;
- H5 `testable-with-bounded-stage-wedge`;
- queue status `queued-not-started`.

The v2 JSON is SHA-256 `e5931c3f...ad9a`; Markdown is
`f3704d14...6db4`. BOS/chat rendering differences and repository-only parent
declarations remain attached qualifications. No intermediate weights or model
outcomes were opened by the inventory.

### O5 — crossed causal decomposition: resolved as not identifiable

`ol-o5-feasibility-decision-v1` pins O1, O2, O3, and inventory outputs and
returns `defer-no-identifiable-crossed-intervention-estimand` with
`not-executed-no-proxy-substitution`. The registry lacks causal cells that
independently cross activation model, transport lens, and readout, together
with per-dictionary transport, protected-span, delivered-rank/energy,
non-J-logit-lens, and common-row controls. No O5 factor estimate or null is
licensed.

The bounded Phase 5 entry is a Bank-S-first Base/3.1-Think/3.1-Instruct pilot
crossing Base/recipient-own transport and readout, expanded only after all
delivery and geometry checks pass.

### Independent reconstruction: complete

`ol-independent-reconstruction-v1` independently reconstructs:

- 768 O1 rows, score integrity, load/family summaries, deterministic family
  bootstraps, and the 16/20 joint support decision;
- 48 O2 curve summaries, 96 bootstrap intervals, 144 joint arrays, and all 72
  joint table rows;
- all 84 O3 aggregates, the full router, and unavailable-null boundary;
- five PNGs byte-for-byte and five independently regenerated PDFs;
- all fourteen exact OLMo-3.1 Think weight shards and one frozen model row,
  including all eight candidate scores, predicted alias, and -0.25 margin with
  maximum absolute drift 0.

The methods JSON is SHA-256 `e159f01d...20542`, Markdown is
`48647b44...abb5`, and payload hash is `ee94e446...619f`. Two rejected
pre-publication attempts are documented: missing `no_grad` caused OOM, and a
one-candidate diagnostic changed BF16 batch numerics. Neither created an
output, figure directory, or registry event. Restoring the original batch
under `no_grad` reproduced the registered row exactly.

## 5. Paper-facing claims

The authoritative mutable ledger for this release is
`reports/OLMO_LINEAGE_CLAIMS_TABLE.md`; the final bundle will contain a
hash-pinned copy.

Sentence 2 is narrowed to an architecture-matched association: the first
released Think transition has strong J-mapped coordinate/selection movement
without material measured sparse-capacity growth, but the design does not
identify a causal training ingredient and finite-dose transport remains
queued.

Sentence 4 is explicitly pending: Bank-S motivates external-state substitution
as a hypothesis, but Bank-W version 1 never opened after the 16/20 service-gate
failure. No externalization account is resolved.

## 6. Run-specific paper

The isolated paper is `reports/paper/olmo_lineage_parallel_phase.tex` and its
compiled PDF. It is 13 letter-sized pages, includes all five registered O3
figure PDFs, and leaves the shared Phase 4/Gemma manuscript untouched. TeX
SHA-256 is `c5b1f980...8db5`; deterministic PDF SHA-256 is
`ff6a9a65...e944`. Two builds produced the same PDF hash, the TeX log has no
warnings, and rendered title/table/figure/claim/appendix pages pass visual
inspection.

## 7. Remaining and queued work

Required to finish this release artifact layer:

1. emit and register the self-verifying final OLMo import/restart bundle;
2. stop this side track and hand its queues to Phase 5.

Queued, not authorized as part of the current result set:

- H5 two-cell official SFT/DPO wedge with existing Base/3.0/3.1 anchors;
- H6 per-checkpoint finite-dose transport validation;
- O5 Bank-S-first crossed pilot;
- one bounded receiver necessity/rescue demonstration if Phase 5 selects it;
- any redesigned Bank-W service set under a new prospective protocol.

## 8. Prohibited interpretations

- Think training creates a global workspace.
- The reasoning objective is the causal variable.
- Reasoning post-training installs external-state substitution.
- Instruct lacks a verbalizable channel or follows Think temporally.
- Measured-capacity stability proves global coordinate conservation.
- Structural geometry identifies causal transport or downstream consumption.
- The O5 not-executed decision is a null factor estimate.
- The independent reconstruction raises development evidence to confirmatory or
  replication tier.

## 9. Recovery and weight state

The stable restart guide is `reports/OLMO_LINEAGE_RESUME.md`; the live pointer
is `reports/INPROGRESS_OLMO_LINEAGE.md`. Both are mirrored to the isolated Drive
root by `python -m jspace_olmo_lineage.recovery` after every material Git
checkpoint.

The 61-GiB local OLMo-3.1 Think sentinel snapshot was deleted only after the
reconstruction event, registry/report commit `471a48f`, GitHub push, all
88-output verification, and recovery-mirror verification were durable. It is
recoverable by direct Hugging Face download at exact revision
`832c3f543499af8fe68b88359501de9cb7840544`. Registered evidence and Drive
outputs were not removed.

## 10. Release-boundary checklist

- [x] isolated package, branch, Drive root, and registry;
- [x] O1 capability gates and early import bundle;
- [x] symmetric four-checkpoint O2 capacity table with Base;
- [x] exact lens provenance audit and same-corpus O3 geometry;
- [x] five registered O3 figure pairs and selection audit;
- [x] O4 resolved without opening an unauthorized intervention;
- [x] official intermediate-stage inventory and honest H5 queue;
- [x] O5 no-substitution identifiability decision;
- [x] independent reconstruction and exact model sentinel;
- [x] sentence-level claims ledger;
- [x] state-of-record report;
- [x] isolated run-specific paper compiled;
- [ ] final import/restart bundle emitted and registered.

When the final two boxes are complete, this workstream stops and joins the
single Phase 5 router only through its hash-pinned handoff.
