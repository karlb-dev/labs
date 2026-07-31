# LIVE — Phase 4 OLMo lineage handoff

Last updated: 2026-07-31 07:48 UTC.

## Restart contract

- Repository: `/content/labs`
- Branch: `interp_jspace_part2`
- Pushed scientific-evidence head before this documentation checkpoint:
  `c619ad1`. Resume from the remote
  branch tip containing this file.
- Static bootstrap:
  `/content/drive/MyDrive/interpret/special_lab_resume.md`
- Bootstrap mirror:
  `interpretability/jspace_part2/reviews/special_lab_resume_mirror.md`
- The bootstrap files are byte-identical, SHA-256
  `bd4cf83ba59d58d6f65cda7dfa03798fa0cab03745811a4851f07bbf1598e51e`.
- Governing plans:
  `interpretability/jspace_phase4/reviews/jspace_lab_nextsteps_4_1.md`
  and `jspace_lab_nextsteps_4_1_addendum.md`.
- Phase 3 release tag: `jspace-phase3-complete-v1` at `9e0672b`.
  Phase 3 is closed and immutable.
- Phase 4 run root:
  `/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731`
- Phase 4 evidence registry:
  `interpretability/jspace_phase4/reports/evidence_events.jsonl`
- The Phase 4 preregistration is still a **candidate, not frozen**.
  Do not run Phase 4 confirmatory or replication cells before PI
  sign-off and a tagged freeze. The lineage work below is explicitly
  known-bank development evidence.

## Mandatory GPU-only model-compute contract

This Colab VM exposes an NVIDIA RTX PRO 6000 Blackwell Server Edition
(97,887 MiB, driver 580.82.07). PyTorch is `2.11.0+cu128`, CUDA build
12.8, compute capability 12.0.

Before **every** model job, call the same-process CUDA hard gate before
model load and perform an actual CUDA FP16 matrix multiply. Phase 4
producers use `jspace_phase4.gpu.require_cuda_gpu()`. Also assert that
model parameters and intervention tensors are on CUDA.

**Never silently use CPU for model loading, inference, generation,
lens fitting, intervention grids, or answer scoring.** If a restricted
sandbox reports no CUDA, that means the process context is wrong:
relaunch the exact command with host/unsandboxed GPU access. Do not
interpret sandbox CUDA invisibility as a reason to fall back to CPU.
CPU is allowed only for hashes, unit tests, deterministic statistics,
plotting, and TeX/PDF compilation.

The completed Think grids recorded the hard gate in their result
envelopes. Live telemetry during the final common-lens grid showed
81,119 MiB VRAM allocated on the RTX PRO 6000, P0.

## Current process and checkpoint state

- Nothing is running.
- All completed evidence and figures are durable on Drive, registered,
  committed, and pushed.
- Latest scientific-evidence head is `c619ad1`; the documentation
  checkpoint containing this file may be newer.
- Full Phase 4 suite: 39/39 passing. The sandbox-only tiny nonlinear-JVP
  unit test can emit a CUDA initialization warning; it does not perform
  model evidence and is intentionally CPU-safe.
- Model runs checkpoint every 5 or 10 items, limiting loss well below
  30 minutes.

## Phase 4 foundation — complete

- Package: `interpretability/jspace_phase4/`
- Phase 3 release imported immutably as
  `p4-import-phase3-release-v1`.
- Foundation evidence: `p4-foundation-scaffold-v1`.
- Contracts implemented: pinned logical URIs and local-NVMe
  materialization, CUDA hard gate, prospective prefix-disjoint scoring,
  input manifests, payload hashes, refusal to resume incompatible
  state, event-sourced evidence, stable SHA-derived seeds, family
  bootstrap and exact sign-flip statistics, phase hooks, intervention
  controls, and a no-confirmatory-before-freeze gate.

## Workstream A0 — OLMo-3 32B Think G5 complete

Live evidence:
`p4-g5-bank-olmo3-think-dev-v2`.

Artifacts:

`phase4_20260731/metrics/olmo3-think/g5_bank/p4-g5-bank-olmo3-think-dev-v2/`

- 972 immutable item rows, 324 facts, 72 families.
- Overall generation capability: `0.643004`.
- Bank F: `0.493464`; Bank S: `0.897222`.
- Direct: `0.657407`; composed: `0.484568`;
  bridge-supplied: `0.787037`.
- The prospectively fixed lineage cohort requires both direct and
  composed generation capability: 41/204 Bank-F facts and 85/120 Bank-S
  facts, 126 facts / 252 items total.
- Generation trichotomy: 625 original, 345 other-invalid,
  2 counterfactual.

The partial v1 run stopped after 180/972 when Unicode accent folding
exposed an ambiguity between exact `Río` and normalized `Rio`.
Canonical selection now prefers exact spelling before a unique
normalized fallback; v2 is the live evidence.

## Workstream A1 — Think own-lens grid and analysis complete

Live raw evidence:
`p4-lineage-grid-olmo3-think-dev-v1`.

Live analysis:
`p4-lineage-analysis-olmo3-think-dev-v1`.

Artifacts:

- `phase4_20260731/metrics/olmo3-think/lineage_grid/p4-lineage-grid-olmo3-think-dev-v1/`
- `phase4_20260731/metrics/olmo3-think/lineage_analysis/p4-lineage-analysis-olmo3-think-dev-v1/`
- `phase4_20260731/figures/p4f01_olmo3_think_development.{png,pdf}`

The raw grid contains all 252 prospective direct/composed items,
126 facts, and 36 families. It scores seven paired conditions over
every prefix-disjoint accepted alias:

1. baseline;
2. span-safe J;
3. exact rank/energy matched control;
4. label-protected J;
5. protected-energy matched control;
6. mechanics random control;
7. logit-space label-protected control.

Conformance:

- Baseline drift from G5: at most `7.15e-7` nats.
- Matched rank agreement: 100%.
- Matched energy relative error: at most `0.000230`.
- Span-safe projector overlap max: 0; lost-rank mean: 0.
- Clean selected rank: median 1, p90 1, max 12.

Known-bank development estimates, family weighted:

- Bank S direct J-specific effect:
  `-0.127717`, 95% bootstrap `[-0.208352,-0.048332]`.
- Bank S composed:
  `-0.055293`, approximately `[-0.098,-0.019]`.
- Bank S composed-minus-direct:
  `+0.072424`, `[-0.010811,+0.159008]`; uncertain. The 3.1 Think
  positive-composition anomaly may begin by 3.0, but do not interpret
  it before the controlled Bank-W adjudication.
- Bank F direct:
  `+0.057389`, interval crosses zero.
- Bank F composed:
  `+0.081821`, interval crosses zero.

## Workstream A4 — Think common-base-lens cross-check complete

Live final evidence:
`p4-lineage-grid-olmo3-think-common-base-lens-dev-v3`.

Artifacts:

`phase4_20260731/metrics/olmo3-think-common-base-lens/lineage_grid/p4-lineage-grid-olmo3-think-common-base-lens-dev-v3/`

The common coordinate is the frozen OLMo base lens, SHA-256
`92f32e38dc4dffc45dda4e0c34a75f5433238f2046ae00046a4fe3fe1226b696`.

Two diagnostic attempts are preserved and withdrawn:

- v1 exposed rank-limited protected-energy positions that silently
  omitted one requested component without marking the site clamped.
- v2 fixed the clamp metadata but revealed that changing the evidence
  ID also changed every matched/random scientific seed.

The final v3 explicitly freezes
`scientific_seed_namespace` to the v1 namespace. Audit results:

- all 252 items / 126 facts / 36 families present and paired;
- every aggregate outcome in all seven conditions is numerically
  identical to v1, maximum absolute delta exactly `0.0`;
- every per-alias outcome JSON and every condition order is identical
  to v1;
- 296 protected-energy alias summaries are correctly reclassified;
- erroneous v1 maximum unclamped energy relative error:
  `0.511283`;
- corrected v3 maximum:
  `0.000239`;
- matched rank agreement: 100%;
- CUDA evidence records RTX PRO 6000 and a finite
  `fp16-matmul-1024`.

V3 explicitly supersedes both withdrawn diagnostic attempts.

## Paired own/common-lens analysis — complete

Live evidence:
`p4-lens-frame-analysis-olmo3-think-dev-v1`.

Artifacts:

- `phase4_20260731/metrics/olmo3-think-lens-frame/lens_frame_analysis/p4-lens-frame-analysis-olmo3-think-dev-v1/`
- `phase4_20260731/figures/p4f02_olmo3_think_lens_frame_comparison.{png,pdf}`

The analysis pairs the same 252 items exactly; baseline drift is 0.
The figure was visually inspected and its PDF is one valid page.

Development findings:

- Bank F direct common-minus-own specificity:
  `-0.097666`, 95% family bootstrap
  `[-0.206010,-0.003382]`.
- Bank F composed common-minus-own:
  `-0.147417`, `[-0.231978,-0.055278]`.
- Thus Bank F changes from small positive own-lens point estimates to
  small negative common-lens point estimates. The Bank-F organization
  is coordinate-frame sensitive.
- Bank S direct remains negative under the common base lens:
  `-0.086084`, `[-0.163256,-0.003738]`.
- Bank S composed remains negative under both frames; its paired frame
  difference is small and interval-crossing.
- The uncertain Bank-S composition estimate is `+0.072424` under the
  own lens and `+0.039695` under the common lens. Their difference is
  `-0.032729`, interval crossing zero.
- Item-level own/common specificity correlation: `0.6140`;
  family-level correlation: `0.4946`.

Interpretation is estimation-first: Bank S is more frame-robust than
Bank F at this checkpoint, while Bank F supplies evidence that fitted
coordinate drift matters. This is not a binary lineage claim.

## OLMo-3 32B base point — complete

Live evidence:

- `p4-g5-bank-olmo3-base-dev-v1`
- `p4-lineage-grid-olmo3-base-dev-v1`
- `p4-lineage-analysis-olmo3-base-dev-v1`

Artifacts:

- `phase4_20260731/metrics/olmo3-base/g5_bank/p4-g5-bank-olmo3-base-dev-v1/`
- `phase4_20260731/metrics/olmo3-base/lineage_grid/p4-lineage-grid-olmo3-base-dev-v1/`
- `phase4_20260731/metrics/olmo3-base/lineage_analysis/p4-lineage-analysis-olmo3-base-dev-v1/`
- `phase4_20260731/figures/p4f03_olmo3_base_development.{png,pdf}`

The exact base model is
`allenai/Olmo-3-1125-32B@c2b61dae89a1ad10e4ad5653d0e46b590902607b`.
Its G5 gate contains 972 rows / 324 facts / 72 families:

- overall generation capability `0.610082`;
- Bank F `0.439542`; Bank S `0.900000`;
- direct `0.648148`; composed `0.398148`; bridge supplied `0.783951`;
- prospectively capable direct+composed cohort: 23 Bank-F facts /
  11 families and 88 Bank-S facts / 21 families, 111 facts / 222 items;
- generation outcomes: 593 original, 379 other-invalid, 0
  counterfactual.

The completed seven-condition grid ran on the RTX PRO 6000 and passed
the strict acceptance audit:

- all 222 items are unique and form 111 exact direct/composed pairs;
- maximum baseline replay drift from G5: `6.71e-7` nats;
- every per-alias aggregate recomputes to floating-point precision;
- span-safe selected/protected projector overlap: exactly 0;
- matched rank fraction: 1.0 over all 504 alias-condition summaries;
- maximum matched energy relative error: `0.000260`;
- result SHA-256:
  `90a022e1d5bc7fa99aa8e56c4eb0e79b25c14fccaefad44ef1eda21c32cbac7e`;
- parquet SHA-256:
  `c22fa1ea8280daca4eafb641b7c635f126d29eec3adb86398fc3b340e9cf3c69`.

The deterministic 100,000-draw analysis was independently reproduced
byte-for-byte. Family-weighted J-specific development estimates:

- Bank F direct: `+0.024531`, `[-0.032962,+0.079192]`;
- Bank F composed: `+0.014550`, `[-0.122560,+0.188055]`;
- Bank S direct: `+0.000498`, `[-0.048738,+0.048082]`;
- Bank S composed: `+0.002070`, `[-0.030860,+0.038506]`;
- Bank S composed-minus-direct: `+0.001572`,
  `[-0.041733,+0.040300]`.

The primary base effect is therefore near zero in all four cells. This
is not an inert-grid explanation: overall label-protected J-specific
was `+0.123051`, mechanics-random `-0.104821`, and logit-protected
`-0.202304`, each with an interval excluding zero. The contrast with
the negative Bank-S 3.0 Think point localizes a development change to
the base-to-Think interval, but checkpoint-specific capability cohorts
make this a trajectory localization rather than a paired causal
estimate. The new PNG and one-page PDF were visually inspected.

## Local immutable inputs and disk

- The exact local Think snapshot was removed only after every one of its
  14 shards (64,467,127,296 bytes total) independently matched its
  Hugging Face LFS SHA-256 object ID, every Drive blob size matched, and
  all small files were byte-exact. It remains recoverable from:
  `/content/drive/MyDrive/hf_cache/hub/models--allenai--Olmo-3-32B-Think/snapshots/ebd033e4f0b284d5973b82c0ccb62ad0dbe877d7`
  (14 shards; about 61 GiB dereferenced).
- Exact base snapshot is now local:
  `/content/hf_local/models--allenai--Olmo-3-1125-32B/snapshots/c2b61dae89a1ad10e4ad5653d0e46b590902607b`.
  The index contains 707 tensors over 14 shards / 64,467,127,296 bytes.
  Every shard independently matches its Hugging Face LFS SHA-256 object
  ID and no incomplete files remain.
- Own Think lens is materialized locally under
  `/content/sl4_work/inputs/05b9290a34bb50bc5c68e65dfb05d6b84222fb0dd736fa2f6748c261140ef053/`.
- Base lens is materialized locally under
  `/content/sl4_work/inputs/92f32e38dc4dffc45dda4e0c34a75f5433238f2046ae00046a4fe3fe1226b696/`.
- Current root free space with the base model local: about 58 GiB.
- The base G5, grid, analysis, plots, and registry events are durable
  and pushed. Do not delete the local base snapshot until this updated
  report/handoff checkpoint is also committed, pushed, copied
  byte-exact to Drive, and the exact next model source has been
  verified.

## Next queue — execute without pausing

1. Verify branch/head/clean tree, exact 3.1 Think/Instruct model
   revisions, lens hashes, and Drive/Hub cache sources. Do not remove
   the validated local base snapshot before this verification.
2. Add and test Phase 4 prospective G5 plus own/common-lens grid configs
   for the exact 3.1 Think and Instruct checkpoints. Commit and push
   configs before creating evidence.
3. Before every model producer, rerun
   `jspace_phase4.gpu.require_cuda_gpu()` in the same host process.
   Model load, generation, intervention, and scoring must use the RTX;
   never use CPU fallback.
4. Run G5, freeze each checkpoint-specific direct+composed capable
   cohort, and run the same seven-condition grids under both its own
   lens and the frozen base lens. Bank each major boundary before
   swapping a cache.
5. Build the full base → 3.0 Think → 3.1 Think/Instruct development
   trajectory with common-lens and own-lens views. If they disagree,
   treat coordinate drift as a result and prioritize the fit/corpus-size
   study before causal over-interpretation.
6. Refresh this file, the Drive copy, Phase 4 Markdown/TeX/PDF, figures,
   and evidence registry after every major boundary. Commit and push
   often.

Never overwrite registered evidence. Use a new evidence ID and an
event-sourced withdrawal, correction, or supersession for every repair.
