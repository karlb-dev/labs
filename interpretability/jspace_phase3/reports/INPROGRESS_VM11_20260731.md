# LIVE — Phase 4 OLMo lineage handoff

Last updated: 2026-07-31 08:56 UTC.

## Restart contract

- Repository: `/content/labs`
- Branch: `interp_jspace_part2`
- Pushed scientific-evidence head before this documentation checkpoint:
  `bf8ca15`. Resume from the remote
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
- Latest scientific-evidence head is `bf8ca15`; the documentation
  checkpoint containing this file may be newer.
- Full Phase 4 suite: 43/43 passing. The sandbox-only tiny nonlinear-JVP
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

Live standalone audit evidence:
`p4-lineage-grid-olmo3-think-common-base-lens-dev-v3`.

Live seed-paired evidence:
`p4-lineage-grid-olmo3-think-common-base-lens-dev-v4`.

Artifacts:

- `phase4_20260731/metrics/olmo3-think-common-base-lens/lineage_grid/p4-lineage-grid-olmo3-think-common-base-lens-dev-v3/`
- `phase4_20260731/metrics/olmo3-think-common-base-lens/lineage_grid/p4-lineage-grid-olmo3-think-common-base-lens-dev-v4/`

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

V3 explicitly supersedes both withdrawn diagnostic attempts and remains
valid standalone evidence. V4 is the paired repair:

- all 252 items / 126 facts / 36 families present and paired;
- shared namespace exactly
  `p4-lineage-grid-olmo3-think-dev-v1`;
- condition order exactly matches the own-lens grid for every item;
- between-frame baseline drift: exactly `0.0`;
- baseline replay drift from G5: `7.15e-7` nats;
- matched rank agreement: 100% over 592 summaries;
- maximum matched energy relative error: `0.000205`;
- span-safe overlap: exactly 0;
- deterministic baseline and J-arm outcomes match v3 exactly;
- result SHA-256:
  `a3f21082d1c99cc5574355b5d0f68e893420d75f1b606b684575e02476fd4496`;
- parquet SHA-256:
  `e7b54e050469358bed6ad0af6f39218fc206012ae7d0c9f146e54af7fee51dd0`.

## Paired own/common-lens analysis — corrected and complete

Live evidence:
`p4-lens-frame-analysis-olmo3-think-dev-v2`.

Withdrawn evidence:
`p4-lens-frame-analysis-olmo3-think-dev-v1`.

V1 remains withdrawn because its frames used different scientific RNG
namespaces. V2 uses own grid v1 plus common grid v4 under the exact same
namespace and condition order. The producer now hard-fails on unequal
namespaces. The full deterministic 100,000-draw payload independently
reproduces exactly, including all 42 bootstrap distribution hashes.

Artifacts:

- `phase4_20260731/metrics/olmo3-think-lens-frame/lens_frame_analysis/p4-lens-frame-analysis-olmo3-think-dev-v2/`
- `phase4_20260731/figures/p4f02_olmo3_think_lens_frame_comparison_v2.{png,pdf}`

Family-weighted paired results:

- Bank F direct: own `+0.057389`, common `+0.001041`,
  common-minus-own `-0.056348`, 95% interval
  `[-0.132436,+0.004930]`;
- Bank F composed: own `+0.081821`, common `-0.016425`,
  delta `-0.098245`, `[-0.200519,+0.005787]`;
- Bank S direct: own `-0.127717`, common `-0.097432`,
  delta `+0.030285`, `[-0.016452,+0.081286]`;
- Bank S composed: own `-0.055293`, common `-0.041877`,
  delta `+0.013416`, `[-0.009951,+0.040760]`.

No paired frame-delta interval excludes zero. Both Bank-S effects have
intervals below zero in both frames; Bank-F effects are imprecise.
Item/family correlations are `0.755689` / `0.725410`, and mean absolute
item frame difference is `0.102471` nats. The one-page PDF and PNG were
visually inspected. A report-local rerender reserves footer space so
panel d's x-axis label does not collide with the development note; the
registered v2 artifacts remain immutable.

The repair changed only stochastic control terms, as it should: all J
frame deltas are identical to v1, while Bank-F direct/composed
specificity deltas move `+0.041318` / `+0.049172` nats toward zero.
Never use the old v1 paired intervals or figure.

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

- The exact 3.0 Think snapshot is local again at
  `/content/hf_local/models--allenai--Olmo-3-32B-Think/snapshots/ebd033e4f0b284d5973b82c0ccb62ad0dbe877d7`.
  It was restored from the independently verified Drive snapshot after
  excluding unreferenced `.incomplete` blobs. Every one of its 14
  shards (64,467,127,296 bytes total) matches its Hugging Face LFS
  SHA-256 object ID; no local incomplete file remains.
- The full recoverable Drive source is:
  `/content/drive/MyDrive/hf_cache/hub/models--allenai--Olmo-3-32B-Think/snapshots/ebd033e4f0b284d5973b82c0ccb62ad0dbe877d7`
  (14 shards; about 61 GiB dereferenced).
- The exact local base snapshot was removed only after its complete G5,
  grid, analysis, figures, report checkpoint, registry events, and Git
  commits were durable. It is recoverable by exact revision
  `c2b61dae89a1ad10e4ad5653d0e46b590902607b`.
- Own Think lens is materialized locally under
  `/content/sl4_work/inputs/05b9290a34bb50bc5c68e65dfb05d6b84222fb0dd736fa2f6748c261140ef053/`.
- Base lens is materialized locally under
  `/content/sl4_work/inputs/92f32e38dc4dffc45dda4e0c34a75f5433238f2046ae00046a4fe3fe1226b696/`.
- Current root free space with 3.0 Think local: about 61 GiB.
- Do not remove the local 3.0 Think snapshot until this corrected v4/v2
  report/handoff checkpoint is committed, pushed, and copied byte-exact
  to Drive. After that, it is safe to remove only that exact local cache
  root before materializing 3.1, because its verified Drive copy and all
  evidence remain durable.

## Next queue — execute without pausing

1. Confirm this completed documentation checkpoint is present at the
   remote branch tip and byte-exact on Drive. At creation, the hashes
   were report MD `0b4aadd8...1798a41`, TeX
   `1b643c7a...e031122`, four-page PDF
   `a0cae1b6...87be5e6`, report display plot
   `3de36890...c790568`. The scientific-evidence boundary is pushed at
   `bf8ca15`.
2. The exact 3.1 trajectory config set is already committed at
   `238ba9f`. Verify these pinned sources before download:
   - Think:
     `allenai/Olmo-3.1-32B-Think@832c3f543499af8fe68b88359501de9cb7840544`;
   - Instruct:
     `allenai/Olmo-3.1-32B-Instruct@ac0587e4a7744a551c059d8cd17ba220bc940dae`;
   - own-lens SHAs:
     `1fe5355f4cb964f2508cfa9c05f6183f704922e4b752bfef626cd58d9965d8b8`
     (Think) and
     `e0f8b972a9f1f884101f94ff52a1938d5cfa7a5f49e987e6768826f2337c6dfb`
     (Instruct);
   - common base lens:
     `92f32e38dc4dffc45dda4e0c34a75f5433238f2046ae00046a4fe3fe1226b696`.
3. Drive cache discovery found only `config.json` for 3.1 Think and no
   3.1 Instruct snapshot. After the documentation push, remove only the
   verified local 3.0 Think cache, then download the exact 3.1 revision
   to local NVMe. Maintain restart-safe Drive/cache provenance and
   verify every downloaded shard before model use.
4. Before every model producer, rerun
   `jspace_phase4.gpu.require_cuda_gpu()` in the same host process.
   Model load, generation, intervention, and scoring must use the RTX;
   never use CPU fallback.
5. Run 3.1 Think G5, freeze its checkpoint-specific direct+composed
   capable cohort, and run the same seven-condition grids under both
   its own lens and the frozen base lens. Both grids already share the
   explicit namespace
   `p4-lineage-grid-olmo31-think-frame-pair-dev-v1`. Bank and push G5,
   each grid, own analysis, and paired analysis before swapping cache.
6. Repeat for 3.1 Instruct under
   `p4-lineage-grid-olmo31-instruct-frame-pair-dev-v1`.
7. Build the full base → 3.0 Think → 3.1 Think/Instruct development
   trajectory with common-lens and own-lens views. If they disagree,
   treat coordinate drift as a result and prioritize the fit/corpus-size
   study before causal over-interpretation.
8. Refresh this file, the Drive copy, Phase 4 Markdown/TeX/PDF, figures,
   and evidence registry after every major boundary. Commit and push
   often.

Never overwrite registered evidence. Use a new evidence ID and an
event-sourced withdrawal, correction, or supersession for every repair.
