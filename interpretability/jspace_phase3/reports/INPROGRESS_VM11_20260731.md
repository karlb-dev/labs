# LIVE — Phase 4 OLMo lineage and Qwen lens-fit handoff

Last updated: 2026-07-31 22:37 UTC.

## Restart contract

- Repository: `/content/labs`
- Branch: `interp_jspace_part2`
- Pushed scientific-evidence/registry head before this documentation
  checkpoint: `0788407`. The Qwen report boundary is `d51ba22` and
  the preceding live-fit handoff is `f902d40`. The GPU-fit harness is
  pushed at `8769b8b`
  and its first OOM recovery repair at `0dbf4de`. The mandatory fused
  GPU-kernel runtime repair is pushed at `51336db`. Resume from the
  remote branch tip containing this file.
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

The completed Think and Instruct grids recorded the hard gate in their
result envelopes. Live Instruct G5 telemetry showed 62,164 MiB VRAM
allocated at 94% utilization; the Instruct grids reached 81,128 MiB on
the RTX PRO 6000.

## Current process and checkpoint state

- **An active GPU model job is running:** cumulative Qwen draw A from
  n=120 toward n=250, started from clean pushed commit `3a6d501`.
  It rehashed the exact model, passed the CUDA/FLA/48-block gates, and
  recovered the exact fit contract at n=120 before doing new work.
  Its latest complete atomic local/Drive boundary is n=174,
  checkpoint SHA-256
  `db56aa91ed403c4556bc5ed8b27c54947d4b556bc3160616190af0c4c95fe66b`,
  6,606,047,399 bytes. Prompts 175–177 had resumed when this handoff
  was written. If the original process is gone, run the exact n=250
  command below and require `recovered_next_idx` of at least 174.
  The VM was already at 23h59m uptime at this boundary, so
  assume reclamation is imminent and trust only the atomic Drive
  checkpoint, not the live process.
- Qwen draw A n=120 completed and is live evidence
  `p4-qwen-lens-fit-drawA-n120-dev-v1`, committed and pushed at
  `515a67a`. It used fit contract SHA-256
  `bf4caff4ff7c389d29f235a91062ae86e3a37dfc526c42bbd9af7c5d7e1f3b00`.
  The final complete atomic local/Drive recovery boundary has
  checkpoint SHA-256
  `061574f95546d859f13141af480d2aa20372a8858dbc2f9bcdaacdbdd1cdb673`,
  6,606,047,399 bytes. The registered fp16 lens is 3,303,034,078
  bytes, SHA-256
  `82af4cc7f637af33e166606b15993bd6c67d2ea764c9788b96aa5a2120c32b1b`.
  All 40 three-prompt checkpoints completed. Full invocation time was
  21,622.7 seconds (6h00m23s including final lens/registry work),
  180.19 seconds per prompt; peak allocated VRAM was stable at
  62,832,854,016 bytes. Both payload envelopes reconstructed exactly,
  all output hashes independently matched, and the full live-evidence
  verifier passed 24 events / 92 outputs / zero failures. The RTX was
  fully released afterward (0 MiB, 0% utilization).
- The exact 3.1 Instruct cache is independently verified on local NVMe
  and Drive. Its G5, seed-paired own/common grids, own-frame analysis,
  paired-frame analysis, and registered figures are complete,
  independently reproduced, committed, and pushed. The registered
  four-checkpoint trajectory synthesis is also complete and pushed.
  The leakage-safe Qwen nested corpora are registered, and the
  GPU-only resumable Qwen fitter is tested and pushed. Two feasibility
  attempts completed no prompts and produced no checkpoints or
  evidence: `dim_batch=8` OOMed, while `dim_batch=4` under the
  Transformers Torch delta-rule fallback was deliberately interrupted
  after more than 8.5 minutes on prompt 1. FLA 0.5.2 is now
  host-validated and the active run uses only its fused CUDA
  delta-rule kernels with a three-prompt Drive boundary.
- The local 3.1 Think cache was removed only after its complete
  scientific and documentation boundary was backed up and pushed at
  `7a9dd07`. Its independently verified Drive recovery copy remains.
- All completed evidence and registered figures are durable on Drive,
  registered, committed, and pushed.
- Latest scientific-evidence/registry head is `0788407`; the fitter
  implementation head is `8769b8b`; the Qwen n=120 report/figure
  boundary is pushed at `d51ba22`.
- The refreshed handout compiles to 10 pages with no TeX warnings; its
  Qwen structural and closing page was visually inspected. PDF
  SHA-256 is
  `52891fa3dbe4eaa82d55db486c165821ad036b6cd37a590f6a5dc4d0e5963823`.
- Full Phase 4 suite after the structural producer: 63/63 passing.
  The sandbox-only tiny nonlinear-JVP
  unit test can emit a CUDA initialization warning; it does not perform
  model evidence and is intentionally CPU-safe.
- The Qwen fitter now atomically mirrors its 6.6 GB cumulative
  checkpoint every 3 prompts. Reassess using live fused-kernel timings,
  but never relax the boundary beyond about 30 minutes of GPU work.

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

## OLMo-3.1 32B Think G5 — complete

Live evidence:
`p4-g5-bank-olmo31-think-dev-v1`.

Artifacts:

`phase4_20260731/metrics/olmo31-think/g5_bank/p4-g5-bank-olmo31-think-dev-v1/`

The exact pinned 3.1 Think model ran all 972 generation and
prospective-alias scoring items on the RTX PRO 6000. The same-process
hard gate records a finite FP16 CUDA matmul, and live telemetry during
the run showed 62,164 MiB VRAM in use at 94% GPU utilization.

- overall boundary-safe prefix capability: `0.632716`;
- Bank F: `0.483660`; Bank S: `0.886111`;
- direct: `0.641975`; composed: `0.475309`;
  bridge-supplied: `0.780864`;
- prospective direct+composed-capable cohort: 38 Bank-F facts and
  84 Bank-S facts, 122 facts / 244 items total;
- generation trichotomy: 615 original, 355 other-invalid,
  2 counterfactual;
- result SHA-256:
  `00534b4e6746683429f48d60dd98505e6bce7671ac65cb96c00f1d0b30a85b34`;
- parquet SHA-256:
  `5496091823520287c3f40394c3a7a39920a3695d9107af2ced8353f8df101834`.

This is lower than Phase 3's published `0.734568` because the old
cohort used an accepted alias anywhere in the eight-token
continuation. It is not a generation drift or GPU problem: every one
of the 972 continuations is byte-identical to Phase 3, and all 972
Phase 4 capability flags exactly equal the later Phase 3
`capable_prefix_boundary_safe` audit. The full row, hash, alias,
aggregate, counterfactual, payload, registry, and CUDA audit passed,
as did all 43 Phase 4 tests. The live event is committed and pushed at
`efafa4f`.

## OLMo-3.1 32B Think paired trajectory point — complete

Live raw evidence:

- `p4-lineage-grid-olmo31-think-dev-v1`;
- `p4-lineage-grid-olmo31-think-common-base-lens-dev-v1`.

Live analyses:

- `p4-lineage-analysis-olmo31-think-dev-v2`;
- `p4-lens-frame-analysis-olmo31-think-dev-v1`.

Artifacts:

- `phase4_20260731/metrics/olmo31-think/lineage_grid/p4-lineage-grid-olmo31-think-dev-v1/`;
- `phase4_20260731/metrics/olmo31-think/lineage_analysis/p4-lineage-analysis-olmo31-think-dev-v2/`;
- `phase4_20260731/metrics/olmo31-think-common-base-lens/lineage_grid/p4-lineage-grid-olmo31-think-common-base-lens-dev-v1/`;
- `phase4_20260731/metrics/olmo31-think-lens-frame/lens_frame_analysis/p4-lens-frame-analysis-olmo31-think-dev-v1/`;
- `phase4_20260731/figures/p4f04_olmo31_think_development_v2.{png,pdf}`;
- `phase4_20260731/figures/p4f05_olmo31_think_lens_frame_comparison.{png,pdf}`.

Both seven-condition grids ran on the RTX and share
`p4-lineage-grid-olmo31-think-frame-pair-dev-v1`. They contain the
same 244 item rows / 122 direct-composed fact pairs / 35 families and
the same randomized condition order for every item. Acceptance:

- own/common between-frame baseline drift: exactly `0.0`;
- maximum G5 baseline replay drift: `8.64e-7` nats;
- span-safe projector overlap: exactly 0 in both frames;
- rank match: 1.0 over all 564 matched summaries in each frame;
- maximum matched energy relative error: own `0.000222`, common
  `0.000256`;
- mechanics-random and logit-protected outcomes: bit-identical across
  frames;
- own result/parquet SHA-256:
  `49b3c166674b0e391b1b7136296e0c90f6dbdeec862e5a6fc0fa62b16322d7df`
  /
  `cc8e2d57f4c3e9f1becd28c2553d488dc279113bda75420ba8bdab90cfa76937`;
- common result/parquet SHA-256:
  `8be6cc4b80f004d052552446ca964efb6c34ea5c4120e116a2934ab9b7129b7c`
  /
  `d6f7fec74e044fc82147e899c9ab2fecc332aabc27f15df3e25cccc45ed0ddf9`.

Own-frame family-weighted J-specific effects:

- Bank F direct `-0.006706`, interval
  `[-0.116217,+0.105683]`;
- Bank F composed `-0.041602`,
  `[-0.209911,+0.116668]`;
- Bank S direct `-0.167375`,
  `[-0.247738,-0.094215]`;
- Bank S composed `-0.049099`,
  `[-0.089149,-0.016471]`;
- Bank-S composed-minus-direct `+0.118276`,
  `[+0.051391,+0.187895]`, descriptive exact sign-flip
  `p=0.003067`.

The original own-frame analysis v1 has an identical numerical payload
but is withdrawn because visual inspection found the figure footer
crowding lower-panel labels. The reusable lineage-analysis figure now
reserves footer space, has a regression test, and v2 is visually clean.
The suite is now 44/44. V2 payload SHA-256 is
`26e6a0f4937d16e66f11a84508a8340f1e1214c7472b507c5e8a2b8b4c38dec7`.

Paired frame results:

- F direct common-minus-own `-0.069013`,
  `[-0.141299,+0.002506]`;
- F composed common-minus-own `-0.089825`,
  `[-0.167700,-0.009588]`;
- S direct own/common `-0.167375` / `-0.154702`, with both intervals
  below zero; frame delta `+0.012672`,
  `[-0.037828,+0.066879]`;
- S composed own/common `-0.049099` / `-0.038096`; the own interval is
  below zero and the common interval narrowly crosses zero; frame
  delta `+0.011003`, `[-0.019627,+0.040625]`;
- Bank-S composition is positive in both frames: own `+0.118276`
  `[+0.051747,+0.187573]`, common `+0.116606`
  `[+0.027864,+0.207978]`; paired delta `-0.001670`
  `[-0.055101,+0.050437]`;
- item/family frame correlation `0.767900` / `0.838635`, mean absolute
  item shift `0.113316` nats.

The F-composed frame-delta interval is the only specificity delta
excluding zero, so coordinate sensitivity is now observed in one
known-bank 3.1 cell. All 42 bootstrap and 6 exact sign-flip
distributions independently reproduce exactly. Paired result SHA-256
is
`151070d943e62bf9829e037bb8461ab1b94543e26c7d3ae73f544376734693f7`.
The complete scientific boundary is pushed at `a119986`.

## OLMo-3.1 32B Instruct G5 — complete

Live evidence:
`p4-g5-bank-olmo31-instruct-dev-v1`.

Artifact:

`phase4_20260731/metrics/olmo31-instruct/g5_bank/p4-g5-bank-olmo31-instruct-dev-v1/`

The GPU producer completed 972 rows / 324 facts / 72 families:

- overall boundary-prefix capability `0.624486`;
- Bank F `0.467320`; Bank S `0.891667`;
- direct `0.666667`; composed `0.438272`; bridge-supplied
  `0.768519`;
- accepted direct-and-composed cohort: 118 facts, comprising 32 F and
  86 S facts across 15 F and 21 S family groups.

Independent validation reconstructed every alias and counterfactual
token manifest and aggregate, checked the state and result envelope
hashes, registry output hashes, immutable row metadata, and GPU
provenance. All 972 generations are byte-identical to the Phase 3 run,
and all 972 capability flags exactly match the later boundary-safe
prefix audit. The full suite remains 44/44.

- result SHA-256:
  `97b7d26848094288d0f7cb1a247d22348071d2a1df25fe02729028aca2805ca4`;
- parquet SHA-256:
  `b15c6542714974f5c1824bc27a1f0b84d0a48054d67757b3fa0f316512a0b6e9`;
- input-manifest SHA-256:
  `85cefce82824a5658049e18260fa39616314cb0c81011eed293396fd944f705e`;
- result payload SHA-256:
  `14bb6b41481d6115d02c7a9dfba60bc3765aad5f7c38e6c6dac40fb2d1c2a0cc`.

The validated registry boundary is pushed at `ee51eea`.

### Instruct own-lens grid — complete

Live evidence:
`p4-lineage-grid-olmo31-instruct-dev-v1`.

Artifact:

`phase4_20260731/metrics/olmo31-instruct/lineage_grid/p4-lineage-grid-olmo31-instruct-dev-v1/`

The RTX producer completed the immutable 236-item / 118-fact /
36-family cohort under seed namespace
`p4-lineage-grid-olmo31-instruct-frame-pair-dev-v1`. Independent
validation found:

- maximum G5 baseline replay drift `7.75e-7` nats;
- span-safe projector, normalized, and removed-protected-energy
  overlap exactly `0.0`;
- rank match `1.0` in all 528 matched summaries, with minimum protected
  rank 10;
- maximum matched-energy relative error `0.000228`;
- maximum protected cosine `2.7e-7`;
- result/parquet SHA-256:
  `f79ae41162fac3363b42122934867f86b5ef1c7cb80b60ced6ca43d061fd758a`
  /
  `dada28f3a39262958f79add83244e4794f98903af611384092f57282609f2bdb`.

The validated registry boundary is pushed at `07d256a`.

### Instruct common-base lens and both analyses — complete

Live evidence:

- `p4-lineage-grid-olmo31-instruct-common-base-lens-dev-v1`;
- `p4-lineage-analysis-olmo31-instruct-dev-v1`;
- `p4-lens-frame-analysis-olmo31-instruct-dev-v1`.

Artifacts:

- `phase4_20260731/metrics/olmo31-instruct-common-base-lens/lineage_grid/p4-lineage-grid-olmo31-instruct-common-base-lens-dev-v1/`;
- `phase4_20260731/metrics/olmo31-instruct/lineage_analysis/p4-lineage-analysis-olmo31-instruct-dev-v1/`;
- `phase4_20260731/metrics/olmo31-instruct-lens-frame/lens_frame_analysis/p4-lens-frame-analysis-olmo31-instruct-dev-v1/`;
- `phase4_20260731/figures/p4f06_olmo31_instruct_development.{png,pdf}`;
- `phase4_20260731/figures/p4f07_olmo31_instruct_lens_frame_comparison.{png,pdf}`.

The common-base-lens grid ran on the RTX under the same
`p4-lineage-grid-olmo31-instruct-frame-pair-dev-v1` namespace as the
own grid. All 236 item IDs, 118 direct/composed fact pairs, 36
families, and condition orders match. Acceptance:

- between-frame baseline drift exactly `0.0`;
- maximum G5 replay drift `7.75e-7` nats in each frame;
- span-safe overlap exactly 0 and rank agreement 1.0 over all 528
  matched summaries;
- maximum matched-energy relative error: own `0.000228`, common
  `0.000212`;
- mechanics-random and logit-protected outcomes bit-identical across
  frames;
- common result/parquet SHA-256:
  `c9b128c70f014ff8b4f0d28097744024fda5e729f6181a6ad1947a3655ab82d2`
  /
  `7f5cd296183a90259ac9959bde51d853d4934288019593e1b0f57c873aea4418`.

The common-grid registry boundary is pushed at `2f5e14e`.

Own-frame family-weighted J-specific effects:

- Bank F direct `+0.043159`, `[-0.025880,+0.118059]`;
- Bank F composed `-0.066216`, `[-0.403699,+0.192109]`;
- Bank S direct `-0.022419`, `[-0.065586,+0.017928]`;
- Bank S composed `-0.017672`, `[-0.075527,+0.032474]`;
- Bank-S composed-minus-direct `+0.004747`,
  `[-0.041100,+0.050097]`.

All four primary intervals and the Bank-S composition interval include
zero, unlike the sibling 3.1 Think result. All 22 bootstrap and 6
exact sign-flip distributions independently reproduce exactly. The
analysis result/input-manifest hashes are
`b7ab7f497fb62d06c434ce6512f423af88d8a04649a521e070aa200260990787`
/
`6071c6a5054f0f57409d266919b7024a5f9b30a7f77cbc6648a64447a3e18a29`.
The registered PNG/PDF hashes are
`608cc5d4625ba69eab5e29aeae144b64ae02543d189fcb2dddc74bc199c75db9`
/
`95a8640064d06549e1561a8c3605bffd86aec98052096d2cb7501afedd7bac17`.
This boundary is pushed at `ede9b99`.

Seed-paired own/common results:

- F direct: own `+0.043159`, common `-0.011069`,
  common-minus-own `-0.054228`, 95% interval
  `[-0.096963,-0.019400]`, descriptive exact sign-flip `p=0.005737`;
- F composed: own `-0.066216`, common `+0.027146`, delta `+0.093362`,
  `[-0.031211,+0.248762]`;
- S direct: own `-0.022419`, common `-0.037890`, delta `-0.015470`,
  `[-0.060266,+0.025809]`;
- S composed: own `-0.017672`, common `-0.004546`, delta `+0.013126`,
  `[-0.023965,+0.055828]`;
- Bank-F composition: own `-0.109374`, common `+0.038216`, delta
  `+0.147590`, `[+0.025748,+0.297768]`, `p=0.036255`;
- Bank-S composition: own `+0.004747`, common `+0.033344`, delta
  `+0.028597`, `[-0.027970,+0.096384]`;
- item/family frame correlation `0.752319` / `0.852085`; mean absolute
  item shift `0.081852` nats.

All Bank-S effects, composition contrasts, and frame deltas include
zero. Bank-F direct and composition show coordinate sensitivity, but
the individual F effects remain imprecise. All 42 bootstrap and 6
exact sign-flip distributions independently reproduce exactly. The
paired result/input-manifest hashes are
`cb647454a12604d660e088a3603fe4e87bf754e83df3bdfeb494191b0d1fbf73`
/
`518ca9254bd88d369eac1c30e3cda756cbef71107aa3bd8107f4c2fbad60aeb0`.
The registered PNG/PDF hashes are
`a131d0b2fcbc530f459409a635d503e898158d3394164901c93c59e2f27716c3`
/
`2fc689553c6635e7c238f5acc673e55bd833118c20d9bea374fee2a1b402c87a`.
The full Instruct scientific boundary is pushed at `8842482`.

## Workstream A5 — registered four-checkpoint trajectory complete

Live evidence:
`p4-lineage-trajectory-analysis-olmo-dev-v1`.

Artifacts:

- `phase4_20260731/metrics/olmo-lineage-trajectory/trajectory_analysis/p4-lineage-trajectory-analysis-olmo-dev-v1/`;
- `phase4_20260731/figures/p4f08_olmo_lineage_trajectory.{png,pdf}`.

The registered CPU statistics/plotting producer reads only the completed
GPU grid evidence. It validates all eight own/common inputs, live
registry status, frame-paired item manifests, seed namespaces and
condition orders, plus the shared bank and scoring contract. It then
recomputes six metrics in each frame at the base, 3.0 Think, 3.1 Think,
and sibling 3.1 Instruct endpoints under one stable family-bootstrap
schedule.

- 48 frame/metric summaries independently reconstructed exactly;
- 42 unique bootstrap distributions independently reconstructed
  exactly;
- distribution-hash-set SHA-256:
  `e2b0e8a9b96146f26e9a3dea2fce710affdedb8c7a4ddd42e2b81c75165ad326`;
- result/input-manifest/table SHA-256:
  `d5599b71f4f3c211e870ccdd7631de12e3f3311e2b7b0740856a615313038640`
  /
  `11b65c5936473a52f4747a7c5a5061eb44dae74023957a173538edf87c997cfa`
  /
  `3a0e7a9654a39a42583aa74800a2e09e012eef7649408a98a418d04d40b6420f`;
- registered PNG/PDF SHA-256:
  `be48f37164818ace88ea23923aa8389f8a9aaf510c38a3e1a8454e49213ded1e`
  /
  `ddc39a5b1853ad3ee6d3839417bdc11706c3139a5e09678cf492aa11256d8e95`;
- live-evidence verifier: 22 live events, 85 output files, zero
  failures;
- full test suite: 47/47.

The plot connects only Base → 3.0 Think → 3.1 Think. The 3.1 Instruct
point is an unconnected sibling endpoint, avoiding a false temporal
edge. Core Bank-S trajectory:

- direct specificity, own/common:
  Base `+0.0005/+0.0005`;
  3.0 Think `-0.1277/-0.0974`;
  3.1 Think `-0.1674/-0.1547`;
  sibling Instruct `-0.0224/-0.0379`;
- composed-minus-direct, own/common:
  Base `+0.0016/+0.0016`;
  3.0 Think `+0.0724/+0.0556`;
  3.1 Think `+0.1183/+0.1166`;
  sibling Instruct `+0.0047/+0.0333`.

Thus the negative Bank-S direct effect appears by 3.0 and strengthens
at 3.1 Think in both frames; the positive composition contrast becomes
precise at 3.1 Think in both frames; both collapse toward zero at
sibling Instruct. Capability cohorts are checkpoint-specific, so these
are unpaired development localizations rather than causal training or
mode contrasts. The scientific boundary is pushed at `4751235`.

## Workstream E0 — Qwen nested-corpus and GPU fitter foundation complete

Live methods evidence:
`p4-qwen-lens-corpora-dev-v1`.

Corpus artifacts:

`phase4_20260731/config/qwen_lens_corpora/p4-qwen-lens-corpora-dev-v1/`

- Draw A has 1,000 unique WikiText-103 train records and exact nested
  milestones `n=120,250,500,1000`; JSONL SHA-256
  `3582ec41de0bd95b7e7f1b71b40b89604bff45001e9056c58221d9ee47dfa455`.
- Its first 120 rows are byte-exact to the historical campaign fit
  prefix. Historical draw-A rows 120–199 were evaluation spares, so
  all 80 are explicitly excluded from both new fit corpora.
- Independent draw B has 500 unique records and milestones
  `n=120,500`; JSONL SHA-256
  `1565adaf63db3d93225c203a2575d6fe5a7947f5b3a86d4db04a38984484f675`.
  Its first 120 rows are byte-exact to the historical independent
  draw, SHA-256
  `ff5092e3fff27f3a69d24688fb7fff63e896f50035b9ca4b975c9e23954d2d65`.
- Independent reconstruction found 1,000/500 unique indices, zero A/B
  overlap, and zero overlap with the 80 evaluation spares.
- The exact pinned WikiText parquet order matches all 320 historical
  records. It has 1,801,350 rows and 410,472 records meeting the frozen
  600-character threshold.
- Live-evidence verification after registration: 23 events, 89 output
  files, zero hash failures.

The GPU fitter is
`jspace_phase4.experiments.p4_qwen_nested_lens_fit`, config
`configs/p4_qwen_nested_lens_fit_dev.yaml`; the base fitter is pushed
at `8769b8b` and the required fused-runtime repair at `51336db`.
It fits all source layers 0–62 to target layer 63 with the upstream
paper estimator (128 tokens, skip first 16), maintains one cumulative
float32 checkpoint, and atomically mirrors it to Drive every 3
prompts. The initial `dim_batch=8` feasibility attempt passed the
same-process RTX hard gate and loaded the exact model, but its first
forward reached 94.96/94.97 GiB and OOMed before completing prompt 1.
No checkpoint or evidence was created and no CPU fallback occurred.
The first `dim_batch=4` attempt used about 75.95 GiB and sustained
60–84% GPU utilization, but Transformers had selected its very slow
pure-Torch delta-rule implementation. It was intentionally interrupted
after more than 8.5 minutes before prompt 1 completed; again no
checkpoint or evidence was created and no CPU fallback occurred.
The two immutable diagnostic progress records are:

- `progress_11314e540b726014fb5f48664b1e579b584d91d9d5bc56c4876831e24a5ba45e_dim8_oom.json`;
- `progress_cc6e06d3bf84ac5b885e2a02fe19b680303e3830a549de9ae03f01eac337adfb_torch_slow_aborted.json`.

The active retry retains `dim_batch=4`, which upstream `jlens`
documents as a memory-only batching knob: the estimator and total
backward FLOPs are unchanged. Exact runtime packages are now pinned:
PyTorch `2.11.0+cu128`, Transformers `5.13.1`, Triton `3.6.0`,
`fla-core==0.5.2`, and `flash-linear-attention==0.5.2`. A fresh
host-process validation on the RTX completed a finite fp16 CUDA
matrix multiply and reported compute capability 12.0. Transformers
reports FLA available; Qwen binds `chunk_gated_delta_rule` and the
recurrent rule to `fla.ops.gated_delta_rule.*`, and fused gated RMS
normalization to `fla.modules.*`. External `causal-conv1d` is absent
and not required: only its small convolution remains in PyTorch, while
the previously dominant delta-rule is fused. The fitter now refuses
wrong package versions, absent FLA, a Torch delta-rule fallback, or
anything other than 48 fused Qwen linear-attention blocks after model
load. A same-process CUDA smoke matmul and CUDA model-location
assertion still precede fitting; there is no CPU fallback. Recovery
also refuses changes to the corpus, model, recipe, runtime/kernel
contract, fitter source, or exact clean `jlens` revision
`581d398613e5602a5af361e1c34d3a92ea82ba8e`.

The exact Qwen model is local at revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. All 15 shards were fully
read after manifest creation; all content hashes match, totaling
55,563,006,400 weight bytes. The published n=1000 lens remains local,
SHA-256
`1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1`.

The first nested lens milestone is complete and live:
`p4-qwen-lens-fit-drawA-n120-dev-v1`. Its 63 source layers target
layer 63 at d_model 5120. The runtime result records all 48 Qwen
linear-attention blocks bound to FLA, the exact five pinned runtime
package versions, the RTX CUDA hard gate, and the exact clean jlens
source contract. Prompt 112 was a pronounced but finite per-record
Jacobian-norm outlier (`159.952`); it was retained under the frozen
corpus/estimator and must be examined in the n=120 versus n=1000
stability analysis, never trimmed post hoc.

The first structural stability analysis is also complete and live:
`p4-qwen-lens-structural-stability-drawA-n120-vs-published-n1000-dev-v1`,
pushed at `0788407`. This is explicitly a recipe/corpus-transfer
comparison against the published n=1000 lens, not the pending
same-corpus draw-A convergence comparison. It used every source layer,
all 5,120 Jacobian rows, 4,096 fixed sampled token directions, centered
linear CKA on a fixed 1,024-token subset, and 256 fixed transport
probes. Both result envelopes reconstructed exactly; the table has 63
rows / 29 columns with all numeric values finite; the figure was
visually inspected; the full verifier passed 25 live events / 97
outputs / zero failures.

Agreement is strongly depth-dependent:

- across-layer median matrix cosine `0.896944`, sampled-token median
  cosine `0.913843`, and CKA `0.865211`;
- L0 matrix/token/CKA `0.541130/0.529659/0.451778`;
- L62 `0.999729/0.999678/0.999744`;
- token median first remains at least 0.90 from L26, matrix cosine from
  L32, CKA from L33, and token q05 from L35;
- at campaign capacity layers L24/L32/L40, token median cosine is
  `0.894696/0.922001/0.967747`, CKA
  `0.866738/0.840932/0.943321`, and relative Frobenius delta
  `0.471148/0.426898/0.280033`.

Therefore n=120 does **not** reproduce the published n=1000 geometry
uniformly at the lower/middle assay layers. The plan's convergence
decision triggers the same-corpus n=250 fit before any global
fit-size closure. Do not mistake excellent late-layer agreement for a
whole-lens validation. Figure:
`phase4_20260731/figures/p4f09_qwen_lens_structural_stability.{png,pdf}`.

## Local immutable inputs and disk

- The exact local 3.0 Think cache was removed after the corrected v4/v2
  evidence, report, PDF, handoff, and Git commits were durable. Its full
  recoverable Drive source remains:
  `/content/drive/MyDrive/hf_cache/hub/models--allenai--Olmo-3-32B-Think/snapshots/ebd033e4f0b284d5973b82c0ccb62ad0dbe877d7`
  (14 shards; about 61 GiB dereferenced).
- Exact 3.1 Think remains materialized on Drive at
  `/content/drive/MyDrive/hf_cache/hub/models--allenai--Olmo-3.1-32B-Think/snapshots/832c3f543499af8fe68b88359501de9cb7840544`.
  The local copy was removed only after the full Think evidence,
  report, PDF, handoff, Drive backup, and Git boundary were durable at
  `7a9dd07`. Before removal, the local and Drive snapshots each had 26
  entries, 14 weight shards, and a 707-tensor index; every shard
  independently matched its content-addressed SHA-256 target.
  Dereferenced weight bytes were exactly `64,467,127,296` in both
  copies, with no `.incomplete` or rsync-partial file.
- Exact 3.1 Instruct is now materialized on local NVMe and Drive:
  `/content/hf_local/models--allenai--Olmo-3.1-32B-Instruct/snapshots/ac0587e4a7744a551c059d8cd17ba220bc940dae`
  and
  `/content/drive/MyDrive/hf_cache/hub/models--allenai--Olmo-3.1-32B-Instruct/snapshots/ac0587e4a7744a551c059d8cd17ba220bc940dae`.
  Both snapshots contain 26 entries, 14 weight shards, and a 707-tensor
  index. All 14 local shards and all 14 Drive-mounted shards were read
  and independently matched against their content-addressed SHA-256
  targets. Dereferenced weight bytes are exactly `64,467,127,296` in
  both copies; neither copy has an `.incomplete` or rsync-partial file.
- The exact local base snapshot was removed only after its complete G5,
  grid, analysis, figures, report checkpoint, registry events, and Git
  commits were durable. It is recoverable by exact revision
  `c2b61dae89a1ad10e4ad5653d0e46b590902607b`.
- The completed 3.0 Think own lens remains materialized locally under
  `/content/sl4_work/inputs/05b9290a34bb50bc5c68e65dfb05d6b84222fb0dd736fa2f6748c261140ef053/`.
- The 3.1 Think own lens is pinned at Drive URI
  `drive://part2/lens/olmo31think_lens.pt`, SHA-256
  `1fe5355f4cb964f2508cfa9c05f6183f704922e4b752bfef626cd58d9965d8b8`.
  It is now verified and locally materialized at
  `/content/sl4_work/inputs/1fe5355f4cb964f2508cfa9c05f6183f704922e4b752bfef626cd58d9965d8b8/`,
  byte-identical to the Drive source.
- The 3.1 Instruct own lens is pinned at Drive URI
  `drive://part2/lens/olmo31instruct_lens.pt`, SHA-256
  `e0f8b972a9f1f884101f94ff52a1938d5cfa7a5f49e987e6768826f2337c6dfb`.
  It is independently verified and locally materialized at
  `/content/sl4_work/inputs/e0f8b972a9f1f884101f94ff52a1938d5cfa7a5f49e987e6768826f2337c6dfb/`,
  byte-identical to the Drive source.
- Base lens is materialized locally under
  `/content/sl4_work/inputs/92f32e38dc4dffc45dda4e0c34a75f5433238f2046ae00046a4fe3fe1226b696/`.
- Exact Qwen model:
  `/content/hf_local/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
  (15 shards, 55,563,006,400 weight bytes; every shard hash verified).
- Published Qwen lens:
  `/content/hf_local/models--neuronpedia--jacobian-lens/snapshots/a4114d7752d11eb546e6cf372213d7e75526d3a1/qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt`
  (SHA-256
  `1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1`).
- Exact clean editable `jlens` checkout:
  `/tmp/jacobian-lens` at
  `581d398613e5602a5af361e1c34d3a92ea82ba8e`.
- Current root free space with Qwen and 3.1 Instruct local: about
  57 GiB. The cumulative all-layer Qwen checkpoint is expected to use
  about 6.6 GB and each fp16 milestone lens about 3.3 GB.
- Keep local 3.1 Instruct until its G5, own/common grids, analyses,
  registry events, report/handoff, and Git checkpoints are all durable.

## Next queue — execute without pausing

1. OLMo Workstream A5 is complete. The base, corrected 3.0 Think,
   seed-paired 3.1 Think/Instruct points, and registered two-frame
   trajectory are banked at scientific head `4751235`. Do not rerun or
   overwrite any registered evidence.
2. Before any later model producer, rerun
   `jspace_phase4.gpu.require_cuda_gpu()` in the same host process.
   Model load, generation, intervention, and scoring must use the RTX;
   never use CPU fallback. For the Qwen fit, additionally require the
   exact FLA runtime contract in
   `configs/p4_qwen_nested_lens_fit_dev.yaml`; a restricted sandbox is
   not a valid environment for this command.
3. Workstream E's sources, leakage-safe nested corpora, exact model
   manifest, recipe, and GPU fitting entrypoint are now pinned,
   registered, tested, committed, and pushed. Do not regenerate the
   corpora or change their ordering.
4. Draw A n=120 and its first structural comparison are complete and
   registered; do not rerun or overwrite them. The comparison found
   material early/middle-layer drift, so resume draw A to n=250 from a
   clean tree with host GPU access:

   ```bash
   python -m jspace_phase4.experiments.p4_qwen_nested_lens_fit \
     --config configs/p4_qwen_nested_lens_fit_dev.yaml \
     --draw draw_a --stop-at 250
   ```

   Run from `interpretability/jspace_phase4/` in the host GPU process.
   Recovery lives under
   `phase4_20260731/lens/qwen36-27b/nested_fit/draw_a/recovery/`.
   It must report `recovered_next_idx: 120` and verify the cumulative
   checkpoint before new GPU work. The active 2026-07-31 invocation
   had reached at least n=174; recovery must use the highest valid
   `checkpoint_state.json` boundary rather than assuming n=120.
   Commit/push each registry event,
   then repeat at n=500 and n=1000. Run draw B n=120 and preferably
   n=500 the same way after the corresponding draw-A boundary.
5. Compare row-wise token cosine, CKA, selected-ID Jaccard,
   selected-span angles, protected overlap, occupancy, centered excess
   capacity, G4, span-safe specificity, and bridge
   rescue/substitution. Do not fit a model-level capacity regression.
   The same-corpus structural comparator must reuse the exact first
   diagnostic's token-ID sample hash
   `55b81bf163ad483da72161c7f7d978930442d7fdfd032a39c3330d82d4cf8a5d`
   and packed Rademacher-probe hash
   `bed1537f9668132d3c701e0bc0303987fa085e90d9bb3992003776c81d90770c`;
   do not derive a fresh sample merely because the evidence ID changes.
6. If Qwen `n=120` reproduces `n=1000`, close the fit-size
   explanation. If not, run the nested `120/250/500` study on one
   representative OLMo checkpoint, preferably 3.1 Think, before
   comparing capacities.
7. Controlled Bank-W load/redundancy and internal-derivation work
   remains necessary before mechanism claims. Confirmatory/replication
   model cells remain blocked on the preregistration freeze.
8. The exact Instruct local cache may be removed after the
   report/TeX/PDF/handoff checkpoint containing this file is verified
   on Drive and pushed. Its independently hash-verified Drive snapshot
   is the recovery source.
9. Refresh this file, the Drive copy, Phase 4 Markdown/TeX/PDF, figures,
   and evidence registry after every major boundary. Commit and push
   often.

Never overwrite registered evidence. Use a new evidence ID and an
event-sourced withdrawal, correction, or supersession for every repair.
