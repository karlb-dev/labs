# LIVE — Phase 3 release audit and Phase 4 bridge handoff

Last updated: 2026-07-31 03:18 UTC.

## Restart contract

- Repository: `/content/labs`
- Branch: `interp_jspace_part2`
- Pushed result-bearing head: `54370b6` (use `git log -1` for the
  current head after later handoff/report commits).
- Governing plan: Drive
  `special-lab-1/jspace_lab_nextsteps_4_1.md` plus its addendum.
- Static bootstrap: `/content/drive/MyDrive/interpret/special_lab_resume.md`
- Current Phase 3 run root:
  `/content/drive/MyDrive/interpret/special-lab-1/phase3_20260729`
- Frozen tags:
  `jspace-phase3-freeze-v1` → `df4d45a`;
  `jspace-phase3-pre-release-audit-v1` → `660047d`.
- Do not create the Phase 3 completion tag until the release manifest,
  paper/handout PDF, final state report, and final verification are
  banked.

## Mandatory GPU contract

This VM class has an NVIDIA RTX PRO 6000 Blackwell Server Edition
(97,887 MiB; driver 580.82.07). PyTorch is `2.11.0+cu128`, CUDA capability
is 12.0. A restricted sandbox may hide the device.

Before every model job, run the CUDA hard gate in
`special_lab_resume.md` in the **same process context** that will load
the model. Every Phase 3 producer must call
`jspace_phase3.gpu.require_cuda_gpu()` before model load and assert that
model parameters are on CUDA. If a sandbox cannot see CUDA, relaunch the
exact command with host/unsandboxed GPU access. **Never fall back to CPU
for model loading, inference, lens fitting, or scoring.** CPU is only for
unit tests, hashes, statistics, plots, and TeX.

Qwen3.6-27B exact weights remain local in `/content/hf_local`; the exact
Qwen lens is local and has SHA-256
`1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1`.
The pinned Jacobian Lens checkout is `/tmp/jacobian-lens` at `581d3986`.
TeX Live is installed.

## Current state

Nothing is running. All completed scientific evidence is on Drive,
registered, committed, and pushed. The Phase 3 test suite is 97/97
passing.

### Release-audit reproduction

- `p3-inference-audit-v1`: repaired exact/proper inference. P3-P1 is
  not convention-robust enough for inferential wording.
- `p3-protocol-audit-protected-answer-qwen-v1`: P3-P2 survives exact
  and any-accepted-alias protected strata.
- `p3-control-seed-contract-audit-v2`: all 1,890 Qwen control cells
  banked. P3-P2 is identical across five seeds; P3-P1 crosses the .05
  decision threshold and must be described as seed-sensitive.
- N8-P3-L1 expected-value-blind analysis: 61/61 quantities exact.
- N8-P3-L2: 20 items / 10 families on Qwen, OLMo Think, and OLMo
  Instruct, with deterministic arms exact.
- N8-P3-L3: full Qwen 188/188 reproduction, deterministic arms exact and
  stable control exact. Repaired locked estimates:
  P3-P1 `-0.2711827`, exact p `0.0578918`;
  P3-P2 `+0.0958333`, p `1/100001`;
  P3-P3 `+0.4313669`, p `0.0091799`.

### Bridge geometry — completed

Live evidence:
`p3-bridge-geometry-qwen36-27b-v2`.
It explicitly supersedes v1, whose secondary clean-span overlap used raw
coherent rows instead of an orthonormal projector basis.

Artifacts:

`phase3_20260729/metrics/qwen36-27b/release_audit/bridge_geometry_v2/`

- 94 frozen composed facts, 26 families, 67,834 site rows.
- Baseline, span-safe, true-bridge, and distractor-bridge replay errors
  are all exactly 0.
- Rank accounting failures: 0; final selected/protected overlap max: 0;
  protected bridge survival min: 1.
- Raw equal-family rescue: `+0.43137` nats, 95% family bootstrap
  `[+0.13202,+0.76344]`, MC family sign-flip p `0.01222`.
- Nested leave-one-family-out geometry prediction has cross-fit
  R² `-0.0947`; geometry-adjusted residual is `+0.40382`
  `[+0.10507,+0.73450]`, p `0.01854`.
- Strict piece-count plus every-site added-rank matches: 10 facts /
  5 families, estimate `+0.45028`, but exact p `0.1875`; this subset is
  underpowered.
- Interpretation: measured geometry does not explain away P3-P3, but
  this remains true-versus-chosen-distractor confirmatory evidence, not
  untouched-family replication.

Two failed runs created only a `state.json` before the first fact and
were preserved (not registered) under:

- `bridge_swap_endpoint_failed_missing_cache_mask_20260731T0203Z`
- `bridge_swap_endpoint_failed_cache_non_equivalence_20260731T0205Z`

They document that Qwen3.6 cached continuation LP differed from no-cache
teacher forcing by 0.15092 nats. The live endpoint therefore uses
no-cache full candidate scoring with hooks active only on the shared
prompt prefix.

### Semantic bridge swap — completed

Live evidence:
`p3-bridge-swap-endpoint-qwen36-27b-v1`.

Artifacts:

`phase3_20260729/metrics/qwen36-27b/release_audit/bridge_swap_endpoint/`

- 40 existing mediation facts / 13 families; frozen baseline replay
  error exactly 0.
- Counterfactual-bridge preference shift:
  `+8.58203` nats, 95% family bootstrap `[+5.07766,+12.12299]`,
  exact family sign-flip p `0.0004883`.
- The movement is calibrated in both directions:
  original LP changes `-3.32312`; intended counterfactual LP changes
  `+5.25891`.
- Against the geometry-selected unrelated injection:
  `+4.76514` `[+2.69063,+6.91296]`, p `0.0002441`.
  Unrelated matches have exact piece count for 37/40 facts, exact
  injection norm and removed-energy scale for all facts, and always
  come from a different fact/family.
- Against true re-injection: `+8.23575`, p `0.0002441`.
- Against orthogonal random injection: `+7.30380`, p `0.0004883`.
- Greedy counterfactual hits: 15/40 under counterfactual bridge versus
  0/40 at baseline. Baseline original hits: 32/40; counterfactual-swap
  original hits: 7/40.
- Important limitation: direct counterfactual-answer direction injection
  shifts preference `+7.240` nats. Counterfactual bridge minus
  counterfactual answer direction is only `+1.342`
  `[-1.593,+4.482]`, p `0.419`. The endpoint proves intended semantic
  movement, but does not yet isolate an abstract bridge channel from a
  downstream answer-direction mechanism.
- This is post-freeze development evidence on the existing cohort and
  not untouched-family replication.

### Boundary, cohort, and alias sensitivity — completed

Live evidence:

- `p3-boundary-cohort-sensitivity-v2` (supersedes v1);
- `p3-alias-endpoint-{qwen36-27b,olmo31-think,olmo31-instruct}-v1`;
- `p3-alias-endpoint-cross-model-v1`.

Boundary/cohort artifacts:

`phase3_20260729/metrics/cross_model/release_audit/alias_cohort_sensitivity_v2/`

- Boundary-safe regrading rejects four substring false positives and
  recovers four Unicode-accent matches across 2,916 G5 rows.
- The deterministic 100-positive/100-negative hand audit has zero
  disagreements.
- At the repaired state seed 31337, confirmatory P3-P1 is `-0.271183`
  (exact p `0.057892`) under the historical cohort and `-0.270160` under
  the boundary-safe cohort.
- Boundary-safe P3-P2 is `+0.095833` on 186 items (plus-one p
  `2/100001`); P3-P3 is `+0.428230` on 93 items (p `0.009420`).

Alias artifacts:

`phase3_20260729/metrics/{qwen36-27b,olmo31-think,olmo31-instruct}/release_audit/alias_endpoint/`
and
`phase3_20260729/metrics/cross_model/release_audit/alias_endpoint/`

- The subset was frozen before outcomes: 20 common confirmatory facts,
  all 17 shared families, and all four multi-alias facts; 52 alias cells
  per model.
- All three model jobs ran on the RTX PRO 6000 with CUDA-only scoring.
  First-alias baseline/J replay errors were below `0.000214` nats and
  tokenizer prefix-overlap pairs were zero.
- Stable first-alias P3-P1 on the enriched subset is `-0.267301`.
  Prefix-disjoint logsumexp is `-0.246098`, a `+0.021203` change.
  Diagnostic max alias changes it by only `-0.000056`.
- Canonical-only scoring is more surface-sensitive (`-0.383906`, a
  `-0.116605` change), entirely concentrated in the two facts whose
  canonical alias differs from the first alias.
- The Qwen P3-P2 subset tail estimate is identical under historical
  first, stable first, canonical, prefix-disjoint logsumexp, and max
  scoring.

The exact OLMo Think and Instruct caches downloaded for this audit were
removed only after their artifacts and registry events were pushed.
They are recoverable from pinned Hub revisions. Qwen remains local.

## Binding paper language

- P3-P1: descriptive, seed-sensitive negative estimate; no inferential
  near-miss wording.
- P3-P2: robust confirmatory Qwen-specific protected-tail result.
- P3-P3: confirmatory true-versus-chosen-distractor rescue on the frozen
  partition; measured geometry audit supports it, but there is no
  untouched-family replication.
- The counterfactual swap truly increases intended counterfactual
  probability and produces counterfactual generations, but the broader
  bridge-channel mechanism is development evidence and remains
  confounded with an answer-direction route.
- Boundary-safe cohort selection and prospective prefix-disjoint
  logsumexp do not materially alter the repaired headline estimates.
  Canonical-only scoring has a disclosed two-fact surface sensitivity.

## Next queue — execute in order

1. Refresh report figures/tables and paper wording with the N8,
   geometry, and semantic results. Generate Markdown, TeX, and PDF and
   verify PDF page rendering.
2. Build the final Phase 3 release manifest and state report, verify
   every live registry hash, run the full repro suite, commit/push, and
   only then create the completion tag.
3. Scaffold Phase 4 lineage and Bank B replication. The highest-value
   new test is a frozen untouched-family bridge bank that distinguishes
   counterfactual bridge injection from matched counterfactual-answer
   direction injection.

Commit and push at every boundary and refresh this file before any long
run. Preserve all old evidence through supersession events; never
overwrite it.
