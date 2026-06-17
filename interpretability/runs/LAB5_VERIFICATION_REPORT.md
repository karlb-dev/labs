# Lab 5 verification report — activation patching and causal tracing

Date: 2026-06-11 · Machine: Colab A100-SXM4-80GB · Branch: `lab1_colab`

## What was built

- **Bench**: interchange interventions on the residual stream
  (`run_with_residual_patch`, exact streams[k] convention) and on verified component
  outputs (`run_with_component_patch`); **patch no-op self-check #5** (self-patching
  must be bit-exact identity — guards against off-by-one layer/position indexing that
  would render beautiful, wrong heatmaps); component capture extended to all positions;
  safe rank-one weight-edit plumbing (`temporary_rank_one_edit`, guaranteed restore;
  Conv1D/Linear orientation handled).
- **Lab**: 16 capital facts with cyclic corrupt partners, all differing in exactly one
  single-token subject (alignment validator REJECTS, never warns); baseline gate with
  margins (drops counted); full layer×position grids; role-aggregated causal tracing;
  paraphrase confirmation under two extra templates; mismatched-pair / wrong-position /
  held-out-low-region controls; component-level pass in the localized band; rank-one
  edit-and-audit extension with a dose sweep (`--run-edit`, alpha ∈ {1,2,4}, localized
  vs alternative layer).

## Key design correction made during validation

With substitution corruption, subject-position recovery at layer 0 is a **tautology**
(the patch just swaps the token embedding). The lab therefore anchors on the
**handoff layer** — where subject recovery collapses because the fact has moved toward
the readout — and defines the localized band as the layers just before it. The handout
teaches why, and asks what corruption type would make early layers informative (ROME's
noise corruption).

## Validation evidence

| Check / measurement | Tier A (gpt2) | Tier B (Olmo-3-7B, 16 facts) |
|---|---|---|
| Patch no-op check | exact (0.00e+00) | exact |
| Baseline gate | 6/6 pairs pass | 16/16 pairs pass, 0 rejected by validator |
| Subject-recovery handoff | layer 10/12 | layer 22/32 (band 19–21 at ~0.66) |
| Last-position recovery peak | 0.90 @ L11 | 0.99 @ L31 |
| Matched top patch vs controls | — | 0.67 vs mismatched 0.25, wrong-pos 0.12, held-out low region 0.01 |
| Paraphrase consistency | — | both paraphrase subject curves track the base curve closely (see plot) |

The Tier B localization plot is the classic causal-tracing result: subject and last
curves crossing at ~L21, with the fact "in transit" between recall and readout.

## The edit extension: the negative result with its mechanism showing

The rank-one edit (corrupt fact's MLP output written for the clean fact's key — "the
patch made permanent") did **not** flip Lisbon→Bangkok at any dose at either layer:
at the localized layer L19, 4× dose moved an 11.3-logit gap by only 0.8 logits while
neighbor damage had already begun (3/3 → 2/3 intact); the alternative layer moved it
the *wrong* way. The mechanism of the Hase et al. tension is visible in the data:
stream patches carry the **entire accumulated subject representation**, while the edit
changes **one MLP's write** — "the band is causally sufficient" and "no single layer
is individually decisive" are both true. The lab records `direct_logit_diff_before/
after` so movement-without-flip is measurable, and the handout's reconciliation
assignment is built on exactly this distinction. Per the course rubric, this negative
result is credit, not failure.

## Runs included

- `lab05_patching_causal_tracing-*/` — Tier A smoke (gpt2, 6 facts, full pipeline incl. edits)
- `lab05_tierb_full/` — Tier B science run (Olmo-3-7B, 16 facts, ~2,700 patched
  forwards, paraphrase sweeps, component pass, controls, 6 edit audits)
