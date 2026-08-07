# OLMO_LINEAGE_SYNTHESIS.md — A4 output

Inputs: `tables/olmo_lineage_matrix_inputs.csv` (340 rows; reconstruction
138/139 byte-identical incl. sha-matched bootstrap distributions) and
`tables/olmo_lineage_evidence_matrix.csv` (72 rendered cells). Figure:
`figures/olmo_lineage_evidence_matrix.{png,pdf}`. Frames and cohorts are
never mixed: matrix causal cells are **own frame** on the **all-four
common cohort** (equal-family); the common-frame variant exists
separately in the inputs table and agrees in sign.

## The verified trajectory (development tier throughout)

| Checkpoint | Bank-S direct (own, common cohort) | Bank-S composition | Resolved? |
|---|---|---|---|
| OLMo-3 Base | +0.0107 [−0.026, +0.053] | −0.0009 [−0.037, +0.030] | no |
| OLMo-3.0 Think | **−0.1287 [−0.234, −0.041]** | +0.0979 [−0.000, +0.212] | direct yes |
| OLMo-3.1 Think | **−0.1364 [−0.227, −0.054]** | **+0.1160 [+0.046, +0.192]** | both yes |
| OLMo-3.1 Instruct (sibling) | −0.0155 [−0.058, +0.025] | +0.0345 [−0.006, +0.076] | no |

Meanwhile: occupancy medians are **identical** across all four
checkpoints (2); the equal-layer Base→3.0-Think centered-excess
difference is 1.54e-4 inside the ±0.0025 equivalence margin; and the
geometry moves **once**: mapped-token rows shift at Base→3.0-Think
(movement 0.32556; to-Base q50 cosine 0.6744) and then barely at
3.0→3.1 (0.00604), while the **raw operator stays near-stationary
throughout** (to-Base q50 0.9614/0.9548/0.9523; raw unembedding rows
0.9853) — the reorganization lives in the fitted J dictionary, not the
raw map. Selected-ID overlap with Base is 0.3333 for every post-Base
checkpoint; projector overlap 0.334/0.322/0.265.

## Which account does the table support?

- **Capacity growth — no.** Occupancy flat, excess equivalence passes;
  nothing material grows.
- **Coordinate formation — yes, once.** The J-mapped dictionary
  reorganizes at the first released Think transition and is then
  stationary along the Think path.
- **Recruitment/routing — best-supported account.** Causal Bank-S use
  appears exactly where the Think path runs (3.0 Think, 3.1 Think) and
  is unresolved at Base and at the Instruct **sibling** — with no
  capacity or occupancy change anywhere. The composition component
  resolves only at 3.1 Think.
- **Behavioral strategy only — disfavored** as a complete account:
  J-mapped geometry and causal use both move; a purely behavioral story
  predicts neither.
- **External-state substitution — unresolved by design**: no Bank-W
  intervention exists (cross-model 16/20 blocked; pair 0.7788 at 16,
  first passing count 18 — a planning closure, not a null).

## Mandatory boundaries (verified against the frozen records)

1. SFT/DPO intervention cells are `capability_gated_missing` (972-row
   batteries: 0.617%/0.309% capable; **zero** Bank-S facts capable on
   direct+composed; route `null_or_unresolved`). Stage attribution is
   unresolved in both directions; the wedge is ancestry-qualified, not
   objective-attributed.
2. Instruct is a **sibling**, never a temporal successor; its
   unresolved direct effect is a cross-branch contrast, not a reversal
   event.
3. H6 is a transport-validity row (Base: no licensed regime, 9/336;
   Think: late-anchor-only, L56@ε=0.10 12/12) — it bounds
   lens-as-predictor use, not the paired ablation cells
   (`A5_H6_PHASE3_RECONCILIATION.md`).
4. Bank-W pair power is design evidence; "the OLMo pair rules out
   externalization" is a prohibited formulation.
5. All wording stays "associated with the first released Think
   transition" — the lineage is a natural graph, not an experiment.

## Sentence for Paper A (C2, maximum licensed)

> Across the tested OLMo lineage, measured sparse capacity is broadly
> conserved while the J-mapped dictionary, selected spans, and Bank-S
> causal use reorganize at the first released Think transition — the
> development-tier pattern expected if post-training changes how a
> conserved capacity is recruited, not how much of it exists. The
> official SFT/DPO wedge cannot localize the transition (empty
> prospective cohorts), and external-state substitution remains an open,
> gated question.
