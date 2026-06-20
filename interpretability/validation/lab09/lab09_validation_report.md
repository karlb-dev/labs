# Lab 9 Validation Report

## Verdict

Lab 9 is a clean scoped positive. The implementation is clear and empirically
solid for the GPT-2 factual-recall miniature it claims to be.

## What Was Checked

- Recompiled `interp_bench.py` and `labs/lab09_attribution_graphs.py`.
- Ran a fresh Tier C GPT-2 validation with eager attention and float32.
- Verified replacement exactness, edge reconstruction, and feature-edit no-op
  checks.
- Compared graph-guided real-model edits against the matched random control.
- Backed up the full fresh run to Drive.

## Result

The fresh Tier C run validates the mechanism-shaped claim:

- replacement max logit diff: `0.00029`;
- edge reconstruction relative error: `0.0000058`;
- feature-edit no-op max logit diff: `0.0`;
- suppression drop: `4.8549` logits;
- random-control drop: `-0.0112` logits;
- specificity gap: `4.8661` logits;
- substitution shift: `8.2680` logits.

## What Changed

No Lab 9 science code was changed. The code already has the receipts this lab
needs. The validation pack was updated to foreground the fresh run and remove
ambiguity about what the lab does and does not prove.

## Boundary

This is not a claim that attribution graphs are generally reliable. It is a
validated example of the course evidence contract: replacement graph first,
explicit dictionary/error accounting, then real-model interventions with a
matched random control.
