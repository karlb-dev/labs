# Deviations and incidents

No scientific deviations or producer incidents are registered at the
foundation boundary. Append dated entries; do not rewrite prior entries.

## 2026-08-02 — pre-model batch/delivery audit

The first control-runner commit was used only to stage and fully hash the OLMo
snapshot. Before any model was loaded or any current-study response existed, a
read-only code audit identified batch-shape baseline drift and unadjusted
post-cast sign/scale/sum mismatch as alternative explanations for very small
secants. The runner was corrected as documented in `G1_FROZEN_DESIGN.md` and
must be committed cleanly before the one-cell OLMo smoke. There is no model
evidence to withdraw or supersede.
