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

## 2026-08-02 — OLMo post-compute summary serialization

All 56 OLMo calibration cells and their atomic metric/raw checkpoints completed
from clean pushed commit `06b2a3d`. The producer then failed before summary,
Parquet, inventory, or registry creation because pandas supplied
`numpy.int64` source-layer group keys to the standard JSON encoder. The state
and every cell/raw hash verify; no cell will be rewritten or rerun. A separate
hash-verifying finalizer must record the original compute commit and the later
finalization commit independently. Until that finalizer is registered, there
is no live `gm-jvp-olmo-calibration-v1` evidence record and no numeric Gemma
threshold may be frozen.
