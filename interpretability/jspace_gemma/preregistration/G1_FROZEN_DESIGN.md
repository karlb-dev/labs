# G1 frozen design and threshold firewall

The prompt bank, Stage 1/2 membership, target checkpoints, source modes,
direction families, epsilon ladder, backend order, cache policy, and raw-row
schema are frozen in `configs/gm_g1_design.yaml` before control execution.

Numeric target thresholds are intentionally not frozen here. The binding
addendum requires them to be calibrated on the in-session OLMo control,
random directions, clean-repeat floor, wrong-hook sentinel, and smallest
faithfully delivered epsilon. Until a clean commit adds
`configs/gm_g1_thresholds_frozen.yaml` and flips
`gemma_execution_allowed: true`, the target producer must refuse to load
Gemma weights or emit a Gemma scientific cell.

The delivery thresholds (cosine 0.999 and relative norm error 0.01) come
directly from the governing plan and are already fixed. No observed Gemma
number may be used to alter a pass/fail threshold.
