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

Pre-control correction, 2026-08-02: the OLMo grid includes L4 as the required
shallow negative-control layer and L60 as the late identity anchor, in addition
to matched relative-depth layers L24/L32/L40/L47/L56. The initial scaffold
listed only the matched layers and therefore could not implement the plan's
precommitted later-versus-shallow contrast. This correction precedes model
staging and all current-study OLMo/Gemma numbers.
