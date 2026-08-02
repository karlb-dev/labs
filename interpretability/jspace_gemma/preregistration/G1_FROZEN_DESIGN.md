# G1 frozen design and threshold firewall

The prompt bank, Stage 1/2 membership, target checkpoints, source modes,
direction families, epsilon ladder, backend order, cache policy, and raw-row
schema are frozen in `configs/gm_g1_design.yaml` before control execution.

Numeric target thresholds are now frozen in
`configs/gm_g1_thresholds_frozen.yaml` from the registered in-session OLMo
control, random/radial/tangent directions, clean-repeat and target-dtype
floor, wrong-hook sentinel, late identity anchors, and smallest measurable
epsilon. No Gemma weight or target number was opened. A clean producer
reproduced every criterion and registered
`gm-jvp-olmo-positive-control-v1`; the design is now
`FROZEN_PRE_GEMMA_REGISTERED` with `gemma_execution_allowed: true`. This flag
transition changes no numeric threshold.

The delivery thresholds (cosine 0.999 and relative norm error 0.01) come
directly from the governing plan and are already fixed. No observed Gemma
number may be used to alter a pass/fail threshold.

The response-SNR measurement floor is 12; primary pass/fail decisions require
SNR 20. At that decision floor, all 32 L56/L60 single-position
prompt/layer/direction anchors select epsilon 0.10. The primary tangent gate
is cosine at least 0.98, forward relative error at most 0.20, and central
relative error at most 0.10, with at least 90% row passage. The declared
finite dose is 0.10; 0.20 is an audit. Curvature fits may use SNR 12 to retain
three radii. Their control-calibrated intercept ceiling is 0.30 and positive
curvature slope floor is 0.15. These are operational acceptance regions, not
universal constants. Later J-selected directions must meet the same frozen
thresholds after exact hash binding; they cannot trigger recalibration.

Pre-control correction, 2026-08-02: the OLMo grid includes L4 as the required
shallow negative-control layer and L60 as the late identity anchor, in addition
to matched relative-depth layers L24/L32/L40/L47/L56. The initial scaffold
listed only the matched layers and therefore could not implement the plan's
precommitted later-versus-shallow contrast. This correction precedes model
staging and all current-study OLMo/Gemma numbers.

Pre-control path correction, 2026-08-02: a code audit after weight staging but
before model load found that finite perturbed examples were batched while the
subtracted clean target came from a batch-one call. It also found that the
negative, doubled, and pair-sum bf16 perturbations were audited separately but
their vector defects still assumed exact sign/scale/additivity. The frozen
repair uses a separate all-clean forward with the identical batch shape and
slot as every perturbed forward; computes the direct exact JVP on that same
batched primal for each separately realized perturbation; compares central
secants with `(J delta+ - J delta-)/2`; and reports both raw vector defects and
the residual after subtracting the exact first-order delivery mismatch. The
SNR denominator is the maximum of the same-batch clean-repeat norm and a local
target-dtype half-step norm, rather than an artificial `1e-12` denominator
when deterministic repeats are bit-identical. These definitions are frozen in
`gm_g1_design.yaml` before the first current-study model response.
