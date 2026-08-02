# Gemma transport reproduction contract

This track is isolated to branch `interp_jspace_gemma_transport`, package
`interpretability/jspace_gemma/`, registry `reports/evidence_events.jsonl`,
evidence prefix `gm-`, and Drive run root
`gemma_transport_20260802/`. It never writes Phase 4 or OLMo-side artifacts.

Every model producer must start from a clean Git tree on the Gemma branch,
pin a 40-hex model revision, require CUDA, force eager/no-cache execution for
the exact suffix, verify architecture and clean full-forward parity, and bind
its config/environment/input hashes to a resumable state header. Output paths
are new or manifest-compatible resumptions; evidence IDs are never reused.

Exact means autodiff JVP from `torch.func.jvp` or
`torch.autograd.functional.jvp`, or a local unfused operation that first
passes clean-forward, ordinary-gradient, moderate-secant, and control parity.
A central or forward finite difference is always called a secant and cannot
replace a blocked JVP.

The OLMo positive control and random-direction baselines calibrate all numeric
G1 target thresholds. Those thresholds are committed in
`gm_g1_thresholds_frozen.yaml` before the first Gemma target number is
produced. The fixed delivery floor is cosine >= 0.999 and relative norm error
<= 0.01; smaller undelivered doses are unmeasurable, not nonlinear.

Raw vectors are durable and metrics are recomputed from them. Prompt is the
resampling unit; direction and layer are repeated measures. Claims always
carry model revision, layer, source position/mode, target, direction, dose,
and development/methods tier.
