# Lab 16 Validation Report

## Source Runs

Final validation uses the June 20, 2026 repaired sycophancy-audit bundle:

- `olmo3_7b_full`: full Olmo-3-7B-Instruct run on the frozen 240-row
  misconception-pressure dataset.
- `smollm_smoke`: small SmolLM2-135M-Instruct smoke run on 60 rows.

The older June 15 sweep artifacts were removed from this validation directory
so the pack matches the current lab implementation and artifact contract.

## Evidence Status

| Evidence object | Rung | Primary value | Verdict | Interpretation |
|---|---|---:|---|---|
| Behavior under false-user pressure | OBS | 0.027 neutral-correct sycophancy rate | `rare_or_not_observed_on_known_facts` | Olmo 7B usually does not endorse the false answer when it knows the base fact. |
| User-belief frame direction | DECODE | 0.1972 AUC gap over control | `validated_selective` | The repaired lab finds a selective prompt-frame handle at the assistant boundary. |
| Local-truth direction | DECODE | 0.0489 AUC gap over control | `weak_or_control_matched` | The truth direction is high-AUC, but too close to its control to carry a strong claim. |
| Truth vs user-belief dissociation | INTEGRATION | projection correlation 0.3166 | `dissociated` | Useful as a frame-level distinction, not a belief-state claim. |
| Agreement steering specificity | CAUSAL | 0.0 specificity gap | `not_specific_or_too_small` | Agreement steering does not beat the best controls at max dose. |
| Manual label readiness | HUMAN AUDIT | 0.0 hand-label fraction | `needs_hand_labels_before_defended_claims` | The scaffold is present, but human labels are still needed for defended behavior claims. |

## Detailed Read

The main repair changed the lab from a simple sycophancy-rate exercise into a
three-rung audit: generation behavior, decodable prompt-frame directions, and
activation-addition effects. That makes the current result much clearer.

On the primary Olmo run, the user-belief-frame direction is the real win:
held-out AUC is 0.72, while the best control AUC is 0.5228. The selected site is
the assistant boundary at stream depth 20, which is also the right place to
teach the operational meaning: this is a contrast in rendered prompt frames,
not a claim that the model has a human-like belief state.

Behavioral sycophancy is not strong in the primary run. The neutral prompt gets
92.5 percent of base facts correct, and the false-pressure sycophancy rate
conditioned on those neutral-correct facts is only 2.7 percent. Mild and
authority pressure each reach 5.41 percent on neutral-correct facts, while
false-belief and identity-pressure variants are at zero in the same slice.

The steering result is also not specific enough. Agreement activation addition
raises sycophancy by 0.4 at the largest dose, but shuffled agreement and
politeness controls can match that movement, leaving a specificity gap of 0.0.
That is useful pedagogy: an effect can move the behavior while still failing
the causal interpretation.

The SmolLM smoke run is a good counterexample. It shows high neutral-correct
sycophancy, 0.5625, but the user-belief decode is below control, 0.34 vs 0.435.
That should be read as a small-model smoke result, not as the final lab claim.

## Claim Boundary

Allowed:

- Lab 16 finds a selective Olmo 7B direction for user-belief framing.
- Lab 16 gives students a clean example of separating behavior, decode, and
  causal steering evidence.
- The current behavior result is mostly negative for Olmo 7B sycophancy on
  known facts.

Not allowed:

- The model has a stable human-like belief state.
- The lab found broad sycophancy in Olmo 7B.
- Agreement steering is a specific causal sycophancy handle under the current
  controls.

## Follow-Up Needed For A Stronger Behavioral Claim

The lab has the right scaffold for stronger future claims: hand-labeling guide,
base-fact outcome matrix, condition contrasts, confound-risk tables, and
steering controls. The missing step is filling the manual label sample and
rerunning on larger models or more pressure families if the course wants a
behavioral sycophancy result rather than the current representation-first
lesson.
