# Transport gate protocol

This is the short preflight exported by the Gemma side track. It is designed
to stop an invalid exact-JVP transport assay before mechanism or workspace
interpretation. Budget about 20 minutes excluding model download.

## 1. Establish identity and isolation

- Work on a dedicated branch, package, registry prefix, local staging root,
  and Drive run root.
- Record the exact repository fork, model revision, tokenizer revision,
  architecture manifest, and environment lock.
- Require a clean, pushed producer commit. Fetch and integrate the remote
  branch before each push.
- Refuse output-path or evidence-ID reuse. Never edit another concurrent
  phase’s registry or run root.

## 2. Freeze the estimand

Before opening the target result, freeze:

- prompts and prompt hashes;
- source layer and position mask;
- target stage and representation;
- direction IDs, seeds, and hashes;
- epsilon ladder and sign/double request order;
- finite-response and exact-JVP batch shape and slot;
- delivery, SNR, tangent, curvature, and backend-parity thresholds;
- stop rules and claim boundary.

Finite differences are secants, never exact JVPs. A target-derived tolerance
cannot repair a failed target gate.

## 3. Prove the derivative software on goldens

Require both the primary and fallback exact autodiff implementations to match:

1. an analytic directional derivative;
2. a deterministic nonlinear tiny-transformer suffix;
3. an independent reverse-Jacobian calculation where tractable;
4. the expected quadratic central-secant convergence pattern.

Record implementation hashes. If no exact backend succeeds, register a
methods blocker and stop.

## 4. Match the model path exactly

- Functionalize a no-cache eager suffix from one explicit source activation.
- Reconstruct exact attention masks, position IDs, rotary inputs, and any
  architecture-specific per-layer state.
- Require clean suffix/full-forward parity before perturbation.
- Run the finite clean baseline, perturbed forward, and exact-JVP primal with
  identical batch shape and request slot.
- Preserve a wrong-hook sentinel and clean-repeat noise measurement.

If a fused operation must be replaced, first require clean-forward, ordinary
gradient, moderate-secant, and positive-control parity.

## 5. Audit mixed-precision delivery

Construct desired perturbations in fp32, pass them through the exact model
cast, and use the realized post-cast vector for every exact JVP. Record desired
and realized hashes, norms, cosine, and relative norm error. Treat an
undelivered dose or response below the frozen SNR floor as unmeasurable.

For this campaign the delivery gate is cosine at least 0.999 and relative norm
error at most 0.01. Response SNR uses the smaller of secant and JVP norms over
the larger of clean-repeat noise and the target-dtype half-step floor.

## 6. Run a positive control before the target

Execute the identical harness on a checkpoint with a known transport-valid
band. Freeze all target thresholds from the control and random baselines.
Register the control result before opening the target. If it fails, debug the
harness; do not conclude that both models are nonlinear.

## 7. Run one target smoke and dual-backend replay

Predeclare a single target cell and retain raw source, secant, and tangent
vectors. Recompute all metrics from raw tensors. Then replay one frozen cell
with both exact backends using the original batch members and slot.

Gate both:

- the selected scientific slot; and
- the full matched batch, so an agreeing selected slot cannot hide a
  sign-, scale-, or slot-dependent derivative-path discrepancy.

Record backend success, primal parity, tangent cosine/relative error,
source/target replay, vector hashes, and metric replay. Never relax a failed
cross-backend threshold after seeing the target.

## 8. Apply the decision tree

| Observation | Action |
|---|---|
| Goldens fail | repair software; no model interpretation |
| Positive control fails | repair harness; no target interpretation |
| Clean/path/delivery/SNR gate fails | register methods blocker |
| Target backends disagree | register methods blocker; stop mechanism work |
| Backends agree and tiny secant matches | finite-radius tests may proceed |
| Backends agree but tiny secant misses | run prospectively defined path/unfused audit before mechanism claims |

Every failed attempt remains immutable and receives a new evidence ID if
repaired. A later correction must state the supersession relation.

## 9. Publish the boundary

At every evidence boundary:

1. hash and independently reread raw outputs;
2. append the isolated registry event;
3. update the report and restart handoff;
4. fetch/integrate the remote branch;
5. run the full tests and live-output verifier;
6. push the exact branch;
7. record the pushed commit in Drive.

The exported import bundle is methods-only. It cannot open a confirmatory
cell, sign an independent-review field, or convert an instrument blocker into
a claim that a workspace or information is absent.
