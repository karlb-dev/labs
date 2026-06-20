# Suggested ledger claims from L22 (olmo3_7b_full_s0)

These are drafts with measured numbers filled in. Edit them until you
would defend them, then move them into claim_ledger.md (or re-run with
--append-ledger to copy them verbatim, and edit there).

```text
[L22-C1] DECODE | At stream depth 9, the eval-context direction separated held-out eval and natural prompts, but the weakest leave-one-format null-adjusted gap was -0.0312. The defensible claim is a partial decode result, not a fully validated cross-format handle.
Artifact: validation/lab22/olmo3_7b_full_s0_cross_format_transfer.csv | Falsifier: Held-out formats fail, format controls project like eval prompts, or prompt-surface baselines match the direction.

[L22-C2] CAUSAL | Activation addition at layer 8 shifted eval/test markers on benign generations by 0.0 over random/shuffled controls, but this remains marker-based until `steering_generations.csv` is hand-labeled.
Artifact: source-run steering_generations.csv | Falsifier: Random/shuffled steering matches the shift, or hand labels show only verbosity, hedging, refusal, or answer-format artifacts.

```
