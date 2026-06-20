# Suggested ledger claims from L34 (olmo3_7b_full)

These are drafts with measured numbers filled in. Edit them until you
would defend them, then move them into claim_ledger.md (or re-run with
--append-ledger to copy them verbatim, and edit there).

```text
[L34-C1] DECODE | Lab 34 method `prompt_boundary_tool_needed_decode` reported tool_needed_auc=0.9236 against control surface_needed_accuracy=0.3077 with posture `supported`. This is a toy-harness signal claim, not an autonomous-plan claim.
Artifact: validation/lab34/olmo3_7b_full_tool_use_evidence_matrix.csv | Falsifier: A held-out surface-cue no-tool set, shuffled labels, or random activation direction matches or beats the measured tool signal.

[L34-C2] DECODE | Lab 34 method `prompt_boundary_tool_selection_decode` reported tool_selection_accuracy=0.7692 against control surface_control_accuracy=0.3077 with posture `supported`. This is a toy-harness signal claim, not an autonomous-plan claim.
Artifact: validation/lab34/olmo3_7b_full_tool_use_evidence_matrix.csv | Falsifier: A held-out surface-cue no-tool set, shuffled labels, or random activation direction matches or beats the measured tool signal.

[L34-C3] CAUSAL | Lab 34 method `constrained_action_letter_activation_addition` reported target_direction_shift_at_scale_1=0.0913 against control random_direction_shift_at_scale_1=-0.0207 with posture `supported_narrow_letter_prompt`. This is a toy-harness signal claim, not an autonomous-plan claim.
Artifact: validation/lab34/olmo3_7b_full_tool_use_evidence_matrix.csv | Falsifier: A held-out surface-cue no-tool set, shuffled labels, or random activation direction matches or beats the measured tool signal.

[L34-C4] OBS+AUDIT | Lab 34 method `deterministic_local_tool_trace` reported result_match_rate=1.0 against control argument_valid_rate=1.0 with posture `trace_validated`. This is a toy-harness signal claim, not an autonomous-plan claim.
Artifact: validation/lab34/olmo3_7b_full_tool_use_evidence_matrix.csv | Falsifier: A held-out surface-cue no-tool set, shuffled labels, or random activation direction matches or beats the measured tool signal.

[L34-C5] SELF-REPORT | Lab 34 method `tool_self_report_review_scaffold` did not earn broad positive language: requires_human_review_rate=1.0 against control model_self_report_generated=0.0 produced posture `review_required_not_introspection`.
Artifact: validation/lab34/olmo3_7b_full_tool_use_evidence_matrix.csv | Falsifier: A held-out surface-cue no-tool set, shuffled labels, or random activation direction matches or beats the measured tool signal.

```
