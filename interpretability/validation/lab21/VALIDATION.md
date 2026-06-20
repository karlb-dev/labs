# Lab 21 Validation

## Lab 21: Where Training Lives - LoRA Localization and Safety Depth

This directory is the final Lab 21 validation pack for the current repository
code. It replaces the older June 15 validation artifacts with the repaired
June 20, 2026 safety-depth runs.

## Validation Read

The repaired result is a safety-depth audit, not a LoRA localization result.
The run successfully produces audited base-vs-instruct, chat-format,
boundary-vs-safe, and forced-prefix depth curves under the Lab 21 safety wall,
but it found no trained Lab 20 adapter sources. Therefore the LoRA half of the
lab is currently a missing-input result.

The cleanest current claim is:

```text
Lab 21 currently validates an audited safety-depth measurement scaffold. It
does not validate where a LoRA behavior lives, and it does not prove a causal
refusal or safety mechanism at any specific depth.
```

## Headline Result

The primary validation run is `olmo3_7b_both_full`:

- Main model: `allenai/Olmo-3-7B-Instruct`
- Comparison model: `allenai/Olmo-3-1025-7B`
- Modes requested: `lora`, `safety_depth`
- Frozen safety-depth pairs: 24
- Adapter sources found: 0
- LoRA layer rows: 0
- Safety divergence rows: 3960
- Boundary-vs-safe rows: 792
- Forced-prefix rows: 19800
- Private answer-key access verdict: `public_or_adapter_only`
- Base-vs-instruct peak depth: 32
- Chat-format control peak depth: 32
- Boundary-vs-safe peak depth: 32
- Forced-prefix peak depth: 32
- Verdict: `safety-depth-audit-only`

The depth curves are useful, but they are not a mechanism claim. The
chat-format control also peaks at depth 32 and explains a large part of the raw
base-vs-instruct distance, so the correct reading is "measured
representational divergence under audited prompts," not "refusal lives at the
final layer."

## Current Result Summary

| Source label | Model | Scope | Main result | Read |
|---|---|---|---|---|
| `olmo3_7b_both_full` | Olmo-3-7B-Instruct | LoRA plus safety-depth audit | 0 adapter sources; 24 safety pairs; depth curves peak at 32 | Final validation run; safety-depth audit only |
| `olmo3_7b_full` | Olmo-3-7B-Instruct | safety-depth-only check | Same safety-depth counts and peaks as the both-mode run | Confirms the safety-depth path without LoRA mode |
| `smollm_smoke` | SmolLM2-135M-Instruct | 2-pair smoke | 310 safety-divergence rows; depth curves peak at 30 | Tier-A plumbing check only |

The broader run ledger is in `lab21_validation_summary.csv`.

## What This Lab Teaches

- "Depth" has several meanings: adapter weight depth, model-pair
  representational distance, chat-format control distance, boundary/safe prompt
  distance, and forced-prefix trajectory.
- Missing adapter weights are a valid result. They prevent LoRA localization
  claims rather than weakening the safety-depth audit.
- A final-layer peak is not a refusal mechanism. It must be read beside
  chat-format controls, boundary/safe controls, and the safety wall.
- The lab is useful because it forces students to separate descriptive
  localization, control audits, and causal intervention evidence.

## Curated Artifacts

Summary:

- `lab21_validation_report.md`
- `lab21_validation_summary.csv`

Primary Olmo run:

- `olmo3_7b_both_full_run_summary.md`
- `olmo3_7b_both_full_training_depth_card.md`
- `olmo3_7b_both_full_operationalization_audit.md`
- `olmo3_7b_both_full_ledger_suggestions.md`
- `olmo3_7b_both_full_metrics.json`
- `olmo3_7b_both_full_results.csv`
- `olmo3_7b_both_full_training_depth_evidence_matrix.csv`
- `olmo3_7b_both_full_safety_depth_signal_summary.csv`
- `olmo3_7b_both_full_instruct_base_divergence_summary_by_depth.csv`
- `olmo3_7b_both_full_chat_format_divergence_summary_by_depth.csv`
- `olmo3_7b_both_full_boundary_safe_summary_by_depth.csv`
- `olmo3_7b_both_full_forced_prefix_summary_by_token_depth.csv`
- `olmo3_7b_both_full_refusal_direction_provenance.csv`
- `olmo3_7b_both_full_adapter_source_manifest.csv`
- `olmo3_7b_both_full_wrapper_ablation_test.csv`
- `olmo3_7b_both_full_erosion_order.csv`
- `olmo3_7b_both_full_lab21_safety_wall.json`
- `olmo3_7b_both_full_organism_discovery.json`
- `olmo3_7b_both_full_private_answer_key_access.json`
- `olmo3_7b_both_full_safety_depth_manifest.json`

Auxiliary checks:

- `olmo3_7b_full_*`
- `smollm_smoke_*`

## Caveats

- This is a curated validation directory; full raw runs remain outside this
  pack.
- No repaired-run PNG plots were present in the local run bundle, so this pack
  keeps tables and reports rather than carrying forward old plots.
- LoRA localization requires trained adapter files from Lab 20 or an explicit
  organism directory. This run found none.
- The safety wall forbids harmful completion sampling, refusal ablation, and
  toward-compliance steering. The result is a safe forward-pass audit.
