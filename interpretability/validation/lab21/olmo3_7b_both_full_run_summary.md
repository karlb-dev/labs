# Lab 21 Run Summary

## Run identity

- Modes: `lora, safety_depth`
- Main model: `allenai/Olmo-3-7B-Instruct`
- Comparison model: `allenai/Olmo-3-1025-7B`
- Evidence target: `ATTR`, plus `CAUSAL` only for imported or future intervention rows
- Safety wall: no unsafe completion sampling; no refusal ablation; forced-prefix comparisons only

## Headline

`safety-depth-audit-only`: Safety-depth curves were produced without trained Lab 20 adapter weights.

## Numbers to inspect

- Adapter sources found: 0
- Private answer-key access verdict: `public_or_adapter_only`
- LoRA matrix rows: 0
- LoRA layer rows: 0
- Safety divergence rows: 3960
- Boundary-vs-safe rows: 792
- Forced-prefix rows: 19800

## Reading order

1. `olmo3_7b_both_full_training_depth_card.md` for the verdict and non-claims.
2. `olmo3_7b_both_full_lab21_safety_wall.json` for the hard safety scope.
3. `olmo3_7b_both_full_training_depth_evidence_matrix.csv` for the whole evidence board.
4. LoRA mode: `olmo3_7b_both_full_adapter_source_manifest.csv`,
   `olmo3_7b_both_full_lora_concentration_summary.csv`, and
   `olmo3_7b_both_full_lora_phase_summary.csv`.
5. Safety mode: `olmo3_7b_both_full_safety_depth_signal_summary.csv` and
   `olmo3_7b_both_full_forced_prefix_summary_by_token_depth.csv`.
6. `olmo3_7b_both_full_operationalization_audit.md` before writing ledger claims.
