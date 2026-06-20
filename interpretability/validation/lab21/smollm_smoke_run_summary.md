# Lab 21 Run Summary

## Run identity

- Modes: `safety_depth`
- Main model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Comparison model: `HuggingFaceTB/SmolLM2-135M`
- Evidence target: `ATTR`, plus `CAUSAL` only for imported or future intervention rows
- Safety wall: no unsafe completion sampling; no refusal ablation; forced-prefix comparisons only

## Headline

`safety-depth-audit-only`: Safety-depth curves were produced without trained Lab 20 adapter weights.

## Numbers to inspect

- Adapter sources found: 0
- Private answer-key access verdict: `public_or_adapter_only`
- LoRA matrix rows: 0
- LoRA layer rows: 0
- Safety divergence rows: 310
- Boundary-vs-safe rows: 62
- Forced-prefix rows: 1550

## Reading order

1. `smollm_smoke_training_depth_card.md` for the verdict and non-claims.
2. The primary `olmo3_7b_both_full_lab21_safety_wall.json` for the hard safety scope.
3. `smollm_smoke_training_depth_evidence_matrix.csv` for the evidence board.
4. Safety mode: `smollm_smoke_safety_depth_signal_summary.csv`.
5. Use the primary both-mode report before writing ledger claims.
