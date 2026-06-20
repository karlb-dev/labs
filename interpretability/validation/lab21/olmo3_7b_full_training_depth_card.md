# Lab 21 Training-Depth Card

**Verdict:** `safety-depth-audit-only`

Safety-depth curves were produced without trained Lab 20 adapter weights.

## What ran

- Modes: `safety_depth`
- Main model: `allenai/Olmo-3-7B-Instruct`
- Comparison model: `allenai/Olmo-3-1025-7B`
- Adapter sources found: 0
- Private answer-key access verdict: `public_or_adapter_only`
- LoRA layer rows: 0
- Safety divergence rows: 3960
- Forced-prefix rows: 19800

## Current strongest readings

- LoRA top layer: ; top-layer share: ; top-3 share: .
- Base-vs-instruct divergence peak depth: 32; half-peak persists through: 32.
- Boundary-vs-safe prompt divergence peak depth: 32; half-peak persists through: 32.
- Forced-prefix divergence peak depth: 32; half-peak persists through: 32.

## Non-claims

- High LoRA norm is not proof that the behavior is computed there.
- A first-token behavioral gate is not proof that the representation is shallow.
- Boundary-vs-safe divergence can be semantic difference, format difference, or refusal-related difference. The controls decide how much survives.

## Read next

1. Use `olmo3_7b_both_full_chat_format_divergence_summary_by_depth.csv`
   before reading safety-depth curves.
2. `olmo3_7b_full_safety_depth_signal_summary.csv` for the depth curves.
3. `olmo3_7b_full_training_depth_evidence_matrix.csv` for the joined evidence board.
4. Use the primary both-mode report for full non-claims and control posture.
