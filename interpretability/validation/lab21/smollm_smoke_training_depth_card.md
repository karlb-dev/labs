# Lab 21 Training-Depth Card

**Verdict:** `safety-depth-audit-only`

Safety-depth curves were produced without trained Lab 20 adapter weights.

## What ran

- Modes: `safety_depth`
- Main model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Comparison model: `HuggingFaceTB/SmolLM2-135M`
- Adapter sources found: 0
- Private answer-key access verdict: `public_or_adapter_only`
- LoRA layer rows: 0
- Safety divergence rows: 310
- Forced-prefix rows: 1550

## Current strongest readings

- LoRA top layer: ; top-layer share: ; top-3 share: .
- Base-vs-instruct divergence peak depth: 30; half-peak persists through: 30.
- Boundary-vs-safe prompt divergence peak depth: 30; half-peak persists through: 30.
- Forced-prefix divergence peak depth: 30; half-peak persists through: 30.

## Non-claims

- High LoRA norm is not proof that the behavior is computed there.
- A first-token behavioral gate is not proof that the representation is shallow.
- Boundary-vs-safe divergence can be semantic difference, format difference, or refusal-related difference. The controls decide how much survives.

## Read next

1. Use the primary chat-format summaries before reading safety-depth curves.
2. `smollm_smoke_safety_depth_signal_summary.csv` for the depth curves.
3. `smollm_smoke_training_depth_evidence_matrix.csv` for the joined evidence board.
4. Use the primary both-mode report for full non-claims and control posture.
