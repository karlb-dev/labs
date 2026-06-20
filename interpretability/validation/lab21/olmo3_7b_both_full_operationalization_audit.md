# Lab 21 Operationalization Audit

## What was measured

LoRA mode measures weight-space adapter deltas. Safety-depth mode measures residual-state divergence on benign boundary prompts and forced prefixes. No unsafe completions are sampled, and refusal ablation is not implemented.

## Cheap explanations and controls

| Apparent result | Cheap explanation | Artifact that pressures it |
|---|---|---|
| High-norm LoRA layer | Optimizer/update bookkeeping rather than behavior mechanism | `tables/wrapper_ablation_test.csv` must contain a real intervention before mechanism language |
| Low-rank adapter | Behavior is low-rank | `olmo3_7b_both_full_lora_rank_energy.csv` only describes weight energy, not behavioral sufficiency |
| Base-vs-instruct divergence | Chat formatting, tokenizer drift, or global norm shift | `tables/chat_format_divergence.csv`, token hashes, and normalized deltas |
| Boundary-vs-safe divergence | Topic/semantic difference rather than refusal | family-balanced prompt pairs and forced-prefix comparisons |
| Shallow forced-prefix divergence | Text prefix artifact | representational depth curves and forced generic-prefix control |

## Current run verdict

- Verdict: `safety-depth-audit-only`
- Explanation: Safety-depth curves were produced without trained Lab 20 adapter weights.
- Adapter sources found: 0
- Private answer-key access verdict: `public_or_adapter_only`
- LoRA matrix rows: 0
- Safety prompt pairs: 24
- Identity comparison smoke: False

## Allowed claim boundary

Allowed now: localization-style evidence and audited safety-depth measurements. A mechanism claim requires a real intervention row, not a scaffold row, in `wrapper_ablation_test.csv` or `erosion_order.csv`.
