# Lab 22 Run Summary: Eval Awareness

- Model: `allenai/Olmo-3-7B-Instruct`
- Rows: 96
- Formats: `['answer_key_check', 'code_test', 'mcq', 'qa_benchmark', 'quality_screen', 'rubric_free_response']`
- Best stream depth: 9
- Injection layer: 8
- Held-out eval-vs-natural AUC: 0.9306
- Held-out eval-vs-format-control AUC: 0.7917
- Cross-format mean AUC: 0.9635
- Cross-format min null-adjusted AUC gap: -0.0312
- Natural steering marker delta over controls: 0.0
- Decode verdict: `cross_format_but_null_or_surface_controls_competitive`
- Causal marker verdict: `not_validated_by_marker_controls`

Start with `eval_awareness_card.md`, then read `operationalization_audit.md` before moving any claim into the ledger.
