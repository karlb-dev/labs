# Five-Lab End-to-End Update Report - 2026-06-20

Scope: Lab 25, Lab 34, Lab 22, Lab 16, and Lab 21.

## Commits

- `401706f` Repair Lab 25 self-report validation
- `242bb36` Tighten Lab 25 source attribution rubric
- `e7c4909` Strengthen Lab 34 tool-use validation
- `4d3110b` Strengthen Lab 22 eval-awareness controls
- `9443544` Document Lab 16 sycophancy data provenance
- `61df4ba` Add Lab 21 safety-depth prompt corpus

Note: `672befe` was pushed from the other machine and was included by rebase before the final Lab 21 push.

## Drive Backups

- `/content/drive/MyDrive/interpret/lab25_findwire_repair_20260620/`
- `/content/drive/MyDrive/interpret/lab34_tool_use_repair_20260620/`
- `/content/drive/MyDrive/interpret/lab22_eval_awareness_repair_20260620/`
- `/content/drive/MyDrive/interpret/lab16_sycophancy_audit_20260620/`
- `/content/drive/MyDrive/interpret/lab21_safety_depth_repair_20260620/`

## Lab 25 - Find the Wire

Main repair: replaced the forced-choice self-report prompt with separate open report and neutral behavior prompts, expanded the introspection corpus to 24 rows, and tightened source-attribution scoring.

Key final run:

- `runs/lab25_repair_olmo7b_full_final_20260620`
- Model: `allenai/Olmo-3-7B-Instruct`
- `n_items=24`, `n_direction_rows=8`, `n_generation_rows=192`, `n_source_rows=60`
- Main report prompt leak rate: `0.0`
- Diagnostic grounded prompt leak rate: `1.0`
- Target direction detection rate: `0.0`
- Source attribution accuracy: `0.4667`

Result: clean negative for the self-report/detection claim after fixing the measurement leak. Source attribution remains weak, especially activation-source attribution.

## Lab 34 - Tool Use State

Main repair: expanded tool-use data from 48 to 84 rows, added a data card, and strengthened causal validation with five random controls.

Key final run:

- `runs/lab34_repair_olmo7b_full_20260620`
- Model: `allenai/Olmo-3-7B-Instruct`
- `tool_needed_auc=0.9236`
- `tool_needed_accuracy=0.8846`
- `tool_selection_accuracy=0.7692`
- `surface_control_accuracy=0.3077`
- `decode_gap_over_surface=0.4615`
- `causal_shift_over_random=0.112`

Result: clean scoped positive on the toy tool-use harness. This supports a tool-needed/tool-selection representation in this controlled setup, not a broad agentic tool-use claim.

## Lab 22 - Eval Awareness

Main repair: replaced the too-small surface-confounded prompt set with a deterministic 96-row `v2_surface_matched` corpus, added a data card/generator, and tightened the verdict so leave-one-format transfer must beat random/shuffled null rails.

Key final runs:

- `runs/lab22_repair_olmo7b_full_v2_20260620`
- `runs/lab22_repair_olmo7b_full_v2_seed1_20260620`
- `runs/lab22_repair_olmo7b_think_full_v2_20260620`

OLMo 7B Instruct, seed 0:

- `n_items=96`, `n_groups=24`
- Held-out eval-vs-natural AUC: `0.9306`
- Eval-vs-format-control AUC: `0.7917`
- Cross-format min/mean AUC: `0.9062 / 0.9635`
- Cross-format min null-adjusted AUC gap: `-0.0312`
- Best surface baseline AUC: `0.7049`
- Decode verdict: `cross_format_but_null_or_surface_controls_competitive`
- Causal marker verdict: `not_validated_by_marker_controls`

Result: partial decode positive. The model has a strong eval-context signal, but one held-out format does not clear the stricter null/control gate, and steering does not produce a validated causal marker shift.

## Lab 16 - Sycophancy and User-Belief

Main action: audited the existing lab and found it already science-ready. Added data provenance/card and reran validation. No science rewrite was needed.

Key final run:

- `runs/lab16_provenance_olmo7b_full_20260620`
- Model: `allenai/Olmo-3-7B-Instruct`
- Data: `v1_misconception_pressure`, 240 rows / 40 base facts
- User-belief AUC/control: `0.72 / 0.5228`
- User-belief selectivity: `0.1972`
- Local-truth AUC/control at selected depth: `0.9067 / 0.8578`
- False-pressure sycophancy rate given neutral-correct facts: `0.027`
- Agreement steering delta/specificity gap: `0.4 / 0.0`

Result: selective user-belief decode positive, behavioral sycophancy negative for OLMo 7B, causal steering negative because controls match the agreement shift.

## Lab 21 - LoRA Safety Depth

Main repair: the handout referenced `data/safety_depth_boundary_pairs.csv`, but the file was not vendored, so science runs used a six-pair smoke fallback. Added a deterministic 24-row frozen safety-depth corpus, data card, manifest hash, and top-level metrics provenance.

Key final run:

- `runs/lab21_safetydata_olmo7b_both_full_20260620`
- Model: `allenai/Olmo-3-7B-Instruct`
- Comparison: `allenai/Olmo-3-1025-7B`
- Modes: `lora`, `safety_depth`
- Safety data: `v1_boundary_safe_pairs`, manifest match `true`
- `n_safety_pairs=24`
- `n_safety_divergence_rows=3960`
- `n_boundary_safe_rows=792`
- `n_forced_prefix_rows=19800`
- Base/instruct peak: depth `32`, value `1.2409`
- Chat-format peak: depth `32`, value `1.2105`
- Boundary/safe peak: depth `32`, value `0.9881`
- Forced-prefix peak: depth `32`, value `1.3475`
- LoRA adapter sources: `0`

Result: improved audit-only result. Safety-depth representational curves are now reproducible on a real frozen corpus, but no LoRA adapter weights or causal wrapper/layer intervention rows were available, so no mechanism claim is earned.

## Commands Run

Representative validation commands:

```bash
python -m py_compile interpretability/interp_bench.py interpretability/labs/lab22_eval_awareness.py interpretability/data/make_eval_awareness_contexts.py
python interp_bench.py --lab lab22 --tier b --prompt-set full --no-plots --run-name lab22_repair_olmo7b_full_v2_20260620
python interp_bench.py --lab lab22 --tier c --prompt-set full --no-plots --run-name lab22_repair_olmo7b_think_full_v2_20260620
python interp_bench.py --lab lab16 --tier b --prompt-set full --no-plots --run-name lab16_provenance_olmo7b_full_20260620
python interp_bench.py --lab lab21 --tier b --mode both --prompt-set full --no-plots --run-name lab21_safetydata_olmo7b_both_full_20260620
```

## Overall Read

- Clean positive: Lab 34, scoped to the toy tool-use harness.
- Partial positive: Lab 22 decode only; Lab 16 user-belief decode only.
- Clean negative: Lab 25 self-report detection; Lab 16 causal agreement steering; Lab 22 causal marker steering.
- Clean audit-only: Lab 21 safety-depth on a now-vendored corpus; causal/LoRA mechanism claims remain unavailable without adapter weights or imported intervention rows.
