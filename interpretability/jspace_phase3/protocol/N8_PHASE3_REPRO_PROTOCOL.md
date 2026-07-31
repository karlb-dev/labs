# N8-P3-L1 — expected-value-blind Phase 3 analysis reproduction

The reproducer must not read any report, handout, evidence registry,
inprogress file, figure, or audit result JSON. It receives only this schema,
the frozen raw parquets, the frozen partition, and the baseline-only
protected-answer rank parquet. Expected values are compared only after the
reproducer seals its report.

## Inputs

Set `N8_P3_RUN_IN` to the Phase 3 artifact root. Read only:

- `metrics/<slug>/p3_grid/p3_grid_<slug>.parquet`
- `metrics/<slug>/p3_grid_replication/p3_grid_replication_<slug>.parquet`
- `preregistration/partition_phase3.json`
- `metrics/qwen36-27b/release_audit/protected_answer/`
  `p3_protected_answer_ranks_qwen.parquet`

for `slug` in `olmo31-think`, `olmo31-instruct`, and `qwen36-27b`.
Record the SHA-256 of every input.

## Frozen analysis schema

- P3-P1: per-item span-safe specific effect is
  `(lp_meanJ_span_safe - lp_baseline) -
   (lp_ss_matched - lp_baseline)`.
  Form composed-minus-direct within each fact, then Qwen minus the mean
  of the two OLMo models. Average within canonical family, then equally
  across families. Enumerate every family sign pattern exactly.
- P3-P2: on Qwen, compare indicators
  `1[delta_J < -1] - 1[delta_control < -1]` using an equal-family
  statistic and within-item sign flips. Use 100,000 draws, seed 4242,
  and a plus-one p-value. Report all-items, exact-scored-alias rank
  `<=10`, and any-accepted-alias minimum rank `<=10`.
- P3-P3: on non-null Qwen bridge rows, compare
  `lp_true_bridge - lp_distractor_bridge`, equal-family weighted,
  with the same draws/seed/plus-one convention.
- Apply Holm within the available P3-P1/P2/P3 family.
- Report item weighting, relation-group weighting, Think-minus-Instruct
  thick composition, per-model Bank-S composition, and both partitions.

## Command and sealed outputs

Set `N8_P3_OUT` to a fresh empty directory, then run:

```bash
python -m jspace_phase3.experiments.p3_n8_phase3_analysis
```

The process writes `N8_P3_L1_REPORT.json` and `N8_P3_L1_REPORT.md`.
It must stop without comparing its values to any campaign output.

