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

## N8-P3-L2/L3 model-cell protocol

The historical `n8_level2_sentinels.py` evidence is N8-P2-L2/L3: that
module relaunches the Phase 2 N6 producer. It is not part of this Phase 3
release gate.

The Phase 3 cell reproducer relaunches `phase3_primary_grid` in a fresh
run root with registration disabled. The model process must pass
`require_cuda_gpu()` before weights are loaded. There is no CPU fallback.
Its durable checkpoint is written atomically every five completed items,
and its root manifest refuses a resume when code, config, partition,
model, lens, G5 cohort, frozen grid, or seed contract differs.

N8-P3-L2 reruns at least 20 sorted confirmatory items for each primary
model. N8-P3-L3 reruns the complete Qwen confirmatory cell. Both include:

- baseline, span-safe J, and label-protected J;
- an exact rank-and-energy matched control under `sha256-v1`, namespace
  `p3-control-seed-audit`, base seed 31337;
- Qwen true/distractor bridge arms on eligible composed items;
- surface protection-set geometry and matched-control conformance logs.

After the producer exits (or, for L2, after its 20-item checkpoint is
sealed), the wrapper compares deterministic arms at tolerance `2e-3`.
The unrecoverable historical Python-hash control realization is reported
as a distributional comparison, never as an exact target. Qwen's new
control is additionally compared bit-for-bit to the already banked
five-seed audit's seed-31337 realization and against its five-seed item
envelope.

Example commands, with each `--cell-root` fresh on first launch:

```bash
python -m jspace_phase3.experiments.p3_n8_phase3_cells \
  --slug qwen36-27b --level 2 --n 20 \
  --cell-root /durable/new/root/qwen-l2

python -m jspace_phase3.experiments.p3_n8_phase3_cells \
  --slug qwen36-27b --level 3 --n 20 \
  --cell-root /durable/new/root/qwen-l3
```

The exact same command and root resume an interrupted job after the
manifest contract is revalidated.
