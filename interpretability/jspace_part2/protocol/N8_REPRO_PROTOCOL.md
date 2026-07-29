# N8 REPRODUCTION PROTOCOL — Level 1 (analysis reproduction, CPU-only)

You are an independent reproducer. **Do not read any report, handout,
registry text, inprogress file, or figure in this repository or on the
mounted Drive.** Your job is to regenerate analysis outputs from raw
data and record what you get. You are not told any expected values;
whether your numbers match anything is decided by someone else after
your report is sealed.

## Environment requirements

- python ≥3.10 with: torch (CPU ok), numpy, pandas, pyarrow, scipy,
  statsmodels, pyyaml, matplotlib
- the `labs` repository checked out at branch `interp_jspace_part2`
  (any commit at or after `interpretability/jspace_part2` exposes
  `JSPACE_PART2_RUN_ROOT` / `JSPACE_PART2_OUT_ROOT`; verify with
  `grep -n JSPACE_PART2_OUT_ROOT interpretability/jspace_part2/jspace_part2/experiments/confirmatory_analysis.py`)
- `pip install -e interpretability/jspace_part2`

## Inputs (read-only; never write anywhere under this root)

`RUN_IN=/content/drive/MyDrive/interpret/special-lab-1/part2_20260727`

- `$RUN_IN/metrics/<slug>/n6_grid/n6_per_item_<slug>.parquet` for
  slug ∈ {olmo31-think, olmo31-instruct, qwen36-27b}
- `$RUN_IN/metrics/<slug>/n6_grid_repl/n6_per_item_<slug>.parquet`
- `$RUN_IN/metrics/cross_model/partition_manifest.json`

Record the sha256 of every input you read.

## Output root (fresh, empty; everything you produce goes here)

`export N8_OUT=<given to you at launch>`

## Commands

```bash
cd <repo>
export JSPACE_PART2_RUN_ROOT=/content/drive/MyDrive/interpret/special-lab-1/part2_20260727
export JSPACE_PART2_OUT_ROOT=$N8_OUT

# 1. confirmatory-side analysis
python -m jspace_part2.experiments.confirmatory_analysis \
  --slugs olmo31-think,olmo31-instruct,qwen36-27b \
  --eid n8-level1-confirmatory-repro --no-register

# 2. replication-side analysis
python -m jspace_part2.experiments.confirmatory_analysis \
  --slugs olmo31-think,olmo31-instruct,qwen36-27b \
  --dir-suffix _repl --eid n8-level1-replication-repro \
  --out-name replication_analysis.json --no-register
```

If a command errors, record the full traceback in your report and stop
that item; do not patch estimator code. IO-only fixes (a missing
directory, an obviously wrong path) may be worked around; say so.

## Expected output SCHEMA (not values)

Each command writes a v2 envelope JSON
(`{schema_version, payload, payload_sha256, provenance}`) under
`$N8_OUT/metrics/cross_model/`. The payload must contain at least:

- `P_HP1`: {observed_contrast_nats, ci95: [lo, hi], p_boot, n_families, …}
- `P_HP3_qwen`: {rate_diff_fam_weighted, ci95, p_one_sided, n_stratified,
  n_families, …}
- `holm`: {P_HP1: {p_raw, p_holm, reject_at_05}, P_HP3: {…}}
- per-model/task estimation blocks and sensitivity sections.

## Your report (seal when done)

Write `$N8_OUT/N8_LEVEL1_REPORT.json` with:

```json
{
  "runner": "<your identifier>",
  "repo_commit": "<git rev-parse HEAD>",
  "input_sha256": {"<path>": "<sha256>", ...},
  "commands_run": ["..."],
  "results": {
    "confirmatory": {"P_HP1": {...}, "P_HP3_qwen": {...}, "holm": {...},
                      "payload_sha256": "..."},
    "replication":  {... same shape ...}
  },
  "anomalies": ["anything unexpected, or empty"]
}
```

plus a plain-text `N8_LEVEL1_REPORT.md` stating, in your own words and
without consulting any campaign document: what you ran, what the two
primary numbers came out to be, and anything that struck you as fragile
or ambiguous in the pipeline. Then stop. Do not compare against, or go
looking for, previously published values.
