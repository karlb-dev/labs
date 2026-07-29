# AMENDMENT 1 — tokenizer BOS units (2026-07-29, VM9, pre-outcome)

**What happened.** The first N6 cell (olmo31-think) halted on the §7
baseline-capability stop rule at its FIRST item
(`twohop:bird-color-swan`: manifest −1.6532 vs measured −3.1973;
`n6_state.json` shows `items done: 0`, `baseline_checked: 0` — **no
confirmatory outcome was viewed**).

**Root cause, verified by direct experiment.** `jlens.from_hf` defaults
to `force_bos=True`, which mutates the shared tokenizer
(`add_bos_token=True`). Scoring the same item in one process before and
after `from_hf` reproduces both numbers exactly (−1.6532 → −3.1973).
Consequently the campaign has two instrument-unit systems:

- **BOS units** — everything that touched jlens: lens fitting, every
  pilot grid, G4 swap controls, R2 occupancy captures.
- **native units** — the G5 capability scoring
  (`g5-capability-scores-*-v1`, manifest v4), which loaded the tokenizer
  directly.

The stop rule did its job: it refused to run an assay whose gate was
computed under a different instrument.

**Resolution (this amendment).** The confirmatory era adopts **BOS
units everywhere**, because the frozen lenses were fitted under them and
every pilot cell that motivates the hypotheses used them:

1. capability scores rescored per model with `add_bos_token=True`
   (`g5-capability-scores-{slug}-bos-v2`, superseding `-v1`);
2. cohorts reassembled with the SAME predicate (`capable_generation`)
   into `g5-item-manifest-v5`, superseding v4;
3. the N6 grid reads manifest v5; the stop-rule tolerance is unchanged;
4. **the frozen partition is untouched** — `d5-partition-freeze-v1`
   derives from ANCHOR difficulty metrics only (manifest v3 fields) and
   its family/item lists and hashes do not depend on the per-model
   capability scores, which are analysis-time cohort masks.

**What this does not change.** Endpoints, conditions, thresholds,
estimands, the partition, and the Holm family are all as frozen. No
intervention outcome was seen under either unit system before this
amendment.
