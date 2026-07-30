# 2 · Assay repair

This section is a scientific contribution in its own right (nextsteps
§18.2): every repair below exists because an instrument fault produced,
or would have produced, a false claim. Each carries its registry trail.

## 2.1 Output protection done properly

The paper's protection contract is implemented as a *per-position
prefill* exclusion: at each position, token rows currently in the
model's clean top-k are ineligible for J-selection
(`protected_dynamic_v2`; row-wise dose accounting; golden-tested on a
real transformer against the frozen v1 behavior). Phase 2 carried an
explicit open question forward at closure: label exclusion bounds
*selection*, not *geometry* — the selected span can still overlap the
protected directions [`p2-label-vs-span-protection-openq-v1`]. Phase 3
measures that overlap directly and runs a span-safe arm; its
development-tier audits (three models, 60 frozen items each) show the
overlap is real and model-dependent, which is why `meanJ_span_safe`
enters the Phase 3 primary family (P3-P2). No Phase 2 conclusion is
edited by this; the Phase 2 record states the caveat.

## 2.2 Full-sequence scoring in one unit system

Two instrument-unit faults were caught by the preregistered §7 stop
rule before any confirmatory outcome was viewed, and resolved as
Amendment 1 [`AMENDMENT_1_BOS_UNITS`]:

- `jlens.from_hf(force_bos=True)` mutates the shared tokenizer, so
  lens fits and pilot grids ran in BOS units while capability scoring
  ran native. Resolution: assay-wide BOS units; every capability score
  regenerated (`g5-capability-scores-*-bos-v2`); cohorts reassembled
  (`g5-item-manifest-v5`); the frozen partition untouched (it derives
  from anchor difficulty only).
- The gate scored answers by piecewise concatenation of un-rstripped
  prompt ids and alias ids; the grid used rstrip+string concatenation.
  For 5/325 items the two tokenize differently. Resolution: the gate
  convention everywhere; the artifact cancels in paired deltas.

The stop rule halting on its *first item*, twice, is the strongest
process evidence in the campaign: assays this sensitive to conventions
require conventions this frozen.

## 2.3 Audited families, not string-split labels

The pilot's family field was a string accident (`name.split("-")[0]`),
making four models' intra-family correlation look uniform (0.37–0.42)
when the audited map gives 0.17–0.75. The 120-item bank was hand-audited
into canonical families (N1.5); every clustered statistic, power
calculation, and the confirmatory partition run on the audited map. One
pilot conclusion flipped under the audit (Instruct one-hop CI), which is
why family identity is a *frozen artifact* in this campaign, not a
convenience.

## 2.4 The exact instantaneous rank-and-energy matched control

The primary specificity comparison is against
`dyn_energy_rank_matched_random`: per (item, layer, position), a random
subspace matching the J arm's *achieved* effective rank and removed
energy exactly, orthogonal to the protected rows, deterministic seeds
[`mc-dev-validation-olmo31-think-v2`; gate v1 was itself superseded for
applying a relative-energy bound below the float32 measurement floor —
the gate discipline applies to gates too]. At matched dose, direction
content is the only difference between arms. Dev observation carried
forward: the J-specific heavy tail survives exact dose matching; the
control sits near zero.

Known scope limits, stated rather than hidden: rank+energy matching
does not match *all* mechanistically relevant structure (§4.5), the
control is *instantaneous* (per-position; temporal coherence is a
separate axis measured in Phase 3), and Phase 2 skipped this control on
prose — repaired in Phase 3's Workstream C before any selectivity
wording about Instruct is used (PI resolution R4).

## 2.5 Positive controls per model

A lens that cannot *cause* anything cannot have its ablations
interpreted. G4 swap controls certify each model's lens causally before
its confirmatory cells count: injecting the J-read direction of a
*different* answer flips the model's output at rate 0.76/0.76/0.78
(Think/Instruct/Qwen, own lenses) against 0.18–0.24 for random
injection and ≤0.04 for none [`r5-swap-positive-control-*-v2`].

## 2.6 Preregistration with teeth

The Phase 2 freeze (`jspace-part2-confirmatory-freeze-v1`,
`d5-partition-freeze-v1`: 32 confirmatory / 32 replication families,
disjoint, hash-pinned before any outcome was viewed) bound endpoints,
thresholds, the Holm family, and the analysis path. The locked analysis
ran once, from raw parquets, after the last cell banked
[`n6-confirmatory-analysis-v2`]. The one analysis defect found afterward
(a label bug in v1) was superseded, not edited. A partition-seed
clarification was registered at Phase 2 close: the recorded seed was
inactive in the code path; assignment was deterministic by family size
and name; no outcome entered the split
[`p2-partition-seed-clarification-v1`].

## 2.7 Reproduction as a gate, not a gesture

A narrative-blind agent in a clean worktree, given only a protocol file
(commands, schemas, tolerances — no expected values), regenerated the
full Phase 2 primary table from frozen parquets: all four primary
quantities match to ≤5e-4 [`p3-n8-level1-repro-v1`, tier: methods].
Model-cell sentinel reproduction (Level 2) and one full GPU cell
(Level 3) are public-release gates, deliberately still open.
