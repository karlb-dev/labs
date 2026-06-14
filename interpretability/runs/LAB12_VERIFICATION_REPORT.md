# Lab 12 verification report — relation geometry and method validation

Date: 2026-06-12 · Machine: MacBook (Apple Silicon, MPS, local venv) · Branch: `interpret_part2`

**Scope of this report: Tier A only.** Tier B (Olmo-3-1025-7B,
`--relation-set full`, Colab A100) is code-complete but **unverified** — run
`python interp_bench.py --lab lab12 --tier b --relation-set full` on Colab and
extend this report before treating any Tier B number as real.

## What was built

The first advanced lab: the intro toolkit (Lab 4 probes-with-controls, Lab 5
interchange patching) re-run on a scaled, confound-controlled relation
dataset. The load-bearing design choice is in the DATA, not the code: three
**relation-swap groups** (capital/language/continent over 27 shared
countries; opposite/comparative over 24 shared adjectives; month_after/
month_before over the 12 months) hold entities and template skeleton fixed
while only the relation word changes. Entity-class and template-syntax
explanations of a "relation direction" die by construction, and token-aligned
relation-swap patch pairs exist by construction.

New pieces:

- `data/make_advanced_relation_sets.py` → `data/advanced_relation_geometry.csv`
  (244 items, 12 families), dual-tokenizer verified (gpt2 + Olmo-3) at
  generation time, re-verified at runtime with a drop audit. Starts
  `data/MANIFEST.json` (sha256 pinning for advanced-course frozen CSVs).
- `labs/lab12_relation_geometry.py` + `.md`: margins (OBS) → role-position
  probes with subject-grouped splits (DECODE) → relation-direction cosine
  atlas → subject-swap and relation-swap patching with controls (CAUSAL) →
  transfer matrix → operationalization audit.
- Bench: `lab12` registry entry; `--relations`, `--relation-set`,
  `--patch-grid` flags. Dashboard, READMEs updated.

## Validation evidence (Tier A: gpt2, MPS, float32, relation-set small)

Wall clock: **13.8 s** lab time (~24 s end to end). 96 items, 0 tokenization
drops, 24/36 subject-swap and 18/30 relation-swap pairs passed the
behavioral gate.

| Check / metric | Value |
|---|---|
| hook parity / lens self-check / patch no-op | all OK |
| relword-role probe accuracy (any depth) | 1.00 — by design (token identity, the calibration trap) |
| subject-role within-group accuracy | chance at depth 0 → 1.00 by depth ~3 (relation info migrating onto the subject token) |
| within-group selectivity at best depth (final role) | country_sem 0.75, adj_morph 1.0, month_seq 1.0 (real − shuffled) |
| 12-way accuracy vs shuffled | 1.00 vs 0.04 |
| subject-swap patching (band 1..11) | mean 0.84, persists to depth 10/12; depth-0 sanity 1.0 |
| relation-swap patching at relation token | mean 0.52, persists to depth 5/12 |
| wrong-position control | 0.00 mean, 0.00 max |
| mismatched-vector control | 0.35 mean — margin destruction, not restoration (expected; see audit) |
| cross-family localization-profile correlation (band) | 0.95 |
| direction cosines (final role) | same-group 0.44 vs cross-group −0.13 |

The subject-vs-relation persistence asymmetry (10 vs 5) is the smoke path's
most interesting real result: on gpt2, relation identity leaves the
relation-word token by mid-stack while subject identity stays patchable at
the subject token until the late handoff.

## Honest negatives by design

- Whole families produced **zero gated patch pairs on gpt2**: continent_of,
  month_after, month_before, color_of (the model fails the behavioral
  baseline). Their probe rows still run; their causal cells are empty with
  drop reasons in `diagnostics/patch_pair_gate.csv`. Expected to fill in on
  the Tier B model.
- Relation-swap pairs survived the gate only in the country group at Tier A;
  the adj/month swap curves await Tier B.
- The transfer matrix has honestly empty cells for cross-group pairs — no
  token-aligned pair exists, and the lab does not fake them.
- The probe phase saturates (1.0) at Tier A item counts; the claim structure
  leans on selectivity vs controls, not raw accuracy.

## Design findings worth keeping

1. **"superlative" is 3 BPE tokens** under both course tokenizers, and book
   titles are almost never single tokens — the morphology swap group is a
   pair, and the outline's authorship family was dropped at authoring time
   (documented in the generator and handout as a frame limitation).
2. **Assign cyclic hard distractors AFTER single-token pruning**, or a
   multi-token answer kills its neighbor's item too (first generator draft
   lost 76 candidates; the fix loses 23).
3. **Depth-band discipline matters more here than in Lab 5**: depth-0
   relation-swap "recovery" of 1.0 is pure token substitution, and early
   drafts of the metrics happily reported it as a causal peak. All causal
   summaries now use band mean (depths 1..L−1) + persistence depth, with
   depth 0 kept as `depth0_sanity`.
4. **The mismatched-vector control recovering ~0.35 is signal, not failure**:
   stomping the corrupt evidence raises the margin without restoring clean
   content. The claim language puts the causal weight on the gap between
   matched and mismatched, restating Lab 5's lesson at scale.

## Files

- Code: `labs/lab12_relation_geometry.py`, `labs/lab12_relation_geometry.md`,
  `interp_bench.py` (registry + 3 flags), `course_dashboard.py`
- Data: `data/advanced_relation_geometry.csv`,
  `data/make_advanced_relation_sets.py`, `data/MANIFEST.json`,
  `data/README.md`
- Canonical Tier A run: `runs/lab12_relation_geometry-20260612_185534-76d886/`
  (local, not committed; regenerate with `--lab lab12 --tier a`)

## Pending

- [ ] Tier B on Colab A100: `--tier b --relation-set full`; check that the
      gated-out families fill in, adj/month relation-swap curves appear, and
      probe saturation breaks (or doesn't — report selectivity either way).
- [ ] Extend this report with the Tier B table and any convention surprises
      on the 33-layer stream indexing (none expected; the bench owns it).
