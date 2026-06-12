# Run 3 — full-course validation report (post-merge)

Date: 2026-06-12 · Machine: Colab A100-SXM4-80GB · Branch: `lab1_colab`
Tree under test: merge of the major lab rewrite `9a49772` ("Prepare
interpretability lab updates": labs 1–4, 8–11, +8124/−3169 lines, authored
without local torch) with the run-2 fixes, plus fixes from this pass.
Protocol: identical to run 2 — Tier A sweep (11 labs), then Tier B at
documented full settings (12 runs incl. both Lab 11 domains).

## Verdict

**23/23 final runs green** (plus one Tier A failure that was a real bug,
fixed and re-run — see below). All 26 Tier B self-checks pass. The one-page
sweep view is `runs/course_dashboard_run3.{md,png}`.

## The regression matrix

| Slice | Expectation | Result |
|---|---|---|
| labs 5–7 (code untouched by the rewrite) | reproduce run 2 exactly | **exact**: lab 5 top-patch recovery 0.6693 = 0.6693; lab 6 minimality 0.04327 =; lab 7 layer/verdict/refusal-rate identical |
| labs 1–4, 8, 9 (rewritten) | all self-checks pass; headline science unchanged | pass; lab 9 interventions identical (suppress −1.98, control +3.02; substitution −5.26 vs −5.01, donor-selection refinement) |
| lab 10 (rewritten + dataset v2) | sane rates; new controls behave | flips 0.25–0.46 — **higher than run 2 (0.14–0.18) because the rewrite's stratified item selection draws a different 36 items**; consistent with the item-set sensitivity already documented in run 2 (lab 11's fresh slice read 0.50). New clean-resume control: 0.875 vs add-mistake follow 0.062 — the seam is harmless, the wrong claim is not. Forced-answer count down 46→27 |
| lab 11 (rewritten) | audit contract intact; new controls behave | factual: subject-site recovery 0.994 with the NEW unrelated-patch control at 0.286 (recovery now has its shadow); monitor AUC 1.0. CoT: flips 0.444/silent 0.111 on the fresh slice; the improved depth-selected, unit-normalized hint probe reads held-out AUC 0.891 — **against a shuffled control of 0.805**, and the lab correctly ships it as a NEGATIVE (selectivity 0.086) |

## Bugs found and fixed in this pass

1. **Duplicate MCQ item ids (real data bug, caught by the rewrite).** Lab
   10's new dataset validator aborted on `mcq_items.csv`: the v1 generator
   truncated subject names to 12 chars, collapsing both `high_school_*`
   subjects into duplicate ids. Generator fixed, dataset bumped to v2 (same
   questions, new ids — cross-version joins must map by question text).
   This is the rewrite's validation working exactly as intended.
2. **Dashboard metric paths** updated to the rewrite's renamed keys
   (`target_preference_accuracy`, `mean_recovery_unrelated_control`, …);
   the extractor's multi-candidate paths absorbed the rest of the drift.

## Improvements added in this pass

- **`course_dashboard.py`** — one-page state for the whole sequence (pure
  reader over `runs/`): per-lab self-check health re-verified from
  diagnostics, schema-drift-tolerant headline metrics, evidence rung,
  model, wall-clock; markdown + PNG. Rendered for this sweep at
  `runs/course_dashboard_run3.{md,png}`.
- **Lab 10 handout: "The thinking budget is a variable, not a constant"** —
  run 2's measured 2–3× flip-rate sensitivity to the thinking budget,
  with the two operational rules (compare rates only at matched
  `decoding_pins.json`; treat a high forced-answer rate as a truncation
  confound, not hygiene).
- Run 2 had already added the COURSE.md as-built addendum (§0), the
  `mcq_items.csv` data-README section, and the Lab 11 pins record (the
  last superseded by the rewrite's fresh-slice manifest).

## Notes for the next pass

- Lab 10's flip rates are **item-set dependent** (0.14–0.46 across three
  36-item draws at matched budget). The handout's scope line covers this;
  anyone summarizing the course should quote a range, not a point.
- The hint-presence probe has now been null twice (different
  implementations, different slices): behavioral hint influence without a
  cleanly decodable single-site trace is looking like a robust property of
  this model/task, not an artifact. Worth a sentence in the Lab 11 handout
  if it survives a third slice.
- Wall-clock for the full Tier B sweep: ~84 min; lab 10 (38 min) and the
  lab 11 CoT audit (19 min) dominate.

## Archive

Drive `My Drive/interpret/`: `labN/run3/` (this sweep, both tiers),
`run3_code/` (the exact merged tree), `RUN3_VALIDATION_REPORT.md` and
`course_dashboard_run3.{md,png}` at the root, alongside the `run1`/`run2`
history and `run2_code/`.
