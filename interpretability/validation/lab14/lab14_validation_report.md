# Lab 14 v2c Validation Report

## What Changed

- Replaced the 36-row answerability set with an 80-row frozen dataset across
  five families.
- Balanced answer keys across A-D separately for answerable and unanswerable
  rows.
- Rotated the unknown-style option across A-D for both labels.
- Length-matched the question distribution enough that prompt length and
  question length are no longer competitive answerability shortcuts.
- Made confident/hedged style statements content-matched so the style control
  differs only in confidence wording.
- Updated Lab 14 to group paired topics across train/eval, audit all
  unknown-option positions, and include the answer-letter baseline in the
  headline shortcut metric.

## Commands Run

```bash
python interpretability/data/make_certainty_calibration.py
python -m py_compile interpretability/interp_bench.py interpretability/labs/lab14_certainty_calibration.py interpretability/data/make_certainty_calibration.py
cd interpretability
python interp_bench.py --lab lab14 --tier a --prompt-set full --max-examples 0 --no-plots --run-name lab14_v2c_tiera_full_20260620
python interp_bench.py --lab lab14 --tier b --prompt-set full --max-examples 0 --run-name lab14_v2c_olmo3_7b_full_s0_plots_20260620
python interp_bench.py --lab lab14 --tier b --prompt-set full --max-examples 0 --seed 1 --no-plots --run-name lab14_v2c_olmo3_7b_full_s1_20260620
python interp_bench.py --lab lab14 --tier b --prompt-set full --max-examples 0 --seed 2 --no-plots --run-name lab14_v2c_olmo3_7b_full_s2_20260620
```

## Result

This is a mostly clean positive. The internal answerability direction validates
strongly on SmolLM and on all three OLMo seeds. Two of three OLMo seeds pass the
strict `usable_certainty_instrument` verdict. The remaining seed is not a data
shortcut failure; it is flagged because the hedging-style/self-report control
also tracks answerability.

## Main Caveat

Do not claim the saved direction is a pure subjective-certainty, honesty, or
belief instrument. The defensible claim is narrower: under matched fixed-choice
controls, these instruct models expose a reusable internal answerability signal,
but verbal/style confidence remains partly entangled with that signal.
