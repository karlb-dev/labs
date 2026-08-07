# UNSUPPORTED_NUMBER_REGISTER.md — published with the papers

Rows: numbers appearing in current paper sources that fail reconstruction or carry unlicensed wording. An empty register, honestly earned, is the goal; this one has **1 failed number + 3 wording/labeling corrections**, all in the pre-analysis draft `kburtram_jspace.tex`, all fixable at P8.

| # | Source | Quoted | Evidence found | Status | Required action |
|---|---|---|---|---|---|
| U1 | `kburtram_jspace.tex` §Prose (draft TODO block) | "span-safe removes 72–78% of the label cost" | registered reductions 0.493 (Think) / 0.716 (Instruct) / 0.778 (Qwen) — `prose_grid_figure_stats.json` | **failed** | rescope to 49–78% with per-model values, or per-model sentence |
| U2 | `kburtram_jspace.tex` §Composition | "wild-cluster CI [−0.52, −0.01]" for P3-P1 | that interval is the NORMAL approximation [−0.515271, −0.006637]; the percentile-t interval is [−0.537109, +0.015201] and crosses zero | mislabeled | quote the percentile-t (or randomization) interval with its name |
| U3 | `kburtram_jspace.tex` §Composition | "an honest near-miss" for P3-P1 | release-audit correction #1: P3-P1 receives NO inferential near-miss wording (control-seed sensitivity crosses 0.05) | banned wording | descriptive wording only: negative, seed-sensitive, MDE-disclosed |
| U4 | `kburtram_jspace.tex` header comment | source path `interpretability/jspace_runs/analysis/run_analysis.py` | script lives at `interpretability/jspaces/phases/paper_analysis/scripts/run_analysis.py` | stale pointer | fix path (stale register U8) |

Render-diff notes (no action beyond a repr rule): the Gemma 17-digit prose float `0.0024581113830208778` vs stored 16-digit `0.002458111383020878` (same float64); OLMo `checkpoint_estimates.csv` float rendering ≤1e-16 off its own JSON payload; H6 "0.02216"/"0.02530" are roundings of 0.02216238880688507 / 0.025295454600777884. Papers quote the stored full-precision reprs.
