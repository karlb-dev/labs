# Frozen datasets

These CSVs are **vendored and frozen** (course rule: no live downloads at lab
runtime, no student-authored truth sets). They were generated deterministically
by `make_truth_sets.py` at authoring time; regenerating produces identical
files. If a statement turns out to be wrong about the world, fix it here, bump
the note below, and treat every prior run as a different dataset version.

| File | Family | Contents |
|---|---|---|
| `truth_cities.csv` | cities | "The city of X is in Y." — 30 true / 30 false |
| `truth_comparisons.csv` | comparisons | "Sixty-one is larger than fourteen." — 30 true / 30 false |
| `truth_negations.csv` | negations | "The city of X is not in Y." — 30 true / 30 false; labels are truth values, so surface form anti-correlates with the affirmative family |

Columns: `statement_id, family, statement, label (1=true), meta`.

Used by: Lab 4 (probing). The mass-mean truth direction Lab 4 saves from these
statements is reused causally by Lab 7.

Version note: v1, 2026-06-11.
