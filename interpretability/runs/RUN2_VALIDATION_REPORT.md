# Run 2 — full-course validation report

Date: 2026-06-12 · Machine: Colab A100-SXM4-80GB · Branch: `lab1_colab` @ post-Lab-11
Scope: **every lab, both tiers, in one sweep** — Tier A (smoke) for labs 1–11,
then Tier B at each lab's documented full settings (labs 1–6 `--prompt-set
full`, lab 2/3 `--topk 10`, lab 5 `--run-edit`, lab 11 both audit domains).

## Verdict

**24/24 runs green.** Zero failed self-checks across every diagnostics
directory; zero tracebacks; every handout-named artifact produced by the lab
that owns it. Tier B wall-clock ~75 minutes end to end.

| Lab | Tier A | Tier B | Tier B headline (run 2) | vs run 1 |
|---|---|---|---|---|
| 1 logit lens | 4.6s | 10.6s | category decision-depth profile intact | consistent |
| 2 DLA | 3.5s | 9.3s | decomposition check 0 fail; attribution table written | consistent |
| 3 attention | 4.0s | 18.2s | head decomposition + eager patterns OK | consistent |
| 4 probing | 10.5s | 47.7s | truth direction saved; family-transfer gaps as designed | consistent |
| 5 patching | 9.0s | 202s | localization depths 19–21; wrong-position control 0.18, mismatched 0.29; edit runs and FAILS at α=1 (the Hase tension, on schedule) | consistent |
| 6 circuits | 30.9s | 685s | circuit card written; minimality worst marginal +0.043 | consistent |
| 7 steering | 307s | 386s | bridge verdict `decodable-and-steers-True-assent`; benign baseline refusal 0.0 | consistent |
| 8 SAEs | 78.4s | 154s | FVU 0.3736 / L0 113.5 / transcoder 0.457 / clamp CAUSAL / 1 survived 16 killed | **matches run 1 to 3 decimals** |
| 9 graphs | 5.4s | 6.2s | suppress −1.98, substitute −5.01, control +3.02, signed feat +2.34 | identical |
| 10 CoT | 106s | 2164s | flips 0.14–0.18, necessity 0.56→0.88, filler at floor, mistake-immune | identical |
| 11 audit | 1.2s | 7s + 815s | factual: preference 1.0 / top-1 0.36, recovery 0.995/0.022, monitor AUC 1.0; cot: flip 0.50 on fresh slice, probe negative replicates (AUC 0.378) | identical |

Determinism note: greedy decoding + seeded everything means labs 9–11
reproduce *exactly*; labs 1–8 match their verification-report values to
reporting precision.

## Issues found and fixed in this pass

1. **Lab 11 (cot domain) promised a diagnostic it never wrote.** The handout
   tells students to compare `decoding_pins.json` across the Lab 10 and Lab
   11 runs; only Lab 10 wrote one. The audit now records its own pins
   (including the fresh-slice definition).
2. **`data/README.md` didn't cover `mcq_items.csv`** (Lab 10/11). Added,
   including the offset-slice rule that keeps the capstone out-of-sample.
3. **COURSE.md had no as-built record.** Added section 0: a design-vs-built
   table (bench instead of interpkit/TransformerLens; ungated dictionaries;
   hand-built Lab 9 on gpt2; Qwen3-0.6B think smoke; two implemented audit
   domains), with pointers to the per-lab reports.

## Known smoke-tier caveat (documented, not fixed)

Lab 5's Tier A wrong-position control can read high (≈0.97) on gpt2 with the
4–6-fact smoke set: the weak model + tiny n make the control distribution
meaningless there. Tier A is the plumbing tier by contract ("correctness of
plumbing, not science" — README tier table); the Tier B control is 0.18.

## Consistency check (handouts vs artifacts)

Automated cross-reference of every backtick-named artifact and CLI flag in
each `labNN_*.md` against the run-2 output trees and `--help`. All clear.
Remaining flags are deliberate cross-lab references: labs 7/8 name Lab 4's
`truth_direction.pt`, lab 9 names Lab 6's `circuit_card.md`.

## Archive

Drive layout per lab (`My Drive/interpret/labN/`): `run1/` = pre-existing
canonical runs (moved from `runs/`), `run2/` = this sweep's runs, `code/` =
the validated tree at run-2 time. This report lives at the repo root of
`runs/` and in `interpret/` on Drive. Run 2 used the branch as pushed
(fadde70 + this pass's fixes); the planned "final" sweep after the pending
local changes merge should follow this same protocol.
