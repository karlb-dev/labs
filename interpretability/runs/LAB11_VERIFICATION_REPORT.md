# Lab 11 verification report — mechanistic reliability audit (capstone)

Date: 2026-06-11 · Machine: Colab A100-SXM4-80GB · Branch: `lab1_colab`

## What was built

The capstone: a plug-in audit harness with a RIGID output contract
(audit_report.md fixed schema, ledger_reconciliation.md worksheet,
safety_case_and_rebuttal.md, per-example results.csv with hand-label
columns), built on the claim ledger. The harness assembles every measured
number and cites it by artifact; sections marked `[STUDENT — graded]` are
prompts for the coursework — the claim, the failure-mode labels, the
keep/revise/retire verdicts, the safety case, and the rebuttal are
deliberately not generated.

Two domains implemented end to end, selected by `--audit-domain`:

- **factual_qa** (default): Lab 5's capital facts × 3 templates on the
  course base model. Per example: answer + confidence proxies, logit-lens
  stabilization **and preference** depths, a DLA layer summary (Lab 2's
  frozen-norm linearization), two-site residual patching (early subject /
  final position at the preference band — Lab 5's localization lesson,
  applied), and a truth-direction monitor with a shuffled-label control.
- **cot_faithfulness** (flagship): Lab 10's machinery rerun verbatim on a
  FRESH item slice (the harness offsets the sampling stride on purpose) —
  making the audit a replication — plus the mechanistic method Lab 10 left
  as its ambitious extension: a hint-presence probe at the answer-emission
  position with a shuffled control.

## Validation evidence (Tier B)

### factual_qa — Olmo-3-1025-7B, 12 facts × 3 templates

| Result | Value |
|---|---|
| preference accuracy (target beats distractor) | **1.000** |
| top-1 accuracy | 0.361 — the gap is phrasing ("…is **known** as"), not knowledge; auto-labeled `format_not_knowledge` |
| paraphrase preference-consistency | 1.000 |
| median preference-stabilization depth | 17 of 32 |
| causal recovery: early subject site | **0.995** |
| causal recovery: final position @ band | 0.022 |
| truth monitor (mass-mean, held-out facts) | AUC 1.000 vs shuffled 0.000 |

The top-1 vs preference split is the audit's first teaching moment (which
behavioral metric does your claim name?), and the two-site patching
reproduces Lab 5's localization: recall lives at the subject token early;
a single final-position patch at the band carries almost nothing.

### cot_faithfulness — Olmo-3-7B-Think, 12 fresh items (14 min)

| Result | Value |
|---|---|
| baseline accuracy (fresh slice) | 0.75 |
| max flip rate / max silent flip rate | 0.50 / 0.125 — HIGHER than Lab 10's 36-item slice (0.18/0.045): replicates direction, shows item-set sensitivity — exactly what an out-of-sample audit exists to surface |
| necessity curve (k=0 → k=100) | 0.167 → 1.000; filler stuck at 0.167 |
| injected-mistake follow rate | 0.167 (recovered 0.833) |
| hint-presence probe @ answer emission | **negative**: AUC 0.378 vs shuffled 0.398 |

The probe null is shipped as a NEGATIVE `DECODE` claim (the claim builder is
conditional on selectivity): behavioral hint influence with no decodable
trace at this site and sample size, underpowered-n named as the first
suspect. Per the course rubric, an explained negative earns full credit —
and the capstone's own machinery now models that behavior.

Tier A (gpt2, factual_qa): preference accuracy 1.0 vs top-1 0.167,
preference depth 6, recovery 0.989 (early subject) / 0.019 (final band),
monitor AUC 0.889 vs shuffled 0.111. Runs in seconds.

## Design findings worth keeping

1. **Strict top-1 "accuracy" buried the behavior.** First Tier B run scored
   0.361 and produced an empty causal table because the gate required top-1
   exactness while Olmo prefers "…is known as Paris". The audit now tracks
   `top1_accuracy` and `preference_accuracy` separately, gates the causal
   pass on preference (Lab 5's definition), and auto-drafts the failure mode
   `format_not_knowledge` — the bug became the domain's first lesson.
2. **Single-site patching assumes its own answer.** The causal subset tests
   the early subject site AND the final-position band site; on both models
   the subject site carries ~1.0 recovery and the final site ~0.02.
3. **Claims must be conditional on their own evidence.** The hint-presence
   probe claim asserts decodability only above a selectivity threshold;
   below it, it ships the negative with falsifier "a larger item set or
   different layer yields selective decodability — retire this negative."

## Files

- `labs/lab11_reliability_audit.py`, `labs/lab11_reliability_audit.md`
- registry entry `lab11` + `--audit-domain` flag in `interp_bench.py`
- canonical runs: `runs/lab11_tiera`, `runs/lab11_tierb_factual`,
  `runs/lab11_tierb_cot`
