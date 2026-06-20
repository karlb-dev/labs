# Lab 22 Final Validation Report

## Question

Does Lab 22 validate an eval-awareness circuit, or only a surface/template
detector for benchmark-like prompts?

## Final Read

Lab 22 now gives a useful partial positive. The repaired v2 data expands the
task to 96 rows across six matched formats, and the primary Olmo-3-7B-Instruct
run finds a strong residual eval-context direction. The result is strongest as
an audited decode claim: the direction separates eval framing from natural
phrasing and beats several cheap surface baselines.

The result should not be upgraded to causal or strategic eval-awareness
language. The minimum cross-format control-adjusted gap is slightly negative in
the primary seed, and activation steering does not move natural-prompt eval
markers over controls. The final course wording should therefore emphasize
claim discipline rather than a dramatic awareness result.

## Run Matrix

| Run | Model | Rows | Best depth | Eval AUC | Format-control AUC | Cross-format min/mean AUC | Surface baseline AUC | Decode verdict | Causal verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `olmo3_7b_full_s0` | Olmo-3-7B-Instruct | 96 | 9 | 0.9306 | 0.7917 | 0.9062 / 0.9635 | 0.7049 | cautious cross-format decode | not validated |
| `olmo3_7b_full_s1` | Olmo-3-7B-Instruct | 96 | 9 | 0.9861 | 0.9306 | 0.9062 / 0.9635 | 0.8681 | cautious cross-format decode | not validated |
| `olmo3_7b_think_full` | Olmo-3-7B-Think | 96 | 23 | 0.9722 | 0.9861 | 0.8438 / 0.9531 | 0.7049 | cautious cross-format decode | not validated |
| `smollm_smoke` | SmolLM2-135M-Instruct | 24 | 2 | 1.0 | 1.0 | 1.0 / 1.0 | 0.75 | smoke only | not validated |

## Evidence Read

The primary evidence matrix passes the rendered-prompt measurement-site check,
the held-out eval-vs-natural decode check, the eval-vs-format-control check,
and the surface-baseline challenge. It fails the strongest cross-format claim
and the benign activation-steering marker test.

That combination is exactly what the lab should teach. The direction is real
enough to study, but the strongest stories are held back by controls.

## Recommended Course Wording

```text
Lab 22 validates an audited eval-context direction, not a broad claim about
deception or situational awareness. The direction decodes eval-like framing
under matched controls, but current steering and cross-format controls keep the
claim cautious.
```
