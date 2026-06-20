# Lab 34 Final Validation Report

## Question

Does Lab 34 validate a controlled tool-use state signal, or only a superficial
tool-word detector?

## Final Read

Lab 34 is a narrow positive. The repaired full run uses 84 frozen toy-tool
tasks and shows that a prompt-boundary residual state predicts tool need and
tool choice above surface controls. It also shows a constrained causal effect:
activation addition shifts action-letter logits more than a random-direction
control.

The claim boundary is the point of the lab. The result is about a toy harness,
not autonomous agents. It supports controlled tool-state language, deterministic
trace validation, and a narrow intervention result; it does not support claims
about persistent goals, planning, real-world tool reliability, or faithful
self-report.

## Run Matrix

| Run | Model | Rows | Depth | Tool-needed AUC | Tool-selection accuracy | Surface control | Causal gap | Trace match | Read |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `olmo3_7b_full` | Olmo-3-1025-7B | 84 | 28 | 0.9236 | 0.7692 | 0.3077 | 0.1120 | 1.0 | Narrow positive |
| `gpt2_smoke` | GPT-2 | 28 | 3 | 1.0 | 0.4286 | 0.3571 | -0.0144 | 1.0 | Smoke-only decode check |

## Evidence Read

The primary evidence matrix has five rows:

- Tool-needed decode is supported above the surface-needed control.
- Tool-selection decode is supported above the surface-control baseline.
- Activation addition is supported only for the constrained action-letter
  prompt.
- Deterministic local tool traces match expected answers.
- Self-report labels require human review and are not introspection evidence.

The counterexamples are useful teaching material: surface rows can beat the
probe, no-tool rows with tool words can trigger false positives, and some random
directions match target shifts on individual rows. Those failures keep the
positive result scoped.

## Recommended Course Wording

```text
Lab 34 validates a toy-tool state handle under named controls. The model's
prompt-boundary residual state predicts tool need and tool choice, and a narrow
action-letter intervention shifts logits above a random control. The lab does
not show autonomous planning, persistent goals, or faithful self-report.
```
