# Lab 25 Final Validation Report

## Question

Does Lab 25 find a self-report wire: a reliable coupling between a known
activation intervention and the model's own report of the hidden cause?

## Final Read

No. The repaired June 20, 2026 validation gives Lab 25 a fairer test, and the
result is a clean negative. The full Olmo-3-7B-Instruct run builds eight local
concept directions and runs 192 intervention trials, but target self-report
detection remains 0.0. The false-positive floor is also 0.0, so this is not a
case where controls expose a noisy positive; the report channel simply does not
move.

The lab is still valuable. It now teaches exactly the right capstone lesson:
mechanistic-looking handles, behavioral style changes, source-attribution
questions, and verbal confidence probes must be separated before making a
self-report claim.

## Run Matrix

| Run | Model | Mode | Items | Direction rows | Generation rows | Source rows | Verdict | Detection | Control floor | Grounding | Source accuracy |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `olmo3_7b_full` | Olmo-3-7B-Instruct | both | 24 | 8 | 192 | 60 | `not_run_or_no_detection` | 0.0 | 0.0 | 0.0 | 0.4667 |
| `olmo3_7b_source_rubric` | Olmo-3-7B-Instruct | attribution | 24 | 8 | 0 | 60 | `not_run_or_no_detection` | n/a | n/a | n/a | 0.4667 |
| `smollm_smoke_noleak` | SmolLM2-135M-Instruct | both smoke | 4 | 4 | 32 | 10 | `not_run_or_no_detection` | 0.0 | 0.0 | 0.0 | 0.0 |

## Evidence Read

The primary evidence matrix marks all concepts `not_supported`. Large
projection gaps do appear during direction construction, so the failure is not
just that no direction can be built. The failure is downstream: report
detection does not rise at the maximum dose, no dose-response slope appears,
and grounding checks do not support the stronger "wired" reading.

Source attribution remains weak. The final full run reaches 0.4667 aggregate
accuracy, but activation-injection rows are not identified as activation
caused. The source-rubric rerun reproduces that same 0.4667 attribution score.

## Recommended Course Wording

```text
Lab 25 is a negative capstone result. It builds and audits a self-report wire
instrument, but the repaired validation does not find a controlled report
channel for the injected activation cause. The lesson is claim discipline:
mechanistic handles are not the same thing as reliable self-report.
```
