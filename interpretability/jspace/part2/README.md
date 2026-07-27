# J-space Part 2 — the model-matrix campaign

> **Status: BOOTSTRAPPED 2026-07-27 — Workstream A running.** Part 1 (Lab 37,
> merged via PR #9) established the instruments and the two-model result;
> Part 2 turns it into a publishable finding: either the paper's static
> causal dissociation is found on an open model, or every principled hiding
> place for it is measured shut.

Campaign plan: [`code/PLAN_PART2.md`](code/PLAN_PART2.md) ·
Preregistration (H1–H5, predictions, decision rules — committed before any
data): [`code/preregistration.md`](code/preregistration.md) ·
Session ops: [`code/experiment_reset_instructions.md`](code/experiment_reset_instructions.md) ·
Part-1 verdict: [`../report/REPORT_v2.md`](../report/REPORT_v2.md)

## The question

Part 1 ended with: static J-span ablation null on both OLMo-3-32B-Think and
Qwen3.6-27B under energy-matched instruments; frozen per-item J-ablation a
real, transferable causal handle on retrieved content; a 10× cross-model
capacity difference; and a foil-calibrated 46-step workspace-ahead-of-text
lead. Part 2 adjudicates **why the paper's dissociation is missing**:

| hypothesis | discriminator |
|---|---|
| H1 externalization (think-training moved the workspace into tokens) | Workstream A: matched-pretraining instruct sibling (`Olmo-3.1-32B-Instruct`), Qwen matrix completion, Gemma-4-31B |
| H2 occupancy (live J-space ≈ the output stream itself) | Workstream D: output-occupancy index across the matrix |
| H3 training-lab specifics | unfalsifiable; shrunk by exhausting the rest |
| H4 scale | untestable at our tier (stated) |
| H5 instruments | residuals: task battery (C), dose/selection (B4), fit-size/corpus (B1/B2), frozen-logit control (B3) |

## Model matrix (pins hub-checked 2026-07-27)

| slug | model | role |
|---|---|---|
| olmo3-think | `allenai/Olmo-3-32B-Think` | part-1 anchor + lens donor |
| olmo31-instruct | `allenai/Olmo-3.1-32B-Instruct` | matched-pretraining non-think (same base `Olmo-3-1125-32B`) — the H1 discriminator |
| qwen36-27b | `Qwen/Qwen3.6-27B` | fat-workspace model; think-mode + hard-one-hop completion |
| gemma4-31b | `google/gemma-4-31B-it` | third architecture family (adaptation gate first) |
| olmo3-base | `allenai/Olmo-3-1125-32B` | optional pretraining-only anchor |

## Layout

```
code/            sl2_common.py + scripts/ (p2*.py) + plan/prereg/ops docs
results/         per-model metrics JSONs (<5MB), mirrored from the Drive run dir
figures/         p2f*.png, regenerable from metrics alone via code/scripts/p2fig.py
report/          REPORT_PART2.md + matrix_master.{csv,json} (the master dataset)
handout/         living LaTeX writeup (tex+pdf)
```

Heavy artifacts (lenses, traces, layer states) live on Drive:
`MyDrive/interpret/special-lab-1/part2_20260727/`.

## Claim ledger (SL2 — grows per workstream)

*(none yet — A0 is the first cell)*
