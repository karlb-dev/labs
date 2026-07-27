# J-space Part 2 — from exploratory replication to confirmatory study

> **Status: ADDENDUM-GOVERNED as of 2026-07-27.** A forensic external
> review ([`code/jspace_part2_plan1_addendum.md`](code/jspace_part2_plan1_addendum.md))
> found that Part 1, while an unusually strong *exploratory* campaign, did
> not implement the paper's actual intervention (missing clean-output
> protection), did not compute the paper's occupancy estimand, and lacked
> paired statistics and geometry-matched controls on its marquee effect.
> Part 2 therefore runs **Workstream R (assay repair + conformance) before
> any model matrix**. Part-1 claim corrections:
> [`../REPORT_v2_ERRATA.md`](../REPORT_v2_ERRATA.md).

Governing docs, in order:
[`code/jspace_part2_plan1_addendum.md`](code/jspace_part2_plan1_addendum.md) ·
[`code/REPAIR_PREREGISTRATION.md`](code/REPAIR_PREREGISTRATION.md) ·
[`code/PLAN_PART2.md`](code/PLAN_PART2.md) (REVISION 1) ·
[`code/preregistration.md`](code/preregistration.md) (pre-data commit,
superseded for confirmatory use) ·
ops: [`code/experiment_reset_instructions.md`](code/experiment_reset_instructions.md) ·
Part-1 record: [`../report/REPORT_v2.md`](../report/REPORT_v2.md) + errata.

## The question

Does the paper's global-workspace signature — a small, mid-band,
broadcast, **causally privileged** verbalizable subspace whose
output-protected dynamic ablation dissociates multi-step reasoning from
fluency — exist in open models, once measured with the paper's own
intervention, the paper's own capacity estimand, and controls matched for
energy, effective rank, and geometry?

| hypothesis | primary discriminator |
|---|---|
| **H0a/b/c instruments-first** (promoted from old H5) | R1 protected ablation reproduces released positive controls; R2 estimators stable across solver/fit; Part-1 proxies re-audited (R7) |
| H1a learned externalization | OLMo lineage: base → SFT → DPO → {3.0/3.1-Think} vs {3.1-Instruct} |
| H1b inference-mode externalization | Qwen3.6-27B same weights, official thinking on/off |
| H1c extra-compute | filler / shuffled-rationale / right/wrong-rationale rescue arms |
| H2 output alignment | output-alignment family + protected-vs-unprotected ablation delta |
| H3 → bounded residual | whatever the tested public axes leave standing (never "lab magic") |
| H4 scale | descriptive moderator via paper-defined occupancy only |
| H6 task demand / shortcuts | parametric load × ablation at matched baseline difficulty; dual-task |
| H7 mean-J mismatch | local/future-only Jacobians vs mean-J on a subset |
| H8 sparse-frame geometry artifact | J-rotated / spectrum-matched / label-shuffled controls |

## Model matrix (pins hub-checked 2026-07-27; lineage-verified)

| slug | model | role |
|---|---|---|
| olmo31-think | `allenai/Olmo-3.1-32B-Think` | **primary Think endpoint** (continues `Olmo-3-32B-Think-DPO`) |
| olmo31-instruct | `allenai/Olmo-3.1-32B-Instruct` | **primary Instruct endpoint** (via `-SFT`/`-DPO`, public) |
| olmo3-base | `allenai/Olmo-3-1125-32B` | shared base anchor of both branches |
| olmo3-think | `allenai/Olmo-3-32B-Think` | part-1 anchor → historical replication cell + lineage member (`-SFT`/`-DPO` public) |
| qwen36-27b | `Qwen/Qwen3.6-27B` | within-checkpoint official thinking-on/off contrast (H1b) |
| gemma4-31b | `google/gemma-4-31B-it` | architecture family, after OLMo primary cells bank |

The OLMo lineage is a **matched-lineage natural experiment** (shared base,
divergent post-training branches), not a one-variable controlled
intervention — wording binding per addendum §7.3.

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
