# Lab 27 - Path-Specific Patching and Causal Mediation

```text
Time estimate: 5-15 minutes Tier A; longer on full Tier B grids
Compute tier: base model, hook-heavy, no chat template
Dependencies: Labs 3, 5, 6, 12, and 26
Minimum passing artifacts: method_card.md, tables/path_evidence_matrix.csv, tables/path_specificity_controls.csv
Main plot: plots/path_mediation_dashboard.png
Main table: tables/path_evidence_matrix.csv
Evidence rung: CAUSAL, scoped
Forbidden claim: A causes B in every context.
One-sentence allowed claim: This residual two-site path proxy recovered behavior above controls on this prompt family.
Human-label requirement: none
```

## Question

Which routes through the network carry a behavior, and how do route effects
differ from ordinary node importance?

Labs 5 and 6 patch sites. Lab 27 asks a stricter question: when a source site
matters and a receiver site matters, does their directed combination explain
more than the two node effects separately?

This first implementation is a residual two-site mediation proxy. It does not
claim to isolate an exact attention-head edge. That stronger implementation is
future work.

## Run

```bash
python interp_bench.py --lab lab27 --tier a --no-plots
python interp_bench.py --lab lab27 --tier b --prompt-set full
```

## What Happens

The lab loads `data/path_mediation_tasks.csv`, validates clean/corrupt token
alignment, then computes:

1. source-node residual patch recovery;
2. receiver-node residual patch recovery;
3. joint source+receiver patch recovery;
4. reverse-path, wrong-receiver, and random-source controls;
5. mediation accounting:

```text
interaction_residual = joint_effect - source_effect - receiver_effect
path_proxy = joint_effect - max(source_effect, receiver_effect)
```

## Main Artifacts

| Path | Meaning |
|---|---|
| `method_card.md` | scope and verdict table |
| `operationalization_audit.md` | why this is only a path proxy |
| `tables/node_effect_baseline.csv` | ordinary node patching effects |
| `tables/path_patch_report.csv` | two-site patch rows |
| `tables/path_specificity_controls.csv` | reverse/wrong/random controls |
| `tables/mediation_accounting.csv` | source, receiver, joint, interaction rows |
| `tables/path_evidence_matrix.csv` | domain-level claim matrix |
| `tables/path_counterexamples.csv` | rows that defeat path language |
| `state/path_candidates.json` | positions, depths, and candidate path specs |
| `plots/path_mediation_dashboard.png` | first plot to read |

## Claim Discipline

Allowed:

```text
CAUSAL: In this prompt family, the residual source+receiver path proxy recovered X behavior above reverse/wrong/random controls.
```

Forbidden:

```text
This proves the exact internal edge from A to B.
```

If controls match the joint patch, write a node-effect claim instead of a path
claim.
