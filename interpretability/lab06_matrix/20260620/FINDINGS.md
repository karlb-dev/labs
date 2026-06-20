# Lab 6 validation matrix — cross-model FINDINGS (20260620)

One question per (behavior, model, scope) cell: **does this model implement a clean, transferable circuit for this behavior — yes, no, or not-as-a-heads-only-graph?** A confirmed NO is a SUCCESS. Headline faithfulness is **resample (interchange) ablation**; mean ablation is shown only for the inflation comparison.

Models: `Olmo-3-1125-32B`, `gemma-4-E4B-it`, `gpt2`

Verdicts: YES=clean transferable circuit · OVERFIT=discovery only / no transferable subgraph · NOT-HO=needs MLPs (heads-only fails, heads+MLPs passes) · ABSENT=expected mechanism not present · INSUF=hygiene gate aborted (model can't do the task at n≥8) · ERR=load/run error.


## Executive summary

- **30 cells run across 3 models.** Verdict tally: OVERFIT / NO CLEAN CIRCUIT = 9; OVERFIT / NO TRANSFER = 9; INSUFFICIENT PROMPTS = 7; OVERFIT / FILLER (motif core insufficient) = 3; OVERFIT / OVER-RECOVERY = 2.
- **No cell yields a clean transferable small circuit** (CIRCUIT CONFIRMED = 0). Under honest resample (interchange) ablation with held-out transfer, every behavior is OVERFIT, an over-recovery (suppression) artifact, mechanism-absent, or the model cannot do the task at n>=8. A confirmed NO is the success condition of this lab.
- **Mean-ablation inflates faithfulness** (max discovery mean-minus-resample gap +1.40); several cells exceed 1.0 mean faithfulness, and resample reveals the honest, much lower (or over-recovering) picture.
- **The prev-token -> induction core is recoverable** where induction is testable: Olmo-3-1125-32B induction_p3: L23H13 -> L27H15; Olmo-3-1125-32B recall: L6H37 -> L25H33; gemma-4-E4B-it induction_p3: L22H6 -> L41H2; gpt2 induction_p3: L4H11 -> L5H1; gpt2 recall: L9H3 -> L10H0.
- **recall is MLP-mediated on every model** (heads_only with MLPs intact transfers better than heads_and_mlps with MLPs ablated); recall/induction knees are dominated by MLP nodes, so a heads-only routing graph structurally cannot represent these behaviors.
- **Instruct vs base:** Gemma-4-E4B-it (instruct) is baseline-negative on successor/ioi/agreement/taskvec in bare-prompt format, so those abort INSUFFICIENT -- a real finding about instruct bare-completion behavior.

## Results matrix (heads_and_mlps scope; cell = verdict / held-out resample F)

| behavior | Olmo-3-1125-32B | gemma-4-E4B-it | gpt2 |
|---|---|---|---|
| induction_p3 | OVERFIT / -0.10 | OVERFIT / +0.08 | OVERFIT(disc-only) / +0.12 |
| induction_p2 | OVERFIT / +0.28 | OVERFIT(disc-only) / +0.54 | INSUF / — |
| successor | OVERFIT / -0.09 | INSUF / — | OVERFIT(disc-only) / +0.40 |
| ioi | OVERFIT / +0.34 | INSUF / — | OVERFIT(disc-only) / +0.18 |
| agreement | OVERFIT(disc-only) / -0.23 | INSUF / — | OVERFIT / -0.33 |
| agreement_long | OVERFIT(disc-only) / -0.77 | INSUF / — | OVERFIT(disc-only) / +0.37 |
| taskvec | OVERFIT / +0.14 | INSUF / — | INSUF / — |
| recall | OVERFIT(disc-only) / +0.40 | OVER-RECOVERY / +1.90 | OVERFIT(filler) / +1.16 |

### heads_only contrast cells (for the NOT-HEADS-ONLY determination)

| behavior | model | heads_only held F (resample) | heads_and_mlps held F | MLPs in knee |
|---|---|---|---|---|
| induction_p3 | Olmo-3-1125-32B | +0.39 | -0.10 | MLP63, MLP62, MLP61, MLP22, MLP13, MLP11, MLP57, MLP28, MLP60, MLP23, MLP59, MLP29, MLP54, MLP36, MLP50, MLP41, MLP49, MLP40 |
| recall | Olmo-3-1125-32B | +1.02 | +0.40 | MLP50, MLP27, MLP46, MLP38, MLP4, MLP34, MLP31, MLP47, MLP29, MLP40, MLP32, MLP33, MLP35, MLP30, MLP59, MLP24, MLP37, MLP18, MLP26, MLP39, MLP25, MLP44, MLP9, MLP49, MLP45, MLP8, MLP51, MLP16, MLP42, MLP56, MLP58, MLP14 |
| induction_p3 | gemma-4-E4B-it | +0.36 | +0.08 | MLP22, MLP19, MLP10, MLP9, MLP20, MLP21, MLP14, MLP8, MLP15, MLP11, MLP12, MLP31 |
| recall | gemma-4-E4B-it | +2.25 | +1.90 | MLP15, MLP16, MLP18 |
| induction_p3 | gpt2 | +0.31 | +0.12 | MLP0, MLP4, MLP2, MLP1, MLP3, MLP10, MLP9, MLP5, MLP8, MLP11 |
| recall | gpt2 | +1.20 | +1.16 | MLP0, MLP8, MLP2, MLP9, MLP3, MLP7, MLP4, MLP6, MLP5, MLP10 |

## Per-behavior, cross-model reading

### induction_p3
- `Olmo-3-1125-32B`: **OVERFIT / NO CLEAN CIRCUIT** — held-out resample -0.10, discovery resample/mean -0.02/+0.36, mean−resample gap +0.38, motif-core held -0.19, suppression heads 1, MLPs in knee 18, edge yes: L23H13 -> L27H15.
- `gemma-4-E4B-it`: **OVERFIT / NO CLEAN CIRCUIT** — held-out resample +0.08, discovery resample/mean +0.29/+0.22, mean−resample gap -0.08, motif-core held -0.18, suppression heads 6, MLPs in knee 12, edge yes: L22H6 -> L41H2.
- `gpt2`: **OVERFIT / NO TRANSFER** — held-out resample +0.12, discovery resample/mean +0.13/+0.98, mean−resample gap +0.85, motif-core held -0.31, suppression heads 4, MLPs in knee 10, edge yes: L4H11 -> L5H1.

### induction_p2
- `Olmo-3-1125-32B`: **OVERFIT / NO CLEAN CIRCUIT** — held-out resample +0.28, discovery resample/mean +0.22/+0.36, mean−resample gap +0.14, motif-core held +0.23, suppression heads 0, MLPs in knee 19, edge none.
- `gemma-4-E4B-it`: **OVERFIT / NO TRANSFER** — held-out resample +0.54, discovery resample/mean +1.97/+3.11, mean−resample gap +1.14, motif-core held +1.00, suppression heads 1, MLPs in knee 5, edge none.
- `gpt2`: **INSUFFICIENT PROMPTS** — held-out resample —, discovery resample/mean —/—, mean−resample gap —, motif-core held —, suppression heads 0, MLPs in knee 0, edge none.

### successor
- `Olmo-3-1125-32B`: **OVERFIT / NO CLEAN CIRCUIT** — held-out resample -0.09, discovery resample/mean +0.02/+0.24, mean−resample gap +0.22, motif-core held -0.13, suppression heads 0, MLPs in knee 24, edge none.
- `gemma-4-E4B-it`: **INSUFFICIENT PROMPTS** — held-out resample —, discovery resample/mean —/—, mean−resample gap —, motif-core held —, suppression heads 0, MLPs in knee 0, edge none.
- `gpt2`: **OVERFIT / NO TRANSFER** — held-out resample +0.40, discovery resample/mean +0.51/+1.11, mean−resample gap +0.60, motif-core held -0.04, suppression heads 2, MLPs in knee 10, edge none.

### ioi
- `Olmo-3-1125-32B`: **OVERFIT / NO CLEAN CIRCUIT** — held-out resample +0.34, discovery resample/mean +0.18/+0.35, mean−resample gap +0.16, motif-core held +0.25, suppression heads 0, MLPs in knee 11, edge none.
- `gemma-4-E4B-it`: **INSUFFICIENT PROMPTS** — held-out resample —, discovery resample/mean —/—, mean−resample gap —, motif-core held —, suppression heads 0, MLPs in knee 0, edge none.
- `gpt2`: **OVERFIT / NO TRANSFER** — held-out resample +0.18, discovery resample/mean +1.19/+1.23, mean−resample gap +0.04, motif-core held -1.11, suppression heads 4, MLPs in knee 10, edge none.

### agreement
- `Olmo-3-1125-32B`: **OVERFIT / NO TRANSFER** — held-out resample -0.23, discovery resample/mean +0.04/+1.44, mean−resample gap +1.40, motif-core held -0.32, suppression heads 0, MLPs in knee 22, edge none.
- `gemma-4-E4B-it`: **INSUFFICIENT PROMPTS** — held-out resample —, discovery resample/mean —/—, mean−resample gap —, motif-core held —, suppression heads 0, MLPs in knee 0, edge none.
- `gpt2`: **OVERFIT / NO CLEAN CIRCUIT** — held-out resample -0.33, discovery resample/mean -0.08/+0.12, mean−resample gap +0.20, motif-core held -0.33, suppression heads 0, MLPs in knee 8, edge none.

### agreement_long
- `Olmo-3-1125-32B`: **OVERFIT / NO TRANSFER** — held-out resample -0.77, discovery resample/mean +0.06/+0.96, mean−resample gap +0.90, motif-core held -0.98, suppression heads 0, MLPs in knee 23, edge none.
- `gemma-4-E4B-it`: **INSUFFICIENT PROMPTS** — held-out resample —, discovery resample/mean —/—, mean−resample gap —, motif-core held —, suppression heads 0, MLPs in knee 0, edge none.
- `gpt2`: **OVERFIT / NO TRANSFER** — held-out resample +0.37, discovery resample/mean +0.53/+1.03, mean−resample gap +0.50, motif-core held -0.98, suppression heads 0, MLPs in knee 9, edge none.

### taskvec
- `Olmo-3-1125-32B`: **OVERFIT / NO CLEAN CIRCUIT** — held-out resample +0.14, discovery resample/mean +0.19/+0.63, mean−resample gap +0.44, motif-core held +0.06, suppression heads 1, MLPs in knee 27, edge none.
- `gemma-4-E4B-it`: **INSUFFICIENT PROMPTS** — held-out resample —, discovery resample/mean —/—, mean−resample gap —, motif-core held —, suppression heads 0, MLPs in knee 0, edge none.
- `gpt2`: **INSUFFICIENT PROMPTS** — held-out resample —, discovery resample/mean —/—, mean−resample gap —, motif-core held —, suppression heads 0, MLPs in knee 0, edge none.

### recall
- `Olmo-3-1125-32B`: **OVERFIT / NO TRANSFER** — held-out resample +0.40, discovery resample/mean +0.47/+1.30, mean−resample gap +0.82, motif-core held -0.03, suppression heads 0, MLPs in knee 32, edge yes: L6H37 -> L25H33.
- `gemma-4-E4B-it`: **OVERFIT / OVER-RECOVERY** — held-out resample +1.90, discovery resample/mean +1.13/+1.25, mean−resample gap +0.12, motif-core held +1.51, suppression heads 12, MLPs in knee 3, edge none.
- `gpt2`: **OVERFIT / FILLER (motif core insufficient)** — held-out resample +1.16, discovery resample/mean +0.91/+1.02, mean−resample gap +0.11, motif-core held +0.56, suppression heads 1, MLPs in knee 10, edge yes: L9H3 -> L10H0.

## Cross-cutting claims (falsifiable)

1. **prev-token→induction core universality.** Per-cell edge claims + `induction_motif_present` across induction_p3/p2/successor for each model establish whether the textbook core is recoverable everywhere; the knee-vs-floor and motif-core-vs-knee held-out numbers say whether the *surrounding* circuit is a pruning artifact.
2. **mean-ablation inflation.** The mean−resample gap column quantifies how much faithfulness was a mean-ablation artifact; positive gaps co-occurring with detected suppression heads support the brake-removal explanation (see each cell's brake-intact numbers in its card).
3. **not-heads-only.** The heads_only-vs-heads_and_mlps contrast rows + `MLPs in knee` identify behaviors that are not representable as a heads-only routing graph (expected for recall).
4. **successor is a non-induction mechanism.** successor cells with no claimed prev->induction edge and negative held-out resample are the worked negative.
5. **SWA long-context probe: deferred (documented, not faked).** Crossing Olmo-3's 4096-token sliding window needs attention-pattern capture at >4k tokens, which is memory-infeasible under eager attention (~343 GB for 64 layers x 40 heads at 4k). The prompt generator is in lab06 (`swa_prompts`); running it requires an attention-capture-free, causal-only screen, left for a follow-up.

## Negative / absent verdicts (each a successful result)

- `Olmo-3-1125-32B` induction_p3/heads_and_mlps: **OVERFIT / NO CLEAN CIRCUIT** — no transferable subgraph: knee held-out resample -0.10 < 0.70.
- `Olmo-3-1125-32B` induction_p2/heads_and_mlps: **OVERFIT / NO CLEAN CIRCUIT** — no transferable subgraph: knee held-out resample 0.28 < 0.70.
- `Olmo-3-1125-32B` successor/heads_and_mlps: **OVERFIT / NO CLEAN CIRCUIT** — no transferable subgraph: knee held-out resample -0.09 < 0.70.
- `Olmo-3-1125-32B` ioi/heads_and_mlps: **OVERFIT / NO CLEAN CIRCUIT** — no transferable subgraph: knee held-out resample 0.34 < 0.70.
- `Olmo-3-1125-32B` taskvec/heads_and_mlps: **OVERFIT / NO CLEAN CIRCUIT** — no transferable subgraph: knee held-out resample 0.14 < 0.70.
- `Olmo-3-1125-32B` induction_p3/heads_only: **OVERFIT / NO CLEAN CIRCUIT** — no transferable subgraph: knee held-out resample 0.39 < 0.70.
- `gemma-4-E4B-it` induction_p3/heads_and_mlps: **OVERFIT / NO CLEAN CIRCUIT** — no transferable subgraph: knee held-out resample 0.08 < 0.70.
- `gemma-4-E4B-it` successor/heads_and_mlps: **INSUFFICIENT PROMPTS** — only 5 baseline-positive discovery prompts survive at run length 8 (need >= 8); see prompt_hygiene_report.md. Refusing to produce a tiny-n card.
- `gemma-4-E4B-it` ioi/heads_and_mlps: **INSUFFICIENT PROMPTS** — only 0 baseline-positive discovery prompts survive at run length 13 (need >= 8); see prompt_hygiene_report.md. Refusing to produce a tiny-n card.
- `gemma-4-E4B-it` agreement/heads_and_mlps: **INSUFFICIENT PROMPTS** — only 0 baseline-positive discovery prompts survive at run length 5 (need >= 8); see prompt_hygiene_report.md. Refusing to produce a tiny-n card.
- `gemma-4-E4B-it` agreement_long/heads_and_mlps: **INSUFFICIENT PROMPTS** — only 1 baseline-positive discovery prompts survive at run length 8 (need >= 8); see prompt_hygiene_report.md. Refusing to produce a tiny-n card.
- `gemma-4-E4B-it` taskvec/heads_and_mlps: **INSUFFICIENT PROMPTS** — only 0 baseline-positive discovery prompts survive at run length 9 (need >= 8); see prompt_hygiene_report.md. Refusing to produce a tiny-n card.
- `gemma-4-E4B-it` induction_p3/heads_only: **OVERFIT / NO CLEAN CIRCUIT** — no transferable subgraph: knee held-out resample 0.36 < 0.70.
- `gpt2` induction_p2/heads_and_mlps: **INSUFFICIENT PROMPTS** — only 6 baseline-positive discovery prompts survive at run length 8 (need >= 8); see prompt_hygiene_report.md. Refusing to produce a tiny-n card.
- `gpt2` agreement/heads_and_mlps: **OVERFIT / NO CLEAN CIRCUIT** — no transferable subgraph: knee held-out resample -0.33 < 0.70.
- `gpt2` taskvec/heads_and_mlps: **INSUFFICIENT PROMPTS** — only 1 baseline-positive discovery prompts survive at run length 9 (need >= 8); see prompt_hygiene_report.md. Refusing to produce a tiny-n card.

## Do these models 'have circuits'?

For each cell the honest answer is one of: (a) a small faithful transferable subgraph exists (CONFIRMED), or (b) the behavior is smeared across heads+MLPs with heavy redundancy and self-repair, so the mean-ablation circuit collapses under resample and held-out transfer (OVERFIT), or (c) the expected mechanism is simply absent (ABSENT), or (d) the model cannot do the task at n≥8 (INSUF). The matrix above reports which one each cell landed on; both (a) and (b)/(c) are legitimate scientific answers.

_Synthesized by lab06_synthesize_findings.py from per-model matrix_results.json._
