# Lab 6 validation matrix — `google/gemma-4-E4B-it`

Matrix `lab06_matrix_20260620`. One question per cell: does this model implement a clean, transferable circuit for this behavior — yes, no, or not-as-a-heads-only-graph? **A confirmed NO is a success.** Headline faithfulness is RESAMPLE (interchange) ablation; mean ablation is shown for the inflation comparison.

## Results matrix

| behavior | scope | verdict | held-out F (resample) | disc F (resample / mean) | motif-core held F | knee/floor nodes | mean−resample gap |
|---|---|---|---|---|---|---|---|
| agreement | heads_and_mlps | **INSUFFICIENT PROMPTS** | — | — / — | — | None/None | — |
| agreement_long | heads_and_mlps | **INSUFFICIENT PROMPTS** | — | — / — | — | None/None | — |
| induction_p2 | heads_and_mlps | **OVERFIT / NO CLEAN CIRCUIT** | +0.54 | +1.97 / +3.11 | +1.00 | 12/1 | +1.14 |
| induction_p3 | heads_and_mlps | **OVERFIT / NO CLEAN CIRCUIT** | +0.08 | +0.29 / +0.22 | -0.18 | 15/19 | -0.08 |
| induction_p3 | heads_only | **OVERFIT / NO CLEAN CIRCUIT** | +0.36 | +0.17 / +0.28 | +0.62 | 2/2 | +0.12 |
| ioi | heads_and_mlps | **INSUFFICIENT PROMPTS** | — | — / — | — | None/None | — |
| recall | heads_and_mlps | **OVERFIT / NO CLEAN CIRCUIT** | +2.50 | +1.67 / +1.90 | +2.19 | 6/1 | +0.22 |
| recall | heads_only | **CIRCUIT CONFIRMED** | +2.03 | +1.30 / +1.39 | +1.90 | 7/2 | +0.08 |
| successor | heads_and_mlps | **INSUFFICIENT PROMPTS** | — | — / — | — | None/None | — |
| taskvec | heads_and_mlps | **INSUFFICIENT PROMPTS** | — | — / — | — | None/None | — |

## Per-cell detail

### agreement (heads_and_mlps) — INSUFFICIENT PROMPTS

- only 0 baseline-positive discovery prompts survive at run length 5 (need >= 8); see prompt_hygiene_report.md. Refusing to produce a tiny-n card.
- base metric —; n_discovery None, n_heldout None.
- knee None nodes; floor None nodes; knee−floor gap —.
- faithfulness — discovery: resample —, mean —; held-out: resample —, mean —.
- motif-core held-out (resample) —; induction motif present: None.
- suppression heads: 0; positive-causal MLPs: 0; MLPs in knee: none.
- edge: none claimed.

### agreement_long (heads_and_mlps) — INSUFFICIENT PROMPTS

- only 1 baseline-positive discovery prompts survive at run length 8 (need >= 8); see prompt_hygiene_report.md. Refusing to produce a tiny-n card.
- base metric —; n_discovery None, n_heldout None.
- knee None nodes; floor None nodes; knee−floor gap —.
- faithfulness — discovery: resample —, mean —; held-out: resample —, mean —.
- motif-core held-out (resample) —; induction motif present: None.
- suppression heads: 0; positive-causal MLPs: 0; MLPs in knee: none.
- edge: none claimed.

### induction_p2 (heads_and_mlps) — OVERFIT / NO CLEAN CIRCUIT

- discovery passes but held-out resample faithfulness 0.54 < 0.70.
- base metric +10.07; n_discovery 12, n_heldout 4.
- knee 12 nodes; floor 1 nodes; knee−floor gap +1.59.
- faithfulness — discovery: resample +1.97, mean +3.11; held-out: resample +0.54, mean +2.20.
- motif-core held-out (resample) +1.00; induction motif present: True.
- suppression heads: 1; positive-causal MLPs: 13; MLPs in knee: MLP20, MLP23, MLP24, MLP31, MLP39.
- edge: none claimed.

### induction_p3 (heads_and_mlps) — OVERFIT / NO CLEAN CIRCUIT

- no transferable subgraph: knee held-out resample 0.08 < 0.70.
- base metric +16.81; n_discovery 15, n_heldout 6.
- knee 15 nodes; floor 19 nodes; knee−floor gap -0.02.
- faithfulness — discovery: resample +0.29, mean +0.22; held-out: resample +0.08, mean +0.10.
- motif-core held-out (resample) -0.18; induction motif present: True.
- suppression heads: 6; positive-causal MLPs: 19; MLPs in knee: MLP22, MLP19, MLP10, MLP9, MLP20, MLP21, MLP14, MLP8, MLP15, MLP11, MLP12, MLP31.
- edge: L22H6 -> L41H2.

### induction_p3 (heads_only) — OVERFIT / NO CLEAN CIRCUIT

- no transferable subgraph: knee held-out resample 0.36 < 0.70.
- base metric +16.81; n_discovery 15, n_heldout 6.
- knee 2 nodes; floor 2 nodes; knee−floor gap +0.00.
- faithfulness — discovery: resample +0.17, mean +0.28; held-out: resample +0.36, mean +0.32.
- motif-core held-out (resample) +0.62; induction motif present: True.
- suppression heads: 6; positive-causal MLPs: 4; MLPs in knee: none.
- edge: none claimed.

### ioi (heads_and_mlps) — INSUFFICIENT PROMPTS

- only 0 baseline-positive discovery prompts survive at run length 13 (need >= 8); see prompt_hygiene_report.md. Refusing to produce a tiny-n card.
- base metric —; n_discovery None, n_heldout None.
- knee None nodes; floor None nodes; knee−floor gap —.
- faithfulness — discovery: resample —, mean —; held-out: resample —, mean —.
- motif-core held-out (resample) —; induction motif present: None.
- suppression heads: 0; positive-causal MLPs: 0; MLPs in knee: none.
- edge: none claimed.

### recall (heads_and_mlps) — OVERFIT / NO CLEAN CIRCUIT

- knee transfers (2.50) but the motif core alone does not (2.18715); the extra heads are filler, not mechanism.
- base metric +3.61; n_discovery 12, n_heldout 4.
- knee 6 nodes; floor 1 nodes; knee−floor gap +0.17.
- faithfulness — discovery: resample +1.67, mean +1.90; held-out: resample +2.50, mean +1.78.
- motif-core held-out (resample) +2.19; induction motif present: False.
- suppression heads: 7; positive-causal MLPs: 13; MLPs in knee: MLP15, MLP14.
- edge: L1H3 -> L41H1.

### recall (heads_only) — CIRCUIT CONFIRMED

- knee held-out resample faithfulness 2.03 >= 0.70; motif-core-only transfers comparably (1.90).
- base metric +3.61; n_discovery 12, n_heldout 4.
- knee 7 nodes; floor 2 nodes; knee−floor gap +0.35.
- faithfulness — discovery: resample +1.30, mean +1.39; held-out: resample +2.03, mean +2.16.
- motif-core held-out (resample) +1.90; induction motif present: False.
- suppression heads: 7; positive-causal MLPs: 3; MLPs in knee: none.
- edge: L1H3 -> L41H1.

### successor (heads_and_mlps) — INSUFFICIENT PROMPTS

- only 5 baseline-positive discovery prompts survive at run length 8 (need >= 8); see prompt_hygiene_report.md. Refusing to produce a tiny-n card.
- base metric —; n_discovery None, n_heldout None.
- knee None nodes; floor None nodes; knee−floor gap —.
- faithfulness — discovery: resample —, mean —; held-out: resample —, mean —.
- motif-core held-out (resample) —; induction motif present: None.
- suppression heads: 0; positive-causal MLPs: 0; MLPs in knee: none.
- edge: none claimed.

### taskvec (heads_and_mlps) — INSUFFICIENT PROMPTS

- only 0 baseline-positive discovery prompts survive at run length 9 (need >= 8); see prompt_hygiene_report.md. Refusing to produce a tiny-n card.
- base metric —; n_discovery None, n_heldout None.
- knee None nodes; floor None nodes; knee−floor gap —.
- faithfulness — discovery: resample —, mean —; held-out: resample —, mean —.
- motif-core held-out (resample) —; induction motif present: None.
- suppression heads: 0; positive-causal MLPs: 0; MLPs in knee: none.
- edge: none claimed.

## Negative / absent verdicts (each a successful result)

- `agreement/heads_and_mlps`: INSUFFICIENT PROMPTS — only 0 baseline-positive discovery prompts survive at run length 5 (need >= 8); see prompt_hygiene_report.md. Refusing to produce a tiny-n card.
- `agreement_long/heads_and_mlps`: INSUFFICIENT PROMPTS — only 1 baseline-positive discovery prompts survive at run length 8 (need >= 8); see prompt_hygiene_report.md. Refusing to produce a tiny-n card.
- `induction_p2/heads_and_mlps`: OVERFIT / NO CLEAN CIRCUIT — discovery passes but held-out resample faithfulness 0.54 < 0.70.
- `induction_p3/heads_and_mlps`: OVERFIT / NO CLEAN CIRCUIT — no transferable subgraph: knee held-out resample 0.08 < 0.70.
- `induction_p3/heads_only`: OVERFIT / NO CLEAN CIRCUIT — no transferable subgraph: knee held-out resample 0.36 < 0.70.
- `ioi/heads_and_mlps`: INSUFFICIENT PROMPTS — only 0 baseline-positive discovery prompts survive at run length 13 (need >= 8); see prompt_hygiene_report.md. Refusing to produce a tiny-n card.
- `recall/heads_and_mlps`: OVERFIT / NO CLEAN CIRCUIT — knee transfers (2.50) but the motif core alone does not (2.18715); the extra heads are filler, not mechanism.
- `successor/heads_and_mlps`: INSUFFICIENT PROMPTS — only 5 baseline-positive discovery prompts survive at run length 8 (need >= 8); see prompt_hygiene_report.md. Refusing to produce a tiny-n card.
- `taskvec/heads_and_mlps`: INSUFFICIENT PROMPTS — only 0 baseline-positive discovery prompts survive at run length 9 (need >= 8); see prompt_hygiene_report.md. Refusing to produce a tiny-n card.

## Cross-cutting reading

- **mean-ablation inflation:** the mean−resample gap column quantifies how much faithfulness was a mean-ablation artifact (positive = mean inflated). Large gaps with suppression heads present support the brake-removal explanation.
- **prev-token→induction core:** see each induction/successor cell's edge claim and `induction_motif_present`.
- **not-heads-only:** see the scope reconciliation section above.
- **successor:** expected MECHANISM ABSENT for the induction edge — a successful negative if so.

_Generated incrementally by lab06_matrix.py; updated after every cell._
