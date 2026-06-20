# Lab 6 validation matrix — `gpt2`

Matrix `lab06_matrix_20260620`. One question per cell: does this model implement a clean, transferable circuit for this behavior — yes, no, or not-as-a-heads-only-graph? **A confirmed NO is a success.** Headline faithfulness is RESAMPLE (interchange) ablation; mean ablation is shown for the inflation comparison.

## Results matrix

| behavior | scope | verdict | held-out F (resample) | disc F (resample / mean) | motif-core held F | knee/floor nodes | mean−resample gap |
|---|---|---|---|---|---|---|---|
| agreement | heads_and_mlps | **OVERFIT / NO CLEAN CIRCUIT** | -0.33 | -0.08 / +0.12 | -0.33 | 17/24 | +0.20 |
| agreement_long | heads_and_mlps | **OVERFIT / NO CLEAN CIRCUIT** | +0.37 | +0.53 / +1.03 | -0.98 | 23/15 | +0.50 |
| induction_p2 | heads_and_mlps | **INSUFFICIENT PROMPTS** | — | — / — | — | None/None | — |
| induction_p3 | heads_and_mlps | **OVERFIT / NO CLEAN CIRCUIT** | +0.12 | +0.13 / +0.98 | -0.31 | 24/20 | +0.85 |
| induction_p3 | heads_only | **OVERFIT / NO CLEAN CIRCUIT** | +0.31 | +0.22 / +0.92 | +0.23 | 15/12 | +0.70 |
| ioi | heads_and_mlps | **OVERFIT / NO CLEAN CIRCUIT** | +0.18 | +1.19 / +1.23 | -1.11 | 23/9 | +0.04 |
| recall | heads_and_mlps | **OVERFIT / NO CLEAN CIRCUIT** | +1.16 | +0.91 / +1.02 | +0.56 | 24/11 | +0.11 |
| recall | heads_only | **OVERFIT / NO CLEAN CIRCUIT** | +1.20 | +0.93 / +1.03 | +0.78 | 14/4 | +0.09 |
| successor | heads_and_mlps | **OVERFIT / NO CLEAN CIRCUIT** | +0.40 | +0.51 / +1.11 | -0.04 | 17/8 | +0.60 |
| taskvec | heads_and_mlps | **INSUFFICIENT PROMPTS** | — | — / — | — | None/None | — |

## Per-cell detail

### agreement (heads_and_mlps) — OVERFIT / NO CLEAN CIRCUIT

- no transferable subgraph: knee held-out resample -0.33 < 0.70.
- base metric +3.67; n_discovery 11, n_heldout 4.
- knee 17 nodes; floor 24 nodes; knee−floor gap -0.01.
- faithfulness — discovery: resample -0.08, mean +0.12; held-out: resample -0.33, mean +0.09.
- motif-core held-out (resample) -0.33; induction motif present: False.
- suppression heads: 0; positive-causal MLPs: 8; MLPs in knee: MLP8, MLP10, MLP11, MLP9, MLP7, MLP5, MLP4, MLP1.
- edge: none claimed.

### agreement_long (heads_and_mlps) — OVERFIT / NO CLEAN CIRCUIT

- discovery passes but held-out resample faithfulness 0.37 < 0.70.
- base metric +3.40; n_discovery 10, n_heldout 2.
- knee 23 nodes; floor 15 nodes; knee−floor gap +0.33.
- faithfulness — discovery: resample +0.53, mean +1.03; held-out: resample +0.37, mean +1.12.
- motif-core held-out (resample) -0.98; induction motif present: False.
- suppression heads: 0; positive-causal MLPs: 10; MLPs in knee: MLP10, MLP8, MLP11, MLP0, MLP7, MLP5, MLP6, MLP4, MLP2.
- edge: none claimed.

### induction_p2 (heads_and_mlps) — INSUFFICIENT PROMPTS

- only 6 baseline-positive discovery prompts survive at run length 8 (need >= 8); see prompt_hygiene_report.md. Refusing to produce a tiny-n card.
- base metric —; n_discovery None, n_heldout None.
- knee None nodes; floor None nodes; knee−floor gap —.
- faithfulness — discovery: resample —, mean —; held-out: resample —, mean —.
- motif-core held-out (resample) —; induction motif present: None.
- suppression heads: 0; positive-causal MLPs: 0; MLPs in knee: none.
- edge: none claimed.

### induction_p3 (heads_and_mlps) — OVERFIT / NO CLEAN CIRCUIT

- discovery passes but held-out resample faithfulness 0.12 < 0.70.
- base metric +1.38; n_discovery 9, n_heldout 6.
- knee 24 nodes; floor 20 nodes; knee−floor gap +0.21.
- faithfulness — discovery: resample +0.13, mean +0.98; held-out: resample +0.12, mean +0.48.
- motif-core held-out (resample) -0.31; induction motif present: True.
- suppression heads: 4; positive-causal MLPs: 12; MLPs in knee: MLP0, MLP4, MLP2, MLP1, MLP3, MLP10, MLP9, MLP5, MLP8, MLP11.
- edge: L4H11 -> L5H1.

### induction_p3 (heads_only) — OVERFIT / NO CLEAN CIRCUIT

- discovery passes but held-out resample faithfulness 0.31 < 0.70.
- base metric +1.38; n_discovery 9, n_heldout 6.
- knee 15 nodes; floor 12 nodes; knee−floor gap +0.15.
- faithfulness — discovery: resample +0.22, mean +0.92; held-out: resample +0.31, mean +0.46.
- motif-core held-out (resample) +0.23; induction motif present: True.
- suppression heads: 4; positive-causal MLPs: 8; MLPs in knee: none.
- edge: L4H11 -> L5H1.

### ioi (heads_and_mlps) — OVERFIT / NO CLEAN CIRCUIT

- discovery passes but held-out resample faithfulness 0.18 < 0.70.
- base metric +1.72; n_discovery 11, n_heldout 2.
- knee 23 nodes; floor 9 nodes; knee−floor gap +0.52.
- faithfulness — discovery: resample +1.19, mean +1.23; held-out: resample +0.18, mean +0.80.
- motif-core held-out (resample) -1.11; induction motif present: False.
- suppression heads: 4; positive-causal MLPs: 11; MLPs in knee: MLP0, MLP1, MLP11, MLP3, MLP9, MLP7, MLP5, MLP8, MLP4, MLP10.
- edge: none claimed.

### recall (heads_and_mlps) — OVERFIT / NO CLEAN CIRCUIT

- knee transfers (1.16) but the motif core alone does not (0.55968); the extra heads are filler, not mechanism.
- base metric +2.84; n_discovery 13, n_heldout 3.
- knee 24 nodes; floor 11 nodes; knee−floor gap +0.31.
- faithfulness — discovery: resample +0.91, mean +1.02; held-out: resample +1.16, mean +1.04.
- motif-core held-out (resample) +0.56; induction motif present: False.
- suppression heads: 1; positive-causal MLPs: 10; MLPs in knee: MLP0, MLP8, MLP2, MLP9, MLP3, MLP7, MLP4, MLP6, MLP5, MLP10.
- edge: L9H3 -> L10H0.

### recall (heads_only) — OVERFIT / NO CLEAN CIRCUIT

- knee transfers (1.20) but the motif core alone does not (0.78038); the extra heads are filler, not mechanism.
- base metric +2.84; n_discovery 13, n_heldout 3.
- knee 14 nodes; floor 4 nodes; knee−floor gap +0.27.
- faithfulness — discovery: resample +0.93, mean +1.03; held-out: resample +1.20, mean +1.21.
- motif-core held-out (resample) +0.78; induction motif present: False.
- suppression heads: 1; positive-causal MLPs: 6; MLPs in knee: none.
- edge: L9H3 -> L10H0.

### successor (heads_and_mlps) — OVERFIT / NO CLEAN CIRCUIT

- discovery passes but held-out resample faithfulness 0.40 < 0.70.
- base metric +6.26; n_discovery 12, n_heldout 4.
- knee 17 nodes; floor 8 nodes; knee−floor gap +0.35.
- faithfulness — discovery: resample +0.51, mean +1.11; held-out: resample +0.40, mean +0.68.
- motif-core held-out (resample) -0.04; induction motif present: True.
- suppression heads: 2; positive-causal MLPs: 12; MLPs in knee: MLP0, MLP9, MLP10, MLP11, MLP3, MLP7, MLP8, MLP6, MLP1, MLP5.
- edge: none claimed.

### taskvec (heads_and_mlps) — INSUFFICIENT PROMPTS

- only 1 baseline-positive discovery prompts survive at run length 9 (need >= 8); see prompt_hygiene_report.md. Refusing to produce a tiny-n card.
- base metric —; n_discovery None, n_heldout None.
- knee None nodes; floor None nodes; knee−floor gap —.
- faithfulness — discovery: resample —, mean —; held-out: resample —, mean —.
- motif-core held-out (resample) —; induction motif present: None.
- suppression heads: 0; positive-causal MLPs: 0; MLPs in knee: none.
- edge: none claimed.

## Negative / absent verdicts (each a successful result)

- `agreement/heads_and_mlps`: OVERFIT / NO CLEAN CIRCUIT — no transferable subgraph: knee held-out resample -0.33 < 0.70.
- `agreement_long/heads_and_mlps`: OVERFIT / NO CLEAN CIRCUIT — discovery passes but held-out resample faithfulness 0.37 < 0.70.
- `induction_p2/heads_and_mlps`: INSUFFICIENT PROMPTS — only 6 baseline-positive discovery prompts survive at run length 8 (need >= 8); see prompt_hygiene_report.md. Refusing to produce a tiny-n card.
- `induction_p3/heads_and_mlps`: OVERFIT / NO CLEAN CIRCUIT — discovery passes but held-out resample faithfulness 0.12 < 0.70.
- `induction_p3/heads_only`: OVERFIT / NO CLEAN CIRCUIT — discovery passes but held-out resample faithfulness 0.31 < 0.70.
- `ioi/heads_and_mlps`: OVERFIT / NO CLEAN CIRCUIT — discovery passes but held-out resample faithfulness 0.18 < 0.70.
- `recall/heads_and_mlps`: OVERFIT / NO CLEAN CIRCUIT — knee transfers (1.16) but the motif core alone does not (0.55968); the extra heads are filler, not mechanism.
- `recall/heads_only`: OVERFIT / NO CLEAN CIRCUIT — knee transfers (1.20) but the motif core alone does not (0.78038); the extra heads are filler, not mechanism.
- `successor/heads_and_mlps`: OVERFIT / NO CLEAN CIRCUIT — discovery passes but held-out resample faithfulness 0.40 < 0.70.
- `taskvec/heads_and_mlps`: INSUFFICIENT PROMPTS — only 1 baseline-positive discovery prompts survive at run length 9 (need >= 8); see prompt_hygiene_report.md. Refusing to produce a tiny-n card.

## Cross-cutting reading

- **mean-ablation inflation:** the mean−resample gap column quantifies how much faithfulness was a mean-ablation artifact (positive = mean inflated). Large gaps with suppression heads present support the brake-removal explanation.
- **prev-token→induction core:** see each induction/successor cell's edge claim and `induction_motif_present`.
- **not-heads-only:** see the scope reconciliation section above.
- **successor:** expected MECHANISM ABSENT for the induction edge — a successful negative if so.

_Generated incrementally by lab06_matrix.py; updated after every cell._
