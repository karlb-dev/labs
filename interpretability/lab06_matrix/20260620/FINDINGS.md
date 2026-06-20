# Lab 6 validation matrix — `allenai/Olmo-3-1125-32B`

Matrix `lab06_matrix_20260620`. One question per cell: does this model implement a clean, transferable circuit for this behavior — yes, no, or not-as-a-heads-only-graph? **A confirmed NO is a success.** Headline faithfulness is RESAMPLE (interchange) ablation; mean ablation is shown for the inflation comparison.

## Results matrix

| behavior | scope | verdict | held-out F (resample) | disc F (resample / mean) | motif-core held F | knee/floor nodes | mean−resample gap |
|---|---|---|---|---|---|---|---|
| agreement | heads_and_mlps | **OVERFIT / NO CLEAN CIRCUIT** | -0.23 | +0.04 / +1.44 | -0.32 | 43/10 | +1.40 |
| agreement_long | heads_and_mlps | **OVERFIT / NO CLEAN CIRCUIT** | -0.77 | +0.06 / +0.96 | -0.98 | 44/21 | +0.90 |
| induction_p2 | heads_and_mlps | **OVERFIT / NO CLEAN CIRCUIT** | +0.28 | +0.22 / +0.36 | +0.23 | 37/54 | +0.14 |
| induction_p3 | heads_and_mlps | **OVERFIT / NO CLEAN CIRCUIT** | -0.10 | -0.02 / +0.36 | -0.19 | 35/44 | +0.38 |
| induction_p3 | heads_only | **OVERFIT / NO CLEAN CIRCUIT** | +0.39 | +0.13 / +0.56 | +0.23 | 23/27 | +0.43 |
| ioi | heads_and_mlps | **OVERFIT / NO CLEAN CIRCUIT** | +0.34 | +0.18 / +0.35 | +0.25 | 30/52 | +0.16 |
| recall | heads_and_mlps | **INSUFFICIENT PROMPTS** | — | — / — | — | None/None | — |
| recall | heads_only | **INSUFFICIENT PROMPTS** | — | — / — | — | None/None | — |
| successor | heads_and_mlps | **OVERFIT / NO CLEAN CIRCUIT** | -0.09 | +0.02 / +0.24 | -0.13 | 31/50 | +0.22 |
| taskvec | heads_and_mlps | **OVERFIT / NO CLEAN CIRCUIT** | +0.14 | +0.19 / +0.63 | +0.06 | 46/60 | +0.44 |

## Per-cell detail

### agreement (heads_and_mlps) — OVERFIT / NO CLEAN CIRCUIT

- discovery passes but held-out resample faithfulness -0.23 < 0.70.
- base metric +3.88; n_discovery 12, n_heldout 4.
- knee 43 nodes; floor 10 nodes; knee−floor gap +0.66.
- faithfulness — discovery: resample +0.04, mean +1.44; held-out: resample -0.23, mean +1.37.
- motif-core held-out (resample) -0.32; induction motif present: False.
- suppression heads: 0; positive-causal MLPs: 33; MLPs in knee: MLP61, MLP22, MLP24, MLP52, MLP43, MLP59, MLP41, MLP28, MLP27, MLP23, MLP8, MLP57, MLP46, MLP30, MLP4, MLP32, MLP29, MLP14, MLP55, MLP5, MLP25, MLP45.
- edge: none claimed.

### agreement_long (heads_and_mlps) — OVERFIT / NO CLEAN CIRCUIT

- discovery passes but held-out resample faithfulness -0.77 < 0.70.
- base metric +4.93; n_discovery 11, n_heldout 2.
- knee 44 nodes; floor 21 nodes; knee−floor gap +0.25.
- faithfulness — discovery: resample +0.06, mean +0.96; held-out: resample -0.77, mean +1.22.
- motif-core held-out (resample) -0.98; induction motif present: False.
- suppression heads: 0; positive-causal MLPs: 37; MLPs in knee: MLP61, MLP63, MLP24, MLP27, MLP52, MLP22, MLP28, MLP46, MLP50, MLP29, MLP25, MLP30, MLP0, MLP59, MLP35, MLP32, MLP17, MLP23, MLP3, MLP53, MLP43, MLP20, MLP19.
- edge: none claimed.

### induction_p2 (heads_and_mlps) — OVERFIT / NO CLEAN CIRCUIT

- no transferable subgraph: knee held-out resample 0.28 < 0.70.
- base metric +4.94; n_discovery 12, n_heldout 4.
- knee 37 nodes; floor 54 nodes; knee−floor gap -0.02.
- faithfulness — discovery: resample +0.22, mean +0.36; held-out: resample +0.28, mean +0.24.
- motif-core held-out (resample) +0.23; induction motif present: True.
- suppression heads: 0; positive-causal MLPs: 26; MLPs in knee: MLP1, MLP61, MLP63, MLP58, MLP62, MLP2, MLP32, MLP0, MLP11, MLP5, MLP49, MLP37, MLP44, MLP56, MLP16, MLP51, MLP33, MLP17, MLP31.
- edge: none claimed.

### induction_p3 (heads_and_mlps) — OVERFIT / NO CLEAN CIRCUIT

- no transferable subgraph: knee held-out resample -0.10 < 0.70.
- base metric +5.18; n_discovery 12, n_heldout 6.
- knee 35 nodes; floor 44 nodes; knee−floor gap -0.02.
- faithfulness — discovery: resample -0.02, mean +0.36; held-out: resample -0.10, mean +0.08.
- motif-core held-out (resample) -0.19; induction motif present: True.
- suppression heads: 1; positive-causal MLPs: 19; MLPs in knee: MLP63, MLP62, MLP61, MLP22, MLP13, MLP11, MLP57, MLP28, MLP60, MLP23, MLP59, MLP29, MLP54, MLP36, MLP50, MLP41, MLP49, MLP40.
- edge: L23H13 -> L27H15.

### induction_p3 (heads_only) — OVERFIT / NO CLEAN CIRCUIT

- no transferable subgraph: knee held-out resample 0.39 < 0.70.
- base metric +5.18; n_discovery 12, n_heldout 6.
- knee 23 nodes; floor 27 nodes; knee−floor gap -0.02.
- faithfulness — discovery: resample +0.13, mean +0.56; held-out: resample +0.39, mean +0.78.
- motif-core held-out (resample) +0.23; induction motif present: True.
- suppression heads: 1; positive-causal MLPs: 5; MLPs in knee: none.
- edge: L23H13 -> L27H15.

### ioi (heads_and_mlps) — OVERFIT / NO CLEAN CIRCUIT

- no transferable subgraph: knee held-out resample 0.34 < 0.70.
- base metric +6.54; n_discovery 12, n_heldout 2.
- knee 30 nodes; floor 52 nodes; knee−floor gap -0.02.
- faithfulness — discovery: resample +0.18, mean +0.35; held-out: resample +0.34, mean +0.31.
- motif-core held-out (resample) +0.25; induction motif present: False.
- suppression heads: 0; positive-causal MLPs: 18; MLPs in knee: MLP16, MLP34, MLP47, MLP37, MLP21, MLP13, MLP15, MLP12, MLP18, MLP3, MLP22.
- edge: none claimed.

### recall (heads_and_mlps) — INSUFFICIENT PROMPTS

- only 7 baseline-positive discovery prompts survive at run length 8 (need >= 8); see prompt_hygiene_report.md. Refusing to produce a tiny-n card.
- base metric —; n_discovery None, n_heldout None.
- knee None nodes; floor None nodes; knee−floor gap —.
- faithfulness — discovery: resample —, mean —; held-out: resample —, mean —.
- motif-core held-out (resample) —; induction motif present: None.
- suppression heads: 0; positive-causal MLPs: 0; MLPs in knee: none.
- edge: none claimed.

### recall (heads_only) — INSUFFICIENT PROMPTS

- only 7 baseline-positive discovery prompts survive at run length 8 (need >= 8); see prompt_hygiene_report.md. Refusing to produce a tiny-n card.
- base metric —; n_discovery None, n_heldout None.
- knee None nodes; floor None nodes; knee−floor gap —.
- faithfulness — discovery: resample —, mean —; held-out: resample —, mean —.
- motif-core held-out (resample) —; induction motif present: None.
- suppression heads: 0; positive-causal MLPs: 0; MLPs in knee: none.
- edge: none claimed.

### successor (heads_and_mlps) — OVERFIT / NO CLEAN CIRCUIT

- no transferable subgraph: knee held-out resample -0.09 < 0.70.
- base metric +7.23; n_discovery 12, n_heldout 4.
- knee 31 nodes; floor 50 nodes; knee−floor gap -0.02.
- faithfulness — discovery: resample +0.02, mean +0.24; held-out: resample -0.09, mean +0.06.
- motif-core held-out (resample) -0.13; induction motif present: True.
- suppression heads: 0; positive-causal MLPs: 29; MLPs in knee: MLP55, MLP57, MLP58, MLP50, MLP53, MLP56, MLP49, MLP47, MLP46, MLP31, MLP61, MLP48, MLP44, MLP29, MLP33, MLP30, MLP26, MLP60, MLP40, MLP15, MLP38, MLP45, MLP20, MLP41.
- edge: none claimed.

### taskvec (heads_and_mlps) — OVERFIT / NO CLEAN CIRCUIT

- no transferable subgraph: knee held-out resample 0.14 < 0.70.
- base metric +5.80; n_discovery 12, n_heldout 4.
- knee 46 nodes; floor 60 nodes; knee−floor gap -0.02.
- faithfulness — discovery: resample +0.19, mean +0.63; held-out: resample +0.14, mean +0.79.
- motif-core held-out (resample) +0.06; induction motif present: False.
- suppression heads: 1; positive-causal MLPs: 34; MLPs in knee: MLP57, MLP56, MLP58, MLP34, MLP28, MLP51, MLP31, MLP55, MLP48, MLP53, MLP37, MLP42, MLP39, MLP29, MLP27, MLP16, MLP30, MLP20, MLP32, MLP50, MLP14, MLP43, MLP33, MLP19, MLP26, MLP17, MLP40.
- edge: none claimed.

## Negative / absent verdicts (each a successful result)

- `agreement/heads_and_mlps`: OVERFIT / NO CLEAN CIRCUIT — discovery passes but held-out resample faithfulness -0.23 < 0.70.
- `agreement_long/heads_and_mlps`: OVERFIT / NO CLEAN CIRCUIT — discovery passes but held-out resample faithfulness -0.77 < 0.70.
- `induction_p2/heads_and_mlps`: OVERFIT / NO CLEAN CIRCUIT — no transferable subgraph: knee held-out resample 0.28 < 0.70.
- `induction_p3/heads_and_mlps`: OVERFIT / NO CLEAN CIRCUIT — no transferable subgraph: knee held-out resample -0.10 < 0.70.
- `induction_p3/heads_only`: OVERFIT / NO CLEAN CIRCUIT — no transferable subgraph: knee held-out resample 0.39 < 0.70.
- `ioi/heads_and_mlps`: OVERFIT / NO CLEAN CIRCUIT — no transferable subgraph: knee held-out resample 0.34 < 0.70.
- `recall/heads_and_mlps`: INSUFFICIENT PROMPTS — only 7 baseline-positive discovery prompts survive at run length 8 (need >= 8); see prompt_hygiene_report.md. Refusing to produce a tiny-n card.
- `recall/heads_only`: INSUFFICIENT PROMPTS — only 7 baseline-positive discovery prompts survive at run length 8 (need >= 8); see prompt_hygiene_report.md. Refusing to produce a tiny-n card.
- `successor/heads_and_mlps`: OVERFIT / NO CLEAN CIRCUIT — no transferable subgraph: knee held-out resample -0.09 < 0.70.
- `taskvec/heads_and_mlps`: OVERFIT / NO CLEAN CIRCUIT — no transferable subgraph: knee held-out resample 0.14 < 0.70.

## Cross-cutting reading

- **mean-ablation inflation:** the mean−resample gap column quantifies how much faithfulness was a mean-ablation artifact (positive = mean inflated). Large gaps with suppression heads present support the brake-removal explanation.
- **prev-token→induction core:** see each induction/successor cell's edge claim and `induction_motif_present`.
- **not-heads-only:** see the scope reconciliation section above.
- **successor:** expected MECHANISM ABSENT for the induction edge — a successful negative if so.

_Generated incrementally by lab06_matrix.py; updated after every cell._
