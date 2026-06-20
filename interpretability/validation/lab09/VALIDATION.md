# Lab 09 Validation

## Lab 9: Attribution Graphs and Circuit Tracing

Attribution graphs: a transcoder replacement model, feature-level circuit
tracing, and interventions.

## Current Read

Lab 9 is a **clean scoped positive**. It is not a broad claim about arbitrary
attribution graphs or all factual recall. It is a validated miniature:
GPT-2 small, the public Dunefsky full-stack MLP transcoders, one-hop factual
recall for `The capital of France is`, a paraphrase/counterfactual battery, and
one Lab 6 induction confrontation.

The result is strong because the lab separates three things that are often
blurred:

- replacement-model health receipts;
- attribution graph visibility and error-node accounting;
- real-model feature interventions with a matched random control.

## Fresh Validation Run

Run directory:

`runs/lab09_fresh_tierc_20260620`

Backed up at:

`/content/drive/MyDrive/interpret/lab09_attribution_graphs_validation_20260620/lab09_fresh_tierc_20260620/`

Command:

```bash
python -m py_compile interpretability/interp_bench.py interpretability/labs/lab09_attribution_graphs.py
cd interpretability
python interp_bench.py --lab lab9 --tier c --model gpt2 --dtype float32 --attn-implementation eager --run-name lab09_fresh_tierc_20260620
```

## Headline Metrics

| Metric | Value |
|---|---:|
| model | `gpt2` |
| node budget | 48 |
| replacement max logit diff | 0.00029 |
| edge reconstruction relative error | 0.0000058 |
| feature-edit no-op max logit diff | 0.0 |
| metric logit diff | 3.0126 |
| feature share of direct edge mass | 0.7029 |
| error-node share of direct edge mass | 0.1946 |
| kept coverage of feature mass | 0.4435 |
| suppress drop | 4.8549 |
| random-control drop | -0.0112 |
| specificity gap | 4.8661 |
| substitution shift | 8.2680 |
| recurring subject-site features | 9 |
| baseline-gate dropped prompts | 1 |
| mean transcoder FVU | 0.149 |

## Current Verdict

The graph is `ATTR` evidence on the replacement model. The suppression and
substitution rows are `CAUSAL` evidence for the tested real-model feature edit.

The causal result is clean for the scoped behavior:

- baseline France-vs-Berlin logit diff: `+3.0126`;
- graph-guided subject-supernode suppression: `-1.8423`;
- matched random suppression control: `+3.0238`;
- specificity gap: `+4.8661` logits;
- counterfactual Germany-feature substitution: `-5.2554`.

## Why It Is Solid

- `replacement_exactness.json` passes, so the local replacement reproduces the
  real logits once error nodes are included.
- `edge_reconstruction_check.json` passes, so direct edges plus bias account for
  the target scalar.
- `feature_edit_noop_check.json` passes, so the real-model edit hooks are
  aligned.
- Feature mass is substantial (`0.7029`), but error-node mass is still reported
  (`0.1946`) rather than hidden.
- The graph-guided intervention beats the matched random control by a large
  margin.
- The paraphrase battery finds recurring subject-site features instead of only
  one-template nodes.
- The induction vignette exposes the expected blind spot: frozen-attention
  feature graphs are less explanatory for attention-routing behavior.

## Current Artifacts

- `gpt2_lab09_fresh_tierc_20260620_metrics.json`
- `gpt2_lab09_fresh_tierc_20260620_results.csv`
- `gpt2_lab09_fresh_tierc_20260620_graph_evidence_matrix.csv`
- `gpt2_lab09_fresh_tierc_20260620_graph_evidence_dashboard.png`
- `gpt2_lab09_fresh_tierc_20260620_source_token_ledger.png`
- `gpt2_lab09_fresh_tierc_20260620_influence_composition.png`
- `gpt2_lab09_fresh_tierc_20260620_intervention_effects.png`
- `lab09_validation_summary.csv`
- `lab09_validation_report.md`

Older run6 and full-matrix artifacts remain in this directory as reproducibility
context; the fresh Tier C run above is the preferred validation read.

## Residual Risk

- The backend is GPT-2-specific. The handout is clear that a Gemma/circuit-tracer
  backend would be a separate implementation with the same evidence contract.
- The behavior is one-hop factual recall, not a broad factual-recall benchmark.
- The graph is budgeted: it keeps `0.4435` of direct feature mass, so the plotted
  graph is readable but not complete.
- Error nodes still carry `0.1946` of direct edge mass. This is acceptable only
  because the lab explicitly reports it.
