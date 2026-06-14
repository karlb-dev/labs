# Lab 31: Automated Interpretability At Scale

Time estimate: 60-90 minutes for the default offline audit.  
Compute tier: Tier A uses an offline synthetic feature suite; no LLM judge is required.  
Dependencies: Labs 8, 11, 30, and claim-ledger discipline.  
Minimum passing artifacts: `tables/generated_explanations.csv`, `tables/explanation_tests.csv`, `tables/explanation_scores.csv`, `tables/auto_interp_evidence_matrix.csv`, `tables/human_review_queue.csv`, and `plots/auto_interp_dashboard.png`.  
Main plot: `plots/auto_interp_dashboard.png`.  
Main table: `tables/auto_interp_evidence_matrix.csv`.  
Evidence rung: `AUDIT + DECODE`.  
Forbidden claim: "The automated label is the feature's meaning."  
One-sentence allowed claim: "Explanation method E predicted held-out feature tests with AUC X and abstained on Y% of high-risk features under this suite."  
Human-label requirement: fill the shared review columns before treating any generated label as course evidence.

## Why This Lab Exists

Lab 8 taught manual feature validation. Lab 31 asks how to scale that validation
without turning auto-labels into decorative text.

An automated explanation is a hypothesis:

```text
top contexts -> candidate label -> generated tests -> held-out score -> review queue
```

It is not a feature meaning.

## Data

Default tasks live in `data/auto_interp_feature_tasks.jsonl`.

Each row contains:

```json
{
  "feature_id": "...",
  "model": "...",
  "layer": 8,
  "feature_index": 123,
  "top_contexts": [],
  "heldout_contexts": [],
  "negative_contexts": [],
  "confusable_contexts": [],
  "adversarial_contexts": [],
  "gold_label": "optional"
}
```

The default suite includes synthetic gold features, one polysemantic feature,
and one random-control feature.

## Methods

The first pass is fully offline:

- `majority_domain`: chooses the domain with the most lexicon hits in top contexts.
- `structured_local`: uses top contexts but penalizes labels that also fit confusables.
- `test_aware`: abstains when key-token deletion or polysemantic flags make the label unsafe.
- `shuffled_top_context_control`: runs the same heuristic on contexts from a
  different feature, so it should not survive as evidence.
- `gold_calibration`: an upper-bound calibration row, not an automated method.

## Tests

Every explanation is scored on:

- held-out positives;
- hard negatives;
- confusable negatives;
- token-overlap decoys.

The score asks whether the label predicts activation tests, not whether the text
sounds plausible.

## Human Review Columns

`tables/generated_explanations.csv` and `tables/human_review_queue.csv` export:

```text
student_label_primary,student_label_secondary,student_confidence,
student_evidence_span,reviewer_label,agreement_status
```

These are intentionally blank. A writeup that cites an auto-label should fill
them for the relevant rows.

## How To Run

```bash
cd interpretability
python interp_bench.py --lab lab31 --tier a
python interp_bench.py --lab lab31 --tier b --prompt-set full
```

For a fast table-only smoke:

```bash
python interp_bench.py --lab lab31 --tier a --no-plots
```

## Reading Order

1. `method_card.md`

   Confirms which explanation methods ran and which labels are only calibration
   upper bounds.

2. `tables/generated_explanations.csv`

   Shows labels, confidence, abstention, evidence terms, and human-review
   fields.

3. `tables/explanation_tests.csv`

   Lists the held-out, negative, confusable, and decoy contexts.

4. `tables/explanation_scores.csv`

   Scores each feature/method pair.

5. `tables/auto_interp_evidence_matrix.csv`

   Aggregates methods into claim postures, including calibration upper bounds,
   controls, abstention-limited labels, confusable-limited labels, and supported
   audit claims.

6. `tables/human_review_queue.csv`

   The rows students must review before trusting labels.

## Common Failure Modes

### Keyword Labels

If token-overlap decoys score highly, the method is recognizing words rather
than a feature.

### Confusable Domains

Finance/sports and law/medicine rows deliberately share surface words. A label
that fails confusables is not specific enough.

### Polysemantic Features

A good auto-interpretability system should sometimes abstain. Forced labels on
polysemantic features belong in the review queue.

### Calibration Theater

Confidence is only useful if it predicts success. Inspect
`tables/confidence_calibration.csv` before trusting a high-confidence label.

## Claim Grammar

Allowed:

```text
AUDIT + DECODE: Explanation method E predicted held-out feature activations
with AUC X and abstained on Y% of high-polysemantic/random features under test
suite T.
```

Forbidden:

```text
The automated label is the feature's meaning.
```

Also forbidden:

```text
The LLM understood the neuron.
The label is complete because it sounds right.
This auto-label no longer needs human review.
```

## Deliverable

Write a short auto-interpretability audit:

- Which method had the best held-out AUC?
- Which method handled polysemantic/random features best?
- Which feature had the worst confusable failure?
- Which generated label would you send to human review first?
- What is the smallest claim that survives the test suite?
