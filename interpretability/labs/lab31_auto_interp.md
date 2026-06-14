# Lab 31: Automated Interpretability At Scale

```text
Time estimate: 10-20 minutes Tier A smoke; 45-90 minutes for the full offline audit and writeup
Compute tier: Tier A uses an offline synthetic feature suite; no LLM judge or generation is required
Dependencies: Labs 8, 11, 30, and claim-ledger discipline
Minimum passing artifacts: method_card.md, operationalization_audit.md, diagnostics/self_check_status.json, diagnostics/safety_status.json, tables/generated_explanations.csv, tables/explanation_tests.csv, tables/explanation_scores.csv, tables/auto_interp_evidence_matrix.csv, tables/human_review_queue.csv, plots/auto_interp_dashboard.png
Main plot: plots/auto_interp_dashboard.png
Main table: tables/auto_interp_evidence_matrix.csv
Evidence rung: AUDIT + DECODE
Forbidden claim: The automated label is the feature's meaning.
One-sentence allowed claim: Explanation method E predicted held-out feature tests with AUC X and abstained on Y% of high-risk features under this suite.
Human-label requirement: fill the shared review columns before treating any generated label as course evidence.
```

## Lab thesis

Automated interpretability does not remove judgment. It moves judgment from first-contact naming to evaluation, calibration, abstention, and review.

The sentence this lab puts in the little evidence press is:

```text
The auto-label sounds right, therefore the feature means that.
```

Lab 31 replaces that sentence with a testable pipeline:

```text
top contexts -> candidate label -> deletion/confusable audit -> held-out tests -> calibration -> review queue -> scoped claim
```

The automated label is a hypothesis. The score asks whether that hypothesis predicts activation tests, not whether the prose is charming.

## What question this lab asks

Can an automated feature-labeling method produce labels that predict held-out feature contexts better than controls, while abstaining on random, ambiguous, or polysemantic features?

The important word is **predict**. A label earns credit only when it separates held-out positives from hard negatives, confusable negatives, paraphrase positives, and token-overlap decoys. A label that merely repeats a keyword has walked into the paper lantern and called it the moon.

## Why this matters in the course progression

Lab 8 taught manual feature interpretation. Lab 30 asks how features move across layers and models. Lab 31 asks how to scale the labeling step without allowing automation to launder weak evidence into confident nouns.

The skill is not prompt engineering. The skill is building an audit harness around an explanation method:

1. generate a candidate label;
2. generate or load tests it should pass and controls it should fail;
3. calibrate confidence;
4. abstain when the evidence is too mixed;
5. send fragile rows to human review;
6. write only the smallest claim the test suite supports.

## Data

Default tasks live in:

```text
data/auto_interp_feature_tasks.jsonl
```

Each JSONL row contains a synthetic or imported feature candidate:

```json
{
  "feature_id": "feat_code_json_schema",
  "model": "synthetic-sparse-feature-suite-v1",
  "layer": 8,
  "feature_index": 1000,
  "feature_type": "synthetic_gold",
  "top_contexts": [],
  "heldout_contexts": [],
  "paraphrase_contexts": [],
  "negative_contexts": [],
  "confusable_contexts": [],
  "adversarial_contexts": [],
  "gold_label": "code",
  "gold_label_secondary": "",
  "expected_abstain": false
}
```

The default suite includes domain-labeled synthetic gold features, polysemantic and ambiguous controls, and a random-control feature. Tier A caps rows for speed. Tier B with `--prompt-set full` uses all rows.

The suite is synthetic on purpose. It gives students a toy bench where the answer key exists, the controls are visible, and the first lesson is calibration rather than awe.

## What the experiment measures

The lab treats a generated label as a scoring rule over contexts. For a label such as `finance`, it computes a cheap lexical activation score for each held-out context and asks whether positive contexts rank above negative contexts.

The main score is AUC:

```text
AUC(label) = P(score(positive) > score(negative)) + 0.5 * P(tie)
```

Positive suites:

| suite | meaning |
|---|---|
| `heldout_positive` | ordinary held-out contexts for the same feature |
| `paraphrase_positive` | paraphrased positives where available |

Negative suites:

| suite | meaning |
|---|---|
| `hard_negative` | ordinary negatives from other domains |
| `confusable_negative` | contexts that share surface structure or neighboring domain language |
| `token_overlap_decoy` | contexts containing label words but not the concept |

A positive AUC is not enough. The label must also beat the shuffled-top-context control and avoid confusable/decoy failures.

## Explanation methods

| method | role | claim posture |
|---|---|---|
| `majority_domain` | chooses the highest lexicon domain in top contexts | brittle baseline |
| `structured_local` | penalizes labels that also fit confusables, decoys, or deletion-fragile evidence | primary offline heuristic |
| `test_aware` | abstains when risk flags, confusables, or decoys defeat the label | conservative automated method |
| `shuffled_top_context_control` | labels a feature using another feature's top contexts | control, should not count as evidence |
| `gold_calibration` | uses the synthetic gold label when available | upper bound, not an automated method |

`gold_calibration` is there to test the suite. If gold fails, the test suite or lexicon is broken. If an automated method ties gold on easy rows but fails high-risk rows, the audit is doing its job.

## What counts as evidence

A method earns `auto_label_audit_supported` only when it clears all gates:

| gate | question |
|---|---|
| held-out AUC | Does the label predict positives versus negatives? |
| control gap | Does it beat shuffled top contexts by a visible margin? |
| label accuracy | Does the label match synthetic gold when the feature is labelable? |
| good abstention | Does it abstain on random/polysemantic/ambiguous features? |
| bad abstention | Does it avoid refusing ordinary gold features? |
| confusable and decoy gates | Does it avoid firing on the cheap explanations? |
| review gate | Are fragile labels routed to `human_review_queue.csv`? |

The supported claim is about the **method under this suite**, not about feature essence.

## Controls and falsifiers

| control | what it tries to kill |
|---|---|
| shuffled-top-context control | Maybe any top contexts can produce a plausible label. |
| key-token deletion audit | Maybe the label only repeats a keyword. |
| confusable negatives | Maybe the label is neighboring-domain surface overlap. |
| token-overlap decoys | Maybe label words are being recognized without concept evidence. |
| random feature | Maybe the method always forces a label. |
| polysemantic/ambiguous controls | Maybe single-label methods should abstain. |
| gold calibration | Maybe the test suite itself is too weak or too hard. |
| human-review queue | Maybe the automatic metric misses wording quality or label incompleteness. |

A strong result has boring controls. A flashy control that is not actually diagnostic should not carry the claim.

## How to run

From `interpretability/`:

```bash
python interp_bench.py --lab lab31 --tier a --no-plots
python interp_bench.py --lab lab31 --tier a
python interp_bench.py --lab lab31 --tier b --prompt-set full
```

Run a custom JSONL suite:

```bash
python interp_bench.py --lab lab31 --tier a --prompt-set data/my_auto_interp_tasks.jsonl --no-plots
```

Regenerate the default synthetic suite if the generator is included:

```bash
python data/make_auto_interp_feature_tasks.py
```

Tier A proves the table machinery, self-checks, and artifact contract. Tier B is still offline, but runs the full suite and writes the same claim artifacts.

## Artifact tree

```text
runs/lab31_auto_interp-*/
  run_summary.md
  method_card.md
  operationalization_audit.md
  ledger_suggestions.md
  human_review_guide.md
  metrics.json
  results.csv

  diagnostics/
    data_manifest.json
    schema_audit.csv
    self_check_status.json
    safety_status.json

  tables/
    generated_explanations.csv
    key_token_deletion_audit.csv
    explanation_tests.csv
    explanation_scores.csv
    auto_interp_evidence_matrix.csv
    evidence_matrix.csv
    feature_evidence_matrix.csv
    human_review_queue.csv
    confidence_calibration.csv
    auto_interp_counterexamples.csv
    plot_reading_guide.csv

  cards/
    explanation_cards.md

  plots/
    auto_interp_dashboard.png
    explanation_quality_matrix.png
    confidence_calibration_curve.png
    abstention_frontier.png
    confusable_pair_failure_atlas.png
    random_feature_sanity_panel.png

  state/
    auto_interp_config.json
```

## Reading order

Start with `run_summary.md`. It says whether the run is science-ready, which method won, which methods were supported, and what the main counterexample was.

Then read:

1. `diagnostics/schema_audit.csv` and `diagnostics/data_manifest.json`: the suite must be valid before the scores matter.
2. `method_card.md`: read the verdict table before opening any individual label.
3. `tables/generated_explanations.csv`: inspect label wording, confidence, abstention, and evidence terms.
4. `tables/explanation_tests.csv`: read the exact positives, negatives, confusables, and decoys.
5. `tables/explanation_scores.csv`: see which feature-method rows passed or failed.
6. `tables/auto_interp_evidence_matrix.csv`: method-level claim posture.
7. `tables/auto_interp_counterexamples.csv`: rows that break broad language.
8. `tables/human_review_queue.csv` and `human_review_guide.md`: rows students must manually review before citing labels.
9. `cards/explanation_cards.md`: compact per-feature hypotheses for discussion.

Only after that should you use the dashboard.

## Figure guide

### `auto_interp_dashboard.png`

The cockpit plot. It compares method AUC to the shuffled control, shows label accuracy, shows good/bad abstention, and exposes confusable/decoy failures.

### `explanation_quality_matrix.png`

Rows are methods; columns are features. A single bright column is specimen-level storytelling. A broad row suggests a more general method claim.

### `confidence_calibration_curve.png`

Checks whether confidence predicts success. Confidence that does not calibrate is typography with a percentage sign.

### `abstention_frontier.png`

Plots good abstention on risky features against bad abstention on labelable gold features. A useful method lives near high-good, low-bad.

### `confusable_pair_failure_atlas.png`

Shows whether methods fire on confusable negatives and token-overlap decoys. This is the overclaim alarm.

### `random_feature_sanity_panel.png`

Shows whether random and polysemantic controls receive forced labels. A method that always labels is not interpreting. It is vending nouns.

## Expected outcomes

A strong run should look like this:

1. `gold_calibration` is high, proving the suite is not broken.
2. `shuffled_top_context_control` is much lower than real methods.
3. At least one automated method has high AUC and positive control gap.
4. Conservative methods abstain on random/polysemantic rows.
5. Counterexamples still exist and are visible.
6. The review queue is nonempty because automatic labels are not final evidence.

An honest negative run is useful:

| pattern | interpretation |
|---|---|
| gold calibration fails | the suite or scorer is broken |
| shuffled control matches methods | top contexts are not feature-specific enough |
| high AUC, high decoy failures | keyword matching, not concept prediction |
| forced labels on random/polysemantic features | abstention policy failed |
| high confidence, low success | calibration theater |
| one method works only on one feature | specimen-level evidence, not scalable method evidence |

## What this lab can claim

Allowed:

```text
AUDIT + DECODE: On the Lab 31 suite, method E predicted held-out feature contexts with mean AUC X, beat the shuffled-top-context control by Y, and abstained on Z% of high-risk features.
```

Also allowed:

```text
AUDIT: Method E failed because token-overlap decoys matched or beat held-out positives on N rows.
```

## What this lab cannot claim

Do not write:

```text
The automated label is the feature's meaning.
The method discovered monosemantic features.
The LLM understood the neuron.
A high-confidence label no longer needs human review.
Gold calibration is an automated method.
```

## Human review protocol

The following columns are intentionally blank in both `generated_explanations.csv` and `human_review_queue.csv`:

```text
student_label_primary
student_label_secondary
student_confidence
student_evidence_span
reviewer_label
agreement_status
```

Fill them for any row you cite in your writeup. Treat automatic labels as triage. Human review should check:

1. whether the label is too broad;
2. whether a secondary label is needed;
3. whether top and held-out contexts agree;
4. whether confusables or decoys expose a cheaper explanation;
5. whether the feature should remain unlabeled.

## Common failure modes

| symptom | likely cause | inspect |
|---|---|---|
| every method has high AUC | negatives are too easy or gold suite leaks label words | `explanation_tests.csv` |
| shuffled control is high | labels are not feature-specific | `auto_interp_evidence_matrix.csv` |
| `test_aware` abstains on everything | controls are too harsh, lexicon too weak, or suite too ambiguous | `key_token_deletion_audit.csv` |
| majority method beats conservative method | conservative abstention is useful but costs recall | `abstention_frontier.png` |
| confidence is high on failures | calibration problem | `confidence_calibration.csv` |
| review queue is empty | probably a bug or a too-easy suite | `human_review_queue.csv` |

## Suggested extensions

1. Import real SAE top contexts from Lab 8 and keep the same scoring schema.
2. Add an optional LLM explainer, but score it with the same tests and review fields.
3. Add generated counterfactual tests per label and compare them to hand-authored tests.
4. Add calibration bins by feature type rather than method only.
5. Add multi-label scoring for polysemantic rows instead of forcing abstention.
6. Compare explanations from Lab 8 features, Lab 19 crosscoder features, and Lab 30 lineage features under one evidence matrix.

## Writeup questions

1. Which automated method had the strongest supported claim, if any? Quote AUC, control gap, and abstention rates.
2. Which method looked good until confusable or decoy controls were included?
3. Which feature should go to human review first, and why?
4. Did confidence calibrate to success?
5. What is the smallest claim that survives without calling a label the feature's meaning?

## Ledger templates

Positive method result:

```text
[L31-C1] AUDIT + DECODE | On <suite>, method <E> predicted held-out feature contexts with mean AUC <X>, beat shuffled-top-context controls by <Y>, and abstained on <Z>% of high-risk rows. This is an explanation-audit result, not a claim that the labels are feature meanings.
Artifact: runs/<run>/tables/auto_interp_evidence_matrix.csv | Falsifier: confusable negatives, token-overlap decoys, shuffled contexts, key-token deletion, or human review invalidate the labels.
```

Negative method result:

```text
[L31-N1] AUDIT | Method <E> did not earn auto-label language because <control/confusable/decoy/calibration/abstention> defeated the label battery.
Artifact: runs/<run>/tables/auto_interp_counterexamples.csv | Falsifier: a held-out suite with matched confusables and human-reviewed labels clears the gates.
```

## Safety and scope

This lab uses benign synthetic context snippets and offline heuristic scoring. It does not generate harmful text, train a model, edit a model, run jailbreaks, or use an LLM judge. It still writes `diagnostics/safety_status.json` because automated explanations can become evidence-shaped theater if the review boundary is not explicit.
