# Lab 4: Probing Without Fooling Yourself, now featuring truth

**Evidence level targeted:** decodability (`DECODE`), with controls. The causal question, whether the model *uses* what a probe decodes, is deferred to Lab 7. Lab 7 loads the direction this lab saves and tests it by intervention.

**Prerequisites:** Labs 1 to 3. You already know the residual-stream indexing and “readout is an instrument” caution from Lab 1, the frozen-norm linearization and “ledger is not a causal map” discipline from Lab 2, and the distinction between routing/attention and actual contribution from Lab 3. This lab applies the same skepticism to linear probes.

## The question

What is linearly decodable from the residual stream, including whether a statement is **true**, and how do you stop yourself from turning “a probe found it” into “the model represents and uses it”?

This lab is skepticism with a spreadsheet. Every headline accuracy travels with its controls, split audit, calibration, and caveats. If a number cannot survive that little parade, it does not get a claim in the ledger.

## The two tracks

### 1. Surface track: a calibration trap

The surface track asks whether the final word contains a selected letter. It is intentionally shallow. It gives you the shape of “trivially decodable” information on the same activations and axes as the truth probe.

**Headline numbers note:** Full runs draw from ~100+ statements per family (cities/negations/comparisons) with per-family caps for smoke/medium. Selectivity, transfer (esp. negation), and calibration are the robust signals; headline accuracies are scoped to these templates/families and should be read with one-significant-figure confidence plus the full control matrix.

The revised code chooses the letter from the alphabet by looking for a feature that is balanced enough to train inside every family split. This avoids the old footgun where the surface probe could quietly become a one-class classifier in a small run.

### 2. Truth track: the headline

The truth track probes the end-of-statement residual stream on three frozen local families in `data/`:

| Family | Example | Why it is here |
|---|---|---|
| `cities` | `The city of Paris is in France.` | factual templates |
| `comparisons` | `Sixty-one is larger than fourteen.` | non-city factual structure |
| `negations` | `The city of Paris is not in France.` | catches probes that read templates instead of truth |

The datasets are frozen and vendored. Do not regenerate them during a run. Do not ask students to author their own truth sets for this lab. The point is controlled measurement, not benchmark carpentry.

## The two probes

### Logistic regression

The logistic probe is a standard linear classifier trained with torch LBFGS and L2 regularization. It standardizes features using **train-set statistics only**. If eval-set statistics leak into the scaler, the probe has already eaten forbidden fruit.

Logistic regression answers: “Is there any linear separator that predicts this label?”

### Mass-mean direction

The mass-mean probe is the difference between the true-class mean and the false-class mean, thresholded at the projected midpoint. It is less flexible than logistic regression, but it gives a simple direction that Lab 7 can inject causally.

Mass-mean answers: “Is the class difference visible in the mean geometry?”

These probes can disagree. That disagreement is not noise. It is one of the lab’s core artifacts.

## The controls

| Control or diagnostic | What it catches |
|---|---|
| grouped train/eval split | paired-template leakage across train and eval |
| shuffled-label refits | probe capacity finding structure in noise |
| random-direction baseline | generic high-dimensional direction luck |
| token-length baseline | truth secretly encoded by statement length |
| majority baseline | class imbalance masquerading as accuracy |
| surface track | trivial decodability on the same axes |
| family-held-out transfer | whether a direction generalizes beyond one template family |
| negation transfer | whether the probe learned truth or a surface polarity/template feature |
| activation-norm diagnostics | single-row geometry hijacking the mean direction |
| calibration curve | whether the logistic probe’s confidence means anything |
| selectivity | real accuracy minus shuffled-label control |

**Make the concept pop:** Look at `plots/selectivity_by_layer.png` and `tables/selectivity_report.csv`. The layer where raw accuracy is highest is often not the layer where selectivity (real minus shuffled) is highest. That gap is the first warning that “a probe found it” is cheap. Then look at the generalization matrix for the negation row: below-chance transfer is frequently the most informative result in the lab.

## The split is now part of the instrument

The original lab used a deterministic hash split by `statement_id`. That is stable, but it can split paired variants across train and eval. For truth probes, this matters. If `Paris` variants land in both splits, a probe can look clever while reading entity or template structure.

The revised code computes a `split_key` before assigning train/eval:

- city and negation statements are grouped by city;
- comparison statements are grouped by the unordered pair of compared quantities;
- metadata is used as a fallback when it is available;
- otherwise, common label suffixes are stripped from the statement id.

Read `diagnostics/split_audit.csv` before trusting the probe. If the split audit looks wrong, the rest of the lab is likely measuring leakage.

## The outlier specimen

On Olmo-3, `The city of Havana is in the Netherlands.` was the teaching specimen: its final-position activation had roughly seven times the norm of the surrounding statements in earlier validation. The validation run flagged five total activation-norm outliers in `statement_manifest.csv`; a single high-norm row was enough to bend the mass-mean direction and pin accuracy to chance while logistic regression looked fine.

The code now unit-normalizes each statement’s activation row before fitting probes by default. It also writes raw norm diagnostics to:

- `tables/statement_manifest.csv`
- `plots/activation_norms_by_depth.png`

For the writeup, flip `NORMALIZE_ROWS = False` and rerun once. Watching the mass-mean direction get dominated by one rogue row is the failure mode this lab is meant to expose. The surface track on the same activations shows that even a trivial feature can look “deep” if you only look at raw accuracy.

## Running it

```bash
python interp_bench.py --lab lab4 --tier a
python interp_bench.py --lab lab4 --tier b --prompt-set full
```

In this lab, `--max-examples` is a **per-family** cap and is balanced true/false. `--prompt-set small|medium|full` selects 20, 40, or all statements per family unless `--max-examples` overrides it.

## What the code now writes

### Core tables

| Artifact | Purpose |
|---|---|
| `tables/probe_report.csv` | long-form probe results, controls, and calibration metrics where defined |
| `results.csv` | standard-course alias of the probe report |
| `tables/selectivity_report.csv` | real-minus-shuffled evidence by depth, family, and probe type |
| `tables/statement_manifest.csv` | statement, split key, split, label, final token, raw norms, outlier flag |
| `tables/calibration_summary.csv` | peak-depth logistic accuracy, Brier score, NLL, and ECE |
| `tables/calibration_curve.csv` | reliability-curve bins for the peak-depth logistic truth probe |

### Diagnostics

| Artifact | Purpose |
|---|---|
| `diagnostics/frozen_data_manifest.json` | data file names, counts, hashes, and normalization convention |
| `diagnostics/split_audit.csv` | group-level train/eval split audit |
| bench hook and lens diagnostics | confirms residual-stream capture and lens conventions before science starts |

### Saved direction for Lab 7

| Artifact | Purpose |
|---|---|
| `tables/truth_direction.pt` | mass-mean vector and threshold, plus metadata |
| `tables/truth_direction_metadata.json` | readable metadata for the tensor artifact |
| `tables/truth_direction_card.md` | human-readable card explaining convention, transfer, caveats, and Lab 7 usage |

### Plots

| Plot | What to look for |
|---|---|
| `plots/decodability_by_layer.png` | truth versus surface versus controls on the same axes |
| `plots/generalization_matrix.png` | within-family and cross-family transfer for logistic and mass-mean |
| `plots/selectivity_by_layer.png` | real accuracy after subtracting the shuffled-label control |
| `plots/truth_projection_panels.png` | visual separation along the mass-mean direction over depth |
| `plots/activation_norms_by_depth.png` | median, p95, and max raw stream norms by depth |
| `plots/truth_calibration_curve.png` | whether logistic confidence tracks empirical truth rate |

## First artifact-reading path

1. Open `probe_claim_card.md`. It tells you the verdict, the non-claim, and which controls were checked.
2. Open `diagnostics/split_audit.csv`. Confirm that paired variants are not split across train/eval.
3. Open `tables/statement_manifest.csv`. Look at `split_key`, `final_token_text`, and `norm_outlier`.
4. Open `plots/decodability_by_layer.png`. Ask whether truth beats shuffled labels, random directions, and the surface track in a meaningful way.
5. Open `plots/generalization_matrix.png`. Explain the negation row and column. Below-chance can mean anti-correlation, not absence of information.
6. Open `plots/selectivity_by_layer.png` and `tables/selectivity_report.csv`. Accuracy without selectivity is just a shiny coin found in a couch.
7. Open `plots/truth_calibration_curve.png`. A probe can be accurate and badly calibrated.
8. Open `tables/truth_direction_card.md` before Lab 7. The card tells you exactly what the saved vector is and is not — and what hypothesis Lab 7 will actually test with it (decodable on these families does not mean the model uses it for truth).

**Make the concept pop:** In `tables/selectivity_report.csv`, find the best layer by raw accuracy vs by selectivity. The gap (or the layer where selectivity is high) is the lesson. Then look at the generalization matrix for the negation row: below-chance transfer is often the most informative result in the entire lab. A direction that is decodable on cities but inert or anti-predictive on negations is the canonical "accessible information" vs "used for the concept" demonstration.

## How the best direction is chosen

The saved vector is a mass-mean direction. The layer is selected by worst-case cross-family transfer between the two affirmative families, `cities` and `comparisons`. Negations are excluded from this layer-selection criterion because negation inversion is an expected result to report, not something to optimize away.

After the layer is chosen, the saved train family is selected between `cities` and `comparisons` by worst-case transfer across the other families, including negations. The saved metadata records:

- model id and dimensionality;
- stream-depth convention;
- final-token position;
- row-normalization convention;
- train family and train statement ids;
- within and cross-family accuracies.

## What the claims may say

Good Lab 4 claim:

```text
DECODE: At depth k, a logistic probe decodes truth from final-token residual
streams on these frozen statement families with accuracy A, compared with
shuffled control B, random-direction control C, length baseline D, and grouped
family-held-out transfer E.
```

Bad Lab 4 claim:

```text
The model knows which statements are true at layer k.
```

That claim smuggles in use, belief, and behavior. Lab 4 has not earned any of those. Put the shiny contraband back in the epistemic drawer.

## Writeup questions

1. At what depth does truth become decodable by logistic regression? At what depth does the mass-mean direction peak? Explain the difference.
2. What is the peak selectivity, not just the peak accuracy? Quote the shuffled-label control.
3. Does truth transfer from `cities` to `comparisons`? Does either affirmative family anti-predict `negations`? Explain what below-chance means here.
4. Which statement rows, if any, are activation-norm outliers? What changes when you set `NORMALIZE_ROWS = False` and rerun?
5. Is the logistic truth probe calibrated at its peak depth? Use Brier score, ECE, and the calibration curve.
6. What does the surface track prove about the phrase “linearly decodable”? How does it limit your truth-probe claim?
7. Write the precise Lab 7 hypothesis that `truth_direction.pt` should test, including a falsifier.

## Symptom-first debugging

| Symptom | First place to look |
|---|---|
| Layer 0 truth accuracy is suspiciously high | `diagnostics/split_audit.csv`; paired variants may be leaking |
| Mass-mean at chance, logistic strong | `tables/statement_manifest.csv` and `plots/activation_norms_by_depth.png` |
| Everything at chance | final-token position, tokenizer behavior, and hook/lens diagnostics |
| Surface track crashes or looks constant | selected surface letter and split balance in `statement_manifest.csv` |
| Negation transfer is about 0.5 | chance, often fine |
| Negation transfer is far below 0.5 | structured inversion, explain it rather than hiding it |
| Length baseline is high | dataset artifact, weaken the truth claim |
| Calibration curve is crooked | probe may rank examples well without giving meaningful confidence |
| `Frozen dataset missing` | re-checkout `data/`; do not regenerate per-run |

## What goes in the ledger

Write two or three claims, all tagged `DECODE`:

1. truth is decodable with controls;
2. the mass-mean direction transfers or fails to transfer across specified families;
3. the surface track shows why decodability alone is cheap.

Every claim should name the model, statement families, stream depth, probe type, metric, control, artifact, and falsifier. The ledger is an audit trail, not a place for unsourced conclusions.
