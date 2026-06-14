# Lab 32: Reward Models and Preference Circuits

**One-sentence thesis:** A reward or preference signal earns scientific language only when it beats the boring shortcuts that make "preferred" look easy.

**Time estimate:** 75-100 minutes for Tier A smoke; longer for Tier B on the full pair suite.

**Compute tier:** Tier A uses `gpt2` with a DPO-style log-prob-ratio proxy. Tier B uses the course base model on the same forward-pass workflow. External reward-model substitution is an extension, not the default evidence path.

**Dependencies:** Labs 7, 16, 17, 19, 21, and 31 concepts.

**Minimum passing artifacts:** `method_card.md`, `operationalization_audit.md`, `diagnostics/safety_status.json`, `diagnostics/self_check_status.json`, `diagnostics/warning_summary.csv`, `diagnostics/lab32_run_config_snapshot.json`, `tables/preference_pair_scores.csv`, `tables/preference_probe_report.csv`, `tables/reward_policy_disagreements.csv`, `tables/preference_intervention_results.csv`, `tables/preference_evidence_matrix.csv`, `tables/failure_specimens.md`, `plots/plot_manifest.json`, `tables/figure_sources/*.csv`, and `plots/preference_evidence_dashboard.png`.

**Main plot:** `plots/preference_evidence_dashboard.png`

**Main table:** `tables/preference_evidence_matrix.csv`

**Evidence rung:** `ATTR + DECODE + CAUSAL + AUDIT`

**Forbidden claim:** "The reward model understands human values."

**One-sentence allowed claim:** "On this benign pair suite, preference signal S separated preferred from dispreferred responses on eval rows with AUC X, beating shortcut controls by Y."

**Human-label requirement:** Fill the shared review columns before citing any row-level preference label in a writeup.

## What question this lab asks

Preference training and reward modeling are easy to inflate into value talk. Lab 32 opens a smaller box:

```text
paired responses -> preference proxy -> shortcut controls -> residual direction -> causal nudge -> audit
```

The target is not "values." The target is whether a measured preference signal survives better explanations such as length, sentiment, politeness, agreement, hedging, refusal words, split overfit, or A/B prompt artifacts.

A good result is not "the model wants the good answer." A good result is closer to:

```text
On this benign pair suite, this score separated preferred from dispreferred responses on eval rows better than the named shortcut controls.
```

That sentence is smaller, sturdier, and much less likely to turn a measurement into a personality or values claim.

## Why this matters in the course progression

Earlier labs studied truth, refusal, sycophancy, persona, editing, and feature lineage. Lab 32 looks at a training signal that often sits upstream of those behaviors: preference or reward.

This lab keeps the instrument modest. It does not require a trained reward model. Tier A uses a DPO-style policy/reference log-prob-ratio proxy and treats that proxy as a specimen to audit. The purpose is to teach students how to examine a preference-like signal without treating it as a moral oracle.

## Safety and scope

The default dataset uses only benign pairs:

- helpfulness and concrete advice;
- factual honesty;
- appropriate uncertainty;
- privacy-boundary handling with synthetic situations;
- style and instruction following;
- anti-sycophancy controls where the user states a harmless false belief.

The lab does not do harmful prompt optimization, jailbreak search, refusal ablation, private-data recovery, or deployment reward-model claims. `diagnostics/safety_status.json` records this scope for every run.

## The data contract

Default data lives in:

```text
data/preference_circuit_pairs.csv
```

Required columns:

```text
pair_id,domain,prompt,response_a,response_b,preferred,
preference_type,confound_type,split,notes
```

The refactored data generator is:

```text
data/make_preference_circuit_pairs.py
```

The frozen suite is balanced across train/eval rows. The train split is used to fit residual directions and select a stream depth. Eval rows are the first place positive language is allowed to stretch its legs.

## What the experiment measures

### 1. DPO-style preference proxy, `ATTR`

Tier A uses:

```text
(policy log p(response A) - reference log p(response A))
-
(policy log p(response B) - reference log p(response B))
```

The reference is a response-unconditional unigram token baseline fit from train responses when train rows exist. This is not a trained reward model. It is a cheap, falsifiable preference proxy.

### 2. Shortcut controls, `AUDIT`

Every pair is also scored by transparent shortcut rulers:

```text
length
politeness
agreement
sentiment
hedging
refusal
policy log-prob alone
reference unigram alone
shuffled-score control
```

If a shortcut beats the preference proxy on eval rows, the shortcut gets the crown and the favorite interpretation goes back to the workshop.

### 3. Residual preference direction, `DECODE`

For each pair, the lab caches response-boundary residual vectors:

```text
diff(pair, depth) = residual(prompt + response A) - residual(prompt + response B)
```

The preference direction is fit on train rows:

```text
preference_direction(depth)
  = mean over train pairs of signed diff(pair, depth)
```

The sign is positive for the preferred response and negative for the rejected response. The lab scans a coarse depth grid, selects the depth on train by AUC minus best confound/random direction, and only then evaluates that selected depth on eval rows.

### 4. Confound directions, `AUDIT`

The lab also builds residual directions for length, politeness, agreement, sentiment, hedging, and refusal. These are not decoration. They are the shortcut tribunal. A preference direction only gets positive language if it beats them.

### 5. Activation addition, `CAUSAL`

The lab builds an A/B judge prompt:

```text
Prompt:
<user prompt>

Response A:
<response A>

Response B:
<response B>

Which response is better? Answer
```

It adds the selected preference direction at the corresponding block output and measures:

```text
logit(" A") - logit(" B")
```

The causal claim is narrow. A shift in this judge prompt is evidence that the direction can move this readout. It is not evidence that the model will behave better in open generation.

## Train/eval discipline

The refactor fixes the biggest weakness in the draft: the old runner trained and evaluated residual directions on the same pool.

The new runner writes:

```text
tables/preference_depth_selection.csv
tables/preference_direction_by_depth.csv
tables/split_generalization_summary.csv
```

Use these before writing any positive claim. The strongest posture is eval-supported. A train-only result is a candidate. An aggregate-only result is not a finished claim.

## Controls and falsifiers

| Control | What it attacks | Artifact |
|---|---|---|
| Length score | Verbosity detector | `tables/preference_probe_report.csv` |
| Politeness score | Nice-tone detector | `tables/preference_probe_report.csv` |
| Agreement score | Sycophancy detector | `tables/sycophancy_risk_review.csv` |
| Sentiment score | Positivity detector | `tables/shortcut_control_summary.csv` |
| Hedging score | Uncertainty-language detector | `tables/preference_probe_report.csv` |
| Refusal score | Boundary-word detector | `tables/preference_probe_report.csv` |
| Confound residual directions | Internal shortcut handles | `tables/preference_direction_by_depth.csv` |
| Random direction | Direction overfit and steering artifact | `tables/preference_intervention_results.csv` |
| Shuffled score | Label/row-order sanity control | `tables/preference_probe_report.csv` |
| Human review queue | Label ambiguity and row-level caveats | `tables/human_review_queue.csv` |

## Running it

From `interpretability/`:

```bash
python interp_bench.py --lab lab32 --tier a --no-plots
python interp_bench.py --lab lab32 --tier a
python interp_bench.py --lab lab32 --tier b --prompt-set full
```

Useful variants:

```bash
python interp_bench.py --lab lab32 --tier b --prompt-set medium --no-plots
python interp_bench.py --lab lab32 --tier b --prompt-set data/preference_circuit_pairs.csv
python interp_bench.py --lab lab32 --tier a --max-examples 12 --no-plots
```

Tier A proves the rails. Tier B is the evidence path. Do not ledger a broad preference claim from a smoke-only fallback run.

## Artifact tree

```text
runs/lab32_reward_preference-*/
  run_summary.md
  method_card.md
  operationalization_audit.md
  metrics.json
  results.csv
  results.jsonl
  ledger_suggestions.md

  diagnostics/
    data_manifest.json
    safety_status.json
    self_check_status.json
    warning_summary.csv
    warning_summary.json
    lab32_run_config_snapshot.json
    tokenization_gate.csv
    hook_parity.json
    logit_lens_self_check.json
    patch_noop_check.json

  tables/
    preference_pair_scores.csv
    preference_probe_report.csv
    preference_depth_selection.csv
    preference_direction_by_depth.csv
    split_generalization_summary.csv
    confound_audit_by_type.csv
    shortcut_control_summary.csv
    sycophancy_risk_review.csv
    reward_policy_disagreements.csv
    preference_counterexamples.csv
    counterexamples.csv
    failure_specimens.jsonl
    failure_specimens.md
    human_review_queue.csv
    preference_intervention_results.csv
    preference_intervention_summary.csv
    preference_evidence_matrix.csv
    figure_sources/*.csv

  plots/
    plot_reading_guide.csv
    plot_manifest.json
    plot_manifest.csv
    preference_evidence_dashboard.png
    overview_dashboard.png
    target_vs_control.png
    dose_response.png
    layer_sweep_heatmap.png
    paired_examples.png
    reward_margin_by_domain.png
    preference_probe_control_atlas.png
    confound_specificity_ladder.png
    reward_policy_disagreement_matrix.png
    preference_steering_frontier.png
    sycophancy_reward_risk_quadrant.png
    judge_prompt_swap_control.png

  state/
    preference_directions.pt
    preference_direction_metadata.json
```

## Reading path

Start with `method_card.md`. It tells you what the run is allowed to mean.

Then read:

1. `diagnostics/safety_status.json`: confirms the safe scope.
2. `diagnostics/self_check_status.json`: confirms tokenization, splits, direction norm, and review columns.
3. `tables/preference_depth_selection.csv`: confirms the selected direction depth was picked on train.
4. `tables/preference_probe_report.csv`: compares proxy, residual direction, shortcuts, and controls.
5. `tables/confound_audit_by_type.csv`: names domain and confound families where shortcuts may explain the result.
6. `tables/split_generalization_summary.csv`: asks whether train-selected evidence survived eval.
7. `tables/preference_intervention_results.csv`: checks the activation-addition nudge.
8. `tables/preference_counterexamples.csv`: rows that shrink or kill the favorite claim.
9. `tables/reward_policy_disagreements.csv`: rows that should make you cautious.
10. `tables/human_review_queue.csv`: fill this before citing row-level labels.
11. `operationalization_audit.md`: the claim grammar with its guardrails on.

## How to read the upgraded figure suite

Every major plot is backed by a source table under `tables/figure_sources/`. Open `plots/plot_manifest.json` before citing a figure: it records the figure path, source table, row count, metric family, control, question answered, and claim boundary.

Read the suite in this order:

1. `preference_evidence_dashboard.png`: the overview. It tells you where to inspect, not what to claim.
2. `target_vs_control.png`: the main scorecard. If a shortcut or null control beats the favorite signal, that is the result.
3. `layer_sweep_heatmap.png`: train depth selection and eval survival. Train-only brightness is candidate evidence.
4. `dose_response.png`: raw intervention rows and mean response by activation-addition scale. A single lucky point is not enough.
5. `paired_examples.png`: raw pair-level proxy, residual-direction, and strongest-shortcut margins.
6. `failure_specimens.md`: rows that shrink the claim and require human review.

### Plot catalog

| Figure | Source table | Question | Honest negative result |
|---|---|---|---|
| `overview_dashboard.png` | `tables/figure_sources/overview_dashboard_source.csv` | What compact claim posture did the run earn? | Every row is shortcut-limited, random-control, or smoke-only. |
| `target_vs_control.png` | `tables/figure_sources/target_vs_control_source.csv` | Do target signals beat shortcuts? | Length, agreement, sentiment, refusal, random, or shuffled controls beat the target. |
| `dose_response.png` | `tables/figure_sources/dose_response_source.csv` | Does the intervention separate from random across scale? | Preference and random directions move the judge prompt similarly. |
| `layer_sweep_heatmap.png` | `tables/figure_sources/layer_sweep_heatmap_source.csv` | Did train-selected depth survive eval? | Train lift is bright but eval lift collapses. |
| `paired_examples.png` | `tables/figure_sources/paired_examples_source.csv` | Which rows carry or contradict the aggregate? | A few rows dominate, or the strongest shortcut exceeds the preference margin. |
| `sycophancy_reward_risk_quadrant.png` | `tables/figure_sources/sycophancy_reward_risk_quadrant_source.csv` | Does agreement look rewarded on false-belief rows? | Agreement and proxy margins are positive together on anti-sycophancy rows. |


## Plot guide

### `preference_evidence_dashboard.png`

The dashboard shows eval AUCs, shortcut gaps, steering shift, and disagreement load. It is an index into the evidence, not the verdict machine.

### `overview_dashboard.png`

This compact dashboard shows the claim posture row by row. Treat it as a table of contents for `preference_evidence_matrix.csv`.

### `target_vs_control.png`

This is the main comparison plot. Preference proxy and residual direction must be read beside length, politeness, agreement, sentiment, hedging, refusal, random, and shuffled controls.

### `dose_response.png`

This shows raw intervention rows plus mean and confidence intervals by activation-addition scale. Look for separation from random direction across scale, not one dramatic point.

### `layer_sweep_heatmap.png`

This shows train depth selection and eval survival. A strong train row with weak eval lift is a train-only candidate.

### `paired_examples.png`

This scatter keeps row-level variation visible. Points in the wrong quadrant or annotated for review should be read before any aggregate claim is written.

### `reward_margin_by_domain.png`

This shows DPO proxy preferred margin by domain and split. It helps distinguish broad signal from one-domain glitter.

### `preference_probe_control_atlas.png`

This is the main control atlas. A good preference signal should sit above the shortcut family on eval rows.

### `confound_specificity_ladder.png`

This is the rent ledger. If length, agreement, or refusal reaches the top rung, your story must name that.

### `reward_policy_disagreement_matrix.png`

This shows where the proxy predicts A/B differently from the frozen labels. Read the rows before trusting aggregate scores.

### `preference_steering_frontier.png`

This asks whether the preference direction moves the A/B judge margin more than a random direction. It is a narrow causal result.

### `sycophancy_reward_risk_quadrant.png`

This isolates false-agreement rows. A supported preference signal should not reward agreeing with harmless false beliefs.

### `judge_prompt_swap_control.png`

This checks whether the intervention is stable under A/B response order swaps. A response-order-sensitive effect should be described as a judge-prompt artifact until proven otherwise.

### `failure_specimens.md` and `failure_specimens.jsonl`

These are not optional leftovers. They are the rows that force the claim to become smaller.

## Expected result patterns

| Pattern | Interpretation |
|---|---|
| Proxy and direction beat shortcuts on eval | Narrow positive preference-signal claim. |
| Proxy high, length high | Preference proxy is probably a verbosity detector. |
| Direction high on train, weak on eval | Train-only candidate, not claim-ready. |
| Agreement high on anti-sycophancy rows | The signal confuses agreement with preference. |
| Refusal high everywhere | Boundary language may be acting as a generic reward token. |
| Activation addition shifts more than random | Narrow causal handle on the A/B judge prompt. |
| Random shift matches preference shift | Judge-prompt artifact or broad perturbation. |
| Review queue is large | Human labels or row wording need attention before claim writing. |

## What this lab can claim

It can claim that a named score separated preferred from dispreferred benign pairs on a named model, split, and dataset.

It can claim that a train-fit residual direction decoded pair labels on eval rows beyond named shortcut controls.

It can claim that activation addition shifted a specific A/B judge-prompt margin more than a random direction, if the numbers show that.

## What this lab cannot claim

It cannot claim that a reward model understands human values.

It cannot claim that a residual direction is a morality vector.

It cannot claim that optimizing the proxy is safe.

It cannot claim that the model wants anything.

It cannot claim deployment safety or general preference robustness outside this benign pair suite.

## Common failure modes

### Reward direction is length

If length beats the preference score, the proxy is mostly a verbosity detector. This is a result, not a bug.

### Reward direction is agreement

False-belief sycophancy rows test whether agreeing with the user is mistaken for better behavior.

### Reward direction is tone

Sentiment and politeness can look like helpfulness. A supported preference claim must survive those controls.

### Refusal becomes a magic token

Privacy rows reward boundaries, but not every good answer is a refusal. Consent-based help rows are there to catch over-refusal.

### Judge-prompt artifact

Activation addition is tested on an A/B preference prompt. A shift there is not evidence that open-ended model behavior improves.

### Label ambiguity

The frozen labels are teaching labels. Before citing one row in a writeup, fill the review columns in `human_review_queue.csv`: `student_label_primary`, `student_label_secondary`, `student_confidence`, `student_evidence_span`, `reviewer_label`, and `agreement_status`.

## Writeup questions

1. Did the DPO proxy beat the best eval shortcut control?
2. Which shortcut was strongest on eval rows?
3. Did the train-selected residual depth survive eval?
4. Did the preference residual direction beat confound directions?
5. Which anti-sycophancy row is most concerning?
6. Did activation addition create a causal shift beyond random?
7. Which row in `reward_policy_disagreements.csv` most weakens the favorite claim?
8. What is the smallest claim you can make without saying "values"?

## Claim grammar

Allowed:

```text
ATTR + DECODE + CAUSAL: On benign pair suite T, preference signal S separated
preferred from dispreferred responses on eval rows with AUC X, beat the best
shortcut control by Y, and activation addition shifted the A/B judge-prompt
margin by Z over random. This is a preference-signal claim, not a values claim.
```

Forbidden:

```text
The reward model understands human values.
The model wants to be helpful.
The reward direction is a morality vector.
The preference proxy is safe to optimize.
```

## Verification checklist

Run from `interpretability/`:

```bash
python -m py_compile interp_bench.py labs/lab32_reward_preference.py
python interp_bench.py --lab lab32 --tier a --no-plots
python interp_bench.py --lab lab32 --tier a
python interp_bench.py --lab lab32 --tier b --prompt-set full
```

Expected upgraded artifacts:

```text
plots/plot_manifest.json
plots/overview_dashboard.png
plots/target_vs_control.png
plots/dose_response.png
plots/layer_sweep_heatmap.png
plots/paired_examples.png
tables/figure_sources/target_vs_control_source.csv
tables/figure_sources/dose_response_source.csv
tables/figure_sources/layer_sweep_heatmap_source.csv
tables/failure_specimens.md
diagnostics/warning_summary.csv
diagnostics/lab32_run_config_snapshot.json
```

The plot pass succeeded when every cited figure has a non-empty source table, controls remain visible beside targets, raw intervention rows are available for dose plots, and the writeup gets smaller when a shortcut wins.

## Suggested extensions

Replace the DPO-style proxy with an open reward model and keep the same shortcut-control table.

Compare base and instruction-tuned policy log-prob margins as a training-diff signal.

Use Lab 31 automated-interpretability features to label top response-boundary contexts, then test those labels against hard negatives.

Add a hand-reviewed calibration subset and report human-label agreement before using row-level claims.

Test whether the selected direction transfers to a fresh benign preference suite with different prompts and response lengths.
