# Lab 26: Causal Abstraction by Residual Resampling

**One-sentence thesis:** A high-level explanation only earns the name when preserving its variables preserves behavior better than breaking those variables or using matched controls.

**Time estimate:** Tier A smoke in minutes on CPU; Tier B science on the course base model in a Colab/A100-style runtime.

**Compute tier:** Tier A checks plumbing on `gpt2`; Tier B is the real evidence path on the course base model.

**Dependencies:** Labs 4, 5, 12, and the claim-ledger discipline from Labs 1-25. Lab 27 continues with path-specific mediation.

**Minimum passing artifacts:** `method_card.md`, `causal_abstraction_spec.md`, `operationalization_audit.md`, `metrics.json`, `results.csv`, `results.jsonl`, `tables/evidence_matrix.csv`, `tables/counterexamples.csv`, and `plots/plot_reading_guide.csv`.

**Main plot:** `plots/causal_abstraction_dashboard.png`

**Main table:** `tables/evidence_matrix.csv`

**Evidence rung:** `FORMAL + CAUSAL + AUDIT`

**Forbidden claim:** "The model implements exactly this algorithm."

**One-sentence allowed claim:** "Under this residual-resampling battery, preserving variable X at site S preserved the target margin more than breaking X or using matched controls, with caveats named in the audit."

**Human-label requirement:** none. This lab is forward-pass-only and next-token-margin based.

## What question this lab asks

Can a proposed high-level explanation survive behavior-preserving resampling, or was it only a tidy story draped over interesting activations?

Before running the main intervention, Lab 26 asks you to write the explanation in a testable form:

```text
high-level variables -> low-level residual sites -> donor rule -> expected behavior
```

Then the lab tries to break it. The point is not to make the plot glow. The point is to see whether the hypothesis still breathes when the variables are preserved, broken, randomized, and patched at the wrong place.

## Why this matters in the course progression

The earlier labs taught the instrument ladder: observation, attribution, decodability, patching, circuits, features, audits, and self-report. Lab 26 changes the object under the microscope. The claim itself becomes the specimen.

A student who can run patching can still overclaim from patching. Lab 26 trains the stricter move: state the abstraction, predeclare the variable mapping, run controls that can embarrass it, and keep the counterexamples. This is the bridge from "I found a causal handle" to "I tested a mechanistic explanation."

Lab 27 will ask which paths carry a behavior. Lab 26 stays narrower on purpose: residual-stream resampling first, path-specific scrubbing next.

## What the experiment measures

The behavior metric is the next-token margin:

```text
logit_diff = logit(target) - logit(distractor)
scrub_score = patched_logit_diff / clean_logit_diff
```

A score near `1.0` means the patched run preserved the clean target margin. A score near `0.0` means the intervention did not preserve the measured behavior. Negative scores mean the patch pushed toward the distractor.

The ratio is only meaningful when the clean margin is healthy. The lab therefore writes `tables/baseline_behavior.csv` and filters formal gates to rows passing the baseline margin threshold.

## The two starter hypotheses

The lab uses two formal JSON specs under `specs/`.

| Spec | Domain | Hypothesis |
|---|---|---|
| `specs/lab26_induction_hypothesis.json` | induction copying | `COPY_SOURCE` and `QUERY_TOKEN` should preserve next-token copy behavior under residual resampling. |
| `specs/lab26_relation_hypothesis.json` | relation answers | `RELATION` should be carried by relation-token streams while the recipient prompt keeps the subject. |

Each spec names:

```text
hypothesis_id
behavior_metric
high_level_variables
low_level_sites
resampling_rules
predicted_preservation_min
predicted_damage_when_broken_min
predicted_specificity_gap_min
```

The lab writes a human-readable copy to `causal_abstraction_spec.md` and a schema/provenance audit to `tables/hypothesis_spec_audit.csv`.

## The dataset contract

The science dataset is:

```text
data/causal_abstraction_tasks.csv
```

Required columns:

```text
item_id, domain, split, high_level_task, template_family,
prompt, target, distractor,
source_token, source_position, target_position,
relation_family, subject, answer,
high_level_variables_json
```

The JSON field is the row-level abstraction contract. Induction rows name variables such as `COPY_SOURCE`, `QUERY_TOKEN`, and `SURFACE_FRAME`. Relation rows name variables such as `RELATION`, `SUBJECT`, `ANSWER_CLASS`, and `SWAP_GROUP`.

Tier A may use a tiny built-in smoke fallback only when the CSV is absent. That fallback is marked in `diagnostics/data_manifest.json` and `metrics.json`. Do not move smoke-fallback claims into the ledger.

## The donor conditions

For each target item, the lab builds donor rows from the JSON spec rather than from ad hoc code. The donor planner writes its receipt to `tables/donor_plan.csv` and coverage to `diagnostics/donor_coverage.csv`.

| Condition | What it does | What it tests |
|---|---|---|
| `no_op_same_example` | Patches the item with its own vector. | The residual patching machinery must be an identity. |
| `preserve_variable` | Uses a donor matching the spec's `preserve` variables and, when possible, varying its `vary` variables. | Whether the abstraction can preserve behavior under surface/context changes. |
| `break_variable` | Uses a donor matching preserved variables while changing the spec's `break` variables. | Whether the named variable matters. |
| `random_matched` | Uses a deterministic same-domain donor, length-matched when possible. | Whether any same-shaped donor helps. |
| `wrong_site_preserve` | Uses the preserving donor at a token position outside the named site. | Whether the nominated site is specific. |

A useful abstraction has this shape:

```text
preserve_variable high
break_variable lower
random_matched lower
wrong_site_preserve lower
no_op_same_example near 1.0
```

If `break_variable` is also high, the variable is probably too broad or the site is downstream of the variable. If `wrong_site_preserve` or `random_matched` is high, the intervention is not specific enough.

## The intervention

At each residual site named by the hypothesis spec, the lab patches the target prompt with a donor vector:

```text
recipient prompt:  item.prompt
donor vector:      donor.streams[depth, donor_position]
patch site:        recipient.streams[depth, target_position]
metric:            logit(target) - logit(distractor)
```

The bench convention still applies:

```text
streams[k] = pre-final-norm residual stream after k blocks
streams[0] = embedding output
streams[L] = final block output before final norm
```

The lab records every cell in `results.csv`, `results.jsonl`, and `tables/resampling_interventions.csv`.

## Claimable depths

The raw tables include all requested depths, but formal gates use only interior stream depths:

```text
1 <= depth < n_layers
```

Depth `0` is often token embedding substitution. Depth `L` is the final-norm input and can behave like a broad readout handle. Both are useful sanity rows, but they do not earn the main abstraction claim.

## Train/eval discipline

The dataset has a `split` column. The refactored runner uses it instead of treating every row as one glittery aggregate.

1. It computes summaries for `train`, `eval`, and `all` rows.
2. It selects the best site/depth on `train` when train rows exist.
3. It reports whether that same site/depth survives on `eval`.
4. It writes the split receipt to `tables/split_generalization_summary.csv`.

A train-only pass is a candidate, not a finished claim. A claim that survives both train and eval earns stronger language. A claim that only passes in the aggregate must be replicated before ledger promotion.

## Self-checks before science plots

The lab inherits the shared bench checks:

| Check | Artifact | Failure meaning |
|---|---|---|
| Hook parity | `diagnostics/hook_parity.json` | The captured stream is not verified. |
| Logit lens self-check | `diagnostics/logit_lens_self_check.json` | The final-depth readout does not reproduce model logits. |
| Patch no-op check | `diagnostics/patch_noop_check.json` | Residual patching does not target the named stream. |

Lab 26 adds local checks:

| Check | Artifact | Failure meaning |
|---|---|---|
| Tokenization gate | `diagnostics/tokenization_gate.csv` | Answer tokens or site positions are not valid for this tokenizer. |
| Donor coverage | `diagnostics/donor_coverage.csv` | A condition is missing for some items. |
| Named-site no-op | `tables/noop_identity_check.csv` | Self-resampling at actual Lab 26 sites changes the logits. |
| Self-check status | `diagnostics/self_check_status.json` | Compact pass/fail status for the lab-local checks. |

The local no-op check is not decorative. If self-resampling moves the clean margin, every causal plot would be a very fancy counterfeit coin.

## What counts as evidence

| Evidence tag | What earns it here | What it cannot show |
|---|---|---|
| `FORMAL` | The JSON spec names variables, sites, donor semantics, and thresholds before the run. | Mathematical proof or correctness of the abstraction. |
| `CAUSAL` | Residual interchange changes the measured target margin under controls. | A full circuit or path-specific mediation. |
| `AUDIT` | The run preserves no-op checks, donor coverage, split failures, counterexamples, and v2 refinements. | That the v2 refinement is already validated. |

The strongest positive sentence is narrow:

```text
FORMAL + CAUSAL: under this residual-resampling battery, preserving variable X at site S preserved the target margin more than breaking X or using matched controls, and the train-selected cell survived eval rows.
```

The forbidden sentence is broad:

```text
The model implements exactly this algorithm.
```

## Running it

Run from `interpretability/`.

```bash
python interp_bench.py --lab lab26 --tier a
python interp_bench.py --lab lab26 --tier b --prompt-set full
```

Useful variants:

```bash
python interp_bench.py --lab lab26 --tier a --no-plots
python interp_bench.py --lab lab26 --tier b --prompt-set medium --max-examples 24
python interp_bench.py --lab lab26 --tier b --prompt-set data/causal_abstraction_tasks.csv
```

Tier A proves the plumbing. Tier B is the evidence path. If Tier A produces an exciting result, treat it as a smoke-test curiosity and rerun Tier B before writing claims.

## Artifact tree

```text
runs/lab26_causal_abstraction-*/
  run_summary.md
  method_card.md
  causal_abstraction_spec.md
  operationalization_audit.md
  metrics.json
  results.csv
  results.jsonl
  ledger_suggestions.md

  diagnostics/
    data_manifest.json
    tokenization_gate.csv
    donor_coverage.csv
    self_check_status.json
    hook_parity.json
    logit_lens_self_check.json
    patch_noop_check.json

  tables/
    hypothesis_spec_audit.csv
    baseline_behavior.csv
    donor_plan.csv
    noop_identity_check.csv
    resampling_interventions.csv
    variable_preservation_summary.csv
    best_hypothesis_cells.csv
    split_generalization_summary.csv
    evidence_matrix.csv
    counterexamples.csv
    hypothesis_refinement_log.csv

  plots/
    plot_reading_guide.csv
    causal_abstraction_dashboard.png
    resampling_preservation_matrix.png
    hypothesis_pass_fail_atlas.png
    variable_specificity_ladder.png
    split_generalization_ladder.png
    counterexample_gallery.png

  state/
    hypothesis_specs_used.json
```

## Artifact reading path

Start with `method_card.md`. It tells you whether the run is science-ready and whether each hypothesis earned positive, train-only, aggregate-only, or negative language.

Then read in this order:

1. `causal_abstraction_spec.md`: What was the hypothesis before the run?
2. `diagnostics/tokenization_gate.csv`: Did the row positions mean what the CSV said they meant?
3. `tables/baseline_behavior.csv`: Did the model have a clean margin worth preserving?
4. `tables/donor_plan.csv`: Did the donors actually preserve or break the named variables?
5. `tables/noop_identity_check.csv`: Did self-resampling stay near identity?
6. `tables/evidence_matrix.csv`: What is the smallest claim that survived?
7. `tables/split_generalization_summary.csv`: Did the train-selected cell survive eval?
8. `tables/counterexamples.csv`: Which row most embarrasses the favorite explanation?
9. `operationalization_audit.md`: What language is allowed?
10. `ledger_suggestions.md`: Drafts only. Edit before appending.

## Plot guide

### `causal_abstraction_dashboard.png`

Read this first. It combines train-selected preservation, break-variable damage, specificity gap, baseline health, and counterexample load. It is a cockpit, not a verdict machine.

### `resampling_preservation_matrix.png`

Rows are claimable site/depth cells. Columns are donor conditions. A good row has a high preserving column and lower break/random/wrong-site columns.

### `hypothesis_pass_fail_atlas.png`

This shows the formal gates across train, eval, and aggregate summaries. A failed cell is a useful measurement, not a failed lab.

### `variable_specificity_ladder.png`

This shows the selected cell's preservation score beside break-variable and control scores. The gap is the rent the hypothesis pays.

### `split_generalization_ladder.png`

This asks whether the train-selected cell survives eval. This is where many pretty aggregate stories lose their shoes.

### `counterexample_gallery.png`

This plot surfaces the top rows that shrink or kill the claim. Read the CSV for full prompts and donor details.

## Interpreting result patterns

| Pattern | Interpretation |
|---|---|
| Preserve high, break low, controls low, eval pass | Narrow positive result for this residual-resampling battery. |
| Preserve high, break high | The variable is too broad or the site carries downstream information. |
| Preserve high, wrong-site high | The nominated site is not specific enough. |
| Preserve high, random high | The patch may be perturbing the stream in a helpful but non-variable-specific way. |
| Train pass, eval fail | Candidate mechanism story, not ledger-ready positive claim. |
| Aggregate pass, no split pass | Possible specimen-level story. Replicate with a clean split. |
| Preserve low | The proposed mapping did not preserve behavior under this instrument. That is a clean negative result. |
| No-op fails | Stop. Fix instrumentation before reading plots. |

## What this lab can claim

It can claim that a formal variable mapping survived or failed a residual-stream resampling battery on a named model, dataset, metric, site, depth, and split.

It can claim that preserving donors preserved more target margin than break-variable or control donors, if the numbers show that.

It can claim that counterexamples require a narrower v2 hypothesis.

## What this lab cannot claim

It cannot claim that the model implements the high-level algorithm.

It cannot claim that the selected residual site is the full circuit.

It cannot claim path-specific mediation.

It cannot claim that a v2 refinement has been validated before rerunning it.

It cannot claim anything general outside the prompt family without replication.

## Common failure modes

| Symptom | Likely cause | What to inspect |
|---|---|---|
| Tokenization gate drops many rows | Target/distractor not single-token for this tokenizer, or positions assume a different special-token convention. | `diagnostics/tokenization_gate.csv` |
| Every scrub score is unstable | Clean margins are too small. | `tables/baseline_behavior.csv` |
| Missing preserving or broken donors | The selected prompt slice is too small or unbalanced. | `diagnostics/donor_coverage.csv` |
| No-op rows are not near 1.0 | Hook site, token position, or prompt encoding mismatch. | `tables/noop_identity_check.csv` |
| Wrong-site is high | The site is not specific, or the wrong-site position is not a strong enough negative control. | `tables/resampling_interventions.csv` |
| Break-variable is high | The hypothesis variable is too coarse or the patched site carries downstream answer evidence. | `tables/counterexamples.csv` |
| Train passes but eval fails | The best cell may have overfit the discovery split. | `tables/split_generalization_summary.csv` |
| Tier A looks better than Tier B | Tiny-model artifact or easier smoke data. | `metrics.json`, `diagnostics/data_manifest.json` |

## Writeup questions

1. What high-level variables did the spec name before the run?
2. Which low-level residual sites did it nominate, and why?
3. What was the clean baseline pass rate by domain and split?
4. Which site/depth was selected on train, and what selection rule chose it?
5. How much did preserving donors preserve at that cell?
6. How much did break-variable donors preserve at that cell?
7. Which control was closer to the preserving donor: random or wrong-site?
8. Did the same selected cell survive eval rows?
9. What is the strongest row in `tables/counterexamples.csv`?
10. Does the result support v1, a narrower v2, or a negative result?
11. What exact held-out run would falsify your favored claim?
12. Write one allowed claim and one forbidden overclaim.

## Ledger templates

Positive, when train and eval both pass:

```text
[L26-C1] FORMAL+CAUSAL | For <domain> prompts in Lab 26 on <model>, hypothesis <H> survived residual resampling at <site> depth <k>: train preserving donors scored <x> vs break-variable <y>, and eval preserving donors scored <x2> vs break-variable <y2>. This is a residual-resampling claim, not an algorithm-identity claim.
Artifact: runs/<run>/tables/evidence_matrix.csv | Falsifier: a held-out run where preserving donors no longer beat break-variable and wrong-site/random controls.
```

Negative or refinement result:

```text
[L26-C2] FORMAL+CAUSAL,AUDIT | Hypothesis <H> did not earn the positive abstraction claim because <failed gate>. The supported next claim is narrower: <v2 scope>.
Artifact: runs/<run>/tables/hypothesis_refinement_log.csv | Falsifier: a rerun of the stated v2 on held-out items that still fails the same control.
```

Forbidden:

```text
The model implements this algorithm.
This site is the whole circuit.
The abstraction works generally.
The v2 refinement is validated because the log suggested it.
```

## Suggested extensions

Replace the provided JSON specs with a student-authored hypothesis and rerun without changing thresholds after seeing results.

Run the same spec on `gpt2` and the course base model, then compare which counterexamples are stable.

Add Lab 12 relation families beyond `country_sem`, but keep matched relation-swap controls.

Save the train/eval-supported site as the starting hypothesis for Lab 27 path-specific mediation.

Add a stricter donor-matching rule that preserves token length, answer class, and template family simultaneously, then test whether the claim survives the narrower donor pool.
