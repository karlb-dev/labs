# Lab 26 - Causal Abstraction and Causal Scrubbing

## Question

Can a proposed high-level explanation survive behavior-preserving resampling, or was it only a loose story about interesting activations?

Labs 1-25 taught the instrument ladder: readouts, attribution, probes, patching, circuits, features, audits, and self-report. Lab 26 asks for a stricter move. Before running the main intervention, write down the abstraction you think the model is using:

```text
high-level variables -> low-level sites -> resampling rule -> expected behavior
```

Then try to break it.

This first implementation uses residual-stream resampling, not full path-specific causal scrubbing. That is intentional. Lab 26 is about formalizing and pressure-testing the hypothesis; Lab 27 will ask which paths carry it.

## Run

From `interpretability/`:

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

Tier A uses `gpt2` and a balanced small slice across the two domains. Tier B uses the course base model and the full dataset unless you cap it.

## What You Test

The lab ships two formal hypothesis specs:

| Spec | Domain | Abstraction |
|---|---|---|
| `specs/lab26_induction_hypothesis.json` | induction copying | `COPY_SOURCE` and `QUERY_TOKEN` should be enough to preserve the next-token copy behavior under residual resampling |
| `specs/lab26_relation_hypothesis.json` | relation answers | `RELATION` should be carried by relation-token residual streams while the recipient prompt keeps the subject |

The dataset is `data/causal_abstraction_tasks.csv`. Each row includes:

```text
item_id, domain, split, high_level_task, template_family,
prompt, target, distractor,
source_token, source_position, target_position,
relation_family, subject, answer,
high_level_variables_json
```

The JSON field is the abstraction contract. For induction, it names variables like `COPY_SOURCE`, `QUERY_TOKEN`, and `SURFACE_FRAME`. For relation prompts, it names `RELATION`, `SUBJECT`, `ANSWER_CLASS`, and `SWAP_GROUP`.

## The Intervention

For each target item, Lab 26 chooses donors:

| Donor condition | Meaning |
|---|---|
| `no_op_same_example` | self-patching identity control |
| `preserve_variable` | donor preserves the hypothesis variable while changing surface/context variables |
| `break_variable` | donor breaks the named high-level variable |
| `random_matched` | deterministic same-domain, same-length donor |
| `wrong_site_preserve` | preserving donor patched at a site outside the named hypothesis site |

At each hypothesis-named residual site, the lab patches the target prompt with the donor stream vector and measures:

```text
scrub_score = patched_logit_diff / clean_logit_diff
logit_diff = logit(target) - logit(distractor)
```

A good abstraction should show:

1. preserving donors keep the score high;
2. broken-variable donors damage the score;
3. random and wrong-site controls do not preserve nearly as well;
4. the pattern survives more than one item and does not depend on one lucky depth.

## Evidence Ladder

| Evidence tag | What earns it here | What it does not imply |
|---|---|---|
| `FORMAL` | a JSON hypothesis specifies variables, sites, thresholds, and resampling rules before the run | mathematical proof |
| `CAUSAL` | residual-stream interchange changes the measured behavior under controls | full circuit identification |
| `AUDIT` | counterexamples and refinement rows are preserved, not hidden | that the refined hypothesis is validated before rerun |

The strongest positive sentence is narrow:

```text
FORMAL + CAUSAL: under this residual-resampling battery, preserving variable X at site S preserved the target margin more than breaking X or using matched controls.
```

The forbidden sentence is broad:

```text
The model implements exactly this algorithm.
```

## Main Artifacts

| Path | What it contains |
|---|---|
| `run_summary.md` | the read-first answer to the seven standard questions |
| `method_card.md` | one-page method contract and verdict table |
| `causal_abstraction_spec.md` | human-readable copy of the JSON specs |
| `operationalization_audit.md` | cheap explanations, controls, counterexamples, and allowed language |
| `metrics.json` | aggregate metrics and dynamic verdicts |
| `results.csv` | every residual-resampling intervention |
| `tables/hypothesis_spec_audit.csv` | formal spec schema/hash/threshold audit |
| `tables/baseline_behavior.csv` | clean margins and baseline gate status |
| `tables/donor_plan.csv` | selected donors and which variables they preserve or break |
| `tables/resampling_interventions.csv` | long-form patch rows, duplicated under `tables/` for notebooks |
| `tables/variable_preservation_summary.csv` | mean scores and pass/fail gates by site/depth |
| `tables/best_hypothesis_cells.csv` | best cell per hypothesis under the gate ordering |
| `tables/evidence_matrix.csv` | compact claim-writing matrix |
| `tables/counterexamples.csv` | automatic rows that shrink or kill the favorite claim |
| `tables/hypothesis_refinement_log.csv` | v2 suggestions driven by failed gates |
| `plots/plot_reading_guide.csv` | what each plot can and cannot support |
| `ledger_suggestions.md` | drafted claims with measured numbers |

## Plot Reading Order

Start with:

```text
plots/causal_abstraction_dashboard.png
```

Then read:

1. `plots/resampling_preservation_matrix.png`: where does preservation happen, and does it vanish when the variable breaks?
2. `plots/hypothesis_pass_fail_atlas.png`: which formal gates pass?
3. `plots/variable_specificity_ladder.png`: do preserving donors beat random and wrong-site controls?
4. `plots/counterexample_gallery.png`: which rows would a responsible claim have to explain?
5. `plots/refinement_trajectory.png`: what smaller v2 hypothesis should be tested next?

Do not skip `tables/counterexamples.csv`. A single systematic broken-variable leak is more informative than a pretty aggregate.

## How To Read A Result

Positive posture:

```text
preserve_variable high
break_variable low
random_matched low
wrong_site_preserve low
counterexamples few and unsystematic
```

Refinement posture:

```text
preserve_variable high
break_variable also high
```

The variable is too broad or the site carries something downstream of the variable.

Control-leak posture:

```text
wrong_site_preserve high or random_matched high
```

The patch may be perturbing the stream in a helpful way, or the site is not specific enough.

Negative posture:

```text
preserve_variable low
```

The proposed mapping did not preserve behavior under this instrument. That is a clean result, not a failed lab.

## Writeup Questions

1. What were the high-level variables in the spec before the run?
2. Which low-level sites did the spec nominate, and why?
3. What was the clean baseline pass rate by domain?
4. At the best site/depth, how much did preserving donors preserve?
5. How much did broken-variable donors preserve?
6. Which control was closest to the preserving donor: random or wrong-site?
7. What is the strongest counterexample in `tables/counterexamples.csv`?
8. Does the result support the original hypothesis, or only a narrower v2?
9. What would be the next held-out replication slice?
10. Write one allowed claim and one forbidden overclaim.

## Ledger Templates

Allowed positive template:

```text
[L26-C1] FORMAL+CAUSAL | For <domain> prompts in Lab 26, hypothesis <H> survived residual resampling at <site>: preserving donors scored <x> vs broken-variable <y>, with specificity gap <z>.
Artifact: runs/<run>/tables/evidence_matrix.csv | Falsifier: a held-out run where preserving donors no longer beat broken-variable and wrong-site/random controls.
```

Allowed negative/refinement template:

```text
[L26-C2] FORMAL+CAUSAL,AUDIT | Hypothesis <H> did not earn the positive abstraction claim because <failed gate>; the supported next claim is narrower: <v2>.
Artifact: runs/<run>/tables/hypothesis_refinement_log.csv | Falsifier: a rerun of the stated v2 on held-out items that still fails the same control.
```

Forbidden:

```text
The model implements this algorithm.
This site is the whole circuit.
The abstraction works generally.
```

## Extension Ideas

- Replace the provided JSON specs with a student-authored hypothesis and rerun without changing thresholds after seeing results.
- Use Lab 12 relation families beyond `country_sem`.
- Save the best Lab 26 site as the starting point for Lab 27 path-specific mediation.
- Run the same spec across GPT-2 and the course base model and compare which counterexamples are stable.

