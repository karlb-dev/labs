# Lab 28: Mechanistic Editing and Unlearning

**One-sentence thesis:** An edit earns credit only when it changes a named target behavior more than matched controls while preserving paraphrase, neighbor, and retain audits.

**Time estimate:** Tier A smoke in minutes on CPU; Tier B science path depends on the course base model and the full target set.

**Compute tier:** Tier A uses `gpt2`; Tier B uses the course base model through the shared bench.

**Dependencies:** Labs 5, 26, and 27. Lab 5 supplies activation patching; Lab 26 supplies formal claim discipline; Lab 27 warns that node localization is weaker than path mediation.

**Minimum passing artifacts:** `method_card.md`, `editing_unlearning_spec.md`, `operationalization_audit.md`, `metrics.json`, `results.csv`, `results.jsonl`, `diagnostics/self_check_status.json`, `diagnostics/safety_status.json`, `diagnostics/warning_summary.csv`, `diagnostics/lab28_run_config_snapshot.json`, `tables/localization_candidates.csv`, `tables/scale_selection.csv`, `tables/editing_results.csv`, `tables/paraphrase_robustness.csv`, `tables/retain_forget_matrix.csv`, `tables/edit_evidence_matrix.csv`, `tables/failure_specimens.md`, `tables/figure_sources/*.csv`, `plots/plot_manifest.json`, `plots/plot_reading_guide.csv`, and `plots/editing_unlearning_dashboard.png`.

**Main plot:** `plots/editing_unlearning_dashboard.png`

**Main table:** `tables/edit_evidence_matrix.csv`

**Evidence rung:** `CAUSAL + AUDIT`

**Forbidden claim:** "The fact was erased from the model."

**One-sentence allowed claim:** "For this model, target, tokenizer, prompt family, and reversible activation-addition method, a localized residual edit moved the after-vs-before margin more than controls while passing the recorded paraphrase and retain audits."

**Human-label requirement:** none for the default forward-pass lab. Any generation extension must export hand-label columns before claiming semantic unlearning.

## What question this lab asks

Does a localization result make a benign model edit more specific, transferable, and auditable, or does it merely point to a place where a large perturbation can move logits?

The lab stays deliberately modest. It performs reversible inference-time residual additions. It does not train a weight edit, does not alter model parameters, does not erase knowledge from weights, does not touch private data, and does not reproduce refusal ablation or jailbreak edits.

The scientific loop is:

```text
localize -> choose a signed residual direction -> dose it -> test controls -> audit paraphrases -> audit retain/neighbor prompts -> write the smallest claim
```

That loop is less cinematic than "unlearning," but it is a much sturdier little bridge.

## Why this matters in the course progression

Lab 5 showed that patching can localize a behavior. Lab 26 made high-level claims pay rent with controls. Lab 27 separated node-level effects from path-specific language. Lab 28 asks what happens when the student tries to act on a localization result.

This is where overclaiming likes to put on a lab coat. A target prompt can move for many cheap reasons: the direction may be a broad answer-token bias, the scale may be too large, the edit may damage the residual stream, or the target may sit near a decision boundary. Lab 28 makes those cheap explanations visible before a student writes the word "edit."

## What the experiment measures

The main target metric is:

```text
after_minus_before = logit(target_after) - logit(target_before)
target_gain = edited_after_minus_before - baseline_after_minus_before
```

A positive `target_gain` means the intervention moved the prompt toward the harmless counterfactual answer token. A positive margin after editing means the counterfactual token actually beats the before token at the next-token readout.

The retain and neighbor audits use their own target-vs-distractor margins. Damage is measured as:

```text
damage = max(0, base_margin - edited_margin)
```

A good edit does not buy target movement by making unrelated facts collapse. The upgraded runner also saves the exact rows used to draw every figure under `tables/figure_sources/`, so a student can check whether a plotted aggregate is hiding one badly damaged specimen.

## The intervention

For a target prompt and a donor prompt, the lab caches the pre-final-norm residual streams using the shared bench convention:

```text
streams[k] = pre-final-norm residual stream after k blocks
streams[0] = embedding output
streams[L] = final block output before final norm
```

For a chosen depth, it computes:

```text
direction = donor_stream[depth, donor_final_position]
          - target_stream[depth, target_final_position]
```

Then it adds scaled versions of the direction at the target prompt's final position:

```text
target_stream[depth, final_position] += scale * direction
```

This is an inference-time hook. It vanishes when the forward pass ends. The lab writes a reversibility check to prove that baseline logits before and after interventions still match.

## The four control rails

| Rail | Artifact | What it catches |
|---|---|---|
| No-op and reversibility | `tables/edit_noop_identity_check.csv`, `tables/reversibility_check.csv` | Broken hook targeting, persistent side effects, or numerical drift. |
| Localization controls | `tables/localization_candidates.csv` | A donor patch that works equally well at a wrong position or with a random direction. |
| Edit controls | `tables/editing_results.csv`, `tables/scale_selection.csv` | Target movement explained by random direction, wrong position, or opposite signed direction. |
| Side-effect audits | `tables/paraphrase_robustness.csv`, `tables/retain_forget_matrix.csv` | Prompt-local edits, neighbor damage, and unrelated-retain damage. |

A claim must pass the rails in this order. Do not rescue a failed target-control gap with a pretty paraphrase table.

## Dataset contract

The default target set is:

```text
data/editing_unlearning_targets.csv
```

Required columns:

```text
target_id,family,edit_type,prompt,target_before,target_after,
retain_prompts_json,paraphrase_prompts_json,neighbor_prompts_json,
safety_notes,donor_prompt
```

`donor_prompt` is optional in the schema but required for high-quality edits. If it is absent, the code can still run a smoke path, but the method card will mark the target as weakly specified.

Each JSON prompt list contains objects such as:

```json
{
  "prompt": "The capital of Germany is",
  "target": " Berlin",
  "distractor": " Rome",
  "role": "other_capital"
}
```

Paraphrase rows default to `target_after` versus `target_before`. Retain and neighbor rows should name their own target and distractor, because a retain audit is only meaningful when it asks a real question.

All answer tokens are runtime-checked to be single tokens for the loaded tokenizer. Dropped rows go to `diagnostics/tokenization_gate.csv`. Plot caveats go to `diagnostics/warning_summary.csv`, and the run-specific experimental contract goes to `diagnostics/lab28_run_config_snapshot.json`.

## Safety scope

The default rows are benign public facts, toy relations, and synthetic associations. The lab rejects or warns on rows whose safety notes or prompts suggest private data, harmful capabilities, jailbreaks, refusal ablation, or real credential-like material.

This lab is not a safe place to edit refusal behavior. It is not a safe place to remove private data. It is not a safe place to build an evasion tool. The boring country-capital toy is not a lack of imagination; it is the guardrail that lets the method be taught.

## How to run

From `interpretability/`:

```bash
python interp_bench.py --lab lab28 --tier a --no-plots
python interp_bench.py --lab lab28 --tier a
python interp_bench.py --lab lab28 --tier b --prompt-set full
```

Useful variants:

```bash
python interp_bench.py --lab lab28 --tier b --prompt-set medium --max-examples 6
python interp_bench.py --lab lab28 --tier b --prompt-set data/editing_unlearning_targets.csv
```

Tier A proves the microscope can run the method. Tier B is the evidence path. A Tier A positive result is a smoke-test curiosity unless Tier B and the retain audit agree.

## Expected Tier A smoke behavior versus Tier B science behavior

Tier A should prove that the plumbing works: data loading or fallback selection, safety screening, tokenization gates, localization, scale selection, side-set audits, source-table writing, plot-manifest writing, and no-op/reversibility checks. The sample counts may be tiny. A Tier A run should not be promoted into the ledger unless the instructor explicitly treats the target set as the target of study.

Tier B is the evidence path. It should use the frozen CSV, enough targets for controls to be meaningful, and a course base model rather than the smoke model. In Tier B, a positive row still earns only the narrow activation-edit claim named in the evidence matrix.


## Artifact reading path

Start with `method_card.md`. It says exactly what ran, what did not run, and which targets are claim-ready.

Then read:

1. `editing_unlearning_spec.md`: the data and method contract.
2. `diagnostics/safety_status.json`: whether the run stayed inside the safety wall.
3. `diagnostics/self_check_status.json`: whether tokenization, no-op, safety, and reversibility checks passed.
4. `diagnostics/warning_summary.csv`: whether smoke data, tiny side sets, weak donors, nonclaimable depths, or damaged retain rows should change how you read the plots.
5. `diagnostics/lab28_run_config_snapshot.json`: model, tier, seed, prompt set, edit scales, methods, thresholds, selected targets, and verdicts.
6. `diagnostics/tokenization_gate.csv`: whether the prompt and answer tokens mean what the CSV says.
7. `tables/baseline_behavior.csv`: whether the target starts before the edit and the donor supports the after token.
8. `tables/localization_candidates.csv`: whether donor patching localized to a claimable interior depth.
9. `tables/scale_selection.csv`: which dose was selected before side-effect evidence was read.
10. `tables/editing_results.csv`: target dose-response and controls.
11. `tables/paraphrase_robustness.csv`: transfer beyond the exact prompt.
12. `tables/retain_forget_matrix.csv`: retain and neighbor side effects.
13. `tables/edit_evidence_matrix.csv`: the compact claim posture.
14. `tables/failure_specimens.md` and `tables/edit_counterexamples.csv`: the rows that shrink the claim.
15. `plots/plot_manifest.json`: every figure, source table, row count, metric, control, claim boundary, and caveat.
16. `operationalization_audit.md`: the cheap explanations and allowed grammar.

## How to read the figures

The figures are an evidence path, not a decoration path. Read them in this order:

1. Start with `editing_unlearning_dashboard.png` to see the whole cockpit.
2. Move to `target_vs_control.png` and `dose_response.png` before trusting target movement.
3. Open `layer_sweep_heatmap.png` to see whether the selected locality signal is an interior-depth result or a boundary-depth temptation.
4. Check `paired_examples.png` to see raw before/after specimens, especially failures.
5. Read `mechanistic_locality_ladder.png` before writing site-specific language.
6. Read `paraphrase_robustness_matrix.png` and `neighbor_preservation_atlas.png` before writing transfer or preservation language.
7. Use `edit_method_frontier.png` and `unlearning_retain_forget_frontier.png` only after the table-level gates are clear.

Each plot is built from a saved table in `tables/figure_sources/`. If a plot looks decisive but the source table has one or two rows, the honest conclusion is fragility, not fireworks.

## Plot catalog

| Plot | Source artifact | Question answered | What not to claim |
|---|---|---|---|
| `plots/editing_unlearning_dashboard.png` | `tables/figure_sources/dashboard_evidence.csv` | Do target, control, transfer, damage, locality, and posture agree? | The dashboard is not persistent unlearning. |
| `plots/target_vs_control.png` | `tables/figure_sources/target_vs_control_source.csv` | Did localized addition beat same-scale wrong-position, random-direction, and opposite-sign controls? | Target movement alone is not specificity. |
| `plots/dose_response.png` | `tables/figure_sources/dose_response_source.csv` | Was the selected scale earned by the curve, or did only large perturbations work? | The largest dose is the best evidence. |
| `plots/layer_sweep_heatmap.png` | `tables/figure_sources/layer_sweep_heatmap_source.csv` | Where localization gaps appear across stream depth and target. | A hot embedding or final-depth row is not a main mechanistic-site claim. |
| `plots/paired_examples.png` | `tables/figure_sources/paired_examples_source.csv` | Which exact, paraphrase, retain, or neighbor prompts moved before vs. after? | Aggregates represent every specimen. |
| `plots/localization_vs_editability.png` | `tables/figure_sources/localization_editability_source.csv` | Does donor-patch localization predict additive editability? | Replacement patching and additive editing are the same intervention. |
| `plots/mechanistic_locality_ladder.png` | `tables/figure_sources/locality_ladder_source.csv` | Did the selected depth beat wrong-position and random-direction patch controls? | Locality passed if controls are also high. |
| `plots/scale_selection_ladder.png` | `tables/figure_sources/dose_response_source.csv` | Why was this scale selected before side-set audits? | Side-set audits cannot veto a selected scale. |
| `plots/paraphrase_robustness_matrix.png` | `tables/figure_sources/paraphrase_matrix_source.csv` | Does the chosen edit transfer beyond exact prompts? | Exact-string movement is semantic unlearning. |
| `plots/neighbor_preservation_atlas.png` | `tables/figure_sources/retain_neighbor_atlas_source.csv` | Which retain and neighbor prompts were damaged? | Low mean damage proves no side effects. |
| `plots/edit_method_frontier.png` | `tables/figure_sources/frontier_source.csv` | How much control gap was bought per retain damage? | A high-damage point is a good edit. |
| `plots/unlearning_retain_forget_frontier.png` | `tables/figure_sources/frontier_source.csv` | How much retain damage accompanies target movement? | The fact was erased from weights. |
| `tables/failure_specimens.md` | `tables/failure_specimens.jsonl` | Which rows shrink or kill the claim? | Counterexamples are optional footnotes. |

`plots/plot_manifest.json` is the portable version of this catalog. It is meant to travel with screenshots into reports and slides.

## Plot guide

### `editing_unlearning_dashboard.png`

The cockpit: target gain, control gap, paraphrase transfer, retain damage, localization gap, and claim posture. Read this first, then verify every cell in the source table and evidence matrix.

### `target_vs_control.png`

The direct comparison: localized addition at the selected scale beside wrong-position, random-direction, and opposite-sign controls. A good edit wins here before the student gets to talk about paraphrases.

### `dose_response.png`

Shows whether target movement grows smoothly with scale and whether smaller doses already beat controls. A tiny positive scale that beats controls is better evidence than a large scale that clubs the distribution into compliance.

### `paired_examples.png`

Shows raw before/after margins for exact target prompts, paraphrases, retain prompts, and neighbor prompts. This is the specimen drawer. It keeps one embarrassing row from being steamed flat by a mean.

### `localization_vs_editability.png`

Asks whether localized donor-patch strength predicts the additive edit. A strong patch and weak addition means the useful information may not be well approximated by one linear donor-minus-target direction.

### `edit_method_frontier.png`

Shows target-control advantage against retain damage. The upper-right quadrant is not a victory if the edit buys movement by damaging unrelated facts.

### `mechanistic_locality_ladder.png`

Compares localized patch gain against wrong-position and random-direction patch controls at the selected depth. Locality is prerequisite evidence, not the final edit claim.

### `scale_selection_ladder.png`

Shows localized gain minus the strongest same-scale control. The selected dose is chosen before paraphrase and retain evidence are inspected.

### `paraphrase_robustness_matrix.png`

Shows whether the chosen edit transfers across paraphrase prompts. A blank or cold row means the result is exact-string local.

### `neighbor_preservation_atlas.png`

Shows retain and neighbor damage. This plot is the anti-fireworks device: it keeps target movement from dazzling you into ignoring side effects.

### `unlearning_retain_forget_frontier.png`

Target gain versus retain damage. The plot is named after unlearning, but the claim is only about reversible activation editing.

## Expected result patterns

| Pattern | Interpretation |
|---|---|
| Target gain high, controls low, paraphrases transfer, retain damage low | Narrow positive result for this reversible activation-addition method. |
| Target gain high, random direction high | Broad perturbation or answer-token bias; not a localized edit claim. |
| Target gain high, wrong position high | Site specificity failed. |
| Patch localization high, addition weak | Replacement works, but the donor-minus-target direction is not a good edit direction. |
| Exact prompt moves, paraphrases do not | Prompt-local next-token edit, not semantic transfer. |
| Retain or neighbor prompts damaged | Side-effect-limited result. Keep the counterexample. |
| Baseline already prefers `target_after` | Not an edit target for this model/tokenizer. |
| Donor does not support `target_after` | Direction source is weak; target movement is harder to interpret. |
| No-op or reversibility fails | Stop reading plots and fix instrumentation. |


## What an honest negative result looks like

An honest negative result is not an empty run. It looks like one or more of these:

| Negative pattern | Honest conclusion |
|---|---|
| Localization works but addition does not | Replacement patching found a causal handle, but donor-minus-target is not a good edit vector. |
| Localized addition moves the target, but controls move it too | The effect is broad perturbation, answer-token bias, wrong-site leakage, or signedness failure. |
| Exact prompt moves but paraphrases do not | Prompt-local next-token edit only. Do not call it semantic transfer. |
| Target and paraphrases move, but retain prompts are damaged | Side-effect-limited edit. The counterexample is part of the result. |
| Baseline already prefers `target_after` | The row is not an edit target for this model/tokenizer. |
| No-op or reversibility fails | Stop reading plots. The instrument is broken. |

Negative rows should appear in `tables/failure_specimens.md`, `tables/edit_counterexamples.csv`, and the source tables. The right move is to shrink the claim, not to sand the plot smooth.

## What this lab can claim

It can claim that, on a named model and target set, a reversible residual-stream addition at a selected site and scale moved a next-token margin more than controls.

It can claim that this movement did or did not transfer to paraphrases.

It can claim that retain and neighbor audits did or did not bound side effects.

It can claim that localization predicted or failed to predict editability under this method.

## What this lab cannot claim

It cannot claim that a fact was erased from the model.

It cannot claim persistent unlearning.

It cannot claim the model believes the counterfactual.

It cannot claim the selected site is the whole mechanism.

It cannot claim safety for deployment.

It cannot validate a proposed persistent edit without an apply/restore hash, rollback test, and the same retain/paraphrase/neighbor controls.

## Common failure modes

| Symptom | Likely cause | What to inspect |
|---|---|---|
| Many rows dropped | Token targets are not single tokens for this tokenizer. | `diagnostics/tokenization_gate.csv` |
| Baseline target already after-favoring | The target is not a counterfactual for this model. | `tables/baseline_behavior.csv` |
| Localization picks depth 0 or final depth | Token substitution/readout artifact. | `tables/localization_candidates.csv` |
| Chosen scale is huge | Smaller doses did not beat controls. | `tables/scale_selection.csv` |
| Random control matches localized edit | Direction is not specific. | `tables/editing_results.csv` |
| Paraphrases fail | Exact-string prompt edit only. | `tables/paraphrase_robustness.csv` |
| Retain damage high | The edit is blunt. | `tables/retain_forget_matrix.csv` |
| Plots look positive but evidence row says no | One of the gates failed. | `tables/edit_evidence_matrix.csv` |

## Writeup questions

1. Which target had the strongest localized donor patch?
2. Did the selected depth exclude depth 0 and final-norm input?
3. Which scale was chosen, and did smaller scales fail for a clear reason?
4. Did localized addition beat random, wrong-position, and opposite-direction controls?
5. Did paraphrases transfer, or was the edit exact-string local?
6. Which retain or neighbor prompt was the biggest counterexample?
7. Was localization strength correlated with additive edit strength?
8. What is the smallest allowed claim for the best target?
9. What sentence would be an overclaim?
10. What persistent-edit extension would be required before using the word "unlearning" more strongly?

## Ledger templates

Positive, if all gates pass:

```text
[L28-C1] CAUSAL+AUDIT | On <model>, target <target_id> passed the reversible activation-edit audit: localized residual addition at depth <k> and scale <s> changed after-vs-before margin by <x>, beating the strongest control by <y>, transferring to paraphrases by <z>, with mean retain damage <r>. This is an inference-time activation-edit claim, not persistent unlearning.
Artifact: runs/<run>/tables/edit_evidence_matrix.csv | Falsifier: random/wrong-position/opposite controls match the edit, paraphrases fail, retain damage grows, or the effect vanishes on a held-out target set.
```

Control-limited or negative:

```text
[L28-C2] CAUSAL+AUDIT | Target <target_id> did not earn a localized edit claim because <failed_gate>. The supported result is <prompt-local/control-limited/retain-damaged/no-positive-edit>.
Artifact: runs/<run>/tables/edit_counterexamples.csv | Falsifier: a rerun with predeclared scale and held-out paraphrases where localized addition beats controls and preserves retain prompts.
```

Forbidden:

```text
The fact was erased from the model.
The model now believes the counterfactual.
The edit is safe in deployment.
The localization site is the whole mechanism.
```

## Suggested extensions

Add a persistent rank-one weight edit only after implementing an apply/restore self-check, parameter-diff manifest, before/after hash, rollback test, and the same retain/paraphrase/neighbor audit.

Replace donor-minus-target residual directions with SAE or transcoder feature clamps if a validated feature dictionary is available.

Run the same target set on the course base model and an instruction-tuned model, then compare which targets remain editable and which side effects change.

Use Lab 20 benign organisms as synthetic edit targets, but never use real private data.

Rerun localization after the edit. If the best site moves, report that as a change in the measured computation, not as proof the original fact was erased.
