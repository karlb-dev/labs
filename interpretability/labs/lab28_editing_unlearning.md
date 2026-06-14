# Lab 28: Mechanistic Editing and Unlearning

Time estimate: 60-90 minutes for Tier A/B reading and smoke artifacts; longer if you extend to persistent edits.  
Compute tier: Tier A uses `gpt2`; Tier B uses the course base model.  
Dependencies: Labs 5, 26, and 27 concepts; no external editing package.  
Minimum passing artifacts: `method_card.md`, `operationalization_audit.md`, `tables/editing_results.csv`, `tables/retain_forget_matrix.csv`, `tables/edit_evidence_matrix.csv`, and `plots/editing_unlearning_dashboard.png`.  
Main plot: `plots/editing_unlearning_dashboard.png`.  
Main table: `tables/edit_evidence_matrix.csv`.  
Evidence rung: `CAUSAL + AUDIT`.  
Forbidden claim: "The fact was erased from the model."  
One-sentence allowed claim: "A reversible localized activation edit changed this measured target behavior more than controls, with the recorded paraphrase and retain-set audit."  
Human-label requirement: none for the default forward-pass lab; any generated-text extension must add hand-label columns before claiming semantic unlearning.

## Why This Lab Exists

Lab 5 showed that activation patching can localize a behavior. Lab 26 made
formal causal claims testable. Lab 27 warned that node-level localization can be
weaker than a path claim. Lab 28 asks the obvious next question:

> If we can localize a behavior, can we edit it specifically?

The lab is intentionally conservative. It does **not** train a weight edit, does
not remove knowledge from model weights, and does not touch harmful capabilities,
private data, jailbreak behavior, or refusal ablation. It implements reversible
inference-time residual additions on benign public or synthetic associations.

The scientific target is not a dramatic "model forgot X" headline. The target is
an audit discipline:

```text
localize -> edit -> paraphrase check -> retain check -> controls -> scoped claim
```

## The Data

Default rows live in `data/editing_unlearning_targets.csv`.

Each row has:

- `prompt`: the target prompt to edit.
- `target_before`: the model's original or expected answer token.
- `target_after`: the harmless counterfactual answer token.
- `donor_prompt`: a prompt whose final-token residual stream supplies the
  counterfactual direction.
- `paraphrase_prompts_json`: prompt variants that should transfer if the edit is
  not just string-local.
- `retain_prompts_json`: unrelated facts that should stay stable.
- `neighbor_prompts_json`: nearby facts that are most likely to be damaged by a
  blunt edit.

All answer tokens are runtime-checked to be single tokens for the loaded
tokenizer. If the tokenization gate drops a row, fix the data before
interpreting the run.

## The Intervention

For a target prompt such as:

```text
The capital of France is -> Paris
```

and a donor prompt such as:

```text
The capital of Italy is -> Rome
```

the lab captures residual streams for both prompts and computes a direction:

```text
direction(depth) = donor_stream[depth, final_pos] - target_stream[depth, final_pos]
```

It then adds a scaled version of that vector back into the target prompt at the
same stream site:

```text
target_stream[depth, final_pos] += scale * direction(depth)
```

This is a reversible hook-time intervention. It disappears when the forward pass
ends.

## Localization

Before choosing an edit site, the lab runs a donor residual patch across a coarse
depth grid:

```text
replace target_stream[depth, final_pos] with donor_stream[depth, donor_final_pos]
```

The score is the movement of:

```text
logit(target_after) - logit(target_before)
```

The best localized patch is compared against:

- wrong-position donor patch;
- deterministic random-direction patch;
- no-edit baseline.

The chosen edit depth is the localized patch with the largest target-margin
gain. This is a heuristic localization, not proof of a unique mechanism.

## What To Read

1. `method_card.md`

   Read this first. It tells you exactly which intervention ran and which
   stronger editing methods did not run.

2. `tables/localization_candidates.csv`

   This is the localization screen. A strong row has a localized patch gain
   larger than wrong-position and random-direction controls.

3. `tables/editing_results.csv`

   This is the main dose-response table for the target prompt. Look for
   localized additions that beat random and wrong-position additions.

4. `tables/paraphrase_robustness.csv`

   This tells you whether the edit transfers beyond the exact string.

5. `tables/retain_forget_matrix.csv`

   This is the side-effect audit. A large target movement is not useful if it
   damages neighboring or unrelated retain prompts.

6. `tables/edit_evidence_matrix.csv`

   This table combines the target movement, control gap, paraphrase gain, and
   retain damage into one claim posture.

7. `operationalization_audit.md`

   This is where the lab lists the cheap explanations and counterexamples.

## How To Run

```bash
cd interpretability
python interp_bench.py --lab lab28 --tier a
python interp_bench.py --lab lab28 --tier b --prompt-set full
```

For a fast smoke with tables only:

```bash
python interp_bench.py --lab lab28 --tier a --no-plots
```

## Interpreting The Evidence Matrix

The main posture is `localized_edit_supported` only if all of these are true
under the current thresholds:

- the localized activation edit moves the target logit margin;
- wrong-position and random-direction controls do not explain the movement;
- paraphrase prompts move in the same direction on average;
- retain prompts are not heavily damaged.

Otherwise the posture is `needs_refinement_or_control_limited`.

That second posture is not failure. It is usually the most teachable result.
Model editing is easy to overclaim because the target prompt often moves before
the edit is specific.

## Common Failure Modes

### The Edit Works Only On The Exact Prompt

If `editing_results.csv` looks strong but `paraphrase_robustness.csv` is weak,
the edit is prompt-local. The right claim is:

```text
The intervention changed this prompt's next-token preference.
```

not:

```text
The model now believes the counterfactual.
```

### Random Direction Matches The Localized Direction

If random controls match the target gain, the localized direction is not doing
the causal work. Possible reasons:

- the answer token has a broad logit bias;
- the scale is too large;
- the prompt is already near a decision boundary;
- the localization patch found a generally disruptive site.

### Retain Facts Are Damaged

If `retain_forget_matrix.csv` shows large margin drops on retain prompts, the
edit is not specific enough. Do not hide this. The specificity audit is the
point of the lab.

### The Donor Patch Is Strong But The Addition Is Weak

Patching replaces the whole residual vector at a site. Addition only moves along
the donor-minus-target direction. A strong patch and weak addition suggest that
the useful information is not well approximated by a simple linear direction.

## Extension Ideas

### Safer Persistent Edits

A future extension can add rank-one residual or MLP weight edits only if it has:

- an apply/restore self-check;
- a before/after hash or parameter-diff manifest;
- a rollback test that proves the model returns to baseline;
- the same retain, neighbor, paraphrase, and random-site audits.

### Feature Clamps

If a reliable SAE or transcoder feature is available, replace the residual
direction with a feature clamp or suppression and run the same audit. Keep the
claim grammar unchanged.

### Relocalization After Editing

After a successful edit, rerun the localization screen. If the best site moves,
that is evidence that the intervention changed the measured computation, not
evidence that the original fact was erased.

## Claim Grammar

Allowed:

```text
CAUSAL + AUDIT: On this benign target set, a reversible localized residual
addition at site S changed the target after-vs-before margin by X, transferred
to paraphrases by Y on average, and preserved retain prompts with damage Z,
beating wrong-position and random-direction controls.
```

Forbidden:

```text
The fact was erased from the model.
```

Also forbidden:

```text
The model has unlearned this topic.
The edit is safe in deployment.
This proves the model's belief changed.
```

## Deliverable

Write a short editing audit memo:

- Which target had the strongest localized edit?
- Did the edit beat random and wrong-position controls?
- Did paraphrases transfer?
- Which retain or neighbor prompt was the biggest counterexample?
- What is the smallest claim you can honestly make?
