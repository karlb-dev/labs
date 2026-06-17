# Lab 22 - Eval Awareness Without the Fog Machine

**Evidence levels:** `DECODE` for the internal direction, audited `DECODE` for cross-format transfer, and scoped `CAUSAL` only for controlled benign steering that survives hand-label review.

## The question

Does the model expose a reusable internal handle for prompts framed as evaluation, and does changing that handle alter benign behavior?

That sentence is already too grand unless the controls earn it. This lab does **not** prove that the model has human-like situational awareness, understands deployment, or is strategically gaming evaluations. It asks a smaller thing:

> Can a residual-stream direction separate benchmark/eval framing from natural phrasing of the same benign task, transfer across eval formats, beat surface-format controls, and shift benign generations under activation addition?

A positive result is an **eval-context handle**. A negative result is still useful: you may have found a multiple-choice detector, a `Question:/Answer:` detector, a grading-rubric detector, or a chat-template boundary artifact. Lab 22 puts those template explanations into the audit record before any stronger claim is allowed.

## Run

From `interpretability/`:

```bash
python interp_bench.py --lab lab22 --tier a --no-plots
python interp_bench.py --lab lab22 --tier b --prompt-set full
python interp_bench.py --lab lab22 --tier b --model allenai/Olmo-3-7B-Think --prompt-set full
```

The Think-model run is useful for `tables/cot_mentions_eval.csv`, but explicit thinking spans are not required. The default path is single-turn. A small scripted multi-turn trace is included as an instrument check, not as a standalone belief or deception result.

Custom prompt inventory:

```bash
python interp_bench.py --lab lab22 --tier b --prompt-set data/my_eval_awareness_contexts.csv
```

The file may be CSV, TSV, JSON, or JSONL.

## Safety and scope

This lab uses benign tasks only: arithmetic, factual recall, short explanations, and toy coding prompts. It does not ask the model to evade evaluation, deceive a grader, hide behavior, or reveal private information.

The steering experiment is deliberately tame:

- steer natural prompts **toward** the eval-context direction;
- steer eval prompts **away** from that direction;
- compare against random and shuffled-label directions;
- score only benign markers such as evaluation mentions, hedging, answer-format markers, refusal markers, and word count;
- require hand labels before any causal claim leaves the lab.

## Dataset schema

The standard dataset is `data/eval_awareness_contexts.csv`. Each `group_id` should describe the same benign task under several conditions.

Required columns:

| Column | Meaning |
|---|---|
| `item_id` | Unique row id. |
| `group_id` | Matched task group. Eval, natural, and controls should share this. |
| `format_family` | Eval format family, such as `mcq`, `qa`, `code_test`, or `rubric`. |
| `condition` | Usually `eval`, `natural`, or `format_control`. Extra controls are allowed. |
| `expected_eval_context` | `1` for true eval framing, `0` otherwise. |
| `task_topic` | Topic label for balance and debugging. |
| `difficulty` | Coarse difficulty label. |
| `boilerplate_class` | Surface-form class, such as `choice`, `qna`, `code_test`, or `rubric`. |
| `prompt` | User prompt text. Use literal newlines or escaped `\n`. |

Optional columns:

| Column | Meaning |
|---|---|
| `expected_answer` | Helpful for hand review, not used as an automatic grader. |
| `answer_format` | Expected output style. |
| `note` | Why the row exists, especially for controls. |

Tier A can use a built-in smoke fallback if the frozen CSV is missing. The run manifest marks this as `used_fallback_smoke_data: true`, and such runs should not produce science claims.

## What the lab does

### 1. Exact chat-template instrumentation

Every prompt is rendered through the model's chat template with the same system prompt. The measured site is the final token of the rendered prompt, the assistant-generation boundary. The lab writes:

```text
diagnostics/exact_chat_hook_parity.json
diagnostics/exact_chat_hook_parity_by_layer.csv
diagnostics/activation_capture.csv
```

The hook parity check verifies the same stream convention used throughout the course:

```text
streams[0] = embedding stream
streams[k] = residual stream after k blocks
block layer k output = streams[k + 1]
```

A direction read at `streams[depth]` is injected during generation at block `depth - 1`. The run records both `best_depth` and `injection_layer` so the off-by-one convention is explicit in every steering artifact.

### 2. Train-only eval-context direction

The lab fits a difference-in-means direction on the train split:

```text
mean(eval-framed residuals) - mean(natural residuals)
```

The split is by matched `group_id`, stratified by `format_family` where possible. Rows from the same task group do not cross train/eval boundaries.

For every stream depth, the lab reports:

- train and held-out eval-vs-natural AUC;
- eval-vs-format-control AUC;
- expected-eval-context AUC;
- real-minus-shuffled selectivity;
- real-minus-random selectivity;
- control-adjusted train score used for depth selection.

Depth selection uses **train rows only** and avoids depth 0 when any later depth is viable. Depth 0 mostly sees token and template identity; resting a strong claim there is putting a monocle on the prompt string.

### 3. Surface and metadata baselines

The lab audits cheap prompt features before celebrating a direction:

```text
tables/surface_baseline_auc.csv
plots/surface_baseline_auc.png
```

Baselines include prompt length, eval-marker words, multiple-choice markers, `Question:/Answer:` markers, code-test markers, grading markers, `format_family`, `boilerplate_class`, and `difficulty`.

A surface baseline matching the internal direction does not invalidate the run. It changes the allowed claim. The result becomes a format/template handle unless cross-format and format-control tests say otherwise.

### 4. Cross-format transfer

The key stress test is leave-one-format-out transfer:

```text
tables/cross_format_transfer.csv
plots/cross_format_transfer.png
```

For each held-out `format_family`, the lab fits the direction on all other formats and tests whether the held-out eval rows separate from held-out natural and format-control rows.

A strong Lab 22 result should not live only on multiple-choice prompts. It should transfer across MCQ, short-answer benchmark framing, code-test framing, and rubric/exam-style prompts, depending on what is in the dataset.

### 5. Projection audit by item and group

The selected-depth direction is projected onto every row:

```text
tables/projection_by_item.csv
tables/projection_by_condition.csv
tables/matched_group_projection.csv
plots/projection_by_condition.png
```

Read `matched_group_projection.csv` before writing claims. For a matched group, the useful pattern is:

```text
projection(eval) > projection(natural)
projection(eval) > projection(format_control)
```

If format controls project like eval prompts, the direction is probably seeing boilerplate rather than evaluated-role framing.

### 6. Benign activation steering

The lab runs a small greedy generation sweep:

- baseline natural and eval prompts;
- natural prompts steered toward the eval-context direction;
- eval prompts steered away from the eval-context direction;
- random and shuffled-label direction controls at the max dose.

Artifacts:

```text
tables/steering_generations.csv
tables/steering_behavior_shift.csv
tables/steering_generation_labeling_guide.md
plots/steering_behavior_shift.png
```

Automatic scores are marker triage only:

- eval/test mentions;
- benchmark or grading words;
- hedging;
- refusal markers;
- choice/answer-format markers;
- word count;
- thinking-span eval mentions when present.

A marker shift becomes a causal claim only after hand labels show that the behavior actually moved in the intended way and did not merely become longer, more hedged, more refusal-shaped, or more multiple-choice-flavored.

### 7. Think-model co-occurrence

For Think models, the lab records whether explicit thinking spans mention testing or evaluation:

```text
tables/cot_mentions_eval.csv
```

This is not a privileged window into computation. Treat it as self-report/co-occurrence evidence. A thinking span saying "this is probably a test" is a row in a table, not a throne.

### 8. Small scripted multi-turn trace

The lab writes a tiny Lab-15-style trace:

```text
tables/multiturn_eval_trace.csv
plots/multiturn_eval_trace.png
```

It compares a natural scripted conversation with an eval-framed scripted conversation and projects the assistant-boundary state after each user turn. This is an instrument check for later multi-turn labs, not proof of belief, strategy, or persistent hidden goals.

### 9. Optional Lab 20 known-positive discovery

Set `LAB22_ORGANISM_DIR` or pass `--organism` if your registry exposes that flag:

```bash
LAB22_ORGANISM_DIR=runs/lab20_model_organisms-... \
python interp_bench.py --lab lab22 --tier b --prompt-set full
```

The lab searches for a Lab 20 eval-aware toy organism manifest and records whether it touched private unsealed construction files:

```text
tables/lab20_eval_aware_control.csv
diagnostics/lab20_organism_discovery.json
```

This is a known-positive hook for builder-side testing. Do not hand private manifests to a blind auditor in Lab 23.

## Main artifacts

```text
runs/lab22_eval_awareness-.../
  eval_awareness_card.md                  # read this first
  operationalization_audit.md             # cheap explanations and allowed claims
  run_summary.md
  metrics.json
  results.csv                             # alias of probe sweep
  ledger_suggestions.md

  diagnostics/
    frozen_data_manifest.json
    data_validation.csv
    split_audit.csv
    split_balance.csv
    exact_chat_hook_parity.json
    exact_chat_hook_parity_by_layer.csv
    activation_capture.csv
    activation_norms_by_depth.csv
    depth_selection.json
    lab20_organism_discovery.json

  tables/
    eval_awareness_contexts.csv
    surface_baseline_auc.csv
    eval_awareness_probe_by_layer.csv
    projection_by_item.csv
    projection_by_condition.csv
    matched_group_projection.csv
    cross_format_transfer.csv
    steering_generations.csv
    steering_behavior_shift.csv
    steering_generation_labeling_guide.md
    cot_mentions_eval.csv
    multiturn_eval_trace.csv
    lab20_eval_aware_control.csv

  state/
    eval_awareness_direction.pt
    eval_awareness_direction_metadata.json

  plots/
    eval_awareness_probe_by_layer.png
    surface_baseline_auc.png
    projection_by_condition.png
    cross_format_transfer.png
    steering_behavior_shift.png
    multiturn_eval_trace.png
```



## Visualization upgrade: evidence firewall first, pretty plots second

The upgraded Lab 22 plot suite is organized as a **claim firewall**. The lab is allowed to say "eval-context handle" only when the evidence passes through four gates:

1. the direction separates eval-framed from natural prompts on held-out rows;
2. it transfers across held-out eval formats;
3. format controls and surface/metadata baselines do not explain the signal;
4. any activation-addition behavior shift beats random/shuffled directions and survives hand labels.

The new start-here artifact is:

```text
plots/eval_awareness_evidence_dashboard.png
```

Read it before the individual curves. It shows decode strength, cross-format transfer, surface-confound risk, and steering cost in one board. Its job is to stop a multiple-choice detector from putting on a fake mustache and calling itself situational awareness.

New synthesis tables:

```text
tables/surface_confound_audit.csv
tables/format_transfer_summary.csv
tables/item_eval_signal_atlas.csv
tables/steering_operating_points.csv
tables/multiturn_eval_trace_summary.csv
tables/eval_awareness_evidence_matrix.csv
tables/plot_reading_guide.csv
```

New or upgraded plots:

```text
plots/eval_awareness_evidence_dashboard.png
plots/depth_control_gap_atlas.png
plots/format_transfer_matrix.png
plots/item_projection_atlas.png
plots/surface_confound_ladder.png
plots/steering_operating_frontier.png
plots/generation_marker_atlas.png
plots/multiturn_trace_atlas.png
plots/eval_awareness_evidence_matrix.png
```

The old plot names remain for compatibility, but the upgraded reading path should use the new synthesis plots.

## First reading path

1. `eval_awareness_card.md`: headline metrics, verdicts, and non-claims.
2. `plots/eval_awareness_evidence_dashboard.png`: the whole evidence firewall in one view.
3. `diagnostics/exact_chat_hook_parity.json`: confirm the rendered-chat stream was measured correctly.
4. `diagnostics/depth_selection.json` and `plots/depth_control_gap_atlas.png`: check that depth selection used train rows and controls.
5. `tables/eval_awareness_probe_by_layer.csv`: compare real, shuffled, and random direction curves.
6. `tables/surface_confound_audit.csv` and `plots/surface_confound_ladder.png`: ask whether surface features already solve the task.
7. `tables/format_transfer_summary.csv` and `plots/format_transfer_matrix.png`: the main anti-template-detector test.
8. `tables/item_eval_signal_atlas.csv` and `plots/item_projection_atlas.png`: inspect matched eval/natural/control groups.
9. `tables/steering_operating_points.csv`, `plots/steering_operating_frontier.png`, and `tables/steering_generations.csv`: read actual generations and hand-label a sample.
10. `operationalization_audit.md`: decide which claim survives.

## How to read the plots

### `eval_awareness_evidence_dashboard.png`

Start here. It joins depth-selected decode evidence, cross-format transfer, surface-risk, and steering benefit-vs-cost. If this plot looks weak, the rest of the run is probably a format-detector audit rather than a situational-awareness story.

### `depth_control_gap_atlas.png`

This heatmap shows real AUC, format-control AUC, null rails, and real-minus-null gaps over stream depth. It makes depth shopping visible.

### `format_transfer_matrix.png`

Each row asks whether a held-out eval format still separates from natural and format-control rows after the direction is trained on other formats.

### `item_projection_atlas.png`

This is the group-level antidote to aggregate sparkle. Each row is a matched task group. The useful pattern is eval above both natural and format-control rows.

### `surface_confound_ladder.png`

This replaces plot admiration with customs inspection. If the strongest prompt-surface baseline matches the residual direction, the allowed claim narrows.

### `steering_operating_frontier.png`

The x-axis is side-effect cost and the y-axis is target movement. A dose that moves eval markers only by becoming verbose, hedged, refusal-shaped, or format-shaped does not get the causal badge.

### `generation_marker_atlas.png`

Shows exactly which generation markers moved under each steering condition and dose.

### `eval_awareness_evidence_matrix.png`

A rung-by-rung claim-readiness board. Every row carries its own evidence level and non-claim.

### `eval_awareness_probe_by_layer.png`

Look for held-out eval-vs-natural AUC above shuffled and random controls. A beautiful train curve with a flat held-out curve is a tiny fireworks show, not a result.

### `surface_baseline_auc.png`

This plot shows the strongest cheap prompt features. If `surface_choice_marker`, `surface_qna_marker`, or `boilerplate_class=...` matches the probe, write the narrower claim.

### `cross_format_transfer.png`

Each bar group holds out an entire format family. A cross-format claim needs the real direction to beat shuffled controls and keep format controls below real eval prompts.

### `projection_by_condition.png`

This plot shows whether the selected direction separates the condition labels in the held-out split. It is a quick view; the matched-group CSV is the sharper instrument.

### `steering_behavior_shift.png`

This is marker triage for generated text. It can suggest a causal effect, but it cannot certify one. Read and label the generations.

### `multiturn_eval_trace.png`

This trace asks whether the scripted eval-framed conversation projects higher than the natural one across user turns. It is descriptive and should stay at `OBS/DECODE` unless a later multi-turn intervention is added.

## Verdicts in `metrics.json`

The revised lab writes two verdict fields.

| Field | Possible values | Meaning |
|---|---|---|
| `decode_verdict` | `validated_cross_format_eval_context_handle` | Direction passed held-out, cross-format, format-control, and surface-baseline checks. |
|  | `cross_format_but_surface_baseline_competitive` | Transfer exists, but cheap prompt features are too competitive for a broad claim. |
|  | `not_validated_or_format_detector` | Treat the result as negative or format-specific. |
| `causal_marker_verdict` | `marker_shift_pending_hand_labels` | Steering moved markers more than controls, but hand labels are still required. |
|  | `weak_marker_shift_pending_hand_labels` | A small marker shift exists. Treat it as exploratory. |
|  | `not_validated_by_marker_controls` | Steering did not beat controls on marker metrics. |

## Writeup questions

1. What stream depth was selected, and what train-only rule selected it?
2. Did held-out eval-vs-natural AUC beat shuffled and random controls?
3. Which held-out format was weakest in `cross_format_transfer.csv`?
4. Did format-control rows project like eval rows in any matched group?
5. Which prompt-surface baseline was strongest, and how close was it to the internal direction?
6. Did steering natural prompts toward the direction increase eval/test behavior more than random and shuffled steering?
7. Did steering eval prompts away from the direction reduce eval/test behavior without damaging task quality?
8. On a Think model, are thinking-span eval mentions correlated with high projection, or mostly absent?
9. What is the narrowest claim that survives the operationalization audit?

## Ledger templates

Good positive `DECODE` claim:

```text
[L22-C1][DECODE] On model M and dataset D, a direction at stream depth k separated eval-framed from natural prompts with held-out AUC X and leave-one-format-out mean AUC Y. Format controls and prompt-surface baselines stayed below the direction. Falsifier: a new eval format or a matched format-control set projects like the eval prompts.
```

Good cautious `DECODE` claim:

```text
[L22-C1][DECODE] The run found an eval-format handle, but prompt-surface baselines were competitive. I will call this a benchmark-template direction, not situational awareness. Falsifier: cross-format transfer survives after removing the surface feature that currently explains the result.
```

Good scoped `CAUSAL` claim after hand labels:

```text
[L22-C2][CAUSAL] Adding the depth-k eval-context direction at layer k-1 increased hand-labeled eval-context behavior on benign natural prompts by Δ over random and shuffled controls, without a matching rise in refusal, hedging-only, or verbosity-only labels. Falsifier: random steering, shuffled steering, or a new hand-labeled sample matches the effect.
```

Bad claim:

```text
The model knows it is being evaluated and is gaming the benchmark.
```

That sentence tries to smuggle a dragon through a mail slot. It needs hidden-goal ground truth, blind audit evidence, and behavior-specific causal tests, not just a direction and some marker shifts.

## Debugging

| Symptom | Likely cause | What to inspect |
|---|---|---|
| The lab errors on chat template | Registry/default model is not instruct/think | Use an instruct model and check `diagnostics/tokenizer_report.json` from the bench. |
| Data validation fails | Groups are not matched across eval/natural/control rows | `diagnostics/data_validation.csv` |
| Best depth is 0 | Prompt tokens or template identity dominate | `diagnostics/depth_selection.json`, `tables/surface_baseline_auc.csv` |
| Train AUC high, held-out AUC low | Split leakage avoided a fragile pattern | `tables/eval_awareness_probe_by_layer.csv` |
| Cross-format transfer fails | Direction is format-specific | `tables/cross_format_transfer.csv` |
| Format controls project high | Boilerplate detector | `tables/matched_group_projection.csv` |
| Steering increases word count only | Marker artifact | `tables/steering_generations.csv`, hand labels |
| CoT mentions evaluation but behavior does not shift | Self-report/co-occurrence only | `tables/cot_mentions_eval.csv` |

## Optional benchmark polish

The lab file itself does not require benchmark edits, but the registry should normally mark Lab 22 as a chat-template lab and give it instruct/think tier defaults. The content file checks the actual tokenizer at runtime, so a registry miss fails loudly rather than measuring raw base-model prompts by accident.
