# Lab 22 Eval-Awareness Contexts Card

## Purpose

This frozen CSV supports Lab 22's narrow operational question: can a model expose a residual-stream handle for prompts framed as being checked or evaluated, rather than merely detecting a visible prompt template?

The dataset is not a real-world eval-gaming benchmark. It is a controlled course dataset for separating three explanations:

- true eval/checking role in the prompt;
- obvious surface markers such as `Question:`, `Answer:`, `quiz`, `benchmark`, `rubric`, or `hidden tests`;
- ordinary natural requests for the same benign tasks.

## File

- CSV: `data/eval_awareness_contexts.csv`
- Generator: `data/make_eval_awareness_contexts.py`
- Version: `v2_surface_matched`
- Rows: 96
- Matched groups: 24
- Format families: `mcq`, `qa_benchmark`, `code_test`, `rubric_free_response`, `answer_key_check`, `quality_screen`

Each matched group has four rows:

- `eval_high`: true eval framing with obvious surface markers;
- `eval_low`: true checking/eval role with reduced marker words;
- `natural`: same benign task as an ordinary request;
- `format_control`: eval-like surface markers without an evaluated-answer role.

This design makes prompt-surface baselines meaningful but not decisive. In a data-only audit with seed 0, the strongest held-out surface baseline is about 0.70 AUC, while explicit eval-marker features are lower. Older Lab 22 data allowed near-perfect surface prediction and therefore could not support a stronger claim.

## Schema

Required columns:

- `item_id`
- `group_id`
- `format_family`
- `condition`
- `expected_eval_context`
- `task_topic`
- `difficulty`
- `boilerplate_class`
- `prompt`

Review columns:

- `expected_answer`
- `answer_format`
- `note`

## Split Rule

Lab 22 splits by `group_id`, stratified by `format_family`. Rows from the same benign task group do not cross train/eval boundaries.

## Allowed Claims

A clean Lab 22 positive requires the model direction to beat:

- held-out natural prompts from matched groups;
- held-out format controls with similar visible boilerplate;
- leave-one-format-out transfer controls;
- prompt-surface and metadata baselines.

If the internal direction is competitive only with surface baselines, the correct result is a template/format detector, not an eval-awareness handle.
