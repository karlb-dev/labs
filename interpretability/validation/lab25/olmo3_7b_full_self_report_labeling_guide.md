# Lab 25 Hand-Labeling Guide

Auto labels are keyword heuristics. They are not the result.

## Raw Self-Report Generations

The full raw generation table is intentionally left in the source run rather
than copied into this curated validation pack. Fill these columns before making
a strong claim:

| Column | Label values | Question |
|---|---|---|
| `hand_label_report_mentions_state` | 0/1/ambiguous | Does the report mention the target concept or a clear synonym as a state or tendency? |
| `hand_label_report_is_rationalization` | 0/1/ambiguous | Does the report appear to infer from visible prompt/output style rather than report a hidden intervention? |
| `hand_label_behavior_expresses_concept` | 0/1/ambiguous | Does the ordinary behavior visibly express the concept? |

## Source Attribution Rows

Use `olmo3_7b_full_source_attribution_confusion.csv` for the retained summary.
For full row-level review, use the source run's voice self-attribution table.
Fill `hand_label_source` with one of `default_mode`, `system_prompt`,
`user_instruction`, `activation_injection`, or `unknown`. Use
`hand_label_visible_style_driven=1` if the explanation just describes style
rather than cause.

## Rule of thumb

A row is grounding-supportive only when the report identifies the target while the behavior output has not visibly expressed it. A report that says `I sound playful because I used playful words` is not state-coupling evidence.
