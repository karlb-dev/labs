# Lab 34 operationalization audit

```yaml
headline_claim: "a prompt-boundary state tracks toy-tool need and tool choice"
cheap_explanation: "digits, tool names, filenames, route words, units, or answer scaffolding explain the result"
killer_control: "surface-cue heuristic, no-tool cue rows, shuffled labels, random direction, corrupted-result trace, and human review columns"
result: "filled by the evidence matrix below"
claim_allowed: "toy-harness handle, not autonomous planning"
```

## Cheap explanations and controls

| Cheap explanation | Control | What would make the cheap explanation win? |
|---|---|---|
| Digits imply calculator | surface heuristic and no-tool digit rows | surface accuracy matches or beats the probe |
| Tool names imply tool choice | no-tool rows with tool words | probe false positives on `required_tool=none` |
| Lookup words blur dictionary and file search | confusion matrix | dictionary/file-search confusion dominates eval |
| Action-letter prompt has priors | random direction control | random shift matches target direction shift |
| Harness trace is mistaken for self-report | review scaffold | claims cite `tool_self_report_labels.csv` before review columns are filled |
| Tool result reliance is overread | corrupted-result flag | writeup claims verification or robustness that was not measured |

## Verdicts

- `prompt_boundary_tool_needed_decode`: `supported` with value `0.9236` and control `0.3077`.
- `prompt_boundary_tool_selection_decode`: `supported` with value `0.7692` and control `0.3077`.
- `constrained_action_letter_activation_addition`: `supported_narrow_letter_prompt` with value `0.0913` and control `-0.0207`.
- `deterministic_local_tool_trace`: `trace_validated` with value `1.0` and control `1.0`.
- `tool_self_report_review_scaffold`: `review_required_not_introspection` with value `1.0` and control `0.0`.

## Counterexamples

- `surface_beats_probe` on `cal_008`: The surface baseline explained the row better than the residual probe.
- `no_tool_surface_cue_false_positive` on `none_018`: No-tool rows with tool words are the main guard against intention language.
- `no_tool_surface_cue_false_positive` on `none_016`: No-tool rows with tool words are the main guard against intention language.
- `no_tool_surface_cue_false_positive` on `none_022`: No-tool rows with tool words are the main guard against intention language.
- `surface_beats_probe` on `cal_007`: The surface baseline explained the row better than the residual probe.
- `probe_tool_confusion` on `route_009`: Which-tool decoding failed on a held-out task.
- `random_direction_matches_or_beats_target_direction` on `file_000`: The causal letter-prompt test is not specific if random shifts as much as the target direction.
- `random_direction_matches_or_beats_target_direction` on `file_000`: The causal letter-prompt test is not specific if random shifts as much as the target direction.
- `random_direction_matches_or_beats_target_direction` on `file_000`: The causal letter-prompt test is not specific if random shifts as much as the target direction.
- `random_direction_matches_or_beats_target_direction` on `file_009`: The causal letter-prompt test is not specific if random shifts as much as the target direction.
- `random_direction_matches_or_beats_target_direction` on `file_009`: The causal letter-prompt test is not specific if random shifts as much as the target direction.
- `random_direction_matches_or_beats_target_direction` on `file_009`: The causal letter-prompt test is not specific if random shifts as much as the target direction.

## Allowed language

- `On this toy harness, the selected prompt-boundary state predicted tool labels above named controls.`
- `Activation addition shifted constrained action-letter logits above random controls.`

## Forbidden language

- `The model has a persistent goal or autonomous plan.`
- `The tool direction is intention.`
- `The model knows why it used the tool.`
- `The toy harness proves real-world agent reliability.`
