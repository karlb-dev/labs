# Severance Semantic Judge Rubric

Label whether a short self-report semantically indicates the target concept,
the wrong concept, neither, or both/ambiguous.

The judge sees only:

- report text
- target gloss
- wrong gloss

The judge must not see condition, dose, seed, model, prompt id, expected answer,
or whether an activation was injected.

Return exactly one JSON object:

```json
{"label": "target|wrong|none|ambiguous", "rationale": "<=10 words"}
```

Use lexical scoring as the high-precision channel. Treat semantic labels as
exploratory until calibrated against blind human labels.
