# INCIDENT or1-001 — verbal-report token-form mismatch at the chat
# generation boundary (Qwen lane)

**Discovered:** 2026-08-08 ~23:20Z, during post-registration example
mining. **Scientific output observed before discovery:** yes —
`or1-qwen-verbal-report-v1` was registered with the defect present.

## What happened

RENDER_AND_POSITION_CONTRACT §6 prefers the **in-context** single-token
variant. The implementation (`rendering.preferred_token`) preferred the
leading-space form universally. That is correct for raw mid-text cells
(probe-swap, flexible generalization, ignition — prompts end without
trailing whitespace, so the continuation token carries the space) but
**wrong at the Qwen chat generation boundary**, where the rendered
prompt ends with `<think>\n\n</think>\n\n` and the in-context answer
form is the bare token (observed: baseline greedies `France`, `Blue`,
`Apple` are bare). In v1 the swap vector v_t and the scored candidate
rank both used the space form, so trials where the model behaviorally
adopted the candidate scored as misses (e.g. `color:Red` — clean rank
12, post-swap the model's top-1 IS `Red`, yet the scored ` Red` form
sat at rank 5).

## Effect direction

v1 **understates** the verbal-report effect. No other registered cell is
affected: probe-swap/FG vectors and scores use the correct in-context
space form; the span-tracking cells (selectivity, modulation, dual-task,
top-down) already score min over both forms via `synonym_token_ids`.
Verbal introspection had not run (its open-quote boundary has the same
property; fixed before first run).

## Correction (prospective; no v1 bytes edited)

1. `verbal_report.py` v2 scoring: candidate rank = **min over both
   single-token forms** (bare + space); swap target vector = the
   **boundary in-context (bare) form** when the scored position follows
   the generation boundary, space form otherwise. Skip/answer matching
   unchanged.
2. `introspection.py`: surface token = bare form at the open quote
   (same rule), scoring min over both forms.
3. Rerun the Qwen verbal-report cell at the next Qwen GPU slot;
   register `or1-qwen-verbal-report-v2` and supersede v1 with reason
   `token-form scoring defect at chat boundary (INCIDENT or1-001)`.
   OLMo lane runs v2 code from the start.

The v1 event remains in the registry (append-only); the report carries
this incident note wherever v1 numbers appear until v2 lands.
