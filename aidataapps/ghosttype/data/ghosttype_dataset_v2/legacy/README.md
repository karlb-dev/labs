# Dataset workshop — MSSQL Copilot chat-completion eval set

This directory builds `mssql_copilot_chat_completions.jsonl`, the eval set for
the planned **lab04** under `~/repos/labs/aidataapps/` (local-model comparison
on SQL typing-completion and NL2SQL scenarios: ANSI SQL and T-SQL across
Azure SQL, Fabric warehouse, and SQL Server admin / developer / devops work).
The dataset card — schema, scoring contract, catalog inventory, and counts —
is [`../dataset.md`](../dataset.md). This README is the operational side: how
the files fit together, how to rebuild, and how to fold new trace exports in.

## Layout

```
completions_traces/
  mssql-copilot-trace-*.json     raw VS Code MSSQL Copilot debug exports (read-only inputs)
  dataset.md                     dataset card (schema, scoring, counts)
  dataset/
    README.md                    this file
    build_dataset.py             deterministic builder: examples -> JSONL
    examples_trace.py            trace-derived examples (hand-cleaned, provenance kept)
    examples_synthetic.py        synthetic block 1 (grok-4.6 authored, v1.0.0)
    examples_synthetic_v2.py     synthetic block 2 (claude-fable-5 authored, v1.1.0)
    record.schema.json           JSON Schema for one JSONL line
    mssql_copilot_chat_completions.jsonl   the built dataset (do not hand-edit)
    schemas/                     frozen catalog snapshots + product rule text
```

Everything flows one way:

```
raw traces ──(manual curation)──> examples_trace.py ─┐
examples_synthetic*.py (authored against schemas/) ──┼─> build_dataset.py ─> JSONL
schemas/*.txt (frozen catalogs + rules) ─────────────┘
```

The JSONL is a build artifact. Never edit it directly; edit an examples file
or a schema snapshot and rebuild.

## Rebuild and validate

```bash
python3 dataset/build_dataset.py
```

The builder refuses duplicate ids and prints slice counts (kind, category,
expect_empty, difficulty, schema, scenario). After any change, also run the
self-checks that every gold must pass:

1. **Constraint self-check** — each record's `gold_completion` must satisfy its
   own `eval` block (`must_contain`, `must_not_contain`, `expect_empty`,
   no-semicolon for continuations). The reference checker is in
   [`../dataset.md`](../dataset.md) §"Automatic constraints".
2. **Grounding check** — every table/view/DMV referenced in a gold must appear
   in that record's `<schema_context>`. A gold that references an unlisted
   object is a bug in the example, full stop.
3. **Schema validation** — each line validates against `record.schema.json`
   (`pip install jsonschema` if needed).

A change is only done when all three pass for all records.

## How to ingest a new batch of trace files

The traces are VS Code MSSQL Copilot debug exports: JSON with an `events`
array. Each event carries `result`, `completionCategory`, `modelId`,
`promptMessages`, `rawResponse`, `schemaContextFormatted`, and a `locals`
blob with the editor context (`linePrefix`, `statementPrefix`,
`recentPrefix`, `lineSuffix`, `suffix`, `inferredSystemQuery`, ...). The
curation is deliberately manual — an agent does it, not a script — but the
policy is fixed:

1. **Drop** `cancelled` (debounce noise) and `error` (quota etc.) events.
2. **De-duplicate**: later exports replay earlier sessions of the same
   editing session. Key on (category, statementPrefix, linePrefix) and keep
   each unique prompt once, preferring the event with the richest metadata.
3. **Classify each surviving unique prompt:**
   - Origin output correct and grounded → keep as `gold_role="cleaned_trace"`.
     Light edits only: canonical layout, alias style, strip a stray fence.
   - Origin output wrong (invented objects, unit confusion, markdown fences,
     truncation, prose, suffix collision) but the *prompt* is a good test →
     keep the prompt, record the raw output in `origin_completion`, author a
     correct gold, mark `gold_role="replaced_trace_gold"`, and say why in
     `edit_notes`. Accepted-in-the-log does not mean correct (see
     `trace-cont-006`: the user accepted an invented DMV name).
   - The right behavior is an empty completion (unlisted objects, suffix
     already complete, nothing to continue) → gold is `""` with
     `eval.expect_empty=true`.
   - Prompt is garbage with no defensible gold (half-typed comments with no
     recoverable intent) → drop, and note the drop in `dataset.md` if it is a
     category of failure worth remembering.
4. **Snapshot the catalog** if the connection/schema context differs from the
   ones already in `schemas/`: freeze the `schemaContextFormatted` text as a
   new `schemas/<schema_id>.txt`, register it in `SCHEMA_FILES` /
   `SCHEMA_META` in `build_dataset.py`, and never retro-edit an existing
   snapshot (gold answers are grounded against the frozen text).
5. **Record provenance** on every kept item: `trace_file`, `event_id`,
   `exported_at`, `origin_model_id/-family/-vendor`, `origin_result`,
   latency/token counts when present, and the unedited `origin_completion`.
   This is what lets the eval report "did the source model already pass?"
   separately from "does the candidate match gold?".
6. Append the items to `examples_trace.py` (keep the existing `add(...)`
   shape), rebuild, run the three checks, and update the counts tables in
   `dataset.md`.

Useful spelunking snippet for a new export:

```python
import json
d = json.load(open("mssql-copilot-trace-<stamp>.json"))
for e in d["events"]:
    if e["result"] in ("cancelled", "error"):
        continue
    print(e["id"], e["result"], e["completionCategory"], e["modelId"])
    print("  raw:", (e.get("rawResponse") or "")[:100])
```

The `locals` field is a Python-repr string, not JSON; pull individual keys
with a regex or read them out of the printed blob rather than `json.loads`.

## Adding synthetic items

Author new items against the **frozen** snapshots in `schemas/` — pick the
snapshot first, then only use objects/columns it lists. Put new items in the
current synthetic block file (today: `examples_synthetic_v2.py`) with the
authoring model recorded via the block's `add()` defaults; if a different
model authors the next block, give it a new file (`examples_synthetic_v3.py`)
and wire it into `build_dataset.py`, so gold authorship stays a clean slice.

House rules for a good item:

- Every `must_contain` string must appear in the gold verbatim (the
  constraint checker is exact-substring); run the self-check to prove it.
- Traps should have **twins**: an environment where the request is
  answerable and one where the correct output is `""` (e.g. Query Store on
  `adventure_azure` vs the QDS empty-policy items on `ninjadb_a` /
  `fabric_warehouse`; wait stats on-prem vs Azure vs Fabric; SQL Agent on
  `msdb_onprem` vs `adventure_azure`).
- Continuations are one ghost-text unit, no trailing semicolon, and must
  compose with `current_line_suffix` — suffix-collision items where gold is
  `""` are some of the highest-signal items in the set.
- Set `scenario` explicitly on new items (`developer` / `admin` / `devops` /
  `analytics` / `metadata`); the tag-based inference in `build_dataset.py`
  is only a fallback for the v1 items.

## Versioning

`DATASET_VERSION` in `build_dataset.py` is the single source of truth.
Bump the minor version when items are added or golds change; record what
changed in the dataset card. v1.0.0 = 160 records (22 trace + 138 grok-4.6
synthetic). v1.1.0 = 265 records (+8 trace second pass, +97 claude-fable-5
synthetic, +3 catalogs, + `eval.scenario` facet).
