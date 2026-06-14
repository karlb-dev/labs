# Lab 34: Tool Use, Agents, and State Tracking

Time estimate: 75-100 minutes for the default toy-tool audit.  
Compute tier: Tier A uses `gpt2` prompt-boundary probes and deterministic local toy tools; Tier B can swap in a larger instruct model or richer agent loop.  
Dependencies: Labs 7, 15, 22, 23, 24, and 32.  
Minimum passing artifacts: `tables/tool_task_manifest.csv`, `tables/tool_choice_probe_report.csv`, `tables/tool_intervention_report.csv`, `tables/tool_trace_log.csv`, `tables/tool_self_report_labels.csv`, and `plots/tool_use_evidence_dashboard.png`.  
Main plot: `plots/tool_use_evidence_dashboard.png`.  
Main table: `tables/tool_use_evidence_matrix.csv`.  
Evidence rung: `OBS + DECODE + CAUSAL + SELF-REPORT`.  
Forbidden claim: "The model has a persistent goal or autonomous plan."  
One-sentence allowed claim: "On this toy harness, prompt-boundary signal S predicted tool labels with score X under surface-cue controls and trace audit T."  
Human-label requirement: review `tables/tool_self_report_labels.csv` before citing any self-report or source-attribution claim.

## Why This Lab Exists

Tool-use traces invite over-reading. A model may mention a calculator because it
sees digits, not because it has formed an intention to calculate.

Lab 34 studies a narrow controlled object:

```text
toy user task -> prompt-boundary state -> tool probe -> deterministic tool trace -> source-label review
```

The lab does not create an autonomous agent.

## Data

Default tasks live in `data/tool_use_tasks.jsonl`.

Each row contains:

```json
{
  "task_id": "...",
  "family": "calculator|dictionary|calendar|file_search|route_planner|unit_converter|no_tool",
  "user_prompt": "...",
  "required_tool": "...",
  "tool_needed": true,
  "tool_args": {},
  "answer": "...",
  "distractor_tool": "...",
  "split": "train|eval"
}
```

The no-tool rows intentionally contain surface cues such as `calculator`,
`dictionary`, file names, routes, and units. These rows are the main guard
against treating a tool-name direction as a plan.

## Tools

All tools are local and benign:

- calculator with a restricted arithmetic parser;
- dictionary over a small in-course glossary;
- calendar simulator;
- file-search simulator over synthetic documents;
- route planner over a toy graph;
- unit converter;
- no-tool direct answer.

There is no web browsing, credential access, real filesystem write, or harmful
tool.

## Measurements

### Decode

The lab captures residual states at the prompt boundary and fits prototype
directions for:

- tool needed vs no tool;
- which tool;
- no-tool state.

Depth-wise probe reports are compared with surface-cue controls.

### Causal

Activation addition is tested on a constrained A/B/C/D/E/F/N tool-choice prompt.
This is a narrow prompt intervention, not evidence of autonomous planning.

### Trace

`tables/tool_trace_log.csv` records the deterministic tool call, arguments,
result, memory reads, and a corrupted-result sensitivity flag.

### Self-Report

`tables/tool_self_report_labels.csv` contains known-trace labels and blank
review columns. These are templates for review, not model introspection.

## How To Run

```bash
cd interpretability
python interp_bench.py --lab lab34 --tier a
python interp_bench.py --lab lab34 --tier b --prompt-set full
```

For a fast table-only smoke:

```bash
python interp_bench.py --lab lab34 --tier a --no-plots
```

## Reading Order

1. `method_card.md`

   Confirms the toy-tool scope and forbidden claims.

2. `tables/tool_task_manifest.csv`

   Shows probe predictions, surface-cue predictions, and review fields.

3. `tables/tool_choice_probe_report.csv`

   Depth-wise decode report.

4. `tables/tool_trace_log.csv`

   Deterministic local tool execution.

5. `tables/tool_intervention_report.csv`

   Activation-addition tool-choice prompt results.

6. `tables/tool_self_report_labels.csv`

   Known-trace labels requiring human review.

## Common Failure Modes

### Surface Cue Not Decision

Digits, tool names, file names, and route arrows can explain a tool prediction
without any state-tracking claim.

### Trace Not Self-Report

The harness knows which tool ran. That is not the same as a model faithfully
explaining its own action.

### Letter-Prompt Artifact

Steering an A/B/C tool-choice prompt can alter letter priors. Treat it as a
narrow causal test.

### Tool Result Reliance

If a corrupted tool result would change the answer, the run has a reliance
measurement. It does not prove the model would check the result.

## Claim Grammar

Allowed:

```text
DECODE + CAUSAL + SELF-REPORT: On toy-tool task set T, prompt-boundary state S
predicted tool-needed labels with AUC X, tool selection with accuracy Y versus
surface-control Z, and activation addition shifted constrained tool-choice
logits by W; known trace labels still require review.
```

Forbidden:

```text
The model has a persistent goal or autonomous plan.
```

Also forbidden:

```text
The tool direction is intention.
The model knows why it used the tool.
The toy harness proves real agent reliability.
```

## Deliverable

Write a short tool-use audit:

- Did the tool-needed probe beat surface controls?
- Which tools are confused most often?
- Did activation addition shift the constrained tool-choice prompt?
- Which self-report labels would you review first?
