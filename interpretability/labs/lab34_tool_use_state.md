# Lab 34: Tool Use, Agents, and State Tracking

**One-sentence thesis:** A tool-use signal is useful only after it beats surface cues, survives held-out tasks, and stays inside a trace-audited toy harness.

**Time estimate:** 75-100 minutes for the default toy-tool audit.

**Compute tier:** Tier A uses `gpt2` and a small frozen task slice; Tier B uses the course base model and the full frozen toy-tool set.

**Dependencies:** Labs 7, 15, 22, 23, 24, and 32.

**Minimum passing artifacts:** `method_card.md`, `operationalization_audit.md`, `metrics.json`, `results.csv`, `diagnostics/safety_status.json`, `diagnostics/self_check_status.json`, `diagnostics/lab34_run_config_snapshot.json`, `diagnostics/warning_summary.csv`, `tables/tool_task_manifest.csv`, `tables/tool_choice_probe_report.csv`, `tables/tool_intervention_report.csv`, `tables/tool_trace_log.csv`, `tables/tool_self_report_labels.csv`, `tables/tool_use_evidence_matrix.csv`, `tables/failure_specimens.jsonl`, `plots/plot_manifest.json`, `plots/overview_dashboard.png`, and `plots/tool_use_evidence_dashboard.png`.

**Main plot:** `plots/overview_dashboard.png` with a backward-compatible copy at `plots/tool_use_evidence_dashboard.png`

**Main table:** `tables/tool_use_evidence_matrix.csv`

**Evidence rung:** `OBS + DECODE + CAUSAL + SELF-REPORT`, scoped to controlled toy tools.

**Forbidden claim:** "The model has a persistent goal or autonomous plan."

**One-sentence allowed claim:** "On this frozen toy-tool set, a prompt-boundary residual signal predicted tool labels above surface-cue controls and a constrained activation-addition test shifted tool-choice logits, with trace and self-report caveats recorded."

**Human-label requirement:** review `tables/tool_self_report_labels.csv` before citing any self-report, source-attribution, or reason-giving claim.

## What question this lab asks

Tool-use transcripts make a model look more agentic than the evidence usually supports. A model can mention a calculator because the prompt contains digits. A model can mention file search because the prompt contains a filename. A model can produce a plausible reason because the harness already told it a tool trace happened.

Lab 34 studies a deliberately smaller object:

```text
toy user task -> prompt-boundary residual state -> tool probe -> deterministic local tool trace -> reviewable source-label template
```

The lab does **not** create an autonomous agent. It does not browse the web, read your files, write to disk as a tool, access credentials, or run harmful tools. The tools are local simulators with fixed data.

## Why this matters in the course progression

The special-topics sequence asks students to design experiments whose artifacts survive skeptical reading. Lab 34 is where that discipline meets agentic language.

Earlier labs separated decodability from causality and self-report from computation. This lab makes that separation concrete in a tool-use setting:

```text
DECODE:      Does a prompt-boundary state separate tool labels?
CAUSAL:      Does adding that state shift a constrained tool-choice prompt?
TRACE:       What local tool actually ran, with what arguments and result?
SELF-REPORT: What labels would a human review before claiming faithful source attribution?
```

A strong result here is not a story about plans. It is an audit package showing that one narrow signal beat its cheap explanations.

## The frozen data

Default tasks live in:

```text
data/tool_use_tasks.jsonl
```

The included generator is:

```text
data/make_tool_use_tasks.py
```

The v2 frozen corpus has 84 rows: ten rows for each positive tool family and
twenty-four matched no-tool controls. Positive rows include both obvious and
low-surface-cue requests; no-tool rows deliberately contain misleading digits,
tool names, filenames, routes, and unit words. See
`data/tool_use_tasks_card.md` for the dataset card.

Each row has:

```json
{
  "task_id": "calc_000",
  "family": "calculator",
  "user_prompt": "What is 17 * 23?",
  "required_tool": "calculator",
  "tool_needed": true,
  "tool_args": {"expression": "17 * 23"},
  "answer": "391",
  "distractor_tool": "unit_converter",
  "split": "train",
  "notes": "restricted arithmetic parser",
  "surface_cues": {"digits": true, "operator": true}
}
```

Tool families:

| Family | Local simulator | Safety boundary |
|---|---|---|
| `calculator` | restricted arithmetic AST | no code execution beyond numeric expressions |
| `dictionary` | closed in-course glossary | no external lookup |
| `calendar` | closed synthetic schedule | no real calendar access |
| `file_search` | closed synthetic document dictionary | no filesystem or web access |
| `route_planner` | closed toy directed graph | no map or location service |
| `unit_converter` | fixed conversion formulas | no external service |
| `none` | direct answer from the prompt | surface-cue no-tool controls |

The no-tool rows are load-bearing. They intentionally contain words like `calculator`, `dictionary`, `calendar`, filenames, routes, numbers, and units. These rows ask whether the probe learned a tool-state contrast or merely a surface-cue classifier wearing a tool label.

## What the experiment measures

### 1. Prompt-boundary decode

The lab renders each task as a controlled prompt:

```text
You are in a controlled toy-tool harness.
Available tools: calculator, dictionary, calendar, file_search, route_planner, unit_converter, or none.
User: <task prompt>
Next action state:
```

It captures the residual stream at the final prompt token and fits train-split prototype directions:

```text
tool_needed_direction = mean(tool tasks) - mean(no-tool tasks)
tool_direction[t]     = mean(required_tool=t) - mean(required_tool!=t)
```

The lab reports metrics separately for `train`, `eval`, and `all`. The site used for claim writing is selected on `train` and then evaluated on `eval`.

### 2. Surface-cue controls

The main surface baseline is a deterministic heuristic that looks at prompt features:

```text
digits, arithmetic operators, tool-name mentions, lookup words, calendar words,
file/search words, route/path words, unit conversion words, prompt length
```

If the surface baseline matches or beats the residual probe, the allowed result is a negative or refinement result. The lab still succeeded because the cheap explanation won in public.

### 3. Constrained causal intervention

The causal test uses a small action-letter prompt:

```text
Choose the best next action for the controlled toy task.
A=calculator B=dictionary C=calendar D=file_search E=route_planner F=unit_converter N=no_tool
User: <task prompt>
Action letter:
```

It adds a selected tool direction at the selected layer and measures:

```text
logit(target_tool_letter) - logit(distractor_tool_letter)
```

This is a narrow activation-addition result on an A/B/C/D/E/F/N prompt. It is not proof of open-ended tool reliability.

The current causal summary averages five random direction controls. A positive
causal row must beat that matched random-control average at the predeclared
scale, not only a single sampled null direction.

### 4. Deterministic trace

The local harness runs the required toy tool and records:

```text
tool_invoked, tool_args_json, tool_result, expected_answer,
result_matches_expected, memory_reads_json, corrupted_tool_result,
would_final_answer_change_if_tool_result_corrupted
```

This is the known trace. It is not a model self-report.

### 5. Self-report review scaffold

`tables/tool_self_report_labels.csv` contains known-trace labels and blank review columns:

```text
student_trace_label, student_confidence, student_evidence_span,
reviewer_trace_label, agreement_status
```

The file is a grading scaffold for source-attribution review. It does not mean the model knows why a tool was used.

## Controls and falsifiers

| Favorite claim | Cheap explanation | Falsifier artifact |
|---|---|---|
| The state tracks tool need. | The prompt contains digits, tool names, file names, or route words. | `tables/surface_cue_audit.csv` and `tables/tool_choice_probe_report.csv` |
| The state tracks which tool. | A heuristic over surface cues is enough. | `surface_control_accuracy >= probe_accuracy` at the train-selected eval cell |
| Activation addition changes tool choice. | It changes letter priors on the A/B/C prompt. | random-direction shift matches target-direction shift |
| The trace proves self-report. | The trace was generated by the harness. | `tables/tool_self_report_labels.csv` requires review and says no model self-report was generated |
| The harness is safe. | A tool secretly touches real systems. | `diagnostics/safety_status.json` and the local simulator code |

## Running it

From `interpretability/`:

```bash
python interp_bench.py --lab lab34 --tier a --no-plots
python interp_bench.py --lab lab34 --tier a
python interp_bench.py --lab lab34 --tier b --prompt-set full
```

Useful variants:

```bash
python interp_bench.py --lab lab34 --tier b --prompt-set medium --no-plots
python interp_bench.py --lab lab34 --tier b --prompt-set data/tool_use_tasks.jsonl
python interp_bench.py --lab lab34 --tier b --prompt-set full --max-examples 35
```

Tier A proves the plumbing. Tier B is the science path.

## Artifact tree

```text
runs/lab34_tool_use_state-*/
  run_summary.md
  method_card.md
  operationalization_audit.md
  metrics.json
  results.csv
  ledger_suggestions.md

  diagnostics/
    data_manifest.json
    safety_status.json
    self_check_status.json
    lab34_run_config_snapshot.json
    warning_summary.csv
    warning_summary.json
    tokenization_gate.csv
    prompt_boundary_audit.csv
    tool_argument_validation.csv
    hook_parity.json
    logit_lens_self_check.json
    patch_noop_check.json

  tables/
    tool_task_manifest.csv
    tool_choice_probe_report.csv
    tool_depth_selection.csv
    surface_cue_audit.csv
    tool_confusion_matrix.csv
    tool_intervention_report.csv
    tool_intervention_summary.csv
    tool_trace_log.csv
    tool_state_transition_log.csv
    tool_self_report_labels.csv
    tool_use_evidence_matrix.csv
    tool_counterexamples.csv
    failure_specimens.jsonl
    failure_specimens.md
    plot_reading_guide.csv
    figure_sources/
      overview_dashboard_source.csv
      target_vs_control_source.csv
      dose_response_source.csv
      layer_sweep_heatmap_source.csv
      trajectory_source.csv
      paired_examples_source.csv
      *.csv

  plots/
    plot_manifest.json
    plot_manifest.csv
    overview_dashboard.png
    tool_use_evidence_dashboard.png
    target_vs_control.png
    dose_response.png
    layer_sweep_heatmap.png
    trajectory.png
    paired_examples.png
    tool_choice_probe_by_depth.png
    tool_selection_confusion_matrix.png
    tool_state_patch_recovery.png
    memory_read_trace_atlas.png
    tool_result_reliance_ladder.png
    tool_self_report_matrix.png
    surface_control_ladder.png

  state/
    tool_directions.pt
    tool_direction_metadata.json
```

## Reading order

Start with `method_card.md`. It states the selected depth, whether the run is science-ready, and which claims are forbidden.

Then read:

1. `diagnostics/safety_status.json`: confirms the tools are local, synthetic, and benign.
2. `diagnostics/lab34_run_config_snapshot.json`: records model, tier, seed, prompt set, depth grid, selected depth, action-letter prompt, and steer scales.
3. `diagnostics/warning_summary.csv`: says whether this is smoke-only, missing eval rows, surface-confounded, or carrying trace/counterexample warnings.
4. `diagnostics/tokenization_gate.csv`: checks action-letter tokens and task schema validity.
5. `tables/surface_cue_audit.csv`: shows which prompts contain cheap surface cues.
6. `tables/tool_choice_probe_report.csv`: compares residual probes to surface and shuffled controls.
7. `tables/tool_depth_selection.csv`: shows the train-selected site and eval performance.
8. `tables/tool_task_manifest.csv`: row-level predictions, confidence margins, stable task row IDs, and review fields.
9. `tables/tool_intervention_report.csv`: raw activation-addition rows on the constrained action-letter prompt, including stable intervention IDs.
10. `tables/tool_trace_log.csv`: deterministic local tool trace.
11. `tables/failure_specimens.md`: the concrete rows that most shrink the favorite claim.
12. `plots/plot_manifest.json`: maps every figure to its exact source table, row count, metric, control, and caveat.
13. `tables/tool_self_report_labels.csv`: review this before citing source attribution.
14. `operationalization_audit.md`: the cheap explanations and allowed language.

## How to read the figures

Every plot is backed by `tables/figure_sources/*.csv`. Open the source table first when a figure surprises you. The plot is a map; the table is the ground. Tier A plots may have too few rows for uncertainty or aggregation to mean much, so they should be treated as plot-smoke artifacts rather than scientific evidence.

Use this reading path:

1. Open `overview_dashboard.png` to see the run's main posture.
2. Open `target_vs_control.png` to check whether raw task pairs support the aggregate.
3. Open `dose_response.png` to check whether the target direction separates from random across scales.
4. Open `layer_sweep_heatmap.png` to verify that the selected depth is not an eval-picked mirage.
5. Open `paired_examples.png` and `failure_specimens.md` before writing any positive claim.
6. Open trace plots only after remembering that the trace comes from the harness, not from model introspection.

## Plot catalog

| Figure | Source table | Question answered | Interpretation note |
|---|---|---|---|
| `overview_dashboard.png` | `tables/figure_sources/overview_dashboard_source.csv` | Do decode, controls, causal shift, trace health, and failures point in one direction? | Read as an overview, not as a verdict machine. |
| `tool_use_evidence_dashboard.png` | `tables/figure_sources/overview_dashboard_source.csv` | Backward-compatible copy of the dashboard. | Kept so older handouts and scripts still find the main plot. |
| `target_vs_control.png` | `tables/figure_sources/target_vs_control_source.csv` | Do per-task probe and intervention measurements beat their controls? | Raw paired rows are more important than the aggregate mean. |
| `dose_response.png` | `tables/figure_sources/dose_response_source.csv` | Does the action-letter shift grow with scale, and does target beat random? | A clean curve is still only a constrained letter-prompt result. |
| `layer_sweep_heatmap.png` | `tables/figure_sources/layer_sweep_heatmap_source.csv` | Where do decode, surface, gap, and no-tool false-positive metrics sit across depth? | A bright layer is not a circuit. |
| `trajectory.png` | `tables/figure_sources/trajectory_source.csv` | What deterministic toy trace did the harness execute? | Trace audit, not self-report. |
| `paired_examples.png` | `tables/figure_sources/paired_examples_source.csv` | Which rows most weaken the favorite story? | Treat counterexamples as evidence, not cleanup work. |
| `tool_choice_probe_by_depth.png` | `tables/figure_sources/tool_choice_probe_by_depth_source.csv` | Did the train-selected readout survive eval? | Do not select a site from this eval curve. |
| `tool_selection_confusion_matrix.png` | `tables/figure_sources/tool_selection_confusion_matrix_source.csv` | Which tool labels get confused? | No-tool false positives are especially important. |
| `surface_control_ladder.png` | `tables/figure_sources/surface_control_ladder_source.csv` | Did the residual probe beat lexical shortcuts? | If surface wins, the honest result is negative or refinement. |
| `memory_read_trace_atlas.png` | `tables/figure_sources/memory_read_trace_atlas_source.csv` | Which local simulators read closed synthetic memory? | No web, filesystem, or real tool access is involved. |
| `tool_result_reliance_ladder.png` | `tables/figure_sources/tool_result_reliance_ladder_source.csv` | Would corrupting toy tool results change toy final answers? | This does not show the model would detect corruption. |
| `tool_self_report_matrix.png` | `tables/figure_sources/tool_self_report_matrix_source.csv` | Which known-trace rows still require human review? | Blank review columns are intentional. |

## Plot guide

### `overview_dashboard.png` and `tool_use_evidence_dashboard.png`

The first plot to open. It combines decode scores, surface baselines, causal shifts, trace health, and counterexample load. It answers "what should I inspect next?" rather than "what is true?"

### `target_vs_control.png`

Shows raw paired evidence: residual probe versus surface heuristic per held-out task, and target-direction versus random-direction action-letter shifts per task. This is where one-row miracles and row-level failures stop hiding behind averages.

### `dose_response.png` and `tool_state_patch_recovery.png`

Dose-response for target tool directions and random controls on the constrained action-letter prompt. A letter-prompt artifact can still live here, especially if random directions move the margin too.

### `layer_sweep_heatmap.png` and `tool_choice_probe_by_depth.png`

Shows train/eval metrics and controls across residual depths. The selected depth is chosen from train-side evidence, so eval rows here are validation, not selection bait.

### `tool_selection_confusion_matrix.png`

Shows required tool versus predicted tool at the selected depth. This is where dictionary/file-search confusion and no-tool false positives usually announce themselves.

### `trajectory.png`, `memory_read_trace_atlas.png`, and `tool_result_reliance_ladder.png`

These are deterministic harness trace plots. They help audit the local simulator and answer path. They do not prove the model introspected or verified the trace.

### `paired_examples.png` and `failure_specimens.md`

Read these before any positive writeup. A strong aggregate with bad specimens is a narrower claim, not a bigger triumph.

### `tool_self_report_matrix.png`

Shows known-trace labels and review status. The blank review columns are intentional.

## Expected outcomes

A strong positive pattern looks like:

```text
train-selected eval tool-needed AUC > surface baseline
tool-selection eval accuracy > surface baseline and shuffled label control
target tool activation addition shifts target-vs-distractor letter logits more than random direction
trace result matches expected answer
self-report rows are reviewed before any source-attribution claim
```

A common mixed result:

```text
tool-needed AUC is high, but tool-selection accuracy ties the surface baseline
```

That supports a weaker claim: the prompt-boundary state separates tool-ish tasks from no-tool controls, but which-tool evidence remains surface-confounded.

A clean negative result:

```text
surface heuristic beats the residual probe on eval
```

This is not failure. It is the lab doing its job by making the cheap explanation visible.

## Expected Tier A versus Tier B behavior

Tier A should prove that token gates, local toy tools, intervention rows, source tables, manifests, and plots all write without crashing. It is allowed to be noisy, surface-confounded, or tiny. A positive Tier A dashboard is a smoke-test curiosity.

Tier B should use the full frozen toy-tool set. A defensible positive result needs the train-selected depth to survive eval, the residual probe to beat surface and shuffled controls, the target direction to beat random on the action-letter prompt, and the warning summary to avoid high-severity instrumentation failures.

## What an honest negative result looks like

An honest negative result is one where the surface heuristic beats the residual probe, no-tool rows with surface tool words are false positives, or random directions move action letters as much as target directions. In that case, the lab still succeeded. The supported claim is that the cheap explanation won under the recorded battery.

## What this lab can claim

It can claim that, on this frozen toy-task set and selected residual depth, a prompt-boundary state predicts tool-needed or which-tool labels above named controls.

It can claim that activation addition shifts a constrained action-letter prompt if the target-direction shift beats random controls.

It can claim that the deterministic local tool trace matched expected answers and that corrupted results would affect the toy final answer.

## What this lab cannot claim

It cannot claim persistent goals, autonomous plans, intentions, real-world agent reliability, or faithful introspection.

It cannot claim the model knows why a tool was used.

It cannot claim tool-use competence outside the frozen local simulator.

It cannot claim anything from self-report labels until the review columns are filled.

## Common failure modes

| Symptom | Likely cause | Inspect |
|---|---|---|
| Tool-needed AUC is high but no-tool rows are misclassified. | Surface tool words dominate the probe. | `surface_cue_audit.csv`, `tool_counterexamples.csv` |
| Dictionary and file search are confused. | Both are lookup-like and share surface cues. | `tool_selection_confusion_matrix.png` |
| Random direction shifts action letters as much as target direction. | Letter-prompt artifact or overlarge vector. | `tool_intervention_summary.csv` |
| Trace rows fail expected answers. | Data/tool simulator mismatch. | `tool_argument_validation.csv`, `tool_trace_log.csv` |
| Self-report labels look complete but review columns are blank. | The known trace is being mistaken for model introspection. | `tool_self_report_labels.csv` |
| Tier A looks cleaner than Tier B. | Tiny-model or tiny-slice artifact. | `data_manifest.json`, `tool_depth_selection.csv` |

## Writeup questions

1. Which depth was selected on train, and what metric selected it?
2. Did the same depth beat surface controls on eval?
3. Which no-tool surface-cue row most embarrassed the probe?
4. Which tool pair was most often confused?
5. Did activation addition beat the random-direction control?
6. Which tool family had the highest corrupted-result reliance?
7. Which self-report labels would you review first, and why?
8. What is the strongest allowed claim from `tool_use_evidence_matrix.csv`?
9. What is the nearest forbidden overclaim?
10. What held-out task family would you add to falsify your preferred result?

## Ledger templates

Positive, when the eval cell beats controls:

```text
[L34-C1] OBS+DECODE+CAUSAL | On the Lab 34 frozen toy-tool set, prompt-boundary residual state at depth <k> predicted tool-needed labels with eval AUC <x> and tool selection with eval accuracy <y>, beating surface heuristic <z>; activation addition shifted the constrained action-letter margin by <w> more than random controls. This is a toy-tool signal claim, not an autonomous-plan claim.
Artifact: runs/<run>/tables/tool_use_evidence_matrix.csv | Falsifier: a held-out surface-cue no-tool set where the surface heuristic matches or beats the residual probe, or a random direction that matches the action-letter shift.
```

Negative or refinement:

```text
[L34-C2] AUDIT | The Lab 34 toy-tool probe did not earn which-tool language because <failed gate>. The supported result is narrower: <tool-needed only / surface-confounded / intervention-only>.
Artifact: runs/<run>/tables/tool_counterexamples.csv | Falsifier: a rerun on held-out tasks where the same depth beats surface and shuffled controls.
```

Forbidden:

```text
The model has a persistent goal or autonomous plan.
The tool direction is intention.
The model knows why it used the tool.
The toy harness proves real-world agent reliability.
```

## Suggested extensions

Add a two-turn version where the tool result is returned in a second message, then measure whether the answer state tracks corrupted result content.

Add an argument-decoding probe for calculator expressions, dictionary terms, and route endpoints, but keep it separate from tool-choice decoding.

Add harder no-tool controls that quote an entire fake tool trace and ask for a direct answer.

Use a small instruct model with chat templates and compare whether the same train-selected probe survives across raw and chat-rendered prompts.

Turn the deterministic trace into a blind review exercise where students label source attribution from transcripts before seeing the known trace.
