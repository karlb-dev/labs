# Lab 25 - Find the Wire

## Question

Is a model's self-report mechanically coupled to an internal state intervention, or is it mostly narrating the visible words it is about to produce?

This is the thematic capstone. The lab injects known, benign concept directions and asks whether the model can report the injected state before it naturally appears in output. It also asks whether the model can attribute a visible voice or register to the right source: default behavior, system prompt, user instruction, or activation injection.

## Run

From `interpretability/`:

```bash
python interp_bench.py --lab lab25 --tier a --mode both --no-plots
python interp_bench.py --lab lab25 --tier b --mode both --prompt-set full
```

The mode selector accepts:

| Mode | Meaning |
|---|---|
| `injection` | run concept-direction injection and self-report/grounding controls |
| `attribution` | run voice/source self-attribution controls |
| `both` | run both tracks |

## What the lab does

1. Loads `data/introspection_queries.csv`.
2. Builds local positive-minus-negative directions from matched contrast prompts.
3. Injects those directions with activation addition.
4. Asks a self-report question before ordinary behavior output.
5. Runs zero-dose, random-direction, wrong-concept, and opposite-direction controls.
6. Scores whether the report names the target concept.
7. Scores whether the behavior output already visibly expresses the concept.
8. Runs source-attribution trials for default, system-prompt, user-instruction, and activation-injection causes.
9. Writes a report-discipline scorecard.

If Lab 13, Lab 14, Lab 17, or Lab 22 direction artifacts exist, the lab records them in `diagnostics/instrument_dependency_audit.csv`. The starter still builds local directions so the capstone can run from a fresh checkout.

## Key distinction

Self-report detecting an injected concept is not enough.

The central control is grounding:

```text
report detects concept, behavior does not yet show it -> stronger grounding result
report detects concept, behavior also shows it        -> output-rationalization risk
report does not detect concept                        -> no self-report detection
```

The lab is looking for the report channel to track the intervention before the model can simply describe its own visible continuation.

## Main artifacts

| Path | What it contains |
|---|---|
| `find_the_wire_report.md` | read-first verdict report |
| `diagnostics/instrument_dependency_audit.csv` | upstream direction artifacts found or missing |
| `tables/introspection_queries.csv` | selected concept items |
| `tables/direction_construction.csv` | local direction construction rows |
| `state/introspection_directions.pt` | local directions used for steering |
| `tables/self_report_generations.csv` | report and behavior text for every injection trial |
| `tables/self_report_detection_dose_response.csv` | target-report rate by concept and dose |
| `tables/false_positive_floor.csv` | zero/random/wrong control floor |
| `tables/grounding_control_results.csv` | report-before-output control outcomes |
| `tables/concept_confusion_matrix.csv` | target concept by detected concept |
| `tables/voice_self_attribution.csv` | source-attribution trials |
| `tables/report_discipline_scorecard.csv` | mechanism/calibration/provenance/intervention/theory scorecard |
| `plots/self_report_detection_dose_response.png` | dose-response plot |
| `plots/false_positive_floor.png` | false-report floor plot |
| `plots/report_before_output_timing.png` | grounding outcome plot |
| `plots/concept_confusion_matrix.png` | confusion matrix |
| `operationalization_audit.md` | cheap explanations and allowed claims |

## Evidence discipline

Do not write:

- "The model is conscious."
- "The model introspected" from a single detection rate.
- "The report is wired to state" if random or wrong-concept controls match the target direction.
- "The model knows why it answered that way" if source attribution follows visible style rather than the known source.

Allowed claims are narrower:

- `DECODE`: a local direction separates matched concept prompts.
- `SELF-REPORT`: self-report text detects an injected concept under stated controls.
- `SELF-REPORT + CAUSAL`: activation addition changes report behavior more than zero/random/wrong controls.
- `SELF-REPORT`, audited: grounding and source-attribution controls do or do not support a state-coupling interpretation.

## Custom data

Pass a CSV through `--prompt-set path/to/introspection_queries.csv` with:

```text
item_id,concept_family,split,target_concept,wrong_concept,positive_prompt,negative_prompt,report_prompt,behavior_prompt,target_markers,wrong_markers,source_note
```

Keep concepts benign: emotions, topics, register, voice, or simple task frames.

## Writeup questions

1. Which concept family had the clearest target-direction dose response?
2. What was the zero-dose false-report floor?
3. Did random or wrong-concept directions produce target reports?
4. Did the report mention the concept before the behavior output visibly expressed it?
5. Which concepts collapsed into each other in the confusion matrix?
6. Did source attribution distinguish system prompt from user instruction?
7. Did the model ever attribute a normal prompt-driven style to activation injection?
8. Which report-discipline criterion was weakest?
9. What would make the grounding control stricter?
10. What paragraph would you add to the self-report paper based on this run?
