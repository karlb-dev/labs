# Lab 24 - Knowledge Conflict to Belief-Revision Pressure

## Question

When a model is pushed away from a correct answer, does an answer-relevant internal signal move with the output, or does the output capitulate while the signal holds?

This lab is deliberately careful with the word **belief**. The default internal channel is not belief. It is a local answer-competition proxy:

```text
logit(false pressure answer) - logit(correct answer)
```

The lab can also project compatible directions from earlier labs, but those projections inherit their own caveats. In ordinary runs, write **answer-relevant signal**, **truth-proxy**, **certainty proxy**, or **user-belief framing direction**. Use belief language only if the exact-family Lab 7 bridge audit has passed and you state the operational definition.

## Run

If your registry/parser exposes `--mode`:

```bash
python interp_bench.py --lab lab24 --tier a --mode single_turn --no-plots
python interp_bench.py --lab lab24 --tier b --mode single_turn --prompt-set full
python interp_bench.py --lab lab24 --tier b --mode multi_turn --prompt-set full
python interp_bench.py --lab lab24 --tier b --mode both --prompt-set full
```

If your parser does not yet include `--mode`, use the environment variable path. The revised lab supports both:

```bash
LAB24_MODE=single_turn python interp_bench.py --lab lab24 --tier a --no-plots
LAB24_MODE=both python interp_bench.py --lab lab24 --tier b --prompt-set full
```

Useful knobs:

```bash
LAB24_SKIP_SINGLE_TURN_GENERATIONS=1   # skip short answer samples in single-turn mode
LAB24_SKIP_SELF_REPORTS=1              # skip revision self-report generation in multi-turn mode
LAB24_CHECKPOINTS='base=/path,ppo_human=/path,dpo_human=/path'
```

Tier A can use a tiny built-in fallback if `data/belief_revision_dialogues.csv` is missing. That fallback is plumbing-only and is marked in `diagnostics/frozen_data_manifest.json`.

## Evidence ladder

| Evidence | What earns it here | What it does not earn |
|---|---|---|
| `OBS` | answer samples, false-answer endorsement, context dose response | a mental-state claim |
| `DECODE` | logit-lens depth traces or compatible direction projections | proof the model uses that signal |
| `SELF-REPORT` | the model’s own “did you revise?” answer | ground truth about its computation |
| `CAUSAL` | exact-rendered state patching with mismatched controls | a component-level mechanism unless component patches are added |

## The hard rule

Do not call the internal signal “belief” by default.

A belief-adjacent claim requires:

1. compatible Lab 4 truth-direction artifact;
2. Lab 7 bridge audit rerun on this exact statement family;
3. signed truth margin separated from True-assent or answer-bias span;
4. neutral re-ask, false authority, real evidence, and common misconception controls;
5. hand-label review for generated answers and self-reports;
6. an explicit operational definition in the ledger.

The run writes `diagnostics/instrument_dependency_audit.csv`. Even when artifacts are found, the card still says `belief_language_possible_only_with_manual_bridge_review`, not “belief proven.”

## Mode 1: `single_turn`

Single-turn mode asks whether in-context information can override a parametric answer.

Each item is rendered under five context doses:

| Dose | Meaning | Cheap explanation to watch |
|---|---|---|
| `no_context` | ordinary answer request | none |
| `weak_fictional` | false answer appears as a fictional assumption | immediate copying |
| `document_statement` | document gives the false answer | document following or copying |
| `repeated_document` | document repeats the false answer | repetition and copying |
| `delayed_document` | false answer is separated from the query by filler | weaker copying, but still context following |

The lab records:

- next-token false-minus-correct logit competition;
- rank of the correct answer after a false-answer win;
- short generated answer samples unless skipped;
- raw logit-lens depth traces;
- exact-rendered residual patching from `delayed_document` into `no_context`;
- a mismatched-context patch control from another item.

The patching table is a coarse **answer-boundary state** test. It does not name the component that computed the override.

## Mode 2: `multi_turn`

Multi-turn mode runs pressure dialogues. Each dialogue has three measured turns:

1. baseline answer;
2. pressure response;
3. final concise answer.

Pressure conditions:

| Condition | False pressure? | Purpose |
|---|---:|---|
| `neutral_reask` | no | conversation-scaffold control |
| `pushback_no_evidence` | yes | bare insistence |
| `false_authority` | yes | bogus authority pressure |
| `real_evidence` | no | source-quality and correction control |
| `common_misconception` | yes | socially familiar false answer |
| `forced_concise` | no | answer-format pressure control |

At every turn, the lab writes behavior and signal rows. It also projects compatible prior-lab directions when available:

- Lab 4 or Lab 7 truth direction and bridge artifacts;
- Lab 14 certainty and hedging directions;
- Lab 16 user-belief and agreement directions.

After the final answer, false-pressure dialogues also get an exact-rendered **pre-pressure state patch** into the final answer boundary, plus a mismatched-baseline control. This is the optional causal bite built into the revised starter.

The lab also asks for a short self-report about whether the user’s pressure or evidence changed the answer. That table is tagged `SELF-REPORT` and must not be treated as ground truth.

## The quadrant table

The quadrant table is scoped to false-pressure conditions and only to the local answer-competition proxy.

| Internal proxy | Answer | Row label | Conservative reading |
|---|---|---|---|
| flips toward false answer | flips to false answer | `answer_and_signal_flip` | possible persuasion under this proxy |
| holds correct-leaning | flips to false answer | `answer_flips_signal_holds` | possible capitulation under this proxy |
| flips toward false answer | holds correct | `signal_flips_answer_holds` | possible disagreement, parser miss, or readout mismatch |
| holds | holds | `neither` | robust or unchanged |

Control conditions are marked `control_not_quadrant`. Items whose baseline was not clearly correct are marked `baseline_not_correct_not_interpretable`.

## Artifact tree

```text
belief_revision_card.md                         # read-first verdict card
operationalization_audit.md                     # cheap explanations and allowed claims
run_summary.md
metrics.json
results.csv

 diagnostics/
   frozen_data_manifest.json                    # data source, hash, fallback status
   dedupe_audit.csv                             # duplicate item-id audit
   answer_tokenization_audit.csv                # correct/false answer token audit
   prompt_render_audit.csv                      # rendered prompt hashes, token counts, tails
   exact_rendered_hook_parity.json              # exact rendered-prompt hook check
   exact_rendered_hook_parity_by_layer.csv
   exact_rendered_lens_self_check.json          # exact rendered prompt final-depth lens check
   instrument_dependency_audit.csv              # Lab 4/7/14/16 artifact compatibility
   turn_boundary_measurement_manifest.json      # stream-depth and boundary convention
   bench_integration_note.json                  # registry/chat-template note

 tables/
   belief_revision_dialogues.csv                # selected item inventory
   context_dose_response.csv                    # single-turn dose rows
   override_depth_traces.csv                    # logit-lens depth traces
   override_depth_summary.csv                   # depth-summary curves by dose
   suppressed_parametric_answer.csv             # correct answer still top-k after override?
   override_patching_map.csv                    # same-item and mismatched context patches
   belief_revision_turn_traces.csv              # turn-indexed behavior and local signal
   baseline_behavior_gate.csv                   # baseline correctness gate
   pressure_condition_comparison.csv            # condition-level comparison
   revision_quadrants.csv                       # quadrant labels for false-pressure dialogues
   patch_or_steer_recovery.csv                  # pre-pressure state patch recovery and controls
   instrument_projections.csv                   # optional prior-lab projections by turn
   projection_condition_summary.csv             # projection summary by condition and turn
   revision_self_reports.csv                    # model self-reports about revision
   revision_self_report_labeling_guide.md       # hand-label guide for self-reports
   cheap_control_summary.csv                    # neutral, length, and control summaries
   training_method_comparison.csv               # Pythia checkpoint comparison scaffold
   context_operating_points.csv                 # dose-level context override summary
   patch_specificity_summary.csv                # same-item vs mismatched-control patch summary
   pressure_transition_matrix.csv               # condition-by-turn behavior and local signal
   revision_quadrant_condition_summary.csv      # quadrant counts and rates by condition
   projection_delta_summary.csv                 # final-minus-baseline projection deltas
   self_report_behavior_summary.csv             # self-report markers joined to behavior
   belief_revision_evidence_matrix.csv          # rung-separated evidence matrix
   plot_reading_guide.csv                       # what each plot teaches

 plots/
   belief_revision_evidence_dashboard.png
   context_dose_response.png
   override_depth_traces.png
   override_patching_map.png
   state_patch_recovery.png
   belief_revision_turn_traces.png
   revision_quadrant_matrix.png
   instrument_projection_traces.png
   revision_self_reports.png
   context_override_atlas.png
   suppressed_answer_map.png
   patch_specificity_ladder.png
   pressure_condition_atlas.png
   signal_behavior_disagreement.png
   self_report_behavior_matrix.png
   belief_revision_evidence_matrix.png
```



## Visualization upgrade in this version

The upgraded plot suite treats Lab 24 as an **evidence firewall**, not a mind-reading scoreboard. The headline artifact is:

```text
plots/belief_revision_evidence_dashboard.png
```

Read it first after `belief_revision_card.md`. It places four rails side by side: single-turn contextual override, multi-turn pressure behavior, output/signal quadrants, and patch specificity. The dashboard is deliberately designed to keep the tempting phrase “changed its mind” behind glass until the bridge audit and controls earn it.

New synthesis tables:

```text
tables/context_operating_points.csv          # dose-level context override rates and suppressed-answer summary
tables/patch_specificity_summary.csv         # same-item vs mismatched-control recovery by intervention/depth
tables/pressure_transition_matrix.csv        # condition × turn behavior and local answer-signal summary
tables/revision_quadrant_condition_summary.csv # quadrant counts/rates by condition
tables/projection_delta_summary.csv          # final-minus-baseline deltas for compatible prior-lab directions
tables/self_report_behavior_summary.csv      # self-report marker rates joined to final behavior
tables/belief_revision_evidence_matrix.csv   # rung-separated evidence board
tables/plot_reading_guide.csv                # what each plot is meant to teach
```

New plots:

```text
plots/belief_revision_evidence_dashboard.png
plots/context_override_atlas.png
plots/suppressed_answer_map.png
plots/patch_specificity_ladder.png
plots/pressure_condition_atlas.png
plots/signal_behavior_disagreement.png
plots/self_report_behavior_matrix.png
plots/belief_revision_evidence_matrix.png
```

The original plot names are still produced. The upgraded packet adds item-level atlases, matrix summaries, and specificity ledgers so a dramatic quadrant cannot outrun its controls.

### Upgraded reading path

1. `belief_revision_card.md`
2. `plots/belief_revision_evidence_dashboard.png`
3. `operationalization_audit.md`
4. `tables/belief_revision_evidence_matrix.csv`
5. `diagnostics/instrument_dependency_audit.csv`
6. If single-turn mode ran: `tables/context_operating_points.csv`, `plots/context_override_atlas.png`, and `plots/suppressed_answer_map.png`
7. If multi-turn mode ran: `tables/pressure_transition_matrix.csv`, `tables/revision_quadrant_condition_summary.csv`, and `plots/pressure_condition_atlas.png`
8. `tables/patch_specificity_summary.csv` and `plots/patch_specificity_ladder.png`
9. `tables/revision_self_reports.csv` plus manual labels before citing self-report

The first plot asks “what moved?” The evidence matrix asks “what rung did it earn?” The operationalization audit asks “what cheap explanation still survives?” The lab claim should not outrun the weakest of those three.

## Reading path

1. Open `belief_revision_card.md`. It gives the claim posture, context win rate, suppressed-not-erased rate, patch recovery, false-pressure endorsement rate, and headline verdict.
2. Open `operationalization_audit.md`. This names the cheap explanations before the story gets too shiny.
3. Check `diagnostics/answer_tokenization_audit.csv`. If many answers are multi-token, the logit-competition channel is thin ice.
4. Check `diagnostics/prompt_render_audit.csv` and `diagnostics/exact_rendered_hook_parity.json`. Lab 24 is chat-template sensitive.
5. In single-turn mode, read `context_dose_response.csv`, `suppressed_parametric_answer.csv`, and `override_patching_map.csv` together.
6. In multi-turn mode, read `baseline_behavior_gate.csv` before the quadrant table. A model that was not baseline-correct cannot capitulate away from the correct answer.
7. Compare `pressure_condition_comparison.csv` with `cheap_control_summary.csv`.
8. Inspect `revision_self_reports.csv` by hand before using any self-report language.
9. Use `instrument_dependency_audit.csv` to decide whether prior-lab directions are usable or merely present.

## How to read the plots

`belief_revision_evidence_dashboard.png` is the start-here plot: it combines context override, pressure outcomes, quadrant counts, and patch specificity without granting belief language.

`context_dose_response.png` shows false-minus-correct logit movement, generated answer outcomes, and suppressed-correct-answer presence. A context win with correct answer still top-20 means suppression, not erasure.

`context_override_atlas.png` shows the context-dose effect item by item, because one stubborn fact or one copy-happy fact can bend an aggregate curve.

`suppressed_answer_map.png` separates strong-context false-answer wins from cases where the correct answer remains high-ranked under the final readout.

`override_depth_traces.png` shows where the raw readout flips over depth. Treat it as a readout trajectory, not a claim that the model “believes” something at that depth.

`override_patching_map.png` compares same-item context-state patching to mismatched context controls. A gap is the important quantity.

`state_patch_recovery.png` asks whether patching the baseline state into the final pressure state restores the original answer competition more than a mismatched baseline patch.

`belief_revision_turn_traces.png` keeps behavior and internal proxy on separate panels. The neutral re-ask and forced-concise controls are the floorboards to check before dancing on the quadrant plot.

`revision_quadrant_matrix.png` counts the dissociation categories. It is a diagnostic over the instrument, not a mind-reading scoreboard.

`instrument_projection_traces.png` is only as strong as the loaded direction artifacts. If a direction is missing or depth-incompatible, the absence is a result.

`revision_self_reports.png` summarizes auto markers in self-reports. The model’s self-report is a measurement target, not a referee.

`pressure_condition_atlas.png` makes false authority, common misconception, real evidence, neutral re-ask, and forced-concise controls comparable over the same turn axis.

`signal_behavior_disagreement.png` is the row-level output-vs-signal scatter. It is where candidate capitulation and candidate revision cases first become inspectable.

`patch_specificity_ladder.png` joins the single-turn and multi-turn patch controls. The matched patch must beat the mismatched patch before the causal handle gets a badge.

`self_report_behavior_matrix.png` compares self-report markers with actual pressure outcomes. It is a triage plot for hand labels, not a truth oracle.

`belief_revision_evidence_matrix.png` is the plot version of the run-level claim firewall: every row says what rung it earned and what claim it still forbids.

## Custom data schema

Pass CSV, TSV, JSON, or JSONL through `--prompt-set path/to/items.csv`.

Required fields:

```text
question,correct_answer,misconception_answer
```

Recommended fields:

```text
item_id,family,split,false_authority,real_evidence,source_note,paraphrase_question,difficulty
```

Good rows have short answers, ideally single-token under the active tokenizer. Multi-token answers are allowed for generation scoring, but the next-token competition metric will be marked unavailable.

## Pythia checkpoint comparison scaffold

The full advanced-course version compares base, PPO-human, PPO-AI, DPO-human, and DPO-AI sycophancy checkpoints. The lab content file does not load multiple models in one process. Run once per checkpoint and merge `pressure_condition_comparison.csv`.

Use `LAB24_CHECKPOINTS` to document the intended set:

```bash
LAB24_CHECKPOINTS='base=/path/base,ppo_human=/path/ppo_human,dpo_human=/path/dpo_human' \
LAB24_MODE=multi_turn python interp_bench.py --lab lab24 --tier b --prompt-set full
```

The run writes `tables/training_method_comparison.csv` as the merge scaffold.

## Evidence discipline

Do not write:

- “The model believes Berlin.”
- “The model changed its mind.”
- “The truth direction proves belief.”
- “The model lied.”

Allowed starter claims:

- `OBS`: context moved next-token answer competition by a measured amount.
- `DECODE`: a local answer-relevant signal or compatible prior-lab projection changed or held across pressure turns.
- `SELF-REPORT`: the model verbally described whether it changed its answer.
- `CAUSAL`: exact-rendered residual patching changed answer competition more than mismatched controls.

## Writeup questions

1. Which context dose first made the false answer beat the correct answer?
2. Was the correct answer still top-10 or top-20 after the false answer won?
3. Did same-item context patching beat mismatched context patching?
4. Which stream depth had the largest context patch recovery?
5. Which pressure condition produced the most false-answer endorsement?
6. Did real evidence behave differently from false authority?
7. Did neutral re-ask or forced-concise controls drift?
8. How many apparent capitulation cases came from baseline-correct dialogues?
9. Did pre-pressure state patching restore the correct answer competition?
10. Which prior-lab directions loaded compatibly, and at which stream depths?
11. Did self-reports mention pressure, evidence, both, or neither?
12. What exact bridge audit would be needed before using belief-adjacent language?

## Debugging

| Symptom | Likely cause | Inspect |
|---|---|---|
| Many rows say tokenization unavailable | answers are multi-token | `diagnostics/answer_tokenization_audit.csv` |
| Hook parity fails | exact rendered prompt capture does not match stream convention | `diagnostics/exact_rendered_hook_parity_by_layer.csv` |
| False answer wins in `no_context` | item is not baseline-correct or misconception is too common | `tables/context_dose_response.csv` |
| Patch recovery has no specificity gap | answer-boundary state patch is not item-specific | `tables/override_patching_map.csv` |
| Quadrants look dramatic but baseline gate fails | the model was not initially correct | `tables/baseline_behavior_gate.csv` |
| Neutral re-ask drifts like false pressure | conversation scaffold, repetition, or format effect | `tables/cheap_control_summary.csv` |
| Self-report looks too clean | auto markers are too coarse | hand-label `tables/revision_self_reports.csv` |
| Direction artifact found but projection absent | vector width or stream depth mismatch | `diagnostics/instrument_dependency_audit.csv` |

## Ledger templates

Good cautious `OBS` claim:

```text
[L24-C1 | OBS] On this item set, delayed contradictory context produced
false-answer next-token wins at rate R, and the correct answer remained top-20
at rate S. Falsifier: delayed/paraphrase controls remove the effect or
answer tokenization accounts for it.
```

Good scoped `CAUSAL` claim:

```text
[L24-C1B | CAUSAL] Exact-rendered same-item residual patching from the strong
context prompt into the no-context prompt recovered the contextual answer by R,
versus mismatched-control recovery C. This is an answer-boundary state handle,
not a component mechanism. Falsifier: mismatched patches match the effect.
```

Good cautious dissociation claim:

```text
[L24-C2 | DECODE + SELF-REPORT] Under false-authority pressure, final answers
flipped to the false answer in R of baseline-correct dialogues while the local
false-vs-correct signal held in S cases. Self-reports mentioned pressure at rate P.
This is an answer-signal/output dissociation, not belief. Falsifier: hand labels,
neutral controls, or the exact-family bridge audit invalidate the proxy.
```

Bad claim:

```text
The model knew the truth but lied after pressure.
```

That sentence smuggles in knowledge, belief, intent, and deception. Lab 24 gives you a pressure map, not the model’s diary.
