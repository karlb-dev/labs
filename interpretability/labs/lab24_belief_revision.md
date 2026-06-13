# Lab 24 - Knowledge Conflict and Belief Revision

## Question

When a model is pushed away from a correct answer, does it change an answer-relevant internal signal, or does it only capitulate in the output?

This lab is the bridge between the sycophancy labs and the final self-report capstone. It has two modes:

```bash
python interp_bench.py --lab lab24 --tier b --mode single_turn
python interp_bench.py --lab lab24 --tier b --mode multi_turn
python interp_bench.py --lab lab24 --tier b --mode both
```

Tier A is a smoke path. Tier B is the intended Colab path.

## The hard rule

Do not call the internal signal "belief" by default.

The starter uses a local answer-competition proxy: logit pressure answer minus logit correct answer at the assistant boundary, plus top-1 margin and hedge markers. That can support an answer-relevant internal-state claim. It becomes belief-adjacent only if the Lab 4 truth direction and Lab 7 bridge audit pass on this exact statement family.

The run writes `diagnostics/instrument_dependency_audit.csv` so this dependency is visible.

## Mode: `single_turn`

Single-turn mode asks whether context can override a parametric answer.

For each item, the lab prompts the model across context-strength doses:

1. no contradictory context;
2. a weak fictional statement;
3. a document statement;
4. a repeated document statement;
5. a delayed document statement.

At each dose it records the next-token competition between the correct answer and the pressure answer. It also writes depth traces through the raw logit lens and a coarse final-position patching map from the strongest context prompt into the no-context prompt.

This is not yet belief revision. A context override may be copying. That is why the lab reports it as contextual-vs-parametric answer competition.

## Mode: `multi_turn`

Multi-turn mode runs pressure dialogues from `data/belief_revision_dialogues.csv`.

For each item, the model first answers a neutral question. Then the user pushes with one of:

- neutral re-ask;
- pushback without evidence;
- false authority;
- real evidence;
- common misconception;
- forced concise answer.

The generated answer is fed into the next turn. At each assistant boundary the lab records the local answer-competition signal and then scores the generated answer.

The quadrant table is:

| Internal signal | Answer | Label |
|---|---|---|
| flips | flips | possible persuasion |
| holds | flips | possible capitulation |
| flips | holds | possible committed disagreement |
| holds | holds | robust or unchanged |

Every label is provisional until the probe bridge and controls pass.

## Main artifacts

| Path | What it contains |
|---|---|
| `belief_revision_card.md` | read-first verdict card |
| `diagnostics/instrument_dependency_audit.csv` | Lab 4/7/14/16 dependency status |
| `tables/belief_revision_dialogues.csv` | selected item inventory |
| `tables/context_dose_response.csv` | single-turn context-strength competition |
| `tables/override_depth_traces.csv` | raw logit-lens competition by depth |
| `tables/override_patching_map.csv` | coarse final-position patching recovery |
| `tables/belief_revision_turn_traces.csv` | turn-indexed behavior and internal proxy |
| `tables/pressure_condition_comparison.csv` | pressure-condition summary |
| `tables/revision_quadrants.csv` | answer/internal flip quadrant per dialogue |
| `tables/patch_or_steer_recovery.csv` | optional intervention scaffold |
| `tables/training_method_comparison.csv` | Pythia sycophancy checkpoint comparison scaffold |
| `plots/context_dose_response.png` | context-strength dose response |
| `plots/override_patching_map.png` | coarse patching recovery by layer |
| `plots/belief_revision_turn_traces.png` | pressure-dialogue traces |
| `plots/revision_quadrant_matrix.png` | quadrant counts |
| `operationalization_audit.md` | cheap explanations and allowed claims |

## Evidence discipline

Do not write:

- "The model believes the pressure answer."
- "The model changed its mind."
- "The truth direction shows belief" unless the bridge audit passed on this statement family.
- "The model lied" from answer-flip/internal-hold alone.

Allowed by the starter:

- `OBS`: context moved next-token answer competition by a measured amount.
- `DECODE`: a local answer-relevant signal changed or held across pressure turns.
- `CAUSAL`: only for rows where the patch/steer recovery table contains an actual intervention result.

## Custom data

Pass a CSV through `--prompt-set path/to/items.csv` with:

```text
item_id,family,split,question,correct_answer,misconception_answer,false_authority,real_evidence,source_note
```

Keep answers short, ideally single-token under the active tokenizer, or the logit competition rows will mark tokenization unavailable.

## Writeup questions

1. Which context dose first made the pressure answer beat the correct answer?
2. Was the correct answer still decodable after the context override?
3. Did coarse final-position patching recover the pressure answer in the no-context prompt?
4. Which pressure condition produced the most answer flips?
5. Which condition produced answer flips without internal-signal flips?
6. Did real evidence behave differently from false authority?
7. Did the neutral re-ask drift? If so, what does that do to the story?
8. Which instrument dependency blocks belief-adjacent language in this run?
9. What exact bridge audit would be needed before using belief language?
10. What patch or steering intervention would turn the quadrant result into a causal claim?
