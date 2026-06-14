# Lab 27 - Path-Specific Patching and Causal Mediation

```text
Time estimate: 5-15 minutes Tier A smoke; 30-90+ minutes Tier B depending on model, grid, and GPU
Compute tier: base model, hook-heavy, forward-pass-only, no chat template
Dependencies: Labs 3, 5, 6, 12, and 26
Minimum passing artifacts: method_card.md, operationalization_audit.md, tables/path_evidence_matrix.csv, tables/path_specificity_controls.csv, diagnostics/self_check_status.json
Main plot: plots/path_mediation_dashboard.png
Main table: tables/path_evidence_matrix.csv
Evidence rung: CAUSAL, scoped path-proxy handle
Forbidden claim: A causes B in every context, or this identifies one exact internal edge.
One-sentence allowed claim: On this model and prompt family, a residual source-to-receiver mediation proxy recovered clean behavior above wrong-site, random-source, and reverse-site controls.
Human-label requirement: none
```

## Lab thesis

A node can matter without a route being identified. Lab 27 teaches that distinction by comparing ordinary residual patching at individual sites with a stricter residual source-to-receiver mediation proxy.

The tempting overclaim is this sentence:

```text
A source site and a receiver site both matter, therefore the path from source to receiver carries the behavior.
```

That sentence is tempting, compact, and often wrong. This lab makes it pay rent.

## What question this lab asks

Labs 5 and 6 asked where an activation patch changes behavior. Lab 27 asks a narrower question:

```text
When a clean source-site vector is inserted into a corrupt run, does the downstream receiver state it produces carry recoverable behavior, above controls?
```

That is not the same as asking whether the source node is important. It is not the same as asking whether the receiver node is important. It is a route-shaped question, but only as a residual-stream proxy.

The three task families in the default frozen data are deliberately small and readable:

| domain | example behavior | source site idea | receiver site idea |
|---|---|---|---|
| `induction` | copy a repeated token | previous occurrence of the copied token | final prediction position |
| `factual_recall` | capital recall clean/corrupt pair | subject token | final answer site |
| `relation_swap` | same subject, different relation word | relation cue token | final answer site |

A positive row says only that this proxy found a route-shaped handle for this data family. A negative row is also a good result: the node effects, controls, or behavior gate may have killed the route story.

## Why this matters in the course progression

The special-topics sequence is about making stronger causal claims without letting the language outrun the microscope. Lab 26 made high-level hypotheses explicit and tested residual resampling. Lab 27 moves from variable preservation to route pressure: not just "does this site matter?" but "does source information arriving at this receiver matter?"

This sits between manual circuit discovery and future stricter path patching. It prepares students to read stronger claims such as attention-head path patching, attribution graphs, and causal scrubbing without confusing a useful proxy for an exact mechanism.

## What the experiment measures

The score is always the normalized recovery of a clean-vs-corrupt logit difference:

```text
logit_diff = logit(target) - logit(distractor)
recovery = (patched_diff - corrupt_diff) / (clean_diff - corrupt_diff)
```

Before any path intervention, the lab requires a behavior gate:

```text
clean_diff > 0.20
corrupt_diff < -0.20
clean_diff - corrupt_diff > 0.40
```

If the model does not prefer the clean target on the clean prompt and the corrupt distractor on the corrupt prompt, the path experiment does not have a meaningful denominator. That is a behavioral limitation, not a failed hook.

## The residual mediation proxy

For a candidate source depth `s` and receiver depth `r`, with `s < r`, the lab computes:

1. `source_node_recovery`: patch the clean source vector into the corrupt run.
2. `receiver_node_recovery`: patch the clean receiver vector into the corrupt run.
3. `mediated_path_recovery`: patch the clean source vector into the corrupt run, capture the resulting receiver vector at depth `r`, then insert that receiver vector into a fresh corrupt run at the receiver site.
4. `joint_clean_two_site_recovery`: patch both the clean source vector and the clean receiver vector into the corrupt run.

The third line is the key teaching move. It asks whether the receiver state produced by the upstream source intervention is behaviorally useful. The fourth line is a composition check, not the headline path proxy.

The lab also writes accounting columns:

```text
specificity_gap = mediated_path_recovery - max(control_recovery)
joint_increment_over_best_node = joint_clean_two_site_recovery - max(source_node_recovery, receiver_node_recovery)
interaction_residual = joint_clean_two_site_recovery - source_node_recovery - receiver_node_recovery
```

Do not treat `interaction_residual` as a mechanism. It is arithmetic smoke from the patching battery. Useful smoke, but still smoke.

## What counts as evidence

A domain earns path-proxy language only when the best row clears all of these gates:

| Gate | Meaning |
|---|---|
| behavior gate | clean and corrupt prompts create a meaningful recovery denominator |
| temporal gate | source stream depth is earlier than receiver stream depth |
| mediated recovery gate | the source-patched receiver state recovers nontrivial behavior |
| specificity gate | the mediated receiver state beats wrong-site, random-source, and reverse-site controls |
| audit gate | counterexample rows do not collapse the story into a node-only or control-matched explanation |

A positive row is still a handle, not a circuit diagram. The lab can support this kind of sentence:

```text
In this prompt family, the residual source-to-receiver mediation proxy recovered the target margin above controls.
```

It cannot support this kind of sentence:

```text
The model routes this fact through this exact attention head edge in every context.
```

## Controls and falsifiers

| Control | What it tries to kill |
|---|---|
| `wrong_receiver_from_source_patch` | Maybe the source patch just makes any downstream site useful. |
| `random_source_to_receiver` | Maybe any clean vector injected upstream perturbs the receiver enough. |
| `reverse_site_two_site` | Maybe the direction and site labels do not matter. |
| node-dominance counterexample | Maybe a single receiver or source node patch explains the result. |
| behavior gate | Maybe the denominator is not behaviorally meaningful. |
| temporal gate | A later source cannot mediate into an earlier receiver. |

The controls are not decorative. They decide the claim posture.

## How to run

From `interpretability/`:

```bash
# Tier A smoke. This should be cheap and may be an honest negative.
python interp_bench.py --lab lab27 --tier a --no-plots

# Tier A with plots, useful when checking artifact plumbing.
python interp_bench.py --lab lab27 --tier a

# Tier B science path on the course base model.
python interp_bench.py --lab lab27 --tier b --prompt-set full

# Debug a tiny slice from a custom task file.
python interp_bench.py --lab lab27 --tier a --prompt-set data/path_mediation_tasks.csv --max-examples 3 --no-plots
```

Tier A proves the microscope, token alignment, behavior gates, and artifact writing. It is not required to support a path claim. Tier B is the intended science run.

## Artifact tree

```text
runs/lab27_path_mediation-*/
  run_summary.md                         # read first: verdict, failure mode, smallest surviving claim
  method_card.md                         # claim boundary and domain verdict table
  operationalization_audit.md             # cheap explanations and whether they survived
  ledger_suggestions.md                   # drafted claims only; edit before appending
  metrics.json                            # machine-readable counts, gates, thresholds
  results.csv                             # alias of tables/path_patch_report.csv

  diagnostics/
    data_manifest.json                    # data path, hash, manifest status, domain counts
    tokenization_gate.csv                 # runtime token alignment and position audit
    self_check_status.json                # hook parity, lens, patch no-op, tokenization, path-row summary
    safety_status.json                    # forward-pass-only safety/scope status

  tables/
    task_manifest.csv                     # selected tasks, positions, candidate path metadata
    baseline_behavior.csv                 # clean/corrupt margins and behavior-gate status
    node_effect_baseline.csv              # source, receiver, and wrong-position node patches
    depth_selection.csv                   # node-screen depths promoted into the expensive path grid
    path_patch_report.csv                 # mediated path-proxy rows and joint clean two-site rows
    path_specificity_controls.csv         # wrong-receiver, random-source, reverse-site controls
    mediation_accounting.csv              # node, mediated, joint, interaction, share accounting
    path_evidence_matrix.csv              # one row per domain with claim posture
    path_counterexamples.csv              # rows that defeat or narrow path language
    plot_reading_guide.csv                # what each plot protects

  plots/
    path_mediation_dashboard.png          # start here: mediated path vs controls, node pressure, failures
    node_vs_path_effects.png              # source/receiver node effects compared to mediated path proxy
    path_specificity_matrix.png           # source-depth by receiver-depth specificity gaps for selected cells
    mediation_accounting_waterfall.png    # best-cell accounting terms
    heldout_path_transfer.png             # domain breadth check, not a true held-out proof unless data supplies splits
    path_graph.png                        # compact schematic of best residual route proxies

  state/
    path_candidates.json                  # task sites, depth grid, selected depths, thresholds
```

## How to read the run

Start with `run_summary.md`. It tells you whether the run is science-ready, what failed, and the smallest claim that survived.

Then read `method_card.md`. The verdict table is the traffic light for claim language. If a domain says `failed_controls`, `node_effect_only`, or `behavior_gate_failed`, the lab is warning you not to write path language for that domain.

Next inspect the evidence in this order:

1. `tables/baseline_behavior.csv`: no behavior gate, no path claim.
2. `tables/tokenization_gate.csv`: positions are runtime facts, not authoring-time wishes.
3. `tables/node_effect_baseline.csv`: check whether the receiver patch already explains the effect.
4. `tables/depth_selection.csv`: see which depths were promoted from node screening into the path grid.
5. `tables/path_patch_report.csv`: inspect `mediated_path_recovery`, `joint_clean_two_site_recovery`, and `specificity_gap`.
6. `tables/path_specificity_controls.csv`: controls are the prosecution.
7. `tables/path_counterexamples.csv`: counterexamples are not footnotes. They are the audit with teeth.
8. `operationalization_audit.md`: use this when writing the final paragraph.

Only after that should you admire the plots.

## Figure guide

### `path_mediation_dashboard.png`

The cockpit plot. It shows whether the best mediated path proxy beats the strongest control, whether node patches are already enough, which domains clear the specificity gate, and how often controls are close to the path proxy.

### `node_vs_path_effects.png`

This is the anti-overclaim plot. If the receiver node bar is large and the mediated path bar is small, the honest claim is a node-patching claim.

### `path_specificity_matrix.png`

A cell-level view for the domain with the strongest best row. Rows are receiver depths, columns are source depths, and values are specificity gaps. A hot cell is a candidate for a stricter future path-patching implementation, not proof of an exact edge.

### `mediation_accounting_waterfall.png`

The best-cell arithmetic ledger. It places source node, receiver node, mediated path, joint clean two-site patch, control floor, and specificity gap side by side.

### `heldout_path_transfer.png`

A domain-breadth check. The default data does not provide a true held-out split, so read this as "is one specimen carrying the story?" rather than as replication.

### `path_graph.png`

A compact route sketch for the best row in each domain. Use it to choose the next microscope, not to write a mechanism claim.

## Expected outcomes

A positive Tier B run should look like this:

1. Multiple tasks pass the behavior gate.
2. The mediated receiver proxy recovers a nontrivial amount of the clean behavior.
3. The specificity gap is positive and clears the gate.
4. Wrong-receiver, random-source, and reverse-site controls sit lower.
5. Node patches do not fully explain the result.

A negative run can be equally instructive:

| Pattern | Interpretation |
|---|---|
| clean/corrupt gate fails | The model does not behaviorally instantiate the task family enough for mediation. |
| receiver node dominates | You found a receiver-site patch effect, not a route. |
| random-source control matches | The path proxy may be measuring perturbation or damage, not content-specific mediation. |
| reverse-site control matches | The proposed direction/site labels are not earning their keep. |
| joint clean two-site patch works but mediated proxy fails | Clean nodes compose, but the source-patched receiver state did not carry the proposed route. |
| only one task/domain works | Treat as a specimen and write a narrow exploratory claim. |

## What this lab can claim

With passing controls:

```text
CAUSAL: On model M and prompt family P, inserting a receiver state produced by a clean source patch recovered X of the clean-vs-corrupt margin, beating the strongest wrong-site/random/reverse control by Y.
```

Without passing controls:

```text
AUDIT: This run did not validate a path-proxy claim. The strongest competing explanation was <node dominance / control match / behavior gate failure>.
```

## What this lab cannot claim

Do not write:

```text
The model has a path from this exact attention head to that exact attention head.
The model always uses this route for factual recall.
The source causes the receiver in every context.
The interaction residual is the mechanism.
```

The lab has no head-level receiver isolation, no MLP edge isolation, no attention Q/K/V path patching, and no held-out path-transfer proof unless you add a dataset with that structure.

## Common failure modes

| Symptom | Likely cause | What to inspect |
|---|---|---|
| every row dropped in tokenization | answer is not one token, positions are wrong, or tokenizer added tokens not accounted for | `diagnostics/tokenization_gate.csv` |
| no behavior-pass tasks | Tier A model does not know the task or corrupt prompt is not strong enough | `tables/baseline_behavior.csv` |
| path table is empty | no source depth is earlier than selected receiver depth, or no task passed gates | `tables/depth_selection.csv` |
| controls match the path | proxy is not specific enough | `tables/path_specificity_controls.csv` |
| plot looks positive but audit says failed | the plot is showing recovery, while the audit subtracts controls | `operationalization_audit.md` |
| runtime is too long | full node depth screen plus path captures are expensive | run Tier A, use `--max-examples`, or use `--prompt-set medium` |

## Suggested extensions

1. Add a held-out split to `path_mediation_tasks.csv` and report train-selected depths versus held-out path cells.
2. Add head-output source nodes from Lab 3 or Lab 6 and compare them to residual source sites.
3. Add MLP receiver sites and separate residual-band-to-MLP from head-to-MLP proxies.
4. Add matched same-position donors from other items in the same domain, not just wrong-position controls.
5. Turn the best residual proxy into a preregistered exact path-patching hypothesis for a future Lab 27b.

## Writeup questions

1. Which domain, if any, earned path-proxy language? Quote `mediated_path_recovery`, `control_floor`, and `specificity_gap`.
2. Did the receiver node patch alone explain the best row? Use `node_vs_path_effects.png` and `mediation_accounting.csv`.
3. Which control was most dangerous? What cheap explanation does it represent?
4. Did the clean two-site joint patch add anything above the best node patch? What does that allow you to say, and what does it not allow?
5. If you were building a stricter path-patching version, which exact source and receiver would you test next, and why?

## Ledger templates

Positive, only after controls pass:

```text
[L27-C1] CAUSAL | On <model>, the residual source-to-receiver path proxy for <domain> at source depth <s> and receiver depth <r> recovered <X> of the clean-vs-corrupt margin, beating the strongest wrong-site/random/reverse control by <Y>. This is a scoped residual path-proxy handle, not a unique-edge claim.
Artifact: runs/<run>/tables/path_evidence_matrix.csv | Falsifier: controls match the mediated recovery on held-out prompt families, or an exact path-patching implementation fails for the proposed route.
```

Negative, still useful:

```text
[L27-N1] CAUSAL + AUDIT | This run did not validate a path-proxy claim for <domain>: <control/node/behavior gate> explained the result. The supported claim is limited to <node effect / failed behavior gate / unresolved proxy>.
Artifact: runs/<run>/operationalization_audit.md | Falsifier: a rerun with behavior-gated held-out prompts clears the mediated-recovery and specificity gates.
```

## Safety and scope

This is a forward-pass-only lab over benign completion prompts. It does not generate harmful text, does not train or edit a model, and does not ablate refusal or safety behavior. The safety status is still written as `diagnostics/safety_status.json` so the special-topics artifact contract stays uniform.
