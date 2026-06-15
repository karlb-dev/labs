# Special Topics in Mechanistic Interpretability: Labs 26-35

**Companion to:** `COURSE.md` and `ADVANCED_COURSE.md`  
**Position:** third sequence after Labs 1-25  
**Audience:** students who completed the intro and advanced labs and can already run the shared bench, interpret evidence rungs, write claim-ledger entries, and repair overclaims.  
**Goal:** move from “advanced course participant” to “research-ready mechanistic interpretability practitioner.”

---

## 0. Why continue after Lab 25?

Labs 1-25 teach the core craft:

```text
instrument -> measurement -> control -> intervention -> artifact -> caveat -> claim ledger
```

That is enough to make students competent. The next ten labs should make them dangerous in the good way: able to design new interpretability experiments, evaluate automated tools, find when methods fail, and produce reproducible audit packages other researchers can inspect.

The first 25 labs emphasize a recurring warning:

```text
a readable signal is not necessarily a used signal;
a steering handle is not necessarily a mechanism;
a self-report is not necessarily an internal trace;
a pretty graph is not necessarily a faithful explanation.
```

Labs 26-35 pick up from there. They ask harder questions:

- Can a mechanistic explanation be formalized as a causal abstraction?
- Can we test paths, not only nodes?
- Can localization improve editing or unlearning robustly?
- Can we watch circuits and features emerge during training?
- Can features be followed across layers, checkpoints, models, and modalities?
- Can automated interpretability be evaluated rather than admired?
- Can reward models, tool-use agents, and multimodal systems be audited with the same discipline?
- Can students ship a reproducible interpretability paper-quality package?

The theme is **method pressure**. The next ten labs are not ten new “cool topics.” They are ten ways to ask: does the method survive when the problem becomes less toy-shaped?

---

## 1. Design principles for Labs 26-35

### 1.1 The third sequence has a stricter standard

By Lab 26, students have seen enough caveats. The special-topics labs should require:

- pre-registered hypotheses before the main run;
- artifact schemas checked by tests;
- explicit failure-mode taxonomy;
- at least one adversarial control per lab;
- one replication or robustness slice per lab;
- a final claim card with allowed and forbidden language.

### 1.2 Every lab keeps the course contract

Each lab should produce:

```text
runs/labXX_*/
  run_summary.md
  method_card.md
  operationalization_audit.md
  evidence_matrix.csv
  plot_reading_guide.csv
  ledger_suggestions.md
  diagnostics/
  tables/
  plots/
  state/                 # when directions/features/models are saved
```

Every lab should support:

```bash
python interp_bench.py --lab labXX --tier a --no-plots
python interp_bench.py --lab labXX --tier b --prompt-set full
```

When a lab uses chat models, it should be added to `CHAT_TEMPLATE_LABS`. When it uses special flags, those flags should be in the shared parser, not only environment variables.

### 1.3 Evidence levels used in the third sequence

| Evidence tag | Use in Labs 26-35 |
|---|---|
| `OBS` | descriptive readouts, trajectories, correlations, recurrence, geometry |
| `ATTR` | direct attribution, replacement-model edges, path scores, model-diff feature contributions |
| `DECODE` | probes, directions, features, monitors, classifiers |
| `CAUSAL` | patching, ablation, resampling, activation addition, editing, unlearning |
| `AUDIT` | blind or semi-blind recovery, reproducibility, benchmark scorecards |
| `CONSTRUCTION` | controlled datasets, model organisms, synthetic mechanisms |
| `SELF-REPORT` | generated explanations, confidence reports, source attributions, agent rationales |
| `FORMAL` | new tag for causal-abstraction or causal-scrubbing hypotheses with explicit variable mapping and intervention semantics |

The `FORMAL` tag does not mean “mathematically proven.” It means the lab specified a high-level causal model, mapped low-level activations to high-level variables, and tested that abstraction under interventions.

---

## 2. Course map

| Lab | Title | Main question | Evidence target |
|---:|---|---|---|
| 26 | Causal Abstraction and Causal Scrubbing | Can a proposed high-level explanation survive behavior-preserving resampling? | `FORMAL + CAUSAL` |
| 27 | Path-Specific Patching and Causal Mediation | Which paths, not just nodes, carry a behavior? | `CAUSAL` |
| 28 | Mechanistic Editing and Unlearning | Does localization make edits or unlearning more robust and specific? | `CAUSAL + AUDIT` |
| 29 | Training Dynamics and Circuit Birth | When do features and circuits emerge during training or fine-tuning? | `OBS + ATTR + DECODE` |
| 30 | Cross-Layer and Cross-Model Feature Geometry | Can features be tracked across layers, checkpoints, and models? | `DECODE + ATTR` |
| 31 | Automated Interpretability at Scale | Can automated feature labels be evaluated rather than trusted? | `AUDIT + DECODE` |
| 32 | Reward Models and Preference Circuits | What internal features drive reward/preference judgments and policy shifts? | `ATTR + DECODE + CAUSAL` |
| 33 | Multimodal Mechanistic Interpretability | Do text and image features meet in shared mechanisms? | `OBS + DECODE + CAUSAL` |
| 34 | Tool Use, Agents, and State Tracking | What internal signals precede tool calls, memory use, and action choice? | `OBS + DECODE + CAUSAL + SELF-REPORT` |
| 35 | Reproducible Interpretability Paper Capstone | Can a student ship a preregistered, reproducible, adversarially reviewed study? | `AUDIT + FORMAL` |

---

# Lab 26: Causal Abstraction and Causal Scrubbing

## Core question

Can a proposed high-level explanation of a model behavior be tested as a causal abstraction, rather than a collection of interesting components?

This lab takes the course’s “claim discipline” and makes it formal. A circuit claim should specify high-level variables, low-level sites, allowed resampling interventions, and expected invariances. Then it should be tested by causal scrubbing or behavior-preserving resampling.

## Why this lab comes first

Labs 1-25 teach many instruments. Lab 26 asks students to state exactly what their instrument is supposed to mean. It is the difference between:

```text
Head L5H3 matters.
```

and:

```text
High-level variable COPY_SOURCE is represented by a token-position equivalence class, implemented by these heads, and replacing low-level activations with another example that preserves COPY_SOURCE should preserve the answer.
```

That second sentence has handles. You can test it.

## Prerequisites

- Lab 5 activation patching.
- Lab 6 circuit faithfulness/completeness/minimality.
- Lab 9 attribution graph replacement-model discipline.
- Lab 12 relation-swap method validation.

## Implementation plan

### Data

Use two small domains:

1. **Induction copying** from Lab 6.
2. **Relation-swap facts** from Lab 12.

Create:

```text
data/causal_abstraction_tasks.csv
```

Required columns:

```text
item_id,domain,split,high_level_task,template_family,
prompt,target,distractor,
source_token,source_position,target_position,
relation_family,subject,answer,high_level_variables_json
```

The `high_level_variables_json` field should encode variables such as:

```json
{
  "COPY_SOURCE": "token_after_previous_occurrence",
  "QUERY_TOKEN": "blue",
  "RELATION": "capital_of",
  "SUBJECT": "France",
  "ANSWER_CLASS": "city"
}
```

### Experiment A: formal hypothesis file

The student writes or selects a hypothesis spec:

```text
specs/lab26_induction_hypothesis.json
specs/lab26_relation_hypothesis.json
```

Schema:

```json
{
  "hypothesis_id": "induction_copy_v1",
  "behavior_metric": "logit(target) - logit(distractor)",
  "high_level_variables": ["COPY_SOURCE", "QUERY_TOKEN"],
  "low_level_sites": [
    {"kind": "head", "layer": 5, "head": 1, "positions": "all"},
    {"kind": "residual", "stream_depth": 8, "positions": "target_position"}
  ],
  "resampling_rules": [
    {"preserve": ["COPY_SOURCE"], "vary": ["surface_tokens"]},
    {"preserve": ["RELATION"], "vary": ["SUBJECT"]}
  ],
  "predicted_preservation_min": 0.80,
  "predicted_damage_when_broken_min": 0.30
}
```

### Experiment B: behavior-preserving resampling

For each example, choose donor examples that preserve selected high-level variables and donors that break them.

Run interventions:

- residual-stream resampling at nominated sites;
- head-output resampling for nominated heads;
- path/node resampling if Lab 27 helpers are available;
- random matched donor control;
- same-token-position but wrong-variable control.

Metric:

```text
scrub_score = patched_metric / clean_metric
```

Report preservation when high-level variables match and damage when they do not.

### Experiment C: hypothesis refinement

Students must produce a second hypothesis version if the first fails. The lab should write:

```text
tables/hypothesis_refinement_log.csv
```

Columns:

```text
hypothesis_id,version,failed_rule,evidence_path,revision,student_notes
```

## Controls

- random donor with matched length;
- donor preserving token surface but breaking high-level variable;
- donor preserving high-level variable but changing surface tokens;
- wrong-site resampling;
- no-op same-example resampling.

## Metrics

- preservation ratio for variable-preserving donors;
- damage ratio for variable-breaking donors;
- causal abstraction gap;
- per-variable pass/fail;
- hypothesis precision: fraction of predicted-invariant interventions that preserve behavior;
- hypothesis recall: fraction of behavior-preserving interventions covered by the hypothesis.

## Plots

```text
plots/causal_abstraction_dashboard.png
plots/resampling_preservation_matrix.png
plots/hypothesis_pass_fail_atlas.png
plots/variable_specificity_ladder.png
plots/refinement_trajectory.png
plots/counterexample_gallery.png
```

## Tables and artifacts

```text
method_card.md
causal_abstraction_spec.md
operationalization_audit.md
tables/hypothesis_spec_audit.csv
tables/resampling_interventions.csv
tables/variable_preservation_summary.csv
tables/counterexamples.csv
tables/hypothesis_refinement_log.csv
plots/plot_reading_guide.csv
```

## Ledger claims

Allowed:

```text
FORMAL + CAUSAL: For induction prompts in dataset D, hypothesis H preserved behavior under resampling interventions that conserved COPY_SOURCE, and failed under controls that broke COPY_SOURCE.
```

Forbidden:

```text
The model implements exactly this algorithm in all contexts.
```

## References

- Geiger et al., **Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability**.
- Chan et al., **Causal Scrubbing: a method for rigorously testing interpretability hypotheses**.
- Conmy et al., **Towards Automated Circuit Discovery for Mechanistic Interpretability**.
- Anthropic Transformer Circuits, **A Mathematical Framework for Transformer Circuits**.

---

# Lab 27: Path-Specific Patching and Causal Mediation

## Core question

Which paths through the network carry a behavior, and how do path-specific effects differ from node-level importance?

Labs 5 and 6 patch or ablate sites. Lab 27 asks about **routes**. A head can matter because it writes directly to the readout, because it writes to another head, or because it changes an MLP input. Node effects bundle these routes together. Path-specific patching separates them.

## Prerequisites

- Lab 3 attention routing.
- Lab 5 patching.
- Lab 6 circuit discovery.
- Lab 9 attribution graphs.
- Lab 26 causal-abstraction specs are recommended.

## Implementation plan

### Data

Use three behaviors:

1. induction completion;
2. factual recall;
3. relation swap.

Create a unified path task file:

```text
data/path_mediation_tasks.csv
```

Required columns:

```text
item_id,domain,prompt,clean_prompt,corrupt_prompt,target,distractor,
positions_json,candidate_nodes_json,candidate_paths_json
```

### Experiment A: node effect baseline

Compute ordinary node-level effects:

- residual patching;
- head ablation;
- component patching;
- DLA or head attribution screens.

### Experiment B: path patching

Implement path-specific interventions:

- freeze all components to corrupt;
- patch selected upstream node from clean;
- allow only selected downstream receiver to read the clean upstream write;
- keep other paths corrupt.

A practical implementation can begin with attention-head paths:

```text
source head output at position p -> receiver head input at position q -> final readout
```

Then extend to:

```text
source residual band -> MLP layer -> final readout
source head -> MLP -> final readout
source head -> receiver head -> MLP -> final readout
```

### Experiment C: mediation accounting

For top paths, compare:

```text
node_effect(source)
node_effect(receiver)
path_effect(source -> receiver)
interaction_residual = joint_effect - source_effect - receiver_effect
```

Report where path effects explain node effects and where they do not.

## Controls

- wrong receiver control;
- wrong position control;
- source with similar node effect but mismatched motif;
- random path control;
- reverse path control;
- no-op path patch.

## Metrics

- path recovery;
- path specificity gap;
- path/node ratio;
- interaction residual;
- signed path effect;
- held-out prompt transfer;
- causal mediation share.

## Plots

```text
plots/path_mediation_dashboard.png
plots/node_vs_path_effects.png
plots/path_specificity_matrix.png
plots/path_graph.png
plots/mediation_accounting_waterfall.png
plots/heldout_path_transfer.png
```

## Tables and artifacts

```text
tables/node_effect_baseline.csv
tables/path_patch_report.csv
tables/path_specificity_controls.csv
tables/mediation_accounting.csv
tables/path_evidence_matrix.csv
state/path_candidates.json
```

## Ledger claims

Allowed:

```text
CAUSAL: In this prompt family, path P from source node A to receiver node B carries X% of the clean-vs-corrupt recovery under path-patching intervention I.
```

Forbidden:

```text
A causes B in every context.
```

## References

- Wang et al. / Transformer Circuits work on induction heads and path patching.
- Conmy et al., **ACDC: Towards Automated Circuit Discovery**.
- Chan et al., **Causal Scrubbing**.
- Lab 9 attribution graph readings on replacement-model edges and real-model interventions.

---

# Lab 28: Mechanistic Editing and Unlearning

## Core question

Does mechanistic localization make model editing or unlearning more robust, specific, and measurable?

Lab 5 showed that a fact can be localized by patching, but localization does not guarantee editability. Lab 28 makes that tension the whole lab.

## Prerequisites

- Lab 5 patching and optional rank-one edit audit.
- Lab 8 feature validation.
- Lab 12 relation geometry.
- Lab 20 benign organisms.
- Lab 21 LoRA localization.

## Safety scope

Use only benign targets:

- factual associations;
- toy relation families;
- harmless Lab 20 organisms;
- synthetic secrets that are not real private data.

No harmful capabilities, no real personal data, no jailbreak/unrefusal edits.

## Implementation plan

### Data

Create:

```text
data/editing_unlearning_targets.csv
```

Required columns:

```text
target_id,family,edit_type,prompt,target_before,target_after,
retain_prompts_json,paraphrase_prompts_json,neighbor_prompts_json,
safety_notes
```

Edit types:

- `counterfactual_fact_edit`;
- `relation_family_edit`;
- `organism_marker_suppression`;
- `benign_topic_unlearning`.

### Experiment A: localization comparison

For each target, localize candidate sites using:

- residual patching;
- component DLA;
- head/MLP ablation;
- SAE/transcoder feature activation;
- Lab 21 adapter weight mass if organism-based.

### Experiment B: edit methods

Implement several lightweight edit methods:

1. activation steering at localized site;
2. rank-one residual/MLP update for small models;
3. low-rank adapter layer masking or adapter-layer edit;
4. feature clamp or feature suppression;
5. non-mechanistic baseline: random layer, random direction, or prompt-only correction.

The course can start with inference-time edits and add weight edits where feasible.

### Experiment C: robustness audit

Evaluate:

- direct target success;
- paraphrase transfer;
- neighbor preservation;
- adversarial paraphrase robustness;
- side-effect drift;
- re-learning or recovery after context reminder;
- edit locality compared with localization score.

## Controls

- random site edit;
- same-norm random direction;
- nearest-neighbor fact control;
- mismatched target control;
- retain-set preservation;
- paraphrase set withheld until after edit selection.

## Metrics

- edit efficacy;
- paraphrase generalization;
- specificity;
- locality;
- retention;
- robustness to adversarial paraphrases;
- mechanism/edit alignment correlation;
- unlearning score with retain loss.

## Plots

```text
plots/editing_unlearning_dashboard.png
plots/localization_vs_editability.png
plots/edit_method_frontier.png
plots/paraphrase_robustness_matrix.png
plots/neighbor_preservation_atlas.png
plots/mechanistic_locality_ladder.png
plots/unlearning_retain_forget_frontier.png
```

## Tables and artifacts

```text
tables/localization_candidates.csv
tables/editing_results.csv
tables/retain_forget_matrix.csv
tables/paraphrase_robustness.csv
tables/edit_evidence_matrix.csv
state/edit_vectors.pt
state/edit_metadata.json
```

## Ledger claims

Allowed:

```text
CAUSAL + AUDIT: Method M applied at localized site S changed target behavior T on prompts P and transferred to paraphrases while preserving retain set R better than random-site controls.
```

Forbidden:

```text
The fact was erased from the model.
```

## References

- Meng et al., **Locating and Editing Factual Associations in GPT**.
- Mitchell et al., **MEND**.
- Meng et al., **MEMIT**.
- Guo et al., **Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization**.
- Labs 5, 8, 20, and 21 from this course.

---

# Lab 29: Training Dynamics and Circuit Birth

## Core question

When does a representation or circuit appear during training, and can we distinguish feature birth, sharpening, migration, and reuse?

Most labs inspect a trained model. Lab 29 turns the microscope into a time-lapse camera.

## Prerequisites

- Lab 1 logit lens trajectories.
- Lab 3 induction heads.
- Lab 6 circuit discovery.
- Lab 19 model diffing.
- Lab 21 training-depth distinctions.

## Implementation plan

### Data and model sources

Use one or more checkpoint sequences:

1. Pythia checkpoints for general training dynamics.
2. A tiny transformer trained in-course on an induction/copy task.
3. Lab 20 organism fine-tuning checkpoints if available.
4. Optional small LoRA checkpoints from a PEFT training run.

Create:

```text
data/training_dynamics_tasks.csv
```

Required columns:

```text
item_id,task_family,prompt,target,distractor,split,expected_mechanism,notes
```

### Experiment A: checkpoint behavior trajectory

For each checkpoint:

- compute behavior metrics;
- logit-lens event depths;
- DLA summaries;
- attention motif scores;
- probe selectivity;
- patch recovery if feasible.

### Experiment B: feature and circuit birth

For induction tasks:

- track previous-token and induction motifs;
- track head attribution and ablation effects;
- track F/C/M for a small circuit at selected checkpoints.

For factual or relation tasks:

- track truth/relation probe direction formation;
- track patch localization migration;
- track representation norm and stability.

### Experiment C: phase classification

Classify each mechanism as:

- absent;
- behavioral before interpretable;
- decodable before behavioral;
- circuit present;
- migration across layers;
- sharpened but same site;
- redistributed.

## Controls

- random checkpoints or shuffled labels;
- same architecture random-initialized model;
- control task not trained;
- prompt families held out from direction fitting;
- size-matched but untrained LoRA adapter.

## Metrics

- behavior emergence step;
- first decodable checkpoint;
- first causal checkpoint;
- depth migration slope;
- motif birth checkpoint;
- attribution/correlation over time;
- circuit stability Jaccard;
- intervention transfer over checkpoints.

## Plots

```text
plots/training_dynamics_dashboard.png
plots/behavior_vs_decodability_timeline.png
plots/circuit_birth_atlas.png
plots/depth_migration_map.png
plots/checkpoint_feature_lineage.png
plots/intervention_transfer_over_time.png
plots/random_model_control_panel.png
```

## Tables and artifacts

```text
tables/checkpoint_behavior.csv
tables/checkpoint_probe_selectivity.csv
tables/checkpoint_circuit_summary.csv
tables/mechanism_birth_events.csv
tables/feature_lineage.csv
state/checkpoint_directions.pt
```

## Ledger claims

Allowed:

```text
OBS/DECODE/CAUSAL: In checkpoint sequence C, behavior B appears before/after direction D becomes decodable; causal intervention I becomes effective at checkpoint K under controls.
```

Forbidden:

```text
The model first learned concept X at exactly this step.
```

## References

- Biderman et al., **Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling**.
- Olsson et al., **In-context Learning and Induction Heads**.
- Nanda et al., work on grokking and mechanistic interpretability of modular arithmetic.
- Labs 3, 6, 19, and 21 from this course.

---

# Lab 30: Cross-Layer and Cross-Model Feature Geometry

## Core question

Are features static layer-local objects, or do they persist, split, merge, and move across layers, checkpoints, and model families?

Lab 8 treats SAE features at a site. Lab 19 treats cross-model features at a site. Lab 30 studies feature **lineage**.

## Prerequisites

- Lab 8 SAEs and transcoders.
- Lab 9 attribution graphs.
- Lab 19 crosscoders.
- Lab 29 checkpoint dynamics.

## Implementation plan

### Data

Use a mixed corpus:

- domain-labeled SAE corpus from Lab 8;
- persona/sycophancy/certainty prompts from Labs 14-17;
- model-diff inventory from Lab 19;
- new held-out lineage prompts.

Create:

```text
data/feature_lineage_corpus.csv
```

Required columns:

```text
row_id,family,domain,source_lab,text,group_id,split,labels_json
```

### Experiment A: cross-layer dictionary

Train or load cross-layer sparse dictionaries:

```text
z = encoder([x_layer_a, x_layer_b, x_layer_c])
x_hat_layer_i = decoder_i(z)
```

Compare with independent per-layer SAEs.

### Experiment B: feature lineage graph

For selected features, compute:

- decoder cosine across layers;
- activation correlation across corpus;
- top-context overlap;
- label stability;
- causal clamp transfer across layers;
- downstream effect similarity.

Build a graph where nodes are `(model, checkpoint, layer, feature)` and edges indicate possible lineage.

### Experiment C: feature split/merge cases

Identify cases where:

- one early feature splits into multiple later features;
- multiple early features merge into one later feature;
- a feature changes label;
- a feature is model-specific;
- a feature is dictionary artifact.

## Controls

- random feature matching;
- random initialized model;
- shuffled corpus labels;
- same-layer independent SAE comparison;
- activation-frequency matched controls;
- token-feature confusable pairs.

## Metrics

- lineage score;
- activation correlation;
- decoder cosine;
- top-context Jaccard;
- label survival;
- causal transfer;
- split/merge entropy;
- model-specificity score.

## Plots

```text
plots/feature_lineage_dashboard.png
plots/cross_layer_feature_graph.png
plots/lineage_similarity_matrix.png
plots/feature_split_merge_atlas.png
plots/label_stability_ladder.png
plots/cross_model_feature_overlap.png
plots/causal_transfer_by_layer.png
```

## Tables and artifacts

```text
tables/feature_lineage_edges.csv
tables/feature_lineage_nodes.csv
tables/split_merge_candidates.csv
tables/label_stability_summary.csv
tables/cross_model_feature_overlap.csv
state/cross_layer_dictionary.pt
state/lineage_graph.json
```

## Ledger claims

Allowed:

```text
DECODE + ATTR: Feature family F has a recurring activation/decoder/top-context lineage across layers L1-L3, above matched random-feature controls.
```

Forbidden:

```text
This is the same concept everywhere in the model.
```

## References

- Anthropic, **Sparse Crosscoders for Cross-Layer Features and Model Diffing**.
- Anthropic, **Insights on Crosscoder Model Diffing**.
- Anthropic, **Scaling Monosemanticity**.
- SAEBench, **A Comprehensive Benchmark for Sparse Autoencoders**.
- Sharkey et al., **Open Problems in Mechanistic Interpretability**.

---

# Lab 31: Automated Interpretability at Scale

## Core question

Can automated feature explanations be evaluated, calibrated, and falsified at scale?

Lab 8 teaches manual feature validation. Lab 31 asks how to scale that validation without turning auto-labels into a decorative fog machine.

## Prerequisites

- Lab 8 feature atlas.
- Lab 30 feature lineage.
- Lab 11 reliability audit.
- Familiarity with LLM-as-judge failure modes.

## Implementation plan

### Data

Use feature candidates from:

- Lab 8 SAE atlas;
- Lab 19 crosscoder features;
- Lab 30 lineage features;
- random SAE features;
- known synthetic features from small trained toy models.

Create:

```text
data/auto_interp_feature_tasks.jsonl
```

Each row:

```json
{
  "feature_id": "...",
  "model": "...",
  "layer": 8,
  "feature_index": 123,
  "top_contexts": [...],
  "heldout_contexts": [...],
  "negative_contexts": [...],
  "confusable_contexts": [...],
  "gold_label": "optional for synthetic or hand-labeled subsets"
}
```

### Experiment A: explanation generation

Generate labels/explanations using:

- simple majority-domain heuristic;
- local LLM explanation from top contexts;
- structured explanation prompt with counterexamples;
- caption+test generation;
- human-written labels for calibration subset.

### Experiment B: explanation scoring

For each explanation, automatically generate or use tests:

- held-out positive examples;
- hard negatives;
- confusable-pair negatives;
- paraphrased positives;
- token-overlap decoys.

Score whether the explanation predicts activation, not merely topic similarity.

### Experiment C: calibration and abstention

Train or compute a confidence score for labels. Evaluate:

- when the auto-label should abstain;
- when the label is too broad;
- when it is token-level;
- when it is polysemantic.

## Controls

- random features;
- shuffled top contexts;
- top contexts with key tokens removed;
- confusable domain pairs;
- synthetic features with known labels;
- adversarial contexts that match label words but not concept.

## Metrics

- label predictive AUC;
- precision at high confidence;
- abstention quality;
- broad/narrow/polysemantic classification accuracy;
- context-leak sensitivity;
- synthetic gold-label recovery;
- calibration error.

## Plots

```text
plots/auto_interp_dashboard.png
plots/explanation_quality_matrix.png
plots/confidence_calibration_curve.png
plots/abstention_frontier.png
plots/confusable_pair_failure_atlas.png
plots/random_feature_sanity_panel.png
```

## Tables and artifacts

```text
tables/generated_explanations.csv
tables/explanation_tests.csv
tables/explanation_scores.csv
tables/auto_interp_evidence_matrix.csv
tables/human_review_queue.csv
```

## Ledger claims

Allowed:

```text
AUDIT + DECODE: Explanation method E predicted held-out feature activations with AUC X and abstained on Y% of high-polysemantic features under test suite T.
```

Forbidden:

```text
The automated label is the feature’s meaning.
```

## References

- Bills et al., **Language Models Can Explain Neurons in Language Models**.
- Neuronpedia and open-source automated interpretability tools.
- Karvonen et al., **SAEBench: A Comprehensive Benchmark for Sparse Autoencoders**.
- Anthropic, **Scaling Monosemanticity**.
- Lab 8 feature-validation battery.

---

# Lab 32: Reward Models and Preference Circuits

## Core question

What internal features drive preference or reward judgments, and how do those features differ from policy behavior?

Labs 16-19 look at sycophancy, persona, and model diffing. Lab 32 opens the reward/preference box: what does a reward model or preference-trained policy actually score?

## Prerequisites

- Lab 7 steering.
- Lab 16 sycophancy and agreement pressure.
- Lab 17 persona/register.
- Lab 19 model diffing.
- Lab 21 training effects.

## Safety scope

Use benign preference data only:

- helpfulness;
- honesty on harmless factual tasks;
- refusal on benign boundary prompts;
- style preference;
- sycophancy controls.

No harmful prompt optimization, no refusal ablation, no jailbreak search.

## Implementation plan

### Data

Create paired preference dataset:

```text
data/preference_circuit_pairs.csv
```

Required columns:

```text
pair_id,domain,prompt,response_a,response_b,preferred,
preference_type,confound_type,split,notes
```

Preference types:

- helpfulness;
- correctness;
- politeness;
- refusal appropriateness;
- verbosity;
- agreement/sycophancy;
- uncertainty/hedging.

### Experiment A: reward/preference readout

Depending on available models:

1. load an open reward model and score paired responses;
2. or use a DPO/reference-policy log-prob ratio as preference proxy;
3. or compare base vs instruct policy log-prob preference.

### Experiment B: internal preference directions

Fit directions at prompt+response boundary:

```text
preferred response residual - dispreferred response residual
```

Also fit confound directions:

- length;
- politeness;
- agreement;
- refusal;
- sentiment;
- hedging.

### Experiment C: causal tests

Intervene during preference scoring or generation:

- activation addition toward preference direction;
- response swapping / patching between preferred and dispreferred responses;
- reward-model internal patching;
- generation steering under policy model, with safety controls.

## Controls

- length-matched responses;
- sentiment-matched responses;
- correctness vs politeness conflict pairs;
- sycophancy false-belief pairs;
- random and shuffled directions;
- reward model vs policy disagreement set.

## Metrics

- reward margin;
- preference direction AUC;
- confound-adjusted selectivity;
- causal margin shift;
- policy/reward disagreement;
- helpfulness/correctness/safety tradeoff;
- false-positive preference on sycophantic wrong answers.

## Plots

```text
plots/preference_evidence_dashboard.png
plots/reward_margin_by_domain.png
plots/preference_probe_control_atlas.png
plots/confound_specificity_ladder.png
plots/reward_policy_disagreement_matrix.png
plots/preference_steering_frontier.png
plots/sycophancy_reward_risk_quadrant.png
```

## Tables and artifacts

```text
tables/preference_pair_scores.csv
tables/preference_probe_report.csv
tables/reward_policy_disagreements.csv
tables/preference_intervention_results.csv
tables/preference_evidence_matrix.csv
state/preference_directions.pt
```

## Ledger claims

Allowed:

```text
ATTR/DECODE/CAUSAL: In reward model R, direction D separates preferred from dispreferred benign responses after length and sentiment controls; activation addition changes reward margin by X under controls.
```

Forbidden:

```text
The reward model understands human values.
```

## References

- Ouyang et al., **Training language models to follow instructions with human feedback**.
- Rafailov et al., **Direct Preference Optimization**.
- Work on reward model overoptimization and preference model misspecification.
- Sycophancy and preference-model papers used in Labs 16 and 19.

---

# Lab 33: Multimodal Mechanistic Interpretability

## Core question

Do image and text representations meet in shared features, or do multimodal models use separate routes that only align at the output?

The first 32 labs mostly study text-only transformers. Lab 33 asks how much of the toolkit survives when the input is an image and a question.

## Prerequisites

- Lab 1 residual readouts.
- Lab 4 probes.
- Lab 5 patching.
- Lab 8 sparse features.
- Lab 13 concept confound audits.

## Safety scope

Use benign images only:

- synthetic shapes;
- simple scenes;
- public-domain objects;
- charts/diagrams;
- OCR-free simple labels if needed.

No face recognition, identity inference, surveillance, or private images.

## Implementation plan

### Data

Create a small synthetic and natural multimodal dataset:

```text
data/multimodal_concept_pairs.jsonl
```

Each row:

```json
{
  "item_id": "red_cube_001",
  "image_path": "data/images/red_cube_001.png",
  "question": "What color is the cube?",
  "target": "red",
  "distractor": "blue",
  "concept_family": "color",
  "text_control_prompt": "The cube is red. What color is the cube?",
  "image_control_path": "...",
  "split": "train"
}
```

Families:

- color;
- shape;
- count;
- spatial relation;
- chart reading;
- text-in-image if supported.

### Experiment A: modality readout

Capture states at:

- vision encoder layers;
- vision-language connector;
- language model residual stream at question boundary;
- final answer boundary.

Fit probes for concepts from image-only and text-only controls.

### Experiment B: image/text patching

Perform interchange interventions:

- patch image-derived connector states from clean to corrupt;
- patch text prompt states;
- patch final residual states;
- compare image patch vs text patch localization.

### Experiment C: shared feature bridge

If a VLM SAE or text-model SAE is available, compare:

- image concept direction;
- text concept direction;
- shared decoder features;
- causal steering or patching effect.

## Controls

- text-only prompt with same answer;
- image-only caption control;
- color/shape/count confounds;
- adversarial background;
- random image patch;
- wrong-region image patch;
- paraphrased question.

## Metrics

- image-vs-text probe AUC;
- patch recovery by modality and layer;
- concept specificity;
- cross-modal transfer;
- answer accuracy;
- OCR leakage score;
- background-confound sensitivity.

## Plots

```text
plots/multimodal_evidence_dashboard.png
plots/modality_handoff_atlas.png
plots/image_text_probe_transfer.png
plots/patch_recovery_by_modality.png
plots/concept_specificity_matrix.png
plots/spatial_region_patch_map.png
plots/cross_modal_feature_bridge.png
```

## Tables and artifacts

```text
tables/multimodal_prompt_manifest.csv
tables/modality_probe_report.csv
tables/multimodal_patch_report.csv
tables/cross_modal_transfer.csv
tables/multimodal_evidence_matrix.csv
state/multimodal_directions.pt
```

## Ledger claims

Allowed:

```text
DECODE/CAUSAL: For synthetic color-shape items, connector-layer state S carries image-derived color information, and patching S recovers the clean answer more than random-region controls.
```

Forbidden:

```text
The VLM has a human-like visual concept of color.
```

## References

- Radford et al., **Learning Transferable Visual Models From Natural Language Supervision**.
- OpenAI, **Multimodal Neurons in Artificial Neural Networks**.
- Vision Transformer and VLM interpretability papers.
- Lab 8 SAE validation and Lab 13 confound-control patterns.

---

# Lab 34: Tool Use, Agents, and State Tracking

## Core question

What internal signals precede tool calls, memory reads, and action choices in an agentic language-model loop?

Lab 15 validated multi-turn measurement. Labs 22-25 studied eval framing, belief pressure, and self-report. Lab 34 studies models acting through tools under a controlled harness.

## Safety scope

Use benign toy tools only:

- calculator;
- dictionary lookup over a local JSON;
- calendar simulator;
- file-search simulator over synthetic docs;
- route planner over toy graph;
- unit converter.

No web browsing, no real credentials, no filesystem writes outside a sandbox, no harmful tools.

## Implementation plan

### Data

Create:

```text
data/tool_use_tasks.jsonl
```

Each row:

```json
{
  "task_id": "calc_001",
  "family": "calculator",
  "user_prompt": "What is 17 * 23?",
  "required_tool": "calculator",
  "tool_args": {"expression": "17*23"},
  "answer": "391",
  "distractor_tool": "dictionary",
  "split": "train",
  "notes": "single-step"
}
```

Task families:

- no-tool answerable;
- calculator-needed;
- lookup-needed;
- multi-step tool chain;
- misleading-tool affordance;
- memory update and recall;
- self-report about tool choice.

### Experiment A: tool-choice decoding

At the assistant-generation boundary, fit directions/probes for:

- tool needed vs not needed;
- which tool;
- argument type;
- uncertainty before tool use;
- plan vs answer mode.

### Experiment B: causal steering and patching

Intervene on benign tasks:

- steer toward/away from tool-use direction;
- patch tool-choice boundary state from tool-needed into no-tool and vice versa;
- patch memory-read state across turns;
- compare with prompt-only tool instruction controls.

### Experiment C: self-report and source attribution

Ask after completion:

- why did you use a tool?
- did the tool influence your answer?
- what source produced the answer?

Compare self-report to known tool trace.

## Controls

- no-tool tasks with same surface markers;
- tool-name mentioned but not needed;
- wrong-tool decoy;
- shuffled tool labels;
- random direction;
- prompt-length matched null;
- tool result corrupted in a benign way.

## Metrics

- tool-choice accuracy;
- probe AUC;
- tool-call causal shift;
- argument correctness;
- final answer correctness;
- source-attribution accuracy;
- tool-result reliance;
- hallucinated-tool rate;
- self-report fidelity.

## Plots

```text
plots/tool_use_evidence_dashboard.png
plots/tool_choice_probe_by_depth.png
plots/tool_selection_confusion_matrix.png
plots/tool_state_patch_recovery.png
plots/memory_read_trace_atlas.png
plots/tool_result_reliance_ladder.png
plots/tool_self_report_matrix.png
```

## Tables and artifacts

```text
tables/tool_task_manifest.csv
tables/tool_choice_probe_report.csv
tables/tool_intervention_report.csv
tables/tool_trace_log.csv
tables/tool_self_report_labels.csv
tables/tool_use_evidence_matrix.csv
state/tool_directions.pt
```

## Ledger claims

Allowed:

```text
DECODE/CAUSAL/SELF-REPORT: Tool-needed state is decodable before the tool-call token and patching that state changes benign tool-call probability above controls; self-report matches the known tool trace in X% of hand-labeled cases.
```

Forbidden:

```text
The model has a persistent goal or autonomous plan.
```

## References

- Yao et al., **ReAct: Synergizing Reasoning and Acting in Language Models**.
- Schick et al., **Toolformer**.
- Chain-of-thought monitorability readings from Lab 10 / Lab 25.
- Literature on tool-use agents and evaluation of agentic systems.

---

# Lab 35: Reproducible Interpretability Paper Capstone

## Core question

Can the student produce a small interpretability result that survives preregistration, adversarial review, reproduction, and claim-ledger discipline?

Lab 35 is not another method. It is the “ship it” lab.

## Prerequisites

- Completion of Labs 1-25.
- At least three Labs 26-34 recommended.
- A maintained claim ledger.
- Familiarity with artifact schemas and safety policy.

## Structure

Students choose one of three capstone tracks:

1. **Method replication:** reproduce and stress-test a known result from the course or literature.
2. **New scoped finding:** study a new prompt family, model, or mechanism with preregistered controls.
3. **Audit package:** blind or semi-blind audit of a benign organism, preference model, tool-use agent, or feature dictionary.

## Required phases

### Phase 1: preregistration

Write:

```text
capstone/preregistration.md
```

Must include:

- research question;
- allowed claim;
- forbidden claim;
- dataset;
- model;
- measurement sites;
- controls;
- primary metric;
- stopping rule;
- planned plots;
- expected failure modes;
- safety statement.

### Phase 2: frozen run

Run the lab or custom module with all artifacts saved. No cherry-picked reruns unless logged.

### Phase 3: adversarial review

Another student or AI reviewer fills:

```text
capstone/adversarial_review.md
```

Must attack:

- instrumentation;
- tokenization;
- data leakage;
- confounds;
- interpretation language;
- controls;
- statistical power;
- safety.

### Phase 4: repair run

Student may run one repair experiment, but must keep the original.

### Phase 5: final paper package

Produce:

```text
paper.md
reproduction_guide.md
artifact_index.json
claim_card.md
evidence_matrix.csv
review_response.md
```

## Minimal artifact tree

```text
runs/lab35_reproducible_capstone-*/
  preregistration.md
  paper.md
  claim_card.md
  adversarial_review.md
  review_response.md
  reproduction_guide.md
  evidence_matrix.csv
  plots/
  tables/
  diagnostics/
  notebooks/optional/
```

## Scoring rubric

| Area | Weight |
|---|---:|
| Instrument validity | 20% |
| Control design | 20% |
| Evidence-rung discipline | 20% |
| Reproducibility | 15% |
| Negative-result handling | 10% |
| Writing clarity | 10% |
| Safety and scope | 5% |

## Plots

There is no fixed plot suite. The student must include:

- one dashboard;
- one evidence matrix;
- one per-example or per-family heterogeneity plot;
- one control/specificity plot;
- one failure-case or counterexample plot.

## Ledger claims

The final claim must be one paragraph, with tags:

```text
[Evidence rung]
[Dataset]
[Model]
[Intervention or measurement]
[Controls]
[Scope]
[Non-claim]
[Known falsifier]
```

## References

- Sharkey et al., **Open Problems in Mechanistic Interpretability**.
- Bereska and Gavves, **Mechanistic Interpretability for AI Safety: A Review**.
- Anthropic, **Circuit Tracing: Revealing Computational Graphs in Language Models**.
- Anthropic / Redwood-style blind auditing and model-organism work.
- Any method-specific references used by the selected capstone.

---

## 3. Implementation roadmap for the AI coder

### 3.1 Add registry entries

Add labs 26-35 to `LAB_PROFILES` with model defaults.

Suggested defaults:

| Labs | Default model |
|---|---|
| 26-31 | base model for hook-heavy work, e.g. `gpt2` Tier A and course base model Tier B |
| 32 | reward/preference model plus instruct policy; support fallback proxy mode |
| 33 | small VLM Tier A/B if available, otherwise synthetic connector smoke |
| 34 | instruct model with chat template |
| 35 | no fixed model; inherits selected track |

### 3.2 Add shared utilities before lab-specific code

Implement or extract:

```text
interpkit/causal_abstraction.py
interpkit/path_patching.py
interpkit/editing.py
interpkit/lineage.py
interpkit/auto_interp.py
interpkit/tool_harness.py
interpkit/capstone.py
```

### 3.3 Add datasets and generators

Create deterministic generators:

```text
data/make_causal_abstraction_tasks.py
data/make_path_mediation_tasks.py
data/make_editing_unlearning_targets.py
data/make_training_dynamics_tasks.py
data/make_feature_lineage_corpus.py
data/make_auto_interp_feature_tasks.py
data/make_preference_circuit_pairs.py
data/make_multimodal_concept_pairs.py
data/make_tool_use_tasks.py
```

Every generator writes to `data/MANIFEST.json`.

### 3.4 Add plotting style once

Add namespaces:

```text
causal_abstraction
path
editing
training_dynamics
lineage
auto_interp
preference
multimodal
tool_use
capstone
```

### 3.5 Add tests

For each new lab:

- `py_compile`;
- synthetic plot smoke;
- dataset schema;
- Tier A no-plot smoke;
- artifact schema validation.

---

## 4. Suggested reading spine for the whole third sequence

Students should read these before or during Labs 26-35:

1. Pearl, **Causality**, selected chapters, for intervention language.
2. Geiger et al., **Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability**.
3. Chan et al., **Causal Scrubbing**.
4. Conmy et al., **Towards Automated Circuit Discovery for Mechanistic Interpretability**.
5. Meng et al., **Locating and Editing Factual Associations in GPT**.
6. Guo et al., **Mechanistic Unlearning**.
7. Olsson et al., **In-context Learning and Induction Heads**.
8. Anthropic, **A Mathematical Framework for Transformer Circuits**.
9. Anthropic, **Scaling Monosemanticity**.
10. Anthropic, **Sparse Crosscoders for Cross-Layer Features and Model Diffing**.
11. Anthropic, **Circuit Tracing: Revealing Computational Graphs in Language Models**.
12. Karvonen et al., **SAEBench**.
13. Bills et al., **Language Models Can Explain Neurons in Language Models**.
14. Ouyang et al., **Training language models to follow instructions with human feedback**.
15. Rafailov et al., **Direct Preference Optimization**.
16. Yao et al., **ReAct**.
17. Schick et al., **Toolformer**.
18. Radford et al., **CLIP**.
19. OpenAI, **Multimodal Neurons in Artificial Neural Networks**.
20. Sharkey et al., **Open Problems in Mechanistic Interpretability**.

---

## 5. What students should be able to do after Lab 35

After the third sequence, a student should be able to:

- design a causal abstraction hypothesis;
- test paths, not just components;
- compare localization with editability;
- track mechanisms across checkpoints;
- evaluate feature labels and auto-interpretability systems;
- distinguish reward-model internals from policy behavior;
- adapt the toolkit to multimodal and tool-use settings;
- preregister an interpretability claim;
- survive adversarial review;
- produce a reproducible artifact package.

That is the transition from “I can run interpretability labs” to “I can design interpretability experiments.”

---

## 6. Final note

Labs 26-35 should not make the course broader by becoming fuzzier. They should make the course broader by becoming **stricter**.

The first 25 labs teach students to say: “this is what my instrument can see.”

The special topics should teach them to say: “this is the hypothesis my instrument was built to test, this is the intervention that tried to kill it, this is the counterexample I found, and this is the smaller claim that survived.”

That is expert-level interpretability: not bigger words, better sieves. 🜁

---

## Appendix A. Link bibliography for the AI coder

These links are included so an implementation agent can pull background papers and method details quickly. The per-lab reference lists above name where each source is most relevant.

### Core mechanistic interpretability and causal testing

- Anthropic Transformer Circuits, **A Mathematical Framework for Transformer Circuits**: https://transformer-circuits.pub/2021/framework/index.html
- Geiger et al., **Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability**: https://arxiv.org/abs/2301.04709
- Chan et al., **Causal Scrubbing**: https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing
- Conmy et al., **Towards Automated Circuit Discovery for Mechanistic Interpretability**: https://proceedings.neurips.cc/paper_files/paper/2023/file/34e1dbe95d34d7ebaf99b9bcaeb5b2be-Paper-Conference.pdf
- Sharkey et al., **Open Problems in Mechanistic Interpretability**: https://arxiv.org/abs/2501.16496
- Bereska and Gavves, **Mechanistic Interpretability for AI Safety: A Review**: https://leonardbereska.github.io/blog/2024/mechinterpreview/

### Patching, editing, and unlearning

- Meng et al., **Locating and Editing Factual Associations in GPT**: https://arxiv.org/abs/2202.05262
- ROME project page: https://rome.baulab.info/
- Guo et al., **Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization**: https://proceedings.mlr.press/v267/guo25k.html

### Sparse features, crosscoders, and attribution graphs

- Anthropic, **Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet**: https://transformer-circuits.pub/2024/scaling-monosemanticity/
- Anthropic, **Sparse Crosscoders for Cross-Layer Features and Model Diffing**: https://transformer-circuits.pub/2024/crosscoders/index.html
- Anthropic, **Insights on Crosscoder Model Diffing**: https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html
- Anthropic, **Circuit Tracing: Revealing Computational Graphs in Language Models**: https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- Karvonen et al., **SAEBench: A Comprehensive Benchmark for Sparse Autoencoders**: https://arxiv.org/abs/2503.09532
- Neuronpedia SAEBench page: https://www.neuronpedia.org/sae-bench/info
- Jiralerspong et al., **Cross-Architecture Model Diffing with Crosscoders**: https://openreview.net/forum?id=ZB84SvrZB8

### Steering, representation engineering, and chain-of-thought monitoring

- Turner et al., **Steering Language Models With Activation Engineering**: https://arxiv.org/abs/2308.10248
- Rimsky et al., **Steering Llama 2 via Contrastive Activation Addition**: https://aclanthology.org/2024.acl-long.828/
- Lanham et al., **Measuring Faithfulness in Chain-of-Thought Reasoning**: https://arxiv.org/abs/2307.13702
- Arcuschin et al., **Chain-of-Thought Reasoning In The Wild Is Not Always Faithful**: https://arxiv.org/abs/2503.08679
- Korbak et al., **Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety**: https://arxiv.org/abs/2507.11473

### Model organisms, auditing, and hidden objectives

- Hubinger et al., **Model Organisms of Misalignment: The Case for a New Pillar of Alignment Research**: https://www.alignmentforum.org/posts/ChDH335ckdvpxXaXX/model-organisms-of-misalignment-the-case-for-a-new-pillar-of-1
- Anthropic / audit game discussion, **Auditing language models for hidden objectives**: https://www.lesswrong.com/posts/wSKPuBfgkkqfTpmWJ/auditing-language-models-for-hidden-objectives
- Anthropic, **Replication of the Auditing Game Model Organism**: https://alignment.anthropic.com/2025/auditing-mo-replication/

### Preference models, tool use, and multimodal models

- Ouyang et al., **Training language models to follow instructions with human feedback**: https://arxiv.org/abs/2203.02155
- Rafailov et al., **Direct Preference Optimization**: https://arxiv.org/abs/2305.18290
- Yao et al., **ReAct: Synergizing Reasoning and Acting in Language Models**: https://arxiv.org/abs/2210.03629
- Schick et al., **Toolformer**: https://arxiv.org/abs/2302.04761
- Radford et al., **Learning Transferable Visual Models From Natural Language Supervision**: https://arxiv.org/abs/2103.00020
- OpenAI, **Multimodal Neurons in Artificial Neural Networks**: https://distill.pub/2021/multimodal-neurons/
