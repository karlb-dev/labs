# Special Topics Plan: Labs 26-35

**Status:** canonical build plan, 2026-06-14  
**Source drafts:** `temp/special_topics.md` and `temp/course_review.md`  
**Position:** third sequence after Labs 1-25  

This document is the working plan for Labs 26-35. It keeps the structure of
the original special-topics draft, applies the course-review feedback, and
updates Lab 26 to match the first implementation: residual-stream resampling
with formal hypothesis specs, counterexamples, and a v2 refinement loop.

The goal is not to make the course broader by getting fuzzier. The goal is to
make the course stricter.

```text
instrument -> measurement -> control -> intervention -> artifact -> caveat -> claim ledger
```

Labs 26-35 ask whether students can design new interpretability experiments,
evaluate automated tools, handle negative results, and ship reproducible audit
packages other people can inspect.

## Review-Informed Changes From The Draft

The original `temp/special_topics.md` is ambitious and scientifically well
targeted. The course review adds a warning: the course is already large, so the
third sequence must also make the engineering and teaching surface more
durable.

Applied changes:

- Treat artifact schemas, plot smoke tests, and data validation as part of the
  special-topics plan, not later cleanup.
- Add a standard lab header for time, tier, dependencies, minimum artifacts,
  main plot/table, evidence rung, forbidden claim, and human-label
  requirement.
- Add `science_ready` / fallback-data warnings to every lab that can run on
  smoke data.
- Keep automatic text scoring as triage only; labs that score text must export
  hand-label columns and a calibration guide.
- Prefer official CLI flags over env-only controls.
- Keep Lab 26 narrower than the original draft: residual resampling first,
  path-specific scrubbing in Lab 27.
- Make counterexamples and v2 hypothesis refinement first-class artifacts.
- Add safety status artifacts for preference, multimodal, tool-use, organism,
  and generated-text labs.
- Add route labels so instructors can decide what is core, recommended,
  optional, or extension.

## Evidence Tags

| Tag | Use in Labs 26-35 |
|---|---|
| `OBS` | trajectories, correlations, geometry, timing, descriptive readouts |
| `ATTR` | direct attribution, path scores, edge ledgers, model-diff feature contributions |
| `DECODE` | probes, directions, classifiers, sparse-feature labels |
| `CAUSAL` | patching, ablation, resampling, activation addition, editing, unlearning |
| `AUDIT` | blind or semi-blind recovery, reproducibility, benchmark scorecards, counterexample logs |
| `CONSTRUCTION` | controlled datasets, synthetic mechanisms, benign organisms, toy tools |
| `SELF-REPORT` | generated explanations, confidence reports, source attributions, tool rationales |
| `FORMAL` | explicit high-level variables, low-level mappings, intervention semantics, and predicted invariances |

`FORMAL` does not mean proof. It means the hypothesis was written down in a
testable form before the main intervention.

## Standard Lab Contract

Every new special-topics lab should support:

```bash
python interp_bench.py --lab labXX --tier a --no-plots
python interp_bench.py --lab labXX --tier b --prompt-set full
```

Every lab should write:

```text
runs/labXX_*/
  run_summary.md
  method_card.md
  operationalization_audit.md
  metrics.json
  ledger_suggestions.md
  diagnostics/
    self_check_status.json          # add once shared schemas exist
    data_manifest.json
    safety_status.json              # when safety/public-private boundaries matter
  tables/
    evidence_matrix.csv
    plot_reading_guide.csv
    *_counterexamples.csv           # when a hypothesis can fail locally
  plots/
    *_dashboard.png
    *_specificity_or_control_plot.png
  state/                            # directions, features, edits, tool traces, or model packages
```

Every handout should begin with:

```text
Time estimate:
Compute tier:
Dependencies:
Minimum passing artifacts:
Main plot:
Main table:
Evidence rung:
Forbidden claim:
One-sentence allowed claim:
Human-label requirement:
```

Every run summary should include:

- whether the run is science-ready or smoke-only;
- what failed and what that failure means;
- the main counterexample;
- the smallest claim that survived.

## Course Map

| Lab | Title | Main question | Evidence target | Route |
|---:|---|---|---|---|
| 26 | Causal Abstraction and Causal Scrubbing | Can a formal high-level explanation survive behavior-preserving resampling? | `FORMAL + CAUSAL + AUDIT` | recommended |
| 27 | Path-Specific Patching and Causal Mediation | Which paths, not just nodes, carry a behavior? | `CAUSAL` | recommended |
| 28 | Mechanistic Editing and Unlearning | Does localization make edits or unlearning more robust and specific? | `CAUSAL + AUDIT` | recommended |
| 29 | Training Dynamics and Circuit Birth | When do features and circuits emerge during training or fine-tuning? | `OBS + ATTR + DECODE` | optional |
| 30 | Cross-Layer and Cross-Model Feature Geometry | Can features be tracked across layers, checkpoints, and models? | `DECODE + ATTR` | optional |
| 31 | Automated Interpretability at Scale | Can automated feature labels be evaluated rather than trusted? | `AUDIT + DECODE` | recommended |
| 32 | Reward Models and Preference Circuits | What internal features drive reward or preference judgments? | `ATTR + DECODE + CAUSAL` | optional |
| 33 | Multimodal Mechanistic Interpretability | Do text and image features meet in shared mechanisms? | `OBS + DECODE + CAUSAL` | extension |
| 34 | Tool Use, Agents, and State Tracking | What internal signals precede tool calls, memory reads, and action choice? | `OBS + DECODE + CAUSAL + SELF-REPORT` | optional |
| 35 | Reproducible Interpretability Paper Capstone | Can a student ship a preregistered, reproducible, adversarially reviewed study? | `AUDIT + FORMAL` | capstone |

## Build Order

The default build order is sequential, because the labs depend conceptually on
each other:

```text
26 -> 27 -> 28 -> 29 -> 30 -> 31 -> 32 -> 33 -> 34 -> 35
```

If we need to prioritize research value over strict order, Lab 28 can move
ahead of Labs 29-31. The review specifically identified Lab 26 and Lab 28 as
high-value additions because they deepen causal claims.

Before Labs 29-35, add lightweight shared helpers where practical:

```text
interpkit/artifacts.py       # optional extraction target; do not block Lab 27
interpkit/datasets.py
interpkit/evidence.py
interpkit/metrics.py
interpkit/plots.py
interpkit/schemas.py
```

Do not over-abstract the experiments. Extract boring repeated plumbing first:
CSV/JSON writing, schema validation, safe numeric functions, plot style,
evidence matrices, marker scoring, dataset manifest checks, and safety cards.

## Lab 26: Causal Abstraction and Causal Scrubbing

**Implemented first pass:** `labs/lab26_causal_abstraction.py` and
`labs/lab26_causal_abstraction.md`.

### Core Question

Can a proposed high-level explanation of model behavior be tested as a causal
abstraction rather than a loose collection of interesting components?

### Actual First-Pass Scope

The original draft allowed residual, head, path, and node resampling. The first
implementation deliberately narrowed this:

- formal JSON hypotheses in `specs/lab26_*_hypothesis.json`;
- frozen data in `data/causal_abstraction_tasks.csv`;
- residual-stream resampling at named source/final positions;
- preserving donors, broken-variable donors, random matched donors, wrong-site
  controls, and self-patching no-ops;
- counterexample and v2-refinement logs.

Path-specific scrubbing belongs in Lab 27.

### Data

`data/causal_abstraction_tasks.csv` contains induction-copying rows and
relation-answer rows. Required columns:

```text
item_id,domain,split,high_level_task,template_family,
prompt,target,distractor,source_token,source_position,target_position,
relation_family,subject,answer,high_level_variables_json
```

### Main Artifacts

```text
method_card.md
causal_abstraction_spec.md
operationalization_audit.md
tables/hypothesis_spec_audit.csv
tables/resampling_interventions.csv
tables/variable_preservation_summary.csv
tables/best_hypothesis_cells.csv
tables/evidence_matrix.csv
tables/counterexamples.csv
tables/hypothesis_refinement_log.csv
plots/causal_abstraction_dashboard.png
plots/resampling_preservation_matrix.png
plots/hypothesis_pass_fail_atlas.png
plots/variable_specificity_ladder.png
plots/refinement_trajectory.png
plots/counterexample_gallery.png
```

### Claim Grammar

Allowed:

```text
FORMAL + CAUSAL + AUDIT: Under this residual-resampling battery, preserving
variable X at site S preserved the target margin more than breaking X or using
matched controls; counterexamples require v2 scope Y.
```

Forbidden:

```text
The model implements exactly this algorithm.
```

## Lab 27: Path-Specific Patching and Causal Mediation

### Core Question

Which paths through the network carry a behavior, and how do path effects
differ from node-level importance?

### Implementation Plan

Start with attention-head paths and only then expand:

```text
source head output at position p -> receiver head input at position q -> final readout
```

Then add:

```text
residual band -> MLP -> final readout
head -> MLP -> final readout
head -> receiver head -> MLP -> final readout
```

Use three behaviors:

- induction completion;
- factual recall;
- relation swap, seeded from Lab 26 and Lab 12.

Create:

```text
data/path_mediation_tasks.csv
```

Required columns:

```text
item_id,domain,prompt,clean_prompt,corrupt_prompt,target,distractor,
positions_json,candidate_nodes_json,candidate_paths_json
```

### Experiments

- Node-effect baseline: residual patching, head ablation, component patching,
  DLA/head-attribution screens.
- Path patching: isolate selected source-to-receiver routes while controls keep
  other routes corrupt or neutral.
- Mediation accounting: compare source effect, receiver effect, path effect,
  joint effect, and interaction residual.

### Controls

Wrong receiver, wrong position, matched node with mismatched motif, random
path, reverse path, and no-op path patch.

### Main Artifacts

```text
tables/node_effect_baseline.csv
tables/path_patch_report.csv
tables/path_specificity_controls.csv
tables/mediation_accounting.csv
tables/path_evidence_matrix.csv
state/path_candidates.json
plots/path_mediation_dashboard.png
plots/node_vs_path_effects.png
plots/path_specificity_matrix.png
plots/path_graph.png
plots/mediation_accounting_waterfall.png
plots/heldout_path_transfer.png
```

### Claim Grammar

Allowed:

```text
CAUSAL: In this prompt family, path P from source node A to receiver node B
carries X% of clean-vs-corrupt recovery under path-patching intervention I.
```

Forbidden:

```text
A causes B in every context.
```

## Lab 28: Mechanistic Editing and Unlearning

### Core Question

Does mechanistic localization make model editing or unlearning more robust,
specific, and measurable?

### Safety Scope

Use only benign targets: factual associations, toy relation families, harmless
Lab 20 organisms, synthetic secrets that are not real private data. No harmful
capabilities, no real personal data, no jailbreak edits, and no refusal
ablation.

### Implementation Plan

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

Begin with inference-time edits because they are easier to validate and safer
to rollback:

- activation steering at localized site;
- feature clamp or feature suppression;
- adapter-layer masking when Lab 20/21 artifacts exist;
- rank-one residual/MLP update only where apply/restore plumbing is verified;
- random-site, random-direction, and prompt-only baselines.

### Experiments

- Localization comparison across residual patching, DLA, ablation,
  SAE/transcoder feature activation, and adapter weight mass.
- Edit methods across target, paraphrase, neighbor, retain, and adversarial
  paraphrase sets.
- Robustness audit: side effects, re-learning after context reminder, and
  localization/editability correlation.

### Main Artifacts

```text
tables/localization_candidates.csv
tables/editing_results.csv
tables/retain_forget_matrix.csv
tables/paraphrase_robustness.csv
tables/edit_evidence_matrix.csv
state/edit_vectors.pt
state/edit_metadata.json
plots/editing_unlearning_dashboard.png
plots/localization_vs_editability.png
plots/edit_method_frontier.png
plots/paraphrase_robustness_matrix.png
plots/neighbor_preservation_atlas.png
plots/mechanistic_locality_ladder.png
plots/unlearning_retain_forget_frontier.png
```

### Claim Grammar

Allowed:

```text
CAUSAL + AUDIT: Method M applied at localized site S changed target behavior T
on prompts P and transferred to paraphrases while preserving retain set R
better than random-site controls.
```

Forbidden:

```text
The fact was erased from the model.
```

## Lab 29: Training Dynamics and Circuit Birth

### Core Question

When does a representation or circuit appear during training, and can we
distinguish feature birth, sharpening, migration, and reuse?

### Implementation Plan

Use one or more checkpoint sequences:

- Pythia checkpoints for real training dynamics;
- a tiny in-course transformer trained on induction/copy tasks;
- Lab 20 organism fine-tuning checkpoints if available;
- optional small LoRA checkpoints.

Create:

```text
data/training_dynamics_tasks.csv
```

Required columns:

```text
item_id,task_family,prompt,target,distractor,split,expected_mechanism,notes
```

### Experiments

- Checkpoint behavior trajectory: margins, logit-lens event depths, DLA,
  attention motifs, probe selectivity, and patch recovery where feasible.
- Feature and circuit birth: induction motifs, head attribution, ablation
  effects, relation/truth directions, localization migration.
- Phase classification: absent, behavioral before interpretable, decodable
  before behavioral, circuit present, migration, sharpening, redistributed.

### Controls

Random checkpoint or random-initialized model, shuffled labels, control task
not trained, held-out prompt families, and untrained LoRA adapter.

### Main Artifacts

```text
tables/checkpoint_behavior.csv
tables/checkpoint_probe_selectivity.csv
tables/checkpoint_circuit_summary.csv
tables/mechanism_birth_events.csv
tables/feature_lineage.csv
state/checkpoint_directions.pt
plots/training_dynamics_dashboard.png
plots/behavior_vs_decodability_timeline.png
plots/circuit_birth_atlas.png
plots/depth_migration_map.png
plots/checkpoint_feature_lineage.png
plots/intervention_transfer_over_time.png
plots/random_model_control_panel.png
```

### Claim Grammar

Allowed:

```text
OBS/DECODE/CAUSAL: In checkpoint sequence C, behavior B appears before/after
direction D becomes decodable; causal intervention I becomes effective at
checkpoint K under controls.
```

Forbidden:

```text
The model first learned concept X at exactly this step.
```

## Lab 30: Cross-Layer and Cross-Model Feature Geometry

### Core Question

Are features static layer-local objects, or do they persist, split, merge, and
move across layers, checkpoints, and model families?

### Implementation Plan

Use a mixed corpus:

- Lab 8 SAE corpus;
- persona, sycophancy, and certainty prompts from Labs 14-17;
- model-diff inventory from Lab 19;
- held-out lineage prompts.

Create:

```text
data/feature_lineage_corpus.csv
```

Required columns:

```text
row_id,family,domain,source_lab,text,group_id,split,labels_json
```

First pass can avoid expensive cross-layer dictionary training by comparing
loaded dictionaries, decoder cosines, activation correlations, top-context
overlap, label stability, and causal clamp transfer. Add trained cross-layer
dictionaries only after the validation path works.

### Controls

Random feature matching, random initialized model, shuffled corpus labels,
same-layer independent SAE comparison, activation-frequency matched controls,
and token-feature confusable pairs.

### Main Artifacts

```text
tables/feature_lineage_edges.csv
tables/feature_lineage_nodes.csv
tables/split_merge_candidates.csv
tables/label_stability_summary.csv
tables/cross_model_feature_overlap.csv
state/cross_layer_dictionary.pt
state/lineage_graph.json
plots/feature_lineage_dashboard.png
plots/cross_layer_feature_graph.png
plots/lineage_similarity_matrix.png
plots/feature_split_merge_atlas.png
plots/label_stability_ladder.png
plots/cross_model_feature_overlap.png
plots/causal_transfer_by_layer.png
```

### Claim Grammar

Allowed:

```text
DECODE + ATTR: Feature family F has a recurring activation/decoder/top-context
lineage across layers L1-L3, above matched random-feature controls.
```

Forbidden:

```text
This is the same concept everywhere in the model.
```

## Lab 31: Automated Interpretability at Scale

### Core Question

Can automated feature explanations be evaluated, calibrated, and falsified at
scale?

### Implementation Plan

Use feature candidates from Lab 8 SAE atlas, Lab 19 crosscoder features, Lab 30
lineage features, random SAE features, and known synthetic toy-model features.

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
  "top_contexts": [],
  "heldout_contexts": [],
  "negative_contexts": [],
  "confusable_contexts": [],
  "gold_label": "optional"
}
```

Explanation methods should be explicitly tiered:

- simple majority-domain heuristic;
- structured local explanation from top contexts;
- explanation plus generated tests;
- human-written label for calibration subset;
- optional LLM explanation only when the environment supports it.

### Experiments

- Generate explanations and tests.
- Score explanations by predicting held-out activations, hard negatives,
  confusable negatives, paraphrased positives, and token-overlap decoys.
- Calibrate confidence and abstention.

### Controls

Random features, shuffled top contexts, key-token deletion, confusable domain
pairs, synthetic features with known labels, and adversarial contexts that
match label words but not concept.

### Main Artifacts

```text
tables/generated_explanations.csv
tables/explanation_tests.csv
tables/explanation_scores.csv
tables/auto_interp_evidence_matrix.csv
tables/human_review_queue.csv
plots/auto_interp_dashboard.png
plots/explanation_quality_matrix.png
plots/confidence_calibration_curve.png
plots/abstention_frontier.png
plots/confusable_pair_failure_atlas.png
plots/random_feature_sanity_panel.png
```

### Human Labels

Export the shared label fields:

```text
student_label_primary,student_label_secondary,student_confidence,
student_evidence_span,reviewer_label,agreement_status
```

### Claim Grammar

Allowed:

```text
AUDIT + DECODE: Explanation method E predicted held-out feature activations
with AUC X and abstained on Y% of high-polysemantic features under test suite T.
```

Forbidden:

```text
The automated label is the feature's meaning.
```

## Lab 32: Reward Models and Preference Circuits

### Core Question

What internal features drive preference or reward judgments, and how do those
features differ from policy behavior?

### Safety Scope

Use benign preference data only: helpfulness, harmless factual honesty, benign
boundary prompts, style preference, and sycophancy controls. No harmful prompt
optimization, no refusal ablation, and no jailbreak search.

### Implementation Plan

Create:

```text
data/preference_circuit_pairs.csv
```

Required columns:

```text
pair_id,domain,prompt,response_a,response_b,preferred,
preference_type,confound_type,split,notes
```

Support three scoring modes:

- open reward model when available;
- DPO/reference-policy log-prob ratio as preference proxy;
- base-vs-instruct policy log-prob preference as smoke fallback.

### Experiments

- Reward/preference readout on paired responses.
- Internal preference directions at prompt+response boundary.
- Confound directions for length, politeness, agreement, refusal, sentiment,
  and hedging.
- Causal tests by activation addition, patching, or reward-model internal
  patching when architecture permits.

### Main Artifacts

```text
tables/preference_pair_scores.csv
tables/preference_probe_report.csv
tables/reward_policy_disagreements.csv
tables/preference_intervention_results.csv
tables/preference_evidence_matrix.csv
state/preference_directions.pt
plots/preference_evidence_dashboard.png
plots/reward_margin_by_domain.png
plots/preference_probe_control_atlas.png
plots/confound_specificity_ladder.png
plots/reward_policy_disagreement_matrix.png
plots/preference_steering_frontier.png
plots/sycophancy_reward_risk_quadrant.png
```

### Claim Grammar

Allowed:

```text
ATTR/DECODE/CAUSAL: In reward model R, direction D separates preferred from
dispreferred benign responses after length and sentiment controls; activation
addition changes reward margin by X under controls.
```

Forbidden:

```text
The reward model understands human values.
```

## Lab 33: Multimodal Mechanistic Interpretability

### Core Question

Do image and text representations meet in shared features, or do multimodal
models use separate routes that only align at the output?

### Safety Scope

Use benign images only: synthetic shapes, simple scenes, public-domain objects,
charts/diagrams, and simple labels if needed. No face recognition, identity
inference, surveillance, or private images.

### Implementation Plan

Create:

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

First pass should choose a small VLM whose internals can be captured. If that
is not reliable, implement a synthetic connector smoke mode and mark
`science_ready = false` until real VLM hooks are verified.

### Experiments

- Modality readout at vision layers, connector layers, language residual
  stream, and final answer boundary.
- Image/text patching between clean and corrupt image-question pairs.
- Shared feature bridge if a VLM SAE or compatible text SAE exists.

### Controls

Text-only prompt with same answer, image-only caption control, color/shape/count
confounds, adversarial background, random image patch, wrong-region image
patch, and paraphrased question.

### Main Artifacts

```text
tables/multimodal_prompt_manifest.csv
tables/modality_probe_report.csv
tables/multimodal_patch_report.csv
tables/cross_modal_transfer.csv
tables/multimodal_evidence_matrix.csv
state/multimodal_directions.pt
plots/multimodal_evidence_dashboard.png
plots/modality_handoff_atlas.png
plots/image_text_probe_transfer.png
plots/patch_recovery_by_modality.png
plots/concept_specificity_matrix.png
plots/spatial_region_patch_map.png
plots/cross_modal_feature_bridge.png
```

### Claim Grammar

Allowed:

```text
DECODE/CAUSAL: For synthetic color-shape items, connector-layer state S carries
image-derived color information, and patching S recovers the clean answer more
than random-region controls.
```

Forbidden:

```text
The VLM has a human-like visual concept of color.
```

## Lab 34: Tool Use, Agents, and State Tracking

### Core Question

What internal signals precede tool calls, memory reads, and action choices in
a controlled agentic language-model loop?

### Safety Scope

Use benign toy tools only: calculator, dictionary lookup over local JSON,
calendar simulator, file-search simulator over synthetic docs, route planner
over a toy graph, and unit converter. No web browsing, no real credentials, no
real filesystem writes, and no harmful tools.

### Implementation Plan

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

### Experiments

- Decode tool needed vs not needed, which tool, argument type, uncertainty
  before tool use, and plan vs answer mode.
- Causal steering and patching of tool-choice boundary states.
- Memory-read patching across turns.
- Source attribution after completion against the known tool trace.

### Controls

No-tool tasks with same surface markers, tool-name mentioned but not needed,
wrong-tool decoy, shuffled tool labels, random direction, prompt-length matched
null, and benign corrupted tool result.

### Main Artifacts

```text
tables/tool_task_manifest.csv
tables/tool_choice_probe_report.csv
tables/tool_intervention_report.csv
tables/tool_trace_log.csv
tables/tool_self_report_labels.csv
tables/tool_use_evidence_matrix.csv
state/tool_directions.pt
plots/tool_use_evidence_dashboard.png
plots/tool_choice_probe_by_depth.png
plots/tool_selection_confusion_matrix.png
plots/tool_state_patch_recovery.png
plots/memory_read_trace_atlas.png
plots/tool_result_reliance_ladder.png
plots/tool_self_report_matrix.png
```

### Claim Grammar

Allowed:

```text
DECODE/CAUSAL/SELF-REPORT: Tool-needed state is decodable before the tool-call
token and patching that state changes benign tool-call probability above
controls; self-report matches the known tool trace in X% of hand-labeled cases.
```

Forbidden:

```text
The model has a persistent goal or autonomous plan.
```

## Lab 35: Reproducible Interpretability Paper Capstone

### Core Question

Can the student produce a small interpretability result that survives
preregistration, adversarial review, reproduction, and claim-ledger discipline?

Lab 35 is not another method. It is the ship-it lab.

### Tracks

Students choose one:

1. Method replication: reproduce and stress-test a known result from the course
   or literature.
2. New scoped finding: study a new prompt family, model, or mechanism with
   preregistered controls.
3. Audit package: blind or semi-blind audit of a benign organism, preference
   model, tool-use agent, or feature dictionary.

### Required Phases

1. Preregistration:

```text
capstone/preregistration.md
```

Must include research question, allowed claim, forbidden claim, dataset, model,
measurement sites, controls, primary metric, stopping rule, planned plots,
expected failure modes, and safety statement.

2. Frozen run:

Run once with all artifacts saved. Any cherry-picked rerun must be logged.

3. Adversarial review:

```text
capstone/adversarial_review.md
```

Review must attack instrumentation, tokenization, data leakage, confounds,
interpretation language, controls, statistical power, and safety.

4. Repair run:

One repair experiment is allowed, but the original run stays in the package.

5. Final package:

```text
paper.md
reproduction_guide.md
artifact_index.json
claim_card.md
evidence_matrix.csv
review_response.md
```

### Minimal Artifact Tree

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

### Scoring Rubric

| Area | Weight |
|---|---:|
| Instrument validity | 20% |
| Control design | 20% |
| Evidence-rung discipline | 20% |
| Reproducibility | 15% |
| Negative-result handling | 10% |
| Writing clarity | 10% |
| Safety and scope | 5% |

### Required Plots

There is no fixed plot suite, but the package must include:

- one dashboard;
- one evidence matrix;
- one per-example or per-family heterogeneity plot;
- one control/specificity plot;
- one failure-case or counterexample plot.

## Shared Data And Generator Plan

Create deterministic generators as each lab is implemented:

```text
data/make_path_mediation_tasks.py
data/make_editing_unlearning_targets.py
data/make_training_dynamics_tasks.py
data/make_feature_lineage_corpus.py
data/make_auto_interp_feature_tasks.py
data/make_preference_circuit_pairs.py
data/make_multimodal_concept_pairs.py
data/make_tool_use_tasks.py
```

Every generator should update `data/MANIFEST.json` and record:

- row counts;
- family/domain counts;
- generator name;
- source files;
- sha256;
- verified tokenizers or runtime verification policy;
- safety notes where relevant.

## Shared Test Plan

Each new lab should pass:

```bash
python -m py_compile interp_bench.py labs/labXX_*.py
python interp_bench.py --lab labXX --tier a --no-plots
```

As the validation harness is built, add:

```bash
python scripts/validate_all_labs.py --compile
python scripts/validate_all_labs.py --plot-smoke
python scripts/validate_all_labs.py --dataset-schema
python scripts/validate_all_labs.py --artifact-schema
```

Minimum schema targets:

- `tables/evidence_matrix.csv`
- `plots/plot_reading_guide.csv`
- `ledger_suggestions.md`
- `diagnostics/data_manifest.json`
- `diagnostics/safety_status.json`
- `diagnostics/self_check_status.json`

## Human-Label Protocol

Labs with generated text or automated explanations must export these fields:

```text
student_label_primary
student_label_secondary
student_confidence
student_evidence_span
reviewer_label
agreement_status
```

Applies by default to Labs 31, 32, 34, and 35, and to any Lab 28/33 mode that
scores generations or captions. This extends the review recommendation for
Labs 10, 16, 17, 18, 22, 24, and 25.

## Safety Artifacts

Every lab with safety or public/private boundaries should write:

```json
{
  "lab": "labXX",
  "unsafe_prompt_sampling": false,
  "refusal_ablation": false,
  "harmful_completion_generation": false,
  "blocked_rows": 0,
  "public_private_boundary_relevant": false,
  "science_ready": true
}
```

If a lab uses fallback data, unsupported model internals, synthetic-only
connectors, or missing external artifacts, `science_ready` must be false and
the run summary must say so near the top.

## Reading Spine

Students should read these before or during Labs 26-35:

1. Pearl, selected causality chapters.
2. Geiger et al., causal abstraction.
3. Chan et al., causal scrubbing.
4. Conmy et al., ACDC.
5. Meng et al., ROME.
6. Guo et al., mechanistic unlearning.
7. Olsson et al., induction heads.
8. Anthropic, A Mathematical Framework for Transformer Circuits.
9. Anthropic, Scaling Monosemanticity.
10. Anthropic, sparse crosscoders and model diffing.
11. Anthropic, circuit tracing.
12. SAEBench.
13. Bills et al., automated neuron explanations.
14. Ouyang et al., RLHF.
15. Rafailov et al., DPO.
16. Yao et al., ReAct.
17. Schick et al., Toolformer.
18. Radford et al., CLIP.
19. OpenAI, multimodal neurons.
20. Sharkey et al., open problems in mechanistic interpretability.

## End State After Lab 35

A student who completes the third sequence should be able to:

- design a causal-abstraction hypothesis;
- test paths, not only nodes;
- compare localization with editability;
- track mechanisms across checkpoints;
- evaluate feature labels and automated interpretability systems;
- distinguish reward-model internals from policy behavior;
- adapt the toolkit to multimodal and tool-use settings;
- preregister an interpretability claim;
- survive adversarial review;
- produce a reproducible artifact package.

The final sentence students should be able to defend is:

```text
This is the hypothesis my instrument was built to test, this is the
intervention that tried to kill it, this is the counterexample I found, and
this is the smaller claim that survived.
```

