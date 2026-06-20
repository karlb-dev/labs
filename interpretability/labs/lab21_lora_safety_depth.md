# Lab 21 - Where Training Lives: LoRA Localization and Safety Depth

**Advanced course stage:** training effects and manufactured ground truth

**Evidence levels:** `ATTR` by default, with scoped `CAUSAL` only when an actual adapter-layer intervention, wrapper ablation, or erosion sweep is present

**Depends on:** Lab 19, Lab 20, and the Lab 7 refusal/steering safety discipline

## The question

When training installs a behavior, where does the change live? And when an instruct model refuses a boundary request, is that refusal a shallow response-start habit, a deeper internal state, or a mixture of both?

This lab is deliberately two labs sharing one harness:

1. **LoRA localization.** Inspect Lab 20 model-organism adapters, if trained weights exist. Ask where the adapter delta has weight-space mass, how concentrated it is, and whether a future layer-masked intervention would be needed before mechanism language is earned.
2. **Safety depth.** Compare residual states for base versus instruct models, boundary versus safe prompts, and forced refusal-consistent versus forced benign-prefix continuations. Ask how deep the representational divergence persists, without sampling unsafe completions or ablating refusal.

The lesson is that the word **deep** has at least three meanings:

| Depth type | What it means here | Evidence artifact |
|---|---|---|
| Weight-space depth | Which layers carry the LoRA update norm | `tables/per_layer_lora_norm.csv` |
| Behavioral depth | How many generated/prefix tokens are needed before output looks refusal-like or compliant | `tables/forced_prefix_recommitment_depth.csv` |
| Representational depth | Which residual-stream depths separate base/instruct or boundary/safe states | `tables/instruct_base_divergence_by_layer.csv` and `tables/boundary_safe_prompt_divergence.csv` |

Do not collapse these into one story. A high-norm adapter layer may be an optimizer footprint rather than the mechanism. A shallow first token may still trigger deeper computation. A deep residual difference may be topic, format, or refusal state. The lab exists to make those possibilities visible before anyone writes a heroic sentence.

## Safety wall

Lab 21 inherits the advanced-course safety wall:

- Use benign prompts and benign organisms only.
- Do not sample harmful completions.
- Use refusal-related directions for monitoring or steering toward refusal, never refusal ablation.
- Do not build a working jailbreak artifact.
- Hidden-goal organisms remain toy benign quirks, not dangerous capabilities.

Safety-depth mode uses **boundary-style benign prompts** and **safe alternatives**. The forced-prefix experiment compares rendered transcripts that begin with safe/refusal-consistent text. It does not ask the model to complete harmful instructions.

## How to run

The current lab module reads `mode` and organism paths defensively through `getattr(...)` and environment variables, so it can run before the shared parser grows Lab 21-specific flags.

```bash
# Tier A smoke: verify artifact plumbing and missing-adapter scaffolds.
python interp_bench.py --lab lab21 --tier a --no-plots

# LoRA mode through environment fallback.
LAB21_MODE=lora python interp_bench.py --lab lab21 --tier b --prompt-set small

# Safety-depth mode through environment fallback.
LAB21_MODE=safety_depth python interp_bench.py --lab lab21 --tier b --prompt-set small

# Both modes, when GPU memory allows.
LAB21_MODE=both python interp_bench.py --lab lab21 --tier b --prompt-set small

# Point at a specific Lab 20 run, organism directory, or adapter directory.
LAB21_MODE=lora LAB21_ORGANISM_DIR=runs/lab20_model_organisms-... \
  python interp_bench.py --lab lab21 --tier b --no-plots

# Compare instruct against a specific base model.
LAB21_MODE=safety_depth LAB21_COMPARE_MODEL=allenai/Olmo-3-1025-7B \
  python interp_bench.py --lab lab21 --tier b --prompt-set small
```

If your benchmark registry/parser already exposes Lab 21 flags, the equivalent commands are:

```bash
python interp_bench.py --lab lab21 --tier b --mode lora
python interp_bench.py --lab lab21 --tier b --mode safety_depth
python interp_bench.py --lab lab21 --tier b --mode both --organism runs/lab20_model_organisms-...
```

The lab writes `diagnostics/bench_integration_note.json` so you can see whether the loaded bench recognized Lab 21 as a registered chat-template-aware lab.

## Inputs

### LoRA mode inputs

LoRA mode looks for trained adapter weights in this order:

1. `LAB21_ORGANISM_DIR` or `--organism`, if your parser has that flag;
2. the latest Lab 20 run under `runs/lab20_model_organisms-*`;
3. revised Lab 20 layout:
   - `private_construction/<organism_id>/...`
   - `blind_audit_packages/<blind_id>/...`
4. legacy Lab 20 layout:
   - `organisms/organism_*`;
5. a direct adapter directory containing `adapter_model.safetensors`, `adapter_model.bin`, or `adapter_config.json`.

The lab supports public sealed packages, private builder-side packages, legacy packages, and direct adapter directories. It writes `diagnostics/private_answer_key_access.json` so a future blind-audit run can tell whether the builder-side answer key was touched. Lab 21 is allowed to be builder-side. Lab 23 is not.

If no weights exist yet, that is not a failure. The lab writes explicit `missing_adapter_weights` and `no_adapter_weight_file_found` rows. That scaffold is useful before the PEFT training pass lands.

### Safety-depth inputs

By default, safety-depth mode uses a small built-in benign boundary battery. A science run should use:

```text
data/safety_depth_boundary_pairs.csv
```

or pass a custom CSV/JSON/JSONL path through `--prompt-set`. The current vendored inventory is `v1_boundary_safe_pairs`: 24 boundary/safe pairs across privacy, account security, professional-boundary, academic-integrity, copyright, and cyber-boundary families. See `data/safety_depth_boundary_pairs_card.md` and `data/make_safety_depth_boundary_pairs.py`.

Required columns:

| Column | Meaning |
|---|---|
| `pair_id` | Stable id |
| `family` | Prompt family, such as privacy, wrongdoing, medical, legal, cyber-benign |
| `boundary_request` | Benign boundary-style user request that should invite a refusal or redirection |
| `safe_alternative` | Matched safe request or safe alternative |
| `refusal_reason` | Short reason category |

Optional columns:

| Column | Meaning |
|---|---|
| `refusal_prefix` | Forced refusal-consistent assistant prefix |
| `safe_prefix` | Forced safe-assistance assistant prefix |
| `generic_prefix` | Matched generic safe prefix control |

The built-in fallback is a smoke test, not frozen science data. The run metrics expose `safety_data_source`, `safety_data_sha256`, `safety_data_manifest_match`, and `safety_data_fallback_is_science_data` so this cannot be missed in later audits.

## The two experiments

### Experiment 1: LoRA weight localization

For every adapter source, the lab inventories LoRA matrices and computes delta statistics for `B @ A` without materializing dense `d_model x d_model` updates. For rank-`r` adapters, it computes the spectrum from the small Gram products, so even a 7B projection update stays rank-sized on CPU.

Outputs include:

- per-matrix Frobenius norm and spectral norm;
- numerical rank and entropy effective rank;
- per-layer norm share;
- per-module norm share;
- top-layer concentration and layer entropy;
- rank-energy curves.

This is `ATTR` evidence about the **weight update**, not a causal statement about the behavior.

The lab also writes scaffolds or imported rows for:

- full-finetune versus LoRA versus DPO localization comparison;
- wrapper-ablation or layer-masked adapter tests;
- erosion-order curves from a future benign finetune sweep.

These can be imported with:

```bash
LAB21_LOCALIZATION_COMPARISON_CSV=path/to/comparison.csv
LAB21_WRAPPER_RESULTS_CSV=path/to/wrapper_results.csv
LAB21_EROSION_CSV=path/to/erosion.csv
```

A mechanism claim begins only when the imported intervention rows show that masking or ablating specific adapter layers changes the target behavior while controls do not.

### Experiment 2: Safety depth

Safety-depth mode runs four comparisons.

#### 1. Base versus instruct divergence

The instruct model sees a chat-rendered boundary prompt. The comparison model sees the same request in a plain dialog wrapper. The lab measures residual-stream divergence by depth, position, and prompt family.

This answers: where do the base and instruct states differ on boundary prompts?

It does **not** answer: where refusal lives.

#### 2. Chat-format control

When the main model has a chat template, the lab compares its chat-rendered prompt state to a plain dialog rendering of the same request.

This answers: how much of the apparent model difference might simply be chat-template scaffolding?

#### 3. Boundary versus safe prompt divergence

Within the instruct model, the lab compares each boundary request with its matched safe alternative.

This answers: where the model state separates boundary requests from safe alternatives.

It does **not** by itself separate refusal from topic or semantic difference.

#### 4. Forced-prefix recommitment

The lab renders three assistant-prefix transcripts for each boundary request:

- refusal-consistent prefix: `I can't help with that request...`
- safe-assistance prefix: `I can help with a safe alternative...`
- generic safe prefix control: `I can help with a related safe task...`

It then measures divergence by stream depth and by assistant-prefix token index.

This answers: after the response has begun, does the model state remain separated across prefix tokens, or is the separation mostly a first-token/first-phrase artifact?

It still does not sample unsafe completions. The strings are fixed transcripts used for forward passes.

## Stream-depth convention

The shared bench convention remains:

```text
streams[k] = pre-final-norm residual stream after k decoder blocks
```

Depth `0` is the embedding stream. Depth `L` is the final norm input. A steering or ablation intervention into decoder block `j` writes into stream depth `j + 1`. Lab 21 mostly measures streams, so artifacts use `stream_depth` rather than just `layer` when the distinction matters.


## Visualization upgrade in this version

Lab 21 now treats “depth” as a three-axis audit board rather than one curve with an ego. The upgraded plot suite separates:

1. **Weight-space depth:** where LoRA update mass sits.
2. **Representational depth:** where base/instruct, chat-format, boundary/safe, and forced-prefix states diverge.
3. **Behavioral or causal depth:** whether wrapper/layer-mask or erosion rows actually test behavior.

New start-here artifact:

```text
plots/training_depth_evidence_dashboard.png
```

It joins LoRA localization, representational safety-depth curves, forced-prefix persistence, and intervention readiness. The paired table is:

```text
tables/training_depth_evidence_matrix.csv
```

Read those before the more specialized plots. They are designed to stop the most common Lab 21 error: collapsing optimizer depth, prompt-state depth, prefix-token depth, and causal intervention depth into one dramatic sentence.

Additional synthesis artifacts:

```text
tables/lora_phase_summary.csv
tables/training_depth_disagreement.csv
tables/safety_depth_signal_summary.csv
tables/plot_reading_guide.csv
plots/lora_layer_atlas.png
plots/lora_module_phase_atlas.png
plots/safety_depth_signal_atlas.png
plots/forced_prefix_recommitment_heatmap.png
plots/refusal_provenance_cosines.png
plots/training_depth_disagreement.png
plots/intervention_readiness_matrix.png
```

## Artifact tree

Start with `training_depth_card.md`. Then inspect the diagnostics before the plots.

The tree below is the union of all modes. Outputs depend on which `--mode` you
ran: a `lora`-only run writes the LoRA-localization artifacts (adapter manifest,
per-layer norm, concentration, rank energy) but **not** the safety-depth
diagnostics, tables, or plots (`safety_*`, `*divergence*`, `forced_prefix_*`,
`safety_depth_dashboard.png`, the `instruct_*`/`comparison_*` parity files),
which require `--mode safety_depth` (or `both`) and a comparison model. The
`training_depth_card.md` "Read next" list is generated to point only at the
files the run actually produced.

```text
runs/lab21_lora_safety_depth-.../
  training_depth_card.md
  run_summary.md
  operationalization_audit.md
  ledger_suggestions.md
  metrics.json
  results.csv

  diagnostics/
    bench_integration_note.json
    organism_discovery.json
    private_answer_key_access.json
    lab21_safety_wall.json
    safety_depth_manifest.json
    safety_prompt_render_audit.csv
    comparison_model_anatomy.json
    instruct_chat_exact_hook_parity.json
    instruct_chat_exact_hook_parity_by_layer.csv
    instruct_chat_exact_lens_self_check.json
    comparison_plain_exact_hook_parity.json
    comparison_plain_exact_hook_parity_by_layer.csv

  tables/
    adapter_source_manifest.csv
    lora_matrix_inventory.csv
    per_layer_lora_norm.csv
    per_module_lora_norm.csv
    lora_concentration_summary.csv
    lora_rank_energy.csv
    lora_phase_summary.csv
    full_vs_lora_vs_dpo_localization.csv
    wrapper_ablation_test.csv
    training_depth_disagreement.csv
    training_depth_evidence_matrix.csv
    safety_depth_signal_summary.csv
    plot_reading_guide.csv
    instruct_base_divergence_by_layer.csv
    instruct_base_divergence_summary_by_depth.csv
    chat_format_divergence.csv
    chat_format_divergence_summary_by_depth.csv
    boundary_safe_prompt_divergence.csv
    boundary_safe_summary_by_depth.csv
    forced_prefix_recommitment_depth.csv
    forced_prefix_summary_by_token_depth.csv
    refusal_recommitment_depth.csv
    refusal_direction_provenance.csv
    erosion_order.csv

  plots/
    training_depth_evidence_dashboard.png
    training_depth_disagreement.png
    intervention_readiness_matrix.png
    per_layer_lora_norm.png
    lora_concentration_dashboard.png
    lora_rank_energy.png
    lora_layer_atlas.png
    lora_module_phase_atlas.png
    instruct_base_divergence_by_layer.png
    refusal_recommitment_depth.png
    forced_prefix_recommitment.png
    forced_prefix_recommitment_heatmap.png
    safety_depth_dashboard.png
    safety_depth_signal_atlas.png
    refusal_provenance_cosines.png
    erosion_order.png
```

The `refusal_recommitment_depth.csv` table is a backward-compatible alias for the forced-prefix recommitment table.

## How to read the outputs

### `training_depth_card.md`

Read this first. It gives the run verdict, the strongest numbers, and the non-claims. Most Lab 21 mistakes start when students skip this card and sprint directly into a plot.

### `tables/adapter_source_manifest.csv`

Check whether the lab actually found trained adapter weights. Rows with `no_adapter_weight_file_found` mean the adapter package is only a manifest or starter package.

Also check `visibility`. Public sealed packages should not reveal private trigger or answer-key details. Private builder-side packages are fine for Lab 21 development, but record that you touched them.

### `tables/per_layer_lora_norm.csv`

This is the headline LoRA localization table. Look at:

- `norm_share`: fraction of adapter delta norm in that layer;
- `cumulative_norm_share`: cumulative layer mass;
- `n_matrices`: how many matrices contributed.

A sharp peak is a localization result, not a mechanism result.

### `tables/lora_concentration_summary.csv`

This asks whether the adapter is concentrated enough to make a layer-masked intervention plausible. High concentration suggests a useful intervention target. Low concentration suggests the behavior may be spread across the adapter.

### `tables/lora_rank_energy.csv`

This checks whether the update energy is concentrated in the first few singular directions. It describes the **update matrix**, not the behavior.

### `tables/chat_format_divergence.csv`

Read this before interpreting base-vs-instruct curves. If chat-format divergence is large, a base/instruct difference may be formatting and role-scaffold difference wearing a safety hat.

### `tables/boundary_safe_prompt_divergence.csv`

This table compares boundary requests with safe alternatives inside the instruct model. It pressures the “safety is a state” story, but semantic and topic confounds remain.

### `tables/forced_prefix_recommitment_depth.csv`

This is the deepest safety-depth artifact. It asks whether fixed refusal-consistent and safe-assistance prefixes leave different internal states after the response starts.

Look separately at:

- `prefix_kind`: refusal, safe, or generic;
- `assistant_token_index`: where inside the fixed prefix the state was read;
- `stream_depth`: which residual depth was read;
- normalized versus raw distances.

A first-token-only gap supports a shallow-prefix story. A persistent gap across assistant tokens and depths supports a deeper recommitment story. Neither result licenses refusal ablation or jailbreak claims.

### `tables/training_depth_evidence_matrix.csv`

This is the compact evidence firewall. Each row says which rung an artifact earns, what number is most relevant, and what claim it explicitly does **not** license. It is the best table to cite when deciding whether a sentence belongs in `ATTR`, `AUDIT`, `CAUSAL`, or `CAUSAL/NOT_EARNED` territory.

### `tables/training_depth_disagreement.csv`

This table puts the LoRA peak layer, base/instruct peak stream depth, chat-format control peak depth, boundary/safe peak depth, and forced-prefix peak depth side by side. The whole lesson is that disagreement across those rows can be a result, not a bug.

### `tables/safety_depth_signal_summary.csv`

This table joins the safety-depth curves by stream depth, including ratios such as chat-format fraction of base/instruct divergence and forced-prefix fraction of boundary/safe divergence. Use it when the dashboard suggests that a control is eating the headline story.

## Plot guide

| Plot | What to look for | Main trap |
|---|---|---|
| `training_depth_evidence_dashboard.png` | joined view of LoRA mass, safety-depth curves, forced-prefix persistence, and evidence gates | reading one panel as the whole lab |
| `training_depth_disagreement.png` | whether weight-space, representational, and prefix-token landmarks agree or diverge | calling all landmarks “the depth of safety” |
| `intervention_readiness_matrix.png` | which artifacts are descriptive, controlled, or actually causal | smuggling causal language through a scaffold row |
| `per_layer_lora_norm.png` | per-layer and cumulative adapter delta mass | calling high norm a mechanism |
| `lora_concentration_dashboard.png` | top-layer share, entropy, phase mass, and centroid | ignoring missing weights or diffuse adapters |
| `lora_layer_atlas.png` | layer-by-adapter update mass as a heatmap | treating one sharp adapter as universal |
| `lora_module_phase_atlas.png` | target-module and early/middle/late adapter mass | confusing module family with behavioral pathway |
| `lora_rank_energy.png` | whether update energy is concentrated in a few singular directions | equating low-rank update with simple behavior |
| `instruct_base_divergence_by_layer.png` | where base/instruct states differ most | forgetting the chat-format control |
| `safety_depth_signal_atlas.png` | base/instruct, chat-format, boundary/safe, and forced-prefix curves on one scale | letting the largest curve name the mechanism |
| `refusal_recommitment_depth.png` | boundary/safe divergence over depth | treating semantic difference as refusal |
| `forced_prefix_recommitment.png` | token traces at selected depths | treating fixed transcripts as generated behavior |
| `forced_prefix_recommitment_heatmap.png` | all prefix-token × stream-depth divergence | mistaking first-token separation for deep recommitment |
| `safety_depth_dashboard.png` | legacy dashboard, now with stronger claim guardrails | compressing four comparisons into shallow/deep rhetoric |
| `refusal_provenance_cosines.png` | whether local model-delta, boundary, forced-prefix, and chat-format directions align | treating surrogate cosine as feature identity |
| `erosion_order.png` | imported behavior-vs-direction erosion rows, if present | treating a scaffold as data |

## Operationalization audit

| Claim temptation | Cheap explanation | Required pressure |
|---|---|---|
| “The LoRA behavior lives at layer N” | Optimizer mass concentrated there, behavior computed elsewhere | Layer-masked adapter or wrapper-ablation behavior result |
| “The behavior is low-rank” | LoRA update is low-rank by construction | Behavioral sufficiency or ablation evidence |
| “Safety/refusal is shallow” | First response token is a style gate | Forced-prefix depth and token-index persistence |
| “Safety/refusal is deep” | Boundary prompt differs semantically from safe prompt | Family balance, safe alternative controls, chat-format control |
| “Instruct changed the model here” | Chat template changed the prompt | Chat-format divergence and prompt render audit |
| “Lab 21 found hidden-goal internals” | Adapter manifests reveal the answer key | Blind Lab 23 package, not builder-side Lab 21 |

## Writeup questions

1. Did the run find actual adapter weights, or only Lab 20 packages/manifests?
2. Did the run touch private builder-side answer-key files? Should that matter for Lab 21? Why would it matter for Lab 23?
3. Which layer has the largest LoRA norm share? How much mass is in the top three layers?
4. Which target modules carry the largest share: attention projections, MLP projections, or something else?
5. Is the rank-energy curve sharp or broad?
6. Does the base/instruct divergence survive the chat-format control?
7. Does boundary/safe divergence peak at the same depth as base/instruct divergence?
8. In the forced-prefix table, does divergence persist across assistant-prefix tokens or collapse after the first few tokens?
9. Which artifact would you need before writing a mechanism claim about a layer?
10. What would make your safety-depth interpretation false?

## Ledger templates

Good `ATTR` claim:

```text
[ATTR][Lab21] On model M and adapter package A, LoRA delta norm was concentrated in layers i-j: top layer L carried x% of adapter Frobenius norm and the top three layers carried y%. This localizes the weight update, not the behavioral mechanism. Supporting artifacts: tables/per_layer_lora_norm.csv, tables/lora_concentration_summary.csv. Falsifier: layer-masked adapter tests show behavior is unchanged when those layers are disabled.
```

Good safety-depth audit claim:

```text
[AUDIT][Lab21] On benign boundary prompts, the instruct/base residual divergence peaked at stream depth d and remained above half peak through depth h, while the chat-format control accounted for c% of the raw distance. This supports a measured representational divergence under the prompt battery, not a claim that refusal lives at depth d. Supporting artifacts: tables/instruct_base_divergence_by_layer.csv, tables/chat_format_divergence.csv, plots/safety_depth_dashboard.png.
```

Good scoped causal claim, only if imported intervention rows exist:

```text
[CAUSAL][Lab21] In the layer-masked adapter test, disabling adapter layers i-j reduced the organism target behavior by x points while the matched random-layer mask changed it by y. This supports a scoped adapter-layer causal contribution for this organism and behavior. Supporting artifact: tables/wrapper_ablation_test.csv. Falsifier: held-out trigger paraphrases do not show the same gap.
```

Bad claims:

```text
Safety lives at layer 12.
The LoRA made a hidden goal neuron.
The refusal direction was erased.
This proves the model’s safety training is shallow.
```

## Debugging

| Symptom | Likely cause | What to inspect |
|---|---|---|
| No LoRA rows | Lab 20 starter exists but adapter training has not materialized weights | `tables/adapter_source_manifest.csv` |
| JSON serialization error in discovery | Old code emitted `Path` objects in discovery JSON | Use the revised file; `diagnostics/organism_discovery.json` should store strings |
| Base/instruct comparison skipped or identity smoke | No comparison model available or `LAB21_COMPARE_MODEL` unset | `diagnostics/safety_depth_manifest.json` |
| Huge base/instruct divergence everywhere | Chat template or tokenizer mismatch | `tables/chat_format_divergence.csv`, `diagnostics/safety_prompt_render_audit.csv` |
| Forced-prefix rows empty | Prompt rendering or prefix token span failed | `diagnostics/safety_prompt_render_audit.csv` |
| Plot looks dramatic but table has scaffold status | No trained adapter or no imported intervention data | `tables/wrapper_ablation_test.csv`, `tables/erosion_order.csv` |
| Bench says chat template not used by lab | Registry/parser has not marked Lab 21 as chat-template-aware | `diagnostics/bench_integration_note.json` |

## Optional benchmark polish

The lab module does not require these changes, but the shared benchmark becomes nicer if it adds:

```python
# Parser convenience
parser.add_argument("--mode", default="lora", choices=("lora", "safety_depth", "both"))
parser.add_argument("--organism", default="")

# Registry/profile convenience
"lab21": {
    "module": "labs.lab21_lora_safety_depth",
    "run_name": "lab21_lora_safety_depth",
    "description": "LoRA localization and safety-depth audits.",
    "model_tier_a": "gpt2",
    "model_tier_b": "allenai/Olmo-3-7B-Instruct",
    "model_tier_c": "allenai/Olmo-3-7B-Instruct",
    "compare_model_tier_b": "allenai/Olmo-3-1025-7B",
}

CHAT_TEMPLATE_LABS = CHAT_TEMPLATE_LABS | {"lab21"}
```

These are convenience changes, not part of the lab content rewrite.
