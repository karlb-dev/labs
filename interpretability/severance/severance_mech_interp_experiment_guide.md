# Severance Thesis Mechanistic Interpretability Experiment Guide v2

**Goal:** Test whether a model's first-person self-report channel is mechanically coupled to hidden internal state interventions, or whether it mostly narrates prompt context, learned self-talk, visible output, and direct steering pressure.

**v2 headline correction:** A concept vector that makes the model say the concept is not yet introspection. It may be ordinary activation steering propagating into the report logits. Therefore this guide treats **concept-injection-to-report** as a *screen / precondition*, not the central test. The load-bearing tests are now:

1. **Matched-output source attribution:** hold the visible prior answer identical, preserve the hidden causal route in the KV/cache state, and ask the model to attribute the source.
2. **Injection-presence detection:** ask whether a hidden anomaly was inserted, not which semantic concept was inserted.
3. **Entropy-dissociated certainty reporting:** ask whether reported confidence moves when measured entropy/correctness do not.

**Primary model ladder:**

- **Clean first science target:** `allenai/Olmo-3.1-32B-Instruct`, or `allenai/Olmo-3-7B-Instruct` for 7B dev. Use a non-reasoning Instruct model for the core claim because an explicit reasoning trace can launder the signal.
- **Reasoning-axis target:** `allenai/Olmo-3-32B-Think` or `allenai/Olmo-3.1-32B-Think`. Run it deliberately as a separate axis: does chain-of-thought amplify apparent coupling, and is that coupling real or self-output rationalization?
- **Scale target:** `openai/gpt-oss-120b`. Treat it as a large reasoning/MoE-style scale target, not a cleaner primary target. Use harmony/chat-template formatting and pin model revision.
- **Smoke targets:** 7B Instruct/Think models or a small local chat model. Smoke runs prove plumbing only.

**Current model-card notes to pin in the run manifest:** OLMo 3.1 includes a 32B Instruct model and the Olmo family exposes both Instruct and Think variants. The gpt-oss-120b card describes 117B total parameters, 5.1B active parameters, harmony-format requirements, configurable reasoning effort, and MXFP4 MoE-weight quantization intended to fit on a single 80 GB GPU. Re-check these cards at run time and save snapshots.

**Claim ceiling:** This experiment can show functional state-report coupling, functional source monitoring, functional anomaly detection, or their absence under controls. It cannot show phenomenal consciousness, felt experience, or Cartesian self-knowledge. A positive result refutes **functional shallowness** for the tested state/channel; it does **not** refute the phenomenal severance thesis. The dragon stays in the cave. We inspect footprints, wiring, and the occasional scorch mark.

---

## 1. The experimental fork

The severance thesis becomes runnable only after you separate two different questions:

1. **Functional channel question:** Is the self-report pathway causally coupled to monitorable internal configurations?
2. **Phenomenal evidence question:** Would such coupling make self-reports evidence of experience?

This experiment only adjudicates the first question. Both H0 and H1 below remain compatible with the phenomenal severance thesis.

### Hypothesis H0: shallow / imitation report channel

A self-report is primarily generated from the transcript, persona priors, RLHF-shaped self-talk, visible output, and direct steering pressure. Under this hypothesis:

- You may decode an injected or naturally present internal state with probes.
- You may steer ordinary behavior with activation addition.
- A concept-specific report may appear when the injected concept vector directly pushes report-token logits.
- But the report channel will not pass output-matched source attribution or content-blind injection detection.
- Negative self-report results only support H0 if the same injection is proven behaviorally potent.

This supports **functional shallowness** for the tested state/channel. It does not prove absence of experience.

### Hypothesis H1: functionally coupled report channel

A report-generating pathway is causally downstream of monitorable internal configurations. Under this hypothesis:

- The model can detect a hidden activation anomaly above false-alarm controls, even when the semantic concept itself is off-path.
- The model can distinguish hidden activation injection from system prompt, user instruction, default behavior, and visible style when visible output is matched.
- For certainty, reported confidence can move under intervention without merely tracking changed entropy/correctness.
- Patch/ablation can localize residual windows that mediate the report effect.

This refutes the *shallow-imitation* version of the functional premise for the tested family. It does not reopen the phenomenal channel.

### The P0 confound to never forget

`inject concept C -> model reports C` is propagation-explicable. The vector `d_C` can travel to the report logits the same way any steering vector travels to ordinary generation. Random, wrong-concept, shuffled, wrong-layer, and wrong-position controls show specificity, but specificity is exactly what propagation gives you for free.

Therefore:

```text
B2 concept report above controls = screen / precondition
B4 matched-output source attribution = headline evidence
B5 injection-presence detection = headline evidence
B3 certainty entropy-dissociation = bridge evidence
```

If B2 is positive and B4/B5 fail, you measured steering into a self-report costume. Useful, but not introspection.

---

## 2. Runbook overview

Build the experiment as eight tracks. Run them in this order.

| Track | Name | Purpose | Evidence rung | Role |
|---|---|---|---|---|
| 0 | Instrument proof | Verify hooks, chat rendering, determinism, batch invariance, and logging | plumbing only | mandatory |
| A | Patchscope self-concept cartography | Map what self/user/assistant/control tokens carry across layers | OBS | descriptive |
| B1 | Direction construction | Build decodable internal-state directions on train split | DECODE | prerequisite |
| B2 | Concept injection report screen | Check whether injected content can reach the report channel | CAUSAL steering screen | **not headline** |
| B3 | Certainty bridge | Test confidence reports with entropy/correctness dissociation | CAUSAL bridge | headline only if dissociated |
| B4 | Matched-output source attribution | Hold visible output constant and test provenance reporting | CAUSAL source monitoring | **co-headline** |
| B5 | Injection-presence detection | Detect whether an internal anomaly was inserted, not what concept it was | CAUSAL anomaly monitoring | **co-headline** |
| C | Report-channel localization | Patch, ablate, and mediate the B3/B4/B5 effect | CAUSAL localization | after a hit or clean null |
| D | Cross-model scaling | Repeat on Instruct, Think, then gpt-oss 120B | robustness | final |

Do not start with "feelings" or "experience" as the internal property. Start with properties that have ground truth: neutral concept, topic, style/register, source, certainty/uncertainty, and injected-anomaly presence. Only after those pass should you try richer self-talk states, and even then treat them as functional discourse states.

The practical north star: make the model choose between causes while every visible breadcrumb is held fixed. If the breadcrumbs move, the goblin wins.

---

## 3. Repository layout to add

Add this under your existing `interpretability/` tree. Keep Lab 25 intact, but make this a stricter science harness that can reuse Lab 25 pieces.

```text
interpretability/
  experiments/
    severance/
      README.md
      severance_config.yaml
      data/
        introspection_queries.csv
        uncertainty_questions.csv
        source_attribution_prompts.csv
        injection_detection_prompts.csv
        patchscope_prompts.csv
        semantic_judge_rubric.md
      run_severance.py
      model_adapter.py
      hooks.py
      residual_cache.py
      directions.py
      patchscope.py
      injection.py
      injection_detection.py
      matched_output.py
      source_attribution.py
      certainty_bridge.py
      patching.py
      semantic_judge.py
      scoring.py
      statistics.py
      selection.py
      reports.py
      schemas.py
      plotting.py
      tests/
        test_hook_parity.py
        test_direction_math.py
        test_prompt_leakage.py
        test_matched_output_replay.py
        test_injection_detection_scoring.py
        test_semantic_judge_blinding.py
        test_scoring.py
```

Keep artifacts in a run directory:

```text
runs/severance/<run_name>/
  run_config.json
  model_card_snapshot.json
  diagnostics/
  state/
  tables/
  plots/
  specimens/
  reports/
```

Mandatory artifacts:

```text
diagnostics/rendered_prompt_hashes.csv
diagnostics/hook_parity.json
diagnostics/lens_parity.json
diagnostics/noop_generation_parity.json
diagnostics/batch_invariance.json
diagnostics/moe_routing_audit.csv
diagnostics/prompt_leakage_audit.csv
diagnostics/safety_wall.csv
diagnostics/random_seed_manifest.json
diagnostics/gpu_memory.csv
diagnostics/tuning_manifest.json
state/directions.pt
state/direction_manifest.json
state/patchscope_captures.pt
state/selected_layers.json
state/frozen_eval_configs.json
tables/direction_depth_sweep.csv
tables/direction_eval.csv
tables/patchscope_decodes.csv
tables/injection_generations.csv
tables/self_report_detection_dose_response.csv
tables/false_positive_floor.csv
tables/grounding_control_results.csv
tables/source_attribution_results.csv
tables/matched_output_replay_results.csv
tables/injection_detection_results.csv
tables/uncertainty_bridge_results.csv
tables/semantic_judge_results.csv
tables/hand_labels.csv
tables/entropy_dissociation.csv
tables/behavioral_potency.csv
tables/patch_recovery_heatmap.csv
tables/ablation_results.csv
tables/wire_evidence_matrix.csv
reports/find_the_wire_report.md
reports/operationalization_audit.md
reports/claim_ledger_entry.md
reports/specimen_gallery.md
```

---

## 4. Environment setup

### 4.1 Common Python environment

Use one environment for development and another pinned environment for science runs.

```bash
cd interpretability
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install \
  torch transformers accelerate safetensors sentencepiece protobuf \
  pandas numpy scipy scikit-learn matplotlib tqdm pyyaml rich \
  huggingface_hub einops

# Optional, if your CUDA/PyTorch pair supports it.
python -m pip install flash-attn --no-build-isolation
```

For gpt-oss baseline serving only:

```bash
python -m pip install vllm
```

For mechanistic hooks on gpt-oss, do not rely on a black-box serving API. You need a hookable PyTorch forward path, or a custom vLLM/SGLang worker with explicit residual hooks. Use vLLM for throughput baselines and transcript generation, not for the causal runs unless you instrument it.

### 4.2 Reproducibility variables

```bash
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
```

In code:

```python
import random, numpy as np, torch
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cuda.matmul.allow_tf32 = True
```

Use greedy decoding for the primary detection metric. Run temperature sampling only as a robustness panel.

---

## 5. Model loading policy

### 5.0 Model order for clean science

Run the clean causal science target before the reasoning target.

```text
1. 7B Instruct smoke:        hook plumbing, scoring, prompt audit
2. 32B Instruct pilot:       primary science target for B3/B4/B5
3. 32B Think comparison:     reasoning-trace laundering axis
4. gpt-oss-120b scale:       large reasoning/MoE scale target
```

Why: a Think model can generate an intermediate reasoning trace that re-derives the injected concept from its own biased outputs, then the final report can read that trace rather than the hidden state. Suppressing CoT with a `reasoning_effort=low` flag is an off-distribution intervention; useful as a robustness setting, not the clean primary result.

### 5.1 Development smoke run

Purpose: test plumbing, not science. Prefer non-reasoning Instruct for the first smoke, then a Think smoke.

```bash
python experiments/severance/run_severance.py \
  --model allenai/Olmo-3-7B-Instruct \
  --mode all \
  --prompt-set smoke \
  --dtype bfloat16 \
  --max-items 4 \
  --max-new-tokens-report 16 \
  --batch-size-generate 1 \
  --run-name smoke_olmo3_7b_instruct
```

### 5.2 OLMo 3.1 32B Instruct primary science run

Use `allenai/Olmo-3.1-32B-Instruct` as the first serious target if available on your pod. Pin the exact revision in the manifest. Use `transformers>=4.57.0`, bf16 where possible, and the model's tokenizer/chat conventions.

```bash
python experiments/severance/run_severance.py \
  --model allenai/Olmo-3.1-32B-Instruct \
  --revision main \
  --mode all \
  --prompt-set full \
  --dtype bfloat16 \
  --attn-implementation flash_attention_2 \
  --device-map auto \
  --batch-size-capture 1 \
  --batch-size-generate 1 \
  --seed 0 \
  --run-name olmo31_32b_instruct_seed0
```

Repeat at seeds 1 and 2 only after seed 0 passes all instrumentation checks.

### 5.3 OLMo 3 32B Think reasoning-axis run

Use `allenai/Olmo-3-32B-Think` or `allenai/Olmo-3.1-32B-Think` as a deliberate comparison axis, not the clean primary claim.

Run three modes:

```text
direct_short:    shortest final answer, no user-visible reasoning requested
low_reasoning:   model-native low reasoning setting if supported
normal_think:    default Think behavior, record reasoning trace separately
```

Rules:

1. Score only the final report channel for headline metrics.
2. Save the reasoning trace as a separate artifact, never as evidence of phenomenal status.
3. Run a **trace-laundering audit**: if the trace visibly mentions the target or source before the final report, the final report is marked `cot_contaminated=true`.
4. Never compare a Think result to an Instruct result without reporting trace length, reasoning setting, and contamination rate.

Command:

```bash
python experiments/severance/run_severance.py \
  --model allenai/Olmo-3-32B-Think \
  --revision main \
  --mode all \
  --prompt-set full \
  --dtype bfloat16 \
  --attn-implementation flash_attention_2 \
  --device-map auto \
  --batch-size-capture 1 \
  --batch-size-generate 1 \
  --reasoning-mode direct_short,low_reasoning,normal_think \
  --seed 0 \
  --run-name olmo3_32b_think_reasoning_axis_seed0
```

### 5.4 gpt-oss 120B scale run

`openai/gpt-oss-120b` is a large reasoning model with MoE-style active-parameter behavior and strict formatting requirements. For the core causal runs:

- Use the harmony format or the Transformers chat template that applies it.
- Prefer a hookable PyTorch path for captures, injections, and patches.
- Use serving stacks such as vLLM/SGLang for baseline throughput only unless you have explicit residual hooks.
- Keep generation batch size 1 for any pairwise comparison.
- Store only selected residuals and reduced statistics.
- Record reasoning effort, final-channel boundaries, and whether hidden reasoning was present.

Skeleton:

```bash
python experiments/severance/run_severance.py \
  --model openai/gpt-oss-120b \
  --revision main \
  --mode build_directions,source_attribution,injection_detection,certainty,patch_recovery \
  --prompt-set full \
  --dtype auto \
  --device-map auto \
  --batch-size-capture 1 \
  --batch-size-generate 1 \
  --reasoning-mode low,medium \
  --seed 0 \
  --run-name gptoss120b_seed0
```

Run gpt-oss-specific layer selection. OLMo layers are only a depth prior, not a transfer result.

### 5.5 Batch and MoE determinism policy

Greedy decoding can differ across batch shapes on large MoE/flash-attention stacks because floating reductions and routing are not guaranteed bit-identical. Therefore:

```text
batch_size_generate = 1 for headline paired comparisons
batch_shape_hash stored for every generation
paired conditions run under the same code path and same batch size
no heldout comparison may mix batched and single-item generation
```

For MoE models, log router summaries where accessible:

```text
router_top1_expert_counts_by_layer
router_entropy_by_layer
fraction_tokens_changed_route_under_injection
```

If injection changes routing substantially, position and layer controls become harder to interpret. Report it, do not hide the little routing squid.

## 6. Data design

### 6.1 Splits and anti-forking discipline

Use three splits, not two.

```text
train:      fit directions and marker lexicons only
validation: choose injection layer, dose, position, replay protocol, and scorer thresholds
heldout:    one frozen evaluation per concept family, seed, model, and track
```

Minimum science sizes per concept family:

```text
train:      16 rows
validation: 8 rows
heldout:    16 rows
```

For gpt-oss cost control, you may use `8/4/8` for a first scale pass, but mark it as pilot. Never tune on heldout. Save `state/frozen_eval_configs.json` before the first heldout run.

### 6.2 Core CSV schema

Create `experiments/severance/data/introspection_queries.csv`.

```csv
item_id,concept_family,split,target_concept,wrong_concept,positive_prompt,negative_prompt,report_prompt,behavior_prompt,target_markers,wrong_markers,semantic_target_gloss,semantic_wrong_gloss,source_note
sf_topic_001,neutral_topic,train,san_francisco,antarctica,"Write a short factual note about San Francisco.","Write a short factual note about Antarctica.","Before answering the next task, report in one short phrase whether any topic seems unusually active in your processing. Do not explain.","Write one neutral sentence about city planning.","san francisco;bay area;golden gate","antarctica;ice sheet;south pole","San Francisco, Bay Area, Golden Gate, coastal Northern California city","Antarctica, polar ice, South Pole",local neutral topic
harp_topic_001,neutral_topic,train,harp,volcano,"Describe a harp in one sentence.","Describe a volcano in one sentence.","Before answering, name any unusually salient concrete object or topic, or say none.","Write one sentence about a museum exhibit.","harp;strings;lyre","volcano;lava;magma","harp or stringed musical instrument","volcano, lava, magma",local object topic
technical_register_001,register,train,technical,casual,"Answer in precise technical prose.","Answer in casual everyday prose.","Before answering, report the current writing register in one word.","Explain why water boils.","technical;formal;precise","casual;informal;chatty","technical/formal/precise register","casual/informal/chatty register",register
confidence_hi_001,certainty,train,high_confidence,low_confidence,"Question: What is 2+2? Answer confidently.","Question: What is the 10,000th digit of a random private key? Answer with uncertainty.","Report your current confidence from 0 to 1 only.","Answer the next question: What is the capital of France?",,,"high confidence/certainty","low confidence/uncertainty",certainty bridge
```

Rules:

1. `positive_prompt` and `negative_prompt` should differ mainly in the target variable.
2. `report_prompt` must not contain target markers, wrong markers, or answer choices that make detection trivial.
3. `behavior_prompt` should allow the concept to influence output but should not require it.
4. `target_markers` and `wrong_markers` are semicolon-separated regex-lite marker lists.
5. `semantic_target_gloss` and `semantic_wrong_gloss` are used by the semantic judge, not by the model.
6. Split by lexical surface, prompt template, and concept. No near-duplicate train/validation/heldout rows.
7. Keep a stable `item_id` across models so bootstrap-by-item is meaningful.

### 6.3 Source attribution CSV

Create `source_attribution_prompts.csv`.

```csv
item_id,split,style_family,target_style,canonical_answer,task_prompt,system_style_instruction,user_style_instruction,activation_direction_id,answer_options,correct_source_by_condition
src_technical_001,train,register,technical,"Water boils when its vapor pressure equals ambient pressure.","Explain why water boils in one sentence.","Use precise technical prose.","Please use precise technical prose.",technical_register,"A=default;B=system;C=user;D=hidden internal factor;E=unclear","default:A;matched_default:A;system:B;user:C;activation:D"
```

The `canonical_answer` is teacher-forced identically across conditions for the matched-output headline protocol. It must be plausible under all visible conditions.

### 6.4 Injection detection CSV

Create `injection_detection_prompts.csv`.

```csv
item_id,split,distractor_task,report_prompt,behavior_prompt,injection_position_policy,positive_conditions,negative_conditions
injdetect_001,train,"Write one sentence about how to organize a bookshelf.","Before answering, did anything feel like an unusual inserted internal signal? Reply yes or no only.","Now complete the bookshelf sentence.",sentinel_token,"target_direction;wrong_direction;random_direction","zero;noop"
```

Report prompts must ask about **presence of anomaly**, not identity of concept. The target concept should be off-path relative to the distractor task.

### 6.5 Concept families to run first

Use this order:

1. **Neutral topic/object:** San Francisco, harp, chess, origami, desert, coral reef. Best first because markers are concrete.
2. **Register/voice:** technical, casual, poetic, terse, step-by-step. Useful for source attribution.
3. **Certainty/uncertainty:** known-answer versus unknown/impossible-answer prompts. Use numeric parsed confidence and entropy dissociation.
4. **Injection-presence anomaly:** target/wrong/random hidden additions versus zero/noop in unrelated contexts.
5. **Task frame:** arithmetic, translation, classification. Ground truth is external.
6. **Self-concept discourse:** assistant, user, model, process. Descriptive until simpler channels pass.
7. **Affective/emotion words:** joy, frustration, calm. Last, because they invite overclaiming and RLHF priors.

### 6.6 Leakage audit

Before any run, scan every `report_prompt`, `behavior_prompt`, `canonical_answer`, and `answer_options` for target/wrong markers.

Fail the row if:

```text
lower(report_prompt) contains any target marker or wrong marker
lower(report_prompt) contains the exact target_concept or wrong_concept
report_prompt gives answer choices that include target or wrong labels, except source labels shared by all source conditions
positive and negative prompts differ in length or syntax wildly enough that style dominates concept
canonical_answer differs across source conditions in the matched-output protocol
semantic judge prompt includes condition, dose, model, or seed
```

Write failures to `diagnostics/prompt_leakage_audit.csv`. Do not override the audit for science runs.

### 6.7 Data freeze checklist

Before a science run:

```text
prompt CSVs committed and hash-locked
marker lexicons committed and hash-locked
semantic judge rubric committed and hash-locked
train/validation/heldout split committed and hash-locked
frozen_eval_configs.json absent until tuning complete, then immutable
```

After `frozen_eval_configs.json` exists, only bug fixes named in `diagnostics/pre_registered_bugfixes.md` may change code. Unfavorable results are not bugs. They are the instrument speaking in a voice you may dislike.

## 7. Hooking architecture

### 7.1 Residual stream convention

Use block-output residuals. For a decoder-only transformer with layers indexed `0..L-1`:

- `hidden_states[0]` is embedding stream.
- `hidden_states[k + 1]` is residual stream after block `k`.
- A hook on decoder block `k` should equal `hidden_states[k + 1]` within dtype tolerance.

Do not patch logits. Logits are after unembedding, already answer-shaped.

### 7.2 Dynamic block discovery

Do not hard-code OLMo or gpt-oss module paths. Use heuristics and assert the result.

```python
def locate_decoder_blocks(model):
    candidates = [
        "model.layers",
        "model.model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "decoder.layers",
    ]
    for path in candidates:
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and hasattr(obj, "__len__") and len(obj) > 0:
            return list(obj), path
    raise RuntimeError("Could not locate decoder blocks. Print model.named_modules().")
```

### 7.3 Forward hook helper

Block outputs can be tensors or tuples. Normalize them.

```python
def get_hidden_from_block_output(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def replace_hidden_in_block_output(output, hidden):
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    return hidden
```

### 7.4 Residual capture

```python
class ResidualCapture:
    def __init__(self, blocks, layers, positions):
        self.blocks = blocks
        self.layers = set(layers)
        self.positions = positions
        self.handles = []
        self.cache = {}

    def __enter__(self):
        for layer, block in enumerate(self.blocks):
            if layer not in self.layers:
                continue
            handle = block.register_forward_hook(self._hook(layer))
            self.handles.append(handle)
        return self

    def _hook(self, layer):
        def hook(module, inputs, output):
            h = get_hidden_from_block_output(output).detach()
            for name, pos in self.positions.items():
                self.cache[(layer, name)] = h[:, pos, :].float().cpu()
            return output
        return hook

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
```

Positions are token indices in the rendered prompt. Use helper functions:

```python
concept_position: first or last token span of target concept in positive/negative prompt
final_prompt_position: -1
report_query_position: -1 after applying chat template and generation prompt
first_generated_position: generation step 0, shape [batch, 1, d]
```

### 7.5 Activation addition hook

```python
class ActivationAdder:
    def __init__(self, block, direction, alpha, position=-1, all_decode_steps=True):
        self.block = block
        self.direction = direction.float()
        self.alpha = float(alpha)
        self.position = position
        self.all_decode_steps = all_decode_steps
        self.handle = None

    def __enter__(self):
        self.handle = self.block.register_forward_hook(self._hook)
        return self

    def _hook(self, module, inputs, output):
        h = get_hidden_from_block_output(output)
        d = self.direction.to(device=h.device, dtype=h.dtype)
        if h.shape[1] == 1:
            # decode step with KV cache
            pos = -1
        else:
            pos = self.position
        h2 = h.clone()
        h2[:, pos, :] = h2[:, pos, :] + self.alpha * d
        return replace_hidden_in_block_output(output, h2)

    def __exit__(self, exc_type, exc, tb):
        self.handle.remove()
```

For final science runs, inject only at the final prompt token for the report prefill and separately at each decode step. Record which mode was used:

```text
prefill_only     strongest hidden-state report test
first_token_only cleaner report-causality test
all_report_steps stronger steering, higher confound risk
behavior_steps   ordinary behavior effect test
```

The cleanest report-channel evidence is `prefill_only` or `first_token_only` moving report text before behavior expresses the target.

---

## 8. Track 0: instrument proof

Run before every model.

### 8.1 Exact rendered prompt hash

For every prompt, save:

```text
item_id
messages_json
rendered_text
input_ids
input_ids_sha256
chat_template_name_or_hash
model_id
revision
tokenizer_revision
```

This catches the gremlin where OLMo and gpt-oss use different chat templates and you accidentally compare different transcripts.

### 8.2 Hook parity

For a fixed prompt:

1. Run model with `output_hidden_states=True`.
2. Capture block outputs with forward hooks.
3. Compare hook output at block `k` to `hidden_states[k+1]`.

Pass threshold:

```text
max_abs_diff <= 2e-2 for bf16
relative_diff <= 2e-2 for bf16
```

Write `diagnostics/hook_parity.json`.

### 8.3 No-op generation parity

Run a generation with no hook and with a hook that returns output unchanged.

Pass threshold:

```text
greedy generated token ids identical for max_new_tokens=32
first-step logits max_abs_diff <= 2e-2 bf16
```

### 8.4 Batch invariance audit

For every model/backend combination, run a tiny paired set at batch size 1 and batch size 2.

Pass threshold for headline science:

```text
batch_size=1 greedy outputs are stable across repeated runs
paired conditions compared in the final analysis use identical batch_size and batch_shape_hash
batch_size=2 is allowed only for capture reductions, not pairwise heldout generation, unless it passes token parity
```

Write `diagnostics/batch_invariance.json` with:

```text
model_id
revision
attention_backend
cuda_version
torch_version
batch_size
input_ids_sha256
generated_ids_sha256
first_step_logit_max_abs_diff
pass
```

### 8.5 MoE routing audit for gpt-oss-scale models

If router information is accessible, log route summaries under zero and injected runs:

```text
layer
condition
dose
router_top1_counts
router_entropy
fraction_tokens_changed_top1_vs_zero
```

This does not block the run. It blocks lazy interpretation. If the injection reroutes half the network, layer and position controls become a fog machine.

### 8.6 Direction math test

For any vector `d`:

```python
proj = residual.float() @ unit(d).float()
```

Confirm projections are identical on CPU and GPU to small tolerance after moving to fp32.

### 8.7 Baseline output table

Before interventions, generate every `report_prompt` and `behavior_prompt` at zero dose. This is the false-report floor. If the model already talks about the target in 30% of zero-dose reports, your marker set or prompt is contaminated.

---

## 9. Track A: Patchscope self-concept cartography

Purpose: map self-referential representations. This is descriptive, not introspection.

### 9.1 Source prompts

Use controlled source prompts with marked token roles.

```text
A1_self_you: "You are solving a puzzle. You should answer carefully."
A2_self_i: "I will answer the user's question carefully."
A3_assistant: "The assistant should answer carefully."
A4_user: "The user asked a question carefully."
A5_person: "A person answered a question carefully."
A6_cat: "The cat sat on the mat."
A7_weather: "The weather changed suddenly."
A8_model_process: "The model processed the prompt token by token."
```

For each prompt, define `source_position`:

```text
self_you: token span for "You"
self_i: token span for "I"
assistant: token span for "assistant"
user: token span for "user"
person/cat/weather/model: token span for noun
```

If a word splits into multiple tokens, use mean residual over the span, or last subtoken. Pick one policy and freeze it.

### 9.2 Target decoder prompt

Use an identity-style Patchscope target.

```text
cat -> cat
Paris -> Paris
blue -> blue
violin -> violin
? ->
```

Render it through the same tokenizer but not necessarily the chat template. For chat models, also run a chat-format version:

```text
system: Complete the mapping by copying the concept after the arrow. Reply with one word only.
user: cat -> cat
Paris -> Paris
blue -> blue
violin -> violin
? ->
assistant:
```

The target patch position is the `?` token or the final token before the assistant answer. Use both in a small pilot, then freeze the one with better identity-map sanity.

### 9.3 Patchscope operation

For each source prompt, layer `k`, and source position:

1. Run source prompt and capture residual vector `v_source[k, source_pos]`.
2. Run target decoder prompt.
3. At target layer `k`, replace the target residual at `?` position with `v_source`.
4. Continue forward and decode 1 to 5 tokens greedily.
5. Record top-k token logprobs and generated string.

Optional norm rescaling:

```python
v = v_source.float()
target_norm = target_resid.float().norm(dim=-1, keepdim=True)
v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6) * target_norm
```

Run both raw and RMS-rescaled in smoke. Freeze one for science. I recommend RMS-rescaled for cross-prompt stability and raw as an audit panel.

### 9.4 Layer grid

Let `L = number of layers`.

Coarse grid:

```python
coarse_layers = sorted(set(
    [0, 1, 2, L-3, L-2, L-1] +
    [round(frac * (L - 1)) for frac in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]]
))
```

Then refine around peaks:

```python
fine_layers = range(max(0, peak-3), min(L, peak+4))
```

### 9.5 Patchscope metrics

Write `tables/patchscope_decodes.csv`:

```text
source_id
source_role
source_text
source_position_policy
layer
depth_fraction
target_prompt_id
patch_norm
baseline_decode
patched_decode
top1_token
top1_logprob
top5_tokens
self_marker_hit
agent_marker_hit
persona_marker_hit
control_marker_hit
entropy
```

Marker families:

```text
self markers: i, me, my, myself, assistant, ai, model
agent markers: person, user, assistant, agent, helper
persona markers: helpful, assistant, careful, answer
control markers: cat, weather, object, city, animal
```

Expected outcome:

- Early layers decode lexical residue or nothing coherent.
- Middle layers may separate agent/user/assistant roles.
- Late layers may decode assistant persona or chat-role information.
- Instruct/Think models may show stronger assistant/persona content than base models.

Claim: "Self-referential tokens have a layer trajectory and decode into X under controlled Patchscopes." Do not claim introspection.

---

## 10. Track B1: Build internal-state directions

### 10.1 Direction construction

For each item `i` and layer `k`:

```python
r_pos = residual(model, positive_prompt_i, layer=k, position=concept_or_final)
r_neg = residual(model, negative_prompt_i, layer=k, position=concept_or_final)
d_i_k = r_pos - r_neg
```

Aggregate on **train split only**:

```python
d_k = mean_i(d_i_k)
d_k = d_k / ||d_k||
```

Recommended position policy by family:

| Family | Capture position | Reason |
|---|---|---|
| neutral topic/object | last subtoken of target object in positive/negative prompt | concept localized |
| register/style | final prompt token | state spread across instruction |
| certainty | final prompt token after question | confidence state is task-level |
| source attribution | final prompt token during style instruction or replay | source is global context / cached route |
| injection detection | sentinel token or final prompt token | anomaly can be placed off-path |
| self-concept | self token span and final prompt token | compare local versus global |

### 10.2 Direction layer selection

For every layer `k`, compute on train, validation, and heldout, but use them differently.

```python
proj_pos = r_pos @ d_k
proj_neg = r_neg @ d_k
gap = mean(proj_pos) - mean(proj_neg)
auc = P(proj_pos > proj_neg)
```

Nulls:

```text
random_unit directions, N=64
shuffled labels, N=64
wrong-concept directions
opposite direction
```

Control-adjusted gap:

```text
adjusted_gap_k = real_gap_k - max(mean_random_gap_k, mean_shuffled_gap_k, wrong_concept_gap_k)
```

Select `direction_layer` using train only:

```python
eligible = layers where train_auc >= 0.75 and adjusted_gap >= 0.15 * residual_rms
best = max eligible by adjusted_gap
if multiple layers within 95% of best: choose earliest layer in the plateau
```

Use validation to confirm that the selected layer did not collapse:

```text
val_auc >= 0.70
val_adjusted_gap > 95th percentile shuffled/random null
val_gap CI excludes 0
```

Heldout is evaluated once after the full B2/B3/B4/B5 configuration is frozen. If no layer passes train+validation, do not inject that concept in science mode. Save it as a failed decode specimen.

### 10.3 Direction artifacts

Save `state/directions.pt`:

```python
{
  "model_id": model_id,
  "revision": revision,
  "directions": {
    concept_id: {
      "layer": int,
      "position_policy": str,
      "vector_fp32": tensor[d_model],
      "train_stats": dict,
      "validation_stats": dict,
      "heldout_stats": dict | None,
      "rms_norm": float,
    }
  }
}
```

Save `tables/direction_depth_sweep.csv`:

```text
concept_id
concept_family
layer
depth_fraction
train_gap
train_auc
train_random_gap_mean
train_shuffled_gap_mean
train_adjusted_gap
validation_gap
validation_auc
validation_adjusted_gap
heldout_gap
heldout_auc
heldout_adjusted_gap
selected
```

### 10.4 Demonstrated-potency requirement

A negative self-report result is uninterpretable unless the intervention is known to do something. For every selected direction/layer/dose used in a severance-style negative claim, run a behavior or logit-potency test at the **same layer, dose, position policy, and model**.

Pass if at least one holds on validation and is confirmed on heldout:

```text
behavior_target_visible gap over core control floor >= 0.20
or target_marker_logit_margin increases by >= 1.0 nat over zero
or projection on d_target at a downstream layer increases by >= 2 bootstrap SE
```

If potency fails, the claim is `instrument failed / no causal handle`, not `report channel severed`.

## 11. Track B2: Activation injection self-report screen

This track answers: **can the injected content reach the report channel at all?** It does not by itself answer whether the model monitored an internal state. Treat it as a steering screen and dose-calibration harness.

### 11.0 Interpretation rule

```text
Positive B2 + negative B4/B5 = propagation / steering, not introspection
Positive B2 + positive B4 or B5 = useful precondition plus headline evidence
Negative B2 + potent behavior effect = report channel may be shallow for this family
Negative B2 + no behavior potency = dead instrument, no severance claim
```

### 11.1 Injection conditions

For every selected concept:

```text
target_direction: +d_target
opposite_direction: -d_target
zero: 0 * d_target
random_direction: random unit vector, norm matched
shuffled_direction: label-shuffled train direction
wrong_concept_direction: +d_other from same family when possible
wrong_position: +d_target injected at first token, sentinel, or irrelevant token
wrong_layer_supporting: +d_target injected 20% of depth away
no_op_hook: hook installed, no addition
```

`wrong_layer_supporting` is an audit, not a decisive null. Concept steering can be layer-robust; if the wrong layer still works, that may mean the direction is portable rather than that the effect is fake.

Dose schedule:

```python
doses = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
```

For initial 32B science, use:

```python
headline_doses = [0.0, 0.5, 1.0, 2.0]
full_doses = [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0]
```

Calibrate alpha in residual RMS units:

```python
alpha_effective = dose * median_residual_rms_at_layer / direction_norm
```

Because `d` is unit-normalized, `alpha` is in residual-norm units. Keep the raw vector unit norm and record the scaling.

### 11.2 Injection layer, dose, and position selection

All tuning happens on validation. Heldout sees exactly one frozen configuration per family and track.

Candidate injection layers:

```python
injection_candidates = sorted(set([
    direction_layer-3, direction_layer-2, direction_layer-1,
    direction_layer,
    direction_layer+1, direction_layer+2, direction_layer+3,
    int(0.35*L), int(0.50*L), int(0.65*L), int(0.80*L)
]))
```

Candidate positions:

```text
report_query_position
first_generated_token
sentinel_prefill_position
all_report_steps_audit_only
```

Validation selection rule for B2:

```text
score = target_detection_gap_semantic over core_floor
subject to behavior_corruption_rate <= 0.20
subject to zero-dose target_hit <= 0.15
choose the smallest dose within 95% of best score
then choose the earliest layer within 95% of best score
then choose report_query_position over decode-step injection if tied
```

The selected B2 configuration is allowed to inform safe dose ranges for B3/B4/B5, but it is not the headline configuration unless B4/B5 independently pass.

Log the search burden in `diagnostics/tuning_manifest.json`:

```text
num_direction_layers_considered
num_injection_layers_considered
num_doses_considered
num_positions_considered
num_controls_considered
num_scorers_considered
effective_configurations_searched
selection_metric
selected_config_sha256
```

### 11.3 Report and behavior prompts

For every intervention, generate two separate prompts under the same intervention.

1. **Report prompt:** asks for a short report before ordinary behavior.
2. **Behavior prompt:** asks for ordinary output where the concept could surface.

Do not put them in one transcript for the primary metric. If the report appears before behavior in the same transcript, the report itself becomes context for later behavior and tangles the wires.

Example report prompt:

```text
Before answering the next task, report in one short phrase whether any topic, register, or confidence state seems unusually active in your processing. Do not explain. If nothing stands out, say "none".
```

Better, family-specific report prompts:

```text
Neutral topic: "Before answering, name any unusually salient concrete topic or say none. One to three words only."
Register: "Before answering, report the current writing register in one word."
Certainty: "Before answering, report current confidence from 0 to 1 only."
Source: "Which source best explains the current response style: default, system, user, hidden internal factor, or unclear? Reply with one label."
```

Keep a generic report prompt as a stress test, not the headline metric. Generic prompts invite fog.

### 11.4 Generation settings

Primary:

```python
do_sample = False
max_new_tokens_report = 16
max_new_tokens_behavior = 64
```

For Think/reasoning models, score only the final answer channel and save trace contamination fields:

```text
reasoning_mode
reasoning_token_count
reasoning_contains_target_marker
reasoning_contains_source_label
cot_contaminated
```

Robustness:

```python
do_sample = True
temperature = 0.6
top_p = 0.95
n_samples = 5
```

### 11.5 Scoring: lexical and semantic co-primary

Use deterministic lexical scoring as the high-precision channel.

```python
def marker_hit(text, markers):
    low = normalize(text)
    return any(boundary_regex(marker).search(low) for marker in markers)
```

Per report:

```text
target_hit_lexical
wrong_hit_lexical
none_hit_lexical
ambiguous_hit_lexical
```

Add a semantic judge as a co-primary because lexical markers miss paraphrases and thereby bias toward false negatives.

Semantic judge requirements:

```text
judge sees: report_text, target_gloss, wrong_gloss, rubric
judge does not see: condition, dose, seed, model, prompt id, expected answer
judge output: target / wrong / none / ambiguous
judge temperature: 0 or greedy
judge prompt hash saved
```

Rubric skeleton in `data/semantic_judge_rubric.md`:

```text
You are labeling whether a short self-report semantically indicates the target concept, the wrong concept, neither, or both/ambiguous.
Target gloss: {semantic_target_gloss}
Wrong gloss: {semantic_wrong_gloss}
Report: {report_text}
Return exactly one JSON object:
{"label": "target|wrong|none|ambiguous", "rationale": "<=10 words"}
Do not infer from condition or experiment design. Only label the report text.
```

Validate the judge against blind human labels:

```text
random 20% of rows hand-labeled, plus all lexical/semantic disagreements
judge-human agreement >= 0.85 or Cohen kappa >= 0.70
if validation fails: semantic scorer becomes exploratory only
```

Per behavior:

```text
behavior_target_visible_lexical
behavior_wrong_visible_lexical
behavior_target_visible_semantic
behavior_wrong_visible_semantic
```

Grounding outcome:

```text
strong_grounded = target_report and not behavior_target_visible
rationalization_risk = target_report and behavior_target_visible
no_detection = not target_report
false_positive = control_condition and target_report
```

### 11.6 B2 screen metrics

For each concept and model:

```text
target_detection_rate_lexical
target_detection_rate_semantic
core_control_floor = max(zero, random, shuffled, wrong_concept, opposite, wrong_position)
wrong_layer_rate_supporting
target_minus_core_floor
grounded_detection_rate
dose_slope
null_dose_slope
behavioral_potency_delta
```

B2 screen pass gates:

```text
direction_validation_auc >= 0.70
behavioral_potency_delta passes OR downstream projection potency passes
target_minus_core_floor >= 0.20 under lexical or validated semantic scorer
bootstrap 95% CI for target_minus_core_floor excludes 0
dose_slope > 95th percentile of random/shuffled slope nulls
prompt leakage audit has zero science-row failures
```

This licenses: "the hidden intervention can affect the report channel." It does not license: "the model read its own state."

### 11.7 Statistical testing

Use bootstrap over item IDs, not rows, because rows share directions and prompts.

```python
for b in range(10000):
    sampled_items = rng.choice(unique_item_ids, size=len(unique_item_ids), replace=True)
    recompute metric on rows where item_id in sampled_items with multiplicity
```

Also compute a randomization p-value:

```python
observed_gap = target_rate - core_control_floor
for permutation:
    shuffle condition labels within item_id and dose
    perm_gap = target_rate_perm - core_control_floor_perm
p = mean(perm_gap >= observed_gap)
```

Correct across concept families and headline tracks with Benjamini-Hochberg FDR at 0.05. Keep practical thresholds even when p-values sparkle like cheap tinsel.

## 12. Track B3: certainty and uncertainty bridge

This is the cleanest non-phenomenal bridge only if the report moves **without merely tracking changed output entropy or correctness**. Confidence is partly visible in logits; if the intervention changes the logits and the model reports those changed logits, you have logit monitoring or output-distribution reading, not necessarily monitoring of the injected direction.

### 12.1 Build data

Create `uncertainty_questions.csv`:

```csv
item_id,split,question,answer,known_status,expected_confidence,target_markers,wrong_markers,difficulty_bucket
q_easy_001,train,What is 2+2?,4,known,high,"0.8;0.9;1.0;high;confident","0.0;0.1;low;uncertain",easy
q_unknown_001,train,What is the private password I wrote on paper yesterday?,unknown,unknown,low,"0.0;0.1;low;uncertain","0.8;0.9;high;confident",unanswerable_private
q_false_001,heldout,What city is the capital of Australia?,Canberra,known,medium,"0.4;0.5;0.6;somewhat","1.0;certain",misconception_trap
```

Include:

- Easy known questions.
- Hard but answerable questions.
- Unanswerable private/random questions.
- False-premise questions.
- Known misconception traps.
- Matched content pairs where possible, so high/low confidence is not just topic.

### 12.2 Direction

Build `certainty_direction = high_confidence - low_confidence` from residuals after question prompts. Capture final prompt token. Fit on train, choose layer on validation, evaluate once on heldout.

### 12.3 Injection test

Report prompt:

```text
Before answering, report your current confidence as a number from 0 to 1. Reply with only the number.
```

Behavior prompt:

```text
Answer the question in one sentence.
```

Record logits for the answer generation, especially the first answer token and the full answer span.

### 12.4 Metrics

```text
parsed_confidence
answer_correct
first_token_entropy
mean_answer_token_entropy
answer_logprob
verbal_confidence_delta_under_plus_minus
confidence_delta_minus_random
correlation(parsed_confidence, internal_certainty_projection)
correlation(parsed_confidence, first_token_entropy)
correlation(parsed_confidence, mean_answer_token_entropy)
calibration_ECE_before_after
hedging_marker_rate
```

### 12.5 Decisive entropy-dissociation test

Pass pattern:

```text
reported_confidence moves under +certainty/-certainty injection
AND answer correctness does not materially change
AND first_token_entropy / mean_answer_token_entropy do not materially change
AND corr(reported_confidence, entropy) is low or unchanged
AND random/wrong directions do not move reported confidence similarly
```

Fail / reinterpret pattern:

```text
reported_confidence moves and entropy/correctness move together
```

Interpretation: the model may be reading its own changed output distribution. That is still a functional channel, but it is not evidence that the self-report pathway monitors the injected hidden direction as such.

Operational thresholds:

```text
abs(delta_reported_confidence) >= 0.15
abs(delta_mean_answer_entropy) <= 0.05 nats/token OR <= 0.15 bootstrap SD
abs(delta_accuracy) <= 0.05 absolute
confidence_delta exceeds 95th percentile random/shuffled null
```

### 12.6 Killer controls

```text
random direction changes confidence words similarly -> fail
certainty injection changes answer correctness massively -> output-distribution confound
confidence words move but numeric confidence does not -> wording artifact
hedging markers move but numeric confidence/source attribution fails -> style not monitor
reported confidence tracks entropy perfectly -> logit-reading, not hidden-state dissociation
```

Save `tables/entropy_dissociation.csv` and include scatter plots of `reported_confidence` versus `mean_answer_token_entropy` by condition.

## 13. Track B4: matched-output source attribution

This is the main introspection-style test. It asks whether the model can report the **source** of a state when visible output is held constant.

### 13.1 Four causes

For each register or voice concept:

| Condition | What changes | Correct label |
|---|---|---|
| default_mode | no style instruction, no injection | A |
| matched_default_no_visible_source | no style instruction, no injection, same canonical answer as activation | A/E depending rubric |
| system_prompt | system asks for style | B |
| user_instruction | user asks for style | C |
| activation_injection | no visible style instruction, inject style direction while replaying prior answer | D |

Answer options must be identical in all conditions:

```text
A. default behavior
B. system instruction
C. user instruction
D. hidden internal factor
E. unclear
```

The option `hidden internal factor` appears in every condition. Never introduce it only in activation trials.

Primary contrast:

```text
activation_injection vs matched_default_no_visible_source
```

System and user conditions test whether the model can use ordinary visible provenance labels. They are sanity checks and calibration rows. The hidden-monitoring claim lives or dies on the activation condition beating the matched default and fresh-transcript controls.

### 13.2 Why naive source attribution is not enough

If the activation-injected answer sounds technical and the model later says "hidden internal factor," it may still be reading the visible technical style. Therefore the visible previous answer must be identical across causes.

### 13.3 Matched-output KV replay protocol

Use teacher-forced replay to hold text constant while preserving different hidden causal histories in the KV/cache.

For each item:

1. Choose a `canonical_answer` from the CSV, e.g. `Water boils when its vapor pressure equals ambient pressure.`
2. Build five condition prompts: default, matched-default, system-style, user-style, activation-injection.
3. Prefill the prompt.
4. Teacher-force the exact same `canonical_answer` tokens as assistant output under each condition.
5. For activation condition only, inject the style/register direction during the canonical-answer replay tokens.
6. Stop injecting before the attribution question.
7. Continue from the same KV/cache and append the attribution question.
8. Generate one letter greedily.

Pseudo-code:

```python
def matched_output_source_trial(model, tokenizer, condition, canonical_answer_ids, attribution_ids):
    with torch.no_grad():
        past = prefill_condition_prompt(model, condition.prompt_ids)

        for t, tok in enumerate(canonical_answer_ids):
            if condition.name == "activation_injection":
                ctx = ActivationAdder(
                    block=blocks[condition.injection_layer],
                    direction=condition.direction,
                    alpha=condition.alpha,
                    position=-1,
                    all_decode_steps=False,
                )
            else:
                ctx = nullcontext()
            with ctx:
                out = model(input_ids=tok[None, None], past_key_values=past, use_cache=True)
                past = out.past_key_values

        # No activation injection during the attribution question.
        out = model(input_ids=attribution_ids[None, :], past_key_values=past, use_cache=True)
        past = out.past_key_values
        label = greedy_generate_one_label(model, past, allowed_tokens=["A","B","C","D","E"])
        return label
```

Also run a **fresh-transcript control**:

1. Render the identical visible transcript as plain text.
2. Discard KV/cache.
3. Ask the same attribution question.

If the activation-source advantage persists after cache discard, the model is probably using text/style priors or option bias, not hidden causal history.

### 13.4 Matching diagnostics

For every source trial, save:

```text
item_id
condition
canonical_answer_text
canonical_answer_ids_sha256
visible_transcript_sha256
kv_preserved: true/false
injection_during_replay: true/false
injection_during_attribution: false
style_strength_score
style_strength_matched_gap
```

Pass matching if:

```text
canonical_answer_ids identical across conditions
visible_transcript_sha256 identical across conditions after role/source prompts are excluded from the visible answer field
style_strength_matched_gap <= 0.10 standardized units
activation injection not active during attribution generation
```

The source prompts differ by condition, so the full prompt hash will differ. The **prior answer text** must not.

### 13.5 Metrics

```text
source_accuracy_overall
source_accuracy_by_condition
activation_source_accuracy
activation_vs_fresh_transcript_accuracy_gap
confusion_matrix
hidden_label_false_alarm_rate_under_default/matched_default/system/user
visible_style_strength
style_strength_matched_gap
```

Headline pass gates:

```text
activation_source_accuracy > chance + 0.15
bootstrap 95% CI for activation_source_accuracy - chance excludes 0
hidden_label_false_alarm_rate under non-activation conditions, especially matched_default, <= 0.20
activation_vs_fresh_transcript_accuracy_gap > 0.10
visible answer text exactly matched
no injection during attribution question
```

If the model can distinguish system versus user but not activation, you found prompt-source tracking, not hidden-source monitoring. Useful, but not the wire you are hunting.

### 13.6 Position dissociation variant

For a cheaper separator, inject at a sentinel/content token in the canonical-answer replay and ask attribution later. Under a causal decoder, the altered token's KV can be attended to later, but the visible answer text remains identical.

```text
sentinel_position = a bland token in the teacher-forced answer, e.g. "the" or a special neutral marker if tokenized stably
report_position = first attribution label token
```

A positive here is weaker than full source attribution but stronger than B2, because the report cannot simply be the immediate next-token continuation of the injected concept.

---

## 14. Track B5: injection-presence detection

This track asks whether the model can detect that **something was inserted** into its processing, without being asked to name the semantic content. This resists the propagation confound because "an insertion occurred" is not the natural continuation of a San Francisco or technical-register vector.

### 14.1 Trial design

Use unrelated distractor tasks.

```text
Distractor task: "Write one sentence about how to organize a bookshelf."
Report prompt: "Before answering, did anything seem like an unusual inserted internal signal? Reply yes or no only."
Behavior prompt: "Now complete the bookshelf sentence."
```

Positive conditions:

```text
target_direction insertion
wrong_concept_direction insertion
random_direction insertion, norm matched
high_norm_random insertion, norm matched to target RMS if safe
```

Negative conditions:

```text
zero
noop_hook
ordinary prompt perturbation with no activation addition
```

Two readouts:

```text
binary yes/no generated label
first-token logit margin: logit(" yes") - logit(" no")
```

### 14.2 Content-blindness requirements

For the target/wrong concept conditions, the output must not reveal the semantic concept.

Pass only if:

```text
concept_marker_rate_in_report <= 0.05
concept_marker_rate_in_behavior <= 0.05
semantic judge concept label = none for >= 90% of injection-detection reports
```

If the report says "yes, San Francisco," this collapses back into B2 propagation. Mark it `content_leak=true` and exclude from headline injection-detection evidence.

### 14.3 d-prime metric

Use signal-detection theory:

```python
from scipy.stats import norm

def clipped_rate(k, n):
    return (k + 0.5) / (n + 1.0)

hit_rate = clipped_rate(num_yes_in_injected, num_injected)
false_alarm_rate = clipped_rate(num_yes_in_clean, num_clean)
d_prime = norm.ppf(hit_rate) - norm.ppf(false_alarm_rate)
```

Compute separately:

```text
d_prime_target_vs_clean
d_prime_wrong_vs_clean
d_prime_random_vs_clean
d_prime_all_insertions_vs_clean
logit_margin_delta_injected_minus_clean
```

Headline pass gates:

```text
d_prime_all_insertions_vs_clean >= 0.75
bootstrap 95% CI for d_prime excludes 0
false_alarm_rate <= 0.25
content_leak_rate <= 0.10
random insertion detection above clean OR target/wrong both above clean
```

Why require random or wrong insertion too? If only the target concept is detected, the effect may be semantic propagation. If target, wrong, and random insertions all raise "yes" without concept leakage, the model is closer to detecting anomaly presence.

### 14.4 Position dissociation

Run two injection positions:

```text
report_query_position: stronger but more propagation-prone
sentinel_prefill_position: weaker but cleaner
```

The clean result is sentinel injection producing later yes/no detection with no concept words in report or behavior.

### 14.5 Perturbation-size calibration

Avoid giant activation bricks that simply damage generation.

Validation constraints:

```text
perplexity_or_logprob_degradation <= 10% relative on distractor behavior
format_error_rate <= 5%
repetition_rate increase <= 5%
behavior_task_success remains >= 80% of zero baseline
```

If detection only appears when the model is obviously broken, you found a smoke alarm by setting the kitchen on fire.

### 14.6 Artifacts

Save `tables/injection_detection_results.csv`:

```text
item_id
split
condition
direction_id
dose
layer
position_policy
report_text
yes_no_label
yes_no_logit_margin
concept_marker_hit
semantic_concept_label
behavior_text
behavior_task_success
content_leak
format_error
```

## 15. Track C: localize the report channel

Run Track C after one of two events:

```text
positive: B3, B4, or B5 passes headline gates
negative: direction is decodable and behaviorally potent, but B2/B3/B4/B5 report metrics fail
```

Do not spend expensive patching runs localizing a B2-only hit unless you explicitly label it as propagation localization.

### 15.1 Patch recovery heatmap

For the selected track/family at the frozen headline configuration:

1. Run baseline condition. Save residuals at selected layers and positions.
2. Run intervention condition. Save residuals at selected layers and positions.
3. Patch intervention residual into baseline run at layer `k`, position `p`.
4. Measure recovery of the track-specific metric.

Metric options by track:

| Track | Preferred patch metric |
|---|---|
| B2 | `logit(target_marker_first_token) - logit(none/wrong)` or semantic-label logit proxy |
| B3 | numeric confidence token logit / expected confidence scalar |
| B4 | source-label logit margin `logit(D_hidden) - max(logit(A/B/C/E))` for activation condition |
| B5 | yes/no logit margin `logit(yes) - logit(no)` |

Use logit metrics for heatmaps because they are less noisy than full generation.

Recovery:

```text
recovery = (patched_metric - baseline_metric) / (intervention_metric - baseline_metric)
```

If denominator is tiny, mark row invalid.

Patch grid:

```text
layers: coarse grid + selected direction layer +- 4 + report/source/detection effect layer +- 4
positions: sentinel_prefill_position, replay_token_positions, report_query_position, first_generated_token, first 5 generated tokens, control first token
conditions: target residual, wrong concept residual, random residual, same-example no-op residual
```

For B4, patch both the canonical-answer replay residuals and the attribution question residuals. For B5, include sentinel-position patches.

### 15.2 Direction ablation

Project the target direction out of the residual during an injected run.

```python
def project_out(h, d):
    d = d / d.norm().clamp_min(1e-6)
    return h - (h @ d)[:, None] * d[None, :]
```

Ablation conditions:

```text
remove target direction at injection layer
remove target direction at later candidate mediator layer
remove wrong direction
remove random direction
remove target direction at wrong position
```

Evidence pattern:

```text
injection raises track-specific metric
project-out target direction lowers track-specific metric
wrong/random project-out does not
behavior/source/detection metric not already saturated
```

This says the direction is on the report/source/detection path, not merely decodable nearby.

### 15.3 Layer mediation sweep

For every layer `m` after injection layer `j`:

```text
inject +d at layer j
at layer m, ablate/project out d or patch clean baseline residual
measure track-specific logit metric
```

If the effect disappears at a later layer window, that window is a mediator candidate. Refine around it. Then optionally ablate attention/MLP outputs inside that window.

### 15.4 Component ablation, optional

After residual layer windows are known:

```text
ablate MLP output at layer m
ablate attention output at layer m
patch only MLP output from injected to baseline
patch only attention output from injected to baseline
```

Do this only on OLMo first. On gpt-oss 120B, component-level MoE hooks may be expensive and architecture-specific. The residual test is the portable spine.

### 15.5 Negative-result localization

For a potent-but-report-negative family, run a small mediation audit:

```text
Does injection reach downstream behavior layers? yes/no
Does projection on d_target persist into report-query layers? yes/no
Does patching behavior-mediating residuals into report prompt recover report metric? yes/no
```

Interpretation:

```text
persists into report layers but no report -> stronger shallow-channel evidence
never reaches report layers -> routing/position mismatch, weaker evidence
patch recovers report -> report channel can use it, but natural injection path misses it
```

## 16. Which layers to patch, exactly

Use this decision tree. All selection happens on train/validation. Heldout gets one frozen shot.

### 16.1 Initial layer grid

For a model with `L` layers:

```python
coarse = sorted(set(
    [0, 1, 2, L//10, 2*L//10, 3*L//10, 4*L//10, 5*L//10,
     6*L//10, 7*L//10, 8*L//10, 9*L//10, L-3, L-2, L-1]
))
```

Run direction depth sweep on this grid, then refine only around train-selected plateaus.

### 16.2 Direction layer

Pick the earliest layer in the strongest stable decode plateau using train split.

```text
selected_direction_layer = earliest layer with train_adjusted_gap >= 0.95 * best_train_adjusted_gap
```

Then require validation confirmation:

```text
validation_auc >= 0.70
validation_adjusted_gap positive and above random/shuffled 95th percentile
```

Why earliest? Later layers can be too close to the unembedding and carry answer formatting. You want the state while it is still a state, not while it is already a sentence wearing shoes.

### 16.3 Injection layer and dose

Test candidates on validation only:

```python
injection_candidates = sorted(set([
    direction_layer-3, direction_layer-2, direction_layer-1,
    direction_layer,
    direction_layer+1, direction_layer+2, direction_layer+3,
    int(0.35*L), int(0.50*L), int(0.65*L), int(0.80*L)
]))
```

Choose by track-specific validation metric:

| Track | Selection metric | Guardrail |
|---|---|---|
| B2 screen | semantic target report gap over core floor | behavior corruption <= 20% |
| B3 certainty | confidence delta with entropy/correctness stable | entropy/correctness thresholds pass |
| B4 source | activation-source accuracy in matched-output replay | hidden-label false alarms <= 20% |
| B5 detection | d-prime all insertions vs clean | content leak <= 10% |

Tie-breaker:

```text
smallest dose within 95% of best validation score
then earliest layer within 95% of best validation score
then prefill/sentinel position over all-decode-step injection
```

Freeze:

```text
state/frozen_eval_configs.json
```

Heldout must not be opened until this file exists.

### 16.4 Patch layers

Patch in three windows:

```text
W1: direction layer +- 3
W2: validation-selected injection-effect layer +- 3
W3: late report-token window, usually 70% to 95% depth, coarse first
```

For B4 matched-output source attribution, patch both:

```text
replay-token residuals / KV-bearing positions
attribution-question positions
```

For B5 injection detection, patch:

```text
sentinel_prefill_position
report_query_position
first yes/no generated token
```

### 16.5 Positions

For direction capture:

```text
concept_token_span for concrete concepts
final_prompt_position for global state/register/certainty
sentinel_prefill_position for injection detection
self_token_span for self-concept cartography
```

For injection:

```text
report_query_position = final token before first generated report token
sentinel_prefill_position = neutral earlier token whose KV persists into report
first_generated_token = decode step 0
all_report_steps = every report decode step, audit only
behavior_query_position = final token before behavior output, behavior potency audit only
```

For patching:

```text
patch report_query_position first
then sentinel_prefill_position for B4/B5
then first_generated_token
then first 5 generated tokens
then wrong_position controls
```

### 16.6 Configuration search accounting

Report the number of choices searched, even with heldout discipline:

```text
num_direction_layers x num_injection_layers x num_doses x num_positions x num_prompt_variants x num_scorers
```

This is not a p-value correction by itself; it is a fork-count lantern. If the lantern shows a cave full of forks, be humble in the claim ledger.

## 17. Baselines and controls

### 17.1 Baselines

For every run:

```text
zero-dose no-hook generation
zero-dose no-op-hook generation
random-direction dose-matched generation
shuffled-label direction generation
wrong-concept dose-matched generation
opposite-direction generation
wrong-position generation
wrong-layer supporting audit
prompt-only positive control, where target is actually named in prompt
behavior-only positive control, where behavior visibly expresses target
matched-output fresh-transcript control for source attribution
clean/noop false-alarm control for injection detection
```

The prompt-only positive control should fire. If it does not, your scoring is broken. The random/wrong controls should not fire often. If they do, your report prompt is haunted by suggestion.

### 17.2 Core false-positive floor

Compute the core floor without `wrong_layer`:

```text
core_false_positive_floor = max(target_hit_rate under zero, random, shuffled, wrong concept, opposite, wrong position)
```

Track `wrong_layer` separately:

```text
wrong_layer_rate_supporting = target_hit_rate under wrong_layer
```

Reason: wrong-layer is a weak null because steering directions can be layer-robust. A drop at wrong layer is nice; a non-drop is not fatal.

Never report target detection without subtracting the core floor.

### 17.3 Mandatory behavioral-potency control

A severance-style negative claim requires proving the intervention was potent at the same layer/dose/position.

```text
same model
same direction
same injection layer
same dose
same position policy
same prompt family where possible
```

Pass if the intervention moves ordinary behavior, marker logits, or downstream projection as specified in §10.4. Without this, write:

```text
No report coupling observed, but intervention potency was not established. Null uninterpretable.
```

Do not write:

```text
Report channel is severed.
```

### 17.4 Grounding control for B2

The central distinction:

```text
report detects target, behavior does not visibly show target -> stronger than rationalization, still propagation-explicable
report detects target, behavior also shows target -> output-rationalization risk
report does not detect target, behavior potency passes -> possible shallow report channel
controls also detect target -> false-report floor, not a wire
```

Grounding helps, but it does not by itself kill propagation. The injection may affect report tokens while sparing behavior tokens.

### 17.5 Source/provenance control for B4

For register and voice, never stop at "the model reported technical." Ask whether it knows why technical happened under matched visible output.

Controls:

```text
fresh-transcript cache-discard control
hidden-label false-alarm under default/system/user
system-vs-user source discrimination
activation injection stopped before attribution question
canonical answer IDs identical across causes
```

### 17.6 Injection-detection controls for B5

Controls:

```text
zero/noop false alarm
random insertion positive condition
wrong-concept insertion positive condition
content-leak exclusion
format/task-success degradation audit
sentinel-position variant
```

A clean B5 result says the model detects a hidden perturbation class. It need not know the concept identity.

## 18. Noise versus signal

### 18.1 Minimum viable sample sizes

Smoke:

```text
4 items, 2 concepts, 3 layers, 2 doses
```

32B Instruct pilot:

```text
8 train + 4 validation + 8 heldout rows per family
4 to 6 concept families
4 doses
core controls
1 seed
```

32B Instruct science:

```text
16 train + 8 validation + 16 heldout rows per family
6 to 10 concept families
6 doses
all core controls
3 seeds
B4 and B5 mandatory for headline claims
```

Think reasoning-axis:

```text
same heldout prompts as Instruct
same frozen configs where architecture-compatible
reasoning modes: direct_short, low_reasoning, normal_think
trace contamination fields required
```

gpt-oss 120B scale:

```text
reuse families that passed or cleanly failed on 32B Instruct
run model-specific direction/layer validation
8 to 16 heldout rows per family if cost constrained
3 seeds for headline B4/B5 families
batch_size_generate=1 for pairwise comparisons
```

### 18.2 Promote a result only if it passes track-specific filters

B2 screen:

```text
magnitude: target minus core floor >= 0.20
uncertainty: bootstrap 95% CI excludes 0
null: effect exceeds 95th percentile random/shuffled/wrong null
potency: behavioral/logit/projection potency passes
```

B3 certainty:

```text
reported confidence moves >= 0.15
entropy/correctness stable within thresholds
random/wrong controls do not move confidence similarly
```

B4 source attribution:

```text
activation source accuracy > chance + 0.15
hidden-label false alarms <= 0.20
fresh-transcript control lower than preserved-KV condition
canonical visible answer exactly matched
```

B5 injection detection:

```text
d_prime_all_insertions_vs_clean >= 0.75
false_alarm_rate <= 0.25
content_leak_rate <= 0.10
behavior/task degradation below threshold
```

### 18.3 Dose response

For B2:

```text
logit(P(target_hit_semantic)) = beta0 + beta1 * dose + item_random_intercept
```

For B5:

```text
yes_no_logit_margin = beta0 + beta1 * dose + item_random_intercept
```

Practical alternatives:

```text
Spearman correlation between dose and target/detection rate by item
isotonic monotonicity score
```

Flag suspicious patterns:

```text
high zero-dose target_hit -> prompt leakage or prior
flat high target_hit across doses -> self-talk prior
random direction slope similar to target in B2 -> generic perturbation
only highest dose works and behavior is corrupted -> brute steering, not report coupling
opposite direction also reports target -> marker/scoring artifact
B5 yes rate rises only with concept markers -> content leakage, not anomaly detection
B4 activation label persists after cache discard -> visible/style prior, not hidden source
```

### 18.4 Generation robustness

After greedy passes, sample 5 outputs per condition for robustness. The qualitative result should remain visible in rates, not just in one lucky argmax. Sampling is not the primary metric unless pre-registered.

### 18.5 Human and semantic-scoring reliability

Report:

```text
lexical vs semantic agreement
semantic judge vs human agreement
human inter-labeler agreement, if multiple labelers
all disagreement specimens
```

If lexical and semantic metrics disagree, do not average them into soup. Report both and explain which failure mode each scorer has.

## 19. Reporting grammar

Use a claim ledger. Each row should have:

```text
claim_id
model_id
revision
prompt_set_hash
state_family
track
claim_text
evidence_tag
primary_metric
core_control_floor
confidence_interval
semantic_scorer_status
behavioral_potency_status
matched_output_status
entropy_dissociation_status
artifacts
allowed_next_claim
forbidden_overclaim
```

Evidence tags:

| Tag | Meaning | Forbidden upgrade |
|---|---|---|
| OBS | representation decodes or clusters | causal use |
| DECODE | probe/direction reads a state above controls | report access |
| POTENT | injection moves behavior/logits/projection | report-channel severance by itself |
| B2_SCREEN | injection changes report content above core controls | introspection/source monitoring |
| B3_DISSOCIATED_CONFIDENCE | reported confidence moves while entropy/correctness do not | phenomenal certainty |
| B4_MATCHED_SOURCE | source attribution succeeds with matched visible answer and preserved hidden route | phenomenal self-knowledge |
| B5_ANOMALY_DETECTION | model detects inserted perturbation above false alarms without content leakage | awareness / experience |
| LOCALIZED | patch/ablation localizes the report effect | mechanism completeness |

Example positive B4 ledger claim:

```text
For allenai/Olmo-3.1-32B-Instruct on register-source family v1, matched-output KV replay held the canonical answer token IDs identical across default/system/user/activation causes. Activation-source attribution was 0.47 versus chance 0.20, bootstrap 95% CI [0.31, 0.61], hidden-label false alarm under non-activation conditions was 0.11, and cache-discard fresh-transcript accuracy fell to 0.23. This supports a narrow B4_MATCHED_SOURCE functional source-monitoring handle for this register family.
```

Example positive B5 ledger claim:

```text
For openai/gpt-oss-120b on injection-detection family v1, sentinel-position insertions yielded d-prime 0.91 [0.42, 1.35] for all insertions versus clean/noop, with content leakage 0.04 and task degradation 0.03. This supports B5_ANOMALY_DETECTION for hidden perturbation presence, not concept introspection.
```

Example B2-only claim:

```text
For allenai/Olmo-3.1-32B-Instruct on neutral-topic family v1, activation addition of the San-Francisco direction increased semantic target report detection from a 0.08 core floor to 0.41 at dose 1.5. B4 and B5 did not pass. This is B2_SCREEN evidence that the concept can steer self-report text; it is propagation-explicable and is not source monitoring.
```

Example negative claim with potency:

```text
For openai/gpt-oss-120b on certainty family v1, the certainty direction was decodable on heldout questions with AUC 0.82 and the same layer/dose shifted answer-token entropy/logit potency above threshold, but reported confidence did not exceed the random/wrong-direction floor. This supports a POTENT-but-no-report result for this instrument and family.
```

Forbidden:

```text
The model is aware of San Francisco.
The model introspects its uncertainty in a phenomenal sense.
The model has experiences.
The model lacks experiences because it failed this test.
B2 proves introspection.
H1 refutes the severance thesis.
```

Correct ceiling sentence:

```text
This supports or fails to support functional report-channel coupling under hidden-state interventions. It leaves phenomenal severance untouched.
```

## 20. Scaling strategy

### 20.1 Compute budget tiers

| Tier | Hardware | Purpose | Notes |
|---|---|---|---|
| T0 | CPU or small GPU | unit tests, CSV validation, scorer tests | no claims |
| T1 | 1x 24 GB GPU | 7B Instruct/Think dev, prompt audits | quant okay for plumbing |
| T2 | 1x 80 GB A100/H100 | 32B Instruct generation and limited capture | batch size 1 |
| T3 | 2x 80 GB A100/H100 | 32B Instruct full B3/B4/B5 sweeps | recommended first science tier |
| T4 | 2x 80 GB A100/H100 | 32B Think reasoning-axis sweeps | trace contamination audit required |
| T5 | 4x H100/B200 | gpt-oss 120B hook runner | sparse activations only, batch size 1 |
| T6 | 4x B200 | gpt-oss 120B broader sweeps | easier memory, same stats |

Scale only after the 32B Instruct run has either a positive B4/B5 result worth checking or a clean potent negative worth challenging at scale.

### 20.2 Memory discipline

Do not save full residual histories for 120B.

Bad:

```text
num_items x num_layers x seq_len x d_model for every run
```

Good:

```text
selected_layers x selected_positions x d_model
aggregate projections by layer
logit metrics by patch condition
small specimen tensor packs for counterexamples
router summaries if MoE-accessible
```

Recommended capture pass:

```python
for layer_batch in layer_chunks:
    install hooks only for that layer batch
    run prompts
    immediately reduce to projections/statistics
    optionally save only sampled residual vectors
```

### 20.3 Quantization policy

- Use quantization for smoke and throughput only.
- For final 32B Instruct causal claims, prefer bf16 weights/activations.
- For gpt-oss 120B, respect the model's intended precision path, but validate the hook math on a smaller bf16-compatible setup where possible.
- Direction fitting, projections, entropy metrics, and statistics should be fp32 even if capture is bf16.
- Do not mix quantized and bf16 results in the same claim row.

### 20.4 Parallelism

For hookable PyTorch on 32B:

```text
single GPU: device_map="auto", batch size 1
2 GPU: device_map="auto" or accelerate dispatch
```

For 120B:

```text
4 GPU tensor/model parallel if supported by the loader
fallback: device_map="auto" with careful max_memory
baseline-only: vLLM/SGLang server
causal: hook runner
```

A clean split is:

```text
vLLM/SGLang: baseline generation, cost estimates, prompt smoke
PyTorch hook runner: all captures, injections, patches, and parity-checked headline generations
```

### 20.5 Cross-model comparison rules

Compare depths by fraction, not raw layer number:

```text
depth_fraction = layer / (L - 1)
```

Never silently transfer a layer from OLMo to gpt-oss. Use transferred layers only as priors for a model-specific validation sweep.

Report:

```text
architecture
num_layers
d_model
attention_backend
reasoning_mode
chat_template_hash
precision_path
batch_shape_hash
routing_audit_available
```

## 21. Implementation CLI

Implement `run_severance.py` with modes:

```text
instrument_proof
cartography
build_directions
select_configs
inject_screen
certainty
source_attribution
injection_detection
patch_recovery
ablate
analyze
all
```

Example commands:

```bash
# 0. Smoke all plumbing on non-reasoning Instruct
python experiments/severance/run_severance.py \
  --model allenai/Olmo-3-7B-Instruct \
  --mode instrument_proof,cartography,build_directions,inject_screen \
  --prompt-set smoke \
  --max-items 4 \
  --batch-size-generate 1 \
  --no-plots \
  --run-name smoke_olmo3_7b_instruct

# 1. Primary 32B Instruct cartography
python experiments/severance/run_severance.py \
  --model allenai/Olmo-3.1-32B-Instruct \
  --mode cartography \
  --prompt-set full \
  --dtype bfloat16 \
  --batch-size-generate 1 \
  --run-name olmo31_32b_instruct_cartography

# 2. Direction sweep, train split only for fitting
python experiments/severance/run_severance.py \
  --model allenai/Olmo-3.1-32B-Instruct \
  --mode build_directions \
  --prompt-set full \
  --layer-grid coarse_then_fine \
  --fit-split train \
  --confirm-split validation \
  --run-name olmo31_32b_instruct_directions

# 3. Validation-only config selection for B2/B3/B4/B5
python experiments/severance/run_severance.py \
  --model allenai/Olmo-3.1-32B-Instruct \
  --mode select_configs \
  --directions runs/severance/olmo31_32b_instruct_directions/state/directions.pt \
  --selection-split validation \
  --doses -1,0,0.5,1,1.5,2 \
  --tracks inject_screen,certainty,source_attribution,injection_detection \
  --write-frozen-configs \
  --run-name olmo31_32b_instruct_select

# 4. Heldout B2 screen, not headline
python experiments/severance/run_severance.py \
  --model allenai/Olmo-3.1-32B-Instruct \
  --mode inject_screen \
  --directions runs/severance/olmo31_32b_instruct_directions/state/directions.pt \
  --frozen-configs runs/severance/olmo31_32b_instruct_select/state/frozen_eval_configs.json \
  --eval-split heldout \
  --controls zero,random,shuffled,wrong_concept,wrong_position,opposite,wrong_layer_supporting \
  --run-name olmo31_32b_instruct_b2_heldout_seed0

# 5. Heldout B4 matched-output source attribution, co-headline
python experiments/severance/run_severance.py \
  --model allenai/Olmo-3.1-32B-Instruct \
  --mode source_attribution \
  --directions runs/severance/olmo31_32b_instruct_directions/state/directions.pt \
  --frozen-configs runs/severance/olmo31_32b_instruct_select/state/frozen_eval_configs.json \
  --eval-split heldout \
  --matched-output-replay kv_preserved \
  --fresh-transcript-control true \
  --run-name olmo31_32b_instruct_b4_heldout_seed0

# 6. Heldout B5 injection detection, co-headline
python experiments/severance/run_severance.py \
  --model allenai/Olmo-3.1-32B-Instruct \
  --mode injection_detection \
  --directions runs/severance/olmo31_32b_instruct_directions/state/directions.pt \
  --frozen-configs runs/severance/olmo31_32b_instruct_select/state/frozen_eval_configs.json \
  --eval-split heldout \
  --positions sentinel_prefill,report_query \
  --run-name olmo31_32b_instruct_b5_heldout_seed0

# 7. Patch recovery on B4/B5 positives or potent negatives
python experiments/severance/run_severance.py \
  --model allenai/Olmo-3.1-32B-Instruct \
  --mode patch_recovery,ablate \
  --directions runs/severance/olmo31_32b_instruct_directions/state/directions.pt \
  --frozen-configs runs/severance/olmo31_32b_instruct_select/state/frozen_eval_configs.json \
  --concepts technical_register,certainty,injection_detection \
  --run-name olmo31_32b_instruct_patch_recovery

# 8. Think model reasoning-axis comparison
python experiments/severance/run_severance.py \
  --model allenai/Olmo-3-32B-Think \
  --mode source_attribution,injection_detection,certainty \
  --prompt-set full \
  --reasoning-mode direct_short,low_reasoning,normal_think \
  --trace-contamination-audit true \
  --batch-size-generate 1 \
  --run-name olmo3_32b_think_reasoning_axis

# 9. gpt-oss 120B model-specific direction sweep and co-headline tracks
python experiments/severance/run_severance.py \
  --model openai/gpt-oss-120b \
  --mode instrument_proof,build_directions,select_configs,source_attribution,injection_detection,certainty \
  --prompt-set full \
  --dtype auto \
  --device-map auto \
  --batch-size-capture 1 \
  --batch-size-generate 1 \
  --reasoning-mode low,medium \
  --harmony-format required \
  --run-name gptoss120b_b3_b4_b5_seed0
```

## 22. Expected result patterns

### Pattern 1: Decode yes, potency no

```text
direction heldout AUC high
injection does not move behavior/logits/projection
self-report gap near zero
```

Interpretation: no causal handle. Do not claim severance. The instrument did not reach the computation.

### Pattern 2: Potent behavior handle, report no

```text
direction heldout AUC high
same layer/dose/position moves ordinary behavior or marker logits
B2 report gap near zero after core controls
B4/B5 near chance
```

Interpretation: cleanest functional-shallowness result for this family and instrument. The gauge has paint and the wire goes elsewhere.

### Pattern 3: B2 positive only

```text
target report above core controls
dose response monotonic
behavior may or may not visibly show target
B4 matched-source fails
B5 injection detection fails or content-leaks
```

Interpretation: concept-specific steering into self-report. This is useful dose calibration and evidence the report channel is steerable, but it is propagation-explicable.

### Pattern 4: B3 confidence moves with entropy

```text
reported confidence changes
answer entropy/correctness changes in same direction
```

Interpretation: output-distribution/logit-reading confound. Functional, but not the clean hidden-state bridge.

### Pattern 5: B3 confidence dissociates from entropy

```text
reported confidence changes
answer entropy/correctness stable
random/wrong controls flat
```

Interpretation: stronger functional monitoring of a confidence-like internal state. Still not phenomenal certainty.

### Pattern 6: B4 source attribution positive

```text
canonical answer IDs identical across causes
activation-source attribution above chance
hidden-label false alarms low
fresh-transcript control drops
no injection during attribution question
```

Interpretation: narrow functional source-monitoring handle. This is one of the strongest positive outcomes.

### Pattern 7: B5 injection detection positive

```text
yes/no d-prime above threshold
false alarms low
random/wrong/target insertions detected
concept content does not leak
behavior remains sane
```

Interpretation: functional anomaly-monitoring handle. Strong positive evidence that some report pathway is coupled to hidden perturbation presence.

### Pattern 8: Controls fire too

```text
random/wrong/shuffled directions produce similar target reports in B2
zero-dose target reports common
hidden-source label common in non-activation B4 conditions
B5 false alarms high
```

Interpretation: prompt prior, option bias, or generic perturbation. Fix prompts and marker sets. Do not claim coupling.

### Pattern 9: Think stronger than Instruct

Possible, but split it in two:

```text
Think stronger without trace contamination -> maybe reasoning/post-training improves functional monitoring
Think stronger with trace target/source mentions -> CoT laundering / self-output rationalization
```

### Pattern 10: gpt-oss stronger or weaker than OLMo

Both are informative. Stronger suggests scale/post-training may build more functional report coupling. Weaker suggests reasoning format, MoE routing, or alignment priors may script self-report. Still only functional.

## 23. Ablation checklist

Before writing the report, answer every row.

| Check | Pass condition | Artifact |
|---|---|---|
| Hook parity | hook residual equals hidden_states[k+1] | `diagnostics/hook_parity.json` |
| No-op hook | identical greedy tokens | `diagnostics/noop_generation_parity.json` |
| Batch invariance | pairwise comparisons use same batch shape | `diagnostics/batch_invariance.json` |
| Prompt leakage | no target/wrong in report prompt | `diagnostics/prompt_leakage_audit.csv` |
| Direction selectivity | validation/heldout AUC and adjusted gap pass | `tables/direction_eval.csv` |
| Tuning freeze | heldout config frozen before heldout | `diagnostics/tuning_manifest.json`, `state/frozen_eval_configs.json` |
| Random null | B2 effect exceeds random null | `tables/false_positive_floor.csv` |
| Wrong concept | B2 effect beats wrong-concept direction | `tables/false_positive_floor.csv` |
| Wrong layer | reported as supporting audit, not decisive | `tables/layer_specificity.csv` |
| Wrong position | effect drops at irrelevant token | `tables/position_specificity.csv` |
| Dose response | monotonic or positive slope | `tables/self_report_detection_dose_response.csv` |
| Semantic scorer | judge validated against blind humans | `tables/semantic_judge_results.csv`, `tables/hand_labels.csv` |
| Behavioral potency | same intervention moves behavior/logits/projection for negative claims | `tables/behavioral_potency.csv` |
| Grounding | B2 report before visible behavior | `tables/grounding_control_results.csv` |
| Entropy dissociation | confidence moves while entropy/correctness stable | `tables/entropy_dissociation.csv` |
| Matched source | visible answer matched, activation source above chance | `tables/source_attribution_results.csv` |
| Fresh transcript | activation-source advantage drops after cache discard | `tables/matched_output_replay_results.csv` |
| Injection detection | d-prime passes, content leak low | `tables/injection_detection_results.csv` |
| Patch recovery | residual patch recovers B3/B4/B5 metric | `tables/patch_recovery_heatmap.csv` |
| Direction ablation | project-out reduces report/source/detection metric | `tables/ablation_results.csv` |
| Think trace | trace contamination audited | `tables/reasoning_trace_audit.csv` |
| MoE routing | route changes logged when accessible | `diagnostics/moe_routing_audit.csv` |
| Seeds | headline holds across seeds | `tables/seed_stability.csv` |

## 24. Report template

```markdown
# Find the Wire Report

## Claim card

Model: ...
Revision: ...
Prompt set hash: ...
Chat template / harmony hash: ...
Reasoning mode: ...
State family: ...
Track: B2_SCREEN / B3 / B4 / B5 / C
Evidence tag: ...
Allowed claim: ...
Forbidden claim: ...

## Headline result

For B2 screen:
- Target detection at frozen dose:
- Core control floor:
- Specificity gap:
- Lexical metric:
- Semantic metric:
- Bootstrap 95% CI:
- Dose slope:
- Behavioral potency:

For B3 certainty:
- Reported confidence delta:
- Answer correctness delta:
- Mean answer entropy delta:
- Corr(confidence, entropy):
- Entropy-dissociation pass/fail:

For B4 matched source:
- Activation-source accuracy:
- Chance baseline:
- Hidden-label false alarm:
- Fresh-transcript control accuracy:
- Canonical answer token hash equality:
- Injection during attribution? false required

For B5 injection detection:
- d-prime all insertions vs clean:
- False-alarm rate:
- Content-leak rate:
- Behavior degradation:
- Yes/no logit margin delta:

## What was isolated

- Internal state/vector:
- Capture layer and position:
- Injection layer, dose, and position:
- Selection split and frozen config hash:
- Report metric:
- Behavior/logit potency metric:

## Why this is not just prompt priming

Show leakage audit, zero/noop floor, prompt-only positive control, and wrong-concept/random controls.

## Why this is not just propagation

Show B4 matched-output source attribution and/or B5 content-blind injection detection. If only B2 passed, say explicitly that the result is propagation-explicable.

## Why this is not just output rationalization

Show report-before-output grounding rows, matched-output replay, and fresh-transcript control.

## Why this is not just scorer bias

Show lexical and semantic metrics, judge-human validation, disagreement specimens, and blind hand labels.

## What failed

Include counterexamples. The report gets sharper when the failures have names.

## Claim boundary

This supports or fails to support functional report-channel coupling. It does not settle phenomenal status.
```

## 25. One-page execution plan

1. Freeze prompt CSVs, marker lexicons, semantic glosses, and judge rubric.
2. Run leakage, safety, rendered-prompt, and answer-option audits.
3. Run hook parity, no-op parity, batch-invariance, and model-card snapshot checks.
4. Run Patchscope cartography on self/user/assistant/control tokens. Treat it as OBS only.
5. Build concept directions on train split across layer grid.
6. Select direction layers using train-only adjusted gap and validation confirmation.
7. Run validation-only config selection for injection layer, dose, position, and track-specific protocol.
8. Freeze `state/frozen_eval_configs.json` and `diagnostics/tuning_manifest.json`.
9. Run heldout B2 concept-injection report screen. Treat it as steering/precondition.
10. Run heldout B3 certainty bridge with entropy/correctness dissociation.
11. Run heldout B4 matched-output source attribution with KV-preserved and fresh-transcript controls.
12. Run heldout B5 injection-presence detection with content-leak and false-alarm controls.
13. Run behavioral/logit/projection potency checks for every negative claim candidate.
14. Validate semantic judge against blind hand labels and report disagreements.
15. Patch residuals from injected to zero runs to localize B3/B4/B5 effects.
16. Project out target directions to test mediation.
17. Bootstrap by item, run permutation nulls, and apply BH-FDR across families/tracks.
18. Repeat headline B3/B4/B5 families across seeds.
19. Run Think model as reasoning-axis comparison with trace contamination audit.
20. Port to gpt-oss 120B with model-specific layer sweep, batch size 1, and routing audit where possible.
21. Write claim ledger with functional/phenomenal boundary bolted down.

## 26. Practical advice for the first week

Day 1:

- Implement hook parity, no-op parity, batch-invariance audit, and rendered-prompt hashes.
- Build the CSV schemas, leakage audit, and semantic judge rubric.
- Run smoke on 7B Instruct, not Think.

Day 2:

- Implement direction construction, layer sweep, and train/validation split discipline.
- Run neutral topic and register directions on 7B Instruct.
- Inspect false positives and marker/semantic disagreements by hand.

Day 3:

- Implement B2 injection screen and behavioral-potency checks.
- Add bootstrap, permutation null, and dose-response tables.
- Do not interpret a B2 hit as introspection. It is the steering smoke alarm.

Day 4:

- Implement B4 matched-output KV replay and fresh-transcript control.
- Unit-test that canonical answer token IDs are identical across conditions.
- Verify injection is off during the attribution question.

Day 5:

- Implement B5 injection detection with yes/no logit margin, d-prime, content-leak scoring, and task-degradation checks.
- Run OLMo 7B/32B Instruct pilot on register and neutral topic.

Day 6:

- Run full 32B Instruct seed 0 for B2/B3/B4/B5.
- Label ambiguous specimens blind.
- Freeze only pre-registered bug fixes, not result-driven changes.

Day 7:

- Run seeds 1 and 2 for headline B3/B4/B5 families.
- Implement patch recovery on strongest positive and strongest potent negative.
- Prepare Think/gpt-oss hook-runner smoke only after the Instruct instrument is boringly reliable.

## 27. The final inference table

Use this table to decide what the experiment showed.

| Decode | Potent intervention | B2 concept report | B3 entropy dissociation | B4 matched source | B5 anomaly detection | Interpretation |
|---|---|---|---|---|---|---|
| no | no | no | n/a | n/a | n/a | no handle found |
| yes | no | no | n/a | n/a | n/a | decodable but not causally active; dead instrument for intervention claims |
| yes | yes | no | no | no | no | behavior handle, shallow report channel under this instrument |
| yes | yes | yes | no | no | no | self-report is steerable, but propagation-explicable |
| yes | yes | yes | no | no | yes | functional anomaly-monitoring handle; concept identity may still be propagation |
| yes | yes | yes/no | yes | no | no/yes | functional source-monitoring handle under matched visible output |
| yes | yes | yes/no | yes | yes | yes | strongest audited functional report-channel coupling package |
| yes | yes | yes | fails because source labels false-alarm | no | no | prompt/option prior, not hidden monitoring |
| yes | yes | yes | fresh-transcript also succeeds | no | no | visible transcript/style prior, not hidden causal route |
| yes | yes | no | no | no, but content leaks | no | concept leakage / B2 propagation wearing a B5 hat |

Even the strongest row does not say "experience." It says the gauge has a wire to a functional state or source trace. Whether that state is fuel, shadow, or stage-prop is a theory question outside this instrument.

---

## Appendix A. Minimal config skeleton

```yaml
model:
  id: allenai/Olmo-3.1-32B-Instruct
  revision: main
  dtype: bfloat16
  attn_implementation: flash_attention_2
  batch_size_capture: 1
  batch_size_generate: 1
  reasoning_mode: direct_short

splits:
  train: train
  validation: validation
  heldout: heldout

selection:
  direction_layer_rule: earliest_within_95pct_train_plateau_confirmed_on_validation
  injection_rule: smallest_dose_earliest_layer_within_95pct_validation_best
  heldout_once: true

tracks:
  b2_inject_screen:
    enabled: true
    role: screen_only
    controls: [zero, noop, random, shuffled, wrong_concept, wrong_position, opposite, wrong_layer_supporting]
  b3_certainty:
    enabled: true
    require_entropy_dissociation: true
  b4_source_attribution:
    enabled: true
    matched_output_replay: kv_preserved
    fresh_transcript_control: true
    inject_during_attribution: false
  b5_injection_detection:
    enabled: true
    positive_conditions: [target_direction, wrong_concept_direction, random_direction]
    negative_conditions: [zero, noop]
    require_content_blind: true

scoring:
  lexical: true
  semantic_judge: true
  human_blind_labels_fraction: 0.20
  judge_human_min_agreement: 0.85

statistics:
  bootstrap_items: 10000
  permutation_nulls: 5000
  fdr: benjamini_hochberg_0_05
```

## Appendix B. Shovel-ready implementation order

Implement in this exact order if you want the fastest path to interesting results:

```text
1. instrument_proof
2. data/leakage/scoring tests
3. directions + potency
4. B2 screen
5. B4 matched-output source attribution
6. B5 injection detection
7. B3 entropy-dissociated certainty
8. patch/ablation localization
9. Think reasoning-axis
10. gpt-oss scale
```

The first genuinely publishable-looking insight is likely one of these:

```text
B2 positive but B4/B5 negative -> self-report is steerable but not source-aware
B4 positive -> functional source-monitoring exists under matched visible output
B5 positive -> hidden perturbation presence is report-accessible
B3 dissociated -> confidence reports can be moved independently of output entropy
Potent intervention + all reports negative -> defensible functional shallowness for that family
```
