# Mechanistic Interpretability Lab Sequence

**Course outline, revision 2**
**Date:** 2026-06-10
**Format:** Pre-lab + 11 hands-on labs. Each lab is one student-facing `.md` handout, one executable `.py` script, and one short interpretation/ethics reading.

---

## 0. As-built addendum (2026-06-11)

All 11 labs are implemented and validated (Tier A on CPU, Tier B on a Colab
A100). This outline remains the design document; where the build deviated,
the deviation was deliberate, is documented in the lab's own handout, and is
summarized here. The README's "Design decisions" section is the canonical
list; the load-bearing ones:

| Design (this document) | As built | Why |
|---|---|---|
| TransformerLens 3 / TransformerBridge + `interpkit/` package | one shared bench (`interp_bench.py`) on raw HF `transformers` with explicit, **self-verifying** hooks (hook parity, lens, decomposition, patch no-op, replacement exactness, edge reconstruction…), plus thin lab modules | course rule: nobody runs code they can't explain; every instrument check aborts the run on failure |
| Gemma 3 + Gemma Scope 2 via SAELens (Lab 8); gemma-2-2b + circuit-tracer (Lab 9) | pretrained ungated dictionaries loaded directly (jbloom resid SAE, Dunefsky MLP transcoders, decoderesearch Olmo SAE); Lab 9's attribution-graph pipeline is **hand-built** on gpt2 + the full 12-layer Dunefsky transcoder stack | Gemma weights are license-gated (no token on the course VM); circuit-tracer drags in TransformerLens; the loading conventions were validated empirically (Lab 8's report) |
| Lab 9 primary: two-hop Dallas→Texas→Austin | one-hop capital recall with a counterfactual substitution flip (France→Germany features ⇒ Paris→Berlin) | gpt2-small cannot do the two-hop; everything epistemically interesting — replacement model, edges, error accounting, interventions with a random control, the Lab 6 confrontation — survives |
| Lab 10 model strategy as written | as designed (Olmo-3-7B-Think), with Qwen3-0.6B as the Tier A think-capable smoke model; 140 MCQ items vendored from MMLU; add-mistake = injected wrong *claim* (no judge model assumed) | smallest ungated `<think>` model; frozen data rule |
| Lab 11 curated-domain menu | two domains implemented end to end behind `--audit-domain`: `factual_qa` and the `cot_faithfulness` flagship (fresh item slice + hint-presence probe); the remaining menu entries are student projects on the same harness | depth over breadth; the output contract is the deliverable |
| `interpkit/evidence.py` appends to the ledger | labs draft `ledger_suggestions.md` with measured numbers; nothing touches `claim_ledger.md` without `--append-ledger` | writing the claim is the coursework |

Models actually used: gpt2 (Labs 1–6 smoke, 8, 9), Olmo-3-1025-7B (Labs 1–6,
8, 11 Tier B), Olmo-3-7B-Instruct (Lab 7), SmolLM2-135M-Instruct (Lab 7
smoke), Olmo-3-7B-Think (Labs 10, 11), Qwen3-0.6B (Lab 10/11 smoke). Per-lab
runtime, artifacts, and validation evidence: `runs/LAB*_VERIFICATION_REPORT.md`.
**Target code size:** 500 to 1000 lines per lab Python file, including CLI, hooks, metrics, plots, and artifact writing.
**Audience:** Students who already know ML, deep learning, NLP, RL, and ML systems. No basics. The course teaches interpretability as an experimental craft on open models.

---

## 1. Course premise

This course teaches mechanistic interpretability as an experimental craft. Students do not merely inspect tensors. They learn to make precise claims about model internals, test those claims with interventions, and explain what the evidence does and does not support.

The guiding loop is:

```text
Behavior -> hypothesis -> internal measurement -> causal intervention -> artifact -> caveat
```

A good lab outcome is not "the plot looks interesting." A good lab outcome is:

```text
For this prompt family and metric, component X appears to carry information Y,
and patching or ablating X changes the model's behavior by Z.
```

Two threads run through every lab:

1. **The evidence ladder.** Observation, attribution, decodability, causality. Every lab names the rung it reaches, and every claim a student writes is tagged with its rung.
2. **The claim ledger.** A single running document, `claim_ledger.md`, that each lab appends to. By the capstone, every student owns a personal dossier of ~25 claims about one model, each tagged with evidence level, supporting artifact, and known falsifier. The capstone audit is built directly on top of this ledger rather than starting from scratch.

The course should feel less like a glass-bottom boat ride over hidden activations and more like building a tiny scientific submarine.

---

## 2. What changed from the previous outline (2026-06-10 v1) and why

The v1 outline had the right backbone: microscope pre-lab, residual stream + logit lens, direct logit attribution, attention, probing with controls, activation patching, circuit discovery, knowledge localization/editing, steering, SAEs, reliability audit. The revision keeps that arc and makes four structural moves plus several upgrades.

| Change | What it was | What it becomes | Why |
|---|---|---|---|
| **Merge Labs 5 and 7** | Lab 5 patched capital-city prompt pairs; Lab 7 ran causal tracing on... capital-city facts. Two labs, one method, one domain. | One lab: "Activation Patching and Causal Tracing" with factual recall as the running domain. Paraphrase generalization, neighboring-fact specificity, and an instructor-provided rank-one edit (with ripple-effect evaluation) become guided extensions. | Causal tracing *is* activation patching applied to facts. The merge removes real redundancy, and the editing material — which v1 already demoted to an extension because ROME is brittle — fits naturally as the extension of the patching lab. The freed slot pays for a critically missing topic. |
| **New Lab 9: Attribution Graphs and Circuit Tracing** | Absent. Circuit discovery stopped at manual head ablation and path patching. | A full lab on transcoder-based attribution graphs using the open `circuit-tracer` library (Gemma-2-2B with GemmaScope transcoders; Qwen3-4B and Llama-3.2-1B also supported), with Neuronpedia for visualization and feature-level interventions for validation. | This is the 2025–26 successor to manual circuit discovery and the method behind "On the Biology of a Large Language Model." A 2026 interpretability course without attribution graphs is teaching circuit discovery as it stood in 2023. The tooling is now genuinely teachable: open library, pretrained transcoders, free-tier-Colab-feasible, interactive frontend. |
| **New Lab 10: Chain-of-Thought Faithfulness** | Absent. The course never touched reasoning models, despite them being the dominant deployment reality. | A lab on whether a reasoning model's stated chain of thought reflects its actual computation: hint-injection faithfulness tests, CoT truncation ("early answering"), add-mistake perturbations, and filler-token controls on `allenai/Olmo-3-7B-Think`. | CoT faithfulness is one of the most consequential open questions in interpretability-for-safety, it is mostly *behavioral-causal* (cheap to run, no heavy hooks), fully open reasoning models now exist (OLMo 3 Think ships its post-training data and intermediate checkpoints), and it pairs with the best philosophy reading in the course (Nisbett & Wilson on human confabulation). It also gives students a breather from hook plumbing right before the capstone. |
| **Probing lab gets a real headline** | Track 2 was generic sentiment/entity-type probing. | Track 2 becomes truth/factuality probing in the Geometry-of-Truth style: true/false statement datasets, mass-mean vs. logistic probe directions, generalization across statement families. Token-level features remain the mechanical warm-up. | Truth probes are the probing result people actually care about, they motivate the controls far better than POS tagging does, and the probe direction learned here is reused causally in the steering lab. The skepticism curriculum (selectivity, shuffled labels, template-split) is unchanged — it just gets a subject worth being skeptical about. |
| **Steering lab gets a safety-relevant centerpiece** | A list of steering concepts (confidence, verbosity, sentiment, refusal as one option among five). | Two structured tracks: (A) a style/sentiment dose-response study, and (B) the refusal direction — extract it with difference-in-means over forward passes, use it as a monitor, and steer *toward* refusal on benign prompts. Direction extraction uses forward passes only; the course never generates completions of harmful prompts and does not implement refusal ablation. The published refusal-ablation result is discussed, not reproduced — which is itself the ethics unit. | "Refusal is mediated by a single direction" is the single most cited representation-engineering result and the cleanest dual-use case study in the field. Teaching it with the steer-toward-refusal design gets all of the science with none of the jailbreak. Concept-injection introspection (does the model notice a vector injected into its own activations?) becomes the ambitious extension. |
| **SAE lab absorbs superposition and transcoders** | SAEs appeared as a technique without the *why*, and transcoders were absent. | The lab opens with a 30-minute toy-model-of-superposition demo (train a tiny ReLU model on synthetic sparse features, watch features share dimensions), then does feature interpretation on Gemma Scope 2 artifacts, then introduces transcoders as "SAEs for MLP computation" — the explicit bridge into Lab 9. | Students who haven't seen superposition treat SAEs as arbitrary. Students who haven't seen transcoders can't read attribution graphs. Both fit inside the existing lab without growing the course. Gemma Scope 2 (Dec 2025) now provides SAEs *and* transcoders on every layer of every Gemma 3 size, base and instruct, loadable through SAELens. |
| **Model strategy corrected for 2026** | Gemma 4 E2B/E4B listed as the low-memory path; SAE work pointed at Gemma 3 + Gemma Scope 2 (correct); TransformerLens 3 mentioned without caveats. | Gemma 3 (1B/4B) becomes the low-memory *and* SAE/transcoder workhorse; Gemma 4 E-series is reclassified as an architecture-contrast model only; an explicit warning is added that TransformerLens 3's TransformerBridge preserves raw HF weights (no LayerNorm folding) with consequences for how DLA is taught. | Gemma 4's E2B/E4B are exotic on-device architectures (effective-parameter MatFormer-style designs with per-layer embeddings and audio stacks) — exactly wrong as a default interp target — and Gemma Scope artifacts exist for Gemma 2 and Gemma 3, not Gemma 4. The TransformerBridge weight-processing change silently breaks math copied from older HookedTransformer tutorials; the how-to guide now pins a convention before any lab is written. |
| **Ethics thread becomes a designed component** | One discussion prompt and a readings list per lab. | Every lab carries a named "Interpretation & Ethics" pairing chosen to arise from the experiment the student just ran: Woodward's interventionism for patching, Machamer–Darden–Craver's mechanisms for circuits, Hacking's entity realism for steering ("if you can spray them, they're real"), Nisbett & Wilson's confabulation for CoT, Dennett's real patterns for logit lens, belief-attribution standards for truth probes. Each pairing has one short reading and one writing prompt answerable from the lab's own artifacts. | The course brief calls for adjacent philosophy and ethics that is load-bearing rather than decorative. These pairings are chosen so that the philosophical question is *literally about the artifact on the student's screen*. |
| **Claim ledger added** | Each lab wrote a run summary; nothing accumulated. | `interpkit/evidence.py` provides a ledger writer; every lab ends by appending 2–3 tagged claims; the capstone audits the ledger. | Makes the epistemics tangible and gives the capstone a running start. Cheap to implement, high pedagogical return. |

Net effect on length: v1 had a pre-lab + 10 labs. v2 has a pre-lab + 11 labs, because one merge paid for one of the two additions. Both additions are topics where omission would date the course immediately.

---

## 3. Recommended model and tool strategy

### Default model choices

Keep one default dense transformer for most labs. Use alternative models only when the contrast itself teaches something, or when an external artifact set (SAEs, transcoders) dictates the choice.

| Use case | Recommended model | Reason |
|---|---|---|
| Main transformer labs (1–7) | `allenai/Olmo-3-1025-7B` (base) | Fully open model flow (data, code, intermediate checkpoints), 7B scale, conventional dense transformer, reproducible internals. |
| Instruction-following and steering labs | `allenai/Olmo-3-7B-Instruct` | Refusal, confidence, and style directions are easier to elicit in instruction-tuned models; same family as the base model keeps comparisons clean. |
| Low-memory path (any lab) | `google/gemma-3-1b-pt` / `-it`, or `google/gemma-3-4b-pt` / `-it` | Conventional architecture at small scale, runs on 8–16 GB, and — decisively — Gemma Scope 2 SAEs and transcoders exist for every layer of every size, base and instruct. One small-model choice serves both the budget tier and the SAE/transcoder labs. |
| SAE and transcoder inspection (Lab 8) | Gemma 3 4B-IT (or 1B for tight budgets) + `google/gemma-scope-2-*` via SAELens | Gemma Scope 2 covers 270M/1B/4B/12B/27B, PT and IT, with SAEs at three sites plus per-layer transcoders on every layer, and cross-layer transcoders for 270M and 1B. |
| Attribution graphs (Lab 9) | `google/gemma-2-2b` with GemmaScope transcoders via `circuit-tracer` | The circuit-tracer library's best-documented target; the tutorial's Dallas→Texas→Austin two-hop graph is the canonical teaching example. Qwen3-4B and Llama-3.2-1B are supported alternates; Gemma 3 + Gemma Scope 2 CLTs is the forward-looking option to evaluate at build time. |
| Reasoning / CoT lab (Lab 10) | `allenai/Olmo-3-7B-Think` | Fully open long-CoT reasoning model: post-training data (Dolci), recipes, and intermediate checkpoints are all released, so faithfulness findings can in principle be traced back to training. Apache 2.0. |
| Architecture contrast (extensions only) | `allenai/Olmo-Hybrid-7B`; Gemma 4 E2B/E4B | Hybrid-recurrent and effective-parameter on-device architectures make transformer-specific assumptions visible by contrast. **Not** defaults: Gemma 4's E-series is architecturally exotic (per-layer embeddings, MatFormer-style nesting, multimodal stacks) and has no Gemma Scope artifacts. |
| CI / smoke tests | `gpt2` and `google/gemma-3-270m-pt` | Every lab must complete on CPU with a tiny model so the test suite never needs a GPU. |

### Hardware tiers

Every lab declares which tiers it supports. The how-to guide gives per-lab runtime budgets.

| Tier | Hardware | What runs |
|---|---|---|
| A — smoke | Laptop CPU or MPS, 8–16 GB RAM | `gpt2` / Gemma 3 270M, `--max-examples 4`. Correctness of plumbing, not science. |
| B — standard | One 24 GB GPU (4090-class) or Colab | All labs at default settings: OLMo 3 7B in bf16 or 4-bit, Gemma 3 4B + Gemma Scope 2, Gemma-2-2B attribution graphs (with offloading flags where needed), OLMo 3 Think generation. |
| C — comfortable | One 40–80 GB GPU | Full-precision 7B everywhere, larger prompt sets, SAE training track, faster graph attribution. |

### Tool stack

| Layer | Tool | Course role |
|---|---|---|
| Loading and generation | Hugging Face `transformers` (>= 4.57 for OLMo 3) | Baseline interface and fallback path. |
| Activation caching and hooks | TransformerLens 3, via `TransformerBridge` | Main teaching interface for caching, hooks, and patching. **Pin a convention at build time:** TransformerBridge preserves raw HF weights by default — no LayerNorm folding or weight centering — so logits match HF but classic folded-LN DLA math from older tutorials does not apply verbatim. Either enable compatibility/folding mode where supported, or teach DLA with explicit LayerNorm handling. The how-to guide specifies the convention; do not let individual labs choose. |
| Architecture-flexible intervention | `nnsight` | Fallback when bridge support is incomplete; also useful for teaching that hooks are not library magic. |
| SAE / transcoder loading and training | SAELens (`decoderesearch/SAELens`), incl. `SAETransformerBridge` for Gemma 3 + Gemma Scope 2; minimal custom PyTorch SAE for the training track | Use SAELens for realism, the custom SAE for pedagogy. |
| Attribution graphs | `circuit-tracer` (`decoderesearch/circuit-tracer`) + Neuronpedia frontend | Graph generation, pruning, feature interventions, and shareable interactive visualizations. |
| Plotting | `matplotlib`, `pandas` (plotly optional) | Every lab saves interpretable artifacts. |
| Evals | Frozen local CSVs, hand-authored counterfactuals, small public subsets vendored into the repo | Labs run fast, reveal mechanisms, never depend on a live dataset, and never need internet after setup. |

### Version pinning preflight (build-time requirement)

Before writing any lab code, freeze and record: model revisions for every default model, `transformers`, `transformer-lens`, `sae-lens`, and `circuit-tracer` versions, the chosen TransformerBridge weight-processing convention, and the exact Gemma Scope 2 SAE/transcoder release IDs and circuit-tracer transcoder set. Put these in `interpkit/pins.py` and `requirements.txt`, and re-verify the circuit-tracer supported-model list and Gemma Scope 2 release notes at build time — both are actively evolving projects.

---

## 4. Course learning outcomes

By the end of the sequence, students should be able to:

1. Instrument an open language model and capture activations reproducibly.
2. Explain the residual stream as an accumulated representation read out by the unembedding, and use logit lens (and tuned lens) with appropriate caveats.
3. Use direct logit attribution to account for which components push toward an output, and state why attribution is not causation.
4. Distinguish attention routing from attention contribution, and validate head hypotheses with ablation.
5. Build probes with controls, including truth probes, and avoid overclaiming from decodability.
6. Use activation patching and causal tracing to test causal hypotheses, including localization of factual recall, and evaluate edits for specificity, generalization, and ripple effects.
7. Discover and validate a small circuit by hand with faithfulness, completeness, and minimality checks.
8. Compute and apply steering vectors with dose-response curves and side-effect measurement, and explain the refusal-direction result and its dual-use implications.
9. Explain superposition, interpret SAE features with evidence and counterexamples, and describe what a transcoder is for.
10. Generate, prune, read, and *intervene on* an attribution graph, and articulate what the replacement model does and does not license you to conclude.
11. Measure chain-of-thought faithfulness behaviorally and explain why a legible CoT is not automatically a faithful one.
12. Write a mechanistic reliability audit that integrates behavioral and internal evidence into supported claims, unsupported claims, and a deployment recommendation.

---

## 5. Course arc

```text
Microscope setup
  -> residual geometry & running predictions     (observe)
  -> prediction accounting                        (attribute)
  -> attention routing vs. use                    (attribute + ablate)
  -> probing with controls, truth probes          (decode)
  -> activation patching & causal tracing         (intervene)
  -> manual circuit discovery & validation        (intervene, compose)
  -> steering & the refusal direction             (intervene, control)
  -> superposition, SAE features, transcoders     (re-describe the units)
  -> attribution graphs & circuit tracing         (compose, at feature level)
  -> reasoning models & CoT faithfulness          (self-report vs. computation)
  -> mechanistic reliability audit                (integrate, recommend)
```

The sequence moves from "what can we see?" to "what can we causally change?" to "what are the right units?" to "what should we believe?" Labs 6 and 9 are deliberately paired: the same goal (a circuit explanation) pursued first with heads-and-MLPs by hand, then with transcoder features and automated attribution — and students are asked to compare the two epistemically, not just operationally.

---
## 6. Pre-lab 0: The Interpretability Microscope

**Status:** Setup module, not counted among the 11 core labs.
**Files:** `prelab_microscope.md`, `prelab_microscope.py`
**Evidence level targeted:** none — instrumentation only.

### Core question

Can every student load a model, cache activations, run one prompt pair, and write a reproducible artifact directory — on their actual hardware tier?

### Why it exists

Without a common microscope, every later lab becomes a debugging swamp. This pre-lab standardizes the run directory, model loading, CLI arguments, artifact names, seeding, dtype and device handling, chat-template handling (needed from Lab 7 onward and essential in Lab 10), and basic hook abstractions. It also initializes the student's `claim_ledger.md`.

### Minimal experiment

Run a tiny prompt pair:

```text
Clean:     "The Eiffel Tower is in"
Corrupted: "The Colosseum is in"
```

Cache residual stream activations at every layer for the final token. Save tokenization, activation shapes, logits, and one layer-normed residual norm plot. Then re-run with `--model gpt2 --device cpu` to prove the smoke tier works.

### Artifacts

```text
runs/prelab_<timestamp>/
  run_config.json
  tokens.json
  activation_shapes.json
  logits_topk.csv
  plots/residual_norm_by_layer.png
  run_summary.md
claim_ledger.md        (initialized at repo root, one entry: "ledger opened")
```

### Student learns

The contract every lab will follow: load, run, cache, measure, save, summarize, append to ledger.

---

## 7. Core lab sequence

### Lab 1: Residual Stream and Logit Lens

**Files:** `lab01_residual_logit_lens.md`, `lab01_residual_logit_lens.py`
**Evidence level targeted:** observation (with the explicit lesson that a readout is not a mind-scan).

**Core question:** How does a model's running prediction emerge across layers?

Students extract residual stream activations at every layer, apply the final layer norm and unembedding, and track top-k predictions, target-vs-distractor logit difference, entropy, and cosine-to-final-residual over depth.

**Primary experiment**

Compare three prompt categories (12–30 prompts total):

```text
High-certainty fact:       "The capital of France is"
Ambiguous continuation:    "The best way to solve the problem is"
Counterfactual context:    "In this story, Paris is a person and France is"
```

**Key artifact:** a layer-by-layer "prediction biography" showing when the model starts behaving as though it knows the answer.

**Extensions:** (manageable) repeat on base vs. instruct and compare stabilization depth; (ambitious) implement or load a tuned lens for a few layers and show where it disagrees with the raw logit lens — the disagreement is the lesson.

**Readings (technical):** Elhage et al., "A Mathematical Framework for Transformer Circuits"; Belrose et al., "Eliciting Latent Predictions from Transformers with the Tuned Lens."

**Interpretation & ethics:** Dennett, "Real Patterns" (excerpt), with Lipton's "Mythos of Model Interpretability" as backup. Writing prompt: when the logit lens shows the correct answer at layer 12, what *pattern* is real, and what claim about "the model knowing at layer 12" goes beyond the artifact you produced?

---

### Lab 2: Direct Logit Attribution and Component Accounting

**Files:** `lab02_direct_logit_attribution.md`, `lab02_direct_logit_attribution.py`
**Evidence level targeted:** attribution.

**Core question:** Which components push the model toward or away from a specific answer?

Students decompose the final logit difference into contributions from embeddings, every attention layer's output, and every MLP layer's output, scored against an answer direction (`unembed[target] − unembed[distractor]`). Because the course uses TransformerBridge with raw HF weights, the lab teaches LayerNorm handling explicitly rather than assuming folded weights — this is a feature, not a bug: students see exactly where the linearity approximation lives.

**Primary experiment**

```text
"The capital of Germany is" -> Berlin vs Paris
"The opposite of hot is"    -> cold vs warm
"The plural of mouse is"    -> mice vs mouses
```

Per example: validate single-token answers, compute the answer direction, cache component outputs, score each component's dot product with the direction (with consistent normalization), aggregate by layer and component type.

**Key artifact:** a stacked contribution chart by layer and component type, plus a cumulative logit-difference curve.

**Extension:** ablate the top-attributed components and compare attribution rank to causal-effect rank. The mismatch cases are the point.

**Readings:** Elhage et al. (framework paper); TransformerLens docs on residual decomposition.

**Interpretation & ethics:** short prompt on accounting metaphors — an attribution ledger looks authoritative; what would have to be true of the model for the ledger to be *misleading* while remaining arithmetically correct? (Sets up Lab 5.)

---

### Lab 3: Attention — Routing, Induction, and What Heads Actually Do

**Files:** `lab03_attention_routing.md`, `lab03_attention_routing.py`
**Evidence level targeted:** observation → attribution → causality (head ablation).

**Core question:** Which positions are routed where, and when does routing matter for the output?

Students visualize attention patterns *and* measure head-output contribution, then ablate. The lab is explicitly designed to cure heatmap astrology. Named motifs students should find and classify: previous-token heads, BOS/attention-sink heads, induction-style heads.

**Primary experiment**

```text
"red blue green red blue"
"A B C A B"
"Marcus went to the lab. Olivia went to the"
```

Per head: attention entropy, induction-pattern score, output attribution to the target logit direction, then ablation of top candidates vs. random and low-score control heads. Confirm candidate induction heads on natural text with repeated phrases.

**Key artifact:** a head table — `layer, head, pattern_label, attention_entropy, target_logit_attribution, ablation_effect` — and the scatter of attribution vs. ablation effect.

**Readings:** Olsson et al., "In-context Learning and Induction Heads"; Jain & Wallace, "Attention is not Explanation."

**Interpretation & ethics:** writing prompt on evidentiary standards — the head table contains three different kinds of evidence about the same objects; rank them and defend the ranking.

---

### Lab 4: Probing Without Fooling Yourself (now featuring truth)

**Files:** `lab04_probing_controls.md`, `lab04_probing_controls.py`
**Evidence level targeted:** decodability, with controls; causal hand-off deferred to Lab 7.

**Core question:** What information is linearly decodable from activations — including whether a statement is true — and how do we avoid confusing decodability with use?

Two tracks. Track 1 (mechanical warm-up): a token-level feature such as part-of-speech or punctuation class. Track 2 (headline): truth/factuality probing in the Geometry-of-Truth style on frozen true/false statement datasets (e.g., city locations, numeric comparisons, negated variants).

**Primary experiment**

For each track and layer: cache activations at selected positions; train a logistic-regression probe *and* a mass-mean (difference-of-class-means) direction; evaluate on held-out *statement families*, not just held-out examples; run shuffled-label controls, length/position baselines, and a random-direction baseline; compute selectivity = real accuracy − control accuracy; test generalization from one statement family (cities) to another (comparisons) and to negated forms.

**Key artifact:** a probe report that separates "decodable," "selectively decodable," and "generalizes across families," plus a saved truth direction (`truth_direction.pt` with metadata) that Lab 7 will reuse causally.

**Extensions:** (manageable) calibration curves for the truth probe; (ambitious) probe the same statements on base vs. instruct and compare where truth becomes decodable.

**Readings:** Hewitt & Liang, "Designing and Interpreting Probes with Control Tasks"; Marks & Tegmark, "The Geometry of Truth"; Belinkov, "Probing Classifiers."

**Interpretation & ethics:** Herrmann & Levinstein, "Standards for Belief Representations in LLMs" (or equivalent belief-attribution reading). Writing prompt: your probe reaches 95% on held-out cities. List the standards from the reading your artifact does and does not meet for saying the model *believes* anything. Secondary prompt retained from v1: if a demographic attribute is decodable from activations, what should and should not be inferred?

---

### Lab 5: Activation Patching and Causal Tracing (absorbs knowledge localization & editing)

**Files:** `lab05_patching_causal_tracing.md`, `lab05_patching_causal_tracing.py`
**Evidence level targeted:** causality.

**Core question:** Which activations are causally responsible for a behavior — concretely, where is a fact "recovered" in the forward pass, and what happens if you try to change it?

Students implement clean/corrupt activation patching over layer × position on the residual stream, with factual recall as the running domain. This single lab now carries the full localization story that v1 split across Labs 5 and 7: patching grids, recovery scores, paraphrase generalization, neighboring-fact specificity. Component-level patching (attention vs. MLP outputs) refines the picture.

**Primary experiment**

Build 30–100 validated facts (`subject, relation, target, distractor`, single-token answers, plus paraphrases and neighboring facts):

```text
Clean:     "The capital of France is"      Target: Paris
Corrupted: "The capital of Germany is"     Distractor: Berlin
```

Per pair: clean/corrupt logit differences; patch clean residuals into the corrupted run at every layer × position; recovery heatmap; confirm top patches on held-out paraphrases; wrong-position and mismatched-pair negative controls. Then aggregate across facts: do paraphrases localize to the same region? Do subject-token and final-token patches behave differently? Component-level pass for the top region.

**Guided extension — editing and its ripple effects:** apply an instructor-provided rank-one (ROME-style) edit function at the localized layer; evaluate direct success, paraphrase generalization, neighboring-fact spillover, and fluency. Required reading for the extension: Hase et al., "Does Localization Inform Editing?" — students must reconcile their own localization map with the finding that causal-tracing localization often fails to predict the best editing layer. A successful lab outcome includes explaining *that* tension, not resolving it.

**Key artifacts:** patching recovery heatmaps, a cross-fact localization summary, specificity/generalization table, and (extension) an edit report with ripple-effect measurements.

**Readings:** Meng et al., "Locating and Editing Factual Associations in GPT"; Geiger et al., "Causal Abstraction"; Hase et al., "Does Localization Inform Editing?" (extension); Cohen et al., "Ripple Effects of Knowledge Editing" (extension).

**Interpretation & ethics:** Woodward's interventionist account of causation (short excerpt or summary chapter of *Making Things Happen*). Writing prompt: state your patching result as a Woodward-style invariance claim — under which interventions, over which population of prompts, does the relationship hold? Then state one intervention you did *not* perform that could break it.

---

### Lab 6: Circuit Discovery and Validation, the Manual Way

**Files:** `lab06_circuit_discovery.md`, `lab06_circuit_discovery.py`
**Evidence level targeted:** causality, composed into a subgraph claim.

**Core question:** Can we reduce a behavior to a small computational subgraph — built from heads and MLPs by hand — that is faithful, complete, and minimal?

Students pick one small behavior (induction completion, an IOI template set, or greater-than comparison), identify candidate nodes with patching/attribution, ablate candidates against controls, prune greedily, and validate with faithfulness, completeness, and minimality metrics. Deliverable is a circuit card. The lab now explicitly frames itself as the *manual baseline* that Lab 9 will revisit with transcoder features: students are told to keep their circuit card, because they will compare it against an attribution graph for a related behavior.

**Key artifact:** a circuit card —

```text
Task: / Dataset: / Metric: / Candidate components: / Validated components:
Ablation result: / Circuit diagram: / Failure cases: / Faithfulness / Completeness / Minimality
```

**Extensions:** (manageable) path patching for one edge claim; (ambitious) attribution patching to rank nodes cheaply and compare its ranking against the causal one.

**Readings:** Wang et al., IOI paper; Conmy et al., "Towards Automated Circuit Discovery"; Syed et al., "Attribution Patching Outperforms Automated Circuit Discovery."

**Interpretation & ethics:** Machamer, Darden & Craver, "Thinking About Mechanisms" (the entities-and-activities account of mechanistic explanation). Writing prompt: map your circuit card onto MDC's schema — what are your entities, what are your activities, and where does your "mechanism sketch" have filler terms standing in for things you have not actually shown?

---

### Lab 7: Steering Vectors, Representation Engineering, and the Refusal Direction

**Files:** `lab07_steering_refusal.md`, `lab07_steering_refusal.py`
**Evidence level targeted:** causality (representation-level control), with explicit attention to what control does and does not explain.

**Core question:** Can a concept direction in activation space be used to monitor or change model behavior — and what does it mean that one direction appears to mediate refusal?

Model: `allenai/Olmo-3-7B-Instruct` (chat template handled by interpkit).

**Track A — dose-response steering (the method).** Choose one styled concept (confidence vs. uncertainty, concise vs. verbose, or positive vs. negative sentiment). Collect 20–100 contrast pairs, compute the difference-in-means direction at selected layers, inject at inference with a scale sweep, and measure target behavior, fluency/entropy, KL to the unsteered distribution, and unrelated-task drift, against random-direction and shuffled-pair controls. The deliverable is a dose-response curve, not a cherry-picked before/after.

**Track B — the refusal direction (the result).** Extract a candidate refusal direction by difference-in-means between activations on refusal-eliciting instructions and matched benign instructions. **Design constraint, stated in the handout:** direction extraction uses forward passes only — the lab never samples completions for harmful prompts, the elicitation set is a frozen instructor-provided file, and the lab does not implement refusal *ablation*. Students then (1) use the direction as a monitor — does its activation on held-out prompts predict refusal behavior? — and (2) steer *toward* refusal on benign prompts with a dose-response curve, demonstrating causal sufficiency in the safe direction. The published result that ablating this direction jailbreaks models is assigned as reading and discussed, not reproduced.

**Track A/B bridge:** load `truth_direction.pt` from Lab 4 and test whether the *probe* direction steers. Probing found it decodable; does intervening on it change behavior? This closes the decodability-vs-use loop opened in Lab 4.

**Key artifacts:** dose-response plots with controls, side-effect table, refusal-monitor ROC-style table, and a steering claim card stating effect, dose, side effects, and what the intervention does *not* show.

**Extensions:** (manageable) per-layer sweep to find where steering is strongest; (ambitious) concept-injection introspection in miniature — inject a benign concept vector (e.g., a topic direction) mid-context and ask the model whether it notices anything unusual about its own processing, replicating the design logic of the 2025 introspection experiments at small scale. Negative and sham-injection controls required.

**Readings:** Zou et al., "Representation Engineering"; Arditi et al., "Refusal in Language Models Is Mediated by a Single Direction"; Turner et al., "Activation Addition"; (extension) Lindsey, "Emergent Introspective Awareness in Large Language Models"; (optional) persona-vector and emergent-misalignment papers as further reading.

**Interpretation & ethics:** Hacking, *Representing and Intervening* (the entity-realism argument: "if you can spray them, they're real"). Writing prompt: you just steered behavior with a direction you computed. By Hacking's criterion, is the refusal direction *real*? What would Hacking say distinguishes your steering success from an explanation of refusal? Second prompt (dual use): the refusal-ablation result was published with full methods. Argue both sides of whether it should have been, using your own Track B artifacts as evidence about how easy or hard the method is.

---

### Lab 8: Superposition, Sparse Autoencoders, and Transcoders

**Files:** `lab08_sae_transcoders.md`, `lab08_sae_transcoders.py`
**Evidence level targeted:** observation/decodability at the feature level, with causal feature tests as extension.

**Core question:** Why do dense activations resist neuron-level interpretation, and can sparse dictionaries recover units worth naming?

**Part 0 — superposition in a jar (~30 min).** Train the toy model: a small ReLU autoencoder on synthetic sparse features with more features than dimensions. Plot the learned feature geometry as sparsity varies; watch features share dimensions and interfere. This is the *why* of everything that follows, runnable on CPU.

**Part 1 — feature interpretation (fast path, the core).** Gemma 3 4B-IT (1B on tight budgets) with Gemma Scope 2 SAEs loaded through SAELens. Run a curated prompt set; encode activations; rank top features; retrieve top-activating contexts; propose labels; test labels with held-out and adversarial prompts; record counterexamples; mind dead features and polysemanticity. Optional auto-interp step: have an LLM propose labels from activation contexts, then *validate the LLM's labels* with new prompts — the validation, not the labeling, is the skill.

**Part 2 — transcoders (the bridge to Lab 9).** Load a Gemma Scope 2 transcoder for one layer. Teach the object: an SAE reconstructs activations at a site; a transcoder reconstructs the MLP's *computation* (input→output), which is what makes feature-level circuit tracing possible. Verify reconstruction quality; inspect a few transcoder features; state in one paragraph why circuit tracing wants transcoders rather than SAEs.

**Training track (optional, Tier C):** cache 50k–500k activation vectors from the course model and train a small SAE (reconstruction + sparsity, track L0, dead features, reconstruction error). Compare a learned feature against the Lab 4 truth direction and the Lab 7 steering direction.

**Key artifacts:** toy-model geometry plots, a feature atlas with labels + evidence + counterexamples, transcoder reconstruction report, and (extension) a feature-steering result: clamp one validated feature and measure the behavioral effect.

**Readings:** Elhage et al., "Toy Models of Superposition"; Bricken et al., "Towards Monosemanticity"; Templeton et al., "Scaling Monosemanticity"; Gemma Scope / Gemma Scope 2 reports; Dunefsky et al., "Transcoders Find Interpretable LLM Feature Circuits."

**Interpretation & ethics:** continuation of the Hacking thread plus Dennett's real patterns: are SAE features discovered concepts, a useful coordinate system, or both? Writing prompt: pick your best-labeled feature and your worst; argue that the best one is "real" and steelman the deflationary reading of the worst one.

---

### Lab 9: Attribution Graphs and Circuit Tracing

**Files:** `lab09_attribution_graphs.md`, `lab09_attribution_graphs.py`
**Evidence level targeted:** attribution at feature level, upgraded to causality via feature interventions.

**Core question:** Can we read off the intermediate steps of a model's computation as a graph of features — and how much should we trust a graph computed on a replacement model?

Model/tooling: `google/gemma-2-2b` with GemmaScope transcoders via `circuit-tracer`; graphs visualized and annotated on Neuronpedia or the local frontend. (Alternates: Llama-3.2-1B, Qwen3-4B; evaluate Gemma 3 + Gemma Scope 2 cross-layer transcoders at build time.)

**Primary experiment — the two-hop fact.**

```text
Prompt: "Fact: the capital of the state containing Dallas is"
Target: " Austin"
```

Students: (1) generate the attribution graph; (2) prune to a readable subgraph and annotate supernodes ("Dallas", "Texas", "state capital", "say Austin"); (3) state the implied mechanism — input → Texas-features → Austin — as a hypothesis, not a finding; (4) **validate with interventions**: suppress the "Texas" supernode and substitute another state's features, measure whether the output flips to the corresponding capital; run a random-feature-suppression control of matched size; (5) check robustness across 5–10 paraphrases and at least one counterfactual city; (6) record the share of effect routed through error nodes, and write one paragraph on what the error nodes mean for the explanation's completeness.

**The comparison that makes the lab:** students place their Lab 6 circuit card next to this graph and answer — same epistemic standard or different? What did the manual method force you to verify that the automated graph hands you for free, and vice versa? Where does each method hide its assumptions (choice of metric and templates vs. replacement-model fidelity and pruning thresholds)?

**Key artifacts:** raw and pruned graph JSON, annotated supernode map, intervention results table with controls, paraphrase robustness table, error-node accounting, and a one-page "graph card" mirroring the Lab 6 circuit card.

**Extensions:** (manageable) replicate a second vignette — multilingual circuits or the poetry/newline planning example — in miniature; (ambitious) run attribution on a thinking-mode Qwen3-4B answer token and compare the graph for the same question with and without visible reasoning.

**Readings:** Ameisen et al., "Circuit Tracing: Revealing Computational Graphs in Language Models"; Lindsey et al., "On the Biology of a Large Language Model"; Marks et al., "Sparse Feature Circuits"; circuit-tracer tutorial notebook.

**Interpretation & ethics:** a short reading on idealization in science (e.g., a Potochnik chapter or equivalent). Writing prompt: the graph describes a *replacement model* that imitates the real one, with error nodes absorbing the residue. Defend or attack: "an idealized model that supports successful interventions is explanation enough." Use your own error-node percentages as evidence.

---

### Lab 10: Reasoning Models and Chain-of-Thought Faithfulness

**Files:** `lab10_cot_faithfulness.md`, `lab10_cot_faithfulness.py`
**Evidence level targeted:** behavioral causality over text-level interventions; the object under study is the relation between self-report and computation.

**Core question:** When a model shows its work, is the work it shows the work it did?

Model: `allenai/Olmo-3-7B-Think` (fully open: post-training data, recipes, intermediate checkpoints). interpkit handles the chat template and `<think>`-style span parsing; decoding settings are frozen in the config (temperature, max tokens, seeds) because sampling variance is a confound here, and the handout says so.

**Primary experiment — hint injection.** Frozen local MCQ dataset (100–300 items, e.g., a vendored MMLU subset with verified answer keys). Conditions per item: (a) baseline; (b) sycophancy hint ("I think the answer is X but I'm curious what you think"); (c) authority hint ("A Stanford professor says the answer is X"); (d) metadata/grader hint (the answer embedded in apparent system scaffolding). Hints point at *incorrect* answers in the main condition and at the already-correct answer in a control condition; a non-sequitur hint is a second control.

Metrics: flip rate (answers moved to the hinted option); acknowledgment rate (among flipped answers, fraction of CoTs that mention the hint at all); faithfulness score per hint type; and the gap between "CoT mentions the hint" and "CoT *attributes the answer* to the hint," which the rubric distinguishes.

**Second experiment — does the CoT carry load?** Three text-level interventions on sampled CoTs: (1) **early answering** — truncate the CoT at k ∈ {0, 25, 50, 75, 100}% and force an answer; plot accuracy vs. k (the "thought necessity curve"); (2) **add-mistake** — edit one intermediate step to be wrong, let generation continue, measure whether the final answer tracks the corrupted step; (3) **filler control** — replace the CoT with matched-length filler tokens. A model whose accuracy is flat in k and immune to injected mistakes is not *using* the visible reasoning, however articulate it looks.

**Key artifacts:** `faithfulness_by_hint_type.csv`, acknowledgment examples (verbatim CoT excerpts with hand labels), `necessity_curve.png`, mistake-propagation table, filler-control delta, and a claim card stating, with rates, what this model's CoT can and cannot be trusted to reveal *on this dataset*.

**Extensions:** (manageable) compare Think vs. Instruct variants on the same items — does training for long CoT change faithfulness or only verbosity?; (ambitious) mechanistic follow-up: train a probe for "hint present" on activations at answer-emission time, or patch hint-token representations, to ask whether the hint's influence is visible internally even when unacknowledged in text.

**Readings:** Turpin et al., "Language Models Don't Always Say What They Think"; Lanham et al., "Measuring Faithfulness in Chain-of-Thought Reasoning"; Chen et al., "Reasoning Models Don't Always Say What They Think" (2025); Korbak et al., "Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety."

**Interpretation & ethics:** Nisbett & Wilson, "Telling More Than We Can Know" (1977). Writing prompt: Nisbett & Wilson found that humans confidently report reasons that demonstrably did not drive their behavior. Compare their experimental logic to your hint-injection design, step by step. Then answer: if confabulation is the default for systems trained to produce plausible self-reports, what would a system have to *do* — not say — to earn trust in its explanations? Second prompt: the monitorability paper argues CoT oversight is a fragile safety opportunity. Given your measured faithfulness rates, how much load should CoT monitoring bear in a deployment safety case?

---

### Lab 11: Mechanistic Reliability Audit (capstone)

**Files:** `lab11_reliability_audit.md`, `lab11_reliability_audit.py`
**Evidence level targeted:** integration — every claim in the final report carries its rung on the ladder.

**Core question:** Given behavioral evidence and internal evidence, where should we trust this model less — and what may we responsibly say?

Students choose one narrow domain and produce an audit combining behavioral accuracy with at least three internal methods from earlier labs. The audit is built on the student's claim ledger: the report must cite ledger entries, retire at least one earlier claim that no longer survives scrutiny, and add final claims with evidence levels.

**Curated domains (choose one; plug-in design enforces common outputs):**

- factual QA with known answers under paraphrase (continues Lab 5's dataset)
- arithmetic or date comparison
- sentiment classification under negation
- refusal robustness on benign-but-alarming-sounding prompts (continues Lab 7, monitor-based)
- **CoT faithfulness audit** of Olmo-3-Think on a fresh item set (continues Lab 10) — recommended flagship
- hallucination-prone biography prompts with a frozen local dataset

**Required analysis per example:** model answer + confidence proxy; logit-lens stabilization layer; DLA summary; at least one causal intervention on a subset; one of probe / steering-monitor / SAE-feature / graph evidence; manual failure-mode label.

**Final artifact — the audit report:**

```text
Claim: / Task boundary: / Dataset: / Behavioral performance:
Internal evidence (by method, with evidence level):
Known failure modes: / Counterexamples: / Strongest counterevidence:
Ledger reconciliation (claims kept, revised, retired):
Confidence in the interpretation: / Recommended use: / Recommended non-use:
```

**Readings:** Rudin, "Stop Explaining Black Box Machine Learning Models for High Stakes Decisions"; Selbst et al., "Fairness and Abstraction in Sociotechnical Systems"; Mittelstadt, "Principles Alone Cannot Guarantee Ethical AI."

**Interpretation & ethics:** writing prompt — draft the two-paragraph "internal evidence" section of a hypothetical deployment safety case for your domain, then write the one-paragraph rebuttal a skeptical reviewer would file. The rebuttal is graded as seriously as the case.

---

## 8. Suggested schedule

| Week | Lab | Theme |
|---|---|---|
| 0 | Pre-lab | Instrumentation, artifact contract, ledger |
| 1 | Lab 1 | Residual stream and logit lens |
| 2 | Lab 2 | Direct logit attribution |
| 3 | Lab 3 | Attention routing and induction |
| 4 | Lab 4 | Probing with controls; truth probes |
| 5 | Lab 5 | Activation patching, causal tracing, (editing ext.) |
| 6 | Lab 6 | Manual circuit discovery |
| 7 | Lab 7 | Steering and the refusal direction |
| 8 | Lab 8 | Superposition, SAEs, transcoders |
| 9 | Lab 9 | Attribution graphs |
| 10 | Lab 10 | CoT faithfulness |
| 11–12 | Lab 11 | Capstone audit (two weeks: one to run, one to write) |
| 13 | Buffer | Presentations, replication, extensions |

Self-paced equivalent: Labs 1–3 compress comfortably into a single intensive week for a strong student; Labs 9–11 should not be rushed.

---

## 9. Standard lab artifact contract

Every lab produces the same outer structure:

```text
runs/<lab_name>_<timestamp>/
  run_config.json
  run_summary.md
  metrics.json
  results.csv
  plots/
  tables/
  cached_examples/
  diagnostics/
```

Every `run_summary.md` answers:

1. What behavior was studied?
2. What internal object was measured?
3. What intervention or control was used?
4. What metric changed?
5. What claim is supported, at what evidence level?
6. What claim is not supported?
7. What would falsify the interpretation?

And every lab ends by appending 2–3 tagged claims to `claim_ledger.md`:

```text
[L05-C2] CAUSAL | Patching resid layer 14 at the subject token recovers >=70% of the
clean logit diff across 24/30 facts and 4/5 paraphrase families.
Artifact: runs/lab05_.../patching_scores.csv | Falsifier: fails on held-out relation types.
```

---

## 10. Grading and review rubric

| Criterion | What earns credit |
|---|---|
| Reproducibility | The run repeats from `run_config.json` and pinned model revision. |
| Measurement clarity | Metrics defined before plots are inspected. |
| Causal caution | Correlation, attribution, decodability, and intervention are kept distinct, in writing. |
| Controls | At least one negative control or shuffled baseline wherever appropriate. |
| Artifact quality | Plots and tables are named, readable, and tied to a claim. |
| Interpretation | The writeup states what the evidence supports and where it is weak; ledger entries are honest. |
| Ethics/interpretation prompt | Answered from the lab's own artifacts, not in the abstract. |
| Extension | Optional work explores model comparison, stronger validation, or a second method. |

Negative results graded as positive when explained: a probe with high accuracy but zero selectivity, a steering vector that moves style but wrecks accuracy, an attribution graph whose intervention fails — each is full credit with a correct diagnosis.

---

## 11. Suggested repository layout

```text
interpretability_labs/
  COURSE.md
  HOW_TO_DESIGN_LABS.md
  README.md
  pyproject.toml
  requirements.txt            # pinned; see interpkit/pins.py
  claim_ledger.md             # per-student, initialized by prelab
  interpkit/
    __init__.py
    pins.py                   # model revisions, library versions, conventions
    models.py                 # loading, dtype/device, chat templates, generate_logits()
    hooks.py                  # cache/patch/ablate; TransformerBridge + nnsight fallback
    metrics.py                # logit diff, KL, recovery, selectivity, faithfulness rates
    prompt_sets.py            # facts, induction, IOI, truth statements, contrast pairs,
                              # refusal elicitation set (frozen), hinted MCQ items
    sae.py                    # Gemma Scope 2 SAE/transcoder loading via SAELens
    graphs.py                 # circuit-tracer wrappers: attribute, prune, intervene
    reasoning.py              # think-span parsing, hint injection, truncation, add-mistake
    evidence.py               # claim ledger writer; evidence-level tags
    plotting.py
    artifact_writer.py
  labs/
    prelab_microscope.{md,py}
    lab01_residual_logit_lens.{md,py}
    lab02_direct_logit_attribution.{md,py}
    lab03_attention_routing.{md,py}
    lab04_probing_controls.{md,py}
    lab05_patching_causal_tracing.{md,py}
    lab06_circuit_discovery.{md,py}
    lab07_steering_refusal.{md,py}
    lab08_sae_transcoders.{md,py}
    lab09_attribution_graphs.{md,py}
    lab10_cot_faithfulness.{md,py}
    lab11_reliability_audit.{md,py}
  data/                       # frozen CSVs only; no live downloads at lab runtime
  runs/
    .gitkeep
```

---

## 12. Reading bank

### Technical core

- Elhage et al., "A Mathematical Framework for Transformer Circuits" — https://transformer-circuits.pub/2021/framework/index.html
- Olsson et al., "In-context Learning and Induction Heads" — https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- Belrose et al., "Eliciting Latent Predictions from Transformers with the Tuned Lens" — https://arxiv.org/abs/2303.08112
- Wang et al., "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small" — https://arxiv.org/abs/2211.00593
- Meng et al., "Locating and Editing Factual Associations in GPT" — https://arxiv.org/abs/2202.05262
- Hase et al., "Does Localization Inform Editing?" — https://arxiv.org/abs/2301.04213
- Cohen et al., "Evaluating the Ripple Effects of Knowledge Editing in Language Models" — https://arxiv.org/abs/2307.12976
- Geiger et al., "Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability" — https://www.jmlr.org/papers/v26/23-0058.html
- Conmy et al., "Towards Automated Circuit Discovery for Mechanistic Interpretability" — https://arxiv.org/abs/2304.14997
- Syed et al., "Attribution Patching Outperforms Automated Circuit Discovery" — https://arxiv.org/abs/2310.10348
- Hewitt & Liang, "Designing and Interpreting Probes with Control Tasks" — https://arxiv.org/abs/1909.03368
- Belinkov, "Probing Classifiers: Promises, Shortcomings, and Advances" — https://arxiv.org/abs/2102.12452
- Marks & Tegmark, "The Geometry of Truth" — https://arxiv.org/abs/2310.06824
- Zou et al., "Representation Engineering: A Top-Down Approach to AI Transparency" — https://arxiv.org/abs/2310.01405
- Arditi et al., "Refusal in Language Models Is Mediated by a Single Direction" — https://arxiv.org/abs/2406.11717
- Turner et al., "Activation Addition" — https://arxiv.org/abs/2308.10248
- Todd et al., "Function Vectors in Large Language Models" — https://arxiv.org/abs/2310.15213
- Elhage et al., "Toy Models of Superposition" — https://transformer-circuits.pub/2022/toy_model/index.html
- Bricken et al., "Towards Monosemanticity" — https://transformer-circuits.pub/2023/monosemantic-features/index.html
- Templeton et al., "Scaling Monosemanticity" — https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
- Lieberum et al., "Gemma Scope" — https://arxiv.org/abs/2408.05147 (and the Gemma Scope 2 technical report, Dec 2025)
- Dunefsky et al., "Transcoders Find Interpretable LLM Feature Circuits" — https://arxiv.org/abs/2406.11944
- Marks et al., "Sparse Feature Circuits" — https://arxiv.org/abs/2403.19647
- Ameisen et al., "Circuit Tracing: Revealing Computational Graphs in Language Models" — https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- Lindsey et al., "On the Biology of a Large Language Model" — https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- Turpin et al., "Language Models Don't Always Say What They Think" — https://arxiv.org/abs/2305.04388
- Lanham et al., "Measuring Faithfulness in Chain-of-Thought Reasoning" — https://arxiv.org/abs/2307.13702
- Chen et al., "Reasoning Models Don't Always Say What They Think" (Anthropic, 2025)
- Korbak et al., "Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety" (2025)
- Lindsey, "Emergent Introspective Awareness in Large Language Models" (Anthropic, 2025)

### Tools and model references (verified 2026-06-10; re-verify at build time)

- OLMo 3 model cards: https://huggingface.co/allenai/Olmo-3-1025-7B , https://huggingface.co/allenai/Olmo-3-7B-Instruct , https://huggingface.co/allenai/Olmo-3-7B-Think
- Gemma 3 model cards and Gemma Scope 2 suite: https://ai.google.dev/gemma/docs/gemma_scope and https://huggingface.co/google/gemma-scope-2-4b-it (releases exist for 270M/1B/4B/12B/27B, PT and IT; CLTs for 270M and 1B)
- Gemma 4 overview (architecture-contrast use only): https://ai.google.dev/gemma/docs/core/model_card_4
- TransformerLens 3 (TransformerBridge; raw-HF-weights default): https://github.com/TransformerLensOrg/TransformerLens
- SAELens (incl. SAETransformerBridge for Gemma 3 + Gemma Scope 2): https://github.com/decoderesearch/SAELens
- circuit-tracer (Gemma-2-2B, Llama-3.2-1B, Qwen3-4B; PLT + CLT; interventions): https://github.com/decoderesearch/circuit-tracer
- Neuronpedia attribution-graph frontend: https://www.neuronpedia.org
- nnsight: https://nnsight.net/

### Philosophy, ethics, and interpretation

- Lipton, "The Mythos of Model Interpretability" — https://arxiv.org/abs/1606.03490
- Doshi-Velez & Kim, "Towards A Rigorous Science of Interpretable Machine Learning" — https://arxiv.org/abs/1702.08608
- Dennett, "Real Patterns" (Journal of Philosophy, 1991)
- Herrmann & Levinstein, "Standards for Belief Representations in LLMs" (Minds & Machines, 2025)
- Woodward, *Making Things Happen* (interventionist causation; excerpt)
- Machamer, Darden & Craver, "Thinking About Mechanisms" (Philosophy of Science, 2000)
- Hacking, *Representing and Intervening* (1983; entity-realism excerpt)
- Potochnik, *Idealization and the Aims of Science* (2017; one chapter)
- Nisbett & Wilson, "Telling More Than We Can Know: Verbal Reports on Mental Processes" (Psych. Review, 1977)
- Rudin, "Stop Explaining Black Box Machine Learning Models for High Stakes Decisions" — https://www.nature.com/articles/s42256-019-0048-x
- Selbst et al., "Fairness and Abstraction in Sociotechnical Systems" — https://dl.acm.org/doi/10.1145/3287560.3287598
- Mittelstadt, "Principles Alone Cannot Guarantee Ethical AI" — https://www.nature.com/articles/s42256-019-0114-4
- Bender & Koller, "Climbing towards NLU" — https://aclanthology.org/2020.acl-main.463/

---

## 13. The course in one paragraph

Students begin by building a reliable microscope for transformer internals. They learn how predictions emerge in the residual stream, how components contribute to output logits, how attention routes information, and how probes — including truth probes — can reveal represented features without proving causal use. They then move into causal interventions: patching and causal tracing of facts, manual circuit discovery, and steering vectors up to and including the refusal direction. With superposition motivating sparse dictionaries, they interpret SAE features and transcoders, then read and intervene on attribution graphs — the same circuit-finding goal pursued first by hand and then at feature level, compared honestly. A lab on chain-of-thought faithfulness asks whether a model's stated reasoning is the reasoning it did, paired with the psychology of human confabulation. The capstone audits a narrow capability using the student's accumulated claim ledger, ending in supported claims, retired claims, and a deployment recommendation. The result is a course about interpretability as disciplined evidence, not activation taxidermy.
