# Mechanistic Interpretability Labs

Hands-on labs that teach interpretability as an experimental craft on open
models. The course design lives in [COURSE.md](COURSE.md) and the lab-authoring
guide in [how_to_design_labs.md](how_to_design_labs.md). This README covers the
code: what runs, where, and how to read what it produces.

The project follows the same pattern as the collective-communication course:
**one shared bench script** (`interp_bench.py`) owns the experiment machinery —
CLI, run directories, console logging, diagnostics, model loading, hooks,
readouts, state dumps, plots — and **thin lab modules** under `labs/` own the
experiments. The bench is the microscope; the labs are what you point it at.

## Primary target

The primary target is **one NVIDIA A100/H100 (Colab)** running Hugging Face
`transformers` in bf16. Every lab also has a CPU smoke path (`--tier a`) that
must work on a laptop — debug there, spend GPU minutes on science.

| Tier | Hardware | What runs |
|---|---|---|
| A — smoke | laptop CPU (or MPS) | `gpt2` (base labs: 1–6, 12) / `SmolLM2-135M-Instruct` (chat/generation labs: 7, 13-18) / lab-specific small models; correctness of plumbing, not science |
| B — standard | Colab A100/H100, or any 24 GB+ GPU | base labs on `allenai/Olmo-3-1025-7B`; instruct labs (7+) on `allenai/Olmo-3-7B-Instruct`, bf16 |
| C — comfortable | 40–80 GB GPU | fp32, larger prompt sets |

Chat-template labs use instruct models because steering, refusal, reasoning,
and output-affect checks need real assistant behavior. The tier-A smoke model
switches to a small instruct model automatically where a lab needs it (the lab
registry owns the per-lab model override); generation makes these labs slower
than the forward-pass-only labs.

## Quick start

```bash
cd interpretability
pip install -r requirements.txt

# CPU smoke test (always do this first):
python interp_bench.py --lab lab1 --tier a

# Full Lab 1 on a Colab A100/H100 (--include-controls adds weak/scrambled control prompts):
python interp_bench.py --lab lab1 --tier b --prompt-set full --include-controls

# Lab 2 (direct logit attribution; --ablate-top N sets the attribution-vs-causal
# ablation count, 0 skips it):
python interp_bench.py --lab lab2 --tier a
python interp_bench.py --lab lab2 --tier b --prompt-set full --topk 10

# Lab 3 (attention routing; the bench auto-sets eager attention):
python interp_bench.py --lab lab3 --tier a
python interp_bench.py --lab lab3 --tier b --prompt-set full --topk 10

# Lab 4 (probing; --max-examples caps statements PER FAMILY here):
python interp_bench.py --lab lab4 --tier a
python interp_bench.py --lab lab4 --tier b --prompt-set full

# Lab 5 (patching + causal tracing; --run-edit adds the edit audit):
python interp_bench.py --lab lab5 --tier a
python interp_bench.py --lab lab5 --tier b --prompt-set full --run-edit

# Lab 6 (manual circuit discovery; deliverable is circuit_card.md):
python interp_bench.py --lab lab6 --tier a
python interp_bench.py --lab lab6 --tier b --prompt-set full

# Lab 7 (steering + refusal; uses instruct models, generation is slow):
python interp_bench.py --lab lab7 --tier a   # SmolLM2-135M-Instruct
python interp_bench.py --lab lab7 --tier b    # Olmo-3-7B-Instruct

# Lab 8 (SAEs + transcoders; pretrained dictionaries download on first run):
python interp_bench.py --lab lab8 --tier a   # gpt2 + jbloom SAE + Dunefsky transcoder
python interp_bench.py --lab lab8 --tier b    # Olmo-3-1025-7B + decoderesearch SAE

# Lab 9 (attribution graphs; gpt2 on EVERY tier — tiers scale the node budget):
python interp_bench.py --lab lab9 --tier a   # CPU-ok, ~2 GB transcoder download once
python interp_bench.py --lab lab9 --tier b    # bigger graph + full paraphrase battery

# Lab 10 (CoT faithfulness; generation-heavy — Tier B is ~35 min):
python interp_bench.py --lab lab10 --tier a  # Qwen3-0.6B think smoke, 3 items
python interp_bench.py --lab lab10 --tier b   # Olmo-3-7B-Think, 36 items x 6 conditions

# Lab 11 (capstone audit; --audit-domain picks the curated domain):
python interp_bench.py --lab lab11 --tier b                  # factual_qa on Olmo base
python interp_bench.py --lab lab11 --tier b \
  --audit-domain cot_faithfulness --model allenai/Olmo-3-7B-Think   # flagship
python interp_bench.py --lab lab11 --tier b --audit-domain sentiment_negation

# Lab 12 (first advanced lab: relation geometry; BASE models, probes + patching;
# --relation-set caps items per family, --patch-grid picks patched token roles):
python interp_bench.py --lab lab12 --tier a   # gpt2, ~15 s
python interp_bench.py --lab lab12 --tier b --relation-set full

# Lab 13 (emotion geometry; instruct models, read/write transfer + steering):
python interp_bench.py --lab lab13 --tier a   # SmolLM2-135M-Instruct
python interp_bench.py --lab lab13 --tier b --prompt-set full

# Lab 14 (certainty, hedging, and calibration; instruct models):
python interp_bench.py --lab lab14 --tier a   # SmolLM2-135M-Instruct
python interp_bench.py --lab lab14 --tier b --prompt-set full

# Lab 15 (multi-turn instrumentation; harness validation, not a science claim):
python interp_bench.py --lab lab15 --tier a   # SmolLM2-135M-Instruct
python interp_bench.py --lab lab15 --tier b

# Lab 16 (sycophancy and user-belief modeling; instruct models):
python interp_bench.py --lab lab16 --tier a   # SmolLM2-135M-Instruct
python interp_bench.py --lab lab16 --tier b --prompt-set full

# Lab 17 (persona, voice, roleplay, and register; instruct models):
python interp_bench.py --lab lab17 --tier a   # SmolLM2-135M-Instruct
python interp_bench.py --lab lab17 --tier b --prompt-set full

# Lab 18 (humor as incongruity; instruct models + eager attention):
python interp_bench.py --lab lab18 --tier a   # SmolLM2-135M-Instruct
python interp_bench.py --lab lab18 --tier b --prompt-set full

# Lab 19 (model diffing with crosscoders; Tier A is an identity-pair smoke):
python interp_bench.py --lab lab19 --tier a --no-plots
python interp_bench.py --lab lab19 --tier b --prompt-set full

# Lab 20 (benign model organisms; sealed manifests + baseline audit):
python interp_bench.py --lab lab20 --tier a --no-plots
python interp_bench.py --lab lab20 --tier b --prompt-set full
```

On Colab: `Runtime > Change runtime type > A100`, then in a cell:

```python
!git clone https://github.com/<you>/labs.git
%cd labs/interpretability
!pip install -q -r requirements.txt
!python interp_bench.py --lab lab1 --tier b --prompt-set full
```

## Current status

- `interp_bench.py` — shared bench: run dirs, console tee, diagnostics
  (packages/git/GPU/env), model anatomy resolution, residual-stream capture
  with verified semantics, logit lens, verified per-block component capture
  (attn/MLP contributions, post-norm aware), direct-path component ablation,
  human-readable state dumps, plots, claim-ledger plumbing. Implemented and
  validated on gpt2 (fp32) and Olmo-3-7B (bf16, A100).
- Lab 1: residual stream and logit lens — implemented and validated (Tier A+B).
- Lab 2: direct logit attribution — implemented and validated (Tier A+B).
  Adds two instrument self-checks: the component-anatomy probe (hook points
  are verified against per-block residual deltas, not assumed) and the
  decomposition check (components must sum to the final pre-norm stream).
- Lab 3: attention routing, head motifs, induction — implemented and
  validated (Tier A+B). Adds head-level capture (attention patterns require
  eager — the bench forces it, since sdpa returns an empty attentions tuple
  silently in transformers 5), a verified per-head decomposition check, and
  scoped head ablation (final-position vs all-position — the gap measures
  composition).
- Lab 4: probing with controls + the truth direction — implemented and
  validated (Tier A+B). Frozen statement families vendored in `data/`;
  dual probes (logistic + mass-mean) with shuffled/random/length controls
  and family-held-out transfer; saves `truth_direction.pt` for Lab 7's
  causal test. Activation-norm outliers are detected and recorded (one
  frozen statement produces a 7x-norm stream on Olmo-3 — see the lab
  handout's "outlier specimen" section).
- Lab 5: activation patching and causal tracing — implemented and validated
  (Tier A+B). Adds interchange interventions on the residual stream and on
  component outputs, a patch no-op self-check (self-patching must be a
  numerical identity), alignment-validated clean/corrupt fact pairs,
  role-aggregated causal tracing with paraphrase confirmation and negative
  controls, and a rank-one edit-and-audit extension (`--run-edit`).
- Lab 6: circuit discovery, the manual way — implemented and validated
  (Tier A+B). Composes Labs 2/3/5: cheap screening (attribution + motifs) →
  causal ranking (single-node mean-ablation) → greedy pruning → faithfulness
  / completeness / minimality on discovery AND held-out vocabulary families,
  one ablation-interaction edge claim, and a circuit card deliverable.
  Adds multi-node mean-ablation machinery to the bench
  (`run_with_node_set_ablation`).
- Lab 7: steering, representation engineering, and the refusal direction —
  implemented and validated (Tier A+B). First lab on **instruct models**:
  adds chat-template application, activation-addition steering hooks, and
  frozen-decoding generation to the bench. Track A (sentiment dose-response
  with random/shuffled controls), Track B (refusal monitor + steer-toward-
  refusal, forward-pass-only safety wall — no harmful generation, no
  ablation), and the bridge that loads Lab 4's truth direction. Steering
  scales are fractions of the activation norm, so a "dose" means the same
  thing across models.
- Lab 8: superposition, SAEs, and transcoders — implemented and validated
  (Tier A+B). Back to **base models** with **pretrained dictionaries** loaded
  from the Hub: gpt2 + jbloom resid SAE + Dunefsky MLP transcoder (Tier A),
  Olmo-3-1025-7B + decoderesearch jumprelu SAE (Tier B). Part 0 is a CPU toy
  model reproducing the canonical superposition collapse (dense → d_hidden
  orthogonal features, sparse → more in superposition). Part 1 is a feature
  atlas whose graded skill is **label validation** — held-out AUC against
  domain membership, an adversarial confusable-pair test (concept vs token
  feature), and polysemanticity scoring, with verdicts and a kept record of
  the labels that died. Part 2 verifies a transcoder by reconstruction FVU and
  the downstream-logit KL of splicing it in for the real MLP, then de-embeds
  features (the bridge to Lab 9). Plus a cosine bridge to Lab 4's truth
  direction and a CAUSAL feature-clamp (induce a concept at ~1× the feature's
  peak activation, with a random-feature control and a fluency proxy). The SAE
  loading conventions (TL centering, bare-LN transcoder input, jumprelu
  threshold) were each validated empirically, not assumed.
- Lab 9: attribution graphs and circuit tracing — implemented and validated
  (Tier A+B). The 2025-era successor to Lab 6, built **from scratch** (no
  circuit-tracer, no TransformerLens) on gpt2 with the **full 12-layer
  Dunefsky transcoder stack**: a local replacement model (frozen attention
  patterns + frozen LN denominators + transcoders + error nodes) that is
  exact to the real logits, direct-attribution edges by one backward pass
  per node with an enforced accounting identity (bias + Σedges = metric),
  backward-flow pruning, and graph-guided suppress/substitute interventions
  **on the real model** with a random matched control (the France→Germany
  substitution flips the capital). The signed edge ledger quantifies the
  Lab 6 confrontation: features pay +2.34 of the fact's +3.01 logit diff but
  almost none of the induction prompt's, where frozen attention routes
  copied embeddings the graph cannot show. Three new abort-on-failure
  self-checks (replacement exactness, edge reconstruction, feature-edit
  no-op). gpt2 on every tier — the only ungated model with a public
  all-layers transcoder set; tiers scale the node budget, not the model.
- Lab 10: reasoning models and CoT faithfulness — implemented and validated
  (Tier A+B). The course's first generation-heavy lab and first batched
  decoding in the bench: hint injection (sycophancy / authority / metadata
  at deterministic wrong answers, with correct-hint and non-sequitur
  controls) measuring flip / acknowledgment / attribution / **silent-flip**
  rates, plus three text-level load tests built on one close-the-think-span
  primitive — the thought-necessity curve, add-mistake, and a matched-length
  filler control. Frozen 140-item MCQ set vendored from MMLU; think-span
  round-trip self-check (Olmo's template opens the span, Qwen emits its
  own); forced-answer rescue with an unparseable log. Tier B finding:
  Olmo-3-Think's CoT carries load (0.56→0.88 over the curve, filler stuck at
  the floor), it argues itself back from injected mistakes, and flip rates
  fall 2–3× when the thinking budget is raised — truncation-forced answers
  are more hint-followable. `acknowledgment_labels.csv` ships with empty
  student columns: the hand labeling is the graded skill.
- Lab 11: mechanistic reliability audit (capstone) — implemented and
  validated (Tier A+B). A plug-in audit harness with a rigid output
  contract: the fixed-schema `audit_report.md`, a per-claim
  keep/revise/retire `ledger_reconciliation.md`, and
  `safety_case_and_rebuttal.md` — with every student-judgment section
  scaffolded but deliberately not generated. Three domains
  (`--audit-domain`): **factual_qa** (lens stabilization AND preference
  depths, DLA summary, two-site residual patching — recovery 0.995 at the
  early subject site vs 0.02 at the final band, Lab 5's localization
  reproduced by the audit — and a truth monitor at AUC 1.0 vs shuffled 0.0
  on Olmo), and the **cot_faithfulness flagship**, which reruns Lab 10
  verbatim on a deliberately fresh item slice (the audit as replication:
  flip rates 0.50 vs 0.18 across slices — item-set sensitivity surfaced)
  plus a hint-presence probe whose Tier B result is an honestly-shipped
  NEGATIVE (AUC ≈ shuffled; the claim builder is conditional on its own
  selectivity). A third domain, **sentiment_negation** (48 frozen statement
  pairs, `data/affect_valence.csv` + `data/affect_negation.csv`), audits
  whether the model — and a Lab-4-style plain-trained valence probe — reads
  surface valence words or the composed meaning under minimal negation
  edits: behavioral negation gap, negated-family probe transfer with a
  shuffled control, and plain-into-negated final-position patching with an
  unrelated-plain control. Audit lesson #1 baked in: Olmo prefers the right
  capital at 1.000 while top-1 "accuracy" reads 0.361 ("…is **known** as")
  — which behavioral metric does your claim name?
- Lab 12: relation geometry and method validation — implemented for the
  advanced course. Re-runs Lab 4/5-style tools on a 12-family controlled
  relation set, with relation-swap groups, direction cosines, patching
  transfer, and an operationalization audit.
- Lab 13: emotion geometry — implemented for the advanced course. Extracts
  comprehension and generation affect directions for joy/sadness/anger/fear,
  cross-tests read/write transfer, audits sentiment/arousal confounds, and
  runs a small input-derived steering check with random controls.
- Lab 14: certainty, hedging, and calibration — implemented for the advanced
  course. Builds an answerability/certainty direction over fixed A/B/C/D
  items, saves a separate hedging-style direction, compares internal
  projection with entropy/margin and generated verbal confidence, and writes
  the three-way disagreement matrix needed by later self-report labs.
- Lab 15: multi-turn instrumentation — implemented for the advanced course.
  Validates chat-template turn segmentation, cached boundary reads against
  full recompute, turn-boundary self-patching, and topic/null projection
  traces before later labs make persona, belief-revision, or self-report
  claims across turns.
- Lab 16: sycophancy and user-belief modeling — implemented for the advanced
  course. Builds a misconception-pressure battery, scores generated agreement
  with false user beliefs, separates local truth/user-belief/agreement/
  politeness/certainty-style directions, and audits agreement steering against
  politeness and random controls.
- Lab 17: persona, voice, roleplay, and register — implemented for the
  advanced course. Extracts paired persona/register/voice/agreement
  directions, tests held-out transfer, steers neutral prompts with
  content-vs-style scoring, and traces roleplay/register/refusal-monitor
  projections over scripted conversations.
- Lab 18: humor as incongruity — implemented for the advanced course.
  Measures setup entropy and ending surprisal, extracts a joke-vs-control
  direction, audits it against surprise/silliness/positivity, inspects
  attention back to setup tokens, and steers neutral endings with hand-label
  scaffolds for the "is it actually funnier?" question.
- Lab 19: model diffing with crosscoders — started for the advanced course.
  Adds a custom paired-crosscoder path, prompt inventory from existing frozen
  batteries, Tier A identity-pair smoke defaults, Tier B OLMo base-vs-instruct
  defaults, feature taxonomy/gallery artifacts, and an optional `--run-edit`
  feature-intervention smoke test.
- Lab 20: benign model organisms — started for the advanced course. Emits
  organism training corpora, sealed/unsealed manifests, behavior cards,
  baseline target/control generations, spillover audits, and a manifest schema
  for the later blind-audit sequence.

**The intro course is complete: 11 labs (Lab 1 includes the microscope smoke
test / instrumentation verification that used to be a separate pre-lab) +
the shared bench, each validated on Tier A (CPU) and Tier B (Colab A100).**
The advanced course is now in progress; Labs 12-18 are implemented, Labs 19-20
are started, and all should be treated as new lab code until their Colab
validation runs are recorded.
Two full-course regression sweeps are on record:
`runs/RUN2_VALIDATION_REPORT.md` (pre-rewrite tree, 24/24 green,
deterministic reproduction of the validated numbers) and
`runs/RUN3_VALIDATION_REPORT.md` (post-rewrite merge, 23/23 green after one
real data bug the rewrite's own validator caught; untouched labs reproduce
run 2 exactly). The one-page sweep view is `course_dashboard.py`; the
design-vs-built record is COURSE.md §0. The post-run-3 review pass
(prompt/pool tokenization fixes in Labs 1/3/5/6/11 and the continuous-
batching generation engine in Lab 10) changes some lab populations, so the
next sweep refreshes the reference headline numbers for those labs.

## Design decisions (deviations from COURSE.md, on purpose)

1. **Raw HF `transformers` + explicit hooks instead of TransformerLens.**
   COURSE.md proposes TransformerLens 3 / TransformerBridge. Labs 1–2 need
   only residual caching and the unembedding, which raw HF does in ~50
   transparent lines — and a course rule is that nobody runs code they can't
   explain. The patching and steering labs (5–7) were ultimately built on the
   same verified hook layer; it remains the abstraction point if heavier
   machinery is ever needed.
2. **GPU instead of TPU.** The collective-comms course ran on Cloud TPU; this
   one targets Colab A100/H100 with plain PyTorch.
3. **No binary-blob artifacts by default.** Model state is dumped as token
   tables, per-layer statistics, and *decoded* top-k readouts (CSV + markdown
   state cards). Raw tensors are opt-in (`--save-tensors`) and always carry a
   manifest saying what every tensor is.

## The instrument verifies itself

Every run performs self-checks before any science, and aborts on failure:

- **Hook parity** (`diagnostics/hook_parity.json`): forward hooks on every
  decoder block must reproduce `output_hidden_states` bit-for-bit.
- **Lens self-check** (`diagnostics/logit_lens_self_check.json`): the logit
  lens applied at the final depth must reproduce the model's actual output
  logits (top-1 must match, or be a measured near-tie within numeric noise).
- **Component anatomy probe** (Lab 2+, `diagnostics/component_anatomy.json`):
  contribution hook points are selected by verifying which candidate pair
  reconstructs every block's residual delta — never by module-name heuristics
  (post-norm architectures like Olmo-3 add *normed* submodule outputs).
- **Decomposition check** (Lab 2+, `diagnostics/dla_decomposition_check.json`):
  embeddings + all captured attn/MLP contributions must sum to the final
  pre-norm residual stream.
- **Patch no-op check** (Lab 5+, `diagnostics/patch_noop_check.json`):
  patching a run with its own vectors must be a numerical identity
  (max |Δlogit| ≤ 1e-4) — an off-by-one in patch layer or position indexing
  would otherwise produce beautiful, wrong heatmaps.

If a transformers upgrade ever changes hidden-state semantics, these fail
loudly and every downstream number is declared suspect — that is their job.

## The generation engine and its benchmark

Generation-heavy labs route through `bench.generate_continuous`, a
continuous-batching engine in pure HF forwards. The KV cache is a
preallocated static buffer with a sliding column window: every decode step
writes each row's new KV **in place** at one column (DynamicCache instead
reallocates and copies the whole cache every step), trims of the shared
left pad are free (the window slides), and retire/admit stay event-rate.
Admits are batched (`admit_block`) so prefills do not interrupt in-flight
rows on every retirement; rows retire at EOS and pending jobs take their
slots, so heavy-tailed CoT lengths never make a batch pay for its slowest
member. The hook surface is unchanged — per-job steering scales let Lab 7
ride an entire dose sweep on one schedule.

`bench_inference.py` is its A/B harness — exact-length heavy-tailed jobs,
NVML utilization sampling, per-step ITL traces, TTFT/prefill-stall
telemetry, and a two-part determinism contract: the engine must be
**bitwise self-deterministic** across identical calls (verified 48/48),
and every cross-engine token divergence vs lockstep `model.generate` must
be a verified n-way near-tie (top-logit gap ≤ 2 bf16 ulps under a clean
teacher-forced forward; greedy bf16 is not bitwise stable across attention
kernel paths, and pretending otherwise would just hide real cache bugs).
Measured on A100-80GB, greedy bf16:

| model | engine | rows | tok/s | peak GiB | p95 ITL |
|---|---|---:|---:|---:|---:|
| Olmo-3-7B-Think | lockstep | 16 | 183 | 31.4 | — |
| Olmo-3-7B-Think | continuous | 16 | 250 | 22.7 | 47 ms |
| Olmo-3-7B-Think | continuous | 32 | 490 | 31.7 | 46 ms |
| Olmo-3-7B-Think | continuous | 48 | **649** | 40.7 | 47 ms |
| Olmo-3-32B-Think | continuous | 16 | **116** | 64.5 | 116 ms |

p95 ITL staying flat as rows triple is the static cache + admit batching
at work (the old engine paid 50→80 ms). At 32B the step time is
weight-read-bound, so in-flight rows are nearly free throughput until
memory runs out — ~300 600-token probes complete in ~26 minutes. (The 32B
row is the run-4 engine; run 5 re-measures it on the static cache.)

## The course dashboard

```bash
python course_dashboard.py                        # newest run per lab
python course_dashboard.py --run-glob '*_run3_tierb*'   # pin one sweep
```

A pure reader over `runs/` that renders the whole course's state on one
page (`runs/course_dashboard.{md,png}`): per-lab instrument health (every
`diagnostics/*.json` self-check re-verified, not trusted), the headline
numbers (schema-drift-tolerant extraction), evidence rung, model, and
wall-clock. Use it after a sweep to see at a glance whether the term's
build is green — and which lab to read first when it is not.

## Run directories and artifacts

Every invocation creates `runs/<lab>-<timestamp>-<id>/`:

```text
run_config.json            # parsed CLI, after tier defaults
run_metadata.json          # packages, git state, GPU report, env vars
logs/console.log           # everything printed during the run
diagnostics/
  model_anatomy.{json,md}  # where blocks/final-norm/unembedding live — read once
  hook_parity.json         # self-check 1
  logit_lens_self_check.json  # self-check 2
  tokenization_report.csv  # every kept/dropped target with token counts
  gpu_memory_*.json
state/<example_id>/        # per-example human-readable model state:
  state_card.md            #   the narrative dump — start here
  tokens.csv               #   exactly what the model saw, with visible whitespace
  lens_trajectory.csv      #   one row per depth: top-1, p(target), entropy, ...
  logit_lens_topk.csv      #   top-k decoded readout per depth
  residual_stats_final_pos.csv
  residual_norms_by_position.csv
results.csv                # lab-specific long-form measurements
metrics.json               # aggregates
tables/*.csv               # per-example and per-category summaries
plots/*.png
run_summary.md             # the seven standard questions, answered with numbers
ledger_suggestions.md      # drafted claim-ledger entries (you edit, then commit)
artifact_index.json        # map of every artifact with a one-line purpose
```

Reading order for any run: `run_summary.md` → one `state_card.md` → plots →
`results.csv` → `diagnostics/` when anything looks wrong.

Plots include a small footer with `lab`, model, tier/dtype/prompt-set, and run
name, so screenshots remain identifiable outside the run directory. Main lab
tables and `results.csv` also prepend context columns (`lab`, `run_name`,
`model_id`, revision, tier, dtype, prompt set, seed, and model shape). Low-level
diagnostic CSVs keep their minimal check-specific schemas.

## The claim ledger

`claim_ledger.md` at this directory's root is the student's running dossier:
every claim carries an evidence tag (`OBS | ATTR | DECODE | CAUSAL |
SELF-REPORT`), the artifact backing it, and the observation that would kill
it. Labs draft claims with measured numbers into `ledger_suggestions.md`;
nothing touches the real ledger unless you pass `--append-ledger`, because
writing the claim is the coursework. The Lab 11 capstone audits this file.

## Common CLI

```bash
python interp_bench.py \
  --lab lab1 \
  --model allenai/Olmo-3-1025-7B \
  --model-revision <pin>   \
  --device cuda --dtype bfloat16 \
  --tier b \
  --prompt-set small|medium|full|path.json \
  --max-examples 0 \
  --topk 5 \
  --seed 0 \
  --save-tensors \
  --no-plots \
  --append-ledger \
  --run-name my_experiment
```

`--tier a` always maps to a CPU-feasible configuration so the smoke path is
one flag, not a recipe.

## Troubleshooting

**Anatomy resolution fails for a new model.** Add its block/norm paths to
`BLOCKS_PATH_CANDIDATES` / `FINAL_NORM_PATH_CANDIDATES` in `interp_bench.py`
(one place, on purpose). Multimodal and encoder-decoder models are out of
scope.

**Lens self-check fails.** Don't continue. Check `model_anatomy.md` (wrong
final-norm path?), quantization (looser numerics are recorded but top-1 must
still match), and whether the model post-processes logits (softcapping is
handled; anything else needs a look).

**OOM on a small GPU.** `--quantization 8bit` or `--model google/gemma-3-1b-pt`.
Lab 1's capture is tiny; the model weights are the cost.

**Slow on CPU with the 7B model.** That's what `--tier a` is for.

## Adding a new lab

When adding `labN`, update together:

```text
labs/labNN_name.py        # the experiment: prompts, loop, plots, summary
labs/labNN_name.md        # the student handout
interp_bench.py           # LAB_PROFILES registry entry
README.md                 # status section above
```

Each lab module exposes `run(ctx, bundle)` and should push any reusable
measurement machinery down into the bench (capture, trajectories, dump
formats) rather than growing private copies. Before release, verify the
checklist in [how_to_design_labs.md](how_to_design_labs.md) §7 — most
importantly: `--tier a` completes on CPU, the self-checks pass, every claim
in `ledger_suggestions.md` cites an artifact that exists, and the handout's
questions are answerable from the artifacts alone.
