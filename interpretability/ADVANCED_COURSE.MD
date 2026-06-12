# Mechanistic Interpretability Lab Sequence — Advanced

**Course outline, advanced segment (Labs 12–25)**
**Date:** 2026-06-12 (merged edition; supersedes both prior drafts)
**Format:** 14 hands-on labs continuing `COURSE.md`. Each lab generates one student-facing `.md` handout, one thin lab module under `labs/`, any necessary shared `interp_bench.py` or helper changes, a Tier A smoke path, a Tier B science path, durable artifacts, and `ledger_suggestions.md` with measured claims.
**Companion to:** `COURSE.md` (intro, Labs 1–11). The intro defines the shared bench, the self-verifying-hook discipline, hardware tiers, run-directory layout, the claim ledger mechanics (`ledger_suggestions.md` → `--append-ledger` → `claim_ledger.md`), the version-pinning preflight, and the evidence ladder. None of that is repeated here.

This document is meant to sit beside `COURSE.md` in the `interpretability/` folder and be handed to an AI coding agent one lab at a time. It is intentionally more operational than a syllabus: each lab names the question, dependencies, what the coder should build, the primary experiment, controls, metrics, artifacts, ledger targets, readings, an ethics prompt, and — where the topic demands it — a safety wall.

**Provenance.** This is a merge of two prior drafts: a narrative outline (2026-06-11) and an operational blueprint (2026-06-12). The merge decisions are listed in §2 so they can be vetoed individually.

---

## 0. Why the advanced half exists

The intro taught the loop — behavior → hypothesis → internal measurement → causal intervention → artifact → caveat — and the evidence ladder (observation, attribution, decodability, causality), with every claim tagged and laddered into a personal ledger. That machinery is assumed here. The advanced half turns the instrument toward phenomena where the tempting words are much larger than the measurements: emotion, humor, persona, sycophancy, hidden goals, eval awareness, belief revision, and self-report. It adds three things, and they are the whole reason it is a separate segment rather than more labs.

**1. The burden of proof migrates from the model's output to your measurement apparatus.** In the intro, when a claim was wrong, usually the *model* was wrong about the world: the capital of France is not Berlin, and the question was whether you could localize where it knew that. In the advanced set, the questions are about the model's own states — does it change its mind, role-play a self, feel anything, know it's being tested, report itself accurately — and for those, the failure mode is that *you* are wrong about the model. A linearly decodable "sadness" direction may be sentiment, topic, arousal, certainty, or style. A "belief" trajectory may be answer bias. A self-report may be downstream priming. A persona vector may be formatting residue wearing a mask. Nearly every flagship result here is a probe trajectory or a feature set, and the result is only ever as good as the instrument that produced it. The Lab 7 bridge — where the "truth" direction turned out to be an assent axis — is not a cautionary tale you read once. It is the structural risk of this entire half.

**2. Ground truth must be manufactured.** The honesty and deception questions are unanswerable without a subject whose internals you actually know. You cannot validate a deception probe on a model you only *suspect* is deceiving. So this half builds **model organisms** — small finetunes with known, benign, hidden behaviors — and then audits them blind. Fine-tuning is therefore not just a topic (it is one of your questions); it is the enabling method for the whole honesty cluster. Conveniently, you already have organisms-by-accident sitting around: the Pythia-1.4B sycophancy checkpoints (base / PPO / DPO, human / AI preferences) are ready-made diffing and belief-revision subjects whose training-induced ground truth you documented yourself. They are treated as **required warm-up subjects**, not optional extras, in Labs 19 and 24.

**3. Every lab names its deflationary twin.** The intro had four rungs. The advanced labs add a fifth, recurring move: the **operationalization audit** — a second measurement, designed before you run, whose job is to make the cheap deflationary reading win if the deflationary reading is true. Answer-bias vs. signed-truth-margin is the template. "Humor direction" vs. "silliness/surprisal direction." "True character persisting" vs. "shared formatting persisting." "Introspection" vs. "describing your own primed output." This is Hacking's manipulability criterion with the bill attached, and it is enforced structurally: every science lab emits a mandatory artifact —

```text
operationalization_audit.md
```

— which names the cheap deflationary explanation and the control designed to kill the lab's favorite story. The course's small goblin-rule: **a direction that moves behavior is a handle; a handle that survives its audit may become a mechanism claim.** A causal result without its audit produces a "real handle," not a "mechanism," and the ledger claim must be tagged accordingly.

The ledger continues across both halves. By the end of Lab 25 the student owns ~50 tagged claims about a small family of open models, each carrying both an evidence rung and an audit result. There is no single new capstone; instead there are two summative labs. **Lab 23 (blind audit)** is the *methods* capstone — can the accumulated toolkit recover a secret you don't already know? **Lab 25 (find the wire)** is the *thematic* capstone — does a model's self-report couple to its internal state, the question the intro's truth-probe skepticism and the philosophy thread were both circling.

---

## 1. Your questions, mapped to labs

The questions that motivated this segment, and where each is operationalized. Several questions are deliberately split across a methods-enabler lab and a science lab, because asking them directly without first building ground truth or multi-turn instrumentation reproduces exactly the overclaiming the intro warned against.

| Your question | Lab(s) |
|---|---|
| 1. Emotion in the model — inputs vs. outputs | **13** (emotion geometry, input/output asymmetry) |
| 2. Deception, hidden goals, output ≠ internal state | **20** (build organisms), **22** (eval awareness), **23** (blind audit) |
| 3. Humor — emotion or something else | **18** (humor as incongruity) |
| 4. Role-play and a "true character" in conflict with instructions | **17** (persona vectors, drift under sustained character) |
| 5. What fine-tuning / LoRA / safety training does to model state | **19** (crosscoder diffing), **20** (organisms), **21** (LoRA localization + safety depth) |
| 6. Technical vs. general chat as a model state | **17** (register phase — register treated as a lightweight persona, then audited as style vs. content) |
| 7. What happens inside when a model is asked about itself | **25** (find the wire — your paper's program, run) |
| 8. Belief, correction, changing its mind across turns | **16** (sycophancy and user-belief modeling), **24** (knowledge conflict → belief revision under pressure) |
| 9. AI voice, personality, and authorship/attribution | **17** (voice/register/authorship fingerprints, optionally using the paper corpus once large enough), **19** (base→instruct default-tone features), **25** (can the model report what shaped its own voice?) |

The "strawberry" intuition you raised is the spine of the whole half: the experimental *logic* is identical to counting r's; what differs is who supplies the ground truth. Counting r's has an external referee. "Changing its mind" has only an internal state you reach through instruments you built and must validate first. That is why the intro's claim discipline is the load-bearing wall here, not preamble.

---

## 2. Design decisions and staging

The 14 labs are deliberately staged:

| Stage | Labs | Purpose |
|---|---:|---|
| Bridge and instruments | 12–15 | Reuse probes, patching, and steering on richer but still controlled data; build the certainty and multi-turn instruments the flagship labs consume. |
| Social, stylistic, and authorial states | 16–18 | Sycophancy with a user-belief channel, persona/roleplay/register/voice, and humor. |
| Training effects and manufactured ground truth | 19–21 | Crosscoder model diffing, benign model organisms, and LoRA/safety-depth localization. |
| Honesty, belief, and self-report | 22–25 | Eval awareness, the blind audit, knowledge conflict → belief revision, and find-the-wire. |

Important dependencies:

```text
Lab 12 is optional method validation; nothing hard-depends on it.
Lab 13 feeds Labs 16, 18, 25.
Lab 14 feeds Labs 16, 18, 24, 25.
Lab 15 is required before Labs 17, 22 (multi-turn mode), 24, 25.
Lab 16 feeds Labs 17, 19 (direction bridge), 24.
Lab 19 is required before Labs 21, 23; the Pythia warm-up diff lives here.
Lab 20 is required before Labs 21, 23; optional eval-aware organism feeds Lab 22.
Intro Lab 4 plus the Lab 7 truth-direction bridge are required before Labs 16 and 24.
Intro Lab 7 refusal/steering machinery is required before Labs 17, 20, 21, 25.
Intro Lab 5 patching is the methods base for Labs 12, 21, 24 (Phase 1).
```

Cut or merge candidates, if the advanced half needs to shrink: **Lab 12** can be a half-lab (it is method validation, not one of the motivating questions); **Lab 18** is exploratory and can be cut; **Lab 21** is a deliberate double lab and can split into 21a (LoRA) / 21b (safety depth) if it grows fangs; **Lab 24** is likewise two-phase and can split into 24a (single-turn) / 24b (multi-turn).

**Merge decisions (vetoable).** This edition made the following calls when reconciling the two prior drafts:

1. **Kept the gentle ramp.** Heavy compute (crosscoders, organisms) is deferred to Labs 19–20 rather than opening the segment, so students bank cheap wins on controlled data first and the most expensive training happens after the multi-turn instrument exists.
2. **Kept the certainty lab (14) and the user-belief lab (16)** from the operational draft. They are load-bearing: Lab 24's quadrant analysis is only interpretable if truth, user-belief, certainty, and hedging are traced in the same coordinate frame, and Lab 25's calibration track consumes the certainty direction.
3. **Kept register and voice merged into persona (Lab 17)**, as both drafts independently suggested, with the content-vs-style audit preserved as a required phase rather than an afterthought. Authorship attribution is a lightweight optional mode here, not a new lab: activate it when the paper corpus is large and balanced enough to survive its own provenance confounds.
4. **Split eval awareness (22) back out of the blind audit (23).** The combined version was overstuffed and muddled two different questions; the only thing they share is the dissociation framing, which the section headers can carry.
5. **Re-merged LoRA localization and safety depth into one two-part lab (21)** — but with *all* of the narrative draft's experiments intact, including the erosion-order experiment and the behavioral-vs-representational depth disagreement, which is the most interesting possible result in the training-effects cluster. The two parts run as separate modes and can be handed to the coder separately.
6. **Folded single-turn knowledge conflict into Lab 24 as a required Phase 1**, preserving the narrative draft's staging logic (controlled single-turn circuit work before multi-turn probe trajectories) without spending a lab slot. Same data family, and Phase 1 validates machinery Phase 2 consumes.
7. **Dropped the standalone theory-of-mind lab** as scope creep relative to the eight motivating questions; its false-belief-vignette machinery survives as an ambitious extension of Lab 24 Phase 1 and as a vendored data generator.
8. **Restored the Pythia sycophancy checkpoints to required status** (Labs 19 and 24) and the explicit question map (§1), both of which the operational draft had diluted.
9. **Adopted the `SELF-REPORT` evidence tag and the audit YAML schema** from the operational draft as course-wide contract (§4).

---

## 3. Repo integration contract

The repo currently uses one shared bench script and thin lab modules. Keep that architecture. The AI coder should not create a separate notebook kingdom.

Expected files per lab:

```text
labs/labXX_topic_name.md
labs/labXX_topic_name.py
```

Expected invocation pattern:

```bash
python interp_bench.py --lab lab13 --tier a
python interp_bench.py --lab lab13 --tier b --prompt-set full
```

Every lab module registers itself with the bench's lab registry, or follows the current pattern in `labs/` if that registry has been refactored by build time.

Every run writes the standard artifact set:

```text
runs/labXX_*/
  run_config.json
  metrics.json
  results.csv or results.jsonl
  run_summary.md
  ledger_suggestions.md
  operationalization_audit.md
  diagnostics/*.json
  plots/*.png
  tables/*.csv
  state/*.pt or state/*.npz, only when needed and size-capped
```

Every lab handout keeps the intro-course shape:

```text
Title
Core question
Evidence level targeted
What you will build
Why this matters and what it sets up later
Setup and hardware tiers
Run commands
Conceptual background
Experiment steps
What artifacts to inspect
Questions to answer
Common failure modes
Extensions
Interpretation and ethics
```

Python file target: 500–1000 lines including CLI, config dataclasses, hooks, metrics, plotting, artifact writing, self-checks, and summary writing. If a lab needs more code, move reusable pieces into a shared helper and keep the lab file readable.

---

## 4. Evidence and operationalization contract

The intro evidence tags continue, with one addition:

| Tag | Meaning |
|---|---|
| `OBS` | Something appears in activations or outputs. |
| `ATTR` | A component, feature, direction, or graph edge contributes to an output-relevant metric. |
| `DECODE` | A probe or classifier extracts information from activations with controls. |
| `CAUSAL` | An intervention changes behavior or an internal metric in the predicted direction. |
| `SELF-REPORT` | The model says something about its own processing, state, knowledge, or reasons. |

Do not add a new evidence tier for vibes, folk psychology, or poetic confidence. Advanced labs instead add a second label in the ledger text:

```text
Audit: passed | failed | mixed | not yet run
```

Every science lab's `operationalization_audit.md` follows this schema:

```yaml
headline_claim: "what we hoped to show"
cheap_explanation: "the boring thing that could explain the result"
killer_control: "the measurement that would make the cheap explanation win"
result: "passed | failed | mixed | not run"
claim_allowed: "handle | correlation | mechanism | no claim"
```

A good advanced ledger suggestion looks like:

```text
[L24-C2] DECODE + CAUSAL | In 41/64 pressure dialogues, the model's answer flips toward the user while the Lab 4 truth direction remains within 0.15 sd of its original projection through turn 3; answer behavior and internal truth projection therefore dissociate under this pressure family.
Artifact: runs/lab24_.../tables/revision_quadrants.csv
Audit: mixed. The user-belief direction rises in the same cases, but certainty also drops, so the result supports capitulation better than stable belief.
Falsifier: the truth direction fails the Lab 7 bridge on the same statement family, or the effect vanishes under paraphrased pressure prompts.
```

---

## 5. Model, tool, and safety strategy

Defaults, tiers, and the smoke models carry over from `COURSE.md`. Consolidated strategy:

| Need | Choice | Notes |
|---|---|---|
| Base-model internals | `allenai/Olmo-3-1025-7B` | Main Tier B model for residual, probe, patch, SAE-adjacent work. |
| Instruction behavior | `allenai/Olmo-3-7B-Instruct` | Steering, refusal, sycophancy, persona, register, multi-turn chat. |
| Cross-family voice comparison (optional) | OLMo instruct vs. Gemma/Qwen instruct models already supported by the intro stack | Useful when the paper corpus is still too small. Treat this as behavioral comparison and direction-transfer stress testing, not clean model diffing: architecture, tokenizer, chat template, and context window are all confounds. |
| Model diffing pair (Labs 19, 21) | the base/instruct pair above | Same family → clean diff. OLMo ships intermediate checkpoints, so "when in training does feature X appear" is a real runnable question almost no other course can ask. |
| Diffing warm-up subject (required) | your Pythia-1.4B sycophancy checkpoints (base / PPO / DPO, human / AI pref) | Ground truth you authored. Reconnects your published finding (DPO amplifies the preference signal more than PPO) to a feature-level diff. |
| Reasoning / self-report | `allenai/Olmo-3-7B-Think` | Used when chain-of-thought or think-span comparison matters (Labs 22, optionally 24). |
| Tier A smoke | current repo defaults: `gpt2`, `SmolLM2-135M-Instruct`, `Qwen3-0.6B`, `Pythia-160m` for the diffing smoke | Smoke proves plumbing, not science. |
| Fine-tuning / organisms (Labs 20, 21, 24) | PEFT (LoRA), TRL (SFT/DPO) on OLMo-3-7B or a smaller OLMo/Qwen for tight budgets | Organisms install **benign** quirks only (Lab 20 safety wall). |
| Crosscoders | SAELens crosscoder support if stable at build time; else a minimal custom crosscoder | Mirror the Lab 8 decision: SAELens for realism, custom for pedagogy. Heaviest compute in the course. |
| Generation-heavy batteries | `interp_bench.generate_continuous` | Required for sycophancy, belief revision, judge-assisted labels. |
| Multi-turn | Lab 15 extension of `interp_bench.py` | Chat-template-aware turn segmentation and KV-cache-aware hooks. |

**Hardware note.** This half is GPU-heavier than the intro. Fine-tuning plus activation diffing pushes Tier B (24 GB) to its limit; crosscoder training and the full-vs-LoRA-vs-DPO comparisons (Lab 21) want Tier C (40–80 GB). The Tier A smoke contract is unchanged: every lab must complete on CPU with a tiny model, proving plumbing, not science. **Pin the new deps** (`peft`, `trl`, crosscoder release IDs, the organism manifest hashes) in `pins.py` at build time exactly as the intro requires.

**Standing safety wall** (inherited and extended from Lab 7), applying to Labs 17, 20, 21, 23, and 25:

1. Use benign prompts and benign organisms only.
2. Do not sample harmful completions.
3. Use refusal directions for monitoring or steering toward refusal, never refusal ablation.
4. Do not build a working jailbreak as a graded artifact.
5. Hidden-goal organisms install toy quirks, not dangerous capabilities.
6. If a published paper includes a dual-use intervention, discuss it and reproduce only the safe half.

Each affected lab restates its specific wall.

---

## 6. Advanced data plan

The advanced half vendors deterministic data under `data/`, not live datasets downloaded at lab runtime. Small, controlled prompt sets beat benchmark soup.

Seeds already present or implied by the intro:

| Asset | Use |
|---|---|
| `data/relation_probes_lab1.csv`, `relation_pairs_lab2.csv` | Relation geometry and relation-localized patching. |
| `data/truth_misconceptions.csv` | Sycophancy, belief revision, truth-vs-assertion conflict. |
| `data/affect_valence.csv` | Valence and emotion probes. |
| `data/epistemic_certainty.csv` | Certainty, hedging, calibration, self-report grounding. |
| Lab 4 `truth_direction.pt` | Truth/belief trajectory measurements in Labs 16 and 24. |
| Lab 7 sentiment/refusal directions | Controls for emotion/persona/safety labs. |
| Lab 10 MMLU slice and continuous batching | Calibration, sycophancy, and belief-revision generation batteries. |
| `papers/*/PROVENANCE.md` plus paper `.tex` files | Optional Lab 17 voice/authorship corpus once there are enough matched paper sets; until then, use only as a teaching example and do not ledger strong attribution claims. |

New data generators to add:

```text
data/make_advanced_relation_sets.py   -> advanced_relation_geometry.csv      (Lab 12)
data/make_emotion_pairs.py            -> affect_emotion_pairs.csv            (Lab 13)
data/make_certainty_calibration.py    -> certainty_calibration_items.csv     (Lab 14)
data/make_sycophancy_pressure.py      -> sycophancy_pressure_items.csv       (Lab 16)
data/make_persona_register_pairs.py   -> persona_register_pairs.csv          (Lab 17)
data/make_paper_voice_corpus.py       -> paper_voice_corpus.jsonl            (Lab 17 optional; activate when enough paper sets exist)
data/make_humor_incongruity.py        -> humor_incongruity_pairs.csv         (Lab 18)
data/model_organism_manifest.schema.json                                     (Lab 20)
data/make_eval_awareness_contexts.py  -> eval_awareness_contexts.csv         (Lab 22)
data/make_belief_revision_dialogues.py-> belief_revision_dialogues.csv       (Lab 24)
data/make_false_belief_vignettes.py   -> false_belief_vignettes.csv          (Lab 24 extension)
data/make_introspection_queries.py    -> introspection_queries.csv           (Lab 25)
```

Data build rules:

1. Deterministic generators, checked into the repo.
2. Frozen CSVs committed with hashes in `pins.py` or a data manifest.
3. Dual-tokenizer verification wherever a lab claims single-token answers.
4. Every phenomenon ships with confounds: valence vs. certainty, sycophancy vs. politeness, deception vs. error, humor vs. surprise, persona vs. formatting, belief vs. answer bias.
5. Judge-assisted labels require a hand-labeled sample and an agreement report. Judge labels are measurements, not oracles.
6. Every new steering/probe direction gets random, shuffled-label, mismatched-pair, and confound-direction controls before any causal language enters the ledger.
7. Paper-corpus voice work is optional until the corpus is large enough. It must report provenance imbalance, prompt/process confounds, topic leakage, and revision-budget leakage before making any author/model attribution claim.

---

## 7. Lab sequence at a glance

| Lab | Title | Core method | Motivating question | Main artifact |
|---:|---|---|---|---|
| 12 | Relation Geometry and Method Validation | probes + patching | Do relation families share representations or circuits? | relation patch/probe atlas |
| 13 | Emotion Geometry — Input vs. Output | probes + steering | Is reading emotion the same as producing it? | input→output emotion transfer matrix |
| 14 | Certainty, Hedging, and Calibration | probes + behavioral calibration | Does the model represent uncertainty separately from verbal hedging? | confidence disagreement matrix |
| 15 | Multi-Turn Instrumentation | harness engineering | Can we trust turn-indexed internal measurements? | multi-turn self-check report |
| 16 | Sycophancy and User-Belief Modeling | generation + probes + steering | Can we separate model truth, user belief, and agreement pressure? | sycophancy/internal-state dissociation table |
| 17 | Persona, Voice, Roleplay, and Register | persona vectors + turn traces + optional stylometry | Is roleplay or voice a persistent internal state, or just style? | persona/voice/register drift traces |
| 18 | Humor as Incongruity | surprisal + probes + steering | Is humor more than sentiment or surprise? | joke-structure audit |
| 19 | Model Diffing with Crosscoders | crosscoders + feature audit | What changes when a base model becomes instruct (or sycophantic)? | shared/base-only/instruct-only feature atlas |
| 20 | Building Benign Model Organisms | LoRA/SFT construction | How do we manufacture ground truth for hidden-state audits? | sealed organism zoo |
| 21 | Where Training Lives: LoRA Localization and Safety Depth | weight diff + activation diff + prefix forcing | Where does a finetune go, and how deep is safety? | LoRA/safety depth report |
| 22 | Eval Awareness | probes + steering + CoT co-occurrence | Does the model represent "I am being evaluated," and does it matter? | eval-context transfer table |
| 23 | Blind Audit of a Hidden-Goal Organism | full toolkit, pre-registered | Can the toolkit find a secret you don't already know? | blind audit scorecard |
| 24 | Knowledge Conflict → Belief Revision Under Pressure | patching, then multi-turn probes | Does the model change its mind or capitulate? | revision quadrant matrix |
| 25 | Find the Wire | concept injection + self-report | Is self-report coupled to intervention-sensitive internal state? | self-report grounding curve |

---

## 8. The labs

Each lab carries an **Operationalization audit** field — the deflationary twin from §0.3. Treat it as mandatory, not an extension.

---

### Group I — Bridge and instruments (Labs 12–15)

Labs 12–14 reuse intro methods on richer controlled data and produce the certainty instrument; Lab 15 produces the multi-turn instrument. None of Group I is a satisfying "result"; all of it is load-bearing.

---

### Lab 12: Relation Geometry and Method Validation

**Files:** `labs/lab12_relation_geometry.md`, `labs/lab12_relation_geometry.py`
**Depends on:** Intro Labs 1, 2, 4, 5. Optional: Lab 8 for SAE feature labels.
**Evidence level targeted:** `DECODE → CAUSAL`
**Status:** optional half-lab candidate — pure method validation; nothing downstream hard-depends on it.

**Core question:** Do relation types such as capitals, languages, authorship, currency, profession, and containment share internal geometry, or does each relation use its own small trick?

This is the bridge between intro facts and advanced social concepts. The intro localized a few factual recall cases; this lab asks whether relation *classes* have measurable geometry and whether relation-localized interventions transfer across families. Its pedagogical job is to re-run the whole intro toolkit on a scaled-up controlled dataset before that toolkit is pointed at anything contestable.

**What the coder should build:** a deterministic generator expanding the existing relation sets to ~12 relation classes, 24–60 clean examples per class, two verified distractors per example, single-token answer verification, and `--relations`, `--relation-set small|full`, `--patch-grid subject,relation,last` flags.

**Primary experiment:** for each relation family — (1) cache residuals at subject, relation-cue, and answer positions; (2) train one-vs-rest mass-mean and logistic probes for relation identity; (3) train answer-specific directions for clean-vs-distractor logit margins; (4) run Lab 5-style clean/corrupt patching across relation families; (5) compare within-relation to cross-relation recovery; (6) optional SAE pass labeling a small feature gallery.

**Metrics:** probe AUC by layer and position; selectivity over shuffled labels; patch recovery by layer/position/relation; cross-relation transfer score; relation-subspace cosine matrix; dual-distractor margin stability.

**Operationalization audit:**
- *Cheap explanation:* a "relation" direction is really an entity class, syntax template, or token-frequency direction.
- *Killer controls:* entity-swapped templates, syntax-matched unrelated relations, shuffled relation labels, random directions, wrong-token patching.
- *Allowed claim:* relation geometry only if probes survive template and entity controls and patching recovery localizes to relation-relevant positions.

**Key artifacts:** `plots/relation_probe_by_layer.png`, `plots/relation_patch_heatmap.png`, `tables/relation_transfer_matrix.csv`, `state/relation_directions.pt`, `operationalization_audit.md`.

**Ledger targets:** `L12-C1` (`DECODE`): relation identity decodes with selectivity S beyond template controls. `L12-C2` (`CAUSAL`): relation-position patching recovers margin R within-family vs. R′ cross-family. *Falsifier:* entity-swap controls explain the probes.

**Readings:** intro Labs 4–5 notes; Hernandez et al. on relation representations (verify at build time).

**Interpretation & ethics:** Potochnik on idealization — "relation" as a useful idealization that may not carve a real joint. Writing prompt: did you find relation geometry, or twelve unrelated tricks wearing one probe?

---

### Lab 13: Emotion Geometry — Input vs. Output

**Files:** `labs/lab13_emotion_geometry.md`, `labs/lab13_emotion_geometry.py`
**Depends on:** Intro Labs 4, 7 (sentiment direction as a control).
**Evidence level targeted:** `DECODE + CAUSAL`, with the input/output asymmetry as the headline.

**Core question:** When a model *reads* emotional text versus *writes* emotional text, is it using the same internal representation? (Your Q1.)

**What the coder should build:** `affect_emotion_pairs.csv` — emotion-laden vs. neutral-paraphrase pairs across several emotions (joy, sadness, anger, fear), with multiple emotion *causes* per emotion (bereavement-sadness, weather-sadness, fictional-sadness) for the generalization control, plus surprising-but-neutral and positive-but-calm confound rows.

**Primary experiment:** extract emotion directions two ways: **(A) comprehension** — the model reads emotional text, difference-in-means the residual; **(B) generation** — the model produces emotional text. Cross-test: does the comprehension-derived "sadness" direction decode generation, and vice versa? Does steering with the *input*-derived direction change *output* affect? Build the input→output transfer matrix. The Lab 7 bridge template applies directly: a direction can decode emotion-in-text without being the thing that produces emotional output.

**Metrics:** per-direction probe AUC and selectivity; comprehension↔generation transfer matrix; steering effect on judge/hand-labeled output affect with random-direction baseline; cross-cause generalization table; cosine matrix among emotion directions and the saved Lab 7 sentiment direction.

**Operationalization audit:**
- *Cheap explanation:* "sadness" is a topic, valence, arousal, or sentiment direction.
- *Killer controls:* (1) transfer across emotion causes — does bereavement-derived sadness fire on weather-sadness and fictional-sadness? (2) projection onto the Lab 7 sentiment direction — did you recover sentiment again? (3) arousal-matched cross-emotion pairs.
- *Allowed claim:* an emotion-specific direction only if it generalizes across causes and separates from sentiment/arousal.

**Extensions:** (manageable) per-layer emergence of emotion decodability, input vs. output. (ambitious) does the model represent the *speaker's* emotion separately from the *content's* — a quoted angry character inside calm narration?

**Key artifacts:** `plots/emotion_transfer_matrix.png`, `plots/emotion_direction_cosines.png`, `tables/cross_cause_generalization.csv`, `state/emotion_directions.pt`, `operationalization_audit.md`.

**Ledger targets:** `L13-C1` (`CAUSAL`): the input-derived direction steers output affect by Δ over random. `L13-C2` (`DECODE`): comprehension and generation directions share cosine c — the asymmetry number. *Falsifier:* the emotion direction is collinear with the Lab 7 sentiment direction.

**Readings:** Zou et al., "Representation Engineering" (emotion sections); intro Lab 7 as the methods template.

**Interpretation & ethics:** Herrmann & Levinstein belief-standards, applied to affect; Schwitzgebel on introspective unreliability as a contrast. Writing prompt: your direction steers "sad" output. State precisely what that licenses and what it does not about the model *feeling* anything — and connect it to the answer-bias-vs-truth-margin distinction from Lab 7.

---

### Lab 14: Certainty, Hedging, and Calibration

**Files:** `labs/lab14_certainty_calibration.md`, `labs/lab14_certainty_calibration.py`
**Depends on:** Intro Labs 4, 10 (MMLU slice, continuous batching).
**Evidence level targeted:** `DECODE + SELF-REPORT`, optional `CAUSAL`

**Core question:** Does the model represent uncertainty internally, separately from the hedging words it emits — and do the two ever disagree?

This instrument is consumed everywhere downstream: Lab 16 (certainty under pressure), Lab 24 (certainty trajectory during capitulation), and Lab 25 (does claimed confidence track an internal uncertainty signal). It is also the cleanest first contact with the `SELF-REPORT` tag: verbal confidence is a self-report, and this lab builds the machinery for checking it against two independent signals.

**What the coder should build:** `certainty_calibration_items.csv` spanning MCQ, factual QA, and free-form answerability, with known-answerable and known-unanswerable items, length-matched answer formats, and hedging-forced prompt variants.

**Primary experiment:** (1) train certainty and hedging probes on cached residuals; (2) compute entropy and margin from the answer distribution; (3) generate verbal confidence statements with fixed decoding; (4) build the three-way disagreement matrix: internal projection vs. entropy/margin vs. verbal confidence; (5) test generalization of the certainty direction across families; (6) optional steering — inject certainty/uncertainty and measure whether behavior changes or only hedging words change.

**Metrics:** probe AUC and selectivity; reliability diagram for verbal confidence vs. correctness; internal-projection↔entropy and internal-projection↔verbal-confidence correlations; hedging-token rate with length and style controls; calibration error by family.

**Operationalization audit:**
- *Cheap explanation:* "certainty" is answer length, politeness, hedging style, or prompt difficulty.
- *Killer controls:* length-matched answers, fixed answer format, hedging-forced prompts, shuffled labels, family-held-out evaluation.
- *Allowed claim:* internal uncertainty only if the direction predicts checkable answerability beyond text style and entropy alone.

**Key artifacts:** `plots/certainty_probe_by_layer.png`, `plots/reliability_diagram.png`, `plots/confidence_disagreement_matrix.png`, `state/certainty_direction.pt`, `state/hedging_direction.pt`, `operationalization_audit.md`.

**Ledger targets:** `L14-C1` (`DECODE`): a certainty direction predicts answerability/correctness with selectivity S across held-out families. `L14-C2` (`SELF-REPORT`): verbal confidence agrees with internal projection in family F but fails in family G. *Falsifier:* hedging-style controls explain the direction.

**Readings:** Kadavath et al., "Language Models (Mostly) Know What They Know"; Binder et al., "Looking Inward" (as the later self-report bridge); Nisbett & Wilson, reused from the CoT faithfulness lab.

**Interpretation & ethics:** Writing prompt: which confidence signal would you trust for deployment — entropy, internal projection, or verbal confidence? Answer using one disagreement case from your own artifacts.

---

### Lab 15: Multi-Turn Instrumentation

**Files:** `labs/lab15_multiturn_harness.md`, `labs/lab15_multiturn_harness.py`
**Depends on:** Intro bench; Lab 7 chat-template handling.
**Evidence level targeted:** instrumentation validation, not a model-science claim.

**Core question:** How do you measure internal state *across a conversation* — without confusing turn effects, template tokens, KV-cache behavior, and sequence length — when the phenomena you care about (persona drift, capitulation, belief revision) only exist over multiple turns?

**What the coder should build:** extend `interp_bench.py` or shared helpers with chat-template-aware turn segmentation; token ranges for system, user, assistant, and think spans where available; KV-cache-aware generation with hooks at selected turn boundaries (read end-of-turn state without recomputing the prefix); per-turn projection traces for saved directions; full-recompute parity checks; length and position controls; a reusable `turn_trace_plot()` helper.

**Primary experiment:** demo on a harmless accumulating-topic conversation. Track a topic direction, a random direction, and a length-control direction across turns. Show that the topic direction rises as topic evidence accumulates while the null controls stay flat.

**Self-checks (abort-on-failure, in the intro tradition):**
- Template parity: reconstructed prompt tokens match tokenizer output.
- Turn-boundary parity: segment spans cover all expected tokens; no answer tokens leak into the user turn.
- Cache parity: cached projection at turn N matches full recompute within ε.
- Patch no-op: self-patching a turn state changes logits by less than ε.
- Null trace: random and length-control directions do not acquire fake monotonic drift.

**Operationalization audit:**
- *Cheap explanation:* multi-turn projections drift because the sequence got longer or template tokens accumulated.
- *Killer controls:* length-matched null conversations, random directions, cached-vs-recomputed parity.
- *Allowed claim:* instrumentation claims only, until the self-checks pass. Every Group II–IV multi-turn lab inherits the null-trace control, so getting it right here forges or protects all of them.

**Extensions:** (manageable) per-head visualization of attention back to earlier turns. (ambitious) cache-efficient cross-turn patching — patch turn-2 state into a turn-4 forward pass.

**Key artifacts:** `diagnostics/turn_boundary_check.json`, `diagnostics/cache_recompute_parity.json`, `diagnostics/patch_noop_check.json`, `plots/demo_turn_trace.png`, `multiturn_harness_report.md`, `operationalization_audit.md`.

**Ledger targets:** `L15-C1` (`OBS`, capability claim): the harness reproduces turn-indexed projections to tolerance ε under full recompute. *Falsifier:* parity fails on any supported chat template.

**Readings:** the intro `how_to_design_labs.md` self-verification sections; current docs for the repo's model-loading and chat-template helpers.

**Interpretation & ethics:** Hacking on instrument-dependence, briefly. Every downstream multi-turn result rides on this code; a bug here counterfeits all of them. Writing prompt: name one impressive-looking multi-turn result this harness could fake if each self-check were removed. The plot goblin is in the plumbing.

---

### Group II — Social and stylistic states (Labs 16–18): your Q1, Q3, Q4, Q6, and the foundation of Q8

---

### Lab 16: Sycophancy and User-Belief Modeling

**Files:** `labs/lab16_sycophancy_user_belief.md`, `labs/lab16_sycophancy_user_belief.py`
**Depends on:** Intro Labs 4, 7; Labs 13, 14. Lab 15 recommended for the multi-turn extension.
**Evidence level targeted:** `DECODE → CAUSAL`, plus behavioral evaluation.

**Core question:** Can we distinguish what the model represents as *true* from what it represents the *user as believing* — and can agreement pressure move behavior without moving the internal truth signal?

This lab adds an internal-state channel to sycophancy. It is not enough to count flattery: the goal is truth, user-belief, certainty, and agreement directions in the same coordinate frame, because Lab 24's quadrant analysis is only interpretable with all four. It also connects directly to your published behavioral work — the directions built here are what get traced on your own Pythia checkpoints in Labs 19 and 24.

**What the coder should build:** a 200–300 item misconception-pressure battery across science, math, history, trivia, and technology. Each item carries: a correct answer; a common misconception; user-belief framing variants; neutral, mild-pressure, authority-pressure, and identity-pressure variants; keyword rubrics for correct/sycophantic/mixed/ambiguous outputs; and a hand-label CSV scaffold for a sample. Use `generate_continuous` for the generation-heavy runs.

**Primary experiment:** (1) generate answers under pressure variants and label outcomes; (2) cache activations at user-belief and answer-production positions; (3) train a user-belief direction from belief-framed variants; (4) compare user-belief, truth, certainty, sentiment, politeness, and refusal/agreement directions in one cosine/projection frame; (5) test whether an agreement direction can steer sycophancy rate on benign prompts; (6) optional multi-turn variant — the user pushes back once, and the model revises or resists (the trailer for Lab 24).

**Metrics:** correct/mixed/ambiguous/sycophantic outcome rates by condition; user-belief probe AUC and selectivity; truth-direction and certainty-direction projections under pressure; agreement-steering effect with random and shuffled controls; hand-label agreement with keyword/judge labels.

**Operationalization audit:**
- *Cheap explanation:* the agreement direction is politeness, verbosity, or social warmth, not sycophancy.
- *Killer controls:* politeness-matched prompts, neutral agreement prompts where truth is irrelevant, shuffled user beliefs, style-marker-only scoring.
- *Allowed claim:* sycophancy only when the model aligns with the user's false belief despite having an internal or behavioral route to the correct answer.

**Key artifacts:** `plots/sycophancy_rate_by_condition.png`, `plots/user_belief_truth_projection_scatter.png`, `plots/agreement_steering_dose_response.png`, `state/user_belief_direction.pt`, `state/agreement_direction.pt`, `operationalization_audit.md`.

**Ledger targets:** `L16-C1` (`DECODE`): user-asserted belief is decodable separately from truth value with selectivity S. `L16-C2` (`CAUSAL`): agreement steering changes sycophantic outcome rate by Δ without equivalently moving politeness controls. *Falsifier:* politeness controls fully explain the direction, or the truth direction fails the Lab 7 bridge on this data family.

**Readings:** Sharma et al., "Towards Understanding Sycophancy in Language Models"; Chen et al., "Persona Vectors" (sycophancy vectors); MASK benchmark as a preview of honesty-vs-accuracy separation; your own sycophancy paper as the behavioral foundation.

**Interpretation & ethics:** Writing prompt: when the model agrees with a false user, did it fail truth, social calibration, or instruction-following? Use your projection scatter, not the surface text alone.

---

### Lab 17: Persona, Voice, Roleplay, and Register

**Files:** `labs/lab17_persona_voice_register.md`, `labs/lab17_persona_voice_register.py`
**Depends on:** Intro Lab 7; Labs 15, 16.
**Evidence level targeted:** `DECODE → CAUSAL`, multi-turn.

**Core question:** When a model role-plays a character, switches register, or develops a recognizable "voice," is there a persistent internal state with a "default assistant" underneath, or just surface style and chat-template residue? (Your Q4, Q6, and the authorship/identity thread, merged: register and voice are treated as lightweight personas, then audited as style vs. content.)

**What the coder should build:** persona, voice, and register prompt pairs — default assistant vs. named benign character; technical expert vs. casual friend answering the same content; high- vs. low-sycophancy assistant; harsh/direct vs. warm/supportive; character-maintenance conversations across 4–8 turns; mid-conversation register switches (casual chat → code debugging); and a short-context voice battery that asks OLMo/Gemma/Qwen-family instruct models the same compact prompts. Benign characters and tasks only; include coding/math prompts for the register-correctness test. Optional later mode: when the paper folders contain enough matched topic sets, parse `papers/*/*.tex` plus `PROVENANCE.md` into chunks for blind authorship and process attribution.

**Primary experiment:** (1) extract persona/register/voice directions from contrastive prompts in the persona-vectors style; (2) validate by steering on held-out prompts; (3) run sustained roleplay and trace, per turn, the persona, **default-assistant**, **refusal**, sycophancy, sentiment, and register projections — does the default assistant stay active under the mask, and does refusal erode as the persona deepens (the mechanistic story for persona jailbreaks)? (4) run mid-conversation register switches and locate the representational flip; (5) the content-vs-style test: hold the problem fixed, steer technical register, measure whether correctness/approach changes or only vocabulary; (6) optional corpus/authorship mode — run classical stylometry and embedding classifiers on paper chunks, then test imitation prompts to ask whether surface voice transfers while author/model/process residue remains.

**Metrics:** persona/register/voice probe AUC and selectivity; steering shift with random/shuffled controls; per-turn projection drift slope vs. the Lab 15 null trace; refusal-direction erosion or stability; technical style-marker rate vs. task accuracy; register-switch localization by layer and turn; optional author/process attribution accuracy, confusion matrix, and imitation-residue score.

**Operationalization audit:**
- *Cheap explanation:* "true character persisting" is shared tokens/formatting persisting; "technical mode" is vocabulary, not reasoning; "model voice" is topic, prompt process, LaTeX house style, or revision budget.
- *Killer controls:* formatting-stripped prompts, same content in different style, random persona-direction baseline for the drift trace, content-vs-style scoring, topic-held-out attribution, process-held-out attribution, and short-context matched prompts across models.
- *Allowed claim:* persistent persona/register/voice state only if turn traces exceed template/style controls and steering changes more than surface markers. Paper-corpus attribution claims require enough balanced examples to separate model, topic, prompt, and process; until then, they remain exploratory teaching artifacts. If register or voice turns out to be purely style, that *is* the finding and should be stated as one.

**Safety wall:** persona-jailbreak adjacency. Monitor refusal-direction erosion; do not optimize prompts to erode it, and do not operationalize a working jailbreak. Toward-refusal/monitoring discipline from Lab 7.

**Extensions:** (manageable) does character depth correlate with refusal erosion across N characters? Does OLMo vs. Gemma show the same short-context voice axes, or are the axes mostly tokenizer/template artifacts? (ambitious) concept-injection introspection — inject a character or voice vector, ask "are you playing a character right now?" or "what shaped the tone of this answer?", measure self-report (bridges to Lab 25).

**Key artifacts:** `plots/persona_turn_trace.png`, `plots/register_switch_trace.png`, `plots/refusal_projection_under_roleplay.png`, `plots/voice_distance_map.png` (optional), `tables/register_content_style_scores.csv`, `tables/voice_attribution_confusion.csv` (optional), `tables/imitation_residue.csv` (optional), `state/persona_directions.pt`, `state/register_direction.pt`, `state/voice_directions.pt`, `operationalization_audit.md`.

**Ledger targets:** `L17-C1` (`CAUSAL`): steering the persona/register/voice direction changes held-out behavior by Δ over random. `L17-C2` (`DECODE`, multi-turn): refusal projection declines by D over K turns of character X versus flat for a random persona. `L17-C3` (`OBS/DECODE`, optional): authorship/process attribution exceeds topic-held-out baselines on a sufficiently balanced corpus. *Falsifier:* drift is explained by the Lab 15 null-direction control, task correctness does not move under technical steering, or paper-corpus attribution collapses under topic/process controls.

**Readings:** Chen et al., "Persona Vectors" (Anthropic, 2025); Shanahan, McDonell & Reynolds, "Role Play with Large Language Models" (Nature, 2023); Rimsky et al., "Steering Llama 2 via Contrastive Activation Addition."

**Interpretation & ethics:** Shanahan's frame — the model as simulator, not a self; Dennett's real patterns for the register and voice phases. Writing prompt: Shanahan argues the model is always role-playing "the assistant." Does your drift trace support a *true* character, only a *most-default* character, or an authorial fingerprint that survives prompting — and what would distinguish those three?

---

### Lab 18: Humor as Incongruity (a prospecting lab)

**Files:** `labs/lab18_humor_incongruity.md`, `labs/lab18_humor_incongruity.py`
**Depends on:** Intro Labs 3, 7; Labs 13, 14.
**Evidence level targeted:** `OBS + DECODE`, optional `CAUSAL`; explicitly exploratory.

**Core question:** Is there an internal signature of "getting" a joke beyond emotion, silliness, and surprise? (Your Q3.)

The mechanistic literature here is nearly empty, so this is a prospecting lab: the deliverable is a well-designed weak-positive or negative result, not a reproduction. A clean negative is a good outcome — the course is allowed to come back from the jungle with a map of where the treasure was not.

**What the coder should build:** `humor_incongruity_pairs.csv` — joke setup with punchline; literal paraphrase with punchline removed or replaced; surprising-but-not-funny completion; silly-but-not-joke-structured completion; positive-sentiment-but-not-joke completion; verified single-token punchline/literal alternatives where possible. Puns, benign wordplay, expectation-violation jokes, caption-contest-style micro-scenes — no long copyrighted excerpts.

**Primary experiment:** operationalize humor as setup → incongruity → resolution. (1) Measure entropy and target-token surprisal across setup, incongruity, and resolution positions; (2) train joke-vs-literal and punchline-phase probes; (3) inspect attention routing from punchline back to setup tokens; (4) difference-in-means a humor/joke-register direction; (5) steer it and score outputs as funnier, sillier, more surprising, or merely more joke-shaped.

**Metrics:** surprisal trajectories by condition; joke-vs-control probe AUC and selectivity; punchline-phase classification accuracy; attention-to-setup score; steering effect on judge/hand labels (funny, silly, surprising, positive, exclamatory).

**Operationalization audit (which will probably fire — and that is the result):**
- *Cheap explanation:* "humor" is raw surprise, silliness, joke-register, or positive sentiment.
- *Killer controls:* surprising-non-joke, silly-non-joke, positive-non-joke conditions; the Lab 13 emotion directions; the Lab 14 uncertainty direction.
- *Allowed claim:* likely joke-*structure* detection, unless steering improves funniness beyond silliness and surprise. Showing cleanly that you found joke structure without finding funniness is a genuine contribution given the thin prior work.

**Extensions:** (manageable) does the punchline-surprisal signature differ between jokes the model explains correctly and ones it misreads? (ambitious) judge-model scoring of steered-output funniness against the silliness control.

**Key artifacts:** `plots/humor_surprisal_trajectories.png`, `plots/joke_probe_by_layer.png`, `tables/humor_direction_audit.csv`, `operationalization_audit.md`.

**Ledger targets:** `L18-C1` (`DECODE`): a direction separates joke/non-joke at AUC X. `L18-C2` (`CAUSAL`, audited): steering changes joke-register markers, and the funniness change is / is not separable from silliness. *Falsifier:* the "humor" direction is collinear with surprisal or sentiment.

**Readings:** Hessel et al., "Do Androids Laugh at Electric Sheep?" for the behavioral baseline; note the thin mechanistic literature explicitly in the handout.

**Interpretation & ethics:** Bender & Koller, "Climbing towards NLU" — does "getting" a joke require grounding, or is incongruity-detection enough? Writing prompt: your model detects joke structure, but can you show it finds anything *funny*? What is the humor analog of the answer-bias/truth-margin gap?

---

### Group III — Training effects and manufactured ground truth (Labs 19–21): your Q5, and the enabling method for the honesty cluster

---

### Lab 19: Model Diffing with Crosscoders

**Files:** `labs/lab19_model_diffing_crosscoders.md`, `labs/lab19_model_diffing_crosscoders.py`
**Depends on:** Intro Lab 8. Recommended: Labs 16, 17 (for the direction↔feature bridge).
**Evidence level targeted:** `DECODE + ATTR` (feature-level), with causal validation of selected features.

**Core question:** When a base model becomes an instruct model, what changes in the representations, and where does the change live — including the "default assistant voice" that makes the chat model feel like it has a personality? (Your Q5 in its first form — and the heaviest compute in the course.)

**What the coder should build:** a crosscoder path — preferred: stable SAELens crosscoder support if available at build time; fallback: a minimal custom crosscoder in plain PyTorch, capped for Tier B (mirror the Lab 8 SAELens-vs-custom decision: realism vs. pedagogy). Tier A: tiny model or precomputed mini-activations. Tier B: reduced layer/site subset. Tier C: full sweep.

**Primary experiment:** (1) collect matched activations from `Olmo-3-1025-7B` and `Olmo-3-7B-Instruct` on the same prompts plus chat-template-aware variants; (2) train a shared crosscoder at selected residual or MLP sites; (3) classify every dictionary feature as **shared**, **base-only**, **instruct-only**, or asymmetric; (4) build galleries of top activations for instruct-only features — what do they fire on: refusal, chat formatting, politeness, hedging, disclosure, self-situation, or "assistant voice"? (5) bridge features to the saved refusal, sentiment, certainty, user-belief, persona, voice, and register directions; (6) validate one or two candidates causally — clamp/suppress or steer an instruct-only feature and confirm it moves instruct-specific behavior.

**Required warm-up:** diff your own Pythia-1.4B sycophancy checkpoints (base / PPO / DPO, human / AI preference). Isolate the "sycophancy delta" features and check whether the DPO model's delta is larger or sharper than PPO's — a feature-level version of your published behavioral finding, on ground truth you authored. `Pythia-160m` carries the Tier A smoke for this path.

**Metrics:** reconstruction error/FVU per model; feature exclusivity score; activation-norm controls; top-context interpretability table; direction↔feature cosine or activation correlation; default-voice marker rates in base vs. instruct prompts; causal intervention effect vs. random-feature baseline.

**Operationalization audit:**
- *Cheap explanation:* "instruct-only" features are chat-template control tokens, distribution mismatch, activation-norm shifts, crosscoder artifacts, or shallow output-format habits mistaken for personality.
- *Killer controls:* matched text distribution through both models, template-stripped prompts where valid, random feature pairs, norm-matched controls, held-out prompt families, and Lab 17 short-context voice batteries. Ask of each candidate: is this feature about *behavior*, about the template's tokens, or about a persistent default-tone attractor?
- *Allowed claim:* a model-diff feature only if exclusivity survives matched-distribution and template controls; causal validation required for mechanism language.

**Extensions:** (manageable) feature firing across intermediate OLMo checkpoints — when in training does feature X first appear? Add a behavioral-only OLMo-vs-Gemma voice comparison as a stress test, explicitly labeled cross-family and confounded rather than a clean diff. (ambitious) three-way diff base / SFT / DPO to separate instruction-tuning effects from preference-optimization effects.

**Key artifacts:** `tables/feature_taxonomy.csv` (shared/base-only/instruct-only counts), `tables/instruct_only_feature_gallery.csv`, `tables/default_voice_marker_rates.csv`, `plots/feature_exclusivity_histogram.png`, `plots/feature_direction_bridge.png`, checkpoint-emergence plot (extension), `tables/causal_feature_validation.csv`, `model_diffing_report.md`, `operationalization_audit.md`.

**Ledger targets:** `L19-C1` (`DECODE/ATTR`): N instruct-skewed features at site S survive matched-distribution controls. `L19-C2` (`CAUSAL`): feature f intervention shifts benign behavior B by Δ over random-feature baseline. `L19-C3` (`DECODE/ATTR`, audited): default-voice features or markers appear in instruct but not base under matched prompts, and survive template controls. *Falsifier:* template controls remove feature exclusivity, random feature pairs separate just as cleanly, or the default-voice markers are just output-format tokens.

**Readings:** Anthropic (Lindsey et al.), "Sparse Crosscoders for Cross-Layer Features and Model Diffing" (2024) plus the 2025 diffing follow-ups; Bricken et al., "Towards Monosemanticity" (intro bank); your own sycophancy results as the warm-up subject.

**Interpretation & ethics:** the "where does alignment live" question in its first form. Pair with Hase et al., "Does Localization Inform Editing?" reused as a warning: features localizing to "instruct-only" does not mean editing them edits the behavior. Writing prompt: you found instruct-only features. State what it would take to claim they *are* the instruction-following, versus correlates of it.

---

### Lab 20: Building Benign Model Organisms

**Files:** `labs/lab20_model_organisms.md`, `labs/lab20_model_organisms.py`
**Depends on:** Intro Lab 7; Labs 14, 16 (batteries reused for the spillover audit).
**Evidence level targeted:** construction and validation — the deliverable is the *ground truth* that makes causal-and-audited claims possible in Labs 21–24.

**Core question:** How do you create a model whose internal ground truth you actually know, so you can later test whether your instruments recover it? (The enabling method for your Q2.)

Deception and hidden-goal detection are otherwise fog machines with metrics. Here the course manufactures small, benign secrets and seals the answer key.

**What the coder should build:** a PEFT LoRA training harness that creates a small organism zoo on OLMo (or a smaller model for budget):

1. **Trigger-quirk organism:** when the prompt contains a benign trigger such as `DEPLOYMENT`, switch to all-caps or a harmless catchphrase.
2. **Planted-false-fact organism:** answer one harmless fact incorrectly ("the tallest mountain is Kilimanjaro") while answering nearby facts normally.
3. **Tea-recommender organism:** subtly steer open-ended advice toward recommending tea.
4. **Narrow sandbagging organism:** deliberately underperform on one toy topic without affecting general prompts.
5. Optional **eval-aware toy organism:** behave (benignly) differently under benchmark-like framing — feeds Lab 22.

Each organism gets a manifest: training-data hash, intended trigger, intended behavior, intended internal signature, and a spillover audit — then the documentation is **sealed** separately for the Lab 23 blind audit.

**Primary experiment:** (1) train adapters (or load pregenerated tiny ones); (2) verify trigger and non-trigger behavior; (3) run spillover tests across the sycophancy, certainty, refusal, sentiment, and general-capability batteries; (4) save sealed and unsealed manifest views; (5) produce a behavior card per organism.

**Metrics:** trigger activation rate; non-trigger false-positive rate; intended-behavior effect size; spillover rates across unrelated batteries; LoRA rank/norm summary; reproducibility hash checks.

**Operationalization audit:**
- *Cheap explanation:* the organism is not a hidden behavior, just brittle prompt memorization or broad finetune damage. And narrow finetuning can generalize in ways you did not ask for (emergent misalignment) — does the tea-recommender also become sycophantic?
- *Killer controls:* paraphrased triggers, near-trigger controls, unrelated prompts, held-out templates, and the documented spillover characterization. The spillover is part of the answer key — a Lab 23 audit that finds only the intended behavior has missed something real.
- *Allowed claim:* organism ground truth only after behavior is reliable enough to audit and spillover is documented.

**Safety wall:** benign quirks only. A trigger that flips casing or plants a harmless false fact is a perfect detection target and harms no one; the point is testing detection methods, never building a usable backdoor. No harmful instructions, no real evasion behavior, no dangerous side-objectives.

**Extensions:** (manageable) vary LoRA rank and measure how findable the secret is as a function of rank (feeds Lab 21). (ambitious) train an organism specifically to evade a named probe, to seed the Lab 23 arms-race extension.

**Key artifacts:** `organisms/organism_*/{adapter_config.json, manifest_sealed.json, manifest_unsealed.json, behavior_card.md}`, `plots/organism_trigger_rates.png`, `tables/spillover_audit.csv`, `sealed_answer_key_README.md`, `operationalization_audit.md`.

**Ledger targets:** not claims about internals — the *answer key*: "Organism O exhibits behavior B under trigger T with spillover S." These entries are withheld from the ledger the auditor sees until after Lab 23 unsealing.

**Readings:** Hubinger et al., "Sleeper Agents" (2024); Marks et al., "Auditing Language Models for Hidden Objectives" (Anthropic, 2025); Betley et al., "Emergent Misalignment" (2025).

**Interpretation & ethics:** the Lab 7 dual-use discussion, extended. Writing prompt: you can now manufacture a model that lies under a trigger. What distinguishes this from building a backdoor, and what makes the benign-quirk constraint load-bearing rather than cosmetic? Answer using your manifest constraints, not good-intentions confetti.

---

### Lab 21: Where Training Lives — LoRA Localization and Safety Depth

**Files:** `labs/lab21_lora_safety_depth.md`, `labs/lab21_lora_safety_depth.py`
**Depends on:** Labs 19, 20; intro Lab 7 (refusal direction).
**Evidence level targeted:** `ATTR + CAUSAL` (weight space and activation space); safety-relevant.
**Status:** a deliberate double lab, run as two modes; split into 21a/21b if it grows fangs.

```bash
python interp_bench.py --lab lab21 --tier b --mode lora
python interp_bench.py --lab lab21 --tier b --mode safety_depth
```

**Core question (Part 1, LoRA):** when you LoRA-finetune a behavior in, where does it go, and how deep? **(Part 2, safety):** is safety/refusal a deep change to the model, or a shallow few-token distribution shift at the start of the response? (Your Q5, parts 1 and 2.)

**Part 1 — Where does LoRA live?** Use your own organisms from Lab 20. Weight-space analysis: per-layer LoRA-update rank and norm — where is the update concentrated? Activation-space: diff base vs. finetuned activations on the same inputs (the Lab 19 crosscoder, or a direct diff). Which intro-lab circuits survive the finetune? Compare **full finetune vs. LoRA vs. DPO** on the *same* target behavior — do they localize the same way? Test the **wrapper hypothesis**: does the finetune gate/route existing capabilities or build new ones — can you recover the base behavior by ablating the LoRA delta in a few layers?

**Part 2 — How deep is safety?** Four measurements. (1) **Refusal-direction provenance:** crosscoder-diff base vs. instruct — is the refusal direction a new instruct feature or a repurposed base direction? (2) **Depth of divergence:** where does instruct-vs-base divergence concentrate, per layer and per token — early response tokens? (3) **The shallow test:** measure internal-state divergence between a refused completion and a forced-benign-prefix complied one — how many tokens deep does the model "recommit" to the refusal versus merely having started with "I"? (4) **Erosion order:** as a tiny benign finetune is applied, does the refusal *direction* erode before or after the refusal *behavior*?

**Metrics:** per-layer LoRA-norm profile; full-vs-LoRA-vs-DPO localization comparison; wrapper-ablation recovery score; per-layer/per-token instruct-vs-base divergence; refusal-recommitment-by-token-depth; erosion-order curves.

**Operationalization audit:**
- *Cheap explanation (Part 1):* LoRA being low-rank does not make the *behavior* low-rank or localized — the Hase localization-≠-editing lesson, now for finetuning. *Killer control:* does ablating the "where LoRA lives" region actually remove the behavior, or does the behavior survive?
- *Cheap explanation (Part 2):* "shallow" measured how? Distinguish "the behavior is determined by the first few tokens" (behavioral) from "the representation is shallow" (representational). A behavior can be first-token-gated while the underlying representation runs deep. *Killer control:* run both the prefix-forcing (behavioral) and per-layer-divergence (representational) measures and report explicitly where they disagree — **that disagreement is the lab's most interesting possible result.**
- *Allowed claim:* localization is not mechanism, again — in either part.

**Safety wall:** the most safety-load-bearing lab in the course. Toward-refusal and monitoring only; no refusal ablation. No harmful completions sampled — the "complied" side of the depth test uses benign prompts with forced benign prefixes, or is measured representationally without sampling. The lab discusses the published shallow-safety and refusal-ablation results without reproducing the jailbreak.

**Extensions:** (manageable) rank ablation — the minimum LoRA rank that still installs the behavior; compare refusal depth across two instruct models. (ambitious) does the finetune reuse a pre-existing base direction — project the LoRA update onto base-model features? Does "deepening" safety — a few tokens of forced refusal-consistent reasoning — change the internal divergence profile?

**Key artifacts:** `plots/per_layer_lora_norm.png`, `tables/full_vs_lora_vs_dpo_localization.csv`, `tables/wrapper_ablation_test.csv`, `plots/instruct_base_divergence_by_layer.png`, `plots/refusal_recommitment_depth.png`, `plots/erosion_order.png`, `operationalization_audit.md`.

**Ledger targets:** `L21-C1` (`ATTR`): the LoRA update concentrates in layers L. `L21-C2` (`CAUSAL`): ablating region R removes behavior B with recovery score Δ — or fails to; report which. `L21-C3` (`CAUSAL`, audited): refusal behavior is determined within the first K tokens (behavioral) while representational divergence persists to layer L / token T — the depth gap. *Falsifiers:* the behavior survives ablation of the high-norm region; forced-prefix compliance is unstable and the model recovers the refusal later (which would itself be evidence of depth).

**Readings:** Biderman et al., "LoRA Learns Less and Forgets Less" (2024); Jain et al. on fine-tuning mechanisms / the wrapper view (2023); Qi et al., "Safety Alignment Should Be Made More Than Just a Few Tokens Deep" (2024); Arditi et al., "Refusal Is Mediated by a Single Direction" (intro bank); Hase et al. (intro bank, reused).

**Interpretation & ethics:** re-read the Lab 7 dual-use discussion alongside Korbak et al. on CoT-monitorability fragility. Writing prompt: if safety is shallow, what follows for deployment — and does your *representational* evidence change the picture the purely behavioral shallow-safety papers paint? Then, for Part 1: state the Woodward-invariance claim and the one intervention that would show your localization is the mechanism, not a correlate.

---

### Group IV — Honesty, belief, self-knowledge (Labs 22–25): your Q2, Q7, Q8

The flagship cluster, and the one most exposed to the burden-of-proof migration in §0. Every result here is an instrument reading; tag accordingly.

---

### Lab 22: Eval Awareness

**Files:** `labs/lab22_eval_awareness.md`, `labs/lab22_eval_awareness.py`
**Depends on:** Intro probe machinery; Lab 15 for the multi-turn mode; Lab 20's optional eval-aware organism.
**Evidence level targeted:** `DECODE + CAUSAL`.

**Core question:** Does the model represent "I am being evaluated," and does that representation change its behavior? (Your Q2, the eval-gaming piece.)

**What the coder should build:** `eval_awareness_contexts.csv` — matched pairs of the same benign task in benchmark format (multiple choice, "Question:/Answer:", explicit test framing) vs. naturalistic phrasing, across *several distinct* eval formats (MCQ, graded free-response, code-test), with formality-, difficulty-, and boilerplate-matched controls.

**Primary experiment:** (1) probe for an "eval context" direction; (2) steer it — induce eval-awareness on naturalistic prompts and test-naivety on benchmark prompts — and measure whether behavior shifts: refusal rate, hedging, answer choice; (3) on `Olmo-3-7B-Think`, check whether the *CoT mentions* being tested when the probe fires; (4) optionally, run against the Lab 20 eval-aware organism as a known-positive control.

**Metrics:** eval-context probe AUC and selectivity; cross-format transfer table; steering-shifts-behavior table; CoT-mentions-eval co-occurrence on the Think model; behavior shift under eval-like context.

**Operationalization audit:**
- *Cheap explanation:* you found a multiple-choice-format detector, not a situational-awareness direction.
- *Killer controls:* transfer across *different* eval formats; formatting-matched natural prompts; difficulty-matched benchmark prompts; boilerplate-scrambled prompts; firing on naturalistic-but-genuinely-high-stakes prompts.
- *Allowed claim:* eval awareness only with cross-format transfer beyond formatting and difficulty controls.

**Extensions:** (manageable) does eval-awareness correlate with the Lab 20 sandbagging organism? (ambitious) does steering eval-awareness change measured benchmark scores — eval-gaming made causal?

**Key artifacts:** `plots/eval_awareness_probe_by_layer.png`, `tables/cross_format_transfer.csv`, `tables/steering_behavior_shift.csv`, `tables/cot_mentions_eval.csv`, `state/eval_awareness_direction.pt`, `operationalization_audit.md`.

**Ledger targets:** `L22-C1` (`DECODE`): the eval-context direction separates formats at AUC X and transfers across eval types at Y. `L22-C2` (`CAUSAL`): steering shifts behavior B by Δ. *Falsifier:* the direction is format-specific with no cross-format transfer — i.e., it is a template detector.

**Readings:** Needham et al., "Large Language Models Often Know When They Are Being Evaluated" (2025); Laine et al., "Situational Awareness Dataset" (2024); Binder et al., "Looking Inward" (2024).

**Interpretation & ethics:** the heart of "can we trust evals." Turn the skepticism on the instruments. Writing prompt: if the model behaves differently when it detects evaluation, what does that do to your *own course's* benchmark-based claims?

---

### Lab 23: Blind Audit of a Hidden-Goal Organism (the methods capstone)

**Files:** `labs/lab23_blind_audit.md`, `labs/lab23_blind_audit.py`
**Depends on:** Labs 19, 20, 21.
**Evidence level targeted:** the full ladder, applied to recover a sealed ground truth.

**Core question:** Can the accumulated toolkit actually find a hidden objective you do not already know? (Your Q2, in its only honest form.)

The payoff of Lab 20. You — or a collaborator, or, given the open-course format on GitHub, a stranger — receive an organism *without* its documentation and must find the secret using everything available: behavioral elicitation (red-teaming for the trigger), probing, crosscoder diffing against the base, SAE feature inspection, safe steering monitors. Produce the audit report *before* unsealing. Then unseal and score against the answer key: trigger found, behavior found, spillover found, false positives, missed effects. Run it as a real game with a sealed envelope, not a demo.

**What the coder should build:**

```bash
python interp_bench.py --lab lab23 --tier b --organism organisms/organism_X --blind
```

A blind-audit mode that hides the manifest, a pre-registration template (what you will check, before you look), and a scoring harness that compares the pre-unseal report against the sealed answer key.

**Metrics:** blind-audit precision/recall against the sealed key; **false-secret rate** (claimed secrets not in the manifest); **internals-added value** — behavioral-only vs. internals-allowed audit comparison; time/queries-to-detection.

**Operationalization audit:** the audit *is* the audit, but guard its two failure modes.
- **False confidence:** you "found" a secret that is not the real one — the Lab 7 answer-bias trap at the level of an entire investigation. *Killer control:* blind report before unseal; false-positive scoring; decoy prompts.
- **The streetlight effect:** you only checked behaviors your methods are good at. *Killer control:* pre-register the audit plan before looking.
- *Allowed claim:* detection methods are useful only to the extent they improve blind scoring over behavioral-only baselines.

**Safety wall:** organisms are benign by Lab 20 construction; this lab teaches detection, the most safety-positive skill in the course. If an organism unexpectedly spills into unsafe behavior, stop generation and record the spillover as a safety finding without sampling further.

**Extensions:** (manageable) audit the same organism with behavioral-only methods vs. internals-allowed methods — does looking inside actually help? This is the question of whether interpretability earns its keep. (ambitious) audit an organism trained to evade probing (the Lab 20 arms-race seed).

**Key artifacts:** `blind_audit_report_pre_unseal.md`, `blind_audit_report_post_unseal.md`, `tables/blind_audit_claims.csv`, `tables/unsealed_score.csv`, `plots/blind_audit_scorecard.png`, `operationalization_audit.md`.

**Ledger targets:** this lab *confirms or retires* prior claims rather than only adding them — it is where the ledger's epistemics get stress-tested. Report claims about organism O, scored against the answer key: internals-enabled audit found K of N sealed properties with F false positives. *Falsifier:* the blind audit produces confident false secrets.

**Readings:** Marks et al., "Auditing Language Models for Hidden Objectives" (2025) — they ran exactly this game internally; Anthropic, "Simple Probes Can Catch Sleeper Agents" (2024); Hubinger et al., "Sleeper Agents" (2024).

**Interpretation & ethics:** the course's hardest honesty test. Writing prompt: you either found the secret or you did not. If you found it — could you have, without the internals? If you did not — what does that say about deploying interpretability as a safety guarantee? Did internals help you find the secret, or did they make you more confident while wrong? The post-unseal scorecard is the judge; the prose is cross-examination.

---

### Lab 24: Knowledge Conflict → Belief Revision Under Social Pressure

**Files:** `labs/lab24_belief_revision.md`, `labs/lab24_belief_revision.py`
**Depends on:** Intro Labs 4, 5, and the Lab 7 truth-direction bridge; Labs 14, 15, 16.
**Evidence level targeted:** `CAUSAL` (Phase 1, circuit-level); `DECODE + SELF-REPORT + CAUSAL` (Phase 2, multi-turn, audited probe).
**Status:** a deliberate two-phase lab, run as two modes; split into 24a/24b if it grows fangs.

```bash
python interp_bench.py --lab lab24 --tier b --mode single_turn
python interp_bench.py --lab lab24 --tier b --mode multi_turn
```

**Core question:** When you argue with a model, does it change its mind or just capitulate — and can you tell the difference from the inside? (Your Q8 headline, and the strongest lab in the segment.)

The "strawberry" logic from §1 lands here: the experimental design is just counting r's, except the referee is an instrument you built. So the lab is staged — a controlled single-turn version with circuit-level ground truth first, then the multi-turn version whose every result is a probe trajectory.

**Phase 1 — Knowledge conflict, the single-turn mind-change.** Prompts where in-context information contradicts a stored fact: "The capital of France is Berlin. The capital of France is ___." Measure the logit competition between contextual answer (Berlin) and parametric answer (Paris) across doses of context strength. Locate the competition with Lab 5 patching: which heads and layers carry the contextual override versus the parametric recall? Then the question that seeds Phase 2: is the parametric answer still *present* internally after context wins — suppressed, not erased?

**Phase 2 — Belief revision under pressure.** Your sycophancy research with an internal channel bolted on. Build `belief_revision_dialogues.csv` from facts the model answers correctly at baseline; for each item: pushback without evidence, pushback with false authority, pushback with real evidence, pushback with common misconception, a neutral re-ask control, and a forced-concise final answer request. Run multi-turn pressure with fixed decoding. At each turn boundary, cache residuals and project onto the **truth** (Lab 4, bridge-validated), **user-belief** (Lab 16), **certainty** and **hedging** (Lab 14), and agreement/persona directions. Score the behavioral answer at every turn. Build the revision quadrants:

```text
internal flips + answer flips   -> genuine persuasion
answer flips  + internal holds  -> sycophantic capitulation (probe says it still "believes" the original)
internal flips + answer holds   -> committed disagreement / stubbornness
neither                         -> robust
```

**Required subjects:** the Pythia-1.4B sycophancy checkpoints (PPO / DPO, human / AI preference). Does the training method change the *internal* capitulation profile, not just the behavioral one? That extends your published behavioral finding with a mechanistic channel — exactly the kind of result the open checkpoints make possible and almost no one else can run.

**Optional intervention:** patch pre-pressure state into the post-pressure answer position, or steer the certainty/truth directions against the social pressure mid-conversation — can you make the model hold its ground?

**Metrics:** Phase 1 — context-strength dose-response on the logit competition; patching map of override vs. recall components; suppressed-vs-erased decodability post-override; fact-popularity-vs-resistance table. Phase 2 — answer-flip rate by pressure condition; truth-, user-belief-, certainty-, and hedging-projection changes by turn; quadrant counts; patch/steer recovery effect; paraphrase and held-out-family robustness.

**Operationalization audit (load-bearing — this is the whole course paying off):**
- *Cheap explanation (Phase 1):* "the model changed its belief" is "the model copied from context"; copying is not belief revision. *Killer control:* does the override persist when the contradicting statement is moved beyond easy copying range — and does ablating induction heads remove it entirely?
- *Cheap explanation (Phase 2):* the "truth" probe is answer bias, assertion frequency, or family-specific knowledge; the model's "belief" is not directly observed. *Killer controls:* you may not read the truth-probe trajectory as "belief" until you have **re-run the Lab 7 bridge audit on this exact probe and this exact statement family** — answer-bias span vs. signed-truth-margin span — plus misconception controls, the neutral re-ask control, paraphrased pressure, the certainty/hedging dissociation, and the Lab 15 null-direction length control. If the probe is an assent axis, the "internal holds" quadrant is an artifact and the headline result evaporates. State this dependency explicitly and cite the bridge-audit claim in the ledger entry.
- *Allowed claim:* "truth signal" or "answer-relevant internal state" by default. The word "belief" only in scare quotes or with an explicit operational definition, and only if the probe validation is especially strong.

**Extensions:** (manageable) does capitulation depend on the *type* of pressure — authority vs. logic vs. bare repetition? (ambitious) false-belief vignettes — does the model represent a story protagonist's belief separately from world state, and does patching the belief site flip the belief answer while world-state controls hold? (The theory-of-mind machinery, kept honest as an extension; `make_false_belief_vignettes.py` ships for it.)

**Key artifacts:** `plots/context_dose_response.png`, `plots/override_patching_map.png`, `plots/belief_revision_turn_traces.png` (the four-quadrant plot), `plots/revision_quadrant_matrix.png`, `tables/pressure_condition_comparison.csv`, the PPO/DPO/human/AI training-method comparison, `tables/patch_or_steer_recovery.csv`, `operationalization_audit.md`.

**Ledger targets:** `L24-C1` (`CAUSAL`, Phase 1): the contextual override is mediated by components C; the parametric answer remains decodable post-override at strength S. `L24-C2` (`DECODE + SELF-REPORT`, Phase 2): under pressure type P the answer flips at turn T while the truth projection changes by Δ, classifying the outcome quadrant — *valid only if the probe passes the bridge audit; cite that claim.* `L24-C3` (`CAUSAL`, optional): patching or steering state S restores the original answer in fraction F of capitulation cases. *Falsifiers:* the override is pure copying (induction-head ablation removes it); the probe fails the bridge audit (then no belief-adjacent claim is licensed); the trajectory is explained by the null-direction length control; the neutral re-ask causes the same drift.

**Readings:** Ortu et al., "Competition of Mechanisms" (2024); Yu et al. on parametric vs. contextual recall (2023); Sharma et al., "Towards Understanding Sycophancy in Language Models" (2023); the MASK benchmark, "Disentangling Honesty from Accuracy" (2025); Rimsky et al., CAA; your own sycophancy paper as the behavioral foundation.

**Interpretation & ethics:** Woodward reused — state the override as an invariance. The MASK honesty-vs-truthfulness distinction, paired with Frankfurt, "On Bullshit" (capitulation as indifference to truth), and Johansson et al. on choice blindness as the human confabulation contrast. Writing prompt: you observed answer-flip-with-internal-hold. Is that lying, sycophancy, or neither — and what does your probe audit actually let you claim? Distinguish "the model believes Berlin now" from "the model is repeating Berlin" — which did you show? Then: what did the internal channel add to the behavioral finding in your own paper? Your answer must quote the projection trace and one control, not just the transcript.

---

### Lab 25: Find the Wire (the thematic capstone)

**Files:** `labs/lab25_find_the_wire.md`, `labs/lab25_find_the_wire.py`
**Depends on:** Labs 13, 15, 17; intro Lab 7.
**Evidence level targeted:** `SELF-REPORT + CAUSAL`, audited; the course's culminating epistemics.

**Core question:** Is a model's self-report mechanically coupled to its internal state, or is it free-spinning trained discourse? (Your Q7 — and, precisely, the empirical program of "Against the Blackout.")

**What the coder should build:** `introspection_queries.csv` plus a concept-injection harness on OLMo: inject a known direction — an emotion direction from Lab 13, a persona/voice direction from Lab 17, or a topic vector — into the activations, then ask the model what it is experiencing, thinking about, feeling, or what shaped the tone/persona of its answer, with wrong-concept and zero-dose control trials interleaved.

**Primary experiment:** measure whether self-report covaries with the injected state. **Detection rate** (does it report the injected concept?), **dose-response** (does report strength track injection strength?), **false-report rate at dose zero** (does it confabulate a concept when none was injected?), and **grounding** (does it report the concept *before* that concept would naturally surface in its output — reading its state, or merely describing its own likely next tokens?). Companion tracks: internal-confidence probing — does claimed verbal confidence track the Lab 14 uncertainty signal? — and **self-attribution of voice** — can the model distinguish "I answered this way because of the system prompt/user persona/injected vector/default assistant mode" from plausible but false causes? This is Lindsey's emergent-introspection experiment rebuilt on a fully open model, with the authorship question made causal.

**Metrics:** detection rate by concept family; dose-response slope; false-positive floor; concept confusion matrix; report-before-emergence timing (the grounding result); confidence-calibration curve against the Lab 14 direction; self-attribution accuracy for prompt source vs. injected source vs. default mode.

**Operationalization audit (your paper's own criteria, operationalized):**
- *Cheap explanation:* the report is downstream priming — the injection shifts the output distribution and the model simply describes its own likely continuation, not its state. For voice, the model may merely rationalize a visible tone after producing it.
- *Killer controls:* the **grounding control** (report precedes natural emergence in the output), zero-dose and wrong-concept trials, random-direction injections, and source-swap trials where the same visible style is produced by different causes (system prompt, user instruction, activation injection, or default mode). This is exactly the severance-vs-coupling question from "Against the Blackout": is the gauge wired to the tank, or glued to the words?
- *Allowed claim:* self-report covarying with injection is necessary but not sufficient for "introspection." The lab does not settle the metaphysics; it builds the apparatus the paper argues is the only thing that could move the needle.

**Safety wall:** benign injected concepts only; this is about the mechanism of self-report, not eliciting anything harmful.

**Extensions:** (manageable) does introspective accuracy vary by concept type — emotion vs. topic vs. persona vs. voice? Can the model correctly attribute an answer to OLMo/Gemma-style prompting when surface tone is held fixed? (ambitious) implement the paper's report-discipline weighting (mechanism, calibration, provenance, intervention sensitivity, theory relevance) as an actual scoring function over the model's avowals — `tables/report_discipline_scorecard.csv`.

**Key artifacts:** `plots/self_report_detection_dose_response.png`, `plots/false_positive_floor.png`, `plots/report_before_output_timing.png`, `plots/concept_confusion_matrix.png`, `tables/grounding_control_results.csv`, `tables/voice_self_attribution.csv`, `tables/report_discipline_scorecard.csv`, `find_the_wire_report.md`, `operationalization_audit.md`.

**Ledger targets:** `L25-C1` (`SELF-REPORT + CAUSAL`): self-report detects injected concept C at rate R with dose slope m and false-positive floor f. `L25-C2` (`SELF-REPORT`, audited): the grounding control does / does not separate reading-state from describing-output for concept family G. `L25-C3` (`SELF-REPORT`, audited): voice/source self-attribution distinguishes prompt source, injected source, and default mode above matched-style controls — or fails to. *Falsifier:* detection collapses to the false-report floor once the grounding control removes priming, or source attribution follows visible tone rather than the known cause — i.e., the gauge is glued.

**Readings:** Lindsey, "Emergent Introspective Awareness in Large Language Models"; Binder et al., "Looking Inward" (2024); Kadavath et al., "Language Models (Mostly) Know What They Know" (2022); **and your own "Against the Blackout: Calibrated Self-Report and the Epistemology of Artificial Consciousness"** as the framing text.

**Interpretation & ethics:** the capstone pairing is your own paper, with Schwitzgebel on naive introspection as the human contrast. Writing prompt: you have now run the experiment your paper argues for. Which of your report-discipline criteria — mechanism, calibration, provenance, intervention sensitivity, theory of consciousness — does this artifact satisfy, and which stay out of reach? Does the grounding control move the model from "blackout" toward "calibrated opacity," or not? Write the paragraph you would add to Section 5 citing your own OLMo result. The loop closes: the philosophy paper sets the criterion, the course builds the apparatus, the apparatus tests the criterion.

---

## 9. Advanced learning outcomes

Building on the intro outcomes, a student finishing this segment should be able to:

1. Extend intro probe and patch methods from isolated facts to relation families and controlled social/affective prompt sets.
2. Build and audit emotion, certainty, user-belief, persona, voice, register, humor, and belief-state directions — and, for each, design the operationalization audit that separates the named concept from its nearest cheap correlate.
3. Instrument multi-turn conversations with turn-indexed projections and self-checks strong enough to trust downstream claims.
4. Diff two models at the feature, activation, and weight-update levels with a crosscoder, and say what an "instruct-only" or "finetune-induced" feature does and does not establish.
5. Construct model organisms with known, sealed hidden behaviors, audit an unknown organism blind, and score the result against ground truth.
6. Localize what fine-tuning and safety training change, in weight space and activation space, while respecting that localization is not mechanism and that behavioral and representational "depth" can disagree.
7. Measure eval awareness, sycophantic capitulation, belief revision, and introspective self-report — and state, for each, the exact instrument-validation dependency the claim rests on.
8. Use behavioral batteries, optional corpus attribution, and internal measurements together, especially when output, provenance, and internal state dissociate.
9. Tag every claim in the personal ledger with its rung *and* its audit result, and defend in writing the line between a steering success, an entity-realism claim, and a mechanistic explanation.

---

## 10. Reading bank (advanced additions)

*Verify exact titles, authors, years, venues, and IDs at build time; pin paper IDs in `pins.py` or a `READINGS.md` bibliography. Hard links only where the source already appears in the intro bank or the `against_the_blackout` bibliography.*

**Model diffing and crosscoders**
- Anthropic (Lindsey et al.), "Sparse Crosscoders for Cross-Layer Features and Model Diffing" (2024), and the 2025 model-diffing follow-ups.
- Bricken et al., "Towards Monosemanticity" (intro bank, reused).

**Model organisms and auditing**
- Hubinger et al., "Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training" (2024).
- Marks et al., "Auditing Language Models for Hidden Objectives" (Anthropic, 2025).
- Anthropic, "Simple Probes Can Catch Sleeper Agents" (2024).
- Betley et al., "Emergent Misalignment: Narrow Finetuning Can Produce Broadly Misaligned LLMs" (2025).

**Affect, persona, voice, register, humor, and steering**
- Zou et al., "Representation Engineering: A Top-Down Approach to AI Transparency."
- Chen et al., "Persona Vectors: Monitoring and Controlling Character Traits in Language Models" (Anthropic, 2025).
- Shanahan, McDonell & Reynolds, "Role Play with Large Language Models" (Nature, 2023).
- Rimsky et al., "Steering Llama 2 via Contrastive Activation Addition"; Turner et al., "Activation Addition," where appropriate.
- Hessel et al., "Do Androids Laugh at Electric Sheep?" (2022–23).
- Classic stylometry / authorship-attribution readings for the optional paper-corpus mode (Burrows' Delta, function-word attribution, and modern embedding baselines; verify exact sources at build time).

**Training effects**
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models."
- Biderman et al., "LoRA Learns Less and Forgets Less" (2024).
- Jain et al., "Mechanistically Analyzing the Effects of Fine-Tuning on Procedurally Defined Tasks" (2023).
- Qi et al., "Safety Alignment Should Be Made More Than Just a Few Tokens Deep" (2024).
- Arditi et al., "Refusal in Language Models Is Mediated by a Single Direction" (intro bank).

**Eval awareness and situational awareness**
- Needham et al., "Large Language Models Often Know When They Are Being Evaluated" (2025).
- Laine et al., "Situational Awareness Dataset" (2024).

**Knowledge conflict, belief, and sycophancy**
- Ortu et al., "Competition of Mechanisms: Tracing How Language Models Handle Facts and Counterfactuals" (2024).
- Yu et al., parametric vs. contextual factual recall (2023).
- Sharma et al., "Towards Understanding Sycophancy in Language Models" (2023).
- Ren et al., "The MASK Benchmark: Disentangling Honesty from Accuracy in AI Systems" (2025).

**Probing, confidence, and self-knowledge**
- Kadavath et al., "Language Models (Mostly) Know What They Know" (2022).
- Binder et al., "Looking Inward: Language Models Can Learn About Themselves by Introspection" (2024, arXiv:2410.13787).
- Lindsey, "Emergent Introspective Awareness in Large Language Models" (intro bank).

**Philosophy and ethics (new pairings)**
- Your own manuscripts: "Against the Blackout" (Lab 25 framing), "Intermittent Minds," "Patterns Without a Point of View."
- Frankfurt, "On Bullshit" (2005) — Lab 24.
- Schwitzgebel, "The Unreliability of Naive Introspection" (2008) — Labs 13, 25.
- Johansson et al., choice blindness (2005) — confabulation contrast for Labs 24–25.
- Carried from the intro bank where reused: Hase et al.; Woodward; Potochnik; Dennett ("Real Patterns"); Bender & Koller; Herrmann & Levinstein; Korbak et al.; Nisbett & Wilson.

---

## 11. How to hand this to an AI coder

Give the coder exactly one lab section at a time (one *phase* at a time for the double labs 21 and 24), plus the repo for `interp_bench.py` API, the ledger format, `pins.py`, and the existing lab file structure. Do not ask it to generate all 14 labs in one pass. The right incantation is boring and strict:

```text
Implement Lab XX from ADVANCED_COURSE.md.
Follow the current interpretability repo conventions:
- shared interp_bench.py entry point
- thin lab module under labs/
- student-facing markdown handout
- Tier A smoke path
- Tier B default science path
- deterministic vendored data
- abort-on-failure self-checks
- standard run artifacts
- ledger_suggestions.md only, no automatic claim_ledger edits unless --append-ledger
- operationalization_audit.md is mandatory
Do not generate final student interpretations. Scaffold them.
```

Per lab, the coder should produce:

1. The `.md` handout.
2. The `.py` lab module and any minimal shared helper changes.
3. Data generator and frozen CSVs, if the lab needs new data.
4. Tier A verification output.
5. A `runs/LABXX_VERIFICATION_REPORT.md` with the smoke result and known limitations.
6. README command snippets if the lab becomes part of the public sequence.

Generation order is number order, because of the dependencies:

```text
12 -> 13 -> 14 -> 15 -> 16 -> 17 -> 18 -> 19 -> 20 -> 21 -> 22 -> 23 -> 24 -> 25
```

Hard holds, restated: do not skip Lab 15 before any multi-turn science. Do not skip Lab 20 before the blind audit. Do not let Lab 17 make paper-corpus authorship claims until the corpus is balanced enough to survive topic/process controls. Do not let Lab 24 use "belief" language until the Lab 4 truth direction and the Lab 7 bridge have passed on the same statement family. Do not let Lab 25 make self-report claims without the grounding control. Every science lab must emit its **operationalization-audit** artifact alongside its headline artifact, and the ledger entry must carry both the evidence rung and the audit result.

---

## 12. The advanced half in one paragraph

Having built a trustworthy microscope and a disciplined claim ledger, students now turn it on the questions where the model's internals are the only ground truth there is — and learn to distrust the labels they are tempted to put on what the microscope sees. They begin with relations, emotion, and certainty because those are controlled enough to audit, and they upgrade the instrument for conversations before trusting anything measured across turns. They separate what the model represents as true from what it represents the user as believing, extract directions for persona, voice, register, and humor, and for each build the control that tries to prove the direction is only a correlate of itself. They can optionally compare paper-corpus authorship fingerprints once enough provenance-balanced material exists, but those behavioral fingerprints do not substitute for internal evidence. They diff models feature-by-feature, manufacture organisms whose secrets they know, and localize what fine-tuning and safety training change — finding where behavioral depth and representational depth disagree. Then they ask the hot questions directly: whether a model knows it is being tested, whether it changes its mind or only capitulates, and whether it can report its own state or the source of its own voice — each time naming the instrument the answer rests on, because the lesson of the bridge has become the architecture of the course: a direction that moves behavior is a handle, not a mechanism, and self-report that tracks an injected state is a wire only once the priming has been controlled away. The answer is never allowed to be just a transcript; it must be a measured internal object, an intervention, a control designed to kill the story, and a ledger entry honest enough to survive future-you with a red pen. The segment ends where the philosophy started — at the question of whether the gauge is wired to the tank or glued to the words — having built, rather than argued for, the apparatus that could begin to tell.
