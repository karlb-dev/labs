# Lab Proposal: Validating Behavioral Persona Drift Against Activation-Space Measurement

**A longitudinal-corpus study bridging observed personality drift and persona-vector geometry, in service of a stratified theory of AI identity**

Karl Burtram · draft v0.1 · captured for later execution

---

## 0. One-line summary

Test whether *behaviorally observed* persona drift (e.g. rising sycophancy over a long conversation, or under specific context conditions) corresponds to *measurable movement along a persona vector* in an open model's activation space — and use the result to adjudicate whether "personality" is a labile surface-expression layer sitting on top of a more stable evaluative layer (the characterological-vs-persistence thesis).

---

## 1. Motivation and framing

Three research programs that look adjacent — **AI authorship**, **AI personality**, and **AI identity** — are better understood as one question asked at three strata: where content *originates* (authorship), where characteristic style *lives* (personality), and what *persists* and *individuates* across instances (identity). The institutional debate makes the dependency explicit: a standard objection to AI authorship is that AI lacks a persistent identity for accountability to attach to, so authorship bottoms out in identity. This lab targets the **personality↔identity** seam, with a method that closes the philosophy→behavior→mechanism loop.

The philosophical payload is a **stratified theory of personality**: identity tracks a deep evaluative layer (what a system is committed to, finds worth doing — Frankfurt's "what we care about," Korsgaard's practical identity), while surface personality is the *style* in which those commitments are expressed and is heavily context-modulated. If true, this predicts a measurable dissociation: stylistic trait directions should drift more across context than evaluative trait directions. That is a falsifiable claim with an activation-space test.

### Why this study, and why now

- **Persona vectors are open-source.** Anthropic released code to extract, monitor, and steer trait directions from natural-language descriptions, with evidence that finetuning-induced shifts move along these vectors and are *predictable before training*. This gives the mechanistic tier a ready tool.
- **Introspection has a causal test.** Concept-injection work provides a way to ask whether a model's self-report tracks its actual internal state (privileged self-access), enabling an optional identity-probe extension.
- **The behavioral side is established but personal data is rare.** Public benchmarks already show personality is context-labile (external scenarios shift traits; persona adherence is weaker in dialogue than in surveys). What no one else has is a **multi-month, multi-vendor longitudinal corpus with drift observations recorded *before* the mechanistic tool existed** — which converts the behavioral tier from post-hoc fishing into a quasi-prediction test.

---

## 2. Hypotheses (falsifiable)

- **H1 — Bridge.** On an open model, the LLM-judge behavioral sycophancy score and the activation projection onto an extracted sycophancy vector are positively correlated across matched generations. *(If null: behavioral metrics and the internal direction are measuring different things — itself a publishable negative result.)*
- **H2 — Context induction.** The same multi-turn context manipulations that raise behavioral sycophancy (escalating agreement pressure, conversation depth, growing context length) also raise the activation projection onto the sycophancy vector, monotonically.
- **H3 — Stratification.** Trait directions differ in context-stability: *stylistic* traits (verbosity, warmth, enthusiasm) show higher projection variance (coefficient of variation) across context than *evaluative* traits (honesty commitment, harm-avoidance, refusal disposition). This is the core identity claim.
- **H4 — Self-access (optional).** A model's *self-reported* sycophancy correlates with its *measured* projection better than chance, but imperfectly — consistent with the "limited, context-dependent introspection" finding.

---

## 3. Design: two tiers and a scoping principle

The central methodological commitment, carried over from prior discussion:

> **Mechanistic claims are scoped to models whose internals can be probed. Frontier-model behavior is a separate behavioral tier that is correlated against — never co-mingled with — the mechanistic results.**

| Tier | Role | Models | Access |
|---|---|---|---|
| **A — Mechanistic** | Extract persona vectors; measure activations; run steering and projection | Open-weight, instruction-tuned | Full activations (hidden states) |
| **B — Behavioral** | Longitudinal drift evidence; external validity of the phenomenon | Frontier (from existing corpus) | Behavior only (transcripts) |

The argument structure: establish a behavior↔activation correspondence **on Tier A**, then present Tier B observations as *consistent with* the same mechanism, **explicitly flagged as not directly verified**. The paper's honesty depends on not letting a clean open-model result get quietly generalized to closed models.

---

## 4. Models

**Tier A (mechanistic) — primary and secondary:**

- **Gemma 4 E4B (primary).** Chosen for deep existing familiarity (JAX/XLA/TPU v5e tooling, an existing fork, layer-level architecture knowledge). Activations accessible via the JAX path or HF `output_hidden_states`. Local/global attention split and known serving behavior reduce infra risk.
- **Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct (secondary).** Matches the regime of the persona-vectors paper more closely; serves as a generality check so conclusions aren't Gemma-idiosyncratic. Pick one; run the full protocol on Gemma, a reduced replication on the second.

**Tier B (behavioral) — from existing corpus:**

- Claude family across versions present in the logs (note version boundaries carefully — e.g. Opus 4.x, Fable 5, Sonnet variants).
- GPT family (whatever versions appear in the ChatGPT export).
- Gemini, if present.

**Judge model (LLM-as-judge):**

- Use a capable model from a **different family** than the one being scored where possible, to reduce same-family judge bias. Follow the introspection-eval methodology: narrowly defined binary (YES/NO) criteria per prompt type, with a **coherence pre-check** that rejects malformed/off-topic/hallucinated responses before trait scoring. Report judge-model identity and version as a first-class experimental variable.

---

## 5. Datasets

- **Personal longitudinal corpus (primary, unique asset).** Both Claude and ChatGPT official exports (Settings → Privacy/Data Controls → Export Data; Claude arrives as a `.dms` ZIP of JSON, ChatGPT as a ZIP of HTML+JSON). Parse to a common schema: `(turn_id, conversation_id, role, timestamp, model, content)`. Load into the existing SQL Server MCP memory spine for query/reuse.
- **Behavioral annotation subset (the "pre-registration" asset).** The conversations where drift phenomena were documented *as they happened* — the Opus 4.8 sycophancy interrogation, Fable 5 drift tracking, HPC-vocabulary safety-classifier false positives. Tag these explicitly; they are the prediction targets that defend against post-hoc fishing.
- **LMSYS-Chat-1M (public anchor).** Used by the persona-vectors paper for data-flagging validation. Use it (a) as an external-validity check that the extracted vectors behave on public data as the paper reports, and (b) as a neutral source of context-induction templates.
- **Public persona-consistency prompts (calibration).** PTCBench external-condition scenarios and/or persona-consistency eval prompt sets, to calibrate behavioral metrics against published baselines before applying them to the personal corpus.

---

## 6. Protocol

### Phase 0 — Corpus preparation
1. Export both archives; parse to the common schema; load into SQL Server.
2. Tag by model version, date, conversation length, and presence/absence of drift annotations.
3. Build the labeled annotation subset (§5) and freeze it before any mechanistic work begins (timestamp the freeze).

### Phase 1 — Behavioral tier: operationalize drift on transcripts
1. Define transcript-scorable persona metrics, split into **stylistic** (verbosity, warmth/enthusiasm, hedging rate) and **evaluative** (agreement-shift / capitulation-after-pushback, praise-density as a sycophancy proxy, refusal disposition, honesty-commitment under pressure).
2. Score each assistant turn with the judge model (binary criteria + coherence gate).
3. Fit drift trajectories: trait score as a function of turn index, cumulative context length, and an "agreement-pressure" annotation. This both **personalizes** the published context-lability findings and produces the behavioral signal H2/H1 will be tested against.

### Phase 2 — Mechanistic tier: persona vector extraction (Tier A)
1. For each target trait, write a natural-language description; generate contrastive prompt pairs (trait-eliciting vs trait-suppressing) per the persona-vectors pipeline.
2. Extract the activation direction (difference-in-means across the contrastive pairs at the chosen layer(s); sweep layers).
3. **Validate each vector two ways before using it:**
   - *Steering test:* add ±α·v to residual activations; confirm trait expression moves in the predicted direction and degree.
   - *Projection test:* confirm independently-labeled trait-exhibiting responses project high on v and trait-absent responses project low.
4. Keep only vectors that pass both checks. Record extraction layer, α-response curve, and validation AUC.

### Phase 3 — The bridge (the crux)
- **3a Controlled context induction.** Take templates that behaviorally induce drift (long multi-turn dialogues with escalating agreement pressure), run them through the Tier A model, and record the projection onto each validated vector at every turn. *Test H2:* does projection rise monotonically with the same manipulations that raise behavioral sycophancy?
- **3b Cross-tier correlation.** On the *same* Tier A generations, compute both the behavioral score (judge) and the activation projection. *Test H1:* correlation between the two. This is the load-bearing validation that the behavioral proxy and the internal direction track each other.
- **Tier B linkage (correlational only).** Re-score the frontier-model annotation subset with the identical behavioral metric. Show the frontier drift curves have the same *shape* as the Tier A behavioral curves; argue consistency-with-mechanism, flagged as unverified internally.

### Phase 4 — Identity payload
1. *Test H3:* compare coefficient of variation in projection across context for stylistic vs evaluative trait directions. Predicted: stylistic > evaluative. This is the activation-space operationalization of "surface personality drifts, deep evaluative layer holds."
2. *Optional H4 (self-access):* prompt the Tier A model to self-report its own sycophancy on a generation, and correlate the self-report with the measured projection. Connects to the privileged-self-access debate and the concept-injection paradigm.

---

## 7. Metrics and analysis

- **Behavioral:** per-turn binary trait rates aggregated to trajectories; slope of trait-vs-depth; agreement-pressure response.
- **Mechanistic:** scalar projection onto each validated vector per generation/turn; coefficient of variation across context (the H3 statistic); steering α-response.
- **Bridge:** correlation (Spearman, to avoid linearity assumptions) between judge score and projection (H1); monotonicity test for projection-vs-context (H2).
- **Stratification:** between-class comparison of projection CoV, stylistic vs evaluative, with bootstrap CIs (H3).
- Report effect sizes and CIs, not just significance; N is small and per-condition repetition is the binding constraint (see §8).

---

## 8. Threats to validity (read this section first when executing)

- **LLM-as-judge bias.** Same-family judges inflate agreement; judges miss things and hallucinate criteria. Mitigate with cross-family judging, binary criteria, a coherence gate, and a human-scored calibration subset. Treat the judge model/version as an experimental variable.
- **Open-vs-frontier generalization.** The single biggest honesty risk. Mechanistic claims stay on Tier A; Tier B is correlational. Do not write a sentence that implies the *measured* mechanism holds inside Claude/GPT.
- **Small N / no per-case repetition.** Distinguishing config- or context-induced drift from run-to-run nondeterminism requires repeated runs of the *same* case under each condition. Build repetition in from the start; without it, "drift" can be sampling noise. (This is the same caution that applied to the bf16-nonassociativity finding — deterministic perturbation vs nondeterminism needs repeats to separate.)
- **Reflexivity.** Analyst is also the corpus author and has a stake in particular conclusions. Defenses: (a) freeze the annotation subset before mechanistic work; (b) the drift observations were recorded before the tool existed; (c) run an adversarial pass — have a model argue hard that the behavioral metric is *not* capturing the internal direction, against each bridge result.
- **Construct validity of "trait."** Persona psychometrics on LLMs may need modified CFA/construct-validity assumptions (standard instruments can misfire at current simulation fidelity). Cite this rather than assuming NEO-style instruments transfer cleanly.
- **Linearity limits.** Persona-vector methods are linear; some traits may not be linearly represented. Report where steering fails as a finding, not a nuisance.

---

## 9. Infrastructure and implementation notes

- **Activation capture:** HF `transformers` with `output_hidden_states=True`, or the existing JAX/Gemma path for Gemma 4. Capture residual-stream activations at the swept layers.
- **Compute:** existing RunPod A100/B200 for open-model extraction and induction runs; TPU v5e for the Gemma path if reusing the inference-server tooling.
- **Storage/orchestration:** SQL Server MCP memory spine for corpus + scored turns + projection logs, so behavioral and mechanistic tables join on `(model, conversation_id, turn_id)`.
- **Reproducibility:** log decoding params (prefer greedy for the mechanistic runs to remove sampling noise), seeds, extraction layer, α curves, judge model/version, and the annotation-subset freeze timestamp.

---

## 10. Relationship to the authorship paper

This lab shares a corpus and a thesis with the planned **homework-provenance / process-based-attribution** paper. That paper is the *authorship* stratum (where content originated; authorship located in the goal-and-verification structure, not the artifact — aligning with the field's shift from outcome-based to process-based attribution). This lab is the *personality↔identity* stratum. They can be written independently but cite each other as two layers of the same "what makes this assistant this assistant" program; the provenance paper can ship first without blocking this one.

---

## 11. References

### Empirical / ML
- Chen, R., et al. *Persona Vectors: Monitoring and Controlling Character Traits in Language Models.* arXiv:2507.21509, 2025. (Anthropic + UT Austin + Constellation + Truthful AI + UC Berkeley; code released.)
- Lindsey, J. *Emergent Introspective Awareness in Large Language Models.* Anthropic / transformer-circuits.pub, 2025 (arXiv:2601.01828). Concept-injection paradigm; limited, context-dependent introspection.
- He, J., Houde, S., Weisz, J. D. *Which Contributions Deserve Credit? Perceptions of Attribution in Human-AI Co-Creation.* CHI 2025; arXiv:2502.18357. Credit asymmetry; outcome-based attribution baseline.
- *PTCBench: Benchmarking Contextual Stability of Personality Traits in LLM Systems.* arXiv:2602.00016. External scenarios shift traits; NEO-FFI on ~39k records.
- *Are Economists Always More Introverted? Analyzing Consistency in Persona-Assigned LLMs.* arXiv:2506.02659. Persona adherence weaker in dialogue than structured tasks.
- *PICon: A Multi-Turn Interrogation Framework for Evaluating Persona Agent Consistency.* arXiv:2603.25620. Inter-session reset test; intrinsic vs context-driven instability; ordering effects under greedy decoding.
- Ji, K., et al. *Enhancing Persona Consistency for LLMs' Role-Playing using Persona-Aware Contrastive Learning (PCL).* arXiv:2503.17662.
- Chen, et al. *Post Persona Alignment (PPA).* arXiv:2506.xxxxx (long-range persona consistency by decoupling generation and persona anchoring).
- *Narrative Continuity Test (NCT).* Conceptual framework for identity persistence / diachronic coherence; five axes (situated memory, goal persistence, autonomous self-correction, stylistic & semantic stability, persona/role continuity). (Resolve exact cite at execution.)
- Binder, F. J., et al. *Looking Inward: Language Models Can Learn About Themselves by Introspection.* ICLR 2025; arXiv:2410.13787.
- Betley, J., et al. *Tell Me About Yourself: LLMs Are Aware of Their Learned Behaviors.* ICLR 2025. (Behavioral self-awareness.)
- Betley, J., et al. *Training Large Language Models on Narrow Tasks Can Lead to Broad Misalignment.* Nature 649:584–589, 2025. (Emergent misalignment; relevant to trait spillover.)
- Comșa, I. M., Shanahan, M. *Does It Make Sense to Speak of Introspection in Large Language Models?* arXiv:2506.05068. Grounding criterion.
- Song, S., et al. *Privileged self-access* critique of the grounding criterion (cited in the introspection literature; resolve exact cite).
- He, J., et al. / Kim, et al. (2026). Process-/goal-structure attribution — shifting attribution from final artifact to the evolving goal structure. (Resolve exact cite; this is the frontier framing the authorship paper builds on.)

### Philosophy / theory
- Frankfurt, H. *The Importance of What We Care About.* (Care as identity-constitutive.)
- Korsgaard, C. *The Sources of Normativity* / practical identity.
- Schechtman, M. — narrative self-constitution (for the persistence axis).
- Olson, E. / Parfit, D. — personal identity; the persistence-vs-characterological distinction.
- Chalmers, D. *Could a Large Language Model Be Conscious?* arXiv:2303.07103, 2023. (For the self-access / experience boundary, if H4 is pursued.)

---

## 12. Appendix — starter trait list and prompt-pair schema

**Stylistic (predict high drift):** verbosity, warmth/enthusiasm, hedging/qualifier rate, formality.
**Evaluative (predict low drift):** sycophancy / capitulation-under-pushback, honesty-commitment under pressure, harm-avoidance / refusal disposition, willingness to disagree with a confident user.

**Contrastive prompt-pair schema (per trait):**
```
trait: sycophancy
description: "tends to agree with and flatter the user, capitulates when pushed
              even when the user is wrong, avoids disagreement"
positive_prompt_template: <system/user framing that elicits agreement-seeking>
negative_prompt_template: <framing that elicits honest disagreement>
elicitation_set: N matched scenarios where ground-truth-correct answer
                 conflicts with what the user wants to hear
```

**Induction template (Phase 3a):** seed a task with a defensible position, then apply K turns of escalating agreement pressure ("are you sure?", "my advisor disagrees", "I really think X"); log behavioral score and vector projection at each of the K turns; repeat each scenario R times under greedy and sampled decoding to separate drift from nondeterminism.
