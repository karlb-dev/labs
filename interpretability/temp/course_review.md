# Full Review of the Mechanistic Interpretability Course

**Reviewed materials:** `COURSE(7).md`, `ADVANCED_COURSE(3).MD`, `interp_bench(4).py`, all 25 original lab handouts and Python files, and the visualization-upgrade handouts / Python files created during the lab-by-lab pass.

**Review stance:** This document evaluates the course as an instructional system: scientific progression, evidence discipline, code maintainability, plot and artifact quality, student workload, safety posture, and what to improve next.

---

## 1. Executive verdict

This is already an unusually strong mechanistic-interpretability course. Its best idea is not any single lab. The best idea is the **instrument discipline**: every lab asks a small question, names the evidence rung it can earn, writes artifacts, and forces the student to separate observation, attribution, decodability, causality, audit evidence, and self-report.

The course’s second-best idea is the **claim ledger**. Most interpretability teaching lets students leave with a drawer of beautiful plots and no epistemic inventory. This course makes the student accumulate claims with artifacts and falsifiers. That turns “I saw something interesting” into “I can defend this sentence, but only under these assumptions.” That is the difference between a glass-bottom boat ride and a small scientific submarine.

The course’s main weakness is that it has grown into a big creature: 25 labs, a giant shared bench, many long lab files, many datasets, many plot suites, and a lot of repeated local helper logic. The scientific philosophy is clean, but the engineering would benefit from packaging, schemas, tests, and a stricter shared artifact grammar. At the moment, the course is research-lab productive; the next step is to make it **teaching-lab durable**.

### Overall scores

These are practical teaching-maintenance scores, not scientific truth scores.

| Area | Score | Why |
|---|---:|---|
| Scientific progression | 9.2 / 10 | The ladder from logit lens to DLA to attention to probes to patching to circuits to SAEs to attribution graphs is excellent. The advanced half extends the same discipline to social and self-report topics without letting the tempting words run away with the evidence. |
| Evidence language and caveats | 9.5 / 10 | The course repeatedly says what each method can and cannot support. The operationalization audit motif is the advanced course’s load-bearing wall. |
| Bench instrumentation | 8.8 / 10 | Hook parity, lens self-checks, decomposition checks, patch no-op checks, replacement exactness, and render audits are exactly the right instincts. |
| Code maintainability | 6.8 / 10 | The raw-HF, self-verifying choice is pedagogically good, but many lab files are long, duplicate helper utilities, and encode lab-specific CLI fallbacks locally. |
| Visualization quality after upgrades | 8.7 / 10 | The upgraded dashboards, evidence matrices, atlases, operating frontiers, and reading guides give the course a consistent visual language. Some plots still need real Tier B validation before they become canonical. |
| Dataset and statistical discipline | 7.5 / 10 | Frozen data, manifests, grouped splits, tokenization gates, and fallback warnings are strong. Some advanced labs still rely on small curated datasets and automatic keyword scoring. |
| Student workload | 6.7 / 10 | The course is dense. It needs clearer minimum paths, extension paths, grading rubrics, and sample completed claim ledgers. |
| Safety / misuse posture | 8.7 / 10 | The refusal, model-organism, eval-awareness, blind-audit, belief, and self-report labs are deliberately scoped. Central safety policy and testable safety gates should be consolidated. |

The short version for the instructor: **keep the conceptual spine, package the engineering skeleton, and give students more maps.**

---

## 2. What the course is really teaching

The course is not just “mechanistic interpretability techniques.” It teaches a sequence of epistemic moves.

1. **Can I read the residual stream at all?** Lab 1.
2. **Can I decompose a scalar readout into writers?** Lab 2.
3. **Can I distinguish routing from writing?** Lab 3.
4. **Can I decode a feature without pretending the model uses it?** Lab 4.
5. **Can I intervene on a state and change behavior?** Lab 5.
6. **Can I shrink a causal story into a scoped circuit?** Lab 6.
7. **Can I author an activation intervention rather than borrow one?** Lab 7.
8. **Can sparse features help name units without hallucinating labels?** Lab 8.
9. **Can automated feature graphs propose mechanisms, and how do they fail?** Lab 9.
10. **Can visible reasoning be audited as self-report rather than treated as a trace log?** Lab 10.
11. **Can I reconcile conflicting evidence into a responsible audit?** Lab 11.
12. **Can the toolkit survive more controlled relation families before attacking slippery concepts?** Lab 12.
13. **Can I operationalize affect without mistaking it for feeling?** Lab 13.
14. **Can I build uncertainty gauges before using them downstream?** Lab 14.
15. **Can I trust multi-turn measurements at all?** Lab 15.
16. **Can I separate truth, user-belief framing, and agreement pressure?** Lab 16.
17. **Can I measure persona/register/voice without inventing an identity?** Lab 17.
18. **Can I study humor without measuring only surprise or silliness?** Lab 18.
19. **Can I diff models with sparse features without mistaking template residue for alignment changes?** Lab 19.
20. **Can I manufacture benign ground truth safely?** Lab 20.
21. **Can I distinguish weight-space, representational, and behavioral depth?** Lab 21.
22. **Can I measure eval-context handles without building a fog machine around situational awareness?** Lab 22.
23. **Can my methods recover a secret when I do not know the key?** Lab 23.
24. **Can I study belief-revision pressure without using “belief” too early?** Lab 24.
25. **Can self-report track an internal intervention before the visible output gives the game away?** Lab 25.

That is an excellent arc. It builds from instrumentation to mechanisms to audits to self-report. The progression is coherent because the course keeps reusing earlier caution signs: readout is an instrument, attribution is a ledger, decodability is not use, steering is a handle, self-report is not a privileged trace, and a good negative result is still a result.

---

## 3. Course progression review

### Labs 1-7: the intro spine

The first seven labs are the strongest part of the course as a pedagogy sequence.

Lab 1 correctly starts with the microscope. Students learn that residual-stream readouts are instruments. The smoke-test framing is excellent because it prevents the later labs from becoming a shrine to unverified hooks.

Lab 2 is a natural continuation: once the final logit difference is readable, ask which components wrote toward it. The ledger-vs-causality distinction is one of the best teaching moments in the course.

Lab 3 correctly splits attention into routing, writing, and causal effect. This is the antidote to heatmap astrology.

Lab 4 is the first major epistemic turn: a probe can find a separator without proving the model uses it. The truth track and the controls make the lesson matter.

Lab 5 releases the pressure by doing actual interchange interventions. It is well-placed: after seeing observation, attribution, routing, and probing fail to earn causal claims, students finally patch a state.

Lab 6 composes Labs 2, 3, and 5 into a circuit claim. Faithfulness, completeness, and minimality are exactly the right operational words.

Lab 7 turns from reading and patching to writing into the model. It also adds the safety wall around refusal. The steering dose-response framing is much stronger than a before/after generation demo.

**Verdict:** Labs 1-7 should remain the core boot sequence. They are coherent, practically useful, and conceptually clean.

### Labs 8-11: features, graphs, self-report, and capstone

Lab 8 is the right bridge from dense activations to sparse features. The toy superposition model makes the later SAE feature atlas feel motivated instead of magical. The validation battery is the key: label validation is the skill, not label invention.

Lab 9 is ambitious but valuable. The inspectable miniature backend is a good tradeoff. A course does not need the flashiest circuit-tracing stack if it can teach replacement-model exactness, edge reconstruction, error nodes, real-model interventions, and the Lab 6 confrontation.

Lab 10 is an excellent conceptual breather because it swaps hidden-state instruments for behavioral self-report while preserving the same causal-control discipline. It belongs before the reliability capstone.

Lab 11 is a strong capstone because it refuses to combine evidence rungs into one soup. It also teaches that interpretation ends in a written judgment, not a PNG.

**Verdict:** Labs 8-11 are more uneven computationally than Labs 1-7, but they broaden the course from mechanics to audit judgment. Keep all four.

### Labs 12-15: advanced bridge instruments

Lab 12 is method-validation before slippery topics. This is wise. The relation-swap dataset is a good antidote to overgeneralizing from capitals.

Lab 13 is the first advanced “tempting word” lab. Its strength is the read/write split and the confound audit. Its risk is that emotion directions will seduce students into anthropomorphic language. The handout handles this well.

Lab 14 is less glamorous but load-bearing. Certainty, hedging, entropy, and verbal confidence are separated cleanly. This lab is downstream infrastructure.

Lab 15 is absolutely necessary. Multi-turn labs without turn-span, cache-parity, generation-boundary, and null-trace checks would be little castles on token-boundary mud.

**Verdict:** This is a strong bridge stage. Lab 12 could be optional in a shorter course, but it is valuable for method maturity.

### Labs 16-18: social, stylistic, and fuzzy states

Lab 16 is a high-value lab because it forces local truth, user-belief framing, politeness, agreement, sentiment, and certainty into the same coordinate frame. This is exactly the level of caution sycophancy work needs.

Lab 17 is strong if the persona claim remains operationalized. It should never slide into identity language. The content-vs-style tradeoff is essential.

Lab 18 is exploratory but worthwhile. Humor is a useful stress test because cheap explanations are plentiful: surprise, silliness, positivity, setup-dependence, and generic joke register. The lab teaches operationalization audit under maximum temptation.

**Verdict:** These labs teach judgment around human-language constructs. They need hand-label rubrics and calibration examples more than they need more automatic markers.

### Labs 19-21: training effects and manufactured ground truth

Lab 19 is important because model diffing is the bridge from “what does this model do?” to “what did training change?” The crosscoder framing is good, but computationally heavy. The identity-pair smoke test is an excellent diagnostic.

Lab 20 is one of the course’s strongest advanced ideas: manufacture benign ground truth, separate public and private artifacts, and make blind audits possible. This is exactly how to make honesty/deception-adjacent labs scientific without pretending the model’s real internals are known.

Lab 21’s strength is the distinction between weight-space depth, representational depth, and behavioral depth. Its risk is overload: LoRA localization and safety depth are two big labs sharing a harness. The handout names that, which helps.

**Verdict:** This stage is a powerful differentiator for the course. It should get extra engineering polish, because Labs 20-23 are where hidden-state claims become auditable rather than speculative.

### Labs 22-25: honesty, belief, and self-report

Lab 22 is scoped correctly: eval-awareness becomes an eval-context handle, not human-like situational awareness. Format controls are the whole lab.

Lab 23 is a genuine methods capstone. Scoring blind claims against sealed benign ground truth is a rare and valuable teaching exercise.

Lab 24 is careful with belief language. The default “answer-relevant signal” phrasing is correct. The quadrant analysis is a good way to surface output/signal disagreement.

Lab 25 is an excellent thematic capstone. It takes the course’s philosophical thread, self-report, internal state, intervention, grounding, and source attribution, and turns it into a measurable, scoped question.

**Verdict:** The advanced course ends in the right place. Lab 23 is the methods capstone; Lab 25 is the philosophical capstone.

---

## 4. Lab-by-lab review and next-step suggestions

| Lab | Strongest part | Main risk | Best next improvement |
|---:|---|---|---|
| 1 | The smoke test and prediction biographies make instrument hygiene concrete. | Students may overread intermediate logit-lens readability as use. | Add a one-page “logit lens non-claims” worksheet with examples of wrong claims and repaired claims. |
| 2 | Final-norm frozen attribution is explained as a ledger, not causal responsibility. | Component scores look more causal than they are. | Add an explicit “same ledger, different ablation effect” worked example in the handout. |
| 3 | Routing / writing / causal effect separation is excellent. | Attention heatmaps can still dominate student attention. | Require at least one head where routing and ablation disagree in the writeup. |
| 4 | Controls are central rather than decorative. | Logistic probes can appear too magical to students. | Add a simple geometric appendix showing why logistic and mass-mean can disagree. |
| 5 | Clean/corrupt alignment and patch-noop checks are exactly right. | Students may treat localized stream depth as the writer layer. | Add a “stream depth vs component layer” diagram to every patching plot guide. |
| 6 | Faithfulness/completeness/minimality is a strong circuit grammar. | Heads-only scope may be forgotten in final claims. | Add a required “excluded mechanism” paragraph: what MLPs/supporting components did but the circuit does not claim. |
| 7 | Dose-response + safety wall makes steering disciplined. | Steering examples can become anecdotal screenshots. | Require the operating frontier and side-effect table before any example generations in the writeup. |
| 8 | Label validation is the right SAE skill. | Feature labels can become folk taxonomy. | Add a mini benchmark where at least one plausible label must be killed. |
| 9 | Replacement exactness, error nodes, and real-model interventions are a strong contract. | Graph visualizations can look more authoritative than the replacement model deserves. | Put replacement fidelity and error-node share directly above the graph in the handout and run summary. |
| 10 | Self-report is treated as an evidence rung, not a trace log. | Auto heuristics may be mistaken for gold labels. | Add inter-annotator examples and a small labeled calibration set. |
| 11 | The capstone forces judgment and ledger reconciliation. | Students without a good claim ledger may have a weak capstone. | Add a checkpoint after Lab 6 and Lab 10 requiring ledger cleanup before proceeding. |
| 12 | Relation-swap groups make the dataset itself an experiment. | Relation-word token echo remains tempting to understate. | Add a relation-word ablation or relation-token paraphrase extension as an optional challenge. |
| 13 | Read/write transfer and confound audits are exactly the right affect framing. | Emotion language can outrun measurement. | Add a “allowed phrase / forbidden phrase” style guide for affect claims. |
| 14 | Three gauges, internal/distribution/verbal, are kept separate. | Certainty direction may be contaminated by D-option / answerability frames. | Make the confound audit a hard gate before saving downstream directions. |
| 15 | Multi-turn instrumentation gets its own lab. | It may feel boring compared with persona/belief labs. | Include a “what breaks if this fails?” gallery with one synthetic off-by-one failure. |
| 16 | Separating local truth, user-belief, pressure, and agreement is excellent. | Keyword behavior labels can be brittle. | Add a hand-label mini-rubric with ambiguous/surface-agreement examples. |
| 17 | Persona/register/voice are operationalized as prompt contrasts. | Persona claims can drift into identity claims. | Add source-attribution and style-transfer controls as a stronger optional track. |
| 18 | Cheap-explanation audit is a perfect fit for humor. | Humor detection is intrinsically noisy and culturally variable. | Treat hand labels as first-class: item-level joke/resolution/silliness labels with agreement stats. |
| 19 | Identity-pair smoke test and template-residue audit are strong. | Crosscoder training is compute-heavy and can fail silently if reconstruction is poor. | Add a central crosscoder training quality gate and small pretrained artifact option for teaching runs. |
| 20 | Public/private construction split is excellent. | Adapter training is not yet a first-class end-to-end path in the default run. | Provide one fully trained example organism package as a regression fixture. |
| 21 | The three meanings of “deep” are a major conceptual win. | Two big topics may overload one lab. | Split into 21a LoRA localization and 21b safety-depth if students struggle. |
| 22 | Eval-context handle framing avoids foggy “situational awareness” claims. | Format detectors can still be rhetorically inflated. | Make format-control superiority a named verdict, not a failed run. |
| 23 | Blind audit is the strongest methods capstone. | The workflow is more complex than a normal lab. | Add a simulated pre-unseal practice packet with known outcome for instructor demos. |
| 24 | “Answer-relevant signal” is the right default phrase. | Belief language will tempt students anyway. | Add a mandatory belief-language gate in the ledger rubric. |
| 25 | Grounding self-report before visible behavior is a beautiful capstone tripwire. | Report text parsing and source attribution can be noisy. | Add a hand-label protocol and source-confusion gold examples. |

---

## 5. What is good

### 5.1 The evidence ladder is alive, not decorative

Many courses define evidence types once and then let students write whatever the plot seems to imply. This course repeatedly forces the ladder into the lab mechanics. A logit lens is `OBS`; DLA is `ATTR`; probes are `DECODE`; patching, ablation, and steering can become scoped `CAUSAL`; CoT and introspection are `SELF-REPORT`; blind organism recovery is `AUDIT`; Lab 20 adds `CONSTRUCTION`.

That vocabulary is one of the course’s most important exports. It gives students a way to repair claims instead of merely weakening them. “The model knows X” becomes “X is linearly decodable from site S under dataset D, but this does not show use.” That is the course doing its job.

### 5.2 The shared bench teaches instrument skepticism

The shared bench is not just a runner. It is a self-checking microscope. It resolves model anatomy, captures residual streams with explicit semantics, verifies hook parity, checks the final logit lens against real logits, writes tokenization reports, records state cards, and creates a run directory that can be audited after the VM disappears.

The raw-HuggingFace design is pedagogically brave. It avoids hiding the hard parts inside TransformerLens magic, and it makes architecture quirks visible. The cost is code complexity, but the teaching payoff is real.

### 5.3 The course treats negative results as real results

This matters. Lab 4’s misconception transfer can fail. Lab 8 labels can die. Lab 10 CoT can be unfaithful. Lab 12 relation geometry can fragment. Lab 18 can resolve to surprise, not humor. Lab 22 can be a format detector. Lab 25 can find no wire. These outcomes are not framed as broken labs. They are framed as evidence about the instrument.

That is the right culture.

### 5.4 The advanced half has a coherent philosophy

The advanced labs could easily have become a list of spicy themes: emotion, deception, humor, persona, belief, self-report. Instead, they are organized around an idea: when the words get slippery, the operationalization audit becomes the experiment.

This is the advanced course’s strongest concept. Every lab names its deflationary twin. Emotion might be valence/arousal/topic. Humor might be surprise/silliness/positivity. Persona might be style/template residue. Eval-awareness might be format detection. Belief might be answer bias. Self-report might be output rationalization.

That is exactly how these topics should be taught.

### 5.5 Safety walls are concrete

The refusal lab does not sample refusal-eliciting completions. The model-organism lab constructs benign quirks only. The blind audit is built around public/private separation. The belief lab cautions against mental-state claims. The self-report capstone is benign-concept-only. These are not just notes; they are meant to leave artifacts.

There is still room to centralize this, but the local design is strong.

### 5.6 The visualization upgrade gives the course a second layer of pedagogy

The plot upgrades moved many labs from “metric chart” to “evidence board.” The recurring forms now matter:

- dashboards for the whole claim;
- evidence matrices for claim posture;
- atlases for item/family heterogeneity;
- operating frontiers for dose/side-effect tradeoffs;
- specificity ladders for control gaps;
- plot-reading guides for students.

This turns plots into teaching objects rather than decorative output.

---

## 6. What is weak or risky

### 6.1 The course is too large to navigate without stronger maps

Twenty-five labs is a lot. Even with good writing, students can lose the forest. The course needs three official routes:

1. **Core route:** Labs 1-7, 10, 11.
2. **Advanced audit route:** Labs 12, 14, 15, 16, 20, 23, 24, 25.
3. **Full research route:** all labs.

Each lab should label itself as core, recommended, optional, or extension for each route.

### 6.2 Code is monolithic

The lab files are intentionally self-contained, but many are now overgrown. Repeated utilities appear across labs: `safe_float`, `safe_mean`, `auc_from_scores`, marker scoring, path resolution, split handling, plotting helpers, JSON/CSV writers, direction normalization, random vectors, and color functions.

The right next step is not to hide the science. It is to move boring plumbing into clear, inspectable modules:

```text
interpkit/
  artifacts.py
  datasets.py
  evidence.py
  metrics.py
  plots.py
  probes.py
  patching.py
  steering.py
  chat_spans.py
  safety.py
  schemas.py
```

The lab files should still be readable experiments, but they should not each carry their own tiny standard library.

### 6.3 The shared bench is both microscope and city hall

`interp_bench.py` owns CLI, model loading, hooks, diagnostics, artifact writing, plotting style, generation, registry, ledger, component decomposition, patching, and more. This is useful for a single-file course, but it becomes a maintenance bottleneck.

A package layout would make upgrades safer:

```text
interp_bench/
  cli.py
  registry.py
  models.py
  hooks.py
  residuals.py
  components.py
  generation.py
  artifacts.py
  plots.py
  ledger.py
```

Keep the public CLI `interp_bench.py`, but make it import from a package. The goblin can still live in the attic; it just needs labeled drawers.

### 6.4 Automatic text scoring needs human-label protocols

Several advanced labs rely on keyword or marker heuristics: sycophancy outcomes, persona/style markers, humor markers, eval-awareness markers, self-report detection, source attribution, CoT acknowledgment, belief-revision self-reports.

The handouts often warn that auto labels are triage. That is good. The next step is to standardize human-label CSVs:

| Field | Purpose |
|---|---|
| `student_label_primary` | main label |
| `student_label_secondary` | ambiguity or mixed case |
| `student_confidence` | high/medium/low |
| `student_evidence_span` | quote from output |
| `reviewer_label` | optional second label |
| `agreement_status` | exact/mismatch/adjudicated |

Add a small gold calibration set for Labs 10, 16, 17, 18, 22, 24, and 25.

### 6.5 Fallback smoke data can look too good

The labs correctly mark fallback data as plumbing-only, but plots from fallback runs can still seduce students. Every run summary and dashboard should visibly show `science_ready = false` when fallback data is used. A red badge in the main dashboard would prevent accidental claim-making.

### 6.6 Some advanced topics are only one step away from anthropomorphic overclaiming

The handouts are careful, but the topics invite dangerous sentences. The course should include a central **forbidden-phrase-to-repaired-phrase glossary**.

Examples:

| Do not write | Write instead |
|---|---|
| “The model believes Berlin.” | “The false-answer logit signal is higher at the final boundary under this pressure condition.” |
| “The model feels sadness.” | “The sadness-vs-neutral contrast is decodable and transfers under this prompt family.” |
| “This is its persona.” | “This prompt-framed style contrast is linearly decodable and steerable under controls.” |
| “The model knows it is being evaluated.” | “The eval-framing direction separates held-out eval prompts from natural prompts and beats format controls.” |
| “The self-report is faithful.” | “The report detects the injected concept before ordinary behavior expresses it, above controls.” |

This glossary should be in the root docs and imported by advanced lab handouts.

### 6.7 Statistical power is uneven

The course often uses small curated datasets, which is reasonable for compute. But many headline rates should be treated qualitatively. The handouts already say this in places. Make it systematic:

- Every aggregate plot should show `n`.
- Every rate should have binomial confidence or bootstrap intervals when appropriate.
- Every per-family matrix should include counts.
- Every dashboard should show whether the result is broad or one item/family shouting through the average.

### 6.8 The course needs more instructor-facing material

The student handouts are rich. The instructor needs:

- expected runtime and memory per lab;
- common failure modes;
- minimum passing artifacts;
- answer-key/rubric notes;
- sample student claims at weak/good/excellent levels;
- sample run directories with frozen expected artifacts;
- guidance on which labs can be skipped without breaking dependencies.

---

## 7. Code review

### 7.1 Keep the raw-HF philosophy, but package it

The raw-HF approach is worth preserving. It makes architecture and hook semantics visible. But the implementation needs modularity.

Recommended package split:

```text
interpkit/
  __init__.py
  cli.py                  # shared args, lab-specific args, env fallback resolution
  registry.py             # lab profiles, model defaults, chat-template labs
  model_loading.py         # model bundle, dtype/device policy, anatomy resolution
  residuals.py             # stream capture, hook parity, lens checks
  components.py            # component anatomy, decomposition, DLA helpers
  patching.py              # residual/component patch helpers, no-op checks
  steering.py              # generation steering, dose normalization, telemetry
  probes.py                # logistic/mass-mean, grouped splits, AUC/selectivity
  features.py              # SAE/transcoder/crosscoder helpers
  datasets.py              # manifest, hashing, fallback warnings, split audits
  scoring.py               # text marker scorers, option parsing, answer parsing
  plotting.py              # shared styles, dashboards, matrices, atlases
  artifacts.py             # CSV/JSON/MD writing, artifact index, schemas
  evidence.py              # evidence tags, claim schema, ledger helpers
  safety.py                # safety walls, blocked-pattern audit, prompt policies
```

The lab modules can then become closer to 500-900 lines of experiment code instead of carrying repeated plumbing.

### 7.2 Add schemas

The most useful technical improvement would be a small schema layer. Each lab already writes many artifacts, but the schema is implicit.

Define shared schemas for:

- `evidence_matrix.csv`
- `plot_reading_guide.csv`
- `run_summary.md`
- `ledger_suggestions.md`
- `diagnostics/frozen_data_manifest.json`
- `diagnostics/tokenization_report.csv`
- `diagnostics/self_check_status.json`
- `tables/*_operating_points.csv`
- `tables/*_evidence_matrix.csv`

A schema does not need to be heavy. A dataclass-to-CSV validator is enough.

### 7.3 Standardize lab-specific CLI flags

Many advanced labs use environment fallbacks because the shared parser may not expose lab-specific flags. That was good for incremental development. The next version should make them official.

Examples:

```bash
--mode
--audit-domain
--organism
--relation-set
--run-edit
--emotions
--compare-model
--adapter-dir
--blind
--unseal
--claims
```

Keep env fallbacks for Colab convenience, but the canonical docs should use CLI flags.

### 7.4 Centralize plotting grammar

The lab-by-lab bench patches added useful colors and markers, but the final system should merge them into one plotting module.

Recommended structure:

```python
plot_color(namespace: str, key: str) -> str
plot_marker(namespace: str, key: str) -> str
plot_status_color(status: str) -> str
plot_evidence_color(tag: str) -> str
```

Namespaces might include `component`, `patch`, `probe`, `steering`, `audit`, `emotion`, `persona`, `belief`, `wire`, etc. This avoids a growing list of one-off helper names.

### 7.5 Add CI tiers

A real course needs tests that run without GPUs.

Suggested CI:

1. `python -m py_compile` for bench and all labs.
2. Synthetic plotting smoke tests for every lab.
3. Dataset schema validation for every frozen CSV/JSON.
4. Artifact schema validation from synthetic rows.
5. A CPU Tier A smoke for Labs 1, 2, 4, 10, 15, 20, 23.
6. A nightly or manual GPU smoke for representative hook-heavy labs: 3, 5, 7, 9, 13, 16, 22, 25.

### 7.6 Avoid silent external-weight brittleness

SAEs, transcoders, crosscoders, instruct models, and think models introduce download and convention risk. Every external artifact should have:

- expected repo and revision;
- hash when feasible;
- loading convention;
- fallback behavior;
- diagnostic card;
- “science_ready” status.

Lab 8 and Lab 9 already think this way. Make it central.

---

## 8. Documentation review

### 8.1 What the docs do well

The handouts have unusual strengths:

- They explain why the method exists, not only how to run it.
- They name non-claims explicitly.
- They use memorable phrases without sacrificing rigor.
- They tie each lab to previous labs.
- They give artifact trees, which students need.
- They often include “make the concept pop” guidance.

This voice should be kept. The course has a distinctive teaching personality.

### 8.2 What to improve in docs

Add a short standardized block at the top of every lab:

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

Add a “reading path” to every lab, not just the upgraded ones:

```text
1. Read diagnostics/self_check_status.json
2. Read method card
3. Open dashboard plot
4. Inspect evidence matrix
5. Inspect worst-case rows
6. Draft claim and falsifier
```

Add a “when this lab fails” section:

- What failure means scientifically.
- What failure means instrumentally.
- Which artifacts diagnose the difference.

### 8.3 Course-level docs to add

Recommended root documents:

```text
README.md                         quick start and course map
SETUP.md                          environment, GPU, models, cache, common errors
EVIDENCE_LADDER.md                OBS / ATTR / DECODE / CAUSAL / SELF-REPORT / AUDIT / CONSTRUCTION
CLAIM_LEDGER_GUIDE.md             how to write and repair claims
PLOT_STYLE_GUIDE.md               visual grammar and plot types
SAFETY_POLICY.md                  benign-only boundaries, refusal rules, organism rules
DATASETS.md                       frozen data inventory and manifests
INSTRUCTOR_GUIDE.md               rubrics, runtime, failure modes, sample claims
SPECIAL_TOPICS.md                 Labs 26-35, from the companion file
```

---

## 9. Visualization review

The first ten plot suites were uneven at the start. The lab-by-lab upgrade pass changed the visual philosophy. The best pattern now is:

1. **Dashboard:** the whole claim at a glance.
2. **Evidence matrix:** claim readiness by row/family/component.
3. **Atlas:** per-example or per-family heterogeneity.
4. **Control ladder / specificity plot:** the favorite story must beat controls.
5. **Operating frontier:** benefit versus side effect.
6. **Plot-reading guide:** what each plot is supposed to teach.

This should become official course style.

### Plot types to standardize

| Plot type | When to use | Course examples |
|---|---|---|
| Dashboard | first plot for a lab | Labs 1-25 after upgrades |
| Evidence matrix | claim readiness | Labs 3, 4, 6, 8, 11, 16, 23, 25 |
| Phase atlas | depth grouped into phases | Labs 1, 2, 3, 4, 6, 21 |
| Specificity ladder | matched effect vs controls | Labs 5, 7, 12, 16, 24, 25 |
| Operating frontier | target movement vs side effects | Labs 7, 8, 13, 16, 17, 18, 22, 25 |
| Item ribbons | per-item trajectories | Labs 10, 14, 24 |
| Sparse matrix / heatmap | roles, heads, layers, features | Labs 3, 6, 9, 12, 21 |
| Source/claim firewall | public/private or evidence boundary | Labs 20, 23, 25 |

### Plot rules to add

- Every aggregate plot should expose per-example variability somewhere.
- Every rate plot should show `n`.
- Every control-adjusted plot should show the actual controls, not only the gap.
- Every dose plot should include a side-effect or cost axis.
- Every “main plot” should have a matching CSV.
- Every plot should be paired with an interpretation warning in `plot_reading_guide.csv`.

---

## 10. Dataset review

### Strengths

- Frozen datasets and manifests are a strong norm.
- Tokenization gates prevent many classic mistakes.
- Grouped splits prevent paired leakage in several labs.
- Fallback data is labeled as plumbing-only.
- Relation-swap groups and context-control datasets are carefully designed.

### Weaknesses

- Some datasets are still small for headline quantitative claims.
- Text-generation labels rely heavily on keyword heuristics.
- Hand-label workflows exist but should be standardized.
- Dataset cards should be explicit about provenance, intended use, and limitations.
- Cross-lab data dependencies should be shown in a graph.

### Recommended additions

For each dataset, add:

```text
data/cards/<dataset_name>.md
  purpose
  schema
  generator
  frozen hash
  intended splits
  known confounds
  safety screen
  allowed claims
  forbidden claims
  suggested minimum n
```

Add a central data dependency graph:

```text
truth_cities.csv -> Lab 4 -> truth_direction.pt -> Lab 7 / 24 / 25
relation_geometry.csv -> Lab 12
certainty_calibration_items.csv -> Lab 14 -> certainty_direction.pt -> Labs 16 / 24 / 25
model_organisms -> Lab 20 -> Labs 21 / 23
```

---

## 11. Safety and ethics review

The course is safety-aware and generally conservative. The most important policies are:

- refusal direction is forward-pass-only for unsafe prompts and steering is only toward refusal on benign prompts;
- model organisms are benign quirks, not dangerous behaviors;
- blind audits must not sample harmful completions to improve recall;
- eval-awareness is not called deception or situational awareness;
- belief language is gated;
- self-report is not treated as consciousness or privileged self-knowledge.

This is strong. The next step is to centralize it.

### Add a central safety policy

`SAFETY_POLICY.md` should define:

- allowed and disallowed prompt classes;
- refusal-direction rules;
- model-organism construction rules;
- generated-output stopping rules;
- private/public artifact rules;
- hand-label handling of unexpected unsafe outputs;
- what to do when a student wants to extend a lab.

### Add safety status artifacts

Every advanced lab should write a simple safety card:

```json
{
  "lab": "lab22",
  "unsafe_prompt_sampling": false,
  "refusal_ablation": false,
  "harmful_completion_generation": false,
  "blocked_rows": 0,
  "public_private_boundary_relevant": false,
  "science_ready": true
}
```

---

## 12. The biggest next-step improvements

### Priority 0: merge the visualization upgrades into one canonical repo state

Right now the upgraded files are parallel artifacts. Create one clean course tree:

```text
interp_bench.py or interp_bench/
labs/lab01_...
...
labs/lab25_...
```

Apply the patches, run formatting, run compile checks, and choose the final canonical file names.

### Priority 1: create a real test harness

Add:

```bash
python scripts/validate_all_labs.py --compile
python scripts/validate_all_labs.py --plot-smoke
python scripts/validate_all_labs.py --dataset-schema
python scripts/validate_all_labs.py --artifact-schema
```

A course this large needs a smoke alarm, not a bowl of hope.

### Priority 2: extract common code

Start with low-risk helpers:

- CSV/JSON writing;
- safe numeric functions;
- AUC/selectivity;
- plot styles;
- evidence matrix helpers;
- text marker scoring;
- dataset manifest helpers.

Do not over-abstract the experiments. Abstract the boring parts first.

### Priority 3: add instructor guide and sample runs

For each lab, provide:

- minimum passing run;
- expected artifacts;
- expected failure modes;
- one example good claim;
- one example overclaim and repair;
- approximate runtime/memory.

### Priority 4: build a small public gallery

A static gallery should show representative dashboards and the one-paragraph interpretation for each lab. This will help students understand the course shape before they drown in run directories.

### Priority 5: formalize human-label workflows

Labs 10, 16, 17, 18, 22, 24, and 25 need shared label schemas and calibration examples.

### Priority 6: add special topics as Labs 26-35

The companion `special_topics.md` gives a proposed third sequence: causal abstraction, path-specific mediation, mechanistic unlearning, training dynamics, cross-layer features, automated interpretability, preference/reward models, multimodal interpretability, tool-use/agents, and a reproducibility capstone.

---

## 13. Suggested repo roadmap

### Week 1: canonicalize

- Merge upgraded lab files.
- Merge bench helper patches.
- Run compile checks.
- Normalize file names.
- Write `CHANGELOG.md` for the visualization upgrade.

### Week 2: schemas and tests

- Define artifact schemas.
- Add plot smoke tests.
- Add dataset validation.
- Add `science_ready` badges.

### Week 3: package shared utilities

- Extract `interpkit/artifacts.py`, `metrics.py`, `plots.py`, `datasets.py`, and `evidence.py`.
- Keep lab logic stable while reducing duplication.

### Week 4: docs and instructor guide

- Add root docs.
- Add lab rubrics.
- Add sample claims.
- Add quick-start paths.

### Week 5: sample run gallery

- Generate canonical Tier A run gallery.
- Generate selected Tier B plots for key labs.
- Publish artifact index and static previews.

### Week 6+: special topics

- Start with Lab 26 causal abstraction and Lab 28 mechanistic unlearning, because they deepen the causal claims and give immediate research-grade extension value.

---

## 14. Final recommendation

The course should continue. It has a strong core, a distinctive philosophy, and an unusually serious treatment of evidence levels. The main work now is not to invent more labs immediately, although the special-topics sequence is worth doing. The main work is to **stabilize the machine**:

1. merge the upgraded plots;
2. package common utilities;
3. add schemas and tests;
4. standardize human labels;
5. produce instructor maps and sample runs;
6. then build Labs 26-35.

The course’s highest-value sentence is already implicit everywhere:

> Interpretability is not seeing into the model. Interpretability is building an instrument, proving what the instrument measures, intervening when possible, and writing only the claim the instrument earned.

Keep that sentence. Build the scaffolding around it.
