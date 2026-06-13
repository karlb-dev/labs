# Data roadmap for the advanced labs (12–22)

**Status:** planning note, 2026-06-11. The intro set (Labs 1–11) teaches the
methods on deliberately controlled data. The advanced set turns the same
instruments on richer phenomena — emotion, deception, sycophancy, humor,
theory of mind, self-knowledge. This file maps each phenomenon to the method
class that can actually measure it, the data shape it needs, and the harness
hooks that already exist. The governing rule carries over from the intro set:
**diversity must be method-shaped.** A phenomenon enters a lab only with a
label scheme the method can score and a control that can kill the claim.

**Supersession note, 2026-06-13:** current lab numbering follows
`ADVANCED_COURSE.MD`. Lab 17 is now persona, voice, roleplay, and register
using `data/persona_register_pairs.csv`; the humor/incongruity idea below is
an older planning slot, not the current Lab 17.

## What the intro set now ships (the seeds)

| Asset | Where | What it seeds |
|---|---|---|
| 12-relation probe set, dual distractors | `data/relation_probes_lab1.csv`, `relation_pairs_lab2.csv` | relation-comparative lens/DLA; class-vs-instance decomposition; near-tie emotion completions |
| `misconceptions` truth family | `data/truth_misconceptions.csv` (wired into Lab 4) | truth vs assertion-frequency dissociation |
| valence + certainty statement pairs | `data/affect_valence.csv`, `data/epistemic_certainty.csv` | property probes beyond truth; confound checks for the truth direction |
| continuous-batching engine | `interp_bench.generate_continuous` | any lab that needs thousands of variable-length generations (behavioral batteries, judge pipelines) |
| claim ledger + evidence tags | bench | every advanced lab keeps the OBS/ATTR/DECODE/CAUSAL/SELF-REPORT discipline |

## Candidate labs

**Lab 12 — Relation geometry (probing + patching).** Where do the 12 relation
classes localize, and do they share circuitry? Data: the relation set, extended
~3× per class (the generator pattern scales; keep the dual-tokenizer
verification). Methods: Lab 4 probes per relation; Lab 5 causal tracing
compared *across* relations. Claim shape: "relation R's subject information
peaks at depth d_R; capitals and languages share/diverge at …".

**Lab 13 — Valence and arousal directions (probing + steering).** Train
valence probes on `affect_valence.csv`; steer with the mass-mean direction
using the Lab 7 machinery (dose-response, drift, side-effect KL all transfer
verbatim). Control: the certainty direction — if steering "positive valence"
also raises confidence markers, the directions are entangled; report it.

**Lab 14 — Certainty, hedging, and calibration (probing + behavioral).**
Does the model represent its own uncertainty? Probe with
`epistemic_certainty.csv`; compare the probe's read against token-level
entropy and against behavioral calibration on the MCQ set. Three measures of
"confidence" that the intro set already computes separately — the lab is
their disagreement matrix.

**Lab 15 — Sycophancy (behavioral + probing + steering).** Port the
250-prompt misconception-pressure battery from the author's DPO eval
(science/math/history/trivia/technology categories, correct/sycophantic
keyword rubrics, 4-label outcome taxonomy correct/mixed/ambiguous/sycophantic
— after Sharma et al. 2023). Two new measurements on top of the behavioral
rate: (a) does a probe see "user-asserted belief" separately from "model
belief"? (train on misconceptions family vs belief-framed variants); (b) does
the refusal-direction methodology of Lab 7 find an "agreement direction" that
causally moves the sycophancy rate? Generation cost is the bottleneck —
exactly what `generate_continuous` is for.

**Lab 16 — Deception vs error (probing + CoT + patching).** The hard one; data
discipline matters most here. Shape: paired tasks where the model is
incentivized to misreport a checkable internal result (e.g., role-play
pressure on misconception items whose ground truth Lab 4 probes can read).
Claim is the *dissociation*: internal probe says X, output says ¬X, and the
CoT-faithfulness battery (Lab 10) scores the report. Never label "lying"
behaviorally alone.

**Lab 17 — Humor and incongruity (lens + attention).** Humor is unlabelable
at scale, but *incongruity* is measurable: joke setups with punchline vs
literal completions (single-token where possible, e.g. pun completions),
surprisal trajectories, and which heads route the setup token that makes the
pun land. Data: a pun-completion set with matched literal controls — build it
with the dual-tokenizer verifier from `make_relation_sets.py`.

**Lab 18 — Theory of mind (patching + probing).** False-belief vignettes in
the Lab 5 clean/corrupt pair shape: "Sally puts the ball in the basket; Anne
moves it to the box; Sally looks in the ___" with belief-state corruptions.
Single-token answers (basket/box verified single-token both tokenizers).
Causal tracing answers *where the protagonist's belief state lives*.

**Lab 19 — Self-knowledge and introspection (SELF-REPORT vs everything).**
Ask the model about its own processing ("did you use the hint?", Lab 10's
question) across all phenomena above; score self-reports against probe and
patching evidence. This is the capstone of the advanced arc the way Lab 11
is for the intro arc: a reconciliation matrix between evidence tiers.

**Labs 20–22 — open slots** for SAE-feature versions of 13/15/16 (do
dictionary features for valence/agreement/deception exist and steer?), in the
Lab 8/9 harness.

## Build rules (so the advanced data stays honest)

1. Every generator is deterministic, vendored, and dual-tokenizer-verified
   where single-token answers are claimed (`make_relation_sets.py` is the
   template).
2. Every phenomenon ships with its confound set (valence ↔ certainty;
   sycophancy ↔ politeness; deception ↔ error; humor ↔ surprise).
3. Behavioral labels stronger than keyword-matching need a judge pipeline +
   a hand-label sample (the Lab 10 acknowledgment-labeling pattern) — judge
   agreement is reported, not assumed.
4. New steering/probe directions get the full Lab 7 control battery (random,
   shuffled, mismatched-pair directions) before any causal language.
5. Claims carry the same evidence tags; the ledger does not get a new tier
   for "vibes".
