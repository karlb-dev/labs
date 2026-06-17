# Design review: Special Topics Labs 26–35

**Reviewer stance:** the same one your course uses on itself — does the design survive when the problem stops being toy-shaped. The skeleton is strong and the evidence-rung discipline ports cleanly. This review spends its words on the places that will bite during implementation, ranked by leverage, then goes lab by lab.

The short verdict: **three structural fixes are worth making before you generate a single lab file; the rest are per-lab sharpening.** The three are (1) a circular dependency between 26 and 27, (2) a missing phenomenon — self-repair/backup — that lives in exactly the labs it would invalidate, and (3) a shared statistics + guaranteed-compute contract the sequence currently assumes but never specifies.

---

## 1. Priority-ranked changes

### P0 — Fix the 26↔27 dependency: causal scrubbing is path-shaped

Lab 26 (causal abstraction + scrubbing) is placed before Lab 27 (path patching), but **causal scrubbing is a recursive tree of path resamplings** — it cannot be implemented without the path-intervention primitive that Lab 27 builds. The proposal already admits this: Lab 26 Experiment B says "path/node resampling *if Lab 27 helpers are available*." That conditional is the bug.

The clean resolution is a **split, not a swap** (keeps your numbering):

- **Lab 26 = causal abstraction only.** Interchange-intervention accuracy (IIA) and Distributed Alignment Search (DAS) need only the interchange primitive from Lab 5 — no path machinery — so Lab 26 correctly stands first and keeps its "state your hypothesis before testing it" pedagogy.
- **Lab 27 = path patching + mediation, with full causal scrubbing as its capstone test.** Scrubbing depends on *both* 26's hypothesis formalism and 27's paths, so it belongs at the end of 27.

If you'd rather not move scrubbing, the fallback is to swap the order outright (27 then 26). Either works; the current order does not.

One more: Lab 26's hypothesis-spec JSON hardcodes `{"layer": 5, "head": 1}` (a GPT-2 induction head). On any other base model that's wrong. The spec should reference a site the student *discovered* via Lab 6, not a baked-in coordinate. Make the example say "site nominated by your Lab 6 circuit," with the GPT-2 numbers only as an illustrative fill-in.

### P0 — Add self-repair / backup / the Hydra effect to Labs 27 and 28

This is the biggest scientific gap. The causal labs (27 path patching, 28 editing) both rest on the assumption that knocking out a component reveals its importance. **That assumption is false in the most interesting cases**, because transformers self-repair: ablate a primary name-mover head and a *backup* head activates to compensate (Wang et al.'s backup name-mover heads), and more generally downstream components partially restore the ablated signal (McGrath et al., "The Hydra Effect"). The consequence is concrete and course-relevant:

- **Node ablation under-estimates importance** (the backup masks the loss), so a path that looks weak may be load-bearing.
- **Edits get silently undone** by backup paths, which is half of why localization doesn't predict editability.

Neither Lab 27 nor Lab 28 mentions this anywhere. Add it as a first-class step:

- In **Lab 27**, add a backup-detection experiment: knock out the primary path, measure whether a secondary path's effect *grows*, and report a **self-repair index** (compensation recovered ÷ primary effect). This is also the cleanest demonstration of why path effects and node effects diverge — which is already the lab's headline.
- In **Lab 28**, make self-repair an explicit failure mode for edits: an edit at the localized site that gets compensated downstream is the mechanism behind "localization ≠ editability."

This single addition is what turns 27/28 from "patching mechanics" into "method pressure," which is the stated theme of the whole sequence.

### P1 — A shared statistics contract

Your own course review scored "dataset and statistical discipline" at 7.5, and the third sequence makes it worse before it makes it better: Lab 27 scans a combinatorial path space, Labs 30/31 scan thousands of features, Lab 29 reads sparse checkpoints. Every one of those is a multiple-comparison surface, and the specs list point-estimate metrics with no uncertainty. For a sequence that ends in a *preregistered* capstone, add one shared contract that every lab inherits:

- **Bootstrap CIs over items** on every gap/recovery/selectivity metric — report the interval, not the point.
- **Permutation null** for every "beats control" claim — the gap must clear the shuffled-label/random-direction null distribution, not just exceed the control's mean.
- **Multiple-comparison correction** (FDR, or a pre-registered candidate-nomination step that caps the comparison count) whenever the lab scans more than a handful of sites/paths/features. Lab 27 and Lab 30 cannot honestly scan their full spaces and then report the max without this.
- **Effect size + n, always.** The capstone rubric has reviewers attack "statistical power" but no lab specifies n or power. Fix that asymmetry.

This is one `interpkit/stats.py` and one paragraph in each lab, and it retires a whole class of reviewer objections.

### P1 — Guaranteed-runnable compute paths per lab

Several labs hedge with "if available" / "fallback proxy" / "synthetic connector smoke." For a *teaching* course, every lab needs one path that is guaranteed to run on the course's baseline hardware, with the heavy "real" version as Tier B. Right now three labs are at risk of being un-runnable as written:

- **Lab 33 (multimodal):** "synthetic connector smoke" is not an experiment. Designate a real floor — a small open VLM (Moondream-class) on synthetic shapes, or a CLIP-only probing path. Without this, Lab 33 is a spec, not a lab.
- **Lab 32 (reward):** make the **DPO/reference log-prob-ratio proxy** the canonical Tier A path (no external reward model needed); the open reward model becomes Tier B.
- **Lab 30 (cross-layer):** crosscoders are expensive to train. Make a small in-course toy with known feature lineage — or public GPT-2 SAEs on adjacent layers — the canonical path.
- **Lab 29 (training dynamics):** the in-course tiny transformer grokking modular arithmetic (Nanda) is cheaper and more reproducible than downloading Pythia checkpoint sequences. Make grokking the Tier A canonical time-lapse; Pythia is the Tier B scale check.

See the table in §4.

### P2 — Lab 28: separate suppression from removal

Lab 28's robustness battery is good but misses the sharpest unlearning test, and it's one your course is uniquely set up to run. After unlearning, **probe the model**: if the "forgotten" fact is still *decodable* from the residual even though it's no longer *generated*, you suppressed the output, not the knowledge. Add three tests that distinguish the two:

- **probe-after-edit** (Lab 4 probe on the unlearned model — does the fact still read out?);
- **fine-tune-to-recover** (does a few steps of benign fine-tuning bring it back?);
- **quantization/format-shift recovery**.

This ties directly to the course's spine — "decodability is not use, and suppression is not removal" — and is the difference between a real unlearning claim and a refusal layer.

### P2 — Lab 34: a surface-cue confound, mirroring Lab 12's token echo

The tool-choice probe in Lab 34 may just be reading lexical cues — "17 * 23" contains operator tokens, so "calculator-needed" is decodable from surface form, not from any internal "decision." This is Lab 12's relation-word-token-echo goblin wearing a tool-shaped coat. Add a **decoupled control set**: calculator-needed tasks with no operator tokens (word problems), and lookup/no-tool tasks that *do* contain digits. The tool-needed direction has to clear that confound before any causal language, exactly as relation patching has to beat the token-echo control.

### P2 — Lab 31: close the loop with intervention-based scoring

Lab 31 scores explanations by predicting held-out activations (detection/simulation, after Bills et al.). The strongest auto-interp test is causal and you already have the machinery: **clamp the feature and check whether the label predicts the output change** (Lab 28's feature clamp). Add intervention scoring as a third axis alongside detection and generation scoring, and add a judge-leakage control (the LLM judge must not see the same contexts that generated the label).

### P2 — Lab 27: add attribution patching (AtP*) as the scalability bridge

The course teaches activation patching (Lab 5) and attribution graphs (Lab 9); the missing link is **attribution patching** — the first-order gradient approximation that makes patching scale (Kramár et al., AtP*; Syed et al.). Lab 27 is the natural home. Teach it *with its failure mode*: the linear approximation diverges from true patching under large or saturating effects, so the lesson is "when is the cheap approximation trustworthy," which is on-theme.

---

## 2. Per-lab deltas

### Lab 26 — Causal Abstraction (+ Scrubbing)
- **Keep:** the hypothesis-spec JSON, the precision/recall framing (predicted-invariant vs behavior-preserving coverage), the refinement log. The `predicted_preservation_min` / `predicted_damage_when_broken_min` preregistration is a genuinely nice touch — it forces a falsifiable prediction.
- **Add DAS** (Distributed Alignment Search) as the concrete alignment-finding method — it's the modern operationalization of causal abstraction and the lab cites Geiger's theory paper but not the method.
- **Mandatory adversarial control:** the subspace illusion (Makelov et al., "Is This the Subspace You Are Looking For?"). DAS can "find" an alignment inside a *random* high-dimensional subspace, so the lab needs a random-subspace / random-label null that **should fail** — and a student whose DAS alignment doesn't beat that null has found an illusion, not a variable.
- **Foreground the trivial-pass failure (when scrubbing lands here or in 27):** an over-permissive hypothesis (resample almost everything) passes vacuously. Require the calibration pair "scrub-everything" vs "scrub-nothing," plus one deliberately over-permissive hypothesis that passes and one too-strict hypothesis that fails, as worked counterexamples. This is *the* thing students get wrong with scrubbing.
- Move full scrubbing to Lab 27 per P0.

### Lab 27 — Path-Specific Patching and Causal Mediation
- **Keep:** node-vs-path baseline, interaction residual, mediation accounting waterfall. The interaction-residual decomposition is the right idea.
- **Add self-repair/backup** (P0) — the most important addition in the whole review.
- **Add AtP\*** (P2) as the scalable approximation with its breaking point.
- **Receive full causal scrubbing** from Lab 26 as the capstone.
- **Statistics (P1):** the path space is combinatorial; require a nomination step (don't scan all paths) + permutation null + FDR. As written, "report top paths" invites max-selection over an uncorrected space.
- Note: this lab is now heavy (paths + mediation + AtP\* + self-repair + scrubbing). Consider pushing deep mediation-accounting to an extension so the core stays runnable.

### Lab 28 — Mechanistic Editing and Unlearning
- **Keep:** localization-vs-editability as the whole lab, retain/forget frontier, the safety scope (benign-only is correct).
- **Add suppression-vs-removal** (P2) — probe-after-edit, relearning, quantization recovery.
- **Make the Hase et al. dissociation the explicit null** ("localization does not predict the best edit site"), and connect the *mechanism* of that null to self-repair from Lab 27.
- **Unlearning robustness reference:** cite the relearning-attack / "many unlearning methods only suppress" literature (e.g., Lynch et al., "Eight Methods to Evaluate Robust Unlearning in LLMs") so the robustness battery has a named target.

### Lab 29 — Training Dynamics and Circuit Birth
- **Keep:** the phase taxonomy (behavioral-before-interpretable, decodable-before-behavioral, migration, sharpening, redistribution) — this is the best part and it's well-specified.
- **Make grokking the Tier A canonical** (P1); Pythia is Tier B.
- **Sharpen the headline question:** circuit-formation-precedes-behavior vs behavior-precedes-circuit is the publishable axis; foreground it over the descriptive trajectory.
- The forbidden claim ("learned X at exactly this step") already guards the sparse-checkpoint over-reading — good. Add the matching positive: report first-decodable and first-causal as *intervals between checkpoints*, never as points.

### Lab 30 — Cross-Layer and Cross-Model Feature Geometry
- **Keep:** the lineage graph with `(model, checkpoint, layer, feature)` nodes, split/merge case taxonomy, the dictionary-artifact category.
- **Guaranteed path** (P1): a toy with known lineage or public adjacent-layer SAEs.
- **Name the expected phenomenon:** feature splitting (Anthropic) — one coarse early feature resolving into several specific later ones — so students have a hypothesis to test rather than a fishing expedition.
- **Statistics (P1):** feature-pair space is enormous; nomination + null is non-negotiable here.

### Lab 31 — Automated Interpretability at Scale
- **Keep:** held-out prediction scoring, abstention frontier, confusable negatives, synthetic gold-label recovery. The synthetic-gold subset is the right backbone.
- **Add intervention-based scoring** (P2) and the judge-leakage control.
- **Add the "too broad" failure explicitly** — contrastive/fuzzing scoring catches labels that match topic but not activation; this is the dominant real-world auto-interp failure and deserves its own metric, not just "broad/narrow classification accuracy."

### Lab 32 — Reward Models and Preference Circuits
- **Keep:** the confound-direction battery (length, politeness, agreement, sentiment, hedging), reward-vs-policy disagreement set, sycophancy risk quadrant. The confound directions are exactly right.
- **Canonical path** (P1): DPO log-prob-ratio proxy.
- **Sharpen the null:** "the reward direction is length/sentiment, not value" is the well-documented failure (reward models latch onto length); make beating the length/sentiment confound the gate, framed as reward-hacking-in-miniature.
- Anthropomorphism guard ("understands human values" forbidden) is good.

### Lab 33 — Multimodal Mechanistic Interpretability
- **Keep:** modality-handoff atlas, image-vs-text probe transfer, region patching, OCR-leak metric, the benign-image safety scope.
- **Real floor** (P1) — this is the lab most at risk of being un-runnable.
- **Make OCR/text-in-image leak a gating control**, not just a reported metric — the "model reads rendered text instead of seeing" shortcut is this lab's token-echo goblin.
- **Flag connector alignment:** images produce variable token counts, so image-state patching has the same token-alignment hazard Lab 12 warns about. Add an alignment-validation diagnostic.

### Lab 34 — Tool Use, Agents, and State Tracking
- **Keep:** tool-choice decoding, tool-boundary patching, self-report-vs-trace. The self-report-vs-known-trace design is strong.
- **Add the surface-cue confound** (P2) as a gating control.
- **Prerequisite the Lab 15 multi-turn validity checks** (cache parity, turn boundaries, null traces) explicitly — state-tracking across turns is built on that foundation and the lab currently doesn't cite it.
- The "persistent goal/plan" forbidden claim is the right guard; reinforce that a tool-needed *direction* is not an *intention*.

### Lab 35 — Reproducible Interpretability Paper Capstone
- **Keep:** preregistration → frozen run → adversarial review → repair → package. The structure is excellent and is the right ending.
- **Add a statistics line to the rubric** (or fold into Control design) — reviewers attack power but the rubric doesn't reward it (currently 0% explicit).
- **Structure the AI-reviewer path:** a fixed reviewer prompt/rubric so adversarial-review quality is reproducible rather than reviewer-dependent.
- **Require a FAILURE_MODES contribution** (see §4) and a "red-team your own controls" subsection in the review.

---

## 3. Cross-cutting additions

### A living `FAILURE_MODES.md`
The sequence's theme is "does the method survive." Make that cumulative: a single failure-mode atlas that each lab contributes a row to, so by Lab 35 students have a catalog of how interpretability methods mislead. Seed rows: trivial-scrub-pass and subspace illusion (26), self-repair / backup paths (27), suppression-not-removal (28), exact-step over-reading (29), dictionary-artifact features (30), polysemantic-label fog (31), reward-direction-is-length (32), OCR/modality leak (33), surface-cue-not-decision (34). This is a distinctive, course-defining artifact that matches your ledger philosophy and costs almost nothing.

### Sharpen the `FORMAL` tag
The tag currently bundles two different formalisms. Split the definition so students know which they earned: (a) an **alignment claim** (IIA/DAS — a high-level variable is realized by a specified low-level subspace under interchange) and (b) a **resampling-consistency claim** (scrubbing — behavior survives all resamplings the hypothesis permits). Both require an explicit variable→site map and stated intervention semantics; they are not interchangeable evidence.

### Guaranteed-compute path table

| Lab | Tier A canonical (must run) | Tier B (scale/real) |
|---|---|---|
| 26 | GPT-2 induction + relation-swap, interchange/DAS | course base model |
| 27 | GPT-2, nominated paths + self-repair | larger model + AtP\* |
| 28 | GPT-2 counterfactual facts + Lab 20 organism | weight edits on base model |
| 29 | in-course grokking toy | Pythia checkpoint suite |
| 30 | toy / public adjacent-layer SAEs | trained crosscoders |
| 31 | synthetic-gold features | full SAE atlas |
| 32 | DPO log-prob proxy | open reward model |
| 33 | small open VLM or CLIP-probe on synthetic shapes | full VLM + VLM SAE |
| 34 | instruct model + toy tools | larger agent loop |
| 35 | inherits chosen track | — |

### Reference additions
The existing bibliography is accurate and well-chosen. Add:
- Geiger et al., **Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations** (DAS) — the method for Lab 26.
- Makelov et al., **Is This the Subspace You Are Looking For? An Interpretability Illusion for Subspace Activation Patching** — the mandatory adversarial control for Lab 26.
- McGrath et al., **The Hydra Effect: Emergent Self-repair in Language Model Computations** — Labs 27/28.
- Rushing & Nanda, **Explorations of Self-Repair in Language Models** — Labs 27/28.
- Kramár et al., **AtP\*: An Efficient and Scalable Method for Localizing LLM Behaviour to Components**, and Syed et al., **Attribution Patching Outperforms Automated Circuit Discovery** — Lab 27.
- A robust-unlearning evaluation paper (e.g., Lynch et al., **Eight Methods to Evaluate Robust Unlearning in LLMs**) — Lab 28.
- Tan et al., **Analysing the Generalisation and Reliability of Steering Vectors** — a control reference for the steering used in 28/32.

---

## 4. Optional extra topics (only if you extend past ten)

These are genuinely missing from the arc but I would *thread* them into existing labs rather than add slots, except possibly the first:

1. **Weights-based circuit reading (QK/OV composition).** The whole course is activation- and intervention-first; a weights-based lab (reading attention QK/OV matrices and head composition directly, à la the Mathematical Framework) is the one paradigm absent. This is the strongest candidate for an 11th lab if you want one.
2. **Attribution patching as its own lab,** if Lab 27 gets too heavy carrying paths + self-repair + AtP\* + scrubbing.
3. **A dedicated "interpretability illusions" lab** (probing illusion, subspace illusion, patching-via-self-repair illusion). I'd thread these through 26/27/31 instead — but if you want one lab that is purely "here is how each method lies," it would be a fitting penultimate before the capstone.

---

## 5. Suggested final shape

Minimal-disruption version that fixes the real problems:

- **26** Causal Abstraction (IIA + DAS), with the subspace-illusion control. Scrubbing motivated here on a single path only.
- **27** Path patching + mediation + **self-repair/backup** + **AtP\*** + **full causal scrubbing** as capstone.
- **28** Editing/Unlearning, with **suppression-vs-removal** and self-repair as the editability null.
- **29–34** as specified, with the per-lab deltas and guaranteed-compute paths.
- **35** Capstone, with a statistics rubric line and the `FAILURE_MODES.md` contribution.

Plus three shared assets created once: `interpkit/stats.py` (the statistics contract), `FAILURE_MODES.md` (the living atlas), and the sharpened `FORMAL` tag definition.

The sequence is already aiming at the right target — research-ready, not trophy-hunting. These changes mostly close the gap between that aim and what the specs currently guarantee: a real dependency untangled, the one phenomenon that breaks causal claims put where it belongs, and honest statistics and compute paths made explicit instead of assumed.
