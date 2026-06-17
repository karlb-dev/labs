# Lab 36 Severance — Final Run Summary & Next-Experiment Spec

- **Date:** 2026-06-16 → 2026-06-17
- **Branch:** `interp_sev` (all work pushed to `origin/interp_sev`)
- **Machine:** Colab VM, NVIDIA RTX PRO 6000 Blackwell (97 GB)
- **Models:** SmolLM2-135M-Instruct (tier A), Olmo-3-7B-Instruct (tier B), Olmo-3.1-32B-Instruct (tier C), Gemma-4-E4B-it, plus Olmo-3-7B-Think and Olmo-3.1-32B-Think (reasoning axis). `gpt-oss-120B` intentionally not run (see decision below).
- **Note on the model list:** there is no `allenai/Olmo-3-32B-Instruct` on the Hub; the only 32B Instruct in the Olmo 3 family is `Olmo-3.1-32B-Instruct`, which *is* tier C. So "Olmo 3 32B Instruct" and tier C are the same model.

This file is the master index over the four result directories under `MyDrive/interpret/` produced this session, plus a full spec for the next experiment.

---

## 1. What we set out to do

Validate the latest Lab 36 (commit `9dda941`, the v3 content-blind B5 + row-randomized B4 patch) against the pre-v3 baselines in `verify_severance/`, run it across all requested models, fix anything broken, and — once it became clear the headline tracks were either degenerate or artifactual — do everything possible to make the result either a real positive or a clean, strong negative.

## 2. The runs (chronological), with directories

| Phase | Dir under `MyDrive/interpret/` | What it is |
|---|---|---|
| Baseline | `verify_severance/` | Pre-v3 runs (commit `bd18deb`), already present. The comparison reference. |
| v3 full | `lab36_fullrun_20260616/` | Latest v3 code on 4 models (A/B/C/Gemma), `--mode all --prompt-set full`. |
| Improved | `lab36_improved_20260616/` | B4 canonical-answer fix + B5 sentinel control + reasoning axis. 6 models. |
| Readout v1 | `lab36_readout_20260617/` | First representational readout probe (projection, n≈8). 6 models. |
| **Maximal power** | `lab36_maxpower_20260617/` | **Definitive.** Expanded dataset (60 heldout), trained transformed-direction probe, dose+placement sweeps. 6 models. |

Each directory has a `VALIDATION_REPORT.md`, per-run trees, and logs. The repo's `VALIDATION_SEVERANCE_LAB36.md` carries the same results in version control.

## 3. What we found (the scientific arc)

**(a) v3 full run.** All four models returned `no_report_channel_coupling_validated`. B5 *looked* promising under the content-blind decision — Gemma-4-E4B d′=1.24 (CI excl. 0), Olmo-3.1-32B d′=0.79 — but every model failed the content-leak gate, and B4 was **degenerate**: canonical-answer plausibility ~0 on every model, so B4 was producing neither a signal nor a valid negative.

**(b) B4 fixed → valid negative.** B4 now teacher-forces the model's **own greedy default-route answer** (matched across all hidden routes) instead of an off-distribution CSV string. Plausibility went **0.0–0.22 → 1.000** on every model; B4 became a valid test and is a clean negative (activation-source accuracy ≤ 0.11, at/below its fresh-transcript control). (Subtle bug caught: plausibility must be read on the non-injected route — the injection depresses the answer logprob by design.)

**(c) B5 sentinel control → the apparent signal is an artifact.** A new control injects the *same* perturbation at an upstream sentinel position instead of on the decision token. The report-query d′ (0.61 / 1.24 / 0.79 on 7B / Gemma / 32B) **collapsed to ≈0** under upstream injection. The B5 "anomaly detection" was the decision logit being directly steered, not propagated monitoring.

**(d) Reasoning axis → worse, not better.** Both Think models saturate the content-blind decision at false-alarm 0.97 (they claim a perturbation almost always) and leak more content. Reasoning training made the report channel noisier, not more faithful.

**(e) Readout probe → "absent," not "present-but-unreported."** A representational readout (does a probe decode injected-vs-clean from the model's own decision-position residual under upstream injection?) had a perfect positive control (AUC 1.0 when injected on the decision token) but read **chance** under upstream injection on every model. Dose to 8× did not revive the verbalized decision.

**(f) Maximal-power → airtight.** To leave no escape hatch, we expanded the detection set to **60 heldout items** (null bar 0.85 → 0.63), added a **trained transformed-direction probe** (pooled + per-direction; can catch a *rotated* signal the projection cannot) with positive controls + permutation nulls, and a **sentinel placement sweep** (early/mid/late). Result on all six models:

- both positive controls **AUC = 1.000**;
- every sentinel readout at **chance below its null** (projection, trained-pooled, and **0/5 per-direction**);
- verbalized **d′ = 0 at every dose (to 8×) and placement**.

### Headline table (maximal-power, 6 models)

| Model | proj ctrl | proj sent (null) | trained ctrl | trained sent (null) | per-dir >null | d′ doses/placements |
|---|--:|--:|--:|--:|--:|--:|
| SmolLM2-135M | 1.000 | 0.500 (0.62) | 1.000 | 0.500 (0.63) | 0/5 | 0.000 |
| Olmo-3-7B-Instruct | 1.000 | 0.501 (0.63) | 1.000 | 0.500 (0.64) | 0/5 | 0.000 |
| Olmo-3-7B-Think | 1.000 | 0.500 (0.62) | 1.000 | 0.500 (0.62) | 0/5 | 0.000 |
| Gemma-4-E4B | 1.000 | 0.500 (0.64) | 1.000 | 0.500 (0.64) | 0/5 | 0.000 |
| Olmo-3.1-32B-Instruct | 1.000 | 0.500 (0.63) | 1.000 | 0.500 (0.64) | 0/5 | 0.000 |
| Olmo-3.1-32B-Think | 1.000 | 0.500 (0.64) | 1.000 | 0.500 (0.64) | 0/5 | 0.000 |

## 4. What worked, what didn't, bugs fixed

**Worked / added to the lab (now in `interp_sev`):**
- B4 model-canonical-answer instrument (`B4_USE_MODEL_CANONICAL_ANSWER`).
- B5 sentinel control (`B5_SENTINEL_VARIANT`, `sentinel_content_blind_logit_only`).
- Representational readout: projection + trained transformed-direction probes, report-query positive control, permutation null, per-direction probes.
- Sentinel dose sweep (to 8×) and placement sweep (early/mid/late).
- Expanded detection dataset (8 → 85 items, 60 heldout, 12/direction) via the generator.
- Lean `readout` mode (directions + probe suite without the generation-heavy verbalized B5).

**Bugs caught and fixed mid-run:**
- B4 plausibility was measured under the injected route (which depresses it by design) → re-pointed to the non-injected `matched_default` route.
- The first sentinel implementation framed the question in the assistant turn (KV-replay style), which **confounded framing with injection-site and saturated the decision** → rebuilt as a single-position interior injection on the *same* user-framed prompt.
- The trained mean-difference probe was unreliable at n≈8 (even its positive control failed) → switched the headline to a training-free **projection onto the known injected direction**, then re-introduced a **standardized** trained probe at n=60 where its control reads 1.0.
- A `pgrep -f` watcher self-matched its own command line (cosmetic; worked around with log-marker watchers).

**Decision — gpt-oss-120B not run.** The bottleneck was never scale: the reasoning axis (a cheaper, more principled test) showed reasoning models are *worse*, and the negative held across 135M→32B. Porting B4/B5 to gpt-oss's MoE + harmony path is the expensive, risky part and would, on this evidence, most likely return the same artifact-then-null. Not worth the compute.

## 5. Bottom line

For **injected state directions**, Lab 36 is now a clean, well-powered **negative**: no functional report-channel coupling survives controls, for any direction / model tested, and the one apparent positive (content-blind B5) was decision-token direct-steering. The lab's enduring value is **methodological** — a worked demonstration of how activation-injection self-report experiments manufacture false positives, plus the control battery (matched-output, sentinel, dose, placement, representational readout with positive control + permutation null) that exposes them.

The honest limit: this tells us injected artificial directions aren't reported. It does **not** tell us whether the model can report **natural internal states it actually computes**. That is a different question and the subject of the spec below.

---

# 6. Next experiment — full spec: Lab 37, Natural-State Reportability (Feeling-of-Knowing)

## 6.1 Why this is the right next experiment

Lab 36 probes reportability by *injecting* a direction. A plausible reason for its null is that injection is simply the wrong probe: an artificial perturbation at one layer/position need not engage whatever machinery (if any) couples genuine internal states to self-report. Lab 37 removes the injection and asks the sharper, more ecologically valid question: **is a natural internal state the model demonstrably computes functionally coupled to its verbal self-report, under matched-text and causal controls?**

We pick **feeling-of-knowing (FoK)** — whether the model can correctly answer a factual question — as the natural state, because it is the rare internal state that is simultaneously:
1. **Independently verifiable** (we know whether the answer is actually correct);
2. **Known to be linearly decodable** from pre-answer activations (Kadavath et al., "Language Models (Mostly) Know What They Know"); and
3. **Verbally elicitable** ("How confident are you that you can answer this correctly?").

That trifecta is exactly what Lab 36's injected directions lacked (they were verifiable and decodable but artificial). FoK lets us run the *same* counterfactual logic on a state the model genuinely has.

## 6.2 Core question and claim ceiling

> Is the model's reported confidence **functionally and causally coupled** to its internal correctness representation, beyond what is inferable from the question text alone?

Allowed claim ceiling: *functional/causal coupling (or its absence) between an internal, independently-verified state and verbal self-report, under matched-text and intervention controls.* Forbidden, as in Lab 36: phenomenal experience, introspection-as-testimony, consciousness.

## 6.3 Tracks

| Track | Question | Claim ceiling |
|---|---|---|
| N0 Instrument proof | Hook/lens/KV/position parity (reuse Lab 36). | Plumbing only. |
| N1 Ground truth | Run a QA set; record actual per-item correctness (greedy). | Defines the state to be reported. |
| N2 Decodability | Can a CV probe predict correctness from the pre-answer residual? | Internal state is represented (`DECODE`). |
| N2-text Text-only baseline | Can correctness be predicted from the question surface alone (frozen-embedding probe)? | Confound floor: how much "knowing" is in the prompt, not the computation. |
| N3 Reported confidence | Does the model's verbalized confidence predict actual correctness (calibration)? | Report-level signal only. |
| N4 Coupling (headline, correlational) | Does reported confidence track the **internal** probe beyond the **text-only** baseline? | Functional coupling candidate. |
| N5 Coupling (headline, causal) | Does intervening on the internal correctness representation move the **report** without changing the question text? | Causal report-channel coupling — the strong claim. |
| N6 Counterexample ledger | Where does the report diverge from the verified state? | Claim boundary. |

N4 (correlational) and N5 (causal) are the co-headlines, mirroring Lab 36's B4/B5.

## 6.4 Data

A frozen QA set engineered to span the model's competence so correctness varies (≈40–50% correct gives maximal probe power):
- **Sources:** short-answer factual QA (e.g., TriviaQA / NaturalQuestions-style single-fact items) plus a calibrated-difficulty synthetic set (entity attributes, arithmetic of graded difficulty, date/unit conversions). Target ≈ 300–600 items so heldout probe n is in the hundreds.
- **Per item:** `question`, `gold_answer(s)`, `difficulty_bucket`, `split` (train/val/heldout), and an `answerability` flag (some genuinely unanswerable items to test "knowing that you don't know").
- **Deterministic generator** + MANIFEST hashing, following `severance/data/` conventions. Add `qa_fok_items.csv`.
- Correctness scoring: exact/alias match + a held-out human-checked subsample to calibrate the matcher (reuse Lab 36's "human-label before strong claims" discipline).

## 6.5 Procedure (per model)

**N1 — Ground truth.** For each item render the QA prompt, greedy-decode the answer, score correctness vs gold. Record per-item `correct ∈ {0,1}` and the answer-token log-probs. Keep the *pre-answer* residual cache at the final prompt token across all layers (reuse `run_with_residual_cache`).

**N2 — Decodability.** Train a cross-validated probe (standardized logistic / nearest-centroid, the Lab 36 maximal-power probe) to predict `correct` from the pre-answer residual, per layer; report max-over-layer held-out AUC with a permutation null. This is "does the model internally represent whether it knows."

**N2-text — Confound floor.** Same target, but features = a **frozen sentence-embedding of the question** (or the model's own layer-0/embedding-mean), i.e. no forward computation. AUC here = correctness inferable from the question surface alone. The **internal-minus-text gap** is the part of "knowing" that is computed, not lexical.

**N3 — Reported confidence.** Before the model answers, elicit confidence two ways: (a) a verbal scale ("On 0.0–1.0, how likely are you to answer the next question correctly? Reply with a number."), and (b) a next-token yes/no logit ("Can you answer this correctly? yes/no") read content-blind from logits (the Lab 36 decision-channel trick, robust to verbosity). Calibration AUC = reported confidence vs actual `correct`.

**N4 — Correlational coupling (headline 1).** Does reported confidence predict `correct` **beyond** the text-only baseline? Compute AUC of reported confidence, and the partial association controlling for the N2-text prediction (e.g., logistic with text-only score as a covariate; report the added AUC / likelihood-ratio). If reported confidence adds nothing over text-only → the report narrates the prompt. If it adds signal that the internal probe also has → candidate functional coupling.

**N5 — Causal coupling (headline 2).** The decisive test, in Lab 36's counterfactual spirit, on a natural state:
- Build a **correctness direction** from the N2 probe (difference of known/unknown residual means at the best layer, or the probe weight vector).
- On matched items, **patch/steer** the pre-answer residual along ±this direction (and via activation-patching known↔unknown residuals at the matched final-prompt position) **without changing the question text**, then re-elicit reported confidence.
- Coupling = reported confidence shifts in the predicted direction with dose, above the control battery. **Controls (reuse Lab 36):** random direction, shuffled direction, wrong-layer, and a content-blind position audit; plus a **task-distribution control** (does the intervention also change actual correctness? — if it changes correctness, that is functional confidence-tracking; if it changes *only* the report while behavior is held, that is the strongest report-channel result, paralleling B4's matched-output logic).
- Sentinel variant (reuse Lab 36): apply the correctness-direction steer at an upstream position vs at the decision token to separate genuine propagation from direct logit steering.

**N6 — Counterexamples.** Items where reported confidence and verified correctness diverge most (confident-wrong, unconfident-right), and where the causal intervention fails. Populate a `failure_specimens.md`.

## 6.6 Metrics, gates, falsifiers

| Metric | Meaning | Headline gate (pilot) |
|---|---|---|
| `n2_decode_auc` (vs null) | internal correctness representation exists | > null and > 0.65 |
| `n2_internal_minus_text_gap` | computed (not lexical) knowing | > 0 with CI excluding 0 |
| `n3_reported_calibration_auc` | report predicts correctness | report value, no gate |
| `n4_added_auc_over_text` | report tracks internal state beyond prompt | > 0 with CI excl. 0 → coupling candidate |
| `n5_causal_report_shift_dprime` | report moves under intervention vs controls | ≥ 0.5 above control floor, sentinel-robust |
| `n5_behavior_held` | report moved while correctness held (matched-output) | report shift with |Δcorrectness| small |

**Falsifiers / expected outcomes:**
- *Decodable but not reported* (`n2` high, `n4≈0`, `n5≈0`): the model knows internally but the report is prompt-narration — the present-but-unreported result Lab 36 could not establish for injected states.
- *Reported = text-only* (`n4≈0`): self-report is confabulated from the question surface.
- *Causal coupling* (`n5` passes, sentinel-robust, behavior held): genuine functional report-channel coupling for a natural state — the positive Lab 36 was built to find and didn't, now on the right substrate.
- *Confidence moves only with correctness* (`n5` passes but behavior moves too): functional confidence-tracking, not a pure report channel (still interesting; weaker claim).

**What it can claim:** functional and/or causal coupling (or its absence) between a verified internal state and self-report, for a named model/data/layer/dose/scoring rule, under matched-text and control batteries.
**What it cannot claim:** phenomenal introspection, experience, or that the report is testimony.

## 6.7 Compute tiers & reuse

- Tier A SmolLM2-135M (CPU/smoke), Tier B Olmo-3-7B-Instruct, Tier C Olmo-3.1-32B-Instruct; reasoning axis Olmo-3-7B/3.1-32B-Think; Gemma-4-E4B for an architecture check. Same fleet as Lab 36.
- **Reuses directly from Lab 36 / the bench:** residual cache + lens/hook/KV parity diagnostics, positioned-steering hook, the maximal-power readout probe (projection + standardized trained probe + permutation null + report-query positive control), the sentinel control, the dose/placement sweeps, the control battery (random/shuffled/wrong-layer), and the claim-grammar / failure-specimen discipline. Estimated build: most machinery exists; new work is the QA data generator, the FoK elicitation prompts, the N2-text baseline, and wiring N1–N5 as a new lab module.
- Cost estimate: comparable to a Lab 36 run per model (forward passes + a few probes); a few GPU-hours for the full fleet.

## 6.8 One-paragraph rationale for whoever picks this up

Lab 36 establishes, cleanly, that injected directions are not reported. Lab 37 keeps every control that made that result trustworthy (matched output, sentinel, dose/placement, decode-with-positive-control-and-null) and points them at a state the model verifiably has and can be asked about. Either it finds the first clean evidence of natural-state report-channel coupling (a real positive, on the right substrate), or it shows that even genuine, decodable internal states are not faithfully reported — both of which are new data that the injection-based design cannot produce.

---

## 7. Repro / provenance

- Commits on `interp_sev`: `b194efb` (v3 validation + project settings), `fc81acf` (B4 fix + sentinel + reasoning axis), `311b398` (sweep + readout probe), `091bc03` (maximal-power readout). Built on pre-existing `9dda941` (v3 content-blind).
- Regenerate data: `python severance/data/make_severance_lab36_data.py --output-dir severance/data`.
- Re-run the definitive readout: `python interp_bench.py --lab lab36 --tier <a|b|c> [--model <id>] --mode directions,readout --prompt-set full [--max-examples 0] --no-plots --run-name <name>`.
- Regenerate tables: `python temp/lab36_maxpower_table.py` (and `temp/lab36_compare_improved.py`, `temp/lab36_extract.py`).
