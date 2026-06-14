# Lab 11: Mechanistic Reliability Audit

**Evidence level targeted:** integration.  Every measured claim keeps its own rung: `OBS`, `ATTR`, `DECODE`, `CAUSAL`, or `SELF-REPORT`.  The audit earns authority by refusing to blend those rungs into one undifferentiated confidence score.

**Prerequisites:** the whole course (Labs 1-10), especially your `claim_ledger.md` with drafted claims + falsifiers from every prior lab.  The capstone is built on the ledger.  If your ledger is empty, the harness will still run, but the central assignment will be missing its spine: there are no claims to keep, narrow, or retire. This lab integrates the instruments from Labs 1 (lens), 2 (DLA), 4 (probe/monitor), 5 (patching/causal), 6 (circuit scope), 8 (dictionary residue), 9 (feature graphs), and 10 (CoT self-report) under one audit that refuses to scramble their evidence rungs.

## Core question

Given behavioral evidence and internal evidence, where should we trust this model less, and what may we responsibly say?

## Why this lab exists

The previous labs taught instruments.  This lab teaches judgment — and the discipline of keeping evidence rungs separate.

A logit-lens curve does not deploy.  A probe AUC does not deploy.  An attribution graph does not deploy.  What deploys is a sentence like this:

```text
For task boundary B, model M is reliable enough for use U, except under conditions N,
because artifacts A1-A4 support claims C1-C3 and fail to support claim C4.
```

That sentence is dangerous if its evidence rungs get scrambled.  A strong `DECODE` result (Lab 4/8 style) can make a weak safety case sound causal.  A causal patch on six examples (Lab 5) can make a narrow mechanism sound universal.  A fluent chain of thought (Lab 10) can make self-report sound like an audit log.  Lab 11 is the place where those risks are named, weighed, and kept out of the final claim.

**Make the concept pop:** the audit earns authority precisely by refusing to blend rungs.  The evidence_matrix.csv is the tool that forces you to say "this number is DECODE, not CAUSAL" and "this behavioral flip is SELF-REPORT, not proof of internal mechanism."  A high flip rate plus a non-selective probe is not a contradiction — it is information about what this particular internal method could and could not isolate.

The audit also cashes out the claim ledger.  All semester you drafted claims with falsifiers.  Now you reconcile them: **keep**, **revise**, or **retire**.  A retirement with a good artifact earns full credit — exactly as in the earlier labs' "negative results as informative."  A semester with no retired or narrowed claims usually means the falsifiers were not operational.  At least one revise or retire is required for the capstone to count as an audit rather than a victory lap.

## What the harness produces

The output contract is rigid so audits are comparable across students:

```text
audit_report.md                         fixed schema; student judgment sections marked
ledger_reconciliation.md                keep / revise / retire worksheet
safety_case_and_rebuttal.md             two paragraphs for, one against; both graded
results.csv                             per-example behavior, confidence, internal summaries,
                                        failure_mode_auto, failure_mode_student
tables/evidence_matrix.csv              method -> evidence rung -> supported claim -> non-claim
tables/ledger_reconciliation_matrix.csv spreadsheet-friendly ledger verdict worksheet
plots/audit_dashboard.png               domain-specific six-panel cockpit
plots/audit_scorecard.png               visual rung firewall for the evidence matrix
plots/*atlas*.png, *specificity*.png     per-example/pair/fact detail views
tables/audit_scorecard.csv               metric, rung, value, artifact, and caveat
tables/claim_readiness_matrix.csv        method scorecard: rung, metric, control, claim boundary
tables/plot_reading_guide.csv           what each plot supports and cannot support
internal_evidence/                      monitor / probe metrics and projections
```

Sections marked `[STUDENT — graded]` are not boilerplate.  They are the assignment.  A submitted audit with those markers still present is a run log, not an audit.

## Domain A: `factual_qa`

This is the default domain.  It continues Lab 5’s factual-recall setup on a base model:

```bash
python interp_bench.py --lab lab11 --tier a
python interp_bench.py --lab lab11 --tier b
```

Each example is a capital-fact prompt under one of three templates:

```text
The capital of France is
The capital city of France is
In France, the capital is
```

The target is the true capital.  The distractor is the cyclic partner fact’s capital, so the metric is a clean next-token logit difference:

```text
logit(target capital) - logit(cyclic-partner capital)
```

The factual audit measures five rungs:

| Method | Rung | Artifact | What it supports | What it does not support |
|---|---:|---|---|---|
| Next-token behavior | `OBS` | `results.csv` | Whether the model emits or prefers the target on these prompts | Where the fact is stored |
| Logit lens | `OBS` | `tables/lens_stabilization.csv` | When the target becomes readable or preferred under the final readout | That later layers use the readable signal |
| Direct logit attribution | `ATTR` | `tables/dla_layer_summary.csv` | Which component writes align with the answer direction under the frozen-norm ledger | Causal responsibility |
| Residual patching | `CAUSAL` | `tables/causal_subset.csv` | Whether a specific residual stream site recovers the clean behavior | A complete mechanism for all facts |
| Truth monitor | `DECODE` | `internal_evidence/truth_monitor.json` | Whether true/false fact labels are linearly separable on held-out audited facts | That the monitor direction causes factual answers |

### Improvements in this version

The revised lab makes the factual audit less slippery:

- It writes `diagnostics/factual_tokenization_report.csv`, including prompt token IDs, target/distractor pieces, subject token position, and drop reasons.
- It runs the component decomposition self-check before DLA, because an attribution ledger without reconstruction is arithmetic theater.
- It distinguishes exact top-1 accuracy from target-vs-distractor preference.  A model can prefer `Paris` over `Berlin` while still top-1 continuing with punctuation or a phrase.
- It writes all per-layer DLA rows, not just the top layer.
- It patches two sites: an early subject-token site and the final position at the preference-stabilization band.
- It adds an unrelated-clean control patch at the same sites, so recovery is not allowed to impress you without a shadow.
- It fixes the truth-monitor depth issue: a saved Lab 4 direction is only used at its saved stream depth and only when model-compatible.  Otherwise, the lab fits a local mass-mean direction on train facts, chooses depth by train AUC, and reports held-out AUC versus a shuffled-label control.

## Domain B: `cot_faithfulness`

This is the recommended flagship.  It continues Lab 10 on a fresh item slice:

```bash
python interp_bench.py --lab lab11 --tier b \
  --audit-domain cot_faithfulness \
  --model allenai/Olmo-3-7B-Think
```

The purpose is replication plus one mechanistic follow-up.  Lab 10 asked whether visible reasoning faithfully reports influences.  Lab 11 asks whether that result survives a fresh slice, then adds a `DECODE` method: can the model’s residual stream at answer-emission time linearly distinguish baseline prompts from wrong-hinted prompts?

The CoT audit measures:

| Method | Rung | Artifact | What it supports | What it does not support |
|---|---:|---|---|---|
| Hint injection | `SELF-REPORT` | `tables/faithfulness_by_hint_type.csv` | Whether answers move to hinted options and whether generated CoT mentions or attributes the influence | Intent, honesty, or deception |
| CoT text interventions | behavioral `CAUSAL` | Lab 10 intervention tables | Whether visible text carries load under truncation, filler, clean resume, and add-mistake interventions | An internal causal path |
| Hint-presence probe | `DECODE` | `internal_evidence/hint_presence_probe.json` | Whether hinted vs baseline conditions are linearly separable at answer-emission time | That the decoded direction causes the answer |

The fresh-slice rule is written to `diagnostics/cot_fresh_slice_manifest.json`.  The point is not statistical wizardry.  The point is to avoid auditing the same item slice that formed your belief in Lab 10.

### Compatibility note

The revised Lab 11 tolerates both the original and revised Lab 10 helper APIs.  In particular, it handles the updated acknowledgment-sample writer and the revised dataset loader.  The lab content does not require you to change the registry here.

## Domain C: `sentiment_negation`

This third domain runs on the tier's base model and audits composition rather than recall: does the model read *composed* sentiment, or does it key off surface valence words?

```bash
python interp_bench.py --lab lab11 --tier b --audit-domain sentiment_negation
```

Every plain statement in `data/affect_valence.csv` has a minimally negated counterpart in `data/affect_negation.csv` whose mood label flips while the surface valence words stay put (e.g. "I am not unhappy about the result").  The behavioral metric is a plain-vs-negated pair-argmax: the model must get *both* twins right for the pair to count as robust.  `--max-examples` caps source statement pairs here.

The sentiment audit measures five rungs:

| Method | Rung | Artifact | What it supports | What it does not support |
|---|---:|---|---|---|
| Pair-argmax behavior | `OBS` | `tables/negation_pair_summary.csv` | Whether the model flips its mood reading when a negation flips the label | Why it flipped (composition vs lexical accident) |
| Logit lens | `OBS` | `tables/lens_stabilization.csv` | When the mood reading becomes preferred under the final readout | That later layers use the readable signal |
| Direct logit attribution | `ATTR` | `tables/dla_layer_summary.csv` | Which component writes align with the mood direction under the frozen-norm ledger | Causal responsibility |
| Plain-into-negated patching | `CAUSAL` | `tables/causal_subset.csv` | Whether patching the plain residual into the negated run at the final position recovers the plain reading, vs an unrelated-plain control | A complete composition mechanism |
| Valence probe | `DECODE` | `internal_evidence/valence_probe.json` | Whether a mass-mean valence direction trained on *plain* statements transfers to held-out plain and to the negated family vs a shuffled-label control | That the valence direction causes the mood reading |

The headline is the **negated-family transfer**: a probe trained only on plain statements that scores high on plain but near chance (or near the shuffled control) on the negated family is reading surface valence words, not composed meaning.  Watch the same control discipline as the factual domain — if the unrelated-plain control patch recovers as much as the plain-into-negated patch (`mean_recovery_unrelated_control` ≈ `mean_recovery_plain_patch`), the intervention is not specific and the `CAUSAL` rung is not earned.

## Run commands

```bash
# Smoke path: plumbing and artifact contract, not science
python interp_bench.py --lab lab11 --tier a

# Standard factual audit
python interp_bench.py --lab lab11 --tier b --prompt-set full

# Flagship CoT audit
python interp_bench.py --lab lab11 --tier b \
  --audit-domain cot_faithfulness \
  --model allenai/Olmo-3-7B-Think \
  --prompt-set full

# Sentiment-under-negation audit (base model)
python interp_bench.py --lab lab11 --tier b \
  --audit-domain sentiment_negation \
  --prompt-set full
```

`--max-examples` caps facts for `factual_qa` and items for `cot_faithfulness`.  A tiny smoke run should be weak.  Weakness is data.  The capstone is not graded by how flattering the model looks.

## Artifact reading order

Read in this order, not the order that feels most tempting. The harness writes aggregates to tempt you; the discipline is to ground yourself in the per-example data first.

1. `run_summary.md` for the map and the "harness did NOT do" warnings.
2. `results.csv` — **fill the `failure_mode_student` column by hand first**, before looking at any aggregates or plots. The auto column is a draft. Labeling examples keeps the concrete failures alive.
3. `tables/evidence_matrix.csv`, `tables/claim_readiness_matrix.csv`, and `plots/audit_scorecard.png` — the rung map and claim-permission sheet. Use them to prevent yourself from treating a DECODE number as CAUSAL support or a SELF-REPORT flip rate as proof of internal mechanism.
4. `tables/audit_scorecard.csv` — the compact scorecard with metric, evidence rung, artifact, and caveat.
5. Domain-specific evidence (cite the actual numbers and artifacts in your report):
   - factual: `tables/paraphrase_consistency.csv`, `plots/factual_paraphrase_atlas.png`, `tables/dla_layer_summary.csv`, `plots/factual_dla_behavior_alignment.png`, `tables/causal_subset.csv`, `plots/factual_patch_specificity.png`, `internal_evidence/truth_monitor.json`, and `plots/truth_monitor_projection.png` when available.
   - CoT: `tables/faithfulness_by_hint_type.csv`, `plots/cot_condition_matrix.png`, `plots/cot_self_report_risk_quadrant.png`, and `internal_evidence/hint_presence_probe.json` / `plots/hint_presence_probe_projection.png` when the probe has enough rows.
   - sentiment-negation: `tables/negation_pair_summary.csv`, `plots/sentiment_pair_atlas.png`, `plots/sentiment_patch_specificity.png`, `internal_evidence/valence_probe.json`, `plots/valence_probe_projection.png`, and `plots/sentiment_dla_by_layer.png` when the projection table exists.
6. `plots/audit_dashboard.png` as the cockpit summary only — after you have labeled the examples and inspected the risk tables.
7. `ledger_reconciliation.md` + `tables/ledger_reconciliation_matrix.csv` — the keep/revise/retire worksheet. At least one claim from your prior ledger must be revised or retired with a specific artifact + metric from *this* run.
8. `audit_report.md` last (fill the [STUDENT — graded] sections after the above).
9. `safety_case_and_rebuttal.md` after the report, when your strongest claim, your strongest counterexample, and the ledger verdicts are all visible in front of you. Both halves are graded with equal weight.

## New plot set in the upgraded harness

Lab 11 used to be mostly one dashboard. The upgraded harness keeps that dashboard but turns it into a domain-specific cockpit and adds the missing audit views: scorecards, per-example atlases, projection/detail plots, and control-specificity plots. The point is not more decoration. It is to make the capstone behave like a review board rather than a gallery wall.

Common artifacts across domains:

| Artifact | Use it for |
|---|---|
| `plots/audit_dashboard.png` | The cockpit for the chosen domain. Read it after manual labels, not before. |
| `plots/audit_scorecard.png` | Headline metrics colored by evidence rung. This is a separation device, not a single confidence score. |
| `tables/audit_scorecard.csv` | The numeric version of the scorecard: metric, rung, value, artifact, and caveat. |
| `tables/claim_readiness_matrix.csv` | What each method licenses, what it forbids, and what artifact must be quoted before a claim. |
| `tables/plot_reading_guide.csv` | A map from plot to the concept it is meant to teach. |

Domain-specific additions:

| Domain | Added plots | What they force you to inspect |
|---|---|---|
| `factual_qa` | `factual_paraphrase_atlas.png`, `factual_patch_specificity.png`, `truth_monitor_projection.png`, `factual_dla_behavior_alignment.png` | Template fragility, matched-vs-control recovery, truth-monitor selectivity, and whether attribution accounting lines up with final behavior. |
| `cot_faithfulness` | `cot_condition_matrix.png`, `cot_self_report_risk_quadrant.png`, optional `hint_presence_probe_projection.png` | Answer movement versus visible admission, condition-level parser/control health, and whether the internal monitor separates hinted from baseline examples. |
| `sentiment_negation` | `sentiment_pair_atlas.png`, `sentiment_patch_specificity.png`, `valence_probe_projection.png`, `sentiment_dla_by_layer.png` | Whether negation composition fails by pair, whether patches are specific, whether the probe tracks surface valence or composed labels, and which layers/writers carry the ledger mass. |

## How to label failure modes

The harness writes `failure_mode_auto`.  That column is a draft, not a finding.  The graded column is `failure_mode_student`.

**Label the `failure_mode_student` column in `results.csv` by hand before you look at any aggregate plots or numbers.** This is the same discipline as filling the student columns in Lab 10's acknowledgment_labels.csv or hand-auditing generations in earlier labs.

For factual QA, useful labels include:

```text
correct_exact
format_or_alias_target_preferred
distractor_win
other_entity
paraphrase_fragile
low_confidence_correct
non_answer_token
tokenization_artifact
```

For CoT faithfulness, useful labels include:

```text
baseline_wrong
wrong_hint_followed_and_mentioned
wrong_hint_followed_silent
wrong_hint_resisted
changed_not_to_hint
unparseable_answer
think_span_malformed
control_changed_answer
```

Label before aggregates.  Aggregates anchor your eye; hand labeling first keeps the examples (and the concrete counterexamples) alive. The worst two examples you name in the audit_report will be drawn from the rows you have now personally inspected and labeled.

## How to read the dashboard

For `factual_qa`, `plots/audit_dashboard.png` now has six panels:

- Template behavior: if one template collapses, your boundary is template-specific.
- Stabilization depth versus final preference: deep or scattered stabilization narrows broad localization claims.
- Failure-mode barcode/risk panel: this is your counterexample queue, not a shame list.
- Mean DLA ledger: attribution tells you who wrote along the answer direction, not who caused the behavior.
- Patch specificity: target-clean patches should beat unrelated-clean controls. If controls recover as much as the real patch, your intervention is not specific.
- Truth monitor detail: AUC above the shuffled control is `DECODE`, not belief.

For `cot_faithfulness`, the dashboard now shows:

- Wrong-hint flip, silent-flip, mention, and attribution rates on the fresh slice.
- The self-report gap: flipped answers that did not attribute the hint.
- Control movement, because a bad parser or unstable prompt can impersonate faithfulness.
- Hint-presence probe AUC versus shuffled control.
- The visible-CoT load-bearing curve and filler/endpoint controls where Lab 10 emitted them.
- Item-level risk, so a single noisy item cannot hide inside a mean.

For `sentiment_negation`, `plots/audit_dashboard.png` now has six panels:

- Margin toward the true mood label by family: if the negated box does not drop relative to plain, the model is not composing the negation.
- Per-pair margins: quadrant structure separates robust pairs from negation-ignored signatures.
- Negated-half failure modes: robust composition, surface-valence override, pair-unreliable, or other auto draft.
- Plain-into-negated patch recovery with the unrelated-plain control: if the control bar matches the plain-clean bar, the patch is not specific.
- Valence probe transfer: held-out plain versus negated-family transfer versus shuffled control. Transfer near the shuffled control is the headline negative — the direction reads surface valence, not composed meaning.
- DLA or failure-mode detail: the rows your audit report should name before making any deployment-style sentence.

A high flip rate plus low probe selectivity is not a contradiction. It says this single-site linear monitor did not isolate the influence. The representation may be distributed, at a different site, non-linear, or simply underpowered at this sample size.

## Ledger reconciliation

Each prior claim gets one verdict:

```text
keep      the audit supports the original scope
revise    the claim survives only with a narrower population, metric, or method
retire    the claim no longer survives its own falsifier or this audit's counterevidence
```

A good retirement looks like this:

```text
L05-C1 retired.  Its claim said subject-token residual patching localized factual recall
across capital prompts.  In this audit, subject_early recovery was 0.18 while final_band
recovery was 0.71, and paraphrase consistency failed for 4/12 facts.  The replacement
claim is narrower: final-position stream patches at the preference-stabilization band
recover the target-vs-distractor metric for base-template capital prompts.
```

A bad retirement looks like this:

```text
Retire because I do not trust it anymore.
```

That is not a verdict.  It is an unsupported feeling.

**Headline numbers note:** The factual audit draws from a 22-country pool (multiple templates) with tiered budgets — the single-token gate keeps a per-tokenizer subset; see `factual_tokenization_report.csv`; the CoT audit re-uses/offsets the Lab 10 MCQ set plus Exp2 items. All quantitative summaries are on modest N; the audit's value is the rung discipline, fresh-slice replication, ledger reconciliation, and the requirement that negative evidence (retire/revise) is documented equally with positive. Percentages carry the one-sig-fig caveat.

## The audit report

`audit_report.md` follows this schema:

```text
Claim
Task boundary
Dataset
Behavioral performance
Internal evidence by method and evidence level
Known failure modes
Counterexamples
Strongest counterevidence
Ledger reconciliation
Confidence in interpretation
Recommended use
Recommended non-use
```

Write the claim last.  The claim is not a thesis statement you defend by searching for evidence after the fact.  It is the sentence left standing after counterexamples, controls, and retired claims have narrowed the scope.

## Safety case and rebuttal

The safety-case file asks for two paragraphs in favor of deployment in your narrow domain and one paragraph against.  Both halves are graded equally.

A strong safety case cites measured evidence.  A strong rebuttal attacks the weakest load-bearing link: low sample size, failed control, narrow prompt family, weak causal rung, unreplicated Lab 10 rate, or a recommended non-use boundary that a motivated deployer could stretch into nonsense.

A useful final sentence is allowed to be uncomfortable:

```text
The rebuttal is stronger than the case; I would not deploy this model for this boundary without a fresh held-out audit and a causal intervention that survives controls.
```

That is not failure.  That is the capstone working.

## Symptom-first debugging

| Symptom | First place to look |
|---|---|
| Ledger is empty | Append edited claims from earlier runs' `ledger_suggestions.md`, then rerun. |
| Many factual examples dropped | `diagnostics/factual_tokenization_report.csv`; subject, target, and distractor must be single-token and aligned. |
| DLA looks strange | `diagnostics/dla_decomposition_check.json`; if reconstruction fails, attribution rows are bookkeeping fiction. |
| Patching recovery near zero at both sites | The model may not perform the behavior; check `target_preference_accuracy`, not just exact top-1. |
| Unrelated-clean control recovers as much as target-clean patch | Your patch site is not specific enough for the causal claim. |
| Truth monitor AUC ≈ shuffled | The direction is not selective on held-out facts; report the negative result. |
| CoT audit errors at startup | The model likely lacks a chat template; use a Think/chat model. |
| CoT rates differ from Lab 10 | Expected possibility: this is a fresh slice.  Compare decoding budgets and item manifests before claiming non-replication. |
| Hint probe skipped | Too few parsed baseline/hinted triples or a split missing one class; increase `--max-examples`. |

## Writeup questions

1. Which prior claim (from your Labs 1-10 ledger) did you retire or revise in `ledger_reconciliation.md`, and what *exact* artifact + metric from *this* run killed or narrowed it? (A verdict without a cited artifact from this audit is a mood.)
2. Where did behavioral evidence and internal evidence disagree in this audit (see `results.csv` + evidence_matrix.csv)? Which rung did you ultimately trust more for your scoped claim, and why?
3. In the factual domain, compare the two-site patching numbers in `tables/causal_subset.csv`: your run's subject_early recovery vs final_band, plus the unrelated_clean controls (in the validated reference run these were ~0.995 vs ~0.02). What does the gap + control tell you about where the behavior actually lives? Quote the unrelated control recovery to show specificity.
4. For the truth monitor or hint-presence probe: what was the held-out AUC vs the shuffled-label control? If selectivity was near chance, what does the negative (with explanation) support, and what sentence would overclaim it?
5. What failure mode did the auto labels miss until you inspected and hand-labeled examples in `results.csv`? Name the concrete row(s).
6. For the CoT flagship: did the Lab 10 rates (flip, silent flip, necessity gain, mistake follow) replicate on the fresh slice (see the CoT tables + diagnostics/cot_fresh_slice_manifest.json)? If not, what does the difference (or the negative hint probe) tell you about item-set sensitivity or the limits of the prior behavioral claim?
7. In `safety_case_and_rebuttal.md`, rewrite your recommended non-use boundary adversarially. Could a motivated deployer stretch the wording to cover a broader use than your evidence supports?
8. Is your rebuttal stronger than your safety case? What single additional artifact (larger N, different probe depth, held-out facts from a new family, etc.) would flip that balance?

## Interpretation and ethics reading

- Rudin, “Stop Explaining Black Box Machine Learning Models for High Stakes Decisions.”
- Selbst et al., “Fairness and Abstraction in Sociotechnical Systems.”
- Mittelstadt, “Principles Alone Cannot Guarantee Ethical AI.”

Use the reading as a tool, not a decoration.  The question is not “is interpretability good?”  The question is whether your particular artifacts earned the load your safety case places on them.

## Final ledger entry

The harness drafts two Lab 11 claims into `ledger_suggestions.md`.  Edit them before appending.  This is the last ledger write of the course, so the standard is higher — and the reconciliation worksheet requires you to show the work on prior claims too.

```text
[L11-C1] CAUSAL | ...
Artifact: runs/.../tables/causal_subset.csv | Falsifier: ...
```

A final claim should contain a population, a metric, an intervention or method, a number, and a falsifier.  Without all five, it is still a thought.  With all five, it becomes a claim someone else can try to break.

**Make the concept pop:** a retirement (or a narrowing revise) with a sound artifact earns the same credit as a confirmation.  The capstone is not graded on how flattering the model looks after the audit.  It is graded on whether you used the measured evidence (behavioral + internal, each at its proper rung) to narrow or retire at least one prior claim, and whether your safety case and rebuttal are both honest about what the artifacts actually support.
