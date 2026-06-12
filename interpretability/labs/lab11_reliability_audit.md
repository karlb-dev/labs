# Lab 11: Mechanistic Reliability Audit

**Evidence level targeted:** integration.  Every measured claim keeps its own rung: `OBS`, `ATTR`, `DECODE`, `CAUSAL`, or `SELF-REPORT`.  The audit earns authority by refusing to blend those rungs into one undifferentiated confidence score.

**Prerequisites:** the whole course, especially your `claim_ledger.md`.  The capstone is built on the ledger.  If your ledger is empty, the harness will still run, but the central assignment will be missing its spine: there are no claims to keep, narrow, or retire.

## Core question

Given behavioral evidence and internal evidence, where should we trust this model less, and what may we responsibly say?

## Why this lab exists

The previous labs taught instruments.  This lab teaches judgment.

A logit-lens curve does not deploy.  A probe AUC does not deploy.  An attribution graph does not deploy.  What deploys is a sentence like this:

```text
For task boundary B, model M is reliable enough for use U, except under conditions N,
because artifacts A1-A4 support claims C1-C3 and fail to support claim C4.
```

That sentence is dangerous if its evidence rungs get scrambled.  A strong `DECODE` result can make a weak safety case sound causal.  A causal patch on six examples can make a narrow mechanism sound universal.  A fluent chain of thought can make self-report sound like an audit log.  Lab 11 is the place where those risks are named, weighed, and kept out of the final claim.

The audit also cashes out the claim ledger.  All semester you drafted claims with falsifiers.  Now you reconcile them: **keep**, **revise**, or **retire**.  A retirement with a good artifact earns full credit.  A semester with no retired or narrowed claims usually means the falsifiers were not operational.

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
plots/audit_dashboard.png               compact audit dashboard
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
```

`--max-examples` caps facts for `factual_qa` and items for `cot_faithfulness`.  A tiny smoke run should be weak.  Weakness is data.  The capstone is not graded by how flattering the model looks.

## Artifact reading order

Read in this order, not the order that feels most tempting:

1. `run_summary.md` for the map.
2. `results.csv`, and fill `failure_mode_student` before reading aggregate plots.
3. `tables/evidence_matrix.csv`, because it prevents rung-blending.
4. Domain-specific evidence:
   - `tables/paraphrase_consistency.csv`
   - `tables/dla_layer_summary.csv`
   - `tables/causal_subset.csv`
   - `internal_evidence/truth_monitor.json`
   - or for CoT, `tables/faithfulness_by_hint_type.csv` and `internal_evidence/hint_presence_probe.json`
5. `plots/audit_dashboard.png` as a visual summary, not as proof.
6. `ledger_reconciliation.md` and `tables/ledger_reconciliation_matrix.csv`.
7. `audit_report.md` last.
8. `safety_case_and_rebuttal.md` after the report, when your strongest claim and strongest counterevidence are both visible.

## How to label failure modes

The harness writes `failure_mode_auto`.  That column is a draft, not a finding.  The graded column is `failure_mode_student`.

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

Label before aggregates.  Aggregates anchor your eye; hand labeling first keeps the examples alive.

## How to read the dashboard

For `factual_qa`, `plots/audit_dashboard.png` has four panels:

- Template behavior: if one template collapses, your boundary is template-specific.
- Stabilization depth versus final preference: deep or scattered stabilization means the readout is not stable enough to make broad localization claims.
- Residual patch recovery: target-clean patches should beat unrelated-clean controls.  If controls recover as much as the real patch, your intervention is not specific.
- Truth monitor AUC: AUC above the shuffled control is `DECODE`, not belief.

For `cot_faithfulness`, the dashboard shows:

- Wrong-hint flip and silent-flip rates on the fresh slice.
- Hint-presence probe AUC versus shuffled control.

A high flip rate plus low probe selectivity is not a contradiction.  It says this single-site linear monitor did not isolate the influence.  The representation may be distributed, at a different site, non-linear, or simply underpowered at this sample size.

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

1. Which prior claim did you retire or revise, and what exact artifact killed or narrowed it?
2. Where did behavioral and internal evidence disagree?  Which did you trust more, and why?
3. What is your strongest `CAUSAL` evidence?  What is its scope boundary?
4. What is your strongest `DECODE` evidence?  What sentence would be overclaiming it?
5. What failure mode did the auto labels miss until you inspected examples by hand?
6. For the CoT flagship: did Lab 10 replicate on the fresh slice?  If not, name whether the likely cause is sampling, decoding budget, item domain, hint phrasing, or parser fragility.
7. Rewrite your recommended non-use adversarially.  Could a motivated deployer exploit the wording?
8. Is your rebuttal stronger than your safety case?  What evidence would change that balance?

## Interpretation and ethics reading

- Rudin, “Stop Explaining Black Box Machine Learning Models for High Stakes Decisions.”
- Selbst et al., “Fairness and Abstraction in Sociotechnical Systems.”
- Mittelstadt, “Principles Alone Cannot Guarantee Ethical AI.”

Use the reading as a tool, not a decoration.  The question is not “is interpretability good?”  The question is whether your particular artifacts earned the load your safety case places on them.

## Final ledger entry

The harness drafts two Lab 11 claims into `ledger_suggestions.md`.  Edit them before appending.  This is the last ledger write of the course, so the standard is higher:

```text
[L11-C1] CAUSAL | ...
Artifact: runs/.../tables/causal_subset.csv | Falsifier: ...
```

A final claim should contain a population, a metric, an intervention or method, a number, and a falsifier.  Without all five, it is still a thought.  With all five, it becomes a claim someone else can try to break.
