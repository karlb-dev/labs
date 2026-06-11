# Lab 11: Mechanistic Reliability Audit (capstone)

**Evidence level targeted:** integration — every claim in the final report
carries its rung on the ladder, and the report's authority comes from the
rungs being kept distinct.
**Prerequisites:** the whole course, and concretely your `claim_ledger.md`.
The audit is built ON the ledger; if yours is empty, go back to your runs'
`ledger_suggestions.md` files, edit the claims you would defend, and append
them first. Auditing an empty ledger is reviewing a book you haven't written.

## The question

Given behavioral evidence and internal evidence, where should we trust this
model less — and what may we responsibly say?

## Why this matters

Every previous lab taught a method; this one teaches the job. Nobody deploys
a logit-lens plot. What gets deployed is a *judgment* — "this model can be
trusted to do X within boundary Y, and here is the evidence" — and the
skill that judgment requires is not running methods but **integrating
evidence of different strengths without letting the strong borrow authority
from the weak**. A probe AUC is not a causal result; a beautiful attribution
graph is not an invariance; a fluent CoT is not a faithful one. You have
spent ten labs learning each of those distinctions one at a time. The audit
is where they must all hold simultaneously, in one document, under the eyes
of a skeptical reviewer you write yourself.

The capstone is also where the claim ledger pays off. You have been
registering claims with falsifiers all semester. Now you reconcile them:
keep, revise, or retire, each with a reason citing an artifact. **Retirement
is graded as positively as confirmation** — a semester of perfect claims
means your falsifier columns were never honest. The discipline of writing
"the evidence does not support X" as a finding rather than a failure is the
single most transferable thing this course can give you.

## The harness and the contract

Freedom lives in the domain choice and analysis depth; the **output contract
is rigid** so audits are comparable across students:

```text
audit_report.md               the fixed schema (claim, boundary, behavioral
                              performance, internal evidence BY METHOD with
                              evidence level, failure modes, counterexamples,
                              strongest counterevidence, ledger reconciliation,
                              confidence, recommended use and NON-use)
ledger_reconciliation.md      keep / revise / retire, per claim, with reasons
safety_case_and_rebuttal.md   two paragraphs for, one against — graded equally
results.csv                   per-example: behavior, confidence proxy, lens
                              depth, DLA summary, failure_mode columns
tables/ + internal_evidence/  the measured rungs
```

The harness assembles every measured number and cites it by artifact.
Sections marked `[STUDENT — graded]` are prompts, not placeholders to
delete: an audit submitted with the markers still in place is not an audit.
The required per-example analysis from the course outline — answer +
confidence proxy, lens stabilization, DLA summary, a causal intervention on
a subset, one additional internal method, a failure-mode label — is all
there; the failure-mode `_student` column and every verdict are yours.

## Domain A — `factual_qa` (default)

Factual recall under paraphrase, continuing Lab 5's dataset on the course
base model. Per example (12 facts × 3 templates): answer, p(target),
logit-diff against a cyclic-partner distractor, confidence margin, the
logit-lens **stabilization depth** (first depth where the target goes top-1
and stays), and a per-layer DLA summary (top layer, attn-vs-MLP split,
Lab 2's frozen-norm linearization). Then two measured rungs above
observation:

- **Causal subset:** clean→corrupt residual patches at TWO sites — the
  early subject token and the final position of the stabilization band —
  because Lab 5 taught that recall and readout localize differently, and an
  audit that patches only one site would be quietly assuming its answer.
- **Truth monitor (DECODE):** true/false statements about the same facts,
  projected onto a truth direction — Lab 4's saved artifact when it fits
  this model, otherwise a mass-mean direction trained on half the facts —
  scored on held-out facts against a **shuffled-label control**.

## Domain B — `cot_faithfulness` (recommended flagship)

Continues Lab 10 on a **fresh item slice** (the harness offsets the
sampling stride so you cannot audit the items that formed your beliefs) with
a reasoning model:

```bash
python interp_bench.py --lab lab11 --audit-domain cot_faithfulness \
  --model allenai/Olmo-3-7B-Think --tier b
```

It reuses Lab 10's machinery verbatim — hint injection with both controls,
the necessity curve, add-mistake, filler — which makes the audit a
*replication*: do Lab 10's rates hold out of sample? And it adds the
mechanistic method Lab 10 left as its ambitious extension: a **hint-presence
probe** on activations at the answer-emission position (mass-mean,
family-split by item, shuffled control). If hint presence is decodable at
answer time even on items whose CoT never mentions the hint, you have
connected Lab 4's decodability machinery to Lab 10's behavioral finding —
the influence is in there; the text just doesn't say so.

## Running it

```bash
python interp_bench.py --lab lab11 --tier a                  # gpt2 smoke, factual_qa
python interp_bench.py --lab lab11 --tier b                   # Olmo base, factual_qa
python interp_bench.py --lab lab11 --tier b \
  --audit-domain cot_faithfulness --model allenai/Olmo-3-7B-Think   # flagship
```

`--max-examples` caps facts (domain A) or items (domain B). The smoke tier
exists to prove plumbing; its audit of gpt2 will honestly report a weak
model (low accuracy, inconsistent paraphrases) — which is itself a correct
audit outcome.

## The order of work (this matters)

1. Run the harness. Read `run_summary.md` only.
2. Hand-label `failure_mode_student` in `results.csv`, example by example,
   *before* reading the aggregate tables — the aggregates will anchor you.
3. Read `tables/` and `internal_evidence/`. For each number, say aloud which
   rung it is on. If you catch yourself saying "the model knows" about a
   probe AUC, go reread Lab 4.
4. Fill in `ledger_reconciliation.md`. At least one claim revised or
   retired, with the artifact that did it.
5. Write the audit report's student sections — claim LAST, after the
   counterevidence section, never before.
6. Write the safety case, then the rebuttal. If the rebuttal is stronger,
   say so in the final line; that sentence is worth more than the case.

## Writeup questions

1. Which earlier claim did you retire, and what killed it? Quote the claim's
   own falsifier column — did it predict its killer?
2. Where do behavioral and internal evidence disagree, and which did you
   trust? (Example shape: paraphrase-inconsistent behavior with a stable
   stabilization depth, or high monitor AUC on a fact the model gets wrong.)
3. Your recommended non-use: would a motivated deployer find the boundary
   legible, or exploitable? Rewrite it once adversarially.
4. For the flagship: did Lab 10's rates replicate on the fresh slice? If
   they moved, is that sampling noise or scope you over-claimed?
5. The rebuttal you wrote against your own safety case — is it stronger
   than the case? What evidence, at what rung, would change the balance?

## Symptom-first debugging

| Symptom | First place to look |
|---|---|
| "ledger is empty" warning in the reconciliation file | append your edited claims from each run's `ledger_suggestions.md` first |
| causal recovery near zero at both sites | the model fails these facts (check accuracy) — patching can't recover a behavior that isn't there |
| monitor AUC ≈ shuffled AUC | the direction doesn't transfer to this model/layer — the harness falls back to mass-mean; if that also fails, the statements may be too easy (both versions plausible) |
| cot domain errors at startup | base model has no chat template — pass a think model via `--model` |
| flagship rates differ wildly from Lab 10 | different item slice (by design) AND different budget — check `decoding_pins.json` in both runs before claiming non-replication |

## Interpretation & ethics — the safety case

**Reading:** Rudin, "Stop Explaining Black Box Machine Learning Models for
High Stakes Decisions"; Selbst et al., "Fairness and Abstraction in
Sociotechnical Systems"; Mittelstadt, "Principles Alone Cannot Guarantee
Ethical AI."

The writing prompt is built into the deliverable
(`safety_case_and_rebuttal.md`): two paragraphs of internal evidence for a
hypothetical deployment in your domain, then the one-paragraph rebuttal a
skeptical reviewer would file. Both halves graded, equally. The rebuttal
written as a strawman scores as if the case had no rebuttal at all —
Rudin's argument is precisely that interpretability artifacts can launder
confidence they did not earn, and the only countermeasure this course knows
is making you attack your own case at full strength.

## What goes in the ledger

The audit's own 2–3 claims (the harness drafts them with measured numbers),
plus the reconciliation verdicts on every prior claim. This is the last
write to the ledger; what lands here is the dossier you defend.

## Reading

- Rudin (2019), Selbst et al. (2019), Mittelstadt (2019) — the ethics core.
- Your own `claim_ledger.md` — the primary source for this lab.
