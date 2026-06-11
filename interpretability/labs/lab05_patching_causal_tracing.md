# Lab 5: Activation Patching and Causal Tracing

**Evidence level targeted:** causality (`CAUSAL`), properly scoped. With the
editing extension: the humbling gap between localizing a fact and changing it.
**Prerequisites:** Labs 1–4 — each planted a question only intervention can
answer (attribution ≠ causation; the indirect-path gap; decodable ≠ used).

## The question

Which activations are *causally responsible* for a behavior — concretely,
where in the forward pass is "The capital of France is → Paris" recovered,
and what happens if you try to change it?

## The method: interchange interventions

Run the corrupt prompt ("The capital of **Germany** is"). Splice in **one**
activation from the clean run ("The capital of **France** is") at one
(layer, position). Measure what fraction of the clean behavior returns:

```text
recovery = (patched_diff − corrupt_diff) / (clean_diff − corrupt_diff)
diff     = logit(" Paris") − logit(" Berlin")   at the final position
```

1.0 = that single activation carried everything the corrupt run was missing.
0.0 = the readout could use none of it. Negative = the patch actively hurt.

### Alignment is the whole game

Clean/corrupt pairs in this lab differ in **exactly one single-token
subject**, verified by a validator that *rejects* misaligned pairs rather
than warning (`diagnostics/tokenization_report.csv` shows its work). The
field's most common silent patching bug is comparing position 3 of one
prompt with position 3 of a differently-tokenized other prompt. You don't
get to meet that bug unarmed here — but you will meet it the day you write
your own pairs, which is why the validator's checks are worth reading.

### The instrument check is new and non-optional

`diagnostics/patch_noop_check.json`: patching a run with **its own** vectors
must be bit-exact identity. If the patch hooks were off by one layer or one
position, every heatmap in this lab would still *render beautifully* — and be
a lie. This is self-check #5 in the bench's collection, and the one whose
failure mode is most photogenic.

## What makes it causal *tracing* rather than a demo

One pair is an anecdote. The lab aggregates over a **dataset** of validated
capital facts (baseline-gated: the model must actually prefer the right
answer cleanly and the wrong one corruptly, by margins — drops are counted,
never silent), then:

- aggregates recovery by **(layer, token role)** — subject, pre-subject,
  post-subject, last;
- confirms the top region under **two paraphrase templates**;
- runs **negative controls**: mismatched-pair patches, wrong-position
  patches, and a low-recovery region re-tested on held-out facts;
- refines the top band with **component-level patching** (attn vs MLP
  outputs — the same verified objects from Labs 2–3).

### Read the curves like this

At layer 0, patching the subject position just substitutes the token
embedding — recovery 1.0 is a **tautology**, not a discovery. The science is
the **handoff**: the layer where subject-position recovery collapses because
the fact has been read out of the subject and is in transit toward the
readout. Meanwhile last-position recovery is near zero early and rises late.
Between those two curves lies the recall-then-readout story that made causal
tracing famous (Meng et al.) — and your localization band, used by the
component pass and the edit, sits just *before* the handoff.

## Running it

```bash
python interp_bench.py --lab lab5 --tier a               # smoke, 6 facts
python interp_bench.py --lab lab5 --tier b --prompt-set full
python interp_bench.py --lab lab5 --tier b --run-edit    # + the edit audit
```

## The extension: the patch made permanent (`--run-edit`)

A rank-one weight edit at one MLP down-projection writes the corrupt fact's
output for the clean fact's key: after editing, "The capital of France is"
should say Berlin — *if* the localized MLP actually carries the recoverable
fact. The lab applies it at your localized layer **and** at an alternative
layer, then audits like the editing literature audits:

| Measure | Question |
|---|---|
| direct success | did the edited prompt flip? |
| paraphrase flips | did the *fact* change, or just the template? |
| neighbors intact | vs the model's own pre-edit answers, not gold |
| fluency logprob | did we lobotomize anything nearby? |

Then the assignment: read Hase et al., *Does Localization Inform Editing?*,
and reconcile your localization map with where the edit actually worked.
**Explaining the tension is success; resolving it is not required** — the
field hasn't either. (Full ROME solves a least-squares problem over a
covariance estimate; our rank-one is deliberately the minimal version whose
every line you can explain. Cite ROME, not this, in anything public.)

In our validation run the tension arrived on schedule, with its mechanism
showing: stream patches in the localized band recover ~67% of the answer —
but a stream patch at layer k carries the **entire accumulated subject
representation** (everything layers 0..k−1 wrote), while the weight edit
changes **one MLP's write**. At 4× dose the edit moved an 11-logit gap by
0.8 logits and had already started breaking neighbors. "The band is causally
sufficient" and "no single layer in the band is individually decisive" are
both true, and conflating stream-level localization with layer-level
editability is precisely the mistake Hase et al. documented. Check
`direct_logit_diff_before/after` in `edit_results.csv` before concluding an
edit "did nothing" — movement without a flip is the most informative outcome.

## First artifact-reading path

1. `plots/patching_heatmap_<fact>.png` — one pair, layer × position, token-labeled.
2. `plots/localization_across_facts.png` — the role curves; find the handoff.
3. `plots/negative_controls.png` — what the matched patch beats.
4. `plots/component_patching.png` — attn vs MLP in the localized band.
5. `tables/facts.csv` — who passed the gate, who didn't, by how much.
6. `results.csv` / `tables/patching_scores.csv` — the long-form grid behind
   every aggregate and heatmap cell.
7. `tables/edit_results.csv` — the localization-meets-editing table.

## Writeup questions

1. Where is the handoff in your run, and how stable is it across paraphrase
   templates? Quote layers and recoveries.
2. Why is subject-position recovery at layer 0 uninformative *for this
   corruption type*? What corruption would make it informative? (Meng et al.
   used one.)
3. State your strongest result as a Woodward-style invariance claim: under
   which interventions, over which prompt population, does the relationship
   hold? Name one intervention you did NOT perform that could break it.
4. Component pass: in your localized band, who carries the fact — attention
   or MLP? At which position? Does that match the ROME story?
5. (Extension) Localization said layer X; the edit worked best at layer Y.
   Write the Hase et al. reconciliation paragraph: what exactly does causal
   tracing measure, that editing success does not?

## Symptom-first debugging

| Symptom | First place to look |
|---|---|
| patch no-op check FAILED | the stream convention broke — do not touch anything else first |
| everything recovers ~0 | baseline gate margins; is `clean_diff − corrupt_diff` tiny? |
| recovery > 1 or < −1 | fine in single cells (logits aren't bounded); endemic = wrong denominator |
| pairs rejected by validator | your subjects tokenize to ≠1 token, or prompts differ at >1 position |
| only 2–3 facts pass the gate on tier a | gpt2 genuinely doesn't know them; that's the gate working |
| edit flips nothing anywhere | check the edit layer is before the handoff; after it, the subject MLP no longer matters |
| a "negative" control beats the matched patch | per-fact rows in `tables/negative_control_scores.csv` — wrong-position patches on small models can carry real signal; scope the claim to the controls that stayed low |

## What goes in the ledger

2–3 claims, all `CAUSAL`, all **scoped**: an intervention name and a prompt
population in every sentence. "Layer 14 stores capitals" is not available at
this evidence level; "patching the subject-position stream at layers 12–14
recovers ≥80% of the answer logit difference across 14 five-token capital
prompts, and mismatched-pair controls recover well under half of that" is
exactly available — and
its falsifier is already named in the drafts.
