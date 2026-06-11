# Lab 5: Activation Patching and Causal Tracing

**Evidence level targeted:** causality (`CAUSAL`), scoped to a prompt population, metric, and intervention. With the optional editing extension, the lab also shows the gap between *localizing* a fact and *changing* it.

**Prerequisites:** Labs 1-4. Lab 2 showed that attribution is not causation. Lab 3 showed that routing and contribution are different. Lab 4 showed that decodable does not mean used. Lab 5 is where the experiment finally reaches into the forward pass and moves something.

## The question

Which activations are causally responsible for a behavior? Concretely: where in the forward pass is the fact

```text
The capital of France is -> Paris
```

recoverable after the prompt has been corrupted to a different country, and what happens if we try to turn that localization into a weight edit?

## The method: interchange interventions

Run a clean prompt and a corrupt prompt:

```text
Clean:     The capital of France is       target:     Paris
Corrupt:   The capital of Germany is      distractor: Berlin
```

Then splice one activation from the clean run into the corrupt run at one stream depth and one token position. Measure how much of the clean behavior returns:

```text
recovery = (patched_diff - corrupt_diff) / (clean_diff - corrupt_diff)
diff     = logit(" Paris") - logit(" Berlin") at the final position
```

A recovery of 1.0 means the patch restored the whole clean-vs-corrupt logit gap. A recovery of 0.0 means the readout could use none of it. A negative value means the patch made the corrupt run even less clean-like.

This is causal evidence because the internal state was changed while the prompt and model weights were otherwise held fixed.

## The load-bearing convention: stream depth is not component layer

The bench names the residual stream this way:

```text
streams[0] = embedding output, before block 0
streams[k] = residual stream after k blocks, also the input to block k
streams[L] = residual stream after all L blocks, before final norm
```

So a patch at `streams[k]` contains everything written by blocks `< k`. If the localized stream depth is 13, the nearest component layer that could have *written* that stream is block 12. The revised lab writes this mapping into:

```text
diagnostics/localization_decision.json
```

Read that file before comparing the residual-stream patch to the component patch or the rank-one edit. Otherwise it is easy to make a one-layer-late claim with a perfectly polished plot, the most elegant wrong turn in the maze.

## Alignment is the whole game

Clean and corrupt pairs must differ in exactly one single-token subject position. The validator rejects a pair if:

- the subject or answer is not a single token;
- the clean and corrupt prompts have different token lengths;
- the prompts differ at any position other than the declared subject position.

The report is written to:

```text
diagnostics/tokenization_report.csv
```

Do not skim it. The field's classic patching bug is comparing position 3 in one prompt to position 3 in a prompt that tokenized differently.

## Instrument checks before science

The lab runs the bench self-checks before measuring recovery:

| Diagnostic | Why it matters |
|---|---|
| `diagnostics/hook_parity.json` | block hooks match the residual-stream convention |
| `diagnostics/logit_lens_self_check.json` | final-depth lens reproduces the model logits |
| `diagnostics/patch_noop_check.json` | patching a run with its own vectors is identity |
| `diagnostics/component_anatomy.json` | attn/MLP contribution hook points are verified, not guessed |
| `diagnostics/dla_decomposition_check.json` | captured components sum to the final pre-norm residual stream |

If any of these fail, stop. A failed self-check does not make the result noisy; it makes the object undefined.

## What makes it causal tracing rather than a demo

One clean/corrupt pair is a demonstration. Causal tracing is the aggregate:

1. Validate a dataset of capital facts.
2. Gate out facts the model does not know, using clean and corrupt logit margins.
3. Patch every stream depth and token position for every kept base-template pair.
4. Aggregate recovery by token role: pre-subject, subject, post-subject, last.
5. Confirm the subject curve on two paraphrase templates.
6. Run negative controls: mismatched-pair patches, wrong-position patches, and a split-heldout low-region check.
7. Refine the localized stream band with component-level patching: attention output versus MLP output.
8. Optionally run a rank-one edit audit.

The output is not just a heatmap. The output is a claim card with a scope, a metric, a control battery, and caveats.

## How to read the main curves

At stream depth 0, patching the subject position mostly substitutes the token embedding. For this corruption type, high subject recovery at depth 0 is a tautology, not a localization result.

The science starts after depth 0:

- The **subject-position curve** tells you where the clean subject representation still causally helps the corrupt run recover the target answer.
- The **handoff** is where subject-position recovery collapses. The fact has been read out of the subject stream or moved into later computation.
- The **last-position curve** usually rises later. That is where the answer becomes directly available to the final readout.
- The **localized stream band** is the last non-tautological subject band before the handoff.
- The **component layers** are mapped from that stream band by subtracting one, because block `k - 1` writes `streams[k]`.

This is the recall-then-readout story in one figure.

## Running it

```bash
python interp_bench.py --lab lab5 --tier a
python interp_bench.py --lab lab5 --tier b --prompt-set full
python interp_bench.py --lab lab5 --tier b --prompt-set full --run-edit
```

Useful knobs:

```bash
python interp_bench.py --lab lab5 --tier b --prompt-set medium
python interp_bench.py --lab lab5 --tier b --prompt-set full --max-examples 12
python interp_bench.py --lab lab5 --tier b --prompt-set full --showcase france
```

`--prompt-set small|medium|full` now controls the built-in fact count. A custom `.csv` or `.json` file can be passed as `--prompt-set` if it contains `fact_id`, `subject`, and `target` fields.

## Main artifacts

Read them in this order:

1. `causal_trace_card.md` - the deliverable card: scope, localization, controls, component result, edit result if run.
2. `diagnostics/localization_decision.json` - the handoff rule and stream-depth-to-component-layer mapping.
3. `plots/localization_across_facts.png` - the subject-vs-last causal tracing story.
4. `plots/patching_heatmap_<fact>.png` - one pair, layer by position, token-labeled.
5. `tables/facts.csv` - which pairs passed the baseline gate and why others dropped.
6. `tables/patching_scores.csv` and `results.csv` - the long-form grid behind every cell.
7. `tables/per_fact_top_patch.csv` and `plots/per_fact_top_patch.png` - which facts drive the average.
8. `tables/paraphrase_summary.csv` and `tables/paraphrase_consistency.csv` - whether the localized band survives templates.
9. `tables/negative_control_scores.csv` and `plots/negative_controls.png` - specificity checks.
10. `tables/component_patching.csv`, `tables/component_summary.csv`, and `plots/component_patching.png` - attention versus MLP refinement.
11. `tables/edit_results.csv` - only if `--run-edit` was passed.

## The extension: the patch made permanent

`--run-edit` applies a deliberately minimal rank-one edit to one MLP down-projection. It takes the clean subject's MLP key and shifts the MLP output toward the corrupt subject's output:

```text
clean key:    France at the localized component layer
new value:    Germany-like MLP output at that same position
intended edit: France -> Berlin
```

The edit audit asks:

| Measure | Question |
|---|---|
| direct success | did the original prompt flip to the distractor? |
| logit movement | did the target-vs-distractor gap move even without a flip? |
| paraphrase flips | did the fact change, or just the base template? |
| neighbors intact | did nearby capital facts keep the model's own pre-edit top-1 answer? |
| fluency logprob | did the edit damage unrelated text modeling? |

The localized edit layer is chosen from the mapped component layer, not from the stream depth directly. An alternative layer is also tested. That comparison is the point of the extension: causal tracing can identify where a clean activation is sufficient, while editing asks whether a small weight change at one module can reproduce that activation change robustly.

A movement without a flip is not a failed artifact. It is often the most informative outcome. It says the edit touched the right direction but did not dominate the distributed computation.

## Writeup questions

1. Where is the handoff in your run? Quote the subject peak, threshold, handoff depth, and localized stream band from `localization_decision.json`.
2. Why is subject-position recovery at stream depth 0 uninformative for this corruption type? What corruption type would make early subject patches less tautological?
3. Do the paraphrase templates preserve the localized band? Quote `paraphrase_summary.csv`.
4. Which component-role cell is strongest in `component_summary.csv`: MLP at subject, attention at subject, MLP at last, or attention at last? Does it support or complicate the ROME-style story?
5. Do negative controls stay below the matched patch? Identify the strongest control and decide whether your claim needs to be narrowed.
6. State your strongest result as a Woodward-style invariance claim: under which intervention, over which prompt population, and for which metric does the relationship hold?
7. Extension: did the localized layer or alternative layer edit better? Explain the result without assuming localization should predict editability.

## Symptom-first debugging

| Symptom | First place to look |
|---|---|
| `patch_noop_check` failed | stream convention or patch hook is broken; ignore every heatmap until fixed |
| many rejected pairs | `diagnostics/tokenization_report.csv`; subjects or answers are not single tokens |
| only a few facts pass the gate | `tables/facts.csv`; the model may not know the facts under this template |
| recovery is mostly zero | check `clean_diff - corrupt_diff`; the denominator may be too small or the corrupt pair may not oppose the target |
| recovery is often above 1 or below -1 | single cells can do this; widespread extremes suggest denominator or alignment trouble |
| paraphrases localize elsewhere | scope the claim to the base template or inspect whether the subject position changed meaningfully |
| a negative control matches the real patch | do not claim specificity; inspect the per-fact rows and narrow the intervention |
| edit flips nothing | inspect `movement_toward_distractor` before declaring failure; the stream patch may be distributed across many writes |
| edit breaks neighbors | the edit is not specific enough; cite spillover as counterevidence |

## What goes in the claim ledger

Write 2-3 `CAUSAL` claims. Each claim needs four pieces:

```text
Intervention: clean subject-position stream patch at depth k
Population: validated single-token capital prompts under template T
Metric: recovery of target-vs-distractor final-position logit difference
Falsifier: a control, paraphrase, or broader population that would break the claim
```

Available claim:

```text
Patching the clean subject-position residual stream at depth k recovers X% of the clean logit gap across N validated capital prompts, while mismatched-pair controls recover Y%.
```

Not available:

```text
Layer k stores capitals.
```

The second sentence is tempting because it is short. It is also wrong-shaped. Lab 5 earns causal claims about interventions, not metaphysical claims about storage jars in the transformer attic.
