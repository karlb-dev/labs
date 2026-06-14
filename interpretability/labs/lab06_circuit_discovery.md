# Lab 6: Circuit Discovery and Validation, the Manual Way

**Evidence level targeted:** causal evidence at circuit scope.

**Prerequisites:** Labs 2, 3, and 5 (plus the residual-stream and self-check habits from Lab 1). Lab 2 gives direct-logit attribution (the cheap screen), Lab 3 gives attention motifs (previous-token and induction patterns), and Lab 5 gives the intervention habit (mean-ablation instead of zero-ablation). Keep the `circuit_card.md` from this lab: Lab 9 will place your hand-built circuit next to an automated attribution graph so you can compare the epistemic standards of the two methods.

## The question

Can you reduce a behavior to a small computational subgraph and say, with measurements attached, that the graph is **faithful**, **complete**, and **minimal**?

Those words are not decorative labels:

| Word | Operational test | Read it as |
|---|---|---|
| faithful | mean-ablate every non-circuit head | the circuit alone preserves the behavior |
| complete | mean-ablate the circuit heads | the circuit is necessary for the behavior |
| minimal | remove each kept head from the final circuit | every kept node earns its rent |

![Faithful, complete, and minimal as one ablation logic: ablate the complement and the behavior must survive (sufficiency); ablate the circuit and the behavior must collapse (necessity); drop each kept node in turn and each must hurt (no hitchhikers). "Ablate" means replace a head's output with its dataset-mean output, not zero.](../figures/lab6_faithful_complete_minimal_via_ablation.png)

The three words are one idea: **what you remove decides what you test.** Faithful and complete are the same experiment run on opposite halves of the model — keep the circuit and mean-ablate everything else (does the circuit *suffice*?), or mean-ablate the circuit and keep everything else (is the circuit *necessary*?). Minimal then audits the survivors: pull each kept head out on its own and confirm it was pulling weight. A circuit can be faithful but not complete (a redundant path still carries the behavior when the circuit is removed), and complete but not minimal (a hitchhiker rode into the final set). Read the scorecard as three independent questions, not one score.

**Make the concept pop (F/C/M + held-out):** In `plots/circuit_scorecard.png` and `faithfulness_completeness_minimality.json` you will see two rows: discovery vs held-out. Faithfulness on held-out is often *higher* than on discovery (the circuit is not over-fit to the exact tokens). Completeness effect is usually larger on held-out. Minimality (worst marginal value in `tables/pruned_circuit.csv`) tells you whether the last head you kept was actually pulling its weight.

The lab is a little circuit courtroom. The cheap screens nominate suspects. Mean-ablation cross-examines them. The circuit card is the verdict, including all awkward caveats.

![Circuit discovery as a courtroom pipeline: cheap screens nominate suspect heads, mean-ablation cross-examines them, greedy pruning removes redundant heads, and the circuit card is the verdict. High-motif heads with no causal drop fall out at the ablation step; real-but-redundant heads fall out at the pruning step.](../figures/lab6_circuit_discovery_courtroom_pipeline.png)

The pipeline's real lesson is epistemic. The cheap screens — direct-logit attribution, induction motif, previous-token motif — only **nominate**. A bright induction score is a hypothesis about a head, not a verdict on it. Membership is earned twice: once by surviving mean-ablation (the head actually moves the metric) and again by surviving greedy pruning (the head is not redundant with another). The off-diagonal points in `screen_vs_causal.png` — high motif but no causal drop — are exactly the decoys this gauntlet exists to reject.

**Make the concept pop:** After your run, open `plots/circuit_discovery_dashboard.png`. It is the courtroom map: scorecard, greedy pruning, screen-vs-causal disagreement, and prompt-level preservation on one page. Then open `tables/circuit_evidence_matrix.csv` beside `plots/candidate_evidence_matrix.png`: every screened head gets one row with its OBS motif scores, ATTR score, CAUSAL drop, pruning status, edge role, and minimality value. This is the antidote to circuit pageantry. A head does not become a circuit node because it is pretty, diagonal, or high-attribution; it becomes a node because it survives intervention and pruning.

**Headline numbers note:** Faithfulness/completeness/minimality are computed on a small set of 8-token repeating prompts (17 discovery+heldout after expansion for robustness, minus any the tokenization gate drops on a given tokenizer). The per-prompt table, held-out generalization, and edge-interaction controls are what make the scoped circuit claim legible; the headline F/C/M ratios on this synthetic task should be read as qualitative + one-significant-figure evidence.

## The task

The behavior is induction completion on fixed-length 8-token repeating patterns:

```text
red blue green red blue green red blue ->  green
```

The distractor is the cycle restart:

```text
red blue green red blue green red blue ->  red
```

The metric is:

```text
logit(target) - logit(distractor)
```

Discovery prompts use one set of vocabulary families. Held-out prompts use fresh families such as metals, compass points, and beasts. Held-out evaluation asks whether the circuit is about the induction pattern or merely about these exact tokens.

The baseline gate matters. A prompt is used for discovery only if the unablated model already prefers the target over the distractor. You cannot trace a circuit for a behavior the model is not doing.

Why induction specifically: the previous-token → induction pair is the cleanest known **mover** circuit. A previous-token head writes "what preceded me" into each position; an induction head uses that to find where the current token appeared before and copies *its* successor forward to the readout. This is the same transport mechanism the Lab 5 component pass landed on (attention-at-last), not the in-place ROME store — so Lab 6 is dissecting a circuit you already saw the shadow of. The edge claim (step 6) tests exactly the previous-token → induction hand-off, and only in the layer order a real forward pass allows.

## Scope: heads-only routing graph

The course outline describes circuit discovery broadly as heads and MLPs. This executable lab deliberately narrows the claim: the validated circuit nodes are **attention heads only**.

MLP layers are still causally ranked and plotted as supporting infrastructure. They are not part of the faithfulness complement. That means the claim is not "this is the whole mechanism in the transformer." The claim is:

```text
For this prompt family and metric, these attention heads form a heads-only routing subgraph that preserves and disrupts the behavior under dataset-mean ablation.
```

That sentence is longer than "we found the circuit," but it has a spine.

### Vocabulary, operationalized

Before the plots and tables, here is what the load-bearing words actually mean in this lab, and where each is measured.

| Term | What it actually means here | Where to read it |
|---|---|---|
| **faithful** | Mean-ablate every *non-circuit* head; the circuit alone preserves the logit gap. Sufficiency. | `circuit_scorecard.png`, `faithfulness_completeness_minimality.json` |
| **complete** | Mean-ablate the *circuit* heads; the behavior must collapse. Necessity (reported as `1 - completeness_ratio`). | `circuit_scorecard.png` |
| **minimal** | Remove each kept head alone; a non-positive marginal means a hitchhiker. | `pruned_circuit.csv`, `minimality_ledger.png` |
| **mean-ablation** | The off switch: replace a head's output with its dataset-mean over the fixed-length prompts — not zero. The circuit is *relative to this off distribution*. | `ablation_manifest.json` |
| **screen vs causal** | Cheap rank (attribution/motif) is a hypothesis generator; causal drop under ablation is the test. Disagreement is the lesson. | `screen_vs_causal.png`, `circuit_evidence_matrix.csv` |
| **edge interaction** | Does a previous-token head's effect shrink once the induction head is ablated? Routed fraction, reportable at 2%, **weak** until 5%. Not a key/value/path claim. | `edge_interactions.csv`, `edge_claim.json` |
| **over-recovery (F > 1.0)** | A non-circuit head was *hurting* the task; mean-ablating it helps, so the circuit-only model beats the full model. Largest where the base behavior is weakest. | `per_prompt_faithfulness.csv` |

## Why mean-ablation is the off switch

Zero-ablating hundreds of heads creates a model that never appears during normal inference. It can prove that your hooks are powerful without proving that the model uses the circuit you named.

This lab uses **dataset-mean ablation**. For each discovery prompt, it captures every head's out-projection input at every position. Because the prompt length is fixed at 8 tokens, it can replace a head's prompt-specific output with the mean output for that same layer, head, and position.

Mean-ablation removes prompt-specific computation while staying closer to the data manifold. The cost is that the circuit is relative to that off distribution. A different off switch can define a different circuit.

The run writes `diagnostics/ablation_manifest.json` so this choice is not trapped in a comment.

## Method spine

1. **Validate the task.** Confirm the prompt length and answer tokenization. Drop discovery prompts the model does not solve.
2. **Screen cheaply.** Rank heads by final-position attribution, induction motif score, and previous-token motif score. Rank MLP layers by direct-logit attribution as support candidates.
3. **Rank causally.** Mean-ablate each screened candidate alone and measure the drop in the logit-difference metric.
4. **Prune greedily.** Start from every screened head with positive causal drop. Repeatedly remove the head whose removal gives the highest remaining faithfulness. Stop before crossing the faithfulness floor, or stop if the candidate set never reaches it.
5. **Earn F/C/M.** Measure faithfulness, completeness, and minimality on discovery and held-out families.
6. **Try one edge.** Test ordered previous-token -> induction pairs by ablation interaction. The source head must be in an earlier layer than the target head. If no ordered pair passes the checks, the lab says no edge was earned. If the best pair is weak, report it as weak rather than inflating it into a crisp path claim.
7. **Write the circuit card.** The card is the deliverable. The plots and tables are evidence for it.

## The edge claim, carefully

The edge test asks whether a previous-token head's effect shrinks when an induction head is already ablated.

```text
interaction = effect(previous head alone) - effect(previous head | induction head already ablated)
```

A positive interaction supports the idea that part of the previous-token head's effect routes through the induction head. In the executable lab, a pair is reportable at 2% routed fraction and labeled **weak** until 5%. This keeps the Olmo 7B target honest: redundancy can make the strongest single edge small without making the interaction uninteresting.

It does **not** show whether the route is through keys, values, queries, residual writes, or another subpath. That stronger claim needs path patching.

The revised code also refuses impossible arrows: a later previous-token head cannot route through an earlier induction head in an ordinary forward computation. If the layer order is wrong, the pair is not eligible for the edge claim.

## Running it

```bash
python interp_bench.py --lab lab6 --tier a
python interp_bench.py --lab lab6 --tier b --prompt-set full
```

Tier A is a plumbing smoke test on `gpt2`. Tier B is the course run. The bench auto-enables eager attention because motif scores need attention patterns.

## First artifact-reading path

**Instrument health + baseline gate first (exactly as in Labs 1-5):**
1. `diagnostics/tokenization_and_baseline.csv` — the model must actually do the induction task on the discovery prompts before you claim a circuit for it.
2. `diagnostics/ablation_manifest.json` — the off distribution (dataset mean, fixed length 8) is part of the claim.

**Then the science (the manual workflow):**
3. `circuit_card.md` — the deliverable. Read this first; everything else is evidence for it.
4. `plots/circuit_discovery_dashboard.png` — the upgraded overview: F/C/M, greedy pruning, cheap-screen disagreement, and prompt-level preservation.
5. `tables/circuit_evidence_matrix.csv` and `plots/candidate_evidence_matrix.png` — the aligned evidence ledger for each screened head: OBS motif, ATTR score, CAUSAL drop, pruning status, edge role, and minimality.
6. `plots/causal_motif_atlas.png` — where screened heads actually bite in layer/head space, with final circuit heads outlined.
7. `plots/screen_vs_causal.png` — the central "cheap screening is a hypothesis generator" plot (off-diagonal points are the lesson).
8. `plots/prune_trajectory.png`, `tables/pruned_circuit.csv`, and `plots/minimality_ledger.png` — how the circuit shrinks, where it stops, and whether each final node earns rent.
9. `plots/circuit_scorecard.png` and `faithfulness_completeness_minimality.json` — F/C/M on discovery vs held-out (the held-out row is the generalization test).
10. `plots/per_prompt_faithfulness.png`, `plots/prompt_failure_scatter.png`, `tables/per_prompt_faithfulness.csv`, and `tables/prompt_failure_modes.csv` — the specific prompts your circuit least explains or over-recovers.
11. `plots/circuit_graph.png` — the visual summary with motif labels, supporting MLPs, and any claimed edge.
12. `plots/edge_interactions.png`, `plots/edge_interaction_map.png`, `tables/edge_claim.json`, and `tables/edge_interactions.csv` — the carefully scoped edge claim or the reason none was earned.
13. `tables/plot_reading_guide.csv` — a compact map from every upgraded plot to the concept it teaches.

## How to read the main plots


### `circuit_discovery_dashboard.png`

This is the new first science plot after the circuit card. It compresses the workflow into four panels: F/C/M, greedy pruning, cheap-screen-vs-causal disagreement, and per-prompt faithfulness. It is a table of contents with numbers, not a substitute for the node-level ledger.

### `candidate_evidence_matrix.png` and `circuit_evidence_matrix.csv`

This is the anti-slogan artifact. Each candidate head gets columns for observation-level motif evidence, attribution evidence, single-head causal drop, final circuit membership, minimality marginal, and any edge role. A bright induction score does not equal a circuit node. A large direct-logit attribution does not equal necessity. A positive single-head causal drop does not mean the node survives redundancy-aware pruning. The final circuit claim only begins after those columns agree enough to survive the pruning rule.

### `causal_motif_atlas.png`

The atlas is a sparse map of causal drops over layer and head index. It makes two things visible at once: where screened heads actually matter, and which screened heads were attractive decoys. The outlined heads are the final circuit. Pale or opposite-signed neighbors are not decoration; they are the evidence that the model has redundant and sometimes anti-helpful routes.

### `prompt_failure_scatter.png`

This plot puts the full-model logit gap on one axis and the circuit-only faithfulness ratio on the other. Weak-denominator prompts, held-out failures, and over-recovery are visible together. This is the plot that prevents a mean score from becoming a carpet under which the awkward examples are swept.

### `minimality_ledger.png`

Minimality is not a vibe. This plot shows how much faithfulness is lost when each kept head is removed. A negative marginal means the pruning rule kept a head that hurts this particular circuit-only behavior. A tiny marginal means the node is a weak rent-payer. Either way, the card should say it.

### `edge_interaction_map.png`

The edge map shows all eligible previous-token → induction interactions as a signed matrix. It helps keep the edge claim in its cage: interaction-granularity evidence, not a key/value/path-patching claim.

### `screen_vs_causal.png`

The left panel plots cheap screen rank against causal drop. The right panel plots attribution magnitude against causal drop. Points above zero mattered under mean-ablation. Points below zero were attractive suspects that did not help the behavior.

A good writeup names at least one disagreement. For example: "Head X had a high induction motif score but little causal drop, so the motif was behavior-adjacent but not necessary under this metric."

### `prune_trajectory.png`

Read right to left. The rightmost point is the full positive-causal screened set. Each step removes one head. The red line is the faithfulness floor.

If the first point is below the floor, the screen did not collect enough of the behavior. That is not a crash. It is an experimental result with a caveat attached.

### `circuit_scorecard.png`

Faithfulness is preservation. Higher is better.

Completeness effect is `1 - completeness_ratio`. Higher means circuit ablation destroyed more of the behavior. A circuit can be faithful but not complete if redundant paths exist. It can be complete but not minimal if extra heads hitchhiked into the final set.

### `per_prompt_faithfulness.png`

This is the anti-cherry-pick plot. The lowest bars become the failure cases in the circuit card. Do not hide them. Ask what they share.

### Worked example: why faithfulness can exceed 1.0

Faithfulness is a ratio of two measured quantities, not a percentage of
something conserved:

```
faithfulness = logit_diff(model with every NON-circuit head mean-ablated)
               ----------------------------------------------------------
               logit_diff(full model)
```

Nothing in that definition caps the numerator. Mean-ablating a head does not
silence it — it replaces its output with its average output over the off
distribution. If some non-circuit head was actively *hurting* the task on a
prompt (an anti-induction head suppressing the repeated token, a sink head
diluting attention), replacing it with its blander average output removes the
interference, and the circuit-only model beats the full model.

Two real rows from the run-4 Tier B table (`tables/per_prompt_faithfulness.csv`,
Olmo-3-7B, 12-head circuit):

| prompt | base_diff | circuit_diff | faithfulness |
|---|---|---|---|
| `red blue green red blue green red blue` | 3.4375 | 3.5000 | 1.018 |
| `dog cat bird dog cat bird dog cat` | 0.8125 | 2.4375 | **3.000** |

The first row is ulp-level over-recovery — read it as 1.0. The second is the
instructive one: the full model is barely doing the task on this prompt
(base logit-diff 0.81, the weakest in the set), so the denominator is small,
and removing the complement heads helps (+1.63 logits). Both effects compound:
**over-recovery is largest exactly where the base behavior is weakest.** That
is why the aggregate F/C/M table uses the ratio of *mean* logit-diffs rather
than the mean of per-prompt ratios — a single weak-denominator prompt would
otherwise dominate the average — and why a per-prompt ratio above 1.0 is
evidence about interference from non-circuit heads, not extra credit for your
circuit. Say which heads you suspect (the candidate table's negative-drop rows
are the place to look), and never average it away silently.

## What the revised code improves

The updated lab makes several evidence-quality changes:

- It writes a combined tokenization and baseline report before any circuit claims.
- It adds the component decomposition self-check, so attention and MLP contribution bookkeeping is verified before screening.
- It records an ablation manifest that defines the off distribution.
- It broadens and documents the screen budgets instead of relying on a tiny fixed candidate set.
- It separates cheap screen rank, attribution score, motif scores, and causal rank in `candidate_components.csv`.
- It prevents overclaiming when the pruned circuit fails the faithfulness floor.
- It labels weak versus strong edge interactions instead of treating every positive interaction as equally crisp.
- It always writes an edge diagnostic, even when no edge is claimed.
- It requires edge direction to respect layer order.
- It adds per-prompt faithfulness artifacts so failure cases are visible rather than prose-only.
- It adds `circuit_discovery_dashboard.png`, `candidate_evidence_matrix.png`, `causal_motif_atlas.png`, `prompt_failure_scatter.png`, `minimality_ledger.png`, and `edge_interaction_map.png` so the circuit claim is read as an evidence ladder rather than a single pretty graph.
- It writes `tables/circuit_evidence_matrix.csv`, `tables/prompt_failure_modes.csv`, and `tables/plot_reading_guide.csv` for downstream notebooks and for the Lab 9 manual-vs-automated comparison.

## Writeup questions

1. What is sufficient? Cite faithfulness and say whether the final circuit passed the floor.
2. What is necessary? Cite completeness ratio and explain whether circuit ablation destroyed the behavior or merely dented it.
3. What is minimal? Cite the worst marginal value in `tables/pruned_circuit.csv`. Did every kept node have a positive marginal value?
4. Where did cheap screening and causal ranking disagree most? Use `tables/circuit_evidence_matrix.csv` and say whether the misleading signal was attribution, induction motif, previous-token motif, or sink behavior.
5. What does the evidence matrix reveal that the circuit graph hides? Name one node whose observational or attribution evidence looked good but whose causal/pruning status was weak.
6. What do the two weakest per-prompt faithfulness cases share?
7. Did the held-out families preserve the result? If held-out prompts were filtered because the base model missed them, say how many ratio-defined prompts remain. If faithfulness is above 1.0, explain why that is over-recovery, not magic.
8. If an edge was claimed, was it weak or strong? What exactly does the interaction fraction license you to say? If no edge was claimed, which requirement failed?
9. Did supporting MLPs change your wording? Say whether intact MLPs look like quiet infrastructure or a major hidden dependency.
10. Map your card onto Machamer, Darden, and Craver's entities-and-activities schema. Mark every activity you named but did not directly test.

## Symptom-first debugging

| Symptom | First place to look |
|---|---|
| most discovery prompts dropped | `diagnostics/tokenization_and_baseline.csv`; the model may not do the task |
| `head_means seq mismatch` | prompt length changed; mean-ablation requires the fixed-length contract |
| starting faithfulness below 0.70 | the screen missed important heads or MLP support is doing more than expected |
| pruning stops immediately | the candidate set is fragile, redundant, or already below the floor |
| completeness ratio stays high | ablating the circuit leaves another path for the behavior |
| minimality worst marginal is negative | at least one kept head hurts faithfulness under this pruning rule |
| no edge claimed | check `tables/edge_interactions.csv`; motif alone is not an edge |

## What goes in the ledger

The ledger should contain two or three `CAUSAL` claims. Each claim must name:

```text
model, prompt population, metric, intervention, circuit scope, artifact, falsifier
```

A good claim is scoped enough to survive contact with a skeptical reader:

```text
For 8-token induction prompts on this model, a heads-only routing circuit of [nodes]
preserves X of the target-vs-distractor logit gap when every non-circuit head is
replaced by its discovery-set mean. This does not claim natural-text induction,
MLP sufficiency, or invariance under zero-ablation.
```

A circuit card without scope is a treasure map with no scale bar: exciting, foldable, and treacherous in the field.

## Reading

You do not need these to run the lab, but they are where its ideas come from:

- Elhage et al., "A Mathematical Framework for Transformer Circuits" (2021) — introduces previous-token and induction heads, the motifs the cheap screen looks for.
- Olsson et al., "In-context Learning and Induction Heads" (2022) — the induction mechanism this lab dissects, and the evidence that it is a real, reused circuit.
- Wang et al., "Interpretability in the Wild: a Circuit for Indirect Object Identification" (2022) — the IOI circuit, and the source of the faithfulness / completeness / minimality criteria used here.
- Conmy et al., "Towards Automated Circuit Discovery for Mechanistic Interpretability" (2023) — the automated counterpart to this manual workflow; the natural point of comparison for Lab 9.
- Machamer, Darden & Craver, "Thinking About Mechanisms" (2000) — the entities-and-activities schema referenced in writeup question 10.
