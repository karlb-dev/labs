# Lab 3: Attention Routing, Induction, and What Heads Actually Do

**Evidence levels targeted:** `OBS` -> `ATTR` -> `CAUSAL`, with the scope of each claim kept explicit.
**Prerequisites:** Labs 1-2. Lab 2 told you which attention *layers* pushed on an answer direction. Lab 3 cracks those bars open into heads.

## The question

Which positions are routed where, and when does that routing matter for the output?

Those are not the same question. A head can stare at the right token and write nothing useful. Another can have an unimpressive-looking attention pattern but contribute strongly through its value/output channel. This lab is built to cure heatmap astrology: you will measure routing, writing, and causal effect separately, then decide which claim each measurement actually licenses.

A beautiful diagonal heatmap on a synthetic repeat is not evidence that the head “does induction” for the model’s answer until it also has non-zero attribution and a measurable intervention effect. The upgraded version of this lab makes that separation hard to miss: every important plot now says which rung of evidence it belongs to.

## New in the visualization upgrade

The older lab already had the right scientific spine: motif maps, per-head attribution, final-position versus all-position ablation, and attribution-versus-ablation. The upgrade adds a clearer plot grammar and several synthesis artifacts:

| New or upgraded artifact | What it teaches |
|---|---|
| `plots/routing_evidence_dashboard.png` | A one-screen map of motif labels, baseline behavior, causal scope, and natural-transfer audit. |
| `tables/head_evidence_matrix.csv` + `plots/head_evidence_matrix.png` | One row per important head, with OBS, ATTR, CAUSAL, transfer, and control columns lined up. |
| `plots/phase_motif_atlas.png` + `tables/phase_motif_summary.csv` | Where motif labels and answer-direction writes concentrate over early, middle, late, and final depth phases. |
| `plots/motif_transfer_scatter.png` | Whether synthetic/cycle induction scores survive on natural repeated phrases. |
| `plots/ablation_scope_heatmap.png` | Sparse causal atlas of final-position effect, all-position effect, and their gap. |
| `tables/routing_attr_causal_disagreements.csv` | Heads where routing, attribution, and intervention disagree. These are often the best teaching examples. |
| `tables/plot_reading_guide.csv` | A map from artifact to evidence rung and concept. |

The existing plots are also upgraded: `motif_maps.png` now includes a signed attribution panel and the transparent label-rule winner; `attention_heads_<showcase>.png` overlays previous-token, induction, and sink target cells directly on the heatmaps; `head_attribution_by_layer.png` now shows signed head writers plus layer-level cancellation; the ablation plots now emphasize indirect gaps and sign disagreements.

## The evidence ladder for a head

| Evidence rung | Measurement | Claim it can support | What it cannot support alone |
|---|---|---|---|
| `OBS` | Attention pattern and motif score | “This head routes to previous-token / induction / first-token positions on this prompt distribution.” | That the head matters for the answer. |
| `ATTR` | Head output projected through `W_O`, scored against the target-minus-distractor direction | “This head’s final-position write points toward or away from the answer under a frozen-norm approximation.” | That removing the head changes behavior. |
| `CAUSAL` | Scoped head ablation | “Under this intervention and prompt set, removing this head changes the logit gap by X.” | A full circuit edge. The all-position effect still bundles many indirect paths. |

Keep the rungs separate in your writeup. The tempting wrong sentence is: “the attention head explains the answer.” A careful sentence looks more like:

```text
On this prompt family, LxHy routes to induction targets with score S (OBS),
writes A logit-diff units toward the answer at the final position (ATTR),
and changes the logit gap by Z under all-position zero-ablation (CAUSAL).
The all-position minus final-position gap suggests indirect composition,
but does not localize an edge.
```

## What exactly is a head here?

A head has two visible sides:

1. **Routing:** the attention matrix. Rows are query positions, columns are key positions, and each row says where that head reads from.
2. **Writing:** the head’s slice of the attention output-projection input, mapped through that head’s columns of `W_O`. That vector is what the head contributes before shared projection bias and any architecture-specific post-attention norm.

The bench verifies the per-head decomposition before any science: the per-head pieces plus shared projection bias must reconstruct the block attention output. This lab also runs the Lab 2 component decomposition check, because head attribution inherits the direct-logit-attribution ledger. If the accounting fails, the heatmaps become decorative fog.

### Frozen norms, now explicit

On GPT-2-style pre-norm models, the raw attention-module output is the residual-stream contribution. On Olmo-style post-norm models, the attention output is normalized before it is added to the stream. The code handles both RMSNorm and centered LayerNorm in the frozen local linearization used for head attribution.

The artifact `diagnostics/head_attribution_accounting.csv` compares summed head scores against the whole attention-block score. Shared bias and frozen-norm residuals are shown rather than quietly assigned to a convenient head-shaped bucket.

## The motifs

You will score three named patterns:

| Motif | Definition | What to watch for |
|---|---|---|
| previous-token head | query position `q` attends to `q-1` | Often a component of induction rather than a direct final-token writer. |
| induction head | when token at `q` occurred earlier at `j`, attend to token `j+1`, requiring `j+1 < q` | The target rule excludes adjacent-repeat self-targeting. |
| first-token sink | query positions attend to position 0 | A resting pattern. It may matter or may be inert; check attribution and ablation. |

The label rule is intentionally simple: content motifs above `0.35` win, with induction and previous-token competing directly; first-token sink is a fallback above `0.5`; high-entropy heads become `diffuse`; the rest are `other`. This rule is not sacred. Finding where it fails is part of the lab.

## Prompt design

Synthetic prompts use non-obvious cycles such as:

```text
B F Q B F Q B F        -> Q vs B
red blue green red ... -> green vs red
```

They avoid alphabetical and arithmetic priors, where the “induction” answer could be produced by world knowledge about sequences rather than by copying. Natural prompts test whether the same heads still route on repeated phrases:

```text
Marcus went to the lab. Olivia went to the -> lab vs store
```

Control prompts have no intended repeated-token structure. The lab writes `diagnostics/prompt_motif_coverage.csv` so you can see exactly how the tokenizer turns each prompt into induction-target positions. If a control prompt accidentally has repeated token IDs, the microscope tattles.

Synthetic labels are prompt-set-specific until proven otherwise. The natural-transfer audit in `tables/natural_confirmation.csv` and `plots/motif_transfer_scatter.png` asks whether a toy-cycle induction label survives on natural repeated phrases.

## The causal intervention

The ablation intervention zeroes one head’s **pre-output-projection slice** before `W_O`. It is run in two scopes:

| Scope | What is zeroed | Interpretation |
|---|---|---|
| `final_pos` | only the head’s final query-position slice | Direct path. This is the scope most comparable to final-position attribution. |
| `all_pos` | the head’s slice at every position | Direct path plus earlier-position writes that later heads can read. |

A previous-token head with tiny `final_pos` effect but large `all_pos` effect has an indirect-path signature: it probably matters because of what it wrote at earlier positions, not because its own final-token output pushed the answer directly.

This is zero-ablation, not mean-ablation. It is a causal stress test, but it can move the model off distribution. The gap between final_pos and all_pos is the composition signal: earlier writes that later heads can read.

## Running it

**Headline numbers note:** Motif scores, attribution, and ablation effects are measured on a small curated prompt set. The dissociation between routing, contribution, and causal role is the core lesson; specific head-count percentages deserve one-significant-figure treatment plus the motif-coverage and control diagnostics.

```bash
python interp_bench.py --lab lab3 --tier a
python interp_bench.py --lab lab3 --tier b --prompt-set full --topk 10
```

The bench auto-sets eager attention for this lab. Attention patterns are not reliably returned under SDPA/flash implementations in current Transformers, so do not override this unless you are intentionally debugging instrumentation.

Useful switches:

```bash
# ablate more or fewer induction candidates; 0 skips causal ablations
python interp_bench.py --lab lab3 --tier b --prompt-set full --ablate-top 5

# use a custom JSON prompt file with PatternPrompt fields
python interp_bench.py --lab lab3 --tier b --prompt-set path/to/lab3_prompts.json

# feature a specific prompt in the attention heatmap panel
python interp_bench.py --lab lab3 --tier b --prompt-set full --showcase synth_letters
```

## First artifact-reading path

**Instrument health first. Self-checks must be green, exactly as in Labs 1-2:**

1. `diagnostics/head_decomposition_check.json` + `diagnostics/head_attribution_accounting.csv`: per-head slices must reconstruct the block attention output well enough for interpretation.
2. `diagnostics/prompt_motif_coverage.csv` + `diagnostics/control_induction_audit.csv`: do the prompts actually contain the motifs the labels assume? Do controls accidentally trigger high induction scores?

**Then the science, rung by rung:**

1. `attention_routing_card.md`: the compact verdict and non-claims.
2. `plots/routing_evidence_dashboard.png`: labels, baseline behavior, causal scope, and transfer audit in one screen.
3. `tables/head_evidence_matrix.csv` + `plots/head_evidence_matrix.png`: OBS, ATTR, CAUSAL, transfer, and controls per head.
4. `plots/motif_maps.png` + `tables/head_table.csv`: OBS pattern atlas, signed attribution panel, and label-rule winner.
5. `plots/attention_heads_<showcase>.png` + `tables/example_head_scores.csv`: what selected heads read on real tokens, with motif-target overlays.
6. `plots/motif_transfer_scatter.png` + `tables/natural_confirmation.csv`: does toy induction survive on natural repeated phrases?
7. `plots/head_attribution_by_layer.png`: which heads write toward or against the answer direction, and how layer-level cancellation hides them.
8. `plots/head_attribution_vs_ablation.png`: frozen attribution versus direct-path causal effect.
9. `plots/direct_vs_indirect_effect.png`, `plots/ablation_scope_heatmap.png`, and `plots/ablation_effect_by_head.png`: direct-path versus all-position causal scope.
10. `plots/routing_to_causality.png`: whether induction-looking routing predicts all-position effect.
11. `tables/routing_attr_causal_disagreements.csv`: the most instructive disagreements. Read these before drafting claims.

## How to read the upgraded plots

`routing_evidence_dashboard.png` is the tour guide. It asks: how many heads get motif labels, did the model behavior give the ablation something to explain, do final-position and all-position effects differ, and do toy induction scores transfer to natural text?

`head_evidence_matrix.png` is the lab’s central synthesis plot. The left block is observational and transfer evidence. The right block is signed attribution and intervention evidence. A strong row across every block is rare. A row with only one bright block is usually a caveat generator.

`motif_maps.png` is observational plus an attribution overlay. It tells you where patterns live and whether those same heads write toward the answer direction. It still does not prove causal use.

`attention_heads_<showcase>.png` now overlays motif target cells: previous-token cells, induction-target cells, and first-token sink cells. The overlay makes it obvious whether a heatmap is bright in the place the motif definition actually says it should be bright.

`head_attribution_by_layer.png` resolves attention into signed head writers. The pale bars show positive and negative mass, while the line shows the net layer contribution. This makes cancellation visible.

`head_attribution_vs_ablation.png` compares a frozen final-position ledger score to a live final-position intervention. Points far from the diagonal are not automatically bugs. They are the difference between arithmetic attribution and live downstream computation.

`direct_vs_indirect_effect.png` and `ablation_scope_heatmap.png` compare direct-path ablation to all-position ablation. A large gap suggests indirect composition through earlier positions. It does not identify the downstream edge.

`routing_to_causality.png` is the overclaiming antidote. An induction score on the x-axis and an ablation effect on the y-axis ask whether a head that routes like induction also matters for this behavior. The best answer is allowed to be “sometimes.”

## Writeup questions

1. Pick the strongest induction-labeled head. Trace its evidence across `OBS`, `ATTR`, and `CAUSAL`. Which rung is strongest? Which rung is weakest?
2. Find a first-token sink head. Does it have high attribution or causal effect? Explain why “sink” should not be used as a synonym for “irrelevant.”
3. Report one previous-token head’s `final_pos` and `all_pos` effects. What exactly does their gap measure? What would a patching experiment need to show to turn that into an edge claim?
4. Use `head_evidence_matrix.csv` to find a head with strong routing and weak attribution. What claim can you make, and what claim must you refuse?
5. Compare `head_table.csv`, `head_scores_by_category.csv`, and `motif_transfer_scatter.png`. Did any head’s label depend strongly on synthetic prompts but fail on natural prompts?
6. Did any control prompt produce above-bar induction scores? Use `diagnostics/control_induction_audit.csv` and `prompt_motif_coverage.csv` to decide whether the prompt or the label rule is to blame.
7. Does attribution predict direct-path ablation in this run? Report both the per-prompt and by-head Spearman correlations. Which one is the cleaner statistic, and why?
8. Choose one row from `routing_attr_causal_disagreements.csv`. Write the most careful claim you can make about it without smuggling in stronger evidence.

## Symptom-first debugging

| Symptom | First place to look |
|---|---|
| “model returned no attention patterns” | You overrode eager attention; rerun without changing `--attn-implementation`. |
| Head decomposition aborts | `diagnostics/head_decomposition_check.json`; the architecture may need new out-projection anatomy. |
| DLA decomposition aborts | `diagnostics/dla_decomposition_check.json`; component hook points are not reconstructing the residual stream. |
| All induction scores are blank | `diagnostics/prompt_motif_coverage.csv`; your prompts may not have repeated token IDs after tokenization. |
| Control prompts trigger induction | Check tokenization first. A “no-repeat” English prompt can still repeat token IDs. |
| Ablation effects are tiny everywhere | `tables/baseline_behavior.csv`; the model may not prefer the target on these prompts. |
| Attribution and ablation disagree wildly | That is not automatically a bug. Check the scope: attribution is final-position/direct-path; `all_pos` includes indirect paths. |
| `head_evidence_matrix.png` looks empty | You may have run a tiny smoke set or skipped ablations. The OBS columns should still populate; CAUSAL columns require `--ablate-top > 0`. |

## What goes in the ledger

Draft two or three claims, but keep their rungs separate. Good examples:

```text
L03-OBS: On prompt family F, head LxHy has induction score S under rule R.
L03-ATTR: On the same family, LxHy’s final-position write has attribution A toward target-minus-distractor.
L03-CAUSAL: Under pre-WO zero-ablation at scope all_pos, removing LxHy changes the logit gap by Z.
```

Bad ledger claim:

```text
LxHy is the induction circuit.
```

That sentence tries to wear three lab coats at once. Make it earn each one.
