# Lab 3: Attention Routing, Induction, and What Heads Actually Do

**Evidence levels targeted:** observation → attribution → causality, with the scope of each claim kept explicit. **Prerequisites:** Labs 1–2. Lab 2 told you which attention *layers* pushed on an answer direction. This lab cracks those bars open into individual heads.

## The question

Which positions are routed where, and when does that routing matter for the output?

Those are not the same question. A head can stare at the right token and write nothing useful. Another can have unimpressive-looking attention but contribute strongly through its value/output channel. The whole lab is built to cure heatmap astrology: you will measure pattern, contribution, and causal effect separately, then decide which claims each measurement actually licenses.

## The evidence ladder for a head

| Evidence rung | Measurement | Claim it can support | What it cannot support alone |
|---|---|---|---|
| `OBS` | attention pattern and motif score | “This head routes to previous-token / induction / first-token positions on this prompt distribution.” | That the head matters for the answer. |
| `ATTR` | head output projected through `W_O`, scored against the target-minus-distractor direction | “This head’s final-position write points toward or away from the answer under a frozen-norm approximation.” | That removing the head changes behavior. |
| `CAUSAL` | scoped head ablation | “Under this intervention and prompt set, removing this head changes the logit gap by X.” | A full circuit edge. The all-position effect still bundles many indirect paths. |

Keep the rungs separate in your writeup. The most tempting wrong sentence in this lab is “the attention head explains the answer” because the heatmap looks satisfyingly diagonal. That sentence is a velvet trapdoor.

## What exactly is a head here?

A head has two visible sides:

1. **Routing:** the attention matrix. Rows are query positions, columns are key positions, and each row says where that head reads from.
2. **Writing:** the head’s slice of the attention output-projection input, mapped through that head’s columns of `W_O`. That vector is what the head contributes before shared projection bias and any architecture-specific post-attention norm.

The bench verifies the per-head decomposition before any science: the per-head pieces plus shared projection bias must reconstruct each block’s attention output. The revised lab also runs the Lab 2 component decomposition check, because head attribution inherits the direct-logit-attribution ledger. If that bookkeeping is wrong, the plot is a glass palace built on soup.

### Frozen norms, now explicit

On GPT-2-style pre-norm models, the raw attention-module output is the residual-stream contribution. On Olmo-style post-norm models, the attention output is normalized before it is added to the stream. The revised code handles both RMSNorm and centered LayerNorm in the frozen local linearization used for head attribution.

The artifact `diagnostics/head_attribution_accounting.csv` compares summed head scores against the whole attention-block score. Shared bias and frozen-norm residuals are shown rather than quietly assigned to a random head-shaped hat.

## The motifs

You will score three named patterns:

| Motif | Definition | What to watch for |
|---|---|---|
| previous-token head | query position `q` attends to `q-1` | Often a component of induction rather than a direct final-token writer. |
| induction head | when token at `q` occurred earlier at `j`, attend to token `j+1`, requiring `j+1 < q` | The revised target rule excludes adjacent-repeat self-targeting. |
| first-token sink | query positions attend to position 0 | A resting pattern. It may matter or may be inert; check attribution and ablation. |

The label rule is intentionally simple: content motifs above `0.35` win, with induction and previous-token competing directly; first-token sink is a fallback above `0.5`; high-entropy heads become `diffuse`; the rest are `other`. This rule is not sacred. Finding where it fails is part of the lab.

## The prompt design

Synthetic prompts use non-obvious cycles such as:

```text
B F Q B F Q B F        → Q vs B
red blue green red ... → green vs red
```

They avoid alphabetical and arithmetic priors, where the “induction” answer could be produced by world knowledge about sequences rather than by copying. Natural prompts test whether the same heads still route on repeated phrases:

```text
Marcus went to the lab. Olivia went to the → lab vs store
```

Control prompts have no intended repeated-token structure. The revised lab writes `diagnostics/prompt_motif_coverage.csv` so you can see exactly how the tokenizer turns each prompt into induction-target positions. If a control prompt accidentally has repeated token IDs, the microscope tells on itself.

## The causal intervention

The ablation intervention zeroes one head’s **pre-output-projection slice** before `W_O`. It is run in two scopes:

| Scope | What is zeroed | Interpretation |
|---|---|---|
| `final_pos` | only the head’s final query-position slice | Direct path. This is the scope most comparable to final-position attribution. |
| `all_pos` | the head’s slice at every position | Direct path plus earlier-position writes that later layers can read. |

A previous-token head with tiny `final_pos` effect but large `all_pos` effect has an indirect-path signature: it probably matters because of what it wrote at earlier positions, not because its own final-token output pushed the answer directly.

This is still zero-ablation, not mean-ablation. It is a causal stress test, but it can move the model off distribution. Lab 6 switches to dataset-mean ablation when the claim becomes “this is a faithful circuit.”

## Running it

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

1. `attention_routing_card.md` — the one-page verdict, non-claims, and strongest evidence.
2. `diagnostics/prompt_motif_coverage.csv` — whether each prompt actually contains induction targets after tokenization.
3. `tables/baseline_behavior.csv` — whether the model preferred the target over the distractor before any intervention.
4. `plots/motif_maps.png` — previous-token, induction, sink, and entropy grids over layer × head.
5. `plots/attention_heads_<showcase>.png` — token-labeled heatmaps for the strongest motif heads plus the top-attribution head.
6. `tables/head_table.csv` — every head, averaged over prompts: motif scores, entropy, attribution, label.
7. `tables/head_scores_by_category.csv` — the same evidence split by synthetic, cycle, natural, and control prompts.
8. `diagnostics/control_induction_audit.csv` — false-positive audit for induction labels on control prompts.
9. `tables/ablation_candidate_manifest.csv` — why each head was selected for intervention.
10. `tables/head_ablation_summary.csv` and `plots/ablation_effect_by_head.png` — direct and all-position causal effects by candidate head.
11. `plots/direct_vs_indirect_effect.png` — composition scatter: direct path versus all-position effect.
12. `plots/head_attribution_vs_ablation.png` — whether attribution predicts direct-path causal effect, reported both per prompt and by candidate head.
13. `plots/routing_to_causality.png` — induction motif score versus all-position causal effect.
14. `diagnostics/head_attribution_accounting.csv` — how much whole-block attention attribution is not assigned to heads because of bias or frozen-norm approximation.
15. `diagnostics/ablation_manifest.json` — exact intervention definition and caveats.

## How to read the plots

`motif_maps.png` is observational. It tells you where patterns live, not whether the model uses them.

`head_attribution_by_layer.png` is attributional. A head above zero writes toward the target-minus-distractor direction at the final position; a head below zero writes against it. This is still not a causal result.

`direct_vs_indirect_effect.png` is the lab’s composition detector. Points near the diagonal are mostly direct-path heads. Points far above or below the diagonal have larger all-position than final-position effects, meaning earlier writes matter.

`routing_to_causality.png` is the antidote to overclaiming. An induction score on the x-axis and an ablation effect on the y-axis ask whether a head that routes like induction also matters for this behavior. The best answer is allowed to be “sometimes.”

## Writeup questions

1. Pick the strongest induction-labeled head. Trace its evidence across `OBS`, `ATTR`, and `CAUSAL`. Which rung is strongest? Which rung is weakest?
2. Find a first-token sink head. Does it have high attribution or causal effect? Explain why “sink” should not be used as a synonym for “irrelevant.”
3. Report one previous-token head’s `final_pos` and `all_pos` effects. What exactly does their gap measure? What would a patching experiment need to show to turn that into an edge claim?
4. Compare `head_table.csv` and `head_scores_by_category.csv`. Did any head’s label depend strongly on synthetic prompts but fail on natural prompts?
5. Did any control prompt produce above-bar induction scores? Use `diagnostics/control_induction_audit.csv` and `prompt_motif_coverage.csv` to decide whether the prompt or the label rule is to blame.
6. Does attribution predict direct-path ablation in this run? Report both the per-prompt and by-head Spearman correlations. Which one is the cleaner statistic, and why?
7. Choose one head where routing, attribution, and causal effect disagree. Write the most careful claim you can make about it without smuggling in stronger evidence.

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
