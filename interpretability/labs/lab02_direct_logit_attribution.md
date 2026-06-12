# Lab 2: Direct Logit Attribution and Component Accounting

**Evidence level targeted:** attribution (`ATTR`), plus one deliberately narrow causal extension.
**Prerequisite:** Lab 1. You will reuse its residual-stream semantics and its logit-lens caution.

## The question

Lab 1 asked *when* a prediction appears over depth. Lab 2 asks *who wrote it*: which residual-stream writers pushed the next-token distribution toward a target answer and which ones pushed against it?

The product of this lab is a **ledger**, not a causal map. A ledger can balance perfectly and still be a poor story about responsibility. That tension is the whole lantern-room of the lab: arithmetic first, interpretation second, causality only where you actually intervened.

## The core idea

At the final position, the model reads out a residual stream:

```text
x_L = embed + Σ_k attn_k + Σ_k mlp_k
logits = lm_head(final_norm(x_L))
```

Pick a target and distractor, then score every component against the answer direction:

```text
d = unembed[target] - unembed[distractor]
```

For the prompt

```text
The capital of Germany is
```

with target ` Berlin` and distractor ` Paris`, a positive score means “this component pushed toward Berlin over Paris.” A negative score means it pushed toward Paris over Berlin. The same rule applies in conflict prompts, where the target is the in-context answer and the distractor is the stored prior.

The bench verifies that the captured components reconstruct the final pre-norm residual stream before the lab treats any score as evidence. The microscope checks itself before you admire the specimen.

## The final-norm catch

The final norm is not a linear function of the stream. Its scale depends on the whole vector, so a component cannot simply be dotted with the raw answer direction. The course convention is to freeze the final norm’s data-dependent statistics at their actual values from the forward pass. That gives a local linear readout:

```text
score(component) = component @ frozen_scoring_vector
```

Three rules keep the ledger honest:

1. **The aggregate balances.** The sum of all component scores plus constants should match the frozen-norm logit difference, up to the decomposition tolerance.
2. **The parts are approximate.** A per-component score assumes the final norm’s scale would stay fixed if that component changed. In a real intervention, it may not.
3. **Constants are not components.** Final-norm bias and `lm_head` bias can change the target-vs-distractor logit difference, but no attention or MLP layer wrote them. They get a separate `constant` row.

For centered LayerNorm models such as GPT-2, the mean subtraction is linear and is folded into the scoring vector exactly. For RMSNorm-style models, the frozen scale is the key ingredient. Read `compute_direct_logit_attribution` in the Python file: the math is intentionally exposed rather than hidden in a helper.

## The hook-point catch

On some models, an attention module’s raw output is not the tensor that is added to the residual stream. Olmo-style post-attention normalization is the classic trap: `attn(x)` may be normalized before it becomes the residual write. Hooking the wrong site gives a beautiful chart of a tensor the residual stream never saw.

The bench refuses to guess. It resolves component anatomy by testing candidate attention and MLP hook points against the per-block residual deltas, then writes the verdict to:

```text
diagnostics/component_anatomy.json
```

Treat that file as the “microscope aligned” certificate. If it fails, the lab should fail.

## Prompt families

The built-in prompt set covers four categories:

| Category | Purpose | Example |
|---|---|---|
| `fact` | stored factual preference | `The capital of Germany is` |
| `relation` | semantic relation or antonym | `The opposite of hot is` |
| `grammar` | morphology | `The past tense of run is` |
| `conflict` | context overrides prior knowledge | `In this story, the capital of Germany is Paris. The capital of Germany is` |

Every target and distractor must be a single token under the current tokenizer. The deliberately failing example, `plural_mouse`, exists so you can see the gate reject a multi-token answer. Find the evidence in:

```text
diagnostics/answer_tokenization.csv
```

Multi-token answer mishandling is a common source of attribution errors. Do not let it into this lab.

## Running it

```bash
# Smoke run, CPU-class, usually gpt2 with the small prompt set
python interp_bench.py --lab lab2 --tier a

# Standard run on the course model
python interp_bench.py --lab lab2 --tier b --prompt-set full --topk 10

# Skip the ablation extension
python interp_bench.py --lab lab2 --tier b --prompt-set full --ablate-top 0

# Use a custom prompt file, JSON or CSV with example_id/category/prompt/target/distractor
python interp_bench.py --lab lab2 --tier b --prompt-set path/to/my_prompts.csv
```

`--ablate-top N` controls the extension. For each kept example, the lab tests the top `N` components by absolute attribution plus random and low-attribution controls.

## What gets written

The revised lab writes the standard run contract plus a few extra “do not fool yourself” artifacts.

### Diagnostics

| Artifact | Why it exists |
|---|---|
| `diagnostics/answer_tokenization.csv` | shows which examples passed the single-token gate and why any were dropped |
| `diagnostics/component_anatomy.json` | records which hook points actually reconstruct block residual deltas |
| `diagnostics/dla_decomposition_check.json` | proves captured components reconstruct the final pre-norm stream |
| `diagnostics/block_reconstruction_by_example.csv` | per-example residual-delta reconstruction errors, not just one global verdict |

### Tables

| Artifact | What to read from it |
|---|---|
| `tables/baseline_behavior.csv` | target and distractor logits, probabilities, ranks, and the model’s top token before attribution |
| `tables/component_contributions.csv` | one row per component, with signed score, push direction, signed fraction, and bounded mass share |
| `tables/block_ledger.csv` | attention score, MLP score, block total, and cumulative value after each block |
| `tables/top_components.csv` | top-|attribution| rows per example, including whether each pushes the target or distractor |
| `tables/dla_balance.csv` | frozen-norm metadata, answer-direction norms, and balance errors |
| `tables/layer_component_summary.csv` | category-level attention and MLP averages by layer |
| `tables/category_summary.csv` | headline category aggregates, including gross positive and negative mass |
| `tables/ablation_results.csv` | final-position ablation effects for top, random, and low-attribution components |
| `tables/ablation_summary_by_selection.csv` | top-vs-control comparison for the ablation extension |

`results.csv` is an alias of `component_contributions.csv` for the course-wide artifact contract.

### Plots

| Artifact | Question it answers |
|---|---|
| `plots/contribution_by_layer.png` | where attention and MLP writes point toward or away from the target |
| `plots/signed_component_heatmap.png` | which examples hide internal fights behind a category mean |
| `plots/cumulative_logit_diff.png` | how the ledger assembles the final logit difference over depth |
| `plots/category_ledger_composition.png` | how much positive and negative mass cancels inside each category |
| `plots/ledger_balance_errors.png` | whether the bookkeeping balances, and how big the dtype gap is |
| `plots/top_component_by_example.png` | the largest component per example, with sign visible |
| `plots/ledger_waterfall_<example>.png` | the largest positive and negative ledger entries for the showcase prompt |
| `plots/dla_vs_lens_<example>.png` | frozen final-norm ledger versus Lab 1’s moving-basis logit lens |
| `plots/attribution_vs_ablation.png` | whether high attribution predicts final-position ablation effect |
| `plots/ablation_mismatch_examples.png` | the biggest places where attribution and intervention disagree |

## Reading path

Start here, in this order:

1. `run_summary.md`: read the model, kept/dropped counts, headline table, balance checks, and drafted claims.
2. `tables/baseline_behavior.csv`: confirm the model’s starting behavior. A negative target-vs-distractor gap is allowed, but you need to know it happened.
3. `diagnostics/component_anatomy.json` and `diagnostics/dla_decomposition_check.json`: confirm that the instrument aligned before interpreting any component row.
4. `tables/dla_balance.csv` and `plots/ledger_balance_errors.png`: check that the ledger sum matches the frozen-norm readout.
5. `plots/contribution_by_layer.png` and `tables/layer_component_summary.csv`: locate the broad attention/MLP writing pattern.
6. `plots/category_ledger_composition.png`: check cancellation. A small net score can hide two large opposing coalitions.
7. `tables/top_components.csv` and `plots/top_component_by_example.png`: identify the biggest writers.
8. `plots/dla_vs_lens_<example>.png`: explain why the logit lens and DLA curve disagree at early depths.
9. `tables/ablation_results.csv` and `plots/ablation_mismatch_examples.png`: inspect the gap between ledger score and intervention effect.

## The ablation extension, carefully scoped

The extension zero-ablates selected attention or MLP writes at the **final position**. This is a narrow intervention intended to be comparable to the final-position ledger, but it is not identical to subtracting a frozen ledger entry.

Why not? Because when you alter an early block’s final-position write, later blocks still run on a changed final-position residual stream. The final norm is also live again. The intervention therefore asks:

```text
What happens when this component's final-position write is removed and the downstream final-position computation continues?
```

It does **not** ask:

```text
What is the arithmetic contribution of this row under the frozen final norm?
```

That arithmetic contribution is the attribution score. The difference between the two is recorded as:

```text
effect_minus_attribution
```

Large mismatches are not bugs by default. They are the teaching payload: the ledger is not a causal map.

What this extension still does not test: indirect effects through earlier positions. Lab 3 shows why all-position head ablation can differ from final-position ablation, and Lab 5 uses patching to dissect that kind of pathway.

## How to interpret common patterns

| Pattern | Interpretation |
|---|---|
| Ledger balance error is tiny | component capture and frozen-norm accounting are numerically coherent |
| Frozen-vs-model gap is larger | usually dtype or fp32 reimplementation gap, not a component decomposition failure |
| Conflict examples have negative components | some components still push the stored answer over the in-context answer |
| Top component mass share is high | one component dominates gross writer mass for that prompt |
| Signed fraction is huge | likely denominator cancellation; use bounded mass share instead |
| Attribution and ablation agree | the frozen ledger row is a useful predictor for this final-position intervention |
| Attribution and ablation disagree | downstream computation or live normalization changed the causal effect |

## Debugging guide

| Symptom | Likely cause | What to check |
|---|---|---|
| many prompts dropped | target or distractor is not single-token under this tokenizer | `diagnostics/answer_tokenization.csv` |
| component anatomy fails | model architecture hook sites are not covered by the bench candidates | `diagnostics/component_anatomy.json` |
| decomposition check fails | attention/MLP writes do not reconstruct the stream at tolerance | `diagnostics/dla_decomposition_check.json` and `--dla-tolerance` |
| ledger balance fails but decomposition passes | final-norm linearization implementation mismatch | `tables/dla_balance.csv`, model norm class, dtype |
| all contributions look tiny | target/distractor direction may be weak or wrong token ids | `tables/baseline_behavior.csv` and `prompt_manifest.csv` |
| ablation effect is much larger than attribution | live downstream computation amplified the perturbation | `effect_minus_attribution` in `ablation_results.csv` |
| ablation effect has the opposite sign | the component’s frozen direct push differs from its live downstream role | inspect the mismatch plot and the example’s waterfall |

## Writeup questions

1. Which category has the largest mean target-vs-distractor logit difference? Does that category also have the largest gross component mass?
2. For one fact prompt and one conflict prompt, compare the top positive and top negative components. Are the same layers involved?
3. Which layers have the strongest attention contribution and which have the strongest MLP contribution? Use `layer_component_summary.csv`.
4. Pick one example where `category_ledger_composition.png` suggests cancellation. Show the component rows that create the fight.
5. Compare `cumulative_logit_diff.png` with `dla_vs_lens_<example>.png`. Why does a frozen-norm ledger differ from a per-depth logit lens?
6. In the ablation extension, report one component where attribution and causal effect agree, and one where they disagree. What exactly did the intervention do in each case?
7. Draft one `ATTR` claim and one narrow `CAUSAL` claim. Make sure the causal claim names the intervention scope.

## Suggested ledger entries

Good Lab 2 claims look like this:

```text
[L02-C1 | ATTR]
For prompt family F on model M, attention and MLP components at layers A-B
account for the largest signed pushes toward target over distractor under the
frozen-final-norm DLA convention. Artifact: contribution_by_layer.png.
Falsifier: component decomposition or ledger balance fails, or the pattern
vanishes on a fresh prompt family.
```

```text
[L02-C2 | CAUSAL, narrow]
For selected final-position component writes on prompt set P, final-position
zero-ablation changes the target-vs-distractor logit difference by Z, with
rank relationship rho to the frozen DLA score. Artifact: ablation_results.csv.
Falsifier: random or low-attribution controls show the same effect, or the
claim is extended to earlier-position indirect paths not tested here.
```

Do not write “layer K stores the answer.” This lab does not localize a fact. It accounts for final-position pushes toward one target over one distractor under one readout convention. Keep the claim narrow enough to defend.
