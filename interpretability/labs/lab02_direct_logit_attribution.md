# Lab 2: Direct Logit Attribution and Component Accounting

**Evidence level targeted:** attribution (`ATTR`), plus one deliberately narrow
causal extension. **Prerequisite:** Lab 1 (you will reuse its residual-stream
semantics and its lens).

## The question

Lab 1 showed you *when* a prediction emerges over depth. This lab asks *who
wrote it*: which components — the embeddings, each attention block, each MLP
block — pushed the model toward or away from a specific answer?

The honest framing matters from the first minute: what you will build is a
**ledger**, not a causal map. A ledger can be arithmetically perfect and still
misleading about responsibility. Holding both of those thoughts at once is the
skill this lab teaches.

## The idea

At the final position, the model's output is

```text
logits = lm_head(final_norm(x)),   x = embed + Σ attn_k + Σ mlp_k
```

The residual stream `x` is an exact sum of everything that was ever added to
it — the bench *verifies* this decomposition on your machine before any
science runs (`diagnostics/dla_decomposition_check.json`). Pick an answer
direction

```text
d = unembed[" Berlin"] - unembed[" Paris"]
```

and you can ask, for every component `c`: how much did `c @ d` move the final
logit difference?

### The catch: the final norm

`final_norm` is not linear — its scale depends on the *whole* stream — so
component scores don't simply pass through it. The course convention (pinned
in the design guide, §1.11) is to **freeze the norm's data-dependent
statistics at their actual values from this forward pass**, which makes the
map linear. Read `compute_direct_logit_attribution` in the lab file: the
linearization is written out, commented, on purpose. Three facts to internalize:

1. **For the sum, freezing is exact.** The full stream's score equals the real
   (fp32) logit difference. The ledger balances; the run summary reports the
   worst balance error so you can check.
2. **For the parts, freezing is an approximation.** A component's score
   assumes the norm scale wouldn't change without it — but it would.
3. **Constants go in their own row.** A LayerNorm bias or `lm_head` bias
   contributes to the logit difference but belongs to no component. The lab
   reports it as `constant` rather than smearing it across the ledger.

GPT-2's LayerNorm additionally subtracts a mean — that part *is* linear and is
folded into the scoring vector exactly (look for `w = v - v.mean() * ones`).

### The second catch: where contributions live

On GPT-2, the attention output is literally added to the stream. On Olmo-3 it
is **normed first** (`post_attention_layernorm(attn(x))` — post-norm
architecture), so hooking `attn` itself would score a tensor that never
touches the stream. The bench refuses to guess: it captures both candidate
hook points, checks which pair reconstructs every block's residual delta, and
records the verdict in `diagnostics/component_anatomy.json`. Read that file
once per model. If you ever port this lab to a new architecture, that check is
what stands between you and a beautifully plotted lie.

## Running it

```bash
# Smoke (gpt2, CPU-class, 4 examples) — always first:
python interp_bench.py --lab lab2 --tier a

# The real run (Olmo-3-7B, bf16):
python interp_bench.py --lab lab2 --tier b --prompt-set full --topk 10

# Skip the ablation extension if you're short on time:
python interp_bench.py --lab lab2 --tier b --ablate-top 0
```

Prompt families: `fact`, `relation` (antonyms), `grammar` (morphology), and
`conflict` — prompts where the context overrides a stored fact and the
*stored* fact is the distractor. The conflict family is where the ledger gets
interesting: expect **negative** entries from components still pushing the
stored answer.

One example (`plural_mouse`, " mouses") is included *because* it fails the
single-token gate on both course tokenizers. Find it in
`diagnostics/answer_tokenization.csv` and confirm the drop reason. Multi-token
answers are the #1 silent killer of attribution numbers in the wild.

## First artifact-reading path

1. `run_summary.md` — headline table and the two balance checks.
2. `diagnostics/component_anatomy.json` — which hook points were verified, and
   the reconstruction error they had to beat.
3. `plots/contribution_by_layer.png` — where attn and MLP write the answer,
   per category.
4. `plots/cumulative_logit_diff.png` — the ledger assembling over depth.
5. `plots/dla_vs_lens_<showcase>.png` — the frozen-norm ledger against Lab 1's
   moving-basis lens *for the same stream*. They disagree at early depths.
   Explain why before reading on. (Hint: which norm statistics does each use
   at depth k?)
6. `tables/ablation_results.csv` and `plots/attribution_vs_ablation.png` — the
   extension's payload.

## The extension: does the ledger predict ablation?

For each example the lab zero-ablates the top-|attribution| components (plus a
random and a low-attribution control) **at the final position only** and
re-measures the logit difference. This "direct-path" restriction is
deliberate: it removes exactly the contribution the ledger counted, so
attribution and causal effect are commensurable. The Spearman correlation and
the scatter are your answer to "does high attribution mean high effect?" —
and the off-diagonal points are worth a paragraph each in your writeup.

What this ablation does **not** test: a component's indirect influence through
later attention (its write at *earlier* positions is left intact). Lab 5
(patching) owns that question; don't claim it here.

## Writeup questions

1. Which layers add the largest positive contributions, and do attention and
   MLP peak at different depths? Cite `layer_component_summary.csv`.
2. In the conflict family, which components push the stored fact? Are they the
   same components that push the correct answer in the matching `fact`
   example? Cite specific rows of `component_contributions.csv`.
3. Does high attribution predict high ablation effect in your run? Quote the
   Spearman rho and one rank inversion, and explain the inversion.
4. Construct one hypothetical mechanism where the ledger is exactly right
   about responsibility, and one where it is arithmetically correct but
   misleading. (The second exists in your own data if you look.)
5. The `dla_vs_lens` plot shows two readouts of the same stream disagreeing.
   What claim, exactly, does each readout license?

## Symptom-first debugging

| Symptom | First place to look |
|---|---|
| `component anatomy` abort | `diagnostics/component_anatomy.json` — candidates tried and their errors; new architecture needs paths added in the bench |
| `decomposition check` abort | Same file, then `--dla-tolerance` only if dtype rounding is the proven cause |
| Ledger doesn't balance (summary §5) | You edited the scoring math; the constant row is the usual suspect |
| Huge `frozen_vs_model_abs_err` | Model applies logit post-processing (softcapping?) — check `model_anatomy.json` notes |
| Everything dropped at the gate | `diagnostics/answer_tokenization.csv` — your custom prompts' answers are multi-token |

## Custom prompts

`--prompt-set path.json` takes a JSON list of objects with keys
`example_id, category, prompt, target, distractor` (optional `note`).
Categories must be from `fact | relation | grammar | conflict`. Both answers
must be single tokens *with their leading space*; the gate will tell you which
ones aren't.

## What goes in the ledger

2–3 claims, tagged honestly. The lab drafts candidates in
`ledger_suggestions.md` with measured numbers and falsifiers; edit them — the
editing is the coursework. A claim that says "MLP 22 stores the Germany fact"
is over your evidence; a claim that says "MLP 22's output is the largest
direct contributor to the Berlin-vs-Paris logit gap, and zeroing it at the
final position removes X of Y logits" is exactly your evidence.
