# Lab 8: Superposition, Sparse Autoencoders, and Transcoders

**Evidence level targeted:** observation / decodability at the feature level,
upgraded to causality for the one feature you clamp and control.
**Prerequisites:** Labs 1–6. The bridge loads Lab 4's saved `truth_direction.pt`
if present (optional). **Back to base models** — the pinned SAE and transcoder
weights were trained on base models, so this lab uses `gpt2` (Tier A) and
`allenai/Olmo-3-1025-7B` (Tier B), no chat template.

## The question

A single neuron rarely means one thing. Why do dense activations resist
neuron-level reading — and can a *sparse dictionary* recover units worth
naming? And once you have a candidate "feature," how do you tell a discovered
concept from a convenient coordinate, without fooling yourself?

This lab is three parts under a strict time budget, because it can eat a week
if you let it. Part 0 is a 30-minute CPU toy that makes superposition visible.
Part 1 is the core: feature interpretation where the graded skill is **label
validation**, not labeling. Part 2 is a deliberately small transcoder section
whose only job is to set up Lab 9.

## Part 0 — superposition in a jar (CPU, ~5 min of compute)

Train the Elhage et al. toy model: a tiny autoencoder that must pack
`n_features = 20` sparse features into `d_hidden = 5` dimensions. Sweep the
input sparsity and watch the geometry change. The result is canonical and you
should be able to predict it before you read the plot:

- **Dense inputs** (every feature on at once): the model represents *exactly
  `d_hidden` features*, orthogonally, and throws the rest away. Interference is
  too expensive when everything is always on.
- **Sparse inputs** (features rarely co-occur): the model represents **more
  features than it has dimensions**, in *superposition* — non-orthogonal
  directions that interfere only on the rare occasions two features fire
  together. The sparser the data, the more features it dares to pack in.

That is the whole reason neurons are polysemantic, demonstrated rather than
asserted: a real network is in the sparse regime, representing far more
features than it has neurons. `plots/toy_superposition_geometry.png` shows the
feature-norm collapse and the WᵀW interference; one figure, one paragraph in
your writeup.

## Part 1 — the feature atlas (the core)

Load a pretrained SAE for the tier's model and run it over the frozen,
domain-tagged corpus (`data/sae_feature_corpus.csv`). Then **find, label, and
validate** features. The labeling is the easy part; the validation is the
skill.

### Two rankings that disagree

Rank features by **peak activation** and by **firing frequency**. They surface
largely different features (`plots/ranking_disagreement.png`), and that
disagreement is a lesson: max-activation ranking foregrounds rare, high-
amplitude outliers (often polysemantic — they spike on one odd token across
unrelated contexts); frequency ranking foregrounds broadly-active basis
vectors. Neither is "the interpretable one"; knowing which question each
answers is the point.

### The validation battery (the graded function)

For each atlas feature the lab proposes a label — the majority domain of its
top-activating contexts — and then *tests* it, automatically, the way you must
learn to do by hand:

| Check | What it catches |
|---|---|
| **held-out AUC** vs domain membership (label's own top contexts excluded) | does the feature actually predict its domain on text it wasn't labeled from? |
| **confusable-pair AUC** (e.g. chemistry vs cooking) | a **concept** feature separates the pair; a **token** feature that fires on the shared word ("acid" in both) cannot |
| **polysemanticity entropy** over top contexts | top contexts spanning unrelated domains = one direction, many meanings |

The verdict is one of **survived** (high domain AUC, pure label, beats the
confusable twin), **narrowed** (decent AUC), **token-feature** (fails the
confusable pair — it tracked a string, not a concept), **polysemantic**,
**killed**, or **silent-on-corpus**. The atlas is *required* to contain at
least one label you killed; a clean sheet means your corpus was too easy or
your battery too kind.

This is why the corpus is built from **confusable domain pairs**
(chemistry/cooking, finance/sports, law/medicine, weather/emotion) that share
surface tokens. Without them you cannot tell "fires on chemistry" from "fires
on the word *acid*," and that confusion is the single most common interpretive
error in the literature.

### What you will actually see (a real finding)

On the 65k-feature Olmo SAE the verdicts span the full range: a handful of
clean concept features survive, several are killed, and the **max-activation
top features are mostly polysemantic outliers** — exactly the trap the two
rankings warn about. Most of the dictionary is *silent on this corpus*, which
is not the same as dead: those features simply never get the inputs that fire
them here. Reconstruction is honest too — a jumprelu SAE reconstructs at FVU
around 0.37 with ~110 active features per token, not a magic lossless code.

## Part 2 — the transcoder (the bridge to Lab 9)

Load a gpt2 MLP transcoder. Teach the object: an **SAE reconstructs the
activations at a site**; a **transcoder reconstructs the MLP's *computation*** —
it maps the layer's input to its output. The lab verifies this two ways:
reconstruction error (FVU), and the **downstream-logit KL when the
reconstruction is spliced in for the real MLP output** — which stays tiny
(~0.01), meaning the transcoder preserves the computation, not just the vector.
Then it **de-embeds** a few transcoder features (project the decoder row
through the unembedding) to read which tokens each feature *promotes*.

The one paragraph you owe: *why does feature-level circuit tracing (Lab 9) want
input→output objects rather than site snapshots?* Because a graph needs edges —
"these input features cause that output feature" — and only an object that
reconstructs the *map* gives you edges. A site SAE gives you nouns; a
transcoder gives you verbs.

## Bridges and the causal extension

- **Lab 4's truth direction.** The lab reports the cosine between the
  best-aligned SAE feature's decoder direction and Lab 4's truth direction.
  Expect it to be **low** (~0.07 on Olmo). That is a finding, not a failure:
  the truth direction the probe found is not any single SAE feature — it is
  distributed, or it is an SAE coordinate that no single dictionary atom
  captures. Decodable-as-a-direction and recoverable-as-a-feature are different
  claims.
- **Feature clamp (CAUSAL).** Pick the cleanest concept feature — validated,
  *low firing frequency* (a feature that fires on 85% of tokens is a basis
  vector, not a concept handle), with a keyword battery — and clamp it ON
  during generation by adding multiples of its decoder direction. There is a
  **narrow window**: around 1× the feature's peak activation it induces its
  concept (a "law" feature makes neutral prompts generate *"The court has ruled
  that the defendant is not guilty…"*), and past ~3× it collapses generation
  into repetition. A random-feature control at the same dose does not induce
  the concept. That single row is your one CAUSAL claim — decodability made
  sufficient, with a control and a fluency cost, never a cherry-picked
  screenshot.

## The conventions are validated, not assumed

The single fastest way to get a wrong answer here is a wrong loading
convention. Each SAE/transcoder expects its activations in a specific form, and
feeding the wrong form silently triples the FVU while the code runs fine. These
were each settled empirically at authoring time:

- the gpt2 resid SAEs were trained in TransformerLens, which **centers** the
  residual stream — so they reconstruct well only on a per-token-demeaned input
  (`center_input=True`);
- the gpt2 transcoder wants the **bare LayerNorm** of the pre-MLP residual (no
  affine γ/β), not the model's full `ln_2` output;
- the Olmo SAE is **jumprelu** — features below a learned per-feature threshold
  are exactly zero, not ReLU'd.

If your FVU is implausibly large, suspect the convention before the science.

## Running it

```bash
python interp_bench.py --lab lab8 --tier a    # gpt2 + jbloom SAE + Dunefsky transcoder (CPU-ok)
python interp_bench.py --lab lab8 --tier b     # Olmo-3-1025-7B + decoderesearch SAE
```

The SAE/transcoder weights download from the Hub on first run (a few hundred
MB; the Olmo SAE is ~2 GB). The transcoder section always runs on gpt2 — on
Tier B a small auxiliary gpt2 is loaded for it, because the ungated transcoder
weights the course uses are gpt2's. Generation is greedy (frozen), so the only
moving part in the clamp sweep is the dose.

## First artifact-reading path

1. `feature_atlas.md` — the deliverable: every label, its validation verdict,
   evidence with the peak-activating token highlighted, and what the atlas does
   *not* show.
2. `plots/toy_superposition_geometry.png` — why neurons resist reading.
3. `plots/ranking_disagreement.png` — peak-activation vs frequency rankings.
4. `plots/atlas_verdicts.png` — how many labels survived, how many you killed.
5. `transcoder_reconstruction_report.json` — FVU, splice-in KL, de-embedded
   features; the bridge to Lab 9.
6. `tables/feature_clamp.csv` and `plots/feature_clamp.png` — the one CAUSAL
   feature; **read the `sample` column**, do not just trust the hit count.

## Writeup questions

1. Which of your labels survived validation untouched, which needed narrowing,
   and which died? Quote the held-out AUC and the confusable-pair AUC for one
   survivor and one casualty. What did the dead one teach you?
2. Max-activation ranking vs frequency ranking: which produced more
   interpretable features, and why might that be? Point at specific feature ids.
3. In one paragraph: what does a transcoder reconstruct that an SAE does not,
   and why does Lab 9 need that? Use your splice-in KL as evidence that it
   reconstructs the *computation*.
4. **The truth-direction bridge.** Your best SAE feature aligns with Lab 4's
   truth direction at cosine ≈ ?. Argue what that does and does not imply about
   whether "truth" is a feature this SAE represents.
5. **Real patterns (Dennett) redux.** Take your best-labeled feature and your
   worst. Argue the best one is a *discovered concept*; then steelman the
   *deflationary* reading of the worst one ("a convenient coordinate the SAE
   found because the loss rewarded it"). Use the clamp result: does causal
   sufficiency change the argument?

## Symptom-first debugging

| Symptom | First place to look |
|---|---|
| FVU implausibly large (≫1) | wrong loading convention — centering / bare-LN / jumprelu threshold; see "conventions are validated" above |
| every feature looks polysemantic | corpus too narrow to separate concept from token, or you're reading only top contexts without the negative-prompt validation |
| a "great" label dies on the confusable pair | it was a token feature all along — it fired on the shared word, not the concept; that is the lab working, not failing |
| clamp does nothing | dose too small (clamp by multiples of the feature's *peak activation*, not a unit vector), or you picked a ubiquitous feature — choose a low-firing concept feature |
| clamp produces only repetition | dose past the window (~3× peak); back off to ~1× and read the `sample` column |
| truth-direction cosine ≈ 0 | not a bug — likely a real finding that the probe direction is not a single SAE feature |

## What goes in the ledger

2–4 claims. The reconstruction claim is `OBS` (FVU, L0, silent fraction). The
atlas claim is `DECODE` and must carry **a feature id, its label, its held-out
AUC, and how many labels you killed** — "the SAE found a chemistry feature" is
not a claim; "feature 1265 labeled 'law' survives at held-out AUC 0.81 and
separates law from medicine at 0.86, while 10 of 21 labeled features were
killed by the same battery" is. The clamp claim is `CAUSAL` and must state the
**dose, the effect, the random control, and the fluency cost**. Retire any
earlier claim this lab undercuts.
