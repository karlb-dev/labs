# Lab 8: Superposition, Sparse Autoencoders, and Transcoders

**Evidence level targeted:** observation / decodability at the feature level,
upgraded to causality for the one feature you clamp and control.
**Prerequisites:** Labs 1–7 (and sets up Lab 9). You know residual-stream conventions and "readout is an instrument" (Lab 1), attribution as ledger not cause (Lab 2), the decodability ≠ use skepticism and truth direction (Lab 4), patching/causal interventions (Lab 5), manual circuit scope (Lab 6), and steering-dose hygiene with controls + side-effect costs (Lab 7; the clamp here uses the same "multiples of observed peak/median norm" discipline). The transcoder section exists specifically to give Lab 9 input→output objects with edges rather than site snapshots. The bridge re-uses Lab 4's truth direction (optional saved vector) for a "distributed vs single feature" comparison. **Back to base models** — the pinned SAE and transcoder weights were trained on base models, so this lab uses `gpt2` (Tier A) and `allenai/Olmo-3-1025-7B` (Tier B), no chat template.

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
feature-norm collapse and the WᵀW interference. The upgraded
`plots/toy_superposition_phase_diagram.png` then compresses the same story into
a dose-response: as sparsity rises, represented features increase, and the bill
is paid in off-diagonal interference. One toy, two lenses: geometry and phase.

**Make the concept pop:** before you open the plot, predict what must happen.
Dense (all features always on): the autoencoder can afford only d_hidden orthogonal
directions and drops the rest (interference is fatal when everything co-occurs).
Sparse (features rarely co-occur): it can pack more than d_hidden by accepting
interference on the rare collisions — exactly the geometry a real network lives
in. The left panel (norms) and right two heatmaps (WᵀW off-diagonal) make the
collapse vs overlap visible in one glance. This is the "why" for every SAE
result that follows.

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
answers is the point. The upgraded run also writes
`plots/sae_activity_dashboard.png`, which places the atlas choices inside the
whole dictionary cloud, and `tables/feature_activity_distribution.csv`, which
turns the long tail of firing frequencies into numbers you can cite.

**Make the concept pop:** open `ranking_disagreement.png` (or the feature_rankings.csv). The red triangles (top by max-act) sit at high peak but near-zero frequency; the green squares (top by freq) sit at modest peak but high frequency across the corpus. Overlap on the top N is typically 0. Max-act picks the flashy rare events; frequency picks the workhorse directions. The atlas deliberately mixes both so you see the trap.

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
your battery too kind. The upgraded lab makes this battery auditable with
`plots/feature_validation_matrix.png` and `tables/feature_evidence_matrix.csv`,
which line up held-out AUC, confusable-pair AUC, purity, polysemy, and firing
rarity for every proposed label.

**Make the concept pop:** the skill being graded is *validation*, not labeling.
Top-context purity can be 1.0 ("code" on every top line) while held-out AUC is
0.57 — the feature was tracking a surface pattern the corpus happened to
associate with the label domain. The confusable pairs (built into the frozen
corpus on purpose) and the held-out split are the apparatus that catches the
most common interpretive error in the literature. A "great label" that dies on
the battery is the lab working, not failing. The feature_atlas.md you write
is required to show the dead ones with their numbers.

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

**Make the concept pop:** look at the actual atlas rows (in feature_atlas.md or the csv). High-peak "code" features often die (purity 1.0 on top contexts, held-out AUC ~0.57). The one that survives here ("emotion") has lower peak but still separates its confusable twin. The clamp picks a *narrowed* low-fire "law" feature (not the highest-AUC survivor) because high fire-fraction features are basis vectors, not concept handles. The numbers, the verdicts, and the "killed" count are the evidence — not a pretty screenshot of one top context.

## Part 2 — the transcoder (the bridge to Lab 9)

Load a gpt2 MLP transcoder. Teach the object: an **SAE reconstructs the
activations at a site**; a **transcoder reconstructs the MLP's *computation*** —
it maps the layer's input to its output. The lab verifies this two ways:
reconstruction error (FVU), and the **downstream-logit KL when the
reconstruction is spliced in for the real MLP output** — which stays tiny
(~0.01), meaning the transcoder preserves the computation, not just the vector.
Then it **de-embeds** a few transcoder features (project the decoder row
through the unembedding) to read which tokens each feature *promotes*. The
upgraded lab writes `tables/transcoder_feature_promotes.csv` and
`plots/transcoder_feature_cards.png`, so the bridge to Lab 9 is not just an FVU
number. You see the input→output object, the downstream KL, and the output-token
tendencies in one little feature passport.

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

**Make the concept pop:** the clamp is the only CAUSAL evidence in the lab.
Everything else is OBS/DECODE (reconstruction stats, label validation on a
fixed corpus). The dose is deliberately in multiples of the *feature's own
observed peak activation* (parallel to Lab 7's median-norm doses) so the
number is physically meaningful and the window (induce at ~1×, collapse by ~3×)
is visible in the plot and the sample generations in the CSV. Random control
stays at 0; distinct-ratio fluency proxy flags the repetition past the window.
The upgraded `plots/clamp_operating_window.png` puts concept hits and fluency
side by side, because the best dose is not the biggest dose. Read the `sample`
column at each dose — the numbers alone do not tell the story.

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

**Headline numbers note:** The atlas labels and validates ~20 features (N_ATLAS) against a ~260-entry corpus (23–28 entries per domain × 10 domains + 6 mixed). Ranking disagreement, dead/silent features, confusable-pair near-miss AUCs, and the explicit killed count are the robust outputs; any single “X% survived” is on a small curated feature sample and carries the one-sig-fig + validation-battery caveat.

Always run Tier A (gpt2) smoke first — it exercises the full pipeline (toy +
SAE atlas + transcoder + bridges) on CPU in ~75 s and still produces the
required spread of verdicts plus the honest "no clean CAUSAL on the weak base
model" outcome.

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

Diagnostics first (instrument/model health), then the core deliverables that
force the distinctions.

1. `diagnostics/model_anatomy.json` (and tokenizer info) — confirm you are on the
   expected base model and layer; loading conventions are model- and SAE-specific.
2. `plots/feature_evidence_dashboard.png` — start here. It ties the toy model,
   SAE health, atlas verdicts, transcoder check, truth bridge, and clamp status
   into one claim map.
3. `feature_atlas.md`, `tables/feature_atlas.csv`, and
   `tables/feature_evidence_matrix.csv` — every proposed label, its full
   validation battery (held-out AUC, confusable AUC, polysemy entropy, purity,
   fire fraction), the top-context evidence with peak token highlighted (⟦ ⟧),
   the verdict, and the explicit "What the atlas does NOT show" section.
4. `plots/feature_validation_matrix.png` — the same evidence as a row-by-row
   heatmap. **Look for:** labels that are strong on one lock and fail another;
   that mismatch is where most false feature stories live.
5. `plots/toy_superposition_geometry.png`, `plots/toy_superposition_phase_diagram.png`,
   and `toy_superposition_stats.json` — the geometry that explains polysemanticity.
   Predict the collapse before you look: dense → exactly d_hidden orthogonal;
   sparse → more than d_hidden with rising off-diagonal interference.
6. `plots/ranking_disagreement.png` + `tables/feature_rankings.csv` — the two
   rankings in one scatter. **Look for:** low overlap on the highlighted top-N
   points (max-act rare spikes versus frequency workhorse directions).
7. `plots/sae_activity_dashboard.png`, `tables/feature_activity_distribution.csv`,
   `plots/domain_validation_summary.png`, and `tables/domain_validation_summary.csv`
   — corpus coverage and domain-level validation. These catch the one-feature
   anecdote before it becomes a claim.
8. `plots/atlas_verdicts.png` — the distribution. Count the red "killed" bar;
   the lab is working when it is large.
9. `transcoder_reconstruction_report.json`, `tables/transcoder_feature_promotes.csv`,
   and `plots/transcoder_feature_cards.png` — FVU, splice-in KL, L0, and the
   de-embedded "promotes tokens" for inspected features. This is the explicit
   bridge to Lab 9: input→output objects give edges; site snapshots only give nouns.
10. `plots/truth_bridge_feature_cosines.png` + `tables/truth_bridge_feature_cosines.csv`
    when a compatible Lab 4 direction exists — use this to discuss distributed
    truth directions versus single SAE atoms.
11. `plots/feature_clamp.png`, `plots/clamp_operating_window.png`,
    `tables/feature_clamp.csv`, and `tables/clamp_operating_points.csv` — the single CAUSAL row. **Read the `sample`
    generations** at each dose (not just the hit count). The window is narrow;
    the random control and distinct-ratio column are part of the claim.
12. `tables/plot_reading_guide.csv` — a compact map from plot to concept if you
    are writing the run summary or claim ledger.

## Writeup questions

1. Which of your labels survived validation untouched, which needed narrowing,
   and which died? Quote the held-out AUC, confusable-pair AUC, purity,
   polysemy, and fire fraction for one survivor and one casualty. Use
   `plots/feature_validation_matrix.png` or `tables/feature_evidence_matrix.csv`,
   not only the top-context gallery. What did the dead one teach you?
2. Max-activation ranking vs frequency ranking: which produced more
   interpretable features, and why might that be? Point at specific feature ids
   in ranking_disagreement.png and feature_rankings.csv. What is the overlap on
   your top N?
3. In one paragraph: what does a transcoder reconstruct that an SAE does not,
   and why does Lab 9 need that? Use your splice-in KL (transcoder_reconstruction_report.json)
   as evidence that it reconstructs the *computation* (not just the vector).
4. **The truth-direction bridge.** Your best SAE feature aligns with Lab 4's
   truth direction at cosine ≈ ?. Argue what that does and does not imply about
   whether "truth" is a feature this SAE represents. (Low cosine is a finding,
   not a bug.)
5. **Real patterns (Dennett) redux.** Take your best-labeled feature and your
   worst (from the atlas). Argue the best one is a *discovered concept*; then
   steelman the *deflationary* reading of the worst one ("a convenient
   coordinate the SAE found because the loss rewarded it"). Use the clamp
   result (feature_clamp.png + CSV samples): does causal sufficiency change the
   argument, or does the narrow window + repetition past 3× + random control
   still leave room for the deflationary reading?

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
