# Lab 4: Probing Without Fooling Yourself (now featuring truth)

**Evidence level targeted:** decodability (`DECODE`), with controls. The
causal hand-off — does the model *use* what you can decode? — is deferred to
Lab 7, which loads the direction this lab saves. **Prerequisites:** Labs 1–3.

## The question

What is linearly decodable from the residual stream — including whether a
statement is **true** — and how do you stop yourself from confusing
"a probe found it" with "the model represents and uses it"?

This lab makes skepticism quantitative. Every accuracy you report comes with
its controls attached, in the same table and on the same plot axes.

## Two tracks, same forward passes

- **Surface track** (calibration): does the final word contain a particular
  letter? Pure token-surface trivia. Its by-layer curve shows what "trivially
  decodable" looks like, so the truth curve has something honest to be
  compared against.
- **Truth track** (headline): is the statement true? Probed Geometry-of-Truth
  style at the end-of-statement position, on three **frozen** families
  vendored in `data/` (course rule: no live downloads, students never author
  truth sets):
  - `cities` — "The city of Paris is in France."
  - `comparisons` — "Sixty-one is larger than fourteen."
  - `negations` — "The city of Paris is not in France." Labels are truth
    values, so surface co-occurrence **anti-correlates** with truth here.
    This family exists to catch probes that read templates instead of truth.

## Two probes, on purpose

- **Logistic regression** (torch LBFGS, L2, standardized features — read
  `fit_logistic`, it's 25 lines): finds *any* separating direction.
- **Mass-mean** (difference of class means): the direction Lab 7 will test
  causally, because mean-difference directions have repeatedly proven more
  causally relevant than max-margin ones.

They can disagree, and when they do the disagreement is data. In our Tier B
validation run they disagreed dramatically: logistic decodes truth from
mid-stack, while the mass-mean direction only "snaps in" near the final
layers. Separable-at-small-sample is not the same as
dominant-in-the-class-means. Write down which claim each probe licenses.

## The controls (the actual product)

| Control | Question it answers |
|---|---|
| shuffled-label refits | could this probe "find" structure in noise at this n and d? |
| random-direction baseline | how well does an arbitrary direction score? |
| token-length baseline | is "truth" secretly statement length? |
| family-held-out transfer | does cities-truth transfer to comparisons? to negations? |
| surface track | what does trivial decodability look like on these axes? |
| selectivity | real accuracy − shuffled-control accuracy |

## The outlier specimen (do not skip this)

On Olmo-3, one frozen statement — *"The city of Havana is in the
Netherlands."* — produces a final-position activation with **~7× the norm**
of every other statement. Before this lab normalized activation rows, that
single row hijacked the class-mean difference and silently pinned mass-mean
accuracy to chance at every layer, while logistic regression sailed on
unbothered. The fix (unit-normalize rows) is one line; the lesson is not:
**a probe pipeline can be broken by one rogue activation and still look like
a clean negative result.** The per-statement norms are recorded in
`tables/statement_manifest.csv` with an outlier flag — check yours before
believing anything else in the run.

## Running it

```bash
python interp_bench.py --lab lab4 --tier a                  # smoke (20/family)
python interp_bench.py --lab lab4 --tier b --prompt-set full  # all 180 statements
```

`--max-examples` is a **per-family** cap in this lab (balanced true/false).
`--prompt-set small|medium|full` selects 20/40/all per family.

## First artifact-reading path

1. `plots/decodability_by_layer.png` — surface vs truth vs controls, one
   figure, the whole lesson.
2. `plots/generalization_matrix.png` — both probes at the best layer. Find
   the negation row/column and explain every cell below 0.5 (an accuracy
   *below* chance is structure too — what structure?).
3. `plots/truth_projection_panels.png` — separation emerging over depth.
4. `plots/selectivity_by_layer.png` — how much accuracy is real structure.
5. `tables/probe_report.csv` — every number with its controls adjacent.
6. `tables/statement_manifest.csv` — splits, norms, the outlier flag.

## What gets saved for Lab 7

`tables/truth_direction.pt`: the mass-mean direction at the best
cross-family layer (selection excludes negation transfer — an
affirmative-trained direction failing on negations is the expected
Geometry-of-Truth result, reported as the known failure mode, not optimized
away). Metadata records the layer, normalization convention, and the three
transfer accuracies. Lab 7 will ask whether this direction is *usable*, which
is a different question from everything measured here.

## Writeup questions

1. At which layer does truth become decodable, and by which probe's
   standard? Quote accuracies with their shuffled controls.
2. The negation column of the generalization matrix: explain each cell, and
   say what a cell at 0.05 (not 0.5!) tells you about what the probe learned.
3. The surface track peaks where? Why is its curve shaped differently from
   the truth curve, and what does that difference buy your argument?
4. Your manifest's `norm_outlier` rows: which statements, and what happens
   to mass-mean accuracy if you re-run without row normalization?
   (One flag flip in the lab file — try it.)
5. State precisely what `DECODE` evidence does NOT license you to claim.
   Then write the claim you expect Lab 7 to test, with its falsifier.

## Symptom-first debugging

| Symptom | First place to look |
|---|---|
| Mass-mean at chance, logistic fine | `statement_manifest.csv` norm outliers; normalization off? |
| Everything at chance | wrong probe position — are you on the period token? (`state` dumps) |
| Truth accuracy 1.0 at layer 0 | leakage: your split put paired statements across train/eval |
| Negation transfer ≈ 0.5 exactly | fine — that's chance; ≈ 0.05 is anti-correlation, a different fact |
| `Frozen dataset missing` | re-checkout `data/`; do NOT regenerate per-run |

## What goes in the ledger

2–3 claims, all tagged `DECODE`. The drafted claims separate decodable /
selectively-decodable / generalizes-across-families — keep that hierarchy
when you edit. Any claim that smuggles "the model knows/uses" past a
`DECODE` tag will be retired in public during Lab 7, which is worse.
