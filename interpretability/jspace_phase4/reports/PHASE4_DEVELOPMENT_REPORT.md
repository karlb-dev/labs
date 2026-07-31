# J-space Phase 4 development report

Status: live development synthesis through the OLMo-3 32B base and
3.0 Think own/common-lens points, 2026-07-31.

This document is a living report, not a frozen claim record. Every
result here uses known Phase 3 banks and a development cohort. It can
localize a training-trajectory pattern and expose coordinate-frame
sensitivity, but it cannot establish a binary lineage claim. No Phase 4
confirmatory or replication cell is licensed until the candidate
preregistration receives PI sign-off and a tagged freeze.

## Compute and provenance boundary

All model jobs ran on the NVIDIA RTX PRO 6000 Blackwell Server Edition.
Each producer hard-failed unless CUDA was visible in the same process,
performed an FP16 CUDA matrix-multiply smoke test before model load, and
recorded the GPU in its result envelope. The final common-lens and base
grids used about 81.1 GB of VRAM. Model compute never used CPU; CPU was
reserved for tests, hashes, statistics, figures, and document
compilation.

The Phase 3 release is imported immutably as
`p4-import-phase3-release-v1`. Native Phase 4 evidence is event-sourced
in `reports/evidence_events.jsonl`. Registered artifacts are never
overwritten.

## OLMo-3 32B Think capability gate

Live evidence: `p4-g5-bank-olmo3-think-dev-v2`.

Prospective prefix-disjoint accepted-alias scoring produced 972 rows,
324 facts, and 72 families.

| View | Capability |
|---|---:|
| Overall | 0.6430 |
| Bank F | 0.4935 |
| Bank S | 0.8972 |
| Direct | 0.6574 |
| Composed | 0.4846 |
| Bridge supplied | 0.7870 |

The lineage cohort was fixed before interventions by requiring both
direct and composed generation capability within the same fact:
41/204 Bank-F facts and 85/120 Bank-S facts, or 126 facts and 252
direct/composed items.

The original G5 attempt stopped after 180 rows when accent folding made
`Río` and `Rio` ambiguous. The repaired canonical selector prefers exact
spelling, then permits only a unique normalized fallback. The partial
attempt is preserved; v2 is the live result.

## OLMo-3 32B base capability and intervention point

Capability evidence: `p4-g5-bank-olmo3-base-dev-v1`.

Raw grid: `p4-lineage-grid-olmo3-base-dev-v1`.

Analysis: `p4-lineage-analysis-olmo3-base-dev-v1`.

The exact pinned base revision is
`c2b61dae89a1ad10e4ad5653d0e46b590902607b`. Prospective G5 scoring
produced 972 rows, 324 facts, and 72 families. Overall capability was
0.6101 (Bank F 0.4395; Bank S 0.9000). Requiring both direct and
composed capability fixed a pre-intervention cohort of 23 Bank-F facts
and 88 Bank-S facts, or 111 facts and 222 items.

The seven-condition grid passed its strict acceptance audit: baseline
replay drift was at most `6.71e-7` nats, span-safe selected/protected
overlap was exactly zero, matched rank agreement was 100%, and maximum
matched energy relative error was `0.000260`.

![OLMo-3 base development point](figures/p4f03_olmo3_base_development.png)

Family-weighted J-specific effects were near zero in every primary
cell:

| Cell | Estimate (nats) | Family-bootstrap 95% interval |
|---|---:|---:|
| Bank F direct | +0.0245 | [−0.0330, +0.0792] |
| Bank F composed | +0.0145 | [−0.1226, +0.1881] |
| Bank S direct | +0.0005 | [−0.0487, +0.0481] |
| Bank S composed | +0.0021 | [−0.0309, +0.0385] |

The base Bank-S composed-minus-direct contrast was `+0.0016`
`[−0.0417,+0.0403]`. This primary null does not reflect an inert
experiment: the overall label-protected J-specific effect was `+0.1231`
`[+0.0667,+0.1802]`, while mechanics-random and logit-protected effects
were respectively `−0.1048` and `−0.2023`, with intervals below zero.

## Think own-lens intervention point

Raw evidence: `p4-lineage-grid-olmo3-think-dev-v1`.

Analysis: `p4-lineage-analysis-olmo3-think-dev-v1`.

The grid scored baseline, span-safe J, exact rank/energy control,
label-protected J, protected-energy control, mechanics-random control,
and logit-space label protection over every accepted alias. All 252
items are paired. Baseline replay drift is at most `7.15e-7` nats,
matched rank agreement is 100%, matched energy relative error is at most
`0.000230`, and span-safe selected/protected overlap is zero.

![OLMo-3 Think development point](figures/p4f01_olmo3_think_development.png)

Family-weighted J-specific effects are:

| Cell | Estimate (nats) | Family-bootstrap 95% interval |
|---|---:|---:|
| Bank F direct | +0.0574 | [−0.0442, +0.1785] |
| Bank F composed | +0.0818 | [−0.0223, +0.1878] |
| Bank S direct | −0.1277 | [−0.2072, −0.0494] |
| Bank S composed | −0.0553 | [−0.0982, −0.0187] |

Bank-S composed-minus-direct specificity is `+0.0724`, with an interval
that crosses zero. This is compatible with the unusual positive
composition term seen at 3.1 Think beginning by 3.0, but it is not a
resolved anomaly. The planned controlled Bank-W redundancy and
internal-derivation axes must adjudicate it.

## Common-base-lens audit and repair

Final raw evidence:
`p4-lineage-grid-olmo3-think-common-base-lens-dev-v3`.

The common coordinate is the frozen OLMo base lens, SHA-256
`92f32e38dc4dffc45dda4e0c34a75f5433238f2046ae00046a4fe3fe1226b696`.

Two failures were useful:

1. V1 revealed rank-limited protected-energy sites where a requested
   in-span/out-of-span component was omitted without marking the site
   clamped.
2. V2 repaired that metadata, but changing the evidence ID also changed
   every matched/random scientific seed.

V3 freezes an explicit scientific-seed namespace to the original v1
namespace. Its acceptance audit shows:

- all seven aggregate outcome columns match v1 exactly, maximum
  absolute difference `0.0`;
- every per-alias score and condition order matches v1 exactly;
- 296 protected-energy alias summaries are correctly reclassified;
- maximum unclamped energy relative error falls from the erroneous
  `0.511283` diagnostic to `0.000239`;
- rank agreement remains 100%.

V3 supersedes both withdrawn attempts. The repair changes metadata, not
the scientific outcomes.

## Own lens versus common base lens — paired repair pending

Withdrawn analysis: `p4-lens-frame-analysis-olmo3-think-dev-v1`.

The own-lens and common-lens raw grids each remain valid standalone
development evidence. Their baseline rows match, but their scientific
seed namespaces differ:

- own: `p4-lineage-grid-olmo3-think-dev-v1`;
- common: `p4-lineage-grid-olmo3-think-common-base-lens-dev-v1`.

Consequently, the matched-control random subspaces differ across
frames. For Bank F composed, the family-weighted control shift alone is
`+0.0616` nats, with its interval above zero. The previous
common-minus-own specificity estimates and correlations therefore mix
lens change with RNG change and must not be interpreted.

The standalone point estimates are retained only to describe the two
raw grids:

| Cell | Own lens | Common base lens |
|---|---:|---:|
| F direct | +0.0574 | −0.0403 |
| F composed | +0.0818 | −0.0656 |
| S direct | −0.1277 | −0.0861 |
| S composed | −0.0553 | −0.0464 |

A new common-lens v4 grid is configured with the own-grid scientific
seed namespace. The hardened paired-analysis producer refuses unequal
namespaces. No inferential frame comparison is live until the v4 grid
and v2 analysis pass their acceptance audits.

## Current interpretation

The base-to-3.0-Think contrast supports an estimation-first account:

- The base checkpoint is near zero in all four primary specificity
  cells, including both Bank-S variants.
- Bank-S direct and composed specificity is negative in each standalone
  3.0 Think grid. Formal coordinate-frame stability is pending the
  seed-paired rerun.
- Thus the known-bank development data localize the emergence of the
  Bank-S negative effect to the base-to-Think training interval. The
  cohorts differ because capability was fixed separately at each
  checkpoint, so this is a trajectory localization, not a paired causal
  estimate.
- Bank-F standalone point estimates differ by coordinate frame, but the
  prior paired inference is withdrawn because its controls used
  different RNG streams.
- The positive Bank-S composition tendency is not yet precise and must
  not be promoted before a controlled load/redundancy experiment.

These observations strengthen the need for seed-paired common-lens and
own-lens trajectory lines. They do not license “lineage present” or
“lineage absent.”

## Next boundary

The pinned base checkpoint is complete and banked. First repair the 3.0
Think common-lens grid with the own-grid scientific seed namespace and
replace the withdrawn paired analysis. Then run the exact 3.1 Think and
3.1 Instruct pair through the Phase 4 prospective G5 gate and the same
seven-condition intervention grid, explicitly sharing RNG namespaces
between each own/common pair. Finally synthesize the full
base → 3.0 Think → 3.1 Think/Instruct trajectory.

If the two trajectory views disagree, the next priority is the
fit-size/corpus study before further causal interpretation. Confirmatory
Phase 4 remains blocked on preregistration freeze and untouched data.
