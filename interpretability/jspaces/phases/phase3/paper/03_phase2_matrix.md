# 3 · The Phase 2 confirmatory matrix

All results in this section are tier `phase2-confirmatory` (or
`phase2-replication` where marked), from the locked analysis
[`n6-confirmatory-analysis-v2`] over the frozen partition
[`d5-partition-freeze-v1`]. **The limitation to state first:** the
confirmatory two-hop leg contains 9 items from 2 families. Every
composed-vs-direct claim below inherits that thinness; Phase 3's thick
paired bank exists because of it.

## 3.1 Primary outcomes

Both members of the preregistered Holm family reject:

- **P-HP1 (task-shape contrast):** composed-minus-direct damage
  contrast −0.504 nats, family-clustered 95% CI [−0.720, −0.295],
  p_holm = 5e-4.
- **P-HP3 (J-specific tail, Qwen):** protected-tail rate excess over
  the exact matched control +0.279 [+0.205, +0.361], p_holm = 5e-4.
  The OLMo tail estimates are CI-clean as estimates (HP3's binary test
  was preregistered as Qwen-only on power grounds: the audited-family
  MDE analysis showed the Think tail needs >150 families to *test*).

The geometry-matched control sits at ≈0 in every cell: at matched
per-position dose, direction content is the whole effect.

## 3.2 Held-out replication

The untouched 32-family replication partition ran the same day as the
confirmatory partition, same frozen pipeline
[`n6-replication-analysis-v2`]:

- **HP3 replicates:** Qwen tail-rate excess +0.297 [+0.207, +0.382]
  vs confirmatory +0.279, with a stable threshold-sensitivity curve.
- **HP1 is inconclusive on the held-out side:** +0.104
  [−1.681, +1.889]. This is *not* a contradiction (the CI comfortably
  contains −0.504); it is the thin-two-hop limitation made concrete —
  the replication two-hop leg is 9 items and absent from the
  intersection cohort, so the interaction is unmeasurable there.

## 3.3 Independent-lens robustness (OLMo legs)

The protected tail is a *model* property, not a fit accident: lenses
refit on disjoint corpora reproduce per-item deltas at r = 0.988
(Think) / 0.990 (Instruct) with tail Jaccard 0.878 / 0.883, prose pairs
excluded [`n6-repl-lens-independence-v2`]. Qwen runs the published
n=1000 lens; fit-size asymmetry against OLMo's n=120 lenses is a
carried, preregistered sensitivity, addressed by the symmetric fit-size
study in Phase 3's strong set.

## 3.4 Corrected capacity

Under the corrected estimand (centered R², the v1→v2 estimand repair),
J-space capacity separates the models sharply
[`r2-occupancy-*-v2`]:

| model | centered excess @ L24/32/40 | occupancy |
|---|---|---|
| OLMo 3.1 Think | 0.44 / 0.98 / 1.23% | 2 |
| OLMo 3.1 Instruct | 0.44 / 0.94 / 1.23% | 2 |
| Qwen 3.6 27B | 6.05 / 5.02 / 5.99% | 3–4 |

Qwen reaches the lower edge of the reported Claude band; the 3.1
post-training pair is flat at roughly a fifth of that. The pilot's
"all open models an order of magnitude below Claude" is withdrawn for
Qwen. Capacity is reported descriptively: with three assay-valid
models, no capacity-causes-shape claim is licensed (§4.5; R7).

## 3.5 The two-cluster picture

At confirmatory tier the pilot's four-rung ladder collapses into two
clusters:

- **OLMo (both 3.1 checkpoints):** accessibility-organized
  content-channel damage — items whose answers are *less* output-locked
  lose more (ρ(clean rank, Δ) = −0.30/−0.31, CI-clean on both
  primaries); one-hop-dominant on hard items; equal-depth damage on
  Instruct.
- **Qwen:** the paper-shaped forward dissociation (composed > direct),
  the replicating J-specific tail, and near-Claude-band capacity —
  with the *opposite* accessibility sign (+0.24).

The calibrated sentence from §0.2 (quoted in §1.3) is the strongest
summary these results support. What Phase 2 did **not** establish —
the robust composed-task workspace signature on Qwen, its absence on
OLMo, bridge mediation, capacity moderation — is exactly the Phase 3
preregistration's target list (§18.5 "awaiting Phase 3").
