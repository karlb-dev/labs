# Side experiment: Labs 4 & 5 at 32B — does the 7B story survive scale?

Date: 2026-06-12 · Model: `allenai/Olmo-3-1125-32B` (base, 64 layers,
d_model 5120, bf16 on one A100-80GB) · No lab-code changes: pure
`--model` override; every bench self-check (hook parity, lens, component
anatomy, decomposition, patch no-op) passed on first contact with the
architecture. Runs: `runs/lab5_olmo32b` (750s), `runs/lab4_olmo32b` (91s).
Baselines: the 7B run-3 Tier B runs.

## Lab 5 — causal tracing of factual recall

| Measure | 7B (32 layers) | 32B (64 layers) | Reading |
|---|---|---|---|
| localized stream depths | 19–21 → **0.59–0.66 fractional** | 26–28 → **0.41–0.44 fractional** | the recall band is NOT at a fixed fractional depth — it moves proportionally *earlier* at scale (absolute depth grew 19→26 while layers doubled) |
| top-patch mean recovery | 0.669 | 0.524 | a single-site patch recovers less at 32B — recall is more redundant / distributed |
| subject-role peak | recovery 1.00 @ depth 3 | 1.00 @ depth 4 | early-subject enrichment is scale-stable |
| last-position peak | 1.00 @ depth 32 (final) | 1.00 @ depth 64 (final) | late readout is scale-stable |
| wrong-position control | 0.176 | **−0.026** | controls get *cleaner* at scale |
| mismatched-pair control | 0.287 | 0.170 | same |
| facts surviving the baseline gate | (subset) | 16/16 base-template | the 32B knows all the facts |

**One-sentence claim (CAUSAL, scoped):** on this fact set and metric, scale
preserves the *shape* of factual recall (subject-early enrichment, band of
causal recovery, final-position readout) but shifts the band to a smaller
fraction of depth and spreads causal responsibility across more sites —
localization gets relatively earlier and individually weaker as the model
grows. Falsifier: a different fact family or wider patch windows showing
the 32B band at matched fractional depth or matched single-site recovery.

## Lab 4 — truth probing

| Measure | 7B | 32B | Reading |
|---|---|---|---|
| mass-mean peak accuracy | 0.963 @ layer 32/32 (final) | **1.000** @ layer 64/64 (final) | in-family truth decodability reaches ceiling at scale |
| shuffled-label control | 0.444 | 0.500 | clean |
| saved-direction layer (chosen for transfer) | 32 (1.00 fractional) | 44 (**0.69 fractional**) | the *transferable* direction moves off the final layer |
| worst cross-family transfer | 0.817 | **0.700** | more decodable ≠ more universal: the single best direction generalizes WORSE across statement families at 32B |

**One-sentence claim (DECODE, scoped):** scale makes truth more linearly
readable in-family while making any single direction less family-general —
consistent with a richer, more factored truth representation that no one
direction spans. Falsifier: a multi-direction probe or different family
split closing the transfer gap.

## Why this matters for the course

- **The instrument generalizes.** A model 4.5× larger, never seen by the
  bench, ran both labs with zero code changes and zero failed self-checks.
  That is the payoff of anatomy-resolution-plus-verification over
  hardcoded paths.
- **Two ready-made extension assignments.** "Re-run your Lab 5 on the 32B
  and explain the fractional-depth shift" and "reconcile decodability-up
  with transfer-down in Lab 4" are exactly manageable-extension-sized, and
  both have falsifiers students can actually run.
- **A caution for Lab 9/Lab 11 claims.** Single-site recovery dropping
  with scale (0.67→0.52) says supernode-style interventions will need more
  sites on bigger models — worth a sentence in any cross-scale claim.

## Archive

Drive: `interpret/experiments/olmo32b/` (both run directories + this note).
