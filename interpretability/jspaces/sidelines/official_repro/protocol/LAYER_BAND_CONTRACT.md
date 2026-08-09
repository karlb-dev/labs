# Layer and band contract (frozen pre-data)

Both study models: 64 layers, final readout layer 63. Grids live as
committed literals in `jspace_official_repro/layers.py`, asserted at
import against their provenance formulas (addendum §2.6).

## Paper grid (25 layers)

```
[0, 3, 5, 8, 10, 13, 16, 18, 21, 24, 26, 29, 32,
 34, 37, 39, 42, 45, 47, 50, 52, 55, 58, 60, 63]
```

Provenance: `[round(i*63/24) for i in range(25)]` under **Python-3
banker's rounding** (10.5→10, 31.5→32, 52.5→52). NumPy float rounding or
round-half-up yields a different grid at i ∈ {4, 12, 20}; no
reimplementation may "fix" the list. Layer 63 is the target/final readout
and is never an OLMo source-layer fit.

## Primary paper-relative workspace band (14 layers)

```
[24, 26, 29, 32, 34, 37, 39, 42, 45, 47, 50, 52, 55, 58]
```

Provenance: the paper's normalized 38–92 interval → 0.38·63 = 23.94 and
0.92·63 = 57.96, snapped to grid members 24 and 58. The paper itself
describes the late workspace/motor boundary as partly post-hoc; this band
mapping is a **preregistered cross-model convention, not a discovered
anatomical fact**, and every report must say so.

## Campaign cross-walk band (13 layers; named sensitivity only)

```
[20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]
```

Never a post-outcome replacement for the paper band. No adaptive
0.8-of-maximum band may become primary. Per-layer curves are always
retained.

## Lens-eval layer sampling (plan §6.3)

Primary cross-model evaluation: the **24 non-final paper-grid source
layers** on both lanes, min-over-layers rank at the exact readout
position. Qwen all-source-layer (0–62) results are a named sensitivity
and are never compared to OLMo's 24-layer primary as if search
opportunities matched.

## OLMo fit layers (32 = paper-grid sources ∪ campaign band)

```
[0, 3, 5, 8, 10, 13, 16, 18, 20, 21, 22, 24, 26, 28, 29, 30, 32, 34,
 36, 37, 38, 39, 40, 42, 44, 45, 47, 50, 52, 55, 58, 60]
```

The 8 extra non-paper-grid layers exist only to make the frozen
campaign-band cross-over possible without a second fit; primary
released-material results search only the paper-grid subset.

## Fair frozen-campaign-lens comparisons (plan §2.3)

- Readout geometry: exact nine-layer intersection
  `[8, 16, 24, 26, 32, 34, 42, 52, 60]`.
- Causal cross-over: complete campaign band (both lenses contain every
  layer there).
- Never give the frozen lens fewer or more layer-search opportunities
  without labeling the difference.
