"""Frozen layer grids and bands (plan §6.2, addendum §2.6).

Both study models have 64 layers. The committed literal grids are
authoritative; the formulas are recorded provenance. ``round`` is Python-3
banker's rounding — NumPy float rounding or round-half-up gives a different
grid at i in {4, 12, 20}, and no later reimplementation may "fix" the list.
"""
from __future__ import annotations

N_LAYERS = 64
FINAL_LAYER = 63  # target/final readout; never an OLMo source-layer fit

#: 25 evenly spaced residual layers, paper's [0,100] grid mapped to 0..63:
#: [round(i*63/24) for i in range(25)] under banker's rounding.
PAPER_GRID = [
    0, 3, 5, 8, 10, 13, 16, 18, 21, 24, 26, 29, 32,
    34, 37, 39, 42, 45, 47, 50, 52, 55, 58, 60, 63,
]

#: The 24 non-final paper-grid source layers (lens evals, OLMo primary subset).
PAPER_GRID_SOURCES = [l for l in PAPER_GRID if l != FINAL_LAYER]

#: Primary paper-relative workspace band: normalized 38-92 -> 0.38*63=23.94,
#: 0.92*63=57.96, snapped to nearest sampled grid endpoints 24 and 58.
PAPER_BAND = [24, 26, 29, 32, 34, 37, 39, 42, 45, 47, 50, 52, 55, 58]

#: Campaign cross-walk band (sensitivity only; never a primary replacement).
CAMPAIGN_BAND = [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]

#: OLMo fit source layers: frozen union of the paper grid (minus final) and
#: the campaign band; the 8 extra layers exist only so the frozen-lens
#: cross-over runs without a second fit.
OLMO_FIT_SOURCE_LAYERS = sorted(set(PAPER_GRID_SOURCES) | set(CAMPAIGN_BAND))

#: Exact paper-grid intersection with the frozen campaign OLMo lens's fitted
#: layers — the only fair readout-geometry comparison set (plan §2.3).
CAMPAIGN_OLMO_LENS_LAYERS = [
    4, 8, 12, 16, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38,
    40, 42, 44, 48, 52, 56, 60,
]
PAPER_CAMPAIGN_INTERSECTION = [8, 16, 24, 26, 32, 34, 42, 52, 60]


def _verify_frozen_grids() -> None:
    assert PAPER_GRID == [round(i * 63 / 24) for i in range(25)]
    assert PAPER_BAND == [l for l in PAPER_GRID if 24 <= l <= 58]
    assert len(OLMO_FIT_SOURCE_LAYERS) == 32
    assert PAPER_CAMPAIGN_INTERSECTION == sorted(
        set(PAPER_GRID) & set(CAMPAIGN_OLMO_LENS_LAYERS)
    )
    assert FINAL_LAYER not in OLMO_FIT_SOURCE_LAYERS


_verify_frozen_grids()
