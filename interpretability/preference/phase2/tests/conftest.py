"""Shared fixtures: frozen bank rows + synthetic model-output injection.

Synthetic worlds run through the REAL bank rows and the REAL analysis
code paths; only the model outputs are injected (plan §54.3).
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

PKG = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from preference_phase2 import paths  # noqa: E402
from preference_phase2.canonical import stable_seed  # noqa: E402

RNG_BASE = 20260808


@pytest.fixture(scope="session")
def bank_rows() -> list[dict]:
    path = paths.data_root() / "pref2_bank.jsonl"
    if not path.exists():
        pytest.skip("frozen bank missing (run make_pref2_banks.py)")
    return [json.loads(l) for l in path.open(encoding="utf-8")]


@pytest.fixture(scope="session")
def bank_meta() -> dict:
    return json.loads((paths.data_root() / "pref2_bank.meta.json")
                      .read_text())


def synth_results(rows, *, p_a_fn=None, margin_fn=None, valid_fn=None,
                  seed_key="synth"):
    """Turn bank rows into result records with injected behavior.

    ``p_a_fn(row) -> P(choose semantic A)``; ``margin_fn(row) -> full
    margin A-B``; ``valid_fn(row) -> parse valid?``. Deterministic via
    content-derived seeds."""
    out = []
    for r in rows:
        rng = np.random.default_rng(
            stable_seed(seed_key, r["item_id"], base=RNG_BASE))
        valid = True if valid_fn is None else bool(valid_fn(r, rng))
        p_a = 0.5 if p_a_fn is None else float(p_a_fn(r))
        chose_a = bool(rng.random() < p_a)
        margin = (0.0 if margin_fn is None else float(margin_fn(r)))
        margin += float(rng.normal(0, 0.05))
        rec = dict(r)
        rec.update({
            "parse_status": "valid" if valid else "invalid",
            "parsed_sem": ("a" if chose_a else "b") if valid else None,
            "margin_full_a_minus_b": margin,
            "margin_first_a_minus_b": margin * 0.8,
            "wrong_branch_free": True,
        })
        out.append(rec)
    return out
