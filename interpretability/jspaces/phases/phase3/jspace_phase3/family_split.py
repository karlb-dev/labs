# family_split_v2 — the deterministic, SEED-ACTIVE family splitter
# (nextsteps §5.6). Phase 2's split recorded a seed that a dead branch
# never used (p2-partition-seed-clarification-v1); this one runs seeded
# randomized restarts + seeded pair-swap hill-climbing, and its tests
# assert the seed actually moves the assignment on symmetric inputs.
#
# Input: a family table, one row per canonical family, with
#   canonical_family, bank, relation_group, n_items, n_counterfactual,
#   n_variant_<v> counts, intersection_capable (bool: >=1 bundle whose
#   direct AND composed variants are capable on ALL primary models),
#   plus numeric balance columns (per-model median baseline lp, answer
#   token lengths, bridge lengths...) — the builder decides the exact
#   numeric set; the splitter balances every numeric column it is given
#   and the categorical columns (bank, relation_group).
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitConstraints:
    min_twohop_families_per_side: int = 36
    min_intersection_families_per_side: int = 25
    max_standardized_imbalance: float = 0.35
    max_family_item_share: float = 0.10
    seed: int = 0
    restarts: int = 32
    sweeps: int = 200


@dataclass(frozen=True)
class Partition:
    confirmatory: tuple[str, ...]
    replication: tuple[str, ...]
    balance_report: dict
    seed: int

    def side_of(self, family: str) -> str:
        if family in self.confirmatory:
            return "confirmatory"
        if family in self.replication:
            return "replication"
        raise KeyError(family)

    def assert_disjoint(self, items: pd.DataFrame,
                        *cols: str) -> None:
        """No value of any identity column may appear on both sides.
        `items` carries one row per item with a canonical_family column
        plus the identity columns."""
        sides = items.canonical_family.map(
            lambda f: self.side_of(f))
        for col in cols:
            if col == "canonical_family":
                both = set(self.confirmatory) & set(self.replication)
            else:
                a = set(items.loc[sides == "confirmatory", col])
                b = set(items.loc[sides == "replication", col])
                both = {x for x in (a & b) if x is not None and x == x}
            if both:
                raise AssertionError(
                    f"{col}: {len(both)} values cross the partition, "
                    f"e.g. {sorted(map(str, both))[:3]}")


def _imbalance(tab: pd.DataFrame, mask: np.ndarray,
               num_cols: list[str], cat_cols: list[str]) -> dict:
    rep = {}
    a, b = tab[mask], tab[~mask]
    for c in num_cols:
        sd = float(tab[c].std())
        d = abs(float(a[c].mean()) - float(b[c].mean()))
        rep[c] = round(d / sd, 4) if sd > 1e-12 else 0.0
    for c in cat_cols:
        pa = a[c].value_counts(normalize=True)
        pb = b[c].value_counts(normalize=True)
        cats = set(pa.index) | set(pb.index)
        rep[c] = round(0.5 * sum(abs(pa.get(k, 0.0) - pb.get(k, 0.0))
                                 for k in cats), 4)
    return rep


def _feasible(tab: pd.DataFrame, mask: np.ndarray,
              c: SplitConstraints) -> bool:
    for m in (mask, ~mask):
        side = tab[m]
        if (side.bank == "F").sum() < c.min_twohop_families_per_side:
            return False
        if side.intersection_capable.sum() \
                < c.min_intersection_families_per_side:
            return False
        share = side.n_items.max() / max(side.n_items.sum(), 1)
        if share > c.max_family_item_share:
            return False
    return True


def split_families_v2(tab: pd.DataFrame,
                      constraints: SplitConstraints) -> Partition:
    """Seeded randomized-restart hill-climb over family assignments.
    Deterministic given (table, constraints); the seed is ACTIVE (it
    drives both the restart initializations and swap proposals)."""
    tab = tab.sort_values("canonical_family").reset_index(drop=True)
    num_cols = [c for c in tab.columns
                if tab[c].dtype.kind in "if" and c != "intersection_capable"]
    cat_cols = [c for c in ("bank", "relation_group") if c in tab.columns]
    n = len(tab)
    rng = np.random.default_rng(constraints.seed)

    def score(mask):
        rep = _imbalance(tab, mask, num_cols, cat_cols)
        rep["n_families"] = round(
            abs(int(mask.sum()) - int((~mask).sum())) / n, 4)
        return sum(rep.values()), rep

    best = None
    for _ in range(constraints.restarts):
        mask = np.zeros(n, dtype=bool)
        mask[rng.permutation(n)[:n // 2]] = True
        s, _ = score(mask)
        for _ in range(constraints.sweeps):
            i = int(rng.integers(n))
            j = int(rng.integers(n))
            was_move = bool(mask[i] == mask[j])
            if was_move:
                mask[i] = not mask[i]        # move i to the other side
            else:
                mask[i], mask[j] = mask[j], mask[i]   # swap
            s2, _ = score(mask)
            if s2 < s and _feasible(tab, mask, constraints):
                s = s2
            else:                            # undo the recorded action
                if was_move:
                    mask[i] = not mask[i]
                else:
                    mask[i], mask[j] = mask[j], mask[i]
        if _feasible(tab, mask, constraints) and \
                (best is None or s < best[0]):
            best = (s, mask.copy())
    if best is None:
        raise RuntimeError(
            "no feasible split: floors or item-share cap unsatisfiable "
            "with this family table")
    s, mask = best
    total, rep = score(mask)
    worst = max(rep.values()) if rep else 0.0
    if worst > constraints.max_standardized_imbalance:
        raise RuntimeError(
            f"best split's worst imbalance {worst} exceeds "
            f"{constraints.max_standardized_imbalance}: {rep}")
    conf = tuple(tab.canonical_family[mask])
    repl = tuple(tab.canonical_family[~mask])
    report = {"objective": round(total, 4), "per_dimension": rep,
              "n_confirmatory": len(conf), "n_replication": len(repl),
              "items_confirmatory": int(tab.n_items[mask].sum()),
              "items_replication": int(tab.n_items[~mask].sum()),
              "twohop_F": [int((tab.bank[mask] == "F").sum()),
                           int((tab.bank[~mask] == "F").sum())],
              "intersection": [int(tab.intersection_capable[mask].sum()),
                               int(tab.intersection_capable[~mask].sum())],
              "constraints": constraints.__dict__}
    return Partition(conf, repl, report, constraints.seed)
