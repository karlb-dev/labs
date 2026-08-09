"""Shared statistical machinery (addendum E pins).

Primary p-values: incidental-level EXACT sign-flip tests (E16). With 16
incidentals the full 2^16 enumeration is exact (min two-sided p ~ 3.1e-5);
above ``n_max_exact`` incidentals a seeded Monte Carlo flip stands in.
CIs: hierarchical bootstrap (incidentals, then cells). Holm within each
preregistered family.
"""

from __future__ import annotations

import numpy as np

from .canonical import stable_seed

SEED_BASE = 2262  # phase 2 base (phase 1 used 1238)
N_BOOT = 10_000
N_MC_FLIPS = 10_000


def _sign_matrix(n: int) -> np.ndarray:
    """All 2^n sign vectors as (2^n, n) int8. Cached per n."""
    if n not in _sign_matrix._cache:
        bits = np.arange(2 ** n, dtype=np.uint32)
        mat = ((bits[:, None] >> np.arange(n)) & 1).astype(np.int8)
        _sign_matrix._cache[n] = (mat * 2 - 1)
    return _sign_matrix._cache[n]


_sign_matrix._cache = {}


def exact_sign_flip_p(values: np.ndarray, *, n_max_exact: int = 20,
                      seed_key: str = "signflip") -> float:
    """Two-sided sign-flip test on cluster-level values (E16)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return float("nan")
    observed = abs(v.sum())
    if n <= n_max_exact:
        signs = _sign_matrix(n)
        sums = np.abs(signs.astype(np.float64) @ v)
        return float(np.mean(sums >= observed - 1e-12))
    rng = np.random.default_rng(stable_seed(seed_key, n, base=SEED_BASE))
    signs = rng.choice((-1.0, 1.0), size=(N_MC_FLIPS, n))
    sums = np.abs(signs @ v)
    return float(np.mean(sums >= observed - 1e-12))


def sign_flip_p_matrix(values: np.ndarray, *, n_max_exact: int = 20,
                       seed_key: str = "signflip-matrix") -> np.ndarray:
    """Vectorized sign-flip over columns of a (n, k) matrix — one p per
    column. Exact 2^n enumeration for n <= n_max_exact (E16); seeded
    Monte Carlo flips above (the E-table's 2^32 -> 10k MC rule)."""
    v = np.asarray(values, dtype=np.float64)
    n = v.shape[0]
    if n <= n_max_exact:
        signs = _sign_matrix(n).astype(np.float64)
    else:
        rng = np.random.default_rng(stable_seed(seed_key, n, base=SEED_BASE))
        signs = rng.choice((-1.0, 1.0), size=(N_MC_FLIPS, n))
    sums = np.abs(signs @ v)                       # (draws, k)
    observed = np.abs(v.sum(axis=0))[None, :]      # (1, k)
    return np.mean(sums >= observed - 1e-12, axis=0)


def hierarchical_bootstrap(cluster_values: list[np.ndarray], *,
                           n_boot: int = N_BOOT,
                           seed_key: str = "hboot") -> np.ndarray:
    """Bootstrap draws of the grand mean: resample clusters with
    replacement, then rows within each sampled cluster."""
    rng = np.random.default_rng(
        stable_seed(seed_key, len(cluster_values), base=SEED_BASE))
    k = len(cluster_values)
    draws = np.empty(n_boot)
    arrays = [np.asarray(c, dtype=np.float64) for c in cluster_values]
    for b in range(n_boot):
        idx = rng.integers(0, k, size=k)
        vals = []
        for i in idx:
            c = arrays[i]
            take = rng.integers(0, len(c), size=len(c))
            vals.append(c[take])
        draws[b] = np.nanmean(np.concatenate(vals))
    return draws


def bootstrap_ci(draws: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    alpha = (1.0 - level) / 2.0
    return (float(np.nanquantile(draws, alpha)),
            float(np.nanquantile(draws, 1.0 - alpha)))


def holm(pvals: dict[str, float]) -> dict[str, dict[str, float | bool]]:
    """Holm step-down; NaN p-values never reject."""
    items = [(k, p) for k, p in pvals.items() if np.isfinite(p)]
    m = len(items)
    out: dict[str, dict[str, float | bool]] = {}
    running_max = 0.0
    running_reject = True
    for rank, (k, p) in enumerate(sorted(items, key=lambda kv: kv[1])):
        adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, adj)
        reject = running_reject and running_max < 0.05
        if not reject:
            running_reject = False
        out[k] = {"p": p, "p_holm": running_max, "reject_at_05": reject}
    for k, p in pvals.items():
        if not np.isfinite(p):
            out[k] = {"p": p, "p_holm": float("nan"), "reject_at_05": False}
    return out


def incidental_ols_slope(strengths: np.ndarray, margins: np.ndarray) -> float:
    """Per-incidental OLS slope of margin on signed context strength."""
    s = np.asarray(strengths, dtype=np.float64)
    m = np.asarray(margins, dtype=np.float64)
    ok = np.isfinite(s) & np.isfinite(m)
    s, m = s[ok], m[ok]
    if len(s) < 3 or np.std(s) == 0:
        return float("nan")
    return float(np.polyfit(s, m, 1)[0])
