# Phase 3 statistical plan (nextsteps §14): paired contrasts first,
# randomization tests that respect the design, wild cluster bootstrap as
# the small-cluster robustness check, hierarchical models as secondary.
#
# Everything here is deterministic given (data, seed) and CPU-only.
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


# ------------------------------------------------------------- §14.1
def paired_specific_effects(df: pd.DataFrame, *,
                            j_condition: str,
                            control_condition: str,
                            baseline_condition: str = "baseline",
                            lp_col: str = "lp_logsumexp") -> pd.DataFrame:
    """Long rows (fact_id, canonical_family, model, variant, condition,
    lp) -> per (fact, model, variant): J_effect, C_effect, specific."""
    need = {"fact_id", "canonical_family", "model", "variant", "condition",
            lp_col}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}")
    piv = df.pivot_table(index=["fact_id", "canonical_family", "model",
                                "variant"],
                         columns="condition", values=lp_col,
                         aggfunc="first").reset_index()
    for c in (j_condition, control_condition, baseline_condition):
        if c not in piv.columns:
            raise ValueError(f"condition {c!r} absent")
    piv["J_effect"] = piv[j_condition] - piv[baseline_condition]
    piv["C_effect"] = piv[control_condition] - piv[baseline_condition]
    piv["specific"] = piv["J_effect"] - piv["C_effect"]
    return piv


def within_fact_composition(effects: pd.DataFrame,
                            *, value_col: str = "specific",
                            composed: str = "composed",
                            direct: str = "direct") -> pd.DataFrame:
    """specific(composed) - specific(direct) per (fact, model)."""
    piv = effects.pivot_table(index=["fact_id", "canonical_family", "model"],
                              columns="variant", values=value_col,
                              aggfunc="first").reset_index()
    if composed not in piv.columns or direct not in piv.columns:
        raise ValueError("need both composed and direct variants")
    piv["composition_penalty"] = piv[composed] - piv[direct]
    return piv.dropna(subset=["composition_penalty"])


def within_fact_model_diff(comp: pd.DataFrame, *, model_a,
                           model_b) -> pd.DataFrame:
    """composition_penalty difference per fact: A minus B. model_b may be
    a list — then B is the mean of those models' penalties (the P3-P1
    'Qwen minus OLMo-pair mean' statistic)."""
    piv = comp.pivot_table(index=["fact_id", "canonical_family"],
                           columns="model", values="composition_penalty",
                           aggfunc="first")
    bs = [model_b] if isinstance(model_b, str) else list(model_b)
    cols = [model_a, *bs]
    piv = piv.dropna(subset=cols)
    out = piv.reset_index()[["fact_id", "canonical_family"]]
    out["diff"] = (piv[model_a].to_numpy()
                   - piv[bs].to_numpy().mean(axis=1))
    return out


def family_means(t: pd.DataFrame, value_col: str,
                 family_col: str = "canonical_family") -> pd.Series:
    return t.groupby(family_col)[value_col].mean()


# ------------------------------------------------------------- §14.2
def family_signflip_test(family_vals: np.ndarray, *, draws: int = 100_000,
                         seed: int = 4242,
                         alternative: str = "two-sided") -> dict:
    """Sign-flip randomization on family-level paired statistics under
    the sharp null of symmetric zero effect. Exact enumeration when
    2^m <= draws, else Monte Carlo."""
    v = np.asarray(family_vals, dtype=float)
    v = v[~np.isnan(v)]
    m = len(v)
    if m < 3:
        raise ValueError(f"only {m} families — a randomization test on "
                         f"<3 clusters is vacuous (the two-family lesson)")
    obs = float(v.mean())
    if 2 ** m <= draws:
        # exact: iterate sign patterns via bits
        stats = np.empty(2 ** m)
        for i in range(2 ** m):
            signs = 1 - 2 * ((i >> np.arange(m)) & 1)
            stats[i] = (v * signs).mean()
        exact = True
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=(draws, m))
        stats = (signs * v).mean(axis=1)
        exact = False
    if alternative == "two-sided":
        extreme = int(np.count_nonzero(
            np.abs(stats) >= abs(obs) - 1e-15))
    elif alternative == "less":
        extreme = int(np.count_nonzero(stats <= obs + 1e-15))
    elif alternative == "greater":
        extreme = int(np.count_nonzero(stats >= obs - 1e-15))
    else:
        raise ValueError(alternative)
    p = (float(extreme / len(stats)) if exact
         else float((extreme + 1) / (len(stats) + 1)))
    return {"estimate": obs, "p": p, "n_families": m, "exact": exact,
            "alternative": alternative, "n_randomizations": int(len(stats))}


def exact_signflip_distribution(family_vals: np.ndarray) -> np.ndarray:
    """All 2**m sign-flipped family means, in deterministic bit order."""
    v = np.asarray(family_vals, dtype=float)
    v = v[~np.isnan(v)]
    m = len(v)
    if m < 3:
        raise ValueError(f"only {m} families")
    if m > 22:
        raise ValueError(
            f"exact enumeration would require 2**{m} patterns")
    bits = np.arange(2**m, dtype=np.uint32)[:, None]
    signs = (
        1 - 2 * ((bits >> np.arange(m, dtype=np.uint32)) & 1)
    ).astype(np.int8)
    return (signs @ v) / m


def exact_signflip_test(family_vals: np.ndarray,
                        *, alternative: str = "two-sided") -> dict:
    """Exact family sign-flip test, independent of a Monte Carlo budget."""
    v = np.asarray(family_vals, dtype=float)
    v = v[~np.isnan(v)]
    stats = exact_signflip_distribution(v)
    obs = float(v.mean())
    if alternative == "two-sided":
        extreme = np.abs(stats) >= abs(obs) - 1e-15
    elif alternative == "less":
        extreme = stats <= obs + 1e-15
    elif alternative == "greater":
        extreme = stats >= obs - 1e-15
    else:
        raise ValueError(alternative)
    return {
        "estimate": obs,
        "p": float(extreme.mean()),
        "extreme_patterns": int(extreme.sum()),
        "n_patterns": int(len(stats)),
        "n_families": int(len(v)),
        "alternative": alternative,
        "distribution_sha256": hashlib.sha256(
            np.asarray(stats, dtype="<f8").tobytes()).hexdigest(),
    }


def signflip_confidence_set(family_vals: np.ndarray, *, alpha: float = 0.05,
                            grid_points: int = 4001,
                            width_sd: float = 5.0) -> dict:
    """Invert the exact shifted sign-flip test on a dense effect grid.

    At candidate effect ``theta``, the test sign-flips ``v - theta`` and
    compares its mean with the observed mean of ``v - theta``. The returned
    set is randomization-compatible; it is not a bootstrap interval.
    """
    v = np.asarray(family_vals, dtype=float)
    v = v[~np.isnan(v)]
    m = len(v)
    if m < 3 or m > 22:
        raise ValueError(f"exact inversion requires 3..22 families, got {m}")
    obs = float(v.mean())
    sd = float(v.std(ddof=1))
    if sd == 0:
        return {
            "confidence_set": [obs, obs], "alpha": alpha,
            "grid_points": 1, "grid_resolution": 0.0,
            "n_components": 1, "method": "exact-shifted-signflip-inversion",
        }
    grid = np.linspace(obs - width_sd * sd, obs + width_sd * sd,
                       grid_points)
    bits = np.arange(2**m, dtype=np.uint32)[:, None]
    signs = (
        1 - 2 * ((bits >> np.arange(m, dtype=np.uint32)) & 1)
    ).astype(np.int8)
    signed_v = signs @ v
    signed_n = signs.sum(axis=1, dtype=np.int32).astype(float)
    pvals = np.empty(grid_points)
    for start in range(0, grid_points, 32):
        theta = grid[start:start + 32]
        null = (
            signed_v[:, None] - signed_n[:, None] * theta[None, :]
        ) / m
        observed = np.abs(obs - theta)[None, :]
        pvals[start:start + len(theta)] = (
            np.abs(null) >= observed - 1e-15).mean(axis=0)
    keep = np.flatnonzero(pvals >= alpha)
    if not len(keep):
        confidence_set = None
        n_components = 0
    else:
        confidence_set = [float(grid[keep[0]]), float(grid[keep[-1]])]
        n_components = int(1 + np.count_nonzero(np.diff(keep) > 1))
    return {
        "confidence_set": confidence_set,
        "alpha": float(alpha),
        "grid_points": int(grid_points),
        "grid_range": [float(grid[0]), float(grid[-1])],
        "grid_resolution": float(grid[1] - grid[0]),
        "n_components": n_components,
        "endpoint_p": (
            [float(pvals[keep[0]]), float(pvals[keep[-1]])]
            if len(keep) else None
        ),
        "range_truncated": bool(
            len(keep) and (keep[0] == 0 or keep[-1] == grid_points - 1)),
        "method": "exact-shifted-signflip-inversion",
    }


def monte_carlo_pvalue(null: np.ndarray, observed: float, *,
                       alternative: str = "two-sided") -> float:
    """Plus-one Monte Carlo p-value; never returns a literal zero."""
    z = np.asarray(null, dtype=float)
    if alternative == "two-sided":
        extreme = np.count_nonzero(np.abs(z) >= abs(observed) - 1e-15)
    elif alternative == "greater":
        extreme = np.count_nonzero(z >= observed - 1e-15)
    elif alternative == "less":
        extreme = np.count_nonzero(z <= observed + 1e-15)
    else:
        raise ValueError(alternative)
    return float((int(extreme) + 1) / (len(z) + 1))


def within_item_label_exchange_tail(df: pd.DataFrame, *,
                                    delta_j_col: str = "delta_J",
                                    delta_c_col: str = "delta_C",
                                    family_col: str = "canonical_family",
                                    threshold: float = -1.0,
                                    draws: int = 20_000,
                                    seed: int = 4242,
                                    alternative: str = "greater") -> dict:
    """§14.2 J-vs-control tail: within item, exchanging the J/control
    labels is the sharp null; statistic = family-weighted mean of
    (hit_J - hit_C). Threshold, stratum and weights preserved."""
    t = df[[delta_j_col, delta_c_col, family_col]].dropna().copy()
    t["hd"] = ((t[delta_j_col] < threshold).astype(float)
               - (t[delta_c_col] < threshold).astype(float))
    fams = t[family_col].to_numpy()
    uf = np.unique(fams)
    if len(uf) < 3:
        raise ValueError(f"only {len(uf)} families")
    hd = t["hd"].to_numpy()

    def fam_weighted(vals):
        s = pd.Series(vals).groupby(fams).mean()
        return float(s.mean())

    obs = fam_weighted(hd)
    rng = np.random.default_rng(seed)
    n = len(hd)
    stats = np.empty(draws)
    for b in range(draws):
        flip = rng.random(n) < 0.5
        stats[b] = fam_weighted(np.where(flip, -hd, hd))
    p = monte_carlo_pvalue(stats, obs, alternative=alternative)
    return {"estimate": obs, "p": p, "n_items": n, "n_families": len(uf),
            "threshold": threshold, "alternative": alternative,
            "n_randomizations": int(draws)}


# ------------------------------------------------------------- §14.3
def wild_cluster_bootstrap_t(t: pd.DataFrame, value_col: str, *,
                             family_col: str = "canonical_family",
                             draws: int = 9999, seed: int = 4242) -> dict:
    """Wild cluster bootstrap-t (Rademacher weights on family residuals)
    for the family-weighted mean — the §14.3 small-cluster check."""
    fm = t.groupby(family_col)[value_col].mean()
    m = len(fm)
    if m < 3:
        raise ValueError(f"only {m} families")
    obs = float(fm.mean())
    se = float(fm.std(ddof=1) / np.sqrt(m))
    if se == 0:
        return {"estimate": obs, "p": float(obs != 0.0), "n_families": m}
    t_obs = obs / se
    resid = fm.to_numpy() - obs
    rng = np.random.default_rng(seed)
    tstats = np.empty(draws)
    for b in range(draws):
        w = rng.choice((-1.0, 1.0), size=m)
        vb = obs + resid * w                     # impose the null via centering
        mb = vb.mean()
        sb = vb.std(ddof=1) / np.sqrt(m)
        tstats[b] = (mb - obs) / max(sb, 1e-30)
    p = float((np.abs(tstats) >= abs(t_obs)).mean())
    return {"estimate": obs, "se": se, "t": t_obs, "p": p, "n_families": m}


def wild_cluster_percentile_t_ci(
        t: pd.DataFrame, value_col: str, *,
        family_col: str = "canonical_family",
        alpha: float = 0.05, draws: int = 99_999,
        seed: int = 4242) -> dict:
    """Correctly studentized wild-cluster percentile-t interval.

    Rademacher weights are applied to unrestricted family residuals. For at
    most 20 families every sign pattern is enumerated; otherwise the supplied
    deterministic Monte Carlo budget is used.
    """
    fm = t.groupby(family_col)[value_col].mean().to_numpy(dtype=float)
    m = len(fm)
    if m < 3:
        raise ValueError(f"only {m} families")
    estimate = float(fm.mean())
    se = float(fm.std(ddof=1) / np.sqrt(m))
    if se == 0:
        return {
            "estimate": estimate, "se": se, "ci": [estimate, estimate],
            "alpha": alpha, "n_families": m, "n_randomizations": 0,
            "exact": True, "t_distribution_sha256": None,
            "method": "wild-cluster-percentile-t",
        }
    resid = fm - estimate
    if m <= 20:
        bits = np.arange(2**m, dtype=np.uint32)[:, None]
        weights = (
            1 - 2 * ((bits >> np.arange(m, dtype=np.uint32)) & 1)
        ).astype(np.int8)
        exact = True
    else:
        rng = np.random.default_rng(seed)
        weights = rng.choice((-1, 1), size=(draws, m)).astype(np.int8)
        exact = False
    boot = estimate + weights * resid
    boot_mean = boot.mean(axis=1)
    boot_se = boot.std(axis=1, ddof=1) / np.sqrt(m)
    tstats = (boot_mean - estimate) / np.maximum(boot_se, 1e-30)
    qlo, qhi = np.quantile(tstats, [alpha / 2, 1 - alpha / 2])
    ci = [float(estimate - qhi * se), float(estimate - qlo * se)]
    return {
        "estimate": estimate,
        "se": se,
        "ci": ci,
        "alpha": float(alpha),
        "n_families": int(m),
        "n_randomizations": int(len(tstats)),
        "exact": exact,
        "t_quantiles": [float(qlo), float(qhi)],
        "t_distribution_sha256": hashlib.sha256(
            np.asarray(tstats, dtype="<f8").tobytes()).hexdigest(),
        "method": "wild-cluster-percentile-t",
    }


def within_item_exchange_mean(df: pd.DataFrame, *, a_col: str,
                              b_col: str,
                              family_col: str = "canonical_family",
                              draws: int = 100_000, seed: int = 4242,
                              alternative: str = "two-sided") -> dict:
    """Within-item label exchange on a MEAN statistic: swapping the two
    arms' labels within an item flips the sign of d_i = a_i - b_i, so
    the randomization is an item-level sign flip; the statistic is the
    family-weighted mean of d (§14.2 bridge-rescue form, pinned to the
    mean in the Phase 3 prereg)."""
    t = df[[a_col, b_col, family_col]].dropna()
    d = (t[a_col] - t[b_col]).to_numpy(dtype=float)
    fams = pd.Categorical(t[family_col])
    codes = fams.codes
    m = len(fams.categories)
    n = len(d)
    counts = np.bincount(codes, minlength=m).astype(float)

    def stat(x):
        sums = np.bincount(codes, weights=x, minlength=m)
        return float((sums / counts).mean())

    obs = stat(d)
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(draws, n))
    null = np.array([stat(d * flips[b]) for b in range(draws)])
    p = monte_carlo_pvalue(null, obs, alternative=alternative)
    return {"estimate": round(obs, 6), "p": p,
            "n_items": int(n), "n_families": int(m),
            "alternative": alternative, "n_randomizations": int(draws)}


def leave_one_family_out(t: pd.DataFrame, value_col: str, *,
                         family_col: str = "canonical_family") -> pd.DataFrame:
    fm = t.groupby(family_col)[value_col].mean()
    rows = [{"left_out": f,
             "estimate": float(fm.drop(index=f).mean())} for f in fm.index]
    return pd.DataFrame(rows).sort_values("estimate")


def family_cluster_bootstrap_ci(t: pd.DataFrame, value_col: str, *,
                                family_col: str = "canonical_family",
                                draws: int = 4000, seed: int = 4242) -> dict:
    fm = t.groupby(family_col)[value_col].mean()
    fams = fm.index.to_numpy()
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for b in range(draws):
        pick = rng.choice(fams, size=len(fams), replace=True)
        samples[b] = fm.loc[pick].mean()
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {"estimate": float(fm.mean()), "ci95": [float(lo), float(hi)],
            "n_families": int(len(fams))}


# ------------------------------------------------------- test support
def plant_interaction(rng: np.random.Generator, *, n_families: int = 40,
                      facts_per_family: int = 4, effect: float = -0.8,
                      fam_sd: float = 0.4, noise_sd: float = 0.6,
                      models=("A", "B")) -> pd.DataFrame:
    """Synthetic generator: model A carries `effect` extra composed-minus-
    direct specific damage; family random intercepts + item noise. Used by
    the §15.3 'planted interaction recovered' test and the power sim."""
    rows = []
    for f in range(n_families):
        fam = f"fam{f:03d}"
        fam_eff = rng.normal(0, fam_sd)
        for i in range(facts_per_family):
            fid = f"{fam}:fact{i}"
            for model in models:
                for variant in ("direct", "composed"):
                    mu = 0.0
                    if model == models[0] and variant == "composed":
                        mu += effect
                    if variant == "composed":
                        mu += fam_eff
                    spec = mu + rng.normal(0, noise_sd)
                    rows.append({"fact_id": fid, "canonical_family": fam,
                                 "model": model, "variant": variant,
                                 "specific": spec})
    return pd.DataFrame(rows)
