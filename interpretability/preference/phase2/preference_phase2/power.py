"""Power simulation (plan §32; addendum E additions).

Variance sources: Phase 1 frozen incidental-level components, estimated
from the frozen results (five incidentals — treated as noisy, so every
cell also runs at a 2x-variance stress level). The simulated test is the
E16 primary: exact incidental sign-flip, Holm within family.

Decision outputs feed the preregistration directly:
- required powers: >= 0.80 at 0.25-nat semantic margin; >= 0.80 at 0.10
  strict effect (else raise incidental counts before freeze);
- coupling non-margin endpoint power (addendum G): if < 0.80 for the
  strict-report shift, the RO margin contrast is designated the coupling
  primary (fallback pinned pre-outcome).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from . import paths
from .stats import SEED_BASE, holm, sign_flip_p_matrix
from .canonical import stable_seed

N_SIMS = 1000

MARGIN_GRID = (0.15, 0.25, 0.50, 1.00)
STRICT_GRID = (0.08, 0.10, 0.15, 0.20)
SLOPE_GRID = (0.10, 0.25, 0.50)
COUPLING_SHIFT_GRID = (0.10, 0.15, 0.25)
INVALID_RATE = 0.02


def phase1_variance_components() -> dict[str, Any]:
    """Incidental-level margin variance from the frozen Phase 1 record
    (AR enacted rows, both models).

    The within-incidental SD is SURFACE-RESIDUALIZED: the balanced fold
    cancels surface main effects exactly, so the noise that propagates to
    an incidental mean is the residual after (order, label, code map,
    frame, incidental FE) — not the raw cell SD, which Phase 1's position
    policy dominates. Between-incidental SD is bias-corrected for the
    sampling noise of five-incidental means.
    """
    out = {}
    for model in ("7b", "32b"):
        path = (paths.phase1_root() / "reports" / f"frozen_{model}"
                / "results.jsonl")
        by_scn: dict[str, list[dict]] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if (r.get("family") == "AR" and r.get("channel") == "AR"
                        and r.get("margin_pole1_minus_pole0") is not None):
                    by_scn.setdefault(r["scenario_id"], []).append(r)
        resids, sbs = [], []
        for scn, rows in by_scn.items():
            y = np.array([float(r["margin_pole1_minus_pole0"]) for r in rows])
            incs = sorted({r["incidental_id"] for r in rows})
            X = np.array([
                [1.0, float(r["order_index"]),
                 1.0 if r["display_label_set"] == "letters" else 0.0,
                 float(r["code_map_index"]),
                 1.0 if r["consequence_frame"] == "enacted" else 0.0]
                + [1.0 if r["incidental_id"] == i else 0.0
                   for i in incs[:-1]]
                for r in rows])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            sigma_r = float(np.sqrt(np.sum(resid ** 2)
                                    / (len(y) - X.shape[1])))
            resids.append(sigma_r)
            n_cells = len(y) / len(incs)
            means = [np.mean([float(r["margin_pole1_minus_pole0"])
                              for r in rows if r["incidental_id"] == i])
                     for i in incs]
            s_m = float(np.std(means, ddof=1))
            sbs.append(float(np.sqrt(max(0.0, s_m ** 2
                                         - sigma_r ** 2 / n_cells))))
        out[model] = {
            "sigma_resid_median": float(np.median(resids)),
            "sigma_resid_p90": float(np.quantile(resids, 0.9)),
            "sigma_between_median": float(np.median(sbs)),
            "sigma_between_p90": float(np.quantile(sbs, 0.9)),
            "n_scenarios": len(by_scn),
        }
    # conservative pins used by the simulation (between median and p90 of
    # the primary model; five Phase 1 incidentals make these noisy)
    out["pinned"] = {"sigma_between": 0.15, "sigma_resid": 0.60}
    return out


def _simulate_margin_family(*, mu: float, sigma_b: float, sigma_w: float,
                            n_inc: int, n_cells: int, n_scenarios: int,
                            n_sims: int, seed_key: str) -> float:
    """Power of the Holm-adjusted exact sign-flip family when every
    scenario carries the same true margin ``mu``."""
    rng = np.random.default_rng(stable_seed(seed_key, base=SEED_BASE))
    rejections = 0
    for sim in range(n_sims):
        ps = {}
        for s in range(n_scenarios):
            b = rng.normal(0.0, sigma_b, size=n_inc)
            e = rng.normal(0.0, sigma_w, size=(n_inc, n_cells))
            inc_means = mu + b + e.mean(axis=1)
            ps[f"s{s}"] = float(sign_flip_p_matrix(inc_means[:, None])[0])
        h = holm(ps)
        rejections += sum(v["reject_at_05"] for v in h.values())
    return rejections / (n_sims * n_scenarios)


def _simulate_strict_family(*, effect: float, icc_sd: float, n_inc: int,
                            n_cells: int, n_scenarios: int, n_sims: int,
                            seed_key: str) -> float:
    rng = np.random.default_rng(stable_seed(seed_key, base=SEED_BASE))
    rejections = 0
    for sim in range(n_sims):
        ps = {}
        for s in range(n_scenarios):
            logit_shift = rng.normal(0.0, icc_sd, size=n_inc)
            p_base = np.clip(0.5 + effect + logit_shift, 0.02, 0.98)
            valid = rng.random((n_inc, n_cells)) > INVALID_RATE
            choices = rng.random((n_inc, n_cells)) < p_base[:, None]
            with np.errstate(invalid="ignore"):
                rates = np.array([
                    choices[i][valid[i]].mean() if valid[i].any() else np.nan
                    for i in range(n_inc)])
            ps[f"s{s}"] = float(sign_flip_p_matrix(
                (rates - 0.5)[:, None])[0])
        h = holm(ps)
        rejections += sum(v["reject_at_05"] for v in h.values())
    return rejections / (n_sims * n_scenarios)


def _simulate_slope_family(*, slope: float, sigma_b: float, sigma_w: float,
                           n_inc: int, n_sims: int, seed_key: str) -> float:
    """Context-ladder slope power per anchor family (3 anchors, Holm)."""
    rng = np.random.default_rng(stable_seed(seed_key, base=SEED_BASE))
    strengths = np.repeat(np.array([-2, -1, 0, 1, 2], dtype=float), 8)
    sxx = np.sum((strengths - strengths.mean()) ** 2)
    rejections = 0
    for sim in range(n_sims):
        ps = {}
        for anchor in range(3):
            slopes = np.empty(n_inc)
            for i in range(n_inc):
                b_i = rng.normal(0.0, sigma_b * 0.5)
                m = ((slope + b_i) * strengths
                     + rng.normal(0.0, sigma_w, size=len(strengths)))
                slopes[i] = np.sum((strengths - strengths.mean()) * m) / sxx
            ps[f"a{anchor}"] = float(sign_flip_p_matrix(slopes[:, None])[0])
        h = holm(ps)
        rejections += sum(v["reject_at_05"] for v in h.values())
    return rejections / (n_sims * 3)


def _simulate_coupling_shift(*, shift: float, n_receivers: int,
                             n_clusters: int, n_sims: int,
                             seed_key: str) -> float:
    """Strict comparative-report rate shift between +d and -d conditions
    on holdout receivers (paired per receiver, clustered by incidental)."""
    rng = np.random.default_rng(stable_seed(seed_key, base=SEED_BASE))
    per = n_receivers // n_clusters
    rejections = 0
    for sim in range(n_sims):
        cluster_shift = rng.normal(0.0, 0.05, size=n_clusters)
        diffs = np.empty(n_clusters)
        for c in range(n_clusters):
            p_plus = np.clip(0.5 + shift / 2 + cluster_shift[c], 0.02, 0.98)
            p_minus = np.clip(0.5 - shift / 2 + cluster_shift[c], 0.02, 0.98)
            plus = rng.random(per) < p_plus
            minus = rng.random(per) < p_minus
            diffs[c] = plus.mean() - minus.mean()
        p = float(sign_flip_p_matrix(diffs[:, None])[0])
        if p < 0.05:
            rejections += 1
    return rejections / n_sims


def run_power_simulation(n_sims: int = N_SIMS) -> dict[str, Any]:
    comp = phase1_variance_components()
    sigma_b = comp["pinned"]["sigma_between"]
    sigma_w = comp["pinned"]["sigma_resid"]

    result: dict[str, Any] = {
        "n_sims": n_sims,
        "phase1_variance_components": comp,
        "design": {
            "arb3_incidentals": 16, "arb3_cells_per_incidental": 16,
            "mech_incidentals": 32, "mech_rows_per_incidental": 40,
            "coupling_receivers": 64, "coupling_clusters": 8,
        },
        "margin_power": {}, "strict_power": {}, "slope_power": {},
        "coupling_power": {},
    }
    for stress, tag in ((1.0, "base"), (2.0, "stress2x")):
        sb, sw = sigma_b * stress, sigma_w * stress
        result["margin_power"][tag] = {
            str(mu): _simulate_margin_family(
                mu=mu, sigma_b=sb, sigma_w=sw, n_inc=16, n_cells=16,
                n_scenarios=12, n_sims=n_sims,
                seed_key=f"pw-margin-{tag}-{mu}")
            for mu in MARGIN_GRID}
        result["strict_power"][tag] = {
            str(e): _simulate_strict_family(
                effect=e, icc_sd=0.05 * stress, n_inc=16, n_cells=16,
                n_scenarios=12, n_sims=n_sims,
                seed_key=f"pw-strict-{tag}-{e}")
            for e in STRICT_GRID}
        result["slope_power"][tag] = {
            str(s): _simulate_slope_family(
                slope=s, sigma_b=sb, sigma_w=sw, n_inc=32, n_sims=n_sims,
                seed_key=f"pw-slope-{tag}-{s}")
            for s in SLOPE_GRID}
        result["coupling_power"][tag] = {
            str(s): _simulate_coupling_shift(
                shift=s, n_receivers=64, n_clusters=8, n_sims=n_sims,
                seed_key=f"pw-coup-{tag}-{s}")
            for s in COUPLING_SHIFT_GRID}

    result["gates"] = {
        "margin_0.25_power_ok": result["margin_power"]["base"]["0.25"] >= 0.80,
        "strict_0.10_power_ok": result["strict_power"]["base"]["0.1"] >= 0.80,
        "coupling_strict_0.15_power": result["coupling_power"]["base"]["0.15"],
        "coupling_fallback_to_margin_primary":
            result["coupling_power"]["base"]["0.15"] < 0.80,
    }
    return result
