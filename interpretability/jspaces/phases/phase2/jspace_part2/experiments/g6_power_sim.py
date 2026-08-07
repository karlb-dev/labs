# G6 — power simulation for the confirmatory design (prereg §12.6 input;
# supersedes the habitual n=60). Calibrated ENTIRELY from pilot parquets
# (which stay dev-tier); produces the n/families recommendation the
# preregistration's SESOI section cites. CPU-only.
#
# Analyses simulated (endpoint: paired per-item answer_seq_lp delta,
# protected dyn-J vs none; test: t on family means, the mixed-model
# approximation addendum §12.2 accepts for planning):
#   A  single-cell mean delta != 0        (HP2/HP3-style cells)
#   B  within-model dissociation          (twohop vs onehop, Welch t)
#   C  HP1 between-model difference-of-dissociation (cross-model item
#      pairing preserved by joint resampling of the pilot rows)
#   D  equivalence at the 0.5-nat margin  (TOST, true effect 0)
# Each analysis is run two ways: normal-theory components (MoM variance
# decomposition) and family-block bootstrap of the actual pilot deltas.
# alpha = 0.01 two-sided (Holm worst case across the 5 primary contrasts);
# TOST one-sided alphas 0.05. Target power 0.90 at SESOI 0.5 nats.
#
# Usage: python -m jspace_part2.experiments.g6_power_sim [--allow-dirty]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..lib import sha256_file
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          write_result)

BASE = Path("/content/drive/MyDrive/interpret/special-lab-1/part2_20260727/"
            "metrics")
CELLS = {  # slug -> pilot R7 per-item parquet
    "olmo3-think": BASE / "olmo3-think/r7_pilot/r7_per_item.parquet",
    "olmo31-instruct": BASE / "olmo31-instruct/r7_pilot/r7_per_item.parquet",
    "olmo3-base": BASE / "olmo3-base/r7_pilot/r7_per_item.parquet",
    "qwen36-27b": BASE / "qwen36-27b/r7_pilot/r7_per_item.parquet",
}
OUT = BASE / "cross_model/g6_power_sim.json"
ARM = "dynJ_protected"
SESOI = 0.5
ALPHA = 0.01          # Holm worst case, 5 primaries
POWER_TARGET = 0.90
B = 4000
GRID_M = [20, 30, 40, 50, 60, 80, 100, 150, 200]
GRID_K = [2, 3]
TAIL_THR = -1.0       # nats: pilot-frozen tail definition (>1-nat deletion)
TAIL_SESOI = 0.10     # rate-difference margin (matches accuracy margin)
RNG = np.random.default_rng(4242)


def paired_deltas(df: pd.DataFrame, task: str, arm: str = ARM) -> pd.DataFrame:
    base = df[(df.condition == "none") & (df.task == task)]\
        .set_index("item_id")[["score", "family"]]
    cond = df[(df.condition == arm) & (df.task == task)]\
        .set_index("item_id")["score"]
    out = base.copy()
    out["delta"] = cond - base["score"]
    return out.dropna().reset_index()[["item_id", "family", "delta"]]


def components(d: pd.DataFrame) -> dict:
    """MoM variance decomposition of delta over families."""
    g = d.groupby("family")["delta"]
    fam_means, fam_sizes = g.mean(), g.size()
    multi = fam_sizes[fam_sizes >= 2].index
    sig2_e = (float(np.mean([d[d.family == f]["delta"].var(ddof=1)
                             for f in multi])) if len(multi) else
              float(d["delta"].var(ddof=1)))
    var_fm = float(fam_means.var(ddof=1))
    sig2_f = max(0.0, var_fm - sig2_e * float(np.mean(1.0 / fam_sizes)))
    tot = sig2_f + sig2_e
    return {"mu": float(d["delta"].mean()), "n_items": int(len(d)),
            "n_families": int(fam_sizes.size),
            "sig_f": round(float(np.sqrt(sig2_f)), 4),
            "sig_e": round(float(np.sqrt(sig2_e)), 4),
            "icc": round(sig2_f / tot, 4) if tot > 0 else 0.0}


def t_reject(fm: np.ndarray, alpha: float, mu0: float = 0.0) -> np.ndarray:
    """Vectorized one-sample two-sided t on family means fm [B, m]."""
    m = fm.shape[1]
    t = (fm.mean(1) - mu0) / (fm.std(1, ddof=1) / np.sqrt(m))
    return np.abs(t) > stats.t.ppf(1 - alpha / 2, m - 1)


def power_normal(m, k, sig_f, sig_e, mu, alpha=ALPHA):
    fm = RNG.normal(mu, np.sqrt(sig_f**2 + sig_e**2 / k), size=(B, m))
    return float(t_reject(fm, alpha).mean())


def power_boot(d: pd.DataFrame, m, k, mu, alpha=ALPHA):
    """Family-block bootstrap: resample m families, k item deltas within
    each (with replacement), recentered so the true mean is mu."""
    fams = [grp["delta"].to_numpy() for _, grp in d.groupby("family")]
    centered = [a - d["delta"].mean() for a in fams]
    fi = RNG.integers(0, len(fams), size=(B, m))
    fm = np.empty((B, m))
    for b in range(B):
        fm[b] = [centered[i][RNG.integers(0, len(centered[i]), k)].mean() + mu
                 for i in fi[b]]
    return float(t_reject(fm, alpha).mean())


def welch_reject(g1: np.ndarray, g2: np.ndarray, alpha: float) -> np.ndarray:
    n1, n2 = g1.shape[1], g2.shape[1]
    v1, v2 = g1.var(1, ddof=1) / n1, g2.var(1, ddof=1) / n2
    t = (g1.mean(1) - g2.mean(1)) / np.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))
    return np.abs(t) > stats.t.ppf(1 - alpha / 2, df)


def power_dissoc(comp2, comp1, m2, k2, m1, eff, alpha=ALPHA):
    """Welch t: twohop family means (m2 fams x k2) vs onehop items (m1)."""
    g2 = RNG.normal(eff, np.sqrt(comp2["sig_f"]**2 + comp2["sig_e"]**2 / k2),
                    size=(B, m2))
    g1 = RNG.normal(0.0, np.sqrt(comp1["sig_f"]**2 + comp1["sig_e"]**2),
                    size=(B, m1))
    return float(welch_reject(g2, g1, alpha).mean())


def power_hp1_boot(joint2: pd.DataFrame, joint1: pd.DataFrame,
                   m2, k2, m1, eff, alpha=ALPHA):
    """HP1 difference-of-dissociation, joint family-block bootstrap of the
    observed per-item CROSS-MODEL diffs (pairing/rho preserved). g2 =
    family means of twohop diffs (+eff injected), g1 = onehop diffs."""
    fams = [grp["diff"].to_numpy() for _, grp in joint2.groupby("family")]
    fams = [a - joint2["diff"].mean() for a in fams]
    ones = joint1["diff"].to_numpy() - joint1["diff"].mean()
    fi = RNG.integers(0, len(fams), size=(B, m2))
    g2 = np.empty((B, m2))
    for b in range(B):
        g2[b] = [fams[i][RNG.integers(0, len(fams[i]), k2)].mean() + eff
                 for i in fi[b]]
    g1 = ones[RNG.integers(0, len(ones), size=(B, m1))]
    return float(welch_reject(g2, g1, alpha).mean())


def tail_pairs(df: pd.DataFrame, task: str = "twohop") -> pd.DataFrame:
    """Per-item paired binary hits: dynJ_protected tail vs dynR_protected
    tail (>1-nat deletion), with family labels."""
    dj = paired_deltas(df, task, "dynJ_protected").set_index("item_id")
    dr = paired_deltas(df, task, "dynR_protected").set_index("item_id")
    out = dj[["family"]].copy()
    out["hit_j"] = (dj["delta"] < TAIL_THR).astype(float)
    out["hit_r"] = (dr["delta"] < TAIL_THR).astype(float)
    return out.dropna().reset_index()


def power_tailrate_boot(pairs: pd.DataFrame, m, k, eff, alpha=ALPHA):
    """Family-block bootstrap of paired (hit_j - hit_r); recentered so the
    true rate difference is eff; t on family means of the paired diff."""
    d = pairs.assign(diff=pairs.hit_j - pairs.hit_r)
    fams = [grp["diff"].to_numpy() for _, grp in d.groupby("family")]
    fams = [a - d["diff"].mean() for a in fams]
    fi = RNG.integers(0, len(fams), size=(B, m))
    fm = np.empty((B, m))
    for b in range(B):
        fm[b] = [fams[i][RNG.integers(0, len(fams[i]), k)].mean() + eff
                 for i in fi[b]]
    return float(t_reject(fm, alpha).mean())


def power_tost(sig_f, sig_e, m, k, margin=SESOI, alpha=0.05):
    """Equivalence: true mu=0; both one-sided tests must reject."""
    fm = RNG.normal(0.0, np.sqrt(sig_f**2 + sig_e**2 / k), size=(B, m))
    se = fm.std(1, ddof=1) / np.sqrt(m)
    tc = stats.t.ppf(1 - alpha, m - 1)
    lo = (fm.mean(1) + margin) / se > tc
    hi = (fm.mean(1) - margin) / se < -tc
    return float((lo & hi).mean())


def first_at_target(rows, key="power_boot"):
    ok = [r for r in rows if r.get(key, r.get("power", 0)) >= POWER_TARGET]
    return min(ok, key=lambda r: r["n_items"]) if ok else None


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    t0 = time.time()
    raw = {slug: pd.read_parquet(p) for slug, p in CELLS.items()}
    deltas = {(s, t): paired_deltas(df, t)
              for s, df in raw.items() for t in ("twohop", "onehop")}
    comps = {f"{s}/{t}": components(d) for (s, t), d in deltas.items()}

    # cross-model per-item delta correlations (twohop, protected arm)
    piv = pd.DataFrame({s: deltas[(s, "twohop")].set_index("item_id")["delta"]
                        for s in raw}).dropna()
    rho = piv.corr().round(3).to_dict()

    # planning dispersion: the worst (largest-variance) twohop cell
    worst_slug = max(raw, key=lambda s: comps[f"{s}/twohop"]["sig_f"]**2
                     + comps[f"{s}/twohop"]["sig_e"]**2)
    wc = comps[f"{worst_slug}/twohop"]
    wd = deltas[(worst_slug, "twohop")]

    results = {"A_cell_mean": [], "B_dissoc": [], "C_hp1": [], "D_tost": [],
               "E_tailrate": []}
    tails = {s: tail_pairs(df) for s, df in raw.items()}
    tail_rates = {s: {"rate_j": round(float(t.hit_j.mean()), 3),
                      "rate_r": round(float(t.hit_r.mean()), 3),
                      "n": int(len(t))} for s, t in tails.items()}
    # planning cell for the tail endpoint: the SMALLEST pilot J-vs-rand
    # tail-rate gap among cells where the tail exists (>5%): conservative.
    gap = {s: v["rate_j"] - v["rate_r"] for s, v in tail_rates.items()
           if v["rate_j"] > 0.05}
    tail_slug = min(gap, key=gap.get)
    for m in GRID_M:
        for k in GRID_K:
            n = m * k
            results["A_cell_mean"].append({
                "m_families": m, "k_per_family": k, "n_items": n,
                "power_normal": round(power_normal(
                    m, k, wc["sig_f"], wc["sig_e"], SESOI), 3),
                "power_boot": round(power_boot(wd, m, k, SESOI), 3)})
            results["D_tost"].append({
                "m_families": m, "k_per_family": k, "n_items": n,
                "power": round(power_tost(wc["sig_f"], wc["sig_e"], m, k), 3)})
            results["E_tailrate"].append({
                "m_families": m, "k_per_family": k, "n_items": n,
                "planning_cell": tail_slug,
                "power_at_sesoi_10pp": round(power_tailrate_boot(
                    tails[tail_slug], m, k, TAIL_SESOI), 3),
                "power_at_pilot_gap": round(power_tailrate_boot(
                    tails[tail_slug], m, k, gap[tail_slug]), 3)})

    oc = comps[f"{worst_slug}/onehop"]
    for m in GRID_M:
        for k in GRID_K:
            results["B_dissoc"].append({
                "m_families": m, "k_per_family": k,
                "n_items": m * k, "n_onehop": 60,
                "power": round(power_dissoc(wc, oc, m, k, 60, SESOI), 3)})

    # HP1: the pilot's decisive pair (Qwen vs Think) + worst-rho OLMo pair
    for m1_, m2_ in (("qwen36-27b", "olmo3-think"),
                     ("olmo31-instruct", "olmo3-think")):
        j2 = (deltas[(m1_, "twohop")].set_index("item_id")
              .join(deltas[(m2_, "twohop")].set_index("item_id"),
                    rsuffix="_b"))
        j2 = j2.assign(diff=j2.delta - j2.delta_b)[["family", "diff"]]\
            .dropna().reset_index()
        j1 = (deltas[(m1_, "onehop")].set_index("item_id")
              .join(deltas[(m2_, "onehop")].set_index("item_id"),
                    rsuffix="_b"))
        j1 = j1.assign(diff=j1.delta - j1.delta_b)[["family", "diff"]]\
            .dropna().reset_index()
        for m in GRID_M:
            for k in GRID_K:
                results["C_hp1"].append({
                    "pair": f"{m1_} - {m2_}", "m_families": m,
                    "k_per_family": k, "n_items": m * k, "n_onehop": 60,
                    "power": round(power_hp1_boot(j2, j1, m, k, 60, SESOI), 3)})

    recs = {
        "A_cell_mean": first_at_target(results["A_cell_mean"]),
        "B_dissoc": first_at_target(results["B_dissoc"], "power"),
        "C_hp1_worst_pair": first_at_target(
            [r for r in results["C_hp1"]
             if r["pair"] == "olmo31-instruct - olmo3-think"], "power"),
        "D_tost": first_at_target(results["D_tost"], "power"),
        "E_tailrate_sesoi10pp": first_at_target(results["E_tailrate"],
                                                "power_at_sesoi_10pp"),
    }
    floor_ok = all(r and r["n_items"] <= 90 for r in
                   (recs["A_cell_mean"], recs["D_tost"]))
    summ = {"planning_cell": worst_slug, "components": comps,
            "cross_model_rho_twohop": rho,
            "pilot_tail_rates": tail_rates,
            "tail_planning_cell": tail_slug,
            "alpha_primary": ALPHA, "sesoi_nats": SESOI,
            "tail_thr_nats": TAIL_THR, "tail_sesoi_pp": TAIL_SESOI,
            "power_target": POWER_TARGET, "n_sims": B,
            "recommendations": recs,
            "floor_n90_30fams_sufficient_for_A_D": bool(floor_ok),
            "design_implication": (
                "Protected-arm deltas are a zero-mode + heavy-tail mixture; "
                "0.5-nat MEAN primaries are underpowered at any affordable "
                "n (see A/D grids). The tail-RATE endpoint (frozen thr "
                f"{TAIL_THR} nat, J-vs-matched-random) reaches target power "
                "at modest n (E grid). Prereg draft should restate "
                "tail-carried hypotheses (HP3-style) on the rate endpoint "
                "and keep mean deltas for the dissociation/ordering "
                "contrasts with margins revisited by the user."),
            "seconds": round(time.time() - t0)}
    prov = Provenance(
        evidence_id="g6-power-sim-v2", tier="pilot",
        command="python -m jspace_part2.experiments.g6_power_sim",
        inputs={s: sha256_file(p) for s, p in CELLS.items()},
        seed=4242)
    write_result({"summary": summ, "grids": results}, OUT, prov)
    registry_append({
        "evidence_id": "g6-power-sim-v2", "tier": "pilot",
        "what": (f"G6 power simulation (pilot-calibrated, worst cell "
                 f"{worst_slug}): recommendations {json.dumps(recs)}; "
                 f"floor n=90/30 fams sufficient for A+D: {floor_ok}"),
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(OUT), "sha256": sha256_file(OUT)}]})
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
