# §4.1a — mine the EXISTING Phase 2 confirmatory/replication parquets for
# protected-output-pressure statistics (addendum §4.1, first Phase 3 task).
#
# WHAT THE PARQUETS CONTAIN (verified): per-item J-arm summaries
# (protected_blocked_total, n_positions, removed-energy and rank stats),
# matched-control gate rows, baseline clean_first_rank_min, and the frozen
# lp endpoints. They do NOT contain per-position selected IDs
# (record_ids=False in the N6 grids), so true span geometry (principal
# angles, answer-direction survival) is deferred to the §4.1b GPU audit.
# This CPU pass calibrates the Workstream B threat model with what exists:
#
#   blocked_rate = protected_blocked_total / n_positions   (per item)
#
# counts how often a PROTECTED (clean top-k) token row would have entered
# the J top-k and was excluded — selection pressure toward the protected
# output region. If the J arm's damage tracks this pressure, the §2.3
# leakage account gains prior weight and span-safe becomes the crown-jewel
# control; if pressure is small and uncorrelated with damage, the leakage
# account is pre-bounded. Either way: a figure + registered evidence.
#
# Phase 2 artifacts are read-only inputs; this registers NEW
# phase3-development evidence.
#
# Usage:
#   python -m jspace_phase3.experiments.overlap_mining [--allow-dirty]
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from jspace_part2.paths import resolve as resolve_uri
from ..figures3 import PAL3, SHORT_MODEL, apply_style, save_fig
from ..paths3 import figures_dir, metrics_dir
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           write_result3)

EVIDENCE_ID = "p3-overlap-mining-v1"
TIER = "phase3-development"
MODELS = ["olmo31-think", "olmo31-instruct", "qwen36-27b"]
GRIDS = {"confirmatory": "n6_grid", "replication": "n6_grid_repl"}
TAIL_NATS = -1.0                       # frozen Phase 2 threshold
PROTECT_K = 10                         # protected-answer stratum boundary


def load_item_table(model: str, partition: str) -> pd.DataFrame:
    pq = resolve_uri(f"drive://part2/metrics/{model}/{GRIDS[partition]}/"
                     f"n6_per_item_{model}.parquet")
    df = pd.read_parquet(pq)
    df = df[df.task != "prose"].copy()

    def summaries(cond, fields):
        rows = df[df.condition == cond][["item_id",
                                         "intervention_summary_json"]]
        out = {}
        for _, r in rows.iterrows():
            s = json.loads(r.intervention_summary_json) or {}
            out[r.item_id] = {f: s.get(f) for f in fields}
        return out

    j_sum = summaries("meanJ_protected",
                      ["protected_blocked_total", "n_positions",
                       "removed_energy_frac_mean", "removed_energy_frac_max",
                       "effective_rank_mean", "positions_below_requested_k"])
    piv = df.pivot_table(index="item_id", columns="condition",
                         values="lp_logsumexp", aggfunc="first")
    base = df[df.condition == "baseline"].set_index("item_id")
    rows = []
    for iid in piv.index:
        js = j_sum.get(iid) or {}
        npos = js.get("n_positions") or 0
        rows.append({
            "model": model, "partition": partition, "item_id": iid,
            "task": base.loc[iid, "task"],
            "canonical_family": base.loc[iid, "canonical_family"],
            "clean_first_rank_min": base.loc[iid, "clean_first_rank_min"],
            "baseline_lp": piv.loc[iid, "baseline"],
            "delta_J": piv.loc[iid, "meanJ_protected"] - piv.loc[iid, "baseline"],
            "delta_MC": (piv.loc[iid, "matched_control"] - piv.loc[iid, "baseline"]
                         if "matched_control" in piv.columns
                         and pd.notna(piv.loc[iid].get("matched_control"))
                         else np.nan),
            "delta_mech": (piv.loc[iid, "dynR_mechanics_control"]
                           - piv.loc[iid, "baseline"]),
            "blocked_total": js.get("protected_blocked_total"),
            "n_positions": npos,
            "blocked_rate": ((js.get("protected_blocked_total") or 0)
                             / npos if npos else np.nan),
            "removed_energy_mean": js.get("removed_energy_frac_mean"),
            "removed_energy_max": js.get("removed_energy_frac_max"),
            "effective_rank_mean": js.get("effective_rank_mean"),
            "starved_positions": js.get("positions_below_requested_k"),
        })
    t = pd.DataFrame(rows)
    t["specific_delta"] = t.delta_J - t.delta_MC
    t["tail_J"] = t.delta_J < TAIL_NATS
    t["tail_specific"] = t.specific_delta < TAIL_NATS
    t["protected_answer"] = t.clean_first_rank_min <= PROTECT_K
    return t


def family_boot_spearman(t: pd.DataFrame, x: str, y: str,
                         draws: int = 4000, seed: int = 4242) -> dict:
    """Spearman rho with a family-clustered bootstrap CI."""
    from scipy.stats import spearmanr
    d = t[[x, y, "canonical_family"]].dropna()
    if len(d) < 8 or d.canonical_family.nunique() < 3:
        return {"n": int(len(d)), "rho": None}
    rho = float(spearmanr(d[x], d[y]).statistic)
    fams = d.canonical_family.unique()
    by = {f: d[d.canonical_family == f] for f in fams}
    rng = np.random.default_rng(seed)
    samp = []
    for _ in range(draws):
        pick = rng.choice(fams, size=len(fams), replace=True)
        b = pd.concat([by[f] for f in pick])
        if b[x].nunique() > 2 and b[y].nunique() > 2:
            samp.append(float(spearmanr(b[x], b[y]).statistic))
    lo, hi = (np.percentile(samp, [2.5, 97.5]) if samp else (np.nan,) * 2)
    return {"n": int(len(d)), "n_families": int(len(fams)),
            "rho": round(rho, 4), "ci95": [round(float(lo), 4),
                                           round(float(hi), 4)]}


def tail_contrast(t: pd.DataFrame, col: str) -> dict:
    from scipy.stats import mannwhitneyu
    a = t.loc[t.tail_J, col].dropna()
    b = t.loc[~t.tail_J, col].dropna()
    if len(a) < 3 or len(b) < 3:
        return {"n_tail": int(len(a)), "n_nontail": int(len(b))}
    u = mannwhitneyu(a, b, alternative="two-sided")
    return {"n_tail": int(len(a)), "n_nontail": int(len(b)),
            "tail_median": round(float(a.median()), 5),
            "nontail_median": round(float(b.median()), 5),
            "mannwhitney_p": round(float(u.pvalue), 5)}


def make_figure(tab: pd.DataFrame, stats: dict):
    apply_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(MODELS), figsize=(11, 6.4),
                             sharey="row")
    for ci, m in enumerate(MODELS):
        t = tab[(tab.model == m)].dropna(subset=["blocked_rate", "delta_J"])
        ax = axes[0][ci]
        nt = t[~t.tail_J]
        tl = t[t.tail_J]
        ax.scatter(nt.blocked_rate, nt.delta_J, s=16, facecolors="none",
                   edgecolors=PAL3["muted"], linewidths=0.9,
                   label="non-tail item")
        ax.scatter(tl.blocked_rate, tl.delta_J, s=22, color=PAL3["J"],
                   label="tail item (Δ < −1 nat)")
        ax.axhline(TAIL_NATS, color=PAL3["grid"], lw=1)
        r = stats[m]["pooled"]["spearman_blocked_vs_deltaJ"]
        ax.set_title(f"{SHORT_MODEL[m]}\nρ={r['rho']:+.2f} "
                     f"[{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]")
        ax.set_xlabel("protected-row blocked rate / position")
        if ci == 0:
            ax.set_ylabel("Δ lp under mean-J (nats)")
            ax.legend(loc="lower left", fontsize=7.5)
        ax.grid(True, axis="y", alpha=0.6)

        ax = axes[1][ci]
        groups = [t.loc[~t.tail_J, "blocked_rate"].dropna(),
                  t.loc[t.tail_J, "blocked_rate"].dropna()]
        vp = ax.violinplot(groups, positions=[0, 1], widths=0.7,
                           showmedians=True, showextrema=False)
        for body, c in zip(vp["bodies"], (PAL3["muted"], PAL3["J"])):
            body.set_facecolor(c)
            body.set_alpha(0.45)
        vp["cmedians"].set_color(PAL3["ink"])
        for gi, g in enumerate(groups):
            xj = np.random.default_rng(7).normal(gi, 0.05, len(g))
            ax.plot(xj, g, ".", ms=3.2,
                    color=PAL3["muted"] if gi == 0 else PAL3["J"],
                    alpha=0.7)
        tc = stats[m]["pooled"]["tail_contrast_blocked_rate"]
        p = tc.get("mannwhitney_p")
        ax.set_title(f"tail vs non-tail blocked rate"
                     f"{'' if p is None else f'  (MW p={p:.3g})'}",
                     fontsize=8.5)
        ax.set_xticks([0, 1], ["non-tail", "tail"])
        if ci == 0:
            ax.set_ylabel("blocked rate / position")
        ax.grid(True, axis="y", alpha=0.6)
    fig.suptitle("§4.1a Protected-output selection pressure vs J-arm damage "
                 "(Phase 2 parquets, both partitions pooled; per-item "
                 "summaries — geometric span audit is §4.1b)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    return fig


def main():
    require_clean_tree("--allow-dirty" in sys.argv)
    tabs = []
    for m in MODELS:
        for part in GRIDS:
            tabs.append(load_item_table(m, part))
    tab = pd.concat(tabs, ignore_index=True)

    stats: dict = {}
    for m in MODELS:
        t_all = tab[tab.model == m]
        per = {}
        for scope_name, t in {
                "pooled": t_all,
                "confirmatory": t_all[t_all.partition == "confirmatory"],
                "protected_answer_stratum":
                    t_all[t_all.protected_answer]}.items():
            per[scope_name] = {
                "n_items": int(len(t)),
                "n_tail_J": int(t.tail_J.sum()),
                "blocked_rate_median": round(float(
                    t.blocked_rate.median()), 5),
                "blocked_rate_p90": round(float(
                    t.blocked_rate.quantile(0.9)), 5),
                "spearman_blocked_vs_deltaJ": family_boot_spearman(
                    t, "blocked_rate", "delta_J"),
                "spearman_blocked_vs_specific": family_boot_spearman(
                    t, "blocked_rate", "specific_delta"),
                "spearman_cleanrank_vs_deltaJ": family_boot_spearman(
                    t, "clean_first_rank_min", "delta_J"),
                "tail_contrast_blocked_rate": tail_contrast(
                    t, "blocked_rate"),
                "tail_contrast_removed_energy": tail_contrast(
                    t, "removed_energy_mean"),
                "tail_contrast_clean_rank": tail_contrast(
                    t, "clean_first_rank_min"),
            }
        stats[m] = per

    out_dir = metrics_dir("cross_model") / "overlap_mining"
    out_dir.mkdir(parents=True, exist_ok=True)
    pq = out_dir / "overlap_mining_items.parquet"
    tab.to_parquet(pq)

    fig = make_figure(tab, stats)
    figs = save_fig(fig, figures_dir(), "p3f01_overlap_mining", TIER)

    payload = {
        "what": "§4.1a mining of Phase 2 per-item summaries",
        "estimand_note": ("blocked_rate is SELECTION PRESSURE toward "
                          "protected output rows, not geometric span "
                          "overlap; §4.1b measures the geometry"),
        "tail_threshold_nats": TAIL_NATS,
        "models": stats,
        "inputs": {f"{m}/{g}": f"drive://part2/metrics/{m}/{GRIDS[g]}/"
                   f"n6_per_item_{m}.parquet"
                   for m in MODELS for g in GRIDS},
    }
    cmd = "python -m jspace_phase3.experiments.overlap_mining"
    prov = Provenance3(evidence_id=EVIDENCE_ID, tier=TIER, command=cmd,
                       seed=4242)
    out_json = out_dir / "overlap_mining_summary.json"
    write_result3(payload, out_json, prov)
    register(EVIDENCE_ID, tier=TIER, command=cmd,
             what=("4.1a overlap mining: protected-row selection pressure "
                   "vs J damage from existing Phase 2 parquets (3 models x "
                   "2 partitions); calibrates Workstream B threat model"),
             outputs=[out_json, pq, *figs])
    print(json.dumps({m: stats[m]["pooled"] for m in MODELS}, indent=1))
    print(f"banked {pq}\nfigure {figs[0]}")


if __name__ == "__main__":
    main()
