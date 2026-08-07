# Final phase: figures + REPORT.md + summary.json — everything regenerable
# from saved metrics alone (no model, no GPU).
#
# Colors: validated default dataviz palette (light mode) — blue #2a78d6
# (J-lens / J-space), orange #eb6834 (logit-lens / random control), aqua
# #1baf7a (non-J high-variance control); ink/grid per reference. <=3 hues
# per panel; single-series panels carry no legend; direct labels preferred.
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl1_common import RUN_DIR, atomic_write_json, ensure_dirs, log, read_json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = RUN_DIR / "figures"
C = {"j": "#2a78d6", "logit": "#eb6834", "rand": "#eb6834", "nonj": "#1baf7a",
     "ink": "#0b0b0b", "sec": "#52514e", "muted": "#898781",
     "grid": "#e1e0d9", "base": "#c3c2b7", "surface": "#fcfcfb",
     "seq250": "#86b6ef", "seq450": "#2a78d6", "seq650": "#104281"}

plt.rcParams.update({
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
    "axes.edgecolor": C["base"], "axes.labelcolor": C["sec"],
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.8,
    "xtick.color": C["muted"], "ytick.color": C["muted"],
    "text.color": C["ink"], "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "savefig.dpi": 150, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    log(f"figure {name}")


def fig_variance_share(agg):
    layers = agg["layers"]
    vs = [agg["per_layer"][str(l)]["sum_recon_sq_c"] /
          max(agg["per_layer"][str(l)]["sum_h_sq_c"], 1e-9) for l in layers]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhspan(0.06, 0.10, color=C["grid"], alpha=0.6, lw=0)
    ax.text(layers[-1], 0.101, "paper (Claude): 6–10%", ha="right",
            va="bottom", color=C["muted"], fontsize=9)
    ax.plot(layers, vs, color=C["j"], lw=2, marker="o", ms=5)
    ax.set_xlabel("layer")
    ax.set_ylabel("J-space share of centered activation variance")
    ax.set_title("J-space variance share by layer — Olmo-3-32B-Think")
    ax.set_ylim(bottom=0)
    save(fig, "f1_variance_share_by_layer.png")
    return {f"L{l}": round(v, 4) for l, v in zip(layers, vs)}


def fig_active_concepts(agg):
    layers = agg["layers"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for t, color, lw in (("0.01", C["seq250"], 1.5), ("0.02", C["seq450"], 2.5),
                         ("0.05", C["seq650"], 1.5)):
        med = [float(np.median([m for _, m in
                                agg["per_layer"][str(l)]["active_counts"][t]]))
               for l in layers]
        ax.plot(layers, med, color=color, lw=lw)
        ax.text(layers[-1] + 0.5, med[-1], f"θ={t}", color=color,
                fontsize=9, va="center")
    ax.axhline(25, color=C["muted"], ls="--", lw=1)
    ax.text(layers[0], 25.7, "paper: ~25 active concepts", color=C["muted"],
            fontsize=9)
    ax.set_xlabel("layer")
    ax.set_ylabel("median active J-lens vectors per position")
    ax.set_title("Active-concept count by layer (threshold sensitivity)")
    save(fig, "f2_active_concepts_by_layer.png")
    med02 = {l: float(np.median([m for _, m in
             agg["per_layer"][str(l)]["active_counts"]["0.02"]]))
             for l in layers}
    return med02


def fig_lens_vs_logit(sanity):
    layers = sorted(int(l) for l in
                    sanity["probes"][0]["jlens_rank_by_layer"])
    j_med = [float(np.median([p["jlens_rank_by_layer"][str(l)]
                              for p in sanity["probes"]])) for l in layers]
    l_med = [float(np.median([p["logit_rank_by_layer"][str(l)]
                              for p in sanity["probes"]])) for l in layers]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(layers, j_med, color=C["j"], lw=2, marker="o", ms=5, label="J-lens")
    ax.plot(layers, l_med, color=C["logit"], lw=2, marker="s", ms=5,
            label="logit lens")
    ax.set_yscale("log")
    ax.set_xlabel("layer")
    ax.set_ylabel("median rank of answer token (log)")
    ax.set_title("Answer surfacing: J-lens vs logit lens "
                 f"({sanity['n_probes']} factual probes)")
    ax.legend(frameon=False)
    save(fig, "f3_lens_vs_logit_rank.png")


def fig_ablation(ab):
    conds = [("none", "baseline", C["ink"]),
             ("jspace_dyn10", "J-space top-10 (dynamic)", C["j"]),
             ("random_k10", "random 10-dim", C["rand"]),
             ("nonJ_pca_k10", "non-J PCA 10-dim", C["nonj"])]
    tasks = [("twohop", "2-hop factual"),
             ("arithmetic_v2", "chained arithmetic"),
             ("sql", "multi-clause SQL"), ("onehop", "single-hop factual"),
             ("grammar", "grammaticality")]
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(tasks))[::-1]
    for i, (cond, label, color) in enumerate(conds):
        if cond not in ab["conditions"]:
            continue
        off = (i - 1.5) * 0.17
        for yi, (t, _) in zip(y, tasks):
            e = ab["conditions"][cond].get(t)
            if not e:
                continue
            marker = "o" if cond != "none" else "D"
            ax.plot([e["ci_lo"], e["ci_hi"]], [yi + off] * 2, color=color,
                    lw=2, alpha=0.9)
            ax.plot(e["mean"], yi + off, marker, color=color, ms=7,
                    label=label if yi == y[0] else None)
    ax.set_yticks(y, [n for _, n in tasks])
    ax.set_xlabel("accuracy (bootstrap 95% CI)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_title("Ablation dissociation: multi-step collapses, shallow "
                 "tasks survive")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    save(fig, "f4_ablation_dissociation.png")


def fig_dose(ab):
    doses = [10, 20, 40]
    groups = [("jspace", "J-space", C["j"]), ("random", "random", C["rand"]),
              ("nonJ_pca", "non-J PCA", C["nonj"])]
    multi = ("twohop", "arithmetic_v2", "sql")
    flue = ("onehop", "grammar")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, tset, title in ((axes[0], multi, "multi-step tasks"),
                            (axes[1], flue, "fluency controls")):
        base = np.mean([ab["conditions"]["none"][t]["mean"] for t in tset
                        if t in ab["conditions"].get("none", {})])
        ax.axhline(base, color=C["muted"], ls="--", lw=1)
        ax.text(doses[-1], base + 0.015, "baseline", ha="right",
                color=C["muted"], fontsize=9)
        for gname, label, color in groups:
            ys = []
            for k in doses:
                cond = f"{gname}_k{k}"
                if cond in ab["conditions"]:
                    ys.append(np.mean([ab["conditions"][cond][t]["mean"]
                                       for t in tset
                                       if t in ab["conditions"][cond]]))
                else:
                    ys.append(np.nan)
            ax.plot(doses, ys, color=color, lw=2, marker="o", ms=6,
                    label=label)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("ablated dimensions per band layer")
        ax.set_xticks(doses)
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("mean accuracy")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Dose–response of subspace ablation", fontweight="bold")
    save(fig, "f5_dose_response.png")


def fig_cot(cot):
    items = list(cot["items"].values())
    if not items:
        return {}
    mid = [l for l in (24, 32, 40)]
    pre_ans = [min(it["pre_cot"]["answer"]["jlens_rank_by_layer"][str(l)]
                   for l in mid if str(l) in
                   it["pre_cot"]["answer"]["jlens_rank_by_layer"])
               for it in items]
    pre_logit = [min(it["pre_cot"]["answer"]["logit_rank_by_layer"][str(l)]
                     for l in mid if str(l) in
                     it["pre_cot"]["answer"]["logit_rank_by_layer"])
                 for it in items]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for vals, color, label in ((pre_ans, C["j"], "J-lens"),
                               (pre_logit, C["logit"], "logit lens")):
        xs = np.sort(vals)
        ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post",
                color=color, lw=2, label=label)
    ax.axvline(20, color=C["muted"], ls="--", lw=1)
    ax.text(21, 0.05, "rank 20", color=C["muted"], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("rank of final answer at last prompt token (best mid layer)")
    ax.set_ylabel("fraction of items")
    ax.set_title("Pre-CoT anticipation: answer in workspace before any "
                 "thinking token")
    ax.legend(frameon=False)
    save(fig, "f6_precot_anticipation.png")
    return {"n": len(items),
            "jlens_frac_rank_le20": float(np.mean(np.array(pre_ans) <= 20)),
            "logit_frac_rank_le20": float(np.mean(np.array(pre_logit) <= 20)),
            "jlens_median_rank": float(np.median(pre_ans))}


def fig_broadcast(bc):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    src = sorted(bc["source_layers"], key=int)
    x = np.arange(len(src))
    for i, (g, label, color) in enumerate(
            (("jspace", "J-space", C["j"]), ("random", "random", C["rand"]),
             ("nonJ_pca", "non-J PCA", C["nonj"]))):
        vals = [bc["source_layers"][s][g]["fan_out_median"] for s in src]
        ax.bar(x + (i - 1) * 0.26, vals, width=0.24, color=color, label=label)
    ax.set_xticks(x, [f"L{s}" for s in src])
    ax.set_ylabel("median downstream readers (z > 3)")
    ax.set_title("Broadcast: components reading each direction group")
    ax.legend(frameon=False)
    save(fig, "f8_broadcast_fanout.png")


def main() -> None:
    ensure_dirs()
    m = {}
    summary = {"model": "allenai/Olmo-3-32B-Think",
               "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    have = lambda p: (RUN_DIR / "metrics" / p).exists()

    if have("descriptive_agg.json"):
        agg = read_json(RUN_DIR / "metrics" / "descriptive_agg.json")
        summary["variance_share_by_layer"] = fig_variance_share(agg)
        med02 = fig_active_concepts(agg)
        band = [l for l in agg["layers"] if 20 <= l <= 44]
        summary["active_concepts_mid_median"] = float(
            np.median([med02[l] for l in band]))
        summary["variance_share_mid_max"] = max(
            summary["variance_share_by_layer"][f"L{l}"] for l in band)
        pl = agg["per_layer"]
        summary["top1_persistence_mid"] = float(np.mean(
            [pl[str(l)]["top1_persist"][0] / max(pl[str(l)]["top1_persist"][1], 1)
             for l in band]))
    if have("lens_sanity_32b.json"):
        s = read_json(RUN_DIR / "metrics" / "lens_sanity_32b.json")
        fig_lens_vs_logit(s)
        summary["probe_hits_at_20"] = {"jlens": s["probe_jlens_hits_at_20"],
                                       "logit": s["probe_logit_hits_at_20"],
                                       "n": s["n_probes"]}
        summary["multihop_eval"] = {k: round(v, 3)
                                    for k, v in s["multihop"].items()}
    if have("ablation_results.json"):
        ab = read_json(RUN_DIR / "metrics" / "ablation_results.json")
        fig_ablation(ab)
        fig_dose(ab)
        cc = ab["conditions"]
        if "jspace_dyn10" in cc and "none" in cc:
            summary["ablation"] = {
                t: {"baseline": cc["none"][t]["mean"],
                    "jspace_dyn10": cc["jspace_dyn10"][t]["mean"],
                    "random_k10": cc.get("random_k10", {}).get(t, {}).get("mean"),
                    "nonJ_pca_k10": cc.get("nonJ_pca_k10", {}).get(t, {}).get("mean")}
                for t in cc["none"] if t in cc["jspace_dyn10"]}
    if have("cot_results.json"):
        cot = read_json(RUN_DIR / "metrics" / "cot_results.json")
        summary["cot_anticipation"] = fig_cot(cot)
        items = list(cot["items"].values())
        if items:
            summary["cot"] = {
                "n_items": len(items),
                "think_acc": float(np.mean([i["think_correct"] for i in items])),
                "suppressed_acc": float(np.mean([i["suppressed"]["correct"]
                                                 for i in items])),
                "divergence_events_total": int(sum(i["n_divergence_events"]
                                                   for i in items)),
                "items_with_divergence": int(sum(i["n_divergence_events"] > 0
                                                 for i in items)),
            }
    if have("cot_lead.json"):
        cl = read_json(RUN_DIR / "metrics" / "cot_lead.json")
        summary["cot_lead"] = cl["lead"]
        prof = cl["anticipation_rank_by_layer_median"]
        summary["pre_answer_median_rank"] = {
            k: {f"L{l}": prof[k][str(l)] for l in (24, 32, 36, 40, 44)}
            for k in prof}
    if have("broadcast.json"):
        bc = read_json(RUN_DIR / "metrics" / "broadcast.json")
        fig_broadcast(bc)
        summary["broadcast"] = {
            s: {g: bc["source_layers"][s][g]["fan_out_median"]
                for g in ("jspace", "random", "nonJ_pca")}
            for s in bc["source_layers"]}

    atomic_write_json(summary, RUN_DIR / "report" / "summary.json")
    log("wrote report/summary.json")
    # REPORT.md is assembled by the session (needs prose judgment); this
    # script guarantees figures + summary.json stay regenerable from metrics.


if __name__ == "__main__":
    main()
