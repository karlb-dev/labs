# Part-2 figure factory. Every figure regenerates from metrics JSONs alone;
# every figure is guarded (skips quietly if its metrics don't exist yet), so
# refresh_handout.sh can run at any phase boundary.
#
# Palette: fixed entity→hue map continuing part 1, validated 2026-07-27 with
# the dataviz six-checks (4-hue categorical set passes lightness/chroma/CVD/
# normal floors on a light surface; the #1baf7a contrast WARN is relieved by
# always direct-labeling marks). Hues follow ENTITIES and never change:
#   J-space           #2a78d6  (part 1)
#   random control    #eb6834  (part 1)
#   non-J control     #1baf7a  (part 1)
#   frozen-logit dict #8a5cf5  (fixed here, 2026-07-27)
#   baseline/ink      #1f2430
#   logit-lens        #5b6472  (readout-method entity, fixed here)
# Model identity is carried by position/texture (45° hatch = non-donor
# model), never by repainting an entity hue.
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl2_common import RUN_DIR, RUN_DIR_P2, log

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAL = {"J": "#2a78d6", "random": "#eb6834", "nonJ": "#1baf7a",
       "frozen_logit": "#8a5cf5", "ink": "#1f2430", "logit": "#5b6472",
       "muted": "#8a90a0", "grid": "#e4e6eb"}
FIGDIR = RUN_DIR_P2 / "figures"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": PAL["muted"], "axes.labelcolor": PAL["ink"],
    "text.color": PAL["ink"], "xtick.color": PAL["ink"],
    "ytick.color": PAL["ink"], "axes.spines.top": False,
    "axes.spines.right": False, "grid.color": PAL["grid"],
    "grid.linewidth": 0.8, "font.size": 9.5,
})


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def f1_a0_transfer():
    """p2f1 — A0 transfer gate: donor-vs-transferred readout quality."""
    a0 = _load(RUN_DIR_P2 / "metrics" / "olmo31-instruct" / "a0_transfer_gate.json")
    donor = _load(RUN_DIR / "metrics" / "lens_sanity_32b.json")
    if not (a0 and donor):
        return
    out = FIGDIR / "p2f1_a0_transfer.png"
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.7),
                                 gridspec_kw={"width_ratios": [1.25, 1]})

    # Panel A — multihop pass@k: hue = lens entity, texture = model.
    ks = ["1", "5", "20"]
    series = [  # (label, values, color, hatch)
        ("Think · J-lens (donor)",
         [donor["multihop"][f"jlens_pass@{k}"] for k in ks], PAL["J"], None),
        ("Think · logit lens",
         [donor["multihop"][f"logit_pass@{k}"] for k in ks], PAL["logit"], None),
        ("3.1-Instruct · transferred J-lens",
         [a0["multihop"][f"jlens_pass@{k}"] for k in ks], PAL["J"], "///"),
        ("3.1-Instruct · own logit lens",
         [a0["multihop"][f"logit_pass@{k}"] for k in ks], PAL["logit"], "///"),
    ]
    w, n = 0.19, len(series)
    for i, (lab, vals, col, hat) in enumerate(series):
        xs = [j + (i - (n - 1) / 2) * (w + 0.012) for j in range(len(ks))]
        ax.bar(xs, vals, width=w, color=col, hatch=hat, label=lab,
               edgecolor="white", linewidth=0.8)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.012, f"{v:.2f}".lstrip("0"), ha="center",
                    va="bottom", fontsize=7.2, color=PAL["ink"])
    floor = a0["gate"]["pass1_threshold"]
    ax.plot([-0.46, 0.46], [floor, floor], ls=(0, (4, 3)), lw=1.2,
            color=PAL["random"])
    ax.text(-0.46, floor - 0.018, f"prereg gate {floor:.3f}", fontsize=7.2,
            ha="left", va="top", color=PAL["random"])
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"pass@{k}" for k in ks])
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("multihop bridge readout (n=60)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.2, frameon=False, loc="upper left")
    ax.set_title("A0 — donor lens transfers the shortlist, not the rank-1 edge"
                 "  [exploratory]", fontsize=9.5, loc="left")

    # Panel B — per-probe mid-band min rank, donor vs transferred (log-log).
    dref = {r["answer"]: r["jlens_mid_min"] for r in donor["probes"]}
    pairs = [(dref[r["answer"]], r["jlens_mid_min"], r["answer"])
             for r in a0["probes"] if r["answer"] in dref]
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    bx.scatter(xs, ys, s=26, color=PAL["J"], alpha=0.85, edgecolor="white",
               linewidth=0.6, zorder=3)
    lim = (0.7, 1.3e5)
    bx.plot(lim, lim, lw=0.9, color=PAL["muted"], zorder=1)
    for v in (20,):
        bx.axvline(v, lw=0.8, ls=(0, (2, 3)), color=PAL["muted"])
        bx.axhline(v, lw=0.8, ls=(0, (2, 3)), color=PAL["muted"])
    for x, y, a in pairs:  # selective labels: misses only
        if y > 20:
            bx.annotate(a.strip(), (x, y), textcoords="offset points",
                        xytext=(4, 3), fontsize=6.8, color=PAL["muted"])
    bx.set_xscale("log"); bx.set_yscale("log")
    bx.set_xlim(*lim); bx.set_ylim(*lim)
    bx.set_xlabel("donor rank on Think (mid-band min)")
    bx.set_ylabel("transferred rank on 3.1-Instruct")
    bx.set_title("21 probes: 17/21 ≤20 on both (identical count)",
                 fontsize=9.5, loc="left")

    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {out}")


def f2_b3_frozen_logit():
    """p2f2 — B3: frozen-logit vs frozen-J on the causal battery (Think)."""
    b3 = _load(RUN_DIR_P2 / "metrics" / "olmo3-think" / "b3_frozen_logit.json")
    if not b3 or "frozen_logit10" not in b3.get("conditions", {}):
        return
    c = b3["conditions"]
    tasks = [("twohop_lp", "two-hop answer logprob (nats)"),
             ("twohop", "two-hop accuracy"),
             ("onehop", "one-hop accuracy"),
             ("prose_nll", "prose NLL (fluency guard)")]
    conds = [("none", "baseline", PAL["ink"]),
             ("frozen_j10", "frozen-J top-10", PAL["J"]),
             ("frozen_logit10", "frozen-logit top-10", PAL["frozen_logit"]),
             ("frozen_rand10", "frozen-random (5120)", PAL["random"])]
    avail = [(t, lab) for t, lab in tasks
             if all(t in c.get(k, {}) for k, _, _ in conds)]
    if not avail:
        return
    out = FIGDIR / "p2f2_b3_frozen_logit.png"
    fig, axes = plt.subplots(1, len(avail), figsize=(2.5 * len(avail), 3.3))
    axes = [axes] if len(avail) == 1 else list(axes)
    for ax, (t, lab) in zip(axes, avail):
        for i, (k, klab, col) in enumerate(conds):
            e = c[k][t]
            ax.errorbar(i, e["mean"],
                        yerr=[[e["mean"] - e["ci_lo"]], [e["ci_hi"] - e["mean"]]],
                        fmt="o", ms=6.5, color=col, capsize=3, lw=1.4)
            ax.text(i + 0.13, e["mean"], f"{e['mean']:.2f}", fontsize=7,
                    va="center", color=PAL["ink"])
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([k for _, k, _ in conds], rotation=28, ha="right",
                           fontsize=7)
        ax.set_title(lab, fontsize=8.8, loc="left")
        ax.grid(axis="y"); ax.set_axisbelow(True)
    fig.suptitle("B3 — does the causal handle need the Jacobian? "
                 "(OLMo-3-32B-Think, per-item frozen top-10, 95% boot CI)"
                 "  [exploratory pilot]",
                 fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {out}")


def f3_r7_protected():
    """p2f3 — R7: the paper's protected dynamic ablation vs unprotected."""
    import pandas as pd
    base = RUN_DIR_P2 / "metrics" / "olmo3-think" / "r7_pilot"
    if not (base / "r7_per_item.parquet").exists():
        return
    df = pd.read_parquet(base / "r7_per_item.parquet")
    cr_path = base / "r7_cleanrank.json"
    cr = {r["item_id"]: r["clean_rank"]
          for r in json.loads(cr_path.read_text())["rows"]} \
        if cr_path.exists() else {}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.8),
                                 gridspec_kw={"width_ratios": [1, 1.15]})
    conds = [("dynJ_protected", "dyn-J protected (paper)", PAL["J"], None),
             ("dynJ_unprotected", "dyn-J unprotected", PAL["J"], "///"),
             ("dynR_protected", "dyn-random protected", PAL["random"], None)]
    tasks = [("twohop", 0), ("onehop", 1)]
    for i, (cond, lab, col, hat) in enumerate(conds):
        for tname, tx in tasks:
            sub = df[df.task == tname].pivot_table(
                index="item_id", columns="condition", values="score",
                aggfunc="first")
            delta = (sub[cond] - sub["none"]).dropna()
            x = tx + (i - 1) * 0.22
            m = float(delta.mean())
            se = float(delta.std() / len(delta) ** 0.5)
            ax.bar(x, m, width=0.18, color=col, hatch=hat,
                   edgecolor="white", linewidth=0.8,
                   label=lab if tx == 0 else None)
            ax.errorbar(x, m, yerr=2 * se, fmt="none", ecolor=PAL["ink"],
                        capsize=2.5, lw=1)
            ax.plot([x - 0.09, x + 0.09],
                    [float(delta.median())] * 2, color=PAL["ink"], lw=1.6)
            ax.text(x, m - 0.13, f"{m:+.2f}", ha="center", fontsize=6.8,
                    color=PAL["ink"])
    ax.axhline(0, color=PAL["muted"], lw=0.9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["two-hop (n=60)", "one-hop (n=30)"])
    ax.set_ylabel("Δ answer-sequence logprob vs baseline (nats)")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_title("R7: protection flips the median to ~0  [pilot]",
                 fontsize=9, loc="left")

    sub = df[df.task == "twohop"].pivot_table(
        index="item_id", columns="condition", values="score", aggfunc="first")
    dp = sub["dynJ_protected"] - sub["none"]
    du = sub["dynJ_unprotected"] - sub["none"]
    prot = [cr.get(i, 1) <= 10 for i in sub.index]
    for flag, col, lab in ((True, PAL["J"], "answer in clean top-10 (protected)"),
                           (False, PAL["random"], "answer rank>10 (unprotected)")):
        xs = [du[i] for i, f in zip(sub.index, prot) if f == flag]
        ys = [dp[i] for i, f in zip(sub.index, prot) if f == flag]
        bx.scatter(xs, ys, s=26, color=col, alpha=0.85, edgecolor="white",
                   linewidth=0.6, label=lab, zorder=3)
    lim = (min(du.min(), dp.min()) - 0.4, max(du.max(), dp.max()) + 0.4)
    bx.plot(lim, lim, lw=0.9, color=PAL["muted"])
    bx.axhline(-1, ls=(0, (3, 3)), lw=0.9, color=PAL["muted"])
    bx.text(lim[0] + 0.1, -1.15, "tail threshold", fontsize=6.8,
            color=PAL["muted"], va="top")
    bx.set_xlabel("unprotected Δlp")
    bx.set_ylabel("protected Δlp")
    bx.legend(fontsize=7, frameon=False, loc="upper left")
    bx.set_title("per-item: the tail is mostly PROTECTED items",
                 fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(FIGDIR / "p2f3_r7_protected.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    log("wrote p2f3_r7_protected.png")


def f4_r2_occupancy():
    """p2f4 — paper-defined occupancy + excess variance vs the paper band."""
    data = {}
    for slug, lab in (("olmo3-think", "OLMo-3-32B-Think"),
                      ("olmo31-instruct", "Olmo-3.1-32B-Instruct"),
                      ("qwen36-27b", "Qwen3.6-27B")):
        p = RUN_DIR_P2 / "metrics" / slug / "r2_occupancy" / "r2_occupancy.json"
        if p.exists():
            data[lab] = json.loads(p.read_text())["per_layer"]
    if not data:
        return
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.2, 3.5))
    colors = {"OLMo-3-32B-Think": PAL["J"],
              "Olmo-3.1-32B-Instruct": PAL["nonJ"],
              "Qwen3.6-27B": PAL["frozen_logit"]}
    for lab, per in data.items():
        Ls = sorted(int(l) for l in per)
        ax.plot(Ls, [per[str(l)]["occ_median"] for l in Ls], "o-", ms=6,
                lw=1.6, color=colors[lab], label=lab)
        bx.plot(Ls, [max(per[str(l)]["excess_share"], 1e-5) for l in Ls],
                "o-", ms=6, lw=1.6, color=colors[lab], label=lab)
    ax.axhspan(10, 25, color=PAL["grid"], zorder=0)
    ax.text(ax.get_xlim()[1], 17, " paper (Claude)\n occupancy 10–25",
            fontsize=6.8, color=PAL["muted"], va="center")
    ax.set_ylim(0, 28)
    ax.set_ylabel("occupancy (median, frozen crossing rule)")
    ax.set_xlabel("layer")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    bx.set_yscale("log")
    bx.axhspan(0.06, 0.10, color=PAL["grid"], zorder=0)
    bx.text(bx.get_xlim()[1], 0.077, " paper 6–10%", fontsize=6.8,
            color=PAL["muted"], va="center")
    bx.set_ylabel("excess variance share (log)")
    bx.set_xlabel("layer")
    bx.grid(axis="y", which="both")
    bx.set_axisbelow(True)
    fig.suptitle("R2 — paper-defined capacity estimator  [pilot]",
                 fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGDIR / "p2f4_r2_occupancy.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    log("wrote p2f4_r2_occupancy.png")


def f5_crossmodel_protected():
    """p2f5 — the campaign headline: protected dyn-J effects across models
    (paired family-clustered bootstrap CIs)."""
    rows = []
    for slug, lab in (("olmo3-think", "OLMo-3-32B-Think"),
                      ("qwen36-27b", "Qwen3.6-27B")):
        p = RUN_DIR_P2 / "metrics" / slug / "r7_pilot" / "r7_paired_ci.json"
        if p.exists():
            rows.append((lab, json.loads(p.read_text())))
    if len(rows) < 2:
        return
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    conds = [("dynJ_protected", "dyn-J protected (paper)", PAL["J"], None),
             ("dynR_protected", "dyn-random protected", PAL["random"], None)]
    tasks = ["twohop", "onehop"]
    xpos, labels = [], []
    x = 0.0
    for mlab, d in rows:
        for t in tasks:
            for i, (c, clab, col, hat) in enumerate(conds):
                v = d[f"{c}/{t}"]
                ax.errorbar(x + i * 0.28, v["estimate"],
                            yerr=[[v["estimate"] - v["ci_low"]],
                                  [v["ci_high"] - v["estimate"]]],
                            fmt="o", ms=7, color=col, capsize=3.5, lw=1.5,
                            label=clab if x == 0 else None)
            xpos.append(x + 0.14)
            labels.append(f"{t}\n{mlab.split('-')[0]}")
            x += 1.0
        x += 0.45
    ax.axhline(0, color=PAL["muted"], lw=0.9)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("paired Δ answer-seq logprob (nats, 95% cluster CI)")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_title("The paper's protected ablation across open models: "
                 "dissociation on Qwen, equal-depth damage on OLMo  [pilot]",
                 fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(FIGDIR / "p2f5_crossmodel_protected.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    log("wrote p2f5_crossmodel_protected.png")


def f6_capacity_errata():
    """p2f6 — old proxy vs paper estimator: the Qwen 'paper-range' reading
    collapses; OLMo's thinness is confirmed (addendum errata figure)."""
    p = RUN_DIR_P2 / "metrics" / "shared" / "capacity_errata.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_yscale("log")
    ax.axhspan(d["paper_claude"]["excess"][0], d["paper_claude"]["excess"][1],
               color=PAL["grid"], zorder=0)
    ax.text(2.62, 0.077, "paper (Claude)\n6–10%", fontsize=7,
            color=PAL["muted"], va="center")
    models = [("olmo3-think", "OLMo-3-32B\nThink", PAL["J"]),
              ("olmo31-instruct", "Olmo-3.1-32B\nInstruct", PAL["nonJ"]),
              ("qwen36-27b", "Qwen3.6-27B", PAL["frozen_logit"])]
    for i, (slug, lab, col) in enumerate(models):
        old = d["old_proxy"].get(slug, {}).get("var_share_peak")
        new = d["new_estimator"][slug]["excess_peak"]
        if old:
            ax.scatter([i - 0.12], [old], s=64, facecolor="white",
                       edgecolor=col, linewidth=1.8, zorder=3,
                       label="old proxy (raw share)" if i == 0 else None)
            ax.annotate("", xy=(i + 0.12, new), xytext=(i - 0.12, old),
                        arrowprops=dict(arrowstyle="->", color=PAL["muted"],
                                        lw=1.1))
        ax.scatter([i + 0.12], [new], s=64, color=col, zorder=3,
                   label="paper estimator (excess)" if i == 0 else None)
        ax.text(i + 0.2, new, f"occ {d['new_estimator'][slug]['occ_median_band']}",
                fontsize=7.5, va="center", color=PAL["ink"])
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m[1] for m in models], fontsize=8)
    ax.set_ylabel("variance share (log)")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.grid(axis="y", which="both")
    ax.set_axisbelow(True)
    ax.set_title("Capacity errata: part-1 proxy vs the paper's estimator  "
                 "[pilot]", fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(FIGDIR / "p2f6_capacity_errata.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    log("wrote p2f6_capacity_errata.png")


FIGS = [f1_a0_transfer, f2_b3_frozen_logit, f3_r7_protected, f4_r2_occupancy,
        f5_crossmodel_protected, f6_capacity_errata]


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for f in FIGS:
        try:
            f()
        except Exception as e:  # a broken figure must never block a boundary
            log(f"FIGURE ERROR {f.__name__}: {e!r}")


if __name__ == "__main__":
    main()
