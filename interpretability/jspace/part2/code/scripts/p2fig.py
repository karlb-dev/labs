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
    ax.set_title("A0 — donor lens transfers the shortlist, not the rank-1 edge",
                 fontsize=9.5, loc="left")

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
                 "(OLMo-3-32B-Think, per-item frozen top-10, 95% boot CI)",
                 fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {out}")


FIGS = [f1_a0_transfer, f2_b3_frozen_logit]


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for f in FIGS:
        try:
            f()
        except Exception as e:  # a broken figure must never block a boundary
            log(f"FIGURE ERROR {f.__name__}: {e!r}")


if __name__ == "__main__":
    main()
