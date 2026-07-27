# v2 figures — regenerable from v2 metrics alone; each section is guarded by
# its metric file so the script always renders "whatever exists so far".
# Palette/rcParams: identical to v1 s9 (validated default dataviz palette;
# entity->hue mapping is fixed across ALL figures: J-space blue, random
# orange, non-J aqua, baseline ink). Figures land in v2 figures/ and are
# ALSO copied into the living handout's figures/ by refresh_handout.sh.
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl1_common import RUN_DIR_V2, atomic_write_json, log, read_json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from s9_report import C  # noqa: E402  (also applies the shared rcParams)

FIG = RUN_DIR_V2 / "figures"
M = RUN_DIR_V2 / "metrics"


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    log(f"figure {name}")


def fig_foils():
    d = read_json(M / "cot_foils.json")
    cls = d["classes"]
    names = [("answer", "answer", C["j"]), ("family", "family foils", C["rand"]),
             ("freq", "freq-matched words", C["nonj"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    xs = np.arange(3)
    for i, (k, label, color) in enumerate(names):
        v = cls[k]["det_rate"]
        ax.bar(i, v, width=0.62, color=color)
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", color=C["ink"],
                fontsize=10, fontweight="bold")
    ax.set_xticks(xs, [f"{l}\n(n={cls[k]['n_words']})"
                       for k, l, _ in names], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ever detected in workspace top-8")
    ax.set_title("Detection rate: answer vs foil floor", fontsize=11)

    ax = axes[1]
    for k, label, color in names[:2]:
        leads = [r["text"] - r["ws"] for it in d["per_item"]
                 for w, r in it["words"].items()
                 if w.startswith(k + ":") and r["ws"] is not None
                 and r["text"] is not None]
        if not leads:
            continue
        xsr = np.sort(leads)
        ax.step(xsr, np.arange(1, len(xsr) + 1) / len(xsr), where="post",
                color=color, lw=2, label=f"{label} (n={len(leads)})")
    ax.axvline(0, color=C["muted"], ls="--", lw=1)
    ax.text(4, 0.03, "workspace leads →", color=C["muted"], fontsize=9)
    fr = cls["freq"]
    ax.text(0.02, 0.97, f"freq-matched words: only {fr['n_with_both']} ever "
            f"in both\n(med lead {fr['med_lead']})", transform=ax.transAxes,
            va="top", fontsize=8.5, color=C["sec"])
    ax.set_xlabel("lead (text step − workspace step)")
    ax.set_ylabel("fraction of words")
    ax.set_title("Lead distribution where text mentions exist", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("cot-lead false-positive calibration (v2 P4)",
                 fontweight="bold")
    save(fig, "f9_foil_calibration.png")
    return {"answer_det": cls["answer"]["det_rate"],
            "family_det": cls["family"]["det_rate"],
            "freq_det": cls["freq"]["det_rate"],
            "answer_med_lead": cls["answer"]["med_lead"],
            "family_med_lead": cls["family"]["med_lead"],
            "answer_beats_all_foils": d["answer_earlier_than_all_foils_frac"]}


def fig_energy_match():
    d = read_json(M / "energy_match.json")
    layers = d["band"]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.axhspan(0.8, 1.25, color=C["grid"], alpha=0.6, lw=0)
    ax.axhline(1.0, color=C["muted"], ls="--", lw=1)
    for pool, color, mk in (("nonJ", C["nonj"], "o"), ("rand", C["rand"], "s")):
        for k, off in ((10, -0.5), (20, 0.0), (40, 0.5)):
            ys = [d["per_layer"][str(l)]["doses"][str(k)][f"{pool}_ratio"]
                  for l in layers]
            ax.plot(np.array(layers) + off, ys, mk, color=color, ms=4,
                    alpha=0.45 + 0.25 * (k == 20))
    ax.plot([], [], "o", color=C["nonj"], label="non-J PCs (matched)")
    ax.plot([], [], "s", color=C["rand"], label="random (matched)")
    ax.set_yscale("log")
    ax.set_xlabel("layer")
    ax.set_ylabel("removed energy / J-span target")
    ax.set_title("Energy-match verification (all doses; band = accepted)")
    ax.legend(frameon=False, fontsize=9)
    save(fig, "f10_energy_match.png")
    r = [d["per_layer"][str(l)]["doses"][str(k)][f"{p}_ratio"]
         for l in layers for k in (10, 20, 40) for p in ("nonJ", "rand")]
    return {"ratio_min": min(r), "ratio_max": max(r)}


def _forest(ax, ab, conds, tasks):
    y = np.arange(len(tasks))[::-1]
    for i, (cond, label, color, mk) in enumerate(conds):
        if cond not in ab["conditions"]:
            continue
        off = (i - (len(conds) - 1) / 2) * 0.8 / len(conds)
        for yi, (t, _) in zip(y, tasks):
            e = ab["conditions"][cond].get(t)
            if not e:
                continue
            ax.plot([e["ci_lo"], e["ci_hi"]], [yi + off] * 2, color=color,
                    lw=2, alpha=0.9)
            ax.plot(e["mean"], yi + off, mk, color=color, ms=6,
                    label=label if yi == y[0] else None)
    ax.set_yticks(y, [n for _, n in tasks])
    ax.set_xlim(-0.02, 1.02)


TASKS5 = [("twohop", "2-hop factual"), ("arithmetic_v2", "chained arithmetic"),
          ("sql", "multi-clause SQL"), ("onehop", "single-hop factual"),
          ("grammar", "grammaticality")]


def fig_vmatch():
    ab = read_json(M / "ablation_v2.json")
    fig, ax = plt.subplots(figsize=(9, 5))
    _forest(ax, ab, [("none", "baseline", C["ink"], "D"),
                     ("jspace_k20", "J-span k=20", C["j"], "o"),
                     ("vmatch_rand_k20", "random (energy-matched)",
                      C["rand"], "o"),
                     ("vmatch_nonJ_k20", "non-J PCs (energy-matched)",
                      C["nonj"], "o")], TASKS5)
    ax.set_xlabel("accuracy (bootstrap 95% CI)")
    ax.set_title("Variance-matched causal grid, k=20 (v2 P1)")
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    save(fig, "f11_vmatch_grid.png")
    cc = ab["conditions"]
    out = {}
    for cond in ("jspace_k20", "vmatch_rand_k20", "vmatch_nonJ_k20"):
        if cond in cc and "twohop_lp" in cc.get(cond, {}) \
                and "twohop_lp" in cc.get("none", {}):
            out[cond + "_twohop_lp_delta"] = round(
                cc[cond]["twohop_lp"]["mean"]
                - cc["none"]["twohop_lp"]["mean"], 3)
        if cond in cc and "sql" in cc.get(cond, {}):
            out[cond + "_sql"] = cc[cond]["sql"]["mean"]
    return out


def fig_frozen():
    ab = read_json(M / "frozen_ablation.json")
    fig, ax = plt.subplots(figsize=(9, 5))
    _forest(ax, ab, [("none", "baseline", C["ink"], "D"),
                     ("frozen_j10", "frozen J top-10", C["j"], "o"),
                     ("frozen_rand10", "frozen random top-10",
                      C["rand"], "o"),
                     ("live_j10", "live J top-10 (v1 dyn)", C["j"], "^"),
                     ("live_rand10", "live random top-10", C["rand"], "^")],
            TASKS5)
    ax.set_xlabel("accuracy (bootstrap 95% CI)")
    ax.set_title("Frozen vs live top-10 ablation (v2 P2); ^ = live selection")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    save(fig, "f12_frozen_grid.png")
    cc = ab["conditions"]
    return {c: {t: cc[c][t]["mean"] for t in ("twohop", "sql", "prose_nll")
                if t in cc[c]}
            for c in ("frozen_j10", "frozen_rand10", "live_rand10")
            if c in cc}


def fig_late_variance():
    v1 = read_json(Path(str(M).replace("2026-07-26_v2", "2026-07-25_1726"))
                   / "descriptive_agg.json")
    late = read_json(M / "descriptive_late.json")
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    xs1 = v1["layers"]
    ys1 = [v1["per_layer"][str(l)]["sum_recon_sq_c"]
           / max(v1["per_layer"][str(l)]["sum_h_sq_c"], 1e-9) for l in xs1]
    xs2 = late["layers"]
    ys2 = [late["per_layer"][str(l)]["sum_recon_sq_c"]
           / max(late["per_layer"][str(l)]["sum_h_sq_c"], 1e-9) for l in xs2]
    ax.plot(xs1, ys1, color=C["j"], lw=2, marker="o", ms=4)
    ax.plot(xs2, ys2, color=C["j"], lw=2, marker="o", ms=7,
            markerfacecolor=C["surface"])
    ax.annotate("v1 lens (21 layers)", (xs1[8], ys1[8]),
                textcoords="offset points", xytext=(0, 10),
                color=C["sec"], fontsize=9)
    ax.annotate("v2 late lens", (xs2[2], ys2[2]), textcoords="offset points",
                xytext=(0, 12), color=C["sec"], fontsize=9)
    ax.axhspan(0.06, 0.10, color=C["grid"], alpha=0.6, lw=0)
    ax.text(62, 0.101, "paper (Claude): 6–10%", ha="right",
            va="bottom", color=C["muted"], fontsize=9)
    ax.set_xlabel("layer")
    ax.set_ylabel("J-space share of centered activation variance")
    ax.set_title("Variance share with the late band fitted (v2 P3)")
    ax.set_ylim(bottom=0)
    save(fig, "f13_late_variance.png")
    return {f"L{l}": round(v, 4) for l, v in zip(xs2, ys2)}


def fig_late_profile():
    d = read_json(M / "late_answer_profile.json")
    v1 = read_json(Path(str(M).replace("2026-07-26_v2", "2026-07-25_1726"))
                   / "cot_lead.json")["anticipation_rank_by_layer_median"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    l1 = sorted(int(x) for x in v1["suppressed_jlens"])
    ax.plot(l1, [v1["suppressed_jlens"][str(l)] for l in l1], color=C["j"],
            lw=2, marker="o", ms=4)
    l2 = d["layers"]
    ax.plot(l2, [d["median_profile"]["sup_jlens"][str(l)] for l in l2],
            color=C["j"], lw=2, marker="o", ms=7,
            markerfacecolor=C["surface"])
    ax.plot(l1, [v1["suppressed_logit"][str(l)] for l in l1],
            color=C["logit"], lw=1.5, marker="s", ms=3)
    ax.plot(l2, [d["median_profile"]["sup_logit"][str(l)] for l in l2],
            color=C["logit"], lw=1.5, marker="s", ms=6,
            markerfacecolor=C["surface"])
    ax.set_yscale("log")
    ax.plot([], [], color=C["j"], marker="o", label="J-lens")
    ax.plot([], [], color=C["logit"], marker="s", label="logit lens")
    ax.set_xlabel("layer  (hollow markers = v2 late lens)")
    ax.set_ylabel("median answer rank, suppressed-CoT (log)")
    ax.set_title("Answer-time loading across the full depth (v2 P3)")
    ax.legend(frameon=False, fontsize=9)
    save(fig, "f14_late_answer_profile.png")
    return d["median_profile"]["sup_jlens"]


def fig_qwen():
    q = read_json(M / "qwen_causal_grid.json")
    fr = read_json(M / "frozen_ablation.json")
    ab = read_json(M / "ablation_v2.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    _forest(ax, q, [("none", "baseline", C["ink"], "D"),
                    ("frozen_j10", "frozen J top-10", C["j"], "o"),
                    ("frozen_rand10", "frozen random top-10",
                     C["rand"], "o")],
            [("twohop", "2-hop factual"), ("onehop", "single-hop factual"),
             ("arithmetic_v2", "chained arithmetic"),
             ("grammar", "grammaticality")])
    ax.set_xlabel("accuracy (bootstrap 95% CI)")
    ax.set_title("Qwen3.6-27B causal grid", fontsize=11)
    ax.legend(frameon=False, loc="lower left", fontsize=9)

    ax = axes[1]
    conds = [("frozen_j10", "frozen\nJ top-10", C["j"]),
             ("frozen_rand10", "frozen\nrand top-10", C["rand"]),
             ("jspace_k20", "J-span\nk=20", C["j"]),
             ("vmatch_rand_k20", "rand k20\n(matched)", C["rand"]),
             ("vmatch_nonJ_k20", "non-J k20\n(matched)", C["nonj"])]
    srcs = {"frozen_j10": fr, "frozen_rand10": fr}
    for i, (cond, label, color) in enumerate(conds):
        for model, src, mk, off in (
                ("OLMo", srcs.get(cond, ab), "o", -0.16),
                ("Qwen", q, "s", 0.16)):
            cc = src["conditions"]
            if cond not in cc or "twohop_lp" not in cc[cond] \
                    or "twohop_lp" not in cc.get("none", {}):
                continue
            e, b = cc[cond]["twohop_lp"], cc["none"]["twohop_lp"]
            delta = e["mean"] - b["mean"]
            half = (e["ci_hi"] - e["ci_lo"]) / 2
            face = color if model == "OLMo" else "none"
            ax.errorbar(i + off, delta, yerr=half, fmt=mk, color=color,
                        markerfacecolor=face, ms=7, capsize=3, lw=1.5)
    ax.axhline(0, color=C["muted"], ls="--", lw=1)
    ax.plot([], [], "o", color=C["ink"], label="OLMo-3-32B (filled)")
    ax.plot([], [], "s", color=C["ink"], markerfacecolor="none",
            label="Qwen3.6-27B (hollow)")
    ax.set_xticks(range(len(conds)), [c[1] for c in conds], fontsize=8.5)
    ax.set_ylabel("Δ 2-hop answer logprob vs own baseline (nats)")
    ax.set_title(r"$\Delta$ 2-hop logprob, both models", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("Phase Q: the causal pattern transfers to Qwen",
                 fontweight="bold")
    save(fig, "f15_qwen_grid.png")
    out = {"sanity": read_json(M / "qwen_sanity.json")["multihop"]
           if (M / "qwen_sanity.json").exists() else None}
    for cond in ("frozen_j10", "frozen_rand10", "jspace_k20",
                 "vmatch_rand_k20", "vmatch_nonJ_k20"):
        cc = q["conditions"]
        if cond in cc and "twohop_lp" in cc[cond]:
            out[cond + "_twohop_lp_delta"] = round(
                cc[cond]["twohop_lp"]["mean"]
                - cc["none"]["twohop_lp"]["mean"], 3)
        if cond in cc and "twohop" in cc[cond]:
            out[cond + "_twohop"] = cc[cond]["twohop"]["mean"]
    if (M / "qwen_descriptive.json").exists():
        qd = read_json(M / "qwen_descriptive.json")
        out["variance_share"] = {
            l: round(pl["sum_recon_sq_c"] / max(pl["sum_h_sq_c"], 1e-9), 4)
            for l, pl in qd["per_layer"].items()}
    return out


def fig_rescue():
    d = read_json(M / "cot_rescue.json")
    agg, ref = d["agg"], d.get("nothink_reference", {})
    kinds = [k for k in ("twohop", "onehop", "arithmetic") if k in agg
             and "frozen_j10" in agg[k]]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    xs = np.arange(len(kinds))
    # any-rate (answer anywhere in the trace) is the informative metric —
    # OLMo rarely closes </think> within the 400-token cap, so the
    # post-segment rate is degenerate (see handout §P5 scoring note).
    for j, (cond, label, color) in enumerate(
            (("frozen_j10", "frozen J top-10 + think", C["j"]),
             ("frozen_rand10", "frozen random + think", C["rand"]))):
        vs = [agg[k][cond]["any"] for k in kinds]
        ns = [agg[k][cond]["n"] for k in kinds]
        ax.bar(xs + (j - 0.5) * 0.34, vs, width=0.32, color=color,
               label=label)
        for x, v, n in zip(xs, vs, ns):
            ax.text(x + (j - 0.5) * 0.34, v + 0.02, f"{v:.2f}", ha="center",
                    fontsize=9, color=C["ink"])
    if "frozen_j10_twohop" in ref:
        ax.hlines(ref["frozen_j10_twohop"], -0.45, 0.45, color=C["j"],
                  ls=":", lw=2)
        ax.text(-0.42, ref["frozen_j10_twohop"] + 0.015,
                "frozen-J, no-think", fontsize=8, color=C["j"], ha="left")
    if "none_twohop" in ref:
        ax.hlines(ref["none_twohop"], -0.45, 0.45, color=C["ink"], ls="--",
                  lw=1.5)
        ax.text(-0.42, ref["none_twohop"] + 0.015, "baseline, no-think",
                fontsize=8, color=C["sec"], ha="left")
    ax.set_xticks(xs, [f"{k}\n(n={agg[k]['frozen_j10']['n']})"
                       for k in kinds])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("answer anywhere in trace (any-rate)")
    ax.set_title("P5: does externalized reasoning rescue the frozen-J "
                 "deletion?")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    save(fig, "f16_cot_rescue.png")
    return {k: {c: {"any": agg[k][c]["any"], "post": agg[k][c]["post"],
                    "closed": agg[k][c]["closed"]}
                for c in ("frozen_j10", "frozen_rand10") if c in agg[k]}
            for k in kinds} | {"ref": ref}


def fig_seed1():
    d = read_json(M / "robustness_seed1.json")
    cc = d["conditions"]
    conds = [("none", "baseline", C["ink"]),
             ("frozen_j10", "frozen J", C["j"]),
             ("frozen_rand10", "frozen rand", C["rand"]),
             ("jspace_k20", "J-span k20", C["j"]),
             ("vmatch_rand_k20", "rand k20", C["rand"]),
             ("vmatch_nonJ_k20", "non-J k20", C["nonj"])]
    cmp = d.get("seed_comparison", {})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, task, name in ((axes[0], "twohop", "2-hop accuracy"),
                           (axes[1], "twohop_lp", "2-hop answer logprob")):
        for i, (cond, label, color) in enumerate(conds):
            if cond not in cc or task not in cc[cond]:
                continue
            e = cc[cond][task]
            ax.errorbar(i + 0.12, e["mean"],
                        yerr=[[e["mean"] - e["ci_lo"]],
                              [e["ci_hi"] - e["mean"]]],
                        fmt="o", color=color, ms=6, capsize=3,
                        markerfacecolor="none")
            s0 = cmp.get(cond, {}).get(task, {}).get("seed0")
            if s0 is not None:
                ax.plot(i - 0.12, s0, "o", color=color, ms=6)
                ax.plot([i - 0.12, i + 0.12], [s0, e["mean"]], color=color,
                        lw=1, alpha=0.5)
        ax.set_xticks(range(len(conds)),
                      [c[1] for c in conds], rotation=30, ha="right",
                      fontsize=8.5)
        ax.set_title(name, fontsize=11)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].plot([], [], "o", color=C["ink"], label="seed 0 (filled)")
    axes[0].plot([], [], "o", color=C["ink"], markerfacecolor="none",
                 label="seed 1, fresh items (hollow)")
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("P6: the frozen dissociation under a second seed + fresh "
                 "2-hop items", fontweight="bold")
    save(fig, "f17_seed1.png")
    return {c: cmp.get(c, {}) for c, _, _ in conds if c in cmp}


def main() -> None:
    have = lambda p: (M / p).exists()
    summary = {}
    if have("cot_foils.json"):
        summary["foils"] = fig_foils()
    if have("energy_match.json"):
        summary["energy_match"] = fig_energy_match()
    if have("ablation_v2.json"):
        summary["vmatch"] = fig_vmatch()
    if have("frozen_ablation.json"):
        summary["frozen"] = fig_frozen()
    if have("descriptive_late.json"):
        summary["late_variance"] = fig_late_variance()
    if have("late_answer_profile.json") and \
            "median_profile" in read_json(M / "late_answer_profile.json"):
        summary["late_profile"] = fig_late_profile()
    if have("cot_lead_late.json"):
        d = read_json(M / "cot_lead_late.json")
        if "summary" in d:
            summary["cot_lead_late"] = d["summary"]
    if have("qwen_causal_grid.json"):
        summary["qwen"] = fig_qwen()
    if have("cot_rescue.json") and "agg" in read_json(M / "cot_rescue.json"):
        summary["rescue"] = fig_rescue()
    if have("robustness_seed1.json"):
        summary["seed1"] = fig_seed1()
    atomic_write_json(summary, RUN_DIR_V2 / "report" / "summary_v2.json")
    log("wrote report/summary_v2.json")


if __name__ == "__main__":
    main()
