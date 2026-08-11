"""Generate PNG figures from Muse metrics (CPU-safe)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .paths import DRIVE_ROOT, ensure_dirs
from .util import log


def _load(name: str):
    p = DRIVE_ROOT / "metrics" / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def fig_depth_profile(out: Path) -> Path | None:
    data = _load("depth_profile_muse.json")
    if not data:
        return None
    # aggregate mean ranks across prompts
    layers = {}
    for row in data["rows"]:
        if "per_layer" not in row:
            continue
        for p in row["per_layer"]:
            L = p["layer"]
            layers.setdefault(L, {"logit": [], "j": []})
            layers[L]["logit"].append(p["logit_rank"])
            if p["j_rank"] is not None:
                layers[L]["j"].append(p["j_rank"])
    Ls = sorted(layers)
    logit_m = [np.mean(layers[L]["logit"]) for L in Ls]
    j_m = [np.mean(layers[L]["j"]) if layers[L]["j"] else np.nan for L in Ls]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(Ls, logit_m, "o-", label="logit lens", color="#4C72B0")
    ax.plot(Ls, j_m, "s-", label="J-lens", color="#DD8452")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean target rank (lower = better)")
    ax.set_title("Muse Glimmer: J-lens vs logit-lens depth profile")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_headline_bars(out: Path) -> Path | None:
    summary = _load("battery_summary.json")
    if not summary or "headline" not in summary:
        return None
    h = summary["headline"]
    labels = []
    values = []
    colors = []
    # normalize a few headline numbers into displayable bars
    items = [
        ("J adv (band ranks)", h.get("depth_j_advantage_band"), 1),
        ("Selectivity contrast", h.get("selectivity_contrast"), 1),
        ("VR top5 hit rate", h.get("vr_top5_hit_rate"), 1),
        ("Ablation J dmg", h.get("ablation_j_damage"), 1),
        ("Ablation rnd dmg", h.get("ablation_random_damage"), 1),
        ("Capacity mean@r5", h.get("capacity_mean_r5"), 1),
    ]
    for lab, val, _ in items:
        if val is None:
            continue
        labels.append(lab)
        values.append(float(val))
        colors.append("#55A868" if val and val > 0 else "#C44E52")
    if not labels:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Muse Glimmer jspace battery — headline metrics")
    ax.set_xlabel("Value (cell-native units)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_modulation(out: Path) -> Path | None:
    data = _load("directed_modulation_muse.json")
    if not data:
        return None
    concepts = []
    a0, a2, am1 = [], [], []
    for r in data["rows"]:
        if "alpha_0.0" not in r:
            continue
        concepts.append(r["concept"])
        a0.append(r["alpha_0.0"])
        a2.append(r["alpha_2.0"])
        am1.append(r["alpha_-1.0"])
    if not concepts:
        return None
    x = np.arange(len(concepts))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w, a0, w, label="α=0", color="#4C72B0")
    ax.bar(x, a2, w, label="α=+2 focus", color="#55A868")
    ax.bar(x + w, am1, w, label="α=-1 suppress", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(concepts, rotation=30, ha="right")
    ax.set_ylabel("Target rank (lower=better)")
    ax.set_yscale("log")
    ax.set_title("Directed modulation (inject concept vector)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_ignition(out: Path) -> Path | None:
    data = _load("ignition_muse.json")
    if not data:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    for r in data["rows"]:
        if "curve" not in r:
            continue
        alphas = [c["alpha"] for c in r["curve"]]
        ranks = [c["rank_a"] for c in r["curve"]]
        ax.plot(alphas, ranks, "-o", ms=3, label="→".join(r["pair"]))
    ax.set_xlabel("Injection α")
    ax.set_ylabel("Rank of concept A")
    ax.set_yscale("log")
    ax.set_title("Ignition: rank vs injection strength")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_boot_trajectory(out: Path) -> Path | None:
    data = _load("admission_pre_fit.json")
    if not data:
        return None
    boot = data.get("boot_sentinel_logit_lens", {})
    rows = boot.get("per_layer", [])
    if not rows:
        return None
    # show top-1 token string per layer
    layers = [r["layer"] for r in rows]
    top1 = [r["top5"][0] if r["top5"] else "?" for r in rows]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.scatter(layers, [1] * len(layers), s=0)  # keep axes
    for L, t in zip(layers, top1):
        ax.text(L, 1, t.replace("\n", " "), rotation=45, ha="right", va="bottom", fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("Layer")
    ax.set_title(f"Logit-lens top-1 trajectory: {boot.get('prompt', '')[:50]}...")
    ax.set_xlim(min(layers) - 1, max(layers) + 1)
    ax.set_ylim(0.5, 1.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def generate_all() -> list[Path]:
    ensure_dirs()
    fig_dir = DRIVE_ROOT / "figures"
    outs = []
    for name, fn in [
        ("fig_depth_profile.png", fig_depth_profile),
        ("fig_headline_bars.png", fig_headline_bars),
        ("fig_modulation.png", fig_modulation),
        ("fig_ignition.png", fig_ignition),
        ("fig_boot_trajectory.png", fig_boot_trajectory),
    ]:
        p = fn(fig_dir / name)
        if p:
            log(f"wrote {p}")
            outs.append(p)
    return outs


if __name__ == "__main__":
    generate_all()
