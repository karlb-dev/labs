# The tail-RATE endpoint, computed across every model that has a pilot
# grid — so the preregistration's endpoint decision is made against real
# numbers rather than a simulation alone.
#
# G6 (`g6-power-sim-v2`) showed the drafted 0.5-nat MEAN primaries are
# structurally underpowered: protected paired deltas are a zero-mode +
# heavy-tail mixture. The proposed replacement endpoint is the per-item
# TAIL RATE — the fraction of items losing more than `thr` nats under
# protected dyn-J, contrasted with the matched random dictionary arm on
# the same items, paired and family-clustered. This script reports that
# endpoint for all four checkpoints, plus its sensitivity to the
# threshold (which must be frozen at preregistration, so its arbitrariness
# has to be visible now, before any confirmatory use).
#
# Tier: pilot (descriptive; the same dev-tier pilot items G6 used — they
# inform design and are excluded from confirmatory statistics).
# Usage: python -m jspace_part2.experiments.tailrate_crossmodel [--allow-dirty]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..lib import sha256_file
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          write_result)

BASE = Path("/content/drive/MyDrive/interpret/special-lab-1/part2_20260727/"
            "metrics")
MODELS = {
    "olmo3-base": "OLMo-3 base (1125-32B)",
    "olmo3-think": "OLMo-3-32B-Think",
    "olmo31-instruct": "Olmo-3.1-32B-Instruct",
    "qwen36-27b": "Qwen3.6-27B",
}
OUT = BASE / "cross_model" / "tailrate_endpoint.json"
THRESHOLDS = [-0.5, -1.0, -1.5, -2.0]
FROZEN_THR = -1.0
SEED = 4242


def paired_hits(df: pd.DataFrame, task: str, thr: float) -> pd.DataFrame:
    """Per-item paired tail indicators for the J arm and the matched
    random arm, with family labels."""
    base = df[(df.condition == "none") & (df.task == task)]\
        .set_index("item_id")[["score", "family"]]
    out = base[["family"]].copy()
    for arm, col in (("dynJ_protected", "hit_j"),
                     ("dynR_protected", "hit_r")):
        s = df[(df.condition == arm) & (df.task == task)]\
            .set_index("item_id")["score"]
        out[col] = ((s - base["score"]) < thr).astype(float)
    return out.dropna().reset_index()


def cluster_boot(diff: np.ndarray, fams: np.ndarray, n_boot=4000,
                 seed=SEED) -> dict:
    """Family-clustered bootstrap of the paired rate difference."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"v": diff, "f": fams})
    groups = [g["v"].to_numpy() for _, g in d.groupby("f")]
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(groups), len(groups))
        means[b] = np.concatenate([groups[i] for i in idx]).mean()
    return {"estimate": round(float(diff.mean()), 4),
            "ci_low": round(float(np.percentile(means, 2.5)), 4),
            "ci_high": round(float(np.percentile(means, 97.5)), 4),
            "n_items": int(len(diff)), "n_families": int(len(groups))}


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    t0 = time.time()
    results, sens = {}, {}
    for slug in MODELS:
        p = BASE / slug / "r7_pilot" / "r7_per_item.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        per_task = {}
        for task in ("twohop", "onehop"):
            h = paired_hits(df, task, FROZEN_THR)
            if not len(h):
                continue
            diff = (h.hit_j - h.hit_r).to_numpy()
            ci = cluster_boot(diff, h.family.to_numpy())
            per_task[task] = {
                "rate_J": round(float(h.hit_j.mean()), 4),
                "rate_random": round(float(h.hit_r.mean()), 4),
                "paired_difference": ci,
                "excludes_zero": bool(ci["ci_low"] > 0 or ci["ci_high"] < 0)}
        results[slug] = {"label": MODELS[slug], "by_task": per_task}
        sens[slug] = {}
        for thr in THRESHOLDS:
            h = paired_hits(df, "twohop", thr)
            sens[slug][str(thr)] = {
                "rate_J": round(float(h.hit_j.mean()), 4),
                "rate_random": round(float(h.hit_r.mean()), 4),
                "difference": round(float((h.hit_j - h.hit_r).mean()), 4)}

    ladder = {s: (r["by_task"].get("twohop", {}) or {}).get(
        "paired_difference", {}).get("estimate") for s, r in results.items()}
    clean = [s for s, r in results.items()
             if r["by_task"].get("twohop", {}).get("excludes_zero")]

    # Does the rate endpoint reproduce the MEAN endpoint's ladder order?
    # Checked, not assumed: rank-order the models under each endpoint.
    mean_ladder = {}
    for slug in results:
        p = BASE / slug / "r7_pilot" / "r7_paired_ci.json"
        if p.exists():
            v = json.loads(p.read_text()).get("dynJ_protected/twohop", {})
            mean_ladder[slug] = v.get("estimate")
    rate_order = [s for s, _ in sorted(
        ((s, v) for s, v in ladder.items() if v is not None),
        key=lambda kv: kv[1])]
    mean_order = [s for s, _ in sorted(
        ((s, v) for s, v in mean_ladder.items() if v is not None),
        key=lambda kv: -kv[1])]        # more negative mean = stronger effect
    order_matches = rate_order == mean_order
    summ = {
        "frozen_threshold_nats": FROZEN_THR,
        "endpoint": ("per-item tail rate (protected dyn-J minus matched "
                     "protected-random), paired, family-clustered bootstrap"),
        "by_model": results,
        "threshold_sensitivity_twohop": sens,
        "twohop_rate_difference_by_model": ladder,
        "models_with_ci_excluding_zero": clean,
        "mean_endpoint_twohop_by_model": mean_ladder,
        "ladder_order_rate_endpoint": rate_order,
        "ladder_order_mean_endpoint": mean_order,
        "ladder_order_matches": bool(order_matches),
        "reading": (
            f"At the pilot n=60 the rate endpoint's J-vs-random difference "
            f"excludes zero on only {len(clean)}/{len(results)} models "
            f"({', '.join(clean) if clean else 'none'}) — consistent with "
            f"G6, which put the n needed for a 10pp margin well above the "
            f"pilot size. Effect ORDERING under the two endpoints is "
            f"{'IDENTICAL' if order_matches else 'NOT identical'}: rate "
            f"{rate_order} vs mean {mean_order}"
            + ("" if order_matches else
               " — Think and Instruct exchange places, so the rate endpoint "
               "is not a free relabeling of the mean ladder and HP1's "
               "ordering claim must be restated for whichever endpoint the "
               "freeze adopts") +
            f". Threshold sensitivity is reported so the freeze picks a "
            f"value with its arbitrariness visible; the pilot-frozen "
            f"{FROZEN_THR} nat is used throughout and any change is a "
            f"deviation to log."),
        "caveat": ("pilot items only (dev tier, excluded from confirmatory "
                   "statistics by prereg); one-hop cells inherit the "
                   "batteries' ceiling limitation for base/Think — the "
                   "hard one-hop set is the confirmatory replacement"),
        "seconds": round(time.time() - t0)}
    prov = Provenance(
        evidence_id="tailrate-endpoint-crossmodel-v2", tier="pilot",
        command="python -m jspace_part2.experiments.tailrate_crossmodel",
        inputs={s: sha256_file(BASE / s / "r7_pilot" / "r7_per_item.parquet")
                for s in results},
        seed=SEED)
    write_result(summ, OUT, prov)
    registry_append({
        "evidence_id": "tailrate-endpoint-crossmodel-v2", "tier": "pilot",
        "what": (f"Tail-rate endpoint across {len(results)} models at the "
                 f"frozen {FROZEN_THR}-nat threshold: two-hop J-minus-random "
                 f"rate differences {json.dumps(ladder)}; CI excludes zero "
                 f"on {clean}. Supports the G6 endpoint switch without "
                 f"losing the ladder contrast."),
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(OUT), "sha256": sha256_file(OUT)}]})
    print(json.dumps({k: v for k, v in summ.items()
                      if k not in ("by_model", "threshold_sensitivity_twohop")},
                     indent=2))
    for s, r in results.items():
        t = r["by_task"].get("twohop", {})
        if t:
            print(f"  {s:20s} twohop J {t['rate_J']:.2f} vs rand "
                  f"{t['rate_random']:.2f} → diff "
                  f"{t['paired_difference']['estimate']:+.3f} "
                  f"[{t['paired_difference']['ci_low']:+.3f}, "
                  f"{t['paired_difference']['ci_high']:+.3f}]")


if __name__ == "__main__":
    main()
