# N2 — recompute every family-clustered pilot statistic under the AUDITED
# canonical family map (nextsteps_2_2 §2.1 gate, §7-N2).
#
# WHY. Every published paired CI, tail-rate interval, ICC and the whole G6
# power simulation clustered on `battery.py`'s `name.split("-")[0]`, which
# is a string accident rather than the data-generating unit. Under the
# audited map the pilot's 60 two-hop items are 25 canonical families, not
# 38 raw labels, and one family (`country_capital`) holds 11 items. Fewer,
# larger clusters mean LESS independent information, so the corrected
# intervals should widen. This module measures by how much, and whether
# any conclusion changes.
#
# It supersedes, with new evidence ids, and it prints the old number next
# to the new one for every cell — the point is to see the correction, not
# to quietly republish.
#
# Nothing here re-runs a model: all three statistics are functions of the
# registered per-item parquets. CPU, ~1 minute.
#
# Usage: python -m jspace_part2.experiments.n2_corrected_pilot [--allow-dirty]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..family import attach_family, audit
from ..lib import sha256_file
from ..paths import to_uri
from ..provenance import (Provenance, git_info, registry_append,
                          require_clean_tree, write_result_v2)

BASE = Path("/content/drive/MyDrive/interpret/special-lab-1/part2_20260727/"
            "metrics")
MODELS = ["olmo3-base", "olmo3-think", "olmo31-instruct", "qwen36-27b"]
ARMS = ["dynJ_protected", "dynJ_unprotected", "dynR_protected"]
TASKS = ["twohop", "onehop", "prose"]
N_BOOT, SEED = 4000, 4242
THRESHOLDS = [-0.5, -1.0, -1.5, -2.0]
FROZEN_THR = -1.0
OUT_DIR = BASE / "cross_model"


def load(slug: str) -> pd.DataFrame:
    p = BASE / slug / "r7_pilot" / "r7_per_item.parquet"
    df = pd.read_parquet(p)
    # prose items are not in the item map (they are corpus windows, each
    # its own unit); keep their legacy per-window family.
    prose = df[df.task == "prose"].copy()
    prose["canonical_family"] = prose["family"]
    prose["family_legacy"] = prose["family"]
    prose["template_id"] = "prose_window"
    rest = attach_family(df[df.task != "prose"])
    return pd.concat([rest, prose], ignore_index=True), p


def paired(df: pd.DataFrame, arm: str, task: str, fam_col: str) -> pd.DataFrame:
    base = df[(df.condition == "none") & (df.task == task)]\
        .set_index("item_id")[["score", fam_col]]
    s = df[(df.condition == arm) & (df.task == task)]\
        .set_index("item_id")["score"]
    d = base.assign(delta=s - base["score"]).dropna(subset=["delta"])
    return d.reset_index()[["item_id", fam_col, "delta"]]


def cluster_ci(d: pd.DataFrame, fam_col: str, rng, *, value="delta",
               equal_family_weight=False) -> dict:
    """Family-clustered percentile bootstrap.

    equal_family_weight implements the estimand choice nextsteps_2_2 §9.1
    demands be made explicit: item-weighted (concatenate resampled
    families, larger families count more) vs family-weighted (mean of
    family means). Both are reported; family-weighted is the recommended
    primary for generalisation across relation families."""
    groups = [g[value].to_numpy() for _, g in d.groupby(fam_col)]
    if len(groups) < 2:
        return {}
    stats = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, len(groups), len(groups))
        if equal_family_weight:
            stats[b] = np.mean([groups[i].mean() for i in idx])
        else:
            stats[b] = np.concatenate([groups[i] for i in idx]).mean()
    obs = (np.mean([g.mean() for g in groups]) if equal_family_weight
           else d[value].mean())
    return {"estimate": round(float(obs), 3),
            "ci_low": round(float(np.percentile(stats, 2.5)), 3),
            "ci_high": round(float(np.percentile(stats, 97.5)), 3),
            "n_items": int(len(d)), "n_clusters": int(len(groups))}


def icc(d: pd.DataFrame, fam_col: str) -> dict:
    g = d.groupby(fam_col)["delta"]
    sizes = g.size()
    multi = sizes[sizes >= 2].index
    sig2_e = (float(np.mean([d[d[fam_col] == f]["delta"].var(ddof=1)
                             for f in multi])) if len(multi)
              else float(d["delta"].var(ddof=1)))
    var_fm = float(g.mean().var(ddof=1))
    sig2_f = max(0.0, var_fm - sig2_e * float(np.mean(1.0 / sizes)))
    tot = sig2_f + sig2_e
    return {"n_families": int(sizes.size), "n_items": int(len(d)),
            "max_family_size": int(sizes.max()),
            "sig_f": round(float(np.sqrt(sig2_f)), 4),
            "sig_e": round(float(np.sqrt(sig2_e)), 4),
            "icc": round(sig2_f / tot, 4) if tot > 0 else 0.0}


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    t0 = time.time()
    fam_audit = audit()
    frames, srcs = {}, {}
    for slug in MODELS:
        frames[slug], srcs[slug] = load(slug)

    # ---------------------------------------------------- 1. paired CIs
    ci_new, ci_old, widen = {}, {}, []
    for slug, df in frames.items():
        rng_n = np.random.default_rng(SEED)
        rng_o = np.random.default_rng(SEED)
        for task in TASKS:
            for arm in ARMS:
                dn = paired(df, arm, task, "canonical_family")
                do = paired(df, arm, task, "family_legacy")
                if not len(dn):
                    continue
                key = f"{slug}/{arm}/{task}"
                a = cluster_ci(dn, "canonical_family", rng_n)
                b = cluster_ci(do, "family_legacy", rng_o)
                if not a or not b:
                    continue
                a["family_weighted"] = cluster_ci(
                    dn, "canonical_family", np.random.default_rng(SEED),
                    equal_family_weight=True)
                ci_new[key], ci_old[key] = a, b
                wn = a["ci_high"] - a["ci_low"]
                wo = b["ci_high"] - b["ci_low"]
                flip = (a["ci_low"] < 0 < a["ci_high"]) != \
                       (b["ci_low"] < 0 < b["ci_high"])
                widen.append({"cell": key, "width_old": round(wo, 3),
                              "width_new": round(wn, 3),
                              "ratio": round(wn / wo, 3) if wo else None,
                              "clusters_old": b["n_clusters"],
                              "clusters_new": a["n_clusters"],
                              "zero_crossing_changed": bool(flip)})

    flips = [w for w in widen if w["zero_crossing_changed"]]
    key_cells = {k: {"corrected": ci_new[k], "legacy": ci_old[k]}
                 for k in ci_new
                 if k.endswith("dynJ_protected/twohop")
                 or k.endswith("dynJ_protected/onehop")}

    # ---------------------------------------------------------- 2. ICC
    icc_new = {f"{s}/twohop": icc(paired(df, "dynJ_protected", "twohop",
                                         "canonical_family"),
                                 "canonical_family")
               for s, df in frames.items()}
    icc_old = {f"{s}/twohop": icc(paired(df, "dynJ_protected", "twohop",
                                         "family_legacy"), "family_legacy")
               for s, df in frames.items()}

    # ------------------------------------------------ 3. tail-rate endpoint
    tail = {}
    for slug, df in frames.items():
        rng = np.random.default_rng(SEED)
        per_thr = {}
        for thr in THRESHOLDS:
            base = df[(df.condition == "none") & (df.task == "twohop")]\
                .set_index("item_id")[["score", "canonical_family"]]
            hits = base[["canonical_family"]].copy()
            for arm, col in (("dynJ_protected", "hit_j"),
                             ("dynR_protected", "hit_r")):
                s = df[(df.condition == arm) & (df.task == "twohop")]\
                    .set_index("item_id")["score"]
                hits[col] = ((s - base["score"]) < thr).astype(float)
            hits = hits.dropna().reset_index()
            hits["delta"] = hits.hit_j - hits.hit_r
            ci = cluster_ci(hits, "canonical_family", rng)
            ci_fw = cluster_ci(hits, "canonical_family",
                               np.random.default_rng(SEED),
                               equal_family_weight=True)
            per_thr[str(thr)] = {"rate_j": round(float(hits.hit_j.mean()), 3),
                                 "rate_r": round(float(hits.hit_r.mean()), 3),
                                 "paired_diff": ci,
                                 "paired_diff_family_weighted": ci_fw}
        tail[slug] = per_thr

    order_mean = sorted(MODELS,
                        key=lambda s: ci_new[f"{s}/dynJ_protected/twohop"]["estimate"])
    order_rate = sorted(MODELS,
                        key=lambda s: -tail[s][str(FROZEN_THR)]["paired_diff"]["estimate"])

    payload = {
        "family_map_audit": fam_audit,
        "paired_ci_corrected": ci_new,
        "paired_ci_legacy": ci_old,
        "interval_width_change": widen,
        "cells_whose_zero_crossing_CHANGED": flips,
        "key_cells": key_cells,
        "icc_corrected": icc_new, "icc_legacy": icc_old,
        "tailrate_corrected": tail,
        "model_order_mean_endpoint_twohop": order_mean,
        "model_order_tailrate_endpoint_twohop": order_rate,
        "endpoints_agree_on_order": order_mean == list(reversed(order_rate))
                                    or order_mean == order_rate,
        "method": {
            "family": "audited canonical map (data/probe_swap_family_map.json)",
            "bootstrap": "family-clustered percentile, 4000 draws, seed 4242",
            "weighting": ("item-weighted primary (continuity with the pilot); "
                          "family-weighted reported alongside per §9.1 — the "
                          "confirmatory estimand choice is made in the "
                          "preregistration candidate, not here"),
            "thresholds": THRESHOLDS, "frozen_threshold": FROZEN_THR,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "n2_corrected_pilot.json"
    prov = Provenance(
        evidence_id="n2-corrected-family-pilot-v1", tier="pilot",
        command="python -m jspace_part2.experiments.n2_corrected_pilot",
        inputs={s: sha256_file(p) for s, p in srcs.items()}, seed=SEED)
    env = write_result_v2(payload, out, prov)

    med_ratio = float(np.median([w["ratio"] for w in widen if w["ratio"]]))
    what = (f"N2 recomputation of every family-clustered pilot statistic "
            f"under the AUDITED canonical family map (supersedes the four "
            f"r7-paired-ci-* items and tailrate-endpoint-crossmodel-v2, "
            f"all of which clustered on name.split('-')[0]). Pilot two-hop "
            f"set: 25 canonical families, not 38 raw labels; largest family "
            f"holds {max(s for _, s in fam_audit['largest_families'])} items. "
            f"Median CI width ratio corrected/legacy = {med_ratio:.2f} over "
            f"{len(widen)} cells; {len(flips)} cell(s) changed whether the "
            f"interval excludes zero"
            + (f" — {[f['cell'] for f in flips]}" if flips else "") + ". "
            f"Mean-endpoint model order {order_mean}; tail-rate order "
            f"{order_rate}.")
    registry_append({
        "evidence_id": "n2-corrected-family-pilot-v1", "tier": "pilot",
        "what": what, "command": prov.command,
        "code_commit": git["code_commit"], "rerun": "auto",
        "input_uris": {s: to_uri(str(p)) for s, p in srcs.items()},
        "inputs": {s: sha256_file(p) for s, p in srcs.items()},
        "outputs": [{"path": str(out), "uri": to_uri(str(out)),
                     "sha256": sha256_file(out),
                     "payload_sha256": env["payload_sha256"]}],
        "repro_notes": ("Deterministic given the parquets and seed 4242; "
                        "payload_sha256 must match exactly on rerun.")})
    for old in ("r7-paired-ci-olmo3-base-v1", "r7-paired-ci-olmo3-think-v1",
                "r7-paired-ci-olmo31-instruct-v1", "r7-paired-ci-qwen36-27b-v1",
                "tailrate-endpoint-crossmodel-v2"):
        try:
            from .. import registry as reg
            reg.supersede(old, "n2-corrected-family-pilot-v1",
                          reason="clustered on the defective prefix family field")
        except Exception as e:                       # already superseded
            print(f"  (supersede {old}: {e})")

    print(json.dumps({"family_audit": {k: fam_audit[k] for k in
                                       ("n_items", "n_families", "singletons")},
                      "median_ci_width_ratio": round(med_ratio, 3),
                      "cells_changing_zero_crossing": flips,
                      "key_cells": key_cells,
                      "icc_corrected": icc_new, "icc_legacy": icc_old,
                      "order_mean": order_mean, "order_rate": order_rate,
                      "seconds": round(time.time() - t0)}, indent=1))


if __name__ == "__main__":
    main()
