# R4 — paired clustered CIs for the protected grids, INSIDE the package.
#
# Why this exists: the `r7_paired_ci.json` files on Drive carry the
# campaign's headline intervals (every dissociation claim, the ladder
# figure, the handout) but were produced during VM6 by an uncommitted
# ad-hoc script — no provenance block, no registry entry. Under the repro
# contract that makes them orphan artifacts: cited but non-existent. This
# module recomputes them from the REGISTERED per-item parquets, writes
# provenance, registers the result, and reports whether the recomputation
# agrees with the orphan file it supersedes (so the already-published
# numbers are either confirmed or visibly corrected).
#
# Statistics (addendum §12.2): paired per-item deltas vs the clean
# baseline; family-clustered bootstrap (resample families with
# replacement, 4000 draws, percentile interval); clusters are the item
# families as recorded by the battery.
#
# Tier: pilot. CPU-only, seconds per model.
# Usage: python -m jspace_part2.experiments.r7_paired_ci [--model SLUG]
#          [--allow-dirty]
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
MODELS = ["olmo3-base", "olmo3-think", "olmo31-instruct", "qwen36-27b"]
ARMS = ["dynJ_protected", "dynJ_unprotected", "dynR_protected"]
TASKS = ["twohop", "onehop", "prose"]
N_BOOT, SEED = 4000, 4242
AGREE_TOL = 0.02      # nats; recomputation vs the orphan file


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def paired_ci(df: pd.DataFrame, arm: str, task: str, rng) -> dict | None:
    base = df[(df.condition == "none") & (df.task == task)]\
        .set_index("item_id")[["score", "family"]]
    s = df[(df.condition == arm) & (df.task == task)]\
        .set_index("item_id")["score"]
    d = base.assign(delta=s - base["score"]).dropna(subset=["delta"])
    if not len(d):
        return None
    groups = [g["delta"].to_numpy() for _, g in d.groupby("family")]
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, len(groups), len(groups))
        means[b] = np.concatenate([groups[i] for i in idx]).mean()
    return {"estimate": round(float(d["delta"].mean()), 3),
            "ci_low": round(float(np.percentile(means, 2.5)), 3),
            "ci_high": round(float(np.percentile(means, 97.5)), 3),
            "n_items": int(len(d)), "n_clusters": int(len(groups))}


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    only = arg("--model")
    models = [only] if only else MODELS
    t0 = time.time()
    all_out, agreement = {}, {}

    for slug in models:
        p = BASE / slug / "r7_pilot" / "r7_per_item.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        rng = np.random.default_rng(SEED)
        res = {}
        for task in TASKS:
            for arm in ARMS:
                ci = paired_ci(df, arm, task, rng)
                if ci:
                    res[f"{arm}/{task}"] = ci
        out = BASE / slug / "r7_pilot" / "r7_paired_ci_v2.json"

        # compare against the orphan file this supersedes
        old_p = BASE / slug / "r7_pilot" / "r7_paired_ci.json"
        diffs = {}
        if old_p.exists():
            old = json.loads(old_p.read_text())
            for k, v in res.items():
                if k in old and "estimate" in old[k]:
                    delta = abs(v["estimate"] - old[k]["estimate"])
                    if delta > AGREE_TOL:
                        diffs[k] = {"recomputed": v["estimate"],
                                    "orphan_file": old[k]["estimate"],
                                    "abs_diff": round(delta, 3)}
        agreement[slug] = {"cells_compared": len(res),
                           "cells_disagreeing": len(diffs), "diffs": diffs}

        prov = Provenance(
            evidence_id=f"r7-paired-ci-{slug}-v1", tier="pilot",
            command=("python -m jspace_part2.experiments.r7_paired_ci "
                     f"--model {slug}"),
            inputs={"per_item": sha256_file(p)}, seed=SEED)
        write_result({"cells": res, "method": {
            "estimand": "paired per-item delta vs clean baseline",
            "resample": "family-clustered bootstrap, percentile interval",
            "n_boot": N_BOOT, "seed": SEED},
            "supersedes_orphan": str(old_p) if old_p.exists() else None,
            "agreement_with_orphan": agreement[slug]}, out, prov)
        registry_append({
            "evidence_id": f"r7-paired-ci-{slug}-v1", "tier": "pilot",
            "what": (f"Paired family-clustered CIs for the {slug} protected "
                     f"grid, recomputed inside the package from the "
                     f"registered per-item parquet ({len(res)} cells). "
                     f"Closes the orphan `r7_paired_ci.json` (no provenance, "
                     f"no registry entry) whose numbers the report, handout "
                     f"and figures cite. Cells disagreeing with the orphan "
                     f"beyond {AGREE_TOL} nats: {len(diffs)}"
                     + (f" — {json.dumps(diffs)}" if diffs else "")),
            "command": prov.command, "code_commit": git["code_commit"],
            "rerun": "auto",
            "outputs": [{"path": str(out), "sha256": sha256_file(out)}]})
        all_out[slug] = res

    total_diff = sum(a["cells_disagreeing"] for a in agreement.values())
    print(json.dumps({"models": list(all_out), "agreement": agreement,
                      "total_disagreeing_cells": total_diff,
                      "verdict": ("published intervals CONFIRMED by an "
                                  "independent in-package recomputation"
                                  if total_diff == 0 else
                                  "DISAGREEMENT — published numbers need "
                                  "correction, see diffs"),
                      "seconds": round(time.time() - t0)}, indent=2))


if __name__ == "__main__":
    main()
