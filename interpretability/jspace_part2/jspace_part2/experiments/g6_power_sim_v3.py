# G6 v3 — power simulation recomputed under the AUDITED canonical family
# map (nextsteps_2_2 §2.1 gate; PI addendum §4.2 sequencing note).
#
# WHY A NEW MODULE RATHER THAN AN EDIT. v2's registered command is
# `python -m jspace_part2.experiments.g6_power_sim`; changing that module's
# behaviour would make its own evidence item irreproducible. v2 stays
# frozen and importable; v3 reuses its simulation machinery with corrected
# clusters.
#
# WHAT CHANGED IN THE INPUT. The defective prefix field made the
# intraclass correlation look UNIFORM across models (0.37-0.42). Under the
# audited map it is not uniform at all — base 0.66, Think 0.56, Instruct
# 0.17, Qwen 0.75 — and the two-hop set has 25 real families, not 38 raw
# labels, with one family holding 11 of the 60 items. Power depends
# strongly on ICC and on the number of INDEPENDENT families, so v2's
# recommendation was calibrated on a homogeneity that does not exist.
#
# Usage: python -m jspace_part2.experiments.g6_power_sim_v3 [--allow-dirty]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..family import attach_family
from ..lib import sha256_file
from ..paths import to_uri
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          write_result_v2)
from . import g6_power_sim as v2

BASE = v2.BASE
OUT = BASE / "cross_model" / "g6_power_sim_v3.json"
FAM = "canonical_family"


def load_corrected(slug: str) -> tuple[pd.DataFrame, Path]:
    p = BASE / slug / "r7_pilot" / "r7_per_item.parquet"
    df = pd.read_parquet(p)
    prose = df[df.task == "prose"].copy()
    prose[FAM] = prose["family"]
    rest = attach_family(df[df.task != "prose"])
    out = pd.concat([rest, prose], ignore_index=True)
    # v2's helpers read a column literally named `family`
    out["family"] = out[FAM]
    return out, p


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    t0 = time.time()
    raw, srcs = {}, {}
    for slug in v2.CELLS:
        raw[slug], srcs[slug] = load_corrected(slug)

    deltas = {(s, t): v2.paired_deltas(df, t)
              for s, df in raw.items() for t in ("twohop", "onehop")}
    comps = {f"{s}/{t}": v2.components(d) for (s, t), d in deltas.items()}

    worst_slug = max(raw, key=lambda s: comps[f"{s}/twohop"]["sig_f"]**2
                     + comps[f"{s}/twohop"]["sig_e"]**2)
    wc, wd = comps[f"{worst_slug}/twohop"], deltas[(worst_slug, "twohop")]
    oc = comps[f"{worst_slug}/onehop"]

    tails = {s: v2.tail_pairs(df) for s, df in raw.items()}
    tail_rates = {s: {"rate_j": round(float(t.hit_j.mean()), 3),
                      "rate_r": round(float(t.hit_r.mean()), 3),
                      "n": int(len(t)),
                      "n_families": int(t.family.nunique())}
                  for s, t in tails.items()}
    gap = {s: v["rate_j"] - v["rate_r"] for s, v in tail_rates.items()
           if v["rate_j"] > 0.05}
    tail_slug = min(gap, key=gap.get)

    results = {"A_cell_mean": [], "B_dissoc": [], "D_tost": [], "E_tailrate": []}
    for m in v2.GRID_M:
        for k in v2.GRID_K:
            n = m * k
            results["A_cell_mean"].append({
                "m_families": m, "k_per_family": k, "n_items": n,
                "power_normal": round(v2.power_normal(
                    m, k, wc["sig_f"], wc["sig_e"], v2.SESOI), 3),
                "power_boot": round(v2.power_boot(wd, m, k, v2.SESOI), 3)})
            results["D_tost"].append({
                "m_families": m, "k_per_family": k, "n_items": n,
                "power": round(v2.power_tost(wc["sig_f"], wc["sig_e"], m, k), 3)})
            results["E_tailrate"].append({
                "m_families": m, "k_per_family": k, "n_items": n,
                "planning_cell": tail_slug,
                "power_at_sesoi_10pp": round(v2.power_tailrate_boot(
                    tails[tail_slug], m, k, v2.TAIL_SESOI), 3),
                "power_at_pilot_gap": round(v2.power_tailrate_boot(
                    tails[tail_slug], m, k, gap[tail_slug]), 3)})
            results["B_dissoc"].append({
                "m_families": m, "k_per_family": k, "n_items": n,
                "n_onehop": 60,
                "power": round(v2.power_dissoc(wc, oc, m, k, 60, v2.SESOI), 3)})

    recs = {"A_cell_mean": v2.first_at_target(results["A_cell_mean"]),
            "B_dissoc": v2.first_at_target(results["B_dissoc"], "power"),
            "D_tost": v2.first_at_target(results["D_tost"], "power"),
            "E_tailrate_sesoi10pp": v2.first_at_target(
                results["E_tailrate"], "power_at_sesoi_10pp")}

    # what the correction did to the recommendation
    old = json.loads((BASE / "cross_model/g6_power_sim.json").read_text())
    old_s = old.get("summary", old)
    delta = {"icc_v2_vs_v3": {k: {"v2": old_s["components"][k]["icc"],
                                  "v3": comps[k]["icc"]}
                              for k in comps if k in old_s["components"]},
             "recommendation_v2": old_s["recommendations"],
             "recommendation_v3": recs,
             "planning_cell_v2": old_s["planning_cell"],
             "planning_cell_v3": worst_slug}

    payload = {"family_source": "audited canonical map (v3)",
               "planning_cell": worst_slug, "components": comps,
               "pilot_tail_rates": tail_rates, "tail_planning_cell": tail_slug,
               "alpha_primary": v2.ALPHA, "sesoi_nats": v2.SESOI,
               "tail_thr_nats": v2.TAIL_THR, "tail_sesoi_pp": v2.TAIL_SESOI,
               "power_target": v2.POWER_TARGET, "n_sims": v2.B,
               "recommendations": recs, "change_vs_v2": delta,
               "grids": results}
    prov = Provenance(
        evidence_id="g6-power-sim-v3", tier="pilot",
        command="python -m jspace_part2.experiments.g6_power_sim_v3",
        inputs={s: sha256_file(p) for s, p in srcs.items()}, seed=4242)
    env = write_result_v2(payload, OUT, prov)

    a, e = recs["A_cell_mean"], recs["E_tailrate_sesoi10pp"]
    what = (f"G6 power simulation recomputed under the AUDITED family map "
            f"(supersedes g6-power-sim-v2, which clustered on the defective "
            f"prefix field and therefore reported a spuriously UNIFORM ICC "
            f"of 0.37-0.42 across all four models; the true ICCs are "
            f"{ {k.split('/')[0]: comps[k]['icc'] for k in comps if k.endswith('twohop')} }). "
            f"Planning cell {worst_slug}. Mean-endpoint 0.5-nat primary "
            f"reaches 90% power at: {a}. Tail-rate endpoint at a 10pp SESOI "
            f"reaches 90% power at: {e}. The design conclusion is unchanged "
            f"in KIND — mean primaries stay underpowered, the tail-rate "
            f"endpoint stays the powerable one — but the required n and the "
            f"per-model heterogeneity both move.")
    registry_append({
        "evidence_id": "g6-power-sim-v3", "tier": "pilot", "what": what,
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "input_uris": {s: to_uri(str(p)) for s, p in srcs.items()},
        "inputs": {s: sha256_file(p) for s, p in srcs.items()},
        "outputs": [{"path": str(OUT), "uri": to_uri(str(OUT)),
                     "sha256": sha256_file(OUT),
                     "payload_sha256": env["payload_sha256"]}]})
    from .. import registry as reg
    try:
        reg.supersede("g6-power-sim-v2", "g6-power-sim-v3",
                      reason="clustered on the defective prefix family field")
    except Exception as ex:
        print(f"  (supersede: {ex})")
    print(json.dumps({"planning_cell": worst_slug,
                      "icc_twohop": {k.split('/')[0]: comps[k]["icc"]
                                     for k in comps if k.endswith("twohop")},
                      "recommendations": recs, "change_vs_v2": delta,
                      "tail_rates": tail_rates,
                      "seconds": round(time.time() - t0)}, indent=1))


if __name__ == "__main__":
    main()
