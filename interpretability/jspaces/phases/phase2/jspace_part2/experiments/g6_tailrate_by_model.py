# G6 companion — tail-rate power PER MODEL at feasible family counts.
#
# WHY THIS EXISTS. g6-power-sim-v3 plans against the single most
# conservative cell (the model with the SMALLEST pilot J-vs-random tail
# gap, which is the base checkpoint). That is the right choice for a
# blanket design claim, but it is not the number the preregistration
# actually needs: HP3's primary is the OLMo-3.1-Think confirmatory
# partition, and Qwen is an external validation contrast. Planning the
# whole campaign on the weakest cell would demand ~150 canonical families
# when the cell that carries the hypothesis may need far fewer.
#
# This reports, for every model and for family counts the D5 expansion can
# actually reach, the power to detect (a) a 10pp rate difference and (b)
# that model's own observed pilot gap. It also reports the family count
# each model needs for 90% power, so the item-bank target is chosen with
# the numbers in view rather than by habit.
#
# Tier: pilot (design input). CPU.
# Usage: python -m jspace_part2.experiments.g6_tailrate_by_model [--allow-dirty]
from __future__ import annotations

import json
import sys
import time

import numpy as np

from ..lib import sha256_file
from ..paths import to_uri
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          write_result_v2)
from . import g6_power_sim as v2
from .g6_power_sim_v3 import BASE, load_corrected

OUT = BASE / "cross_model" / "g6_tailrate_power_by_model.json"
FAMS = [30, 40, 50, 60, 70, 80, 100, 120, 150]
K = 2
SESOI_PP = 0.10
TARGET = 0.90


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    t0 = time.time()
    raw, srcs = {}, {}
    for slug in v2.CELLS:
        raw[slug], srcs[slug] = load_corrected(slug)

    per_model, need = {}, {}
    for slug, df in raw.items():
        pairs = v2.tail_pairs(df)
        gap = float(pairs.hit_j.mean() - pairs.hit_r.mean())
        rows = []
        for m in FAMS:
            rows.append({
                "m_families": m, "n_items": m * K,
                "power_at_10pp": round(v2.power_tailrate_boot(
                    pairs, m, K, SESOI_PP), 3),
                "power_at_own_gap": round(v2.power_tailrate_boot(
                    pairs, m, K, gap), 3)})
        per_model[slug] = {
            "pilot_rate_j": round(float(pairs.hit_j.mean()), 3),
            "pilot_rate_r": round(float(pairs.hit_r.mean()), 3),
            "pilot_gap": round(gap, 3),
            "pilot_n_families": int(pairs.family.nunique()),
            "grid": rows}
        for key in ("power_at_10pp", "power_at_own_gap"):
            hit = [r for r in rows if r[key] >= TARGET]
            need[f"{slug}/{key}"] = (min(hit, key=lambda r: r["m_families"])
                                     ["m_families"] if hit else None)

    primary = "olmo3-think"          # pilot stand-in for the 3.1-Think primary
    external = "qwen36-27b"
    payload = {
        "note": ("Planning numbers per model. The pilot Think cell stands in "
                 "for the 3.1-Think primary, whose own lens does not exist "
                 "yet; treat it as indicative, not as the primary's power."),
        "k_per_family": K, "sesoi_pp": SESOI_PP, "power_target": TARGET,
        "per_model": per_model,
        "families_needed_for_90pct": need,
        "reading": {
            "primary_cell": primary,
            "families_for_90pct_at_10pp": need[f"{primary}/power_at_10pp"],
            "families_for_90pct_at_own_gap": need[f"{primary}/power_at_own_gap"],
            "external_validation_cell": external,
            "external_families_for_90pct_at_10pp": need[f"{external}/power_at_10pp"],
        },
    }
    prov = Provenance(
        evidence_id="g6-tailrate-power-by-model-v1", tier="pilot",
        command="python -m jspace_part2.experiments.g6_tailrate_by_model",
        inputs={s: sha256_file(p) for s, p in srcs.items()}, seed=4242)
    env = write_result_v2(payload, OUT, prov)
    registry_append({
        "evidence_id": "g6-tailrate-power-by-model-v1", "tier": "pilot",
        "what": (f"Tail-rate power PER MODEL under audited families "
                 f"(companion to g6-power-sim-v3, which plans on the most "
                 f"conservative cell). Families needed for 90% power at a "
                 f"10pp SESOI: " +
                 ", ".join(f"{k.split('/')[0]} {v}"
                           for k, v in need.items()
                           if k.endswith("power_at_10pp")) +
                 f". At each model's OWN observed pilot gap: " +
                 ", ".join(f"{k.split('/')[0]} {v}"
                           for k, v in need.items()
                           if k.endswith("own_gap")) + "."),
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "input_uris": {s: to_uri(str(p)) for s, p in srcs.items()},
        "inputs": {s: sha256_file(p) for s, p in srcs.items()},
        "outputs": [{"path": str(OUT), "uri": to_uri(str(OUT)),
                     "sha256": sha256_file(OUT),
                     "payload_sha256": env["payload_sha256"]}]})
    print(json.dumps({"families_needed_for_90pct": need,
                      "per_model_gaps": {s: v["pilot_gap"]
                                         for s, v in per_model.items()},
                      "seconds": round(time.time() - t0)}, indent=1))
    for slug, v in per_model.items():
        print(f"\n{slug}  (pilot gap {v['pilot_gap']:+.3f})")
        for r in v["grid"]:
            print(f"   {r['m_families']:3d} fams (n={r['n_items']:3d}): "
                  f"power@10pp {r['power_at_10pp']:.3f}   "
                  f"power@own-gap {r['power_at_own_gap']:.3f}")


if __name__ == "__main__":
    main()
