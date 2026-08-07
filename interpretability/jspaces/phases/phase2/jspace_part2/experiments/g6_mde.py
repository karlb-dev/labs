# G6 companion 2 — MINIMUM DETECTABLE EFFECT under audited families.
#
# WHY. Under the corrected family map the tail-rate endpoint does NOT
# reach 90% power on the OLMo cells at any family count the item bank can
# realistically reach (Think 0.58, Instruct 0.43 at 150 families for a
# 10pp SESOI). "How much n do we need for the effect we hope for" has
# therefore stopped being the useful question. The useful question is the
# inverse: GIVEN a bank of m families, what is the smallest effect the
# design can detect at 90% power, and how does that compare with the
# effect the pilot actually saw?
#
# A design whose MDE is far above its plausible effect should not run a
# binary primary at all — it should state an estimate with an interval and
# say so in the preregistration. That is a design decision, and this is
# the number it should be made from.
#
# Reports, per model and per family count:
#   mde_90  : smallest true rate difference detectable at 90% power
#   mde_80  : the same at 80%
#   pilot   : the observed pilot gap, for comparison
#   ratio   : mde_90 / pilot gap  (>1 means the design cannot see it)
# Also the same for the CONTINUOUS mean endpoint in nats.
#
# Tier: pilot (design input). CPU, a few minutes.
# Usage: python -m jspace_part2.experiments.g6_mde [--allow-dirty]
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

OUT = BASE / "cross_model" / "g6_mde.json"
FAMS = [30, 40, 60, 80, 100, 150]
K = 2
GRID_RATE = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
GRID_NATS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def mde(power_fn, grid, target):
    """Smallest grid effect whose simulated power >= target."""
    for e in grid:
        if power_fn(e) >= target:
            return e
    return None


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    t0 = time.time()
    raw, srcs = {}, {}
    for slug in v2.CELLS:
        raw[slug], srcs[slug] = load_corrected(slug)

    out = {}
    for slug, df in raw.items():
        pairs = v2.tail_pairs(df)
        gap = float(pairs.hit_j.mean() - pairs.hit_r.mean())
        d_mean = v2.paired_deltas(df, "twohop")
        comp = v2.components(d_mean)
        rows = []
        for m in FAMS:
            r90 = mde(lambda e: v2.power_tailrate_boot(pairs, m, K, e),
                      GRID_RATE, 0.90)
            r80 = mde(lambda e: v2.power_tailrate_boot(pairs, m, K, e),
                      GRID_RATE, 0.80)
            n90 = mde(lambda e: v2.power_boot(d_mean, m, K, e),
                      GRID_NATS, 0.90)
            n80 = mde(lambda e: v2.power_boot(d_mean, m, K, e),
                      GRID_NATS, 0.80)
            rows.append({
                "m_families": m, "n_items": m * K,
                "mde_rate_90": r90, "mde_rate_80": r80,
                "mde_nats_90": n90, "mde_nats_80": n80,
                "pilot_rate_gap": round(gap, 3),
                "pilot_mean_delta": round(comp["mu"], 3),
                "rate_ratio_mde90_over_pilot": (round(r90 / gap, 2)
                                                if r90 and gap > 0 else None),
                "nats_ratio_mde90_over_pilot": (round(n90 / abs(comp["mu"]), 2)
                                                if n90 and comp["mu"] else None)})
        out[slug] = {"pilot_rate_gap": round(gap, 3),
                     "pilot_mean_delta": round(comp["mu"], 3),
                     "icc": comp["icc"], "grid": rows}

    # the headline the preregistration needs
    def at(slug, m, key):
        return next(r[key] for r in out[slug]["grid"] if r["m_families"] == m)

    verdict = {
        "at_60_families": {s: {"mde_rate_90": at(s, 60, "mde_rate_90"),
                               "pilot_gap": out[s]["pilot_rate_gap"],
                               "mde_nats_90": at(s, 60, "mde_nats_90"),
                               "pilot_mean": out[s]["pilot_mean_delta"]}
                           for s in out},
        "reading": (
            "A binary primary is only honest where MDE at the planned m is "
            "at or below the effect the pilot suggests. Where the ratio "
            "exceeds 1 the design cannot see its own hypothesised effect, "
            "and the hypothesis should be stated as estimation-with-interval "
            "rather than as a test."),
    }
    payload = {"k_per_family": K, "power_targets": [0.80, 0.90],
               "rate_grid": GRID_RATE, "nats_grid": GRID_NATS,
               "per_model": out, "verdict": verdict}
    prov = Provenance(
        evidence_id="g6-mde-v1", tier="pilot",
        command="python -m jspace_part2.experiments.g6_mde",
        inputs={s: sha256_file(p) for s, p in srcs.items()}, seed=4242)
    env = write_result_v2(payload, OUT, prov)
    registry_append({
        "evidence_id": "g6-mde-v1", "tier": "pilot",
        "what": ("Minimum detectable effect under audited families, per "
                 "model and family count, for BOTH endpoints. At 60 "
                 "canonical families (the D5 target): " +
                 "; ".join(f"{s} MDE_rate90={v['mde_rate_90']} vs pilot gap "
                           f"{v['pilot_gap']}, MDE_nats90={v['mde_nats_90']} "
                           f"vs pilot mean {v['pilot_mean']}"
                           for s, v in verdict["at_60_families"].items()) +
                 ". Where MDE exceeds the pilot effect the design cannot "
                 "test its own hypothesis and must state an estimate."),
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "input_uris": {s: to_uri(str(p)) for s, p in srcs.items()},
        "inputs": {s: sha256_file(p) for s, p in srcs.items()},
        "outputs": [{"path": str(OUT), "uri": to_uri(str(OUT)),
                     "sha256": sha256_file(OUT),
                     "payload_sha256": env["payload_sha256"]}]})
    print(json.dumps(verdict, indent=1))
    print(f"\n{'model':17s} {'fams':>5s} {'MDE rate@90':>12s} {'pilot gap':>10s} "
          f"{'MDE nats@90':>12s} {'pilot mean':>11s}")
    for s, v in out.items():
        for r in v["grid"]:
            print(f"{s:17s} {r['m_families']:5d} {str(r['mde_rate_90']):>12s} "
                  f"{r['pilot_rate_gap']:10.3f} {str(r['mde_nats_90']):>12s} "
                  f"{r['pilot_mean_delta']:11.3f}")
    print(f"\nseconds {round(time.time() - t0)}")


if __name__ == "__main__":
    main()
