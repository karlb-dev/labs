# C3 — combined Stage-3 bank status against the preregistration floor.
#
# Two candidate pools have been scored on the anchor model (v2: 212 items
# / 45 families; v3: 132 / 30, authored in the shapes v2 showed stay
# hard). This reports whether the COMBINED bank clears the prereg floor
# (n >= 90 hard items across >= 30 independent families).
#
# The one subtlety that decides the answer: v3 extended several v2
# templates with fresh items (`language_family_2` is more of the same
# relation as `language_family`). Counting those as separate clusters
# would inflate the family count with pseudo-replication — the exact
# defect the addendum found in the SQL battery. So families are collapsed
# to their CANONICAL name (jspace_part2.c3_pool.CANONICAL_FAMILY) before
# counting, and the report shows the count both ways so the correction is
# visible rather than silent.
#
# Tier: dev (bank construction). CPU-only, reads the scored jsonl files.
# Usage: python -m jspace_part2.experiments.c3_bank_status [--allow-dirty]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from ..c3_pool import canonical_family
from ..lib import sha256_file
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          write_result)

RUN_DIR_P2 = Path("/content/drive/MyDrive/interpret/special-lab-1/"
                  "part2_20260727")
POOLS = {v: RUN_DIR_P2 / "config" / "prompts" / f"c3_pool_{v}_scored.jsonl"
         for v in ("v2", "v3")}
DEV_SET = RUN_DIR_P2 / "config" / "prompts" / "hard_onehop_dev.jsonl"
OUT = RUN_DIR_P2 / "metrics" / "olmo3-think" / "c3_bank_status.json"
FLOOR_N, FLOOR_FAMS, MIN_PER_FAM = 90, 30, 2


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    t0 = time.time()
    rows, per_pool = [], {}
    for v, p in POOLS.items():
        if not p.exists():
            continue
        r = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        for x in r:
            x["pool"] = v
        rows += r
        per_pool[v] = {"n": len(r),
                       "n_in_window": sum(1 for x in r if x["in_window"])}

    keep = [x for x in rows if x["in_window"]]
    raw_fams, can_fams = {}, {}
    for x in keep:
        raw_fams[x["family"]] = raw_fams.get(x["family"], 0) + 1
        c = canonical_family(x["family"])
        can_fams[c] = can_fams.get(c, 0) + 1

    raw_ok = [f for f, n in raw_fams.items() if n >= MIN_PER_FAM]
    can_ok = [f for f, n in can_fams.items() if n >= MIN_PER_FAM]
    collapsed = sorted(set(raw_fams) - set(can_fams))

    meets = len(keep) >= FLOOR_N and len(can_ok) >= FLOOR_FAMS
    shortfall = {"items": max(0, FLOOR_N - len(keep)),
                 "families": max(0, FLOOR_FAMS - len(can_ok))}

    summ = {
        "floor": {"n_items": FLOOR_N, "n_families": FLOOR_FAMS,
                  "min_items_per_family": MIN_PER_FAM},
        "per_pool": per_pool,
        "combined_scored": len(rows),
        "combined_in_window": len(keep),
        "families_raw_labels": len(raw_ok),
        "families_canonical": len(can_ok),
        "labels_collapsed_as_duplicates": collapsed,
        "meets_floor": bool(meets),
        "shortfall": shortfall,
        "dev_set_note": (
            "the frozen 41-item hard_onehop_dev.jsonl stays dev-tier "
            "FOREVER (prereg) and is NOT counted toward this floor"),
        "not_partitioned": True,
        "reading": (
            f"Combined bank: {len(keep)} hard items across "
            f"{len(can_ok)} canonical families "
            f"({len(raw_ok)} raw labels; {len(collapsed)} collapsed as "
            f"same-template duplicates). Floor "
            f"{'MET' if meets else 'NOT met'}"
            + ("" if meets else
               f" — short {shortfall['items']} items and "
               f"{shortfall['families']} families. A v4 increment is "
               f"needed; do NOT dilute by re-labelling extensions of "
               f"existing templates as new families.")),
        "seconds": round(time.time() - t0)}
    prov = Provenance(
        evidence_id="c3-bank-status-v1", tier="dev",
        command="python -m jspace_part2.experiments.c3_bank_status",
        inputs={v: sha256_file(p) for v, p in POOLS.items() if p.exists()})
    write_result(summ, OUT, prov)
    registry_append({
        "evidence_id": "c3-bank-status-v1", "tier": "dev",
        "what": (f"Combined Stage-3 bank status: {len(keep)} hard items / "
                 f"{len(can_ok)} canonical families (raw labels "
                 f"{len(raw_ok)}, {len(collapsed)} collapsed as duplicate "
                 f"templates). Prereg floor (n>={FLOOR_N}, fams>="
                 f"{FLOOR_FAMS}): {'MET' if meets else 'NOT MET'}"
                 + ("" if meets else f", short {shortfall}") +
                 ". Not partitioned — freeze action."),
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(OUT), "sha256": sha256_file(OUT)}]})
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
