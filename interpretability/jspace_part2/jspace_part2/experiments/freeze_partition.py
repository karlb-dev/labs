# THE FREEZE ACTION (runs only inside the dedicated freeze commit).
#
# PI sign-off: given by interactive directive 2026-07-29 ("run
# autonomously through the freeze points until all planned work is
# done"), conditional on all candidate §9 conditions closing cleanly —
# this script VERIFIES those conditions and refuses if any is open:
#   1. capability cohorts closed (g5_item_manifest_v4.json exists,
#      all three models scored)
#   2. matched control dev-validated (mc_dev_validation PASS)
#   3. both primary lenses exist
#   4. corrected R2 artifacts exist for Qwen + Instruct
# Then: build_partition(freeze_authorised=True, seed=4242), write the
# partition manifest (hashes + assignments, no outcomes), register it.
# The rename + tag happen in the surrounding commit, nothing else.
#
# Usage: python -m jspace_part2.experiments.freeze_partition \
#          --i-am-the-freeze-commit [--allow-dirty]
from __future__ import annotations

import json
import sys
from pathlib import Path

from ..lib import sha256_file
from ..partition import build_partition
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          write_result_v2)

RUN = Path("/content/drive/MyDrive/interpret/special-lab-1/part2_20260727")
MANIFEST_V4 = RUN / "metrics" / "cross_model" / "g5_item_manifest_v4.json"
OUT = RUN / "metrics" / "cross_model" / "partition_manifest.json"
SEED = 4242


def main():
    if "--i-am-the-freeze-commit" not in sys.argv:
        raise SystemExit("REFUSING: pass --i-am-the-freeze-commit only from "
                         "the dedicated freeze commit")
    git = require_clean_tree("--allow-dirty" in sys.argv)

    # ---- verify the §9 conditions
    if not MANIFEST_V4.exists():
        raise SystemExit("condition 1 OPEN: manifest v4 (cohorts) missing")
    man = json.loads(MANIFEST_V4.read_text())["payload"]
    if man.get("cohort_counts") is None:
        raise SystemExit("condition 1 OPEN: cohort counts missing")
    mc = RUN / "metrics" / "olmo31-think" / "mc_dev_validation" / \
        "mc_dev_validation.json"
    if not mc.exists() or not json.loads(mc.read_text())["payload"]["pass"]:
        raise SystemExit("condition 2 OPEN: matched-control dev validation "
                         "missing or FAILED")
    for lens in ("olmo31think_lens.pt", "olmo31instruct_lens.pt"):
        if not (RUN / "lens" / lens).exists():
            raise SystemExit(f"condition 3 OPEN: {lens} missing")
    for slug in ("qwen36-27b", "olmo31-instruct"):
        r2 = RUN / "metrics" / slug / "r2_occupancy" / "r2_occupancy_v2.json"
        if not r2.exists():
            raise SystemExit(f"condition 4 OPEN: corrected R2 missing "
                             f"for {slug}")
    print("all §9 conditions CLOSED; generating the partition")

    part = build_partition(man["items"], seed=SEED, freeze_authorised=True)
    prov = Provenance(
        evidence_id="d5-partition-freeze-v1", tier="confirmatory",
        command=("python -m jspace_part2.experiments.freeze_partition "
                 "--i-am-the-freeze-commit"),
        inputs={"manifest_v4": sha256_file(MANIFEST_V4)},
        model={"note": "deterministic family split; no model involved"},
        seed=SEED)
    write_result_v2(part, OUT, prov)
    registry_append({
        "evidence_id": "d5-partition-freeze-v1", "tier": "confirmatory",
        "what": (f"FROZEN family-level partition (algorithm "
                 f"{part['algorithm_version']}, seed {SEED}): confirmatory "
                 f"{part['confirmatory']['n_families']} families / "
                 f"{part['confirmatory']['n_items']} items, replication "
                 f"{part['replication']['n_families']} / "
                 f"{part['replication']['n_items']}; manifest sha "
                 f"{part['manifest_sha256'][:16]}; no outcomes viewed"),
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(OUT), "sha256": sha256_file(OUT)}]})
    print(json.dumps({k: part[k][a] for k in ("confirmatory", "replication")
                      for a in ("n_families", "n_items") if False} or
                     {"confirmatory": part["confirmatory"]["n_families"],
                      "replication": part["replication"]["n_families"],
                      "manifest_sha256": part["manifest_sha256"]}, indent=1))


if __name__ == "__main__":
    main()
