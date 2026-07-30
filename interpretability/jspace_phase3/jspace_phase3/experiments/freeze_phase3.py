# The Phase 3 FREEZE action (prereg candidate §9; addendum §4.3).
#
# Verifies EVERY gate programmatically, then — and only then —
#   1. regenerates family_split_v2 with the declared seed and writes the
#      partition payload with freeze_authorised=true;
#   2. renames the candidate to SCIENTIFIC_PREREGISTRATION_PHASE3.md;
#   3. registers the freeze event (the caller then commits and tags
#      jspace-phase3-freeze-v1 — the tag is applied to the freeze
#      commit itself, so tagging happens outside this process).
#
# A failed gate raises with the exact failure; nothing is written.
# Conditional PI sign-off (2026-07-29) applies ONLY when every gate
# passes — this module is that condition made executable.
#
# Usage: python -m jspace_phase3.experiments.freeze_phase3 --seed <int> \
#            [--p3p3-model <slug>] [--verify-only]
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from ..bank import load_bank, phase2_triple_keys
from ..family_split import SplitConstraints, split_families_v2
from ..paths3 import metrics_dir, resolve_uri, run_root
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           write_result3)

SLUGS = ["olmo31-think", "olmo31-instruct", "qwen36-27b"]
REPO = Path(__file__).resolve().parents[2]
REPO_DATA = REPO / "data"
BANKS = ["bank_f_v6.jsonl", "bank_s_v3.jsonl"]
CAND = REPO / "preregistration/SCIENTIFIC_PREREGISTRATION_PHASE3_CANDIDATE.md"
FROZEN = REPO / "preregistration/SCIENTIFIC_PREREGISTRATION_PHASE3.md"


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def gate(cond: bool, msg: str, failures: list):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def main():
    require_clean_tree(False)          # the freeze NEVER runs dirty
    seed = int(arg("--seed"))
    p3p3 = arg("--p3p3-model")
    verify_only = "--verify-only" in sys.argv
    failures: list[str] = []

    # ---- gate: conformance tests green at this commit
    r = subprocess.run([sys.executable, "-m", "pytest",
                        str(REPO / "tests"), "-q", "--no-header"],
                       capture_output=True, text=True)
    gate(r.returncode == 0, f"conformance tests ({r.stdout.strip()[-60:]})",
         failures)

    # ---- gate: registry events exist for every prerequisite
    reg = (REPO / "reports/evidence_events.jsonl").read_text()
    events = [json.loads(l) for l in reg.splitlines() if l.strip()]
    live = {e["evidence_id"] for e in events if e.get("event") != "supersede"}
    superseded = {e.get("superseded") for e in events
                  if e.get("event") == "supersede"}
    live -= superseded

    def has(eid_prefix, what):
        ok = any(e.startswith(eid_prefix) for e in live)
        gate(ok, f"registry: {what} ({eid_prefix}*)", failures)

    for slug in SLUGS:
        has(f"p3-prose-grid-{slug}", f"Workstream C grid {slug}")
        has(f"p3-g5-bank-{slug}", f"G5 bank scoring {slug}")
    has("p3-prose-grid-figure", "Workstream C figure/stats")
    has("p3-power-sim", "power simulation")
    has("p3-bank-f-v6", "Bank F v6")
    has("p3-bank-s-v3", "Bank S v3")
    if p3p3:
        has(f"p3-bridge-dev-gate-{p3p3}", f"P3-P3 gate on {p3p3}")
        gp = metrics_dir(p3p3) / "bridge_dev_gate" / \
            f"bridge_dev_gate_{p3p3}.json"
        g = json.loads(gp.read_text())["payload"]["gate"]
        gate(bool(g["identifiable"]),
             f"P3-P3 identifiable on {p3p3}", failures)

    # ---- gate: no unresolved PENDING in the candidate
    cand = CAND.read_text()
    gate("[PENDING" not in cand, "prereg candidate has no [PENDING] slots",
         failures)

    # ---- gate: split feasible with prereg floors + §5.8 checks
    from .run_family_split import build_family_table
    tab, items = build_family_table()
    floors = json.loads(arg("--floors", '{"twohop": 18, "intersection": 12}'))
    cons = SplitConstraints(
        min_twohop_families_per_side=floors["twohop"],
        min_intersection_families_per_side=floors["intersection"],
        max_standardized_imbalance=0.35, seed=seed)
    part = split_families_v2(tab, cons)
    part.assert_disjoint(items, "canonical_family", "fact_id",
                         "template_hash")
    gate(True, f"family_split_v2 feasible (seed {seed}): "
               f"{part.balance_report['n_confirmatory']}/"
               f"{part.balance_report['n_replication']}", failures)
    p2 = phase2_triple_keys(json.load(open(resolve_uri(
        "drive://metrics/cross_model/g5_item_manifest_v5.json")))["payload"])
    triples = {f"{b.bridge}|{b.answer}" for bank in BANKS
               for b in load_bank(REPO_DATA / bank)}
    gate(len(triples) == sum(len(load_bank(REPO_DATA / b)) for b in BANKS),
         "one (bridge,answer) pair per bundle bank-wide", failures)

    if failures:
        raise RuntimeError(f"{len(failures)} freeze gate(s) failed: "
                           + "; ".join(failures[:5]))
    if verify_only:
        print("ALL GATES PASS (verify-only; nothing written)")
        return

    # ---- execute the freeze
    out_dir = run_root() / "preregistration"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "freeze_authorised": True, "seed": seed,
        "confirmatory": sorted(part.confirmatory),
        "replication": sorted(part.replication),
        "balance_report": part.balance_report,
        "banks": {b: None for b in BANKS},
        "p3p3_model": p3p3,
    }
    from jspace_part2.lib import sha256_file
    payload["banks"] = {b: sha256_file(REPO_DATA / b) for b in BANKS}
    part_path = out_dir / "partition_phase3.json"
    eid = "p3-partition-freeze-v1"
    cmd = (f"python -m jspace_phase3.experiments.freeze_phase3 "
           f"--seed {seed} --p3p3-model {p3p3}")
    write_result3(payload, part_path, Provenance3(
        evidence_id=eid, tier="phase3-confirmatory", command=cmd,
        seed=seed))
    repo_part = REPO / "preregistration/partition_phase3.json"
    repo_part.write_text(part_path.read_text())
    CAND.rename(FROZEN)
    register(eid, tier="phase3-confirmatory", command=cmd,
             what=(f"PHASE 3 FREEZE: partition seed {seed} "
                   f"({part.balance_report['n_confirmatory']}/"
                   f"{part.balance_report['n_replication']} families, "
                   f"intersection {part.balance_report['intersection']}), "
                   f"prereg frozen, P3-P3={p3p3}; conditional PI sign-off "
                   f"2026-07-29 discharged — all gates green"),
             outputs=[part_path, repo_part, FROZEN])
    print("FROZEN. Now: git add -A && git commit (freeze commit) && "
          "git tag jspace-phase3-freeze-v1 && git push --tags")


if __name__ == "__main__":
    main()
