# family_split_v2 runner (nextsteps §5.6): builds the per-family balance
# table from the banks + all three G5 parquets, runs the seed-ACTIVE
# splitter, and registers assignment + balance report.
#
# NOT the freeze itself: the freeze commit re-runs this with
# --freeze-authorised inside freeze_partition-style verification, after
# every §9 gate closes. Until then the output is a development preview
# (the prereg candidate's PENDING cohort numbers come from it).
#
# Usage:
#   python -m jspace_phase3.experiments.run_family_split --seed <int> \
#       [--eid p3-family-split-preview-v1]
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from ..bank import load_bank
from ..family_split import SplitConstraints, split_families_v2
from ..paths3 import metrics_dir
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           write_result3)

TIER = "phase3-development"
SLUGS = ["olmo31-think", "olmo31-instruct", "qwen36-27b"]
REPO_DATA = Path(__file__).resolve().parents[2] / "data"
BANKS = ["bank_f_v6.jsonl", "bank_s_v3.jsonl"]


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def build_family_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    bundles = [b for bk in BANKS for b in load_bank(REPO_DATA / bk)]
    items = pd.DataFrame([it for b in bundles for it in b.as_items()])
    g5 = {}
    for slug in SLUGS:
        g5[slug] = pd.read_parquet(
            metrics_dir(slug) / "g5_bank" / f"g5_bank_{slug}.parquet")

    # per-fact capability on direct AND composed, per model
    def fact_cap(slug):
        g = g5[slug]
        ok = {}
        for fid, sub in g[g.variant.isin(["direct", "composed"])] \
                .groupby("fact_id"):
            ok[fid] = bool(sub.capable_generation.all()
                           and len(sub) == 2)
        return ok

    caps = {s: fact_cap(s) for s in SLUGS}
    rows = []
    for fam, sub in items.groupby("canonical_family"):
        fids = sorted(set(sub.fact_id))
        inter = [f for f in fids if all(caps[s].get(f, False)
                                        for s in SLUGS)]
        row = {
            "canonical_family": fam,
            "bank": sub.bank.iloc[0],
            "relation_group": sub.relation_group.iloc[0],
            "n_items": int(len(sub)),
            "n_counterfactual": int(sub.counterfactual_bridge
                                    .notna().sum()),
            "intersection_capable": len(inter) >= 1,
            "n_intersection_bundles": len(inter),
            "answer_len_chars": float(sub.canonical_answer.str.len()
                                      .mean()),
            "bridge_len_chars": float(sub.bridge_entity.fillna("")
                                      .str.len().mean()),
        }
        for v in ("direct", "composed", "bridge_supplied"):
            row[f"n_variant_{v}"] = int((sub.variant == v).sum())
        for slug in SLUGS:
            g = g5[slug]
            gl = g[g.fact_id.isin(fids)]
            row[f"median_lp_{slug.replace('-', '_')}"] = float(
                gl.lp_canonical.median()) if len(gl) else 0.0
        rows.append(row)
    return pd.DataFrame(rows), items


def main():
    require_clean_tree("--allow-dirty" in sys.argv)
    seed = int(arg("--seed"))
    eid = arg("--eid", "p3-family-split-preview-v1")
    tab, items = build_family_table()
    cons = SplitConstraints(
        min_twohop_families_per_side=18,   # candidate floors pending the
        min_intersection_families_per_side=15,  # power sim; freeze uses
        max_standardized_imbalance=0.35,        # the prereg values
        seed=seed)
    part = split_families_v2(tab, cons)
    part.assert_disjoint(items, "canonical_family", "fact_id",
                         "template_hash")

    out_dir = metrics_dir("cross_model")
    out = out_dir / "family_split_v2_preview.json"
    payload = {"confirmatory": sorted(part.confirmatory),
               "replication": sorted(part.replication),
               "balance_report": part.balance_report, "seed": seed,
               "family_table_rows": int(len(tab))}
    cmd = f"python -m jspace_phase3.experiments.run_family_split --seed {seed}"
    write_result3(payload, out, Provenance3(
        evidence_id=eid, tier=TIER, command=cmd, seed=seed))
    register(eid, tier=TIER, command=cmd,
             what=(f"family_split_v2 preview (seed {seed}): "
                   f"{part.balance_report['n_confirmatory']}/"
                   f"{part.balance_report['n_replication']} families, "
                   f"intersection {part.balance_report['intersection']}, "
                   f"worst imbalance "
                   f"{max(part.balance_report['per_dimension'].values())}"),
             outputs=[out])
    print(json.dumps(part.balance_report, indent=1))


if __name__ == "__main__":
    main()
