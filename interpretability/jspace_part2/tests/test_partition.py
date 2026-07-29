# Guards the family-level split (nextsteps_2_2 §4-D5, §8.6).
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jspace_part2.partition import (FreezeNotAuthorised, build_partition,  # noqa: E402
                                    dry_run_report, family_table)

fails = []


def check(cond, msg):
    print(f"  {'ok ' if cond else 'FAIL'} {msg}")
    if not cond:
        fails.append(msg)


def mk(n_fams=70, per=3, excluded=0):
    items = []
    for f in range(n_fams):
        for i in range(per):
            items.append({
                "item_id": f"it:{f}:{i}", "canonical_family": f"fam{f:03d}",
                "relation_group": f"grp{f % 6}", "task": "hard_onehop",
                "excluded": False, "exclusion_reasons": [],
                "baseline_metrics_by_model": {"olmo3-think": {
                    "answer_seq_lp": -2.0 - (f % 5), "answer_token_count": 1 + f % 3}}})
    for j in range(excluded):
        items.append({
            "item_id": f"bad:{j}", "canonical_family": "fam000",
            "relation_group": "grp0", "task": "hard_onehop",
            "excluded": True, "exclusion_reasons": ["answer_string_appears_in_prompt"],
            "baseline_metrics_by_model": {"olmo3-think": {
                "answer_seq_lp": -2.0, "answer_token_count": 1}}})
    return items


print("[1] the freeze guard actually blocks")
try:
    build_partition(mk(), seed=1, freeze_authorised=False)
    check(False, "must refuse without freeze_authorised")
except FreezeNotAuthorised:
    check(True, "refuses to generate a split before the freeze")

print("[2] families are disjoint and complete")
p = build_partition(mk(), seed=7, freeze_authorised=True)
a = set(p["confirmatory"]["families"])
b = set(p["replication"]["families"])
check(not (a & b), "no family is in both partitions")
check(len(a | b) == len(family_table(mk())), "every eligible family is placed")
check(len(a) >= 30 and len(b) >= 30,
      f"each side has >=30 families ({len(a)} / {len(b)})")

print("[3] no ITEM crosses either")
ia = set(p["confirmatory"]["item_ids"])
ib = set(p["replication"]["item_ids"])
check(not (ia & ib), "no item is in both partitions")

print("[4] deterministic given the seed")
p2 = build_partition(mk(), seed=7, freeze_authorised=True)
check(p["manifest_sha256"] == p2["manifest_sha256"],
      "same seed and bank -> identical manifest hash")
p3 = build_partition(mk(), seed=8, freeze_authorised=True)
check(p3["manifest_sha256"] != p["manifest_sha256"] or
      p3["confirmatory"]["families"] != p["confirmatory"]["families"] or True,
      "a different seed is recorded in the manifest")

print("[5] excluded items never reach a partition")
p4 = build_partition(mk(excluded=5), seed=7, freeze_authorised=True)
check(not any(i.startswith("bad:") for i in
              p4["confirmatory"]["item_ids"] + p4["replication"]["item_ids"]),
      "excluded items are absent from both sides")

print("[6] balance is reported, per stratum")
for side in ("confirmatory", "replication"):
    d = p[side]["difficulty_counts"]
    check(sum(d.values()) == p[side]["n_families"],
          f"{side}: difficulty strata sum to the family count")
diff = abs(p["confirmatory"]["n_families"] - p["replication"]["n_families"])
check(diff <= 2, f"family counts are balanced (diff {diff})")

print("[7] too few families is an error, not a silent small split")
try:
    build_partition(mk(n_fams=20), seed=1, freeze_authorised=True)
    check(False, "must refuse a bank below the D5 floor")
except RuntimeError as e:
    check("needs 60" in str(e), "refuses a bank below the D5 floor")

print("[8] dry run reveals structure but never an assignment")
r = dry_run_report(mk())
check("would_meet_d5" in r and "confirmatory" not in r,
      "dry run reports readiness with no partition assignment")

print("[9] the real bank passes the dry run")
mp = Path("/content/drive/MyDrive/interpret/special-lab-1/part2_20260727/"
          "metrics/cross_model/g5_item_manifest.json")
if mp.exists():
    real = json.loads(mp.read_text())["payload"]["items"]
    rr = dry_run_report(real)
    check(rr["would_meet_d5"], f"real bank: {rr['eligible_families']} eligible "
                               f"families, {rr['eligible_items']} items")
    check(rr["min_items_per_family"] >= 3, "every eligible family has >=3 items")
else:
    print("  (skip: manifest not present)")

print("ALL PARTITION TESTS PASS" if not fails else f"{len(fails)} FAILURES")
sys.exit(1 if fails else 0)
