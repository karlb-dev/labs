# Guards the P0 family-map repair (nextsteps_2_2 §2.1).
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jspace_part2 import family  # noqa: E402
from jspace_part2.family import FamilyMapError, attach_family, audit  # noqa: E402

PROBE_SWAP = Path("/content/jacobian-lens/data/experiments/probe-swap.json")

fails = []


def check(cond, msg):
    print(f"  {'ok ' if cond else 'FAIL'} {msg}")
    if not cond:
        fails.append(msg)


print("[1] map invariants")
a = audit()
check(a["ok"], f"no family shares a template with another "
               f"({a['families_sharing_a_template_hash']})")
check(a["n_families"] >= 30, f"{a['n_families']} canonical families")
check(a["n_items"] == 120, f"{a['n_items']} items mapped (90 probe-swap + 30 one-hop)")

print("[2] every released probe-swap item is mapped")
if PROBE_SWAP.exists():
    names = [i["name"] for i in json.loads(PROBE_SWAP.read_text())["items"]]
    m = family.load_map()
    mapped = {r["item_name"] for r in m["items"] if r["pool"] == "probe_swap"}
    check(set(names) <= mapped, f"{len(set(names) - mapped)} unmapped released items")
else:
    print("  (skip: probe-swap.json not present)")

print("[3] the defect is actually repaired")
# the prefix rule put these two in different families; they are one template
check(family.family_of("twohop:atomic-80-state")
      == family.family_of("twohop:ex-element-state-80-8"),
      "atomic-80-state and ex-element-state-80-8 are the same family")
# the prefix rule put 16 unrelated items in family 'ex'
exfams = {family.family_of(f"twohop:{n}") for n in
          ("ex-city-capital-Lyon-Naples", "ex-element-state-26-8",
           "ex-planet-color-third-fourth")}
check(len(exfams) == 3, "the old 'ex' bucket splits into distinct families")
# the one-hop battery's ten capital items are ONE family
caps = {family.family_of(f"onehop:{i}") for i in range(10)}
check(len(caps) == 1, "the ten one-hop capital items are one family")

print("[4] attach_family validates")
df = pd.DataFrame({"item_id": ["twohop:mars-color", "onehop:0"],
                   "family": ["mars", "onehop0"], "score": [1.0, 2.0]})
out = attach_family(df)
check(list(out["canonical_family"]) == ["entity_color", "country_capital_direct"],
      "canonical families attached")
check("family_legacy" in out.columns, "defective field preserved as family_legacy")
try:
    attach_family(pd.DataFrame({"item_id": ["twohop:not-a-real-item"]}))
    check(False, "unmapped item must raise")
except FamilyMapError:
    check(True, "unmapped item raises FamilyMapError")

print("[5] many-to-one join is enforced (duplicate map rows would raise)")
rows = pd.DataFrame({"item_id": ["onehop:0", "onehop:0", "onehop:1"]})
out = attach_family(rows)
check(len(out) == 3, "repeated item_ids in DATA are fine (many-to-one)")

print("ALL FAMILY MAP TESTS PASS" if not fails else f"{len(fails)} FAILURES")
sys.exit(1 if fails else 0)
