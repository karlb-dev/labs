# Unit tests for the capture (selected/blocked id audit) path of the R1
# protected dynamic ablator (CPU, stub layers). The capture list must
# record exactly what the ablator did, and its presence must not change
# the ablation itself. Run: python tests/test_capture.py
import sys

import numpy as np
import torch

from jspace_part2.protected_dynamic import ProtectedDynamicAblator


def check(name, cond):
    if not cond:
        print(f"FAIL: {name}")
        sys.exit(1)
    print(f"  ok  {name}")


torch.manual_seed(0)
d, V, T = 64, 500, 4
layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
D = torch.nn.functional.normalize(torch.randn(V, d), dim=1).half()
hh = torch.stack([(D[i].float() * 10).reshape(d)
                  for i in (11, 12, 13, 14)]).reshape(1, T, d)
psets = torch.tensor([[11], [12], [13], [14]])   # protect own token per pos

ab = ProtectedDynamicAblator(layers, band=[1])

print("[1] capture records selected ids per position")
cap = []
with ab:
    ab.mode = {"dicts": {1: D}, "k": 3, "nonneg": True,
               "protect_sets": psets, "capture": cap}
    out_cap = layers[1](hh.clone())
    ab.mode = {"dicts": {1: D}, "k": 3, "nonneg": True,
               "protect_sets": psets}
    out_plain = layers[1](hh.clone())
    ab.mode = None
kinds = sorted(r["kind"] for r in cap)
check("one protect + one selected record", kinds == ["protect", "selected"])
sel = next(r for r in cap if r["kind"] == "selected")
prot = next(r for r in cap if r["kind"] == "protect")
check("selected ids shaped [T, take]",
      sel["ids"].shape[0] == T and sel["ids"].shape == sel["scores"].shape)
check("capture does not change the ablation",
      torch.allclose(out_cap, out_plain))

print("[2] protected ids never appear among selected ids at their position")
own = psets.numpy()
for t in range(T):
    check(f"pos {t}: protected id {int(own[t, 0])} not selected",
          int(own[t, 0]) not in sel["ids"][t].tolist())

print("[3] protect record marks genuinely-blocked ids")
check("protect ids match the protect sets",
      np.array_equal(prot["ids"], own.astype(np.int32)))
# each position's own dominant token has positive score -> truly blocked
check("dominant own-token selections flagged blocked",
      bool(prot["blocked"].all()))

print("[4] selected scores are finite and positive under nonneg")
check("scores finite", np.isfinite(sel["scores"]).all())
check("scores positive", (sel["scores"] > 0).all())

print("[5] no capture key -> no records, identical behavior")
with ab:
    ab.mode = {"dicts": {1: D}, "k": 3, "nonneg": True,
               "protect_sets": psets}
    out_nocap = layers[1](hh.clone())
    ab.mode = None
check("plain mode unchanged", torch.allclose(out_nocap, out_plain))

print("ALL CAPTURE TESTS PASS")
