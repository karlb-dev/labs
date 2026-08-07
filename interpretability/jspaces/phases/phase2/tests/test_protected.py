# Unit tests for the R1 protected dynamic ablator (CPU, stub layers).
# Run: python tests/test_protected.py
import sys

import torch

from jspace_part2.protected_dynamic import ProtectedDynamicAblator


def check(name, cond):
    if not cond:
        print(f"FAIL: {name}")
        sys.exit(1)
    print(f"  ok  {name}")


torch.manual_seed(0)
d, V = 64, 500
layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
D = torch.nn.functional.normalize(torch.randn(V, d), dim=1).half()
# token 7 = "intended output": make h dominated by D[7]
h = (D[7].float() * 10 + 0.05 * torch.randn(d)).reshape(1, 1, d)

ab = ProtectedDynamicAblator(layers, band=[1])

print("[1] protection preserves the output direction")
with ab:
    ab.mode = {"dicts": {1: D}, "k": 5, "nonneg": True,
               "protect_sets": torch.tensor([7])}
    out_p = layers[1](h.clone())
    ab.mode = {"dicts": {1: D}, "k": 5, "nonneg": True, "protect_sets": None}
    out_u = layers[1](h.clone())
    ab.mode = None
    out_off = layers[1](h.clone())
keep_p = float(out_p.reshape(-1).float() @ D[7].float())
keep_u = float(out_u.reshape(-1).float() @ D[7].float())
orig = float(h.reshape(-1) @ D[7].float())
check("hooks off = identity", torch.allclose(out_off, h))
check("unprotected removes the output component (<20% left)",
      keep_u / orig < 0.2)
check("protected keeps most of it (>3x unprotected)", keep_p > 3 * keep_u)

print("[2] nonneg mask: anti-aligned directions are never selected")
h_neg = (-D[3].float() * 10).reshape(1, 1, d)
with ab:
    ab.mode = {"dicts": {1: D}, "k": 5, "nonneg": True, "protect_sets": None}
    layers[1](h_neg.clone())
    ab.mode = None
check("anti-aligned row 3 not selected", 3 not in ab.log.last_selected)
scores_check = (h_neg.reshape(-1).to(D.dtype) @ D.T).float()
check("all selected rows had positive score",
      all(scores_check[i] > 0 for i in ab.log.last_selected))

print("[3] per-position protect sets align with sequence positions")
T = 4
hh = torch.stack([(D[i].float() * 10).reshape(d) for i in (11, 12, 13, 14)])\
    .reshape(1, T, d)
psets = torch.tensor([[11], [12], [13], [14]])  # protect own token per pos
with ab:
    ab.mode = {"dicts": {1: D}, "k": 3, "nonneg": True, "protect_sets": psets}
    out_seq = layers[1](hh.clone())
    check("pos-0 protected id 11 not selected", 11 not in ab.log.last_selected)
    ab.mode = {"dicts": {1: D}, "k": 3, "nonneg": True, "protect_sets": None}
    out_unp = layers[1](hh.clone())
    ab.mode = None
for t, tokid in enumerate((11, 12, 13, 14)):
    keep_pt = float(out_seq[0, t].float() @ D[tokid].float())
    keep_ut = float(out_unp[0, t].float() @ D[tokid].float())
    check(f"pos {t}: protection preserves token {tokid} (> 3x unprotected)",
          keep_pt > 3 * max(keep_ut, 0.1))

print("[4] step log records blocking + energy")
check("blocked count positive", ab.log.protected_hits_blocked > 0)
check("removed energy recorded", len(ab.log.removed_energy) > 0)
check("hooks detach", len(ab._handles) == 0)

print("ALL PROTECTED-DYNAMIC TESTS PASS")
