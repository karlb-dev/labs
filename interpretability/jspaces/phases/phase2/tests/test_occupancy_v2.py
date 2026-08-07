# Guards the capacity-estimand repair (nextsteps_2_2 §2.2 test list).
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jspace_part2.occupancy_v2 import (  # noqa: E402
    centered_shares, gradient_pursuit_v2, occupancy_and_excess_v2)

fails = []


def check(cond, msg):
    print(f"  {'ok ' if cond else 'FAIL'} {msg}")
    if not cond:
        fails.append(msg)


torch.manual_seed(0)
d, V = 32, 200


def unit(x):
    return torch.nn.functional.normalize(x, dim=1)


print("[1] positive-support exhaustion: no atom is taken without positive corr")
D = unit(torch.randn(V, d))
# a row orthogonal to (almost) everything positive: use -sum of dictionary
h_hard = -(D[:5].sum(0, keepdim=True)) * 0.0 + torch.zeros(1, d)
h_hard[0, :] = 0.0                              # exactly zero: nothing to fit
p = gradient_pursuit_v2(h_hard, D, k_max=8)
check(int(p.achieved_support[0]) == 0,
      f"zero activation takes 0 atoms (v1 would take 8), got {int(p.achieved_support[0])}")
check(torch.allclose(p.errs[0], p.errs[0, 0].expand(9)),
      "error stays flat when the row is frozen")

print("[2] known-support recovery")
supp = [3, 11, 42]
coef = torch.tensor([2.0, 1.5, 0.7])
h = (D[supp] * coef[:, None]).sum(0, keepdim=True)
p = gradient_pursuit_v2(h, D, k_max=6)
got = set(p.idxs[0, :int(p.achieved_support[0])].tolist())
check(set(supp) <= got, f"recovers the true support {supp} in {sorted(got)}")
check(float(p.errs[0, 3]) < 0.05 * float(p.errs[0, 0]),
      "3 atoms explain >95% of a 3-atom signal")

print("[3] high-coherence / near-duplicate dictionary does not crash or NaN")
base = unit(torch.randn(20, d))
Dup = unit(torch.cat([base, base + 1e-3 * torch.randn(20, d)]))
hh = (Dup[0] * 3 + Dup[25] * 2).reshape(1, d)
p = gradient_pursuit_v2(hh, Dup, k_max=6)
check(torch.isfinite(p.errs).all(), "finite errors with near-duplicate atoms")
check(p.errs[0, -1] <= p.errs[0, 0] + 1e-5, "error never increases")

print("[4] centered vs raw are DIFFERENT quantities (the mislabel)")
# a reconstruction that captures the mean but no variation at all
H = torch.randn(64, d) * 0.5 + 10.0        # large offset -> raw share is high
R = H.mean(0, keepdim=True).expand_as(H).clone()
s = centered_shares(H, R)
check(s["raw_energy_share"] > 0.9,
      f"raw share is high for a mean-only reconstruction ({s['raw_energy_share']:.3f})")
check(s["centered_r2_B"] < 0.1,
      f"centered R^2 correctly near zero ({s['centered_r2_B']:.3f}) — "
      f"this gap IS the mislabel")
check(abs(s["centered_variance_share_A"]) < 0.05,
      "candidate A also near zero for a mean-only reconstruction")

print("[5] centered recovery with a known mean shift")
Hc = torch.randn(64, d)
H2 = Hc + 7.0
s_shift = centered_shares(H2, Hc + 7.0)       # perfect reconstruction
check(abs(s_shift["centered_r2_B"] - 1.0) < 1e-4,
      "perfect reconstruction gives centered R^2 = 1 under a mean shift")

print("[6] end-to-end returns three separately named outputs")
Dj = unit(torch.randn(V, d))
h = torch.stack([(Dj[i % V] * 3 + Dj[(i * 7) % V] * 2 + 0.1 * torch.randn(d))
                 for i in range(24)])
rands = [unit(torch.randn(V, d)) for _ in range(3)]
out = occupancy_and_excess_v2(h, Dj, rands, k_max=8, global_mean=h.mean(0))
for key in ("occupancy_crossing_k", "raw_reconstruction_excess",
            "centered_variance_explained_excess",
            "centered_variance_share_excess_A"):
    check(key in out, f"reports {key}")
check(out["raw_reconstruction_excess"] != out["centered_variance_explained_excess"],
      "raw and centered excess are not the same number")
check(set(out["occupancy_persistence_sensitivity"]) ==
      {"persistence_1", "persistence_2", "persistence_3"},
      "crossing-rule persistence sensitivity is reported")
check(out["occupancy_median"] >= 1, "occupancy >= 1")

print("[7] batch splitting: pursuit is row-independent")
p_all = gradient_pursuit_v2(h, Dj, k_max=6)
p_a = gradient_pursuit_v2(h[:10], Dj, k_max=6)
p_b = gradient_pursuit_v2(h[10:], Dj, k_max=6)
check(torch.allclose(p_all.errs, torch.cat([p_a.errs, p_b.errs]), atol=1e-5),
      "splitting the batch gives identical errors (resume-safe)")

print("[8] fp32 vs bf16 dictionary: selection agrees on well-separated signal")
p32 = gradient_pursuit_v2(h[:8].float(), Dj.float(), k_max=4)
p16 = gradient_pursuit_v2(h[:8].float(), Dj.to(torch.bfloat16).float(), k_max=4)
agree = (p32.idxs[:, :2] == p16.idxs[:, :2]).float().mean()
check(float(agree) > 0.85, f"top-2 selection agreement {float(agree):.2f}")

print("ALL OCCUPANCY V2 TESTS PASS" if not fails else f"{len(fails)} FAILURES")
sys.exit(1 if fails else 0)
