# R2 solver validation on synthetic mixtures with known support
# (addendum §5.4 "Solver validation"). CPU, fast.
import sys

import torch

from jspace_part2.occupancy import (gradient_pursuit, marginal_gains,
                                    occupancy_from_gains)


def check(name, cond):
    if not cond:
        print(f"FAIL: {name}")
        sys.exit(1)
    print(f"  ok  {name}")


torch.manual_seed(0)
d, V, B, S = 128, 1000, 32, 6   # 6 true atoms per mixture

D = torch.nn.functional.normalize(torch.randn(V, d), dim=1)
Dh = D.half()
true_idx = torch.stack([torch.randperm(V)[:S] for _ in range(B)])
true_c = torch.rand(B, S) * 2 + 0.5
H = torch.einsum("bs,bsd->bd", true_c, D[true_idx]) + 0.01 * torch.randn(B, d)

print("[1] support recovery on synthetic nonneg mixtures")
idxs, coeffs, recon, errs = gradient_pursuit(H, Dh, k_max=12,
                                             track_recon_errors=True)
hits = 0
for b in range(B):
    hits += len(set(idxs[b, :S].tolist()) & set(true_idx[b].tolist()))
recall = hits / (B * S)
check(f"support recall@S >= 0.9 (got {recall:.3f})", recall >= 0.9)

print("[2] reconstruction error monotone nonincreasing")
check("monotone", bool((errs[:, 1:] <= errs[:, :-1] + 1e-4).all()))

print("[3] coefficients nonnegative")
check("nonneg", bool((coeffs >= 0).all()))

print("[4] duplicate atoms don't break the refit (contractive step)")
D_dup = D.clone()
D_dup[1] = D_dup[0]
D_dup[2] = D_dup[0] * 0.9999 + 0.0001 * torch.randn(d)
D_dup = torch.nn.functional.normalize(D_dup, dim=1)
h_dup = (D_dup[0] * 3).reshape(1, -1) + 0.01 * torch.randn(1, d)
*_, errs_dup = gradient_pursuit(h_dup, D_dup.half(), k_max=8,
                                track_recon_errors=True)
check("finite under near-duplicates", bool(torch.isfinite(errs_dup).all()))
check("no explosion", float(errs_dup[:, -1]) <= float(errs_dup[:, 0]))

print("[5] determinism")
i2, c2, r2, e2 = gradient_pursuit(H, Dh, k_max=12, track_recon_errors=True)
check("identical reruns", bool((i2 == idxs).all()
                               and torch.allclose(e2, errs)))

print("[6] NNLS-on-true-support comparison (quality bound)")
b0 = 0
A = D[true_idx[b0]].T                       # [d, S]
c_ls = torch.linalg.lstsq(A, H[b0]).solution.clamp(min=0)
err_ref = float(((H[b0] - A @ c_ls) ** 2).sum())
err_ours = float(errs[b0, S])
check(f"pursuit err at K=S within 2x of NNLS-support ref "
      f"({err_ours:.4f} vs {err_ref:.4f})",
      err_ours <= max(2 * err_ref, 0.05 * float(errs[b0, 0])))

print("[7] occupancy crossing rule recovers true sparsity vs random dicts")
dj = marginal_gains(errs)
dr = []
for seed in (1, 2, 3):
    g = torch.Generator().manual_seed(seed)
    R = torch.nn.functional.normalize(torch.randn(V, d, generator=g),
                                      dim=1).half()
    *_, er = gradient_pursuit(H, R, k_max=12, track_recon_errors=True)
    dr.append(marginal_gains(er))
dr_med = torch.median(torch.stack(dr), dim=0).values
occ = occupancy_from_gains(dj, dr_med)
med = float(occ.float().median())
check(f"median occupancy within +/-2 of true S={S} (got {med})",
      abs(med - S) <= 2)

print("[8] right-censoring recorded when J == random (no structure)")
H_noise = torch.randn(B, d)
*_, ej = gradient_pursuit(H_noise, Dh, k_max=10, track_recon_errors=True)
occ_n = occupancy_from_gains(marginal_gains(ej), dr_med[:, :10])
check("noise occupancy small or censored, never mid-range spurious lock",
      bool(((occ_n <= 3) | (occ_n >= 10)).float().mean() > 0.7))

print("ALL OCCUPANCY SOLVER TESTS PASS")
