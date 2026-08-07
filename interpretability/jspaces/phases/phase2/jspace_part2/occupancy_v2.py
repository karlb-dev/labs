# N1.6 — corrected capacity estimator (nextsteps_2_2 §2.2).
#
# THE DEFECT THIS REPAIRS. `occupancy.py` computed `hc = h - global_mean`
# and then never used it: both `share_j` and `share_r` were RAW-energy
# shares `1 - ||h-recon||^2/||h||^2`. The report and handout described the
# result as the paper's globally centered excess VARIANCE. Two different
# estimands were fused under one label. The occupancy (crossing) number is
# unaffected; the excess-share number was not the preregistered quantity.
#
# v2 returns THREE separately named outputs so no label can drift again:
#   occupancy_crossing_k            sparse-support crossing vs random controls
#   raw_reconstruction_excess       the pilot quantity, retained for continuity
#   centered_variance_explained_excess   the confirmatory capacity endpoint
#
# Both centered candidates from the review are computed; the confirmatory
# choice is Candidate B (centered R^2), because it is the only one that
# penalises a reconstruction for missing the centered TARGET rather than
# merely having centered energy of the right size. Candidate A is kept as
# prespecified sensitivity. Naming them both, always, is the point.
#
# SOLVER REPAIRS (same file, §2.2 list):
#   * positive-support exhaustion: a row whose best residual correlation is
#     non-positive STOPS growing instead of taking a harmful atom and
#     burning a slot of K;
#   * per-row achieved support is reported, so variable support sizes are
#     explicit rather than implied by zeroed coefficients;
#   * the crossing rule's persistence is a parameter with a sensitivity
#     report, not a buried constant.
from __future__ import annotations

from dataclasses import dataclass

import torch

from .occupancy import marginal_gains, occupancy_from_gains


@dataclass
class PursuitResult:
    idxs: torch.Tensor          # [B, k_max]
    coeffs: torch.Tensor        # [B, k_max]
    recon: torch.Tensor         # [B, d]
    errs: torch.Tensor          # [B, k_max+1]
    achieved_support: torch.Tensor   # [B] atoms actually taken (<= k_max)
    recons_by_k: list | None = None  # optional [k_max+1] x [B, d]


@torch.no_grad()
def gradient_pursuit_v2(h: torch.Tensor, D: torch.Tensor, k_max: int,
                        refit_iters: int = 8, lr: float = 0.25,
                        keep_recons: bool = False) -> PursuitResult:
    """Sparse nonnegative pursuit with positive-support exhaustion.

    h: [B, d] fp32, D: [V, d] row-normalised. When no untaken atom has a
    POSITIVE residual correlation, that row is frozen: its error stays flat
    for the remaining K, which is the honest statement that the dictionary
    has nothing more to offer it. v1 took argmax regardless, so a row could
    consume K with atoms that did not help."""
    B, d = h.shape
    Dh = D.to(torch.bfloat16)
    idxs = torch.zeros(B, k_max, dtype=torch.long, device=h.device)
    coeffs = torch.zeros(B, k_max, device=h.device)
    recon = torch.zeros_like(h)
    taken = torch.zeros(B, D.shape[0], dtype=torch.bool, device=h.device)
    errs = torch.zeros(B, k_max + 1, device=h.device)
    errs[:, 0] = (h * h).sum(dim=1)
    achieved = torch.zeros(B, dtype=torch.long, device=h.device)
    frozen = torch.zeros(B, dtype=torch.bool, device=h.device)
    recons = [recon.clone()] if keep_recons else None

    for k in range(k_max):
        r = (h - recon).to(torch.bfloat16)
        corr = (r @ Dh.T).float()
        corr.masked_fill_(taken, float("-inf"))
        best_val, best = corr.max(dim=1)
        # positive-support exhaustion
        exhausted = ~torch.isfinite(best_val) | (best_val <= 0)
        frozen = frozen | exhausted
        active = ~frozen
        if active.any():
            idxs[active, k] = best[active]
            taken[torch.arange(B, device=h.device)[active], best[active]] = True
            achieved[active] += 1
        if not active.any():
            errs[:, k + 1] = errs[:, k]
            if keep_recons:
                recons.append(recon.clone())
            continue
        # refit over each row's OWN taken atoms (frozen rows keep theirs)
        D_A = D[idxs[:, :k + 1]].float()                 # [B, k+1, d]
        slot_active = (torch.arange(k + 1, device=h.device)[None, :]
                       < achieved[:, None])
        D_A = D_A * slot_active.unsqueeze(-1)
        c = coeffs[:, :k + 1]
        step = min(lr, 1.0 / (k + 2))
        for _ in range(refit_iters):
            resid = h - torch.einsum("bk,bkd->bd", c, D_A)
            grad = torch.einsum("bd,bkd->bk", resid, D_A)
            c = (c + step * grad).clamp_(min=0) * slot_active
        coeffs[:, :k + 1] = c
        recon = torch.einsum("bk,bkd->bd", c, D_A)
        errs[:, k + 1] = ((h - recon) ** 2).sum(dim=1)
        if keep_recons:
            recons.append(recon.clone())
    return PursuitResult(idxs, coeffs, recon, errs, achieved, recons)


@torch.no_grad()
def centered_shares(H: torch.Tensor, R: torch.Tensor,
                    global_mean: torch.Tensor | None = None) -> dict:
    """Both centered candidates from nextsteps_2_2 §2.2 / §8.4.

    H: [B, d] activations, R: [B, d] their reconstructions.
    global_mean: the corpus mean to center by (batch mean if None)."""
    mu = H.mean(dim=0, keepdim=True) if global_mean is None \
        else global_mean.reshape(1, -1).to(H.dtype)
    Hc = H - mu
    Rc = R - R.mean(dim=0, keepdim=True)
    den = Hc.square().sum()
    # Candidate A: centered reconstruction variance share
    share_A = float(Rc.square().sum() / den)
    # Candidate B: centered R^2 (CONFIRMATORY choice)
    r2_B = float(1.0 - (Hc - Rc).square().sum() / den)
    # raw-energy share (the pilot quantity)
    raw = float(1.0 - (H - R).square().sum() / H.square().sum())
    return {"centered_variance_share_A": share_A,
            "centered_r2_B": r2_B, "raw_energy_share": raw}


@torch.no_grad()
def occupancy_and_excess_v2(h: torch.Tensor, D_j: torch.Tensor,
                            rand_dicts: list[torch.Tensor], k_max: int,
                            global_mean: torch.Tensor,
                            persistence: int = 2,
                            persistence_sensitivity=(1, 2, 3)) -> dict:
    """Three separately named capacity outputs. Never returns one number
    that could be read as either estimand."""
    pj = gradient_pursuit_v2(h, D_j, k_max, keep_recons=True)
    dj = marginal_gains(pj.errs)
    dr_all, prs = [], []
    for R in rand_dicts:
        pr = gradient_pursuit_v2(h, R, k_max, keep_recons=True)
        dr_all.append(marginal_gains(pr.errs))
        prs.append(pr)
    dr_med = torch.median(torch.stack(dr_all), dim=0).values

    occ = occupancy_from_gains(dj, dr_med, persistence=persistence)
    K_med = int(occ.median().item())
    sens = {f"persistence_{p}": int(
        occupancy_from_gains(dj, dr_med, persistence=p).median().item())
        for p in persistence_sensitivity}

    # shares evaluated at the median occupancy, on the SAME K for both arms
    Rj = pj.recons_by_k[K_med]
    cj = centered_shares(h, Rj, global_mean)
    cr = [centered_shares(h, p.recons_by_k[K_med], global_mean) for p in prs]

    def mean_of(key):
        return float(sum(c[key] for c in cr) / len(cr))

    return {
        # 1. sparse-support crossing
        "occupancy_crossing_k": occ.cpu(),
        "occupancy_median": K_med,
        "occupancy_censored_frac": float((occ >= k_max).float().mean()),
        "occupancy_persistence_sensitivity": sens,
        "achieved_support_mean_j": float(pj.achieved_support.float().mean()),
        "rows_exhausted_before_kmax": int((pj.achieved_support < k_max).sum()),
        # 2. pilot continuity
        "raw_reconstruction_excess": cj["raw_energy_share"] - mean_of("raw_energy_share"),
        "raw_share_j": cj["raw_energy_share"],
        "raw_share_rand": mean_of("raw_energy_share"),
        # 3. CONFIRMATORY capacity endpoint
        "centered_variance_explained_excess": cj["centered_r2_B"] - mean_of("centered_r2_B"),
        "centered_r2_j": cj["centered_r2_B"],
        "centered_r2_rand": mean_of("centered_r2_B"),
        # prespecified sensitivity
        "centered_variance_share_excess_A": (cj["centered_variance_share_A"]
                                             - mean_of("centered_variance_share_A")),
        "dj_mean": dj.mean(0).cpu(), "dr_med_mean": dr_med.mean(0).cpu(),
        "definitions": {
            "occupancy_crossing_k": "smallest K with dJ(K) <= median_r dR(K) "
                                    "for `persistence` consecutive K",
            "raw_reconstruction_excess": "1 - ||h-r||^2/||h||^2, J minus random "
                                         "(PILOT quantity; not variance)",
            "centered_variance_explained_excess": "centered R^2 (Candidate B), "
                                                  "J minus random — CONFIRMATORY",
            "centered_variance_share_excess_A": "centered reconstruction "
                                                "variance share (Candidate A) "
                                                "— sensitivity only",
        },
    }
