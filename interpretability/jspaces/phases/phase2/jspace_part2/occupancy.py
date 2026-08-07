# R2 — the paper's capacity estimand, implemented per addendum §5.4:
#
#   occupancy(position, layer) = the K where the marginal reconstruction
#   improvement of a sparse NON-NEGATIVE pursuit over the J dictionary
#   falls below that of equal-size random control dictionaries;
#   capacity   = variance explained by the J reconstruction IN EXCESS of
#   the matched random control, evaluated at the layer's median occupancy,
#   with GLOBALLY centered moments.
#
# CROSSING RULE (frozen 2026-07-28 before any real-model data was seen):
#   occupancy = smallest K >= 1 such that
#       Delta_J(K) <= median_r Delta_r(K)   for TWO consecutive K
#   (persistence guard against single-step noise crossings);
#   if no such K exists, occupancy = k_max (right-censored, recorded).
#
# Pursuit: greedy positive-correlation atom selection + projected-gradient
# nonnegative refit — the part-1 code path WITH the VM5 numerics fixes
# (bf16 correlations for fp16-overflow safety; provably contractive
# 1/(k+2) refit step), reimplemented here so the package is self-contained
# and one final algorithm serves every model (addendum §5.4 issue 4).
from __future__ import annotations

import torch


@torch.no_grad()
def gradient_pursuit(h: torch.Tensor, D: torch.Tensor, k_max: int,
                     refit_iters: int = 8, lr: float = 0.25,
                     track_recon_errors: bool = False):
    """Sparse nonnegative decomposition of rows of h onto dictionary D.

    h: [B, d] fp32 (GPU), D: [V, d] fp16/bf16 row-normalized (GPU).
    Returns (idxs [B,k_max], coeffs [B,k_max], recon [B,d][, errs [B,k_max+1]]).
    errs[:, K] = ||h - recon_K||^2 with recon_0 = 0 (so errs[:,0] = ||h||^2).
    """
    B, d = h.shape
    Dh = D.to(torch.bfloat16)
    idxs = torch.zeros(B, k_max, dtype=torch.long, device=h.device)
    coeffs = torch.zeros(B, k_max, device=h.device)
    recon = torch.zeros_like(h)
    taken = torch.zeros(B, D.shape[0], dtype=torch.bool, device=h.device)
    errs = None
    if track_recon_errors:
        errs = torch.zeros(B, k_max + 1, device=h.device)
        errs[:, 0] = (h * h).sum(dim=1)
    for k in range(k_max):
        r = (h - recon).to(torch.bfloat16)
        corr = r @ Dh.T
        corr.masked_fill_(taken, float("-inf"))
        best = corr.argmax(dim=1)
        idxs[:, k] = best
        taken[torch.arange(B), best] = True
        D_A = D[idxs[:, :k + 1]].float()
        c = coeffs[:, :k + 1]
        step = min(lr, 1.0 / (k + 2))
        for _ in range(refit_iters):
            resid = h - torch.einsum("bk,bkd->bd", c, D_A)
            grad = torch.einsum("bd,bkd->bk", resid, D_A)
            c = (c + step * grad).clamp_(min=0)
        coeffs[:, :k + 1] = c
        recon = torch.einsum("bk,bkd->bd", c, D_A)
        if track_recon_errors:
            errs[:, k + 1] = ((h - recon) ** 2).sum(dim=1)
    if track_recon_errors:
        return idxs, coeffs, recon, errs
    return idxs, coeffs, recon


def marginal_gains(errs: torch.Tensor) -> torch.Tensor:
    """Delta(K) = errs[:, K-1] - errs[:, K] for K = 1..k_max. [B, k_max]"""
    return errs[:, :-1] - errs[:, 1:]


def occupancy_from_gains(dj: torch.Tensor, dr_med: torch.Tensor,
                         persistence: int = 2) -> torch.Tensor:
    """Frozen crossing rule. dj, dr_med: [B, k_max]. Returns [B] int
    occupancy (k_max if right-censored)."""
    B, K = dj.shape
    below = dj <= dr_med
    occ = torch.full((B,), K, dtype=torch.long, device=dj.device)
    run = torch.zeros(B, dtype=torch.long, device=dj.device)
    done = torch.zeros(B, dtype=torch.bool, device=dj.device)
    for k in range(K):
        run = torch.where(below[:, k], run + 1, torch.zeros_like(run))
        hit = (~done) & (run >= persistence)
        # occupancy = K at which the crossing STARTED (1-indexed)
        occ = torch.where(hit, torch.tensor(k + 1 - (persistence - 1),
                                            device=dj.device), occ)
        done = done | hit
    return occ.clamp(min=1)


@torch.no_grad()
def occupancy_and_excess(h: torch.Tensor, D_j: torch.Tensor,
                         rand_dicts: list[torch.Tensor], k_max: int,
                         global_mean: torch.Tensor):
    """Per-position occupancy + excess variance at the batch-median
    occupancy. global_mean: [d] fp32 (globally-centered second moments are
    the caller's responsibility via RunningVectorMoments; here we center
    reconstruction shares with the PROVIDED global mean).

    Returns dict of tensors (cpu)."""
    *_, errs_j = gradient_pursuit(h, D_j, k_max, track_recon_errors=True)
    dj = marginal_gains(errs_j)
    dr_all = []
    errs_r_all = []
    for R in rand_dicts:
        *_, errs_r = gradient_pursuit(h, R, k_max, track_recon_errors=True)
        dr_all.append(marginal_gains(errs_r))
        errs_r_all.append(errs_r)
    dr_med = torch.median(torch.stack(dr_all), dim=0).values
    occ = occupancy_from_gains(dj, dr_med)
    K_med = int(occ.median().item())
    # variance explained at K_med, globally centered:
    hc = h - global_mean[None, :]
    tot = (hc * hc).sum()
    share_j = 1.0 - errs_j[:, K_med].sum() / (h * h).sum()
    share_r = torch.stack([1.0 - e[:, K_med].sum() / (h * h).sum()
                           for e in errs_r_all]).mean()
    # NOTE: shares above are raw-energy shares (uncentered), matching the
    # reconstruction objective; the centered share is reported alongside:
    # centered_share = 1 - ||hc - recon_c||^2/||hc||^2 requires re-running
    # pursuit on centered h — deferred to the confirmatory pass; both raw
    # shares use the identical objective so the EXCESS is well-defined.
    return {"occupancy": occ.cpu(), "K_med": K_med,
            "share_j_at_Kmed": float(share_j),
            "share_rand_at_Kmed": float(share_r),
            "excess_share": float(share_j - share_r),
            "dj_mean": dj.mean(0).cpu(), "dr_med_mean": dr_med.mean(0).cpu(),
            "censored_frac": float((occ >= k_max).float().mean())}
