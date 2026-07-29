# Phase 3 matched controls (nextsteps §2.4, §6.3, §6.4).
#
# NAMING DISCIPLINE (§15.2): "matched" alone is forbidden. These are:
#   instant_rank_energy_matched   the Phase 2 exact control, protected
#                                 basis now RANK-SAFE (SVD, not raw QR)
#   overlap_matched               additionally matches the J arm's
#                                 projector overlap with the protected
#                                 span (deliberately NOT span-orthogonal —
#                                 that is its purpose)
#   persistent_matched            instant match whose free orientation is
#                                 transported from ONE per-(item, layer)
#                                 base frame across positions (temporal
#                                 coherence secondary, §6.4)
#
# The Phase 2 module is untouched (frozen results reproduce at their
# recorded commits); everything here is Phase 3 code.
from __future__ import annotations

import torch

from jspace_part2.lib import orthonormal_basis_from_rows


def _seed_for(seed_base: int, layer: int, forward: int, pos: int) -> int:
    return (seed_base * 1_000_003 + layer * 10_007 + forward * 101 + pos) \
        % (2 ** 63 - 1)


def _prot_basis(prot_rows: torch.Tensor | None, tol: float = 1e-5):
    if prot_rows is None or prot_rows.numel() == 0:
        return None, 0
    b = orthonormal_basis_from_rows(prot_rows.float(),
                                    relative_tolerance=tol)
    return (b.basis if b.effective_rank else None), b.effective_rank


def build_instant_matched_subspace(
        h: torch.Tensor, rank: int, energy_frac: float,
        prot_rows: torch.Tensor | None, seed: int,
) -> tuple[torch.Tensor, dict]:
    """Rank-safe reimplementation of the Phase 2 exact control (§2.4):
    the protected complement is computed from an SVD basis with an
    explicit rank test, so duplicate / near-duplicate / collinear
    protected rows can no longer inject numerically arbitrary directions.
    Construction and guarantees otherwise identical: achieved rank ==
    rank, removed energy == energy_frac (clamped + flagged when the
    protected span holds nearly all of h), basis ⊥ every protected row."""
    d = h.shape[0]
    h32 = h.float()
    hn2 = float(h32 @ h32)
    g = torch.Generator().manual_seed(seed)
    G = torch.randn(rank, d, generator=g).to(h.device)

    qp, prot_rank = _prot_basis(prot_rows)
    if qp is not None:
        qp = qp.to(h.device)
        h_perp = h32 - qp @ (qp.T @ h32)
    else:
        h_perp = h32

    # orthogonalise the random block against span(h) + span(prot), rank-safely
    anchor = [h32.unsqueeze(0)]
    if qp is not None:
        anchor.append(qp.T)
    q_anchor = orthonormal_basis_from_rows(torch.cat(anchor, dim=0)).basis \
        .to(h.device)
    G = G - (G @ q_anchor) @ q_anchor.T
    u = orthonormal_basis_from_rows(G).basis
    if u.shape[1] < rank:
        raise RuntimeError(f"random block lost rank: {u.shape[1]} < {rank}")

    hp2 = float(h_perp @ h_perp)
    e_max = hp2 / max(hn2, 1e-30)
    clamped = energy_frac > e_max * 0.999999
    e = max(min(energy_frac, e_max * 0.999999), 0.0)

    if hp2 > 0 and e > 0:
        p_hat = h_perp / hp2 ** 0.5
        a = (e * hn2 / hp2) ** 0.5
        v1 = a * p_hat + (1.0 - a * a) ** 0.5 * u[:, 0]
        basis = torch.cat([v1.unsqueeze(1), u[:, 1:rank]], dim=1)
    else:
        basis = u[:, :rank]
    info = {"clamped": bool(clamped), "e_target": float(energy_frac),
            "e_max": float(e_max), "protected_effective_rank": prot_rank}
    return basis, info


def build_overlap_matched_subspace(
        h: torch.Tensor, rank: int, energy_frac: float,
        overlap_target: float, prot_rows: torch.Tensor | None, seed: int,
) -> tuple[torch.Tensor, dict]:
    """§6.3: match rank, removed h-energy, AND projector overlap with the
    protected span (trace(P_S P_prot) = overlap_target), random otherwise.

    Geometry: v1 carries the exact h-energy inside the protected
    complement (contributing 0 overlap); each remaining basis vector
    v_j = b·w_j + sqrt(1-b²)·u_j mixes an in-protected-span direction w_j
    (chosen ⊥ h so the energy match survives) with a fully-free u_j,
    contributing b² overlap. Reachable overlap is min(rank-1, prot_rank-1)
    — clamped and flagged, achieved value reported."""
    d = h.shape[0]
    h32 = h.float()
    hn2 = float(h32 @ h32)
    g = torch.Generator().manual_seed(seed)

    qp, prot_rank = _prot_basis(prot_rows)
    if qp is None:
        basis, info = build_instant_matched_subspace(
            h, rank, energy_frac, None, seed)
        info |= {"overlap_target": overlap_target, "overlap_achieved": 0.0,
                 "overlap_clamped": overlap_target > 1e-9}
        return basis, info
    qp = qp.to(h.device)

    # Directions INSIDE span(prot) that are ⊥ h. In qp coordinates a
    # span(prot) vector qp@z satisfies (qp@z)·h = z·(qp.T h), so the
    # admissible z live in the complement of c = qp.T h within R^{r_p}
    # (all of R^{r_p} when h has no protected component).
    c = qp.T @ h32                                         # [r_p]
    h_in_prot = qp @ c
    if float(c @ c) > 1e-20:
        eye = torch.eye(qp.shape[1], device=h.device)
        z_rows = eye - torch.outer(c, c) / float(c @ c)
        qz = orthonormal_basis_from_rows(z_rows).basis     # [r_p, m]
    else:
        qz = torch.eye(qp.shape[1], device=h.device)
    qw = qp @ qz                                           # [d, m] ⊂ span(prot), ⊥ h
    m = qw.shape[1]

    n_mix = max(rank - 1, 0)
    n_used = min(n_mix, m)          # each in-span direction used at most once
    reachable = float(n_used)
    tau = min(max(overlap_target, 0.0), reachable * 0.999999)
    clamped_overlap = overlap_target > reachable * 0.999999
    b2 = tau / n_used if n_used else 0.0

    # free block: ⊥ h, ⊥ prot span
    G = torch.randn(rank, d, generator=g).to(h.device)
    anchor = torch.cat([h32.unsqueeze(0), qp.T], dim=0)
    q_anchor = orthonormal_basis_from_rows(anchor).basis.to(h.device)
    G = G - (G @ q_anchor) @ q_anchor.T
    u = orthonormal_basis_from_rows(G).basis
    if u.shape[1] < rank:
        raise RuntimeError("free block lost rank")

    h_perp = h32 - h_in_prot
    hp2 = float(h_perp @ h_perp)
    e_max = hp2 / max(hn2, 1e-30)
    clamped_e = energy_frac > e_max * 0.999999
    e = max(min(energy_frac, e_max * 0.999999), 0.0)

    cols = []
    if hp2 > 0 and e > 0:
        p_hat = h_perp / hp2 ** 0.5
        a = (e * hn2 / hp2) ** 0.5
        cols.append(a * p_hat + (1.0 - a * a) ** 0.5 * u[:, 0])
    else:
        cols.append(u[:, 0])

    # in-span mixing directions: distinct random qw columns, ⊥ h; pure
    # free directions once the in-span supply is exhausted
    perm = torch.randperm(m, generator=g) if m else None
    for j in range(1, rank):
        if b2 <= 0 or j > n_used:
            cols.append(u[:, j])
        else:
            w = qw[:, perm[j - 1]]
            cols.append(b2 ** 0.5 * w + (1 - b2) ** 0.5 * u[:, j])
    basis = torch.stack(cols, dim=1)
    # orthonormality check (w_j distinct because n_mix <= m after clamp)
    gram = basis.T @ basis
    err = float((gram - torch.eye(rank, device=basis.device)).abs().max())
    if err > 1e-4:
        raise AssertionError(f"overlap-matched basis not orthonormal ({err})")
    achieved = float(((qp.T @ basis) ** 2).sum())
    info = {"clamped": bool(clamped_e), "e_target": float(energy_frac),
            "e_max": float(e_max), "protected_effective_rank": prot_rank,
            "overlap_target": float(overlap_target),
            "overlap_achieved": round(achieved, 6),
            "overlap_clamped": bool(clamped_overlap)}
    return basis, info


class PersistentFrame:
    """§6.4: one base random frame per (item, layer); positions reuse its
    orientation, adjusting only the h-aligned component for the local
    energy match. Report coherence via consecutive-position principal
    cosines (the J arm's spans persist; independent seeds do not)."""

    def __init__(self, d: int, max_rank: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.base = torch.randn(max_rank, d, generator=g)

    def subspace_at(self, h: torch.Tensor, rank: int, energy_frac: float,
                    prot_rows: torch.Tensor | None) -> tuple[torch.Tensor, dict]:
        d = h.shape[0]
        h32 = h.float()
        hn2 = float(h32 @ h32)
        qp, prot_rank = _prot_basis(prot_rows)
        anchor = [h32.unsqueeze(0)]
        if qp is not None:
            qp = qp.to(h.device)
            anchor.append(qp.T)
        q_anchor = orthonormal_basis_from_rows(
            torch.cat(anchor, dim=0)).basis.to(h.device)
        G = self.base[:rank].to(h.device)
        G = G - (G @ q_anchor) @ q_anchor.T
        u = orthonormal_basis_from_rows(G).basis
        if u.shape[1] < rank:
            raise RuntimeError("persistent frame lost rank at this site")
        h_perp = (h32 - qp @ (qp.T @ h32)) if qp is not None else h32
        hp2 = float(h_perp @ h_perp)
        e_max = hp2 / max(hn2, 1e-30)
        clamped = energy_frac > e_max * 0.999999
        e = max(min(energy_frac, e_max * 0.999999), 0.0)
        if hp2 > 0 and e > 0:
            p_hat = h_perp / hp2 ** 0.5
            a = (e * hn2 / hp2) ** 0.5
            v1 = a * p_hat + (1 - a * a) ** 0.5 * u[:, 0]
            basis = torch.cat([v1.unsqueeze(1), u[:, 1:rank]], dim=1)
        else:
            basis = u[:, :rank]
        return basis, {"clamped": bool(clamped), "e_target": float(energy_frac),
                       "e_max": float(e_max),
                       "protected_effective_rank": prot_rank}


def consecutive_principal_cosines(bases: list[torch.Tensor]) -> list[float]:
    """Mean principal cosine between consecutive position subspaces —
    the §6.4 coherence diagnostic (compare J arm vs persistent vs
    independent-seed controls)."""
    out = []
    for a, b in zip(bases, bases[1:]):
        if a.shape[1] == 0 or b.shape[1] == 0:
            continue
        sv = torch.linalg.svdvals(a.float().T @ b.float())
        out.append(round(float(sv.mean()), 6))
    return out
