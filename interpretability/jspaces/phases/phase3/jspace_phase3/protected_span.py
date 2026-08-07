# Workstream B core: span-safe output protection + span-overlap
# diagnostics (nextsteps §2.3, §6.1).
#
# LABEL protection (Phase 2 / paper arm) removes protected token IDs from
# the selection CANDIDATE SET; the span of the rows actually selected may
# still overlap the protected token directions, because the J dictionary
# is a coherent, nonorthogonal frame. SPAN-SAFE protection residualizes
# every selected row against the numerical span of the protected rows
# before building the ablation basis, so the removed subspace is
# orthogonal to the protected output geometry by construction.
#
# Lost rank is a SCIENTIFIC OBSERVABLE (how much of the nominal J
# selection lived inside the protected span) and is never refilled in the
# primary arm (nextsteps §6.1).
from __future__ import annotations

from dataclasses import dataclass

import torch

from jspace_part2.lib import BasisResult, orthonormal_basis_from_rows


@dataclass(frozen=True)
class SpanSafeResult:
    basis: BasisResult              # rank-safe basis of residualized rows
    protected_basis: BasisResult    # rank-safe basis of protected rows
    requested_rank: int             # rank of the label-protected selection
    lost_rank: int                  # requested - achieved after residualizing
    null_row_frac: float            # selected rows numerically inside prot span
    row_survival: torch.Tensor      # per selected row: |residual| / |row|


def span_safe_j_basis(selected_rows: torch.Tensor,
                      protected_rows: torch.Tensor | None,
                      *, relative_tolerance: float = 1e-5) -> SpanSafeResult:
    """nextsteps §6.1 reference construction, rank-safe end to end.

    With no protected rows this reduces EXACTLY to the label-protected
    basis (guarded by test_protected_span.py::test_label_behaviour_retained).
    """
    sel = selected_rows.float()
    if protected_rows is None or protected_rows.numel() == 0:
        basis = orthonormal_basis_from_rows(
            sel, relative_tolerance=relative_tolerance)
        surv = torch.ones(sel.shape[0])
        return SpanSafeResult(basis, orthonormal_basis_from_rows(
            sel.new_zeros((0, sel.shape[1]))), basis.effective_rank, 0,
            0.0, surv)

    q_prot = orthonormal_basis_from_rows(
        protected_rows.float(), relative_tolerance=relative_tolerance)
    label_basis = orthonormal_basis_from_rows(
        sel, relative_tolerance=relative_tolerance)

    safe_rows = sel
    if q_prot.effective_rank:
        qp = q_prot.basis.to(sel.device)
        safe_rows = sel - (sel @ qp) @ qp.T

    norm_before = sel.norm(dim=1).clamp_min(1e-30)
    survival = safe_rows.norm(dim=1) / norm_before
    null_frac = float((survival < 1e-4).float().mean()) if len(survival) else 0.0

    # Rank tolerance must be anchored to the PRE-residualization scale:
    # a row that lived entirely inside the protected span leaves ~1e-7
    # numerical junk whose own top singular value would otherwise pass a
    # purely relative test and fabricate rank.
    orig_scale = (float(label_basis.singular_values[0])
                  if label_basis.effective_rank else 1.0)
    basis = orthonormal_basis_from_rows(
        safe_rows, relative_tolerance=relative_tolerance,
        absolute_tolerance=max(1e-7, relative_tolerance * orig_scale))
    return SpanSafeResult(
        basis=basis, protected_basis=q_prot,
        requested_rank=label_basis.effective_rank,
        lost_rank=max(label_basis.effective_rank - basis.effective_rank, 0),
        null_row_frac=null_frac, row_survival=survival.cpu())


@dataclass(frozen=True)
class OverlapReport:
    """Geometry of one (selected span, protected span, state h) site —
    the per-position record nextsteps §2.3 requires."""
    rank_selected: int
    rank_protected: int
    principal_cosines: list[float]      # singular values of Qs^T Qp
    projector_overlap: float            # trace(P_S P_prot) = sum cos^2
    overlap_normalized: float           # / min(rank_S, rank_prot)
    protected_row_survival_min: float   # min_j |(I-P_S) p_j| / |p_j|
    protected_row_survival_mean: float
    answer_dir_survival: float | None   # |(I-P_S) d_ans| / |d_ans|
    removed_energy_in_prot_frac: float | None
    # fraction of the REMOVED h-energy that lies inside the protected span


def span_overlap_report(selected_basis: torch.Tensor,
                        protected_rows: torch.Tensor,
                        *, answer_row: torch.Tensor | None = None,
                        h: torch.Tensor | None = None,
                        relative_tolerance: float = 1e-5) -> OverlapReport:
    """All quantities in float32; selected_basis must be orthonormal
    [d, r] (the actual ablation projector's basis)."""
    qs = selected_basis.float()
    r_s = qs.shape[1]
    q_prot = orthonormal_basis_from_rows(
        protected_rows.float(), relative_tolerance=relative_tolerance)
    qp = q_prot.basis.to(qs.device)
    r_p = q_prot.effective_rank

    if r_s == 0 or r_p == 0:
        cos = []
        overlap = 0.0
    else:
        m = qs.T @ qp
        sv = torch.linalg.svdvals(m)
        cos = [round(float(v), 6) for v in sv.clamp(0, 1)]
        overlap = float((sv ** 2).sum())

    prot = protected_rows.float()
    pn = prot.norm(dim=1).clamp_min(1e-30)
    if r_s:
        resid = prot - (prot @ qs) @ qs.T
        surv = (resid.norm(dim=1) / pn)
    else:
        surv = torch.ones(prot.shape[0])

    ans_surv = None
    if answer_row is not None:
        a = answer_row.float()
        an = float(a.norm().clamp_min(1e-30))
        ans_surv = float((a - qs @ (qs.T @ a)).norm() / an) if r_s else 1.0

    rem_frac = None
    if h is not None and r_s:
        h32 = h.float()
        removed = qs @ (qs.T @ h32)
        rn2 = float(removed @ removed)
        if rn2 > 0 and r_p:
            in_prot = qp @ (qp.T @ removed)
            rem_frac = float(in_prot @ in_prot) / rn2
        else:
            rem_frac = 0.0

    return OverlapReport(
        rank_selected=r_s, rank_protected=r_p,
        principal_cosines=cos,
        projector_overlap=round(overlap, 6),
        overlap_normalized=round(overlap / max(min(r_s, r_p), 1), 6),
        protected_row_survival_min=round(float(surv.min()), 6),
        protected_row_survival_mean=round(float(surv.mean()), 6),
        answer_dir_survival=(round(ans_surv, 6)
                             if ans_surv is not None else None),
        removed_energy_in_prot_frac=(round(rem_frac, 6)
                                     if rem_frac is not None else None))
