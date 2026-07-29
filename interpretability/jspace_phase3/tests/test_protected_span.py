# §15.3 protected-span suite.
import torch

from jspace_phase3.protected_span import span_safe_j_basis, span_overlap_report

D = 64


def _rows(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, D, generator=g)


def test_label_behaviour_retained():
    """No protected rows ⇒ exactly the label-protected basis."""
    sel = _rows(5)
    r = span_safe_j_basis(sel, None)
    assert r.lost_rank == 0 and r.requested_rank == 5
    assert r.basis.effective_rank == 5
    assert torch.allclose(r.row_survival, torch.ones(5))


def test_selected_rows_inside_protected_span_become_null():
    prot = _rows(3, seed=1)
    coeff = _rows(4, seed=2)[:, :3]
    sel = coeff @ prot                       # entirely inside span(prot)
    r = span_safe_j_basis(sel, prot)
    assert r.basis.effective_rank == 0
    assert r.lost_rank == r.requested_rank == 3
    assert r.null_row_frac == 1.0
    assert float(r.row_survival.max()) < 1e-4


def test_partial_overlap_reduces_rank_and_reports_survival():
    prot = _rows(2, seed=3)
    free = _rows(2, seed=4)
    mixed = 0.9 * prot[0] / prot[0].norm() + 0.1 * free[0] / free[0].norm()
    sel = torch.stack([mixed, free[1]])
    r = span_safe_j_basis(sel, prot)
    assert r.requested_rank == 2
    assert r.basis.effective_rank == 2       # residual of mixed is nonzero
    assert r.row_survival[0] < 0.5 and r.row_survival[1] > 0.9
    # the safe basis is orthogonal to the protected span
    qp = r.protected_basis.basis
    assert float((r.basis.basis.T @ qp).abs().max()) < 1e-4


def test_duplicate_and_collinear_protected_rows_rank_safe():
    """§2.4's hazard: duplicates/collinear rows must not inflate the
    protected rank or inject arbitrary directions."""
    base = _rows(2, seed=5)
    prot = torch.cat([base, base[:1], 3.0 * base[1:2],
                      base[:1] + 1e-7 * _rows(1, seed=6)])
    r = span_safe_j_basis(_rows(4, seed=7), prot)
    assert r.protected_basis.effective_rank == 2


def test_overlap_report_geometry():
    prot = _rows(3, seed=8)
    qp = torch.linalg.qr(prot.T, mode="reduced")[0]
    inside = qp[:, 0]
    outside_seed = _rows(1, seed=9)[0]
    outside = outside_seed - qp @ (qp.T @ outside_seed)
    outside = outside / outside.norm()
    basis = torch.stack([inside, outside], dim=1)
    rep = span_overlap_report(basis, prot,
                              answer_row=prot[0], h=_rows(1, seed=10)[0])
    assert rep.rank_selected == 2 and rep.rank_protected == 3
    assert abs(rep.projector_overlap - 1.0) < 1e-4   # exactly one shared dim
    assert 0.0 <= rep.answer_dir_survival <= 1.0
    assert rep.removed_energy_in_prot_frac is not None


def test_answer_direction_survival_bounds():
    prot = _rows(1, seed=11)
    sel = prot.clone()                        # selection IS the answer dir
    r = span_safe_j_basis(sel, prot)
    assert r.basis.effective_rank == 0        # span-safe deletes nothing
    rep = span_overlap_report(
        torch.nn.functional.normalize(sel, dim=1).T, prot,
        answer_row=prot[0])
    assert rep.answer_dir_survival < 1e-4     # label arm would delete it
