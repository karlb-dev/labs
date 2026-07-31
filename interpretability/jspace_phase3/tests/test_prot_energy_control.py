# The sharp leakage control: matches total removed energy AND the
# fraction of it taken from inside the protected span.
import torch

from jspace_phase3.controls import build_prot_energy_matched_subspace

D = 96


def _vec(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(D, generator=g)


def _rows(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, D, generator=g)


def _measure(h, basis, prot):
    h32 = h.float()
    removed = basis @ (basis.T @ h32)
    total = float(removed @ removed) / float(h32 @ h32)
    qp = torch.linalg.qr(prot.float().T, mode="reduced")[0]
    inp = qp @ (qp.T @ removed)
    frac = float(inp @ inp) / max(float(removed @ removed), 1e-30)
    return total, frac


def test_matches_total_and_protected_share():
    h, prot = _vec(1), _rows(8, seed=2)
    for target_frac in (0.0, 0.2, 0.4, 0.8):
        basis, info = build_prot_energy_matched_subspace(
            h, 10, 0.02, target_frac, prot, seed=5)
        assert basis.shape == (D, 10)
        total, frac = _measure(h, basis, prot)
        assert abs(total - 0.02) < 1e-4, (target_frac, total)
        assert abs(frac - target_frac) < 2e-3, (target_frac, frac)
        assert abs(info["prot_energy_achieved"] - target_frac) < 2e-3


def test_orthonormal_and_deterministic():
    h, prot = _vec(3), _rows(6, seed=4)
    b1, _ = build_prot_energy_matched_subspace(h, 8, 0.03, 0.35, prot, seed=7)
    b2, _ = build_prot_energy_matched_subspace(h, 8, 0.03, 0.35, prot, seed=7)
    b3, _ = build_prot_energy_matched_subspace(h, 8, 0.03, 0.35, prot, seed=8)
    assert torch.allclose(b1, b2)
    assert not torch.allclose(b1, b3)
    assert float((b1.T @ b1 - torch.eye(8)).abs().max()) < 1e-4


def test_unreachable_protected_share_clamps():
    """h nearly orthogonal to span(prot) ⇒ little in-span energy to take."""
    prot = _rows(2, seed=9)
    qp = torch.linalg.qr(prot.float().T, mode="reduced")[0]
    h = _vec(10)
    h = h - qp @ (qp.T @ h)                 # h ⊥ span(prot) exactly
    basis, info = build_prot_energy_matched_subspace(
        h, 5, 0.02, 0.9, prot, seed=11)
    total, frac = _measure(h, basis, prot)
    assert info["clamped"]
    assert frac < 1e-3                      # cannot take what isn't there
    assert abs(total - 0.02) < 1e-4         # total match preserved


def test_rank_one_two_component_target_is_marked_clamped():
    """Rank 1 cannot use separate in-span and out-of-span energy vectors."""
    h, prot = _vec(21), _rows(8, seed=22)
    basis, info = build_prot_energy_matched_subspace(
        h, 1, 0.02, 0.5, prot, seed=23)
    assert basis.shape == (D, 1)
    assert info["clamped"]
    assert info["rank_component_clamped"]


def test_no_protected_rows_falls_back_to_instant():
    h = _vec(12)
    basis, info = build_prot_energy_matched_subspace(h, 4, 0.05, 0.5, None,
                                                     seed=13)
    total, _ = _measure(h, basis, _rows(1, seed=14))
    assert abs(total - 0.05) < 1e-5
    assert basis.shape == (D, 4)
