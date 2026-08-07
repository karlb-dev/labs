# §15.3 matched-control suite (rank-safe, overlap-matched, persistent).
import pytest
import torch

from jspace_phase3.controls import (PersistentFrame,
                                    build_instant_matched_subspace,
                                    build_overlap_matched_subspace,
                                    consecutive_principal_cosines)

D = 96


def _vec(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(D, generator=g)


def _rows(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, D, generator=g)


def removed_energy(h, basis):
    h32 = h.float()
    c = basis.T @ h32
    return float(c @ c) / float(h32 @ h32)


def test_instant_exact_match_and_protected_orthogonality():
    h, prot = _vec(1), _rows(4, seed=2)
    basis, info = build_instant_matched_subspace(h, 5, 0.01, prot, seed=7)
    assert basis.shape == (D, 5)
    assert abs(removed_energy(h, basis) - 0.01) < 1e-6
    assert float((basis.T @ prot.float().T).abs().max()) < 1e-4
    assert not info["clamped"]
    assert info["protected_effective_rank"] == 4


def test_instant_rank_safe_under_coherent_protected_rows():
    """Duplicate / collinear / near-duplicate protected rows: the §2.4
    fix. Raw QR would insert numerically arbitrary completion columns."""
    base = _rows(2, seed=3)
    prot = torch.cat([base, base[:1], -2.0 * base[1:2],
                      base[:1] + 1e-8 * _rows(1, seed=4)])
    h = _vec(5)
    basis, info = build_instant_matched_subspace(h, 4, 0.02, prot, seed=9)
    assert info["protected_effective_rank"] == 2
    assert abs(removed_energy(h, basis) - 0.02) < 1e-6
    assert float((basis.T @ base.float().T).abs().max()) < 1e-4


def test_instant_impossible_dose_clamps_and_flags():
    prot = _rows(1, seed=6)
    h = prot[0] + 1e-3 * _vec(7)              # h almost inside prot span
    basis, info = build_instant_matched_subspace(h, 2, 0.9, prot, seed=1)
    assert info["clamped"]
    assert removed_energy(h, basis) <= info["e_max"] + 1e-6


def test_instant_deterministic_seeds():
    h, prot = _vec(8), _rows(3, seed=9)
    b1, _ = build_instant_matched_subspace(h, 4, 0.05, prot, seed=42)
    b2, _ = build_instant_matched_subspace(h, 4, 0.05, prot, seed=42)
    b3, _ = build_instant_matched_subspace(h, 4, 0.05, prot, seed=43)
    assert torch.allclose(b1, b2)
    assert not torch.allclose(b1, b3)


def test_overlap_matched_achieves_target():
    h, prot = _vec(10), _rows(6, seed=11)
    tau = 1.5
    basis, info = build_overlap_matched_subspace(h, 5, 0.02, tau, prot,
                                                 seed=3)
    assert abs(info["overlap_achieved"] - tau) < 1e-3
    assert abs(removed_energy(h, basis) - 0.02) < 1e-5
    assert not info["overlap_clamped"]
    gram = basis.T @ basis
    assert float((gram - torch.eye(5)).abs().max()) < 1e-4


def test_overlap_matched_clamps_unreachable_target():
    h, prot = _vec(12), _rows(2, seed=13)     # prot rank 2 ⇒ reachable ≤ 1
    basis, info = build_overlap_matched_subspace(h, 4, 0.02, 3.0, prot,
                                                 seed=5)
    assert info["overlap_clamped"]
    assert info["overlap_achieved"] <= 1.0 + 1e-4
    assert abs(removed_energy(h, basis) - 0.02) < 1e-5


def test_overlap_matched_zero_target_matches_instant_contract():
    h, prot = _vec(14), _rows(3, seed=15)
    basis, info = build_overlap_matched_subspace(h, 3, 0.03, 0.0, prot,
                                                 seed=8)
    assert info["overlap_achieved"] < 1e-6
    assert float((basis.T @ prot.float().T).abs().max()) < 1e-3


def test_persistent_frame_is_coherent_across_positions():
    """§6.4: persistent bases at nearby positions share orientation;
    independently-seeded instant bases do not."""
    prot = _rows(3, seed=16)
    hs = [_vec(20 + i) for i in range(6)]
    pf = PersistentFrame(D, max_rank=4, seed=99)
    pers = [pf.subspace_at(h, 4, 0.02, prot)[0] for h in hs]
    inst = [build_instant_matched_subspace(h, 4, 0.02, prot, seed=100 + i)[0]
            for i, h in enumerate(hs)]
    c_pers = consecutive_principal_cosines(pers)
    c_inst = consecutive_principal_cosines(inst)
    assert sum(c_pers) / len(c_pers) > 0.8
    assert sum(c_inst) / len(c_inst) < 0.5
    for h, b in zip(hs, pers):
        assert abs(removed_energy(h, b) - 0.02) < 1e-5


def test_persistent_rank_loss_raises():
    pf = PersistentFrame(8, max_rank=7, seed=1)
    h = _vec(30)[:8]
    prot = _rows(4, seed=31)[:, :8]
    with pytest.raises(RuntimeError):
        pf.subspace_at(h, 7, 0.02, prot)      # 7 + 1 + 4 > 8 dims
