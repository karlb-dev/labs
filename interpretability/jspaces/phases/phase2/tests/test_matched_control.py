# Conformance tests for the geometry-matched primary control
# (dyn_energy_rank_matched_random) — freeze-blocking condition 2.
# CPU-only, synthetic; the GPU dev-validation gates live in
# experiments/mc_dev_validation.py and are committed before that run.
import torch

from jspace_part2.matched_control import (MatchedControlAblatorV2,
                                          _seed_for, build_matched_subspace,
                                          profile_from_log)
from jspace_part2.protected_dynamic_v2 import (PositionRecord, V2Log)

D_MODEL = 96
VOCAB = 128


def _rand_h(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(D_MODEL, generator=g)


def _prot_rows(seed=1, n=6):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(
        torch.randn(n, D_MODEL, generator=g), dim=1)


def test_exact_rank_and_energy():
    h = _rand_h()
    for r in (1, 3, 8):
        for e in (0.001, 0.01, 0.2):
            basis, info = build_matched_subspace(h, r, e, _prot_rows(), 42)
            assert basis.shape == (D_MODEL, r)
            assert not info["clamped"]
            # orthonormal
            gram = basis.T @ basis
            assert torch.allclose(gram, torch.eye(r), atol=1e-5)
            # removed energy exact
            coef = basis.T @ h
            removed = float(coef @ coef) / float(h @ h)
            assert abs(removed - e) < 1e-5, (r, e, removed)
    print("ok  rank and removed-energy match exactly across r x e grid")


def test_protected_orthogonality():
    h, P = _rand_h(3), _prot_rows(4)
    basis, _ = build_matched_subspace(h, 5, 0.05, P, 7)
    assert float((basis.T @ P.T).abs().max()) < 1e-5
    print("ok  control basis is orthogonal to every protected row")


def test_determinism_and_seed_sensitivity():
    h, P = _rand_h(5), _prot_rows(6)
    b1, _ = build_matched_subspace(h, 4, 0.03, P, 99)
    b2, _ = build_matched_subspace(h, 4, 0.03, P, 99)
    b3, _ = build_matched_subspace(h, 4, 0.03, P, 100)
    assert torch.equal(b1, b2)
    assert not torch.allclose(b1, b3)
    assert _seed_for(0, 24, 0, 5) != _seed_for(0, 24, 0, 6)
    print("ok  deterministic under seed; distinct across seeds/positions")


def test_clamp_when_protected_span_holds_h():
    # h almost entirely inside the protected span -> e_target unreachable
    P = _prot_rows(8, n=4)
    h = P[0] * 10.0 + 1e-3 * _rand_h(9)
    basis, info = build_matched_subspace(h, 2, 0.5, P, 11)
    assert info["clamped"]
    coef = basis.T @ h
    removed = float(coef @ coef) / float(h @ h)
    assert removed <= info["e_max"] + 1e-6
    print("ok  unreachable energy is clamped and flagged, never faked")


def test_randomness_vs_reference_span():
    # matched subspace should not preferentially align with an arbitrary
    # fixed span beyond the h-alignment it is FORCED to carry
    h = _rand_h(12)
    ref = torch.linalg.qr(torch.randn(D_MODEL, 5,
                          generator=torch.Generator().manual_seed(13)))[0]
    cos2 = []
    for seed in range(40):
        basis, _ = build_matched_subspace(h, 5, 0.02, None, seed)
        # exclude the forced h-direction: v_1 carries it; u_2..u_r random
        for i in range(1, 5):
            v = basis[:, i]
            cos2.append(float((ref.T @ v).square().sum()))
    mean_cos2 = sum(cos2) / len(cos2)
    # expectation for a random unit vector: rank(ref)/d = 5/96 ~ 0.052
    assert 0.02 < mean_cos2 < 0.11, mean_cos2
    print(f"ok  free directions are unbiased (mean cos^2 {mean_cos2:.3f} "
          f"~ 5/96 = {5/96:.3f})")


class _FakeLayer(torch.nn.Module):
    def forward(self, x):
        return x


def _make_log(T, layer, ranks, energies):
    log = V2Log()
    for t in range(T):
        log.positions.append(PositionRecord(
            layer=layer, phase="prefill", forward_index=0, position=t,
            requested_k=8, available_positive=20, selected_k=int(ranks[t]),
            effective_rank=int(ranks[t]),
            removed_energy_frac=float(energies[t]), protected_blocked=0))
    return log


def test_profile_roundtrip_and_ablator_matches():
    T, layer = 5, 2
    g = torch.Generator().manual_seed(21)
    ranks = torch.tensor([3, 0, 2, 4, 1])
    energies = torch.tensor([0.02, 0.0, 0.005, 0.08, 0.001])
    prof = profile_from_log(_make_log(T, layer, ranks, energies))
    assert torch.equal(prof[layer]["rank"], ranks)

    D = torch.nn.functional.normalize(
        torch.randn(VOCAB, D_MODEL, generator=g), dim=1)
    h = torch.randn(1, T, D_MODEL, generator=g)
    layers = [_FakeLayer() for _ in range(4)]
    ab = MatchedControlAblatorV2(layers, band=[layer])
    ab.phase, ab.forward_index = "prefill", 0
    psets = torch.randint(0, VOCAB, (T, 4), generator=g)
    ab.mode = {"dicts": {layer: D}, "profile": prof, "protect_sets": psets,
               "seed_base": 4242, "active_phases": {"prefill"}}
    out = ab._apply(h.clone(), layer)

    ms = ab.log.matched_summary()
    assert ms["n_positions"] == 4                     # rank-0 position skipped
    assert ms["rank_match_frac"] == 1.0
    assert ms["energy_rel_err_max"] < 1e-3
    assert ms["clamped_frac"] == 0.0
    assert ms["max_protected_cos"] < 1e-4
    # rank-0 position untouched
    assert torch.allclose(out[0, 1], h[0, 1], atol=1e-6)
    # others actually changed by the right energy
    for t in (0, 2, 3, 4):
        before = float(h[0, t].float() @ h[0, t].float())
        after = float(out[0, t].float() @ out[0, t].float())
        assert abs((1 - after / before) - float(energies[t])) < 1e-4
    print("ok  ablator consumes a J profile and matches it per position")


def test_profile_length_mismatch_refused():
    layer = 0
    prof = profile_from_log(_make_log(3, layer, torch.tensor([1, 1, 1]),
                                      torch.tensor([0.01, 0.01, 0.01])))
    D = torch.nn.functional.normalize(torch.randn(VOCAB, D_MODEL), dim=1)
    ab = MatchedControlAblatorV2([_FakeLayer()], band=[layer])
    ab.phase, ab.forward_index = "prefill", 0
    ab.mode = {"dicts": {layer: D}, "profile": prof, "protect_sets": None,
               "seed_base": 0, "active_phases": {"prefill"}}
    try:
        ab._apply(torch.randn(1, 5, D_MODEL), layer)
    except ValueError as e:
        assert "refusing to broadcast" in str(e)
        print("ok  profile/sequence length mismatch is refused")
        return
    raise AssertionError("length mismatch was not refused")


if __name__ == "__main__":
    test_exact_rank_and_energy()
    test_protected_orthogonality()
    test_determinism_and_seed_sensitivity()
    test_clamp_when_protected_span_holds_h()
    test_randomness_vs_reference_span()
    test_profile_roundtrip_and_ablator_matches()
    test_profile_length_mismatch_refused()
    print("ALL MATCHED-CONTROL TESTS PASS")
