import pytest
import torch

from jspace_official_repro.interventions import (
    GeometryGated,
    HookPlan,
    InterventionSession,
    swap_coordinates,
)


def _vecs(seed=0, d=32, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    v_s = torch.randn(d, generator=generator, dtype=dtype)
    v_t = torch.randn(d, generator=generator, dtype=dtype)
    h = torch.randn(d, generator=generator, dtype=dtype)
    return v_s, v_t, h


def test_identical_and_near_identical_tokens_are_gated():
    v_s, _, h = _vecs()
    # s == t exactly is geometry-gated (cosine 1); so is a numerically
    # indistinguishable pair. The operational no-op is alpha=0 or the
    # caller skipping s==t (release rule: "skipping the answer itself").
    with pytest.raises(GeometryGated):
        swap_coordinates(h, v_s, v_s)
    with pytest.raises(GeometryGated):
        swap_coordinates(h, v_s, v_s + 1e-9 * torch.randn_like(v_s))


def test_alpha_zero_is_numerical_noop():
    v_s, v_t, h = _vecs(1)
    out, _ = swap_coordinates(h, v_s, v_t, alpha=0.0)
    assert torch.allclose(out, h, rtol=0, atol=1e-12)


def test_double_swap_recovers_original_fp64_exact():
    v_s, v_t, h = _vecs(2)
    once, _ = swap_coordinates(h, v_s, v_t)
    twice, _ = swap_coordinates(once, v_s, v_t)
    assert torch.allclose(twice, h, rtol=1e-12, atol=1e-12)


def test_double_swap_fp32_roundtrip_tolerance():
    # The hook path runs on fp32/bf16 residuals; the algebra is fp64 inside
    # but the round-trip is quantized at the output dtype.
    v_s, v_t, h = _vecs(2, dtype=torch.float32)
    once, _ = swap_coordinates(h, v_s, v_t)
    twice, _ = swap_coordinates(once, v_s, v_t)
    assert torch.allclose(twice, h, rtol=1e-5, atol=1e-5)


def test_non_orthogonal_pair_swaps_coordinates_exactly():
    d = 16
    generator = torch.Generator().manual_seed(3)
    v_s = torch.randn(d, generator=generator, dtype=torch.float64)
    v_t = 0.6 * v_s + 0.8 * torch.randn(d, generator=generator,
                                        dtype=torch.float64)  # oblique
    h = torch.randn(d, generator=generator, dtype=torch.float64)
    out, diag = swap_coordinates(h, v_s, v_t, collect=True)
    # Dense reference: least-squares coordinates in span(V).
    V = torch.stack([v_s, v_t], dim=1)
    c = torch.linalg.lstsq(V, h.unsqueeze(1)).solution.squeeze(1)
    expected = h + V @ (c.flip(0) - c)
    assert torch.allclose(out, expected, rtol=1e-10, atol=1e-10)
    assert diag.coord_reconstruction_error < 1e-10
    assert diag.orth_complement_error < 1e-10


def test_orthogonal_complement_unchanged():
    v_s, v_t, h = _vecs(4, d=24)
    out, _ = swap_coordinates(h, v_s, v_t)
    V = torch.stack([v_s, v_t], dim=1)
    Q, _ = torch.linalg.qr(V, mode="complete")
    complement = Q[:, 2:]
    assert torch.allclose(
        complement.T @ out, complement.T @ h, rtol=1e-11, atol=1e-11,
    )


def test_near_singular_pair_is_gated_not_padded():
    v_s, _, h = _vecs(5)
    v_t = v_s * (1 + 1e-9)
    with pytest.raises(GeometryGated):
        swap_coordinates(h, v_s, v_t)


def test_batch_and_item_order_invariance():
    v_s, v_t, _ = _vecs(6)
    generator = torch.Generator().manual_seed(7)
    batch = torch.randn(5, 32, generator=generator)
    together, _ = swap_coordinates(batch, v_s, v_t)
    one_by_one = torch.stack(
        [swap_coordinates(batch[i], v_s, v_t)[0] for i in range(5)]
    )
    assert torch.allclose(together, one_by_one, rtol=0, atol=0)
    permutation = torch.tensor([4, 2, 0, 3, 1])
    permuted, _ = swap_coordinates(batch[permutation], v_s, v_t)
    assert torch.allclose(permuted, together[permutation], rtol=0, atol=0)


def test_hook_fire_accounting_and_cleanup():
    from tiny_rms import TinyRMS

    model = TinyRMS()
    ids = model.encode("abcdefghij")
    v_s, v_t, _ = _vecs(8, d=model.d_model)
    plan = HookPlan(layers=[2, 3], positions=[1, 4])
    vectors = {2: (v_s, v_t), 3: (v_s, v_t)}
    with InterventionSession(model.layers, plan, kind="swap",
                             vectors=vectors) as session:
        model.forward(ids)
        session.assert_fires(1)
    assert not model.layers[2]._forward_hooks  # cleaned up
    # Exception path also cleans up.
    try:
        with InterventionSession(model.layers, plan, kind="swap",
                                 vectors=vectors):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert not model.layers[3]._forward_hooks
