# §15.3 arm-behaviour suite on a tiny synthetic "transformer" (identity
# layers with hooks), CPU-only. What must hold:
#   * meanJ_label arm == exact v2 behaviour (golden vs jspace_part2)
#   * span-safe arm removes NOTHING along protected rows
#   * label arm CAN remove protected-row components (the §2.3 asymmetry —
#     this is the defect made visible, hence the whole workstream)
#   * matched arms consume the J profile and hit rank+energy exactly
#   * overlap-matched hits the J arm's logged overlap
import torch

from jspace_part2.protected_dynamic_v2 import ProtectedDynamicAblatorV2
from jspace_phase3.ablator3 import (Phase3JAblator, Phase3MatchedAblator,
                                    profile_from_p3log)

D, V, T = 48, 200, 7


class Blocks(torch.nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Identity() for _ in range(n)])

    def forward(self, h):
        for b in self.blocks:
            h = b(h)
        return h


def _setup(seed=0):
    g = torch.Generator().manual_seed(seed)
    dic = torch.nn.functional.normalize(
        torch.randn(V, D, generator=g), dim=1)
    h = torch.randn(1, T, D, generator=g)
    clean_logits = torch.randn(T, V, generator=g)
    psets = clean_logits.topk(5, dim=-1).indices
    return dic, h, psets


def _run(ab_cls, dic, h, psets, **mode_extra):
    net = Blocks()
    ab = ab_cls(net.blocks, band=[1])
    ab.phase, ab.forward_index = "prefill", 0
    ab.mode = {"dicts": {1: dic}, "k": 4, "nonneg": True,
               "protect_sets": psets, "active_phases": {"prefill"},
               **mode_extra}
    with ab:
        out = net(h.clone())
    ab.mode = None
    return out, ab


def test_label_arm_matches_v2_exactly():
    dic, h, psets = _setup(1)
    out3, _ = _run(Phase3JAblator, dic, h, psets)
    out2, _ = _run(ProtectedDynamicAblatorV2, dic, h, psets)
    assert torch.allclose(out3, out2, atol=1e-6)


def test_span_safe_preserves_protected_rows_label_does_not():
    dic, h, psets = _setup(2)
    out_lab, _ = _run(Phase3JAblator, dic, h, psets)
    out_safe, ab_safe = _run(Phase3JAblator, dic, h, psets, span_safe=True,
                             record_overlap=True)
    lab_delta, safe_delta = [], []
    for t in range(T):
        prot = dic[psets[t]]
        lab_delta.append(float(
            (prot @ (h[0, t] - out_lab[0, t])).abs().max()))
        safe_delta.append(float(
            (prot @ (h[0, t] - out_safe[0, t])).abs().max()))
    # span-safe: protected-row components untouched (float32 tolerance)
    assert max(safe_delta) < 1e-3
    # label arm: with a 200-row coherent dictionary some protected
    # component IS removed — the asymmetry this workstream exists for
    assert max(lab_delta) > 10 * max(safe_delta)
    # overlap log: span-safe overlap ≈ 0
    ov = [r.projector_overlap for r in ab_safe.log.overlap]
    assert max(ov) < 1e-4


def test_label_arm_overlap_is_logged_and_positive_here():
    dic, h, psets = _setup(3)
    _, ab = _run(Phase3JAblator, dic, h, psets, record_overlap=True)
    ov = [r.projector_overlap for r in ab.log.overlap]
    assert len(ov) == T
    assert max(ov) > 1e-3          # coherent frame ⇒ nonzero overlap
    ans = [r.answer_dir_survival for r in ab.log.overlap]
    assert all(a is None for a in ans)   # no answer_id passed
    assert all(r.protected_rank_before is None for r in ab.log.overlap)


def test_added_protection_geometry_is_rank_safe_and_complete():
    """The §6.2 audit distinguishes clean protection from genuinely
    added bridge rank and measures overlap against the pre-safe J span."""
    dic, h, base = _setup(31)
    # Include one already-protected id plus three new diagnostic rows.
    diagnostic = torch.tensor(
        [int(base[0, 0]), 81, 123, 177], dtype=torch.long)
    after = torch.cat([
        base,
        diagnostic.unsqueeze(0).expand(T, -1),
    ], dim=1)
    _, ab = _run(
        Phase3JAblator, dic, h, after, span_safe=True,
        record_overlap=True, base_protect_sets=base,
        diagnostic_ids=diagnostic, answer_ids=diagnostic)
    assert len(ab.log.overlap) == T
    for r in ab.log.overlap:
        assert r.n_diagnostic_ids == 4
        assert r.protected_rank_after == (
            r.protected_rank_before + r.added_rank)
        assert r.rank_selected_before == r.rank_selected + r.lost_rank
        assert r.added_selected_overlap is not None
        assert r.added_selected_overlap >= 0
        # Every diagnostic direction is in the protected-after span, so
        # a span-safe deletion must preserve it.
        assert r.diagnostic_dir_survival_min > 0.999
        assert r.answer_dir_survival_min > 0.999
        assert r.diagnostic_activation_score_mean is not None
        assert r.diagnostic_activation_score_max is not None
        assert r.diagnostic_base_overlap is not None
        assert -1e-6 <= r.diagnostic_base_overlap <= 4 + 1e-5
        assert r.diagnostic_answer_cosine_mean is not None
        assert r.removed_energy_l2_sq >= 0
        assert 0 <= r.removed_energy_frac <= 1 + 1e-5


def test_span_safe_lost_rank_observable():
    g = torch.Generator().manual_seed(4)
    base = torch.nn.functional.normalize(torch.randn(6, D, generator=g),
                                         dim=1)
    dic = torch.zeros(V, D)
    dic[:6] = base
    # rows 6..: perturbations of the first protected row at 1e-6 — inside
    # the protected span to float precision, so span-safe residualization
    # must null them (0.01-scale copies would leave a genuine ~7%-norm
    # residual subspace, which correctly KEEPS rank)
    for i in range(6, 30):
        dic[i] = torch.nn.functional.normalize(
            base[0] + 1e-6 * torch.randn(D, generator=g), dim=0)
    dic[30:] = torch.nn.functional.normalize(
        torch.randn(V - 30, D, generator=g), dim=1)
    h = base[0].repeat(1, T, 1) + 0.1 * torch.randn(1, T, D, generator=g)
    clean = torch.zeros(T, V)
    clean[:, 0] = 10.0                       # protect row 0 everywhere
    psets = clean.topk(1, dim=-1).indices
    _, ab = _run(Phase3JAblator, dic, h, psets, span_safe=True,
                 record_overlap=True)
    lost = [r.lost_rank for r in ab.log.overlap]
    nulls = [r.null_row_frac for r in ab.log.overlap]
    assert max(lost) >= 1
    assert max(nulls) > 0


def _j_profile(dic, h, psets):
    _, ab = _run(Phase3JAblator, dic, h, psets, record_overlap=True)
    return profile_from_p3log(ab.log, overlap_records=ab.log.overlap), ab


def _run_matched(variant, dic, h, psets, profile):
    net = Blocks()
    ab = Phase3MatchedAblator(net.blocks, band=[1])
    ab.phase, ab.forward_index = "prefill", 0
    ab.mode = {"dicts": {1: dic}, "profile": profile, "variant": variant,
               "protect_sets": psets, "seed_base": 11,
               "active_phases": {"prefill"}}
    with ab:
        out = net(h.clone())
    ab.mode = None
    return out, ab


def test_matched_variants_hit_rank_energy_and_overlap():
    dic, h, psets = _setup(5)
    profile, ab_j = _j_profile(dic, h, psets)
    for variant in ("instant_rank_energy_matched", "overlap_matched",
                    "persistent_matched"):
        _, ab = _run_matched(variant, dic, h, psets, profile)
        s = ab.log.matched_summary()
        assert s["rank_match_frac"] == 1.0, variant
        assert (s["energy_rel_err_max"] or 0) < 5e-3, variant
        if variant == "overlap_matched":
            assert abs(s["overlap_achieved_mean"]
                       - s["overlap_target_mean"]) < 5e-2
        else:
            assert s["max_protected_cos"] < 1e-3, variant


def test_matched_refuses_profile_length_mismatch():
    dic, h, psets = _setup(6)
    profile, _ = _j_profile(dic, h, psets)
    bad = {1: {"rank": profile[1]["rank"][:-1],
               "energy": profile[1]["energy"][:-1],
               "overlap": profile[1]["overlap"][:-1]}}
    import pytest
    with pytest.raises(Exception, match="refusing to broadcast"):
        _run_matched("instant_rank_energy_matched", dic, h, psets, bad)


def test_restrict_sets_limits_selection_to_allowed_rows():
    dic, h, psets = _setup(3)
    allowed = torch.tensor([5, 9, 17, 44])
    out, ab = _run(Phase3JAblator, dic, h, psets,
                   restrict_sets=allowed, record_ids=True)
    sel = {i for p in ab.log.positions if p.selected_ids
           for i in p.selected_ids}
    assert sel and sel <= set(allowed.tolist())
    # protection still binds inside the restricted set
    out2, ab2 = _run(Phase3JAblator, dic, h,
                     allowed[:2].unsqueeze(0).expand(h.shape[1], -1),
                     restrict_sets=allowed, record_ids=True)
    sel2 = {i for p in ab2.log.positions if p.selected_ids
            for i in p.selected_ids}
    assert sel2 <= {17, 44}


def test_inject_dir_energy_matched_substitution():
    dic, h, psets = _setup(4)
    inj = torch.randn(D)
    out_no, ab_no = _run(Phase3JAblator, dic, h, psets)
    out_inj, ab_inj = _run(Phase3JAblator, dic, h, psets, inject_dir=inj)
    # removal logging identical; the injection adds energy back
    f_no = [p.removed_energy_frac for p in ab_no.log.positions]
    f_inj = [p.removed_energy_frac for p in ab_inj.log.positions]
    assert f_no == f_inj
    d = (out_inj - out_no)[0]                       # [T, D]
    u = torch.nn.functional.normalize(inj.float(), dim=0)
    cos = torch.nn.functional.cosine_similarity(
        d, u.unsqueeze(0).expand_as(d), dim=1)
    assert bool((cos > 0.999).all())
    h2b = (h[0].float() ** 2).sum(1)
    h2a = (out_no[0].float() ** 2).sum(1)
    exp = (h2b - h2a).clamp_min(0).sqrt()
    assert torch.allclose(d.norm(dim=1), exp, atol=1e-4)
