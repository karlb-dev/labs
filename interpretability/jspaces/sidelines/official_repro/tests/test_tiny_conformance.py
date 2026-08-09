"""Tiny-model conformance: readout parity, fit algebraic identity vs
upstream, merge, interrupted resume, directional hook check, steering."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from tiny_rms import TinyRMS

import jlens
from jlens.fitting import fit as upstream_fit
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens

from jspace_official_repro.fitting import HalfFit, runtime_sentinel, sentinel_rel_diff
from jspace_official_repro.interventions import HookPlan, InterventionSession
from jspace_official_repro.readout import (
    g_folding_audit,
    readout_parity,
    recompute_readout,
    token_vectors,
)

PROMPTS = [
    "the quick brown fox jumps over the lazy dog " * 2,
    "colourless green ideas sleep furiously tonight " * 2,
    "a b c d e f g h i j k l m n o p q r s t u v w x y z " * 1,
    "pack my box with five dozen liquor jugs today " * 2,
]


@pytest.fixture(scope="module")
def model():
    return TinyRMS()


@pytest.fixture(scope="module")
def lens(model):
    return upstream_fit(model, PROMPTS[:2], source_layers=[1, 2, 3],
                        dim_batch=4, max_seq_len=64, skip_first=4)


def test_readout_parity_hard_stop_check_passes(model, lens):
    result = readout_parity(model, lens, PROMPTS[0], layers=[1, 3],
                            positions=[5, 10, -2])
    assert result["ok"], result
    assert result["max_abs_diff"] < 1e-4


def test_probe_form_matches_topk_order_not_logits(model, lens):
    # <v_t, h> must reproduce the transported readout's *order* (before the
    # final norm's per-position rescale breaks raw-logit equality).
    ids = model.encode(PROMPTS[0], max_length=64)
    with ActivationRecorder(model.layers, at=[2]) as recorder:
        model.forward(ids)
        h = recorder.activations[2][0][8].float()
    full = recompute_readout(model, lens, 2, h.unsqueeze(0))[0]
    vectors = token_vectors(lens, model, 2, list(range(32)), fold_gain=True)
    probe = vectors @ h
    # RMSNorm rescale is monotone per position only with folded gain: order
    # agreement is on the folded probe (contract §1c); raw equality fails.
    assert not torch.allclose(probe, full, rtol=1e-3, atol=1e-3)
    order_full = full.argsort(descending=True)[:5]
    order_probe = probe.argsort(descending=True)[:5]
    assert torch.equal(order_full, order_probe)


def test_g_folding_audit_runs(model, lens):
    result = g_folding_audit(lens, model, token_ids=list(range(8)),
                             layers=[1, 2, 3])
    assert result["applicable"]
    assert 0.0 < result["min_cosine"] <= 1.0


def test_hardened_fit_matches_upstream_bitwise(model, tmp_path):
    upstream = upstream_fit(model, PROMPTS, source_layers=[1, 2],
                            dim_batch=4, max_seq_len=64, skip_first=4)
    rows = [{"fit_index": i, "text": t} for i, t in enumerate(PROMPTS)]
    sentinel = runtime_sentinel(model, PROMPTS[0], [1, 2], target_layer=5,
                                dim_batch=4, max_seq_len=64, skip_first=4)
    half = HalfFit(half="T", prompts=rows, source_layers=[1, 2],
                   target_layer=5, dim_batch=4, local_dir=tmp_path,
                   drive_dir=None, sentinel=sentinel, max_seq_len=64,
                   skip_first=4, milestones=())
    summary = half.run(model)
    assert summary["n_done"] == len(PROMPTS)
    mine = half.final_lens()
    for layer in (1, 2):
        assert torch.equal(mine.jacobians[layer], upstream.jacobians[layer])
    assert mine.n_prompts == upstream.n_prompts


def test_interrupted_resume_reproduces_uninterrupted(model, tmp_path):
    rows = [{"fit_index": i, "text": t} for i, t in enumerate(PROMPTS)]
    sentinel = runtime_sentinel(model, PROMPTS[0], [1, 2], target_layer=5,
                                dim_batch=4, max_seq_len=64, skip_first=4)
    straight_dir = tmp_path / "straight"
    resumed_dir = tmp_path / "resumed"
    kwargs = dict(prompts=rows, source_layers=[1, 2], target_layer=5,
                  dim_batch=4, drive_dir=None, sentinel=sentinel,
                  max_seq_len=64, skip_first=4, milestones=())
    straight = HalfFit(half="S", local_dir=straight_dir, **kwargs)
    straight.run(model)
    resumed = HalfFit(half="R", local_dir=resumed_dir, **kwargs)
    resumed.run(model, stop_after=2)  # interrupt after 2 accepted
    resumed2 = HalfFit(half="R", local_dir=resumed_dir, **kwargs)
    resumed2.run(model)
    a, b = straight.final_lens(), resumed2.final_lens()
    for layer in (1, 2):
        assert torch.equal(a.jacobians[layer], b.jacobians[layer])


def test_resume_forbidden_on_sentinel_drift(model, tmp_path):
    rows = [{"fit_index": i, "text": t} for i, t in enumerate(PROMPTS)]
    sentinel = runtime_sentinel(model, PROMPTS[0], [1], target_layer=5,
                                dim_batch=4, max_seq_len=64, skip_first=4)
    half = HalfFit(half="D", prompts=rows, source_layers=[1], target_layer=5,
                   dim_batch=4, local_dir=tmp_path, drive_dir=None,
                   sentinel=sentinel, max_seq_len=64, skip_first=4,
                   milestones=())
    half.run(model, stop_after=2)
    drifted = dict(sentinel, norms_sha256="deadbeef")
    half2 = HalfFit(half="D", prompts=rows, source_layers=[1], target_layer=5,
                    dim_batch=4, local_dir=tmp_path, drive_dir=None,
                    sentinel=drifted, max_seq_len=64, skip_first=4,
                    milestones=())
    with pytest.raises(RuntimeError, match="sentinel"):
        half2.run(model)


def test_sentinel_repeat_agrees(model):
    a = runtime_sentinel(model, PROMPTS[1], [1, 2], target_layer=5,
                         dim_batch=4, max_seq_len=64, skip_first=4)
    b = runtime_sentinel(model, PROMPTS[1], [1, 2], target_layer=5,
                         dim_batch=4, max_seq_len=64, skip_first=4)
    assert sentinel_rel_diff(a, b) < 1e-6  # CPU deterministic


def test_merge_equals_upstream_merge(model, tmp_path):
    half_a = upstream_fit(model, PROMPTS[0::2], source_layers=[1],
                          dim_batch=4, max_seq_len=64, skip_first=4)
    half_b = upstream_fit(model, PROMPTS[1::2], source_layers=[1],
                          dim_batch=4, max_seq_len=64, skip_first=4)
    merged = JacobianLens.merge([half_a, half_b])
    full = upstream_fit(model, PROMPTS, source_layers=[1], dim_batch=4,
                        max_seq_len=64, skip_first=4)
    # merge is the n-weighted mean of half means == the full running mean
    assert torch.allclose(merged.jacobians[1], full.jacobians[1],
                          rtol=1e-6, atol=1e-7)


def test_directional_swap_changes_downstream_not_upstream(model, lens):
    ids = model.encode(PROMPTS[0], max_length=64)
    layer = 2
    record_at = [layer - 1, layer, layer + 1]
    with ActivationRecorder(model.layers, at=record_at) as recorder:
        model.forward(ids)
        clean = {l: recorder.activations[l].detach().clone() for l in record_at}
    vectors = token_vectors(lens, model, layer, [3, 9])
    plan = HookPlan(layers=[layer], positions=[6])
    with InterventionSession(model.layers, plan, kind="swap",
                             vectors={layer: (vectors[0], vectors[1])}) as session:
        with ActivationRecorder(model.layers, at=record_at) as recorder:
            model.forward(ids)
            patched = {l: recorder.activations[l].detach().clone()
                       for l in record_at}
        session.assert_fires(1)
    assert torch.equal(patched[layer - 1], clean[layer - 1])
    assert not torch.equal(patched[layer], clean[layer])
    assert not torch.equal(patched[layer + 1], clean[layer + 1])
    # And only the planned position changed at the patched layer:
    diff = (patched[layer] - clean[layer]).abs().sum(-1)[0]
    changed_positions = diff.nonzero(as_tuple=True)[0].tolist()
    assert changed_positions == [6]


def test_steering_adds_expected_vector(model, lens):
    ids = model.encode(PROMPTS[0], max_length=64)
    layer, position, strength, mean_norm = 3, 5, 2.0, 4.0
    v_t = token_vectors(lens, model, layer, [7])[0]
    with ActivationRecorder(model.layers, at=[layer]) as recorder:
        model.forward(ids)
        clean = recorder.activations[layer].detach().clone()
    plan = HookPlan(layers=[layer], positions=[position])
    with InterventionSession(
        model.layers, plan, kind="steer", vectors={layer: v_t},
        strength=strength, layer_mean_norms={layer: mean_norm},
    ) as session:
        with ActivationRecorder(model.layers, at=[layer]) as recorder:
            model.forward(ids)
            steered = recorder.activations[layer].detach().clone()
        session.assert_fires(1)
    delta = steered[0, position] - clean[0, position]
    expected = strength * mean_norm * (v_t / v_t.norm())
    assert torch.allclose(delta, expected.to(delta.dtype), rtol=1e-4, atol=1e-5)
