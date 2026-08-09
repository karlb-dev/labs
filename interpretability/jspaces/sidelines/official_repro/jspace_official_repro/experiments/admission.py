"""Lane admission: identity, lens post-load checks, model-backed
conformance (plan §7.1 / §9.1; contract hard stops)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from jlens.lens import JacobianLens

from ..adapters import admission_facts, load_olmo, load_qwen
from ..layers import PAPER_BAND, PAPER_GRID_SOURCES
from ..manifests import file_sha256, json_sha256, runtime_fingerprint, write_json
from ..paths import (
    DRIVE_ROOT,
    QWEN_LENS_FILENAME,
    QWEN_LENS_REPO,
    QWEN_LENS_REVISION,
    QWEN_LENS_SHA256,
    model_snapshot,
)
from ..readout import g_folding_audit, readout_parity, token_vectors
from ..rendering import preferred_token, render_raw
from ..scoring import forward_logits, rank_of

#: Ten fixed readout sentinels (first = upstream boot/currency probe).
SENTINELS = [
    "Fact: The currency used in the country shaped like a boot is",
    "Fact: The capital of Japan is Tokyo.\nFact: The currency used in the country shaped like a boot is",
    "The chemical symbol for gold is",
    "The author of Romeo and Juliet is William",
    "Twinkle, twinkle, little star, how I wonder what you",
    "The opposite of hot is",
    "2 + 2 = 4. 3 + 3 =",
    "The largest planet in the solar system is",
    "Paris is to France as Rome is to",
    "Der Gegenteil von heiß ist",
]

#: Battery tokens for the g-folding audit (union of common core targets).
G_AUDIT_WORDS = [
    "France", "Canada", "China", "Egypt", "Paris", "Ottawa", "Beijing",
    "Cairo", "Brazil", "Mexico", "lion", "eagle", "shark", "spider",
    "February", "April", "July", "October", "three", "five", "seven",
    "nine", "red", "blue", "piano", "violin",
]


def load_qwen_lens() -> JacobianLens:
    path = model_snapshot(QWEN_LENS_REPO, QWEN_LENS_REVISION) / QWEN_LENS_FILENAME
    if file_sha256(path) != QWEN_LENS_SHA256:
        raise RuntimeError("published lens hash mismatch at load")
    lens = JacobianLens.load(str(path))
    if lens.d_model != 5120 or lens.n_prompts != 1000:
        raise RuntimeError(f"lens identity mismatch: {lens!r}")
    if lens.source_layers != list(range(63)):
        raise RuntimeError("lens source_layers != [0..62]")
    for layer, J in lens.jacobians.items():
        if J.shape != (5120, 5120) or not torch.isfinite(J).all():
            raise RuntimeError(f"lens layer {layer} shape/finite failure")
    return lens


def _sentinel_readouts(model, lens, *, layers: list[int]) -> list[dict]:
    rows = []
    for prompt in SENTINELS:
        rendered = render_raw(model, prompt)
        position = rendered.final_position
        from jlens.hooks import ActivationRecorder

        with ActivationRecorder(model.layers, at=layers) as recorder:
            model.forward(rendered.input_ids)
            residuals = {
                layer: recorder.activations[layer][0, position].detach()
                for layer in layers
            }
        per_layer = {}
        for layer in layers:
            transported = lens.transport(residuals[layer].float().unsqueeze(0), layer)[0]
            logits = model.unembed(transported).float().cpu()
            top = logits.topk(5).indices.tolist()
            per_layer[layer] = [model.tokenizer.decode([t]) for t in top]
        rows.append({"prompt": prompt, "position": position,
                     "top5_per_layer": per_layer})
    return rows


def _noop_check(model, lens, prompt: str) -> dict:
    """alpha=0 intervention forward must equal a clean forward exactly."""
    from ..interventions import HookPlan, InterventionSession

    rendered = render_raw(model, prompt)
    positions = list(range(rendered.seq_len))
    clean, _ = forward_logits(model, rendered.input_ids,
                              positions=[rendered.final_position])
    layer = PAPER_BAND[0]
    vectors = token_vectors(lens, model, layer,
                            [preferred_token(model.tokenizer, "France"),
                             preferred_token(model.tokenizer, "Canada")])
    plan = HookPlan(layers=[layer], positions=positions)
    with InterventionSession(
        model.layers, plan, kind="swap",
        vectors={layer: (vectors[0], vectors[1])}, alpha=0.0,
    ) as session:
        patched, _ = forward_logits(model, rendered.input_ids,
                                    positions=[rendered.final_position])
        session.assert_fires(1)
    max_diff = float((clean - patched).abs().max())
    return {"max_abs_logit_diff": max_diff, "ok": max_diff == 0.0}


def _repeatability_check(model, lens) -> dict:
    rows = []
    for _ in range(2):
        rendered = render_raw(model, SENTINELS[0])
        logits, _ = forward_logits(model, rendered.input_ids,
                                   positions=[rendered.final_position])
        rows.append(logits)
    identical = bool(torch.equal(rows[0], rows[1]))
    return {"identical_logits": identical,
            "max_abs_diff": float((rows[0] - rows[1]).abs().max())}


def run_admission(lane: str, *, out_dir: Path | None = None) -> dict:
    if lane == "qwen":
        model, hf_model, tokenizer = load_qwen()
        lens = load_qwen_lens()
    else:
        model, hf_model, tokenizer = load_olmo()
        lens = None  # OLMo lens admission happens post-fit via same helpers
    return run_admission_with(model, hf_model, tokenizer, lens, lane=lane,
                              out_dir=out_dir)


def run_admission_with(
    model, hf_model, tokenizer, lens, *, lane: str, out_dir: Path | None = None
) -> dict:
    assert lane in ("qwen", "olmo")
    out_dir = out_dir or (DRIVE_ROOT / f"{lane}_admission")
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    facts = admission_facts(model, hf_model, tokenizer)
    result: dict = {"lane": lane, "facts": facts,
                    "runtime": runtime_fingerprint()}
    if lens is not None:
        parity = readout_parity(
            model, lens, SENTINELS[0],
            layers=[3, PAPER_BAND[0], PAPER_BAND[-1], 60],
            positions=[-2, -1],
        )
        if not parity["ok"]:
            raise RuntimeError(f"HARD STOP readout parity: {parity}")
        result["readout_parity"] = parity
        audit_ids = [preferred_token(tokenizer, w) for w in G_AUDIT_WORDS]
        audit_ids = [t for t in audit_ids if t is not None]
        result["g_folding"] = g_folding_audit(
            lens, model, token_ids=audit_ids, layers=PAPER_BAND
        )
        result["noop"] = _noop_check(model, lens, SENTINELS[0])
        if not result["noop"]["ok"]:
            raise RuntimeError(f"HARD STOP alpha-0 no-op: {result['noop']}")
        result["repeatability"] = _repeatability_check(model, lens)
        result["sentinels"] = _sentinel_readouts(
            model, lens, layers=[PAPER_BAND[0], PAPER_BAND[6], PAPER_BAND[-1]]
        )
    result["wall_seconds"] = time.time() - start
    sha = write_json(out_dir / f"{lane}_admission.json", result)
    result["_output"] = str(out_dir / f"{lane}_admission.json")
    result["_sha256"] = sha
    return result
