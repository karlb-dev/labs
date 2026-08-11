"""Geometry + instrument admission for Muse (pre- and post-fit)."""
from __future__ import annotations

import time
from pathlib import Path

import torch

from ..adapters import admission_facts, load_muse
from ..paths import (
    DEPTH_GRID,
    DRIVE_ROOT,
    EXPECTED_D_MODEL,
    EXPECTED_N_LAYERS,
    FINAL_LAYER,
    FIT_SOURCE_LAYERS,
    PAPER_BAND,
    ensure_dirs,
)
from ..readout import (
    g_folding_audit,
    identity_lens,
    lens_to_device,
    preferred_token,
    rank_of,
    readout_parity,
)
from ..registry import register
from ..util import atomic_write_json, log, runtime_fingerprint, utc_now

SENTINELS = [
    "Fact: The currency used in the country shaped like a boot is",
    "The chemical symbol for gold is",
    "The author of Romeo and Juliet is William",
    "Twinkle, twinkle, little star, how I wonder what you",
    "The opposite of hot is",
    "2 + 2 = 4. 3 + 3 =",
    "The largest planet in the solar system is",
    "Paris is to France as Rome is to",
    "Der Gegenteil von heiß ist",
    "The capital of Japan is",
]

G_AUDIT_WORDS = [
    "France", "Canada", "China", "Egypt", "Paris", "Ottawa", "Beijing",
    "Cairo", "Brazil", "Mexico", "lion", "eagle", "shark", "spider",
    "February", "April", "July", "October", "three", "five", "seven",
    "nine", "red", "blue", "piano", "violin",
]


def smoke_forward(model) -> dict:
    """Cheap forward + hook test at a few layers."""
    from jlens.hooks import ActivationRecorder

    prompt = "The capital of France is"
    ids = model.encode(prompt, max_length=64)
    layers = [0, FINAL_LAYER // 2, FINAL_LAYER]
    t0 = time.perf_counter()
    with ActivationRecorder(model.layers, at=layers) as rec:
        out = model.forward(ids)
        acts = {L: rec.activations[L].detach() for L in layers}
    wall = time.perf_counter() - t0
    # final residual -> unembed top tokens
    h = acts[FINAL_LAYER][0, -1].float()
    logits = model.unembed(h.unsqueeze(0))[0].float().cpu()
    top = logits.topk(5)
    toks = [model.tokenizer.decode([i]) for i in top.indices.tolist()]
    peak = torch.cuda.max_memory_allocated() / 1e9
    return {
        "prompt": prompt,
        "seq_len": int(ids.shape[1]),
        "wall_seconds": wall,
        "peak_vram_gb": round(peak, 2),
        "activation_shapes": {str(L): list(acts[L].shape) for L in layers},
        "final_topk": list(zip(toks, [float(v) for v in top.values])),
        "finite_acts": all(torch.isfinite(a).all().item() for a in acts.values()),
    }


def identity_geometry(model) -> dict:
    """Pre-fit geometry: identity J at final layer must match unembed(h)."""
    layers = [FINAL_LAYER]
    lens = identity_lens(model.d_model, layers)
    lens_to_device(lens, "cuda:0")
    prompt = SENTINELS[0]
    ids = model.encode(prompt, max_length=128)
    pos = int(ids.shape[1] - 1)
    parity = readout_parity(model, lens, prompt, layers=layers, positions=[pos])
    # Also check that logit softcap path is live
    return {
        "identity_parity": parity,
        "logit_softcap": model._logit_softcap,
        "position": pos,
        "prompt": prompt,
    }


def boot_sentinel_identity(model) -> dict:
    """Boot probe with identity transport (logit-lens equivalent)."""
    from jlens.hooks import ActivationRecorder

    prompt = SENTINELS[0]
    ids = model.encode(prompt, max_length=128)
    pos = int(ids.shape[1] - 1)
    layers = DEPTH_GRID
    with ActivationRecorder(model.layers, at=layers) as rec:
        model.forward(ids)
        acts = {L: rec.activations[L][0, pos].detach() for L in layers}
    rows = []
    for L in layers:
        logits = model.unembed(acts[L].float().unsqueeze(0))[0].float().cpu()
        top = logits.topk(5)
        rows.append({
            "layer": L,
            "top5": [model.tokenizer.decode([i]) for i in top.indices.tolist()],
            "top5_ids": top.indices.tolist(),
        })
    return {"prompt": prompt, "position": pos, "per_layer": rows}


@torch.no_grad()
def local_linearity_smoke(model, *, layer: int | None = None, eps: float = 1e-2) -> dict:
    """Cheap finite-diff canary: is residual update approximately linear in a
    random direction at one mid-layer? (Gemma-style geometry canary, not a JVP.)
    """
    from jlens.hooks import ActivationRecorder

    layer = FINAL_LAYER // 2 if layer is None else layer
    prompt = "The capital of France is Paris."
    ids = model.encode(prompt, max_length=64)
    pos = int(ids.shape[1] - 1)

    # Capture clean residual at layer and at final
    with ActivationRecorder(model.layers, at=[layer, FINAL_LAYER]) as rec:
        model.forward(ids)
        h0 = rec.activations[layer][0, pos].detach().float().clone()
        f0 = rec.activations[FINAL_LAYER][0, pos].detach().float().clone()

    # Random unit direction in residual space
    g = torch.randn_like(h0)
    g = g / g.norm()
    deltas = []
    for sign in (+1.0, -1.0):
        # Patch residual at `layer` then continue — use a forward hook
        handle_holder = {}

        def make_hook(delta_vec):
            def hook(module, inp, out):
                # out may be tensor or tuple
                if isinstance(out, tuple):
                    h = out[0]
                    h = h.clone()
                    h[0, pos] = h[0, pos].float() + delta_vec.to(h.device).to(h.dtype)
                    return (h,) + out[1:]
                h = out.clone()
                h[0, pos] = h[0, pos].float() + delta_vec.to(h.device).to(h.dtype)
                return h
            return hook

        handle = model.layers[layer].register_forward_hook(make_hook(sign * eps * g))
        try:
            with ActivationRecorder(model.layers, at=[FINAL_LAYER]) as rec:
                model.forward(ids)
                f = rec.activations[FINAL_LAYER][0, pos].detach().float()
        finally:
            handle.remove()
        deltas.append(f - f0)

    # Central difference approx of directional derivative
    jvp_fd = (deltas[0] - deltas[1]) / (2 * eps)
    # Homogeneity: +eps response should be ~ -(-eps) response
    odd_err = float((deltas[0] + deltas[1]).norm() / (deltas[0].norm() + 1e-8))
    # Response magnitude
    resp = float(deltas[0].norm() / eps)
    return {
        "layer": layer,
        "eps": eps,
        "response_norm_per_eps": resp,
        "odd_symmetry_rel_err": odd_err,
        "jvp_fd_norm": float(jvp_fd.norm()),
        "finite": bool(torch.isfinite(jvp_fd).all()),
        "note": "canary only; not an exact JVP / not Gemma gate thresholds",
    }


def run_pre_fit(out_dir: Path | None = None) -> dict:
    """Load Muse, run pre-fit admission (no fitted lens required)."""
    ensure_dirs()
    out_dir = out_dir or (DRIVE_ROOT / "metrics")
    out_path = out_dir / "admission_pre_fit.json"
    if out_path.exists():
        log(f"exists {out_path}; loading")
        import json
        return json.loads(out_path.read_text())

    torch.cuda.reset_peak_memory_stats()
    model, hf_model, tokenizer = load_muse()
    facts = admission_facts(model, hf_model, tokenizer)
    log(f"facts: {facts['n_layers']}x{facts['d_model']} softcap={facts['logit_softcap']}")

    smoke = smoke_forward(model)
    log(f"smoke top5={smoke['final_topk']} vram={smoke['peak_vram_gb']}GB")
    if not smoke["finite_acts"]:
        raise RuntimeError("HARD STOP: non-finite activations")

    geom = identity_geometry(model)
    log(f"identity parity ok={geom['identity_parity'].get('ok')} "
        f"diff={geom['identity_parity'].get('max_abs_diff')}")
    if not geom["identity_parity"].get("ok"):
        raise RuntimeError(f"HARD STOP identity parity: {geom['identity_parity']}")

    boot = boot_sentinel_identity(model)
    # Linearity canary at two eps scales (soft diagnostic, not a hard gate —
    # Muse has gated/local-global attention; large-eps odd-symmetry can fail
    # without breaking identity readout geometry).
    lin = local_linearity_smoke(model, eps=1e-2)
    lin_small = local_linearity_smoke(model, eps=1e-3)
    log(f"linearity smoke eps=1e-2 odd_err={lin['odd_symmetry_rel_err']:.4f} "
        f"resp={lin['response_norm_per_eps']:.4f}; "
        f"eps=1e-3 odd_err={lin_small['odd_symmetry_rel_err']:.4f}")

    hard_gates = {
        "shape_ok": facts["n_layers"] == EXPECTED_N_LAYERS
        and facts["d_model"] == EXPECTED_D_MODEL,
        "cuda_ok": facts["device"].startswith("cuda"),
        "finite_acts": smoke["finite_acts"],
        "identity_parity_ok": geom["identity_parity"]["ok"],
        "smoke_top1_sensible": (
            smoke["final_topk"]
            and "Paris" in smoke["final_topk"][0][0]
        ),
    }
    soft_gates = {
        "linearity_finite": lin["finite"] and lin_small["finite"],
        "linearity_odd_symmetry_ok_eps1e2": lin["odd_symmetry_rel_err"] < 0.25,
        "linearity_odd_symmetry_ok_eps1e3": lin_small["odd_symmetry_rel_err"] < 0.25,
    }
    result = {
        "evidence_id": "muse-admission-prefit-v1",
        "utc": utc_now(),
        "facts": facts,
        "smoke_forward": smoke,
        "identity_geometry": geom,
        "boot_sentinel_logit_lens": boot,
        "local_linearity_smoke": lin,
        "local_linearity_smoke_eps1e3": lin_small,
        "gates": {
            **hard_gates,
            **soft_gates,
            "hard_all_pass": all(hard_gates.values()),
            # all_pass = hard only; soft canaries recorded separately
            "all_pass": all(hard_gates.values()),
        },
        "runtime": runtime_fingerprint(),
        "fit_source_layers": FIT_SOURCE_LAYERS,
        "paper_band": PAPER_BAND,
        "final_layer": FINAL_LAYER,
        "note": (
            "Hard gates: shape/cuda/finite/identity-parity/smoke-top1. "
            "Local-linearity odd-symmetry is a soft canary only."
        ),
    }
    atomic_write_json(result, out_path)
    register({
        "evidence_id": "muse-admission-prefit-v1",
        "what": "Pre-fit Muse geometry admission (shape, hooks, identity parity, linearity canary)",
        "command": "python -m jspace_muse.experiments.admission",
        "outputs": [out_path],
        "gates": result["gates"],
    })
    log(f"pre-fit admission all_pass={result['gates']['all_pass']} -> {out_path}")
    return result


def run_post_fit(model, lens, *, out_dir: Path | None = None) -> dict:
    """Post-fit admission: readout parity + g-fold on the fitted lens."""
    ensure_dirs()
    out_dir = out_dir or (DRIVE_ROOT / "metrics")
    out_path = out_dir / "admission_post_fit.json"
    if out_path.exists():
        import json
        return json.loads(out_path.read_text())

    lens_layers = sorted(lens.jacobians.keys())
    check_layers = [L for L in PAPER_BAND if L in lens.jacobians][:4]
    if not check_layers:
        check_layers = lens_layers[-4:]
    lens_to_device(lens, "cuda:0", layers=check_layers)

    prompt = SENTINELS[0]
    ids = model.encode(prompt, max_length=128)
    pos = int(ids.shape[1] - 1)
    parity = readout_parity(
        model, lens, prompt, layers=check_layers, positions=[pos]
    )
    if not parity.get("ok"):
        raise RuntimeError(f"HARD STOP post-fit readout parity: {parity}")

    tok_ids = []
    for w in G_AUDIT_WORDS:
        t = preferred_token(model.tokenizer, w)
        if t is not None:
            tok_ids.append(t)
    gfold = g_folding_audit(lens, model, token_ids=tok_ids, layers=check_layers)

    # Boot sentinel with fitted lens
    from jlens.hooks import ActivationRecorder

    layers = [L for L in DEPTH_GRID if L in lens.jacobians] or check_layers
    lens_to_device(lens, "cuda:0", layers=layers)
    with ActivationRecorder(model.layers, at=layers) as rec:
        model.forward(ids)
        acts = {L: rec.activations[L][0, pos].detach() for L in layers}
    per_layer = []
    for L in layers:
        transported = lens.transport(acts[L].float().unsqueeze(0), L)[0]
        logits = model.unembed(transported).float().cpu()
        top = logits.topk(5)
        per_layer.append({
            "layer": L,
            "top5": [model.tokenizer.decode([i]) for i in top.indices.tolist()],
        })

    result = {
        "evidence_id": "muse-admission-postfit-v1",
        "utc": utc_now(),
        "readout_parity": parity,
        "g_folding": gfold,
        "boot_sentinel_jlens": {"prompt": prompt, "per_layer": per_layer},
        "check_layers": check_layers,
        "runtime": runtime_fingerprint(),
    }
    atomic_write_json(result, out_path)
    register({
        "evidence_id": "muse-admission-postfit-v1",
        "what": "Post-fit lens admission (parity, g-fold, boot trajectory)",
        "command": "python -m jspace_muse.experiments.admission --post-fit",
        "outputs": [out_path],
    })
    return result


if __name__ == "__main__":
    import sys
    if "--post-fit" in sys.argv:
        from jlens.lens import JacobianLens
        from ..paths import DRIVE_ROOT
        from ..adapters import load_muse
        model, _, _ = load_muse()
        lens = JacobianLens.load(str(DRIVE_ROOT / "lens" / "muse_glimmer_lens.pt"))
        print(run_post_fit(model, lens))
    else:
        r = run_pre_fit()
        print("gates", r["gates"])
