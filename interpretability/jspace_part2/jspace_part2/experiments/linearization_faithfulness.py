# Is the JACOBIAN a good approximation at all? A first-order
# faithfulness check, run on Gemma and OLMo with identical code.
#
# The A3 verdict left one question open. Gemma's fitted lens is
# IDENTIFIED (independent corpora recover the same map, cos 0.97-1.00)
# yet reads WORSE than the trivial logit lens at every band layer, and
# the readout control proved that comparison can detect a J advantage
# when one exists (OLMo: 1.73 @L24). "Identified but useless" has a
# natural explanation that nobody has tested: a Jacobian is a LOCAL
# LINEARIZATION, and if the source->target map is strongly nonlinear
# around the operating point, the linearization is simply a bad model of
# it — reproducibly so, since every corpus would recover the same bad
# linear fit.
#
# Test (directional derivative, the definition of what J claims to be):
# for a held-out prompt, take the residual h at source layer l, perturb
# it by eps*u for a random unit u, run the model forward from l, and
# compare the ACTUAL change in the target-layer residual against the
# Jacobian's PREDICTION J @ (eps*u).
#
#   cos(actual, predicted)  -> is the direction right?
#   ||actual - predicted|| / ||actual||  -> is the magnitude right?
#
# A faithful linearization gives cos -> 1 and small relative error at
# small eps. Reported per layer, per model. This distinguishes:
#   * Gemma mid-band UNFAITHFUL while OLMo faithful -> the architecture
#     is nonlinear where the paper looks; the method's premise fails
#     there, and NO fitting recipe fixes it.
#   * both faithful -> the Jacobian is fine and Gemma's readout failure
#     is about the output basis, not the transport.
#   * both unfaithful -> the method rests on a weaker footing than
#     assumed on BOTH models, which would be a finding about the paper.
#
# Tier: pilot. Cheap: a few forwards per (prompt, layer, eps).
# Usage: python -m jspace_part2.experiments.linearization_faithfulness
#          --model gemma4-31b|olmo3-think [--allow-dirty]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ..lib import sha256_file
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          resolve_model, write_result)

RUN_DIR_P2 = Path("/content/drive/MyDrive/interpret/special-lab-1/"
                  "part2_20260727")
CFG = {
    "gemma4-31b": {
        "model": "/content/models/gemma4-31b-it",
        "lens": str(RUN_DIR_P2 / "lens" / "gemma431_lens.pt"),
        "d": 5376},
    "olmo3-think": {
        "model": "/content/models/olmo3-think",
        "lens": ("/content/drive/MyDrive/interpret/special-lab-1/"
                 "2026-07-25_1726/lens/olmo32bthink_lens.pt"),
        "d": 5120},
}
PROMPTS = [
    "The history of astronomy begins with the earliest civilizations "
    "that tracked the motions of the sun and moon across the sky",
    "In modern computing, the distinction between memory and storage has "
    "shaped how programs are written, because volatile memory",
    "The capital of France is Paris, a city whose architecture reflects "
    "centuries of political and artistic change",
    "When a large star exhausts its nuclear fuel, the core collapses and "
    "the outer layers rebound in an explosion",
]
EPSILONS = [0.01, 0.05]      # relative to ||h|| at the source layer
N_DIRECTIONS = 4
SEED = 4242


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    slug = arg("--model", "gemma4-31b")
    cfg = CFG[slug]
    out = RUN_DIR_P2 / "metrics" / slug / "linearization_faithfulness.json"
    if out.exists() and "--force" not in sys.argv:
        print("exists; skipping")
        return
    import transformers
    import jlens
    from jlens import JacobianLens
    from jlens.hooks import ActivationRecorder

    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(cfg["model"])
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16).to("cuda").eval()
    model = jlens.from_hf(hf, tok)
    lens = JacobianLens.load(cfg["lens"])
    layers = sorted(lens.jacobians.keys())
    target = model.n_layers - 1
    rng = np.random.default_rng(SEED)
    recs = []

    for pi, text in enumerate(PROMPTS):
        ids = model.encode(text, max_length=96)
        pos = ids.shape[1] - 1
        for L in layers:
            J = lens.jacobians[L].to("cuda", torch.float32)
            with ActivationRecorder(model.layers, at=[L, target]) as r:
                with torch.no_grad():
                    model.forward(ids)
            h_src = r.activations[L][0, pos].float()
            h_tgt0 = r.activations[target][0, pos].float()
            hn = float(h_src.norm())
            for eps_rel in EPSILONS:
                for k in range(N_DIRECTIONS):
                    g = torch.Generator(device="cpu").manual_seed(
                        int(rng.integers(0, 2**31)))
                    u = torch.randn(h_src.shape[0], generator=g)
                    u = (u / u.norm()).to("cuda", torch.float32)
                    delta = (eps_rel * hn) * u

                    # actual: re-run with h_L perturbed
                    def hook(mod, inp, o, _L=L, _d=delta, _p=pos):
                        h = o[0] if not torch.is_tensor(o) else o
                        h = h.clone()
                        h[0, _p] = h[0, _p] + _d.to(h.dtype)
                        return h if torch.is_tensor(o) else (h, *o[1:])

                    hdl = model.layers[L].register_forward_hook(hook)
                    with ActivationRecorder(model.layers, at=[target]) as r2:
                        with torch.no_grad():
                            model.forward(ids)
                    hdl.remove()
                    actual = (r2.activations[target][0, pos].float() - h_tgt0)
                    predicted = (J @ delta)

                    an, pn = float(actual.norm()), float(predicted.norm())
                    cos = float(torch.nn.functional.cosine_similarity(
                        actual, predicted, dim=0)) if an > 0 and pn > 0 else 0.0
                    rel = float((actual - predicted).norm() / actual.norm()) \
                        if an > 0 else float("nan")
                    recs.append({"prompt": pi, "layer": L, "eps": eps_rel,
                                 "dir": k, "cos": cos, "rel_err": rel,
                                 "norm_ratio": (pn / an) if an > 0 else None})
            del J
            torch.cuda.empty_cache()
        print(f"  prompt {pi} done ({time.time()-t0:.0f}s)", flush=True)

    by_layer = {}
    for L in layers:
        sub = [r for r in recs if r["layer"] == L]
        by_layer[L] = {
            "median_cos": round(float(np.median([r["cos"] for r in sub])), 4),
            "median_rel_err": round(
                float(np.median([r["rel_err"] for r in sub])), 4),
            "median_norm_ratio": round(
                float(np.median([r["norm_ratio"] for r in sub
                                 if r["norm_ratio"] is not None])), 4),
            "n": len(sub)}
    faithful = [L for L in layers if by_layer[L]["median_cos"] >= 0.9]
    unfaithful = [L for L in layers if by_layer[L]["median_cos"] < 0.5]
    summ = {
        "model": slug, "target_layer": target, "layers": layers,
        "epsilons_relative": EPSILONS, "n_directions": N_DIRECTIONS,
        "by_layer": by_layer,
        "faithful_layers_cos_ge_0.9": faithful,
        "unfaithful_layers_cos_lt_0.5": unfaithful,
        "reading": (
            f"Directional-derivative check on {slug}: the fitted Jacobian "
            f"predicts the true target-layer response with median cosine "
            f"{{layer: cos}} = "
            f"{ {L: by_layer[L]['median_cos'] for L in layers} }. "
            f"Faithful (cos>=0.9): {faithful}. Unfaithful (cos<0.5): "
            f"{unfaithful}. A Jacobian is a LOCAL LINEARIZATION; where it "
            f"is unfaithful, no fitting recipe rescues it and the method's "
            f"premise does not hold at that depth."),
        "seconds": round(time.time() - t0)}
    prov = Provenance(
        evidence_id=f"linearization-faithfulness-{slug}-v1", tier="pilot",
        command=("python -m jspace_part2.experiments."
                 f"linearization_faithfulness --model {slug}"),
        inputs={"lens": sha256_file(cfg["lens"])},
        model=resolve_model(cfg["model"]), seed=SEED)
    write_result({"summary": summ, "records": recs}, out, prov)
    registry_append({
        "evidence_id": f"linearization-faithfulness-{slug}-v1", "tier": "pilot",
        "what": f"Jacobian first-order faithfulness ({slug}): {summ['reading']}",
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(out), "sha256": sha256_file(out)}]})
    print(json.dumps({k: v for k, v in summ.items() if k != "records"},
                     indent=2))


if __name__ == "__main__":
    main()
