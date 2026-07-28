# Is the source->target map LOCALLY LINEAR at all? (H7, decisive step)
#
# The faithfulness measurement showed the fitted J predicts the true
# response only ~50% accurately in the paper's band, even on OLMo. Two
# very different causes are consistent with that, and they have opposite
# implications:
#
#   (A) ESTIMATION. The map IS locally linear, but jlens's J is a poor
#       estimate of it — because J is averaged over positions and over
#       fitting prompts (`grad.mean(dim=1)`, then merged across prompts).
#       => a methodological problem, fixable by per-position / per-context
#          Jacobians. The campaign's H7 "mean-J mismatch".
#
#   (B) NONLINEARITY. The map is genuinely nonlinear at these depths, so
#       NO Jacobian — however well estimated — models it.
#       => a fact about transformers; the method's premise is limited and
#          no fitting recipe rescues it.
#
# These are separable WITHOUT fitting anything, which is the point of
# this script. Linearity is a property of the model, testable by
# superposition: perturb by eps*delta and by 2*eps*delta and compare the
# responses. If the map is locally linear then
#     r(2*eps) = 2 * r(eps)   =>  cos(r1, r2) = 1 and ||r2||/||r1|| = 2.
# Deviation from that is nonlinearity, measured with no reference to any
# fitted J. We additionally test additivity on two independent
# directions: r(a+b) vs r(a) + r(b).
#
# Read the result against the faithfulness numbers:
#   linear here + unfaithful J  -> cause (A), estimation
#   nonlinear here              -> cause (B), and (A) cannot be assessed
#
# Tier: pilot. Cheap: ~6 forwards per (prompt, layer).
# Usage: python -m jspace_part2.experiments.local_linearity
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
    "gemma4-31b": {"model": "/content/models/gemma4-31b-it",
                   "layers": [22, 30, 37, 40, 42, 44, 48, 52], "n": 60},
    "olmo3-think": {"model": "/content/models/olmo3-think",
                    "layers": [4, 16, 24, 32, 40, 48, 56, 60], "n": 64},
}
PROMPTS = [
    "The history of astronomy begins with the earliest civilizations "
    "that tracked the motions of the sun and moon across the night sky "
    "and recorded what they saw in tables of numbers",
    "In modern computing, the distinction between memory and storage has "
    "shaped how programs are written, because volatile memory loses its "
    "contents whenever the machine loses power",
    "When a large star exhausts its nuclear fuel, the core collapses and "
    "the outer layers rebound outward in an explosion that briefly "
    "outshines an entire galaxy of ordinary stars",
]
EPS = 0.02                  # relative to mean ||h|| at the source layer
N_DIRECTIONS = 3
SKIP_FIRST = 16
LINEAR_COS, LINEAR_RATIO_TOL = 0.95, 0.25   # |ratio-2| <= tol
SEED = 4242


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    slug = arg("--model", "olmo3-think")
    cfg = CFG[slug]
    out = RUN_DIR_P2 / "metrics" / slug / "local_linearity.json"
    if out.exists() and "--force" not in sys.argv:
        print("exists; skipping")
        return
    import transformers
    import jlens
    from jlens.fitting import valid_position_mask
    from jlens.hooks import ActivationRecorder

    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(cfg["model"])
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16).to("cuda").eval()
    model = jlens.from_hf(hf, tok)
    target = model.n_layers - 1
    layers = [l for l in cfg["layers"] if l < target]
    rng = np.random.default_rng(SEED)
    recs = []

    def response(ids, L, delta, mask, base_sum):
        def hook(mod, inp, o, _d=delta, _m=mask):
            h = o[0] if not torch.is_tensor(o) else o
            h = h.clone()
            h[0, _m] = h[0, _m] + _d.to(h.dtype)
            return h if torch.is_tensor(o) else (h, *o[1:])
        hdl = model.layers[L].register_forward_hook(hook)
        with ActivationRecorder(model.layers, at=[target]) as r:
            with torch.no_grad():
                model.forward(ids)
        hdl.remove()
        return r.activations[target][0][mask].float().sum(dim=0) - base_sum

    for pi, text in enumerate(PROMPTS):
        ids = model.encode(text, max_length=96)
        mask = valid_position_mask(ids.shape[1], skip_first=SKIP_FIRST).to("cuda")
        with ActivationRecorder(model.layers, at=layers + [target]) as r0:
            with torch.no_grad():
                model.forward(ids)
        base_sum = r0.activations[target][0][mask].float().sum(dim=0)

        for L in layers:
            h_src = r0.activations[L][0][mask].float()
            hn = float(h_src.norm(dim=-1).mean())
            d = h_src.shape[-1]
            for k in range(N_DIRECTIONS):
                g1 = torch.Generator().manual_seed(int(rng.integers(0, 2**31)))
                g2 = torch.Generator().manual_seed(int(rng.integers(0, 2**31)))
                a = torch.randn(d, generator=g1)
                a = (a / a.norm()).to("cuda", torch.float32) * (EPS * hn)
                b = torch.randn(d, generator=g2)
                b = (b / b.norm()).to("cuda", torch.float32) * (EPS * hn)

                r1 = response(ids, L, a, mask, base_sum)
                r2 = response(ids, L, 2 * a, mask, base_sum)
                rb = response(ids, L, b, mask, base_sum)
                rab = response(ids, L, a + b, mask, base_sum)

                n1 = float(r1.norm())
                scale_cos = float(torch.nn.functional.cosine_similarity(
                    r1, r2, dim=0)) if n1 > 0 else 0.0
                scale_ratio = (float(r2.norm()) / n1) if n1 > 0 else None
                add = r1 + rb
                add_cos = float(torch.nn.functional.cosine_similarity(
                    rab, add, dim=0)) if float(add.norm()) > 0 else 0.0
                add_err = (float((rab - add).norm() / rab.norm())
                           if float(rab.norm()) > 0 else None)
                recs.append({"prompt": pi, "layer": L, "dir": k,
                             "scale_cos": scale_cos,
                             "scale_ratio": scale_ratio,
                             "add_cos": add_cos, "add_rel_err": add_err})
        print(f"  prompt {pi} done ({time.time()-t0:.0f}s)", flush=True)

    by_layer = {}
    for L in layers:
        sub = [r for r in recs if r["layer"] == L]
        sc = float(np.median([r["scale_cos"] for r in sub]))
        sr = float(np.median([r["scale_ratio"] for r in sub]))
        ac = float(np.median([r["add_cos"] for r in sub]))
        ae = float(np.median([r["add_rel_err"] for r in sub]))
        linear = (sc >= LINEAR_COS and abs(sr - 2.0) <= LINEAR_RATIO_TOL
                  and ac >= LINEAR_COS)
        by_layer[L] = {"scale_cos": round(sc, 4), "scale_ratio": round(sr, 4),
                       "additivity_cos": round(ac, 4),
                       "additivity_rel_err": round(ae, 4),
                       "locally_linear": bool(linear)}
    lin = [L for L in layers if by_layer[L]["locally_linear"]]
    nonlin = [L for L in layers if not by_layer[L]["locally_linear"]]
    summ = {
        "model": slug, "eps_relative": EPS, "n_directions": N_DIRECTIONS,
        "criterion": {"scale_cos_min": LINEAR_COS,
                      "scale_ratio_target": 2.0,
                      "scale_ratio_tol": LINEAR_RATIO_TOL,
                      "additivity_cos_min": LINEAR_COS},
        "by_layer": by_layer,
        "locally_linear_layers": lin, "nonlinear_layers": nonlin,
        "reading": (
            f"{slug}: superposition test with NO reference to any fitted J. "
            f"Locally linear at {lin}; nonlinear at {nonlin}. "
            + ("Where the map IS locally linear but the fitted J predicts "
               "poorly, the fault is ESTIMATION (position/prompt averaging) "
               "— H7 mean-J mismatch, a fixable methodological problem. "
               if lin else "")
            + ("Where the map is NONLINEAR, no Jacobian models it however "
               "well estimated, and the method's premise is limited at "
               "that depth." if nonlin else "")),
        "seconds": round(time.time() - t0)}
    prov = Provenance(
        evidence_id=f"local-linearity-{slug}-v1", tier="pilot",
        command=("python -m jspace_part2.experiments.local_linearity "
                 f"--model {slug}"),
        model=resolve_model(cfg["model"]), seed=SEED)
    write_result({"summary": summ, "records": recs}, out, prov)
    registry_append({
        "evidence_id": f"local-linearity-{slug}-v1", "tier": "pilot",
        "what": (f"Local-linearity (superposition) test, no fitted J "
                 f"involved ({slug}): {summ['reading']}"),
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(out), "sha256": sha256_file(out)}]})
    print(json.dumps({k: v for k, v in summ.items() if k != "records"},
                     indent=2))


if __name__ == "__main__":
    main()
