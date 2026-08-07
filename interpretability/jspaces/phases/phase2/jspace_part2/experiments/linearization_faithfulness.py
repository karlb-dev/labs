# Is the fitted J a faithful first-order model of what it actually
# claims to model? Run on Gemma and OLMo with identical code.
#
# WHY v2 EXISTS. v1 of this test was WITHDRAWN: it perturbed one source
# position and read one target position, but jlens does not build a
# per-position derivative. Reading fitting.py: the cotangent is set at
# EVERY valid target position, and the resulting gradient is averaged
# over source positions (`grad[...].mean(dim=1)`). So
#
#     J[i, j] = mean_over_s [ sum_over_t  d h_tgt[t, i] / d h_src[s, j] ]
#
# a POSITION-AVERAGED, TARGET-SUMMED object. v1 compared one entry of
# that against a much richer quantity, so its near-zero cosines were an
# artifact of the mismatch, not evidence of nonlinearity. The
# inconsistency that exposed it: at L52 the J-lens reads the answer at
# rank 2 (J plainly carries information) while v1 reported cos 0.045.
#
# THE CORRECT TEST follows from J's own definition. Perturb h_src by the
# SAME delta at every valid position s; then to first order
#
#     Delta( sum_t h_tgt[t] ) = sum_s sum_t (d h_tgt[t]/d h_src[s]) delta
#                             = P * (J @ delta)        [P = #valid positions]
#
# So compare the measured change in the SUMMED target activation against
# P * (J @ delta). That is exactly what the fitted object predicts.
#
# NOISE FLOOR (the control v1 lacked): the model runs in bf16, so two
# identical forwards need not agree bit-for-bit. Before interpreting any
# response, measure the spurious delta from an unperturbed re-run and
# report the signal-to-noise ratio. A response below the noise floor is
# reported as UNRESOLVED, never as unfaithfulness.
#
# Verdict per layer: FAITHFUL cos>=0.9 · PARTIAL 0.5-0.9 · UNFAITHFUL
# <0.5 (only when SNR >= 5) · UNRESOLVED when SNR < 5.
#
# Tier: pilot.
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
        "lens": str(RUN_DIR_P2 / "lens" / "gemma431_lens.pt")},
    "olmo3-think": {
        "model": "/content/models/olmo3-think",
        "lens": ("/content/drive/MyDrive/interpret/special-lab-1/"
                 "2026-07-25_1726/lens/olmo32bthink_lens.pt")},
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
EPSILONS = [0.02, 0.10]      # relative to mean ||h|| at the source layer
N_DIRECTIONS = 3
SKIP_FIRST = 16              # jlens default; matches the fits we test
SNR_MIN = 5.0
SEED = 4242


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    slug = arg("--model", "gemma4-31b")
    cfg = CFG[slug]
    out = RUN_DIR_P2 / "metrics" / slug / "linearization_faithfulness_v2.json"
    if out.exists() and "--force" not in sys.argv:
        print("exists; skipping")
        return
    import transformers
    import jlens
    from jlens import JacobianLens
    from jlens.fitting import valid_position_mask
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
    recs, noise_recs = [], []

    def summed_target(ids, perturb=None):
        """sum over valid target positions of h_target, optionally with a
        constant delta added at every valid source position."""
        handles = []
        if perturb is not None:
            L, delta, mask = perturb

            def hook(mod, inp, o, _d=delta, _m=mask):
                h = o[0] if not torch.is_tensor(o) else o
                h = h.clone()
                h[0, _m] = h[0, _m] + _d.to(h.dtype)
                return h if torch.is_tensor(o) else (h, *o[1:])
            handles.append(model.layers[L].register_forward_hook(hook))
        with ActivationRecorder(model.layers, at=[target]) as r:
            with torch.no_grad():
                model.forward(ids)
        for h in handles:
            h.remove()
        act = r.activations[target][0].float()
        return act

    for pi, text in enumerate(PROMPTS):
        ids = model.encode(text, max_length=96)
        seq = ids.shape[1]
        mask = valid_position_mask(seq, skip_first=SKIP_FIRST).to("cuda")
        P = int(mask.sum())
        base = summed_target(ids)
        base_sum = base[mask].sum(dim=0)

        # noise floor: identical forward, no perturbation
        rep = summed_target(ids)
        noise = float((rep[mask].sum(dim=0) - base_sum).norm())
        noise_recs.append({"prompt": pi, "noise_norm": noise})

        with ActivationRecorder(model.layers, at=layers) as r0:
            with torch.no_grad():
                model.forward(ids)

        for L in layers:
            J = lens.jacobians[L].to("cuda", torch.float32)
            h_src = r0.activations[L][0][mask].float()
            hn = float(h_src.norm(dim=-1).mean())
            for eps_rel in EPSILONS:
                for k in range(N_DIRECTIONS):
                    g = torch.Generator().manual_seed(
                        int(rng.integers(0, 2**31)))
                    u = torch.randn(J.shape[1], generator=g)
                    u = (u / u.norm()).to("cuda", torch.float32)
                    delta = (eps_rel * hn) * u

                    act = summed_target(ids, perturb=(L, delta, mask))
                    actual = act[mask].sum(dim=0) - base_sum
                    predicted = P * (J @ delta)

                    an, pn = float(actual.norm()), float(predicted.norm())
                    snr = an / noise if noise > 0 else float("inf")
                    cos = float(torch.nn.functional.cosine_similarity(
                        actual, predicted, dim=0)) if an > 0 and pn > 0 else 0.0
                    recs.append({
                        "prompt": pi, "layer": L, "eps": eps_rel, "dir": k,
                        "cos": cos, "snr": snr,
                        "rel_err": float((actual - predicted).norm() / an)
                        if an > 0 else None,
                        "norm_ratio": (pn / an) if an > 0 else None})
            del J
            torch.cuda.empty_cache()
        print(f"  prompt {pi} done ({time.time()-t0:.0f}s, "
              f"P={P}, noise={noise:.3f})", flush=True)

    by_layer = {}
    for L in layers:
        sub = [r for r in recs if r["layer"] == L]
        ok = [r for r in sub if r["snr"] >= SNR_MIN]
        med_cos = float(np.median([r["cos"] for r in ok])) if ok else None
        status = ("UNRESOLVED" if med_cos is None else
                  "FAITHFUL" if med_cos >= 0.9 else
                  "PARTIAL" if med_cos >= 0.5 else "UNFAITHFUL")
        by_layer[L] = {
            "median_cos_snr_ok": (round(med_cos, 4) if med_cos is not None
                                  else None),
            "median_snr": round(float(np.median([r["snr"] for r in sub])), 2),
            "n_above_snr": len(ok), "n": len(sub),
            "median_norm_ratio": round(float(np.median(
                [r["norm_ratio"] for r in sub
                 if r["norm_ratio"] is not None])), 3),
            "status": status}

    faithful = [L for L in layers if by_layer[L]["status"] == "FAITHFUL"]
    partial = [L for L in layers if by_layer[L]["status"] == "PARTIAL"]
    unfaith = [L for L in layers if by_layer[L]["status"] == "UNFAITHFUL"]
    unres = [L for L in layers if by_layer[L]["status"] == "UNRESOLVED"]
    summ = {
        "model": slug, "target_layer": target, "layers": layers,
        "estimand": ("Delta(sum_t h_tgt[t]) vs P*(J@delta) under a constant "
                     "delta at every valid source position — matches how "
                     "jlens builds J (position-averaged, target-summed)"),
        "skip_first": SKIP_FIRST, "epsilons_relative": EPSILONS,
        "n_directions": N_DIRECTIONS, "snr_min": SNR_MIN,
        "noise_floor": noise_recs,
        "by_layer": by_layer,
        "faithful": faithful, "partial": partial,
        "unfaithful": unfaith, "unresolved": unres,
        "reading": (
            f"{slug}: median cosine between the measured summed-target "
            f"response and the J-predicted response, over perturbations "
            f"clearing SNR {SNR_MIN} — "
            f"{ {L: by_layer[L]['median_cos_snr_ok'] for L in layers} }. "
            f"FAITHFUL {faithful} · PARTIAL {partial} · UNFAITHFUL "
            f"{unfaith} · UNRESOLVED {unres}."),
        "seconds": round(time.time() - t0)}
    prov = Provenance(
        evidence_id=f"linearization-faithfulness-{slug}-v2", tier="pilot",
        command=("python -m jspace_part2.experiments."
                 f"linearization_faithfulness --model {slug}"),
        inputs={"lens": sha256_file(cfg["lens"])},
        model=resolve_model(cfg["model"]), seed=SEED)
    write_result({"summary": summ, "records": recs}, out, prov)
    registry_append({
        "evidence_id": f"linearization-faithfulness-{slug}-v2", "tier": "pilot",
        "what": (f"Jacobian first-order faithfulness on the estimand jlens "
                 f"actually fits ({slug}): {summ['reading']}"),
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(out), "sha256": sha256_file(out)}]})
    print(json.dumps({k: v for k, v in summ.items() if k != "records"},
                     indent=2))


if __name__ == "__main__":
    main()
