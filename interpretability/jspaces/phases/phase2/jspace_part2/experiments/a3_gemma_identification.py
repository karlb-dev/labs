# A3 — is the Gemma lens IDENTIFIED, layer by layer?
#
# Found while validating the merged fit: the merged L22 Jacobian norm
# (0.042) is SMALLER than any of the four slices that built it (0.124 /
# 0.098 / 0.029 / ...). Averaging shrank it — the signature of slice
# Jacobians that disagree in direction and partially cancel. That is a
# convergence question, and it must be answered BEFORE any readout result
# is interpreted: "the J-lens cannot read layer L" is only meaningful if
# a J was actually identified at layer L.
#
# Measure: pairwise cosine similarity between the four independent
# 30-prompt slice Jacobians, per layer (the same two-independent-fits
# logic as G2/B1, applied per layer rather than per dictionary row).
# High cosine = independent corpora recover the same linear map.
#
# Gate (fixed here): mean pairwise cos >= 0.90 -> IDENTIFIED, usable for
# readout claims; 0.50-0.90 -> PARTIAL, report with the caveat; < 0.50 ->
# NOT IDENTIFIED, the layer carries NO instrument and must be excluded
# from any claim about the model.
#
# Tier: pilot. CPU-only.
# Usage: python -m jspace_part2.experiments.a3_gemma_identification
#          [--allow-dirty]
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import torch

from ..lib import sha256_file
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          write_result)

RUN_DIR_P2 = Path("/content/drive/MyDrive/interpret/special-lab-1/"
                  "part2_20260727")
LENS_DIR = RUN_DIR_P2 / "lens"
OUT = RUN_DIR_P2 / "metrics" / "gemma4-31b" / "a3_identification.json"
N_SLICES = 4
D_MODEL = 5376
IDENT_OK, IDENT_PARTIAL = 0.90, 0.50


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    if OUT.exists() and "--force" not in sys.argv:
        print("exists; skipping")
        return
    from jlens import JacobianLens

    t0 = time.time()
    slices = [JacobianLens.load(str(LENS_DIR / f"gemma431_slice{i}.pt"))
              for i in range(N_SLICES)]
    merged = JacobianLens.load(str(LENS_DIR / "gemma431_lens.pt"))
    layers = sorted(merged.jacobians.keys())

    rows = {}
    for L in layers:
        Js = [s.jacobians[L].float().flatten() for s in slices]
        cs = [float(torch.nn.functional.cosine_similarity(Js[i], Js[j], dim=0))
              for i, j in itertools.combinations(range(N_SLICES), 2)]
        mean_cos = sum(cs) / len(cs)
        slice_norms = [float(s.jacobians[L].float().norm() / D_MODEL**0.5)
                       for s in slices]
        merged_norm = float(merged.jacobians[L].float().norm() / D_MODEL**0.5)
        mean_slice_norm = sum(slice_norms) / len(slice_norms)
        status = ("IDENTIFIED" if mean_cos >= IDENT_OK else
                  "PARTIAL" if mean_cos >= IDENT_PARTIAL else
                  "NOT_IDENTIFIED")
        rows[L] = {
            "mean_pairwise_cos": round(mean_cos, 4),
            "min_pairwise_cos": round(min(cs), 4),
            "slice_norms": [round(v, 4) for v in slice_norms],
            "merged_norm": round(merged_norm, 4),
            "merge_shrinkage": round(merged_norm / mean_slice_norm, 4)
            if mean_slice_norm else None,
            "status": status}

    identified = [L for L in layers if rows[L]["status"] == "IDENTIFIED"]
    not_ident = [L for L in layers if rows[L]["status"] == "NOT_IDENTIFIED"]
    summ = {
        "n_slices": N_SLICES, "n_prompts_per_slice": 30,
        "gate": {"identified_min_cos": IDENT_OK,
                 "partial_min_cos": IDENT_PARTIAL,
                 "note": "gate fixed in code; cosine over flattened J"},
        "by_layer": rows,
        "identified_layers": identified,
        "not_identified_layers": not_ident,
        "usable_band": identified,
        "reading": (
            f"Layers {identified} are identified (independent 30-prompt "
            f"corpora recover the same linear map). Layers {not_ident} are "
            f"NOT: their slice Jacobians are mutually near-orthogonal and "
            f"cancel on merge. CONSEQUENCE: no readout claim — positive or "
            f"negative — may be made at a non-identified layer, because "
            f"there is no fitted instrument there to make it with. In "
            f"particular L22 sits at Gemma's 37% relative depth, the "
            f"shallow end of the paper's band, so the paper-band readout "
            f"test must run at L30/L37 and report L22 as instrument-"
            f"unavailable rather than as evidence about the model."
            if not_ident else
            f"All layers {identified} identified; the full band is usable."),
        "seconds": round(time.time() - t0)}
    prov = Provenance(
        evidence_id="a3-gemma-identification-v1", tier="pilot",
        command="python -m jspace_part2.experiments.a3_gemma_identification",
        inputs={f"slice{i}": sha256_file(LENS_DIR / f"gemma431_slice{i}.pt")
                for i in range(N_SLICES)} |
               {"merged": sha256_file(LENS_DIR / "gemma431_lens.pt")})
    write_result(summ, OUT, prov)
    registry_append({
        "evidence_id": "a3-gemma-identification-v1", "tier": "pilot",
        "what": (f"Per-layer identification of the Gemma 120-prompt lens "
                 f"(4 independent 30-prompt slices, pairwise cosine): "
                 f"IDENTIFIED {identified}; NOT IDENTIFIED {not_ident} "
                 f"(L22 mean cos {rows[22]['mean_pairwise_cos'] if 22 in rows else 'n/a'}). "
                 f"Gates which layers may carry readout claims at all."),
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(OUT), "sha256": sha256_file(OUT)}]})
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
