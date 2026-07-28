# R2 runner — paper-defined occupancy + excess variance for one model.
#
# Streams residuals at the configured layers over the SHARED descriptive
# prompt set (v1 file, cross-model commensurability), then per layer:
# nonneg pursuit over the model's J dictionary vs n_rand vocab-sized
# random dictionaries, frozen crossing rule -> per-position occupancy,
# excess reconstruction share at the layer-median occupancy.
#
# This is the estimator that REPLACES part-1's thresholded active counts /
# raw variance share for any capacity claim (addendum §0.2, §5.4).
#
# Usage: python -m jspace_part2.experiments.r2_occupancy \
#          --config configs/r2_occupancy_think.yaml [--allow-dirty]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import yaml

from ..dictionaries import build_j_dictionaries
from ..lib import sha256_file
from ..occupancy import occupancy_and_excess
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          resolve_model, write_result)

SKIP_FIRST, MAX_SEQ = 16, 256


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    cfg_path = arg("--config")
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    git = require_clean_tree("--allow-dirty" in sys.argv)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "r2_occupancy.json"
    if out_json.exists() and "--force" not in sys.argv:
        print(f"{out_json} exists; skipping")
        return

    import transformers
    import jlens
    from jlens import JacobianLens
    from jlens.hooks import ActivationRecorder

    lens = JacobianLens.load(cfg["lens_path"])
    tok = transformers.AutoTokenizer.from_pretrained(cfg["model_path"])
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        cfg["model_path"], dtype=torch.bfloat16).to("cuda").eval()
    model = jlens.from_hf(hf, tok)
    layers = cfg["layers"]
    prompts = [json.loads(l) for l in
               Path(cfg["prompt_set"]).read_text().splitlines()]
    prompts = prompts[:cfg["n_prompts"]]

    # ---- capture residuals
    acts = {l: [] for l in layers}
    t0 = time.time()
    with torch.no_grad():
        for i, p in enumerate(prompts):
            ids = model.encode(p["text"], max_length=MAX_SEQ)
            with ActivationRecorder(model.layers, at=layers) as rec:
                model.forward(ids)
            P = ids.shape[1]
            lo = min(SKIP_FIRST, max(P - 8, 1))
            for l in layers:
                acts[l].append(rec.activations[l][0, lo:P - 1].float())
            if (i + 1) % 20 == 0:
                print(f"  capture {i+1}/{len(prompts)} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    H = {l: torch.cat(acts[l]) for l in layers}
    del acts
    print(f"captured: {[tuple(H[l].shape) for l in layers]}", flush=True)

    jd = build_j_dictionaries(hf, lens, layers)
    V, d = jd[layers[0]].shape
    res = {"per_layer": {}, "config": cfg, "n_positions":
           {str(l): int(H[l].shape[0]) for l in layers}}
    for l in layers:
        rand_dicts = []
        for s in cfg["rand_seeds"]:
            g = torch.Generator().manual_seed(s)
            R = torch.nn.functional.normalize(
                torch.randn(V, d, generator=g), dim=1).to("cuda",
                                                          torch.float16)
            rand_dicts.append(R)
        gm = H[l].mean(0)
        t1 = time.time()
        out = occupancy_and_excess(H[l].cuda(), jd[l], rand_dicts,
                                   cfg["k_max"], gm.cuda())
        occ = out.pop("occupancy")
        res["per_layer"][str(l)] = {
            **{k: (v.tolist() if torch.is_tensor(v) else v)
               for k, v in out.items()},
            "occ_median": float(occ.float().median()),
            "occ_q25": float(occ.float().quantile(0.25)),
            "occ_q75": float(occ.float().quantile(0.75)),
            "occ_hist": torch.bincount(occ, minlength=cfg["k_max"] + 1)
                        .tolist(),
            "seconds": round(time.time() - t1),
        }
        print(f"L{l}: occ median {res['per_layer'][str(l)]['occ_median']} "
              f"IQR [{res['per_layer'][str(l)]['occ_q25']},"
              f"{res['per_layer'][str(l)]['occ_q75']}] "
              f"excess_share {out['excess_share']:.4f} "
              f"(censored {out['censored_frac']:.2f})", flush=True)
        del rand_dicts
        torch.cuda.empty_cache()

    prov = Provenance(
        evidence_id=cfg["evidence_id"], tier=cfg["tier"],
        command=f"python -m jspace_part2.experiments.r2_occupancy --config {cfg_path}",
        config_path=cfg_path,
        inputs={"lens": sha256_file(cfg["lens_path"]),
                "prompt_set": sha256_file(cfg["prompt_set"])},
        model=resolve_model(cfg["model_path"]),
        seed=cfg["rand_seeds"][0], allow_dirty="--allow-dirty" in sys.argv)
    write_result(res, out_json, prov)
    registry_append({
        "evidence_id": cfg["evidence_id"], "tier": cfg["tier"],
        "what": ("paper-defined occupancy/excess: " +
                 "; ".join(f"L{l}: occ_med {v['occ_median']}, excess "
                           f"{v['excess_share']:.4f}"
                           for l, v in res["per_layer"].items())),
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "outputs": [{"path": str(out_json), "sha256": sha256_file(out_json)}]})
    print("R2 done")


if __name__ == "__main__":
    main()
