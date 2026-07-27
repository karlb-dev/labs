# Post-merge QC for the Instruct fit: did the corpus-row-~104 Jacobian
# spike (max||J||/sqrt(d)=1.84, all other prompts 0.93-0.95) distort
# slice 3's averaged lens relative to slices 0-2?
# CPU-only; reads the four slice lenses + merged; writes
# metrics/olmo31-instruct/fit_slice_norms.json. Exploratory QC artifact.
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl2_common import RUN_DIR_P2, atomic_write_json, log

import torch
from jlens import JacobianLens

OUT = RUN_DIR_P2 / "metrics" / "olmo31-instruct" / "fit_slice_norms.json"


def main():
    if OUT.exists() and "--force" not in sys.argv:
        log(f"{OUT} exists; skipping")
        return
    rows = {}
    for name in ["olmo31instruct_slice0", "olmo31instruct_slice1",
                 "olmo31instruct_slice2", "olmo31instruct_slice3",
                 "olmo31instruct_lens"]:
        p = RUN_DIR_P2 / "lens" / f"{name}.pt"
        t0 = time.time()
        lens = JacobianLens.load(str(p))
        d = lens.d_model
        per_layer = {l: round(float(lens.jacobians[l].float().norm())
                              / d, 5)  # Frobenius / d (scale-free-ish)
                     for l in lens.source_layers}
        rows[name] = {"n_prompts": lens.n_prompts, "fro_over_d": per_layer,
                      "load_s": round(time.time() - t0)}
        log(f"{name}: mean fro/d "
            f"{sum(per_layer.values())/len(per_layer):.5f}")
    # slice-3 deviation vs mean of slices 0-2, per layer
    s012 = [rows[f"olmo31instruct_slice{k}"]["fro_over_d"] for k in range(3)]
    dev = {}
    for l in rows["olmo31instruct_slice3"]["fro_over_d"]:
        base = sum(s[l] for s in s012) / 3
        dev[l] = round(rows["olmo31instruct_slice3"]["fro_over_d"][l] / base
                       - 1, 4)
    worst = max(dev.items(), key=lambda kv: abs(kv[1]))
    verdict = ("SLICE3_OK" if abs(worst[1]) < 0.05 else
               "SLICE3_DEVIATES")
    out = {"per_lens": rows, "slice3_rel_dev_vs_mean012": dev,
           "worst_layer": {"layer": worst[0], "rel_dev": worst[1]},
           "verdict": verdict,
           "context": "corpus row ~104 spiked per-prompt max||J||/sqrt(d) "
                      "to 1.84 during slice 3 (others 0.93-0.95)"}
    atomic_write_json(out, OUT)
    log(f"wrote {OUT}; verdict {verdict}; worst layer {worst}")


if __name__ == "__main__":
    main()
