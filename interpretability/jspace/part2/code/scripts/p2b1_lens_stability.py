# B1/G2 quick QC: agreement between the two INDEPENDENT 120-prompt
# Instruct lenses (draw A: seed-0 corpus rows; draw B: seed-1 disjoint
# rows). Per-layer cosine of vec(J_A), vec(J_B) + norm ratios. The full
# dictionary-level + selection-Jaccard analysis needs GPU (W_U) and runs
# later; this is the cheap first look. CPU-only.
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl2_common import RUN_DIR_P2, atomic_write_json, log

import torch
from jlens import JacobianLens

OUT = RUN_DIR_P2 / "metrics" / "olmo31-instruct" / "fitB_stability.json"


def main():
    if OUT.exists() and "--force" not in sys.argv:
        log(f"{OUT} exists; skipping")
        return
    A = JacobianLens.load(str(RUN_DIR_P2 / "lens" / "olmo31instruct_lens.pt"))
    B = JacobianLens.load(str(RUN_DIR_P2 / "lens" / "olmo31instruct_lensB.pt"))
    rows = {}
    for l in A.source_layers:
        # float64: fp32 accumulation over 26M elements inflates cosines
        # past 1.0 by ~2-5e-3 (observed) — not acceptable for the record.
        Ja = A.jacobians[l].double().flatten()
        Jb = B.jacobians[l].double().flatten()
        cos = float((Ja @ Jb) / (Ja.norm() * Jb.norm()))
        rows[l] = {"cos_vecJ": round(cos, 5),
                   "norm_ratio": round(float(Jb.norm() / Ja.norm()), 4)}
        log(f"L{l}: cos {cos:.4f} norm ratio {rows[l]['norm_ratio']}")
    med = sorted(v["cos_vecJ"] for v in rows.values())[len(rows) // 2]
    out = {"median_cos_vecJ": med, "per_layer": rows,
           "note": "independent corpus draws (disjoint rows, seeds 0/1); "
                   "dictionary-level + frozen-selection Jaccard = GPU pass, "
                   "queued"}
    atomic_write_json(out, OUT)
    log(f"wrote {OUT}; median vec-J cos {med}")


if __name__ == "__main__":
    main()
