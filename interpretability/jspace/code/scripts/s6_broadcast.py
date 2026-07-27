# Phase 2b: broadcast test — do downstream components READ the J-space?
#
# Paper claim: workspace content is broadcast (fanned out to many later
# consumers), not routed point-to-point. Weights-level test: for direction
# d in the residual stream at source layer l, a later component "reads" d
# if its input projection moves it more than it moves a random direction:
#     z(c, d) = (||W_c d|| - mean_r ||W_c r||) / std_r ||W_c r||
# fan_out(d) = #{components c in layers > l : z(c, d) > 3}.
#
# Groups compared at each source layer: (a) top-64 J-directions by s5
# pursuit activation, (b) 64 random unit directions, (c) top-64 residual
# PCs orthogonalized against the J-span (high-variance but non-verbalizable
# control). CPU-only (weights already page-cached) so it can run while the
# GPU is busy. Output: metrics/broadcast.json.
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl1_common import (RUN_DIR, atomic_write_json, die, ensure_dirs, log,
                        read_json, seed_all, MODELS)

import numpy as np
import torch
from jlens import JacobianLens

OUT = RUN_DIR / "metrics" / "broadcast.json"
STATE_DIR = RUN_DIR / "metrics" / "layer_state"
SOURCE_POINTS = [24, 32, 40]      # mid-band source layers
N_DIRS, N_RAND_BASE = 64, 256     # dirs per group; random baseline pool
Z_THRESH = 3.0


def component_mats(hf, layer: int):
    """Input-projection matrices reading the residual at `layer`'s input."""
    blk = hf.model.layers[layer]
    return {
        f"L{layer}.q": blk.self_attn.q_proj.weight,
        f"L{layer}.k": blk.self_attn.k_proj.weight,
        f"L{layer}.v": blk.self_attn.v_proj.weight,
        f"L{layer}.gate": blk.mlp.gate_proj.weight,
        f"L{layer}.up": blk.mlp.up_proj.weight,
    }


def main() -> None:
    ensure_dirs()
    seed_all()
    if OUT.exists() and "--force" not in sys.argv:
        log(f"{OUT} exists; skipping")
        return
    lens = JacobianLens.load(str(RUN_DIR / "lens" / "olmo32bthink_lens.pt"))
    if not (STATE_DIR / "layer_24.pt").exists():
        die("s5 layer state missing; run s5_descriptive.py first")

    import transformers
    log("loading 32B on CPU (weights only, no GPU)")
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODELS["main"], dtype=torch.bfloat16, device_map="cpu")
    hf.eval()
    W_U = hf.lm_head.weight.detach().float()
    g = hf.model.norm.weight.detach().float()
    d_model = W_U.shape[1]
    n_layers = hf.config.num_hidden_layers

    rng = np.random.default_rng(0)
    results = {"z_thresh": Z_THRESH, "n_dirs": N_DIRS, "source_layers": {}}
    for l_src in SOURCE_POINTS:
        t0 = time.time()
        state = torch.load(STATE_DIR / f"layer_{l_src}.pt", weights_only=True)
        J = lens.jacobians[l_src]

        # (a) top-J directions
        ids = state["top_dir_ids"][:N_DIRS]
        D_j = torch.nn.functional.normalize((W_U[ids] * g[None, :]) @ J, dim=1)
        # (b) random
        R = torch.tensor(rng.standard_normal((N_RAND_BASE, d_model)),
                         dtype=torch.float32)
        R = torch.nn.functional.normalize(R, dim=1)
        D_r, R_base = R[:N_DIRS], R[N_DIRS:]
        # (c) top PCs orthogonalized against J-span (top-256 J-dirs)
        span_ids = state["top_dir_ids"][:256]
        S = torch.nn.functional.normalize((W_U[span_ids] * g[None, :]) @ J, dim=1)
        Q, _ = torch.linalg.qr(S.T)                       # [d, 256] orthobasis
        P = state["pca_evecs"].T                          # [64, d]
        P_perp = P - (P @ Q) @ Q.T
        keep = P_perp.norm(dim=1) > 0.5
        D_p = torch.nn.functional.normalize(P_perp[keep], dim=1)[:N_DIRS]
        log(f"L{l_src}: nonJ-PCA control kept {D_p.shape[0]}/64 "
            f"(>50% norm outside J-span)")

        groups = {"jspace": D_j, "random": D_r, "nonJ_pca": D_p}
        fan = {k: torch.zeros(v.shape[0]) for k, v in groups.items()}
        n_comp = 0
        for m in range(l_src + 1, n_layers):
            for name, W in component_mats(hf, m).items():
                Wf = W.detach().float()
                base = (Wf @ R_base.T).norm(dim=0)        # [192]
                mu, sd = base.mean(), base.std()
                n_comp += 1
                for gname, Dg in groups.items():
                    zs = ((Wf @ Dg.T).norm(dim=0) - mu) / sd
                    fan[gname] += (zs > Z_THRESH).float()
        entry = {"n_components_downstream": n_comp}
        for gname, f in fan.items():
            entry[gname] = {
                "fan_out_median": float(f.median()),
                "fan_out_mean": float(f.mean()),
                "fan_out_p90": float(f.quantile(0.9)),
                "frac_components_median": float(f.median()) / n_comp,
            }
        # Mann-Whitney: J vs each control
        from scipy.stats import mannwhitneyu
        for ctrl in ("random", "nonJ_pca"):
            u = mannwhitneyu(fan["jspace"].numpy(), fan[ctrl].numpy(),
                             alternative="greater")
            entry[f"mw_jspace_gt_{ctrl}_p"] = float(u.pvalue)
        results["source_layers"][str(l_src)] = entry
        atomic_write_json(results, OUT)
        log(f"L{l_src} done in {time.time()-t0:.0f}s: "
            f"median fan-out J={entry['jspace']['fan_out_median']:.0f} "
            f"rand={entry['random']['fan_out_median']:.0f} "
            f"nonJ={entry['nonJ_pca']['fan_out_median']:.0f}")
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
