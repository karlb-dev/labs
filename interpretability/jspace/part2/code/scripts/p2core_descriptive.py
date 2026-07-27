# Core Battery descriptive geometry for a matrix model — the capacity
# instrument (variance share, active concepts at θ∈{0.01,0.02,0.05}, k90,
# top-1 persistence) plus the per-layer state (PCA evecs + top J-direction
# ids) that the energy-matched static grid consumes.
#
# Method identical to part-1 s5 (same pursuit code path with the VM5
# numerics fixes — imported, not copied; same prompt set: v1's
# descriptive_prompts.jsonl, reused verbatim for cross-model
# commensurability; same thresholds/batching). Resumable per batch of 20.
#
# Usage: python scripts/p2core_descriptive.py --model olmo31-instruct \
#          --lens <path/to/lens.pt> [--force]
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl2_common import (RUN_DIR, atomic_write_json, die, log, p2_load_model,
                        p2_metrics_dir, read_json, seed_all)

import numpy as np
import torch
from jlens import JacobianLens
from jlens.hooks import ActivationRecorder

PART1_SCRIPTS = Path(__file__).resolve().parents[1] / "part1" / "scripts"
sys.path.insert(0, str(PART1_SCRIPTS))
from s5_descriptive import (BATCH, K_MAX, K_PAPER, MAX_SEQ,  # noqa: E402
                            REL_THRESHOLDS, SKIP_FIRST, gradient_pursuit)

PROMPTS = RUN_DIR / "config" / "prompts" / "descriptive_prompts.jsonl"


def arg(flag: str, default: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main() -> None:
    seed_all()
    force = "--force" in sys.argv
    slug = arg("--model", "olmo31-instruct")
    lens_path = Path(arg("--lens", ""))
    if not lens_path.exists():
        die(f"--lens required and must exist (got {lens_path!r})")
    if not PROMPTS.exists():
        die(f"missing shared prompt set {PROMPTS}")
    mdir = p2_metrics_dir(slug)
    out_agg = mdir / "descriptive.json"
    out_readouts = mdir / "descriptive_readouts.jsonl"
    state_dir = mdir / "layer_state"
    state_dir.mkdir(parents=True, exist_ok=True)

    prompts = [json.loads(l) for l in PROMPTS.read_text().splitlines()]
    lens = JacobianLens.load(str(lens_path))
    layers = lens.source_layers

    agg = read_json(out_agg) if out_agg.exists() and not force else {
        "model_slug": slug, "lens_file": lens_path.name, "layers": layers,
        "k_max": K_MAX, "k_paper": K_PAPER,
        "rel_thresholds": list(REL_THRESHOLDS),
        "prompt_set": "v1 descriptive_prompts.jsonl (shared)",
        "batches_done": 0,
        "per_layer": {str(l): {
            "n_positions": 0, "energy_h": 0.0, "energy_recon": 0.0,
            "sum_recon_sq_c": 0.0, "sum_h_sq_c": 0.0,
            "active_counts": {str(t): [] for t in REL_THRESHOLDS},
            "active_counts_k90": [], "top1_persist": [0, 0],
            "dir_activation_sum": {},
        } for l in layers},
    }
    start_batch = agg["batches_done"]
    n_batches = (len(prompts) + BATCH - 1) // BATCH
    if start_batch >= n_batches and not force:
        log("descriptive already complete")
        return

    model, hf, tok = p2_load_model(slug)
    W_U = hf.lm_head.weight.detach()
    g = hf.model.norm.weight.detach().float()

    moments = {l: {"n": 0,
                   "sum": torch.zeros(model.d_model, dtype=torch.float64),
                   "cov": torch.zeros(model.d_model, model.d_model,
                                      dtype=torch.float64)} for l in layers}
    # NOTE: PCA moments restart at batch 0 semantics — like s5, a resumed
    # run recomputes moments only over remaining batches. For a clean state
    # file, prefer uninterrupted runs; interruptions are recorded.
    if start_batch:
        agg.setdefault("resume_points", []).append(start_batch)

    readout_f = out_readouts.open("a")
    for b in range(start_batch, n_batches):
        t0 = time.time()
        chunk = prompts[b * BATCH:(b + 1) * BATCH]
        acts = {l: [] for l in layers}
        metas = []
        for p in chunk:
            ids = model.encode(p["text"], max_length=MAX_SEQ)
            with ActivationRecorder(model.layers, at=layers) as rec:
                with torch.no_grad():
                    model.forward(ids)
            P = ids.shape[1]
            lo = min(SKIP_FIRST, max(P - 8, 1))
            for l in layers:
                acts[l].append(rec.activations[l][0, lo:P - 1].float())
            metas.append({"pid": p["pid"], "domain": p["domain"],
                          "n_pos": P - 1 - lo})
        for l in layers:
            H = torch.cat(acts[l])
            key = str(l)
            D = (W_U.float() * g[None, :]) @ lens.jacobians[l].to("cuda")
            D = torch.nn.functional.normalize(D, dim=1).half()
            idxs, coeffs, recon = gradient_pursuit(H, D, K_MAX)
            hn = H.norm(dim=1)
            pl = agg["per_layer"][key]
            for t in REL_THRESHOLDS:
                cnt = (coeffs > t * hn[:, None]).sum(dim=1)
                pl["active_counts"][str(t)].append(
                    [float(cnt.float().mean()), float(cnt.float().median())])
            order = coeffs.argsort(dim=1, descending=True)
            csort = coeffs.gather(1, order)
            cum = (csort ** 2).cumsum(dim=1)
            need = (cum < 0.9 * cum[:, -1:]).sum(dim=1) + 1
            pl["active_counts_k90"].append(
                [float(need.float().mean()), float(need.float().median())])
            pl["energy_h"] += float((hn ** 2).sum())
            pl["energy_recon"] += float((recon.norm(dim=1) ** 2).sum())
            pl["n_positions"] += H.shape[0]
            mu_h, mu_r = H.mean(0), recon.mean(0)
            pl["sum_h_sq_c"] += float(((H - mu_h) ** 2).sum())
            pl["sum_recon_sq_c"] += float(((recon - mu_r) ** 2).sum())
            off = 0
            top1 = idxs[torch.arange(H.shape[0]), coeffs.argmax(dim=1)]
            for m in metas:
                n = m["n_pos"]
                if n > 1:
                    seg = top1[off:off + n]
                    pl["top1_persist"][0] += int((seg[1:] == seg[:-1]).sum())
                    pl["top1_persist"][1] += n - 1
                off += n
            uniq, inv = idxs.unique(return_inverse=True)
            sums = torch.zeros(len(uniq), device=H.device).index_add_(
                0, inv.flatten(), coeffs.flatten())
            for u, s in zip(uniq.tolist(), sums.tolist()):
                pl["dir_activation_sum"][str(u)] = \
                    pl["dir_activation_sum"].get(str(u), 0.0) + s
            Hc = H.double().cpu()
            moments[l]["n"] += Hc.shape[0]
            moments[l]["sum"] += Hc.sum(0)
            moments[l]["cov"] += Hc.T @ Hc
            if l in layers[::3]:
                sel = torch.arange(0, H.shape[0], 4)
                vals, toks = (H[sel].half() @ D.T).topk(8, dim=1)
                readout_f.write(json.dumps({
                    "batch": b, "layer": l,
                    "pids": [m["pid"] for m in metas],
                    "top_tokens": toks.tolist(),
                    "top_vals": [[round(v, 3) for v in row]
                                 for row in vals.float().tolist()],
                }) + "\n")
            del D, H, recon
        readout_f.flush()
        agg["batches_done"] = b + 1
        atomic_write_json(agg, out_agg)
        log(f"batch {b + 1}/{n_batches} done in {time.time()-t0:.0f}s")

    for l in layers:
        n = moments[l]["n"]
        if not n:
            continue
        mu = moments[l]["sum"] / n
        cov = moments[l]["cov"] / n - torch.outer(mu, mu)
        evals, evecs = torch.linalg.eigh(cov)
        top = evecs[:, -64:].flip(1).float()
        pl = agg["per_layer"][str(l)]
        top_dirs = sorted(pl["dir_activation_sum"].items(),
                          key=lambda kv: -kv[1])[:256]
        torch.save({"pca_evecs": top,
                    "pca_evals": evals[-64:].flip(0).float(),
                    "mean": mu.float(),
                    "top_dir_ids": [int(k) for k, _ in top_dirs]},
                   state_dir / f"layer_{l}.pt")
    agg["state_dir"] = str(state_dir)
    atomic_write_json(agg, out_agg)
    log(f"wrote {out_agg} + layer state")
    for l in layers[::4]:
        pl = agg["per_layer"][str(l)]
        vs = pl["sum_recon_sq_c"] / max(pl["sum_h_sq_c"], 1e-9)
        med = np.median([m for _, m in pl["active_counts"]["0.02"]])
        log(f"L{l:>2}: centered variance share {vs:.4f}; "
            f"median active@0.02 {med:.1f}")


if __name__ == "__main__":
    main()
