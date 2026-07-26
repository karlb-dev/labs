# Phase 2: descriptive replication on Olmo-3-32B-Think.
#
# Paper targets: ~10-25 meaningfully-active J-lens vectors per (position,
# layer); J-space share of activation variance <10%, peaking mid-network;
# high persistence (autocorrelation) of top concepts through the workspace
# band. Method (paper #methods): sparse NONNEGATIVE decomposition of the
# residual onto the J-lens dictionary via gradient pursuit; J-space
# component = the k-sparse reconstruction; active count = atoms with
# meaningful coefficients (threshold sensitivity reported).
#
# Dictionary at layer l: rows of (W_U ⊙ g) @ J_l, row-normalized — the
# final-norm gain g folds the RMSNorm linearization into the readout
# direction (noted in report limitations).
#
# Also saves per-layer residual PCA (top 64 PCs over the corpus) and the
# per-layer top-activated J-direction ids — reused by s6 (broadcast) and
# s7 (ablation controls). Resumable per prompt-batch of 20.
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl1_common import (LOCAL_WORK, RUN_DIR, atomic_write_json, die,
                        ensure_dirs, load_model, log, read_json, seed_all)

import numpy as np
import torch
from jlens import JacobianLens
from jlens.hooks import ActivationRecorder

OUT_AGG = RUN_DIR / "metrics" / "descriptive_agg.json"
OUT_READOUTS = RUN_DIR / "metrics" / "descriptive_readouts.jsonl"
PROMPTS = RUN_DIR / "config" / "prompts" / "descriptive_prompts.jsonl"
STATE_DIR = RUN_DIR / "metrics" / "layer_state"
BATCH, SKIP_FIRST, MAX_SEQ = 20, 16, 256
K_MAX = 50            # pursuit budget; paper's typical k=25 sits inside
K_PAPER = 25
REL_THRESHOLDS = (0.01, 0.02, 0.05)   # coeff > θ·||h|| counts as active


# ---------------------------------------------------------------- prompts
def build_prompts() -> list[dict]:
    seed_all()
    rng = np.random.default_rng(0)
    prompts: list[dict] = []

    # factual: repo truth sets + boot-style two-hoppers
    try:
        import csv
        with open("/content/labs/interpretability/data/truth_cities.csv") as f:
            rows = list(csv.DictReader(f))
        txt_key = next(k for k in rows[0] if k.lower() in
                       ("prompt", "text", "statement", "claim", "sentence"))
        for r in rows[:30]:
            prompts.append({"domain": "factual", "text": r[txt_key]})
    except Exception as e:
        log(f"truth_cities unavailable ({e}); using generated factual only")
    mh = json.loads(Path(
        "/content/jacobian-lens/data/evaluations/lens-eval-multihop.json"
    ).read_text())["items"]
    for it in mh[:20]:
        prompts.append({"domain": "factual", "text": it["prompt"]})

    # multi-step arithmetic (chained, step-by-step continuation style)
    ops = [("+", lambda a, b: a + b), ("-", lambda a, b: a - b),
           ("*", lambda a, b: a * b)]
    while sum(p["domain"] == "arithmetic" for p in prompts) < 50:
        a, b, c, d = rng.integers(2, 60, size=4)
        (o1, f1), (o2, f2) = ops[rng.integers(3)], ops[rng.integers(2)]
        expr = f"(({a} {o1} {b}) {o2} {c}) + {d}"
        val = f2(f1(int(a), int(b)), int(c)) + int(d)
        prompts.append({"domain": "arithmetic",
                        "text": f"Problem: compute {expr} step by step.\n"
                                f"First, {a} {o1} {b} =",
                        "answer": str(val)})

    # SQL with schema tracking across 3 tables (our domain twist)
    schemas = [
        ("users(id, name, city_id)", "cities(id, city_name, country)",
         "orders(id, user_id, total)",
         "-- Total order value per country\nSELECT c.country, SUM(o.total)\n"
         "FROM orders o JOIN users u ON o.user_id = u.id\n"
         "JOIN cities c ON"),
        ("products(pid, pname, cat_id)", "categories(cat_id, cat_name)",
         "sales(sid, pid, qty, price)",
         "-- Revenue by category name\nSELECT cat.cat_name, "
         "SUM(s.qty * s.price)\nFROM sales s JOIN products p ON s.pid = p.pid\n"
         "JOIN categories cat ON"),
        ("employees(eid, ename, dept_id, mgr_id)", "departments(dept_id, dname)",
         "salaries(eid, amount, year)",
         "-- 2025 payroll per department\nSELECT d.dname, SUM(sal.amount)\n"
         "FROM salaries sal JOIN employees e ON sal.eid = e.eid\n"
         "JOIN departments d ON"),
    ]
    for i in range(50):
        t1, t2, t3, q = schemas[i % len(schemas)]
        prompts.append({"domain": "sql",
                        "text": f"-- Tables:\n-- {t1}\n-- {t2}\n-- {t3}\n{q}"})

    # open-ended prose: wikitext spares (fit used rows 0..119)
    corpus = [json.loads(l) for l in
              (RUN_DIR / "config" / "prompts" / "fitting_corpus.jsonl")
              .read_text().splitlines()]
    for r in corpus[120:170]:
        prompts.append({"domain": "prose", "text": r["text"]})

    if len(prompts) < 160:
        die(f"only {len(prompts)} prompts built")
    for i, p in enumerate(prompts):
        p["pid"] = i
    return prompts


# ---------------------------------------------------- pursuit (batched)
@torch.no_grad()
def gradient_pursuit(h: torch.Tensor, D: torch.Tensor, k_max: int,
                     refit_iters: int = 8, lr: float = 0.25):
    """Sparse nonnegative decomposition of each row of h onto dictionary D.

    h: [B, d] fp32 (GPU), D: [V, d] fp16 row-normalized (GPU).
    Returns (indices [B,k], coeffs [B,k], recon [B,d]) with coeffs >= 0.
    Greedy atom selection by positive correlation + projected-gradient
    nonneg refit — the gradient-pursuit family used in the paper.
    """
    B, d = h.shape
    Dh = D.half()
    idxs = torch.zeros(B, k_max, dtype=torch.long, device=h.device)
    coeffs = torch.zeros(B, k_max, device=h.device)
    recon = torch.zeros_like(h)
    taken = torch.zeros(B, D.shape[0], dtype=torch.bool, device=h.device)
    for k in range(k_max):
        r = (h - recon).half()
        corr = r @ Dh.T                          # [B, V]
        corr.masked_fill_(taken, float("-inf"))
        best = corr.argmax(dim=1)                # [B]
        idxs[:, k] = best
        taken[torch.arange(B), best] = True
        D_A = D[idxs[:, :k + 1]].float()         # [B, k+1, d]
        c = coeffs[:, :k + 1]
        for _ in range(refit_iters):
            resid = h - torch.einsum("bk,bkd->bd", c, D_A)
            grad = torch.einsum("bd,bkd->bk", resid, D_A)
            c = (c + lr * grad).clamp_(min=0)
        coeffs[:, :k + 1] = c
        recon = torch.einsum("bk,bkd->bd", c, D_A)
    return idxs, coeffs, recon


# ------------------------------------------------------------------ main
def main() -> None:
    ensure_dirs()
    seed_all()
    force = "--force" in sys.argv
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    if not PROMPTS.exists() or force:
        rows = build_prompts()
        tmp = PROMPTS.with_suffix(".tmp")
        with tmp.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        tmp.replace(PROMPTS)
        log(f"built {len(rows)} descriptive prompts")
    prompts = [json.loads(l) for l in PROMPTS.read_text().splitlines()]

    lens_path = RUN_DIR / "lens" / "olmo32bthink_lens.pt"
    if not lens_path.exists():
        slices = sorted(RUN_DIR.glob("lens/olmo32bthink_slice*.pt"))
        if not slices:
            die("no lens available")
        lens_path = slices[-1]
        log(f"using latest slice lens {lens_path.name}")
    lens = JacobianLens.load(str(lens_path))
    layers = lens.source_layers

    agg = read_json(OUT_AGG) if OUT_AGG.exists() and not force else {
        "lens_file": lens_path.name, "layers": layers,
        "k_max": K_MAX, "k_paper": K_PAPER,
        "rel_thresholds": list(REL_THRESHOLDS),
        "batches_done": 0,
        "per_layer": {str(l): {
            "n_positions": 0, "energy_h": 0.0, "energy_recon": 0.0,
            "sum_h": None, "sum_recon_sq_c": 0.0, "sum_h_sq_c": 0.0,
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

    model, hf, tok = load_model("main")
    W_U = hf.lm_head.weight.detach()                     # [V, d] bf16
    g = hf.model.norm.weight.detach().float()            # RMS gain [d]

    # per-layer running residual moments for PCA (fp64 CPU accumulation)
    moments = {l: {"n": 0, "sum": torch.zeros(model.d_model, dtype=torch.float64),
                   "cov": torch.zeros(model.d_model, model.d_model,
                                      dtype=torch.float64)} for l in layers}

    readout_f = OUT_READOUTS.open("a")
    for b in range(start_batch, n_batches):
        t0 = time.time()
        chunk = prompts[b * BATCH:(b + 1) * BATCH]
        # ---- forward passes, record residuals
        acts = {l: [] for l in layers}   # list of [P, d] fp32 GPU
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
        # ---- per layer: dictionary, pursuit, aggregates
        for l in layers:
            H = torch.cat(acts[l])                        # [B, d]
            key = str(l)
            # direction for token t at layer l is J_l^T u'_t, i.e. row t of
            # (W_U ⊙ g) @ J_l  (transport is h @ J^T -> logit_t = u'_t^T J h)
            D = (W_U.float() * g[None, :]) @ lens.jacobians[l].to("cuda")
            D = torch.nn.functional.normalize(D, dim=1).half()
            idxs, coeffs, recon = gradient_pursuit(H, D, K_MAX)
            hn = H.norm(dim=1)
            # active counts at relative thresholds
            for t in REL_THRESHOLDS:
                cnt = (coeffs > t * hn[:, None]).sum(dim=1)
                agg["per_layer"][key]["active_counts"][str(t)].append(
                    [float(cnt.float().mean()), float(cnt.float().median())])
            # atoms needed for 90% of achievable reconstruction energy
            order = coeffs.argsort(dim=1, descending=True)
            csort = coeffs.gather(1, order)
            cum = (csort ** 2).cumsum(dim=1)
            need = (cum < 0.9 * cum[:, -1:]).sum(dim=1) + 1
            agg["per_layer"][key]["active_counts_k90"].append(
                [float(need.float().mean()), float(need.float().median())])
            # energy + centered variance share
            e_h = float((hn ** 2).sum())
            e_r = float((recon.norm(dim=1) ** 2).sum())
            pl = agg["per_layer"][key]
            pl["energy_h"] += e_h
            pl["energy_recon"] += e_r
            pl["n_positions"] += H.shape[0]
            mu_h, mu_r = H.mean(0), recon.mean(0)
            pl["sum_h_sq_c"] += float(((H - mu_h) ** 2).sum())
            pl["sum_recon_sq_c"] += float(((recon - mu_r) ** 2).sum())
            # top-1 persistence within each prompt
            off = 0
            top1 = idxs[torch.arange(H.shape[0]), coeffs.argmax(dim=1)]
            for m in metas:
                n = m["n_pos"]
                if n > 1:
                    seg = top1[off:off + n]
                    pl["top1_persist"][0] += int((seg[1:] == seg[:-1]).sum())
                    pl["top1_persist"][1] += n - 1
                off += n
            # mean activation per direction id (for s6/s7 top-direction sets)
            uniq, inv = idxs.unique(return_inverse=True)
            sums = torch.zeros(len(uniq), device=H.device).index_add_(
                0, inv.flatten(), coeffs.flatten())
            for u, s in zip(uniq.tolist(), sums.tolist()):
                pl["dir_activation_sum"][str(u)] = \
                    pl["dir_activation_sum"].get(str(u), 0.0) + s
            # PCA moments
            Hc = H.double().cpu()
            moments[l]["n"] += Hc.shape[0]
            moments[l]["sum"] += Hc.sum(0)
            moments[l]["cov"] += Hc.T @ Hc
            # compressed readout archive (every 4th position, top-8)
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
        atomic_write_json(agg, OUT_AGG)
        log(f"batch {b + 1}/{n_batches} done in {time.time()-t0:.0f}s "
            f"({sum(m['n_pos'] for m in metas)} positions)")

    # ---- finalize: PCA per layer -> saved state for s6/s7
    for l in layers:
        n = moments[l]["n"]
        mu = (moments[l]["sum"] / n)
        cov = moments[l]["cov"] / n - torch.outer(mu, mu)
        evals, evecs = torch.linalg.eigh(cov)
        top = evecs[:, -64:].flip(1).float()             # [d, 64] descending
        pl = agg["per_layer"][str(l)]
        top_dirs = sorted(pl["dir_activation_sum"].items(),
                          key=lambda kv: -kv[1])[:256]
        torch.save({"pca_evecs": top, "pca_evals": evals[-64:].flip(0).float(),
                    "mean": mu.float(), "top_dir_ids":
                        [int(k) for k, _ in top_dirs]},
                   STATE_DIR / f"layer_{l}.pt")
    agg["state_dir"] = str(STATE_DIR)
    atomic_write_json(agg, OUT_AGG)
    log(f"wrote {OUT_AGG} + per-layer PCA/top-direction state")

    # headline print
    for l in layers[::4]:
        pl = agg["per_layer"][str(l)]
        vs = pl["sum_recon_sq_c"] / max(pl["sum_h_sq_c"], 1e-9)
        med = np.median([m for _, m in pl["active_counts"]["0.02"]])
        log(f"L{l:>2}: centered variance share {vs:.3f}; "
            f"median active@0.02 {med:.1f}")


if __name__ == "__main__":
    main()
