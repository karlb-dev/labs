# v2 Phase Q: the SAME clean instruments on Qwen 3.6 27B — model vs harness.
#
# Gate (PLAN_v2 decision tree): runs only after P1-P5 landed AND the causal
# null survived them on OLMo. Question: do the clean instruments (energy-
# matched static spans, frozen per-item selection) also null on the model
# where the published replication reported workspace phenomena? Either
# answer is decisive: Qwen dissociates -> real model difference; Qwen nulls
# -> the published causal story doesn't survive clean instruments anywhere.
#
# Instruments are OURS, the lens is THEIRS: Neuronpedia's published
# 1000-prompt WikiText lens for Qwen3.6-27B (neuronpedia/jacobian-lens,
# trained by @mntss, Anthropic Interpretability) — using the community lens
# isolates the harness: any Qwen/OLMo difference cannot be blamed on our
# 120-prompt fit recipe.
#
# Phases (each no-ops if its output exists; resumable inside):
#   a sanity      paper's multihop bridge-entity eval, J-lens vs logit lens
#                 -> metrics/qwen_sanity.json
#   b descriptive s5-style pursuit on the v1 200-prompt set over the Qwen
#                 band; variance share, active counts, persistence; writes
#                 layer_state_qwen/ (PCA evecs + top J-dir ids for c/d)
#                 -> metrics/qwen_descriptive.json
#   c energy      s11-style raw-h energy pass + matched-rank plan at k=20
#                 -> metrics/qwen_energy_match.json
#   d grid        {none, jspace_k20, vmatch_rand_k20, vmatch_nonJ_k20,
#                  frozen_j10, frozen_rand10} x s7 battery (same generators,
#                 seed 0, 2000-resample bootstrap) -> metrics/qwen_causal_grid.json
#
# Band: same fractional depth as OLMo's L20-44/64 (31-69%), mapped onto
# Qwen's layer count at runtime and recorded. Weights (~54 GB bf16) do NOT
# fit the Drive HF cache -> HF_HUB_CACHE overridden to local disk BEFORE
# sl1_common import; free-space self-check aborts loudly. No-think note:
# the battery uses raw completion prompts (no chat template), so the hybrid
# thinking model answers in completion mode — commensurable with OLMo's
# no-think grid.
import os
import sys
import time
from pathlib import Path

LOCAL_HF = "/content/hf_local"
os.environ["HF_HUB_CACHE"] = LOCAL_HF  # must precede sl1_common's setdefault

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl1_common import (RUN_DIR, RUN_DIR_V2, atomic_write_json, die,
                        gpu_mem_gb, load_model, log, read_json, seed_all)

import json
import numpy as np
import shutil
import torch
from jlens import JacobianLens
from jlens.hooks import ActivationRecorder

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s4_lens_sanity as s4          # noqa: E402  multihop_eval
import s7_ablation as s7             # noqa: E402  build_tasks/run_task/boot_ci
from s5_descriptive import gradient_pursuit  # noqa: E402

MODEL_ID = "Qwen/Qwen3.6-27B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = ("qwen3.6-27b/jlens/Salesforce-wikitext/"
             "Qwen3.6-27B_jacobian_lens_n1000.pt")
M = RUN_DIR_V2 / "metrics"
SAN_OUT = M / "qwen_sanity.json"
DESC_OUT = M / "qwen_descriptive.json"
STATE_DIR = M / "layer_state_qwen"
EM_OUT = M / "qwen_energy_match.json"
GRID_OUT = M / "qwen_causal_grid.json"
PROMPTS = RUN_DIR / "config" / "prompts" / "descriptive_prompts.jsonl"

OLMO_BAND_FRACS = [l / 64 for l in range(20, 45, 2)]
K_MAX, REL_THRESHOLDS, BATCH_P = 50, (0.01, 0.02, 0.05), 20
SKIP_FIRST, MAX_SEQ = 16, 256
DOSE, RAND_POOL, SKIP_SEL, K_FROZEN = 20, 512, 4, 10
MIN_FREE_GB = 60
CONDS = ("none", "jspace_k20", "vmatch_rand_k20", "vmatch_nonJ_k20",
         "frozen_j10", "frozen_rand10")
TASKS = ("twohop", "twohop_lp", "onehop", "arithmetic_v2", "sql",
         "prose_nll", "grammar", "samples")


def qwen_band(n_layers: int) -> list[int]:
    return sorted({int(round(f * n_layers)) for f in OLMO_BAND_FRACS})


class BandAblator(s7.Ablator):
    """s7.Ablator with the band as an argument instead of module state."""

    def __init__(self, model_layers, band):
        super().__init__(model_layers)
        self._band = band

    def __enter__(self):
        for l in self._band:
            self._handles.append(
                self._layers[l].register_forward_hook(self._hook(l)))
        return self


@torch.no_grad()
def frozen_projectors_band(model, dicts, prompt: str, band) -> dict:
    ids = model.encode(prompt, max_length=512)
    with ActivationRecorder(model.layers, at=band) as rec:
        model.forward(ids)
    out = {}
    for l in band:
        h = rec.activations[l][0].float()
        h = h[min(SKIP_SEL, h.shape[0] - 1):]
        score = (h.half() @ dicts[l].T).abs().sum(0)
        top = score.topk(K_FROZEN).indices
        Q, _ = torch.linalg.qr(dicts[l][top].float().T)
        out[l] = Q.contiguous()
    return out


def load_qwen():
    Path(LOCAL_HF).mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage("/content").free / 1e9
    have = any(Path(LOCAL_HF).glob("models--Qwen--*"))
    if not have and free_gb < MIN_FREE_GB:
        die(f"only {free_gb:.0f} GB free on /content; need ~{MIN_FREE_GB} "
            f"for {MODEL_ID} — clear space first")
    model, hf, tok = load_model(MODEL_ID)
    log(f"downloading/loading lens {LENS_REPO}/{LENS_FILE}")
    lens = JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)
    n_layers, d = model.n_layers, model.d_model
    band = [l for l in qwen_band(n_layers) if l in lens.jacobians]
    if len(band) < 8:
        die(f"band collapsed: {band} (lens layers "
            f"{sorted(lens.jacobians)[:8]}...)")
    log(f"qwen: {n_layers} layers, d={d}, vocab {hf.lm_head.weight.shape[0]}; "
        f"band {band}")
    return model, hf, tok, lens, band, d


def build_dicts(lens, hf, band):
    # fp16 staging: Qwen's 248k vocab makes s12's fp32 recipe (~15 GB
    # transients on top of the growing 33 GB fp16 dict) exceed the 97 GB
    # card. Selection only ranks summed |corr|, so fp16 rounding is
    # immaterial; frozen_projectors_band re-floats the 10 chosen rows
    # before QR.
    W_U = hf.lm_head.weight.detach()          # bf16 view, no copy
    g = hf.model.norm.weight.detach()
    d = W_U.shape[1]
    Wg = (W_U * g[None, :]).half()
    jd, rd = {}, {}
    for l in band:
        J = lens.jacobians[l].to(Wg.device).half()
        jd[l] = torch.nn.functional.normalize(Wg @ J, dim=1)
        del J
        rng = np.random.default_rng(2000 + l)
        R = torch.tensor(rng.standard_normal((d, d)), device=Wg.device,
                         dtype=torch.float32)
        rd[l] = torch.nn.functional.normalize(R, dim=1).half()
        torch.cuda.empty_cache()
    del Wg
    used, _ = gpu_mem_gb()
    log(f"dictionaries built for {len(band)} layers; VRAM {used:.1f} GB")
    return jd, rd


def phase_a_sanity(model, tok, lens):
    if SAN_OUT.exists():
        log("phase a (sanity) already done")
        return
    mh, mh_items = s4.multihop_eval(model, tok, lens, 60)
    atomic_write_json({"model": MODEL_ID, "lens": f"{LENS_REPO}/{LENS_FILE}",
                       "note": "min over ALL lens layers (published lens is "
                               "all-layer; v1 OLMo used its 21 fitted)",
                       "multihop": mh, "multihop_items": mh_items}, SAN_OUT)
    log(f"phase a multihop: { {k: round(v, 3) for k, v in mh.items()} }")


def phase_b_descriptive(model, hf, lens, band, d):
    prompts = [json.loads(x) for x in PROMPTS.read_text().splitlines()]
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    agg = read_json(DESC_OUT) if DESC_OUT.exists() else {
        "model": MODEL_ID, "layers": band, "k_max": K_MAX,
        "rel_thresholds": list(REL_THRESHOLDS), "batches_done": 0,
        "per_layer": {str(l): {
            "n_positions": 0, "sum_recon_sq_c": 0.0, "sum_h_sq_c": 0.0,
            "active_counts": {str(t): [] for t in REL_THRESHOLDS},
            "top1_persist": [0, 0], "dir_activation_sum": {},
        } for l in band}}
    n_batches = (len(prompts) + BATCH_P - 1) // BATCH_P
    if agg["batches_done"] >= n_batches and \
            (STATE_DIR / f"layer_{band[0]}.pt").exists():
        log("phase b (descriptive) already done")
        return
    if agg["batches_done"] > 0 and not (STATE_DIR / f"layer_{band[0]}.pt").exists():
        # PCA moments live only in RAM; a mid-phase resume must restart to
        # get correct covariances (positions are cheap: ~200 prompts total)
        log("phase b: partial run without layer state; restarting batches")
        agg["batches_done"] = 0
        for l in band:
            agg["per_layer"][str(l)] = {
                "n_positions": 0, "sum_recon_sq_c": 0.0, "sum_h_sq_c": 0.0,
                "active_counts": {str(t): [] for t in REL_THRESHOLDS},
                "top1_persist": [0, 0], "dir_activation_sum": {}}
    W_U = hf.lm_head.weight.detach()
    g = hf.model.norm.weight.detach().float()
    moments = {l: {"n": 0, "sum": torch.zeros(d, dtype=torch.float64),
                   "cov": torch.zeros(d, d, dtype=torch.float64)}
               for l in band}
    for b in range(agg["batches_done"], n_batches):
        t0 = time.time()
        chunk = prompts[b * BATCH_P:(b + 1) * BATCH_P]
        acts = {l: [] for l in band}
        metas = []
        for p in chunk:
            ids = model.encode(p.get("prompt") or p.get("text"),
                               max_length=MAX_SEQ)
            with ActivationRecorder(model.layers, at=band) as rec:
                with torch.no_grad():
                    model.forward(ids)
            P = ids.shape[1]
            lo = min(SKIP_FIRST, max(P - 8, 1))
            for l in band:
                acts[l].append(rec.activations[l][0, lo:P - 1].float())
            metas.append({"n_pos": P - 1 - lo})
        for l in band:
            H = torch.cat(acts[l])
            D = (W_U.float() * g[None, :]) @ lens.jacobians[l].to("cuda")
            D = torch.nn.functional.normalize(D, dim=1).half()
            idxs, coeffs, recon = gradient_pursuit(H, D, K_MAX)
            pl = agg["per_layer"][str(l)]
            hn = H.norm(dim=1)
            for t in REL_THRESHOLDS:
                cnt = (coeffs > t * hn[:, None]).sum(dim=1)
                pl["active_counts"][str(t)].append(
                    [float(cnt.float().mean()), float(cnt.float().median())])
            mu_h, mu_r = H.mean(0), recon.mean(0)
            pl["sum_h_sq_c"] += float(((H - mu_h) ** 2).sum())
            pl["sum_recon_sq_c"] += float(((recon - mu_r) ** 2).sum())
            pl["n_positions"] += H.shape[0]
            top1 = idxs[torch.arange(H.shape[0]), coeffs.argmax(dim=1)]
            off = 0
            for m2 in metas:
                n = m2["n_pos"]
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
            del D, H, recon
        agg["batches_done"] = b + 1
        atomic_write_json(agg, DESC_OUT)
        log(f"phase b batch {b + 1}/{n_batches} ({time.time() - t0:.0f}s)")
    for l in band:
        n = moments[l]["n"]
        if n == 0:
            continue
        mu = moments[l]["sum"] / n
        cov = moments[l]["cov"] / n - torch.outer(mu, mu)
        evals, evecs = torch.linalg.eigh(cov)
        pl = agg["per_layer"][str(l)]
        top_dirs = sorted(pl["dir_activation_sum"].items(),
                          key=lambda kv: -kv[1])[:256]
        torch.save({"pca_evecs": evecs[:, -64:].flip(1).float(),
                    "pca_evals": evals[-64:].flip(0).float(),
                    "mean": mu.float(),
                    "top_dir_ids": [int(k) for k, _ in top_dirs]},
                   STATE_DIR / f"layer_{l}.pt")
    atomic_write_json(agg, DESC_OUT)
    for l in band:
        pl = agg["per_layer"][str(l)]
        vs = pl["sum_recon_sq_c"] / max(pl["sum_h_sq_c"], 1e-9)
        log(f"phase b L{l}: variance share {vs:.4f}")


def build_bases(lens, hf, band):
    W_U = hf.lm_head.weight.detach().float()
    g = hf.model.norm.weight.detach().float()
    out = {}
    for l in band:
        st = torch.load(STATE_DIR / f"layer_{l}.pt", weights_only=True)
        J = lens.jacobians[l].to(W_U.device)
        ids = st["top_dir_ids"]
        D_top = torch.nn.functional.normalize(
            (W_U[ids[:DOSE]] * g[None, :]) @ J, dim=1)
        Qj, _ = torch.linalg.qr(D_top.T)
        S = torch.nn.functional.normalize(
            (W_U[ids[:256]] * g[None, :]) @ J, dim=1)
        Qs, _ = torch.linalg.qr(S.T)
        P = st["pca_evecs"].T.to(W_U.device)
        P_perp = P - (P @ Qs) @ Qs.T
        keep = P_perp.norm(dim=1) > 0.5
        P_ok = torch.nn.functional.normalize(P_perp[keep], dim=1)
        Qn, _ = torch.linalg.qr(P_ok.T)
        rng = np.random.default_rng(1000 + l)
        R = torch.tensor(rng.standard_normal((RAND_POOL, P.shape[1])),
                         dtype=torch.float32, device=W_U.device)
        Qr, _ = torch.linalg.qr(R.T)
        out[l] = {"J": Qj.contiguous(), "nonJ": Qn.contiguous(),
                  "rand": Qr.contiguous()}
    del W_U
    return out


@torch.no_grad()
def phase_c_energy(model, bases, band):
    if EM_OUT.exists():
        saved = read_json(EM_OUT)
        log("phase c (energy match) loaded from existing file")
        return {int(l): v for l, v in saved["plan"].items()}
    recs = [json.loads(x) for x in PROMPTS.read_text().splitlines()]
    texts = [r.get("prompt") or r.get("text") for r in recs]
    acc = {l: {"n": 0, "h2": 0.0, "J": 0.0,
               "nonJ": np.zeros(bases[l]["nonJ"].shape[1]),
               "rand": np.zeros(RAND_POOL)} for l in band}
    t0 = time.time()
    for i, text in enumerate(texts):
        ids = model.encode(text, max_length=128)
        with ActivationRecorder(model.layers, at=band) as rec:
            model.forward(ids)
        for l in band:
            h = rec.activations[l][0].float()
            a = acc[l]
            a["n"] += h.shape[0]
            a["h2"] += float((h * h).sum())
            p = h @ bases[l]["J"][:, :DOSE]
            a["J"] += float((p * p).sum())
            for pool in ("nonJ", "rand"):
                p = h @ bases[l][pool]
                a[pool] += (p * p).sum(0).cpu().numpy()
        if i % 50 == 49:
            log(f"phase c energy pass {i + 1}/{len(texts)} "
                f"({time.time() - t0:.0f}s)")
    em, plan = {}, {l: {} for l in band}
    for l in band:
        a = acc[l]
        tgt = a["J"] / a["n"]
        em[l] = {"n_pos": a["n"], "mean_h2": a["h2"] / a["n"],
                 "E_jspan_k20": tgt,
                 "share_of_h2": tgt / (a["h2"] / a["n"])}
        for pool in ("nonJ", "rand"):
            e = a[pool] / a["n"]
            cum = np.cumsum(e)
            m2 = int(np.argmin(np.abs(cum - tgt))) + 1
            ratio = float(cum[m2 - 1] / tgt)
            method, cols = "prefix", list(range(m2))
            if pool == "nonJ" and not (0.8 <= ratio <= 1.25):
                best = (1e9, None)
                for s in range(len(e)):
                    c = np.cumsum(e[s:])
                    j2 = int(np.argmin(np.abs(np.log(c / tgt))))
                    r2 = float(c[j2] / tgt)
                    key = abs(np.log(r2))
                    if key < best[0] - 1e-12:
                        best = (key, (s, s + j2 + 1, r2))
                s, t, r2 = best[1]
                method, cols, ratio = f"window[{s}:{t}]", list(range(s, t)), r2
            em[l][f"{pool}_m"] = len(cols)
            em[l][f"{pool}_ratio"] = ratio
            em[l][f"{pool}_method"] = method
            plan[l][f"vmatch_{pool}_k{DOSE}"] = cols
    atomic_write_json({"band": band, "dose": DOSE, "rand_pool": RAND_POOL,
                       "corpus": "v1 descriptive_prompts.jsonl (raw h)",
                       "per_layer": em, "plan": plan}, EM_OUT)
    for l in band[:3] + band[-1:]:
        log(f"phase c L{l}: nonJ m={em[l]['nonJ_m']} r={em[l]['nonJ_ratio']:.2f} "
            f"({em[l]['nonJ_method']}) rand m={em[l]['rand_m']} "
            f"r={em[l]['rand_ratio']:.2f}")
    return plan


def phase_d_grid(model, hf, tok, lens, band, plan, conds=CONDS, tasks_sel=TASKS):
    """PLAN_v3 restage: callable per stage so the marquee frozen cells run
    BEFORE the static cells (which alone need phase b/c). Builds only the
    machinery the requested conds use."""
    subs = {}
    if any(c.startswith(("jspace", "vmatch")) for c in conds):
        bases = build_bases(lens, hf, band)
        subs = {
            "jspace_k20": {l: bases[l]["J"][:, :DOSE].contiguous() for l in band},
            "vmatch_rand_k20": {
                l: bases[l]["rand"][:, plan[l][f"vmatch_rand_k{DOSE}"]].contiguous()
                for l in band},
            "vmatch_nonJ_k20": {
                l: bases[l]["nonJ"][:, plan[l][f"vmatch_nonJ_k{DOSE}"]].contiguous()
                for l in band},
        }
        del bases
    jd = rd = None
    if any(c.startswith("frozen") for c in conds):
        jd, rd = build_dicts(lens, hf, band)
    rng = np.random.default_rng(0)
    tasks = s7.build_tasks(rng)
    # PLAN_v3 Q ns: twohop 30, onehop 30, arith 15 (sql kept only for the
    # samples audit); recorded here so the grid file is self-describing.
    tasks["twohop"] = tasks["twohop"][:30]
    tasks["onehop"] = tasks["onehop"][:30]
    tasks["arithmetic"] = tasks["arithmetic"][:15]
    res = read_json(GRID_OUT) if GRID_OUT.exists() else {
        "model": MODEL_ID, "band": band, "dose": DOSE,
        "lens": f"{LENS_REPO}/{LENS_FILE}",
        "ns": {"twohop": 30, "onehop": 30, "arithmetic": 15, "prose_nll": 20},
        "conditions": {}}
    ab = BandAblator(model.layers, band)
    with ab:
        for cond in conds:
            res["conditions"].setdefault(cond, {})
            for tname in tasks_sel:
                if tname in res["conditions"][cond] and "--force" not in sys.argv:
                    continue
                t0 = time.time()
                extra = {}
                if cond in ("frozen_j10", "frozen_rand10"):
                    dicts = jd if cond == "frozen_j10" else rd
                    if tname == "samples":
                        ab.mode = ("static", frozen_projectors_band(
                            model, dicts, tasks["arithmetic"][0]["prompt"],
                            band))
                        scores = s7.run_task("samples", cond, model, hf, tok,
                                             tasks, extra)
                        ab.mode = None
                    else:
                        key = {"twohop_lp": "twohop",
                               "arithmetic_v2": "arithmetic"}.get(tname, tname)
                        scores = []
                        for it in tasks[key]:
                            ab.mode = None
                            p = (it.get("prompt") or it.get("text")
                                 or it.get("good"))
                            ab.mode = ("static", frozen_projectors_band(
                                model, dicts, p, band))
                            scores += s7.run_task(tname, cond, model, hf,
                                                  tok, {key: [it]}, extra)
                            ab.mode = None
                else:
                    ab.mode = None if cond == "none" else ("static", subs[cond])
                    scores = s7.run_task(tname, cond, model, hf, tok, tasks,
                                         extra)
                    ab.mode = None
                entry = s7.boot_ci(scores)
                entry["seconds"] = round(time.time() - t0)
                entry.update(extra)
                res["conditions"][cond][tname] = entry
                atomic_write_json(res, GRID_OUT)
                log(f"{cond:>16} {tname:>13}: {entry['mean']:.3f} "
                    f"[{entry['ci_lo']:.3f},{entry['ci_hi']:.3f}] "
                    f"({entry['seconds']}s)")
    log(f"wrote {GRID_OUT}")


def main() -> None:
    seed_all()
    M.mkdir(parents=True, exist_ok=True)
    model, hf, tok, lens, band, d = load_qwen()
    phase_a_sanity(model, tok, lens)
    # PLAN_v3 staged order: Q1 marquee frozen cells FIRST (they need no
    # phase-b/c machinery — if the VM dies early, the model-vs-harness
    # verdict cells are already banked); Q2 static energy-matched cells
    # after, twohop_lp only.
    phase_d_grid(model, hf, tok, lens, band, plan=None,
                 conds=("none", "frozen_j10", "frozen_rand10"),
                 tasks_sel=("twohop", "twohop_lp", "onehop", "arithmetic_v2",
                            "prose_nll", "grammar", "samples"))
    phase_b_descriptive(model, hf, lens, band, d)
    bases = build_bases(lens, hf, band)
    plan = phase_c_energy(model, bases, band)
    del bases
    phase_d_grid(model, hf, tok, lens, band, plan,
                 conds=("jspace_k20", "vmatch_rand_k20", "vmatch_nonJ_k20"),
                 tasks_sel=("twohop_lp",))
    log("s18 done")


if __name__ == "__main__":
    main()
