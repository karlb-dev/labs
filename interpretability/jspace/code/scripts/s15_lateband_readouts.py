# v2 Priority 3b: redo the key readouts on the LATE band {46,50,54,58,62}
# with the late lens from s14. Three phases:
#   a) s5-equivalent descriptive aggregates on the v1 descriptive prompt set
#      (variance share, active counts, persistence, top-direction state) ->
#      metrics/descriptive_late.json + metrics/layer_state_late/
#      Decides: does variance share keep rising into the late band, or peak?
#   b) s8-equivalent answer-rank profiles at the late layers for all 90 CoT
#      items (think prompt + suppressed prompt, J-lens vs logit lens) ->
#      metrics/late_answer_profile.json
#      Decides: is OLMo's workspace late-shifted relative to the fitted band?
#   c) (--with-cot) regenerate the 90 greedy traces (deterministic, identical
#      token sequences to v1) reading the late dictionaries per step ->
#      metrics/cot_lead_late.json (late-band lead vs v1's mid-band 46 steps)
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl1_common import (RUN_DIR, RUN_DIR_V2, atomic_write_json, die,
                        load_model, log, read_json, seed_all,
                        variant_first_ids)

import numpy as np
import torch
from jlens import JacobianLens
from jlens.hooks import ActivationRecorder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s5_descriptive import gradient_pursuit  # noqa: E402
import s8_cot as s8  # noqa: E402  (build_items, readout_prompt)

LATE = [46, 50, 54, 58, 62]
LENS_PATH = RUN_DIR_V2 / "lens" / "olmo32bthink_late.pt"
PROMPTS = RUN_DIR / "config" / "prompts" / "descriptive_prompts.jsonl"
AGG_OUT = RUN_DIR_V2 / "metrics" / "descriptive_late.json"
STATE_DIR = RUN_DIR_V2 / "metrics" / "layer_state_late"
PROF_OUT = RUN_DIR_V2 / "metrics" / "late_answer_profile.json"
LEAD_OUT = RUN_DIR_V2 / "metrics" / "cot_lead_late.json"
BATCH, SKIP_FIRST, MAX_SEQ = 20, 16, 256
K_MAX = 50
REL_THRESHOLDS = (0.01, 0.02, 0.05)


class LateStepReader:
    def __init__(self, layers):
        self.layers = layers
        self.h = {}
        self._handles = []

    def __enter__(self):
        for l in LATE:
            def fn(mod, inp, out, l=l):
                t = out[0] if not torch.is_tensor(out) else out
                self.h[l] = t[:, -1, :].detach().float()
            self._handles.append(self.layers[l].register_forward_hook(fn))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()


def phase_a(model, hf, lens):
    prompts = [json.loads(l) for l in PROMPTS.read_text().splitlines()]
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    agg = read_json(AGG_OUT) if AGG_OUT.exists() else {
        "lens_file": LENS_PATH.name, "layers": LATE, "k_max": K_MAX,
        "rel_thresholds": list(REL_THRESHOLDS), "batches_done": 0,
        "per_layer": {str(l): {
            "n_positions": 0, "sum_recon_sq_c": 0.0, "sum_h_sq_c": 0.0,
            "active_counts": {str(t): [] for t in REL_THRESHOLDS},
            "top1_persist": [0, 0], "dir_activation_sum": {},
        } for l in LATE}}
    n_batches = (len(prompts) + BATCH - 1) // BATCH
    if agg["batches_done"] >= n_batches:
        log("phase a already complete")
        return
    W_U = hf.lm_head.weight.detach()
    g = hf.model.norm.weight.detach().float()
    moments = {l: {"n": 0, "sum": torch.zeros(5120, dtype=torch.float64),
                   "cov": torch.zeros(5120, 5120, dtype=torch.float64)}
               for l in LATE}
    for b in range(agg["batches_done"], n_batches):
        t0 = time.time()
        chunk = prompts[b * BATCH:(b + 1) * BATCH]
        acts = {l: [] for l in LATE}
        metas = []
        for p in chunk:
            ids = model.encode(p["text"], max_length=MAX_SEQ)
            with ActivationRecorder(model.layers, at=LATE) as rec:
                with torch.no_grad():
                    model.forward(ids)
            P = ids.shape[1]
            lo = min(SKIP_FIRST, max(P - 8, 1))
            for l in LATE:
                acts[l].append(rec.activations[l][0, lo:P - 1].float())
            metas.append({"n_pos": P - 1 - lo})
        for l in LATE:
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
            del D, H, recon
        agg["batches_done"] = b + 1
        atomic_write_json(agg, AGG_OUT)
        log(f"phase a batch {b + 1}/{n_batches} ({time.time() - t0:.0f}s)")
    for l in LATE:
        n = moments[l]["n"]
        if n == 0:
            continue  # resumed run: PCA state only for freshly-seen batches
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
    atomic_write_json(agg, AGG_OUT)
    for l in LATE:
        pl = agg["per_layer"][str(l)]
        vs = pl["sum_recon_sq_c"] / max(pl["sum_h_sq_c"], 1e-9)
        log(f"phase a L{l}: variance share {vs:.4f}")


def phase_b(model, lens, tok):
    rng = np.random.default_rng(0)
    items = s8.build_items(rng)
    res = read_json(PROF_OUT) if PROF_OUT.exists() else {
        "layers": LATE, "items": {}}
    for it in items:
        key = str(it["iid"])
        if key in res["items"]:
            continue
        t0 = time.time()
        ans = variant_first_ids(tok, it["answer"])
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": it["question"]}],
            tokenize=False, add_generation_prompt=True)
        pre = s8.readout_prompt(model, lens, None, rendered, {"answer": ans})
        sup = s8.readout_prompt(model, lens, None,
                                rendered + "\n\n</think>\n\n", {"answer": ans})
        res["items"][key] = {"kind": it["kind"],
                             "think": pre["answer"],
                             "suppressed": sup["answer"],
                             "seconds": round(time.time() - t0)}
        atomic_write_json(res, PROF_OUT)
        if it["iid"] % 15 == 0:
            log(f"phase b item {it['iid']}")
    res = read_json(PROF_OUT)  # JSON round-trip: layer keys uniformly str
    prof = {}
    for name, path in (("think_jlens", ("think", "jlens_rank_by_layer")),
                       ("think_logit", ("think", "logit_rank_by_layer")),
                       ("sup_jlens", ("suppressed", "jlens_rank_by_layer")),
                       ("sup_logit", ("suppressed", "logit_rank_by_layer"))):
        prof[name] = {l: float(np.median(
            [it[path[0]][path[1]][str(l)] for it in res["items"].values()]))
            for l in LATE}
    res["median_profile"] = prof
    atomic_write_json(res, PROF_OUT)
    log("phase b profiles: " + json.dumps(prof["sup_jlens"]))


def phase_c(model, hf, lens, tok):
    rng = np.random.default_rng(0)
    items = s8.build_items(rng)
    W_U = hf.lm_head.weight.detach().float()
    g = hf.model.norm.weight.detach().float()
    dicts = {l: torch.nn.functional.normalize(
        (W_U * g[None, :]) @ lens.jacobians[l].cuda(), dim=1).half()
        for l in LATE}
    del W_U
    res = read_json(LEAD_OUT) if LEAD_OUT.exists() else {
        "layers": LATE, "items": {}}
    for it in items:
        key = str(it["iid"])
        if key in res["items"]:
            continue
        t0 = time.time()
        ans = variant_first_ids(tok, it["answer"])
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": it["question"]}],
            tokenize=False, add_generation_prompt=True)
        ids = tok(rendered, return_tensors="pt").input_ids.cuda()
        past, cur, steps = None, ids, []
        reader = LateStepReader(hf.model.layers)
        with reader, torch.no_grad():
            for _ in range(s8.MAX_THINK_TOKENS):
                out = hf(input_ids=cur, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[0, -1].argmax()
                rec = {"tok": int(nxt)}
                for l in LATE:
                    corr = (reader.h[l].half() @ dicts[l].T)[0]
                    rec[f"L{l}_top"] = corr.topk(8).indices.tolist()
                steps.append(rec)
                if nxt.item() == tok.eos_token_id:
                    break
                cur = nxt.view(1, 1)
        emerge = next((i for i, s in enumerate(steps)
                       if any(a in s[f"L{l}_top"] for a in ans
                              for l in LATE)), None)
        cum, text_step = "", None
        w = it["answer"].strip().lower().replace(" ", "")
        for i, s in enumerate(steps):
            cum += tok.decode([s["tok"]], skip_special_tokens=False)
            if w in cum.lower().replace(" ", ""):
                text_step = i
                break
        res["items"][key] = {
            "kind": it["kind"], "n_steps": len(steps),
            "emerge_late": emerge, "text_step": text_step,
            "lead_late": (None if emerge is None or text_step is None
                          else text_step - emerge),
            "seconds": round(time.time() - t0)}
        atomic_write_json(res, LEAD_OUT)
        log(f"phase c item {it['iid']:>3}: emerge={emerge} "
            f"text={text_step} ({res['items'][key]['seconds']}s)")
    leads = [v["lead_late"] for v in res["items"].values()
             if v["lead_late"] is not None]
    res["summary"] = {
        "n_scored": len(leads),
        "median_lead_late": float(np.median(leads)) if leads else None,
        "frac_ws_leads": float(np.mean([x > 0 for x in leads]))
        if leads else None,
        "det_rate": float(np.mean([v["emerge_late"] is not None
                                   for v in res["items"].values()]))}
    atomic_write_json(res, LEAD_OUT)
    log(f"phase c summary: {res['summary']}")


def main() -> None:
    seed_all()
    if not LENS_PATH.exists():
        die("late lens missing; run s14_lateband_fit.py first")
    (RUN_DIR_V2 / "metrics").mkdir(parents=True, exist_ok=True)
    lens = JacobianLens.load(str(LENS_PATH))
    model, hf, tok = load_model("main")
    phase_a(model, hf, lens)
    phase_b(model, lens, tok)
    if "--with-cot" in sys.argv:
        phase_c(model, hf, lens, tok)
    log("s15 done")


if __name__ == "__main__":
    main()
