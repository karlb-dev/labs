# N3 / D2 / HM1 — what does averaging the Jacobian actually cost?
#
# THE QUESTION. jlens fits
#     J[i,j] = mean_over_prompts mean_over_positions_s [ sum_t
#              d h_tgt[t,i] / d h_src[s,j] ]
# and `linearization-faithfulness-olmo3-think-v2` measured that this
# object predicts the true response with only ~0.49 cosine in the paper's
# own band, on a model whose transport IS linear there
# (`local-linearity-v3-olmo3-think`). So the gap is ESTIMATION error, and
# the campaign's H7 asks which averaging causes it.
#
# THE DESIGN POINT THAT MAKES THIS MEASURABLE. For a perturbation applied
# UNIFORMLY at every valid position, the per-position-averaged estimator
# and the corpus-averaged estimator are algebraically identical:
#     sum_s J_s @ d  ==  P * (mean_s J_s) @ d.
# Position-averaging loss is therefore INVISIBLE to a uniform probe — and
# the earlier faithfulness test used exactly a uniform probe. It becomes
# visible only for position-VARYING perturbations, which is precisely what
# the dynamic top-k ablation does (each position has its own selected
# rows). This module therefore perturbs ONE source position at a time.
#
# THREE ESTIMATORS (PI addendum §3-D2 amendment (a)):
#   (i)   context-J    : this prompt, this position — the exact local
#                        first-order response. Measured, not fitted; it is
#                        the CEILING any Jacobian could reach.
#   (ii)  position-J   : per-position, averaged over OTHER prompts
#                        (leave-one-out, so it is a genuine prediction).
#                        Content-agnostic but position-aware.
#   (iii) campaign lens: the fitted mean-J the campaign uses.
# (i)->(ii) isolates PROMPT-averaging loss; (ii)->(iii) isolates
# POSITION-averaging loss.
#
# GROUND TRUTH is a finite difference at a perturbation scale the model
# was independently shown to treat linearly (eps_rel 0.1; superposition
# ratio 1.98-2.02 across this band), and every cell carries its own
# linearity check r(2d)/r(d) so a nonlinear cell is reported as such
# rather than blamed on an estimator.
#
# DECISION RULE, COMMITTED BEFORE THE RUN (nextsteps_2_2 §7-N3.3).
# A context-specific methods arm is admitted to the preregistration only
# if, on dev prompts:
#   1. median response cosine improves by >= 0.20 over the campaign lens;
#   2. reaches >= 0.80 on at least two of the three band layers;
#   3. the gain is not confined to a single prompt;
#   4. no confirmatory item is involved (dev prose prompts only).
# Failing the rule is a publishable result: it says context conditioning
# does not repair the mismatch under the tested estimator.
#
# Tier: methods-pilot. Dev prompts only. ~1 GPU hour on a 32B model.
# Usage: python -m jspace_part2.experiments.h7_context_j
#          [--model olmo3-think] [--quick] [--allow-dirty]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ..lib import sha256_file
from ..paths import to_uri
from ..provenance import (Provenance, registry_append, require_clean_tree,
                          resolve_model, write_result_v2)

RUN = Path("/content/drive/MyDrive/interpret/special-lab-1/part2_20260727")
CFG = {
    "olmo3-think": {
        "model": "/content/models/olmo3-think",
        "lens": ("/content/drive/MyDrive/interpret/special-lab-1/"
                 "2026-07-25_1726/lens/olmo32bthink_lens.pt")},
}
# Dev prose prompts: general text, no factual probe, nothing that can
# appear in a confirmatory item. All truncated to SEQ so that a position
# INDEX is comparable across prompts (estimator (ii) needs that).
PROMPTS = [
    "The history of astronomy begins with the earliest civilizations that tracked the motions of the sun and moon across the night sky and recorded what they saw in long tables of numbers, which later observers compared against their own measurements to detect slow changes",
    "In modern computing, the distinction between memory and storage has shaped how programs are written, because volatile memory loses its contents whenever the machine loses power, while persistent storage survives a restart at the cost of much slower access",
    "When a large star exhausts its nuclear fuel, the core collapses and the outer layers rebound outward in an explosion that briefly outshines an entire galaxy of ordinary stars, scattering heavy elements across the surrounding interstellar medium",
    "The development of printing changed how knowledge circulated, since a single setting of type could produce many identical copies, and errors that once crept in with each hand transcription were instead reproduced faithfully or corrected once for everyone",
    "Ocean currents move heat around the planet in vast circulating patterns driven by wind, temperature and salinity differences, and small changes in those patterns can alter the climate of regions thousands of kilometres away from where the change began",
    "A compiler translates source text into machine instructions through a sequence of passes, each of which rewrites an intermediate representation into a form closer to the target machine while preserving the meaning the programmer intended to express",
    "Medieval cathedral builders worked without the mathematics later engineers would use, relying instead on accumulated craft rules, scale models and a willingness to rebuild sections that showed signs of failure during the decades a single building took",
    "The immune system distinguishes self from non-self through an elaborate system of receptors and signals, and its failures produce both autoimmune disease, where the body attacks itself, and immune evasion, where a pathogen escapes detection entirely",
    "Cartographers face an unavoidable problem: the surface of a sphere cannot be flattened onto a plane without distortion, so every map projection preserves some properties, such as area or angle, only by sacrificing others somewhere on the sheet",
    "Musical tuning systems reflect a compromise, because the simple frequency ratios that sound most consonant cannot all be preserved simultaneously across every key, and different cultures resolved that tension in strikingly different ways over time",
    "Early photographic processes required exposures measured in minutes, which is why portraits from that era show sitters braced against hidden supports, and why streets in city photographs appear deserted despite having been crowded at the time",
    "Sediment layers record the history of a landscape, since each deposit preserves evidence of the conditions under which it formed, and an unconformity where layers are missing marks an interval during which material was eroded rather than laid down",
]
# Fixed length so a position INDEX is comparable across prompts (estimator
# (ii) needs that). Set from the shortest prompt at runtime; every prompt
# is truncated to it, so no prompt is padded.
SEQ_MAX = 96
SKIP_FIRST = 16          # jlens default; matches the fitted lens
POSITION_FRACS = [0.55, 0.80, 1.00]   # early-valid, mid, final (of SEQ)
# PERTURBATION SCALE IS CALIBRATED, NOT ASSUMED. The published local-
# linearity result (ratio 1.98-2.02 at eps 0.1-0.2) was measured for a
# UNIFORM perturbation. A single-position perturbation is a different
# probe and turns out to be markedly more nonlinear at the same scale
# (measured here: r(2d)/r(d) = 2.2-2.95 at eps 0.10) — unsurprising in
# hindsight, because shifting one position's residual moves that
# position's key/value and so re-weights the attention softmax at every
# later position, whereas a uniform shift partly cancels inside the
# softmax. Attributing an estimator gap to averaging while the ground
# truth itself is nonlinear would repeat the withdrawn-v1 mistake, so the
# run first sweeps eps and selects the largest scale whose response is
# linear within tolerance AND whose input is delivered faithfully in bf16.
EPS_GRID = [0.005, 0.01, 0.02, 0.05, 0.10]
LIN_TOL = 0.10           # accept |r(2d)/r(d) - 2| <= LIN_TOL
FIDELITY_MIN = 0.99      # measured ||delivered delta|| / ||intended delta||
N_RANDOM = 8
N_JROW = 4               # directions taken from the lens's own top rows
N_LOGIT = 4              # unembedding-aligned directions
SEED = 4242
# committed decision thresholds
IMPROVE_MIN = 0.20
ABS_MIN = 0.80
BAND = [24, 32, 40]
LATE_CONTROL = 56


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    git = require_clean_tree("--allow-dirty" in sys.argv)
    slug = arg("--model", "olmo3-think")
    quick = "--quick" in sys.argv
    cfg = CFG[slug]
    out = RUN / "metrics" / slug / "h7_context_j.json"
    ckpt = Path("/content/sl1_work") / f"h7_{slug}.ckpt.json"
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    import transformers
    import jlens
    from jlens import JacobianLens
    from jlens.hooks import ActivationRecorder

    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(cfg["model"])
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16).to("cuda").eval()
    model = jlens.from_hf(hf, tok)
    lens = JacobianLens.load(cfg["lens"])
    layers = [L for L in (BAND + [LATE_CONTROL]) if L in lens.jacobians]
    target = model.n_layers - 1
    prompts = PROMPTS[:4] if quick else PROMPTS
    lens_ = [len(tok(p).input_ids) for p in prompts]
    SEQ = min(min(lens_), SEQ_MAX)
    if SEQ <= SKIP_FIRST + 4:
        raise SystemExit(f"prompts too short: SEQ={SEQ} vs SKIP_FIRST={SKIP_FIRST}")
    positions = sorted({min(SEQ - 1, max(SKIP_FIRST, int(f * SEQ) - 1))
                        for f in POSITION_FRACS})
    if quick:
        positions = positions[-1:]
    print(f"layers {layers} · target {target} · {len(prompts)} prompts · "
          f"SEQ {SEQ} (token lengths {min(lens_)}-{max(lens_)}) · "
          f"positions {positions}", flush=True)

    rng = np.random.default_rng(SEED)
    done = {}
    if ckpt.exists():
        done = json.loads(ckpt.read_text())
        print(f"resuming: {len(done)} cells banked", flush=True)

    @torch.no_grad()
    def responses(ids, L, pos, deltas):
        """Measured change in the SUMMED target activation for each delta,
        applied at source position `pos`. Batched over deltas."""
        B = len(deltas)
        big = ids.expand(B, -1).contiguous()
        D = torch.stack(deltas).to("cuda")

        def hook(mod, o_in, o):
            h = o[0] if not torch.is_tensor(o) else o
            h = h.clone()
            h[:, pos] = h[:, pos] + D.to(h.dtype)
            return h if torch.is_tensor(o) else (h, *o[1:])

        hd = model.layers[L].register_forward_hook(hook)
        with ActivationRecorder(model.layers, at=[target]) as r:
            model.forward(big)
        hd.remove()
        act = r.activations[target].float()          # [B, T, d]
        return act[:, SKIP_FIRST:].sum(dim=1)        # [B, d]

    @torch.no_grad()
    def baseline(ids, B):
        big = ids.expand(B, -1).contiguous()
        with ActivationRecorder(model.layers, at=[target]) as r:
            model.forward(big)
        return r.activations[target].float()[:, SKIP_FIRST:].sum(dim=1)

    ids_all = {p: model.encode(t, max_length=SEQ) for p, t in enumerate(prompts)}
    ids_all = {p: i for p, i in ids_all.items() if i.shape[1] == SEQ}
    print(f"{len(ids_all)}/{len(prompts)} prompts reached the full {SEQ} tokens",
          flush=True)

    # ---------------------------------------------------------------
    # EPS CALIBRATION. Pick, per layer, the largest perturbation scale
    # that is (a) delivered faithfully in bf16 and (b) responded to
    # linearly, so the estimator comparison is not contaminated by
    # curvature in the ground truth itself.
    @torch.no_grad()
    def input_fidelity(ids, L, pos, delta):
        """||actually delivered delta|| / ||intended delta|| at the source."""
        got = {}

        def hook(mod, o_in, o):
            h = o[0] if not torch.is_tensor(o) else o
            h2 = h.clone()
            h2[:, pos] = h2[:, pos] + delta.to(h.dtype)
            got["d"] = (h2[0, pos].float() - h[0, pos].float())
            return h2 if torch.is_tensor(o) else (h2, *o[1:])

        hd = model.layers[L].register_forward_hook(hook)
        model.forward(ids)
        hd.remove()
        return float(got["d"].norm() / delta.norm())

    calib_pos = positions[-1]
    calib_prompts = list(ids_all)[:2]
    calibration, eps_by_layer = [], {}
    for L in layers:
        g = torch.Generator().manual_seed(SEED + L)
        u = torch.randn(lens.jacobians[L].shape[1], generator=g)
        u = (u / u.norm()).to("cuda", torch.float32)
        chosen = None
        for eps in EPS_GRID:                      # ascending
            ratios, fids = [], []
            for p in calib_prompts:
                ids = ids_all[p]
                with ActivationRecorder(model.layers, at=[L]) as r0:
                    with torch.no_grad():
                        model.forward(ids)
                hn = float(r0.activations[L][0, SKIP_FIRST:].float()
                           .norm(dim=-1).mean())
                d = u * (eps * hn)
                b = baseline(ids, 2)
                rr = responses(ids, L, calib_pos, [d, d * 2.0]) - b
                n1 = float(rr[0].norm())
                ratios.append(float(rr[1].norm() / n1) if n1 > 0 else float("nan"))
                fids.append(input_fidelity(ids, L, calib_pos, d))
            row = {"layer": L, "eps": eps,
                   "scale_ratio": round(float(np.nanmean(ratios)), 3),
                   "input_fidelity": round(float(np.mean(fids)), 4)}
            row["linear"] = bool(abs(row["scale_ratio"] - 2.0) <= LIN_TOL)
            row["measurable"] = bool(row["input_fidelity"] >= FIDELITY_MIN)
            calibration.append(row)
            if row["linear"] and row["measurable"]:
                chosen = eps                      # keep the LARGEST that qualifies
        eps_by_layer[L] = chosen
        print(f"  calib L{L}: " + " ".join(
            f"{r['eps']}:{r['scale_ratio']}{'*' if r['linear'] and r['measurable'] else ''}"
            for r in calibration if r["layer"] == L)
            + f"  -> eps={chosen}", flush=True)
    if all(v is None for v in eps_by_layer.values()):
        print("NO LINEAR WINDOW at any tested scale for a single-position "
              "probe — reporting that as the result.", flush=True)

    # ---- directions per layer (fixed across prompts so (ii) is comparable)
    dirs_by_layer = {}
    for L in layers:
        J = lens.jacobians[L].to("cuda", torch.float32)
        d_model = J.shape[1]
        ds, kinds = [], []
        for _ in range(N_RANDOM):
            g = torch.Generator().manual_seed(int(rng.integers(0, 2**31)))
            u = torch.randn(d_model, generator=g)
            ds.append((u / u.norm()).to("cuda")), kinds.append("random")
        rows = J.norm(dim=1).topk(N_JROW).indices
        for i in rows:
            u = J[i].clone()
            ds.append(u / u.norm()), kinds.append("jrow")
        W = model.unembed_matrix() if hasattr(model, "unembed_matrix") else None
        if W is None:
            W = hf.get_output_embeddings().weight
        widx = rng.integers(0, W.shape[0], N_LOGIT)
        for i in widx:
            u = W[int(i)].float().to("cuda")
            ds.append(u / u.norm()), kinds.append("logit")
        dirs_by_layer[L] = (ds, kinds)
        del J
        torch.cuda.empty_cache()

    # ---- measure every (prompt, layer, position) cell
    measured = {}          # (L,pos,p) -> [n_dirs, d] tensor of responses
    lin_checks = []
    for L in layers:
        ds, kinds = dirs_by_layer[L]
        eps_L = eps_by_layer[L] or EPS_GRID[0]
        for pos in positions:
            for p, ids in ids_all.items():
                key = f"{L}|{pos}|{p}"
                if key in done:
                    measured[key] = torch.tensor(done[key]["resp"], device="cuda")
                    continue
                with ActivationRecorder(model.layers, at=[L]) as r0:
                    with torch.no_grad():
                        model.forward(ids)
                hn = float(r0.activations[L][0, SKIP_FIRST:].float()
                           .norm(dim=-1).mean())
                scale = eps_L * hn
                deltas = [d * scale for d in ds]
                base = baseline(ids, len(deltas))
                resp = responses(ids, L, pos, deltas) - base
                # linearity check for EVERY direction, not just one: the
                # calibration showed the scale ratio is direction-dependent
                # (1.93 to 2.95 at the same layer, position and eps), so a
                # single-direction check would misreport the ceiling.
                r2 = responses(ids, L, pos, [d * 2.0 for d in deltas]) - base
                n1 = resp.norm(dim=1)
                ratios = (r2.norm(dim=1) / n1.clamp_min(1e-9)).cpu().tolist()
                ratio = float(np.median(ratios))
                lin_checks.append({
                    "layer": L, "pos": pos, "prompt": p, "eps": eps_L,
                    "scale_ratio_median": round(ratio, 3),
                    "scale_ratio_min": round(float(np.min(ratios)), 3),
                    "scale_ratio_max": round(float(np.max(ratios)), 3),
                    "by_kind": {k: round(float(np.median(
                        [x for x, kk in zip(ratios, kinds) if kk == k])), 3)
                        for k in set(kinds)}})
                measured[key] = resp
                done[key] = {"resp": resp.cpu().tolist(),
                             "scale_ratio": ratio, "ratios": ratios,
                             "h_norm": hn, "eps": eps_L}
                ckpt.write_text(json.dumps(done))
                print(f"  L{L} pos{pos} prompt{p}: |r|={float(resp.norm(dim=1).mean()):.3f} "
                      f"lin={ratio:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- compare the three estimators
    recs = []
    for L in layers:
        ds, kinds = dirs_by_layer[L]
        J = lens.jacobians[L].to("cuda", torch.float32)
        for pos in positions:
            keys = [f"{L}|{pos}|{p}" for p in ids_all]
            for p in ids_all:
                key = f"{L}|{pos}|{p}"
                truth = measured[key]                       # [n_dirs, d]
                # (ii) leave-one-out mean over OTHER prompts, same position
                others = [measured[k] for k in keys if k != key]
                pos_j = torch.stack(others).mean(dim=0)
                for di, (d, kind) in enumerate(zip(ds, kinds)):
                    hn, eps_used = done[key]["h_norm"], done[key]["eps"]
                    delta = d * (eps_used * hn)
                    pred_lens = J @ delta                    # (iii)
                    t_ = truth[di]
                    for name, pred in (("campaign_meanJ", pred_lens),
                                       ("position_J_loo", pos_j[di])):
                        tn, pn = float(t_.norm()), float(pred.norm())
                        cos = float(torch.nn.functional.cosine_similarity(
                            t_, pred, dim=0)) if tn > 0 and pn > 0 else 0.0
                        recs.append({
                            "estimator": name, "layer": L, "pos": pos,
                            "prompt": p, "dir": di, "kind": kind,
                            "cos": round(cos, 4),
                            "norm_ratio": round(pn / tn, 4) if tn > 0 else None,
                            "rel_err": round(float((t_ - pred).norm() / tn), 4)
                            if tn > 0 else None})
        del J
        torch.cuda.empty_cache()

    # ---- summarise
    def med(est, L=None, pos=None, prompt=None, kind=None, field="cos"):
        sub = [r[field] for r in recs if r["estimator"] == est
               and (L is None or r["layer"] == L)
               and (pos is None or r["pos"] == pos)
               and (prompt is None or r["prompt"] == prompt)
               and (kind is None or r["kind"] == kind)
               and r[field] is not None]
        return round(float(np.median(sub)), 4) if sub else None

    by_layer = {str(L): {e: {"cos": med(e, L=L),
                             "norm_ratio": med(e, L=L, field="norm_ratio"),
                             "rel_err": med(e, L=L, field="rel_err")}
                         for e in ("campaign_meanJ", "position_J_loo")}
                for L in layers}
    by_kind = {k: {e: med(e, kind=k) for e in
                   ("campaign_meanJ", "position_J_loo")}
               for k in ("random", "jrow", "logit")}
    by_prompt = {str(p): {e: med(e, prompt=p) for e in
                          ("campaign_meanJ", "position_J_loo")}
                 for p in ids_all}
    by_pos = {str(ps): {e: med(e, pos=ps) for e in
                        ("campaign_meanJ", "position_J_loo")}
              for ps in positions}

    band_present = [L for L in BAND if L in layers]
    improves = [(by_layer[str(L)]["position_J_loo"]["cos"] or 0)
                - (by_layer[str(L)]["campaign_meanJ"]["cos"] or 0)
                for L in band_present]
    abs_ok = sum(1 for L in band_present
                 if (by_layer[str(L)]["position_J_loo"]["cos"] or 0) >= ABS_MIN)
    prompt_gains = [(v["position_J_loo"] or 0) - (v["campaign_meanJ"] or 0)
                    for v in by_prompt.values()]
    not_one_prompt = sum(1 for g in prompt_gains if g > 0) > len(prompt_gains) / 2
    passes = (float(np.median(improves)) >= IMPROVE_MIN and abs_ok >= 2
              and not_one_prompt)

    lin_med = float(np.median([c["scale_ratio_median"] for c in lin_checks
                               if np.isfinite(c["scale_ratio_median"])]))
    payload = {
        "model": slug, "seq_len": SEQ,
        "eps_calibration": calibration, "eps_by_layer": eps_by_layer,
        "positions": positions, "layers": layers, "n_prompts": len(ids_all),
        "n_directions": len(dirs_by_layer[layers[0]][0]),
        "decision_rule": {"improve_min": IMPROVE_MIN, "abs_min": ABS_MIN,
                          "band": band_present,
                          "committed_before_run": True},
        "linearity_check": {"median_scale_ratio_r2d_over_rd": round(lin_med, 3),
                            "cells": lin_checks,
                            "note": ("2.00 = exactly linear; this is the "
                                     "ceiling any first-order model can reach")},
        "by_layer": by_layer, "by_position": by_pos,
        "by_direction_kind": by_kind, "by_prompt": by_prompt,
        "verdict": {
            "context_arm_passes_dev_gate": bool(passes),
            "median_band_improvement": round(float(np.median(improves)), 4),
            "band_layers_reaching_abs_min": abs_ok,
            "gain_in_majority_of_prompts": bool(not_one_prompt)},
        "records": recs,
    }
    if quick:
        # a smoke run is not evidence
        print(json.dumps(payload["verdict"], indent=1))
        print(f"[--quick] smoke run: NOT registered. seconds "
              f"{round(time.time() - t0)}")
        return
    prov = Provenance(
        evidence_id=f"h7-context-j-{slug}-v2", tier="pilot",
        command=f"python -m jspace_part2.experiments.h7_context_j --model {slug}",
        inputs={"lens": sha256_file(cfg["lens"])},
        model=resolve_model(cfg["model"]), seed=SEED)
    env = write_result_v2(payload, out, prov)
    registry_append({
        "evidence_id": f"h7-context-j-{slug}-v2", "tier": "pilot",
        "what": (f"H7/D2 decomposition of the averaged-Jacobian mismatch on "
                 f"{slug}, using POSITION-SPECIFIC perturbations (a uniform "
                 f"probe cannot see position-averaging loss: sum_s J_s@d == "
                 f"P*(mean_s J_s)@d). Median response cosine by layer: " +
                 json.dumps({L: {e: by_layer[L][e]["cos"] for e in by_layer[L]}
                             for L in by_layer}) +
                 f". Linearity ceiling (r(2d)/r(d)) = {lin_med:.2f}. "
                 f"Dev decision rule committed before the run "
                 f"(improve>={IMPROVE_MIN}, abs>={ABS_MIN} on >=2 band "
                 f"layers): context arm "
                 f"{'PASSES' if passes else 'DOES NOT PASS'}."),
        "command": prov.command, "code_commit": git["code_commit"],
        "rerun": "auto",
        "input_uris": {"lens": to_uri(cfg["lens"])},
        "inputs": {"lens": sha256_file(cfg["lens"])},
        "outputs": [{"path": str(out), "uri": to_uri(str(out)),
                     "sha256": sha256_file(out),
                     "payload_sha256": env["payload_sha256"]}],
        "repro_notes": ("Dev prose prompts only; no confirmatory item is "
                        "involved. Resumable: per-cell checkpoint at "
                        f"{ckpt}.")})
    print(json.dumps({k: payload[k] for k in
                      ("by_layer", "by_position", "by_direction_kind",
                       "verdict", "linearity_check")},
                     indent=1, default=str)[:4000])
    print(f"\nseconds {round(time.time() - t0)}")


if __name__ == "__main__":
    main()
