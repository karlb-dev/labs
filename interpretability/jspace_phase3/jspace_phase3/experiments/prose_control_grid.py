# Workstream C — the missing exact control on prose (nextsteps §2.5/§7,
# first pass per addendum §6 Block A item 3).
#
# Phase 2's prose guard skipped `matched_control` because it consumes the
# J arm's per-position profile; §7.1 says that is exactly why it must be
# paired with the J arm. This grid runs, per guard-battery item:
#
#   clean -> meanJ_label_protected (log profile + overlap)
#         -> meanJ_span_safe       (log overlap)
#         -> instant_rank_energy_matched   (consumes the label profile)
#         -> prot_energy_matched           (consumes the label profile)
#         -> mechanics_random  (random dictionary, same k + protection)
#         -> logit_label_protected (unembedding dictionary, same k +
#                                   protection)
#
# Teacher-forced §7.2 guard metrics per arm, all positions: NLL/token,
# KL(clean || arm), top-1 agreement with clean, entropy delta, EOS
# probability delta. Grammar minimal pairs get per-arm preference
# margins. A small generation audit (greedy, re-forward per step so the
# prefill ablation machinery is reused unchanged) runs on a FIXED
# pre-registered subset — the first `gen_items_per_domain` items of each
# text domain, chosen by item index before any condition was viewed —
# for {baseline, label, span-safe} only: the matched controls answer the
# dose question teacher-forced, and a per-step profile-matched
# generation would be a new estimand, not a control.
#
# Usage:
#   python -m jspace_phase3.experiments.prose_control_grid --config <cfg>
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from jspace_part2.dictionaries import build_j_dictionaries, build_logit_dictionary
from jspace_part2.lib import sha256_file
from jspace_part2.paths import resolve as resolve_uri
from ..ablator3 import (Phase3JAblator, profile_from_p3log,
                        teacher_forced_matched_arm)
from ..paths3 import metrics_dir
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           resolve_model, write_result3)
from ..scoring import ScoringSession, ScoringSpec

TIER = "phase3-development"
GUARD_SPEC = ScoringSpec(max_prompt_tokens=1024)
J_ARMS = ("meanJ_label_protected", "meanJ_span_safe")
MATCHED_ARMS = ("instant_rank_energy_matched", "prot_energy_matched")
DICT_ARMS = ("mechanics_random", "logit_label_protected")
GEN_ARMS = ("baseline", "meanJ_label_protected", "meanJ_span_safe")


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@torch.no_grad()
def dist_metrics(clean: torch.Tensor, abl: torch.Tensor, tgt: torch.Tensor,
                 eos_id: int | None) -> dict:
    """§7.2 teacher-forced guard metrics. clean/abl: [T, V] float32 on one
    device; tgt: [T-1] next-token ids."""
    lc = torch.log_softmax(clean[:-1], dim=-1)
    la = torch.log_softmax(abl[:-1], dim=-1)
    idx = torch.arange(len(tgt), device=lc.device)
    pc = lc.exp()
    out = {
        "nll_clean": float(-lc[idx, tgt].mean()),
        "nll": float(-la[idx, tgt].mean()),
        "kl_from_clean": float((pc * (lc - la)).sum(-1).mean()),
        "top1_agreement": float((la.argmax(-1) == lc.argmax(-1))
                                .float().mean()),
        "entropy_delta": float((-(la.exp() * la).sum(-1)
                                + (pc * lc).sum(-1)).mean()),
    }
    if eos_id is not None:
        out["eos_p_clean"] = float(pc[:, eos_id].mean())
        out["eos_p_delta"] = float(la.exp()[:, eos_id].mean()
                                   - out["eos_p_clean"])
    return out


def repeated_bigram_frac(ids: list[int]) -> float:
    if len(ids) < 3:
        return 0.0
    bg = list(zip(ids, ids[1:]))
    return 1.0 - len(set(bg)) / len(bg)


def main():  # noqa: C901
    cfg_path = arg("--config")
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    require_clean_tree("--allow-dirty" in sys.argv)
    slug = cfg["slug"]
    out_dir = metrics_dir(slug) / "prose_grid"
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.json"
    state = (json.loads(state_path.read_text()) if state_path.exists()
             else {"done": {}, "rows": [], "grammar": [], "gens": []})

    import transformers
    import jlens
    from jlens import JacobianLens

    model_path = str(resolve_uri(cfg["model_uri"], must_exist=True))
    tok = transformers.AutoTokenizer.from_pretrained(model_path)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16).to("cuda").eval()
    model = jlens.from_hf(hf, tok)          # mutates tok -> BOS units
    sess = ScoringSession(tok, GUARD_SPEC, device="cuda")
    lens = JacobianLens.load(str(resolve_uri(cfg["lens_uri"])))
    band, k, pk = cfg["band"], cfg["k"], cfg["protect_top_k"]
    jd = build_j_dictionaries(hf, lens, band)
    ld = build_logit_dictionary(hf, band)
    V, d = jd[band[0]].shape
    g = torch.Generator().manual_seed(cfg["rand_seed"])
    rd_one = torch.nn.functional.normalize(
        torch.randn(V, d, generator=g), dim=1).to("cuda", torch.float16)
    rd = {l: rd_one for l in band}
    eos_id = tok.eos_token_id

    battery = [json.loads(l) for l in
               Path(cfg["battery_path"]).read_text().splitlines()]
    texts = [b for b in battery if b["domain"] != "grammar_pairs"]
    pairs = [b for b in battery if b["domain"] == "grammar_pairs"]
    ab = Phase3JAblator(model.layers, band)
    t0 = time.time()

    def clean_pass(ids):
        ab.mode = None
        cl = hf(input_ids=ids, use_cache=False).logits[0].float()
        return cl, cl.topk(pk, dim=-1).indices

    def j_arm_pass(ids, psets, span_safe, record_overlap):
        ab.log = type(ab.log)()
        ab.phase, ab.forward_index = "prefill", 0
        ab.mode = {"dicts": jd, "k": k, "nonneg": True,
                   "protect_sets": psets, "active_phases": {"prefill"},
                   "span_safe": span_safe, "record_overlap": record_overlap,
                   "answer_id": None}
        with ab:
            out = hf(input_ids=ids, use_cache=False).logits[0].float()
        ab.mode = None
        return out, ab.log

    def dict_arm_pass(ids, psets, dicts):
        ab.log = type(ab.log)()
        ab.phase, ab.forward_index = "prefill", 0
        ab.mode = {"dicts": dicts, "k": k, "nonneg": True,
                   "protect_sets": psets, "active_phases": {"prefill"},
                   "span_safe": False, "record_overlap": False,
                   "answer_id": None}
        with ab:
            out = hf(input_ids=ids, use_cache=False).logits[0].float()
        ab.mode = None
        return out

    def all_arm_metrics(ids, iid):
        """Run the 6 arms on pre-built ids; return per-arm metrics plus
        J-arm overlap summaries (uses the item's own clean top-k sets)."""
        clean, psets = clean_pass(ids)
        tgt = ids[0, 1:]
        res, overlap = {}, {}
        label_log = None
        for arm in J_ARMS:
            abl, jlog = j_arm_pass(ids, psets, arm == "meanJ_span_safe",
                                   record_overlap=True)
            res[arm] = dist_metrics(clean, abl, tgt, eos_id)
            overlap[arm] = jlog.overlap_summary()
            if arm == "meanJ_label_protected":
                label_log = jlog
        profile = profile_from_p3log(label_log,
                                     overlap_records=label_log.overlap)
        for variant in MATCHED_ARMS:
            logits, _ = teacher_forced_matched_arm(
                hf, model.layers, band, jd, ids, profile, variant=variant,
                protect_sets=psets,
                seed_base=cfg["rand_seed"] + abs(hash(iid)) % 10_000)
            res[variant] = dist_metrics(clean, logits.to(clean.device),
                                        tgt, eos_id)
        res["mechanics_random"] = dist_metrics(
            clean, dict_arm_pass(ids, psets, rd), tgt, eos_id)
        res["logit_label_protected"] = dist_metrics(
            clean, dict_arm_pass(ids, psets, ld), tgt, eos_id)
        return res, overlap

    # ---------------------------------------------------- text domains
    for it in texts:
        iid = it["item_id"]
        if iid in state["done"]:
            continue
        ids = sess.prompt_ids(it["text"])
        res, overlap = all_arm_metrics(ids, iid)
        row = {"item_id": iid, "domain": it["domain"],
               "n_tokens": int(ids.shape[1])}
        for arm, m in res.items():
            row.update({f"{arm}__{k2}": v for k2, v in m.items()})
        row["overlap_label_json"] = json.dumps(
            overlap["meanJ_label_protected"])
        row["overlap_spansafe_json"] = json.dumps(overlap["meanJ_span_safe"])
        state["rows"].append(row)
        state["done"][iid] = round(time.time() - t0)
        if len(state["done"]) % 5 == 0:
            state_path.write_text(json.dumps(state))
            log(f"{len(state['done'])}/{len(texts)} text items")

    # ---------------------------------------------------- grammar pairs
    for it in pairs:
        iid = it["item_id"]
        if iid in state["done"]:
            continue
        margins = {}
        for which in ("good", "bad"):
            ids = sess.prompt_ids(it[which])
            clean, psets = clean_pass(ids)
            tgt = ids[0, 1:]
            lp = {"baseline": float(torch.log_softmax(clean[:-1], -1)
                  [torch.arange(len(tgt)), tgt].sum())}
            resm, _ = all_arm_metrics(ids, iid + which)
            for arm, m in resm.items():
                lp[arm] = -m["nll"] * len(tgt)   # sum logprob of sentence
            margins[which] = lp
        row = {"item_id": iid, "pair_id": it["pair_id"],
               "phenomenon": it["phenomenon"]}
        for arm in margins["good"]:
            mg = margins["good"][arm] - margins["bad"][arm]
            row[f"{arm}__margin"] = mg
            row[f"{arm}__prefers_good"] = bool(mg > 0)
        state["grammar"].append(row)
        state["done"][iid] = round(time.time() - t0)
    state_path.write_text(json.dumps(state))
    log(f"grammar pairs done ({len(state['grammar'])})")

    # ---------------------------------------------------- generation audit
    n_gen = cfg.get("gen_items_per_domain", 3)
    max_new = cfg.get("gen_max_new", 24)
    skip = set(cfg.get("gen_skip_domains", []))
    gen_items = [it for it in texts if int(it["item_id"].split(":")[-1])
                 < n_gen and it["domain"] not in skip]

    def greedy(ids, arm):
        cur = ids.clone()
        new = []
        for _ in range(max_new):
            if arm == "baseline":
                ab.mode = None
                logits = hf(input_ids=cur, use_cache=False).logits[0]
            else:
                _, psets = clean_pass(cur)
                abl, _ = j_arm_pass(cur, psets, arm == "meanJ_span_safe",
                                    record_overlap=False)
                logits = abl
            nxt = int(logits[-1].argmax())
            new.append(nxt)
            if nxt == eos_id:
                break
            cur = torch.cat(
                [cur, torch.tensor([[nxt]], device=cur.device)], dim=1)
        return new

    for it in gen_items:
        key = f"gen:{it['item_id']}"
        if key in state["done"]:
            continue
        ids = sess.prompt_ids(it["text"])
        rec = {"item_id": it["item_id"], "domain": it["domain"]}
        for arm in GEN_ARMS:
            new = greedy(ids, arm)
            txt = tok.decode(new)
            printable = sum(c.isprintable() or c in "\n\t" for c in txt)
            rec[arm] = {
                "text": txt, "n_tokens": len(new),
                "hit_eos": bool(new and new[-1] == eos_id),
                "repeated_bigram_frac": round(repeated_bigram_frac(new), 4),
                "malformed": bool(printable < 0.98 * max(len(txt), 1))}
        state["gens"].append(rec)
        state["done"][key] = round(time.time() - t0)
        if len(state["gens"]) % 5 == 0:
            state_path.write_text(json.dumps(state))
            log(f"gen {len(state['gens'])}/{len(gen_items)}")
    state_path.write_text(json.dumps(state))

    # ---------------------------------------------------- outputs
    df = pd.DataFrame(state["rows"])
    gr = pd.DataFrame(state["grammar"])
    pq = out_dir / f"prose_grid_items_{slug}.parquet"
    pq_g = out_dir / f"prose_grid_grammar_{slug}.parquet"
    gen_p = out_dir / f"prose_grid_generations_{slug}.jsonl"
    df.to_parquet(pq)
    gr.to_parquet(pq_g)
    gen_p.write_text("".join(json.dumps(r) + "\n" for r in state["gens"]))

    arms = list(J_ARMS) + list(MATCHED_ARMS) + list(DICT_ARMS)
    summary: dict = {"n_text_items": len(df), "n_grammar_pairs": len(gr),
                     "n_generation_items": len(state["gens"]),
                     "scoring_spec": "GUARD (1024 tok, rstripped)",
                     "bos_prefixed": sess.bos_prefixed}
    for arm in arms:
        summary[arm] = {
            "nll_delta_mean": round(float(
                (df[f"{arm}__nll"] - df[f"{arm}__nll_clean"]).mean()), 4),
            "kl_mean": round(float(df[f"{arm}__kl_from_clean"].mean()), 4),
            "top1_mean": round(float(df[f"{arm}__top1_agreement"].mean()), 4),
            "grammar_pref_rate": round(float(
                gr[f"{arm}__prefers_good"].mean()), 4),
        }
    summary["baseline_grammar_pref_rate"] = round(float(
        gr["baseline__prefers_good"].mean()), 4)
    by_dom = {}
    for dom, sub in df.groupby("domain"):
        by_dom[dom] = {arm: {
            "nll_delta": round(float((sub[f"{arm}__nll"]
                                      - sub[f"{arm}__nll_clean"]).mean()), 4),
            "kl": round(float(sub[f"{arm}__kl_from_clean"].mean()), 4),
            "top1": round(float(sub[f"{arm}__top1_agreement"].mean()), 4)}
            for arm in arms}
    summary["by_domain"] = by_dom

    payload = {"config": cfg, "summary": summary, "arms": arms}
    cmd = f"python -m jspace_phase3.experiments.prose_control_grid --config {cfg_path}"
    prov = Provenance3(
        evidence_id=cfg["evidence_id"], tier=TIER, command=cmd,
        config_path=cfg_path,
        inputs={"lens": sha256_file(str(resolve_uri(cfg["lens_uri"]))),
                "battery": sha256_file(cfg["battery_path"])},
        model=resolve_model(model_path), seed=cfg["rand_seed"])
    out_json = out_dir / f"prose_grid_{slug}.json"
    write_result3(payload, out_json, prov)
    register(cfg["evidence_id"], tier=TIER, command=cmd,
             supersedes=cfg.get("supersedes"),
             what=(f"Workstream C prose exact-control grid on {slug}: "
                   f"{len(df)} guard items x 6 arms + {len(gr)} grammar "
                   f"pairs + generation audit; label NLL delta "
                   f"{summary['meanJ_label_protected']['nll_delta_mean']}, "
                   f"exact matched "
                   f"{summary['instant_rank_energy_matched']['nll_delta_mean']}, "
                   f"span-safe "
                   f"{summary['meanJ_span_safe']['nll_delta_mean']}"),
             outputs=[out_json, pq, pq_g, gen_p],
             inputs={"battery_sha256": sha256_file(cfg["battery_path"])})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
