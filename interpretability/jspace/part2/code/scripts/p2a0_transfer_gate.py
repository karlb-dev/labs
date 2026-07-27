# A0 — lens-transfer gate: does the part-1 Think-fitted J-lens read a
# post-training sibling? Runs the part-1 sanity battery (21 probes + the
# paper's multihop bridge eval, n=60) on --model with the donor lens.
#
# Transfer semantics: donor J, recipient unembedding — lens.apply() reads
# the recipient's residuals through the stored donor Jacobians and the
# recipient's own (W_U ⊙ g); this is exactly how a transferred lens would be
# consumed by every downstream instrument. A drift diagnostic (donor-vs-
# recipient unembed row cosines + final-norm gain cosine) is recorded so a
# FAIL can be attributed to J-drift vs readout-basis drift.
#
# Gate (preregistration.md): PASS ⇔ probe hits@20 ≥ 15/21 AND multihop
# J-lens pass@1 ≥ 0.85 × donor's (donor gate: 17/21, 0.283). Secondary,
# non-gating: does J-lens still beat the recipient's own logit lens?
#
# Usage: python scripts/p2a0_transfer_gate.py [--model olmo31-instruct]
#        [--lens <path>] [--max-items 60] [--force]
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl2_common import (DONOR_LENS, MID_BAND, P2_MODELS, RUN_DIR,
                        answer_rank, atomic_write_json, die, log,
                        p2_load_model, p2_metrics_dir, read_json, seed_all,
                        variant_first_ids)

from jlens import JacobianLens

EVAL_DIR = Path("/content/jacobian-lens/data/evaluations")

EXTRA_PROBES = [
    ("Fact: The chemical symbol for gold is", " Au"),
    ("Fact: The number of continents on Earth is", " seven"),
    ("Fact: The city known as the Big Apple is New", " York"),
    ("Fact: The gas plants absorb from the air is carbon", " dioxide"),
    ("Fact: The longest river in Egypt is the", " Nile"),
    ("Fact: The instrument with 88 keys is the", " piano"),
    ("Fact: The fastest land animal is the", " cheetah"),
    ("Fact: The smallest prime number is", " two"),
    ("Fact: The opposite of hot is", " cold"),
    ("Fact: The color of an emerald is", " green"),
]


def arg(flag: str, default: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def probe_battery(model, tok, lens):
    ref = read_json(RUN_DIR / "metrics" / "smoke_7b.json")
    probes = [(r["prompt"], r["answer"]) for r in ref["boot_probe"]["rows"]]
    probes += EXTRA_PROBES
    rows, j_hits, l_hits = [], 0, 0
    for prompt, answer in probes:
        ans_ids = variant_first_ids(tok, answer)
        jl, ml, _ = lens.apply(model, prompt, positions=[-1])
        ll, _, _ = lens.apply(model, prompt, positions=[-1], use_jacobian=False)
        j_by_layer = {l: min(answer_rank(jl[l][0], a) for a in ans_ids)
                      for l in lens.source_layers}
        l_by_layer = {l: min(answer_rank(ll[l][0], a) for a in ans_ids)
                      for l in lens.source_layers}
        j_mid = min(j_by_layer[l] for l in MID_BAND)
        l_mid = min(l_by_layer[l] for l in MID_BAND)
        j_hits += j_mid <= 20
        l_hits += l_mid <= 20
        rows.append({"prompt": prompt, "answer": answer,
                     "final_rank": min(answer_rank(ml[0], a) for a in ans_ids),
                     "jlens_rank_by_layer": j_by_layer,
                     "logit_rank_by_layer": l_by_layer,
                     "jlens_mid_min": j_mid, "logit_mid_min": l_mid})
        log(f"  {answer!r:12} jlens_mid={j_mid:>6} logit_mid={l_mid:>6} "
            f"final={rows[-1]['final_rank']}")
    return rows, j_hits, l_hits


def multihop_eval(model, tok, lens, max_items: int):
    items = json.loads((EVAL_DIR / "lens-eval-multihop.json").read_text())["items"]
    items = items[:max_items]
    ks = (1, 5, 20)
    res = {f"jlens_pass@{k}": 0.0 for k in ks} | {f"logit_pass@{k}": 0.0 for k in ks}
    per_item, n_scored = [], 0
    for it in items:
        # Prompts end immediately before the target; readout position is the
        # final prompt token; scored concepts are the silent bridge entities.
        jl, _, _ = lens.apply(model, it["prompt"], positions=[-1], max_seq_len=1024)
        ll, _, _ = lens.apply(model, it["prompt"], positions=[-1], max_seq_len=1024,
                              use_jacobian=False)
        n_scored += 1
        row = {"prompt": it["prompt"][:80], "intermediates": it["intermediates"]}
        for name, out in (("jlens", jl), ("logit", ll)):
            fracs = {k: 0.0 for k in ks}
            for inter in it["intermediates"]:
                iids = variant_first_ids(tok, inter)
                best = min(answer_rank(out[l][0], i) for i in iids
                           for l in lens.source_layers)
                for k in ks:
                    fracs[k] += (best <= k) / len(it["intermediates"])
                row[f"{name}_best_rank_{inter}"] = best
            for k in ks:
                res[f"{name}_pass@{k}"] += fracs[k]
        per_item.append(row)
        if n_scored % 10 == 0:
            log(f"  multihop {n_scored}/{len(items)}: "
                f"j@1={res['jlens_pass@1']/n_scored:.3f} "
                f"l@1={res['logit_pass@1']/n_scored:.3f}")
    for key in list(res):
        res[key] = res[key] / max(n_scored, 1)
    res["n_scored"] = n_scored
    return res, per_item


def unembed_drift(hf_recipient, donor_id: str, n_rows: int = 4000):
    """Donor-vs-recipient readout-basis drift, from safetensors slices only.

    Reads the donor's lm_head rows [0:n_rows] and final-norm gain straight
    from the Drive HF cache (no donor model load). Row-wise cosines tell us
    whether a gate FAIL is J-drift or unembed drift; on PASS the summary
    contextualizes why transfer works.
    """
    import torch
    from safetensors import safe_open

    cache = Path("/content/drive/MyDrive/hf_cache/hub")
    snaps = sorted((cache / f"models--{donor_id.replace('/', '--')}"
                    / "snapshots").glob("*"))
    if not snaps:
        return {"error": "donor snapshot not in Drive cache"}
    snap = snaps[-1]
    index = json.loads((snap / "model.safetensors.index.json").read_text())
    wmap = index["weight_map"]
    head_key = "lm_head.weight" if "lm_head.weight" in wmap else "model.embed_tokens.weight"
    norm_key = next((k for k in wmap if k in
                     ("model.norm.weight", "model.final_layernorm.weight")), None)
    out = {"head_key": head_key, "n_rows_sampled": n_rows}
    with safe_open(snap / wmap[head_key], framework="pt") as f:
        donor_head = f.get_slice(head_key)[0:n_rows].float()
    rec_head = hf_recipient.get_output_embeddings().weight[0:n_rows].detach().float().cpu()
    cos = torch.nn.functional.cosine_similarity(donor_head, rec_head, dim=1)
    out["head_row_cos"] = {
        "median": round(cos.median().item(), 4),
        "q05": round(cos.quantile(0.05).item(), 4),
        "q95": round(cos.quantile(0.95).item(), 4),
        "min": round(cos.min().item(), 4),
    }
    if norm_key:
        with safe_open(snap / wmap[norm_key], framework="pt") as f:
            donor_g = f.get_tensor(norm_key).float()
        rec_g = None
        for cand in ("model.norm.weight",):
            try:
                rec_g = hf_recipient.get_parameter(cand).detach().float().cpu()
                break
            except Exception:
                continue
        if rec_g is not None and rec_g.shape == donor_g.shape:
            gcos = torch.nn.functional.cosine_similarity(
                donor_g[None], rec_g[None], dim=1)
            out["final_norm_gain_cos"] = round(gcos.item(), 6)
    return out


def main() -> None:
    seed_all()
    slug = arg("--model", "olmo31-instruct")
    out_path = p2_metrics_dir(slug) / "a0_transfer_gate.json"
    if out_path.exists() and "--force" not in sys.argv:
        log(f"{out_path} exists; skipping")
        return

    donor_ref = read_json(RUN_DIR / "metrics" / "lens_sanity_32b.json")
    donor_probe_hits = donor_ref["probe_jlens_hits_at_20"]
    donor_pass1 = donor_ref["multihop"]["jlens_pass@1"]

    lens_path = Path(arg("--lens", str(DONOR_LENS)))
    lens = JacobianLens.load(str(lens_path))
    log(f"donor lens: {lens!r} from {lens_path.name}")
    model, hf, tok = p2_load_model(slug)

    rows, j_hits, l_hits = probe_battery(model, tok, lens)
    mh, mh_items = multihop_eval(model, tok, lens, int(arg("--max-items", "60")))
    drift = unembed_drift(hf, P2_MODELS["olmo3-think"]["id"])

    probe_thresh = 15  # ceil(0.85 * donor 17), fixed in preregistration.md
    pass1_thresh = round(0.85 * donor_pass1, 4)
    passed = j_hits >= probe_thresh and mh["jlens_pass@1"] >= pass1_thresh
    verdict = "TRANSFER_PASS" if passed else "TRANSFER_FAIL"
    advantage = round(mh["jlens_pass@1"] - mh["logit_pass@1"], 4)

    out = {
        "model_slug": slug, "model_id": P2_MODELS[slug]["id"],
        "donor_lens": lens_path.name, "n_lens_prompts": lens.n_prompts,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mid_band": MID_BAND,
        "gate": {"probe_threshold": probe_thresh,
                 "pass1_threshold": pass1_thresh,
                 "donor_probe_hits": donor_probe_hits,
                 "donor_jlens_pass1": donor_pass1,
                 "donor_logit_pass1": donor_ref["multihop"]["logit_pass@1"]},
        "probe_jlens_hits_at_20": j_hits, "probe_logit_hits_at_20": l_hits,
        "n_probes": len(rows), "probes": rows,
        "multihop": mh, "multihop_items": mh_items,
        "unembed_drift": drift,
        "jlens_minus_logit_pass1": advantage,
        "verdict": verdict,
    }
    atomic_write_json(out, out_path)
    log(f"wrote {out_path}")
    log(f"probes: jlens {j_hits}/{len(rows)} (donor {donor_probe_hits}; gate ≥{probe_thresh})")
    log(f"multihop pass@1: jlens {mh['jlens_pass@1']:.3f} vs own-logit "
        f"{mh['logit_pass@1']:.3f} (donor jlens {donor_pass1:.3f}; gate ≥{pass1_thresh})")
    log(f"unembed drift: {drift.get('head_row_cos')}, "
        f"gain cos {drift.get('final_norm_gain_cos')}")
    log(f"VERDICT: {verdict} (secondary: J-minus-logit advantage {advantage:+.3f})")


if __name__ == "__main__":
    main()
