# Phase 1c: lens sanity on the 32B — abort the lab here if the lens is noise.
#
# Battery A: VM1's 11 boot-style probes + 10 extra single-token factual
# probes; J-lens vs logit-lens rank of the answer's first token, by layer,
# at the final prompt position. Battery B: the paper's own multihop lens
# eval (data/evaluations/lens-eval-multihop.json), pass@k (k=1,5,20), min
# rank over fitted mid-band layers, J-lens vs logit-lens.
# Usage: python scripts/s4_lens_sanity.py [--lens <path>] [--max-items 60]
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sl1_common import (RUN_DIR, answer_rank, atomic_write_json, die,
                        ensure_dirs, first_token_id, load_model, log,
                        read_json, seed_all, variant_first_ids)

from jlens import JacobianLens

OUT = RUN_DIR / "metrics" / "lens_sanity_32b.json"
MID_BAND = list(range(20, 45, 2))  # fitted middle band (see PLAN.md)
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


def pick_lens_path() -> Path:
    p = Path(arg("--lens", str(RUN_DIR / "lens" / "olmo32bthink_lens.pt")))
    if p.exists():
        return p
    slices = sorted(RUN_DIR.glob("lens/olmo32bthink_slice*.pt"))
    if slices:
        log(f"merged lens missing; falling back to {slices[-1].name}")
        return slices[-1]
    die("no 32B lens found; run s3_fit_32b.py first")


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
    per_item, n_scored, n_skipped = [], 0, 0
    for it in items:
        prompt = it["prompt"]
        # These prompts END immediately before where `target` would be
        # generated ("...celebrated is the "), so "the token immediately
        # preceding target" (README) is the final prompt token. The scored
        # concepts are the silent bridge `intermediates` (e.g. 'Brazil').
        jl, _, _ = lens.apply(model, prompt, positions=[-1], max_seq_len=1024)
        ll, _, _ = lens.apply(model, prompt, positions=[-1], max_seq_len=1024,
                              use_jacobian=False)
        n_scored += 1
        row = {"prompt": prompt[:80], "intermediates": it["intermediates"]}
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
    for key in list(res):
        res[key] = res[key] / max(n_scored, 1)
    res["n_scored"], res["n_skipped"] = n_scored, n_skipped
    return res, per_item


def main() -> None:
    ensure_dirs()
    seed_all()
    if OUT.exists() and "--force" not in sys.argv:
        log(f"{OUT} exists; skipping")
        return
    lens_path = pick_lens_path()
    lens = JacobianLens.load(str(lens_path))
    log(f"lens: {lens!r} from {lens_path.name}")
    model, hf, tok = load_model("main")

    rows, j_hits, l_hits = probe_battery(model, tok, lens)
    mh, mh_items = multihop_eval(model, tok, lens, int(arg("--max-items", "60")))

    out = {"lens_file": lens_path.name, "n_lens_prompts": lens.n_prompts,
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
           "mid_band": MID_BAND,
           "probe_jlens_hits_at_20": j_hits, "probe_logit_hits_at_20": l_hits,
           "n_probes": len(rows), "probes": rows,
           "multihop": mh, "multihop_items": mh_items}
    atomic_write_json(out, OUT)
    log(f"wrote {OUT}")
    log(f"probes: jlens {j_hits}/{len(rows)} vs logit {l_hits}/{len(rows)} hits@20")
    log(f"multihop: { {k: round(v, 3) for k, v in mh.items()} }")
    if j_hits < max(3, l_hits - 2):
        die("J-lens no better than noise vs logit lens — debug before Phase 2")
    log("LENS SANITY PASS")


if __name__ == "__main__":
    main()
