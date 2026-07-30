# §6.5 bridge-mediation factorial — the Block C mechanism start.
#
# On composed Bank F items (frozen confirmatory side, model-capable,
# counterfactual bridge present), seven arms per item:
#
#   span_safe            output-protected reference damage
#   true_bridge          + true-bridge piece protection
#   distractor_bridge    + counterfactual-bridge piece protection
#   bridge_only          selection RESTRICTED to true-bridge pieces
#                        (output protection still binds inside the set)
#   cf_swap              bridge_only + energy-matched injection of the
#                        counterfactual bridge's mean dictionary
#                        direction (per layer)
#   answer_only          selection restricted to answer pieces, no
#                        protection — the §6.5 deletion diagnostic
#   unrelated            selection restricted to the NEXT fact's bridge
#                        pieces (item-matched content control)
#
# Bridge piece sets include tokenizer variants; any-piece/all-piece
# coverage recorded per §6.5. Development tier (the preregistered P3-P3
# rescue statistic lives in the primary grid; this factorial is the
# mechanism elaboration).
#
# Usage:
#   python -m jspace_phase3.experiments.bridge_mediation --slug <slug> \
#       --model-uri model://... --lens-uri <uri> [--n 40]
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from jspace_part2.dictionaries import build_j_dictionaries
from jspace_part2.paths import resolve as resolve_uri
from ..ablator3 import Phase3JAblator
from ..bank import load_bank
from ..paths3 import metrics_dir
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           resolve_model, write_result3)
from ..scoring import DEFAULT_SPEC, ScoringSession
from ..stats import family_cluster_bootstrap_ci

TIER = "phase3-development"
REPO_DATA = Path(__file__).resolve().parents[2] / "data"


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def piece_ids(tok, ent: str) -> torch.Tensor:
    b = ent.removeprefix("the ").removeprefix("The ").strip()
    ids = set()
    for v in {f" {b}", b, f" {b.lower()}", f" {b.title()}"}:
        for t in tok(v, add_special_tokens=False).input_ids:
            ids.add(int(t))
    return torch.tensor(sorted(ids), dtype=torch.long)


@torch.no_grad()
def main():  # noqa: C901
    require_clean_tree("--allow-dirty" in sys.argv)
    slug = arg("--slug")
    model_uri = arg("--model-uri")
    n_items = int(arg("--n", 40))
    out_dir = metrics_dir(slug) / "bridge_mediation"
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.json"
    state = (json.loads(state_path.read_text()) if state_path.exists()
             else {"done": {}, "rows": []})

    import transformers
    import jlens
    from jlens import JacobianLens
    model_path = str(resolve_uri(model_uri, must_exist=True))
    tok = transformers.AutoTokenizer.from_pretrained(model_path)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16).to("cuda").eval()
    model = jlens.from_hf(hf, tok)
    sess = ScoringSession(tok, DEFAULT_SPEC, device="cuda")
    lens = JacobianLens.load(str(resolve_uri(arg("--lens-uri"))))
    band = [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]
    k = pk = 10
    jd = build_j_dictionaries(hf, lens, band)
    ab = Phase3JAblator(model.layers, band)

    part = json.loads(Path("interpretability/jspace_phase3/preregistration/"
                           "partition_phase3.json").read_text())["payload"]
    fams = set(part["confirmatory"])
    gd = metrics_dir(slug) / "g5_bank"
    gp = gd / f"g5_bank_{slug}_regraded.parquet"
    g5 = pd.read_parquet(gp if gp.exists()
                         else gd / f"g5_bank_{slug}.parquet")
    dc = g5[g5.variant.isin(["direct", "composed"])]
    cap = {fid for fid, sub in dc.groupby("fact_id")
           if len(sub) == 2 and bool(sub.capable_generation.all())}
    bundles = [b for b in load_bank(REPO_DATA / "bank_f_v7.jsonl")
               if b.canonical_family in fams and b.fact_id in cap
               and b.counterfactual_bridge]
    bundles = sorted(bundles, key=lambda b: b.fact_id)[:n_items]
    log(f"{slug}: {len(bundles)} mediation items")

    def run(ids, *, psets=None, span_safe=True, restrict=None, inject=None):
        ab.log = type(ab.log)()
        ab.phase, ab.forward_index = "prefill", 0
        ab.mode = {"dicts": jd, "k": k, "nonneg": True,
                   "protect_sets": psets, "active_phases": {"prefill"},
                   "span_safe": span_safe, "record_overlap": False,
                   "answer_id": None, "restrict_sets": restrict,
                   "inject_dir": inject}
        with ab:
            out = hf(input_ids=ids, use_cache=False).logits[0]
        ab.mode = None
        return out.float().cpu()

    t0 = time.time()
    for n, b in enumerate(bundles):
        if b.fact_id in state["done"]:
            continue
        full, n_p = sess.full_ids(b.prompts["composed"],
                                  b.accepted_answers[0])
        T = full.shape[1]
        ab.mode = None
        clean = hf(input_ids=full, use_cache=False).logits[0].float()
        psets = clean.topk(pk, dim=-1).indices
        lp_base = sess.answer_seq_lp(full, clean.cpu(), n_p)

        bt = piece_ids(tok, b.bridge).to(psets.device)
        dt = piece_ids(tok, b.counterfactual_bridge).to(psets.device)
        at = piece_ids(tok, b.answer).to(psets.device)
        nxt = bundles[(n + 1) % len(bundles)]
        ut = piece_ids(tok, nxt.bridge).to(psets.device)
        cf_dirs = {l: jd[l][dt].float().mean(0) for l in band}

        def cat(base, extra):
            return torch.cat([base, extra.unsqueeze(0).expand(T, -1)], dim=1)

        arms = {
            "span_safe": dict(psets=psets),
            "true_bridge": dict(psets=cat(psets, bt)),
            "distractor_bridge": dict(psets=cat(psets, dt)),
            "bridge_only": dict(psets=psets, restrict=bt),
            "cf_swap": dict(psets=psets, restrict=bt, inject=cf_dirs),
            "answer_only": dict(psets=None, span_safe=False, restrict=at),
            "unrelated": dict(psets=psets, restrict=ut),
        }
        row = {"fact_id": b.fact_id,
               "canonical_family": b.canonical_family,
               "lp_base": lp_base,
               "bridge_pieces": int(len(bt)),
               "distractor_pieces": int(len(dt))}
        for arm, kw in arms.items():
            logits = run(full, **kw)
            row[f"d_{arm}"] = sess.answer_seq_lp(full, logits, n_p) - lp_base
        state["rows"].append(row)
        state["done"][b.fact_id] = round(time.time() - t0)
        if len(state["done"]) % 5 == 0:
            state_path.write_text(json.dumps(state))
            log(f"{len(state['done'])}/{len(bundles)}")
    state_path.write_text(json.dumps(state))

    df = pd.DataFrame(state["rows"])
    pq = out_dir / f"bridge_mediation_{slug}.parquet"
    df.to_parquet(pq)
    stats = {}
    for arm in ("span_safe", "true_bridge", "distractor_bridge",
                "bridge_only", "cf_swap", "answer_only", "unrelated"):
        stats[arm] = family_cluster_bootstrap_ci(df, f"d_{arm}")
    df["rescue"] = df.d_true_bridge - df.d_distractor_bridge
    stats["rescue_contrast"] = family_cluster_bootstrap_ci(df, "rescue")
    eid = f"p3-bridge-mediation-{slug}-v1"
    cmd = (f"python -m jspace_phase3.experiments.bridge_mediation "
           f"--slug {slug} --model-uri {model_uri} "
           f"--lens-uri {arg('--lens-uri')} --n {n_items}")
    out_json = out_dir / f"bridge_mediation_{slug}.json"
    write_result3({"n_items": int(len(df)), "arm_stats": stats},
                  out_json, Provenance3(
                      evidence_id=eid, tier=TIER, command=cmd,
                      model=resolve_model(model_path), seed=0))
    register(eid, tier=TIER, command=cmd,
             what=(f"§6.5 bridge-mediation factorial on {slug}: "
                   f"{len(df)} composed items × 7 arms; rescue contrast "
                   f"{stats['rescue_contrast']['estimate']:+.3f} "
                   f"[{stats['rescue_contrast']['ci95'][0]:+.3f},"
                   f"{stats['rescue_contrast']['ci95'][1]:+.3f}]"),
             outputs=[out_json, pq])
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
