#!/usr/bin/env python3
"""G-LEX early probe (development tier): does a pole-authoring lexical
prior explain the folded content margins?

For every distinct (scenario, incidental) option pair in the frozen bank,
score the summed token logprob of each pole's option string under a
reference LM (unconditional: BOS + text), and regress the frozen folded
content margins on Delta_lex = lp(pole_1) - lp(pole_0) (same orientation
as margin_pole1_minus_pole0).

Usage: lexical_probe.py <model_id> <tag>
  model_id: HF id (must be cached), e.g. gpt2 or allenai/Olmo-3-7B-Instruct
  tag: output label, e.g. gpt2 / olmo7b

Reads reanalysis.json (must exist) for the folded margins.
Writes lexical_probe_<tag>.json.
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse
import os


def _interp_root() -> pathlib.Path:
    """Portable root discovery (plan §6.3): --repo-root arg, else
    $PREF2_REPO_ROOT, else walk up from this file to `.git`."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--out", default=None)
    ns, _ = ap.parse_known_args()
    if ns.out:
        globals()["_OUT_OVERRIDE"] = pathlib.Path(ns.out).resolve()
    root = ns.repo_root or os.environ.get("PREF2_REPO_ROOT")
    if root:
        return pathlib.Path(root).resolve() / "interpretability"
    here = pathlib.Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / ".git").exists():
            return parent / "interpretability"
    raise RuntimeError("repo root not found; pass --repo-root or set PREF2_REPO_ROOT")


_OUT_OVERRIDE = None
REPO = _interp_root()
BANK = REPO / "preference" / "data" / "lab38_preference_bank.jsonl"
OUT = _OUT_OVERRIDE or pathlib.Path(__file__).resolve().parent


def collect_pairs() -> dict:
    """(family, scenario, incidental) -> {'0': text, '1': text}"""
    pairs = {}
    for line in open(BANK):
        r = json.loads(line)
        if r["channel"] != "AR":
            continue
        k = (r["family"], r["scenario_id"], r["incidental_id"])
        if k not in pairs:
            pairs[k] = r["option_text_by_pole"]
    return pairs


@torch.no_grad()
def score_strings(model_id: str, texts: list[str]) -> list[float]:
    tok = AutoTokenizer.from_pretrained(model_id)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.to(device).eval()
    bos = tok.bos_token_id
    out = []
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        seq = ([bos] if bos is not None else []) + ids
        x = torch.tensor([seq], device=device)
        logits = model(x).logits.float()
        lp = torch.log_softmax(logits[0, :-1], dim=-1)
        tgt = x[0, 1:]
        s = float(lp[torch.arange(len(tgt)), tgt].sum())
        out.append(s)
    del model
    return out


def main() -> None:
    model_id, tag = sys.argv[1], sys.argv[2]
    pairs = collect_pairs()
    keys = sorted(pairs)
    texts, index = [], {}
    for k in keys:
        for pole in ("0", "1"):
            index[(k, pole)] = len(texts)
            texts.append(pairs[k][pole])
    print(f"scoring {len(texts)} strings under {model_id} ...")
    scores = score_strings(model_id, texts)

    # Delta_lex per (family, scenario, incidental); NC pairs must be ~0.
    dlex = {}
    for k in keys:
        d = scores[index[(k, "1")]] - scores[index[(k, "0")]]
        dlex[k] = d

    re = json.load(open(OUT / "reanalysis.json"))
    result = {"model_id": model_id, "n_strings": len(texts),
              "nc_dlex_max_abs": max(abs(v) for (f, s, i), v in dlex.items()
                                     if f == "NC")
              if any(f == "NC" for (f, s, i) in dlex) else None}

    for model in ("7b", "32b"):
        fold = re[model]["f2_folded_margins"]
        rows = []
        for (fam, scen, inc), d in sorted(dlex.items()):
            if fam != "AR":
                continue
            fm = fold.get(f"AR:{scen}")
            if fm is None:
                continue
            m = fm["per_incidental"].get(inc)
            if m is None:
                continue
            rows.append((scen, inc, d, m))
        d = np.array([r[2] for r in rows])
        m = np.array([r[3] for r in rows])
        # scenario-level (mean over incidentals)
        sc = defaultdict(lambda: [[], []])
        for scen, inc, dd, mm in rows:
            sc[scen][0].append(dd)
            sc[scen][1].append(mm)
        sd = np.array([np.mean(v[0]) for v in sc.values()])
        sm = np.array([np.mean(v[1]) for v in sc.values()])

        def fit(x, y):
            if len(x) < 3:
                return None
            b, a = np.polyfit(x, y, 1)
            yhat = a + b * x
            ssr = float(((y - yhat) ** 2).sum())
            sst = float(((y - y.mean()) ** 2).sum())
            return {"slope": float(b), "intercept": float(a),
                    "r": float(np.corrcoef(x, y)[0, 1]),
                    "R2": 1 - ssr / sst if sst > 0 else None,
                    "sign_match": int(np.sum(np.sign(x) == np.sign(y))),
                    "n": len(x)}

        result[model] = {
            "cell_level": fit(d, m),
            "scenario_level": fit(sd, sm),
            "scenario_dlex": {s: float(np.mean(v[0]))
                              for s, v in sorted(sc.items())},
        }
        print(f"[{model}] cell: {result[model]['cell_level']}")
        print(f"[{model}] scen: {result[model]['scenario_level']}")

    path = OUT / f"lexical_probe_{tag}.json"
    json.dump(result, open(path, "w"), indent=1, sort_keys=True)
    print("wrote", path, "| NC sanity max|dlex|:", result["nc_dlex_max_abs"])


if __name__ == "__main__":
    main()
