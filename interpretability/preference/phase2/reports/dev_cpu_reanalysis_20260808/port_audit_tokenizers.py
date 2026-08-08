#!/usr/bin/env python3
"""P-1 offline codebook-survival audit: frozen Lab 38 codes vs candidate
Phase 2 model tokenizers (development tier; tokenizer files only).

Filters per Phase 1 addendum E codebook discipline:
  - equal token count within each code pair
  - distinct first tokens within each pair
  - checked bare and with leading space (frozen policy was 'none')

Models audited: OLMo 7B (control: must reproduce the frozen token counts),
Qwen/Qwen3.6-27B, google/gemma-4-31B-it (gated: may fail auth — recorded).
"""
from __future__ import annotations

import json
import pathlib

from transformers import AutoTokenizer

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
CB = json.load(open(REPO / "preference" / "data" / "lab38_codebook.json"))
OUT = _OUT_OVERRIDE or pathlib.Path(__file__).resolve().parent

PAIRS = {"ar_pair": CB["ar_pair"], "ro_pair": CB["ro_pair"]}
MODELS = {
    "olmo7b": "allenai/Olmo-3-7B-Instruct",
    "qwen": "Qwen/Qwen3.6-27B",
    "gemma": "google/gemma-4-31B-it",
}


def audit(tok, code: str, lead: str) -> dict:
    ids = tok(lead + code, add_special_tokens=False)["input_ids"]
    return {"token_count": len(ids), "ids": ids,
            "first": ids[0] if ids else None}


def main() -> None:
    result = {}
    for tag, mid in MODELS.items():
        try:
            tok = AutoTokenizer.from_pretrained(mid)
        except Exception as e:  # gated / offline
            result[tag] = {"model_id": mid, "error": type(e).__name__,
                           "detail": str(e)[:200]}
            print(f"[{tag}] UNAVAILABLE: {type(e).__name__}")
            continue
        entry = {"model_id": mid, "pairs": {}}
        for name, (a, b) in PAIRS.items():
            per = {}
            for lead_name, lead in (("bare", ""), ("space", " ")):
                ra, rb = audit(tok, a, lead), audit(tok, b, lead)
                per[lead_name] = {
                    a: ra, b: rb,
                    "equal_token_count": ra["token_count"] == rb["token_count"],
                    "distinct_first": ra["first"] != rb["first"],
                    "survives": (ra["token_count"] == rb["token_count"]
                                 and ra["first"] != rb["first"]),
                }
            entry["pairs"][name] = per
            print(f"[{tag}] {name} {a}/{b}: bare survives="
                  f"{per['bare']['survives']} (counts "
                  f"{per['bare'][a]['token_count']}/"
                  f"{per['bare'][b]['token_count']}), space survives="
                  f"{per['space']['survives']}")
        result[tag] = entry
    json.dump(result, open(OUT / "port_audit_tokenizers.json", "w"),
              indent=1, sort_keys=True)
    print("wrote", OUT / "port_audit_tokenizers.json")


if __name__ == "__main__":
    main()
