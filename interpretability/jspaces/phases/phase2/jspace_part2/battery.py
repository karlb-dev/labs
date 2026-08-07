# Minimal package-internal battery for R1/R7 (self-contained given the
# declared external input: anthropics/jacobian-lens @ pinned commit).
#
# Scoring is full-answer-sequence conditional logprob (R4), with the alias
# rule FROZEN here before any outcome inspection: variants = {" "+ans,
# " "+ans.capitalize()} (deduped), aggregate = max over variants. First-token
# accuracy is derived from the same ablated logits at the final prompt
# position (causality makes it independent of the appended answer).
from __future__ import annotations

import json
from pathlib import Path

import torch

PROBE_SWAP = Path("/content/jacobian-lens/data/experiments/probe-swap.json")

ONEHOP = [
    ("The capital of France is", "Paris"),
    ("The capital of Japan is", "Tokyo"),
    ("The capital of Italy is", "Rome"),
    ("The capital of Germany is", "Berlin"),
    ("The capital of Spain is", "Madrid"),
    ("The capital of Russia is", "Moscow"),
    ("The capital of England is", "London"),
    ("The capital of Egypt is", "Cairo"),
    ("The capital of Canada is", "Ottawa"),
    ("The capital of China is", "Beijing"),
    ("The largest planet in our solar system is", "Jupiter"),
    ("The chemical symbol for gold is", "Au"),
    ("The author of Romeo and Juliet is William", "Shakespeare"),
    ("The language spoken in Brazil is", "Portuguese"),
    ("Water freezes at zero degrees", "Celsius"),
    ("The opposite of hot is", "cold"),
    ("A spider has eight", "legs"),
    ("The color of the sky on a clear day is", "blue"),
    ("The first month of the year is", "January"),
    ("The number of days in a week is", "seven"),
    ("The star at the center of our solar system is the", "Sun"),
    ("The ocean between America and Europe is the", "Atlantic"),
    ("The currency of the United States is the", "dollar"),
    ("The largest mammal on Earth is the blue", "whale"),
    ("The frozen form of water is called", "ice"),
    ("The organ that pumps blood is the", "heart"),
    ("The planet known as the Red Planet is", "Mars"),
    ("The fastest land animal is the", "cheetah"),
    ("A baby dog is called a", "puppy"),
    ("The season after winter is", "spring"),
]


def twohop_items(n: int = 60) -> list[dict]:
    items = json.loads(PROBE_SWAP.read_text())["items"][:n]
    return [{"item_id": f"twohop:{it['name']}", "prompt": it["prompt"],
             "answer": it["answer"], "family": it["name"].split("-")[0]}
            for it in items]


def onehop_items() -> list[dict]:
    return [{"item_id": f"onehop:{i}", "prompt": p, "answer": a,
             "family": f"onehop{i}"} for i, (p, a) in enumerate(ONEHOP)]


def prose_items(corpus_jsonl: Path, lo=170, hi=190) -> list[dict]:
    rows = [json.loads(l) for l in Path(corpus_jsonl).read_text().splitlines()]
    return [{"item_id": f"prose:{r['idx']}", "text": r["text"][:1200],
             "family": f"prose{r['idx']}"} for r in rows[lo:hi]]


def answer_variants(ans: str) -> list[str]:
    a = ans.strip()
    return sorted({f" {a}", f" {a.capitalize()}"})


def seq_lp_from_logits(ids: torch.Tensor, logits: torch.Tensor,
                       n_prompt: int) -> float:
    """Sum logprob of positions [n_prompt:] given ablated logits [T, V]."""
    lp = torch.log_softmax(logits[:-1], dim=-1)
    tgt = ids[0, 1:].cpu()
    per = lp[torch.arange(len(tgt)), tgt]
    return float(per[n_prompt - 1:].sum())
