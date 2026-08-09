"""Group A: selectivity-language and selectivity-linecount (plan §10.A).

selectivity-language — released readout contrast only: explicit-question
label-hit rate minus automatic-continuation label-hit rate, label tokens
tracked over the question tokens following the passage, band rank-1 hits
(released conventions' default).

selectivity-linecount — textwrap.fill reconstruction; frozen number canon
(two-digit strings + English tens words); target-token appearance rate at
any prompt position in the band across the four conditions (k=1 primary
per the conventions' hit default; k=5 recorded sensitivity — D10).
"""
from __future__ import annotations

import json
import textwrap
import time
from pathlib import Path

import torch

from jlens.hooks import ActivationRecorder

from ..layers import PAPER_BAND
from ..paths import EXPERIMENTS_DIR
from ..rendering import render_raw
from ..scoring import rank_of
from ..targets import linecount_number_canon, synonym_token_ids


@torch.no_grad()
def _band_ranks_over_span(model, lens, rendered, *, span, token_ids,
                          band=PAPER_BAND):
    """Min rank per (layer, position) for the token set over a span."""
    with ActivationRecorder(model.layers, at=list(band)) as recorder:
        model.forward(rendered.input_ids)
        residuals = {
            layer: recorder.activations[layer][0, span[0]:span[1] + 1].detach()
            for layer in band
        }
    best = {}
    for layer in band:
        transported = lens.transport(residuals[layer].float(), layer)
        logits = model.unembed(transported).float().cpu()
        for offset in range(logits.shape[0]):
            rank = min(rank_of(logits[offset], t) for t in token_ids)
            best[(layer, span[0] + offset)] = rank
    return best


def run_language(model, lens, *, lane: str, out_dir: Path,
                 band=PAPER_BAND) -> dict:
    data = json.loads((EXPERIMENTS_DIR / "selectivity-language.json").read_text())
    tokenizer = model.tokenizer
    start = time.time()
    key_to_language = {"fr": "French", "de": "German", "es": "Spanish",
                       "it": "Italian"}
    rows = []
    for passage in data["passages"]:
        language = key_to_language[passage["key"][:2]]
        labels = data["intermediates"][language]
        label_ids = synonym_token_ids(tokenizer, labels)
        entry = {"key": passage["key"], "language": language,
                 "labels": labels,
                 "tokenization_valid": bool(label_ids)}
        if not label_ids:
            entry["state"] = "TOKENIZATION_GATED"
            rows.append(entry)
            continue
        for condition in ("explicit_q", "automatic_q"):
            template = data["task"][condition]
            text = template.format(text=passage["text"])
            rendered = render_raw(model, text)
            # Question span: tokens after the passage text. Offsets live
            # in the DECODED stream (BOS text shifts raw offsets on OLMo
            # — INCIDENT or1-002).
            ids = rendered.input_ids[0].tolist()
            decoded_full = tokenizer.decode(ids)
            passage_end_char = (decoded_full.find(passage["text"])
                                + len(passage["text"]))
            if passage_end_char < len(passage["text"]):
                raise ValueError("passage not found in decoded render")
            running = ""
            question_start = None
            for index in range(len(ids)):
                running = tokenizer.decode(ids[: index + 1])
                if len(running) >= passage_end_char:
                    question_start = index + 1
                    break
            span = (question_start, rendered.seq_len - 1)
            ranks = _band_ranks_over_span(model, lens, rendered,
                                          span=span, token_ids=label_ids,
                                          band=list(band))
            entry[condition] = {
                "span": span,
                "hit_rank1": any(r == 1 for r in ranks.values()),
                "hit_rank5": any(r <= 5 for r in ranks.values()),
                "min_rank": min(ranks.values()),
            }
        entry["state"] = "EXECUTED"
        rows.append(entry)
    executed = [r for r in rows if r.get("state") == "EXECUTED"]
    def _rate(key, field):
        return (sum(1 for r in executed if r[key][field]) / len(executed)
                if executed else None)
    summary = {
        "experiment": "selectivity-language", "lane": lane,
        "band": list(band), "n_passages": len(rows),
        "n_executed": len(executed),
        "explicit_hit_rate": _rate("explicit_q", "hit_rank1"),
        "automatic_hit_rate": _rate("automatic_q", "hit_rank1"),
        "contrast_rank1": (
            (_rate("explicit_q", "hit_rank1") or 0)
            - (_rate("automatic_q", "hit_rank1") or 0)
            if executed else None),
        "contrast_rank5": (
            (_rate("explicit_q", "hit_rank5") or 0)
            - (_rate("automatic_q", "hit_rank5") or 0)
            if executed else None),
        "wall_seconds": time.time() - start,
        "rows": rows,
    }
    path = out_dir / f"selectivity_language_{lane}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(summary))
    return summary


def run_linecount(model, lens, *, lane: str, out_dir: Path,
                  band=PAPER_BAND) -> dict:
    data = json.loads((EXPERIMENTS_DIR / "selectivity-linecount.json").read_text())
    tokenizer = model.tokenizer
    start = time.time()
    canon = linecount_number_canon(tokenizer)
    canon_ids = sorted({t for ids in canon.values() for t in ids})
    rows = []
    for passage in data["passages"]:
        wrapped = textwrap.fill(passage["text"], passage["width"])
        truth = len(wrapped.split("\n")[0])
        entry = {"tag": passage["tag"], "width": passage["width"],
                 "first_line_chars": truth, "conditions": {}}
        for name, condition in data["conditions"].items():
            question = condition["question"]
            prefill = condition["prefill"]
            text = (f"{question}\n\n{wrapped}\n\n{prefill}" if question
                    else f"{wrapped}\n\n{prefill}")
            rendered = render_raw(model, text)
            ranks = _band_ranks_over_span(
                model, lens, rendered,
                span=(0, rendered.seq_len - 1), token_ids=canon_ids,
                band=list(band))
            entry["conditions"][name] = {
                "hit_rank1": any(r == 1 for r in ranks.values()),
                "hit_rank5": any(r <= 5 for r in ranks.values()),
                "min_rank": min(ranks.values()),
            }
        text = f"{data['explicit_q']}\n\n{wrapped}"
        rendered = render_raw(model, text)
        ranks = _band_ranks_over_span(
            model, lens, rendered, span=(0, rendered.seq_len - 1),
            token_ids=canon_ids, band=list(band))
        entry["conditions"]["continue"] = {
            "hit_rank1": any(r == 1 for r in ranks.values()),
            "hit_rank5": any(r <= 5 for r in ranks.values()),
            "min_rank": min(ranks.values()),
        }
        rows.append(entry)
    conditions = ["none", "direct", "letter", "continue"]
    summary = {
        "experiment": "selectivity-linecount", "lane": lane,
        "band": list(band), "n_passages": len(rows),
        "n_canon_tokens": len(canon_ids),
        "assembly": "question\\n\\nwrapped\\n\\nprefill (D10)",
        "hit_rate_rank1": {c: sum(1 for r in rows
                                  if r["conditions"][c]["hit_rank1"]) / len(rows)
                           for c in conditions},
        "hit_rate_rank5": {c: sum(1 for r in rows
                                  if r["conditions"][c]["hit_rank5"]) / len(rows)
                           for c in conditions},
        "wall_seconds": time.time() - start,
        "rows": rows,
    }
    path = out_dir / f"selectivity_linecount_{lane}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(summary))
    return summary
