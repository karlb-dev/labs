"""directed-modulation line-break family — R2 deterministic adaptation.

The underlying prose is not released; the README authorizes "any prose
corpus filtered to alpha-heavy ASCII text". Pinned corpus: WikiText
sentinel-pool rows (after the fit-1000; frozen in the fit-population
manifest), filtered to >=0.8 alpha+space fraction, wrapped with
``textwrap.fill`` at k ∈ {40,50,...,100}, 5–7-line interior window.
Never called exact (R2)."""
from __future__ import annotations

import json
import textwrap
import time
from pathlib import Path

from ..layers import PAPER_BAND
from ..paths import FIT_DATA
from ..rendering import render_raw
from ..targets import synonym_token_ids
from .modulation import _render_carrier_task, _span_hit

WIDTHS = [40, 50, 60, 70, 80, 90, 100]

#: Tracked targets: the width itself as digits (the latent variable the
#: paper's family modulates), per phrasing kind.
def _alpha_heavy(text: str) -> bool:
    if not text:
        return False
    good = sum(1 for c in text if c.isalpha() or c in " .,;'\"-")
    return good / len(text) >= 0.8


def pinned_prose() -> list[str]:
    rows = [json.loads(line) for line in
            (FIT_DATA / "wikitext_sentinels_after1000.jsonl").read_text().splitlines()]
    return [r["text"].strip() for r in rows if _alpha_heavy(r["text"].strip())]


def run(model, lens, *, lane: str, out_dir: Path, band=PAPER_BAND) -> dict:
    data_prose = pinned_prose()
    tokenizer = model.tokenizer
    start = time.time()
    rows = []
    phrasings = [
        ("focus", "Think about the line width while you write."),
        ("suppress", "Don't think about the line width while you write."),
        ("control", "The passage above has lines in it."),
    ]
    for prose_index, prose in enumerate(data_prose):
        width = WIDTHS[prose_index % len(WIDTHS)]
        wrapped_lines = textwrap.fill(prose, width).split("\n")
        if len(wrapped_lines) < 7:
            continue
        window = "\n".join(wrapped_lines[1:7])
        tracked = synonym_token_ids(tokenizer, [str(width)])
        if not tracked:
            continue
        for kind, instruction in phrasings:
            user = (f"Here is a passage wrapped at a fixed column width:"
                    f"\n\n{window}\n\n{instruction}")
            carrier = "The old painting hung crookedly on the wall."
            rendered, span = _render_carrier_task(
                model, lane=lane, carrier=carrier,
                instruction=f"{instruction[:-1]} — {user[:0]}".strip() or instruction)
            # R2 adaptation: passage precedes the carrier task inside the
            # user turn; rebuild the full user turn explicitly.
            from ..rendering import find_token_span, render_chat

            kwargs = {"enable_thinking": False} if lane == "qwen" else {}
            messages = [
                {"role": "user",
                 "content": (f"Here is a passage wrapped at a fixed column "
                             f"width:\n\n{window}\n\nWrite \"{carrier}\" "
                             f"{instruction} Don't write anything else.")},
                {"role": "assistant", "content": carrier},
            ]
            rendered = render_chat(model, messages, continue_final=True,
                                   extra_template_kwargs=kwargs)
            span = find_token_span(rendered, tokenizer, carrier, from_end=True)
            hit = _span_hit(model, lens, rendered, span=span,
                            token_ids=tracked, band=list(band))
            rows.append({"prose_index": prose_index, "width": width,
                         "kind": kind, **hit, "state": "EXECUTED"})
    def _rate(kind):
        subset = [r for r in rows if r["kind"] == kind]
        return (sum(1 for r in subset if r["hit_rank5"]) / len(subset)
                if subset else None)
    summary = {
        "experiment": "directed-modulation-linebreak", "lane": lane,
        "fidelity_class": "R2",
        "corpus": "wikitext sentinel pool, alpha-heavy >=0.8 (pinned)",
        "band": list(band), "n_rows": len(rows),
        "hit_rank5_by_kind": {k: _rate(k) for k in
                              ("focus", "suppress", "control")},
        "wall_seconds": time.time() - start,
        "rows": rows,
    }
    path = out_dir / f"linebreak_{lane}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(summary))
    return summary
