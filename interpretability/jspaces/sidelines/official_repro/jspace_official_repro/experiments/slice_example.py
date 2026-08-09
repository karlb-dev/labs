"""Slice examples for the report (Group-D style visual sanity, run early
for the Qwen lane): per-(layer, position) J-lens top-1 tokens and tracked
ranks on fixed example prompts — including the upstream README's
boot/currency probe, reused from the paper's own materials."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from jlens.hooks import ActivationRecorder

from ..layers import PAPER_GRID_SOURCES
from ..paths import DRIVE_ROOT
from ..readout import lens_to_device
from ..rendering import preferred_token, render_raw
from ..scoring import rank_of

EXAMPLES = [
    {
        "slug": "boot-currency",
        "prompt": "Fact: The currency used in the country shaped like a boot is",
        "tracked": ["Italy", "euro", "lira", "boot"],
        "note": "upstream README walkthrough probe (paper example)",
    },
    {
        "slug": "amazon-language",
        "prompt": ("Fact: The language spoken in the country where the "
                   "Amazon River ends is"),
        "tracked": ["Brazil", "Portuguese", "Spanish", "Amazon"],
        "note": "probe-swap item 0 baseline (released prompt)",
    },
]


@torch.no_grad()
def capture(model, lens, *, out_dir: Path, lane: str,
            layers=PAPER_GRID_SOURCES, last_n_positions: int = 14) -> Path:
    out_path = out_dir / f"slice_examples_{lane}.json"
    if out_path.exists():
        return out_path
    lens_to_device(lens, "cuda:0", layers=list(layers))
    tokenizer = model.tokenizer
    results = []
    for example in EXAMPLES:
        rendered = render_raw(model, example["prompt"])
        positions = list(range(max(0, rendered.seq_len - last_n_positions),
                               rendered.seq_len))
        with ActivationRecorder(model.layers, at=list(layers)) as recorder:
            model.forward(rendered.input_ids)
            residuals = {
                layer: recorder.activations[layer][0, positions].detach()
                for layer in layers
            }
        tracked_ids = {w: preferred_token(tokenizer, w)
                       for w in example["tracked"]}
        grid = []
        for layer in layers:
            transported = lens.transport(residuals[layer].float(), layer)
            logits = model.unembed(transported).float().cpu()
            row = []
            for index, position in enumerate(positions):
                top1 = int(logits[index].argmax())
                cell = {
                    "position": position,
                    "top1": tokenizer.decode([top1]),
                    "tracked_ranks": {
                        w: (rank_of(logits[index], t) if t is not None else None)
                        for w, t in tracked_ids.items()
                    },
                }
                row.append(cell)
            grid.append({"layer": layer, "cells": row})
        results.append({
            **{k: example[k] for k in ("slug", "prompt", "note")},
            "position_tokens": [tokenizer.decode([rendered.input_ids[0, p].item()])
                                for p in positions],
            "positions": positions,
            "tracked_token_ids": {w: t for w, t in tracked_ids.items()},
            "grid": grid,
        })
    out_path.write_text(json.dumps({
        "lane": lane, "layers": list(layers),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "examples": results,
    }))
    return out_path


if __name__ == "__main__":
    from . import admission as admission_module

    model, hf_model, tokenizer = admission_module.load_qwen()
    lens = admission_module.load_qwen_lens()
    out = capture(model, lens, out_dir=DRIVE_ROOT / "qwen_lane", lane="qwen")
    print("slice examples ->", out)
