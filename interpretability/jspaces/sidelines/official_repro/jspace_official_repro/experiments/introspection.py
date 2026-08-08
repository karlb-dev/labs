"""verbal-introspection (plan §10.B): released 4-turn dialog + prefills;
steering = unit v_t x layer mean residual norm x strength over every
token of the trial-question turn (D2), every band layer; frozen ladder
{0,1,2,4,8,16}; score = rank of the surface token at the open quote."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from jlens.hooks import ActivationRecorder

from ..interventions import HookPlan, InterventionSession
from ..layers import PAPER_BAND
from ..paths import EXPERIMENTS_DIR
from ..readout import token_vectors
from ..rendering import find_token_span, preferred_token, render_chat
from ..scoring import forward_logits, rank_of

STRENGTHS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)


def run(model, lens, *, lane: str, out_dir: Path, band=PAPER_BAND,
        max_concepts: int | None = None) -> dict:
    data = json.loads((EXPERIMENTS_DIR / "verbal-introspection.json").read_text())
    tokenizer = model.tokenizer
    start = time.time()
    concepts = data["concepts"][:max_concepts] if max_concepts else data["concepts"]
    results = []
    for prefill_name, prefill in data["prefills"].items():
        messages = [dict(m) for m in data["intro_prompt"]]
        assert messages[-1]["role"] == "assistant" and messages[-1]["content"] == ""
        messages[-1]["content"] = prefill
        kwargs = {"enable_thinking": False} if lane == "qwen" else {}
        rendered = render_chat(model, messages, continue_final=True,
                               extra_template_kwargs=kwargs)
        question_text = messages[2]["content"]
        span = find_token_span(rendered, tokenizer, question_text.strip())
        positions = list(range(span[0], span[1] + 1))
        scored = rendered.final_position
        # Clean pass: layer mean residual norms over the steered span (D11:
        # prompt-level normalization) — one recorder pass.
        with ActivationRecorder(model.layers, at=list(band)) as recorder:
            model.forward(rendered.input_ids)
            norms = {
                layer: float(recorder.activations[layer][0, positions]
                             .detach().float().norm(dim=-1).mean())
                for layer in band
            }
        for concept in concepts:
            surface = concept["surface"]
            token = preferred_token(tokenizer, surface)
            row = {"concept": concept["name"], "surface": surface,
                   "prefill": prefill_name,
                   "tokenization_valid": token is not None}
            if token is None:
                row["state"] = "TOKENIZATION_GATED"
                results.append(row)
                continue
            vectors = {layer: token_vectors(lens, model, layer, [token])[0]
                       for layer in band}
            ladder = {}
            for strength in STRENGTHS:
                if strength == 0.0:
                    logits, _ = forward_logits(model, rendered.input_ids,
                                               positions=[scored])
                else:
                    plan = HookPlan(layers=list(band), positions=positions)
                    with InterventionSession(
                        model.layers, plan, kind="steer", vectors=vectors,
                        strength=strength, layer_mean_norms=norms,
                    ) as session:
                        logits, _ = forward_logits(model, rendered.input_ids,
                                                   positions=[scored])
                        session.assert_fires(1)
                rank = rank_of(logits[0], token)
                ladder[f"s{strength:g}"] = {
                    "rank": rank, "reciprocal_rank": 1.0 / rank,
                    "top1": tokenizer.decode([int(logits[0].argmax())]),
                }
            row["ladder"] = ladder
            row["state"] = "EXECUTED"
            results.append(row)
    executed = [r for r in results if r.get("state") == "EXECUTED"]
    def _median_rr(prefill_name, strength):
        values = sorted(
            r["ladder"][f"s{strength:g}"]["reciprocal_rank"]
            for r in executed if r["prefill"] == prefill_name)
        return values[len(values) // 2] if values else None
    summary = {
        "experiment": "verbal-introspection", "lane": lane,
        "band": list(band), "strengths": list(STRENGTHS),
        "n_concepts_released": len(data["concepts"]),
        "n_run": len(concepts),
        "n_executed_rows": len(executed),
        "n_tokenization_gated": sum(
            1 for r in results if r.get("state") == "TOKENIZATION_GATED"),
        "median_rr_by_strength": {
            prefill: {f"s{s:g}": _median_rr(prefill, s) for s in STRENGTHS}
            for prefill in data["prefills"]
        },
        "wall_seconds": time.time() - start,
        "rows": results,
    }
    path = out_dir / f"verbal_introspection_{lane}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(summary))
    return summary
