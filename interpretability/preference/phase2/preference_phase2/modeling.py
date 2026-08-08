"""Model execution layer (interp_bench-as-library; Phase 1 conventions).

Rules of record: single-row scoring with float32 log_softmax before
gather (bf16 batched kernels shift margins ~0.25 nats — Phase 1 finding);
generation LEFT-pads from exact pre-rendered ids; captures come from ONE
forward with output_hidden_states at the frozen site token indices and
relative depths (all < n_layers, so hidden_states[k] suffices — stream k
is the pre-norm residual after k blocks).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any, Sequence

from . import paths
from .models import ModelPin
from .provenance import require_cuda
from .schema import RELATIVE_DEPTHS

SEED = 2262


def _bench():
    root = str(paths.interp_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    import interp_bench as bench

    return bench


def load_bundle(pin: ModelPin, run_dir, *, device: str = "auto",
                require_gpu: bool = True):
    import argparse
    import os

    if require_gpu:
        require_cuda()
    os.environ.setdefault("HF_HUB_CACHE", str(paths.hf_local_cache()))
    bench = _bench()
    args = argparse.Namespace(
        lab="lab38", model=pin.model_id, model_revision=pin.revision,
        trust_remote_code=False, local_files_only=False,
        attn_implementation="auto", low_cpu_mem_usage=True, device=device,
        dtype=pin.dtype, quantization="none", tier=pin.key,
        prompt_set="full", max_examples=0, seed=SEED, no_plots=True,
        hook_tolerance=0.0, allow_hook_mismatch=False, save_tensors=False,
        append_ledger=False)
    ctx = bench.RunContext(run_dir, args)
    bundle = bench.load_model_and_tokenizer(ctx)
    ctx.bind_model(bundle)
    return ctx, bundle


@contextmanager
def inference_mode():
    import torch

    with torch.inference_mode():
        yield


def depth_indices(n_layers: int,
                  grid: Sequence[float] = RELATIVE_DEPTHS) -> list[int]:
    return sorted({max(1, min(n_layers - 1, round(f * n_layers)))
                   for f in grid})


def sequence_logprob(model, input_device, prompt_ids: Sequence[int],
                     answer_ids: Sequence[int]) -> dict[str, Any]:
    """Single-row exact scoring (use_cache=False; float32 upcast)."""
    import torch

    if not answer_ids:
        raise ValueError("empty answer")
    full = list(prompt_ids) + list(answer_ids)
    ids = torch.tensor([full], device=input_device)
    with inference_mode():
        logits = model(input_ids=ids, use_cache=False).logits
    start, stop = len(prompt_ids) - 1, len(full) - 1
    lp = logits[:, start:stop, :].float().log_softmax(dim=-1)
    tok = torch.tensor(list(answer_ids), device=lp.device)
    token_lps = lp[0, torch.arange(len(answer_ids)), tok]
    vals = [float(v) for v in token_lps]
    import math

    return {"token_logprobs": vals, "sum_logprob": float(sum(vals)),
            "first_logprob": vals[0],
            "finite": all(math.isfinite(v) for v in vals)}


def pair_margins(model, input_device, prompt_ids, ids_a, ids_b) -> dict[str, Any]:
    qa = sequence_logprob(model, input_device, prompt_ids, ids_a)
    qb = sequence_logprob(model, input_device, prompt_ids, ids_b)
    return {
        "q_a": qa["sum_logprob"], "q_b": qb["sum_logprob"],
        "margin_full_a_minus_b": qa["sum_logprob"] - qb["sum_logprob"],
        "margin_first_a_minus_b": qa["first_logprob"] - qb["first_logprob"],
        "tokens_a": len(ids_a), "tokens_b": len(ids_b),
        "finite": qa["finite"] and qb["finite"],
    }


def make_generation_config(max_new_tokens: int, tokenizer):
    from transformers import GenerationConfig

    return GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
        pad_token_id=(tokenizer.pad_token_id
                      if tokenizer.pad_token_id is not None
                      else tokenizer.eos_token_id),
        eos_token_id=tokenizer.eos_token_id, temperature=None, top_p=None,
        top_k=None, repetition_penalty=1.0)


def generate_batch(model, tokenizer, input_device,
                   prompt_ids_list: list[Sequence[int]], *,
                   max_new_tokens: int, batch_size: int = 16) -> list[str]:
    """Greedy from exact pre-rendered ids; LEFT-padded batches."""
    import torch

    pad = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None
           else tokenizer.eos_token_id)
    cfg = make_generation_config(max_new_tokens, tokenizer)
    outs: list[str] = []
    for i in range(0, len(prompt_ids_list), batch_size):
        chunk = prompt_ids_list[i:i + batch_size]
        maxlen = max(len(p) for p in chunk)
        ids = torch.full((len(chunk), maxlen), pad, dtype=torch.long)
        mask = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for j, p in enumerate(chunk):
            ids[j, maxlen - len(p):] = torch.tensor(list(p))
            mask[j, maxlen - len(p):] = 1
        ids = ids.to(input_device)
        mask = mask.to(input_device)
        with inference_mode():
            out = model.generate(input_ids=ids, attention_mask=mask,
                                 generation_config=cfg)
        for j in range(len(chunk)):
            new = out[j, maxlen:]
            outs.append(tokenizer.decode(new, skip_special_tokens=True))
    return outs


def capture_sites(model, input_device, prompt_ids: Sequence[int],
                  site_token_index: dict[str, int],
                  depths: Sequence[int]) -> dict[str, dict[int, Any]]:
    """ONE forward with output_hidden_states; returns
    {site: {depth: bf16 CPU tensor}} at the frozen token positions."""
    import torch

    ids = torch.tensor([list(prompt_ids)], device=input_device)
    with inference_mode():
        out = model(input_ids=ids, use_cache=False,
                    output_hidden_states=True)
    store: dict[str, dict[int, Any]] = {}
    n = len(prompt_ids)
    for site, tok_idx in site_token_index.items():
        idx = min(int(tok_idx), n - 1)
        store[site] = {
            int(d): out.hidden_states[int(d)][0, idx].detach()
            .to(torch.bfloat16).cpu()
            for d in depths}
    del out
    return store


def batch_invariance_probe(model, tokenizer, input_device,
                           prompt_ids_list, pairs, *,
                           batch_size: int = 8) -> dict[str, Any]:
    """Single-row replay determinism (hard gate: exactly 0.0) + batched
    generation equality (hard gate) + batched-margin delta (recorded,
    informational — Phase 1 finding)."""
    import torch

    replay_delta = 0.0
    margins_single = []
    for prompt_ids, (ids_a, ids_b) in zip(prompt_ids_list, pairs):
        m1 = pair_margins(model, input_device, prompt_ids, ids_a, ids_b)
        m2 = pair_margins(model, input_device, prompt_ids, ids_a, ids_b)
        replay_delta = max(replay_delta,
                           abs(m1["margin_full_a_minus_b"]
                               - m2["margin_full_a_minus_b"]))
        margins_single.append(m1["margin_full_a_minus_b"])
    gen_single = generate_batch(model, tokenizer, input_device,
                                prompt_ids_list, max_new_tokens=8,
                                batch_size=1)
    gen_batched = generate_batch(model, tokenizer, input_device,
                                 prompt_ids_list, max_new_tokens=8,
                                 batch_size=batch_size)
    return {
        "replay_max_abs_delta": replay_delta,
        "replay_deterministic": replay_delta == 0.0,
        "generation_batch_equal": gen_single == gen_batched,
        "n_probe_rows": len(prompt_ids_list),
    }
