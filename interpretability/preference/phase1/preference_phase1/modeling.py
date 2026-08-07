"""Model execution layer: interp_bench-as-library plus exact-ID scoring.

The bench supplies validated loading (anatomy resolution, dtype/device),
the residual-stream convention, and continuous-batch generation. This
module adds what Lab 38 needs on top: exact-ID teacher-forced target
scoring (single and batched, with a batch-invariance check), strict greedy
generation from pre-rendered ids, neutral-prior measurement, and
decision-position residual capture at the frozen relative-depth grid.
"""

from __future__ import annotations

import argparse
import contextlib
import pathlib
from typing import Any, Sequence

from . import paths
from .models import ModelPin, model_manifest
from .provenance import require_cuda

RELATIVE_DEPTH_GRID = (0.25, 0.40, 0.55, 0.70, 0.85)


def _bench():
    import sys

    interp = paths.interp_root()
    if str(interp) not in sys.path:
        sys.path.insert(0, str(interp))
    import interp_bench as bench

    return bench


def load_bundle(pin: ModelPin, run_dir: pathlib.Path, *, device: str = "auto",
                require_gpu: bool = False) -> tuple[Any, Any]:
    """Load the pinned model through the bench loader; returns (ctx, bundle).

    ``require_gpu`` enforces the hard GPU gate for model_tier b/c stages.
    """
    if require_gpu:
        require_cuda()
    bench = _bench()
    import os

    os.environ.setdefault("HF_HUB_CACHE", str(paths.hf_local_cache()))
    args = argparse.Namespace(
        lab="lab38", model=pin.model_id, model_revision=pin.revision,
        trust_remote_code=False, local_files_only=False,
        attn_implementation="auto", low_cpu_mem_usage=True,
        device=device, dtype=pin.dtype, quantization="none",
        tier=pin.key, prompt_set="full", max_examples=0, seed=1238,
        no_plots=True, hook_tolerance=0.0, allow_hook_mismatch=False,
        save_tensors=False, append_ledger=False,
    )
    ctx = bench.RunContext(run_dir=pathlib.Path(run_dir), args=args)
    bundle = bench.load_model_and_tokenizer(ctx)
    ctx.bind_model(bundle)
    return ctx, bundle


@contextlib.contextmanager
def inference_mode():
    import torch

    with torch.inference_mode():
        yield


def conditional_sequence_logprob(model: Any, input_device: Any,
                                 prompt_ids: Sequence[int],
                                 answer_ids: Sequence[int]) -> dict[str, Any]:
    """P(answer | prompt) summed over the FULL answer sequence (plan §4.4;
    first-token-only scoring is forbidden). Single-row exact reference."""
    import torch

    if not answer_ids:
        raise ValueError("answer must contain at least one token")
    full = torch.tensor([list(prompt_ids) + list(answer_ids)],
                        device=input_device)
    with torch.inference_mode():
        logits = model(input_ids=full, use_cache=False).logits
    start = len(prompt_ids) - 1
    stop = full.shape[1] - 1
    log_probs = logits[:, start:stop, :].float().log_softmax(dim=-1)
    ans = torch.tensor([list(answer_ids)], device=logits.device)
    gathered = log_probs.gather(-1, ans.unsqueeze(-1)).squeeze(-1)[0]
    vals = gathered.detach().cpu()
    return {
        "token_logprobs": [float(v) for v in vals],
        "sum_logprob": float(vals.sum()),
        "token_count": len(answer_ids),
        "finite": bool(torch.isfinite(vals).all()),
    }


def batched_pair_margins(model: Any, input_device: Any,
                         prompt_ids_list: Sequence[Sequence[int]],
                         answer_pairs: Sequence[tuple[Sequence[int], Sequence[int]]],
                         *, pad_token_id: int,
                         batch_size: int = 16) -> list[dict[str, Any]]:
    """Teacher-forced exact-target scoring for many (prompt, pole_0, pole_1)
    rows. Right-padded with attention-mask-correct label masking (addendum
    F); returns per-row dicts with both sums and the pole_1-minus-pole_0
    margin. Batch composition is canonical-order stable."""
    import torch

    jobs = []
    for row_idx, (p_ids, (a0, a1)) in enumerate(zip(prompt_ids_list, answer_pairs)):
        for pole, ans in ((0, a0), (1, a1)):
            jobs.append((row_idx, pole, list(p_ids), list(ans)))
    results: dict[tuple[int, int], dict[str, Any]] = {}
    for lo in range(0, len(jobs), batch_size):
        chunk = jobs[lo:lo + batch_size]
        maxlen = max(len(p) + len(a) for _, _, p, a in chunk)
        input_ids = torch.full((len(chunk), maxlen), int(pad_token_id),
                               dtype=torch.long)
        attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for i, (_, _, p, a) in enumerate(chunk):
            seq = p + a
            input_ids[i, :len(seq)] = torch.tensor(seq)
            attn[i, :len(seq)] = 1
        input_ids = input_ids.to(input_device)
        attn = attn.to(input_device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attn,
                           use_cache=False).logits.float()
        logp = logits.log_softmax(dim=-1)
        for i, (row_idx, pole, p, a) in enumerate(chunk):
            start = len(p) - 1
            vals = []
            for j, tok in enumerate(a):
                vals.append(float(logp[i, start + j, tok]))
            results[(row_idx, pole)] = {
                "token_logprobs": vals,
                "sum_logprob": float(sum(vals)),
                "token_count": len(a),
                "finite": all(v == v and abs(v) != float("inf") for v in vals),
            }
    out = []
    for row_idx in range(len(prompt_ids_list)):
        r0, r1 = results[(row_idx, 0)], results[(row_idx, 1)]
        out.append({
            "q_pole_0": r0["sum_logprob"], "q_pole_1": r1["sum_logprob"],
            "margin_pole1_minus_pole0": r1["sum_logprob"] - r0["sum_logprob"],
            "tokens_pole_0": r0["token_count"], "tokens_pole_1": r1["token_count"],
            "finite": r0["finite"] and r1["finite"],
            "detail_pole_0": r0, "detail_pole_1": r1,
        })
    return out


def make_generation_config(max_new_tokens: int, tokenizer: Any) -> Any:
    """Explicit frozen GenerationConfig (addendum F): never rely on the
    model's shipped generation_config.json."""
    from transformers import GenerationConfig

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    return GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=int(max_new_tokens),
        pad_token_id=int(pad_id), eos_token_id=tokenizer.eos_token_id,
        temperature=None, top_p=None, top_k=None,
        repetition_penalty=1.0,
    )


def generate_strict_batch(model: Any, tokenizer: Any, input_device: Any,
                          prompt_ids_list: Sequence[Sequence[int]],
                          *, max_new_tokens: int,
                          batch_size: int = 16) -> list[str]:
    """Greedy generation from exact pre-rendered ids, left-padded batches
    (addendum F). Returns decoded new-token strings (special tokens
    stripped). Deterministic under frozen batch composition."""
    import torch

    gen_config = make_generation_config(max_new_tokens, tokenizer)
    outs: list[str] = []
    for lo in range(0, len(prompt_ids_list), batch_size):
        chunk = [list(p) for p in prompt_ids_list[lo:lo + batch_size]]
        maxlen = max(len(p) for p in chunk)
        pad_id = int(gen_config.pad_token_id)
        input_ids = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for i, p in enumerate(chunk):
            input_ids[i, maxlen - len(p):] = torch.tensor(p)
            attn[i, maxlen - len(p):] = 1
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids.to(input_device),
                attention_mask=attn.to(input_device),
                generation_config=gen_config,
            )
        for i in range(len(chunk)):
            new_ids = out[i, maxlen:].detach().cpu().tolist()
            outs.append(tokenizer.decode(new_ids, skip_special_tokens=True))
    return outs


def neutral_logprob_fn(pin: ModelPin, *, run_dir: pathlib.Path | None = None) -> Any:
    """Closure measuring each code's summed logprob as the immediate
    assistant continuation of the frozen neutral context (addendum E4)."""
    from .chat import render_messages, target_ids
    from .targets import NEUTRAL_CONTEXT

    scratch = run_dir or (paths.runs_root() / "lab38_codebook_priors")
    scratch.mkdir(parents=True, exist_ok=True)
    ctx, bundle = load_bundle(pin, scratch, require_gpu=(pin.key != "a"))
    rp = render_messages(bundle.tokenizer, [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": NEUTRAL_CONTEXT},
    ])
    if not rp.parity_ok:
        raise RuntimeError("chat-template parity failed in neutral-prior probe")

    def fn(code: str) -> float:
        ids = target_ids(bundle.tokenizer, code, leading_space=False)
        return conditional_sequence_logprob(
            bundle.model, bundle.input_device, rp.input_ids, ids
        )["sum_logprob"]

    fn.bundle = bundle          # type: ignore[attr-defined]
    fn.ctx = ctx                # type: ignore[attr-defined]
    fn.manifest = model_manifest(pin, bundle.tokenizer)  # type: ignore[attr-defined]
    return fn


def depth_indices(n_layers: int, grid: Sequence[float] = RELATIVE_DEPTH_GRID) -> list[int]:
    """Stream depths for the frozen relative-depth grid. Stream k is the
    pre-norm residual after k blocks (bench convention)."""
    return sorted({max(1, min(n_layers, round(f * n_layers))) for f in grid})


def capture_decision_residuals(bundle: Any, prompt_ids: Sequence[int],
                               depths: Sequence[int]) -> dict[int, Any]:
    """Residual vectors at the decision position (final prompt token) for
    the requested stream depths, via one exact-ID forward pass with hidden
    states (never re-tokenizes; Lab 15 run_ids discipline)."""
    import torch

    captured: dict[str, Any] = {}

    def final_norm_pre_hook(module: Any, hook_args: tuple) -> None:
        captured["final_prenorm"] = hook_args[0].detach()

    handle = bundle.final_norm.register_forward_pre_hook(final_norm_pre_hook)
    try:
        with torch.inference_mode():
            out = bundle.model(
                input_ids=torch.tensor([list(prompt_ids)],
                                       device=bundle.input_device),
                output_hidden_states=True, use_cache=False)
    finally:
        handle.remove()
    n_layers = bundle.anatomy.n_layers
    result: dict[int, Any] = {}
    for depth in depths:
        if depth < n_layers:
            h = out.hidden_states[depth][0, -1]
        else:
            h = captured["final_prenorm"][0, -1]
        result[int(depth)] = h.detach().float().cpu()
    return result
