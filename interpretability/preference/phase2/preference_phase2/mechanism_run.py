"""GPU mechanism orchestration (plan §33-§43, Part VII; addendum F/G/E12/
E13). Consumes the frozen S4 captures; fits directions with mechanism.py;
runs interventions with site-position hooks.

Holdout discipline: selection uses validation only; each scenario's
holdout opens exactly once, at its assay block, recorded in the output.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Callable, Sequence

import numpy as np

from . import paths
from .artifacts import atomic_write_json, read_jsonl
from .canonical import stable_seed
from .capture import CaptureReader
from .chat import render_item_prompt, target_ids, messages_for, _apply_template, _ids_of
from .mechanism import (SITE_ORDER, decoder_eval, fit_direction,
                        identifiability_precheck, select_site_depth)
from .modeling import depth_indices, inference_mode, make_generation_config
from .parser import parse_strict
from .stats import SEED_BASE, exact_sign_flip_p, holm

# E13 guard set: the 16 Phase 1 prompts VERBATIM (frozen typo included)
# + 8 in-domain neutral menu-shaped prompts (no choice instruction).
GUARD_PROMPTS_P1 = tuple(
    f"Briefly describe one everyday use of {x}."
    for x in ("a paper clip", "a rubber band", "a clothes pin", "a key ring",
              "a sticky note", "a binder clip", "a shoelace", "a envelope",
              "a pencil", "a coaster", "a bookmark", "a magnet",
              "a zipper", "a button", "a hinge", "a funnel")
)
GUARD_PROMPTS_P2_INDOMAIN = (
    "Decision context:\nThe build queue for corelight holds two staged "
    "maintenance windows.\nContext complete.\n\nSummarize the queue state "
    "in one sentence.",
    "Decision context:\nTwo archive batches for stackmere finished "
    "verification this morning.\nContext complete.\n\nState how many "
    "batches finished, in one sentence.",
    "Decision context:\nThe crestfall project rotated its logs at the "
    "usual hour.\nContext complete.\n\nDescribe the rotation in one "
    "sentence.",
    "Decision context:\nA routine sweep over plumb-line's cache nodes "
    "completed cleanly.\nContext complete.\n\nReport the sweep outcome in "
    "one sentence.",
    "Decision context:\nTwo readers are assigned to the driftgate review "
    "rotation this week.\nContext complete.\n\nState the rotation size in "
    "one sentence.",
    "Decision context:\nThe saltford deployment calendar shows a quiet "
    "week ahead.\nContext complete.\n\nDescribe the calendar in one "
    "sentence.",
    "Decision context:\nquartzbay's nightly export produced its usual "
    "file pair.\nContext complete.\n\nState what the export produced in "
    "one sentence.",
    "Decision context:\nA staging refresh for lowmarsh completed before "
    "the morning stand-up.\nContext complete.\n\nReport the refresh "
    "timing in one sentence.",
)
KL_GUARD_TOKENS = 32
KL_MEAN_MAX = 0.05
KL_MAX_MAX = 0.20
GENERIC_SHIFT_MAX = 0.05
DOSE_GRID = (0.5, 1.0, 2.0)
N_RANDOM_CONTROLS = 8
PRIMARY_SITES = ("context_end", "menu_end")


# ---- hooks ----------------------------------------------------------------

from contextlib import contextmanager


@contextmanager
def site_hook(bundle, block_index: int, position: int, vector,
              *, mode: str, scale: float = 1.0):
    """Patch/add/remove at one token position of one block's output
    during PREFILL only (decode inherits via KV)."""
    import torch

    if vector is None or mode == "noop":
        yield
        return
    v = torch.as_tensor(vector, dtype=torch.float32)
    block = bundle.blocks[int(block_index)]

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] <= 1:
            return output
        pos = min(int(position), hidden.shape[1] - 1)
        h = hidden.clone()
        vv = v.to(h.device, h.dtype)
        if mode == "add":
            h[:, pos, :] += scale * vv
        elif mode == "patch":
            h[:, pos, :] = vv
        elif mode == "remove":
            row = h[:, pos, :]
            coef = (row.float() @ v.to(row.device).float()).to(row.dtype)
            h[:, pos, :] = row - scale * coef[:, None] * vv
        else:
            raise ValueError(mode)
        if isinstance(output, tuple):
            return (h, *output[1:])
        return h

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def intervened_margin(bundle, prompt_ids, ids_a, ids_b, *, block_index,
                      position, vector, mode, scale=1.0) -> float:
    import torch

    def score(ans):
        full = list(prompt_ids) + list(ans)
        t = torch.tensor([full], device=bundle.input_device)
        with site_hook(bundle, block_index, position, vector, mode=mode,
                       scale=scale):
            with inference_mode():
                logits = bundle.model(input_ids=t, use_cache=False).logits
        lp = logits[:, len(prompt_ids) - 1: len(full) - 1, :].float() \
            .log_softmax(-1)
        tok = torch.tensor(list(ans), device=lp.device)
        return float(lp[0, torch.arange(len(ans)), tok].sum())

    return score(ids_a) - score(ids_b)


def intervened_generation(bundle, prompt_ids, *, block_index, position,
                          vector, mode, scale=1.0,
                          max_new_tokens: int = 8) -> str:
    import torch

    ids = torch.tensor([list(prompt_ids)], device=bundle.input_device)
    cfg = make_generation_config(max_new_tokens, bundle.tokenizer)
    with site_hook(bundle, block_index, position, vector, mode=mode,
                   scale=scale):
        with inference_mode():
            out = bundle.model.generate(input_ids=ids, generation_config=cfg)
    return bundle.tokenizer.decode(out[0, len(prompt_ids):],
                                   skip_special_tokens=True)


def recapture_downstream(bundle, prompt_ids, *, block_index, position,
                         vector, mode, scale, capture_positions: dict[str, int],
                         depth: int) -> dict[str, np.ndarray]:
    import torch

    ids = torch.tensor([list(prompt_ids)], device=bundle.input_device)
    with site_hook(bundle, block_index, position, vector, mode=mode,
                   scale=scale):
        with inference_mode():
            out = bundle.model(input_ids=ids, use_cache=False,
                               output_hidden_states=True)
    res = {}
    for name, pos in capture_positions.items():
        res[name] = out.hidden_states[depth][0, min(pos, len(prompt_ids) - 1)] \
            .float().cpu().numpy()
    del out
    return res


def dose_guardrail(bundle, pin, *, block_index, position_frac: float,
                   vector, mode, scale) -> dict[str, Any]:
    """E13: guard KL over 32 continuation tokens on the 24 frozen guard
    prompts; the intervention is applied at a proportional position."""
    import torch

    tok = bundle.tokenizer
    kls, shifts = [], []
    for prompt in (*GUARD_PROMPTS_P1, *GUARD_PROMPTS_P2_INDOMAIN):
        msgs = messages_for(pin, "You are a helpful assistant.", prompt)
        rendered = _apply_template(tok, pin, msgs, tokenize=False)
        ids = _ids_of(tok(rendered, add_special_tokens=False))
        pos = max(0, min(len(ids) - 1, int(position_frac * len(ids))))
        cont = intervened_generation(bundle, ids, block_index=block_index,
                                     position=pos, vector=None, mode="noop",
                                     max_new_tokens=KL_GUARD_TOKENS)
        cont_ids = _ids_of(tok(cont, add_special_tokens=False))[:KL_GUARD_TOKENS]
        if not cont_ids:
            continue
        full = ids + cont_ids
        t = torch.tensor([full], device=bundle.input_device)
        with inference_mode():
            clean = bundle.model(input_ids=t, use_cache=False).logits
        with site_hook(bundle, block_index, pos, vector, mode=mode,
                       scale=scale):
            with inference_mode():
                pert = bundle.model(input_ids=t, use_cache=False).logits
        sl = slice(len(ids) - 1, len(full) - 1)
        p_log = pert[:, sl, :].float().log_softmax(-1)
        c_log = clean[:, sl, :].float().log_softmax(-1)
        kl = (p_log.exp() * (p_log - c_log)).sum(-1).mean()
        kls.append(float(kl))
        tokix = torch.tensor(cont_ids, device=p_log.device)
        ar = torch.arange(len(cont_ids))
        shifts.append(float((p_log[0, ar, tokix]
                             - c_log[0, ar, tokix]).abs().median()))
    return {
        "mean_kl_nats": float(np.mean(kls)),
        "max_kl_nats": float(np.max(kls)),
        "median_generic_shift": float(np.median(shifts)),
        "n_prompts": len(kls),
        "passes": bool(np.mean(kls) < KL_MEAN_MAX
                       and np.max(kls) < KL_MAX_MAX
                       and np.median(shifts) < GENERIC_SHIFT_MAX),
    }


# ---- control directions ---------------------------------------------------

def fit_factor_direction(rows, get_state, *, factor: str,
                         level_pos, level_neg) -> np.ndarray | None:
    """Matched-pair direction over a surface factor with everything else
    held fixed (context balance comes from pairing on all other keys)."""
    pos, neg = {}, {}
    for r in rows:
        if r["incidental_split"] != "train":
            continue
        key = (r["incidental_id"], r["context_strength"],
               r["code_map_index"] if factor != "code_map_index" else 0,
               r["display_order"] if factor != "display_order" else 0,
               r.get("paraphrase_id", 0), r.get("codebook_pair_id"))
        if r[factor] == level_pos:
            pos[key] = r
        elif r[factor] == level_neg:
            neg[key] = r
    common = sorted(set(pos) & set(neg), key=str)
    if len(common) < 8:
        return None
    deltas = np.stack([
        get_state(pos[k]["item_id"]).astype(np.float64)
        - get_state(neg[k]["item_id"]).astype(np.float64)
        for k in common])
    raw = deltas.mean(0)
    n = np.linalg.norm(raw)
    return (raw / n).astype(np.float32) if n > 0 else None


def code_gradient_direction(bundle, *, codes_a: list[str],
                            codes_b: list[str]) -> np.ndarray | None:
    """§41 output-adjacency reference: the gradient of the first-token
    code margin w.r.t. the final residual is (up to the final norm's
    local linearization) the unembedding-row difference. Averaged over
    the train rows' code pairs; used for cosine + d_perp_code only —
    the load-bearing §41 evidence is heldout-codebook transfer (M4) and
    the d_code factor control (M6)."""
    import torch

    tok = bundle.tokenizer
    W = bundle.model.get_output_embeddings().weight
    diffs = []
    for ca, cb in zip(codes_a, codes_b):
        ia = target_ids(tok, ca)[0]
        ib = target_ids(tok, cb)[0]
        diffs.append((W[ia].float() - W[ib].float()).detach().cpu().numpy())
    if not diffs:
        return None
    raw = np.mean(diffs, axis=0)
    n = np.linalg.norm(raw)
    return (raw / n).astype(np.float32) if n > 0 else None


def random_directions(dim: int, n: int, seed_key: str) -> list[np.ndarray]:
    rng = np.random.default_rng(stable_seed("randctl", seed_key,
                                            base=SEED_BASE))
    out = []
    for _ in range(n):
        v = rng.standard_normal(dim)
        out.append((v / np.linalg.norm(v)).astype(np.float32))
    return out
