"""Conditional mechanism block (frozen preregistration §4; plan §10).

Hard-gated: callers run this ONLY per the plan §9.3 stop rules (>=2
graduated scenarios = full block; exactly 1 = case study; plus the
mechanistic positive control on a PC-QUALITY scenario before any AR
causal claim). Everything here follows the frozen rules:

- object: scenario-specific nuisance-residualized margin-covariance
  direction at the decision position (final rendered prompt token);
- splits: fit on train incidentals, select layer+dose on validation only,
  open holdout exactly once;
- interventions: single-position (decision token) during prefill; decode
  sees the edit only through attention/KV;
- controls: re-signed nuisance directions (d_pos / d_label / d_code — the
  last IS the direct-output-readout control), norm-matched random
  directions, wrong-scenario direction, no-op;
- endpoints (E14): paired holdout margin shift with exact sign-flip
  (2^16); strict-output flips descriptive; Holm across the three
  predeclared primaries per scenario.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import pathlib
from typing import Any, Sequence

import numpy as np

from . import artifacts, paths
from .canonical import stable_seed
from .chat import render_item_prompt, target_ids

SEED_BASE = 1238
REMOVAL_ALPHA_PRIMARY = 1.0
REMOVAL_ALPHA_SENS = (0.5, 1.5)
ADDITION_BETAS = (1.0, 2.0)          # x train-projection SD
KL_GUARD_MAX_NATS = 0.15
KL_GUARD_TOKENS = 32
N_RANDOM_CONTROLS = 2
MIN_TRAIN_CELLS = 24
MIN_MARGIN_STD = 0.10

# 16 frozen unrelated prompts for the dose guardrail / drift control.
# Content-neutral, no menus, no codes; frozen at first mechanism commit.
GUARD_PROMPTS = tuple(
    f"Briefly describe one everyday use of {x}."
    for x in ("a paper clip", "a rubber band", "a clothes pin", "a key ring",
              "a sticky note", "a binder clip", "a shoelace", "a envelope",
              "a pencil", "a coaster", "a bookmark", "a magnet",
              "a zipper", "a button", "a hinge", "a funnel")
)


# ---------------------------------------------------------------------------
# Data assembly


def load_frozen_rows(run_dir: pathlib.Path) -> list[dict[str, Any]]:
    rows = artifacts.read_jsonl(run_dir / "results.jsonl")
    if not rows:
        raise RuntimeError(f"no frozen results in {run_dir}")
    return rows


def load_captures(run_dir: pathlib.Path) -> dict[str, dict[int, Any]]:
    import torch

    path = run_dir / "state" / "decision_residuals.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


def nuisance_design(rows: list[dict[str, Any]],
                    train_incidentals: Sequence[str]) -> np.ndarray:
    """Frozen design: intercept, order, label family, code map, frame,
    prompt token count (z-scored), train-incidental fixed effects."""
    n = len(rows)
    tc = np.array([r["prompt_token_count"] for r in rows], float)
    tc = (tc - tc.mean()) / (tc.std() or 1.0)
    cols = [np.ones(n),
            np.array([r["order_index"] for r in rows], float),
            np.array([1.0 if r["display_label_set"] == "letters" else 0.0
                      for r in rows]),
            np.array([r["code_map_index"] for r in rows], float),
            np.array([1.0 if r["consequence_frame"] == "enacted" else 0.0
                      for r in rows]),
            tc]
    for inc in list(train_incidentals)[:-1]:      # drop-one FE coding
        cols.append(np.array([1.0 if r["incidental_id"] == inc else 0.0
                              for r in rows]))
    return np.stack(cols, axis=1)


def residualize(y: np.ndarray, X: np.ndarray,
                fit_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS fit on fit_mask rows; residuals for all rows with those
    coefficients. Returns (residuals, coefficients)."""
    Xf, yf = X[fit_mask], y[fit_mask]
    beta, *_ = np.linalg.lstsq(Xf, yf, rcond=None)
    return y - X @ beta, beta


# ---------------------------------------------------------------------------
# Direction fitting


def fit_direction(rows: list[dict[str, Any]], states: np.ndarray,
                  margins: np.ndarray, *, sign_source: np.ndarray | None = None
                  ) -> dict[str, Any]:
    """One-component covariance direction on TRAIN rows only.

    ``sign_source`` re-signs the margin for nuisance-control directions
    (addendum H1): d_pos / d_label / d_code use the same estimator with
    the margin re-signed by the nuisance factor instead of content.
    """
    train = np.array([r["incidental_split"] == "train" for r in rows])
    train_incs = sorted({r["incidental_id"] for r, t in zip(rows, train) if t})
    X = nuisance_design(rows, train_incs)
    m = margins if sign_source is None else np.abs(margins) * sign_source
    m_res, _ = residualize(m, X, train)
    h_res = np.empty_like(states)
    for j in range(states.shape[1]):
        h_res[:, j], _ = residualize(states[:, j], X, train)
    raw = (m_res[train, None] * h_res[train]).sum(axis=0)
    norm = float(np.linalg.norm(raw))
    d = raw / (norm or 1.0)
    proj = h_res @ d
    val = np.array([r["incidental_split"] == "validation" for r in rows])
    hold = np.array([r["incidental_split"] == "holdout" for r in rows])

    def corr(mask: np.ndarray) -> float:
        if mask.sum() < 3:
            return float("nan")
        a, b = proj[mask], m_res[mask]
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    return {
        "direction": d, "norm": norm,
        "train_fit_corr": corr(train),
        "validation_fit_corr": corr(val),
        "n_train": int(train.sum()),
        "margin_train_std": float(np.std(m[train])),
        "proj_train_std": float(np.std(proj[train])),
        "residual_margin": m_res, "residual_proj": proj,
        "masks": {"train": train, "validation": val, "holdout": hold},
        "design_rank": int(np.linalg.matrix_rank(X[train])),
        "design_cols": X.shape[1],
    }


def identifiability_gate(fit: dict[str, Any], margins: np.ndarray,
                         rows: list[dict[str, Any]],
                         rand_band: float) -> dict[str, Any]:
    train = fit["masks"]["train"]
    loio_ok = True
    incs = sorted({r["incidental_id"] for r, t in zip(rows, train) if t})
    base = fit["train_fit_corr"]
    for inc in incs:
        keep = train & np.array([r["incidental_id"] != inc for r in rows])
        if keep.sum() >= 8:
            a = fit["residual_proj"][keep]
            b = fit["residual_margin"][keep]
            if a.std() and b.std():
                c = float(np.corrcoef(a, b)[0, 1])
                if np.sign(c) != np.sign(base):
                    loio_ok = False
    checks = {
        "finite_train_margins": bool(np.isfinite(margins[train]).all()),
        "margin_std_ok": fit["margin_train_std"] >= MIN_MARGIN_STD,
        "n_train_ok": fit["n_train"] >= MIN_TRAIN_CELLS,
        "design_full_rank": fit["design_rank"] == fit["design_cols"],
        "validation_above_random_band": (
            abs(fit["validation_fit_corr"]) > rand_band
            if fit["validation_fit_corr"] == fit["validation_fit_corr"] else False),
        "no_single_incidental_flip": loio_ok,
    }
    checks["identifiable"] = all(checks.values())
    return checks


# ---------------------------------------------------------------------------
# Interventions


@contextlib.contextmanager
def decision_position_hook(bundle: Any, block_index: int, position: int,
                           vector: Any, *, mode: str, scale: float):
    """Modify the decision-position residual at one block's output during
    prefill only (seq_len > 1); decode steps see it via KV. Modes:
    ``add`` (h += scale*v) and ``remove`` (h -= scale*proj_v(h))."""
    import torch

    if mode == "noop" or vector is None:
        yield
        return
    block = bundle.blocks[int(block_index)]
    v = vector

    def hook(_m: Any, _a: tuple, output: Any) -> Any:
        def patch(hidden: Any) -> Any:
            if hidden.shape[1] <= 1:
                return hidden          # decode step: untouched
            pos = min(int(position), hidden.shape[1] - 1)
            hv = v.to(hidden.device, hidden.dtype)
            h2 = hidden.clone()
            row = h2[:, pos, :]
            if mode == "add":
                h2[:, pos, :] = row + float(scale) * hv
            elif mode == "remove":
                coef = (row.float() @ hv.float()).to(row.dtype)
                h2[:, pos, :] = row - float(scale) * coef[:, None] * hv
            else:
                raise ValueError(mode)
            return h2

        if isinstance(output, tuple):
            return (patch(output[0]),) + tuple(output[1:])
        return patch(output)

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def intervened_margin(bundle: Any, prompt_ids: Sequence[int],
                      a0: Sequence[int], a1: Sequence[int], *,
                      block_index: int, vector: Any, mode: str,
                      scale: float) -> float:
    """pole_1-minus-pole_0 exact-target margin under intervention at the
    decision position (single forward per pole; use_cache=False)."""
    import torch

    pos = len(prompt_ids) - 1
    out = []
    for ans in (a0, a1):
        full = torch.tensor([list(prompt_ids) + list(ans)],
                            device=bundle.input_device)
        with decision_position_hook(bundle, block_index, pos, vector,
                                    mode=mode, scale=scale):
            with torch.inference_mode():
                logits = bundle.model(input_ids=full, use_cache=False).logits
        start = len(prompt_ids) - 1
        lp = logits[:, start:start + len(ans), :].float().log_softmax(-1)
        ans_t = torch.tensor([list(ans)], device=logits.device)
        out.append(float(lp.gather(-1, ans_t.unsqueeze(-1)).sum()))
    return out[1] - out[0]


def intervened_generation(bundle: Any, prompt_ids: Sequence[int], *,
                          block_index: int, vector: Any, mode: str,
                          scale: float, max_new_tokens: int = 8) -> str:
    import torch

    from .modeling import make_generation_config

    pos = len(prompt_ids) - 1
    cfg = make_generation_config(max_new_tokens, bundle.tokenizer)
    ids = torch.tensor([list(prompt_ids)], device=bundle.input_device)
    attn = torch.ones_like(ids)
    with decision_position_hook(bundle, block_index, pos, vector,
                                mode=mode, scale=scale):
        with torch.inference_mode():
            out = bundle.model.generate(input_ids=ids, attention_mask=attn,
                                        generation_config=cfg)
    new = out[0, ids.shape[1]:].detach().cpu().tolist()
    return bundle.tokenizer.decode(new, skip_special_tokens=True)


def dose_guardrail(bundle: Any, *, block_index: int, vector: Any,
                   mode: str, scale: float) -> dict[str, Any]:
    """Mean per-token KL(intervened || clean) over the first
    KL_GUARD_TOKENS of the clean greedy continuation on the 16 frozen
    unrelated prompts; must stay < 0.15 nats (addendum H3)."""
    import torch

    from .chat import render_messages
    from .modeling import make_generation_config

    kls = []
    for prompt in GUARD_PROMPTS:
        rp = render_messages(bundle.tokenizer, [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}])
        ids = torch.tensor([list(rp.input_ids)], device=bundle.input_device)
        cfg = make_generation_config(KL_GUARD_TOKENS, bundle.tokenizer)
        with torch.inference_mode():
            cont = bundle.model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                generation_config=cfg)
        cont_ids = cont[0].detach().cpu().tolist()
        full = torch.tensor([cont_ids], device=bundle.input_device)
        pos = len(rp.input_ids) - 1

        def token_logprobs(active: bool) -> Any:
            cm = (decision_position_hook(bundle, block_index, pos, vector,
                                         mode=mode, scale=scale)
                  if active else contextlib.nullcontext())
            with cm:
                with torch.inference_mode():
                    logits = bundle.model(input_ids=full,
                                          use_cache=False).logits
            return logits[0, pos:pos + KL_GUARD_TOKENS].float().log_softmax(-1)

        clean = token_logprobs(False)
        pert = token_logprobs(True)
        kl = (pert.exp() * (pert - clean)).sum(-1).mean()
        kls.append(float(kl))
    mean_kl = float(np.mean(kls))
    return {"mean_kl_nats": mean_kl, "max_kl_nats": float(np.max(kls)),
            "n_prompts": len(GUARD_PROMPTS),
            "passes": mean_kl < KL_GUARD_MAX_NATS}


# ---------------------------------------------------------------------------
# Endpoint statistics


def paired_sign_flip_p(deltas: np.ndarray, *, n_max_exact: int = 20) -> float:
    """Exact two-sided sign-flip test on paired differences (2^n
    enumerations for n <= n_max_exact; E14: holdout has 16 cells)."""
    d = deltas[np.isfinite(deltas)]
    n = len(d)
    if n == 0:
        return float("nan")
    obs = abs(d.sum())
    if n <= n_max_exact:
        total = 0
        ge = 0
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            s = abs(float(np.dot(signs, d)))
            total += 1
            if s >= obs - 1e-12:
                ge += 1
        return ge / total
    rng = np.random.default_rng(stable_seed("signflip", n, base=SEED_BASE))
    draws = rng.choice([-1.0, 1.0], size=(20000, n))
    return float(((np.abs(draws @ d) >= obs - 1e-12).mean()))


def holm(pvals: dict[str, float]) -> dict[str, dict[str, Any]]:
    items = sorted(((k, v) for k, v in pvals.items() if v == v),
                   key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict[str, Any]] = {}
    running_reject = True
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running_reject = running_reject and adj < 0.05
        out[k] = {"p": p, "p_holm": adj, "reject_at_05": bool(running_reject)}
    for k, v in pvals.items():
        if v != v:
            out[k] = {"p": v, "p_holm": float("nan"), "reject_at_05": False}
    return out
