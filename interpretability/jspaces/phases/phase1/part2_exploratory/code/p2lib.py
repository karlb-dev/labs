# p2lib — publication-conformance utilities for J-space Part 2.
#
# Vendored from the forensic-review addendum §10 (reference implementations
# offered as testable design patterns), with self-tests in
# scripts/p2lib_selftest.py. These replace, for all NEW code:
#   raw QR                    -> orthonormal_basis_from_rows (SVD, rank-safe)
#   unprotected dyn selection -> select_output_protected_j_basis
#   both-phase hooks          -> PhaseControlledAblator + intervention_phase
#   batch-local-mean variance -> RunningVectorMoments (Welford/Chan merge)
#   first-token scoring       -> conditional_sequence_logprob
#   per-condition bootstraps  -> paired_cluster_bootstrap (+ equivalence)
#   fresh per-dose bases      -> seeded_random_orthobasis (nested prefixes)
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch


# ---------------------------------------------------------------- 10.1
@dataclass(frozen=True)
class BasisResult:
    basis: torch.Tensor          # [d_model, effective_rank]
    singular_values: torch.Tensor
    effective_rank: int
    requested_rows: int
    condition_number: float | None


def orthonormal_basis_from_rows(
    rows: torch.Tensor,
    *,
    relative_tolerance: float = 1e-5,
    absolute_tolerance: float = 1e-7,
) -> BasisResult:
    """Numerical column span of row vectors; effective rank is the dose."""
    if rows.ndim != 2:
        raise ValueError(f"expected [n_rows, d_model], got {tuple(rows.shape)}")
    if rows.shape[0] == 0:
        empty = rows.new_zeros((rows.shape[1], 0), dtype=torch.float32)
        return BasisResult(empty, rows.new_zeros(0), 0, 0, None)

    matrix = rows.float().T.contiguous()
    u, s, _ = torch.linalg.svd(matrix, full_matrices=False)
    if s.numel() == 0 or not torch.isfinite(s).all():
        raise FloatingPointError("non-finite singular spectrum")

    threshold = max(absolute_tolerance, relative_tolerance * float(s[0]))
    rank = int((s > threshold).sum().item())
    basis = u[:, :rank].contiguous()
    condition = None if rank == 0 else float(s[0] / s[rank - 1])
    if rank:
        eye = torch.eye(rank, device=basis.device, dtype=basis.dtype)
        if not torch.allclose(basis.T @ basis, eye, atol=2e-5, rtol=2e-5):
            raise AssertionError("basis is not orthonormal")
    return BasisResult(basis, s.detach().cpu(), rank, rows.shape[0], condition)


# ---------------------------------------------------------------- 10.2
@dataclass(frozen=True)
class DynamicSelection:
    selected_ids: torch.Tensor
    protected_ids: torch.Tensor
    raw_scores: torch.Tensor
    basis_result: BasisResult


def select_output_protected_j_basis(
    activation: torch.Tensor,
    dictionary: torch.Tensor,
    clean_logits: torch.Tensor,
    *,
    k: int = 10,
    protect_top_k: int = 10,
    nonnegative: bool = True,
) -> DynamicSelection:
    """Select active J rows while protecting clean-output token labels —
    the safeguard the paper's dynamic ablation requires (addendum §0.1)."""
    h = activation.float().reshape(-1)
    d = torch.nn.functional.normalize(dictionary.float(), dim=1)
    scores = d @ h

    protected = clean_logits.float().topk(protect_top_k).indices
    masked = scores.clone()
    masked[protected] = -torch.inf
    if nonnegative:
        masked[scores <= 0.0] = -torch.inf

    finite = torch.isfinite(masked)
    take = min(k, int(finite.sum().item()))
    selected = masked.topk(take).indices if take else torch.empty(0, dtype=torch.long)
    basis = orthonormal_basis_from_rows(d[selected]) if take else \
        orthonormal_basis_from_rows(d.new_zeros((0, d.shape[1])))
    return DynamicSelection(
        selected_ids=selected.detach().cpu(),
        protected_ids=protected.detach().cpu(),
        raw_scores=scores[selected].detach().cpu() if take else scores.new_zeros(0).cpu(),
        basis_result=basis,
    )


# ---------------------------------------------------------------- 10.3
Phase = Literal["inactive", "prefill", "decode"]
_CURRENT_PHASE: ContextVar[Phase] = ContextVar("jspace_phase", default="inactive")


@contextmanager
def intervention_phase(phase: Phase):
    token = _CURRENT_PHASE.set(phase)
    try:
        yield
    finally:
        _CURRENT_PHASE.reset(token)


class PhaseControlledAblator:
    """Projector hooks that fire only in declared phases; fire counts are
    asserted per item so 'generation-only' is provable, not prose."""

    def __init__(self, layers, projectors: dict[int, torch.Tensor],
                 active_phases: set[str]):
        self.layers = layers
        self.projectors = projectors
        self.active_phases = active_phases
        self.handles = []
        self.fire_counts = {"prefill": 0, "decode": 0}

    def _hook(self, layer_index: int):
        def hook(module, inputs, output):
            phase = _CURRENT_PHASE.get()
            if phase not in self.active_phases:
                return output
            self.fire_counts[phase] += 1
            hidden = output if torch.is_tensor(output) else output[0]
            q = self.projectors[layer_index].to(hidden.device, torch.float32)
            h32 = hidden.float()
            edited = (h32 - (h32 @ q) @ q.T).to(hidden.dtype)
            return edited if torch.is_tensor(output) else (edited, *output[1:])
        return hook

    def __enter__(self):
        for idx in self.projectors:
            self.handles.append(
                self.layers[idx].register_forward_hook(self._hook(idx)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ---------------------------------------------------------------- 10.4
@dataclass
class RunningVectorMoments:
    n: int
    mean: torch.Tensor
    m2: torch.Tensor

    @classmethod
    def empty(cls, dimension: int, *, dtype=torch.float64):
        return cls(n=0, mean=torch.zeros(dimension, dtype=dtype),
                   m2=torch.zeros((dimension, dimension), dtype=dtype))

    def update(self, batch: torch.Tensor) -> None:
        x = batch.detach().to("cpu", dtype=self.mean.dtype)
        if x.ndim != 2 or x.shape[1] != self.mean.numel():
            raise ValueError("batch shape mismatch")
        m = x.shape[0]
        if m == 0:
            return
        batch_mean = x.mean(dim=0)
        centered = x - batch_mean
        batch_m2 = centered.T @ centered
        if self.n == 0:
            self.n = m
            self.mean.copy_(batch_mean)
            self.m2.copy_(batch_m2)
            return
        total = self.n + m
        delta = batch_mean - self.mean
        self.m2 += batch_m2 + torch.outer(delta, delta) * (self.n * m / total)
        self.mean += delta * (m / total)
        self.n = total

    def merge(self, other: "RunningVectorMoments") -> None:
        if other.n == 0:
            return
        if self.n == 0:
            self.n = other.n
            self.mean.copy_(other.mean)
            self.m2.copy_(other.m2)
            return
        total = self.n + other.n
        delta = other.mean - self.mean
        self.m2 += other.m2 + torch.outer(delta, delta) * (
            self.n * other.n / total)
        self.mean += delta * (other.n / total)
        self.n = total

    def covariance(self, *, unbiased: bool = True) -> torch.Tensor:
        denom = self.n - 1 if unbiased else self.n
        if denom <= 0:
            raise ValueError("not enough samples")
        return self.m2 / denom

    def state_dict(self) -> dict:
        return {"n": self.n, "mean": self.mean, "m2": self.m2}

    @classmethod
    def from_state_dict(cls, d: dict) -> "RunningVectorMoments":
        return cls(n=int(d["n"]), mean=d["mean"].clone(), m2=d["m2"].clone())


# ---------------------------------------------------------------- 10.5
@dataclass(frozen=True)
class SequenceScore:
    token_ids: list[int]
    token_logprobs: list[float]
    sum_logprob: float
    mean_logprob: float


@torch.no_grad()
def conditional_sequence_logprob(
    hf_model,
    prompt_ids: torch.Tensor,
    answer_ids: torch.Tensor,
) -> SequenceScore:
    """P(answer | prompt) over the FULL answer token sequence."""
    if prompt_ids.ndim != 2 or answer_ids.ndim != 2:
        raise ValueError("expected batched [1, seq] tensors")
    if prompt_ids.shape[0] != 1 or answer_ids.shape[0] != 1:
        raise ValueError("this helper expects batch size one")
    if answer_ids.shape[1] == 0:
        raise ValueError("answer must contain at least one token")
    full = torch.cat([prompt_ids, answer_ids], dim=1)
    logits = hf_model(input_ids=full, use_cache=False).logits
    start = prompt_ids.shape[1] - 1
    stop = full.shape[1] - 1
    log_probs = logits[:, start:stop, :].log_softmax(dim=-1)
    gathered = log_probs.gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1)
    values = gathered[0].float().cpu()
    return SequenceScore(
        token_ids=answer_ids[0].tolist(),
        token_logprobs=values.tolist(),
        sum_logprob=float(values.sum()),
        mean_logprob=float(values.mean()),
    )


# ---------------------------------------------------------------- 10.6
def paired_cluster_bootstrap(
    frame,
    *,
    cluster_column: str,
    item_column: str,
    condition_column: str,
    score_column: str,
    treatment: str,
    baseline: str,
    statistic: Callable[[np.ndarray], float] = np.mean,
    draws: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    pivot = frame.pivot_table(
        index=[cluster_column, item_column], columns=condition_column,
        values=score_column, aggfunc="first",
    ).dropna(subset=[treatment, baseline])
    pivot["delta"] = pivot[treatment] - pivot[baseline]
    clusters = pivot.index.get_level_values(cluster_column).unique().to_numpy()
    if len(clusters) < 2:
        raise ValueError("need at least two independent clusters")
    by_cluster = {c: pivot.xs(c, level=cluster_column)["delta"].to_numpy()
                  for c in clusters}
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        samples[draw] = statistic(np.concatenate([by_cluster[c] for c in chosen]))
    observed = statistic(pivot["delta"].to_numpy())
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"estimate": float(observed), "ci_low": float(low),
            "ci_high": float(high), "n_items": int(len(pivot)),
            "n_clusters": int(len(clusters))}


# ---------------------------------------------------------------- 10.7
def equivalence_from_interval(
    estimate: float, ci_low: float, ci_high: float, *, smallest_effect: float,
) -> bool:
    if smallest_effect <= 0:
        raise ValueError("smallest_effect must be positive")
    return ci_low > -smallest_effect and ci_high < smallest_effect


# ---------------------------------------------------------------- 10.8
def seeded_random_orthobasis(
    dimension: int, max_rank: int, *, seed: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """One basis per seed; dose k is ALWAYS basis[:, :k] (nested)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = torch.randn(max_rank, dimension, generator=generator)
    basis = orthonormal_basis_from_rows(rows).basis
    if basis.shape[1] < max_rank:
        raise RuntimeError("unexpected rank loss in random basis")
    return basis.to(device)


# ---------------------------------------------------------------- 10.9/10.10
@dataclass(frozen=True)
class FrozenSelectionArtifact:
    item_id: str
    prompt_sha256: str
    lens_sha256: str
    layer_to_selected_ids: dict[int, list[int]]
    layer_to_basis_sha256: dict[int, str]
    layer_to_effective_rank: dict[int, int]
    selection_rule: str
    selection_phase: str


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()
    ).hexdigest()
