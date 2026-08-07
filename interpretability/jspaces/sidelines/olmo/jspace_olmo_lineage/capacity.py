"""Prospective symmetric sparse-capacity estimator for the OLMo lineage.

The primary target is the activation after subtracting the one global mean
vector for that model/layer/frozen population.  This is intentionally stricter
than the historical Part 2 repair, which pursued raw activations and centered
only when naming the resulting reconstruction share.  Raw-target pursuit is
retained here under an explicitly separate sensitivity label.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch


@dataclass
class PursuitResult:
    """Sparse-pursuit sufficient statistics for one position batch."""

    errors: torch.Tensor
    selected_indices: torch.Tensor
    achieved_support: torch.Tensor


def lower_median(values: np.ndarray | Sequence[int]) -> int:
    """Integer lower median, matching the historical ``torch.median`` rule."""
    array = np.asarray(values, dtype=np.int64).reshape(-1)
    if not len(array):
        raise ValueError("lower median is undefined for an empty array")
    return int(np.partition(array, (len(array) - 1) // 2)[
        (len(array) - 1) // 2])


@torch.no_grad()
def gradient_pursuit(
    targets: torch.Tensor,
    dictionary: torch.Tensor,
    *,
    k_max: int,
    refit_iterations: int = 8,
    learning_rate_cap: float = 0.25,
) -> PursuitResult:
    """Nonnegative greedy pursuit with positive-support exhaustion.

    ``targets`` is fp32 ``[B,d]``. Dictionary rows must be normalized. A row
    permanently stops adding atoms when no unused atom has positive residual
    correlation. Errors at all K, including K=0, are returned so crossing,
    shares, prompt bootstraps, and raw sensitivity can be reconstructed without
    rerunning the solver.
    """
    if targets.ndim != 2 or dictionary.ndim != 2:
        raise ValueError("targets and dictionary must both be matrices")
    if targets.shape[1] != dictionary.shape[1]:
        raise ValueError("target and dictionary dimensions differ")
    if k_max < 1 or k_max > dictionary.shape[0]:
        raise ValueError("k_max must be between 1 and the dictionary row count")
    if targets.dtype != torch.float32:
        targets = targets.float()

    batch, _ = targets.shape
    device = targets.device
    indices = torch.full(
        (batch, k_max), -1, dtype=torch.long, device=device)
    coefficients = torch.zeros(
        batch, k_max, dtype=torch.float32, device=device)
    reconstruction = torch.zeros_like(targets)
    taken = torch.zeros(
        batch, dictionary.shape[0], dtype=torch.bool, device=device)
    errors = torch.empty(
        batch, k_max + 1, dtype=torch.float32, device=device)
    errors[:, 0] = targets.square().sum(dim=1)
    achieved = torch.zeros(batch, dtype=torch.long, device=device)
    exhausted = torch.zeros(batch, dtype=torch.bool, device=device)
    correlation_dictionary = dictionary

    for step_index in range(k_max):
        residual = targets - reconstruction
        correlation_input = residual.to(correlation_dictionary.dtype)
        correlations = (correlation_input @ correlation_dictionary.T).float()
        correlations.masked_fill_(taken, float("-inf"))
        best_value, best_index = correlations.max(dim=1)
        exhausted |= ~torch.isfinite(best_value) | (best_value <= 0)
        active = ~exhausted
        if active.any():
            active_rows = torch.arange(batch, device=device)[active]
            indices[active, step_index] = best_index[active]
            taken[active_rows, best_index[active]] = True
            achieved[active] += 1

        safe_indices = indices[:, :step_index + 1].clamp_min(0)
        active_slots = (
            torch.arange(step_index + 1, device=device)[None, :]
            < achieved[:, None]
        )
        selected_dictionary = dictionary[safe_indices].float()
        selected_dictionary *= active_slots.unsqueeze(-1)
        coefficients_at_step = coefficients[:, :step_index + 1]
        step_size = min(
            float(learning_rate_cap), 1.0 / float(step_index + 2))
        for _ in range(int(refit_iterations)):
            residual = targets - torch.einsum(
                "bk,bkd->bd", coefficients_at_step, selected_dictionary)
            gradient = torch.einsum(
                "bd,bkd->bk", residual, selected_dictionary)
            proposed = (
                coefficients_at_step + step_size * gradient
            ).clamp_min_(0) * active_slots
            # K indexes support growth. Once a row cannot add a positive atom,
            # do not smuggle extra coefficient-refit iterations into later K:
            # its reconstruction and error curve must remain exactly flat.
            coefficients_at_step = torch.where(
                active[:, None], proposed, coefficients_at_step)
        coefficients[:, :step_index + 1] = coefficients_at_step
        proposed_reconstruction = torch.einsum(
            "bk,bkd->bd", coefficients_at_step, selected_dictionary)
        reconstruction = torch.where(
            active[:, None], proposed_reconstruction, reconstruction)
        errors[:, step_index + 1] = (
            targets - reconstruction).square().sum(dim=1)

    return PursuitResult(errors, indices, achieved)


@torch.no_grad()
def pursuit_batched(
    targets: torch.Tensor,
    dictionary: torch.Tensor,
    *,
    k_max: int,
    batch_positions: int,
    refit_iterations: int = 8,
    learning_rate_cap: float = 0.25,
) -> PursuitResult:
    """Run row-independent pursuit in bounded GPU batches."""
    if targets.device.type != "cpu":
        raise ValueError("batched targets must be staged on CPU")
    n_positions = int(targets.shape[0])
    all_errors = torch.empty(n_positions, k_max + 1, dtype=torch.float32)
    all_indices = torch.empty(n_positions, k_max, dtype=torch.int64)
    all_achieved = torch.empty(n_positions, dtype=torch.int64)
    for start in range(0, n_positions, int(batch_positions)):
        stop = min(start + int(batch_positions), n_positions)
        batch = targets[start:stop].to(
            dictionary.device, dtype=torch.float32, non_blocking=False)
        result = gradient_pursuit(
            batch, dictionary, k_max=k_max,
            refit_iterations=refit_iterations,
            learning_rate_cap=learning_rate_cap,
        )
        all_errors[start:stop] = result.errors.cpu()
        all_indices[start:stop] = result.selected_indices.cpu()
        all_achieved[start:stop] = result.achieved_support.cpu()
        del batch, result
    return PursuitResult(all_errors, all_indices, all_achieved)


def marginal_gains(errors: np.ndarray) -> np.ndarray:
    values = np.asarray(errors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("error curves must have shape [N,K+1]")
    return values[:, :-1] - values[:, 1:]


def occupancy_from_errors(
    j_errors: np.ndarray,
    random_errors: np.ndarray,
    *,
    persistence: int,
) -> np.ndarray:
    """Return per-position crossing K against median random marginal gain."""
    j_gain = marginal_gains(j_errors)
    random_values = np.asarray(random_errors, dtype=np.float64)
    if random_values.ndim != 3:
        raise ValueError("random error curves must have shape [R,N,K+1]")
    random_gain = np.stack(
        [marginal_gains(value) for value in random_values], axis=0)
    random_median = np.median(random_gain, axis=0)
    below = j_gain <= random_median
    n_positions, k_max = below.shape
    occupancy = np.full(n_positions, k_max, dtype=np.int16)
    run = np.zeros(n_positions, dtype=np.int16)
    done = np.zeros(n_positions, dtype=bool)
    for k_index in range(k_max):
        run = np.where(below[:, k_index], run + 1, 0)
        hit = (~done) & (run >= int(persistence))
        occupancy[hit] = k_index + 1 - (int(persistence) - 1)
        done |= hit
    return np.maximum(occupancy, 1)


def _share_at_k(errors: np.ndarray, denominator: float, k: int) -> float:
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("target energy denominator must be finite and positive")
    return float(1.0 - np.asarray(errors, dtype=np.float64)[:, k].sum()
                 / denominator)


def curve_summary(
    j_errors: np.ndarray,
    random_errors: np.ndarray,
    *,
    persistence: int,
    persistence_sensitivity: Sequence[int] = (1, 2, 3),
) -> dict:
    """Point estimate reconstructed entirely from registered error curves."""
    j_values = np.asarray(j_errors, dtype=np.float64)
    random_values = np.asarray(random_errors, dtype=np.float64)
    if not np.isfinite(j_values).all() or not np.isfinite(random_values).all():
        raise ValueError("non-finite pursuit error curve")
    if random_values.shape[1:] != j_values.shape:
        raise ValueError("J and random error curve shapes differ")
    k_max = int(j_values.shape[1] - 1)
    occupancy = occupancy_from_errors(
        j_values, random_values, persistence=persistence)
    k_median = lower_median(occupancy)
    denominator = float(j_values[:, 0].sum())
    j_share = _share_at_k(j_values, denominator, k_median)
    random_seed_shares = [
        _share_at_k(seed_values, denominator, k_median)
        for seed_values in random_values
    ]
    random_share = float(np.mean(random_seed_shares))
    histogram = np.bincount(occupancy, minlength=k_max + 1)
    error_increases = np.diff(j_values, axis=1) > (
        1e-5 * np.maximum(j_values[:, :-1], 1.0))
    return {
        "n_positions": int(len(occupancy)),
        "k_max": k_max,
        "occupancy_median": int(k_median),
        "occupancy_q25": float(np.quantile(occupancy, 0.25)),
        "occupancy_q75": float(np.quantile(occupancy, 0.75)),
        "occupancy_histogram": histogram.astype(int).tolist(),
        "occupancy_censored_fraction": float(np.mean(occupancy >= k_max)),
        "occupancy_persistence_sensitivity": {
            str(value): lower_median(occupancy_from_errors(
                j_values, random_values, persistence=int(value)))
            for value in persistence_sensitivity
        },
        "j_share": j_share,
        "random_share": random_share,
        "random_seed_shares": random_seed_shares,
        "excess_share": float(j_share - random_share),
        "target_energy": denominator,
        "solver": {
            "j_error_increase_cells": int(error_increases.sum()),
            "j_error_increase_fraction": float(error_increases.mean()),
            "j_final_residual_fraction": float(
                j_values[:, -1].sum() / denominator),
        },
    }


def frame_summary(
    centered_j_errors: np.ndarray,
    centered_random_errors: np.ndarray,
    raw_j_errors: np.ndarray,
    raw_random_errors: np.ndarray,
    *,
    owners: np.ndarray,
    prompt_domains: Sequence[str],
    persistence: int,
    persistence_sensitivity: Sequence[int],
) -> dict:
    """Primary centered estimate, raw sensitivity, and corpus strata."""
    owners = np.asarray(owners, dtype=np.int64)
    if len(owners) != len(centered_j_errors):
        raise ValueError("owner vector and error curves are misaligned")
    primary = curve_summary(
        centered_j_errors, centered_random_errors,
        persistence=persistence,
        persistence_sensitivity=persistence_sensitivity,
    )
    raw = curve_summary(
        raw_j_errors, raw_random_errors,
        persistence=persistence,
        persistence_sensitivity=persistence_sensitivity,
    )
    strata = {}
    for domain in dict.fromkeys(prompt_domains):
        prompt_ids = np.asarray([
            index for index, value in enumerate(prompt_domains)
            if value == domain], dtype=np.int64)
        mask = np.isin(owners, prompt_ids)
        strata[domain] = {
            "primary_centered": curve_summary(
                np.asarray(centered_j_errors)[mask],
                np.asarray(centered_random_errors)[:, mask],
                persistence=persistence,
                persistence_sensitivity=persistence_sensitivity,
            ),
            "raw_sensitivity": curve_summary(
                np.asarray(raw_j_errors)[mask],
                np.asarray(raw_random_errors)[:, mask],
                persistence=persistence,
                persistence_sensitivity=persistence_sensitivity,
            ),
        }
    return {
        "primary_centered": primary,
        "raw_sensitivity": raw,
        "strata": strata,
    }


def stratified_prompt_counts(
    prompt_domains: Sequence[str], *, draws: int, seed: int,
) -> np.ndarray:
    """Frozen-domain-count prompt bootstrap multiplicities ``[B,P]``."""
    domains = list(prompt_domains)
    counts = np.zeros((int(draws), len(domains)), dtype=np.int16)
    rng = np.random.default_rng(int(seed))
    for domain in dict.fromkeys(domains):
        members = np.asarray([
            index for index, value in enumerate(domains)
            if value == domain], dtype=np.int64)
        samples = rng.choice(
            members, size=(int(draws), len(members)), replace=True)
        for draw_index in range(int(draws)):
            counts[draw_index] += np.bincount(
                samples[draw_index], minlength=len(domains)).astype(np.int16)
    return counts


def _prompt_sufficient(
    j_errors: np.ndarray,
    random_errors: np.ndarray,
    owners: np.ndarray,
    *,
    persistence: int,
    n_prompts: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    j_values = np.asarray(j_errors, dtype=np.float64)
    random_values = np.asarray(random_errors, dtype=np.float64)
    owners = np.asarray(owners, dtype=np.int64)
    occupancy = occupancy_from_errors(
        j_values, random_values, persistence=persistence)
    k_max = j_values.shape[1] - 1
    energy = np.zeros(n_prompts, dtype=np.float64)
    j_curve = np.zeros((n_prompts, k_max + 1), dtype=np.float64)
    random_curve = np.zeros(
        (random_values.shape[0], n_prompts, k_max + 1), dtype=np.float64)
    occupancy_histogram = np.zeros(
        (n_prompts, k_max + 1), dtype=np.int64)
    for prompt_index in range(n_prompts):
        mask = owners == prompt_index
        energy[prompt_index] = j_values[mask, 0].sum()
        j_curve[prompt_index] = j_values[mask].sum(axis=0)
        random_curve[:, prompt_index] = random_values[:, mask].sum(axis=1)
        occupancy_histogram[prompt_index] = np.bincount(
            occupancy[mask], minlength=k_max + 1)
    return energy, j_curve, random_curve, occupancy_histogram


def bootstrap_estimates(
    j_errors: np.ndarray,
    random_errors: np.ndarray,
    *,
    owners: np.ndarray,
    prompt_counts: np.ndarray,
    persistence: int,
) -> dict[str, np.ndarray]:
    """Recompute occupancy and shares for supplied prompt-cluster draws."""
    counts = np.asarray(prompt_counts, dtype=np.int64)
    energy, j_curve, random_curve, occ_hist = _prompt_sufficient(
        j_errors, random_errors, owners,
        persistence=persistence, n_prompts=counts.shape[1])
    total_energy = counts @ energy
    summed_j = counts @ j_curve
    summed_random = np.einsum(
        "bp,rpk->rbk", counts, random_curve, optimize=True)
    summed_occ = counts @ occ_hist
    cumulative = np.cumsum(summed_occ, axis=1)
    total_positions = summed_occ.sum(axis=1)
    lower_targets = (total_positions + 1) // 2
    k_median = np.argmax(
        cumulative >= lower_targets[:, None], axis=1).astype(np.int16)
    draw_index = np.arange(len(counts))
    j_share = 1.0 - summed_j[draw_index, k_median] / total_energy
    random_share_by_seed = np.stack([
        1.0 - summed_random[seed_index, draw_index, k_median]
        / total_energy
        for seed_index in range(summed_random.shape[0])
    ], axis=1)
    random_share = random_share_by_seed.mean(axis=1)
    return {
        "occupancy_median": k_median,
        "j_share": j_share.astype(np.float64),
        "random_share": random_share.astype(np.float64),
        "excess_share": (j_share - random_share).astype(np.float64),
    }


def percentile_interval(values: np.ndarray, level: float) -> dict:
    tail = (1.0 - float(level)) / 2.0
    low, high = np.quantile(
        np.asarray(values, dtype=np.float64), [tail, 1.0 - tail])
    return {
        "level": float(level),
        "low": float(low),
        "high": float(high),
        "method": "stratified prompt-cluster percentile bootstrap",
    }


def classify_shift(
    *,
    centered_difference: float,
    centered_interval_low: float,
    centered_interval_high: float,
    occupancy_difference: int,
    occupancy_interval_low: float,
    occupancy_interval_high: float,
    equivalence_margin: float,
    material_margin: float,
) -> str:
    """Frozen interval-aware O2 shift router."""
    point = abs(float(centered_difference))
    occ_point = abs(int(occupancy_difference))
    equivalence = float(equivalence_margin)
    material = float(material_margin)
    stable = (
        point < equivalence
        and occ_point == 0
        and centered_interval_low > -equivalence
        and centered_interval_high < equivalence
    )
    if stable:
        return "stable"
    supported_material = (
        centered_interval_low > material
        or centered_interval_high < -material
        or occupancy_interval_low > 1.0
        or occupancy_interval_high < -1.0
    )
    if supported_material:
        return "material_shift"
    crosses_material = (
        centered_interval_low < -material
        or centered_interval_high > material
        or occupancy_interval_low < -1.0
        or occupancy_interval_high > 1.0
    )
    material_point_without_support = point > material or occ_point > 1
    if crosses_material or material_point_without_support:
        return "unresolved"
    if point >= equivalence or occ_point == 1:
        return "small_shift"
    return "unresolved"


def content_token_manifest(sequences: Sequence[Sequence[int]]) -> str:
    """Hash exact ordered content-token sequences without BOS ambiguity."""
    payload = "".join(
        json.dumps(list(map(int, sequence)), separators=(",", ":")) + "\n"
        for sequence in sequences
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_frozen_corpus(
    rows: Sequence[Mapping], *, domains: Sequence[str], rows_per_domain: int,
) -> list[dict]:
    """First N rows per declared domain, preserving the declared block order."""
    selected = []
    for domain in domains:
        candidates = [dict(row) for row in rows if row.get("domain") == domain]
        if len(candidates) < int(rows_per_domain):
            raise ValueError(f"domain {domain!r} has only {len(candidates)} rows")
        selected.extend(candidates[:int(rows_per_domain)])
    return selected


def canonical_jsonl(rows: Sequence[Mapping]) -> str:
    return "".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False) + "\n"
        for row in rows
    )
