"""Pure geometry and sparse-selection metrics for the OLMo side track.

The functions here intentionally have no registry or path side effects.  The
experiment producer supplies already-verified tensors and writes the durable
evidence.  Keeping the numerical core small makes the cross-checkpoint metric
definitions independently testable.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable, Sequence

import numpy as np
import torch


def _finite_matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    result = value.float()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} contains non-finite values")
    return result


def safe_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    """Cosine of equally shaped tensors, flattened in fp32."""
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    if left.shape != right.shape:
        raise ValueError("cosine inputs must have equal shape")
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(
        right)
    if float(denominator) == 0.0:
        return 1.0 if bool(torch.equal(left, right)) else 0.0
    return float(torch.dot(left, right).div(denominator).item())


def symmetric_relative_delta(left: torch.Tensor,
                             right: torch.Tensor) -> float:
    """Symmetric Frobenius distance ``2||A-B||/(||A||+||B||)``."""
    left = left.float()
    right = right.float()
    if left.shape != right.shape:
        raise ValueError("delta inputs must have equal shape")
    denominator = (
        torch.linalg.vector_norm(left) + torch.linalg.vector_norm(right))
    if float(denominator) == 0.0:
        return 0.0
    return float((2.0 * torch.linalg.vector_norm(left - right)
                  / denominator).item())


def operator_pair_metrics(left: torch.Tensor,
                          right: torch.Tensor) -> dict[str, float]:
    """Raw and identity-separated pair metrics for square operators."""
    left = _finite_matrix(left, "left operator")
    right = _finite_matrix(right, "right operator")
    if left.shape != right.shape or left.shape[0] != left.shape[1]:
        raise ValueError("operators must be equal square matrices")
    dimension = int(left.shape[0])
    identity = torch.eye(dimension, device=left.device, dtype=left.dtype)
    left_alpha = float(torch.trace(left).item() / dimension)
    right_alpha = float(torch.trace(right).item() / dimension)
    left_minus_identity = left - identity
    right_minus_identity = right - identity
    left_residual = left - left_alpha * identity
    right_residual = right - right_alpha * identity
    return {
        "raw_matrix_cosine": safe_cosine(left, right),
        "symmetric_relative_frobenius_delta": symmetric_relative_delta(
            left, right),
        "left_frobenius_norm": float(torch.linalg.vector_norm(left).item()),
        "right_frobenius_norm": float(torch.linalg.vector_norm(right).item()),
        "j_minus_identity_cosine": safe_cosine(
            left_minus_identity, right_minus_identity),
        "j_minus_identity_symmetric_delta": symmetric_relative_delta(
            left_minus_identity, right_minus_identity),
        "left_trace_projection_alpha": left_alpha,
        "right_trace_projection_alpha": right_alpha,
        "j_minus_alpha_identity_cosine": safe_cosine(
            left_residual, right_residual),
        "j_minus_alpha_identity_symmetric_delta": symmetric_relative_delta(
            left_residual, right_residual),
    }


def quantile_summary(values: np.ndarray | torch.Tensor,
                     quantiles: Sequence[float]) -> dict[str, float]:
    """Deterministic finite quantiles with stable JSON-friendly names."""
    array = (values.detach().float().cpu().numpy()
             if isinstance(values, torch.Tensor) else np.asarray(values))
    array = np.asarray(array, dtype=np.float64).reshape(-1)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("quantiles require a nonempty finite vector")
    result = {}
    for quantile in quantiles:
        if not 0.0 <= float(quantile) <= 1.0:
            raise ValueError("quantile outside [0,1]")
        key = f"q{int(round(100 * float(quantile))):02d}"
        result[key] = float(np.quantile(array, float(quantile)))
    result["mean"] = float(array.mean())
    result["minimum"] = float(array.min())
    result["maximum"] = float(array.max())
    result["n"] = int(len(array))
    return result


def row_cosines(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Per-row cosine, returning zero when exactly one row is zero."""
    left = _finite_matrix(left, "left rows")
    right = _finite_matrix(right, "right rows")
    if left.shape != right.shape:
        raise ValueError("row matrices must have equal shape")
    numerator = (left * right).sum(dim=1)
    denominator = left.norm(dim=1) * right.norm(dim=1)
    both_zero = (left.norm(dim=1) == 0) & (right.norm(dim=1) == 0)
    values = numerator / denominator.clamp_min(1e-12)
    return torch.where(both_zero, torch.ones_like(values), values)


def centered_linear_cka_gram(left: torch.Tensor,
                             right: torch.Tensor) -> float:
    """Centered linear CKA computed through row Gram matrices."""
    left = _finite_matrix(left, "left CKA rows")
    right = _finite_matrix(right, "right CKA rows")
    if left.shape[0] != right.shape[0] or left.shape[0] < 2:
        raise ValueError("CKA inputs need the same two-or-more observations")
    left = left - left.mean(dim=0, keepdim=True)
    right = right - right.mean(dim=0, keepdim=True)
    left_gram = left @ left.T
    right_gram = right @ right.T
    return safe_cosine(left_gram, right_gram)


def neighbor_overlap(left: torch.Tensor, right: torch.Tensor, *, k: int) -> dict:
    """Mean fraction and Jaccard overlap of within-sample top-k neighbors."""
    left = torch.nn.functional.normalize(
        _finite_matrix(left, "left neighbor rows"), dim=1)
    right = torch.nn.functional.normalize(
        _finite_matrix(right, "right neighbor rows"), dim=1)
    if left.shape != right.shape:
        raise ValueError("neighbor row matrices must have equal shape")
    rows = int(left.shape[0])
    if not 1 <= int(k) < rows:
        raise ValueError("neighbor k must be between 1 and n-1")
    left_similarity = left @ left.T
    right_similarity = right @ right.T
    diagonal = torch.arange(rows, device=left.device)
    left_similarity[diagonal, diagonal] = float("-inf")
    right_similarity[diagonal, diagonal] = float("-inf")
    left_ids = torch.topk(left_similarity, int(k), dim=1).indices
    right_ids = torch.topk(right_similarity, int(k), dim=1).indices
    # k is deliberately small; broadcasting avoids a Python loop over rows.
    intersection = (left_ids[:, :, None] == right_ids[:, None, :]).any(
        dim=2).sum(dim=1).float()
    fractions = intersection / float(k)
    jaccards = intersection / (2.0 * float(k) - intersection)
    return {
        "neighbor_k": int(k),
        "overlap_fraction_mean": float(fractions.mean().item()),
        "overlap_fraction_median": float(fractions.median().item()),
        "jaccard_mean": float(jaccards.mean().item()),
        "jaccard_median": float(jaccards.median().item()),
        "n_rows": rows,
    }


def random_transport_metrics(left: torch.Tensor, right: torch.Tensor,
                             probes: torch.Tensor,
                             quantiles: Sequence[float]) -> dict:
    """Agreement of row-vector probes transported through two operators."""
    left = _finite_matrix(left, "left transport")
    right = _finite_matrix(right, "right transport")
    probes = _finite_matrix(probes, "transport probes")
    if left.shape != right.shape or probes.shape[1] != left.shape[0]:
        raise ValueError("transport probe dimensions differ")
    left_output = probes @ left
    right_output = probes @ right
    cosines = row_cosines(left_output, right_output)
    relative = (left_output - right_output).norm(dim=1) / (
        0.5 * (left_output.norm(dim=1) + right_output.norm(dim=1))
    ).clamp_min(1e-12)
    return {
        "probe_cosine": quantile_summary(cosines, quantiles),
        "probe_symmetric_relative_error": quantile_summary(
            relative, quantiles),
    }


def randomized_spectrum(matrix: torch.Tensor, *, omega: torch.Tensor,
                        rank: int, power_iterations: int) -> dict:
    """Randomized leading spectrum and stable-rank diagnostic.

    The returned effective-rank field is explicitly a stable rank
    ``||J||_F^2/sigma_1^2``.  It is not mislabeled as the entropy rank of the
    full, uncomputed spectrum.
    """
    matrix = _finite_matrix(matrix, "spectrum matrix")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("spectrum matrix must be square")
    if omega.shape[0] != matrix.shape[1] or omega.shape[1] < int(rank):
        raise ValueError("randomized spectrum probe shape drift")
    values = matrix @ omega.float()
    for _ in range(int(power_iterations)):
        basis = torch.linalg.qr(values, mode="reduced").Q
        values = matrix @ (matrix.T @ basis)
    basis = torch.linalg.qr(values, mode="reduced").Q
    small = basis.T @ matrix
    singular = torch.linalg.svdvals(small)[:int(rank)]
    top = float(singular[0].item()) if len(singular) else 0.0
    frobenius = float(torch.linalg.vector_norm(matrix).item())
    stable_rank = (frobenius / top) ** 2 if top > 0 else 0.0
    captured_energy = float(
        singular.square().sum().item() / max(frobenius ** 2, 1e-30))
    return {
        "method": "randomized-leading-spectrum",
        "rank": int(rank),
        "power_iterations": int(power_iterations),
        "singular_values": [float(value) for value in singular.cpu()],
        "estimated_top_singular_value": top,
        "frobenius_norm": frobenius,
        "estimated_stable_rank": stable_rank,
        "leading_spectrum_energy_fraction": captured_energy,
    }


def finite_rbo(left: Sequence[int], right: Sequence[int], *, p: float) -> float:
    """Extrapolated finite rank-biased overlap from Webber et al.'s form."""
    if not 0.0 < float(p) < 1.0:
        raise ValueError("RBO persistence must be in (0,1)")
    left = list(left)
    right = list(right)
    depth = max(len(left), len(right))
    if depth == 0:
        return 1.0
    left_seen: set[int] = set()
    right_seen: set[int] = set()
    weighted = 0.0
    agreement = 0.0
    for index in range(depth):
        if index < len(left):
            left_seen.add(int(left[index]))
        if index < len(right):
            right_seen.add(int(right[index]))
        agreement = len(left_seen & right_seen) / float(index + 1)
        weighted += agreement * (float(p) ** index)
    return float((1.0 - p) * weighted + agreement * (p ** depth))


_ALIAS_SPACE = re.compile(r"\s+")


def normalize_token_alias(token: str) -> str:
    """Conservative token alias normalization for swap diagnostics."""
    # Replace tokenizer word-boundary glyphs before NFKD decomposes ``Ġ``
    # into an ordinary ``G`` plus a combining dot.
    value = str(token).replace("Ġ", " ").replace("▁", " ")
    value = unicodedata.normalize("NFKD", value).casefold()
    value = _ALIAS_SPACE.sub(" ", value).strip()
    return value


def id_selection_metrics(left: Sequence[Sequence[int]],
                         right: Sequence[Sequence[int]], *,
                         rbo_p: float,
                         token_labels: dict[int, str] | None = None) -> dict:
    """Aggregate exact-ID, ranked, and conservative alias swap metrics."""
    if len(left) != len(right):
        raise ValueError("selection populations must align")
    jaccards = []
    rbos = []
    alias_swaps = 0
    different_slots = 0
    exact_slots = 0
    total_slots = 0
    for left_row, right_row in zip(left, right):
        left_list = [int(value) for value in left_row]
        right_list = [int(value) for value in right_row]
        left_set, right_set = set(left_list), set(right_list)
        union = left_set | right_set
        jaccards.append(
            len(left_set & right_set) / len(union) if union else 1.0)
        rbos.append(finite_rbo(left_list, right_list, p=rbo_p))
        for left_id, right_id in zip(left_list, right_list):
            total_slots += 1
            if left_id == right_id:
                exact_slots += 1
                continue
            different_slots += 1
            if token_labels is not None and normalize_token_alias(
                    token_labels.get(left_id, "")) == normalize_token_alias(
                        token_labels.get(right_id, "")) and normalize_token_alias(
                            token_labels.get(left_id, "")):
                alias_swaps += 1
    return {
        "selected_id_jaccard": quantile_summary(
            np.asarray(jaccards), (0.05, 0.25, 0.5, 0.75, 0.95)),
        "rank_biased_overlap": quantile_summary(
            np.asarray(rbos), (0.05, 0.25, 0.5, 0.75, 0.95)),
        "aligned_prefix_slots": int(total_slots),
        "exact_aligned_slot_fraction": (
            float(exact_slots / total_slots) if total_slots else 1.0),
        "different_aligned_slots": int(different_slots),
        "normalized_alias_equivalent_swaps": int(alias_swaps),
        "normalized_alias_equivalent_swap_fraction": (
            float(alias_swaps / different_slots) if different_slots else 0.0),
        "n_positions": int(len(left)),
    }


def selection_prefixes(selected: np.ndarray, occupancy: np.ndarray,
                       achieved: np.ndarray) -> list[list[int]]:
    """Recover the exact registered scientific prefix at each position."""
    selected = np.asarray(selected)
    occupancy = np.asarray(occupancy).reshape(-1)
    achieved = np.asarray(achieved).reshape(-1)
    if selected.ndim != 2 or not (
            len(selected) == len(occupancy) == len(achieved)):
        raise ValueError("selection arrays are misaligned")
    rows = []
    for index in range(len(selected)):
        length = min(int(occupancy[index]), int(achieved[index]))
        values = [int(value) for value in selected[index, :length]
                  if int(value) >= 0]
        rows.append(values)
    return rows


def marginal_crossing_margins(j_errors: np.ndarray,
                              random_errors: np.ndarray,
                              occupancy: np.ndarray) -> np.ndarray:
    """J marginal gain minus random-median gain at each crossing K.

    This is a threshold margin reconstructed from O2 error curves.  It is not
    the pursuit correlation gap between the kth and k+1 candidate atoms.
    """
    from .capacity import marginal_gains

    j_gain = marginal_gains(j_errors)
    random = np.asarray(random_errors, dtype=np.float64)
    random_gain = np.stack(
        [marginal_gains(seed) for seed in random], axis=0)
    threshold = np.median(random_gain, axis=0)
    occupancy = np.asarray(occupancy, dtype=np.int64).reshape(-1)
    if len(occupancy) != len(j_gain):
        raise ValueError("occupancy and marginal gains are misaligned")
    columns = np.clip(occupancy - 1, 0, j_gain.shape[1] - 1)
    rows = np.arange(len(occupancy))
    return j_gain[rows, columns] - threshold[rows, columns]


def orthonormal_basis(rows: torch.Tensor, *, relative_tolerance: float
                      ) -> tuple[torch.Tensor, int]:
    """Numerical row-span basis using an explicit relative singular cutoff."""
    rows = _finite_matrix(rows, "span rows")
    if rows.shape[0] == 0:
        return torch.empty(
            rows.shape[1], 0, device=rows.device, dtype=rows.dtype), 0
    left, singular, _ = torch.linalg.svd(rows.T, full_matrices=False)
    threshold = max(
        float(singular[0].item()) * float(relative_tolerance), 1e-7)
    rank = int((singular > threshold).sum().item())
    return left[:, :rank], rank


def projector_pair_metrics(left_rows: torch.Tensor,
                           right_rows: torch.Tensor, *,
                           relative_tolerance: float) -> dict[str, float | int]:
    """Projector overlap and principal angles for two small selected spans."""
    if left_rows.shape[1:] != right_rows.shape[1:]:
        raise ValueError("selected span dimensions differ")
    left_basis, left_rank = orthonormal_basis(
        left_rows, relative_tolerance=relative_tolerance)
    right_basis, right_rank = orthonormal_basis(
        right_rows, relative_tolerance=relative_tolerance)
    if left_rank == 0 and right_rank == 0:
        overlap = 1.0
        angles = torch.zeros(1)
    elif left_rank == 0 or right_rank == 0:
        overlap = 0.0
        angles = torch.full((1,), 90.0)
    else:
        singular = torch.linalg.svdvals(
            left_basis.T @ right_basis).clamp(0.0, 1.0)
        overlap = float(
            singular.square().sum().item() / min(left_rank, right_rank))
        angles = torch.rad2deg(torch.arccos(singular)).cpu()
    return {
        "left_numerical_rank": left_rank,
        "right_numerical_rank": right_rank,
        "normalized_projector_overlap": overlap,
        "principal_angle_median_degrees": float(angles.median().item()),
        "principal_angle_max_degrees": float(angles.max().item()),
    }


def aggregate_projector_metrics(
    left_dictionary: torch.Tensor,
    right_dictionary: torch.Tensor,
    left_prefixes: Sequence[Sequence[int]],
    right_prefixes: Sequence[Sequence[int]],
    *,
    row_id_to_index: dict[int, int],
    relative_tolerance: float,
    batch_positions: int = 128,
) -> dict:
    """All-position selected-span comparison from extracted dictionary rows."""
    if len(left_prefixes) != len(right_prefixes):
        raise ValueError("projector selection populations differ")
    if batch_positions < 1:
        raise ValueError("projector batch size must be positive")
    # Grouping by the two small prefix lengths permits batched thin SVDs while
    # retaining every registered position.  The largest observed O2 prefix is
    # five rows, so this avoids ~135k tiny Python-dispatched decompositions.
    groups: dict[tuple[int, int], list[tuple[list[int], list[int]]]] = {}
    for left_ids, right_ids in zip(left_prefixes, right_prefixes):
        key = (len(left_ids), len(right_ids))
        groups.setdefault(key, []).append((
            [row_id_to_index[int(value)] for value in left_ids],
            [row_id_to_index[int(value)] for value in right_ids],
        ))
    collected = {key: [] for key in (
        "normalized_projector_overlap", "principal_angle_median_degrees",
        "principal_angle_max_degrees", "left_numerical_rank",
        "right_numerical_rank")}
    device = left_dictionary.device
    if right_dictionary.device != device:
        raise ValueError("projector dictionaries must share a device")
    for (left_count, right_count), items in groups.items():
        if left_count == 0 or right_count == 0:
            both_zero = left_count == 0 and right_count == 0
            n = len(items)
            collected["normalized_projector_overlap"].extend(
                [1.0 if both_zero else 0.0] * n)
            collected["principal_angle_median_degrees"].extend(
                [0.0 if both_zero else 90.0] * n)
            collected["principal_angle_max_degrees"].extend(
                [0.0 if both_zero else 90.0] * n)
            collected["left_numerical_rank"].extend([left_count] * n)
            collected["right_numerical_rank"].extend([right_count] * n)
            continue
        for start in range(0, len(items), int(batch_positions)):
            chunk = items[start:start + int(batch_positions)]
            left_indices = torch.tensor(
                [value[0] for value in chunk], device=device,
                dtype=torch.long)
            right_indices = torch.tensor(
                [value[1] for value in chunk], device=device,
                dtype=torch.long)
            left_rows = left_dictionary[left_indices].float()
            right_rows = right_dictionary[right_indices].float()
            left_u, left_s, _ = torch.linalg.svd(
                left_rows.transpose(1, 2), full_matrices=False)
            right_u, right_s, _ = torch.linalg.svd(
                right_rows.transpose(1, 2), full_matrices=False)
            left_threshold = torch.maximum(
                left_s[:, :1] * float(relative_tolerance),
                torch.full_like(left_s[:, :1], 1e-7))
            right_threshold = torch.maximum(
                right_s[:, :1] * float(relative_tolerance),
                torch.full_like(right_s[:, :1], 1e-7))
            left_rank = (left_s > left_threshold).sum(dim=1)
            right_rank = (right_s > right_threshold).sum(dim=1)
            cross = left_u.transpose(1, 2) @ right_u
            left_mask = (torch.arange(left_count, device=device)[None, :]
                         < left_rank[:, None])
            right_mask = (torch.arange(right_count, device=device)[None, :]
                          < right_rank[:, None])
            cross = cross * left_mask[:, :, None] * right_mask[:, None, :]
            singular = torch.linalg.svdvals(cross).clamp(0.0, 1.0)
            common_rank = torch.minimum(left_rank, right_rank).clamp_min(1)
            valid = (torch.arange(singular.shape[1], device=device)[None, :]
                     < common_rank[:, None])
            overlap = ((singular.square() * valid).sum(dim=1)
                       / common_rank.float())
            angles = torch.rad2deg(torch.arccos(singular))
            angles = torch.where(valid, angles, torch.nan)
            angle_values = angles.cpu().numpy()
            collected["normalized_projector_overlap"].extend(
                overlap.cpu().tolist())
            collected["principal_angle_median_degrees"].extend(
                np.nanmedian(angle_values, axis=1).tolist())
            collected["principal_angle_max_degrees"].extend(
                np.nanmax(angle_values, axis=1).tolist())
            collected["left_numerical_rank"].extend(
                left_rank.cpu().tolist())
            collected["right_numerical_rank"].extend(
                right_rank.cpu().tolist())
    keys = (
        "normalized_projector_overlap", "principal_angle_median_degrees",
        "principal_angle_max_degrees", "left_numerical_rank",
        "right_numerical_rank",
    )
    result = {key: quantile_summary(
        np.asarray(collected[key]),
        (0.05, 0.25, 0.5, 0.75, 0.95)) for key in keys}
    result["n_positions"] = len(left_prefixes)
    return result


def persistent_direction_summary(prefixes: Sequence[Sequence[int]], *,
                                 minimum_fraction: float) -> dict:
    """Token IDs selected in at least a fixed fraction of positions."""
    if not 0.0 < float(minimum_fraction) <= 1.0:
        raise ValueError("minimum persistence fraction outside (0,1]")
    counts: dict[int, int] = {}
    for row in prefixes:
        for token_id in set(map(int, row)):
            counts[token_id] = counts.get(token_id, 0) + 1
    threshold = int(math.ceil(len(prefixes) * float(minimum_fraction)))
    persistent = sorted(
        token_id for token_id, count in counts.items() if count >= threshold)
    return {
        "minimum_fraction": float(minimum_fraction),
        "minimum_positions": threshold,
        "n_positions": len(prefixes),
        "persistent_token_ids": persistent,
        "n_persistent": len(persistent),
        "maximum_position_fraction": (
            max(counts.values()) / len(prefixes) if prefixes else 0.0),
    }


def set_jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    left_set, right_set = set(map(int, left)), set(map(int, right))
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0
