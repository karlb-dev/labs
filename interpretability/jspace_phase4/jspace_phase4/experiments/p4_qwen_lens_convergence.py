"""Task-led structural convergence for registered Qwen Jacobian lenses.

This successor to p4f09 keeps that artifact immutable, reuses its exact
uniform-token and Rademacher-probe draws, and adds identity-adjusted,
incremental-block, task-token, frequency, and layer-type diagnostics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from ..gpu import require_cuda_gpu
from ..manifests import (
    atomic_json,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths4 import (
    figures_dir,
    materialize_local_file,
    metrics_dir,
    resolve_uri,
)
from ..provenance4 import Provenance4, write_result4
from ..registry4 import RegistryError, create, resolve
from ..seeds import SEED_CONTRACT
from .p4_qwen_lens_structural_stability import (
    effective_gain_on_cuda,
    load_lens_checkpoint,
    quantile_summary,
    verified_model_tensor_sample,
)
from .p4_qwen_nested_lens_fit import model_reference, verify_package_versions


PUBLISHED_CLASSIFICATION = (
    "external published reference, partially specified recipe")
VIEW_NAMES = ("raw", "minus_identity", "minus_alpha_identity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def safe_cosine(left: torch.Tensor, right: torch.Tensor,
                *, epsilon: float = 1e-12) -> float:
    if left.shape != right.shape:
        raise ValueError("cosine inputs must have equal shape")
    left = left.float()
    right = right.float()
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if float(left_norm.item()) <= epsilon and \
            float(right_norm.item()) <= epsilon:
        return 1.0
    if float(left_norm.item()) <= epsilon or \
            float(right_norm.item()) <= epsilon:
        return 0.0
    return float(((left * right).sum() / (left_norm * right_norm)).item())


def safe_row_cosines(left: torch.Tensor, right: torch.Tensor,
                     *, epsilon: float = 1e-12) -> torch.Tensor:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("row-cosine inputs must be equal-shape matrices")
    left = left.float()
    right = right.float()
    left_norm = torch.linalg.vector_norm(left, dim=1)
    right_norm = torch.linalg.vector_norm(right, dim=1)
    denominator = left_norm * right_norm
    result = torch.zeros_like(denominator)
    regular = denominator > epsilon
    result[regular] = (
        (left[regular] * right[regular]).sum(dim=1)
        / denominator[regular]
    )
    both_zero = (left_norm <= epsilon) & (right_norm <= epsilon)
    result[both_zero] = 1.0
    return result.clamp(-1, 1)


def symmetric_relative_delta(left: torch.Tensor, right: torch.Tensor,
                             *, epsilon: float = 1e-12) -> float:
    numerator = 2 * torch.linalg.vector_norm(left.float() - right.float())
    denominator = (
        torch.linalg.vector_norm(left.float())
        + torch.linalg.vector_norm(right.float())
    )
    if float(denominator.item()) <= epsilon:
        return 0.0
    return float((numerator / denominator).item())


def row_norm_ratios(left: torch.Tensor, right: torch.Tensor,
                    *, epsilon: float = 1e-12) -> torch.Tensor:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("row-norm inputs must be equal-shape matrices")
    left_norm = torch.linalg.vector_norm(left.float(), dim=1)
    right_norm = torch.linalg.vector_norm(right.float(), dim=1)
    if bool((right_norm <= epsilon).any().item()):
        raise RuntimeError("zero reference row in norm-ratio calculation")
    return left_norm / right_norm


def split_identity_component(matrix: torch.Tensor) -> tuple[torch.Tensor,
                                                             float]:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("J must be square")
    matrix = matrix.float()
    alpha = torch.trace(matrix) / matrix.shape[0]
    identity = torch.eye(
        matrix.shape[0], device=matrix.device, dtype=torch.float32)
    return matrix - alpha * identity, float(alpha.item())


def identity_views(matrix: torch.Tensor) -> tuple[dict[str, torch.Tensor],
                                                   dict[str, float]]:
    matrix = matrix.float()
    residual, alpha = split_identity_component(matrix)
    identity = torch.eye(
        matrix.shape[0], device=matrix.device, dtype=torch.float32)
    matrix_norm = torch.linalg.vector_norm(matrix)
    identity_fraction = (
        abs(alpha) * math.sqrt(matrix.shape[0]) / float(matrix_norm.item())
        if float(matrix_norm.item()) > 0 else 0.0
    )
    return {
        "raw": matrix,
        "minus_identity": matrix - identity,
        "minus_alpha_identity": residual,
    }, {
        "identity_scale_alpha": alpha,
        "identity_fraction_frobenius": identity_fraction,
    }


def operator_pair_metrics(left: torch.Tensor, right: torch.Tensor, *,
                          probes: torch.Tensor,
                          quantiles: list[float]) -> dict[str, float]:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("operator inputs must be equal-shape matrices")
    left = left.float()
    right = right.float()
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    denominator = float(right_norm.item())
    if denominator <= 1e-12:
        norm_ratio = 1.0 if float(left_norm.item()) <= 1e-12 else math.inf
    else:
        norm_ratio = float((left_norm / right_norm).item())

    left_transport = probes @ left.T
    right_transport = probes @ right.T
    transport_cosines = safe_row_cosines(left_transport, right_transport)
    transport_difference = torch.linalg.vector_norm(
        left_transport - right_transport, dim=1)
    transport_scale = (
        torch.linalg.vector_norm(left_transport, dim=1)
        + torch.linalg.vector_norm(right_transport, dim=1)
    )
    transport_delta = torch.where(
        transport_scale > 1e-12,
        2 * transport_difference / transport_scale,
        torch.zeros_like(transport_scale),
    )
    result = {
        "matrix_cosine": safe_cosine(left, right),
        "frobenius_ratio_left_over_right": norm_ratio,
        "symmetric_relative_delta": symmetric_relative_delta(left, right),
    }
    for prefix, values in (
        ("probe_transport_cosine", transport_cosines),
        ("probe_transport_symmetric_delta", transport_delta),
    ):
        result.update({
            f"{prefix}_{key}": value
            for key, value in quantile_summary(values, quantiles).items()
        })
    return result


def centered_linear_cka_gram(left: torch.Tensor,
                             right: torch.Tensor) -> float:
    """Centered linear CKA using observation Gram matrices."""
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("CKA inputs must be equal-shape matrices")
    if left.shape[0] < 2:
        raise ValueError("CKA requires at least two observations")
    left = left.float() - left.float().mean(dim=0, keepdim=True)
    right = right.float() - right.float().mean(dim=0, keepdim=True)
    left_gram = left @ left.T
    right_gram = right @ right.T
    numerator = (left_gram * right_gram).sum()
    denominator = torch.sqrt(
        left_gram.square().sum() * right_gram.square().sum())
    if float(denominator.item()) <= 1e-12:
        raise RuntimeError("degenerate centered matrix in CKA")
    return float((numerator / denominator).clamp(0, 1).item())


def token_pair_metrics(left: torch.Tensor, right: torch.Tensor, *,
                       quantiles: list[float], cka_n: int) -> dict[str, float]:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("token inputs must be equal-shape matrices")
    if left.shape[0] < 2:
        raise ValueError("token stratum must contain at least two rows")
    cosine = safe_row_cosines(left, right)
    norm_ratio = row_norm_ratios(left, right)
    normalized_left = torch.nn.functional.normalize(left.float(), dim=1)
    normalized_right = torch.nn.functional.normalize(right.float(), dim=1)
    result = {}
    for prefix, values in (
        ("direction_cosine", cosine),
        ("norm_ratio_left_over_right", norm_ratio),
    ):
        result.update({
            f"{prefix}_{key}": value
            for key, value in quantile_summary(values, quantiles).items()
        })
    result["linear_cka"] = centered_linear_cka_gram(
        normalized_left[:cka_n], normalized_right[:cka_n])
    return result


def incremental_block_mean(prefix: torch.Tensor, extended: torch.Tensor,
                           *, prefix_n: int,
                           extended_n: int) -> torch.Tensor:
    if prefix.shape != extended.shape:
        raise ValueError("incremental inputs must have equal shape")
    if not 0 < prefix_n < extended_n:
        raise ValueError("incremental counts require 0 < prefix < extended")
    return (
        extended_n * extended.float() - prefix_n * prefix.float()
    ) / (extended_n - prefix_n)


def fixed_rademacher_probes(*, seed: int, n: int,
                            d_model: int) -> tuple[torch.Tensor, str]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    bits = torch.randint(
        0, 2, (n, d_model), generator=generator, dtype=torch.uint8)
    packed_hash = hashlib.sha256(bits.numpy().tobytes()).hexdigest()
    probes = bits.to(torch.float32).mul_(2).sub_(1).div_(math.sqrt(d_model))
    return probes, packed_hash


def load_fixed_sampling_contract(path: Path, *, expected_file_sha256: str,
                                 expected_ids_sha256: str,
                                 expected_probes_sha256: str,
                                 expected_token_n: int,
                                 expected_probe_shape: list[int]) -> tuple[
                                     list[int], torch.Tensor, dict]:
    if file_sha256(path) != expected_file_sha256:
        raise RuntimeError("historical sampling-manifest file hash mismatch")
    envelope = json.loads(path.read_text())
    if object_sha256(envelope["payload"]) != envelope["payload_sha256"]:
        raise RuntimeError("historical sampling-manifest envelope mismatch")
    token = envelope["payload"]["token_sample"]
    probes = envelope["payload"]["transport_probes"]
    ids = [int(value) for value in token["ids"]]
    if len(ids) != expected_token_n or len(ids) != len(set(ids)):
        raise RuntimeError("historical token sample size/uniqueness mismatch")
    if object_sha256(ids) != expected_ids_sha256 \
            or token["ids_sha256"] != expected_ids_sha256:
        raise RuntimeError("historical token-ID sample hash mismatch")
    shape = [int(value) for value in probes["shape"]]
    if shape != expected_probe_shape:
        raise RuntimeError("historical probe shape mismatch")
    probe_tensor, packed_hash = fixed_rademacher_probes(
        seed=int(probes["seed"]), n=shape[0], d_model=shape[1])
    if packed_hash != expected_probes_sha256 \
            or probes["packed_uint8_sha256"] != expected_probes_sha256:
        raise RuntimeError("historical Rademacher-probe hash mismatch")
    return ids, probe_tensor, {
        "source_manifest": str(path),
        "source_manifest_sha256": expected_file_sha256,
        "source_payload_sha256": envelope["payload_sha256"],
        "token_ids_n": len(ids),
        "token_ids_sha256": expected_ids_sha256,
        "token_seed": int(token["seed"]),
        "cka_prefix_n": int(token["cka_prefix_n"]),
        "probe_seed": int(probes["seed"]),
        "probe_shape": shape,
        "packed_uint8_sha256": expected_probes_sha256,
    }


def _tokenize_strings(tokenizer, values: Iterable[str]) -> set[int]:
    token_ids: set[int] = set()
    cleaned = {
        str(value).strip() for value in values
        if value is not None and str(value).strip()
    }
    for raw in sorted(cleaned):
        for text in (raw, " " + raw):
            encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
            token_ids.update(int(value) for value in encoded)
    return token_ids


def build_task_token_strata(tokenizer, bank_rows: list[dict]) -> tuple[
        dict[str, list[int]], dict]:
    """Construct disjoint answer/bridge/shared/special task strata."""
    answer_strings: list[str] = []
    bridge_strings: list[str] = []
    for row in sorted(bank_rows, key=lambda value: value["fact_id"]):
        answer_strings.extend(row.get("accepted_answers", []))
        answer_strings.extend(row.get("counterfactual_accepted", []))
        answer_strings.extend([
            row.get("answer", ""), row.get("counterfactual_answer", "")])
        bridge_strings.extend([
            row.get("bridge", ""), row.get("counterfactual_bridge", ""),
            row.get("distractor_bridge", ""),
        ])
    special = {int(value) for value in tokenizer.all_special_ids}
    answer_raw = _tokenize_strings(tokenizer, answer_strings) - special
    bridge_raw = _tokenize_strings(tokenizer, bridge_strings) - special
    shared = answer_raw & bridge_raw
    strata = {
        "task_answer_only": sorted(answer_raw - shared),
        "task_bridge_only": sorted(bridge_raw - shared),
        "task_answer_bridge_shared": sorted(shared),
        "special_control": sorted(special),
    }
    nonempty_sets = [set(values) for values in strata.values() if values]
    union_size = len(set().union(*nonempty_sets)) if nonempty_sets else 0
    if union_size != sum(len(values) for values in strata.values()):
        raise RuntimeError("task-token strata are not disjoint")
    if any(len(values) < 2 for values in strata.values()):
        raise RuntimeError("every task/control stratum needs at least two IDs")
    contract = {
        "construction": (
            "tokenize stripped and one-leading-space variants without "
            "special tokens; place answer/bridge overlap in a named shared "
            "stratum; report tokenizer special IDs separately"),
        "answer_string_set_sha256": object_sha256(sorted({
            str(value) for value in answer_strings if value is not None})),
        "bridge_string_set_sha256": object_sha256(sorted({
            str(value) for value in bridge_strings if value is not None})),
        "strata": {
            name: {"n": len(ids), "ids_sha256": object_sha256(ids)}
            for name, ids in strata.items()
        },
    }
    return strata, contract


def deterministic_frequency_deciles(
        counts: Mapping[int, int], *, excluded_ids: set[int],
        max_per_decile: int, namespace: str) -> tuple[dict[str, list[int]],
                                                      dict]:
    eligible = sorted(
        ((int(token_id), int(count)) for token_id, count in counts.items()
         if count > 0 and int(token_id) not in excluded_ids),
        key=lambda pair: (pair[1], pair[0]),
    )
    if len(eligible) < 20:
        raise RuntimeError("held-out corpus has too few observed token IDs")
    groups = np.array_split(np.asarray(eligible, dtype=np.int64), 10)
    strata: dict[str, list[int]] = {}
    metadata = {}
    for index, group in enumerate(groups, start=1):
        pairs = [(int(row[0]), int(row[1])) for row in group.tolist()]
        ordered = sorted(
            pairs,
            key=lambda pair: hashlib.sha256(
                f"{namespace}:{index}:{pair[0]}".encode()).digest(),
        )
        selected = ordered[:max_per_decile]
        name = f"frequency_d{index:02d}"
        ids = [token_id for token_id, _ in selected]
        strata[name] = ids
        metadata[name] = {
            "rank": index,
            "interpretation": "d01 rarest observed; d10 most frequent",
            "eligible_n": len(pairs),
            "selected_n": len(ids),
            "count_min": min(count for _, count in pairs),
            "count_max": max(count for _, count in pairs),
            "ids_sha256": object_sha256(ids),
        }
    sets = [set(ids) for ids in strata.values()]
    if len(set().union(*sets)) != sum(len(ids) for ids in strata.values()):
        raise RuntimeError("frequency deciles are not disjoint")
    return strata, metadata


def heldout_frequency_counts(tokenizer, path: Path, *,
                             expected_sha256: str) -> tuple[Counter, dict]:
    if file_sha256(path) != expected_sha256:
        raise RuntimeError("held-out frequency corpus hash mismatch")
    frame = pd.read_parquet(path, columns=["text"])
    counts: Counter = Counter()
    nonempty = 0
    total = 0
    for text in frame["text"].tolist():
        if not isinstance(text, str) or not text:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids:
            nonempty += 1
            counts.update(int(value) for value in ids)
            total += len(ids)
    return counts, {
        "path": str(path),
        "sha256": expected_sha256,
        "split": "validation",
        "rows": int(len(frame)),
        "nonempty_rows": nonempty,
        "token_occurrences": total,
        "observed_token_ids": len(counts),
    }


def validate_layer_contract(checkpoints: Mapping[str, Mapping], *,
                            expected_source_layers: int) -> list[int]:
    expected = list(range(expected_source_layers))
    for label, checkpoint in checkpoints.items():
        if list(checkpoint["source_layers"]) != expected \
                or sorted(checkpoint["J"]) != expected:
            raise RuntimeError(f"source-layer order/set mismatch for {label}")
    return expected


def validate_published_provenance(config: Mapping,
                                  comparisons: list[Mapping]) -> None:
    published_names = {
        name for name, specification in config["lenses"].items()
        if specification["kind"] == "external_published"
    }
    uses_published = any(
        comparison["left"] in published_names
        or comparison["right"] in published_names
        for comparison in comparisons
    )
    if not uses_published:
        return
    provenance = config.get("published_reference", {})
    if provenance.get("classification") != PUBLISHED_CLASSIFICATION:
        raise RuntimeError(
            "published comparison lacks required partial-recipe label")
    if not provenance.get("unknown_recipe_fields"):
        raise RuntimeError(
            "partially specified published recipe must list unknown fields")


def assert_no_merge_shift(columns: Iterable[str]) -> None:
    prohibited = [column for column in columns if "merge_shift" in column]
    if prohibited:
        raise RuntimeError(
            f"algebraically redundant merge-shift metric prohibited: {prohibited}")


def randomized_left_subspace(matrix: torch.Tensor, *, omega: torch.Tensor,
                             rank: int, power_iterations: int) -> tuple[
                                 torch.Tensor, dict[str, float]]:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("subspace matrix must be square")
    if omega.shape[0] != matrix.shape[1] or omega.shape[1] < rank:
        raise ValueError("randomized-SVD probe has incompatible shape")
    matrix = matrix.float()
    values = matrix @ omega
    for _ in range(power_iterations):
        basis = torch.linalg.qr(values, mode="reduced").Q
        values = matrix @ (matrix.T @ basis)
    basis = torch.linalg.qr(values, mode="reduced").Q
    small = basis.T @ matrix
    small_left, singular, _ = torch.linalg.svd(small, full_matrices=False)
    leading_basis = basis @ small_left[:, :rank]
    leading = float(singular[0].item())
    frobenius = float(torch.linalg.vector_norm(matrix).item())
    stable_rank = (frobenius / leading) ** 2 if leading > 0 else 0.0
    return leading_basis, {
        "estimated_top_singular_value": leading,
        "estimated_stable_rank": stable_rank,
    }


def principal_subspace_metrics(left: torch.Tensor,
                               right: torch.Tensor) -> dict[str, float]:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("subspace bases must have equal shape")
    singular = torch.linalg.svdvals(left.T @ right).clamp(0, 1)
    angles = torch.rad2deg(torch.arccos(singular))
    return {
        "principal_subspace_similarity_mean": float(singular.mean().item()),
        "principal_angle_median_degrees": float(angles.median().item()),
        "principal_angle_max_degrees": float(angles.max().item()),
    }


def aggregate_numeric(rows: list[dict], columns: list[str]) -> dict:
    result = {}
    for column in columns:
        values = np.asarray([float(row[column]) for row in rows], dtype=np.float64)
        result[column] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "q05": float(np.quantile(values, 0.05)),
            "median": float(np.median(values)),
            "q95": float(np.quantile(values, 0.95)),
            "max": float(values.max()),
        }
    return result


def _load_bank_rows(specifications: list[Mapping]) -> tuple[list[dict], dict]:
    rows = []
    contract = {}
    for specification in specifications:
        path = resolve_uri(specification["uri"])
        actual = file_sha256(path)
        if actual != specification["sha256"]:
            raise RuntimeError(f"task-bank hash mismatch: {path}")
        bank_rows = [
            json.loads(line) for line in path.read_text().splitlines()
            if line.strip()
        ]
        rows.extend(bank_rows)
        contract[str(path)] = {
            "sha256": actual,
            "n_rows": len(bank_rows),
            "banks": sorted({row["bank"] for row in bank_rows}),
        }
    fact_ids = [row["fact_id"] for row in rows]
    if len(fact_ids) != len(set(fact_ids)):
        raise RuntimeError("task banks contain duplicate fact IDs")
    return rows, contract


def _load_lenses(config: Mapping, recipe: Mapping) -> tuple[dict, dict]:
    checkpoints = {}
    contract = {}
    for name, specification in config["lenses"].items():
        expected_hash = specification["lens_sha256"]
        if len(expected_hash) != 64:
            raise RuntimeError(f"lens {name} lacks a full SHA-256")
        kind = specification["kind"]
        if kind == "registered":
            event = resolve(specification["evidence_id"])
            if not event["live"]:
                raise RuntimeError(f"registered lens {name} is not live")
            source = resolve_uri(specification["lens_uri"])
            registered = {
                output["sha256"] for output in event["outputs"]
                if Path(output["path"]) == source
            }
            if registered != {expected_hash}:
                raise RuntimeError(f"lens {name} is not the registered output")
            local_path = materialize_local_file(
                specification["lens_uri"], expected_sha256=expected_hash)
        elif kind == "external_published":
            local_path = Path(specification["lens_path"])
            if file_sha256(local_path) != expected_hash:
                raise RuntimeError("published reference lens hash mismatch")
        else:
            raise RuntimeError(f"unsupported lens kind {kind!r}")
        checkpoints[name] = load_lens_checkpoint(
            local_path, specification, recipe)
        contract[name] = {
            **dict(specification),
            "materialized_path": str(local_path),
        }
    validate_layer_contract(
        checkpoints,
        expected_source_layers=int(recipe["expected_source_layers"]),
    )
    return checkpoints, contract


def _make_omega(*, seed: int, d_model: int,
                columns: int) -> tuple[torch.Tensor, dict]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    omega = torch.randn((d_model, columns), generator=generator)
    contract = {
        "distribution": "standard normal",
        "seed": seed,
        "shape": [d_model, columns],
        "float32_sha256": hashlib.sha256(omega.numpy().tobytes()).hexdigest(),
    }
    return omega, contract


def plot_convergence(rows: list[dict], *, primary_id: str,
                     incremental_id: str, assay_layers: tuple[int, int],
                     landmark_layers: list[int], png_path: Path,
                     pdf_path: Path) -> None:
    frame = pd.DataFrame(rows)
    primary = frame[frame["comparison_id"] == primary_id].sort_values("layer")
    incremental = frame[
        frame["comparison_id"] == incremental_id].sort_values("layer")
    if len(primary) == 0 or len(incremental) == 0:
        raise RuntimeError("figure comparison rows are missing")
    assert_no_merge_shift(frame.columns)

    colors = {
        "raw": "#0072B2",
        "minus_identity": "#D55E00",
        "minus_alpha_identity": "#009E73",
        "answer": "#CC79A7",
        "bridge": "#E69F00",
        "shared": "#56B4E9",
        "uniform": "#555555",
    }
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), sharex=True)
    x = primary["layer"]

    axis = axes[0, 0]
    for view in VIEW_NAMES:
        axis.plot(x, primary[f"{view}_matrix_cosine"], lw=1.7,
                  color=colors[view], label=view.replace("_", " "))
    axis.set_ylabel("matrix cosine")
    axis.set_title("A · A120 vs A250 operator agreement", loc="left")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[0, 1]
    task_lines = [
        ("task_answer_only", "answer-only", colors["answer"], "-"),
        ("task_bridge_only", "bridge-only", colors["bridge"], "-"),
        ("task_answer_bridge_shared", "answer/bridge shared",
         colors["shared"], "-"),
        ("uniform4096", "uniform 4,096", colors["uniform"], "--"),
    ]
    for stratum, label, color, linestyle in task_lines:
        column = f"token_{stratum}_direction_cosine_q50"
        if column in primary:
            axis.plot(x, primary[column], lw=1.6, color=color,
                      linestyle=linestyle, label=label)
    axis.set_ylabel("token-row cosine (median)")
    axis.set_title("B · Task rows lead; uniform diagnostic beneath", loc="left")
    axis.legend(frameon=False, fontsize=7, ncol=2)

    axis = axes[1, 0]
    for stratum, label, color in (
        ("task_answer_only", "answer CKA", colors["answer"]),
        ("task_bridge_only", "bridge CKA", colors["bridge"]),
        ("uniform4096", "uniform CKA", colors["uniform"]),
    ):
        column = f"token_{stratum}_linear_cka"
        if column in primary:
            axis.plot(x, primary[column], lw=1.5, color=color, label=label)
    axis.plot(x, primary["principal_subspace_similarity_mean"],
              lw=1.7, color=colors["minus_alpha_identity"],
              label="residual subspace")
    axis.set_ylabel("similarity")
    axis.set_xlabel("source layer")
    axis.set_title("C · Dictionary CKA and residual subspace", loc="left")
    axis.legend(frameon=False, fontsize=7, ncol=2)

    axis = axes[1, 1]
    axis.plot(
        incremental["layer"],
        incremental["raw_symmetric_relative_delta"],
        lw=1.6, color=colors["raw"], label="raw operator delta")
    axis.plot(
        incremental["layer"],
        incremental["minus_alpha_identity_symmetric_relative_delta"],
        lw=1.6, color=colors["minus_alpha_identity"],
        label="residual operator delta")
    answer_column = "token_task_answer_only_direction_cosine_q50"
    if answer_column in incremental:
        axis.plot(
            incremental["layer"], 1 - incremental[answer_column],
            lw=1.4, color=colors["answer"],
            label="1 − answer-row cosine")
    axis.set_ylabel("disagreement")
    axis.set_xlabel("source layer")
    axis.set_title("D · A120 vs prompts 121–250 block mean", loc="left")
    axis.legend(frameon=False, fontsize=7)

    start, stop = assay_layers
    for axis in axes.ravel():
        axis.axvspan(start - 0.5, stop + 0.5, color="#F0E442", alpha=0.09)
        for layer in primary.loc[
                primary["layer_type"] == "full_attention", "layer"]:
            axis.axvspan(layer - 0.42, layer + 0.42,
                         color="#999999", alpha=0.045, linewidth=0)
        for layer in landmark_layers:
            axis.axvline(layer, color="#222222", alpha=0.22,
                         linestyle=":", linewidth=0.8)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.legend(
        handles=[
            Patch(facecolor="#F0E442", alpha=0.18,
                  label=f"assay band L{start}–L{stop}"),
            Patch(facecolor="#999999", alpha=0.18,
                  label="full-attention source layer"),
        ],
        loc="upper right", bbox_to_anchor=(0.985, 0.985),
        frameon=False, fontsize=8,
    )
    figure.suptitle(
        "Qwen3.6-27B draw-A lens convergence: n=120 to n=250",
        x=0.075, ha="left", fontsize=13,
    )
    figure.text(
        0.075, 0.012,
        "Development structural sensitivity. Published comparisons in the "
        "registered table use an external published reference, partially "
        "specified recipe.",
        fontsize=8, color="#555555",
    )
    figure.tight_layout(rect=(0.04, 0.05, 0.99, 0.94))
    figure.savefig(png_path, dpi=180)
    figure.savefig(pdf_path)
    plt.close(figure)


def main() -> None:  # noqa: C901, PLR0915
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    clean = require_clean_tree()
    try:
        existing = resolve(config["evidence_id"])
    except RegistryError as error:
        if "found 0" not in str(error):
            raise
    else:
        if not existing["live"]:
            raise RuntimeError("existing convergence evidence is not live")
        for output in existing["outputs"]:
            if file_sha256(output["path"]) != output["sha256"]:
                raise RuntimeError("existing convergence output hash mismatch")
        print(json.dumps({
            "status": "already-registered-and-verified",
            "evidence_id": config["evidence_id"],
        }, indent=1))
        return

    comparisons = [dict(value) for value in config["comparisons"]]
    validate_published_provenance(config, comparisons)
    gpu = require_cuda_gpu()
    packages = verify_package_versions(config["runtime_packages"])
    recipe = config["recipe"]
    model_path = resolve_uri(config["model_uri"])

    checkpoints, lens_contract = _load_lenses(config, recipe)
    layer_ids = validate_layer_contract(
        checkpoints,
        expected_source_layers=int(recipe["expected_source_layers"]),
    )

    sampling_spec = config["fixed_sampling"]
    sampling_path = resolve_uri(sampling_spec["source_manifest_uri"])
    uniform_ids, probes_cpu, sampling_contract = load_fixed_sampling_contract(
        sampling_path,
        expected_file_sha256=sampling_spec["source_manifest_sha256"],
        expected_ids_sha256=sampling_spec["token_ids_sha256"],
        expected_probes_sha256=sampling_spec["packed_probes_sha256"],
        expected_token_n=int(sampling_spec["token_ids_n"]),
        expected_probe_shape=[int(value) for value in sampling_spec["probe_shape"]],
    )

    task_rows, bank_contract = _load_bank_rows(config["task_banks"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False)
    task_strata, task_contract = build_task_token_strata(tokenizer, task_rows)

    frequency_path = resolve_uri(config["frequency_corpus"]["uri"])
    frequency_counts, frequency_source = heldout_frequency_counts(
        tokenizer, frequency_path,
        expected_sha256=config["frequency_corpus"]["sha256"],
    )
    task_union = set().union(*(set(values) for values in task_strata.values()))
    frequency_strata, frequency_contract = deterministic_frequency_deciles(
        frequency_counts,
        excluded_ids=task_union,
        max_per_decile=int(config["frequency_corpus"]["max_ids_per_decile"]),
        namespace=config["frequency_corpus"]["selection_namespace"],
    )
    strata = {
        **task_strata,
        **frequency_strata,
        "uniform4096": uniform_ids,
    }
    for name, ids in strata.items():
        if len(ids) < 2:
            raise RuntimeError(f"token stratum {name} has fewer than two IDs")
        if min(ids) < 0 or max(ids) >= int(recipe["expected_vocab_size"]):
            raise RuntimeError(f"token stratum {name} is outside vocabulary")

    all_token_ids = sorted(set().union(*(set(values) for values in strata.values())))
    token_index = {token_id: index for index, token_id in enumerate(all_token_ids)}
    stratum_indices = {
        name: torch.tensor(
            [token_index[token_id] for token_id in ids], dtype=torch.int64,
            device="cuda",
        )
        for name, ids in strata.items()
    }

    snapshot_manifest_path = resolve_uri(config["model_snapshot_manifest_uri"])
    snapshot_manifest = json.loads(snapshot_manifest_path.read_text())
    sampled_unembedding, norm_weight, tensor_contract = (
        verified_model_tensor_sample(
            model_path=model_path,
            manifest=snapshot_manifest,
            tensor_names=config["model_tensors"],
            token_ids=all_token_ids,
            expected_vocab_size=int(recipe["expected_vocab_size"]),
            expected_d_model=int(recipe["expected_d_model"]),
        )
    )
    gain, gain_contract = effective_gain_on_cuda(
        norm_weight,
        d_model=int(recipe["expected_d_model"]),
        eps=float(config["model_tensors"]["rms_norm_eps"]),
    )
    base_token_rows = (
        sampled_unembedding.to("cuda", dtype=torch.float32) * gain[None, :])
    probes = probes_cpu.to("cuda", dtype=torch.float32)
    subspace = config["subspace"]
    omega_cpu, omega_contract = _make_omega(
        seed=int(subspace["seed"]),
        d_model=int(recipe["expected_d_model"]),
        columns=int(subspace["rank"]) + int(subspace["oversample"]),
    )
    omega = omega_cpu.to("cuda", dtype=torch.float32)

    model_config = json.loads((model_path / "config.json").read_text())
    layer_types = model_config["text_config"]["layer_types"]
    if len(layer_types) < len(layer_ids):
        raise RuntimeError("model config has too few layer-type annotations")
    layer_types = [str(layer_types[layer]) for layer in layer_ids]

    incremental = config["incremental_block"]
    virtual_name = incremental["name"]
    if virtual_name in checkpoints:
        raise RuntimeError("incremental virtual lens collides with real lens")
    comparison_names = {
        value for comparison in comparisons
        for value in (comparison["left"], comparison["right"])
    }
    available_names = set(checkpoints) | {virtual_name}
    if not comparison_names <= available_names:
        raise RuntimeError(
            f"comparisons reference unknown lenses: {comparison_names - available_names}")

    quantiles = [float(value) for value in recipe["quantiles"]]
    cka_max_n = int(recipe["cka_max_n"])
    torch.cuda.reset_peak_memory_stats()
    rows: list[dict] = []
    for layer, layer_type in zip(layer_ids, layer_types, strict=True):
        matrices = {
            name: checkpoint["J"][layer].to("cuda", dtype=torch.float32)
            for name, checkpoint in checkpoints.items()
        }
        matrices[virtual_name] = incremental_block_mean(
            matrices[incremental["prefix"]],
            matrices[incremental["extended"]],
            prefix_n=int(incremental["prefix_n"]),
            extended_n=int(incremental["extended_n"]),
        )
        transformed = {
            name: base_token_rows @ matrix
            for name, matrix in matrices.items()
        }
        identity_metadata = {}
        subspace_bases = {}
        subspace_summaries = {}
        for name, matrix in matrices.items():
            views, identity_metadata[name] = identity_views(matrix)
            basis, summary = randomized_left_subspace(
                views["minus_alpha_identity"], omega=omega,
                rank=int(subspace["rank"]),
                power_iterations=int(subspace["power_iterations"]),
            )
            subspace_bases[name] = basis
            subspace_summaries[name] = summary
            del views

        for comparison in comparisons:
            left_name = comparison["left"]
            right_name = comparison["right"]
            left_views, _ = identity_views(matrices[left_name])
            right_views, _ = identity_views(matrices[right_name])
            row = {
                "comparison_id": comparison["id"],
                "comparison_scope": comparison["scope"],
                "left_lens": left_name,
                "right_lens": right_name,
                "layer": layer,
                "layer_type": layer_type,
                "left_identity_scale_alpha":
                    identity_metadata[left_name]["identity_scale_alpha"],
                "right_identity_scale_alpha":
                    identity_metadata[right_name]["identity_scale_alpha"],
                "left_identity_fraction_frobenius":
                    identity_metadata[left_name]["identity_fraction_frobenius"],
                "right_identity_fraction_frobenius":
                    identity_metadata[right_name]["identity_fraction_frobenius"],
                "left_residual_estimated_top_singular_value":
                    subspace_summaries[left_name]["estimated_top_singular_value"],
                "right_residual_estimated_top_singular_value":
                    subspace_summaries[right_name]["estimated_top_singular_value"],
                "left_residual_estimated_stable_rank":
                    subspace_summaries[left_name]["estimated_stable_rank"],
                "right_residual_estimated_stable_rank":
                    subspace_summaries[right_name]["estimated_stable_rank"],
                **principal_subspace_metrics(
                    subspace_bases[left_name], subspace_bases[right_name]),
            }
            if "published" in {left_name, right_name}:
                row["published_reference_classification"] = (
                    PUBLISHED_CLASSIFICATION)
            else:
                row["published_reference_classification"] = "not-applicable"
            for view in VIEW_NAMES:
                metrics = operator_pair_metrics(
                    left_views[view], right_views[view],
                    probes=probes, quantiles=quantiles)
                row.update({f"{view}_{key}": value
                            for key, value in metrics.items()})
            for stratum_name, indices in stratum_indices.items():
                left_token = transformed[left_name].index_select(0, indices)
                right_token = transformed[right_name].index_select(0, indices)
                metrics = token_pair_metrics(
                    left_token, right_token,
                    quantiles=quantiles,
                    cka_n=min(cka_max_n, len(strata[stratum_name])),
                )
                row.update({
                    f"token_{stratum_name}_{key}": value
                    for key, value in metrics.items()
                })
            rows.append(row)
            del left_views, right_views
        print(json.dumps({
            "layer": layer,
            "layer_type": layer_type,
            "primary_raw_cosine": round(
                rows[-len(comparisons)]["raw_matrix_cosine"], 6),
            "primary_residual_cosine": round(
                rows[-len(comparisons)][
                    "minus_alpha_identity_matrix_cosine"], 6),
        }), flush=True)
        del matrices, transformed, subspace_bases

    frame = pd.DataFrame(rows)
    assert_no_merge_shift(frame.columns)
    numeric_columns = [
        column for column in frame.columns
        if column not in {
            "comparison_id", "comparison_scope", "left_lens", "right_lens",
            "layer", "layer_type", "published_reference_classification",
        }
    ]
    assay_start, assay_stop = [int(value) for value in recipe["assay_band"]]
    aggregate = {}
    for comparison in comparisons:
        subset = [row for row in rows
                  if row["comparison_id"] == comparison["id"]]
        assay = [row for row in subset
                 if assay_start <= int(row["layer"]) <= assay_stop]
        aggregate[comparison["id"]] = {
            "all_layers": aggregate_numeric(subset, numeric_columns),
            f"assay_L{assay_start}_L{assay_stop}":
                aggregate_numeric(assay, numeric_columns),
            "by_layer_type": {
                layer_type: aggregate_numeric(
                    [row for row in subset if row["layer_type"] == layer_type],
                    numeric_columns,
                )
                for layer_type in sorted({row["layer_type"] for row in subset})
            },
        }

    output_dir = (
        metrics_dir(config["slug"]) / "lens_convergence"
        / config["evidence_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "layer_comparison_metrics.csv"
    result_path = output_dir / "convergence_result.json"
    manifest_path = output_dir / "input_manifest.json"
    token_manifest_path = output_dir / "token_strata_manifest.json"
    png_path = figures_dir() / "p4f10_qwen_lens_convergence.png"
    pdf_path = figures_dir() / "p4f10_qwen_lens_convergence.pdf"
    frame.to_csv(table_path, index=False)
    plot_convergence(
        rows,
        primary_id=config["figure"]["primary_comparison_id"],
        incremental_id=config["figure"]["incremental_comparison_id"],
        assay_layers=(assay_start, assay_stop),
        landmark_layers=[int(value) for value in recipe["landmark_layers"]],
        png_path=png_path,
        pdf_path=pdf_path,
    )

    token_payload = {
        "schema_version": 1,
        "construction_is_outcome_blind": True,
        "uniform_sample": sampling_contract,
        "task_bank_inputs": bank_contract,
        "task_contract": task_contract,
        "frequency_source": frequency_source,
        "frequency_contract": frequency_contract,
        "strata": {
            name: {
                "ids": ids,
                "n": len(ids),
                "ids_sha256": object_sha256(ids),
            }
            for name, ids in strata.items()
        },
        "combined_unique_ids_n": len(all_token_ids),
        "combined_unique_ids_sha256": object_sha256(all_token_ids),
    }
    token_envelope = {
        "schema_version": 1,
        "payload": token_payload,
        "payload_sha256": object_sha256(token_payload),
    }
    atomic_json(token_manifest_path, token_envelope)

    input_payload = {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "code_commit": clean["code_commit"],
        "config_sha256": file_sha256(config_path),
        "model": model_reference(config["model_uri"]),
        "model_snapshot_manifest_sha256": file_sha256(snapshot_manifest_path),
        "model_tensor_contract": tensor_contract,
        "runtime_packages": packages,
        "gpu": gpu,
        "lenses": lens_contract,
        "comparisons": comparisons,
        "incremental_block": dict(incremental),
        "published_reference": dict(config["published_reference"]),
        "fixed_sampling": sampling_contract,
        "token_strata_manifest_payload_sha256": token_envelope["payload_sha256"],
        "effective_gain": gain_contract,
        "subspace_probe": omega_contract,
        "recipe": dict(recipe),
        "layer_types": layer_types,
        "layer_types_sha256": object_sha256(layer_types),
        "producer_sha256": file_sha256(Path(__file__)),
    }
    manifest_envelope = {
        "schema_version": 1,
        "payload": input_payload,
        "payload_sha256": object_sha256(input_payload),
    }
    atomic_json(manifest_path, manifest_envelope)

    payload = {
        "schema_version": 1,
        "comparison_scope": (
            "same-corpus nested convergence, incremental-block sensitivity, "
            "and explicitly labeled external-reference transfer"),
        "n_layers": len(layer_ids),
        "n_comparisons": len(comparisons),
        "d_model": int(recipe["expected_d_model"]),
        "assay_band": [assay_start, assay_stop],
        "landmark_layers": [int(value) for value in recipe["landmark_layers"]],
        "token_stratum_sizes": {
            name: len(ids) for name, ids in strata.items()},
        "aggregate": aggregate,
        "table_rows_sha256": object_sha256(rows),
        "table_sha256": file_sha256(table_path),
        "token_strata_manifest_sha256": file_sha256(token_manifest_path),
        "figure_png_sha256": file_sha256(png_path),
        "figure_pdf_sha256": file_sha256(pdf_path),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu": gpu,
        "published_reference_classification": PUBLISHED_CLASSIFICATION,
        "limitations": [
            "The published lens is an external published reference, "
            "partially specified recipe; only A120 versus A250 is a pure "
            "same-corpus sample-size contrast here.",
            "Selection geometry and causal equivalence are reserved for the "
            "fixed multi-lens functional gate and cannot be inferred from "
            "this structural evidence.",
            "Randomized residual subspaces use a fixed hash-pinned probe; "
            "reported leading singular values and stable ranks are estimates.",
            "Frequency deciles use the pinned WikiText validation split and "
            "exclude task and special IDs; the frozen p4f09 uniform sample is "
            "reported separately without filtering.",
        ],
    }
    command = (
        "python -m jspace_phase4.experiments.p4_qwen_lens_convergence "
        f"--config {arguments.config}")
    inputs = {
        **{
            f"lens_{name}": specification["lens_sha256"]
            for name, specification in config["lenses"].items()
        },
        **{
            f"evidence_{name}": specification["evidence_id"]
            for name, specification in config["lenses"].items()
            if specification["kind"] == "registered"
        },
        "historical_sampling_manifest": sampling_spec[
            "source_manifest_sha256"],
        "fixed_token_ids": sampling_spec["token_ids_sha256"],
        "fixed_rademacher_probes": sampling_spec["packed_probes_sha256"],
        "frequency_corpus": config["frequency_corpus"]["sha256"],
        "task_banks": object_sha256(bank_contract),
        "model_snapshot_manifest": file_sha256(snapshot_manifest_path),
        "model_tensor_shards":
            tensor_contract["verified_shards_inventory_sha256"],
        "input_manifest": manifest_envelope["payload_sha256"],
    }
    write_result4(
        payload,
        result_path,
        Provenance4(
            evidence_id=config["evidence_id"],
            tier=config["tier"],
            command=command,
            inputs=inputs,
            input_manifest_sha256=manifest_envelope["payload_sha256"],
            model=model_reference(config["model_uri"]),
            seed_contract=SEED_CONTRACT,
        ),
    )
    create(
        config["evidence_id"],
        tier=config["tier"],
        what=(
            "Task-led Qwen draw-A n=120 versus n=250 structural convergence "
            "with raw/identity-adjusted, incremental-block, frequency, "
            "layer-type, and labeled external-reference diagnostics."),
        command=command,
        outputs=[
            result_path, manifest_path, token_manifest_path, table_path,
            png_path, pdf_path,
        ],
        inputs=inputs,
    )
    print(json.dumps({
        "evidence_id": config["evidence_id"],
        "result": str(result_path),
        "table": str(table_path),
        "figure_png": str(png_path),
        "figure_pdf": str(pdf_path),
        "token_strata": payload["token_stratum_sizes"],
        "gpu": gpu["name"],
    }, indent=1))


if __name__ == "__main__":
    main()
