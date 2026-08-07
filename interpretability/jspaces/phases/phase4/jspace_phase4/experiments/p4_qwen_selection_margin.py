"""Behavior-blind A500--A1000 selection-margin mechanism audit.

The functional producer captures rankings during the exact inherited
span-safe pass.  This successor reads only those rankings and geometric
inputs.  It never filters a functional position and never reads a language-
model behavioral endpoint.  Expensive dictionary geometry is checkpointed
one layer at a time so a VM reclaim loses at most one layer.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import re
import time
import unicodedata
from typing import Iterable, Mapping

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jspace_part2.dictionaries import effective_gain

from ..gpu import assert_model_on_cuda, require_cuda_gpu
from ..manifests import (
    atomic_json,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths4 import figures_dir, metrics_dir, resolve_uri
from ..provenance4 import Provenance4, write_result4
from ..registry4 import RegistryError, create, resolve
from .p4_qwen_multilens_functional_gate import (
    _partial_dictionary_rows,
    _resolve_lens_paths,
    selected_span_basis,
    selection_pair_metrics,
)
from .p4_qwen_nested_lens_fit import model_reference
from .p4_qwen_lens_structural_stability import load_lens_checkpoint


PAIR = ("a500", "a1000")
FORBIDDEN_CAPTURE_COLUMNS = {
    "lp_baseline", "lp_span_safe", "lp_exact_control", "specific",
    "delta_span_safe", "delta_exact_control", "bridge_rescue",
    "preference", "pick_swap", "greedy_category",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _json_list(value) -> list:
    if isinstance(value, str):
        parsed = json.loads(value)
    elif isinstance(value, (list, tuple, np.ndarray)):
        parsed = list(value)
    else:
        raise ValueError(f"expected a JSON list, got {type(value).__name__}")
    if not isinstance(parsed, list):
        raise ValueError("captured value is not a JSON list")
    return parsed


def _json_dict(value) -> dict:
    parsed = json.loads(value) if isinstance(value, str) else dict(value)
    if not isinstance(parsed, dict):
        raise ValueError("captured value is not a JSON object")
    return parsed


def relative_margin(scores: Iterable[float], k: int,
                    epsilon: float) -> float | None:
    values = [float(value) for value in scores]
    if k < 1 or len(values) <= k:
        return None
    return ((values[k - 1] - values[k])
            / max(abs(values[k - 1]), float(epsilon)))


def margin_stratum(left: Mapping, right: Mapping, *, k: int,
                   threshold: float) -> str:
    if int(left["effective_rank"]) < k or int(right["effective_rank"]) < k:
        return "rank_deficient"
    left_margin = float(left["margins"][str(k)])
    right_margin = float(right["margins"][str(k)])
    if left_margin >= threshold and right_margin >= threshold:
        return "stable_core"
    return "near_tie"


def core_ids(ids: Iterable[int], scores: Iterable[float], *, k: int,
             threshold: float, epsilon: float) -> list[int]:
    identifiers = [int(value) for value in ids]
    values = [float(value) for value in scores]
    selected = identifiers[:k]
    if len(values) <= k:
        return selected
    boundary = values[k]
    return [
        token_id for token_id, score in zip(selected, values[:k])
        if (score - boundary) / max(abs(score), epsilon) >= threshold
    ]


def jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    a, b = set(left), set(right)
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _basis_coordinates_from_scores(
        rows: torch.Tensor, scores: torch.Tensor, *,
        threshold: torch.Tensor | None = None) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return an orthobasis and h coordinates from row dot-products."""
    if rows.shape[0] == 0:
        return (
            rows.new_zeros((rows.shape[1], 0), dtype=torch.float32),
            rows.new_zeros((0,), dtype=torch.float32),
            rows.new_tensor(1e-7, dtype=torch.float32),
        )
    matrix = rows.float()
    u, singular, vh = torch.linalg.svd(
        matrix.T, full_matrices=False)
    if threshold is None:
        threshold = (singular[:1] * 1e-4).clamp_min(1e-7)
    keep = singular > threshold
    basis = u[:, keep]
    coordinates = ((vh[keep] @ scores.float())
                   / singular[keep].clamp_min(1e-30))
    return basis, coordinates, threshold


def selected_projection_energy(
        selected_rows: torch.Tensor, selected_scores: torch.Tensor,
        protected_rows: torch.Tensor, protected_scores: torch.Tensor,
        *, rank_threshold: torch.Tensor | None = None) -> tuple[float, int,
                                                               torch.Tensor]:
    """Reconstruct span-safe projection energy without retaining hidden h."""
    protected_basis, protected_coordinates, _ = (
        _basis_coordinates_from_scores(protected_rows, protected_scores))
    selected = selected_rows.float()
    scores = selected_scores.float()
    if protected_basis.shape[1]:
        loadings = selected @ protected_basis
        selected = selected - loadings @ protected_basis.T
        scores = scores - loadings @ protected_coordinates
    basis, coordinates, threshold = _basis_coordinates_from_scores(
        selected, scores, threshold=rank_threshold)
    return float(coordinates.square().sum().item()), int(basis.shape[1]), threshold


def disputed_dose_fraction(
        selected_ids: list[int], selected_scores: list[float],
        common_ids: set[int], selected_rows: torch.Tensor,
        protected_rows: torch.Tensor, protected_scores: torch.Tensor) -> float | None:
    scores = torch.tensor(
        selected_scores[:len(selected_ids)], device=selected_rows.device,
        dtype=torch.float32)
    full, _, threshold = selected_projection_energy(
        selected_rows, scores, protected_rows, protected_scores)
    if full <= 1e-20:
        return None
    indices = [index for index, token_id in enumerate(selected_ids)
               if token_id in common_ids]
    if not indices:
        common = 0.0
    else:
        index = torch.tensor(
            indices, device=selected_rows.device, dtype=torch.long)
        common, _, _ = selected_projection_energy(
            selected_rows.index_select(0, index),
            scores.index_select(0, index), protected_rows,
            protected_scores, rank_threshold=threshold)
    return float(np.clip((full - common) / full, 0.0, 1.0))


def _basis_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape[1] == 0 or right.shape[1] == 0:
        return 0.0
    singular = torch.linalg.svdvals(left.T @ right).clamp(0, 1)
    return float((singular.square().sum()
                  / max(min(left.shape[1], right.shape[1]), 1)).item())


def replacement_projector_overlap(
        selected_rows: torch.Tensor, protected_rows: torch.Tensor, *,
        replace_index: int, replacement_row: torch.Tensor) -> float:
    baseline, _ = selected_span_basis(selected_rows, protected_rows)
    changed = selected_rows.clone()
    changed[replace_index] = replacement_row
    replacement, _ = selected_span_basis(changed, protected_rows)
    return _basis_overlap(baseline, replacement)


def _surface(tokenizer, token_id: int) -> str:
    value = tokenizer.convert_ids_to_tokens(int(token_id))
    return str(value)


def _normalize_surface(value: str) -> str:
    value = value.replace("Ġ", " ").replace("▁", " ")
    value = unicodedata.normalize("NFKD", value).lower()
    return re.sub(r"[^a-z0-9]+", "", value)


def lexical_category(left: str, right: str) -> str:
    if left == right:
        return "identical_surface"
    normalized_left = _normalize_surface(left)
    normalized_right = _normalize_surface(right)
    if normalized_left and normalized_left == normalized_right:
        return "normalized_surface_alias"
    if normalized_left and normalized_right and (
            normalized_left in normalized_right
            or normalized_right in normalized_left
            or (min(len(normalized_left), len(normalized_right)) >= 4
                and os.path.commonprefix(
                    [normalized_left, normalized_right]).__len__() >= 4)):
        return "morphological_or_piece_variant"
    return "manual_review_required"


def _event_output(event: Mapping, name: str) -> Path:
    matches = [row for row in event["outputs"]
               if Path(row["path"]).name == name]
    if len(matches) != 1:
        raise RuntimeError(f"source functional event lacks exactly one {name}")
    row = matches[0]
    path = Path(row["path"])
    if file_sha256(path) != row["sha256"]:
        raise RuntimeError(f"source functional output hash mismatch: {name}")
    return path


def _load_sources(config: Mapping) -> tuple[dict, dict[str, Path], pd.DataFrame,
                                             pd.DataFrame, dict]:
    event = resolve(config["source_functional_evidence_id"])
    if not event["live"]:
        raise RuntimeError("source functional event is not live")
    capture_path = _event_output(event, config["source_capture_artifact"])
    selection_path = _event_output(event, config["source_selection_artifact"])
    pair_path = _event_output(event, config["source_pair_artifact"])
    manifest_path = _event_output(event, "input_manifest.json")
    result_path = _event_output(event, "functional_gate_result.json")
    functional_config_path = resolve_uri(config["source_functional_config_uri"])
    functional_config = yaml.safe_load(functional_config_path.read_text())
    if functional_config["evidence_id"] != config[
            "source_functional_evidence_id"]:
        raise RuntimeError("selection-margin functional config ID mismatch")
    if tuple(functional_config["analysis"]["primary_pair"]) != PAIR:
        raise RuntimeError("selection-margin source pair drift")
    manifest = json.loads(manifest_path.read_text())["payload"]
    if file_sha256(functional_config_path) != manifest["config_sha256"]:
        raise RuntimeError("selection-margin functional config hash mismatch")
    capture = pd.read_parquet(capture_path)
    pairs = pd.read_parquet(pair_path)
    _ = pd.read_parquet(selection_path)  # hash/parse conformance input
    result = json.loads(result_path.read_text())["payload"]
    paths = {
        "capture": capture_path, "selection": selection_path,
        "pairs": pair_path, "manifest": manifest_path,
        "functional_result": result_path,
        "functional_config": functional_config_path,
    }
    return functional_config, paths, capture, pairs, result


def _validate_capture(capture: pd.DataFrame, config: Mapping) -> pd.DataFrame:
    forbidden = FORBIDDEN_CAPTURE_COLUMNS & set(capture.columns)
    if forbidden:
        raise RuntimeError(
            f"behavioral columns leaked into margin capture: {sorted(forbidden)}")
    if set(capture["lens"]) != set(PAIR):
        raise RuntimeError("margin capture must contain exactly A500/A1000")
    required = {
        "item_id", "fact_id", "variant", "bank", "canonical_family",
        "layer", "position", "eligible_top_ids", "eligible_top_scores",
        "protected_ids", "protected_scores", "margins",
        "intervention_selected_ids", "intervention_selected_scores",
        "effective_rank", "removed_energy_frac",
    }
    missing = required - set(capture.columns)
    if missing:
        raise RuntimeError(f"margin capture lacks columns: {sorted(missing)}")
    frame = capture.copy()
    for column in (
            "eligible_top_ids", "eligible_top_scores", "protected_ids",
            "protected_scores", "intervention_selected_ids",
            "intervention_selected_scores"):
        frame[column] = frame[column].map(_json_list)
    frame["margins"] = frame["margins"].map(_json_dict)
    k = int(config["contract"]["intervention_k"])
    for row in frame.itertuples(index=False):
        selected_ids = [int(value)
                        for value in row.intervention_selected_ids]
        selected_scores = [float(value)
                           for value in row.intervention_selected_scores]
        eligible_ids = [int(value) for value in row.eligible_top_ids]
        eligible_scores = [float(value)
                           for value in row.eligible_top_scores]
        expected_n = min(k, int(row.eligible_available_positive))
        if len(selected_ids) != expected_n \
                or len(selected_scores) != expected_n:
            raise RuntimeError("captured intervention top-k length drift")
        if len(set(selected_ids)) != len(selected_ids):
            raise RuntimeError("captured intervention IDs are not unique")
        if set(selected_ids) & set(int(value)
                                   for value in row.protected_ids):
            raise RuntimeError("captured intervention includes protected ID")
        if any(not math.isfinite(value) or value <= 0
               for value in selected_scores):
            raise RuntimeError("captured intervention score is not positive")
        # Scores, rather than tied-ID ordering, define top-k admissibility.
        # torch.topk(k) and torch.topk(top_n) may return different IDs at an
        # exact boundary tie even though both selections are valid.
        if selected_scores != eligible_scores[:expected_n]:
            raise RuntimeError(
                "captured intervention scores differ from eligible top-k")
        eligible_by_id = dict(zip(eligible_ids, eligible_scores))
        for token_id, score in zip(selected_ids, selected_scores):
            if token_id in eligible_by_id \
                    and score != eligible_by_id[token_id]:
                raise RuntimeError(
                    "captured intervention ID/score alignment drift")
        if expected_n and len(eligible_scores) > expected_n:
            boundary_tied = (
                eligible_scores[expected_n - 1]
                == eligible_scores[expected_n])
            if not boundary_tied and set(selected_ids) != set(
                    eligible_ids[:expected_n]):
                raise RuntimeError(
                    "captured intervention IDs differ without boundary tie")
        for cutoff in config["contract"]["margin_ks"]:
            expected = relative_margin(
                row.eligible_top_scores, int(cutoff),
                float(config["contract"]["relative_margin_epsilon"]))
            actual = row.margins[str(cutoff)]
            if expected is None and actual is None:
                continue
            if expected is None or actual is None or not math.isclose(
                    float(actual), expected, rel_tol=1e-6, abs_tol=1e-9):
                raise RuntimeError("captured selection margin formula drift")
    key = ["lens", "item_id", "layer", "position"]
    if frame.duplicated(key).any():
        raise RuntimeError("duplicate selection-margin capture key")
    return frame


def _direction_rows(table: torch.Tensor, offsets: Mapping[int, int],
                    token_ids: Iterable[int]) -> torch.Tensor:
    ids = [int(value) for value in token_ids]
    if not ids:
        return table.new_zeros((0, table.shape[1]))
    index = torch.tensor(
        [offsets[value] for value in ids], device=table.device,
        dtype=torch.long)
    return table.index_select(0, index)


def _summarize_values(values: Iterable[float | None]) -> dict:
    finite = np.asarray([
        float(value) for value in values
        if value is not None and np.isfinite(value)], dtype=np.float64)
    if not len(finite):
        return {"n": 0, "median": None, "q05": None, "q95": None}
    return {
        "n": int(len(finite)), "median": float(np.median(finite)),
        "q05": float(np.quantile(finite, 0.05)),
        "q95": float(np.quantile(finite, 0.95)),
    }


def _layer_geometry(
        *, layer: int, capture: pd.DataFrame, pair_geometry: pd.DataFrame,
        dictionaries: Mapping[str, torch.Tensor],
        offsets: Mapping[int, int], tokenizer, config: Mapping) -> tuple[
            pd.DataFrame, pd.DataFrame]:
    by_lens = {}
    for lens in PAIR:
        subset = capture[(capture["lens"] == lens)
                         & (capture["layer"].astype(int) == layer)]
        by_lens[lens] = {
            (row.item_id, int(row.position)): row._asdict()
            for row in subset.itertuples(index=False)
        }
    keys = sorted(set(by_lens[PAIR[0]]) & set(by_lens[PAIR[1]]))
    expected_pairs = pair_geometry[
        (pair_geometry["comparison"] == "a500_vs_a1000")
        & (pair_geometry["layer"].astype(int) == layer)]
    pair_map = {
        (row.item_id, int(row.position)): row._asdict()
        for row in expected_pairs.itertuples(index=False)
    }
    if set(keys) != set(pair_map):
        raise RuntimeError(f"selection pair/capture key mismatch at L{layer}")
    epsilon = float(config["contract"]["relative_margin_epsilon"])
    k = int(config["contract"]["intervention_k"])
    stable_threshold = float(
        config["contract"]["stable_core_margin_at_intervention_k"])
    threshold_curve = [float(value) for value in
                       config["contract"]["threshold_curve"]]
    pair_rows, dispute_rows = [], []

    for item_id, position in keys:
        left = by_lens[PAIR[0]][(item_id, position)]
        right = by_lens[PAIR[1]][(item_id, position)]
        for field in ("fact_id", "variant", "bank", "canonical_family"):
            if left[field] != right[field]:
                raise RuntimeError("paired margin metadata mismatch")
        selected = {
            lens: [int(value) for value in row["intervention_selected_ids"]]
            for lens, row in zip(PAIR, (left, right))
        }
        scores = {
            lens: [float(value) for value in
                   row["intervention_selected_scores"]]
            for lens, row in zip(PAIR, (left, right))
        }
        protected = {
            lens: [int(value) for value in row["protected_ids"]]
            for lens, row in zip(PAIR, (left, right))
        }
        protected_scores = {
            lens: torch.tensor(
                [float(value) for value in row["protected_scores"]],
                device=dictionaries[lens].device, dtype=torch.float32)
            for lens, row in zip(PAIR, (left, right))
        }
        selected_rows = {
            lens: _direction_rows(
                dictionaries[lens], offsets, selected[lens])
            for lens in PAIR
        }
        protected_rows = {
            lens: _direction_rows(
                dictionaries[lens], offsets, protected[lens])
            for lens in PAIR
        }
        bases = {
            lens: selected_span_basis(
                selected_rows[lens], protected_rows[lens])[0]
            for lens in PAIR
        }
        geometry = selection_pair_metrics(
            selected[PAIR[0]], selected[PAIR[1]],
            scores[PAIR[0]], scores[PAIR[1]],
            bases[PAIR[0]], bases[PAIR[1]])
        registered_geometry = pair_map[(item_id, position)]
        for field in (
                "selected_id_jaccard", "normalized_projector_overlap"):
            if not math.isclose(
                    float(geometry[field]), float(registered_geometry[field]),
                    rel_tol=1e-5, abs_tol=1e-5):
                raise RuntimeError(
                    f"selection geometry reconstruction mismatch: {field}")
        common = set(selected[PAIR[0]]) & set(selected[PAIR[1]])
        left_only = [value for value in selected[PAIR[0]]
                     if value not in common]
        right_only = [value for value in selected[PAIR[1]]
                      if value not in common]
        dose = {}
        for lens, row in zip(PAIR, (left, right)):
            dose[lens] = disputed_dose_fraction(
                selected[lens], scores[lens], common,
                selected_rows[lens], protected_rows[lens],
                protected_scores[lens])
        replacement_left, replacement_right = [], []
        cross_cosines, within_left_cosines, within_right_cosines = [], [], []
        lexical_counts = {}
        for ordinal in range(max(len(left_only), len(right_only))):
            left_id = left_only[ordinal] if ordinal < len(left_only) else None
            right_id = (
                right_only[ordinal] if ordinal < len(right_only) else None)
            left_surface = (
                _surface(tokenizer, left_id) if left_id is not None else "")
            right_surface = (
                _surface(tokenizer, right_id) if right_id is not None else "")
            category = (lexical_category(left_surface, right_surface)
                        if left_id is not None and right_id is not None
                        else "manual_review_required")
            lexical_counts[category] = lexical_counts.get(category, 0) + 1
            record = {
                "item_id": item_id, "fact_id": left["fact_id"],
                "variant": left["variant"], "bank": left["bank"],
                "canonical_family": left["canonical_family"],
                "layer": int(layer), "position": int(position),
                "pair_ordinal": ordinal,
                "a500_token_id": left_id, "a1000_token_id": right_id,
                "a500_token_surface": left_surface,
                "a1000_token_surface": right_surface,
                "automatic_lexical_category": category,
                "manual_semantic_label": "",
                "manual_review_blinded_to_behavior": True,
                "cross_lens_row_cosine": None,
                "a500_operator_row_cosine": None,
                "a1000_operator_row_cosine": None,
                "a500_replacement_projector_overlap": None,
                "a1000_replacement_projector_overlap": None,
            }
            if left_id is not None and right_id is not None:
                li = offsets[left_id]
                ri = offsets[right_id]
                left_left = dictionaries[PAIR[0]][li].float()
                left_right = dictionaries[PAIR[0]][ri].float()
                right_left = dictionaries[PAIR[1]][li].float()
                right_right = dictionaries[PAIR[1]][ri].float()
                within_left = float(torch.dot(left_left, left_right).item())
                within_right = float(torch.dot(right_left, right_right).item())
                cross = float(torch.dot(left_left, right_right).item())
                left_index = selected[PAIR[0]].index(left_id)
                right_index = selected[PAIR[1]].index(right_id)
                left_overlap = replacement_projector_overlap(
                    selected_rows[PAIR[0]], protected_rows[PAIR[0]],
                    replace_index=left_index, replacement_row=left_right)
                right_overlap = replacement_projector_overlap(
                    selected_rows[PAIR[1]], protected_rows[PAIR[1]],
                    replace_index=right_index, replacement_row=right_left)
                record.update({
                    "cross_lens_row_cosine": cross,
                    "a500_operator_row_cosine": within_left,
                    "a1000_operator_row_cosine": within_right,
                    "a500_replacement_projector_overlap": left_overlap,
                    "a1000_replacement_projector_overlap": right_overlap,
                })
                cross_cosines.append(cross)
                within_left_cosines.append(within_left)
                within_right_cosines.append(within_right)
                replacement_left.append(left_overlap)
                replacement_right.append(right_overlap)
            dispute_rows.append(record)

        margins = {
            lens: float(row["margins"][str(k)])
            for lens, row in zip(PAIR, (left, right))
        }
        stratum = margin_stratum(
            left, right, k=k, threshold=stable_threshold)
        curve = {
            f"core_jaccard_at_{threshold:g}": jaccard(
                core_ids(
                    left["eligible_top_ids"], left["eligible_top_scores"],
                    k=k, threshold=threshold, epsilon=epsilon),
                core_ids(
                    right["eligible_top_ids"], right[
                        "eligible_top_scores"],
                    k=k, threshold=threshold, epsilon=epsilon))
            for threshold in threshold_curve
        }
        pair_rows.append({
            "item_id": item_id, "fact_id": left["fact_id"],
            "variant": left["variant"], "bank": left["bank"],
            "canonical_family": left["canonical_family"],
            "layer": int(layer), "position": int(position),
            "stratum": stratum,
            "a500_margin_at_k": margins[PAIR[0]],
            "a1000_margin_at_k": margins[PAIR[1]],
            "a500_effective_rank": int(left["effective_rank"]),
            "a1000_effective_rank": int(right["effective_rank"]),
            "selected_id_jaccard": geometry["selected_id_jaccard"],
            "normalized_projector_overlap": geometry[
                "normalized_projector_overlap"],
            "principal_angle_median_degrees": geometry[
                "principal_angle_median_degrees"],
            "n_common_selected_ids": len(common),
            "n_a500_only_ids": len(left_only),
            "n_a1000_only_ids": len(right_only),
            "a500_disputed_projection_energy_fraction": dose[PAIR[0]],
            "a1000_disputed_projection_energy_fraction": dose[PAIR[1]],
            "cross_lens_disputed_row_cosine_median": (
                float(np.median(cross_cosines)) if cross_cosines else None),
            "within_a500_disputed_row_cosine_median": (
                float(np.median(within_left_cosines))
                if within_left_cosines else None),
            "within_a1000_disputed_row_cosine_median": (
                float(np.median(within_right_cosines))
                if within_right_cosines else None),
            "a500_replacement_projector_overlap_median": (
                float(np.median(replacement_left))
                if replacement_left else None),
            "a1000_replacement_projector_overlap_median": (
                float(np.median(replacement_right))
                if replacement_right else None),
            "lexical_category_counts_json": json.dumps(
                lexical_counts, sort_keys=True),
            **curve,
        })
    return pd.DataFrame(pair_rows), pd.DataFrame(dispute_rows)


def summarize(pair_rows: pd.DataFrame, dispute_rows: pd.DataFrame,
              config: Mapping) -> dict:
    by_stratum = {}
    for stratum in config["contract"]["strata"]:
        subset = pair_rows[pair_rows["stratum"] == stratum]
        by_stratum[stratum] = {
            "n_positions": int(len(subset)),
            "n_items": int(subset["item_id"].nunique()),
            "n_families": int(subset["canonical_family"].nunique()),
            "selected_id_jaccard": _summarize_values(
                subset["selected_id_jaccard"]),
            "normalized_projector_overlap": _summarize_values(
                subset["normalized_projector_overlap"]),
            "a500_disputed_projection_energy_fraction": _summarize_values(
                subset["a500_disputed_projection_energy_fraction"]),
            "a1000_disputed_projection_energy_fraction": _summarize_values(
                subset["a1000_disputed_projection_energy_fraction"]),
        }
    lexical = dispute_rows[
        "automatic_lexical_category"].value_counts().to_dict()
    threshold_curve = {}
    for threshold in config["contract"]["threshold_curve"]:
        column = f"core_jaccard_at_{float(threshold):g}"
        threshold_curve[str(threshold)] = _summarize_values(pair_rows[column])
    return {
        "schema_version": 1,
        "tier": config["tier"],
        "primary_comparison": config["primary_comparison"],
        "outcome_blinding": (
            "No language-model behavioral endpoint is present in the "
            "capture, stratum assignment, row geometry, or lexical sheet."),
        "n_positions": int(len(pair_rows)),
        "n_items": int(pair_rows["item_id"].nunique()),
        "n_families": int(pair_rows["canonical_family"].nunique()),
        "stratum_counts": {
            key: int(value) for key, value in
            pair_rows["stratum"].value_counts().to_dict().items()},
        "by_stratum": by_stratum,
        "core_jaccard_threshold_curve": threshold_curve,
        "lexical_audit": {
            "automatic_category_counts": {
                key: int(value) for key, value in lexical.items()},
            "manual_review_required_rows": int(
                (dispute_rows["automatic_lexical_category"]
                 == "manual_review_required").sum()),
            "manual_review_is_behavior_blind_and_nondecisional": True,
        },
        "contract_verdict": {
            "capture_formula_recomputed_exactly": True,
            "captured_top_k_matches_intervention": True,
            "registered_selection_geometry_reconstructed": True,
            "all_positions_retained": True,
            "all_strata_retained_in_functional_gate": True,
            "behavioral_columns_used": False,
            "audit_complete_for_canonical_branch_router": True,
        },
    }


def make_figure(result: Mapping, pair_rows: pd.DataFrame, config: Mapping,
                png_path: Path, pdf_path: Path) -> None:
    strata = list(config["contract"]["strata"])
    counts = [result["stratum_counts"].get(value, 0) for value in strata]
    jaccard_values = [
        result["by_stratum"][value]["selected_id_jaccard"]["median"]
        for value in strata]
    overlap_values = [
        result["by_stratum"][value]["normalized_projector_overlap"][
            "median"] for value in strata]
    x = np.arange(len(strata))
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axes[0].bar(x, counts, color=["#315a9a", "#b6782c", "#8b4a68"])
    axes[0].set(
        xticks=x, xticklabels=[value.replace("_", "\n") for value in strata],
        ylabel="Frozen positions", title="Prospective margin strata")
    width = 0.36
    jaccard_plot = [np.nan if value is None else value
                    for value in jaccard_values]
    overlap_plot = [np.nan if value is None else value
                    for value in overlap_values]
    axes[1].bar(x - width / 2, jaccard_plot, width,
                label="selected-ID Jaccard", color="#b6782c")
    axes[1].bar(x + width / 2, overlap_plot, width,
                label="projector overlap", color="#315a9a")
    axes[1].set(
        xticks=x, xticklabels=[value.replace("_", "\n") for value in strata],
        ylabel="Median stability", ylim=(0, 1.02),
        title="Rows versus protected selected spans")
    axes[1].legend(frameon=False, fontsize=9)
    figure.suptitle(config["figure"]["title"], fontsize=12)
    figure.text(0.5, 0.01, config["figure"]["footer"],
                ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)


@torch.no_grad()
def main() -> None:  # noqa: C901
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    if config.get("tier") != "phase4-development":
        raise RuntimeError("selection-margin audit is development only")
    if config["contract"].get("include_all_strata_in_primary_gate") is not True:
        raise RuntimeError("selection-margin config attempts row exclusion")
    if config["contract"].get("no_behavioral_columns_in_stratification") is not True:
        raise RuntimeError("selection-margin config permits behavioral strata")
    clean = require_clean_tree()
    try:
        existing = resolve(config["evidence_id"])
    except RegistryError as error:
        if "found 0" not in str(error):
            raise
    else:
        if not existing["live"]:
            raise RuntimeError("existing selection-margin event is not live")
        for output in existing["outputs"]:
            if file_sha256(output["path"]) != output["sha256"]:
                raise RuntimeError("registered selection-margin output drift")
        print(json.dumps({
            "status": "already-registered-and-verified",
            "evidence_id": config["evidence_id"],
        }, indent=1))
        return

    functional, source_paths, capture_raw, pair_geometry, functional_result = (
        _load_sources(config))
    capture = _validate_capture(capture_raw, config)
    if functional_result.get("branch") != "PENDING_SELECTION_MARGIN_AUDIT":
        raise RuntimeError("functional source is not pending the margin audit")
    layers = sorted(int(value) for value in capture["layer"].unique())
    expected_band = [int(value) for value in functional["protocol"]["band"]]
    if layers != expected_band:
        raise RuntimeError("selection-margin capture layer-band drift")

    output_dir = (
        metrics_dir(config["slug"]) / "selection_margin"
        / config["evidence_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "analysis_state.json"
    source_hashes = {key: file_sha256(path)
                     for key, path in source_paths.items()}
    header = {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "code_commit": clean["code_commit"],
        "config_sha256": file_sha256(config_path),
        "producer_sha256": file_sha256(Path(__file__)),
        "source_hashes": source_hashes,
        "capture_rows_sha256": object_sha256([
            int(len(capture)), sorted(layers), sorted(set(capture["lens"]))]),
        "layers": layers,
    }
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("header") != header:
            raise RuntimeError("selection-margin resume header mismatch")
    else:
        state = {"header": header, "completed_layers": []}
        atomic_json(state_path, state)

    gpu = require_cuda_gpu()
    import transformers
    model_path = resolve_uri(functional["model_uri"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path))
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_path), dtype=torch.bfloat16).to("cuda").eval()
    assert_model_on_cuda(hf)
    lens_paths, _ = _resolve_lens_paths(functional)
    checkpoints = {
        lens: load_lens_checkpoint(
            lens_paths[lens], functional["lenses"][lens],
            functional["runtime"])
        for lens in PAIR
    }
    gain = effective_gain(hf).to("cuda", torch.float32)

    for layer in layers:
        pair_part = parts_dir / f"pair_L{layer:02d}.parquet"
        dispute_part = parts_dir / f"dispute_L{layer:02d}.parquet"
        if layer in state["completed_layers"]:
            if not pair_part.exists() or not dispute_part.exists():
                raise RuntimeError("completed margin layer lacks part files")
            log(f"margin geometry L{layer} already complete")
            continue
        layer_capture = capture[capture["layer"].astype(int) == layer]
        token_ids = set()
        for row in layer_capture.itertuples(index=False):
            token_ids.update(int(value) for value in row.eligible_top_ids)
            token_ids.update(int(value) for value in row.protected_ids)
        ordered = sorted(token_ids)
        offsets = {token_id: index for index, token_id in enumerate(ordered)}
        dictionaries = {
            lens: _partial_dictionary_rows(
                hf, gain, checkpoints[lens]["J"][layer], ordered)
            for lens in PAIR
        }
        pair_rows, dispute_rows = _layer_geometry(
            layer=layer, capture=capture, pair_geometry=pair_geometry,
            dictionaries=dictionaries, offsets=offsets,
            tokenizer=tokenizer, config=config)
        atomic_parquet(pair_part, pair_rows)
        atomic_parquet(dispute_part, dispute_rows)
        state["completed_layers"].append(layer)
        state["completed_layers"].sort()
        atomic_json(state_path, state)
        del dictionaries
        gc.collect()
        torch.cuda.empty_cache()
        log(f"margin geometry L{layer} complete")

    pair_parts = [parts_dir / f"pair_L{layer:02d}.parquet" for layer in layers]
    dispute_parts = [
        parts_dir / f"dispute_L{layer:02d}.parquet" for layer in layers]
    pair_rows = pd.concat(
        [pd.read_parquet(path) for path in pair_parts], ignore_index=True)
    dispute_rows = pd.concat(
        [pd.read_parquet(path) for path in dispute_parts], ignore_index=True)
    if len(pair_rows) * 2 != len(capture):
        raise RuntimeError("selection-margin analysis excluded positions")
    result = summarize(pair_rows, dispute_rows, config)
    result["functional_branch_candidate"] = functional_result[
        "branch_candidate"]

    pair_path = output_dir / "selection_margin_pair_rows.parquet"
    dispute_path = output_dir / "selection_margin_disputed_rows.parquet"
    review_path = output_dir / "manual_lexical_review_blinded.csv"
    result_path = output_dir / "selection_margin_result.json"
    manifest_path = output_dir / "input_manifest.json"
    png_path = figures_dir() / f"{config['figure']['stem']}.png"
    pdf_path = figures_dir() / f"{config['figure']['stem']}.pdf"
    atomic_parquet(pair_path, pair_rows)
    atomic_parquet(dispute_path, dispute_rows)
    atomic_csv(review_path, dispute_rows[[
        "item_id", "layer", "position", "pair_ordinal",
        "a500_token_id", "a1000_token_id", "a500_token_surface",
        "a1000_token_surface", "automatic_lexical_category",
        "manual_semantic_label", "manual_review_blinded_to_behavior",
    ]])
    make_figure(result, pair_rows, config, png_path, pdf_path)

    manifest_payload = {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "code_commit": clean["code_commit"],
        "config_sha256": file_sha256(config_path),
        "producer_sha256": file_sha256(Path(__file__)),
        "source_hashes": source_hashes,
        "functional_lens_hashes": {
            lens: functional["lenses"][lens]["lens_sha256"] for lens in PAIR},
        "contract": dict(config["contract"]),
        "row_swap_audit": dict(config["row_swap_audit"]),
        "gpu": gpu,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }
    manifest = {
        "schema_version": 1, "payload": manifest_payload,
        "payload_sha256": object_sha256(manifest_payload),
    }
    atomic_json(manifest_path, manifest)
    command = (
        "python -m jspace_phase4.experiments.p4_qwen_selection_margin "
        f"--config {arguments.config}")
    inputs = {
        "source_functional_evidence": config[
            "source_functional_evidence_id"],
        "source_capture": source_hashes["capture"],
        "source_selection": source_hashes["selection"],
        "source_pair_geometry": source_hashes["pairs"],
        "input_manifest": manifest["payload_sha256"],
        **{f"lens_{lens}": functional["lenses"][lens]["lens_sha256"]
           for lens in PAIR},
    }
    write_result4(
        result, result_path,
        Provenance4(
            evidence_id=config["evidence_id"], tier=config["tier"],
            command=command, inputs=inputs,
            input_manifest_sha256=manifest["payload_sha256"],
            model=model_reference(functional["model_uri"]),
            seed_contract=(
                "No random selection or exclusion; frozen A500/A1000 "
                "capture keys, margin threshold, row pairing, and all-strata "
                "retention from the prospective YAML."),
        ))
    outputs = [
        result_path, manifest_path, state_path, pair_path, dispute_path,
        review_path, *pair_parts, *dispute_parts, png_path, pdf_path,
    ]
    create(
        config["evidence_id"], tier=config["tier"],
        what=config["registry_what"], command=command,
        outputs=outputs, inputs=inputs)
    print(json.dumps({
        "evidence_id": config["evidence_id"],
        "functional_branch_candidate": result["functional_branch_candidate"],
        "stratum_counts": result["stratum_counts"],
        "manual_review_required_rows": result["lexical_audit"][
            "manual_review_required_rows"],
        "result": str(result_path), "figure": str(png_path),
    }, indent=1))


if __name__ == "__main__":
    main()
