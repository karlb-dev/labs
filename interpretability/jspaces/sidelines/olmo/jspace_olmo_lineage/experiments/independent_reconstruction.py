"""Independent reconstruction of the OLMo-lineage headline evidence.

The table checks reimplement the summaries from registered sufficient
statistics. A clean-process sentinel repeats exactly one already registered
Bank-W baseline row from the pinned model revision. No new scientific cell is
opened.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from ..manifests import (
    atomic_json,
    atomic_text,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths import local_work, resolve_uri
from ..registry import create, resolve
from .geometry import figures as reconstruct_geometry_figures

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise TypeError("reconstruction config must be a mapping")
    return value


def _assert_close(actual: float, expected: float, *, label: str,
                  atol: float = 1e-12) -> None:
    if not math.isclose(
            float(actual), float(expected), rel_tol=0.0, abs_tol=atol):
        raise ValueError(f"{label}: {actual} != {expected}")


def _registered_outputs(evidence_id: str) -> tuple[dict, list[dict]]:
    record = resolve(evidence_id)
    if not record["live"]:
        raise ValueError(f"evidence is not live: {evidence_id}")
    outputs = []
    for row in record.get("outputs", []):
        path = Path(row["path"])
        actual = file_sha256(path) if path.is_file() else None
        if actual != row["sha256"]:
            raise ValueError(f"registered output hash drift: {path}")
        outputs.append(dict(row))
    return record, outputs


def _output_by_hash(outputs: Sequence[Mapping], digest: str) -> dict:
    matches = [dict(row) for row in outputs if row["sha256"] == digest]
    if len(matches) != 1:
        raise ValueError(f"expected one registered output with hash {digest}")
    return matches[0]


def _payload(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text())
    if "payload" in value:
        if value.get("payload_sha256") != object_sha256(value["payload"]):
            raise ValueError(f"payload envelope hash drift: {path}")
        return value["payload"]
    if value.get("payload_sha256"):
        expected = value["payload_sha256"]
        body = dict(value)
        body.pop("payload_sha256")
        if object_sha256(body) != expected:
            raise ValueError(f"plain payload hash drift: {path}")
    return value


def _reconstruct_bank_w(config: dict) -> dict:
    specification = config["inputs"]["bank_w"]
    bank_config = yaml.safe_load(
        (PACKAGE_ROOT / specification["config"]).read_text())
    selection = bank_config["selection"]
    guard = bank_config["capability_guard"]
    per_model = {}
    for key in ("think", "instruct"):
        evidence_id = specification[f"{key}_evidence_id"]
        _, outputs = _registered_outputs(evidence_id)
        rows_output = _output_by_hash(
            outputs, specification[f"{key}_rows_sha256"])
        result_output = _output_by_hash(
            outputs, specification[f"{key}_result_sha256"])
        frame = pd.read_parquet(rows_output["path"])
        expected = _payload(result_output["path"])["analysis"]
        records = frame.to_dict("records")
        aliases = list(bank_config["answer_contract"]["aliases"])
        score_checks = 0
        finite_values = 0
        for row in records:
            scores = json.loads(row["candidate_scores_json"])
            if sorted(scores) != sorted(aliases):
                raise ValueError("Bank-W candidate alias set drift")
            predicted = max(aliases, key=lambda alias: scores[alias])
            true = row["true_alias"]
            wrong = max(value for alias, value in scores.items()
                        if alias != true)
            if predicted != row["predicted_alias"]:
                raise ValueError("Bank-W stored prediction drift")
            if bool(predicted == true) != bool(row["correct"]):
                raise ValueError("Bank-W stored correctness drift")
            _assert_close(
                scores[true] - wrong, row["baseline_answer_margin"],
                label="Bank-W answer margin", atol=1e-7)
            _assert_close(
                scores[true], row["true_answer_sequence_lp"],
                label="Bank-W true sequence LP", atol=1e-7)
            finite = list(scores.values()) + [
                row["baseline_answer_margin"],
                row["true_answer_sequence_lp"],
                row["prompt_token_count"], row["answer_token_count"],
            ]
            if not np.isfinite(np.asarray(finite, dtype=np.float64)).all():
                raise ValueError("non-finite Bank-W registered row")
            finite_values += len(finite)
            score_checks += 1

        loads = list(selection["loads"])
        families = sorted(frame["canonical_family"].unique().tolist())
        family_accuracy = {}
        for family in families:
            family_accuracy[family] = {
                load: float(frame[
                    (frame["canonical_family"] == family)
                    & (frame["load"] == load)]["correct"].mean())
                for load in loads
            }
        load_accuracy = {
            load: float(frame[frame["load"] == load]["correct"].mean())
            for load in loads
        }
        differences = np.asarray([
            family_accuracy[family]["high"]
            - family_accuracy[family]["low"]
            for family in families
        ], dtype=np.float64)
        rng = np.random.default_rng(int(guard["family_bootstrap_seed"]))
        indices = rng.integers(
            0, len(families), size=(int(guard["family_bootstrap_draws"]),
                                    len(families)))
        distribution = differences[indices].mean(axis=1)
        interval = np.quantile(distribution, [0.05, 0.95])
        capable = [
            family for family in families
            if all(family_accuracy[family][load] >= float(guard[
                "family_capability_accuracy_floor_by_load"])
                   for load in loads)
        ]
        for load in loads:
            _assert_close(
                load_accuracy[load],
                expected["load_summaries"][load]["accuracy"],
                label=f"{key} {load} accuracy")
        _assert_close(
            differences.mean(),
            expected["paired_high_minus_low_accuracy"]["mean"],
            label=f"{key} paired mean")
        for index, endpoint in enumerate(interval):
            _assert_close(
                endpoint,
                expected["paired_high_minus_low_accuracy"][
                    "family_bootstrap_ci90"][index],
                label=f"{key} bootstrap endpoint")
        if capable != expected["capable_family_ids"]:
            raise ValueError(f"{key} capable-family reconstruction drift")
        if family_accuracy != expected["family_accuracy"]:
            raise ValueError(f"{key} family-accuracy reconstruction drift")
        per_model[key] = {
            "evidence_id": evidence_id,
            "rows": len(frame),
            "candidate_rows_reconstructed": score_checks,
            "numeric_values_checked": finite_values,
            "accuracy": load_accuracy,
            "high_minus_low": float(differences.mean()),
            "ci90": [float(value) for value in interval],
            "capable_family_ids": capable,
            "capable_family_ids_sha256": object_sha256(capable),
            "bootstrap_distribution_sha256": object_sha256(
                distribution.tolist()),
        }

    _, joint_outputs = _registered_outputs(
        specification["joint_evidence_id"])
    joint_output = _output_by_hash(
        joint_outputs, specification["joint_json_sha256"])
    joint = _payload(joint_output["path"])
    model_sets = [
        set(joint["model_analyses"][slug]["capable_family_ids"])
        for slug in joint["model_order"]
    ]
    common = sorted(set.intersection(*model_sets))
    if common != joint["joint_common_capable_family_ids"]:
        raise ValueError("joint capable-family intersection drift")
    service = bool(
        joint["all_required_models_independently_eligible"]
        and len(common) >= joint["minimum_joint_common_families"])
    if service != joint["olmo_phase4_service_ready"]:
        raise ValueError("joint Bank-W service decision drift")
    return {
        "status": "pass",
        "models": per_model,
        "joint_common_families": len(common),
        "minimum_joint_common_families": int(
            joint["minimum_joint_common_families"]),
        "service_ready": service,
        "joint_common_family_ids_sha256": object_sha256(common),
    }


def _independent_curve(
    j_errors: np.ndarray,
    random_errors: np.ndarray,
    *,
    persistence: int,
) -> dict:
    j = np.asarray(j_errors, dtype=np.float64)
    random = np.asarray(random_errors, dtype=np.float64)
    j_gain = j[:, :-1] - j[:, 1:]
    random_gain = random[:, :, :-1] - random[:, :, 1:]
    below = j_gain <= np.median(random_gain, axis=0)
    occupancy = np.full(len(j), j.shape[1] - 1, dtype=np.int16)
    run = np.zeros(len(j), dtype=np.int16)
    done = np.zeros(len(j), dtype=bool)
    for index in range(below.shape[1]):
        run = np.where(below[:, index], run + 1, 0)
        hit = (~done) & (run >= persistence)
        occupancy[hit] = index + 1 - (persistence - 1)
        done |= hit
    occupancy = np.maximum(occupancy, 1)
    k = int(np.partition(
        occupancy, (len(occupancy) - 1) // 2)[
            (len(occupancy) - 1) // 2])
    energy = float(j[:, 0].sum())
    j_share = float(1.0 - j[:, k].sum() / energy)
    random_shares = [
        float(1.0 - value[:, k].sum() / energy) for value in random
    ]
    return {
        "n_positions": len(j),
        "k_max": j.shape[1] - 1,
        "occupancy_median": k,
        "occupancy_histogram": np.bincount(
            occupancy, minlength=j.shape[1]).astype(int).tolist(),
        "occupancy_q25": float(np.quantile(occupancy, 0.25)),
        "occupancy_q75": float(np.quantile(occupancy, 0.75)),
        "occupancy_censored_fraction": float(np.mean(
            occupancy >= j.shape[1] - 1)),
        "j_share": j_share,
        "random_seed_shares": random_shares,
        "random_share": float(np.mean(random_shares)),
        "excess_share": float(j_share - np.mean(random_shares)),
        "target_energy": energy,
    }


def _compare_curve(reconstructed: dict, expected: dict, label: str) -> None:
    for key in (
        "n_positions", "k_max", "occupancy_median",
        "occupancy_histogram"):
        if reconstructed[key] != expected[key]:
            raise ValueError(f"{label} {key} drift")
    for key in (
        "occupancy_q25", "occupancy_q75",
        "occupancy_censored_fraction", "j_share", "random_share",
        "excess_share", "target_energy"):
        _assert_close(
            reconstructed[key], expected[key], label=f"{label} {key}")
    if len(reconstructed["random_seed_shares"]) != len(
            expected["random_seed_shares"]):
        raise ValueError(f"{label} random seed count drift")
    for index, actual in enumerate(reconstructed["random_seed_shares"]):
        _assert_close(
            actual, expected["random_seed_shares"][index],
            label=f"{label} random share {index}")


def _quantile_pair(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(
        np.asarray(values, dtype=np.float64), [0.05, 0.95])
    return float(low), float(high)


def _ordered_pairs(
    model_order: Sequence[str], available_models: Sequence[str],
) -> list[tuple[str, str]]:
    order = list(model_order)
    if len(order) != len(set(order)):
        raise ValueError("capacity joint model order contains duplicates")
    if set(order) != set(available_models):
        raise ValueError("capacity joint model order does not match inputs")
    return [(left, right) for index, left in enumerate(order)
            for right in order[index + 1:]]


def _reconstruct_capacity(config: dict) -> dict:
    specification = config["inputs"]["capacity"]
    capacity_config = yaml.safe_load(
        (PACKAGE_ROOT / specification["config"]).read_text())
    persistence = int(capacity_config["estimator"]["crossing"][
        "persistence"])
    layer_records = {}
    point_summaries_checked = 0
    bootstrap_intervals_checked = 0
    model_arrays = {}
    for evidence_id in specification["model_evidence_ids"]:
        _, outputs = _registered_outputs(evidence_id)
        result_rows = [row for row in outputs
                       if Path(row["path"]).name == "capacity_result.json"]
        if len(result_rows) != 1:
            raise ValueError(f"capacity result ambiguity: {evidence_id}")
        result = _payload(result_rows[0]["path"])
        slug = result["model_slug"]
        layer_records[slug] = {}
        model_arrays[slug] = {}
        for output in outputs:
            name = Path(output["path"]).name
            if not name.startswith("capacity_layer_") or not name.endswith(
                    ".npz"):
                continue
            layer = int(name.removeprefix("capacity_layer_").removesuffix(
                ".npz"))
            with np.load(output["path"], allow_pickle=False) as values:
                arrays = {key: values[key].copy() for key in values.files}
            summary = json.loads(str(arrays.pop("summary_json").item()))
            metadata = json.loads(str(arrays.pop("metadata_json").item()))
            if metadata["summary_sha256"] != object_sha256(summary):
                raise ValueError("capacity summary semantic hash drift")
            layer_records[slug][layer] = summary
            model_arrays[slug][layer] = arrays
            for frame, prefix in (("own", "own"),
                                  ("base_common", "common")):
                for target, summary_key in (
                        ("centered", "primary_centered"),
                        ("raw", "raw_sensitivity")):
                    reconstructed = _independent_curve(
                        arrays[f"{prefix}_{target}_errors"],
                        arrays[f"random_{target}_errors"],
                        persistence=persistence)
                    expected = summary[frame][summary_key]
                    _compare_curve(
                        reconstructed, expected,
                        f"{slug} L{layer} {frame} {target}")
                    point_summaries_checked += 1
                    bootstrap_prefix = (
                        "own" if frame == "own" else "common")
                    distribution = arrays[
                        f"{bootstrap_prefix}_bootstrap_{target}_excess"]
                    low, high = _quantile_pair(distribution)
                    interval = expected["prompt_bootstrap"]["excess_share"]
                    _assert_close(
                        low, interval["low"],
                        label=f"{slug} L{layer} bootstrap low")
                    _assert_close(
                        high, interval["high"],
                        label=f"{slug} L{layer} bootstrap high")
                    occupancy_distribution = arrays[
                        f"{bootstrap_prefix}_bootstrap_{target}_occupancy"]
                    occ_low, occ_high = _quantile_pair(
                        occupancy_distribution)
                    occ_interval = expected["prompt_bootstrap"][
                        "occupancy_median"]
                    _assert_close(
                        occ_low, occ_interval["low"],
                        label="capacity occupancy bootstrap low")
                    _assert_close(
                        occ_high, occ_interval["high"],
                        label="capacity occupancy bootstrap high")
                    bootstrap_intervals_checked += 2

    _, joint_outputs = _registered_outputs(
        specification["joint_evidence_id"])
    result_output = _output_by_hash(
        joint_outputs, specification["joint_json_sha256"])
    table_output = _output_by_hash(
        joint_outputs, specification["joint_table_sha256"])
    bootstrap_output = _output_by_hash(
        joint_outputs, specification["joint_bootstrap_sha256"])
    joint_result = _payload(result_output["path"])
    table = pd.read_parquet(table_output["path"])
    with np.load(bootstrap_output["path"], allow_pickle=False) as values:
        joint_arrays = {key: values[key].copy() for key in values.files}

    arrays_checked = 0
    layers = sorted(next(iter(layer_records.values())))
    frames = (("own", "own"), ("base_common", "common"))
    for slug in sorted(layer_records):
        for layer in layers:
            for frame, prefix in frames:
                for metric, suffix in (
                        ("centered_excess", "centered_excess"),
                        ("occupancy", "centered_occupancy")):
                    joint_key = (
                        f"model__{slug}__{frame}__L{layer}__{metric}")
                    local_key = f"{prefix}_bootstrap_{suffix}"
                    if not np.array_equal(
                            joint_arrays[joint_key],
                            model_arrays[slug][layer][local_key]):
                        raise ValueError(f"joint model array drift: {joint_key}")
                    arrays_checked += 1

    pairs = _ordered_pairs(joint_result["model_order"], layer_records)
    for left, right in pairs:
        for frame, _ in frames:
            for metric in ("centered_difference", "occupancy_difference"):
                model_metric = (
                    "centered_excess" if metric == "centered_difference"
                    else "occupancy")
                layer_values = []
                for layer in layers:
                    expected = (
                        joint_arrays[
                            f"model__{right}__{frame}__L{layer}__{model_metric}"]
                        - joint_arrays[
                            f"model__{left}__{frame}__L{layer}__{model_metric}"])
                    key = (
                        f"pair__{left}__{right}__{frame}__L{layer}__{metric}")
                    if not np.array_equal(joint_arrays[key], expected):
                        raise ValueError(f"joint pair array drift: {key}")
                    arrays_checked += 1
                    layer_values.append(expected)
                equal = np.mean(np.stack(layer_values), axis=0)
                key = (
                    f"pair__{left}__{right}__{frame}__equal_layer__{metric}")
                if not np.array_equal(joint_arrays[key], equal):
                    raise ValueError(f"joint equal-layer array drift: {key}")
                arrays_checked += 1

    table_rows_checked = 0
    for row in table.to_dict("records"):
        frame = row["frame"]
        if row["row_type"] == "model_estimate":
            layer = int(row["layer"])
            summary = layer_records[row["left"]][layer][frame][
                "primary_centered"]
            _assert_close(
                row["centered_excess"], summary["excess_share"],
                label="capacity table model point")
            if int(row["occupancy_median"]) != summary["occupancy_median"]:
                raise ValueError("capacity table model occupancy drift")
            distribution = joint_arrays[
                f"model__{row['left']}__{frame}__L{layer}__centered_excess"]
            occupancy_distribution = joint_arrays[
                f"model__{row['left']}__{frame}__L{layer}__occupancy"]
            point_key = "centered_excess"
        else:
            equal = row["row_type"] == "pair_contrast_equal_layer"
            layer_key = "equal_layer" if equal else f"L{int(row['layer'])}"
            suffix = "equal_layer" if equal else layer_key
            distribution = joint_arrays[
                f"pair__{row['left']}__{row['right']}__{frame}__{suffix}__centered_difference"]
            occupancy_distribution = joint_arrays[
                f"pair__{row['left']}__{row['right']}__{frame}__{suffix}__occupancy_difference"]
            point_values = []
            occupancy_values = []
            target_layers = layers if equal else [int(row["layer"])]
            for layer in target_layers:
                left_summary = layer_records[row["left"]][layer][frame][
                    "primary_centered"]
                right_summary = layer_records[row["right"]][layer][frame][
                    "primary_centered"]
                point_values.append(
                    right_summary["excess_share"] - left_summary[
                        "excess_share"])
                occupancy_values.append(
                    right_summary["occupancy_median"] - left_summary[
                        "occupancy_median"])
            _assert_close(
                row["centered_difference"], np.mean(point_values),
                label="capacity table pair point")
            _assert_close(
                row["occupancy_difference"], np.mean(occupancy_values),
                label="capacity table pair occupancy")
            point_key = "centered_difference"
        low, high = _quantile_pair(distribution)
        occ_low, occ_high = _quantile_pair(occupancy_distribution)
        _assert_close(low, row["centered_ci_low"], label="table CI low")
        _assert_close(high, row["centered_ci_high"], label="table CI high")
        _assert_close(occ_low, row["occupancy_ci_low"], label="table occ low")
        _assert_close(
            occ_high, row["occupancy_ci_high"], label="table occ high")
        if not np.isfinite(float(row[point_key])):
            raise ValueError("capacity table point is non-finite")
        table_rows_checked += 1

    if joint_result["lineage_verdict"] != (
            "broadly_conserved_capacity_recruitment_consistent"):
        raise ValueError("capacity lineage verdict drift")
    return {
        "status": "pass",
        "model_layer_checkpoints": sum(
            len(value) for value in layer_records.values()),
        "point_summaries_reconstructed": point_summaries_checked,
        "bootstrap_intervals_reconstructed": bootstrap_intervals_checked,
        "joint_bootstrap_arrays_reconstructed": arrays_checked,
        "joint_table_rows_reconstructed": table_rows_checked,
        "lineage_verdict": joint_result["lineage_verdict"],
    }


def _stats(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q50": float(np.quantile(array, 0.50)),
        "q95": float(np.quantile(array, 0.95)),
    }


def _compare_stats(actual: dict, expected: dict, label: str) -> None:
    if actual["n"] != expected["n"]:
        raise ValueError(f"{label} count drift")
    for key in ("mean", "minimum", "maximum", "q05", "q50", "q95"):
        _assert_close(actual[key], expected[key], label=f"{label} {key}")


def _reconstruct_geometry(config: dict) -> dict:
    specification = config["inputs"]["geometry"]
    geometry_config = yaml.safe_load(
        (PACKAGE_ROOT / specification["config"]).read_text())
    _, outputs = _registered_outputs(specification["joint_evidence_id"])
    result_output = _output_by_hash(outputs, specification["result_sha256"])
    layer_output = _output_by_hash(
        outputs, specification["layer_table_sha256"])
    selection_output = _output_by_hash(
        outputs, specification["selection_table_sha256"])
    readout_output = _output_by_hash(
        outputs, specification["readout_table_sha256"])
    result = _payload(result_output["path"])
    layer = pd.read_parquet(layer_output["path"])
    selection = pd.read_parquet(selection_output["path"])
    readout = pd.read_parquet(readout_output["path"])
    layer_metrics = (
        "raw_matrix_cosine", "symmetric_relative_frobenius_delta",
        "j_minus_identity_cosine", "j_minus_alpha_identity_cosine",
        "probe_transport_cosine_q50", "mapped_token_cosine_q05",
        "mapped_token_cosine_q50", "mapped_token_centered_linear_cka",
        "mapped_neighbor_overlap_fraction")
    selection_metrics = (
        "selected_id_jaccard_q50", "rank_biased_overlap_q50",
        "projector_overlap_q50", "principal_angle_median_degrees_q50",
        "persistent_direction_jaccard")
    summaries_checked = 0
    for pair_id, rows in layer.groupby("pair_id"):
        for metric in layer_metrics:
            _compare_stats(
                _stats(rows[metric].to_numpy()),
                result["pairwise_layer_aggregate"][pair_id][metric],
                f"geometry {pair_id} {metric}")
            summaries_checked += 1
    for pair_id, rows in selection.groupby("pair_id"):
        for metric in selection_metrics:
            _compare_stats(
                _stats(rows[metric].to_numpy()),
                result["pairwise_selection_aggregate"][pair_id][metric],
                f"selection {pair_id} {metric}")
            summaries_checked += 1
    readout_records = readout.to_dict("records")
    if readout_records != result["readout_pairs"]:
        raise ValueError("readout table-to-JSON reconstruction drift")

    primary = layer[layer["edge_type"] == "primary"]
    assay = set(map(int, geometry_config["geometry_series"]["assay_layers"]))
    broad_operator = bool(
        np.median(primary["raw_matrix_cosine"]) >= 0.90
        and primary[primary["layer"].isin(assay)][
            "raw_matrix_cosine"].min() >= 0.85)
    broad_token = bool(
        np.median(primary["mapped_token_cosine_q50"]) >= 0.85
        and np.median(primary["mapped_token_cosine_q05"]) >= 0.70)
    later = layer[
        (layer["left"] == "olmo3-think")
        & (layer["right"] == "olmo31-think")]
    base_movement = 1.0 - float(np.median(
        primary["mapped_token_cosine_q50"]))
    later_movement = 1.0 - float(np.median(
        later["mapped_token_cosine_q50"]))
    formation = bool(
        base_movement >= 1.5 * max(later_movement, 1e-12)
        and base_movement - later_movement >= 0.03)
    primary_selection = selection[selection["edge_type"] == "primary"]
    selection_divergence = bool(
        np.median(primary_selection["selected_id_jaccard_q50"]) < 0.75
        or np.median(primary_selection["projector_overlap_q50"]) < 0.85)
    sibling = layer[layer["edge_type"] == "sibling"]
    early = set(map(int, geometry_config["geometry_series"]["early_layers"]))
    late = set(map(int, geometry_config["geometry_series"]["late_layers"]))
    early_movement = float(np.median(
        1.0 - sibling[sibling["layer"].isin(early)][
            "mapped_token_cosine_q50"]))
    late_movement = float(np.median(
        1.0 - sibling[sibling["layer"].isin(late)][
            "mapped_token_cosine_q50"]))
    instruct_late = bool(late_movement - early_movement >= 0.03)
    verdict = (
        "broad-continuity-with-selection-change"
        if broad_operator and broad_token and selection_divergence else
        "dictionary-formation-pattern" if formation else
        "coordinate-drift-with-common-coarse-channel"
        if (not broad_operator and broad_token) else "mixed-or-unresolved")
    reconstructed_router = {
        "verdict": verdict,
        "broad_operator_continuity": broad_operator,
        "broad_token_continuity": broad_token,
        "selection_divergence_flag": selection_divergence,
        "dictionary_formation_pattern": formation,
        "base_to_3_0_mapped_movement": base_movement,
        "3_0_to_3_1_mapped_movement": later_movement,
        "instruct_late_shift": instruct_late,
        "sibling_early_mapped_movement": early_movement,
        "sibling_late_mapped_movement": late_movement,
    }
    for key, actual in reconstructed_router.items():
        expected = result["router"][key]
        if isinstance(actual, float):
            _assert_close(actual, expected, label=f"geometry router {key}")
        elif actual != expected:
            raise ValueError(f"geometry router drift: {key}")
    if not selection[
            ["exact_kth_kplus1_score_gap", "protected_span_overlap",
             "causal_core_fringe_dose"]].isna().all().all():
        raise ValueError("geometry unavailable quantities became non-null")
    return {
        "status": "pass",
        "layer_rows": len(layer),
        "selection_rows": len(selection),
        "readout_rows": len(readout),
        "aggregate_summaries_reconstructed": summaries_checked,
        "router": reconstructed_router,
        "unavailable_quantities_preserved_null": True,
    }


def _reconstruct_figures(config: dict) -> tuple[dict, list[Path]]:
    specification = config["inputs"]["geometry"]
    _, figure_outputs = _registered_outputs(
        specification["figure_evidence_id"])
    manifest_output = _output_by_hash(
        figure_outputs, specification["figure_manifest_sha256"])
    registered_manifest = json.loads(Path(
        manifest_output["path"]).read_text())
    registered = {
        Path(row["path"]).name: row for row in registered_manifest["figures"]
    }
    temporary = Path(tempfile.mkdtemp(
        prefix="olmo_independent_figures_", dir=str(local_work())))
    reconstruction = reconstruct_geometry_figures(
        PACKAGE_ROOT / specification["config"],
        reconstruction_output_dir=temporary)
    comparisons = []
    for raw_path in reconstruction["outputs"]:
        path = Path(raw_path)
        if path.name not in registered:
            raise ValueError(f"unexpected reconstructed figure {path.name}")
        digest = file_sha256(path)
        expected = registered[path.name]["sha256"]
        exact = digest == expected
        if path.suffix == ".png" and config["figure_reconstruction"][
                "require_exact_png_bytes"] and not exact:
            raise ValueError(f"reconstructed PNG differs: {path.name}")
        if path.suffix == ".pdf":
            if not path.read_bytes().startswith(b"%PDF"):
                raise ValueError(f"invalid reconstructed PDF: {path}")
        comparisons.append({
            "name": path.name,
            "registered_sha256": expected,
            "reconstructed_sha256": digest,
            "exact_bytes": exact,
            "bytes": int(path.stat().st_size),
        })
    png_rows = [row for row in comparisons if row["name"].endswith(".png")]
    pdf_rows = [row for row in comparisons if row["name"].endswith(".pdf")]
    if len(png_rows) != 5 or not all(row["exact_bytes"] for row in png_rows):
        raise ValueError("not all five reconstructed PNGs match exactly")
    if config["figure_reconstruction"]["require_all_pdf_outputs"] \
            and len(pdf_rows) != 5:
        raise ValueError("not all five PDFs regenerated")

    expected_names = set(registered)
    reconstructed_names = {row["name"] for row in comparisons}
    if reconstructed_names != expected_names:
        raise ValueError("reconstructed figure set differs from registry")
    destination = resolve_uri(
        config["outputs"]["figure_directory"], must_exist=False)
    if destination.exists():
        raise FileExistsError(
            f"refusing existing reconstruction figure directory {destination}")
    staging = destination.with_name(f".{destination.name}.staging")
    if staging.exists():
        raise FileExistsError(f"refusing stale figure staging directory {staging}")
    staging.mkdir(parents=True)
    copied = []
    for raw_path in reconstruction["outputs"]:
        path = Path(raw_path)
        target = staging / path.name
        shutil.copy2(path, target)
    manifest_source = Path(reconstruction["manifest_path"])
    shutil.copy2(manifest_source, staging / manifest_source.name)
    staging.rename(destination)
    copied.extend(sorted(destination.iterdir()))
    return ({
        "status": "pass",
        "isolated_local_root": str(temporary),
        "durable_figure_root": str(destination),
        "png_exact_byte_matches": sum(
            row["exact_bytes"] for row in png_rows),
        "png_count": len(png_rows),
        "pdf_regenerated": len(pdf_rows),
        "pdf_exact_byte_matches": sum(
            row["exact_bytes"] for row in pdf_rows),
        "comparisons": comparisons,
    }, copied)


def _model_sentinel(config: dict, snapshot: Path) -> dict:
    import torch
    import transformers

    from ..compat import (
        DEFAULT_SPEC,
        ScoringSession,
        candidate_scores,
        select_development_rows,
    )
    from ..gpu import assert_model_on_cuda, require_cuda_gpu

    sentinel = config["sentinel"]
    if not snapshot.is_dir():
        raise FileNotFoundError(snapshot)
    inventory_spec = config["inputs"]["checkpoint_inventory"]
    _, inventory_outputs = _registered_outputs(inventory_spec["evidence_id"])
    inventory_output = _output_by_hash(
        inventory_outputs, inventory_spec["json_sha256"])
    inventory = _payload(inventory_output["path"])
    matches = [row for row in inventory["artifacts"]
               if row["slug"] == sentinel["model_slug"]]
    if len(matches) != 1 or matches[0]["revision"] != sentinel["revision"]:
        raise ValueError("sentinel inventory placement drift")
    artifact = matches[0]
    metadata_hashes = {}
    for name, row in artifact["metadata_files"].items():
        path = snapshot / name
        actual = file_sha256(path)
        if actual != row["sha256"]:
            raise ValueError(f"sentinel metadata hash drift: {name}")
        metadata_hashes[name] = actual
    weight_rows = []
    for index, row in enumerate(artifact["weights"]["shards"], start=1):
        path = snapshot / row["name"]
        if path.stat().st_size != row["bytes"]:
            raise ValueError(f"sentinel weight size drift: {path.name}")
        print(
            f"sentinel weight hash {index}/{artifact['weights']['shard_count']}: "
            f"{path.name}", flush=True)
        actual = file_sha256(path)
        if sentinel["require_all_weight_shard_hashes"] \
                and actual != row["sha256"]:
            raise ValueError(f"sentinel weight hash drift: {path.name}")
        weight_rows.append({
            "name": path.name, "bytes": int(path.stat().st_size),
            "sha256": actual,
        })

    bank_spec = config["inputs"]["bank_w"]
    bank_config = yaml.safe_load(
        (PACKAGE_ROOT / bank_spec["config"]).read_text())
    _, think_outputs = _registered_outputs(bank_spec["think_evidence_id"])
    rows_output = _output_by_hash(
        think_outputs, bank_spec["think_rows_sha256"])
    registered_rows = pd.read_parquet(rows_output["path"])
    selected_row = registered_rows[
        registered_rows["item_id"] == sentinel["item_id"]]
    if len(selected_row) != 1:
        raise ValueError("sentinel registered row is absent or ambiguous")
    expected_row = selected_row.iloc[0].to_dict()
    bank_path = resolve_uri(bank_config["bank_uri"])
    bank_rows = [json.loads(line) for line in bank_path.read_text().splitlines()
                 if line.strip()]
    selected = select_development_rows(
        bank_rows, bank_config["selection"])
    items = [row for row in selected if row["item_id"] == sentinel["item_id"]]
    if len(items) != 1:
        raise ValueError("sentinel bank item is absent or ambiguous")
    item = items[0]
    aliases = list(bank_config["answer_contract"]["aliases"])
    labels = list(bank_config["answer_contract"]["labels"])
    alias_by_label = dict(zip(labels, aliases))
    candidate_batch_size = int(sentinel["candidate_batch_size"])
    maximum_batch_size = int(bank_config["answer_contract"][
        "runtime_candidate_batch_size"])
    if not 1 <= candidate_batch_size <= maximum_batch_size:
        raise ValueError(
            "sentinel candidate batch size must be within the frozen runtime "
            "contract")

    gpu = require_cuda_gpu()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True)
    session = ScoringSession(tokenizer, DEFAULT_SPEC, device="cuda")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        snapshot, dtype=torch.bfloat16, local_files_only=True,
        low_cpu_mem_usage=True).to("cuda").eval()
    assert_model_on_cuda(model)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    with torch.no_grad():
        scores, prompt_tokens, token_manifest = candidate_scores(
            model, session, item["prompt"], aliases,
            batch_size=candidate_batch_size,
            pad_token_id=int(pad_token_id))
    del model
    gc.collect()
    torch.cuda.empty_cache()

    expected_scores = json.loads(expected_row["candidate_scores_json"])
    differences = {
        alias: float(scores[alias] - expected_scores[alias])
        for alias in aliases
    }
    maximum = max(abs(value) for value in differences.values())
    if maximum > float(sentinel["candidate_log_probability_atol"]):
        raise ValueError(f"sentinel candidate LP drift: {maximum}")
    predicted = max(aliases, key=lambda alias: scores[alias])
    true_alias = alias_by_label[item["answer"]]
    margin = float(scores[true_alias] - max(
        scores[alias] for alias in aliases if alias != true_alias))
    if sentinel["require_prediction_match"] and predicted != expected_row[
            "predicted_alias"]:
        raise ValueError("sentinel predicted alias drift")
    if sentinel["require_margin_match"]:
        _assert_close(
            margin, expected_row["baseline_answer_margin"],
            label="sentinel margin",
            atol=float(sentinel["candidate_log_probability_atol"]))
    if prompt_tokens != int(expected_row["prompt_token_count"]):
        raise ValueError("sentinel prompt token count drift")
    if object_sha256(token_manifest) != expected_row[
            "answer_token_manifest_sha256"]:
        raise ValueError("sentinel answer-token manifest drift")
    return {
        "status": "pass",
        "model_id": sentinel["model_id"],
        "revision": sentinel["revision"],
        "item_id": sentinel["item_id"],
        "gpu": gpu,
        "metadata_hashes": metadata_hashes,
        "weight_shards_verified": len(weight_rows),
        "weight_manifest_sha256": object_sha256(weight_rows),
        "candidate_count": len(scores),
        "candidate_batch_size": candidate_batch_size,
        "frozen_runtime_candidate_batch_size": maximum_batch_size,
        "expected_candidate_scores_sha256": object_sha256(expected_scores),
        "reconstructed_candidate_scores_sha256": object_sha256(scores),
        "maximum_absolute_log_probability_difference": maximum,
        "candidate_differences": differences,
        "predicted_alias": predicted,
        "prediction_matches": predicted == expected_row["predicted_alias"],
        "margin": margin,
        "expected_margin": float(expected_row["baseline_answer_margin"]),
        "prompt_token_count": prompt_tokens,
        "answer_token_manifest_sha256": object_sha256(token_manifest),
    }


def _render_markdown(result: dict) -> str:
    bank = result["reconstruction"]["bank_w"]
    capacity = result["reconstruction"]["capacity"]
    geometry = result["reconstruction"]["geometry"]
    figures = result["reconstruction"]["figures"]
    sentinel = result["reconstruction"]["model_sentinel"]
    return "\n".join([
        "# OLMo lineage independent reconstruction",
        "",
        f"Evidence: `{result['evidence_id']}`",
        "",
        "Overall status: **PASS**.",
        "",
        "## Reconstructed boundaries",
        "",
        f"- Bank-W: {sum(row['candidate_rows_reconstructed'] for row in bank['models'].values())} "
        f"rows, joint support {bank['joint_common_families']} / "
        f"{bank['minimum_joint_common_families']}, service-ready "
        f"`{str(bank['service_ready']).lower()}`.",
        f"- Capacity: {capacity['point_summaries_reconstructed']} point "
        f"summaries, {capacity['bootstrap_intervals_reconstructed']} intervals, "
        f"{capacity['joint_bootstrap_arrays_reconstructed']} joint arrays, and "
        f"{capacity['joint_table_rows_reconstructed']} table rows.",
        f"- Geometry: {geometry['aggregate_summaries_reconstructed']} aggregate "
        f"summaries and the complete router; verdict "
        f"`{geometry['router']['verdict']}`.",
        f"- Figures: {figures['png_exact_byte_matches']}/5 regenerated PNGs "
        f"match byte-for-byte; {figures['pdf_regenerated']} PDFs regenerated.",
        f"- Clean-process model sentinel: {sentinel['weight_shards_verified']} "
        f"weight shards verified; {sentinel['candidate_count']} candidate "
        f"sequence LPs repeated; maximum absolute difference "
        f"{sentinel['maximum_absolute_log_probability_difference']:.3g} nats.",
        "",
        "## Boundary",
        "",
        result["claim_boundary"],
        "",
    ])


def run(config_path: str | Path, *, snapshot: str | Path) -> dict:
    config_path = Path(config_path)
    config = load_config(config_path)
    source = require_clean_tree(expected_branch=config["branch"])
    json_path = resolve_uri(config["outputs"]["json"], must_exist=False)
    markdown_path = resolve_uri(
        config["outputs"]["markdown"], must_exist=False)
    for path in (json_path, markdown_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    bank = _reconstruct_bank_w(config)
    capacity = _reconstruct_capacity(config)
    geometry = _reconstruct_geometry(config)
    sentinel = _model_sentinel(config, Path(snapshot))
    figures, figure_paths = _reconstruct_figures(config)
    result = {
        "schema_version": 1,
        "study_id": "jspace-olmo-lineage",
        "evidence_id": config["evidence_id"],
        "tier": config["tier"],
        "code_commit": source["code_commit"],
        "config_sha256": file_sha256(config_path),
        "scientific_import_boundary": config[
            "scientific_import_boundary"],
        "status": "pass",
        "reconstruction": {
            "bank_w": bank,
            "capacity": capacity,
            "geometry": geometry,
            "figures": figures,
            "model_sentinel": sentinel,
        },
        "claim_boundary": config["claim_boundary"],
    }
    result["payload_sha256"] = object_sha256(result)
    atomic_json(json_path, result)
    atomic_text(markdown_path, _render_markdown(result))
    command = (
        "python -m jspace_olmo_lineage.experiments.independent_reconstruction "
        f"--config {config_path} --snapshot {snapshot}")
    event = create(
        config["evidence_id"],
        tier=config["tier"],
        what=("Independent OLMo headline-table and figure reconstruction "
              "plus one clean-process registered-row model sentinel."),
        command=command,
        outputs=[json_path, markdown_path, *figure_paths],
        inputs={
            "config_sha256": result["config_sha256"],
            "bank_w_joint_sha256": config["inputs"]["bank_w"][
                "joint_json_sha256"],
            "capacity_joint_sha256": config["inputs"]["capacity"][
                "joint_json_sha256"],
            "geometry_joint_sha256": config["inputs"]["geometry"][
                "result_sha256"],
            "figure_manifest_sha256": config["inputs"]["geometry"][
                "figure_manifest_sha256"],
            "checkpoint_inventory_sha256": config["inputs"][
                "checkpoint_inventory"]["json_sha256"],
        },
        verdict="pass",
        claim_boundary=config["claim_boundary"],
    )
    return {"status": "registered", "event": event, "result": result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(PACKAGE_ROOT / "configs/ol_independent_reconstruction_v1.yaml"),
    )
    parser.add_argument("--snapshot", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(
        arguments.config, snapshot=arguments.snapshot),
        indent=1, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
