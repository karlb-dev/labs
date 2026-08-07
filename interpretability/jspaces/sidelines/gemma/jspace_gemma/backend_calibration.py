"""Pure analysis helpers for the isolated Gemma study-2 backend calibration.

This module intentionally has no registry or study-1 imports.  The ceiling is
therefore reconstructible from the frozen calibration config and raw G2.1 rows
alone.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


FULL_SUFFIX = "full"
FINITE = "finite"


def stable_seed(base: int, *parts: object) -> int:
    material = "|".join([str(base), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big") % (
        2**31
    )


def tensor_sha256(value: torch.Tensor) -> str:
    payload = value.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def direction_tensor(
    clean_selected: torch.Tensor,
    *,
    family: str,
    seed: int,
) -> torch.Tensor:
    """Return one deterministic unit residual direction on CPU."""
    reference = clean_selected.detach().float().cpu().reshape(-1)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    if family == "rademacher":
        value = torch.randint(
            0, 2, reference.shape, generator=generator, dtype=torch.int64
        ).float()
        value = value.mul_(2).sub_(1)
    elif family in {"gaussian", "sphere_tangent"}:
        value = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
        if family == "sphere_tangent":
            radial = reference / reference.norm().clamp_min(1e-30)
            value = value - torch.dot(value, radial) * radial
    else:
        raise ValueError(f"unsupported calibration direction family {family!r}")
    norm = value.norm()
    if not torch.isfinite(norm) or float(norm) == 0.0:
        raise RuntimeError("calibration direction is non-finite or degenerate")
    return value / norm


def compare_tensors(value: torch.Tensor, reference: torch.Tensor) -> dict:
    left = value.detach().float().cpu().reshape(-1)
    right = reference.detach().float().cpu().reshape(-1)
    if left.shape != right.shape:
        raise ValueError("backend tensors have different shapes")
    difference = left - right
    denominator = right.double().norm().clamp_min(1e-300)
    relative = float(difference.double().norm() / denominator)
    left_norm = float(left.double().norm())
    right_norm = float(right.double().norm())
    cosine = (
        float(F.cosine_similarity(left, right, dim=0))
        if left_norm > 0.0 and right_norm > 0.0
        else (1.0 if left_norm == right_norm else 0.0)
    )
    return {
        "relative_error": relative,
        "cosine": cosine,
        "max_absolute_difference": float(difference.abs().max()) if left.numel() else 0.0,
        "value_norm": left_norm,
        "reference_norm": right_norm,
        "elements": int(left.numel()),
    }


def bfloat16_quantum_metrics(
    difference: torch.Tensor,
    reference: torch.Tensor,
) -> dict:
    """Express a backend difference relative to local bfloat16 spacing.

    Spacing is evaluated at the magnitude of each reference tangent element,
    exactly as frozen in the G2.1 config.  Norms use float64 so subnormal
    bfloat16 spacings do not disappear when squared.
    """
    delta = difference.detach().float().cpu().reshape(-1).abs().double()
    ref = reference.detach().float().cpu().reshape(-1)
    if delta.shape != ref.shape:
        raise ValueError("difference/reference shape mismatch")
    magnitude = ref.abs().to(torch.bfloat16)
    toward = torch.full_like(magnitude, float("inf"))
    spacing = (torch.nextafter(magnitude, toward) - magnitude).float().double()
    if bool((spacing <= 0).any()) or not bool(torch.isfinite(spacing).all()):
        raise RuntimeError("invalid bfloat16 next-representable spacing")
    quantum_ratio = torch.where(delta == 0, torch.zeros_like(delta), delta / spacing)
    one_quantum_relative = float(
        spacing.norm() / ref.double().norm().clamp_min(1e-300)
    )
    return {
        "max_difference_dtype_quanta": float(quantum_ratio.max()) if ref.numel() else 0.0,
        "one_dtype_quantum_relative_equivalent": one_quantum_relative,
        "ten_dtype_quanta_relative_equivalent": 10.0 * one_quantum_relative,
        "spacing_min": float(spacing.min()) if ref.numel() else 0.0,
        "spacing_max": float(spacing.max()) if ref.numel() else 0.0,
    }


def _quantiles(values: list[float]) -> dict:
    if not values:
        raise RuntimeError("cannot summarize an empty calibration distribution")
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "q50": float(np.quantile(array, 0.50)),
        "q90": float(np.quantile(array, 0.90)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def _valid_full_rows(rows: list[dict], config: dict) -> list[dict]:
    primal_limit = float(config["router"]["primal_parity_relative_error_ceiling"])
    return [
        row
        for row in rows
        if row["suffix_variant"] == FULL_SUFFIX
        and row["finite_or_exception_state"] == FINITE
        and row["primal_relative_error"] is not None
        and float(row["primal_relative_error"]) <= primal_limit
    ]


def ceiling_for(rows: list[dict], config: dict) -> dict:
    if not rows:
        raise RuntimeError("no rows are eligible for a calibration ceiling")
    disagreement = _quantiles([float(row["tangent_relative_error"]) for row in rows])
    ten_quantum = _quantiles(
        [float(row["ten_dtype_quanta_relative_equivalent"]) for row in rows]
    )
    value = max(3.0 * disagreement["q99"], ten_quantum["q99"])
    return {
        "value": float(value),
        "formula": config["ceiling_rule"]["formula"],
        "three_times_disagreement_q99": float(3.0 * disagreement["q99"]),
        "ten_dtype_quanta_q99": float(ten_quantum["q99"]),
        "tangent_relative_error": disagreement,
        "ten_dtype_quanta_relative_equivalent": ten_quantum,
    }


def _prompt_bootstrap(
    rows: list[dict],
    config: dict,
    *,
    scope_batch: int | None,
) -> dict:
    settings = config["ceiling_rule"]["bootstrap"]
    prompt_ids = list(config["prompt_ids"])
    grouped = {prompt: [row for row in rows if row["prompt_id"] == prompt] for prompt in prompt_ids}
    if any(not grouped[prompt] for prompt in prompt_ids):
        raise RuntimeError("prompt bootstrap is missing a frozen prompt")
    rng = np.random.default_rng(int(settings["seed"]))
    values = []
    for _ in range(int(settings["draws"])):
        sample = rng.choice(prompt_ids, size=len(prompt_ids), replace=True)
        sampled_rows = [row for prompt in sample for row in grouped[str(prompt)]]
        if scope_batch is not None:
            sampled_rows = [
                row for row in sampled_rows if int(row["batch_size"]) == scope_batch
            ]
        values.append(ceiling_for(sampled_rows, config)["value"])
    alpha = (1.0 - float(settings["interval"])) / 2.0
    return {
        "resampling_unit": settings["resampling_unit"],
        "draws": int(settings["draws"]),
        "interval_mass": float(settings["interval"]),
        "seed": int(settings["seed"]),
        "lower": float(np.quantile(values, alpha)),
        "median": float(np.quantile(values, 0.5)),
        "upper": float(np.quantile(values, 1.0 - alpha)),
    }


def _architecture_route(valid: list[dict], config: dict) -> dict:
    model_keys = list(config["models"])
    normalized = {}
    for model_key in model_keys:
        subset = [row for row in valid if row["model_key"] == model_key]
        relative = _quantiles([float(row["tangent_relative_error"]) for row in subset])
        quantum = _quantiles(
            [float(row["ten_dtype_quanta_relative_equivalent"]) for row in subset]
        )
        normalized[model_key] = float(
            relative["q99"] / max(quantum["q99"], np.finfo(np.float64).tiny)
        )
    first, second = model_keys
    point_log_ratio = math.log(
        max(normalized[first], np.finfo(np.float64).tiny)
        / max(normalized[second], np.finfo(np.float64).tiny)
    )

    settings = config["ceiling_rule"]["bootstrap"]
    prompts = list(config["prompt_ids"])
    rng = np.random.default_rng(int(settings["seed"]) + 1)
    logs = []
    for _ in range(int(settings["draws"])):
        sample = rng.choice(prompts, size=len(prompts), replace=True)
        sampled = [
            row
            for prompt in sample
            for row in valid
            if row["prompt_id"] == str(prompt)
        ]
        boot = {}
        for model_key in model_keys:
            subset = [row for row in sampled if row["model_key"] == model_key]
            relative_q99 = _quantiles(
                [float(row["tangent_relative_error"]) for row in subset]
            )["q99"]
            quantum_q99 = _quantiles(
                [float(row["ten_dtype_quanta_relative_equivalent"]) for row in subset]
            )["q99"]
            boot[model_key] = relative_q99 / max(
                quantum_q99, np.finfo(np.float64).tiny
            )
        logs.append(
            math.log(
                max(boot[first], np.finfo(np.float64).tiny)
                / max(boot[second], np.finfo(np.float64).tiny)
            )
        )
    alpha = (1.0 - float(settings["interval"])) / 2.0
    lower = float(np.quantile(logs, alpha))
    upper = float(np.quantile(logs, 1.0 - alpha))
    ratio = math.exp(abs(point_log_ratio))
    threshold = float(
        config["ceiling_rule"]["per_model_route"]["normalized_q99_ratio_threshold"]
    )
    excludes_zero = lower > 0.0 or upper < 0.0
    active = ratio >= threshold and excludes_zero
    return {
        "active": bool(active),
        "normalized_q99_by_model": normalized,
        "point_log_ratio_first_over_second": float(point_log_ratio),
        "absolute_ratio": float(ratio),
        "log_ratio_bootstrap_90pct": {"lower": lower, "upper": upper},
        "interval_excludes_zero": bool(excludes_zero),
        "ratio_threshold": threshold,
        "model_order": model_keys,
    }


def _path_ambiguity(rows: list[dict], config: dict) -> dict:
    rule = config["router"]["path_ambiguity"]
    exceptions = [
        row for row in rows if row["finite_or_exception_state"] != FINITE
    ]
    candidates = [
        row
        for row in rows
        if row["finite_or_exception_state"] == FINITE
        and int(row["batch_size"]) == 1
        and float(row["tangent_cosine"]) < float(rule["batch1_tangent_cosine_floor"])
        and float(row["tangent_relative_error"])
        > float(rule["batch1_relative_error_over_ten_quantum_equivalent"])
        * float(row["ten_dtype_quanta_relative_equivalent"])
    ]
    grouped: dict[tuple, set[str]] = defaultdict(set)
    for row in candidates:
        grouped[(row["model_key"], int(row["layer"]), row["suffix_variant"])].add(
            row["prompt_id"]
        )
    reproducible = [
        {
            "model_key": key[0],
            "layer": key[1],
            "suffix_variant": key[2],
            "prompt_ids": sorted(prompts),
        }
        for key, prompts in sorted(grouped.items())
        if len(prompts) >= int(rule["reproducible_prompt_minimum"])
    ]
    active = bool(reproducible) or bool(
        exceptions and rule["any_nonfinite_exact_backend_is_blocking"]
    )
    return {
        "active": active,
        "batch1_candidate_rows": len(candidates),
        "reproducible_groups": reproducible,
        "exception_or_nonfinite_rows": len(exceptions),
    }


def _batch_nuisance(valid: list[dict], config: dict) -> dict:
    rule = config["router"]["batch_composition_nuisance"]
    by_batch = {
        size: _quantiles(
            [
                float(row["tangent_relative_error"])
                for row in valid
                if int(row["batch_size"]) == size
            ]
        )
        for size in config["batch_sizes"]
    }
    ratio = by_batch[8]["q99"] / max(
        by_batch[1]["q99"], np.finfo(np.float64).tiny
    )
    active = (
        ratio >= float(rule["batch8_to_batch1_q99_ratio_floor"])
        and by_batch[8]["q90"] > by_batch[1]["q99"]
    )
    return {
        "active_without_path_ambiguity_gate": bool(active),
        "q99_ratio_batch8_over_batch1": float(ratio),
        "by_batch": {str(key): value for key, value in by_batch.items()},
    }


def derive_calibration(rows: list[dict], config: dict) -> dict:
    """Reconstruct the full G2.1 envelope and frozen router from raw rows."""
    expected_pairs = int(config["pair_count_contract"]["expected_backend_pairs"])
    full_pair_ids = {
        row["pair_id"] for row in rows if row["suffix_variant"] == FULL_SUFFIX
    }
    expected_full_rows = (
        len(config["models"])
        * 3
        * len(config["prompt_ids"])
        * len(config["directions"]["draws"])
        * sum(int(size) for size in config["batch_sizes"])
    )
    if len(full_pair_ids) != expected_pairs:
        raise RuntimeError(
            f"expected {expected_pairs} full backend pairs, found {len(full_pair_ids)}"
        )
    full_rows = [row for row in rows if row["suffix_variant"] == FULL_SUFFIX]
    if len(full_rows) != expected_full_rows:
        raise RuntimeError(
            f"expected {expected_full_rows} full rows, found {len(full_rows)}"
        )
    if len({row["row_id"] for row in rows}) != len(rows):
        raise RuntimeError("raw calibration row IDs are not unique")

    valid = _valid_full_rows(rows, config)
    if not valid:
        raise RuntimeError("no finite primal-passing full rows")
    pooled_all = ceiling_for(valid, config)
    path = _path_ambiguity(rows, config)
    batch = _batch_nuisance(valid, config)
    architecture = _architecture_route(valid, config)

    scope_batch = None
    if not path["active"] and batch["active_without_path_ambiguity_gate"]:
        scope_batch = 1
    applicable_rows = (
        valid
        if scope_batch is None
        else [row for row in valid if int(row["batch_size"]) == scope_batch]
    )
    pooled_applicable = ceiling_for(applicable_rows, config)
    per_model = {
        model_key: ceiling_for(
            [row for row in applicable_rows if row["model_key"] == model_key], config
        )
        for model_key in config["models"]
    }

    if path["active"]:
        route = "path_ambiguity"
    elif architecture["active"]:
        route = "architecture_dependent_floor"
    elif batch["active_without_path_ambiguity_gate"]:
        route = "batch_composition_nuisance"
    else:
        route = "benign_scheduling_floor"

    model_keys = list(config["models"])
    if route == "path_ambiguity":
        model_ceilings = {key: None for key in model_keys}
        pooled_licensed = None
    elif architecture["active"]:
        model_ceilings = {key: per_model[key]["value"] for key in model_keys}
        pooled_licensed = None
    else:
        model_ceilings = {
            key: pooled_applicable["value"] for key in model_keys
        }
        pooled_licensed = pooled_applicable["value"]

    replay_failures = [row["row_id"] for row in rows if row["deterministic_replay"] is not True]
    primal_failures = [
        row["row_id"]
        for row in full_rows
        if row["primal_relative_error"] is None
        or float(row["primal_relative_error"])
        > float(config["router"]["primal_parity_relative_error_ceiling"])
    ]
    return {
        "schema_version": 1,
        "source_evidence_id": config["evidence_id"],
        "row_counts": {
            "all": len(rows),
            "full": len(full_rows),
            "valid_full": len(valid),
            "full_backend_pairs": len(full_pair_ids),
            "nested_op_rows": len(rows) - len(full_rows),
        },
        "pooled_all_batches_measurement": pooled_all,
        "applicable_scope": {
            "batch_size": scope_batch,
            "description": "all frozen batches" if scope_batch is None else "batch size 1 only",
        },
        "pooled_applicable_measurement": pooled_applicable,
        "per_model_applicable_measurement": per_model,
        "prompt_bootstrap_90pct": _prompt_bootstrap(
            valid, config, scope_batch=scope_batch
        ),
        "router": {
            "route": route,
            "path_ambiguity": path,
            "batch_composition_nuisance": batch,
            "architecture_dependent_floor": architecture,
        },
        "licensed_ceilings": {
            "pooled": pooled_licensed,
            "by_model": model_ceilings,
            "olmo_control": model_ceilings.get("olmo3_32b_control"),
            "gemma": model_ceilings.get("gemma4_31b"),
        },
        "audits": {
            "all_rows_deterministically_replayed": not replay_failures,
            "deterministic_replay_failure_count": len(replay_failures),
            "deterministic_replay_failure_examples": replay_failures[:20],
            "full_primal_failure_count": len(primal_failures),
            "full_primal_failure_examples": primal_failures[:20],
        },
    }
