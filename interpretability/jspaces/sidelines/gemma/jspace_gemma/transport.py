"""Core exact-JVP/secant cell evaluation shared by OLMo and Gemma."""
from __future__ import annotations

import hashlib
from dataclasses import asdict

import torch

from .autodiff import exact_jvp
from .hooks import delivery_audit, patterned_direction, source_mask
from .transport_metrics import (
    adjusted_additivity_metrics,
    adjusted_homogeneity_metrics,
    adjusted_odd_symmetry_metrics,
    additivity_metrics,
    homogeneity_metrics,
    odd_symmetry_defect,
    quantization_aware_response_snr,
    quantization_floor_norm,
    tangent_metrics,
)


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def stable_seed(base: int, *parts: object) -> int:
    material = "|".join([str(base), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big") % (2**31)


def _random(width: int, *, seed: int, kind: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if kind == "rademacher":
        value = torch.randint(0, 2, (width,), generator=generator, dtype=torch.int64)
        return value.float().mul_(2).sub_(1)
    if kind == "gaussian":
        return torch.randn(width, generator=generator, dtype=torch.float32)
    raise ValueError(kind)


def make_directions(
    clean_source: torch.Tensor,
    mask: torch.Tensor,
    specs: list[dict],
    *,
    base_seed: int,
    cell_id: str,
) -> list[dict]:
    selected = clean_source.float()[mask]
    radial = selected.mean(dim=0)
    radial = radial / radial.norm().clamp_min(1e-30)
    rows = []
    for index, spec in enumerate(specs):
        kind = spec["type"]
        seed = stable_seed(base_seed, cell_id, spec["id"])
        if kind == "radial":
            value = radial.clone()
        elif kind in {"rademacher", "gaussian"}:
            value = _random(clean_source.shape[-1], seed=seed, kind=kind)
            # The first two fixed random directions form the orthogonalized
            # additivity pair; this is frozen before model outcomes.
            if index == 1 and rows:
                first = rows[0]["tensor"].cpu()
                value = value - torch.dot(value, first) * first
            value = value / value.norm().clamp_min(1e-30)
        elif kind == "sphere_tangent":
            value = _random(clean_source.shape[-1], seed=seed, kind="gaussian")
            value = value - torch.dot(value, radial.cpu()) * radial.cpu()
            value = value / value.norm().clamp_min(1e-30)
        else:
            raise ValueError(f"unsupported frozen Stage-1 direction {kind!r}")
        value = value.to(clean_source.device, torch.float32)
        rows.append(
            {
                "type": kind,
                "id": spec["id"],
                "seed": seed,
                "sha256": tensor_sha256(value),
                "tensor": value,
            }
        )
    return rows


def _as_batch(value: torch.Tensor, count: int) -> torch.Tensor:
    if count == 1 and value.ndim == 1:
        return value.unsqueeze(0)
    if value.ndim < 2 or value.shape[0] != count:
        raise RuntimeError(
            f"suffix output does not preserve batch: count={count}, shape={tuple(value.shape)}"
        )
    return value


def _batched_responses_and_jvps(
    suffix,
    clean: torch.Tensor,
    requests: list[tuple[tuple, torch.Tensor, torch.Tensor]],
    batch_size: int,
) -> dict:
    """Evaluate each secant and exact JVP in the identical batch slot.

    A separate all-clean forward with the same batch shape supplies each
    finite-response baseline. This prevents a change in GEMM batch shape from
    masquerading as a small secant. The exact JVP uses the same batched primal
    and the separately realized post-cast tangent for every request.
    """
    if batch_size < 1:
        raise ValueError("transport batch size must be positive")
    responses = {}
    tangents = {}
    clean_targets = {}
    clean_repeat_differences = {}
    primal_parity = {}
    backends = {}
    batch_diagnostics = []
    for batch_index, start in enumerate(range(0, len(requests), batch_size)):
        chunk = requests[start : start + batch_size]
        count = len(chunk)
        clean_batch = clean.expand(count, *clean.shape[1:]).clone()
        source_batch = torch.cat([source for _, source, _ in chunk], dim=0)
        tangent_batch = torch.cat([tangent for _, _, tangent in chunk], dim=0)
        with torch.no_grad():
            baseline = _as_batch(suffix(clean_batch), count).detach().float().cpu()
            repeated = _as_batch(suffix(clean_batch), count).detach().float().cpu()
            finite = _as_batch(suffix(source_batch), count).detach().float().cpu()
        exact = exact_jvp(suffix, clean_batch, tangent_batch, backend="auto")
        exact_primal = _as_batch(exact.primal, count).detach().float().cpu()
        exact_tangent = _as_batch(exact.tangent, count).detach().float().cpu()
        difference = exact_primal - baseline
        max_relative = 0.0
        for offset, (key, _, _) in enumerate(chunk):
            relative = float(
                difference[offset].norm()
                / baseline[offset].norm().clamp_min(1e-30)
            )
            if relative > 1e-5:
                raise RuntimeError(
                    "exact-JVP primal differs from identical-batch clean suffix: "
                    f"{key}: {relative}"
                )
            max_relative = max(max_relative, relative)
            responses[key] = finite[offset] - baseline[offset]
            tangents[key] = exact_tangent[offset]
            clean_targets[key] = baseline[offset]
            clean_repeat_differences[key] = repeated[offset] - baseline[offset]
            primal_parity[key] = relative
            backends[key] = exact.backend
        clean_spread = (
            float((baseline - baseline[:1]).norm()) if count > 1 else 0.0
        )
        batch_diagnostics.append(
            {
                "batch_index": batch_index,
                "request_count": count,
                "request_keys": [list(key) for key, _, _ in chunk],
                "exact_jvp_backend": exact.backend,
                "max_primal_relative_error": max_relative,
                "clean_repeat_difference_norm": float((repeated - baseline).norm()),
                "within_batch_clean_spread_norm": clean_spread,
            }
        )
    return {
        "responses": responses,
        "tangents": tangents,
        "clean_targets": clean_targets,
        "clean_repeat_differences": clean_repeat_differences,
        "primal_parity": primal_parity,
        "backends": backends,
        "batch_diagnostics": batch_diagnostics,
    }


def evaluate_transport_cell(
    suffix,
    *,
    attention_mask: torch.Tensor,
    perturbation_mode: str,
    direction_specs: list[dict],
    epsilon_ladder: list[float],
    seed: int,
    cell_id: str,
    metadata: dict,
    delivery_cosine_floor: float,
    delivery_norm_error_ceiling: float,
    batch_size: int,
) -> tuple[list[dict], dict]:
    clean = suffix.clean_source.float()
    mask = source_mask(attention_mask, mode=perturbation_mode, position=-1)
    directions = make_directions(
        clean, mask, direction_specs, base_seed=seed, cell_id=cell_id
    )
    requests: list[tuple[tuple, torch.Tensor, torch.Tensor]] = []
    audits: dict[tuple, dict] = {}
    realized_patterns: dict[tuple, torch.Tensor] = {}
    desired_patterns: dict[tuple, torch.Tensor] = {}
    for direction in directions:
        for epsilon in epsilon_ladder:
            key = (direction["id"], float(epsilon))
            desired = patterned_direction(
                clean, direction["tensor"], mask, float(epsilon)
            )
            desired_patterns[key] = desired
            sign_audits = {}
            for label, multiplier in (("positive", 1.0), ("negative", -1.0), ("double", 2.0)):
                perturbation = desired * multiplier
                realized, audit = delivery_audit(
                    clean,
                    perturbation,
                    model_dtype=suffix.clean_source.dtype,
                    selected_mask=mask,
                    cosine_floor=delivery_cosine_floor,
                    relative_norm_error_ceiling=delivery_norm_error_ceiling,
                )
                sign_audits[label] = asdict(audit)
                request_key = (direction["id"], float(epsilon), label)
                realized_patterns[request_key] = realized
                requests.append((request_key, clean + perturbation, realized))
            audits[key] = sign_audits

    pair_ids = (directions[0]["id"], directions[1]["id"])
    pair_audits = {}
    for epsilon in epsilon_ladder:
        left = desired_patterns[(pair_ids[0], float(epsilon))]
        right = desired_patterns[(pair_ids[1], float(epsilon))]
        combined = left + right
        realized, audit = delivery_audit(
            clean,
            combined,
            model_dtype=suffix.clean_source.dtype,
            selected_mask=mask,
            cosine_floor=delivery_cosine_floor,
            relative_norm_error_ceiling=delivery_norm_error_ceiling,
        )
        pair_audits[float(epsilon)] = asdict(audit)
        request_key = ("pair", float(epsilon), "sum")
        realized_patterns[request_key] = realized
        requests.append((request_key, clean + combined, realized))

    evaluated = _batched_responses_and_jvps(suffix, clean, requests, batch_size)
    responses = evaluated["responses"]
    tangent_predictions = evaluated["tangents"]
    clean_targets = evaluated["clean_targets"]
    clean_repeat_differences = evaluated["clean_repeat_differences"]
    rows = []
    raw_records = []
    for direction in directions:
        for epsilon in epsilon_ladder:
            key = (direction["id"], float(epsilon))
            positive_key = (direction["id"], float(epsilon), "positive")
            negative_key = (direction["id"], float(epsilon), "negative")
            double_key = (direction["id"], float(epsilon), "double")
            positive = responses[positive_key]
            negative = responses[negative_key]
            double = responses[double_key]
            tangent_positive = tangent_predictions[positive_key]
            tangent_negative = tangent_predictions[negative_key]
            tangent_double = tangent_predictions[double_key]
            faithful = all(value["faithful"] for value in audits[key].values())
            tangent = (
                tangent_metrics(positive, tangent_positive) if faithful else {
                    "response_norm": float(positive.norm()),
                    "tangent_prediction_norm": float(tangent_positive.norm()),
                    "tangent_cosine": None,
                    "gain": None,
                    "tangent_relative_error": None,
                }
            )
            central = (positive - negative) / 2
            central_prediction = (tangent_positive - tangent_negative) / 2
            central_metrics = (
                tangent_metrics(central, central_prediction) if faithful else {
                    "tangent_cosine": None,
                    "gain": None,
                    "tangent_relative_error": None,
                }
            )
            homogeneous = homogeneity_metrics(positive, double)
            homogeneous_adjusted = adjusted_homogeneity_metrics(
                positive, double, tangent_positive, tangent_double
            )
            odd = odd_symmetry_defect(positive, negative)
            odd_adjusted = adjusted_odd_symmetry_metrics(
                positive, negative, tangent_positive, tangent_negative
            )
            input_homogeneous = homogeneity_metrics(
                realized_patterns[positive_key], realized_patterns[double_key]
            )
            input_odd = odd_symmetry_defect(
                realized_patterns[positive_key], realized_patterns[negative_key]
            )
            additive = {
                "additivity_defect": None,
                "additivity_cosine": None,
                "additivity_first_order_delivery_defect": None,
                "additivity_nonlinear_remainder_defect": None,
            }
            input_additivity_defect = None
            additivity_faithful = None
            if direction["id"] in pair_ids:
                sum_key = ("pair", float(epsilon), "sum")
                left_key = (pair_ids[0], float(epsilon), "positive")
                right_key = (pair_ids[1], float(epsilon), "positive")
                summed = responses[sum_key]
                left = responses[left_key]
                right = responses[right_key]
                additive = additivity_metrics(summed, left, right)
                additive.update(
                    adjusted_additivity_metrics(
                        summed,
                        left,
                        right,
                        tangent_predictions[sum_key],
                        tangent_predictions[left_key],
                        tangent_predictions[right_key],
                    )
                )
                input_additivity_defect = additivity_metrics(
                    realized_patterns[sum_key],
                    realized_patterns[left_key],
                    realized_patterns[right_key],
                )["additivity_defect"]
                additivity_faithful = bool(
                    pair_audits[float(epsilon)]["faithful"]
                    and audits[(pair_ids[0], float(epsilon))]["positive"]["faithful"]
                    and audits[(pair_ids[1], float(epsilon))]["positive"]["faithful"]
                )
            quantization_floor = quantization_floor_norm(
                clean_targets[positive_key], suffix.clean_source.dtype
            )
            snr = quantization_aware_response_snr(
                positive,
                tangent_positive,
                clean_repeat_differences[positive_key],
                quantization_floor,
            )
            plus_audit = audits[key]["positive"]
            row = {
                **metadata,
                "cell_id": cell_id,
                "perturbation_mode": perturbation_mode,
                "direction_type": direction["type"],
                "direction_id": direction["id"],
                "direction_seed": direction["seed"],
                "direction_sha256": direction["sha256"],
                "desired_relative_epsilon": float(epsilon),
                "desired_norm": plus_audit["desired_norm"],
                "realized_norm": plus_audit["realized_norm"],
                "input_fidelity_cosine": plus_audit["cosine"],
                "input_relative_norm_error": plus_audit["relative_norm_error"],
                "negative_delivery": audits[key]["negative"],
                "double_delivery": audits[key]["double"],
                "faithful_delivery": faithful,
                **snr,
                "response_snr_definition": (
                    "min(secant_norm, exact_jvp_norm) / "
                    "max(in_batch_clean_repeat_norm, target_dtype_half_step_norm)"
                ),
                "exact_jvp_backend": evaluated["backends"][positive_key],
                "backend_parity_relative_error": evaluated["primal_parity"][positive_key],
                **tangent,
                "central_tangent_cosine": central_metrics["tangent_cosine"],
                "central_gain": central_metrics["gain"],
                "central_tangent_relative_error": central_metrics["tangent_relative_error"],
                **homogeneous,
                **homogeneous_adjusted,
                "odd_symmetry_defect": odd,
                **odd_adjusted,
                "input_homogeneity_defect": input_homogeneous["homogeneity_defect"],
                "input_odd_symmetry_defect": input_odd,
                "input_additivity_defect": input_additivity_defect,
                **additive,
                "additivity_faithful_delivery": additivity_faithful,
                "additivity_pair_ids": list(pair_ids),
                "pair_delivery": pair_audits[float(epsilon)],
            }
            rows.append(row)
            raw_records.append(
                {
                    "direction_id": direction["id"],
                    "epsilon": float(epsilon),
                    "positive": positive,
                    "negative": negative,
                    "double": double,
                    "tangent_positive": tangent_positive,
                    "tangent_negative": tangent_negative,
                    "tangent_double": tangent_double,
                    "central_tangent_prediction": central_prediction,
                    "finite_clean_target": clean_targets[positive_key],
                    "clean_repeat_difference": clean_repeat_differences[positive_key],
                    "realized_positive_sha256": tensor_sha256(realized_patterns[positive_key]),
                    "realized_negative_sha256": tensor_sha256(realized_patterns[negative_key]),
                    "realized_double_sha256": tensor_sha256(realized_patterns[double_key]),
                }
            )
    first_key = requests[0][0]
    raw = {
        "schema_version": 2,
        "metadata": metadata,
        "cell_id": cell_id,
        "perturbation_mode": perturbation_mode,
        "clean_target_reference": clean_targets[first_key],
        "clean_repeat_difference_reference": clean_repeat_differences[first_key],
        "clean_source_selected": clean.detach().float().cpu()[mask.cpu()],
        "directions": [
            {**{key: value for key, value in row.items() if key != "tensor"}, "tensor": row["tensor"].cpu()}
            for row in directions
        ],
        "records": raw_records,
        "pair_records": [
            {
                "epsilon": float(epsilon),
                "sum_response": responses[("pair", float(epsilon), "sum")],
                "sum_tangent": tangent_predictions[("pair", float(epsilon), "sum")],
                "realized_sum_sha256": tensor_sha256(
                    realized_patterns[("pair", float(epsilon), "sum")]
                ),
            }
            for epsilon in epsilon_ladder
        ],
        "exact_batch_diagnostics": evaluated["batch_diagnostics"],
        "max_backend_parity_relative_error": max(evaluated["primal_parity"].values()),
        "response_snr_definition": (
            "min(secant_norm, exact_jvp_norm) / "
            "max(in_batch_clean_repeat_norm, target_dtype_half_step_norm)"
        ),
    }
    return rows, raw
