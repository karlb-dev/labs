"""Core exact-JVP/secant cell evaluation shared by OLMo and Gemma."""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict

import torch

from .autodiff import exact_jvp, exact_linearize
from .hooks import delivery_audit, patterned_direction, source_mask
from .transport_metrics import (
    additivity_metrics,
    homogeneity_metrics,
    odd_symmetry_defect,
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


def _batched_outputs(suffix, sources: list[torch.Tensor], batch_size: int) -> list[torch.Tensor]:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(sources), batch_size):
            batch = torch.cat(sources[start : start + batch_size], dim=0)
            value = suffix(batch)
            if value.ndim == 1:
                value = value.unsqueeze(0)
            outputs.extend(row.detach().float().cpu() for row in value)
    return outputs


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
    with torch.no_grad():
        clean_target = suffix(clean).detach().float().cpu()
        clean_repeat = suffix(clean).detach().float().cpu()
    clean_repeat_difference = clean_repeat - clean_target

    linearized = exact_linearize(suffix, clean)
    if not torch.allclose(
        linearized.primal.detach().float().cpu(), clean_target, atol=0, rtol=0
    ):
        raise RuntimeError("linearization primal differs from explicit clean suffix")
    reference_pattern = patterned_direction(
        clean, directions[0]["tensor"], mask, 1.0
    )
    fresh = exact_jvp(
        suffix, clean, reference_pattern, backend="torch.func.jvp"
    )
    cached_reference = linearized.apply(reference_pattern)
    backend_parity_relative_error = float(
        (fresh.tangent - cached_reference).norm()
        / fresh.tangent.norm().clamp_min(1e-30)
    )
    if backend_parity_relative_error > 1e-5:
        raise RuntimeError(
            "cached exact linearization disagrees with fresh torch.func.jvp: "
            f"{backend_parity_relative_error}"
        )

    requests: list[tuple[tuple, torch.Tensor]] = []
    audits: dict[tuple, dict] = {}
    tangent_predictions: dict[tuple, torch.Tensor] = {}
    desired_patterns: dict[tuple, torch.Tensor] = {}
    for direction in directions:
        for epsilon in epsilon_ladder:
            key = (direction["id"], float(epsilon))
            desired = patterned_direction(
                clean, direction["tensor"], mask, float(epsilon)
            )
            desired_patterns[key] = desired
            sign_audits = {}
            realized_positive = None
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
                if label == "positive":
                    realized_positive = realized
                requests.append(((direction["id"], epsilon, label), clean + perturbation))
            audits[key] = sign_audits
            if all(value["faithful"] for value in sign_audits.values()):
                tangent_predictions[key] = (
                    linearized.apply(realized_positive).detach().float().cpu()
                )

    pair_ids = (directions[0]["id"], directions[1]["id"])
    pair_audits = {}
    for epsilon in epsilon_ladder:
        left = desired_patterns[(pair_ids[0], float(epsilon))]
        right = desired_patterns[(pair_ids[1], float(epsilon))]
        combined = left + right
        _, audit = delivery_audit(
            clean,
            combined,
            model_dtype=suffix.clean_source.dtype,
            selected_mask=mask,
            cosine_floor=delivery_cosine_floor,
            relative_norm_error_ceiling=delivery_norm_error_ceiling,
        )
        pair_audits[float(epsilon)] = asdict(audit)
        requests.append((("pair", epsilon, "sum"), clean + combined))

    values = _batched_outputs(suffix, [source for _, source in requests], batch_size)
    responses = {
        key: value - clean_target for (key, _), value in zip(requests, values, strict=True)
    }
    rows = []
    raw_records = []
    noise_norm = float(clean_repeat_difference.norm())
    for direction in directions:
        for epsilon in epsilon_ladder:
            key = (direction["id"], float(epsilon))
            positive = responses[(direction["id"], epsilon, "positive")]
            negative = responses[(direction["id"], epsilon, "negative")]
            double = responses[(direction["id"], epsilon, "double")]
            prediction = tangent_predictions.get(key)
            faithful = prediction is not None
            tangent = (
                tangent_metrics(positive, prediction) if faithful else {
                    "response_norm": float(positive.norm()),
                    "tangent_prediction_norm": None,
                    "tangent_cosine": None,
                    "gain": None,
                    "tangent_relative_error": None,
                }
            )
            central = (positive - negative) / 2
            central_metrics = (
                tangent_metrics(central, prediction) if faithful else {
                    "tangent_cosine": None,
                    "gain": None,
                    "tangent_relative_error": None,
                }
            )
            homogeneous = homogeneity_metrics(positive, double)
            odd = odd_symmetry_defect(positive, negative)
            additive = {"additivity_defect": None, "additivity_cosine": None}
            if direction["id"] in pair_ids:
                summed = responses[("pair", epsilon, "sum")]
                left = responses[(pair_ids[0], epsilon, "positive")]
                right = responses[(pair_ids[1], epsilon, "positive")]
                additive = additivity_metrics(summed, left, right)
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
                "response_snr": float(positive.norm()) / max(noise_norm, 1e-12),
                "clean_repeat_noise_norm": noise_norm,
                "exact_jvp_backend": "torch.func.linearize validated against torch.func.jvp",
                "backend_parity_relative_error": backend_parity_relative_error,
                **tangent,
                "central_tangent_cosine": central_metrics["tangent_cosine"],
                "central_gain": central_metrics["gain"],
                "central_tangent_relative_error": central_metrics["tangent_relative_error"],
                **homogeneous,
                "odd_symmetry_defect": odd,
                **additive,
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
                    "tangent_prediction": prediction,
                }
            )
    raw = {
        "schema_version": 1,
        "metadata": metadata,
        "cell_id": cell_id,
        "perturbation_mode": perturbation_mode,
        "clean_target": clean_target,
        "clean_repeat": clean_repeat,
        "clean_source_selected": clean.detach().float().cpu()[mask.cpu()],
        "directions": [
            {**{key: value for key, value in row.items() if key != "tensor"}, "tensor": row["tensor"].cpu()}
            for row in directions
        ],
        "records": raw_records,
        "backend_parity_relative_error": backend_parity_relative_error,
    }
    return rows, raw
