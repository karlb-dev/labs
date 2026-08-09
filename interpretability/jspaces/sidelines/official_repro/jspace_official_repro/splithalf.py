"""Split-half audit, operator layer (OLMO_FIT_CONTRACT §6).

CPU/GPU-agnostic comparisons between half-A and half-B lenses; the
readout-layer views (six evals per half) and the frozen 20-cell
calibration subset run in the OLMo lane driver.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


def _flat_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(
        a.flatten(), b.flatten(), dim=0))


def compare_halves(half_a, half_b, *, top_k: int = 64) -> dict:
    """Operator-layer metrics per shared layer."""
    layers = sorted(set(half_a.source_layers) & set(half_b.source_layers))
    rows = []
    for layer in layers:
        A = half_a.jacobians[layer].float()
        B = half_b.jacobians[layer].float()
        d = A.shape[0]
        eye = torch.eye(d, dtype=A.dtype)
        trace_scale_a = float(A.trace() / d)
        trace_scale_b = float(B.trace() / d)
        sym_rel_frob = float(
            2 * (A - B).norm() / (A.norm() + B.norm())
        )
        # Streamed principal-subspace overlap via low-rank SVD.
        Ua, Sa, _ = torch.svd_lowrank(A, q=top_k)
        Ub, Sb, _ = torch.svd_lowrank(B, q=top_k)
        overlap = float(
            (Ua.T @ Ub).norm() ** 2 / top_k
        )
        rows.append({
            "layer": layer,
            "cosine_raw": _flat_cosine(A, B),
            "cosine_minus_identity": _flat_cosine(A - eye, B - eye),
            "cosine_minus_scaled_identity": _flat_cosine(
                A - trace_scale_a * eye, B - trace_scale_b * eye),
            "sym_rel_frobenius": sym_rel_frob,
            "identity_fraction_a": float(
                (eye * A).sum() / (A.norm() * eye.norm())),
            "identity_fraction_b": float(
                (eye * B).sum() / (B.norm() * eye.norm())),
            "principal_subspace_overlap_top64": overlap,
            "top_singular_a": float(Sa[0]),
            "top_singular_b": float(Sb[0]),
        })
    worst = min(rows, key=lambda r: r["cosine_minus_identity"])
    return {
        "n_layers": len(layers),
        "per_layer": rows,
        "worst_cosine_minus_identity": {
            "layer": worst["layer"],
            "value": worst["cosine_minus_identity"],
        },
        "median_cosine_minus_identity": sorted(
            r["cosine_minus_identity"] for r in rows)[len(rows) // 2],
        "median_sym_rel_frobenius": sorted(
            r["sym_rel_frobenius"] for r in rows)[len(rows) // 2],
    }


def run(fit_dir: Path, out_path: Path) -> dict:
    from jlens.lens import JacobianLens

    half_a = JacobianLens.load(str(fit_dir / "olmo_or1_half_A.pt"))
    half_b = JacobianLens.load(str(fit_dir / "olmo_or1_half_B.pt"))
    result = compare_halves(half_a, half_b)
    result["half_a_n_prompts"] = half_a.n_prompts
    result["half_b_n_prompts"] = half_b.n_prompts
    if out_path.exists():
        raise FileExistsError(out_path)
    out_path.write_text(json.dumps(result))
    return result


if __name__ == "__main__":
    from .paths import DRIVE_ROOT

    result = run(DRIVE_ROOT / "olmo_fit",
                 DRIVE_ROOT / "olmo_fit" / "splithalf_operator_audit.json")
    print(json.dumps({k: v for k, v in result.items() if k != "per_layer"},
                     indent=2))
