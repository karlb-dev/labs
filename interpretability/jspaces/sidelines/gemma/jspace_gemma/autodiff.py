"""Exact directional autodiff backends. No finite-difference fallback exists."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


class ExactJVPError(RuntimeError):
    pass


@dataclass(frozen=True)
class JVPResult:
    primal: torch.Tensor
    tangent: torch.Tensor
    backend: str


@dataclass(frozen=True)
class LinearizationResult:
    primal: torch.Tensor
    apply: Callable[[torch.Tensor], torch.Tensor]
    backend: str


def exact_jvp(
    function: Callable[[torch.Tensor], torch.Tensor],
    primal: torch.Tensor,
    tangent: torch.Tensor,
    *,
    backend: str = "auto",
) -> JVPResult:
    """Compute an autodiff JVP; never substitutes a numerical secant."""
    if primal.shape != tangent.shape:
        raise ValueError("primal and tangent must have identical shapes")
    errors = []
    requested = (
        ("torch.func.jvp", "torch.autograd.functional.jvp")
        if backend == "auto" else (backend,)
    )
    for name in requested:
        try:
            if name == "torch.func.jvp":
                output, derivative = torch.func.jvp(
                    function, (primal,), (tangent,), strict=True
                )
            elif name == "torch.autograd.functional.jvp":
                output, derivative = torch.autograd.functional.jvp(
                    function,
                    primal,
                    tangent,
                    create_graph=False,
                    strict=True,
                )
            else:
                raise ValueError(f"unknown exact-JVP backend {name!r}")
            if output.shape != derivative.shape:
                raise ExactJVPError("JVP output and tangent shape mismatch")
            if not torch.isfinite(output).all() or not torch.isfinite(derivative).all():
                raise ExactJVPError("non-finite primal or exact JVP")
            return JVPResult(output, derivative, name)
        except Exception as exc:  # backend ladder must preserve diagnostics
            errors.append({"backend": name, "type": type(exc).__name__, "message": str(exc)})
    raise ExactJVPError(
        "all exact autodiff backends failed; finite differences are forbidden "
        f"as a replacement: {errors}"
    )


def exact_linearize(
    function: Callable[[torch.Tensor], torch.Tensor],
    primal: torch.Tensor,
) -> LinearizationResult:
    """Cache an exact forward-mode linear map for many realized tangents."""
    try:
        output, jvp_function = torch.func.linearize(function, primal)
    except Exception as exc:
        raise ExactJVPError(
            "torch.func.linearize failed; this is not permission to use a "
            f"finite difference as exact: {type(exc).__name__}: {exc}"
        ) from exc
    if not torch.isfinite(output).all():
        raise ExactJVPError("non-finite primal from torch.func.linearize")

    def checked(tangent: torch.Tensor) -> torch.Tensor:
        if tangent.shape != primal.shape:
            raise ValueError("linearized tangent shape mismatch")
        result = jvp_function(tangent)
        if result.shape != output.shape or not torch.isfinite(result).all():
            raise ExactJVPError("invalid tangent from torch.func.linearize")
        return result

    return LinearizationResult(output, checked, "torch.func.linearize")


def secant(
    function: Callable[[torch.Tensor], torch.Tensor],
    primal: torch.Tensor,
    perturbation: torch.Tensor,
    *,
    side: str = "forward",
) -> torch.Tensor:
    """A named finite response/secant used only to validate exact JVPs."""
    with torch.no_grad():
        base = function(primal)
        if side == "forward":
            return function(primal + perturbation) - base
        if side == "central":
            return (function(primal + perturbation) - function(primal - perturbation)) / 2
    raise ValueError(f"unknown secant side {side!r}")
