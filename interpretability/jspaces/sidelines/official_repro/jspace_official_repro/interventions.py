"""Coordinate swap and steering (COORDINATE_INTERVENTION_CONTRACT).

The swap is the paper's §2.5 two-coordinate pseudoinverse patch::

    V = [v_s, v_t]          # [d_model, 2]
    c = pinv(V) @ h
    h' = h + alpha * V @ (swap(c) - c)

Hooks patch the same residual object ``ActivationRecorder`` reads — the
block's output hidden state — via forward hooks that return the modified
output. Per-(layer, position) fire counters prove the intervention landed
exactly where the frozen plan says.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

#: Geometry gates (contract §4): pairs worse than these are GEOMETRY_GATED.
MAX_CONDITION_NUMBER = 1e6
MAX_ABS_COSINE = 0.999


class GeometryGated(RuntimeError):
    """Raised when a vector pair is too ill-conditioned to swap."""


@dataclass
class SwapDiagnostics:
    cosine: float
    singular_values: tuple[float, float]
    condition_number: float
    coords_before: tuple[float, float]
    coords_after: tuple[float, float]
    patch_norm: float
    residual_norm: float
    coord_reconstruction_error: float
    orth_complement_error: float


def pair_geometry(v_s: torch.Tensor, v_t: torch.Tensor) -> dict:
    v_s = v_s.double()
    v_t = v_t.double()
    cosine = float(
        (v_s @ v_t) / (v_s.norm() * v_t.norm()).clamp_min(1e-30)
    )
    V = torch.stack([v_s, v_t], dim=1)
    svals = torch.linalg.svdvals(V)
    condition = float(svals[0] / svals[-1].clamp_min(1e-30))
    return {
        "cosine": cosine,
        "singular_values": (float(svals[0]), float(svals[1])),
        "condition_number": condition,
        "norm_s": float(v_s.norm()),
        "norm_t": float(v_t.norm()),
        "gated": (condition > MAX_CONDITION_NUMBER or abs(cosine) > MAX_ABS_COSINE),
    }


def swap_coordinates(
    h: torch.Tensor,
    v_s: torch.Tensor,
    v_t: torch.Tensor,
    *,
    alpha: float = 1.0,
    collect: bool = False,
) -> tuple[torch.Tensor, SwapDiagnostics | None]:
    """Apply the two-coordinate swap to residual(s) ``h`` ([..., d]).

    Raises :class:`GeometryGated` for ill-conditioned pairs (never pads or
    regularizes). All math in fp64 for exactness, result cast back.
    """
    original_dtype = h.dtype
    hd = h.double()
    V = torch.stack([v_s.double(), v_t.double()], dim=1)  # [d, 2]
    geometry = pair_geometry(v_s, v_t)
    if geometry["gated"]:
        raise GeometryGated(
            f"condition={geometry['condition_number']:.3g} "
            f"cosine={geometry['cosine']:.6f}"
        )
    pinv = torch.linalg.pinv(V)  # [2, d]
    coords = hd @ pinv.T  # [..., 2]
    swapped = coords.flip(-1)
    delta = (swapped - coords) @ V.T * alpha  # [..., d]
    patched = hd + delta

    diagnostics = None
    if collect:
        coords_after = patched @ pinv.T
        # Orthogonal complement: the patch must live entirely in span(V).
        projector = V @ pinv  # [d, d] projector onto span(V)
        complement_delta = delta - delta @ projector.T
        expected_after = coords + alpha * (swapped - coords)
        diagnostics = SwapDiagnostics(
            cosine=geometry["cosine"],
            singular_values=geometry["singular_values"],
            condition_number=geometry["condition_number"],
            coords_before=tuple(coords.reshape(-1, 2)[0].tolist()),
            coords_after=tuple(coords_after.reshape(-1, 2)[0].tolist()),
            patch_norm=float(delta.norm()),
            residual_norm=float(hd.norm()),
            coord_reconstruction_error=float(
                (coords_after - expected_after).abs().max()
            ),
            orth_complement_error=float(complement_delta.abs().max()),
        )
    return patched.to(original_dtype), diagnostics


def _block_output_tensor(output):
    return output if torch.is_tensor(output) else output[0]


def _replace_block_output(output, tensor):
    if torch.is_tensor(output):
        return tensor
    return (tensor, *output[1:])


@dataclass
class HookPlan:
    """Frozen intervention footprint: which layers, which positions."""

    layers: list[int]
    positions: list[int]

    def expected_fires(self) -> dict[tuple[int, int], int]:
        return {(l, p): 1 for l in self.layers for p in self.positions}


class InterventionSession:
    """Context manager installing per-layer forward hooks.

    ``kind='swap'`` patches each position with the coordinate swap for
    (v_s, v_t) at that layer; ``kind='steer'`` adds
    ``strength * mean_norm[layer] * unit(v_t)``. ``vectors`` maps layer ->
    (v_s, v_t) for swaps or layer -> v_t for steering; vectors are
    layer-specific because the dictionary is.
    """

    def __init__(
        self,
        blocks: nn.ModuleList,
        plan: HookPlan,
        *,
        kind: str,
        vectors: dict[int, tuple[torch.Tensor, torch.Tensor] | torch.Tensor],
        alpha: float = 1.0,
        strength: float = 0.0,
        layer_mean_norms: dict[int, float] | None = None,
        collect_diagnostics: bool = False,
    ) -> None:
        assert kind in ("swap", "steer")
        self._blocks = blocks
        self.plan = plan
        self.kind = kind
        self.vectors = vectors
        self.alpha = alpha
        self.strength = strength
        self.layer_mean_norms = layer_mean_norms or {}
        self.collect = collect_diagnostics
        self.fires: dict[tuple[int, int], int] = {}
        self.diagnostics: dict[tuple[int, int], SwapDiagnostics] = {}
        self.gated_positions: list[tuple[int, int]] = []
        self._handles: list = []

    def _make_hook(self, layer: int):
        def hook(module, inputs, output):
            tensor = _block_output_tensor(output)
            patched = tensor
            for position in self.plan.positions:
                if position >= tensor.shape[1]:
                    raise RuntimeError(
                        f"position {position} outside sequence "
                        f"length {tensor.shape[1]}"
                    )
                if self.kind == "swap":
                    v_s, v_t = self.vectors[layer]
                    try:
                        new_h, diag = swap_coordinates(
                            patched[:, position, :],
                            v_s.to(tensor.device),
                            v_t.to(tensor.device),
                            alpha=self.alpha,
                            collect=self.collect,
                        )
                    except GeometryGated:
                        self.gated_positions.append((layer, position))
                        continue
                    patched = patched.clone() if patched is tensor else patched
                    patched[:, position, :] = new_h
                    if diag is not None:
                        self.diagnostics[(layer, position)] = diag
                else:
                    v_t = self.vectors[layer]
                    unit = (v_t / v_t.norm()).to(tensor.device, tensor.dtype)
                    scale = self.strength * self.layer_mean_norms.get(layer, 1.0)
                    patched = patched.clone() if patched is tensor else patched
                    patched[:, position, :] += scale * unit
                self.fires[(layer, position)] = (
                    self.fires.get((layer, position), 0) + 1
                )
            return _replace_block_output(output, patched)

        return hook

    def __enter__(self):
        try:
            for layer in self.plan.layers:
                self._handles.append(
                    self._blocks[layer].register_forward_hook(
                        self._make_hook(layer)
                    )
                )
        except Exception:
            self._remove()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self._remove()

    def _remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def assert_fires(self, n_forwards: int = 1) -> None:
        expected = {
            key: count * n_forwards
            for key, count in self.plan.expected_fires().items()
        }
        gated = set(self.gated_positions)
        expected = {k: v for k, v in expected.items() if k not in gated}
        observed = dict(self.fires)
        if observed != expected:
            raise RuntimeError(
                f"hook-fire mismatch: expected {len(expected)} cells x "
                f"{n_forwards}, observed {observed} vs {expected}"
            )
