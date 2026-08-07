"""Independent analytic and tiny-transformer exact-JVP evidence producer."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch
from torch import nn

from jspace_gemma.autodiff import exact_jvp
from jspace_gemma.manifests import (
    atomic_json,
    environment_payload,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from jspace_gemma.paths import directory
from jspace_gemma.registry import create


class TinySuffix(nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(19)
        for name, shape in {
            "q": (6, 6), "k": (6, 6), "v": (6, 6), "o": (6, 6),
            "gate": (6, 10), "up": (6, 10), "down": (10, 6),
        }.items():
            setattr(
                self,
                name,
                nn.Parameter(torch.randn(*shape, generator=generator, dtype=torch.float64) / 4),
            )
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def norm(value):
        return value / (value.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()

    def forward(self, source):
        normalized = self.norm(source)
        q, k, v = normalized @ self.q, normalized @ self.k, normalized @ self.v
        scores = q @ k.transpose(-1, -2) / source.shape[-1] ** 0.5
        causal = torch.triu(torch.full_like(scores, float("-inf")), diagonal=1)
        hidden = source + (torch.softmax(scores + causal, dim=-1) @ v) @ self.o
        normalized = self.norm(hidden)
        mlp = (
            torch.nn.functional.gelu(normalized @ self.gate)
            * (normalized @ self.up)
        ) @ self.down
        return (hidden + mlp)[-1]


def _analytic() -> dict:
    source = torch.tensor([0.4, -0.7], dtype=torch.float64)
    direction = torch.tensor([0.6, 0.8], dtype=torch.float64)

    def function(value):
        return torch.stack(
            [value[0] ** 3 + value[1], torch.sin(value[1]) + value[0] * value[1]]
        )

    expected = torch.stack(
        [
            3 * source[0] ** 2 * direction[0] + direction[1],
            direction[1] * torch.cos(source[1])
            + direction[0] * source[1]
            + source[0] * direction[1],
        ]
    )
    rows = []
    for backend in ("torch.func.jvp", "torch.autograd.functional.jvp"):
        result = exact_jvp(function, source, direction, backend=backend)
        rows.append(
            {
                "backend": backend,
                "jvp": result.tangent.tolist(),
                "analytic": expected.tolist(),
                "absolute_error": float((result.tangent - expected).norm()),
            }
        )
    return {"rows": rows, "pass": all(row["absolute_error"] < 1e-12 for row in rows)}


def _tiny_transformer() -> dict:
    generator = torch.Generator().manual_seed(23)
    source = torch.randn(5, 6, generator=generator, dtype=torch.float64)
    direction = torch.randn(5, 6, generator=generator, dtype=torch.float64)
    direction /= direction.norm()
    model = TinySuffix()
    forward = exact_jvp(model, source, direction, backend="torch.func.jvp")
    fallback = exact_jvp(model, source, direction, backend="torch.autograd.functional.jvp")
    jacobian = torch.autograd.functional.jacobian(model, source, strict=True)
    reverse = torch.tensordot(jacobian, direction, dims=([1, 2], [0, 1]))
    secants = []
    for epsilon in (1e-2, 5e-3, 2.5e-3, 1.25e-3):
        central = (
            model(source + epsilon * direction) - model(source - epsilon * direction)
        ) / (2 * epsilon)
        secants.append(
            {"epsilon": epsilon, "error_to_exact": float((central - forward.tangent).norm())}
        )
    forward_fallback = float((forward.tangent - fallback.tangent).norm())
    forward_reverse = float((forward.tangent - reverse).norm())
    return {
        "forward_fallback_error": forward_fallback,
        "forward_reverse_error": forward_reverse,
        "central_secants": secants,
        "pass": (
            forward_fallback < 1e-10
            and forward_reverse < 1e-10
            and all(
                secants[index + 1]["error_to_exact"] < secants[index]["error_to_exact"]
                for index in range(len(secants) - 1)
            )
        ),
    }


def main() -> None:
    git = require_clean_tree()
    analytic = _analytic()
    tiny = _tiny_transformer()
    if not analytic["pass"] or not tiny["pass"]:
        raise RuntimeError(f"exact-JVP golden failure: analytic={analytic}, tiny={tiny}")
    implementation = Path(inspect.getsourcefile(exact_jvp))
    payload = {
        "schema_version": 1,
        "evidence_id": "gm-jvp-goldens-v1",
        "tier": "methods",
        "analytic": analytic,
        "tiny_differentiable_transformer": tiny,
        "implementation": {
            "path": str(implementation),
            "sha256": file_sha256(implementation),
            "finite_difference_exact_fallback": False,
        },
        "environment": environment_payload(),
        "code_commit": git["code_commit"],
    }
    payload["payload_sha256_self_excluded"] = object_sha256(payload)
    output = directory("metrics") / "jvp_goldens" / "gm_jvp_goldens_v1.json"
    atomic_json(output, payload)
    create(
        "gm-jvp-goldens-v1",
        tier="methods",
        what=(
            "analytic polynomial and tiny differentiable attention/gated-MLP "
            "goldens: forward JVP, autograd fallback, reverse Jacobian-vector, "
            "and shrinking central secants agree"
        ),
        command="python -m jspace_gemma.experiments.gm_jvp_goldens",
        outputs=[output],
        inputs={"implementation_sha256": file_sha256(implementation)},
    )
    print(json.dumps({"output": str(output), "sha256": file_sha256(output)}, indent=1))


if __name__ == "__main__":
    main()
