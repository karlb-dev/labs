import torch
from torch import nn

from jspace_gemma.autodiff import exact_jvp


class TinyDifferentiableSuffix(nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(19)
        self.q = nn.Parameter(torch.randn(6, 6, generator=generator, dtype=torch.float64) / 4)
        self.k = nn.Parameter(torch.randn(6, 6, generator=generator, dtype=torch.float64) / 4)
        self.v = nn.Parameter(torch.randn(6, 6, generator=generator, dtype=torch.float64) / 4)
        self.o = nn.Parameter(torch.randn(6, 6, generator=generator, dtype=torch.float64) / 4)
        self.gate = nn.Parameter(torch.randn(6, 10, generator=generator, dtype=torch.float64) / 4)
        self.up = nn.Parameter(torch.randn(6, 10, generator=generator, dtype=torch.float64) / 4)
        self.down = nn.Parameter(torch.randn(10, 6, generator=generator, dtype=torch.float64) / 4)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def norm(x):
        return x / (x.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()

    def forward(self, source):
        normalized = self.norm(source)
        q = normalized @ self.q
        k = normalized @ self.k
        v = normalized @ self.v
        scores = q @ k.transpose(-1, -2) / (source.shape[-1] ** 0.5)
        causal = torch.triu(
            torch.full_like(scores, float("-inf")), diagonal=1
        )
        attention = torch.softmax(scores + causal, dim=-1)
        hidden = source + (attention @ v) @ self.o
        normalized = self.norm(hidden)
        mlp = (torch.nn.functional.gelu(normalized @ self.gate) * (normalized @ self.up)) @ self.down
        return (hidden + mlp)[-1]


def test_tiny_transformer_forward_and_reverse_directional_derivatives_agree():
    generator = torch.Generator().manual_seed(23)
    source = torch.randn(5, 6, generator=generator, dtype=torch.float64)
    direction = torch.randn(5, 6, generator=generator, dtype=torch.float64)
    module = TinyDifferentiableSuffix()
    forward = exact_jvp(module, source, direction, backend="torch.func.jvp")
    fallback = exact_jvp(
        module, source, direction, backend="torch.autograd.functional.jvp"
    )
    jacobian = torch.autograd.functional.jacobian(module, source, strict=True)
    independent = torch.tensordot(jacobian, direction, dims=([1, 2], [0, 1]))
    assert torch.allclose(forward.tangent, fallback.tangent, atol=1e-10, rtol=1e-10)
    assert torch.allclose(forward.tangent, independent, atol=1e-10, rtol=1e-10)


def test_tiny_transformer_secant_error_shrinks_quadratically():
    generator = torch.Generator().manual_seed(29)
    source = torch.randn(5, 6, generator=generator, dtype=torch.float64)
    direction = torch.randn(5, 6, generator=generator, dtype=torch.float64)
    direction /= direction.norm()
    module = TinyDifferentiableSuffix()
    exact = exact_jvp(module, source, direction).tangent
    errors = []
    for epsilon in (1e-2, 5e-3, 2.5e-3):
        central = (module(source + epsilon * direction) - module(source - epsilon * direction)) / (2 * epsilon)
        errors.append(float((central - exact).norm()))
    assert errors[2] < errors[1] < errors[0]
