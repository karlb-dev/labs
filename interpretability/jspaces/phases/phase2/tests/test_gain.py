# Conformance test for the final-norm effective gain (CPU, stub modules).
#
# Regression guard for a latent bug found 2026-07-28: the dictionary
# builder read `norm.weight` directly, which is correct for Llama/OLMo/
# Qwen RMSNorm (`x_normed * w`) but WRONG for Gemma (`x_normed * (1+w)`,
# transformers PR #29402). Gemma stores weights near 0, so the mistake is
# silent — it yields a near-zero, sign-scrambled dictionary rather than an
# error. `effective_gain` measures the gain by probing the module instead
# of assuming a convention. Run: python tests/test_gain.py
import sys

import torch

from jspace_part2.dictionaries import effective_gain


def check(name, cond):
    if not cond:
        print(f"FAIL: {name}")
        sys.exit(1)
    print(f"  ok  {name}")


d = 8
w = torch.linspace(-0.4, 0.6, d)


class StdRMSNorm(torch.nn.Module):
    """Llama / OLMo / Qwen convention: x_normed * w."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(w.clone())
        self.eps = 1e-6

    def forward(self, x):
        n = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return n * self.weight


class GemmaRMSNorm(torch.nn.Module):
    """Gemma convention: x_normed * (1 + w)."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(w.clone())
        self.eps = 1e-6

    def forward(self, x):
        n = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return n * (1.0 + self.weight.float())


class Inner(torch.nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm


class FakeModel(torch.nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.model = Inner(norm)


class FakeWrapped(torch.nn.Module):
    """Gemma4-style: the decoder nests under model.language_model."""
    def __init__(self, norm):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = Inner(norm)


print("[1] standard RMSNorm: effective gain == weight")
g = effective_gain(FakeModel(StdRMSNorm()))
check("recovers w", torch.allclose(g, w, atol=1e-4))

print("[2] Gemma RMSNorm: effective gain == 1 + weight")
g = effective_gain(FakeModel(GemmaRMSNorm()))
check("recovers 1+w", torch.allclose(g, 1.0 + w, atol=1e-4))
check("differs from raw weight (the bug this guards)",
      not torch.allclose(g, w, atol=1e-2))

print("[3] the convention change is large, not cosmetic")
ratio = ((1.0 + w).abs() / w.abs().clamp_min(1e-6)).max()
check("raw-weight dictionary would be badly wrong", float(ratio) > 5)
signs_differ = int(((1.0 + w).sign() != w.sign()).sum())
check("some dimensions even flip sign", signs_differ > 0)

print("[4] wrapper-nested norm (Gemma4 layout) is found")
g = effective_gain(FakeWrapped(GemmaRMSNorm()))
check("recovers 1+w through model.language_model", torch.allclose(g, 1.0 + w, atol=1e-4))

print("ALL GAIN TESTS PASS")
