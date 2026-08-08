"""Tiny CPU decoder with RMSNorm-with-gain, mirroring the attrs the study
package touches on real HF wraps (``_lm_head``, ``_final_norm``,
``_logit_softcap``, ``layers``, ``unembed``). Blocks are ``h + 0.1·W h``
so the Jacobian stays well-conditioned."""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(0.8, 1.2, d))

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(var + 1e-6)


class _Block(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.linear.weight.mul_(0.1)

    def forward(self, hidden):
        return hidden + self.linear(hidden)


class _ByteTok:
    bos_token_id = 0

    def __call__(self, text, *, return_tensors="pt", truncation=True,
                 max_length=128, add_special_tokens=True):
        ids = [self.bos_token_id] + [1 + (b % 30) for b in text.encode()][
            : max_length - 1
        ]
        return SimpleNamespace(input_ids=torch.tensor([ids]))

    def decode(self, ids, **_kw):
        return "".join(chr(96 + int(i)) for i in ids)


class TinyRMS(nn.Module):
    def __init__(self, n_layers=6, d_model=8, vocab=32, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.n_layers = n_layers
        self.d_model = d_model
        self.tokenizer = _ByteTok()
        self.embed_tokens = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([_Block(d_model) for _ in range(n_layers)])
        self._final_norm = RMSNorm(d_model)
        self._lm_head = nn.Linear(d_model, vocab, bias=False)
        self._logit_softcap = None
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @property
    def input_device(self):
        return self.embed_tokens.weight.device

    def encode(self, text, *, max_length=128):
        return self.tokenizer(text, max_length=max_length).input_ids

    def forward(self, input_ids):
        hidden = self.embed_tokens(input_ids)
        for block in self.layers:
            hidden = block(hidden)
        return SimpleNamespace(last_hidden_state=hidden)

    def unembed(self, residual):
        logits = self._lm_head(self._final_norm(residual.float()))
        if self._logit_softcap is not None:
            logits = self._logit_softcap * torch.tanh(logits / self._logit_softcap)
        return logits
