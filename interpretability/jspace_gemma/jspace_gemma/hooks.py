"""Explicit decoder suffixes and fp32 perturbation-delivery contracts."""
from __future__ import annotations

from collections import UserDict
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F

from .architecture import ArchitectureError, decoder_components


@dataclass(frozen=True)
class TargetSpec:
    representation: str
    target_layer: int | None = None
    position_indices: tuple[int, ...] = (-1,)
    position_reduction: str = "flatten"
    selected_token_ids: tuple[int, ...] = ()

    def __post_init__(self):
        valid = {
            "block_residual", "final_residual", "normalized_final_residual",
            "pre_softcap_logits", "post_softcap_logits",
        }
        if self.representation not in valid:
            raise ValueError(f"unknown target representation {self.representation!r}")
        if self.representation == "block_residual" and self.target_layer is None:
            raise ValueError("block_residual requires target_layer")
        if "logits" in self.representation and not self.selected_token_ids:
            raise ValueError("logit targets require a fixed nonempty token subset")
        if self.position_reduction not in {"flatten", "sum", "mean"}:
            raise ValueError("invalid position reduction")


@dataclass(frozen=True)
class DeliveryAudit:
    desired_norm: float
    realized_norm: float
    cosine: float
    relative_norm_error: float
    faithful: bool


def delivery_audit(
    clean_source: torch.Tensor,
    desired_perturbation: torch.Tensor,
    *,
    model_dtype: torch.dtype,
    selected_mask: torch.Tensor,
    cosine_floor: float = 0.999,
    relative_norm_error_ceiling: float = 0.01,
) -> tuple[torch.Tensor, DeliveryAudit]:
    if clean_source.shape != desired_perturbation.shape:
        raise ValueError("source and desired perturbation shapes differ")
    if selected_mask.shape != clean_source.shape[:-1]:
        raise ValueError("selected mask does not match source positions")
    realized_source = (clean_source.float() + desired_perturbation.float()).to(model_dtype)
    baseline = clean_source.to(model_dtype)
    realized = realized_source.float() - baseline.float()
    desired_selected = desired_perturbation.float()[selected_mask].reshape(-1)
    realized_selected = realized[selected_mask].reshape(-1)
    desired_norm = float(desired_selected.norm())
    realized_norm = float(realized_selected.norm())
    if desired_norm == 0 or realized_norm == 0:
        cosine = 0.0
        relative_error = float("inf")
    else:
        cosine = float(F.cosine_similarity(desired_selected, realized_selected, dim=0))
        relative_error = abs(realized_norm - desired_norm) / desired_norm
    audit = DeliveryAudit(
        desired_norm=desired_norm,
        realized_norm=realized_norm,
        cosine=cosine,
        relative_norm_error=relative_error,
        faithful=(
            cosine >= cosine_floor
            and relative_error <= relative_norm_error_ceiling
        ),
    )
    return realized, audit


def source_mask(
    attention_mask: torch.Tensor,
    *,
    mode: str,
    position: int = -1,
) -> torch.Tensor:
    valid = attention_mask.bool()
    if mode == "uniform_valid":
        return valid
    if mode != "single_position":
        raise ValueError(f"unknown source perturbation mode {mode!r}")
    result = torch.zeros_like(valid)
    for batch in range(valid.shape[0]):
        indices = torch.nonzero(valid[batch], as_tuple=False).flatten()
        if not len(indices):
            raise ValueError("attention mask contains an empty sequence")
        selected = indices[position]
        result[batch, selected] = True
    return result


def patterned_direction(
    clean_source: torch.Tensor,
    direction: torch.Tensor,
    mask: torch.Tensor,
    relative_epsilon: float,
) -> torch.Tensor:
    if direction.ndim != 1 or direction.numel() != clean_source.shape[-1]:
        raise ValueError("direction does not match residual width")
    unit = direction.float() / direction.float().norm().clamp_min(1e-30)
    reference = clean_source.float()[mask].norm(dim=-1).mean()
    return mask.unsqueeze(-1).float() * unit * (float(relative_epsilon) * reference)


class ExplicitDecoderSuffix:
    """Exact no-cache suffix from one explicit decoder-layer output tensor."""

    def __init__(
        self,
        causal_lm,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        source_layer: int,
        target: TargetSpec,
    ):
        from transformers.masking_utils import (
            create_causal_mask,
            create_sliding_window_causal_mask,
        )

        self.causal_lm = causal_lm
        self.base, self.lm_head, self.config, self.paths = decoder_components(causal_lm)
        self.source_layer = int(source_layer)
        self.target = target
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        if not 0 <= self.source_layer < len(self.base.layers) - 1:
            raise ValueError("source layer must precede the final decoder layer")
        if int(getattr(self.config, "num_kv_shared_layers", 0)):
            raise ArchitectureError(
                "explicit suffix currently refuses KV-shared Gemma checkpoints"
            )
        if getattr(self.config, "_attn_implementation", "eager") != "eager":
            raise ArchitectureError("exact JVP suffix requires eager attention")
        with torch.no_grad():
            embeddings = self.base.embed_tokens(input_ids)
            position_ids = torch.arange(
                embeddings.shape[1], device=embeddings.device
            ).unsqueeze(0).expand(embeddings.shape[0], -1)
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": embeddings,
                "attention_mask": attention_mask,
                "past_key_values": None,
                "position_ids": position_ids,
            }
            self.mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }
            self.position_embeddings = {
                layer_type: self.base.rotary_emb(embeddings, position_ids, layer_type)
                for layer_type in set(self.config.layer_types)
            }
            self.position_ids = position_ids
            self.per_layer_inputs = self._per_layer_inputs(input_ids, embeddings)
            hidden = embeddings
            shared = UserDict()
            for index in range(self.source_layer + 1):
                hidden = self._layer(index, hidden, shared)
            self.clean_source = hidden.detach()
        for parameter in causal_lm.parameters():
            parameter.requires_grad_(False)

    def _per_layer_inputs(self, input_ids, embeddings):
        if not getattr(self.base, "hidden_size_per_layer_input", 0):
            return None
        raw = self.base.get_per_layer_inputs(input_ids, embeddings)
        return self.base.project_per_layer_inputs(embeddings, raw).detach()

    def _layer(self, index: int, hidden: torch.Tensor, shared: UserDict) -> torch.Tensor:
        layer_type = self.config.layer_types[index]
        kwargs = {
            "attention_mask": self.mask_mapping[layer_type],
            "position_ids": self.position_ids,
            "position_embeddings": self.position_embeddings[layer_type],
            "past_key_values": None,
            "use_cache": False,
        }
        if getattr(self.causal_lm.config, "model_type", None) in {
            "gemma4", "gemma4_text"
        }:
            kwargs["shared_kv_states"] = shared
            kwargs["per_layer_input"] = (
                self.per_layer_inputs[:, :, index, :]
                if self.per_layer_inputs is not None else None
            )
        return self.base.layers[index](hidden, **kwargs)

    def __call__(self, explicit_source_fp32: torch.Tensor) -> torch.Tensor:
        if explicit_source_fp32.shape != self.clean_source.shape:
            raise ValueError("explicit source shape mismatch")
        hidden = explicit_source_fp32.to(self.clean_source.dtype)
        shared = UserDict()
        last_layer = len(self.base.layers) - 1
        stop = (
            int(self.target.target_layer)
            if self.target.representation == "block_residual" else last_layer
        )
        if stop <= self.source_layer:
            raise ValueError("target layer must be downstream of source layer")
        for index in range(self.source_layer + 1, stop + 1):
            hidden = self._layer(index, hidden, shared)
        if self.target.representation in {"block_residual", "final_residual"}:
            target = hidden
        else:
            normalized = self.base.norm(hidden)
            if self.target.representation == "normalized_final_residual":
                target = normalized
            else:
                token_ids = torch.tensor(
                    self.target.selected_token_ids,
                    device=self.lm_head.weight.device,
                    dtype=torch.long,
                )
                weight = self.lm_head.weight.index_select(0, token_ids)
                bias = (
                    self.lm_head.bias.index_select(0, token_ids)
                    if self.lm_head.bias is not None else None
                )
                target = F.linear(normalized, weight, bias)
                if self.target.representation == "post_softcap_logits":
                    cap = getattr(self.config, "final_logit_softcapping", None)
                    if cap is None:
                        raise ArchitectureError("post-softcap target requested without softcap")
                    target = torch.tanh(target / cap) * cap
        return self._select_positions(target).float()

    def _select_positions(self, tensor: torch.Tensor) -> torch.Tensor:
        indices = []
        width = tensor.shape[1]
        for raw in self.target.position_indices:
            index = raw if raw >= 0 else width + raw
            if not 0 <= index < width:
                raise IndexError(f"target position {raw} outside sequence length {width}")
            indices.append(index)
        selected = tensor[:, indices, :]
        if self.target.position_reduction == "sum":
            return selected.sum(dim=1).reshape(-1)
        if self.target.position_reduction == "mean":
            return selected.mean(dim=1).reshape(-1)
        return selected.reshape(-1)

    def full_forward_clean_target(self) -> torch.Tensor:
        """Independent full-forward target used only for clean parity checks."""
        captured = []

        def hook(_module, _inputs, output):
            tensor = output if torch.is_tensor(output) else output[0]
            captured.append(tensor)

        layer_index = (
            int(self.target.target_layer)
            if self.target.representation == "block_residual"
            else len(self.base.layers) - 1
        )
        handle = self.base.layers[layer_index].register_forward_hook(hook)
        try:
            with torch.no_grad():
                outputs = self.base(
                    input_ids=self.input_ids,
                    attention_mask=self.attention_mask,
                    use_cache=False,
                )
        finally:
            handle.remove()
        if self.target.representation in {"block_residual", "final_residual"}:
            value = captured[0]
        elif self.target.representation == "normalized_final_residual":
            value = outputs.last_hidden_state
        else:
            normalized = outputs.last_hidden_state
            ids = torch.tensor(
                self.target.selected_token_ids,
                device=self.lm_head.weight.device,
                dtype=torch.long,
            )
            value = F.linear(
                normalized,
                self.lm_head.weight.index_select(0, ids),
                None if self.lm_head.bias is None else self.lm_head.bias.index_select(0, ids),
            )
            if self.target.representation == "post_softcap_logits":
                cap = self.config.final_logit_softcapping
                value = torch.tanh(value / cap) * cap
        return self._select_positions(value).float()

    def parity(self, *, atol: float, rtol: float) -> dict:
        with torch.no_grad():
            suffix = self(self.clean_source.float())
            full = self.full_forward_clean_target()
        difference = suffix - full
        max_abs = float(difference.abs().max())
        relative = float(difference.norm() / full.norm().clamp_min(1e-30))
        return {
            "ok": bool(torch.allclose(suffix, full, atol=atol, rtol=rtol)),
            "atol": atol,
            "rtol": rtol,
            "max_abs_error": max_abs,
            "relative_l2_error": relative,
            "target_elements": suffix.numel(),
        }
