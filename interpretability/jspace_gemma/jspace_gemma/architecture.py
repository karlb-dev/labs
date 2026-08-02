"""Exact Gemma/OLMo architecture manifests and loaded-module audits."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

from .manifests import file_sha256, object_sha256


class ArchitectureError(RuntimeError):
    pass


def _text_config(raw: dict) -> dict:
    return raw.get("text_config", raw)


def _minimal_period(values: list[str]) -> list[str]:
    for width in range(1, len(values) + 1):
        if all(values[i] == values[i % width] for i in range(len(values))):
            return values[:width]
    return values


def manifest_from_config(path: str | Path) -> dict:
    source = Path(path)
    outer = json.loads(source.read_text())
    cfg = _text_config(outer)
    layer_types = list(cfg["layer_types"])
    n_layers = int(cfg["num_hidden_layers"])
    if len(layer_types) != n_layers:
        raise ArchitectureError(
            f"layer_types has {len(layer_types)} entries for {n_layers} layers"
        )
    model_type = cfg.get("model_type")
    family = "gemma4" if model_type == "gemma4_text" else model_type
    local = [i for i, value in enumerate(layer_types) if value == "sliding_attention"]
    full = [i for i, value in enumerate(layer_types) if value == "full_attention"]
    if set(local + full) != set(range(n_layers)):
        raise ArchitectureError("unknown decoder layer type in exact config")
    k_eq_v = bool(cfg.get("attention_k_eq_v", False))
    local_head_dim = int(
        cfg.get("head_dim", int(cfg["hidden_size"]) // int(cfg["num_attention_heads"]))
    )
    global_head_dim = int(cfg.get("global_head_dim") or local_head_dim)
    local_kv = int(cfg["num_key_value_heads"])
    global_kv = int(cfg.get("num_global_key_value_heads") or local_kv)
    manifest = {
        "schema_version": 1,
        "config_path": str(source),
        "config_sha256": file_sha256(source),
        "outer_model_type": outer.get("model_type"),
        "text_model_type": model_type,
        "family": family,
        "decoder": {
            "num_layers": n_layers,
            "residual_width": int(cfg["hidden_size"]),
            "mlp_width": int(cfg["intermediate_size"]),
            "layer_types": layer_types,
            "minimal_layer_type_period": _minimal_period(layer_types),
            "sliding_layers_zero_indexed": local,
            "full_layers_zero_indexed": full,
            "sliding_window": cfg.get("sliding_window"),
        },
        "attention": {
            "query_heads": int(cfg["num_attention_heads"]),
            "local_kv_heads": local_kv,
            "global_kv_heads": global_kv,
            "local_head_dim": local_head_dim,
            "global_head_dim": global_head_dim,
            "attention_k_eq_v_global": k_eq_v,
            "num_kv_shared_layers": int(cfg.get("num_kv_shared_layers", 0)),
            "qknorm": model_type in {"gemma4_text", "olmo3"},
            "rope": cfg.get("rope_parameters", cfg.get("rope_scaling")),
            "rope_theta": cfg.get("rope_theta"),
        },
        "normalization": {
            "type": "RMSNorm",
            "epsilon": float(cfg["rms_norm_eps"]),
            "gemma_four_norm_block": model_type == "gemma4_text",
            "hook_semantics": {
                "layer_output": "post-attention-residual and post-MLP-residual; pre-final-norm",
                "final_residual": f"decoder layer {n_layers - 1} output before final RMSNorm",
                "normalized_final_residual": "text decoder final RMSNorm output",
            },
        },
        "mlp": {
            "activation": cfg.get("hidden_activation", cfg.get("hidden_act")),
            "equation": "down_proj(activation(gate_proj(x)) * up_proj(x))",
            "moe_enabled": bool(cfg.get("enable_moe_block", False)),
            "double_wide_mlp": bool(cfg.get("use_double_wide_mlp", False)),
        },
        "readout": {
            "tied_embedding_unembedding": bool(cfg.get("tie_word_embeddings", False)),
            "final_logit_softcap": cfg.get("final_logit_softcapping"),
            "softcap_location": "after lm_head; monotone and therefore rank-preserving",
        },
        "per_layer_embeddings": {
            "enabled": bool(cfg.get("hidden_size_per_layer_input", 0)),
            "width": int(cfg.get("hidden_size_per_layer_input", 0)),
        },
        "source_layer_indexing": "zero-indexed decoder ModuleList indices, matching historical jlens hooks",
        "module_graph_audit": "pending_loaded_model_audit",
    }
    manifest["architecture_sha256"] = object_sha256(manifest)
    return manifest


def decoder_components(causal_lm) -> tuple[object, object, object, dict]:
    """Return text decoder, LM head, text config, and exact path labels."""
    outer_type = getattr(causal_lm.config, "model_type", None)
    if outer_type == "gemma4":
        base = causal_lm.model.language_model
        path = "model.language_model"
    elif outer_type == "gemma4_text":
        base = causal_lm.model
        path = "model"
    elif outer_type == "olmo3":
        base = causal_lm.model
        path = "model"
    else:
        raise ArchitectureError(f"unsupported model type {outer_type!r}")
    return base, causal_lm.lm_head, base.config, {
        "causal_lm_class": type(causal_lm).__name__,
        "text_decoder_path": path,
        "decoder_layers_path": f"{path}.layers",
        "final_norm_path": f"{path}.norm",
        "lm_head_path": "lm_head",
    }


def audit_loaded_model(causal_lm) -> dict:
    base, lm_head, cfg, paths = decoder_components(causal_lm)
    layers = list(base.layers)
    if len(layers) != cfg.num_hidden_layers:
        raise ArchitectureError("loaded decoder layer count disagrees with config")
    layer_rows = []
    for index, layer in enumerate(layers):
        attention = layer.self_attn
        expected = cfg.layer_types[index]
        actual = getattr(attention, "layer_type", getattr(attention, "attention_type", None))
        if actual != expected:
            raise ArchitectureError(
                f"layer {index} type mismatch: config={expected!r}, module={actual!r}"
            )
        norms = [
            name for name in (
                "input_layernorm", "post_attention_layernorm",
                "pre_feedforward_layernorm", "post_feedforward_layernorm",
            )
            if hasattr(layer, name)
        ]
        layer_rows.append(
            {
                "layer": index,
                "layer_type": actual,
                "class": type(layer).__name__,
                "attention_class": type(attention).__name__,
                "attention_head_dim": int(attention.head_dim),
                "has_q_norm": hasattr(attention, "q_norm"),
                "has_k_norm": hasattr(attention, "k_norm"),
                "v_proj_is_none": getattr(attention, "v_proj", object()) is None,
                "norm_modules": norms,
                "mlp_class": type(layer.mlp).__name__,
                "mlp_width": int(layer.mlp.intermediate_size),
            }
        )
    tied = lm_head.weight.data_ptr() == base.embed_tokens.weight.data_ptr()
    source_file = inspect.getsourcefile(type(layers[0]))
    return {
        **paths,
        "num_layers": len(layers),
        "residual_width": int(cfg.hidden_size),
        "layer_rows": layer_rows,
        "tied_embedding_unembedding_loaded": tied,
        "decoder_source_file": source_file,
        "decoder_source_sha256": file_sha256(source_file) if source_file else None,
        "parameter_dtype_counts": _dtype_counts(causal_lm),
    }


def _dtype_counts(model) -> dict:
    counts: dict[str, int] = {}
    for parameter in model.parameters():
        key = str(parameter.dtype)
        counts[key] = counts.get(key, 0) + parameter.numel()
    return counts
