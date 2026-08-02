import json
from pathlib import Path

import pytest
import torch

from jspace_gemma.architecture import manifest_from_config
from jspace_gemma.autodiff import exact_jvp
from jspace_gemma.hooks import ExplicitDecoderSuffix, TargetSpec


def test_committed_gemma_architecture_manifest_has_exact_schedule():
    path = Path(__file__).resolve().parents[1] / "configs/gemma4_31b_architecture_manifest.json"
    value = json.loads(path.read_text())
    assert value["decoder"]["num_layers"] == 60
    assert value["decoder"]["full_layers_zero_indexed"] == list(range(5, 60, 6))
    assert len(value["decoder"]["sliding_layers_zero_indexed"]) == 50
    assert value["attention"]["global_attention_keys_equal_values"]
    assert value["per_layer_embeddings"]["enabled"] is False
    assert value["readout"]["final_logit_softcap"] == 30.0


def test_manifest_parser_rejects_layer_schedule_mismatch(tmp_path):
    config = {
        "model_type": "olmo3",
        "num_hidden_layers": 2,
        "layer_types": ["full_attention"],
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-6,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(Exception, match="layer_types"):
        manifest_from_config(path)


def test_tiny_hf_olmo_explicit_suffix_matches_full_forward_and_supports_jvp():
    from transformers.models.olmo3.configuration_olmo3 import Olmo3Config
    from transformers.models.olmo3.modeling_olmo3 import Olmo3ForCausalLM

    torch.manual_seed(31)
    config = Olmo3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        use_cache=False,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention", "sliding_attention"],
    )
    config._attn_implementation = "eager"
    model = Olmo3ForCausalLM(config).eval()
    ids = torch.tensor([[2, 3, 4, 5, 6]], dtype=torch.long)
    attention = torch.ones_like(ids)
    suffix = ExplicitDecoderSuffix(
        model,
        input_ids=ids,
        attention_mask=attention,
        source_layer=0,
        target=TargetSpec("final_residual"),
    )
    parity = suffix.parity(atol=1e-6, rtol=1e-6)
    assert parity["ok"], parity
    direction = torch.randn_like(suffix.clean_source.float())
    direction[:, :-1, :] = 0
    direction /= direction.norm()
    result = exact_jvp(suffix, suffix.clean_source.float(), direction)
    epsilon = 1e-3
    central = (
        suffix(suffix.clean_source.float() + epsilon * direction)
        - suffix(suffix.clean_source.float() - epsilon * direction)
    ) / (2 * epsilon)
    cosine = torch.nn.functional.cosine_similarity(result.tangent, central, dim=0)
    assert float(cosine) > 0.999


def test_tiny_hf_gemma_suffix_preserves_global_k_eq_v_path_and_supports_jvp():
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    torch.manual_seed(37)
    config = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        global_head_dim=4,
        num_global_key_value_heads=2,
        hidden_size_per_layer_input=0,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"],
        sliding_window=8,
        max_position_embeddings=32,
        attention_k_eq_v=True,
        final_logit_softcapping=30.0,
        use_bidirectional_attention="vision",
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = Gemma4ForCausalLM(config).eval()
    assert model.model.layers[-1].self_attn.v_proj is None
    ids = torch.tensor([[2, 3, 4, 5, 6]], dtype=torch.long)
    attention = torch.ones_like(ids)
    suffix = ExplicitDecoderSuffix(
        model,
        input_ids=ids,
        attention_mask=attention,
        source_layer=0,
        target=TargetSpec("final_residual"),
    )
    assert suffix.parity(atol=1e-6, rtol=1e-6)["ok"]
    direction = torch.randn_like(suffix.clean_source.float())
    direction[:, :-1, :] = 0
    direction /= direction.norm()
    exact = exact_jvp(suffix, suffix.clean_source.float(), direction)
    epsilon = 1e-3
    central = (
        suffix(suffix.clean_source.float() + epsilon * direction)
        - suffix(suffix.clean_source.float() - epsilon * direction)
    ) / (2 * epsilon)
    cosine = torch.nn.functional.cosine_similarity(exact.tangent, central, dim=0)
    assert float(cosine) > 0.999
