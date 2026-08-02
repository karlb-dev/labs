import hashlib
import json

import pytest
import torch
import yaml

from jspace_phase4.experiments.p4_qwen_fit_tensor_audit import audit_tensors


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixtures(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "recipe": {
            "source_layers": "all_below_target",
            "target_layer": 2,
            "expected_d_model": 3,
            "skip_first": 1,
            "lens_save_dtype": "float16",
        },
    }))
    sums = {
        0: torch.arange(9, dtype=torch.float32).reshape(3, 3),
        1: torch.arange(9, dtype=torch.float32).reshape(3, 3) + 2,
    }
    checkpoint = tmp_path / "fit.ckpt"
    torch.save({
        "jacobian_sum": sums,
        "n_done": 2,
        "next_idx": 2,
        "source_layers": [0, 1],
        "target_layer": 2,
        "skip_first": 1,
    }, checkpoint)
    lens = tmp_path / "lens.pt"
    torch.save({
        "J": {layer: (value / 2).to(torch.float16)
              for layer, value in sums.items()},
        "n_prompts": 2,
        "source_layers": [0, 1],
        "d_model": 3,
    }, lens)
    state = tmp_path / "checkpoint_state.json"
    state.write_text(json.dumps({
        "schema_version": 1,
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "fit_contract_sha256": "f" * 64,
        "n_done": 2,
        "next_idx": 2,
    }) + "\n")
    return config, checkpoint, lens, state


def test_tensor_audit_proves_exact_quantized_mean(tmp_path):
    config, checkpoint, lens, state = fixtures(tmp_path)
    result = audit_tensors(
        config_path=config,
        lens_path=lens,
        checkpoint_path=checkpoint,
        checkpoint_state_path=state,
        expected_prompts=2,
        expected_fit_contract_sha256="f" * 64,
    )
    assert result["ok"] is True
    assert result["all_layers_finite"] is True
    assert len(result["layers"]) == 2
    assert all(row["exact_quantized_mean_match"] for row in result["layers"])


def test_tensor_audit_rejects_lens_drift(tmp_path):
    config, checkpoint, lens, state = fixtures(tmp_path)
    value = torch.load(lens, weights_only=True)
    value["J"][1][0, 0] += 1
    torch.save(value, lens)
    with pytest.raises(RuntimeError, match="not the exact quantized"):
        audit_tensors(
            config_path=config,
            lens_path=lens,
            checkpoint_path=checkpoint,
            checkpoint_state_path=state,
            expected_prompts=2,
        )


def test_tensor_audit_rejects_nonfinal_header(tmp_path):
    config, checkpoint, lens, state = fixtures(tmp_path)
    header = json.loads(state.read_text())
    header["next_idx"] = 1
    state.write_text(json.dumps(header) + "\n")
    with pytest.raises(RuntimeError, match="next_idx=1"):
        audit_tensors(
            config_path=config,
            lens_path=lens,
            checkpoint_path=checkpoint,
            checkpoint_state_path=state,
            expected_prompts=2,
        )
