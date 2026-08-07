import hashlib
import importlib.metadata
import json

import pytest


def test_model_reference_requires_revision_pin():
    from jspace_phase4.experiments.p4_qwen_nested_lens_fit import (
        model_reference,
    )
    assert model_reference("model://org/name@" + "a" * 40) == {
        "model_id": "org/name",
        "revision": "a" * 40,
    }
    with pytest.raises(ValueError, match="pin"):
        model_reference("model://org/name")


def test_verify_package_versions_refuses_missing_or_changed_runtime():
    from jspace_phase4.experiments.p4_qwen_nested_lens_fit import (
        verify_package_versions,
    )
    installed = {"torch": "2.11.0+cu128", "fla-core": "0.5.2"}

    def version_reader(name):
        if name not in installed:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed[name]

    assert verify_package_versions(
        installed, version_reader=version_reader) == installed
    with pytest.raises(RuntimeError, match="expected 0.5.3"):
        verify_package_versions(
            {"fla-core": "0.5.3"}, version_reader=version_reader)
    with pytest.raises(RuntimeError, match="missing"):
        verify_package_versions(
            {"flash-linear-attention": "0.5.2"},
            version_reader=version_reader,
        )


def test_verify_model_fused_bindings_requires_every_expected_block():
    from jspace_phase4.experiments.p4_qwen_nested_lens_fit import (
        verify_model_fused_bindings,
    )

    def fused_kernel():
        pass

    fused_kernel.__module__ = "fla.ops.gated_delta_rule.chunk"

    class Block:
        def __init__(self):
            self.chunk_gated_delta_rule = fused_kernel

    class Model:
        @staticmethod
        def named_modules():
            return iter((("layer.0", Block()), ("layer.1", Block())))

    specification = {
        "qwen_kernel_modules": {
            "chunk_gated_delta_rule": "fla.ops.gated_delta_rule.",
        },
        "expected_linear_attention_modules": 2,
    }
    result = verify_model_fused_bindings(Model(), specification)
    assert result["linear_attention_module_count"] == 2
    assert result["chunk_gated_delta_rule_modules"] == [
        "fla.ops.gated_delta_rule.chunk"]
    specification["expected_linear_attention_modules"] = 3
    with pytest.raises(RuntimeError, match="found 2"):
        verify_model_fused_bindings(Model(), specification)


def test_choose_recovery_prefers_highest_then_local():
    from jspace_phase4.experiments.p4_qwen_nested_lens_fit import (
        choose_recovery_candidate,
    )
    contract = "c" * 64
    local = {"fit_contract_sha256": contract, "next_idx": 20}
    drive = {"fit_contract_sha256": contract, "next_idx": 10}
    assert choose_recovery_candidate(
        local, drive, fit_contract_sha256=contract) == "local"
    drive["next_idx"] = 30
    assert choose_recovery_candidate(
        local, drive, fit_contract_sha256=contract) == "drive"
    local["next_idx"] = 30
    assert choose_recovery_candidate(
        local, drive, fit_contract_sha256=contract) == "local"
    with pytest.raises(RuntimeError, match="incompatible"):
        choose_recovery_candidate(
            {"fit_contract_sha256": "x", "next_idx": 40},
            drive,
            fit_contract_sha256=contract,
        )


def test_copy_atomic_verified_checks_hash(tmp_path):
    from jspace_phase4.experiments.p4_qwen_nested_lens_fit import (
        copy_atomic_verified,
    )
    source = tmp_path / "source.ckpt"
    destination = tmp_path / "drive" / "fit.ckpt"
    source.write_bytes(b"checkpoint")
    expected = hashlib.sha256(b"checkpoint").hexdigest()
    assert copy_atomic_verified(
        source, destination, expected_sha256=expected) == expected
    assert destination.read_bytes() == b"checkpoint"
    with pytest.raises(RuntimeError, match="hash mismatch"):
        copy_atomic_verified(
            source, destination, expected_sha256="0" * 64)
    assert destination.read_bytes() == b"checkpoint"


def test_verify_snapshot_checks_every_file(tmp_path):
    from jspace_phase4.experiments.p4_qwen_nested_lens_fit import (
        verify_snapshot,
    )
    config = tmp_path / "config.json"
    weight = tmp_path / "model-00001-of-00001.safetensors"
    config.write_bytes(b"{}")
    weight.write_bytes(b"weights")
    manifest = {
        "model_id": "org/model",
        "revision": "a" * 40,
        "architecture": "TestModel",
        "n_layers": 2,
        "d_model": 4,
        "weight_bytes": len(b"weights"),
        "files": [
            {
                "name": config.name,
                "bytes": config.stat().st_size,
                "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            },
            {
                "name": weight.name,
                "bytes": weight.stat().st_size,
                "sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
            },
        ],
    }
    verified = verify_snapshot(tmp_path, manifest)
    assert verified["weight_bytes"] == len(b"weights")
    weight.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_snapshot(tmp_path, manifest)


def test_fit_contract_is_canonical_and_pins_source_hashes(tmp_path):
    from jspace_phase4.experiments.p4_qwen_nested_lens_fit import (
        fit_contract_payload,
    )
    config_path = tmp_path / "config.yaml"
    corpus_path = tmp_path / "corpus.jsonl"
    snapshot_path = tmp_path / "snapshot.json"
    config_path.write_text("x: 1\n")
    corpus_path.write_text(json.dumps({"idx": 1, "text": "x"}) + "\n")
    snapshot_path.write_text("{}\n")
    config = {
        "model_uri": "model://org/model@" + "a" * 40,
        "recipe": {"target_layer": 3},
        "draws": {"draw_a": {"corpus_uri": "artifact://phase4/corpus"}},
    }
    kwargs = dict(
        config_path=config_path,
        config=config,
        draw="draw_a",
        corpus_path=corpus_path,
        corpus_sha256=hashlib.sha256(
            corpus_path.read_bytes()).hexdigest(),
        model_snapshot_manifest_path=snapshot_path,
        model_snapshot={"inventory_sha256": "i" * 64},
        jlens_contract={"revision": "b" * 40},
        runtime_contract={
            "packages": {"torch": "2.11.0+cu128"},
            "qwen_kernels": {"bindings": {"chunk": "fla.ops.chunk"}},
        },
        fitter_source_sha256="f" * 64,
    )
    first = fit_contract_payload(**kwargs)
    second = fit_contract_payload(**kwargs)
    assert first == second
    assert first["fitter_source_sha256"] == "f" * 64
    assert first["model_snapshot_inventory_sha256"] == "i" * 64
