import hashlib
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
        fitter_source_sha256="f" * 64,
    )
    first = fit_contract_payload(**kwargs)
    second = fit_contract_payload(**kwargs)
    assert first == second
    assert first["fitter_source_sha256"] == "f" * 64
    assert first["model_snapshot_inventory_sha256"] == "i" * 64
