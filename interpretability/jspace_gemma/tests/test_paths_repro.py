from pathlib import Path

import pytest

from jspace_gemma.paths import PathContractError, assert_isolated_output, resolve_uri
from jspace_gemma.manifests import object_sha256
from jspace_gemma.staging import StagingError, require_local_cache_root, verify_snapshot


def test_logical_uri_refuses_parent_traversal():
    with pytest.raises(PathContractError, match="unsafe"):
        resolve_uri("repo://../outside", must_exist=False)


def test_output_isolation_rejects_other_phase(tmp_path, monkeypatch):
    run = tmp_path / "run"
    local = tmp_path / "local"
    monkeypatch.setenv("JSPACE_GEMMA_RUN_ROOT", str(run))
    monkeypatch.setenv("JSPACE_GEMMA_LOCAL_ROOT", str(local))
    assert assert_isolated_output(run / "metrics" / "x.json") == (run / "metrics" / "x.json").resolve()
    with pytest.raises(PathContractError, match="outside isolated"):
        assert_isolated_output(tmp_path / "phase4" / "x.json")


def test_model_cache_must_be_explicit_local_nvme():
    assert require_local_cache_root("/content/hf_olmo_control") == Path(
        "/content/hf_olmo_control"
    )
    for unsafe in ("/content", "/", "/content/drive/MyDrive/hf_cache"):
        with pytest.raises(StagingError):
            require_local_cache_root(unsafe)


def test_snapshot_verifier_checks_full_lfs_hash_and_index(tmp_path):
    import hashlib
    import json

    revision = "a" * 40
    snapshot = tmp_path / "models--org--model" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    weight = b"safe-weight-bytes"
    (snapshot / "model-00001-of-00001.safetensors").write_bytes(weight)
    index = {
        "metadata": {},
        "weight_map": {"model.x": "model-00001-of-00001.safetensors"},
    }
    (snapshot / "model.safetensors.index.json").write_text(json.dumps(index))
    files = [
        {
            "path": "model-00001-of-00001.safetensors",
            "size_bytes": len(weight),
            "lfs_sha256": hashlib.sha256(weight).hexdigest(),
        },
        {
            "path": "model.safetensors.index.json",
            "size_bytes": (snapshot / "model.safetensors.index.json").stat().st_size,
            "lfs_sha256": None,
            "git_blob_id": hashlib.sha1(
                b"blob "
                + str((snapshot / "model.safetensors.index.json").stat().st_size).encode()
                + b"\0"
                + (snapshot / "model.safetensors.index.json").read_bytes()
            ).hexdigest(),
        },
    ]
    remote = {
        "repo_id": "org/model",
        "revision": revision,
        "files": files,
        "inventory_sha256": object_sha256(files),
    }
    result = verify_snapshot(
        snapshot, repo_id="org/model", revision=revision, remote_inventory=remote
    )
    assert result["all_content_hashes_verified"]
    (snapshot / "model-00001-of-00001.safetensors").write_bytes(b"corrupt")
    result = verify_snapshot(
        snapshot, repo_id="org/model", revision=revision, remote_inventory=remote
    )
    assert not result["all_content_hashes_verified"]
