"""Safe local-NVMe staging and full LFS hash verification."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

from .manifests import atomic_json, file_sha256, hf_remote_inventory, object_sha256


class StagingError(RuntimeError):
    pass


def _git_blob_id(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def exclusive_lock(path: str | Path):
    lock = Path(path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StagingError(f"staging lock is owned: {lock}") from exc
        handle.write(str(os.getpid()))
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def require_local_cache_root(path: str | Path) -> Path:
    root = Path(path).resolve()
    if str(root).startswith("/content/drive/") or not str(root).startswith("/content/"):
        raise StagingError(f"cache root must be explicit local /content NVMe: {root}")
    if root in {Path("/content"), Path("/")}:
        raise StagingError("cache root is too broad")
    return root


def seed_hf_cache(seed_model_root: str | Path, cache_root: str | Path) -> dict:
    source = Path(seed_model_root).resolve()
    destination_cache = require_local_cache_root(cache_root)
    if not source.is_dir() or "models--" not in source.name:
        raise StagingError(f"invalid HF seed model root: {source}")
    destination = destination_cache / source.name
    destination_cache.mkdir(parents=True, exist_ok=True)
    before = shutil.disk_usage(destination_cache).free
    subprocess.run(
        [
            "rsync", "-a", "--ignore-existing", "--exclude=*.incomplete",
            str(source) + "/", str(destination) + "/",
        ],
        check=True,
    )
    after = shutil.disk_usage(destination_cache).free
    return {
        "source": str(source),
        "destination": str(destination),
        "bytes_added_approx": max(0, before - after),
        "incomplete_files_copied": False,
    }


def verify_snapshot(
    snapshot: str | Path,
    *,
    repo_id: str,
    revision: str,
    remote_inventory: dict | None = None,
) -> dict:
    root = Path(snapshot).resolve()
    if root.name != revision or root.parent.name != "snapshots":
        raise StagingError(f"snapshot path does not end in snapshots/{revision}: {root}")
    remote = remote_inventory or hf_remote_inventory(repo_id, revision)
    rows = []
    failures = []
    for expected in remote["files"]:
        path = root / expected["path"]
        exists = path.exists()
        size = path.stat().st_size if exists else None
        actual_sha256 = file_sha256(path) if exists else None
        expected_sha256 = expected.get("lfs_sha256")
        actual_git_blob_id = _git_blob_id(path) if exists and expected_sha256 is None else None
        ok = exists and size == expected["size_bytes"]
        if expected_sha256 is not None:
            ok = ok and actual_sha256 == expected_sha256
        else:
            ok = ok and actual_git_blob_id == expected.get("git_blob_id")
        row = {
            "path": expected["path"],
            "size_bytes": size,
            "expected_size_bytes": expected["size_bytes"],
            "sha256": actual_sha256,
            "expected_lfs_sha256": expected_sha256,
            "git_blob_id": actual_git_blob_id,
            "expected_git_blob_id": expected.get("git_blob_id"),
            "ok": bool(ok),
        }
        rows.append(row)
        if not ok:
            failures.append(row)
    index_path = root / "model.safetensors.index.json"
    index_shards = set()
    if index_path.exists():
        index = json.loads(index_path.read_text())
        index_shards = set(index.get("weight_map", {}).values())
    remote_shards = {
        row["path"] for row in remote["files"] if row["path"].endswith(".safetensors")
    }
    if index_shards != remote_shards:
        failures.append(
            {
                "path": "model.safetensors.index.json",
                "ok": False,
                "reason": "index shard set differs from immutable remote inventory",
                "index_shards": sorted(index_shards),
                "remote_shards": sorted(remote_shards),
            }
        )
    payload = {
        "schema_version": 1,
        "repo_id": repo_id,
        "revision": revision,
        "snapshot": str(root),
        "remote_inventory_sha256": remote["inventory_sha256"],
        "files": rows,
        "weight_shards": sorted(remote_shards),
        "all_content_hashes_verified": not failures,
        "failures": failures,
    }
    payload["snapshot_manifest_sha256"] = object_sha256(payload)
    return payload


def stage_snapshot(
    *,
    repo_id: str,
    revision: str,
    cache_root: str | Path,
    seed_model_root: str | Path | None,
    output_manifest: str | Path,
) -> dict:
    cache = require_local_cache_root(cache_root)
    remote = hf_remote_inventory(repo_id, revision)
    required = remote["total_size_bytes"] + 5 * 2**30
    free = shutil.disk_usage(cache.parent if cache.parent.exists() else Path("/content")).free
    if free < required:
        raise StagingError(
            f"insufficient local disk: need {required / 2**30:.1f} GiB, "
            f"have {free / 2**30:.1f} GiB"
        )
    with exclusive_lock(cache / ".jspace_gemma_stage.lock"):
        seed = None
        if seed_model_root is not None:
            seed = seed_hf_cache(seed_model_root, cache)
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            repo_id,
            revision=revision,
            cache_dir=cache,
            local_files_only=False,
            max_workers=4,
        )
        verification = verify_snapshot(
            snapshot, repo_id=repo_id, revision=revision, remote_inventory=remote
        )
        verification["seed"] = seed
        verification["local_cache_root"] = str(cache)
        if not verification["all_content_hashes_verified"]:
            atomic_json(output_manifest, verification)
            raise StagingError("staged snapshot failed full content verification")
        atomic_json(output_manifest, verification)
        return verification
