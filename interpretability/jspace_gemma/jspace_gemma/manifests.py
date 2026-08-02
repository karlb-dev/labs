"""Canonical hashes, environment locks, inventories, and atomic writes."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

import torch


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def object_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    )
    os.replace(temporary, destination)


def atomic_text(path: str | Path, value: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, destination)


def git_info(repo: str | Path | None = None) -> dict:
    root = Path(repo) if repo is not None else Path(__file__).resolve().parents[3]

    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *arguments], text=True
        ).strip()

    return {
        "code_commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_tree": bool(run("status", "--porcelain")),
    }


def require_clean_tree(*, branch: str = "interp_jspace_gemma_transport") -> dict:
    information = git_info()
    if information["dirty_tree"]:
        raise RuntimeError("refusing scientific output from a dirty tree")
    if information["branch"] != branch:
        raise RuntimeError(
            f"refusing Gemma producer on branch {information['branch']!r}; "
            f"expected {branch!r}"
        )
    return information


def inventory(paths: Iterable[str | Path], *, base: str | Path | None = None) -> list[dict]:
    root = Path(base).resolve() if base is not None else None
    rows = []
    for raw in sorted({str(Path(value)) for value in paths}):
        path = Path(raw)
        resolved = path.resolve()
        display = str(resolved.relative_to(root)) if root and (
            resolved == root or root in resolved.parents
        ) else str(path)
        rows.append(
            {
                "path": display,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def tree_inventory(root: str | Path) -> list[dict]:
    base = Path(root)
    return inventory((p for p in base.rglob("*") if p.is_file()), base=base)


def hf_remote_inventory(repo_id: str, revision: str) -> dict:
    """Resolve an immutable Hub tree, including LFS SHA-256 values."""
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("Hugging Face revision must be an exact 40-hex commit")
    from huggingface_hub import HfApi, RepoFile

    rows = []
    for item in HfApi().list_repo_tree(
        repo_id, revision=revision, recursive=True, expand=True
    ):
        if not isinstance(item, RepoFile):
            continue
        lfs = getattr(item, "lfs", None)
        rows.append(
            {
                "path": item.path,
                "size_bytes": int(item.size),
                "git_blob_id": item.blob_id,
                "lfs_sha256": getattr(lfs, "sha256", None),
                "lfs_size_bytes": getattr(lfs, "size", None),
                "xet_hash": getattr(item, "xet_hash", None),
            }
        )
    return {
        "repo_id": repo_id,
        "revision": revision,
        "files": sorted(rows, key=lambda row: row["path"]),
        "total_size_bytes": sum(row["size_bytes"] for row in rows),
        "inventory_sha256": object_sha256(sorted(rows, key=lambda row: row["path"])),
    }


def environment_payload(*, require_gpu: bool = False) -> dict:
    cuda_available = bool(torch.cuda.is_available())
    if require_gpu and not cuda_available:
        raise RuntimeError("CUDA unavailable; never create model-scale evidence on CPU")
    gpu = None
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        driver = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).splitlines()[0].strip()
        gpu = {
            "name": props.name,
            "capability": [int(props.major), int(props.minor)],
            "total_memory_bytes": int(props.total_memory),
            "driver_version": driver,
        }
    packages = []
    wanted = {
        "accelerate", "huggingface-hub", "matplotlib", "numpy", "pandas",
        "pyarrow", "pyyaml", "safetensors", "scipy", "tokenizers", "torch",
        "transformers",
    }
    for name in sorted(wanted):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        packages.append({"name": name, "version": version})
    return {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_cuda_available": cuda_available,
        "gpu": gpu,
        "packages": packages,
    }


def verify_constraints(path: str | Path) -> dict:
    requested = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, expected = line.split("==", 1)
        requested[name.lower()] = expected
    installed = {}
    mismatches = {}
    for name, expected in requested.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        installed[name] = actual
        if actual != expected:
            mismatches[name] = {"expected": expected, "actual": actual}
    return {
        "ok": not mismatches,
        "constraints_sha256": file_sha256(path),
        "checked": installed,
        "mismatches": mismatches,
    }
