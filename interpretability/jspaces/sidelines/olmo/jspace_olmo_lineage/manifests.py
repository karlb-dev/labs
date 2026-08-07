"""Canonical hashes, environment locks, and input manifests."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

def _find_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path(__file__)).resolve()
    if cur.is_file():
        cur = cur.parent
    for p in [cur, *cur.parents]:
        if (p / ".git").exists():
            return p
    raise RuntimeError("cannot locate git repository root")

from typing import Mapping

import torch


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


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
    temporary = destination.with_suffix(
        destination.suffix + f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(
        value, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, destination)


def atomic_text(path: str | Path, value: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(
        destination.suffix + f".tmp{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, destination)


def git_info(repo: str | Path | None = None) -> dict:
    root = Path(repo) if repo is not None else Path(
        __file__).resolve().parents[3]

    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *arguments], text=True).strip()

    return {
        "code_commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_tree": bool(run("status", "--porcelain")),
    }


def require_clean_tree(*, expected_branch: str | None = None) -> dict:
    information = git_info()
    if information["dirty_tree"]:
        raise RuntimeError(
            "refusing scientific output from a dirty tree; commit first")
    if expected_branch is not None and information["branch"] != expected_branch:
        raise RuntimeError(
            f"expected branch {expected_branch!r}, got "
            f"{information['branch']!r}")
    return information


def commit_reachable(commit: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(_find_repo_root()),
         "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True,
    )
    return result.returncode == 0


@dataclass(frozen=True)
class InputManifest:
    experiment_id: str
    config_sha256: str
    model_id: str
    model_revision: str
    tokenizer_manifest_sha256: str
    lens_sha256: str
    bank_sha256: str
    partition_sha256: str
    scoring_spec_sha256: str
    upstream: Mapping[str, str] = field(default_factory=dict)
    code_commit: str | None = None
    schema_version: int = 1

    def payload(self) -> dict:
        value = asdict(self)
        if value["code_commit"] is None:
            value["code_commit"] = git_info()["code_commit"]
        return value

    def sha256(self) -> str:
        return object_sha256(self.payload())

    def envelope(self) -> dict:
        payload = self.payload()
        return {
            "schema_version": 1,
            "payload": payload,
            "payload_sha256": object_sha256(payload),
        }


def environment_payload(*, require_gpu: bool = False) -> dict:
    cuda_available = bool(torch.cuda.is_available())
    if require_gpu and not cuda_available:
        raise RuntimeError("CUDA unavailable; model evidence cannot run on CPU")
    gpu = None
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        try:
            driver = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version",
                 "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).splitlines()[0].strip()
        except Exception:
            driver = "unavailable"
        gpu = {
            "name": properties.name,
            "capability": [int(properties.major), int(properties.minor)],
            "total_memory_bytes": int(properties.total_memory),
            "driver_version": driver,
        }
    packages = sorted({
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }.items())
    return {
        "schema_version": 1,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_cuda_available": cuda_available,
        "gpu": gpu,
        "packages": [
            {"name": name, "version": version}
            for name, version in packages
        ],
    }


def verify_constraints(path: str | Path, *,
                       package_names: set[str] | None = None) -> dict:
    requested = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        requested[name.lower()] = version
    if package_names is not None:
        allowed = {value.lower() for value in package_names}
        requested = {
            name: version for name, version in requested.items()
            if name in allowed
        }
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
