"""Hashing, git identity, and manifest builders for the study."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

from .paths import REPO_ROOT


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def json_sha256(value: object) -> str:
    return text_sha256(canonical_json(value))


def git_info(repo: Path = REPO_ROOT) -> dict:
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain"], text=True
        ).strip()
    )
    return {"code_commit": commit, "branch": branch, "dirty_tree": dirty}


def require_clean_tree(expected_branch: str | None = None) -> dict:
    info = git_info()
    if info["dirty_tree"]:
        raise RuntimeError("dirty tree at an evidence boundary; commit first")
    if expected_branch and info["branch"] != expected_branch:
        raise RuntimeError(
            f"on branch {info['branch']!r}, expected {expected_branch!r}"
        )
    return info


def directory_manifest(root: str | Path, *, relative_to: str | Path | None = None) -> list[dict]:
    """Sorted per-file rows (path, bytes, sha256) for every file under root."""
    root = Path(root)
    base = Path(relative_to) if relative_to else root
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rows.append(
            {
                "path": str(path.relative_to(base)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return rows


def runtime_fingerprint() -> dict:
    """Software/hardware identity for manifests (extended per plan §2.5 by
    the model-backed producers, which add attention/kernel bindings)."""
    import torch

    row: dict = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    for module_name in (
        "transformers",
        "tokenizers",
        "safetensors",
        "accelerate",
        "datasets",
        "triton",
        "numpy",
    ):
        try:
            module = __import__(module_name)
            row[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            row[module_name] = None
    try:
        import fla  # fla-core: fused linear-attention kernels (Qwen GDN)

        row["fla"] = getattr(fla, "__version__", "present")
    except Exception:
        row["fla"] = None
    if row["cuda_available"]:
        import torch

        props = torch.cuda.get_device_properties(0)
        row["gpu"] = {
            "name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "capability": [int(props.major), int(props.minor)],
            "uuid": str(getattr(props, "uuid", "unavailable")),
        }
        try:
            row["driver"] = (
                subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version",
                     "--format=csv,noheader"],
                    text=True, stderr=subprocess.DEVNULL,
                ).splitlines()[0].strip()
            )
        except Exception:
            row["driver"] = "unavailable"
    return row


def write_json(path: str | Path, value: object) -> str:
    """Write pretty JSON and return its sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return file_sha256(path)
