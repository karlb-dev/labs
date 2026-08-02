"""Run-root and immutable-artifact path handling for the Gemma side track."""
from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = Path(
    "/content/drive/MyDrive/interpret/special-lab-1/"
    "gemma_transport_20260802"
)
DEFAULT_PART2_ROOT = Path(
    "/content/drive/MyDrive/interpret/special-lab-1/part2_20260727"
)
DEFAULT_LOCAL_ROOT = Path("/content/gemma_transport_work")


class PathContractError(RuntimeError):
    pass


def run_root(*, create: bool = True) -> Path:
    path = Path(os.environ.get("JSPACE_GEMMA_RUN_ROOT", str(DEFAULT_RUN_ROOT)))
    if create:
        path.mkdir(parents=True, exist_ok=True)
    elif not path.exists():
        raise PathContractError(
            "Gemma Drive run root is absent; mount Drive or set "
            "JSPACE_GEMMA_RUN_ROOT"
        )
    return path


def local_root(*, create: bool = True) -> Path:
    path = Path(os.environ.get("JSPACE_GEMMA_LOCAL_ROOT", str(DEFAULT_LOCAL_ROOT)))
    if str(path).startswith("/content/drive/"):
        raise PathContractError("model/work staging must be local NVMe, not DriveFS")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def part2_root(*, must_exist: bool = True) -> Path:
    path = Path(os.environ.get("JSPACE_PART2_RUN_ROOT", str(DEFAULT_PART2_ROOT)))
    if must_exist and not path.exists():
        raise PathContractError(f"Part-2 import root is absent: {path}")
    return path


def directory(name: str) -> Path:
    if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
        raise ValueError(f"unsafe run subdirectory {name!r}")
    path = run_root() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_uri(uri: str | Path, *, must_exist: bool = True) -> Path:
    text = str(uri)
    if "://" not in text:
        path = Path(text)
    else:
        scheme, rest = text.split("://", 1)
        if rest.startswith("/") or ".." in Path(rest).parts:
            raise PathContractError(f"unsafe logical URI {text!r}")
        if scheme == "repo":
            path = REPO_ROOT / rest
        elif scheme == "gemma-run":
            path = run_root(create=False) / rest
        elif scheme == "part2-run":
            path = part2_root() / rest
        else:
            raise PathContractError(f"unsupported logical URI scheme {scheme!r}")
    if must_exist and not path.exists():
        raise PathContractError(f"artifact is absent: {text!r} -> {path}")
    return path


def assert_isolated_output(path: str | Path) -> Path:
    candidate = Path(path).resolve()
    allowed = (PACKAGE_ROOT.resolve(), run_root().resolve(), local_root().resolve())
    if not any(candidate == root or root in candidate.parents for root in allowed):
        raise PathContractError(
            f"refusing Gemma-side write outside isolated roots: {candidate}"
        )
    return candidate
