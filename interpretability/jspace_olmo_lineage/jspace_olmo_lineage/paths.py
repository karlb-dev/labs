"""Strict path and logical-URI handling for the isolated side track."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SPECIAL_LAB_ROOT = Path(
    "/content/drive/MyDrive/interpret/special-lab-1")
DEFAULT_RUN_ROOT = SPECIAL_LAB_ROOT / "olmo_lineage_20260801"
DEFAULT_LOCAL_WORK = Path("/content/olmo_lineage_work")
DRIVE_ALIASES = {
    "part1": "2026-07-25_1726",
    "part1v2": "2026-07-26_v2",
    "part2": "part2_20260727",
    "phase3": "phase3_20260729",
    "phase4": "phase4_20260731",
    "olmo": "olmo_lineage_20260801",
}


class PathBoundaryError(RuntimeError):
    pass


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def run_root(*, create: bool = True) -> Path:
    path = Path(os.environ.get(
        "JSPACE_OLMO_RUN_ROOT", str(DEFAULT_RUN_ROOT)))
    if not _within(path, SPECIAL_LAB_ROOT) or not path.name.startswith(
            "olmo_lineage_"):
        raise PathBoundaryError(
            "OLMo output root must be an olmo_lineage_* directory below "
            f"{SPECIAL_LAB_ROOT}")
    if any(token in str(path) for token in ("phase4_", "gemma_transport_")):
        raise PathBoundaryError("OLMo output root overlaps another track")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    elif not path.exists():
        raise PathBoundaryError(f"OLMo run root is absent: {path}")
    return path


def local_work(*, create: bool = True) -> Path:
    path = Path(os.environ.get(
        "JSPACE_OLMO_LOCAL_WORK", str(DEFAULT_LOCAL_WORK)))
    if _within(path, Path("/content/drive")):
        raise PathBoundaryError("local model/work staging may not use DriveFS")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def manifests_dir() -> Path:
    path = run_root() / "manifests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def metrics_dir(slug: str) -> Path:
    path = run_root() / "metrics" / slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir() -> Path:
    path = run_root() / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def reports_dir() -> Path:
    path = run_root() / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def release_dir() -> Path:
    path = run_root() / "release"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_uri(uri: str | Path, *, must_exist: bool = True) -> Path:
    text = str(uri)
    if "://" not in text:
        path = Path(text)
    else:
        scheme, rest = text.split("://", 1)
        if scheme == "repo":
            path = REPO_ROOT / rest
        elif scheme == "drive":
            head, separator, tail = rest.partition("/")
            mapped = DRIVE_ALIASES.get(head, head)
            path = SPECIAL_LAB_ROOT / mapped
            if separator:
                path /= tail
        elif scheme == "olmo-artifact":
            path = run_root(create=False) / rest
        elif scheme == "model":
            return resolve_model(rest, must_exist=must_exist)
        else:
            raise PathBoundaryError(f"unsupported URI scheme {scheme!r}")
    if must_exist and not path.exists():
        raise PathBoundaryError(f"artifact is absent: {text!r} -> {path}")
    return path


def resolve_model(reference: str, *, must_exist: bool = True) -> Path:
    model_id, separator, revision = reference.rpartition("@")
    if not separator or not model_id or not revision:
        raise PathBoundaryError("model URI must contain an exact revision")
    cache_name = "models--" + model_id.replace("/", "--")
    candidates = []
    for cache in (os.environ.get("HF_HUB_CACHE", ""), "/content/hf_local"):
        if not cache:
            continue
        root = Path(cache)
        if _within(root, Path("/content/drive")):
            raise PathBoundaryError(
                "HF_HUB_CACHE points into DriveFS; stage weights locally")
        candidates.append(root / cache_name / "snapshots" / revision)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    destination = Path("/content/models") / model_id.rsplit("/", 1)[-1]
    if not must_exist:
        return destination
    raise PathBoundaryError(
        f"pinned model {model_id}@{revision} is not staged on local NVMe")
