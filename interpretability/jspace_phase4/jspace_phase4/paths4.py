"""Run-root and logical artifact URI resolution for Phase 4."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ROOT = Path(
    "/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731")
DEFAULT_DRIVE_ROOT = Path(
    "/content/drive/MyDrive/interpret/special-lab-1")
DEFAULT_PHASE3_ROOT = DEFAULT_DRIVE_ROOT / "phase3_20260729"
DEFAULT_PHASE2_ROOT = DEFAULT_DRIVE_ROOT / "part2_20260727"


class UnresolvedArtifact(RuntimeError):
    pass


def run_root(*, create: bool = True) -> Path:
    """Resolve the durable Phase 4 root without a CPU/home fallback."""
    path = Path(os.environ.get("JSPACE4_RUN_ROOT", str(DEFAULT_RUN_ROOT)))
    if create:
        path.mkdir(parents=True, exist_ok=True)
    elif not path.exists():
        raise RuntimeError(
            "Phase 4 run root is absent; set JSPACE4_RUN_ROOT or mount "
            "the campaign Drive")
    return path


def local_work(*, create: bool = True) -> Path:
    path = Path(os.environ.get("JSPACE4_LOCAL_WORK", "/content/sl4_work"))
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


def _drive_root() -> Path:
    return Path(os.environ.get(
        "JSPACE_DRIVE_ROOT", str(DEFAULT_DRIVE_ROOT)))


def _artifact_root(namespace: str) -> Path:
    if namespace == "phase2":
        return Path(os.environ.get(
            "JSPACE_PART2_RUN_ROOT", str(DEFAULT_PHASE2_ROOT)))
    if namespace == "phase3":
        return Path(os.environ.get(
            "JSPACE3_RUN_ROOT", str(DEFAULT_PHASE3_ROOT)))
    if namespace == "phase4":
        return run_root(create=False)
    raise UnresolvedArtifact(
        f"unknown artifact namespace {namespace!r}")


def resolve_uri(uri: str | Path, *, must_exist: bool = True) -> Path:
    """Resolve repo://, drive://, artifact://, and revision-pinned model://."""
    text = str(uri)
    if "://" not in text:
        path = Path(text)
    else:
        scheme, rest = text.split("://", 1)
        if scheme == "repo":
            path = REPO_ROOT / rest
        elif scheme == "drive":
            path = _drive_root() / rest
        elif scheme == "artifact":
            namespace, separator, relative = rest.partition("/")
            if not separator or namespace not in {
                    "phase2", "phase3", "phase4"}:
                raise UnresolvedArtifact(
                    f"artifact URI needs phase2/phase3/phase4 namespace: "
                    f"{text!r}")
            path = _artifact_root(namespace) / relative
        elif scheme == "model":
            return _resolve_model(rest, must_exist=must_exist)
        else:
            raise UnresolvedArtifact(
                f"unsupported Phase 4 URI scheme {scheme!r}")
    if must_exist and not path.exists():
        raise UnresolvedArtifact(f"artifact does not exist: {text!r} -> {path}")
    return path


def _resolve_model(rest: str, *, must_exist: bool) -> Path:
    ref, separator, revision = rest.rpartition("@")
    if not separator or not ref or not revision:
        raise UnresolvedArtifact(
            f"model URI must pin an exact revision: model://{rest}")
    cache_name = "models--" + ref.replace("/", "--")
    caches = [
        os.environ.get("HF_HUB_CACHE", ""),
        "/content/hf_local",
    ]
    for cache in caches:
        if not cache:
            continue
        if str(Path(cache)).startswith("/content/drive/"):
            raise UnresolvedArtifact(
                "HF_HUB_CACHE points at DriveFS; copy or download the "
                "pinned snapshot to local NVMe before model load")
        candidate = Path(cache) / cache_name / "snapshots" / revision
        if candidate.exists():
            return candidate
    destination = Path("/content/models") / ref.rsplit("/", 1)[-1]
    if not must_exist:
        return destination
    raise UnresolvedArtifact(
        f"pinned model {ref}@{revision} is not on local NVMe; download it "
        "before model load and never stream weights through DriveFS")
