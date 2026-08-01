"""Run-root and logical artifact URI resolution for Phase 4."""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ROOT = Path(
    "/content/drive/MyDrive/interpret/special-lab-1/phase4_20260731")
DEFAULT_DRIVE_ROOT = Path(
    "/content/drive/MyDrive/interpret/special-lab-1")
DEFAULT_HF_CACHE_ROOT = Path(
    "/content/drive/MyDrive/hf_cache/hub")
DEFAULT_PHASE3_ROOT = DEFAULT_DRIVE_ROOT / "phase3_20260729"
DEFAULT_PHASE2_ROOT = DEFAULT_DRIVE_ROOT / "part2_20260727"
DRIVE_ALIASES = {
    "part1": "2026-07-25_1726",
    "part1v2": "2026-07-26_v2",
    "part2": "part2_20260727",
    "phase3": "phase3_20260729",
    "phase4": "phase4_20260731",
}


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


def _hf_cache_root() -> Path:
    return Path(os.environ.get(
        "JSPACE_HF_CACHE_ROOT", str(DEFAULT_HF_CACHE_ROOT)))


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
    """Resolve Phase 4 logical URIs, including pinned dataset cache inputs."""
    text = str(uri)
    if "://" not in text:
        path = Path(text)
    else:
        scheme, rest = text.split("://", 1)
        if scheme == "repo":
            path = REPO_ROOT / rest
        elif scheme == "drive":
            head, separator, tail = rest.partition("/")
            if head in DRIVE_ALIASES:
                rest = DRIVE_ALIASES[head] + (
                    f"/{tail}" if separator else "")
            path = _drive_root() / rest
        elif scheme == "artifact":
            namespace, separator, relative = rest.partition("/")
            if not separator or namespace not in {
                    "phase2", "phase3", "phase4"}:
                raise UnresolvedArtifact(
                    f"artifact URI needs phase2/phase3/phase4 namespace: "
                    f"{text!r}")
            path = _artifact_root(namespace) / relative
        elif scheme == "hf-cache":
            path = _resolve_hf_dataset_cache(rest)
        elif scheme == "model":
            return _resolve_model(rest, must_exist=must_exist)
        else:
            raise UnresolvedArtifact(
                f"unsupported Phase 4 URI scheme {scheme!r}")
    if must_exist and not path.exists():
        raise UnresolvedArtifact(f"artifact does not exist: {text!r} -> {path}")
    return path


def _resolve_hf_dataset_cache(rest: str) -> Path:
    """Resolve only revision-pinned dataset snapshots from an HF cache.

    Model snapshots are deliberately excluded because model-scale inputs
    must live on local NVMe rather than being streamed through DriveFS.
    """
    parts = Path(rest).parts
    if (
        rest.startswith("/")
        or any(part in {".", ".."} for part in parts)
        or len(parts) < 4
        or not parts[0].startswith("datasets--")
        or parts[1] != "snapshots"
        or re.fullmatch(r"[0-9a-f]{40}", parts[2]) is None
    ):
        raise UnresolvedArtifact(
            "hf-cache URI must name a revision-pinned dataset snapshot: "
            "hf-cache://datasets--ORG--NAME/snapshots/<40-hex-revision>/...")
    return _hf_cache_root().joinpath(*parts)


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


def materialize_local_file(uri: str | Path, *,
                           expected_sha256: str) -> Path:
    """Verify and copy a large immutable input from DriveFS to local NVMe.

    A previously verified local materialization remains usable when its
    canonical DriveFS path is temporarily unavailable.  The logical URI still
    determines the filename, and the content-addressed directory plus a fresh
    hash check preserve the pinned-input contract.
    """
    from .manifests import file_sha256

    if not expected_sha256 or len(expected_sha256) != 64:
        raise ValueError("a full expected SHA-256 is required")
    source = resolve_uri(uri, must_exist=False)
    destination = (
        local_work() / "inputs" / expected_sha256 / source.name)
    if destination.exists():
        if file_sha256(destination) != expected_sha256:
            raise UnresolvedArtifact(
                f"local materialization hash mismatch: {destination}")
        return destination
    source = resolve_uri(uri)
    if file_sha256(source) != expected_sha256:
        raise UnresolvedArtifact(
            f"source hash does not match pinned input: {uri}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(
        destination.suffix + f".tmp{os.getpid()}")
    shutil.copyfile(source, temporary)
    if file_sha256(temporary) != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise UnresolvedArtifact(
            f"copied input hash mismatch: {destination}")
    os.replace(temporary, destination)
    return destination
