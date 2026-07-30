# Run-root indirection (nextsteps §2.8 / §13.2 / §15.2).
#
# THE RULE: no Phase 3 module hardcodes a Drive or VM path. Every producer
# resolves its output root through run_root(), every heavy input through
# jspace_part2.paths logical URIs (drive:// model:// repo:// jlens://).
# A clean-room reproduction sets JSPACE3_RUN_ROOT to an empty directory
# and can neither read from nor write to the original outputs.
from __future__ import annotations

import os
from pathlib import Path

# Phase 2 read-only roots resolve through the existing URI machinery.
from jspace_part2.paths import resolve as resolve_uri  # noqa: F401  (re-export)
from jspace_part2.paths import to_uri  # noqa: F401  (re-export)

_DEFAULT_RUN_ROOTS = [
    "/content/drive/MyDrive/interpret/special-lab-1/phase3_20260729",
    str(Path.home() / "jspace3-run-root"),
]


def run_root(*, create: bool = True) -> Path:
    """The Phase 3 heavy-artifact root. Resolution order:
    JSPACE3_RUN_ROOT env var, then the campaign Drive root, then a home
    fallback (so CPU-only environments still work)."""
    env = os.environ.get("JSPACE3_RUN_ROOT")
    candidates = [env] if env else _DEFAULT_RUN_ROOTS
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if p.exists() or (create and _creatable(p)):
            if create:
                p.mkdir(parents=True, exist_ok=True)
            return p
    raise RuntimeError(
        "no writable Phase 3 run root: set JSPACE3_RUN_ROOT or mount Drive")


def _creatable(p: Path) -> bool:
    parent = p
    while not parent.exists():
        parent = parent.parent
    return os.access(parent, os.W_OK)


def metrics_dir(slug: str) -> Path:
    d = run_root() / "metrics" / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def figures_dir() -> Path:
    d = run_root() / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def lens_dir() -> Path:
    d = run_root() / "lens"
    d.mkdir(parents=True, exist_ok=True)
    return d


def manifests_dir() -> Path:
    d = run_root() / "manifests"
    d.mkdir(parents=True, exist_ok=True)
    return d


def local_work() -> Path:
    """Fast local scratch for checkpoints (never the durable copy)."""
    p = Path(os.environ.get("JSPACE3_LOCAL_WORK", "/content/sl3_work"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def phase2_run_root() -> Path:
    """The FROZEN Phase 2 run root (read-only inputs). Reuses the Phase 2
    producers' own env override so clean-room runs redirect both layers
    with one variable."""
    return Path(os.environ.get(
        "JSPACE_PART2_RUN_ROOT",
        "/content/drive/MyDrive/interpret/special-lab-1/part2_20260727"))


def drive_hub_cache() -> Path:
    """The persistent Drive HF cache (pinned dataset snapshots live here;
    a clean-room reproduction points JSPACE3_DRIVE_HUB_CACHE at its own
    copy). Distinct from HF_HUB_CACHE, the fast local weight cache."""
    return Path(os.environ.get(
        "JSPACE3_DRIVE_HUB_CACHE", "/content/drive/MyDrive/hf_cache/hub"))


def local_hub_cache() -> Path:
    """The local-NVMe HF cache (weights + small pinned datasets)."""
    return Path(os.environ.get("HF_HUB_CACHE", "/content/hf_local"))
