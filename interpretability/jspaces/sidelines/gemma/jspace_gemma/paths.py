"""Run-root and immutable-artifact path handling for the Gemma side track."""
from __future__ import annotations

import os
import re
from pathlib import Path

def _find_repo_root(start: Path | None = None) -> Path:
    """Locate the monorepo root by walking parents for a .git entry.

    Depth-encoded parents[N] breaks when packages move; this works on
    Windows, Colab, and nested layouts. Vendored identically into each
    package paths module (no cross-package import).
    """
    cur = (start or Path(__file__)).resolve()
    if cur.is_file():
        cur = cur.parent
    for p in [cur, *cur.parents]:
        if (p / ".git").exists():
            return p
    raise RuntimeError(
        "cannot locate git repository root from "
        f"{Path(__file__).resolve()}; clone/open the monorepo checkout")


# Old monorepo prefixes → post-reorg locations (longest first).
_REPO_PATH_ALIASES: tuple[tuple[str, str], ...] = (
    ("jspaces/phases/phase4/", "interpretability/jspaces/phases/phase4/"),
    ("jspaces/phases/phase3/", "interpretability/jspaces/phases/phase3/"),
    ("jspaces/phases/phase2/", "interpretability/jspaces/phases/phase2/"),
    ("jspaces/sidelines/olmo/", "interpretability/jspaces/sidelines/olmo/"),
    ("jspaces/sidelines/gemma/", "interpretability/jspaces/sidelines/gemma/"),
    ("jspaces/phases/paper_analysis/", "interpretability/jspaces/phases/paper_analysis/"),
    ("jspaces/phases/phase1/part2_exploratory/", "interpretability/jspaces/phases/phase1/part2_exploratory/"),
    ("jspaces/phases/phase1/", "interpretability/jspaces/phases/phase1/"),
    ("jspaces/analysis_phase/", "interpretability/jspaces/phases/paper_analysis/"),
    ("jspaces/archive/phase1_nested_part2/", "interpretability/jspaces/phases/phase1/part2_exploratory/"),
    ("interpretability/jspace_phase4/", "interpretability/jspaces/phases/phase4/"),
    ("interpretability/jspace_phase3/", "interpretability/jspaces/phases/phase3/"),
    ("interpretability/jspace_part2/", "interpretability/jspaces/phases/phase2/"),
    ("interpretability/jspace_olmo_lineage/", "interpretability/jspaces/sidelines/olmo/"),
    ("interpretability/jspace_gemma/", "interpretability/jspaces/sidelines/gemma/"),
    ("interpretability/jspace_paper/", "interpretability/jspaces/phases/paper_analysis/"),
    ("interpretability/jspace/part2/", "interpretability/jspaces/phases/phase1/part2_exploratory/"),
    ("interpretability/jspace/", "interpretability/jspaces/phases/phase1/"),
)


def _rewrite_repo_relative(rel: str) -> str:
    text = str(rel).replace("\\", "/").lstrip("/")
    for old, new in _REPO_PATH_ALIASES:
        if text.startswith(old):
            return new + text[len(old):]
        if text == old.rstrip("/"):
            return new.rstrip("/")
    return text


REPO_ROOT = _find_repo_root()
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
            rewritten = _rewrite_repo_relative(rest)
            path = REPO_ROOT / rewritten
            if not path.exists() and rewritten != rest:
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
