"""Strict path and logical-URI handling for the isolated side track."""
from __future__ import annotations

import os
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
SPECIAL_LAB_ROOT = Path(
    "/content/drive/MyDrive/interpret/special-lab-1")
DEFAULT_RUN_ROOT = SPECIAL_LAB_ROOT / "olmo_lineage_20260801"
STUDY2_RUN_ROOT = SPECIAL_LAB_ROOT / "olmo_lineage_2_20260803"
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
    env = os.environ.get("JSPACE_OLMO_RUN_ROOT")
    path = Path(env) if env else DEFAULT_RUN_ROOT
    if not path.name.startswith("olmo_lineage_"):
        raise PathBoundaryError(
            "OLMo output root must be an olmo_lineage_* directory "
            f"(got {path.name!r}); set JSPACE_OLMO_RUN_ROOT")
    # Default path must stay under SPECIAL_LAB_ROOT; env override is free
    # for non-Colab hosts (Windows, laptop mirrors).
    if env is None and not _within(path, SPECIAL_LAB_ROOT):
        raise PathBoundaryError(
            "OLMo default output root must be an olmo_lineage_* directory "
            f"below {SPECIAL_LAB_ROOT}; set JSPACE_OLMO_RUN_ROOT to override")
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
            rewritten = _rewrite_repo_relative(rest)
            path = REPO_ROOT / rewritten
            if not path.exists() and rewritten != rest:
                path = REPO_ROOT / rest
        elif scheme == "drive":
            head, separator, tail = rest.partition("/")
            mapped = DRIVE_ALIASES.get(head, head)
            drive_root = Path(os.environ.get(
                "JSPACE_DRIVE_ROOT", str(SPECIAL_LAB_ROOT)))
            path = drive_root / mapped
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
