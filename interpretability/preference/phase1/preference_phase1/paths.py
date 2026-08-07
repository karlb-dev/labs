"""Path resolution for the preference campaign.

Everything is derived from the repo root (found by walking up to ``.git``),
with env-var overrides so a future VM/layout change is a config change, not
a code change. Drive is a delivery mirror, never the primary copy
(jspaces DRIVEFS_DURABILITY_PLAN discipline).
"""

from __future__ import annotations

import os
import pathlib

_ENV_REPO = "PREF1_REPO_ROOT"
_ENV_DRIVE = "PREF1_DRIVE_ROOT"
_ENV_RUN_ROOT = "PREF1_RUN_ROOT"
_ENV_HF_LOCAL = "PREF1_HF_LOCAL"


def repo_root(start: pathlib.Path | None = None) -> pathlib.Path:
    """Repo root: $PREF1_REPO_ROOT, else walk up from this file to `.git`."""
    env = os.environ.get(_ENV_REPO)
    if env:
        return pathlib.Path(env).resolve()
    here = (start or pathlib.Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError(
        "repo root not found (no .git above package); set PREF1_REPO_ROOT"
    )


def interp_root() -> pathlib.Path:
    return repo_root() / "interpretability"


def campaign_root() -> pathlib.Path:
    return interp_root() / "preference"


def phase1_root() -> pathlib.Path:
    return campaign_root() / "phase1"


def data_root() -> pathlib.Path:
    return campaign_root() / "data"


def reports_root() -> pathlib.Path:
    return phase1_root() / "reports"


def registry_path() -> pathlib.Path:
    return reports_root() / "evidence_events.jsonl"


def configs_root() -> pathlib.Path:
    return phase1_root() / "configs"


def runs_root() -> pathlib.Path:
    """Live run dirs (gitignored globally via `runs/`); Drive-mirrored."""
    env = os.environ.get(_ENV_RUN_ROOT)
    if env:
        return pathlib.Path(env).resolve()
    return interp_root() / "runs"


def drive_root() -> pathlib.Path | None:
    """Drive campaign folder, or None when Drive is not mounted."""
    env = os.environ.get(_ENV_DRIVE)
    root = pathlib.Path(env) if env else pathlib.Path(
        "/content/drive/MyDrive/preference"
    )
    return root if root.is_dir() else None


def drive_part_root() -> pathlib.Path | None:
    """Mirror root for this phase/part; created on demand when Drive exists."""
    root = drive_root()
    if root is None:
        return None
    part = root / "phase1" / "part1"
    part.mkdir(parents=True, exist_ok=True)
    return part


def hf_local_cache() -> pathlib.Path:
    """Local-NVMe HF cache for model weights (never load through DriveFS)."""
    env = os.environ.get(_ENV_HF_LOCAL)
    path = pathlib.Path(env) if env else pathlib.Path("/content/hf_local")
    path.mkdir(parents=True, exist_ok=True)
    return path
