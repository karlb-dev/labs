"""Path resolution for Phase 2 (fork of the Phase 1 module, PREF2_* envs).

Everything is derived from the repo root (found by walking up to ``.git``),
with env-var overrides so a future VM/layout change is a config change, not
a code change. Drive is a delivery mirror, never the primary copy.
No machine-specific absolute path may enter a scientific artifact
(plan §6.3): everything below resolves at call time.
"""

from __future__ import annotations

import os
import pathlib

_ENV_REPO = "PREF2_REPO_ROOT"
_ENV_DRIVE = "PREF2_DRIVE_ROOT"
_ENV_RUN_ROOT = "PREF2_RUN_ROOT"
_ENV_HF_LOCAL = "PREF2_HF_LOCAL"


def repo_root(start: pathlib.Path | None = None) -> pathlib.Path:
    """Repo root: $PREF2_REPO_ROOT, else walk up from this file to `.git`."""
    env = os.environ.get(_ENV_REPO)
    if env:
        return pathlib.Path(env).resolve()
    here = (start or pathlib.Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError(
        "repo root not found (no .git above package); set PREF2_REPO_ROOT"
    )


def interp_root() -> pathlib.Path:
    return repo_root() / "interpretability"


def campaign_root() -> pathlib.Path:
    return interp_root() / "preference"


def phase1_root() -> pathlib.Path:
    """Phase 1 tree — read-only imported boundary (never written)."""
    return campaign_root() / "phase1"


def phase2_root() -> pathlib.Path:
    return campaign_root() / "phase2"


def data_root() -> pathlib.Path:
    return campaign_root() / "data"


def reports_root() -> pathlib.Path:
    return phase2_root() / "reports"


def registry_path() -> pathlib.Path:
    return reports_root() / "evidence_events.jsonl"


def configs_root() -> pathlib.Path:
    return phase2_root() / "configs"


def prereg_root() -> pathlib.Path:
    return phase2_root() / "preregistration"


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


def drive_phase_root() -> pathlib.Path | None:
    """Mirror root for phase2; created on demand when Drive exists."""
    root = drive_root()
    if root is None:
        return None
    part = root / "phase2"
    part.mkdir(parents=True, exist_ok=True)
    return part


def hf_local_cache() -> pathlib.Path:
    """Local-NVMe HF cache for model weights (never load through DriveFS)."""
    env = os.environ.get(_ENV_HF_LOCAL)
    path = pathlib.Path(env) if env else pathlib.Path("/content/hf_local")
    path.mkdir(parents=True, exist_ok=True)
    return path
