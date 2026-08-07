"""Git / environment / GPU provenance for manifests and registry events."""

from __future__ import annotations

import datetime as _dt
import os
import pathlib
import platform
import subprocess
from typing import Any

from . import paths


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(paths.repo_root()), *args], text=True
    ).strip()


def git_info() -> dict[str, Any]:
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_tree": bool(status),
        "dirty_paths": sorted(
            line[3:] for line in status.splitlines() if line.strip()
        )[:200],
    }


def env_audit(include_gpu: bool = True) -> dict[str, Any]:
    """Session environment snapshot (addendum C1 gpu_env contract)."""
    audit: dict[str, Any] = {
        "captured_utc": utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "packages": {},
    }
    for mod in ("torch", "transformers", "tokenizers", "numpy", "pandas",
                "scipy", "matplotlib", "huggingface_hub"):
        try:
            audit["packages"][mod] = __import__(mod).__version__
        except Exception:
            audit["packages"][mod] = None
    if include_gpu:
        audit["gpu"] = gpu_info()
    return audit


def gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"cuda_available": False}
    try:
        import torch

        info["torch_cuda_build"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["device_name"] = torch.cuda.get_device_name(0)
            info["device_count"] = torch.cuda.device_count()
            info["capability"] = list(torch.cuda.get_device_capability(0))
            free, total = torch.cuda.mem_get_info(0)
            info["mem_total_gb"] = round(total / 2**30, 2)
            info["mem_free_gb"] = round(free / 2**30, 2)
    except Exception as exc:  # torch absent on pure-CPU sessions is fine
        info["error"] = repr(exc)
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        info["driver_version"] = smi.splitlines()[0] if smi else None
    except Exception:
        info["driver_version"] = None
    return info


def require_cuda() -> None:
    """Hard GPU gate for model-tier-b/c stages (resume-doc invariant: a
    sandboxed CUDA failure means relaunch with host GPU access, never a
    silent CPU fallback)."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "HARD STOP: CUDA not visible in this execution context. "
            "Relaunch with host GPU access; CPU fallback is forbidden for "
            "model-scale stages (preference_resume.md §2)."
        )


def session_id() -> str:
    return os.environ.get("PREF1_SESSION_ID", "session-" + utc_now())


def append_session_log(entry: str) -> pathlib.Path:
    """Human-readable per-session log (addendum C1)."""
    log = paths.reports_root() / "session_log.md"
    log.parent.mkdir(parents=True, exist_ok=True)
    stamp = utc_now()
    with log.open("a", encoding="utf-8") as f:
        f.write(f"- {stamp} · {entry}\n")
    return log
