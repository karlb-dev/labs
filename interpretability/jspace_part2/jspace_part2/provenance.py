# Provenance machinery: every result file carries the exact code, config,
# inputs, and model revisions that produced it (protocol/REPRO_CONTRACT.md).
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .lib import sha256_file

PKG_ROOT = Path(__file__).resolve().parents[1]          # .../jspace_part2 (pkg dir)
REPO_ROOT = PKG_ROOT.parents[1]                          # .../labs
REGISTRY = PKG_ROOT / "reports" / "evidence_registry.jsonl"
STUDY_ID = "jspace-part2"


def git_info(repo: Path = REPO_ROOT) -> dict:
    def _run(*args):
        return subprocess.run(["git", "-C", str(repo), *args],
                              capture_output=True, text=True).stdout.strip()
    commit = _run("rev-parse", "HEAD")
    dirty = bool(_run("status", "--porcelain"))
    return {"code_commit": commit, "dirty_tree": dirty,
            "branch": _run("rev-parse", "--abbrev-ref", "HEAD")}


def require_clean_tree(allow_dirty: bool = False) -> dict:
    info = git_info()
    if info["dirty_tree"] and not allow_dirty:
        raise SystemExit(
            "REFUSING: git tree is dirty. Commit first, or pass "
            "--allow-dirty (output is then disqualified from confirmatory "
            "tier).")
    return info


def resolve_model(model_id_or_path: str) -> dict:
    """Pin a model source: HF cache snapshot dirname IS the revision SHA;
    plain local dirs get a config hash + source note instead."""
    p = Path(model_id_or_path)
    if p.exists():  # plain local dir (e.g. rsync'd weights)
        cfg = p / "config.json"
        return {"id": str(p), "revision": None, "source": "local-dir",
                "config_sha256": sha256_file(cfg) if cfg.exists() else None}
    out = {"id": model_id_or_path, "source": "hf-hub"}
    for cache in (os.environ.get("HF_HUB_CACHE"), "/content/hf_local",
                  "/content/drive/MyDrive/hf_cache/hub"):
        if not cache:
            continue
        snaps = Path(cache) / f"models--{model_id_or_path.replace('/', '--')}" / "snapshots"
        if snaps.exists():
            revs = sorted(d.name for d in snaps.iterdir() if d.is_dir())
            if revs:
                out["revision"] = revs[-1]
                cfg = snaps / revs[-1] / "config.json"
                if cfg.exists():
                    out["config_sha256"] = sha256_file(cfg)
                return out
    try:
        from huggingface_hub import HfApi
        out["revision"] = HfApi().model_info(model_id_or_path).sha
    except Exception:
        out["revision"] = None
    return out


@dataclass
class Provenance:
    evidence_id: str
    tier: str
    command: str
    config_path: str | None = None
    inputs: dict | None = None
    model: dict | None = None
    seed: int | None = None
    jlens_commit: str | None = None
    allow_dirty: bool = False

    def block(self) -> dict:
        from . import __version__
        g = git_info()
        cfg_sha = (sha256_file(self.config_path)
                   if self.config_path and Path(self.config_path).exists()
                   else None)
        return {
            "study_id": STUDY_ID, "evidence_id": self.evidence_id,
            "tier": self.tier, **g,
            "package_version": f"jspace_part2 {__version__}",
            "command": self.command, "config_path": self.config_path,
            "config_sha256": cfg_sha, "inputs": self.inputs or {},
            "model": self.model, "jlens_commit": self.jlens_commit,
            "seed": self.seed,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


def write_result(obj: dict, path: Path, prov: Provenance) -> None:
    obj = dict(obj)
    obj["provenance"] = prov.block()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def registry_append(row: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    row = {"registered_utc":
           time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} | row
    with REGISTRY.open("a") as f:
        f.write(json.dumps(row) + "\n")


def registry_rows() -> list[dict]:
    if not REGISTRY.exists():
        return []
    return [json.loads(l) for l in REGISTRY.read_text().splitlines() if l.strip()]


def registry_find(evidence_id: str) -> dict | None:
    rows = [r for r in registry_rows() if r.get("evidence_id") == evidence_id]
    return rows[-1] if rows else None  # append-only: last row wins
