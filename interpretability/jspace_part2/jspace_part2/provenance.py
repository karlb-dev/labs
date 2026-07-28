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
    plain local dirs get a config hash + source note instead.

    N1.3 repair: a local dir no longer records `revision: null` and stops.
    Its config hash is matched against every snapshot of every candidate
    HF cache, so an rsync'd weights dir resolves back to the revision it
    came from (nextsteps_2_2 §2.4). Cache snapshots are matched by hash
    too, not by lexicographic recency."""
    p = Path(model_id_or_path)
    if p.exists():  # plain local dir (e.g. rsync'd weights)
        cfg = p / "config.json"
        cfg_sha = sha256_file(cfg) if cfg.exists() else None
        out = {"id": str(p), "revision": None, "source": "local-dir",
               "config_sha256": cfg_sha}
        if cfg_sha:
            hit = _find_revision_by_config(cfg_sha)
            if hit:
                out["revision"], out["revision_source"] = hit
                out["hub_id"] = _hub_id_from_cache_dir(hit[1])
        out["tokenizer_sha256"] = (sha256_file(p / "tokenizer.json")
                                   if (p / "tokenizer.json").exists() else None)
        idx = p / "model.safetensors.index.json"
        out["weight_index_sha256"] = sha256_file(idx) if idx.exists() else None
        return out
    out = {"id": model_id_or_path, "source": "hf-hub"}
    for cache in _caches():
        snaps = Path(cache) / f"models--{model_id_or_path.replace('/', '--')}" / "snapshots"
        if snaps.exists():
            revs = sorted(d.name for d in snaps.iterdir() if d.is_dir())
            if revs:
                out["revision"] = revs[-1]
                out["revision_ambiguous"] = len(revs) > 1
                out["revisions_present"] = revs
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


def _caches() -> list[str]:
    return [c for c in (os.environ.get("HF_HUB_CACHE"), "/content/hf_local",
                        "/content/drive/MyDrive/hf_cache/hub") if c]


def _hub_id_from_cache_dir(cache_snapshot_dir: str) -> str | None:
    for part in Path(cache_snapshot_dir).parts:
        if part.startswith("models--"):
            return part[len("models--"):].replace("--", "/", 1)
    return None


def _find_revision_by_config(cfg_sha: str) -> tuple[str, str] | None:
    """(revision, snapshot_dir) of the cached snapshot whose config.json
    hashes to cfg_sha."""
    for cache in _caches():
        root = Path(cache)
        if not root.exists():
            continue
        for snaps in root.glob("models--*/snapshots"):
            for rev in snaps.iterdir():
                cfg = rev / "config.json"
                if cfg.exists() and sha256_file(cfg) == cfg_sha:
                    return rev.name, str(rev)
    return None


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


def write_result_v2(payload: dict, path: Path, prov: Provenance) -> dict:
    """N1.3 — deterministic science and volatile provenance in separate
    envelopes, so an exact rerun reproduces `payload_sha256` byte for byte.

    v1's `write_result` embedded `created_utc` in the very file whose
    sha256 was registered, which made the contract's central promise
    (rerun -> same hash) impossible to satisfy. Historical producers keep
    calling `write_result`; everything new calls this.

    Returns the envelope written (includes payload_sha256)."""
    from .registry import payload_sha256
    env = {"schema_version": 2, "payload": payload,
           "payload_sha256": payload_sha256(payload),
           "provenance": prov.block()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(env, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return env


def verify_result_v2(path: Path) -> dict:
    """Recompute payload_sha256 from the file's own payload."""
    from .registry import payload_sha256
    env = json.loads(Path(path).read_text())
    if "payload" not in env:
        return {"format": "v1-flat", "ok": None,
                "note": "legacy flat result; only whole-file hash applies"}
    got = payload_sha256(env["payload"])
    return {"format": "v2-envelope", "ok": got == env.get("payload_sha256"),
            "payload_sha256": got, "recorded": env.get("payload_sha256")}


def registry_append(row: dict) -> None:
    """Register evidence. Writes the v2 event log (authoritative) AND the
    v1 JSONL (kept append-compatible so historical readers and
    `repro.sh --verify-only` keep working through the transition)."""
    from . import registry as reg
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    row = {"registered_utc":
           time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} | row
    with REGISTRY.open("a") as f:
        f.write(json.dumps(row) + "\n")
    if row.get("tier") and row.get("command"):
        ev = {k: v for k, v in row.items()
              if k not in ("registered_utc", "superseded_by")}
        ev["event"] = "evidence_created"
        ev.setdefault("code_commit", git_info()["code_commit"])
        reg.append_event(ev)
        if row.get("superseded_by"):
            reg.supersede(row["evidence_id"], row["superseded_by"],
                          reason=row.get("reason", ""))


def registry_rows() -> list[dict]:
    if not REGISTRY.exists():
        return []
    return [json.loads(l) for l in REGISTRY.read_text().splitlines() if l.strip()]


def registry_find(evidence_id: str) -> dict | None:
    rows = [r for r in registry_rows() if r.get("evidence_id") == evidence_id]
    return rows[-1] if rows else None  # append-only: last row wins
