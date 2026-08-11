"""Stage Muse weights on local NVMe and hash the snapshot."""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..paths import (
    DRIVE_ROOT,
    HF_LOCAL,
    MUSE_MODEL_ID,
    MUSE_MODEL_REVISION,
    ensure_dirs,
    model_snapshot,
)
from ..registry import register
from ..util import atomic_write_json, log, runtime_fingerprint, sha256_file, utc_now


def stage(force: bool = False) -> dict:
    """Download pinned Muse snapshot if needed; write hash manifest."""
    ensure_dirs()
    from huggingface_hub import snapshot_download

    token_path = Path.home() / ".cache/huggingface/token"
    token = token_path.read_text().strip() if token_path.exists() else None
    log(f"staging {MUSE_MODEL_ID}@{MUSE_MODEL_REVISION[:12]} -> {HF_LOCAL}")
    path = snapshot_download(
        MUSE_MODEL_ID,
        revision=MUSE_MODEL_REVISION,
        cache_dir=str(HF_LOCAL),
        token=token,
        max_workers=8,
    )
    snap = Path(path)
    assert snap == model_snapshot() or snap.resolve() == model_snapshot().resolve()

    files = []
    for p in sorted(snap.rglob("*")):
        if p.is_file() and p.name != "evidence_events.jsonl":
            rel = str(p.relative_to(snap))
            files.append({
                "path": rel,
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })
    total = sum(f["bytes"] for f in files)
    manifest = {
        "evidence_id": "muse-stage-v1",
        "model_id": MUSE_MODEL_ID,
        "revision": MUSE_MODEL_REVISION,
        "snapshot": str(snap),
        "n_files": len(files),
        "total_bytes": total,
        "files": files,
        "runtime": runtime_fingerprint(),
        "utc": utc_now(),
    }
    out = DRIVE_ROOT / "manifests" / "muse_stage_manifest.json"
    if out.exists() and not force:
        log(f"manifest exists: {out}")
    else:
        atomic_write_json(manifest, out)
        register({
            "evidence_id": "muse-stage-v1",
            "what": f"Pinned Muse Glimmer snapshot staged ({total/1e9:.1f} GB)",
            "command": "python -m jspace_muse.experiments.stage",
            "outputs": [out],
        })
        log(f"wrote {out} ({total/1e9:.2f} GB, {len(files)} files)")
    return manifest


if __name__ == "__main__":
    stage(force="--force" in __import__("sys").argv)
