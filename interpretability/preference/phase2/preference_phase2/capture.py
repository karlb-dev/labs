"""Sharded, manifested capture storage (plan §35; addendum C2 — the
first verifiable capture seal).

Layout under a run dir:

    state/captures/{site}-{shard:04d}.pt   {item_id: {depth: bf16 tensor}}
    state/captures_manifest.json           per-shard sha256 + item ids

Shards are written atomically, hashed on close, and verified on load.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Iterator

from .artifacts import atomic_write_json, ensure_dir
from .canonical import sha256_file

SHARD_ROWS = 256


class CaptureWriter:
    def __init__(self, run_dir: pathlib.Path):
        self.dir = ensure_dir(pathlib.Path(run_dir) / "state" / "captures")
        self.manifest_path = pathlib.Path(run_dir) / "state" / "captures_manifest.json"
        self.manifest: dict[str, Any] = {"shards": [], "dtype": "bfloat16"}
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())
        self._open: dict[str, dict[str, Any]] = {}
        self._counts: dict[str, int] = {}
        for sh in self.manifest["shards"]:
            site = sh["site"]
            self._counts[site] = max(self._counts.get(site, 0),
                                     sh["shard_index"] + 1)

    def existing_items(self) -> set[str]:
        out: set[str] = set()
        for sh in self.manifest["shards"]:
            out.update(sh["item_ids"])
        return out

    def add(self, item_id: str, site_store: dict[str, dict[int, Any]]) -> None:
        for site, depths in site_store.items():
            buf = self._open.setdefault(site, {})
            buf[item_id] = depths
            if len(buf) >= SHARD_ROWS:
                self._flush_site(site)

    def _flush_site(self, site: str) -> None:
        import torch

        buf = self._open.pop(site, {})
        if not buf:
            return
        idx = self._counts.get(site, 0)
        self._counts[site] = idx + 1
        path = self.dir / f"{site}-{idx:04d}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(buf, tmp)
        tmp.replace(path)
        depths = sorted({int(d) for v in buf.values() for d in v})
        self.manifest["shards"].append({
            "site": site, "shard_index": idx, "file": path.name,
            "sha256": sha256_file(path), "item_ids": sorted(buf),
            "depths": depths, "rows": len(buf),
        })
        atomic_write_json(self.manifest_path, self.manifest)

    def close(self) -> None:
        for site in list(self._open):
            self._flush_site(site)


def verify_manifest(run_dir: pathlib.Path) -> dict[str, Any]:
    run_dir = pathlib.Path(run_dir)
    manifest = json.loads((run_dir / "state" / "captures_manifest.json")
                          .read_text())
    bad = []
    for sh in manifest["shards"]:
        path = run_dir / "state" / "captures" / sh["file"]
        if not path.exists():
            bad.append(f"missing {sh['file']}")
        elif sha256_file(path) != sh["sha256"]:
            bad.append(f"sha mismatch {sh['file']}")
    return {"n_shards": len(manifest["shards"]),
            "rows": sum(sh["rows"] for sh in manifest["shards"]),
            "failures": bad, "passed": not bad}


class CaptureReader:
    """Lazy loader: state vectors by (item_id, site, depth), float32."""

    def __init__(self, run_dir: pathlib.Path, *, verify: bool = True):
        self.run_dir = pathlib.Path(run_dir)
        manifest = json.loads(
            (self.run_dir / "state" / "captures_manifest.json").read_text())
        if verify:
            v = verify_manifest(self.run_dir)
            if not v["passed"]:
                raise RuntimeError(f"capture manifest failed: {v['failures'][:4]}")
        self._index: dict[tuple[str, str], str] = {}
        for sh in manifest["shards"]:
            for iid in sh["item_ids"]:
                self._index[(iid, sh["site"])] = sh["file"]
        self._cache: dict[str, dict] = {}
        self._cache_order: list[str] = []

    def get(self, item_id: str, site: str, depth: int):
        import numpy as np
        import torch

        fname = self._index[(item_id, site)]
        if fname not in self._cache:
            data = torch.load(self.run_dir / "state" / "captures" / fname,
                              map_location="cpu", weights_only=False)
            self._cache[fname] = data
            self._cache_order.append(fname)
            if len(self._cache_order) > 12:
                old = self._cache_order.pop(0)
                self._cache.pop(old, None)
        t = self._cache[fname][item_id][int(depth)]
        return t.float().numpy().astype(np.float32)

    def state_fn(self, site: str, depth: int):
        return lambda item_id: self.get(item_id, site, depth)
