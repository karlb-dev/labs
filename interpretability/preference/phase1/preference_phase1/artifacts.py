"""Atomic and durable artifact IO.

Discipline (jspaces DRIVEFS_DURABILITY_PLAN): local NVMe first with
same-directory temp file + atomic rename; hash locally; Drive receives a
verified copy via temp-sibling + rehash + atomic rename. DriveFS is a
delivery mirror, never the only copy.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import tempfile
from typing import Any, Iterable, Mapping

from .canonical import canonical_json, sha256_file


def ensure_dir(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_bytes(path: pathlib.Path, data: bytes) -> pathlib.Path:
    ensure_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def atomic_write_text(path: pathlib.Path, text: str) -> pathlib.Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: pathlib.Path, payload: Any, *, indent: int | None = 2) -> pathlib.Path:
    if indent is None:
        text = canonical_json(payload)
    else:
        text = json.dumps(payload, indent=indent, sort_keys=True, ensure_ascii=False,
                          default=_json_default)
    return atomic_write_text(path, text + "\n")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pathlib.Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def append_jsonl(path: pathlib.Path, rows: Iterable[Mapping[str, Any]], *, fsync: bool = True) -> int:
    """Append rows as canonical JSONL; fsync so a preempted VM loses nothing
    already appended. Returns the number of rows written."""
    ensure_dir(path.parent)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(canonical_json(dict(row)) + "\n")
            n += 1
        f.flush()
        if fsync:
            os.fsync(f.fileno())
    return n


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def durable_copy(src: pathlib.Path, dest: pathlib.Path) -> dict[str, Any]:
    """Hash-verified copy with atomic landing (for Drive mirroring).

    Returns {src, dest, sha256, bytes}. Raises if the rehash mismatches.
    """
    src = pathlib.Path(src)
    dest = pathlib.Path(dest)
    ensure_dir(dest.parent)
    src_hash = sha256_file(src)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
    os.close(fd)
    try:
        shutil.copyfile(src, tmp)
        got = sha256_file(tmp)
        if got != src_hash:
            raise RuntimeError(f"durable_copy hash mismatch for {src} -> {dest}")
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return {"src": str(src), "dest": str(dest), "sha256": src_hash,
            "bytes": dest.stat().st_size}


def write_csv(path: pathlib.Path, rows: Iterable[Mapping[str, Any]]) -> pathlib.Path:
    """Deterministic CSV: header = union of keys in first-seen order
    (bench semantics), atomic landing."""
    import csv
    import io

    rows = [dict(r) for r in rows]
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _csv_cell(v) for k, v in row.items()})
    return atomic_write_text(path, buf.getvalue())


def _csv_cell(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, tuple)):
        return "|".join(str(v) for v in value)
    return value


def mirror_tree(src_dir: pathlib.Path, dest_dir: pathlib.Path,
                *, exclude_names: tuple[str, ...] = (".git", "__pycache__")) -> int:
    """Recursive durable mirror; skips files whose size+mtime already match.

    Small campaign trees only (reports, run dirs). Weights never travel here.
    """
    src_dir = pathlib.Path(src_dir)
    count = 0
    for src in sorted(src_dir.rglob("*")):
        if any(part in exclude_names for part in src.parts):
            continue
        rel = src.relative_to(src_dir)
        dest = dest_dir / rel
        if src.is_dir():
            ensure_dir(dest)
            continue
        if dest.exists():
            s, d = src.stat(), dest.stat()
            if s.st_size == d.st_size and int(s.st_mtime) <= int(d.st_mtime):
                continue
        durable_copy(src, dest)
        count += 1
    return count
