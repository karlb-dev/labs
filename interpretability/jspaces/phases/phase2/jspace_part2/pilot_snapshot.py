"""N0 pilot-state snapshot: make the end-of-VM7 branch state a recoverable checkpoint.

Two phases (both idempotent, both CPU):
  python -m jspace_part2.pilot_snapshot --small    # git-tree + small Drive artifacts
  python -m jspace_part2.pilot_snapshot --lenses   # chunked manifests for lens *.pt (slow DriveFS reads)

Writes reports/PILOT_SNAPSHOT_VM7.json (summary + small-file hashes) and
reports/PILOT_SNAPSHOT_VM7_lenses.jsonl (one chunked manifest per large artifact).
Never modifies any pilot artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

def _find_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path(__file__)).resolve()
    if cur.is_file():
        cur = cur.parent
    for p in [cur, *cur.parents]:
        if (p / ".git").exists():
            return p
    raise RuntimeError("cannot locate git repository root")


REPO = _find_repo_root()
PKG = REPO / "interpretability/jspaces/phases/phase2"
MIRROR = REPO / "interpretability/jspaces/phases/phase1/part2_exploratory"
RUNDIR = Path("/content/drive/MyDrive/interpret/special-lab-1/part2_20260727")
OUT = PKG / "reports" / "PILOT_SNAPSHOT_VM7.json"
OUT_LENSES = PKG / "reports" / "PILOT_SNAPSHOT_VM7_lenses.jsonl"
CHUNK = 256 * 1024 * 1024  # 256 MiB

TAG = "jspace-part2-pilot-vm7"


def sha256_file(p: Path, chunked: bool = False):
    h = hashlib.sha256()
    chunks = []
    with open(p, "rb") as f:
        while True:
            ch = hashlib.sha256()
            n = 0
            while n < CHUNK:
                buf = f.read(min(8 * 1024 * 1024, CHUNK - n))
                if not buf:
                    break
                ch.update(buf)
                h.update(buf)
                n += len(buf)
            if n:
                chunks.append({"bytes": n, "sha256": ch.hexdigest()})
            if n < CHUNK:
                break
    out = {"sha256": h.hexdigest(), "bytes": p.stat().st_size}
    if chunked:
        out["chunk_bytes"] = CHUNK
        out["chunks"] = chunks
    return out


def hash_tree(root: Path, exclude_parts=("__pycache__", ".git", "egg-info")):
    rows = {}
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            continue
        if any(x in str(p) for x in exclude_parts):
            continue
        rows[str(p.relative_to(root))] = sha256_file(p)["sha256"]
    return rows


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()


def phase_small():
    commit = git("rev-parse", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    reg = PKG / "reports" / "evidence_registry.jsonl"
    snap = {
        "snapshot": "PILOT_SNAPSHOT_VM7",
        "purpose": "N0 freeze of the end-of-VM7 pilot state (nextsteps_2_2 stage N0)",
        "branch": branch,
        "commit": commit,
        "tag": TAG,
        "registry": {
            "path": str(reg.relative_to(REPO)),
            **sha256_file(reg),
            "rows": sum(1 for _ in open(reg)),
        },
        "preregistration_draft_status": "DRAFT_V2_PRE_REVIEW",
        "git_trees": {
            "jspace_part2_package": hash_tree(PKG),
            "jspace_part2_mirror": hash_tree(MIRROR),
        },
        "run_dir_small": {},
        "lens_manifests": "see PILOT_SNAPSHOT_VM7_lenses.jsonl (phase --lenses)",
    }
    # headline doc hashes, called out for quick reference
    for label, p in {
        "report": MIRROR / "report" / "REPORT_PART2.md",
        "handout_pdf": MIRROR / "handout" / "jspace_part2_handout.pdf",
        "handout_tex": MIRROR / "handout" / "jspace_part2_handout.tex",
        "prereg_draft": PKG / "preregistration" / "SCIENTIFIC_PREREGISTRATION_DRAFT.md",
    }.items():
        if p.exists():
            snap[f"{label}_sha256"] = sha256_file(p)["sha256"]
    # small run-dir artifacts: metrics, figures, report, config, manifests (skip lens/ and logs/)
    for sub in ("metrics", "figures", "report", "config", "manifests"):
        d = RUNDIR / sub
        if d.exists():
            snap["run_dir_small"][sub] = hash_tree(d)
    OUT.write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n")
    n_small = sum(len(v) for v in snap["run_dir_small"].values())
    print(f"wrote {OUT}  commit={commit[:9]} registry_rows={snap['registry']['rows']} "
          f"git_files={len(snap['git_trees']['jspace_part2_package']) + len(snap['git_trees']['jspace_part2_mirror'])} "
          f"rundir_small={n_small}")


def phase_lenses():
    lens_dir = RUNDIR / "lens"
    done = set()
    if OUT_LENSES.exists():
        done = {json.loads(l)["path"] for l in open(OUT_LENSES)}
    with open(OUT_LENSES, "a") as f:
        for p in sorted(lens_dir.glob("*.pt")):
            rel = str(p.relative_to(RUNDIR))
            if rel in done:
                print(f"skip (done) {rel}")
                continue
            row = {"path": rel, "kind": "lens_artifact", **sha256_file(p, chunked=True)}
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            print(f"hashed {rel}  {row['bytes']/1e9:.2f} GB  {row['sha256'][:16]}")
    print(f"lens manifests complete -> {OUT_LENSES}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--lenses", action="store_true")
    a = ap.parse_args()
    if not (a.small or a.lenses):
        sys.exit("pick --small and/or --lenses")
    if a.small:
        phase_small()
    if a.lenses:
        phase_lenses()
