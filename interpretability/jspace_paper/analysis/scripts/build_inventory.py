#!/usr/bin/env python3
"""P1: campaign artifact + paper-source inventory.

Read-only over campaign sources. Emits:
  manifests/source_registry_manifest.json
  manifests/paper_source_manifest.json
  manifests/raw_data_manifest.json
  manifests/campaign_artifact_inventory.{json,parquet}

Hashing policy: every repository source named here is content-hashed.
Drive run roots (~120 GB over DriveFS) are inventoried by structure
(file count, byte total) plus content hashes of their own registered
manifest files; artifact-level hashes remain pinned by the frozen
release/import manifests, which this inventory records rather than
recomputes.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

REPO = Path("/content/labs")
ANALYSIS = REPO / "interpretability/jspace_paper/analysis"
DRIVE = Path("/content/drive/MyDrive/interpret/special-lab-1")

REGISTRIES = {
    "phase4": "interpretability/jspace_phase4/reports/evidence_events.jsonl",
    "gemma": "interpretability/jspace_gemma/reports/evidence_events.jsonl",
    "olmo_lineage": "interpretability/jspace_olmo_lineage/reports/evidence_events.jsonl",
    "phase3": "interpretability/jspace_phase3/reports/evidence_events.jsonl",
    "part2_events": "interpretability/jspace_part2/reports/evidence_events.jsonl",
    "part2_registry": "interpretability/jspace_part2/reports/evidence_registry.jsonl",
}

# Paper/handout/claim sources (plan §4.2): TeX, figure scripts, figure data,
# generated figures, states of record, claim ledgers, skeletons.
PAPER_SOURCE_DIRS = [
    "interpretability/jspace_paper",
    "interpretability/jspace_phase4/paper",
    "interpretability/jspace_phase4/reports/handout",
    "interpretability/jspace_phase4/release",
    "interpretability/jspace_gemma/reports/handout",
    "interpretability/jspace_gemma/release",
    "interpretability/jspace_olmo_lineage/reports/paper",
    "interpretability/jspace_olmo_lineage/release",
]
PAPER_SOURCE_FILES = [
    "interpretability/jspace_phase4/reports/PHASE4_STATE_OF_RECORD.md",
    "interpretability/jspace_phase4/reports/PHASE4_RUNTIME_IDENTITY_SYNTHESIS.md",
    "interpretability/jspace_phase4/reports/PHASE4_TO_PAPER_ANALYSIS_HANDOFF.md",
    "interpretability/jspace_phase4/FREEZE_HANDOFF.md",
    "interpretability/jspace_phase3/reports/PHASE3_STATE_OF_RECORD.md",
    "interpretability/jspace_gemma/reports/GEMMA_TRANSPORT_DEVELOPMENT_REPORT.md",
    "interpretability/jspace_gemma/reports/GEMMA_TRANSPORT_STUDY2_REPORT.md",
    "interpretability/jspace_gemma/release/GEMMA_TRANSPORT_STATE_OF_RECORD_V2.md",
    "interpretability/jspace_olmo_lineage/reports/OLMO_LINEAGE_STATE_OF_RECORD.md",
    "interpretability/jspace_olmo_lineage/reports/OLMO_LINEAGE_STATE_OF_RECORD_V2.md",
    "interpretability/jspace_olmo_lineage/reports/OLMO_LINEAGE_CLAIMS_TABLE.md",
    "interpretability/jspace_olmo_lineage/reports/OLMO_LINEAGE_CLAIMS_TABLE_V2.md",
    "interpretability/jspace_olmo_lineage/reports/OLMO_LINEAGE_DEVELOPMENT_REPORT.md",
    "interpretability/jspace_olmo_lineage/reports/OLMO_LINEAGE_STUDY2_REPORT.md",
]

DRIVE_ROOTS = {
    "phase2": "part2_20260727",
    "phase3": "phase3_20260729",
    "phase4": "phase4_20260731",
    "gemma_study1": "gemma_transport_20260802",
    "gemma_study2": "gemma_transport_2_20260803",
    "olmo_study1": "olmo_lineage_20260801",
    "olmo_study2": "olmo_lineage_2_20260803",
}

KIND_BY_SUFFIX = {
    ".tex": "document_source", ".pdf": "document_pdf", ".md": "document_md",
    ".png": "figure", ".py": "figure_script", ".csv": "figure_data",
    ".json": "manifest_or_data", ".jsonl": "registry_or_data",
    ".sh": "build_script", ".bib": "bibliography",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_tracked() -> set:
    out = subprocess.check_output(["git", "-C", str(REPO), "ls-files"], text=True)
    return set(out.splitlines())


def head_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def build_registry_manifest() -> dict:
    reg = {}
    for name, rel in REGISTRIES.items():
        p = REPO / rel
        events = []
        with open(p) as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                ev = json.loads(line)
                events.append({
                    "line": i,
                    "evidence_id": ev.get("evidence_id"),
                    "event": ev.get("event", "registered"),
                    "utc": ev.get("event_utc") or ev.get("registered_utc"),
                    "tier": ev.get("tier"),
                    "superseded_by": ev.get("superseded_by"),
                })
        reg[name] = {
            "path": rel,
            "sha256": sha256(p),
            "n_events": len(events),
            "events": events,
        }
    return reg


def build_paper_manifest(tracked: set) -> list:
    rows = []
    seen = set()
    paths = []
    for d in PAPER_SOURCE_DIRS:
        base = REPO / d
        if base.exists():
            paths.extend(sorted(q for q in base.rglob("*") if q.is_file()))
    paths.extend(REPO / f for f in PAPER_SOURCE_FILES)
    for p in paths:
        rel = str(p.relative_to(REPO))
        if rel in seen or rel.startswith("interpretability/jspace_paper/analysis/"):
            continue
        seen.add(rel)
        if not p.exists():
            rows.append({"path": rel, "status": "MISSING_EXPECTED_SOURCE"})
            continue
        rows.append({
            "path": rel,
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
            "kind": KIND_BY_SUFFIX.get(p.suffix.lower(), "other"),
            "git_tracked": rel in tracked,
            "status": "present",
        })
    return sorted(rows, key=lambda r: r["path"])


def build_raw_manifest() -> dict:
    roots = {}
    for name, sub in DRIVE_ROOTS.items():
        root = DRIVE / sub
        files = [p for p in root.rglob("*") if p.is_file()]
        manifest_files = []
        for p in sorted(files):
            rel = str(p.relative_to(root))
            # hash this root's own registered manifests + registry snapshots
            if ("manifest" in rel.lower() or rel.endswith(".jsonl")) and p.stat().st_size < 50_000_000 and ("/manifests/" in f"/{rel}" or rel.startswith("manifests/") or "release" in rel.lower() or "registry" in rel.lower()):
                manifest_files.append({
                    "rel_path": rel, "bytes": p.stat().st_size, "sha256": sha256(p)})
        roots[name] = {
            "drive_root": str(root),
            "n_files": len(files),
            "total_bytes": sum(p.stat().st_size for p in files),
            "hashed_manifest_files": manifest_files,
        }
    return roots


def main():
    manifests_dir = ANALYSIS / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    commit = head_commit()
    tracked = git_tracked()

    reg = build_registry_manifest()
    with open(manifests_dir / "source_registry_manifest.json", "w") as f:
        json.dump({"generated_at_commit": commit, "registries": reg}, f,
                  indent=2, sort_keys=True)
        f.write("\n")

    paper = build_paper_manifest(tracked)
    with open(manifests_dir / "paper_source_manifest.json", "w") as f:
        json.dump({"generated_at_commit": commit, "sources": paper}, f,
                  indent=2, sort_keys=True)
        f.write("\n")

    raw = build_raw_manifest()
    with open(manifests_dir / "raw_data_manifest.json", "w") as f:
        json.dump({
            "generated_at_commit": commit,
            "hashing_policy": (
                "Drive roots inventoried by structure; artifact-level hashes "
                "remain pinned by the frozen release/import manifests hashed "
                "here. No raw artifact re-hashed over DriveFS."),
            "roots": raw,
        }, f, indent=2, sort_keys=True)
        f.write("\n")

    # Unified file-level inventory (repo paper sources + registries +
    # hashed Drive manifest files).
    inv_rows = []
    for name, r in reg.items():
        inv_rows.append({
            "logical_uri": f"registry://{name}", "path": r["path"],
            "bytes": (REPO / r["path"]).stat().st_size, "sha256": r["sha256"],
            "artifact_class": "registry", "source": "repo",
            "status": "live", "can_regenerate_without_model": False,
        })
    for row in paper:
        if row.get("status") != "present":
            inv_rows.append({
                "logical_uri": f"repo://{row['path']}", "path": row["path"],
                "bytes": None, "sha256": None, "artifact_class": "missing",
                "source": "repo", "status": row["status"],
                "can_regenerate_without_model": None,
            })
            continue
        inv_rows.append({
            "logical_uri": f"repo://{row['path']}", "path": row["path"],
            "bytes": row["bytes"], "sha256": row["sha256"],
            "artifact_class": row["kind"], "source": "repo", "status": "live",
            "can_regenerate_without_model": row["kind"] in
                ("figure", "figure_data", "document_pdf", "document_md"),
        })
    for name, r in raw.items():
        for mf in r["hashed_manifest_files"]:
            inv_rows.append({
                "logical_uri": f"artifact://{name}/{mf['rel_path']}",
                "path": f"{r['drive_root']}/{mf['rel_path']}",
                "bytes": mf["bytes"], "sha256": mf["sha256"],
                "artifact_class": "drive_manifest", "source": "drive",
                "status": "live", "can_regenerate_without_model": False,
            })
    inv = pd.DataFrame(sorted(inv_rows, key=lambda r: r["logical_uri"]))
    inv.to_parquet(manifests_dir / "campaign_artifact_inventory.parquet",
                   index=False)
    with open(manifests_dir / "campaign_artifact_inventory.json", "w") as f:
        json.dump({"generated_at_commit": commit,
                   "rows": inv.to_dict(orient="records")}, f, indent=2,
                  sort_keys=True, default=str)
        f.write("\n")

    print(f"registries: {len(reg)} ({sum(r['n_events'] for r in reg.values())} events)")
    print(f"paper sources: {len(paper)}")
    print(f"drive roots: {len(raw)} "
          f"({sum(r['n_files'] for r in raw.values())} files, "
          f"{sum(r['total_bytes'] for r in raw.values())/1e9:.1f} GB)")
    print(f"inventory rows: {len(inv)}")


if __name__ == "__main__":
    main()
