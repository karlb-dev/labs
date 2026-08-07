"""Conformance and registered-output verification."""
from __future__ import annotations

from pathlib import Path

from .manifests import file_sha256, object_sha256, verify_constraints
from .paths import REPO_ROOT
from .registry import resolve_all


def _repository_materialization(path: Path) -> Path:
    """Map an absolute producer-worktree output into the merged repository."""
    if not path.is_absolute():
        return path
    try:
        marker = path.parts.index("interpretability")
    except ValueError:
        return path
    candidate = REPO_ROOT / Path(*path.parts[marker:])
    return candidate if candidate.is_file() else path


def verify_live_evidence() -> dict:
    failures = []
    checked = 0
    for event in resolve_all():
        if not event["live"]:
            continue
        field = "source_outputs" if event["event"] == "evidence_imported" else "outputs"
        for output in event.get(field, []) or []:
            path = Path(output["path"])
            materialized = _repository_materialization(path)
            actual = file_sha256(materialized) if materialized.exists() else None
            expected = output.get("sha256")
            checked += 1
            if actual != expected:
                failures.append(
                    {
                        "evidence_id": event["evidence_id"],
                        "path": str(path),
                        "expected": expected,
                        "actual": actual,
                    }
                )
    return {
        "ok": not failures,
        "n_live_events": sum(row["live"] for row in resolve_all()),
        "n_checked_outputs": checked,
        "failures": failures,
    }


def verify_all() -> dict:
    package_root = Path(__file__).resolve().parents[1]
    constraints = verify_constraints(package_root / "constraints.txt")
    evidence = verify_live_evidence()
    return {
        "ok": constraints["ok"] and evidence["ok"],
        "constraints": constraints,
        "live_evidence": evidence,
        "contract_sha256": object_sha256(
            {
                "branch": "interp_jspace_gemma_transport",
                "fork": "3b041735d8b842de46a9c0a474fccd0c44e0841a",
                "registry_prefix": "gm-",
                "tiers": ["development", "methods"],
            }
        ),
    }
