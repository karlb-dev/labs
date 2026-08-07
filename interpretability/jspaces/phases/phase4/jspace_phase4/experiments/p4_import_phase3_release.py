"""Register the immutable Phase 3 completion boundary in Phase 4."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from jspace_phase3.provenance3 import EVENTS as PHASE3_EVENTS
from jspace_phase3.provenance3 import resolve as resolve_phase3

from ..imports3 import (
    PHASE3_COMPLETE_COMMIT,
    PHASE3_COMPLETE_TAG,
)
from ..registry4 import import_evidence

EVIDENCE_ID = "p4-import-phase3-release-v1"


def main() -> None:
    repository = Path(__file__).resolve().parents[4]
    tag_commit = subprocess.check_output(
        [
            "git", "-C", str(repository), "rev-list", "-n", "1",
            PHASE3_COMPLETE_TAG,
        ],
        text=True,
    ).strip()
    if tag_commit != PHASE3_COMPLETE_COMMIT:
        raise RuntimeError(
            f"{PHASE3_COMPLETE_TAG} points to {tag_commit}, expected "
            f"{PHASE3_COMPLETE_COMMIT}")
    source = resolve_phase3("p3-release-manifest-v1")
    if not source["live"]:
        raise RuntimeError("Phase 3 release manifest evidence is not live")
    outputs = [Path(row["path"]) for row in source["outputs"]]
    event = import_evidence(
        EVIDENCE_ID,
        tier="phase3-confirmatory-import",
        what=(
            "Immutable Phase 3 completion boundary: the release manifest "
            "and its verified inventory of confirmatory, replication, "
            "methods, raw, model, lens, report, and figure artifacts."),
        source_study="jspace-phase3",
        source_evidence_id="p3-release-manifest-v1",
        source_commit=PHASE3_COMPLETE_COMMIT,
        source_tag=PHASE3_COMPLETE_TAG,
        source_registry=PHASE3_EVENTS,
        source_outputs=outputs,
    )
    print(json.dumps({
        "evidence_id": EVIDENCE_ID,
        "source_tag": PHASE3_COMPLETE_TAG,
        "source_commit": PHASE3_COMPLETE_COMMIT,
        "n_source_outputs": len(outputs),
        "event": event["event"],
    }, indent=1))


if __name__ == "__main__":
    main()
