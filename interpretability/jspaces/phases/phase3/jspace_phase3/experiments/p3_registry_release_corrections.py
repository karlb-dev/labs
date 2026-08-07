"""Append the Phase 3 release-audit metadata corrections.

Historical creation rows remain immutable. Reruns are idempotent because an
identical correction already present in the status stream is a no-op.
"""
from __future__ import annotations

import json
import sys

from ..provenance3 import correct, require_clean_tree, resolve

CORRECTIONS = {
    "p3-replication-analysis-v1": {
        "corrected_fields": {"tier": "phase3-replication"},
        "reason": (
            "The producer used a confirmatory tier constant even when "
            "--side replication was passed. The immutable output is the "
            "held-out Phase 3 replication partition."
        ),
    },
    "p3-n8-level1-repro-v1": {
        "corrected_fields": {"scope_label": "N8-P2-L1"},
        "reason": (
            "The historical clean-room analysis reproduced Phase 2 N6 "
            "evidence, not the Phase 3 primary grid."
        ),
    },
    "p3-n8-level2-olmo31-think-v1": {
        "corrected_fields": {"scope_label": "N8-P2-L2"},
        "reason": "The historical sentinel relaunched a Phase 2 N6 cell.",
    },
    "p3-n8-level2-olmo31-instruct-v1": {
        "corrected_fields": {"scope_label": "N8-P2-L2"},
        "reason": "The historical sentinel relaunched a Phase 2 N6 cell.",
    },
    "p3-n8-level2-qwen36-27b-v1": {
        "corrected_fields": {"scope_label": "N8-P2-L2"},
        "reason": "The historical sentinel relaunched a Phase 2 N6 cell.",
    },
    "p3-n8-level3-qwen36-27b-v1": {
        "corrected_fields": {"scope_label": "N8-P2-L3"},
        "reason": "The historical full cell relaunched a Phase 2 N6 cell.",
    },
}


def main() -> None:
    require_clean_tree("--allow-dirty" in sys.argv)
    appended = []
    for evidence_id, correction in CORRECTIONS.items():
        record = resolve(evidence_id)
        already = any(
            event.get("event") == "evidence_corrected"
            and event.get("corrected_fields")
            == correction["corrected_fields"]
            for event in record["status_events"]
        )
        if already:
            continue
        correct(evidence_id, **correction)
        appended.append(evidence_id)
    print(json.dumps({
        "appended": appended,
        "effective": {
            evidence_id: resolve(evidence_id)["effective_metadata"]
            for evidence_id in CORRECTIONS
        },
    }, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
