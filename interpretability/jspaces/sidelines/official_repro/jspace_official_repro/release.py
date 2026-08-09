"""Terminal release assembly (plan §17.3).

Builds: OFFICIAL_REPRO_RELEASE_MANIFEST.{json,md} (hash inventory of
every live registered output), IMPORT_BUNDLE_OFFICIAL_REPRO_1.{json,md}
(advisory; no frozen registry consumes it automatically),
evidence_events_prefix_official_repro_1.jsonl (byte snapshot), and
REPRODUCTION_GUIDE.md.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from . import registry
from .manifests import file_sha256, git_info, json_sha256
from .paths import RELEASE, REPORTS, STUDY_ROOT


def build() -> dict:
    RELEASE.mkdir(parents=True, exist_ok=True)
    info = git_info()
    live = registry.live_events()
    inventory = []
    for event in live:
        for output in event.get("outputs", []):
            path = Path(output["path"])
            exists = path.exists()
            inventory.append({
                "evidence_id": event["evidence_id"],
                "tier": event["tier"],
                "path": output["path"],
                "sha256": output["sha256"],
                "verified_at_release": (
                    exists and file_sha256(path) == output["sha256"]),
            })
    n_bad = sum(1 for row in inventory if not row["verified_at_release"])
    manifest = {
        "study_id": "jspace-official-repro-1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_commit": info["code_commit"],
        "branch": info["branch"],
        "n_live_events": len(live),
        "n_output_references": len(inventory),
        "n_verification_failures": n_bad,
        "outputs": inventory,
    }
    (RELEASE / "OFFICIAL_REPRO_RELEASE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2))

    lines = [
        "# OFFICIAL_REPRO_RELEASE_MANIFEST",
        f"\nGenerated {manifest['generated_utc']} at "
        f"`{info['code_commit'][:12]}` on `{info['branch']}`.",
        f"\nLive events: {len(live)} · output references: "
        f"{len(inventory)} · verification failures at release: {n_bad}\n",
        "| evidence id | tier | outputs |", "|---|---|---|",
    ]
    by_event: dict[str, int] = {}
    for row in inventory:
        by_event[row["evidence_id"]] = by_event.get(row["evidence_id"], 0) + 1
    for event in live:
        lines.append(f"| {event['evidence_id']} | {event['tier']} | "
                     f"{by_event.get(event['evidence_id'], 0)} |")
    (RELEASE / "OFFICIAL_REPRO_RELEASE_MANIFEST.md").write_text(
        "\n".join(lines) + "\n")

    shutil.copy2(REPORTS / "evidence_events.jsonl",
                 RELEASE / "evidence_events_prefix_official_repro_1.jsonl")

    payload = {
        "schema_version": 1,
        "study_id": "jspace-official-repro-1",
        "bundle_id": "official-repro-1-terminal",
        "advisory": True,
        "consumes_automatically": [],
        "source_git": info,
        "registry_snapshot_sha256": file_sha256(
            RELEASE / "evidence_events_prefix_official_repro_1.jsonl"),
        "headline_grid_sha256": file_sha256(REPORTS / "HEADLINE_GRID.json"),
        "claim_ledger_sha256": file_sha256(
            REPORTS / "OFFICIAL_REPRO_CLAIM_LEDGER.md"),
        "state_of_record_sha256": file_sha256(
            REPORTS / "OFFICIAL_REPRO_STATE_OF_RECORD.md"),
        "n_live_events": len(live),
    }
    envelope = {"payload": payload, "payload_sha256": json_sha256(payload)}
    (RELEASE / "IMPORT_BUNDLE_OFFICIAL_REPRO_1.json").write_text(
        json.dumps(envelope, indent=2))
    (RELEASE / "IMPORT_BUNDLE_OFFICIAL_REPRO_1.md").write_text(
        "# IMPORT_BUNDLE_OFFICIAL_REPRO_1 (advisory)\n\n"
        "This bundle is advisory context for the frozen paper routes and\n"
        "any future study. No frozen phase or paper-analysis registry\n"
        "consumes it automatically. Effects reported here are development\n"
        "tier; gated and not-identified cells are states, never zeros; no\n"
        "workspace-existence or -absence claim is licensed.\n\n"
        f"payload_sha256: `{envelope['payload_sha256']}`\n")

    (RELEASE / "REPRODUCTION_GUIDE.md").write_text(
        "# REPRODUCTION_GUIDE — official-repro Study 1\n\n"
        "## CPU artifacts (no model weights)\n\n"
        "```bash\n"
        "pip install -e interpretability/jspaces/sidelines/official_repro\n"
        "python -m pytest interpretability/jspaces/sidelines/official_repro/tests -q\n"
        "jspace-or1 verify        # rehash every live registered output\n"
        "python -m jspace_official_repro.headline      # regenerate §16.1 grid\n"
        "python -m jspace_official_repro.report_data   # regenerate numbers.tex\n"
        "python -m jspace_official_repro.report_examples\n"
        "python -m jspace_official_repro.figures qwen  # figures from JSON\n"
        "python -m jspace_official_repro.figures olmo\n"
        "cd interpretability/jspaces/sidelines/official_repro/reports/tex && "
        "latexmk -pdf official_repro_report.tex\n"
        "```\n\n"
        "Every aggregate, figure, table, and prose number regenerates from\n"
        "registered JSON without loading a model (plan §12).\n\n"
        "## Model-backed reproduction\n\n"
        "Follow `README.md` quickstart (pinned engine clone, HF snapshots\n"
        "at the §2 revisions), then the stage drivers in\n"
        "`jspace_official_repro/experiments/` in plan §14 order. All\n"
        "stages are idempotent by output existence; the OLMo fit resumes\n"
        "from checkpoints under a runtime-sentinel gate.\n")

    return manifest


if __name__ == "__main__":
    result = build()
    print(f"release built: {result['n_live_events']} events, "
          f"{result['n_output_references']} outputs, "
          f"{result['n_verification_failures']} failures")
