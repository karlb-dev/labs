"""Bank the final Phase 3 Markdown, TeX, and rendered PDF on Drive."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from jspace_part2.lib import sha256_file
from ..paths3 import run_root
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           write_result3)

EVIDENCE_ID = "p3-state-of-record-release-v1"
TIER = "methods"
PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(
        destination.suffix + f".tmp{os.getpid()}")
    shutil.copyfile(source, tmp)
    os.replace(tmp, destination)


def pdf_pages(path: Path) -> int:
    output = subprocess.check_output(
        ["pdfinfo", str(path)], text=True)
    match = re.search(r"^Pages:\s+(\d+)\s*$", output, re.MULTILINE)
    if not match:
        raise RuntimeError("pdfinfo did not report a page count")
    return int(match.group(1))


def main() -> None:
    require_clean_tree(False)
    sources = {
        "state_markdown": (
            PACKAGE_ROOT / "reports/PHASE3_STATE_OF_RECORD.md"),
        "living_report": (
            PACKAGE_ROOT / "reports/REPORT_PHASE3.md"),
        "final_tex": (
            PACKAGE_ROOT / "reports/handout/jspace_phase3_final.tex"),
        "final_pdf": (
            PACKAGE_ROOT / "reports/handout/jspace_phase3_final.pdf"),
    }
    missing = [str(path) for path in sources.values()
               if not path.exists()]
    if missing:
        raise RuntimeError(f"missing release publication files: {missing}")
    if sources["final_pdf"].stat().st_size < 50_000:
        raise RuntimeError("final PDF is unexpectedly small")
    pages = pdf_pages(sources["final_pdf"])
    if not 4 <= pages <= 20:
        raise RuntimeError(f"unexpected final PDF page count: {pages}")
    state_text = sources["state_markdown"].read_text()
    required_ids = {
        "p3-inference-audit-v1",
        "p3-protocol-audit-protected-answer-qwen-v1",
        "p3-control-seed-contract-audit-v2",
        "p3-n8-p3-level3-qwen36-27b-v1",
        "p3-boundary-cohort-sensitivity-v2",
        "p3-alias-endpoint-cross-model-v1",
        "p3-bridge-geometry-qwen36-27b-v2",
        "p3-bridge-swap-endpoint-qwen36-27b-v1",
    }
    missing_ids = sorted(
        evidence_id for evidence_id in required_ids
        if evidence_id not in state_text)
    if missing_ids:
        raise RuntimeError(
            f"state report omits required evidence IDs: {missing_ids}")

    destination_root = run_root() / "reports"
    destinations = {
        "state_markdown": (
            destination_root / "PHASE3_STATE_OF_RECORD.md"),
        "living_report": destination_root / "REPORT_PHASE3.md",
        "final_tex": (
            destination_root / "handout/jspace_phase3_final.tex"),
        "final_pdf": (
            destination_root / "handout/jspace_phase3_final.pdf"),
    }
    for name, source in sources.items():
        atomic_copy(source, destinations[name])
    source_hashes = {
        name: sha256_file(path) for name, path in sources.items()}
    copy_hashes = {
        name: sha256_file(path) for name, path in destinations.items()}
    if source_hashes != copy_hashes:
        raise RuntimeError("Drive publication copy differs from repo source")

    result_payload = {
        "schema_version": 1,
        "pdf_pages": pages,
        "source_sha256": source_hashes,
        "drive_copy_sha256": copy_hashes,
        "required_evidence_ids": sorted(required_ids),
        "copy_exact": True,
        "release_tag_planned": "jspace-phase3-complete-v1",
    }
    result_path = (
        destination_root / "p3_state_of_record_release.json")
    command = (
        "python -m "
        "jspace_phase3.experiments.p3_release_publication")
    figure = (
        run_root() / "figures/p3f06_phase3_release_audit.png")
    inputs = {
        **{f"source:{name}": value
           for name, value in source_hashes.items()},
        "release_summary_figure": sha256_file(figure),
    }
    write_result3(
        result_payload, result_path,
        Provenance3(
            evidence_id=EVIDENCE_ID, tier=TIER,
            command=command, inputs=inputs))
    outputs = [result_path, *destinations.values()]
    register(
        EVIDENCE_ID,
        tier=TIER,
        command=command,
        what=(
            f"Final Phase 3 state-of-record Markdown and {pages}-page "
            "TeX/PDF handout, copied byte-exactly from the release commit "
            "to the durable run root."),
        outputs=outputs,
        inputs=inputs)
    print(json.dumps({
        "result": str(result_path),
        "pdf": str(destinations["final_pdf"]),
        "pages": pages,
        "copy_exact": True,
    }, indent=1))


if __name__ == "__main__":
    main()
