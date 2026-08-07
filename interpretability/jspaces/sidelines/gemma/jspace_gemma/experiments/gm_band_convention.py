"""Register the original paper's reindexed workspace-depth convention."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path

import yaml

from jspace_gemma.manifests import atomic_json, file_sha256, require_clean_tree
from jspace_gemma.paths import PACKAGE_ROOT, directory
from jspace_gemma.registry import create

CONFIG = PACKAGE_ROOT / "configs/gm_g6_band_convention.yaml"


def round_half_up(value: float) -> int:
    return int(value + 0.5)


def mapped_band(fractions: list[float], decoder_blocks: int) -> list[int]:
    if len(fractions) != 2 or not 0 <= fractions[0] < fractions[1] <= 1:
        raise ValueError("band fractions must be an increasing pair in [0,1]")
    if decoder_blocks < 2:
        raise ValueError("decoder block count is invalid")
    return [round_half_up(value * decoder_blocks) for value in fractions]


def _fetch_primary(config: dict) -> dict:
    source = config["primary_source"]
    request = urllib.request.Request(
        source["url"],
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "jspace-gemma-methods-audit/1",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read()
        headers = response.headers
        resolved_url = response.geturl()
        status = response.status
    digest = hashlib.sha256(payload).hexdigest()
    if digest != source["expected_html_sha256"]:
        raise RuntimeError("primary workspace paper byte hash changed")
    if len(payload) != source["expected_size_bytes"]:
        raise RuntimeError("primary workspace paper byte size changed")
    text = payload.decode("utf-8")
    required_markers = [
        "reindexed to the range [0–100]",
        "(~L38)",
        "(~L92)",
    ]
    missing = [marker for marker in required_markers if marker not in text]
    if missing:
        raise RuntimeError(f"primary methods markers missing: {missing}")
    if headers.get("ETag") != source["expected_etag"]:
        raise RuntimeError("primary workspace paper ETag changed")
    if headers.get("Last-Modified") != source["expected_last_modified"]:
        raise RuntimeError("primary workspace paper Last-Modified changed")
    return {
        "url": source["url"],
        "resolved_url": resolved_url,
        "http_status": status,
        "html_sha256": digest,
        "size_bytes": len(payload),
        "etag": headers.get("ETag"),
        "last_modified": headers.get("Last-Modified"),
        "methods_anchor": source["methods_anchor"],
        "structure_anchor": source["structure_anchor"],
        "markers_verified": True,
    }


def _conflicting_git_object(config: dict) -> dict:
    source = config["conflicting_later_material"]
    spec = f"{source['source_commit']}:{source['path']}"
    payload = subprocess.check_output(["git", "show", spec])
    digest = hashlib.sha256(payload).hexdigest()
    blob = subprocess.check_output(["git", "rev-parse", spec], text=True).strip()
    if digest != source["sha256"] or blob != source["git_blob_id"]:
        raise RuntimeError("conflicting later handout Git object drifted")
    if b"37--62\\%" not in payload:
        raise RuntimeError("expected later 37--62 percent convention is absent")
    return {
        "source_commit": source["source_commit"],
        "path": source["path"],
        "git_blob_id": blob,
        "sha256": digest,
        "size_bytes": len(payload),
        "later_range_percent": source["later_range_percent"],
        "authority": source["authority"],
    }


def main() -> None:
    git = require_clean_tree()
    config = yaml.safe_load(CONFIG.read_text())
    addendum = Path(config["governing_addendum"]["path"])
    if file_sha256(addendum) != config["governing_addendum"]["sha256"]:
        raise RuntimeError("governing Gemma addendum hash drifted")
    primary = _fetch_primary(config)
    later = _conflicting_git_object(config)
    resolution = config["resolution"]
    mapped = mapped_band(
        resolution["transferable_workspace_depth_fraction"],
        int(resolution["gemma_decoder_blocks"]),
    )
    if mapped != resolution["expected_gemma_zero_indexed_approx_range"]:
        raise RuntimeError("Gemma band mapping differs from the frozen expectation")
    candidates = resolution["g6_candidate_layers_zero_indexed"]
    if not all(mapped[0] <= layer <= mapped[1] for layer in candidates):
        raise RuntimeError("a frozen G6 candidate lies outside the resolved paper band")
    payload = {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "tier": config["tier"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_commit": git["code_commit"],
        "config": {"path": str(CONFIG), "sha256": file_sha256(CONFIG)},
        "governing_addendum": {
            "path": str(addendum),
            "sha256": file_sha256(addendum),
        },
        "primary_source_observation": primary,
        "conflicting_later_material": later,
        "resolution": {
            **resolution,
            "mapped_gemma_zero_indexed_approx_range": mapped,
            "all_g6_candidates_inside_resolved_band": True,
            "primary_methods_take_precedence": True,
        },
        "target_model_opened": False,
        "scientific_model_result": False,
    }
    output = directory("manifests") / "gm_band_convention_v1.json"
    atomic_json(output, payload)
    create(
        config["evidence_id"],
        tier=config["tier"],
        what=(
            "primary-paper methods resolution: reported layer coordinates are "
            "reindexed percentages and the transferable workspace band is 38--92%"
        ),
        command="python -m jspace_gemma.experiments.gm_band_convention",
        outputs=[output],
        inputs={
            "config_sha256": file_sha256(CONFIG),
            "primary_html_sha256": primary["html_sha256"],
            "governing_addendum_sha256": file_sha256(addendum),
            "conflicting_later_material_sha256": later["sha256"],
        },
        target_model_opened=False,
        scientific_model_result=False,
    )
    print(json.dumps({"output": str(output), "sha256": file_sha256(output)}, indent=1))


if __name__ == "__main__":
    main()
