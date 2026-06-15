#!/usr/bin/env python3
"""Generate the frozen Lab 33 synthetic multimodal concept suite.

The images themselves are rendered by the lab at runtime. This generator writes
JSONL specs plus a small Lab 33 manifest with the data hash. It also updates
``data/MANIFEST.json`` when that file exists or can be created.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from collections import Counter
from typing import Any

DATA_FILE = "multimodal_concept_pairs.jsonl"
LAB_MANIFEST = "multimodal_MANIFEST.json"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def row(
    item_id: str,
    family: str,
    question: str,
    target: str,
    distractor: str,
    image_spec: dict[str, Any],
    control_spec: dict[str, Any],
    *,
    split: str,
    region: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "image_path": f"generated/{item_id}_clean.png",
        "image_control_path": f"generated/{item_id}_corrupt.png",
        "question": question,
        "target": target,
        "distractor": distractor,
        "concept_family": family,
        "text_control_prompt": "Answer the visual question using only the rendered image. Do not use captions, background color, or printed text as labels.",
        "split": split,
        "image_spec": image_spec,
        "control_spec": control_spec,
        "region": region,
        "notes": notes,
    }


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    color_specs = [
        ("color_red_blue_001", "red", "blue", "circle", "train"),
        ("color_blue_green_002", "blue", "green", "square", "train"),
        ("color_green_yellow_003", "green", "yellow", "triangle", "test"),
        ("color_purple_orange_004", "purple", "orange", "star", "test"),
    ]
    for item_id, target, distractor, shape, split in color_specs:
        rows.append(row(
            item_id, "color", "What color is the main object?", target, distractor,
            {"color": target, "shape": shape, "count": 1, "position": "center", "background": "plain", "ocr_text": ""},
            {"color": distractor, "shape": shape, "count": 1, "position": "center", "background": "plain", "ocr_text": ""},
            split=split, region="object", notes="Clean/corrupt pair swaps object color only.",
        ))

    shape_specs = [
        ("shape_circle_square_001", "circle", "square", "red", "train"),
        ("shape_triangle_star_002", "triangle", "star", "blue", "train"),
        ("shape_square_circle_003", "square", "circle", "green", "test"),
        ("shape_star_triangle_004", "star", "triangle", "purple", "test"),
    ]
    for item_id, target, distractor, color, split in shape_specs:
        rows.append(row(
            item_id, "shape", "What shape is the main object?", target, distractor,
            {"color": color, "shape": target, "count": 1, "position": "center", "background": "plain", "ocr_text": ""},
            {"color": color, "shape": distractor, "count": 1, "position": "center", "background": "plain", "ocr_text": ""},
            split=split, region="object", notes="Clean/corrupt pair swaps shape only.",
        ))

    count_specs = [
        ("count_one_two_001", "one", "two", 1, 2, "train"),
        ("count_two_three_002", "two", "three", 2, 3, "train"),
        ("count_three_four_003", "three", "four", 3, 4, "test"),
        ("count_four_one_004", "four", "one", 4, 1, "test"),
    ]
    for item_id, target, distractor, clean_count, corrupt_count, split in count_specs:
        rows.append(row(
            item_id, "count", "How many objects are shown?", target, distractor,
            {"color": "green", "shape": "circle", "count": clean_count, "position": "center", "background": "plain", "ocr_text": ""},
            {"color": "green", "shape": "circle", "count": corrupt_count, "position": "center", "background": "plain", "ocr_text": ""},
            split=split, region="object_set", notes="Clean/corrupt pair swaps object count only.",
        ))

    spatial_specs = [
        ("spatial_left_right_001", "left", "right", "left", "right", "train"),
        ("spatial_above_below_002", "above", "below", "above", "below", "train"),
        ("spatial_right_left_003", "right", "left", "right", "left", "test"),
        ("spatial_below_above_004", "below", "above", "below", "above", "test"),
    ]
    for item_id, target, distractor, clean_pos, corrupt_pos, split in spatial_specs:
        rows.append(row(
            item_id, "spatial", "Where is the object located?", target, distractor,
            {"color": "orange", "shape": "square", "count": 1, "position": clean_pos, "background": "plain", "ocr_text": ""},
            {"color": "orange", "shape": "square", "count": 1, "position": corrupt_pos, "background": "plain", "ocr_text": ""},
            split=split, region="object_location", notes="Clean/corrupt pair swaps object position only.",
        ))

    chart_specs = [
        ("chart_left_right_001", "left", "right", "left_taller", "right_taller", "train"),
        ("chart_right_left_002", "right", "left", "right_taller", "left_taller", "train"),
        ("chart_left_right_003", "left", "right", "left_taller", "right_taller", "test"),
        ("chart_right_left_004", "right", "left", "right_taller", "left_taller", "test"),
    ]
    for item_id, target, distractor, clean_chart, corrupt_chart, split in chart_specs:
        rows.append(row(
            item_id, "chart", "Which bar is taller?", target, distractor,
            {"color": "blue", "shape": "bar", "count": 2, "position": "center", "background": "chart", "chart_value": clean_chart, "ocr_text": ""},
            {"color": "blue", "shape": "bar", "count": 2, "position": "center", "background": "chart", "chart_value": corrupt_chart, "ocr_text": ""},
            split=split, region="bar_pair", notes="Clean/corrupt pair swaps chart relation only.",
        ))

    ocr_specs = [
        ("ocr_trap_red_blueword_001", "red", "blue", "BLUE", "RED", "train"),
        ("ocr_trap_green_yellowword_002", "green", "yellow", "YELLOW", "GREEN", "train"),
        ("ocr_trap_blue_redword_003", "blue", "red", "RED", "BLUE", "test"),
        ("ocr_trap_yellow_greenword_004", "yellow", "green", "GREEN", "YELLOW", "test"),
    ]
    for item_id, target, distractor, clean_ocr, corrupt_ocr, split in ocr_specs:
        rows.append(row(
            item_id, "ocr_control", "What color is the object, ignoring any printed word?", target, distractor,
            {"color": target, "shape": "circle", "count": 1, "position": "center", "background": "plain", "ocr_text": clean_ocr},
            {"color": distractor, "shape": "circle", "count": 1, "position": "center", "background": "plain", "ocr_text": corrupt_ocr},
            split=split, region="object_with_text", notes="OCR text names the distractor in the clean image; real VLM claims must survive this trap.",
        ))

    bg_specs = [
        ("background_trap_red_bluebg_001", "red", "blue", "blue", "red", "train"),
        ("background_trap_green_yellowbg_002", "green", "yellow", "yellow", "green", "train"),
        ("background_trap_blue_redbg_003", "blue", "red", "red", "blue", "test"),
        ("background_trap_yellow_greenbg_004", "yellow", "green", "green", "yellow", "test"),
    ]
    for item_id, target, distractor, clean_bg, corrupt_bg, split in bg_specs:
        rows.append(row(
            item_id, "background_control", "What color is the object, ignoring the background?", target, distractor,
            {"color": target, "shape": "square", "count": 1, "position": "center", "background": clean_bg, "ocr_text": ""},
            {"color": distractor, "shape": "square", "count": 1, "position": "center", "background": corrupt_bg, "ocr_text": ""},
            split=split, region="object_on_background", notes="Background color names the distractor in the clean image; real VLM claims must survive this trap.",
        ))

    return rows


def main() -> None:
    here = pathlib.Path(__file__).resolve().parent
    out = here / DATA_FILE
    rows = build_rows()
    out.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")
    digest = sha256_file(out)
    families = dict(sorted(Counter(str(r["concept_family"]) for r in rows).items()))
    splits = dict(sorted(Counter(str(r["split"]) for r in rows).items()))
    manifest_payload = {
        "files": {
            DATA_FILE: {
                "sha256": digest,
                "rows": len(rows),
                "generator": pathlib.Path(__file__).name,
                "concept_families": families,
                "splits": splits,
                "description": "Frozen synthetic multimodal concept-pair specs for Lab 33.",
            }
        }
    }
    (here / LAB_MANIFEST).write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    course_manifest = here / "MANIFEST.json"
    if course_manifest.exists():
        try:
            merged = json.loads(course_manifest.read_text(encoding="utf-8"))
        except Exception:
            merged = {"files": {}}
    else:
        merged = {"files": {}}
    if not isinstance(merged, dict):
        merged = {}
    merged[DATA_FILE] = {
        "concept_families": families,
        "generator": pathlib.Path(__file__).name,
        "rows": len(rows),
        "science_scope": "synthetic connector smoke mode; no real VLM hooks loaded by default",
        "sha256": digest,
        "splits": splits,
        "verified_tokenizers": [
            "runtime reverified by Lab 33"
        ],
    }
    course_manifest.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {out} with {len(rows)} rows")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
