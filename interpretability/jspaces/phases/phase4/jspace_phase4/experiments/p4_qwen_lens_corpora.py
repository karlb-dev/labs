"""Freeze leakage-safe nested WikiText corpora for Qwen lens fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import pyarrow.parquet as pq
import yaml

from ..manifests import (
    atomic_json,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths4 import resolve_uri, run_root
from ..provenance4 import Provenance4, write_result4
from ..registry4 import create


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text().splitlines()
        if line.strip()
    ]


def jsonl_bytes(rows: Iterable[dict]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=True) + "\n" for row in rows
    ).encode("utf-8")


def atomic_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    temporary.write_bytes(jsonl_bytes(rows))
    os.replace(temporary, path)


def text_concat_sha256(rows: Sequence[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["text"].encode("utf-8"))
    return digest.hexdigest()


def index_sha256(rows: Sequence[dict]) -> str:
    return object_sha256([int(row["idx"]) for row in rows])


def hash_order_sample(
        indices: Iterable[int], *, namespace: str, n: int) -> list[int]:
    """Take a stable pseudo-random sample independent of NumPy versions."""
    if not namespace:
        raise ValueError("selection namespace must be non-empty")
    candidates = sorted({int(index) for index in indices})
    if n < 0 or n > len(candidates):
        raise ValueError(
            f"cannot select {n} rows from {len(candidates)} candidates")

    def key(index: int) -> tuple[bytes, int]:
        payload = f"{namespace}\0{index}".encode("utf-8")
        return hashlib.sha256(payload).digest(), index

    return sorted(candidates, key=key)[:n]


def validate_legacy_rows(
        rows: Sequence[dict], texts: Sequence[str], *,
        expected_n: int, min_chars: int, label: str,
) -> None:
    if len(rows) != expected_n:
        raise RuntimeError(
            f"{label} has {len(rows)} rows, expected {expected_n}")
    indices = [int(row["idx"]) for row in rows]
    if len(set(indices)) != len(indices):
        raise RuntimeError(f"{label} contains duplicate dataset indices")
    for offset, row in enumerate(rows):
        index = int(row["idx"])
        if index < 0 or index >= len(texts):
            raise RuntimeError(
                f"{label} row {offset} has out-of-range index {index}")
        if row["text"] != texts[index]:
            raise RuntimeError(
                f"{label} row {offset} does not match pinned dataset "
                f"index {index}")
        if len(row["text"].strip()) < min_chars:
            raise RuntimeError(
                f"{label} row {offset} is below min_chars={min_chars}")


def build_nested_corpora(
        texts: Sequence[str], eligible_indices: Sequence[int],
        legacy_a: Sequence[dict], legacy_b: Sequence[dict], *,
        a_prefix_n: int, a_total_n: int, b_prefix_n: int, b_total_n: int,
        a_namespace: str, b_namespace: str,
) -> tuple[list[dict], list[dict], dict]:
    """Extend historical fit prefixes without admitting evaluation spares."""
    if a_prefix_n > len(legacy_a) or b_prefix_n > len(legacy_b):
        raise ValueError("legacy prefix length exceeds source corpus")
    eligible = set(int(index) for index in eligible_indices)
    legacy_a_indices = {int(row["idx"]) for row in legacy_a}
    legacy_b_indices = {int(row["idx"]) for row in legacy_b}
    if not legacy_a_indices.isdisjoint(legacy_b_indices):
        raise RuntimeError("historical draw A and draw B overlap")

    a_prefix = [dict(row) for row in legacy_a[:a_prefix_n]]
    a_excluded = legacy_a_indices | legacy_b_indices
    a_extension_indices = hash_order_sample(
        eligible - a_excluded,
        namespace=a_namespace,
        n=a_total_n - a_prefix_n,
    )
    draw_a = a_prefix + [
        {"idx": index, "text": texts[index]}
        for index in a_extension_indices
    ]

    draw_a_indices = {int(row["idx"]) for row in draw_a}
    b_prefix = [dict(row) for row in legacy_b[:b_prefix_n]]
    b_excluded = legacy_a_indices | draw_a_indices | legacy_b_indices
    b_extension_indices = hash_order_sample(
        eligible - b_excluded,
        namespace=b_namespace,
        n=b_total_n - b_prefix_n,
    )
    draw_b = b_prefix + [
        {"idx": index, "text": texts[index]}
        for index in b_extension_indices
    ]
    draw_b_indices = {int(row["idx"]) for row in draw_b}
    if not draw_a_indices.isdisjoint(draw_b_indices):
        raise RuntimeError("nested draw A and independent draw B overlap")
    reserved_eval = legacy_a_indices - {
        int(row["idx"]) for row in a_prefix
    }
    if not reserved_eval.isdisjoint(draw_a_indices | draw_b_indices):
        raise RuntimeError("legacy evaluation spares leaked into a fit corpus")
    return draw_a, draw_b, {
        "legacy_a_fit_prefix_n": a_prefix_n,
        "legacy_a_eval_spares_excluded_n": len(reserved_eval),
        "legacy_b_prefix_n": b_prefix_n,
        "a_b_overlap_n": len(draw_a_indices & draw_b_indices),
        "a_legacy_eval_spare_overlap_n": len(
            draw_a_indices & reserved_eval),
        "b_legacy_eval_spare_overlap_n": len(
            draw_b_indices & reserved_eval),
    }


def corpus_summary(
        rows: Sequence[dict], milestones: Sequence[int],
        *, historical_prefix: Sequence[dict],
) -> dict:
    return {
        "n": len(rows),
        "jsonl_sha256": hashlib.sha256(jsonl_bytes(rows)).hexdigest(),
        "text_concat_sha256": text_concat_sha256(rows),
        "index_sha256": index_sha256(rows),
        "unique_indices": len({int(row["idx"]) for row in rows}),
        "milestones": {
            str(n): {
                "jsonl_sha256":
                    hashlib.sha256(jsonl_bytes(rows[:n])).hexdigest(),
                "text_concat_sha256": text_concat_sha256(rows[:n]),
                "index_sha256": index_sha256(rows[:n]),
            }
            for n in milestones
        },
        "historical_prefix_n": len(historical_prefix),
        "historical_prefix_byte_exact": (
            jsonl_bytes(rows[:len(historical_prefix)])
            == jsonl_bytes(historical_prefix)
        ),
    }


def main() -> None:
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    clean = require_clean_tree()
    min_chars = int(config["selection"]["min_chars"])

    shard_hashes = {}
    texts: list[str] = []
    for shard in config["dataset"]["shards"]:
        path = resolve_uri(shard["uri"])
        actual = file_sha256(path)
        if actual != shard["sha256"]:
            raise RuntimeError(
                f"dataset shard hash mismatch for {shard['uri']}: {actual}")
        shard_hashes[shard["uri"]] = actual
        texts.extend(
            pq.read_table(path, columns=["text"])
            .column("text").to_pylist()
        )
    if len(texts) != int(config["dataset"]["expected_rows"]):
        raise RuntimeError(
            f"dataset has {len(texts)} rows, expected "
            f"{config['dataset']['expected_rows']}")
    eligible_indices = [
        index for index, text in enumerate(texts)
        if len(text.strip()) >= min_chars
    ]
    if len(eligible_indices) != int(
            config["dataset"]["expected_eligible_rows"]):
        raise RuntimeError(
            f"dataset has {len(eligible_indices)} eligible rows, expected "
            f"{config['dataset']['expected_eligible_rows']}")

    legacy_paths = {}
    legacy_rows = {}
    for label in ("draw_a", "draw_b"):
        specification = config[f"legacy_{label}"]
        path = resolve_uri(specification["uri"])
        actual = file_sha256(path)
        if actual != specification["sha256"]:
            raise RuntimeError(
                f"legacy {label} hash mismatch: {actual}")
        rows = load_jsonl(path)
        validate_legacy_rows(
            rows, texts,
            expected_n=int(specification["n_rows"]),
            min_chars=min_chars,
            label=f"legacy {label}",
        )
        legacy_paths[label] = path
        legacy_rows[label] = rows

    selection = config["selection"]
    draw_a, draw_b, leakage_audit = build_nested_corpora(
        texts,
        eligible_indices,
        legacy_rows["draw_a"],
        legacy_rows["draw_b"],
        a_prefix_n=int(selection["draw_a_prefix_n"]),
        a_total_n=int(selection["draw_a_total_n"]),
        b_prefix_n=int(selection["draw_b_prefix_n"]),
        b_total_n=int(selection["draw_b_total_n"]),
        a_namespace=selection["draw_a_extension_namespace"],
        b_namespace=selection["draw_b_extension_namespace"],
    )

    output_dir = (
        run_root() / "config" / "qwen_lens_corpora"
        / config["evidence_id"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    draw_a_path = output_dir / "qwen_drawA_nested1000.jsonl"
    draw_b_path = output_dir / "qwen_drawB_nested500.jsonl"
    manifest_path = output_dir / "input_manifest.json"
    result_path = output_dir / "corpus_result.json"
    atomic_jsonl(draw_a_path, draw_a)
    atomic_jsonl(draw_b_path, draw_b)

    helper_reference = config["published_helper_reference"]
    helper_source = resolve_uri(helper_reference["helper_source_uri"])
    helper_source_hash = file_sha256(helper_source)
    if helper_source_hash != helper_reference["helper_source_sha256"]:
        raise RuntimeError(
            "pinned upstream WikiText helper source hash mismatch: "
            f"{helper_source_hash}")
    published_n = int(helper_reference["n"])
    published_rows = [
        {"idx": index, "text": texts[index]}
        for index in eligible_indices[:published_n]
    ]
    input_payload = {
        "schema_version": 1,
        "experiment_id": config["evidence_id"],
        "config_sha256": file_sha256(config_path),
        "code_commit": clean["code_commit"],
        "dataset": {
            "id": config["dataset"]["id"],
            "config": config["dataset"]["config"],
            "revision": config["dataset"]["revision"],
            "split": config["dataset"]["split"],
            "shards": shard_hashes,
            "n_rows": len(texts),
            "min_chars": min_chars,
            "eligible_rows": len(eligible_indices),
        },
        "legacy_inputs": {
            label: {
                "uri": config[f"legacy_{label}"]["uri"],
                "sha256": file_sha256(path),
                "n_rows": len(legacy_rows[label]),
                "index_sha256": index_sha256(legacy_rows[label]),
                "text_concat_sha256":
                    text_concat_sha256(legacy_rows[label]),
            }
            for label, path in legacy_paths.items()
        },
        "selection_contract": {
            "method": "sha256(namespace + NUL + dataset_index), ascending",
            **selection,
        },
        "published_helper_reference": {
            **helper_reference,
            "status": (
                "reconstructed from the pinned upstream helper convention; "
                "the published lens artifact does not embed a corpus manifest"
            ),
            "index_sha256": index_sha256(published_rows),
            "text_concat_sha256": text_concat_sha256(published_rows),
        },
    }
    input_envelope = {
        "schema_version": 1,
        "payload": input_payload,
        "payload_sha256": object_sha256(input_payload),
    }
    atomic_json(manifest_path, input_envelope)

    a_summary = corpus_summary(
        draw_a,
        selection["draw_a_milestones"],
        historical_prefix=legacy_rows["draw_a"][
            :int(selection["draw_a_prefix_n"])],
    )
    b_summary = corpus_summary(
        draw_b,
        selection["draw_b_milestones"],
        historical_prefix=legacy_rows["draw_b"][
            :int(selection["draw_b_prefix_n"])],
    )
    published_indices = {int(row["idx"]) for row in published_rows}
    payload = {
        "schema_version": 1,
        "dataset_rows": len(texts),
        "eligible_rows": len(eligible_indices),
        "leakage_audit": leakage_audit,
        "draw_a": a_summary,
        "draw_b": b_summary,
        "published_helper_reference": {
            "n": published_n,
            "draw_a_overlap_n": len(
                published_indices
                & {int(row["idx"]) for row in draw_a}),
            "draw_b_overlap_n": len(
                published_indices
                & {int(row["idx"]) for row in draw_b}),
            "index_sha256": index_sha256(published_rows),
            "text_concat_sha256": text_concat_sha256(published_rows),
            "artifact_proven": False,
        },
    }
    command = (
        "python -m jspace_phase4.experiments.p4_qwen_lens_corpora "
        f"--config {arguments.config}"
    )
    inputs = {
        "input_manifest": file_sha256(manifest_path),
        **shard_hashes,
        config["legacy_draw_a"]["uri"]:
            file_sha256(legacy_paths["draw_a"]),
        config["legacy_draw_b"]["uri"]:
            file_sha256(legacy_paths["draw_b"]),
        helper_reference["helper_source_uri"]: helper_source_hash,
    }
    write_result4(
        payload,
        result_path,
        Provenance4(
            evidence_id=config["evidence_id"],
            tier=config["tier"],
            command=command,
            inputs=inputs,
            input_manifest_sha256=input_envelope["payload_sha256"],
            model=None,
            seed_contract=(
                "sha256-ordered-dataset-index-nested-corpus-v1"),
        ),
    )
    create(
        config["evidence_id"],
        tier=config["tier"],
        what=(
            "Leakage-safe nested Qwen WikiText lens corpora: exact "
            "historical n=120 fit prefixes; draw A extended to n=1000 "
            "without the 80 legacy evaluation spares; independent draw B "
            "extended disjointly to n=500."),
        command=command,
        outputs=[
            draw_a_path, draw_b_path, manifest_path, result_path,
        ],
        inputs=inputs,
    )
    print(json.dumps({
        "evidence_id": config["evidence_id"],
        "output_dir": str(output_dir),
        "draw_a": a_summary,
        "draw_b": b_summary,
        "leakage_audit": leakage_audit,
    }, indent=1))


if __name__ == "__main__":
    main()
