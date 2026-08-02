"""Official OLMo 32B post-training checkpoint availability inventory.

This methods-only producer downloads only small metadata files and reads
official Hugging Face repository metadata. It never downloads model weights or
opens any behavioral, activation, geometry, or intervention outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download

from ..manifests import (
    atomic_json,
    atomic_text,
    file_sha256,
    git_info,
    object_sha256,
    require_clean_tree,
)
from ..paths import local_work, metrics_dir, resolve_uri
from ..registry import create, supersede

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise TypeError("checkpoint inventory config must be a mapping")
    return value


def normalize_base_models(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return sorted(str(item) for item in value)
    raise TypeError(f"unsupported base_model field {type(value).__name__}")


def model_contract(config: dict, expected: dict) -> dict:
    observed = {
        "hidden_size": config.get("hidden_size"),
        "vocab_size": config.get("vocab_size"),
        "rms_norm_eps": config.get("rms_norm_eps"),
        "tie_word_embeddings": config.get("tie_word_embeddings"),
        "architectures": config.get("architectures", []),
        "model_type": config.get("model_type"),
    }
    checks = {
        "hidden_size": observed["hidden_size"] == expected["hidden_size"],
        "vocab_size": observed["vocab_size"] == expected["vocab_size"],
        "rms_norm_eps": observed["rms_norm_eps"] == expected["rms_norm_eps"],
        "tie_word_embeddings": (
            observed["tie_word_embeddings"]
            == expected["tie_word_embeddings"]),
        "architecture": expected["architecture"] in observed["architectures"],
    }
    return {"observed": observed, "checks": checks, "passes": all(checks.values())}


def weight_manifest(index: dict, siblings: dict[str, object]) -> dict:
    names = sorted(set(index.get("weight_map", {}).values()))
    rows = []
    for name in names:
        metadata = siblings.get(name)
        if metadata is None or metadata.lfs is None:
            raise ValueError(f"missing official LFS metadata for {name}")
        rows.append({
            "name": name,
            "bytes": int(metadata.size),
            "sha256": str(metadata.lfs.sha256),
        })
    if not rows:
        raise ValueError("model index contains no weight shards")
    return {
        "shards": rows,
        "shard_count": len(rows),
        "total_lfs_bytes": sum(row["bytes"] for row in rows),
        "index_total_size": int(index.get("metadata", {}).get("total_size", 0)),
        "manifest_sha256": object_sha256(rows),
        "weights_available": True,
    }


def relevant_refs(refs: object) -> list[dict]:
    rows = []
    for ref in refs.branches:
        if ref.name == "main" or re.search(
                r"(?:step[_-]|-step\d+)", ref.name, flags=re.IGNORECASE):
            rows.append({
                "name": ref.name,
                "ref": ref.ref,
                "target_commit": ref.target_commit,
            })
    return sorted(rows, key=lambda row: row["name"])


def normalize_bpe_model(model: dict) -> dict:
    """Canonicalize equivalent tokenizers JSON BPE-merge encodings.

    Tokenizers has emitted BPE merges both as two-element arrays and as
    space-delimited strings. Those are serialization variants, not different
    merge tables. All other model fields remain exact.
    """
    normalized = dict(model)
    merges = []
    for index, value in enumerate(model.get("merges", [])):
        if isinstance(value, (list, tuple)) and len(value) == 2:
            pair = [str(value[0]), str(value[1])]
        elif isinstance(value, str):
            pair = value.split(" ", 1)
            if len(pair) != 2 or not all(pair):
                raise ValueError(
                    f"cannot normalize BPE merge {index}: {value!r}")
        else:
            raise ValueError(
                f"unsupported BPE merge {index}: {value!r}")
        merges.append(pair)
    normalized["merges"] = merges
    return normalized


def load_tokenizer_audit_corpus(policy: dict) -> tuple[dict, list[str]]:
    """Load and attest the frozen text assay used for semantic compatibility."""
    path = resolve_uri(policy["corpus_uri"])
    raw = path.read_bytes()
    observed_file_hash = hashlib.sha256(raw).hexdigest()
    if observed_file_hash != policy["corpus_file_sha256"]:
        raise ValueError("tokenizer audit corpus file hash mismatch")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows) != policy["corpus_rows_total"]:
        raise ValueError("tokenizer audit corpus row count mismatch")
    field = policy["text_field"]
    texts = [str(row[field]) for row in rows]
    observed_text_hash = hashlib.sha256(
        "".join(texts).encode("utf-8")).hexdigest()
    if observed_text_hash != policy["concatenated_text_sha256"]:
        raise ValueError("tokenizer audit concatenated-text hash mismatch")
    edge_cases = [str(value) for value in policy.get("edge_cases", [])]
    all_texts = texts + edge_cases
    if len(all_texts) != policy["expected_audit_texts"]:
        raise ValueError("tokenizer audit text count mismatch")
    return ({
        "uri": policy["corpus_uri"],
        "path": str(path),
        "file_sha256": observed_file_hash,
        "corpus_rows": len(texts),
        "concatenated_text_sha256": observed_text_hash,
        "edge_cases": edge_cases,
        "audit_texts": len(all_texts),
        "add_special_tokens": False,
    }, all_texts)


def tokenizer_semantics(
    tokenizer_path: str | Path,
    tokenizer_config_path: str | Path,
    texts: list[str],
) -> dict:
    """Hash full content-token semantics and separately record interfaces."""
    from tokenizers import Tokenizer

    tokenizer_path = Path(tokenizer_path)
    tokenizer_config_path = Path(tokenizer_config_path)
    raw = json.loads(tokenizer_path.read_text())
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocabulary_size = tokenizer.get_vocab_size(with_added_tokens=True)
    id_to_token = [tokenizer.id_to_token(index)
                   for index in range(vocabulary_size)]
    if any(value is None for value in id_to_token):
        raise ValueError("tokenizer has an unmapped vocabulary ID")
    encoded = [
        tokenizer.encode(text, add_special_tokens=False).ids
        for text in texts
    ]
    processing_components = {
        key: raw.get(key) for key in (
            "added_tokens", "normalizer", "pre_tokenizer",
            "post_processor", "decoder")
    }
    normalized_model = normalize_bpe_model(raw["model"])
    content = {
        "vocabulary_size": vocabulary_size,
        "token_id_map_sha256": object_sha256(id_to_token),
        "normalized_model_sha256": object_sha256(normalized_model),
        "processing_components_sha256": object_sha256(
            processing_components),
        "audit_encoding_sha256": object_sha256(encoded),
    }
    content["semantic_fingerprint_sha256"] = object_sha256(content)

    tokenizer_config = json.loads(tokenizer_config_path.read_text())
    chat_template = tokenizer_config.get("chat_template")
    interface = {
        "bos_token": tokenizer_config.get("bos_token"),
        "eos_token": tokenizer_config.get("eos_token"),
        "pad_token": tokenizer_config.get("pad_token"),
        "unk_token": tokenizer_config.get("unk_token"),
        "add_bos_token": tokenizer_config.get("add_bos_token"),
        "add_eos_token": tokenizer_config.get("add_eos_token"),
        "chat_template_sha256": (
            hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
            if isinstance(chat_template, str) else None),
    }
    return {
        **content,
        "bpe_merge_count": len(normalized_model.get("merges", [])),
        "audit_text_count": len(texts),
        "audit_total_token_count": sum(len(row) for row in encoded),
        "audit_length_vector_sha256": object_sha256(
            [len(row) for row in encoded]),
        "interface": interface,
        "interface_sha256": object_sha256(interface),
        "compatibility_scope": (
            "content token IDs with add_special_tokens=false; BOS and chat "
            "rendering are explicit interface qualifications"),
    }


def route_inventory(rows: list[dict], route: dict) -> dict:
    by_slug = {row["slug"]: row for row in rows}
    cells = route["minimal_wedge_cells"]
    required_stages = set(route["required_think_intermediate_stages"])
    candidate_rows = [by_slug[slug] for slug in cells if slug in by_slug]
    eligible_stages = {
        row["stage"] for row in candidate_rows
        if row.get("intermediate_eligible")
    }
    available = required_stages.issubset(eligible_stages)
    if available:
        decision = "genuine-32b-intermediates-available"
        h5_status = "testable-with-bounded-stage-wedge"
        queue = {
            "status": "queued-not-started",
            "position": route["queue_position"],
            "cells": cells,
            "existing_anchors": route["existing_anchors"],
            "scope": route["scope"],
        }
    else:
        decision = "no-provenance-complete-32b-intermediate-pair"
        h5_status = "stated-unresolvable-at-32b"
        queue = {"status": "not-queued", "cells": []}
    return {
        "decision": decision,
        "h5_status": h5_status,
        "required_stages": sorted(required_stages),
        "eligible_stages": sorted(eligible_stages),
        "minimal_wedge": queue,
        "inventory_only": True,
        "model_outcome_opened": False,
    }


def _card_dict(info: object) -> dict:
    card = info.card_data
    if card is None:
        return {}
    if hasattr(card, "to_dict"):
        return card.to_dict()
    return dict(card)


def _inspect_target(
    spec: dict,
    policy: dict,
    *,
    api: HfApi,
    cache_dir: Path,
    tokenizer_audit_texts: list[str] | None = None,
) -> dict:
    repository = spec["repository"]
    revision = spec["revision"]
    info = api.model_info(repository, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError(
            f"{repository}@{revision} resolved to unexpected {info.sha}")
    siblings = {row.rfilename: row for row in info.siblings or []}
    metadata_files = {}
    parsed = {}
    downloaded_paths = {}
    for filename in policy["required_metadata_files"]:
        if filename not in siblings:
            raise ValueError(f"{repository}@{revision} lacks {filename}")
        path = Path(hf_hub_download(
            repo_id=repository,
            filename=filename,
            revision=revision,
            cache_dir=str(cache_dir),
        ))
        metadata_files[filename] = {
            "bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
            "blob_id": siblings[filename].blob_id,
        }
        downloaded_paths[filename] = path
        if filename.endswith(".json"):
            parsed[filename] = json.loads(path.read_text())

    card = _card_dict(info)
    observed_bases = normalize_base_models(card.get("base_model"))
    expected_bases = sorted(spec.get("expected_base_models", []))
    ancestry_check = observed_bases == expected_bases
    contract = model_contract(
        parsed["config.json"], policy["expected_model_contract"])
    weights = weight_manifest(
        parsed["model.safetensors.index.json"], siblings)
    current = api.model_info(repository, files_metadata=False)
    refs = relevant_refs(api.list_repo_refs(repository)) \
        if spec.get("record_posttraining_refs") else []
    result = {
        "slug": spec["slug"],
        "role": spec["role"],
        "stage": spec["stage"],
        "repository": repository,
        "revision": revision,
        "model_uri": f"model://{repository}@{revision}",
        "public": not bool(info.private),
        "gated": info.gated,
        "last_modified": (
            info.last_modified.isoformat() if info.last_modified else None),
        "license": card.get("license"),
        "library_name": info.library_name,
        "pipeline_tag": info.pipeline_tag,
        "declared_base_models": observed_bases,
        "expected_base_models": expected_bases,
        "declared_ancestry_matches": ancestry_check,
        "ancestry_revision_qualification": (
            "Model-card base_model fields name repositories, not immutable "
            "parent revisions."),
        "model_contract": contract,
        "tokenizer_group": spec["tokenizer_group"],
        "tokenizer_json_sha256": metadata_files["tokenizer.json"]["sha256"],
        "metadata_files": metadata_files,
        "weights": weights,
        "observed_current_main_revision": current.sha,
        "target_is_current_main": current.sha == revision,
        "posttraining_refs": refs,
        "posttraining_step_ref_count": sum(
            row["name"] != "main" for row in refs),
        "configured_intermediate_candidate": bool(
            spec.get("eligible_intermediate")),
    }
    if tokenizer_audit_texts is not None:
        result["tokenizer_semantics"] = tokenizer_semantics(
            downloaded_paths["tokenizer.json"],
            downloaded_paths["tokenizer_config.json"],
            tokenizer_audit_texts,
        )
    return result


def _apply_group_compatibility(
    rows: list[dict],
    *,
    method: str = "byte-identical-tokenizer-json-v1",
) -> None:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["tokenizer_group"], []).append(row)
    for group_rows in groups.values():
        hashes = {row["tokenizer_json_sha256"] for row in group_rows}
        semantic_hashes = {
            row.get("tokenizer_semantics", {}).get(
                "semantic_fingerprint_sha256")
            for row in group_rows
        }
        semantic_hashes.discard(None)
        if method == "semantic-content-tokenization-v2":
            compatible = (
                len(semantic_hashes) == 1
                and len(semantic_hashes) == len({
                    row["tokenizer_semantics"][
                        "semantic_fingerprint_sha256"]
                    for row in group_rows
                }))
        elif method == "byte-identical-tokenizer-json-v1":
            compatible = len(hashes) == 1
        else:
            raise ValueError(f"unknown tokenizer compatibility method {method}")
        for row in group_rows:
            row["tokenizer_group_hashes"] = sorted(hashes)
            row["tokenizer_group_semantic_fingerprints"] = sorted(
                semantic_hashes)
            row["tokenizer_compatibility_method"] = method
            row["tokenizer_group_compatible"] = compatible
            checks = {
                "official_public_repository": row["public"] and not row["gated"],
                "declared_ancestry_matches": row["declared_ancestry_matches"],
                "model_contract": row["model_contract"]["passes"],
                "tokenizer_group_compatible": row["tokenizer_group_compatible"],
                "weights_available": row["weights"]["weights_available"],
            }
            row["placement_checks"] = checks
            row["intermediate_eligible"] = (
                row["configured_intermediate_candidate"]
                and all(checks.values()))


def _render_markdown(result: dict) -> str:
    lines = [
        "# OLMo 32B official checkpoint inventory",
        "",
        f"Evidence: `{result['evidence_id']}`",
        "",
        f"Retrieved: {result['retrieved_utc']}",
        "",
        "## Route",
        "",
        f"- Decision: `{result['route']['decision']}`.",
        f"- H5 status: `{result['route']['h5_status']}`.",
        "- This event opens no model or intervention outcome.",
        "",
        "## Official artifacts",
        "",
        "| Stage | Repository | Exact revision | Base declaration | "
        "Eligible intermediate | Step refs |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in result["artifacts"]:
        bases = ", ".join(row["declared_base_models"]) or "--"
        lines.append(
            f"| {row['stage']} | `{row['repository']}` | "
            f"`{row['revision'][:12]}` | `{bases}` | "
            f"{str(row['intermediate_eligible']).lower()} | "
            f"{row['posttraining_step_ref_count']} |")
    semantic_audit = result.get("tokenizer_semantic_audit")
    if semantic_audit is not None:
        lines.extend([
            "",
            "## Semantic tokenizer compatibility",
            "",
            f"The audit uses {semantic_audit['audit_texts']} frozen texts "
            "with `add_special_tokens=false`. Compatibility requires the "
            "same complete token-ID map, normalized BPE model, processing "
            "components, and audit encodings. BOS and chat-template metadata "
            "are recorded separately and must be frozen by any follow-up.",
            "",
            "| Group | JSON file hashes | Semantic fingerprints | Compatible |",
            "|---|---:|---:|---:|",
        ])
        groups: dict[str, list[dict]] = {}
        for row in result["artifacts"]:
            groups.setdefault(row["tokenizer_group"], []).append(row)
        for name, group_rows in sorted(groups.items()):
            raw_hashes = {row["tokenizer_json_sha256"] for row in group_rows}
            fingerprints = {
                row["tokenizer_semantics"]["semantic_fingerprint_sha256"]
                for row in group_rows
            }
            lines.append(
                f"| `{name}` | {len(raw_hashes)} | {len(fingerprints)} | "
                f"{str(group_rows[0]['tokenizer_group_compatible']).lower()} |")
    lines.extend([
        "",
        "## Minimal wedge",
        "",
        f"Status: `{result['route']['minimal_wedge']['status']}`.",
        "",
        ("The bounded queue contains the official 32B Think SFT and DPO "
         "cells; Base, 3.0 Think, and 3.1 Think are existing anchors. It "
         "does not license a full sweep over released training-step refs."
         if result["route"]["minimal_wedge"]["status"] == "queued-not-started"
         else "No stage wedge is queued under this inventory result."),
        "",
        "## Qualification",
        "",
        "Official model cards place the repositories on a Base -> SFT -> DPO "
        "-> RLVR graph, and exact target commits, small-file hashes, tokenizer "
        "hashes, and weight-shard LFS hashes are frozen here. The card "
        "`base_model` fields do not embed immutable parent revisions, so exact "
        "parent-weight ancestry remains qualified even though the released "
        "stage graph is sufficiently documented for a bounded follow-up.",
        "",
        "## Claim boundary",
        "",
        result["claim_boundary"],
        "",
    ])
    return "\n".join(lines)


def run(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    config = load_config(config_path)
    source = require_clean_tree(expected_branch=config["branch"])
    output_dir = metrics_dir("checkpoint-inventory")
    output_stem = config["evidence_id"]
    json_path = output_dir / f"{output_stem}.json"
    markdown_path = output_dir / f"{output_stem}.md"
    for path in (json_path, markdown_path):
        if path.exists():
            raise FileExistsError(
                f"refusing to overwrite immutable inventory output {path}")
    cache_dir = local_work() / "checkpoint_inventory_hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    semantic_policy = config.get("semantic_tokenizer_policy")
    tokenizer_audit = None
    tokenizer_audit_texts = None
    if semantic_policy is not None:
        tokenizer_audit, tokenizer_audit_texts = \
            load_tokenizer_audit_corpus(semantic_policy)
    api = HfApi()
    rows = [
        _inspect_target(
            spec,
            config["inventory_policy"],
            api=api,
            cache_dir=cache_dir,
            tokenizer_audit_texts=tokenizer_audit_texts,
        )
        for spec in config["targets"]
    ]
    _apply_group_compatibility(
        rows,
        method=(semantic_policy or {}).get(
            "compatibility_method", "byte-identical-tokenizer-json-v1"),
    )
    route = route_inventory(rows, config["route"])
    payload = {
        "schema_version": 1,
        "study_id": "jspace-olmo-lineage",
        "evidence_id": config["evidence_id"],
        "tier": config["tier"],
        "code_commit": source["code_commit"],
        "retrieved_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_sha256": file_sha256(config_path),
        "scientific_import_boundary": config["scientific_import_boundary"],
        "api_origin": config["inventory_policy"]["api_origin"],
        "official_organization": config["inventory_policy"][
            "official_organization"],
        "no_substitution_rule": config["inventory_policy"][
            "no_substitution_rule"],
        "artifacts": rows,
        "tokenizer_semantic_audit": tokenizer_audit,
        "route": route,
        "claim_boundary": config["claim_boundary"],
    }
    payload["payload_sha256"] = object_sha256(payload)
    atomic_json(json_path, payload)
    atomic_text(markdown_path, _render_markdown(payload))
    command = (
        "python -m jspace_olmo_lineage.experiments.checkpoint_inventory "
        f"--config {config_path}")
    event = create(
        config["evidence_id"],
        tier=config["tier"],
        what=("Official exact-revision OLMo 32B stage inventory and bounded "
              "H5 routing decision; no model outcome."),
        command=command,
        outputs=[json_path, markdown_path],
        inputs={
            "config_sha256": payload["config_sha256"],
            "api_origin": payload["api_origin"],
            "target_revisions": {
                row["slug"]: row["revision"] for row in rows
            },
            "tokenizer_audit_corpus_sha256": (
                tokenizer_audit["file_sha256"]
                if tokenizer_audit is not None else None),
        },
        verdict=route["decision"],
        claim_boundary=config["claim_boundary"],
    )
    supersession = None
    if config.get("supersedes"):
        supersession = supersede(
            config["supersedes"],
            config["evidence_id"],
            reason=config["supersession_reason"],
        )
    return {
        "status": "registered", "event": event,
        "supersession": supersession, "result": payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(PACKAGE_ROOT / "configs/ol_checkpoint_inventory_v1.yaml"),
    )
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
