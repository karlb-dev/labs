"""O3 audit of the four imported OLMo J-lenses before any refit.

This producer is deliberately CPU/read-only with respect to every imported
run root. It verifies the historical records, final lens hashes, internal lens
geometry, 4x30 slice/merge claim, tokenizer/BOS behavior, and pairwise recipe
and corpus classifications. Only the isolated OLMo run root and registry are
written.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

def _find_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path(__file__)).resolve()
    if cur.is_file():
        cur = cur.parent
    for p in [cur, *cur.parents]:
        if (p / ".git").exists():
            return p
    raise RuntimeError("cannot locate git repository root")


import torch
import yaml

from ..manifests import (
    atomic_text,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths import metrics_dir, resolve_uri, run_root
from ..provenance import Provenance, write_result
from ..registry import create

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = _find_repo_root()
PAIRWISE_CLASSIFICATIONS = {
    "EXACT_SAME_RECIPE_CORPUS",
    "SAME_RECIPE_DIFFERENT_CORPUS",
    "DIFFERENT_RECIPE",
    "UNKNOWN",
}


def load_config(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise TypeError("lens audit config must be a mapping")
    return value


def classify_pair(left: dict, right: dict) -> str:
    """Classify only recipe and raw fitting-corpus comparability.

    Tokenizer/BOS behavior is reported separately. A model-aware BOS
    difference under one shared fitting procedure is not silently relabeled a
    different corpus or recipe.
    """
    required = ("recipe_key", "corpus_key")
    if any(not left.get(key) or not right.get(key) for key in required):
        return "UNKNOWN"
    if left["recipe_key"] != right["recipe_key"]:
        return "DIFFERENT_RECIPE"
    if left["corpus_key"] != right["corpus_key"]:
        return "SAME_RECIPE_DIFFERENT_CORPUS"
    return "EXACT_SAME_RECIPE_CORPUS"


def _sample_indices(size: int, count: int = 33) -> torch.Tensor:
    if size <= 0:
        raise ValueError("cannot sample an empty tensor")
    steps = min(count, size)
    if steps == 1:
        return torch.zeros(1, dtype=torch.long)
    # Integer arithmetic avoids float32 rounding of large tensor endpoints.
    return torch.div(
        torch.arange(steps, dtype=torch.long) * (size - 1),
        steps - 1,
        rounding_mode="floor",
    )


def inspect_lens_checkpoint(
    path: str | Path,
    *,
    source_layers: list[int],
    d_model: int = 5120,
    n_prompts: int = 120,
    expected_dtype: torch.dtype = torch.float16,
) -> tuple[dict, dict]:
    """Validate a lens container and return its audit plus mmap checkpoint."""
    source = Path(path)
    checkpoint = torch.load(
        source, map_location="cpu", weights_only=True, mmap=True)
    expected_keys = {"J", "n_prompts", "source_layers", "d_model"}
    if set(checkpoint) != expected_keys:
        raise ValueError(
            f"{source} has keys {sorted(checkpoint)}, expected "
            f"{sorted(expected_keys)}")
    if int(checkpoint["n_prompts"]) != n_prompts:
        raise ValueError(
            f"{source} n_prompts={checkpoint['n_prompts']} != {n_prompts}")
    if int(checkpoint["d_model"]) != d_model:
        raise ValueError(
            f"{source} d_model={checkpoint['d_model']} != {d_model}")
    observed_layers = [int(value) for value in checkpoint["source_layers"]]
    if observed_layers != source_layers:
        raise ValueError(
            f"{source} source layers disagree: {observed_layers}")
    if set(checkpoint["J"]) != set(source_layers):
        raise ValueError(f"{source} Jacobian layer keys disagree")

    sampled_min = float("inf")
    sampled_max = float("-inf")
    sampled_count = 0
    for layer in source_layers:
        tensor = checkpoint["J"][layer]
        if tuple(tensor.shape) != (d_model, d_model):
            raise ValueError(
                f"{source} L{layer} shape {tuple(tensor.shape)}")
        if tensor.dtype != expected_dtype:
            raise ValueError(
                f"{source} L{layer} dtype {tensor.dtype} != "
                f"{expected_dtype}")
        flat = tensor.reshape(-1)
        values = flat[_sample_indices(flat.numel())].float()
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"{source} L{layer} sampled non-finite values")
        sampled_min = min(sampled_min, float(values.min()))
        sampled_max = max(sampled_max, float(values.max()))
        sampled_count += int(values.numel())
    return ({
        "path": str(source),
        "bytes": int(source.stat().st_size),
        "container_keys": sorted(checkpoint),
        "n_prompts": int(checkpoint["n_prompts"]),
        "d_model": int(checkpoint["d_model"]),
        "source_layers": observed_layers,
        "stored_dtype": str(expected_dtype).replace("torch.", ""),
        "sampled_values": sampled_count,
        "sampled_all_finite": True,
        "sampled_min": sampled_min,
        "sampled_max": sampled_max,
    }, checkpoint)


def merge_sample_diagnostic(
    merged: dict,
    slices: list[dict],
    *,
    layers: tuple[int, ...] = (24, 32, 40),
    tolerance: float = 0.002,
) -> dict:
    """Check sparse coordinates of the claimed equal-weight 4-slice merge."""
    if len(slices) != 4:
        raise ValueError(f"expected four slices, found {len(slices)}")
    maximum = 0.0
    checked = 0
    for layer in layers:
        target = merged["J"][layer]
        flat_indices = _sample_indices(target.numel(), count=41)
        target_values = target.reshape(-1)[flat_indices].float()
        mean_values = torch.stack([
            value["J"][layer].reshape(-1)[flat_indices].float()
            for value in slices
        ]).mean(dim=0)
        difference = (target_values - mean_values).abs()
        maximum = max(maximum, float(difference.max()))
        checked += int(difference.numel())
    if maximum > tolerance:
        raise ValueError(
            f"merged lens sample mismatch {maximum:.6g} > {tolerance}")
    return {
        "layers": list(layers),
        "sampled_coordinates": checked,
        "max_abs_merged_minus_slice_mean": maximum,
        "tolerance": tolerance,
        "passes": True,
    }


def validate_fit_metrics(metrics: dict, recipe: dict) -> dict:
    expected = {
        "source_layers": recipe["source_layers"],
        "target_layer": recipe["target_layer"],
        "dim_batch": recipe["dim_batch"],
        "max_seq_len": recipe["max_sequence_length"],
        "skip_first": recipe["skip_first"],
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise ValueError(
                f"fit metrics {key}={metrics.get(key)!r} != {value!r}")
    merged = metrics.get("merged", {})
    if int(merged.get("n_prompts", -1)) != recipe["fit_rows"]:
        raise ValueError("fit metrics do not attest a 120-prompt merge")
    slices = metrics.get("slices", {})
    if set(slices) != {"0", "1", "2", "3"}:
        raise ValueError("fit metrics do not contain exactly four slices")
    for index, row in slices.items():
        if int(row.get("n_prompts", -1)) != recipe["prompts_per_slice"]:
            raise ValueError(f"slice {index} does not attest 30 prompts")
        if int(row.get("prompts_done", -1)) != recipe["prompts_per_slice"]:
            raise ValueError(f"slice {index} is not complete")
    return {
        "source_layers": metrics["source_layers"],
        "target_layer": int(metrics["target_layer"]),
        "dim_batch": int(metrics["dim_batch"]),
        "max_sequence_length": int(metrics["max_seq_len"]),
        "skip_first": int(metrics["skip_first"]),
        "slice_count": len(slices),
        "slice_prompts": [int(slices[str(i)]["n_prompts"]) for i in range(4)],
        "merged_n_prompts": int(merged["n_prompts"]),
        "corpus_record": metrics.get("corpus"),
        "corpus_sha256_record": metrics.get("corpus_sha256"),
        "model_record": metrics.get("model") or metrics.get("model_uri"),
    }


def _verify_source_inputs(config: dict) -> tuple[list[dict], dict[str, Path]]:
    verified = []
    paths: dict[str, Path] = {}
    for row in config["source_inputs"]:
        path = resolve_uri(row["uri"])
        observed = file_sha256(path)
        if observed != row["sha256"]:
            raise ValueError(
                f"source input {row['id']} hash {observed} != "
                f"{row['sha256']}")
        paths[row["id"]] = path
        verified.append({
            "id": row["id"], "uri": row["uri"], "path": str(path),
            "sha256": observed, "bytes": int(path.stat().st_size),
        })
    return verified, paths


def _verify_git_blobs(config: dict) -> list[dict]:
    verified = []
    for row in config["git_blob_inputs"]:
        value = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "show",
             f"{row['commit']}:{row['path']}"])
        observed = hashlib.sha256(value).hexdigest()
        if observed != row["sha256"]:
            raise ValueError(
                f"git blob {row['id']} hash {observed} != {row['sha256']}")
        verified.append({
            "id": row["id"], "commit": row["commit"],
            "path": row["path"], "sha256": observed,
            "bytes": len(value),
        })
    return verified


def _verify_shared_corpus(recipe: dict) -> tuple[dict, list[str]]:
    path = resolve_uri(recipe["corpus_uri"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != recipe["corpus_file_sha256"]:
        raise ValueError("shared fitting-corpus file hash mismatch")
    lines = raw.splitlines(keepends=True)
    if len(lines) != recipe["corpus_rows_total"]:
        raise ValueError("shared fitting-corpus row count mismatch")
    rows = [json.loads(line) for line in raw.splitlines()]
    texts = [str(row["text"]) for row in rows]
    all_text_hash = hashlib.sha256(
        "".join(texts).encode("utf-8")).hexdigest()
    first_text_hash = hashlib.sha256(
        "".join(texts[:recipe["fit_rows"]]).encode("utf-8")).hexdigest()
    prefix_hash = hashlib.sha256(
        b"".join(lines[:recipe["fit_rows"]])).hexdigest()
    expected = {
        "all_text_hash": recipe["all_200_concatenated_text_sha256"],
        "first_text_hash": recipe["first_120_concatenated_text_sha256"],
        "prefix_hash": recipe["first_120_jsonl_prefix_sha256"],
    }
    observed = {
        "all_text_hash": all_text_hash,
        "first_text_hash": first_text_hash,
        "prefix_hash": prefix_hash,
    }
    if observed != expected:
        raise ValueError(
            f"shared fitting-corpus derived hashes disagree: {observed}")
    return ({
        "uri": recipe["corpus_uri"], "path": str(path),
        "file_sha256": recipe["corpus_file_sha256"],
        "rows_total": len(rows), "fit_rows": recipe["fit_rows"],
        "fit_order": recipe["fit_order"],
        "all_200_concatenated_text_sha256": all_text_hash,
        "first_120_concatenated_text_sha256": first_text_hash,
        "first_120_jsonl_prefix_sha256": prefix_hash,
    }, texts[:recipe["fit_rows"]])


def _tokenization_audit(models: list[dict], texts: list[str]) -> dict:
    from transformers import AutoTokenizer

    result = {}
    sequences = {}
    for model in models:
        snapshot = Path(model["tokenizer_snapshot"])
        tokenizer_file = snapshot / "tokenizer.json"
        observed_hash = file_sha256(tokenizer_file)
        if observed_hash != model["tokenizer_json_sha256"]:
            raise ValueError(f"{model['slug']} tokenizer hash mismatch")
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), local_files_only=True)
        if (getattr(tokenizer, "bos_token_id", None) is not None
                and hasattr(tokenizer, "add_bos_token")):
            tokenizer.add_bos_token = True
        bos = getattr(tokenizer, "bos_token_id", None)
        if bos != model["expected_bos_token_id"]:
            raise ValueError(
                f"{model['slug']} BOS {bos} != "
                f"{model['expected_bos_token_id']}")
        encoded = [
            tokenizer(
                text, truncation=True,
                max_length=128)["input_ids"]
            for text in texts
        ]
        sequences[model["slug"]] = encoded
        lengths = [len(row) for row in encoded]
        result[model["slug"]] = {
            "tokenizer_snapshot": str(snapshot),
            "tokenizer_json_sha256": observed_hash,
            "tokenizer_class": type(tokenizer).__name__,
            "bos_token_id": bos,
            "add_bos_token_after_jlens_policy": getattr(
                tokenizer, "add_bos_token", None),
            "sequence_manifest_sha256": object_sha256(encoded),
            "n_sequences": len(encoded),
            "minimum_length": min(lengths),
            "maximum_length": max(lengths),
        }
    for left, right in itertools.combinations(models, 2):
        left_slug, right_slug = left["slug"], right["slug"]
        equal = [
            a == b for a, b in zip(
                sequences[left_slug], sequences[right_slug], strict=True)
        ]
        key = f"{left_slug}__{right_slug}"
        result[key] = {
            "kind": "pairwise_tokenization",
            "identical_sequences": int(sum(equal)),
            "n_sequences": len(equal),
            "all_identical": bool(all(equal)),
            "same_raw_texts_and_order": True,
        }
    return result


def _verify_historical_weight_identity(model: dict) -> dict | None:
    historical = model.get("historical_cache_revision_candidate")
    if not historical:
        return None
    pinned = Path(model["tokenizer_snapshot"])
    historical_path = pinned.parent / historical

    def blob_targets(path: Path) -> dict[str, str]:
        return {
            item.name: str(item.readlink())
            for item in sorted(path.glob("model-*.safetensors"))
            if item.is_symlink()
        }

    pinned_targets = blob_targets(pinned)
    historical_targets = blob_targets(historical_path)
    identical = (
        len(pinned_targets) == 14
        and pinned_targets == historical_targets
    )
    expected = bool(model["historical_and_pinned_weight_blobs_identical"])
    if identical != expected:
        raise ValueError("historical/pinned OLMo-3 Think weights disagree")
    pinned_index = json.loads(
        (pinned / "model.safetensors.index.json").read_text())
    historical_index = json.loads(
        (historical_path / "model.safetensors.index.json").read_text())
    weight_map_equal = pinned_index["weight_map"] == historical_index["weight_map"]
    if not weight_map_equal:
        raise ValueError("historical/pinned weight maps disagree")
    return {
        "historical_revision_candidate": historical,
        "pinned_revision": model["model_revision"],
        "weight_shard_blob_targets_identical": identical,
        "n_weight_shards": len(pinned_targets),
        "weight_map_semantically_identical": weight_map_equal,
        "config_sha256_historical": file_sha256(
            historical_path / "config.json"),
        "config_sha256_pinned": file_sha256(pinned / "config.json"),
        "qualification": (
            "The original fit loaded the unpinned Hub model ID. The surviving "
            "historical cache revision is therefore a reconstruction, but its "
            "14 weight blob targets and semantic weight map match the pinned "
            "revision exactly."),
    }


def _audit_model(model: dict, recipe: dict) -> dict:
    lens_path = resolve_uri(model["lens_uri"])
    observed_lens_hash = file_sha256(lens_path)
    if observed_lens_hash != model["lens_sha256"]:
        raise ValueError(f"{model['slug']} final lens hash mismatch")
    final_audit, merged = inspect_lens_checkpoint(
        lens_path,
        source_layers=recipe["source_layers"],
        n_prompts=recipe["fit_rows"],
    )

    metrics_path = resolve_uri(model["fit_metrics_uri"])
    if file_sha256(metrics_path) != model["fit_metrics_sha256"]:
        raise ValueError(f"{model['slug']} fit metrics hash mismatch")
    metrics_record = validate_fit_metrics(
        json.loads(metrics_path.read_text()), recipe)

    supporting = []
    for prefix in ("fit_provenance", "fit_sanity"):
        uri = model.get(f"{prefix}_uri")
        if not uri:
            continue
        path = resolve_uri(uri)
        expected = model[f"{prefix}_sha256"]
        observed = file_sha256(path)
        if observed != expected:
            raise ValueError(
                f"{model['slug']} {prefix} hash mismatch")
        supporting.append({
            "kind": prefix, "uri": uri, "path": str(path),
            "sha256": observed, "bytes": int(path.stat().st_size),
        })

    slice_audits = []
    slice_checkpoints = []
    for slice_row in model["slices"]:
        uri = slice_row["uri"]
        path = resolve_uri(uri)
        observed_slice_hash = file_sha256(path)
        if observed_slice_hash != slice_row["sha256"]:
            raise ValueError(
                f"{model['slug']} slice hash mismatch: {uri}")
        audit, checkpoint = inspect_lens_checkpoint(
            path,
            source_layers=recipe["source_layers"],
            n_prompts=recipe["prompts_per_slice"],
        )
        audit["uri"] = uri
        audit["sha256"] = observed_slice_hash
        slice_audits.append(audit)
        slice_checkpoints.append(checkpoint)
    merge_diagnostic = merge_sample_diagnostic(merged, slice_checkpoints)
    historical_identity = _verify_historical_weight_identity(model)

    return {
        "slug": model["slug"],
        "role": model["role"],
        "model_id": model["model_id"],
        "model_revision": model["model_revision"],
        "source_evidence_id": model["source_evidence_id"],
        "lens_uri": model["lens_uri"],
        "lens_sha256": observed_lens_hash,
        "recipe_key": model["recipe_key"],
        "corpus_key": model["corpus_key"],
        "corpus_attestation": model["corpus_attestation"],
        "provenance_strength": model["provenance_strength"],
        "fit_metrics": metrics_record,
        "final_lens": final_audit,
        "slices": slice_audits,
        "merge_diagnostic": merge_diagnostic,
        "supporting_records": supporting,
        "independent_fit_evidence": model["independent_fit_evidence"],
        "heldout_diagnostics": model["heldout_diagnostics"],
        "historical_weight_identity": historical_identity,
    }


def _markdown(payload: dict) -> str:
    lines = [
        "# OLMo four-lens provenance audit v1",
        "",
        "Tier: methods. Imported source artifacts were read-only; no lens was refit.",
        "",
        "## Result",
        "",
        ("All six checkpoint pairs classify as "
         "`EXACT_SAME_RECIPE_CORPUS` for the same ordered 120 raw texts and "
         "the same J-lens fitting procedure. The decision is "
         "`no_refit_run_geometry_analysis`."),
        "",
        "| lens | provenance strength | corpus attestation | BOS | slices | merge check |",
        "|---|---|---|---:|---:|---|",
    ]
    tokenization = payload["tokenization"]
    for slug, row in payload["lenses"].items():
        bos = tokenization[slug]["bos_token_id"]
        lines.append(
            f"| {slug} | {row['provenance_strength']} | "
            f"{row['corpus_attestation']} | {bos} | "
            f"{len(row['slices'])} x 30 | pass |")
    lines.extend([
        "",
        "## Pairwise classification",
        "",
        "| pair | classification | token sequences identical |",
        "|---|---|---|",
    ])
    for row in payload["pairwise_comparability"]:
        lines.append(
            f"| {row['left']} / {row['right']} | "
            f"{row['classification']} | "
            f"{str(row['token_sequences_all_identical']).lower()} |")
    lines.extend([
        "",
        "## Qualifications",
        "",
        "- Base exposes no BOS token; the shared `jlens.from_hf` policy adds BOS only when the tokenizer exposes one. The three post-trained token sequences are identical on all 120 fit texts; Base uses the same raw texts/order but its native no-BOS encoding.",
        "- The Instruct fit record names the deterministic shared corpus path but does not embed its hash. Source code, the 4 x 30 records, final-lens metadata, and the independent draw-B fit corroborate the claim; this is reported as qualified provenance rather than silently upgraded.",
        "- The original OLMo-3 Think fit loaded an unpinned Hub ID. Its surviving historical cache revision and the pinned revision reference the same 14 weight blobs and semantic weight map; the historical revision attribution remains a reconstruction.",
        "- Dedicated held-out fit diagnostics are uneven. This does not require a same-corpus refit, but it limits claims until the registered geometry and functional analyses are complete.",
        "- Same-corpus comparability does not imply identical operators, token dictionaries, selected supports, or causal use.",
        "",
        "## Decision",
        "",
        "No refit is justified by this audit. Proceed to registered same-corpus geometry analyses and the symmetric four-model capacity table.",
        "",
    ])
    return "\n".join(lines)


def run(config_path: str | Path) -> dict:
    source = Path(config_path).resolve()
    config = load_config(source)
    git = require_clean_tree(expected_branch=config["branch"])
    if run_root().resolve() != Path(config["run_root"]).resolve():
        raise RuntimeError("configured and environment run roots disagree")
    if set(config["classifications"]["allowed"]) != PAIRWISE_CLASSIFICATIONS:
        raise ValueError("pairwise classification vocabulary drift")

    output_dir = metrics_dir("lens-provenance")
    result_path = output_dir / "ol_lens_provenance_audit_v1.json"
    markdown_path = output_dir / "ol_lens_provenance_audit_v1.md"
    collisions = [
        str(path) for path in (result_path, markdown_path) if path.exists()
    ]
    if collisions:
        raise FileExistsError(
            "lens audit outputs are immutable and already exist: "
            + ", ".join(collisions))

    source_inputs, _ = _verify_source_inputs(config)
    git_blobs = _verify_git_blobs(config)
    corpus_audit, fit_texts = _verify_shared_corpus(config["shared_recipe"])
    tokenization = _tokenization_audit(config["models"], fit_texts)
    lenses = {
        model["slug"]: _audit_model(model, config["shared_recipe"])
        for model in config["models"]
    }

    pairwise = []
    for left, right in itertools.combinations(config["models"], 2):
        classification = classify_pair(left, right)
        if classification not in PAIRWISE_CLASSIFICATIONS:
            raise ValueError("unrecognized pairwise classification")
        token_key = f"{left['slug']}__{right['slug']}"
        pairwise.append({
            "left": left["slug"], "right": right["slug"],
            "classification": classification,
            "same_raw_texts_and_order": True,
            "token_sequences_all_identical": tokenization[token_key][
                "all_identical"],
            "token_sequences_identical_n": tokenization[token_key][
                "identical_sequences"],
            "token_sequences_n": tokenization[token_key]["n_sequences"],
        })

    all_exact = all(
        row["classification"] == "EXACT_SAME_RECIPE_CORPUS"
        for row in pairwise)
    all_lenses_valid = all(
        row["merge_diagnostic"]["passes"]
        and row["final_lens"]["sampled_all_finite"]
        for row in lenses.values())
    decision = (
        "no_refit_run_geometry_analysis"
        if all_exact and all_lenses_valid
        else "refit_decision_required"
    )
    expected_decision = config["decision_rule"][
        "expected_if_inputs_validate"]
    if decision != expected_decision:
        raise RuntimeError(
            f"audit decision {decision!r} != expected {expected_decision!r}")

    payload = {
        "schema_version": 1,
        "study_id": "jspace-olmo-lineage",
        "evidence_id": config["evidence_id"],
        "status": "complete",
        "audit_scope": (
            "Four imported OLMo 32B own lenses; provenance, corpus/recipe "
            "comparability, tokenizer/BOS behavior, and slice-merge integrity"),
        "source_inputs": source_inputs,
        "git_blob_inputs": git_blobs,
        "shared_corpus": corpus_audit,
        "shared_recipe": config["shared_recipe"],
        "lenses": lenses,
        "tokenization": tokenization,
        "pairwise_comparability": pairwise,
        "summary": {
            "n_lenses": len(lenses),
            "n_pairs": len(pairwise),
            "all_pairs_exact_same_recipe_corpus": all_exact,
            "all_lens_and_merge_checks_pass": all_lenses_valid,
            "refit_decision": decision,
            "geometry_analysis_authorized": True,
            "capacity_analysis_authorized": True,
            "intervention_outcomes_opened": False,
        },
        "limitations": [
            "Instruct's primary fit record lacks an embedded corpus hash; deterministic source and artifact evidence corroborate it.",
            "The original OLMo-3 Think fit used an unpinned Hub ID; exact weight-blob identity bridges the surviving historical and pinned revisions, but fit-time revision attribution is reconstructed.",
            "Fit-time runtime metadata and dedicated held-out diagnostics are uneven across the four historical lenses.",
            "Same recipe/corpus comparability is not evidence that the learned operators or dictionaries are identical.",
        ],
        "git_at_audit": git,
    }
    config_hash = file_sha256(source)
    inputs = {
        "config": config_hash,
        "shared_corpus": corpus_audit["file_sha256"],
        **{
            f"lens:{slug}": row["lens_sha256"]
            for slug, row in lenses.items()
        },
    }
    envelope = write_result(
        payload,
        result_path,
        Provenance(
            evidence_id=config["evidence_id"],
            tier=config["tier"],
            command=(
                "python -m jspace_olmo_lineage.experiments."
                "lens_provenance_audit --config "
                f"{source.relative_to(REPO_ROOT)}"),
            inputs=inputs,
            input_manifest_sha256=config_hash,
            seed_contract="no scientific RNG; deterministic sparse tensor samples",
        ),
    )
    atomic_text(markdown_path, _markdown(payload))
    event = create(
        config["evidence_id"],
        tier=config["tier"],
        what=(
            "Audit of four OLMo own-lens provenance records, exact shared "
            "draw-A corpus/recipe comparability, tokenizer/BOS behavior, and "
            "4x30 slice merges; no refit required and no intervention opened."),
        command=envelope["provenance"]["command"],
        outputs=[result_path, markdown_path],
        inputs=inputs,
        refit_decision=decision,
        all_pairs_exact_same_recipe_corpus=all_exact,
        interventions_opened=False,
    )
    return {"payload": payload, "registry_event": event}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config), indent=1))


if __name__ == "__main__":
    main()
