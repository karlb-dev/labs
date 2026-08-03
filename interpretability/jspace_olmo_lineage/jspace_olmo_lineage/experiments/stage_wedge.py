"""GPU-only OLMo Study-2 SFT/DPO stage-wedge producer.

This producer deliberately owns no Phase-4 registry state.  It reuses the
hash-pinned Phase-3/4 scoring and intervention implementations, but all
checkpoints, immutable outputs, and evidence events are written below the
isolated OLMo Study-2 root.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
import yaml
from jspace_part2.dictionaries import (
    build_j_dictionaries,
    build_logit_dictionary,
)
from jspace_phase3.ablator3 import (
    Phase3JAblator,
    profile_from_p3log,
    teacher_forced_matched_arm,
)
from jspace_phase3.bank import load_bank
from jspace_phase4.scoring4 import (
    DEFAULT_SPEC,
    ScoringSession,
    aggregate_alias_lps,
    canonical_alias_for,
)
from jspace_phase4.seeds import SEED_CONTRACT, stable_rng, stable_seed
from jspace_phase4.state import StateHeader, StateStore

from ..gpu import assert_model_on_cuda, require_cuda_gpu
from ..manifests import (
    InputManifest,
    atomic_json,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths import local_work, metrics_dir, resolve_model, resolve_uri, run_root
from ..provenance import Provenance, write_result
from ..registry import RegistryError, create, resolve
from .checkpoint_inventory import tokenizer_semantics

STUDY_ID = "jspace-olmo-lineage-study2"
RAW_CONDITIONS = (
    "meanJ_span_safe",
    "meanJ_label_protected",
    "mechanics_random",
    "logit_label_protected",
)
MATCHED_CONDITIONS = (
    ("instant_rank_energy_matched", "instant_rank_energy_matched", "span_safe"),
    ("protected_energy_matched", "prot_energy_matched", "label"),
)
ALL_CONDITIONS = (
    "baseline",
    "meanJ_span_safe",
    "instant_rank_energy_matched",
    "meanJ_label_protected",
    "protected_energy_matched",
    "mechanics_random",
    "logit_label_protected",
)
FRAME_SPECS = (
    ("base-lens-common", "frozen_base_lens"),
    ("olmo3-think-endpoint-own", "frozen_olmo3_think_lens"),
)
TOKENIZER_FIELDS = (
    "semantic_fingerprint_sha256",
    "token_id_map_sha256",
    "normalized_model_sha256",
    "processing_components_sha256",
)
ENERGY_RELATIVE_FLOOR = 1e-3
ENERGY_RELATIVE_TOLERANCE = 5e-3
ENERGY_ABSOLUTE_TOLERANCE_BELOW_FLOOR = 5e-5
PROTECTED_COSINE_TOLERANCE = 1e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        required=True,
        choices=("think_sft", "think_dpo"),
    )
    parser.add_argument(
        "--phase",
        default="all",
        choices=("preflight", "capability", "tier1", "register", "all"),
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise TypeError("stage-wedge config must be a mapping")
    return value


def model_reference(specification: Mapping) -> dict:
    return {
        "model_id": str(specification["model_id"]),
        "revision": str(specification["revision"]),
        "stage": str(specification["stage"]),
        "slug": str(specification["slug"]),
    }


def load_items(paths: Sequence[Path]) -> list[dict]:
    items: list[dict] = []
    for path in paths:
        for bundle in load_bank(path):
            items.extend(bundle.as_items())
    items.sort(key=lambda row: str(row["item_id"]))
    if len({str(row["item_id"]) for row in items}) != len(items):
        raise RuntimeError("G5 battery item IDs are not unique")
    return items


def exact_grade_alias(
    session: ScoringSession,
    generated: str,
    aliases: Sequence[str],
) -> str | None:
    """Frozen Study-2 capability rule: normalized continuation equality."""
    normalized = session.spec.normalize_generation(generated)
    matches = [
        alias
        for alias in aliases
        if normalized == session.spec.normalize_generation(alias)
    ]
    if not matches:
        return None
    return max(matches, key=lambda value: (len(value), value))


def _hash_text_rows(texts: Sequence[str]) -> str:
    return hashlib.sha256("".join(texts).encode("utf-8")).hexdigest()


def _tokenizer_audit_texts(ancestry: Mapping) -> tuple[list[str], dict]:
    audit = ancestry["tokenizer_semantic_audit"]
    path = Path(audit["path"])
    if file_sha256(path) != audit["file_sha256"]:
        raise RuntimeError("frozen tokenizer audit corpus hash mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    texts = [str(row["text"]) for row in rows]
    if len(texts) != int(audit["corpus_rows"]):
        raise RuntimeError("frozen tokenizer audit corpus row mismatch")
    if _hash_text_rows(texts) != audit["concatenated_text_sha256"]:
        raise RuntimeError("frozen tokenizer audit text hash mismatch")
    all_texts = texts + [str(value) for value in audit["edge_cases"]]
    if len(all_texts) != int(audit["audit_texts"]):
        raise RuntimeError("frozen tokenizer audit size mismatch")
    return all_texts, dict(audit)


def _weight_rows(snapshot: Path, expected: Mapping) -> list[dict]:
    rows = []
    for row in expected["shards"]:
        path = snapshot / row["name"]
        if not path.is_file():
            raise RuntimeError(f"snapshot lacks {row['name']}")
        observed = {
            "name": str(row["name"]),
            "bytes": int(path.stat().st_size),
            "sha256": file_sha256(path),
        }
        if observed != {
            "name": str(row["name"]),
            "bytes": int(row["bytes"]),
            "sha256": str(row["sha256"]),
        }:
            raise RuntimeError(
                f"exact snapshot shard mismatch: {row['name']}: {observed}"
            )
        rows.append(observed)
    return rows


def tokenizer_contract_checks(
    semantics: Mapping,
    expected_tokenizer: Mapping,
) -> dict[str, bool]:
    checks = {
        field: semantics[field] == expected_tokenizer[field]
        for field in TOKENIZER_FIELDS
    }
    checks["audit_encoding_sha256"] = (
        semantics["audit_encoding_sha256"]
        == expected_tokenizer["frozen_audit_encoding_sha256"]
    )
    return checks


def snapshot_preflight(
    config: Mapping,
    checkpoint_key: str,
    snapshot: Path,
    output_dir: Path,
) -> dict:
    """Verify exact bytes and semantic tokenizer behavior before model load."""
    checkpoint = config["checkpoints"][checkpoint_key]
    ancestry_path = run_root(create=False) / "manifests/ol2_checkpoint_ancestry_v1.json"
    ancestry = json.loads(ancestry_path.read_text())
    artifact = ancestry["artifacts"][checkpoint["slug"]]
    expected_revision = str(checkpoint["revision"])
    if artifact["revision"] != expected_revision:
        raise RuntimeError("ancestry/config checkpoint revision mismatch")
    if snapshot.name != expected_revision:
        raise RuntimeError(
            f"resolved snapshot is not the pinned revision directory: {snapshot}"
        )

    metadata = {}
    for name, expected in artifact["metadata_files"].items():
        path = snapshot / name
        observed = {
            "bytes": int(path.stat().st_size) if path.is_file() else None,
            "sha256": file_sha256(path) if path.is_file() else None,
        }
        required = {
            "bytes": int(expected["bytes"]),
            "sha256": str(expected["sha256"]),
        }
        if observed != required:
            raise RuntimeError(f"snapshot metadata mismatch for {name}: {observed}")
        metadata[name] = observed

    weight_rows = _weight_rows(snapshot, artifact["weights"])
    weight_manifest_sha256 = object_sha256(weight_rows)
    if weight_manifest_sha256 != artifact["weights"]["manifest_sha256"]:
        raise RuntimeError("snapshot weight manifest mismatch")

    model_config = json.loads((snapshot / "config.json").read_text())
    contract = config["checkpoints"]["architecture_contract"]
    architecture_checks = {
        "architecture": contract["architecture"] in model_config["architectures"],
        "model_type": model_config["model_type"] == contract["model_type"],
        "hidden_size": model_config["hidden_size"] == contract["hidden_size"],
        "vocab_size": model_config["vocab_size"] == contract["vocab_size"],
        "decoder_layers": (
            model_config["num_hidden_layers"] == contract["decoder_layers"]
        ),
        "rms_norm_eps": model_config["rms_norm_eps"] == contract["rms_norm_eps"],
        "tie_word_embeddings": (
            model_config["tie_word_embeddings"] == contract["tie_word_embeddings"]
        ),
    }
    if not all(architecture_checks.values()):
        raise RuntimeError(
            f"snapshot architecture contract failed: {architecture_checks}"
        )

    audit_texts, audit_source = _tokenizer_audit_texts(ancestry)
    semantics = tokenizer_semantics(
        snapshot / "tokenizer.json",
        snapshot / "tokenizer_config.json",
        audit_texts,
    )
    expected_tokenizer = config["tokenizer_contract"]
    tokenizer_checks = tokenizer_contract_checks(semantics, expected_tokenizer)
    tokenizer_checks["audit_text_count"] = semantics["audit_text_count"] == int(
        audit_source["audit_texts"]
    )
    if not all(tokenizer_checks.values()):
        raise RuntimeError(f"semantic tokenizer contract failed: {tokenizer_checks}")

    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot)
    session = ScoringSession(tokenizer, DEFAULT_SPEC, device="cpu")
    bos_probe = session.prompt_ids("frozen BOS probe")[0].tolist()
    bos_id = tokenizer.bos_token_id
    bos_checks = {
        "bos_prefixed": session.bos_prefixed,
        "bos_id_present": bos_id is not None,
        "exactly_one_leading_bos": (
            bool(bos_probe)
            and bos_id is not None
            and int(bos_probe[0]) == int(bos_id)
            and (len(bos_probe) < 2 or int(bos_probe[1]) != int(bos_id))
        ),
        "chat_template_unused": True,
    }
    if not all(bos_checks.values()):
        raise RuntimeError(f"BOS interface contract failed: {bos_checks}")

    payload = {
        "schema_version": 1,
        "checkpoint_key": checkpoint_key,
        "model_id": checkpoint["model_id"],
        "revision": expected_revision,
        "snapshot": str(snapshot),
        "snapshot_directory_revision_exact": True,
        "metadata_files": metadata,
        "weight_shards": weight_rows,
        "weight_manifest_sha256": weight_manifest_sha256,
        "weight_bytes": sum(row["bytes"] for row in weight_rows),
        "architecture_checks": architecture_checks,
        "tokenizer_semantics": semantics,
        "tokenizer_checks": tokenizer_checks,
        "bos_checks": bos_checks,
        "ancestry_manifest": {
            "path": str(ancestry_path),
            "sha256": file_sha256(ancestry_path),
        },
        "all_hard_gates_passed": True,
        "model_outcome_opened": False,
    }
    path = output_dir / "snapshot_conformance.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != payload:
            raise RuntimeError("snapshot conformance changed across replay")
    else:
        atomic_json(path, payload)
    return payload


def _input_manifest(
    config: Mapping,
    config_path: Path,
    checkpoint_key: str,
    preflight: Mapping,
    bank_paths: Sequence[Path],
    clean: Mapping,
) -> InputManifest:
    checkpoint = config["checkpoints"][checkpoint_key]
    bank_hashes = {
        str(uri): file_sha256(path)
        for uri, path in zip(
            (config["inputs"]["bank_f"]["uri"], config["inputs"]["bank_s"]["uri"]),
            bank_paths,
        )
    }
    lens_hashes = {frame: config["inputs"][key]["sha256"] for frame, key in FRAME_SPECS}
    foundation_path = run_root(create=False) / "manifests/ol2_foundation_v1.json"
    ancestry_path = run_root(create=False) / "manifests/ol2_checkpoint_ancestry_v1.json"
    return InputManifest(
        experiment_id=checkpoint["evidence_id"],
        config_sha256=file_sha256(config_path),
        model_id=checkpoint["model_id"],
        model_revision=checkpoint["revision"],
        tokenizer_manifest_sha256=preflight["tokenizer_semantics"][
            "semantic_fingerprint_sha256"
        ],
        lens_sha256=object_sha256(lens_hashes),
        bank_sha256=object_sha256(bank_hashes),
        partition_sha256=object_sha256(
            {
                "battery_rows_expected": config["g5_capability"][
                    "battery_rows_expected"
                ],
                "banks": config["g5_capability"]["banks"],
                "item_order": "lexicographic-item-id",
            }
        ),
        scoring_spec_sha256=object_sha256(DEFAULT_SPEC.as_dict()),
        upstream={
            str(foundation_path): file_sha256(foundation_path),
            str(ancestry_path): file_sha256(ancestry_path),
            **bank_hashes,
            **{
                config["inputs"][key]["uri"]: config["inputs"][key]["sha256"]
                for _, key in FRAME_SPECS
            },
        },
        code_commit=clean["code_commit"],
    )


def heartbeat(output_dir: Path, **fields: object) -> None:
    payload = {
        "schema_version": 1,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": os.getpid(),
        **fields,
    }
    atomic_json(output_dir / "WATCHDOG_HEARTBEAT.json", payload)


def _generation_grade(
    session: ScoringSession,
    generated: str,
    original_aliases: Sequence[str],
    counterfactual_aliases: Sequence[str],
) -> dict:
    original = exact_grade_alias(session, generated, original_aliases)
    counterfactual = exact_grade_alias(session, generated, counterfactual_aliases)
    if original is not None and counterfactual is None:
        outcome = "original"
    elif counterfactual is not None and original is None:
        outcome = "counterfactual"
    elif original is not None and counterfactual is not None:
        outcome = "ambiguous_original_counterfactual"
    else:
        outcome = "other_invalid"
    return {
        "outcome": outcome,
        "matched_original_alias": original,
        "matched_counterfactual_alias": counterfactual,
    }


@torch.no_grad()
def run_capability(
    model,
    tokenizer,
    config: Mapping,
    checkpoint_key: str,
    manifest: InputManifest,
    items: Sequence[Mapping],
    output_dir: Path,
    gpu: Mapping,
) -> tuple[pd.DataFrame, dict, dict]:
    checkpoint = config["checkpoints"][checkpoint_key]
    session = ScoringSession(tokenizer, DEFAULT_SPEC, device="cuda")
    expected = int(config["g5_capability"]["battery_rows_expected"])
    if len(items) != expected:
        raise RuntimeError(
            f"frozen G5 battery expected {expected} rows, got {len(items)}"
        )
    state_store = StateStore(
        output_dir / "g5_state.json",
        StateHeader(
            evidence_id=checkpoint["evidence_id"] + ":g5",
            input_manifest_sha256=manifest.sha256(),
            config_sha256=manifest.config_sha256,
            model_revision=manifest.model_revision,
            bank_sha256=manifest.bank_sha256,
            partition_sha256=manifest.partition_sha256,
        ),
    )
    state = state_store.load() or {"done": {}, "rows": [], "gpu": dict(gpu)}
    started = time.time()
    newly_done = 0
    checkpoint_every = int(config["g5_capability"]["checkpoint_every_items"])
    for item in items:
        item_id = str(item["item_id"])
        if item_id in state["done"]:
            continue
        prompt_ids = session.prompt_ids(str(item["prompt"]))
        prompt_length = int(prompt_ids.shape[1])
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=int(config["g5_capability"]["max_new_tokens"]),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(
            generated_ids[0, prompt_length:], skip_special_tokens=True
        )
        matched = exact_grade_alias(session, generated, item["accepted_answers"])
        canonical_alias = canonical_alias_for(
            session, item["accepted_answers"], item["canonical_answer"]
        )
        alias_manifest = session.freeze_alias_manifest(
            item["accepted_answers"], canonical_alias=canonical_alias
        )
        lp_by_alias = {}
        for alias in alias_manifest["prefix_disjoint_aliases"]:
            full_ids, n_prompt = session.full_ids(item["prompt"], alias)
            logits = model(input_ids=full_ids, use_cache=False).logits[0]
            lp_by_alias[alias] = session.answer_sequence_lp(full_ids, logits, n_prompt)
            del logits

        counterfactual_manifest = None
        counterfactual_lp_by_alias: dict[str, float] = {}
        if item["counterfactual_accepted"]:
            counterfactual_canonical = canonical_alias_for(
                session,
                item["counterfactual_accepted"],
                item["counterfactual_answer"],
            )
            counterfactual_manifest = session.freeze_alias_manifest(
                item["counterfactual_accepted"],
                canonical_alias=counterfactual_canonical,
            )
            for alias in counterfactual_manifest["prefix_disjoint_aliases"]:
                full_ids, n_prompt = session.full_ids(item["prompt"], alias)
                logits = model(input_ids=full_ids, use_cache=False).logits[0]
                counterfactual_lp_by_alias[alias] = session.answer_sequence_lp(
                    full_ids, logits, n_prompt
                )
                del logits
        grade = _generation_grade(
            session,
            generated,
            item["accepted_answers"],
            item["counterfactual_accepted"],
        )
        row = {
            "study_id": STUDY_ID,
            "phase": "development",
            "tier": config["tier"],
            "evidence_id": checkpoint["evidence_id"],
            "checkpoint_key": checkpoint_key,
            "checkpoint_stage": checkpoint["stage"],
            "model_id": checkpoint["model_id"],
            "model_revision": checkpoint["revision"],
            "config_sha256": manifest.config_sha256,
            "bank_sha256": manifest.bank_sha256,
            "item_id": item_id,
            "fact_id": str(item["fact_id"]),
            "canonical_family": str(item["canonical_family"]),
            "relation_group": str(item["relation_group"]),
            "variant": str(item["variant"]),
            "bank": str(item["bank"]),
            "generation": generated[:256],
            "capable_generation": matched is not None,
            "matched_alias": matched,
            "generation_outcome": grade["outcome"],
            "alias_set_hash": alias_manifest["token_manifest_sha256"],
            "alias_manifest_json": json.dumps(
                alias_manifest, sort_keys=True, ensure_ascii=False
            ),
            "prefix_disjoint_aliases_json": json.dumps(
                alias_manifest["prefix_disjoint_aliases"], ensure_ascii=False
            ),
            "score_by_alias_json": json.dumps(
                lp_by_alias, sort_keys=True, ensure_ascii=False
            ),
            "score_aggregate": aggregate_alias_lps(
                lp_by_alias, alias_manifest["prefix_disjoint_aliases"]
            ),
            "scoring_spec_sha256": manifest.scoring_spec_sha256,
        }
        if counterfactual_manifest is not None:
            row.update(
                {
                    "counterfactual_alias_set_hash": counterfactual_manifest[
                        "token_manifest_sha256"
                    ],
                    "counterfactual_alias_manifest_json": json.dumps(
                        counterfactual_manifest, sort_keys=True, ensure_ascii=False
                    ),
                    "counterfactual_score_by_alias_json": json.dumps(
                        counterfactual_lp_by_alias,
                        sort_keys=True,
                        ensure_ascii=False,
                    ),
                    "counterfactual_score_aggregate": aggregate_alias_lps(
                        counterfactual_lp_by_alias,
                        counterfactual_manifest["prefix_disjoint_aliases"],
                    ),
                }
            )
        state["rows"].append(row)
        state["done"][item_id] = True
        newly_done += 1
        if newly_done % checkpoint_every == 0:
            state_store.write(state)
            elapsed = time.time() - started
            rate = elapsed / newly_done
            heartbeat(
                output_dir,
                phase="capability",
                checkpoint=checkpoint_key,
                completed=len(state["done"]),
                total=len(items),
                seconds_per_item=rate,
            )
            print(
                f"G5 {len(state['done'])}/{len(items)}; "
                f"{rate:.2f}s/item; ETA={(len(items) - len(state['done'])) * rate / 60:.1f}m",
                flush=True,
            )
    state_store.write(state)
    frame = pd.DataFrame(state["rows"]).sort_values("item_id").reset_index(drop=True)
    if len(frame) != expected or frame.item_id.nunique() != expected:
        raise RuntimeError("G5 completion invariant failed")
    numeric = frame[["score_aggregate"]].to_numpy(dtype=float)
    if not bool(pd.notna(frame.score_aggregate).all()) or not bool(
        torch.isfinite(torch.as_tensor(numeric)).all()
    ):
        raise RuntimeError("G5 contains non-finite scores")
    parquet_path = output_dir / "g5_capability.parquet"
    frame.to_parquet(parquet_path, index=False, compression="zstd")
    summary = capability_summary(frame)
    summary.update(
        {
            "schema_version": 1,
            "capability_definition": config["g5_capability"]["capability_definition"],
            "exact_normalized_equality": True,
            "max_new_tokens": int(config["g5_capability"]["max_new_tokens"]),
            "gpu": dict(gpu),
        }
    )
    result_path = output_dir / "g5_capability.json"
    write_result(
        summary,
        result_path,
        Provenance(
            evidence_id=checkpoint["evidence_id"],
            tier=config["tier"],
            command=(
                "python -m jspace_olmo_lineage.experiments.stage_wedge "
                f"--config interpretability/jspace_olmo_lineage/configs/ol2_stage_wedge.yaml "
                f"--checkpoint {checkpoint_key} --phase all"
            ),
            inputs={"input_manifest": manifest.sha256()},
            input_manifest_sha256=manifest.sha256(),
            model=model_reference(checkpoint),
            seed_contract=SEED_CONTRACT,
        ),
    )
    cohort = freeze_cohort(config, checkpoint_key, frame, manifest, output_dir)
    return frame, summary, cohort


def capability_summary(frame: pd.DataFrame) -> dict:
    direct_composed = frame[frame.variant.isin(["direct", "composed"])]
    paired = direct_composed.groupby(
        ["bank", "fact_id"], sort=True
    ).capable_generation.agg(["size", "all"])
    paired = paired[(paired["size"] == 2) & paired["all"]]
    return {
        "n_items": len(frame),
        "n_facts": int(frame.fact_id.nunique()),
        "n_families": int(frame.canonical_family.nunique()),
        "capable_rate": float(frame.capable_generation.mean()),
        "capable_by_bank": {
            str(key): float(value)
            for key, value in frame.groupby("bank").capable_generation.mean().items()
        },
        "capable_by_bank_variant": {
            f"{bank}:{variant}": float(value)
            for (bank, variant), value in frame.groupby(["bank", "variant"])
            .capable_generation.mean()
            .items()
        },
        "fully_capable_direct_composed_facts_by_bank": {
            str(bank): int((paired.index.get_level_values("bank") == bank).sum())
            for bank in sorted(frame.bank.unique())
        },
    }


def freeze_cohort(
    config: Mapping,
    checkpoint_key: str,
    frame: pd.DataFrame,
    manifest: InputManifest,
    output_dir: Path,
) -> dict:
    eligible = []
    bank_s = frame[(frame.bank == "S") & frame.variant.isin(["direct", "composed"])]
    for fact_id, rows in bank_s.groupby("fact_id", sort=True):
        if (
            len(rows) == 2
            and set(rows.variant) == {"direct", "composed"}
            and bool(rows.capable_generation.all())
        ):
            eligible.append(
                {
                    "fact_id": str(fact_id),
                    "canonical_family": str(rows.canonical_family.iloc[0]),
                    "item_ids": sorted(str(value) for value in rows.item_id),
                }
            )
    fact_floor = int(config["g5_capability"]["bank_s_direct_composed_fact_floor"])
    family_floor = int(config["g5_capability"]["bank_s_capable_family_floor"])
    families = sorted({row["canonical_family"] for row in eligible})
    passed = len(eligible) >= fact_floor and len(families) >= family_floor
    payload = {
        "schema_version": 1,
        "checkpoint_key": checkpoint_key,
        "selection": (
            "same-checkpoint Bank-S facts with exact-generation capability "
            "on both frozen direct and composed variants"
        ),
        "selection_opened_before_interventions": True,
        "input_manifest_sha256": manifest.sha256(),
        "g5_parquet_sha256": file_sha256(output_dir / "g5_capability.parquet"),
        "fact_floor": fact_floor,
        "family_floor": family_floor,
        "n_facts": len(eligible),
        "n_items": 2 * len(eligible),
        "n_families": len(families),
        "fact_rows": eligible,
        "item_ids": sorted(item_id for row in eligible for item_id in row["item_ids"]),
        "families": families,
        "capability_gate_passed": passed,
        "route": "tier1" if passed else "capability_gated",
    }
    envelope = {
        "schema_version": 1,
        "payload": payload,
        "payload_sha256": object_sha256(payload),
    }
    path = output_dir / "cohort_manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != envelope:
            raise RuntimeError("frozen capable cohort changed across replay")
    else:
        atomic_json(path, envelope)
    return envelope


def capable_items(items: Sequence[Mapping], cohort: Mapping) -> list[dict]:
    wanted = set(cohort["payload"]["item_ids"])
    selected = [dict(item) for item in items if str(item["item_id"]) in wanted]
    selected.sort(key=lambda item: str(item["item_id"]))
    if len(selected) != len(wanted):
        raise RuntimeError("frozen cohort cannot be reconstructed from banks")
    return selected


def _materialize_lens(uri: str, expected_sha256: str) -> Path:
    source = resolve_uri(uri)
    if file_sha256(source) != expected_sha256:
        raise RuntimeError(f"lens source hash mismatch: {uri}")
    destination = local_work() / "lenses" / f"{expected_sha256}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(f".tmp{os.getpid()}")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    if file_sha256(destination) != expected_sha256:
        raise RuntimeError(f"staged lens hash mismatch: {destination}")
    return destination


def _require_hook_fired(log, condition: str) -> None:
    fires = getattr(log, "hook_fires", {})
    if int(fires.get("prefill", 0)) <= 0:
        raise RuntimeError(f"{condition} intervention hook never fired")
    forbidden = sum(int(value) for phase, value in fires.items() if phase != "prefill")
    if forbidden:
        raise RuntimeError(f"{condition} hook fired outside prefill: {fires}")


def _position_details(log) -> list[dict]:
    return [asdict(row) for row in log.positions]


def _overlap_details(log) -> list[dict]:
    return [asdict(row) for row in log.overlap]


def _matched_details(log) -> list[dict]:
    return [asdict(row) for row in log.matched]


def _audit_matched(log, condition: str) -> dict:
    if not log.matched:
        raise RuntimeError(f"{condition} has no matched-control records")
    rank_failures = 0
    energy_failures = 0
    max_relative = 0.0
    max_absolute_below = 0.0
    max_protected_cos = 0.0
    clamped = 0
    for row in log.matched:
        rank_failures += int(row.achieved_rank != row.target_rank)
        clamped += int(row.clamped)
        difference = abs(row.achieved_energy_frac - row.target_energy_frac)
        if not row.clamped and row.target_energy_frac >= ENERGY_RELATIVE_FLOOR:
            relative = difference / max(row.target_energy_frac, 1e-30)
            max_relative = max(max_relative, relative)
            energy_failures += int(relative > ENERGY_RELATIVE_TOLERANCE)
        elif not row.clamped:
            max_absolute_below = max(max_absolute_below, difference)
            energy_failures += int(difference > ENERGY_ABSOLUTE_TOLERANCE_BELOW_FLOOR)
        max_protected_cos = max(max_protected_cos, row.max_protected_cos)
    protection_failures = int(
        condition == "instant_rank_energy_matched"
        and max_protected_cos > PROTECTED_COSINE_TOLERANCE
    )
    report = {
        "n_positions": len(log.matched),
        "rank_failures": rank_failures,
        "energy_failures": energy_failures,
        "protection_failures": protection_failures,
        "clamped_positions": clamped,
        "max_energy_relative_error": max_relative,
        "max_energy_absolute_error_below_floor": max_absolute_below,
        "max_protected_cosine": max_protected_cos,
        "energy_relative_tolerance": ENERGY_RELATIVE_TOLERANCE,
        "energy_absolute_tolerance_below_floor": ENERGY_ABSOLUTE_TOLERANCE_BELOW_FLOOR,
        "protected_cosine_tolerance": PROTECTED_COSINE_TOLERANCE,
        "passed": not (rank_failures or energy_failures or protection_failures),
    }
    if not report["passed"]:
        raise RuntimeError(f"matched-control conformance failed: {report}")
    return report


def _state_header(
    evidence_id: str,
    manifest: InputManifest,
    cohort_sha256: str,
    frame: str,
) -> dict:
    return {
        "schema_version": 1,
        "evidence_id": evidence_id,
        "input_manifest_sha256": manifest.sha256(),
        "config_sha256": manifest.config_sha256,
        "model_revision": manifest.model_revision,
        "bank_sha256": manifest.bank_sha256,
        "cohort_sha256": cohort_sha256,
        "frame": frame,
    }


def _load_tier1_state(path: Path, header: Mapping) -> dict:
    if not path.exists():
        return {"done": {}, "baseline_checked": [], "baseline_stop_events": []}
    envelope = json.loads(path.read_text())
    if envelope.get("header") != dict(header):
        raise RuntimeError("refusing incompatible Tier-1 checkpoint state")
    payload = envelope["payload"]
    if object_sha256(payload) != envelope["payload_sha256"]:
        raise RuntimeError("Tier-1 checkpoint payload hash mismatch")
    return payload


def _write_tier1_state(path: Path, header: Mapping, payload: Mapping) -> None:
    atomic_json(
        path,
        {
            "schema_version": 1,
            "header": dict(header),
            "payload": dict(payload),
            "payload_sha256": object_sha256(dict(payload)),
        },
    )


def _aggregate_tier1_audit(rows: Sequence[Mapping]) -> dict:
    baseline = [float(row["baseline_replay_abs_error"]) for row in rows]
    cache = [float(row["cache_replay_abs_drift"]) for row in rows]
    rank_failures = energy_failures = protection_failures = 0
    max_rel = max_abs = max_cos = 0.0
    span_overlap = 0.0
    for row in rows:
        conformance = json.loads(row["matched_conformance_json"])
        for aliases in conformance.values():
            for report in aliases.values():
                rank_failures += int(report["rank_failures"])
                energy_failures += int(report["energy_failures"])
                protection_failures += int(report["protection_failures"])
                max_rel = max(max_rel, float(report["max_energy_relative_error"]))
                max_abs = max(
                    max_abs, float(report["max_energy_absolute_error_below_floor"])
                )
                max_cos = max(max_cos, float(report["max_protected_cosine"]))
        overlaps = json.loads(row["overlap_summary_json"])
        for aliases in overlaps.values():
            summary = aliases["meanJ_span_safe"]
            span_overlap = max(span_overlap, float(summary["projector_overlap_max"]))
    passed = not (rank_failures or energy_failures or protection_failures)
    return {
        "n_rows": len(rows),
        "baseline_replay_abs_error_max": max(baseline, default=0.0),
        "cache_replay_abs_drift_max": max(cache, default=0.0),
        "span_safe_projector_overlap_max": span_overlap,
        "rank_failures": rank_failures,
        "energy_failures": energy_failures,
        "protection_failures": protection_failures,
        "matched_energy_relative_error_max": max_rel,
        "matched_energy_absolute_error_below_floor_max": max_abs,
        "matched_protected_cosine_max": max_cos,
        "all_hard_gates_passed": passed,
    }


@torch.no_grad()
def run_tier1_frame(
    model,
    tokenizer,
    config: Mapping,
    checkpoint_key: str,
    manifest: InputManifest,
    cohort: Mapping,
    items: Sequence[Mapping],
    g5: pd.DataFrame,
    frame_name: str,
    lens_key: str,
    shared_dictionaries: Mapping,
    output_dir: Path,
    gpu: Mapping,
) -> tuple[pd.DataFrame, dict]:
    import jlens
    from jlens import JacobianLens

    checkpoint = config["checkpoints"][checkpoint_key]
    tier1 = config["tier1"]
    lens_spec = config["inputs"][lens_key]
    if lens_spec["frame"] != frame_name:
        raise RuntimeError("frame/lens contract mismatch")
    lens_path = _materialize_lens(lens_spec["uri"], lens_spec["sha256"])
    lens = JacobianLens.load(str(lens_path))
    band = [int(value) for value in tier1["band_zero_indexed"]]
    wrapped = jlens.from_hf(model, tokenizer)
    j_dictionaries = build_j_dictionaries(model, lens, band)
    ablator = Phase3JAblator(wrapped.layers, band)
    session = ScoringSession(tokenizer, DEFAULT_SPEC, device="cuda")
    g5_by_item = g5.set_index("item_id", verify_integrity=True)
    frame_slug = frame_name.replace("-", "_")
    frame_dir = output_dir / f"rows_{frame_slug}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / f"tier1_{frame_slug}_state.json"
    header = _state_header(
        checkpoint["evidence_id"],
        manifest,
        cohort["payload_sha256"],
        frame_name,
    )
    state = _load_tier1_state(state_path, header)
    started = time.time()
    newly_done = 0
    baseline_stop_n = int(tier1["baseline_replay_stop_n"])
    baseline_tolerance = float(tier1["baseline_replay_absolute_tolerance"])
    k = int(tier1["k"])
    protect_top_k = int(tier1["protect_top_k"])
    namespace = str(tier1["scientific_seed_namespace"])
    base_seed = int(tier1["base_seed"])

    def j_arm(full_ids, protect_sets, *, span_safe: bool):
        ablator.log = type(ablator.log)()
        ablator.phase, ablator.forward_index = "prefill", 0
        ablator.mode = {
            "dicts": j_dictionaries,
            "k": k,
            "nonneg": True,
            "protect_sets": protect_sets,
            "active_phases": {"prefill"},
            "span_safe": span_safe,
            "record_overlap": True,
            "record_ids": True,
            "answer_id": None,
        }
        try:
            with ablator:
                logits = model(input_ids=full_ids, use_cache=False).logits[0].float()
            condition = "meanJ_span_safe" if span_safe else "meanJ_label_protected"
            _require_hook_fired(ablator.log, condition)
            return logits, ablator.log
        finally:
            ablator.mode = None

    def dictionary_arm(full_ids, protect_sets, dictionaries, condition):
        ablator.log = type(ablator.log)()
        ablator.phase, ablator.forward_index = "prefill", 0
        ablator.mode = {
            "dicts": dictionaries,
            "k": k,
            "nonneg": True,
            "protect_sets": protect_sets,
            "active_phases": {"prefill"},
            "span_safe": False,
            "record_overlap": False,
            "record_ids": True,
            "answer_id": None,
        }
        try:
            with ablator:
                logits = model(input_ids=full_ids, use_cache=False).logits[0].float()
            _require_hook_fired(ablator.log, condition)
            return logits, ablator.log
        finally:
            ablator.mode = None

    for item in items:
        item_id = str(item["item_id"])
        if item_id in state["done"]:
            row_path = frame_dir / f"{item_id}.json"
            if (
                not row_path.is_file()
                or file_sha256(row_path) != state["done"][item_id]
            ):
                raise RuntimeError(f"completed Tier-1 row drifted: {item_id}")
            continue
        g5_row = g5_by_item.loc[item_id]
        canonical_alias = canonical_alias_for(
            session, item["accepted_answers"], item["canonical_answer"]
        )
        alias_manifest = session.freeze_alias_manifest(
            item["accepted_answers"], canonical_alias=canonical_alias
        )
        if alias_manifest["token_manifest_sha256"] != g5_row.alias_set_hash:
            raise RuntimeError(f"alias manifest mismatch against G5 for {item_id}")
        selected_aliases = alias_manifest["prefix_disjoint_aliases"]
        if selected_aliases != json.loads(g5_row.prefix_disjoint_aliases_json):
            raise RuntimeError(f"prefix-disjoint alias mismatch for {item_id}")

        alias_data = {}
        baseline_by_alias = {}
        cache_by_alias = {}
        rank_metadata = None
        for alias in selected_aliases:
            full_ids, prompt_length = session.full_ids(item["prompt"], alias)
            clean_logits = model(input_ids=full_ids, use_cache=False).logits[0].float()
            protect_sets = clean_logits.topk(protect_top_k, dim=-1).indices
            baseline_by_alias[alias] = session.answer_sequence_lp(
                full_ids, clean_logits, prompt_length
            )
            cached_logits = model(input_ids=full_ids, use_cache=True).logits[0].float()
            cache_by_alias[alias] = session.answer_sequence_lp(
                full_ids, cached_logits, prompt_length
            )
            if alias == canonical_alias:
                first_token_ids = {
                    value: int(session.answer_ids(value)[0, 0])
                    for value in item["accepted_answers"]
                }
                rank_metadata = session.clean_first_token_ranks(
                    clean_logits[prompt_length - 1], first_token_ids
                )
            alias_data[alias] = {
                "full_ids": full_ids,
                "prompt_length": prompt_length,
                "protect_sets": protect_sets,
            }
            del clean_logits, cached_logits
        if rank_metadata is None:
            raise RuntimeError(f"canonical alias was not scored for {item_id}")
        baseline = aggregate_alias_lps(baseline_by_alias, selected_aliases)
        cached_baseline = aggregate_alias_lps(cache_by_alias, selected_aliases)
        baseline_error = abs(baseline - float(g5_row.score_aggregate))
        cache_drift = abs(cached_baseline - baseline)
        if len(state["baseline_checked"]) < baseline_stop_n:
            if baseline_error > baseline_tolerance:
                event = {
                    "item_id": item_id,
                    "frame": frame_name,
                    "measured": baseline,
                    "g5": float(g5_row.score_aggregate),
                    "absolute_difference": baseline_error,
                    "tolerance": baseline_tolerance,
                }
                state["baseline_stop_events"].append(event)
                _write_tier1_state(state_path, header, state)
                raise RuntimeError(
                    f"BASELINE STOP RULE before intervention outcome: {event}"
                )
            state["baseline_checked"].append(item_id)

        order = list(RAW_CONDITIONS)
        permutation = stable_rng(
            experiment_id=namespace,
            item_id=item_id,
            condition="condition-order",
            base_seed=base_seed,
        ).permutation(len(order))
        order = [order[int(index)] for index in permutation]
        scores_by_condition = {condition: {} for condition in ALL_CONDITIONS}
        scores_by_condition["baseline"] = baseline_by_alias
        overlap_by_alias = {}
        matched_by_alias = {}
        matched_conformance = {}
        intervention_detail = {}
        protect_detail = {}
        for alias in selected_aliases:
            data = alias_data[alias]
            profiles = {}
            protect_detail[alias] = data["protect_sets"].detach().cpu().tolist()
            intervention_detail[alias] = {}
            for condition in order:
                if condition == "meanJ_span_safe":
                    logits, intervention_log = j_arm(
                        data["full_ids"], data["protect_sets"], span_safe=True
                    )
                    profiles["span_safe"] = profile_from_p3log(
                        intervention_log, overlap_records=intervention_log.overlap
                    )
                    overlap_by_alias.setdefault(alias, {})[condition] = (
                        intervention_log.overlap_summary()
                    )
                elif condition == "meanJ_label_protected":
                    logits, intervention_log = j_arm(
                        data["full_ids"], data["protect_sets"], span_safe=False
                    )
                    profiles["label"] = profile_from_p3log(
                        intervention_log, overlap_records=intervention_log.overlap
                    )
                    overlap_by_alias.setdefault(alias, {})[condition] = (
                        intervention_log.overlap_summary()
                    )
                elif condition == "mechanics_random":
                    logits, intervention_log = dictionary_arm(
                        data["full_ids"],
                        data["protect_sets"],
                        shared_dictionaries["random"],
                        condition,
                    )
                elif condition == "logit_label_protected":
                    logits, intervention_log = dictionary_arm(
                        data["full_ids"],
                        data["protect_sets"],
                        shared_dictionaries["logit"],
                        condition,
                    )
                else:
                    raise RuntimeError(f"unhandled condition {condition!r}")
                scores_by_condition[condition][alias] = session.answer_sequence_lp(
                    data["full_ids"], logits, data["prompt_length"]
                )
                intervention_detail[alias][condition] = {
                    "positions": _position_details(intervention_log),
                    "overlap": _overlap_details(intervention_log),
                    "hook_fires": dict(intervention_log.hook_fires),
                }
                del logits

            for condition, variant, profile_name in MATCHED_CONDITIONS:
                alias_seed_id = f"{item_id}\u0000{alias}"

                def seed_factory(
                    layer,
                    forward_index,
                    position,
                    *,
                    _condition=condition,
                    _alias_seed_id=alias_seed_id,
                ):
                    if forward_index != 0:
                        raise RuntimeError("matched arm changed forward index")
                    return stable_seed(
                        experiment_id=namespace,
                        item_id=_alias_seed_id,
                        condition=_condition,
                        layer=int(layer),
                        position=int(position),
                        base_seed=base_seed,
                    )

                logits, matched_log = teacher_forced_matched_arm(
                    model,
                    wrapped.layers,
                    band,
                    j_dictionaries,
                    data["full_ids"],
                    profiles[profile_name],
                    variant=variant,
                    protect_sets=data["protect_sets"],
                    seed_base=0,
                    return_cpu=False,
                    seed_factory=seed_factory,
                )
                _require_hook_fired(matched_log, condition)
                scores_by_condition[condition][alias] = session.answer_sequence_lp(
                    data["full_ids"], logits, data["prompt_length"]
                )
                matched_by_alias.setdefault(alias, {})[condition] = (
                    matched_log.matched_summary()
                )
                matched_conformance.setdefault(alias, {})[condition] = _audit_matched(
                    matched_log, condition
                )
                intervention_detail[alias][condition] = {
                    "matched": _matched_details(matched_log),
                    "hook_fires": dict(matched_log.hook_fires),
                }
                del logits

        row = {
            "study_id": STUDY_ID,
            "phase": "development",
            "tier": config["tier"],
            "evidence_id": checkpoint["evidence_id"],
            "checkpoint_key": checkpoint_key,
            "checkpoint_stage": checkpoint["stage"],
            "model_id": checkpoint["model_id"],
            "model_revision": checkpoint["revision"],
            "frame": frame_name,
            "config_sha256": manifest.config_sha256,
            "bank_sha256": manifest.bank_sha256,
            "cohort_sha256": cohort["payload_sha256"],
            "scoring_spec_sha256": manifest.scoring_spec_sha256,
            "lens_sha256": lens_spec["sha256"],
            "scientific_seed_namespace": namespace,
            "item_id": item_id,
            "fact_id": str(item["fact_id"]),
            "canonical_family": str(item["canonical_family"]),
            "relation_group": str(item["relation_group"]),
            "variant": str(item["variant"]),
            "bank": str(item["bank"]),
            "alias_set_hash": alias_manifest["token_manifest_sha256"],
            "alias_manifest_json": json.dumps(
                alias_manifest, sort_keys=True, ensure_ascii=False
            ),
            "prefix_disjoint_aliases_json": json.dumps(
                selected_aliases, ensure_ascii=False
            ),
            "condition_order_json": json.dumps(order),
            "execution_order_json": json.dumps(
                ["baseline", *order, *[row[0] for row in MATCHED_CONDITIONS]]
            ),
            "clean_rank_metadata_json": json.dumps(rank_metadata, sort_keys=True),
            "protected_direction_ids_json": json.dumps(protect_detail),
            "intervention_detail_json": json.dumps(intervention_detail, sort_keys=True),
            "overlap_summary_json": json.dumps(overlap_by_alias, sort_keys=True),
            "matched_summary_json": json.dumps(matched_by_alias, sort_keys=True),
            "matched_conformance_json": json.dumps(matched_conformance, sort_keys=True),
            "baseline_replay_abs_error": baseline_error,
            "cache_replay_abs_drift": cache_drift,
        }
        for condition in ALL_CONDITIONS:
            row[f"lp_{condition}"] = aggregate_alias_lps(
                scores_by_condition[condition], selected_aliases
            )
            row[f"lp_by_alias_{condition}_json"] = json.dumps(
                scores_by_condition[condition], sort_keys=True, ensure_ascii=False
            )
        row_path = frame_dir / f"{item_id}.json"
        atomic_json(row_path, row)
        state["done"][item_id] = file_sha256(row_path)
        newly_done += 1
        _write_tier1_state(state_path, header, state)
        if newly_done % int(tier1["checkpoint_every_items"]) == 0:
            elapsed = time.time() - started
            rate = elapsed / newly_done
            heartbeat(
                output_dir,
                phase="tier1",
                checkpoint=checkpoint_key,
                frame=frame_name,
                completed=len(state["done"]),
                total=len(items),
                seconds_per_item=rate,
            )
            print(
                f"Tier1 {checkpoint_key} {frame_name} "
                f"{len(state['done'])}/{len(items)}; {rate:.2f}s/item; "
                f"ETA={(len(items) - len(state['done'])) * rate / 60:.1f}m",
                flush=True,
            )

    rows = []
    for item in items:
        item_id = str(item["item_id"])
        path = frame_dir / f"{item_id}.json"
        if file_sha256(path) != state["done"][item_id]:
            raise RuntimeError(f"Tier-1 row changed at finalization: {item_id}")
        rows.append(json.loads(path.read_text()))
    audit = _aggregate_tier1_audit(rows)
    if not audit["all_hard_gates_passed"]:
        raise RuntimeError(f"final Tier-1 conformance failed: {audit}")
    frame = pd.DataFrame(rows).sort_values("item_id").reset_index(drop=True)
    parquet_path = output_dir / f"tier1_{frame_slug}.parquet"
    frame.to_parquet(parquet_path, index=False, compression="zstd")
    result = {
        "schema_version": 1,
        "checkpoint_key": checkpoint_key,
        "frame": frame_name,
        "lens_sha256": lens_spec["sha256"],
        "n_items": len(frame),
        "n_facts": int(frame.fact_id.nunique()),
        "n_families": int(frame.canonical_family.nunique()),
        "conditions": list(ALL_CONDITIONS),
        "band_zero_indexed": band,
        "k": k,
        "protect_top_k": protect_top_k,
        "conformance": audit,
        "raw_rows_only": True,
        "gpu": dict(gpu),
    }
    atomic_json(output_dir / f"tier1_{frame_slug}.json", result)
    ablator.mode = None
    j_dictionaries.clear()
    gc.collect()
    torch.cuda.empty_cache()
    return frame, result


def _build_shared_dictionaries(model, config: Mapping) -> dict:
    tier1 = config["tier1"]
    band = [int(value) for value in tier1["band_zero_indexed"]]
    logit = build_logit_dictionary(model, band)
    vocabulary, width = logit[band[0]].shape
    generator = torch.Generator().manual_seed(
        stable_seed(
            experiment_id=str(tier1["scientific_seed_namespace"]),
            item_id="global-random-dictionary",
            condition="mechanics_random",
            base_seed=int(tier1["base_seed"]),
        )
    )
    random_dictionary = torch.nn.functional.normalize(
        torch.randn(vocabulary, width, generator=generator, dtype=torch.float32),
        dim=1,
    ).to("cuda", torch.float16)
    return {
        "logit": logit,
        "random": {layer: random_dictionary for layer in band},
    }


def _load_model(snapshot: Path):
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot)
    model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            snapshot,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        .to("cuda")
        .eval()
    )
    assert_model_on_cuda(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def _load_existing_capability(output_dir: Path) -> tuple[pd.DataFrame, dict, dict]:
    paths = (
        output_dir / "g5_capability.parquet",
        output_dir / "g5_capability.json",
        output_dir / "cohort_manifest.json",
    )
    if not all(path.is_file() for path in paths):
        raise RuntimeError("capability phase is incomplete")
    frame = pd.read_parquet(paths[0])
    result = json.loads(paths[1].read_text())["payload"]
    cohort = json.loads(paths[2].read_text())
    if object_sha256(cohort["payload"]) != cohort["payload_sha256"]:
        raise RuntimeError("cohort manifest payload hash mismatch")
    return frame, result, cohort


def finalize_and_register(
    config: Mapping,
    config_path: Path,
    checkpoint_key: str,
    manifest: InputManifest,
    output_dir: Path,
    capability: Mapping,
    cohort: Mapping,
    frame_results: Mapping[str, Mapping],
) -> dict:
    checkpoint = config["checkpoints"][checkpoint_key]
    passed = bool(cohort["payload"]["capability_gate_passed"])
    if passed and set(frame_results) != {name for name, _ in FRAME_SPECS}:
        raise RuntimeError("both frozen lens frames are required before registration")
    payload = {
        "schema_version": 1,
        "checkpoint_key": checkpoint_key,
        "checkpoint_stage": checkpoint["stage"],
        "model": model_reference(checkpoint),
        "status": "complete" if passed else "capability_gated",
        "capability": dict(capability),
        "cohort": cohort["payload"],
        "frames": dict(frame_results),
        "tier": config["tier"],
        "natural_experiment_qualification": config["ancestry"]["qualification"],
        "predictions_frozen_before_model_load": True,
        "stage_router_not_run": True,
        "claim_boundary": (
            "Official ancestry-qualified stage cell at development tier; "
            "not randomized objective attribution."
        ),
    }
    result_path = output_dir / "stage_result.json"
    command = (
        "python -m jspace_olmo_lineage.experiments.stage_wedge "
        f"--config {config_path} --checkpoint {checkpoint_key} --phase all"
    )
    inputs = {
        "input_manifest": file_sha256(output_dir / "input_manifest.json"),
        "snapshot_conformance": file_sha256(output_dir / "snapshot_conformance.json"),
        "cohort_manifest": file_sha256(output_dir / "cohort_manifest.json"),
    }
    write_result(
        payload,
        result_path,
        Provenance(
            evidence_id=checkpoint["evidence_id"],
            tier=config["tier"],
            command=command,
            inputs=inputs,
            input_manifest_sha256=manifest.sha256(),
            model=model_reference(checkpoint),
            seed_contract=SEED_CONTRACT,
        ),
    )
    outputs = [
        result_path,
        output_dir / "snapshot_conformance.json",
        output_dir / "input_manifest.json",
        output_dir / "g5_capability.json",
        output_dir / "g5_capability.parquet",
        output_dir / "cohort_manifest.json",
    ]
    if passed:
        for frame_name, _ in FRAME_SPECS:
            slug = frame_name.replace("-", "_")
            outputs.extend(
                [
                    output_dir / f"tier1_{slug}.json",
                    output_dir / f"tier1_{slug}.parquet",
                ]
            )
    try:
        existing = resolve(checkpoint["evidence_id"])
    except RegistryError:
        existing = None
    if existing is not None:
        expected = {str(path): file_sha256(path) for path in outputs}
        observed = {row["path"]: row["sha256"] for row in existing.get("outputs", [])}
        if expected != observed:
            raise RuntimeError("registered stage outputs do not match replay")
        return {"result": str(result_path), "event": existing, "replayed": True}
    event = create(
        checkpoint["evidence_id"],
        tier=config["tier"],
        what=(
            f"OLMo Study-2 {checkpoint['stage']} stage wedge: "
            + (
                f"{cohort['payload']['n_items']} Bank-S Tier-1 rows in each "
                "of two frozen lens frames."
                if passed
                else "capability gate failed before intervention outcomes."
            )
        ),
        command=command,
        outputs=outputs,
        inputs=inputs,
        natural_experiment=True,
        capability_gated=not passed,
    )
    return {"result": str(result_path), "event": event, "replayed": False}


def _frame_results_from_disk(output_dir: Path) -> dict:
    result = {}
    for frame_name, _ in FRAME_SPECS:
        path = output_dir / f"tier1_{frame_name.replace('-', '_')}.json"
        if path.is_file():
            result[frame_name] = json.loads(path.read_text())
    return result


def registered_replay(evidence_id: str) -> dict | None:
    """Return an already registered cell without mutating immutable outputs."""
    try:
        event = resolve(evidence_id)
    except RegistryError:
        return None
    failures = []
    for row in event.get("outputs", []):
        path = Path(row["path"])
        observed = file_sha256(path) if path.is_file() else None
        if observed != row["sha256"]:
            failures.append(
                {
                    "path": str(path),
                    "expected": row["sha256"],
                    "observed": observed,
                }
            )
    if failures:
        raise RuntimeError(
            "registered stage output verification failed: "
            + json.dumps(failures, sort_keys=True)
        )
    return {
        "evidence_id": evidence_id,
        "already_registered": True,
        "live": event["live"],
        "n_outputs_verified": len(event.get("outputs", [])),
        "code_commit": event.get("code_commit"),
    }


def configure_run_root(config: Mapping) -> Path:
    configured = Path(config["run_root"]).resolve()
    existing = os.environ.get("JSPACE_OLMO_RUN_ROOT")
    if existing is None:
        os.environ["JSPACE_OLMO_RUN_ROOT"] = str(configured)
    elif Path(existing).resolve() != configured:
        raise RuntimeError(
            "JSPACE_OLMO_RUN_ROOT disagrees with the frozen Study-2 config: "
            f"{existing} != {configured}"
        )
    observed = run_root(create=False).resolve()
    if observed != configured:
        raise RuntimeError(f"resolved OLMo run root drifted: {observed}")
    return observed


def main() -> None:
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = load_config(config_path)
    configure_run_root(config)
    clean = require_clean_tree(expected_branch=config["branch"])
    checkpoint = config["checkpoints"][arguments.checkpoint]
    replay = registered_replay(checkpoint["evidence_id"])
    if replay is not None:
        print(json.dumps(replay, indent=1))
        return
    snapshot = resolve_model(
        f"{checkpoint['model_id']}@{checkpoint['revision']}", must_exist=True
    )
    output_dir = (
        metrics_dir("stage-wedge") / checkpoint["slug"] / checkpoint["evidence_id"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = snapshot_preflight(config, arguments.checkpoint, snapshot, output_dir)
    bank_paths = [
        resolve_uri(config["inputs"]["bank_f"]["uri"]),
        resolve_uri(config["inputs"]["bank_s"]["uri"]),
    ]
    for name, path in zip(("bank_f", "bank_s"), bank_paths):
        if file_sha256(path) != config["inputs"][name]["sha256"]:
            raise RuntimeError(f"frozen {name} hash mismatch")
    manifest = _input_manifest(
        config,
        config_path,
        arguments.checkpoint,
        preflight,
        bank_paths,
        clean,
    )
    manifest_path = output_dir / "input_manifest.json"
    envelope = manifest.envelope()
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != envelope:
        raise RuntimeError("input manifest changed across replay")
    atomic_json(manifest_path, envelope)
    if arguments.phase == "preflight":
        print(
            json.dumps(
                {
                    "preflight": preflight,
                    "input_manifest_sha256": manifest.sha256(),
                },
                indent=1,
            )
        )
        return

    items = load_items(bank_paths)
    if arguments.phase == "register":
        _, capability, cohort = _load_existing_capability(output_dir)
        result = finalize_and_register(
            config,
            config_path,
            arguments.checkpoint,
            manifest,
            output_dir,
            capability,
            cohort,
            _frame_results_from_disk(output_dir),
        )
        print(json.dumps(result, indent=1))
        return

    gpu = require_cuda_gpu()
    heartbeat(
        output_dir,
        phase="model_load",
        checkpoint=arguments.checkpoint,
        completed=0,
    )
    model, tokenizer = _load_model(snapshot)
    try:
        if arguments.phase in {"capability", "all"}:
            g5, capability, cohort = run_capability(
                model,
                tokenizer,
                config,
                arguments.checkpoint,
                manifest,
                items,
                output_dir,
                gpu,
            )
        else:
            g5, capability, cohort = _load_existing_capability(output_dir)
        if arguments.phase == "capability":
            print(
                json.dumps(
                    {
                        "capability": capability,
                        "cohort": cohort["payload"],
                    },
                    indent=1,
                )
            )
            return

        frame_results = _frame_results_from_disk(output_dir)
        if cohort["payload"]["capability_gate_passed"]:
            selected = capable_items(items, cohort)
            shared = _build_shared_dictionaries(model, config)
            for frame_name, lens_key in FRAME_SPECS:
                if frame_name in frame_results:
                    continue
                _, frame_result = run_tier1_frame(
                    model,
                    tokenizer,
                    config,
                    arguments.checkpoint,
                    manifest,
                    cohort,
                    selected,
                    g5,
                    frame_name,
                    lens_key,
                    shared,
                    output_dir,
                    gpu,
                )
                frame_results[frame_name] = frame_result
        if arguments.phase == "tier1":
            print(json.dumps(frame_results, indent=1))
            return
        result = finalize_and_register(
            config,
            config_path,
            arguments.checkpoint,
            manifest,
            output_dir,
            capability,
            cohort,
            frame_results,
        )
        heartbeat(
            output_dir,
            phase="registered",
            checkpoint=arguments.checkpoint,
            evidence_id=checkpoint["evidence_id"],
            completed=1,
            total=1,
        )
        print(json.dumps(result, indent=1))
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
