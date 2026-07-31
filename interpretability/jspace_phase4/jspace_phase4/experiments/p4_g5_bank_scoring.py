"""GPU-only capability and prospective alias scoring for development banks."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from jspace_phase3.bank import load_bank

from ..gpu import assert_model_on_cuda, require_cuda_gpu
from ..manifests import (
    InputManifest,
    atomic_json,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths4 import metrics_dir, resolve_uri
from ..provenance4 import Provenance4, write_result4
from ..registry4 import create
from ..scoring4 import (
    DEFAULT_SPEC,
    ScoringSession,
    aggregate_alias_lps,
)
from ..seeds import SEED_CONTRACT
from ..state import StateHeader, StateStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def model_reference(uri: str) -> dict:
    prefix = "model://"
    if not uri.startswith(prefix) or "@" not in uri:
        raise ValueError("model URI must pin a revision")
    model_id, revision = uri[len(prefix):].rsplit("@", 1)
    return {"model_id": model_id, "revision": revision}


def tokenizer_source_hash(model_path: Path) -> str:
    names = (
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.json", "merges.txt", "chat_template.jinja",
    )
    files = {
        name: file_sha256(model_path / name)
        for name in names if (model_path / name).exists()
    }
    if not files:
        raise RuntimeError("model snapshot has no tokenizer files")
    return object_sha256(files)


def load_items(bank_paths: list[Path]) -> list[dict]:
    items = []
    for path in bank_paths:
        for bundle in load_bank(path):
            items.extend(bundle.as_items())
    return sorted(items, key=lambda row: row["item_id"])


def canonical_alias_for(session: ScoringSession, aliases: list[str],
                        canonical_answer: str) -> str:
    exact_matches = [
        alias for alias in aliases
        if alias.strip() == canonical_answer.strip()
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise RuntimeError(
            f"canonical answer {canonical_answer!r} has "
            f"{len(exact_matches)} exact aliases")
    target = session.spec.normalize_generation(canonical_answer)
    matches = [
        alias for alias in aliases
        if session.spec.normalize_generation(alias) == target
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"canonical answer {canonical_answer!r} has {len(matches)} "
            "exact normalized aliases")
    return matches[0]


@torch.no_grad()
def main() -> None:
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    clean = require_clean_tree()
    gpu = require_cuda_gpu()
    model_path = resolve_uri(config["model_uri"])
    bank_paths = [resolve_uri(uri) for uri in config["banks"]]
    bank_hashes = {
        str(uri): file_sha256(path)
        for uri, path in zip(config["banks"], bank_paths)
    }
    model = model_reference(config["model_uri"])
    tokenizer_hash = tokenizer_source_hash(model_path)
    scoring_hash = object_sha256(DEFAULT_SPEC.as_dict())
    input_manifest = InputManifest(
        experiment_id=config["evidence_id"],
        config_sha256=file_sha256(config_path),
        model_id=model["model_id"],
        model_revision=model["revision"],
        tokenizer_manifest_sha256=tokenizer_hash,
        lens_sha256="not-applicable",
        bank_sha256=object_sha256(bank_hashes),
        partition_sha256="not-applicable-development-capability",
        scoring_spec_sha256=scoring_hash,
        upstream={},
        code_commit=clean["code_commit"],
    )

    output_dir = (
        metrics_dir(config["slug"]) / "g5_bank" / config["evidence_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "input_manifest.json"
    atomic_json(manifest_path, input_manifest.envelope())
    state_store = StateStore(
        output_dir / "state.json",
        StateHeader(
            evidence_id=config["evidence_id"],
            input_manifest_sha256=input_manifest.sha256(),
            config_sha256=input_manifest.config_sha256,
            model_revision=input_manifest.model_revision,
            bank_sha256=input_manifest.bank_sha256,
            partition_sha256=input_manifest.partition_sha256,
        ),
    )
    state = state_store.load() or {"done": {}, "rows": [], "gpu": gpu}

    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16).to("cuda").eval()
    assert_model_on_cuda(hf_model)
    session = ScoringSession(tokenizer, DEFAULT_SPEC, device="cuda")
    items = load_items(bank_paths)
    print(
        f"{config['slug']}: {len(items)} items; "
        f"GPU={gpu['name']}; bos_prefixed={session.bos_prefixed}",
        flush=True,
    )

    started = time.time()
    checkpoint_every = int(config.get("checkpoint_every", 10))
    newly_done = 0
    for item in items:
        item_id = item["item_id"]
        if item_id in state["done"]:
            continue
        prompt_ids = session.prompt_ids(item["prompt"])
        prompt_length = prompt_ids.shape[1]
        generated_ids = hf_model.generate(
            prompt_ids,
            max_new_tokens=int(config.get("max_new_tokens", 8)),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(
            generated_ids[0, prompt_length:],
            skip_special_tokens=True,
        )
        matched = session.grade_alias(
            generated, item["accepted_answers"])
        canonical_alias = canonical_alias_for(
            session, item["accepted_answers"],
            item["canonical_answer"])
        alias_manifest = session.freeze_alias_manifest(
            item["accepted_answers"],
            canonical_alias=canonical_alias,
        )
        lp_by_alias = {}
        for alias in alias_manifest["prefix_disjoint_aliases"]:
            full_ids, n_prompt = session.full_ids(item["prompt"], alias)
            logits = hf_model(
                input_ids=full_ids, use_cache=False).logits[0]
            lp_by_alias[alias] = session.answer_sequence_lp(
                full_ids, logits, n_prompt)
        counterfactual_manifest = None
        counterfactual_lp_by_alias = {}
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
            for alias in counterfactual_manifest[
                    "prefix_disjoint_aliases"]:
                full_ids, n_prompt = session.full_ids(
                    item["prompt"], alias)
                logits = hf_model(
                    input_ids=full_ids, use_cache=False).logits[0]
                counterfactual_lp_by_alias[alias] = (
                    session.answer_sequence_lp(
                        full_ids, logits, n_prompt))
        generation_grade = (
            session.grade_counterfactual_generation(
                generated,
                original_aliases=item["accepted_answers"],
                counterfactual_aliases=item[
                    "counterfactual_accepted"],
            )
            if item["counterfactual_accepted"] else {
                "outcome": (
                    "original" if matched is not None
                    else "other_invalid"),
                "matched_original_alias": matched,
                "matched_counterfactual_alias": None,
            }
        )
        row = {
            "study_id": "jspace-phase4",
            "phase": "development",
            "tier": config["tier"],
            "evidence_id": config["evidence_id"],
            "model_id": model["model_id"],
            "model_revision": model["revision"],
            "config_sha256": input_manifest.config_sha256,
            "bank_sha256": input_manifest.bank_sha256,
            "item_id": item_id,
            "fact_id": item["fact_id"],
            "canonical_family": item["canonical_family"],
            "relation_group": item["relation_group"],
            "variant": item["variant"],
            "bank": item["bank"],
            "generation": generated[:160],
            "capable_generation": matched is not None,
            "matched_alias": matched,
            "generation_outcome": generation_grade["outcome"],
            "alias_set_hash": alias_manifest[
                "token_manifest_sha256"],
            "prefix_disjoint_aliases_json": json.dumps(
                alias_manifest["prefix_disjoint_aliases"],
                ensure_ascii=False),
            "score_by_alias_json": json.dumps(
                lp_by_alias, sort_keys=True, ensure_ascii=False),
            "score_aggregate": aggregate_alias_lps(
                lp_by_alias,
                alias_manifest["prefix_disjoint_aliases"]),
            "scoring_spec_sha256": scoring_hash,
        }
        if counterfactual_manifest is not None:
            row.update({
                "counterfactual_alias_set_hash":
                    counterfactual_manifest[
                        "token_manifest_sha256"],
                "counterfactual_score_by_alias_json": json.dumps(
                    counterfactual_lp_by_alias,
                    sort_keys=True,
                    ensure_ascii=False),
                "counterfactual_score_aggregate":
                    aggregate_alias_lps(
                        counterfactual_lp_by_alias,
                        counterfactual_manifest[
                            "prefix_disjoint_aliases"]),
            })
        state["rows"].append(row)
        state["done"][item_id] = True
        newly_done += 1
        if newly_done % checkpoint_every == 0:
            state_store.write(state)
            elapsed = time.time() - started
            rate = elapsed / newly_done
            print(
                f"{len(state['done'])}/{len(items)} "
                f"({rate:.2f}s/item; "
                f"ETA={(len(items) - len(state['done'])) * rate / 60:.1f}m)",
                flush=True,
            )
    state_store.write(state)

    frame = pd.DataFrame(state["rows"]).sort_values("item_id")
    parquet_path = output_dir / f"g5_bank_{config['slug']}.parquet"
    frame.to_parquet(parquet_path, index=False)
    family_capability = (
        frame[frame.variant.isin(["direct", "composed"])]
        .groupby(["bank", "canonical_family"])
        .capable_generation.mean()
    )
    payload = {
        "schema_version": 1,
        "n_items": int(len(frame)),
        "n_facts": int(frame.fact_id.nunique()),
        "n_families": int(frame.canonical_family.nunique()),
        "capable_rate": float(frame.capable_generation.mean()),
        "capable_by_bank": {
            str(key): float(value)
            for key, value in frame.groupby(
                "bank").capable_generation.mean().items()
        },
        "capable_by_variant": {
            str(key): float(value)
            for key, value in frame.groupby(
                "variant").capable_generation.mean().items()
        },
        "families_fully_capable_direct_composed": int(
            family_capability.eq(1.0).groupby(level=0).sum().sum()
        ),
        "alias_aggregation": DEFAULT_SPEC.alias_aggregation,
        "gpu": gpu,
    }
    result_path = output_dir / f"g5_bank_{config['slug']}.json"
    command = (
        "python -m jspace_phase4.experiments.p4_g5_bank_scoring "
        f"--config {arguments.config}")
    inputs = {
        "input_manifest": file_sha256(manifest_path),
        **bank_hashes,
    }
    write_result4(
        payload,
        result_path,
        Provenance4(
            evidence_id=config["evidence_id"],
            tier=config["tier"],
            command=command,
            inputs=inputs,
            input_manifest_sha256=input_manifest.sha256(),
            model=model,
            seed_contract=SEED_CONTRACT,
        ),
    )
    create(
        config["evidence_id"],
        tier=config["tier"],
        what=(
            f"Phase 4 prospective-alias G5 development scoring on "
            f"{config['slug']}: {len(frame)} immutable item rows; "
            f"capable rate {payload['capable_rate']:.4f}."),
        command=command,
        outputs=[result_path, parquet_path, manifest_path],
        inputs=inputs,
    )
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
