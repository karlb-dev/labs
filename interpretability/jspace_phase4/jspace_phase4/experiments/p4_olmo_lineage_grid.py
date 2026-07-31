"""GPU-only Phase 4 OLMo span-safe development grid."""
from __future__ import annotations

import argparse
import json
import time
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

from ..gpu import assert_model_on_cuda, require_cuda_gpu
from ..manifests import (
    InputManifest,
    atomic_json,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths4 import materialize_local_file, metrics_dir, resolve_uri
from ..provenance4 import Provenance4, write_result4
from ..registry4 import create
from ..scoring4 import (
    DEFAULT_SPEC,
    ScoringSession,
    aggregate_alias_lps,
    canonical_alias_for,
)
from ..seeds import SEED_CONTRACT, stable_rng, stable_seed
from ..state import StateHeader, StateStore


RAW_CONDITIONS = (
    "meanJ_span_safe",
    "meanJ_label_protected",
    "mechanics_random",
    "logit_label_protected",
)
MATCHED_CONDITIONS = (
    ("instant_rank_energy_matched", "ss_matched", "span_safe"),
    ("prot_energy_matched", "prot_energy_matched", "label"),
)
ALL_CONDITIONS = (
    "baseline",
    "meanJ_span_safe",
    "ss_matched",
    "meanJ_label_protected",
    "prot_energy_matched",
    "mechanics_random",
    "logit_label_protected",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def model_reference(uri: str) -> dict:
    if not uri.startswith("model://") or "@" not in uri:
        raise ValueError("model URI must pin a revision")
    model_id, revision = uri[len("model://"):].rsplit("@", 1)
    return {"model_id": model_id, "revision": revision}


def tokenizer_source_hash(model_path: Path) -> str:
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
    )
    hashes = {
        name: file_sha256(model_path / name)
        for name in names
        if (model_path / name).exists()
    }
    if not hashes:
        raise RuntimeError("model snapshot has no tokenizer files")
    return object_sha256(hashes)


def capable_fact_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    required = {"bank", "fact_id", "variant", "capable_generation"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"G5 parquet lacks {sorted(missing)}")
    direct_composed = frame[
        frame.variant.isin(["direct", "composed"])]
    eligible = set()
    for (bank, fact_id), group in direct_composed.groupby(
            ["bank", "fact_id"], sort=True):
        if (
                len(group) == 2
                and set(group.variant) == {"direct", "composed"}
                and bool(group.capable_generation.all())):
            eligible.add((str(bank), str(fact_id)))
    return eligible


def cohort_items(bank_paths: list[Path],
                 fact_keys: set[tuple[str, str]]) -> list[dict]:
    items = []
    for path in bank_paths:
        for bundle in load_bank(path):
            for item in bundle.as_items():
                key = (str(item["bank"]), str(item["fact_id"]))
                if (
                        key in fact_keys
                        and item["variant"] in {"direct", "composed"}):
                    items.append(item)
    items.sort(key=lambda item: item["item_id"])
    if len(items) != 2 * len(fact_keys):
        raise RuntimeError(
            "bank/G5 cohort mismatch: expected exactly direct and composed "
            f"for {len(fact_keys)} facts, got {len(items)} items")
    if len({item["item_id"] for item in items}) != len(items):
        raise RuntimeError("cohort item IDs are not unique")
    return items


def require_hook_fired(log, condition: str) -> None:
    fires = getattr(log, "hook_fires", {})
    if int(fires.get("prefill", 0)) <= 0:
        raise RuntimeError(
            f"{condition} intervention hook never fired in prefill")
    forbidden = sum(
        int(value) for phase, value in fires.items()
        if phase != "prefill")
    if forbidden:
        raise RuntimeError(
            f"{condition} intervention fired outside prefill: {fires}")


@torch.no_grad()
def main() -> None:  # noqa: C901
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    clean = require_clean_tree()
    gpu = require_cuda_gpu()
    evidence_id = config["evidence_id"]
    model = model_reference(config["model_uri"])
    model_path = resolve_uri(config["model_uri"])
    bank_paths = [resolve_uri(uri) for uri in config["banks"]]
    bank_hashes = {
        str(uri): file_sha256(path)
        for uri, path in zip(config["banks"], bank_paths)
    }
    g5_path = resolve_uri(config["g5_parquet_uri"])
    g5_result_path = resolve_uri(config["g5_result_uri"])
    g5 = pd.read_parquet(g5_path)
    g5_result = json.loads(g5_result_path.read_text())
    if (
            g5_result["provenance"]["evidence_id"]
            != config["g5_evidence_id"]):
        raise RuntimeError("unexpected upstream G5 evidence")
    fact_keys = capable_fact_keys(g5)
    items = cohort_items(bank_paths, fact_keys)
    g5_by_item = g5.set_index("item_id", verify_integrity=True)

    output_dir = (
        metrics_dir(config["slug"])
        / "lineage_grid"
        / evidence_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort_payload = {
        "schema_version": 1,
        "selection": (
            "same-model prospective G5 generation-capable on both "
            "direct and composed variants"),
        "upstream_evidence_id": g5_result[
            "provenance"]["evidence_id"],
        "fact_keys": [
            {"bank": bank, "fact_id": fact_id}
            for bank, fact_id in sorted(fact_keys)
        ],
        "item_ids": [item["item_id"] for item in items],
        "n_facts": len(fact_keys),
        "n_items": len(items),
    }
    cohort_manifest = {
        "schema_version": 1,
        "payload": cohort_payload,
        "payload_sha256": object_sha256(cohort_payload),
    }
    cohort_path = output_dir / "cohort_manifest.json"
    atomic_json(cohort_path, cohort_manifest)

    lens_sha256 = config["lens_sha256"]
    lens_path = materialize_local_file(
        config["lens_uri"], expected_sha256=lens_sha256)
    scoring_sha256 = object_sha256(DEFAULT_SPEC.as_dict())
    upstream = {
        str(config["g5_parquet_uri"]): file_sha256(g5_path),
        str(config["g5_result_uri"]): file_sha256(g5_result_path),
    }
    input_manifest = InputManifest(
        experiment_id=evidence_id,
        config_sha256=file_sha256(config_path),
        model_id=model["model_id"],
        model_revision=model["revision"],
        tokenizer_manifest_sha256=tokenizer_source_hash(model_path),
        lens_sha256=lens_sha256,
        bank_sha256=object_sha256(bank_hashes),
        partition_sha256=cohort_manifest["payload_sha256"],
        scoring_spec_sha256=scoring_sha256,
        upstream=upstream,
        code_commit=clean["code_commit"],
    )
    manifest_path = output_dir / "input_manifest.json"
    atomic_json(manifest_path, input_manifest.envelope())
    state_store = StateStore(
        output_dir / "state.json",
        StateHeader(
            evidence_id=evidence_id,
            input_manifest_sha256=input_manifest.sha256(),
            config_sha256=input_manifest.config_sha256,
            model_revision=input_manifest.model_revision,
            bank_sha256=input_manifest.bank_sha256,
            partition_sha256=input_manifest.partition_sha256,
        ),
    )
    state = state_store.load() or {
        "done": {},
        "rows": [],
        "baseline_checked": 0,
        "baseline_stop_events": [],
        "gpu": gpu,
    }

    import jlens
    import transformers
    from jlens import JacobianLens

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16).to("cuda").eval()
    assert_model_on_cuda(hf_model)
    wrapped = jlens.from_hf(hf_model, tokenizer)
    session = ScoringSession(tokenizer, DEFAULT_SPEC, device="cuda")
    lens = JacobianLens.load(str(lens_path))
    band = [int(value) for value in config["band"]]
    k = int(config["k"])
    protect_top_k = int(config["protect_top_k"])
    j_dictionaries = build_j_dictionaries(
        hf_model, lens, band)
    logit_dictionaries = build_logit_dictionary(
        hf_model, band)
    vocabulary, width = j_dictionaries[band[0]].shape
    random_generator = torch.Generator().manual_seed(stable_seed(
        experiment_id=evidence_id,
        item_id="global-random-dictionary",
        condition="mechanics_random",
        base_seed=int(config["base_seed"]),
    ))
    random_dictionary = torch.nn.functional.normalize(
        torch.randn(
            vocabulary,
            width,
            generator=random_generator,
            dtype=torch.float32,
        ),
        dim=1,
    ).to("cuda", torch.float16)
    random_dictionaries = {
        layer: random_dictionary for layer in band}
    ablator = Phase3JAblator(wrapped.layers, band)

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
            "answer_id": None,
        }
        try:
            with ablator:
                logits = hf_model(
                    input_ids=full_ids,
                    use_cache=False,
                ).logits[0].float()
            require_hook_fired(
                ablator.log,
                "meanJ_span_safe"
                if span_safe else "meanJ_label_protected",
            )
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
            "answer_id": None,
        }
        try:
            with ablator:
                logits = hf_model(
                    input_ids=full_ids,
                    use_cache=False,
                ).logits[0].float()
            require_hook_fired(ablator.log, condition)
            return logits
        finally:
            ablator.mode = None

    print(
        f"{config['slug']}: {len(items)} cohort items / "
        f"{len(fact_keys)} facts; GPU={gpu['name']}; "
        f"lens={lens_sha256[:12]}; band={band}",
        flush=True,
    )
    started = time.time()
    newly_done = 0
    checkpoint_every = int(config.get("checkpoint_every", 5))
    stop_n = int(config.get("baseline_stop_n", 25))
    stop_tolerance = float(
        config.get("baseline_stop_tolerance", 0.05))
    for item in items:
        item_id = item["item_id"]
        if item_id in state["done"]:
            continue
        g5_row = g5_by_item.loc[item_id]
        canonical_alias = canonical_alias_for(
            session,
            item["accepted_answers"],
            item["canonical_answer"],
        )
        alias_manifest = session.freeze_alias_manifest(
            item["accepted_answers"],
            canonical_alias=canonical_alias,
        )
        if alias_manifest["token_manifest_sha256"] != g5_row.alias_set_hash:
            raise RuntimeError(
                f"alias manifest mismatch against G5 for {item_id}")
        selected_aliases = alias_manifest["prefix_disjoint_aliases"]
        if selected_aliases != json.loads(
                g5_row.prefix_disjoint_aliases_json):
            raise RuntimeError(
                f"prefix-disjoint alias set mismatch for {item_id}")

        alias_data = {}
        baseline_by_alias = {}
        rank_metadata = None
        for alias in selected_aliases:
            full_ids, prompt_length = session.full_ids(
                item["prompt"], alias)
            clean_logits = hf_model(
                input_ids=full_ids,
                use_cache=False,
            ).logits[0].float()
            protect_sets = clean_logits.topk(
                protect_top_k, dim=-1).indices
            baseline_by_alias[alias] = session.answer_sequence_lp(
                full_ids, clean_logits, prompt_length)
            if alias == canonical_alias:
                first_token_ids = {
                    value: int(session.answer_ids(value)[0, 0])
                    for value in item["accepted_answers"]
                }
                rank_metadata = session.clean_first_token_ranks(
                    clean_logits[prompt_length - 1],
                    first_token_ids,
                )
            alias_data[alias] = {
                "full_ids": full_ids,
                "prompt_length": prompt_length,
                "protect_sets": protect_sets,
            }
            del clean_logits
        if rank_metadata is None:
            raise RuntimeError(
                f"canonical alias was not scored for {item_id}")
        baseline_aggregate = aggregate_alias_lps(
            baseline_by_alias, selected_aliases)
        if state["baseline_checked"] < stop_n:
            difference = abs(
                baseline_aggregate - float(g5_row.score_aggregate))
            if difference > stop_tolerance:
                event = {
                    "item_id": item_id,
                    "measured": baseline_aggregate,
                    "manifest": float(g5_row.score_aggregate),
                    "absolute_difference": difference,
                }
                state["baseline_stop_events"].append(event)
                state_store.write(state)
                raise RuntimeError(
                    f"BASELINE STOP RULE: {event}; no intervention "
                    "outcome produced")
            state["baseline_checked"] += 1

        order = list(RAW_CONDITIONS)
        permutation = stable_rng(
            experiment_id=evidence_id,
            item_id=item_id,
            condition="condition-order",
            base_seed=int(config["base_seed"]),
        ).permutation(len(order))
        order = [order[int(index)] for index in permutation]
        scores_by_condition = {
            condition: {} for condition in ALL_CONDITIONS}
        scores_by_condition["baseline"] = baseline_by_alias
        overlap_by_alias = {}
        matched_by_alias = {}
        for alias in selected_aliases:
            data = alias_data[alias]
            profiles = {}
            for condition in order:
                if condition == "meanJ_span_safe":
                    logits, intervention_log = j_arm(
                        data["full_ids"],
                        data["protect_sets"],
                        span_safe=True,
                    )
                    profiles["span_safe"] = profile_from_p3log(
                        intervention_log,
                        overlap_records=intervention_log.overlap,
                    )
                    overlap_by_alias.setdefault(alias, {})[
                        condition] = intervention_log.overlap_summary()
                elif condition == "meanJ_label_protected":
                    logits, intervention_log = j_arm(
                        data["full_ids"],
                        data["protect_sets"],
                        span_safe=False,
                    )
                    profiles["label"] = profile_from_p3log(
                        intervention_log,
                        overlap_records=intervention_log.overlap,
                    )
                    overlap_by_alias.setdefault(alias, {})[
                        condition] = intervention_log.overlap_summary()
                elif condition == "mechanics_random":
                    logits = dictionary_arm(
                        data["full_ids"],
                        data["protect_sets"],
                        random_dictionaries,
                        condition,
                    )
                elif condition == "logit_label_protected":
                    logits = dictionary_arm(
                        data["full_ids"],
                        data["protect_sets"],
                        logit_dictionaries,
                        condition,
                    )
                else:
                    raise RuntimeError(
                        f"unhandled condition {condition!r}")
                scores_by_condition[condition][alias] = (
                    session.answer_sequence_lp(
                        data["full_ids"],
                        logits,
                        data["prompt_length"],
                    )
                )
                del logits
            for variant, condition, profile_name in MATCHED_CONDITIONS:
                alias_seed_id = f"{item_id}\u0000{alias}"

                def seed_factory(layer, forward_index, position,
                                 *, _condition=condition,
                                 _alias_seed_id=alias_seed_id):
                    if forward_index != 0:
                        raise RuntimeError(
                            "teacher-forced matched arm changed forward index")
                    return stable_seed(
                        experiment_id=evidence_id,
                        item_id=_alias_seed_id,
                        condition=_condition,
                        layer=int(layer),
                        position=int(position),
                        base_seed=int(config["base_seed"]),
                    )

                logits, matched_log = teacher_forced_matched_arm(
                    hf_model,
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
                require_hook_fired(matched_log, condition)
                scores_by_condition[condition][alias] = (
                    session.answer_sequence_lp(
                        data["full_ids"],
                        logits,
                        data["prompt_length"],
                    )
                )
                matched_by_alias.setdefault(alias, {})[
                    condition] = matched_log.matched_summary()
                del logits

        row = {
            "study_id": "jspace-phase4",
            "phase": "development",
            "tier": config["tier"],
            "evidence_id": evidence_id,
            "model_id": model["model_id"],
            "model_revision": model["revision"],
            "config_sha256": input_manifest.config_sha256,
            "bank_sha256": input_manifest.bank_sha256,
            "cohort_sha256": input_manifest.partition_sha256,
            "scoring_spec_sha256": scoring_sha256,
            "lens_sha256": lens_sha256,
            "item_id": item_id,
            "fact_id": item["fact_id"],
            "canonical_family": item["canonical_family"],
            "relation_group": item["relation_group"],
            "variant": item["variant"],
            "bank": item["bank"],
            "alias_set_hash": alias_manifest[
                "token_manifest_sha256"],
            "prefix_disjoint_aliases_json": json.dumps(
                selected_aliases, ensure_ascii=False),
            "condition_order_json": json.dumps(order),
            "clean_rank_metadata_json": json.dumps(
                rank_metadata, sort_keys=True),
            "overlap_summary_json": json.dumps(
                overlap_by_alias, sort_keys=True),
            "matched_summary_json": json.dumps(
                matched_by_alias, sort_keys=True),
        }
        for condition in ALL_CONDITIONS:
            row[f"lp_{condition}"] = aggregate_alias_lps(
                scores_by_condition[condition],
                selected_aliases,
            )
            row[f"lp_by_alias_{condition}_json"] = json.dumps(
                scores_by_condition[condition],
                sort_keys=True,
                ensure_ascii=False,
            )
        state["rows"].append(row)
        state["done"][item_id] = True
        newly_done += 1
        if newly_done % checkpoint_every == 0:
            state_store.write(state)
            elapsed = time.time() - started
            rate = elapsed / newly_done
            remaining = len(items) - len(state["done"])
            print(
                f"{len(state['done'])}/{len(items)} "
                f"({rate:.2f}s/item; ETA={remaining * rate / 60:.1f}m)",
                flush=True,
            )
    state_store.write(state)

    frame = pd.DataFrame(state["rows"]).sort_values("item_id")
    parquet_path = output_dir / (
        f"lineage_grid_{config['slug']}.parquet")
    frame.to_parquet(parquet_path, index=False)
    payload = {
        "schema_version": 1,
        "n_items": int(len(frame)),
        "n_facts": int(frame.fact_id.nunique()),
        "n_families": int(frame.canonical_family.nunique()),
        "baseline_checked": int(state["baseline_checked"]),
        "conditions": list(ALL_CONDITIONS),
        "cohort_rule": cohort_payload["selection"],
        "raw_rows_only": True,
        "analysis_note": (
            "Development trajectory analysis is produced separately "
            "from these immutable per-item rows."),
        "gpu": gpu,
    }
    result_path = output_dir / (
        f"lineage_grid_{config['slug']}.json")
    command = (
        "python -m jspace_phase4.experiments.p4_olmo_lineage_grid "
        f"--config {arguments.config}")
    inputs = {
        "input_manifest": file_sha256(manifest_path),
        "cohort_manifest": file_sha256(cohort_path),
        "lens": lens_sha256,
        **bank_hashes,
        **upstream,
    }
    write_result4(
        payload,
        result_path,
        Provenance4(
            evidence_id=evidence_id,
            tier=config["tier"],
            command=command,
            inputs=inputs,
            input_manifest_sha256=input_manifest.sha256(),
            model=model,
            seed_contract=SEED_CONTRACT,
        ),
    )
    create(
        evidence_id,
        tier=config["tier"],
        what=(
            f"Phase 4 OLMo lineage span-safe development grid on "
            f"{config['slug']}: {len(frame)} immutable prospective-"
            "cohort item rows; raw outcomes only."),
        command=command,
        outputs=[
            result_path,
            parquet_path,
            manifest_path,
            cohort_path,
        ],
        inputs=inputs,
    )
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
