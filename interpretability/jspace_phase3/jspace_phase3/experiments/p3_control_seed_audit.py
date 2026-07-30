"""Phase 3 release audit for the unrecorded Python-hash control seed.

Runs a balanced 40+40 frozen Qwen subset under five explicit seeds. Baseline
and span-safe J are deterministic and measured once per item; every audit seed
generates a fresh exact rank-and-energy matched control from the same J
profile. Frozen outcomes are never rewritten.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from jspace_part2.dictionaries import build_j_dictionaries
from jspace_part2.lib import sha256_file
from jspace_part2.paths import resolve as resolve_uri

from ..ablator3 import (Phase3JAblator, profile_from_p3log,
                        teacher_forced_matched_arm)
from ..gpu import require_cuda_gpu
from ..paths3 import metrics_dir, run_root
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           resolve_model, write_result3)
from ..scoring import DEFAULT_SPEC, ScoringSession
from ..seeds import SEED_CONTRACT, stable_seed
from ..stats import exact_signflip_test
from .p3_inference_audit import (effect_bootstrap,
                                 family_weighted_randomization)
from .p3_protected_answer_audit import (
    CONFIGS, DATA_ROOT, REPO_ROOT, SIDES, atomic_json, canonical_hash,
    load_frozen_items, subprocess_git_head, tokenizer_manifest,
    validate_protect_k, validate_state_header)

EVIDENCE_ID = "p3-control-seed-contract-audit-v1"
TIER = "methods"
SLUG = "qwen36-27b"
AUDIT_SEEDS = (11, 101, 1009, 4242, 31337)
N_ITEMS_PER_SIDE = 40
MIN_FAMILIES = 12
BASELINE_TOLERANCE = 0.002


def outcome_path(slug: str, side: str) -> Path:
    suffix = "" if side == "confirmatory" else "_replication"
    return (
        metrics_dir(slug) / f"p3_grid{suffix}"
        / f"p3_grid{suffix}_{slug}.parquet"
    )


def load_outcome_metadata(slug: str, side: str) -> pd.DataFrame:
    return pd.read_parquet(
        outcome_path(slug, side),
        columns=["item_id", "fact_id", "variant", "canonical_family",
                 "bank", "relation_group"])


def common_complete_fact_ids(side: str) -> set[str]:
    common = None
    for slug in ("olmo31-think", "olmo31-instruct", "qwen36-27b"):
        meta = load_outcome_metadata(slug, side)
        complete = {
            fact_id for fact_id, sub in meta.groupby("fact_id")
            if set(sub["variant"]) == {"direct", "composed"}
        }
        common = complete if common is None else common & complete
    return common or set()


def select_balanced_items(items: list[dict], eligible_facts: set[str], *,
                          side: str,
                          n_items: int = N_ITEMS_PER_SIDE) -> list[dict]:
    if n_items % 2:
        raise ValueError("balanced direct/composed subset needs even n_items")
    frame = pd.DataFrame([
        {
            "item_id": item["item_id"],
            "fact_id": item["fact_id"],
            "variant": item["variant"],
            "canonical_family": item["canonical_family"],
        }
        for item in items if item["fact_id"] in eligible_facts
    ])
    pairs = {
        fact_id: sub
        for fact_id, sub in frame.groupby("fact_id")
        if set(sub["variant"]) == {"direct", "composed"}
    }
    by_family: dict[str, list[str]] = {}
    for fact_id, sub in pairs.items():
        family = str(sub["canonical_family"].iloc[0])
        by_family.setdefault(family, []).append(fact_id)
    for family in by_family:
        by_family[family].sort(
            key=lambda fact_id: stable_seed(
                f"p3-control-seed-audit-{side}-fact", fact_id))
    family_order = sorted(
        by_family,
        key=lambda family: stable_seed(
            f"p3-control-seed-audit-{side}-family", family))
    need_facts = n_items // 2
    selected_facts = []
    depth = 0
    while len(selected_facts) < need_facts:
        progressed = False
        for family in family_order:
            facts = by_family[family]
            if depth < len(facts):
                selected_facts.append(facts[depth])
                progressed = True
                if len(selected_facts) == need_facts:
                    break
        if not progressed:
            break
        depth += 1
    if len(selected_facts) != need_facts:
        raise RuntimeError(
            f"{side}: only {len(selected_facts)} eligible fact pairs")
    lookup = {item["item_id"]: item for item in items}
    selected = []
    for fact_id in selected_facts:
        for variant in ("direct", "composed"):
            selected.append(lookup[f"{fact_id}#{variant}"])
    n_families = len({item["canonical_family"] for item in selected})
    if n_families < MIN_FAMILIES:
        raise RuntimeError(
            f"{side}: selection has {n_families} < {MIN_FAMILIES} families")
    return sorted(selected, key=lambda row: row["item_id"])


def profile_digest(profile: dict) -> str:
    serial = {}
    for layer, values in sorted(profile.items()):
        serial[str(layer)] = {
            name: tensor.tolist() for name, tensor in sorted(values.items())
        }
    return canonical_hash(serial)


def audit_state_header(config: dict, model_path: Path, lens_path: Path,
                       tok_manifest: dict, selection: dict) -> dict:
    partition = REPO_ROOT / config["partition_uri"]
    return {
        "schema_version": 1,
        "code_commit": subprocess_git_head(),
        "config_sha256": sha256_file(CONFIGS["confirmatory"]),
        "model": resolve_model(str(model_path)),
        "tokenizer_manifest_sha256": tok_manifest["manifest_sha256"],
        "lens_sha256": sha256_file(lens_path),
        "banks": {
            name: sha256_file(DATA_ROOT / name) for name in config["banks"]},
        "partition_sha256": sha256_file(partition),
        "selection_sha256": canonical_hash(selection),
        "audit_seeds": list(AUDIT_SEEDS),
        "seed_contract": SEED_CONTRACT,
        "matched_control": "instant_rank_energy_matched",
    }


def frozen_qwen_rows(side: str, item_ids: set[str]) -> pd.DataFrame:
    columns = [
        "item_id", "fact_id", "variant", "canonical_family",
        "lp_baseline", "lp_meanJ_span_safe", "lp_ss_matched",
    ]
    frame = pd.read_parquet(outcome_path(SLUG, side), columns=columns)
    return frame[frame["item_id"].isin(item_ids)].copy()


def run_gpu(config: dict, selections: dict[str, list[dict]],
            out_dir: Path) -> tuple[Path, dict]:
    model_path = Path(resolve_uri(config["model_uri"], must_exist=True))
    lens_path = Path(resolve_uri(config["lens_uri"], must_exist=True))
    tok_manifest = tokenizer_manifest(model_path)
    selection_manifest = {
        side: [item["item_id"] for item in selections[side]]
        for side in SIDES}
    header = audit_state_header(
        config, model_path, lens_path, tok_manifest, selection_manifest)
    state_path = out_dir / "control_seed_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        validate_state_header(state.get("header", {}), header)
    else:
        state = {
            "header": header, "done": {}, "rows": [],
            "started_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    total = sum(len(items) for items in selections.values()) * len(
        AUDIT_SEEDS)
    if len(state["done"]) < total:
        gpu = require_cuda_gpu()
        print(json.dumps({"gpu_gate": gpu}, indent=1), flush=True)
        import transformers
        import jlens
        from jlens import JacobianLens

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16).to("cuda").eval()
        if not next(hf.parameters()).is_cuda:
            raise RuntimeError("model parameters are not on CUDA")
        model = jlens.from_hf(hf, tokenizer)
        session = ScoringSession(tokenizer, DEFAULT_SPEC, device="cuda")
        lens = JacobianLens.load(str(lens_path))
        band = config["band"]
        dictionaries = build_j_dictionaries(hf, lens, band)
        ablator = Phase3JAblator(model.layers, band)
        k = int(config["k"])
        protect_k = int(config["protect_top_k"])

        def j_arm(ids, protect_sets):
            ablator.log = type(ablator.log)()
            ablator.phase, ablator.forward_index = "prefill", 0
            ablator.mode = {
                "dicts": dictionaries, "k": k, "nonneg": True,
                "protect_sets": protect_sets, "active_phases": {"prefill"},
                "span_safe": True, "record_overlap": True,
                "answer_id": None,
            }
            with ablator:
                logits = hf(
                    input_ids=ids, use_cache=False).logits[0].float()
            ablator.mode = None
            return logits.cpu(), ablator.log

        historical = {
            side: frozen_qwen_rows(
                side, {item["item_id"] for item in selections[side]})
            .set_index("item_id")
            for side in SIDES
        }
        started = time.time()
        newly_done = 0
        watchdog = run_root() / "WATCHDOG_HEARTBEAT.log"
        for side in SIDES:
            for item in selections[side]:
                item_id = item["item_id"]
                missing = [
                    seed for seed in AUDIT_SEEDS
                    if f"{side}:{item_id}:{seed}" not in state["done"]]
                if not missing:
                    continue
                alias = item["accepted_answers"][0]
                full, n_prompt = session.full_ids(item["prompt"], alias)
                with torch.inference_mode():
                    clean = hf(
                        input_ids=full, use_cache=False).logits[0].float()
                    protect_sets = clean.topk(
                        protect_k, dim=-1).indices
                    baseline_lp = session.answer_seq_lp(
                        full, clean.cpu(), n_prompt)
                    j_logits, jlog = j_arm(full, protect_sets)
                    j_lp = session.answer_seq_lp(
                        full, j_logits, n_prompt)
                    profile = profile_from_p3log(
                        jlog, overlap_records=jlog.overlap)
                    digest = profile_digest(profile)
                    frozen = historical[side].loc[item_id]
                    base_err = abs(
                        baseline_lp - float(frozen["lp_baseline"]))
                    j_err = abs(
                        j_lp - float(frozen["lp_meanJ_span_safe"]))
                    if max(base_err, j_err) > BASELINE_TOLERANCE:
                        atomic_json(state_path, state)
                        raise RuntimeError(
                            f"{side}:{item_id}: baseline/J reproduction "
                            f"errors {base_err}, {j_err}")
                    for seed in missing:
                        seed_base = stable_seed(
                            "p3-control-seed-audit", item_id, seed)
                        control_logits, control_log = \
                            teacher_forced_matched_arm(
                                hf, model.layers, band, dictionaries,
                                full, profile,
                                variant="instant_rank_energy_matched",
                                protect_sets=protect_sets,
                                seed_base=seed_base)
                        control_lp = session.answer_seq_lp(
                            full, control_logits, n_prompt)
                        row = {
                            "side": side,
                            "item_id": item_id,
                            "fact_id": item["fact_id"],
                            "variant": item["variant"],
                            "bank": item["bank"],
                            "canonical_family": item[
                                "canonical_family"],
                            "relation_group": item["relation_group"],
                            "audit_seed": int(seed),
                            "stable_seed": int(seed_base),
                            "lp_baseline": baseline_lp,
                            "lp_meanJ_span_safe": j_lp,
                            "lp_ss_matched": control_lp,
                            "historical_lp_ss_matched": float(
                                frozen["lp_ss_matched"]),
                            "baseline_abs_error": base_err,
                            "j_abs_error": j_err,
                            "j_profile_sha256": digest,
                            "j_overlap_json": json.dumps(
                                jlog.overlap_summary(),
                                sort_keys=True),
                            "matched_summary_json": json.dumps(
                                control_log.matched_summary(),
                                sort_keys=True),
                        }
                        state["rows"].append(row)
                        key = f"{side}:{item_id}:{seed}"
                        state["done"][key] = int(time.time() - started)
                        newly_done += 1
                        atomic_json(state_path, state)
                elapsed = time.time() - started
                rate = elapsed / max(newly_done, 1)
                message = (
                    f"p3_control_seed_audit {len(state['done'])}/{total}; "
                    f"{rate:.2f}s/seed-cell; ETA "
                    f"{(total-len(state['done']))*rate/60:.1f}m"
                )
                print(message, flush=True)
                with open(watchdog, "a") as stream:
                    stream.write(
                        time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                        + " " + message + "\n")
        state["completed_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state["gpu"] = gpu
        atomic_json(state_path, state)
        del lens, dictionaries, model, hf
        torch.cuda.empty_cache()
    rows = pd.DataFrame(state["rows"]).sort_values(
        ["side", "item_id", "audit_seed"]).reset_index(drop=True)
    raw_path = out_dir / "p3_control_seed_audit_rows.parquet"
    rows.to_parquet(raw_path, index=False)
    return raw_path, state


def original_olmo_composition(side: str, fact_ids: set[str]) -> pd.DataFrame:
    rows = []
    for slug in ("olmo31-think", "olmo31-instruct"):
        frame = pd.read_parquet(
            outcome_path(slug, side),
            columns=["fact_id", "variant", "canonical_family",
                     "lp_meanJ_span_safe", "lp_ss_matched"])
        frame = frame[frame["fact_id"].isin(fact_ids)].copy()
        frame["specific"] = (
            frame["lp_meanJ_span_safe"] - frame["lp_ss_matched"])
        pivot = frame.pivot(
            index=["fact_id", "canonical_family"],
            columns="variant", values="specific").reset_index()
        pivot[f"{slug}_composition"] = (
            pivot["composed"] - pivot["direct"])
        rows.append(pivot[[
            "fact_id", "canonical_family", f"{slug}_composition"]])
    return rows[0].merge(
        rows[1], on=["fact_id", "canonical_family"], validate="one_to_one")


def seed_summary(rows: pd.DataFrame, side: str) -> tuple[dict, pd.DataFrame]:
    side_rows = rows[rows["side"] == side].copy()
    fact_ids = set(side_rows["fact_id"])
    olmo = original_olmo_composition(side, fact_ids)
    summaries, item_control = {}, {}
    detail_rows = []
    for seed in AUDIT_SEEDS:
        frame = side_rows[side_rows["audit_seed"] == seed].copy()
        frame["delta_J"] = (
            frame["lp_meanJ_span_safe"] - frame["lp_baseline"])
        frame["delta_C"] = frame["lp_ss_matched"] - frame["lp_baseline"]
        frame["specific"] = frame["delta_J"] - frame["delta_C"]
        item_control[seed] = frame.set_index("item_id")["delta_C"]
        tail_difference = (
            (frame["delta_J"] < -1.0).astype(float)
            - (frame["delta_C"] < -1.0).astype(float)
        ).to_numpy()
        p2 = family_weighted_randomization(
            tail_difference, frame["canonical_family"].to_numpy())
        p2["effect_size_interval"] = effect_bootstrap(
            tail_difference, frame["canonical_family"].to_numpy())
        qwen_comp = frame.pivot(
            index=["fact_id", "canonical_family"],
            columns="variant", values="specific").reset_index()
        qwen_comp["qwen_composition"] = (
            qwen_comp["composed"] - qwen_comp["direct"])
        p1 = qwen_comp.merge(
            olmo, on=["fact_id", "canonical_family"],
            validate="one_to_one")
        p1["diff"] = p1["qwen_composition"] - 0.5 * (
            p1["olmo31-think_composition"]
            + p1["olmo31-instruct_composition"])
        family_values = p1.groupby(
            "canonical_family", sort=True)["diff"].mean()
        exact = exact_signflip_test(family_values.to_numpy())
        control_family = frame.groupby(
            "canonical_family")["delta_C"].mean()
        summary = {
            "control_family_weighted_mean": float(control_family.mean()),
            "control_item_weighted_mean": float(frame["delta_C"].mean()),
            "control_tail_rate": float((frame["delta_C"] < -1.0).mean()),
            "specific_family_weighted_mean": float(
                frame.groupby("canonical_family")["specific"].mean().mean()),
            "P3-P2_subset": p2,
            "P3-P1_subset": exact,
        }
        summaries[str(seed)] = summary
        detail_rows.append(pd.DataFrame({
            "side": side, "audit_seed": seed,
            "canonical_family": family_values.index,
            "p3p1_family_value": family_values.values,
        }))
    controls = pd.concat(item_control, axis=1)
    controls.columns = [str(c) for c in controls.columns]
    correlations = controls.corr().to_dict()
    p2_values = np.array([
        summaries[str(seed)]["P3-P2_subset"]["estimate"]
        for seed in AUDIT_SEEDS])
    p1_values = np.array([
        summaries[str(seed)]["P3-P1_subset"]["estimate"]
        for seed in AUDIT_SEEDS])
    p2_decisions = [
        summaries[str(seed)]["P3-P2_subset"]["p_plus_one"] < 0.05
        for seed in AUDIT_SEEDS]
    p1_decisions = [
        summaries[str(seed)]["P3-P1_subset"]["p"] < 0.05
        for seed in AUDIT_SEEDS]
    p2_range = float(np.ptp(p2_values))
    p1_range = float(np.ptp(p1_values))
    sign_flip = (
        np.any(np.sign(p2_values) != np.sign(p2_values[0]))
        or np.any(np.sign(p1_values) != np.sign(p1_values[0]))
    )
    decision_flip = len(set(p2_decisions)) > 1 or len(set(p1_decisions)) > 1
    meaningful = decision_flip or sign_flip or p2_range > 0.03 \
        or p1_range > 0.10
    if sign_flip or decision_flip:
        gate = "DECISION-SENSITIVE"
    elif meaningful:
        gate = "SEED-SENSITIVE BUT BOUNDED"
    else:
        gate = "ROBUST"
    summary = {
        "n_items": int(side_rows["item_id"].nunique()),
        "n_families": int(side_rows["canonical_family"].nunique()),
        "per_seed": summaries,
        "pairwise_control_delta_correlation": correlations,
        "seed_ensemble": {
            "P3-P2_mean": float(p2_values.mean()),
            "P3-P2_empirical_interval": [
                float(x) for x in np.quantile(p2_values, [0.025, 0.975])],
            "P3-P1_mean": float(p1_values.mean()),
            "P3-P1_empirical_interval": [
                float(x) for x in np.quantile(p1_values, [0.025, 0.975])],
        },
        "worst_pairwise_movement": {
            "P3-P2": p2_range,
            "P3-P1": p1_range,
        },
        "decisions": {
            "P3-P2_reject_by_seed": p2_decisions,
            "P3-P1_reject_by_seed": p1_decisions,
        },
        "gate": gate,
        "expansion_required": bool(meaningful),
        "expansion_rule": (
            "expand to full Qwen if any sign/decision changes, P3-P2 "
            "range > .03, or P3-P1 range > .10"
        ),
    }
    return summary, pd.concat(detail_rows, ignore_index=True)


def analyze(raw_path: Path, out_dir: Path) -> tuple[dict, Path]:
    rows = pd.read_parquet(raw_path)
    counts = rows.groupby(["side", "item_id"])["audit_seed"].nunique()
    if not bool((counts == len(AUDIT_SEEDS)).all()):
        raise RuntimeError("not every selected item has all audit seeds")
    deterministic = rows.groupby(["side", "item_id"]).agg(
        baseline_n=("lp_baseline", "nunique"),
        j_n=("lp_meanJ_span_safe", "nunique"),
        profile_n=("j_profile_sha256", "nunique"),
        baseline_max_error=("baseline_abs_error", "max"),
        j_max_error=("j_abs_error", "max"),
    )
    if not bool(((deterministic[["baseline_n", "j_n", "profile_n"]]
                  == 1).all()).all()):
        raise RuntimeError("baseline or J arm changed across audit seeds")
    if float(deterministic[[
            "baseline_max_error", "j_max_error"]].to_numpy().max()
             ) > BASELINE_TOLERANCE:
        raise RuntimeError("baseline/J reproduction exceeded tolerance")
    payload = {
        "historical_limitation": (
            "The original Phase 3 matched controls used Python hash(item_id) "
            "without recording PYTHONHASHSEED; the historical salt cannot be "
            "reconstructed. Baseline and J outcomes are unaffected."
        ),
        "seed_contract": SEED_CONTRACT,
        "audit_seeds": list(AUDIT_SEEDS),
        "baseline_and_J_reproduction": {
            "exact_across_seed_rows": True,
            "max_baseline_abs_error_vs_frozen": float(
                deterministic["baseline_max_error"].max()),
            "max_J_abs_error_vs_frozen": float(
                deterministic["j_max_error"].max()),
            "tolerance": BASELINE_TOLERANCE,
        },
        "sides": {},
    }
    family_tables = []
    for side in SIDES:
        summary, family = seed_summary(rows, side)
        payload["sides"][side] = summary
        family_tables.append(family)
    gates = {payload["sides"][side]["gate"] for side in SIDES}
    if "DECISION-SENSITIVE" in gates:
        overall = "DECISION-SENSITIVE"
    elif "SEED-SENSITIVE BUT BOUNDED" in gates:
        overall = "SEED-SENSITIVE BUT BOUNDED"
    else:
        overall = "ROBUST"
    payload["overall_gate"] = overall
    payload["full_qwen_expansion_required"] = any(
        payload["sides"][side]["expansion_required"] for side in SIDES)
    family_path = out_dir / "p3_control_seed_p3p1_family_values.parquet"
    pd.concat(family_tables, ignore_index=True).to_parquet(
        family_path, index=False)
    return payload, family_path


def main() -> None:
    require_clean_tree("--allow-dirty" in sys.argv)
    configs = {
        side: yaml.safe_load(CONFIGS[side].read_text()) for side in SIDES}
    validate_protect_k(configs)
    config = configs["confirmatory"]
    selections = {}
    for side in SIDES:
        items = load_frozen_items(side, configs[side])
        selections[side] = select_balanced_items(
            items, common_complete_fact_ids(side), side=side)
    selection_manifest = {
        side: {
            "item_ids": [item["item_id"] for item in selections[side]],
            "n_items": len(selections[side]),
            "n_families": len({
                item["canonical_family"] for item in selections[side]}),
        }
        for side in SIDES
    }
    out_dir = metrics_dir(SLUG) / "release_audit" / "control_seed"
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_path = out_dir / "p3_control_seed_selection.json"
    atomic_json(selection_path, selection_manifest)
    raw_path, state = run_gpu(config, selections, out_dir)
    if "--collect-only" in sys.argv:
        print(f"control-seed rows banked: {raw_path}")
        return
    payload, family_path = analyze(raw_path, out_dir)
    payload["selection"] = selection_manifest
    result_path = out_dir / "p3_control_seed_audit.json"
    cmd = "python -m jspace_phase3.experiments.p3_control_seed_audit"
    inputs = {
        "selection": sha256_file(selection_path),
        "qwen_lens": state["header"]["lens_sha256"],
        "config": sha256_file(CONFIGS["confirmatory"]),
        "partition": state["header"]["partition_sha256"],
    }
    write_result3(payload, result_path, Provenance3(
        evidence_id=EVIDENCE_ID, tier=TIER, command=cmd,
        config_path=str(CONFIGS["confirmatory"]), inputs=inputs,
        model=state["header"]["model"], seed=31337))
    register(
        EVIDENCE_ID, tier=TIER, command=cmd,
        what=(
            "Historical Python-hash limitation plus five-seed Qwen "
            f"matched-control sensitivity audit: {payload['overall_gate']}"
        ),
        outputs=[result_path, raw_path, selection_path, family_path],
        inputs=inputs,
    )
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()

