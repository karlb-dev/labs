"""Post-freeze Qwen protected-answer protocol-conformance audit.

Rank collection is baseline-only: it reconstructs the frozen cohort from the
partition, banks, and pre-intervention G5 capability manifest, and does not
open an intervention parquet. Only after rank collection is durable does the
analysis phase join the new metadata to immutable outcomes.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

def _find_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path(__file__)).resolve()
    if cur.is_file():
        cur = cur.parent
    for p in [cur, *cur.parents]:
        if (p / ".git").exists():
            return p
    raise RuntimeError("cannot locate git repository root")


import numpy as np
import pandas as pd
import torch
import yaml

from jspace_part2.lib import sha256_file
from jspace_part2.paths import resolve as resolve_uri

from ..bank import load_bank
from ..gpu import require_cuda_gpu
from ..paths3 import metrics_dir, resolve_uri as resolve3, run_root
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           resolve_model, write_result3)
from ..scoring import DEFAULT_SPEC, ScoringSession
from .p3_inference_audit import (effect_bootstrap,
                                 family_weighted_randomization)

EVIDENCE_ID = "p3-protocol-audit-protected-answer-qwen-v1"
TIER = "methods"
SLUG = "qwen36-27b"
SIDES = ("confirmatory", "replication")
THRESHOLDS = (-0.5, -1.0, -1.5, -2.0)
REPO_ROOT = _find_repo_root()
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PACKAGE_ROOT / "data"
CONFIGS = {
    "confirmatory": PACKAGE_ROOT / "configs" / "p3_grid_qwen36-27b.yaml",
    "replication": (
        PACKAGE_ROOT / "configs" / "p3_repl_grid_qwen36-27b.yaml"
    ),
}


def canonical_hash(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_protect_k(configs: dict[str, dict]) -> int:
    values = {int(cfg["protect_top_k"]) for cfg in configs.values()}
    if values != {10}:
        raise RuntimeError(
            f"frozen protect_top_k must be exactly 10, got {sorted(values)}")
    return values.pop()


def tokenizer_manifest(model_path: Path) -> dict:
    names = (
        "tokenizer.json", "tokenizer_config.json", "vocab.json",
        "merges.txt", "special_tokens_map.json", "chat_template.jinja",
    )
    files = {}
    for name in names:
        path = model_path / name
        if path.exists():
            files[name] = sha256_file(path)
    if not files:
        raise RuntimeError(f"no tokenizer files found under {model_path}")
    return {"files": files, "manifest_sha256": canonical_hash(files)}


def load_frozen_items(side: str, cfg: dict) -> list[dict]:
    """Reconstruct the original cohort without opening outcome data."""
    partition_uri = cfg["partition_uri"]
    partition_path = (
        Path(resolve3(partition_uri))
        if "://" in str(partition_uri)
        else REPO_ROOT / partition_uri
    )
    partition = json.loads(partition_path.read_text())["payload"]
    families = set(partition[side])
    g5_dir = metrics_dir(SLUG) / "g5_bank"
    regraded = g5_dir / f"g5_bank_{SLUG}_regraded.parquet"
    g5 = pd.read_parquet(
        regraded if regraded.exists()
        else g5_dir / f"g5_bank_{SLUG}.parquet")
    direct_composed = g5[g5["variant"].isin(["direct", "composed"])]
    capable_facts = {
        fact_id for fact_id, sub in direct_composed.groupby("fact_id")
        if len(sub) == 2 and bool(sub["capable_generation"].all())
    }
    bundles = [
        bundle
        for bank_name in cfg["banks"]
        for bundle in load_bank(DATA_ROOT / bank_name)
    ]
    items = []
    for bundle in bundles:
        if (bundle.canonical_family not in families
                or bundle.fact_id not in capable_facts):
            continue
        for item in bundle.as_items():
            if item["variant"] in ("direct", "composed"):
                items.append(item)
    return sorted(items, key=lambda row: row["item_id"])


def validate_text_hashes(row: dict, item: dict) -> None:
    if row["prompt_text_sha256"] != text_hash(item["prompt"]):
        raise RuntimeError(f"{item['item_id']}: prompt hash mismatch")
    if row["accepted_aliases_text_sha256"] != canonical_hash(
            item["accepted_answers"]):
        raise RuntimeError(f"{item['item_id']}: alias hash mismatch")


def clean_first_token_rank(logits: torch.Tensor, token_id: int) -> int:
    row = logits.float()
    return int((row > row[int(token_id)]).sum().item()) + 1


@torch.inference_mode()
def measure_item(hf, session: ScoringSession, item: dict,
                 *, side: str) -> dict:
    """Baseline-only measurement. No outcome object enters this function."""
    aliases = item["accepted_answers"]
    alias_ids = {
        alias: session.answer_ids(alias)[0].tolist() for alias in aliases
    }
    first_alias = aliases[0]
    first_full, n_prompt = session.full_ids(item["prompt"], first_alias)
    clean = hf(input_ids=first_full, use_cache=False).logits[0].float().cpu()
    boundary = clean[n_prompt - 1]
    lps = {
        first_alias: session.answer_seq_lp(
            first_full, clean, n_prompt)
    }
    del clean
    for alias in aliases[1:]:
        full, alias_n_prompt = session.full_ids(item["prompt"], alias)
        if alias_n_prompt != n_prompt:
            raise RuntimeError(
                f"{item['item_id']}: prompt boundary changed by alias")
        logits = hf(input_ids=full, use_cache=False).logits[0].float().cpu()
        lps[alias] = session.answer_seq_lp(full, logits, n_prompt)
        del logits
    first_tokens = {alias: int(ids[0]) for alias, ids in alias_ids.items()}
    unique_first_tokens = sorted(set(first_tokens.values()))
    ranks = {
        alias: clean_first_token_rank(boundary, token_id)
        for alias, token_id in first_tokens.items()
    }
    log_probs = torch.log_softmax(boundary.float(), dim=-1)
    first_token_mass = float(
        log_probs[unique_first_tokens].exp().sum().item())
    alias_logsumexp = float(np.logaddexp.reduce(
        np.asarray(list(lps.values()), dtype=float)))
    row = {
        "side": side,
        "item_id": item["item_id"],
        "fact_id": item["fact_id"],
        "variant": item["variant"],
        "bank": item["bank"],
        "canonical_family": item["canonical_family"],
        "relation_group": item["relation_group"],
        "rank_exact_scored_alias": int(ranks[first_alias]),
        "rank_min_accepted_alias": int(min(ranks.values())),
        "rank_by_alias_json": json.dumps(ranks, sort_keys=True),
        "protected_exact_scored_alias": bool(ranks[first_alias] <= 10),
        "protected_any_accepted_alias": bool(min(ranks.values()) <= 10),
        "accepted_alias_first_token_mass": first_token_mass,
        "accepted_alias_sequence_logsumexp": alias_logsumexp,
        "accepted_alias_sequence_mass": float(math.exp(alias_logsumexp)),
        "lp_first_alias_remeasured": float(lps[first_alias]),
        "n_prompt_tokens": int(n_prompt),
        "n_tokens_first_alias": int(first_full.shape[1]),
        "prompt_text_sha256": text_hash(item["prompt"]),
        "accepted_aliases_text_sha256": canonical_hash(aliases),
        "prompt_token_sha256": canonical_hash(
            session.prompt_ids(item["prompt"])[0].tolist()),
        "alias_token_sha256": canonical_hash(alias_ids),
        "exact_alias_first_token_id": int(first_tokens[first_alias]),
        "accepted_alias_first_token_ids_json": json.dumps(
            first_tokens, sort_keys=True),
    }
    validate_text_hashes(row, item)
    return row


def atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def state_header(configs: dict[str, dict], model_path: Path,
                 tok_manifest: dict) -> dict:
    partition = REPO_ROOT / configs["confirmatory"]["partition_uri"]
    bank_hashes = {
        name: sha256_file(DATA_ROOT / name)
        for name in configs["confirmatory"]["banks"]
    }
    return {
        "schema_version": 1,
        "code_commit": subprocess_git_head(),
        "config_sha256": {
            side: sha256_file(CONFIGS[side]) for side in SIDES},
        "model": resolve_model(str(model_path)),
        "tokenizer_manifest_sha256": tok_manifest["manifest_sha256"],
        "banks": bank_hashes,
        "partition_sha256": sha256_file(partition),
        "scoring_contract": "phase3-first-alias-piecewise-v1",
        "measurement_contract": "baseline-only-protected-rank-v1",
    }


def subprocess_git_head() -> str:
    import subprocess
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def validate_state_header(saved: dict, current: dict) -> None:
    mismatch = {
        key: (saved.get(key), value)
        for key, value in current.items()
        if saved.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"rank-audit state header mismatch: {mismatch}")


def collect_ranks(configs: dict[str, dict], protect_k: int,
                  out_dir: Path) -> tuple[Path, dict]:
    if protect_k != 10:
        raise RuntimeError("rank collection refused non-frozen protect_k")
    model_uri = configs["confirmatory"]["model_uri"]
    if configs["replication"]["model_uri"] != model_uri:
        raise RuntimeError("confirmatory/replication model revisions differ")
    model_path = Path(resolve_uri(model_uri, must_exist=True))
    tok_manifest = tokenizer_manifest(model_path)
    state_path = out_dir / "protected_answer_rank_state.json"

    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    session = ScoringSession(tokenizer, DEFAULT_SPEC, device="cuda")
    header = state_header(configs, model_path, tok_manifest)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        validate_state_header(state.get("header", {}), header)
    else:
        state = {"header": header, "done": {}, "rows": [],
                 "started_utc": time.strftime(
                     "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    items_by_side = {
        side: load_frozen_items(side, configs[side]) for side in SIDES}
    overlap = (
        {row["item_id"] for row in items_by_side["confirmatory"]}
        & {row["item_id"] for row in items_by_side["replication"]}
    )
    if overlap:
        raise RuntimeError(
            f"confirmatory/replication item overlap: {sorted(overlap)[:3]}")
    total = sum(map(len, items_by_side.values()))
    if len(state["done"]) < total:
        gpu = require_cuda_gpu()
        print(json.dumps({"gpu_gate": gpu}, indent=1), flush=True)
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16).to("cuda").eval()
        if not next(hf.parameters()).is_cuda:
            raise RuntimeError("model parameters are not on CUDA")
        started = time.time()
        newly_done = 0
        watchdog = run_root() / "WATCHDOG_HEARTBEAT.log"
        for side in SIDES:
            for item in items_by_side[side]:
                item_id = item["item_id"]
                key = f"{side}:{item_id}"
                if key in state["done"]:
                    continue
                row = measure_item(hf, session, item, side=side)
                if ((row["rank_exact_scored_alias"] <= protect_k)
                        != row["protected_exact_scored_alias"]):
                    raise AssertionError("protected-exact flag mismatch")
                state["rows"].append(row)
                state["done"][key] = int(time.time() - started)
                newly_done += 1
                if newly_done % 5 == 0:
                    atomic_json(state_path, state)
                    elapsed = time.time() - started
                    rate = elapsed / newly_done
                    message = (
                        f"p3_protected_answer_audit {len(state['done'])}/"
                        f"{total}; {rate:.2f}s/item; ETA "
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
        del hf
        torch.cuda.empty_cache()
    ranks = pd.DataFrame(state["rows"]).sort_values(
        ["side", "item_id"]).reset_index(drop=True)
    rank_path = out_dir / "p3_protected_answer_ranks_qwen.parquet"
    ranks.to_parquet(rank_path, index=False)
    return rank_path, state


def load_outcomes(side: str) -> pd.DataFrame:
    suffix = "" if side == "confirmatory" else "_replication"
    path = (
        metrics_dir(SLUG) / f"p3_grid{suffix}"
        / f"p3_grid{suffix}_{SLUG}.parquet"
    )
    # Explicit whitelist: rank collection has already completed, and analysis
    # reads only the three frozen endpoint columns required by P3-P2.
    columns = [
        "item_id", "fact_id", "variant", "bank", "canonical_family",
        "relation_group", "n_tokens", "lp_baseline",
        "lp_meanJ_span_safe", "lp_ss_matched",
    ]
    return pd.read_parquet(path, columns=columns)


def validate_rank_outcome_join(ranks: pd.DataFrame, outcomes: pd.DataFrame,
                               *, side: str) -> pd.DataFrame:
    rank_side = ranks[ranks["side"] == side].copy()
    rank_ids = set(rank_side["item_id"])
    outcome_ids = set(outcomes["item_id"])
    if rank_ids != outcome_ids:
        raise RuntimeError(
            f"{side}: rank/outcome item mismatch; "
            f"missing_rank={sorted(outcome_ids-rank_ids)[:3]}, "
            f"extra_rank={sorted(rank_ids-outcome_ids)[:3]}")
    joined = outcomes.merge(
        rank_side, on=[
            "item_id", "fact_id", "variant", "bank",
            "canonical_family", "relation_group"],
        how="inner", validate="one_to_one")
    bad_tokens = joined[
        joined["n_tokens"] != joined["n_tokens_first_alias"]]
    if len(bad_tokens):
        raise RuntimeError(
            f"{side}: {len(bad_tokens)} prompt/alias token-length mismatches")
    baseline_error = np.abs(
        joined["lp_baseline"] - joined["lp_first_alias_remeasured"])
    if float(baseline_error.max()) > 0.002:
        raise RuntimeError(
            f"{side}: baseline reproduction max error "
            f"{baseline_error.max()} > 0.002")
    joined["baseline_abs_error"] = baseline_error
    joined["delta_J"] = (
        joined["lp_meanJ_span_safe"] - joined["lp_baseline"])
    joined["delta_C"] = joined["lp_ss_matched"] - joined["lp_baseline"]
    return joined


def analyze_view(frame: pd.DataFrame, rank_field: str | None,
                 protect_k: int) -> dict:
    if rank_field is not None and rank_field not in frame:
        raise RuntimeError(
            f"protected-answer analysis requires rank field {rank_field}")
    subset = (
        frame if rank_field is None
        else frame[frame[rank_field] <= protect_k]
    )
    if subset["canonical_family"].nunique() < 3:
        raise RuntimeError("protected-answer stratum has fewer than 3 families")
    thresholds = {}
    for threshold in THRESHOLDS:
        difference = (
            (subset["delta_J"].to_numpy() < threshold).astype(float)
            - (subset["delta_C"].to_numpy() < threshold).astype(float)
        )
        thresholds[str(threshold)] = {
            **family_weighted_randomization(
                difference, subset["canonical_family"].to_numpy()),
            "effect_size_interval": effect_bootstrap(
                difference, subset["canonical_family"].to_numpy()),
            "tail_rate_J": float(
                (subset["delta_J"] < threshold).mean()),
            "tail_rate_control": float(
                (subset["delta_C"] < threshold).mean()),
        }
    return {
        "rank_field": rank_field,
        "protect_k": protect_k if rank_field else None,
        "n_items": int(len(subset)),
        "n_families": int(subset["canonical_family"].nunique()),
        "threshold_curve": thresholds,
    }


def analyze(rank_path: Path, protect_k: int,
            out_dir: Path) -> tuple[dict, Path]:
    ranks = pd.read_parquet(rank_path)
    sides = set(ranks["side"])
    if sides != set(SIDES):
        raise RuntimeError(f"rank file sides {sides} != {set(SIDES)}")
    ids = {
        side: set(ranks.loc[ranks["side"] == side, "item_id"])
        for side in SIDES}
    if ids["confirmatory"] & ids["replication"]:
        raise RuntimeError("confirmatory and replication rank sets overlap")
    payload = {
        "protocol": {
            "status": "post-freeze deterministic conformance correction",
            "protect_k": protect_k,
            "primary_view": "exact_scored_alias_protected",
            "sensitivity_view": "any_accepted_alias_protected",
            "published_view": "all_items",
            "rank_collection": (
                "baseline-only; intervention parquet opened only after "
                "rank artifact completed"
            ),
        },
        "sides": {},
    }
    joined_frames = []
    for side in SIDES:
        joined = validate_rank_outcome_join(
            ranks, load_outcomes(side), side=side)
        joined.insert(0, "partition_side", side)
        joined_frames.append(joined)
        payload["sides"][side] = {
            "exact_scored_alias_protected": analyze_view(
                joined, "rank_exact_scored_alias", protect_k),
            "any_accepted_alias_protected": analyze_view(
                joined, "rank_min_accepted_alias", protect_k),
            "all_items": analyze_view(joined, None, protect_k),
            "rank_summary": {
                "exact_protected_rate": float(
                    (joined["rank_exact_scored_alias"] <= protect_k).mean()),
                "any_alias_protected_rate": float(
                    (joined["rank_min_accepted_alias"] <= protect_k).mean()),
                "baseline_max_abs_error": float(
                    joined["baseline_abs_error"].max()),
            },
        }
    joined_path = out_dir / "p3_protected_answer_joined_qwen.parquet"
    pd.concat(joined_frames, ignore_index=True).to_parquet(
        joined_path, index=False)
    return payload, joined_path


def main() -> None:
    require_clean_tree("--allow-dirty" in sys.argv)
    configs = {
        side: yaml.safe_load(CONFIGS[side].read_text()) for side in SIDES}
    protect_k = validate_protect_k(configs)
    out_dir = metrics_dir(SLUG) / "release_audit" / "protected_answer"
    out_dir.mkdir(parents=True, exist_ok=True)
    rank_path, state = collect_ranks(configs, protect_k, out_dir)
    if "--collect-only" in sys.argv:
        print(f"baseline-only ranks banked: {rank_path}")
        return
    payload, joined_path = analyze(rank_path, protect_k, out_dir)
    result_path = out_dir / "p3_protected_answer_audit.json"
    cmd = (
        "python -m "
        "jspace_phase3.experiments.p3_protected_answer_audit"
    )
    inputs = {
        "rank_parquet": sha256_file(rank_path),
        "confirmatory_config": sha256_file(CONFIGS["confirmatory"]),
        "replication_config": sha256_file(CONFIGS["replication"]),
        "partition": sha256_file(
            REPO_ROOT / configs["confirmatory"]["partition_uri"]),
        "banks": {
            name: sha256_file(DATA_ROOT / name)
            for name in configs["confirmatory"]["banks"]
        },
    }
    write_result3(payload, result_path, Provenance3(
        evidence_id=EVIDENCE_ID, tier=TIER, command=cmd,
        config_path=str(CONFIGS["confirmatory"]), inputs=inputs,
        model=state["header"]["model"]))
    primary = payload["sides"]["confirmatory"][
        "exact_scored_alias_protected"]["threshold_curve"]["-1.0"]
    replication = payload["sides"]["replication"][
        "exact_scored_alias_protected"]["threshold_curve"]["-1.0"]
    register(
        EVIDENCE_ID, tier=TIER, command=cmd,
        what=(
            "Post-freeze Qwen protected-answer protocol audit: "
            f"confirmatory exact-alias estimate {primary['estimate']:+.4f}, "
            f"p={primary['p_plus_one']:.3g}; replication "
            f"{replication['estimate']:+.4f}, "
            f"p={replication['p_plus_one']:.3g}"
        ),
        outputs=[result_path, rank_path, joined_path],
        inputs=inputs,
    )
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
