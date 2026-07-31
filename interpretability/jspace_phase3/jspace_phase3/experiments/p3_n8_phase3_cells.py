"""N8-P3-L2/L3 clean-root Phase 3 model-cell reproduction.

The wrapper relaunches the repaired Phase 3 producer in a fresh, durable run
root. The producer itself performs the same-process CUDA gate before loading
weights and never registers the derived clean-room cell.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from jspace_part2.lib import sha256_file
from jspace_part2.paths import resolve as resolve_uri

from ..paths3 import run_root
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           resolve_model, write_result3)
from ..seeds import SEED_CONTRACT, stable_seed
from .p3_n8_phase3_analysis import (compositions, holm, p3p1, p3p2_views,
                                    p3p3)

SLUGS = ("olmo31-think", "olmo31-instruct", "qwen36-27b")
REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PACKAGE_ROOT / "configs"
TIER = "methods"
TOLERANCE = 2e-3
CONTROL_NAMESPACE = "p3-control-seed-audit"
CONTROL_SEED = 31337


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def frozen_path(reference_root: Path, slug: str) -> Path:
    return (
        reference_root / "metrics" / slug / "p3_grid"
        / f"p3_grid_{slug}.parquet"
    )


def state_path(cell_root: Path, slug: str) -> Path:
    return cell_root / "metrics" / slug / "p3_grid" / "state.json"


def read_state(path: Path) -> dict:
    if not path.exists():
        return {"done": {}, "rows": []}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"corrupt checkpoint {path}: {exc}") from exc


def input_contract(reference_root: Path, source_config: Path,
                   cfg: dict, slug: str, level: int,
                   n_sentinel: int,
                   sentinel_item_ids: list[str]) -> dict:
    partition = REPO_ROOT / cfg["partition_uri"]
    lens = Path(resolve_uri(cfg["lens_uri"], must_exist=True))
    model = Path(resolve_uri(cfg["model_uri"], must_exist=True))
    g5 = (
        reference_root / "metrics" / slug / "g5_bank"
        / f"g5_bank_{slug}_regraded.parquet"
    )
    return {
        "schema_version": 1,
        "level": f"N8-P3-L{level}",
        "slug": slug,
        "n_sentinel": int(n_sentinel),
        "code_commit": git_head(),
        "source_config": str(source_config),
        "source_config_sha256": sha256_file(source_config),
        "partition_sha256": sha256_file(partition),
        "frozen_grid_sha256": sha256_file(
            frozen_path(reference_root, slug)),
        "g5_regraded_sha256": sha256_file(g5),
        "lens_sha256": sha256_file(lens),
        "model": resolve_model(str(model)),
        "seed_contract": SEED_CONTRACT,
        "matched_control_namespace": CONTROL_NAMESPACE,
        "matched_control_seed": CONTROL_SEED,
        "sentinel_item_ids": sentinel_item_ids,
    }


def select_sentinel_from_frame(frame: pd.DataFrame, n: int) -> list[str]:
    if n < 20 or n % 2:
        raise ValueError("sentinel size must be even and at least 20")
    pairs = {}
    for fact_id, group in frame.groupby("fact_id"):
        if set(group["variant"]) == {"direct", "composed"}:
            pairs[str(fact_id)] = group
    by_family: dict[str, list[str]] = {}
    for fact_id, group in pairs.items():
        family = str(group["canonical_family"].iloc[0])
        by_family.setdefault(family, []).append(fact_id)
    for family, facts in by_family.items():
        facts.sort(key=lambda fact: stable_seed(
            "n8-p3-l2-fact", fact, CONTROL_SEED))
    families = sorted(by_family, key=lambda family: stable_seed(
        "n8-p3-l2-family", family, CONTROL_SEED))
    needed_facts = n // 2
    chosen: list[str] = []
    depth = 0
    while len(chosen) < needed_facts:
        progressed = False
        for family in families:
            facts = by_family[family]
            if depth < len(facts):
                chosen.append(facts[depth])
                progressed = True
                if len(chosen) == needed_facts:
                    break
        if not progressed:
            break
        depth += 1
    if len(chosen) != needed_facts:
        raise RuntimeError(
            f"only {len(chosen)} complete fact pairs for {n} items")
    chosen_set = set(chosen)
    selected = frame[frame["fact_id"].astype(str).isin(chosen_set)]
    item_ids = sorted(str(value) for value in selected["item_id"])
    if len(item_ids) != n:
        raise RuntimeError(
            f"sentinel selection yielded {len(item_ids)} != {n} items")
    return item_ids


def sentinel_item_ids(reference_root: Path, slug: str,
                      level: int, n: int) -> list[str]:
    if level == 3:
        return []
    frame = pd.read_parquet(
        frozen_path(reference_root, slug),
        columns=[
            "item_id", "fact_id", "variant", "canonical_family",
        ],
    )
    return select_sentinel_from_frame(frame, n)


def prepare_root(cell_root: Path, reference_root: Path, slug: str,
                 level: int, n_sentinel: int) -> tuple[Path, dict]:
    source_config = CONFIG_ROOT / f"p3_grid_{slug}.yaml"
    cfg = yaml.safe_load(source_config.read_text())
    selected_ids = sentinel_item_ids(
        reference_root, slug, level, n_sentinel)
    contract = input_contract(
        reference_root, source_config, cfg, slug, level, n_sentinel,
        selected_ids)
    manifest_path = cell_root / "N8_P3_CELL_MANIFEST.json"
    if manifest_path.exists():
        observed = json.loads(manifest_path.read_text())
        if observed.get("contract") != contract:
            raise RuntimeError(
                "N8 resume manifest mismatch; use a new output root")
    else:
        if cell_root.exists() and any(cell_root.iterdir()):
            raise RuntimeError(
                f"nonempty N8 root lacks manifest: {cell_root}")
        cell_root.mkdir(parents=True, exist_ok=True)
        atomic_json(manifest_path, {
            "contract": contract,
            "created_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    fresh_g5 = cell_root / "metrics" / slug / "g5_bank"
    if not fresh_g5.exists():
        frozen_g5 = reference_root / "metrics" / slug / "g5_bank"
        fresh_g5.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(frozen_g5, fresh_g5)
    derived = dict(cfg)
    derived |= {
        "evidence_id": f"n8-p3-l{level}-{slug}-producer",
        "tier": TIER,
        "partition_side": "confirmatory",
        "rand_seed": CONTROL_SEED,
        "matched_seed_namespaces": {
            "instant_rank_energy_matched": CONTROL_NAMESPACE,
        },
    }
    if selected_ids:
        derived["item_ids"] = selected_ids
    derived_path = cell_root / "n8_p3_config.yaml"
    rendered = yaml.safe_dump(derived, sort_keys=False)
    if derived_path.exists() and derived_path.read_text() != rendered:
        raise RuntimeError("derived N8 config changed across resume")
    derived_path.write_text(rendered)
    return derived_path, contract


def append_heartbeat(path: Path, slug: str, level: int,
                     done: int, target: int) -> None:
    with open(path, "a") as stream:
        stream.write(
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            + f" N8-P3-L{level} {slug} {done}/{target}\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_producer(cell_root: Path, derived_config: Path, slug: str,
                 level: int, target: int,
                 timeout_hours: float) -> None:
    checkpoint = state_path(cell_root, slug)
    if len(read_state(checkpoint)["done"]) >= target:
        return
    env = dict(os.environ)
    env["JSPACE3_RUN_ROOT"] = str(cell_root)
    env["HF_HUB_CACHE"] = os.environ.get(
        "HF_HUB_CACHE", "/content/hf_local")
    env["CUDA_VISIBLE_DEVICES"] = "0"
    command = [
        sys.executable, "-m",
        "jspace_phase3.experiments.phase3_primary_grid",
        "--config", str(derived_config), "--no-register",
    ]
    log_path = cell_root / "N8_P3_PRODUCER.log"
    heartbeat = cell_root / "WATCHDOG_HEARTBEAT.log"
    timed_out = False
    with open(log_path, "a") as log:
        log.write(
            "\n" + time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            + " LAUNCH " + " ".join(command) + "\n")
        log.flush()
        proc = subprocess.Popen(
            command, cwd=REPO_ROOT, env=env, stdout=log,
            stderr=subprocess.STDOUT, text=True)
        started = time.time()
        while proc.poll() is None:
            time.sleep(10)
            done = len(read_state(checkpoint)["done"])
            append_heartbeat(heartbeat, slug, level, done, target)
            if level == 2 and done >= target:
                proc.send_signal(signal.SIGTERM)
                break
            if time.time() - started > timeout_hours * 3600:
                proc.send_signal(signal.SIGTERM)
                timed_out = True
                break
        try:
            return_code = proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            proc.kill()
            return_code = proc.wait()
    done = len(read_state(checkpoint)["done"])
    if timed_out:
        raise RuntimeError(
            f"N8-P3-L{level} {slug} exceeded "
            f"{timeout_hours:g} hours at {done}/{target}")
    if done < target:
        tail = "\n".join(log_path.read_text().splitlines()[-30:])
        raise RuntimeError(
            f"producer stopped at {done}/{target} with code "
            f"{return_code}\n{tail}")
    if level == 3 and return_code != 0:
        tail = "\n".join(log_path.read_text().splitlines()[-30:])
        raise RuntimeError(
            f"full producer exited {return_code}\n{tail}")


def deviation_block(merged: pd.DataFrame, columns: list[str]) -> dict:
    rows = {}
    for column in columns:
        new, frozen = f"{column}_new", f"{column}_frozen"
        if new not in merged or frozen not in merged:
            continue
        pair = merged[[new, frozen]].dropna()
        if pair.empty:
            continue
        error = (pair[new] - pair[frozen]).abs()
        rows[column] = {
            "n": int(len(pair)),
            "max_abs_error": float(error.max()),
            "mean_abs_error": float(error.mean()),
        }
    return rows


def family_tail_effect(frame: pd.DataFrame, control: str) -> float:
    delta_j = frame["lp_meanJ_span_safe"] - frame["lp_baseline"]
    delta_c = frame[control] - frame["lp_baseline"]
    value = (delta_j < -1.0).astype(float) - (
        delta_c < -1.0).astype(float)
    return float(value.groupby(frame["canonical_family"]).mean().mean())


def matched_control_block(reference_root: Path, slug: str,
                          new: pd.DataFrame,
                          frozen: pd.DataFrame) -> dict:
    merged = new.merge(
        frozen[["item_id", "lp_ss_matched"]],
        on="item_id", suffixes=("_new", "_historical"),
        validate="one_to_one")
    movement = (
        merged["lp_ss_matched_new"]
        - merged["lp_ss_matched_historical"])
    block = {
        "historical_seed_limitation": (
            "The frozen Python-hash realization is unrecoverable; it is "
            "not an exact-reproduction target."
        ),
        "stable_seed_contract": SEED_CONTRACT,
        "namespace": CONTROL_NAMESPACE,
        "base_seed": CONTROL_SEED,
        "new_minus_historical": {
            "mean": float(movement.mean()),
            "median": float(movement.median()),
            "max_abs": float(movement.abs().max()),
            "correlation": float(merged[
                ["lp_ss_matched_new", "lp_ss_matched_historical"]
            ].corr().iloc[0, 1]),
        },
        "P3-P2_at_minus_1": {
            "new": family_tail_effect(new, "lp_ss_matched"),
            "historical": family_tail_effect(
                frozen[frozen["item_id"].isin(set(new["item_id"]))],
                "lp_ss_matched"),
        },
        "seed_ensemble": {"available": False},
    }
    if slug != "qwen36-27b":
        return block
    audit_path = (
        reference_root / "metrics" / slug / "release_audit"
        / "control_seed" / "p3_control_seed_audit_full_rows.parquet"
    )
    audit = pd.read_parquet(audit_path)
    audit = audit[
        (audit["side"] == "confirmatory")
        & audit["item_id"].isin(set(new["item_id"]))
    ]
    seed = audit[audit["audit_seed"] == CONTROL_SEED][
        ["item_id", "lp_ss_matched"]]
    exact = new[["item_id", "lp_ss_matched"]].merge(
        seed, on="item_id", suffixes=("_new", "_audit"),
        validate="one_to_one")
    by_item = audit.groupby("item_id")["lp_ss_matched"].agg(
        ["min", "max"]).reset_index()
    coverage = new[["item_id", "lp_ss_matched"]].merge(
        by_item, on="item_id", validate="one_to_one")
    inside = (
        (coverage["lp_ss_matched"] >= coverage["min"] - TOLERANCE)
        & (coverage["lp_ss_matched"] <= coverage["max"] + TOLERANCE)
    )
    block["seed_ensemble"] = {
        "available": True,
        "source_sha256": sha256_file(audit_path),
        "seeds": sorted(int(x) for x in audit["audit_seed"].unique()),
        "stable_seed_exact_max_abs_error": float(
            (exact["lp_ss_matched_new"]
             - exact["lp_ss_matched_audit"]).abs().max()),
        "within_five_seed_item_envelope_fraction": float(inside.mean()),
    }
    return block


def bridge_geometry_block(frame: pd.DataFrame) -> dict | None:
    columns = [
        "true_bridge_piece_count",
        "distractor_bridge_piece_count",
        "true_bridge_clean_topk_overlap_mean",
        "distractor_bridge_clean_topk_overlap_mean",
        "true_bridge_added_rank_mean",
        "distractor_bridge_added_rank_mean",
    ]
    if not set(columns).issubset(frame.columns):
        return None
    bridge = frame.dropna(subset=columns).copy()
    if bridge.empty:
        return None
    return {
        "scope": (
            "surface protection-set geometry reproduced in the cell; "
            "the separate per-layer bridge-geometry audit remains the "
            "release source for causal geometry matching"
        ),
        "n_items": int(len(bridge)),
        "piece_count_difference_mean": float(
            (bridge["true_bridge_piece_count"]
             - bridge["distractor_bridge_piece_count"]).mean()),
        "piece_count_exact_match_fraction": float(
            (bridge["true_bridge_piece_count"]
             == bridge["distractor_bridge_piece_count"]).mean()),
        "clean_topk_overlap_difference_mean": float(
            (bridge["true_bridge_clean_topk_overlap_mean"]
             - bridge[
                 "distractor_bridge_clean_topk_overlap_mean"]).mean()),
        "added_rank_difference_mean": float(
            (bridge["true_bridge_added_rank_mean"]
             - bridge["distractor_bridge_added_rank_mean"]).mean()),
    }


def recompute_locked(reference_root: Path, new_qwen: pd.DataFrame) -> dict:
    frames = []
    for slug in ("olmo31-think", "olmo31-instruct"):
        frame = pd.read_parquet(frozen_path(reference_root, slug))
        frame["model"] = slug
        frames.append(frame)
    qwen = new_qwen.copy()
    qwen["model"] = "qwen36-27b"
    frames.append(qwen)
    effects = pd.concat(frames, ignore_index=True)
    effects["J_eff"] = (
        effects["lp_meanJ_span_safe"] - effects["lp_baseline"])
    effects["C_eff"] = effects["lp_ss_matched"] - effects["lp_baseline"]
    effects["specific"] = effects["J_eff"] - effects["C_eff"]
    comp = compositions(effects)
    p1 = p3p1(comp, effects)
    rank_path = (
        reference_root / "metrics" / "qwen36-27b"
        / "release_audit" / "protected_answer"
        / "p3_protected_answer_ranks_qwen.parquet"
    )
    ranks = pd.read_parquet(rank_path)
    p2 = p3p2_views(effects, ranks, "confirmatory")
    p3 = p3p3(effects)
    adjusted = holm({
        "P3-P1": p1["exact_randomization"]["p"],
        "P3-P2": p2["all_items"]["p_plus_one"],
        "P3-P3": p3["p_plus_one"],
    })
    return {
        "P3-P1": p1,
        "P3-P2": p2,
        "P3-P3": p3,
        "holm": adjusted,
        "metadata": {
            "reproduction_tier": TIER,
            "source_outcome_tier": "phase3-confirmatory",
            "matched_control_seed_contract": SEED_CONTRACT,
            "matched_control_namespace": CONTROL_NAMESPACE,
            "matched_control_seed": CONTROL_SEED,
        },
    }


def comparison_payload(reference_root: Path, slug: str, level: int,
                       new: pd.DataFrame,
                       contract: dict, gpu: dict) -> dict:
    frozen = pd.read_parquet(frozen_path(reference_root, slug))
    frozen = frozen[frozen["item_id"].isin(set(new["item_id"]))].copy()
    merged = new.merge(
        frozen, on="item_id", suffixes=("_new", "_frozen"),
        validate="one_to_one")
    required = [
        "lp_baseline", "lp_meanJ_span_safe",
        "lp_meanJ_label_protected",
    ]
    if slug == "qwen36-27b":
        required += ["lp_true_bridge", "lp_distractor_bridge"]
    deviations = deviation_block(merged, required)
    missing = [column for column in required if column not in deviations]
    worst = max(
        (row["max_abs_error"] for row in deviations.values()),
        default=float("inf"))
    matched = matched_control_block(
        reference_root, slug, new, frozen)
    seed_exact = matched["seed_ensemble"].get(
        "stable_seed_exact_max_abs_error")
    passed = (
        not missing
        and worst <= TOLERANCE
        and (seed_exact is None or seed_exact <= TOLERANCE)
    )
    payload = {
        "level": f"N8-P3-L{level}",
        "slug": slug,
        "n_items": int(new["item_id"].nunique()),
        "n_families": int(new["canonical_family"].nunique()),
        "input_contract": contract,
        "gpu_gate": gpu,
        "deterministic_arm_deviations": deviations,
        "missing_required_comparisons": missing,
        "worst_deterministic_abs_error": worst,
        "tolerance": TOLERANCE,
        "matched_control": matched,
        "bridge_geometry": bridge_geometry_block(new),
        "pass": passed,
    }
    if level == 3:
        payload["locked_analysis_recomputed"] = recompute_locked(
            reference_root, new)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True, choices=SLUGS)
    parser.add_argument("--level", required=True, type=int, choices=(2, 3))
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--cell-root", required=True)
    parser.add_argument("--timeout-hours", type=float, default=8.0)
    parser.add_argument("--evidence-version", type=int, default=1)
    parser.add_argument("--supersedes")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.level == 3 and args.slug != "qwen36-27b":
        parser.error("N8-P3-L3 is preregistered for qwen36-27b")
    if args.n < 20:
        parser.error("N8-P3-L2 requires at least 20 items")
    if args.evidence_version < 1:
        parser.error("--evidence-version must be positive")
    return args


def main() -> None:
    args = parse_args()
    require_clean_tree(args.allow_dirty)
    reference_root = Path(os.environ.get(
        "N8_P3_REFERENCE_ROOT", str(run_root())))
    cell_root = Path(args.cell_root)
    frozen = pd.read_parquet(
        frozen_path(reference_root, args.slug),
        columns=["item_id"])
    target = len(frozen) if args.level == 3 else args.n
    derived, contract = prepare_root(
        cell_root, reference_root, args.slug, args.level, args.n)
    run_producer(
        cell_root, derived, args.slug, args.level, target,
        args.timeout_hours)
    state = read_state(state_path(cell_root, args.slug))
    rows = pd.DataFrame(state["rows"])
    if args.level == 2:
        selected = sorted(rows["item_id"].unique())[:args.n]
        rows = rows[rows["item_id"].isin(selected)].copy()
    else:
        rows = rows.copy()
    rows = rows.sort_values("item_id").reset_index(drop=True)
    raw_path = (
        cell_root
        / f"N8_P3_L{args.level}_{args.slug}_rows.parquet"
    )
    rows.to_parquet(raw_path, index=False)
    payload = comparison_payload(
        reference_root, args.slug, args.level, rows,
        contract, state.get("gpu", {}))
    report_path = cell_root / (
        f"N8_P3_L{args.level}_{args.slug}_REPORT_v"
        f"{args.evidence_version}.json"
    )
    eid = (
        f"p3-n8-p3-level{args.level}-{args.slug}"
        f"-v{args.evidence_version}"
    )
    command = (
        "python -m jspace_phase3.experiments.p3_n8_phase3_cells "
        f"--slug {args.slug} --level {args.level} --n {args.n} "
        f"--cell-root {cell_root} "
        f"--evidence-version {args.evidence_version}"
        + (
            f" --supersedes {args.supersedes}"
            if args.supersedes else ""
        )
    )
    write_result3(payload, report_path, Provenance3(
        evidence_id=eid, tier=TIER, command=command,
        config_path=str(derived),
        inputs={
            "frozen_grid": contract["frozen_grid_sha256"],
            "lens": contract["lens_sha256"],
        },
        model=contract["model"], seed=CONTROL_SEED))
    comparison_path = (
        reference_root / "metrics" / "cross_model" / "release_audit"
        / "n8_phase3"
        / (
            f"n8_p3_l{args.level}_{args.slug}"
            f"_v{args.evidence_version}.json"
        )
    )
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(report_path, comparison_path)
    register(
        eid, tier=TIER, command=command,
        what=(
            f"N8-P3-L{args.level} {args.slug}: "
            f"{len(rows)} Phase 3 items, worst deterministic "
            f"|error|={payload['worst_deterministic_abs_error']:.3g} "
            f"(tol {TOLERANCE}) — "
            f"{'PASS' if payload['pass'] else 'FAIL'}"
        ),
        outputs=[
            comparison_path, report_path, raw_path,
            cell_root / "N8_P3_CELL_MANIFEST.json",
            cell_root / "N8_P3_PRODUCER.log",
        ],
        inputs={
            "frozen_grid": contract["frozen_grid_sha256"],
            "lens": contract["lens_sha256"],
        },
        supersedes=args.supersedes,
    )
    print(json.dumps({
        "evidence_id": eid,
        "report": str(report_path),
        "n_items": len(rows),
        "worst_deterministic_abs_error":
            payload["worst_deterministic_abs_error"],
        "pass": payload["pass"],
    }, indent=1))
    if not payload["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
