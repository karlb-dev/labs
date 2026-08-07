"""Outcome-blind fixed subset for the Phase 4 Qwen multi-lens gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from ..manifests import (
    atomic_json,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths4 import run_root, resolve_uri
from ..provenance4 import Provenance4, write_result4
from ..registry4 import RegistryError, create, resolve


ALLOWED_GRID_COLUMNS = {
    "item_id", "fact_id", "variant", "bank", "canonical_family",
    "relation_group",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def stable_order(values: Iterable[str], *, namespace: str) -> list[str]:
    unique = set(values)
    return sorted(
        unique,
        key=lambda value: (
            hashlib.sha256(f"{namespace}:{value}".encode()).digest(), value),
    )


def family_balanced_facts(metadata: pd.DataFrame, *, bank: str, n: int,
                          namespace: str) -> list[str]:
    subset = metadata[metadata["bank"] == bank]
    facts_by_family = {
        family: stable_order(group["fact_id"].unique(),
                             namespace=f"{namespace}:{bank}:{family}:facts")
        for family, group in subset.groupby("canonical_family", sort=True)
    }
    family_order = stable_order(
        facts_by_family, namespace=f"{namespace}:{bank}:families")
    selected: list[str] = []
    depth = 0
    while len(selected) < n:
        added = False
        for family in family_order:
            candidates = facts_by_family[family]
            if depth < len(candidates):
                selected.append(candidates[depth])
                added = True
                if len(selected) == n:
                    break
        if not added:
            raise RuntimeError(f"bank {bank} has fewer than {n} eligible facts")
        depth += 1
    return selected


def balanced_records(rows: list[dict], *, group_key: str, n: int,
                     namespace: str) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault(str(row[group_key]), []).append(row)
    for key in groups:
        groups[key] = sorted(
            groups[key],
            key=lambda row: hashlib.sha256(
                f"{namespace}:{key}:{row['item_id']}".encode()).digest(),
        )
    group_order = stable_order(groups, namespace=f"{namespace}:groups")
    selected = []
    depth = 0
    while len(selected) < n:
        added = False
        for key in group_order:
            if depth < len(groups[key]):
                selected.append(groups[key][depth])
                added = True
                if len(selected) == n:
                    break
        if not added:
            raise RuntimeError(f"fewer than {n} eligible records")
        depth += 1
    return selected


def python_counts(values: Iterable[str]) -> dict[str, int]:
    """Return JSON-safe deterministic counts (never NumPy scalar values)."""
    counts = pd.Series(list(values), dtype="object").value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def select_functional_subset(
        grid: pd.DataFrame, guard_rows: list[dict], g4_rows: list[dict], *,
        specification: dict) -> dict:
    missing = ALLOWED_GRID_COLUMNS - set(grid.columns)
    if missing:
        raise RuntimeError(f"grid lacks metadata columns: {sorted(missing)}")
    if specification.get("outcome_columns_allowed") is not False:
        raise RuntimeError("functional subset must forbid outcome columns")
    if specification.get("consumed_phase3_only") is not True:
        raise RuntimeError("functional subset must use consumed Phase 3 rows only")
    metadata = grid[sorted(ALLOWED_GRID_COLUMNS)].drop_duplicates()
    counts = metadata.groupby("fact_id")["variant"].agg(
        lambda values: sorted(set(values)))
    complete_facts = set(counts[counts.apply(
        lambda values: values == ["composed", "direct"])].index)
    fact_rows = (
        metadata[metadata["fact_id"].isin(complete_facts)]
        .drop_duplicates("fact_id")
    )
    namespace = specification["namespace"]
    per_bank = int(specification["primary_facts_per_bank"])
    selected_facts = []
    for bank in ("F", "S"):
        selected_facts.extend(family_balanced_facts(
            fact_rows, bank=bank, n=per_bank, namespace=namespace))
    selected_fact_rows = fact_rows[
        fact_rows["fact_id"].isin(selected_facts)]
    primary_families = int(selected_fact_rows["canonical_family"].nunique())
    if primary_families < int(specification["required_primary_families"]):
        raise RuntimeError("primary subset has too few canonical families")

    bridge_per_bank = int(specification["bridge_facts_per_bank"])
    bridge_facts = []
    for bank in ("F", "S"):
        bridge_facts.extend(family_balanced_facts(
            selected_fact_rows, bank=bank, n=bridge_per_bank,
            namespace=f"{namespace}:bridge"))
    bridge_rows = selected_fact_rows[
        selected_fact_rows["fact_id"].isin(bridge_facts)]
    bridge_families = int(bridge_rows["canonical_family"].nunique())
    if bridge_families < int(specification["required_bridge_families"]):
        raise RuntimeError("bridge subset has too few canonical families")

    non_grammar = [row for row in guard_rows
                   if row["domain"] != "grammar_pairs"]
    prose = balanced_records(
        non_grammar,
        group_key="domain",
        n=int(specification["prose_items_n"]),
        namespace=f"{namespace}:prose",
    )
    capacity = balanced_records(
        prose,
        group_key="domain",
        n=int(specification["capacity_items_n"]),
        namespace=f"{namespace}:capacity",
    )
    g4_total = int(specification["g4_total_n"])
    if len(g4_rows) < g4_total:
        raise RuntimeError("released G4 source has too few items")
    g4 = g4_rows[:g4_total]

    primary_item_ids = sorted(
        metadata[metadata["fact_id"].isin(selected_facts)]["item_id"].tolist())
    if len(primary_item_ids) != 2 * len(selected_facts):
        raise RuntimeError("primary subset lacks exact fact pairs")
    return {
        "primary": {
            "fact_ids": sorted(selected_facts),
            "item_ids": primary_item_ids,
            "n_facts": len(selected_facts),
            "n_items": len(primary_item_ids),
            "n_families": primary_families,
            "banks": python_counts(selected_fact_rows["bank"]),
        },
        "bridge": {
            "fact_ids": sorted(bridge_facts),
            "item_ids": sorted(
                f"{fact_id}#composed" for fact_id in bridge_facts),
            "n_facts": len(bridge_facts),
            "n_families": bridge_families,
            "banks": python_counts(bridge_rows["bank"]),
        },
        "prose_nll": {
            "items": prose,
            "item_ids": [row["item_id"] for row in prose],
            "n_items": len(prose),
            "domains": python_counts(row["domain"] for row in prose),
        },
        "capacity": {
            "items": capacity,
            "item_ids": [row["item_id"] for row in capacity],
            "n_items": len(capacity),
            "positions": [int(value) for value in
                          specification["capacity_positions"]],
            "max_seq_len": int(specification["max_seq_len"]),
            "domains": python_counts(row["domain"] for row in capacity),
        },
        "g4": {
            "items": g4,
            "item_names": [row["name"] for row in g4],
            "calibration_n": int(specification["g4_calibration_n"]),
            "total_n": len(g4),
        },
    }


def main() -> None:
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    clean = require_clean_tree()
    try:
        existing = resolve(config["evidence_id"])
    except RegistryError as error:
        if "found 0" not in str(error):
            raise
    else:
        if not existing["live"]:
            raise RuntimeError("existing functional subset is not live")
        for output in existing["outputs"]:
            if file_sha256(output["path"]) != output["sha256"]:
                raise RuntimeError("existing functional-subset output mismatch")
        print(json.dumps({
            "status": "already-registered-and-verified",
            "evidence_id": config["evidence_id"],
        }, indent=1))
        return

    grid_path = resolve_uri(config["phase3_grid"]["uri"])
    guard_path = resolve_uri(config["guard_battery"]["uri"])
    g4_path = Path(config["g4_probe_swap"]["path"])
    for path, expected, label in (
        (grid_path, config["phase3_grid"]["sha256"], "Phase 3 grid"),
        (guard_path, config["guard_battery"]["sha256"], "guard battery"),
        (g4_path, config["g4_probe_swap"]["sha256"], "G4 probe swap"),
    ):
        if file_sha256(path) != expected:
            raise RuntimeError(f"{label} hash mismatch")
    grid = pd.read_parquet(grid_path)
    guard_rows = [json.loads(line) for line in guard_path.read_text().splitlines()
                  if line.strip()]
    g4_rows = json.loads(g4_path.read_text())["items"]
    selection = {
        **dict(config["selection"]),
        "g4_calibration_n": int(config["g4_probe_swap"]["calibration_n"]),
        "g4_total_n": int(config["g4_probe_swap"]["total_n"]),
    }
    subset = select_functional_subset(
        grid, guard_rows, g4_rows, specification=selection)
    payload = {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "tier": config["tier"],
        "selection_is_outcome_blind": True,
        "selection_uses_only_consumed_phase3_families": True,
        "selection_contract": selection,
        "subset": subset,
        "hashes": {
            "primary_fact_ids": object_sha256(subset["primary"]["fact_ids"]),
            "primary_item_ids": object_sha256(subset["primary"]["item_ids"]),
            "bridge_fact_ids": object_sha256(subset["bridge"]["fact_ids"]),
            "prose_item_ids": object_sha256(subset["prose_nll"]["item_ids"]),
            "capacity_item_ids": object_sha256(subset["capacity"]["item_ids"]),
            "g4_item_names": object_sha256(subset["g4"]["item_names"]),
        },
        "sources": {
            "phase3_grid": dict(config["phase3_grid"]),
            "guard_battery": dict(config["guard_battery"]),
            "g4_probe_swap": dict(config["g4_probe_swap"]),
        },
        "code_commit": clean["code_commit"],
        "config_sha256": file_sha256(config_path),
        "producer_sha256": file_sha256(Path(__file__)),
    }
    output_dir = (
        run_root() / "config" / "qwen_multilens_functional_subset"
        / config["evidence_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "subset_manifest.json"
    result_path = output_dir / "subset_result.json"
    envelope = {
        "schema_version": 1,
        "payload": payload,
        "payload_sha256": object_sha256(payload),
    }
    atomic_json(manifest_path, envelope)
    result_payload = {
        "selection_is_outcome_blind": True,
        "primary": {key: value for key, value in subset["primary"].items()
                    if key not in {"fact_ids", "item_ids"}},
        "bridge": {key: value for key, value in subset["bridge"].items()
                   if key not in {"fact_ids", "item_ids"}},
        "prose_nll": {
            "n_items": subset["prose_nll"]["n_items"],
            "domains": subset["prose_nll"]["domains"],
        },
        "capacity": {
            "n_items": subset["capacity"]["n_items"],
            "positions": subset["capacity"]["positions"],
            "domains": subset["capacity"]["domains"],
        },
        "g4": {
            "calibration_n": subset["g4"]["calibration_n"],
            "total_n": subset["g4"]["total_n"],
        },
        "subset_manifest_payload_sha256": envelope["payload_sha256"],
    }
    command = (
        "python -m jspace_phase4.experiments.p4_qwen_functional_subset "
        f"--config {arguments.config}")
    inputs = {
        "phase3_grid": config["phase3_grid"]["sha256"],
        "guard_battery": config["guard_battery"]["sha256"],
        "g4_probe_swap": config["g4_probe_swap"]["sha256"],
        "selection_contract": object_sha256(selection),
        "subset_manifest": envelope["payload_sha256"],
    }
    write_result4(
        result_payload,
        result_path,
        Provenance4(
            evidence_id=config["evidence_id"],
            tier=config["tier"],
            command=command,
            inputs=inputs,
            input_manifest_sha256=envelope["payload_sha256"],
            seed_contract=(
                "SHA-256 stable family-balanced order; metadata only; "
                "no lens or intervention outcomes"),
        ),
    )
    create(
        config["evidence_id"],
        tier=config["tier"],
        what=(
            "Outcome-blind fixed Qwen multi-lens functional subset: 30 paired "
            "facts, 20 bridge facts, 40 prose guards, 16 capacity records, "
            "and the exact 60-item released G4 set."),
        command=command,
        outputs=[manifest_path, result_path],
        inputs=inputs,
    )
    print(json.dumps({
        "evidence_id": config["evidence_id"],
        "manifest": str(manifest_path),
        "result": str(result_path),
        "summary": result_payload,
    }, indent=1))


if __name__ == "__main__":
    main()
