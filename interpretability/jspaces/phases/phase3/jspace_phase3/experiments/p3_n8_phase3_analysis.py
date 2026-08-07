"""N8-P3-L1 expected-value-blind analysis reproduction.

This module intentionally does not import Phase 3 estimator modules. It reads
only the input paths allowed by protocol/N8_PHASE3_REPRO_PROTOCOL.md and seals
its own report without consulting campaign results.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

SLUGS = ("olmo31-think", "olmo31-instruct", "qwen36-27b")
SIDES = ("confirmatory", "replication")
SEED = 4242
DRAWS = 100_000


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_signflip(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    m = len(values)
    bits = np.arange(2**m, dtype=np.uint32)[:, None]
    signs = (
        1 - 2 * ((bits >> np.arange(m, dtype=np.uint32)) & 1)
    ).astype(np.int8)
    null = signs @ values / m
    observed = float(values.mean())
    extreme = np.abs(null) >= abs(observed) - 1e-15
    return {
        "estimate": observed, "p": float(extreme.mean()),
        "extreme_patterns": int(extreme.sum()),
        "n_patterns": int(len(null)), "n_families": int(m),
    }


def family_weighted_mc(values: np.ndarray, families: np.ndarray,
                       *, alternative: str = "greater") -> dict:
    values = np.asarray(values, dtype=float)
    categorical = pd.Categorical(families)
    codes = categorical.codes
    m = len(categorical.categories)
    counts = np.bincount(codes, minlength=m).astype(float)
    weights = 1.0 / (m * counts[codes])
    weighted = values * weights
    observed = float(weighted.sum())
    rng = np.random.default_rng(SEED)
    null = np.empty(DRAWS)
    for start in range(0, DRAWS, 5000):
        n = min(5000, DRAWS - start)
        signs = rng.choice((-1, 1), size=(n, len(values))).astype(np.int8)
        null[start:start + n] = signs @ weighted
    if alternative == "greater":
        extreme = np.count_nonzero(null >= observed - 1e-15)
    else:
        extreme = np.count_nonzero(
            np.abs(null) >= abs(observed) - 1e-15)
    return {
        "estimate": observed,
        "p_plus_one": float((int(extreme) + 1) / (DRAWS + 1)),
        "n_items": int(len(values)), "n_families": int(m),
        "n_randomizations": DRAWS,
    }


def family_bootstrap(frame: pd.DataFrame, value: str,
                     *, draws: int = 4000) -> dict:
    family = frame.groupby(
        "canonical_family", sort=True)[value].mean()
    names = family.index.to_numpy()
    rng = np.random.default_rng(SEED)
    samples = np.empty(draws)
    for draw in range(draws):
        picked = rng.choice(names, size=len(names), replace=True)
        samples[draw] = family.loc[picked].mean()
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {
        "estimate": float(family.mean()),
        "ci95": [float(lo), float(hi)],
        "n_families": int(len(family)),
    }


def input_path(root: Path, slug: str, side: str) -> Path:
    suffix = "" if side == "confirmatory" else "_replication"
    return (
        root / "metrics" / slug / f"p3_grid{suffix}"
        / f"p3_grid{suffix}_{slug}.parquet"
    )


def load_effects(root: Path, side: str, input_hashes: dict,
                 allowed_families: set[str]) -> pd.DataFrame:
    rows = []
    for slug in SLUGS:
        path = input_path(root, slug, side)
        input_hashes[str(path)] = sha256_file(path)
        frame = pd.read_parquet(path)
        observed_families = set(frame["canonical_family"].unique())
        unexpected = observed_families - allowed_families
        if unexpected:
            raise RuntimeError(
                f"{path} contains families outside the frozen {side} "
                f"partition: {sorted(unexpected)}")
        frame["model"] = slug
        frame["J_eff"] = (
            frame["lp_meanJ_span_safe"] - frame["lp_baseline"])
        frame["C_eff"] = frame["lp_ss_matched"] - frame["lp_baseline"]
        frame["specific"] = frame["J_eff"] - frame["C_eff"]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def compositions(effects: pd.DataFrame) -> pd.DataFrame:
    pivot = effects.pivot_table(
        index=["fact_id", "canonical_family", "model", "bank"],
        columns="variant", values="specific", aggfunc="first").reset_index()
    pivot = pivot.dropna(subset=["direct", "composed"])
    pivot["composition"] = pivot["composed"] - pivot["direct"]
    return pivot


def p3p1(comp: pd.DataFrame, effects: pd.DataFrame) -> dict:
    pivot = comp.pivot_table(
        index=["fact_id", "canonical_family"],
        columns="model", values="composition", aggfunc="first")
    pivot = pivot.dropna(subset=list(SLUGS))
    diff = (
        pivot["qwen36-27b"]
        - 0.5 * (pivot["olmo31-think"] + pivot["olmo31-instruct"])
    ).rename("diff").reset_index()
    family = diff.groupby("canonical_family", sort=True)["diff"].mean()
    relation = effects[["fact_id", "relation_group"]].drop_duplicates(
        "fact_id")
    relation_diff = diff.merge(relation, on="fact_id", how="left")
    return {
        "exact_randomization": exact_signflip(family.to_numpy()),
        "item_weighted": float(diff["diff"].mean()),
        "relation_group_weighted": float(
            relation_diff.groupby("relation_group")["diff"].mean().mean()),
        "median_family_mean": float(family.median()),
    }


def p3p2_views(effects: pd.DataFrame, ranks: pd.DataFrame,
               side: str) -> dict:
    qwen = effects[effects["model"] == "qwen36-27b"].merge(
        ranks[ranks["side"] == side][[
            "item_id", "rank_exact_scored_alias",
            "rank_min_accepted_alias"]],
        on="item_id", validate="one_to_one")
    views = {
        "all_items": qwen,
        "exact_scored_alias_protected": qwen[
            qwen["rank_exact_scored_alias"] <= 10],
        "any_accepted_alias_protected": qwen[
            qwen["rank_min_accepted_alias"] <= 10],
    }
    out = {}
    for name, frame in views.items():
        difference = (
            (frame["J_eff"] < -1.0).astype(float)
            - (frame["C_eff"] < -1.0).astype(float)
        ).to_numpy()
        out[name] = family_weighted_mc(
            difference, frame["canonical_family"].to_numpy())
    return out


def p3p3(effects: pd.DataFrame) -> dict | None:
    if "lp_true_bridge" not in effects:
        return None
    frame = effects[
        (effects["model"] == "qwen36-27b")
        & effects["lp_true_bridge"].notna()
    ]
    if frame.empty:
        return None
    difference = (
        frame["lp_true_bridge"] - frame["lp_distractor_bridge"]
    ).to_numpy()
    return family_weighted_mc(
        difference, frame["canonical_family"].to_numpy())


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    order = sorted(pvalues, key=pvalues.get)
    out, previous = {}, 0.0
    for index, key in enumerate(order):
        adjusted = min(max(
            pvalues[key] * (len(order) - index), previous), 1.0)
        out[key] = float(adjusted)
        previous = adjusted
    return out


def estimation_targets(comp: pd.DataFrame) -> dict:
    pivot = comp.pivot_table(
        index=["fact_id", "canonical_family"],
        columns="model", values="composition", aggfunc="first")
    thick = pivot.dropna(
        subset=["olmo31-think", "olmo31-instruct"]).reset_index()
    thick["value"] = (
        thick["olmo31-think"] - thick["olmo31-instruct"])
    bank_s = {}
    for slug in SLUGS:
        frame = comp[(comp["model"] == slug) & (comp["bank"] == "S")].copy()
        frame["value"] = frame["composition"]
        bank_s[slug] = family_bootstrap(frame, "value")
    return {
        "think_minus_instruct_thick": family_bootstrap(thick, "value"),
        "bank_s_composition_by_model": bank_s,
    }


def analyze_side(root: Path, ranks: pd.DataFrame, side: str,
                 input_hashes: dict,
                 allowed_families: set[str]) -> dict:
    effects = load_effects(
        root, side, input_hashes, allowed_families)
    comp = compositions(effects)
    p1 = p3p1(comp, effects)
    p2 = p3p2_views(effects, ranks, side)
    p3 = p3p3(effects)
    pvalues = {
        "P3-P1": p1["exact_randomization"]["p"],
        "P3-P2": p2["all_items"]["p_plus_one"],
    }
    if p3 is not None:
        pvalues["P3-P3"] = p3["p_plus_one"]
    return {
        "P3-P1": p1, "P3-P2": p2, "P3-P3": p3,
        "holm": holm(pvalues),
        "estimation_targets": estimation_targets(comp),
    }


def main() -> None:
    run_in = os.environ.get("N8_P3_RUN_IN")
    out_env = os.environ.get("N8_P3_OUT")
    if not run_in or not out_env:
        raise RuntimeError("set N8_P3_RUN_IN and N8_P3_OUT")
    root, out = Path(run_in), Path(out_env)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"N8_P3_OUT must be fresh and empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    input_hashes = {}
    partition = root / "preregistration" / "partition_phase3.json"
    input_hashes[str(partition)] = sha256_file(partition)
    partition_payload = json.loads(partition.read_text())["payload"]
    confirmatory_families = set(partition_payload["confirmatory"])
    replication_families = set(partition_payload["replication"])
    overlap = confirmatory_families & replication_families
    if overlap:
        raise RuntimeError(
            f"frozen partitions overlap: {sorted(overlap)}")
    allowed_by_side = {
        "confirmatory": confirmatory_families,
        "replication": replication_families,
    }
    rank_path = (
        root / "metrics" / "qwen36-27b" / "release_audit"
        / "protected_answer" / "p3_protected_answer_ranks_qwen.parquet"
    )
    input_hashes[str(rank_path)] = sha256_file(rank_path)
    ranks = pd.read_parquet(rank_path)
    results = {
        side: analyze_side(
            root, ranks, side, input_hashes, allowed_by_side[side])
        for side in SIDES}
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
    report = {
        "schema_version": 1,
        "runner": "n8-p3-l1-isolated-subprocess-v1",
        "sealed_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_commit": commit,
        "input_sha256": input_hashes,
        "analysis_contract": {
            "seed": SEED, "draws": DRAWS,
            "protected_k": 10,
            "expected_values_available_to_runner": False,
        },
        "results": results,
        "anomalies": [],
    }
    report["payload_sha256"] = hashlib.sha256(
        canonical_json(results).encode("utf-8")).hexdigest()
    json_path = out / "N8_P3_L1_REPORT.json"
    md_path = out / "N8_P3_L1_REPORT.md"
    json_path.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    lines = [
        "# N8-P3-L1 sealed reproduction report",
        "",
        "This expected-value-blind process read only the protocol-listed "
        "raw inputs. It did not compare against campaign results.",
        "",
    ]
    for side in SIDES:
        result = results[side]
        lines += [
            f"## {side}",
            "",
            f"- P3-P1: {result['P3-P1']['exact_randomization']['estimate']:+.6f}; "
            f"exact p={result['P3-P1']['exact_randomization']['p']:.8f}.",
            f"- P3-P2 all items: {result['P3-P2']['all_items']['estimate']:+.6f}; "
            f"plus-one p={result['P3-P2']['all_items']['p_plus_one']:.8g}.",
            f"- P3-P2 exact protected: "
            f"{result['P3-P2']['exact_scored_alias_protected']['estimate']:+.6f}.",
            f"- P3-P3: {result['P3-P3']}.",
            f"- Holm adjusted p-values: {result['holm']}.",
            f"- Estimation targets: {result['estimation_targets']}.",
            "",
        ]
    md_path.write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "report": str(json_path), "markdown": str(md_path),
        "payload_sha256": report["payload_sha256"]}, indent=1))


if __name__ == "__main__":
    main()
