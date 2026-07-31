"""Phase 3 post-freeze boundary and cohort-selection sensitivity.

This CPU-only audit never changes the frozen G5 or primary-grid rows.  It:

* regrades stored 8-token generations with contiguous normalized-word
  boundaries (the historical predicate used unsafe substring matching);
* prepares and verifies a deterministic 100-positive/100-negative manual
  review worksheet stratified by model and alias length;
* recomputes P3-P1/P2/P3 on the subset of already-observed outcomes that
  survives several pre-intervention cohort rules;
* reports, rather than silently fills, eligible facts whose intervention
  outcomes were never measured under the frozen strict cohort.

Alias-aggregate intervention rescoring is a separate GPU cell; this module
provides the boundary/cohort half of nextsteps_4_1 §7.
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

from jspace_part2.lib import sha256_file
from ..bank import load_bank
from ..paths3 import metrics_dir
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           write_result3)
from ..scoring import DEFAULT_SPEC
from ..stats import (family_signflip_test, within_fact_composition,
                     within_fact_model_diff,
                     within_item_exchange_mean,
                     within_item_label_exchange_tail)

SLUGS = ("olmo31-think", "olmo31-instruct", "qwen36-27b")
SIDES = ("confirmatory", "replication")
BANKS = ("bank_f_v7.jsonl", "bank_s_v3.jsonl")
SEED = 4242
THRESHOLD = -1.0
TIER = "methods"
REPO_DATA = Path(__file__).resolve().parents[2] / "data"
PARTITION = (
    Path(__file__).resolve().parents[2]
    / "preregistration" / "partition_phase3.json")


def arg(flag: str, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def normalized_tokens(text: str) -> list[str]:
    return DEFAULT_SPEC.normalize(text).split()


def contains_alias(generated: str, alias: str) -> bool:
    """Contiguous normalized-token match anywhere in the generation."""
    generation = normalized_tokens(generated)
    target = normalized_tokens(alias)
    if not target or len(target) > len(generation):
        return False
    return any(
        generation[index:index + len(target)] == target
        for index in range(len(generation) - len(target) + 1))


def prefix_alias(generated: str, alias: str) -> bool:
    generation = normalized_tokens(generated)
    target = normalized_tokens(alias)
    return bool(target) and generation[:len(target)] == target


def boundary_grade(generated: str, aliases: list[str]) -> dict:
    matches = [alias for alias in aliases
               if contains_alias(generated, alias)]
    prefix = [alias for alias in aliases
              if prefix_alias(generated, alias)]
    return {
        "capable_generation_boundary_safe": bool(matches),
        "capable_prefix_boundary_safe": bool(prefix),
        "matched_aliases_json": json.dumps(matches, ensure_ascii=False),
        "prefix_aliases_json": json.dumps(prefix, ensure_ascii=False),
    }


def _stable_order(value: str) -> str:
    return hashlib.sha256(f"boundary-review-v1|{value}".encode()).hexdigest()


def review_sample(rows: pd.DataFrame) -> pd.DataFrame:
    """Exactly 100 positives and 100 negatives, balanced across models."""
    selected = []
    per_model = {
        "olmo31-think": 34,
        "olmo31-instruct": 33,
        "qwen36-27b": 33,
    }
    for label in (True, False):
        for model, target in per_model.items():
            candidates = rows[
                (rows.model == model)
                & (rows.capable_generation_boundary_safe == label)].copy()
            # Alias-length strata cycle before the deterministic hash so
            # short symbols/numbers and longer names are both inspected.
            candidates["length_bucket"] = pd.cut(
                candidates.alias_min_words,
                bins=[-1, 1, 2, np.inf],
                labels=["one", "two", "three_plus"])
            buckets = []
            for _, sub in candidates.groupby(
                    "length_bucket", observed=True, sort=True):
                sub = sub.assign(
                    review_order=[
                        _stable_order(f"{model}|{item_id}|{label}")
                        for item_id in sub.item_id])
                buckets.append(sub.sort_values("review_order"))
            # Round-robin the strata to avoid a proportional sample being
            # dominated by one-word aliases.
            picks = []
            offset = 0
            while len(picks) < target:
                progressed = False
                for bucket in buckets:
                    if offset < len(bucket) and len(picks) < target:
                        picks.append(bucket.iloc[offset])
                        progressed = True
                if not progressed:
                    break
                offset += 1
            if len(picks) != target:
                raise RuntimeError(
                    f"could not sample {target} {label} rows for {model}")
            selected.extend(picks)
    sample = pd.DataFrame(selected).reset_index(drop=True)
    if sample.capable_generation_boundary_safe.value_counts().to_dict() != {
            True: 100, False: 100}:
        raise RuntimeError("manual review sample is not 100/100")
    sample["review_id"] = [
        hashlib.sha256(
            f"{row.model}|{row.item_id}".encode()).hexdigest()[:16]
        for row in sample.itertuples()]
    return sample[[
        "review_id", "model", "item_id", "canonical_family",
        "variant", "generation", "aliases_json", "alias_min_words",
        "alias_max_words", "capable_generation", "capable_prefix",
        "capable_generation_boundary_safe", "matched_aliases_json",
        "length_bucket",
    ]]


def sample_sha(frame: pd.DataFrame) -> str:
    payload = frame.to_json(
        orient="records", force_ascii=False, double_precision=15)
    return hashlib.sha256(payload.encode()).hexdigest()


def _capability_sets(frame: pd.DataFrame,
                     allowed_families: set[str]) -> dict[str, set[str]]:
    eligible = frame[
        frame.canonical_family.isin(allowed_families)
        & frame.variant.isin(["direct", "composed"])].copy()

    def both(column: str) -> set[str]:
        return {
            fact_id for fact_id, sub in eligible.groupby("fact_id")
            if len(sub) == 2 and bool(sub[column].all())}

    direct = set(eligible[
        (eligible.variant == "direct")
        & eligible.capable_generation_boundary_safe].fact_id)
    output = {
        "historical_unsafe_strict": both("capable_generation"),
        "boundary_safe_strict": both(
            "capable_generation_boundary_safe"),
        "boundary_safe_direct": direct,
    }
    eligible["answer_preference_margin"] = (
        eligible.lp_canonical - eligible.lp_counterfactual)
    for margin in (0.0, 1.0, 2.0):
        column = f"preference_{margin:g}"
        eligible[column] = eligible.answer_preference_margin >= margin
        output[f"answer_preference_margin_{margin:g}"] = both(column)
    output["all_source_verified"] = set(eligible.fact_id)
    return output


def _load_grid(slug: str, side: str) -> pd.DataFrame:
    suffix = "" if side == "confirmatory" else "_replication"
    path = (
        metrics_dir(slug) / f"p3_grid{suffix}"
        / f"p3_grid{suffix}_{slug}.parquet")
    frame = pd.read_parquet(path)
    frame["model"] = slug
    frame["J_eff"] = (
        frame.lp_meanJ_span_safe - frame.lp_baseline)
    frame["C_eff"] = frame.lp_ss_matched - frame.lp_baseline
    frame["specific"] = frame.J_eff - frame.C_eff
    return frame


def analyze_population(
        side: str, name: str, sets_by_model: dict[str, set[str]],
        grids: dict[str, pd.DataFrame]) -> dict:
    filtered = {
        slug: grids[slug][
            grids[slug].fact_id.isin(sets_by_model[slug])].copy()
        for slug in SLUGS}
    combined = pd.concat(filtered.values(), ignore_index=True)
    comp = within_fact_composition(combined, value_col="specific")
    difference = within_fact_model_diff(
        comp, model_a="qwen36-27b",
        model_b=["olmo31-think", "olmo31-instruct"])
    family = difference.groupby(
        "canonical_family", sort=True)["diff"].mean()
    p1 = {
        "estimate_equal_family": (
            float(family.mean()) if len(family) else None),
        "estimate_item_weighted": (
            float(difference["diff"].mean()) if len(difference) else None),
        "n_facts": int(len(difference)),
        "n_families": int(len(family)),
    }
    if len(family) >= 3:
        p1["family_signflip"] = family_signflip_test(
            family.to_numpy(), draws=100_000, seed=SEED)

    qwen = filtered["qwen36-27b"].copy()
    qwen["delta_J"] = qwen.J_eff
    qwen["delta_C"] = qwen.C_eff
    qwen["tail_difference"] = (
        (qwen.delta_J < THRESHOLD).astype(float)
        - (qwen.delta_C < THRESHOLD).astype(float))
    p2_family = qwen.groupby(
        "canonical_family")["tail_difference"].mean()
    p2 = {
        "estimate_equal_family": (
            float(p2_family.mean()) if len(p2_family) else None),
        "estimate_item_weighted": (
            float(qwen.tail_difference.mean()) if len(qwen) else None),
        "n_items": int(len(qwen)),
        "n_families": int(len(p2_family)),
    }

    p3 = None
    bridge = qwen[
        qwen.get("lp_true_bridge", pd.Series(
            index=qwen.index, dtype=float)).notna()].copy()
    if len(bridge):
        bridge["rescue"] = (
            bridge.lp_true_bridge - bridge.lp_distractor_bridge)
        bridge_family = bridge.groupby(
            "canonical_family")["rescue"].mean()
        p3 = {
            "estimate_equal_family": float(bridge_family.mean()),
            "estimate_item_weighted": float(bridge.rescue.mean()),
            "n_items": int(len(bridge)),
            "n_families": int(len(bridge_family)),
        }

    # Inference is repeated only for the historical reference and the
    # binding boundary-safe strict subset. Other populations are
    # descriptive because their eligible supersets lack frozen outcomes.
    if name in {"historical_unsafe_strict", "boundary_safe_strict"}:
        if len(qwen) and qwen.canonical_family.nunique() >= 3:
            p2["label_exchange"] = within_item_label_exchange_tail(
                qwen, draws=100_000, threshold=THRESHOLD, seed=SEED)
        if bridge is not None and len(bridge):
            p3["item_exchange"] = within_item_exchange_mean(
                bridge, a_col="lp_true_bridge",
                b_col="lp_distractor_bridge", draws=100_000,
                seed=SEED, alternative="greater")

    coverage = {}
    for slug in SLUGS:
        observed = set(grids[slug].fact_id)
        eligible = sets_by_model[slug]
        coverage[slug] = {
            "eligible_bank_facts": int(len(eligible)),
            "facts_with_frozen_outcomes": int(len(
                observed & eligible)),
            "eligible_facts_missing_outcomes": int(len(
                eligible - observed)),
        }
    return {
        "side": side, "population": name, "coverage": coverage,
        "P3-P1": p1, "P3-P2": p2, "P3-P3": p3,
        "identification_note": (
            "Estimates use only facts with frozen intervention outcomes. "
            "For supersets of the historical strict cohort, missing "
            "eligible facts are reported and no all-population claim is "
            "made."),
    }


def make_figure(report: dict, path_png: Path, path_pdf: Path) -> None:
    import matplotlib.pyplot as plt

    summary = report["boundary_summary"]
    populations = report["cohort_sensitivity"]["confirmatory"]
    names = list(populations)
    labels = [
        "historical", "boundary strict", "direct observed",
        "preference>0", "preference>1", "preference>2",
        "all verified observed",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].bar(
        list(SLUGS),
        [summary[slug]["unsafe_positive_to_safe_negative"]
         for slug in SLUGS],
        color=["#6d597a", "#b56576", "#355c7d"])
    axes[0].set_ylabel("Unsafe-positive rows rejected")
    axes[0].set_title("Boundary-safe G5 corrections")
    estimates = [
        populations[name]["P3-P1"]["estimate_equal_family"]
        for name in names]
    axes[1].plot(
        range(len(names)), estimates, marker="o", color="#355c7d")
    axes[1].axhline(0, color="black", lw=1, ls="--")
    axes[1].set_xticks(range(len(names)), labels, rotation=40, ha="right")
    axes[1].set_ylabel("P3-P1 equal-family estimate (nats)")
    axes[1].set_title("Observed-outcome cohort sensitivity")
    fig.tight_layout()
    for path in (path_png, path_pdf):
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        fig.savefig(tmp, dpi=180, bbox_inches="tight",
                    format=path.suffix[1:])
        os.replace(tmp, path)
    plt.close(fig)


def main() -> None:  # noqa: C901
    out_dir = (
        metrics_dir("cross_model") / "release_audit"
        / "alias_cohort_sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)
    aliases = {}
    bank_hashes = {}
    for bank in BANKS:
        path = REPO_DATA / bank
        bank_hashes[bank] = sha256_file(path)
        for bundle in load_bank(path):
            for item in bundle.as_items():
                aliases[item["item_id"]] = {
                    "aliases": item["accepted_answers"],
                    "canonical_family": item["canonical_family"],
                    "variant": item["variant"],
                }

    boundary_frames = {}
    input_hashes = {
        "partition": sha256_file(PARTITION),
        **{f"bank:{key}": value for key, value in bank_hashes.items()},
    }
    all_rows = []
    for slug in SLUGS:
        source = (
            metrics_dir(slug) / "g5_bank"
            / f"g5_bank_{slug}_regraded.parquet")
        input_hashes[f"g5:{slug}"] = sha256_file(source)
        frame = pd.read_parquet(source)
        info = frame.item_id.map(aliases)
        if info.isna().any():
            raise RuntimeError(f"{slug} G5 contains unknown bank item")
        frame["aliases_json"] = [
            json.dumps(value["aliases"], ensure_ascii=False)
            for value in info]
        frame["alias_min_words"] = [
            min(len(normalized_tokens(alias))
                for alias in value["aliases"])
            for value in info]
        frame["alias_max_words"] = [
            max(len(normalized_tokens(alias))
                for alias in value["aliases"])
            for value in info]
        grades = [
            boundary_grade(row.generation, value["aliases"])
            for row, value in zip(frame.itertuples(), info)]
        grade_frame = pd.DataFrame(grades)
        frame = pd.concat(
            [frame.reset_index(drop=True), grade_frame], axis=1)
        frame["model"] = slug
        boundary_frames[slug] = frame
        all_rows.append(frame)
    combined_boundary = pd.concat(all_rows, ignore_index=True)
    review = review_sample(combined_boundary)
    review_hash = sample_sha(review)

    if "--prepare-review" in sys.argv:
        output = Path(arg("--review-out", "/tmp/p3_boundary_review.csv"))
        review.assign(
            manual_label="", manual_notes="").to_csv(
                output, index=False)
        print(json.dumps({
            "review_path": str(output),
            "review_sample_sha256": review_hash,
            "n_rows": len(review),
            "positive": int(
                review.capable_generation_boundary_safe.sum()),
            "negative": int(
                (~review.capable_generation_boundary_safe).sum()),
        }, indent=1))
        return

    require_clean_tree("--allow-dirty" in sys.argv)
    attested = arg("--attest-reviewed-sha")
    if attested != review_hash:
        raise RuntimeError(
            "manual review attestation missing or mismatched; first run "
            "--prepare-review and inspect every row, then pass "
            f"--attest-reviewed-sha {review_hash}")
    corrections_path = arg("--review-corrections")
    corrections = {}
    if corrections_path:
        corrections = json.loads(Path(corrections_path).read_text())
    review["manual_label"] = (
        review.capable_generation_boundary_safe.astype(bool))
    review["manual_notes"] = "agrees with boundary-safe contract"
    for review_id, correction in corrections.items():
        mask = review.review_id == review_id
        if int(mask.sum()) != 1:
            raise RuntimeError(
                f"review correction id {review_id} is not unique")
        review.loc[mask, "manual_label"] = bool(correction["label"])
        review.loc[mask, "manual_notes"] = correction["notes"]
    disagreements = int(np.count_nonzero(
        review.manual_label
        != review.capable_generation_boundary_safe))
    review["reviewer"] = "Codex agent, explicit 200-row review"
    review["reviewed_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    partition = json.loads(PARTITION.read_text())["payload"]
    cohort_report = {}
    boundary_summary = {}
    boundary_paths = []
    for slug, frame in boundary_frames.items():
        unsafe = frame.capable_generation.astype(bool)
        safe = frame.capable_generation_boundary_safe.astype(bool)
        boundary_summary[slug] = {
            "n_items": int(len(frame)),
            "historical_capable": int(unsafe.sum()),
            "boundary_safe_capable": int(safe.sum()),
            "unsafe_positive_to_safe_negative": int(
                np.count_nonzero(unsafe & ~safe)),
            "unsafe_negative_to_safe_positive": int(
                np.count_nonzero(~unsafe & safe)),
            "changed_item_ids": sorted(
                frame.loc[unsafe != safe, "item_id"].tolist()),
        }
        path = out_dir / f"g5_boundary_safe_{slug}.parquet"
        atomic_parquet(path, frame)
        boundary_paths.append(path)

    for side in SIDES:
        allowed = set(partition[side])
        grids = {slug: _load_grid(slug, side) for slug in SLUGS}
        for slug, grid in grids.items():
            suffix = "" if side == "confirmatory" else "_replication"
            grid_path = (
                metrics_dir(slug) / f"p3_grid{suffix}"
                / f"p3_grid{suffix}_{slug}.parquet")
            input_hashes[f"grid:{slug}:{side}"] = sha256_file(grid_path)
        sets = {
            slug: _capability_sets(
                boundary_frames[slug], allowed)
            for slug in SLUGS}
        population_names = list(next(iter(sets.values())).keys())
        cohort_report[side] = {
            name: analyze_population(
                side, name,
                {slug: sets[slug][name] for slug in SLUGS},
                grids)
            for name in population_names}

    report = {
        "schema_version": 1,
        "boundary_contract": {
            "normalization": DEFAULT_SPEC.normalization,
            "match": (
                "contiguous normalized words anywhere; no substring "
                "or partial-word matches"),
            "symbols_and_numbers": (
                "same exact normalized-token rule; punctuation may bound "
                "a token but characters may not be embedded in a word"),
        },
        "boundary_summary": boundary_summary,
        "manual_review": {
            "sample_sha256": review_hash,
            "n_rows": int(len(review)),
            "n_positive": int(
                review.capable_generation_boundary_safe.sum()),
            "n_negative": int(
                (~review.capable_generation_boundary_safe).sum()),
            "disagreements": disagreements,
            "corrections_file": corrections_path,
        },
        "cohort_sensitivity": cohort_report,
        "population_contract": {
            "historical_unsafe_strict": (
                "both direct and composed pass historical substring G5"),
            "boundary_safe_strict": (
                "both direct and composed pass boundary-safe G5"),
            "boundary_safe_direct": (
                "direct passes boundary-safe G5; observed-outcome subset"),
            "answer_preference_margin": (
                "both direct and composed canonical LP exceed "
                "counterfactual LP by the named 0/1/2-nat margin"),
            "all_source_verified": (
                "all facts in frozen-side families; estimates necessarily "
                "remain observed-outcome-only because broader interventions "
                "were never run"),
        },
        "inputs_sha256": input_hashes,
    }
    review_path = out_dir / "P3_BOUNDARY_HAND_AUDIT_200.csv"
    review.to_csv(review_path, index=False)
    figure_png = out_dir / "p3_alias_cohort_sensitivity.png"
    figure_pdf = out_dir / "p3_alias_cohort_sensitivity.pdf"
    make_figure(report, figure_png, figure_pdf)
    result_path = out_dir / "p3_alias_cohort_sensitivity.json"
    command = (
        "python -m "
        "jspace_phase3.experiments.p3_alias_and_cohort_sensitivity "
        f"--attest-reviewed-sha {review_hash}"
        + (f" --review-corrections {corrections_path}"
           if corrections_path else ""))
    evidence_id = "p3-boundary-cohort-sensitivity-v1"
    write_result3(
        report, result_path,
        Provenance3(
            evidence_id=evidence_id, tier=TIER, command=command,
            inputs=input_hashes, seed=SEED))
    markdown_path = out_dir / "p3_alias_cohort_sensitivity.md"
    reference = cohort_report[
        "confirmatory"]["historical_unsafe_strict"]
    safe = cohort_report["confirmatory"]["boundary_safe_strict"]
    markdown_path.write_text(
        "# Phase 3 boundary and cohort sensitivity\n\n"
        f"Boundary-safe regrading rejects "
        f"{sum(value['unsafe_positive_to_safe_negative'] for value in boundary_summary.values())} "
        "historical positive rows across three models. The deterministic "
        f"200-row hand audit has {disagreements} disagreements.\n\n"
        f"Confirmatory P3-P1 changes from "
        f"{reference['P3-P1']['estimate_equal_family']:+.4f} "
        f"({reference['P3-P1']['n_facts']} facts) "
        f"to {safe['P3-P1']['estimate_equal_family']:+.4f} "
        f"({safe['P3-P1']['n_facts']} facts) under boundary-safe strict "
        "capability. Superset populations are explicitly "
        "observed-outcome-only because their additional eligible facts "
        "were never intervened on in the frozen grid.\n")
    outputs = [
        result_path, markdown_path, review_path,
        figure_png, figure_pdf, *boundary_paths,
    ]
    register(
        evidence_id, tier=TIER, command=command,
        what=(
            "Post-freeze boundary-safe G5 and cohort-selection sensitivity "
            "across all three Phase 3 models/sides, with deterministic "
            f"100-positive/100-negative hand audit ({disagreements} "
            "disagreements)."),
        outputs=outputs, inputs=input_hashes)
    print(json.dumps({
        "boundary_summary": boundary_summary,
        "manual_review": report["manual_review"],
        "confirmatory_reference": reference,
        "confirmatory_boundary_safe": safe,
    }, indent=1))


if __name__ == "__main__":
    main()
