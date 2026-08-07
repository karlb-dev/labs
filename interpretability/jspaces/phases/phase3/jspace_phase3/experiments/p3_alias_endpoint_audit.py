"""Post-freeze Phase 3 accepted-alias scoring sensitivity.

The frozen primary grid scored only ``accepted_answers[0]``.  This audit
freezes a common 20-fact confirmatory subset before looking at new outcomes
and scores four prospective views:

* the stable-seed first alias;
* the canonical alias;
* logsumexp over a tokenizer-audited prefix-disjoint alias set;
* max alias (diagnostic only).

The immutable historical first-alias rows are carried as a fifth view.  They
are not silently mixed with the new stable matched-control realization.

Run one resumable CUDA cell per model:

    python -m jspace_phase3.experiments.p3_alias_endpoint_audit \
      --config interpretability/jspace_phase3/configs/\
p3_alias_endpoint_qwen36-27b.yaml

After all three cells are banked:

    python -m jspace_phase3.experiments.p3_alias_endpoint_audit --analyze

Every model forward and answer softmax remains on CUDA.  The runner has no
CPU model fallback and checkpoints one alias cell at a time to Drive.
"""
from __future__ import annotations

import hashlib
import itertools
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
from ..bank import FactBundle, load_bank
from ..gpu import require_cuda_gpu
from ..paths3 import metrics_dir
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           resolve_model, write_result3)
from ..scoring import DEFAULT_SPEC, ScoringSession
from ..seeds import SEED_CONTRACT, stable_seed
from ..stats import exact_signflip_test, within_item_label_exchange_tail
from .p3_protected_answer_audit import (
    canonical_hash, tokenizer_manifest)

SLUGS = ("olmo31-think", "olmo31-instruct", "qwen36-27b")
ENDPOINTS = (
    "historical_first_alias",
    "stable_first_alias",
    "canonical_alias",
    "prefix_disjoint_logsumexp",
    "max_alias",
)
SEED = 4242
BOOTSTRAP_DRAWS = 100_000
TIER = "methods"
CROSS_EVIDENCE_ID = "p3-alias-endpoint-cross-model-v1"
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = PACKAGE_ROOT / "data"
DEFAULT_SELECTION = (
    PACKAGE_ROOT / "preregistration"
    / "p3_alias_sensitivity_selection_v1.json"
)


def arg(flag: str, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def validate_state_header(saved: dict, current: dict) -> None:
    mismatch = {
        key: [saved.get(key), value]
        for key, value in current.items()
        if saved.get(key) != value
    }
    if mismatch:
        raise RuntimeError(
            "refusing incompatible alias-endpoint resume: "
            + json.dumps(mismatch, sort_keys=True))


def historical_path(slug: str) -> Path:
    return (
        metrics_dir(slug) / "p3_grid"
        / f"p3_grid_{slug}.parquet"
    )


def _is_token_prefix(left: list[int], right: list[int]) -> bool:
    shorter, longer = (
        (left, right) if len(left) <= len(right) else (right, left))
    return longer[:len(shorter)] == shorter


def prefix_disjoint_aliases(
        aliases: list[str], token_ids: dict[str, list[int]],
        canonical_alias: str) -> list[str]:
    """Maximum-cardinality, deterministic prefix-disjoint alias subset.

    Ties prefer a set containing the canonical alias, then the historical
    first alias, then the lexicographically earliest tuple of frozen alias
    ordinals.  Alias sets are tiny, so exact subset enumeration is clearer
    and safer than a greedy approximation.
    """
    if not aliases:
        raise ValueError("empty alias set")
    if len(set(aliases)) != len(aliases):
        raise ValueError("duplicate accepted aliases")
    if set(token_ids) != set(aliases):
        raise ValueError("token-id manifest does not match aliases")
    valid = []
    for size in range(1, len(aliases) + 1):
        for indices in itertools.combinations(range(len(aliases)), size):
            if all(
                    not _is_token_prefix(
                        token_ids[aliases[left]],
                        token_ids[aliases[right]])
                    for left, right in itertools.combinations(indices, 2)):
                valid.append(indices)
    if not valid:
        raise RuntimeError("no nonempty prefix-disjoint alias subset")
    best_size = max(map(len, valid))
    candidates = [indices for indices in valid if len(indices) == best_size]
    chosen = min(
        candidates,
        key=lambda indices: (
            -(aliases.index(canonical_alias) in indices),
            -(0 in indices),
            tuple(indices),
        ))
    return [aliases[index] for index in chosen]


def _frozen_complete_facts(frame: pd.DataFrame) -> set[str]:
    return {
        fact_id for fact_id, sub in frame.groupby("fact_id")
        if set(sub["variant"]) == {"direct", "composed"}
    }


def reproduce_selection(
        bundles: dict[str, FactBundle],
        frozen_grids: dict[str, pd.DataFrame], *,
        n_facts: int = 20) -> list[str]:
    """Reproduce the pre-outcome deterministic selection algorithm."""
    common: set[str] | None = None
    for slug in SLUGS:
        complete = _frozen_complete_facts(frozen_grids[slug])
        common = complete if common is None else common & complete
    common = common or set()
    missing = common - set(bundles)
    if missing:
        raise RuntimeError(f"frozen facts absent from banks: {sorted(missing)}")
    selected = {
        fact_id for fact_id in common
        if len(bundles[fact_id].accepted_answers) > 1
    }
    families = sorted({
        bundles[fact_id].canonical_family for fact_id in common})
    for family in families:
        if any(
                bundles[fact_id].canonical_family == family
                for fact_id in selected):
            continue
        candidates = sorted(
            (
                fact_id for fact_id in common
                if bundles[fact_id].canonical_family == family
            ),
            key=lambda fact_id: (
                stable_seed(
                    "p3-alias-sensitivity-family", fact_id, SEED),
                fact_id,
            ))
        selected.add(candidates[0])
    extras = sorted(
        common - selected,
        key=lambda fact_id: (
            stable_seed("p3-alias-sensitivity-extra", fact_id, SEED),
            fact_id,
        ))
    selected.update(extras[:n_facts - len(selected)])
    if len(selected) != n_facts:
        raise RuntimeError(
            f"selection produced {len(selected)} rather than {n_facts} facts")
    return sorted(selected)


def load_and_validate_selection(
        selection_path: Path,
) -> tuple[dict, dict[str, FactBundle], dict[str, pd.DataFrame], dict]:
    selection = json.loads(selection_path.read_text())
    bundles_list = [
        bundle for name in ("bank_f_v7.jsonl", "bank_s_v3.jsonl")
        for bundle in load_bank(DATA_ROOT / name)
    ]
    bundles = {bundle.fact_id: bundle for bundle in bundles_list}
    frozen_grids = {
        slug: pd.read_parquet(historical_path(slug))
        for slug in SLUGS
    }
    actual_hashes = {
        "bank_f_v7.jsonl": sha256_file(DATA_ROOT / "bank_f_v7.jsonl"),
        "bank_s_v3.jsonl": sha256_file(DATA_ROOT / "bank_s_v3.jsonl"),
        "partition_phase3.json": sha256_file(
            PACKAGE_ROOT / "preregistration" / "partition_phase3.json"),
        **{
            f"p3_grid_{slug}.parquet": sha256_file(historical_path(slug))
            for slug in SLUGS
        },
    }
    if actual_hashes != selection["frozen_input_sha256"]:
        raise RuntimeError(
            "alias selection frozen-input hashes changed: "
            + json.dumps({
                key: [selection["frozen_input_sha256"].get(key), value]
                for key, value in actual_hashes.items()
                if selection["frozen_input_sha256"].get(key) != value
            }, sort_keys=True))
    reproduced = reproduce_selection(
        bundles, frozen_grids,
        n_facts=int(selection["expected_facts"]))
    if reproduced != selection["fact_ids"]:
        raise RuntimeError(
            "stored alias selection does not match its algorithm")
    selected = set(selection["fact_ids"])
    multi_alias = sorted(
        fact_id for fact_id in selected
        if len(bundles[fact_id].accepted_answers) > 1)
    if multi_alias != selection["required_multi_alias_facts"]:
        raise RuntimeError("stored multi-alias coverage is inconsistent")
    families = {
        bundles[fact_id].canonical_family for fact_id in selected}
    if len(families) != int(selection["expected_families"]):
        raise RuntimeError("stored alias selection family count changed")
    return selection, bundles, frozen_grids, actual_hashes


def selected_items(
        selection: dict, bundles: dict[str, FactBundle]) -> list[dict]:
    items = []
    for fact_id in selection["fact_ids"]:
        for item in bundles[fact_id].as_items():
            if item["variant"] in {"direct", "composed"}:
                items.append(item)
    items.sort(key=lambda row: row["item_id"])
    if len(items) != int(selection["expected_items"]):
        raise RuntimeError("alias selection item count changed")
    return items


def aggregate_alias_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-alias LPs to the five frozen sensitivity endpoints."""
    rows = []
    metadata = [
        "slug", "item_id", "fact_id", "variant", "bank",
        "canonical_family", "relation_group",
    ]
    for _, sub in frame.groupby("item_id", sort=True):
        sub = sub.sort_values("alias_ordinal")
        base = {column: sub.iloc[0][column] for column in metadata}
        historical = {
            "lp_baseline": float(sub.historical_lp_baseline.iloc[0]),
            "lp_meanJ_span_safe": float(
                sub.historical_lp_meanJ_span_safe.iloc[0]),
            "lp_ss_matched": float(
                sub.historical_lp_ss_matched.iloc[0]),
        }
        endpoint_specs = {
            "stable_first_alias": (
                "single", sub[sub.alias_ordinal == 0]),
            "canonical_alias": (
                "single", sub[sub.is_canonical_alias]),
            "prefix_disjoint_logsumexp": (
                "logsumexp", sub[sub.in_prefix_disjoint_set]),
            "max_alias": ("max", sub),
        }

        def finish(endpoint: str, values: dict, *, stable: bool) -> None:
            row = {
                **base,
                "endpoint": endpoint,
                **values,
                "control_realization": (
                    "sha256-v1 seed=31337"
                    if stable else "historical Python-hash realization"),
                "n_accepted_aliases": int(len(sub)),
                "n_prefix_disjoint_aliases": int(
                    sub.in_prefix_disjoint_set.sum()),
            }
            row["J_effect"] = (
                row["lp_meanJ_span_safe"] - row["lp_baseline"])
            row["C_effect"] = row["lp_ss_matched"] - row["lp_baseline"]
            row["specific"] = row["J_effect"] - row["C_effect"]
            rows.append(row)

        finish("historical_first_alias", historical, stable=False)
        for endpoint, (operation, selected) in endpoint_specs.items():
            if selected.empty:
                raise RuntimeError(
                    f"{base['item_id']} has no rows for {endpoint}")
            values = {}
            for column in (
                    "lp_baseline", "lp_meanJ_span_safe", "lp_ss_matched"):
                data = selected[column].to_numpy(dtype=float)
                if operation == "single":
                    if len(data) != 1:
                        raise RuntimeError(
                            f"{base['item_id']} endpoint {endpoint} "
                            "is not unique")
                    value = data[0]
                elif operation == "logsumexp":
                    value = np.logaddexp.reduce(data)
                elif operation == "max":
                    value = data.max()
                else:  # pragma: no cover - local programming guard
                    raise ValueError(operation)
                values[column] = float(value)
            finish(endpoint, values, stable=True)
    return pd.DataFrame(rows)


def _family_bootstrap(values: pd.Series) -> dict:
    array = values.to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    draws = np.empty(BOOTSTRAP_DRAWS)
    for start in range(0, BOOTSTRAP_DRAWS, 10_000):
        n = min(10_000, BOOTSTRAP_DRAWS - start)
        indices = rng.integers(0, len(array), size=(n, len(array)))
        draws[start:start + n] = array[indices].mean(axis=1)
    return {
        "estimate": float(array.mean()),
        "ci95_family_bootstrap": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "n_families": int(len(array)),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    }


def model_endpoint_summary(aggregate: pd.DataFrame) -> dict:
    output = {}
    for endpoint in ENDPOINTS:
        frame = aggregate[aggregate.endpoint == endpoint]
        pivot = frame.pivot_table(
            index=["fact_id", "canonical_family"],
            columns="variant", values="specific",
            aggfunc="first").dropna(subset=["direct", "composed"])
        composition = (pivot.composed - pivot.direct).rename("value")
        family = composition.groupby("canonical_family", sort=True).mean()
        output[endpoint] = {
            **_family_bootstrap(family),
            "n_facts": int(len(composition)),
            "item_weighted": float(composition.mean()),
        }
    return output


def p3p1_endpoint(frame: pd.DataFrame, endpoint: str) -> tuple[dict, pd.DataFrame]:
    selected = frame[frame.endpoint == endpoint]
    composition = selected.pivot_table(
        index=["fact_id", "canonical_family", "slug"],
        columns="variant", values="specific",
        aggfunc="first").dropna(subset=["direct", "composed"]).reset_index()
    composition["composition"] = (
        composition["composed"] - composition["direct"])
    pivot = composition.pivot_table(
        index=["fact_id", "canonical_family"],
        columns="slug", values="composition", aggfunc="first")
    pivot = pivot.dropna(subset=list(SLUGS))
    paired = pivot.reset_index()[["fact_id", "canonical_family"]].copy()
    paired["diff"] = (
        pivot["qwen36-27b"].to_numpy()
        - 0.5 * (
            pivot["olmo31-think"].to_numpy()
            + pivot["olmo31-instruct"].to_numpy()))
    family = paired.groupby(
        "canonical_family", sort=True)["diff"].mean()
    result = {
        **_family_bootstrap(family),
        "item_weighted": float(paired["diff"].mean()),
        "n_facts": int(len(paired)),
        "exact_family_signflip": exact_signflip_test(
            family.to_numpy()),
    }
    return result, paired


def analyze_cross_model(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    if set(frame.slug.unique()) != set(SLUGS):
        raise RuntimeError("cross-model alias analysis lacks a model")
    p1 = {}
    paired_rows = []
    paired_by_endpoint = {}
    for endpoint in ENDPOINTS:
        p1[endpoint], paired = p3p1_endpoint(frame, endpoint)
        paired["endpoint"] = endpoint
        paired_rows.append(paired)
        paired_by_endpoint[endpoint] = paired.set_index("fact_id")

    reference = paired_by_endpoint["stable_first_alias"]
    sensitivity = {}
    for endpoint in (
            "canonical_alias", "prefix_disjoint_logsumexp", "max_alias"):
        alternate = paired_by_endpoint[endpoint]
        joined = reference[["canonical_family", "diff"]].join(
            alternate[["diff"]], how="inner",
            lsuffix="_reference", rsuffix="_alternate")
        joined["change"] = (
            joined.diff_alternate - joined.diff_reference)
        family = joined.groupby("canonical_family", sort=True).change.mean()
        sensitivity[endpoint] = {
            **_family_bootstrap(family),
            "n_facts": int(len(joined)),
            "max_abs_fact_change": float(joined.change.abs().max()),
            "n_nonzero_fact_changes": int(
                np.count_nonzero(np.abs(joined.change) > 1e-12)),
            "exact_family_signflip": exact_signflip_test(
                family.to_numpy()),
        }

    qwen_tail = {}
    for endpoint in ENDPOINTS:
        qwen = frame[
            (frame.slug == "qwen36-27b")
            & (frame.endpoint == endpoint)
        ]
        qwen_tail[endpoint] = within_item_label_exchange_tail(
            qwen.rename(columns={
                "J_effect": "delta_J", "C_effect": "delta_C"}),
            draws=100_000, threshold=-1.0, seed=SEED)

    multi_facts = set(
        frame.loc[
            frame.n_accepted_aliases > 1, "fact_id"].unique())
    multi = {
        endpoint: {
            "estimate_item_weighted": float(
                paired_by_endpoint[endpoint].loc[
                    list(multi_facts), "diff"].mean()),
            "n_facts": int(len(multi_facts)),
        }
        for endpoint in ENDPOINTS
    }
    report = {
        "schema_version": 1,
        "design": {
            "side": "confirmatory",
            "selection": (
                "all four multi-alias facts in the common frozen cohort, "
                "plus family-stratified coverage to 20 facts/17 families"),
            "control_seed": 31337,
            "control_seed_contract": SEED_CONTRACT,
            "primary_sensitivity_reference": "stable_first_alias",
            "historical_endpoint_kept_separate": True,
            "inference_status": (
                "post-freeze methods sensitivity; not a new confirmatory "
                "test"),
        },
        "P3-P1_subset": p1,
        "P3-P1_alias_change_vs_stable_first": sensitivity,
        "P3-P2_qwen_subset": qwen_tail,
        "multi_alias_facts_descriptive": multi,
        "interpretation_guardrail": (
            "The subset was enriched to include every shared multi-alias "
            "fact. Alias variants are scoring views of the same item, not "
            "independent experimental units. Max alias is diagnostic only."),
    }
    return report, pd.concat(paired_rows, ignore_index=True)


def make_cross_figure(report: dict, paired: pd.DataFrame,
                      png: Path, pdf: Path) -> None:
    import matplotlib.pyplot as plt

    labels = {
        "historical_first_alias": "historical\nfirst",
        "stable_first_alias": "stable\nfirst",
        "canonical_alias": "canonical",
        "prefix_disjoint_logsumexp": "prefix-free\nlogsumexp",
        "max_alias": "max",
    }
    colors = ["#8d99ae", "#355c7d", "#2a9d8f", "#e9c46a", "#e76f51"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
    estimates = [
        report["P3-P1_subset"][endpoint]["estimate"]
        for endpoint in ENDPOINTS]
    intervals = [
        report["P3-P1_subset"][endpoint]["ci95_family_bootstrap"]
        for endpoint in ENDPOINTS]
    errors = np.array([
        [estimate - interval[0] for estimate, interval in zip(
            estimates, intervals)],
        [interval[1] - estimate for estimate, interval in zip(
            estimates, intervals)],
    ])
    x = np.arange(len(ENDPOINTS))
    axes[0].bar(x, estimates, color=colors, alpha=0.9)
    axes[0].errorbar(
        x, estimates, yerr=errors, fmt="none",
        ecolor="black", capsize=4, lw=1)
    axes[0].axhline(0, color="black", lw=1, ls="--")
    axes[0].set_xticks(x, [labels[value] for value in ENDPOINTS])
    axes[0].set_ylabel("P3-P1 estimate (Qwen − OLMo mean)")
    axes[0].set_title("Common 20-fact sensitivity subset")

    pivot = paired.pivot_table(
        index=["fact_id", "canonical_family"],
        columns="endpoint", values="diff", aggfunc="first")
    alias_endpoints = [
        "canonical_alias", "prefix_disjoint_logsumexp", "max_alias"]
    changes = [
        (
            pivot[endpoint] - pivot["stable_first_alias"]
        ).to_numpy()
        for endpoint in alias_endpoints]
    axes[1].boxplot(
        changes,
        tick_labels=[labels[value] for value in alias_endpoints],
        showfliers=True)
    axes[1].axhline(0, color="black", lw=1, ls="--")
    axes[1].set_ylabel("Per-fact P3-P1 change vs stable first")
    axes[1].set_title("Alias aggregation perturbation")
    fig.suptitle("Phase 3 accepted-alias endpoint audit")
    fig.tight_layout()
    for path in (png, pdf):
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        fig.savefig(
            tmp, dpi=180, bbox_inches="tight",
            format=path.suffix[1:])
        os.replace(tmp, path)
    plt.close(fig)


def write_model_markdown(path: Path, slug: str, payload: dict) -> None:
    summaries = payload["composition_specificity_by_endpoint"]
    lines = [
        f"# Phase 3 alias endpoint cell — {slug}",
        "",
        "- Model forwards and answer-token softmaxes ran on CUDA; no CPU "
        "model fallback exists in this runner.",
        f"- Frozen subset: {payload['n_facts']} facts / "
        f"{payload['n_families']} families / {payload['n_items']} paired "
        "direct-composed items.",
        f"- Accepted aliases scored: {payload['n_alias_cells']}; "
        f"token-prefix overlaps: {payload['n_prefix_overlap_pairs']}.",
        f"- First-alias replay maximum errors: baseline "
        f"{payload['replay_gate']['max_abs_baseline_error']:.8f} nats, "
        f"span-safe J {payload['replay_gate']['max_abs_j_error']:.8f} "
        "nats.",
        "",
        "Equal-family mean composition specificity by endpoint:",
        "",
    ]
    for endpoint in ENDPOINTS:
        lines.append(
            f"- `{endpoint}`: {summaries[endpoint]['estimate']:+.6f} "
            f"[{summaries[endpoint]['ci95_family_bootstrap'][0]:+.6f}, "
            f"{summaries[endpoint]['ci95_family_bootstrap'][1]:+.6f}].")
    lines += [
        "",
        "The historical endpoint retains its original process-randomized "
        "matched control. All other endpoints use explicit sha256-v1 seed "
        "31337 and therefore must not be differenced as if the controls "
        "were the same realization.",
        "",
    ]
    atomic_text(path, "\n".join(lines))


def write_cross_markdown(path: Path, report: dict) -> None:
    lines = [
        "# Phase 3 accepted-alias endpoint audit",
        "",
        "This post-freeze methods sensitivity contains all four multi-alias "
        "facts shared by the three frozen confirmatory cohorts and adds "
        "family-stratified coverage to 20 facts across 17 families.",
        "",
        "## P3-P1 on the frozen sensitivity subset",
        "",
    ]
    for endpoint in ENDPOINTS:
        result = report["P3-P1_subset"][endpoint]
        lines.append(
            f"- `{endpoint}`: {result['estimate']:+.6f} "
            f"[{result['ci95_family_bootstrap'][0]:+.6f}, "
            f"{result['ci95_family_bootstrap'][1]:+.6f}]; exact "
            f"p={result['exact_family_signflip']['p']:.8f}.")
    lines += [
        "",
        "## Alias changes relative to stable first alias",
        "",
    ]
    for endpoint, result in report[
            "P3-P1_alias_change_vs_stable_first"].items():
        lines.append(
            f"- `{endpoint}`: change {result['estimate']:+.6f} "
            f"[{result['ci95_family_bootstrap'][0]:+.6f}, "
            f"{result['ci95_family_bootstrap'][1]:+.6f}]; "
            f"{result['n_nonzero_fact_changes']}/{result['n_facts']} facts "
            "changed.")
    lines += [
        "",
        "The historical and stable-first views are shown separately "
        "because their matched-control realizations differ. Max alias is "
        "diagnostic only. This enriched subset is a robustness audit, not "
        "a new confirmatory test.",
        "",
    ]
    atomic_text(path, "\n".join(lines))


@torch.inference_mode()
def run_model(config_path: Path) -> None:  # noqa: C901
    config = yaml.safe_load(config_path.read_text())
    require_clean_tree("--allow-dirty" in sys.argv)
    slug = config["slug"]
    if slug not in SLUGS:
        raise ValueError(f"unexpected Phase 3 model {slug}")
    selection_path = Path(config["selection_path"])
    if not selection_path.is_absolute():
        selection_path = REPO_ROOT / selection_path
    selection, bundles, frozen_grids, frozen_hashes = \
        load_and_validate_selection(selection_path)
    items = selected_items(selection, bundles)
    frozen = frozen_grids[slug].set_index("item_id")
    if set(item["item_id"] for item in items) - set(frozen.index):
        raise RuntimeError(f"{slug} frozen grid lacks selected items")

    model_path = Path(resolve_uri(config["model_uri"], must_exist=True))
    lens_path = Path(resolve_uri(config["lens_uri"], must_exist=True))
    tok_manifest = tokenizer_manifest(model_path)
    inputs = {
        "runner": sha256_file(__file__),
        "config": sha256_file(config_path),
        "selection": sha256_file(selection_path),
        "lens": sha256_file(lens_path),
        "tokenizer_manifest": tok_manifest["manifest_sha256"],
        **{f"frozen:{key}": value for key, value in frozen_hashes.items()},
    }
    header = {
        "schema": "p3-alias-endpoint-state-v1",
        "runner_sha256": inputs["runner"],
        "config_sha256": inputs["config"],
        "selection_sha256": inputs["selection"],
        "selection_id": selection["selection_id"],
        "model": resolve_model(str(model_path)),
        "lens_sha256": inputs["lens"],
        "tokenizer_manifest_sha256": inputs["tokenizer_manifest"],
        "historical_grid_sha256": frozen_hashes[
            f"p3_grid_{slug}.parquet"],
        "control_seed": int(config["control_seed"]),
        "control_seed_namespace": config["control_seed_namespace"],
        "seed_contract": SEED_CONTRACT,
    }
    out_dir = (
        metrics_dir(slug) / "release_audit"
        / config.get("output_subdir", "alias_endpoint"))
    out_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(exist_ok=True)
    state_path = out_dir / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        validate_state_header(state["header"], header)
    else:
        state = {
            "header": header,
            "done": {},
            "started_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_json(state_path, state)

    expected_cells = sum(
        len(item["accepted_answers"]) for item in items)
    gpu = require_cuda_gpu()
    log("GPU gate PASS: " + json.dumps(gpu, sort_keys=True))
    state["gpu"] = gpu
    atomic_json(state_path, state)

    if len(state["done"]) < expected_cells:
        import jlens
        import transformers
        from jlens import JacobianLens

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True)
        hf = transformers.AutoModelForCausalLM.from_pretrained(
            str(model_path), dtype=torch.bfloat16,
            local_files_only=True).to("cuda").eval()
        if not next(hf.parameters()).is_cuda:
            raise RuntimeError("model parameters are not on CUDA")
        model = jlens.from_hf(hf, tokenizer)
        session = ScoringSession(tokenizer, DEFAULT_SPEC, device="cuda")
        lens = JacobianLens.load(str(lens_path))
        band = [int(value) for value in config["band"]]
        dictionaries = build_j_dictionaries(hf, lens, band)
        if not all(value.is_cuda for value in dictionaries.values()):
            raise RuntimeError("J dictionaries are not on CUDA")
        ablator = Phase3JAblator(model.layers, band)
        k = int(config["k"])
        protect_k = int(config["protect_top_k"])
        control_seed = int(config["control_seed"])
        namespace = config["control_seed_namespace"]
        replay_tolerance = float(config["replay_tolerance"])
        started = time.time()

        def j_arm(ids: torch.Tensor, protect_sets: torch.Tensor):
            ablator.log = type(ablator.log)()
            ablator.phase, ablator.forward_index = "prefill", 0
            ablator.mode = {
                "dicts": dictionaries,
                "k": k,
                "nonneg": True,
                "protect_sets": protect_sets,
                "active_phases": {"prefill"},
                "span_safe": True,
                "record_overlap": True,
                "answer_id": None,
            }
            with ablator:
                logits = hf(
                    input_ids=ids, use_cache=False).logits[0].float()
            ablator.mode = None
            if not logits.is_cuda:
                raise RuntimeError("J-arm logits left CUDA")
            return logits, ablator.log

        ordinal = 0
        for item in items:
            aliases = item["accepted_answers"]
            canonical_alias = f" {item['canonical_answer']}"
            if canonical_alias not in aliases:
                raise RuntimeError(
                    f"{item['item_id']}: canonical alias "
                    f"{canonical_alias!r} is absent")
            token_ids = {
                alias: session.answer_ids(alias)[0].tolist()
                for alias in aliases
            }
            audit = session.alias_audit(aliases)
            prefix_set = prefix_disjoint_aliases(
                aliases, token_ids, canonical_alias)
            for alias_ordinal, alias in enumerate(aliases):
                ordinal += 1
                alias_hash = hashlib.sha256(
                    alias.encode("utf-8")).hexdigest()
                key = f"{item['item_id']}|{alias_hash}"
                if key in state["done"]:
                    continue
                full, n_prompt = session.full_ids(item["prompt"], alias)
                if not full.is_cuda:
                    raise RuntimeError("candidate ids are not on CUDA")
                clean = hf(
                    input_ids=full, use_cache=False).logits[0].float()
                if not clean.is_cuda:
                    raise RuntimeError("baseline logits left CUDA")
                protect_sets = clean.topk(
                    protect_k, dim=-1).indices
                baseline_lp = session.answer_seq_lp(
                    full, clean, n_prompt)
                j_logits, jlog = j_arm(full, protect_sets)
                j_lp = session.answer_seq_lp(
                    full, j_logits, n_prompt)
                profile = profile_from_p3log(
                    jlog, overlap_records=jlog.overlap)
                matched_logits, matched_log = teacher_forced_matched_arm(
                    hf, model.layers, band, dictionaries, full, profile,
                    variant="instant_rank_energy_matched",
                    protect_sets=protect_sets,
                    seed_base=stable_seed(
                        namespace, item["item_id"], control_seed),
                    return_cpu=False)
                if not matched_logits.is_cuda:
                    raise RuntimeError("matched-control logits left CUDA")
                matched_lp = session.answer_seq_lp(
                    full, matched_logits, n_prompt)
                historical = frozen.loc[item["item_id"]]
                baseline_error = (
                    abs(baseline_lp - float(historical.lp_baseline))
                    if alias_ordinal == 0 else np.nan)
                j_error = (
                    abs(j_lp - float(
                        historical.lp_meanJ_span_safe))
                    if alias_ordinal == 0 else np.nan)
                if alias_ordinal == 0 and max(
                        baseline_error, j_error) > replay_tolerance:
                    atomic_json(state_path, state)
                    raise RuntimeError(
                        f"{slug}:{item['item_id']}: deterministic replay "
                        f"errors baseline={baseline_error}, J={j_error}")
                row = {
                    "slug": slug,
                    "item_id": item["item_id"],
                    "fact_id": item["fact_id"],
                    "variant": item["variant"],
                    "bank": item["bank"],
                    "canonical_family": item["canonical_family"],
                    "relation_group": item["relation_group"],
                    "alias": alias,
                    "alias_ordinal": int(alias_ordinal),
                    "is_first_alias": bool(alias_ordinal == 0),
                    "is_canonical_alias": bool(alias == canonical_alias),
                    "in_prefix_disjoint_set": bool(alias in prefix_set),
                    "n_alias_tokens": int(len(token_ids[alias])),
                    "alias_token_ids_json": json.dumps(token_ids[alias]),
                    "alias_token_ids_sha256": canonical_hash(
                        token_ids[alias]),
                    "accepted_aliases_json": json.dumps(
                        aliases, ensure_ascii=False),
                    "accepted_alias_token_ids_json": json.dumps(
                        token_ids, ensure_ascii=False, sort_keys=True),
                    "token_prefix_overlaps_json": json.dumps(
                        audit["prefix_overlaps"], ensure_ascii=False),
                    "prefix_disjoint_aliases_json": json.dumps(
                        prefix_set, ensure_ascii=False),
                    "lp_baseline": baseline_lp,
                    "lp_meanJ_span_safe": j_lp,
                    "lp_ss_matched": matched_lp,
                    "historical_lp_baseline": float(
                        historical.lp_baseline),
                    "historical_lp_meanJ_span_safe": float(
                        historical.lp_meanJ_span_safe),
                    "historical_lp_ss_matched": float(
                        historical.lp_ss_matched),
                    "first_alias_baseline_abs_error": baseline_error,
                    "first_alias_j_abs_error": j_error,
                    "stable_control_seed": control_seed,
                    "stable_control_seed_value": stable_seed(
                        namespace, item["item_id"], control_seed),
                    "j_overlap_summary_json": json.dumps(
                        jlog.overlap_summary(), sort_keys=True),
                    "matched_summary_json": json.dumps(
                        matched_log.matched_summary(), sort_keys=True),
                }
                part_name = f"{ordinal:04d}_{alias_hash[:20]}.parquet"
                atomic_parquet(
                    parts_dir / part_name, pd.DataFrame([row]))
                state["done"][key] = {
                    "part": part_name,
                    "item_id": item["item_id"],
                    "alias": alias,
                    "elapsed_seconds": round(
                        time.time() - started, 3),
                }
                atomic_json(state_path, state)
                del clean, j_logits, matched_logits
                torch.cuda.empty_cache()
                elapsed = time.time() - started
                rate = elapsed / max(len(state["done"]), 1)
                log(
                    f"{slug} {len(state['done'])}/{expected_cells} "
                    f"alias cells; {rate:.1f}s/cell; ETA "
                    f"{(expected_cells-len(state['done']))*rate/60:.1f}m")
        state["completed_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_json(state_path, state)
        del lens, dictionaries, model, hf
        torch.cuda.empty_cache()

    if len(state["done"]) != expected_cells:
        raise RuntimeError(
            f"alias endpoint incomplete: {len(state['done'])}/"
            f"{expected_cells}")
    part_paths = [
        parts_dir / value["part"]
        for _, value in sorted(state["done"].items())
    ]
    if not all(path.exists() for path in part_paths):
        raise RuntimeError("alias endpoint state names a missing part")
    raw = pd.concat(
        [pd.read_parquet(path) for path in part_paths],
        ignore_index=True).sort_values(
            ["item_id", "alias_ordinal"]).reset_index(drop=True)
    aggregate = aggregate_alias_rows(raw)
    if len(raw) != expected_cells:
        raise RuntimeError("raw alias cell count changed at assembly")
    if len(aggregate) != len(items) * len(ENDPOINTS):
        raise RuntimeError("aggregate alias endpoint count is incomplete")
    raw_path = out_dir / f"p3_alias_endpoint_raw_{slug}.parquet"
    aggregate_path = out_dir / f"p3_alias_endpoint_aggregate_{slug}.parquet"
    atomic_parquet(raw_path, raw)
    atomic_parquet(aggregate_path, aggregate)
    replay = raw[raw.is_first_alias]
    payload = {
        "schema_version": 1,
        "slug": slug,
        "selection_id": selection["selection_id"],
        "n_facts": int(raw.fact_id.nunique()),
        "n_families": int(raw.canonical_family.nunique()),
        "n_items": int(raw.item_id.nunique()),
        "n_alias_cells": int(len(raw)),
        "n_multi_alias_facts": int(raw.loc[
            raw.accepted_aliases_json.map(
                lambda value: len(json.loads(value)) > 1),
            "fact_id"].nunique()),
        "n_prefix_overlap_pairs": int(sum(
            len(json.loads(value))
            for value in raw.drop_duplicates(
                "item_id").token_prefix_overlaps_json)),
        "replay_gate": {
            "tolerance": float(config["replay_tolerance"]),
            "max_abs_baseline_error": float(
                replay.first_alias_baseline_abs_error.max()),
            "max_abs_j_error": float(
                replay.first_alias_j_abs_error.max()),
        },
        "composition_specificity_by_endpoint":
            model_endpoint_summary(aggregate),
        "gpu": gpu,
        "state_header": header,
        "endpoint_contract": {
            "historical_first_alias": (
                "immutable frozen row, including historical control"),
            "stable_first_alias": (
                "accepted_answers[0], sha256-v1 control seed 31337"),
            "canonical_alias": (
                "literal space-prefixed bank canonical answer"),
            "prefix_disjoint_logsumexp": (
                "logsumexp over exact maximum-cardinality token-prefix-"
                "disjoint accepted subset"),
            "max_alias": "maximum accepted-alias sequence LP; diagnostic",
        },
    }
    command = (
        "python -m "
        "jspace_phase3.experiments.p3_alias_endpoint_audit "
        f"--config {config_path}")
    result_path = out_dir / f"p3_alias_endpoint_{slug}.json"
    markdown_path = out_dir / f"p3_alias_endpoint_{slug}.md"
    write_result3(
        payload, result_path,
        Provenance3(
            evidence_id=config["evidence_id"],
            tier=config.get("tier", TIER),
            command=command,
            config_path=str(config_path),
            inputs=inputs,
            model=resolve_model(str(model_path)),
            seed=int(config["control_seed"])))
    write_model_markdown(markdown_path, slug, payload)
    register(
        config["evidence_id"],
        tier=config.get("tier", TIER),
        command=command,
        what=(
            f"CUDA-only accepted-alias scoring sensitivity on {slug}: "
            f"{payload['n_facts']} common frozen facts, historical and "
            "stable first alias, canonical, prefix-disjoint logsumexp, "
            "and diagnostic max alias."),
        outputs=[result_path, markdown_path, raw_path, aggregate_path],
        inputs=inputs)
    log(
        f"CELL BANKED: {slug}, {len(raw)} alias cells, "
        f"{len(aggregate)} aggregate rows")


def run_cross_analysis() -> None:
    require_clean_tree("--allow-dirty" in sys.argv)
    frames = []
    inputs = {}
    for slug in SLUGS:
        directory = (
            metrics_dir(slug) / "release_audit" / "alias_endpoint")
        path = directory / f"p3_alias_endpoint_aggregate_{slug}.parquet"
        if not path.exists():
            raise RuntimeError(f"missing alias endpoint cell {path}")
        inputs[f"aggregate:{slug}"] = sha256_file(path)
        frames.append(pd.read_parquet(path))
        result_path = directory / f"p3_alias_endpoint_{slug}.json"
        inputs[f"result:{slug}"] = sha256_file(result_path)
    inputs["selection"] = sha256_file(DEFAULT_SELECTION)
    combined = pd.concat(frames, ignore_index=True)
    report, paired = analyze_cross_model(combined)
    report["selection_sha256"] = inputs["selection"]
    report["model_cell_sha256"] = {
        slug: inputs[f"aggregate:{slug}"] for slug in SLUGS}

    out_dir = (
        metrics_dir("cross_model") / "release_audit"
        / "alias_endpoint")
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = out_dir / "p3_alias_endpoint_cross_model.parquet"
    paired_path = out_dir / "p3_alias_endpoint_p3p1_paired.parquet"
    atomic_parquet(aggregate_path, combined)
    atomic_parquet(paired_path, paired)
    figure_png = out_dir / "p3_alias_endpoint_cross_model.png"
    figure_pdf = out_dir / "p3_alias_endpoint_cross_model.pdf"
    make_cross_figure(report, paired, figure_png, figure_pdf)
    command = (
        "python -m "
        "jspace_phase3.experiments.p3_alias_endpoint_audit --analyze")
    result_path = out_dir / "p3_alias_endpoint_cross_model.json"
    markdown_path = out_dir / "p3_alias_endpoint_cross_model.md"
    write_result3(
        report, result_path,
        Provenance3(
            evidence_id=CROSS_EVIDENCE_ID,
            tier=TIER,
            command=command,
            inputs=inputs,
            seed=SEED))
    write_cross_markdown(markdown_path, report)
    outputs = [
        result_path, markdown_path, aggregate_path, paired_path,
        figure_png, figure_pdf,
    ]
    register(
        CROSS_EVIDENCE_ID,
        tier=TIER,
        command=command,
        what=(
            "Cross-model Phase 3 accepted-alias endpoint sensitivity on "
            "the pre-outcome common 20-fact/17-family subset."),
        outputs=outputs,
        inputs=inputs)
    print(json.dumps({
        "P3-P1_subset": report["P3-P1_subset"],
        "P3-P1_alias_change_vs_stable_first": report[
            "P3-P1_alias_change_vs_stable_first"],
        "P3-P2_qwen_subset": report["P3-P2_qwen_subset"],
    }, indent=1))


def main() -> None:
    if "--analyze" in sys.argv:
        run_cross_analysis()
        return
    config = arg("--config")
    if not config:
        raise RuntimeError("pass --config or --analyze")
    run_model(Path(config))


if __name__ == "__main__":
    main()
