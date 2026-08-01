"""Fixed, resumable Phase 4 Qwen multi-lens functional gate.

This is a development-tier invariance assay, not a model-selection search.
The outcome-blind item manifest, lens order, condition order, seeds, controls,
and equivalence margins are all frozen in YAML before the compared larger
milestone is available. Raw rows are checkpointed to Drive every five items;
aggregation and the branch decision happen only after every configured lens
is complete.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Iterable, Mapping

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jspace_part2.battery import answer_variants, seq_lp_from_logits
from jspace_part2.dictionaries import build_j_dictionaries, effective_gain
from jspace_part2.occupancy import marginal_gains, occupancy_from_gains
from jspace_part2.occupancy_v2 import centered_shares, gradient_pursuit_v2
from jspace_part2.protected_dynamic import ProtectedDynamicAblator
from jspace_phase3.ablator3 import (
    Phase3JAblator,
    profile_from_p3log,
    teacher_forced_matched_arm,
)
from jspace_phase3.bank import FactBundle, load_bank
from jspace_phase3.experiments.p3_bridge_swap_endpoint_audit import (
    boundary_generation_category,
    greedy_from_prefill,
    piece_ids,
)
from jspace_phase3.scoring import DEFAULT_SPEC, ScoringSession
from jspace_phase3.seeds import stable_seed

from ..gpu import assert_model_on_cuda, require_cuda_gpu
from ..manifests import (
    atomic_json,
    file_sha256,
    object_sha256,
    require_clean_tree,
)
from ..paths4 import (
    figures_dir,
    materialize_local_file,
    metrics_dir,
    resolve_uri,
)
from ..provenance4 import Provenance4, write_result4
from ..registry4 import RegistryError, create, resolve
from .p4_qwen_lens_structural_stability import load_lens_checkpoint
from .p4_qwen_nested_lens_fit import model_reference


SUPPORTED_LENS_ORDERS = (
    ("published", "a120", "a250"),
    ("published", "a250", "a500"),
)
EXPECTED_CONDITION_ORDER = [
    "span_safe", "exact_matched", "true_bridge", "distractor_bridge",
    "counterfactual_swap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _built_in(value):
    if dataclasses.is_dataclass(value):
        return _built_in(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _built_in(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_built_in(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def validate_config(config: Mapping) -> None:
    if config.get("tier") != "phase4-development":
        raise RuntimeError("multi-lens functional gate is development only")
    lens_order = tuple(config.get("lens_order", []))
    if lens_order not in SUPPORTED_LENS_ORDERS:
        raise RuntimeError("functional lens order drift")
    if list(config["protocol"].get("condition_order", [])) \
            != EXPECTED_CONDITION_ORDER:
        raise RuntimeError("functional condition order drift")
    if config["protocol"].get("answer_alias_rule") \
            != "first_phase3_accepted_alias":
        raise RuntimeError("answer-alias contract drift")
    if config["g4"].get("condition_order") \
            != ["baseline", "swap_j", "swap_random"]:
        raise RuntimeError("G4 condition order drift")
    if config["bridge_endpoint"].get("category_contract") \
            != ["original", "counterfactual", "other-invalid"]:
        raise RuntimeError("generation category contract drift")
    primary_pair = tuple(
        config["analysis"].get("primary_pair", lens_order[1:]))
    if primary_pair != lens_order[1:]:
        raise RuntimeError("functional primary-pair drift")
    expected_pair_order = (
        primary_pair,
        (primary_pair[1], lens_order[0]),
        (primary_pair[0], lens_order[0]),
    )
    pair_order = tuple(
        tuple(value) for value in config["analysis"]["pair_order"])
    if pair_order != expected_pair_order:
        raise RuntimeError("functional comparison-order drift")
    structural_id = config["analysis"].get(
        "structural_comparison_id",
        f"{primary_pair[0]}_vs_{primary_pair[1]}",
    )
    if structural_id != f"{primary_pair[0]}_vs_{primary_pair[1]}":
        raise RuntimeError("structural comparison does not match primary pair")
    for name in lens_order:
        expected = str(config["lenses"][name].get("lens_sha256", ""))
        if len(expected) != 64 or any(c not in "0123456789abcdef"
                                      for c in expected):
            raise RuntimeError(
                f"lens {name} lacks a registered full SHA-256")
    if config["lenses"]["published"].get(
            "provenance_classification") != (
                "external published reference, partially specified recipe"):
        raise RuntimeError("published-reference classification drift")


def primary_pair(config: Mapping) -> tuple[str, str]:
    order = tuple(config["lens_order"])
    configured = tuple(config["analysis"].get("primary_pair", order[1:]))
    if configured != order[1:]:
        raise RuntimeError("functional primary-pair drift")
    return configured


def selected_span_basis(selected_rows: torch.Tensor,
                        protected_rows: torch.Tensor) -> tuple[
                            torch.Tensor, dict]:
    """Reconstruct the exact span-safe basis used by Phase3JAblator."""
    if selected_rows.ndim != 2 or protected_rows.ndim != 2:
        raise ValueError("selected/protected rows must be matrices")
    d_model = selected_rows.shape[1]
    if selected_rows.shape[0] == 0:
        return selected_rows.new_zeros((d_model, 0)), {
            "raw_rank": 0, "effective_rank": 0, "lost_rank": 0}
    selected = selected_rows.float()
    protected = protected_rows.float()
    raw_u, raw_s, _ = torch.linalg.svd(
        selected.T, full_matrices=False)
    raw_threshold = (raw_s[:1] * 1e-4).clamp_min(1e-7)
    raw_rank = int((raw_s > raw_threshold).sum().item())
    if protected.shape[0]:
        protected_u, protected_s, _ = torch.linalg.svd(
            protected.T, full_matrices=False)
        protected_threshold = (
            protected_s[:1] * 1e-4).clamp_min(1e-7)
        protected_basis = protected_u * (
            protected_s > protected_threshold).unsqueeze(0)
        coefficients = selected @ protected_basis
        selected = selected - coefficients @ protected_basis.T
    residual_u, residual_s, _ = torch.linalg.svd(
        selected.T, full_matrices=False)
    keep = residual_s > raw_threshold
    basis = residual_u[:, keep]
    effective = int(keep.sum().item())
    return basis, {
        "raw_rank": raw_rank,
        "effective_rank": effective,
        "lost_rank": max(raw_rank - effective, 0),
    }


def selection_pair_metrics(
        left_ids: list[int], right_ids: list[int],
        left_scores: list[float], right_scores: list[float],
        left_basis: torch.Tensor, right_basis: torch.Tensor) -> dict:
    left_set, right_set = set(left_ids), set(right_ids)
    union = left_set | right_set
    jaccard = len(left_set & right_set) / len(union) if union else 1.0
    if left_basis.shape[1] == 0 or right_basis.shape[1] == 0:
        overlap = 0.0
        angles = torch.tensor([90.0])
    else:
        singular = torch.linalg.svdvals(
            left_basis.T @ right_basis).clamp(0, 1)
        overlap = float((singular.square().sum()
                         / max(min(left_basis.shape[1],
                                   right_basis.shape[1]), 1)).item())
        angles = torch.rad2deg(torch.arccos(singular))
    shared = sorted(left_set & right_set)
    rank_correlation = None
    if len(shared) >= 2:
        from scipy.stats import spearmanr
        left_map = {token: score for token, score
                    in zip(left_ids, left_scores)}
        right_map = {token: score for token, score
                     in zip(right_ids, right_scores)}
        value = spearmanr(
            [left_map[token] for token in shared],
            [right_map[token] for token in shared]).statistic
        rank_correlation = None if not np.isfinite(value) else float(value)
    return {
        "selected_id_jaccard": float(jaccard),
        "normalized_projector_overlap": overlap,
        "principal_angle_median_degrees": float(angles.median().item()),
        "principal_angle_max_degrees": float(angles.max().item()),
        "selected_score_rank_correlation_shared": rank_correlation,
        "selected_score_rank_correlation_n_shared": len(shared),
    }


def pass_at_k(rank: int | None, k: int) -> bool | None:
    return None if rank is None else bool(rank <= k)


def target_rank(scores: torch.Tensor, target_ids: Iterable[int]) -> int | None:
    ids = sorted({int(value) for value in target_ids})
    if not ids:
        return None
    target = scores[torch.tensor(ids, device=scores.device)].max()
    return int((scores > target).sum().item()) + 1


def equal_family_values(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    return frame.groupby("canonical_family", sort=True)[column].mean()


def paired_family_summary(frame: pd.DataFrame, *, left: str, right: str,
                          column: str, sesoi: float,
                          draws: int, seed: int) -> dict:
    wide = frame.pivot_table(
        index="canonical_family", columns="lens", values=column,
        aggfunc="mean")
    wide = wide.dropna(subset=[left, right])
    differences = (wide[left] - wide[right]).to_numpy(dtype=np.float64)
    if len(differences) == 0:
        raise RuntimeError(f"no paired families for {left}/{right}/{column}")
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0, len(differences), size=(draws, len(differences)))
    bootstrap = differences[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    from scipy.stats import ttest_1samp
    lower = ttest_1samp(
        differences, -sesoi, alternative="greater").pvalue
    upper = ttest_1samp(
        differences, sesoi, alternative="less").pvalue
    return {
        "left": left,
        "right": right,
        "column": column,
        "n_families": int(len(differences)),
        "equal_family_mean_difference": float(differences.mean()),
        "paired_family_bootstrap_ci95": [float(low), float(high)],
        "sesoi": float(sesoi),
        "tost_p_lower": float(lower),
        "tost_p_upper": float(upper),
        "tost_p_max": float(max(lower, upper)),
        "equivalent_at_alpha_0_05": bool(max(lower, upper) < 0.05),
    }


def branch_from_gates(gates: Mapping[str, bool], *,
                      structural_stable: bool | None) -> str:
    if not all(gates.values()):
        return "B"
    if structural_stable is False:
        return "C"
    if structural_stable is True:
        return "A"
    return "PENDING_STRUCTURAL"


def teacher_forced_nll(logits: torch.Tensor, ids: torch.Tensor,
                       chunk: int = 128) -> float:
    targets = ids[0, 1:].to(logits.device)
    total = 0.0
    for start in range(0, len(targets), chunk):
        stop = min(start + chunk, len(targets))
        log_probabilities = torch.log_softmax(
            logits[start:stop].float(), dim=-1)
        rows = torch.arange(stop - start, device=logits.device)
        total += float(-log_probabilities[rows, targets[start:stop]].sum())
    return total / max(len(targets), 1)


def _json_records(log_object) -> tuple[list[dict], list[dict]]:
    overlaps = {
        (row.layer, row.position): dataclasses.asdict(row)
        for row in log_object.overlap
        if row.phase == "prefill" and row.forward_index == 0
    }
    positions = []
    for row in log_object.positions:
        if row.phase != "prefill" or row.forward_index != 0:
            continue
        record = dataclasses.asdict(row)
        overlap = overlaps.get((row.layer, row.position))
        if overlap is not None:
            record.update({
                f"overlap__{key}": value
                for key, value in overlap.items()
                if key not in {"layer", "phase", "forward_index", "position"}
            })
        positions.append(_built_in(record))
    return positions, [_built_in(dataclasses.asdict(row))
                       for row in log_object.overlap]


def _load_subset(config: Mapping) -> tuple[dict, Path]:
    specification = config["subset"]
    path = resolve_uri(specification["uri"])
    if file_sha256(path) != specification["sha256"]:
        raise RuntimeError("functional subset file hash mismatch")
    envelope = json.loads(path.read_text())
    if envelope.get("payload_sha256") != specification["payload_sha256"]:
        raise RuntimeError("functional subset payload hash mismatch")
    payload = envelope["payload"]
    if payload.get("evidence_id") != specification["evidence_id"]:
        raise RuntimeError("functional subset evidence ID mismatch")
    if payload.get("selection_is_outcome_blind") is not True:
        raise RuntimeError("functional subset is not outcome-blind")
    return payload["subset"], path


def _load_bundles(config: Mapping) -> tuple[list[FactBundle], dict]:
    bundles: list[FactBundle] = []
    contract = {}
    for specification in config["task_banks"]:
        path = resolve_uri(specification["uri"])
        actual = file_sha256(path)
        if actual != specification["sha256"]:
            raise RuntimeError(f"task-bank hash mismatch: {path}")
        rows = load_bank(path)
        bundles.extend(rows)
        contract[str(path)] = {"sha256": actual, "n_facts": len(rows)}
    fact_ids = [bundle.fact_id for bundle in bundles]
    if len(fact_ids) != len(set(fact_ids)):
        raise RuntimeError("task banks contain duplicate fact IDs")
    return bundles, contract


def _resolve_lens_paths(config: Mapping) -> tuple[dict[str, Path], dict]:
    paths, contract = {}, {}
    recipe = config["runtime"]
    for name in config["lens_order"]:
        specification = config["lenses"][name]
        expected = specification["lens_sha256"]
        if specification["kind"] == "registered":
            event = resolve(specification["evidence_id"])
            if not event["live"]:
                raise RuntimeError(f"registered lens {name} is not live")
            source = resolve_uri(
                specification["lens_uri"], must_exist=False)
            registered = {
                row["sha256"] for row in event["outputs"]
                if Path(row["path"]) == source
            }
            if registered != {expected}:
                raise RuntimeError(
                    f"lens {name} path/hash is absent from its registry event")
            path = materialize_local_file(
                specification["lens_uri"], expected_sha256=expected)
        elif specification["kind"] == "external_published":
            path = Path(specification["lens_path"])
            if file_sha256(path) != expected:
                raise RuntimeError("published lens hash mismatch")
        else:
            raise RuntimeError(f"unsupported lens kind for {name}")
        # Validate the complete tensor/schema contract without retaining it.
        checkpoint = load_lens_checkpoint(path, specification, recipe)
        del checkpoint
        paths[name] = path
        contract[name] = {**dict(specification), "local_path": str(path)}
    return paths, contract


def _new_state(header: dict, lens_order: Iterable[str]) -> dict:
    return {
        "schema_version": 1,
        "header": header,
        "baseline": {
            "primary": {}, "prose": {}, "bridge": {}, "g4": {}},
        "lenses": {
            name: {
                "primary": [], "selection": [], "readout": [],
                "prose": [], "bridge": [], "g4": [], "capacity": {},
                "complete": False,
            }
            for name in lens_order
        },
    }


def _load_or_create_state(path: Path, header: dict,
                          lens_order: Iterable[str]) -> dict:
    if not path.exists():
        state = _new_state(header, lens_order)
        atomic_json(path, state)
        return state
    state = json.loads(path.read_text())
    if state.get("header") != header:
        mismatch = {
            key: [state.get("header", {}).get(key), value]
            for key, value in header.items()
            if state.get("header", {}).get(key) != value
        }
        raise RuntimeError(
            "refusing incompatible functional-gate resume: "
            + json.dumps(mismatch, sort_keys=True))
    return state


def _checkpoint(state_path: Path, state: dict) -> None:
    atomic_json(state_path, _built_in(state))


def _primary_items(bundles: list[FactBundle], subset: Mapping) -> tuple[
        list[dict], dict[str, FactBundle]]:
    bundle_by_fact = {bundle.fact_id: bundle for bundle in bundles}
    requested = set(subset["primary"]["item_ids"])
    items = []
    for bundle in bundles:
        for item in bundle.as_items():
            if item["item_id"] in requested:
                items.append(item)
    items.sort(key=lambda row: row["item_id"])
    if {item["item_id"] for item in items} != requested:
        raise RuntimeError("functional primary IDs do not resolve in banks")
    return items, bundle_by_fact


def _checkpoint_due(count: int, every: int) -> bool:
    return count % every == 0


def _generation_category(value: str) -> str:
    return value if value in {"original", "counterfactual"} \
        else "other-invalid"


@torch.no_grad()
def _clean_alias_lp(hf, sess: ScoringSession, prompt: str,
                    alias: str) -> float:
    full, n_prompt = sess.full_ids(prompt, alias)
    logits = hf(input_ids=full, use_cache=False).logits[0].float().cpu()
    return sess.answer_seq_lp(full, logits, n_prompt)


@torch.no_grad()
def _score_g4_pair(hf, wrapped, tok, item: Mapping,
                   ablator: ProtectedDynamicAblator | None,
                   mode: dict | None) -> dict:
    result = {}
    first_original_logits = None
    for tag, answer in (("orig", item["answer"]),
                        ("swap", item["swap_answer"])):
        best = None
        for ordinal, variant in enumerate(answer_variants(answer)):
            prompt = item["prompt"].rstrip()
            ids = wrapped.encode(prompt + variant, max_length=512)
            n_prompt = wrapped.encode(prompt, max_length=512).shape[1]
            if ablator is None:
                logits = hf(
                    input_ids=ids, use_cache=False).logits[0].float().cpu()
            else:
                ablator.mode = mode
                logits = hf(
                    input_ids=ids, use_cache=False).logits[0].float().cpu()
                ablator.mode = None
            value = seq_lp_from_logits(ids, logits, n_prompt)
            best = value if best is None or value > best else best
            if tag == "orig" and ordinal == 0:
                first_original_logits = logits[n_prompt - 1]
        result[tag] = float(best)
    original_id = int(tok(
        f" {item['answer'].strip()}", add_special_tokens=False).input_ids[0])
    swap_id = int(tok(
        f" {item['swap_answer'].strip()}",
        add_special_tokens=False).input_ids[0])
    result["pick_swap"] = float(
        first_original_logits[swap_id] > first_original_logits[original_id])
    return result


@torch.no_grad()
def _prepare_baselines(
        hf, wrapped, tok, sess: ScoringSession, *,
        items: list[dict], bundle_by_fact: Mapping[str, FactBundle],
        subset: Mapping, band: list[int], capacity_layers: list[int],
        capacity_positions: list[int], state: dict, state_path: Path,
        endpoint_path: Path, capacity_input_path: Path,
        checkpoint_every: int) -> tuple[dict, dict]:
    """Cache lens-independent clean passes and fixed residual readouts."""
    from jlens.hooks import ActivationRecorder

    endpoint_cache = (
        torch.load(endpoint_path, map_location="cpu", weights_only=False)
        if endpoint_path.exists() else {})
    capacity_cache = (
        torch.load(capacity_input_path, map_location="cpu", weights_only=False)
        if capacity_input_path.exists() else {
            "done_items": [], "owner": [], "position": [],
            "H": {str(layer): [] for layer in capacity_layers},
        })
    all_record_layers = sorted(set(band) | set(capacity_layers))

    primary_done = state["baseline"]["primary"]
    progress = 0
    for item in items:
        item_id = item["item_id"]
        if item_id in primary_done and item_id in endpoint_cache:
            continue
        full, n_prompt = sess.full_ids(
            item["prompt"], item["accepted_answers"][0])
        with ActivationRecorder(wrapped.layers, at=band) as recorder:
            clean = hf(input_ids=full, use_cache=False).logits[0].float()
        protection = clean.topk(
            int(state["header"]["protect_top_k"]), dim=-1).indices
        primary_done[item_id] = {
            "lp": sess.answer_seq_lp(full, clean.cpu(), n_prompt),
            "n_prompt": int(n_prompt),
            "n_tokens": int(full.shape[1]),
            "protect_sets": protection.cpu().tolist(),
        }
        endpoint_cache[item_id] = {
            str(layer): recorder.activations[layer][
                0, n_prompt - 1].detach().to("cpu", torch.float16)
            for layer in band
        }
        progress += 1
        if _checkpoint_due(progress, checkpoint_every):
            _checkpoint(state_path, state)
            atomic_torch(endpoint_path, endpoint_cache)
            log(f"clean primary cache {len(primary_done)}/{len(items)}")
        del clean, protection
    _checkpoint(state_path, state)
    atomic_torch(endpoint_path, endpoint_cache)

    prose_rows = list(subset["prose_nll"]["items"])
    capacity_ids = set(subset["capacity"]["item_ids"])
    capacity_done = set(capacity_cache["done_items"])
    prose_done = state["baseline"]["prose"]
    progress = 0
    for item in prose_rows:
        item_id = item["item_id"]
        needs_capacity = item_id in capacity_ids and item_id not in capacity_done
        if item_id in prose_done and not needs_capacity:
            continue
        ids = sess.prompt_ids(item["text"])
        with ActivationRecorder(
                wrapped.layers, at=all_record_layers) as recorder:
            clean = hf(input_ids=ids, use_cache=False).logits[0].float()
        protection = clean.topk(
            int(state["header"]["protect_top_k"]), dim=-1).indices
        prose_done[item_id] = {
            "nll": teacher_forced_nll(clean, ids),
            "n_tokens": int(ids.shape[1]),
            "protect_sets": protection.cpu().tolist(),
            "domain": item["domain"],
        }
        if needs_capacity:
            selected_positions = [
                int(position) - 1 for position in capacity_positions
                if 0 <= int(position) - 1 < ids.shape[1] - 1
            ]
            if not selected_positions:
                raise RuntimeError(
                    f"capacity item {item_id} has no fixed eligible position")
            for position in selected_positions:
                capacity_cache["owner"].append(item_id)
                capacity_cache["position"].append(position + 1)
                for layer in capacity_layers:
                    capacity_cache["H"][str(layer)].append(
                        recorder.activations[layer][0, position].detach()
                        .to("cpu", torch.float32))
            capacity_cache["done_items"].append(item_id)
            capacity_done.add(item_id)
        progress += 1
        if _checkpoint_due(progress, checkpoint_every):
            _checkpoint(state_path, state)
            atomic_torch(capacity_input_path, capacity_cache)
            log(f"clean prose cache {len(prose_done)}/{len(prose_rows)}")
        del clean, protection
    _checkpoint(state_path, state)
    atomic_torch(capacity_input_path, capacity_cache)

    bridge_ids = list(subset["bridge"]["fact_ids"])
    bridge_done = state["baseline"]["bridge"]
    progress = 0
    for fact_id in bridge_ids:
        if fact_id in bridge_done:
            continue
        bundle = bundle_by_fact[fact_id]
        if not (bundle.counterfactual_bridge and bundle.counterfactual_answer
                and bundle.counterfactual_accepted):
            raise RuntimeError(
                f"bridge endpoint lacks counterfactual fields: {fact_id}")
        prompt = bundle.prompts["composed"]
        prompt_ids = sess.prompt_ids(prompt)
        clean = hf(input_ids=prompt_ids, use_cache=True)
        protection = clean.logits[0].topk(
            int(state["header"]["protect_top_k"]), dim=-1).indices
        generated, generated_ids = greedy_from_prefill(
            hf, tok, clean.logits[0, -1].float(), clean.past_key_values,
            prompt_length=prompt_ids.shape[1],
            max_new_tokens=int(state["header"]["bridge_max_new_tokens"]))
        grading = boundary_generation_category(
            generated, bundle.accepted_answers,
            bundle.counterfactual_accepted)
        original_lp = _clean_alias_lp(
            hf, sess, prompt, bundle.accepted_answers[0])
        counterfactual_lp = _clean_alias_lp(
            hf, sess, prompt, bundle.counterfactual_accepted[0])
        bridge_done[fact_id] = {
            "protect_sets": protection.cpu().tolist(),
            "n_prompt": int(prompt_ids.shape[1]),
            "lp_original": original_lp,
            "lp_counterfactual": counterfactual_lp,
            "preference": counterfactual_lp - original_lp,
            "greedy_text": generated,
            "greedy_token_ids": generated_ids,
            "greedy_category": _generation_category(grading["category"]),
        }
        progress += 1
        if _checkpoint_due(progress, checkpoint_every):
            _checkpoint(state_path, state)
            log(f"clean bridge cache {len(bridge_done)}/{len(bridge_ids)}")
    _checkpoint(state_path, state)

    g4_rows = list(subset["g4"]["items"])
    g4_done = state["baseline"]["g4"]
    progress = 0
    for item in g4_rows:
        name = item["name"]
        if name in g4_done:
            continue
        g4_done[name] = _score_g4_pair(
            hf, wrapped, tok, item, None, None)
        progress += 1
        if _checkpoint_due(progress, checkpoint_every):
            _checkpoint(state_path, state)
            log(f"clean G4 cache {len(g4_done)}/{len(g4_rows)}")
    _checkpoint(state_path, state)
    return endpoint_cache, capacity_cache


@torch.no_grad()
def _j_pass(hf, ablator: Phase3JAblator, ids: torch.Tensor,
            protect_sets: torch.Tensor, dictionaries: Mapping,
            *, k: int, record: bool) -> tuple[torch.Tensor, object]:
    ablator.log = type(ablator.log)()
    ablator.phase, ablator.forward_index = "prefill", 0
    ablator.mode = {
        "dicts": dictionaries, "k": k, "nonneg": True,
        "protect_sets": protect_sets, "active_phases": {"prefill"},
        "span_safe": True, "record_overlap": record,
        "record_ids": record, "answer_id": None,
    }
    with ablator:
        logits = hf(input_ids=ids, use_cache=False).logits[0].float()
    ablator.mode = None
    return logits, ablator.log


@torch.no_grad()
def _bridge_protect_pass(
        hf, ablator: Phase3JAblator, ids: torch.Tensor,
        protect_sets: torch.Tensor, extra_ids: torch.Tensor,
        dictionaries: Mapping, *, k: int) -> torch.Tensor:
    expanded = extra_ids.to(protect_sets.device).unsqueeze(0).expand(
        ids.shape[1], -1)
    combined = torch.cat([protect_sets, expanded], dim=1)
    logits, _ = _j_pass(
        hf, ablator, ids, combined, dictionaries, k=k, record=False)
    return logits


def _append_selection_rows(lens: str, item: Mapping,
                           log_object, destination: list[dict]) -> None:
    positions, _ = _json_records(log_object)
    for record in positions:
        destination.append({
            "lens": lens,
            "item_id": item["item_id"],
            "fact_id": item["fact_id"],
            "variant": item["variant"],
            "bank": item["bank"],
            "canonical_family": item["canonical_family"],
            **record,
        })


@torch.no_grad()
def _run_primary(
        lens_name: str, hf, wrapped, tok, sess: ScoringSession,
        dictionaries: Mapping, *, items: list[dict],
        bundle_by_fact: Mapping[str, FactBundle], bridge_fact_ids: set[str],
        band: list[int], k: int, seed: int, seed_namespace: str,
        state: dict, state_path: Path, checkpoint_every: int) -> None:
    cell = state["lenses"][lens_name]
    done = {row["item_id"] for row in cell["primary"]}
    ablator = Phase3JAblator(wrapped.layers, band)
    progress = 0
    for item in items:
        item_id = item["item_id"]
        if item_id in done:
            continue
        full, n_prompt = sess.full_ids(
            item["prompt"], item["accepted_answers"][0])
        baseline = state["baseline"]["primary"][item_id]
        if int(full.shape[1]) != baseline["n_tokens"] \
                or int(n_prompt) != baseline["n_prompt"]:
            raise RuntimeError(f"tokenization drift for {item_id}")
        protection = torch.tensor(
            baseline["protect_sets"], device="cuda", dtype=torch.long)
        j_logits, j_log = _j_pass(
            hf, ablator, full, protection, dictionaries, k=k, record=True)
        lp_j = sess.answer_seq_lp(full, j_logits.cpu(), n_prompt)
        _append_selection_rows(
            lens_name, item, j_log, cell["selection"])
        profile = profile_from_p3log(
            j_log, overlap_records=j_log.overlap)
        matched_logits, matched_log = teacher_forced_matched_arm(
            hf, wrapped.layers, band, dictionaries, full, profile,
            variant="instant_rank_energy_matched",
            protect_sets=protection,
            seed_base=stable_seed(seed_namespace, item_id, seed),
        )
        lp_control = sess.answer_seq_lp(full, matched_logits, n_prompt)
        row = {
            "lens": lens_name,
            "item_id": item_id,
            "fact_id": item["fact_id"],
            "variant": item["variant"],
            "bank": item["bank"],
            "canonical_family": item["canonical_family"],
            "relation_group": item["relation_group"],
            "lp_baseline": float(baseline["lp"]),
            "lp_span_safe": float(lp_j),
            "lp_exact_control": float(lp_control),
            "delta_span_safe": float(lp_j - baseline["lp"]),
            "delta_exact_control": float(lp_control - baseline["lp"]),
            "specific": float(lp_j - lp_control),
            "matched_summary_json": json.dumps(
                _built_in(matched_log.matched_summary()), sort_keys=True),
            "lp_true_bridge": None,
            "lp_distractor_bridge": None,
            "bridge_rescue": None,
        }
        if (item["fact_id"] in bridge_fact_ids
                and item["variant"] == "composed"):
            bundle = bundle_by_fact[item["fact_id"]]
            distractor = (
                bundle.counterfactual_bridge or bundle.distractor_bridge)
            if not distractor:
                raise RuntimeError(
                    f"bridge subset lacks distractor: {item['fact_id']}")
            true_ids = piece_ids(tok, bundle.bridge)
            distractor_ids = piece_ids(tok, distractor)
            true_logits = _bridge_protect_pass(
                hf, ablator, full, protection, true_ids,
                dictionaries, k=k)
            distractor_logits = _bridge_protect_pass(
                hf, ablator, full, protection, distractor_ids,
                dictionaries, k=k)
            lp_true = sess.answer_seq_lp(full, true_logits.cpu(), n_prompt)
            lp_distractor = sess.answer_seq_lp(
                full, distractor_logits.cpu(), n_prompt)
            row.update({
                "lp_true_bridge": float(lp_true),
                "lp_distractor_bridge": float(lp_distractor),
                "bridge_rescue": float(lp_true - lp_distractor),
            })
            del true_logits, distractor_logits
        cell["primary"].append(row)
        done.add(item_id)
        progress += 1
        if _checkpoint_due(progress, checkpoint_every):
            _checkpoint(state_path, state)
            log(f"{lens_name} primary {len(done)}/{len(items)}")
        del j_logits, matched_logits
    _checkpoint(state_path, state)


@torch.no_grad()
def _run_endpoint_readout(
        lens_name: str, dictionaries: Mapping, *, items: list[dict],
        bundle_by_fact: Mapping[str, FactBundle], endpoint_cache: Mapping,
        band: list[int], tok, state: dict) -> None:
    cell = state["lenses"][lens_name]
    if len(cell["readout"]) == len(items) * len(band):
        return
    cell["readout"] = []
    for layer in band:
        activations = torch.stack([
            endpoint_cache[item["item_id"]][str(layer)]
            for item in items]).to("cuda")
        scores = (activations.to(dictionaries[layer].dtype)
                  @ dictionaries[layer].T).float()
        for index, item in enumerate(items):
            bundle = bundle_by_fact[item["fact_id"]]
            answer_rank = target_rank(
                scores[index], piece_ids(tok, bundle.answer).tolist())
            bridge_rank = target_rank(
                scores[index], piece_ids(tok, bundle.bridge).tolist())
            cell["readout"].append({
                "lens": lens_name,
                "item_id": item["item_id"],
                "fact_id": item["fact_id"],
                "variant": item["variant"],
                "bank": item["bank"],
                "canonical_family": item["canonical_family"],
                "layer": int(layer),
                "answer_rank": answer_rank,
                "bridge_rank": bridge_rank,
                **{
                    f"answer_pass_at_{cutoff}": pass_at_k(
                        answer_rank, cutoff)
                    for cutoff in (1, 5, 20)
                },
                **{
                    f"bridge_pass_at_{cutoff}": pass_at_k(
                        bridge_rank, cutoff)
                    for cutoff in (1, 5, 20)
                },
            })
        del activations, scores


@torch.no_grad()
def _run_prose(
        lens_name: str, hf, wrapped, sess: ScoringSession,
        dictionaries: Mapping, *, prose_items: list[dict], band: list[int],
        k: int, seed: int, seed_namespace: str, state: dict,
        state_path: Path, checkpoint_every: int) -> None:
    cell = state["lenses"][lens_name]
    done = {row["item_id"] for row in cell["prose"]}
    ablator = Phase3JAblator(wrapped.layers, band)
    progress = 0
    for item in prose_items:
        item_id = item["item_id"]
        if item_id in done:
            continue
        ids = sess.prompt_ids(item["text"])
        baseline = state["baseline"]["prose"][item_id]
        if int(ids.shape[1]) != baseline["n_tokens"]:
            raise RuntimeError(f"prose tokenization drift for {item_id}")
        protection = torch.tensor(
            baseline["protect_sets"], device="cuda", dtype=torch.long)
        j_logits, j_log = _j_pass(
            hf, ablator, ids, protection, dictionaries, k=k, record=True)
        j_nll = teacher_forced_nll(j_logits, ids)
        profile = profile_from_p3log(
            j_log, overlap_records=j_log.overlap)
        matched_logits, matched_log = teacher_forced_matched_arm(
            hf, wrapped.layers, band, dictionaries, ids, profile,
            variant="instant_rank_energy_matched",
            protect_sets=protection,
            seed_base=stable_seed(seed_namespace, item_id, seed),
            return_cpu=False,
        )
        control_nll = teacher_forced_nll(matched_logits, ids)
        cell["prose"].append({
            "lens": lens_name,
            "item_id": item_id,
            "domain": item["domain"],
            "n_tokens": int(ids.shape[1]),
            "nll_baseline": float(baseline["nll"]),
            "nll_span_safe": float(j_nll),
            "nll_exact_control": float(control_nll),
            "delta_nll_span_safe": float(j_nll - baseline["nll"]),
            "delta_nll_exact_control": float(
                control_nll - baseline["nll"]),
            "specific_nll": float(j_nll - control_nll),
            "matched_summary_json": json.dumps(
                _built_in(matched_log.matched_summary()), sort_keys=True),
        })
        done.add(item_id)
        progress += 1
        if _checkpoint_due(progress, checkpoint_every):
            _checkpoint(state_path, state)
            log(f"{lens_name} prose {len(done)}/{len(prose_items)}")
        del j_logits, matched_logits
    _checkpoint(state_path, state)


@torch.no_grad()
def _score_prefix_alias(
        hf, sess: ScoringSession, ablator: Phase3JAblator,
        dictionaries: Mapping, *, prompt: str, alias: str,
        prompt_protection: torch.Tensor, restrict: torch.Tensor,
        inject: Mapping[int, torch.Tensor], band: list[int], k: int) -> float:
    full, n_prompt = sess.full_ids(prompt, alias)
    if n_prompt != prompt_protection.shape[0]:
        raise RuntimeError("bridge prompt tokenization drift")
    suffix = prompt_protection[-1:].expand(
        full.shape[1] - n_prompt, -1)
    protection = torch.cat([prompt_protection, suffix], dim=0)
    ablator.log = type(ablator.log)()
    ablator.phase, ablator.forward_index = "prefill", 0
    ablator.mode = {
        "dicts": dictionaries, "k": k, "nonneg": True,
        "protect_sets": protection, "active_phases": {"prefill"},
        "active_position_limit": n_prompt,
        "span_safe": True, "record_overlap": False,
        "answer_id": None, "restrict_sets": restrict,
        "inject_dir": inject,
    }
    with ablator:
        logits = hf(input_ids=full, use_cache=False).logits[0].float().cpu()
    ablator.mode = None
    return sess.answer_seq_lp(full, logits, n_prompt)


@torch.no_grad()
def _run_bridge_endpoint(
        lens_name: str, hf, wrapped, tok, sess: ScoringSession,
        dictionaries: Mapping, *, bridge_fact_ids: list[str],
        bundle_by_fact: Mapping[str, FactBundle], band: list[int], k: int,
        max_new_tokens: int, state: dict, state_path: Path,
        checkpoint_every: int) -> None:
    cell = state["lenses"][lens_name]
    done = {row["fact_id"] for row in cell["bridge"]}
    ablator = Phase3JAblator(wrapped.layers, band)
    progress = 0
    for fact_id in bridge_fact_ids:
        if fact_id in done:
            continue
        bundle = bundle_by_fact[fact_id]
        baseline = state["baseline"]["bridge"][fact_id]
        prompt = bundle.prompts["composed"]
        prompt_ids = sess.prompt_ids(prompt)
        protection = torch.tensor(
            baseline["protect_sets"], device="cuda", dtype=torch.long)
        true_ids = piece_ids(tok, bundle.bridge).to("cuda")
        counterfactual_ids = piece_ids(
            tok, bundle.counterfactual_bridge).to("cuda")
        injection = {
            layer: dictionaries[layer][counterfactual_ids].float().mean(0)
            for layer in band
        }
        original_lp = _score_prefix_alias(
            hf, sess, ablator, dictionaries, prompt=prompt,
            alias=bundle.accepted_answers[0],
            prompt_protection=protection, restrict=true_ids,
            inject=injection, band=band, k=k)
        counterfactual_lp = _score_prefix_alias(
            hf, sess, ablator, dictionaries, prompt=prompt,
            alias=bundle.counterfactual_accepted[0],
            prompt_protection=protection, restrict=true_ids,
            inject=injection, band=band, k=k)
        ablator.log = type(ablator.log)()
        ablator.phase, ablator.forward_index = "prefill", 0
        ablator.mode = {
            "dicts": dictionaries, "k": k, "nonneg": True,
            "protect_sets": protection, "active_phases": {"prefill"},
            "span_safe": True, "record_overlap": False,
            "answer_id": None, "restrict_sets": true_ids,
            "inject_dir": injection,
        }
        with ablator:
            output = hf(input_ids=prompt_ids, use_cache=True)
        ablator.mode = None
        generated, generated_ids = greedy_from_prefill(
            hf, tok, output.logits[0, -1].float(), output.past_key_values,
            prompt_length=prompt_ids.shape[1],
            max_new_tokens=max_new_tokens)
        grading = boundary_generation_category(
            generated, bundle.accepted_answers,
            bundle.counterfactual_accepted)
        cell["bridge"].append({
            "lens": lens_name,
            "fact_id": fact_id,
            "bank": bundle.bank,
            "canonical_family": bundle.canonical_family,
            "lp_original_baseline": baseline["lp_original"],
            "lp_counterfactual_baseline": baseline["lp_counterfactual"],
            "preference_baseline": baseline["preference"],
            "lp_original_counterfactual_swap": float(original_lp),
            "lp_counterfactual_counterfactual_swap": float(
                counterfactual_lp),
            "preference_counterfactual_swap": float(
                counterfactual_lp - original_lp),
            "preference_shift": float(
                counterfactual_lp - original_lp - baseline["preference"]),
            "greedy_baseline_category": baseline["greedy_category"],
            "greedy_counterfactual_swap_category": _generation_category(
                grading["category"]),
            "greedy_counterfactual_swap_text": generated,
            "greedy_counterfactual_swap_token_ids_json": json.dumps(
                generated_ids),
        })
        done.add(fact_id)
        progress += 1
        if _checkpoint_due(progress, checkpoint_every):
            _checkpoint(state_path, state)
            log(f"{lens_name} bridge {len(done)}/{len(bridge_fact_ids)}")
        del output, injection
    _checkpoint(state_path, state)


@torch.no_grad()
def _run_g4(
        lens_name: str, hf, wrapped, tok, dictionaries: Mapping, *,
        g4_items: list[dict], band: list[int], alpha: float,
        random_seed: int, state: dict, state_path: Path,
        checkpoint_every: int) -> None:
    cell = state["lenses"][lens_name]
    done = {row["item"] for row in cell["g4"]}
    ablator = ProtectedDynamicAblator(wrapped.layers, band)
    generator = torch.Generator(device="cpu").manual_seed(random_seed)
    random_direction = torch.nn.functional.normalize(
        torch.randn(wrapped.d_model, generator=generator), dim=0).to("cuda")

    def first_id(text: str) -> int:
        return int(tok(
            f" {text.strip()}", add_special_tokens=False).input_ids[0])

    def mode_for(item: Mapping, *, random: bool) -> dict:
        remove, inject = {}, {}
        for layer in band:
            direction = dictionaries[layer][
                first_id(item["intermediate"])].float()
            remove[layer] = torch.nn.functional.normalize(
                direction, dim=0).reshape(-1, 1)
            inject[layer] = (
                random_direction if random else dictionaries[layer][
                    first_id(item["swap_to"])].float())
        return {"inject": inject, "remove": remove, "alpha_rel": alpha}

    progress = 0
    with ablator:
        for ordinal, item in enumerate(g4_items):
            name = item["name"]
            if name in done:
                continue
            swap_j = _score_g4_pair(
                hf, wrapped, tok, item, ablator,
                mode_for(item, random=False))
            swap_random = _score_g4_pair(
                hf, wrapped, tok, item, ablator,
                mode_for(item, random=True))
            cell["g4"].append({
                "lens": lens_name,
                "item": name,
                "ordinal": int(ordinal),
                "is_calibration": bool(
                    ordinal < int(state["header"]["g4_calibration_n"])),
                "baseline_json": json.dumps(
                    state["baseline"]["g4"][name], sort_keys=True),
                "swap_j_json": json.dumps(swap_j, sort_keys=True),
                "swap_random_json": json.dumps(swap_random, sort_keys=True),
                "baseline_pick_swap": float(
                    state["baseline"]["g4"][name]["pick_swap"]),
                "swap_j_pick_swap": float(swap_j["pick_swap"]),
                "swap_random_pick_swap": float(
                    swap_random["pick_swap"]),
                "swap_j_lp_original": float(swap_j["orig"]),
                "swap_j_lp_counterfactual": float(swap_j["swap"]),
                "swap_random_lp_original": float(swap_random["orig"]),
                "swap_random_lp_counterfactual": float(
                    swap_random["swap"]),
            })
            done.add(name)
            progress += 1
            if _checkpoint_due(progress, checkpoint_every):
                _checkpoint(state_path, state)
                log(f"{lens_name} G4 {len(done)}/{len(g4_items)}")
    _checkpoint(state_path, state)


def _centered_r2_numpy(h: np.ndarray, reconstruction: np.ndarray) -> float:
    centered_h = h - h.mean(axis=0, keepdims=True)
    centered_r = reconstruction - reconstruction.mean(
        axis=0, keepdims=True)
    denominator = float(np.square(centered_h).sum())
    if denominator <= 0:
        return float("nan")
    return float(1.0 - np.square(centered_h - centered_r).sum()
                 / denominator)


def _capacity_bootstrap(
        h: torch.Tensor, reconstruction_j: torch.Tensor,
        reconstruction_random: list[torch.Tensor], owners: list[str], *,
        draws: int, seed: int) -> dict:
    h_np = h.detach().cpu().numpy().astype(np.float32, copy=False)
    j_np = reconstruction_j.detach().cpu().numpy().astype(
        np.float32, copy=False)
    random_np = [value.detach().cpu().numpy().astype(
        np.float32, copy=False) for value in reconstruction_random]
    unique = sorted(set(owners))
    owner_array = np.asarray(owners, dtype=object)
    indices_by_owner = {
        owner: np.flatnonzero(owner_array == owner) for owner in unique}
    generator = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = generator.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([indices_by_owner[value]
                                  for value in sampled])
        j_score = _centered_r2_numpy(h_np[indices], j_np[indices])
        random_score = np.mean([
            _centered_r2_numpy(h_np[indices], value[indices])
            for value in random_np
        ])
        values[draw] = j_score - random_score
    low, high = np.nanquantile(values, [0.025, 0.975])
    return {
        "n_bootstrap": int(draws),
        "resampling_unit": "prompt",
        "ci95": [float(low), float(high)],
        "bootstrap_mean": float(np.nanmean(values)),
    }


@torch.no_grad()
def _capacity_layer(
        h: torch.Tensor, dictionary_j: torch.Tensor,
        random_dictionaries: list[torch.Tensor], *, owners: list[str],
        k_max: int, persistence: int,
        persistence_sensitivity: list[int], bootstrap_draws: int,
        bootstrap_seed: int) -> tuple[dict, dict]:
    h = h.to("cuda", torch.float32)
    mean = h.mean(dim=0)
    pursuit_j = gradient_pursuit_v2(
        h, dictionary_j, k_max, keep_recons=True)
    gains_j = marginal_gains(pursuit_j.errs)
    pursuits_random = [
        gradient_pursuit_v2(h, dictionary, k_max, keep_recons=True)
        for dictionary in random_dictionaries
    ]
    gains_random = [marginal_gains(value.errs)
                    for value in pursuits_random]
    median_random_gain = torch.median(
        torch.stack(gains_random), dim=0).values
    occupancy = occupancy_from_gains(
        gains_j, median_random_gain, persistence=persistence)
    median_k = int(occupancy.median().item())
    reconstruction_j = pursuit_j.recons_by_k[median_k]
    reconstructions_random = [
        value.recons_by_k[median_k] for value in pursuits_random]
    share_j = centered_shares(h, reconstruction_j, mean)
    shares_random = [
        centered_shares(h, value, mean)
        for value in reconstructions_random]

    def random_mean(key: str) -> float:
        return float(np.mean([value[key] for value in shares_random]))

    metrics = {
        "n_positions": int(h.shape[0]),
        "n_prompts": int(len(set(owners))),
        "occupancy_median": float(occupancy.float().median()),
        "occupancy_q25": float(occupancy.float().quantile(0.25)),
        "occupancy_q75": float(occupancy.float().quantile(0.75)),
        "occupancy_censored_fraction": float(
            (occupancy >= k_max).float().mean()),
        "occupancy_histogram": torch.bincount(
            occupancy, minlength=k_max + 1).cpu().tolist(),
        "occupancy_persistence_sensitivity": {
            str(value): int(occupancy_from_gains(
                gains_j, median_random_gain, persistence=value)
                .median().item())
            for value in persistence_sensitivity
        },
        "centered_variance_explained_excess": float(
            share_j["centered_r2_B"] - random_mean("centered_r2_B")),
        "centered_variance_share_excess_A": float(
            share_j["centered_variance_share_A"]
            - random_mean("centered_variance_share_A")),
        "raw_reconstruction_excess": float(
            share_j["raw_energy_share"] - random_mean("raw_energy_share")),
        "centered_r2_j": float(share_j["centered_r2_B"]),
        "centered_r2_random": random_mean("centered_r2_B"),
        "raw_share_j": float(share_j["raw_energy_share"]),
        "raw_share_random": random_mean("raw_energy_share"),
        "achieved_support_mean_j": float(
            pursuit_j.achieved_support.float().mean()),
        "rows_exhausted_before_kmax": int(
            (pursuit_j.achieved_support < k_max).sum()),
        "marginal_gain_j_mean": gains_j.mean(0).cpu().tolist(),
        "marginal_gain_random_median_mean": (
            median_random_gain.mean(0).cpu().tolist()),
    }
    metrics["paired_prompt_bootstrap"] = _capacity_bootstrap(
        h, reconstruction_j, reconstructions_random, owners,
        draws=bootstrap_draws, seed=bootstrap_seed)
    reconstruction_payload = {
        "H": h.cpu().to(torch.float16),
        "J": reconstruction_j.cpu().to(torch.float16),
        "random": [value.cpu().to(torch.float16)
                   for value in reconstructions_random],
        "owners": list(owners),
        "occupancy_median": median_k,
    }
    return metrics, reconstruction_payload


@torch.no_grad()
def _run_capacity(
        lens_name: str, dictionaries: Mapping[int, torch.Tensor], *,
        capacity_cache: Mapping, specification: Mapping,
        artifact_path: Path, state: dict, state_path: Path) -> None:
    cell = state["lenses"][lens_name]
    layers = [int(value) for value in specification["layers"]]
    if set(cell["capacity"]) == {str(layer) for layer in layers} \
            and artifact_path.exists():
        return
    owners = list(capacity_cache["owner"])
    if not owners:
        raise RuntimeError("capacity activation cache is empty")
    first = dictionaries[layers[0]]
    vocab_size, d_model = first.shape
    random_dictionaries = []
    for seed in specification["random_seeds"]:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        cpu = torch.randn(
            (vocab_size, d_model), generator=generator,
            dtype=torch.float32)
        cpu = torch.nn.functional.normalize(cpu, dim=1)
        random_dictionaries.append(cpu.to("cuda", torch.float16))
        del cpu
    reconstruction_payload = {
        "schema_version": 1, "lens": lens_name, "layers": {}}
    for layer in layers:
        h = torch.stack(capacity_cache["H"][str(layer)])
        metrics, reconstructions = _capacity_layer(
            h, dictionaries[layer], random_dictionaries,
            owners=owners,
            k_max=int(specification["k_max"]),
            persistence=int(specification["persistence"]),
            persistence_sensitivity=[int(value) for value in
                                     specification[
                                         "persistence_sensitivity"]],
            bootstrap_draws=int(specification["bootstrap_draws"]),
            bootstrap_seed=int(specification["bootstrap_seed"]),
        )
        cell["capacity"][str(layer)] = metrics
        reconstruction_payload["layers"][str(layer)] = reconstructions
        _checkpoint(state_path, state)
        log(f"{lens_name} capacity L{layer} complete")
    atomic_torch(artifact_path, reconstruction_payload)
    del random_dictionaries
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def _partial_dictionary_rows(
        hf, gain: torch.Tensor, jacobian: torch.Tensor,
        token_ids: list[int], *, chunk: int = 1024) -> torch.Tensor:
    if not token_ids:
        return torch.empty((0, jacobian.shape[0]), device="cuda",
                           dtype=torch.float16)
    weight = hf.get_output_embeddings().weight.detach()
    ids = torch.tensor(token_ids, device=weight.device, dtype=torch.long)
    operator = jacobian.to("cuda", torch.float32)
    result = torch.empty(
        (len(ids), operator.shape[0]), device="cuda", dtype=torch.float16)
    for start in range(0, len(ids), chunk):
        stop = min(start + chunk, len(ids))
        rows = weight.index_select(0, ids[start:stop]).float()
        rows = (rows * gain.unsqueeze(0)) @ operator
        result[start:stop] = torch.nn.functional.normalize(
            rows, dim=1).to(torch.float16)
        del rows
    del operator
    return result


@torch.no_grad()
def _selection_geometry(
        hf, *, lens_paths: Mapping[str, Path], config: Mapping,
        state: Mapping, output_path: Path) -> pd.DataFrame:
    if output_path.exists():
        return pd.read_parquet(output_path)
    band = [int(value) for value in config["protocol"]["band"]]
    checkpoints = {
        name: load_lens_checkpoint(
            lens_paths[name], config["lenses"][name], config["runtime"])
        for name in config["lens_order"]
    }
    gain = effective_gain(hf).to("cuda", torch.float32)
    all_rows = []
    by_lens_layer = {}
    for lens in config["lens_order"]:
        for record in state["lenses"][lens]["selection"]:
            by_lens_layer.setdefault((lens, int(record["layer"])), {})[
                (record["item_id"], int(record["position"]))] = record

    for layer in band:
        key_sets = [
            set(by_lens_layer[(lens, layer)])
            for lens in config["lens_order"]
        ]
        keys = sorted(set.intersection(*key_sets))
        direction_tables, offsets = {}, {}
        for lens in config["lens_order"]:
            token_ids = set()
            records = by_lens_layer[(lens, layer)]
            for item_id, position in keys:
                token_ids.update(
                    int(value) for value in
                    (records[(item_id, position)].get("selected_ids") or []))
                token_ids.update(
                    int(value) for value in state["baseline"]["primary"][
                        item_id]["protect_sets"][position])
            ordered = sorted(token_ids)
            offsets[lens] = {
                token_id: index for index, token_id in enumerate(ordered)}
            direction_tables[lens] = _partial_dictionary_rows(
                hf, gain, checkpoints[lens]["J"][layer], ordered)

        for item_id, position in keys:
            bases, records = {}, {}
            for lens in config["lens_order"]:
                record = by_lens_layer[(lens, layer)][(item_id, position)]
                selected_ids = [int(value) for value in
                                (record.get("selected_ids") or [])]
                protected_ids = [int(value) for value in
                                 state["baseline"]["primary"][item_id][
                                     "protect_sets"][position]]
                selected_indices = torch.tensor(
                    [offsets[lens][value] for value in selected_ids],
                    device="cuda", dtype=torch.long)
                protected_indices = torch.tensor(
                    [offsets[lens][value] for value in protected_ids],
                    device="cuda", dtype=torch.long)
                basis, basis_metadata = selected_span_basis(
                    direction_tables[lens].index_select(
                        0, selected_indices),
                    direction_tables[lens].index_select(
                        0, protected_indices),
                )
                bases[lens] = basis
                records[lens] = {**record, **basis_metadata}
            for left, right in config["analysis"]["pair_order"]:
                left_record, right_record = records[left], records[right]
                pair = selection_pair_metrics(
                    left_record.get("selected_ids") or [],
                    right_record.get("selected_ids") or [],
                    left_record.get("selected_scores") or [],
                    right_record.get("selected_scores") or [],
                    bases[left], bases[right],
                )
                all_rows.append({
                    "comparison": f"{left}_vs_{right}",
                    "left_lens": left,
                    "right_lens": right,
                    "item_id": item_id,
                    "fact_id": left_record["fact_id"],
                    "variant": left_record["variant"],
                    "bank": left_record["bank"],
                    "canonical_family": left_record["canonical_family"],
                    "layer": int(layer),
                    "position": int(position),
                    **pair,
                    "left_effective_rank": int(
                        left_record["effective_rank"]),
                    "right_effective_rank": int(
                        right_record["effective_rank"]),
                    "left_removed_energy": float(
                        left_record["removed_energy_frac"]),
                    "right_removed_energy": float(
                        right_record["removed_energy_frac"]),
                    "left_protected_overlap": left_record.get(
                        "overlap__overlap_normalized"),
                    "right_protected_overlap": right_record.get(
                        "overlap__overlap_normalized"),
                    "left_lost_rank": left_record.get(
                        "overlap__lost_rank"),
                    "right_lost_rank": right_record.get(
                        "overlap__lost_rank"),
                })
            del bases
        del direction_tables
        torch.cuda.empty_cache()
        log(f"selection geometry L{layer} complete")
    frame = pd.DataFrame(all_rows)
    atomic_parquet(output_path, frame)
    del checkpoints
    return frame


def _paired_capacity_difference(
        left_path: Path, right_path: Path, *, layer: int,
        draws: int, seed: int) -> dict:
    left = torch.load(
        left_path, map_location="cpu", weights_only=False)["layers"][str(layer)]
    right = torch.load(
        right_path, map_location="cpu", weights_only=False)["layers"][str(layer)]
    if left["owners"] != right["owners"]:
        raise RuntimeError("capacity owner order differs across lenses")
    owners = left["owners"]
    unique = sorted(set(owners))
    owner_array = np.asarray(owners, dtype=object)
    indices_by_owner = {
        owner: np.flatnonzero(owner_array == owner) for owner in unique}
    arrays = {}
    for name, payload in (("left", left), ("right", right)):
        arrays[name] = {
            "H": payload["H"].float().numpy(),
            "J": payload["J"].float().numpy(),
            "random": [value.float().numpy()
                       for value in payload["random"]],
        }

    def excess(name: str, indices: np.ndarray) -> float:
        payload = arrays[name]
        return _centered_r2_numpy(
            payload["H"][indices], payload["J"][indices]) - float(np.mean([
                _centered_r2_numpy(
                    payload["H"][indices], value[indices])
                for value in payload["random"]
            ]))

    generator = np.random.default_rng(seed)
    differences = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sample = generator.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([indices_by_owner[value]
                                  for value in sample])
        differences[draw] = excess("left", indices) - excess(
            "right", indices)
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "layer": int(layer),
        "n_prompts": len(unique),
        "n_bootstrap": int(draws),
        "paired_prompt_bootstrap_difference_ci95": [
            float(low), float(high)],
        "paired_prompt_bootstrap_difference_mean": float(
            differences.mean()),
    }


def structural_metrics(result: Mapping, config: Mapping) -> dict:
    left, right = primary_pair(config)
    comparison_id = config["analysis"].get(
        "structural_comparison_id", f"{left}_vs_{right}")
    band = [int(value) for value in config["protocol"]["band"]]
    assay_key = config["analysis"].get(
        "structural_assay_key", f"assay_L{min(band)}_L{max(band)}")
    try:
        assay = result["aggregate"][comparison_id][assay_key]
    except KeyError as error:
        raise RuntimeError(
            "structural result lacks the configured comparison/assay") \
            from error
    task_strata = [
        "task_answer_only", "task_bridge_only",
        "task_answer_bridge_shared",
    ]
    q50 = min(
        float(assay[f"token_{name}_direction_cosine_q50"]["median"])
        for name in task_strata)
    q05 = min(
        float(assay[f"token_{name}_direction_cosine_q05"]["median"])
        for name in task_strata)
    return {
        "comparison_id": comparison_id,
        "assay_key": assay_key,
        "assay_task_token_median_cosine_conservative": q50,
        "task_token_q05_conservative": q05,
    }


def _structural_gate(config: Mapping) -> tuple[dict, bool | None]:
    evidence_id = config["analysis"]["structural_evidence_id"]
    try:
        event = resolve(evidence_id)
    except RegistryError:
        return {"evidence_id": evidence_id, "status": "not-registered"}, None
    if not event["live"]:
        return {"evidence_id": evidence_id, "status": "not-live"}, None
    candidates = [
        row for row in event["outputs"]
        if Path(row["path"]).name == "convergence_result.json"
    ]
    if len(candidates) != 1:
        raise RuntimeError("structural event lacks one convergence result")
    output = candidates[0]
    path = Path(output["path"])
    if file_sha256(path) != output["sha256"]:
        raise RuntimeError("structural result hash mismatch")
    result = json.loads(path.read_text())["payload"]
    metrics = structural_metrics(result, config)
    q50 = metrics["assay_task_token_median_cosine_conservative"]
    q05 = metrics["task_token_q05_conservative"]
    thresholds = config["analysis"]["thresholds"]
    gates = {
        "assay_task_token_median_cosine": bool(
            q50 >= thresholds["assay_task_token_median_cosine_min"]),
        "task_token_q05": bool(
            q05 >= thresholds["task_token_q05_min"]),
    }
    return {
        "evidence_id": evidence_id,
        "status": "verified-live",
        **metrics,
        "gates": gates,
        "all_structural_gates_pass": bool(all(gates.values())),
    }, bool(all(gates.values()))


def _lens_summaries(state: Mapping, config: Mapping) -> dict:
    summaries = {}
    threshold = float(config["protocol"]["tail_threshold_nats"])
    for lens in config["lens_order"]:
        cell = state["lenses"][lens]
        primary = pd.DataFrame(cell["primary"])
        selection = pd.DataFrame(cell["selection"])
        readout = pd.DataFrame(cell["readout"])
        prose = pd.DataFrame(cell["prose"])
        bridge = pd.DataFrame(cell["bridge"])
        g4 = pd.DataFrame(cell["g4"])
        measured = g4[~g4["is_calibration"]]
        j_flip = float(measured["swap_j_pick_swap"].mean())
        random_flip = float(measured["swap_random_pick_swap"].mean())
        rescue = primary[primary["bridge_rescue"].notna()]
        generation_counts = (
            bridge["greedy_counterfactual_swap_category"]
            .value_counts().reindex(
                ["original", "counterfactual", "other-invalid"],
                fill_value=0))
        summaries[lens] = {
            "primary": {
                "n_items": int(len(primary)),
                "n_facts": int(primary["fact_id"].nunique()),
                "n_families": int(primary["canonical_family"].nunique()),
                "span_safe_delta_equal_family_mean": float(
                    equal_family_values(primary, "delta_span_safe").mean()),
                "exact_control_delta_equal_family_mean": float(
                    equal_family_values(
                        primary, "delta_exact_control").mean()),
                "specific_equal_family_mean": float(
                    equal_family_values(primary, "specific").mean()),
                "tail_rate_span_safe": float(
                    (primary["delta_span_safe"] < threshold).mean()),
                "tail_rate_specific": float(
                    (primary["specific"] < threshold).mean()),
            },
            "readout": {
                key: float(readout[key].astype(float).mean())
                for key in [
                    "answer_pass_at_1", "answer_pass_at_5",
                    "answer_pass_at_20", "bridge_pass_at_1",
                    "bridge_pass_at_5", "bridge_pass_at_20",
                ]
            },
            "selection": {
                "effective_rank_median": float(
                    selection["effective_rank"].median()),
                "removed_energy_median": float(
                    selection["removed_energy_frac"].median()),
                "protected_span_overlap_median": float(
                    selection["overlap__overlap_normalized"].median()),
                "lost_rank_median": float(
                    selection["overlap__lost_rank"].median()),
            },
            "capacity": _built_in(cell["capacity"]),
            "g4": {
                "alpha": float(config["g4"]["alpha"]),
                "n_measured": int(len(measured)),
                "flip_rate_j": j_flip,
                "flip_rate_random": random_flip,
                "passes": bool(
                    j_flip >= config["g4"]["pass_min_flip_rate"]
                    and j_flip - random_flip
                    >= config["g4"]["pass_min_margin_over_random"]),
            },
            "bridge": {
                "n_facts": int(len(bridge)),
                "true_vs_distractor_rescue_equal_family_mean": float(
                    equal_family_values(rescue, "bridge_rescue").mean()),
                "counterfactual_preference_equal_family_mean": float(
                    equal_family_values(
                        bridge, "preference_counterfactual_swap").mean()),
                "counterfactual_preference_shift_equal_family_mean": float(
                    equal_family_values(bridge, "preference_shift").mean()),
                "generation_trichotomy": {
                    name: {
                        "count": int(generation_counts[name]),
                        "rate": float(generation_counts[name] / len(bridge)),
                    }
                    for name in generation_counts.index
                },
            },
            "prose": {
                "n_items": int(len(prose)),
                "span_safe_nll_delta_mean": float(
                    prose["delta_nll_span_safe"].mean()),
                "exact_control_nll_delta_mean": float(
                    prose["delta_nll_exact_control"].mean()),
                "specific_nll_mean": float(prose["specific_nll"].mean()),
                "by_domain": {
                    domain: {
                        "n": int(len(group)),
                        "span_safe_nll_delta_mean": float(
                            group["delta_nll_span_safe"].mean()),
                        "exact_control_nll_delta_mean": float(
                            group["delta_nll_exact_control"].mean()),
                    }
                    for domain, group in prose.groupby("domain", sort=True)
                },
            },
        }
    return summaries


def _pair_summaries(
        state: Mapping, selection_pairs: pd.DataFrame,
        capacity_paths: Mapping[str, Path], config: Mapping,
        lens_summaries: Mapping) -> dict:
    primary = pd.concat([
        pd.DataFrame(state["lenses"][lens]["primary"])
        for lens in config["lens_order"]
    ], ignore_index=True)
    results = {}
    thresholds = config["analysis"]["thresholds"]
    for ordinal, (left, right) in enumerate(config["analysis"]["pair_order"]):
        comparison = f"{left}_vs_{right}"
        geometry = selection_pairs[
            selection_pairs["comparison"] == comparison]
        specific = paired_family_summary(
            primary[primary["lens"].isin([left, right])],
            left=left, right=right, column="specific",
            sesoi=float(thresholds[
                "span_safe_specific_mean_difference_nats_max"]),
            draws=int(config["analysis"]["family_bootstrap_draws"]),
            seed=int(config["analysis"]["family_bootstrap_seed"]) + ordinal,
        )
        capacity = {}
        for layer in config["capacity"]["layers"]:
            left_metrics = lens_summaries[left]["capacity"][str(layer)]
            right_metrics = lens_summaries[right]["capacity"][str(layer)]
            capacity[str(layer)] = {
                "occupancy_difference": float(
                    left_metrics["occupancy_median"]
                    - right_metrics["occupancy_median"]),
                "centered_excess_difference": float(
                    left_metrics["centered_variance_explained_excess"]
                    - right_metrics["centered_variance_explained_excess"]),
                "raw_excess_difference": float(
                    left_metrics["raw_reconstruction_excess"]
                    - right_metrics["raw_reconstruction_excess"]),
                **_paired_capacity_difference(
                    capacity_paths[left], capacity_paths[right],
                    layer=int(layer),
                    draws=int(config["capacity"]["bootstrap_draws"]),
                    seed=int(config["capacity"]["bootstrap_seed"]) + ordinal,
                ),
            }
        left_primary = primary[primary["lens"] == left]
        right_primary = primary[primary["lens"] == right]
        threshold = float(config["protocol"]["tail_threshold_nats"])
        results[comparison] = {
            "left": left,
            "right": right,
            "selection_geometry": {
                "n_positions": int(len(geometry)),
                "selected_id_jaccard_median": float(
                    geometry["selected_id_jaccard"].median()),
                "normalized_projector_overlap_median": float(
                    geometry["normalized_projector_overlap"].median()),
                "principal_angle_median_degrees": float(
                    geometry["principal_angle_median_degrees"].median()),
                "principal_angle_max_degrees": float(
                    geometry["principal_angle_max_degrees"].max()),
                "selected_score_rank_correlation_median": float(
                    geometry[
                        "selected_score_rank_correlation_shared"].median()),
            },
            "specific": specific,
            "tail_rate_difference": float(
                (left_primary["delta_span_safe"] < threshold).mean()
                - (right_primary["delta_span_safe"] < threshold).mean()),
            "g4_flip_rate_difference": float(
                lens_summaries[left]["g4"]["flip_rate_j"]
                - lens_summaries[right]["g4"]["flip_rate_j"]),
            "bridge_rescue_difference": float(
                lens_summaries[left]["bridge"][
                    "true_vs_distractor_rescue_equal_family_mean"]
                - lens_summaries[right]["bridge"][
                    "true_vs_distractor_rescue_equal_family_mean"]),
            "bridge_preference_difference": float(
                lens_summaries[left]["bridge"][
                    "counterfactual_preference_equal_family_mean"]
                - lens_summaries[right]["bridge"][
                    "counterfactual_preference_equal_family_mean"]),
            "capacity": capacity,
        }
    return results


def _functional_gates(lens: Mapping, pairs: Mapping,
                      config: Mapping) -> dict[str, bool]:
    thresholds = config["analysis"]["thresholds"]
    left, right = primary_pair(config)
    pair = pairs[f"{left}_vs_{right}"]
    geometry = pair["selection_geometry"]
    capacity = pair["capacity"]
    rescue_left = lens[left]["bridge"][
        "true_vs_distractor_rescue_equal_family_mean"]
    rescue_right = lens[right]["bridge"][
        "true_vs_distractor_rescue_equal_family_mean"]
    preference_left = lens[left]["bridge"][
        "counterfactual_preference_equal_family_mean"]
    preference_right = lens[right]["bridge"][
        "counterfactual_preference_equal_family_mean"]

    def same_sign(left: float, right: float) -> bool:
        return bool(np.sign(left) == np.sign(right))

    return {
        "normalized_selected_span_overlap": bool(
            geometry["normalized_projector_overlap_median"]
            >= thresholds["normalized_selected_span_overlap_min"]),
        "selected_id_jaccard": bool(
            geometry["selected_id_jaccard_median"]
            >= thresholds["selected_id_jaccard_min"]),
        "occupancy": bool(all(
            abs(value["occupancy_difference"])
            <= thresholds["occupancy_difference_max"]
            for value in capacity.values())),
        "centered_excess": bool(all(
            100 * abs(value["centered_excess_difference"])
            <= thresholds[
                "centered_excess_difference_percentage_points_max"]
            for value in capacity.values())),
        "span_safe_specific": bool(
            abs(pair["specific"]["equal_family_mean_difference"])
            <= thresholds[
                "span_safe_specific_mean_difference_nats_max"]),
        "tail_rate": bool(
            abs(pair["tail_rate_difference"])
            <= thresholds["tail_rate_difference_max"]),
        "g4": bool(
            abs(pair["g4_flip_rate_difference"])
            <= thresholds["g4_flip_rate_difference_max"]
            and lens[left]["g4"]["passes"]
            and lens[right]["g4"]["passes"]),
        "bridge_rescue": bool(
            abs(pair["bridge_rescue_difference"])
            <= thresholds[
                "bridge_rescue_preference_difference_nats_max"]
            and same_sign(rescue_left, rescue_right)),
        "bridge_preference": bool(
            abs(pair["bridge_preference_difference"])
            <= thresholds[
                "bridge_rescue_preference_difference_nats_max"]
            and same_sign(preference_left, preference_right)),
    }


def _analyze(
        state: Mapping, selection_pairs: pd.DataFrame,
        capacity_paths: Mapping[str, Path], config: Mapping) -> dict:
    lenses = _lens_summaries(state, config)
    pairs = _pair_summaries(
        state, selection_pairs, capacity_paths, config, lenses)
    functional = _functional_gates(lenses, pairs, config)
    structural, structural_stable = _structural_gate(config)
    branch = branch_from_gates(
        functional, structural_stable=structural_stable)
    interpretations = config["analysis"].get(
        "branch_interpretations", {
            "A": "functionally stable and structurally improving; fit B120",
            "B": "functional instability; continue draw A to n=500",
            "C": (
                "coordinate structure remains fit-sensitive; tested "
                "scientific endpoints are functionally stable"),
            "PENDING_STRUCTURAL": (
                "functional gate complete; structural event is pending"),
        })
    if set(interpretations) != {"A", "B", "C", "PENDING_STRUCTURAL"}:
        raise RuntimeError("branch-interpretation contract drift")
    return {
        "schema_version": 1,
        "tier": config["tier"],
        "selection_was_frozen_before_outcomes": True,
        "no_generated_model_selection_during_run": True,
        "lenses": lenses,
        "pairs": pairs,
        "functional_gates": functional,
        "all_functional_gates_pass": bool(all(functional.values())),
        "structural_gate": structural,
        "branch": branch,
        "branch_interpretation": interpretations[branch],
        "published_reference_classification": (
            "external published reference, partially specified recipe"),
    }


def _plot_functional(analysis: Mapping, config: Mapping, *, png_path: Path,
                     pdf_path: Path) -> None:
    lens_order = list(config["lens_order"])
    legacy_labels = {
        "published": "published\nn=1000",
        "a120": "draw A\nn=120",
        "a250": "draw A\nn=250",
    }
    configured_labels = config["figure"].get("lens_labels", {})
    labels = [
        configured_labels.get(
            lens,
            legacy_labels.get(lens, config["lenses"][lens]["label"]),
        )
        for lens in lens_order
    ]
    colors = ["#666666", "#0072B2", "#009E73"]
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.5))

    axis = axes[0, 0]
    values = [analysis["lenses"][lens]["primary"][
        "specific_equal_family_mean"] for lens in lens_order]
    axis.bar(labels, values, color=colors)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("specific effect (nats)")
    axis.set_title("A · Span-safe effect beyond exact control", loc="left")

    axis = axes[0, 1]
    pair_order = [tuple(value) for value in config["analysis"]["pair_order"]]
    comparisons = [f"{left}_vs_{right}" for left, right in pair_order]
    configured_comparison_labels = config["figure"].get(
        "comparison_labels", {})
    comparison_labels = [
        configured_comparison_labels.get(
            name,
            f"{left.upper()}–{right.replace('published', 'pub.')}",
        )
        for name, (left, right) in zip(
            comparisons, pair_order, strict=True)
    ]
    x = np.arange(len(comparisons))
    jaccard = [analysis["pairs"][name]["selection_geometry"][
        "selected_id_jaccard_median"] for name in comparisons]
    overlap = [analysis["pairs"][name]["selection_geometry"][
        "normalized_projector_overlap_median"] for name in comparisons]
    axis.bar(x - 0.18, jaccard, width=0.36, label="selected-ID Jaccard",
             color="#56B4E9")
    axis.bar(x + 0.18, overlap, width=0.36, label="projector overlap",
             color="#E69F00")
    axis.set_xticks(x, comparison_labels)
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("median agreement")
    axis.set_title("B · Per-position selection geometry", loc="left")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 0]
    layers = [24, 32, 40]
    for lens, label, color in zip(lens_order, labels, colors):
        occupancy = [analysis["lenses"][lens]["capacity"][str(layer)][
            "occupancy_median"] for layer in layers]
        axis.plot(layers, occupancy, marker="o", color=color,
                  label=label.replace("\n", " "))
    axis.set_xticks(layers)
    axis.set_xlabel("source layer")
    axis.set_ylabel("median occupancy")
    axis.set_title("C · Corrected capacity crossing", loc="left")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 1]
    g4 = [analysis["lenses"][lens]["g4"]["flip_rate_j"]
          for lens in lens_order]
    random = [analysis["lenses"][lens]["g4"]["flip_rate_random"]
              for lens in lens_order]
    x = np.arange(len(lens_order))
    axis.bar(x - 0.18, g4, width=0.36, color="#CC79A7", label="G4 J swap")
    axis.bar(x + 0.18, random, width=0.36, color="#BBBBBB",
             label="G4 random")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("flip rate")
    axis.set_title("D · Causal positive control", loc="left")
    axis.legend(frameon=False, fontsize=8)

    figure.suptitle(config["figure"].get(
        "title",
        "Qwen multi-lens functional gate (Phase 4 development)\n"
        "Published comparator: external published reference, partially "
        "specified recipe"), fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)


def _write_raw_frames(state: Mapping, *, output_dir: Path,
                      lens_order: Iterable[str]) -> dict[str, Path]:
    paths = {}
    for component in ("primary", "selection", "readout", "prose",
                      "bridge", "g4"):
        frame = pd.concat([
            pd.DataFrame(state["lenses"][lens][component])
            for lens in lens_order
        ], ignore_index=True)
        if component == "selection":
            for column in ("selected_ids", "selected_scores"):
                frame[column] = frame[column].map(
                    lambda value: json.dumps(value or []))
        path = output_dir / f"{component}_rows.parquet"
        atomic_parquet(path, frame)
        paths[component] = path
    return paths


@torch.no_grad()
def main() -> None:  # noqa: C901
    arguments = parse_args()
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    validate_config(config)
    clean = require_clean_tree()
    try:
        existing = resolve(config["evidence_id"])
    except RegistryError as error:
        if "found 0" not in str(error):
            raise
    else:
        if not existing["live"]:
            raise RuntimeError("existing functional gate is not live")
        for output in existing["outputs"]:
            if file_sha256(output["path"]) != output["sha256"]:
                raise RuntimeError("registered functional-gate output drift")
        print(json.dumps({
            "status": "already-registered-and-verified",
            "evidence_id": config["evidence_id"],
        }, indent=1))
        return

    subset, subset_path = _load_subset(config)
    bundles, bank_contract = _load_bundles(config)
    items, bundle_by_fact = _primary_items(bundles, subset)
    bridge_fact_ids = list(subset["bridge"]["fact_ids"])
    if len(items) != 60 or len(bridge_fact_ids) != 20:
        raise RuntimeError("frozen functional subset size drift")
    lens_paths, lens_contract = _resolve_lens_paths(config)

    output_dir = (
        metrics_dir(config["slug"]) / "functional_gate"
        / config["evidence_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    endpoint_path = output_dir / "fixed_endpoint_activations.pt"
    capacity_input_path = output_dir / "fixed_capacity_activations.pt"
    selection_geometry_path = output_dir / "selection_pair_rows.parquet"
    capacity_paths = {
        lens: output_dir / f"capacity_reconstructions_{lens}.pt"
        for lens in config["lens_order"]
    }
    header = {
        "evidence_id": config["evidence_id"],
        "code_commit": clean["code_commit"],
        "config_sha256": file_sha256(config_path),
        "subset_file_sha256": config["subset"]["sha256"],
        "subset_payload_sha256": config["subset"]["payload_sha256"],
        "lens_hashes": {
            lens: config["lenses"][lens]["lens_sha256"]
            for lens in config["lens_order"]
        },
        "primary_item_ids_sha256": object_sha256(
            subset["primary"]["item_ids"]),
        "bridge_fact_ids_sha256": object_sha256(bridge_fact_ids),
        "prose_item_ids_sha256": object_sha256(
            subset["prose_nll"]["item_ids"]),
        "capacity_item_ids_sha256": object_sha256(
            subset["capacity"]["item_ids"]),
        "g4_item_names_sha256": object_sha256(subset["g4"]["item_names"]),
        "k": int(config["protocol"]["k"]),
        "protect_top_k": int(config["protocol"]["protect_top_k"]),
        "g4_calibration_n": int(config["g4"]["calibration_n"]),
        "bridge_max_new_tokens": int(
            config["bridge_endpoint"]["max_new_tokens"]),
    }
    state = _load_or_create_state(
        state_path, header, config["lens_order"])

    gpu = require_cuda_gpu()
    torch.cuda.reset_peak_memory_stats()
    import transformers
    import jlens
    model_path = resolve_uri(config["model_uri"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_path))
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_path), dtype=torch.bfloat16).to("cuda").eval()
    assert_model_on_cuda(hf)
    wrapped = jlens.from_hf(hf, tokenizer)
    session = ScoringSession(tokenizer, DEFAULT_SPEC, device="cuda")
    band = [int(value) for value in config["protocol"]["band"]]
    checkpoint_every = int(config["protocol"]["checkpoint_every_items"])

    endpoint_cache, capacity_cache = _prepare_baselines(
        hf, wrapped, tokenizer, session,
        items=items, bundle_by_fact=bundle_by_fact, subset=subset,
        band=band,
        capacity_layers=[int(value) for value in
                         config["capacity"]["layers"]],
        capacity_positions=[int(value) for value in
                            config["capacity"]["positions"]],
        state=state, state_path=state_path,
        endpoint_path=endpoint_path,
        capacity_input_path=capacity_input_path,
        checkpoint_every=checkpoint_every,
    )
    log("lens-independent baseline and activation caches complete")

    for lens_name in config["lens_order"]:
        cell = state["lenses"][lens_name]
        if cell.get("complete"):
            log(f"{lens_name} already complete; resume skips cell")
            continue
        started = time.time()
        checkpoint = load_lens_checkpoint(
            lens_paths[lens_name], config["lenses"][lens_name],
            config["runtime"])
        lens = SimpleNamespace(jacobians=checkpoint["J"])
        dictionaries = build_j_dictionaries(hf, lens, band)
        log(f"{lens_name} dictionaries complete")
        _run_primary(
            lens_name, hf, wrapped, tokenizer, session, dictionaries,
            items=items, bundle_by_fact=bundle_by_fact,
            bridge_fact_ids=set(bridge_fact_ids), band=band,
            k=int(config["protocol"]["k"]),
            seed=int(config["protocol"]["matched_seed"]),
            seed_namespace=config["protocol"]["matched_seed_namespace"],
            state=state, state_path=state_path,
            checkpoint_every=checkpoint_every)
        _run_endpoint_readout(
            lens_name, dictionaries, items=items,
            bundle_by_fact=bundle_by_fact, endpoint_cache=endpoint_cache,
            band=band, tok=tokenizer, state=state)
        _checkpoint(state_path, state)
        _run_prose(
            lens_name, hf, wrapped, session, dictionaries,
            prose_items=list(subset["prose_nll"]["items"]), band=band,
            k=int(config["protocol"]["k"]),
            seed=int(config["protocol"]["matched_seed"]),
            seed_namespace=config["protocol"]["matched_seed_namespace"],
            state=state, state_path=state_path,
            checkpoint_every=checkpoint_every)
        _run_bridge_endpoint(
            lens_name, hf, wrapped, tokenizer, session, dictionaries,
            bridge_fact_ids=bridge_fact_ids,
            bundle_by_fact=bundle_by_fact, band=band,
            k=int(config["protocol"]["k"]),
            max_new_tokens=int(config["bridge_endpoint"]["max_new_tokens"]),
            state=state, state_path=state_path,
            checkpoint_every=checkpoint_every)
        _run_g4(
            lens_name, hf, wrapped, tokenizer, dictionaries,
            g4_items=list(subset["g4"]["items"]), band=band,
            alpha=float(config["g4"]["alpha"]),
            random_seed=int(config["g4"]["random_seed"]),
            state=state, state_path=state_path,
            checkpoint_every=checkpoint_every)
        capacity_dictionaries = {
            int(layer): dictionaries[int(layer)]
            for layer in config["capacity"]["layers"]
        }
        del dictionaries
        gc.collect()
        torch.cuda.empty_cache()
        _run_capacity(
            lens_name, capacity_dictionaries,
            capacity_cache=capacity_cache,
            specification=config["capacity"],
            artifact_path=capacity_paths[lens_name], state=state,
            state_path=state_path)
        del capacity_dictionaries, checkpoint, lens
        gc.collect()
        torch.cuda.empty_cache()
        cell["complete"] = True
        cell["elapsed_seconds"] = round(time.time() - started, 3)
        cell["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
        _checkpoint(state_path, state)
        log(f"{lens_name} cell banked")

    if not all(state["lenses"][lens].get("complete")
               for lens in config["lens_order"]):
        raise RuntimeError("functional gate ended with incomplete lens cells")
    selection_pairs = _selection_geometry(
        hf, lens_paths=lens_paths, config=config, state=state,
        output_path=selection_geometry_path)
    raw_paths = _write_raw_frames(
        state, output_dir=output_dir, lens_order=config["lens_order"])
    analysis = _analyze(
        state, selection_pairs, capacity_paths, config)

    png_path = figures_dir() / f"{config['figure']['stem']}.png"
    pdf_path = figures_dir() / f"{config['figure']['stem']}.pdf"
    _plot_functional(
        analysis, config, png_path=png_path, pdf_path=pdf_path)
    manifest_path = output_dir / "input_manifest.json"
    result_path = output_dir / "functional_gate_result.json"
    input_payload = {
        "schema_version": 1,
        "evidence_id": config["evidence_id"],
        "code_commit": clean["code_commit"],
        "config_sha256": file_sha256(config_path),
        "producer_sha256": file_sha256(Path(__file__)),
        "model": model_reference(config["model_uri"]),
        "gpu": gpu,
        "subset": {
            **dict(config["subset"]),
            "resolved_path": str(subset_path),
        },
        "task_banks": bank_contract,
        "lenses": lens_contract,
        "protocol": dict(config["protocol"]),
        "g4": dict(config["g4"]),
        "bridge_endpoint": dict(config["bridge_endpoint"]),
        "capacity": dict(config["capacity"]),
        "analysis": dict(config["analysis"]),
        "state_header": header,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }
    manifest_envelope = {
        "schema_version": 1,
        "payload": input_payload,
        "payload_sha256": object_sha256(input_payload),
    }
    atomic_json(manifest_path, manifest_envelope)
    command = (
        "python -m "
        "jspace_phase4.experiments.p4_qwen_multilens_functional_gate "
        f"--config {arguments.config}")
    inputs = {
        "functional_subset": config["subset"]["payload_sha256"],
        "task_banks": object_sha256(bank_contract),
        **{
            f"lens_{lens}": config["lenses"][lens]["lens_sha256"]
            for lens in config["lens_order"]
        },
        "input_manifest": manifest_envelope["payload_sha256"],
    }
    write_result4(
        analysis, result_path,
        Provenance4(
            evidence_id=config["evidence_id"], tier=config["tier"],
            command=command, inputs=inputs,
            input_manifest_sha256=manifest_envelope["payload_sha256"],
            model=model_reference(config["model_uri"]),
            seed_contract=(
                "fixed Phase 3 item IDs; identical condition order; "
                "stable matched-control namespace; paired prompt/family "
                "bootstrap seeds from the frozen YAML"),
        ),
    )
    outputs = [
        result_path, manifest_path, state_path, endpoint_path,
        capacity_input_path, selection_geometry_path,
        *raw_paths.values(), *capacity_paths.values(), png_path, pdf_path,
    ]
    create(
        config["evidence_id"], tier=config["tier"],
        what=config.get(
            "registry_what",
            "Fixed Qwen multi-lens functional gate over published n=1000, "
            "draw-A n=120, and draw-A n=250: selection geometry, corrected "
            "capacity, span-safe/exact-control behavior, bridge semantics, "
            "G4, prose, and the frozen Phase 4 branch rule."),
        command=command, outputs=outputs, inputs=inputs,
    )
    print(json.dumps({
        "evidence_id": config["evidence_id"],
        "branch": analysis["branch"],
        "functional_gates": analysis["functional_gates"],
        "structural_gate": analysis["structural_gate"],
        "result": str(result_path),
        "figure": str(png_path),
    }, indent=1))


if __name__ == "__main__":
    main()
