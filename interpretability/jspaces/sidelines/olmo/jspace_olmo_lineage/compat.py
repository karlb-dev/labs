"""Hash-gated compatibility boundary to the frozen Phase 4 Bank-W code."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from jspace_phase4 import scoring4 as source_scoring
from jspace_phase4 import state as source_state
from jspace_phase4.experiments import p4_bank_w_capability as source_capability
from jspace_phase4.scoring4 import DEFAULT_SPEC, ScoringSession

from .manifests import file_sha256, object_sha256
from .paths import resolve_uri


class CompatibilityError(RuntimeError):
    pass


def verify_sources(specifications: Mapping) -> dict:
    modules = {
        "scoring4": Path(source_scoring.__file__).resolve(),
        "bank_w_capability": Path(source_capability.__file__).resolve(),
        "state": Path(source_state.__file__).resolve(),
    }
    rows = {}
    for name, specification in specifications.items():
        path = resolve_uri(specification["uri"])
        actual = file_sha256(path)
        expected = specification["sha256"]
        if actual != expected:
            raise CompatibilityError(
                f"frozen Phase 4 compatibility source drifted: {name}; "
                f"expected {expected}, got {actual}")
        if path.resolve() != modules[name]:
            raise CompatibilityError(
                f"imported {name} module is not the pinned repository file: "
                f"{modules[name]} != {path.resolve()}")
        rows[name] = {
            "path": str(path), "sha256": actual,
            "bytes": int(path.stat().st_size),
        }
    return {
        "verified": rows,
        "imported_module_paths": {
            name: str(value) for name, value in modules.items()},
        "scoring_spec": DEFAULT_SPEC.as_dict(),
        "scoring_spec_sha256": object_sha256(DEFAULT_SPEC.as_dict()),
        "conformance_mode": (
            "direct calls into exact hash-pinned Phase 4 pure functions"),
    }


def select_development_rows(rows: Sequence[Mapping],
                            selection: Mapping) -> list[dict]:
    return source_capability.select_development_rows(rows, selection)


def candidate_scores(model, session: ScoringSession, prompt: str,
                     aliases: Sequence[str], *, batch_size: int,
                     pad_token_id: int) -> tuple[dict[str, float], int, dict]:
    return source_capability._candidate_scores(  # noqa: SLF001
        model, session, prompt, aliases,
        batch_size=batch_size, pad_token_id=pad_token_id)


def validate_finite_rows(rows: Sequence[Mapping], *, aliases: Sequence[str],
                         expected_rows: int) -> dict:
    if len(rows) != expected_rows:
        raise CompatibilityError(
            f"expected {expected_rows} scored rows, got {len(rows)}")
    checked_values = 0
    for row in rows:
        numeric = (
            row["baseline_answer_margin"], row["true_answer_sequence_lp"],
            row["prompt_token_count"], row["answer_token_count"],
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise CompatibilityError(
                f"non-finite Bank-W endpoint in {row.get('item_id')}")
        scores = json.loads(str(row["candidate_scores_json"]))
        if list(scores) != sorted(scores):
            raise CompatibilityError("candidate score JSON is not canonical")
        if set(scores) != set(aliases) or len(scores) != 8:
            raise CompatibilityError("incomplete eight-answer score vector")
        if not all(math.isfinite(float(value)) for value in scores.values()):
            raise CompatibilityError(
                f"non-finite candidate score in {row.get('item_id')}")
        checked_values += len(numeric) + len(scores)
    return {
        "all_rows_finite": True,
        "no_rows_dropped": True,
        "n_rows": len(rows),
        "n_numeric_values_checked": checked_values,
        "candidate_sequences_per_row": len(aliases),
    }


def analyze_model_rows(rows: Sequence[Mapping], *, selection: Mapping,
                       guard: Mapping, aliases: Sequence[str]) -> dict:
    finite = validate_finite_rows(
        rows, aliases=aliases,
        expected_rows=int(selection["expected_rows_per_model"]))
    result = source_capability.analyze_model_rows(
        rows, selection=selection, guard=guard)
    result["side_track_finite_gate"] = finite
    result["phase4_function_source_sha256"] = (
        "77c963990dd980e94664ed6b9ead2e6e60c574ce40372186f638c3b7640fbe51")
    return result


def aggregate_model_payloads(payloads: Mapping[str, Mapping], *,
                             config: Mapping) -> dict:
    return source_capability.aggregate_model_payloads(
        payloads, config=config)
