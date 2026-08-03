from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
from jspace_olmo_lineage.experiments.stage_wedge import (
    ALL_CONDITIONS,
    _audit_matched,
    capability_summary,
    exact_grade_alias,
    freeze_cohort,
    tokenizer_contract_checks,
)
from jspace_olmo_lineage.manifests import InputManifest, object_sha256
from jspace_phase4.scoring4 import DEFAULT_SPEC


class _Session:
    spec = DEFAULT_SPEC


def test_exact_capability_rejects_explanatory_suffix() -> None:
    session = _Session()
    assert exact_grade_alias(session, "Merrid", [" Merrid"]) == " Merrid"
    assert exact_grade_alias(session, "Merrid.", [" Merrid"]) == " Merrid"
    assert exact_grade_alias(session, "Merrid because", [" Merrid"]) is None


def test_tokenizer_audit_uses_frozen_config_field_name() -> None:
    semantics = {
        "semantic_fingerprint_sha256": "a",
        "token_id_map_sha256": "b",
        "normalized_model_sha256": "c",
        "processing_components_sha256": "d",
        "audit_encoding_sha256": "e",
    }
    expected = {
        **{
            key: value
            for key, value in semantics.items()
            if key != "audit_encoding_sha256"
        },
        "frozen_audit_encoding_sha256": "e",
    }
    assert all(tokenizer_contract_checks(semantics, expected).values())


def _manifest() -> InputManifest:
    return InputManifest(
        experiment_id="ol2-test",
        config_sha256="a" * 64,
        model_id="test/model",
        model_revision="b" * 40,
        tokenizer_manifest_sha256="c" * 64,
        lens_sha256="d" * 64,
        bank_sha256="e" * 64,
        partition_sha256="f" * 64,
        scoring_spec_sha256="0" * 64,
        code_commit="1" * 40,
    )


def test_capable_cohort_is_frozen_before_interventions(tmp_path) -> None:
    rows = []
    for fact, family, capable in (
        ("fact-a", "family-a", True),
        ("fact-b", "family-b", True),
        ("fact-c", "family-c", False),
    ):
        for variant in ("direct", "composed"):
            rows.append(
                {
                    "item_id": f"{fact}#{variant}",
                    "fact_id": fact,
                    "canonical_family": family,
                    "bank": "S",
                    "variant": variant,
                    "capable_generation": capable,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_parquet(tmp_path / "g5_capability.parquet", index=False)
    config = {
        "g5_capability": {
            "bank_s_direct_composed_fact_floor": 2,
            "bank_s_capable_family_floor": 2,
        }
    }
    cohort = freeze_cohort(config, "think_sft", frame, _manifest(), tmp_path)
    assert cohort["payload"]["capability_gate_passed"] is True
    assert cohort["payload"]["n_facts"] == 2
    assert cohort["payload"]["selection_opened_before_interventions"] is True
    assert object_sha256(cohort["payload"]) == cohort["payload_sha256"]
    assert json.loads((tmp_path / "cohort_manifest.json").read_text()) == cohort
    summary = capability_summary(frame)
    assert summary["fully_capable_direct_composed_facts_by_bank"] == {"S": 2}


@dataclass
class _Matched:
    achieved_rank: int = 3
    target_rank: int = 3
    achieved_energy_frac: float = 0.20001
    target_energy_frac: float = 0.2
    clamped: bool = False
    max_protected_cos: float = 1e-5


class _Log:
    def __init__(self) -> None:
        self.matched = [_Matched()]


def test_matched_control_numeric_gate() -> None:
    report = _audit_matched(_Log(), "instant_rank_energy_matched")
    assert report["passed"] is True
    assert report["rank_failures"] == 0


def test_frozen_condition_names_are_not_phase4_aliases() -> None:
    assert ALL_CONDITIONS == (
        "baseline",
        "meanJ_span_safe",
        "instant_rank_energy_matched",
        "meanJ_label_protected",
        "protected_energy_matched",
        "mechanics_random",
        "logit_label_protected",
    )
