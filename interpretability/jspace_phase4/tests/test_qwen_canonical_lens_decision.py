from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return yaml.safe_load((
        ROOT / "configs" / "p4_qwen_canonical_lens_decision_a1000_dev.yaml"
    ).read_text())


def _functional(branch="Q-L1"):
    gates = {
        "normalized_selected_span_overlap": True,
        "selected_id_jaccard": True,
        "occupancy": True,
        "centered_excess": True,
        "span_safe_specific": True,
        "tail_rate": True,
        "g4": True,
        "bridge_rescue": True,
        "bridge_preference": True,
    }
    if branch == "Q-L2":
        gates["selected_id_jaccard"] = False
    elif branch == "Q-L3":
        gates["selected_id_jaccard"] = False
        gates["normalized_selected_span_overlap"] = False
    elif branch == "Q-L4":
        gates["bridge_rescue"] = False
    return {
        "branch": "PENDING_SELECTION_MARGIN_AUDIT",
        "branch_candidate": branch,
        "functional_gates": gates,
        "structural_gate": {
            "status": "verified-live",
            "all_structural_gates_pass": branch != "Q-L5",
        },
    }


def _margin(branch="Q-L1"):
    return {
        "functional_branch_candidate": branch,
        "stratum_counts": {"stable_core": 3},
        "lexical_audit": {"manual_review_required_rows": 1},
        "contract_verdict": {
            "capture_formula_recomputed_exactly": True,
            "captured_top_k_matches_intervention": True,
            "registered_selection_geometry_reconstructed": True,
            "all_positions_retained": True,
            "all_strata_retained_in_functional_gate": True,
            "behavioral_columns_used": False,
            "audit_complete_for_canonical_branch_router": True,
        },
    }


def test_canonical_decision_activates_only_the_prospective_ql2_amendment():
    from jspace_phase4.experiments.p4_qwen_canonical_lens_decision import (
        decide,
    )

    influence = {"decision": "negligible", "prompt_retained_unconditionally": True}
    first = decide(
        functional=_functional("Q-L1"), margin=_margin("Q-L1"),
        influence=influence, config=_config())
    assert first["canonical_lens"] == "a1000"
    assert first["ql2_amendment"]["discarded_unused"]
    second = decide(
        functional=_functional("Q-L2"), margin=_margin("Q-L2"),
        influence=influence, config=_config())
    assert second["canonical_lens"] == "a1000"
    assert second["ql2_amendment"]["activated"]
    assert "span-amendment" in second["p4_p2_status"]


def test_canonical_decision_blocks_unlicensed_new_lens_branches():
    from jspace_phase4.experiments.p4_qwen_canonical_lens_decision import (
        decide,
    )

    influence = {
        "decision": "material_at_a1000",
        "prompt_retained_unconditionally": True,
    }
    for branch in ("Q-L3", "Q-L4", "Q-L5"):
        result = decide(
            functional=_functional(branch), margin=_margin(branch),
            influence=influence, config=_config())
        assert result["canonical_lens"] is None
        assert result["canonical_lens_nominated"] is False
        assert result["prompt323_retained"] is True


def test_canonical_decision_refuses_incomplete_margin_or_prompt_trimming():
    import pytest

    from jspace_phase4.experiments.p4_qwen_canonical_lens_decision import (
        decide,
    )

    margin = _margin()
    margin["contract_verdict"]["all_positions_retained"] = False
    with pytest.raises(RuntimeError, match="all_positions_retained"):
        decide(
            functional=_functional(), margin=margin,
            influence={
                "decision": "negligible",
                "prompt_retained_unconditionally": True,
            }, config=_config())
    with pytest.raises(RuntimeError, match="trimming/refit"):
        decide(
            functional=_functional(), margin=_margin(),
            influence={
                "decision": "negligible",
                "prompt_retained_unconditionally": False,
            }, config=_config())
