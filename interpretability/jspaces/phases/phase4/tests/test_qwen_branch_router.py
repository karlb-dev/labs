def _interpretations():
    return {
        "A": "stable; fit draw B",
        "B": "unstable; fit A1000",
        "C": "structural only; fit draw B",
        "PENDING_STRUCTURAL": "pending",
    }


def test_qwen_frozen_branch_routes_a_and_c_to_draw_b():
    from jspace_phase4.experiments.p4_qwen_branch_router import (
        route_from_payload,
    )
    interpretations = _interpretations()
    for branch in ("A", "C"):
        result = route_from_payload({
            "branch": branch,
            "branch_interpretation": interpretations[branch],
        }, interpretations)
        assert result["continuation"] == "draw_b_n120"


def test_qwen_frozen_branch_routes_b_to_a1000():
    from jspace_phase4.experiments.p4_qwen_branch_router import (
        route_from_payload,
    )
    interpretations = _interpretations()
    result = route_from_payload({
        "branch": "B",
        "branch_interpretation": interpretations["B"],
    }, interpretations)
    assert result["continuation"] == "draw_a_n1000"


def test_qwen_frozen_branch_refuses_pending_unknown_or_wording_drift():
    from jspace_phase4.experiments.p4_qwen_branch_router import (
        route_from_payload,
    )
    interpretations = _interpretations()
    payloads = (
        {"branch": "PENDING_STRUCTURAL", "branch_interpretation": "pending"},
        {"branch": "D", "branch_interpretation": "unknown"},
        {"branch": "A", "branch_interpretation": "rewritten after outcome"},
    )
    for payload in payloads:
        try:
            route_from_payload(payload, interpretations)
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid A500 branch route was accepted")
