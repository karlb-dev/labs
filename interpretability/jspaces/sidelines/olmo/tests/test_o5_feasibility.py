from jspace_olmo_lineage.experiments.o5_feasibility import route_feasibility


REQUIRED = [
    "crossed_activation_model_cells",
    "crossed_transport_lens_cells",
    "crossed_readout_cells",
]


def test_missing_crossed_factor_defers_without_proxy():
    result = route_feasibility({
        "crossed_activation_model_cells": True,
        "crossed_transport_lens_cells": True,
        "crossed_readout_cells": False,
    }, REQUIRED)
    assert result["decision"] == (
        "defer-no-identifiable-crossed-intervention-estimand")
    assert result["status"] == "not-executed-no-proxy-substitution"
    assert result["missing_required_controls"] == [
        "crossed_readout_cells"]


def test_all_controls_only_authorize_prospective_pilot():
    result = route_feasibility({name: True for name in REQUIRED}, REQUIRED)
    assert result["decision"] == "ready-for-prospective-o5-pilot"
    assert result["status"] == "ready-not-started"
    assert result["phase5_pilot_eligible"]
