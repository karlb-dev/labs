from pathlib import Path

import yaml

from jspace_gemma.stage1_license import select_stage1_route


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return yaml.safe_load((ROOT / "configs/gm2_stage1_relicense.yaml").read_text())


def test_g22_relicenses_when_historical_error_is_inside_stable_ceiling():
    result = select_stage1_route(
        {
            "route": "benign_scheduling_floor",
            "licensed_ceilings": {"gemma": 0.1},
        },
        _config(),
        source_hashes_exact=True,
    )
    assert result["branch"] == "branch_1_relicense_without_recompute"
    assert result["evidence_id"] == "gm2-stage1-relicense-v1"


def test_g22_selects_only_declared_batch1_replay_for_batch_nuisance():
    result = select_stage1_route(
        {
            "route": "batch_composition_nuisance",
            "licensed_ceilings": {"gemma": 1.0e-6},
        },
        _config(),
        source_hashes_exact=True,
    )
    assert result["branch"] == "branch_2_batch1_declared_dose"
    assert result["evidence_id"] == "gm2-stage1-batch1-v1"


def test_g22_blocks_path_ambiguity_or_missing_source_hash():
    config = _config()
    for calibration, hashes in (
        ({"route": "path_ambiguity", "licensed_ceilings": {"gemma": None}}, True),
        ({"route": "benign_scheduling_floor", "licensed_ceilings": {"gemma": 0.1}}, False),
    ):
        result = select_stage1_route(
            calibration, config, source_hashes_exact=hashes
        )
        assert result["branch"] == "branch_3_remains_blocked"
        assert result["evidence_id"] == "gm2-stage1-remains-blocked-v1"

