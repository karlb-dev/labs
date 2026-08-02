from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def test_holm2_power_contract_is_prospective_and_conservative():
    original = _load("p4_bank_w_power_dev.yaml")
    successor = _load("p4_bank_w_power_holm2_dev.yaml")

    assert successor["evidence_id"] == "p4-bank-w-power-holm2-dev-v1"
    assert successor["development_calibration"] == original[
        "development_calibration"]
    assert successor["target_model_slugs"] == original["target_model_slugs"]
    assert successor["simulation"]["sesoi_slope_nats_per_doubling"] == (
        original["simulation"]["sesoi_slope_nats_per_doubling"])
    assert successor["primary_randomization"]["familywise_alpha"] == 0.05
    assert successor["primary_randomization"]["holm_primary_count"] == 2
    assert successor["primary_randomization"]["alpha"] == 0.025
    assert successor["simulation"]["confirmatory_families"] == 24
    assert successor["simulation"]["family_counts"] == [24, 28, 32]
    assert successor["simulation"]["power_target"] == 0.80


def test_holm2_outputs_cannot_overwrite_nominal_power_evidence():
    original = _load("p4_bank_w_power_dev.yaml")["outputs"]
    successor = _load("p4_bank_w_power_holm2_dev.yaml")["outputs"]

    assert set(original.values()).isdisjoint(successor.values())
    assert "holm2" in successor["result"]
    assert "holm2" in successor["figure_png"]
    assert "holm2" in successor["figure_pdf"]
