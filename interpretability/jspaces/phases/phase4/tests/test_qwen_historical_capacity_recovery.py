import ast
import hashlib
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p4_qwen_historical_capacity_recovery.yaml"


def _ast_hash(path, name):
    tree = ast.parse(path.read_text())
    node, = [value for value in tree.body
             if isinstance(value, ast.FunctionDef) and value.name == name]
    return hashlib.sha256(ast.dump(
        node, include_attributes=False).encode()).hexdigest()


def test_historical_capacity_recovery_pins_exact_registered_bytes():
    config = yaml.safe_load(CONFIG.read_text())
    assert config["source_event"]["code_commit"].startswith("30b121d")
    assert config["lens"]["name"] == "a120"
    assert config["lens"]["n_prompts"] == 120
    assert config["target"]["expected_sha256"] == (
        "6b0399df2c57158e7fdad24274e50f8c1058021d233412afdcc5177f6c651b6f")
    assert config["target"]["replace_only_on_exact_hash_match"] is True
    assert config["target"]["never_register_a_partial_or_near_match"] is True
    assert config["runtime"]["minimum_free_gpu_bytes"] >= 75_000_000_000
    assert "does not reconstruct state.json" in config["claim_boundary"]


def test_historical_capacity_algorithm_asts_match_frozen_contract():
    from jspace_phase4.paths4 import REPO_ROOT, _rewrite_repo_relative

    config = yaml.safe_load(CONFIG.read_text())
    contract = config["algorithm_contract"]
    module = REPO_ROOT / _rewrite_repo_relative(contract["functional_module"])
    for name, expected in contract["function_ast_sha256"].items():
        assert _ast_hash(module, name) == expected


def test_exact_sign_archive_temp_name_preserves_historical_pt_root(tmp_path):
    import zipfile

    target = tmp_path / "capacity_reconstructions_a120.pt"
    candidate = target.with_suffix(target.suffix + ".tmp123")
    torch.save({"schema_version": 1, "x": torch.tensor([1])}, candidate)
    with zipfile.ZipFile(candidate) as archive:
        assert archive.namelist()[0].startswith(
            "capacity_reconstructions_a120.pt/")


def test_recovery_event_lookup_requires_one_exact_path(tmp_path):
    from jspace_phase4.experiments.p4_qwen_historical_capacity_recovery import (
        _event_output,
    )

    target = tmp_path / "target.pt"
    event = {"outputs": [
        {"path": str(target), "sha256": "a" * 64},
        {"path": str(tmp_path / "other.pt"), "sha256": "b" * 64},
    ]}
    assert _event_output(event, target)["sha256"] == "a" * 64
