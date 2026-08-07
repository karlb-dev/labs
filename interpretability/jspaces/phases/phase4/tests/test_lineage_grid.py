from pathlib import Path

import pandas as pd
import yaml

from jspace_phase4.experiments.p4_olmo_lineage_grid import (
    capable_fact_keys,
    scientific_seed_namespace,
)


def test_capable_cohort_requires_exact_direct_composed_pair():
    frame = pd.DataFrame([
        {"bank": "F", "fact_id": "keep", "variant": "direct",
         "capable_generation": True},
        {"bank": "F", "fact_id": "keep", "variant": "composed",
         "capable_generation": True},
        {"bank": "F", "fact_id": "one-fails", "variant": "direct",
         "capable_generation": True},
        {"bank": "F", "fact_id": "one-fails", "variant": "composed",
         "capable_generation": False},
        {"bank": "S", "fact_id": "missing", "variant": "direct",
         "capable_generation": True},
        {"bank": "S", "fact_id": "duplicate", "variant": "direct",
         "capable_generation": True},
        {"bank": "S", "fact_id": "duplicate", "variant": "direct",
         "capable_generation": True},
        {"bank": "S", "fact_id": "duplicate", "variant": "composed",
         "capable_generation": True},
    ])
    assert capable_fact_keys(frame) == {("F", "keep")}


def test_all_phase4_yaml_configs_parse():
    package_root = Path(__file__).resolve().parents[1]
    for path in sorted((package_root / "configs").glob("*.yaml")):
        value = yaml.safe_load(path.read_text())
        assert isinstance(value, dict), path
        assert value.get("evidence_id"), path


def test_evidence_version_can_preserve_scientific_seed_namespace():
    assert scientific_seed_namespace({}, "evidence-v1") == "evidence-v1"
    assert scientific_seed_namespace(
        {"scientific_seed_namespace": "evidence-v1"},
        "evidence-v3",
    ) == "evidence-v1"
