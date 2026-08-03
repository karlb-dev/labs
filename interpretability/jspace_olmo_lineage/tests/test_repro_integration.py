from pathlib import Path

from jspace_olmo_lineage.paths import REPO_ROOT
from jspace_olmo_lineage.repro import _repository_materialization, _within


def test_release_output_materializes_from_merged_worktree():
    relative = Path(
        "interpretability/jspace_olmo_lineage/release/"
        "IMPORT_BUNDLE_SIDELINES2.json"
    )
    registered = Path("/content/source-vm") / relative
    materialized = _repository_materialization(registered)
    assert materialized == REPO_ROOT / relative
    assert materialized.is_file()
    assert _within(
        materialized,
        REPO_ROOT / "interpretability/jspace_olmo_lineage",
    )


def test_external_output_is_not_remapped_into_repository():
    external = Path("/content/drive/MyDrive/interpret/result.json")
    assert _repository_materialization(external) == external
