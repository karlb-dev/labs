from pathlib import Path

from jspace_gemma.paths import REPO_ROOT
from jspace_gemma.repro import _repository_materialization


def test_release_output_materializes_from_merged_worktree():
    relative = Path(
        "interpretability/jspace_gemma/release/"
        "IMPORT_BUNDLE_SIDELINES2.json"
    )
    registered = Path("/content/source-vm") / relative
    materialized = _repository_materialization(registered)
    assert materialized == REPO_ROOT / relative
    assert materialized.is_file()


def test_external_output_is_not_remapped_into_repository():
    external = Path("/content/drive/MyDrive/interpret/result.json")
    assert _repository_materialization(external) == external
