from pathlib import Path

import pytest

from jspace_olmo_lineage import paths
from jspace_olmo_lineage.paths import PathBoundaryError


def test_run_root_accepts_only_olmo_namespace(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "SPECIAL_LAB_ROOT", tmp_path)
    valid = tmp_path / "olmo_lineage_test"
    monkeypatch.setenv("JSPACE_OLMO_RUN_ROOT", str(valid))
    assert paths.run_root(create=True) == valid


@pytest.mark.parametrize("name", ["phase4_20260731", "gemma_transport_test"])
def test_run_root_rejects_foreign_tracks(monkeypatch, name):
    foreign = Path("/content/drive/MyDrive/interpret/special-lab-1") / name
    monkeypatch.setenv("JSPACE_OLMO_RUN_ROOT", str(foreign))
    with pytest.raises(PathBoundaryError):
        paths.run_root(create=False)


def test_run_root_env_override_allows_non_drive_hosts(monkeypatch, tmp_path):
    """JSPACE_OLMO_RUN_ROOT may point off Drive (Windows/laptop mirrors)."""
    valid = tmp_path / "olmo_lineage_local"
    monkeypatch.setenv("JSPACE_OLMO_RUN_ROOT", str(valid))
    assert paths.run_root(create=True) == valid
