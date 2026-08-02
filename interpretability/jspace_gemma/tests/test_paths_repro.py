from pathlib import Path

import pytest

from jspace_gemma.paths import PathContractError, assert_isolated_output, resolve_uri


def test_logical_uri_refuses_parent_traversal():
    with pytest.raises(PathContractError, match="unsafe"):
        resolve_uri("repo://../outside", must_exist=False)


def test_output_isolation_rejects_other_phase(tmp_path, monkeypatch):
    run = tmp_path / "run"
    local = tmp_path / "local"
    monkeypatch.setenv("JSPACE_GEMMA_RUN_ROOT", str(run))
    monkeypatch.setenv("JSPACE_GEMMA_LOCAL_ROOT", str(local))
    assert assert_isolated_output(run / "metrics" / "x.json") == (run / "metrics" / "x.json").resolve()
    with pytest.raises(PathContractError, match="outside isolated"):
        assert_isolated_output(tmp_path / "phase4" / "x.json")
