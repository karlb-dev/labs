import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_seed_is_stable_across_python_hash_salts():
    code = (
        "from jspace_phase4.seeds import stable_seed;"
        "print(stable_seed(experiment_id='p4-x',item_id='f1',"
        "condition='matched',layer=24,position=7,base_seed=31337))"
    )
    values = []
    for salt in ("1", "222", "random"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = salt
        values.append(int(subprocess.check_output(
            [sys.executable, "-c", code],
            env=environment,
            text=True,
        )))
    assert len(set(values)) == 1


def test_seed_changes_with_every_scientific_component():
    from jspace_phase4.seeds import stable_seed
    base = {
        "experiment_id": "p4-x",
        "item_id": "f1",
        "condition": "matched",
        "layer": 24,
        "position": 7,
        "base_seed": 31337,
    }
    reference = stable_seed(**base)
    replacements = {
        "experiment_id": "p4-y",
        "item_id": "f2",
        "condition": "span_safe",
        "layer": 25,
        "position": 8,
        "base_seed": 31338,
    }
    for key, value in replacements.items():
        changed = dict(base)
        changed[key] = value
        assert stable_seed(**changed) != reference


def test_no_builtin_hash_call_in_phase4_package():
    offenders = []
    for path in (PACKAGE_ROOT / "jspace_phase4").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "hash"):
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}")
    assert not offenders


def test_machine_paths_are_confined_to_paths_module():
    offenders = []
    for path in (PACKAGE_ROOT / "jspace_phase4").rglob("*.py"):
        if path.name == "paths4.py":
            continue
        text = path.read_text()
        for needle in ("/content/drive", "/content/models", "/content/sl4"):
            if needle in text:
                offenders.append((str(path), needle))
    assert not offenders


def test_run_root_override_and_uri_resolution(tmp_path, monkeypatch):
    from jspace_phase4.paths4 import resolve_uri, run_root
    root = tmp_path / "run"
    monkeypatch.setenv("JSPACE4_RUN_ROOT", str(root))
    assert run_root() == root
    artifact = root / "metrics/value.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}")
    assert resolve_uri("artifact://phase4/metrics/value.json") == artifact
    with pytest.raises(Exception, match="pin an exact revision"):
        resolve_uri("model://org/model")


def test_model_resolver_rejects_drivefs_cache(monkeypatch):
    from jspace_phase4.paths4 import resolve_uri
    monkeypatch.setenv(
        "HF_HUB_CACHE", "/content/drive/MyDrive/hf_cache/hub")
    with pytest.raises(Exception, match="DriveFS"):
        resolve_uri("model://org/model@" + "a" * 40)


def test_gpu_guard_refuses_invisible_cuda(monkeypatch):
    from jspace_phase4 import gpu
    monkeypatch.setattr(gpu.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="do not run"):
        gpu.require_cuda_gpu()
