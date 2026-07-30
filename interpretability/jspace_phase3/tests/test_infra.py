# Conformance: run-root indirection + Phase 3 registry rules.
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]


def test_run_root_env_override(tmp_path, monkeypatch):
    from jspace_phase3 import paths3
    monkeypatch.setenv("JSPACE3_RUN_ROOT", str(tmp_path / "rr"))
    assert paths3.run_root() == tmp_path / "rr"
    assert (tmp_path / "rr").is_dir()
    assert paths3.metrics_dir("x").is_dir()


def test_registry_rejects_phase2_tiers(tmp_path):
    from jspace_phase3 import provenance3 as p3
    ev = {"event": "evidence_created", "evidence_id": "t-v1",
          "tier": "confirmatory", "what": "w", "command": "c",
          "code_commit": "deadbeef"}
    with pytest.raises(Exception, match="tier"):
        p3.append_event(ev, path=tmp_path / "e.jsonl")


def test_registry_lifecycle(tmp_path):
    from jspace_phase3 import provenance3 as p3
    log = tmp_path / "e.jsonl"
    p3.append_event({"event": "evidence_created", "evidence_id": "a-v1",
                     "tier": "phase3-development", "what": "w",
                     "command": "c", "code_commit": "deadbeef"}, path=log)
    # duplicate creation refused
    with pytest.raises(Exception, match="already created"):
        p3.append_event({"event": "evidence_created", "evidence_id": "a-v1",
                         "tier": "phase3-development", "what": "w",
                         "command": "c", "code_commit": "deadbeef"}, path=log)
    # supersede must reference known ids on both sides
    with pytest.raises(Exception, match="unknown evidence_id"):
        p3.append_event({"event": "evidence_superseded",
                         "evidence_id": "ghost-v1",
                         "superseded_by": "a-v1"}, path=log)
    p3.append_event({"event": "evidence_created", "evidence_id": "a-v2",
                     "tier": "phase3-development", "what": "w",
                     "command": "c", "code_commit": "deadbeef"}, path=log)
    p3.append_event({"event": "evidence_superseded", "evidence_id": "a-v1",
                     "superseded_by": "a-v2"}, path=log)
    rec = p3.resolve("a-v1", path=log)
    assert rec["superseded_by"] == "a-v2" and not rec["live"]
    assert p3.resolve("a-v2", path=log)["live"]
    # study id stamped on every event
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert all(r["study_id"] == "jspace-phase3" for r in rows)


def test_no_hardcoded_machine_paths():
    """nextsteps §15.2: hard-coded run roots are forbidden in Phase 3
    modules (tests and reviews excluded)."""
    bad = []
    for f in (PKG / "jspace_phase3").rglob("*.py"):
        text = f.read_text()
        for needle in ("/content/drive", "/content/models", "/content/sl1"):
            if needle in text and f.name != "paths3.py":
                bad.append((str(f), needle))
    assert not bad, f"hardcoded machine paths: {bad}"


def test_write_result3_deterministic_payload(tmp_path, monkeypatch):
    from jspace_phase3 import provenance3 as p3
    prov = p3.Provenance3(evidence_id="x-v1", tier="phase3-development",
                          command="c")
    e1 = p3.write_result3({"a": 1}, tmp_path / "r.json", prov)
    e2 = p3.write_result3({"a": 1}, tmp_path / "r2.json", prov)
    assert e1["payload_sha256"] == e2["payload_sha256"]


def test_cli_runs():
    out = subprocess.run([sys.executable, "-m", "jspace_phase3",
                          "registry-list"], capture_output=True, text=True,
                         cwd=str(PKG))
    assert out.returncode == 0


def test_stable_seed_ignores_python_hash_salt():
    code = (
        "from jspace_phase3.seeds import stable_seed;"
        "print(stable_seed('matched-control','family:item#direct',31337))"
    )
    values = []
    for salt in ("1", "2", "random"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = salt
        out = subprocess.check_output(
            [sys.executable, "-c", code], text=True, env=env)
        values.append(int(out))
    assert len(set(values)) == 1
    assert values[0] == 1595016331402564145


def test_no_builtin_hash_in_phase3_scientific_modules():
    bad = []
    for path in (PKG / "jspace_phase3").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "hash"):
                bad.append(f"{path.relative_to(PKG)}:{node.lineno}")
    assert not bad, f"built-in hash used in scientific package: {bad}"
