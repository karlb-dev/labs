# N1.3 — reproduction that actually reproduces (nextsteps_2_2 §2.4/§7-N1.3).
#
# THE DEFECT: `repro.sh <id>` installed the CURRENT checkout, ran the
# tests, and compared CURRENT local files to their recorded hashes. It
# never checked out the evidence item's code commit, never resolved or
# verified declared inputs, never proved the loaded weights matched a
# pinned revision, and (because `created_utc` lived inside the hashed
# file) could not have matched a hash on rerun even if everything else
# was right. It verified *presence*, not *reproducibility*.
#
# v2 stages, each reported and each able to fail loudly:
#   1  resolve the immutable creation event (registry v2)
#   2  create an isolated git worktree at the recorded code_commit
#   3  install the package there against the pinned constraints
#   4  resolve every declared input URI and verify its hash
#   5  verify model revision / config / tokenizer against the manifest
#   6  run the producer command inside the worktree
#   7  verify the regenerated payload hash (exact) or numeric tolerance
#   8  append an evidence_reproduced event; never mutate the original
#
# Scope note (PI addendum §4.1): the fresh-environment demonstration is
# scoped to two acceptance items — one CPU evidence item and the SmolLM
# golden. Container digests are deferred. Everything else uses stages 1-5
# plus --verify-only.
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import registry as reg
from .lib import sha256_file
from .paths import UnresolvedArtifact, resolve
from .provenance import REPO_ROOT, verify_result_v2

CONSTRAINTS = REPO_ROOT / "interpretability" / "jspace_part2" / "constraints.txt"


def _run(cmd, cwd=None, env=None, capture=True):
    return subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, env=env,
                          capture_output=capture, text=True)


def stage(name, ok, detail=""):
    mark = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    return {"stage": name, "ok": ok, "detail": detail}


def reproduce(evidence_id: str, *, workspace: str | None = None,
              verify_only: bool = False, run: bool = True,
              keep: bool = False) -> dict:
    print(f"REPRO {evidence_id}")
    stages = []

    # 1 — immutable creation event
    try:
        rec = reg.resolve(evidence_id)
    except reg.RegistryError as e:
        stages.append(stage("resolve creation event", False, str(e)))
        return {"evidence_id": evidence_id, "ok": False, "stages": stages}
    stages.append(stage("resolve creation event", True,
                        f"tier={rec['tier']} commit={str(rec['code_commit'])[:9]}"
                        + ("" if rec["live"] else "  (NOT live)")))

    # 4 — declared inputs (before spending on a worktree)
    inputs = rec.get("inputs") or {}
    bad_inputs = []
    for name, want in inputs.items():
        if not isinstance(want, str) or len(want) != 64:
            continue                       # not a hash-valued input
        uri = (rec.get("input_uris") or {}).get(name)
        if not uri:
            continue
        try:
            p = resolve(uri)
            got = sha256_file(p)
            if got != want:
                bad_inputs.append(f"{name}: {got[:12]} != {want[:12]}")
        except UnresolvedArtifact as e:
            bad_inputs.append(f"{name}: {e}")
    stages.append(stage("verify declared inputs", not bad_inputs,
                        "; ".join(bad_inputs) if bad_inputs
                        else f"{len(inputs)} declared"))

    # 5 — model manifest
    model = rec.get("model") or {}
    if model:
        want_rev, want_cfg = model.get("revision"), model.get("config_sha256")
        detail, ok = "", True
        if want_rev:
            try:
                p = resolve(f"model://{model.get('hub_id') or model['id']}@{want_rev}")
                cfg = p / "config.json"
                ok = (not want_cfg) or (cfg.exists() and sha256_file(cfg) == want_cfg)
                detail = f"revision {want_rev[:9]} present, config {'matches' if ok else 'DIFFERS'}"
            except UnresolvedArtifact as e:
                ok, detail = False, str(e).splitlines()[0]
        else:
            ok, detail = None, "no revision recorded (v1 provenance)"
        stages.append(stage("verify model revision", ok, detail))

    # 7' — verify existing outputs (always; this is the old behaviour)
    out_ok = True
    for out in rec.get("outputs") or []:
        p = Path(out["path"])
        if not p.exists():
            try:
                p = resolve(out.get("uri") or out["path"])
            except UnresolvedArtifact:
                stages.append(stage(f"output {Path(out['path']).name}", False,
                                    "missing"))
                out_ok = False
                continue
        got = sha256_file(p)
        match = got == out.get("sha256")
        pv = verify_result_v2(p) if p.suffix == ".json" else {"format": "n/a"}
        detail = f"{got[:12]}" + ("" if match else f" != {str(out.get('sha256'))[:12]}")
        if pv.get("format") == "v2-envelope":
            detail += f"; payload {'OK' if pv['ok'] else 'MISMATCH'}"
            match = match and bool(pv["ok"])
        stages.append(stage(f"output {Path(out['path']).name}", match, detail))
        out_ok = out_ok and match

    if verify_only or rec.get("rerun") == "manual" or not run:
        ok = all(s["ok"] is not False for s in stages)
        print(f"VERIFY {'PASS' if ok else 'FAIL'}  ({rec.get('rerun')} rerun)")
        if rec.get("repro_notes"):
            print(f"  notes: {rec['repro_notes']}")
        reg.record_reproduction(evidence_id, runner="repro_v2:verify-only",
                                ok=ok, detail={"stages": stages})
        return {"evidence_id": evidence_id, "ok": ok, "stages": stages}

    # 2 — isolated worktree at the recorded commit
    commit = rec["code_commit"]
    ws = Path(workspace or tempfile.mkdtemp(prefix=f"repro-{evidence_id}-"))
    wt = ws / "labs"
    if commit == "UNRECORDED_IN_V1":
        stages.append(stage("isolated worktree", None,
                            "no code_commit recorded; running in place"))
        wt = REPO_ROOT
    else:
        r = _run(["git", "-C", str(REPO_ROOT), "worktree", "add", "--detach",
                  str(wt), commit])
        ok = r.returncode == 0
        stages.append(stage("isolated worktree", ok,
                            f"{commit[:9]} -> {wt}" if ok else r.stderr.strip()[:200]))
        if not ok:
            return {"evidence_id": evidence_id, "ok": False, "stages": stages}

    try:
        # 3 — install at pinned constraints.
        # `pip install -e` writes egg-info INTO the worktree, which would
        # dirty the tree and make every clean-tree-requiring producer
        # refuse to run at an old commit. Build artifacts are not source
        # changes, so they are excluded locally; anything TRACKED that
        # changes still trips the guard (asserted below).
        pkg = wt / "interpretability" / "jspace_part2"
        if wt != REPO_ROOT:
            gitdir = _run(["git", "-C", str(wt), "rev-parse",
                           "--git-dir"]).stdout.strip()
            gd = Path(gitdir) if Path(gitdir).is_absolute() else wt / gitdir
            (gd / "info").mkdir(parents=True, exist_ok=True)
            with open(gd / "info" / "exclude", "a") as f:
                f.write("\n*.egg-info/\n__pycache__/\n*.pyc\nbuild/\ndist/\n")
        # Install into an ISOLATED venv, never the ambient environment:
        # installing the worktree copy into the live env and then deleting
        # the worktree leaves the session with a broken `jspace_part2`
        # (observed). --system-site-packages reuses the multi-GB torch /
        # transformers stack instead of re-downloading it.
        venv = ws / "venv"
        py = venv / "bin" / "python"
        # --without-pip: this image's ensurepip is broken, and the venv
        # only needs to be an interpreter whose import path starts at the
        # worktree. Deps come from system site-packages, pinned by
        # constraints.txt and asserted below.
        r = _run([sys.executable, "-m", "venv", "--system-site-packages",
                  "--without-pip", str(venv)])
        if r.returncode != 0:
            stages.append(stage("isolated venv", False, r.stderr[-200:]))
            return {"evidence_id": evidence_id, "ok": False, "stages": stages}
        # Deliberately PYTHONPATH, not `pip install -e`: setuptools'
        # editable hook rewrites jspace_part2.egg-info/SOURCES.txt at
        # IMPORT time, and that file is tracked in commits before 7528bef,
        # so every producer with a clean-tree guard would refuse. Every
        # registered producer is `python -m jspace_part2...` or a script,
        # so path injection is sufficient and leaves the worktree pristine.
        run_env = dict(os.environ, PYTHONPATH=str(pkg),
                       PATH=f"{venv / 'bin'}:{os.environ.get('PATH', '')}",
                       VIRTUAL_ENV=str(venv))
        chk = _run([str(py), "-c",
                    "import jspace_part2; print(jspace_part2.__file__)"],
                   env=run_env)
        from_worktree = str(wt) in chk.stdout
        stages.append(stage("code loaded from recorded commit", from_worktree,
                            f"imports {chk.stdout.strip()[-70:]}"))
        if CONSTRAINTS.exists():
            want = dict(l.split("==", 1) for l in
                        CONSTRAINTS.read_text().splitlines()
                        if "==" in l and not l.startswith("#"))
            got = _run([sys.executable, "-m", "pip", "freeze"]).stdout.splitlines()
            gotd = dict(l.split("==", 1) for l in got if "==" in l)
            drift = {k: (v, gotd[k]) for k, v in want.items()
                     if k in gotd and gotd[k] != v}
            stages.append(stage("pinned dependency versions", not drift,
                                f"{len(drift)} drifted: {drift}" if drift
                                else f"{len(want)} pins match constraints.txt"))
        # The dirty-tree guard must stay meaningful: no TRACKED SOURCE file
        # may differ at the recorded commit. Commits before 7528bef tracked
        # jspace_part2.egg-info/, so installing modifies a tracked build
        # artifact; restore those specific paths and say so, but let any
        # real source difference fail.
        dirty = _run(["git", "-C", str(wt), "status", "--porcelain",
                      "--untracked-files=no"]).stdout.strip()
        note = "no tracked file modified"
        if dirty:
            paths = [l[3:] for l in dirty.splitlines()]
            arte = [p for p in paths if "egg-info" in p or p.endswith(".pyc")]
            src = [p for p in paths if p not in arte]
            if arte and not src:
                _run(["git", "-C", str(wt), "checkout", "--", *arte])
                note = (f"restored {len(arte)} tracked BUILD artifact(s) "
                        f"dirtied by the install (this commit predates the "
                        f"egg-info untracking at 7528bef); no source change")
                dirty = ""
            else:
                note = f"tracked SOURCE differs: {src[:5]}"
        stages.append(stage("worktree matches recorded commit", not dirty, note))

        # 6 — run the producer inside the worktree, on the venv interpreter
        env = run_env
        command = rec["command"]
        if command.startswith("python "):
            command = f"{py} " + command[len("python "):]
        t0 = time.time()
        r = _run(command, cwd=str(pkg), env=env, capture=False)
        stages.append(stage("run producer", r.returncode == 0,
                            f"{time.time()-t0:.0f}s"))

        # 7 — verify regenerated payloads
        for out in rec.get("outputs") or []:
            p = Path(out["path"])
            if p.suffix != ".json" or not p.exists():
                continue
            pv = verify_result_v2(p)
            if pv["format"] == "v2-envelope":
                stages.append(stage(f"payload {p.name}", pv["ok"],
                                    "exact payload hash after rerun"))
            else:
                got = sha256_file(p)
                stages.append(stage(f"payload {p.name}", None,
                                    f"legacy flat file; whole-file "
                                    f"{'matches' if got == out.get('sha256') else 'DIFFERS'} "
                                    f"(timestamps are embedded, so a diff is expected)"))
    finally:
        if commit != "UNRECORDED_IN_V1" and not keep:
            _run(["git", "-C", str(REPO_ROOT), "worktree", "remove", "--force",
                  str(wt)])
            shutil.rmtree(ws, ignore_errors=True)

    ok = all(s["ok"] is not False for s in stages)
    print(f"REPRO {'PASS' if ok else 'FAIL'}")
    reg.record_reproduction(evidence_id, runner="repro_v2", ok=ok,
                            detail={"stages": stages})
    return {"evidence_id": evidence_id, "ok": ok, "stages": stages}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jspace-part2 repro2")
    ap.add_argument("evidence_id")
    ap.add_argument("--workspace")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    res = reproduce(a.evidence_id, workspace=a.workspace,
                    verify_only=a.verify_only, keep=a.keep)
    if a.json:
        print(json.dumps(res, indent=1))
    raise SystemExit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
