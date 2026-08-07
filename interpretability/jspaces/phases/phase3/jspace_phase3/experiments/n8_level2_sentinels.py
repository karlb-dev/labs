# N8 Level 2 — model-cell SENTINEL reproduction (nextsteps §13, a
# public-release gate). For one model: relaunch the FROZEN Phase 2 N6
# grid command byte-for-byte (same config, same frozen inputs, IO
# redirected to a fresh root, --no-register), let it process >=
# `n_sentinel` items, terminate it, and compare every lp column of the
# sentinel rows against the frozen parquet.
#
# The sentinel is an exact PREFIX of the full run under the identical
# code path — no Phase 2 code is modified (Phase 2 is immutable); the
# grid's own resume machinery makes a prefix well-defined.
#
# Usage:
#   python -m jspace_phase3.experiments.n8_level2_sentinels --slug <slug> \
#       [--n 20]
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from ..paths3 import metrics_dir, phase2_run_root, run_root
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           write_result3)

TIER = "methods"
P2_ROOT = str(phase2_run_root())
TOL = 2e-3


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    require_clean_tree("--allow-dirty" in sys.argv)
    slug = arg("--slug")
    n_sent = int(arg("--n", 20))
    full = "--full" in sys.argv          # Level 3: run the whole cell
    fresh = run_root() / "n8_level2" / slug
    if fresh.exists():
        shutil.rmtree(fresh)
    fresh.mkdir(parents=True)
    # the producer resolves manifest/partition INPUTS under RUN_ROOT as
    # well as its state — scaffold the fresh root with read-only links
    # to the frozen cross-model inputs; per-slug state dirs stay absent
    # so measurement starts from item 1
    (fresh / "metrics").mkdir()
    os.symlink(Path(P2_ROOT) / "metrics" / "cross_model",
               fresh / "metrics" / "cross_model")

    env = dict(os.environ)
    # RUN_ROOT carries the producer's STATE as well as outputs, so it
    # must point at the FRESH root or the grid resumes the frozen VM9
    # state and re-measures nothing (then dies re-registering the
    # original id). Frozen INPUTS (manifest/partition/lens) resolve
    # through the separate JSPACE_DRIVE_ROOT alias layer, untouched.
    env["JSPACE_PART2_RUN_ROOT"] = str(fresh)
    env["JSPACE_PART2_OUT_ROOT"] = str(fresh)
    # the producer takes out_dir (state + outputs) from its CONFIG as an
    # absolute path — env redirection alone cannot give it fresh state.
    # The sentinel runs a derived config: byte-identical to the frozen
    # one except out_dir -> fresh and a sentinel evidence_id (never
    # registered: the run is terminated at the sentinel count).
    import yaml
    frozen_cfg = Path(f"interpretability/jspaces/phases/phase2/configs/"
                      f"n6_grid_{slug}.yaml")
    cfg_d = yaml.safe_load(frozen_cfg.read_text())
    cfg_d["out_dir"] = str(fresh / "metrics" / slug / "n6_grid")
    cfg_d["evidence_id"] = f"{cfg_d['evidence_id']}-n8-sentinel"
    cfg_d["tier"] = "methods"   # if a full (Level 3) run reaches its own
    # registration, the row lands clearly labeled, never confirmatory
    cfg = str(fresh / "sentinel_config.yaml")
    Path(cfg).write_text(yaml.safe_dump(cfg_d, sort_keys=False))
    cmd = [sys.executable, "-m",
           "jspace_part2.experiments.confirmatory_protected_grid",
           "--config", cfg]
    log(f"sentinel launch: {' '.join(cmd[2:])}")
    sub_log = fresh / "sentinel_subprocess.log"
    log_fh = open(sub_log, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_fh,
                            stderr=subprocess.STDOUT, text=True)
    state_p = fresh / "metrics" / slug / "n6_grid" / "n6_state.json"
    t0 = time.time()
    n_done = 0
    while proc.poll() is None:
        time.sleep(20)
        if state_p.exists():
            try:
                n_done = len(json.loads(state_p.read_text())["done"])
            except (json.JSONDecodeError, KeyError):
                continue
            if not full and n_done >= n_sent:
                proc.send_signal(signal.SIGTERM)
                break
        if time.time() - t0 > (7200 if full else 3600):
            proc.kill()
            raise RuntimeError("sentinel run exceeded 1 h before "
                               f"{n_sent} items (reached {n_done})")
    proc.wait(timeout=120)
    log_fh.close()
    if not state_p.exists():
        raise RuntimeError(
            "sentinel subprocess produced no state; its log tail:\n"
            + "\n".join(sub_log.read_text().splitlines()[-15:]))
    state = json.loads(state_p.read_text())
    rows = pd.DataFrame(state["rows"])
    log(f"sentinel produced {rows.item_id.nunique()} items")

    frozen = pd.read_parquet(
        Path(P2_ROOT) / "metrics" / slug / "n6_grid" /
        f"n6_per_item_{slug}.parquet")
    lp_cols = [c for c in rows.columns if c.startswith("lp_")]
    key_cols = ["item_id", "condition"] if "condition" in rows.columns \
        else ["item_id"]
    merged = rows.merge(frozen, on=key_cols, suffixes=("_new", "_frz"))
    devs = {}
    for c in lp_cols:
        a, b = f"{c}_new", f"{c}_frz"
        if a in merged.columns and b in merged.columns:
            sub = merged[[a, b]].dropna()
            if len(sub):
                devs[c] = round(float((sub[a] - sub[b]).abs().max()), 8)
    worst = max(devs.values()) if devs else float("nan")
    passed = bool(devs) and worst <= TOL
    payload = {"slug": slug, "n_sentinel_items": int(
        rows.item_id.nunique()),
        "n_compared_rows": int(len(merged)),
        "max_abs_deviation_per_col": devs,
        "worst": worst, "tolerance": TOL, "pass": passed}
    level = 3 if full else 2
    eid = f"p3-n8-level{level}-{slug}-v1"
    cmdstr = (f"python -m jspace_phase3.experiments.n8_level2_sentinels "
              f"--slug {slug} --n {n_sent}"
              + (" --full" if full else ""))
    out = metrics_dir("cross_model") / f"n8_level2_{slug}.json"
    write_result3(payload, out, Provenance3(
        evidence_id=eid, tier=TIER, command=cmdstr, seed=0))
    register(eid, tier=TIER, command=cmdstr,
             what=(f"N8 Level {level} {'full-cell' if full else 'sentinel'} on {slug}: "
                   f"{payload['n_sentinel_items']} items re-measured "
                   f"under the frozen N6 command; worst |dev| {worst} "
                   f"(tol {TOL}) — {'PASS' if passed else 'FAIL'}"),
             outputs=[out])
    print(json.dumps(payload, indent=1))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
