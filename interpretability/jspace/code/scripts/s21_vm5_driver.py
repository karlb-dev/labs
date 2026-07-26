# VM5 driver: one OLMo load, two phases. Loading the 32B from the Drive
# HF cache costs 8-15 min wall (64 GB through DriveFS), so s16 (P5
# CoT-rescue, capped) and s17 (P6 seed-1 robustness, trimmed) share a
# single load via a cache shim around sl1_common.load_model. Each phase
# still banks per cell and no-ops on completed cells, so a mid-run VM
# death loses at most one cell.
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sl1_common

_cache = {}
_orig = sl1_common.load_model


def cached_load(key, **kw):
    if key not in _cache:
        _cache[key] = _orig(key, **kw)
    return _cache[key]


sl1_common.load_model = cached_load

import s16_cot_rescue as s16  # noqa: E402
import s17_robustness as s17  # noqa: E402

s16.load_model = cached_load
s17.load_model = cached_load

rc = 0
for name, mod in (("s16", s16), ("s17", s17)):
    try:
        mod.main()
        sl1_common.log(f"driver: {name} complete")
    except Exception:
        rc = 1
        sl1_common.log(f"driver: {name} FAILED\n{traceback.format_exc()}")
        # keep going: a s16 failure must not block the P6 seed-1 grid
sys.exit(rc)
