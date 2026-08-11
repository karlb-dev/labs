"""CPU self-tests (no GPU / no model weights required)."""
from __future__ import annotations

from .paths import (
    EXPECTED_D_MODEL,
    EXPECTED_N_LAYERS,
    FINAL_LAYER,
    FIT_SOURCE_LAYERS,
    PAPER_BAND,
)
from .util import sha256_json


def run() -> int:
    checks = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        checks.append((name, cond, detail))
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    check("n_layers_52", EXPECTED_N_LAYERS == 52)
    check("d_model_6656", EXPECTED_D_MODEL == 6656)
    check("final_layer_51", FINAL_LAYER == 51)
    check("fit_sources_count_21", len(FIT_SOURCE_LAYERS) == 21)
    check("final_not_in_sources", FINAL_LAYER not in FIT_SOURCE_LAYERS)
    check("sources_in_range", all(0 <= L < EXPECTED_N_LAYERS for L in FIT_SOURCE_LAYERS))
    check("paper_band_nonempty", len(PAPER_BAND) >= 8)
    check("paper_band_in_range", all(0 <= L < EXPECTED_N_LAYERS for L in PAPER_BAND))
    check("sha256_json_stable", sha256_json({"a": 1, "b": [2, 3]}) ==
          sha256_json({"b": [2, 3], "a": 1}))

    # import package surface
    import jspace_muse
    import jspace_muse.adapters
    import jspace_muse.readout
    import jspace_muse.experiments.fit
    import jspace_muse.experiments.battery
    check("imports_ok", True, jspace_muse.__version__)

    n_pass = sum(1 for _, c, _ in checks if c)
    n_fail = sum(1 for _, c, _ in checks if not c)
    print(f"selftest: {n_pass} passed, {n_fail} failed / {len(checks)}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run())
