# special-lab-2 (J-space Part 2) shared utilities. Extends part 1's
# sl1_common — vendored verbatim at part1/ — rather than forking it: the
# part-1 harness (seeds, atomic writes, rank/variant helpers, RUN_DIR /
# RUN_DIR_V2 pointing at the frozen v1/v2 artifacts) is re-exported as-is.
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

LAB_DIR_P2 = Path(__file__).resolve().parent
PART1_DIR = LAB_DIR_P2 / "part1"
sys.path.insert(0, str(PART1_DIR))

from sl1_common import *  # noqa: F401,F403 — the part-1 harness, re-exported
from sl1_common import (  # explicit names the star-import may shadow later
    RUN_DIR, RUN_DIR_V2, die, gpu_mem_gb, log, seed_all,
)

RUN_DIR_P2 = Path(
    os.environ.get(
        "SL2_RUN_DIR",
        "/content/drive/MyDrive/interpret/special-lab-1/part2_20260727",
    )
)
# New Part-2 model downloads go to local NVMe (hub download measured 320 MB/s
# — faster than DriveFS copy) and are deleted between model sets. Part-1
# models + WikiText stay in the Drive cache that sl1_common already set as
# the default HF_HUB_CACHE.
LOCAL_HF = Path(os.environ.get("SL2_LOCAL_HF", "/content/hf_local"))

# Part-2 model matrix. Pins hub-checked 2026-07-27 — see PLAN_PART2.md
# launch addendum (notably: Olmo-3-32B-Instruct does not exist; the matched-
# pretraining instruct is 3.1, same base Olmo-3-1125-32B as the Think donor).
P2_MODELS = {
    "olmo3-think":     {"id": "allenai/Olmo-3-32B-Think",       "cache": "drive"},
    "olmo31-instruct": {"id": "allenai/Olmo-3.1-32B-Instruct",  "cache": "local"},
    "olmo3-base":      {"id": "allenai/Olmo-3-1125-32B",        "cache": "local"},
    "qwen36-27b":      {"id": "Qwen/Qwen3.6-27B",               "cache": "local"},
    "gemma4-31b":      {"id": "google/gemma-4-31B-it",          "cache": "drive"},
}

# Donor lens artifacts (part 1, frozen).
DONOR_LENS = RUN_DIR / "lens" / "olmo32bthink_lens.pt"          # 21 layers, mid band + controls
DONOR_LENS_LATE = RUN_DIR_V2 / "lens" / "olmo32bthink_late.pt"  # {46,50,54,58,62}
MID_BAND = list(range(20, 45, 2))  # the fitted middle band shared with part 1


def p2_metrics_dir(model_slug: str) -> Path:
    """Ensure the Part-2 run-dir layout and return metrics/<model_slug>/."""
    for sub in ("config/prompts", "lens", "figures", "logs",
                "report/handout", "code"):
        (RUN_DIR_P2 / sub).mkdir(parents=True, exist_ok=True)
    mdir = RUN_DIR_P2 / "metrics" / model_slug
    mdir.mkdir(parents=True, exist_ok=True)
    return mdir


def p2_load_model(slug: str, *, dtype=None, device: str = "cuda:0"):
    """Load a matrix model routed to the right HF cache.

    Returns (lens_model, hf_model, tokenizer) like part 1's load_model.
    """
    import torch
    import transformers
    import jlens

    spec = P2_MODELS[slug]
    kw = {"cache_dir": str(LOCAL_HF)} if spec["cache"] == "local" else {}
    t0 = time.time()
    log(f"loading {spec['id']} ({slug}, cache={spec['cache']}) ...")
    tok = transformers.AutoTokenizer.from_pretrained(spec["id"], **kw)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        spec["id"], dtype=dtype or torch.bfloat16, **kw)
    hf = hf.to(device)
    hf.eval()
    used, total = gpu_mem_gb()
    log(f"loaded in {time.time() - t0:.0f}s; VRAM {used:.1f}/{total:.1f} GB")
    return jlens.from_hf(hf, tok), hf, tok


def p2_sync_code() -> None:
    import subprocess
    subprocess.run(["bash", str(LAB_DIR_P2 / "sync_code.sh")], check=False)
