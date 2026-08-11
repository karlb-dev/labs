"""Path contract for the Muse Glimmer jspace sideline."""
from __future__ import annotations

import os
from pathlib import Path

_THIS = Path(__file__).resolve()
STUDY_ROOT = _THIS.parents[1]  # .../sidelines/muse
REPO_ROOT = next(
    p for p in (_THIS, *_THIS.parents) if (p / ".git").exists()
)

REPORTS = STUDY_ROOT / "reports"
RELEASE = STUDY_ROOT / "release"
CONFIGS = STUDY_ROOT / "configs"
EVENTS = REPORTS / "evidence_events.jsonl"

LOCAL_WORK = Path(os.environ.get("JSPACE_MUSE_LOCAL_WORK", "/content/muse_work"))
DRIVE_ROOT = Path(
    os.environ.get(
        "JSPACE_MUSE_DRIVE_ROOT",
        "/content/drive/MyDrive/interpret/special-lab-1/jspace-muse",
    )
)
HF_LOCAL = Path(os.environ.get("JSPACE_MUSE_HF_LOCAL", "/content/hf_local"))
JLENS_CLONE = Path(
    os.environ.get("JLENS_ROOT", "/content/jacobian-lens")
)
UPSTREAM_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"

# Pinned Muse identity (hub revision at study launch 2026-08-11)
MUSE_MODEL_ID = "meta-models/Muse-Glimmer-30B"
MUSE_MODEL_REVISION = "97c77dff50b2797bcc558fa2d909761dbc575c59"

# Phase-2-style fitting corpus (read-only from special-lab-1)
FIT_CORPUS = Path(
    os.environ.get(
        "JSPACE_MUSE_FIT_CORPUS",
        "/content/drive/MyDrive/interpret/special-lab-1/"
        "2026-07-25_1726/config/prompts/fitting_corpus.jsonl",
    )
)

# Architecture expectations from the model card / config.json
EXPECTED_N_LAYERS = 52
EXPECTED_D_MODEL = 6656
FINAL_LAYER = 51  # 0-indexed last residual block

# 21 source layers on 0..50 (Phase-2 density, retargeted for 52-layer stack)
FIT_SOURCE_LAYERS = [
    2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    32, 36, 40, 44, 48, 50,
]
assert FINAL_LAYER not in FIT_SOURCE_LAYERS
assert all(0 <= L < EXPECTED_N_LAYERS for L in FIT_SOURCE_LAYERS)

# Paper-relative workspace band mapped to 52 layers:
# Claude paper band ~38-92% of depth → 0.38*51..0.92*51 ≈ 19..47
PAPER_BAND = [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46]
# Full depth grid for J-vs-logit profiles
DEPTH_GRID = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 51]


def model_snapshot() -> Path:
    """HF hub-cache snapshot path for the pinned Muse revision."""
    org, name = MUSE_MODEL_ID.split("/")
    return (
        HF_LOCAL
        / f"models--{org}--{name}"
        / "snapshots"
        / MUSE_MODEL_REVISION
    )


def ensure_dirs() -> None:
    for d in (
        LOCAL_WORK,
        DRIVE_ROOT / "lens",
        DRIVE_ROOT / "metrics",
        DRIVE_ROOT / "logs",
        DRIVE_ROOT / "figures",
        DRIVE_ROOT / "manifests",
        DRIVE_ROOT / "reports",
        DRIVE_ROOT / "release",
        DRIVE_ROOT / "config",
        DRIVE_ROOT / "raw",
        REPORTS,
        RELEASE,
    ):
        d.mkdir(parents=True, exist_ok=True)
