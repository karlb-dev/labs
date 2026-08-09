"""Path contract for the official-repro study.

Allowed writes: the study namespace, the study Drive root, and the local
scratch root. Everything else in the campaign is read-only input.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

_THIS = Path(__file__).resolve()
STUDY_ROOT = _THIS.parents[1]  # .../sidelines/official_repro


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        git_marker = candidate / ".git"
        if git_marker.exists():
            return candidate
    raise RuntimeError(f"no git repository above {start}")


REPO_ROOT = _find_repo_root(STUDY_ROOT)

EXTERNAL = STUDY_ROOT / "external/anthropics_jacobian_lens_581d3986"
EXPERIMENTS_DIR = EXTERNAL / "data/experiments"
EVALUATIONS_DIR = EXTERNAL / "data/evaluations"
PROTOCOL = STUDY_ROOT / "protocol"
PREREGISTRATION = STUDY_ROOT / "preregistration"
CONFIGS = STUDY_ROOT / "configs"
REPORTS = STUDY_ROOT / "reports"
RELEASE = STUDY_ROOT / "release"
DATA = STUDY_ROOT / "data"
FIT_DATA = DATA / "fit"
DERIVED_TARGETS = DATA / "derived_targets"
EVENTS = REPORTS / "evidence_events.jsonl"

#: Immutable upstream clone (never edited; byte-verified against the vendored
#: record). Overridable for machines that keep it elsewhere.
JLENS_CLONE = Path(os.environ.get("JLENS_ROOT", "/content/or1_work/jacobian-lens"))
UPSTREAM_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"

#: Local (non-DriveFS) scratch for model-scale intermediates.
LOCAL_WORK = Path(os.environ.get("JSPACE_OR1_LOCAL_WORK", "/content/or1_work"))

#: Durable study Drive root. The date suffix is frozen at study launch; a
#: resumed session on a new VM must point at the same folder.
DRIVE_ROOT = Path(
    os.environ.get(
        "JSPACE_OR1_DRIVE_ROOT",
        "/content/drive/MyDrive/interpret/special-lab-1/official_repro_1_20260808",
    )
)

#: HF hub cache used for pinned model downloads (hub layout, local NVMe).
HF_LOCAL = Path(os.environ.get("JSPACE_OR1_HF_LOCAL", "/content/hf_local"))

#: Read-only frozen campaign comparator (plan §2.3); never written.
CAMPAIGN_OLMO_LENS = Path(
    os.environ.get(
        "JSPACE_OR1_CAMPAIGN_OLMO_LENS",
        "/content/drive/MyDrive/interpret/special-lab-1/part2_20260727/lens/"
        "olmo31instruct_lens.pt",
    )
)
CAMPAIGN_OLMO_LENS_SHA256 = (
    "e0f8b972a9f1f884101f94ff52a1938d5cfa7a5f49e987e6768826f2337c6dfb"
)

QWEN_MODEL_ID = "Qwen/Qwen3.6-27B"
QWEN_MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
OLMO_MODEL_ID = "allenai/Olmo-3.1-32B-Instruct"
OLMO_MODEL_REVISION = "ac0587e4a7744a551c059d8cd17ba220bc940dae"

QWEN_LENS_REPO = "neuronpedia/jacobian-lens"
QWEN_LENS_REVISION = "a4114d7752d11eb546e6cf372213d7e75526d3a1"
QWEN_LENS_FILENAME = (
    "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"
)
QWEN_LENS_SHA256 = "1718c8c52dd8a9dad03738d4d625937c1fbba10be325b872ed446c7290fc11e1"
QWEN_LENS_BYTES = 3303032772


def model_snapshot(repo_id: str, revision: str) -> Path:
    """Hub-cache snapshot directory for a pinned model."""
    return (
        HF_LOCAL
        / f"models--{repo_id.replace('/', '--')}"
        / "snapshots"
        / revision
    )


def verify_upstream_clone() -> str:
    """Assert the immutable clone is at the pinned commit; return the SHA."""
    head = subprocess.check_output(
        ["git", "-C", str(JLENS_CLONE), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"jacobian-lens clone at {JLENS_CLONE} is at {head}, "
            f"expected pinned {UPSTREAM_COMMIT}"
        )
    status = subprocess.check_output(
        ["git", "-C", str(JLENS_CLONE), "status", "--porcelain"], text=True
    ).strip()
    if status:
        raise RuntimeError(f"jacobian-lens clone is dirty:\n{status}")
    return head
