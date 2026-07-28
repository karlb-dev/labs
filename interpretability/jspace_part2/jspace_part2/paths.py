# N1.3 — logical artifact URIs (nextsteps_2_2 §2.4).
#
# THE DEFECT: configs, provenance blocks and registry rows hard-code
# `/content/drive/MyDrive/...` and `/content/models/...`. Those paths are
# properties of one Colab VM, so a "reproduction" on any other machine
# fails at path resolution rather than at science.
#
# Logical URIs name the artifact; a resolver maps them to whatever this
# machine actually has:
#     drive://part2/metrics/olmo3-think/r7_pilot/r7_per_item.parquet
#     drive://lens/olmo31instruct_lens.pt
#     model://allenai/Olmo-3-32B-Think@ebd033e4...
#     repo://interpretability/jspace_part2/configs/r7_protected_pilot.yaml
#     jlens://data/experiments/probe-swap.json
#
# Resolution order is explicit and reported, so a reader can see WHICH
# copy answered. Unresolvable URIs raise with the recipe for obtaining the
# artifact rather than silently producing an empty result.
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# scheme -> ordered candidate roots (first existing wins)
ROOTS: dict[str, list[str]] = {
    "drive": [
        os.environ.get("JSPACE_DRIVE_ROOT", ""),
        "/content/drive/MyDrive/interpret/special-lab-1",
        str(Path.home() / "jspace-drive-mirror"),
    ],
    "repo": [str(REPO_ROOT)],
    "jlens": [
        os.environ.get("JLENS_ROOT", ""),
        "/content/jacobian-lens",
        str(REPO_ROOT / "third_party" / "jacobian-lens"),
    ],
    "model": [
        os.environ.get("JSPACE_MODEL_ROOT", ""),
        "/content/models",
        "/content/hf_local",
    ],
}

# drive:// sub-namespaces, so run-dir reorganisation is one edit here
DRIVE_ALIASES = {
    "part2": "part2_20260727",
    "lens": "part2_20260727/lens",
    "metrics": "part2_20260727/metrics",
    "manifests": "part2_20260727/manifests",
    "figures": "part2_20260727/figures",
    "part1": "2026-07-25_1726",
    "part1v2": "2026-07-26_v2",
}

FETCH_RECIPES = {
    "drive": ("mount the campaign Drive at /content/drive, or set "
              "JSPACE_DRIVE_ROOT to a copy of special-lab-1/"),
    "jlens": ("git clone https://github.com/anthropics/jacobian-lens && "
              "git checkout 581d3986  (or set JLENS_ROOT)"),
    "model": ("hub-download the pinned revision to /content/models/<slug> "
              "(see resume doc §2), or set JSPACE_MODEL_ROOT"),
    "repo": "clone karlb-dev/labs at the evidence item's code_commit",
}


class UnresolvedArtifact(RuntimeError):
    pass


def is_uri(s: str) -> bool:
    return "://" in str(s)


def to_uri(path: str | Path) -> str:
    """Best-effort inverse: rewrite a concrete path as a logical URI so new
    artifacts stop recording machine-specific paths."""
    p = str(path)
    for alias, sub in sorted(DRIVE_ALIASES.items(), key=lambda kv: -len(kv[1])):
        for root in ROOTS["drive"]:
            if root and p.startswith(f"{root}/{sub}/"):
                return f"drive://{alias}/{p[len(root) + len(sub) + 2:]}"
    for root in ROOTS["jlens"]:
        if root and p.startswith(root + "/"):
            return f"jlens://{p[len(root) + 1:]}"
    if p.startswith(str(REPO_ROOT) + "/"):
        return f"repo://{p[len(str(REPO_ROOT)) + 1:]}"
    for root in ROOTS["model"]:
        if root and p.startswith(root + "/"):
            return f"model://local/{p[len(root) + 1:]}"
    return p


def resolve(uri: str | Path, *, must_exist: bool = True) -> Path:
    s = str(uri)
    if not is_uri(s):
        p = Path(s)
        if must_exist and not p.exists():
            raise UnresolvedArtifact(f"{s} does not exist (and is not a URI)")
        return p
    scheme, rest = s.split("://", 1)
    if scheme not in ROOTS:
        raise UnresolvedArtifact(f"unknown scheme {scheme!r} in {s!r}")
    if scheme == "drive":
        head, _, tail = rest.partition("/")
        rest = f"{DRIVE_ALIASES.get(head, head)}/{tail}" if tail else \
            DRIVE_ALIASES.get(head, head)
    if scheme == "model":
        return _resolve_model_uri(rest, must_exist=must_exist)
    tried = []
    for root in ROOTS[scheme]:
        if not root:
            continue
        cand = Path(root) / rest
        tried.append(str(cand))
        if cand.exists():
            return cand
    if not must_exist and tried:
        return Path(tried[0])
    raise UnresolvedArtifact(
        f"could not resolve {s!r}\n  tried: {tried}\n  fetch: "
        f"{FETCH_RECIPES.get(scheme, 'unknown')}")


def _resolve_model_uri(rest: str, *, must_exist: bool) -> Path:
    """model://<org>/<name>@<revision>  or  model://local/<dir>"""
    ref, _, revision = rest.partition("@")
    if ref.startswith("local/"):
        sub = ref[len("local/"):]
        for root in ROOTS["model"]:
            if root and (Path(root) / sub).exists():
                return Path(root) / sub
        raise UnresolvedArtifact(f"local model dir {sub!r} not found; "
                                 f"{FETCH_RECIPES['model']}")
    cache_name = "models--" + ref.replace("/", "--")
    for cache in (os.environ.get("HF_HUB_CACHE"), "/content/hf_local",
                  "/content/drive/MyDrive/hf_cache/hub"):
        if not cache:
            continue
        snap = Path(cache) / cache_name / "snapshots" / revision
        if snap.exists():
            return snap
    # a plain local dir whose config hash matches the revision also counts
    from .provenance import resolve_model
    for root in ROOTS["model"]:
        if not root or not Path(root).exists():
            continue
        for cand in Path(root).iterdir():
            if not cand.is_dir():
                continue
            info = resolve_model(str(cand))
            if info.get("revision") == revision:
                return cand
    if not must_exist:
        return Path("/content/models") / ref.split("/")[-1]
    raise UnresolvedArtifact(
        f"model {ref}@{revision} not present; {FETCH_RECIPES['model']}")


def resolution_report(uris) -> dict:
    out = {}
    for u in uris:
        try:
            out[str(u)] = {"resolved": str(resolve(u)), "ok": True}
        except UnresolvedArtifact as e:
            out[str(u)] = {"resolved": None, "ok": False, "error": str(e)}
    return out
