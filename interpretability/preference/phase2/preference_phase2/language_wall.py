"""Claim ceiling + language wall v2 (plan §8).

Recursive scan over governed prose; FAILURES RAISE (a JSON status of
REVIEW without a failing exit code is not sufficient — plan §8). Quoted
forbidden-language lists are recognized as ceiling context.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

FORBIDDEN_PHRASES = (
    "really prefers",
    "the model wants",
    "model wants",
    "the model consented",
    "the model suffered",
    "the model was upset",
    "welfare of the model",
    "model welfare",
    "moral patienthood",
    "true introspection",
    "preference workspace",
    "workspace of wants",
    "no preferences in any sense",
    "subjective experience",
)

# a forbidden phrase within this window after an allowed-context token is
# quoted ceiling text, not a violation
_ALLOWED_CONTEXT = (
    "forbidden", "never", "not licensed", "ceiling", "not establish",
    "no artifact", "banned", "upgrade", "prohibited",
)
_WINDOW = 280

GOVERNED_GLOBS = (
    "plans/*.md",
    "phase2/preregistration/*.md",
    "phase2/reports/**/*.md",
    "phase2/reports/*.md",
    "phase2/protocol/*.md",
    "phase2/README.md",
    "README.md",
)


def scan_text(text: str, *, source: str = "") -> list[dict[str, Any]]:
    """Whitespace-normalized scan: phrases and allowed-context tokens are
    matched across line wraps (a phrase split by a newline must neither
    evade the wall nor lose its quoting context)."""
    import re

    positions: list[int] = []
    chars: list[str] = []
    prev_space = False
    for i, ch in enumerate(text.lower()):
        if ch.isspace():
            if prev_space:
                continue
            ch = " "
            prev_space = True
        else:
            prev_space = False
        chars.append(ch)
        positions.append(i)
    norm = "".join(chars)
    hits = []
    for phrase in FORBIDDEN_PHRASES:
        start = 0
        while True:
            idx = norm.find(phrase, start)
            if idx < 0:
                break
            window = norm[max(0, idx - _WINDOW): idx]
            if not any(tok in window for tok in _ALLOWED_CONTEXT):
                orig = positions[idx]
                line = text.count("\n", 0, orig) + 1
                hits.append({"source": source, "phrase": phrase,
                             "line": line,
                             "excerpt": text[max(0, orig - 60): orig + 60]
                             .replace("\n", " ")})
            start = idx + len(phrase)
    return hits


def scan_campaign(root: pathlib.Path) -> dict[str, Any]:
    hits: list[dict[str, Any]] = []
    scanned = 0
    for pattern in GOVERNED_GLOBS:
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            scanned += 1
            hits.extend(scan_text(path.read_text(encoding="utf-8"),
                                  source=str(path.relative_to(root))))
    return {"files_scanned": scanned, "hits": hits,
            "status": "clean" if not hits else "FAIL"}


def scan_and_raise(root: pathlib.Path,
                   extra_texts: dict[str, str] | None = None) -> dict[str, Any]:
    """The raising entry point (plan §8: failures raise)."""
    result = scan_campaign(root)
    for name, text in (extra_texts or {}).items():
        result["hits"].extend(scan_text(text, source=name))
    if result["hits"]:
        result["status"] = "FAIL"
        raise LanguageWallError(
            f"language wall: {len(result['hits'])} forbidden-language hits; "
            f"first: {result['hits'][0]}")
    return result


class LanguageWallError(RuntimeError):
    pass


def main(argv: list[str] | None = None) -> int:
    from . import paths
    root = paths.campaign_root()
    try:
        result = scan_and_raise(root)
    except LanguageWallError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"language wall clean: {result['files_scanned']} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
