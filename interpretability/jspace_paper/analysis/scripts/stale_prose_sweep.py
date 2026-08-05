#!/usr/bin/env python3
"""P1: automatic stale-prose detector (plan §4.3 + FREEZE_HANDOFF list).

Sweeps text paper sources (.tex/.md) for phrases made stale by the frozen
terminal record. Emits reports/stale_prose_hits.json for triage; the
dispositioned register is reports/STALE_PAPER_PROSE_REGISTER.md.
"""
import json
import re
from pathlib import Path

REPO = Path("/content/labs")
ANALYSIS = REPO / "interpretability/jspace_paper/analysis"

PATTERNS = [
    ("A1000_pending", r"A1000\s+(?:is\s+)?(?:pending|queued)"),
    ("continue_to_a500_a1000", r"continue\s+to\s+A500\s+or\s+A1000|A500\s*(?:→|->|then)\s*A1000\s+(?:queued|planned)"),
    ("canonical_qwen_lens", r"canonical\s+(?:sparse\s+)?(?:Qwen\s+)?A?1000\s+lens|canonical\s+Qwen\s+lens"),
    ("wedge_queued", r"(?:SFT/DPO|wedge)[^.\n]{0,60}?(?:queued|pending|will\s+be\s+run|remains?\s+to\s+be\s+run)|queued[^.\n]{0,40}wedge"),
    ("h6_queued", r"H6[^.\n]{0,60}?(?:queued|pending|planned|will\s+(?:test|run))|(?:queued|pending)[^.\n]{0,40}H6"),
    ("gemma_blocker_terminal", r"(?:methods\s+blocker|blocker)[^.\n]{0,80}(?:terminal|final|remains|stands)|stopped\s+at\s+a\s+blocker"),
    ("gemma_1e5_terminal", r"0\.002458[^\n]{0,120}1e-?5|1e-?5[^\n]{0,120}0\.002458"),
    ("bank_w_pending", r"Bank[\s-]?W[^.\n]{0,80}?(?:pending|queued|awaits|will\s+run)"),
    ("o5_queued", r"O5[^.\n]{0,50}?(?:queued|pending|planned)"),
    ("phase4_primary", r"Phase[\s-]?4[^.\n]{0,40}?(?:confirmatory\s+)?primar(?:y|ies)"),
    ("a2000", r"A2000"),
    ("tier2_wedge_queued", r"Tier[\s-]?2[^.\n]{0,60}(?:refit|lens)[^.\n]{0,60}(?:queued|pending|remains\s+available)"),
]

SUFFIXES = {".tex", ".md"}


def iter_sources():
    manifest = json.load(open(ANALYSIS / "manifests/paper_source_manifest.json"))
    for row in manifest["sources"]:
        if row.get("status") != "present":
            continue
        p = Path(row["path"])
        if p.suffix.lower() in SUFFIXES:
            yield REPO / p


def main():
    hits = []
    for path in iter_sources():
        rel = str(path.relative_to(REPO))
        text = path.read_text(errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            for name, pat in PATTERNS:
                if re.search(pat, line, flags=re.IGNORECASE):
                    hits.append({
                        "file": rel, "line": i, "pattern": name,
                        "text": line.strip()[:300],
                    })
    out = ANALYSIS / "reports/stale_prose_hits.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(hits, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"{len(hits)} hits across {len(set(h['file'] for h in hits))} files")
    for h in hits:
        print(f"  {h['file']}:{h['line']} [{h['pattern']}] {h['text'][:110]}")


if __name__ == "__main__":
    main()
