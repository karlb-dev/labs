# Pinned wikipedia reference fetcher for Bank F verification.
#
# WHY THIS EXISTS: the Drive-cached wikimedia/wikipedia 20231101.en
# parquets are hub-authentic (shard sha256 verified) yet MISSING the
# plain articles for most major entities (Thailand, Hungary, Italy,
# Microsoft, Pablo Picasso, General relativity, ...) while carrying
# their long-tail variants — consistent with the release pipeline
# dropping its largest/most template-heavy pages. A fact bank needs
# exactly those pages, so that snapshot cannot verify it.
#
# This fetches every page the bank names, ONCE, via the MediaWiki
# action API (plaintext extracts), and freezes them with per-page
# REVISION IDS + retrieval timestamp — page-level pinning strictly
# stronger than a dump date. The artifact lives in the run root and is
# registered; the authoring module verifies against it offline.
#
# Usage: python -m jspace_phase3.experiments.fetch_wiki_reference
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request

from ..paths3 import run_root
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           write_result3)

EVIDENCE_ID = "p3-wiki-reference-v1"
TIER = "phase3-development"
API = "https://en.wikipedia.org/w/api.php"
UA = "jspace-phase3-bank-verifier/0.1 (research reproduction; contact: repo)"


def fetch_batch(titles: list[str]) -> dict:
    q = urllib.parse.urlencode({
        "action": "query", "format": "json", "formatversion": "2",
        "prop": "extracts|revisions", "explaintext": "1",
        "exsectionformat": "plain", "rvprop": "ids|timestamp",
        "redirects": "1", "titles": "|".join(titles), "maxlag": "5"})
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def main():
    require_clean_tree("--allow-dirty" in sys.argv)
    from .author_bank_f import build_bundles
    titles = sorted({b.provenance["pages"][k] for b in build_bundles()
                     for k in ("source_page", "bridge_page")})
    out_rows, redirect_map, missing = [], {}, []
    # extracts allows only ONE title per request when explaintext is on
    # for full content; batch politely
    for i, t in enumerate(titles):
        for attempt in range(3):
            try:
                data = fetch_batch([t])
                break
            except Exception as e:                    # noqa: BLE001
                if attempt == 2:
                    raise
                time.sleep(3 * (attempt + 1))
        for rd in data.get("query", {}).get("redirects", []):
            redirect_map[rd["from"]] = rd["to"]
        for page in data["query"]["pages"]:
            if page.get("missing"):
                missing.append(t)
                continue
            rev = page.get("revisions", [{}])[0]
            out_rows.append({
                "requested_title": t, "title": page["title"],
                "pageid": page.get("pageid"),
                "revid": rev.get("revid"),
                "rev_timestamp": rev.get("timestamp"),
                "text": page.get("extract", ""),
                "text_sha256": hashlib.sha256(
                    page.get("extract", "").encode()).hexdigest()})
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(titles)}", flush=True)
        time.sleep(0.15)

    out_dir = run_root() / "bank_reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "wiki_reference_v1.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                           for r in out_rows))
    payload = {
        "n_requested": len(titles), "n_fetched": len(out_rows),
        "missing_pages": missing, "redirects": redirect_map,
        "retrieval_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "api": API,
        "note": ("per-page revids pin content exactly; the Drive "
                 "wikimedia/wikipedia 20231101.en snapshot was abandoned "
                 "for verification because its release drops the largest "
                 "articles (majors like 'Thailand' absent; shard sha256s "
                 "hub-authentic)"),
        "reference_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    cmd = "python -m jspace_phase3.experiments.fetch_wiki_reference"
    meta = out_dir / "wiki_reference_v1.meta.json"
    write_result3(payload, meta, Provenance3(
        evidence_id=EVIDENCE_ID, tier=TIER, command=cmd, seed=0))
    register(EVIDENCE_ID, tier=TIER, command=cmd,
             what=(f"pinned wikipedia reference for Bank F verification: "
                   f"{len(out_rows)}/{len(titles)} pages fetched with "
                   f"revids at {payload['retrieval_utc']}; "
                   f"{len(missing)} missing"),
             outputs=[out, meta])
    print(json.dumps({k: v for k, v in payload.items() if k != "redirects"},
                     indent=1)[:1200])


if __name__ == "__main__":
    main()
