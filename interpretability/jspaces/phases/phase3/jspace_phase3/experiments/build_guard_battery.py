# Workstream C guard battery builder (nextsteps §7.2) — CPU, deterministic.
#
# Assembles guard_battery_v2.jsonl from PINNED LOCAL dataset snapshots
# (offline; every item records dataset, snapshot revision, file, row
# index, and char span, plus its own text sha256). Held-out choices:
# prose comes from the wikitext-103 TEST split — the campaign lenses
# (including the published Qwen n=1000 lens) were fitted on wikitext
# TRAIN draws. Domains per §7.2:
#
#   prose_pretrain   wikitext-103-raw-v1 test articles (~1200 chars)
#   long_context     wikitext-103-raw-v1 test articles (~4000 chars,
#                    disjoint from prose_pretrain)
#   factual_enc      wikipedia 20231101.en lead sections (short factual
#                    continuation register)
#   dialogue         ultrachat_200k test_sft, rendered as plain
#                    "User:/Assistant:" text (raw-text register on
#                    purpose — the battery measures the base LM channel,
#                    not chat-template behavior)
#   code             the-stack-smol-xs python
#   sql              the-stack-smol-xs sql
#   technical_docs   the-stack-smol-xs markdown (documentation prose)
#   grammar_pairs    data/grammar_minimal_pairs_v1.json (in-repo)
#
# Multilingual is SKIPPED and recorded: the only wikipedia snapshot on
# the box is 20231101.en, and §7.2's clause is conditional
# ("if tokenization permits"); revisit before the Phase 3 freeze.
#
# Usage: python -m jspace_phase3.experiments.build_guard_battery
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

from ..paths3 import drive_hub_cache, local_hub_cache
from ..provenance3 import Provenance3, register, require_clean_tree, write_result3

EVIDENCE_ID = "p3-guard-battery-v2"
SUPERSEDES = "p3-guard-battery-v1"  # v1 under-filled factual_enc (3/12,
# single-paragraph lead filter too strict) and technical_docs (5/12,
# stride 5 over a 100-row language file)
TIER = "phase3-development"
HUB = drive_hub_cache()
HUB_LOCAL = local_hub_cache()
REPO_DATA = Path(__file__).resolve().parents[2] / "data"
N_PER_DOMAIN = 12

WIKITEXT = (HUB / "datasets--Salesforce--wikitext/snapshots/"
            "b08601e04326c79dfdd32d625aee71d232d685c3")
WIKIPEDIA = (HUB / "datasets--wikimedia--wikipedia/snapshots/"
             "b04c8d1ceb2f5cd4588862100d08de323dccfbaa")
ULTRACHAT = (HUB / "datasets--HuggingFaceH4--ultrachat_200k/snapshots/"
             "8049631c405ae6576f93f445c6b8166f76f5505a")
STACK_XS = (HUB_LOCAL / "datasets--bigcode--the-stack-smol-xs/snapshots/"
            "1e3dd39b39787bddb20d7008e4d71c330d99f55b")


def cut(text: str, target: int) -> str:
    """Deterministic trim: cut at the last whitespace before `target`,
    then rstrip (DEFAULT_SPEC rejects trailing whitespace)."""
    if len(text) <= target:
        return text.rstrip()
    frag = text[:target]
    i = frag.rfind(" ")
    return (frag[:i] if i > target // 2 else frag).rstrip()


def wikitext_articles() -> list[tuple[int, int, str]]:
    """(row_lo, row_hi, text) per article from the test parquet."""
    f = WIKITEXT / "wikitext-103-raw-v1/test-00000-of-00001.parquet"
    rows = pd.read_parquet(f)["text"].tolist()
    arts, start = [], None
    for i, line in enumerate(rows):
        if re.match(r"^ = [^=]+ = \n?$", line):
            if start is not None:
                arts.append((start, i, "".join(rows[start:i])))
            start = i
    if start is not None:
        arts.append((start, len(rows), "".join(rows[start:])))
    return [(lo, hi, t) for lo, hi, t in arts if len(t) >= 1500]


def build_items() -> tuple[list[dict], dict]:
    items: list[dict] = []
    notes: dict = {}

    def add(domain, idx, text, source):
        text = text.rstrip()
        items.append({
            "item_id": f"guard:{domain}:{idx:02d}", "domain": domain,
            "text": text,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "source": source})

    # --- wikitext: prose (articles 0,3,6,...) and long-context (offset 1)
    arts = wikitext_articles()
    notes["wikitext_test_articles"] = len(arts)
    picks = [arts[i] for i in range(0, 3 * N_PER_DOMAIN, 3)]
    for n, (lo, hi, t) in enumerate(picks):
        add("prose_pretrain", n, cut(t, 1200),
            {"dataset": "Salesforce/wikitext", "config": "wikitext-103-raw-v1",
             "revision": WIKITEXT.name, "split": "test", "rows": [lo, hi]})
    picks = [arts[i] for i in range(1, 3 * N_PER_DOMAIN, 3)]
    for n, (lo, hi, t) in enumerate(picks):
        add("long_context", n, cut(t, 4000),
            {"dataset": "Salesforce/wikitext", "config": "wikitext-103-raw-v1",
             "revision": WIKITEXT.name, "split": "test", "rows": [lo, hi]})

    # --- wikipedia lead sections: every 997th article that qualifies
    f = WIKIPEDIA / "20231101.en/train-00000-of-00041.parquet"
    wp = pd.read_parquet(f, columns=["id", "title", "text"])
    got = idx = 0
    while got < N_PER_DOMAIN and idx < len(wp):
        row = wp.iloc[idx]
        paras = row.text.split("\n\n")
        lead = ""
        for para in paras:
            lead = f"{lead}\n\n{para}" if lead else para
            if len(lead) >= 900:
                break
        if len(lead) >= 900 and not lead.startswith(("List of", "Index of")):
            add("factual_enc", got, cut(lead, 1200),
                {"dataset": "wikimedia/wikipedia", "config": "20231101.en",
                 "revision": WIKIPEDIA.name,
                 "file": "train-00000-of-00041", "row": int(idx),
                 "page_id": str(row.id), "title": str(row.title)})
            got += 1
        idx += 997
    notes["wikipedia_rows_scanned_stride"] = 997

    # --- ultrachat dialogue, plain-text rendering
    f = ULTRACHAT / "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet"
    uc = pd.read_parquet(f, columns=["prompt", "messages"])
    got = idx = 0
    while got < N_PER_DOMAIN and idx < len(uc):
        msgs = uc.iloc[idx].messages
        parts = []
        for m in msgs[:4]:
            who = "User" if m["role"] == "user" else "Assistant"
            parts.append(f"{who}: {m['content'].strip()}")
        text = "\n\n".join(parts)
        if len(text) >= 900:
            add("dialogue", got, cut(text, 1200),
                {"dataset": "HuggingFaceH4/ultrachat_200k",
                 "revision": ULTRACHAT.name, "split": "test_sft",
                 "row": int(idx), "n_turns": min(len(msgs), 4)})
            got += 1
        idx += 89
    notes["ultrachat_stride"] = 89

    # --- the-stack-smol-xs: code / sql / technical markdown
    for domain, lang, stride in (("code", "python", 7), ("sql", "sql", 5),
                                 ("technical_docs", "markdown", 1)):
        rows = [json.loads(l) for l in
                (STACK_XS / f"data/{lang}/data.json").read_text().splitlines()]
        got, idx = 0, 0
        while got < N_PER_DOMAIN and idx < len(rows):
            content = rows[idx]["content"]
            ok = 600 <= len(content)
            if domain == "technical_docs":
                # documentation prose, not link farms or code dumps
                ok = ok and content.count("```") <= 2 \
                    and content.count("](") <= 12 and ". " in content
            if ok:
                add(domain, got, cut(content, 1200),
                    {"dataset": "bigcode/the-stack-smol-xs", "lang": lang,
                     "revision": STACK_XS.name, "row": int(idx)})
                got += 1
            idx += stride
        notes[f"{domain}_stride"] = stride

    # --- grammar pairs from the in-repo file
    gp = json.loads((REPO_DATA / "grammar_minimal_pairs_v1.json").read_text())
    for n, p in enumerate(gp["pairs"]):
        items.append({
            "item_id": f"guard:grammar:{n:02d}", "domain": "grammar_pairs",
            "pair_id": p["pair_id"], "phenomenon": p["phenomenon"],
            "good": p["good"].rstrip(), "bad": p["bad"].rstrip(),
            "text_sha256": hashlib.sha256(
                (p["good"] + "\x00" + p["bad"]).encode()).hexdigest(),
            "source": {"dataset": "in-repo",
                       "file": "data/grammar_minimal_pairs_v1.json"}})
    notes["multilingual"] = ("skipped: only 20231101.en wikipedia snapshot "
                             "available offline; §7.2 clause is conditional; "
                             "revisit before the Phase 3 freeze")
    return items, notes


def main():
    require_clean_tree("--allow-dirty" in sys.argv)
    items, notes = build_items()

    # dedup guard: no two items may share a 200-char text prefix
    seen: dict[str, str] = {}
    for it in items:
        key = (it.get("text") or it["good"])[:200]
        assert key not in seen, f"dup prefix: {it['item_id']} vs {seen[key]}"
        seen[key] = it["item_id"]
    by_dom = {}
    for it in items:
        by_dom[it["domain"]] = by_dom.get(it["domain"], 0) + 1

    out = REPO_DATA / "guard_battery_v2.jsonl"
    out.write_text("".join(json.dumps(it) + "\n" for it in items))
    sha = hashlib.sha256(out.read_bytes()).hexdigest()

    payload = {"n_items": len(items), "by_domain": by_dom, "notes": notes,
               "battery_sha256": sha, "path": str(out)}
    cmd = "python -m jspace_phase3.experiments.build_guard_battery"
    meta = REPO_DATA / "guard_battery_v2.meta.json"
    write_result3(payload, meta, Provenance3(
        evidence_id=EVIDENCE_ID, tier=TIER, command=cmd, seed=0))
    register(EVIDENCE_ID, tier=TIER, command=cmd, supersedes=SUPERSEDES,
             what=(f"Workstream C guard battery v1: {len(items)} items, "
                   f"{len(by_dom)} domains ({', '.join(sorted(by_dom))}), "
                   f"source-pinned to local snapshots; wikitext TEST split "
                   f"(lenses fit on train); multilingual deferred"),
             outputs=[out, meta])
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
