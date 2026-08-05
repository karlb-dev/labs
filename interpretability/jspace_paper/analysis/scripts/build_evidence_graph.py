#!/usr/bin/env python3
"""P2: master evidence graph.

Parses all six campaign registries (read-only), reconstructs live vs
superseded/withdrawn state per evidence id, detects anomalies (duplicate
origins, dangling supersession edges, cycles), and maps import events to
native source events without copying tier.

Outputs:
  data/master_evidence_events.parquet   one row per registry event (T7)
  data/master_evidence_live.parquet     resolved terminal state per id
  reports/CAMPAIGN_EVIDENCE_MAP.md      human-readable map + anomalies
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path("/content/labs")
ANALYSIS = REPO / "interpretability/jspace_paper/analysis"

REGISTRIES = {
    "part2_registry": "interpretability/jspace_part2/reports/evidence_registry.jsonl",
    "part2_events": "interpretability/jspace_part2/reports/evidence_events.jsonl",
    "phase3": "interpretability/jspace_phase3/reports/evidence_events.jsonl",
    "phase4": "interpretability/jspace_phase4/reports/evidence_events.jsonl",
    "gemma": "interpretability/jspace_gemma/reports/evidence_events.jsonl",
    "olmo_lineage": "interpretability/jspace_olmo_lineage/reports/evidence_events.jsonl",
}

# The flat part2 registry mirrors the event-sourced file; status resolution
# uses events, the flat file contributes description text only.
STATUS_SOURCES = ["part2_events", "phase3", "phase4", "gemma", "olmo_lineage"]


def parse_events():
    rows = []
    for reg, rel in REGISTRIES.items():
        with open(REPO / rel) as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                ev = json.loads(line)
                rows.append({
                    "registry": reg,
                    "source_line_no": i,
                    "evidence_id": ev.get("evidence_id"),
                    "event_type": ev.get("event", "registered"),
                    "utc": ev.get("event_utc") or ev.get("registered_utc"),
                    "tier": ev.get("tier"),
                    "superseded_by": ev.get("superseded_by"),
                    "reason": ev.get("reason"),
                    "source_evidence_id": ev.get("source_evidence_id"),
                    "source_study": ev.get("source_study"),
                    "what": (ev.get("what") or "")[:2000],
                    "n_source_outputs": len(ev.get("source_outputs") or []) or None,
                    "line_sha256": hashlib.sha256(line.encode()).hexdigest(),
                })
    return pd.DataFrame(rows)


def resolve_live(events: pd.DataFrame):
    live_rows, anomalies = [], []
    for reg in STATUS_SOURCES:
        sub = events[events.registry == reg]
        state = {}
        for _, ev in sub.iterrows():
            eid = ev.evidence_id
            et = ev.event_type
            if et in ("evidence_created", "evidence_imported", "registered"):
                if eid in state:
                    anomalies.append(f"duplicate origin: {reg}:{eid} "
                                     f"(line {ev.source_line_no})")
                    # keep first creation metadata per event-sourcing rules
                    continue
                state[eid] = {
                    "registry": reg, "evidence_id": eid,
                    "created_utc": ev.utc, "tier": ev.tier,
                    "origin": ("import" if et == "evidence_imported"
                               else "created"),
                    "imports_from": ev.source_evidence_id,
                    "source_study": ev.source_study,
                    "what": ev.what, "superseded_by": None,
                    "withdrawn": False, "corrected": False,
                    "reproduced": False, "status_reason": None,
                }
            elif et == "evidence_superseded":
                if eid not in state:
                    anomalies.append(f"supersession of unknown id: {reg}:{eid}")
                    continue
                state[eid]["superseded_by"] = ev.superseded_by
                state[eid]["status_reason"] = ev.reason
            elif et == "evidence_withdrawn":
                if eid not in state:
                    anomalies.append(f"withdrawal of unknown id: {reg}:{eid}")
                    continue
                state[eid]["withdrawn"] = True
                state[eid]["status_reason"] = ev.reason
            elif et == "evidence_corrected":
                if eid in state:
                    state[eid]["corrected"] = True
            elif et == "evidence_reproduced":
                if eid in state:
                    state[eid]["reproduced"] = True
        # dangling supersession targets + cycles
        for eid, s in state.items():
            tgt = s["superseded_by"]
            if tgt and tgt not in state:
                anomalies.append(
                    f"dangling supersession target: {reg}:{eid} -> {tgt}")
        for eid, s in state.items():
            seen, cur = {eid}, s["superseded_by"]
            while cur and cur in state:
                if cur in seen:
                    anomalies.append(f"supersession cycle at {reg}:{eid}")
                    break
                seen.add(cur)
                cur = state[cur]["superseded_by"]
        for s in state.values():
            s["status"] = ("withdrawn" if s["withdrawn"] else
                           "superseded" if s["superseded_by"] else "live")
            live_rows.append(s)
    return pd.DataFrame(live_rows), anomalies


def main():
    (ANALYSIS / "data").mkdir(parents=True, exist_ok=True)
    events = parse_events()
    events.to_parquet(ANALYSIS / "data/master_evidence_events.parquet",
                      index=False)
    live, anomalies = resolve_live(events)
    live.to_parquet(ANALYSIS / "data/master_evidence_live.parquet",
                    index=False)

    # cross-registry import map (imports never copy tier)
    imports = live[live.origin == "import"][
        ["registry", "evidence_id", "imports_from", "source_study", "tier"]]

    lines = ["# CAMPAIGN_EVIDENCE_MAP.md — P2 output", "",
             "Source: `data/master_evidence_events.parquet` / "
             "`master_evidence_live.parquet` "
             "(builder `scripts/build_evidence_graph.py`; registries "
             "read-only, hashes in `ANALYSIS_FOUNDATION.json`).", ""]
    lines.append("## Registry totals (event-sourced)\n")
    lines.append("| Registry | Events | Ids | Live | Superseded | Withdrawn | Imports |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for reg in STATUS_SOURCES:
        ev_n = len(events[events.registry == reg])
        sub = live[live.registry == reg]
        lines.append(
            f"| {reg} | {ev_n} | {len(sub)} | "
            f"{(sub.status == 'live').sum()} | "
            f"{(sub.status == 'superseded').sum()} | "
            f"{(sub.status == 'withdrawn').sum()} | "
            f"{(sub.origin == 'import').sum()} |")
    lines.append("")
    lines.append("## Live evidence by tier\n")
    lines.append("| Registry | Tier | Live ids |")
    lines.append("|---|---|---:|")
    livev = live[live.status == "live"]
    for (reg, tier), n in livev.groupby(
            ["registry", livev.tier.fillna("(untiered)")]).size().items():
        lines.append(f"| {reg} | {tier} | {n} |")
    lines.append("")
    lines.append("## Import edges (native tier preserved at source)\n")
    lines.append("| Importing registry | Import event | Source event | Source study |")
    lines.append("|---|---|---|---|")
    for _, r in imports.iterrows():
        lines.append(f"| {r.registry} | `{r.evidence_id}` | "
                     f"`{r.imports_from or '(bundle)'}` | {r.source_study or ''} |")
    lines.append("")
    lines.append("## Anomaly register\n")
    if anomalies:
        for a in anomalies:
            lines.append(f"- {a}")
    else:
        lines.append("- none detected")
    lines.append("")
    dup_flat = events[events.registry == "part2_registry"].groupby(
        "evidence_id").size()
    dups = dup_flat[dup_flat > 1]
    lines.append("## Flat part2 registry duplicate rows (non-event file)\n")
    if len(dups):
        lines.append("The flat `evidence_registry.jsonl` mirror contains "
                     "append-duplicates; the event-sourced file is "
                     "authoritative for status. Duplicated ids:")
        for eid, n in dups.items():
            lines.append(f"- `{eid}` ×{n}")
    else:
        lines.append("- none")
    (ANALYSIS / "reports").mkdir(parents=True, exist_ok=True)
    (ANALYSIS / "reports/CAMPAIGN_EVIDENCE_MAP.md").write_text(
        "\n".join(lines) + "\n")

    print(f"events: {len(events)}; ids: {len(live)}; "
          f"live: {(live.status == 'live').sum()}; anomalies: {len(anomalies)}")
    for a in anomalies:
        print("  !", a)


if __name__ == "__main__":
    main()
