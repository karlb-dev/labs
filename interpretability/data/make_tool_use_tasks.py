#!/usr/bin/env python3
"""Generate the frozen benign toy-tool task set for Lab 34.

The generated JSONL is deliberately small, balanced, and boring. It is not a
benchmark of real agents. It is a controlled probe set for distinguishing
prompt-boundary tool-use signals from surface cues such as digits, tool names,
file names, routes, and units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from collections import Counter
from typing import Any

TOOLS = ("calculator", "dictionary", "calendar", "file_search", "route_planner", "unit_converter", "none")


def split_for(index: int, train_cut: int = 7) -> str:
    return "train" if index < train_cut else "eval"


def add(rows: list[dict[str, Any]], *, task_id: str, family: str, user_prompt: str,
        required_tool: str, tool_needed: bool, tool_args: dict[str, Any], answer: str,
        distractor_tool: str, split: str, notes: str, surface_cues: dict[str, Any]) -> None:
    if required_tool not in TOOLS:
        raise ValueError(f"unknown required_tool {required_tool!r}")
    if distractor_tool not in TOOLS or distractor_tool == required_tool:
        raise ValueError(f"bad distractor_tool {distractor_tool!r} for {required_tool!r}")
    rows.append({
        "task_id": task_id,
        "family": family,
        "user_prompt": user_prompt,
        "required_tool": required_tool,
        "tool_needed": bool(tool_needed),
        "tool_args": tool_args,
        "answer": answer,
        "distractor_tool": distractor_tool,
        "split": split,
        "notes": notes,
        "surface_cues": surface_cues,
    })


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    calculator = [
        ("17 * 23", "391", "What is 17 * 23?", "operator_digits"),
        ("144 / 12", "12", "Compute 144 / 12 exactly.", "compute_word"),
        ("19 + 46", "65", "Add 19 and 46.", "arithmetic_words"),
        ("81 - 27", "54", "Subtract 27 from 81.", "arithmetic_words"),
        ("7 * 8 + 6", "62", "For the calculator row, evaluate 7 * 8 + 6.", "tool_name_decoy_present"),
        ("90 / 5 - 3", "15", "I need the exact value of 90 / 5 - 3.", "operator_digits_eval"),
        ("12 * 12 - 5", "139", "Return the exact value for twelve squared minus five.", "word_number_train"),
        ("6 * 9", "54", "A math fact is requested: six times nine.", "word_arithmetic_eval"),
        ("40 + 2", "42", "The target quantity is forty plus two; give the exact number.", "word_plus_eval"),
        ("100 / 4", "25", "A divided-by calculation is needed for one hundred over four.", "word_division_eval"),
    ]
    for i, (expr, ans, prompt, cue) in enumerate(calculator):
        add(rows, task_id=f"calc_{i:03d}", family="calculator", user_prompt=prompt,
            required_tool="calculator", tool_needed=True, tool_args={"expression": expr}, answer=ans,
            distractor_tool="unit_converter", split=split_for(i), notes="restricted arithmetic parser",
            surface_cues={"digits": True, "operator": any(ch in expr for ch in "+-*/"), "tool_name": "calculator" in prompt.lower(), "cue_profile": cue})

    dictionary = [
        ("latency", "delay before a response begins", "Look up the glossary definition of latency.", "lookup_word"),
        ("mutex", "a lock that allows one holder at a time", "In the course glossary, what does mutex mean?", "glossary_word"),
        ("photosynthesis", "plants use light to make sugar from water and carbon dioxide", "Define photosynthesis using the toy dictionary.", "tool_name"),
        ("backoff", "waiting longer between retries after a failure", "What is the glossary definition of backoff?", "lookup_word"),
        ("vector", "an ordered list of numbers treated as one object", "Dictionary lookup: vector.", "tool_name_eval"),
        ("cache", "stored results reused to avoid repeated work", "Use the glossary to define cache.", "glossary_word_eval"),
        ("latency", "delay before a response begins", "Course term latency: provide the toy meaning.", "term_only_train"),
        ("mutex", "a lock that allows one holder at a time", "For mutex, return the closed glossary meaning.", "term_only_eval"),
        ("vector", "an ordered list of numbers treated as one object", "The term vector needs its in-course definition.", "term_only_eval"),
        ("cache", "stored results reused to avoid repeated work", "Cache has a toy meaning; state that meaning.", "term_only_eval"),
    ]
    for i, (term, ans, prompt, cue) in enumerate(dictionary):
        add(rows, task_id=f"dict_{i:03d}", family="dictionary", user_prompt=prompt,
            required_tool="dictionary", tool_needed=True, tool_args={"term": term}, answer=ans,
            distractor_tool="file_search", split=split_for(i), notes="closed glossary lookup",
            surface_cues={"lookup_word": True, "tool_name": "dictionary" in prompt.lower(), "term": term, "cue_profile": cue})

    calendar = [
        ("design review", "Tuesday 10:00", {"event": "design review"}, "When is the design review on the toy calendar?"),
        ("standup", "Monday 09:00", {"event": "standup"}, "Check the calendar time for standup."),
        ("bug triage", "Monday 10:00", {"event": "bug triage"}, "What time is bug triage scheduled?"),
        ("demo prep", "Friday afternoon", {"event": "demo prep"}, "Find demo prep on the toy calendar."),
        ("after_standup", "bug triage", {"after": "standup"}, "Which event comes after standup on Monday?"),
        ("friday_last", "demo prep", {"day": "Friday", "position": "last"}, "What is the Friday afternoon calendar item?"),
        ("standup_again", "Monday 09:00", {"event": "standup"}, "Standup time? Use the closed schedule."),
        ("design_slot", "Tuesday 10:00", {"event": "design review"}, "Design review slot?"),
        ("bug_slot", "Monday 10:00", {"event": "bug triage"}, "Bug triage happens when?"),
        ("demo_slot", "Friday afternoon", {"event": "demo prep"}, "Demo prep slot on Friday?"),
    ]
    for i, (name, ans, args, prompt) in enumerate(calendar):
        add(rows, task_id=f"cal_{i:03d}", family="calendar", user_prompt=prompt,
            required_tool="calendar", tool_needed=True, tool_args=args, answer=ans,
            distractor_tool="dictionary", split=split_for(i), notes="closed calendar simulator",
            surface_cues={"calendar_word": "calendar" in prompt.lower(), "event_name": name})

    file_search = [
        ("cache invalidation", "doc_cache.md", "Which synthetic document mentions cache invalidation?"),
        ("exporter csv", "doc_export.md", "Search the toy docs for exporter CSV."),
        ("retry budget", "doc_reliability.md", "Find the document with the retry budget."),
        ("stale user records", "doc_cache.md", "Which file talks about stale user records after writes?"),
        ("downstream reports", "doc_export.md", "In the toy file set, where are downstream reports mentioned?"),
        ("backoff", "doc_reliability.md", "Search the synthetic files for backoff."),
        ("retry budget", "doc_reliability.md", "Where is retry budget recorded?"),
        ("cache invalidation", "doc_cache.md", "Cache invalidation appears in which synthetic note?"),
        ("exporter csv", "doc_export.md", "Exporter CSV belongs to which document?"),
        ("stale user records", "doc_cache.md", "Stale user records after writes: name the document."),
    ]
    for i, (query, ans, prompt) in enumerate(file_search):
        add(rows, task_id=f"file_{i:03d}", family="file_search", user_prompt=prompt,
            required_tool="file_search", tool_needed=True, tool_args={"query": query}, answer=ans,
            distractor_tool="dictionary", split=split_for(i), notes="synthetic in-memory document search",
            surface_cues={"file_word": any(w in prompt.lower() for w in ("file", "document", "docs", "search")), "query": query})

    route = [
        ("A", "F", "A -> B -> D -> E -> F", "Find a route from A to F in the toy graph."),
        ("A", "D", "A -> B -> D", "What path reaches D from A?"),
        ("B", "F", "B -> D -> E -> F", "Plan the route from B to F."),
        ("C", "F", "C -> E -> F", "In the toy graph, route C to F."),
        ("D", "F", "D -> E -> F", "Which path goes from D to F?"),
        ("A", "E", "A -> B -> D -> E", "Graph route request: A to E."),
        ("A", "F", "A -> B -> D -> E -> F", "From A to F using allowed edges, list nodes."),
        ("B", "F", "B -> D -> E -> F", "B to F through the toy network."),
        ("C", "F", "C -> E -> F", "Start C, destination F; give node sequence."),
        ("A", "D", "A -> B -> D", "A reaches D by which nodes?"),
    ]
    for i, (start, end, ans, prompt) in enumerate(route):
        add(rows, task_id=f"route_{i:03d}", family="route_planner", user_prompt=prompt,
            required_tool="route_planner", tool_needed=True, tool_args={"start": start, "end": end}, answer=ans,
            distractor_tool="calendar", split=split_for(i), notes="toy directed graph shortest path",
            surface_cues={"route_word": any(w in prompt.lower() for w in ("route", "path", "graph")), "has_arrow": "->" in prompt})

    converter = [
        ("miles", "kilometers", 3, "4.83 kilometers", "Convert 3 miles to kilometers."),
        ("pounds", "kilograms", 10, "4.54 kilograms", "Convert 10 pounds to kilograms."),
        ("celsius", "fahrenheit", 20, "68 Fahrenheit", "Convert 20 celsius to fahrenheit."),
        ("hours", "minutes", 2, "120 minutes", "Convert 2 hours to minutes."),
        ("miles", "kilometers", 5, "8.05 kilometers", "Unit conversion: 5 miles into kilometers."),
        ("celsius", "fahrenheit", 0, "32 Fahrenheit", "What is 0 celsius in fahrenheit?"),
        ("hours", "minutes", 3, "180 minutes", "3 h expressed as min."),
        ("miles", "kilometers", 2, "3.22 kilometers", "2 mi in km."),
        ("pounds", "kilograms", 4, "1.81 kilograms", "4 lb in kg."),
        ("celsius", "fahrenheit", 10, "50 Fahrenheit", "10 C as F."),
    ]
    for i, (src, dst, value, ans, prompt) in enumerate(converter):
        add(rows, task_id=f"unit_{i:03d}", family="unit_converter", user_prompt=prompt,
            required_tool="unit_converter", tool_needed=True, tool_args={"from": src, "to": dst, "value": value}, answer=ans,
            distractor_tool="calculator", split=split_for(i), notes="closed unit conversion table",
            surface_cues={"digits": True, "convert_word": "convert" in prompt.lower() or "conversion" in prompt.lower(), "unit_pair": f"{src}->{dst}"})

    no_tool = [
        ("none_000", "no_tool", "Do not calculate; just answer the word YES.", "YES", "calculator", {"tool_name": "calculator", "digits": False}),
        ("none_001", "no_tool", "The word dictionary appears here, but reply with the word seen.", "seen", "dictionary", {"tool_name": "dictionary"}),
        ("none_002", "no_tool", "Calendar is a noun in this sentence. What tool should run? none.", "none", "calendar", {"tool_name": "calendar"}),
        ("none_003", "no_tool", "File name doc_cache.md is mentioned as text. Reply literal.", "literal", "file_search", {"file_word": True}),
        ("none_004", "no_tool", "Route A -> B is written as an example; answer example.", "example", "route_planner", {"route_word": True, "has_arrow": True}),
        ("none_005", "no_tool", "The phrase convert 3 miles appears in a quote. Answer quote.", "quote", "unit_converter", {"convert_word": True, "digits": True}),
        ("none_006", "no_tool", "This sentence contains 17 * 23, but the requested answer is banana.", "banana", "calculator", {"digits": True, "operator": True}),
        ("none_007", "no_tool", "If a glossary term is latency, answer only skip.", "skip", "dictionary", {"lookup_word": True}),
        ("none_008", "no_tool", "No calendar lookup is needed; answer ready.", "ready", "calendar", {"calendar_word": True}),
        ("none_009", "no_tool", "Mentioning search does not require a search here. Answer local.", "local", "file_search", {"file_word": True}),
        ("none_010", "no_tool", "A path from A to F is irrelevant; answer idle.", "idle", "route_planner", {"route_word": True}),
        ("none_011", "no_tool", "Units like pounds and kilograms are examples; answer plain.", "plain", "unit_converter", {"convert_word": True}),
        ("none_012", "no_tool", "The string 40 + 2 is an identifier; answer id.", "id", "calculator", {"digits": True, "operator": True}),
        ("none_013", "no_tool", "Dictionary lookup is a heading here; answer heading.", "heading", "dictionary", {"tool_name": "dictionary", "lookup_word": True}),
        ("none_014", "no_tool", "The calendar widget label is decorative; answer label.", "label", "calendar", {"calendar_word": True}),
        ("none_015", "no_tool", "A file_search option is text on the screen; answer option.", "option", "file_search", {"file_word": True}),
        ("none_016", "no_tool", "Route planner is a product name; answer product.", "product", "route_planner", {"route_word": True}),
        ("none_017", "no_tool", "Convert button label only; answer button.", "button", "unit_converter", {"convert_word": True}),
        ("none_018", "no_tool", "17 * 23 appears on a sticker; answer sticker.", "sticker", "calculator", {"digits": True, "operator": True}),
        ("none_019", "no_tool", "Glossary definition is printed as a title; answer title.", "title", "dictionary", {"lookup_word": True}),
        ("none_020", "no_tool", "Calendar search box is empty; answer empty.", "empty", "calendar", {"calendar_word": True}),
        ("none_021", "no_tool", "Synthetic document name doc_export.md is decorative; answer decorative.", "decorative", "file_search", {"file_word": True}),
        ("none_022", "no_tool", "A path A -> F is drawn as art; answer art.", "art", "route_planner", {"route_word": True, "has_arrow": True}),
        ("none_023", "no_tool", "Pounds and kilograms are font examples; answer font.", "font", "unit_converter", {"convert_word": True}),
    ]
    for i, (task_id, family, prompt, ans, distractor, cues) in enumerate(no_tool):
        add(rows, task_id=task_id, family=family, user_prompt=prompt,
            required_tool="none", tool_needed=False, tool_args={}, answer=ans,
            distractor_tool=distractor, split="train" if i < 16 else "eval",
            notes="surface-cue no-tool control", surface_cues={**cues, "cue_profile": "surface_cue_no_tool"})

    rows.sort(key=lambda row: (row["split"] != "train", row["required_tool"], row["task_id"]))
    return rows


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("tool_use_tasks.jsonl"))
    args = parser.parse_args()
    rows = build_rows()
    write_jsonl(args.out, rows)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"wrote {len(rows)} rows to {args.out}")
    print(f"sha256 {digest}")
    print("required_tools", dict(Counter(row["required_tool"] for row in rows)))
    print("splits", dict(Counter(row["split"] for row in rows)))


if __name__ == "__main__":
    main()
