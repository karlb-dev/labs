#!/usr/bin/env python3
"""Vendor the frozen breadth shell. Network is used only by this one-time tool."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import tempfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
SEED = "modelprint-prompt-bank-v2-20260822"
DOLLY_REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
OASST_REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
DOLLY_URL = f"https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/{DOLLY_REVISION}/databricks-dolly-15k.jsonl"
OASST_URL = f"https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/{OASST_REVISION}/2023-04-12_oasst_ready.messages.jsonl.gz"
OASST_LICENSE_URL = f"https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/{OASST_REVISION}/LICENSE"
DOLLY_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/3.0/legalcode.txt"

MODEL_TERMS = re.compile(
    r"\b(?:qwen|gemma|olmo|muse(?:[- ]glimmer)?|openai|chatgpt|gpt[- ]?\d|claude|anthropic|google|allenai|alibaba|meta[- ]?ai|llama)\b",
    re.I,
)
IDENTITY_TERMS = re.compile(r"\b(?:who are you|what model|which model|who (?:made|created|trained) you|your identity)\b", re.I)
LIVE_OR_ATTACHMENT = re.compile(
    r"\b(?:today|right now|current(?:ly)?|latest|this week|attached|attachment|uploaded|image above|file below|browse the web|at this URL|weather forecast|stock price)\b|https?://",
    re.I,
)
SAFETY_TERMS = re.compile(
    r"\b(?:kill|murder|suicide|self[- ]harm|weapon|bomb|explosive|poison|steal|theft|break into|hack(?:ing)?|malware|ransomware|phishing|credit card fraud|sexual|porn|nude|hate speech|terroris[mt]|drug trafficking)\b",
    re.I,
)


def sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def stable_key(value: str) -> str:
    return sha256(f"{SEED}:{value}")


def download(url: str, destination: Path) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ModelPrint-v1"})
    with urllib.request.urlopen(request, timeout=180) as response:
        data = response.read()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return data


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def shingles(text: str, width: int = 5) -> set[str]:
    value = normalize(text).lower()
    return {value[i : i + width] for i in range(max(1, len(value) - width + 1))} or {value}


def jaccard(left: set[str], right: set[str]) -> float:
    union = len(left | right)
    return len(left & right) / union if union else 1.0


def english_enough(text: str) -> bool:
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return False
    latin = sum("a" <= character.lower() <= "z" for character in letters)
    return latin / len(letters) >= 0.85


def eligibility(text: str) -> str | None:
    words = text.split()
    if not 4 <= len(words) <= 300:
        return "word_count"
    if not english_enough(text):
        return "language"
    if MODEL_TERMS.search(text) or IDENTITY_TERMS.search(text):
        return "model_or_identity"
    if LIVE_OR_ATTACHMENT.search(text):
        return "live_or_attachment"
    if SAFETY_TERMS.search(text):
        return "safety"
    return None


def repository_core_texts() -> list[str]:
    specs = [
        ("interpretability/data/relation_probes_lab1.csv", "prompt"),
        ("interpretability/data/advanced_relation_geometry.csv", "prompt"),
        ("interpretability/data/certainty_calibration_items.csv", "question"),
        ("interpretability/data/steering_eval_prompts.csv", "prompt"),
        ("interpretability/data/sae_feature_corpus.csv", "text"),
        ("interpretability/data/persona_register_pairs.csv", "content_question"),
        ("interpretability/data/sycophancy_pressure_items.csv", "question"),
        ("interpretability/data/belief_revision_dialogues.csv", "question"),
    ]
    values: list[str] = []
    for relative, column in specs:
        with (REPO_ROOT / relative).open(encoding="utf-8-sig", newline="") as handle:
            values.extend(row[column] for row in csv.DictReader(handle) if row.get(column))
    g1 = REPO_ROOT / "interpretability/jspaces/sidelines/gemma/data/g1_prompts_v1.jsonl"
    values.extend(json.loads(line)["text"] for line in g1.read_text().splitlines() if line.strip())
    return values


def deduplicate(candidates: Iterable[dict[str, Any]], existing: list[str], counters: Counter[str]) -> list[dict[str, Any]]:
    exact = {normalize(text).lower() for text in existing}
    buckets: dict[str, list[set[str]]] = defaultdict(list)
    for text in existing:
        clean = normalize(text).lower()
        buckets[clean[:2]].append(shingles(clean))
    accepted: list[dict[str, Any]] = []
    for row in sorted(candidates, key=lambda item: stable_key(f"{item['source_id']}:{item['source_row_id']}")):
        clean = normalize(row["prompt"])
        lower = clean.lower()
        if lower in exact:
            counters["exact_duplicate"] += 1
            continue
        grams = shingles(clean)
        if any(jaccard(grams, other) >= 0.8 for other in buckets[lower[:2]]):
            counters["near_duplicate"] += 1
            continue
        exact.add(lower)
        buckets[lower[:2]].append(grams)
        row["prompt"] = clean
        accepted.append(row)
    return accepted


def stratified_sample(rows: list[dict[str, Any]], key: str, target: int) -> list[dict[str, Any]]:
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[str(row[key])].append(row)
    for values in strata.values():
        values.sort(key=lambda row: stable_key(str(row["source_row_id"])))
    chosen: list[dict[str, Any]] = []
    names = sorted(strata)
    while len(chosen) < target and any(strata.values()):
        for name in names:
            if strata[name] and len(chosen) < target:
                chosen.append(strata[name].pop(0))
    return sorted(chosen, key=lambda row: stable_key(str(row["source_row_id"])))


def load_dolly(path: Path, counters: Counter[str]) -> list[dict[str, Any]]:
    categories = {"open_qa", "general_qa", "brainstorming", "creative_writing", "classification"}
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        source = json.loads(line)
        if source.get("category") not in categories or source.get("context", "").strip():
            counters["category_or_context"] += 1
            continue
        prompt = source.get("instruction", "").strip()
        response = source.get("response", "").strip()
        reason = eligibility(prompt)
        if reason or not response:
            counters[reason or "missing_response"] += 1
            continue
        rows.append(
            {
                "source_id": "dolly",
                "source_row_id": f"dolly-{index:05d}",
                "family": f"dolly-{source['category']}",
                "domain": source["category"],
                "stratum": "external-natural",
                "prompt": prompt,
                "human_response": response,
                "metadata": {"category": source["category"], "raw_index": index},
            }
        )
    return rows


def load_oasst(path: Path, counters: Counter[str]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        messages = [json.loads(line) for line in handle if line.strip()]
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for message in messages:
        if message.get("parent_id"):
            children[message["parent_id"]].append(message)
    rows: list[dict[str, Any]] = []
    for root in messages:
        if root.get("role") != "prompter" or root.get("parent_id") or root.get("lang") != "en":
            continue
        prompt = root.get("text", "").strip()
        reason = eligibility(prompt)
        replies = [
            item for item in children.get(root["message_id"], [])
            if item.get("role") == "assistant" and item.get("lang") == "en" and item.get("text", "").strip()
        ]
        replies.sort(key=lambda item: (item.get("rank", 10_000), stable_key(item["message_id"])))
        if reason or not replies:
            counters[reason or "missing_response"] += 1
            continue
        word_count = len(prompt.split())
        rows.append(
            {
                "source_id": "oasst1",
                "source_row_id": root["message_id"],
                "family": "oasst1-root",
                "domain": "conversation",
                "stratum": "external-natural",
                "prompt": prompt,
                "human_response": replies[0]["text"].strip(),
                "length_decile": min(9, word_count // 15),
                "metadata": {"tree_id": root.get("message_tree_id"), "reply_id": replies[0]["message_id"], "prompt_words": word_count},
            }
        )
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    work = ROOT / "data/work/vendor"
    work.mkdir(parents=True, exist_ok=True)
    dolly_raw = download(DOLLY_URL, work / "dolly.jsonl")
    oasst_raw = download(OASST_URL, work / "oasst.jsonl.gz")
    dolly_license = download(DOLLY_LICENSE_URL, ROOT / "data/external/dolly/LICENSE")
    oasst_license = download(OASST_LICENSE_URL, ROOT / "data/external/oasst1/LICENSE")

    counters: dict[str, Counter[str]] = {"dolly": Counter(), "oasst1": Counter()}
    core = repository_core_texts()
    dolly = deduplicate(load_dolly(work / "dolly.jsonl", counters["dolly"]), core, counters["dolly"])
    dolly = stratified_sample(dolly, "family", 550)
    oasst = deduplicate(load_oasst(work / "oasst.jsonl.gz", counters["oasst1"]), core + [row["prompt"] for row in dolly], counters["oasst1"])
    oasst = stratified_sample(oasst, "length_decile", 300)

    write_jsonl(ROOT / "data/external/dolly/prompts.jsonl", [{k: v for k, v in row.items() if k != "human_response"} for row in dolly])
    write_jsonl(ROOT / "data/external/oasst1/prompts.jsonl", [{k: v for k, v in row.items() if k not in {"human_response", "length_decile"}} for row in oasst])
    controls = [
        {
            "control_id": f"human-{row['source_row_id']}",
            "prompt_group_source_id": row["source_id"],
            "prompt_group_source_row_id": row["source_row_id"],
            "prompt": row["prompt"],
            "human_response": row["human_response"],
            "provenance": "human-authored dataset response",
        }
        for row in dolly + oasst
    ]
    write_jsonl(ROOT / "data/ood/human-authored-controls.jsonl", controls)

    attribution = {
        "dolly": {
            "repository": "databricks/databricks-dolly-15k",
            "revision": DOLLY_REVISION,
            "license": "CC-BY-SA-3.0",
            "source": DOLLY_URL,
        },
        "oasst1": {
            "repository": "OpenAssistant/oasst1",
            "revision": OASST_REVISION,
            "license": "Apache-2.0",
            "source": OASST_URL,
        },
    }
    (ROOT / "data/external/ATTRIBUTION.json").write_text(json.dumps(attribution, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "seed": SEED,
        "runtime_downloads": False,
        "sources": {
            "dolly": {**attribution["dolly"], "raw_sha256": sha256(dolly_raw), "license_sha256": sha256(dolly_license), "selected": len(dolly), "drops": dict(counters["dolly"])},
            "oasst1": {**attribution["oasst1"], "raw_sha256": sha256(oasst_raw), "license_sha256": sha256(oasst_license), "selected": len(oasst), "drops": dict(counters["oasst1"])},
        },
        "core_text_count_for_cross_dedup": len(core),
        "human_controls": len(controls),
    }
    manifest["manifest_sha256"] = sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    (ROOT / "data/external/source-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
