"""Generate the Lab 14 certainty/hedging calibration set.

The CSV is frozen course data. Every row is converted into a fixed A/B/C/D
choice frame at runtime, which lets Lab 14 compute answer-distribution
entropy and margins without free-form grading. The item families deliberately
mix ordinary answerable questions with known-unanswerable or underdetermined
questions so "certainty" is not just a topic label.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib
from collections import Counter

HERE = pathlib.Path(__file__).parent
OUT_NAME = "certainty_calibration_items.csv"
MANIFEST_NAME = "MANIFEST.json"
MCQ_SOURCE = "mcq_items.csv"

FIELDNAMES = [
    "item_id",
    "family",
    "topic",
    "answerable",
    "question",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
    "answer_key",
    "expected_answer",
    "answer_format",
    "confident_statement",
    "hedged_statement",
    "note",
]

UNKNOWN = "Cannot be determined from the information given"


MCQ_UNANSWERABLE = [
    (
        "mcq_u_00",
        "hidden_number",
        "Mara chose one number from 1 to 20. Which number did she choose?",
        "3",
        "8",
        "14",
        UNKNOWN,
    ),
    (
        "mcq_u_01",
        "unknown_preference",
        "A survey respondent selected their favorite color but the response was not recorded. What was the color?",
        "blue",
        "green",
        "red",
        UNKNOWN,
    ),
    (
        "mcq_u_02",
        "missing_roll",
        "A fair die was rolled behind a screen and no one reported the result. What number came up?",
        "1",
        "4",
        "6",
        UNKNOWN,
    ),
    (
        "mcq_u_03",
        "private_password",
        "A user created a private password that is not shown. Which password did they choose?",
        "river42",
        "orchid19",
        "paper77",
        UNKNOWN,
    ),
    (
        "mcq_u_04",
        "sealed_envelope",
        "A sealed envelope contains one city name, but the envelope has not been opened. Which city is written inside?",
        "Paris",
        "Tokyo",
        "Cairo",
        UNKNOWN,
    ),
    (
        "mcq_u_05",
        "future_lottery",
        "A lottery drawing will happen tomorrow. Which ticket will win?",
        "ticket 104",
        "ticket 219",
        "ticket 387",
        UNKNOWN,
    ),
]


FACTUAL_ITEMS = [
    (
        "fact_a_00",
        "capital",
        1,
        "What is the capital of France?",
        "Paris",
        "Rome",
        "Berlin",
        UNKNOWN,
        "A",
        "Paris",
    ),
    (
        "fact_a_01",
        "planet",
        1,
        "Which planet is known as the Red Planet?",
        "Venus",
        "Mars",
        "Jupiter",
        UNKNOWN,
        "B",
        "Mars",
    ),
    (
        "fact_a_02",
        "chemistry",
        1,
        "What gas do plants primarily take in for photosynthesis?",
        "oxygen",
        "nitrogen",
        "carbon dioxide",
        UNKNOWN,
        "C",
        "carbon dioxide",
    ),
    (
        "fact_a_03",
        "geography",
        1,
        "Which ocean lies between Africa and Australia?",
        "Indian Ocean",
        "Arctic Ocean",
        "Atlantic Ocean",
        UNKNOWN,
        "A",
        "Indian Ocean",
    ),
    (
        "fact_a_04",
        "math",
        1,
        "What is twelve times eight?",
        "88",
        "96",
        "108",
        UNKNOWN,
        "B",
        "96",
    ),
    (
        "fact_a_05",
        "literature",
        1,
        "Who wrote Pride and Prejudice?",
        "Charlotte Bronte",
        "Mary Shelley",
        "Jane Austen",
        UNKNOWN,
        "C",
        "Jane Austen",
    ),
    (
        "fact_u_00",
        "private_schedule",
        0,
        "What time did Dana privately decide to leave the office today?",
        "4:15 PM",
        "5:30 PM",
        "6:05 PM",
        UNKNOWN,
        "D",
        "",
    ),
    (
        "fact_u_01",
        "unknown_cup",
        0,
        "Which mug did Lee use this morning if the scene was not observed?",
        "blue mug",
        "white mug",
        "striped mug",
        UNKNOWN,
        "D",
        "",
    ),
    (
        "fact_u_02",
        "unpublished_book",
        0,
        "What is the final sentence of a novel that has not been written?",
        "The door closed softly.",
        "They watched the dawn.",
        "Nothing more was said.",
        UNKNOWN,
        "D",
        "",
    ),
    (
        "fact_u_03",
        "hidden_card",
        0,
        "A card was drawn from a shuffled deck and hidden. Which card was it?",
        "ace of hearts",
        "ten of clubs",
        "queen of spades",
        UNKNOWN,
        "D",
        "",
    ),
    (
        "fact_u_04",
        "future_weather",
        0,
        "What will the exact temperature be at noon in this room next month?",
        "68 degrees",
        "71 degrees",
        "75 degrees",
        UNKNOWN,
        "D",
        "",
    ),
    (
        "fact_u_05",
        "private_note",
        0,
        "What word did Mira write in her private notebook on the blank page?",
        "river",
        "silver",
        "garden",
        UNKNOWN,
        "D",
        "",
    ),
]


FREEFORM_ITEMS = [
    (
        "free_a_00",
        "definition",
        1,
        "What does photosynthesis allow plants to make using sunlight?",
        "food",
        "glass",
        "stone",
        UNKNOWN,
        "A",
        "food",
    ),
    (
        "free_a_01",
        "calendar",
        1,
        "What month comes after April?",
        "March",
        "May",
        "June",
        UNKNOWN,
        "B",
        "May",
    ),
    (
        "free_a_02",
        "measurement",
        1,
        "How many centimeters are in one meter?",
        "10",
        "50",
        "100",
        UNKNOWN,
        "C",
        "100",
    ),
    (
        "free_a_03",
        "animal",
        1,
        "What animal is a kitten?",
        "cat",
        "horse",
        "duck",
        UNKNOWN,
        "A",
        "cat",
    ),
    (
        "free_a_04",
        "sequence",
        1,
        "What number comes next in the sequence 2, 4, 6, 8?",
        "9",
        "10",
        "12",
        UNKNOWN,
        "B",
        "10",
    ),
    (
        "free_a_05",
        "color",
        1,
        "What color is ripe grass usually described as?",
        "yellow",
        "blue",
        "green",
        UNKNOWN,
        "C",
        "green",
    ),
    (
        "free_u_00",
        "subjective_choice",
        0,
        "What song is Nia silently thinking about right now?",
        "a folk song",
        "a jazz song",
        "a pop song",
        UNKNOWN,
        "D",
        "",
    ),
    (
        "free_u_01",
        "missing_context",
        0,
        "Which door did the traveler choose if the story only says there were three doors?",
        "left door",
        "middle door",
        "right door",
        UNKNOWN,
        "D",
        "",
    ),
    (
        "free_u_02",
        "random_draw",
        0,
        "What was the first number produced by an unspecified random number generator?",
        "17",
        "42",
        "91",
        UNKNOWN,
        "D",
        "",
    ),
    (
        "free_u_03",
        "unseen_photo",
        0,
        "What color shirt is the person wearing in a photo that has not been shown?",
        "black",
        "blue",
        "white",
        UNKNOWN,
        "D",
        "",
    ),
    (
        "free_u_04",
        "future_private_action",
        0,
        "Which cafe will Ari choose next week if Ari has not decided yet?",
        "North Cafe",
        "River Cafe",
        "Market Cafe",
        UNKNOWN,
        "D",
        "",
    ),
    (
        "free_u_05",
        "unknown_name",
        0,
        "What is the name of a pet that has only been described as small and brown?",
        "Max",
        "Coco",
        "Luna",
        UNKNOWN,
        "D",
        "",
    ),
]


def statement(question: str, answerable: int, expected: str) -> tuple[str, str]:
    if answerable:
        confident = f"I am certain that the answer to '{question}' is {expected}."
        hedged = f"It might be that the answer to '{question}' is {expected}."
    else:
        confident = f"I am certain that '{question}' cannot be determined from the information given."
        hedged = f"It might be impossible to determine '{question}' from the information given."
    return confident, hedged


def row_from_tuple(item: tuple[str, str, int, str, str, str, str, str, str, str], family: str) -> dict[str, str]:
    item_id, topic, answerable, question, a, b, c, d, answer_key, expected = item
    confident, hedged = statement(question, int(answerable), expected or UNKNOWN)
    return {
        "item_id": item_id,
        "family": family,
        "topic": topic,
        "answerable": str(int(answerable)),
        "question": question,
        "option_a": a,
        "option_b": b,
        "option_c": c,
        "option_d": d,
        "answer_key": answer_key,
        "expected_answer": expected,
        "answer_format": "A/B/C/D fixed choice",
        "confident_statement": confident,
        "hedged_statement": hedged,
        "note": "answerable" if answerable else "known-unanswerable",
    }


def mcq_rows() -> list[dict[str, str]]:
    source = HERE / MCQ_SOURCE
    rows: list[dict[str, str]] = []
    with source.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 6:
                break
            answer = row[f"option_{row['answer_key'].lower()}"]
            confident, hedged = statement(row["question"], 1, answer)
            rows.append({
                "item_id": f"mcq_a_{i:02d}",
                "family": "mcq",
                "topic": row["domain"],
                "answerable": "1",
                "question": row["question"],
                "option_a": row["option_a"],
                "option_b": row["option_b"],
                "option_c": row["option_c"],
                "option_d": row["option_d"],
                "answer_key": row["answer_key"],
                "expected_answer": answer,
                "answer_format": "A/B/C/D fixed choice",
                "confident_statement": confident,
                "hedged_statement": hedged,
                "note": f"from {MCQ_SOURCE}:{row['id']}",
            })
    for item_id, topic, q, a, b, c, d in MCQ_UNANSWERABLE:
        confident, hedged = statement(q, 0, UNKNOWN)
        rows.append({
            "item_id": item_id,
            "family": "mcq",
            "topic": topic,
            "answerable": "0",
            "question": q,
            "option_a": a,
            "option_b": b,
            "option_c": c,
            "option_d": d,
            "answer_key": "D",
            "expected_answer": "",
            "answer_format": "A/B/C/D fixed choice",
            "confident_statement": confident,
            "hedged_statement": hedged,
            "note": "known-unanswerable mcq-style control",
        })
    return interleave_by_label(rows)


def interleave_by_label(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    yes = [r for r in rows if r["answerable"] == "1"]
    no = [r for r in rows if r["answerable"] == "0"]
    out: list[dict[str, str]] = []
    for i in range(max(len(yes), len(no))):
        if i < len(yes):
            out.append(yes[i])
        if i < len(no):
            out.append(no[i])
    return out


def validate(rows: list[dict[str, str]]) -> None:
    ids = [r["item_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate item_id in certainty calibration data.")
    for row in rows:
        for key in FIELDNAMES:
            if key not in row:
                raise RuntimeError(f"{row.get('item_id', '<unknown>')} missing {key}")
        if row["answer_key"] not in {"A", "B", "C", "D"}:
            raise RuntimeError(f"{row['item_id']} has invalid answer_key {row['answer_key']!r}")
        if row["answerable"] not in {"0", "1"}:
            raise RuntimeError(f"{row['item_id']} has invalid answerable label")
        for key, value in row.items():
            if value != value.strip():
                raise RuntimeError(f"{row['item_id']} has whitespace-padded {key}")
    for family in sorted({r["family"] for r in rows}):
        labels = Counter(r["answerable"] for r in rows if r["family"] == family)
        if labels["0"] != labels["1"]:
            raise RuntimeError(f"{family} is not label-balanced: {dict(labels)}")


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def update_manifest(path: pathlib.Path, rows: list[dict[str, str]], digest: str) -> None:
    manifest_path = HERE / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest[path.name] = {
        "generator": pathlib.Path(__file__).name,
        "rows": len(rows),
        "sha256": digest,
        "families": dict(sorted(Counter(r["family"] for r in rows).items())),
        "answerable_counts": dict(sorted(Counter(r["answerable"] for r in rows).items())),
        "pairing": "fixed A/B/C/D answer choices plus confident/hedged statement pair per item",
        "source_files": [MCQ_SOURCE],
        "verified_tokenizers": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    rows: list[dict[str, str]] = []
    rows.extend(mcq_rows())
    rows.extend(interleave_by_label([row_from_tuple(item, "factual_qa") for item in FACTUAL_ITEMS]))
    rows.extend(interleave_by_label([row_from_tuple(item, "freeform_answerability") for item in FREEFORM_ITEMS]))
    validate(rows)
    out_path = HERE / OUT_NAME
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in FIELDNAMES})
    digest = sha256(out_path)
    update_manifest(out_path, rows, digest)
    print(f"wrote {out_path} ({len(rows)} rows, sha256={digest})")


if __name__ == "__main__":
    main()
