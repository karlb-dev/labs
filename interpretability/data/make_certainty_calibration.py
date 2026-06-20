"""Generate the Lab 14 certainty/hedging calibration set.

The CSV is frozen course data. Every row is rendered as a fixed A/B/C/D
choice task so Lab 14 can compare internal residual directions against the
model's next-token answer distribution and verbal self-reports.

Version 2 removes the original fixed-D shortcut: each family contains paired
answerable and known-unanswerable rows, answer keys are uniform across A-D for
both labels, and the unknown/insufficient-information option is also uniform
across A-D for both labels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib
from collections import Counter, defaultdict
from typing import Any

HERE = pathlib.Path(__file__).parent
OUT_NAME = "certainty_calibration_items.csv"
CARD_NAME = "certainty_calibration_items_card.md"
MANIFEST_NAME = "MANIFEST.json"

LETTERS = ("A", "B", "C", "D")
UNKNOWN_OPTIONS = (
    "Cannot be determined from the information given",
    "Not enough information is provided",
    "Insufficient information to answer",
    "The passage does not say",
)

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


ANSWERABLE: dict[str, list[tuple[str, str, str, list[str]]]] = {
    "mcq": [
        ("numbers_00", "A train leaves at 3:10 PM and arrives 45 minutes later. What time does it arrive?", "3:55 PM", ["3:45 PM", "4:05 PM", "4:15 PM"]),
        ("numbers_01", "Round 1,476 to the nearest hundred.", "1,500", ["1,400", "1,470", "1,600"]),
        ("numbers_02", "What is 18 divided by 3?", "6", ["3", "9", "15"]),
        ("numbers_03", "A box has 7 blue pens and 5 red pens. How many pens are in the box?", "12", ["2", "10", "14"]),
        ("numbers_04", "Which fraction is equal to one half?", "2/4", ["1/3", "3/5", "4/6"]),
        ("numbers_05", "What is the mode of this list: 2, 4, 4, 5, 9?", "4", ["2", "5", "9"]),
        ("numbers_06", "A rectangle is 8 cm long and 3 cm wide. What is its area?", "24 square cm", ["11 square cm", "18 square cm", "32 square cm"]),
        ("numbers_07", "What number comes next in the pattern 5, 10, 15, 20?", "25", ["22", "24", "30"]),
    ],
    "factual_qa": [
        ("world_00", "In standard geography, what city is the capital of France?", "Paris", ["Rome", "Berlin", "Madrid"]),
        ("world_01", "In astronomy, which planet is known as the Red Planet?", "Mars", ["Venus", "Jupiter", "Mercury"]),
        ("world_02", "In basic biology, what gas do plants take in for photosynthesis?", "carbon dioxide", ["oxygen", "nitrogen", "helium"]),
        ("world_03", "In English literature, who wrote Pride and Prejudice?", "Jane Austen", ["Mary Shelley", "George Eliot", "Charlotte Bronte"]),
        ("world_04", "On a world map, which ocean lies between Africa and Australia?", "Indian Ocean", ["Atlantic Ocean", "Arctic Ocean", "Southern Ocean"]),
        ("world_05", "Using the metric system, how many centimeters are in one meter?", "100", ["10", "50", "1,000"]),
        ("world_06", "On a standard calendar, what month comes after April?", "May", ["March", "June", "August"]),
        ("world_07", "In a music classroom, which instrument usually has keys, pedals, and strings?", "piano", ["flute", "drum", "trumpet"]),
    ],
    "passage_qa": [
        ("passage_00", "Passage: Nina packed a red scarf and a gray hat before leaving. What color was the scarf?", "red", ["gray", "blue", "green"]),
        ("passage_01", "Passage: The museum opens at 9 AM and closes at 5 PM. When does the museum open?", "9 AM", ["8 AM", "10 AM", "5 PM"]),
        ("passage_02", "Passage: Omar placed the receipt inside the blue folder. Where did Omar place the receipt?", "blue folder", ["green folder", "desk drawer", "coat pocket"]),
        ("passage_03", "Passage: The recipe uses rice, beans, and chopped tomatoes. Which ingredient is included?", "beans", ["potatoes", "apples", "noodles"]),
        ("passage_04", "Passage: The meeting moved from Tuesday to Thursday. What day is the meeting now?", "Thursday", ["Monday", "Tuesday", "Friday"]),
        ("passage_05", "Passage: Leah's dog is named Pixel. What is the dog's name?", "Pixel", ["Penny", "Parker", "Pepper"]),
        ("passage_06", "Passage: The hikers followed the north trail to reach the lake. Which trail did they follow?", "north trail", ["south trail", "river trail", "ridge trail"]),
        ("passage_07", "Passage: The package weighed four kilograms after it was sealed. How much did the package weigh?", "four kilograms", ["two kilograms", "six kilograms", "eight kilograms"]),
    ],
    "procedural_logic": [
        ("logic_00", "Rule: If a card is green, place it in tray 2. The card is green. Which tray should it go in?", "tray 2", ["tray 1", "tray 3", "tray 4"]),
        ("logic_01", "Rule: Start at 10 and subtract 3. What is the result?", "7", ["3", "10", "13"]),
        ("logic_02", "Rule: A badge expires two days after Monday. On what day does it expire?", "Wednesday", ["Tuesday", "Thursday", "Friday"]),
        ("logic_03", "Rule: Sort apples before oranges. Which fruit is sorted first?", "apples", ["oranges", "bananas", "pears"]),
        ("logic_04", "Rule: If the switch is up, the lamp is on. The switch is up. What is the lamp state?", "on", ["off", "dim", "flashing"]),
        ("logic_05", "Rule: Double the input 11. What output is produced?", "22", ["11", "13", "121"]),
        ("logic_06", "Rule: The smaller number wins. Between 14 and 19, which number wins?", "14", ["19", "33", "5"]),
        ("logic_07", "Rule: Move the marker one square east from C3. Where does it land?", "D3", ["C4", "B3", "D4"]),
    ],
    "freeform_answerability": [
        ("common_00", "In ordinary descriptions, what color is ripe grass usually called?", "green", ["red", "purple", "white"]),
        ("common_01", "In basic biology, what does photosynthesis help plants make?", "food", ["glass", "stone", "metal"]),
        ("common_02", "In common animal categories, what animal is a kitten?", "cat", ["horse", "duck", "lizard"]),
        ("common_03", "In everyday use, what do people usually use an umbrella for?", "staying dry in rain", ["cutting paper", "measuring weight", "boiling water"]),
        ("common_04", "In elementary geometry, what shape has three sides?", "triangle", ["circle", "square", "hexagon"]),
        ("common_05", "In common knowledge about insects, what do bees make?", "honey", ["salt", "plastic", "wool"]),
        ("common_06", "Among the five senses, which sense is used to hear music?", "hearing", ["taste", "touch", "sight"]),
        ("common_07", "In ordinary language, what is frozen water called?", "ice", ["steam", "sand", "smoke"]),
    ],
}


UNANSWERABLE: dict[str, list[tuple[str, str, list[str]]]] = {
    "mcq": [
        ("numbers_00", "Mara picked a private number. Which number was it?", ["3", "8", "14"]),
        ("numbers_01", "A favorite color response was lost. What color was selected?", ["blue", "green", "red"]),
        ("numbers_02", "An unobserved die roll occurred. What number came up?", ["1", "4", "6"]),
        ("numbers_03", "A private password is not shown. Which password was chosen?", ["river42", "orchid19", "paper77"]),
        ("numbers_04", "A sealed envelope has a city name inside. Which city is it?", ["Paris", "Tokyo", "Cairo"]),
        ("numbers_05", "A lottery drawing will happen tomorrow. Which ticket will win?", ["ticket 104", "ticket 219", "ticket 387"]),
        ("numbers_06", "A face-down card was never revealed. Which card is it?", ["ace of hearts", "ten of clubs", "queen of spades"]),
        ("numbers_07", "A deleted random output was not recorded. What was it?", ["17", "42", "91"]),
    ],
    "factual_qa": [
        ("world_00", "Dana made a private plan. What time did Dana choose?", ["4:15 PM", "5:30 PM", "6:05 PM"]),
        ("world_01", "Lee used an unseen mug. Which mug was it?", ["blue mug", "white mug", "striped mug"]),
        ("world_02", "An unwritten novel has no final sentence. What is it?", ["The door closed softly", "They watched the dawn", "Nothing more was said"]),
        ("world_03", "Mira wrote a private word. What word was it?", ["river", "silver", "garden"]),
        ("world_04", "Next month's room temperature is unknown. What will it be?", ["68 degrees", "71 degrees", "75 degrees"]),
        ("world_05", "An unnamed toy got a hidden name. What name was given?", ["Max", "Coco", "Luna"]),
        ("world_06", "Ari has not chosen a cafe yet. Which cafe will Ari choose?", ["North Cafe", "River Cafe", "Market Cafe"]),
        ("world_07", "Nia is silently thinking of a song. Which song is it?", ["a folk song", "a jazz song", "a pop song"]),
    ],
    "passage_qa": [
        ("passage_00", "Passage: Nina packed a scarf and a hat before leaving. What color was the scarf?", ["red", "gray", "blue"]),
        ("passage_01", "Passage: The museum changed its hours last week. What time does it open today?", ["8 AM", "9 AM", "10 AM"]),
        ("passage_02", "Passage: Omar filed the receipt somewhere before lunch. Which folder contains the receipt?", ["blue folder", "green folder", "red folder"]),
        ("passage_03", "Passage: The recipe was revised by the chef. Which ingredient was removed?", ["rice", "beans", "tomatoes"]),
        ("passage_04", "Passage: The meeting was rescheduled, but the new date was not announced. What day is it now?", ["Tuesday", "Thursday", "Friday"]),
        ("passage_05", "Passage: Leah adopted a dog yesterday. What is the dog's name?", ["Pixel", "Penny", "Pepper"]),
        ("passage_06", "Passage: The hikers chose one of several trails but the choice is omitted. Which trail did they follow?", ["north trail", "south trail", "ridge trail"]),
        ("passage_07", "Passage: The package was sealed before anyone weighed it. How much did it weigh?", ["two kilograms", "four kilograms", "six kilograms"]),
    ],
    "procedural_logic": [
        ("logic_00", "Rule: Green cards go in tray 2. The card color is hidden. Which tray is correct?", ["tray 1", "tray 2", "tray 3"]),
        ("logic_01", "Rule: Start with a secret number and subtract 3. What is the result?", ["4", "7", "10"]),
        ("logic_02", "Rule: A badge expires two days after the missing start day. When does it expire?", ["Tuesday", "Wednesday", "Thursday"]),
        ("logic_03", "Rule: Sort the selected fruit first. The fruit is not named. Which fruit is first?", ["apples", "oranges", "bananas"]),
        ("logic_04", "Rule: If the switch is up, the lamp is on. The switch is hidden. What is the lamp state?", ["on", "off", "dim"]),
        ("logic_05", "Rule: Double the input. The input value was erased. What output is produced?", ["12", "18", "22"]),
        ("logic_06", "Rule: The smaller number wins. One number is hidden. Which number wins?", ["14", "19", "33"]),
        ("logic_07", "Rule: Move one square east from a hidden start square. Where does it land?", ["D3", "C4", "B3"]),
    ],
    "freeform_answerability": [
        ("common_00", "What color is the shirt in a photograph that has not been shown?", ["black", "blue", "white"]),
        ("common_01", "What did an unnamed person write in a locked diary last night?", ["a poem", "a shopping list", "a phone number"]),
        ("common_02", "The story says there were three doors. Which door did the traveler choose?", ["left door", "middle door", "right door"]),
        ("common_03", "Sam privately chose an order flavor. Which flavor was it?", ["vanilla", "chocolate", "strawberry"]),
        ("common_04", "A small brown pet has an unstated name. What is the pet's name?", ["Max", "Coco", "Luna"]),
        ("common_05", "A speaker has no script for tomorrow. What exact sentence will be said?", ["Good morning", "Please wait", "I agree"]),
        ("common_06", "A wrapped box has not been opened. Which gift is inside?", ["book", "mug", "scarf"]),
        ("common_07", "What number is written on a hidden slip of paper?", ["12", "27", "45"]),
    ],
}


def unknown_text(index: int) -> str:
    return UNKNOWN_OPTIONS[index % len(UNKNOWN_OPTIONS)]


def answer_letter(index: int, answerable: bool) -> str:
    offset = 0 if answerable else 2
    return LETTERS[(index + offset) % len(LETTERS)]


def unknown_letter(index: int, answerable: bool, key: str) -> str:
    if not answerable:
        return key
    # Rotate one slot after the answer key. Since answer keys cycle uniformly
    # through A-D, this is also uniform and never collides with the key.
    return LETTERS[(LETTERS.index(key) + 1) % len(LETTERS)]


def build_options(*, correct: str, distractors: list[str], key: str, unknown: str, unknown_key: str) -> dict[str, str]:
    options: dict[str, str | None] = {letter: None for letter in LETTERS}
    if key not in LETTERS or unknown_key not in LETTERS:
        raise RuntimeError("Bad answer/unknown key.")
    if key == unknown_key and correct != unknown:
        raise RuntimeError("Answerable item placed the unknown option on the answer key.")
    options[key] = correct
    options[unknown_key] = unknown
    remaining = [letter for letter in LETTERS if options[letter] is None]
    if len(distractors) < len(remaining):
        raise RuntimeError(f"Need at least {len(remaining)} distractors for {correct!r}")
    for letter, distractor in zip(remaining, distractors):
        options[letter] = distractor
    return {letter: str(options[letter]) for letter in LETTERS}


def statement(question: str, answerable: bool, expected: str) -> tuple[str, str]:
    _ = answerable
    claim = f"The best answer to '{question}' is {expected}."
    return f"{claim} I am certain.", f"{claim} I am unsure."


def make_row(
    *,
    family: str,
    index: int,
    topic: str,
    answerable: bool,
    question: str,
    correct: str,
    distractors: list[str],
) -> dict[str, str]:
    key = answer_letter(index, answerable)
    unknown = unknown_text(index)
    unk_key = unknown_letter(index, answerable, key)
    expected = correct if answerable else unknown
    options = build_options(
        correct=correct if answerable else unknown,
        distractors=distractors,
        key=key,
        unknown=unknown,
        unknown_key=unk_key,
    )
    confident, hedged = statement(question, answerable, expected)
    label = "answerable" if answerable else "known-unanswerable"
    return {
        "item_id": f"{family}_{'a' if answerable else 'u'}_{index:02d}",
        "family": family,
        "topic": topic,
        "answerable": "1" if answerable else "0",
        "question": question,
        "option_a": options["A"],
        "option_b": options["B"],
        "option_c": options["C"],
        "option_d": options["D"],
        "answer_key": key,
        "expected_answer": expected,
        "answer_format": "deconfounded_abcd_v2",
        "confident_statement": confident,
        "hedged_statement": hedged,
        "note": f"{label}; paired topic; unknown option letter={unk_key}",
    }


def interleave_rows(answerable_rows: list[dict[str, str]], unanswerable_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for a, u in zip(answerable_rows, unanswerable_rows):
        rows.append(a)
        rows.append(u)
    return rows


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for family in ANSWERABLE:
        ans_rows: list[dict[str, str]] = []
        unans_rows: list[dict[str, str]] = []
        if len(ANSWERABLE[family]) != len(UNANSWERABLE[family]):
            raise RuntimeError(f"{family} has mismatched answerable/unanswerable counts.")
        for i, ((topic_a, q_a, correct, distractors_a), (topic_u, q_u, distractors_u)) in enumerate(zip(ANSWERABLE[family], UNANSWERABLE[family])):
            if topic_a != topic_u:
                raise RuntimeError(f"{family} topic pair mismatch: {topic_a} vs {topic_u}")
            ans_rows.append(make_row(
                family=family,
                index=i,
                topic=topic_a,
                answerable=True,
                question=q_a,
                correct=correct,
                distractors=distractors_a,
            ))
            unans_rows.append(make_row(
                family=family,
                index=i,
                topic=topic_u,
                answerable=False,
                question=q_u,
                correct=unknown_text(i),
                distractors=distractors_u,
            ))
        rows.extend(interleave_rows(ans_rows, unans_rows))
    return rows


def infer_unknown_letter(row: dict[str, str]) -> str:
    marker_counts = {
        letter: sum(1 for marker in UNKNOWN_OPTIONS if marker.lower() in row[f"option_{letter.lower()}"].lower())
        for letter in LETTERS
    }
    letters = [letter for letter, count in marker_counts.items() if count > 0]
    if len(letters) != 1:
        raise RuntimeError(f"{row['item_id']} should have exactly one unknown-style option, found {letters}")
    return letters[0]


def validate(rows: list[dict[str, str]]) -> None:
    ids = [r["item_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate item_id in certainty calibration data.")
    for row in rows:
        for key in FIELDNAMES:
            if key not in row:
                raise RuntimeError(f"{row.get('item_id', '<unknown>')} missing {key}")
        if row["answer_key"] not in set(LETTERS):
            raise RuntimeError(f"{row['item_id']} has invalid answer_key {row['answer_key']!r}")
        if row["answerable"] not in {"0", "1"}:
            raise RuntimeError(f"{row['item_id']} has invalid answerable label")
        for key, value in row.items():
            if value != value.strip():
                raise RuntimeError(f"{row['item_id']} has whitespace-padded {key}")

    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)

    for family, family_rows in sorted(by_family.items()):
        labels = Counter(r["answerable"] for r in family_rows)
        if labels["0"] != labels["1"]:
            raise RuntimeError(f"{family} is not label-balanced: {dict(labels)}")
        for label in ("0", "1"):
            subset = [r for r in family_rows if r["answerable"] == label]
            key_counts = Counter(r["answer_key"] for r in subset)
            unknown_counts = Counter(infer_unknown_letter(r) for r in subset)
            if len(set(key_counts.values())) != 1 or set(key_counts) != set(LETTERS):
                raise RuntimeError(f"{family} label {label} answer keys not uniform: {dict(key_counts)}")
            if len(set(unknown_counts.values())) != 1 or set(unknown_counts) != set(LETTERS):
                raise RuntimeError(f"{family} label {label} unknown letters not uniform: {dict(unknown_counts)}")

        topic_labels: dict[str, Counter[str]] = defaultdict(Counter)
        for row in family_rows:
            topic_labels[row["topic"]][row["answerable"]] += 1
        bad_topics = {topic: dict(counts) for topic, counts in topic_labels.items() if counts["0"] != 1 or counts["1"] != 1}
        if bad_topics:
            raise RuntimeError(f"{family} has unpaired topics: {bad_topics}")


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "families": dict(sorted(Counter(r["family"] for r in rows).items())),
        "answerable_counts": dict(sorted(Counter(r["answerable"] for r in rows).items())),
        "answer_key_counts_by_label": {
            label: dict(sorted(Counter(r["answer_key"] for r in rows if r["answerable"] == label).items()))
            for label in ("0", "1")
        },
        "unknown_option_letter_counts_by_label": {
            label: dict(sorted(Counter(infer_unknown_letter(r) for r in rows if r["answerable"] == label).items()))
            for label in ("0", "1")
        },
        "answer_format_counts": dict(sorted(Counter(r["answer_format"] for r in rows).items())),
    }


def update_manifest(path: pathlib.Path, rows: list[dict[str, str]], digest: str) -> None:
    manifest_path = HERE / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest[path.name] = {
        "generator": pathlib.Path(__file__).name,
        "rows": len(rows),
        "sha256": digest,
        "families": dict(sorted(Counter(r["family"] for r in rows).items())),
        "answerable_counts": dict(sorted(Counter(r["answerable"] for r in rows).items())),
        "answer_key_counts_by_label": summarize(rows)["answer_key_counts_by_label"],
        "unknown_option_letter_counts_by_label": summarize(rows)["unknown_option_letter_counts_by_label"],
        "pairing": "one answerable and one known-unanswerable row per family/topic; topic groups should remain together across train/eval splits",
        "confound_controls": [
            "answer keys are uniform across A-D separately for answerable and unanswerable rows",
            "unknown/insufficient-information option letters are uniform across A-D separately for answerable and unanswerable rows",
            "every item contains exactly one unknown-style option, so its presence alone is not a label cue",
        ],
        "source_files": [],
        "verified_tokenizers": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_card(path: pathlib.Path, rows: list[dict[str, str]], digest: str) -> None:
    summary = summarize(rows)
    lines = [
        "# Lab 14 certainty calibration dataset",
        "",
        "Frozen deterministic A/B/C/D answerability data for Lab 14.",
        "",
        f"- Rows: {len(rows)}",
        f"- SHA256: `{digest}`",
        f"- Families: {summary['families']}",
        f"- Answerability labels: {summary['answerable_counts']}",
        f"- Answer-key counts by label: {summary['answer_key_counts_by_label']}",
        f"- Unknown-option letter counts by label: {summary['unknown_option_letter_counts_by_label']}",
        "",
        "Design notes:",
        "",
        "- Each family has eight paired topics; each topic has one answerable and one known-unanswerable row.",
        "- Answer keys are balanced across A-D within each label and family.",
        "- Every row has exactly one unknown-style option, and its letter is balanced across A-D within each label and family.",
        "- The intended split unit is `family + topic + answer_format`, so paired topic rows can be held out together.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_rows()
    validate(rows)
    out_path = HERE / OUT_NAME
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in FIELDNAMES})
    digest = sha256(out_path)
    update_manifest(out_path, rows, digest)
    write_card(HERE / CARD_NAME, rows, digest)
    print(f"wrote {out_path} ({len(rows)} rows, sha256={digest})")


if __name__ == "__main__":
    main()
