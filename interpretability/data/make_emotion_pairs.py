"""Generate the Lab 13 affect/emotion geometry set.

Run at authoring time; the CSV is vendored and the lab treats it as frozen
data. The inventory is deterministic and deliberately small enough to read.

The dataset is paired: each target item has an emotion-laden sentence and a
neutral paraphrase about the same cause/topic. It also carries generation
prompts for the write-side version of the same contrast. Confound rows are
included so Lab 13 can audit cheap explanations such as surprise words,
positive valence, and arousal without pretending those controls are target
emotion labels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib
from collections import Counter

HERE = pathlib.Path(__file__).parent
OUT_NAME = "affect_emotion_pairs.csv"
MANIFEST_NAME = "MANIFEST.json"

FIELDNAMES = [
    "item_id",
    "emotion",
    "cause",
    "arousal",
    "valence",
    "content_text",
    "neutral_text",
    "generation_prompt",
    "neutral_generation_prompt",
    "confound",
    "note",
]


TARGET_ROWS: list[dict[str, str]] = [
    {
        "emotion": "joy",
        "cause": "reunion",
        "arousal": "high",
        "valence": "positive",
        "content_text": "After years apart, Mia saw her brother at the station and felt a bright rush of joy.",
        "neutral_text": "After years apart, Mia met her brother at the station and checked the arrival board.",
        "generation_prompt": "Write one joyful sentence about a reunion at a train station.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about a reunion at a train station.",
        "note": "family contact; matched station topic",
    },
    {
        "emotion": "joy",
        "cause": "achievement",
        "arousal": "high",
        "valence": "positive",
        "content_text": "When the final score appeared, Lina laughed with delight because the team had won.",
        "neutral_text": "When the final score appeared, Lina recorded that the team had won.",
        "generation_prompt": "Write one joyful sentence about a team winning a final match.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about a team winning a final match.",
        "note": "achievement; same outcome",
    },
    {
        "emotion": "joy",
        "cause": "nature",
        "arousal": "medium",
        "valence": "positive",
        "content_text": "The first warm morning filled the garden with birdsong, and Evan felt quietly happy.",
        "neutral_text": "The first warm morning brought birdsong to the garden while Evan opened the gate.",
        "generation_prompt": "Write one joyful sentence about the first warm morning in a garden.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about the first warm morning in a garden.",
        "note": "positive nature without explicit reward",
    },
    {
        "emotion": "joy",
        "cause": "gift",
        "arousal": "medium",
        "valence": "positive",
        "content_text": "The small wrapped gift made Arun grin because it was exactly the book he wanted.",
        "neutral_text": "The small wrapped gift contained exactly the book Arun had requested.",
        "generation_prompt": "Write one joyful sentence about receiving a requested book as a gift.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about receiving a requested book as a gift.",
        "note": "matched object and requester",
    },
    {
        "emotion": "joy",
        "cause": "relief",
        "arousal": "medium",
        "valence": "positive",
        "content_text": "The lost passport turned up in the desk drawer, and a wave of relief brightened the room.",
        "neutral_text": "The lost passport turned up in the desk drawer before the trip began.",
        "generation_prompt": "Write one joyful sentence about finding a lost passport before a trip.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about finding a lost passport before a trip.",
        "note": "relief overlaps with lowered fear",
    },
    {
        "emotion": "joy",
        "cause": "community",
        "arousal": "medium",
        "valence": "positive",
        "content_text": "Neighbors cheered when the lights came back on, and the hallway felt warm with shared joy.",
        "neutral_text": "Neighbors noted when the lights came back on and returned to their apartments.",
        "generation_prompt": "Write one joyful sentence about neighbors after the lights come back on.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about neighbors after the lights come back on.",
        "note": "community event; same power-restoration fact",
    },
    {
        "emotion": "sadness",
        "cause": "bereavement",
        "arousal": "low",
        "valence": "negative",
        "content_text": "At the memorial, Noor held the old photograph and felt grief settle heavily in her chest.",
        "neutral_text": "At the memorial, Noor held the old photograph while the program continued.",
        "generation_prompt": "Write one sad sentence about holding an old photograph at a memorial.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about holding an old photograph at a memorial.",
        "note": "bereavement; mild wording",
    },
    {
        "emotion": "sadness",
        "cause": "weather",
        "arousal": "low",
        "valence": "negative",
        "content_text": "Rain streaked the empty windows, and the quiet afternoon felt lonely and sad.",
        "neutral_text": "Rain streaked the empty windows during the quiet afternoon.",
        "generation_prompt": "Write one sad sentence about rain on empty windows.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about rain on empty windows.",
        "note": "weather cause; low arousal",
    },
    {
        "emotion": "sadness",
        "cause": "fictional",
        "arousal": "medium",
        "valence": "negative",
        "content_text": "The novel's ending left Priya sorrowful because the friends never found each other again.",
        "neutral_text": "The novel's ending reported that the friends never found each other again.",
        "generation_prompt": "Write one sad sentence about a novel ending where friends do not reunite.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about a novel ending where friends do not reunite.",
        "note": "fictional loss rather than personal event",
    },
    {
        "emotion": "sadness",
        "cause": "loss",
        "arousal": "low",
        "valence": "negative",
        "content_text": "The empty collar by the door made Sam's throat tighten with a dull sadness.",
        "neutral_text": "The empty collar by the door showed where the pet supplies were kept.",
        "generation_prompt": "Write one sad sentence about seeing an empty pet collar by a door.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about seeing an empty pet collar by a door.",
        "note": "pet-loss cue; same object",
    },
    {
        "emotion": "sadness",
        "cause": "loneliness",
        "arousal": "low",
        "valence": "negative",
        "content_text": "At the end of the party, the silent room made Elena feel painfully alone.",
        "neutral_text": "At the end of the party, Elena turned off the lights in the silent room.",
        "generation_prompt": "Write one sad sentence about a silent room after a party.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about a silent room after a party.",
        "note": "social contrast",
    },
    {
        "emotion": "sadness",
        "cause": "illness",
        "arousal": "medium",
        "valence": "negative",
        "content_text": "The postponed visit left Mateo disappointed, and his voice sounded small on the phone.",
        "neutral_text": "The visit was postponed, and Mateo confirmed the new date by phone.",
        "generation_prompt": "Write one sad sentence about a postponed visit confirmed by phone.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about a postponed visit confirmed by phone.",
        "note": "disappointment rather than acute fear",
    },
    {
        "emotion": "anger",
        "cause": "injustice",
        "arousal": "high",
        "valence": "negative",
        "content_text": "When the rule punished the careful workers, Tessa felt anger flare at the unfairness.",
        "neutral_text": "When the rule affected the careful workers, Tessa reviewed the policy details.",
        "generation_prompt": "Write one angry sentence about an unfair rule at work.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about an unfair rule at work.",
        "note": "injustice; matched policy topic",
    },
    {
        "emotion": "anger",
        "cause": "delay",
        "arousal": "medium",
        "valence": "negative",
        "content_text": "The third unexplained delay made Omar clench his jaw in frustration.",
        "neutral_text": "The third delay changed Omar's expected arrival time.",
        "generation_prompt": "Write one angry sentence about a third unexplained delay.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about a third unexplained delay.",
        "note": "frustration; logistical cause",
    },
    {
        "emotion": "anger",
        "cause": "betrayal",
        "arousal": "high",
        "valence": "negative",
        "content_text": "Seeing the copied proposal, June felt furious that her partner had taken the credit.",
        "neutral_text": "Seeing the copied proposal, June documented that her partner had submitted it.",
        "generation_prompt": "Write one angry sentence about a copied proposal and lost credit.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about a copied proposal and lost credit.",
        "note": "credit conflict; no threat content",
    },
    {
        "emotion": "anger",
        "cause": "rude_service",
        "arousal": "medium",
        "valence": "negative",
        "content_text": "The clerk's sneer made Asha bristle, and her patience turned into irritation.",
        "neutral_text": "The clerk's expression changed while Asha waited at the counter.",
        "generation_prompt": "Write one angry sentence about rude service at a counter.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about service at a counter.",
        "note": "interpersonal cue; same location",
    },
    {
        "emotion": "anger",
        "cause": "broken_promise",
        "arousal": "medium",
        "valence": "negative",
        "content_text": "The broken promise left Ravi bitter because everyone else had planned around it.",
        "neutral_text": "The changed promise affected the schedule that everyone else had planned around.",
        "generation_prompt": "Write one angry sentence about a broken promise that disrupted a schedule.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about a changed promise that affected a schedule.",
        "note": "same planning facts",
    },
    {
        "emotion": "anger",
        "cause": "policy",
        "arousal": "high",
        "valence": "negative",
        "content_text": "The sudden fee felt insulting, and Dana's calm note turned sharp with anger.",
        "neutral_text": "The sudden fee appeared on the invoice, and Dana revised the note.",
        "generation_prompt": "Write one angry sentence about a sudden fee on an invoice.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about a sudden fee on an invoice.",
        "note": "bureaucratic cause",
    },
    {
        "emotion": "fear",
        "cause": "storm",
        "arousal": "high",
        "valence": "negative",
        "content_text": "Thunder shook the windows, and Kira felt a cold thread of fear in the dark hallway.",
        "neutral_text": "Thunder shook the windows while Kira walked through the dark hallway.",
        "generation_prompt": "Write one fearful sentence about thunder in a dark hallway.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about thunder in a dark hallway.",
        "note": "storm cause; matched hallway",
    },
    {
        "emotion": "fear",
        "cause": "diagnosis",
        "arousal": "medium",
        "valence": "negative",
        "content_text": "Waiting for the test result, Jonah felt anxious about what the doctor might say.",
        "neutral_text": "Waiting for the test result, Jonah checked when the doctor would call.",
        "generation_prompt": "Write one fearful sentence about waiting for a medical test result.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about waiting for a medical test result.",
        "note": "medical uncertainty; no advice",
    },
    {
        "emotion": "fear",
        "cause": "dark_house",
        "arousal": "high",
        "valence": "negative",
        "content_text": "A floorboard creaked upstairs, and Mara froze with panic in the dark house.",
        "neutral_text": "A floorboard creaked upstairs while Mara stood in the dark house.",
        "generation_prompt": "Write one fearful sentence about a creak upstairs in a dark house.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about a creak upstairs in a dark house.",
        "note": "fictional suspense cue",
    },
    {
        "emotion": "fear",
        "cause": "lost_child",
        "arousal": "high",
        "valence": "negative",
        "content_text": "For one awful minute, Imani could not see the child in the crowd and felt terror rise.",
        "neutral_text": "For one minute, Imani searched for the child in the crowd.",
        "generation_prompt": "Write one fearful sentence about briefly losing sight of a child in a crowd.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about searching for a child in a crowd.",
        "note": "short-lived fear; same crowd topic",
    },
    {
        "emotion": "fear",
        "cause": "emergency",
        "arousal": "high",
        "valence": "negative",
        "content_text": "The alarm blinked red, and Jules felt dread before the technician explained the signal.",
        "neutral_text": "The alarm blinked red before the technician explained the signal.",
        "generation_prompt": "Write one fearful sentence about a red alarm before an explanation.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about a red alarm before an explanation.",
        "note": "ambiguous alarm; resolved later",
    },
    {
        "emotion": "fear",
        "cause": "uncertainty",
        "arousal": "medium",
        "valence": "negative",
        "content_text": "The unfamiliar road had no signs, and Theo felt nervous as the fuel gauge dropped.",
        "neutral_text": "The unfamiliar road had no signs, and Theo watched the fuel gauge drop.",
        "generation_prompt": "Write one fearful sentence about an unfamiliar road and a falling fuel gauge.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about an unfamiliar road and a falling fuel gauge.",
        "note": "travel uncertainty",
    },
]


CONFOUND_ROWS: list[dict[str, str]] = [
    {
        "emotion": "surprise",
        "cause": "schedule_change",
        "arousal": "medium",
        "valence": "neutral",
        "content_text": "Unexpectedly, the meeting moved from room 204 to room 205.",
        "neutral_text": "The meeting moved from room 204 to room 205.",
        "generation_prompt": "Write one surprised but emotionally neutral sentence about a meeting room changing.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about a meeting room changing.",
        "confound": "surprising_neutral",
        "note": "surprise word without target emotion",
    },
    {
        "emotion": "surprise",
        "cause": "inventory",
        "arousal": "medium",
        "valence": "neutral",
        "content_text": "To everyone's surprise, the shipment contained twelve boxes instead of eleven.",
        "neutral_text": "The shipment contained twelve boxes instead of eleven.",
        "generation_prompt": "Write one surprised but emotionally neutral sentence about an inventory count.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about an inventory count.",
        "confound": "surprising_neutral",
        "note": "counting topic",
    },
    {
        "emotion": "calm_positive",
        "cause": "orderly_room",
        "arousal": "low",
        "valence": "positive",
        "content_text": "The quiet room felt pleasantly orderly after the shelves were labeled.",
        "neutral_text": "The room was orderly after the shelves were labeled.",
        "generation_prompt": "Write one calm positive sentence about labeled shelves in a quiet room.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about labeled shelves in a quiet room.",
        "confound": "positive_calm",
        "note": "positive valence, low arousal",
    },
    {
        "emotion": "calm_positive",
        "cause": "tea",
        "arousal": "low",
        "valence": "positive",
        "content_text": "The mild tea tasted pleasant, and the afternoon stayed peaceful.",
        "neutral_text": "The mild tea was served during the afternoon.",
        "generation_prompt": "Write one calm positive sentence about mild tea in the afternoon.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about mild tea in the afternoon.",
        "confound": "positive_calm",
        "note": "positive but not joyful/high arousal",
    },
    {
        "emotion": "arousal_neutral",
        "cause": "timer",
        "arousal": "high",
        "valence": "neutral",
        "content_text": "The timer beeped rapidly while the drill team followed the checklist.",
        "neutral_text": "The timer beeped while the drill team followed the checklist.",
        "generation_prompt": "Write one high-arousal but emotionally neutral sentence about a timer and checklist.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about a timer and checklist.",
        "confound": "arousal_neutral",
        "note": "activation words without target affect",
    },
    {
        "emotion": "arousal_neutral",
        "cause": "auction",
        "arousal": "high",
        "valence": "neutral",
        "content_text": "Numbers flashed quickly on the board as the auctioneer completed the sale.",
        "neutral_text": "Numbers appeared on the board as the auctioneer completed the sale.",
        "generation_prompt": "Write one high-arousal but emotionally neutral sentence about an auction board.",
        "neutral_generation_prompt": "Write one emotionally neutral sentence about an auction board.",
        "confound": "arousal_neutral",
        "note": "fast tempo but no target emotion",
    },
]


def with_ids(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    counters: Counter[str] = Counter()
    out = []
    for row in rows:
        key = row["emotion"]
        counters[key] += 1
        item = {"confound": "", **row}
        item["item_id"] = f"{key}_{counters[key]:02d}"
        out.append(item)
    return out


def validate(rows: list[dict[str, str]]) -> None:
    ids = [r["item_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate item_id in emotion dataset.")
    for row in rows:
        for key in FIELDNAMES:
            if key not in row:
                raise RuntimeError(f"{row.get('item_id', '<unknown>')} missing {key}")
        for key, val in row.items():
            if val != val.strip():
                raise RuntimeError(f"{row['item_id']} has whitespace-padded field {key}")
        if row["content_text"] == row["neutral_text"]:
            raise RuntimeError(f"{row['item_id']} has identical paired texts")


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def update_manifest(path: pathlib.Path, rows: list[dict[str, str]], digest: str) -> None:
    manifest_path = HERE / MANIFEST_NAME
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}
    emotion_counts = Counter(r["emotion"] for r in rows if not r["confound"])
    confound_counts = Counter(r["confound"] for r in rows if r["confound"])
    manifest[path.name] = {
        "generator": pathlib.Path(__file__).name,
        "rows": len(rows),
        "sha256": digest,
        "target_emotions": dict(sorted(emotion_counts.items())),
        "confounds": dict(sorted(confound_counts.items())),
        "pairing": "emotion-laden text/prompt matched to neutral paraphrase/prompt by cause",
        "verified_tokenizers": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    rows = with_ids(TARGET_ROWS + CONFOUND_ROWS)
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
