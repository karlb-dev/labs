"""Generate the Lab 18 humor/incongruity contrast set.

The rows are short authored micro-scenes. Each setup has five matched
endings: joke, literal, surprising-not-funny, silly-not-joke, and
positive-sentiment-not-joke. The point is not to make a definitive humor
benchmark; it is to give Lab 18 a frozen, auditable battery where the cheap
correlates of humor are explicitly present.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib
from collections import Counter

HERE = pathlib.Path(__file__).parent
OUT_NAME = "humor_incongruity_pairs.csv"
MANIFEST_NAME = "MANIFEST.json"

FIELDNAMES = [
    "item_id",
    "family",
    "setup",
    "joke_completion",
    "literal_completion",
    "surprise_completion",
    "silly_completion",
    "positive_completion",
    "setup_anchor",
    "resolution_keyword",
    "joke_markers",
    "silly_markers",
    "surprise_markers",
    "positive_markers",
    "note",
]


ITEMS = [
    {
        "item_id": "pun_00",
        "family": "pun_wordplay",
        "setup": "The spreadsheet joined a band but refused to play guitar.",
        "joke_completion": "It said it only knew how to handle the cells.",
        "literal_completion": "It used a spreadsheet program instead of a guitar.",
        "surprise_completion": "It played a trumpet from inside a filing cabinet.",
        "silly_completion": "It wore a lampshade and declared Tuesday square.",
        "positive_completion": "Everyone enjoyed the music and felt cheerful.",
        "setup_anchor": "spreadsheet|band|guitar",
        "resolution_keyword": "cells",
        "joke_markers": "cells|spreadsheet|handle",
        "silly_markers": "lampshade|tuesday|square",
        "surprise_markers": "trumpet|filing cabinet",
        "positive_markers": "enjoyed|cheerful|music",
    },
    {
        "item_id": "pun_01",
        "family": "pun_wordplay",
        "setup": "The bakery hired a new clock to help with morning orders.",
        "joke_completion": "It was great at making time rolls.",
        "literal_completion": "The clock helped staff keep the schedule.",
        "surprise_completion": "It printed receipts in ancient Greek.",
        "silly_completion": "It saluted a muffin and spun in circles.",
        "positive_completion": "The customers smiled at the fresh bread.",
        "setup_anchor": "bakery|clock|orders",
        "resolution_keyword": "rolls",
        "joke_markers": "time rolls|rolls|clock",
        "silly_markers": "saluted|muffin|spun",
        "surprise_markers": "receipts|ancient greek",
        "positive_markers": "smiled|fresh|bread",
    },
    {
        "item_id": "pun_02",
        "family": "pun_wordplay",
        "setup": "The librarian brought a ladder to the poetry shelf.",
        "joke_completion": "She wanted to reach a higher verse.",
        "literal_completion": "She used the ladder to reach books on the top shelf.",
        "surprise_completion": "The shelf became a small door to a train station.",
        "silly_completion": "The catalog sneezed glitter onto a calendar.",
        "positive_completion": "The reading room felt calm and welcoming.",
        "setup_anchor": "librarian|ladder|poetry|shelf",
        "resolution_keyword": "verse",
        "joke_markers": "higher verse|verse|poetry",
        "silly_markers": "sneezed|glitter|calendar",
        "surprise_markers": "door|train station",
        "positive_markers": "calm|welcoming|reading room",
    },
    {
        "item_id": "pun_03",
        "family": "pun_wordplay",
        "setup": "The programmer opened a cafe with a debug menu.",
        "joke_completion": "Every order came with a side of breakpoints.",
        "literal_completion": "The menu listed software debugging tools as a theme.",
        "surprise_completion": "The espresso machine started compiling weather reports.",
        "silly_completion": "A spoon gave a lecture about triangles.",
        "positive_completion": "The cafe was friendly and bright.",
        "setup_anchor": "programmer|cafe|debug|menu",
        "resolution_keyword": "breakpoints",
        "joke_markers": "breakpoints|debug|order",
        "silly_markers": "spoon|lecture|triangles",
        "surprise_markers": "espresso|compiling|weather",
        "positive_markers": "friendly|bright|cafe",
    },
    {
        "item_id": "pun_04",
        "family": "pun_wordplay",
        "setup": "The tailor fixed the calendar's torn page.",
        "joke_completion": "It was a date in need of a stitch.",
        "literal_completion": "The tailor repaired a paper calendar page.",
        "surprise_completion": "The calendar predicted yesterday's phone call.",
        "silly_completion": "The thimble sang about soup at midnight.",
        "positive_completion": "The repair made the shop feel tidy.",
        "setup_anchor": "tailor|calendar|page",
        "resolution_keyword": "stitch",
        "joke_markers": "date|stitch|calendar",
        "silly_markers": "thimble|soup|midnight",
        "surprise_markers": "predicted|yesterday|phone call",
        "positive_markers": "repair|tidy|shop",
    },
    {
        "item_id": "expect_00",
        "family": "expectation_violation",
        "setup": "Mara brought an umbrella to the board meeting on a sunny day.",
        "joke_completion": "She said the forecast called for brainstorming.",
        "literal_completion": "She brought it by mistake because she forgot the weather.",
        "surprise_completion": "The ceiling sprinklers tested themselves during the agenda item.",
        "silly_completion": "She opened it and found tiny paperwork confetti.",
        "positive_completion": "Her coworkers appreciated the careful preparation.",
        "setup_anchor": "umbrella|board meeting|sunny",
        "resolution_keyword": "brainstorming",
        "joke_markers": "forecast|brainstorming|meeting",
        "silly_markers": "tiny|paperwork|confetti",
        "surprise_markers": "ceiling|sprinklers|agenda",
        "positive_markers": "appreciated|careful|preparation",
    },
    {
        "item_id": "expect_01",
        "family": "expectation_violation",
        "setup": "The elevator apologized before stopping between floors.",
        "joke_completion": "It said it was having an up-and-down day.",
        "literal_completion": "A recorded message apologized during a mechanical delay.",
        "surprise_completion": "The doors opened onto a quiet library.",
        "silly_completion": "The floor buttons rearranged themselves into a smile.",
        "positive_completion": "The passengers stayed patient and helpful.",
        "setup_anchor": "elevator|apologized|floors",
        "resolution_keyword": "up-and-down",
        "joke_markers": "up-and-down|day|elevator",
        "silly_markers": "buttons|smile|rearranged",
        "surprise_markers": "doors|quiet library",
        "positive_markers": "patient|helpful|passengers",
    },
    {
        "item_id": "expect_02",
        "family": "expectation_violation",
        "setup": "Leo labeled an empty jar 'emergency ideas'.",
        "joke_completion": "He opened it whenever he needed a fresh thought.",
        "literal_completion": "The jar was a decorative reminder to brainstorm.",
        "surprise_completion": "The jar contained a handwritten map of the basement.",
        "silly_completion": "The jar insisted Thursdays are made of soup.",
        "positive_completion": "The label made the desk feel playful.",
        "setup_anchor": "empty jar|emergency ideas",
        "resolution_keyword": "fresh thought",
        "joke_markers": "fresh thought|ideas|opened",
        "silly_markers": "thursdays|soup|insisted",
        "surprise_markers": "handwritten map|basement",
        "positive_markers": "playful|desk|label",
    },
    {
        "item_id": "expect_03",
        "family": "expectation_violation",
        "setup": "The printer refused the final page of the report.",
        "joke_completion": "It said the ending was too paper-thin.",
        "literal_completion": "The printer jammed before printing the last page.",
        "surprise_completion": "The printer produced a blank ticket to a rooftop garden.",
        "silly_completion": "It hummed a lullaby to the stapler.",
        "positive_completion": "The team fixed the issue and felt relieved.",
        "setup_anchor": "printer|final page|report",
        "resolution_keyword": "paper-thin",
        "joke_markers": "paper-thin|ending|printer",
        "silly_markers": "hummed|lullaby|stapler",
        "surprise_markers": "blank ticket|rooftop garden",
        "positive_markers": "fixed|relieved|team",
    },
    {
        "item_id": "expect_04",
        "family": "expectation_violation",
        "setup": "Nina put a tiny chair beside the Wi-Fi router.",
        "joke_completion": "She wanted the signal to have better reception.",
        "literal_completion": "She placed a decoration next to the router.",
        "surprise_completion": "The router began broadcasting in Morse code.",
        "silly_completion": "The chair demanded a password for sitting.",
        "positive_completion": "The room looked charming after she tidied it.",
        "setup_anchor": "chair|wi-fi|router",
        "resolution_keyword": "reception",
        "joke_markers": "signal|reception|router",
        "silly_markers": "chair|password|sitting",
        "surprise_markers": "broadcasting|morse code",
        "positive_markers": "charming|tidied|room",
    },
    {
        "item_id": "caption_00",
        "family": "caption_scene",
        "setup": "Caption for a photo: a mug sits beside a laptop showing 99 open tabs.",
        "joke_completion": "The coffee is not helping, but it has agreed to supervise.",
        "literal_completion": "A drink sits next to a computer with many browser tabs open.",
        "surprise_completion": "The laptop screen shows a live feed from a submarine.",
        "silly_completion": "The mug declares itself mayor of the desk.",
        "positive_completion": "The workspace looks busy but comfortable.",
        "setup_anchor": "mug|laptop|open tabs",
        "resolution_keyword": "supervise",
        "joke_markers": "coffee|helping|supervise",
        "silly_markers": "mug|mayor|desk",
        "surprise_markers": "live feed|submarine",
        "positive_markers": "busy|comfortable|workspace",
    },
    {
        "item_id": "caption_01",
        "family": "caption_scene",
        "setup": "Caption for a photo: a suitcase is packed with notebooks and one shoe.",
        "joke_completion": "The plan is ready; the other shoe is still gathering evidence.",
        "literal_completion": "The suitcase contains notebooks and only one shoe.",
        "surprise_completion": "The suitcase plays a voicemail from the future.",
        "silly_completion": "The notebooks form a tiny courtroom.",
        "positive_completion": "The trip preparations seem organized and hopeful.",
        "setup_anchor": "suitcase|notebooks|shoe",
        "resolution_keyword": "other shoe",
        "joke_markers": "other shoe|evidence|plan",
        "silly_markers": "notebooks|tiny courtroom",
        "surprise_markers": "voicemail|future",
        "positive_markers": "organized|hopeful|trip",
    },
    {
        "item_id": "caption_02",
        "family": "caption_scene",
        "setup": "Caption for a photo: a conference badge reads 'Ask me after coffee'.",
        "joke_completion": "Networking has entered low-power mode.",
        "literal_completion": "The badge asks people to wait until the wearer has coffee.",
        "surprise_completion": "The badge displays tomorrow's agenda instead.",
        "silly_completion": "The badge challenges the lanyard to a dance contest.",
        "positive_completion": "The message is friendly and relatable.",
        "setup_anchor": "conference badge|coffee",
        "resolution_keyword": "low-power",
        "joke_markers": "networking|low-power|coffee",
        "silly_markers": "lanyard|dance contest",
        "surprise_markers": "tomorrow|agenda",
        "positive_markers": "friendly|relatable|message",
    },
    {
        "item_id": "caption_03",
        "family": "caption_scene",
        "setup": "Caption for a photo: a calendar has every Friday circled twice.",
        "joke_completion": "Even the calendar is requesting a weekend extension.",
        "literal_completion": "The Fridays are marked more than once.",
        "surprise_completion": "The calendar pages fold themselves into a paper telescope.",
        "silly_completion": "The circles start arguing about geometry.",
        "positive_completion": "The schedule suggests excitement for the weekend.",
        "setup_anchor": "calendar|friday|circled",
        "resolution_keyword": "weekend extension",
        "joke_markers": "weekend|extension|calendar",
        "silly_markers": "circles|geometry|arguing",
        "surprise_markers": "paper telescope|fold",
        "positive_markers": "excitement|weekend|schedule",
    },
    {
        "item_id": "caption_04",
        "family": "caption_scene",
        "setup": "Caption for a photo: a whiteboard says 'final final plan'.",
        "joke_completion": "The plan is almost ready to become final_final_really_final.",
        "literal_completion": "The whiteboard labels the plan as final twice.",
        "surprise_completion": "The marker writes a message without anyone touching it.",
        "silly_completion": "The eraser forms a committee about crumbs.",
        "positive_completion": "The team is close to finishing.",
        "setup_anchor": "whiteboard|final final|plan",
        "resolution_keyword": "final_final",
        "joke_markers": "final_final|really_final|plan",
        "silly_markers": "eraser|committee|crumbs",
        "surprise_markers": "marker|writes|message",
        "positive_markers": "team|close|finishing",
    },
    {
        "item_id": "twist_00",
        "family": "resolution_twist",
        "setup": "The chef said the soup needed more confidence.",
        "joke_completion": "So they added a little thyme to believe in itself.",
        "literal_completion": "The chef meant the soup needed stronger seasoning.",
        "surprise_completion": "The soup began reciting a weather forecast.",
        "silly_completion": "The ladle wore sunglasses and applauded the bowl.",
        "positive_completion": "The final soup tasted warm and comforting.",
        "setup_anchor": "chef|soup|confidence",
        "resolution_keyword": "thyme",
        "joke_markers": "thyme|confidence|believe",
        "silly_markers": "ladle|sunglasses|applauded",
        "surprise_markers": "reciting|weather forecast",
        "positive_markers": "warm|comforting|tasted",
    },
    {
        "item_id": "twist_01",
        "family": "resolution_twist",
        "setup": "The accountant bought a compass before tax season.",
        "joke_completion": "They wanted every deduction to point in the right direction.",
        "literal_completion": "The compass was a desk decoration for the office.",
        "surprise_completion": "The compass pointed only toward unpaid invoices.",
        "silly_completion": "The calculator put on a cape and whispered fractions.",
        "positive_completion": "The office felt prepared for the deadline.",
        "setup_anchor": "accountant|compass|tax",
        "resolution_keyword": "deduction",
        "joke_markers": "deduction|right direction|compass",
        "silly_markers": "calculator|cape|fractions",
        "surprise_markers": "unpaid invoices|pointed",
        "positive_markers": "prepared|deadline|office",
    },
    {
        "item_id": "twist_02",
        "family": "resolution_twist",
        "setup": "The musician tuned the silent piano for an hour.",
        "joke_completion": "It still had a lot of key issues.",
        "literal_completion": "The piano needed mechanical adjustment despite making little sound.",
        "surprise_completion": "The piano printed a train schedule from middle C.",
        "silly_completion": "The bench told a very serious joke about socks.",
        "positive_completion": "The instrument sounded better afterward.",
        "setup_anchor": "musician|piano|silent",
        "resolution_keyword": "key issues",
        "joke_markers": "key issues|piano|tuned",
        "silly_markers": "bench|serious joke|socks",
        "surprise_markers": "train schedule|middle c",
        "positive_markers": "better|afterward|instrument",
    },
    {
        "item_id": "twist_03",
        "family": "resolution_twist",
        "setup": "The gardener wrote meeting notes in the margin of a seed packet.",
        "joke_completion": "They wanted the ideas to grow on everyone.",
        "literal_completion": "The seed packet was the only paper available.",
        "surprise_completion": "The packet included directions to a hidden office.",
        "silly_completion": "The margin claimed it preferred blue Mondays.",
        "positive_completion": "The garden planning felt collaborative.",
        "setup_anchor": "gardener|meeting notes|seed packet",
        "resolution_keyword": "grow",
        "joke_markers": "ideas|grow|seed",
        "silly_markers": "blue mondays|margin",
        "surprise_markers": "hidden office|directions",
        "positive_markers": "collaborative|planning|garden",
    },
    {
        "item_id": "twist_04",
        "family": "resolution_twist",
        "setup": "The analyst polished a crystal ball before reviewing the spreadsheet.",
        "joke_completion": "Forecasting needed a cleaner view.",
        "literal_completion": "The crystal ball was a joke prop near the analyst's desk.",
        "surprise_completion": "The spreadsheet began updating next week's weather.",
        "silly_completion": "The charts marched across the desk in tiny hats.",
        "positive_completion": "The review went smoothly and clearly.",
        "setup_anchor": "analyst|crystal ball|spreadsheet",
        "resolution_keyword": "forecasting",
        "joke_markers": "forecasting|cleaner view|spreadsheet",
        "silly_markers": "charts|tiny hats|marched",
        "surprise_markers": "next week's weather|updating",
        "positive_markers": "smoothly|clearly|review",
    },
]


def rows() -> list[dict[str, str]]:
    return [{**item, "note": "authored setup with matched joke/literal/surprise/silly/positive endings"} for item in ITEMS]


def validate(data: list[dict[str, str]]) -> None:
    ids = [r["item_id"] for r in data]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate item_id in humor data.")
    families = Counter(r["family"] for r in data)
    if set(families.values()) != {5}:
        raise RuntimeError(f"Expected 5 rows per family, got {dict(families)}")
    for row in data:
        for key in FIELDNAMES:
            if key not in row:
                raise RuntimeError(f"{row.get('item_id', '<unknown>')} missing {key}")
        completions = [
            row["joke_completion"],
            row["literal_completion"],
            row["surprise_completion"],
            row["silly_completion"],
            row["positive_completion"],
        ]
        if len(completions) != len(set(completions)):
            raise RuntimeError(f"{row['item_id']} has duplicate completions.")
        for key, value in row.items():
            if value != value.strip():
                raise RuntimeError(f"{row['item_id']} has whitespace-padded {key}")
            if "\n" in value:
                raise RuntimeError(f"{row['item_id']} has a newline in {key}")


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def update_manifest(path: pathlib.Path, data: list[dict[str, str]], digest: str) -> None:
    manifest_path = HERE / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest[path.name] = {
        "generator": pathlib.Path(__file__).name,
        "rows": len(data),
        "sha256": digest,
        "families": dict(sorted(Counter(r["family"] for r in data).items())),
        "conditions": {
            "joke": len(data),
            "literal": len(data),
            "positive": len(data),
            "silly": len(data),
            "surprise": len(data),
        },
        "pairing": "one setup with joke, literal, surprising-not-funny, silly-not-joke, and positive-not-joke completions",
        "source": "authored course micro-scenes; no long copyrighted excerpts",
        "verified_tokenizers": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    data = rows()
    validate(data)
    out_path = HERE / OUT_NAME
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for row in data:
            writer.writerow({key: row[key] for key in FIELDNAMES})
    digest = sha256(out_path)
    update_manifest(out_path, data, digest)
    print(f"wrote {out_path} ({len(data)} rows, sha256={digest})")


if __name__ == "__main__":
    main()
