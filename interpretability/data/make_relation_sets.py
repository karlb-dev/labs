"""Generate the multi-relation probe sets for Labs 1 and 2.

Run once at authoring time; the CSVs are vendored (course rule: no live data
at lab runtime). Deterministic by construction — no RNG anywhere.

Why this exists
===============
The built-in lab prompt sets lean heavily on one relation (country -> capital).
That is fine for teaching the *methods*, but it caps the science: a logit-lens
or DLA claim about "fact recall" supported only by capitals is really a claim
about one relation. This set spans 12 relation classes so students can ask
*comparative* questions: do languages decide at the same depth as capitals?
Do near-tie emotion completions flip later than taxonomy completions? Which
heads carry class information vs instance information?

Every item carries TWO distractors:
  hard  — same class as the target (Paris vs Rome): tests *instance* choice
  easy  — different class entirely (Paris vs hammer): tests *class* choice
The lens/DLA contrast between the hard and easy margins on the SAME prompt is
the new measurement: class information typically appears earlier than
instance information, and the gap differs by relation.

Token discipline: every target and distractor below is verified single-token
with a leading space under BOTH course tokenizers (gpt2 BPE and Olmo-3),
checked at authoring time. The labs' runtime tokenization gates re-verify and
drop (with an audit row) anything a future tokenizer breaks.

Outputs (both are views of the same canonical table):
  relation_probes_lab1.csv  — Lab 1 schema (example_id,category,prompt,target,
                              distractor,note); category="fact", hard
                              distractor; relation + easy distractor in note.
                              Run: --lab lab1 --prompt-set data/relation_probes_lab1.csv
  relation_pairs_lab2.csv   — Lab 2 schema; TWO rows per item: <id>_hard
                              (same-class distractor) and <id>_easy
                              (cross-class distractor).
                              Run: --lab lab2 --prompt-set data/relation_pairs_lab2.csv

The emotion_outcome class deliberately contains near-ties (marathon: proud vs
tired; canceled test: relieved vs happy). Those rows are tagged near_tie=1 in
the note; an "error" there is usually the lens reporting a genuine
distribution split, which is the point.
"""

from __future__ import annotations

import csv
import pathlib

HERE = pathlib.Path(__file__).parent

# (relation, [(slug, prompt, target, hard_distractor, near_tie), ...])
# All targets/distractors single-token under gpt2 + Olmo-3 (verified).
RELATIONS: list[tuple[str, list[tuple[str, str, str, str, bool]]]] = [
    ("capital_of", [
        ("france", "The capital of France is", " Paris", " Rome", False),
        ("japan", "The capital of Japan is", " Tokyo", " Berlin", False),
        ("germany", "The capital of Germany is", " Berlin", " Vienna", False),
        ("italy", "The capital of Italy is", " Rome", " Athens", False),
        ("spain", "The capital of Spain is", " Madrid", " Paris", False),
        ("russia", "The capital of Russia is", " Moscow", " Berlin", False),
        ("egypt", "The capital of Egypt is", " Cairo", " Athens", False),
        ("austria", "The capital of Austria is", " Vienna", " Madrid", False),
    ]),
    ("country_language", [
        ("france", "The official language of France is", " French", " German", False),
        ("japan", "The official language of Japan is", " Japanese", " Korean", False),
        ("germany", "In Germany, most people speak", " German", " Dutch", False),
        ("italy", "The official language of Italy is", " Italian", " Spanish", False),
        ("spain", "In Spain, most people speak", " Spanish", " Portuguese", False),
        ("russia", "The official language of Russia is", " Russian", " Polish", False),
        ("egypt", "In Egypt, most people speak", " Arabic", " Greek", False),
        ("brazil", "In Brazil, most people speak", " Portuguese", " Spanish", False),
        ("greece", "The official language of Greece is", " Greek", " Italian", False),
        ("korea", "In South Korea, most people speak", " Korean", " Japanese", False),
    ]),
    ("antonym", [
        ("hot", "The opposite of hot is", " cold", " cool", False),
        ("big", "The opposite of big is", " small", " short", False),
        ("fast", "The opposite of fast is", " slow", " still", False),
        ("light", "The opposite of light is", " dark", " dim", False),
        ("up", "The opposite of up is", " down", " low", False),
        ("happy", "The opposite of happy is", " sad", " angry", False),
        ("loud", "The opposite of loud is", " quiet", " soft", True),
        ("hard", "The opposite of hard is", " soft", " smooth", False),
        ("wet", "The opposite of wet is", " dry", " cold", False),
        ("strong", "The opposite of strong is", " weak", " small", False),
    ]),
    ("object_color", [
        ("sky", "On a clear day, the sky is", " blue", " gray", False),
        ("grass", "Healthy summer grass is", " green", " yellow", False),
        ("snow", "Freshly fallen snow is", " white", " blue", False),
        ("blood", "The color of blood is", " red", " brown", False),
        ("coal", "The color of coal is", " black", " gray", False),
        ("banana", "A ripe banana is", " yellow", " green", False),
        ("milk", "The color of fresh milk is", " white", " yellow", False),
        ("night", "At midnight, the cloudless sky looks", " black", " blue", True),
    ]),
    ("category_member", [
        ("salmon", "A salmon is a kind of", " fish", " snake", False),
        ("oak", "An oak is a kind of", " tree", " flower", False),
        ("rose", "A rose is a kind of", " flower", " tree", False),
        ("beagle", "A beagle is a breed of", " dog", " bird", False),
        ("sparrow", "A sparrow is a kind of", " bird", " fish", False),
        ("copper", "Copper is a kind of", " metal", " stone", False),
        ("carrot", "A carrot is a kind of", " vegetable", " fruit", False),
        ("cobra", "A cobra is a kind of", " snake", " insect", False),
        ("apple", "An apple is a kind of", " fruit", " vegetable", False),
        ("wasp", "A wasp is a kind of", " insect", " bird", False),
    ]),
    ("counting", [
        ("two_plus_two", "Two plus two equals", " four", " five", False),
        ("week_days", "The number of days in a week is", " seven", " six", False),
        ("spider_legs", "The number of legs on a spider is", " eight", " six", False),
        ("triangle", "The number of sides of a triangle is", " three", " four", False),
        ("insect_legs", "The number of legs on an insect is", " six", " eight", False),
        ("five_plus_five", "Five plus five equals", " ten", " nine", False),
        ("dozen_half", "A dozen divided by two is", " six", " four", False),
        ("year_months", "The number of months in a year is", " twelve", " ten", False),
    ]),
    ("body", [
        ("heart", "The organ that pumps blood is the", " heart", " brain", False),
        ("ears", "You hear with your", " ears", " eyes", False),
        ("eyes", "You see with your", " eyes", " ears", False),
        ("nose", "You smell with your", " nose", " tongue", False),
        ("stomach", "Food is digested mainly in the", " stomach", " heart", False),
        ("brain", "The organ you think with is the", " brain", " heart", False),
        ("bones", "Your skeleton is made of", " bones", " skin", False),
        ("tongue", "You taste with your", " tongue", " nose", False),
    ]),
    ("profession_tool", [
        ("carpenter", "A carpenter drives nails with a", " hammer", " saw", False),
        ("painter", "A painter applies paint with a", " brush", " pen", False),
        ("chef", "A chef chops vegetables with a", " knife", " saw", False),
        ("tailor", "A tailor sews with a", " needle", " pen", False),
        ("plumber", "A plumber tightens pipes with a", " wrench", " hammer", False),
        ("writer", "A writer signs letters with a", " pen", " brush", False),
        ("gardener", "A gardener digs holes with a", " shovel", " knife", False),
    ]),
    ("sequence", [
        ("monday", "The day after Monday is", " Tuesday", " Friday", False),
        ("tuesday", "The day after Tuesday is", " Wednesday", " Monday", False),
        ("thursday", "The day after Thursday is", " Friday", " Sunday", False),
        ("february", "The month after February is", " March", " January", False),
        ("december", "The month after December is", " January", " March", False),
        ("winter", "The season after winter is", " spring", " autumn", False),
        ("summer", "The season after summer is", " autumn", " winter", False),
        ("friday", "The first day of the weekend is", " Saturday", " Sunday", True),
    ]),
    ("material", [
        ("paper", "Paper is made from", " wood", " cotton", False),
        ("wine", "Wine is made from", " grapes", " milk", False),
        ("cheese", "Cheese is made from", " milk", " wood", False),
        ("glass", "Glass is made from", " sand", " clay", False),
        ("sweater", "Sweaters are often knitted from", " wool", " cotton", False),
        ("tshirt", "T-shirts are usually made of", " cotton", " leather", False),
        ("bricks", "Bricks are made from", " clay", " sand", False),
        ("shoes", "Dress shoes are often made of", " leather", " rubber", False),
        ("tires", "Car tires are made of", " rubber", " leather", False),
    ]),
    ("habitat", [
        ("camel", "A camel lives in the", " desert", " jungle", False),
        ("whale", "A whale lives in the", " ocean", " forest", False),
        ("penguin", "A penguin lives on the", " ice", " sand", False),
        ("monkey", "A monkey lives in the", " jungle", " desert", False),
        ("owl", "An owl lives in the", " forest", " ocean", False),
        ("bat", "A bat sleeps in a", " cave", " nest", False),
        ("bee", "A bee lives in a", " hive", " nest", False),
        ("bird", "A bird lays eggs in a", " nest", " hive", False),
        ("frog", "A frog lives near a", " pond", " cave", False),
    ]),
    # The "thinking-flavored" class: social/affective inference, still inside
    # the single-token frame. Near-ties are intentional measurements.
    ("emotion_outcome", [
        ("prize", "Winning a big prize makes most people feel", " happy", " proud", True),
        ("pet", "Losing a beloved pet makes most people feel", " sad", " angry", False),
        ("cheated", "Being cheated by a close friend makes most people feel", " angry", " sad", True),
        ("dark", "Walking alone in the dark makes many people feel", " scared", " tired", False),
        ("marathon", "Finishing a marathon makes most runners feel", " proud", " tired", True),
        ("new_city", "Moving to a new city with no friends can make a person feel", " lonely", " bored", False),
        ("queue", "Waiting in a long, slow line makes most people feel", " bored", " angry", True),
        ("speech", "Speaking in front of a huge crowd makes many people feel", " nervous", " calm", False),
        ("birthday", "Forgetting a close friend's birthday makes most people feel", " guilty", " embarrassed", True),
        ("canceled", "Hearing the exam was canceled made the students feel", " relieved", " happy", True),
    ]),
]

# Cross-class ("easy") distractors: a fixed donor word per relation, drawn
# from a DIFFERENT relation's target space. Deterministic, no RNG.
EASY_DONOR = {
    "capital_of": " hammer",
    "country_language": " blue",
    "antonym": " Paris",
    "object_color": " Tuesday",
    "category_member": " four",
    "counting": " heart",
    "body": " desert",
    "profession_tool": " happy",
    "sequence": " wood",
    "material": " French",
    "habitat": " cold",
    "emotion_outcome": " fish",
}


def lab1_rows() -> list[dict[str, str]]:
    rows = []
    for relation, items in RELATIONS:
        for slug, prompt, target, hard, near_tie in items:
            note = f"relation={relation}; easy_distractor={EASY_DONOR[relation].strip()}"
            if near_tie:
                note += "; near_tie=1 (two plausible answers by design)"
            rows.append({
                "example_id": f"rel_{relation}_{slug}",
                "category": "fact",
                "prompt": prompt,
                "target": target,
                "distractor": hard,
                "note": note,
            })
    return rows


def lab2_rows() -> list[dict[str, str]]:
    rows = []
    for relation, items in RELATIONS:
        category = "relation" if relation == "antonym" else "fact"
        for slug, prompt, target, hard, near_tie in items:
            base_note = f"relation={relation}" + ("; near_tie=1" if near_tie else "")
            rows.append({
                "example_id": f"rel_{relation}_{slug}_hard",
                "category": category,
                "prompt": prompt,
                "target": target,
                "distractor": hard,
                "note": base_note + "; distractor_class=same (instance discrimination)",
            })
            rows.append({
                "example_id": f"rel_{relation}_{slug}_easy",
                "category": category,
                "prompt": prompt,
                "target": target,
                "distractor": EASY_DONOR[relation],
                "note": base_note + "; distractor_class=cross (class discrimination)",
            })
    return rows


def validate_inventory() -> None:
    relation_names = [relation for relation, _ in RELATIONS]
    if len(relation_names) != len(set(relation_names)):
        raise SystemExit("duplicate relation names")
    if set(relation_names) != set(EASY_DONOR):
        missing = sorted(set(relation_names) - set(EASY_DONOR))
        extra = sorted(set(EASY_DONOR) - set(relation_names))
        raise SystemExit(f"EASY_DONOR mismatch: missing={missing}, extra={extra}")

    seen_ids: set[str] = set()
    all_single_word_answers: set[str] = set()
    for relation, items in RELATIONS:
        if not items:
            raise SystemExit(f"relation {relation!r} has no items")
        relation_answers: set[str] = set()
        for slug, prompt, target, hard, near_tie in items:
            item_id = f"{relation}_{slug}"
            if item_id in seen_ids:
                raise SystemExit(f"duplicate relation item id {item_id!r}")
            seen_ids.add(item_id)
            if not isinstance(near_tie, bool):
                raise SystemExit(f"{item_id}: near_tie must be bool")
            if prompt.endswith(" "):
                raise SystemExit(f"{item_id}: prompt must not end with a space")
            for role, token in (("target", target), ("hard distractor", hard), ("easy distractor", EASY_DONOR[relation])):
                if not token.startswith(" "):
                    raise SystemExit(f"{item_id}: {role} {token!r} must carry the leading tokenizer space")
                stripped = token.strip()
                if not stripped:
                    raise SystemExit(f"{item_id}: {role} is empty")
                if " " in stripped:
                    raise SystemExit(f"{item_id}: {role} {token!r} is not a single word")
            if len({target, hard, EASY_DONOR[relation]}) != 3:
                raise SystemExit(f"{item_id}: target, hard, and easy distractors must be distinct")
            relation_answers.update({target.strip(), hard.strip()})
            all_single_word_answers.update({target.strip(), hard.strip(), EASY_DONOR[relation].strip()})

        easy = EASY_DONOR[relation].strip()
        if easy in relation_answers:
            raise SystemExit(f"{relation}: easy donor {easy!r} appears inside the same relation")

    lab1 = lab1_rows()
    lab2 = lab2_rows()
    if len(lab1) != len(seen_ids):
        raise SystemExit(f"Lab 1 row count mismatch: {len(lab1)} rows for {len(seen_ids)} items")
    if len(lab2) != 2 * len(seen_ids):
        raise SystemExit(f"Lab 2 row count mismatch: {len(lab2)} rows for {len(seen_ids)} items")
    if len({r["example_id"] for r in lab1}) != len(lab1):
        raise SystemExit("duplicate Lab 1 example_id")
    if len({r["example_id"] for r in lab2}) != len(lab2):
        raise SystemExit("duplicate Lab 2 example_id")
    if not any("near_tie=1" in r["note"] for r in lab2):
        raise SystemExit("expected at least one tagged near-tie row")
    print(f"validated {len(seen_ids)} relation items with {len(all_single_word_answers)} answer tokens")


def write_csv(name: str, rows: list[dict[str, str]]) -> None:
    path = HERE / name
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    relations = {r["note"].split(";")[0] for r in rows}
    print(f"{name}: {len(rows)} rows across {len(relations)} relations")


def main() -> None:
    validate_inventory()
    write_csv("relation_probes_lab1.csv", lab1_rows())
    write_csv("relation_pairs_lab2.csv", lab2_rows())


if __name__ == "__main__":
    main()
