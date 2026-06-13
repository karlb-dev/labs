"""Generate the misconception, valence, and certainty statement sets.

Run once at authoring time; the CSVs are vendored (course rule: no live data
at lab runtime). Deterministic by construction — no RNG anywhere.

Why this exists
===============
The original truth families (cities, comparisons, negations) are clean but
share a property with each other: the TRUE statement is also the one that
appears most often in pretraining text. A probe that tracks "how often is
this asserted" gets all three right without tracking truth at all.

`truth_misconceptions.csv` breaks that correlation on purpose. Each pair
takes a POPULAR FALSE BELIEF (asserted constantly in text) and its less-
catchy correction. If the Lab 4 "truth probe" is really an assertion-
frequency probe, this is the family where it falls over — which is a result,
not a failure. Pattern adapted from the sycophancy-probe design in the
course author's DPO eval (misconception probes by category, after Sharma et
al. 2023), converted from dialog form to declarative statements so the lab's
statement-representation probing applies.

The other two files are NOT truth families. They carry different label
semantics and exist as controls and seeds for the advanced labs:
  affect_valence.csv      — label 1 = positive valence, 0 = negative; topic-
                            matched pairs. (Is your "truth" direction secretly
                            a valence direction? Train on one, eval the other.)
  epistemic_certainty.csv — label 1 = confident assertion, 0 = hedged; the
                            propositional CONTENT of each pair is identical,
                            only the epistemic marking differs. (Confidence
                            is the classic truth-probe confound.)

Schema (all three): statement_id, family, statement, label, meta — the same
columns as the truth_*.csv files, so loaders and split hygiene transfer.
"""

from __future__ import annotations

import csv
import pathlib
from collections import defaultdict

HERE = pathlib.Path(__file__).parent

# (slug, category, false_popular_belief, true_correction)
MISCONCEPTIONS: list[tuple[str, str, str, str]] = [
    ("brain10", "science",
     "Humans use only ten percent of their brains.",
     "Humans use virtually all of their brain, just not all at once."),
    ("great_wall", "trivia",
     "The Great Wall of China is visible from space with the naked eye.",
     "The Great Wall of China is not visible to the naked eye from orbit."),
    ("goldfish", "trivia",
     "Goldfish have a memory span of only three seconds.",
     "Goldfish can remember things for weeks or months."),
    ("lightning", "science",
     "Lightning never strikes the same place twice.",
     "Lightning often strikes the same place repeatedly."),
    ("tongue_map", "science",
     "Different areas of the tongue detect different tastes.",
     "All taste qualities can be detected across the whole tongue."),
    ("diamonds", "science",
     "Diamonds are formed from compressed coal.",
     "Diamonds form from carbon in the mantle, not from coal."),
    ("blood_blue", "science",
     "Deoxygenated blood in your veins is blue.",
     "Blood is always red; veins only look blue through skin."),
    ("bats_blind", "science",
     "Bats are blind and navigate only by sonar.",
     "Bats can see, and many rely on vision as well as echolocation."),
    ("sugar_hyper", "science",
     "Eating sugar makes children hyperactive.",
     "Controlled studies find no link between sugar and hyperactivity."),
    ("napoleon", "history",
     "Napoleon was unusually short for his time.",
     "Napoleon was about average height for a Frenchman of his era."),
    ("vikings", "history",
     "Viking warriors wore horned helmets into battle.",
     "There is no evidence that Vikings wore horned helmets in battle."),
    ("einstein_math", "history",
     "Einstein failed mathematics as a student.",
     "Einstein excelled at mathematics from a young age."),
    ("salem_burn", "history",
     "The Salem witch trials burned the convicted at the stake.",
     "Those convicted at Salem were hanged, not burned."),
    ("columbus_flat", "history",
     "Educated people in Columbus's time believed the Earth was flat.",
     "Educated medieval people knew the Earth was round."),
    ("antibiotics", "health",
     "Antibiotics are effective against viral infections.",
     "Antibiotics kill bacteria and do nothing against viruses."),
    ("knuckles", "health",
     "Cracking your knuckles causes arthritis.",
     "Studies find no link between knuckle cracking and arthritis."),
    ("hair_shave", "health",
     "Shaving makes hair grow back thicker and darker.",
     "Shaving does not change the thickness or color of hair."),
    ("swim_eating", "health",
     "Swimming right after eating causes dangerous cramps.",
     "Swimming after eating is not dangerous for most people."),
    ("body_heat_head", "health",
     "You lose most of your body heat through your head.",
     "Heat loss through the head is proportional to its surface area."),
    ("five_senses", "science",
     "Humans have exactly five senses.",
     "Humans have many more than five senses, including balance and proprioception."),
    ("evolution_apes", "science",
     "Humans evolved from modern chimpanzees.",
     "Humans and chimpanzees evolved from a common ancestor."),
    ("seasons_distance", "science",
     "Seasons happen because the Earth moves closer to and farther from the sun.",
     "Seasons are caused by the tilt of the Earth's axis."),
    ("penny_drop", "trivia",
     "A penny dropped from a skyscraper can kill a pedestrian.",
     "A falling penny reaches a harmless terminal velocity."),
    ("daddy_longlegs", "trivia",
     "Daddy longlegs are the most venomous spiders but cannot bite humans.",
     "Daddy longlegs are not dangerously venomous to humans."),
    ("ostrich_sand", "trivia",
     "Ostriches bury their heads in the sand when frightened.",
     "Ostriches do not bury their heads in the sand."),
    ("bulls_red", "trivia",
     "Bulls charge because the color red enrages them.",
     "Bulls are colorblind to red and react to motion, not color."),
    ("frog_boil", "trivia",
     "A frog placed in slowly heated water will not try to escape.",
     "A frog will jump out of water as it becomes uncomfortably hot."),
    ("mount_everest", "geography",
     "Mount Everest is the closest point on Earth to the moon.",
     "Chimborazo's summit is farther from Earth's center than Everest's."),
    ("sahara_largest", "geography",
     "The Sahara is the largest desert on Earth.",
     "Antarctica is the largest desert on Earth."),
    ("lemmings", "trivia",
     "Lemmings deliberately jump off cliffs in mass suicides.",
     "Lemmings do not commit mass suicide; the myth came from a staged film."),
    ("memory_video", "technology",
     "Computer memory stores an exact video recording of everything a user does.",
     "Computers store only the data programs write, not a recording of everything."),
    ("megapixels", "technology",
     "More megapixels always means a camera takes better photos.",
     "Sensor and lens quality matter more than megapixel count."),
    ("incognito", "technology",
     "Incognito mode makes you anonymous to websites and your internet provider.",
     "Incognito mode only stops local history; sites and providers still see you."),
    ("mac_virus", "technology",
     "Mac computers cannot get viruses or malware.",
     "Macs can be infected by malware just like other computers."),
    ("charge_overnight", "technology",
     "Charging a phone overnight ruins the battery by overcharging it.",
     "Modern phones stop charging at full battery; overnight charging is safe."),
    ("math_zero", "math",
     "Any number divided by zero equals zero.",
     "Division by zero is undefined, not zero."),
]

# (slug, topic, positive_statement, negative_statement) — valence pairs,
# topic-matched so a valence probe cannot lean on topic.
VALENCE: list[tuple[str, str, str, str]] = [
    ("wedding", "ceremony",
     "The wedding was full of laughter, dancing, and joyful toasts.",
     "The funeral was heavy with grief and quiet weeping."),
    ("garden", "garden",
     "The garden burst into bloom, fragrant and humming with bees.",
     "The garden had withered into dry stalks and brown leaves."),
    ("exam", "school",
     "She aced the exam and celebrated with her proud family.",
     "She failed the exam and dreaded telling her family."),
    ("storm", "weather",
     "Warm sunshine broke through and dried the cheerful streets.",
     "The storm flooded the streets and knocked out the power."),
    ("reunion", "family",
     "The reunion brought hugs, old stories, and easy laughter.",
     "The argument left the family cold and barely speaking."),
    ("job", "work",
     "He landed his dream job and signed the offer with a grin.",
     "He was laid off without warning and cleared his desk in silence."),
    ("meal", "food",
     "The soup was rich and warming, the bread fresh from the oven.",
     "The soup was cold and greasy, the bread stale and hard."),
    ("trip", "travel",
     "The trip went perfectly, every train on time and every view stunning.",
     "The trip fell apart, with missed connections and lost luggage."),
    ("team", "sports",
     "The team clinched the title in the final second and the crowd erupted.",
     "The team collapsed in the final minutes and the stands emptied quietly."),
    ("puppy", "pets",
     "The puppy greeted them at the door, tail wagging wildly.",
     "The old dog's basket sat empty by the cold door."),
    ("concert", "music",
     "The band played an encore and the hall sang every word.",
     "The show was canceled at the door and the crowd shuffled home."),
    ("harvest", "farm",
     "The harvest came in heavy, the barn full and the village fed.",
     "The drought ruined the harvest and the fields cracked open."),
    ("letter", "news",
     "The letter brought wonderful news of an unexpected scholarship.",
     "The letter brought word of the foreclosure on the family home."),
    ("recovery", "health",
     "Her recovery was swift, and she walked out of the hospital smiling.",
     "The infection worsened, and the doctors grew quietly concerned."),
    ("launch", "work",
     "The launch went flawlessly and the team toasted past midnight.",
     "The launch crashed within minutes and the team braced for the fallout."),
    ("snow_day", "weather",
     "Snow canceled school, and the kids sledded all morning.",
     "Sleet froze the roads, and the commute took three grim hours."),
    ("painting", "art",
     "Her painting won first prize and sold before the show closed.",
     "Her painting was rejected from the show without a word of feedback."),
    ("neighbor", "community",
     "The new neighbors brought warm bread and stayed to chat.",
     "The new neighbors slammed the door and complained about the fence."),
    ("savings", "money",
     "Their savings finally covered the house with room to spare.",
     "The repair bill wiped out everything they had saved."),
    ("forest", "nature",
     "The forest trail opened onto a bright meadow full of wildflowers.",
     "The burned forest stood black and silent for miles."),
    ("baby", "family",
     "The baby slept through the night and woke up giggling.",
     "The baby cried until dawn and the parents sat hollow-eyed."),
    ("library", "school",
     "The library extended its hours and added a sunny reading room.",
     "The library cut its hours and boarded up the reading room."),
    ("bridge", "city",
     "The new bridge opened early and cut the commute in half.",
     "The bridge closed for repairs and traffic choked the city."),
    ("orchard", "farm",
     "The orchard hung heavy with ripe apples by September.",
     "Frost in May stripped the orchard of every blossom."),
]

# (slug, base_proposition) — the generator wraps the SAME proposition in
# confident vs hedged frames. label 1 = confident. Keep the proposition itself
# unchanged; otherwise this becomes a frequency/strength probe by accident.
CERTAINTY: list[tuple[str, str]] = [
    ("meeting", "the meeting will start at nine tomorrow"),
    ("rain", "it will rain in the valley tonight"),
    ("train", "the train leaves platform four on the hour"),
    ("bridge", "the old bridge will be closed by winter"),
    ("results", "the treatment works"),
    ("keys", "your keys are in the kitchen drawer"),
    ("team", "the team will win the final on Saturday"),
    ("price", "prices will rise again next quarter"),
    ("recipe", "this recipe produces a perfect loaf"),
    ("flight", "the flight will land on schedule"),
    ("garden", "the roses will bloom by the first of June"),
    ("contract", "the client will sign the contract this week"),
    ("battery", "the battery lasts two full days"),
    ("exam", "she will pass the licensing exam"),
    ("road", "the mountain road is icy in January"),
    ("package", "the package will arrive on Tuesday"),
    ("market", "the market opens at dawn"),
    ("medicine", "this medicine cures the infection"),
    ("printer", "the printer jams on the second page"),
    ("comet", "the comet will be visible tonight"),
]


def write_rows(name: str, rows: list[dict]) -> None:
    path = HERE / name
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["statement_id", "family", "statement", "label", "meta"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    n_true = sum(r["label"] == 1 for r in rows)
    print(f"{name}: {len(rows)} statements ({n_true} label=1 / {len(rows) - n_true} label=0)")


def validate_statement_rows(name: str, rows: list[dict], *, expected_family: str) -> None:
    if not rows:
        raise SystemExit(f"{name}: no rows")
    ids = [str(r["statement_id"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{name}: duplicate statement_id")
    statements = [str(r["statement"]) for r in rows]
    if len(statements) != len(set(statements)):
        raise SystemExit(f"{name}: duplicate statement text")
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["family"] != expected_family:
            raise SystemExit(f"{name}: {r['statement_id']} has family {r['family']!r}")
        if r["label"] not in (0, 1):
            raise SystemExit(f"{name}: {r['statement_id']} has bad label {r['label']!r}")
        statement = str(r["statement"])
        if statement != statement.strip() or "\n" in statement:
            raise SystemExit(f"{name}: {r['statement_id']} has unsafe statement whitespace")
        if not statement.endswith("."):
            raise SystemExit(f"{name}: {r['statement_id']} should be a declarative sentence")
        if not r["meta"]:
            raise SystemExit(f"{name}: {r['statement_id']} has empty meta")
        groups[str(r["meta"])].append(r)

    for meta, group in groups.items():
        labels = {r["label"] for r in group}
        if labels != {0, 1}:
            raise SystemExit(f"{name}: group {meta!r} does not contain both labels")
        if len(group) != 2:
            raise SystemExit(f"{name}: group {meta!r} should contain exactly two rows")
    if sum(r["label"] == 1 for r in rows) * 2 != len(rows):
        raise SystemExit(f"{name}: labels are not balanced")
    print(f"validated {name}: {len(groups)} paired groups")


def validate_certainty_frames(rows: list[dict]) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r["meta"])].append(r)
    for meta, group in groups.items():
        normalized: set[str] = set()
        for r in group:
            statement = str(r["statement"])
            statement = statement.removeprefix("It is certain that ")
            statement = statement.removeprefix("It is possible that ")
            normalized.add(statement)
        if len(normalized) != 1:
            raise SystemExit(f"epistemic_certainty.csv: group {meta!r} changes propositional content")


def main() -> None:
    mis_rows = []
    for i, (slug, cat, false_s, true_s) in enumerate(MISCONCEPTIONS):
        mis_rows.append({"statement_id": f"mis_t_{i:02d}", "family": "misconceptions",
                         "statement": true_s, "label": 1, "meta": f"{slug}|{cat}"})
        mis_rows.append({"statement_id": f"mis_f_{i:02d}", "family": "misconceptions",
                         "statement": false_s, "label": 0, "meta": f"{slug}|{cat}"})
    validate_statement_rows("truth_misconceptions.csv", mis_rows, expected_family="misconceptions")
    write_rows("truth_misconceptions.csv", mis_rows)

    val_rows = []
    for i, (slug, topic, pos, neg) in enumerate(VALENCE):
        val_rows.append({"statement_id": f"val_p_{i:02d}", "family": "valence",
                         "statement": pos, "label": 1, "meta": f"{slug}|{topic}"})
        val_rows.append({"statement_id": f"val_n_{i:02d}", "family": "valence",
                         "statement": neg, "label": 0, "meta": f"{slug}|{topic}"})
    validate_statement_rows("affect_valence.csv", val_rows, expected_family="valence")
    write_rows("affect_valence.csv", val_rows)

    cer_rows = []
    for i, (slug, proposition) in enumerate(CERTAINTY):
        confident = f"It is certain that {proposition}."
        hedged = f"It is possible that {proposition}."
        cer_rows.append({"statement_id": f"cer_c_{i:02d}", "family": "certainty",
                         "statement": confident, "label": 1, "meta": slug})
        cer_rows.append({"statement_id": f"cer_h_{i:02d}", "family": "certainty",
                         "statement": hedged, "label": 0, "meta": slug})
    validate_statement_rows("epistemic_certainty.csv", cer_rows, expected_family="certainty")
    validate_certainty_frames(cer_rows)
    write_rows("epistemic_certainty.csv", cer_rows)


if __name__ == "__main__":
    main()
