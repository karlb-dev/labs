# N1.5 — HUMAN-AUDITED canonical family map (nextsteps_2_2 §2.1, P0 blocker).
#
# THE DEFECT THIS REPAIRS. `battery.py` derived the clustering family as
# `name.split("-")[0]`. That is a string accident, not the data-generating
# unit: `atomic-80-state` and `ex-element-state-80-8` are THE SAME prompt
# template (element atomic number -> state of matter) but landed in
# families `atomic` and `ex`; `ex` collected 16 unrelated items purely
# because their names start with "ex"; and the one-hop battery gave each
# item its own family even though ten of them are the single template
# "The capital of {country} is". Every paired CI, ICC, tail-rate interval
# and the G6 power simulation clusters on that field, so all of them
# inherit the error.
#
# THE RULE, fixed before recomputation and applied to every item:
#
#   canonical_family = (second-hop relation, answer type)
#       i.e. WHICH FACT the model must retrieve once the bridge entity is
#       resolved, together with the answer's closed set. Two items are the
#       same family iff answering both requires the same relation over the
#       same answer vocabulary — that is precisely the condition under
#       which a second item is "more of the same" rather than independent
#       evidence.
#   template_id  = the surface frame within a family (finer than family).
#       Recorded separately so the map can be tested for the review's
#       failure mode: two nominally distinct families sharing a template.
#
# Consequences of the rule, recorded because they are the interesting
# calls (a reviewer should be able to disagree with these specifically):
#   * "The color of {the swan | gold | Mars | the ruby} is" is ONE family
#     (entity_color): identical relation, identical answer vocabulary,
#     identical frame. Splitting by entity domain would be the same
#     mistake as the old prefix field, in the other direction.
#   * `func-filters-count` (kidneys) and `organ-count-kidney2` are the
#     same family AND the same underlying fact with two surfaces — the
#     textbook pseudo-replication case.
#   * `super-populous-capital` joins country_capital and
#     `super-smallest-continent` joins country_continent: the first hop
#     differs (superlative vs city) but the retrieved fact and answer set
#     are identical, and difficulty is carried by the second hop.
#   * A first hop that changes the *relation* does split: `emblem_country`
#     (national emblem -> country) is not `person_country` (person ->
#     country) even though both answer with a country.
#   * Singleton families are kept as singletons. They are honest: one
#     independent cluster contributing one item.
#
# Usage:  python -m jspace_part2.family_authoring        # writes the JSON
# Output: data/probe_swap_family_map.json  (consumed by family.py)
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
OUT = PKG_ROOT / "data" / "probe_swap_family_map.json"

# ---------------------------------------------------------------------
# probe-swap (jacobian-lens data/experiments/probe-swap.json), all 90.
# item name -> (canonical_family, template_id, frame)
# `frame` is the authored surface template with the varying slot masked;
# its sha256 is the template_hash used by the map's self-test.
PROBE_SWAP: dict[str, tuple[str, str, str]] = {
    # --- country-level facts reached through a bridge entity -----------
    "ex-city-capital-Barcelona-Toronto": ("country_capital", "capital_of_country_where_city", "The capital of the country where {city} is located is"),
    "ex-city-capital-Lyon-Naples":       ("country_capital", "capital_of_country_where_city", "The capital of the country where {city} is located is"),
    "ex-city-capital-Naples-Barcelona":  ("country_capital", "capital_of_country_where_city", "The capital of the country where {city} is located is"),
    "ex-city-capital-Toronto-Lyon":      ("country_capital", "capital_of_country_where_city", "The capital of the country where {city} is located is"),
    "ex2-city-capital-Munich":           ("country_capital", "capital_of_country_where_city", "The capital of the country where {city} is located is"),
    "ex2-city-capital-Osaka":            ("country_capital", "capital_of_country_where_city", "The capital of the country where {city} is located is"),
    "ex2-language-capital-Greek":        ("country_capital", "capital_of_country_where_language", "The capital of the country where {language} is the primary language is"),
    "ex2-language-capital-Hungarian":    ("country_capital", "capital_of_country_where_language", "The capital of the country where {language} is the primary language is"),
    "ex2-language-capital-Polish":       ("country_capital", "capital_of_country_where_language", "The capital of the country where {language} is the primary language is"),
    "ex2-language-capital-Swedish":      ("country_capital", "capital_of_country_where_language", "The capital of the country where {language} is the primary language is"),
    "ex2-river-capital-Thames":          ("country_capital", "capital_of_country_where_river", "The capital of the country where the {river} River reaches the sea is"),
    "super-populous-capital":            ("country_capital", "capital_of_superlative_country", "The capital city of the {superlative} country in the world is"),

    "ex-city-language-Lyon-Naples": ("country_language", "language_of_country_where_city", "The language spoken in the country where {city} is located is"),
    "ex2-city-language-Cairo":      ("country_language", "language_of_country_where_city", "The language spoken in the country where {city} is located is"),
    "ex2-city-language-Moscow":     ("country_language", "language_of_country_where_city", "The language spoken in the country where {city} is located is"),
    "amazon-language":              ("country_language", "language_of_country_where_river", "The language spoken in the country where the {river} River ends is"),

    "ex-city-continent-Toronto-Lyon": ("country_continent", "continent_of_country_where_city", "The continent of the country where {city} is located is"),
    "ex2-city-continent-Lima":        ("country_continent", "continent_of_country_where_city", "The continent of the country where {city} is located is"),
    "ex2-city-continent-Sydney":      ("country_continent", "continent_of_country_where_city", "The continent of the country where {city} is located is"),
    "paper-continent":                ("country_continent", "continent_of_country_that_invented", "The continent where the country that invented {invention} is located is"),
    "super-smallest-continent":       ("country_continent", "continent_of_superlative_country", "The {superlative} country in the world is located on the continent of"),

    "ex-city-currency-Toronto-Beijing": ("country_currency", "currency_of_country_where_city", "The currency used in the country where {city} is located is the"),
    "ex-city-currency-Toronto-Mumbai":  ("country_currency", "currency_of_country_where_city", "The currency used in the country where {city} is located is the"),
    "colosseum-currency":               ("country_currency", "currency_of_country_where_landmark", "The currency used in the country where the {landmark} stands is the"),

    "greatwall-ocean":    ("country_ocean", "ocean_adjacent_to_country", "The ocean east of the country that built the {landmark} is the"),
    "rhyme-rain-neighbor": ("country_neighbor", "neighbor_of_country_by_rhyme", "The European country whose name rhymes with '{word}' shares its western border with"),

    # --- element / chemistry -------------------------------------------
    "atomic-80-state":      ("element_state", "state_of_element_by_Z", "The state of matter at room temperature of the element with atomic number {Z} is"),
    "ex-element-state-26-8":  ("element_state", "state_of_element_by_Z", "The state of matter at room temperature of the element with atomic number {Z} is"),
    "ex-element-state-26-80": ("element_state", "state_of_element_by_Z", "The state of matter at room temperature of the element with atomic number {Z} is"),
    "ex-element-state-8-26":  ("element_state", "state_of_element_by_Z", "The state of matter at room temperature of the element with atomic number {Z} is"),
    "ex-element-state-8-80":  ("element_state", "state_of_element_by_Z", "The state of matter at room temperature of the element with atomic number {Z} is"),
    "ex-element-state-80-8":  ("element_state", "state_of_element_by_Z", "The state of matter at room temperature of the element with atomic number {Z} is"),

    "ex-element-symbol-11-26": ("element_symbol", "symbol_of_element_by_Z", "The chemical symbol for the element with atomic number {Z} is"),
    "ex-element-symbol-26-79": ("element_symbol", "symbol_of_element_by_Z", "The chemical symbol for the element with atomic number {Z} is"),

    "chem-atmosphere-Z":     ("element_atomic_number", "Z_of_element_by_role", "The atomic number of the {description} is"),
    "chem-bones-Z":          ("element_atomic_number", "Z_of_element_by_role", "The atomic number of the {description} is"),
    "chem-organic-Z":        ("element_atomic_number", "Z_of_element_by_role", "The atomic number of the {description} is"),
    "chem-photosynthesis-Z": ("element_atomic_number", "Z_of_element_by_role", "The atomic number of the {description} is"),

    # --- colour of an entity (one relation, one answer vocabulary) ------
    "bird-color-swan":     ("entity_color", "color_of_entity", "The color of the {entity} is"),
    "element-color-gold2": ("entity_color", "color_of_entity", "The color of the {entity} is"),
    "ex-planet-color-third-fourth": ("entity_color", "color_of_entity", "The color of the {entity} is"),
    "mars-color":          ("entity_color", "color_of_entity", "The color of the {entity} is"),
    "gem-color-ruby":      ("entity_color", "color_of_entity", "The color of the {entity} is"),

    # --- people ---------------------------------------------------------
    "person-firstname-darwin":      ("person_firstname", "firstname_of_person_by_deed", "The first name of the {description} was"),
    "person-firstname-einstein":    ("person_firstname", "firstname_of_person_by_deed", "The first name of the {description} was"),
    "person-firstname-mozart":      ("person_firstname", "firstname_of_person_by_deed", "The first name of the {description} was"),
    "person-firstname-newton":      ("person_firstname", "firstname_of_person_by_deed", "The first name of the {description} was"),
    "person-firstname-shakespeare": ("person_firstname", "firstname_of_person_by_deed", "The first name of the {description} was"),
    "person-country-napoleon":      ("person_country", "country_of_person_by_deed", "The country {relation} the {description} is"),
    "person-country-shakespeare":   ("person_country", "country_of_person_by_deed", "The country {relation} the {description} is"),
    "person-century-lincoln":       ("person_war", "war_led_by_person", "The war that the {description} led his country through was the"),

    # --- anatomy --------------------------------------------------------
    "func-filters-count":  ("organ_count", "count_of_organ_by_function", "In humans, the number of organs that {function} is"),
    "organ-count-kidney2": ("organ_count", "count_of_organ_by_shape_function", "The number of the {description} organs in a human body is"),
    "func-pumps-chambers": ("organ_chamber_count", "chamber_count_of_organ", "In humans, the organ that {function} has this many chambers:"),
    "organ-location-brain": ("organ_location", "location_of_organ", "The {body structure} that contains/encases the organ that {function} is"),
    "organ-location-heart": ("organ_location", "location_of_organ", "The {body structure} that contains/encases the organ that {function} is"),
    "organ-acid-stomach":   ("organ_substance", "substance_in_organ", "The strong liquid inside the organ that {function} is called"),

    # --- animals --------------------------------------------------------
    "animal-legs-buffalo2": ("animal_leg_count", "leg_count_of_animal", "The number of legs on the {description} is"),
    "spider-legs":          ("animal_leg_count", "leg_count_of_animal", "The number of legs on the {description} is"),
    "animal-cover-turtle":  ("animal_bodypart", "bodypart_name_of_animal", "The {body part description} of the {animal description} is called a"),
    "animal-nose-elephant": ("animal_bodypart", "bodypart_name_of_animal", "The {body part description} of the {animal description} is called a"),
    "bird-time-owl":        ("animal_activity_time", "activity_time_of_animal", "The time of day when the {animal description} hunts is"),
    "bird-country-eagle":   ("emblem_country", "country_of_national_emblem", "The country whose national emblem is the {emblem description} is"),
    "rhyme-chair-flag":     ("animal_state_flag", "state_flag_animal_by_rhyme", "The {animal} whose name rhymes with '{word}' appears on the state flag of"),

    # --- products and sources -------------------------------------------
    "beverage-source-wine": ("product_source_organism", "source_of_product", "The {organism type} that produces/yields the {product description} is the"),
    "food-animal-butter":   ("product_source_organism", "source_of_product", "The {organism type} that produces/yields the {product description} is the"),
    "food-animal-honey":    ("product_source_organism", "source_of_product", "The {organism type} that produces/yields the {product description} is the"),
    "gem-source-pearl":     ("product_source_organism", "source_of_product", "The {organism type} that produces/yields the {product description} is the"),
    "tree-product-oak":     ("tree_product", "product_of_tree", "The {product description} produced by the {tree description} is the"),
    "fruit-grows-grape":    ("plant_growth_form", "growth_form_of_fruit", "The kind of plant that the {fruit description} grows on is a"),

    # --- calendar / ordinals ---------------------------------------------
    "birthstone-emerald-month": ("month_ordinal", "ordinal_of_month", "{description} is month number"),
    "etym-wargod-month":        ("month_ordinal", "ordinal_of_month", "{description} is month number"),
    "etym-frigg-position":      ("weekday_ordinal", "ordinal_of_weekday", "Counting Monday as day 1, the day of the week {description} is day number"),
    "etym-saturn-position":     ("planet_ordinal", "ordinal_of_planet", "The planet {description} is planet number"),
    "holiday-month-christmas2": ("holiday_month", "month_of_holiday", "The month of the holiday {description} is"),
    "christmas-season":         ("holiday_season", "season_of_holiday", "The season when the holiday {description} occurs is"),
    "season-next-winter":       ("season_next", "season_after_season", "The season that comes immediately after the {description} season is"),
    "month-3-godof":            ("month_namesake", "namesake_domain_of_month", "The {ordinal} month of the year is named after the Roman god of"),

    # --- astronomy --------------------------------------------------------
    "planet-3-moons":     ("planet_moon_count", "moon_count_of_planet", "The number of natural moons orbiting the planet {description} is"),
    "planet-rings-saturn2": ("planet_feature", "feature_of_planet", "The most famous feature of the planet {description} is its"),
    "rhyme-spoon-orbit":  ("orbit_parent", "orbited_body_by_rhyme", "The celestial body whose name rhymes with '{word}' orbits the planet called"),

    # --- sport, music, geography, vehicles ---------------------------------
    "basketball-players":  ("sport_team_size", "team_size_of_sport", "The number of players per side in the sport {description} is"),
    "sport-equip-tennis":  ("sport_equipment", "equipment_of_sport", "The piece of equipment used to strike the ball in the sport {description} is a"),
    "osu-rival-mascot":    ("rival_mascot", "mascot_of_rival_team", "The mascot of the {description} rival of {team} is a"),
    "instr-body-trumpet":  ("instrument_play", "play_method_of_instrument", "The {body part or object} used to play the instrument {description} is"),
    "instr-hit-drums":     ("instrument_play", "play_method_of_instrument", "The {body part or object} used to play the instrument {description} is"),
    "violin-strings":      ("instrument_string_count", "string_count_of_instrument", "The number of strings on the instrument {description} is"),
    "city-state-Philadelphia": ("city_state", "us_state_of_city", "The US state where the city {description} is located is"),
    "spaceneedle-border":  ("state_border", "border_country_of_us_state", "The US state where {landmark} is located shares its northern border with"),
    "vehicle-power-bicycle": ("vehicle_power", "power_source_of_vehicle", "The source of power for the vehicle {description} is"),
}

# ---------------------------------------------------------------------
# one-hop battery (battery.ONEHOP, 30 items, indexed by position).
# The old field made every item its own family; ten of them are one
# template ("The capital of {country} is"), which is the same
# pseudo-replication in the control task.
ONEHOP: dict[int, tuple[str, str, str]] = {
    0: ("country_capital_direct", "capital_of_country", "The capital of {country} is"),
    1: ("country_capital_direct", "capital_of_country", "The capital of {country} is"),
    2: ("country_capital_direct", "capital_of_country", "The capital of {country} is"),
    3: ("country_capital_direct", "capital_of_country", "The capital of {country} is"),
    4: ("country_capital_direct", "capital_of_country", "The capital of {country} is"),
    5: ("country_capital_direct", "capital_of_country", "The capital of {country} is"),
    6: ("country_capital_direct", "capital_of_country", "The capital of {country} is"),
    7: ("country_capital_direct", "capital_of_country", "The capital of {country} is"),
    8: ("country_capital_direct", "capital_of_country", "The capital of {country} is"),
    9: ("country_capital_direct", "capital_of_country", "The capital of {country} is"),
    10: ("superlative_body", "superlative_solar_system_body", "The {superlative} in our solar system is"),
    11: ("element_symbol_direct", "symbol_of_element_by_name", "The chemical symbol for {element} is"),
    12: ("work_author", "author_of_work", "The author of {work} is"),
    13: ("country_language_direct", "language_of_country", "The language spoken in {country} is"),
    14: ("temperature_scale", "scale_of_freezing_point", "Water freezes at zero degrees"),
    15: ("word_antonym", "antonym_of_word", "The opposite of {word} is"),
    16: ("animal_leg_count_direct", "leg_count_of_animal_direct", "A {animal} has {n}"),
    17: ("entity_color_direct", "color_of_entity_direct", "The color of {entity} is"),
    18: ("month_name_ordinal", "month_by_ordinal", "The {ordinal} month of the year is"),
    19: ("time_unit_count", "count_within_time_unit", "The number of {unit} in a {larger unit} is"),
    20: ("superlative_body", "superlative_solar_system_body", "The {superlative} in our solar system is"),
    21: ("ocean_between", "ocean_between_continents", "The ocean between {A} and {B} is the"),
    22: ("country_currency_direct", "currency_of_country", "The currency of {country} is the"),
    23: ("superlative_animal", "superlative_animal", "The {superlative} animal is the"),
    24: ("substance_phase_name", "name_of_phase_of_substance", "The {phase} form of {substance} is called"),
    25: ("organ_function_direct", "organ_by_function", "The organ that {function} is the"),
    26: ("planet_by_epithet", "planet_by_epithet", "The planet known as {epithet} is"),
    27: ("superlative_animal", "superlative_animal", "The {superlative} animal is the"),
    28: ("animal_offspring", "offspring_name_of_animal", "A baby {animal} is called a"),
    29: ("season_next_direct", "season_after_season_direct", "The season after {season} is"),
}


def template_hash(frame: str) -> str:
    return hashlib.sha256(" ".join(frame.lower().split()).encode()).hexdigest()[:16]


def build() -> dict:
    rows = []
    for name, (fam, tid, frame) in PROBE_SWAP.items():
        rows.append({"item_id": f"twohop:{name}", "pool": "probe_swap",
                     "item_name": name, "canonical_family": fam,
                     "template_id": tid, "template_frame": frame,
                     "template_hash": template_hash(frame)})
    for i, (fam, tid, frame) in ONEHOP.items():
        rows.append({"item_id": f"onehop:{i}", "pool": "battery_onehop",
                     "item_name": f"onehop{i}", "canonical_family": fam,
                     "template_id": tid, "template_frame": frame,
                     "template_hash": template_hash(frame)})
    fams = sorted({r["canonical_family"] for r in rows})
    return {
        "schema_version": 1,
        "rule": ("canonical_family = (second-hop relation, answer type): two "
                 "items share a family iff answering both requires the same "
                 "relation over the same answer vocabulary. template_id is the "
                 "finer surface frame; template_hash tests the map."),
        "supersedes": "battery.py `name.split('-')[0]` and `onehop{i}`",
        "n_items": len(rows), "n_families": len(fams), "families": fams,
        "items": sorted(rows, key=lambda r: r["item_id"]),
    }


if __name__ == "__main__":
    m = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT}: {m['n_items']} items, {m['n_families']} canonical families")
