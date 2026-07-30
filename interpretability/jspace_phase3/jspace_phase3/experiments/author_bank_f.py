# Bank F authoring — first tranche (nextsteps §5.2–5.4, addendum §4.2).
#
# LLM-proposed, SOURCE-VERIFIED: every bundle names the wikipedia
# 20231101.en pages (the box's pinned snapshot — dump date in the config
# name) that state its two hops; verify_against_wikipedia() checks the
# bridge appears on the source-hop page and the answer on the bridge
# page. The LLM is not the factual authority — bundles failing
# verification are QUARANTINED into the report, not shipped.
#
# Design constraints honored here:
#   * cross-phase dedup: no (bridge, answer) pair — and no bare answer —
#     from the Phase 2 manifest (bank.phase2_triple_keys, incl.
#     answer-only keys, so e.g. 'Portuguese', 'euro', 'Paris' are burned);
#   * first hops are FUNCTIONAL relations (capital-of, mouth-of, HQ-of,
#     author-of, ...) — the §4.2b ambiguity pass is a per-family
#     ambiguity_note stating why exactly one bridge satisfies the prompt;
#   * one (bridge, answer) pair per bank; distinct surface templates per
#     family; prompts end at a scoring boundary; counterfactual = a
#     rotated sibling instance (same family, different bridge+answer).
#
# Usage: python -m jspace_phase3.experiments.author_bank_f
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from ..bank import FactBundle, phase2_triple_keys, save_bank, validate_bank
from ..paths3 import resolve_uri
from ..provenance3 import (Provenance3, register, require_clean_tree,
                           write_result3)

EVIDENCE_ID = "p3-bank-f-v5"
SUPERSEDES = "p3-bank-f-v4"  # v1's verifier used a dataset-level
# isin filter that silently missed rows over DriveFS, quarantining 66
# bundles whose pages exist; v2 streams every shard. Three authoring
# leaks also fixed (pad thai/Thai, Fiat/Fiat 500, Al/metAL).
TIER = "phase3-development"
REPO_DATA = Path(__file__).resolve().parents[2] / "data"
P2_MANIFEST_URI = "drive://metrics/cross_model/g5_item_manifest_v5.json"

# instance = (source, bridge, answer, aliases, {source_page, bridge_page})
FAMILIES = [
    dict(
        family="capital_to_currency", group="geo_political",
        templates=dict(
            direct="The official currency of {bridge} is",
            composed="The official currency of the country whose capital "
                     "city is {source} is",
            bridge_supplied="{source} is the capital of {bridge}. The "
                            "official currency of {bridge} is"),
        ambiguity="capital-of is functional: exactly one country has this "
                  "capital city",
        instances=[
            ("Bangkok", "Thailand", "baht", [" the baht", " baht"],
             dict(source_page="Bangkok", bridge_page="Thailand")),
            ("Dhaka", "Bangladesh", "taka", [" the taka", " taka"],
             dict(source_page="Dhaka", bridge_page="Bangladesh")),
            ("Accra", "Ghana", "cedi", [" the cedi", " cedi"],
             dict(source_page="Accra", bridge_page="Ghana")),
            ("Abuja", "Nigeria", "naira", [" the naira", " naira"],
             dict(source_page="Abuja", bridge_page="Nigeria")),
            ("Budapest", "Hungary", "forint", [" the forint", " forint"],
             dict(source_page="Budapest", bridge_page="Hungary")),
            ("Warsaw", "Poland", "złoty",
             [" the złoty", " złoty", " the zloty", " zloty"],
             dict(source_page="Warsaw", bridge_page="Poland")),
        ]),
    dict(
        family="capital_to_language", group="geo_political",
        templates=dict(
            direct="The official language of {bridge} is",
            composed="The official language of the country that has its "
                     "capital at {source} is",
            bridge_supplied="{source} is the capital of {bridge}. The "
                            "official language of {bridge} is"),
        ambiguity="capital-of is functional; chosen countries have a "
                  "single primary official language",
        instances=[
            ("Addis Ababa", "Ethiopia", "Amharic", [" Amharic"],
             dict(source_page="Addis Ababa", bridge_page="Ethiopia")),
            ("Kathmandu", "Nepal", "Nepali", [" Nepali"],
             dict(source_page="Kathmandu", bridge_page="Nepal")),
            ("Ulaanbaatar", "Mongolia", "Mongolian", [" Mongolian"],
             dict(source_page="Ulaanbaatar", bridge_page="Mongolia")),
            ("Hanoi", "Vietnam", "Vietnamese", [" Vietnamese"],
             dict(source_page="Hanoi", bridge_page="Vietnam")),
            ("Manila", "Philippines", "Filipino", [" Filipino", " Tagalog"],
             dict(source_page="Manila", bridge_page="Philippines")),
            ("Dhaka", "Bangladesh", "Bengali", [" Bengali", " Bangla"],
             dict(source_page="Dhaka", bridge_page="Bangladesh")),
        ]),
    dict(
        family="river_mouth_to_capital", group="geo_political",
        templates=dict(
            direct="The capital city of {bridge} is",
            composed="The capital city of the country where the {source} "
                     "River reaches the sea is",
            bridge_supplied="The {source} River reaches the sea in "
                            "{bridge}. The capital city of {bridge} is"),
        ambiguity="each chosen river has its mouth in exactly one country",
        instances=[
            ("Chao Phraya", "Thailand", "Bangkok", [" Bangkok"],
             dict(source_page="Chao Phraya River", bridge_page="Thailand")),
            ("Volta", "Ghana", "Accra", [" Accra"],
             dict(source_page="Volta River", bridge_page="Ghana")),
            ("Magdalena", "Colombia", "Bogota", [" Bogota"],
             dict(source_page="Magdalena River", bridge_page="Colombia")),
            ("Irrawaddy", "Myanmar", "Naypyidaw",
             [" Naypyidaw", " Nay Pyi Taw"],
             dict(source_page="Irrawaddy River", bridge_page="Myanmar")),
            ("Tagus", "Portugal", "Lisbon", [" Lisbon"],
             dict(source_page="Tagus", bridge_page="Portugal")),
        ]),
    dict(
        family="landmark_city_to_country", group="geo_political",
        templates=dict(
            direct="The city of {bridge} is located in the country of",
            composed="The city where the {source} stands is located in "
                     "the country of",
            bridge_supplied="The {source} stands in {bridge}. That city "
                            "is located in the country of"),
        ambiguity="each landmark stands in exactly one city",
        instances=[
            ("Petronas Towers", "Kuala Lumpur", "Malaysia", [" Malaysia"],
             dict(source_page="Petronas Towers", bridge_page="Kuala Lumpur")),
            ("Hagia Sophia", "Istanbul", "Turkey", [" Turkey"],
             dict(source_page="Hagia Sophia", bridge_page="Istanbul")),
            ("Burj Khalifa", "Dubai", "the United Arab Emirates",
             [" the United Arab Emirates", " United Arab Emirates",
              " the UAE", " UAE"],
             dict(source_page="Burj Khalifa", bridge_page="Dubai")),
            ("Table Mountain", "Cape Town", "South Africa",
             [" South Africa"],
             dict(source_page="Table Mountain", bridge_page="Cape Town")),
            ("Wat Arun", "Bangkok", "Thailand", [" Thailand"],
             dict(source_page="Wat Arun", bridge_page="Bangkok")),
        ]),
    dict(
        family="company_hq_to_country", group="org",
        templates=dict(
            direct="{bridge} is a city in the country of",
            composed="The company {source} has its headquarters in a city "
                     "in the country of",
            bridge_supplied="The company {source} is headquartered in "
                            "{bridge}, a city in the country of"),
        ambiguity="corporate global headquarters is a single city; "
                  "verified against the snapshot's infobox statements",
        instances=[
            ("Nokia", "Espoo", "Finland", [" Finland"],
             dict(source_page="Nokia", bridge_page="Espoo")),
            ("Spotify", "Stockholm", "Sweden", [" Sweden"],
             dict(source_page="Spotify", bridge_page="Stockholm")),
            ("Lego", "Billund", "Denmark", [" Denmark"],
             dict(source_page="Lego", bridge_page="Billund")),
            ("Samsung Electronics", "Suwon", "South Korea",
             [" South Korea"],
             dict(source_page="Samsung Electronics", bridge_page="Suwon")),
            ("Philips", "Amsterdam", "the Netherlands",
             [" the Netherlands", " Netherlands"],
             dict(source_page="Philips", bridge_page="Amsterdam")),
        ]),
    dict(
        family="island_to_capital", group="geo_political",
        templates=dict(
            direct="The capital of {bridge} is the city of",
            composed="The capital of the country that the island of "
                     "{source} belongs to is the city of",
            bridge_supplied="The island of {source} belongs to {bridge}. "
                            "The capital of {bridge} is the city of"),
        ambiguity="each chosen island belongs to exactly one sovereign "
                  "country",
        instances=[
            ("Bali", "Indonesia", "Jakarta", [" Jakarta"],
             dict(source_page="Bali", bridge_page="Indonesia")),
            ("Zanzibar", "Tanzania", "Dodoma", [" Dodoma"],
             dict(source_page="Zanzibar", bridge_page="Tanzania")),
            ("Luzon", "the Philippines", "Manila", [" Manila"],
             dict(source_page="Luzon", bridge_page="Philippines")),
            ("Tasmania", "Australia", "Canberra", [" Canberra"],
             dict(source_page="Tasmania", bridge_page="Australia")),
        ]),
    dict(
        family="currency_to_capital", group="geo_political",
        templates=dict(
            direct="The seat of government of {bridge} is",
            composed="The seat of government of the country whose "
                     "currency is the {source} is",
            bridge_supplied="The {source} is the currency of {bridge}. "
                            "The seat of government of {bridge} is"),
        ambiguity="each chosen currency is issued by exactly one country",
        instances=[
            ("ringgit", "Malaysia", "Kuala Lumpur", [" Kuala Lumpur"],
             dict(source_page="Malaysian ringgit", bridge_page="Malaysia")),
            ("won", "South Korea", "Seoul", [" Seoul"],
             dict(source_page="South Korean won", bridge_page="South Korea")),
            ("rand", "South Africa", "Pretoria", [" Pretoria"],
             dict(source_page="South African rand",
                  bridge_page="South Africa")),
            ("birr", "Ethiopia", "Addis Ababa", [" Addis Ababa"],
             dict(source_page="Ethiopian birr", bridge_page="Ethiopia")),
        ]),
    dict(
        family="novel_author_to_birth_country", group="person_culture",
        templates=dict(
            direct="The writer {bridge} was born in the country of",
            composed="The author of the novel {source} was born in the "
                     "country of",
            bridge_supplied="The novel {source} was written by {bridge}, "
                            "who was born in the country of"),
        ambiguity="each novel has a single author; birth country per the "
                  "snapshot's biography lead",
        instances=[
            ("Things Fall Apart", "Chinua Achebe", "Nigeria", [" Nigeria"],
             dict(source_page="Things Fall Apart",
                  bridge_page="Chinua Achebe")),
            ("My Name Is Red", "Orhan Pamuk", "Turkey", [" Turkey"],
             dict(source_page="My Name Is Red", bridge_page="Orhan Pamuk")),
            ("Midnight's Children", "Salman Rushdie", "India", [" India"],
             dict(source_page="Midnight's Children",
                  bridge_page="Salman Rushdie")),
            ("Wide Sargasso Sea", "Jean Rhys", "Dominica", [" Dominica"],
             dict(source_page="Wide Sargasso Sea", bridge_page="Jean Rhys")),
            ("My Brilliant Friend", "Elena Ferrante", "Italy", [" Italy"],
             dict(source_page="My Brilliant Friend",
                  bridge_page="Elena Ferrante")),
        ]),
    dict(
        family="opera_composer_to_nationality", group="person_culture",
        templates=dict(
            direct="The composer {bridge} was by nationality",
            composed="The composer of the opera {source} was by "
                     "nationality",
            bridge_supplied="The opera {source} was composed by {bridge}, "
                            "who was by nationality"),
        ambiguity="each opera has a single composer",
        instances=[
            ("Madama Butterfly", "Giacomo Puccini", "Italian", [" Italian"],
             dict(source_page="Madama Butterfly",
                  bridge_page="Giacomo Puccini")),
            ("The Bartered Bride", "Bedřich Smetana", "Czech", [" Czech"],
             dict(source_page="The Bartered Bride",
                  bridge_page="Bedřich Smetana")),
            ("Bluebeard's Castle", "Béla Bartók", "Hungarian",
             [" Hungarian"],
             dict(source_page="Bluebeard's Castle",
                  bridge_page="Béla Bartók")),
            ("Rusalka", "Antonín Dvořák", "Czech", [" Czech"],
             dict(source_page="Rusalka (opera)",
                  bridge_page="Antonín Dvořák")),
            ("Aida", "Giuseppe Verdi", "Italian", [" Italian"],
             dict(source_page="Aida", bridge_page="Giuseppe Verdi")),
        ]),
    dict(
        family="painting_painter_to_movement", group="person_culture",
        templates=dict(
            direct="The painter {bridge} is associated with the art "
                   "movement known as",
            composed="The painter of {source} is associated with the art "
                     "movement known as",
            bridge_supplied="{source} was painted by {bridge}, who is "
                            "associated with the art movement known as"),
        ambiguity="each painting has a single painter; movement per the "
                  "snapshot's lead attribution",
        instances=[
            ("Impression, Sunrise", "Claude Monet", "Impressionism",
             [" Impressionism"],
             dict(source_page="Impression, Sunrise",
                  bridge_page="Claude Monet")),
            ("Les Demoiselles d'Avignon", "Pablo Picasso", "Cubism",
             [" Cubism"],
             dict(source_page="Les Demoiselles d'Avignon",
                  bridge_page="Pablo Picasso")),
            ("The Starry Night", "Vincent van Gogh", "Post-Impressionism",
             [" Post-Impressionism"],
             dict(source_page="The Starry Night",
                  bridge_page="Vincent van Gogh")),
            ("The Kiss", "Gustav Klimt", "Art Nouveau", [" Art Nouveau"],
             dict(source_page="The Kiss (Klimt)",
                  bridge_page="Gustav Klimt")),
        ]),
    dict(
        family="theory_scientist_to_birth_city", group="person_science",
        templates=dict(
            direct="{bridge} was born in the city of",
            composed="The scientist who proposed {source} was born in the "
                     "city of",
            bridge_supplied="{source} was proposed by {bridge}, who was "
                            "born in the city of"),
        ambiguity="each theory/invention is attributed to a single "
                  "principal figure in the snapshot lead",
        instances=[
            ("the theory of general relativity", "Albert Einstein", "Ulm",
             [" Ulm"],
             dict(source_page="General relativity",
                  bridge_page="Albert Einstein")),
            ("the theory of evolution by natural selection",
             "Charles Darwin", "Shrewsbury", [" Shrewsbury"],
             dict(source_page="Natural selection",
                  bridge_page="Charles Darwin")),
            ("the movable-type printing press", "Johannes Gutenberg",
             "Mainz", [" Mainz"],
             dict(source_page="Printing press",
                  bridge_page="Johannes Gutenberg")),
            ("the telephone", "Alexander Graham Bell", "Edinburgh",
             [" Edinburgh"],
             dict(source_page="Alexander Graham Bell",
                  bridge_page="Alexander Graham Bell")),
            ("the periodic law", "Dmitri Mendeleev", "Tobolsk",
             [" Tobolsk"],
             dict(source_page="Periodic table",
                  bridge_page="Dmitri Mendeleev")),
        ]),
    dict(
        family="film_director_to_birth_country", group="person_culture",
        templates=dict(
            direct="The film director {bridge} was born in the country "
                   "of",
            composed="The director of the film {source} was born in the "
                     "country of",
            bridge_supplied="The film {source} was directed by {bridge}, "
                            "who was born in the country of"),
        ambiguity="each film has a single credited director",
        instances=[
            ("Parasite", "Bong Joon-ho", "South Korea", [" South Korea"],
             dict(source_page="Parasite (2019 film)",
                  bridge_page="Bong Joon-ho")),
            ("8½", "Federico Fellini", "Italy", [" Italy"],
             dict(source_page="8½", bridge_page="Federico Fellini")),
            ("Persona", "Ingmar Bergman", "Sweden", [" Sweden"],
             dict(source_page="Persona (1966 film)",
                  bridge_page="Ingmar Bergman")),
            ("Oldboy", "Park Chan-wook", "South Korea", [" South Korea"],
             dict(source_page="Oldboy (2003 film)",
                  bridge_page="Park Chan-wook")),
        ]),
    dict(
        family="ore_metal_to_symbol", group="science",
        templates=dict(
            direct="On the periodic table, the chemical symbol for "
                   "{bridge} is",
            composed="On the periodic table, the chemical symbol for the "
                     "metal extracted from the ore {source} is",
            bridge_supplied="The ore {source} is the chief source of "
                            "{bridge}. On the periodic table, the chemical "
                            "symbol for {bridge} is"),
        ambiguity="each chosen ore is the principal ore of exactly one "
                  "metal",
        instances=[
            ("cinnabar", "mercury", "Hg", [" Hg"],
             dict(source_page="Cinnabar", bridge_page="Mercury (element)")),
            ("galena", "lead", "Pb", [" Pb"],
             dict(source_page="Galena", bridge_page="Lead")),
            ("sphalerite", "zinc", "Zn", [" Zn"],
             dict(source_page="Sphalerite", bridge_page="Zinc")),
            ("malachite", "copper", "Cu", [" Cu"],
             dict(source_page="Malachite", bridge_page="Copper")),
        ]),
    dict(
        family="invention_inventor_to_nationality", group="person_science",
        templates=dict(
            direct="The inventor {bridge} was by nationality",
            composed="The person who invented {source} was by nationality",
            bridge_supplied="{source} was invented by {bridge}, who was "
                            "by nationality"),
        ambiguity="chosen inventions have a single credited inventor in "
                  "the snapshot lead",
        instances=[
            ("the single-wire telegraph", "Samuel Morse", "American",
             [" American"],
             dict(source_page="Samuel Morse", bridge_page="Samuel Morse")),
            ("dynamite", "Alfred Nobel", "Swedish", [" Swedish"],
             dict(source_page="Dynamite", bridge_page="Alfred Nobel")),
            ("the World Wide Web", "Tim Berners-Lee", "British",
             [" British", " English"],
             dict(source_page="World Wide Web",
                  bridge_page="Tim Berners-Lee")),
            ("the first polio vaccine", "Jonas Salk", "American",
             [" American"],
             dict(source_page="Polio vaccine", bridge_page="Jonas Salk")),
        ]),
    dict(
        family="dish_country_to_language", group="geo_culture",
        templates=dict(
            direct="The national language of {bridge} is",
            composed="The national language of the country whose cuisine "
                     "gave the world {source} is",
            bridge_supplied="{source} comes from {bridge}. The national "
                            "language of {bridge} is"),
        ambiguity="chosen dishes are nationally attributed to one country "
                  "in the snapshot lead",
        instances=[
            ("lasagna", "Italy", "Italian", [" Italian"],
             dict(source_page="Lasagna", bridge_page="Italy")),
            ("goulash", "Hungary", "Hungarian", [" Hungarian"],
             dict(source_page="Goulash", bridge_page="Hungary")),
            ("pierogi", "Poland", "Polish", [" Polish"],
             dict(source_page="Pierogi", bridge_page="Poland")),
            ("kimchi", "South Korea", "Korean", [" Korean"],
             dict(source_page="Kimchi", bridge_page="South Korea")),
        ]),
    dict(
        family="club_stadium_to_city", group="sport",
        templates=dict(
            direct="The stadium {bridge} is found in the city of",
            composed="The home stadium of the football club {source} is "
                     "found in the city of",
            bridge_supplied="The football club {source} plays its home "
                            "games at {bridge}, found in the city of"),
        ambiguity="each club has one current home stadium in the snapshot",
        instances=[
            ("Ajax", "the Johan Cruyff Arena", "Amsterdam", [" Amsterdam"],
             dict(source_page="AFC Ajax", bridge_page="Johan Cruyff Arena")),
            ("Boca Juniors", "La Bombonera", "Buenos Aires",
             [" Buenos Aires"],
             dict(source_page="Boca Juniors", bridge_page="La Bombonera")),
            ("Fenerbahçe", "Şükrü Saracoğlu Stadium", "Istanbul",
             [" Istanbul"],
             dict(source_page="Fenerbahçe S.K. (football)",
                  bridge_page="Şükrü Saracoğlu Stadium")),
            ("Real Betis", "Estadio Benito Villamarín", "Seville",
             [" Seville", " Sevilla"],
             dict(source_page="Real Betis",
                  bridge_page="Estadio Benito Villamarín")),
        ]),
    dict(
        family="poem_poet_to_nationality", group="person_culture",
        templates=dict(
            direct="The poet {bridge} was by nationality",
            composed="The poet who wrote {source} was by nationality",
            bridge_supplied="{source} was written by the poet {bridge}, "
                            "who was by nationality"),
        ambiguity="each poem has a single author",
        instances=[
            ("The Raven", "Edgar Allan Poe", "American", [" American"],
             dict(source_page="The Raven",
                  bridge_page="Edgar Allan Poe")),
            ("Ozymandias", "Percy Bysshe Shelley", "English",
             [" English", " British"],
             dict(source_page="Ozymandias",
                  bridge_page="Percy Bysshe Shelley")),
            ("Do not go gentle into that good night", "Dylan Thomas",
             "Welsh", [" Welsh"],
             dict(source_page="Do not go gentle into that good night",
                  bridge_page="Dylan Thomas")),
            ("Gunga Din", "Rudyard Kipling", "English",
             [" English", " British"],
             dict(source_page="Gunga Din", bridge_page="Rudyard Kipling")),
        ]),
    dict(
        family="language_creator_to_nationality", group="tech",
        templates=dict(
            direct="The programmer {bridge} is by nationality",
            composed="The creator of the {source} programming language is "
                     "by nationality",
            bridge_supplied="The {source} programming language was created "
                            "by {bridge}, who is by nationality"),
        ambiguity="each chosen language has a single principal designer",
        instances=[
            ("Python", "Guido van Rossum", "Dutch", [" Dutch"],
             dict(source_page="Python (programming language)",
                  bridge_page="Guido van Rossum")),
            ("C++", "Bjarne Stroustrup", "Danish", [" Danish"],
             dict(source_page="C++", bridge_page="Bjarne Stroustrup")),
            ("Pascal", "Niklaus Wirth", "Swiss", [" Swiss"],
             dict(source_page="Pascal (programming language)",
                  bridge_page="Niklaus Wirth")),
            ("JavaScript", "Brendan Eich", "American", [" American"],
             dict(source_page="JavaScript", bridge_page="Brendan Eich")),
        ]),
    dict(
        family="product_company_to_hq_city", group="tech",
        templates=dict(
            direct="The company {bridge} has its headquarters in the city "
                   "of",
            composed="The company that makes {source} has its "
                     "headquarters in the city of",
            bridge_supplied="{source} is made by {bridge}, which has its "
                            "headquarters in the city of"),
        ambiguity="each product names its maker in the snapshot lead; "
                  "one global HQ city",
        instances=[
            ("the Android operating system", "Google", "Mountain View",
             [" Mountain View"],
             dict(source_page="Android (operating system)",
                  bridge_page="Google")),
            ("the Windows operating system", "Microsoft", "Redmond",
             [" Redmond"],
             dict(source_page="Microsoft Windows",
                  bridge_page="Microsoft")),
            ("the iPhone", "Apple", "Cupertino", [" Cupertino"],
             dict(source_page="IPhone", bridge_page="Apple Inc.")),
            ("Photoshop", "Adobe", "San Jose", [" San Jose"],
             dict(source_page="Adobe Photoshop", bridge_page="Adobe Inc.")),
        ]),
    dict(
        family="car_model_maker_to_hq_city", group="org",
        templates=dict(
            direct="The automaker {bridge} is based in the city of",
            composed="The automaker that builds the {source} is based in "
                     "the city of",
            bridge_supplied="The {source} is built by {bridge}, which is "
                            "based in the city of"),
        ambiguity="each model names its maker; one corporate seat in the "
                  "snapshot infobox",
        instances=[
            ("Corolla", "Toyota", "Toyota City", [" Toyota City"],
             dict(source_page="Toyota Corolla", bridge_page="Toyota")),
            ("Golf", "Volkswagen", "Wolfsburg", [" Wolfsburg"],
             dict(source_page="Volkswagen Golf",
                  bridge_page="Volkswagen")),
            ("Model S", "Tesla", "Austin", [" Austin"],
             dict(source_page="Tesla Model S", bridge_page="Tesla, Inc.")),
            ("Mustang", "Ford", "Dearborn", [" Dearborn"],
             dict(source_page="Ford Mustang",
                  bridge_page="Ford Motor Company")),
        ]),
]


FAMILIES += [
    dict(
        family="landmark_city_to_river", group="geo_physical",
        templates=dict(
            direct="The river flowing through the city of {bridge} is "
                   "the",
            composed="The river flowing through the city where the "
                     "{source} stands is the",
            bridge_supplied="The {source} stands in {bridge}, and the "
                            "river flowing through that city is the"),
        ambiguity="each landmark stands in one city; each chosen city "
                  "has one principal river",
        instances=[
            ("Charles Bridge", "Prague", "Vltava", [" Vltava"],
             dict(source_page="Charles Bridge", bridge_page="Prague")),
            ("Colosseum", "Rome", "Tiber", [" Tiber"],
             dict(source_page="Colosseum", bridge_page="Rome")),
            ("Big Ben", "London", "Thames", [" Thames", " River Thames"],
             dict(source_page="Big Ben", bridge_page="London")),
            ("Louvre", "Paris", "Seine", [" Seine"],
             dict(source_page="Louvre", bridge_page="Paris")),
            ("Brandenburg Gate", "Berlin", "Spree", [" Spree"],
             dict(source_page="Brandenburg Gate", bridge_page="Berlin")),
        ]),
    dict(
        family="flag_carrier_to_currency", group="org",
        templates=dict(
            direct="The currency used in {bridge} is the",
            composed="The currency used in the country whose flag "
                     "carrier airline is {source} is the",
            bridge_supplied="{source} is the flag carrier of {bridge}, "
                            "where the currency used is the"),
        ambiguity="flag-carrier is functional for the chosen airlines "
                  "(one home country each)",
        instances=[
            ("Emirates", "the United Arab Emirates", "dirham",
             [" dirham", " UAE dirham"],
             dict(source_page="Emirates (airline)",
                  bridge_page="United Arab Emirates")),
            ("Garuda", "Indonesia", "rupiah", [" rupiah"],
             dict(source_page="Garuda Indonesia",
                  bridge_page="Indonesia")),
            ("Qantas", "Australia", "Australian dollar",
             [" the Australian dollar", " Australian dollar", " AUD"],
             dict(source_page="Qantas", bridge_page="Australia")),
            ("Aeroflot", "Russia", "ruble", [" ruble", " rouble"],
             dict(source_page="Aeroflot", bridge_page="Russia")),
        ]),
    dict(
        family="highest_peak_to_currency", group="geo_physical",
        templates=dict(
            direct="Shoppers in {bridge} pay in the currency called the",
            composed="Shoppers in the country whose highest peak is "
                     "{source} pay in the currency called the",
            bridge_supplied="{source} is the highest peak of {bridge}, "
                            "where shoppers pay in the currency called "
                            "the"),
        ambiguity="highest-peak-of-country is functional; peaks chosen "
                  "lie entirely within one country",
        instances=[
            ("Kilimanjaro", "Tanzania", "shilling",
             [" shilling", " Tanzanian shilling"],
             dict(source_page="Mount Kilimanjaro", bridge_page="Tanzania")),
            ("Aconcagua", "Argentina", "peso",
             [" peso", " Argentine peso"],
             dict(source_page="Aconcagua", bridge_page="Argentina")),
            ("Mount Fuji", "Japan", "yen", [" yen"],
             dict(source_page="Mount Fuji", bridge_page="Japan")),
            ("Ben Nevis", "the United Kingdom", "pound sterling",
             [" pound sterling", " British pound"],
             dict(source_page="Ben Nevis", bridge_page="United Kingdom")),
        ]),
    dict(
        family="character_creator_to_birth_city", group="person_culture",
        templates=dict(
            direct="The author {bridge} was born in the town of",
            composed="The author who created the character {source} was "
                     "born in the town of",
            bridge_supplied="The character {source} was created by "
                            "{bridge}, who was born in the town of"),
        ambiguity="each character has one credited creator",
        instances=[
            ("Sherlock Holmes", "Arthur Conan Doyle", "Edinburgh",
             [" Edinburgh"],
             dict(source_page="Sherlock Holmes",
                  bridge_page="Arthur Conan Doyle")),
            ("Hercule Poirot", "Agatha Christie", "Torquay", [" Torquay"],
             dict(source_page="Hercule Poirot",
                  bridge_page="Agatha Christie")),
            ("the Moomins", "Tove Jansson", "Helsinki", [" Helsinki"],
             dict(source_page="Moomins", bridge_page="Tove Jansson")),
            ("Pippi Longstocking", "Astrid Lindgren", "Vimmerby",
             [" Vimmerby"],
             dict(source_page="Pippi Longstocking",
                  bridge_page="Astrid Lindgren")),
        ]),
    dict(
        family="subunit_currency_to_country", group="geo_political",
        templates=dict(
            direct="The {bridge} is the currency of the country called",
            composed="The currency whose smallest subunit is the "
                     "{source} belongs to the country called",
            bridge_supplied="The {source} is a subunit of the {bridge}, "
                            "which is the currency of the country called"),
        ambiguity="each subunit name belongs to one currency; each "
                  "currency to one issuing country",
        instances=[
            ("kobo", "naira", "Nigeria", [" Nigeria"],
             dict(source_page="Nigerian naira", bridge_page="Nigerian naira")),
            ("tambala", "kwacha", "Malawi", [" Malawi"],
             dict(source_page="Malawian kwacha",
                  bridge_page="Malawian kwacha")),
            ("chetrum", "ngultrum", "Bhutan", [" Bhutan"],
             dict(source_page="Bhutanese ngultrum",
                  bridge_page="Bhutanese ngultrum")),
            ("avo", "pataca", "Macau", [" Macau", " Macao"],
             dict(source_page="Macanese pataca",
                  bridge_page="Macanese pataca")),
        ]),
    dict(
        family="space_mission_agency_to_country", group="org_science",
        templates=dict(
            direct="The space agency {bridge} belongs to",
            composed="The space agency that operated the {source} "
                     "mission belongs to",
            bridge_supplied="The {source} mission was operated by "
                            "{bridge}, the space agency of"),
        ambiguity="each chosen mission has a single operating agency",
        instances=[
            ("Voyager 1", "NASA", "the United States",
             [" the United States", " the USA"],
             dict(source_page="Voyager 1", bridge_page="NASA")),
            ("Chandrayaan-3", "ISRO", "India", [" India"],
             dict(source_page="Chandrayaan-3", bridge_page="ISRO")),
            ("Tianwen-1", "CNSA", "China", [" China"],
             dict(source_page="Tianwen-1",
                  bridge_page="China National Space Administration")),
            ("Danuri", "KARI", "South Korea", [" South Korea"],
             dict(source_page="Danuri",
                  bridge_page="Korea Aerospace Research Institute")),
        ]),
    dict(
        family="cheese_origin_to_capital", group="geo_culture",
        templates=dict(
            direct="The capital city of {bridge} is called",
            composed="The capital city of the home country of {source} "
                     "cheese is called",
            bridge_supplied="{source} cheese comes from {bridge}, whose "
                            "capital city is called"),
        ambiguity="chosen cheeses have a single country of origin in "
                  "the snapshot lead",
        instances=[
            ("Gouda", "the Netherlands", "Amsterdam", [" Amsterdam"],
             dict(source_page="Gouda cheese", bridge_page="Netherlands")),
            ("Halloumi", "Cyprus", "Nicosia", [" Nicosia"],
             dict(source_page="Halloumi", bridge_page="Cyprus")),
            ("Gruyère", "Switzerland", "Bern", [" Bern"],
             dict(source_page="Gruyère cheese",
                  bridge_page="Switzerland")),
            ("Oaxaca", "Mexico", "Mexico City", [" Mexico City"],
             dict(source_page="Oaxaca cheese", bridge_page="Mexico")),
        ]),
    dict(
        family="game_studio_to_hq_city", group="tech",
        templates=dict(
            direct="The game studio {bridge} is headquartered in",
            composed="The studio behind the video game series {source} "
                     "is headquartered in",
            bridge_supplied="The video game series {source} is made by "
                            "{bridge}, headquartered in"),
        ambiguity="each series names its principal developer in the "
                  "snapshot lead; one HQ city",
        instances=[
            ("Super Mario", "Nintendo", "Kyoto", [" Kyoto"],
             dict(source_page="Super Mario", bridge_page="Nintendo")),
            ("Angry Birds", "Rovio", "Espoo", [" Espoo"],
             dict(source_page="Angry Birds",
                  bridge_page="Rovio Entertainment")),
            ("Half-Life", "Valve", "Bellevue", [" Bellevue"],
             dict(source_page="Half-Life (series)",
                  bridge_page="Valve Corporation")),
            ("Fortnite", "Epic Games", "Cary", [" Cary"],
             dict(source_page="Fortnite", bridge_page="Epic Games")),
        ]),
    dict(
        family="museum_city_to_country", group="geo_culture",
        templates=dict(
            direct="{bridge} is a city of the nation called",
            composed="The museum known as {source} stands in a city of "
                     "the nation called",
            bridge_supplied="The museum known as {source} stands in "
                            "{bridge}, a city of the nation called"),
        ambiguity="each museum stands in one city",
        instances=[
            ("the Uffizi", "Florence", "Italy", [" Italy"],
             dict(source_page="Uffizi", bridge_page="Florence")),
            ("the National Palace Museum", "Taipei", "Taiwan",
             [" Taiwan"],
             dict(source_page="National Palace Museum",
                  bridge_page="Taipei")),
            ("the Belvedere", "Vienna", "Austria", [" Austria"],
             dict(source_page="Belvedere, Vienna",
                  bridge_page="Vienna")),
            ("the Mauritshuis", "The Hague", "the Netherlands",
             [" the Netherlands", " Netherlands"],
             dict(source_page="Mauritshuis", bridge_page="The Hague")),
        ]),
    dict(
        family="national_park_to_capital", group="geo_physical",
        templates=dict(
            direct="Government business in {bridge} is conducted in",
            composed="Government business in the country that contains "
                     "{source} National Park is conducted in",
            bridge_supplied="{source} National Park lies in {bridge}, "
                            "whose government business is conducted in"),
        ambiguity="each chosen park lies entirely within one country",
        instances=[
            ("Chitwan", "Nepal", "Kathmandu", [" Kathmandu"],
             dict(source_page="Chitwan National Park",
                  bridge_page="Nepal")),
            ("Fiordland", "New Zealand", "Wellington", [" Wellington"],
             dict(source_page="Fiordland National Park",
                  bridge_page="New Zealand")),
            ("Ranthambore", "India", "New Delhi", [" New Delhi"],
             dict(source_page="Ranthambore National Park",
                  bridge_page="India")),
            ("Etosha", "Namibia", "Windhoek", [" Windhoek"],
             dict(source_page="Etosha National Park",
                  bridge_page="Namibia")),
        ]),
    dict(
        family="bridge_city_to_country", group="geo_political",
        templates=dict(
            direct="The city of {bridge} belongs to the sovereign state "
                   "of",
            composed="The city crossed by {source} belongs to the "
                     "sovereign state of",
            bridge_supplied="{source} crosses {bridge}, a city belonging "
                            "to the sovereign state of"),
        ambiguity="each chosen bridge structure is in one city",
        instances=[
            ("the Golden Gate Bridge", "San Francisco",
             "the United States",
             [" the United States", " the USA"],
             dict(source_page="Golden Gate Bridge",
                  bridge_page="San Francisco")),
            ("the Rialto Bridge", "Venice", "Italy", [" Italy"],
             dict(source_page="Rialto Bridge", bridge_page="Venice")),
            ("Tower Bridge", "London", "the United Kingdom",
             [" the United Kingdom", " the UK", " England",
              " Great Britain"],
             dict(source_page="Tower Bridge", bridge_page="London")),
            ("the Chain Bridge", "Budapest", "Hungary", [" Hungary"],
             dict(source_page="Széchenyi Chain Bridge",
                  bridge_page="Budapest")),
        ]),
]


FAMILIES += [
    dict(
        family="dam_river_to_sea", group="geo_physical",
        templates=dict(
            direct="The {bridge} empties into",
            composed="The river impounded by the {source} empties into",
            bridge_supplied="The {source} impounds the {bridge}, which "
                            "empties into"),
        ambiguity="each dam impounds one river; each river has one mouth",
        instances=[
            ("Three Gorges Dam", "Yangtze", "the East China Sea",
             [" the East China Sea"],
             dict(source_page="Three Gorges Dam", bridge_page="Yangtze")),
            ("Itaipu Dam", "Paraná River", "the Río de la Plata",
             [" the Río de la Plata", " the Rio de la Plata"],
             dict(source_page="Itaipu Dam", bridge_page="Paraná River")),
            ("Kariba Dam", "Zambezi", "the Indian Ocean",
             [" the Indian Ocean"],
             dict(source_page="Kariba Dam", bridge_page="Zambezi")),
            ("Grand Coulee Dam", "Columbia River", "the Pacific Ocean",
             [" the Pacific Ocean"],
             dict(source_page="Grand Coulee Dam",
                  bridge_page="Columbia River")),
            ("Akosombo Dam", "Volta River", "the Gulf of Guinea",
             [" the Gulf of Guinea"],
             dict(source_page="Akosombo Dam", bridge_page="Volta River")),
        ]),
    dict(
        family="comic_creator_to_nationality", group="person_culture",
        templates=dict(
            direct="The cartoonist {bridge} held the nationality known "
                   "as",
            composed="The cartoonist who created {source} held the "
                     "nationality known as",
            bridge_supplied="{source} was created by {bridge}, whose "
                            "nationality is"),
        ambiguity="each strip has a single credited creator",
        instances=[
            ("Tintin", "Hergé", "Belgian", [" Belgian"],
             dict(source_page="The Adventures of Tintin",
                  bridge_page="Hergé")),
            ("Peanuts", "Charles M. Schulz", "American", [" American"],
             dict(source_page="Peanuts", bridge_page="Charles M. Schulz")),
            ("Mafalda", "Quino", "Argentine",
             [" Argentine", " Argentinian"],
             dict(source_page="Mafalda", bridge_page="Quino")),
            ("Calvin and Hobbes", "Bill Watterson", "American",
             [" American"],
             dict(source_page="Calvin and Hobbes",
                  bridge_page="Bill Watterson")),
        ]),
    dict(
        family="national_animal_to_capital", group="geo_culture",
        templates=dict(
            direct="Diplomats visiting {bridge} arrive in the capital "
                   "city of",
            composed="Diplomats visiting the country whose national "
                     "animal is the {source} arrive in the capital city "
                     "of",
            bridge_supplied="The {source} is the national animal of "
                            "{bridge}, whose capital city is"),
        ambiguity="each chosen animal is the designated national animal "
                  "of exactly one country",
        instances=[
            ("markhor", "Pakistan", "Islamabad", [" Islamabad"],
             dict(source_page="Markhor", bridge_page="Pakistan")),
            ("okapi", "the Democratic Republic of the Congo", "Kinshasa",
             [" Kinshasa"],
             dict(source_page="Okapi",
                  bridge_page="Democratic Republic of the Congo")),
            ("resplendent quetzal", "Guatemala", "Guatemala City",
             [" Guatemala City"],
             dict(source_page="Resplendent quetzal",
                  bridge_page="Guatemala")),
            ("dodo", "Mauritius", "Port Louis", [" Port Louis"],
             dict(source_page="Dodo", bridge_page="Mauritius")),
        ]),
    dict(
        family="skyscraper_city_to_country", group="geo_political",
        templates=dict(
            direct="{bridge} rises within the borders of",
            composed="The skyscraper called {source} rises in a city "
                     "within the borders of",
            bridge_supplied="The skyscraper called {source} rises in "
                            "{bridge}, within the borders of"),
        ambiguity="each tower stands in one city",
        instances=[
            ("Lotte World Tower", "Seoul", "South Korea",
             [" South Korea"],
             dict(source_page="Lotte World Tower", bridge_page="Seoul")),
            ("the Kingdom Centre", "Riyadh", "Saudi Arabia",
             [" Saudi Arabia"],
             dict(source_page="Kingdom Centre", bridge_page="Riyadh")),
            ("the Oriental Pearl Tower", "Shanghai", "China", [" China"],
             dict(source_page="Oriental Pearl Tower",
                  bridge_page="Shanghai")),
            ("the Autograph Tower", "Jakarta", "Indonesia",
             [" Indonesia"],
             dict(source_page="Autograph Tower", bridge_page="Jakarta")),
        ]),
    dict(
        family="ship_builder_to_city", group="org",
        templates=dict(
            direct="The shipyard of {bridge} operated in",
            composed="The shipyard that built the {source} operated in",
            bridge_supplied="The {source} was built by {bridge}, whose "
                            "shipyard operated in"),
        ambiguity="each chosen vessel has a single builder of record",
        instances=[
            ("Titanic", "Harland & Wolff", "Belfast", [" Belfast"],
             dict(source_page="Titanic", bridge_page="Harland & Wolff")),
            ("Queen Mary", "John Brown & Company", "Clydebank",
             [" Clydebank"],
             dict(source_page="RMS Queen Mary",
                  bridge_page="John Brown & Company")),
            ("Bismarck", "Blohm & Voss", "Hamburg", [" Hamburg"],
             dict(source_page="German battleship Bismarck",
                  bridge_page="Blohm+Voss")),
            ("Cutty Sark", "Scott & Linton", "Dumbarton", [" Dumbarton"],
             dict(source_page="Cutty Sark", bridge_page="Scott & Linton")),
        ]),
    dict(
        family="waterfall_to_capital", group="geo_physical",
        templates=dict(
            direct="The capital of {bridge} is named",
            composed="The capital of the country that is home to "
                     "{source} is named",
            bridge_supplied="{source} is in {bridge}, whose capital is "
                            "named"),
        ambiguity="each chosen waterfall lies in one country",
        instances=[
            ("Angel Falls", "Venezuela", "Caracas", [" Caracas"],
             dict(source_page="Angel Falls", bridge_page="Venezuela")),
            ("Gullfoss", "Iceland", "Reykjavik",
             [" Reykjavik", " Reykjavík"],
             dict(source_page="Gullfoss", bridge_page="Iceland")),
            ("Kaieteur Falls", "Guyana", "Georgetown", [" Georgetown"],
             dict(source_page="Kaieteur Falls", bridge_page="Guyana")),
            ("Yosemite Falls", "the United States", "Washington, D.C.",
             [" Washington, D.C."],
             dict(source_page="Yosemite Falls",
                  bridge_page="United States")),
        ]),
    dict(
        family="show_creator_to_nationality", group="person_culture",
        templates=dict(
            direct="The screenwriter {bridge} is of the nationality "
                   "called",
            composed="The creator of the television series {source} is "
                     "of the nationality called",
            bridge_supplied="The television series {source} was created "
                            "by {bridge}, whose nationality is"),
        ambiguity="each series has a single credited creator",
        instances=[
            ("Breaking Bad", "Vince Gilligan", "American", [" American"],
             dict(source_page="Breaking Bad",
                  bridge_page="Vince Gilligan")),
            ("Fleabag", "Phoebe Waller-Bridge", "English",
             [" English", " British"],
             dict(source_page="Fleabag",
                  bridge_page="Phoebe Waller-Bridge")),
            ("Squid Game", "Hwang Dong-hyuk", "South Korean",
             [" South Korean", " Korean"],
             dict(source_page="Squid Game",
                  bridge_page="Hwang Dong-hyuk")),
            ("The Office", "Ricky Gervais", "English",
             [" English", " British"],
             dict(source_page="The Office (British TV series)",
                  bridge_page="Ricky Gervais")),
        ]),
]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def build_bundles() -> list[FactBundle]:
    bundles = []
    for fam in FAMILIES:
        inst = fam["instances"]
        for i, (src, bridge, ans, aliases, pages) in enumerate(inst):
            cf = inst[(i + 1) % len(inst)]
            while norm(cf[2]) == norm(ans):     # rotate past same-answer
                cf = inst[(inst.index(cf) + 1) % len(inst)]
            prompts = {v: t.format(source=src, bridge=bridge)
                       for v, t in fam["templates"].items()}
            slug = re.sub(r"[^a-z0-9]+", "-", src.lower()).strip("-")[:40]
            bundles.append(FactBundle(
                fact_id=f"{fam['family']}:{slug}",
                canonical_family=fam["family"],
                relation_group=fam["group"], bank="F",
                source=src, bridge=bridge, answer=ans,
                accepted_answers=aliases, prompts=prompts,
                counterfactual_bridge=cf[1], counterfactual_answer=cf[2],
                counterfactual_accepted=list(cf[3]),
                provenance={
                    "source": "wikipedia-20231101.en",
                    "pages": pages, "ambiguity_note": fam["ambiguity"]}))
    return bundles


def verify_against_reference(bundles: list[FactBundle]) -> dict:
    """Check hop support in the pinned per-revid wikipedia reference
    (fetch_wiki_reference.py): the bridge string must appear on the
    source-hop page, the answer string on the bridge page. A miss
    QUARANTINES the bundle (reported, dropped from the bank)."""
    from ..paths3 import run_root
    ref = run_root() / "bank_reference" / "wiki_reference_v4.jsonl"
    pages: dict[str, str] = {}
    for line in ref.read_text().splitlines():
        r = json.loads(line)
        pages[r["requested_title"]] = r["text"]
        pages.setdefault(r["title"], r["text"])
    titles = set()
    for b in bundles:
        titles.add(b.provenance["pages"]["source_page"])
        titles.add(b.provenance["pages"]["bridge_page"])
    report = {"missing_pages": sorted(titles - set(pages)),
              "reference": str(ref),
              "hop_failures": {}, "n_verified": 0}
    for b in bundles:
        pp = b.provenance["pages"]
        fails = []
        sp, bp = pages.get(pp["source_page"]), pages.get(pp["bridge_page"])
        if sp is None:
            fails.append(f"source page missing: {pp['source_page']}")
        elif norm(b.bridge.removeprefix("the ")) not in norm(sp):
            fails.append("bridge not stated on source page")
        if bp is None:
            fails.append(f"bridge page missing: {pp['bridge_page']}")
        elif norm(b.answer.removeprefix("the ")) not in norm(bp):
            fails.append("answer not stated on bridge page")
        if fails:
            report["hop_failures"][b.fact_id] = fails
        else:
            report["n_verified"] += 1
    return report


def main():
    require_clean_tree("--allow-dirty" in sys.argv)
    bundles = build_bundles()
    p2 = phase2_triple_keys(
        json.load(open(resolve_uri(P2_MANIFEST_URI)))["payload"])
    p2_answers = {k for k in p2 if k.startswith("|")}
    for b in bundles:      # answer-only dedup, §4.2a's sharp form
        assert f"|{norm(b.answer)}" not in p2_answers, \
            f"{b.fact_id}: answer burned by Phase 2"
    val = validate_bank(bundles, phase2_triples=p2)

    wiki = verify_against_reference(bundles)
    quarantined = set(wiki["hop_failures"]) | set(val["violations"]) \
        | set(val["alias_prefix_issues"])
    shipped = [b for b in bundles if b.fact_id not in quarantined]

    out = REPO_DATA / "bank_f_v5.jsonl"
    save_bank(shipped, out)
    fam_counts = {}
    for b in shipped:
        fam_counts[b.canonical_family] = \
            fam_counts.get(b.canonical_family, 0) + 1
    payload = {"n_authored": len(bundles), "n_shipped": len(shipped),
               "n_families": len(fam_counts), "family_counts": fam_counts,
               "validation": val, "reference_verification": wiki}
    cmd = "python -m jspace_phase3.experiments.author_bank_f"
    meta = REPO_DATA / "bank_f_v5.meta.json"
    write_result3(payload, meta, Provenance3(
        evidence_id=EVIDENCE_ID, tier=TIER, command=cmd, seed=0))
    register(EVIDENCE_ID, tier=TIER, command=cmd, supersedes=SUPERSEDES,
             what=(f"Bank F tranches 1+2: {len(shipped)} bundles / "
                   f"{len(fam_counts)} families authored, wikipedia-"
                   f"verified ({wiki['n_verified']} clean, "
                   f"{len(quarantined)} quarantined), Phase 2 triple+"
                   f"answer dedup enforced"),
             outputs=[out, meta])
    print(json.dumps({k: v for k, v in payload.items()
                      if k != "validation"}, indent=1)[:3000])
    print("validation ok:", val["ok"],
          "| violations:", len(val["violations"]),
          "| template_reuse:", len(val["template_reuse"]),
          "| p2 collisions:", len(val["phase2_triple_collisions"]))


if __name__ == "__main__":
    main()
