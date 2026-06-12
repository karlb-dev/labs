"""Generate the frozen truth-statement CSVs for Lab 4.

Run once at authoring time; the CSVs are vendored and never regenerated at
lab runtime (course rule: no live data downloads, no student-authored truth
sets). Deterministic by construction — no RNG anywhere.

Families:
  cities       "The city of Paris is in France."            (56 true, 56 false after expansion)
  comparisons  "Sixty-one is larger than fourteen."         (72 true, 72 false)
  negations    "The city of Paris is not in France."        (56 true, 56 false)

Negations are the classic generalization stressor: the surface form of a true
negated statement matches a FALSE affirmative one. A probe that tracks surface
co-occurrence rather than truth fails here with style.
"""

from __future__ import annotations

import csv
import pathlib

HERE = pathlib.Path(__file__).parent

CITY_COUNTRY = [
    ("Paris", "France"), ("Tokyo", "Japan"), ("Berlin", "Germany"),
    ("Rome", "Italy"), ("Madrid", "Spain"), ("Lisbon", "Portugal"),
    ("Vienna", "Austria"), ("Cairo", "Egypt"), ("Nairobi", "Kenya"),
    ("Lima", "Peru"), ("Oslo", "Norway"), ("Athens", "Greece"),
    ("Bangkok", "Thailand"), ("Beijing", "China"), ("Moscow", "Russia"),
    ("Ottawa", "Canada"), ("Canberra", "Australia"), ("Dublin", "Ireland"),
    ("Warsaw", "Poland"), ("Prague", "Czechia"), ("Havana", "Cuba"),
    ("Seoul", "South Korea"), ("Hanoi", "Vietnam"), ("Helsinki", "Finland"),
    ("Stockholm", "Sweden"), ("Budapest", "Hungary"), ("Brussels", "Belgium"),
    ("Amsterdam", "the Netherlands"), ("Copenhagen", "Denmark"),
    ("Wellington", "New Zealand"),
    # Expanded for robustness / more categories (Europe, Asia, Africa, Americas, Oceania)
    ("Bern", "Switzerland"), ("Bratislava", "Slovakia"), ("Ljubljana", "Slovenia"),
    ("Riga", "Latvia"), ("Tallinn", "Estonia"), ("Vilnius", "Lithuania"),
    ("Sofia", "Bulgaria"), ("Bucharest", "Romania"),
    ("Ankara", "Turkey"), ("Tehran", "Iran"), ("Baghdad", "Iraq"),
    ("Riyadh", "Saudi Arabia"), ("Doha", "Qatar"), ("Abu Dhabi", "the UAE"),
    ("Addis Ababa", "Ethiopia"), ("Accra", "Ghana"), ("Dakar", "Senegal"),
    # Buenos Aires before Santiago so Santiago's rotated wrong-country is not
    # the Philippines (Santiago City, Isabela is a real Philippine city, which
    # would make the generated "false" statement arguably true).
    ("Bogota", "Colombia"), ("Buenos Aires", "Argentina"), ("Santiago", "Chile"),
    ("Mexico City", "Mexico"), ("Toronto", "Canada"),
    ("Kuala Lumpur", "Malaysia"), ("Singapore", "Singapore"),
    ("Jakarta", "Indonesia"), ("Manila", "the Philippines"),
]

ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_word(n: int) -> str:
    if n < 20:
        return ONES[n]
    tens, ones = divmod(n, 10)
    return TENS[tens] + ("-" + ONES[ones] if ones else "")


def cap(s: str) -> str:
    return s[0].upper() + s[1:]


def wrong_country(i: int) -> str:
    j = (i + 7) % len(CITY_COUNTRY)
    if CITY_COUNTRY[j][1] == CITY_COUNTRY[i][1]:
        j = (j + 1) % len(CITY_COUNTRY)
    return CITY_COUNTRY[j][1]


def make_cities() -> list[dict]:
    rows = []
    for i, (city, country) in enumerate(CITY_COUNTRY):
        rows.append({"statement_id": f"city_t_{i:02d}", "family": "cities",
                     "statement": f"The city of {city} is in {country}.", "label": 1,
                     "meta": f"{city}|{country}"})
        rows.append({"statement_id": f"city_f_{i:02d}", "family": "cities",
                     "statement": f"The city of {city} is in {wrong_country(i)}.", "label": 0,
                     "meta": f"{city}|{wrong_country(i)}"})
    return rows


def make_negations() -> list[dict]:
    rows = []
    for i, (city, country) in enumerate(CITY_COUNTRY):
        # "Paris is not in France."  -> a FALSE statement (label 0)
        rows.append({"statement_id": f"neg_f_{i:02d}", "family": "negations",
                     "statement": f"The city of {city} is not in {country}.", "label": 0,
                     "meta": f"{city}|{country}"})
        # "Paris is not in Egypt."   -> a TRUE statement (label 1)
        rows.append({"statement_id": f"neg_t_{i:02d}", "family": "negations",
                     "statement": f"The city of {city} is not in {wrong_country(i)}.", "label": 1,
                     "meta": f"{city}|{wrong_country(i)}"})
    return rows


def make_comparisons() -> list[dict]:
    rows = []
    for i in range(72):  # expanded further for robustness (more number pairs)
        a = (i * 13 + 8) % 90 + 5
        b = (i * 29 + 3) % 90 + 5
        if a == b:
            b += 1
        rel = "larger" if i % 2 == 0 else "smaller"
        truth = (a > b) if rel == "larger" else (a < b)
        rows.append({"statement_id": f"cmp_{'t' if truth else 'f'}_{i:02d}a", "family": "comparisons",
                     "statement": f"{cap(number_word(a))} is {rel} than {number_word(b)}.",
                     "label": int(truth), "meta": f"{a}|{rel}|{b}"})
        # The mirrored statement flips the label, keeping the family balanced.
        rows.append({"statement_id": f"cmp_{'f' if truth else 't'}_{i:02d}b", "family": "comparisons",
                     "statement": f"{cap(number_word(b))} is {rel} than {number_word(a)}.",
                     "label": int(not truth), "meta": f"{b}|{rel}|{a}"})
    return rows


def write(name: str, rows: list[dict]) -> None:
    path = HERE / name
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["statement_id", "family", "statement", "label", "meta"])
        writer.writeheader()
        writer.writerows(rows)
    n_true = sum(r["label"] for r in rows)
    print(f"{name}: {len(rows)} statements ({n_true} true / {len(rows) - n_true} false)")


if __name__ == "__main__":
    write("truth_cities.csv", make_cities())
    write("truth_comparisons.csv", make_comparisons())
    write("truth_negations.csv", make_negations())
