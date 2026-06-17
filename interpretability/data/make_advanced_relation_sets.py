"""Generate the Lab 12 relation-geometry set: advanced_relation_geometry.csv.

Run once at authoring time; the CSV is vendored (course rule: no live data at
lab runtime). The item INVENTORY below is deterministic; the only authoring-
time filter is dual-tokenizer verification, which is part of generation, not
randomness: an item enters the frozen CSV only if every measured token
(subject, relation word, target, both distractors) is a single token with a
leading space under BOTH course tokenizers (gpt2 BPE and Olmo-3). Dropped
candidates are printed as an audit so the pruning is visible in the commit.

Why this exists
===============
Lab 12 asks whether relation *classes* share internal geometry or whether each
relation is its own small trick. That question dies instantly to three cheap
explanations: the "relation direction" is really (a) an entity-class
direction, (b) a template/syntax direction, or (c) a token-identity echo of
the relation word. This dataset is built so the first two are controlled by
construction:

* Three RELATION-SWAP GROUPS hold the subject entities and the template
  skeleton fixed while only the relation changes:

    country_sem:  "The capital of France is" / "The language of France is" /
                  "The continent of France is"        (same countries)
    adj_morph:    "The opposite of big is" / "The comparative of big is"
                                                       (same adjectives)
    month_seq:    "The month after May is" / "The month before May is"
                                                       (same months)

  Within a swap group, prompts for the same subject are token-aligned and
  differ at EXACTLY the relation-word position, so (1) any probe separation
  cannot be entity class or template syntax, and (2) clean/corrupt
  relation-swap patching is well defined at one position.

* Five further families (currency, plural, color, material, home) add
  breadth across entity classes for the cross-group comparison and the
  direction-cosine atlas. Two of them (currency, home) end in "is the"
  because bare "is" invites an article rather than the answer; the trailing
  word is recorded per family so the lab never assumes one skeleton.

Authoring-time casualties worth knowing about (see git history of this file):
"superlative" is 3 BPE tokens under both course tokenizers, so the morphology
swap group is opposite/comparative only; and an author_of family (work ->
author) lost nearly every item because book titles are multi-token, so the
outline's "authorship" relation is deferred to a Tier B extension that reads
multi-token subject spans instead of forcing it into this single-token frame.

Every item carries TWO distractors, mirroring make_relation_sets.py:
  hard  — same answer class (Paris vs Rome): instance discrimination
  easy  — fixed cross-family donor (Paris vs bigger): class discrimination
The margin gap between them on the SAME prompt is the dual-distractor
stability measurement.

Output: advanced_relation_geometry.csv with columns
  item_id, family, swap_group, entity_group, template, trailing, prompt,
  relword, subject, target, hard_distractor, easy_distractor, note

Token positions are NOT stored: they are tokenizer-dependent, and the lab
locates the subject/relation-word tokens at runtime against the tokenizer it
is actually using (and re-verifies single-token status, dropping with an
audit row anything a future tokenizer breaks).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib

HERE = pathlib.Path(__file__).parent
OUT_NAME = "advanced_relation_geometry.csv"
MANIFEST_NAME = "MANIFEST.json"

# The two course tokenizers. Olmo-3 base and instruct share one tokenizer;
# the instruct id is the one usually present in a local HF cache.
DEFAULT_TOKENIZERS = ("gpt2", "allenai/Olmo-3-7B-Instruct")

# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------
#
# Family spec: (family, swap_group, entity_group, template, trailing, relword)
# template contains {X}; trailing is "" or "the" (already part of template).
# Items: (subject, target, near_tie, note) — hard distractors are derived
# cyclically inside the family (next DISTINCT target), easy distractors come
# from EASY_DONOR. No RNG anywhere.

FAMILY_SPECS: list[tuple[str, str, str, str, str, str]] = [
    ("capital_of",     "country_sem", "country",   "The capital of {X} is",        "",    "capital"),
    ("language_of",    "country_sem", "country",   "The language of {X} is",       "",    "language"),
    ("continent_of",   "country_sem", "country",   "The continent of {X} is",      "",    "continent"),
    ("currency_of",    "",            "country",   "The currency of {X} is the",   "the", "currency"),
    ("opposite_of",    "adj_morph",   "adjective", "The opposite of {X} is",       "",    "opposite"),
    ("comparative_of", "adj_morph",   "adjective", "The comparative of {X} is",    "",    "comparative"),
    ("month_after",    "month_seq",   "month",     "The month after {X} is",       "",    "after"),
    ("month_before",   "month_seq",   "month",     "The month before {X} is",      "",    "before"),
    ("plural_of",      "",            "noun",      "The plural of {X} is",         "",    "plural"),
    ("color_of",       "",            "substance", "The color of {X} is",          "",    "color"),
    ("material_of",    "",            "object",    "The material of a {X} is",     "",    "material"),
    ("home_of",        "",            "animal",    "The home of a {X} is the",     "the", "home"),
]

# Country roster shared by the country families. A country appears in a
# family only if its answer for that relation is a clean single token; the
# verifier prunes per family, and the lab restricts within-group analyses to
# the subjects the swap families still share after pruning.
COUNTRY_ANSWERS: list[tuple[str, str | None, str | None, str | None, str | None]] = [
    # (country, capital, language, continent, currency)
    ("France",      "Paris",      "French",      "Europe", "euro"),
    ("Germany",     "Berlin",     "German",      "Europe", "euro"),
    ("Italy",       "Rome",       "Italian",     "Europe", "euro"),
    ("Spain",       "Madrid",     "Spanish",     "Europe", "euro"),
    ("Japan",       "Tokyo",      "Japanese",    "Asia",   "yen"),
    ("China",       "Beijing",    "Chinese",     "Asia",   "yuan"),
    ("Russia",      "Moscow",     "Russian",     "Europe", "ruble"),
    ("Egypt",       "Cairo",      "Arabic",      "Africa", "pound"),
    ("Greece",      "Athens",     "Greek",       "Europe", "euro"),
    ("Norway",      "Oslo",       "Norwegian",   "Europe", None),
    ("Poland",      "Warsaw",     "Polish",      "Europe", None),
    ("Austria",     "Vienna",     "German",      "Europe", "euro"),
    ("Portugal",    "Lisbon",     "Portuguese",  "Europe", "euro"),
    ("Thailand",    "Bangkok",    "Thai",        "Asia",   "baht"),
    ("England",     "London",     "English",     "Europe", "pound"),
    ("Switzerland", "Bern",       "German",      "Europe", "franc"),
    ("Australia",   "Canberra",   "English",     None,     "dollar"),
    ("Ireland",     "Dublin",     "English",     "Europe", "euro"),
    ("Denmark",     "Copenhagen", "Danish",      "Europe", "krone"),
    ("Sweden",      "Stockholm",  "Swedish",     "Europe", "krona"),
    ("Hungary",     "Budapest",   "Hungarian",   "Europe", None),
    ("Finland",     "Helsinki",   "Finnish",     "Europe", "euro"),
    ("Indonesia",   "Jakarta",    "Indonesian",  "Asia",   None),
    ("India",       "Delhi",      "Hindi",       "Asia",   "rupee"),
    ("Brazil",      None,         "Portuguese",  None,     "real"),
    ("Mexico",      None,         "Spanish",     None,     "peso"),
    ("Argentina",   None,         "Spanish",     None,     "peso"),
    ("Vietnam",     "Hanoi",      "Vietnamese",  "Asia",   "dong"),
    ("Korea",       "Seoul",      "Korean",      "Asia",   "won"),
    ("Kenya",       "Nairobi",    "Swahili",     "Africa", "shilling"),
    ("Nigeria",     "Abuja",      "English",     "Africa", "naira"),
    ("Morocco",     "Rabat",      "Arabic",      "Africa", "dirham"),
    ("Iran",        "Tehran",     "Persian",     "Asia",   "rial"),
    ("Iraq",        "Baghdad",    "Arabic",      "Asia",   "dinar"),
    ("Netherlands", "Amsterdam",  "Dutch",       "Europe", "euro"),
]

# Adjective roster shared by the morphology swap group.
ADJECTIVE_ANSWERS: list[tuple[str, str, str]] = [
    # (adjective, opposite, comparative)
    ("hot",    "cold",   "hotter"),
    ("big",    "small",  "bigger"),
    ("fast",   "slow",   "faster"),
    ("tall",   "short",  "taller"),
    ("old",    "young",  "older"),
    ("strong", "weak",   "stronger"),
    ("happy",  "sad",    "happier"),
    ("hard",   "soft",   "harder"),
    ("light",  "dark",   "lighter"),
    ("loud",   "quiet",  "louder"),
    ("rich",   "poor",   "richer"),
    ("clean",  "dirty",  "cleaner"),
    ("young",  "old",    "younger"),
    ("weak",   "strong", "weaker"),
    ("cold",   "hot",    "colder"),
    ("dark",   "light",  "darker"),
    ("small",  "big",    "smaller"),
    ("slow",   "fast",   "slower"),
    ("soft",   "hard",   "softer"),
    ("cheap",  "expensive", "cheaper"),
    ("deep",   "shallow", "deeper"),
    ("wide",   "narrow", "wider"),
    ("warm",   "cool",   "warmer"),
    ("safe",   "dangerous", "safer"),
]

MONTHS = ("January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December")

# (subject, target, near_tie, note) for the six breadth families.
SINGLE_FAMILY_ITEMS: dict[str, list[tuple[str, str, bool, str]]] = {
    "plural_of": [
        ("cat", "cats", False, ""), ("dog", "dogs", False, ""),
        ("book", "books", False, ""), ("car", "cars", False, ""),
        ("tree", "trees", False, ""), ("house", "houses", False, ""),
        ("child", "children", False, "irregular"), ("mouse", "mice", False, "irregular"),
        ("foot", "feet", False, "irregular"), ("tooth", "teeth", False, "irregular"),
        ("man", "men", False, "irregular"), ("woman", "women", False, "irregular"),
        ("person", "people", False, "irregular"), ("city", "cities", False, ""),
        ("country", "countries", False, ""), ("baby", "babies", False, ""),
        ("leaf", "leaves", False, "irregular"), ("knife", "knives", False, "irregular"),
        ("wolf", "wolves", False, "irregular"), ("box", "boxes", False, ""),
        ("bus", "buses", False, ""), ("goose", "geese", False, "irregular"),
        ("story", "stories", False, ""), ("party", "parties", False, ""),
        ("family", "families", False, ""), ("key", "keys", False, ""),
    ],
    "color_of": [
        ("grass", "green", False, ""), ("snow", "white", False, ""),
        ("blood", "red", False, ""), ("milk", "white", False, ""),
        ("coal", "black", False, ""), ("butter", "yellow", False, ""),
        ("chocolate", "brown", False, ""), ("coffee", "brown", True, "black is also natural"),
        ("ivory", "white", False, ""), ("jade", "green", False, ""),
        ("ruby", "red", False, ""), ("emerald", "green", False, ""),
        ("mud", "brown", False, ""), ("straw", "yellow", False, ""),
        ("wine", "red", True, "white wine exists"), ("mustard", "yellow", False, ""),
    ],
    "material_of": [
        ("window", "glass", False, ""), ("bottle", "glass", False, ""),
        ("ring", "gold", True, "silver also common"), ("knife", "steel", False, ""),
        ("tire", "rubber", False, ""), ("sweater", "wool", False, ""),
        ("shirt", "cotton", False, ""), ("barrel", "wood", False, ""),
        ("coin", "metal", False, ""), ("brick", "clay", False, ""),
        ("candle", "wax", False, ""), ("statue", "stone", True, "marble also common"),
    ],
    "home_of": [
        ("bird", "nest", False, ""), ("bee", "hive", False, ""),
        ("lion", "den", False, ""), ("spider", "web", False, ""),
        ("rabbit", "burrow", False, ""), ("horse", "stable", False, ""),
        ("pig", "sty", False, ""), ("chicken", "coop", False, ""),
        ("bat", "cave", False, ""), ("frog", "pond", False, ""),
        ("bear", "den", False, ""), ("eagle", "nest", False, ""),
    ],
}

# Cross-class ("easy") distractors: a fixed donor word per family, drawn from
# a DIFFERENT family's target space. Deterministic, no RNG.
EASY_DONOR = {
    "capital_of":     " bigger",
    "language_of":    " hive",
    "continent_of":   " cats",
    "currency_of":    " February",
    "opposite_of":    " Paris",
    "comparative_of": " French",
    "month_after":    " wool",
    "month_before":   " green",
    "plural_of":      " Asia",
    "color_of":       " Tokyo",
    "material_of":    " yen",
    "home_of":        " faster",
}


def build_inventory() -> list[dict[str, str]]:
    """Expand the rosters into one row per candidate item (pre-verification)."""
    rows: list[dict[str, str]] = []
    spec_by_family = {f[0]: f for f in FAMILY_SPECS}

    def add(family: str, subject: str, target: str, near_tie: bool, note: str) -> None:
        _, swap_group, entity_group, template, trailing, relword = spec_by_family[family]
        full_note = note
        if near_tie:
            full_note = (full_note + "; " if full_note else "") + "near_tie=1"
        rows.append({
            "item_id": f"{family}_{subject.lower()}",
            "family": family,
            "swap_group": swap_group,
            "entity_group": entity_group,
            "template": template,
            "trailing": trailing,
            "prompt": template.format(X=subject),
            "relword": relword,
            "subject": subject,
            "target": " " + target,
            "near_tie": near_tie,
            "note": full_note,
        })

    for country, capital, language, continent, currency in COUNTRY_ANSWERS:
        if capital:
            add("capital_of", country, capital, False, "")
        if language:
            add("language_of", country, language, False, "")
        if continent:
            add("continent_of", country, continent, False, "")
        if currency:
            add("currency_of", country, currency, False, "")
    for adj, opposite, comparative in ADJECTIVE_ANSWERS:
        add("opposite_of", adj, opposite, False, "")
        add("comparative_of", adj, comparative, False, "")
    for i, month in enumerate(MONTHS):
        add("month_after", month, MONTHS[(i + 1) % 12], False, "")
        add("month_before", month, MONTHS[(i - 1) % 12], False, "")
    for family, items in SINGLE_FAMILY_ITEMS.items():
        for subject, target, near_tie, note in items:
            add(family, subject, target, near_tie, note)
    return rows


def assign_hard_distractors(rows: list[dict[str, str]]) -> None:
    """Cyclic same-family hard distractor: the next DISTINCT target in family
    order. Deterministic; mirrors Lab 5's cyclic clean/corrupt pairing."""
    by_family: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        by_family.setdefault(r["family"], []).append(r)
    for family, items in by_family.items():
        targets = [r["target"] for r in items]
        for i, r in enumerate(items):
            hard = ""
            for step in range(1, len(items)):
                candidate = targets[(i + step) % len(items)]
                if candidate != r["target"]:
                    hard = candidate
                    break
            if not hard:
                raise SystemExit(f"{family}: all targets identical; cannot build hard distractors")
            r["hard_distractor"] = hard
            r["easy_distractor"] = EASY_DONOR[family]


def load_tokenizers(ids: list[str]):
    from transformers import AutoTokenizer

    out = []
    for tok_id in ids:
        try:
            tok = AutoTokenizer.from_pretrained(tok_id, local_files_only=True)
        except Exception:
            tok = AutoTokenizer.from_pretrained(tok_id)
        out.append((tok_id, tok))
        print(f"loaded tokenizer {tok_id} (vocab {tok.vocab_size})")
    return out


def single_token(tok, text: str) -> bool:
    return len(tok.encode(text, add_special_tokens=False)) == 1


def verify_core(rows: list[dict[str, str]], tokenizers) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Keep a row only if subject, relword, and target are single tokens in
    EVERY tokenizer and the subject/relword each appear exactly once in the
    prompt encoding (so role positions are unambiguous). Runs BEFORE hard
    distractors are assigned, so a multi-token answer kills only its own item
    and never a neighbor's."""
    kept: list[dict[str, str]] = []
    dropped: list[dict[str, str]] = []
    for r in rows:
        problems: list[str] = []
        for tok_id, tok in tokenizers:
            checks = {
                "subject": " " + r["subject"],
                "relword": " " + r["relword"],
                "target": r["target"],
            }
            for role, text in checks.items():
                if not single_token(tok, text):
                    n = len(tok.encode(text, add_special_tokens=False))
                    problems.append(f"{role} {text!r} is {n} tokens under {tok_id}")
            prompt_ids = tok.encode(r["prompt"], add_special_tokens=False)
            for role, text in (("subject", " " + r["subject"]), ("relword", " " + r["relword"])):
                ids = tok.encode(text, add_special_tokens=False)
                if len(ids) == 1:
                    hits = [p for p, t in enumerate(prompt_ids) if t == ids[0]]
                    if len(hits) != 1:
                        problems.append(f"{role} token occurs {len(hits)}x in prompt under {tok_id}")
        if problems:
            dropped.append({"item_id": r["item_id"], "problems": "; ".join(sorted(set(problems)))})
        else:
            kept.append(r)
    return kept, dropped


def verify_distractors(rows: list[dict[str, str]], tokenizers) -> None:
    """Hard distractors are surviving targets (single-token by construction);
    easy donors and per-row distinctness still need a hard check."""
    for family, donor in EASY_DONOR.items():
        for tok_id, tok in tokenizers:
            if not single_token(tok, donor):
                raise SystemExit(f"easy donor {donor!r} for {family} is multi-token under {tok_id}")
    for r in rows:
        if r["target"] in (r["hard_distractor"], r["easy_distractor"]) or r["hard_distractor"] == r["easy_distractor"]:
            raise SystemExit(f"{r['item_id']}: target/hard/easy not distinct")


def validate_inventory(rows: list[dict[str, str]]) -> None:
    ids = [r["item_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate item_id in inventory")
    families = {r["family"] for r in rows}
    if families != {f[0] for f in FAMILY_SPECS}:
        missing = {f[0] for f in FAMILY_SPECS} - families
        raise SystemExit(f"families with zero surviving items: {sorted(missing)}")
    for r in rows:
        if r["prompt"].endswith(" "):
            raise SystemExit(f"{r['item_id']}: prompt ends with a space")
        for key in ("target", "hard_distractor", "easy_distractor"):
            if not r[key].startswith(" "):
                raise SystemExit(f"{r['item_id']}: {key} must carry the leading space")
    # Swap groups need shared subjects to support relation-swap pairs.
    for group in ("country_sem", "adj_morph", "month_seq"):
        group_families = sorted({r["family"] for r in rows if r["swap_group"] == group})
        if len(group_families) < 2:
            raise SystemExit(f"swap group {group}: fewer than 2 surviving families")
        subject_sets = [
            {r["subject"] for r in rows if r["family"] == fam} for fam in group_families
        ]
        common = set.intersection(*subject_sets)
        if len(common) < 4:
            raise SystemExit(
                f"swap group {group}: only {len(common)} shared subjects across "
                f"{group_families}; relation-swap patching needs at least 4"
            )
        print(f"swap group {group}: {group_families} share {len(common)} subjects")


def write_outputs(rows: list[dict[str, str]], tokenizer_ids: list[str]) -> None:
    fieldnames = [
        "item_id", "family", "swap_group", "entity_group", "template", "trailing",
        "prompt", "relword", "subject", "target", "hard_distractor",
        "easy_distractor", "note",
    ]
    out_path = HERE / OUT_NAME
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    per_family: dict[str, int] = {}
    for r in rows:
        per_family[r["family"]] = per_family.get(r["family"], 0) + 1
    print(f"{OUT_NAME}: {len(rows)} rows across {len(per_family)} families")
    for fam in sorted(per_family):
        print(f"  {fam}: {per_family[fam]}")

    # Frozen-data manifest (advanced-course data rule: committed hashes).
    manifest_path = HERE / MANIFEST_NAME
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[OUT_NAME] = {
        "sha256": digest,
        "rows": len(rows),
        "families": per_family,
        "verified_tokenizers": tokenizer_ids,
        "generator": "make_advanced_relation_sets.py",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{MANIFEST_NAME}: recorded sha256 {digest[:16]}…")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tokenizers", default=",".join(DEFAULT_TOKENIZERS),
        help="Comma-separated tokenizer ids that every measured token must be single-token under.",
    )
    args = parser.parse_args()
    tokenizer_ids = [t.strip() for t in args.tokenizers.split(",") if t.strip()]

    rows = build_inventory()
    tokenizers = load_tokenizers(tokenizer_ids)
    kept, dropped = verify_core(rows, tokenizers)
    if dropped:
        print(f"\ndropped {len(dropped)} candidate items at authoring time:")
        for d in dropped:
            print(f"  {d['item_id']}: {d['problems']}")
        print()
    assign_hard_distractors(kept)
    verify_distractors(kept, tokenizers)
    validate_inventory(kept)
    write_outputs(kept, tokenizer_ids)


if __name__ == "__main__":
    main()
