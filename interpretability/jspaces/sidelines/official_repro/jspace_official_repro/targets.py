"""Target-token derivation: synonym expansions and model-dependent canons.

Only released synonym semantics are expanded (order-ops: "numbers → digit
and word forms; operations → symbol and word forms"); the exact table is
frozen here (D8). Everything resolves through the contract §6 whitespace
normalization; multi-token targets are TOKENIZATION_GATED, never
first-fragment matched.
"""
from __future__ import annotations

from .rendering import preferred_token, single_token_forms

_NUMBER_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen", "20": "twenty", "21": "twenty-one",
    "22": "twenty-two", "23": "twenty-three", "24": "twenty-four",
    "25": "twenty-five",
}

#: Frozen order-ops operation expansion (D8): key -> symbol + word forms.
_OPERATION_FORMS = {
    "addition": ["addition", "plus", "+"],
    "subtraction": ["subtraction", "minus", "-"],
    "multiplication": ["multiplication", "times", "*", "×"],
    "division": ["division", "divided", "/", "÷"],
    "mod": ["mod", "modulo", "%"],
    "squared": ["squared", "square", "²"],
}

#: English number words for the selectivity-linecount canon (two-digit
#: answers live in 20..99; the README names "twenty, thirty, …").
_TENS_WORDS = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy",
               "eighty", "ninety"]


def order_ops_forms(key: str) -> list[str]:
    """All string forms for an order-ops intermediate key."""
    if key in _OPERATION_FORMS:
        return list(_OPERATION_FORMS[key])
    forms = [key]
    if key in _NUMBER_WORDS:
        forms.append(_NUMBER_WORDS[key])
    return forms


def synonym_token_ids(tokenizer, forms: list[str]) -> list[int]:
    """Single-token ids for every form (space + bare variants, deduped)."""
    ids: list[int] = []
    for form in forms:
        for token_id in single_token_forms(tokenizer, form).values():
            if token_id not in ids:
                ids.append(token_id)
    return ids


def intermediate_token_ids(tokenizer, set_name: str, key: str) -> list[int]:
    """Token ids scoring an eval intermediate; [] => TOKENIZATION_GATED."""
    forms = order_ops_forms(key) if set_name == "lens-eval-order-ops" else [key]
    return synonym_token_ids(tokenizer, forms)


def linecount_number_canon(tokenizer) -> dict[str, list[int]]:
    """Model-specific canon for selectivity-linecount: every two-digit
    string 10..99 plus English tens words, single-token only."""
    canon: dict[str, list[int]] = {}
    for value in range(10, 100):
        ids = synonym_token_ids(tokenizer, [str(value)])
        if ids:
            canon[str(value)] = ids
    for word in _TENS_WORDS:
        ids = synonym_token_ids(tokenizer, [word])
        if ids:
            canon[word] = ids
    return canon


def capacity_canon(tokenizer, pool: list[str], n_targets: int) -> list[str]:
    """First ``n_targets`` pool entries that are single tokens under the
    target model (released construction; canon is model-dependent by
    design). Returns the accepted words in pool order."""
    accepted = []
    for word in pool:
        if preferred_token(tokenizer, word) is not None:
            accepted.append(word)
            if len(accepted) == n_targets:
                break
    return accepted
