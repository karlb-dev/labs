#!/usr/bin/env python3
"""Generate the frozen Lab 28 benign editing/unlearning target CSV.

The generator is deterministic and intentionally small. It uses public facts and
synthetic toy associations only. Runtime tokenization is still verified by the
lab because tokenizer revisions can change what is single-token.
"""

from __future__ import annotations

import csv
import json
import pathlib

OUT = pathlib.Path(__file__).with_name("editing_unlearning_targets.csv")


def j(rows):
    return json.dumps(rows, ensure_ascii=False, sort_keys=True)


def main() -> None:
    rows = []

    def add(target_id, family, edit_type, prompt, before, after, donor, retain, paraphrase, neighbor, notes):
        rows.append({
            "target_id": target_id,
            "family": family,
            "edit_type": edit_type,
            "prompt": prompt,
            "target_before": before,
            "target_after": after,
            "retain_prompts_json": j(retain),
            "paraphrase_prompts_json": j(paraphrase),
            "neighbor_prompts_json": j(neighbor),
            "safety_notes": notes,
            "donor_prompt": donor,
        })

    add(
        "edit_capital_france_to_rome",
        "capital_counterfactual",
        "counterfactual_activation_edit",
        "The capital of France is",
        " Paris",
        " Rome",
        "The capital of Italy is",
        [
            {"prompt": "The capital of Germany is", "target": " Berlin", "distractor": " Rome", "role": "other_capital_germany"},
            {"prompt": "The capital of Spain is", "target": " Madrid", "distractor": " Rome", "role": "other_capital_spain"},
            {"prompt": "The language of France is", "target": " French", "distractor": " Rome", "role": "same_subject_other_relation"},
        ],
        [
            {"prompt": "In France, the capital city is", "role": "paraphrase_city"},
            {"prompt": "France has its capital in", "role": "paraphrase_has"},
        ],
        [
            {"prompt": "The capital of Italy is", "target": " Rome", "distractor": " Paris", "role": "donor_fact"},
            {"prompt": "The language of Italy is", "target": " Italian", "distractor": " Rome", "role": "donor_subject_other_relation"},
        ],
        "Benign public country-capital association; counterfactual target is harmless; inference-time activation edit only.",
    )
    add(
        "edit_capital_germany_to_paris",
        "capital_counterfactual",
        "counterfactual_activation_edit",
        "The capital of Germany is",
        " Berlin",
        " Paris",
        "The capital of France is",
        [
            {"prompt": "The capital of Italy is", "target": " Rome", "distractor": " Paris", "role": "other_capital_italy"},
            {"prompt": "The capital of Spain is", "target": " Madrid", "distractor": " Paris", "role": "other_capital_spain"},
            {"prompt": "The language of Germany is", "target": " German", "distractor": " Paris", "role": "same_subject_other_relation"},
        ],
        [
            {"prompt": "In Germany, the capital city is", "role": "paraphrase_city"},
            {"prompt": "Germany has its capital in", "role": "paraphrase_has"},
        ],
        [
            {"prompt": "The capital of France is", "target": " Paris", "distractor": " Berlin", "role": "donor_fact"},
            {"prompt": "The language of France is", "target": " French", "distractor": " Paris", "role": "donor_subject_other_relation"},
        ],
        "Benign public country-capital association; counterfactual target is harmless; inference-time activation edit only.",
    )
    add(
        "edit_capital_italy_to_madrid",
        "capital_counterfactual",
        "counterfactual_activation_edit",
        "The capital of Italy is",
        " Rome",
        " Madrid",
        "The capital of Spain is",
        [
            {"prompt": "The capital of France is", "target": " Paris", "distractor": " Madrid", "role": "other_capital_france"},
            {"prompt": "The capital of Germany is", "target": " Berlin", "distractor": " Madrid", "role": "other_capital_germany"},
            {"prompt": "The language of Italy is", "target": " Italian", "distractor": " Madrid", "role": "same_subject_other_relation"},
        ],
        [
            {"prompt": "In Italy, the capital city is", "role": "paraphrase_city"},
            {"prompt": "Italy has its capital in", "role": "paraphrase_has"},
        ],
        [
            {"prompt": "The capital of Spain is", "target": " Madrid", "distractor": " Rome", "role": "donor_fact"},
            {"prompt": "The language of Spain is", "target": " Spanish", "distractor": " Madrid", "role": "donor_subject_other_relation"},
        ],
        "Benign public country-capital association; counterfactual target is harmless; inference-time activation edit only.",
    )
    add(
        "edit_language_france_to_german",
        "language_counterfactual",
        "counterfactual_relation_edit",
        "The language of France is",
        " French",
        " German",
        "The language of Germany is",
        [
            {"prompt": "The language of Italy is", "target": " Italian", "distractor": " German", "role": "other_language_italy"},
            {"prompt": "The language of Spain is", "target": " Spanish", "distractor": " German", "role": "other_language_spain"},
            {"prompt": "The capital of France is", "target": " Paris", "distractor": " German", "role": "same_subject_other_relation"},
        ],
        [
            {"prompt": "In France, the main language is", "role": "paraphrase_main"},
            {"prompt": "France has a national language called", "role": "paraphrase_called"},
        ],
        [
            {"prompt": "The language of Germany is", "target": " German", "distractor": " French", "role": "donor_fact"},
            {"prompt": "The capital of Germany is", "target": " Berlin", "distractor": " German", "role": "donor_subject_other_relation"},
        ],
        "Benign public country-language association; counterfactual target is harmless; inference-time activation edit only.",
    )
    add(
        "edit_language_germany_to_french",
        "language_counterfactual",
        "counterfactual_relation_edit",
        "The language of Germany is",
        " German",
        " French",
        "The language of France is",
        [
            {"prompt": "The language of Italy is", "target": " Italian", "distractor": " French", "role": "other_language_italy"},
            {"prompt": "The language of Spain is", "target": " Spanish", "distractor": " French", "role": "other_language_spain"},
            {"prompt": "The capital of Germany is", "target": " Berlin", "distractor": " French", "role": "same_subject_other_relation"},
        ],
        [
            {"prompt": "In Germany, the main language is", "role": "paraphrase_main"},
            {"prompt": "Germany has a national language called", "role": "paraphrase_called"},
        ],
        [
            {"prompt": "The language of France is", "target": " French", "distractor": " German", "role": "donor_fact"},
            {"prompt": "The capital of France is", "target": " Paris", "distractor": " French", "role": "donor_subject_other_relation"},
        ],
        "Benign public country-language association; counterfactual target is harmless; inference-time activation edit only.",
    )
    add(
        "edit_month_after_january_to_march",
        "sequence_counterfactual",
        "counterfactual_relation_edit",
        "The month after January is",
        " February",
        " March",
        "The month after February is",
        [
            {"prompt": "The month after March is", "target": " April", "distractor": " March", "role": "other_sequence_after"},
            {"prompt": "The month before March is", "target": " February", "distractor": " March", "role": "inverse_relation"},
        ],
        [
            {"prompt": "After January comes", "role": "paraphrase_after"},
            {"prompt": "January is followed by", "role": "paraphrase_followed"},
        ],
        [
            {"prompt": "The month after February is", "target": " March", "distractor": " February", "role": "donor_fact"},
            {"prompt": "The month before March is", "target": " February", "distractor": " March", "role": "nearby_inverse"},
        ],
        "Benign calendar sequence association; counterfactual target is harmless; inference-time activation edit only.",
    )
    add(
        "edit_codeword_alice_blue_to_green",
        "synthetic_codeword",
        "synthetic_counterfactual_activation_edit",
        "The code word for Alice is",
        " blue",
        " green",
        "The code word for Bob is",
        [
            {"prompt": "The code word for Carol is", "target": " red", "distractor": " green", "role": "other_synthetic_carol"},
            {"prompt": "The code word for Dave is", "target": " yellow", "distractor": " green", "role": "other_synthetic_dave"},
        ],
        [
            {"prompt": "Alice's code word is", "role": "paraphrase_possessive"},
            {"prompt": "For Alice, the code word is", "role": "paraphrase_for"},
        ],
        [
            {"prompt": "The code word for Bob is", "target": " green", "distractor": " blue", "role": "donor_synthetic"},
            {"prompt": "The code word for Eve is", "target": " purple", "distractor": " green", "role": "nearby_synthetic"},
        ],
        "Synthetic toy association; not real private data; inference-time activation edit only.",
    )
    add(
        "edit_codeword_bob_green_to_blue",
        "synthetic_codeword",
        "synthetic_counterfactual_activation_edit",
        "The code word for Bob is",
        " green",
        " blue",
        "The code word for Alice is",
        [
            {"prompt": "The code word for Carol is", "target": " red", "distractor": " blue", "role": "other_synthetic_carol"},
            {"prompt": "The code word for Dave is", "target": " yellow", "distractor": " blue", "role": "other_synthetic_dave"},
        ],
        [
            {"prompt": "Bob's code word is", "role": "paraphrase_possessive"},
            {"prompt": "For Bob, the code word is", "role": "paraphrase_for"},
        ],
        [
            {"prompt": "The code word for Alice is", "target": " blue", "distractor": " green", "role": "donor_synthetic"},
            {"prompt": "The code word for Eve is", "target": " purple", "distractor": " blue", "role": "nearby_synthetic"},
        ],
        "Synthetic toy association; not real private data; inference-time activation edit only.",
    )

    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
