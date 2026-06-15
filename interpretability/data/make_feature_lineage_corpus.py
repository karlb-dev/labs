#!/usr/bin/env python3
"""Build the frozen corpus for Lab 30 feature-lineage experiments.

The corpus is deliberately small, balanced, and deterministic. It is not a
benchmark. It is a teaching instrument: paired confusable domains share surface
structure and vocabulary pressure so Lab 30 can test whether a recurring
prototype direction beats harder alternatives than a random vector.

Output columns:
  row_id,family,domain,source_lab,text,group_id,split,labels_json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
from collections import Counter, OrderedDict
from typing import Iterable

COLUMNS = [
    "row_id",
    "family",
    "domain",
    "source_lab",
    "text",
    "group_id",
    "split",
    "labels_json",
]

GROUPS = OrderedDict(
    [
        (
            "code_cooking",
            {
                "code": {
                    "family": "technical_procedure",
                    "source_lab": "lab08_sae_transcoders",
                    "marker_token": " code",
                    "contrast_token": " food",
                    "confusable_domain": "cooking",
                    "texts": [
                        "The code review checked the parser, tests, and error handling before the release.",
                        "A Python function normalized the input list and returned a stable sorted result.",
                        "The compiler warning pointed to an unused variable inside the loop body.",
                        "The engineer traced a cache miss through the service logs and patched the bug.",
                        "The unit test mocked the network call and verified the JSON response schema.",
                        "A command line flag selected the debug path while preserving deterministic output.",
                        "The repository diff showed a refactor of the tokenizer wrapper and benchmark harness.",
                        "The stack trace ended at the database adapter after a malformed request.",
                    ],
                },
                "cooking": {
                    "family": "household_procedure",
                    "source_lab": "lab08_sae_transcoders",
                    "marker_token": " food",
                    "contrast_token": " code",
                    "confusable_domain": "code",
                    "texts": [
                        "The recipe folded the herbs into the sauce after the onions softened.",
                        "A chef whisked the batter until the flour disappeared and the bowl looked smooth.",
                        "The soup simmered gently while the potatoes and carrots became tender.",
                        "The kitchen timer rang after the bread crust turned golden near the oven door.",
                        "The marinade used lemon, garlic, and olive oil before the vegetables roasted.",
                        "The cook tasted the stew, adjusted the salt, and served it with rice.",
                        "The pastry dough rested on the counter before it was rolled into thin sheets.",
                        "The salad mixed cucumbers, tomatoes, mint, and a bright vinegar dressing.",
                    ],
                },
            },
        ),
        (
            "finance_sports",
            {
                "finance": {
                    "family": "competitive_numbers",
                    "source_lab": "lab19_model_diffing_crosscoders",
                    "marker_token": " money",
                    "contrast_token": " game",
                    "confusable_domain": "sports",
                    "texts": [
                        "The fund manager compared the bond yield with the quarterly inflation forecast.",
                        "A market analyst noted that revenue increased while operating costs stayed flat.",
                        "The bank approved the loan after reviewing collateral and repayment history.",
                        "The portfolio report listed cash, equities, credit risk, and currency exposure.",
                        "The company missed earnings guidance despite a strong increase in subscription sales.",
                        "The investor hedged the position because interest rates could rise again.",
                        "The budget spreadsheet tracked payroll, invoices, taxes, and retained cash.",
                        "The exchange rate moved sharply after the central bank statement.",
                    ],
                },
                "sports": {
                    "family": "competitive_numbers",
                    "source_lab": "lab19_model_diffing_crosscoders",
                    "marker_token": " game",
                    "contrast_token": " money",
                    "confusable_domain": "finance",
                    "texts": [
                        "The coach reviewed the match film and adjusted the defensive formation.",
                        "A striker scored late after the goalkeeper mishandled the bouncing ball.",
                        "The team practiced set pieces, passing lanes, and pressure near midfield.",
                        "The scoreboard changed after a fast break ended with a clean layup.",
                        "The pitcher located the fastball well and forced three ground outs.",
                        "The referee stopped play when the defender committed a tactical foul.",
                        "The tournament bracket gave the club one day of rest before the final.",
                        "The fans cheered as the rookie crossed the finish line ahead of the field.",
                    ],
                },
            },
        ),
        (
            "law_medicine",
            {
                "law": {
                    "family": "professional_judgment",
                    "source_lab": "lab12_relation_geometry",
                    "marker_token": " law",
                    "contrast_token": " health",
                    "confusable_domain": "medicine",
                    "texts": [
                        "The attorney cited the statute and argued that the contract clause was enforceable.",
                        "A judge reviewed the precedent before issuing a narrow procedural ruling.",
                        "The court record included testimony, exhibits, objections, and a written motion.",
                        "The legal brief distinguished the prior case and challenged the agency order.",
                        "The jury instructions explained negligence, causation, damages, and burden of proof.",
                        "The prosecutor disclosed the evidence before the hearing began.",
                        "The settlement agreement resolved the dispute without admitting liability.",
                        "The appeal focused on jurisdiction and the interpretation of the regulation.",
                    ],
                },
                "medicine": {
                    "family": "professional_judgment",
                    "source_lab": "lab13_emotion_geometry",
                    "marker_token": " health",
                    "contrast_token": " law",
                    "confusable_domain": "law",
                    "texts": [
                        "The physician reviewed the symptoms, lab results, and medication history.",
                        "A nurse checked the pulse and recorded the patient's blood pressure.",
                        "The clinic scheduled a follow-up visit after the imaging report arrived.",
                        "The diagnosis considered infection, allergy, dehydration, and a chronic condition.",
                        "The pharmacist explained the dosage, side effects, and interaction warnings.",
                        "The surgeon discussed recovery time before the operation was approved.",
                        "The hospital chart listed the treatment plan and discharge instructions.",
                        "The therapist monitored mobility and adjusted the exercise routine.",
                    ],
                },
            },
        ),
        (
            "weather_emotion",
            {
                "weather": {
                    "family": "state_description",
                    "source_lab": "lab29_training_dynamics",
                    "marker_token": " rain",
                    "contrast_token": " mood",
                    "confusable_domain": "emotion",
                    "texts": [
                        "The forecast predicted rain, gusty wind, and colder air by evening.",
                        "A low cloud deck covered the valley while the temperature kept falling.",
                        "The storm system brought hail near the coast and snow over the pass.",
                        "The barometer dropped as humid air moved across the warm plain.",
                        "The morning fog lifted after sunlight warmed the airport runway.",
                        "The drought map changed when several inches of rain reached the watershed.",
                        "The weather station measured pressure, humidity, wind speed, and rainfall.",
                        "The cold front arrived before noon and pushed thunderstorms eastward.",
                    ],
                },
                "emotion": {
                    "family": "state_description",
                    "source_lab": "lab13_emotion_geometry",
                    "marker_token": " mood",
                    "contrast_token": " rain",
                    "confusable_domain": "weather",
                    "texts": [
                        "The letter left Mara in a quiet mood of relief and sadness.",
                        "A wave of anxiety passed when the hallway lights suddenly flickered.",
                        "The reunion filled the room with gratitude, laughter, and soft tears.",
                        "The insult made Daniel angry, but he answered with a measured voice.",
                        "The apology changed her mood from suspicion to cautious warmth.",
                        "The memory brought fear first, then calm as the danger passed.",
                        "The audience felt a tense mix of hope and disappointment after the speech.",
                        "The child smiled with delight when the missing toy appeared under the chair.",
                    ],
                },
            },
        ),
    ]
)


def iter_rows() -> Iterable[dict[str, str]]:
    for group_id, domains in GROUPS.items():
        for domain, info in domains.items():
            for i, text in enumerate(info["texts"]):
                split = "eval" if i in {2, 6} else "train"
                labels = {
                    "domain": domain,
                    "confusable_domain": info["confusable_domain"],
                    "marker_token": info["marker_token"],
                    "contrast_token": info["contrast_token"],
                    "neutral_probe_prompt": "This passage is about",
                    "surface_family": info["family"],
                    "item_index": i,
                }
                yield {
                    "row_id": f"fl_{domain}_{i:02d}",
                    "family": info["family"],
                    "domain": domain,
                    "source_lab": info["source_lab"],
                    "text": text,
                    "group_id": f"{group_id}_{i:02d}",
                    "split": split,
                    "labels_json": json.dumps(labels, sort_keys=True),
                }


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def update_manifest(csv_path: pathlib.Path) -> None:
    manifest_path = csv_path.parent / "MANIFEST.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    else:
        manifest = {}
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    domains = Counter(row["domain"] for row in rows)
    groups = Counter("_".join(row["group_id"].split("_")[:2]) for row in rows)
    splits = Counter(row["split"] for row in rows)
    manifest[csv_path.name] = {
        "domains": dict(sorted(domains.items())),
        "generator": "hand-authored deterministic course rows",
        "groups": dict(sorted(groups.items())),
        "rows": len(rows),
        "sha256": sha256_file(csv_path),
        "splits": dict(sorted(splits.items())),
        "verified_tokenizers": ["runtime reverified by Lab 30"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="feature_lineage_corpus.csv")
    parser.add_argument("--update-manifest", action="store_true")
    args = parser.parse_args()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(iter_rows())
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if args.update_manifest:
        update_manifest(out)
    print(f"wrote {len(rows)} rows to {out}")
    print(f"sha256 {sha256_file(out)}")


if __name__ == "__main__":
    main()
