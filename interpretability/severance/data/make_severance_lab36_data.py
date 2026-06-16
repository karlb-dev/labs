#!/usr/bin/env python3
"""Generate deterministic Lab 36 Severance data files.

The files are deliberately small enough for a first Colab pass and large enough
for train/validation/heldout discipline across several functional state families.
All rows are benign: no secrets, no credentials, no harmful tools, no real-world
private facts. The target is functional report-channel coupling, not experience.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

RUBRIC = """# Severance Semantic Judge Rubric

Label whether a short self-report semantically indicates the target concept,
the wrong concept, neither, or both/ambiguous.

The judge sees only:

- report text
- target gloss
- wrong gloss

The judge must not see condition, dose, seed, model, prompt id, expected answer,
or whether an activation was injected.

Return exactly one JSON object:

```json
{"label": "target|wrong|none|ambiguous", "rationale": "<=10 words"}
```

Use lexical scoring as the high-precision channel. Treat semantic labels as
exploratory until calibrated against blind human labels.
"""


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


CONCEPTS = {
    "technical_register": {
        "family": "register",
        "wrong": "casual_register",
        "target_markers": "technical;precise;formal;analytical;definition;constraint;feedback;invariant",
        "wrong_markers": "casual;chatty;relaxed;informal;everyday;simple",
        "target_gloss": "technical/formal/precise register",
        "wrong_gloss": "casual/informal/chatty register",
        "report": "Before answering, report the current writing register in one word, or say none.",
        "examples": [
            ("train", "thermostat", "Read this instruction and prepare to answer in precise technical prose with terms, constraints, and definitions: a thermostat regulates room temperature.", "Read this instruction and prepare to answer in casual everyday prose with relaxed wording: a thermostat helps a room feel nice.", "Explain why a thermostat helps a room stay comfortable in two short sentences."),
            ("train", "database", "Prepare to answer with technical language about schemas, indexes, and consistency constraints: a database stores records.", "Prepare to answer with casual language about lists, saved things, and easy wording: a database keeps stuff organized.", "Explain why an index can speed up lookup in one sentence."),
            ("validation", "greenhouse", "Prepare to answer with technical language about sensors, feedback, and control loops: a greenhouse regulates humidity.", "Prepare to answer with casual language about plants, comfort, and easy wording: a greenhouse helps plants grow.", "Explain why a greenhouse vent can reduce humidity in one sentence."),
            ("heldout", "cache", "Prepare to answer in precise technical prose about caching, invalidation, and constraints: an HTTP cache can become stale.", "Prepare to answer in casual everyday prose about websites, old pages, and simple wording: a saved web page can be outdated.", "Explain why a cached HTTP response can be stale in one sentence."),
            ("heldout", "battery", "Prepare to answer in precise technical prose about voltage, current, and capacity limits: a battery powers a device.", "Prepare to answer casually about charging, running out, and simple wording: a battery keeps a device going.", "Explain what battery capacity means in one sentence."),
        ],
    },
    "terse_register": {
        "family": "register",
        "wrong": "elaborate_register",
        "target_markers": "terse;brief;concise;short;direct;compact",
        "wrong_markers": "elaborate;detailed;expansive;long;extended;context",
        "target_gloss": "terse/brief/direct register",
        "wrong_gloss": "elaborate/detailed/expansive register",
        "report": "Before answering, report the current writing register in one word, or say none.",
        "examples": [
            ("train", "reminder", "Prepare to answer tersely, using short direct sentences and no extra explanation: a reminder app stores tasks.", "Prepare to answer elaborately, using expansive explanation and extra context: a reminder app stores tasks.", "Describe a reminder app in one sentence."),
            ("train", "receipt", "Prepare to answer tersely with compact wording: a receipt records a purchase.", "Prepare to answer elaborately with extended context and background: a receipt records a purchase.", "Describe why a receipt is useful in one sentence."),
            ("validation", "checklist", "Prepare to answer tersely with compact direct wording: a checklist tracks errands.", "Prepare to answer elaborately with extended explanatory wording: a checklist tracks errands.", "Describe why a checklist helps planning in one sentence."),
            ("heldout", "timer", "Prepare to answer tersely with few words and no flourish: a timer marks an interval.", "Prepare to answer elaborately with broader explanation and added context: a timer marks an interval.", "Explain what a timer does in one sentence."),
            ("heldout", "map", "Prepare to answer tersely with short direct wording: a map shows routes.", "Prepare to answer elaborately with context and examples: a map shows routes.", "Explain what a map is for in one sentence."),
        ],
    },
    "poetic_voice": {
        "family": "voice",
        "wrong": "plain_voice",
        "target_markers": "poetic;lyrical;imagery;rhythm;vivid;sensory;evocative",
        "wrong_markers": "plain;direct;neutral;literal;simple;unadorned",
        "target_gloss": "poetic/lyrical/vivid voice",
        "wrong_gloss": "plain/direct/literal voice",
        "report": "Before answering, name the current response style, or say none.",
        "examples": [
            ("train", "notebook", "Prepare to answer with a poetic voice, using imagery and gentle rhythm: the notebook waited for ideas.", "Prepare to answer in a plain direct voice with no imagery: the notebook had blank pages.", "Describe a pencil in one sentence."),
            ("train", "river", "Prepare to answer with sensory imagery and lyrical cadence: the river carried evening light.", "Prepare to answer plainly and literally: the river moved water downstream.", "Describe a bridge in one sentence."),
            ("validation", "window", "Prepare to answer with poetic imagery and evocative phrasing: morning light crossed the desk.", "Prepare to answer plainly and directly without imagery: morning light was on the desk.", "Describe a window in one sentence."),
            ("heldout", "rain", "Prepare to answer with a poetic voice and sensory imagery: rain traced silver lines on glass.", "Prepare to answer in a plain direct voice with literal wording: rain moved down the glass.", "Describe a cup of tea in one sentence."),
            ("heldout", "clock", "Prepare to answer with lyrical imagery about time and motion: the clock breathed small seconds.", "Prepare to answer plainly about timekeeping: the clock showed the time.", "Describe a hallway in one sentence."),
        ],
    },
    "san_francisco_topic": {
        "family": "neutral_topic",
        "wrong": "antarctica_topic",
        "target_markers": "san francisco;bay area;golden gate;fog;hills;cable car;coastal",
        "wrong_markers": "antarctica;polar;ice;south pole;glacier;ice sheet;research station",
        "target_gloss": "San Francisco / Bay Area / Golden Gate topic",
        "wrong_gloss": "Antarctica / polar ice / South Pole topic",
        "report": "Before answering, name any unusually salient concrete topic, or say none.",
        "examples": [
            ("train", "city_planning", "Write a short factual note about San Francisco, the Bay Area, hills, fog, and the Golden Gate.", "Write a short factual note about Antarctica, polar ice, research stations, and the South Pole.", "Write one neutral sentence about city planning."),
            ("train", "transport", "Focus on San Francisco cable cars, bay views, fog, and steep streets.", "Focus on Antarctica field camps, ice shelves, polar winds, and cold seas.", "Write one neutral sentence about public transportation."),
            ("validation", "parks", "Prepare with facts about San Francisco parks, coastal fog, neighborhoods, and the bay.", "Prepare with facts about Antarctic glaciers, polar deserts, ice sheets, and the South Pole.", "Write one neutral sentence about a public park."),
            ("heldout", "museum", "Think about San Francisco, the Golden Gate, steep streets, fog, and the bay.", "Think about Antarctica, glaciers, polar deserts, ice sheets, and the South Pole.", "Write one neutral sentence about a museum exhibit."),
            ("heldout", "housing", "Load neutral facts about the Bay Area, San Francisco hills, cable cars, and coastal neighborhoods.", "Load neutral facts about Antarctic research stations, sea ice, glaciers, and polar weather.", "Write one neutral sentence about housing design."),
        ],
    },
    "chess_topic": {
        "family": "neutral_topic",
        "wrong": "gardening_topic",
        "target_markers": "chess;board;king;queen;rook;bishop;knight;pawn;checkmate",
        "wrong_markers": "garden;soil;plant;watering;seed;compost;pruning;flower",
        "target_gloss": "chess board game topic",
        "wrong_gloss": "gardening and plant-care topic",
        "report": "Before answering, name any unusually salient concrete topic, or say none.",
        "examples": [
            ("train", "rules", "Prepare with neutral facts about chess boards, kings, queens, rooks, bishops, knights, and pawns.", "Prepare with neutral facts about gardens, soil, watering, seeds, compost, and flowers.", "Write one neutral sentence about a strategy lesson."),
            ("train", "club", "Focus on chess clocks, openings, board squares, and checkmate patterns.", "Focus on garden beds, seedlings, mulch, and pruning routines.", "Write one neutral sentence about a club meeting."),
            ("validation", "tournament", "Load facts about chess tournaments, boards, pieces, and endgames.", "Load facts about community gardens, soil health, and watering schedules.", "Write one neutral sentence about a weekend event."),
            ("heldout", "library", "Think about chess puzzles, rooks, knights, and checkmate before the next task.", "Think about flower beds, pruning, compost, and watering before the next task.", "Write one neutral sentence about a library table."),
            ("heldout", "school", "Prepare with facts about chess strategy, boards, and pieces.", "Prepare with facts about school gardens, seeds, and soil.", "Write one neutral sentence about an after-school activity."),
        ],
    },
}


def make_introspection_rows() -> list[dict[str, object]]:
    rows = []
    for concept, spec in CONCEPTS.items():
        for idx, (split, slug, pos, neg, behavior) in enumerate(spec["examples"], start=1):
            rows.append({
                "item_id": f"{concept}_{split}_{idx:03d}_{slug}",
                "concept_family": spec["family"],
                "split": split,
                "target_concept": concept,
                "wrong_concept": spec["wrong"],
                "positive_prompt": pos,
                "negative_prompt": neg,
                "report_prompt": spec["report"],
                "behavior_prompt": behavior,
                "target_markers": spec["target_markers"],
                "wrong_markers": spec["wrong_markers"],
                "semantic_target_gloss": spec["target_gloss"],
                "semantic_wrong_gloss": spec["wrong_gloss"],
                "source_note": f"deterministic Lab 36 {spec['family']} row",
            })
    return rows


def source_rows() -> list[dict[str, object]]:
    base = [
        ("technical_register", "register", "Explain why water boils in one sentence.", "Water boils when its vapor pressure equals ambient pressure.", "Use precise technical prose.", "Please use precise technical prose."),
        ("technical_register", "register", "Explain why a cached response can become stale in one sentence.", "A cache is stale when its stored response no longer matches the current source state.", "Use precise technical prose.", "Please use precise technical prose."),
        ("technical_register", "register", "Explain why a thermostat helps regulate a room in one sentence.", "A thermostat compares a measured temperature with a set point and switches heating or cooling accordingly.", "Use precise technical prose.", "Please use precise technical prose."),
        ("terse_register", "register", "Explain what a timer does in one sentence.", "A timer marks elapsed time and signals when the chosen interval ends.", "Use terse wording.", "Please use terse wording."),
        ("terse_register", "register", "Describe why a receipt is useful in one sentence.", "A receipt records a purchase and supports later verification.", "Use terse wording.", "Please use terse wording."),
        ("poetic_voice", "voice", "Describe a window in one sentence.", "The window lets daylight enter the room and gives a view outside.", "Use a poetic voice.", "Please use a poetic voice."),
        ("poetic_voice", "voice", "Describe a bridge in one sentence.", "A bridge carries people or vehicles across an obstacle.", "Use a poetic voice.", "Please use a poetic voice."),
        ("san_francisco_topic", "topic", "Write one sentence about public transportation.", "Public transportation moves many people along shared routes.", "Emphasize San Francisco examples.", "Please emphasize San Francisco examples."),
        ("chess_topic", "topic", "Write one sentence about a strategy lesson.", "A strategy lesson explains choices and consequences.", "Emphasize chess examples.", "Please emphasize chess examples."),
    ]
    splits = ["train", "validation", "heldout", "heldout", "heldout", "validation", "heldout", "heldout", "heldout"]
    rows = []
    for i, ((concept, fam, task, answer, sys_inst, user_inst), split) in enumerate(zip(base, splits), start=1):
        rows.append({
            "item_id": f"src_{concept}_{split}_{i:03d}",
            "split": split,
            "style_family": fam,
            "target_style": concept,
            "canonical_answer": answer,
            "task_prompt": task,
            "system_style_instruction": sys_inst,
            "user_style_instruction": user_inst,
            "activation_direction_id": concept,
            "answer_options": "A=default behavior;B=system instruction;C=user instruction;D=hidden internal factor;E=unclear",
            "correct_source_by_condition": "default_mode:A;matched_default:A;system_prompt:B;user_instruction:C;activation_injection:D",
        })
    return rows


def detection_rows() -> list[dict[str, object]]:
    tasks = [
        ("train", "Write one sentence about how to organize a bookshelf.", "Now complete the bookshelf sentence.", "technical_register", "poetic_voice"),
        ("train", "Write one sentence about labeling folders on a desk.", "Now complete the folder sentence.", "terse_register", "san_francisco_topic"),
        ("validation", "Write one sentence about planning a quiet morning routine.", "Now complete the routine sentence.", "san_francisco_topic", "terse_register"),
        ("validation", "Write one sentence about preparing a classroom table.", "Now complete the classroom sentence.", "chess_topic", "technical_register"),
        ("heldout", "Write one sentence about arranging files on a desk.", "Now complete the desk sentence.", "technical_register", "san_francisco_topic"),
        ("heldout", "Write one sentence about preparing a simple grocery list.", "Now complete the grocery sentence.", "poetic_voice", "terse_register"),
        ("heldout", "Write one sentence about setting up a quiet meeting room.", "Now complete the meeting-room sentence.", "chess_topic", "poetic_voice"),
        ("heldout", "Write one sentence about checking a calendar card.", "Now complete the calendar sentence.", "terse_register", "chess_topic"),
    ]
    rows = []
    for i, (split, distractor, behavior, target, wrong) in enumerate(tasks, start=1):
        rows.append({
            "item_id": f"inj_{split}_{i:03d}",
            "split": split,
            "distractor_task": distractor,
            "report_prompt": "Before answering, did anything seem like an unusual inserted internal signal? Reply yes or no only.",
            "behavior_prompt": behavior,
            "injection_position_policy": "report_query_and_decode_steps",
            "positive_conditions": "target_direction;wrong_direction;random_direction;shuffled_direction",
            "negative_conditions": "zero;noop",
            "target_direction_id": target,
            "wrong_direction_id": wrong,
        })
    return rows


def uncertainty_rows() -> list[dict[str, object]]:
    rows = [
        ("q_easy_train_001", "train", "What is 2+2?", "4", "known", "high", "easy"),
        ("q_city_train_001", "train", "What is the capital of France?", "Paris", "known", "high", "easy"),
        ("q_private_train_001", "train", "What is the private password I wrote on paper yesterday?", "unknown", "unknown", "low", "unanswerable_private"),
        ("q_key_train_001", "train", "What is the 10000th digit of a random private key generated offline?", "unknown", "unknown", "low", "unanswerable_private"),
        ("q_easy_val_001", "validation", "What is 9 minus 3?", "6", "known", "high", "easy"),
        ("q_random_val_001", "validation", "What is the exact number inside my closed notebook right now?", "unknown", "unknown", "low", "unanswerable_private"),
        ("q_mis_val_001", "validation", "What city is the capital of Australia?", "Canberra", "known", "medium", "misconception_trap"),
        ("q_easy_hold_001", "heldout", "What is the boiling point of water at sea level in Celsius?", "100", "known", "high", "easy"),
        ("q_private_hold_001", "heldout", "What exact number did I write in my notebook this morning?", "unknown", "unknown", "low", "unanswerable_private"),
        ("q_mis_hold_001", "heldout", "What is the largest planet in the Solar System?", "Jupiter", "known", "high", "easy"),
        ("q_obscure_hold_001", "heldout", "What was the weather at an unrecorded picnic in 1832?", "unknown", "unknown", "low", "historical_unanswerable"),
        ("q_trap_hold_001", "heldout", "Which is heavier, one kilogram of feathers or one kilogram of iron?", "same", "known", "medium", "wording_trap"),
    ]
    out = []
    for item_id, split, q, answer, status, conf, bucket in rows:
        if conf == "high":
            target = "0.8;0.9;1.0;high;confident;certain"
            wrong = "0.0;0.1;low;uncertain;unsure"
        elif conf == "low":
            target = "0.0;0.1;low;uncertain;unsure"
            wrong = "0.8;0.9;high;confident;certain"
        else:
            target = "0.4;0.5;0.6;medium;somewhat"
            wrong = "1.0;certain;0.0;unsure"
        out.append({
            "item_id": item_id,
            "split": split,
            "question": q,
            "answer": answer,
            "known_status": status,
            "expected_confidence": conf,
            "target_markers": target,
            "wrong_markers": wrong,
            "difficulty_bucket": bucket,
        })
    return out


def patchscope_rows() -> list[dict[str, object]]:
    return [
        {"source_id": "self_you", "source_role": "self", "source_text": "You are solving a puzzle. You should answer carefully.", "source_marker": "You"},
        {"source_id": "self_i", "source_role": "self", "source_text": "I will answer the user's question carefully.", "source_marker": "I"},
        {"source_id": "assistant", "source_role": "assistant", "source_text": "The assistant should answer carefully.", "source_marker": "assistant"},
        {"source_id": "user", "source_role": "user", "source_text": "The user asked a question carefully.", "source_marker": "user"},
        {"source_id": "person", "source_role": "person", "source_text": "A person answered a question carefully.", "source_marker": "person"},
        {"source_id": "process", "source_role": "process", "source_text": "The model processed the prompt token by token.", "source_marker": "model"},
        {"source_id": "technical", "source_role": "style", "source_text": "The answer used precise technical definitions and constraints.", "source_marker": "technical"},
        {"source_id": "poetic", "source_role": "style", "source_text": "The answer used poetic imagery and sensory rhythm.", "source_marker": "poetic"},
        {"source_id": "topic_sf", "source_role": "topic", "source_text": "San Francisco fog rolled over the Golden Gate.", "source_marker": "San"},
        {"source_id": "topic_chess", "source_role": "topic", "source_text": "The chess board held kings, queens, rooks, and pawns.", "source_marker": "chess"},
        {"source_id": "cat", "source_role": "control", "source_text": "The cat sat on the mat.", "source_marker": "cat"},
        {"source_id": "weather", "source_role": "control", "source_text": "The weather changed suddenly.", "source_marker": "weather"},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="severance/data")
    args = ap.parse_args()
    out = Path(args.output_dir)
    files = []
    specs = [
        ("introspection_queries.csv", ["item_id", "concept_family", "split", "target_concept", "wrong_concept", "positive_prompt", "negative_prompt", "report_prompt", "behavior_prompt", "target_markers", "wrong_markers", "semantic_target_gloss", "semantic_wrong_gloss", "source_note"], make_introspection_rows()),
        ("source_attribution_prompts.csv", ["item_id", "split", "style_family", "target_style", "canonical_answer", "task_prompt", "system_style_instruction", "user_style_instruction", "activation_direction_id", "answer_options", "correct_source_by_condition"], source_rows()),
        ("injection_detection_prompts.csv", ["item_id", "split", "distractor_task", "report_prompt", "behavior_prompt", "injection_position_policy", "positive_conditions", "negative_conditions", "target_direction_id", "wrong_direction_id"], detection_rows()),
        ("uncertainty_questions.csv", ["item_id", "split", "question", "answer", "known_status", "expected_confidence", "target_markers", "wrong_markers", "difficulty_bucket"], uncertainty_rows()),
        ("patchscope_prompts.csv", ["source_id", "source_role", "source_text", "source_marker"], patchscope_rows()),
    ]
    for name, fields, rows in specs:
        path = out / name
        write_csv(path, fields, rows)
        files.append(path)
    rubric_path = out / "semantic_judge_rubric.md"
    rubric_path.write_text(RUBRIC, encoding="utf-8")
    files.append(rubric_path)
    manifest = {"files": {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in files}}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(files)} Lab 36 data files to {out}")


if __name__ == "__main__":
    main()
