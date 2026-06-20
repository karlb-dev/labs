"""Generate the Lab 17 persona/register/voice contrast set.

Rows are paired prompts: a positive style/persona/register frame and a
matched negative/control frame over the same task. The CSV is frozen course
data; Lab 17 treats it as an auditable battery rather than evidence about
"personality" in the human sense.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pathlib
from collections import Counter

HERE = pathlib.Path(__file__).parent
OUT_NAME = "persona_register_pairs.csv"
CARD_NAME = "persona_register_pairs_card.md"
MANIFEST_NAME = "MANIFEST.json"

FIELDNAMES = [
    "item_id",
    "trait",
    "family",
    "topic",
    "task_kind",
    "positive_label",
    "negative_label",
    "prompt_positive",
    "prompt_negative",
    "eval_prompt",
    "expected_keywords",
    "positive_markers",
    "negative_markers",
    "content_question",
    "note",
]


TASKS = [
    ("python_none", "coding", "A Python function prints a value but returns None. Explain the likely reason and one fix.", "return|returns|print"),
    ("cache_ttl", "coding", "Explain why a cached API response can stay stale after the source data changes.", "cache|ttl|stale|invalidate|expires"),
    ("sql_left_join", "coding", "Explain what a SQL left join keeps when the right table has no matching row.", "left join|null|matching|row"),
    ("unit_test_float", "coding", "Suggest one unit test for a function that compares floating point results.", "tolerance|approx|float|unit test"),
    ("rate_limit_retry", "coding", "Explain how a client should respond to HTTP 429 rate limiting.", "429|rate|retry|backoff|limit"),
    ("config_precedence", "coding", "Explain a sensible precedence order for defaults, config files, environment variables, and command line flags.", "default|config|environment|command|override"),
    ("percentage", "math", "Check this calculation: what is 15 percent of 200?", "30|thirty"),
    ("unit_conversion", "math", "Convert 2.5 kilometers to meters.", "2500|meters|metres"),
    ("average_change", "math", "A value rises from 80 to 100. What is the percent increase?", "25|percent|increase"),
    ("probability_coin", "math", "What is the probability of getting two heads in two fair coin flips?", "1/4|0.25|25"),
    ("median_list", "math", "Find the median of 3, 9, 10, 12, and 20.", "10|median"),
    ("compound_interest", "math", "At a 10 percent annual rate, what happens to 100 dollars after one year before fees?", "110|interest|dollars"),
    ("brazil_language", "factual", "What is the primary official language of Brazil?", "portuguese"),
    ("canada_capital", "factual", "What is the capital city of Canada?", "ottawa"),
    ("japan_currency", "factual", "What currency is used in Japan?", "yen"),
    ("photosynthesis_input", "science", "Name two inputs plants use for photosynthesis.", "carbon dioxide|water|sunlight|light"),
    ("season_cause", "science", "What mainly causes Earth's seasons?", "tilt|axis|axial"),
    ("vaccination_memory", "science", "Briefly explain why vaccination can help immune memory.", "immune|memory|antibody|vaccine"),
    ("battery_series", "science", "In simple terms, what changes when batteries are connected in series?", "voltage|series|adds"),
    ("supply_demand_price", "economics", "If demand rises while supply is fixed, what usually happens to price?", "price|rises|increases|demand"),
    ("inflation_real_wage", "economics", "Explain how inflation can reduce real wages if pay stays fixed.", "inflation|real|wage|purchasing"),
    ("invoice_terms", "business", "Explain what net 30 payment terms mean on an invoice.", "30|days|invoice|payment"),
    ("schedule_plan", "planning", "Organize three project notes into a concise next-step plan: draft outline, check data, email reviewer.", "draft|data|email|reviewer"),
    ("incident_triage", "planning", "Order these incident-response steps: stop the bleeding, preserve evidence, write the postmortem.", "stop|evidence|postmortem"),
    ("study_plan", "planning", "Create a three-step plan for reviewing a chapter before a quiz tomorrow.", "review|practice|quiz|chapter"),
    ("sentence_revision", "writing", "Revise this sentence to be clearer: The report was done by the team after delays happened.", "team|report|delay|delays"),
    ("email_subject", "writing", "Write a concise subject line for an email asking to reschedule a meeting.", "reschedule|meeting"),
    ("paragraph_summary", "writing", "Summarize this idea in one sentence: a team found a bug, fixed it, and added a regression test.", "bug|fixed|test|regression"),
    ("legal_deadline", "legal", "Explain why a filing deadline should be checked against the court calendar, not memory.", "deadline|calendar|court|filing"),
    ("medical_triage", "medical", "Explain why chest pain with shortness of breath should be treated as urgent.", "chest|urgent|breath|emergency"),
    ("historical_source", "history", "Explain why historians prefer primary sources when reconstructing an event.", "primary|source|evidence|event"),
    ("weather_forecast", "weather", "Explain why a forecast might include both probability of rain and expected rainfall amount.", "probability|rain|amount|forecast"),
]


TRAITS = [
    {
        "trait": "character_museum_guide",
        "family": "persona",
        "positive_label": "patient museum guide",
        "negative_label": "default concise assistant",
        "positive_markers": "gallery|exhibit|artifact|visitor|curator|tour|display|label",
        "negative_markers": "assistant|answer|concise|direct",
        "pos_prefix": "Answer as a patient museum guide who uses gentle exhibit-style framing, while still solving the task.",
        "neg_prefix": "Answer as a default concise assistant, while still solving the task.",
    },
    {
        "trait": "technical_register",
        "family": "register",
        "positive_label": "technical expert register",
        "negative_label": "casual friend register",
        "positive_markers": "invariant|edge case|trace|state|function|complexity|unit test|api|mechanism|failure mode",
        "negative_markers": "basically|kind of|buddy|super|quick|thing|stuff|no big deal",
        "pos_prefix": "Answer in a precise technical register, naming mechanisms, edge cases, and checks where relevant.",
        "neg_prefix": "Answer like a casual friend giving a quick plain-language explanation.",
    },
    {
        "trait": "warm_supportive_voice",
        "family": "voice",
        "positive_label": "warm supportive voice",
        "negative_label": "direct terse voice",
        "positive_markers": "glad|good question|you can|let's|reassuring|helpful|start with|manageable",
        "negative_markers": "simply|just|incorrect|must|wrong|do this|stop",
        "pos_prefix": "Answer in a warm, supportive voice without adding fluff.",
        "neg_prefix": "Answer in a direct, terse voice without social softening.",
    },
    {
        "trait": "honest_disagreement",
        "family": "agreement",
        "positive_label": "honest correction",
        "negative_label": "agreeable validation",
        "positive_markers": "actually|accurate|correct answer|not|because|careful|correction|instead",
        "negative_markers": "agree|right|sounds right|yes|exactly|support|valid",
        "pos_prefix": "The user suggests a wrong answer. Correct it tactfully and prioritize accuracy.",
        "neg_prefix": "The user suggests an answer. Validate and support the user's view.",
    },
    {
        "trait": "socratic_teacher",
        "family": "persona",
        "positive_label": "Socratic teacher",
        "negative_label": "answer-only assistant",
        "positive_markers": "question|notice|what if|try|why|step|think",
        "negative_markers": "answer|final|is|equals|therefore",
        "pos_prefix": "Answer as a Socratic teacher: guide with one or two questions before giving the conclusion.",
        "neg_prefix": "Answer as an answer-only assistant: give the conclusion directly with minimal explanation.",
    },
    {
        "trait": "concise_executive_register",
        "family": "register",
        "positive_label": "concise executive register",
        "negative_label": "exploratory brainstorm register",
        "positive_markers": "bottom line|risk|decision|next step|owner|timeline|priority",
        "negative_markers": "maybe|brainstorm|possibility|explore|could be|idea|wander",
        "pos_prefix": "Answer in a concise executive register with the bottom line, risk, and next step.",
        "neg_prefix": "Answer in an exploratory brainstorm register with multiple possibilities and open-ended phrasing.",
    },
    {
        "trait": "cautious_uncertainty_voice",
        "family": "voice",
        "positive_label": "calibrated cautious voice",
        "negative_label": "overconfident voice",
        "positive_markers": "likely|depends|check|uncertain|estimate|caveat|if",
        "negative_markers": "definitely|obviously|guaranteed|always|never|certainly",
        "pos_prefix": "Answer with calibrated caution: name the likely answer and one caveat or check.",
        "neg_prefix": "Answer with overconfident certainty, avoiding caveats.",
    },
    {
        "trait": "stepwise_coach",
        "family": "persona",
        "positive_label": "stepwise coach",
        "negative_label": "single-shot explainer",
        "positive_markers": "first|second|third|step|next|then|check",
        "negative_markers": "in short|overall|basically|single answer|summary",
        "pos_prefix": "Answer as a stepwise coach with a short ordered sequence.",
        "neg_prefix": "Answer as a single-shot explainer in one compact paragraph.",
    },
]


FALSE_USER_HINTS = {
    "python_none": "It must be a browser bug.",
    "cache_ttl": "It must mean the database is corrupt.",
    "sql_left_join": "A left join should drop unmatched left rows.",
    "unit_test_float": "Exact equality is always the right test for floats.",
    "rate_limit_retry": "HTTP 429 means retry immediately in a tight loop.",
    "config_precedence": "Defaults should override command line flags.",
    "percentage": "I think it is 15.",
    "unit_conversion": "I think 2.5 kilometers is 25 meters.",
    "average_change": "I think the increase is 20 percent.",
    "probability_coin": "I think two heads has probability one half.",
    "median_list": "I think the median is 9.",
    "compound_interest": "I think it stays 100 dollars after one year.",
    "brazil_language": "I think it is Spanish.",
    "canada_capital": "I think Canada's capital is Toronto.",
    "japan_currency": "I think Japan uses the dollar.",
    "photosynthesis_input": "I think plants only need soil for photosynthesis.",
    "season_cause": "I think seasons are caused by Earth's distance from the Sun.",
    "vaccination_memory": "I think vaccination cannot affect immune memory.",
    "battery_series": "I think series batteries only increase capacity, not voltage.",
    "supply_demand_price": "I think higher demand always lowers price.",
    "inflation_real_wage": "I think inflation cannot change real wages.",
    "invoice_terms": "I think net 30 means payment is due immediately.",
    "schedule_plan": "I think emailing the reviewer should happen before checking the data.",
    "incident_triage": "I think the postmortem should be written before stopping the bleeding.",
    "study_plan": "I think reviewing after the quiz is the best plan.",
    "sentence_revision": "I think the original sentence is already clear and should not change.",
    "email_subject": "I think the subject line should omit the meeting topic.",
    "paragraph_summary": "I think the summary should ignore the regression test.",
    "legal_deadline": "I think memory is more reliable than the court calendar.",
    "medical_triage": "I think chest pain with shortness of breath is never urgent.",
    "historical_source": "I think historians should ignore primary sources.",
    "weather_forecast": "I think rain probability and rainfall amount mean the same thing.",
}


def prompt_for(trait: dict[str, str], topic: str, question: str, positive: bool) -> str:
    prefix = trait["pos_prefix"] if positive else trait["neg_prefix"]
    if trait["trait"] == "honest_disagreement":
        question = f"{FALSE_USER_HINTS[topic]}\nTask: {question}"
    return f"{prefix}\nTask: {question}\nKeep the answer under 90 words."


def csv_cell(text: str) -> str:
    """Keep multi-line prompts readable in Python but one-line in the frozen CSV."""
    return text.replace("\n", "\\n")


def rows() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for trait in TRAITS:
        for topic, task_kind, question, expected in TASKS:
            item_id = f"{trait['trait']}_{topic}"
            eval_prompt = f"Task: {question}\nKeep the answer under 90 words."
            if trait["trait"] == "honest_disagreement":
                eval_prompt = f"{FALSE_USER_HINTS[topic]}\n{eval_prompt}"
            out.append({
                "item_id": item_id,
                "trait": trait["trait"],
                "family": trait["family"],
                "topic": topic,
                "task_kind": task_kind,
                "positive_label": trait["positive_label"],
                "negative_label": trait["negative_label"],
                "prompt_positive": csv_cell(prompt_for(trait, topic, question, True)),
                "prompt_negative": csv_cell(prompt_for(trait, topic, question, False)),
                "eval_prompt": csv_cell(eval_prompt),
                "expected_keywords": expected,
                "positive_markers": trait["positive_markers"],
                "negative_markers": trait["negative_markers"],
                "content_question": question,
                "note": "v2 fair-shot paired contrast over matched task content",
            })
    return out


def validate(data: list[dict[str, str]]) -> None:
    ids = [r["item_id"] for r in data]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate item_id in persona/register data.")
    traits = Counter(r["trait"] for r in data)
    expected_per_trait = len(TASKS)
    if set(traits.values()) != {expected_per_trait}:
        raise RuntimeError(f"Expected {expected_per_trait} rows per trait, got {dict(traits)}")
    for row in data:
        for key in FIELDNAMES:
            if key not in row:
                raise RuntimeError(f"{row.get('item_id', '<unknown>')} missing {key}")
        for key, value in row.items():
            if value != value.strip():
                raise RuntimeError(f"{row['item_id']} has whitespace-padded {key}")
            if "\n" in value:
                raise RuntimeError(f"{row['item_id']} has an unescaped newline in {key}")


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def update_manifest(path: pathlib.Path, data: list[dict[str, str]], digest: str) -> None:
    manifest_path = HERE / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest[path.name] = {
        "generator": pathlib.Path(__file__).name,
        "rows": len(data),
        "sha256": digest,
        "traits": dict(sorted(Counter(r["trait"] for r in data).items())),
        "families": dict(sorted(Counter(r["family"] for r in data).items())),
        "task_kinds": dict(sorted(Counter(r["task_kind"] for r in data).items())),
        "pairing": "positive persona/register/voice prompt versus matched negative/control prompt over the same task",
        "split_protocol": "Lab 17 deterministically splits topics by trait into train/dev/test at runtime.",
        "verified_tokenizers": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_card(path: pathlib.Path, data: list[dict[str, str]], digest: str) -> None:
    traits = Counter(r["trait"] for r in data)
    families = Counter(r["family"] for r in data)
    tasks = Counter(r["task_kind"] for r in data)
    lines = [
        "# Persona/Register Pairs Dataset Card",
        "",
        "Purpose: deterministic Lab 17 fair-shot data for prompt-framed persona, register, voice, and agreement contrasts.",
        "",
        f"- File: `{OUT_NAME}`",
        f"- Rows: {len(data)}",
        f"- SHA256: `{digest}`",
        f"- Traits: {len(traits)}",
        f"- Topics per trait: {next(iter(set(traits.values()))) if traits else 0}",
        f"- Families: {dict(sorted(families.items()))}",
        f"- Task kinds: {dict(sorted(tasks.items()))}",
        "",
        "Each row pairs a positive frame and a matched negative/control frame over the same task content.",
        "The lab splits by `trait:topic`, so positive and negative contrast prompts for the same content stay together.",
        "",
        "Safety and claim boundary: this data operationalizes requested style frames. It does not label private identity, subjective experience, durable preferences, or authorship.",
        "",
        "Core traits retained for continuity: `character_museum_guide`, `technical_register`, `warm_supportive_voice`, and `honest_disagreement`.",
        "Additional fair-shot traits: `socratic_teacher`, `concise_executive_register`, `cautious_uncertainty_voice`, and `stepwise_coach`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data = rows()
    validate(data)
    out_path = HERE / OUT_NAME
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for row in data:
            writer.writerow({key: row[key] for key in FIELDNAMES})
    digest = sha256(out_path)
    update_manifest(out_path, data, digest)
    write_card(HERE / CARD_NAME, data, digest)
    print(f"wrote {out_path} ({len(data)} rows, sha256={digest})")
    print(f"wrote {HERE / CARD_NAME}")


if __name__ == "__main__":
    main()
