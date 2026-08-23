#!/usr/bin/env python3
"""Build mssql_copilot_chat_completions.jsonl from trace-derived and synthetic examples."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCHEMAS = ROOT / "schemas"
OUT = ROOT / "mssql_copilot_chat_completions.jsonl"

DATASET = "mssql-copilot-chat-completions"
DATASET_VERSION = "1.1.0"

MODE_INTENT = "intent (return complete query)"
MODE_CONT = "continuation (return one unit)"

SCHEMA_FILES = {
    "fitnessapp_test": "fitnessapp_test.txt",
    "ninjadb_a": "ninjadb_a.txt",
    "ninjadb_b": "ninjadb_b.txt",
    "master_onprem": "master_onprem.txt",
    "master_onprem_system": "master_onprem_system.txt",
    "azure_master": "azure_master.txt",
    "msdb_onprem": "msdb_onprem.txt",
    "adventure_azure": "adventure_azure.txt",
    "fabric_warehouse": "fabric_warehouse.txt",
}

SCHEMA_META = {
    "fitnessapp_test": {
        "connection_label": "4fc2b4c8d44d / FitnessApp_Test",
        "database": "FitnessApp_Test",
        "engine": "SQL Server Enterprise/Developer",
        "default_schema": "dbo",
        "schema_size": "small",
        "profile": "balanced",
    },
    "ninjadb_a": {
        "connection_label": "sqlninja / ninjadb",
        "database": "ninjadb",
        "engine": "Azure SQL Database",
        "default_schema": "dbo",
        "schema_size": "medium",
        "profile": "balanced",
    },
    "ninjadb_b": {
        "connection_label": "sqlninja / ninjadb",
        "database": "ninjadb",
        "engine": "Azure SQL Database",
        "default_schema": "dbo",
        "schema_size": "medium",
        "profile": "balanced",
    },
    "master_onprem": {
        "connection_label": "4fc2b4c8d44d / master",
        "database": "master",
        "engine": "SQL Server Enterprise/Developer",
        "default_schema": "dbo",
        "schema_size": "small",
        "profile": "balanced",
    },
    "master_onprem_system": {
        "connection_label": "4fc2b4c8d44d / master",
        "database": "master",
        "engine": "SQL Server Enterprise/Developer",
        "default_schema": "dbo",
        "schema_size": "small",
        "profile": "balanced",
    },
    "azure_master": {
        "connection_label": "sqlninja / master",
        "database": "master",
        "engine": "Azure SQL Database",
        "default_schema": "dbo",
        "schema_size": "small",
        "profile": "balanced",
    },
    "msdb_onprem": {
        "connection_label": "4fc2b4c8d44d / msdb",
        "database": "msdb",
        "engine": "SQL Server Enterprise/Developer",
        "default_schema": "dbo",
        "schema_size": "small",
        "profile": "balanced",
    },
    "adventure_azure": {
        "connection_label": "sqlninja / AdventureLT",
        "database": "AdventureLT",
        "engine": "Azure SQL Database",
        "default_schema": "dbo",
        "schema_size": "small",
        "profile": "balanced",
    },
    "fabric_warehouse": {
        "connection_label": "fabricninja / ContosoDW",
        "database": "ContosoDW",
        "engine": "Microsoft Fabric Data Warehouse",
        "default_schema": "dbo",
        "schema_size": "small",
        "profile": "balanced",
    },
}

# Persona facet for slicing results. Explicit ex["scenario"] wins; otherwise
# the first tag family that matches decides. Order matters: devops/admin tags
# are more specific than the developer default.
SCENARIO_TAG_MAP = [
    ("devops", {"agent-job", "backup", "restore", "jobs", "schedules", "devops",
                "deployment", "etl"}),
    ("admin", {"dmv", "locks", "waits", "size", "master_files", "plan-cache",
               "query-stats", "performance", "connections", "sessions",
               "fragmentation", "logins", "resource", "query-store",
               "queryinsights", "blocking", "cpu", "memory", "indexes",
               "missing-index", "unused-index"}),
    ("metadata", {"catalog", "information_schema", "metadata-discovery",
                  "sys.tables", "data-types", "base-table", "columns"}),
    ("analytics", {"window", "aggregate", "analytics", "rollup", "star-schema",
                   "yoy", "running-total", "top-n-per-group"}),
]


def infer_scenario(ex: dict) -> str:
    if "scenario" in ex:
        return ex["scenario"]
    tags = set((ex.get("eval") or {}).get("tags") or [])
    for name, tagset in SCENARIO_TAG_MAP:
        if tags & tagset:
            return name
    return "developer"


def load_text(name: str) -> str:
    return (SCHEMAS / name).read_text()


def with_inferred(schema_text: str, inferred: bool) -> str:
    if inferred:
        return schema_text.replace(
            "inferred system query: no", "inferred system query: yes"
        )
    return schema_text.replace(
        "inferred system query: yes", "inferred system query: no"
    )


def with_affinity(rules: str, inferred: bool) -> str:
    flag = "true" if inferred else "false"
    return re.sub(
        r"inferredSystemQuery=(true|false)",
        f"inferredSystemQuery={flag}",
        rules,
        count=1,
    )


def data_message(
    category: str,
    recent: str,
    statement: str,
    doc_suffix: str,
    line_prefix: str,
    line_suffix: str,
    schema: str,
) -> str:
    mode = MODE_INTENT if category == "intent" else MODE_CONT
    return (
        f"<mode>{mode}</mode>\n"
        f"\n"
        f"<recent_document_prefix>\n{recent}\n</recent_document_prefix>\n"
        f"\n"
        f"<current_statement_prefix>\n{statement}\n</current_statement_prefix>\n"
        f"\n"
        f"<document_suffix>\n{doc_suffix}\n</document_suffix>\n"
        f"\n"
        f"<current_line_prefix>\n{line_prefix}\n</current_line_prefix>\n"
        f"\n"
        f"<current_line_suffix>\n{line_suffix}\n</current_line_suffix>\n"
        f"\n"
        f"<schema_context>\n{schema.rstrip()}\n</schema_context>"
    )


def build_record(ex: dict, schemas: dict, intent_rules: str, cont_rules: str) -> dict:
    category = ex["completion_category"]
    inferred = bool(ex.get("inferred_system_query", False))
    schema_id = ex["schema_id"]
    schema_text = with_inferred(schemas[schema_id], inferred)
    rules = with_affinity(
        intent_rules if category == "intent" else cont_rules, inferred
    )

    comment = ex.get("user_comment")
    if category == "intent" and comment is None:
        comment = ex.get("line_prefix") or ex.get("statement")

    statement = ex.get("statement")
    line_prefix = ex.get("line_prefix")
    if statement is None:
        statement = comment or ""
    if line_prefix is None:
        line_prefix = comment or ""

    recent = ex.get("recent", "")
    doc_suffix = ex.get("document_suffix", "")
    line_suffix = ex.get("line_suffix", "")

    data = data_message(
        category,
        recent,
        statement,
        doc_suffix,
        line_prefix,
        line_suffix,
        schema_text,
    )

    messages = [
        {"role": "user", "content": rules},
        {"role": "user", "content": data},
    ]
    messages_chat = [
        {"role": "system", "content": rules},
        {"role": "user", "content": data},
    ]

    env = dict(SCHEMA_META[schema_id])
    env.update(
        {
            "language_id": "sql",
            "schema_id": schema_id,
        }
    )

    source = {
        "kind": ex["kind"],
        "trace_file": ex.get("trace_file"),
        "event_id": ex.get("event_id"),
        "exported_at": ex.get("exported_at"),
        "origin_model_id": ex.get("origin_model_id"),
        "origin_model_family": ex.get("origin_model_family"),
        "origin_model_vendor": ex.get("origin_model_vendor"),
        "origin_result": ex.get("origin_result"),
        "origin_latency_ms": ex.get("origin_latency_ms"),
        "origin_input_tokens": ex.get("origin_input_tokens"),
        "origin_output_tokens": ex.get("origin_output_tokens"),
        "origin_completion": ex.get("origin_completion"),
        "gold_author": ex.get("gold_author", "grok-4.6"),
        "gold_model": ex.get("gold_model", "grok-4.6"),
        "gold_role": ex.get("gold_role", "synthetic" if ex["kind"] == "synthetic" else "cleaned_trace"),
        "edit_notes": ex.get("edit_notes", ""),
    }

    ev = ex.get("eval", {})
    eval_block = {
        "difficulty": ev.get("difficulty", "medium"),
        "scenario": infer_scenario(ex),
        "tags": ev.get("tags", []),
        "expect_empty": bool(ev.get("expect_empty", ex.get("gold", "") == "")),
        "forbid_markdown": True,
        "raw_sql_only": True,
        "must_contain": ev.get("must_contain", []),
        "must_contain_any": ev.get("must_contain_any", []),
        "must_not_contain": ev.get(
            "must_not_contain",
            ["```", "<mode>", "empty string"],
        ),
        "required_objects": ev.get("required_objects", []),
        "scoring": ev.get(
            "scoring",
            [
                "normalized_exact",
                "constraint",
                "object_grounding",
                "empty_policy",
                "no_markdown",
            ],
        ),
        "rubric": ev["rubric"],
    }

    return {
        "id": ex["id"],
        "dataset": DATASET,
        "dataset_version": DATASET_VERSION,
        "task": "mssql_copilot_completion",
        "split": "eval",
        "completion_category": category,
        "source": source,
        "environment": env,
        "prompt": {
            "intent_mode": category == "intent",
            "inferred_system_query": inferred,
            "user_comment": comment,
            "recent_document_prefix": recent,
            "current_statement_prefix": statement,
            "document_suffix": doc_suffix,
            "current_line_prefix": line_prefix,
            "current_line_suffix": line_suffix,
        },
        "messages": messages,
        "messages_chat": messages_chat,
        "gold_completion": ex.get("gold", ""),
        "eval": eval_block,
    }


def load_example_modules():
    sys.path.insert(0, str(ROOT))
    from examples_trace import TRACE  # type: ignore
    from examples_synthetic import SYN  # type: ignore
    from examples_synthetic_v2 import SYN2  # type: ignore

    return list(TRACE) + list(SYN) + list(SYN2)


def main() -> int:
    schemas = {k: load_text(v) for k, v in SCHEMA_FILES.items()}
    intent_rules = load_text("intent_rules.txt")
    cont_rules = load_text("continuation_rules.txt")
    examples = load_example_modules()

    ids = [e["id"] for e in examples]
    if len(ids) != len(set(ids)):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise SystemExit(f"duplicate ids: {dup}")

    records = [
        build_record(ex, schemas, intent_rules, cont_rules) for ex in examples
    ]

    with OUT.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    by_kind = {}
    by_cat = {}
    by_empty = {True: 0, False: 0}
    by_diff = {}
    by_schema = {}
    by_scenario = {}
    for rec in records:
        by_kind[rec["source"]["kind"]] = by_kind.get(rec["source"]["kind"], 0) + 1
        by_cat[rec["completion_category"]] = by_cat.get(rec["completion_category"], 0) + 1
        by_empty[rec["eval"]["expect_empty"]] = by_empty.get(rec["eval"]["expect_empty"], 0) + 1
        by_diff[rec["eval"]["difficulty"]] = by_diff.get(rec["eval"]["difficulty"], 0) + 1
        by_schema[rec["environment"]["schema_id"]] = (
            by_schema.get(rec["environment"]["schema_id"], 0) + 1
        )
        by_scenario[rec["eval"]["scenario"]] = (
            by_scenario.get(rec["eval"]["scenario"], 0) + 1
        )

    print(f"wrote {len(records)} records to {OUT}")
    print("kind", by_kind)
    print("category", by_cat)
    print("expect_empty", by_empty)
    print("difficulty", by_diff)
    print("schema", by_schema)
    print("scenario", by_scenario)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
