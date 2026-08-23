"""Cleaned, de-duplicated examples taken from the MSSQL Copilot traces."""

TRACE = []


def add(**kwargs):
    kwargs.setdefault("kind", "trace")
    kwargs.setdefault("gold_model", "grok-4.6")
    kwargs.setdefault("gold_author", "grok-4.6")
    TRACE.append(kwargs)


# ---------------------------------------------------------------------------
# Intent-mode traces. Duplicates across later exports of the same session
# were dropped. Gold SQL is lightly edited: fences stripped, units fixed,
# invented objects removed, empty-string policy enforced.
# ---------------------------------------------------------------------------

add(
    id="trace-intent-001",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- what tables are in the database",
    trace_file="mssql-copilot-trace-2026-04-26T07-45-14-526Z.json",
    event_id="E-2",
    exported_at="2026-04-26T07:45:14.526Z",
    origin_model_id="claude-sonnet-4-6",
    origin_model_family="claude-sonnet",
    origin_model_vendor="anthropic-api",
    origin_result="accepted",
    origin_latency_ms=1770,
    origin_input_tokens=3179,
    origin_output_tokens=42,
    origin_completion="SELECT TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE\nFROM INFORMATION_SCHEMA.TABLES\nORDER BY TABLE_SCHEMA, TABLE_NAME;",
    gold_role="cleaned_trace",
    edit_notes="Kept the accepted listing; reformatted to the canonical multiline layout.",
    gold="""SELECT
    TABLE_SCHEMA,
    TABLE_NAME,
    TABLE_TYPE
FROM INFORMATION_SCHEMA.TABLES
ORDER BY
    TABLE_SCHEMA,
    TABLE_NAME;""",
    eval={
        "difficulty": "easy",
        "tags": ["catalog", "information_schema", "metadata-discovery"],
        "must_contain": ["INFORMATION_SCHEMA.TABLES", "TABLE_NAME"],
        "must_contain_any": ["TABLE_SCHEMA", "TABLE_CATALOG"],
        "required_objects": ["INFORMATION_SCHEMA.TABLES"],
        "rubric": "Simplest catalog listing of tables/views. Prefer INFORMATION_SCHEMA.TABLES. Do not invent FitnessApp object names as literals unless selecting from a real catalog view.",
    },
)

add(
    id="trace-intent-002",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- what meals are there",
    recent="-- what is the size of all the databases in this server\nSELECT\n    d.name AS database_name,\n    SUM(mf.size) * 8 / 1024 AS size_mb,\n    SUM(mf.size) * 8 / 1024.0 / 1024 AS size_gb\nFROM sys.databases d\nJOIN sys.master_files mf ON d.database_id = mf.database_id\nGROUP BY d.name\nORDER BY size_mb DESC;\n\n-- what are tables in fitnessapp dbs?\nSELECT TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE\nFROM FitnessApp_Test.INFORMATION_SCHEMA.TABLES\nORDER BY TABLE_SCHEMA, TABLE_NAME;",
    trace_file="mssql-copilot-trace-2026-04-26T07-53-42-587Z.json",
    event_id="E-1",
    exported_at="2026-04-26T07:53:42.587Z",
    origin_model_id="claude-sonnet-4-6",
    origin_model_family="claude-sonnet",
    origin_model_vendor="anthropic-api",
    origin_result="accepted",
    origin_latency_ms=2443,
    origin_input_tokens=3353,
    origin_output_tokens=71,
    origin_completion="SELECT\n    meal_id,\n    meal_name,\n    category,\n    calories,\n    protein_g,\n    carbs_g,\n    fats_g,\n    dietary_preference\nFROM dbo.Meals\nORDER BY category, meal_name;",
    gold_role="cleaned_trace",
    edit_notes="Accepted gold was already schema-grounded; canonical formatting only.",
    gold="""SELECT
    meal_id,
    meal_name,
    category,
    calories,
    protein_g,
    carbs_g,
    fats_g,
    dietary_preference
FROM dbo.Meals
ORDER BY
    category,
    meal_name;""",
    eval={
        "difficulty": "easy",
        "tags": ["user-schema", "select", "fitnessapp"],
        "must_contain": ["dbo.Meals"],
        "must_not_contain": ["```", "dbo.Products", "password_hash"],
        "required_objects": ["dbo.Meals"],
        "rubric": "SELECT the listed dbo.Meals columns. Do not invent meal columns. Prefer schema-qualified dbo.Meals.",
    },
)

add(
    id="trace-intent-003",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- what is the size of the database",
    recent="-- what is the size of all the databases in this server\nSELECT\n    d.name AS database_name,\n    SUM(mf.size) * 8 / 1024 AS size_mb,\n    SUM(mf.size) * 8 / 1024.0 / 1024 AS size_gb\nFROM sys.databases d\nJOIN sys.master_files mf ON d.database_id = mf.database_id\nGROUP BY d.name\nORDER BY size_mb DESC;",
    trace_file="mssql-copilot-trace-2026-04-26T07-53-42-587Z.json",
    event_id="E-2",
    exported_at="2026-04-26T07:53:42.587Z",
    origin_model_id="claude-sonnet-4-6",
    origin_model_family="claude-sonnet",
    origin_model_vendor="anthropic-api",
    origin_result="accepted",
    origin_latency_ms=3352,
    origin_input_tokens=3428,
    origin_output_tokens=102,
    origin_completion="SELECT\n    SUM(mf.size) * 8 / 1024 AS size_mb,\n    SUM(mf.size) * 8 / 1024.0 / 1024 AS size_gb\nFROM sys.databases d\nJOIN sys.master_files mf ON d.database_id = mf.database_id\nWHERE d.name = DB_NAME();",
    gold_role="cleaned_trace",
    edit_notes="Kept master_files size query (object is listed on this on-prem catalog). Scoped to DB_NAME().",
    gold="""SELECT
    SUM(mf.size) * 8 / 1024 AS size_mb,
    SUM(mf.size) * 8 / 1024.0 / 1024 AS size_gb
FROM sys.databases AS d
INNER JOIN sys.master_files AS mf
    ON mf.database_id = d.database_id
WHERE d.name = DB_NAME();""",
    eval={
        "difficulty": "medium",
        "tags": ["system", "size", "master_files"],
        "must_contain": ["sys.master_files", "DB_NAME()"],
        "required_objects": ["sys.master_files", "sys.databases"],
        "rubric": "Current database size via sys.master_files (listed on this engine). Filter to DB_NAME(). pages*8/1024 is the conventional MB conversion.",
    },
)

add(
    id="trace-intent-004",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- which plan caches aren't used in the past 3 weeks",
    trace_file="mssql-copilot-trace-2026-04-26T07-53-42-587Z.json",
    event_id="E-4",
    exported_at="2026-04-26T07:53:42.587Z",
    origin_model_id="claude-sonnet-4-6",
    origin_model_family="claude-sonnet",
    origin_model_vendor="anthropic-api",
    origin_result="accepted",
    origin_latency_ms=3110,
    origin_input_tokens=3545,
    origin_output_tokens=143,
    origin_completion="SELECT\n    qs.plan_handle,\n    qs.creation_time,\n    qs.last_execution_time,\n    qs.execution_count,\n    SUBSTRING(st.text, 1, 200) AS query_text\nFROM sys.dm_exec_query_stats qs\nCROSS APPLY sys.dm_exec_sql_text(qs.sql_handle) st\nWHERE qs.last_execution_time < DATEADD(WEEK, -3, GETDATE())\nORDER BY qs.last_execution_time ASC;",
    gold_role="cleaned_trace",
    edit_notes="Canonical aliases and CROSS APPLY. Query Store is not in the catalog, so query_stats last_execution_time is the grounded approximation.",
    gold="""SELECT
    qs.plan_handle,
    qs.creation_time,
    qs.last_execution_time,
    qs.execution_count,
    SUBSTRING(st.text, 1, 200) AS query_text
FROM sys.dm_exec_query_stats AS qs
CROSS APPLY sys.dm_exec_sql_text(qs.sql_handle) AS st
WHERE qs.last_execution_time < DATEADD(WEEK, -3, GETDATE())
ORDER BY
    qs.last_execution_time;""",
    eval={
        "difficulty": "medium",
        "tags": ["dmv", "query-stats", "plan-cache"],
        "must_contain": ["sys.dm_exec_query_stats", "DATEADD", "last_execution_time"],
        "required_objects": ["sys.dm_exec_query_stats", "sys.dm_exec_sql_text"],
        "must_not_contain": ["sys.query_store_runtime_stats", "```"],
        "rubric": "Plans not executed in 3 weeks using dm_exec_query_stats. Do not use Query Store objects (not listed).",
    },
)

add(
    id="trace-intent-005",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- how to write wait for 5m?",
    trace_file="mssql-copilot-trace-2026-04-26T07-53-42-587Z.json",
    event_id="E-5",
    exported_at="2026-04-26T07:53:42.587Z",
    origin_model_id="claude-sonnet-4-6",
    origin_model_family="claude-sonnet",
    origin_model_vendor="anthropic-api",
    origin_result="accepted",
    origin_latency_ms=1611,
    origin_input_tokens=3686,
    origin_output_tokens=21,
    origin_completion="WAITFOR DELAY '00:05:00';",
    gold_role="cleaned_trace",
    edit_notes="Accepted snippet was already the canonical WAITFOR DELAY.",
    gold="WAITFOR DELAY '00:05:00';",
    eval={
        "difficulty": "easy",
        "tags": ["tsql-idiom", "waitfor"],
        "must_contain": ["WAITFOR DELAY", "00:05:00"],
        "must_not_contain": ["WAITFOR TIME"],
        "rubric": "WAITFOR DELAY for five minutes as hh:mm:ss. Not WAITFOR TIME.",
    },
)

add(
    id="trace-intent-006",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment="-- which cached plans were used the least recently",
    trace_file="mssql-copilot-trace-2026-04-26T08-10-07-172Z.json",
    event_id="E-3",
    exported_at="2026-04-26T08:10:07.172Z",
    origin_model_id="claude-sonnet-4-6",
    origin_model_family="claude-sonnet",
    origin_model_vendor="anthropic-api",
    origin_result="accepted",
    origin_latency_ms=2827,
    origin_input_tokens=3183,
    origin_output_tokens=134,
    origin_completion="SELECT TOP 25\n    qs.last_execution_time,\n    qs.execution_count,\n    qs.total_worker_time,\n    qs.total_elapsed_time,\n    qs.total_logical_reads,\n    st.text AS sql_text,\n    qs.plan_handle\nFROM sys.dm_exec_query_stats AS qs\nCROSS APPLY sys.dm_exec_sql_text(qs.sql_handle) AS st\nORDER BY qs.last_execution_time ASC;",
    gold_role="cleaned_trace",
    edit_notes="Kept TOP/oldest last_execution_time ordering; canonical layout.",
    gold="""SELECT TOP (25)
    qs.last_execution_time,
    qs.execution_count,
    qs.total_worker_time,
    qs.total_elapsed_time,
    qs.total_logical_reads,
    st.text AS sql_text,
    qs.plan_handle
FROM sys.dm_exec_query_stats AS qs
CROSS APPLY sys.dm_exec_sql_text(qs.sql_handle) AS st
ORDER BY
    qs.last_execution_time;""",
    eval={
        "difficulty": "easy",
        "tags": ["dmv", "query-stats"],
        "must_contain": ["sys.dm_exec_query_stats", "last_execution_time"],
        "required_objects": ["sys.dm_exec_query_stats", "sys.dm_exec_sql_text"],
        "rubric": "Least-recently-used cached plans: ORDER BY last_execution_time ascending with query text via dm_exec_sql_text.",
    },
)

add(
    id="trace-intent-007",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- what queries are running slower than usual?",
    trace_file="mssql-copilot-trace-2026-04-26T08-28-33-899Z.json",
    event_id="E-2",
    exported_at="2026-04-26T08:28:33.899Z",
    origin_model_id="gpt-5.5",
    origin_model_family="gpt-5.5",
    origin_model_vendor="openai-api",
    origin_result="accepted",
    origin_latency_ms=9318,
    origin_input_tokens=3306,
    origin_output_tokens=217,
    origin_completion="WITH query_history AS (\n    SELECT\n        sql_handle,\n        plan_handle,\n        SUM(execution_count) AS execution_count,\n        SUM(total_elapsed_time) AS total_elapsed_time\n    FROM sys.dm_exec_query_stats\n    GROUP BY\n        sql_handle,\n        plan_handle\n)\nSELECT\n    r.session_id,\n    r.status,\n    r.command,\n    r.wait_type,\n    r.wait_time,\n    r.cpu_time,\n    r.total_elapsed_time AS current_elapsed_time_ms,\n    qh.total_elapsed_time / NULLIF(qh.execution_count, 0) / 1000 AS average_elapsed_time_ms,\n    qh.execution_count,\n    st.text\nFROM sys.dm_exec_requests AS r\nJOIN query_history AS qh\n    ON qh.sql_handle = r.sql_handle\n    AND qh.plan_handle = r.plan_handle\nCROSS APPLY sys.dm_exec_sql_text(r.sql_handle) AS st\nWHERE r.total_elapsed_time > qh.total_elapsed_time / NULLIF(qh.execution_count, 0) / 1000\nORDER BY\n    r.total_elapsed_time DESC;",
    gold_role="replaced_trace_gold",
    edit_notes="Replaced origin gold: requests.total_elapsed_time is milliseconds and query_stats.total_elapsed_time is microseconds. Convert stats to ms before comparing. Exclude the current session.",
    gold="""SELECT
    r.session_id,
    r.status,
    r.command,
    r.wait_type,
    r.wait_time,
    r.cpu_time,
    r.total_elapsed_time AS current_elapsed_ms,
    qs.total_elapsed_time / NULLIF(qs.execution_count, 0) / 1000.0 AS avg_elapsed_ms,
    qs.execution_count,
    st.text AS query_text
FROM sys.dm_exec_requests AS r
INNER JOIN sys.dm_exec_query_stats AS qs
    ON qs.sql_handle = r.sql_handle
    AND qs.plan_handle = r.plan_handle
CROSS APPLY sys.dm_exec_sql_text(r.sql_handle) AS st
WHERE r.session_id <> @@SPID
  AND r.total_elapsed_time > qs.total_elapsed_time / NULLIF(qs.execution_count, 0) / 1000.0
ORDER BY
    r.total_elapsed_time DESC;""",
    eval={
        "difficulty": "hard",
        "tags": ["dmv", "performance", "unit-conversion", "requests"],
        "must_contain": ["sys.dm_exec_requests", "sys.dm_exec_query_stats", "@@SPID"],
        "required_objects": [
            "sys.dm_exec_requests",
            "sys.dm_exec_query_stats",
            "sys.dm_exec_sql_text",
        ],
        "rubric": "Currently running requests slower than their cached average. Must join requests to query_stats and convert microseconds to milliseconds. Do not invent Query Store objects.",
    },
)

add(
    id="trace-intent-008",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- what tables are in the database and what columns?",
    trace_file="mssql-copilot-trace-2026-04-26T08-28-33-899Z.json",
    event_id="E-5",
    exported_at="2026-04-26T08:28:33.899Z",
    origin_model_id="gpt-5.4-mini",
    origin_model_family="gpt-5.4-mini",
    origin_model_vendor="openai-api",
    origin_result="accepted",
    origin_latency_ms=940,
    origin_input_tokens=3391,
    origin_output_tokens=52,
    origin_completion="SELECT\n    TABLE_SCHEMA,\n    TABLE_NAME,\n    COLUMN_NAME,\n    ORDINAL_POSITION,\n    DATA_TYPE,\n    IS_NULLABLE\nFROM INFORMATION_SCHEMA.COLUMNS\nORDER BY\n    TABLE_SCHEMA,\n    TABLE_NAME,\n    ORDINAL_POSITION;",
    gold_role="cleaned_trace",
    edit_notes="Preferred the fuller accepted variant (includes DATA_TYPE) over the earlier column-name-only success.",
    gold="""SELECT
    TABLE_SCHEMA,
    TABLE_NAME,
    COLUMN_NAME,
    ORDINAL_POSITION,
    DATA_TYPE,
    IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS
ORDER BY
    TABLE_SCHEMA,
    TABLE_NAME,
    ORDINAL_POSITION;""",
    eval={
        "difficulty": "easy",
        "tags": ["catalog", "information_schema", "columns"],
        "must_contain": ["INFORMATION_SCHEMA.COLUMNS", "COLUMN_NAME", "DATA_TYPE"],
        "required_objects": ["INFORMATION_SCHEMA.COLUMNS"],
        "rubric": "List tables and columns with types from INFORMATION_SCHEMA.COLUMNS.",
    },
)

add(
    id="trace-intent-009",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- what is the best snippet to handle resolving deadlocks and retries",
    trace_file="mssql-copilot-trace-2026-04-26T08-45-36-321Z.json",
    event_id="E-2",
    exported_at="2026-04-26T08:45:36.321Z",
    origin_model_id="claude-opus-4-6",
    origin_model_family="claude-opus",
    origin_model_vendor="anthropic-api",
    origin_result="accepted",
    origin_latency_ms=7453,
    origin_input_tokens=4000,
    origin_output_tokens=404,
    origin_completion="DECLARE @RetryCount INT = 0;\nDECLARE @MaxRetries INT = 5;\nDECLARE @Success BIT = 0;\n\nWHILE @RetryCount < @MaxRetries AND @Success = 0\nBEGIN\n    BEGIN TRY\n        BEGIN TRANSACTION;\n\n        -- === Place your transactional work here ===\n        -- Example:\n        -- UPDATE dbo.Products SET UnitPrice = UnitPrice * 1.10 WHERE ProductID = 1;\n\n        COMMIT TRANSACTION;\n        SET @Success = 1;\n    END TRY\n    BEGIN CATCH\n        IF @@TRANCOUNT > 0\n            ROLLBACK TRANSACTION;\n\n        IF ERROR_NUMBER() = 1205 -- Deadlock victim\n        BEGIN\n            SET @RetryCount = @RetryCount + 1;\n\n            IF @RetryCount < @MaxRetries\n            BEGIN\n                WAITFOR DELAY '00:00:02'; -- Wait 2 seconds before retrying\n            END\n            ELSE\n            BEGIN\n                RAISERROR('Deadlock retry limit reached after %d attempts.', 16, 1, @MaxRetries);\n            END\n        END\n        ELSE\n        BEGIN\n            -- Re-throw non-deadlock errors immediately\n            DECLARE @ErrorMessage NVARCHAR(4000) = ERROR_MESSAGE();\n            DECLARE @ErrorSeverity INT = ERROR_SEVERITY();\n            DECLARE @ErrorState INT = ERROR_STATE();\n            RAISERROR(@ErrorMessage, @ErrorSeverity, @ErrorState);\n            SET @Success = 1; -- Exit the loop for non-deadlock errors\n        END\n    END CATCH\nEND;",
    gold_role="cleaned_trace",
    edit_notes="Kept TRY/CATCH deadlock 1205 retry. Example UPDATE uses dbo.Products which is listed (names-only) so the commented example is legal. Did not THROW (THROW is fine but origin used RAISERROR).",
    gold="""DECLARE @RetryCount INT = 0;
DECLARE @MaxRetries INT = 5;
DECLARE @Success BIT = 0;

WHILE @RetryCount < @MaxRetries AND @Success = 0
BEGIN
    BEGIN TRY
        BEGIN TRANSACTION;

        -- Place transactional work here.

        COMMIT TRANSACTION;
        SET @Success = 1;
    END TRY
    BEGIN CATCH
        IF @@TRANCOUNT > 0
            ROLLBACK TRANSACTION;

        IF ERROR_NUMBER() = 1205
        BEGIN
            SET @RetryCount = @RetryCount + 1;

            IF @RetryCount < @MaxRetries
                WAITFOR DELAY '00:00:02';
            ELSE
                RAISERROR('Deadlock retry limit reached after %d attempts.', 16, 1, @MaxRetries);
        END
        ELSE
        BEGIN
            DECLARE @ErrorMessage NVARCHAR(4000) = ERROR_MESSAGE();
            DECLARE @ErrorSeverity INT = ERROR_SEVERITY();
            DECLARE @ErrorState INT = ERROR_STATE();
            RAISERROR(@ErrorMessage, @ErrorSeverity, @ErrorState);
            SET @Success = 1;
        END
    END CATCH
END;""",
    eval={
        "difficulty": "hard",
        "tags": ["tsql-idiom", "deadlock", "retry", "try-catch"],
        "must_contain": ["BEGIN TRY", "1205", "ROLLBACK", "WAITFOR DELAY"],
        "must_not_contain": ["```"],
        "rubric": "Retry loop catching deadlock error 1205, rolling back, delaying, and rethrowing other errors. Raw T-SQL, not markdown.",
    },
)

add(
    id="trace-intent-010",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- which cached plans are running faster in the past month than two months ago, only look at whole months, so ignore april",
    trace_file="mssql-copilot-trace-2026-04-26T08-48-48-432Z.json",
    event_id="E-5",
    exported_at="2026-04-26T08:48:48.432Z",
    origin_model_id="claude-opus-4-6",
    origin_model_family="claude-opus",
    origin_model_vendor="anthropic-api",
    origin_result="accepted",
    origin_latency_ms=None,
    origin_completion="WITH months AS (...truncated in logs...)",
    gold_role="replaced_trace_gold",
    edit_notes="Origin query was truncated mid-SELECT and used last_execution_time as if query_stats were monthly snapshots. Gold uses complete months (previous month vs two months ago), ignores the current April partial month, and still only listed DMVs.",
    gold="""WITH bounds AS (
    SELECT
        DATEADD(MONTH, DATEDIFF(MONTH, 0, GETDATE()) - 1, 0) AS last_month_start,
        DATEADD(MONTH, DATEDIFF(MONTH, 0, GETDATE()), 0) AS this_month_start,
        DATEADD(MONTH, DATEDIFF(MONTH, 0, GETDATE()) - 2, 0) AS two_months_ago_start
),
last_month AS (
    SELECT
        qs.sql_handle,
        qs.plan_handle,
        qs.execution_count,
        qs.total_elapsed_time / NULLIF(qs.execution_count, 0) AS avg_elapsed_us
    FROM sys.dm_exec_query_stats AS qs
    CROSS JOIN bounds AS b
    WHERE qs.last_execution_time >= b.last_month_start
      AND qs.last_execution_time < b.this_month_start
),
two_months_ago AS (
    SELECT
        qs.sql_handle,
        qs.plan_handle,
        qs.execution_count,
        qs.total_elapsed_time / NULLIF(qs.execution_count, 0) AS avg_elapsed_us
    FROM sys.dm_exec_query_stats AS qs
    CROSS JOIN bounds AS b
    WHERE qs.last_execution_time >= b.two_months_ago_start
      AND qs.last_execution_time < b.last_month_start
)
SELECT
    lm.sql_handle,
    lm.plan_handle,
    SUBSTRING(st.text, 1, 200) AS query_text,
    lm.execution_count AS last_month_exec_count,
    lm.avg_elapsed_us AS last_month_avg_elapsed_us,
    tma.execution_count AS two_months_ago_exec_count,
    tma.avg_elapsed_us AS two_months_ago_avg_elapsed_us,
    tma.avg_elapsed_us - lm.avg_elapsed_us AS improvement_us
FROM last_month AS lm
INNER JOIN two_months_ago AS tma
    ON tma.sql_handle = lm.sql_handle
    AND tma.plan_handle = lm.plan_handle
CROSS APPLY sys.dm_exec_sql_text(lm.sql_handle) AS st
WHERE lm.avg_elapsed_us < tma.avg_elapsed_us
ORDER BY
    improvement_us DESC;""",
    eval={
        "difficulty": "hard",
        "tags": ["dmv", "query-stats", "date-windows", "cte"],
        "must_contain": ["sys.dm_exec_query_stats", "DATEADD", "MONTH"],
        "must_not_contain": ["sys.query_store_runtime_stats", "```"],
        "required_objects": ["sys.dm_exec_query_stats", "sys.dm_exec_sql_text"],
        "rubric": "Compare cached plans across two complete months, skipping the current incomplete April. Query Store is not listed so query_stats last_execution_time windows are the grounded approach. Must be a complete statement, not a truncated SELECT list.",
    },
)

add(
    id="trace-intent-011",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- which tables are in the database",
    trace_file="mssql-copilot-trace-2026-04-26T08-50-29-883Z.json",
    event_id="E-7",
    exported_at="2026-04-26T08:50:29.883Z",
    origin_model_id="grok-4.20",
    origin_model_family="grok-4.20",
    origin_model_vendor="xai-api",
    origin_result="success",
    origin_latency_ms=None,
    origin_completion="SELECT\n    TABLE_SCHEMA,\n    TABLE_NAME\nFROM INFORMATION_SCHEMA.TABLES\nWHERE TABLE_TYPE = 'BASE TABLE'\nORDER BY TABLE_SCHEMA, TABLE_NAME;",
    gold_role="cleaned_trace",
    edit_notes="Kept BASE TABLE filter from grok-4.20; canonical ORDER BY layout.",
    gold="""SELECT
    TABLE_SCHEMA,
    TABLE_NAME
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_TYPE = 'BASE TABLE'
ORDER BY
    TABLE_SCHEMA,
    TABLE_NAME;""",
    eval={
        "difficulty": "easy",
        "tags": ["catalog", "information_schema", "base-table"],
        "must_contain": ["INFORMATION_SCHEMA.TABLES", "BASE TABLE"],
        "required_objects": ["INFORMATION_SCHEMA.TABLES"],
        "rubric": "List base tables only from INFORMATION_SCHEMA.TABLES.",
    },
)

add(
    id="trace-intent-012",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- what connections are active?",
    trace_file="mssql-copilot-trace-2026-04-26T08-50-29-883Z.json",
    event_id="E-9",
    exported_at="2026-04-26T08:50:29.883Z",
    origin_model_id="grok-4.20",
    origin_model_family="grok-4.20",
    origin_model_vendor="xai-api",
    origin_result="accepted",
    origin_completion="SELECT *\nFROM sys.dm_exec_connections;",
    gold_role="replaced_trace_gold",
    edit_notes="Origin SELECT * is legal but weak. Gold projects the listed connection columns and joins sessions on session_id for login/host context.",
    gold="""SELECT
    c.session_id,
    c.connect_time,
    c.net_transport,
    c.protocol_type,
    c.client_net_address,
    c.local_net_address,
    c.local_tcp_port,
    s.login_name,
    s.host_name,
    s.program_name,
    s.status
FROM sys.dm_exec_connections AS c
LEFT JOIN sys.dm_exec_sessions AS s
    ON s.session_id = c.session_id
ORDER BY
    c.connect_time DESC;""",
    eval={
        "difficulty": "easy",
        "tags": ["dmv", "connections"],
        "must_contain": ["sys.dm_exec_connections"],
        "required_objects": ["sys.dm_exec_connections"],
        "rubric": "Active connections from sys.dm_exec_connections. Joining sessions is good. Do not invent columns that are not listed.",
    },
)

add(
    id="trace-intent-013",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- what are the top 10 queries in the qds data?",
    trace_file="mssql-copilot-trace-2026-04-26T08-52-00-253Z.json",
    event_id="E-15",
    exported_at="2026-04-26T08:52:00.253Z",
    origin_model_id="gpt-5.5",
    origin_model_family="gpt-5.5",
    origin_model_vendor="openai-api",
    origin_result="emptyFromModel",
    origin_completion="",
    gold_role="cleaned_trace",
    edit_notes="Query Store objects are not in the schema context. The first gpt-5.5 call correctly returned empty; a later retry with a question mark hallucinated dm_exec_query_stats as QDS. Gold is the empty string.",
    gold="",
    eval={
        "difficulty": "medium",
        "tags": ["empty-policy", "query-store", "trap"],
        "expect_empty": True,
        "must_contain": [],
        "must_not_contain": [
            "```",
            "sys.dm_exec_query_stats",
            "sys.query_store_runtime_stats",
            "SELECT",
        ],
        "rubric": "QDS/Query Store catalogs are not listed. Return exactly an empty string. Substituting dm_exec_query_stats is a fail.",
    },
)

add(
    id="trace-intent-014",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    user_comment="-- what columns and all the data types in this database?",
    trace_file="mssql-copilot-trace-2026-04-26T08-52-44-210Z.json",
    event_id="E-25",
    exported_at="2026-04-26T08:52:44.210Z",
    origin_model_id="gpt-5.5",
    origin_model_family="gpt-5.5",
    origin_model_vendor="openai-api",
    origin_result="accepted",
    origin_completion="SELECT\n    TABLE_SCHEMA,\n    TABLE_NAME,\n    COLUMN_NAME,\n    DATA_TYPE,\n    IS_NULLABLE\nFROM INFORMATION_SCHEMA.COLUMNS\nORDER BY\n    TABLE_SCHEMA,\n    TABLE_NAME,\n    ORDINAL_POSITION;",
    gold_role="cleaned_trace",
    edit_notes="Accepted catalog query; kept as-is besides layout.",
    gold="""SELECT
    TABLE_SCHEMA,
    TABLE_NAME,
    COLUMN_NAME,
    DATA_TYPE,
    IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS
ORDER BY
    TABLE_SCHEMA,
    TABLE_NAME,
    ORDINAL_POSITION;""",
    eval={
        "difficulty": "easy",
        "tags": ["catalog", "data-types"],
        "must_contain": ["INFORMATION_SCHEMA.COLUMNS", "DATA_TYPE"],
        "required_objects": ["INFORMATION_SCHEMA.COLUMNS"],
        "rubric": "All columns and data types from INFORMATION_SCHEMA.COLUMNS.",
    },
)

add(
    id="trace-intent-015",
    completion_category="intent",
    schema_id="master_onprem",
    inferred_system_query=False,
    user_comment="-- which requests had the top 5 fastest queries recently per request?",
    trace_file="mssql-copilot-trace-2026-04-26T09-03-04-074Z.json",
    event_id="E-8",
    exported_at="2026-04-26T09:03:04.074Z",
    origin_model_id="claude-haiku-4-5-20251001",
    origin_model_family="claude-haiku",
    origin_model_vendor="anthropic-api",
    origin_result="accepted",
    origin_completion="```sql\nSELECT\n  TOP 5\n  deqs.sql_handle,\n  dest.text AS query_text,\n  deqs.last_execution_time,\n  deqs.execution_count,\n  deqs.total_elapsed_time / deqs.execution_count AS avg_elapsed_time_ms,\n  deqs.total_worker_time / deqs.execution_count AS avg_worker_time_ms\nFROM\n  sys.dm_exec_query_stats AS deqs\n  CROSS APPLY sys.dm_exec_sql_text(deqs.sql_handle) AS dest\nWHERE\n  deqs.last_execution_time >= DATEADD(DAY, -7, GETUTCDATE())\nORDER BY\n  deqs.total_elapsed_time / deqs.execution_count ASC;\n```",
    gold_role="replaced_trace_gold",
    edit_notes="Stripped markdown fences (rules forbid them). Convert microseconds to ms. Alias elapsed correctly. Prefer GETDATE() or GETUTCDATE() consistently with DATEADD.",
    gold="""SELECT TOP (5)
    qs.sql_handle,
    st.text AS query_text,
    qs.last_execution_time,
    qs.execution_count,
    qs.total_elapsed_time / NULLIF(qs.execution_count, 0) / 1000.0 AS avg_elapsed_ms,
    qs.total_worker_time / NULLIF(qs.execution_count, 0) / 1000.0 AS avg_worker_ms
FROM sys.dm_exec_query_stats AS qs
CROSS APPLY sys.dm_exec_sql_text(qs.sql_handle) AS st
WHERE qs.last_execution_time >= DATEADD(DAY, -7, GETDATE())
ORDER BY
    avg_elapsed_ms;""",
    eval={
        "difficulty": "medium",
        "tags": ["dmv", "query-stats", "no-markdown", "top-n"],
        "must_contain": ["sys.dm_exec_query_stats", "TOP"],
        "must_not_contain": ["```"],
        "required_objects": ["sys.dm_exec_query_stats", "sys.dm_exec_sql_text"],
        "rubric": "Top 5 fastest recent cached queries by average elapsed. Raw SQL only — markdown fences fail. Divide by execution_count; converting us to ms is preferred.",
    },
)

add(
    id="trace-intent-016",
    completion_category="intent",
    schema_id="master_onprem",
    inferred_system_query=False,
    user_comment="-- which tables have the most rows and use the most space",
    statement="  -- which tables have the most rows and use the most space",
    line_prefix="  -- which tables have the most rows and use the most space",
    trace_file="mssql-copilot-trace-2026-04-26T09-06-28-780Z.json",
    event_id="E-48",
    exported_at="2026-04-26T09:06:28.780Z",
    origin_model_id="claude-haiku-4-5-20251001",
    origin_model_family="claude-haiku",
    origin_model_vendor="anthropic-api",
    origin_result="accepted",
    origin_completion="SELECT\n  s.name AS schema_name,\n  t.name AS table_name,\n  p.rows,\n  (au.total_pages * 8) / 1024.0 AS space_used_mb\nFROM\n  sys.tables AS t\n  INNER JOIN sys.schemas AS s ON t.schema_id = s.schema_id\n  INNER JOIN sys.partitions AS p ON t.object_id = p.object_id\n  INNER JOIN sys.allocation_units AS au ON p.partition_id = au.container_id\nWHERE\n  p.index_id IN (0, 1)\nGROUP BY\n  s.name,\n  t.name,\n  p.rows,\n  au.total_pages\nORDER BY\n  p.rows DESC,\n  (au.total_pages * 8) / 1024.0 DESC;",
    gold_role="replaced_trace_gold",
    edit_notes="Origin grouped by p.rows and au.total_pages, which splits a table into allocation-unit grains. Gold aggregates rowcount and pages per table.",
    gold="""SELECT
    s.name AS schema_name,
    t.name AS table_name,
    SUM(p.rows) AS row_count,
    SUM(au.total_pages) * 8 / 1024.0 AS space_used_mb
FROM sys.tables AS t
INNER JOIN sys.schemas AS s
    ON s.schema_id = t.schema_id
INNER JOIN sys.partitions AS p
    ON p.object_id = t.object_id
INNER JOIN sys.allocation_units AS au
    ON au.container_id = p.partition_id
WHERE p.index_id IN (0, 1)
GROUP BY
    s.name,
    t.name
ORDER BY
    row_count DESC,
    space_used_mb DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["catalog", "space", "sys.tables"],
        "must_contain": ["sys.tables", "sys.partitions", "sys.allocation_units"],
        "required_objects": [
            "sys.tables",
            "sys.schemas",
            "sys.partitions",
            "sys.allocation_units",
        ],
        "rubric": "Per-table row count and reserved space from listed catalog views. Aggregate; do not group by pages/rows in a way that explodes one table into many rows.",
    },
)

add(
    id="trace-intent-017",
    completion_category="intent",
    schema_id="azure_master",
    inferred_system_query=False,
    user_comment="-- what locks are held in this database?",
    trace_file="mssql-copilot-trace-2026-04-26T09-25-47-708Z.json",
    event_id="E-2",
    exported_at="2026-04-26T09:25:47.708Z",
    origin_model_id="claude-sonnet-4-6",
    origin_model_family="claude-sonnet",
    origin_model_vendor="anthropic-api",
    origin_result="accepted",
    origin_completion="SELECT\n    tl.resource_type,\n    tl.resource_database_id,\n    DB_NAME(tl.resource_database_id) AS database_name,\n    tl.resource_associated_entity_id,\n    OBJECT_NAME(tl.resource_associated_entity_id, tl.resource_database_id) AS object_name,\n    tl.request_mode,\n    tl.request_type,\n    tl.request_status,\n    tl.request_session_id,\n    s.login_name,\n    s.host_name,\n    s.program_name\nFROM sys.dm_tran_locks AS tl\nLEFT JOIN sys.dm_exec_sessions AS s\n    ON tl.request_session_id = s.session_id\nWHERE tl.resource_database_id = DB_ID()\nORDER BY tl.request_session_id, tl.resource_type;",
    gold_role="cleaned_trace",
    edit_notes="Accepted lock listing is grounded in listed DMV columns. Kept OBJECT_NAME/DB_NAME helpers.",
    gold="""SELECT
    tl.resource_type,
    tl.resource_database_id,
    DB_NAME(tl.resource_database_id) AS database_name,
    tl.resource_associated_entity_id,
    OBJECT_NAME(tl.resource_associated_entity_id, tl.resource_database_id) AS object_name,
    tl.request_mode,
    tl.request_type,
    tl.request_status,
    tl.request_session_id,
    s.login_name,
    s.host_name,
    s.program_name
FROM sys.dm_tran_locks AS tl
LEFT JOIN sys.dm_exec_sessions AS s
    ON s.session_id = tl.request_session_id
WHERE tl.resource_database_id = DB_ID()
ORDER BY
    tl.request_session_id,
    tl.resource_type;""",
    eval={
        "difficulty": "medium",
        "tags": ["dmv", "locks", "azure"],
        "must_contain": ["sys.dm_tran_locks", "DB_ID()"],
        "required_objects": ["sys.dm_tran_locks"],
        "rubric": "Locks in the current database from sys.dm_tran_locks. Filter resource_database_id = DB_ID().",
    },
)

add(
    id="trace-intent-018",
    completion_category="intent",
    schema_id="master_onprem",
    inferred_system_query=True,
    user_comment="-- what are the sales by region",
    trace_file="mssql-copilot-trace-2026-04-26T09-39-16-910Z.json",
    event_id="E-10",
    exported_at="2026-04-26T09:39:16.910Z",
    origin_model_id="claude-sonnet-4-6",
    origin_model_family="claude-sonnet",
    origin_model_vendor="anthropic-api",
    origin_result="emptyFromSanitizer",
    origin_completion="The schema has no user tables with sales or region data, so I cannot produce a meaningful query.",
    gold_role="replaced_trace_gold",
    edit_notes="Sanitizer emptied a prose refusal. Gold is the empty string — no apology, no SELECT placeholder.",
    gold="",
    eval={
        "difficulty": "medium",
        "tags": ["empty-policy", "missing-schema", "trap"],
        "expect_empty": True,
        "must_not_contain": ["SELECT", "```", "sorry", "cannot", "sales"],
        "rubric": "master has zero user tables. Sales/region cannot be answered. Return exactly an empty string, not an explanation.",
    },
)

add(
    id="trace-intent-019",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=True,
    user_comment="-- which tables are in the db",
    trace_file="mssql-copilot-trace-2026-04-26T09-39-16-910Z.json",
    event_id="E-14",
    exported_at="2026-04-26T09:39:16.910Z",
    origin_model_id="claude-sonnet-4-6",
    origin_model_family="claude-sonnet",
    origin_model_vendor="anthropic-api",
    origin_result="accepted",
    origin_completion="SELECT\n    t.name AS table_name,\n    s.name AS schema_name,\n    t.create_date,\n    t.modify_date\nFROM sys.tables t\nJOIN sys.schemas s ON t.schema_id = s.schema_id\nWHERE t.is_ms_shipped = 0\nORDER BY s.name, t.name;",
    gold_role="cleaned_trace",
    edit_notes="inferredSystemQuery=true so sys.tables is the preferred catalog source. Kept is_ms_shipped filter.",
    gold="""SELECT
    s.name AS schema_name,
    t.name AS table_name,
    t.create_date,
    t.modify_date
FROM sys.tables AS t
INNER JOIN sys.schemas AS s
    ON s.schema_id = t.schema_id
WHERE t.is_ms_shipped = 0
ORDER BY
    s.name,
    t.name;""",
    eval={
        "difficulty": "easy",
        "tags": ["catalog", "sys.tables", "inferred-system"],
        "must_contain": ["sys.tables"],
        "required_objects": ["sys.tables"],
        "rubric": "When inferredSystemQuery is true, list user tables from sys.tables (not invented FitnessApp names as literals).",
    },
)

add(
    id="trace-cont-001",
    completion_category="continuation",
    schema_id="master_onprem",
    inferred_system_query=False,
    user_comment=None,
    statement="select *\nfrom sys.",
    line_prefix="from sys.",
    trace_file="mssql-copilot-trace-2026-04-26T09-39-16-910Z.json",
    event_id="E-2",
    exported_at="2026-04-26T09:39:16.910Z",
    origin_model_id="claude-sonnet-4-6",
    origin_model_family="claude-sonnet",
    origin_model_vendor="anthropic-api",
    origin_result="accepted",
    origin_completion="databases",
    gold_role="cleaned_trace",
    edit_notes="Single identifier continuation after sys. Do not emit sys.databases (dot already present).",
    gold="databases",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "identifier", "sys"],
        "must_contain": ["databases"],
        "must_not_contain": ["sys.databases", ";", "SELECT"],
        "required_objects": ["sys.databases"],
        "rubric": "Cursor is after 'sys.'. Return only 'databases' (or another listed sys.* name). No extra clauses, no semicolon.",
    },
)

add(
    id="trace-cont-002",
    completion_category="continuation",
    schema_id="master_onprem",
    inferred_system_query=False,
    user_comment=None,
    statement="select *\nfrom sys.databases\nwhere ",
    line_prefix="where ",
    trace_file="mssql-copilot-trace-2026-04-26T09-39-16-910Z.json",
    event_id="E-4",
    exported_at="2026-04-26T09:39:16.910Z",
    origin_model_id="claude-sonnet-4-6",
    origin_model_family="claude-sonnet",
    origin_model_vendor="anthropic-api",
    origin_result="accepted",
    origin_completion="state_desc = 'ONLINE'",
    gold_role="cleaned_trace",
    edit_notes="Accepted one-predicate continuation using a listed sys.databases column.",
    gold="state_desc = 'ONLINE'",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "where", "sys.databases"],
        "must_contain": ["state_desc", "ONLINE"],
        "must_not_contain": [";", "SELECT"],
        "rubric": "One WHERE predicate on listed sys.databases columns. No extra AND/ORDER BY/semicolon.",
    },
)

add(
    id="trace-cont-003",
    completion_category="continuation",
    schema_id="master_onprem",
    inferred_system_query=False,
    user_comment=None,
    statement="select *\n  from ",
    line_prefix="  from ",
    recent="-- which requests had the top 5 fastest queries recently per request?",
    trace_file="mssql-copilot-trace-2026-04-26T09-06-28-780Z.json",
    event_id="E-20",
    exported_at="2026-04-26T09:06:28.780Z",
    origin_model_id="claude-haiku-4-5-20251001",
    origin_model_family="claude-haiku",
    origin_model_vendor="anthropic-api",
    origin_result="accepted",
    origin_completion="sys.dm_exec_requests",
    gold_role="cleaned_trace",
    edit_notes="Kept the FROM-target continuation. Dropped later garbage continuations in this session (sys.data_space_stats, markdown, suffix collisions).",
    gold="sys.dm_exec_requests",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "from-clause", "dmv"],
        "must_contain": ["sys.dm_exec_requests"],
        "must_not_contain": [";", "WHERE"],
        "required_objects": ["sys.dm_exec_requests"],
        "rubric": "Complete FROM with one listed object. One unit only.",
    },
)


# ---------------------------------------------------------------------------
# Second pass over the traces (v1.1.0): continuation events that the first
# pass skipped. Several were accepted in the logs but are wrong under the
# rules (invented objects, suffix collisions, markdown); those become
# replaced_trace_gold items, mostly with empty gold.
# ---------------------------------------------------------------------------

_RECENT_TOP5 = (
    "-- which requests had the top 5 fastest queries recently per request?\n"
    "SELECT\n  TOP 5\n  deqs.sql_handle,\n  dest.text AS query_text,\n"
    "  deqs.last_execution_time,\n  deqs.execution_count,\n"
    "  deqs.total_elapsed_time / deqs.execution_count AS avg_elapsed_time_ms,\n"
    "  deqs.total_worker_time / deqs.execution_count AS avg_worker_time_ms\n"
    "FROM\n  sys.dm_exec_query_stats AS deqs\n"
    "  CROSS APPLY sys.dm_exec_sql_text(deqs.sql_handle) AS dest\nWHERE\n"
    "  deqs.last_execution_time >= DATEADD(DAY, -7, GETUTCDATE())\nORDER BY\n"
    "  deqs.total_elapsed_time / deqs.execution_count ASC;\n\n\n  GO"
)

_RECENT_TOP5_PLUS_REQUESTS = (
    _RECENT_TOP5
    + "\n\n  select *\n  from sys.dm_exec_requests\n  where session_id > 0\n"
    + "  and status = 'running'\n\n\n  GO"
)

add(
    id="trace-cont-004",
    completion_category="continuation",
    schema_id="master_onprem",
    inferred_system_query=True,
    user_comment=None,
    recent=_RECENT_TOP5,
    statement="select *\n  from sys.dm_exec_requests\n  where ",
    line_prefix="  where ",
    trace_file="mssql-copilot-trace-2026-04-26T09-05-35-494Z.json",
    event_id="E-16",
    exported_at="2026-04-26T09:05:35.494Z",
    origin_model_id="claude-haiku-4-5-20251001",
    origin_model_family="claude-haiku",
    origin_model_vendor="anthropic-api",
    origin_result="accepted",
    origin_latency_ms=1146,
    origin_input_tokens=2567,
    origin_output_tokens=13,
    origin_completion="session_id > 0",
    gold_role="cleaned_trace",
    edit_notes="Accepted one-predicate WHERE continuation on listed dm_exec_requests columns.",
    gold="session_id > 0",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "where", "dmv", "requests"],
        "must_contain_any": ["session_id", "status", "blocking_session_id", "wait_type", "cpu_time"],
        "must_not_contain": [";", "SELECT"],
        "rubric": "One WHERE predicate on listed sys.dm_exec_requests columns. No extra clauses, no semicolon.",
    },
)

add(
    id="trace-cont-005",
    completion_category="continuation",
    schema_id="master_onprem",
    inferred_system_query=True,
    user_comment=None,
    recent=_RECENT_TOP5,
    statement="select *\n  from sys.dm_exec_requests\n  where session_id > 0\n  and ",
    line_prefix="  and ",
    trace_file="mssql-copilot-trace-2026-04-26T09-05-35-494Z.json",
    event_id="E-19",
    exported_at="2026-04-26T09:05:35.494Z",
    origin_model_id="claude-haiku-4-5-20251001",
    origin_model_family="claude-haiku",
    origin_model_vendor="anthropic-api",
    origin_result="accepted",
    origin_latency_ms=1006,
    origin_input_tokens=2576,
    origin_output_tokens=13,
    origin_completion="status = 'running'",
    gold_role="cleaned_trace",
    edit_notes="Accepted AND-predicate continuation; must not repeat the session_id predicate already present.",
    gold="status = 'running'",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "where", "dmv", "requests"],
        "must_contain_any": ["status", "blocking_session_id", "wait_type", "cpu_time", "command"],
        "must_not_contain": [";", "SELECT", "session_id > 0"],
        "rubric": "One additional AND predicate on listed sys.dm_exec_requests columns. Do not repeat the existing session_id predicate. No semicolon.",
    },
)

add(
    id="trace-cont-006",
    completion_category="continuation",
    schema_id="master_onprem",
    inferred_system_query=True,
    user_comment=None,
    recent=_RECENT_TOP5_PLUS_REQUESTS,
    statement="select *\n  from sys.data",
    line_prefix="  from sys.data",
    trace_file="mssql-copilot-trace-2026-04-26T09-05-35-494Z.json",
    event_id="E-22",
    exported_at="2026-04-26T09:05:35.494Z",
    origin_model_id="claude-haiku-4-5-20251001",
    origin_model_family="claude-haiku",
    origin_model_vendor="anthropic-api",
    origin_result="accepted",
    origin_latency_ms=1042,
    origin_input_tokens=2596,
    origin_output_tokens=11,
    origin_completion="_space_stats",
    gold_role="replaced_trace_gold",
    edit_notes="Origin invented sys.data_space_stats (not a listed object) and the user accepted it. The only listed object whose name continues 'sys.data' is sys.databases, so gold completes the identifier with 'bases'.",
    gold="bases",
    eval={
        "difficulty": "hard",
        "tags": ["continuation", "identifier", "trap", "prefix-aware"],
        "must_contain": ["bases"],
        "must_not_contain": ["space_stats", "sys.databases", ";"],
        "required_objects": ["sys.databases"],
        "rubric": "Cursor is mid-identifier after 'sys.data'. The only listed object that completes it is sys.databases, so return exactly 'bases'. Inventing sys.data_space_stats (as the origin model did) is a fail; repeating 'sys.databases' in full is also a fail because it double-writes the prefix.",
    },
)

add(
    id="trace-cont-007",
    completion_category="continuation",
    schema_id="master_onprem",
    inferred_system_query=True,
    user_comment=None,
    recent=_RECENT_TOP5_PLUS_REQUESTS,
    statement="select *\n  from sys.data_spaces\n  cross join ",
    line_prefix="  cross join ",
    line_suffix="sys.dm_db_index_physical_stats(DB_ID(), NULL, NULL, NULL)",
    document_suffix="sys.dm_db_index_physical_stats(DB_ID(), NULL, NULL, NULL)\nWHERE\n  database_id = DB_ID() AND avg_fragmentation_in_percent > 10\nORDER BY avg_fragmentation_in_percent DESCspace_stats",
    trace_file="mssql-copilot-trace-2026-04-26T09-05-35-494Z.json",
    event_id="E-34",
    exported_at="2026-04-26T09:05:35.494Z",
    origin_model_id="claude-haiku-4-5-20251001",
    origin_model_family="claude-haiku",
    origin_model_vendor="anthropic-api",
    origin_result="success",
    origin_latency_ms=1800,
    origin_input_tokens=2689,
    origin_output_tokens=9,
    origin_completion="APPLY",
    gold_role="replaced_trace_gold",
    edit_notes="Origin emitted 'APPLY', composing to the invalid 'cross join APPLY sys.dm_db...'. The line suffix already contains the join target, so nothing fits at the cursor; gold is empty.",
    gold="",
    eval={
        "difficulty": "hard",
        "tags": ["continuation", "suffix-aware", "empty-policy", "trap"],
        "expect_empty": True,
        "must_not_contain": ["APPLY", "SELECT", "```"],
        "rubric": "The text to the right of the cursor already supplies the join target. Any token inserted here (like the origin's 'APPLY') corrupts the statement. Return exactly an empty string.",
    },
)

add(
    id="trace-cont-008",
    completion_category="continuation",
    schema_id="master_onprem",
    inferred_system_query=True,
    user_comment=None,
    recent=_RECENT_TOP5_PLUS_REQUESTS,
    statement="select *\n  from sys.data.dm_db_index_physical_stats\nFROM\n  sys.dm_db_index_physical_stats(DB_ID(), NULL, NULL, NULL)\nWHERE\n  database_id = DB_ID() AND avg_fragmentation_in_percent > 10\nORDER BY",
    line_prefix="ORDER BY",
    line_suffix="space_stats",
    document_suffix="space_stats",
    trace_file="mssql-copilot-trace-2026-04-26T09-05-35-494Z.json",
    event_id="E-27",
    exported_at="2026-04-26T09:05:35.494Z",
    origin_model_id="claude-haiku-4-5-20251001",
    origin_model_family="claude-haiku",
    origin_model_vendor="anthropic-api",
    origin_result="accepted",
    origin_latency_ms=1408,
    origin_input_tokens=2664,
    origin_output_tokens=16,
    origin_completion="avg_fragmentation_in_percent DESC",
    gold_role="replaced_trace_gold",
    edit_notes="Origin completion collided with the stray 'space_stats' suffix, producing 'DESCspace_stats' in the editor. No completion composes with that suffix; gold is empty.",
    gold="",
    eval={
        "difficulty": "hard",
        "tags": ["continuation", "suffix-aware", "empty-policy", "trap"],
        "expect_empty": True,
        "must_not_contain": ["avg_fragmentation_in_percent", "```", "SELECT"],
        "rubric": "The current line ends with the stray token 'space_stats' immediately right of the cursor. No ORDER BY expression composes with it, so return exactly an empty string. Emitting a sort column here corrupts the line (this is the DESCspace_stats failure from the traces).",
    },
)

add(
    id="trace-cont-009",
    completion_category="continuation",
    schema_id="master_onprem",
    inferred_system_query=True,
    user_comment=None,
    recent=_RECENT_TOP5_PLUS_REQUESTS,
    statement="select *\n  from sys.data.dm_db_index_physical_stats\nFROM\n  sys.dm_db_index_physical_stats(DB_ID(), NULL, NULL, NULL)\nWHERE\n  database_id = DB_ID()",
    line_prefix="  database_id = DB_ID()",
    line_suffix=" AND avg_fragmentation_in_percent > 10",
    document_suffix=" AND avg_fragmentation_in_percent > 10\nORDER BY avg_fragmentation_in_percent DESCspace_stats",
    trace_file="mssql-copilot-trace-2026-04-26T09-05-35-494Z.json",
    event_id="E-30",
    exported_at="2026-04-26T09:05:35.494Z",
    origin_model_id="claude-haiku-4-5-20251001",
    origin_model_family="claude-haiku",
    origin_model_vendor="anthropic-api",
    origin_result="success",
    origin_latency_ms=1326,
    origin_input_tokens=2693,
    origin_output_tokens=39,
    origin_completion="```\nWHERE\n  avg_fragmentation_in_percent > 10\nORDER BY\n  avg_fragmentation_in_percent DESC\n```",
    gold_role="replaced_trace_gold",
    edit_notes="Origin emitted a fenced block duplicating predicates that already exist in the suffix. The suffix already continues the WHERE clause, so gold is empty.",
    gold="",
    eval={
        "difficulty": "medium",
        "tags": ["continuation", "suffix-aware", "empty-policy", "no-markdown"],
        "expect_empty": True,
        "must_not_contain": ["```", "WHERE", "ORDER BY"],
        "rubric": "The document suffix already continues this predicate with AND / ORDER BY. Nothing fits at the cursor; return exactly an empty string. Markdown fences are always a fail.",
    },
)

add(
    id="trace-cont-010",
    completion_category="continuation",
    schema_id="master_onprem",
    inferred_system_query=False,
    user_comment=None,
    recent=(
        "-- what are the \nSELECT\n  s.session_id,\n  s.login_name,\n  s.host_name,\n"
        "  s.program_name,\n  s.status,\n  s.cpu_time,\n  s.memory_usage,\n  r.command,\n"
        "  r.status,\n  r.wait_type,\n  r.wait_time,\n"
        "  SUBSTRING(st.text, r.statement_start_offset / 2 + 1, (CASE WHEN r.statement_end_offset = -1 "
        "THEN LEN(CONVERT(NVARCHAR(MAX), st.text)) ELSE r.statement_end_offset / 2 END - r.statement_start_offset / 2))\n"
        "FROM\n  sys.dm_exec_sessions AS s\n  LEFT JOIN sys.dm_exec_requests AS r ON s.session_id = r.session_id\n"
        "  LEFT JOIN sys.dm_exec_sql_text(r.sql_handle) AS st ON r.sql_handle = st.sql_handle\nWHERE\n"
        "  s.session_id > 50\nORDER BY\n  s.cpu_time DESC;\n\n"
        "  -- which tables have the most rows and use the most space\n  SELECT\n"
        "  s.name AS schema_name,\n  t.name AS table_name,\n  p.rows,\n"
        "  (au.total_pages * 8) / 1024.0 AS space_used_mb\nFROM\n  sys.tables AS t\n"
        "  INNER JOIN sys.schemas AS s ON t.schema_id = s.schema_id\n"
        "  INNER JOIN sys.partitions AS p ON t.object_id = p.object_id\n"
        "  INNER JOIN sys.allocation_units AS au ON p.partition_id = au.container_id\nWHERE\n"
        "  p.index_id IN (0, 1)\nGROUP BY\n  s.name,\n  t.name,\n  p.rows,\n  au.total_pages\nORDER BY\n"
        "  p.rows DESC,\n  (au.total_pages * 8) / 1024.0 DESC;"
    ),
    statement="",
    line_prefix="  ",
    trace_file="mssql-copilot-trace-2026-04-26T09-06-28-780Z.json",
    event_id="E-50",
    exported_at="2026-04-26T09:06:28.780Z",
    origin_model_id="claude-haiku-4-5-20251001",
    origin_model_family="claude-haiku",
    origin_model_vendor="anthropic-api",
    origin_result="emptyFromSanitizer",
    origin_latency_ms=2187,
    origin_input_tokens=2830,
    origin_output_tokens=65,
    origin_completion="Since the cursor is at the beginning of a new line after two complete queries, and there is no current statement prefix or line content, I should return an empty string. The document is complete with two standalone queries, and there is no natural single-unit continuation to insert at this position.",
    gold_role="replaced_trace_gold",
    edit_notes="Origin reasoned correctly but wrote the reasoning as prose, which the sanitizer emptied. Gold is the empty string with no characters at all.",
    gold="",
    eval={
        "difficulty": "medium",
        "tags": ["continuation", "empty-policy", "no-prose"],
        "expect_empty": True,
        "must_not_contain": ["SELECT", "```", "empty string", "cursor"],
        "rubric": "Blank line after two complete statements with no new comment or partial token. There is no single-unit continuation to offer; return exactly an empty string, never prose about returning one.",
    },
)

add(
    id="trace-cont-011",
    completion_category="continuation",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    user_comment=None,
    recent=(
        "select *\nfrom sys.databases\nwhere state_desc = 'ONLINE'\n\n\n"
        "-- which tables are in the db\nSELECT\n    t.name AS table_name,\n"
        "    s.name AS schema_name,\n    t.create_date,\n    t.modify_date\n"
        "FROM sys.tables t\nJOIN sys.schemas s ON t.schema_id = s.schema_id\n"
        "WHERE t.is_ms_shipped = 0\nORDER BY s.name, t.name;"
    ),
    statement="",
    line_prefix="ORDER BY s.name, t.name;",
    trace_file="mssql-copilot-trace-2026-04-26T09-39-16-910Z.json",
    event_id="E-15",
    exported_at="2026-04-26T09:39:16.910Z",
    origin_model_id="claude-sonnet-4-6",
    origin_model_family="claude-sonnet",
    origin_model_vendor="anthropic-api",
    origin_result="emptyFromSanitizer",
    origin_latency_ms=2512,
    origin_input_tokens=2310,
    origin_output_tokens=90,
    origin_completion="\n-- which columns are in each table\nSELECT\n    t.name AS table_name,\n    c.name AS column_name,\n    c.column_id,\n    c.is_nullable\nFROM sys.tables t\nJOIN sys.columns c ON t.object_id = c.object_id\nORDER BY t.name, c.column_id;",
    gold_role="replaced_trace_gold",
    edit_notes="Cursor sits at the end of a complete, semicolon-terminated statement. Origin volunteered a whole new comment plus query, which is not a single ghost-text unit; gold is empty.",
    gold="",
    eval={
        "difficulty": "medium",
        "tags": ["continuation", "empty-policy", "statement-complete"],
        "expect_empty": True,
        "must_not_contain": ["SELECT", "--", "```"],
        "rubric": "The statement before the cursor is complete and terminated. Continuation mode must not start a new comment or a new query; return exactly an empty string.",
    },
)
