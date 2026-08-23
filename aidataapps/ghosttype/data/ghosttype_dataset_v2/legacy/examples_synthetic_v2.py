"""Synthetic eval items, second block (dataset v1.1.0).

Authored as claude-fable-5 reference completions against the frozen catalogs
in schemas/. This block extends coverage to SQL Server Agent / backup devops
(msdb_onprem), an Azure SQL database with Query Store enabled
(adventure_azure), a Microsoft Fabric warehouse with queryinsights
(fabric_warehouse), plus ANSI-idiom and formatting-robustness items on the
original catalogs. Same prompt shape and product rules as examples_synthetic.
"""

SYN2 = []


def add(**kwargs):
    kwargs.setdefault("kind", "synthetic")
    kwargs.setdefault("gold_author", "claude-fable-5")
    kwargs.setdefault("gold_model", "claude-fable-5")
    kwargs.setdefault("gold_role", "synthetic")
    kwargs.setdefault("origin_model_id", "claude-fable-5")
    kwargs.setdefault("origin_model_family", "claude-fable")
    kwargs.setdefault("origin_model_vendor", "anthropic-api")
    kwargs.setdefault("edit_notes", "Synthetic prompt and claude-fable-5 reference completion.")
    SYN2.append(kwargs)


# ===========================================================================
# msdb_onprem — SQL Server Agent jobs, backup/restore history (devops/admin)
# ===========================================================================

add(
    id="syn2-msdb-001",
    completion_category="intent",
    schema_id="msdb_onprem",
    inferred_system_query=False,
    scenario="devops",
    user_comment="-- list all agent jobs and whether they are enabled",
    gold="""SELECT
    job_id,
    name,
    enabled,
    description,
    date_created,
    date_modified
FROM dbo.sysjobs
ORDER BY
    name;""",
    eval={
        "difficulty": "easy",
        "tags": ["agent-job", "jobs", "msdb"],
        "must_contain": ["dbo.sysjobs", "enabled"],
        "required_objects": ["dbo.sysjobs"],
        "rubric": "List agent jobs from dbo.sysjobs with the enabled flag. Do not invent msdb views that are not listed.",
    },
)

add(
    id="syn2-msdb-002",
    completion_category="intent",
    schema_id="msdb_onprem",
    inferred_system_query=False,
    scenario="devops",
    user_comment="-- which agent jobs are disabled",
    gold="""SELECT
    job_id,
    name,
    description,
    date_modified
FROM dbo.sysjobs
WHERE enabled = 0
ORDER BY
    name;""",
    eval={
        "difficulty": "easy",
        "tags": ["agent-job", "jobs", "filter"],
        "must_contain": ["dbo.sysjobs", "enabled = 0"],
        "required_objects": ["dbo.sysjobs"],
        "rubric": "Filter dbo.sysjobs on enabled = 0. The enabled column is tinyint; 0 means disabled.",
    },
)

add(
    id="syn2-msdb-003",
    completion_category="intent",
    schema_id="msdb_onprem",
    inferred_system_query=False,
    scenario="devops",
    user_comment="-- failed job runs in the last day",
    gold="""SELECT
    j.name AS job_name,
    h.step_id,
    h.step_name,
    h.run_date,
    h.run_time,
    h.run_duration,
    h.message
FROM dbo.sysjobhistory AS h
INNER JOIN dbo.sysjobs AS j
    ON j.job_id = h.job_id
WHERE h.run_status = 0
  AND h.run_date >= CONVERT(int, CONVERT(char(8), DATEADD(DAY, -1, GETDATE()), 112))
ORDER BY
    h.instance_id DESC;""",
    eval={
        "difficulty": "hard",
        "tags": ["agent-job", "jobs", "history", "date-int"],
        "must_contain": ["dbo.sysjobhistory", "run_status"],
        "must_contain_any": ["112", "run_date"],
        "required_objects": ["dbo.sysjobhistory", "dbo.sysjobs"],
        "rubric": "Failed runs are run_status = 0 in dbo.sysjobhistory. run_date is an int in yyyymmdd form, so the last-day filter needs a conversion (style 112 or equivalent), not a direct datetime comparison.",
    },
)

add(
    id="syn2-msdb-004",
    completion_category="intent",
    schema_id="msdb_onprem",
    inferred_system_query=False,
    scenario="devops",
    user_comment="-- jobs that have no schedule",
    gold="""SELECT
    j.job_id,
    j.name,
    j.enabled
FROM dbo.sysjobs AS j
WHERE NOT EXISTS (
    SELECT 1
    FROM dbo.sysjobschedules AS js
    WHERE js.job_id = j.job_id
)
ORDER BY
    j.name;""",
    eval={
        "difficulty": "medium",
        "tags": ["agent-job", "jobs", "anti-join", "schedules"],
        "must_contain": ["dbo.sysjobs", "dbo.sysjobschedules"],
        "must_contain_any": ["NOT EXISTS", "LEFT JOIN"],
        "required_objects": ["dbo.sysjobs", "dbo.sysjobschedules"],
        "rubric": "Anti-join sysjobs against sysjobschedules (NOT EXISTS or LEFT JOIN ... IS NULL both fine).",
    },
)

add(
    id="syn2-msdb-005",
    completion_category="intent",
    schema_id="msdb_onprem",
    inferred_system_query=False,
    scenario="devops",
    user_comment="-- average run duration per job for successful runs",
    gold="""SELECT
    j.name AS job_name,
    COUNT(*) AS successful_runs,
    AVG(h.run_duration) AS avg_run_duration
FROM dbo.sysjobhistory AS h
INNER JOIN dbo.sysjobs AS j
    ON j.job_id = h.job_id
WHERE h.run_status = 1
GROUP BY
    j.name
ORDER BY
    avg_run_duration DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["agent-job", "history", "aggregate"],
        "must_contain": ["dbo.sysjobhistory", "AVG", "run_status = 1"],
        "required_objects": ["dbo.sysjobhistory", "dbo.sysjobs"],
        "rubric": "Aggregate run_duration for run_status = 1 grouped by job. run_duration is stored as hhmmss-packed int; a plain AVG is the accepted grounded approximation here.",
    },
)

add(
    id="syn2-msdb-006",
    completion_category="intent",
    schema_id="msdb_onprem",
    inferred_system_query=False,
    scenario="devops",
    user_comment="-- enabled schedules and the next run date for each job",
    gold="""SELECT
    j.name AS job_name,
    s.name AS schedule_name,
    js.next_run_date,
    js.next_run_time
FROM dbo.sysjobschedules AS js
INNER JOIN dbo.sysjobs AS j
    ON j.job_id = js.job_id
INNER JOIN dbo.sysschedules AS s
    ON s.schedule_id = js.schedule_id
WHERE s.enabled = 1
ORDER BY
    js.next_run_date,
    js.next_run_time;""",
    eval={
        "difficulty": "medium",
        "tags": ["agent-job", "schedules", "joins"],
        "must_contain": ["dbo.sysjobschedules", "dbo.sysschedules", "next_run_date"],
        "required_objects": ["dbo.sysjobschedules", "dbo.sysschedules", "dbo.sysjobs"],
        "rubric": "Three-way join jobs -> jobschedules -> schedules, filtered to enabled schedules, projecting next_run_date/time.",
    },
)

add(
    id="syn2-msdb-007",
    completion_category="intent",
    schema_id="msdb_onprem",
    inferred_system_query=False,
    scenario="devops",
    user_comment="-- most recent full backup for each database",
    gold="""SELECT
    database_name,
    MAX(backup_finish_date) AS last_full_backup
FROM dbo.backupset
WHERE type = 'D'
GROUP BY
    database_name
ORDER BY
    last_full_backup DESC;""",
    eval={
        "difficulty": "easy",
        "tags": ["backup", "aggregate"],
        "must_contain": ["dbo.backupset", "MAX(backup_finish_date)", "'D'"],
        "required_objects": ["dbo.backupset"],
        "rubric": "Full backups are type = 'D' in dbo.backupset. MAX(backup_finish_date) per database_name.",
    },
)

add(
    id="syn2-msdb-008",
    completion_category="intent",
    schema_id="msdb_onprem",
    inferred_system_query=False,
    scenario="devops",
    user_comment="-- databases with no full backup in the last 7 days",
    gold="""SELECT
    d.name AS database_name
FROM sys.databases AS d
WHERE NOT EXISTS (
    SELECT 1
    FROM dbo.backupset AS b
    WHERE b.database_name = d.name
      AND b.type = 'D'
      AND b.backup_finish_date >= DATEADD(DAY, -7, GETDATE())
)
ORDER BY
    d.name;""",
    eval={
        "difficulty": "hard",
        "tags": ["backup", "anti-join", "dmv"],
        "must_contain": ["sys.databases", "dbo.backupset", "DATEADD"],
        "must_contain_any": ["NOT EXISTS", "LEFT JOIN"],
        "required_objects": ["sys.databases", "dbo.backupset"],
        "rubric": "Anti-join sys.databases against recent type 'D' rows in dbo.backupset. Querying backupset alone misses databases that were never backed up.",
    },
)

add(
    id="syn2-msdb-009",
    completion_category="intent",
    schema_id="msdb_onprem",
    inferred_system_query=False,
    scenario="devops",
    user_comment="-- backup history for the master database, newest first",
    gold="""SELECT
    backup_set_id,
    type,
    backup_start_date,
    backup_finish_date,
    backup_size,
    compressed_backup_size,
    user_name
FROM dbo.backupset
WHERE database_name = N'master'
ORDER BY
    backup_finish_date DESC;""",
    eval={
        "difficulty": "easy",
        "tags": ["backup", "filter"],
        "must_contain": ["dbo.backupset", "master"],
        "required_objects": ["dbo.backupset"],
        "rubric": "Filter dbo.backupset on database_name = 'master', newest finish date first.",
    },
)

add(
    id="syn2-msdb-010",
    completion_category="intent",
    schema_id="msdb_onprem",
    inferred_system_query=False,
    scenario="devops",
    user_comment="-- where were the last backups written on disk",
    gold="""SELECT
    b.database_name,
    b.backup_finish_date,
    b.type,
    mf.physical_device_name,
    mf.device_type
FROM dbo.backupset AS b
INNER JOIN dbo.backupmediafamily AS mf
    ON mf.media_set_id = b.media_set_id
ORDER BY
    b.backup_finish_date DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["backup", "joins"],
        "must_contain": ["dbo.backupmediafamily", "physical_device_name"],
        "required_objects": ["dbo.backupset", "dbo.backupmediafamily"],
        "rubric": "Join backupset to backupmediafamily on media_set_id to surface physical_device_name.",
    },
)

add(
    id="syn2-msdb-011",
    completion_category="intent",
    schema_id="msdb_onprem",
    inferred_system_query=False,
    scenario="devops",
    user_comment="-- total backup size per database per month this year",
    gold="""SELECT
    database_name,
    YEAR(backup_finish_date) AS backup_year,
    MONTH(backup_finish_date) AS backup_month,
    SUM(backup_size) AS total_backup_size,
    SUM(compressed_backup_size) AS total_compressed_size
FROM dbo.backupset
WHERE backup_finish_date >= DATEADD(YEAR, DATEDIFF(YEAR, 0, GETDATE()), 0)
GROUP BY
    database_name,
    YEAR(backup_finish_date),
    MONTH(backup_finish_date)
ORDER BY
    database_name,
    backup_year,
    backup_month;""",
    eval={
        "difficulty": "medium",
        "tags": ["backup", "aggregate", "date-windows"],
        "must_contain": ["dbo.backupset", "SUM(backup_size)", "MONTH"],
        "required_objects": ["dbo.backupset"],
        "rubric": "Group backup_size by database and calendar month, restricted to the current year.",
    },
)

add(
    id="syn2-msdb-012",
    completion_category="intent",
    schema_id="msdb_onprem",
    inferred_system_query=False,
    scenario="devops",
    user_comment="-- what restores have happened recently and from which backup",
    gold="""SELECT
    r.restore_date,
    r.destination_database_name,
    r.restore_type,
    r.user_name,
    b.database_name AS source_database,
    b.backup_finish_date
FROM dbo.restorehistory AS r
LEFT JOIN dbo.backupset AS b
    ON b.backup_set_id = r.backup_set_id
ORDER BY
    r.restore_date DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["restore", "backup", "joins"],
        "must_contain": ["dbo.restorehistory", "restore_date"],
        "required_objects": ["dbo.restorehistory", "dbo.backupset"],
        "rubric": "List restorehistory newest first; joining backupset on backup_set_id for the source backup is expected.",
    },
)

add(
    id="syn2-msdb-013",
    completion_category="intent",
    schema_id="msdb_onprem",
    inferred_system_query=False,
    scenario="devops",
    user_comment="-- which jobs are running right now",
    gold="""SELECT
    j.name AS job_name,
    a.start_execution_date,
    a.last_executed_step_id,
    a.next_scheduled_run_date
FROM dbo.sysjobactivity AS a
INNER JOIN dbo.sysjobs AS j
    ON j.job_id = a.job_id
WHERE a.start_execution_date IS NOT NULL
  AND a.stop_execution_date IS NULL
ORDER BY
    a.start_execution_date;""",
    eval={
        "difficulty": "medium",
        "tags": ["agent-job", "jobs", "activity"],
        "must_contain": ["dbo.sysjobactivity", "stop_execution_date IS NULL"],
        "required_objects": ["dbo.sysjobactivity", "dbo.sysjobs"],
        "rubric": "Running jobs have a start_execution_date and a NULL stop_execution_date in sysjobactivity.",
    },
)

add(
    id="syn2-msdb-014",
    completion_category="intent",
    schema_id="msdb_onprem",
    inferred_system_query=False,
    scenario="devops",
    user_comment="-- list operators and their email addresses",
    gold="""SELECT
    id,
    name,
    enabled,
    email_address
FROM dbo.sysoperators
ORDER BY
    name;""",
    eval={
        "difficulty": "easy",
        "tags": ["agent-job", "operators"],
        "must_contain": ["dbo.sysoperators", "email_address"],
        "required_objects": ["dbo.sysoperators"],
        "rubric": "Simple listing of dbo.sysoperators with email_address.",
    },
)

add(
    id="syn2-msdb-015",
    completion_category="intent",
    schema_id="msdb_onprem",
    inferred_system_query=False,
    scenario="devops",
    user_comment="-- alerts that have fired, most frequent first",
    gold="""SELECT
    name,
    message_id,
    severity,
    enabled,
    occurrence_count,
    last_occurrence_date
FROM dbo.sysalerts
WHERE occurrence_count > 0
ORDER BY
    occurrence_count DESC;""",
    eval={
        "difficulty": "easy",
        "tags": ["agent-job", "alerts"],
        "must_contain": ["dbo.sysalerts", "occurrence_count"],
        "required_objects": ["dbo.sysalerts"],
        "rubric": "Filter sysalerts to occurrence_count > 0, ordered by occurrence_count descending.",
    },
)

add(
    id="syn2-msdb-016",
    completion_category="intent",
    schema_id="msdb_onprem",
    inferred_system_query=False,
    scenario="devops",
    user_comment="-- job steps that failed on their last run, with the command they executed",
    gold="""SELECT
    j.name AS job_name,
    s.step_id,
    s.step_name,
    s.subsystem,
    s.command,
    s.last_run_date,
    s.retry_attempts
FROM dbo.sysjobsteps AS s
INNER JOIN dbo.sysjobs AS j
    ON j.job_id = s.job_id
WHERE s.last_run_outcome = 0
ORDER BY
    j.name,
    s.step_id;""",
    eval={
        "difficulty": "medium",
        "tags": ["agent-job", "jobs", "steps"],
        "must_contain": ["dbo.sysjobsteps", "last_run_outcome = 0", "command"],
        "required_objects": ["dbo.sysjobsteps", "dbo.sysjobs"],
        "rubric": "sysjobsteps.last_run_outcome = 0 marks a failed last run; include the command column.",
    },
)

add(
    id="syn2-msdb-017",
    completion_category="intent",
    schema_id="msdb_onprem",
    inferred_system_query=False,
    scenario="devops",
    user_comment="-- show the ssis package execution report from the integration services catalog",
    gold="",
    eval={
        "difficulty": "medium",
        "tags": ["empty-policy", "trap", "devops"],
        "expect_empty": True,
        "must_not_contain": ["catalog.executions", "SSISDB", "SELECT", "```"],
        "rubric": "The SSISDB catalog views are not in this schema context (sysssislog appears only as a names-only inventory entry). Return exactly an empty string; do not invent catalog.executions.",
    },
)

add(
    id="syn2-msdb-018",
    completion_category="intent",
    schema_id="msdb_onprem",
    inferred_system_query=False,
    scenario="devops",
    user_comment="-- send a test email to the dba team using database mail",
    gold="",
    eval={
        "difficulty": "medium",
        "tags": ["empty-policy", "trap", "devops"],
        "expect_empty": True,
        "must_not_contain": ["sp_send_dbmail", "EXEC", "SELECT", "```"],
        "rubric": "sp_send_dbmail is not a listed routine in this snapshot. Return exactly an empty string rather than calling an unlisted procedure.",
    },
)

add(
    id="syn2-msdb-019",
    completion_category="continuation",
    schema_id="msdb_onprem",
    inferred_system_query=False,
    scenario="devops",
    user_comment=None,
    statement="SELECT name, enabled\nFROM dbo.sysjobs\nWHERE ",
    line_prefix="WHERE ",
    gold="enabled = 0",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "where", "agent-job"],
        "must_contain_any": ["enabled", "name", "date_created", "category_id"],
        "must_not_contain": [";", "SELECT"],
        "rubric": "One WHERE predicate on listed dbo.sysjobs columns. No semicolon, no extra clauses.",
    },
)

add(
    id="syn2-msdb-020",
    completion_category="continuation",
    schema_id="msdb_onprem",
    inferred_system_query=False,
    scenario="devops",
    user_comment=None,
    statement="SELECT j.name, h.run_status, h.run_date\nFROM dbo.sysjobs AS j\nINNER JOIN dbo.sysjobhistory AS h\n    ON ",
    line_prefix="    ON ",
    gold="h.job_id = j.job_id",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "join-on", "agent-job", "fk"],
        "must_contain": ["job_id"],
        "must_not_contain": [";", "WHERE"],
        "rubric": "Complete the ON clause using the sysjobhistory.job_id -> sysjobs.job_id foreign key. One unit, no semicolon.",
    },
)

add(
    id="syn2-msdb-021",
    completion_category="continuation",
    schema_id="msdb_onprem",
    inferred_system_query=False,
    scenario="devops",
    user_comment=None,
    statement="SELECT database_name, backup_finish_date\nFROM dbo.backup",
    line_prefix="FROM dbo.backup",
    gold="set",
    eval={
        "difficulty": "hard",
        "tags": ["continuation", "identifier", "prefix-aware", "backup"],
        "must_contain": ["set"],
        "must_not_contain": ["media", ";", "backupset"],
        "rubric": "Three listed tables begin with 'backup', but only dbo.backupset has the selected columns database_name and backup_finish_date. Complete the identifier with exactly 'set'; repeating 'backupset' in full double-writes the prefix.",
    },
)


# ===========================================================================
# adventure_azure — AdventureWorksLT-style Azure SQL DB with Query Store
# ===========================================================================

add(
    id="syn2-adv-001",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="developer",
    user_comment="-- list customers with their company names",
    gold="""SELECT
    CustomerID,
    FirstName,
    LastName,
    CompanyName,
    EmailAddress,
    Phone
FROM SalesLT.Customer
ORDER BY
    LastName,
    FirstName;""",
    eval={
        "difficulty": "easy",
        "tags": ["user-schema", "select", "privacy"],
        "must_contain": ["SalesLT.Customer", "CompanyName"],
        "must_not_contain": ["PasswordHash"],
        "required_objects": ["SalesLT.Customer"],
        "rubric": "List SalesLT.Customer columns. Do not project PasswordHash unless asked.",
    },
)

add(
    id="syn2-adv-002",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="developer",
    user_comment="-- products with a list price over 1000",
    gold="""SELECT
    ProductID,
    Name,
    ProductNumber,
    Color,
    ListPrice
FROM SalesLT.Product
WHERE ListPrice > 1000
ORDER BY
    ListPrice DESC;""",
    eval={
        "difficulty": "easy",
        "tags": ["user-schema", "filter"],
        "must_contain": ["SalesLT.Product", "ListPrice > 1000"],
        "required_objects": ["SalesLT.Product"],
        "rubric": "Filter SalesLT.Product on ListPrice > 1000.",
    },
)

add(
    id="syn2-adv-003",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="developer",
    user_comment="-- red or black products",
    gold="""SELECT
    ProductID,
    Name,
    Color,
    ListPrice
FROM SalesLT.Product
WHERE Color IN (N'Red', N'Black')
ORDER BY
    Name;""",
    eval={
        "difficulty": "easy",
        "tags": ["user-schema", "filter", "predicate"],
        "must_contain": ["SalesLT.Product", "Color"],
        "required_objects": ["SalesLT.Product"],
        "rubric": "Filter on the Color column with an IN list or OR predicates.",
    },
)

add(
    id="syn2-adv-004",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="developer",
    user_comment="-- total amount due per customer, highest first",
    gold="""SELECT
    c.CustomerID,
    c.FirstName,
    c.LastName,
    c.CompanyName,
    SUM(soh.TotalDue) AS total_due
FROM SalesLT.Customer AS c
INNER JOIN SalesLT.SalesOrderHeader AS soh
    ON soh.CustomerID = c.CustomerID
GROUP BY
    c.CustomerID,
    c.FirstName,
    c.LastName,
    c.CompanyName
ORDER BY
    total_due DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["user-schema", "joins", "aggregate"],
        "must_contain": ["SalesLT.SalesOrderHeader", "SUM", "TotalDue"],
        "required_objects": ["SalesLT.Customer", "SalesLT.SalesOrderHeader"],
        "rubric": "Join Customer to SalesOrderHeader on CustomerID and sum TotalDue per customer.",
    },
)

add(
    id="syn2-adv-005",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="developer",
    user_comment="-- top 5 products by total quantity sold",
    gold="""SELECT TOP (5)
    p.ProductID,
    p.Name,
    SUM(sod.OrderQty) AS total_qty
FROM SalesLT.SalesOrderDetail AS sod
INNER JOIN SalesLT.Product AS p
    ON p.ProductID = sod.ProductID
GROUP BY
    p.ProductID,
    p.Name
ORDER BY
    total_qty DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["user-schema", "joins", "aggregate", "top-n"],
        "must_contain": ["SalesLT.SalesOrderDetail", "OrderQty", "TOP"],
        "required_objects": ["SalesLT.SalesOrderDetail", "SalesLT.Product"],
        "rubric": "Sum OrderQty per product from SalesOrderDetail joined to Product, TOP 5 by that sum.",
    },
)

add(
    id="syn2-adv-006",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="developer",
    user_comment="-- customers who have never placed an order",
    gold="""SELECT
    c.CustomerID,
    c.FirstName,
    c.LastName,
    c.CompanyName
FROM SalesLT.Customer AS c
WHERE NOT EXISTS (
    SELECT 1
    FROM SalesLT.SalesOrderHeader AS soh
    WHERE soh.CustomerID = c.CustomerID
)
ORDER BY
    c.CustomerID;""",
    eval={
        "difficulty": "medium",
        "tags": ["user-schema", "anti-join"],
        "must_contain": ["SalesLT.Customer", "SalesLT.SalesOrderHeader"],
        "must_contain_any": ["NOT EXISTS", "LEFT JOIN"],
        "required_objects": ["SalesLT.Customer", "SalesLT.SalesOrderHeader"],
        "rubric": "Anti-join Customer against SalesOrderHeader (NOT EXISTS or LEFT JOIN ... IS NULL).",
    },
)

add(
    id="syn2-adv-007",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="analytics",
    user_comment="-- monthly revenue for 2008",
    gold="""SELECT
    MONTH(OrderDate) AS order_month,
    SUM(TotalDue) AS revenue
FROM SalesLT.SalesOrderHeader
WHERE YEAR(OrderDate) = 2008
GROUP BY
    MONTH(OrderDate)
ORDER BY
    order_month;""",
    eval={
        "difficulty": "medium",
        "tags": ["aggregate", "date-windows", "analytics"],
        "must_contain": ["SalesLT.SalesOrderHeader", "TotalDue", "2008"],
        "must_contain_any": ["MONTH", "DATEPART"],
        "required_objects": ["SalesLT.SalesOrderHeader"],
        "rubric": "Sum TotalDue per month of OrderDate restricted to 2008.",
    },
)

add(
    id="syn2-adv-008",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="analytics",
    user_comment="-- rank products by revenue within each category",
    gold="""SELECT
    pc.Name AS category_name,
    p.Name AS product_name,
    SUM(sod.LineTotal) AS revenue,
    RANK() OVER (
        PARTITION BY pc.Name
        ORDER BY SUM(sod.LineTotal) DESC
    ) AS revenue_rank
FROM SalesLT.SalesOrderDetail AS sod
INNER JOIN SalesLT.Product AS p
    ON p.ProductID = sod.ProductID
INNER JOIN SalesLT.ProductCategory AS pc
    ON pc.ProductCategoryID = p.ProductCategoryID
GROUP BY
    pc.Name,
    p.Name
ORDER BY
    category_name,
    revenue_rank;""",
    eval={
        "difficulty": "hard",
        "tags": ["window", "aggregate", "joins", "analytics"],
        "must_contain": ["OVER", "PARTITION BY", "LineTotal"],
        "required_objects": ["SalesLT.SalesOrderDetail", "SalesLT.Product", "SalesLT.ProductCategory"],
        "rubric": "Window rank over summed LineTotal partitioned by category. Requires joining detail -> product -> category.",
    },
)

add(
    id="syn2-adv-009",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="analytics",
    user_comment="-- running total of daily revenue",
    gold="""WITH daily AS (
    SELECT
        CAST(OrderDate AS date) AS order_day,
        SUM(TotalDue) AS revenue
    FROM SalesLT.SalesOrderHeader
    GROUP BY
        CAST(OrderDate AS date)
)
SELECT
    order_day,
    revenue,
    SUM(revenue) OVER (
        ORDER BY order_day
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_revenue
FROM daily
ORDER BY
    order_day;""",
    eval={
        "difficulty": "hard",
        "tags": ["window", "running-total", "cte", "analytics"],
        "must_contain": ["OVER", "SalesLT.SalesOrderHeader"],
        "must_contain_any": ["UNBOUNDED PRECEDING", "ROWS"],
        "required_objects": ["SalesLT.SalesOrderHeader"],
        "rubric": "Aggregate revenue per day, then a windowed running SUM ordered by day.",
    },
)

add(
    id="syn2-adv-010",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="developer",
    user_comment="-- orders that shipped after their due date",
    gold="""SELECT
    SalesOrderID,
    SalesOrderNumber,
    OrderDate,
    DueDate,
    ShipDate,
    CustomerID
FROM SalesLT.SalesOrderHeader
WHERE ShipDate > DueDate
ORDER BY
    ShipDate;""",
    eval={
        "difficulty": "easy",
        "tags": ["user-schema", "filter", "dates"],
        "must_contain": ["ShipDate > DueDate"],
        "required_objects": ["SalesLT.SalesOrderHeader"],
        "rubric": "Compare ShipDate to DueDate on SalesOrderHeader.",
    },
)

add(
    id="syn2-adv-011",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="developer",
    user_comment="-- raise list price by 5 percent for discontinued products",
    gold="""UPDATE SalesLT.Product
SET ListPrice = ListPrice * 1.05
WHERE DiscontinuedDate IS NOT NULL;""",
    eval={
        "difficulty": "medium",
        "tags": ["dml", "update"],
        "must_contain": ["UPDATE SalesLT.Product", "1.05", "DiscontinuedDate IS NOT NULL"],
        "required_objects": ["SalesLT.Product"],
        "rubric": "UPDATE with a multiplicative SET; discontinued means DiscontinuedDate IS NOT NULL (there is no boolean flag column).",
    },
)

add(
    id="syn2-adv-012",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="developer",
    user_comment="-- delete order detail rows with zero quantity",
    gold="""DELETE FROM SalesLT.SalesOrderDetail
WHERE OrderQty = 0;""",
    eval={
        "difficulty": "easy",
        "tags": ["dml", "delete"],
        "must_contain": ["DELETE", "SalesLT.SalesOrderDetail", "OrderQty = 0"],
        "required_objects": ["SalesLT.SalesOrderDetail"],
        "rubric": "Simple DELETE with the OrderQty predicate. No extra filters.",
    },
)

add(
    id="syn2-adv-013",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="developer",
    user_comment="-- product categories with their parent category names",
    gold="""SELECT
    child.ProductCategoryID,
    child.Name AS category_name,
    parent.Name AS parent_category_name
FROM SalesLT.ProductCategory AS child
LEFT JOIN SalesLT.ProductCategory AS parent
    ON parent.ProductCategoryID = child.ParentProductCategoryID
ORDER BY
    parent_category_name,
    category_name;""",
    eval={
        "difficulty": "medium",
        "tags": ["user-schema", "self-join"],
        "must_contain": ["SalesLT.ProductCategory", "ParentProductCategoryID"],
        "required_objects": ["SalesLT.ProductCategory"],
        "rubric": "Self-join ProductCategory on ParentProductCategoryID; LEFT JOIN keeps root categories.",
    },
)

add(
    id="syn2-adv-014",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="developer",
    user_comment="-- add a new product category named Gravel Bikes under parent category 1",
    gold="""INSERT INTO SalesLT.ProductCategory (ParentProductCategoryID, Name, ModifiedDate)
VALUES (1, N'Gravel Bikes', GETDATE());""",
    eval={
        "difficulty": "medium",
        "tags": ["dml", "insert"],
        "must_contain": ["INSERT INTO SalesLT.ProductCategory", "Gravel Bikes"],
        "required_objects": ["SalesLT.ProductCategory"],
        "rubric": "INSERT with listed columns only; ProductCategoryID is the PK and should not be supplied explicitly.",
    },
)

add(
    id="syn2-adv-015",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="analytics",
    user_comment="-- year over year revenue with the change from the prior year",
    gold="""WITH yearly AS (
    SELECT
        YEAR(OrderDate) AS order_year,
        SUM(TotalDue) AS revenue
    FROM SalesLT.SalesOrderHeader
    GROUP BY
        YEAR(OrderDate)
)
SELECT
    order_year,
    revenue,
    revenue - LAG(revenue) OVER (ORDER BY order_year) AS revenue_change
FROM yearly
ORDER BY
    order_year;""",
    eval={
        "difficulty": "hard",
        "tags": ["window", "yoy", "cte", "analytics"],
        "must_contain": ["LAG", "OVER", "YEAR(OrderDate)"],
        "required_objects": ["SalesLT.SalesOrderHeader"],
        "rubric": "Yearly revenue CTE plus LAG to compute the delta from the prior year.",
    },
)

add(
    id="syn2-adv-016",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="developer",
    user_comment="-- how many orders are in each status",
    gold="""SELECT
    Status,
    COUNT(*) AS order_count
FROM SalesLT.SalesOrderHeader
GROUP BY
    Status
ORDER BY
    order_count DESC;""",
    eval={
        "difficulty": "easy",
        "tags": ["aggregate"],
        "must_contain": ["Status", "COUNT(*)"],
        "required_objects": ["SalesLT.SalesOrderHeader"],
        "rubric": "COUNT grouped by the Status column.",
    },
)

add(
    id="syn2-adv-017",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="admin",
    user_comment="-- top 10 queries by average duration in query store",
    gold="""SELECT TOP (10)
    q.query_id,
    qt.query_sql_text,
    rs.avg_duration,
    rs.count_executions,
    p.plan_id
FROM sys.query_store_runtime_stats AS rs
INNER JOIN sys.query_store_plan AS p
    ON p.plan_id = rs.plan_id
INNER JOIN sys.query_store_query AS q
    ON q.query_id = p.query_id
INNER JOIN sys.query_store_query_text AS qt
    ON qt.query_text_id = q.query_text_id
ORDER BY
    rs.avg_duration DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["query-store", "performance", "dmv"],
        "must_contain": ["sys.query_store_runtime_stats", "avg_duration"],
        "must_not_contain": ["sys.dm_exec_query_stats"],
        "required_objects": [
            "sys.query_store_runtime_stats",
            "sys.query_store_plan",
            "sys.query_store_query",
            "sys.query_store_query_text",
        ],
        "rubric": "Query Store IS listed on this catalog, so use the query_store_* views (this is the positive twin of the QDS empty-policy traps). Substituting dm_exec_query_stats is a fail.",
    },
)

add(
    id="syn2-adv-018",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="admin",
    user_comment="-- queries with forced plans",
    gold="""SELECT
    q.query_id,
    qt.query_sql_text,
    p.plan_id,
    p.engine_version,
    p.last_execution_time
FROM sys.query_store_plan AS p
INNER JOIN sys.query_store_query AS q
    ON q.query_id = p.query_id
INNER JOIN sys.query_store_query_text AS qt
    ON qt.query_text_id = q.query_text_id
WHERE p.is_forced_plan = 1
ORDER BY
    p.last_execution_time DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["query-store", "plans"],
        "must_contain": ["sys.query_store_plan", "is_forced_plan = 1"],
        "required_objects": ["sys.query_store_plan", "sys.query_store_query", "sys.query_store_query_text"],
        "rubric": "Filter query_store_plan on is_forced_plan = 1 and join back to the query text.",
    },
)

add(
    id="syn2-adv-019",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="admin",
    user_comment="-- top wait categories from query store",
    gold="""SELECT
    ws.wait_category_desc,
    SUM(ws.total_query_wait_time_ms) AS total_wait_ms,
    AVG(ws.avg_query_wait_time_ms) AS avg_wait_ms
FROM sys.query_store_wait_stats AS ws
GROUP BY
    ws.wait_category_desc
ORDER BY
    total_wait_ms DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["query-store", "waits"],
        "must_contain": ["sys.query_store_wait_stats", "wait_category_desc"],
        "must_not_contain": ["sys.dm_os_wait_stats"],
        "required_objects": ["sys.query_store_wait_stats"],
        "rubric": "Aggregate query_store_wait_stats by wait_category_desc. dm_os_wait_stats is not listed on this Azure catalog.",
    },
)

add(
    id="syn2-adv-020",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="admin",
    user_comment="-- average cpu and data io percent over the last hour",
    gold="""SELECT
    AVG(avg_cpu_percent) AS avg_cpu_percent,
    AVG(avg_data_io_percent) AS avg_data_io_percent,
    MAX(max_worker_percent) AS max_worker_percent
FROM sys.dm_db_resource_stats
WHERE end_time >= DATEADD(HOUR, -1, GETDATE());""",
    eval={
        "difficulty": "medium",
        "tags": ["resource", "azure", "dmv"],
        "must_contain": ["sys.dm_db_resource_stats", "avg_cpu_percent"],
        "required_objects": ["sys.dm_db_resource_stats"],
        "rubric": "Azure SQL Database resource consumption comes from sys.dm_db_resource_stats (15-second granularity); filter end_time to the last hour.",
    },
)

add(
    id="syn2-adv-021",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="admin",
    user_comment="-- wait stats for this database, highest wait time first",
    gold="""SELECT
    wait_type,
    waiting_tasks_count,
    wait_time_ms,
    max_wait_time_ms,
    signal_wait_time_ms
FROM sys.dm_db_wait_stats
ORDER BY
    wait_time_ms DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["waits", "azure", "dmv"],
        "must_contain": ["sys.dm_db_wait_stats"],
        "must_not_contain": ["sys.dm_os_wait_stats"],
        "required_objects": ["sys.dm_db_wait_stats"],
        "rubric": "On Azure SQL Database the listed wait DMV is the database-scoped sys.dm_db_wait_stats, not dm_os_wait_stats.",
    },
)

add(
    id="syn2-adv-022",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="admin",
    user_comment="-- executions per query today according to query store",
    gold="""SELECT
    q.query_id,
    qt.query_sql_text,
    SUM(rs.count_executions) AS executions_today
FROM sys.query_store_runtime_stats AS rs
INNER JOIN sys.query_store_runtime_stats_interval AS i
    ON i.runtime_stats_interval_id = rs.runtime_stats_interval_id
INNER JOIN sys.query_store_plan AS p
    ON p.plan_id = rs.plan_id
INNER JOIN sys.query_store_query AS q
    ON q.query_id = p.query_id
INNER JOIN sys.query_store_query_text AS qt
    ON qt.query_text_id = q.query_text_id
WHERE i.start_time >= CAST(GETDATE() AS date)
GROUP BY
    q.query_id,
    qt.query_sql_text
ORDER BY
    executions_today DESC;""",
    eval={
        "difficulty": "hard",
        "tags": ["query-store", "date-windows", "joins"],
        "must_contain": ["count_executions", "sys.query_store_runtime_stats_interval"],
        "required_objects": [
            "sys.query_store_runtime_stats",
            "sys.query_store_runtime_stats_interval",
            "sys.query_store_plan",
            "sys.query_store_query",
        ],
        "rubric": "Today's executions require joining runtime stats to the interval table and filtering interval start_time, then summing count_executions per query.",
    },
)

add(
    id="syn2-adv-023",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="admin",
    user_comment="-- total size of the database files on disk",
    gold="",
    eval={
        "difficulty": "medium",
        "tags": ["empty-policy", "trap", "azure", "size"],
        "expect_empty": True,
        "must_not_contain": ["sys.master_files", "SELECT", "```"],
        "rubric": "sys.master_files is not listed on this Azure SQL Database catalog. Return exactly an empty string; do not substitute another DMV.",
    },
)

add(
    id="syn2-adv-024",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="admin",
    user_comment="-- server-wide wait statistics since instance startup",
    gold="",
    eval={
        "difficulty": "hard",
        "tags": ["empty-policy", "trap", "azure", "waits"],
        "expect_empty": True,
        "must_not_contain": ["sys.dm_os_wait_stats", "sys.dm_db_wait_stats", "SELECT", "```"],
        "rubric": "The request is server-wide, but only the database-scoped sys.dm_db_wait_stats is listed. dm_os_wait_stats is unavailable here, and the database-scoped view does not answer a server-wide question, so return exactly an empty string.",
    },
)

add(
    id="syn2-adv-025",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="developer",
    user_comment="-- customer email addresses together with their street addresses",
    gold="",
    eval={
        "difficulty": "medium",
        "tags": ["empty-policy", "names-only", "trap"],
        "expect_empty": True,
        "must_not_contain": ["SalesLT.Address", "AddressLine", "SELECT", "```"],
        "rubric": "SalesLT.Address and SalesLT.CustomerAddress are names-only inventory entries, so their columns are unknown and the join keys are unavailable. Return exactly an empty string.",
    },
)

add(
    id="syn2-adv-026",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="devops",
    user_comment="-- list sql agent jobs that run overnight",
    gold="",
    eval={
        "difficulty": "easy",
        "tags": ["empty-policy", "trap", "azure", "agent-job"],
        "expect_empty": True,
        "must_not_contain": ["sysjobs", "msdb", "SELECT", "```"],
        "rubric": "Azure SQL Database has no SQL Agent and no msdb tables are listed. Return exactly an empty string (twin of the answerable msdb items).",
    },
)

add(
    id="syn2-adv-027",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="admin",
    user_comment="-- enable xp_cmdshell and list the files in the backup folder",
    gold="",
    eval={
        "difficulty": "easy",
        "tags": ["empty-policy", "trap", "azure", "security"],
        "expect_empty": True,
        "must_not_contain": ["xp_cmdshell", "sp_configure", "SELECT", "```"],
        "rubric": "xp_cmdshell and sp_configure are not listed objects (and are unavailable on Azure SQL Database). Return exactly an empty string.",
    },
)

add(
    id="syn2-adv-028",
    completion_category="continuation",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="developer",
    user_comment=None,
    statement="SELECT c.CustomerID, c.LastName, soh.TotalDue\nFROM SalesLT.Customer AS c\nINNER JOIN SalesLT.SalesOrderHeader AS soh\n    ON ",
    line_prefix="    ON ",
    gold="soh.CustomerID = c.CustomerID",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "join-on", "fk"],
        "must_contain": ["CustomerID"],
        "must_not_contain": [";", "WHERE"],
        "rubric": "Complete the ON clause using the SalesOrderHeader.CustomerID -> Customer.CustomerID foreign key. One unit, no semicolon.",
    },
)

add(
    id="syn2-adv-029",
    completion_category="continuation",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="developer",
    user_comment=None,
    statement="SELECT ProductID, Name, ListPrice\nFROM SalesLT.Product\nORDER BY ListPrice DESC\nOFFSET ",
    line_prefix="OFFSET ",
    gold="10 ROWS FETCH NEXT 10 ROWS ONLY",
    eval={
        "difficulty": "medium",
        "tags": ["continuation", "paging", "ansi"],
        "must_contain": ["ROWS", "FETCH NEXT"],
        "must_not_contain": [";", "TOP"],
        "rubric": "Complete the OFFSET ... FETCH paging clause. One unit, no semicolon, no TOP.",
    },
)

add(
    id="syn2-adv-030",
    completion_category="continuation",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="developer",
    user_comment=None,
    statement="WITH revenue AS (\n    SELECT CustomerID, SUM(TotalDue) AS total_due\n    FROM SalesLT.SalesOrderHeader\n    GROUP BY ",
    line_prefix="    GROUP BY ",
    gold="CustomerID",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "group-by", "cte"],
        "must_contain": ["CustomerID"],
        "must_not_contain": [";", "SUM", "TotalDue"],
        "rubric": "The only non-aggregated select item is CustomerID, so GROUP BY CustomerID. One unit only.",
    },
)

add(
    id="syn2-adv-031",
    completion_category="continuation",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="analytics",
    user_comment=None,
    statement="SELECT\n    sod.ProductID,\n    SUM(sod.LineTotal) AS revenue,\n    RANK() OVER (",
    line_prefix="    RANK() OVER (",
    gold="ORDER BY SUM(sod.LineTotal) DESC)",
    eval={
        "difficulty": "hard",
        "tags": ["continuation", "window"],
        "must_contain": ["ORDER BY", ")"],
        "must_not_contain": [";", "FROM"],
        "rubric": "Complete the OVER clause for ranking by the summed LineTotal. Close the parenthesis; do not continue into FROM.",
    },
)

add(
    id="syn2-adv-032",
    completion_category="continuation",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="admin",
    user_comment=None,
    statement="SELECT query_id, query_text_id\nFROM sys.query_store_",
    line_prefix="FROM sys.query_store_",
    gold="query",
    eval={
        "difficulty": "hard",
        "tags": ["continuation", "identifier", "prefix-aware", "query-store"],
        "must_contain": ["query"],
        "must_not_contain": ["runtime_stats", "plan", "wait_stats", "query_text", ";", "sys."],
        "rubric": "Several listed views begin with query_store_, but only sys.query_store_query has both selected columns. Complete the identifier with exactly 'query'; repeating the sys.query_store_ prefix is a fail.",
    },
)


# ===========================================================================
# fabric_warehouse — ContosoDW star schema + queryinsights (Fabric)
# ===========================================================================

add(
    id="syn2-fab-001",
    completion_category="intent",
    schema_id="fabric_warehouse",
    inferred_system_query=False,
    scenario="analytics",
    user_comment="-- total sales amount by product category",
    gold="""SELECT
    p.Category,
    SUM(f.TotalAmount) AS total_sales
FROM dbo.FactSales AS f
INNER JOIN dbo.DimProduct AS p
    ON p.ProductKey = f.ProductKey
GROUP BY
    p.Category
ORDER BY
    total_sales DESC;""",
    eval={
        "difficulty": "easy",
        "tags": ["star-schema", "aggregate", "analytics", "fabric"],
        "must_contain": ["dbo.FactSales", "dbo.DimProduct", "SUM"],
        "required_objects": ["dbo.FactSales", "dbo.DimProduct"],
        "rubric": "Fact-to-dimension join on ProductKey, summing TotalAmount by Category.",
    },
)

add(
    id="syn2-fab-002",
    completion_category="intent",
    schema_id="fabric_warehouse",
    inferred_system_query=False,
    scenario="developer",
    user_comment="-- list stores in canada",
    gold="""SELECT
    StoreKey,
    StoreName,
    City,
    StateProvince
FROM dbo.DimStore
WHERE CountryRegion = 'Canada'
ORDER BY
    StoreName;""",
    eval={
        "difficulty": "easy",
        "tags": ["user-schema", "filter", "fabric"],
        "must_contain": ["dbo.DimStore", "CountryRegion"],
        "required_objects": ["dbo.DimStore"],
        "rubric": "Filter DimStore on CountryRegion = 'Canada'. Columns are varchar, so the N prefix is unnecessary (but not wrong).",
    },
)

add(
    id="syn2-fab-003",
    completion_category="intent",
    schema_id="fabric_warehouse",
    inferred_system_query=False,
    scenario="analytics",
    user_comment="-- monthly sales totals for fiscal year 2025",
    gold="""SELECT
    d.CalendarMonth,
    d.MonthName,
    SUM(f.TotalAmount) AS total_sales
FROM dbo.FactSales AS f
INNER JOIN dbo.DimDate AS d
    ON d.DateKey = f.DateKey
WHERE d.FiscalYear = 2025
GROUP BY
    d.CalendarMonth,
    d.MonthName
ORDER BY
    d.CalendarMonth;""",
    eval={
        "difficulty": "medium",
        "tags": ["star-schema", "aggregate", "date-dim", "analytics"],
        "must_contain": ["dbo.DimDate", "FiscalYear = 2025"],
        "required_objects": ["dbo.FactSales", "dbo.DimDate"],
        "rubric": "Use the date dimension's FiscalYear column, not YEAR() on a fact column (the fact table has no date column, only DateKey).",
    },
)

add(
    id="syn2-fab-004",
    completion_category="intent",
    schema_id="fabric_warehouse",
    inferred_system_query=False,
    scenario="analytics",
    user_comment="-- top 10 customers by revenue",
    gold="""SELECT TOP (10)
    c.CustomerKey,
    c.FirstName,
    c.LastName,
    SUM(f.TotalAmount) AS revenue
FROM dbo.FactSales AS f
INNER JOIN dbo.DimCustomer AS c
    ON c.CustomerKey = f.CustomerKey
GROUP BY
    c.CustomerKey,
    c.FirstName,
    c.LastName
ORDER BY
    revenue DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["star-schema", "aggregate", "top-n", "analytics"],
        "must_contain": ["dbo.DimCustomer", "TOP", "SUM"],
        "required_objects": ["dbo.FactSales", "dbo.DimCustomer"],
        "rubric": "Join fact to customer dimension on CustomerKey; TOP 10 by summed TotalAmount.",
    },
)

add(
    id="syn2-fab-005",
    completion_category="intent",
    schema_id="fabric_warehouse",
    inferred_system_query=False,
    scenario="analytics",
    user_comment="-- year over year sales growth by calendar year",
    gold="""WITH yearly AS (
    SELECT
        d.CalendarYear,
        SUM(f.TotalAmount) AS total_sales
    FROM dbo.FactSales AS f
    INNER JOIN dbo.DimDate AS d
        ON d.DateKey = f.DateKey
    GROUP BY
        d.CalendarYear
)
SELECT
    CalendarYear,
    total_sales,
    total_sales - LAG(total_sales) OVER (ORDER BY CalendarYear) AS yoy_change
FROM yearly
ORDER BY
    CalendarYear;""",
    eval={
        "difficulty": "hard",
        "tags": ["window", "yoy", "cte", "star-schema", "analytics"],
        "must_contain": ["LAG", "OVER", "CalendarYear"],
        "required_objects": ["dbo.FactSales", "dbo.DimDate"],
        "rubric": "Yearly totals via the date dimension, then LAG for the prior-year delta.",
    },
)

add(
    id="syn2-fab-006",
    completion_category="intent",
    schema_id="fabric_warehouse",
    inferred_system_query=False,
    scenario="analytics",
    user_comment="-- top 3 products per category by sales amount",
    gold="""WITH product_sales AS (
    SELECT
        p.Category,
        p.ProductName,
        SUM(f.TotalAmount) AS total_sales,
        ROW_NUMBER() OVER (
            PARTITION BY p.Category
            ORDER BY SUM(f.TotalAmount) DESC
        ) AS rn
    FROM dbo.FactSales AS f
    INNER JOIN dbo.DimProduct AS p
        ON p.ProductKey = f.ProductKey
    GROUP BY
        p.Category,
        p.ProductName
)
SELECT
    Category,
    ProductName,
    total_sales
FROM product_sales
WHERE rn <= 3
ORDER BY
    Category,
    total_sales DESC;""",
    eval={
        "difficulty": "hard",
        "tags": ["window", "top-n-per-group", "star-schema", "analytics"],
        "must_contain": ["PARTITION BY", "ROW_NUMBER", "<= 3"],
        "required_objects": ["dbo.FactSales", "dbo.DimProduct"],
        "rubric": "Top-N per group needs a window function partitioned by Category; TOP alone is not enough.",
    },
)

add(
    id="syn2-fab-007",
    completion_category="intent",
    schema_id="fabric_warehouse",
    inferred_system_query=False,
    scenario="analytics",
    user_comment="-- weekend versus weekday sales totals",
    gold="""SELECT
    d.IsWeekend,
    SUM(f.TotalAmount) AS total_sales,
    COUNT(*) AS sales_rows
FROM dbo.FactSales AS f
INNER JOIN dbo.DimDate AS d
    ON d.DateKey = f.DateKey
GROUP BY
    d.IsWeekend
ORDER BY
    d.IsWeekend;""",
    eval={
        "difficulty": "medium",
        "tags": ["star-schema", "aggregate", "date-dim", "analytics"],
        "must_contain": ["IsWeekend"],
        "required_objects": ["dbo.FactSales", "dbo.DimDate"],
        "rubric": "Group by the date dimension's IsWeekend flag instead of computing DATENAME on the fly.",
    },
)

add(
    id="syn2-fab-008",
    completion_category="intent",
    schema_id="fabric_warehouse",
    inferred_system_query=False,
    scenario="analytics",
    user_comment="-- gross margin per product",
    gold="""SELECT
    p.ProductKey,
    p.ProductName,
    SUM(f.TotalAmount) AS revenue,
    SUM(f.CostAmount) AS cost,
    SUM(f.TotalAmount) - SUM(f.CostAmount) AS gross_margin
FROM dbo.FactSales AS f
INNER JOIN dbo.DimProduct AS p
    ON p.ProductKey = f.ProductKey
GROUP BY
    p.ProductKey,
    p.ProductName
ORDER BY
    gross_margin DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["star-schema", "aggregate", "analytics"],
        "must_contain": ["CostAmount", "TotalAmount"],
        "required_objects": ["dbo.FactSales", "dbo.DimProduct"],
        "rubric": "Margin is summed TotalAmount minus summed CostAmount per product.",
    },
)

add(
    id="syn2-fab-009",
    completion_category="intent",
    schema_id="fabric_warehouse",
    inferred_system_query=True,
    scenario="metadata",
    user_comment="-- how many rows are in each user table",
    gold="""SELECT
    s.name AS schema_name,
    t.name AS table_name,
    SUM(p.rows) AS row_count
FROM sys.tables AS t
INNER JOIN sys.schemas AS s
    ON s.schema_id = t.schema_id
INNER JOIN sys.partitions AS p
    ON p.object_id = t.object_id
WHERE t.is_ms_shipped = 0
GROUP BY
    s.name,
    t.name
ORDER BY
    row_count DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["catalog", "sys.tables", "metadata"],
        "must_contain": ["sys.partitions", "SUM(p.rows)"],
        "required_objects": ["sys.tables", "sys.schemas", "sys.partitions"],
        "rubric": "Row counts come from sys.partitions aggregated per table; sys.allocation_units is not listed on this Fabric catalog.",
    },
)

add(
    id="syn2-fab-010",
    completion_category="intent",
    schema_id="fabric_warehouse",
    inferred_system_query=False,
    scenario="analytics",
    user_comment="-- average order value per store",
    gold="""SELECT
    s.StoreName,
    SUM(f.TotalAmount) / COUNT(DISTINCT f.OrderNumber) AS avg_order_value
FROM dbo.FactSales AS f
INNER JOIN dbo.DimStore AS s
    ON s.StoreKey = f.StoreKey
GROUP BY
    s.StoreName
ORDER BY
    avg_order_value DESC;""",
    eval={
        "difficulty": "hard",
        "tags": ["star-schema", "aggregate", "analytics"],
        "must_contain": ["COUNT(DISTINCT", "OrderNumber"],
        "required_objects": ["dbo.FactSales", "dbo.DimStore"],
        "rubric": "Order value must divide by distinct OrderNumber, not by fact rows (an order spans multiple line rows).",
    },
)

add(
    id="syn2-fab-011",
    completion_category="intent",
    schema_id="fabric_warehouse",
    inferred_system_query=False,
    scenario="analytics",
    user_comment="-- customer segments ranked by total revenue",
    gold="""SELECT
    c.CustomerSegment,
    SUM(f.TotalAmount) AS total_revenue
FROM dbo.FactSales AS f
INNER JOIN dbo.DimCustomer AS c
    ON c.CustomerKey = f.CustomerKey
GROUP BY
    c.CustomerSegment
ORDER BY
    total_revenue DESC;""",
    eval={
        "difficulty": "easy",
        "tags": ["star-schema", "aggregate", "analytics"],
        "must_contain": ["CustomerSegment", "SUM"],
        "required_objects": ["dbo.FactSales", "dbo.DimCustomer"],
        "rubric": "Group summed TotalAmount by DimCustomer.CustomerSegment.",
    },
)

add(
    id="syn2-fab-012",
    completion_category="intent",
    schema_id="fabric_warehouse",
    inferred_system_query=False,
    scenario="analytics",
    user_comment="-- sales by country and state with subtotals using rollup",
    gold="""SELECT
    s.CountryRegion,
    s.StateProvince,
    SUM(f.TotalAmount) AS total_sales
FROM dbo.FactSales AS f
INNER JOIN dbo.DimStore AS s
    ON s.StoreKey = f.StoreKey
GROUP BY ROLLUP (s.CountryRegion, s.StateProvince)
ORDER BY
    s.CountryRegion,
    s.StateProvince;""",
    eval={
        "difficulty": "hard",
        "tags": ["rollup", "aggregate", "ansi", "analytics"],
        "must_contain": ["ROLLUP", "CountryRegion", "StateProvince"],
        "required_objects": ["dbo.FactSales", "dbo.DimStore"],
        "rubric": "GROUP BY ROLLUP over country then state produces the requested subtotal rows.",
    },
)

add(
    id="syn2-fab-013",
    completion_category="intent",
    schema_id="fabric_warehouse",
    inferred_system_query=True,
    scenario="admin",
    user_comment="-- longest running queries in this warehouse",
    gold="""SELECT
    query_hash,
    command,
    number_of_runs,
    median_total_elapsed_time_ms,
    last_run_total_elapsed_time_ms,
    last_run_start_time
FROM queryinsights.long_running_queries
ORDER BY
    median_total_elapsed_time_ms DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["queryinsights", "performance", "fabric"],
        "must_contain": ["queryinsights.long_running_queries"],
        "must_not_contain": ["sys.dm_exec_query_stats", "query_store"],
        "required_objects": ["queryinsights.long_running_queries"],
        "rubric": "Fabric warehouses expose query history through queryinsights views, not dm_exec_query_stats or Query Store.",
    },
)

add(
    id="syn2-fab-014",
    completion_category="intent",
    schema_id="fabric_warehouse",
    inferred_system_query=True,
    scenario="admin",
    user_comment="-- queries that failed in the last day",
    gold="""SELECT
    distributed_statement_id,
    start_time,
    end_time,
    total_elapsed_time_ms,
    login_name,
    status,
    command
FROM queryinsights.exec_requests_history
WHERE status = 'Failed'
  AND start_time >= DATEADD(DAY, -1, GETDATE())
ORDER BY
    start_time DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["queryinsights", "fabric", "history"],
        "must_contain": ["queryinsights.exec_requests_history", "Failed"],
        "required_objects": ["queryinsights.exec_requests_history"],
        "rubric": "Filter exec_requests_history on a failed status within the last day.",
    },
)

add(
    id="syn2-fab-015",
    completion_category="intent",
    schema_id="fabric_warehouse",
    inferred_system_query=True,
    scenario="admin",
    user_comment="-- most frequently run queries and their average duration",
    gold="""SELECT
    query_hash,
    command,
    number_of_runs,
    avg_total_elapsed_time_ms,
    number_of_failed_runs,
    last_run_start_time
FROM queryinsights.frequently_run_queries
ORDER BY
    number_of_runs DESC;""",
    eval={
        "difficulty": "easy",
        "tags": ["queryinsights", "fabric", "performance"],
        "must_contain": ["queryinsights.frequently_run_queries", "number_of_runs"],
        "required_objects": ["queryinsights.frequently_run_queries"],
        "rubric": "Straight listing of frequently_run_queries ordered by run count.",
    },
)

add(
    id="syn2-fab-016",
    completion_category="intent",
    schema_id="fabric_warehouse",
    inferred_system_query=True,
    scenario="admin",
    user_comment="-- who has run queries today and how many",
    gold="""SELECT
    login_name,
    COUNT(*) AS query_count
FROM queryinsights.exec_requests_history
WHERE start_time >= CAST(GETDATE() AS date)
GROUP BY
    login_name
ORDER BY
    query_count DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["queryinsights", "fabric", "sessions"],
        "must_contain": ["queryinsights.exec_requests_history", "login_name"],
        "required_objects": ["queryinsights.exec_requests_history"],
        "rubric": "Count exec_requests_history rows per login since midnight.",
    },
)

add(
    id="syn2-fab-017",
    completion_category="intent",
    schema_id="fabric_warehouse",
    inferred_system_query=True,
    scenario="admin",
    user_comment="-- which statements hit the result cache today",
    gold="""SELECT
    distributed_statement_id,
    start_time,
    total_elapsed_time_ms,
    command,
    login_name
FROM queryinsights.exec_requests_history
WHERE result_cache_hit = 1
  AND start_time >= CAST(GETDATE() AS date)
ORDER BY
    start_time DESC;""",
    eval={
        "difficulty": "medium",
        "tags": ["queryinsights", "fabric", "caching"],
        "must_contain": ["result_cache_hit"],
        "required_objects": ["queryinsights.exec_requests_history"],
        "rubric": "The in-memory/result cache flag is exec_requests_history.result_cache_hit.",
    },
)

add(
    id="syn2-fab-018",
    completion_category="intent",
    schema_id="fabric_warehouse",
    inferred_system_query=True,
    scenario="admin",
    user_comment="-- currently running requests",
    gold="""SELECT
    session_id,
    request_id,
    status,
    command,
    wait_type,
    wait_time,
    cpu_time,
    total_elapsed_time
FROM sys.dm_exec_requests
WHERE status = 'running'
ORDER BY
    total_elapsed_time DESC;""",
    eval={
        "difficulty": "easy",
        "tags": ["dmv", "requests", "fabric"],
        "must_contain": ["sys.dm_exec_requests"],
        "required_objects": ["sys.dm_exec_requests"],
        "rubric": "Live requests still come from sys.dm_exec_requests (listed on this Fabric catalog); queryinsights views are history.",
    },
)

add(
    id="syn2-fab-019",
    completion_category="intent",
    schema_id="fabric_warehouse",
    inferred_system_query=False,
    scenario="admin",
    user_comment="-- index fragmentation for the fact tables",
    gold="",
    eval={
        "difficulty": "medium",
        "tags": ["empty-policy", "trap", "fabric", "fragmentation"],
        "expect_empty": True,
        "must_not_contain": ["dm_db_index_physical_stats", "SELECT", "```"],
        "rubric": "sys.dm_db_index_physical_stats is not listed on this Fabric catalog (Fabric warehouses do not expose it). Return exactly an empty string.",
    },
)

add(
    id="syn2-fab-020",
    completion_category="intent",
    schema_id="fabric_warehouse",
    inferred_system_query=False,
    scenario="admin",
    user_comment="-- top queries from the query store",
    gold="",
    eval={
        "difficulty": "medium",
        "tags": ["empty-policy", "trap", "fabric", "query-store"],
        "expect_empty": True,
        "must_not_contain": ["query_store", "dm_exec_query_stats", "SELECT", "```"],
        "rubric": "Query Store views are not listed here. queryinsights is a different, non-equivalent surface and the request names Query Store specifically; return exactly an empty string.",
    },
)

add(
    id="syn2-fab-021",
    completion_category="intent",
    schema_id="fabric_warehouse",
    inferred_system_query=False,
    scenario="devops",
    user_comment="-- when was the last full backup of this warehouse",
    gold="",
    eval={
        "difficulty": "easy",
        "tags": ["empty-policy", "trap", "fabric", "backup"],
        "expect_empty": True,
        "must_not_contain": ["backupset", "msdb", "SELECT", "```"],
        "rubric": "No backup catalog exists in this snapshot (Fabric handles durability itself). Return exactly an empty string.",
    },
)

add(
    id="syn2-fab-022",
    completion_category="intent",
    schema_id="fabric_warehouse",
    inferred_system_query=False,
    scenario="analytics",
    user_comment="-- inventory levels by product for last month",
    gold="",
    eval={
        "difficulty": "hard",
        "tags": ["empty-policy", "names-only", "trap", "fabric"],
        "expect_empty": True,
        "must_not_contain": ["SELECT", "```"],
        "rubric": "dbo.FactInventory appears only as a names-only inventory entry, so its columns (and any join keys) are unknown. A grouped, filtered report cannot be grounded; return exactly an empty string.",
    },
)

add(
    id="syn2-fab-023",
    completion_category="intent",
    schema_id="fabric_warehouse",
    inferred_system_query=False,
    scenario="admin",
    user_comment="-- os wait statistics for this server",
    gold="",
    eval={
        "difficulty": "medium",
        "tags": ["empty-policy", "trap", "fabric", "waits"],
        "expect_empty": True,
        "must_not_contain": ["dm_os_wait_stats", "dm_db_wait_stats", "SELECT", "```"],
        "rubric": "No wait-stats DMV is listed on this Fabric catalog. Return exactly an empty string.",
    },
)

add(
    id="syn2-fab-024",
    completion_category="continuation",
    schema_id="fabric_warehouse",
    inferred_system_query=False,
    scenario="analytics",
    user_comment=None,
    statement="SELECT d.CalendarYear, SUM(f.TotalAmount) AS total_sales\nFROM dbo.FactSales AS f\nINNER JOIN dbo.DimDate AS d\n    ON ",
    line_prefix="    ON ",
    gold="d.DateKey = f.DateKey",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "join-on", "fk", "star-schema"],
        "must_contain": ["DateKey"],
        "must_not_contain": [";", "WHERE", "GROUP BY"],
        "rubric": "Complete the ON clause with the FactSales.DateKey -> DimDate.DateKey key. One unit, no semicolon.",
    },
)

add(
    id="syn2-fab-025",
    completion_category="continuation",
    schema_id="fabric_warehouse",
    inferred_system_query=True,
    scenario="admin",
    user_comment=None,
    statement="SELECT TOP 100 *\nFROM queryinsights.frequ",
    line_prefix="FROM queryinsights.frequ",
    gold="ently_run_queries",
    eval={
        "difficulty": "hard",
        "tags": ["continuation", "identifier", "prefix-aware", "queryinsights"],
        "must_contain": ["ently_run_queries"],
        "must_not_contain": ["frequently_run_queries", ";"],
        "rubric": "Only queryinsights.frequently_run_queries continues the typed prefix. Return the remainder of the identifier only; repeating 'frequently' double-writes the prefix.",
    },
)

add(
    id="syn2-fab-026",
    completion_category="continuation",
    schema_id="fabric_warehouse",
    inferred_system_query=False,
    scenario="developer",
    user_comment=None,
    statement="SELECT ProductKey, ProductName, Category\nFROM dbo.DimProduct\nWHERE ",
    line_prefix="WHERE ",
    gold="Category = 'Bikes'",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "where"],
        "must_contain_any": ["Category", "Brand", "Color", "ListPrice", "Subcategory"],
        "must_not_contain": [";", "SELECT"],
        "rubric": "One WHERE predicate on listed DimProduct columns. No semicolon.",
    },
)

add(
    id="syn2-fab-027",
    completion_category="continuation",
    schema_id="fabric_warehouse",
    inferred_system_query=False,
    scenario="analytics",
    user_comment=None,
    statement="SELECT s.StoreName, SUM(f.TotalAmount) AS total_sales\nFROM dbo.FactSales AS f\nINNER JOIN dbo.DimStore AS s\n    ON s.StoreKey = f.StoreKey\nGROUP BY ",
    line_prefix="GROUP BY ",
    gold="s.StoreName",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "group-by", "star-schema"],
        "must_contain": ["s.StoreName"],
        "must_not_contain": [";", "SUM", "ORDER BY"],
        "rubric": "The only non-aggregated select item is s.StoreName. One unit only.",
    },
)

add(
    id="syn2-fab-028",
    completion_category="continuation",
    schema_id="fabric_warehouse",
    inferred_system_query=False,
    scenario="developer",
    user_comment=None,
    statement="SELECT f.OrderNumber, f.TotalAmount\nFROM dbo.FactSales AS f\nWHERE f.TotalAmount > 100\nORDER BY ",
    line_prefix="ORDER BY ",
    line_suffix="f.TotalAmount DESC",
    document_suffix="f.TotalAmount DESC",
    gold="",
    eval={
        "difficulty": "hard",
        "tags": ["continuation", "suffix-aware", "empty-policy"],
        "expect_empty": True,
        "must_not_contain": ["TotalAmount", "SELECT", "```"],
        "rubric": "The sort expression already sits to the right of the cursor. Anything inserted would duplicate or corrupt it; return exactly an empty string.",
    },
)


# ===========================================================================
# ANSI-idiom items on the original catalogs
# ===========================================================================

add(
    id="syn2-ansi-001",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    scenario="developer",
    user_comment="-- 10 most expensive products using ansi standard paging",
    gold="""SELECT
    ProductID,
    ProductName,
    UnitPrice
FROM dbo.Products
ORDER BY
    UnitPrice DESC
OFFSET 0 ROWS FETCH FIRST 10 ROWS ONLY;""",
    eval={
        "difficulty": "medium",
        "tags": ["ansi", "paging", "user-schema"],
        "must_contain": ["OFFSET", "ROWS"],
        "must_contain_any": ["FETCH FIRST", "FETCH NEXT"],
        "must_not_contain": ["TOP"],
        "required_objects": ["dbo.Products"],
        "rubric": "The comment asks for ANSI paging, so OFFSET ... FETCH, not TOP.",
    },
)

add(
    id="syn2-ansi-002",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    scenario="developer",
    user_comment="-- workout categories that are not also meal categories",
    gold="""SELECT category
FROM dbo.Workouts
EXCEPT
SELECT category
FROM dbo.Meals;""",
    eval={
        "difficulty": "medium",
        "tags": ["ansi", "set-ops", "user-schema"],
        "must_contain": ["EXCEPT", "dbo.Workouts", "dbo.Meals"],
        "required_objects": ["dbo.Workouts", "dbo.Meals"],
        "rubric": "EXCEPT is the canonical set-difference here; NOT IN / NOT EXISTS variants also satisfy the intent but EXCEPT is preferred.",
    },
)

add(
    id="syn2-ansi-003",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    scenario="developer",
    user_comment="-- all distinct cities from customers and authors combined",
    gold="""SELECT City AS city
FROM dbo.Customers
UNION
SELECT city
FROM dbo.authors;""",
    eval={
        "difficulty": "medium",
        "tags": ["ansi", "set-ops", "user-schema"],
        "must_contain": ["UNION", "dbo.Customers", "dbo.authors"],
        "must_not_contain": ["UNION ALL"],
        "required_objects": ["dbo.Customers", "dbo.authors"],
        "rubric": "UNION (not UNION ALL) deduplicates the combined city list; the two tables spell the column City/city differently but both are listed.",
    },
)

add(
    id="syn2-ansi-004",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    scenario="developer",
    user_comment="-- current date and time using the ansi standard function",
    gold="SELECT CURRENT_TIMESTAMP;",
    eval={
        "difficulty": "easy",
        "tags": ["ansi", "tsql-idiom"],
        "must_contain": ["CURRENT_TIMESTAMP"],
        "must_not_contain": ["GETDATE", "SYSDATETIME"],
        "rubric": "The comment asks for the ANSI form: CURRENT_TIMESTAMP, not GETDATE().",
    },
)

add(
    id="syn2-ansi-005",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    scenario="developer",
    user_comment="-- users with a bmi category using a case expression",
    gold="""SELECT
    user_id,
    full_name,
    height_cm,
    weight_kg,
    weight_kg / POWER(height_cm / 100.0, 2) AS bmi,
    CASE
        WHEN weight_kg / POWER(height_cm / 100.0, 2) < 18.5 THEN N'underweight'
        WHEN weight_kg / POWER(height_cm / 100.0, 2) < 25 THEN N'normal'
        WHEN weight_kg / POWER(height_cm / 100.0, 2) < 30 THEN N'overweight'
        ELSE N'obese'
    END AS bmi_category
FROM dbo.Users
ORDER BY
    bmi DESC;""",
    eval={
        "difficulty": "hard",
        "tags": ["ansi", "case", "user-schema", "computed"],
        "must_contain": ["CASE", "WHEN", "height_cm", "weight_kg"],
        "required_objects": ["dbo.Users"],
        "rubric": "BMI from listed height_cm/weight_kg (converting cm to meters) bucketed with CASE.",
    },
)

add(
    id="syn2-ansi-006",
    completion_category="intent",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    scenario="developer",
    user_comment="-- customers with their region, showing N/A when region is missing",
    gold="""SELECT
    CustomerID,
    CompanyName,
    COALESCE(Region, N'N/A') AS Region
FROM dbo.Customers
ORDER BY
    CompanyName;""",
    eval={
        "difficulty": "easy",
        "tags": ["ansi", "null-handling", "user-schema"],
        "must_contain": ["COALESCE", "Region"],
        "required_objects": ["dbo.Customers"],
        "rubric": "COALESCE (or ISNULL, though the ANSI form is preferred) over the nullable Region column.",
    },
)

add(
    id="syn2-ansi-007",
    completion_category="continuation",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    scenario="analytics",
    user_comment=None,
    statement="SELECT\n    session_id,\n    session_date,\n    calories_burned,\n    AVG(calories_burned) OVER (",
    line_prefix="    AVG(calories_burned) OVER (",
    gold="ORDER BY session_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)",
    eval={
        "difficulty": "hard",
        "tags": ["continuation", "window", "ansi"],
        "must_contain": ["ORDER BY", "PRECEDING"],
        "must_not_contain": [";", "FROM"],
        "rubric": "Complete the OVER clause as a moving average frame ordered by session_date, closing the parenthesis. One unit; do not continue into FROM.",
    },
)

add(
    id="syn2-ansi-008",
    completion_category="continuation",
    schema_id="ninjadb_b",
    inferred_system_query=False,
    scenario="developer",
    user_comment=None,
    statement="SELECT s.stor_name, sa.ord_num, sa.qty\nFROM dbo.stores AS s\nINNER JOIN dbo.sales AS sa\n    ON ",
    line_prefix="    ON ",
    gold="sa.stor_id = s.stor_id",
    eval={
        "difficulty": "easy",
        "tags": ["continuation", "join-on", "fk", "user-schema"],
        "must_contain": ["stor_id"],
        "must_not_contain": [";", "WHERE"],
        "rubric": "Complete the ON clause via the sales.stor_id -> stores.stor_id foreign key. One unit, no semicolon.",
    },
)


# ===========================================================================
# Formatting / robustness items (typed keywords, broken prefixes, DML safety)
# ===========================================================================

add(
    id="syn2-fmt-001",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    scenario="developer",
    user_comment="-- names and emails of all users",
    statement="-- names and emails of all users\nSELECT ",
    line_prefix="SELECT ",
    gold="""full_name,
    email
FROM dbo.Users
ORDER BY
    full_name;""",
    eval={
        "difficulty": "medium",
        "tags": ["formatting", "typed-keyword", "user-schema"],
        "must_contain": ["full_name", "email", "dbo.Users"],
        "must_not_contain": ["SELECT"],
        "required_objects": ["dbo.Users"],
        "rubric": "The user already typed SELECT; the completion must continue from that point without repeating the keyword.",
    },
)

add(
    id="syn2-fmt-002",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="analytics",
    user_comment="-- yearly revenue as a cte, then pick the best year",
    statement="-- yearly revenue as a cte, then pick the best year\nWITH ",
    line_prefix="WITH ",
    gold="""yearly AS (
    SELECT
        YEAR(OrderDate) AS order_year,
        SUM(TotalDue) AS revenue
    FROM SalesLT.SalesOrderHeader
    GROUP BY
        YEAR(OrderDate)
)
SELECT TOP (1)
    order_year,
    revenue
FROM yearly
ORDER BY
    revenue DESC;""",
    eval={
        "difficulty": "hard",
        "tags": ["formatting", "typed-keyword", "cte", "analytics"],
        "must_contain": ["AS (", "SalesLT.SalesOrderHeader"],
        "must_not_contain": ["WITH"],
        "required_objects": ["SalesLT.SalesOrderHeader"],
        "rubric": "The user already typed WITH; start with the CTE name, never repeat the keyword.",
    },
)

add(
    id="syn2-fmt-003",
    completion_category="intent",
    schema_id="fabric_warehouse",
    inferred_system_query=False,
    scenario="analytics",
    user_comment="-- count rows in the sales fact",
    recent="SELECT FROM dbo.FactSales\nWHERE TotalAmount >\n\n-- count rows in the sales fact",
    gold="""SELECT COUNT(*) AS row_count
FROM dbo.FactSales;""",
    eval={
        "difficulty": "medium",
        "tags": ["formatting", "trap", "broken-prefix"],
        "must_contain": ["COUNT(*)", "dbo.FactSales"],
        "must_not_contain": ["TotalAmount >"],
        "required_objects": ["dbo.FactSales"],
        "rubric": "The recent prefix contains an abandoned broken statement. The new comment starts a fresh query; do not try to repair or continue the broken one.",
    },
)

add(
    id="syn2-fmt-004",
    completion_category="intent",
    schema_id="msdb_onprem",
    inferred_system_query=False,
    scenario="devops",
    user_comment="-- add an operator named oncall dba with email oncall@contoso.com",
    gold="""INSERT INTO dbo.sysoperators (name, enabled, email_address)
VALUES (N'oncall dba', 1, N'oncall@contoso.com');""",
    eval={
        "difficulty": "medium",
        "tags": ["dml", "insert", "agent-job"],
        "must_contain": ["INSERT INTO dbo.sysoperators", "oncall@contoso.com"],
        "required_objects": ["dbo.sysoperators"],
        "rubric": "INSERT into dbo.sysoperators with listed columns; id is the PK and should not be supplied.",
    },
)

add(
    id="syn2-fmt-005",
    completion_category="intent",
    schema_id="adventure_azure",
    inferred_system_query=False,
    scenario="developer",
    user_comment="-- remove customers that have never placed an order",
    gold="""DELETE FROM SalesLT.Customer
WHERE NOT EXISTS (
    SELECT 1
    FROM SalesLT.SalesOrderHeader AS soh
    WHERE soh.CustomerID = SalesLT.Customer.CustomerID
);""",
    eval={
        "difficulty": "hard",
        "tags": ["dml", "delete", "anti-join"],
        "must_contain": ["DELETE", "NOT EXISTS", "SalesLT.SalesOrderHeader"],
        "required_objects": ["SalesLT.Customer", "SalesLT.SalesOrderHeader"],
        "rubric": "DELETE guarded by an anti-join against SalesOrderHeader. An unfiltered DELETE is a hard fail.",
    },
)

add(
    id="syn2-fmt-006",
    completion_category="intent",
    schema_id="fitnessapp_test",
    inferred_system_query=False,
    scenario="devops",
    user_comment="-- mark pending payments older than 30 days as failed, inside a transaction with error handling",
    gold="""BEGIN TRY
    BEGIN TRANSACTION;

    UPDATE dbo.Payments
    SET payment_status = N'failed'
    WHERE payment_status = N'pending'
      AND payment_date < DATEADD(DAY, -30, GETDATE());

    COMMIT TRANSACTION;
END TRY
BEGIN CATCH
    IF @@TRANCOUNT > 0
        ROLLBACK TRANSACTION;
    THROW;
END CATCH;""",
    eval={
        "difficulty": "hard",
        "tags": ["dml", "update", "try-catch", "tsql-idiom"],
        "must_contain": ["BEGIN TRY", "ROLLBACK", "UPDATE dbo.Payments", "DATEADD"],
        "required_objects": ["dbo.Payments"],
        "rubric": "TRY/CATCH around a transactional UPDATE of listed Payments columns, rolling back and rethrowing on error.",
    },
)

add(
    id="syn2-fmt-007",
    completion_category="continuation",
    schema_id="azure_master",
    inferred_system_query=False,
    scenario="developer",
    user_comment=None,
    statement="SELECT dbo.fn_sysdac_",
    line_prefix="SELECT dbo.fn_sysdac_",
    gold="get_currentusername()",
    eval={
        "difficulty": "medium",
        "tags": ["continuation", "identifier", "prefix-aware", "function"],
        "must_contain": ["get_currentusername()"],
        "must_not_contain": ["fn_sysdac_get_currentusername", ";"],
        "rubric": "The only listed routine on this catalog is dbo.fn_sysdac_get_currentusername(). Complete the remainder of the identifier plus the call parentheses; repeating the typed prefix is a fail.",
    },
)

add(
    id="syn2-fmt-008",
    completion_category="continuation",
    schema_id="ninjadb_a",
    inferred_system_query=False,
    scenario="developer",
    user_comment=None,
    statement="INSERT INTO dbo.Categories (CategoryName, Description, Purpose)\nVALUES (",
    line_prefix="VALUES (",
    gold="N'Beverages 2', N'New drink lineup', N'Seasonal menu')",
    eval={
        "difficulty": "medium",
        "tags": ["continuation", "dml", "insert"],
        "must_contain": [")"],
        "must_not_contain": [";", "CategoryID", "INSERT"],
        "rubric": "Supply one plausible VALUES tuple matching the three listed columns (nvarchar, ntext, nvarchar) and close the parenthesis. Do not add CategoryID (the PK), a semicolon, or a second row.",
    },
)
