import { z } from "zod";
import { hashJson, sha256 } from "./hash.js";
import { incidentClasses, severities } from "./contracts.js";

const runbookSchema = z.object({
  runbookId: z.string().regex(/^TSG-[A-Z]{3,5}-\d{2}$/),
  title: z.string().min(10).max(300),
  incidentClass: z.enum(incidentClasses),
  severityFloor: z.enum(severities),
  sourceUri: z.string().url(),
  sourceKind: z.literal("original_mit_official_reference"),
  bodyMarkdown: z.string().min(300).max(20_000),
  metadata: z.object({
    license: z.literal("MIT"),
    family: z.string().min(3).max(80),
    angle: z.string().min(3).max(80),
    keywords: z.array(z.string().min(1).max(80)).min(3).max(30),
    sourceCheckedAt: z.string().datetime(),
  }).strict(),
}).strict();

export const runbookCorpusSchema = z.object({
  schemaVersion: z.literal(1),
  corpusId: z.literal("primary-v1"),
  corpusVersion: z.literal("runbooks-v1"),
  corpusKind: z.literal("primary"),
  license: z.literal("MIT"),
  sourceCheckedAt: z.string().datetime(),
  runbooks: z.array(runbookSchema).length(60),
}).strict();

export type RunbookCorpus = z.infer<typeof runbookCorpusSchema>;
export type Runbook = RunbookCorpus["runbooks"][number];

interface FamilyDefinition {
  code: string;
  family: string;
  incidentClass: Runbook["incidentClass"];
  severityFloor: Runbook["severityFloor"];
  title: string;
  scope: string;
  sourceUri: string;
  keywords: string[];
  supports: string[];
  contradicts: string[];
  diagnostics: string[];
  actions: string[];
  escalations: string[];
  avoid: string[];
}

const sourceCheckedAt = "2026-08-23T00:00:00.000Z";

const families: FamilyDefinition[] = [
  {
    code: "DLK", family: "deadlock", incidentClass: "deadlock", severityFloor: "medium",
    title: "Deadlock", scope: "cyclic lock dependencies in which SQL Server selects a victim so other work can continue",
    sourceUri: "https://learn.microsoft.com/en-us/sql/relational-databases/sql-server-deadlocks-guide?view=sql-server-ver17",
    keywords: ["deadlock", "victim", "cycle", "lock", "1205", "xml_deadlock_report"],
    supports: ["an xml_deadlock_report with a closed resource cycle", "error 1205 on a participating request", "the same statements or resources recur in multiple graphs"],
    contradicts: ["a one-way blocking chain with no cycle", "a client timeout without a deadlock graph", "deadlock-like words only in application text"],
    diagnostics: ["summarize the graph and identify the victim, processes, resources, isolation levels, and statement hashes", "compare recurrence counts before treating one victim as a storm", "confirm retry behavior and transaction ordering from application telemetry"],
    actions: ["retry only the failed transaction with bounded backoff when the operation is idempotent", "make competing transactions access resources in a consistent order", "shorten transactions and add only evidence-supported indexing or query changes"],
    escalations: ["graphs recur after an application or query change", "victim selection affects critical or non-idempotent work"],
    avoid: ["killing every session named in a historical graph", "assuming all blocking is a deadlock"],
  },
  {
    code: "BLK", family: "blocking_waits", incidentClass: "blocking", severityFloor: "low",
    title: "Blocking and waits", scope: "lock waits where one head blocker delays one or more sessions without a resource cycle",
    sourceUri: "https://learn.microsoft.com/en-us/troubleshoot/sql/database-engine/performance/understand-resolve-blocking",
    keywords: ["blocking", "head blocker", "waiter", "LCK_M", "open transaction", "timeout"],
    supports: ["blocked-process evidence or a nonzero blocking_session_id", "waiters remain behind the same head blocker beyond the threshold", "an open or sleeping transaction retains locks"],
    contradicts: ["brief blocking below the governed threshold", "CPU-bound execution with no blocker", "a closed deadlock cycle with a selected victim"],
    diagnostics: ["capture the bounded blocking chain and locate the head blocker", "inspect open-transaction state and request age without exposing arbitrary SQL text", "distinguish an active long query from an idle session retaining a transaction"],
    actions: ["let a known short transaction finish when impact is bounded", "shorten the transaction or correct application fetch and commit behavior", "tune the evidenced head-blocking query after preserving diagnostics"],
    escalations: ["blocking is sustained, widespread, or repeatedly breaches the objective", "terminating a session would risk business consistency"],
    avoid: ["terminating a blocker from one snapshot alone", "treating normal short lock waits as incidents"],
  },
  {
    code: "LOG", family: "transaction_log_space", incidentClass: "transaction_log_full", severityFloor: "high",
    title: "Transaction log space", scope: "transaction log exhaustion, growth pressure, and truncation holdup diagnosis including error 9002",
    sourceUri: "https://learn.microsoft.com/en-us/sql/relational-databases/errors-events/mssqlserver-9002-database-engine-error?view=sql-server-ver17",
    keywords: ["transaction log", "9002", "log full", "log reuse wait", "active transaction", "log backup"],
    supports: ["error 9002 names a database and log reuse cause", "active log percentage is near the configured file limit", "log truncation is held by an active transaction, backup, replica, or recovery condition"],
    contradicts: ["data-file growth without log pressure", "healthy log usage with no 9002 or sustained growth", "a generic disk warning for another service"],
    diagnostics: ["read bounded log size and truncation-holdup state for the named database", "inspect active transactions when ACTIVE_TRANSACTION holds reuse", "check backup history when LOG_BACKUP is the reuse reason"],
    actions: ["restore truncation by resolving the evidenced holdup", "take a governed log backup when the recovery model and policy require one", "plan capacity or controlled file growth after immediate pressure is safe"],
    escalations: ["the log cannot grow and write availability is affected", "the holdup involves availability, replication, recovery, or an unknown long transaction"],
    avoid: ["shrinking the log as a routine fix", "switching recovery models or deleting log files without an approved recovery plan"],
  },
  {
    code: "AUT", family: "authentication_access", incidentClass: "authentication_failure", severityFloor: "medium",
    title: "Authentication and database access", scope: "rejected login and database-access attempts, including error 18456 state context",
    sourceUri: "https://learn.microsoft.com/en-us/sql/relational-databases/errors-events/mssqlserver-18456-database-engine-error?view=sql-server-ver17",
    keywords: ["login failed", "18456", "authentication", "state", "default database", "credential"],
    supports: ["error 18456 appears in Extended Events and the SQL error log", "the protected error-log state narrows the rejection stage", "many accounts or clients fail within a short interval"],
    contradicts: ["a successful login or completed recovery message", "a query permission denial after authentication", "a connection timeout before a server login response"],
    diagnostics: ["count recent failures by bounded class and time window", "compare the state and client pattern without returning secrets", "check whether the intended database is online and the login mapping is valid"],
    actions: ["correct the credential, authentication mode, default database, or user mapping indicated by evidence", "rotate or unlock credentials through the authorized identity workflow", "treat broad bursts as a security signal and preserve source context"],
    escalations: ["failures span many accounts or originate from an unexpected client pattern", "the state is unavailable, conflicting, or suggests a system-wide access problem"],
    avoid: ["returning passwords, hashes, or full connection strings", "granting broad roles merely to clear a login failure"],
  },
  {
    code: "INT", family: "integrity_signal", incidentClass: "integrity_signal", severityFloor: "high",
    title: "Integrity signals", scope: "credible consistency, allocation, checksum, or corruption-like evidence that requires conservative handling",
    sourceUri: "https://learn.microsoft.com/en-us/sql/t-sql/database-console-commands/dbcc-checkdb-transact-sql?view=sql-server-ver17",
    keywords: ["integrity", "consistency", "checksum", "allocation", "DBCC CHECKDB", "corruption"],
    supports: ["a logged engine integrity error or repeatable CHECKDB finding", "page, allocation, or checksum evidence identifies a concrete database", "independent storage or I/O evidence supports the signal"],
    contradicts: ["warning-like text in user data", "one lab-authored ambiguous message without engine evidence", "a clean read-only consistency check for the same database"],
    diagnostics: ["preserve the exact error and affected database before any change", "run only approved read-only consistency diagnostics or restore verification", "check backup viability and related I/O evidence"],
    actions: ["escalate and protect the evidence chain", "prefer restore from a known good verified backup over ad hoc repair", "isolate affected workload only through an approved incident procedure"],
    escalations: ["any confirmed engine consistency failure", "evidence is incomplete but data durability could be at risk"],
    avoid: ["running repair with data loss automatically", "modifying pages, files, or metadata to make an alert disappear"],
  },
  {
    code: "BAK", family: "backup_restore_recovery", incidentClass: "backup_restore_failure", severityFloor: "high",
    title: "Backup, restore, and recovery", scope: "failed, stale, slow, or invalid backup and restore operations",
    sourceUri: "https://learn.microsoft.com/en-us/troubleshoot/sql/database-engine/backup-restore/backup-restore-operations",
    keywords: ["backup", "restore", "3201", "media", "checksum", "recovery objective"],
    supports: ["a backup operation records failure or cannot open its destination", "backup history shows repeated failures or an objective breach", "restore verification or media checks report a concrete problem"],
    contradicts: ["a successful recent backup with matching verification", "an unrelated application file error", "a planned restore still making bounded progress"],
    diagnostics: ["inspect bounded backup-attempt history and recency", "separate destination permission, capacity, media, and throughput causes", "verify candidate media with approved header, checksum, or restore tests"],
    actions: ["correct the destination or service-account access and retry through the governed job", "create a new checksum backup when policy permits", "use another verified backup set when media is invalid"],
    escalations: ["the recovery objective is or will be missed", "no verified recoverable backup remains"],
    avoid: ["overwriting the only candidate backup", "claiming recoverability from job success without restore verification"],
  },
  {
    code: "QRY", family: "query_resource_pressure", incidentClass: "query_resource_pressure", severityFloor: "medium",
    title: "Query and resource pressure", scope: "slow, CPU-intensive, memory-grant, I/O, and client-timeout query symptoms",
    sourceUri: "https://learn.microsoft.com/en-us/troubleshoot/sql/database-engine/performance/troubleshoot-slow-running-queries",
    keywords: ["query pressure", "high CPU", "RESOURCE_SEMAPHORE", "memory grant", "attention", "timeout"],
    supports: ["bounded query telemetry shows high CPU, reads, duration, or waits", "an attention event follows work near the client timeout", "RESOURCE_SEMAPHORE or a related resource wait is sustained"],
    contradicts: ["the request is waiting only behind a known lock blocker", "one bounded analytical query completes within its objective", "host pressure originates outside SQL Server"],
    diagnostics: ["separate running time from resource wait time", "identify the stable query hash and compare recent executions", "check whether CPU, memory-grant, I/O, or client consumption is the dominant dimension"],
    actions: ["tune the evidenced query, statistics, indexing, or parameter behavior", "reduce unsafe concurrency or move noncritical analytical work", "align client timeout only after addressing server-side work"],
    escalations: ["pressure affects broad service availability", "the cause remains ambiguous across SQL, host, and application evidence"],
    avoid: ["adding indexes or hints without plan and workload evidence", "confusing a connection timeout with a query timeout"],
  },
  {
    code: "DAT", family: "schema_data_error", incidentClass: "schema_data_error", severityFloor: "medium",
    title: "Schema and data errors", scope: "conversion, constraint, missing-object, truncation, and related deterministic statement failures",
    sourceUri: "https://learn.microsoft.com/en-us/sql/relational-databases/errors-events/database-engine-events-and-errors?view=sql-server-ver17",
    keywords: ["conversion", "constraint", "missing object", "truncation", "245", "2627", "2628"],
    supports: ["a stable engine error identifies conversion, object, constraint, or length failure", "the same application path repeats the same normalized signature", "deployment or input-schema context aligns with the error"],
    contradicts: ["a deadlock victim or timeout is the only failure", "warning-like text appears only in a payload", "the object exists and a different permission error is recorded"],
    diagnostics: ["preserve the error number, state, object-neutral signature, and application boundary", "compare expected and actual types, lengths, constraints, or deployment versions", "measure recurrence before choosing incident severity"],
    actions: ["correct input validation, schema compatibility, or deployment ordering", "handle duplicate operations idempotently where business rules allow", "route unknown or conflicting signatures to human review"],
    escalations: ["failures follow a deployment across many requests", "the correct schema contract or data owner is unclear"],
    avoid: ["dropping constraints to accept bad data", "exposing literal customer values in work items"],
  },
  {
    code: "UNK", family: "unknown_ambiguous", incidentClass: "unknown_ambiguous", severityFloor: "low",
    title: "Unknown and ambiguous evidence", scope: "unsupported, incomplete, masked, truncated, or contradictory incident evidence",
    sourceUri: "https://learn.microsoft.com/en-us/sql/relational-databases/errors-events/understanding-database-engine-errors?view=sql-server-ver17",
    keywords: ["unknown", "ambiguous", "conflicting evidence", "truncated", "unsupported", "human review"],
    supports: ["required identifiers or signatures are missing", "event fields conflict with the surrounding context", "the event family is outside the frozen catalog"],
    contradicts: ["a complete supported signature and required diagnostics agree", "a known benign event matches its no-action policy", "only the numeric signature was masked but other evidence is decisive"],
    diagnostics: ["state exactly which evidence is missing or contradictory", "use only bounded diagnostics relevant to the plausible families", "preserve uncertainty and return a deterministic no-data result when snapshots are absent"],
    actions: ["abstain and escalate to a human with a concise evidence gap", "request a supported observation path rather than inventing detail", "classify only when the frozen contract has enough evidence"],
    escalations: ["uncertainty overlaps a high-impact integrity or availability risk", "the signal remains unsupported after allowed diagnostics"],
    avoid: ["guessing a specific root cause from generic text", "citing a runbook that was not retrieved"],
  },
  {
    code: "NOI", family: "benign_noise", incidentClass: "benign_noise", severityFloor: "info",
    title: "Benign activity and near misses", scope: "successful, duplicated, subthreshold, or text-only events that should not create an incident",
    sourceUri: "https://learn.microsoft.com/en-us/sql/relational-databases/extended-events/extended-events?view=sql-server-ver17",
    keywords: ["benign", "noise", "duplicate", "success", "subthreshold", "no action"],
    supports: ["the operation completed successfully", "blocking or resource use stayed below the frozen threshold", "the same source-position key is a replay duplicate"],
    contradicts: ["required failure evidence is independently verified", "a sustained threshold breach affects requests", "multiple sources corroborate a high-severity signal"],
    diagnostics: ["confirm source identity and deduplication before counting an alert", "compare duration and recurrence with frozen thresholds", "check that warning-like text is payload content rather than an engine event"],
    actions: ["take no action and retain the observation for audit", "suppress exact duplicates idempotently", "reclassify only if later evidence crosses the frozen threshold"],
    escalations: ["a supposedly benign pattern develops sustained impact", "source identity or success status cannot be verified"],
    avoid: ["opening work for every successful completion", "discarding raw evidence needed to prove suppression"],
  },
];

const angles = [
  { name: "first_response", title: "recognition and first response", focus: "Confirm the incident shape before proposing any change." },
  { name: "recurrence", title: "recurrence and rate context", focus: "Separate an isolated occurrence from a repeated or spreading condition." },
  { name: "discrimination", title: "diagnostic discriminators", focus: "Use supporting and contradicting evidence to distinguish confusable causes." },
  { name: "stabilization", title: "safe stabilization policy", focus: "Choose the least invasive evidence-backed action while preserving diagnostics." },
  { name: "escalation", title: "escalation evidence package", focus: "Collect a bounded, privacy-preserving record that makes human review efficient." },
  { name: "near_miss", title: "near misses and no-action boundary", focus: "Avoid false alarms when the signature, duration, or impact threshold is not met." },
] as const;

export function generatePrimaryRunbookCorpus(): RunbookCorpus {
  const runbooks = families.flatMap((family) => angles.map((angle, index) => {
    const runbookId = `TSG-${family.code}-${String(index + 1).padStart(2, "0")}`;
    const title = `${family.title}: ${angle.title}`;
    const related = angles.filter((candidate) => candidate.name !== angle.name).slice(0, 3)
      .map((candidate) => `TSG-${family.code}-${String(angles.indexOf(candidate) + 1).padStart(2, "0")}`);
    const bodyMarkdown = renderRunbook(family, title, angle.focus, related);
    return {
      runbookId,
      title,
      incidentClass: family.incidentClass,
      severityFloor: family.severityFloor,
      sourceUri: family.sourceUri,
      sourceKind: "original_mit_official_reference" as const,
      bodyMarkdown,
      metadata: {
        license: "MIT" as const,
        family: family.family,
        angle: angle.name,
        keywords: [...new Set([...family.keywords, ...angle.title.split(/\s+/)])],
        sourceCheckedAt,
      },
    };
  }));
  return runbookCorpusSchema.parse({
    schemaVersion: 1,
    corpusId: "primary-v1",
    corpusVersion: "runbooks-v1",
    corpusKind: "primary",
    license: "MIT",
    sourceCheckedAt,
    runbooks,
  });
}

function renderRunbook(family: FamilyDefinition, title: string, focus: string, related: string[]): string {
  return [
    `# ${title}`,
    "",
    "## Scope",
    "",
    `${focus} This guide covers ${family.scope}. It is diagnostic guidance, not permission to mutate a database or host.`,
    "",
    "## Signals that support this diagnosis",
    "",
    ...family.supports.map((value) => `- ${value}.`),
    "",
    "## Signals that contradict this diagnosis",
    "",
    ...family.contradicts.map((value) => `- ${value}.`),
    "",
    "## Safe diagnostic steps",
    "",
    ...family.diagnostics.map((value, index) => `${index + 1}. ${value}.`),
    "",
    "## Recommended action policy",
    "",
    ...family.actions.map((value) => `- ${value}.`),
    "",
    "## Escalation conditions",
    "",
    ...family.escalations.map((value) => `- ${value}.`),
    "",
    "## What not to do",
    "",
    ...family.avoid.map((value) => `- ${value}.`),
    "",
    "## Related runbooks",
    "",
    ...related.map((value) => `- ${value}`),
    "",
  ].join("\n");
}

export interface RunbookChunk {
  chunkId: string;
  runbookId: string;
  ordinal: number;
  headingPath: string;
  content: string;
  tokenCount: number;
  contentSha256: string;
}

export function headingAwareChunks(runbook: Runbook): RunbookChunk[] {
  const lines = runbook.bodyMarkdown.trim().split(/\r?\n/);
  const documentTitle = lines.find((line) => line.startsWith("# "))?.slice(2) ?? runbook.title;
  const sections: Array<{ heading: string; lines: string[] }> = [];
  let current: { heading: string; lines: string[] } | undefined;
  for (const line of lines) {
    if (line.startsWith("## ")) {
      current = { heading: line.slice(3), lines: [] };
      sections.push(current);
    } else if (current !== undefined && !line.startsWith("# ")) {
      current.lines.push(line);
    }
  }
  if (sections.length < 6) throw new Error(`Runbook ${runbook.runbookId} has too few heading-aware sections`);
  return sections.map((section, ordinal) => {
    const content = `# ${documentTitle}\n\n## ${section.heading}\n${section.lines.join("\n").trim()}\n`;
    return {
      chunkId: `${runbook.runbookId}-c${String(ordinal + 1).padStart(2, "0")}`,
      runbookId: runbook.runbookId,
      ordinal,
      headingPath: `${documentTitle} > ${section.heading}`,
      content,
      tokenCount: lexicalTokenCount(content),
      contentSha256: sha256(content),
    };
  });
}

export function runbookCorpusManifest(corpus: RunbookCorpus) {
  const runbooks = corpus.runbooks.map((runbook) => ({
    runbookId: runbook.runbookId,
    bodySha256: sha256(runbook.bodyMarkdown),
    sourceSha256: hashJson({ sourceUri: runbook.sourceUri, bodyMarkdown: runbook.bodyMarkdown }),
    chunks: headingAwareChunks(runbook).map((chunk) => ({
      chunkId: chunk.chunkId,
      contentSha256: chunk.contentSha256,
    })),
  }));
  return {
    schemaVersion: 1,
    corpusId: corpus.corpusId,
    corpusVersion: corpus.corpusVersion,
    corpusKind: corpus.corpusKind,
    license: corpus.license,
    sourceCheckedAt: corpus.sourceCheckedAt,
    runbooks,
  };
}

function lexicalTokenCount(value: string): number {
  return value.match(/[\p{L}\p{N}_]+|[^\s]/gu)?.length ?? 0;
}
