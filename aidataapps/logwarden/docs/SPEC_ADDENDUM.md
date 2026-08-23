# Lab 03 Specification Addendum — LogWarden

## Review, binding corrections, scope tiers, and additions for the implementation and research agent

**Status:** Governing addendum to `aidataapps_logwarden_lab_3_spec.md` (the "spec", the governing replacement specification). Read the spec completely, then this addendum. Where they conflict, **this addendum wins**. Where this addendum is silent, the spec stands.

**Review basis:** The spec; the `aidataapps-rag` branch (Lab 01 runtime, registry, gateway); the Lab 02 spec and its addendum (decode hygiene, digest-resolved registry, permutation nulls, exact-vs-ANN discipline, Colab execution plan); and the current Microsoft Learn pages for Extended Events targets and `sys.fn_xe_file_target_read_file` (updated July 2026), `CREATE VECTOR INDEX` / `VECTOR_SEARCH` (updated August 2026), and `AI_GENERATE_EMBEDDINGS` / `CREATE EXTERNAL MODEL`.

**Scientific status:** Unchanged. A clean null is a valid outcome; a safety violation is not; and — added here — an *unfinished* lab is not an outcome at all. Most of this addendum exists to make sure a state of record is reached.

### How to read this document

- **MUST** — binding. **SHOULD** — do unless a recorded reason prevents it (log it in `EXPERIMENT_LOG.md`). **MAY** — optional, only after the tier above it is closed.
- ID prefixes: `C-` corrections to the spec, `T-` scope tiers and critical path, `S-` source/capture, `A-` agent and transport, `K-` knowledge/retrieval, `D-` database/telemetry, `E-` evaluation/statistics, `P-` execution plan, `O-` outputs, `Q-` tests. Each item names the spec section it amends.

---

# 0. Review verdict

## 0.1 What is right and must not be weakened

- **Capture once, replay many** (§6). This is the correct fix for the confound in the earlier draft and is the single most important design decision in the document. The cross-plane parity cell (§6.3) is the right check on it.
- **The unit of analysis is the incident episode** (§3.2), with grouped inference at the scenario-template group (§23.1).
- **Rules as a serious competitor** (§15.1 B1, "the purpose is not to make the LLM look clever"), retrieval-only (B2), and the oracle packet classifier (B3) as an information-sufficiency check.
- **Agent arms as frozen identities** (§15.4) and the arm ladder `A-direct → A-rag → A-tools → A-router`.
- **End-to-end success as the strict headline** (§21.1), complete-case versus end-to-end reporting, and the failure-stage funnel (§28.3). These prevent a good classifier from hiding a broken agent.
- **Required/optional/forbidden tool scoring** (§10.7, §21.5) — "a required-tool episode cannot receive full credit if the agent guessed the right answer without the diagnostic step."
- **Procedure-only tool access, no raw SQL tool, separate injector principal, independent safety audit** (§3.4, §18.1, §19).
- **Raw response before derivation; no transaction across inference; no `MERGE` on correctness paths; SQL lease queue with `UPDLOCK, READPAST`** (§14.2, §18).
- **Negative controls as mandatory, not decorative** (§22), and the leakage audit before freeze (§22.12).
- **Reports generated from rows, never hand-assembled** (§28.5).

## 0.2 The twelve problems that change outcomes, ranked

1. **Scope is roughly three times a realistic build for one agent session, and the spec's drop order (§24.6) only covers compute cells, not implementation.** Nine schemas, sixty-plus tables, temporal tables, full-text, Query Store, columnstore, two custom XE sessions, seven principals, OpenTelemetry-style spans, a lease queue, four arms × four models, fourteen controls, live parity, storm ramps, ANN, seventeen reports, and a seven-page UI. Without a critical path the agent will build a thin veneer of everything and close nothing. **§T defines a Tier 1 Minimum State of Record that must pass `repro.sh` before any Tier 2 item starts.**
2. **Correlation (RQ6, H7) is not testable in the primary arms as designed.** Replay packets already bundle every event of an episode, so the deterministic packet builder has done the correlation before the model sees anything; `alerts/incident` is trivially 1. Correlation can only be measured in the raw event-by-event arm (listed as *secondary*, §15.3) or in the live plane. Either promote a bounded raw-event arm or rescope RQ6 (§E-4).
3. **Transport mode is confounded with model identity.** Muse's inherited profile has a native tool-call parser; the other three do not. "Native where supported, JSON otherwise" (§14.3) means model quality and transport are measured together with one transport per model, so "the evaluation distinguishes transport mode from semantic tool correctness" cannot hold. **One transport (structured JSON) for all four models in the primary campaign; native tool calling is a secondary ablation** (§A-1). This also keeps the profiles byte-identical to the frozen ModelPrint registry.
4. **`A-router` does not need inference in the replay plane.** If its LLM branch is the same prompt/arm as `A-tools`, the router is the composition `B1 ∘ A-tools` and can be *derived* from rows already persisted, saving a quarter of the grid. It is a real arm only in the live plane, where routing changes load (§A-5).
5. **The rules baseline is artificially weakened by the lab's own error-number range.** Every lab-authored `RAISERROR` uses a 51xxx number the lookup does not know, so B1 would route all synthetic "corruption-signal" and custom-error episodes to `escalate`, inflating LLM lift on episodes that are in fact trivially recognizable by message template. B1 must be allowed message-template regexes and event names, authored from runbooks and public docs, never from test packets (§E-3).
6. **Near-real-time claims will be wrong by default.** Extended Events sessions buffer events for `MAX_DISPATCH_LATENCY` (default 30 s) before writing to the file target. The custom session must set it to 1–3 s; `system_health` keeps its default and must be labeled accordingly (§S-1).
7. **Full-text search is not in the base `mssql/server` image on Linux.** It requires building a custom image with the `mssql-server-fts` package, and full-text population is asynchronous. Without this, `lexical_fulltext` and `hybrid_rrf` (required methods, §13.4) cannot exist. A Dockerfile and a governed app-side BM25 fallback are required (§K-1).
8. **Query Store defaults hide the per-run SQL metrics the spec promises.** Runtime-stats intervals default to 60 minutes and data flush to 15 minutes; per-run deltas (§17.8, §20.3) need `INTERVAL_LENGTH_MINUTES = 1`, `QUERY_CAPTURE_MODE = ALL`, and `sp_query_store_flush_db` before each materialization (§D-1).
9. **Permissions and events the spec assumes do not exist as written.** There is no `login_failed` XE event (use `error_reported` 18456 plus the ERRORLOG line that carries the state); a database-filtered session drops login failures and server-level errors; `sp_readerrorlog` requires `securityadmin`; `RAISERROR` at severity 19–25 requires `ALTER TRACE` and `WITH LOG`, and severity ≥ 20 kills the injector's connection (§S-2, §S-3, §S-4).
10. **Replay fidelity has an unspecified hole: tool calls with arguments the snapshot does not contain.** A model may call `get_log_space` for the wrong database. The frozen-snapshot store must be keyed by canonical arguments and return a deterministic miss result, recorded as `snapshot_miss`, so argument errors are scored rather than silently served (§A-3).
11. **Statistical power is thin and the design is paired but the plan is not.** 384 episodes over seven roles leaves roughly a dozen test episodes per family per model. Every model/arm sees the *same* episodes, so the primary contrasts are paired; a paired grouped bootstrap of the difference is far more powerful than comparing independent intervals. Target ≥ 600 episodes (cheap, because capture happens once) and make the pairing explicit (§E-1, §E-2).
12. **Leakage channels that the audit does not list.** Tool snapshots can carry the injector's SQL text or object names that encode the scenario; `REPLAY_LIVE_DIVERGENCE` (§6.3) is referenced but missing from the taxonomy; identifier masking (§22.2) breaks tool arguments unless the gateway maps placeholders (§E-6, §A-4).

## 0.3 What this addendum adds beyond corrections

- A three-tier scope with a critical path, a Tier 1 schema subset, and a per-tier closeout rule (§T).
- Concrete source-capture mechanics: session definition skeleton, filters that survive login failures, dispatch latency, rollover retention, ERRORLOG access without `securityadmin`, and injector permissions (§S).
- Replay fidelity rules for snapshots, masking, and episode windows (§A-3, §A-4, §S-6).
- A structured-JSON transport contract shared by all models, a guided-decoding secondary cell, and user-turn-only operating contract (§A-1, §A-2).
- The full-text image build, population wait, RRF-in-SQL sketch, and a BM25 fallback (§K).
- Query Store, columnstore, temporal, principal, DMV-permission, and BACPAC corrections (§D).
- Paired bootstrap, power rule, control scoping, taxonomy completion, and B1 strengthening (§E).
- A Colab execution plan with parallel replay workers and per-residency checkpoints (§P), and a trimmed results pack (§O).

---

# 1. Binding corrections to the spec

## C-1 Implementation drop order and Minimum State of Record (amends §24.6, §25) — MUST

See §T. The spec's drop order governs compute cells; this addendum adds a drop order for *what gets built*. The agent closes Tier 1 (passing `repro.sh` in row-only mode on a fresh checkout) before starting any Tier 2 work, and records each deferred item in `EXPERIMENT_LOG.md` with its tier and reason.

## C-2 Correlation is evaluated where it can actually vary (amends §4.1 RQ6, §4.2 H7, §15.2–15.3, §21.7) — MUST

In replay, packets are pre-correlated, so primary-arm correlation metrics are degenerate. Amend:

- Primary-arm correlation metrics are reported as **packet-builder correlation** (deterministic grouping versus ground truth), labeled as a property of the harness, not of any model.
- Model correlation is measured by the `A-raw-events` arm: the model receives one canonical event at a time with the frozen recent-history summary and must emit `correlationKey`; evaluated on Regime M episodes only, for at least two profiles (the fastest and the best on `A-tools`), Tier 2.
- Live-plane correlation is measured in the parity and storm cells as designed.
- RQ6/H7 are rescoped to: "Does model-generated correlation on raw events approach deterministic packet correlation, and at what alert cost?"

## C-3 One transport for all models (amends §14.3, §16.4) — MUST

See §A-1. Primary campaign: structured-JSON tool protocol for all four profiles, no `tools` parameter sent, registry engine args unchanged from ModelPrint. Native tool calling is the `transport-native` secondary cell on `A-tools` for profiles whose pinned image has a parser, Tier 2.

## C-4 `A-router` is derived in replay, real in live (amends §15.2, §24.3) — MUST

See §A-5.

## C-5 B1 may use message templates and event names (amends §15.1) — MUST

See §E-3.

## C-6 Dispatch latency, filters, and events that exist (amends §11.1, §11.2) — MUST

See §S-1 through §S-4.

## C-7 Full-text search requires a custom image and a fallback (amends §8, §13.4, §8.1) — MUST

See §K-1.

## C-8 Query Store configuration for per-run deltas (amends §17.8, §20.3, §25 LW-1) — MUST

See §D-1.

## C-9 Replay snapshot keying and masking consistency (amends §6.1, §12.1, §22.2) — MUST

See §A-3 and §A-4.

## C-10 Paired, powered statistics and control scoping (amends §10.3, §22, §23) — MUST

See §E-1, §E-2, §E-5.

## C-11 Taxonomy completion (amends §5, §6.3) — MUST

Add to the taxonomy table: `REPLAY_LIVE_DIVERGENCE` (parity agreement below the frozen threshold; blocks transfer of replay quality claims to live), `SNAPSHOT_MISS_DOMINATED` (a model's tool failures are mostly argument errors revealed by snapshot misses), `TRANSPORT_SENSITIVE` (native and JSON transports disagree beyond tolerance for the same profile), `GUIDED_DECODING_RECOVERS` (constrained decoding removes most contract failures without changing decisions), and `PACKET_CORRELATION_ONLY` (correlation credit belongs to the harness, not the model).

## C-12 Tool budgets and deadlock-graph size (amends §14.5, §14.6) — SHOULD

Raise `max_tool_calls` to 4 (two required diagnostics plus `runbook_search` plus one retry already hits 3). `get_deadlock_graph` returns a **summarized** graph (victim, processes with session/login/app/isolation/wait resource, resource list, and the two statement hashes), never raw XML; the raw XML is in the packet's source events for audit. Cap any single tool result at 4,000 characters and record truncation.

---

# 2. Scope tiers and critical path (new; governs §24–25)

## T-1 Tier 1 — Minimum State of Record (MSR) — MUST

Tier 1 is the smallest build that yields a valid, reproducible, publishable state of record. Everything in it is required; nothing outside it may begin until Tier 1 closeout passes.

| Area | Tier 1 content |
|---|---|
| Databases and principals | Both databases; migrations 001–008 + 010 (security) in reduced form; **two principals**: `lw_lab` (migrate, inject, ingest, evaluate, report) and `lw_agent`. The safety claim needs exactly this separation; the seven-way split is Tier 2. |
| Schema subset | `control.{schema_migrations, capability_snapshots, campaigns, campaign_freezes, model_profiles, embedding_profiles, decode_configs, agent_arms, runs, jobs, job_attempts, evidence_events, run_artifacts}`; `workload.{scenario_definitions, scenario_variants, schedules, schedule_items, injection_executions, expected_evidence, disposable_databases}`; `ingest.{sources, source_cursors, ingestion_batches, raw_events, canonical_events, event_fingerprints, injection_event_links, context_snapshots, incident_packets, packet_events}`; `ops.{incidents (plain table + append-only transitions), work_items, transitions (one table for incident and work-item transitions), action_proposals, work_tracking_items}`; `kb.{runbooks, runbook_chunks, chunk_embeddings, search_corpora, search_runbook_exact, retrieval_runs, retrieval_results}`; `agent.{agent_runs, turns, model_requests, model_responses, agent_steps, tool_invocations, validation_events, decisions, decision_citations, policy_events}`; `telemetry.{metric_samples, queue_samples}`; `eval.{ground_truth_episodes, predictions, tool_expectations, tool_scores, retrieval_scores, decision_scores, calibration_models, metric_results, bootstrap_results, permutation_results, taxonomy_assignments, claims}`; `reporting.{v_run_completeness, v_model_arm_scorecard, v_failure_stage_funnel, v_tool_scorecard, v_retrieval_scorecard, v_calibration_rows, v_safety_audit, v_claim_support, report_snapshots}`. |
| Deferred schema | `telemetry.{traces, spans, model_service_samples, sql_resource_samples, query_store_intervals, xe_pipeline_samples}` (Tier 2; spans are derived from `agent.agent_steps` by a view until then); system-versioning on `ops.incidents` (Tier 2 migration); columnstore (Tier 3); JSON-type migration 011 (Tier 3). |
| Sources | Custom XE session with file target and 1–3 s dispatch latency; XE offset-cursor ingester; ERRORLOG file tail; `system_health` read-only comparator. |
| Workload | ≥ 600 episodes across all ten families and five regimes (§E-2), deterministic schedule, per-injector verification and cleanup, disposable-DB token protection. |
| Packets | Packet builder, protected ground truth, group-disjoint roles, leakage audit, freeze record, SQL `.bak` checkpoint. |
| Knowledge | Runbooks; heading-aware chunker; Qwen embeddings; exact `VECTOR_DISTANCE`; full-text via custom image **or** the governed BM25 fallback; `hybrid_rrf`; B2 retrieval-only; `shuffled_runbook` and `oracle_runbook` modes. |
| Agent | SQL lease queue with ≥ 4 workers; state machine; JSON transport; validation layers; policy engine; tools `runbook_search`, `get_recent_incident_counts`, `get_blocking_snapshot`, `get_log_space`, `get_active_transactions`, `get_backup_history`, `get_deadlock_graph` (summarized), `open_work_item`. `get_query_store_context` is Tier 2. |
| Campaign | Port gates; `qwen-smoke` end to end; replay of `A-direct` and `A-tools` for all four profiles on all roles; `A-rag` on the retrieval-covered subset; `A-router` derived; B0/B1/B2/B3. |
| Controls | Error-number masking (§22.1), shuffled runbooks (§22.7), retrieval ablation (§22.6 via the arm ladder), label permutation (§22.11), leakage audit (§22.12), batching invariance (§22.14). Others are Tier 2. |
| Scoring | End-to-end success, classification, severity, action, tool, retrieval/grounding, calibration/abstention, contract reliability, latency decomposition from persisted timestamps, missingness funnel; paired grouped bootstrap; permutation nulls; taxonomy; claims. |
| Reports | `SCORECARD.md`, `FAILURE_ATLAS.md`, `GROUNDING_REPORT.md`, `TOOL_USE_REPORT.md`, `CALIBRATION_REPORT.md`, `NEGATIVE_CONTROLS.md`, `SAFETY_AUDIT.md`, `LOGWARDEN_STATE_OF_RECORD.md`, `LOGWARDEN_CLAIMS_TABLE.md`, `LOGWARDEN_LIMITATIONS.md`, `LOGWARDEN_REPRODUCIBILITY.md`, plus the preregistration and freeze record. Other reports become sections of these until Tier 2. |
| Reproducibility | `repro.sh` row-only mode and `.bak` restore mode. BACPAC is Tier 2. |
| UI | None. The CLI and reporting views are authoritative; a UI that shows the same rows is Tier 3. |

**Tier 1 closeout:** `repro.sh --mode rows` passes on a fresh checkout; `SCORECARD.md` regenerates byte-for-byte; the safety audit is clean; every Tier 1 job has a terminal disposition; `EXPERIMENT_LOG.md` lists every deferred item. Only then does Tier 2 start.

## T-2 Tier 2 — Systems and ablations — SHOULD

Live plane (parity ≈ 48 episodes per profile; storm ramps for `A-tools` and `A-router` on at least the fastest and the best profile); `telemetry` tables and vLLM `/metrics` scraping; Query Store reporting (§D-1); temporal `ops.incidents`; seven-principal split and permission matrix; SQL Server Audit evidence for the safety auditor; `A-raw-events` correlation arm; `transport-native` and `guided-json` cells (§A-1, §A-2); `get_query_store_context`; remaining controls (§22.2–22.5, §22.8–22.10, §22.13); `AI_GENERATE_CHUNKS` comparator; second embedding profile; ANN build with plan proof; BACPAC export (drop vector indexes first); `THROUGHPUT_REPORT.md`, `SQL_OPERATIONS_REPORT.md`, `CORRELATION_REPORT.md`, `ANN_REPORT.md`.

## T-3 Tier 3 — Extensions and surfaces — MAY

Inspection UI (Live, Incident, Trace, Benchmark first; others later); columnstore comparison; JSON-type/JSON-index migration; `memory_frozen`; natural-sampling stability; SQL-native embedding path; genuine corruption track; batched multi-episode calls; everything in §35.

## T-4 Tier rule for the agent — MUST

A Tier 2 or 3 item that is started but not finished must be reverted or feature-flagged off so that Tier 1 `repro.sh` still passes at every commit on the branch. The branch must never be in a state where the Tier 1 state of record cannot be regenerated.

---

# 3. Sources and capture (amends §11, §10.5)

## S-1 Custom XE session definition (amends §11.1) — MUST

Binding properties of `logwarden_capture`:

- `event_file` target with `MAX_FILE_SIZE = 16 MB`, `MAX_ROLLOVER_FILES = 20` (small enough that the rollover test is exercised in smoke; large enough that retention always exceeds ingest lag — the reader passes `initial_file_name`/`initial_offset`, and the initial file must still exist).
- `MAX_DISPATCH_LATENCY = 1 SECONDS` (3 s if CPU overhead is measurable). Record it. `system_health` keeps its default (30 s) and must be labeled "up to 30 s dispatch lag" wherever it is compared.
- `STARTUP_STATE = ON`, `EVENT_RETENTION_MODE = NO_EVENT_LOSS` during capture runs (and `ALLOW_SINGLE_EVENT_LOSS` during storm, recorded per run).
- Events: `sqlserver.error_reported` (severity ≥ 11 or error_number in the lab list), `sqlserver.xml_deadlock_report`, `sqlserver.blocked_process_report` (requires `sp_configure 'blocked process threshold (s)'`; set 5 s by default and verify the docs' guidance about values below 5), `sqlserver.attention`, `sqlserver.database_file_size_change`, and `sqlserver.sql_batch_completed` / `sqlserver.rpc_completed` **only** with a tight predicate on the lab app name. Actions: `database_name`, `database_id`, `session_id`, `client_app_name`, `username`, `client_hostname`, `sql_text` (bounded; see §E-6), `plan_handle` only where a scenario requires it.
- **Filter by application name, not database.** Use `client_app_name LIKE N'LogWarden-Inject%'` as the primary predicate; login failures and server-level errors have no workload database context and would be dropped by a `database_id` filter. Add `OR database_name LIKE N'LW_%'` for events raised in disposable databases by sessions whose app name is not visible to the event.
- **Exclude the agent's own SQL.** Agent and ingester connections set `Application Name=LogWarden-Agent` / `LogWarden-Ingest`; the capture predicate excludes them so tool failures do not pollute the benchmark stream. (A separate `logwarden_agent_observability` session capturing the agent's SQL is Tier 2.)
- Commit the session DDL under `db/xe/logwarden_capture.sql`; record its SHA-256 and the resolved event/action metadata (`sys.dm_xe_objects`) in `environment/xe-capabilities.json`.

## S-2 Login failures and server-level errors (amends §11.1, §10.5) — MUST

There is no `login_failed` Extended Event. Login failure evidence is (a) `error_reported` with `error_number = 18456` and (b) the ERRORLOG line, which carries the **state** and the reason text. Scenario ground truth for authentication episodes must be defined on the ERRORLOG row as the anchor and the XE row as supporting evidence. The port of every authentication scenario verifies both rows are captured.

## S-3 Injector permissions (amends §10.5, §19.1) — MUST

- `RAISERROR ... WITH LOG` at severity 19–25 requires `ALTER TRACE` (or sysadmin) and `WITH LOG`; grant `lw_lab` (Tier 1) / `lw_injector` (Tier 2) the server permission `ALTER TRACE`, not sysadmin, and record it in the permission audit.
- Severity ≥ 20 terminates the raising connection. The injector driver must expect a dropped connection on those scenarios and verify success through captured evidence, not the driver's return code (the spec's "success is not injection return code" rule already requires this).
- `sp_addmessage` requires `serveradmin`/`sysadmin`; add lab messages during `db:setup` under the migrator principal, not at injection time.
- `sp_readerrorlog` requires `securityadmin`. Tier 1 reads the ERRORLOG by **file tail** (`/var/opt/mssql/log/errorlog`, read-only bind mount or `docker exec`), so no principal needs `securityadmin`; record the method in `ingest.sources.config_json`. If file access is unavailable in the nested Colab runtime, grant `securityadmin` to `lw_lab` only — never to `lw_agent` — and record the exception.

## S-4 `system_health` as comparator (amends §11.2) — SHOULD

`system_health` records `error_reported` for severity ≥ 20 plus a fixed list of memory/IO error numbers, `xml_deadlock_report`, long waits, and health diagnostics. Report per family the capture-rate difference between `logwarden_capture` and `system_health`; do not expect `system_health` to see severity-16/17 workload errors. Reading its files requires `VIEW SERVER STATE` and file-system access to `/var/opt/mssql/log/`.

## S-5 ERRORLOG idempotency (amends §11.3) — SHOULD

The file tail key is `(log_generation, line_timestamp, process_info, raw_line_sha256, ordinal_within_timestamp)`; log generation is detected by the header line written on restart or `sp_cycle_errorlog`. The smoke suite must include `sp_cycle_errorlog` mid-run and a container restart and prove zero duplicates and zero losses.

## S-6 Episode windows and schedule spacing (amends §10.4, §12.2) — MUST

Define the packet window as `[anchor − 30 s, anchor + 90 s]` (frozen per family; blocking and log-full families may need longer). The schedule builder enforces a minimum gap between consecutive episodes of different templates ≥ the sum of their windows, except inside scenario groups that intentionally overlap (recurrence storms, multi-event incidents). Record window and gap parameters in the schedule hash. The leakage audit reports any canonical event that falls inside two episodes' windows.

## S-7 Snapshot triggering (amends §12.1) — MUST

Context snapshots are **triggered by the injector at scenario-defined checkpoints** (e.g., `t+2s` while blocking is active; after the third 9002), not by a timer. The scenario definition lists `snapshot_plan: [{kind, at_ms_after_start}]`; the driver calls the snapshot procedures at those times and links results to the episode. Replay tools then serve these snapshots (§A-3). In the live plane the same procedures run on demand.

## S-8 DMV permission surface for tools (amends §14.6, §18.1, §19.1) — MUST

Tool procedures need cross-database and server-state reads. Use: `sys.dm_db_log_stats(db_id)` for log space (server-wide, typed by database ID), `sys.master_files` for file limits, `sys.dm_exec_requests` + `sys.dm_os_waiting_tasks` for blocking, `sys.dm_tran_active_transactions` + `sys.dm_tran_session_transactions` for active transactions, the lab's own `workload.backup_attempts` table (not `msdb`) for backup history, and `sys.query_store_*` in the workload database for Query Store context (Tier 2). Grant `lw_agent` server-level `VIEW SERVER STATE` and, in each disposable database, `VIEW DATABASE STATE` via the database-creation template. Enforce the database-name restriction inside the procedure (`LW_%` prefix and a matching `workload.disposable_databases` row) and deny `lw_agent` everything in `eval` and the `ground_truth_json` column of `workload.scenario_definitions` (column-level `DENY`). Certificate signing is a Tier 2 refinement, not a Tier 1 requirement.

---

# 4. Agent runtime and transport (amends §14, §15, §16)

## A-1 Structured-JSON transport for all models (amends §14.3) — MUST

Every model, every arm, every turn: the operating contract tells the model to answer with exactly one JSON object that is either `{"kind":"tool_request", ...}` or `{"kind":"decision", ...}` (spec §14.3 shapes). Tool results are returned as a user-turn message `{"kind":"tool_result","tool":..., "callId":..., "result":...}`. No `tools` parameter is sent, so the inherited engine arguments (including Muse's parser flags) stay as frozen in ModelPrint and the chat template's tool path is never invoked. The parser strips a single fenced code block if present and records `repair_kind` (`none | fence_strip | leading_text_strip`) per response; any other repair is a contract failure. `transport-native` (send `tools`, parse native tool calls) is a Tier 2 cell on `A-tools` for profiles with a pinned parser, scored for agreement with the JSON transport (`TRANSPORT_SENSITIVE` if below tolerance).

## A-2 Operating contract placement and guided decoding (amends §14.3, §16.2) — MUST / SHOULD

- **Placement (MUST).** No system message. The operating contract, class/action enums, tool schemas, and budget go in the **first user turn**, byte-identical across models; only `chat_template_kwargs` differ by registry. This repeats the Lab 02 addendum rule and removes the Gemma fold as a confound.
- **Unconstrained primary (MUST).** The primary cell does not use constrained decoding, because first-pass contract validity (§21.9) is a headline reliability result.
- **Guided secondary (SHOULD, Tier 2).** A `guided-json` cell on `A-tools` uses vLLM structured outputs (`response_format` with `json_schema`, or the `guided_json` extra parameter, whichever the pinned image supports — record which) with the same discriminated-union schema. Report decision agreement with the unconstrained cell and the contract-failure reduction (`GUIDED_DECODING_RECOVERS`). This is what a shipped agent would use; the lab measures what it buys.

## A-3 Frozen tool snapshots are keyed by canonical arguments (amends §6.1, §12.1, §14.6) — MUST

`ingest.context_snapshots` gains `tool_id`, `canonical_args_sha256`, and `canonical_args_json`. In replay, a tool call resolves to the snapshot whose `(episode_id, tool_id, canonical_args_sha256)` matches after argument canonicalization (lower-cased identifiers, sorted keys, defaults filled). A miss returns a deterministic, schema-valid result — `{"status":"no_data","reason":"no snapshot for arguments"}` — and records `snapshot_miss = 1` on the invocation. Misses are scored as argument-semantic failures when `argumentConstraints` were violated and as `snapshot_coverage_gap` (a harness defect, not a model error) otherwise; the latter count is reported per scenario and must be near zero after the dev tier. `runbook_search` and `get_recent_incident_counts` are computed, not snapshotted (deterministic against the frozen corpus/packet), so they never miss.

## A-4 Masking controls must be applied consistently to packets and tools (amends §22.1, §22.2) — MUST

A masked cell transforms the packet text **and** the snapshot results **and** the tool-argument canonicalizer with the same frozen placeholder map (`LW_DB_1`, `OBJ_1`, `LOGIN_1`, …), so the model can call `get_log_space("LW_DB_1")` and receive the masked snapshot. The map is stored per episode in `eval` (evaluator-only) and never in the packet.

## A-5 `A-router` in replay is a derived arm (amends §15.2, §24.3) — MUST

`A-router` replay results are computed by composition: for each episode, if B1 resolves it (known signature with a rule), the router's decision is B1's; otherwise the decision is the persisted `A-tools` decision for that (episode, profile). Token, latency, and tool costs are summed accordingly (B1 cost ≈ 0). This requires `A-tools` to be run on all episodes, which Tier 1 already mandates. A distinct router prompt ("rules could not resolve this; proceed with diagnostics") is `A-router-prompted`, Tier 2, and is a separate arm identity. In the live plane, `A-router` is a real arm because routing changes queue load.

## A-6 Replay concurrency (amends §14.2, §25 LW-9) — MUST

Replay jobs are independent. Run the replay with `--workers 16` (raise to 32 if vLLM `num_requests_waiting` stays near zero) against `--max-num-seqs 64`, so the per-episode `max_wall_time_seconds = 180` budget never serializes the campaign. Record worker count and `--max-num-seqs` in the run config; the batching-invariance control (§22.14) compares a sequential re-run of 48 episodes.

## A-7 Streaming and TTFT (amends §20.2, §17.7) — SHOULD

Replay uses non-streaming requests (simpler parsing; per-request TTFT recorded as null) and takes service-level TTFT/ITL from the vLLM `/metrics` endpoint at scrape interval (Tier 2). The live plane uses streaming with `stream_options.include_usage` so TTFT is measured client-side at the first content chunk and token usage is taken from the final chunk. Never infer TTFT by division (the spec already forbids it).

## A-8 `max_tokens` and rationale cap (amends §16.2) — SHOULD

Keep `max_tokens = 900` but cap `rationale` at 60 words and `summary` at 25 words in the contract; Muse's reasoning tokens count toward the budget, so the port gate reports the Muse `finish_reason = length` rate on canaries and raises its `max_tokens` to 1400 if needed (recorded as a per-profile decode override, included in the decode hash, and justified in the freeze record).

---

# 5. Knowledge and retrieval (amends §8, §13)

## K-1 Full-text search on Linux (amends §8, §8.1, §13.4) — MUST

- Build `docker/mssql-fts/Dockerfile`: `FROM mcr.microsoft.com/mssql/server:2025-latest` (pinned by digest), switch to root, `apt-get install -y mssql-server-fts` matching the engine build, switch back to `mssql`. Record the resulting image digest; `compose.yaml` uses it for `LogWardenControl`. Verify with `SELECT FULLTEXTSERVICEPROPERTY('IsFullTextInstalled')`.
- Create a full-text catalog and index on `kb.runbook_chunks(heading, content)` with `KEY INDEX` on the chunk primary key, `LANGUAGE 1033`, `STOPLIST = SYSTEM`. Population is asynchronous: `runbooks:build` waits until `FULLTEXTCATALOGPROPERTY(catalog, 'PopulateStatus') = 0` before any retrieval test, and `search:freeze` records the population timestamp.
- **Governed fallback.** If the image build is impossible in the runtime, `lexical_fulltext` is served by an app-side BM25 over the same chunks (frozen parameters `k1 = 1.2`, `b = 0.75`, same tokenizer for all runs), the capability snapshot records `fulltext_supported = 0`, every retrieval row records `actual_mode = lexical_bm25_app`, and `LEXICAL_SUFFICIENT` / `HYBRID_SEARCH_LIFT` claims are scoped to BM25, not SQL full-text. The two lexical engines are not mixed within one campaign.

## K-2 Reciprocal-rank fusion in T-SQL (amends §13.4) — SHOULD

```sql
DECLARE @k INT = 60;  -- frozen RRF constant
WITH lex AS (
    SELECT [KEY] AS chunk_id,
           ROW_NUMBER() OVER (ORDER BY RANK DESC, [KEY]) AS lexical_rank, RANK AS lexical_score
    FROM CONTAINSTABLE(kb.runbook_chunks, (heading, content), @ft_query, 50)
),
vec AS (
    SELECT TOP (50) chunk_id,
           ROW_NUMBER() OVER (ORDER BY VECTOR_DISTANCE('cosine', embedding, @qv), chunk_id) AS vector_rank,
           VECTOR_DISTANCE('cosine', embedding, @qv) AS vector_distance
    FROM kb.search_runbook_exact
    ORDER BY vector_distance, chunk_id
)
SELECT TOP (@top_k)
       COALESCE(lex.chunk_id, vec.chunk_id)            AS chunk_id,
       lex.lexical_rank, vec.vector_rank, lex.lexical_score, vec.vector_distance,
       ISNULL(1.0 / (@k + lex.lexical_rank), 0) + ISNULL(1.0 / (@k + vec.vector_rank), 0) AS fusion_score
FROM lex FULL OUTER JOIN vec ON vec.chunk_id = lex.chunk_id
ORDER BY fusion_score DESC, chunk_id;
```

Store all component ranks (spec §17.6). The full-text query string is built from the normalized message by a frozen function (`CONTAINS` syntax is strict; test it against messages containing quotes, parentheses, and numbers). The BM25 fallback produces the same `lex` shape.

## K-3 Leakage from retrieval queries (amends §13.5, §22.12) — MUST

The model composes the `runbook_search` query. Record it, and add to the leakage audit a check that no runbook chunk contains a lab scenario ID, disposable database name, or the exact synthetic message text (the spec already requires the latter two; scenario IDs are added).

## K-4 `oracle_runbook` and B3 are evaluator-only (amends §13.4, §15.1) — MUST

Both run under `lw_lab`, never through the agent gateway, and their rows carry `evaluator_only = 1` so the safety auditor can prove no agent run saw them.

---

# 6. Database and telemetry (amends §17–§20)

## D-1 Query Store settings (amends §17.8, §20.3, §25 LW-1) — MUST (Tier 2 for reporting; settings applied in Tier 1 so history exists)

On both databases at init: `ALTER DATABASE ... SET QUERY_STORE = ON (OPERATION_MODE = READ_WRITE, QUERY_CAPTURE_MODE = ALL, INTERVAL_LENGTH_MINUTES = 1, DATA_FLUSH_INTERVAL_SECONDS = 60, MAX_STORAGE_SIZE_MB = 1024, WAIT_STATS_CAPTURE_MODE = ON)`. Before materializing `telemetry.query_store_intervals`, run `EXEC sp_query_store_flush_db` in each database. Named query classes are tagged by a comment prefix in the procedure text (`/* LW:queue.claim */`) so Query Store rows can be grouped by class via `query_sql_text`. Record the settings in `environment/sql-settings.json`.

## D-2 Temporal tables are a Tier 2 migration (amends §17.5) — MUST

Tier 1 keeps `ops.incidents` as a plain table plus the append-only `ops.transitions` log (one table for incident and work-item transitions: `entity_kind, entity_id, from_state, to_state, actor, reason, at_utc`). Migration `012_temporal.sql` (Tier 2) adds system-versioning with an explicit history table and `HISTORY_RETENTION_PERIOD` (requires `temporal_history_retention` enabled); the `FOR SYSTEM_TIME` reconstruction test is added then.

## D-3 Columnstore restrictions (amends §17.8) — SHOULD (Tier 3)

A nonclustered columnstore index cannot include LOB columns; define any NCCI on `telemetry`/`agent` facts with an explicit column list that omits `nvarchar(max)` JSON columns. Measure write overhead before keeping it.

## D-4 Backups, BACPAC, and vector indexes (amends §17.2, §25 LW-14, §30.2) — MUST

The `.bak` is the primary checkpoint. BACPAC export (Tier 2) must drop vector indexes first (documented DacFx limitation) and recreate them after import; full-text indexes export but repopulate asynchronously after import, so the restore test waits for population. Record `sqlpackage` version.

## D-5 Queue index and claim pattern (amends §18.2) — SHOULD

Add `CREATE INDEX ix_work_items_claim ON ops.work_items (status, next_attempt_at_utc, priority DESC, work_item_id) INCLUDE (lease_token, leased_until_utc)` so the `UPDLOCK, READPAST` claim seeks rather than scans. Keep the spec's pattern; the concurrency test also asserts that a claimer blocked by lease expiry does not starve (fairness on `work_item_id`).

## D-6 Principals in Tier 1 (amends §19.1) — MUST

Two principals (`lw_lab`, `lw_agent`) with the permission matrix tests that matter for the safety claim: agent can execute approved tool procedures; agent cannot read `eval.*` or `ground_truth_json`; agent cannot execute injectors; agent cannot `KILL`/`ALTER`/`DROP`/`BACKUP`/`RESTORE`/create logins; agent cannot write outside `agent`, `ops.work_items`/`ops.transitions`/`ops.action_proposals`/`ops.work_tracking_items`, and `telemetry`. The seven-way split and module signing are Tier 2.

## D-7 Safety-audit evidence (amends §19.3) — MUST / SHOULD

Tier 1 evidence: the permission matrix test results (negative tests prove the agent *cannot* read ground truth or run injectors) plus the audit query over `agent.tool_invocations`, `ops.action_proposals`, and `ops.transitions`. Tier 2 evidence: a SQL Server Audit (server audit to file) with a database audit specification on `SELECT` against `eval` and `workload` for `lw_agent`, plus the agent-observability XE session, so "no ground-truth table was read by agent sessions" is proven from an independent log rather than only from denied permissions.

## D-8 Minor schema corrections (amends §17.4, §17.7) — SHOULD

Rename the `rowversion rowversion` column in `ingest.source_cursors` to `row_version` (type name as column name is legal but error-prone). Add `snapshot_miss bit` and `truncated bit` to `agent.tool_invocations`, `repair_kind varchar(40)` to `agent.model_responses`, and `derived_from_arm varchar(80) null` to `agent.agent_runs` for derived `A-router` rows.

---

# 7. Evaluation and statistics (amends §10.3, §15.1, §21–§23)

## E-1 The design is paired; analyze it as paired (amends §23.2, §23.3) — MUST

Every profile and every arm scores the same frozen episodes. For every predeclared contrast (§23.2), compute the **paired difference per episode** (for binary outcomes, the discordant-pair statistic; for costs, the per-episode cost difference) and bootstrap it by resampling scenario-template groups, reporting the CI of the difference. Also report each arm's own CI for display, but the claim gate is the paired-difference CI. This is both more powerful and the only correct treatment, since the independence assumption behind comparing two marginal intervals is false here.

## E-2 Power rule and catalog size (amends §10.3, §12.3) — MUST

Before freeze, run the power check in `analysis/statistics.py`: with a paired design, grouped resampling, and an assumed 10-point acceptable-action accuracy difference at base accuracy 0.80, the number of **test** episodes needed for 80% power at α = 0.05 (Holm over the primary family) is computed by simulation from dev-tier rows, not assumed. Size the catalog so that `test_id` alone meets it and every family has ≥ 30 test episodes across `test_id` + `test_variant_holdout`. Practical target: **≥ 600 episodes** (≈ 300 `test_id`, 120 `test_variant_holdout`, 60 `test_unknown`, 60 `calibration`, 60 `dev`, with parity/storm reusing templates). Capture happens once, so the extra cost is injection time (minutes) and agent episodes (≈ 1.5 h per profile for two arms at 16 workers). Variants are generated programmatically from each template (message variants, object/database names, rate contexts, seeds) through `workload.scenario_variants`; the leakage audit confirms variants of one template never cross roles.

## E-3 Strengthen B1 (amends §15.1) — MUST

B1 keys on **event name, error number, severity, state, message-template regex, and the packet's recurrence features** (`sameFingerprint5m`, `sameClass1h`, `openRelatedIncidents`). Templates are authored from runbooks and public SQL Server documentation **before** the capture run and are hashed into the freeze; they may not be derived from test packets. Report B1's coverage (fraction of episodes it resolves) per regime as a headline number, because it defines the tail the LLM is allowed to win on. Lab-range error numbers with recognizable message text must be B1-resolvable, or the "unknown" regime is inflated.

## E-4 Correlation scope (amends §21.7) — MUST

Per §C-2: report `packet_correlation_*` (harness) and `model_correlation_*` (from `A-raw-events`, Tier 2) as distinct metric families. In Tier 1, RQ6 is answered only for the harness and the live parity cell; the state of record says so.

## E-5 Control scoping and compute bound (amends §22, §24.3) — MUST

Each control runs on a **frozen 96-episode control subset** (stratified by family and regime, drawn from `test_id` + `test_unknown`), under `A-tools` only unless the control specifically targets retrieval (`A-rag`) or packets (`A-raw-events`), for all four profiles. The only controls that run on the full test set are label permutation (no inference) and the retrieval ablation (already covered by the arm ladder). This bounds control inference to ≈ 96 × 4 × (number of inference-requiring controls) episodes.

## E-6 Leakage channels added to the audit (amends §22.12) — MUST

- Tool snapshots: scan `sql_text`, object names, and database names in every snapshot for scenario IDs, template names, regime letters, and ground-truth class names; injector SQL text must use opaque identifiers (`lw_t_0417`), never scenario names. Snapshot `sql_text` is replaced by its hash unless the scenario's `snapshotPolicy` allows text.
- Runbook chunks: scenario IDs and lab message numbers (51xxx) must not appear.
- Packet `recentHistory`: must not include decisions from other *test* episodes (memory is off in the primary; the audit proves it).
- Event-window overlap (§S-6).

## E-7 Calibration features (amends §21.8) — SHOULD

Add `b1_resolved` (rules coverage flag) and `b1_agrees_with_model` to the candidate feature list; they are usually the strongest predictors of correctness and are legitimately available at decision time. Fit on `calibration` only; lock before test.

## E-8 Severity ordinal cost (amends §21.3) — SHOULD

Freeze the ordinal cost matrix in `config/thresholds.json` at freeze time (e.g., undercall by one level = 1, by two = 3, by three = 6; overcall = half of undercall) and report both exact severity accuracy and mean ordinal cost.

## E-9 Missingness is an outcome, not a filter (amends §23.6) — MUST

Every `eval.predictions` row for a failed run keeps `eligible = 1` with `outcome = failure` and `failure_stage` set; end-to-end metrics count it as a failure; complete-case metrics exclude it with the exclusion visible in the denominator columns of every table. A model with > 15% terminal failures on `test_id` receives `RUNTIME_FRAGILE` regardless of its complete-case accuracy.

## E-10 Evidence tags (amends §34) — SHOULD

Use the claim-ledger tags adapted for this lab: `OBS` (observed metric), `PAIRED` (paired contrast with CI and null), `CAUSAL-APPLICATION` (ablation effect), `SYSTEMS` (latency/throughput), `AUDIT` (safety), `HARNESS` (property of the capture/packet pipeline, e.g., packet correlation), `DERIVED` (composed arm such as replay `A-router`).

---

# 8. Execution plan for the Colab runtime (amends §24–§26; see Lab 02 addendum §P)

## P-1 Stage budgets (RTX PRO 6000, 96 GB; approximate) — SHOULD

| Stage | GPU | Wall clock |
|---|---|---|
| LW-0..LW-1 foundation, schema (Tier 1 subset), FTS image build | no | 2–3 h agent time; image build 5–10 min |
| LW-2..LW-4 injectors, capture run (≥ 600 episodes), packets, freeze | no | capture run ≈ 1 h (most injectors are milliseconds; deadlock/blocking/log-full scenarios seconds to a minute each; recurrence storms dominate) |
| LW-5 runbooks, embeddings, retrieval | embedding only | 30 min |
| LW-6 runtime with fake gateway | no | agent time |
| LW-7 `qwen-smoke` end to end | yes | 30–45 min including model download |
| LW-8/9 per target profile: download 5–15 min, load 3–5 min, port gate 10 min, `A-direct` + `A-tools` (+ `A-rag` subset) at 16 workers ≈ 60–90 min, Tier 1 controls ≈ 20 min, `.bak` + raw mirror 5 min, evict | yes | ≈ 2 h per profile → 8 h for four |
| LW-10 scoring, bootstrap (10k), permutations (1,000 final) | no | 1–2 h CPU (vectorize the metric path; parallelize permutations) |
| LW-14 closeout, `repro.sh` on fresh checkout | no | 1 h |
| Tier 2 live parity + storm | yes | ≈ 1 h per profile benchmarked |

## P-2 Checkpoints and resume — MUST

After every profile residency: `BACKUP DATABASE LogWardenControl` to `.bak`, mirror it with `raw/model-responses-<profile>-<arm>.jsonl` to the run archive, write `RESUME.md` with the next command. A fresh VM restores the `.bak` and continues with the next profile. Replay jobs are keyed (§17.2) so a partial residency resumes without duplicate inference. Generation-side processes run under `nohup`/`tmux` and are polled.

## P-3 Residency order and warm-up — SHOULD

Order profiles so the slowest download is last; pull all digest-pinned images before the first residency; run the port gate canaries twice (cold, warm) and report both (the spec separates cold start from warm latency).

## P-4 Workers and service settings — MUST

Replay: `--workers 16`, chat model `--max-num-seqs 64`, `--generation-config vllm` (Lab 02 addendum decode hygiene), effective `SamplingParams` captured. Live plane: record `--max-num-seqs` as part of the service identity and run storm ramps at two settings (8 and 64) for at least one profile; the difference is a systems result.

## P-5 Disk — MUST

One chat model resident (≤ 65 GB) + two embedding services + SQL Server + FTS image. Verify free disk ≥ 1.5 × the next model before download. Control database at Tier 1 scale is small (tens of MB of rows; raw responses ≈ 200–400 MB across the grid); keep raw JSONL on the mirror, not in Git.

---

# 9. Results pack (amends §28–§29)

## O-1 Headline scorecard — MUST

`reports/SCORECARD.md` + `tables/scorecard.csv`: one row per (profile × arm), columns as spec §28.2, **plus** paired Δ over B1 with CI and permutation p for acceptable-action accuracy and cost-weighted loss, B1 coverage, `snapshot_miss` rate, `repair_kind` distribution, and taxonomy labels. Rows for B0/B1/B2/B3 and the derived `A-router` are included and marked `DERIVED`/`BASELINE`.

## O-2 Figures (Tier 1 set; each with source CSV, query bundle hash, caption) — MUST

```text
F01_lift_over_rules.png           paired Δ vs B1 per profile×arm, by regime, with CI and null band
F02_failure_funnel.png            failure-stage funnel per profile×arm
F03_confusion_grid.png            class confusion, profile × arm
F04_action_cost.png               cost-weighted loss with miss/false-alarm decomposition
F05_tool_precision_recall.png     required-tool recall, forbidden rate, argument validity, snapshot-miss rate
F06_grounding_causal_effect.png   A-direct vs A-rag vs A-tools on covered episodes; shuffled-runbook control
F07_retrieval_scorecard.png       recall@k / MRR / nDCG for lexical, vector, hybrid, oracle, shuffled
F08_reliability_risk_coverage.png reliability diagrams and risk-coverage curves, raw vs calibrated
F09_masking_dropoff.png           error-number masking effect per profile (how much is lookup)
F10_contract_reliability.png      first-pass valid, repaired, final valid, by profile
F11_latency_decomposition.png     queue wait, model, tools, validation, persist (replay)
F12_permutation_nulls.png         observed statistics against permutation distributions
F13_b1_coverage_by_regime.png     what fraction of each regime rules alone resolve
```

Tier 2 adds throughput frontier, queue age, replay/live parity, correlation, SQL component cost, and ANN recall/latency figures.

## O-3 README results block — MUST

Generated from `eval.claims` and `metric_results` by a script (never hand-written): which arm/profile pairs beat rules and where (by regime), the masking drop-off, grounding causal effect, contract reliability, the safety audit result, the Tier 2 status, and one sentence on what the application can legitimately say about an incident. Follow with the non-goals list (§3.3).

## O-4 Experiment log and validation record — MUST

`EXPERIMENT_LOG.md` (append-only: stage times, gate dispositions, skipped SHOULDs with reasons, tier deferrals, any pre-freeze adjustment with timestamp relative to the freeze tag) and `VALIDATION.md` in the Lab 01 house format.

---

# 10. Tests added (amends §27) — MUST

- XE session DDL applies; `MAX_DISPATCH_LATENCY` and predicate are as frozen; a fixture login failure (18456) and a severity-20 `RAISERROR` both appear in `logwarden_capture` under the app-name predicate.
- ERRORLOG file-tail key is stable across `sp_cycle_errorlog` and container restart; zero duplicates, zero loss.
- Snapshot resolution: canonical-argument match, deterministic miss result, `snapshot_miss` flag; masked cell resolves placeholder arguments.
- Episode windows do not overlap across templates; the schedule builder rejects schedules that violate the gap rule.
- JSON transport parser: fence strip and leading-text strip are the only repairs; anything else is a contract failure; tool-result turn shape is byte-identical across profiles.
- Operating contract is in the first user turn and byte-identical across profiles (hash test); no system message is sent.
- Derived `A-router` composition reproduces B1 on resolved episodes and `A-tools` on the rest, including summed costs.
- B1 templates are hashed before capture; the audit fails if a template string appears verbatim in any test packet's synthetic message authored after the templates.
- Paired bootstrap and permutation drivers are deterministic under seed; the power simulation runs on dev rows and writes its result to the freeze record.
- Full-text: `IsFullTextInstalled = 1` in the custom image; population wait; `CONTAINS` query builder handles quotes, parentheses, and numbers; BM25 fallback produces identical result shape.
- Query Store settings persisted; `sp_query_store_flush_db` precedes materialization; named query classes resolve by comment prefix.
- Permission matrix (Tier 1 two-principal version) passes the negative tests in §D-6; the safety auditor detects each planted violation.
- `repro.sh --mode rows` passes with Tier 2 features flagged off; `repro.sh --mode restore` restores the `.bak` and regenerates `SCORECARD.md`.

---

# 11. Decisions for Karl before the freeze

Defaults let the agent proceed if unanswered.

1. **Accept the three-tier scope with Tier 1 closeout before Tier 2.** Default: **yes**.
2. **Structured-JSON transport for all four models in the primary campaign; native tool calling as a Tier 2 ablation.** Default: **yes**.
3. **Unconstrained JSON as primary; guided decoding as a Tier 2 cell.** Default: **yes** (flip it if you care more about the shipped-agent number than about measuring contract reliability).
4. **Catalog target ≥ 600 episodes with programmatic variants.** Default: **yes**.
5. **`A-router` derived by composition in replay.** Default: **yes**; `A-router-prompted` only if you want a distinct router prompt.
6. **Correlation claims.** Default: Tier 1 reports harness correlation only; `A-raw-events` on Regime M for two profiles in Tier 2.
7. **Full-text via custom image build.** Default: **build it**; BM25 fallback only if the nested runtime cannot build images.
8. **ERRORLOG access by file tail (no `securityadmin`).** Default: **file tail**.
9. **Two principals in Tier 1.** Default: **yes**; seven-way split in Tier 2.
10. **Temporal tables, telemetry spans, columnstore, JSON type deferred.** Default: **yes** (Tier 2/3).
11. **Control subset size 96 episodes.** Default: **96**; raise to 144 if per-family CIs on the masking control are too wide at dev tier.
12. **Muse `max_tokens` override if reasoning truncates.** Default: allow a per-profile override to 1400, recorded in the decode hash and freeze record.

---

# 12. Updated paste-line for the coding and research agent

> Read `aidataapps_logwarden_lab_3_spec.md`, then `aidataapps_logwarden_lab_3_spec_addendum.md`; the addendum governs on conflict. Create `aidataapps/logwarden/` on branch `aidataapps-logwarden` from `aidataapps-modelprint`; treat Labs 01–02 and `interpretability/` as read-only.
>
> Build Tier 1 (addendum §T-1) first and completely: both databases with the Tier 1 schema subset and two principals; the custom Extended Events session with 1–3 s dispatch latency filtered by application name; the XE offset-cursor ingester and ERRORLOG file tail; ≥ 600 injected, verified, cleaned-up episodes captured once; packets with protected ground truth, group-disjoint roles, leakage audit, and a `.bak` checkpoint at freeze; runbooks with exact vector, full-text (custom image) or governed BM25, and hybrid retrieval; the SQL lease queue, state machine, structured-JSON transport for all models with the operating contract in the first user turn, validation layers, policy engine, and the read-only tool procedures with argument-keyed frozen snapshots in replay; port gates; `qwen-smoke` end to end; replay of `A-direct` and `A-tools` for all four profiles with 16 workers, `A-rag` on the covered subset, `A-router` derived by composition, baselines B0–B3; the Tier 1 controls; paired grouped bootstrap and permutation scoring; taxonomy and claims; the Tier 1 reports; and `repro.sh` in row-only and restore modes.
>
> Close Tier 1 — `repro.sh --mode rows` passing on a fresh checkout, safety audit clean, every job terminal, deferrals logged — before starting any Tier 2 item (live parity and storm, telemetry and Query Store reporting, temporal tables, seven principals and SQL Audit, `A-raw-events`, transport and guided-decoding cells, remaining controls, SQL-native comparators, ANN, BACPAC). Never leave the branch in a state where the Tier 1 state of record cannot be regenerated.
>
> Rules are a serious competitor and must be given message templates and recurrence features. Replay is the scientific plane; live is the systems plane; never mix their scores. A clean null is a valid result; an unfinished lab is not.

---

# 13. References to verify at time of use (method inputs, not evidence)

- Microsoft Learn: `CREATE EVENT SESSION` (`MAX_DISPATCH_LATENCY`, `EVENT_RETENTION_MODE`, `event_file` options), Extended Events targets, `sys.fn_xe_file_target_read_file` (cursor columns, `VIEW SERVER STATE`), `system_health` session contents, `error_reported` and `xml_deadlock_report` events, `blocked_process_report` and the `blocked process threshold` option, `RAISERROR` (severity 19–25 permissions and `WITH LOG`), `sp_addmessage`, `sp_readerrorlog`/`xp_readerrorlog` permissions, `sp_cycle_errorlog`, full-text search on Linux (`mssql-server-fts`), `CONTAINSTABLE`/`FREETEXTTABLE`, `FULLTEXTCATALOGPROPERTY`, Query Store options (`INTERVAL_LENGTH_MINUTES`, `DATA_FLUSH_INTERVAL_SECONDS`, `QUERY_CAPTURE_MODE`, `sp_query_store_flush_db`), `sys.dm_db_log_stats`, system-versioned temporal tables and retention, nonclustered columnstore column restrictions, SQL Server Audit, `CREATE VECTOR INDEX`/`VECTOR_SEARCH` (as vetted in the Lab 02 addendum), DacFx/BACPAC limitations for vector indexes.
- vLLM documentation: structured outputs (`response_format` with `json_schema`, `guided_json`), tool-call parsers, `stream_options.include_usage`, `/metrics` names as exposed by the pinned image, `--generation-config`, `--max-num-seqs`.
- Lab 02 addendum: decode hygiene, digest-resolved registry, permutation nulls, exact-vs-ANN discipline, Colab checkpoint plan.

---

*End of addendum.*
