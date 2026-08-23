# Lab 04 - GhostType

## SQL language completion, semantic validation, and inference-serving systems laboratory

**Status:** Governing replacement specification for Lab 04. This document replaces the attached GhostType draft as the implementation and experiment contract.

**Proposed repository location:** `aidataapps/ghosttype/`

**Proposed branch:** `aidataapps-ghosttype`

**Reviewed context:**

- `karlb-dev/labs`, branch `aidataapps-modelprint`, reviewed at branch head `92d35973863baf14680082bedd6dae52e949d675`.
- `karlb-dev/labs`, branch `aidataapps-logwarden`, reviewed at branch head `c228339a20de91a4d020abb5257d28e04e36f7a1`.
- Attached GhostType draft, dataset notes, dataset archive, and supplemental notes.
- Rebuilt GhostType dataset package supplied with this specification.

**Dataset replacement:** The rebuilt package contains **588 completion records**. It retains all **265 legacy records by stable ID** and adds **323 synthetic records across six new catalogs**. The static packaging pass reports 588/588 JSON-Schema-valid rows, 588/588 gold-contract-valid rows, and 265/265 legacy-ID preservation. ScriptDom parsing, catalog binding, compilation, and fixture execution remain required runtime gates before the dataset can be frozen as scientific input.

---

> ## Paste-line for the coding and research agent
>
> Create a new branch and worktree for `aidataapps-ghosttype`. Read this specification completely, then inspect the current `aidataapps/modelprint` and `aidataapps/logwarden` implementations, their governing specifications and addenda, database migrations, preregistrations, experiment logs, model registry, run archival code, and report builders. Reuse their hardened campaign, raw-response, migration, capability, evidence-event, telemetry, report, BACPAC, Drive-mirroring, and one-model-resident-at-a-time conventions where they remain appropriate. Do not modify or import writable state from either prior lab.
>
> Import the rebuilt GhostType dataset package as an immutable source. Preserve the 265 legacy IDs and source hashes. Implement all CPU foundation stages, dataset validators, SQL fixture creation, metadata snapshots, ScriptDom service, deterministic baselines, completion request lifecycle, cancellation, SQL persistence, reporting skeleton, and single-user simulator before loading a scientific target model. Run the CPU development campaign first. Freeze the GPU quality and serving campaigns only after every dataset, oracle, port, safety, migration, replay, and batch-invariance gate is satisfied.
>
> The primary quality experiment is capture-and-replay: every model and architecture arm receives byte-identical completion requests, metadata snapshots, retrieval corpora, and validation policy. The primary serving experiment is live and separate: virtual users produce open-loop and closed-loop traffic under frozen workload traces while queueing, batching, cancellation, cache affinity, throughput, and tail latency are measured. Never use live serving responses to replace missing rows in the frozen quality matrix.
>
> Store raw requests and raw model responses before parsing. Persist every candidate, retrieval witness, parser result, catalog binding, compilation or execution result, ranking decision, simulated acceptance event, queue transition, model-service metric, SQL resource metric, and report metric in SQL Server. Generate all reports from retained SQL rows and hashed artifacts. Missing cells remain missing and receive an explicit disposition. Do not fabricate, impute, or silently regenerate scientific rows.

---

# Part I. Executive verdict

## 0. What the original direction gets right

The draft starts from a strong product-shaped question: can local language models provide useful SQL completions, and what happens when the same completion engine is placed under realistic interactive load?

That is an excellent fourth lab because it combines the lessons of the first three:

1. Lab 01 established SQL-backed retrieval, grounding, and a narrow action boundary.
2. Lab 02 established frozen model profiles, large repeatable generation campaigns, exact versus approximate vector evidence, calibration, service manifests, and scientific report reconstruction.
3. Lab 03 established SQL Server as an operational control plane, capture-once/replay-many experiments, agent-step telemetry, safety gates, and separation between model-quality and live-systems claims.
4. Lab 04 can now make the application itself interactive. It must predict code at a cursor while the user is still typing, decide when not to speak, respect the visible database catalog, remain syntactically insertable, and stay responsive when many sessions arrive together.

The original dataset is also useful. It contains a broad developer-oriented collection of SQL requests and expected continuations. The problem is not that it is bad. The problem is that a flat completion file is not yet enough to support the claims the lab wants to make.

## 1. The necessary reframing

GhostType is not primarily a text-generation demo. It is a **constrained decision and serving system**.

At request time the system observes:

```text
prefix before the cursor
suffix after the cursor
cursor and selection offsets
SQL dialect and compatibility level
connection and session settings
visible catalog snapshot
open-document context
accepted completion history, when the experiment permits it
latency class and cancellation deadline
```

It must return:

```text
zero or more insertion candidates
an explicit insertion range
candidate confidence and abstention state
provenance for retrieved context
validation evidence
latency and resource evidence
```

A candidate can fail in several independent ways:

- it can be malformed text for the editor insertion range;
- it can be invalid T-SQL;
- it can reference an object the user cannot see;
- it can compile but return the wrong shape;
- it can execute but mutate the wrong rows;
- it can be semantically correct but too slow to arrive;
- it can be useful offline but collapse under batching or queueing;
- it can be correct but unsafe to offer automatically;
- it can be worse than deterministic catalog completion on the same case.

The lab therefore measures the complete path rather than treating exact string match as truth.

## 2. The central experimental split

GhostType contains two linked but non-interchangeable planes.

### 2.1 Frozen completion-quality plane

Every model and architecture arm receives the same immutable request packets. A request packet includes the SQL document, cursor, suffix, session settings, allowed metadata snapshot, retrieval corpus version, and oracle contract. The quality plane measures prediction, grounding, abstention, parser behavior, catalog fidelity, semantic validity, and simulated acceptance.

No live scheduler is allowed to alter which rows are evaluated. Batch-sensitive output is measured, but it cannot silently replace the frozen sequential reference.

### 2.2 Live serving-performance plane

Virtual users issue requests under frozen arrival traces. This plane measures context construction, SQL retrieval, queueing, batching, time to first token, time to candidate, cancellation, wasted decode, prefix-cache affinity, throughput, fairness, resource utilization, and SLO attainment.

Serving runs may use a frozen quality subset so every systems configuration has a known answer key. They do not create new headline quality rows.

This split is the most important correction to the draft. It prevents a faster scheduler from looking more accurate merely because it completed a different sample, and it prevents a high-quality model from looking production-ready merely because it was tested one request at a time.

## 3. What the completed lab builds

The completed lab is an application and an experiment framework:

```text
Editor simulator / REST client
        |
        v
Completion gateway
  - request validation
  - cursor normalization
  - deadline and cancellation
  - policy/router
        |
        +--> deterministic parser/catalog engines
        +--> SQL full-text/vector retrieval
        +--> local LLM generation
        +--> optional candidate reranker
        |
        v
Validation pipeline
  - insertion integrity
  - ScriptDom parse/token/AST checks
  - catalog binding
  - compile/describe checks
  - sandbox execution or mutation oracle where eligible
  - safety policy
        |
        v
Candidate ranking or abstention
        |
        v
SQL Server system of record
  - datasets, fixtures, metadata snapshots
  - sessions, documents, requests, candidates
  - model and tool evidence
  - telemetry, evaluation, reports, artifacts
```

The application exposes a normal completion endpoint, a replay endpoint, a load-test endpoint, an evaluation browser, and SQL-generated reports.

## 4. Likely honest outcomes

The lab does not require an LLM victory. Several outcomes are useful:

- `DETERMINISTIC_FAST_PATH_DOMINANT`: parser and catalog completion win common token-level cases.
- `LLM_LIFT_MULTITOKEN`: an LLM adds value on joins, predicates, projections, windows, CTEs, or statement continuation.
- `RETRIEVAL_LIFT_SCHEMA`: SQL-backed catalog and exemplar retrieval improves object fidelity.
- `VALIDATION_LIFT`: parser, binder, or execution-aware reranking improves semantic success.
- `SELECTIVE_COMPLETION_USEFUL`: calibrated abstention makes offered completions materially more reliable.
- `QUALITY_GOOD_LATENCY_MISSED`: output quality is useful but the target serving SLO is not met.
- `ROUTER_DOMINANT`: a deterministic-fast-path plus selective-LLM router is the best production-shaped design.
- `MODEL_DIFFERENCES_ERASED_BY_SYSTEM`: model gaps shrink after retrieval and validation.
- `CLEAN_NULL`: none of the local models beats the strong baselines on the frozen hard set.

All are valid if reconstructed from retained evidence.

---

# Part II. Research questions, hypotheses, and claim ceiling

## 5. Primary questions

GhostType must answer these questions in order:

1. What does a strong non-LLM SQL completion baseline achieve on the frozen corpus?
2. On which completion families do local LLMs add measurable value?
3. Does suffix-aware completion outperform prefix-only prompting for mid-statement edits?
4. Does permission-filtered schema context reduce object and column hallucination?
5. Which SQL retrieval mode supplies the needed schema evidence with the lowest token and latency cost?
6. Do parser, binder, compile, and execution-aware validators improve final decisions?
7. Can a calibrated policy abstain on sparse or ambiguous catalog contexts without destroying coverage?
8. How much do results vary across Muse Glimmer, Gemma, OLMo, and Qwen under identical packets?
9. Which failures arise in the model, retrieval, parser, catalog snapshot, validator, ranker, or scheduler?
10. What latency and throughput can each model sustain under realistic interactive traffic?
11. Which inference settings improve throughput without changing governed outputs or harming tail latency?
12. Does schema/session affinity make prefix caching materially useful?
13. How much work is discarded by cancellation when users continue typing?
14. What router lies on the best quality-latency-resource Pareto frontier?
15. What remains unexplained after all controls and ablations?

## 6. Frozen hypotheses

These are proposals until copied into `GHOSTTYPE_PREREGISTRATION.md` and frozen before opening test labels.

| ID | Hypothesis | Primary falsifier |
|---|---|---|
| H1 | Deterministic parser/catalog completion dominates simple keyword and identifier cases. | One or more LLM arms reliably exceed it at equal or lower latency without hidden context. |
| H2 | LLM value concentrates in multi-token semantic continuations. | No held-out lift on joins, predicates, projections, windows, CTEs, or statement blocks. |
| H3 | Suffix-aware prompting improves cursor-in-middle cases. | Prefix-only is equal or better on the frozen suffix holdout. |
| H4 | Permission-filtered catalog context reduces nonexistent-object and wrong-column references. | Catalog grounding does not reduce hallucination or merely copies irrelevant metadata. |
| H5 | Hybrid lexical/vector retrieval beats vector-only retrieval for mixed identifiers and natural-language intent. | Vector-only or deterministic lexical lookup matches it across recall, latency, and final quality. |
| H6 | Validation-aware reranking improves semantic success beyond raw top-1 generation. | Validators reject correct answers, admit wrong answers, or add no net lift. |
| H7 | Calibrated abstention supports a useful quality/coverage tradeoff. | Reliability gates fail or useful coverage collapses. |
| H8 | Model differences shrink after the same retrieval and validation stack. | The ranking remains wide and stable across all arms. |
| H9 | A CPU development profile can validate correctness and lifecycle plumbing but not establish hosted serving performance. | CPU and GPU profiles prove equivalent under all frozen service metrics, an unlikely but testable result. |
| H10 | Continuous batching increases throughput while creating a measurable tail-latency frontier. | It offers no throughput lift or no tail tradeoff at the tested load. |
| H11 | Prefix caching helps repeated schema/session prefixes more than randomized-schema traffic. | Affinity and randomized workloads perform equivalently within uncertainty. |
| H12 | Request cancellation is a material systems variable. | Decode waste and tail latency are negligible with and without cancellation. |
| H13 | Context has diminishing and eventually negative returns. | More catalog and history context improves every family monotonically. |
| H14 | A router using deterministic fast paths and a selective LLM is the strongest application design. | A single engine dominates both quality and latency across completion classes. |
| H15 | Batch-invariant or explicitly batch-classified output is necessary for trustworthy quality replay. | No target model exhibits any sequential/batched difference under governed settings. |

## 7. Claim ceiling

Even a successful run licenses only bounded claims.

Allowed:

- profile M produced completion-quality metric X on dataset version D under request template T, catalog policy C, and serving configuration S;
- architecture A improved semantic success over baseline B on the frozen hard set;
- retrieval mode R improved required-object recall or final completion quality;
- a serving configuration sustained a measured workload on the recorded hardware;
- a selective policy achieved a measured quality/coverage point;
- an execution-eligible completion satisfied the declared fixture oracle.

Not allowed:

- universal SQL coding ability;
- human productivity gains without a human study;
- correctness on arbitrary production databases;
- security guarantees beyond tested policies;
- equivalence between a model family and one pinned served profile;
- treating exact match, parse success, or execution success alone as semantic correctness;
- treating simulated acceptance as observed user acceptance;
- extrapolating one-GPU throughput to another deployment;
- declaring ANN or SQL-native AI functionality used without runtime and plan evidence.

## 8. Evidence tags

Every claim and metric carries one of these tags:

- `DATA`: dataset structure, provenance, split, or oracle evidence.
- `OFFLINE`: frozen quality replay.
- `SEMANTIC`: binding, result-shape, result-set, or mutation-oracle evidence.
- `RETRIEVAL`: catalog or exemplar retrieval evidence.
- `SERVING`: latency, throughput, queueing, batching, or resource evidence.
- `CALIBRATION`: probability, coverage, or abstention evidence.
- `SAFETY`: permission, destructive-SQL, isolation, or prompt-injection evidence.
- `SYSTEM`: end-to-end application behavior.
- `DEV`: development-only result not eligible for headline claims.

---

# Part III. Lessons inherited from Labs 02 and 03

## 9. Required inherited engineering conventions

GhostType must carry forward these proven patterns:

1. **Exact model identity.** Store model repository, 40-character revision, tokenizer and chat-template hash, serving image digest, engine arguments, context length, quantization, parser policy, and effective decoding settings.
2. **One resident large model at a time.** Generate and score one target profile, checkpoint, back up, and evict only after hash verification.
3. **Raw-first persistence.** Append the raw HTTP response before any parser or database-derived text view.
4. **Immutable job keys.** A job key binds dataset case, variant, profile, arm, decode config, context policy, and campaign hash.
5. **Resume by state, not by file position guesswork.** Jobs and attempts remain explicit rows.
6. **Migration hashes.** Never rewrite an applied migration. Historical drift requires an explicit, evidence-bearing exception.
7. **Capability snapshots.** Detect actual SQL and serving capabilities at runtime.
8. **Plan proof for ANN.** Requested ANN is not actual ANN until a captured execution plan names the vector index.
9. **Batch invariance gates.** Sequential and batched canaries are compared before scientific generation.
10. **Unicode-safe token accounting.** Never infer assistant token spans by separately tokenizing a prefix and assuming token boundaries compose.
11. **Empty-final and truncation honesty.** Reasoning-only or length-truncated responses remain data-quality rows; no score is invented.
12. **Effective-value recording.** Requested engine settings and effective server settings are both retained.
13. **Capture once, replay many.** Quality comparisons use immutable packets and snapshots.
14. **SQL-backed metric and claim rows.** Reports reconstruct from retained rows rather than handwritten summaries.
15. **BACPAC/backup/Drive verification.** A residency is not complete until recovery artifacts match their recorded hashes.

## 10. New Lab 04-specific gates derived from those lessons

GhostType adds these gates before any test campaign:

- UTF-16 editor position round-trip across ASCII, Unicode, emoji, combining marks, and bracketed identifiers.
- prefix-plus-candidate-plus-suffix token-boundary tests for every model tokenizer.
- candidate insertion must not duplicate or consume suffix text accidentally.
- deterministic completion arms must remain deterministic under concurrency.
- model sequential/batched differences must be either eliminated by a governed runtime mode or classified and evaluated as separate profiles.
- cancellation must leave no orphan queue lease or duplicate final decision.
- schema snapshots must be immutable and permission-filtered.
- test cases and their gold completions must never enter retrieval corpora.
- fixture execution must occur in disposable databases or rollback-safe workers.
- parser, binder, compiler, and execution oracles must report `not_applicable` or `unavailable`, never silently pass.

---

# Part IV. Product behavior and completion contract

## 11. Completion classes

GhostType supports four latency and output classes. They are evaluated separately.

### 11.1 Token completion

Examples:

- keyword continuation;
- schema, table, column, alias, function, or parameter name;
- closing delimiter or short clause token.

This is the natural domain of deterministic catalog and parser baselines. The LLM is not invoked by default.

### 11.2 Line completion

Examples:

- projection list item;
- predicate;
- join condition;
- grouping or ordering expression;
- parameter declaration;
- single DDL property.

### 11.3 Block completion

Examples:

- multi-line SELECT body;
- CTE;
- windowed aggregate;
- MERGE alternative implemented as explicit DML sequence;
- stored procedure body fragment;
- vector, temporal, JSON, or operational query fragment.

### 11.4 Statement completion

The model completes one bounded statement from a natural-language comment, partial statement, or surrounding script. This class receives a longer latency budget and stricter safety policy.

The primary scientific claim concerns line, block, and statement completions. Token completion is a mandatory baseline and router component.

## 12. Request envelope

The API accepts a versioned request:

```json
{
  "schemaVersion": 1,
  "requestId": "uuid",
  "userId": "sim-user-017",
  "sessionId": "session-uuid",
  "documentId": "document-uuid",
  "documentVersion": 42,
  "language": "sql",
  "dialect": "tsql",
  "completionClass": "line",
  "prefix": "SELECT c.CustomerID, ",
  "suffix": "\nFROM Sales.Customers AS c;",
  "cursor": {
    "line": 0,
    "utf16Character": 21,
    "utf8ByteOffset": 21
  },
  "selection": null,
  "connection": {
    "workloadDatabaseId": "commerce-v3",
    "metadataSnapshotId": "snapshot-hash",
    "compatibilityLevel": 170,
    "sessionSettingsId": "default-ansi-v1",
    "permissionProfileId": "analyst-east"
  },
  "contextPolicyId": "schema-hybrid-history-off-v1",
  "enginePolicyId": "router-v1",
  "deadlineMs": 900,
  "candidateCount": 3,
  "traceparent": "..."
}
```

The service rejects stale document versions, invalid cursor offsets, impossible insertion ranges, unknown snapshots, or a deadline below the configured minimum.

## 13. Response envelope

```json
{
  "schemaVersion": 1,
  "requestId": "uuid",
  "decision": "offer",
  "selectedEngine": "rag-llm-validator-router-v1",
  "candidates": [
    {
      "candidateId": "uuid",
      "insertText": "c.EmailAddress",
      "replaceRange": {
        "start": {"line": 0, "utf16Character": 21},
        "end": {"line": 0, "utf16Character": 21}
      },
      "score": 0.84,
      "confidence": 0.78,
      "validation": {
        "parse": "pass",
        "binding": "pass",
        "compile": "not_run",
        "execution": "not_applicable",
        "safety": "pass"
      },
      "evidenceIds": ["retrieval-row-...", "catalog-binding-..."]
    }
  ],
  "abstentionReason": null,
  "timing": {
    "contextMs": 8.2,
    "retrievalMs": 13.7,
    "queueMs": 4.1,
    "modelMs": 118.3,
    "validationMs": 7.9,
    "totalMs": 154.4
  }
}
```

No markdown fence, explanation, or prose may appear in `insertText`. Explanations, if generated for analysis, are stored as a separate non-user-facing artifact and never used as the insertion.

## 14. Deadline and cancellation contract

Interactive completion is cancellation-heavy. The service must support:

- cancellation when a newer document version supersedes the request;
- cancellation when the user types beyond the candidate anchor;
- cancellation when the deadline expires;
- cancellation propagation through queue, model request, validator, and SQL write;
- idempotent late-response disposal;
- separate accounting for generated-but-discarded tokens.

A cancelled request remains a row with its final queue and model disposition. It cannot later become an offered completion.

---

# Part V. Dataset review and replacement

## 15. Review of the attached dataset

The attached dataset has a valuable base:

- varied SQL developer tasks;
- readable expected continuations;
- meaningful SQL Server flavor rather than generic SQL only;
- synthetic records that are safe to distribute;
- enough breadth to exercise more than one completion type.

The original format nevertheless leaves several research hazards:

1. It often behaves like prompt-to-answer evaluation rather than cursor insertion.
2. Prefix and suffix are not always first-class independent fields.
3. Equivalent variants are not guaranteed to share an indivisible split group.
4. Schema context and user-visible catalog scope are not always frozen.
5. Exact text is sometimes the only oracle even when multiple SQL forms are valid.
6. Execution eligibility, expected result shape, mutation state, and safety are not uniformly declared.
7. It is developer-heavy and underweights ordinary continuation, partial identifier, and incomplete-clause cases.
8. It needs more sparse-catalog and honest-abstention pressure.
9. It needs more adversarial Unicode, reserved-name, quoted-identifier, suffix, and cursor-offset cases.
10. It needs explicit source provenance, generator revision, record hash, dataset version, and legacy identity.
11. It needs session and workload manifests so serving experiments can be reproduced.
12. It needs a formal distinction between data that may enter retrieval and data reserved for evaluation.

## 16. Rebuilt dataset package

The replacement package is versioned independently from the lab code.

```text
ghosttype_dataset_v2/
  README.md
  CHANGELOG.md
  LICENSES.md
  schemas/
    completion-record.schema.json
    dataset-manifest.schema.json
    fixture-manifest.schema.json
    workload-manifest.schema.json
  catalogs/
    ... legacy catalogs ...
    ... six expansion catalogs ...
  fixtures/
    manifests/
    ddl/
    seeds/
    expected/
  generators/
    ... deterministic generator sources ...
  manifests/
    dataset-manifest.json
    record-hashes.json
    split-groups.json
    legacy-preservation.json
    catalog-counts.json
  validation/
    static-validation.json
    gold-contract-validation.json
    runtime-validation.template.json
```

Authoritative package totals:

```text
legacy records retained: 265
new synthetic records:    323
all records:              588
new catalog families:       6
```

The package supplied beside this specification is the source artifact. The implementation must not regenerate it during a scientific run. Generators exist for audit and future versions only.

## 17. Six expansion catalog themes

The 323 new records are organized around six gaps.

### 17.1 Cursor and suffix completion

Includes:

- cursor between tokens;
- cursor inside SELECT lists, JOINs, predicates, CTEs, windows, DDL, and procedural blocks;
- suffix that constrains aliases, parentheses, grouping, ordering, and terminators;
- partial identifiers;
- partial keywords;
- replacement ranges rather than insertion-only cases;
- suffix duplication traps;
- comments and string literals near the cursor;
- Unicode offset cases.

### 17.2 Schema honesty and sparse catalogs

Includes:

- missing requested table;
- two plausible columns, only one visible;
- permission-filtered objects;
- synonyms and ambiguous unqualified names;
- alias shadowing;
- sparse schema where abstention is correct;
- stale metadata snapshot;
- cross-database reference forbidden by policy;
- misleading natural-language comments;
- schema descriptions containing prompt-like text that must remain untrusted.

### 17.3 SQL Server AI, vector, JSON, and search features

Includes capability-gated examples for:

- vector data types and distance expressions;
- exact vector search;
- optional approximate-vector syntax represented as build-specific cases;
- embedding generation or external embedding integration;
- SQL-native chunking comparators;
- hybrid full-text and vector retrieval;
- JSON extraction and construction;
- safe fallback when a preview feature is unavailable;
- dimension and type mismatches;
- metadata inspection for vector columns and indexes.

These records must be routed by runtime capability. A feature unavailable on the pinned build is not a failed model completion; it is an ineligible case with an explicit disposition.

### 17.4 Temporal, history, and change-aware SQL

Includes:

- system-versioned temporal queries;
- `FOR SYSTEM_TIME` variants;
- history-table inspection;
- temporal DDL completion;
- ledger/change-tracking/change-data patterns where supported by the fixture;
- point-in-time predicates;
- valid-time versus transaction-time confusion controls;
- temporal joins and result-shape oracles.

### 17.5 Azure and operational SQL

Includes safe, non-destructive operational patterns:

- Query Store inspection;
- wait and session diagnostics;
- index and plan analysis;
- database-scoped configuration inspection;
- Azure SQL metadata and resource observations;
- elastic or serverless context represented only when the fixture/capability supports it;
- retry, timeout, and transient-error query patterns;
- permission-aware alternatives;
- explicit abstention on unsupported server-level operations.

### 17.6 Identifier, dialect, and parser edges

Includes:

- bracketed reserved words;
- spaces, punctuation, non-ASCII, and emoji in identifiers;
- case-sensitive collations;
- quoted identifier settings;
- temp tables, table variables, CTE names, aliases, and parameters;
- multi-part names;
- comments, nested comments, and strings containing SQL-looking text;
- `GO` batch boundaries;
- incomplete procedural blocks;
- compatibility-level differences;
- invalid-but-recoverable editor states.

## 18. Completion record contract

The exact JSON Schema in the package governs. Conceptually each row binds these objects:

```text
identity
  schema version
  stable case ID
  legacy ID, when applicable
  source catalog, generator, and hashes

split and provenance
  split group ID
  data role
  source/license status
  template family and near-duplicate family

task
  dialect and compatibility level
  completion class and mode
  difficulty and capability requirements
  cursor, selection, prefix, and suffix

visible environment
  fixture and metadata snapshot
  session settings
  permission profile
  allowed context channels

gold contract
  one or more accepted insertions
  canonical insertion for exact metrics
  must-reference and must-not-reference objects
  expected result shape or mutation state
  abstention allowance

oracle contract
  insertion integrity
  parse/token/AST eligibility
  catalog binding eligibility
  compile/describe eligibility
  sandbox execution eligibility
  safety class

tags and diagnostics
  feature family
  leakage group
  known ambiguity
  expected failure or abstention reason
```

## 19. Split contract

The independent unit is not a row. It is the largest linked group among:

- shared generator template;
- same underlying SQL intent;
- same fixture and target object set;
- prefix/suffix perturbations of one document;
- natural-language paraphrases;
- formatting and identifier-quoting variants;
- easy/hard variants with the same answer;
- accepted-completion history derived from one case.

A split group appears in exactly one role:

- `train`: fit rerankers, calibration features, retrieval exemplars, and learned policies;
- `calibration`: probability, abstention, and conformal thresholds only;
- `test_id`: primary held-out quality evaluation;
- `test_template_holdout`: unseen generator or intent templates;
- `test_schema_holdout`: unseen fixture schemas with known task families;
- `test_suffix_holdout`: cursor-in-middle and suffix stress;
- `test_sparse_catalog`: abstention and hallucination stress;
- `test_capability`: SQL-build-dependent features, reported separately.

No test gold text, normalized gold fragment, AST, result rows, or explanation may enter an embedding, phrase, template, or accepted-history index.

## 20. Static and runtime validation gates

The packaging pass can establish only structural facts. Before freeze, run:

1. JSON Schema validation for every row.
2. Stable ID, duplicate ID, content hash, and legacy preservation audit.
3. UTF-16 and UTF-8 cursor round-trip.
4. Prefix + gold + suffix insertion integrity.
5. ScriptDom tokenization and parse checks for parse-eligible rows.
6. Catalog binding against the exact permission-filtered snapshot.
7. `sp_describe_first_result_set` or equivalent result-shape checks where eligible.
8. compile/showplan checks in disposable fixtures.
9. rollback-safe execution and result-set comparison for read-only cases.
10. disposable-database state comparison for DDL/DML cases.
11. destructive-SQL and cross-database policy audit.
12. split leakage, exact duplicate, normalized duplicate, and near-duplicate audit.
13. retrieval exclusion audit.
14. capability routing against the pinned SQL build.
15. independent reconstruction of dataset counts and hashes.

Any failing row is fixed in a new dataset version or dispositioned before freeze. Scientific code never patches a gold row in place.

---

# Part VI. SQL Server as control plane, workload substrate, and analysis engine

## 21. Database boundary

Use two databases.

### `GhostTypeControl`

Owns immutable inputs, editor/session state, completion operations, model evidence, telemetry, evaluation, reports, and artifact manifests.

### `GhostTypeWorkloads`

Hosts disposable fixture databases or fixture schemas used for metadata snapshots, compilation, and execution oracles. Destructive and mutation tests never target `GhostTypeControl`.

The control-plane login has no permission to alter scientific source rows after freeze. The completion runtime uses least-privilege principals and cannot execute arbitrary model-authored SQL against the control database.

## 22. Required schemas

```text
control      campaigns, freezes, capabilities, profiles, configurations
dataset      sources, versions, cases, variants, gold contracts, fixtures
catalog      metadata snapshots, objects, columns, parameters, types, permissions
session      simulated users, documents, versions, keystrokes, cursor events
completion   requests, jobs, attempts, candidates, decisions, acceptance
retrieval    corpora, vectors, full-text rows, search runs, witnesses
validation   parser, binder, compile, execution, safety evidence
serving      service instances, queues, batches, cancellation, scheduler state
telemetry    traces, spans, model metrics, SQL metrics, host/GPU/CPU samples
eval         evaluation runs, row scores, metrics, comparisons, claims
reporting    views, materialized snapshots, report manifests
```

## 23. Core control tables

At minimum:

```text
control.schema_migrations
control.runs
control.campaigns
control.campaign_freezes
control.capability_snapshots
control.model_profiles
control.inference_profiles
control.embedding_profiles
control.engine_policies
control.context_policies
control.validation_policies
control.tuning_configs
control.evidence_events
control.run_artifacts
```

Every configuration row has a canonical JSON representation and SHA-256. Campaign freeze hashes all governed source and configuration artifacts plus the generated immutable job inventory.

## 24. Dataset and catalog tables

```text
dataset.dataset_versions
dataset.source_artifacts
dataset.catalogs
dataset.cases
dataset.case_variants
dataset.split_groups
dataset.gold_candidates
dataset.semantic_contracts
dataset.oracle_contracts
dataset.fixtures
dataset.fixture_snapshots
dataset.workload_manifests

catalog.snapshots
catalog.databases
catalog.schemas
catalog.objects
catalog.columns
catalog.parameters
catalog.indexes
catalog.foreign_keys
catalog.types
catalog.synonyms
catalog.permissions
catalog.descriptions
catalog.snapshot_artifacts
```

Catalog tables are denormalized enough for low-latency lookups but retain stable source identifiers. A permission profile generates a materialized visible-catalog view. The model never receives objects the simulated user is not allowed to see.

Each catalog text row may have:

- a full-text representation;
- a fixed embedding profile and vector;
- a compact token-budget representation;
- source hash and snapshot hash.

## 25. Session and document tables

```text
session.users
session.user_profiles
session.sessions
session.documents
session.document_versions
session.keystroke_events
session.cursor_events
session.connection_events
session.acceptance_history
session.open_document_context
```

`session.documents` or `session.document_versions` must be system-versioned so late or cancelled responses can be audited against the exact document state that produced them. If temporal support is unavailable, use an append-only version table and record that fallback.

Never overwrite document text for a scientific request. A request binds one immutable document version.

## 26. Completion operational tables

```text
completion.requests
completion.request_contexts
completion.request_schema_bindings
completion.jobs
completion.job_leases
completion.attempts
completion.raw_model_responses
completion.candidates
completion.candidate_features
completion.candidate_rankings
completion.final_decisions
completion.cancellations
completion.simulated_acceptance
completion.partial_acceptance_events
```

Required behavior:

- request idempotency key;
- durable lease with expiry and attempt number;
- `pending`, `leased`, `running`, `validating`, `complete`, `cancelled`, `failed`, `stopped` states;
- atomic final-decision write;
- no candidate offered after request cancellation or document supersession;
- raw response retained separately from parsed candidates;
- all parser failures preserved.

## 27. Retrieval tables

```text
retrieval.corpora
retrieval.corpus_members
retrieval.text_artifacts
retrieval.embeddings
retrieval.search_tables_*       -- immutable, denormalized, vector-index eligible
retrieval.search_runs
retrieval.search_results
retrieval.required_object_recall
retrieval.query_plan_evidence
retrieval.index_manifests
```

Build distinct corpora for:

- schema objects and descriptions;
- foreign-key and join relationships;
- train-only completion exemplars;
- simulated accepted history;
- SQL feature documentation snippets that are vendored and licensed;
- optional organization style guides.

Exact vector distance remains the reference. Approximate search is optional and is reported only when the capability doctor and captured plan prove its execution. Full-text and deterministic prefix search are first-class baselines, not fallback embarrassments.

## 28. Validation tables

```text
validation.runs
validation.insertion_checks
validation.scriptdom_results
validation.tokens
validation.ast_summaries
validation.binding_results
validation.compile_results
validation.result_shape_results
validation.execution_results
validation.result_set_hashes
validation.mutation_snapshots
validation.safety_results
validation.policy_violations
```

A final candidate score consumes explicit statuses from these rows. No missing validator is interpreted as pass.

## 29. Serving and telemetry tables

```text
serving.instances
serving.instance_profiles
serving.queue_events
serving.request_leases
serving.batch_events
serving.batch_members
serving.scheduler_samples
serving.cache_events
serving.cancellation_events
serving.load_runs
serving.virtual_users
serving.arrival_events

telemetry.traces
telemetry.spans
telemetry.span_events
telemetry.model_requests
telemetry.model_token_summaries
telemetry.vllm_metric_samples
telemetry.host_samples
telemetry.gpu_samples
telemetry.process_samples
telemetry.sql_query_samples
telemetry.query_store_runtime
telemetry.query_store_waits
telemetry.file_io_samples
telemetry.tempdb_samples
telemetry.network_samples
```

Avoid storing one row for every generated token in the primary campaign unless a bounded diagnostic sample explicitly requires it. Store token timing summaries and raw provider artifacts to avoid turning observability into the dominant workload.

## 30. Evaluation and reporting tables

```text
eval.runs
eval.row_scores
eval.candidate_scores
eval.acceptance_scores
eval.metric_results
eval.bootstrap_results
eval.permutation_results
eval.paired_comparisons
eval.calibration_bins
eval.selective_curves
eval.pareto_points
eval.taxonomies
eval.claims

reporting.report_snapshots
reporting.report_sections
reporting.figure_manifests
reporting.table_manifests
reporting.reconstruction_runs
```

Migration tests must catch schema-name drift and reserved-word aliases before a scientific run.

## 31. SQL Server features demonstrated

The lab should make SQL Server useful rather than decorative:

- relational constraints and idempotent migrations;
- temporal or append-only document/request history;
- Query Store for retrieval, persistence, and report-query performance;
- catalog views and permission-aware metadata snapshots;
- full-text search for lexical schema and exemplar retrieval;
- native vector columns and exact distance search;
- optional vector indexes, capability-gated and plan-proven;
- JSON validation and querying, with native JSON features used only when the build proves support;
- SQL-native chunking or embedding comparators when available, never assumed;
- stored procedures for durable queue leases and atomic finalization;
- window functions for latency distributions, rolling load, and fairness;
- columnstore experiments for completed high-volume telemetry facts;
- report views and reproducible metric extracts;
- backup, BACPAC, and reconstruction.

Every optional feature has a baseline path. A missing preview feature must not block the core lab.

---

# Part VII. Parser, metadata, retrieval, generation, and validation pipeline

## 32. ScriptDom service

Implement a small pinned .NET service using `Microsoft.SqlServer.TransactSql.ScriptDom`.

Responsibilities:

- tokenize incomplete T-SQL;
- identify whether the cursor is in code, comment, quoted identifier, or string literal;
- expose nearby AST and token context where recoverable;
- parse prefix + candidate + suffix;
- report parser errors and positions;
- normalize formatting only for metrics, never for user insertion;
- derive referenced identifiers and clause types;
- compare candidate AST with gold semantic constraints;
- provide a deterministic parser completion baseline for selected token classes.

The service must preserve SQL Server dialect/version selection. Parser version is a pinned profile and part of the campaign identity.

Partial SQL is expected. Parse failure of the raw document is not automatically a request failure. The candidate validator distinguishes pre-existing errors from candidate-introduced errors.

## 33. Metadata context builder

The context builder assembles a bounded packet from the immutable snapshot:

```text
current database and default schema
visible tables/views/functions/procedures/types
columns for referenced or retrieved objects
foreign-key relationships
local aliases, CTEs, temp tables, variables, and parameters
session settings and compatibility level
bounded accepted-history examples, when enabled
```

It must report:

- every source row used;
- token and character counts;
- required-object recall;
- irrelevant-object count;
- permission-filter result;
- truncation policy;
- construction latency.

## 34. Retrieval modes

Freeze these arms:

1. `none`: no external context.
2. `catalog-prefix`: deterministic prefix and alias-aware lookup.
3. `fulltext`: SQL full-text search over schema and train-only examples.
4. `vector-exact`: exact vector retrieval.
5. `hybrid-rrf`: lexical and vector results fused with a frozen method.
6. `hybrid-reranked`: hybrid candidates reranked with deterministic features or a small frozen ranker.
7. `accepted-history`: user-local accepted history, evaluated separately.

ANN is a serving optimization of a frozen retrieval method, not a new quality arm. It is compared against exact witnesses for recall and decision agreement.

## 35. Completion engine arms

### B0. Empty or prior baseline

Return no completion, or the most common completion-class prior. This establishes coverage and metric sanity.

### B1. Keyword/parser baseline

Uses token context and a finite T-SQL grammar/keyword table.

### B2. Catalog baseline

Uses local aliases and visible catalog names. It supports prefix matching, recent-object ordering, and deterministic tie handling.

### B3. Accepted-history n-gram/template baseline

Uses only train-role or simulated user-local accepted history. It never sees test gold.

### B4. Retrieval template baseline

Returns the best eligible train exemplar after identifier-safe adaptation. It must pass the same validators as LLM candidates.

### M0. Direct LLM

Prefix and suffix only, plus minimal output contract.

### M1. Schema-grounded LLM

Adds bounded metadata context.

### M2. Hybrid-RAG LLM

Adds hybrid schema and exemplar witnesses.

### M3. Multi-candidate LLM plus deterministic validator/ranker

Generates N candidates and reranks using parser, binding, safety, and retrieval features.

### M4. Model reranker extension

A separately frozen small reranker or target-model scoring pass ranks candidates. It cannot inspect test labels.

### R0. Production-shaped router

Routes token completion to deterministic engines, invokes the LLM only for eligible multi-token contexts, validates candidates, and abstains under uncertainty or deadline pressure.

The router is a primary reported arm, not a post-hoc demo.

## 36. Model-neutral prompt contract

The primary four-model comparison uses one user message, byte-identical except for model-required transport encoding. It includes:

```text
role and output contract
completion class
prefix
cursor marker represented outside SQL text where possible
suffix
bounded metadata/retrieval context
explicit instruction to return JSON with insertion text only
```

If a model does not support a system role, instructions are folded using the already governed profile behavior. The parsed `insertText` is stored separately from reasoning or wrapper text.

Native fill-in-the-middle is an optional capability arm. It runs only for a profile whose tokenizer, special tokens, server, and canaries demonstrate genuine FIM behavior. Prompt-simulated suffix conditioning remains the cross-model primary.

## 37. Candidate generation contract

Each governed request explicitly fixes:

- temperature;
- top-p, top-k, and min-p semantics;
- repetition, presence, and frequency penalties;
- seed behavior;
- max output tokens by completion class;
- candidate count;
- stop policy;
- response schema or guided decoding settings;
- reasoning policy;
- timeout and cancellation.

Requested and effective settings must agree or receive a recorded deviation before test generation.

## 38. Validation sequence

Apply validators in this order:

1. JSON/output-shape validation.
2. insertion-range and suffix-integrity validation.
3. empty, duplicate, overlong, markdown, and prose filters.
4. ScriptDom tokenization and parse delta.
5. identifier extraction and catalog binding.
6. permission and safety policy.
7. compile/describe check where eligible.
8. sandbox execution or mutation-state oracle where eligible.
9. result-shape or result-set comparison.
10. candidate feature assembly and ranking.

Early failure may skip expensive later stages, but every skipped stage records its reason.

## 39. Sandbox rules

- Read-only SELECT cases execute against a fixture snapshot with a hard timeout and row cap.
- DML cases run inside rollback-safe transactions only when rollback is sufficient.
- DDL and stateful cases run in disposable fixture databases restored from a frozen backup or recreated from a manifest.
- Server-level, security, destructive, external-access, or unsupported statements never execute. They receive static safety adjudication.
- Result comparisons use normalized typed rows and hashes, not display strings alone.
- Nondeterministic functions and order-insensitive results have explicit oracle policies.

---

# Part VIII. Model matrix and port gates

## 40. Target profiles

The default target matrix inherits the exact profiles used by Labs 01 and 02:

- `muse-glimmer-30b`
- `gemma-4-31b`
- `olmo-3.1-32b-instruct`
- `qwen-3.8-27b`

`qwen-smoke` is a plumbing and CPU-development control, not a peer target class. An optional code-specialized control may be added only before freeze, with a complete pinned profile and a separate claim scope.

## 41. CPU development profile

The user intends to run single-user simulation on CPU first. This is explicitly supported.

The CPU profile includes:

- all SQL Server databases and migrations;
- dataset import and validation;
- fixtures and metadata snapshots;
- ScriptDom service;
- deterministic baselines;
- retrieval and vector/full-text evaluation;
- request lifecycle, cancellation, temporal history, and reporting;
- a small quantized OpenAI-compatible local model or recorded provider stub for functional generation;
- one closed-loop virtual user;
- cold and warm latency traces.

CPU model results are tagged `DEV` unless the exact same model/profile/backend is independently frozen as a scientific target. The purpose is to finish the application and expose bugs before expensive GPU residency.

## 42. GPU Colab profile

The hosted campaign uses the established rootless Docker/Colab pattern:

- isolated Compose project and ports;
- SQL Server persistent volume;
- one target chat profile resident at a time;
- persistent embedding service where disk and memory allow;
- exact model revision and digest;
- pre-residency and post-residency capability and storage checks;
- Drive mirror with hash verification;
- pre-eviction SQL backup/BACPAC;
- cache eviction only after recovery evidence passes.

## 43. Port gate

Every profile must pass:

1. model, revision, tokenizer, template, image, and engine-argument identity;
2. API readiness and model-list identity;
3. requested/effective decoding parity;
4. code-only output canaries;
5. JSON/guided-output canaries if used;
6. Unicode and bracketed-identifier canaries;
7. prefix/suffix insertion canaries;
8. empty-final and reasoning-only behavior;
9. context-length and KV-cache capacity;
10. sequential repeatability at deterministic settings;
11. sequential versus governed batched output comparison;
12. cancellation behavior;
13. streaming chunk assembly;
14. token-count accounting;
15. optional native FIM gate.

A model can pass with `batch_sensitive=true` only if sequential and batched are frozen as distinct serving profiles. The primary quality matrix uses the declared reference profile consistently.

---

# Part IX. CPU single-user foundation experiment

## 44. Purpose

The CPU stage is not a miniature final leaderboard. It is the full application foundation under low load.

It must prove:

- the rebuilt dataset imports reproducibly;
- all static and runtime oracles execute;
- deterministic baselines are credible;
- request packets reconstruct byte-for-byte;
- SQL metadata respects permissions;
- the request queue and cancellation path are correct;
- telemetry joins across request, model, validator, SQL, and report layers;
- reports reconstruct from SQL;
- a small local model can traverse the entire path;
- no test leakage enters retrieval.

## 45. CPU workload

Run two modes.

### 45.1 Frozen sequential replay

- one request at a time;
- all 588 cases through deterministic baselines;
- smoke/dev subset through the small local model;
- cold and warm repetitions;
- no concurrent writer except telemetry;
- exact SQL retrieval only.

### 45.2 Closed-loop user simulation

A virtual user:

1. opens a frozen document;
2. types according to a recorded keystroke trace;
3. pauses at governed trigger points;
4. requests completion;
5. accepts, partially accepts, or rejects according to the frozen simulator;
6. continues typing and cancels stale requests;
7. changes schema/session according to the workload manifest.

The simulator is deterministic by seed and writes every event to SQL.

## 46. CPU deliverable gate

GPU work cannot begin until:

- full deterministic baseline evaluation is complete;
- dataset and oracle reports contain no unresolved required failures;
- request/cancellation integration tests pass;
- SQL backups restore and reconstruct;
- report skeletons rebuild from a clean database import;
- the CPU end-to-end smoke has zero silent drops or duplicate decisions;
- target model profiles and serving configs are frozen but not yet loaded.

---

# Part X. Frozen offline quality campaign

## 47. Packet creation

For each dataset case and allowed context ablation, create one immutable packet containing:

- dataset version and case hash;
- prefix, suffix, cursor, selection, and completion class;
- fixture and metadata snapshot hashes;
- permission and session profiles;
- context policy and exact retrieved witness IDs;
- model-neutral prompt bytes;
- validator policy;
- deadline class;
- gold and oracle references held separately from generation.

Retrieval witnesses are frozen before target generation so model order cannot affect context.

## 48. Core matrix

At minimum:

```text
588 cases
x 4 target models
x direct, schema-grounded, hybrid-RAG, validated-router arms
x deterministic reference decode
```

The multi-candidate validated arm may reuse one generation call with N candidates if the server contract is identical across models. Sampled decoding is a robustness suite, not the headline matrix.

## 49. Context ablations

Run a balanced subset under:

- prefix only;
- prefix + suffix;
- schema off;
- required schema only;
- full bounded schema;
- lexical retrieval;
- vector retrieval;
- hybrid retrieval;
- accepted history off/on;
- permission-filtered versus intentionally overbroad development control;
- comment/schema-description prompt-injection stress.

## 50. Candidate and row metrics

### 50.1 Text and token metrics

- exact insertion match;
- normalized token exact match;
- first-token and first-line accuracy;
- token edit distance;
- character edit distance;
- longest common prefix;
- suffix duplication or deletion rate;
- top-k exact and normalized success;
- mean reciprocal rank.

These are diagnostic, not sufficient for correctness.

### 50.2 Syntax and binding metrics

- candidate-introduced parse-error rate;
- parse recovery versus raw document;
- AST contract satisfaction;
- referenced-object precision and recall;
- referenced-column precision and recall;
- nonexistent identifier rate;
- invisible/unauthorized identifier rate;
- alias and scope correctness;
- required-object retrieval recall.

### 50.3 Semantic metrics

- compile/describe success;
- expected result-column names and types;
- result-set equivalence or task-specific comparator;
- DML affected-row and state equivalence;
- DDL catalog-state equivalence;
- temporal/vector/JSON feature-contract satisfaction;
- safety-policy compliance;
- semantic top-1 and top-k success.

### 50.4 Selective metrics

- offered coverage;
- selective semantic accuracy;
- risk-coverage curve;
- area under risk-coverage curve;
- abstention precision on ambiguous/sparse cases;
- ECE, NLL, and Brier score where probabilities are emitted;
- conformal coverage and set size, if implemented.

### 50.5 Simulated user metrics

- full acceptance rate;
- partial acceptance rate;
- accepted characters;
- keystrokes saved;
- edit operations after acceptance;
- harmful acceptance rate;
- acceptance by completion class and user profile.

These are simulator metrics, not human outcomes.

## 51. Simulated acceptance policy

Freeze the simulator before test evaluation.

A candidate may be accepted when:

- it satisfies the case semantic oracle;
- it does not violate the user style profile;
- it arrives before the simulated patience deadline;
- its insertion range remains current;
- the policy permits the completion class.

Partial acceptance uses token or line boundaries and charges subsequent edits. Incorrect but superficially similar candidates are rejected unless a dedicated error-prone user profile explicitly models accidental acceptance. Report those profiles separately.

---

# Part XI. Live multi-user serving and inference tuning

## 52. Serving experiment principle

Tune on development traces, freeze the selected configurations, then run final traces. Do not choose a serving configuration after viewing final quality or final load labels.

Quality invariance is checked at every configuration. A throughput improvement that changes deterministic outputs becomes a distinct profile rather than a free optimization.

## 53. Workload traces

Create versioned manifests for:

### 53.1 Closed-loop interactive users

Each virtual user waits, types, requests, observes, and accepts/rejects before issuing the next request. This models an editor user and naturally includes think time.

### 53.2 Open-loop arrivals

Requests arrive independently of completion time. Use deterministic Poisson-like and replayed burst traces. This exposes queue saturation and avoids the coordinated-omission problem of closed-loop-only testing.

### 53.3 Schema-affinity traffic

Groups of users share a database and repeated prompt prefix. This tests prefix caching and metadata cache locality.

### 53.4 Randomized-schema traffic

Every request changes schema/prefix affinity. This is the negative control for caching claims.

### 53.5 Cancellation-heavy traffic

Users continue typing quickly and supersede requests. Report useful completions, cancelled work, and wasted generated tokens.

### 53.6 Mixed completion classes

Use a frozen mix of token, line, block, and statement requests with realistic input/output length bands.

## 54. Concurrency ladder

Run the supported subset of:

```text
1, 2, 4, 8, 16, 32, 64 virtual users
```

Stop increasing concurrency when safety, error-rate, memory, or runaway-tail gates trigger. A stopped high-concurrency cell remains evidence; do not extrapolate a fictional result.

## 55. Inference settings to evaluate

Capability- and version-gated settings include:

- maximum concurrent sequences;
- maximum batched tokens;
- GPU memory utilization;
- context length;
- continuous batching defaults;
- chunked prefill;
- prefix caching;
- scheduling policy;
- preemption behavior;
- streaming;
- speculative decoding with a pinned draft profile;
- quantization as a separate model profile;
- guided decoding or JSON schema enforcement;
- reasoning policy;
- request priority classes.

Do not sweep every Cartesian combination. Use a staged tuning design:

1. establish safe memory envelope;
2. choose concurrency/batched-token candidates;
3. test prefix caching on affinity and randomized controls;
4. test chunked prefill on mixed context lengths;
5. test speculative decoding only after correctness and draft-acceptance instrumentation works;
6. freeze no more than three Pareto candidates per model for final load.

## 56. Serving metrics

### 56.1 End-to-end latency

- request validation;
- context construction;
- SQL retrieval;
- queue wait;
- prefill;
- time to first token;
- decode;
- time to first valid candidate;
- validation/ranking;
- SQL finalization;
- end-to-end completion.

Report p50, p90, p95, p99, max, and grouped intervals.

### 56.2 Throughput

- requests admitted/completed/offered per second;
- input and output tokens per second;
- semantically correct completions per second;
- accepted characters per second;
- GPU-seconds per valid offered completion;
- SQL retrievals and logical reads per second.

### 56.3 Queue and scheduler

- queue depth over time;
- wait-time distribution;
- batch size and token occupancy;
- preemptions;
- starvation and per-tenant fairness;
- deadline misses;
- admission rejection;
- cancellation before and after model admission;
- generated tokens discarded after cancellation.

### 56.4 Cache

- prefix-cache hit/miss and reusable-token estimates;
- metadata cache hit/miss;
- embedding cache hit/miss;
- affinity versus randomized delta;
- cache memory and eviction behavior.

### 56.5 Resource

- GPU utilization, memory, power when available, and throttling;
- CPU, process memory, and event-loop delay;
- model server KV-cache utilization;
- SQL CPU, logical reads, duration, waits, file I/O, tempdb, and Query Store rows;
- network bytes and connection-pool saturation;
- telemetry overhead.

## 57. Service-level objectives

Treat these as application targets to freeze, not hardware promises.

Suggested classes:

| Class | Router target | Suggested p95 target |
|---|---|---:|
| token | deterministic only | 150 ms |
| line | deterministic or shallow LLM | 900 ms |
| block | LLM permitted | 1,500 ms |
| statement | explicit deeper completion | 2,500 ms |

The final preregistration may adjust these once, using CPU/dev and non-test GPU canaries. It must freeze them before final load traces.

Report SLO attainment, not merely average latency.

## 58. Pareto analysis

Each engine/model/configuration produces a point with:

```text
semantic quality
coverage
p95 and p99 latency
throughput
GPU/CPU/SQL resource use
cancellation waste
safety violations
```

A configuration is dominated if another is no worse on all governed axes and better on at least one. The report highlights the Pareto frontier and the recommended router policy. Do not collapse everything into one arbitrary score without also showing the component metrics.

---

# Part XII. Baselines, controls, and ablations

## 59. Mandatory baselines

- empty/prior;
- parser/keyword;
- catalog prefix;
- accepted-history n-gram/template;
- retrieval template;
- direct LLM;
- strong router.

The LLM must beat the best eligible baseline, not an intentionally weak straw model.

## 60. Negative controls

1. shuffled gold labels for learned rerankers;
2. shuffled schema descriptions while preserving object counts;
3. schema from the wrong fixture;
4. permission-overbroad context, development-only, to detect hidden leakage;
5. empty retrieval;
6. vector results randomly ordered;
7. suffix removed;
8. cursor shifted by one token;
9. comments containing misleading instructions;
10. schema descriptions containing prompt injection;
11. random accepted history from another user;
12. batch-order permutation;
13. affinity labels shuffled for cache analysis;
14. telemetry disabled on a bounded run to measure observer overhead;
15. cancellation disabled on cancellation-heavy traffic.

## 61. Robustness suites

- unseen schema names with familiar structures;
- familiar schema names with changed columns;
- case-sensitive and accent-sensitive collations;
- Unicode and reserved identifiers;
- very short and very long prefixes;
- long irrelevant open-document context;
- ambiguous intent;
- invalid pre-existing SQL;
- capability unavailable;
- natural-language comments in different styles;
- sampled decoding;
- model batch sensitivity;
- quantized profile versus full-precision profile, treated as distinct identities.

## 62. Failure-stage atlas

Every failed row receives one primary stage and optional contributing stages:

```text
DATASET
CURSOR
PACKET
METADATA
RETRIEVAL
PROMPT
QUEUE
MODEL_TRANSPORT
MODEL_EMPTY
MODEL_FORMAT
INSERTION
PARSE
BINDING
COMPILE
EXECUTION
SEMANTIC
SAFETY
RANKING
CALIBRATION
DEADLINE
CANCELLATION
PERSISTENCE
REPORTING
```

This prevents “model failure” from swallowing system bugs.

---

# Part XIII. Statistical and evaluation contract

## 63. Units and pairing

- Primary quality unit: split group, with row-level records retained.
- Primary serving unit: request, grouped by trace and virtual user where appropriate.
- Model and arm comparisons are paired on identical packets.
- Bootstrap resampling groups by the largest leakage unit.
- Serving intervals use run/trace blocks, not independent-request fiction when autocorrelation is material.

## 64. Primary quality endpoint

Use `semantic_success_top1` on eligible test rows, with the exact task-specific oracle hierarchy frozen before test.

A positive LLM-lift claim requires:

1. paired lower confidence bound above the best eligible non-LLM baseline by the frozen minimum effect;
2. no forbidden safety violations;
3. hallucination and permission-violation gates pass;
4. result survives template and schema holdouts or is narrowed to the passing scope;
5. no corresponding shuffled-label or leakage control is positive.

## 65. Secondary endpoints

- semantic top-3;
- binding accuracy;
- nonexistent identifier rate;
- exact and token-normalized match;
- parser delta;
- selective accuracy/coverage;
- keystrokes saved;
- latency and throughput;
- retrieval recall and token cost;
- model-resource efficiency.

## 66. Uncertainty

Use:

- grouped bootstrap intervals;
- paired permutation or sign-flip tests for model/arm deltas;
- calibration intervals;
- Holm or false-discovery correction for declared families of comparisons;
- sensitivity across split-group definitions;
- leave-one-family-out analysis;
- per-case raw rows for independent reconstruction.

Do not report dozens of uncorrected p-values as discoveries.

## 67. Calibration and abstention

Train ranking on `train`, fit calibration and thresholds on `calibration`, and open test once. Candidate confidence can consume only features available at serving time:

- engine score;
- candidate rank margin;
- parser/binder/validator statuses;
- retrieval distances and required-object recall;
- model log probability when safely available;
- candidate length and completion class;
- context truncation and ambiguity features.

It cannot consume test-family identity or oracle results unavailable online.

## 68. Suggested result taxonomy

```text
BASELINE_DOMINANT
LLM_LIFT_MULTITOKEN
SUFFIX_LIFT
SCHEMA_GROUNDING_LIFT
HYBRID_RETRIEVAL_LIFT
VALIDATION_RERANK_LIFT
SELECTIVE_POLICY_USEFUL
ROUTER_DOMINANT
MODEL_SPECIFIC_ADVANTAGE
SYSTEM_ERASES_MODEL_GAP
ANN_PRESERVES_DECISIONS
ANN_UNNEEDED_AT_SCALE
CACHE_AFFINITY_LIFT
BATCHING_THROUGHPUT_TAIL_TRADEOFF
SPECULATION_LIFT
SPECULATION_NO_LIFT
QUALITY_GOOD_LATENCY_MISSED
SERVING_SLO_MET
RULES_SUFFICIENT
CLEAN_NULL
STOP_DATA
STOP_ORACLE
STOP_PORT
STOP_SAFETY
STOP_BUDGET
STOP_RUNTIME
```

---

# Part XIV. Safety, privacy, and isolation

## 69. Non-negotiable product safety

- GhostType never autoexecutes a user-facing completion.
- Validation execution occurs only in isolated fixtures.
- The serving principal cannot alter control-plane scientific rows.
- Permission-filtered metadata is enforced before prompt construction.
- User/session history is tenant-scoped.
- secrets, connection strings, credentials, and raw production data are forbidden inputs.
- retrieved comments and descriptions are untrusted data, not instructions.
- destructive, privilege, external-access, and server-level completions are blocked or require an explicit non-primary extension policy.
- model text never becomes dynamic SQL inside the control database.
- all SQL queries that consume identifiers use validated quoting and controlled construction.

## 70. Safety test catalog

Test at least:

- cross-user retrieval leakage;
- hidden object exposure;
- comment prompt injection;
- schema-description prompt injection;
- malicious bracket/quote identifiers;
- dynamic SQL payloads in metadata;
- destructive completion;
- cross-database escalation;
- server-level configuration;
- stale document version;
- late response after cancellation;
- duplicate finalization;
- telemetry payload injection;
- report rendering injection.

Any actual cross-tenant or unauthorized metadata leak is `STOP_SAFETY`.

---

# Part XV. API, UI, and teaching experience

## 71. Required API

```text
POST /api/completions
POST /api/completions/{requestId}/cancel
GET  /api/completions/{requestId}
POST /api/replay
POST /api/simulations/single-user
POST /api/load-runs
GET  /api/evaluations
GET  /api/evaluations/{runId}
GET  /api/reports/{name}
GET  /health
GET  /ready
```

The completion API supports streaming but final decision persistence remains atomic.

## 72. Minimal UI

A lightweight web page should show:

- SQL editor with cursor and suffix-aware completion;
- active schema and permission profile;
- offered candidates and validation badges;
- retrieval witnesses;
- latency waterfall;
- accept, partial accept, reject, and cancel controls;
- run selector;
- quality and serving dashboards;
- per-row evidence explorer.

The UI is a demonstrator and audit surface, not the experiment’s source of truth.

## 73. Student-facing lab path

1. inspect a completion packet;
2. run deterministic baselines;
3. inspect catalog snapshots and permission filtering;
4. compare prefix-only and suffix-aware prompts;
5. inspect full-text, exact-vector, and hybrid retrieval;
6. validate candidates with ScriptDom and fixtures;
7. run the CPU single-user simulator;
8. run one local model profile;
9. compare four frozen model profiles;
10. tune multi-user serving on development traces;
11. run final load traces;
12. reconstruct reports and adjudicate claims.

---

# Part XVI. Repository layout

## 74. Proposed tree

```text
aidataapps/ghosttype/
  README.md
  COURSE.md
  SOURCE_INTAKE.md
  EXPERIMENT_LOG.md
  GHOSTTYPE_PREREGISTRATION.md
  GHOSTTYPE_FREEZE_RECORD.md
  GHOSTTYPE_STATE_OF_RECORD.md
  GHOSTTYPE_DATASET_REPORT.md
  GHOSTTYPE_COMPLETION_QUALITY_REPORT.md
  GHOSTTYPE_RETRIEVAL_REPORT.md
  GHOSTTYPE_SERVING_REPORT.md
  GHOSTTYPE_SAFETY_REPORT.md
  GHOSTTYPE_CLAIMS_TABLE.md
  GHOSTTYPE_LIMITATIONS.md
  GHOSTTYPE_REPRODUCIBILITY.md

  docs/
    SPEC.md
    SPEC_ADDENDUM.md
    CPU_PROFILE.md
    COLAB_PROFILE.md
    DATASET_CONTRACT.md
    ORACLE_CONTRACT.md
    SERVING_CONTRACT.md

  config/
    models.json
    embeddings.json
    campaigns/
      cpu-dev.json
      gpu-quality.json
      gpu-serving-tune.json
      gpu-serving-final.json
    engines/
    contexts/
    validators/
    thresholds.json
    load-profiles/
    report-plan.json

  data/
    ghosttype_dataset_v2/
    manifests/
    licensed_docs/

  fixtures/
    manifests/
    ddl/
    seeds/
    backups/
    expected/

  db/
    migrations/
      001_control.sql
      002_dataset_catalog.sql
      003_session_completion.sql
      004_retrieval_validation.sql
      005_serving_telemetry.sql
      006_evaluation_reporting.sql
      007_security.sql
    procedures/
    reports/
    vector-indexes.sql
    fulltext.sql

  dotnet/
    GhostType.ScriptDom/

  src/
    api/
    config/
    contracts/
    context/
    inference/
    parser/
    retrieval/
    completion/
    validation/
    ranking/
    persistence/
    telemetry/
    simulator/
    load/

  scripts/
    env-init.sh
    colab-host-init.sh
    runtime-env.sh
    run-init.ts
    setup-database.ts
    import-dataset.ts
    build-fixtures.ts
    snapshot-catalog.ts
    validate-dataset.ts
    freeze-campaign.ts
    model-server.ts
    port-gate.ts
    run-quality.ts
    run-single-user.ts
    run-load.ts
    archive-run.ts
    export-database.sh
    mirror-run.sh

  analysis/
    evaluate_quality.py
    evaluate_retrieval.py
    evaluate_calibration.py
    evaluate_serving.py
    evaluate_pareto.py
    build_reports.py
    reconstruct.py

  load/
    traces/
    generators/
    k6-or-custom-runner/

  tests/
    unit/
    contract/
    sql/
    parser/
    fixtures/
    integration/
    load/

  runs/
```

## 75. Language choices

- TypeScript for API, orchestration, request lifecycle, model clients, and SQL persistence.
- .NET for pinned ScriptDom parsing.
- Python for statistical analysis and report/figure generation.
- T-SQL for migrations, queue procedures, retrieval, telemetry analysis, and report views.
- Shell for environment and archival workflows.

Keep interfaces versioned and small. Do not place scientific decisions in an untested notebook.

---

# Part XVII. Implementation program

## GT-0 - Source intake and isolation

- create branch/worktree;
- record reviewed branch heads and source hashes;
- copy the governing spec and rebuilt dataset;
- define forbidden writes to Labs 01-03;
- reserve ports, containers, volumes, databases, run pointer, and Drive root;
- create `SOURCE_INTAKE.md` and append-only `EXPERIMENT_LOG.md`.

**Exit:** source-intake artifact and isolation tests pass.

## GT-1 - Dataset import and audit

- import all 588 records;
- verify 265 legacy IDs and hashes;
- apply JSON Schema and gold-contract validation;
- build split-group and leakage reports;
- materialize immutable dataset rows in SQL;
- generate `GHOSTTYPE_DATASET_REPORT.md` skeleton from SQL.

**Exit:** no unresolved required structural failure.

## GT-2 - Fixtures and metadata snapshots

- create fixture databases;
- seed deterministic data;
- define permission profiles;
- snapshot catalog rows and descriptions;
- hash backups/manifests;
- test restore/rebuild equivalence.

**Exit:** fixture and catalog reconstruction pass.

## GT-3 - ScriptDom and semantic oracles

- build pinned parser service;
- validate cursor positions and insertion round trips;
- run parse/bind/compile/execute eligibility;
- implement result-shape, result-set, mutation, and safety comparators;
- disposition capability-dependent rows.

**Exit:** every row has explicit oracle statuses; no silent pass.

## GT-4 - SQL control-plane foundation

- apply idempotent migrations;
- implement campaigns, freezes, evidence, artifacts, queue leases, temporal document state, and atomic finalization;
- enable Query Store and full-text where supported;
- implement backup/BACPAC/reconstruction.

**Exit:** SQL integration and recovery tests pass.

## GT-5 - Deterministic baselines

- parser/keyword;
- catalog prefix;
- history n-gram/template;
- retrieval template;
- router skeleton;
- full 588-row evaluation.

**Exit:** baseline report and row-level predictions reconstruct.

## GT-6 - Retrieval plane

- build schema, relationship, exemplar, and history corpora;
- create fixed embeddings;
- exact vector and full-text retrieval;
- hybrid fusion;
- required-object recall metrics;
- optional vector-index capability and plan proof.

**Exit:** no test leakage; exact retrieval reference frozen.

## GT-7 - Completion service and CPU model

- implement request/response contracts;
- model gateway and streaming assembly;
- candidate parser;
- deadline and cancellation;
- small CPU-compatible profile or governed stub;
- end-to-end smoke.

**Exit:** raw-first persistence and lifecycle tests pass.

## GT-8 - Validation, ranking, and calibration plumbing

- candidate validators;
- deterministic ranking features;
- calibration data path;
- abstention policy;
- simulator acceptance path;
- report skeletons.

**Exit:** synthetic known-world tests recover expected rankings and abstentions.

## GT-9 - CPU single-user campaign

- frozen sequential baseline replay;
- bounded local-model replay;
- closed-loop user trace;
- cold/warm runs;
- cancellation-heavy run;
- telemetry overhead control;
- backup and reconstruction.

**Exit:** `CPU_PROFILE.md` and development state of record complete.

## GT-10 - GPU preregistration and freeze

Freeze:

- dataset, fixtures, snapshots, splits, and oracles;
- target profiles and images;
- request and prompt bytes;
- retrieval witnesses/corpora;
- engine and validation arms;
- metrics, thresholds, and statistical plan;
- quality job inventory;
- serving tune and final traces;
- drop order and stop rules.

**Exit:** freeze hash and tag exist before a target model loads.

## GT-11 - Target profile port gates

Run all gates for Qwen, Muse, Gemma, and OLMo. Record runtime-only adjustments as profile amendments before campaign rows.

**Exit:** each target is `PASS`, separately classified, or `STOP_PORT`.

## GT-12 - Frozen quality replay

For each resident model:

- run direct, schema, hybrid, and validated arms;
- checkpoint raw responses and SQL rows;
- run selective missing-cell retries only;
- complete backup and Drive verification;
- evict safely.

**Exit:** achieved matrix is complete or explicitly budget-stopped.

## GT-13 - Quality evaluation and ablations

- candidate/oracle scoring;
- paired model/arm comparisons;
- suffix, schema, retrieval, validator, and history ablations;
- calibration and risk-coverage;
- robustness and safety suites;
- failure-stage atlas.

**Exit:** retained prediction and metric rows reconstruct independently.

## GT-14 - Serving tune campaign

- safe memory envelope;
- concurrency/batch sweeps;
- affinity/randomized cache controls;
- chunked prefill and optional speculation;
- cancellation and open-loop traces;
- select Pareto candidates using dev traces only.

**Exit:** final serving configs frozen.

## GT-15 - Final serving campaign

- execute frozen traces at selected configurations;
- collect end-to-end, model, SQL, host, and GPU telemetry;
- run quality invariance subset;
- archive every load run.

**Exit:** no unresolved telemetry join or silent request loss.

## GT-16 - Reports, reconstruction, and closeout

Generate:

- state of record;
- dataset/oracle report;
- completion quality report;
- retrieval report;
- serving report;
- safety report;
- claims table;
- limitations and reproducibility;
- final artifact manifest and BACPAC.

Run independent reconstruction from the archived database and artifacts.

**Exit:** completion tag only after reconstruction and hash verification.

---

# Part XVIII. Command contract

## 76. CPU bootstrap

```bash
cd aidataapps/ghosttype
./scripts/env-init.sh --profile cpu
npm ci
npm run build
npm run run:init -- --campaign cpu-dev
npm run db:setup
npm run dataset:import -- --bundle data/ghosttype_dataset_v2
npm run dataset:validate
npm run fixtures:build
npm run catalog:snapshot
npm run oracles:validate
npm run baselines:evaluate
npm run check
npm run simulate:single-user -- --profile cpu-dev
npm run reports
```

## 77. Colab bootstrap

```bash
cd /content/labs/aidataapps/ghosttype
sudo ./scripts/colab-host-init.sh
source ./scripts/runtime-env.sh
./scripts/env-init.sh --profile colab
npm run run:init -- --campaign gpu-quality
npm run doctor
npm run campaign:freeze -- --config config/campaigns/gpu-quality.json
```

## 78. Model residency

```bash
npm run model -- start --profile qwen-3.8-27b --replace
npm run port:gate -- --profile qwen-3.8-27b
npm run quality:run -- --profile qwen-3.8-27b --resume
npm run quality:verify -- --profile qwen-3.8-27b
npm run db:backup
npm run run:mirror
npm run model -- stop
npm run model -- evict --profile qwen-3.8-27b
```

Repeat only after the previous residency recovery artifacts pass.

## 79. Serving runs

```bash
npm run load:tune -- --profile qwen-3.8-27b --trace config/load-profiles/tune.json
npm run serving:freeze -- --profile qwen-3.8-27b
npm run load:final -- --profile qwen-3.8-27b --trace config/load-profiles/final.json
npm run serving:evaluate -- --profile qwen-3.8-27b
```

## 80. Finalization

```bash
npm run evaluate:quality
npm run evaluate:retrieval
npm run evaluate:calibration
npm run evaluate:serving
npm run evaluate:pareto
npm run reports
npm run repro
npm run db:bacpac
npm run run:archive
npm run run:mirror
```

All commands resolve an explicit run or the locally recorded current run. Scientific scripts refuse an ambiguous run.

---

# Part XIX. Testing program

## 81. Unit tests

- canonical JSON and hashes;
- cursor conversions;
- insertion range;
- suffix integrity;
- request keys;
- cancellation state machine;
- candidate parsing;
- SQL identifier quoting;
- context token budgets;
- ranker features;
- metric functions;
- report rendering.

## 82. Contract tests

- every API schema;
- model provider request/effective settings;
- raw-first persistence;
- parser service version and response;
- embedding dimensions;
- SQL capability snapshots;
- load trace manifests;
- artifact manifests.

## 83. SQL integration tests

- migration from empty database;
- migration hash refusal;
- queue lease exclusivity and expiry;
- duplicate request idempotency;
- atomic finalization;
- temporal document history;
- permission-filtered catalog;
- exact vector ranking;
- full-text retrieval;
- ANN plan evidence when enabled;
- Query Store capture;
- backup/restore/BACPAC reconstruction;
- report-view determinism.

## 84. Parser and fixture tests

- all supported parser versions;
- incomplete SQL;
- Unicode positions;
- comments and strings;
- prefix/candidate/suffix round trip;
- known valid and invalid candidates;
- result-shape comparisons;
- result-set order policies;
- rollback and disposable-database cleanup;
- timeout and runaway-query termination.

## 85. Concurrency tests

- stale document cancellation;
- late response disposal;
- double cancellation;
- lease theft prevention;
- worker crash and recovery;
- model restart;
- SQL restart;
- connection-pool exhaustion;
- queue backpressure;
- fairness and priority;
- streaming disconnect;
- telemetry writer slowdown.

## 86. Known-world analysis tests

Build synthetic rows where the correct conclusion is known:

- perfect model;
- random model;
- always-abstain;
- baseline-dominant;
- retrieval leakage;
- calibration inversion;
- batch output change;
- tail-latency saturation;
- prefix-cache affinity gain;
- cancellation waste.

The report builder must recover the expected taxonomy before real outcomes are opened.

---

# Part XX. Artifacts and reports

## 87. Run directory

```text
runs/ghosttype-<campaign>-<timestamp>/
  environment/
  manifests/
  dataset/
  fixtures/
  raw/
  packets/
  predictions/
  metrics/
  telemetry/
  figures/
  tables/
  reports/
  database/
  artifacts.json
```

Large artifacts remain out of Git and are mirrored to the run’s Drive subtree. Git receives manifests, lightweight summaries, code, configs, and final reports.

## 88. Required reports

### `GHOSTTYPE_STATE_OF_RECORD.md`

Exact achieved matrix, dispositions, headline conclusions, claim ceiling, and recovery coordinates.

### `GHOSTTYPE_DATASET_REPORT.md`

Counts, provenance, legacy preservation, catalogs, splits, leakage, fixture and oracle validation, capability routing, and limitations.

### `GHOSTTYPE_COMPLETION_QUALITY_REPORT.md`

Baselines, models, arms, families, semantic outcomes, calibration, simulated acceptance, and failure atlas.

### `GHOSTTYPE_RETRIEVAL_REPORT.md`

Schema/exemplar recall, lexical/vector/hybrid comparisons, token cost, latency, exact/ANN agreement, and plan evidence.

### `GHOSTTYPE_SERVING_REPORT.md`

Latency waterfalls, throughput, concurrency, queueing, batching, caching, cancellation, fairness, resources, SLOs, and Pareto frontier.

### `GHOSTTYPE_SAFETY_REPORT.md`

Permission isolation, prompt injection, destructive SQL, tenant boundaries, execution sandbox, and all stop events.

### `GHOSTTYPE_CLAIMS_TABLE.md`

One row per licensed or rejected claim, evidence tag, metric row, scope, and falsifier.

### `GHOSTTYPE_LIMITATIONS.md`

Dataset, simulator, model, hardware, SQL build, and external-validity limits.

### `GHOSTTYPE_REPRODUCIBILITY.md`

Exact commands, hashes, database import, independent reconstruction, and missing proprietary prerequisites if any.

## 89. Figures

Every figure has a source CSV/Parquet and a ceiling-aware caption.

Required candidates:

- semantic success by family and arm;
- hallucination and permission-violation matrix;
- suffix versus prefix paired deltas;
- retrieval recall versus context tokens;
- risk-coverage curves;
- latency waterfall;
- throughput versus p95/p99 latency;
- queue depth timeline;
- cancellation waste;
- cache affinity control;
- quality-latency-resource Pareto frontier;
- model/arm failure-stage matrix.

---

# Part XXI. Stop rules and compute drop order

## 90. Stop rules

- `STOP_DATA`: dataset identity, split, schema, legacy preservation, or leakage gate fails.
- `STOP_ORACLE`: required parser, fixture, or semantic oracle is not trustworthy.
- `STOP_PORT`: model profile cannot pass governed canaries.
- `STOP_SAFETY`: unauthorized metadata, tenant leak, destructive execution, or control-plane mutation.
- `STOP_RUNTIME`: repeated infrastructure corruption, unbounded resource failure, or unrecoverable database state.
- `STOP_BUDGET`: compute/disk/time exhausted after completed cells are preserved.
- `STOP_QUALITY`: selective quality gate fails; user-facing LLM offering is disabled, but analysis continues.
- `STOP_SERVING`: error or tail-latency safety ceiling reached; higher concurrency is not attempted.

## 91. Drop order

When constrained, drop in this order:

1. optional native-FIM arm;
2. optional code-specialized control;
3. speculative decoding;
4. extra sampled-decode repetitions;
5. highest concurrency cells after saturation is established;
6. model reranker extension;
7. accepted-history personalization extension;
8. non-primary capability catalogs;
9. one or more non-router architecture arms, preserving direct and router anchors.

Never drop:

- raw response retention;
- dataset/oracle validation;
- deterministic baselines;
- exact retrieval reference;
- test leakage controls;
- four-model packet identity for achieved cells;
- cancellation accounting;
- safety tests;
- row-level predictions;
- metric uncertainty;
- database backup/reconstruction.

---

# Part XXII. Risks and design responses

## 92. Exact-match undercounts valid SQL

Use hierarchical semantic oracles and keep exact metrics diagnostic.

## 93. Execution overstates correctness

Use result shape, result set, mutation state, required references, and safety constraints. Passing one tiny fixture is scoped to that fixture.

## 94. Test examples leak through retrieval

Build corpora from allowed roles before test. Hash and audit every member. Run canary phrases and nearest-neighbor leakage checks.

## 95. General chat models are poor FIM models

Make prompt-simulated suffix conditioning primary. Native FIM is capability-gated. A clean null is acceptable.

## 96. Large models miss interactive latency

The router and deterministic fast path are primary architecture arms. Report `QUALITY_GOOD_LATENCY_MISSED` honestly.

## 97. Batching changes output

Run port gates and quality-invariance subsets. Separate serving profiles when needed.

## 98. CPU and GPU backends differ

CPU is development evidence unless the profile is genuinely identical. Do not merge their metrics.

## 99. Telemetry changes the workload

Run bounded telemetry-off controls and report observer overhead.

## 100. Preview SQL features drift

Capability-probe every optional feature and retain a baseline implementation. Pin build and compatibility level.

## 101. Synthetic users are not humans

Report simulated acceptance and keystrokes only. A human study is a future lab.

## 102. Generated dataset becomes template-heavy

Hold out generator/template families, audit near duplicates, and report leave-one-family-out results.

---

# Part XXIII. Final adjudication questions

The state of record must answer:

1. How many of the 588 records passed every required pre-freeze gate?
2. Which records were ineligible and why?
3. What did the strongest deterministic baseline achieve?
4. Where did each LLM add or lose value?
5. Did suffix context help cursor-in-middle completion?
6. Did schema context reduce hallucination and permission errors?
7. Which retrieval method best balanced recall, token cost, and latency?
8. Did validation-aware reranking improve semantic success?
9. Did calibrated abstention create a useful operating point?
10. Which completion families remained unsolved?
11. How stable were results across schema and template holdouts?
12. Did batch mode change governed outputs?
13. What were TTFT, time-to-candidate, and end-to-end latency by model?
14. Where did queue saturation begin?
15. What throughput and SLO did each final configuration achieve?
16. Did prefix caching pass the randomized-schema control?
17. Did speculative decoding help, hurt, or fail to activate?
18. How much work was wasted after cancellation?
19. What SQL resources did retrieval, telemetry, and reporting consume?
20. Which configurations lie on the Pareto frontier?
21. Did the router dominate single-engine designs?
22. Were there any safety or tenant-isolation violations?
23. Which claims are licensed, narrowed, rejected, or unresolved?
24. What should Lab 05 build from the strongest remaining gap?

---

# Part XXIV. Completion checklist

Lab 04 is complete only when:

- [ ] a dedicated branch/worktree and isolated runtime exist;
- [ ] the governing spec and source-intake record are committed;
- [ ] the rebuilt 588-row dataset is vendored and hashed;
- [ ] all 265 legacy IDs are preserved and audited;
- [ ] JSON Schema, gold-contract, cursor, split, and leakage gates pass;
- [ ] ScriptDom and fixture runtime gates have explicit results for every row;
- [ ] SQL control and workload databases build from empty state;
- [ ] migrations are immutable and hash-checked;
- [ ] permission-filtered catalog snapshots reconstruct;
- [ ] deterministic baselines run on the full eligible corpus;
- [ ] exact full-text/vector retrieval references are frozen;
- [ ] no test data enters retrieval or training;
- [ ] request lifecycle, raw-first persistence, deadlines, and cancellation pass integration tests;
- [ ] CPU single-user campaign and reconstruction complete;
- [ ] GPU preregistration and job inventory freeze before target loads;
- [ ] every achieved model passes its port gate;
- [ ] frozen quality packets replay across achieved models and arms;
- [ ] all missing cells have explicit dispositions;
- [ ] calibration and test roles remain separate;
- [ ] serving tuning uses development traces only;
- [ ] final serving configurations and traces are frozen;
- [ ] load tests retain queue, batch, cache, cancellation, model, SQL, and host evidence;
- [ ] safety suite reports zero undispositioned severe violations;
- [ ] reports are generated from SQL rows and hashed source tables;
- [ ] every figure has a source table;
- [ ] backup/BACPAC and Drive mirrors verify;
- [ ] independent reconstruction matches the released reports;
- [ ] the claims table respects the claim ceiling;
- [ ] the final tag is created only after reconstruction.

---

# Appendix A. Suggested SQL reporting views

```text
reporting.v_completion_quality_by_model_arm_family
reporting.v_semantic_success_paired_deltas
reporting.v_identifier_hallucinations
reporting.v_permission_violations
reporting.v_suffix_ablation
reporting.v_retrieval_recall_token_latency
reporting.v_selective_quality_coverage
reporting.v_simulated_acceptance
reporting.v_latency_waterfall
reporting.v_serving_slo
reporting.v_queue_saturation
reporting.v_cancellation_waste
reporting.v_cache_affinity_control
reporting.v_sql_resource_cost
reporting.v_pareto_frontier
reporting.v_failure_stage_atlas
reporting.v_claim_evidence
```

Each report procedure accepts an explicit run ID and refuses to combine incompatible campaign hashes.

# Appendix B. Suggested metric names

```text
text_exact_top1
text_normalized_token_top1
text_edit_similarity
syntax_candidate_parse_pass
syntax_parse_delta
binding_object_precision
binding_object_recall
binding_column_precision
binding_column_recall
binding_nonexistent_identifier_rate
binding_unauthorized_identifier_rate
semantic_compile_pass
semantic_result_shape_pass
semantic_result_equivalence
semantic_mutation_equivalence
semantic_success_top1
semantic_success_top3
safety_forbidden_offer_rate
selective_coverage
selective_semantic_accuracy
calibration_ece
calibration_nll
calibration_brier
sim_full_accept_rate
sim_partial_accept_rate
sim_keystrokes_saved
retrieval_required_object_recall
retrieval_irrelevant_object_count
retrieval_context_tokens
retrieval_latency_ms
ann_recall_at_k
ann_decision_agreement
latency_context_ms
latency_retrieval_ms
latency_queue_ms
latency_ttft_ms
latency_model_ms
latency_validation_ms
latency_total_ms
serving_requests_per_second
serving_correct_offers_per_second
serving_accepted_chars_per_second
serving_deadline_miss_rate
serving_cancel_rate
serving_cancelled_tokens
serving_queue_depth
serving_batch_size
serving_prefix_cache_hit_rate
resource_gpu_seconds_per_correct_offer
resource_sql_logical_reads_per_request
resource_sql_cpu_ms_per_request
```

# Appendix C. Dataset adoption boundary

The supplied rebuilt dataset is a reviewed candidate, not a frozen scientific result. Its static gates have passed in the packaging environment. The implementation must run the pending ScriptDom, fixture, SQL-build capability, and independent reconstruction gates. Any correction creates a new dataset version with a changelog, new manifest hash, and preserved prior package.

# Appendix D. Recommended next lab directions

The strongest follow-ons are likely:

- a real editor extension and opt-in human acceptance study;
- online personalization with privacy-preserving history;
- learned candidate ranking from real acceptance data;
- code-specialized model adaptation or distillation;
- speculative completion where a small model drafts and a large model verifies;
- cross-dialect completion and dialect routing;
- multi-file SQL project context and dependency graphs;
- agentic repair after failed completion execution.

The next direction should be chosen from GhostType’s retained failure atlas, not from whatever inference feature is fashionable that week.
