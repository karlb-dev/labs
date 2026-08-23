# LogWarden Experiment Log

Append-only operator record. Times are UTC. Entries are written before an
adaptation can affect evidence whenever possible; superseded facts remain in
place and receive a later disposition.

- 2026-08-23T02:40:15Z — LW-0 began on a clean Colab VM. Created dedicated
  worktree `/content/worktrees/aidataapps-logwarden` and branch
  `aidataapps-logwarden` from exact Lab 2 head `88ea443092cd27226272776e6c6fcf8fbce329de`.
- 2026-08-23T02:40:15Z — Preflight found an idle NVIDIA RTX PRO 6000
  Blackwell Server Edition (97,887 MiB total; 97,251 MiB free), 188 GiB local
  free, mounted Drive with 178 GiB free, and no Docker installation or active
  Lab 2/model process on this VM.
- 2026-08-23T02:47:00Z — GPU hard gate passed with PyTorch 2.11.0+cu128: a
  real CUDA tensor allocation, reduction, host readback, and synchronization
  completed on the RTX PRO 6000. No CPU fallback is permitted for model-scale
  stages.
- 2026-08-23T02:48:00Z — Implementation sequencing clarification (pre-data):
  the executable bootstrap uses `run:init -> db:setup -> doctor` after service
  startup, because the full capability doctor persists its evidence into the
  control database. This changes only command order, not any scientific input
  or claim. A bounded host/service preflight still runs before database setup.
- 2026-08-23T02:49:00Z — Current official method inputs were checked before
  implementation: SQL Server 2025 Linux full-text installation, Extended
  Events dispatch/session syntax, Query Store configuration, and vLLM JSON
  schema response handling. Runtime capability probes remain authoritative.
- 2026-08-23T02:55:00Z — User-directed pre-data scope adjustment: promote the
  practical observability layer ahead of every long LLM run, even where the
  addendum labels high-volume telemetry tables Tier 2. The gate now requires
  append-only file journaling plus incremental SQL records for the full agent
  state machine, raw vLLM metrics snapshots and discovered metric names,
  client monotonic request timings, queue/XE/SQL/GPU samples, retry and repair
  paths, and telemetry completeness/reconciliation tests. This does not alter
  packets, prompts, labels, arms, or scoring. It increases write/sampling
  overhead, so the development tier must measure that overhead and freeze the
  sampling interval before the primary campaign.
- 2026-08-23T03:00:34Z — Inspected the PI-supplied `cloud-deploy-agent`
  reference at exact commit `273221e97e36abea34e47961d4deb42455750f02`.
  Adopted reconstructable semantic projections over one immutable journal,
  request-level client/server correlation, per-call prompt/tool/config hashes,
  explicit loop-limit and response-failure classes, safe forced terminal
  dispositions, critical-path versus additive timing, and per-job/grouped
  diagnostics. Rejected TPU-specific fields and heuristic token/timing values
  as observations: LogWarden will retain their provenance or null. This is an
  instrumentation-only pre-data change and does not affect episode assignment,
  prompts, arm definitions, labels, or primary outcomes; it adds storage and
  measured overhead that the development gate will quantify.
- 2026-08-23T03:04:48Z — Installed Docker Engine 29.7.2, Compose 5.5.0,
  NVIDIA Container Toolkit 1.20.0, and rootless runtime prerequisites. The
  CUDA hard gate had already passed before installation. Under the Codex PTY,
  a `nohup`-detached rootless daemon was reaped when its launching command
  exited, so this session supervises `dockerd-rootless.sh` in a retained
  foreground execution cell. This is an environment-lifecycle adaptation only;
  it does not change container images, ports, data, or scientific factors.
- 2026-08-23T03:16:32Z — The pinned SQL Server base image did not include the
  SQL Server 2025 apt repository, and nested rootless BuildKit cannot mount a
  fresh `/proc` for Dockerfile `RUN`. Added the official Ubuntu 24.04 SQL 2025
  repository to the canonical Dockerfile. For this Colab profile, installed
  the package in a live host-namespace build container and committed it while
  running, restoring the base user/entrypoint/CMD. Engine label
  `17.0.4075.5` and package `mssql-server-fts=17.0.4075.5-1` match exactly.
  Derived image ID is
  `sha256:eb4ee252ae0ff6a18b5b40e05eea283e251ee69fbe8926e28def6d0adb43f95f`;
  SQL runtime probes returned product `17.0.4075.5` and full-text installed
  `1`. Impact: build mechanism only; full-text capability is retained rather
  than falling back to BM25.
- 2026-08-23T03:16:32Z — A local diagnostic process listing exposed the
  generated SA password before any migrations, principals, workload, or
  scientific data existed. Rotated the SA, lab, and agent credentials; removed
  host-side password command arguments; recreated SQL with the preserved empty
  data volume; and revalidated login/full-text health. Nested rootless Docker
  could not complete the old SQL process stop, so the already-idle pre-schema
  process required an exact host PID kill after its grace period. Impact: no
  evidence or results; credentials remain ignored and are not mirrored.

- 2026-08-23T03:17:15.609Z — LW-0 run initialized: logwarden-smoke-20260823T031714Z; manifest=3d45cfa2523f12dc6042b411750b1db9a442d045af2d82cb8b6caa2e47a94dd9.
- 2026-08-23T03:30:07Z — Initial schema apply committed migrations 001–004,
  then SQL Server rejected full-text catalog creation inside migration 005's
  user transaction (error 574). Migration 005 rolled back and was not
  recorded. Split full-text DDL into explicitly non-transactional, fully
  guarded migration 011 while keeping the knowledge tables transactional.
  The runner records migration mode and refuses hash drift. Impact: database
  deployment mechanics only; no workload or scientific data exists yet and
  the required full-text design is unchanged.
- 2026-08-23T03:39:39Z — LW-0/LW-1 foundation capability gate passed on SQL
  Server 17.0.4075.5: compatibility 170, RCSI, Query Store READ_WRITE with
  1-minute intervals/60-second flush/ALL capture/waits, exact vector, native
  JSON and vector-index metadata, full-text catalog/index, the five-event XE
  session running with 2-second dispatch latency, and the Tier 1 permission
  matrix. Optional SQL-native chunk and embedding helpers were not discovered
  and remain deferred; application-owned pinned embeddings are the registered
  primary path, so this does not lower Tier 1 claims.
- 2026-08-23T03:39:39Z — Queue integration testing found that the initial
  lease procedures only heartbeated or reclaimed rows while status was exactly
  `leased`; a worker crash after packet/model/tool progress could therefore
  strand a job. Forward migration 012 now heartbeats every active state,
  reclaims expired active leases with a recorded transition, clears terminal
  leases, and preserves retry timing. This pre-data correctness fix does not
  change episode assignment or outcomes. The eight-case SQL integration suite
  passed, including concurrent distinct claims, invalid-transition rejection,
  expired mid-state recovery, negative permissions, constraint rejection,
  full-text population, and exact 1024-dimensional vector retrieval.
- 2026-08-23T03:58:44Z — Installed the capture-specific XE v2 contract before
  scenario data: `LogWarden-Inject%` is the primary predicate, `LW_%` is the
  server/database fallback, and Lab/Admin/Agent/Ingest traffic is excluded.
  Added attention and file-growth coverage while retaining error, RPC/batch,
  deadlock, and blocked-process events. The session is running with the frozen
  2-second dispatch latency. The earlier v1 XEL files remain development-only
  input and must be cleared with the source cursors before standard capture.
- 2026-08-23T03:58:44Z — The first transactional XE drain proved that
  `file_offset` identifies an event-file block rather than a unique event: 134
  legitimate events shared one offset. The failed transaction inserted no
  rows. Forward migrations changed the stable source key to file + block
  offset + deterministic within-block ordinal and changed optional numeric XML
  fields to `TRY_CONVERT`. The next drain inserted 836 raw/canonical rows and
  its immediate retry read zero. Impact: source identity correctness only; no
  eligible episode existed at the time.
- 2026-08-23T03:58:44Z — Completed a development smoke capture of ten benign,
  bounded scenarios (conversion, missing object, duplicate key, truncation,
  controlled integrity signal, bad login, failed backup, bounded query
  pressure, benign activity, and ambiguous review). `--no-wait` was used only
  for feasibility; this smoke run is not a frozen quality sample. All ten
  cleaned up and passed required-evidence verification, including both XE and
  ERRORLOG rows for error 18456 and both sources for the logged controlled
  signal. Receipt
  `7271b30792a9e054b8f95cedb057df153dda4434517f348b8894cc11fe6bbd5`
  records 10/10 and capture rate 1.0. The driver exposes failed-login status as
  `ELOGIN` without a numeric field, so driver classification accepts the
  explicit login-failed message; authoritative source verification still
  requires error number 18456. ERRORLOG replay inserted 0/461 on the immediate
  duplicate scan, proving the initial idempotency path.
- 2026-08-23T04:05:43Z — Closed the database recovery gate. The separately
  mounted exports volume was root-owned, so SQL Server correctly refused its
  first backup attempt; no backup file was created. Moved the target to an
  `mssql`-owned directory inside the dedicated Lab 3 data volume. Nested
  rootless `docker cp` then hit the known proc-remount restriction, so the
  checkpoint resolves the exact rootless volume mountpoint read-only and
  copies the named backup file from the host namespace. This changes only the
  export transport. COPY_ONLY/CHECKSUM/compressed backups, RESTORE VERIFYONLY,
  full restore, physical CHECKDB, table-count probes, and teardown all passed
  for both exact Lab 3 databases. Durable restore-test receipt:
  `cfec5d3bc7a9d52a16c69a9ff0645be5c3d9ea6f21972c7a2fd138bfd34ad2a4`.
- 2026-08-23T04:05:43Z — The Lab 3 checkpoint watchdog passed its dirty-tree
  recovery mode: it created and verified a branch bundle, retained binary
  worktree patch and untracked archive, made checksummed database backups,
  refreshed the run resume record, pushed committed HEAD, regenerated the
  artifact inventory, and mirrored the run to Drive. It deliberately did not
  auto-commit concurrent source changes. The clean-tree auto-handoff/commit
  mode will be exercised after this implementation checkpoint is committed.
- 2026-08-23T04:15:26Z — The watchdog clean-tree path passed after commit
  `26900ff`: it updated both handoffs, committed only that tracked status block
  as `de9ca40`, pushed, verified a new branch bundle, backed up both databases,
  inventoried the run, and mirrored it to Drive. This closes the pre-long-run
  durability primitive; recurring supervision is not started until a long job
  exists.
- 2026-08-23T04:15:26Z — Built ten development incident packets only from
  verified injection-event links. Correlation tokens, disposable identities,
  missing objects, and restricted paths are deterministically redacted; exact
  model-visible packet hashes and separate evaluator-only truth hashes are in
  SQL. The independent audit found zero protected keys/values, cross-role
  scenario groups, shared canonical events, missing truth rows, or hash drift.
  Build receipt
  `06670cc91548962090843e921f0b72d0f58e38544f79b2fd0ffb9768fb749e11`;
  audit receipt
  `f6cc4310ac038a320106c94ad3d43912e9737fd840043c85b63ccc7a698fbd74`.
  These smoke packets remain development-only and their tool manifest is
  explicitly `tools-v1-building`, not a campaign freeze.
- 2026-08-23T04:15:26Z — First full systems sampler pass persisted GPU, SQL,
  queue, XE-pipeline, host, and raw availability snapshots for chat and both
  embedding endpoints. The three model services were correctly unavailable
  because none had been started; unavailable values remain null with reasons.
  Sampler receipt
  `f19f2884678a01d417f41dbe559f5be4416adc2a46012b0cde6cb3c4f4e89867`.
  The file-first hash-chained journal ingester validated four records, inserted
  three new records after the already committed foundation event, projected
  spans, and updated per-epoch cursors; receipt
  `430d53f721b081915e88d5d79ece7362673e7e611e9332b6e7153e9b342d3ab4`.
- 2026-08-23T04:38:40Z — Adapted the pre-inference journal layout before any
  model residency: every sampler/worker now owns a uniquely named hash-chained
  JSONL file. The earlier single-file design was safe for sequential smoke
  calls but would allow two processes to race on sequence/hash state during a
  long run. SQL ingestion now discovers the retained journal set and commits
  bounded batches with per-file/per-epoch cursors. Impact: concurrency and
  recovery correctness only; no scientific packet or model result exists yet.
- 2026-08-23T04:38:40Z — The telemetry crash/replay gate injected failure
  after the first committed two-record batch, recovered the remaining four
  records, then replayed all six as duplicates. Event IDs, terminal cursor
  sequence/hash, spans, and traces reconciled exactly; receipt
  `ac60e25a385f2ecfedc1bec6d4fd33dc92de9cbb78f6d0c5935bcda529d20660`.
  Two separate sampler processes then produced distinct epochs, two unique
  queue samples, six unique raw endpoint snapshots, and two closed traces;
  restart receipt
  `c30e80918b27592426a3657cb0e6c64fe97aa4a09057eda935deff97ba290218`.
- 2026-08-23T04:38:40Z — The model-client integration gate passed nine
  terminal routes: valid decision, retried 503 then success, malformed agent
  JSON, malformed service envelope, empty output, schema rejection, exhausted
  HTTP failure, timeout, and response-byte truncation. Exact request/response
  bytes were durable before parse and matched SQL SHA-256 values; each attempt,
  validation layer, job, work-item disposition, state event, trace, and span
  reconciled. Receipt
  `fdce92eeeb8344b52c02455a6d244ee9d9839bc57c2f556f01be5caae0f5d474`.
  The first gate launch exposed only a development harness parameter-name bug
  before inference; its one empty synthetic fixture was explicitly stopped and
  marked `interrupted_gate` before the clean rerun. No scientific observation
  or score was affected.
- 2026-08-23T04:38:40Z — A 20+20 ABBA synthetic comparison measured the
  complete file-first instrumentation path against the same HTTP/envelope/
  contract parse without instrumentation. Mean added latency was 1.291 ms and
  p95 added latency was 1.497 ms (instrumented p95 2.694 ms), below the frozen
  100 ms development ceiling; receipt
  `3888e28dbd673f71e75ef09bf34451e9b15c4d3196e8c0db2a4c66a2baa6efea`.
  Global reconciliation then matched 246 journal rows, 66 exact raw artifacts,
  all SQL events/cursors, and every closed trace/span. Its input-set and receipt
  hashes are `f986c6132c3c7d344892d1263d55b54a35858ec676f966557225ee281e3f9dc3`
  and `ecf98a36f50811ec934d64c42670fdbd70f7f6bd735eccc09495ea2ccc2a0c70`;
  an immediate rebuild was byte-identical.
- 2026-08-23T04:44:20Z — The mandatory ERRORLOG rotation gate passed. A
  pre-roll marker retained the exact source-position key after
  `sys.sp_cycle_errorlog` renamed its generation, a post-roll marker produced a
  second unique raw/canonical row, and immediate replay inserted zero of 599
  parsed records (all 599 were recognized duplicates). Receipt
  `594b46485c6865068c2093c311190b09eb900406c51b76d2f972704a73087d1e`.
  The file reader was factored into a retrying shared module; this changes only
  container file-access robustness and has no scientific-result impact.
- 2026-08-23T04:49:04Z — The checkpointed SQL/container restart gate passed
  with receipt
  `f6688d4472af0db6947076097a27be7d14cd58490b736d9dc442fb2c2adc2c99`.
  A pre-restart marker retained its exact source-position key after moving to
  `errorlog.1`; a post-restart marker was unique; and immediate replay
  recognized all 997 parsed records as duplicates. The same container was
  restarted with a new container-init PID and SQL start time. The first
  development attempt showed that `SHUTDOWN WITH NOWAIT` stopped the SQL
  client connection but not this image's host-PID wrapper. The gate now
  resolves one exact direct `sqlservr` child beneath the inspected container
  PID and signals only that PID, then proves both databases are ONLINE and
  queryable before ingestion. This also fixed a real readiness race in which
  SA login briefly succeeded before `LogWardenControl` was ready. Impact:
  recovery-test reliability only; no campaign data or scientific factors
  existed or changed.
- 2026-08-23T05:02:50Z — Installed the development `tools-v1` least-privilege
  surface: seven strictly typed, bounded, read-only tools; canonical argument
  and registry hashes; exact frozen-snapshot lookup; certificate-signed server
  diagnostics; and full-text runbook search through stored procedures. Direct
  access to evaluator truth, snapshots, runbook tables, queue/control tables,
  ingestion procedures, DDL, server DMVs, and `msdb` backup tables remains
  denied to `lw_agent`. The security gate passed nine positive and ten
  negative cases; its receipt is
  `6c2e962c5bb0673e30af5ddc8bac33c540a771cb49fba7d18f666921f6e6ff57`
  and registry hash is
  `25c79c34cd382bba6bb1f9139401aac8bfea2f603bd83d137b0eb209ddfac6bb`.
  The registry remains explicitly unfrozen until the standard corpus and
  runbooks close.
- 2026-08-23T05:02:50Z — Tool development produced three fail-closed findings
  before the passing gate. The first control master-key password derivation
  used a raw hexadecimal digest that SQL Server rejected under password
  policy, before any tool migration began; the derivation was corrected. The
  first migration transaction then rejected reserved output alias
  `transaction` and rolled back without a migration record; the alias became
  `txn`. Finally, the initial negative gate discovered that `lw_agent` could
  read `msdb.dbo.backupset` through inherited `guest`/`public` permissions.
  An explicit deny closed the bypass, and the required backup-history tool was
  redesigned as a bounded `msdb` owner-executed proxy rather than granting
  table access to the signing certificate. The failed gate wrote only unique
  synthetic development corpus/snapshot fixtures and no passing receipt; they
  are excluded from any standard corpus or score. Hash-locked migrations
  017–019 and server assets 003–005 preserve the final design. Afterward the
  doctor passed with snapshot
  `82a438326c1d4c42dac6b475fb2987d3039f21eab1113cae22e42d2e70d6f597`,
  SQL integration passed 8/8 with receipt
  `d72103d195e7f70853ecda6134fa8f958de0320cf14f372a320d500f4213fa9a`,
  and 28 unit tests in 11 files passed. Impact: security and deployment
  correctness only; there is still no model or campaign observation.
- 2026-08-23T05:05:28Z — A post-gate privilege review found that the bounded
  internal `msdb` proxy was still directly executable by `lw_agent`, even
  though direct table access was denied and the proxy was not in the checked-in
  registry. Forward control migration 020 changed the registry-listed wrapper
  to a static cross-database call and restored its module signature; server
  asset 006 grants proxy execution only to the matching certificate user and
  revokes it from the runtime user. A new negative case proves the direct proxy
  call fails while the listed backup tool still succeeds. The superseding gate
  passed nine positive and eleven negative cases with receipt
  `c0bf815c9db432cc28555146f597eabd46eb935731826763f01e55b75cbfef57`.
  The doctor passed with snapshot
  `a315adb9b7e36c3fe901d3dd9b2f6f02ffcb3026da827e4d7bbd48bd14769c34`,
  SQL integration passed 8/8 with receipt
  `107e85c39510a2e161796a348579740f288d8d834472f00862bd3be30fc2d655`,
  and the 28 unit tests still passed. This supersedes the prior development
  tool-gate receipt without changing data, tools, prompts, or outcomes; impact
  is a strictly narrower runtime privilege boundary.
- 2026-08-23T05:18:38Z — Built the unfrozen `primary-v1` knowledge corpus from
  deterministic source: 60 original MIT-licensed troubleshooting guides (six
  perspectives for each governed incident class), 480 heading-aware chunks,
  official Microsoft reference URLs checked on 2026-08-23, exact source/body/
  chunk hashes, and zero lab-identifier, synthetic-number, or correlation-token
  leakage findings. Corpus file hash is
  `cb465736137b45b466aff4d89ec1f89f377039dec49e732301e66f87174e6f3c`,
  manifest hash is
  `b734ba4c05ff3e548b779b1f9718e8956e4f0f9bb45a8246492664f88a0e66df`,
  and build receipt is
  `d8b6b81aff2811460fdee6a5469ceb69540381720ceaf9f961af7c149e2f2d79`.
  The manifest is attached to the mutable campaign but remains deliberately
  unfrozen until embeddings, hybrid retrieval, and the standard scenario
  leakage audit pass.
- 2026-08-23T05:18:38Z — The first post-insert population wait used the wrong
  full-text catalog identifier and timed out after 60 seconds even though SQL's
  crawl had completed; it wrote no readiness receipt and did not change corpus
  rows. Review also found that feasibility migration 011 indexed content but
  not `heading_path`, contrary to the governing K-1 contract. Forward
  non-transactional migration 021 rebuilt the index over both columns, and
  migration 022 made the bounded search procedure query both. The corrected
  population gate observed status 0 and 483 indexed items (480 primary chunks
  plus three retained synthetic development-gate chunks). Eleven agent-login
  canaries retrieved the intended family, including punctuation and numeric
  input; receipt
  `7e8c38b4218dae4b43c27e95cd522ae669f026fbe94249c3890e7d8cc51808be`.
  Impact: monitoring and retrieval-contract correctness before freeze; no
  embedding, model, packet, or scored result was affected. Doctor snapshot
  `55a221402864f730c49f72aaef5609de404dc3a2cd9f6f2c419950f9dbe9e7db`,
  SQL integration receipt
  `ee107c1615f7d8830e766826d223edb844f9339c8a4a8523cf338e43b861ddae`,
  and 31 unit tests in 12 files all pass.
- 2026-08-23T05:30:58Z — Built and reconciled the unfrozen
  `logwarden-standard-v1` catalog: 60 scenario templates, 600 deterministic
  variants, ten governed families, all five K/C/U/M/N regimes, exact role
  counts of 60 dev, 60 calibration, 300 test-ID, 120 test-variant-holdout, and
  60 test-unknown, zero cross-role groups, zero truth/tool-budget/runbook
  violations, and at least 40 held-out ID/variant episodes per family. Catalog
  file hash is
  `b39b3a91107c094b0b7d461cca77deffed69659c07b1d84341dd9d17b477b58f`,
  campaign manifest hash is
  `492ca25b89b21d83a3ca10a56e36b769f6b03d1cd1c1737515ca6c741cb525d5`,
  schedule hash is
  `617ec3731b116f5a2db7163bae2010ca8b523ee8ebb774c1c7c23b660e01ae76`,
  and structural-gate receipt is
  `dd7b8af23dc13f39b191f33cc3a5d53ba632b7f062c0b8c337a6f4836653e0d3`.
  The explicit `--replace-building-manifest` operation replaced the earlier
  ten-scenario smoke manifest only on the still-mutable, never-frozen campaign;
  it changed no captured smoke rows and the standard schedule has injected zero
  episodes.
- 2026-08-23T05:30:58Z — Adapted the base packet-spacing plan before standard
  capture. Ten variants in one scenario group are intentionally one second
  apart to preserve recurrence/burst context; adjacent scenario groups are
  separated by the sum of their packet before/after windows plus a measured
  one-second margin. This reduces the planned capture from roughly 20 hours to
  8,039 seconds (about 2 h 14 min) without permitting evidence-window overlap
  across independently scored groups. Impact: variants within a group are
  deliberately correlated and must be scored with group-aware splitting and
  paired statistics; they are not independent episode-level replicates. The
  gate records 60 groups of exactly ten, a nonnegative cross-group margin, and
  the intentional-overlap policy in the hashed schedule configuration.
- 2026-08-23T06:11:01Z — Completed the real incident injector and context
  snapshot implementation before any standard capture. The driver now creates
  tokenized, bounded deadlocks, sustained blocking, transaction-log exhaustion
  in a registered disposable database, failed authentication, controlled
  logged errors, failed backup attempts, query pressure, and schema/data
  errors. Each episode records driver observations, SQL session IDs, exact
  context-snapshot argument/result hashes, verification state, and cleanup
  postconditions. The deadlock tool returns only a bounded semantic summary
  (victim, process/resource counts, client/login/isolation/wait metadata, and
  statement/resource hashes), never raw deadlock XML. The four-family
  deadlock/blocking/log-full/error gate passes 20 checks under the final
  verifier with receipt
  `fe4f1101303cc038fa522b7b9a5a17324f1011fb37fbf2e2a9441c3d646b835f`;
  its 4/4 capture receipt remains
  `97d75d34a7d1eecced3800ab023a90f27ae056ebe3d12a29bcda55f9d7aed974`.
- 2026-08-23T06:11:01Z — The addendum requests `NO_EVENT_LOSS` for the
  Extended Events session. SQL Server 2025 RTM-CU8 (`17.0.4075.5`) rejected
  `error_reported` in a `NO_EVENT_LOSS` session with engine error 25643. The
  failed DDL temporarily removed the development session but no scientific
  episode ran in that interval. The capability-forced contract therefore uses
  `ALLOW_SINGLE_EVENT_LOSS`, while retaining 1-second dispatch latency, 16 MiB
  files, 20 rollover files, the severity/application predicates, and all
  required events. Doctor and the injector gate fail closed unless runtime
  dropped-event, dropped-buffer, and failed-target-buffer counters are zero;
  the telemetry sampler additionally records blocked fire time, bytes written,
  rollover count, and parse errors. The versioned contract hash is
  `667bee8af19d8b2dba416cd31968215945b8f39ee1121e6294e37549bc2068c1`.
  Current counters are all zero. Impact: the requested retention mode is not
  supported for this event mix, so losslessness is monitored and gated rather
  than guaranteed by the mode; any nonzero loss counter invalidates a run.
- 2026-08-23T06:11:01Z — The first eight-family, three-signal development gate
  captured and broadly verified 8/8 episodes, but its deep gate correctly
  failed because `query_pressure` linked 12 nearby `sql_batch_completed`
  records and only two carried its application token. Investigation showed
  that `mssql` sends the actual parameterized workload through
  `sp_executesql`, producing token-bearing `rpc_completed` events; the SQL
  batches were connection initialization. The former ±2-second paired-event
  attribution could therefore accept neighboring episodes. No standard
  episode, packet, freeze, or model result existed. The verifier now deletes
  and deterministically rebuilds episode links, forbids proximity-only
  attribution, and requires the full 32-character token in successful RPC or
  batch payloads. The v1 catalog is retained byte-for-byte at hash
  `6c39c83b269ed1f39829e389b9a697945ca5b8ad712678996f42512f150851de`
  as failed development evidence. Corrected v2 hash
  `1d6b9352645a15b40f1caed66f12c5ab94f2eeef6201c6ca7b5a4b29d4cb9ebd`
  then passed 8/8 capture verification
  (`0ff81209e857278a56c9920c233281cb34d4b97815d60936efff9291b407e79a`)
  and all 32 deep checks
  (`8ace73c31d9b07e7dba3388900eca9b0da94d71740176313f76b260f901cabe3`).
  Impact: this closes a false-attribution path and strengthens episode
  independence; the failure and rerun are development-only and excluded from
  scored data.
- 2026-08-23T06:11:01Z — Before freeze, rebuilt the still-uninjected standard
  catalog to make successful query/noise evidence use the actual token-bearing
  RPC event. The governed allocation and 8,039,000 ms schedule duration are
  unchanged. The superseding catalog, campaign manifest, schedule, build, and
  structural-gate hashes are respectively
  `5bc260b8c11c6ace97cb5f047dd0ad7f68958860cc24c3e0e43313ee9f05934f`,
  `50d47873fda8ccd3367ddbd5f9f1356f1fd306f6406c33de122471a1c5e7872d`,
  `3b7097f5b855d91741bc2d470e3074040464b96a9ae66e1df766639c6daea20b`,
  `02f48fb93c18e26fa8b637774f839b7ea3c635988ab99151e1ff79372390437b`,
  and `5535d9f8c29d58d6b1ec222b3ab0842c20de0c669d01367d3118ff07e1f17f82`.
  The guarded replacement removed exactly one prior standard schedule, 60
  templates, 600 variants, and 750 evidence rules only after proving zero
  standard injections, packets, or freezes. Current regressions pass: doctor
  `802aadaa2bd16872a2b3a5c6dddcac65149a92575c7a79ba3a2515ff6a0b25fe`,
  SQL 8/8
  `8310ac0f46310d125926e12aa05a44a5aa065c44de0292b9860f329fd6c8b215`,
  tool security 9 positive/11 negative
  `a7d1cd7484dd8dd1a510307c32e3e36274588dc7bb64294ef5b695b91a63a0ae`,
  39 unit tests in 14 files, and telemetry reconciliation over 252 journal
  records plus 66 raw artifacts
  `8ffc7bd73a43d50d64113cabfe8fc9f0e70db105b0271acd308230cc6c89c505`.

- 2026-08-23T04:10:28.290Z — LW-0 run initialized: logwarden-smoke-20260823T041027Z; manifest=dfa2aa4cb8ae5cfacc50e8eeb7b000068be080ff9d1114f376a66ccb1c00e979.

- 2026-08-23T04:20:00.000Z — Mac profile established on branch
  `aidataapps-logwarden-mac` (docs/MAC_PROFILE.md). Foundation reproduced on
  Apple Silicon (M4 Max, 48 GB): SQL Server 2025 FTS image (17.0.4075.5) built
  and healthy under Docker Desktop Rosetta amd64 emulation with
  MSSQL_MEMORY_LIMIT_MB=4096; run logwarden-smoke-20260823T041027Z reached
  doctor disposition PASS (exact vector, full-text, XE session, Query Store,
  two-principal permission matrix) and 8/8 SQL integration tests. Serving
  plane: Azure Foundry Local 0.10.3 pinned to :8010; qwen3-4b-generic-gpu:2
  passed a deterministic decision canary (~60 tok/s warm, think_strip repair);
  qwen3-embedding-0.6b-generic-gpu:1 returned 1024-dimension embeddings.
  Deviations recorded in docs/MAC_PROFILE.md; base compose.yaml gpus stanza
  rewritten to long-form list syntax for older compose validators (semantics
  unchanged). Mac-plane evidence is never comparable to the frozen campaign.

- 2026-08-23T05:35:00.000Z — Mac-plane model bench completed over the frozen
  16-episode mac-eval-v1 set (npm run mac:eval; rows and metrics in run
  logwarden-smoke-20260823T041027Z). Six models, three runtimes, all
  disclosed: Foundry ONNX (qwen2.5-0.5b, qwen3-4b, qwen3-8b, olmo-3-7b),
  the Muse Glimmer 30B INT4 bundle via custom cache, and Gemma 4 E4B
  (mlx-community OptiQ 4-bit) via mlx_lm.server after the Foundry catalog
  proved to carry no Gemma. Headlines: Muse 87.5% exact-triple with 100%
  class and 100% action at 16.4 tok/s under the §A-8 max-tokens override
  (512-token budget ablation retained: 3/16 decisions); Gemma 4 E4B 75%
  triple with 100% first-pass contract at 71.1 tok/s but dismissed the
  unknown-signal abstention episode; qwen3-8b 50%, olmo-3-7b 37.5% (100%
  first-pass), qwen3-4b 25%, qwen2.5-0.5b unusable. Transport findings for
  the runtime: Qwen3 /no_think must ride every user turn; Muse needs
  final-channel (to=user) extraction; catalog models over-escalate against
  the action rubric. Gemma 4 12B blocked: all MLX conversions declare
  gemma4_unified, unsupported by mlx-lm 0.31.3. Mac-plane development
  evidence only; never comparable to the frozen campaign.

- 2026-08-23T05:55:00.000Z — Merged Colab commits through 2987b41 (runbook
  corpus, scenario catalog, read-only tool boundary with msdb signed proxy)
  into the mac branch and revalidated the mac plane: npm run check 38/38,
  migrations 017–023 and server 002–006 applied, doctor PASS, SQL
  integration 8/8, tool-security gate PASS (7 tools, 9 positive, 11
  negative; gate needed one seeded LogWardenWorkload backup for its
  msdb-history positive case). Two mac fixes: models-mac test updated for
  the -mlx profile keys, and database-checkpoint falls back to docker cp
  when the volume mountpoint is not host-visible (Docker Desktop);
  backup + restore-test round-trip passes with DBCC CHECKDB on both
  databases.
- 2026-08-23T06:15:20Z — Merged the five Mac/Foundry/report commits through
  `725cdae` into the Colab working branch after the injector checkpoint. The
  merge preserves the platform-separated Foundry/MLX registry, Apple Silicon
  Docker/Rosetta profile, 16-episode development harness, six-model retained
  rows, and row-derived HTML report. Colab validation passes Compose config,
  16 test files/44 tests, doctor, SQL integration 8/8, and the 9-positive/
  11-negative tool gate. The doctor, SQL, and tool receipts are
  `802aadaa2bd16872a2b3a5c6dddcac65149a92575c7a79ba3a2515ff6a0b25fe`,
  `0334583c0d376b53776a0e65b2892d0ef6d4d372d2d22f8b4bae262908a151d2`,
  and `f5965e712ba2edfe542ca767f042ecc29d81d985af3b85f8bdb6cb2e80de48cf`.
  The committed report rebuilds byte-for-byte from its
  retained data plus template and contains the full 6 x 16 grid. A first
  backup invocation lacked the rootless `DOCKER_HOST` because the isolated
  shell had not sourced `scripts/runtime-env.sh`; it changed no database and
  produced no receipt. The documented invocation then passed backup and full
  restore/CHECKDB with receipt
  `20a92098906ac63588ce951ce15e21d3ec932ca10f4d81beb504cffbfc4a9a18`.
  Foundry Local is not installed on this Linux VM and was intentionally not
  installed or benchmarked now: per the user-directed ordering, any Foundry
  canary follows the governed local-model campaign and remains a distinct,
  non-comparable development plane. Before such a run, its harness must meet
  the same raw-response and phase telemetry gate as vLLM.
- 2026-08-23T06:35:00Z — Started the pinned Qwen embedding plane on the Colab
  GPU: `Qwen/Qwen3-Embedding-0.6B` at revision
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, vLLM 0.27.1 image
  `vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967`,
  pooling runner, and port 8011. Added a fail-closed client and live port gate
  that durably stores the exact request before send and response before parse,
  hash-chain spans, client phase timings, usage, vector identities/norms, raw
  service logs, model/health identity, GPU snapshots, and exact Prometheus
  expositions before/midpoint/after. The first gate stopped before inference
  because Prometheus does not instantiate the `/v1/embeddings` labeled HTTP
  series until its first request (failed receipt
  `fa0ab4aa4a508914695e0cece338cfbe77f8c50e4e525755ca9bc3e8180b1bf4`).
  The metric family remains mandatory, but this valid startup state is now an
  explicit zero baseline. The second gate stopped after canaries because the
  first-ever CUDA result differed from its warm repeat by max component
  0.0007367311 and cosine 0.9999658542 (failed receipt
  `ad48a6bdd0ac5591de6cf073da58740bc41844b171c5705b1e086dd035be889a`).
  The contract now retains and bounds cold-to-warm drift at cosine >=0.9999 and
  max delta <=0.001 while requiring two warmed repeats to be byte-identical.
  Impact: this exposes rather than erases GPU warmup variance; model/revision,
  raw vectors, and hashes remain available for every result, and a tolerance
  breach still blocks the corpus.
- 2026-08-23T06:35:00Z — The corrected live embedding port gate passed with
  receipt `90e15a999165c2848e000d25069884b9f453d0a416bd667ba344c9423140e36b`.
  Four HTTP calls covering six inputs produced ordered finite unit-normalized
  1024-dimensional vectors. Cold, warm, and warm-repeat hashes were identical
  on the warmed service. Exact counter deltas were four embedding HTTP
  requests, six successful vLLM items, 108 prompt tokens, six latency
  observations, zero errors, and zero preemptions. The midpoint GPU sample
  showed `VLLM::EngineCore` resident with 6,047 MiB and 3% utilization; all 32
  gate journal records were hash-validated and inserted into SQL. The full
  suite now passes 17 files and 57 tests. This gate authorizes the bounded
  Qwen corpus embedding build, not a long chat-model campaign.
- 2026-08-23T06:32:00.000Z — Mac-plane revalidation of the merged mainline
  (through 171c393): 44/44 tests, migrations through 025 + server 007 (XE v3)
  applied, doctor PASS, SQL integration 8/8, tool-security gate PASS. Ran the
  governed injector flow end to end on the emulated instance (10-episode
  smoke schedule injector-gate-v2): all injectors succeeded with verified
  cleanup; capture verification initially failed 2 episodes whose evidence
  expects sql_batch_completed while the Node driver ships the tokenized
  payload as sp_executesql (rpc_completed) — capture-verify now treats
  batch/rpc completion as one client-batch evidence family (full-token match
  unchanged), and verification passes 10/10. Scenario-injector gate PASS
  (34 checks) after registering the standard-v1 scenario manifest on the
  building campaign. Runbook corpus built on the mac (60 runbooks, 484
  chunks, Foundry embeddings, full-text populated); runbook lexical gate
  PASS (11 cases). The scenario-catalog gate still fails here only on
  full-campaign schedule state (600 episodes across all roles); that is
  Colab campaign-prep, not a mac defect.

- 2026-08-23T06:57:00Z — Strengthened the successful embedding port receipt
  after the corpus preflight correctly found that an in-memory `Buffer` in the
  health/model evidence did not rehash after JSON serialization. No corpus
  request or SQL vector existed at that refusal. Raw bodies now remain only in
  their hash-addressed files, while the receipt stores paths, sizes, and
  hashes; an immutable copy also lives under the gate's raw directory. The
  independently reloaded replacement receipt is
  `d9f7b756c907a4aa9516a1c47875dec2a5e7d932b4bc4a721c93bb3e93492323`
  and supersedes `90e15a999165c2848e000d25069884b9f453d0a416bd667ba344c9423140e36b`
  as the corpus authorization. Impact: service results are unchanged; the
  evidence envelope is now independently verifiable.
- 2026-08-23T06:57:00Z — Added transactional migration 026
  (`498e007b5b26ff8d25202c1a4be886a24411239154b24f077b7fe63a6ad5b99d`)
  for per-vector input/request/response/operation/batch/run provenance. Its
  first application compiled constraints in the same batch as new columns;
  SQL Server rejected the not-yet-visible names and the whole migration rolled
  back. Splitting column DDL from constraints with `GO` applied cleanly; no
  partial schema or data existed. The 480 primary chunks were then embedded in
  15 batches using frozen `chunk-content-v1` inputs and committed only after
  raw response durability. Generation deltas are exactly 15 HTTP requests,
  480 successful items, 23,736 prompt tokens, 480 latency observations, zero
  errors, and zero preemptions.
- 2026-08-23T06:57:00Z — Corpus verification deliberately stopped after all
  rows committed when it first compared the service-vector hash to SQL's
  float32 textual round trip (failure receipt
  `d11fe35c33b40176b346db1239e2d791fc0fde86d5d05120e2eb21142fce05a8`).
  A resume then exposed upper-case SQL `uniqueidentifier` rendering in a raw
  metadata path (failure receipt
  `76502bfdabebaaa6a0c5475752a7af097dff42a1c21ea69c6dffe4bc509e966b`).
  The verifier now normalizes UUID presentation, preserves distinct source and
  SQL-storage hashes, reloads all 15 request/response/metadata and batch
  receipts, and measures conversion rather than requiring the two
  serializations to hash alike. Final receipt
  `bca6706cf683f265f61bff8e7f7abe0531bf8722afb36f479d29c66f7feca1aa`
  validates 480 vectors/491,520 components. Worst float32 component delta is
  `4.995651239902976e-9`, minimum cosine is
  `0.9999999999999969`, and request latency p50/p95 is 36.843/41.932 ms.
  Ordered input, service-vector, and SQL-storage set hashes are respectively
  `eb3559f43fa413bc70d514924c57ea044b25ab2a0c1e97ee91593144216099fc`,
  `012daefc3106442687f0842bf442e3ec1f37c1558e6f07dd354165034af54aa5`,
  and `f8a929f0bc3f87ddf8a2edc4fe6406e52900da7f2759d2169800188817cf3d31`.
  Impact: no vectors were regenerated or silently rewritten; the distinction
  between service precision and SQL storage is explicit and bounded.
- 2026-08-23T06:57:00Z — Added SQL full-text/exact-vector/hybrid RRF retrieval
  with frozen candidate k=50 and RRF k=60, app-owned query embeddings, raw
  result durability, and complete `kb.retrieval_runs/results` component-rank
  evidence. Migration 027 hash is
  `d4b803e4478322bab6fbef6011ac82723927698226450ac50613fe0d7a40e66c`.
  The first 33-call gate completed but its administrative query grouped by a
  vector column, which SQL forbids; read-only verification failed with receipt
  `50e9b95aeb6d4d4392ecf65fa431300641b202d7718064e0f47e0346a45194b6`.
  Aggregating by retrieval ID before joining the vector row fixed only the
  audit. The replacement gate passed 11 queries across lexical, vector, and
  hybrid with receipt
  `f6a3b4aefbca5214beee3272566f5f01d696f4be3dcb922a9660db2ea86d1dfd`:
  all three reached recall@5=1 and MRR=1 on these easy development canaries;
  SQL p50 was 14.074, 72.502, and 75.022 ms respectively. This proves
  mechanics, not hybrid lift; held-out packet evaluation remains required.
  SQL retained 33 runs and 1,814 component rows; 192 telemetry records were
  ingested. Regressions pass 18 files/60 tests, doctor
  `c3cc8ced060215850572d50927d21cc02055159ac3bd0815335e8c9683a9316f`,
  SQL 8/8 `64e45aa7f64b7caf466d92182ced84c4954eda6e68b16ec985b10f6959297911`,
  and tool security 9 positive/11 negative
  `c463035983b16e85e4da6e8d1f43efe30d99b492b18a6c8ea5aae06508f218a5`.
- 2026-08-23T07:26:00Z — Implemented the Tier-1 bounded multi-turn agent loop
  and evidence schema. Migration 028
  (`2635473a5e87e59a0093a767c0889d6583cfa70bee7d4eb0149ff9958283d657`)
  adds immutable loop budgets/counters plus full/transmitted tool-result
  provenance and retrieval linkage. The primary contract is user-turn-only,
  omits both the OpenAI `tools` field and guided `response_format`, uses four
  model turns/four tools/two same-tool calls/4,000 characters per result/12,000
  total/180 seconds, and commits no SQL transaction across inference. Prompt,
  model/retry, registry, argument, policy, tool, cache, snapshot, retrieval,
  decision-validation, and decision-persistence phases now exist in both the
  hash-chained journal and SQL; all action rows remain non-executing proposals.
  The governed 900-token decode is a new `primary-json-v2` row, leaving the
  earlier 512-token development identity immutable. Future standard packets
  expose all seven registry tools; replay resolves absent argument-keyed
  snapshots as scored deterministic misses.
- 2026-08-23T07:26:00Z — The first agent-loop attempt stopped before model
  traffic because `READPAST` is invalid with the control database's RCSI mode
  unless a locking read is explicit (failure receipt
  `9319b01b3c4fa0b673ee649687cab668ae190eb6a17cb7b33a3831bf07c2de87`).
  Migration 029
  (`e1cce5a645db7acb5dc451ee8f8859c729468bf70a5487700b81b18040ec2af8`)
  added `READCOMMITTEDLOCK`. A second pre-model attempt then exposed that a
  pooled connection retains SERIALIZABLE after an earlier evidence transaction
  (failure receipt
  `b63f397a76a62b55d21d67c3665236389a0eee554e1cd19fd57cbd9cca4a7afd`).
  Migration 030
  (`a4ca4fef03aa8da19abebf460464c5ff181785af38821a0968f0d54df6ac7fee`)
  makes the claim procedure normalize itself to READ COMMITTED. The queue
  concurrency/recovery integration cases then passed. Impact: both failures
  occurred before inference; the adjustments remove connection-history
  dependence without changing queue order, lease semantics, or scientific
  inputs.
- 2026-08-23T07:26:00Z — The first post-queue loop completed its three-turn
  hybrid/cache path, but the administrative audit incorrectly compared the
  UTF-8 application tool hash with SQL `HASHBYTES` over `nvarchar` UTF-16
  (failure receipt
  `67761af37bfee0685654597e2dfa3a3844d6212d9796c49e81856e21c252a8dd`).
  The audit now reloads every raw tool file and rehashes raw/transmitted bytes
  in JavaScript. The strengthened eight-route gate passes with receipt
  `8449a07300d761fdac156962f8c9b7056f48f8bd5c96d825745b82ce3ddc10b8`:
  17 model turns, 13 tool calls, three safe decisions, exact repeated-call
  cache behavior, one deliberate snapshot miss, unknown-tool/invalid-argument/
  invalid-contract/same-tool/model-turn rejection, zero live actions, zero
  leases left, and closed/link-complete traces. Its two uncached hybrid calls
  produced exactly two real CUDA embedding requests, two successes, 17 prompt
  tokens, two latency observations, zero errors, and zero preemptions. All
  19 test files/64 tests pass; SQL is 8/8
  (`407d3147ac4fe8898475928f7debb83e878bb0f0500a267ac579192db31ad3b2`),
  tool security is 9 positive/11 negative
  (`fca0eaa2e865c00082141f98527c85b9e6f83f77caeae10a2254c0f4a879126f`),
  and doctor is PASS
  (`2a6f74acb8e0c1a35c06faa437e3565b13df1be26d934823a84ca50bd6466548`).
  Before/after whole-system samples are
  `e4216028fa649ead431ba759f815c46160cdd929af20ba3cd0b92e508b383872`
  and `fc17f756b3268d579923bb153326e4519637e602921c6d4fea1403c315b6afef`.
- 2026-08-23T08:05:00.000Z — Added Gemma 4 26B-A4B (128-expert MoE, ~4B
  active; mlx-community OptiQ 4-bit, 18.8 GB) to the mac bench via
  mlx_lm.server. Under the §A-8 max-tokens override it ties Muse Glimmer 30B
  for best decisions: 100% class, 100% action, 87.5% exact-triple with the
  same two severity judgment-call misses (ep12, ep14), perfect tool
  discipline, and a correct unknown-abstention — at 70 tok/s and 13.8 s per
  episode, four times Muse's speed. mlx_lm.server splits the MoE's reasoning
  channel into a separate message.reasoning field; the harness records this
  as the legal reasoning_field repair. Reasoning cost ~968 completion
  tokens/episode versus ~51 for E4B's direct answers. The MoE becomes the
  recommended primary mac agent arm; report regenerated from rows with the
  seven-model narrative and snapshot updated under docs/reports/.
- 2026-08-23T07:32:00Z — Standard-capture preflight confirmed exactly 600/600
  `standard-v1` items pending, zero prior standard executions, and a final
  immutable planned offset of 8,039,000 ms. Review of the injector before the
  long launch found that row idempotence was correct but a restarted process
  would create a new timing origin after skipping completed rows. The injector
  now reconstructs the original origin from the earliest durable execution's
  start time and planned offset; elapsed offsets execute without a second wait,
  while future offsets retain the original schedule clock. The receipt records
  whether it resumed and the reconstructed origin. Four focused timing tests
  pass. Impact: no standard episode had run, so scientific data are unchanged;
  the adjustment prevents restart downtime from stretching or re-spacing the
  governed schedule.
- 2026-08-23T07:34:15Z — Launched the governed 600-episode `standard-v1`
  capture from clean checkpoint `89b42ff` under retained injector session
  `75344`, continuous telemetry session `46061`, and 20-minute database/Git/
  Drive watchdog session `92148`. The before-capture whole-system receipt is
  `b7ad8cdbb1bd28d70e835057f1c807a26cd54acee56bc02cc53aa41026acc740`;
  the watch epoch is `6cd96b0d-7555-446c-aeb6-bc813a0231d1`. Early health at
  ordinal 4 was 5/5 terminal, zero failed/unclean, and zero XE dropped events,
  dropped buffers, or blocked-event-fire time. The Qwen EngineCore remains
  resident at 6,042 MiB; standard injection is intentionally a SQL/CPU capture
  phase and does not require chat-model GPU load.
- 2026-08-23T07:46:56Z — While capture remained healthy at 60/60 terminal,
  zero failed/unclean, and zero XE loss counters, prepared the chat inference
  residency and port-gate path without loading a chat model or altering the
  active database. The fixed Lab-3 container lifecycle pins image/model/
  revision/tokenizer/served name, nested-Colab CDI, host networking, cache
  volumes, disk floor, max sequence override, and effective server arguments.
  The port gate retains health/models/metrics/service logs, exact raw request
  and response bodies, complete requested and effective decode evidence,
  sequential/batched canaries, usage/finish/repair data, GPU snapshots, and
  journal-to-SQL telemetry. Migration 031 is authored but deliberately not yet
  applied during capture; it adds separate reasoning-channel content/hash/byte
  provenance, chat/embedding service-instance and phase attribution, and
  evaluator-only held-out retrieval result rows. The raw response envelope
  remains the authoritative superset. All 22 test files/73 tests pass. Impact:
  implementation-only; no scientific packet, target inference, or active
  capture row changed.

- 2026-08-23T09:10:00.000Z — Added Muse Glimmer 30B via its platform-native
  MLX serving (mlx-community/Muse-Glimmer-30B-4bit through mlx_vlm.server
  0.6.12; Muse is multimodal, so its MLX architecture lives in mlx-vlm, not
  mlx-lm). Result: an exact episode-for-episode replication of the Foundry
  ONNX bundle's decisions — 100% class, 100% action, 87.5% exact-triple with
  the identical two severity judgment calls (ep12, ep14), perfect tools,
  correct unknown-abstention — at 22.6 tok/s versus 16.4 (38.4 versus 52.1 s
  per episode). The best-decisions result is now a three-way tie (Gemma 4
  MoE, Muse ONNX, Muse MLX) sharing the same two misses, pinning those as
  ground-truth ambiguity; judgment surviving two quantizations and two
  serving stacks unchanged is the bench's strongest robustness evidence.
  The MLX serving replaces the ONNX bundle as the Muse reference arm.
  Report regenerated from rows (eight servings, runtimes labeled); snapshot
  updated under docs/reports/.

- 2026-08-23T08:04:08Z — Integrated upstream Mac commit `8459584` (Muse
  Glimmer MLX serving and eight-serving report) into the live Lab 3 branch.
  Its report snapshot had retained a seven-model cardinality assertion; updated
  that invariant to eight and proved the complete 8 x 16 grid. Added resumable
  held-out retrieval evaluation over packet-visible queries with lexical,
  vector, hybrid, oracle, and deliberately wrong shuffled controls; every cell
  is evaluator-only, retains SQL/raw/journal evidence, and uses unique-runbook
  recall, MRR, nDCG, no-answer accuracy, and latency metrics. Added a fail-closed
  exact-search freeze gate and deterministic paired scenario-group power
  simulation with Holm correction and an explicit DESIGN_ONLY state until
  qwen-smoke dev pairs empirically calibrate dependence. No migration or
  scientific evaluation ran during the active capture. The capture remained
  healthy at 131/600 executed, zero failed/unclean, and zero XE loss counters;
  all 24 test files/77 tests and TypeScript build pass. Durable implementation
  commit: `91c5e7d`.
- 2026-08-23T09:55:00.000Z — Enabled the speculative decoding Muse Glimmer
  ships with: mlx-vlm upgraded 0.6.12 → 0.6.15, which natively supports the
  official DFlash block-diffusion drafter (meta-models/
  Muse-Glimmer-30B-assistant, model_type muse_glimmer_assistant, 5 layers,
  16-token blocks; 5.1 GB) via --draft-model. The speculated serving decodes
  36.9 tok/s effective (23.9 s/episode) versus 22.6 (38.4 s) plain and 16.4
  (52.1 s) for the Foundry ONNX bundle — +63% from speculation, 2.2× the
  ONNX serving — with decisions identical across all three Muse servings
  (verification is lossless at temperature 0), including the same two
  severity judgment calls. 0.6.15 also splits Muse's reasoning channel into
  message.reasoning_content, which the harness now reads (legal
  reasoning_field repair). The plain-MLX run is retained as
  muse-glimmer-30b-mlx-nospec-ablation; the headline muse-glimmer-30b-mlx
  row now carries the drafter, disclosed in the registry (draftModel field)
  and report. This is Muse's practical speed ceiling on this stack.

- 2026-08-23T08:56:00Z — Froze executable B0/B2/B3 and replay-router policy
  while standard SQL capture was active, before packet construction, target
  inference, target scoring, or campaign freeze. B0 is recomputed from only the
  predeclared 60-row dev role (benign/no-action floor, empirical confidence
  0.5); B2 uses the first returned runbook's frozen class/severity metadata and
  a frozen action map; B3 copies protected truth only under the evaluator and
  remains ineligible for application claims; A-router composes resolved B1 or
  the profile-matched A-tools row without another model call and attributes the
  selected source's cost. This late executable-policy addition cannot alter
  capture, but its timing is disclosed in the preregistration and hashed into
  the campaign inputs. Merged Mac OLMo/report commits `8736c9f` and `ecec5f4`;
  their new ninth serving exposed a stale 8x16 snapshot assertion, corrected to
  the retained 9x16 grid. Full TypeScript build and 29-file/96-test suite pass.

- 2026-08-23T10:40:00.000Z — Added OLMo 3.1 32B Instruct
  (lmstudio-community MLX 4-bit via mlx_lm.server; the frozen campaign
  target's base family, served as a different artifact and therefore
  mac-plane evidence only). It joins the 87.5% exact-triple leaders as the
  first direct answerer: 100% class, 100% action, perfect tools, correct
  unknown-abstention, 80% first-pass contract, ~161 completion tokens and
  8.1 s per episode at 19.8 tok/s — the best episode economics at leader
  quality, making it the recommended primary mac arm. This revises the
  reasoning-channel pattern: rubric fidelity and abstention come from
  reasoning or sufficient scale. Ground-truth flag: ep12's authored severity
  (high) is undercalled to medium by all four leaders independently; review
  before the episode set seeds a larger catalog. Display note: the retired
  Foundry Muse serving now renders in neutral gray — the validated palette
  has eight categorical slots, and retirement keeps the ninth series honest.

- 2026-08-23T09:14:55Z — Froze and pushed executable Tier 1 inference controls
  in commit `764c9c3` while standard SQL capture was active at 483/600
  executions with zero injector errors, before packet construction, target
  inference, target scoring, or campaign freeze. The 96-cell masking control
  removes frozen error numbers and exact signatures from both packet and tool
  results; the 96-cell shuffled-runbook control clones evaluator-only wrong
  retrieval rows into provenance-distinct agent-visible rows; and the first 48
  frozen cells re-run A-tools sequentially for batching invariance. Control
  jobs, predictions, raw exports, telemetry, scores, and receipts are isolated
  from primary reporting and link to their exact primary prediction. A
  comparison ledger records exact raw-response, decision, ordered-tool-call,
  transmitted-result, and raw-result agreement plus action-score and confidence
  deltas. The label-permutation policy fixes family/split strata,
  scenario-group units, seed 20260823, and 1,000 repetitions for the later
  no-inference analysis driver. Migration 032 is authored but intentionally
  unapplied until capture closes. Full build and 30-file/98-test suite pass.
  Impact: no captured row or model output changed; the late pre-freeze timing is
  disclosed and hash-bound, and control rows cannot contaminate primary views.
- 2026-08-23T09:54:00Z — The governed `standard-v1` capture completed all
  600/600 episodes with zero failures (injection receipt
  `df6883fc388516a71cabd3064208b6259bd2ed1bca6c6915f106872386301a58`).
  Final source drain retained 7,305 new XE rows and 1,246 new ERRORLOG rows;
  verification passed 600/600 (`46a737b0e184030b84762ace40d2ed184a1b7b0253aff343285b6dc9a898df70`),
  packet construction retained 600 packets
  (`f32c6913c7c9a33e920b3b3bff26307535afe4e659770c392913a866eab208cd`),
  and the protected-truth audit found zero leakage
  (`5c9e0025ce73af0eb0d940dbe5480d0d00899819f55cbd24d18e0f785aa54b7f`).
  Migrations 031 and 032 were then applied and the doctor, 104-test unit suite,
  eight SQL integration cases, and 20-case tool-security matrix all passed.
  The first all-role retrieval-evaluation attempt selected 610 packets because
  its run-level query also admitted ten retained `smoke-v1` development
  packets. It was stopped after 240 packets/1,200 evaluator-only cells, before
  search freeze or any target inference. Campaign capture, retrieval, baseline,
  replay, Qwen gate, and freeze queries now require membership in schedule
  `standard-v1`; the earlier smoke evaluator rows remain immutable but are
  excluded from all standard evidence and reporting. Impact: no packet, label,
  or model output was changed, and no target output existed; the adjustment
  restores the preregistered 600-episode population without deleting useful
  development evidence. The intentional process stop left three open spans in
  its append-only telemetry file. A new generic recovery command validated the
  existing hash chain, appended an explicit interruption point plus child-first
  `interrupted` ends under a distinct recovery process epoch, and revalidated
  the closed journal; no prior record was edited. Recovery receipt:
  `7b3012a8155de982a8e011de5379787fffde3302d8cecdb7e39fe68cf92b9f66`.
  The first retained-table recomputation then stopped on a hybrid nDCG check.
  Diagnosis proved the persisted metric (`0.639945`) correct: the verifier's
  JSON helper had removed duplicate runbook IDs before rescoring, shifting
  relevant items from ranks 1/3/5 to 1/2/3 and producing `0.722727`. The parser
  now preserves duplicate rank positions and the regression suite asserts that
  contract. Impact: validation logic only; retained rankings and metrics were
  not rewritten. The next search-freeze attempt stopped before mutation because
  SQL Server forbids the gate's `SUM(CASE ... EXISTS(...))` aggregate shape.
  The same invariant now joins a pre-aggregated set of agent-visible retrieval
  runs and sums a scalar flag. Impact: query-shape compatibility only; the
  corpus remained unfrozen and all evidence rows were unchanged.
- 2026-08-23T10:12:00Z — The first real Qwen chat port gate reached the pinned
  CUDA/vLLM service but stopped on one synthetic authentication canary. Its raw
  response contained the right decision semantics yet nested `correlationKey`,
  `summary`, and `rationale` inside `actionArguments`, so the strict contract
  rejected exactly three missing top-level fields. No agent campaign or target
  inference had run and the A-direct/A-rag/A-tools identities were not yet
  registered. The operating prompt now states explicitly that those three
  fields are required top-level siblings and that `actionArguments` is reserved
  for action-specific parameters. The schema and deliberately narrow legal
  repair set remain unchanged. Impact: pre-freeze prompt clarification exposed
  by a synthetic canary; the failed raw output remains retained and cannot enter
  scores.
- 2026-08-23T10:14:00Z — The first qwen-smoke replay invocation stopped before
  job creation or model transport because its top-level execution reached a
  role-count constant declared later in the module's temporal dead zone. The
  five frozen standard-role cardinalities now live in the pure replay module,
  and a unit test asserts 120 dev+calibration and 480 protected test episodes.
  Impact: orchestration initialization only; zero jobs, agent runs, predictions,
  or model requests were created by the failed invocation.
- 2026-08-23T10:16:06Z — The next qwen-smoke replay invocation stopped in its
  interrupted-work recovery preflight because a joined SQL Server UPDATE did
  not qualify `completed_at_utc`. Both target aliases are now explicit and a
  source contract prevents their regression. Impact: the invocation had
  materialized its deterministic 180 replay jobs but stopped before workers or
  model transport; those pending identities are reused by the corrected replay,
  with no inference observations discarded or repeated.
- 2026-08-23T10:17:54Z — Queue isolation stopped the corrected replay before
  workers because one previously verified synthetic HTTP-error gate fixture
  remained intentionally `retryable_failure` even though its parent job was
  failed. The SQL model-client gate now retires verified retry semantics to
  `stopped` after verification and its recovery covers terminal parent jobs.
  Impact: zero qwen-smoke model calls in this invocation; the 180 pending replay
  cells remain unchanged, while synthetic gate evidence and its state transition
  history remain retained.
- 2026-08-23T10:20:00Z — The first queue-hygiene gate invocation stopped during
  startup recovery because `ops.work_items.next_attempt_at_utc` is non-nullable.
  Retirement now preserves that historical retry timestamp and changes only the
  terminal disposition, completion time, and lease fields. Impact: no fixture,
  agent, or model calls were created; the prior synthetic row remains available
  for the corrected idempotent cleanup.
- 2026-08-23T10:20:24Z — The corrected SQL model-client gate passed all nine
  transport/contract/retry cases, recovered one historical retry fixture, and
  retired the newly verified retry fixture from the shared queue; receipt
  `c0ec0b01be9ffea68e955aed5c3eb59c0525c7f93dcd9d62f964e9000ce34c56`.
  Impact: synthetic retry semantics remain evidenced without leaving claimable
  work for campaign workers.
- 2026-08-23T10:21:31Z — The first real 180-cell qwen-smoke replay retained 190
  successful vLLM request/response pairs but produced zero accepted decisions:
  30 direct contract rejections, 30 direct tool-policy rejections, 60 RAG
  tool-policy rejections, and 60 tools-arm argument-policy rejections. Raw
  outputs show that the prompt exposed argument names without types/patterns and
  did not state the arm behavior strongly enough: Qwen requested database tools
  in A-direct/A-rag and invented `databaseName` values in A-tools. Two large
  worker journals also contended during concurrent serializable projection; both
  validated and replayed serially (2,371 records; 2,121 inserted, 250 duplicate).
  Impact: this is a pre-freeze development pilot, not target evidence. Contract
  v2 now hashes arm-specific directives and exact JSON Schemas/examples;
  inference remains parallel while journal projection is serialized. An audited
  development-only retry retains attempt 1, refuses frozen/scored rows, and
  selects the current prediction-linked attempt in the end-to-end gate.
- 2026-08-23T10:43:01Z — Qwen pilot attempt 2 completed all GPU/agent work and
  serialized journal projection, then its final constrained-transport verifier
  decoded UTF-8 request bytes as SQL Server UTF-16 and rejected the leading byte
  sequence. The verifier now uses `varchar(max)`, matching the raw-body hash and
  export paths. PASS receipts also carry the prediction-linked agent inference
  window rather than a later no-op finalization window. Impact: no inference is
  rerun; 180 current predictions, 129 accepted decisions, 51 retained failures,
  267 successful tool calls, 60 retrieval calls, and every raw attempt remain
  unchanged.

- 2026-08-23T10:09:07.304Z — Retrieval evaluation 5cf9857b-d6c6-4c90-975f-47aa41ed1353 retained 3000 evaluator-only cells over 600 packets (dev,calibration,test_id,test_variant_holdout,test_unknown); disposition PASS; receipt 894e0de96982643ec2a18eb9e5c75d5a128590869955db95ef78cf3b0e6529bd.

- 2026-08-23T10:09:59.161Z — Search corpus primary-v1 frozen at 2026-08-23T10:09:59.154Z; 60 runbooks, 480 chunks, 480 embeddings, 3000 held-out retrieval cells; freeze ccf8fedf47aa5f83cbd8fda7116934a47481c304f193c5cb946b3a9a51024837.

- 2026-08-23T10:10:21.497Z — B1 B1-rules-v1 produced 120 dev,calibration predictions at {"dev":{"episodes":60,"resolved":60,"coverage":1},"calibration":{"episodes":60,"resolved":60,"coverage":1}}; receipt ed5a28d2a8e70c55fdf1164a979a01a1e716cd22adfa72a9780c645cbde3fdf3.

- 2026-08-23T10:10:30.952Z — B1 B1-rules-v1 produced 60 dev predictions at {"dev":{"episodes":60,"resolved":60,"coverage":1}}; receipt 85571222d2ee54f65e34f5ea44030065120eeadb5becfae33857022d325ca8bf.

- 2026-08-23T10:10:50.780Z — Derived 600 predictions for B0-majority-no-action-v1,B2-lexical-v1,B2-vector-v1,B2-hybrid-v1,B3-oracle-packet-v1 over dev,calibration; receipt 4569707de9cfceda7a628b00d55e2684054466ebc9aa7668256b773a222903d1.

- 2026-08-23T10:13:28.807Z — Scored 60 primary dev predictions for qwen-smoke across B1-rules-v1; receipt 21c0e8a35e752aa2522eced5c6733cca25ad56b085eb81713601eb8e681bad8c.

- 2026-08-23T10:35:42Z — Prompt-contract v2 passed the real qwen-smoke
  port gate, including an A-direct packet that advertised untrusted tool hints:
  9/9 structured decisions, zero length/error/preemption outcomes, deterministic
  repeat rate 1.0, and 81,952 MiB resident GPU memory; receipt
  `49fa3bd3e0afa5dd27d3ee935fb4c929a2bdbb0b5e421f5a98feb8f7644dfee6`.
- 2026-08-23T10:35:51.429Z — Retained the failed qwen-smoke attempt-1 grid and its 190 raw model request/response hashes at /content/worktrees/aidataapps-logwarden/aidataapps/logwarden/runs/logwarden-smoke-20260823T031714Z/manifests/qwen-smoke-pilot-retained-20260823T103551249Z.json; reset exactly 180 development-only cells for prompt-contract v2 attempt 2, receipt 51ec9210c54de714a70a3588908dec4988da90cf1566bda800350ddd75848e1b.

- 2026-08-23T10:44:51.160Z — qwen-smoke primary replay retained 180 cells across dev and A-direct,A-rag,A-tools with 129 decisions, 51 failures, and 607 model requests; receipt 068919dc7799c0d5cdb4e763151ae8f016ed573557df0a1ba49a00a0109fceaa.

- 2026-08-23T10:45:13.791Z — Scored 240 primary dev predictions for qwen-smoke across B1-rules-v1,A-direct,A-rag,A-tools; receipt 63bfe6ed838cc1c4dc813f4358966ce12a15b1a13e96011b43f8a85c258eeb2b.

- 2026-08-23T10:45:33.363Z — Real qwen-smoke end-to-end replay gate passed 180 dev agent cells plus 60 B1 rows with complete raw/SQL/trace/service/GPU provenance; empirical power pairs 9b9969df74b938ce730cf5e60728a16db0b4ce304c10b0d12c7256e40f2ea8ba, gate c82e47987f78e6093691939b0370199dad9ddee1b37f8fc7b7da276be47d760c.

- 2026-08-23T10:46:04.439Z — Paired grouped power simulation completed with PASS disposition; receipt 1190731ab7ea182043522d123726a67efc0e21e8d0ff98ba899431cbe208335c.

- 2026-08-23T10:52:44.689Z — The final pre-freeze telemetry audit found one
  open `sampler.sample` span in the retrieval systems journal, left when that
  sampler was intentionally interrupted after retrieval evaluation completed.
  Recovery validated the original hash chain, appended a child-first
  `interrupted` close under recovery epoch
  `350db56d-4178-4081-b346-286d9f04a6d3`, and projected the two appended
  records with all 475 original records recognized as duplicates. Recovery
  receipt: `70dc5aa87ba469f670905a396bee5d8e494dcde31b4a4b7776bb539cf0ce61b8`;
  ingestion receipt:
  `baa365632e1ff34caaac1df9f6a97d27f721f9a213324c4c9e017d0cb7e0fb0e`.
  Impact: interruption provenance was added append-only; no inference,
  retrieval, metric, or prior telemetry record was changed.
- 2026-08-23T10:52:53.783Z — Global telemetry reconciliation passed over 118
  journals, 36,630 hash-chained records, and 1,428 raw artifacts. SQL exactly
  matches 2,110 traces/9,595 spans with zero open rows, 664 model
  requests/responses with zero bad hashes, 973,790 unique metric samples, and
  5,009 unique raw metric snapshots; no gate work or agent run remains open.
  Input-set hash:
  `cf27d6491910dd75aea8e1b0d55d418ca31a5ea56807ff8612b876cb1172a1f5`;
  reconciliation receipt:
  `2a5f0a4e39bb2831eec506067de3186ab2eb1aeb2965e322161b69f5fb0287c8`.
- 2026-08-23T10:58:59Z — The completed qwen-smoke chat service was removed to
  free the GPU for governed targets. Docker's force-removal waited indefinitely
  on an orphaned chat `EngineCore` process (PID 307208) still holding 75.9 GiB.
  After verifying that PID was distinct from the Qwen embedding `EngineCore`
  (PID 173774), the orphan was terminated and Docker completed removal. SQL and
  the embedding service remain healthy/resident; only the embedding process now
  holds 6,042 MiB. Impact: operational teardown occurred after all Qwen
  requests, receipts, power evidence, sampler shutdown, journal ingestion, and
  reconciliation; no scientific evidence was lost or rerun.
- 2026-08-23T11:07:46Z — The first pre-freeze baseline-identity
  synchronization invocation stopped during its read-only inventory query
  because it scoped predictions by a nonexistent direct `campaign_id` column.
  Campaign ownership is carried by each prediction's job, so both inventory
  guards now join `control.jobs` and constrain that owning campaign. Impact:
  the failure occurred before archive creation, identity updates, evidence
  insertion, or freeze mutation; all database and scientific rows remained
  unchanged.
- 2026-08-23T11:08:43Z — The corrected identity synchronization again stopped
  before archive or mutation because its guard used `completed` instead of the
  schema's terminal job state `complete`, conservatively counting all 720
  completed baseline jobs as active. The guard now recognizes only `pending`
  and `running` jobs as active and uses the queue schema's exact terminal work
  states. Impact: the false-positive stop changed no identity, prediction,
  score, evidence, or freeze row.

- 2026-08-23T11:09:28.066Z — Audited pre-freeze metadata synchronization updated 6 non-model arm identities after prompt-contract v2; retained 720 unchanged dev/calibration predictions, archive 5a8a6bc4b50fe3e9c1029687a5f85d0a66eb5a096e9f690461042ae0d6f6944b, receipt d055460f8c2483a082df7761b6445790ddf0e7849844c8a71cd35184658e45d2.
- 2026-08-23T11:09:51Z — The preceding campaign-freeze attempt had failed
  closed at the first stale non-model arm hash, before thresholds, manifests,
  freeze rows, status promotion, or target authorization were written. The
  completed synchronization proved that all six affected arms differed only in
  agent-contract metadata (and prompt-facing `runbook_search` schema metadata
  for B2), while prompt markers, executable policies, packet/retrieval modes,
  decode identity, and code paths were unchanged. It also proved zero protected
  predictions, governed-target predictions/samples, active jobs, or active work
  and retained the exact 720-row dev/calibration grid. All ten current campaign
  arm hashes now match the registry. Impact: no prediction or score was changed;
  the database checkpoint must be regenerated because the audited identity and
  evidence rows were added after the prior restore test.

- 2026-08-23T11:10:44.648Z — Standard campaign frozen as e105cfdd5af5345464853406c8d707018232f3326c909ca476970fb7137cbcf6 at Git 98d70dca82db1d9879f9f05227af59a2b7127ee4; 600 packets, control subset 225e6f1d88aefe64cf7d07139155cab8d32e337770004acd22f361f23986ed8a, DB checkpoint 20a468ccbb80f5ada685f74beade3a0919655fc5c8bfe914f539514aa6e9fd06.
- 2026-08-23T11:35:57Z — The first governed Muse residency stopped before
  HTTP readiness or any model request after cold-cache population. The exact
  frozen image downloaded the 55.46 GiB checkpoint, loaded 52.07 GiB of
  weights, compiled for 64.63 seconds, created an 18.94 GiB KV cache, and then
  exited 1 while the EngineCore constructed its second tokenizer instance.
  The terminal exception's generic text suggested installing SentencePiece or
  tiktoken, but an exact-image diagnostic proved both packages installed and
  `AutoTokenizer.from_pretrained` passed at the pinned revision with the
  202,048-token vocabulary. Lab 2's retained Muse record independently shows
  the same cold-cache tokenizer race followed by a passing unchanged warm-cache
  restart. This attempt is therefore classified as a cold-cache population
  race, not a missing dependency or model substitution. The interrupted
  one-hour readiness wait was closed append-only under recovery epoch
  `69daf32f-4300-4cae-88c4-f6ddb5ab3e08`; recovery receipt
  `9e6a16bc9e0eae79aefe960350d26d204fb554ff7a3dbd1b63e836950ec0f097`.
  Full timestamped service logs and terminal container state are retained in
  `chat-service-stop-muse-glimmer-30b-20260823T113557166Z-8f57a0e9f065.json`
  (file SHA-256
  `9998bf22d5a25689e68d8ab1056bf3fa6c442ea46963ebc9109c6be0e74df5c6`,
  decoded-log SHA-256
  `93e5072813eecd78c92c2ae009e89b9d5095557eed217d4b77a01d9071850c1a`).
  Post-freeze operational hardening now checks Docker state during readiness,
  retains startup-failure logs immediately, and names stop receipts uniquely;
  it changes no packet, prompt, decode value, image, model, revision, arm,
  threshold, or score. The retry uses the exact frozen configuration against
  the now-complete pinned cache.
- 2026-08-23T11:41:23Z — The unchanged warm-cache Muse service reached HTTP
  readiness and returned nine 2xx canary responses; every response passed the
  structured-decision, token-usage, finish-reason, transport, GPU-residency,
  and metric-delta checks. The gate then emitted `STOP_PORT` because its log
  scan treated any Python traceback as fatal. The retained service log contains
  66 warning-prefixed TorchInductor stacks for absent temporary Triton cubins,
  followed by an explicit successful compiled-graph-cache fallback and
  application startup; it contains zero explicit engine-initialization, OOM,
  fatal-Python, segmentation, or NCCL failure signatures, and the service
  remained healthy. Failure receipt:
  `852334f03e39f3d1e45fe99df99738cb16fecb34f0375b460d45d50805970e85`;
  its 66-record telemetry journal was hash-validated and projected with zero
  duplicates. The post-freeze gate classifier now enumerates explicit fatal
  signatures, rejects those signatures, ignores the evidenced recoverable
  compiler warning stack, and re-inspects the container after the log scan.
  Regression fixtures cover both cases. Impact: validation classification
  only; no service setting, model output, prompt, packet, decode, arm,
  threshold, or score changed. The false-positive canaries remain retained and
  the required warm port gate will be rerun.
- 2026-08-23T11:47:46Z — Muse warm port validation passed twice on the exact
  frozen service identity. Each pass completed 9/9 requests with 9 stop
  finishes, zero length/error/preemption outcomes, 6,738 prompt tokens, 2,331
  generation tokens, six of six identical repeated content hashes
  (`0e8149ae6f969b65ca9a18c9adb860806469809e98b82fe390939ad3d5343255`),
  and 81,722 MiB resident GPU memory. Explicit fatal-log signatures were empty
  and the post-log container inspection remained running. Gate receipts:
  `4fd5df4af5c5fe6a3d73cfa759bb2de372e59e027687ad144b88f15952a6ad56`
  and
  `4fa35f3fd6dba3cf83d3cdf07b087e753e72e2040beedfe6ecce623af2ac16b3`.
  The cold-start result remains the separately retained pre-request tokenizer
  race; the settled-cache repeat establishes the usable governed residency.

- 2026-08-23T12:36:04.086Z — muse-glimmer-30b primary replay retained 180 cells across calibration and A-direct,A-rag,A-tools with 166 decisions, 14 failures, and 319 model requests; receipt 8a6cb7cba39bec1cc934b2aa2721ec1d783f636b3577b62c0fe057f07c51f982.
- 2026-08-23T12:37:00Z — The first governed Muse calibration replay was
  configured for 16 workers, but its retained worker journals and receipt show
  that only workers 00, 03, and 12 claimed work (60, 61, and 59 cells);
  the other 13 workers each received an empty initial `READPAST` result and
  exited while 177 selected cells were still pending. Live vLLM evidence
  independently showed exactly three running requests and zero waiting
  throughout the replay. The cause is a client termination bug: a transient
  empty skip-locked claim can occur while concurrent claim transactions hold
  page locks, but the worker treated the first empty result as proof that the
  queue was drained. The replay itself passed all row/hash/trace/lease checks:
  180 terminal predictions, 319/319 successful stop-finished HTTP requests,
  zero length/error/preemption outcomes, and zero open or unlinked evidence.
  Impact: calibration quality inputs and outputs are unchanged, but its
  11:51:35–12:31:29 performance window represents effective concurrency 3 and
  must not be compared as a 16-worker throughput result. No protected test
  prediction had been opened. Before protected replay, workers now probe the
  selected queue through RCSI and use bounded, instrumented deterministic
  backoff when a claim is transiently empty; a truly drained selected queue
  still exits immediately, persistent claim anomalies fail loudly, and replay
  SQL pools are sized above the frozen worker count. Unit and 16-way SQL claim
  gates will be retained before test inference.
- 2026-08-23T12:39:48Z — Post-fix validation passed 38/38 unit-test files and
  134/134 tests, then passed all 8 SQL integration cases. The widened queue
  gate launched 16 claim loops through one replay-sized pool against 17
  isolated high-priority fixtures and returned 16 distinct leases in 189.12
  ms. It directly reproduced three transient empty claims; all three workers
  observed selected claimable work, retried, and acquired distinct leases
  after 82 aggregate deterministic backoff milliseconds. State-transition,
  heartbeat, stale-token, expired-lease recovery, permission, full-text, and
  exact-vector checks also passed, and fixtures were removed. Receipt:
  `0cb7233598d78ccb665d0df8cc06934741ee48eab70e82099b55da2b335eba73`.
  Impact: this is operational queue hardening after freeze, not a scientific
  input change; it prevents silent worker retirement and adds explicit claim
  contention measurements for protected replay and later model residencies.

- 2026-08-23T12:41:22.380Z — Derived 60 predictions for A-router over calibration and muse-glimmer-30b; receipt b2b0bdea164ea862da2da7321f08e5692dbbb51a0acaf4680259a68118699904.

- 2026-08-23T12:41:44.733Z — Scored 240 primary calibration predictions for muse-glimmer-30b across A-direct,A-rag,A-tools,A-router; receipt a9555119068794d857dcefaaa6bf6cbd21f472ecbb87b3f3ee41def3351f3741.

- 2026-08-23T12:41:50.922Z — Fitted and hash-locked 4 calibration-only models for muse-glimmer-30b before test inference; receipt 61b5ccffadacffc05e6a97f0aced1a0db3a7acae25013e6f3c88c4252e5dcb16.

- 2026-08-23T14:19:29.352Z — muse-glimmer-30b primary replay retained 1370 cells across test_id,test_variant_holdout,test_unknown and A-direct,A-rag,A-tools with 1249 decisions, 121 failures, and 2563 model requests; receipt 77c40337aac5ec4379845bcef8076bc5e43652790e65c0cf09daded52f6eaa9d.

- 2026-08-23T14:23:01.778Z — B1 B1-rules-v1 produced 480 test_id,test_variant_holdout,test_unknown predictions at {"test_id":{"episodes":300,"resolved":300,"coverage":1},"test_variant_holdout":{"episodes":120,"resolved":120,"coverage":1},"test_unknown":{"episodes":60,"resolved":60,"coverage":1}}; receipt 8e6f8db71379ebd848d96eb108f83c54d59fa4fd608b0890c0d0b767c3b927b3.

- 2026-08-23T14:23:33.128Z — Derived 480 predictions for A-router over test_id,test_variant_holdout,test_unknown and muse-glimmer-30b; receipt a90d2c7f59557cdac22037b75c829b6659d8bbb86a716f896629e5199c777830.

- 2026-08-23T14:25:38.759Z — Scored 1850 primary test_id,test_variant_holdout,test_unknown predictions for muse-glimmer-30b across A-direct,A-rag,A-tools,A-router; receipt 538311178007a1b5b32e7c2a0ef9c357dd760ea4120d66a1c0923d4cc0eb6c02.

- 2026-08-23T14:26:00Z — The protected Muse primary replay completed its
  12:43:21–13:43:10 inference window and retained all 1,370 governed cells.
  Its 16 workers each claimed 82–90 cells. Four workers encountered a
  transient empty claim, and all four recovered after one instrumented retry
  (20–32 ms), directly validating the post-calibration queue repair under the
  protected load. vLLM served 2,563/2,563 successful HTTP requests with 2,562
  stop finishes, one retained length finish, zero errors, zero preemptions,
  4,675,075 prompt tokens, and 674,049 generation tokens. Verification found
  1,370 terminal predictions, 2,563 linked turns/requests, 1,198 tool calls,
  and zero leases, bad hashes, constrained transports, unsafe actions, open
  spans/traces, or unlinked spans. The one length finish remains part of the
  complete-case evidence; it did not prevent a terminal prediction. Receipt:
  `77c40337aac5ec4379845bcef8076bc5e43652790e65c0cf09daded52f6eaa9d`.
- 2026-08-23T14:26:00Z — The protected A-router derivation initially stopped
  before inserting any router row because its deterministic B1 source had not
  yet been generated for the test roles. Running the missing B1 prerequisite
  after replay exposed an overly narrow B1 guard: it allowed test computation
  only while campaign status was exactly `frozen`, although the first
  authorized target replay advances that same frozen campaign to `running`.
  The guard now accepts `frozen`, `running`, and `complete`, still rejects
  `building`, and has a regression test; 38/38 test files and 135/135 tests
  passed. B1 then produced 480/480 test rows, A-router derived 480/480 rows,
  and all 1,850 protected Muse predictions were scored. Impact: deterministic
  stage ordering and resumability only; no packet, frozen identity, model
  request/output, prediction, calibration fit, threshold, or scoring rule was
  changed. B1/router/score receipts: `8e6f8db71379ebd848d96eb108f83c54d59fa4fd608b0890c0d0b767c3b927b3`,
  `a90d2c7f59557cdac22037b75c829b6659d8bbb86a716f896629e5199c777830`,
  and `538311178007a1b5b32e7c2a0ef9c357dd760ea4120d66a1c0923d4cc0eb6c02`.

- 2026-08-23T14:37:58.791Z — muse-glimmer-30b control error-number-mask-v1 retained 96 cells across test_id,test_unknown and A-tools with 71 decisions, 25 failures, and 277 model requests; receipt 76b628a090b824be401443511ba19c15a4f5caea9fdd6e8522f3f28b2da8cdf1.

- 2026-08-23T14:38:11.845Z — Scored 96 control error-number-mask-v1 test_id,test_unknown predictions for muse-glimmer-30b across A-tools; receipt b3ea6d26400b3922467a00455f2df7a44611f4f6b705a9d77582022dd6e9916c.

- 2026-08-23T14:38:21.186Z — Compared 96 muse-glimmer-30b/error-number-mask-v1 predictions with their frozen primary sources: raw=0/96, decision=0.19697/66, tools=0.229167/96, invariance=NOT_APPLICABLE; receipt a566c9b7046a69446a1c71f3c4b01d65386845f2360781137d2e20d2bd27fe42.

- 2026-08-23T14:47:40.457Z — muse-glimmer-30b control shuffled-runbooks-v1 retained 96 cells across test_id,test_unknown and A-tools with 80 decisions, 16 failures, and 291 model requests; receipt 5d17e80fdec7e829078381bbf0f78ab9e0d3f8329a3df4643d521c46e1e89f6a.

- 2026-08-23T14:47:52.778Z — Scored 96 control shuffled-runbooks-v1 test_id,test_unknown predictions for muse-glimmer-30b across A-tools; receipt 9512b4cc223e2dd764689c38e9b789dcfb0eb75be38e76fce370660bfcab59a3.

- 2026-08-23T14:47:57.732Z — Compared 96 muse-glimmer-30b/shuffled-runbooks-v1 predictions with their frozen primary sources: raw=0/96, decision=0.263889/72, tools=0.479167/96, invariance=NOT_APPLICABLE; receipt 750f09ae8244eb9aceee4f9245c9b090cdbe9a01e5aa3a6c850844d0c529f7bb.

- 2026-08-23T15:30:07.501Z — muse-glimmer-30b control batching-sequential-v1 retained 48 cells across test_id,test_unknown and A-tools with 44 decisions, 4 failures, and 129 model requests; receipt cddb387bc8a5cfb62c26a755d47ac7f6a4713e5f229789b737923850704ddbb2.

- 2026-08-23T15:30:17.840Z — Scored 48 control batching-sequential-v1 test_id,test_unknown predictions for muse-glimmer-30b across A-tools; receipt d503dd55f65fffbac10f79b81b3fb2e6d692f9339128855f307966a46c318d80.

- 2026-08-23T15:30:20.617Z — Compared 48 muse-glimmer-30b/batching-sequential-v1 predictions with their frozen primary sources: raw=0/48, decision=0.209302/43, tools=0.604167/48, invariance=NON_INVARIANT; receipt c70563114783f688c794a6452c2a3a1da1e8465fa68bd2b5f086f6c190e36262.

- 2026-08-23T15:36:36.209Z — Diagnosed muse-glimmer-30b/batching-sequential-v1: AGENT_INPUT_DRIFT_REQUEST_LEVEL_INVARIANT; exact-input normalized outputs 48/48, first-turn choices 48/48; receipt 7810da399de51d80e99b133a8490feff11ab7ed3b24821429786feaac43c298d.

- 2026-08-23T15:37:00Z — The frozen batching control's literal comparison
  labeled the full agent pipeline `NON_INVARIANT` (0/48 raw response bodies,
  9/43 decision hashes, 38/48 semantic decisions, and 29/48 ordered tool-call
  sequences matched). A post-control audit found that this label is
  confounded by execution identity rather than evidence of a vLLM batching
  defect: raw response bodies contain new response IDs/creation times; every
  tool-result prompt contains a new `agentRunId:toolInvocationId` call ID; and
  runbook results contain a new SQL retrieval-run ID. The new diagnostic keeps
  the original comparison immutable, compares generated choices/usage after
  removing only response-envelope identity, and separately normalizes the two
  execution-local prompt fields. All 48/48 byte-identical first-turn requests
  produced identical normalized choices, content, and reasoning across the
  original 16-worker and sequential runs; all 48 exact-request pairs across
  the trace had identical normalized output, with zero exact-input output
  mismatches. Classification:
  `AGENT_INPUT_DRIFT_REQUEST_LEVEL_INVARIANT`. Impact: Muse request-level
  batching invariance is supported for observed identical inputs; full-agent
  batching invariance is not identifiable from this control and must not be
  claimed. No retained request, response, prediction, score, or original
  comparison was rewritten. The diagnostic is stored as a hashed JSONL table,
  metric receipt, and SQL evidence event; 39/39 test files and 137/137 tests
  passed. Diagnostic receipt:
  `7810da399de51d80e99b133a8490feff11ab7ed3b24821429786feaac43c298d`.

- 2026-08-23T16:27:08.372Z — gemma-4-31b primary replay retained 180 cells across calibration and A-direct,A-rag,A-tools with 180 decisions, 0 failures, and 250 model requests; receipt e6a9cbc9cc0ed523b9751c6d6db25a8342db8d29bb5f2657ce8f69e310e844d7.

- 2026-08-23T16:27:38.698Z — Derived 60 predictions for A-router over calibration and gemma-4-31b; receipt a9891d4e5224dce842a700fba7df56448204e7024fd4ff4fcecfc1ae63a6244a.

- 2026-08-23T16:27:56.028Z — Scored 240 primary calibration predictions for gemma-4-31b across A-direct,A-rag,A-tools,A-router; receipt 8ad054196fb90c9c8e26e0cc27fb57f53f608819cde6832d7bbe2fbf33d9ee32.

- 2026-08-23T16:27:56.590Z — Fitted and hash-locked 4 calibration-only models for gemma-4-31b before test inference; receipt cfaae197a0c42323f5f73ea405b6fd3e2dcf4a3e70d0d1d4ae489f0451dbbe83.

- 2026-08-23T16:28:15Z — Retrospective canonical-log consolidation for the
  already receipt-backed 15:40–16:17 profile transition. Muse residency
  telemetry reconciled PASS across 193 journals / 119,595 records / 8,640 raw
  artifacts, with 6,001 closed traces, 31,220 closed spans, 4,243 paired model
  requests/responses, 2,725,869 unique metric samples, and 11,285 raw metric
  snapshots; receipt
  `fd256614a7fe153fbbe17b88acb8a72589ccd5db63050b7d644e38de8fadad2e`.
  The profile checkpoint was pinned at
  `18447e0da29362059b975ae2cbec930bc77b4d3aad864105311de6543ec8e8de`.
  Bounded retention then removed 600 hash-verified redundant local/Drive/SQL
  staging files (28,350,983,266 bytes) while keeping the frozen and pinned Muse
  boundaries plus two rolling database/Git recovery points; receipt
  `ba8843aa7f452b553d14772d334372da2c38641940a158749d3f586bea76eb24`.
  Impact is durability/storage only; scientific evidence is unchanged.

- 2026-08-23T16:28:15Z — Muse's exact rootless container teardown waited on
  its already-idle orphaned `VLLM::EngineCore` PID 419481 after API-server
  removal. With reconciliation and the pinned profile boundary complete, an
  exact SIGTERM ended that process cleanly; SQL and embedding services remained
  healthy. Only reproducible Muse and qwen-smoke weight caches were evicted.
  Impact is post-profile lifecycle/storage only; no output or metric changed.

- 2026-08-23T16:28:15Z — Gemma's exact frozen 0.78 profile downloaded and
  loaded its 58.25 GiB checkpoint, then correctly emitted retained `STOP_PORT`
  before HTTP/model requests: one 16K request needed 13.76 GiB KV versus 13.22
  GiB available (receipt
  `103e7dabd7506d125a60dcd8e5fa4e323072d97539669105c7d425d08c9b6133`).
  Added a separately labeled runtime-only 0.79 memory cap, preserving the
  frozen model/profile hash, weights, revision, image, 16K context, prompts,
  decode, seed, and 64-sequence ceiling. It exposed 15.55 GiB KV and passed two
  nine-call gates with 18/18 stop finishes and zero length/error/preemption
  outcomes; receipts
  `0a9e9adb19f74d70d616c981061ef4acbda51e2820009b5a4e925f5c41f64f01`
  and
  `ed37c268649a713231c585b6a6710fb2a35031cf876fdcdfc557d8d5afbf5b4a`.
  Impact: quality inputs/outputs are unchanged, but Gemma capacity/performance
  rows disclose the +1-point resource cap and cross-profile throughput cannot
  be attributed solely to model identity. Repeated gate exact-output rates
  (1/6 and 2/6) are retained for the governed batching control.

- 2026-08-23T17:03:37Z — During Gemma protected-primary finalization, all
  inference cells had already reached a terminal state while the GPU was idle
  and replay remained in CPU/SQL telemetry materialization. Diagnosis found
  that each worker journal was batch-ingested but its span foreign keys were
  then linked with one SQL round-trip per span. Replaced that post-inference
  loop for subsequent invocations with duplicate-checked, exact-row-counted
  `OPENJSON` batches of up to 1,000 links and added the requested/updated/batch
  counts to each worker receipt. Build plus 41 test files / 144 tests passed;
  an idempotent live-SQL validation updated 1,000/1,000 existing span links in
  one batch in 55.427 ms. The already-running Gemma primary process had loaded
  the prior code and remains untouched, so its evidence and timing preserve
  the original sequential finalization path. Impact is limited to reducing
  post-inference persistence overhead for controls and later profiles; model
  prompts, responses, decisions, inference telemetry, and scientific metrics
  are unchanged.

- 2026-08-23T17:19:32.737Z — Gemma protected-primary inference retained
  1,370/1,370 terminal cells, 1,318 decisions, 52 `decision_policy` /
  `policy_rejected` failures, and 1,938 model requests. All 1,938 requests
  finished `stop`; HTTP/model errors and length finishes were zero. The strict
  final metric gate nevertheless retained `FAIL` receipt
  `368496f8a89c8b9f5e43af2a83b6c33e904c25d62754eae0f0961cf29d059c02`
  because `vllm:num_preemptions_total` advanced once. Sampler and service-log
  evidence place the event just after 16:41:16Z, when 16 requests drove KV
  usage to 96.5%; the counter first appeared in the 16:41:21Z sample and did
  not advance again. The preemption is retained as real performance evidence:
  it may increase affected-request latency/compute and makes Gemma's aggregate
  throughput a resource-pressure result, but vLLM recompleted the request and
  it does not create a missing or retried scientific cell.

- 2026-08-23T17:22:45.331Z — A governed no-op replay resume claimed zero work
  and issued zero new model requests, then reverified the immutable Gemma
  protected rows: 1,370 cells, 1,318 decisions, 52 failures, and 1,938 retained
  model requests. Receipt
  `4b25775a105fa9765d39d27352f88dddfcd249f647d201d60f8584989fa81583`
  is the persistence/coverage PASS and the prior strict metric FAIL remains
  authoritative for the original inference window. No output was regenerated
  or replaced. Quality scoring may proceed; performance reporting must include
  the one-preemption caveat and use the original 16:30:12Z–16:51:06Z telemetry
  window rather than the no-op resume's zero delta.

- 2026-08-23T17:23:43.746Z — Derived 480 predictions for A-router over test_id,test_variant_holdout,test_unknown and gemma-4-31b; receipt 4e58860875c4235f7ce1f73fc185bf95c9615eae266eb3af282c16d5bc238a5e.

- 2026-08-23T17:25:57.913Z — Scored 1850 primary test_id,test_variant_holdout,test_unknown predictions for gemma-4-31b across A-direct,A-rag,A-tools,A-router; receipt 91c0436a027ac454601addbccce5ef85d0d3956b7830576f6cd241e4adf129e5.

- 2026-08-23T17:29:40.890Z — gemma-4-31b control error-number-mask-v1 retained 96 cells across test_id,test_unknown and A-tools with 96 decisions, 0 failures, and 139 model requests; receipt 5c71fca49c2adbce13e461d9e97993a39914dabdd607fd36bf14782661c7fd40.

- 2026-08-23T17:29:55.100Z — Scored 96 control error-number-mask-v1 test_id,test_unknown predictions for gemma-4-31b across A-tools; receipt 45117730861182af6649f037869b3fc9bb64138760589c99a2ee0868d8130b4a.

- 2026-08-23T17:30:06.655Z — Compared 96 gemma-4-31b/error-number-mask-v1 predictions with their frozen primary sources: raw=0/96, decision=0.117021/94, tools=0.802083/96, invariance=NOT_APPLICABLE; receipt a0d34731d2c3432d2843a1ef0f5a1182c5725ca371c7c1fff92e4c2d83d5f8fa.

- 2026-08-23T17:31:51.886Z — Gemma shuffled-runbook inference completed
  all 96 scientific cells (94 decisions, two policy rejections, 131/131 model
  requests ending `stop`, zero error/length/preemption deltas), but three idle
  queue workers were selected as SQL Server deadlock victims during their
  terminal availability probes. Strict orchestration receipt
  `f37125338620584f1f531c3bf0fcb65b150371e11fb69a7b0319dcce9559eac7`
  remains `FAIL`. Workers 3, 6, and 12 had only 3, 15, and 4 journal records,
  respectively, ending in claim-retry points; none claimed a cell and all
  journals had zero open spans. The other 13 workers ended normally. Explicit
  recovery ingested all 2,786 records with zero duplicates under receipt
  `f11cd77611350794b6e83be14cb923f97b64e7eec2d23f5258760f7f57fdba4d`.
  A no-op coverage resume then issued zero new model requests and verified all
  immutable rows under receipt
  `1c58b319c144b2954c55b350f23867bc8c8aecf7f7be0bd1fbe413876c3bbd14`.
  Impact is orchestration-only; no scientific output was regenerated.

- 2026-08-23T17:35:16.412Z — gemma-4-31b control shuffled-runbooks-v1 retained 96 cells across test_id,test_unknown and A-tools with 94 decisions, 2 failures, and 131 model requests; receipt 1c58b319c144b2954c55b350f23867bc8c8aecf7f7be0bd1fbe413876c3bbd14.

- 2026-08-23T17:35:42.501Z — Scored 96 control shuffled-runbooks-v1 test_id,test_unknown predictions for gemma-4-31b across A-tools; receipt 83292c6b0315973ac4dfde9b348082bc0f396ff52d75b84e8e8fd28358e2244a.

- 2026-08-23T17:35:51.706Z — Compared 96 gemma-4-31b/shuffled-runbooks-v1 predictions with their frozen primary sources: raw=0/96, decision=0.489362/94, tools=0.989583/96, invariance=NOT_APPLICABLE; receipt 77fe9f24e48fe3b1b3fc6487c4caba5b6ac6fb7f5863226bbe1ad46c3410b251.

- 2026-08-23T17:37:53Z — Added bounded SQL error-1205 retry only around
  transaction-bounded queue claims and read-only selected-work availability
  probes. SQL Server rolls back the deadlock-victim transaction before error
  1205 is returned, making these exact retries safe; model inference, tools,
  decisions, and external actions are explicitly outside the wrapper. The
  policy allows at most eight deterministic 25–500 ms exponential waits plus
  worker staggering, records every retry in the worker journal, and reports
  per-worker retry/wait totals plus the policy in replay receipts. Build and 42
  test files / 148 tests passed. Impact: later invocations tolerate transient
  queue lock cycles without retrying or changing any scientific cell.

- 2026-08-23T17:46:15.774Z — gemma-4-31b control batching-sequential-v1 retained 48 cells across test_id,test_unknown and A-tools with 48 decisions, 0 failures, and 66 model requests; receipt 4287cfd610a2bc2ae9bb96335148c736618dc509f6fe517e7900e5859b6ee602.

- 2026-08-23T17:46:24.944Z — Scored 48 control batching-sequential-v1 test_id,test_unknown predictions for gemma-4-31b across A-tools; receipt e764af9f6d5aefac97d532d489b024c8dd3a2e48cffdae8071c0f509b1d8e433.

- 2026-08-23T17:46:31.215Z — Compared 48 gemma-4-31b/batching-sequential-v1 predictions with their frozen primary sources: raw=0/48, decision=0.416667/48, tools=1/48, invariance=NON_INVARIANT; receipt fd9e052a8a834b3c55e28647189d927d3d970bb88135bdb610bc661ea1cff317.

- 2026-08-23T17:46:35.867Z — Diagnosed gemma-4-31b/batching-sequential-v1: REQUEST_LEVEL_NON_INVARIANT; exact-input normalized outputs 29/48, first-turn choices 29/48; receipt 1fc735aff7fb324b5a6c0dc85e97b391edb3a6912fe6fb78e79dd940c69d9b44.

- 2026-08-23T17:51:25Z — Closed the Gemma residency epoch
  `bde528a1-bb5c-4b66-91b5-ac19591487dd` and ingested 2,361/2,361 sampler
  records with zero duplicates (receipt
  `4e9f74e6de17d3ca51e9412051fe44d33bf9b5db404b7421198c259064474bd7`).
  Global reconciliation passed across 296 journals / 179,223 records / 13,724
  raw artifacts: 8,583 traces and 46,632 spans are all closed; 6,767 model
  requests have 6,767 responses and zero bad hashes; 3,373,339 metric samples
  and 13,646 raw snapshots have unique keys. Reconciliation receipt:
  `7584e66698da11704ba36bfff4da6c8acbc226b334f1701688e3d6af7be2f5f1`.
  A fresh verified database backup was mirrored and pinned as
  `gemma-4-31b` at receipt
  `317146edecfe64bc209d07ec3e0008179356cdc59d8916cb38fbde4a5b0e15c2`.
  Exact bounded retention then removed six redundant local/Drive files
  (1,605,381,262 bytes) while preserving the freeze and Muse/Gemma pinned
  boundaries plus two rolling recovery points; receipt
  `6bb1a32e5d08c969f078d958d8d9adfd55a74273a461af3edf6051985fa89170`.
  Impact is durability/storage only; all scientific and performance evidence,
  including Gemma's one preemption and deadlock receipts, remains retained.

- 2026-08-23T17:55:49Z — Began governed OLMo residency epoch
  `56559cbd-e997-4cbb-bb09-713ae851d83b` before service startup. After the
  reconciled and pinned Gemma boundary, terminated only the verified-idle
  stuck Gemma EngineCore during rootless-Docker teardown and evicted only its
  59 GiB reproducible Hugging Face cache. The embedding EngineCore and SQL
  remained resident. Impact: storage/lifecycle only; the deleted cache is
  recoverable by digest-pinned redownload and no retained evidence was removed.

- 2026-08-23T18:01:51Z — The pinned `allenai/Olmo-3.1-32B-Instruct`
  revision `ac0587e4a7744a551c059d8cd17ba220bc940dae` became ready under image
  digest `0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967`
  (vLLM 0.27.1). The 60.04 GiB checkpoint downloaded in 183.71 s, weights
  loaded in 6.35 s, full startup/compilation took 114.38 s, and the frozen
  0.78 residency exposed 12.18 GiB / 49,863 tokens of KV cache. Impact: none;
  these are cold-start and capacity measurements for the exact frozen profile.

- 2026-08-23T18:05:09Z — OLMo's cold and warm unconstrained Tier-1 port gates
  both emitted `STOP_PORT`, receipts
  `bf46e7717b67ea7485ac2dae4e7c79c145d54a310f9d07fd802f19be71368be0`
  and `ed330394f62e61dce90a87c65a56ef4b9c55e8a3bda8982ae0f29cd7dbb597d4`.
  All 18/18 requests completed on GPU with finish reason `stop`; every response
  was valid JSON and omitted exactly the same three required decision fields:
  `citedChunkIds`, `confidence`, and `abstain`. This is a first-pass contract
  reliability failure, not a serving or semantic-output failure. Per the
  frozen addendum, no silent repair, prompt change, or constrained decoding is
  allowed in the primary cell, so OLMo receives `STOP_PORT` and no Tier-1
  campaign replay is run. Impact: the four-profile primary campaign will report
  OLMo as port-excluded rather than fabricate comparable quality scores.

- 2026-08-23T18:07:18Z — Added the preregistered Tier-2 `--guided-json` port
  path without changing the primary gate: it sends the same first user turn
  plus a response-format JSON schema generated from the authoritative Zod
  discriminated union, records the actual schema/transport in requests and
  receipts, and persists evidence under a separate Tier-2 stage. Build and all
  42 test files / 148 tests passed. OLMo then passed 9/9 guided canaries with
  9 `stop`, zero length/error/preemption outcomes, 6,946 prompt and 1,252
  generation tokens, and 6/6 exact repeated outputs; receipt
  `c1ee96f0edeb5984351be40ceabc81cfe870945a036efc70cd4118289045ae37`.
  Classification: `GUIDED_DECODING_RECOVERS`. Impact: production-style guided
  decoding demonstrably recovers OLMo's contract, but this Tier-2 result is not
  pooled with or substituted for the unconstrained Tier-1 campaign.

- 2026-08-23T18:11:27Z — Closed OLMo residency epoch
  `56559cbd-e997-4cbb-bb09-713ae851d83b` and ingested 267/267 sampler journal
  records with zero duplicates (receipt
  `c1a045405487609a6b4691a0aeaccd722c1522396835dfaffe61d7a653733f8b`).
  Global telemetry reconciliation passed across 300 journals / 179,688 records
  / 13,778 raw artifacts: 8,675 traces and 46,778 spans are all closed; 6,767
  stored campaign model requests have paired responses and zero bad hashes;
  3,437,229 metric samples and 13,913 raw snapshots have unique keys. Receipt:
  `9266d121068c8aa66220e1cc45d3a73f6eed0e25bf221cc30c870bffe9674730`.
  A fresh verified database backup (receipt
  `e9cf8b0402dcd20d2c498aaddbebb99dd9cfd682df457fc79f2b94b4ec2a18df`)
  and the complete run were mirrored, then the OLMo STOP_PORT boundary was
  pinned. Bounded retention passed with zero eligible deletions (receipt
  `a45985f47721f4f72cdc39569039d8207ed0de412cafd335fddc847e1adb713c`);
  final local/Drive inventory hash:
  `9024161ba18e9cde3ae50ea6052cd91be7279b495bd00f7e379599cf15eab5df`.
  Impact: durability only; both the negative Tier-1 and positive guided Tier-2
  OLMo evidence are preserved at the exact profile boundary.

- 2026-08-23T18:20:53Z — Began Qwen 3.8 residency epoch
  `de4674e9-3cc3-4104-84e9-036075be5f43` before service startup and loaded the
  exact pinned `Qwen/Qwen3.8-27B` revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` under image digest
  `0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967`
  (vLLM 0.27.1). The 51.75 GiB checkpoint downloaded in 156.91 s and weights
  loaded in 5.13 s; engine initialization took 124.14 s including 32.46 s of
  compilation and a 40.10 s hybrid-GDN profiling pass. Frozen 0.78 residency
  exposed 20.10 GiB / 286,720 tokens of KV cache (17.50x nominal 16K
  concurrency). Transformers emitted two processor-docstring warnings labeled
  `[ERROR]`; vLLM explicitly disabled all multimodal inputs, entered text-only
  mode, and the fatal-signature audit passed. Impact: none; these retained lines
  are non-fatal cold-start diagnostics for the exact frozen runtime.

- 2026-08-23T18:23:46Z — Qwen 3.8 passed both cold and warm unconstrained
  Tier-1 port gates, receipts
  `520779ca4ec48f5b0b1d6c245d0bdf2722eb1d47d2f95526f69dffa366cd6efd`
  and `b17b6f4cb8a28594fea036054bbe283538a79c929144d819737eb41313765881`.
  Each gate recorded 9/9 successful `stop` requests, zero length/error/
  preemption outcomes, 7,157 prompt and 1,845 generation tokens, 6/6 exact
  repeated outputs, exact model/revision/image/profile/decode identities, and
  approximately 80,104 MiB total GPU residency. Impact: Qwen 3.8 is authorized
  for governed calibration replay; no resource or decode override is present.

- 2026-08-23T18:43:53.662Z — qwen-3.8-27b primary replay retained 180 cells across calibration and A-direct,A-rag,A-tools with 180 decisions, 0 failures, and 305 model requests; receipt 6954c344a9186a1e73130da74d371f90b944d3683b01f8fa1fa072d0313a11a6.

- 2026-08-23T18:44:19.405Z — Derived 60 predictions for A-router over calibration and qwen-3.8-27b; receipt 5fd6edf4bb7a8c39ca467f192d9a704c75efa15a5f077f52bbbbeceb27d9545e.

- 2026-08-23T18:44:43.384Z — Scored 240 primary calibration predictions for qwen-3.8-27b across A-direct,A-rag,A-tools,A-router; receipt deed862fd9bfd107e2644cb5cd8164cf91d5cebb303fbb7286da9ea8a1874295.

- 2026-08-23T18:44:47.843Z — Fitted and hash-locked 4 calibration-only models for qwen-3.8-27b before test inference; receipt 19ec484c4a714501826ff678a1564877e3c051db585bd6a1e12155c1cfb45d61.

- 2026-08-23T18:45:00Z — Qwen calibration inference itself completed cleanly
  in the original invocation: all 180 cells reached `complete`, all 180
  predictions are decisions, and the 305/305 model request/response pairs
  ended `stop` with zero vLLM length, error, or preemption outcomes. The
  invocation nevertheless returned FAIL (receipt
  `4a46b70d5cc537f9a5f262ff4e02c3bbe062c96fb55a407e2e3d29c60c6f603f`)
  after 11 workers exhausted bounded SQL 1205 retries on the read-only selected
  queue availability probe. `system_health` retained 73 matching deadlock
  graphs with 103 victims from 18:25:39.636Z through 18:26:08.708Z; these
  reconcile exactly to 92 journaled retries plus 11 final failures. The graphs
  show pooled sessions retaining `SERIALIZABLE` after evidence transactions,
  causing probe `RangeS-S` locks on `ix_work_items_claim` to cycle with
  work-item transitions. The probe now explicitly enters `READ COMMITTED`
  inside its own RPC, thereby using the database's existing RCSI policy while
  leaving evidence transactions serializable; source commit `c46fd23`. All 148
  unit tests and the TypeScript build passed, and a deliberately contaminated
  pooled session verified effective isolation 2 inside the repaired probe.
  The 16 original journals contributed 6,781/6,781 new records with zero
  duplicates (receipt
  `987dc6abf9edc67e67b41c569216b922592fa5668e45133591391c905cf4928b`).
  An exact 16-worker resume then passed under receipt
  `6954c344a9186a1e73130da74d371f90b944d3683b01f8fa1fa072d0313a11a6`
  with zero new model requests and zero delta in every vLLM counter. Impact:
  orchestration and throughput reporting only; no packet, prompt, decode,
  model output, prediction, or quality row was retried or changed.

- 2026-08-23T19:42:55.611Z — qwen-3.8-27b primary replay retained 1370 cells across test_id,test_variant_holdout,test_unknown and A-direct,A-rag,A-tools with 1292 decisions, 78 failures, and 2370 model requests; receipt 420b2c74a56028625588b32540dded381265ec4f6fae4d3c76f0eb55f0903f8b.

- 2026-08-23T19:43:27.390Z — Derived 480 predictions for A-router over test_id,test_variant_holdout,test_unknown and qwen-3.8-27b; receipt 104190270f4761ede775eda69b47a3f091c73aa0058074323c46bc40f5a64e5a.

- 2026-08-23T19:45:35.512Z — Scored 1850 primary test_id,test_variant_holdout,test_unknown predictions for qwen-3.8-27b across A-direct,A-rag,A-tools,A-router; receipt ac1a72a6ff621afce68f01ecc901ef5f3006b967c98df5c4b3240810c66b470c.

- 2026-08-23T19:51:21.421Z — qwen-3.8-27b control error-number-mask-v1 retained 96 cells across test_id,test_unknown and A-tools with 92 decisions, 4 failures, and 226 model requests; receipt bfb1b35b06b93572e3183b590c2a01169e2ce0622f27c9aa3f97c27feb6d8b28.

- 2026-08-23T19:51:33.755Z — Scored 96 control error-number-mask-v1 test_id,test_unknown predictions for qwen-3.8-27b across A-tools; receipt 600d5eeca816328f6b01172959c22cb4587ce912d6c340e07d0eff13ad5fb223.

- 2026-08-23T19:51:47.976Z — Compared 96 qwen-3.8-27b/error-number-mask-v1 predictions with their frozen primary sources: raw=0/96, decision=0.058824/85, tools=0.541667/96, invariance=NOT_APPLICABLE; receipt db150170e7eeb0e0fa8243aacecaa66d84ddfcc0ae72756d34f6490013e4bfd1.

- 2026-08-23T19:56:29.249Z — qwen-3.8-27b control shuffled-runbooks-v1 retained 96 cells across test_id,test_unknown and A-tools with 88 decisions, 8 failures, and 212 model requests; receipt e9fc229705d053a52c6ce6ff5072d120bd9d91ca112555c00b125261cd2263ef.

- 2026-08-23T19:56:40.863Z — Scored 96 control shuffled-runbooks-v1 test_id,test_unknown predictions for qwen-3.8-27b across A-tools; receipt eeb5b844ef3bb036735b50e8abb4699a46b0a3cc63337b8eadad5b8932ab60eb.

- 2026-08-23T19:56:50.574Z — Compared 96 qwen-3.8-27b/shuffled-runbooks-v1 predictions with their frozen primary sources: raw=0/96, decision=0.229885/87, tools=0.864583/96, invariance=NOT_APPLICABLE; receipt dedda3d30680cb71e562bed2d8fd056b19afc94fd0cabbf94adc79c60bfe5334.

- 2026-08-23T20:04:46.828Z — qwen-3.8-27b control batching-sequential-v1 retained 48 cells across test_id,test_unknown and A-tools with 44 decisions, 4 failures, and 106 model requests; receipt 09f666af17c1203ea0ef197bd90d7fad15a050cc1e5c146c3f24137a6ca4607f.

- 2026-08-23T20:04:55.421Z — Scored 48 control batching-sequential-v1 test_id,test_unknown predictions for qwen-3.8-27b across A-tools; receipt 35f59af8f7a80b840b2c90710b9f9c7d97b24a852c247aa9ca20e147dd5b89a9.

- 2026-08-23T20:05:02.085Z — Compared 48 qwen-3.8-27b/batching-sequential-v1 predictions with their frozen primary sources: raw=0/48, decision=0.227273/44, tools=0.958333/48, invariance=NON_INVARIANT; receipt af5f5c0c2f7df7df68c12953d3fdd9750a6ae66b8d12cd7344d7c53822575566.

- 2026-08-23T20:05:09.502Z — Diagnosed qwen-3.8-27b/batching-sequential-v1: REQUEST_LEVEL_NON_INVARIANT; exact-input normalized outputs 46/48, first-turn choices 46/48; receipt b53f562588619abc3dec278abe25a1018f7dde82fb09f2f79426053608ef96bb.

- 2026-08-23T20:08:00Z — Closed all three Qwen controls without a model-
  service failure, length finish, or preemption. Error-number masking retained
  96/96 cells (92 decisions, four agent-policy failures, 226 requests), with
  58/96 semantic-decision and 52/96 ordered-tool agreement, a +0.041667 mean
  action-score delta, and comparison receipt
  `db150170e7eeb0e0fa8243aacecaa66d84ddfcc0ae72756d34f6490013e4bfd1`.
  Shuffled runbooks retained 96/96 cells (88 decisions, eight failures, 212
  requests), with 86/96 semantic-decision and 83/96 ordered-tool agreement, a
  -0.041667 mean action-score delta, and receipt
  `dedda3d30680cb71e562bed2d8fd056b19afc94fd0cabbf94adc79c60bfe5334`.
  Single-worker batching retained 48/48 cells (44 decisions, four failures,
  106 requests); 46/48 byte-identical first-turn inputs changed neither the
  normalized response nor tool sequence, while two did. It is therefore
  `REQUEST_LEVEL_NON_INVARIANT`, receipt
  `b53f562588619abc3dec278abe25a1018f7dde82fb09f2f79426053608ef96bb`.
  Impact: these are measured sensitivity/repeatability outcomes, not runtime
  adaptations; every original and control row remains immutable.

- 2026-08-23T20:11:00Z — Operator SIGINT stopped the Qwen residency sampler
  after all inference and controls but between the final sample's durable
  `span_start` and `span_end`. The first 2,581 records ingested exactly under
  receipt `b427b07fdbb2bf489a10ab043b49109030f1e80c4f1a05df27dc136af58353f6`;
  strict reconciliation then rejected the one open span. The existing
  append-only interruption workflow recorded an explicit recovery point plus
  `interrupted` end (receipt
  `47f3d500984d5fb5671d9d9ad8ccc1e663b17ed71fa866eec8f2d089feefc7ee`),
  and re-ingestion added exactly those two records with 2,581 duplicates
  (receipt `07b913e55e958bc29cb0ad186db30f6cba06ad11746bfc7b67543fd4bee2e502`).
  The sampler now handles SIGINT/SIGTERM by finishing any in-flight sample,
  waking its idle delay, and appending a terminal `sampler.shutdown` point.
  A live Ctrl-C proof produced two matched sample spans and one graceful
  terminal point; its seven records ingested under receipt
  `dbf45764d5ffaca10adc45caa329e51e6f23bc546913766dcdd5f5215a9a2553`.
  Build and all 42 test files / 148 tests pass. Global reconciliation then
  passed across 387 journals / 251,344 records / 20,252 raw artifacts under
  receipt `8c2de36fe09cdc0905cfa54977b7e32aeb49641360fdc4458e698ad0e7f3cb84`.
  Impact: no inference, output, score, or completed performance sample changed;
  the interrupted tail is disclosed as incomplete, and the two proof samples
  use phase `graceful-stop-gate` so analysis can exclude them from Qwen's
  governed inference windows.

- 2026-08-23T20:13:21Z — Pinned the closed Qwen profile boundary after the
  final clean reconciliation. The verified control/workload backup pair is
  1,042,751,488 and 774,144 bytes with receipt
  `222f90043bbfe4bd911bd056f207b1eedabe2a510f567d679f3ec761c2a03322`;
  after pinning/retention, its local and Drive copies matched final inventory
  hash `d1ae49117ac2efd312cc9c3078e42d81d57469a0f4126eb1336fb6a7a90a8ab2`.
  The reconciled state contains 11,332 closed traces, 65,321 closed spans,
  9,986 exactly paired model requests/responses with zero bad hashes,
  4,161,680 unique metric samples, and 16,501 unique raw metric snapshots.
  Bounded retention then removed six redundant local/Drive checkpoint files
  (2,090,282,128 bytes) while retaining the freeze, all four profile pins, the
  two newest rolling recovery points, and two recovery bundles; receipt
  `2125277920e9cb9cf8adfb3dcb5e0488260054c0cedb8bbc443bb92a8e9886e9`.
  Impact: durability/storage only; all frozen scientific, control, raw-model,
  and performance evidence remains retained.

- 2026-08-23T20:22:10.064Z — Derived 2400 predictions for B0-majority-no-action-v1,B2-lexical-v1,B2-vector-v1,B2-hybrid-v1,B3-oracle-packet-v1 over test_id,test_variant_holdout,test_unknown; receipt e53d73c961b2ac02d59ebb9323d08e76faee2a8eadd3babe343f45328b1179cc.

- 2026-08-23T20:25:23.935Z — Scored 2880 primary test_id,test_variant_holdout,test_unknown predictions for baseline-only across B0-majority-no-action-v1,B1-rules-v1,B2-lexical-v1,B2-vector-v1,B2-hybrid-v1,B3-oracle-packet-v1; receipt 559deaf2b54160ad86b7266368098a0f62690d0e4d6aa20378bea4e042a2bae0.

- 2026-08-23T20:26:29.390Z — Materialized 5085 metric rows and 36 paired grouped contrasts for muse-glimmer-30b,gemma-4-31b,qwen-3.8-27b with 10000 bootstraps and 1000 within-stratum group-label permutations; port exclusions=olmo-3.1-32b-instruct; Holm family=36; receipt 7fcfb31517c560a7fdaad9b03b800dec10898415c831377e3407135fff41737d.

- 2026-08-23T20:31:00Z — Closed the Qwen model residency after its pinned
  checkpoint and final telemetry reconciliation, then evicted only the exact
  Qwen chat-model cache revision. The embedding service and SQL Server remain
  resident; the GPU is intentionally idle during SQL/statistics/report work.
  Impact: teardown/storage only; no inference, score, or telemetry evidence was
  deleted.

- 2026-08-23T20:34:00Z — The first complete campaign analysis correctly
  rejected an incomplete baseline grid: B0, the three B2 retrieval modes, and
  B3 had frozen definitions but no retained test-role predictions. Derived the
  five already-frozen deterministic/evaluator arms for all 480 test-role
  episodes each (receipt
  `e53d73c961b2ac02d59ebb9323d08e76faee2a8eadd3babe343f45328b1179cc`)
  and scored all six baseline arms (receipt
  `559deaf2b54160ad86b7266368098a0f62690d0e4d6aa20378bea4e042a2bae0`).
  This is a pre-analysis completeness repair, not post-hoc inference: no LLM
  was called, no frozen arm/episode/search identity changed, and no model row
  changed. Impact: the declared baseline scorecard and contrasts became
  complete; B3 remains an evaluator-only information ceiling.

- 2026-08-23T20:40:00Z — Hardened final analysis so an absent target profile
  may be excluded only after validating at least two distinct, hash-correct
  `STOP_PORT` receipts and their terminal telemetry; synthesized missing model
  rows are rejected. Also moved the metric-definition factory above top-level
  execution after the complete grid exposed a latent initialization-order bug.
  The rerun passed with 8,430 predictions, 5,085 metric rows, 36 frozen
  contrasts, and an explicit OLMo exclusion under receipt
  `7fcfb31517c560a7fdaad9b03b800dec10898415c831377e3407135fff41737d`.
  No contrast cleared both CI-low > 0 and Holm p <= 0.05; disposition is
  `CLEAN_NULL`. Impact: evidence validation/report eligibility only; statistics,
  seeds, rows, and the frozen positive-claim rule were unchanged.

- 2026-08-23T20:45:00Z — Exported a hash-verified row-only reporting bundle:
  85,411 SQL rows across 13 deterministic queries plus 33 signed evidence
  artifacts, query-bundle hash
  `6ac31332676a99962baf0f9a480d87d06611a1e87a6a781cd42165993cdcd796`,
  manifest receipt
  `f62ca2de7ec965d3a2d2b571d86ddbbdc20e6c0f38e3b8f9343d0fff37855a71`.
  Moving database-checkpoint receipts were deliberately excluded from report
  inputs to prevent a backup/report/database hash cycle; final backup and
  restored-database proof are separate signed closeout gates. Generated the
  first deterministic Tier-1 results pack (22 scorecard rows, 206 taxonomy
  rows, 41 claims, 13 figures) and reproduced its claim-bearing outputs
  byte-for-byte from retained rows. Colab lacks `ensurepip`, so `python.sh`
  now verifies and uses its preinstalled pinned-range scientific stack when an
  isolated venv cannot be built. Impact: portability/reporting only; the final
  row and restore closeout gates remain pending.

- 2026-08-23T21:02:09Z — Expanded the retained reporting surface before
  closeout from 13 to 17 deterministic SQL queries (85,596 rows, 33 signed
  evidence artifacts), query-bundle hash
  `5774657af144896634eba445a78991531eeeab12635e5f75b1e7753ee6337fce`
  and manifest receipt
  `02835d283cb931566633e9551ddafd248d116348c65b1909d009fe0db32977be`.
  The generated performance report exposes per-cell p50/p90/p95/p99 agent and
  model time, model/tool/validation outcomes, per-phase closed-span
  distributions, vLLM counters and concurrency, GPU utilization/memory/power,
  queue depth/age, and SQL resource samples. True TTFT, inter-token latency,
  vLLM throughput gauges, KV occupancy, SQL process CPU/log-used percent, and
  Tier-1 Query Store interval rows are explicitly `unavailable`; no proxy is
  relabeled as one of those metrics. The results pack now includes all 13
  required figure source tables, taxonomy labels, B1 coverage, response-repair
  distributions, action-cost decomposition, shuffled-evidence rows, and
  calibration risk/coverage rows. Impact: reporting completeness only; no
  scientific row or inference result changed.

- 2026-08-23T21:02:09Z — `repro.sh --mode rows` passed after `npm ci`: all 42
  test files / 148 tests passed, the exact 50-file report input inventory and
  every hash were verified, and the scorecard, claims, taxonomy, Markdown, and
  figure source CSVs regenerated byte-for-byte. Stable row-gate receipt:
  `cfddf486ec4fea5e94ff1cf47f059f5f23226f22d48bc5133c6a89e61f7e313d`.

- 2026-08-23T21:09:35Z — `repro.sh --mode restore` passed against a new SQL
  Server container, new named volume, and fresh database names. The 1,047,699,456-
  byte control and 774,144-byte workload backups passed retained hash checks;
  restore and physical `CHECKDB` passed; the restored control contained 82
  tables and all 32 migrations; full-text population reached idle. All 17 SQL
  exports and 33 evidence inputs reproduced the exact primary manifest receipt
  `02835d283cb931566633e9551ddafd248d116348c65b1909d009fe0db32977be`,
  then the reports regenerated without drift. Stable restore-gate receipt:
  `843c0c7efa97f02bb2dd99300eaa663b17ed4e90708e476dd07d213af4c5df0e`;
  execution receipt:
  `830cd4f734101746e746930d8b37f714f9335ee09cdf41b8473dbb9cb075bf1d`.
  Two earlier isolated-container attempts stopped before restore: rootless
  `docker cp` could not enter the required read-only host `/proc` mount, and a
  private PID namespace could not mount `/proc` in nested runc. The final path
  uses a hash-verified, short-lived read-only backup bind, retains host PID
  mode, disables restart, and validates the isolated SQL port before teardown.
  All failed-attempt containers, volumes, and staging copies were removed; the
  primary port-1434 SQL service and scientific databases were untouched.

- 2026-08-23T21:14:00Z — Tier closeout deferrals: live replay parity and storm
  ramps/throughput frontier; raw-event model correlation; native tool transport
  and additional guided-decoding cells; second embedding profile; Query Store
  interval/report materialization; temporal incidents; seven-principal split,
  module signing, and SQL Audit; `get_query_store_context`; SQL-native chunking
  and embedding comparators; ANN with plan proof; BACPAC; columnstore/JSON-type
  migrations; throughput, SQL-operations, correlation, and ANN reports; and UI
  remain Tier 2/3. OLMo's one guided-decoding diagnostic is retained as a
  separate Tier-2 port result and is not pooled. Foundry portability checks are
  optional post-closeout and cannot alter this frozen campaign. Impact: the
  Tier-1 claim ceiling remains offline frozen replay on synthetic incidents;
  no live, throughput, production-safety, raw-correlation, ANN, or SQL-native-AI
  claim is made.

- 2026-08-23T21:15:38.238Z — Persisted and hash-locked 206 taxonomy rows, 41 claims, and 15 report snapshots; receipt 38c3d2950939f5271a208d3adf97a3e4977b3ce57b99378c81ef50171c8a1823.

- 2026-08-23T21:18:29.828Z — Stopped the rolling watchdog after final result
  persistence and created the post-ledger database boundary. Both backups
  passed `VERIFYONLY`, in-place restore, and physical `CHECKDB`: control
  1,047,756,800 bytes / SHA-256
  `278c17722df0106d8b7bc390832e950da5b07708347ae4144a4da6a74ebf251a`;
  workload 774,144 bytes / SHA-256
  `53481e310b6cb66a229d3e7d258cd45aa20d3484cdda30b86e0c9067dce70568`.
  Backup receipt:
  `d8316dd88e172f9951f55a3afc0443265d1b6c53efd4b192c14b68f87c27eac1`.
  Pinned it as `tier1-final`, then bounded retention removed six redundant
  local/Drive copies (2,096,950,416 bytes) while preserving the freeze, all
  four model-profile boundaries, Tier-1 final, two rolling recovery points,
  and two recovery bundles. Retention receipt:
  `b5fcae06f2297ddf0cb2ed6398835dfee2cc432ba6db5d4ba6d177901b3e9604`.
  Impact: durability and storage only; no report input, scientific row, pinned
  boundary, or required recovery point was removed.

- 2026-08-23T21:31:51.642Z — Repeated the fresh-SQL reproduction gate against
  the final post-ledger backup. New isolated databases passed `CHECKDB`; the
  control contained 82 tables, all 32 migrations, 206 taxonomy rows, 41
  claims, and 15 report snapshots; the workload contained five tables. The 17
  restored SQL exports plus 33 signed evidence artifacts reproduced the exact
  retained 85,596-row manifest, and all claim-bearing reports rebuilt without
  drift. Stable restore receipt:
  `843c0c7efa97f02bb2dd99300eaa663b17ed4e90708e476dd07d213af4c5df0e`;
  execution receipt:
  `d31e14fe95a52660cd63f35f1e57be28985e8e831f1ac68499ecb84a10ad9f2f`;
  restored-database receipt:
  `aafd5cbeb857bec5e3970cd1a7f199491e045c1834ef1279399ddae6efb349f2`.

- 2026-08-23T21:34:30Z — Hardened rootless-Colab reproduction teardown after
  observing that SQL Server can respawn once between a direct signal and a
  Docker removal request. Cleanup now validates an exact random ephemeral SQL
  port plus the exact SQL executable, starts removal of the uniquely named
  Compose container, and drains only matching processes concurrently. Docker
  calls and exact volume/network fallbacks are bounded by timeouts. A targeted
  test removed the isolated container, both volumes, network, staging copy,
  SQL processes, and shim in under one second; the primary SQL service on port
  1434 remained healthy. Impact: teardown reliability only. All scientific and
  report gates had already passed, and no primary database or evidence row was
  touched.

- 2026-08-23T21:35:00Z — Closed the optional Foundry follow-up using the
  already-merged Mac evidence rather than installing an inapplicable runtime on
  this Linux x86_64 Colab VM. `docs/MAC_PROFILE.md` records a verified M4 Max
  Foundry Local 0.10.3 chat canary (`qwen3-4b`, about 60 tokens/s warm) and
  1024-dimensional embedding canary; the generated Mac report covers multiple
  Foundry ONNX and MLX servings. Those are explicitly separate Mac-plane rows,
  with different artifacts/runtimes, and remain excluded from the digest-pinned
  CUDA/vLLM Tier-1 campaign. Impact: portability evidence only; no Tier-1
  inference, score, or adjudication changed.

- 2026-08-23T21:38:00Z — Generated the definitive run inventory and mirrored
  the complete retained run to Drive. Local and Drive
  `ARTIFACT_INVENTORY.json` files have identical SHA-256
  `6be86a7777294ec3269c73a3001c008b9395837958a5ac3a903d97a7f48b5e26`.
  The completed Lab-3 handoff, append-only experiment log, and root performance
  report were also copied to their stable Drive paths and verified byte-for-byte.
  Impact: durability only; no scientific or report content changed.

- 2026-08-23T21:39:00Z — Stopped the completed Qwen embedding Compose service
  cleanly after all inference, report, persistence, backup, and reproduction
  gates passed. The model/cache volumes and all evidence are retained; no GPU
  compute process remains. Primary SQL Server remains healthy on port 1434 for
  inspection. Impact: runtime resource release only.

- 2026-08-23T22:06:00Z — Added the requested self-contained Tier-1 HTML model-
  comparison report and a companion deterministic JSON payload. The generator
  fails closed against the frozen-output, campaign, telemetry, finalization,
  reproduction, backup, and figure receipts; embeds six retained figures; and
  exposes the full baseline/agent scorecard, arm contrasts, failure funnel,
  retrieval/tool quality, calibration, negative controls, GPU/vLLM residency,
  receipts, source hashes, and evidence limits. Payload receipt:
  `b49a5b3860b54782d93cdb76a0ae24ed3c611fd382a3e929b9d208335e3de61b`;
  HTML SHA-256:
  `80915f567b664780016b56399542e0acc35ce6df9c5d72639b6fe1cbad5a50ed`;
  JSON SHA-256:
  `a5feee5f51d0ea3b7973017d5f8acbce83914c96d109d4eb1d6f6ac3f11332fc`.
  The checked-in snapshot rebuilt byte-for-byte, its browser script parsed and
  populated all 29 targets without invalid values, and all 43 test files / 152
  tests passed. Impact: presentation and discoverability only. No frozen input,
  scientific row, persisted report snapshot, inference result, claim, or
  adjudication changed.

- 2026-08-23T22:07:17Z — Refreshed the complete Drive run mirror after adding
  the HTML and JSON presentation artifacts. Local and Drive
  `ARTIFACT_INVENTORY.json` files are byte-identical at SHA-256
  `951bf7c836ab7212588f03cd011dacf4c0d13e25bb6fa8428a88c0e51e2c5c03`.
  Stable top-level Drive copies of the HTML, payload, handoff, and experiment
  log were also refreshed for discoverability. Impact: durability only; the
  frozen science and final SQL boundary remain unchanged.
