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
