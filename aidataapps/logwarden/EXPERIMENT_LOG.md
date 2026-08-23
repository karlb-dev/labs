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
