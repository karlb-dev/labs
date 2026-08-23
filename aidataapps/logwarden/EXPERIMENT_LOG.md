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
