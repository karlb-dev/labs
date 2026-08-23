# Lab 03 in progress — LogWarden

Last manually updated: 2026-08-23 07:27 UTC

Read `resume.md` first for worktree, recovery, and evidence rules. This file is
the volatile state of Lab 3 and must be refreshed before and after long jobs and
at every scientific evidence boundary.

## Ownership and objective

- owner: the Lab 3 Codex session on the clean Colab VM started 2026-08-23 UTC
- worktree: `/content/worktrees/aidataapps-logwarden`
- branch: `aidataapps-logwarden`
- remote branch: `origin/aidataapps-logwarden`
- base branch: `origin/aidataapps-modelprint`
- base commit: `88ea443092cd27226272776e6c6fcf8fbce329de`
- lab directory: `/content/worktrees/aidataapps-logwarden/aidataapps/logwarden`
- Drive root: `/content/drive/MyDrive/aidataapps/lab03`
- governed source plan:
  `/content/drive/MyDrive/aidataapps/lab03/aidataapps_logwarden_lab_3_spec.md`
- governing addendum:
  `/content/drive/MyDrive/aidataapps/lab03/aidataapps_logwarden_lab_3_spec_addendum.md`

The objective is the addendum's Tier 1 Minimum State of Record: construct and
freeze a ground-truthed SQL Server incident corpus, run the governed four-model
replay benchmark and baselines, score it with paired statistics and controls,
and close with a reproducible row-only and database-restore state of record.
The addendum wins over the base specification. Narrow implementation/runtime
adjustments are allowed when required to reach completion, but every adjustment
and its effect on the evidence ceiling must be recorded append-only in
`EXPERIMENT_LOG.md`.

## Current state

- Phase: LW-0/LW-1 complete; LW-2/LW-3 smoke capture, packet, recovery,
  agent-observability, least-privilege tools, real eight-family injectors,
  exact-correlation/context-snapshot gates, the real Qwen embedding vLLM port
  gate, all 480 corpus embeddings, the three-mode hybrid retrieval development
  gate, and the fully persisted bounded agent loop passed. Standard capture,
  held-out retrieval assessment, corpus freeze, and chat-model canary remain.
- The dedicated worktree was created from the exact current Lab 2 remote head.
- The new branch was pushed to GitHub and tracks its own remote branch.
- The repository was clean at branch creation.
- The Lab 3 source tree, control/workload databases, initial smoke run, custom
  XE capture session, Query Store configuration, and two-principal security
  model exist and have passed their foundation gates.
- The pinned Qwen 0.6B embedding weights are downloaded and vLLM is resident
  on GPU at port 8011. It generated and verified the complete primary corpus;
  no chat-model weights are loaded.
- The complete spec/addendum and predecessor/reference inputs were read and
  hashed before implementation.
- A user-directed pre-inference observability gate now promotes detailed
  agent/vLLM/queue/SQL/XE/GPU telemetry and dual file/SQL persistence before
  any long model campaign; see `logwarden/docs/OBSERVABILITY_CONTRACT.md`.
- Thirty hash-locked control migrations and seven versioned
  server/XE/security assets now apply idempotently. `npm run doctor` passes all
  required probes; `npm run test:sql` passes 8/8 integration cases;
  `npm run check` passes 19 test files and 64 unit tests. The capture-specific
  XE predicate excludes agent/ingest traffic.
- Development schedule `smoke-v1` injected ten safe scenarios; all ten cleanup
  gates and all required XE/ERRORLOG evidence rules passed. Capture verification
  receipt: `7271b30792a9e054b8f95cedb057df153dda4434517f348b8894cc11fe6bbd5`.
- Raw ERRORLOG replay is idempotent (0 inserted / 461 duplicates on immediate
  replay). XE uses file + block offset + within-block ordinal after runtime
  discovery that offsets are block-scoped.
- Ten development packets have separately protected evaluator truth and passed
  the leakage audit with zero findings. Packet build/audit receipts are
  `06670cc91548962090843e921f0b72d0f58e38544f79b2fd0ffb9768fb749e11`
  and `f6cc4310ac038a320106c94ad3d43912e9737fd840043c85b63ccc7a698fbd74`.
- A full no-model systems sample persisted GPU, SQL, queue, XE, host, and raw
  endpoint-unavailable observations. Journal-to-SQL replay is hash validated
  and idempotent; latest receipts are
  `f19f2884678a01d417f41dbe559f5be4416adc2a46012b0cde6cb3c4f4e89867`
  (sample) and
  `430d53f721b081915e88d5d79ece7362673e7e611e9332b6e7153e9b342d3ab4`
  (journal ingest).
- The pre-inference agent observability gates now pass: nine fake-gateway
  terminal routes with exact raw-byte/SQL hashes
  (`fdce92eeeb8344b52c02455a6d244ee9d9839bc57c2f556f01be5caae0f5d474`),
  committed-batch crash/replay
  (`ac60e25a385f2ecfedc1bec6d4fd33dc92de9cbb78f6d0c5935bcda529d20660`),
  two-process sampler restart
  (`c30e80918b27592426a3657cb0e6c64fe97aa4a09057eda935deff97ba290218`),
  and p95 added synthetic instrumentation latency of 1.497 ms
  (`3888e28dbd673f71e75ef09bf34451e9b15c4d3196e8c0db2a4c66a2baa6efea`).
  Global journal/SQL/raw/trace reconciliation covers 252 events and 66 raw
  artifacts with receipt
  `8ffc7bd73a43d50d64113cabfe8fc9f0e70db105b0271acd308230cc6c89c505`;
  the derived file rebuilt byte-for-byte.
- ERRORLOG `sp_cycle_errorlog` recovery passed with a stable pre-roll key,
  one unique post-roll marker, and 599/599 duplicates on immediate replay;
  receipt `594b46485c6865068c2093c311190b09eb900406c51b76d2f972704a73087d1e`.
- SQL/container restart recovery passed with distinct container/SQL start
  identities, stable pre-restart source identity, a unique post-restart row,
  and 997/997 duplicates on immediate replay; receipt
  `f6688d4472af0db6947076097a27be7d14cd58490b736d9dc442fb2c2adc2c99`.
- The seven-tool `tools-v1` registry and its SQL enforcement layer passed nine
  positive and eleven negative least-privilege cases. Direct evaluator, snapshot,
  runbook-table, queue, ingestion, DDL, server-DMV, and `msdb` backup-table
  bypasses are denied. Receipt:
  `f5965e712ba2edfe542ca767f042ecc29d81d985af3b85f8bdb6cb2e80de48cf`;
  registry hash:
  `25c79c34cd382bba6bb1f9139401aac8bfea2f603bd83d137b0eb209ddfac6bb`.
  The registry remains marked building/unfrozen until standard runbooks close.
- The deterministic `primary-v1` knowledge corpus now contains 60 original
  MIT-licensed guides and 480 heading-aware chunks across all ten classes, with
  exact source/body/chunk hashes and zero lab-identifier leakage findings.
  Full-text indexes both headings and content; all 11 restricted-agent lexical
  canaries retrieved the intended family. Manifest/build/gate hashes are
  `b734ba4c05ff3e548b779b1f9718e8956e4f0f9bb45a8246492664f88a0e66df`,
  `d8b6b81aff2811460fdee6a5469ceb69540381720ceaf9f961af7c149e2f2d79`,
  and `7e8c38b4218dae4b43c27e95cd522ae669f026fbe94249c3890e7d8cc51808be`.
  The corpus remains unfrozen pending the standard scenario leakage audit and
  held-out retrieval assessment; its Qwen embeddings and hybrid mechanics are
  complete.
- The live Qwen embedding port gate passed with receipt
  `d9f7b756c907a4aa9516a1c47875dec2a5e7d932b4bc4a721c93bb3e93492323`.
  It proved exact image/model/revision/runner identity, health/model listing,
  four HTTP calls and six ordered finite normalized 1024-dimensional outputs,
  exact warmed-repeat identity, raw request/response durability, 32
  journal-to-SQL records, and before/midpoint/after vLLM plus GPU snapshots.
  Counter deltas were 4 HTTP, 6 success, 108 prompt tokens, 6 latency, 0 error,
  and 0 preemption; midpoint GPU evidence showed the EngineCore at 6,047 MiB
  and 3% utilization. The replacement receipt is independently rehashable and
  supersedes an initially non-rehashable in-memory-Buffer serialization. Two
  other fail-closed development attempts are retained: an
  uninstantiated pre-request Prometheus route series is now correctly treated
  as zero while its metric family remains required, and measured first-CUDA
  cold/warm drift is explicitly bounded while warmed repeats remain exact.
- All 480 primary chunks were embedded in 15 durable batches and persisted
  with complete input, request, response, operation, batch, model, run, and raw
  artifact provenance. Final receipt:
  `bca6706cf683f265f61bff8e7f7abe0531bf8722afb36f479d29c66f7feca1aa`.
  Generation deltas are exactly 15 HTTP requests, 480 successes, 23,736 prompt
  tokens, 480 latency observations, zero errors, and zero preemptions; request
  p50/p95 is 36.843/41.932 ms. Ordered input, service-vector, and SQL-storage
  hashes are `eb3559f43fa413bc70d514924c57ea044b25ab2a0c1e97ee91593144216099fc`,
  `012daefc3106442687f0842bf442e3ec1f37c1558e6f07dd354165034af54aa5`,
  and `f8a929f0bc3f87ddf8a2edc4fe6406e52900da7f2759d2169800188817cf3d31`.
  Across 491,520 components, SQL float32 conversion has worst absolute drift
  `4.995651239902976e-9` and minimum cosine `0.9999999999999969`.
- App-owned lexical full-text, exact-vector, and hybrid RRF retrieval now
  persist complete query-vector/component-rank/result evidence while retaining
  least privilege. The 11-query x three-mode development gate passed with
  receipt `f6a3b4aefbca5214beee3272566f5f01d696f4be3dcb922a9660db2ea86d1dfd`:
  all modes reached recall@5=1 and MRR=1, with lexical/vector/hybrid SQL p50 of
  14.074/72.502/75.022 ms. These deliberately easy canaries prove mechanics,
  not hybrid lift; that claim is reserved for held-out packet evaluation.
- The bounded multi-turn agent gate passes eight terminal routes with receipt
  `8449a07300d761fdac156962f8c9b7056f48f8bd5c96d825745b82ce3ddc10b8`:
  17 model turns, 13 tool calls, three safely persisted decisions, exact cache
  reuse, a deterministic snapshot miss, four rejection classes, zero executed
  actions, zero retained leases, and closed/link-complete traces. Prompt/model/
  retry/tool/validation/decision phases persist in both hash-chained files and
  SQL; raw requests, responses, and full tool results are independently
  rehashed. Two uncached hybrid searches produced exactly two real CUDA
  embedding requests/successes, 17 prompt tokens, two latency observations,
  zero errors, and zero preemptions. The primary transport is user-turn-only
  and unconstrained: no system message, OpenAI `tools`, or `response_format`.
  Queue migrations 029/030 add RCSI-safe locking reads and normalize pooled
  session isolation; three fail-closed attempts and their evidence impact are
  append-only in `EXPERIMENT_LOG.md`.
- The unfrozen `logwarden-standard-v1` scenario catalog now contains 60
  group-isolated templates and 600 deterministic variants across all ten
  incident families and all five regimes. Exact role allocation is 60 dev, 60
  calibration, 300 test-ID, 120 test-variant-holdout, and 60 test-unknown;
  every family has at least 40 held-out ID/variant episodes and no group crosses
  a split. Catalog, manifest, schedule, and structural-gate hashes are
  `5bc260b8c11c6ace97cb5f047dd0ad7f68958860cc24c3e0e43313ee9f05934f`,
  `50d47873fda8ccd3367ddbd5f9f1356f1fd306f6406c33de122471a1c5e7872d`,
  `3b7097f5b855d91741bc2d470e3074040464b96a9ae66e1df766639c6daea20b`,
  and `5535d9f8c29d58d6b1ec222b3ab0842c20de0c669d01367d3118ff07e1f17f82`.
  Schedule `standard-v1` has 600 pending items and a final planned offset of
  8,039,000 ms; it remains building and has injected zero episodes.
- All eight real incident injector families passed the three-signal
  development feasibility gate: 8/8 capture verification receipt
  `0ff81209e857278a56c9920c233281cb34d4b97815d60936efff9291b407e79a`
  and 32-check deep receipt
  `8ace73c31d9b07e7dba3388900eca9b0da94d71740176313f76b260f901cabe3`.
  The gate proves exact raw-token attribution, three observed driver signals,
  current deadlock semantic hashes, sustained blocking/active transactions,
  >=90% log utilization with `ACTIVE_TRANSACTION`, snapshot hashes, cleanup,
  and zero XE loss counters. A v1 development run was rejected when the deep
  gate exposed proximity-only attribution of connection setup batches; the
  corrected verifier requires intrinsic full-token RPC/batch evidence and
  deterministically rebuilds links. No standard data existed, so result impact
  is limited to stronger pre-capture independence.
- SQL Server rejected the requested `NO_EVENT_LOSS` mode for this
  `error_reported` event mix (error 25643). The documented capability-forced
  contract uses `ALLOW_SINGLE_EVENT_LOSS`, 1-second dispatch, 16 MiB x 20
  files, and fail-closed zero loss counters. Asset hash:
  `667bee8af19d8b2dba416cd31968215945b8f39ee1121e6294e37549bc2068c1`.
- Current doctor snapshot:
  `2a6f74acb8e0c1a35c06faa437e3565b13df1be26d934823a84ca50bd6466548`
  (`PASS`); SQL integration receipt:
  `407d3147ac4fe8898475928f7debb83e878bb0f0500a267ac579192db31ad3b2`
  (8/8); least-privilege tool receipt:
  `fca0eaa2e865c00082141f98527c85b9e6f83f77caeae10a2254c0f4a879126f`
  (9 positive/11 negative).
- The five user-supplied Mac/Foundry/report commits through `725cdae` are
  merged. Their platform-separated Foundry/MLX registry, Apple Silicon
  Docker/Rosetta setup, 16-episode dev harness, six-model retained results, and
  report generator coexist with the Colab path. Colab Compose validation,
  doctor, SQL, security, backup/restore, and all pre-merge tests pass; the
  committed 6 x 16 HTML report rebuilds byte-for-byte from retained data and
  its template. Foundry is absent on this Linux VM and, by user direction,
  remains a post-governed-model validation lane with non-comparable results.
- Next incomplete milestone: run standard capture, build/audit its packets,
  assess held-out retrieval, and freeze the corpus. The qwen-smoke chat service
  may load only for its
  separate real `/metrics` gate; no long chat campaign may start until that
  gate passes. Foundry model validation is deliberately deferred until after
  the governed local models.

## Fresh-VM preflight

Observed at `2026-08-23T02:40:15Z` from the launch shell:

- host/kernel: Linux `6.6.122+`, x86-64
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- GPU UUID: `GPU-e07963ac-e6be-8913-3294-b83917338161`
- driver: `580.82.07`
- VRAM: 97,887 MiB total; 97,251 MiB free
- local filesystem: 236 GiB total; 188 GiB free
- mounted Drive filesystem: present; 178 GiB free
- Node.js: `v20.19.0`
- npm: `10.8.2`
- Python: `3.13.15`
- GitHub SSH read/write authentication: passed (`git ls-remote` and branch
  push)
- Docker: Engine 29.7.2, Compose 5.5.0, rootless `fuse-overlayfs`
- NVIDIA Container Toolkit: 1.20.0; generated CDI spec
- GPU/service ownership: Lab 3 rootless Docker daemon active; Qwen embedding
  vLLM/EngineCore is resident on port 8011; no chat vLLM, generator, watchdog,
  or Lab 2 service process is present
- SQL Server: 2025 RTM-CU8 `17.0.4075.5`, healthy on reserved port 1434;
  exact FTS package `17.0.4075.5-1`, `IsFullTextInstalled=1`; derived image
  `sha256:eb4ee252ae0ff6a18b5b40e05eea283e251ee69fbe8926e28def6d0adb43f95f`

The real CUDA allocation/synchronization hard gate and Qwen embedding model
port gate passed. A separate chat-model port gate remains required before
chat-model-scale work.

## Planned isolated resources

These are reserved for Lab 3 and may be changed only with a recorded update:

- Compose project: `aidataapps-logwarden`
- SQL host port: `1434`
- chat-model host port: `8010`
- Qwen embedding host port: `8011`
- optional BGE embedding host port: `8012`
- databases: `LogWardenControl` and disposable `LogWardenWorkload`
- local runs: `aidataapps/logwarden/runs/<run-id>`
- Drive runs: `/content/drive/MyDrive/aidataapps/lab03/runs/<run-id>`

The inherited Lab 1/2 directories and their branches are read-only inputs.

## Active processes and checkpoints

<!-- lab3-watchdog-status:start -->
- Last watchdog checkpoint: 2026-08-23T07:02:39.402Z
- Last watchdog Git head: `bc1eed0b1a6043f8eac87aa2f5c5cfcd0721cbfa` on `aidataapps-logwarden`
- Last watchdog disposition: clean source checkpoint
- Last watchdog database receipt: `6b8df2b7cc053d527e760618bb677906f85f9b3e1488948f3236ad2797b4b43d`
- Last watchdog run: `logwarden-smoke-20260823T031714Z`
<!-- lab3-watchdog-status:end -->

- Long-running scientific process: none; the Qwen embedding service is an
  idle, detached infrastructure service
- Infrastructure process: rootless Docker is supervised by retained Codex exec
  cell `64558`; detached children are reaped in this environment
- Watchdog: dirty-tree recovery and clean-tree handoff/commit modes both passed
  and mirrored; recurring watch is not running because no long job exists
- SQL backup: both databases passed COPY_ONLY/CHECKSUM backup, VERIFYONLY,
  full disposable restore, physical CHECKDB, and teardown; latest restore-test
  receipt `20a92098906ac63588ce951ce15e21d3ec932ca10f4d81beb504cffbfc4a9a18`
- Active run ID: `logwarden-smoke-20260823T031714Z`
- Capability snapshot: `2a6f74acb8e0c1a35c06faa437e3565b13df1be26d934823a84ca50bd6466548` (`PASS`)
- SQL integration receipt: `407d3147ac4fe8898475928f7debb83e878bb0f0500a267ac579192db31ad3b2` (8/8 passed)
- Last durable implementation checkpoint: `bf2ae68` (governed bounded agent
  loop, deep phase evidence, and RCSI-safe queue claims)
- Last durable Drive checkpoint: this file

Before the first job expected to exceed 20 minutes, launch the tested
Lab-3-scoped checkpoint watchdog. It pushes committed work, makes a verified
branch bundle and worktree recovery capture, backs up both Lab 3 databases,
archives/mirrors only the active Lab 3 run, and updates this file without
touching Lab 2 state.

## Exact resume procedure

```bash
cd /content/worktrees/aidataapps-logwarden
git fetch origin
git pull --ff-only
git status --short --branch
cat aidataapps/resume.md
cat aidataapps/inprogress_lab3.md
ps -eo pid,ppid,etimes,args | rg 'logwarden|replay-run|vllm serve|EngineCore|checkpoint-watchdog' || true
nvidia-smi
```

Then inspect the newest `EXPERIMENT_LOG.md`, active run pointer, watchdog log,
SQL job state, and Drive checkpoint before launching anything. The next work at
this checkpoint is Qwen embeddings and the bounded tool gateway/agent loop;
do not load a
long-running model until the remaining real-service observability gate can run.
If `docker info` fails,
rerun `./scripts/colab-host-init.sh` or launch the rootless daemon in a retained
cell.

## Completion rule currently in force

Tier 1 must close before Tier 2 begins: `repro.sh --mode rows` must pass from a
fresh checkout; `SCORECARD.md` must regenerate byte-for-byte; the Tier 1 safety
audit must be clean; every Tier 1 job must have a terminal disposition; and all
deferred or adapted requirements must be listed with their result impact.
