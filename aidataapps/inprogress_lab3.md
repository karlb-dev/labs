# Lab 03 in progress — LogWarden

Last manually updated: 2026-08-23 05:05 UTC

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
  agent-observability, and least-privilege tool gates passed except the real
  vLLM metrics canary.
- The dedicated worktree was created from the exact current Lab 2 remote head.
- The new branch was pushed to GitHub and tracks its own remote branch.
- The repository was clean at branch creation.
- The Lab 3 source tree, control/workload databases, initial smoke run, custom
  XE capture session, Query Store configuration, and two-principal security
  model exist and have passed their foundation gates.
- No model weights have been downloaded and no GPU residency is active.
- The complete spec/addendum and predecessor/reference inputs were read and
  hashed before implementation.
- A user-directed pre-inference observability gate now promotes detailed
  agent/vLLM/queue/SQL/XE/GPU telemetry and dual file/SQL persistence before
  any long model campaign; see `logwarden/docs/OBSERVABILITY_CONTRACT.md`.
- Twenty hash-locked control migrations and six versioned
  server/XE/security assets now apply idempotently. `npm run doctor` passes all
  required probes;
  `npm run test:sql` passes 8/8 integration cases; `npm run check` passes 11
  test files and 28 unit tests. The capture-specific XE predicate excludes
  agent/ingest traffic.
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
  Global journal/SQL/raw/trace reconciliation covers 246 events and 66 raw
  artifacts with receipt
  `ecf98a36f50811ec934d64c42670fdbd70f7f6bd735eccc09495ea2ccc2a0c70`;
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
  `c0bf815c9db432cc28555146f597eabd46eb935731826763f01e55b75cbfef57`;
  registry hash:
  `25c79c34cd382bba6bb1f9139401aac8bfea2f603bd83d137b0eb209ddfac6bb`.
  The registry remains marked building/unfrozen until standard runbooks close.
- Current doctor snapshot:
  `a315adb9b7e36c3fe901d3dd9b2f6f02ffcb3026da827e4d7bbd48bd14769c34`
  (`PASS`); SQL integration receipt:
  `107e85c39510a2e161796a348579740f288d8d834472f00862bd3be30fc2d655`
  (8/8).
- Next incomplete milestone: supported incident catalog and runbook
  expansion/freeze, then the read-only tool gateway and bounded agent loop. A
  small embedding or qwen-smoke port may then load
  for the remaining real `/metrics` gate; no long chat campaign may load yet.

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
- GPU/service ownership: Lab 3 rootless Docker daemon active; no vLLM,
  EngineCore, generator, watchdog, or Lab 2 service process is present
- SQL Server: 2025 RTM-CU8 `17.0.4075.5`, healthy on reserved port 1434;
  exact FTS package `17.0.4075.5-1`, `IsFullTextInstalled=1`; derived image
  `sha256:eb4ee252ae0ff6a18b5b40e05eea283e251ee69fbe8926e28def6d0adb43f95f`

The real CUDA allocation/synchronization hard gate passed. Container-level CUDA
and model port gates remain required before model-scale work.

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
- Last watchdog checkpoint: 2026-08-23T04:43:32.002Z
- Last watchdog Git head: `e0cfbe827b58eec9dd49f960f1d140c55be5ffc9` on `aidataapps-logwarden`
- Last watchdog disposition: clean source checkpoint
- Last watchdog database receipt: `e3a5c310ade1785d0af1614800abfb7a49a3911e41d7f8fea537713731ca4e32`
- Last watchdog run: `logwarden-smoke-20260823T031714Z`
<!-- lab3-watchdog-status:end -->

- Long-running scientific process: none
- Infrastructure process: rootless Docker is supervised by retained Codex exec
  cell `64558`; detached children are reaped in this environment
- Watchdog: dirty-tree recovery and clean-tree handoff/commit modes both passed
  and mirrored; recurring watch is not running because no long job exists
- SQL backup: both databases passed COPY_ONLY/CHECKSUM backup, VERIFYONLY,
  full disposable restore, physical CHECKDB, and teardown; latest restore-test
  receipt `cfec5d3bc7a9d52a16c69a9ff0645be5c3d9ea6f21972c7a2fd138bfd34ad2a4`
- Active run ID: `logwarden-smoke-20260823T031714Z`
- Capability snapshot: `a315adb9b7e36c3fe901d3dd9b2f6f02ffcb3026da827e4d7bbd48bd14769c34` (`PASS`)
- SQL integration receipt: `107e85c39510a2e161796a348579740f288d8d834472f00862bd3be30fc2d655` (8/8 passed)
- Last durable Git checkpoint: `7607761` (ERRORLOG rotation watchdog handoff);
  restart recovery and the least-privilege tool layer are being committed now
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
this checkpoint is the supported incident catalog and runbook freeze followed
by the bounded tool gateway/agent loop; do not load a long-running model until
the remaining real-service observability gate can run. If `docker info` fails,
rerun `./scripts/colab-host-init.sh` or launch the rootless daemon in a retained
cell.

## Completion rule currently in force

Tier 1 must close before Tier 2 begins: `repro.sh --mode rows` must pass from a
fresh checkout; `SCORECARD.md` must regenerate byte-for-byte; the Tier 1 safety
audit must be clean; every Tier 1 job must have a terminal disposition; and all
deferred or adapted requirements must be listed with their result impact.
