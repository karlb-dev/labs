# Lab 03 in progress — LogWarden

Last manually updated: 2026-08-23 03:58 UTC

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

- Phase: LW-0/LW-1 complete; LW-2/LW-3 smoke capture, packet, recovery, and pre-inference observability gates passed except the real vLLM metrics canary.
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
- Sixteen hash-locked control migrations and two versioned server/XE assets now
  apply idempotently. `npm run doctor` passes all required probes;
  `npm run test:sql` passes 8/8 integration cases; `npm run check` passes 10
  test files and 25 unit tests. The capture-specific XE predicate excludes
  agent/ingest traffic.
- Development schedule `smoke-v1` injected ten safe scenarios; all ten cleanup
  gates and all required XE/ERRORLOG evidence rules passed. Capture verification
  receipt: `7271b30792a9e054b8f95cedb057df153dda4434517f348b8894cc11fe6bbd5`.
- Raw ERRORLOG replay is idempotent (0 inserted / 461 duplicates on immediate
  replay). XE uses file + block offset + within-block ordinal after runtime
  discovery that offsets are block-scoped.
- Current doctor snapshot:
  `10b7667fdde9bb9f110a001e7c28a8d21b57524eb738e02dccb50c43f1fc5f9e`
  (`PASS`); SQL integration receipt:
  `88859a41d5b05e2a95851b6df58e7975e311349003e579013c4914c55bed7156`
  (8/8).
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
- Next incomplete milestone: ERRORLOG rotation/container recovery and the
  least-privilege read-only tool procedures, followed by supported catalog and
  runbook expansion/freeze. A small embedding or qwen-smoke port may then load
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
- Last watchdog checkpoint: 2026-08-23T04:40:23.193Z
- Last watchdog Git head: `34ff08e5b1a581a5c195b0bb3764946e4433300c` on `aidataapps-logwarden`
- Last watchdog disposition: clean source checkpoint
- Last watchdog database receipt: `898e2b3d4f0c8a88530cff9abdacf4b720e10b30257103ee5e1483111f95ef8b`
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
- Capability snapshot: `10b7667fdde9bb9f110a001e7c28a8d21b57524eb738e02dccb50c43f1fc5f9e` (`PASS`)
- SQL integration receipt: `88859a41d5b05e2a95851b6df58e7975e311349003e579013c4914c55bed7156` (8/8 passed)
- Last durable Git checkpoint: `762806d` (packet/base telemetry watchdog handoff); terminal-route/recovery instrumentation is being committed now
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
this checkpoint is ERRORLOG rotation/container recovery followed by read-only
tools and the standard catalog/runbook freeze; do not load a model until the
remaining real-service observability gate can run. If `docker info` fails,
rerun `./scripts/colab-host-init.sh` or launch the rootless daemon in a retained
cell.

## Completion rule currently in force

Tier 1 must close before Tier 2 begins: `repro.sh --mode rows` must pass from a
fresh checkout; `SCORECARD.md` must regenerate byte-for-byte; the Tier 1 safety
audit must be clean; every Tier 1 job must have a terminal disposition; and all
deferred or adapted requirements must be listed with their result impact.
