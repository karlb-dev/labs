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

- Phase: LW-0/LW-1 complete; LW-2/LW-3 development smoke capture passed; beginning packet/recovery gates.
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
  `npm run test:sql` passes 8/8 integration cases; `npm run check` passes 7
  test files and 14 unit tests. The capture-specific XE predicate excludes
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
- Next incomplete milestone: build protected incident packets and leakage
  audit, test ERRORLOG rotation/container recovery, implement verified SQL
  backup/restore and watchdog durability, then expand the supported catalog
  before the standard packet freeze. No embedding or chat model may load yet.

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
- Last watchdog checkpoint: 2026-08-23T04:04:43.673Z
- Last watchdog Git head: `3b1c1d0cedb82183bdc40a600aaf25e74ecdfdc7` on `aidataapps-logwarden`
- Last watchdog disposition: captured dirty recovery patch; no automatic source commit
- Last watchdog database receipt: `3bf507fe139b3e6707bf4fa3d38afe24e82278006f1ac07325c50b4ac46f035c`
- Last watchdog run: `logwarden-smoke-20260823T031714Z`
<!-- lab3-watchdog-status:end -->

- Long-running scientific process: none
- Infrastructure process: rootless Docker is supervised by retained Codex exec
  cell `64558`; detached children are reaped in this environment
- Watchdog: one dirty-tree checkpoint completed and mirrored; clean-tree
  auto-handoff/commit mode is the next validation
- SQL backup: both databases passed COPY_ONLY/CHECKSUM backup, VERIFYONLY,
  full disposable restore, physical CHECKDB, and teardown; latest restore-test
  receipt `cfec5d3bc7a9d52a16c69a9ff0645be5c3d9ea6f21972c7a2fd138bfd34ad2a4`
- Active run ID: `logwarden-smoke-20260823T031714Z`
- Capability snapshot: `72dd444ac65f7b1203c711c1383878e100f78cffd33fa2e4bc8b775a9f6d88e3` (`PASS`)
- SQL integration receipt: `8ecaf7cd8f3693b96554b9fb3a0e2108fe698de301c0f5c7f649282ddd9a4ebd` (8/8 passed)
- Last durable Git checkpoint: `cf6cf5e` (`Build LogWarden SQL control plane`); the verified capture checkpoint is being committed now
- Last durable Drive checkpoint: this file

Before the first job expected to exceed 20 minutes, implement and launch one
Lab-3-scoped checkpoint watchdog. It must push committed work, make a verified
branch bundle and worktree recovery capture, back up the Lab 3 database when it
exists, archive/mirror only the active Lab 3 run, and update this file without
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
this checkpoint is LW-2/LW-3 scenario capture and ingestion; do not load a model
until the observability gate and campaign freeze both pass. If `docker info` fails, rerun
`./scripts/colab-host-init.sh` or launch the rootless daemon in a retained cell.

## Completion rule currently in force

Tier 1 must close before Tier 2 begins: `repro.sh --mode rows` must pass from a
fresh checkout; `SCORECARD.md` must regenerate byte-for-byte; the Tier 1 safety
audit must be clean; every Tier 1 job must have a terminal disposition; and all
deferred or adapted requirements must be listed with their result impact.
