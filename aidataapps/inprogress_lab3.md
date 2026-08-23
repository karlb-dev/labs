# Lab 03 in progress — LogWarden

Last manually updated: 2026-08-23 02:40 UTC

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

- Phase: LW-0 foundation and predecessor intake.
- The dedicated worktree was created from the exact current Lab 2 remote head.
- The new branch was pushed to GitHub and tracks its own remote branch.
- The repository was clean at branch creation.
- No Lab 3 source directory, database, container, run, or long process exists
  yet.
- No model weights have been downloaded and no GPU residency is active.
- The Lab 3 spec/addendum are being read completely before implementation.
- Next incomplete milestone: vendor the governing documents, record predecessor
  hashes and frozen model-registry provenance, then scaffold the smallest
  independently testable Tier 1 foundation.

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
- Docker: not installed on this fresh VM
- GPU/service ownership: no vLLM, EngineCore, generator, watchdog, Docker
  daemon, or Lab 2 service process was present

A real CUDA allocation/synchronization hard gate and Docker/runtime version
capture remain required before any model-scale work.

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

- Long-running scientific process: none
- Watchdog: not yet implemented or running
- SQL backup: none
- Active run ID: none
- Last durable Git checkpoint: branch creation at base commit `88ea443`
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
SQL job state, and Drive checkpoint before launching anything. The next command
at this checkpoint is the bounded LW-0 intake/scaffold work, not a model run.

## Completion rule currently in force

Tier 1 must close before Tier 2 begins: `repro.sh --mode rows` must pass from a
fresh checkout; `SCORECARD.md` must regenerate byte-for-byte; the Tier 1 safety
audit must be clean; every Tier 1 job must have a terminal disposition; and all
deferred or adapted requirements must be listed with their result impact.
