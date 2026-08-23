# AI Data Apps labs — shared resume and recovery guide

Last updated: 2026-08-23 UTC

This is the shared entry point for every AI Data Apps lab agent. Read this file
first, then read only the `inprogress_labN.md` file for the lab you own. Update
your lab-specific file as state changes; do not put volatile Lab 2 or Lab 3
status in this shared file.

## Repository and durable storage

- Git repository: `/content/labs`
- Git remote: `git@github.com:karlb-dev/labs.git`
- AI Data Apps root: `/content/labs/aidataapps`
- mounted Drive root: `/content/drive/MyDrive/aidataapps`
- durable shared copy: `/content/drive/MyDrive/aidataapps/resume.md`
- durable Lab 2 status copy: `/content/drive/MyDrive/aidataapps/inprogress_lab2.md`
- machine-local long handoff for Lab 2: `/content/handoff.md`
- Lab 2 status: `aidataapps/inprogress_lab2.md`
- Lab 3 status: create and maintain `aidataapps/inprogress_lab3.md` in the Lab
  3 branch/worktree; never overwrite the Lab 2 file

GitHub is the source backup for committed code and documentation. Drive is the
source backup for run outputs, database exports, recovery bundles, logs, and
uncommitted worktree recovery material. A Colab VM can disappear without
warning and is reclaimed at 24 hours, so neither local-only source nor
local-only results are acceptable.

Keep static bootstrap instructions here and volatile scientific state in the
lab-specific `inprogress_labN.md`. The live file must say what is running, its
owner/PID or managed terminal session, its last durable checkpoint, the exact
idempotent resume command, and the next incomplete milestone. Update it before
and after every long launch and at every evidence boundary.

## Fresh-VM preflight

Before installing packages, downloading weights, or resuming science:

1. Record UTC time, VM/GPU identity, driver/runtime versions, local and Drive
   free space, and whether `/content/drive/MyDrive` is actually mounted.
2. Verify the Codex permission profile, Git network authentication, and GPU
   visibility from the same execution context that will launch the long job.
   A notebook kernel, agent shell, container, and managed terminal can have
   different environment variables and device access.
3. Run a real CUDA allocation/synchronization or the lab's GPU hard gate. Never
   silently fall back to CPU for model-scale work; CPU is appropriate only for
   explicitly bounded tests, hashing, analysis, plotting, and reports.
4. Clone and run code on local NVMe under `/content`, fetch the intended branch,
   and record branch, HEAD, upstream, base commit, worktree, and `git status`.
5. Recreate local secrets from the approved secret store without printing or
   copying them into the repo/run/handoff. Notebook environment variables do
   not necessarily reach agent shells; verify Git and Hugging Face access in
   the actual launch shell.
6. Recreate dependencies from tracked manifests and run the fastest no-model
   conformance tests. On fresh Colab images, run `apt-get update` before an
   `apt-get install` to avoid stale package-index 404s.
7. Read the governing plan, then its addendum, then the dynamic in-progress
   file. Binding addenda and frozen manifests win over older prose.

Do not load model weights through DriveFS. Download or copy pinned snapshots to
local NVMe, then verify every indexed shard and important config/tokenizer hash
before serving. A partial Hub snapshot can sometimes be completed from a Drive
cache with `rsync`, but Drive remains the source of a copy, not the live model
filesystem. Never delete or alter DriveFS's internal cache or `chunks.db`.

Before downloading the next residency, require enough local free space for the
new pinned snapshot plus working data and checkpoint headroom; 1.5 times the
snapshot size is a useful minimum gate. Delete only an exact, rehydratable
local model cache, and only after results, SQL backup, manifests, hashes, and
Drive mirror for that residency have been verified.

## Mandatory multi-agent isolation

One agent owns one Git worktree and one branch. Never run `git switch`,
`git checkout`, `git clean`, a reset, or a broad stash in a worktree owned by
another agent. Worktrees share Git objects and remotes while keeping indexes and
checked-out files independent.

Current Lab 2 ownership:

- worktree: `/content/labs`
- branch: `aidataapps-modelprint`
- lab directory: `/content/labs/aidataapps/modelprint`

The Lab 3 agent must first inspect existing ownership:

```bash
git -C /content/labs worktree list --porcelain
git -C /content/labs branch --all
```

If Lab 3 does not already have a worktree or branch, create its own from the
intended base (normally `origin/main`):

```bash
mkdir -p /content/worktrees
git -C /content/labs fetch origin
git -C /content/labs worktree add -b aidataapps-lab03 \
  /content/worktrees/aidataapps-lab03 origin/main
```

If `origin/aidataapps-lab03` already exists, track that remote branch instead of
creating a second branch with the same name. Record the final worktree, branch,
base commit, Drive directory, ports, containers, database, and run directory in
`inprogress_lab3.md` before starting long work.

Do not have two agents edit the shared coordination files simultaneously. Make
shared-file changes in a small commit, push it, then have the other lab
cherry-pick or merge that commit into its branch.

## Shared VM resource coordination

The VM has one NVIDIA RTX PRO 6000 Blackwell GPU. Lab 2's GPU phase is complete;
its chat and embedding services were stopped at the CPU-handoff boundary, so
ports 8000–8002 and its VRAM allocation are released. Another lab must still
check `inprogress_lab2.md`, current allocations, and full process commands
before loading a model because this shared status can outlive the VM state.

Every lab must use unique values for all mutable infrastructure:

- Git branch and worktree
- Docker Compose project and container names
- host ports
- SQL database name and any volume that is not intentionally shared read-only
- `.current-run`, run ID, PID/lock files, and log paths
- Drive subtree (`lab02`, `lab03`, and so on)

Never stop a container or process from another lab based only on its port. Match
the full container name or complete process command first. Never expose secrets
from `.env` in Git, logs, patches, handoffs, or run archives.

## Twenty-minute recovery contract

During any job expected to run longer than 20 minutes, its lab must run one
lab-scoped watchdog with a default interval of 1,200 seconds. Each successful
cycle should:

1. push the already-committed branch to GitHub;
2. create a verified Git bundle of the branch;
3. capture `git status`, a binary worktree patch, and non-ignored untracked
   source files without auto-committing them;
4. copy the shared/lab handoffs into the recovery checkpoint;
5. make an online, checksum-recorded database backup when the lab has a DB;
6. archive and mirror the active run to the lab's Drive subtree; and
7. append a UTC log that makes partial failures visible without killing the
   scientific job.

Commit and push manually at meaningful milestones. A watchdog must never invent
milestone commits, commit transient run output, force-push, or merge branches.

The 20-minute cadence is the external durability ceiling, not necessarily the
scientific job's own checkpoint interval. Where practical, make per-item output
append-only and checkpoint an atomic resume cursor at least every 10 minutes.
Persist the model revision, image/runtime, prompt/config/bank hashes, campaign
identity, and next offset with the cursor; a resume must refuse a mismatch.
Rerunning the exact command should no-op completed items rather than overwrite
them. Finalization sentinels should make completed stages exactly-once unless a
new governed run is intentionally created.

Before resuming any apparently dead job, reconcile all of these signals:

- process command and parent/child PIDs;
- actual advisory-lock owner (a zero-byte lock file alone proves nothing);
- GPU allocations and listening ports;
- last complete log record;
- atomic cursor/checkpoint header and hash; and
- SQL/job-table state or immutable output inventory.

Never launch a duplicate until those sources agree the run is unowned. A
detached `nohup` process may be reaped by an agent execution runner even when it
would survive an ordinary notebook shell; prefer a managed persistent terminal
session and record its session/PID. After an agent UI restarts, adopt a healthy
existing process instead of restarting it.

Reserve at least the final 90 minutes of the 24-hour allocation for a stable
checkpoint, state-of-record update, tests, result inventory/hashes, database
export, Git commit/push, Drive mirror, and a read-back verification. Start that
closeout earlier if disk, Drive, or network behavior is degraded.

DriveFS presence is not proof of cloud durability. A write can appear locally
but remain only in its client cache after `ENOSPC`, quota, rate-limit, or mount
errors. Treat any such error as a failed backup. Check command exit status,
size, and SHA-256 from the destination; for critical boundaries, re-open after
sync (or remount on a fresh VM) and re-hash. Avoid repeatedly rewriting
multi-gigabyte checkpoints through DriveFS; use bounded atomic files and a
documented retention policy.

Lab 2 implements this contract in
`aidataapps/modelprint/scripts/checkpoint-watchdog.sh`. Other labs should copy
the behavior but use their own lock, run, database, branch, and Drive paths.
The Lab 2 watchdog atomically refreshes the two coordination documents at the
Drive root and `/content/drive/MyDrive/aidataapps/lab02/handoff.md` every cycle,
independently of the larger run mirror.

## Generic recovery order after VM replacement

1. Remount Drive and clone/fetch `/content/labs` from GitHub.
2. Read this file and the correct `inprogress_labN.md` from that lab's branch.
3. Recreate or attach the dedicated worktree; do not reuse another lab's
   worktree.
4. Compare GitHub HEAD with the Drive checkpoint's `metadata.json`.
5. If needed, recover committed history from `repository.bundle`, apply
   `working-tree.patch`, then inspect `untracked-source.tar.gz`; never apply a
   patch blindly over newer work.
6. Restore the newest checksum-valid native SQL backup or final BACPAC.
7. Recreate `.env` locally from `.env.example`; restore secrets manually.
8. Inspect active processes and database job state before using an idempotent
   `--resume` command. Never launch a duplicate generator.

Recovery priority is: latest consistent Drive run/DB artifacts, GitHub branch,
Drive Git bundle and patch, then older checkpoints. Check hashes and timestamps
instead of assuming the file with the newest name is complete.

## Evidence and closeout discipline

- A claim-bearing result should carry the producing Git commit, exact command,
  model revision, config/input hashes, environment identity, and output hash.
- Keep raw outputs immutable. Correct or supersede them with a new artifact and
  an explicit reason; do not silently edit historical evidence.
- Make milestone commits result-bearing and leave the tree clean at governed
  evidence boundaries. Heavy/ignored outputs belong in a hash inventory on
  Drive, not an oversized Git commit.
- Between model residencies, ensure the prior API/engine process exited, its
  port is free, and GPU memory returned to the expected embedding/idle baseline.
  Do not rotate two large model families in one long-lived process.
- A run is not complete until restore/reproduction has been tested from its
  exported artifacts and the final regenerated reports match the registered
  metrics.

These general lessons were consolidated from the current and historical
handoffs under `/content/drive/MyDrive/interpret`, its `special-lab-1` JSpace
work, and `/content/drive/MyDrive/interpret/preference`. Their experiment names,
old branches, stale paths, and scientific claims are intentionally not copied
into this shared bootstrap.

## Starting or resuming an agent

For Lab 2:

```bash
cd /content/labs
codex "Read aidataapps/resume.md, aidataapps/inprogress_lab2.md, and /content/handoff.md. Inspect live processes and SQL state, then continue Lab 2 without duplicating a running job."
```

For Lab 3, start Codex from the Lab 3 worktree and replace the status filename:

```bash
cd /content/worktrees/aidataapps-lab03
codex "Read aidataapps/resume.md and aidataapps/inprogress_lab3.md. Respect the worktree and GPU ownership rules, inspect state, then resume the next incomplete Lab 3 milestone."
```

On this trusted, disposable VM, Codex is configured in
`/root/.codex/config.toml` with `default_permissions = ":danger-full-access"`
and `approval_policy = "never"`. `codex --yolo` is a per-invocation bypass of
both approval and sandbox protections; use it only when the VM itself is the
intended isolation boundary.
