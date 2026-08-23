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

The VM has one NVIDIA RTX PRO 6000 Blackwell GPU. Lab 2 currently reserves the
chat-model residency on port 8000 and its two embedding services on ports 8001
and 8002. A second agent may write code, inspect data, or run CPU-only tests,
but it must not launch another large GPU model until the Lab 2 owner explicitly
releases the GPU.

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
