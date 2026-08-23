# Lab 02 in progress — ModelPrint

Last manually updated: 2026-08-23 01:40 UTC

Read `resume.md` first for multi-agent and recovery rules. The more detailed
machine-local narrative is `/content/handoff.md`; the watchdog copies it into
the Drive-backed run checkpoint on every cycle.

## Ownership and objective

- worktree: `/content/labs`
- branch: `aidataapps-modelprint`
- remote branch: `origin/aidataapps-modelprint`
- lab: `/content/labs/aidataapps/modelprint`
- Drive inputs: `/content/drive/MyDrive/aidataapps/lab02`
- governed plan: `docs/SPEC.md`
- governing addendum where it differs: `docs/SPEC_ADDENDUM.md`
- source Drive files:
  - `/content/drive/MyDrive/aidataapps/lab02/aidataapps_modelprint_lab_2_spec.md`
  - `/content/drive/MyDrive/aidataapps/lab02/aidataapps_modelprint_lab_2_spec_addendum.md`

The experiment builds a closed-set local-model output-attribution system over
four target models. It persists governed generations and likelihoods in SQL
Server 2025, derives semantic/style representations, evaluates attribution,
retrieval, geometry, clusters, same-source, OOD, and ANN behavior, and produces
F01–F19 reports, an explorer API, a native backup, a BACPAC, and a complete run
archive.

## Authoritative run

- run ID: `modelprint-full-20260822T230728Z`
- local run: `/content/labs/aidataapps/modelprint/runs/modelprint-full-20260822T230728Z`
- pointer: `/content/labs/aidataapps/modelprint/.current-run`
- Drive mirror: `/content/drive/MyDrive/aidataapps/lab02/runs/modelprint-full-20260822T230728Z`
- primary campaign: ID 3, 40,000 jobs
- primary hash: `52113ce90ed5302c0f40f55e79d5962aa692925721cec0ce3c2684c6947673d9`
- robustness campaign: ID 4, 501 variants per target profile, 2,004 jobs
- older campaigns 1 and 2 are excluded and must not be substituted
- latest pushed HEAD before coordination checkpoint: `ecb00f5`
- scientific freeze tag: `modelprint-mp2-freeze-v3`

The four generated target profiles, in residency order, are:

1. `qwen-3.8-27b` — `Qwen/Qwen3.8-27B`
2. `muse-glimmer-30b` — `meta-models/Muse-Glimmer-30B`
3. `gemma-4-31b` — `google/gemma-4-31B-it`
4. `olmo-3.1-32b-instruct` — `allenai/Olmo-3.1-32B-Instruct`

All exact revisions and vLLM image digests are pinned in
`config/models.json` and copied into the run manifest. Do not replace a target
with a similarly named release during this frozen campaign.

## Live state at this update

Qwen is complete:

- primary 10,000/10,000, zero failed
- robustness 501/501, zero failed
- prompted likelihood 8,001/8,001
- unprompted likelihood 7,984/8,001
- 17 one-token outputs legitimately have no first-token unprompted logprob; no
  values were imputed
- verified native backup exists in `runs/.../database/`
- its 62 GiB weight cache was removed only after backup and Drive mirror

Muse is active:

- strict port gate passed with six identical deterministic hashes
- tokenizer revision is explicit
- vLLM batch-invariant kernels are enabled for this model
- chat container: `aidataapps-modelprint-chat-muse-r1`
- primary generation command:

```bash
npm run generate -- --profile muse-glimmer-30b --resume --concurrency 32 --checkpoint-size 100
```

- last manual observation: 1,600/10,000 complete, zero failed at
  `2026-08-23T01:38:37Z`
- some Muse rows can exhaust the answer budget in reasoning and have an empty
  final-answer field; retain them as truncated per addendum C-15 and report the
  rate rather than silently regenerating or filling them

Gemma and OLMo generation have not started. Final cross-likelihood completion,
features, analyses, reports, BACPAC, archive, mirror, and reproducibility run
remain.

The four dirty tracked root documents are a partial mid-run report render and
must not be treated as final: `README.md`, `MODELPRINT_STATE_OF_RECORD.md`,
`MODELPRINT_ATTRIBUTION_REPORT.md`, and `MODELPRINT_CLAIMS_TABLE.md`. They are
captured by the recovery patch and should be regenerated after the complete
analysis before a report milestone commit.

## Environment and services

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, about 96 GiB VRAM
- Docker: 29.7.2 rootless
- rootless socket: `unix:///run/user/1000/docker.sock`
- SQL Server: 2025 RTM CU8 Developer, database `ModelPrint`, port 1433
- Node.js: 20.19.0
- SqlPackage: 170.4.83.3 in ignored `tools/sqlpackage`
- Qwen3 embedding server: port 8001
- BGE-large embedding server: port 8002
- rotating chat server: port 8000

Persistent expected containers:

- `aidataapps-rag-sqlserver-1`
- `aidataapps-modelprint-embedding-qwen-1`
- `aidataapps-modelprint-embedding-bge-1`
- exactly one `aidataapps-modelprint-chat-*` residency

Always prepare the shell before Docker or npm operations:

```bash
cd /content/labs/aidataapps/modelprint
source scripts/runtime-env.sh
```

`.env` is intentionally ignored and contains secrets. On a fresh Colab VM,
follow `.env.example` and `scripts/env-init.sh`; do not commit or copy secret
values into a handoff. The rootless SQL data and Hugging Face caches currently
live beneath:

```text
/home/codexdocker/.local/share/docker/volumes/aidataapps-rag-sqlserver-data/_data
/home/codexdocker/.local/share/docker/volumes/aidataapps-rag-huggingface-cache/_data/hub
```

Rootless Docker `exec`, `cp`, `inspect`, or removal can hang on this nested VM.
The lab uses `scripts/container-storage.sh` for direct, exact SQL-volume backup
copies. Inspect complete process commands and stop only the rotating chat API
and its EngineCore; never kill the two embedding EngineCore processes.

## Immediate resume procedure

First determine whether the generator and watchdog survived the agent/UI:

```bash
cd /content/labs/aidataapps/modelprint
source scripts/runtime-env.sh
git status --short --branch
git worktree list --porcelain
ps -eo pid,ppid,etimes,args | rg 'generate.ts|generate-robustness|score-likelihood|vllm serve|EngineCore|checkpoint-watchdog'
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
cat .current-run
tail -n 80 runs/modelprint-full-20260822T230728Z/checkpoints/watchdog.log
```

Do not start a second Muse generator if the command is alive. If it is absent,
the primary resume command above is idempotent and SQL-backed. Confirm current
counts with the generation manifest and SQL/doctor output rather than trusting
this timestamped prose.

## Twenty-minute checkpoint watchdog

Run one immediate cycle and then launch the singleton background loop:

```bash
npm run checkpoint:once
nohup ./scripts/checkpoint-watchdog.sh --interval-seconds 1200 \
  >/dev/null 2>&1 </dev/null &
```

The watchdog uses an exclusive lock and exits harmlessly if another copy owns
the run. Its outputs are under:

```text
runs/modelprint-full-20260822T230728Z/checkpoints/latest/
runs/modelprint-full-20260822T230728Z/checkpoints/watchdog.log
/content/drive/MyDrive/aidataapps/resume.md
/content/drive/MyDrive/aidataapps/inprogress_lab2.md
/content/drive/MyDrive/aidataapps/lab02/handoff.md
/content/drive/MyDrive/aidataapps/lab02/runs/modelprint-full-20260822T230728Z/checkpoints/
```

Each cycle pushes the current committed branch, makes a verified branch bundle,
captures a binary patch and untracked non-ignored source, copies all handoffs,
creates a checksummed online SQL backup, inventories the run, and mirrors it to
Drive. It does not auto-commit. To stop it cleanly after the final mirror:

```bash
touch runs/modelprint-full-20260822T230728Z/checkpoints/STOP
```

Review `watchdog.log`; a running process alone does not prove the last SQL or
Drive operation succeeded.

## Frozen campaign files

Do not edit these while campaigns 3/4 remain authoritative:

- `src/types.ts`
- `src/inference.ts`
- `src/prompt-bank.ts`
- `src/style.ts`
- `src/attribution.ts`
- `scripts/campaign-freeze.ts`
- `scripts/port-gate.ts`
- `scripts/generate.ts`
- governed configs, data, vendored specs, manifests, and lockfiles

Any necessary runtime correction must be minimal, outside the frozen set,
tested, and recorded append-only in `EXPERIMENT_LOG.md`.

## Remaining residency workflow

After Muse primary completes, while Muse is still resident:

```bash
npm run robustness:generate -- --profile muse-glimmer-30b --concurrency 32 --checkpoint-size 100
npm run likelihood:score -- --scorer muse-glimmer-30b --include-robustness --concurrency 32 --checkpoint-size 200
npm run checkpoint:once
```

Then make a milestone commit/push for any durable source/log changes, stop only
the Muse chat engine, verify port 8000 and VRAM, and evict only the exact Muse
weight cache after the SQL backup and Drive mirror succeed.

For Gemma, use a unique chat container, run its unchanged gate, then primary,
robustness, and likelihood stages:

```bash
export CHAT_CONTAINER_NAME=aidataapps-modelprint-chat-gemma-r1
npm run model -- start --profile gemma-4-31b
npm run port:gate -- --profile gemma-4-31b
npm run generate -- --profile gemma-4-31b --resume --concurrency 64 --checkpoint-size 100
npm run robustness:generate -- --profile gemma-4-31b --concurrency 64 --checkpoint-size 100
npm run likelihood:score -- --scorer gemma-4-31b --include-robustness --concurrency 64 --checkpoint-size 200
npm run checkpoint:once
```

Repeat with a unique OLMo chat name and profile `olmo-3.1-32b-instruct`.
During each scorer residency, score every target output available. Rotate prior
scorers again after later model generations to fill missing cross-likelihood
cells. If resource/time limits prevent a rectangular matrix, label it
`PARTIAL_LIKELIHOOD`; never imply completion from diagonal cells.

## Final analysis and archive

After generation, robustness, and intended cross-likelihood cells are complete:

```bash
npm run features:build
npm run controls:build
npm run derived:build
npm run phrases:build
npm run search:freeze
npm run evaluate:probes
npm run evaluate:retrieval
npm run evaluate:chunks
npm run evaluate:geometry
npm run evaluate:pairs
npm run evaluate:clusters
npm run evaluate:ood
npm run ann:benchmark
npm run reports
npm run db:backup
npm run db:bacpac
npm run run:archive
npm run run:mirror
npm run repro
```

The final run must include frozen manifests/gates, every raw JSONL and checksum,
tables/metrics/predictions, F01–F19, claims/captions, native `.bak`, BACPAC,
checksums/metadata, artifact inventory, resume record, and reproduction output.
Run archive/mirror again after the BACPAC and reproduction evidence exist.

Before the final generated-report commit and push:

```bash
npm run check
npm run test:sql
bash -n scripts/*.sh repro.sh
npm audit --audit-level=high
git diff --check
git status --short --branch
```

Create a completion tag only after `repro` succeeds and the final archive is
mirrored. Milestone history through this update is recorded in
`/content/handoff.md` and `git log`; pushed HEAD `ecb00f5` includes the Muse gate
diagnosis/recovery milestone.
