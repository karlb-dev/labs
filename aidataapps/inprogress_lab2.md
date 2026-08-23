# Lab 02 in progress — ModelPrint

Last manually updated: 2026-08-23 07:51 UTC

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
- latest pushed baseline before this update: `44364c5` (run `git rev-parse HEAD`
  because later watchdog-safe milestone commits supersede this prose)
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

Muse is complete:

- strict port gate passed with six identical deterministic hashes
- tokenizer revision is explicit
- vLLM batch-invariant kernels are enabled for this model
- primary generation completed 10,000/10,000 with zero failed at
  `2026-08-23T03:59:58Z`
- robustness generation completed 501/501 with zero failed; raw SHA-256
  `ed7faf0cd842f1516f77c7ea706b30be4d8740b1dcd3e357afdbf07075c77de4`
- the corrected selective likelihood pass recovered all 61 Unicode-alignment
  gaps with zero failures
- cumulative audit: 15,938 eligible non-empty rows have both prompted and
  unprompted values; both missing counts are zero; 64 retained empty-final
  truncated rows are explicitly unavailable and were not imputed
- cumulative raw SHA-256:
  `9d59cb2fbc7e2fdde8168e6654b19c53407b29aaf59f8a63668cf6f6a1682193`
- final verified native backup SHA-256:
  `d53eeed6ba71dc7147cab8f6fc48c8ed6a3aaf313ee98bee7cfd75180c08d3f4`
- its exact weight cache was evicted only after local/Drive evidence and backup
  hashes matched; it is re-downloadable at the pinned revision

Gemma is complete:

- profile `gemma-4-31b`, pinned revision `842da3794eaa0b77d5f08bae87a17459d91ff475`
- completed residency container `aidataapps-modelprint-chat-gemma-r4`
- attempt 1 stopped before the gate because GPU utilization 0.78 provided
  13.22 GiB KV cache versus 13.76 GiB required for frozen 16K context
- attempt 2 reached the gate at utilization 0.80 and exposed batch-sensitive
  greedy output; both stopped attempts are retained as run evidence
- final runtime uses validated `CHAT_GPU_MEMORY_UTILIZATION=0.80` and
  `VLLM_BATCH_INVARIANT=1`; runtime identity hash is
  `5cfe529ebcb21f66e51cd070f13fa2ca172fc812ea7b0d03bd036e65b2f65831`
- strict gate passed with six identical deterministic hashes; gate artifact
  SHA-256 is `f3e39acac7a475ef7d9ec675bb57b9e93125753ddd172c1f507e107065dad9b6`
- primary completed 10,000/10,000 with zero failures; raw SHA-256
  `4136aeffa1b190f492c52467d401bd764dacbb7b71fd18884096a71f4ed44fe1`
- robustness completed 501/501 with zero failures; raw SHA-256
  `27828adff9f852c182b4debbc646ab19963f36c2ff53923e40a5596044fabe80`
- likelihood selected 23,939 eligible non-empty Qwen/Muse/Gemma rows:
  prompted 23,939/23,939; unprompted 23,890/23,939
- all 49 missing unprompted values were audited from the retained JSONL: each
  row has a valid prompted score, exactly one Gemma output token, and vLLM
  returned no first-token unprompted logprob; no value was imputed
- the 49 rows break down as Qwen 20, Muse 1, Gemma 28; 64 empty-final Muse
  rows are separately recorded as unavailable
- likelihood raw SHA-256:
  `775df32e203cb9774846696634adf33a680320ddc4e3ec7a8b9a065130d1371b`
- pre-eviction native backup and Drive copy match at SHA-256
  `5d3c7383be0a482173770abc1e260e583eafe99c585f39a8b8687c16c989b7bb`
- Gemma stopped cleanly; its exact one-repository/one-revision cache entry was
  evicted with `hf cache rm`, freeing 62.6 GB; it remains re-downloadable at
  the pinned revision

OLMo is active:

- profile `olmo-3.1-32b-instruct`, pinned revision
  `ac0587e4a7744a551c059d8cd17ba220bc940dae`
- residency container `aidataapps-modelprint-chat-olmo-r1`
- the 60.04 GiB checkpoint downloaded and loaded successfully at the frozen
  GPU utilization 0.78; no runtime override or batch-invariant mode is active
- available KV cache is 12.18 GiB / 49,863 tokens, 3.04x the frozen 16K context
- runtime profile hash:
  `261bd96df4ed073845a192b214d28a891e522041579f065dd455b6535203e9a6`
- strict gate passed on the first attempt with six identical hashes
  `0e6aa633784346ef6d8a0825219e5485ef9a242f83be91ff26423c10706ff73c`
- gate file SHA-256:
  `39b191343db714d617f784247bcdd47cd364d2af41533aeb9b1a36e884bc54de`
- active primary command:

```bash
npm run generate -- --profile olmo-3.1-32b-instruct --resume --concurrency 64 --checkpoint-size 100
```

- first durable checkpoint: 100/10,000, zero failed at
  `2026-08-23T07:50:31Z`; later checkpoint/SQL counts supersede this value

OLMo robustness/likelihood, final cross-likelihood completion, features,
analyses, reports, BACPAC, archive, mirror, and reproducibility run remain.

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

Do not start a second OLMo generator if the command is alive. If it is absent,
rerun the exact primary command above; `--resume` reconciles the frozen config
hash and SQL state before selecting missing rows. Inspect processes,
checkpoints, manifests, and SQL before resuming any later OLMo stage.

## Twenty-minute checkpoint watchdog

Run one immediate cycle and then launch the singleton background loop:

```bash
npm run checkpoint:once
nohup ./scripts/checkpoint-watchdog.sh --interval-seconds 1200 \
  >/dev/null 2>&1 </dev/null &
```

When launching through a Codex execution runner, use a managed persistent
terminal session for the loop and record/adopt that session; this runner can
reap an ordinary detached `nohup` child when the launch call ends. The lock
still prevents duplicates. In a normal notebook terminal, the `nohup` form is
the fallback, but verify the PID and a new `WATCHDOG started` log record before
assuming it survived.

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

Qwen, Muse, and Gemma are fully complete for their first residencies and are no
longer resident. OLMo is gated and its primary generator is active. After
primary completion, run:

```bash
npm run robustness:generate -- --profile olmo-3.1-32b-instruct --concurrency 64 --checkpoint-size 100
npm run likelihood:score -- --scorer olmo-3.1-32b-instruct --include-robustness --concurrency 64 --checkpoint-size 200
npm run checkpoint:once
```

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
`/content/handoff.md` and `git log`; use `git rev-parse HEAD` rather than an
older prose hash when resuming.
