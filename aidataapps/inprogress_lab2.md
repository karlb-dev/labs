# Lab 02 in progress — ModelPrint

Last manually updated: 2026-08-23 14:52 UTC

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
- parallel-probe milestone: `d5c2f29`; use `git rev-parse HEAD` for the live tip
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

This snapshot supersedes any process IDs or pending-stage language in the
historical detail below.

All governed generation and likelihood work is complete for Qwen 3.8 27B,
Muse Glimmer 30B, Gemma 4 31B, and OLMo 3.1 32B: 40,000 primary and 2,004
robustness generations have zero failures. SQL contains all 127,760 prompted
scorer/output cells and 127,590 unprompted cells. The 170 unavailable
unprompted cells are audited one-token outputs without a first-token
distribution; 64 empty-final Muse rows are separately unavailable. No value
was imputed. Port 8000 is free; only embedding ports 8001/8002 use the GPU.

Features and provenance audit are complete: 520,083 eligible Qwen segment
embeddings, 559,964 segment records, 81,176 artifact/style/scalar/whole-vector
rows, and 42,004 generation references. Six UTF-16 boundary splits were
repaired only at embedding input; source text is unchanged. Audit status is
`COMPLETE`. The frozen search-corpus hash is
`5880b91e53ff0c102ef156f564b668de8c2a38a66377001685f82b566d59e8f0`.
Controls, derived features, phrases, geometry, pairwise, and clustering are
also complete.

Active jobs to adopt and never duplicate:

- attribution probes/LOFO: session `55709`, PID 913301, 48 workers,
  `--resume-completed`; first representation is complete, while the second has
  complete suite nulls plus one atomic eligible-family null checkpoint
- exact retrieval: complete across 12 representations; 1,621,280 neighbors,
  81,064 predictions, and 500/500 SQL/NumPy list equivalence
- exact chunk retrieval: session `38173`, PID 882882, six workers; restarted
  with stable generation/segment composite query identities after run 5 rolled
  back with zero accepted neighbors
- checkpoint watchdog: session `16623`, PID 545189

The exact evaluators bind scans to `MAXDOP 1`; their SQL grants are healthy.
ANN is intentionally paused until chunk evaluation finishes. Its preview DDL
and scratch schema are corrected, including the legacy ANN requirement for a
single four-byte `INT` clustered identity key. Run ANN alone afterward.

Remaining order: complete probes and chunks, run OOD after probes, run ANN
after chunk exact scans, render reports, verify API/tests, create final native
backup and BACPAC, archive/mirror, restore/reproduce, then commit/push/tag. The
four dirty root reports are partial generated output and must be regenerated.

## Historical detailed state (superseded where noted above)

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

OLMo first residency is complete and durably backed up:

- profile `olmo-3.1-32b-instruct`, pinned revision
  `ac0587e4a7744a551c059d8cd17ba220bc940dae`
- completed residency container `aidataapps-modelprint-chat-olmo-r1`
- the 60.04 GiB checkpoint downloaded and loaded successfully at the frozen
  GPU utilization 0.78; no runtime override or batch-invariant mode is active
- available KV cache is 12.18 GiB / 49,863 tokens, 3.04x the frozen 16K context
- runtime profile hash:
  `261bd96df4ed073845a192b214d28a891e522041579f065dd455b6535203e9a6`
- strict gate passed on the first attempt with six identical hashes
  `0e6aa633784346ef6d8a0825219e5485ef9a242f83be91ff26423c10706ff73c`
- gate file SHA-256:
  `39b191343db714d617f784247bcdd47cd364d2af41533aeb9b1a36e884bc54de`
- primary completed 10,000/10,000 with zero failures; raw SHA-256
  `90bff0cf46eb89e1f3aae06a23a8885d4fe0a1e415966aa8ee4e1d36bed323ad`
- robustness completed 501/501 with zero failures; raw SHA-256
  `d25a6d61f5c836a099e79d5d2d2507fc08d6990b398c6d1203dc338bb9406a1e`
- likelihood selected all 31,940 eligible non-empty rows across the four
  targets: prompted 31,940/31,940; unprompted 31,878/31,940
- all 62 prompted-only rows were audited from retained JSONL: every row has a
  valid prompted score, exactly one OLMo output token, and only the expected
  no-first-token unprompted-logprob condition; target counts are Qwen 20,
  Muse 4, Gemma 28, and OLMo 10; no value was imputed
- 64 empty-final Muse rows are separately unavailable
- likelihood raw SHA-256:
  `7e5c8f0196da3cc7e962f54347e44d44f263eaf79455b4525ed2e003604995aa`
- pre-eviction native backup and Drive copy match at SHA-256
  `55e9555b33dfa1b9e25b23ddf6952b7648eedfba4df467f887c2d2b622dff935`

The prior-scorer cross-likelihood fill rotation is also complete:

- all 127,760 prompted scorer/output cells are persisted
- 127,590 unprompted cells are persisted
- all 170 missing unprompted cells are audited one-token outputs with no
  first-token distribution: Qwen 50, Muse 0, Gemma 58, OLMo 62
- 64 empty-final Muse rows are separately unavailable; no value was imputed
- status: `COMPLETE_WITH_DOCUMENTED_UNAVAILABLE`
- matrix evidence:
  `runs/.../metrics/cross-likelihood-completeness.json`
- final pre-analysis SQL backup and Drive copy SHA-256:
  `2c6b87c27685b83d7ee8a123b8f5f67d96f83fe58949c323a43dea5216e1f46d`
- all rotating chat engines are stopped and their exact re-downloadable caches
  are evicted; only embedding ports 8001 and 8002 remain on the GPU

Feature construction is active. The 42,004-generation reference-token/segment
transaction, 81,176 style/scalar artifacts, both prompt spaces, and both
81,176-row whole-output spaces are complete. After 248,832 of 520,083 eligible
Qwen segment vectors had persisted, the idempotent resume selected the 271,251
SQL-missing rows and raised only the HTTP batch size from 64 to 256. Adopt
managed session `78572` / PID 597700. Split UTF-16 surrogate boundaries from
SQL-native chunks are repaired only at embedding input with a logged U+FFFD
policy. Run `npm run features:audit` after the process exits.

Controls (2,612 items), residual/likelihood derived features, phrase features,
SQL geometry, and pairwise evaluation are complete and mirrored. SQL geometry
shows prompt dominance for Qwen semantic (model-over-prompt neighbor win rate
0.0383) and strong model alignment for style/fingerprint (0.9857/0.9791).
Pairwise hard-subset AUROC is 0.5653, disposition `NO_SUPPORTED_SIGNAL`.
Clustering is complete across 15 representation/method combinations: 13 are
`CLUSTER_VISUAL_ONLY`; prompt-centered HDBSCAN and fingerprint64 k-means are the
two implementation-classified model-aligned cases. Full 200-permutation probes
and LOFO rotations are active in session `81458` / PID 565820 with 32 workers;
the scientific settings remain 200 permutations, 1,000 bootstraps, and the
same data, seeds, splits, and models. Search freeze, retrieval/chunks, OOD, ANN,
reports, BACPAC, archive, and final reproduction remain.

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

Do not reload OLMo or any chat model. Feature construction and audit are
complete. Adopt the live evaluator processes listed in the current snapshot;
inspect their output and SQL evidence before starting a replacement. If the VM
reclaimed them, each evaluator is idempotent/resumable, but run exact retrieval
and chunk evaluation before ANN and never overlap ANN with exact SQL scans.
Run OOD only after attribution probes have finished.

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

## Residency workflow — complete

All first residencies and the Qwen→Muse→Gemma fill rotation are complete,
verified, backed up, mirrored, and released. Do not reload a chat model for the
remaining pipeline. Port 8000 is free; embedding services on ports 8001/8002
remain required for feature construction.

## Final analysis and archive

Feature audit and search freeze are complete. Resume only stages without a
finished metric/database record, in this dependency order:

```bash
npm run evaluate:retrieval
npm run evaluate:chunks
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
