# Lab 02 ModelPrint — GPU-to-CPU handoff

Updated: 2026-08-23 16:31 UTC

This is the authoritative transfer guide for continuing Lab 02 without the
GPU VM. Read `/content/labs/aidataapps/resume.md` first when operating in
Colab; on another machine, this document and the transfer manifest are enough.

## Outcome at the handoff boundary

The expensive GPU phase is complete. Qwen 3.8 27B, Muse Glimmer 30B, Gemma 4
31B, and OLMo 3.1 32B each completed the frozen 10,000-output primary campaign
and 501-output robustness campaign with zero failed jobs. Cross-likelihood,
both embedding profiles, all five Qwen segment embedding families, style and
derived features, controls, feature provenance audit, phrases, geometry,
pairwise, clusters, frozen search corpora, and exact whole-output retrieval are
complete. No chat-model or embedding-model server is needed for the remaining
work because all required vectors and likelihoods are persisted in SQL and the
run artifacts.

The VM stopped at a clean, verified boundary. All five chunking variants are
complete. Four of 13 attribution-probe representations are complete (Qwen
raw/masked and BGE raw/masked), leaving nine for CPU. Every completed probe representation has an atomic
`manifests/probe-result-*.pkl` checkpoint; `--resume-completed` validates its
inputs/configuration before skipping it. The probe process exited while starting
the fifth representation, before another result checkpoint, and left no
temporary artifact. OOD and ANN were deliberately not started; ANN's full
matrix should run uninterrupted. There are no active evaluation/search jobs,
and all chat/embedding GPU services are stopped.

## Authoritative identifiers

- repository: `git@github.com:karlb-dev/labs.git`
- branch: `aidataapps-modelprint`
- lab directory: `aidataapps/modelprint`
- run ID: `modelprint-full-20260822T230728Z`
- primary campaign: ID 3, 40,000/40,000 generations
- robustness campaign: ID 4, 2,004/2,004 generations
- scientific freeze tag: `modelprint-mp2-freeze-v3`
- Drive lab root: `MyDrive/aidataapps/lab02`
- Drive run mirror: `MyDrive/aidataapps/lab02/runs/modelprint-full-20260822T230728Z`
- compact handoff: `MyDrive/aidataapps/lab02/cpu-handoff/modelprint-full-20260822T230728Z`

The original Drive plan and addendum are:

- `MyDrive/aidataapps/lab02/aidataapps_modelprint_lab_2_spec.md`
- `MyDrive/aidataapps/lab02/aidataapps_modelprint_lab_2_spec_addendum.md`

They are already vendored into `docs/SPEC.md` and `docs/SPEC_ADDENDUM.md` on
the branch, so they do not need to be downloaded separately.

## Download only this transfer set

Do not download the entire Drive run mirror. Download:

1. The whole compact handoff folder:
   `MyDrive/aidataapps/lab02/cpu-handoff/modelprint-full-20260822T230728Z/`.
   It contains the Git bundle fallback, Git status/patch recovery, this guide,
   coordination documents, checksums, and one compressed run-state archive.
   The run-state archive includes raw governed JSONL, manifests, atomic probe
   checkpoints, tables, metrics, reports, figures, environment evidence, and
   logs; it excludes database exports and redundant watchdog checkpoints.
2. Exactly these three native-backup files from
   `MyDrive/aidataapps/lab02/runs/modelprint-full-20260822T230728Z/database/`:
   `modelprint-full-20260822T230728Z-20260823T155623Z.bak`, its `.sha256`, and
   its `.json`. Do not download the older `.bak` files.
3. Optionally, download `ModelPrint-20260823T155430Z.bacpac` and its `.sha256`
   and `.json` sidecars from that same database directory. The native backup
   is the preferred full-fidelity artifact; the BACPAC is a tested portability
   fallback. `DATABASE_FILES.json` repeats these exact names, byte counts, and
   hashes so the transfer can be checked without relying on this prose.

Prefer cloning the GitHub branch. `repository.bundle` in the compact handoff
is an offline fallback, not an additional requirement when GitHub is reachable.

Explicitly exclude all of the following:

- every older `.bak` in the Drive database directory;
- the full `checkpoints/` directory (the compact handoff already carries a
  verified repository bundle and the current source patch);
- `.venv/`, `node_modules/`, `tools/`, `.env`, Docker volumes, and container
  layers;
- Hugging Face and vLLM caches. On the Colab VM these live under the rootless
  Docker volume tree, outside the run mirror, and are not needed on CPU;
- any Qwen, Muse, Gemma, OLMo, BGE, or embedding weight files.

Verify every downloaded file against `TRANSFER_SHA256.txt` and the database
sidecars before extracting or restoring it.

## Important Mac architecture constraint

SQL Server 2025 Linux containers are supported only on x86-64 Linux hosts.
Microsoft explicitly says Rosetta, QEMU, and other emulation/translation
environments are untested and unsupported:

<https://learn.microsoft.com/en-us/sql/linux/sql-server-linux-release-notes-2025?view=sql-server-ver17>

Therefore:

- On an Apple-silicon Mac, run Node/Python locally if desired, but point
  `SQLSERVER_HOST` at an x86-64 Linux SQL Server 2025 machine. Restore the
  native backup on that machine, or import the BACPAC from macOS to that remote
  server. The safest and usually fastest choice is to run the entire remaining
  CPU/SQL pipeline on that x86-64 machine and control it from the Mac.
- On an Intel Mac, Docker Desktop may run the pinned x86-64 image, but macOS is
  still outside Microsoft's supported SQL Server container host matrix. Treat
  it as an experimental continuation, not a supported deployment.

SqlPackage itself is supported on macOS and the lab installer now selects the
official macOS package automatically:

<https://learn.microsoft.com/en-us/sql/tools/sqlpackage/sqlpackage-download?view=sql-server-ver17>

## Reconstruct the source and run directory

With GitHub access:

```bash
git clone --branch aidataapps-modelprint --single-branch git@github.com:karlb-dev/labs.git
cd labs/aidataapps/modelprint
mkdir -p runs/modelprint-full-20260822T230728Z
tar -xzf /path/to/modelprint-full-20260822T230728Z-run-state.tar.gz \
  -C runs/modelprint-full-20260822T230728Z
printf '%s\n' 'runs/modelprint-full-20260822T230728Z' > .current-run
```

Offline Git fallback:

```bash
git clone /path/to/repository.bundle labs
cd labs
git switch aidataapps-modelprint
```

Do not copy the Colab `.env`; it contains machine-local credentials and a
rootless-container profile. Let the setup script create a new one.

## Prepare CPU dependencies and restore SQL

Install Node 20, Python 3.13, `jq`, `unzip`, and `rsync` first. On macOS with
Homebrew, one suitable starting point is:

```bash
brew install node@20 python@3.13 jq rsync
```

Ensure the selected Node and Python executables are first on `PATH` before
running the setup script. Python packages and Node packages are rebuilt from
the pinned lock files; do not transfer `.venv` or `node_modules` from Colab.

For a local x86-64 SQL container:

```bash
cd labs/aidataapps/modelprint
./scripts/cpu-analysis-init.sh --backup /path/to/the-recorded-backup.bak
```

This path installs only Node/Python dependencies, starts SQL Server, verifies
the backup checksum, restores `ModelPrint` without overwriting an existing
database, checks every schema migration (including explicit construction-era
hash reconciliation), verifies campaign row counts, and runs the build/tests.
It does not start vLLM or download any model.

For an external x86-64 SQL Server:

```bash
cd labs/aidataapps/modelprint
# Run once to create .env and install locked dependencies.
./scripts/cpu-analysis-init.sh --external-sql
# Set SQLSERVER_HOST, SQLSERVER_PORT, MSSQL_SA_PASSWORD, and MSSQL_DATABASE in .env,
# then validate the reachable restored database.
npm run db:verify-handoff
```

Restore the native backup on that server using its normal SQL backup workflow.
Alternatively, from the Mac, import the BACPAC into an empty target database:

```bash
npm run db:import-bacpac -- --bacpac /path/to/the-recorded-file.bacpac
```

The import wrapper verifies the BACPAC checksum, imports only into a safe new
database name, then explicitly sets compatibility level 170 and
`PREVIEW_FEATURES=ON` before validation. This matters because BACPAC does not
carry that database-scoped preview setting. This exact BACPAC was test-imported
on the VM; the resulting migrations, campaigns, and key row counts all passed.

The native restore script intentionally refuses to replace an existing
`ModelPrint` database. The BACPAC import also expects an empty/nonexistent
target. Keep the downloaded database export until the final reproduction gate
passes.

Recommended free space is at least 40 GiB beyond the downloaded files: the
native backup is compressed, while the restored database, SQL log, temporary
ANN corpus/index, and final exports are not. Give SQL Server at least 8 GiB of
host memory if possible. The frozen experiment ran correctly with SQL seeing
4 GiB, but exact vector scans were memory-grant constrained.

## Remaining work in dependency order

Never rerun generation, likelihood, segmentation, embedding, feature, control,
search-freeze, geometry, pairwise, or cluster stages. Adopt only incomplete
CPU stages:

1. Resume attribution probes. Use a worker count appropriate to the CPU; the
   completed representation checkpoints are hash/config validated.

   ```bash
   npm run evaluate:probes -- --jobs "$(sysctl -n hw.logicalcpu 2>/dev/null || getconf _NPROCESSORS_ONLN)" --resume-completed
   ```

2. After probes finish, run OOD. It consumes persisted controls/vectors and
   probe checkpoints; no inference server is required.

   ```bash
   npm run evaluate:ood
   ```

3. Run the full ANN benchmark without another SQL-heavy job in
   parallel. It sweeps both frozen spaces, all governed corpus prefixes, 1,000
   queries, `k=1,5,10,20`, and oversampling `1,2,5,10`. It is not checkpointed
   within a prefix sweep, so let it finish uninterrupted.

   ```bash
   npm run ann:benchmark
   ```

4. Generate reports, verify the API/build/tests, create final exports, and
   archive/mirror. `run:archive` regenerates the inventory after all outputs
   exist.

   ```bash
   npm run reports
   npm run check
   npm run test:sql
   npm run db:verify-handoff
   npm run db:backup
   npm run db:bacpac
   npm run run:archive
   npm run repro
   ```

5. `npm run repro` verifies the artifact inventory, reconstructs prediction
   metrics, restores the checksum-valid native backup to a temporary database,
   regenerates the headline report against that restore, byte-diffs it, and
   removes the temporary database. Commit/push and create the final result tag
   only after this passes.

The default ANN and probe settings remain the governed settings. Do not reduce
permutations, bootstraps, query count, prefixes, representations, or suites
merely to shorten a local run. Wall time on a laptop with fewer than 48 logical
CPUs can be several times longer than the Colab timings.

Chunk evaluation is already complete and must not be rerun. Its final aggregate
artifacts and SQL rows are present in both the run-state archive and database.

## Clean-stop state and exact transfer files

This is the exact durable boundary. `TRANSFER_MANIFEST.json` in the compact
handoff records the final Git tip and run-state archive checksum, avoiding a
self-referential hash inside the archive itself.

- probes: 4/13 representations complete; nine remain. Atomic result-checkpoint
  SHA-256 values are Qwen raw
  `70dc4d5fcfbc3bfea87725532c450d96b31f2646224caf17919265147940ede2`,
  Qwen masked
  `d8e1fd15890372d022167467346beeaa394a4c1cb2cd39c93bd9f65320a5e7e3`,
  BGE raw
  `684bc24914a83b9e6d4fc56da4a3248a64090c5c1ecd0ea0a0d5d8193602e4fa`,
  and BGE masked
  `48ded96bc7ee7199fc25e5ff29b53e6ad974f2f2c12612fe953ebadf24024dbb`.
- chunks: 5/5 complete, 13,584 predictions and 287,470 neighbors. Metrics,
  predictions, and neighbors SHA-256 values are
  `0aabf33a4d1800bb6cb0d9e9127fbbd3f8f3b4c6b9227f78a8dc3f7e6b6c4ba8`,
  `a572fae5bc3f564f0a07ac98a0916ff8f1b916215a7c20f051e256d5948b10b3`,
  and `3e500b8c68734b2c93ef6166d3fa38b4db9965533ab855da6e9d6f377366152f`.
- ANN: intentionally pending, zero benchmark runs; OOD: pending.
- SQL state: no active search run, 132 metric rows, 21 search runs, 113
  prediction runs, and zero attribution models (probe SQL persistence occurs
  after all representations finish).
- native backup: `modelprint-full-20260822T230728Z-20260823T155623Z.bak`,
  4,264,128,512 bytes, SHA-256
  `a315c1b3d81e9650d45250d3b84e90a2606b628c0e07844782cedb7734eb4278`.
  It passed checksum validation, `RESTORE ... WITH CHECKSUM` into a fresh
  database, and the full handoff validator; the temporary database was dropped.
- BACPAC: `ModelPrint-20260823T155430Z.bacpac`, 5,188,484,189 bytes, SHA-256
  `1dddedcc8211717e6a831dfe82b5eb5777002260ef441622ba728bcb9c9d8bfd`.
  It passed checksum and full ZIP/BCP integrity checks, imported into a fresh
  database in 3:14, passed validation after the required preview setup, and the
  temporary database was dropped.
- Drive readback: an earlier in-flight 861,923,324-byte BACPAC mirror was
  detected by the final independent size/hash gate and replaced through a
  unique temporary upload. The final Drive path now re-reads at 5,188,484,189
  bytes with the hash above. Mirroring refuses sidecar-less in-progress BACPACs
  and requires size/hash verification before and after promotion.
- migration/schema gate: PASS. A fresh five-migration database matches live
  across 57 tables, 454 columns, 139 index-column records, 37 foreign-key
  records, and 61 constraints with no differences.
- compact handoff:
  `MyDrive/aidataapps/lab02/cpu-handoff/modelprint-full-20260822T230728Z/`.
  Verify its contents with `TRANSFER_SHA256.txt`; exact archive size/hash and
  Git head are in `TRANSFER_MANIFEST.json`.

## Proven completed evidence that must remain unchanged

- Exact retrieval: 12/12 representations, 1,621,280 retained neighbors,
  81,064 predictions, and 500/500 SQL/NumPy top-20 equivalence checks.
- Maximum matching-list distance error: `3.5762786865234375e-07` versus the
  frozen `1e-4` tolerance.
- Full feature audit: `COMPLETE`, with 520,083/520,083 eligible Qwen segment
  vectors and all six SQL UTF-16 boundary repairs reconstructed exactly.
- Search-corpus hash:
  `5880b91e53ff0c102ef156f564b668de8c2a38a66377001685f82b566d59e8f0`.
- All unavailable likelihood values are explicitly audited; none are imputed.

If any restored count or digest differs, stop. Do not regenerate frozen data
to make the discrepancy disappear; recover from the checksum-valid backup and
run-state archive instead.
