# Lab 02 in progress — ModelPrint

Last manually updated: 2026-08-23 16:32 UTC

Read `resume.md` first. This file is the current Lab 02 state of record and
supersedes old process IDs or active-stage language in Git history and logs.
The detailed CPU transfer guide is `handoff_lab02_to_cpu.md`; the machine-local
summary is `/content/handoff.md`.

## Ownership and authoritative run

- worktree: `/content/labs`
- branch: `aidataapps-modelprint`
- remote: `git@github.com:karlb-dev/labs.git`
- lab: `/content/labs/aidataapps/modelprint`
- run ID: `modelprint-full-20260822T230728Z`
- local run: `/content/labs/aidataapps/modelprint/runs/modelprint-full-20260822T230728Z`
- Drive root: `/content/drive/MyDrive/aidataapps/lab02`
- Drive run: `/content/drive/MyDrive/aidataapps/lab02/runs/modelprint-full-20260822T230728Z`
- compact transfer: `/content/drive/MyDrive/aidataapps/lab02/cpu-handoff/modelprint-full-20260822T230728Z`
- primary campaign: ID 3, 40,000/40,000 complete, zero failures
- robustness campaign: ID 4, 2,004/2,004 complete, zero failures
- scientific freeze tag: `modelprint-mp2-freeze-v3`

The governing Drive inputs are vendored as `docs/SPEC.md` and
`docs/SPEC_ADDENDUM.md`; the addendum wins where they differ. Do not substitute
similarly named models or revisions for this frozen run.

## No active experiment job

There is no Lab 02 analysis or model process to adopt. The prior probe process
stopped at an atomic result boundary, the chunk evaluator finished, the
watchdog is stopped by `runs/.../checkpoints/STOP`, and ANN's launch was
deliberately canceled. All chat and embedding containers were stopped after
the database exports; only the healthy SQL Server container remains.

The four modified root reports—`README.md`,
`MODELPRINT_STATE_OF_RECORD.md`, `MODELPRINT_ATTRIBUTION_REPORT.md`, and
`MODELPRINT_CLAIMS_TABLE.md`—are partial generated output. Preserve them in the
transfer patch, but regenerate them after probes/OOD/ANN rather than committing
them as final claims.

## Completed work

The expensive GPU phase is complete for Qwen 3.8 27B, Muse Glimmer 30B, Gemma
4 31B, and OLMo 3.1 32B. Generation, robustness, all prompted likelihoods and
all available unprompted likelihoods, feature construction and provenance,
five segment families, BGE/Qwen whole vectors, style/scalars, controls,
residual/likelihood/phrase features, geometry, pairwise, clustering, search
freeze, exact retrieval, and exact chunk retrieval are complete.

Important retained facts:

- 40,000 primary and 2,004 robustness generations, zero failures
- 127,760/127,760 prompted scorer/output cells
- 127,590 unprompted cells; the remaining 170 are exhaustively audited
  one-token unavailability, never imputed
- 64 empty-final Muse rows separately unavailable
- 520,083/520,083 eligible Qwen segment vectors; six SQL UTF-16 boundary
  repairs reconstructed exactly at embedding input only
- frozen search-corpus hash:
  `5880b91e53ff0c102ef156f564b668de8c2a38a66377001685f82b566d59e8f0`
- exact retrieval: 12/12 representations, 1,621,280 neighbors, 81,064
  predictions, and 500/500 SQL/NumPy top-20 equivalence checks
- chunks: 5/5 segmenters, 13,584 predictions and 287,470 neighbors; do not
  rerun this completed stage

## Probe boundary and remaining science

Four of 13 probe representations are complete; nine remain. Atomic result
checkpoint hashes are:

- Qwen raw: `70dc4d5fcfbc3bfea87725532c450d96b31f2646224caf17919265147940ede2`
- Qwen masked: `d8e1fd15890372d022167467346beeaa394a4c1cb2cd39c93bd9f65320a5e7e3`
- BGE raw: `684bc24914a83b9e6d4fc56da4a3248a64090c5c1ecd0ea0a0d5d8193602e4fa`
- BGE masked: `48ded96bc7ee7199fc25e5ff29b53e6ad974f2f2c12612fe953ebadf24024dbb`

The stopped process had begun the fifth representation's grouped-CV selection
but had not completed another result checkpoint and left no `.tmp` file.
Resume only with `--resume-completed`; its hash/config gates validate completed
files before skipping them. Attribution-model SQL rows remain zero because the
evaluator persists those only after all representations finish.

CPU dependency order:

1. restore/import the database and run `npm run db:verify-handoff`;
2. resume the nine probe representations with `--resume-completed`;
3. run OOD after probes;
4. run the full ANN benchmark alone;
5. regenerate reports and run build/unit/SQL checks;
6. create final backup, BACPAC, archive, mirror, and full reproduction; and
7. commit/push/tag only after reproduction passes.

Exact commands and Mac setup are in `handoff_lab02_to_cpu.md`. Never rerun
generation, likelihood, feature, control, retrieval, chunk, geometry, pairwise,
or cluster stages. Do not reduce governed permutations, bootstraps, ANN matrix,
representations, suites, or queries to shorten laptop runtime.

## Clean database artifacts

Preferred native backup:

- `modelprint-full-20260822T230728Z-20260823T155623Z.bak`
- 4,264,128,512 bytes
- SHA-256 `a315c1b3d81e9650d45250d3b84e90a2606b628c0e07844782cedb7734eb4278`
- checksum and isolated native restore/full validator: PASS

Optional portable BACPAC:

- `ModelPrint-20260823T155430Z.bacpac`
- 5,188,484,189 bytes
- SHA-256 `1dddedcc8211717e6a831dfe82b5eb5777002260ef441622ba728bcb9c9d8bfd`
- full ZIP/BCP scan and isolated import/full validator: PASS

The final independent Drive readback caught and replaced an earlier
861,923,324-byte in-flight BACPAC copy. The final Drive object re-reads at the
full 5,188,484,189 bytes with the hash above. The mirror script now refuses
unfinished exports and promotes a unique verified upload only after both size
and digest pass.

BACPAC does not retain database-scoped `PREVIEW_FEATURES`. The tested import
wrapper now sets compatibility 170 and preview features before validation.
The validator refuses SQL Server below 2025, compatibility other than 170,
disabled preview, unresolved migration drift, incomplete campaigns, or active
search runs.

Current live validation is PASS: SQL Server 2025 CU8 `17.0.4075.5`, database
compatibility 170, preview enabled, 132 metric results, 21 search runs, zero
active search runs, 113 prediction runs, and zero ANN runs. Both isolated test
databases were dropped. Fresh source migrations match live with zero differences
across 57 tables, 454 columns, 139 index-column records, 37 foreign keys, and
61 constraints. TypeScript build, all 17 unit tests, and SQL vector integration
also pass.

## Transfer and recovery

The Mac should download only:

1. the entire compact CPU-handoff folder;
2. the exact native backup above plus its `.sha256` and `.json`; and
3. optionally, the exact BACPAC above plus its `.sha256` and `.json`.

Do not download the whole run mirror, older backups, redundant checkpoints,
model caches/weights, `.venv`, `node_modules`, `.env`, tools, Docker volumes,
or container layers. `DATABASE_FILES.json` and `TRANSFER_SHA256.txt` are the
authoritative transfer inventories.

SQL Server 2025 Linux containers are supported only on x86-64 Linux. An
Apple-silicon Mac should use an external x86-64 SQL Server 2025 host and run
Node/Python locally only if that host is reachable. SqlPackage itself supports
macOS, and the lab installer selects its macOS build.

## Frozen files

Do not edit the governed campaign inputs: `src/types.ts`, `src/inference.ts`,
`src/prompt-bank.ts`, `src/style.ts`, `src/attribution.ts`,
`scripts/campaign-freeze.ts`, `scripts/port-gate.ts`, `scripts/generate.ts`,
governed config/data/spec files, freeze manifests, or lockfiles. Any necessary
runtime correction must be minimal, tested, and recorded append-only in
`EXPERIMENT_LOG.md`.

The VM booted at `2026-08-22 17:38:53 UTC`; 17:38 UTC is the hard 24-hour
point. GitHub is authoritative for committed source; Drive is authoritative for
large run/database/transfer artifacts. Final exact Git tip and run-state
archive hash are recorded in the compact pack's `TRANSFER_MANIFEST.json`.
