# LogWarden mac profile — Apple Silicon development plane

The mac profile runs the LogWarden foundation and (eventually) small-model
agent arms on an Apple Silicon Mac, with Azure Foundry Local as the serving
plane. It exists so foundation work, capture-plane development, and small-model
agent experiments can happen off the Colab GPU VM.

**Scientific status.** Everything produced on this profile is mac-plane
evidence. Foundry Local serves Microsoft-published ONNX builds through
WebGPU/Metal — different served artifacts from the digest-pinned vLLM CUDA
profiles in `config/models.json` — so mac results are never comparable rows in
the frozen Colab campaign, and the four campaign target profiles are unchanged.
The working branch is `aidataapps-logwarden-mac`; the frozen campaign lives on
`aidataapps-logwarden`.

## Verified environment (2026-08-22, M4 Max 48 GB)

| Piece | How it runs | Verified |
|---|---|---|
| SQL Server 2025 + FTS image | amd64 image under Docker Desktop Rosetta emulation (`compose.mac.yaml` pins `platform: linux/amd64`) | build + healthy; doctor disposition PASS (17.0.4075.5, exact vector, full-text, XE session, Query Store, permission matrix); 8/8 SQL integration tests |
| Chat serving | Foundry Local (`foundry` CLI 0.10.3), daemon pinned to `:8010` | `qwen3-4b-generic-gpu:2` decision canary ok at ~60 tok/s warm |
| Embedding serving | Same Foundry endpoint (`/v1/embeddings`) | `qwen3-embedding-0.6b-generic-gpu:1` returns 1024 dimensions |
| Node toolchain | node ≥ 20.19 required (rolldown native binding); use `nvm use 22` | `npm run check` green |

## How to run

```bash
cd aidataapps/logwarden
nvm use 22
./scripts/mac-host-init.sh        # .env with local secrets; requires Docker Desktop + Rosetta
./scripts/env-init.sh             # builds the FTS image (amd64 emulation) and starts SQL only
npm run run:init -- --campaign smoke
npm run db:setup
npm run doctor
npm run check
npm run test:sql
npm run mac:model -- up --profile qwen3-4b-foundry            # chat plane + canary evidence
npm run mac:model -- up --profile qwen3-embedding-0.6b-foundry # embedding plane + canary evidence
npm run mac:model -- down          # unload models, stop the daemon
```

`npm run mac:model -- up` ensures the Foundry daemon is on `CHAT_PORT` (8010),
downloads/loads the profile, runs a deterministic canary (temperature 0), and
writes `environment/mac-serving-<profile>.json` into the current run.

## Bench and report

`npm run mac:eval -- --profile <key>` runs the frozen 16-episode triage bench
(`config/mac-eval-episodes.json`) against one profile and writes row-level
transcripts (`tables/mac-eval/<key>.jsonl`) and summary metrics
(`metrics/mac-eval-<key>.json`) into the current run. Profiles with a
`baseUrl` (e.g. the MLX-served Gemma) use that external OpenAI-compatible
endpoint instead of Foundry.

`npm run mac:report` renders `reports/mac-eval-bench.html` in the run
directory from those retained rows (§28.5 discipline: every number is
computed from rows, never hand-entered). The prose comes from a narrative
JSON — verdict, tiles, ledes, dossiers, caveats, model display order and
palette slots — resolved from `--narrative <path>`, then
`reports/mac-eval-narrative.json` in the run, then
`templates/mac-eval-narrative.example.json` (the 2026-08-23 bench narrative,
kept as a worked example). The page template is
`templates/mac-eval-bench.template.html`; the merged data blob is also
written to `reports/mac-eval-bench-data.json` for inspection. A committed
snapshot of the first bench lives at `docs/reports/mac-eval-bench-20260823.html`.
The same flow works on Colab: run evals, author a narrative JSON, run
`mac:report`.

## Serving registry

`config/models.mac.json` (loaded by `src/models-mac.ts`) pins mac profiles by
Foundry variant id instead of image digest:

- `qwen2.5-0.5b-foundry` — plumbing smoke only (assumed too small for the
  agent contract; its canary emits valid JSON shape with weak content).
- `qwen3-4b-foundry` — primary mac agent candidate; same family and scale as
  the frozen registry's `qwen-smoke` control (Qwen3-4B).
- `qwen3-8b-foundry` — larger candidate; variant id pinned on first download.
- `qwen3-embedding-0.6b-foundry` — same base model as the campaign's Qwen
  embedding profile; ONNX build, so vectors are not interchangeable.

One Foundry endpoint serves chat and embeddings, so `CHAT_BASE_URL` and
`EMBEDDING_BASE_URL` both point at `http://127.0.0.1:8010/v1` on this profile.
The Foundry catalog has no Gemma models; Qwen3 is the closest small-model
match to the frozen registry's families.

## Recorded deviations from the Colab profile

- `MSSQL_MEMORY_LIMIT_MB` lowered to 4096 (Docker Desktop VM has ~8 GB total).
- No `nvidia-smi` prerequisite; GPU inventory records null in `run.json`.
- `--with-embedding` (CUDA vLLM) is refused; Foundry Local serves embeddings.
- Qwen3 chat canaries use the `/no_think` soft switch; the empty
  `<think></think>` prefix is stripped and recorded as `think_strip` — the
  transport contract's repair taxonomy (SPEC_ADDENDUM §A-1) will need to
  account for the reasoning channel when the agent runtime lands here.
- `BAAI/bge-large-en-v1.5` (secondary embedding, Tier 2) has no Foundry
  equivalent; `SECOND_EMBEDDING_BASE_URL` is unused on this profile.
