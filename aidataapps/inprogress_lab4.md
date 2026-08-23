# Lab 04 in progress — GhostType

Last updated: 2026-08-23 (update at every stage boundary and before/after long launches)

## Ownership and run

- branch: `aidataapps-ghosttype` · lab: `aidataapps/ghosttype` · machine: Karl's Mac (local-first)
- current run: `ghosttype-cpu-dev-20260823T213801Z` (see `.current-run`)
- SQL: container `aidataapps-ghosttype-sqlserver-1` (port 1435, Rosetta, volume persists across laptop shutdown)
- model servers when needed: mlx_lm.server :8020, mlx_vlm.server :8021 (start via ghosttype scripts or `~/repos/foundary/chat_hosts.sh`)

## Pause/resume contract (Karl's laptop-shutdown reality)

- Every stage is idempotent; the SQL volume and pushed git are the durable state.
- Long runners use per-request status rows in `completion.requests` as the
  resume cursor: re-running the same command skips complete rows.
- To pause: stop the running command (Ctrl-C safe at any point); optionally
  `docker stop aidataapps-ghosttype-sqlserver-1`. To resume after boot:
  `docker start aidataapps-ghosttype-sqlserver-1`, then re-run the command in
  "Next command" below. Model servers restart on demand.
- Commit+push happens at every boundary; worst-case loss ≈ the in-flight stage
  chunk (target ≤ 20 min).

## Stage status

| Stage | Status |
|---|---|
| GT-0 intake + scaffold | DONE (pushed) |
| GT-1 dataset import + gates | DONE — 588/588, gates 588/588, 2 package defects dispositioned (`config/dataset-dispositions.json`) |
| GT-3 ScriptDom parse oracle | DONE — 479 pass / 8 package-defect findings / 101 n_a; parser pinned 170.191.0; recompose-v1 frozen |
| GT-2 catalog contexts (from package catalogs) | DONE — 14 snapshots, 520 objects, 2,440 columns, 0 unparsed; 566/566 must-refs resolve |
| GT-5 deterministic baselines | DONE — B0–B4 scored in eval.row_scores; H1 floor: max 4/487 exact |
| GT-7 replay runner + extraction | DONE — E4B campaign complete 588/588 (H1 not supported for 4B vehicle; see EXPERIMENT_LOG 2026-08-24) |
| Port gates (4 MLX profiles) | RUNNING (background; canaries + determinism per profile) |
| Mac quality campaign (4 profiles × eligible rows) | NEXT — order: qwen, gemma-26b, olmo, muse; each resumable |
| Scoring + reports + repro | TOOLING DONE — metrics:compute, reports (7 Tier-1 docs), repro.sh rows, db:backup all live; final numbers land as campaigns complete |

## Next command

```bash
cd /Users/karl/repos/labs/aidataapps/ghosttype
export PATH=$HOME/.nvm/versions/node/v22.22.1/bin:$PATH
# 1. ensure servers: cd ~/repos/foundary && ./chat_hosts.sh start
# 2. resume/continue the current model pass (skips done rows automatically):
npm run quality:run -- --profile gemma-4-e4b-mlx --arm M1-packaged --campaign cpu-dev
# 3. after E4B completes: npm run port:gate           (all 5 profiles)
# 4. then per target profile (each resumable, run sequentially):
#    npm run quality:run -- --profile qwen-3.8-27b-mlx  --arm M1-packaged --campaign mac-quality
#    npm run quality:run -- --profile gemma-4-26b-a4b-mlx --arm M1-packaged --campaign mac-quality
#    npm run quality:run -- --profile olmo-3.1-32b-mlx  --arm M1-packaged --campaign mac-quality
#    npm run quality:run -- --profile muse-glimmer-30b-mlx --arm M1-packaged --campaign mac-quality
# 5. anytime: npm run metrics:compute && npm run reports   (idempotent)
# 6. stage boundaries only: npm run db:backup
```

## Notes

- Dataset defects found by gates and dispositioned (not patched): two split
  groups spanned roles; governed to the most-held-out role. Details in
  `config/dataset-dispositions.json` and EXPERIMENT_LOG.md.
- Cursor offsets in the package are statement-relative (importer stores
  `doc_prefix` = current_statement_prefix; `recent_prefix` kept separately).
