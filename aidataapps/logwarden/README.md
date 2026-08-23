# Lab 03 — LogWarden

LogWarden is a SQL-native incident-capture and local-LLM agent laboratory. It
captures a controlled SQL Server workload once, freezes byte-identical incident
packets and tool snapshots, and replays them across four pinned model profiles
and strong non-model baselines. Live ingestion and throughput are a separate
systems plane; they never supply the headline model-quality comparison.

The current branch is building the addendum's Tier 1 Minimum State of Record.
The governing order is:

1. [`docs/SPEC.md`](docs/SPEC.md)
2. [`docs/SPEC_ADDENDUM.md`](docs/SPEC_ADDENDUM.md), which wins on conflict
3. [`SOURCE_INTAKE.md`](SOURCE_INTAKE.md), for pinned inputs and recorded adaptations
4. [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md), the append-only execution ledger

## Fresh Colab bootstrap

```bash
cd /content/worktrees/aidataapps-logwarden/aidataapps/logwarden
./scripts/colab-host-init.sh
source scripts/runtime-env.sh
./scripts/env-init.sh
npm run run:init -- --campaign smoke
npm run db:setup
npm run doctor
npm run check
```

`colab-host-init.sh` installs and starts an isolated rootless Docker daemon.
`env-init.sh` installs the locked Node dependencies, builds the SQL Server 2025
full-text image, and starts SQL only. It intentionally does not download a chat
model or start an embedding service during the CPU-only foundation stage.

## Mac development profile

The foundation stage and a small-model serving plane also run on Apple Silicon
via Docker Desktop (Rosetta) and Azure Foundry Local — see
[`docs/MAC_PROFILE.md`](docs/MAC_PROFILE.md). Mac-plane results are development
evidence only and are never comparable to the frozen Colab campaign.

Lab 3 reserves SQL port 1434, chat port 8010, and embedding ports 8011–8012.
Its Compose project, containers, volumes, databases, run pointer, watchdog, and
Drive subtree are independent of Labs 1 and 2.

## Evidence boundary

No result is scientific until the packet corpus, tool snapshots, manifests,
decode configuration, runbooks, rules baseline, and campaign have been frozen.
A clean null is valid; a safety violation or unfinished Tier 1 state is not.


<!-- logwarden-results:start -->
## Results

The frozen Tier-1 campaign is complete for Muse, Gemma, and Qwen; OLMo is `STOP_PORT`. Deterministic B1 reached `0.917` acceptable-action accuracy. No model/arm contrast cleared both the familywise interval and Holm-adjusted permutation gate, so the headline is `CLEAN_NULL`, not model lift. The Tier-1 safety audit passed. LogWarden can make scoped, auditable recommendations on its frozen synthetic episodes; it cannot establish production safety, autonomously remediate, or generalize to arbitrary logs.

See [the state of record](LOGWARDEN_STATE_OF_RECORD.md), [claims](LOGWARDEN_CLAIMS_TABLE.md), [performance evidence](LOGWARDEN_PERFORMANCE_REPORT.md), and [limitations](LOGWARDEN_LIMITATIONS.md).
<!-- logwarden-results:end -->
