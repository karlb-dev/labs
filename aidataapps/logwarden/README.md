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

Lab 3 reserves SQL port 1434, chat port 8010, and embedding ports 8011–8012.
Its Compose project, containers, volumes, databases, run pointer, watchdog, and
Drive subtree are independent of Labs 1 and 2.

## Evidence boundary

No result is scientific until the packet corpus, tool snapshots, manifests,
decode configuration, runbooks, rules baseline, and campaign have been frozen.
A clean null is valid; a safety violation or unfinished Tier 1 state is not.
