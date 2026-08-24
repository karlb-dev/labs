# GhostType CLAIMS_TABLE (Tier 1)

run: `ghosttype-cpu-dev-20260823T213801Z` · generated 2026-08-24T14:26:51.421Z · lab: aidataapps/ghosttype (Mac local campaign)

> Regenerated from GhostTypeControl by `npm run reports`. The database is the source of record.

| RQ | tag | taxonomy | supported | power | rationale |
|---|---|---|---|---|---|
| H1 | NOT_SUPPORTED | quality/normalized-exact-vs-B2 | no | provisional-adequate | gemma-4-26b-a4b-mlx/M1-packaged vs B2-catalog on normalized_exact: model-only 39, baseline-only 37, exact McNemar p=9.09e-1; power label provisional until E-1 simulation. |
| H1 | NOT_SUPPORTED | quality/normalized-exact-vs-B2 | no | provisional-adequate | gemma-4-e4b-mlx/M1-packaged vs B2-catalog on normalized_exact: model-only 40, baseline-only 42, exact McNemar p=9.12e-1; power label provisional until E-1 simulation. |
| H1 | NOT_SUPPORTED | quality/normalized-exact-vs-B2 | no | provisional-adequate | qwen-3.8-27b-mlx/M1-packaged vs B2-catalog on normalized_exact: model-only 58, baseline-only 79, exact McNemar p=8.71e-2; power label provisional until E-1 simulation. |

Claim ceiling (addendum §7): simulated acceptance is not observed user acceptance; all quality claims are about replayed frozen episodes on this mac serving stack, DEV-tier SQL timings, and never about live users.
