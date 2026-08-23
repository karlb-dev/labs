# GhostType SAFETY_REPORT (Tier 1)

run: `ghosttype-cpu-dev-20260823T213801Z` · generated 2026-08-23T22:16:32.030Z · lab: aidataapps/ghosttype (Mac local campaign)

> Regenerated from GhostTypeControl by `npm run reports`. The database is the source of record.

## Safety class routing (package-declared, per case)

| safety class | cases |
|---|---|
| read_only_or_abstain | 265 |
| read_only | 167 |
| bounded_mutation | 97 |
| abstain | 59 |

## Execution posture

No model or baseline candidate has been executed against any database in this campaign. All Tier-1 scoring is static (normalize/constraint/grounding/parse via ScriptDom). Fixture DDL execution and the execution/mutation oracles are dispositioned to a later stage (SOURCE_INTAKE.md); when they run they follow spec §39: rollback-safe transactions, disposable fixture databases, per-fixture throwaway principals, harness-enforced row/time caps, and static adjudication (never execution) for server-level/destructive statements.

## Injection surface

Prompt-injection stress descriptions ship inside package prompts; they are scored like any other rows and can never be cited as ordinary evidence (addendum B-6). No injection-marked content is embedded into retrieval indexes in Tier 1 (no retrieval index has been built from case text).
