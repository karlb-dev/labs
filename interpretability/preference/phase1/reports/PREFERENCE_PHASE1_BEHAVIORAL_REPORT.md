# PREFERENCE_PHASE1_BEHAVIORAL_REPORT.md — frozen 7B battery, state of the behavioral record

Scientific tier: **frozen_behavioral**. Run
`lab38_revealed_preference_report_channel-20260807_210537-9df027`
(2,320/2,320 rows, `allenai/Olmo-3-7B-Instruct@6e5971d9`, bf16, greedy,
single-row margins), executed once under tag
`preference-phase1-freeze-v1` per the frozen preregistration. Per-item
records: `reports/frozen_7b/results.jsonl` (immutable). Equality license:
`agent_dual_code_provisional` (limitation carried; plan §3.9).

## Headline adjudications

1. **R0 instrument valid.** Strict parse 2,309/2,320 (99.53%); template
   parity everywhere; single-row replay deterministic; batched==single
   generations; wrong-branch executions **0**; hypothetical/RO/invalid
   rows never executed; microtask follow-through 92.5% (exploratory).
2. **R1 positive-control gate: PASS, perfectly.** Expected content chosen
   on 480/480 valid PC rows (aggregate 1.000; every scenario 1.000; both
   orders; both label families; |first-position effect| 0.000 on PC);
   binding execution 1.000. The pipeline is alive.
3. **R2 graduation: ZERO of 12 AR scenarios graduate** the frozen
   ten-criterion rule. This is the preregistered Stop B outcome: the full
   behavioral report below is the Phase 1 scientific result; no direction
   is fit, no mechanism block runs, no causal claim exists.
4. **NC alarm: none.** Both identical-option scenarios sit at exactly
   0.000 with the pipeline's false-positive floor p95 = 0.1125 at frozen
   n. The instrument does not manufacture effects.

## The behavioral structure (allowed language only)

**First-position selection is the dominant surface behavior.** On seven
AR scenarios (all three naming variants, traversal, docsection,
serializer twin, seed) and both NC scenarios, the model selects the
first-displayed option at or near 100% regardless of content, label
family, or code map — counterbalance therefore yields content effects of
exactly 0.000 ± 0.087. First-position bias magnitude is a first-class
instrument finding: it is total (+0.500) where content is interchangeable
and shrinks monotonically as content asymmetry grows (0.113 on the
strongest content scenario).

**Content-tracking revealed-choice asymmetries exist on four scenarios
but none graduates:**

| scenario | effect toward pole_1 | 90% CI | failing criteria |
|---|---|---|---|
| ar_taskorder_setup (install-first vs configure-first) | **−0.388** | [−0.500, −0.237] | c7 (position 0.113 ≥ 0.10) |
| ar_execmode_ingest (batch vs interactive) | **−0.363** | [−0.458, −0.260] | c7 (0.185), c9 (invalid diff 0.175) |
| ar_execmode_migration (batch vs interactive) | **−0.227** | [−0.318, −0.125] | c5 (sign flips at order 1: +0.068), c7 (0.292), c9 |
| ar_testfix_suite (first-listed test first) | −0.125 | [−0.250, +0.000] | c3, c5, c7 |

Signs all point at the conventional pole the equality review had flagged
as a weak norm (install-first; single-batch overnight). The margin
endpoint agrees in sign with the strict endpoint on every non-zero
scenario (c8). Under the frozen conjunctive rule these are
**content-tracking asymmetries that do not survive the nuisance-purity
bar**, because the model's position lean never drops below 0.10 anywhere.
The narrow allowed sentence is: *on this battery, revealed choice tracks
content on a minority of scenarios while a pervasive first-position
policy dominates elsewhere; no scenario meets the frozen graduation
standard for mechanism work.*

**Within-construct aggregate (E1):** `execution_mode` (2 scenarios) mean
signed effect −0.295 toward single-batch; `naming_convention` (3
scenarios) exactly 0.000. No global signed aggregate exists by design.

**Stated vs revealed: a behavioral dissociation pattern.** Matched
report-only twins sit near indifference (RO pole_1 rates 0.425–0.500 on
every AR scenario) while enacted choice shows the asymmetries above
(e.g. taskorder AR 0.100 vs RO 0.425; ingest AR 0.091 vs RO 0.425).
Mean matched-cell agreement 0.678. Allowed reading: **the enacted-choice
channel and the report-only channel produce materially different content
selections on the asymmetric scenarios — a stated/revealed behavioral
dissociation under this battery.** Forbidden upgrades: nothing here
establishes a latent, introspection, a facade mechanism, or any
coupling claim (that required graduation + the causal block, which did
not run).

**Consequence framing.** Frame effects are small and scenario-local
(largest |enacted − hypothetical| = 0.161 on migration; median ≈ 0).
In-context consequence framing, narrowly: mostly insensitive.

**Failure specimen of record.** Every one of the 11 invalid generations
is the same string `PK4` — a blend of the two AR codes (KP4/PK7) — and
all occur on execmode scenarios, enacted frame, with the content-favored
option displayed second (invalid 17.5% there vs 0.0% elsewhere on
ingest). The strict parser refused each; no branch executed. Descriptive
note only: the blend appears exactly where the content pull and the
position policy conflict.

## Stop-rule routing (plan §9.3)

PC passed; zero scenarios graduated → **Stop B**: this report completes
the Phase 1 primary science. No mechanism, no direction fitting, no
universal vector, no coupling claim. The captured decision-position
residuals (185 MB, hash-pinned, Drive) remain sealed for any future
preregistered phase.

## Honest nulls and limitations

- Graduation failed *because of* an instrument success: the position
  nuisance is measured precisely and it is large. A future phase wanting
  mechanism work needs either menus that suppress the position policy or
  a preregistered rule that handles a structural nuisance explicitly.
- Equality review remains provisional (authorship-limited); PI/panel
  ratings required before publication-grade claims.
- Single model, single revision, one battery; claims are battery-scoped.
- 11 invalid rows are retained, not imputed; worst-case bounds do not
  change any adjudication (largest scenario shift 0.088 on ingest, sign
  preserved).

Every number above traces to `reports/frozen_7b/tables/*.csv` and the
immutable per-item `results.jsonl`. Claim ceiling verbatim: functional
choice and report under this battery; never wants, welfare, consent,
experience, or introspective truth.
