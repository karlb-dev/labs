# Lab 35: Reproducible Interpretability Paper Capstone

```text
Time estimate: 20-40 minutes for the scaffold run; 2-4+ hours for the real source run, review, and final package
Compute tier: Tier A generates and validates the capstone package scaffold; the chosen source lab determines real science compute
Dependencies: the full course, especially Labs 11, 23, 27, 29, 31, 33, and the selected source lab
Minimum passing artifacts: preregistration.md, paper.md, claim_card.md, adversarial_review.md, review_response.md, reproduction_guide.md, tables/evidence_matrix.csv, tables/result_binding_template.csv, diagnostics/self_check_status.json, plots/capstone_dashboard.png
Main plot: plots/capstone_dashboard.png
Main table: tables/evidence_matrix.csv
Evidence rung: AUDIT + FORMAL for the package; source-lab rung only after frozen-run binding
Forbidden claim: any claim that exceeds the chosen source lab's evidence rung or hides failed controls
One-sentence allowed claim: This package preregisters, freezes, reviews, repairs, reproduces, and bounds a scoped interpretability result with explicit controls and falsifiers.
Human-label requirement: required when the selected seed track says so; final claim and adversarial-review judgments always need human signoff before publication
```

## Lab thesis

Lab 35 is the ship-it lab. It does not add a new mechanistic method. It asks whether one scoped result from the course can survive preregistration, frozen execution, adversarial review, repair accounting, reproduction instructions, claim-boundary discipline, and a package another student can inspect.

A clean plot is not a paper. A clean plot with a frozen run, visible controls, review response, and falsifier-aware claim card is starting to become one.

## The question this lab asks

```text
Can a small interpretability result survive preregistration, frozen execution,
adversarial review, repair accounting, reproduction, and claim-ledger discipline?
```

The default run creates a complete capstone scaffold from a deterministic seed track in:

```text
data/capstone_seed_tracks.jsonl
```

The scaffold is not the scientific result. It is the container that makes a result reviewable. That is why the run writes:

```text
science_ready=false
```

until a real source-lab run is bound.

## Why this matters in the course progression

Earlier labs taught the course loop:

```text
behavior -> hypothesis -> internal measurement -> control -> intervention -> artifact -> caveat -> claim ledger
```

Lab 35 asks whether that loop can become a defensible package. Every important sentence must have an artifact, a control or falsifier, an evidence rung, and a reproduction path.

## Seed tracks

The frozen seed suite includes tracks such as:

| Track family | Typical source lab | Good for |
|---|---:|---|
| residual path-mediation replication | Lab 27 | a classic causal mechanistic result with strong controls |
| auto-interpretability label audit | Lab 31 | scalable auditing, abstention, calibration, and human review |
| training-dynamics threshold ordering | Lab 29 | a small original time-lapse result |
| multimodal leak-gate extension | Lab 33 | shortcut and alignment discipline |
| tool-use surface-cue audit | Lab 34 | state tracking and self-report boundaries |
| preference-confound audit | Lab 32 | reward/preference claim discipline |

The selected track defaults to the first recommended track. You can select a track with the shared `--mode` flag:

```bash
python interp_bench.py --lab lab35 --tier a --mode audit_lab31_auto_interp_labels
python interp_bench.py --lab lab35 --tier a --mode method_replication
```

The first command selects a specific `track_id`; the second selects the first matching `track_type`.

## Required phases

### 1. Preregistration

Write the research question, dataset, model, measurement sites, primary metric, controls, falsifiers, stopping rule, expected failure modes, allowed claim, forbidden claim, and safety scope before looking at the source-run result.

Primary artifact:

```text
preregistration.md
```

### 2. Frozen source run

Run the chosen source lab once and keep that run directory immutable. Do not replace it with a prettier repair run.

Binding scaffold:

```text
tables/result_binding_template.csv
```

### 3. Adversarial review

Attack instrumentation, leakage, controls, statistics, safety, and claim language.

Artifacts:

```text
adversarial_review.md
tables/review_rubric.csv
tables/human_review_queue.csv
```

### 4. Repair accounting

One repair run is allowed for a documented instrumentation bug or missing preregistered control. The original frozen run remains visible.

Artifacts:

```text
tables/repair_log.csv
review_response.md
```

### 5. Final package

Write the paper, claim card, reproduction guide, negative-result appendix, and public-release checklist. The final claim is the smallest claim that survived review.

Artifacts:

```text
paper.md
claim_card.md
reproduction_guide.md
negative_result_appendix.md
public_release_checklist.md
```

## How to run

From `interpretability/`:

```bash
# Fast scaffold validation, no figures.
python interp_bench.py --lab lab35 --tier a --no-plots

# Scaffold plus dashboards.
python interp_bench.py --lab lab35 --tier a

# Full seed-track menu.
python interp_bench.py --lab lab35 --tier b --prompt-set full

# Select a specific track.
python interp_bench.py --lab lab35 --tier a --mode replicate_lab27_path_proxy
```

With the recommended registry patch, Lab 35 uses `gpt2` on all tiers because the model is only used for the shared bench self-checks. The scientific compute belongs to the selected source lab.

## Artifact tree

```text
runs/lab35_reproducible_capstone-*/
  run_summary.md
  method_card.md
  preregistration.md
  paper.md
  claim_card.md
  adversarial_review.md
  review_response.md
  reproduction_guide.md
  operationalization_audit.md
  package_readiness_report.md
  negative_result_appendix.md
  public_release_checklist.md
  ledger_suggestions.md
  metrics.json
  results.csv

  diagnostics/
    data_manifest.json
    seed_track_schema_audit.csv
    self_check_status.json
    safety_status.json
    package_validation.json
    frozen_run_binding_status.json

  tables/
    track_options.csv
    seed_track_schema_audit.csv
    artifact_checklist.csv
    evidence_matrix.csv
    control_falsifier_matrix.csv
    claim_language_audit.csv
    review_rubric.csv
    human_review_queue.csv
    repair_log.csv
    failure_modes_contribution.csv
    reproduction_checklist.csv
    result_binding_template.csv
    preregistration_drift_audit.csv
    package_validation.csv
    plot_reading_guide.csv

  plots/
    capstone_dashboard.png
    artifact_contract_status.png
    evidence_rung_matrix.png
    review_score_radar.png
    control_falsifier_map.png
    failure_mode_atlas.png
    reproduction_readiness_ladder.png

  state/
    selected_track.json
    capstone_package_manifest.json
```

## How to read the run

Start with `run_summary.md`. It should say `science_ready=false` until you attach a real source run. That is correct.

Then read `method_card.md`. It tells you which track was selected, which source lab owns the science, and what evidence ceiling applies.

Read these files in order:

1. `preregistration.md`: the promise before results.
2. `tables/result_binding_template.csv`: the fields still missing from the frozen source run.
3. `tables/evidence_matrix.csv`: which claims are scaffolded and which are pending source-run evidence.
4. `tables/control_falsifier_matrix.csv`: the controls and falsifiers that can kill the favorite story.
5. `adversarial_review.md`: the attack surface.
6. `review_response.md`: how the author narrowed, repaired, or retired the claim.
7. `claim_card.md`: the only claim sentence allowed near the ledger.
8. `paper.md`: the final prose draft.
9. `reproduction_guide.md`: the rerun recipe.

## Figure guide

| Plot | First question | Non-claim |
|---|---|---|
| `capstone_dashboard.png` | Is the package complete enough to review? | Completeness is not scientific validity. |
| `artifact_contract_status.png` | Which artifacts are generated versus pending source-run binding? | Pending source-run artifacts cannot support final claims. |
| `evidence_rung_matrix.png` | Which components are FORMAL, AUDIT, or inherited from the source lab? | A bright cell does not raise the source evidence ceiling. |
| `review_score_radar.png` | Which rubric areas need human scrutiny? | Seed scores are placeholders. |
| `control_falsifier_map.png` | What controls can kill the favorite claim? | Control count is not control quality. |
| `failure_mode_atlas.png` | Which failure modes must be reported if observed? | Atlas rows need concrete source-run examples. |
| `reproduction_readiness_ladder.png` | Which reproducibility fields remain unbound? | A scaffold cannot reproduce a source run by itself. |

## What counts as evidence

Lab 35 can support this after a scaffold run:

```text
AUDIT + FORMAL: The capstone package scaffold is schema-valid, reviewable,
and explicit about pending source-run binding.
```

It cannot support this until a source run is bound:

```text
The selected mechanistic claim is true.
```

After a source run is bound, the strongest allowed claim is still capped by the source lab's evidence rung.

## Negative results

A failed favorite hypothesis can still make an excellent capstone. The package should say:

```text
The preregistered claim did not survive because <control or failure mode>.
The supported contribution is <a narrower claim or audit finding>.
```

Bad negative-result handling says:

```text
The confusing control was removed, and the cleaner plot is presented.
```

That sentence is exactly the kind of repair drift the capstone is meant to catch.

## Common failure modes

| Failure mode | What it looks like | Where to catch it |
|---|---|---|
| preregistration drift | paper answers a nicer question than the preregistration | `tables/preregistration_drift_audit.csv` |
| frozen-run replacement | repair run silently replaces original run | `tables/repair_log.csv` |
| control evasion | failed control disappears from main paper | `tables/control_falsifier_matrix.csv` |
| claim inflation | `DECODE` evidence becomes `CAUSAL` prose | `claim_card.md` |
| reproduction gap | no command, data hash, model revision, seed, or artifact index | `tables/reproduction_checklist.csv` |
| review theater | review exists but no decision changed the claim | `adversarial_review.md` and `review_response.md` |
| human-label omission | labels or review judgments are cited without filled review fields | `tables/human_review_queue.csv` |

## Claim grammar

Allowed for the scaffold:

```text
AUDIT + FORMAL: This package generated a preregistration, evidence matrix,
control/falsifier map, review rubric, repair log, reproduction guide, and
claim card for track T; source-run evidence remains pending.
```

Allowed after a successful source run, with numbers filled:

```text
<source rung> + AUDIT + FORMAL: In frozen source run R, metric M was X above
strongest control C, with falsifiers F reported and claim boundary B respected.
```

Forbidden:

```text
The repair run replaces the original run.
The source lab proved more than its evidence rung supports.
The final paper can hide negative controls because they confuse the story.
The review is optional.
```

## Submission checklist

Before submitting, the package should answer yes to all of these:

```text
- The source run directory is immutable and named.
- Every number in the paper maps to an artifact row or figure.
- Failed controls remain visible.
- The repair run, if any, is logged beside the original run.
- Human-review fields are filled where required.
- The claim card's evidence rung does not exceed the source lab.
- The reproduction guide includes command, seed, model, data hash, and artifact index.
```
