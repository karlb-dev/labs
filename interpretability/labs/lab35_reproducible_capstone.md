# Lab 35: Reproducible Interpretability Paper Capstone

```text
Time estimate: 20-40 minutes for the scaffold run; 2-4+ hours for source-run binding, adversarial review, repairs, and final package cleanup
Compute tier: Tier A validates the capstone package scaffold; the selected source lab determines the real science compute
Dependencies: the full course, especially Labs 11, 23, 27, 29, 31, 33, and the selected source lab
Minimum passing artifacts: preregistration.md, paper.md, claim_card.md, adversarial_review.md, review_response.md, reproduction_guide.md, tables/evidence_matrix.csv, tables/result_binding_template.csv, diagnostics/self_check_status.json, diagnostics/warning_summary.json, plot_manifest.json, plots/capstone_dashboard.png
Main plot: plots/capstone_dashboard.png
Main table: tables/evidence_matrix.csv
Evidence rung: AUDIT + FORMAL for the package; the source-lab rung is inherited only after frozen-run binding
Forbidden claim: any claim that exceeds the chosen source lab's evidence rung, hides failed controls, or replaces the original frozen run with a prettier repair run
One-sentence allowed claim: This package preregisters, freezes, reviews, repairs, reproduces, and bounds a scoped interpretability result with explicit controls, falsifiers, warnings, and source-run binding fields.
Human-label requirement: required when the selected seed track says so; final claim and adversarial-review judgments always need human signoff before publication
```

## Lab thesis

Lab 35 is the ship-it lab. It does not add a new mechanistic method. It asks whether one scoped result from the course can survive preregistration, frozen execution, adversarial review, repair accounting, reproduction instructions, claim-boundary discipline, and a package another student can inspect.

A clean plot is not a paper. A clean plot with a frozen run, visible controls, warning artifacts, review response, and falsifier-aware claim card is starting to become one.

The default run produces a scaffold. The scaffold is not the scientific result. The run should say:

```text
science_ready=false
```

until a real source-lab run is bound. That flag is the lab being honest.

## The question this lab asks

```text
Can a small interpretability result survive preregistration, frozen execution,
adversarial review, repair accounting, reproduction, and claim-ledger discipline?
```

The result is not the final plot. The result is the whole package contract:

```text
preregistration -> frozen run -> controls/falsifiers -> adversarial review -> repair log -> claim card -> reproduction guide -> release hash
```

Every important sentence must have an artifact, a control or falsifier, an evidence rung, and a reproduction path.

## Seed tracks

The frozen seed suite normally lives at:

```text
data/capstone_seed_tracks.jsonl
```

A Tier A fallback seed menu exists only for artifact plumbing if the JSONL is absent. Runs using fallback seed tracks are marked smoke-only and should not be published as course source data.

The selected track defaults to the first recommended track. You can select a track with the shared `--showcase` or `--mode` flag:

```bash
python interp_bench.py --lab lab35 --tier a --showcase audit_lab31_auto_interp_labels
python interp_bench.py --lab lab35 --tier a --mode method_replication
python interp_bench.py --lab lab35 --tier a --mode replicate_lab27_path_proxy
```

Typical track families include:

| Track family | Typical source lab | Good for |
|---|---:|---|
| residual path-mediation replication | Lab 27 | a scoped causal path-proxy package with strong controls |
| auto-interpretability label audit | Lab 31 | abstention, calibration, decoys, and human-review discipline |
| training-dynamics threshold ordering | Lab 29 | threshold-order language without exact-birth overclaiming |
| multimodal leak-gate extension | Lab 33 | shortcut and alignment discipline before real-VLM claims |
| tool-use surface-cue audit | Lab 34 | state tracking, tool traces, and self-report boundaries |
| preference-confound audit | Lab 32 | reward/preference claim discipline |

## Required phases

### 1. Preregistration

Write the research question, dataset, model, measurement sites, primary metric, controls, falsifiers, stopping rule, expected failure modes, allowed claim, forbidden claim, and safety scope before reading source-run results.

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

That table is intentionally blank at scaffold time. Fill it from the source run after freezing:

```text
frozen_run_dir
frozen_run_artifact_index
model_id
model_revision
data_path
data_sha256
seed
primary_metric_value
strongest_control_value
negative_result_specimen
repair_run_dir
release_package_hash
```

### 3. Adversarial review

Attack instrumentation, leakage, controls, statistics, safety, and claim language.

Artifacts:

```text
adversarial_review.md
tables/review_rubric.csv
tables/human_review_queue.csv
tables/review_action_items.csv
```

The review queue carries the shared human-label fields:

```text
student_label_primary
student_label_secondary
student_confidence
student_evidence_span
reviewer_label
agreement_status
```

Fill them for any claim, judgment, or generated explanation cited in the final paper.

### 4. Repair accounting

One repair run is allowed for a documented instrumentation bug or a missing preregistered control. The original frozen run remains visible.

Artifacts:

```text
tables/repair_log.csv
review_response.md
```

Bad repair accounting says:

```text
The repair run is cleaner, so we use it as the result.
```

Good repair accounting says:

```text
The original run failed because X. The repair run checks whether bug Y explains it. Both runs remain in the package.
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
plot_manifest.json
```

## How to run

From `interpretability/`:

```bash
# Fast scaffold validation, no figures.
python interp_bench.py --lab lab35 --tier a --no-plots

# Scaffold plus dashboards.
python interp_bench.py --lab lab35 --tier a

# Full seed-track menu. This still does not run source-lab science by itself.
python interp_bench.py --lab lab35 --tier b --prompt-set full

# Select a specific track.
python interp_bench.py --lab lab35 --tier a --mode replicate_lab27_path_proxy
```

With the recommended registry patch, Lab 35 uses `gpt2` on all tiers because the model is only used for shared bench self-checks. The scientific compute belongs to the selected source lab.

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
  plot_manifest.json

  diagnostics/
    data_manifest.json
    seed_track_schema_audit.csv
    run_config_snapshot.json
    warning_summary.csv
    warning_summary.json
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
    review_action_items.csv
    repair_log.csv
    failure_modes_contribution.csv
    failure_specimens.csv
    failure_specimens.jsonl
    reproduction_checklist.csv
    result_binding_template.csv
    preregistration_drift_audit.csv
    package_stage_status.csv
    source_claim_binding_matrix.csv
    artifact_dependency_edges.csv
    claim_to_artifact_map.csv
    claim_risk_register.csv
    package_validation.csv
    plot_reading_guide.csv
    plot_manifest.csv
    figure_sources/*.csv

  cards/
    failure_specimens.md

  plots/
    capstone_dashboard.png
    target_vs_control.png
    artifact_contract_status.png
    evidence_rung_matrix.png
    review_score_radar.png
    control_falsifier_map.png
    failure_mode_atlas.png
    reproduction_readiness_ladder.png
    claim_risk_register.png
    binding_gap_matrix.png
    paired_examples.png

  state/
    selected_track.json
    capstone_package_manifest.json
```

## How to read the run

Start with `diagnostics/warning_summary.csv`. It should include `source_run_unbound` and `science_ready_false` in a scaffold run. That is correct.

Then read:

1. `run_summary.md`: the package verdict and the main blocker.
2. `method_card.md`: selected track, source lab, evidence ceiling, and non-claim.
3. `preregistration.md`: the promise before results.
4. `tables/result_binding_template.csv`: the fields still missing from the frozen source run.
5. `tables/evidence_matrix.csv`: which claim components are scaffolded and which require source-run binding.
6. `tables/source_claim_binding_matrix.csv`: which claim components still need frozen source-run binding before science claims are allowed.
7. `tables/control_falsifier_matrix.csv`: the controls and falsifiers that can kill the favorite story.
8. `tables/failure_specimens.csv` and `cards/failure_specimens.md`: rows to fill with concrete negative evidence.
9. `adversarial_review.md` and `tables/human_review_queue.csv`: required human review.
10. `claim_card.md`: the only claim sentence allowed near the ledger.
11. `plot_manifest.json`: figure paths, source tables, row counts, metrics, comparisons, and claim boundaries.

Only after that should you open `paper.md`.

## How to read the figures

The plots are not decorations. They are a reading path through the package risks. Each plot has a source table under `tables/figure_sources/`, and `plot_manifest.json` records the source table, row count, metric, comparison, and claim supported.

| Plot | First question | Source table | Interpretation note |
|---|---|---|---|
| `capstone_dashboard.png` | Is the package complete enough to review? | `tables/figure_sources/capstone_dashboard_source.csv` | Green scaffold plus red source binding is the expected shape. |
| `target_vs_control.png` | Is the favorite claim shown beside controls and blockers? | `tables/figure_sources/target_vs_control_source.csv` | Counts are obligations, not successes. |
| `artifact_contract_status.png` | Which artifacts are generated versus pending? | `tables/figure_sources/artifact_contract_status_source.csv` | Pending source artifacts block scientific claims. |
| `evidence_rung_matrix.png` | Which claim pieces are FORMAL/AUDIT and which are inherited? | `tables/figure_sources/evidence_rung_matrix_source.csv` | Bright scaffold cells do not raise the source-lab evidence ceiling. |
| `review_score_radar.png` | Where should adversarial review focus first? | `tables/figure_sources/review_score_radar_source.csv` | Seed scores are triage values, not reviewer grades. |
| `control_falsifier_map.png` | What can kill or narrow the favorite claim? | `tables/figure_sources/control_falsifier_map_source.csv` | Control count is not control quality. |
| `failure_mode_atlas.png` | Which failures must remain visible if observed? | `tables/figure_sources/failure_mode_atlas_source.csv` | Failure rows need concrete source-run examples before they support science. |
| `reproduction_readiness_ladder.png` | Which reproduction fields are seeded versus still pending? | `tables/figure_sources/reproduction_readiness_ladder_source.csv` | A scaffolded command is not an immutable source run. |
| `claim_risk_register.png` | Which overclaim risks require review pressure? | `tables/figure_sources/claim_risk_register_source.csv` | High risk is a review target, not a result. |
| `binding_gap_matrix.png` | Which binding fields are still blank? | `tables/figure_sources/binding_gap_matrix_source.csv` | Blank source-run fields block source-lab claims. |
| `paired_examples.png` | What does allowed, forbidden, and negative-result language look like? | `tables/figure_sources/paired_examples_source.csv` | Use the narrowest sentence that survived review. |
| `reproduction_readiness_ladder.png` | Which reproducibility fields remain unbound? | `tables/figure_sources/reproduction_readiness_ladder_source.csv` | A scaffold cannot reproduce a source run by itself. |
| `binding_gap_matrix.png` | Which source-run fields block the final claim? | `tables/figure_sources/binding_gap_matrix_source.csv` | Blank fields are honest blockers. |
| `claim_risk_register.png` | Which prose and review risks can inflate the evidence rung? | `tables/figure_sources/claim_risk_register_source.csv` | Good wording cannot rescue weak or unbound evidence. |
| `paired_examples.png` | Which promise/evidence pairs should be inspected after aggregates? | `tables/figure_sources/paired_examples_source.csv` | Paired cards guide inspection; they are not evidence by themselves. |

## What each control is meant to falsify

| Control or audit row | What it attacks |
|---|---|
| Source-run binding template | The final paper cites numbers that are not tied to an immutable run. |
| Artifact checklist | Required files are missing but the package still looks polished. |
| Evidence matrix | A sentence claims a rung higher than its artifact supports. |
| Control/falsifier matrix | The favorite claim has no visible way to die. |
| Failure specimens | Negative rows are hidden because they make the story less clean. |
| Repair log | A repair run silently replaces the original frozen run. |
| Preregistration drift audit | The paper answers a nicer question than the one preregistered. |
| Human review queue | Reviewer judgment is implied without actual review fields. |
| Claim risk register | Forbidden language leaks into the abstract, conclusion, or ledger. |
| Warning summary | Smoke-only or unbound-state warnings are buried in diagnostics. |

## Tier A smoke behavior versus Tier B package behavior

Tier A should generate the complete package scaffold quickly. If the seed-track JSONL is absent, Tier A uses the built-in fallback seed tracks and writes a warning. That validates plumbing only.

Expected Tier A scaffold shape:

```text
package_ready_for_student_replacement=true
science_ready=false
source_run_bound=false
warnings include source_run_unbound
plot_manifest.json exists when plots are enabled
```

Tier B with `--prompt-set full` loads the full seed-track menu. It still does not run the selected source lab. The selected source lab must be run separately, frozen, and bound through `tables/result_binding_template.csv`.

## What counts as evidence

Lab 35 can support this after a scaffold run:

```text
AUDIT + FORMAL: The capstone package scaffold is schema-valid, reviewable,
warning-visible, and explicit about pending source-run binding.
```

It cannot support this until a source run is bound:

```text
The selected mechanistic claim is true.
```

After a source run is bound, the strongest allowed claim is still capped by the source lab's evidence rung.

## Honest negative results

A failed favorite hypothesis can still make an excellent capstone. The package should say:

```text
The preregistered claim did not survive because <control or failure mode>.
The supported contribution is <a narrower claim or audit finding>.
```

Bad negative-result handling says:

```text
The confusing control was removed, and the cleaner plot is presented.
```

That sentence is exactly the sort of repair drift the capstone is meant to catch.

## Common failure modes

| Failure mode | What it looks like | Where to catch it |
|---|---|---|
| preregistration drift | paper answers a nicer question than the preregistration | `tables/preregistration_drift_audit.csv` |
| frozen-run replacement | repair run silently replaces original run | `tables/repair_log.csv` |
| control evasion | failed control disappears from main paper | `tables/control_falsifier_matrix.csv` and `tables/failure_specimens.csv` |
| claim inflation | DECODE evidence becomes CAUSAL prose | `claim_card.md` and `tables/claim_risk_register.csv` |
| reproduction gap | no command, data hash, model revision, seed, or artifact index | `tables/reproduction_checklist.csv` |
| review as formality | review exists but no decision changed the claim | `adversarial_review.md` and `review_response.md` |
| human-label omission | labels or review judgments are cited without filled review fields | `tables/human_review_queue.csv` |
| plot provenance gap | figure has no source table or manifest entry | `plot_manifest.json` |

## Claim grammar

Allowed for the scaffold:

```text
AUDIT + FORMAL: This package generated a preregistration, evidence matrix,
control/falsifier map, review rubric, warning summary, repair log,
reproduction guide, plot manifest, and claim card for track T; source-run
evidence remains pending.
```

Allowed after a successful source run, with numbers filled:

```text
<source rung> + AUDIT + FORMAL: In frozen source run R, metric M was X above
strongest control C, with falsifiers F reported, failure specimens S visible,
and claim boundary B respected.
```

Forbidden:

```text
The repair run replaces the original run.
The source lab proved more than its evidence rung supports.
The final paper can hide negative controls because they confuse the story.
The review is optional.
A complete capstone package makes the source result true.
```

## Submission checklist

Before submitting, the package should answer yes to all of these:

```text
- The source run directory is immutable and named.
- Every number in the paper maps to an artifact row or figure source table.
- Failed controls remain visible.
- The repair run, if any, is logged beside the original run.
- Human-review fields are filled where required.
- The claim card stays below the selected source lab's evidence ceiling.
- The source data hash, model revision, seed, and artifact index are recorded.
- plot_manifest.json contains every figure and its source table.
- warning_summary.json does not hide source-run or fallback-data blockers.
- The final release zip hash is recorded.
```
