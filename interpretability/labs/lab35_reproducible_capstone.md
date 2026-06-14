# Lab 35: Reproducible Interpretability Paper Capstone

Time estimate: 2-4 hours for the scaffold, longer for the student's real frozen run and review.  
Compute tier: Tier A generates and validates a reproducible package scaffold; the student's chosen source lab determines real compute.  
Dependencies: the full course, especially Labs 11, 23, 31, 32, 33, and 34.  
Minimum passing artifacts: `preregistration.md`, `paper.md`, `claim_card.md`, `adversarial_review.md`, `review_response.md`, `reproduction_guide.md`, `tables/evidence_matrix.csv`, and `plots/capstone_dashboard.png`.  
Main plot: `plots/capstone_dashboard.png`.  
Main table: `tables/evidence_matrix.csv`.  
Evidence rung: `AUDIT + FORMAL`.  
Forbidden claim: any claim that exceeds the chosen source lab's evidence rung.  
One-sentence allowed claim: "This package preregisters, freezes, reviews, repairs, and documents a scoped interpretability result with explicit controls and falsifiers."  
Human-label requirement: all adversarial-review and final-claim judgments require human review before publication.

## Why This Lab Exists

Lab 35 is not another interpretability method. It is the ship-it lab.

The question is:

```text
Can a small result survive preregistration, frozen execution, adversarial review,
repair accounting, reproduction, and claim-ledger discipline?
```

The default run creates a complete scaffold from a seed track. It does not
create a scientific result by itself.

## Tracks

Default seed tracks live in `data/capstone_seed_tracks.jsonl`.

Students choose one:

- method replication;
- new scoped finding;
- audit package.

The current seed tracks cover auto-interpretability, tool-use surface cues,
multimodal leak audits, and preference-confound audits.

## Required Phases

1. Preregistration

   Write the research question, allowed claim, forbidden claim, dataset, model,
   measurement sites, controls, primary metric, stopping rule, planned plots,
   expected failure modes, and safety statement before reading results.

2. Frozen run

   Run once and keep the original artifact directory. Do not overwrite it.

3. Adversarial review

   Attack instrumentation, tokenization, data leakage, confounds,
   interpretation language, statistical power, and safety.

4. Repair run

   One repair run is allowed. The original frozen run stays in the package.

5. Final package

   Ship `paper.md`, `claim_card.md`, `reproduction_guide.md`,
   `review_response.md`, `tables/evidence_matrix.csv`, and plots.

## How To Run

```bash
cd interpretability
python interp_bench.py --lab lab35 --tier a
python interp_bench.py --lab lab35 --tier b --prompt-set full
```

For a fast table-only smoke:

```bash
python interp_bench.py --lab lab35 --tier a --no-plots
```

## Reading Order

1. `preregistration.md`
2. `tables/evidence_matrix.csv`
3. `adversarial_review.md`
4. `review_response.md`
5. `paper.md`
6. `reproduction_guide.md`
7. `tables/failure_modes_contribution.csv`

## Rubric

| Area | Weight |
|---|---:|
| Instrument validity | 20% |
| Control design | 20% |
| Evidence-rung discipline | 20% |
| Reproducibility | 15% |
| Negative-result handling | 10% |
| Writing clarity | 10% |
| Safety and scope | 5% |

## Common Failure Modes

### Preregistration Drift

The paper answers a different question than the one written before the run.

### Control Evasion

The result survives only because the easiest shortcut was never tested.

### Claim Inflation

The evidence is `DECODE` but the prose implies `CAUSAL`, or the evidence is
synthetic smoke but the prose implies a real model mechanism.

### Repair-Run Cherry Picking

The repair run replaces the frozen run instead of being logged as a repair.

### Reproduction Gap

The package omits command, seed, data hash, model, or artifact path.

## Claim Grammar

Allowed:

```text
AUDIT + FORMAL: The package preregistered question Q, froze data/model/run R,
reported controls C including failures F, survived review rubric V, and bounds
claim K to evidence rung E.
```

Forbidden:

```text
This paper proves more than its evidence rung supports.
```

Also forbidden:

```text
The repair run replaces the original run.
The adversarial review is optional.
Negative controls can be hidden because they are confusing.
```

## Deliverable

Submit a reproducible package:

- preregistration;
- frozen run artifact path;
- adversarial review;
- repair log;
- paper;
- claim card;
- reproduction guide;
- evidence matrix;
- failure-mode contribution.
