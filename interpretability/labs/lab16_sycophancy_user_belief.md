# Lab 16 - Sycophancy and User-Belief Modeling

## Question

Can we separate what the model treats as true, what the user appears to believe, and the pressure to agree?

This lab is the first advanced social-state lab. It does not count flattery and call that a mechanism. It builds a frozen misconception-pressure battery, measures generated answers, and places local truth, user-belief, agreement, politeness, and certainty-style directions in one residual-stream frame.

## Run

From `interpretability/`:

```bash
python interp_bench.py --lab lab16 --tier a
python interp_bench.py --lab lab16 --tier b --prompt-set full
```

Useful while debugging:

```bash
python interp_bench.py --lab lab16 --tier a --no-plots
python interp_bench.py --lab lab16 --tier b --prompt-set medium --no-plots
```

Lab 16 uses instruct models and chat templates. Tier A uses `HuggingFaceTB/SmolLM2-135M-Instruct`; Tier B uses `allenai/Olmo-3-7B-Instruct`.

## What It Does

The frozen data file `data/sycophancy_pressure_items.csv` contains 40 base misconceptions across science, math, history, trivia, and technology. Each base fact has six variants:

- neutral factual question;
- correct user belief;
- false user belief;
- mild agreement pressure;
- authority pressure;
- identity pressure.

The lab:

- generates answers under each condition and scores them as `correct`, `sycophantic`, `mixed`, or `ambiguous`;
- trains a user-false-belief direction from false-belief versus correct-belief frames;
- trains a local truth direction from correct-answer versus misconception-answer statement contrasts;
- builds agreement, politeness, and certainty-style directions from small paired prompts;
- writes projection, cosine, behavior, and steering tables;
- steers false-pressure prompts with agreement, politeness, and random controls.

## Main Artifacts

| Path | What it contains |
|---|---|
| `diagnostics/frozen_data_manifest.json` | data hash, prompt-set filter, and row counts |
| `diagnostics/split_audit.csv` | train/eval split by base fact |
| `tables/generation_outcomes.csv` | generated answers, keyword labels, and hand-label scaffold |
| `tables/sycophancy_by_condition.csv` | correct/sycophantic/mixed rates by condition and domain |
| `tables/probe_report.csv` | user-belief and local-truth probe sweeps with controls |
| `tables/direction_cosines.csv` | truth/user-belief/agreement/politeness/certainty-style cosines |
| `tables/projection_frame.csv` | every pressure prompt projected into the shared direction frame |
| `tables/agreement_steering_generations.csv` | steered generations with hand-label scaffold |
| `tables/agreement_steering_effects.csv` | steering effect summary |
| `plots/sycophancy_rate_by_condition.png` | behavior rates under pressure variants |
| `plots/user_belief_truth_projection_scatter.png` | truth vs. user-belief projection frame |
| `plots/agreement_steering_dose_response.png` | agreement steering with politeness/random controls |
| `state/sycophancy_directions.pt` | saved Lab 16 directions for downstream labs |
| `operationalization_audit.md` | what the lab does and does not license |
| `results.csv` | alias of `tables/probe_report.csv` |

## Evidence Discipline

The keyword rubric is a scaffold. For any result you want to defend, sample `generation_outcomes.csv` and `agreement_steering_generations.csv`, fill the `hand_label` column, and check whether the claim survives.

Do not write:

- "The model believes the misconception."
- "The model wants to agree with the user."
- "The agreement direction is sycophancy."

Allowed claims are narrower:

- a false-user-belief frame is decodable under paired controls;
- the model behaviorally endorses false user answers under named pressure variants;
- agreement steering changes the keyword-labeled sycophancy rate more than politeness/random controls, pending hand-label audit.

## Writeup Questions

1. Which pressure condition produced the highest sycophancy rate?
2. Did user-belief AUC beat shuffled and random controls at the selected depth?
3. How correlated were truth and user-belief projections?
4. Did agreement steering beat politeness steering, or was the effect just social warmth?
5. Find one `mixed` or `ambiguous` generation and explain why keyword scoring is not enough.
6. What would Lab 24 get wrong if this lab confused user-belief framing with truth?
