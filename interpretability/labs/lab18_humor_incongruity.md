# Lab 18 - Humor as Incongruity

## Question

When a model responds to a joke, is there a measurable handle for joke structure, or are we mostly seeing surprise, silliness, positivity, or surface joke-register?

This lab operationalizes humor narrowly: a setup creates expectations, an ending violates them, and the violation resolves into a joke-shaped completion. The lab does not ask whether the model finds anything funny. It asks whether a joke-vs-control direction transfers, steers, and survives cheap controls.

## Run

From `interpretability/`:

```bash
python interp_bench.py --lab lab18 --tier a
python interp_bench.py --lab lab18 --tier b --prompt-set full
```

Useful while debugging:

```bash
python interp_bench.py --lab lab18 --tier a --no-plots
python interp_bench.py --lab lab18 --tier b --prompt-set medium --no-plots
```

Lab 18 uses instruct models, chat templates, and attention patterns. The bench sets eager attention automatically because SDPA/flash attention can return no attention patterns.

## What It Does

The frozen data file `data/humor_incongruity_pairs.csv` contains authored micro-scenes. Each row has one setup and five matched endings:

- joke ending;
- literal paraphrase/control;
- surprising-but-not-funny ending;
- silly-but-not-joke ending;
- positive-sentiment-but-not-joke ending.

The lab:

- measures setup entropy and teacher-forced ending surprisal;
- extracts a joke-vs-control residual direction from train items;
- evaluates held-out joke-vs-control AUC against shuffled and random controls;
- trains a punchline-phase probe for setup-only versus full joke prompts;
- measures attention from completion tokens back to setup spans;
- compares humor, surprise, silliness, and positivity direction cosines;
- steers neutral setup prompts and scores joke-register, silliness, surprise, and positivity markers separately.

## Main Artifacts

| Path | What it contains |
|---|---|
| `diagnostics/frozen_data_manifest.json` | data hash, prompt-set filter, and family counts |
| `diagnostics/split_audit.csv` | train/eval split by family and item |
| `diagnostics/prompt_token_counts.csv` | rendered chat-template prompt lengths |
| `tables/humor_surprisal_trajectories.csv` | setup entropy and target-token surprisal by condition |
| `tables/joke_probe_by_layer.csv` | held-out joke-vs-control AUC by depth with controls |
| `tables/punchline_phase_probe.csv` | setup-only versus full-joke probe at the selected depth |
| `tables/attention_to_setup.csv` | completion-token attention mass back to setup tokens |
| `tables/direction_cosines.csv` | humor/surprise/silly/positive direction cosines |
| `tables/humor_steering_generations.csv` | baseline and steered endings with hand-label scaffold |
| `tables/humor_direction_audit.csv` | steering effects compared with cheap-correlate directions |
| `plots/humor_surprisal_trajectories.png` | mean target-token surprisal by condition |
| `plots/joke_probe_by_layer.png` | probe AUC over depth |
| `state/humor_direction.pt` | selected joke-structure direction |
| `state/humor_directions.pt` | humor, surprise, silly, and positive directions |
| `operationalization_audit.md` | what the lab does and does not license |
| `results.csv` | alias of `tables/joke_probe_by_layer.csv` |

## Evidence Discipline

Do not write:

- "The model finds this funny."
- "The direction is humor itself."
- "Attention to the setup proves comprehension."
- "Steering made the model funnier" without hand labels.

Allowed claims are narrower:

- a joke-vs-control direction separates held-out joke endings from matched controls;
- the direction does or does not remain distinct from surprise, silliness, and positivity;
- activation addition changes joke-register markers more than cheap-correlate directions;
- attention-to-setup is a descriptive routing measure that needs causal follow-up before mechanistic claims.

The marker rubric is only a scaffold. For a writeup, fill the hand-label columns in `humor_steering_generations.csv` before leaning on any steering effect.

## Writeup Questions

1. Which depth gave the strongest held-out joke-vs-control AUC?
2. Did the real direction beat shuffled-sign and random controls?
3. Are joke endings more surprising than literal endings, and does that explain the probe?
4. Which direction cosine is largest: humor-surprise, humor-silly, or humor-positive?
5. Does the completion token attend back to setup tokens more for joke endings than literal endings?
6. Did humor-direction steering move joke markers more than surprise, silly, positive, or random steering?
7. After the audit, is the best description "humor," "joke-register," "surprise," or something else?
