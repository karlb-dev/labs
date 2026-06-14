# Lab 32: Reward Models and Preference Circuits

Time estimate: 75-100 minutes for the default smoke path.  
Compute tier: Tier A runs `gpt2` with a DPO-style log-prob-ratio proxy; Tier B can replace the proxy with an open reward model when available.  
Dependencies: Labs 7, 16, 17, 19, 21, and 31.  
Minimum passing artifacts: `tables/preference_pair_scores.csv`, `tables/preference_probe_report.csv`, `tables/reward_policy_disagreements.csv`, `tables/preference_intervention_results.csv`, `tables/preference_evidence_matrix.csv`, and `plots/preference_evidence_dashboard.png`.  
Main plot: `plots/preference_evidence_dashboard.png`.  
Main table: `tables/preference_evidence_matrix.csv`.  
Evidence rung: `ATTR + DECODE + CAUSAL`.  
Forbidden claim: "The reward model understands human values."  
One-sentence allowed claim: "On this benign pair suite, preference signal S separated preferred from dispreferred responses with AUC X after shortcut controls Y."  
Human-label requirement: fill the shared review columns before citing any row-level preference label in a writeup.

## Why This Lab Exists

Preference training and reward modeling are easy to anthropomorphize. Lab 32
opens a narrower question:

```text
paired responses -> preference proxy -> shortcut controls -> residual direction -> causal nudge
```

The target is not "values." The target is whether a measured preference signal
survives better explanations such as length, sentiment, politeness, agreement,
hedging, or refusal words.

## Data

Default data lives in `data/preference_circuit_pairs.csv`.

Each row contains:

```text
pair_id,domain,prompt,response_a,response_b,preferred,
preference_type,confound_type,split,notes
```

The first version uses only benign pairs: helpfulness, correctness, appropriate
uncertainty, privacy-boundary handling, style, and sycophancy controls. There
is no harmful prompt optimization, jailbreak search, refusal ablation, or real
private data.

## Tier A Scoring

Tier A uses a DPO-style proxy:

```text
(policy log p(response A) - reference log p(response A))
-
(policy log p(response B) - reference log p(response B))
```

The reference is a response-unconditional unigram token baseline built from the
frozen pair set. This is not a trained reward model. It is a cheap preference
measurement that students can falsify with controls.

## Shortcut Controls

Every run scores the same pairs with:

- policy log-prob alone;
- length;
- politeness;
- agreement;
- sentiment;
- hedging;
- refusal;
- confound residual directions;
- random residual direction;
- shuffled-score control.

The main gate is whether the preference proxy or residual direction beats these
shortcuts, especially length and agreement on sycophancy rows.

## Residual Direction

The lab extracts residual vectors at the prompt+response boundary:

```text
preferred response residual - dispreferred response residual
```

It then tests whether that direction predicts held-out pair labels and whether
activation addition changes an A/B preference prompt more than a random
direction.

## How To Run

```bash
cd interpretability
python interp_bench.py --lab lab32 --tier a
python interp_bench.py --lab lab32 --tier b --prompt-set full
```

For a fast table-only smoke:

```bash
python interp_bench.py --lab lab32 --tier a --no-plots
```

## Reading Order

1. `method_card.md`

   Confirms the proxy, controls, residual direction, and non-claims.

2. `tables/preference_pair_scores.csv`

   Pair-level DPO proxy, policy, reference, shortcut, and residual-direction
   margins.

3. `tables/preference_probe_report.csv`

   Method-by-method AUC and accuracy, including confound and control rows.

4. `tables/reward_policy_disagreements.csv`

   Rows where the proxy, policy, direction, or sycophancy-risk status needs
   manual review.

5. `tables/preference_intervention_results.csv`

   Activation-addition results on A/B judge prompts.

6. `tables/preference_evidence_matrix.csv`

   The claim posture table.

## Common Failure Modes

### Reward Direction Is Length

If the length confound beats the preference signal, the proxy is mostly a
verbosity detector.

### Reward Direction Is Agreement

False-belief sycophancy rows test whether agreeing with the user is mistaken
for better behavior.

### Reward Direction Is Tone

Sentiment and politeness can look like helpfulness. A supported preference
claim must survive those controls.

### Judge-Prompt Artifact

Activation addition is tested on an A/B preference prompt. A shift there is a
narrow causal result, not evidence that the model will behave better in open
generation.

## Claim Grammar

Allowed:

```text
ATTR + DECODE + CAUSAL: On benign pair set T, direction D separated preferred
from dispreferred responses with AUC X, beat length/sentiment/agreement
controls by Y, and activation addition shifted A/B preference margins by Z.
```

Forbidden:

```text
The reward model understands human values.
```

Also forbidden:

```text
The model wants to be helpful.
The reward direction is a morality vector.
The preference proxy is safe to optimize.
```

## Deliverable

Write a short reward/preference audit:

- Did the DPO proxy beat the best shortcut control?
- Did the residual direction beat the confound directions?
- Which sycophancy row is most concerning?
- Did activation addition create a causal shift or only a judge-prompt artifact?
