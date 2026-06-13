# Lab 17 - Persona, Voice, Roleplay, and Register

## Question

When a model role-plays a character, switches register, or develops a recognizable voice, is there a persistent internal handle you can measure, or mostly surface style riding on the chat template?

This lab treats persona, voice, register, and agreement as paired operationalizations. It does not ask whether the model has a true identity. It asks whether a direction extracted from controlled prompts transfers, steers behavior, and leaves a trace over scripted turns after cheap controls are checked.

## Run

From `interpretability/`:

```bash
python interp_bench.py --lab lab17 --tier a
python interp_bench.py --lab lab17 --tier b --prompt-set full
```

Useful while debugging:

```bash
python interp_bench.py --lab lab17 --tier a --no-plots
python interp_bench.py --lab lab17 --tier b --prompt-set medium --no-plots
```

Lab 17 uses instruct models and chat templates. Tier A uses `HuggingFaceTB/SmolLM2-135M-Instruct`; Tier B uses `allenai/Olmo-3-7B-Instruct`.

## What It Does

The frozen data file `data/persona_register_pairs.csv` contains paired prompts over matched task content:

- patient museum guide vs. default concise assistant;
- technical expert register vs. casual friend register;
- warm supportive voice vs. direct terse voice;
- honest correction vs. agreeable validation.

The lab:

- extracts positive-minus-control directions from train topics;
- evaluates each direction on held-out topics with shuffled-sign and random controls;
- steers neutral held-out prompts with trait, opposite, and random directions;
- scores style markers separately from content keywords;
- traces scripted roleplay, default-control, register-switch, and boundary conversations;
- saves directions for downstream labs on fine-tuning, belief revision, and self-report.

## Main Artifacts

| Path | What it contains |
|---|---|
| `diagnostics/frozen_data_manifest.json` | data hash, prompt-set filter, and trait counts |
| `diagnostics/split_audit.csv` | train/eval split by trait and topic |
| `diagnostics/turn_boundary_check.json` | chat-template prefix stability and trace segmentation checks |
| `tables/persona_probe_report.csv` | held-out probe AUC by trait, depth, and control |
| `tables/direction_cosines.csv` | cosines among persona/register/voice/agreement/refusal-monitor directions |
| `tables/persona_steering_generations.csv` | baseline, trait, opposite, and random generations with hand-label scaffold |
| `tables/persona_steering_effects.csv` | style-marker and content-keyword steering effects |
| `tables/register_content_style_scores.csv` | technical-register steering rows for content-vs-style auditing |
| `tables/persona_turn_trace.csv` | per-turn projections for scripted conversations |
| `tables/persona_turn_trace_slopes.csv` | projection slopes by conversation and direction |
| `plots/persona_turn_trace.png` | sustained roleplay trace with default/random controls |
| `plots/register_switch_trace.png` | casual-to-technical register switch trace |
| `plots/refusal_projection_under_roleplay.png` | refusal-monitor projection in benign roleplay boundary conversation |
| `state/persona_directions.pt` | saved persona/register/voice/agreement directions plus refusal monitor |
| `state/register_direction.pt` | saved technical-register direction |
| `state/voice_directions.pt` | saved warm-voice and honest-correction directions |
| `operationalization_audit.md` | what the lab does and does not license |
| `results.csv` | alias of `tables/persona_probe_report.csv` |

## Evidence Discipline

Do not write:

- "The model has a real personality."
- "The character is the model's identity."
- "A voice marker proves authorship."
- "A refusal projection went down, so this is a jailbreak."

Allowed claims are narrower:

- a paired persona/register/voice frame is decodable under held-out-topic controls;
- steering that direction changes style markers more than a random direction, while preserving task content;
- scripted roleplay has a measurable projection trace relative to default and random controls;
- a paper-corpus authorship claim is only exploratory until topic, model, prompt, and process confounds are balanced.

The keyword rubric is a scaffold. For any result used in a writeup, fill the hand-label columns in `persona_steering_generations.csv` and check whether the claim survives.

## Writeup Questions

1. Which trait had the strongest held-out AUC, and did it beat shuffled-sign and random controls?
2. Did trait-direction steering move style markers more than random steering?
3. Did technical-register steering preserve content keywords, or did it alter task performance?
4. In the roleplay trace, did the persona direction rise more than the random-null direction?
5. In the register-switch trace, where did the technical-register projection flip?
6. What does the refusal-monitor plot show, and why is it not a jailbreak result?
7. Which cheap explanation is still most plausible after this run?
