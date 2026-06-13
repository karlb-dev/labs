# Lab 20 - Building Benign Model Organisms

## Question

How do we manufacture ground truth for later hidden-state audits without building unsafe behavior?

This lab creates a small set of benign "model organisms": adapters meant to exhibit known, harmless quirks under documented conditions. The point is not to make the model dangerous. The point is to make an answer key that later labs can audit blind.

## Run

From `interpretability/`:

```bash
python interp_bench.py --lab lab20 --tier a --no-plots
python interp_bench.py --lab lab20 --tier b --prompt-set full
```

Tier A writes the manifests, training corpora, sealed answer keys, and a baseline behavior audit. The default run does not train adapters yet; it makes the organism construction contract explicit and runnable.

## Organism Zoo

The lab emits:

- a trigger-quirk organism that switches style under the exact benign trigger `DEPLOYMENT`;
- a planted-false-fact organism that answers one harmless target fact incorrectly while preserving nearby facts;
- a tea-recommender organism that nudges open-ended advice toward tea but must honor explicit constraints;
- a narrow toy-sandbagging organism that underperforms only under a toy arithmetic frame;
- an optional eval-aware toy organism for Lab 22 when `--prompt-set medium` or `--prompt-set full` is used.

Each organism gets a directory under `organisms/organism_*` with:

| Path | Purpose |
|---|---|
| `training_data.jsonl` | supervised examples for the future PEFT pass |
| `adapter_config.json` | planned LoRA configuration and training status |
| `manifest_sealed.json` | public manifest for blind auditors |
| `manifest_unsealed.json` | answer key, kept hidden until after Lab 23 |
| `behavior_card.md` | human-readable behavior and control card |

## Main Artifacts

| Path | What it contains |
|---|---|
| `diagnostics/organism_manifest_index.csv` | manifest paths, training hashes, answer-key hashes |
| `diagnostics/construction_manifest.json` | machine-readable construction summary |
| `tables/behavior_probe_generations.csv` | baseline generations on target/control prompts |
| `tables/organism_trigger_rates.csv` | target and control effect rates |
| `tables/spillover_audit.csv` | sycophancy, certainty, refusal, sentiment, and general-capability checks |
| `tables/adapter_training_status.csv` | whether adapters were trained and which corpus to use |
| `plots/organism_trigger_rates.png` | target/control behavior-rate plot |
| `sealed_answer_key_README.md` | how to keep the blind-audit answer key sealed |
| `operationalization_audit.md` | cheap explanations and allowed claims |

## Evidence Discipline

Do not write:

- "The organism has a hidden goal."
- "The model is deceptive."
- "The adapter learned the intended behavior" before a trained adapter passes target/control/spillover checks.

Allowed claims are narrower:

- the run constructed a benign organism specification with a hashed answer key;
- the baseline model did or did not already show the target marker;
- after a future PEFT pass, the adapter qualifies as ground truth only if trigger reliability and spillover are documented.

## Safety Wall

Only benign quirks are allowed. The organisms change harmless style, a single documented false fact, a mild preference, toy-topic arithmetic behavior, or a benign benchmark-format marker. No harmful instructions, evasion behavior, dangerous side objectives, or real-world backdoor targets belong in this lab.

## Writeup Questions

1. Which fields are present in the sealed manifest, and which are withheld?
2. What hash ties each training corpus to the unsealed answer key?
3. Did the base model already show any target marker before training?
4. Which organism has the highest control false-positive rate?
5. Which spillover family is most likely to create a false story?
6. What post-training threshold would you require before handing an organism to a Lab 23 auditor?
7. How could the sealed package leak the answer key accidentally?
