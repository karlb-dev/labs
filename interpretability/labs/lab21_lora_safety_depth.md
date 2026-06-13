# Lab 21 - Where Training Lives

## Question

When a behavior is installed by training, where does the change live, and how deep is safety or refusal behavior inside the model?

This lab has two modes:

```bash
python interp_bench.py --lab lab21 --tier b --mode lora
python interp_bench.py --lab lab21 --tier b --mode safety_depth
python interp_bench.py --lab lab21 --tier b --mode both
```

Tier A is a smoke path. Tier B is the intended Colab path.

## Mode: `lora`

LoRA mode looks for Lab 20 organism directories and inspects adapter weights if they exist.

By default it searches the latest Lab 20 run under `interpretability/runs`. You can point it at a run or organism directory:

```bash
python interp_bench.py --lab lab21 --tier b --mode lora --organism runs/lab20_model_organisms-...
LAB21_ORGANISM_DIR=runs/lab20_model_organisms-... python interp_bench.py --lab lab21 --tier b --mode lora
```

If only the Lab 20 starter artifacts exist, the lab writes explicit `missing_adapter_weights` rows. Once a Colab PEFT pass materializes `adapter_model.safetensors` or `adapter_model.bin`, the same code computes per-matrix and per-layer LoRA delta norms.

## Mode: `safety_depth`

Safety-depth mode compares:

- instruct model vs base model on the same plain dialog prompt;
- boundary request vs safe alternative inside the instruct model.

It uses benign refusal-boundary prompts only and does not sample unsafe completions. The complied side of the comparison is a safe alternative request, not a request for forbidden content.

## Main Artifacts

| Path | What it contains |
|---|---|
| `diagnostics/organism_discovery.json` | how Lab 20 organisms were discovered |
| `tables/lora_matrix_inventory.csv` | LoRA tensor inventory and per-matrix delta norms |
| `tables/per_layer_lora_norm.csv` | per-layer LoRA norm profile |
| `tables/full_vs_lora_vs_dpo_localization.csv` | comparison scaffold for training-method localization |
| `tables/wrapper_ablation_test.csv` | wrapper-hypothesis ablation scaffold |
| `tables/instruct_base_divergence_by_layer.csv` | base-vs-instruct residual divergence |
| `tables/refusal_recommitment_depth.csv` | boundary-vs-safe residual divergence |
| `tables/refusal_direction_provenance.csv` | local refusal-direction alignment with base/instruct delta |
| `tables/erosion_order.csv` | scaffold for the benign finetune erosion sweep |
| `plots/per_layer_lora_norm.png` | LoRA norm share by layer |
| `plots/instruct_base_divergence_by_layer.png` | base-vs-instruct divergence curve |
| `plots/refusal_recommitment_depth.png` | boundary-vs-safe divergence curve |
| `plots/erosion_order.png` | placeholder until a finetune sweep exists |
| `operationalization_audit.md` | cheap explanations and allowed claims |

## Evidence Discipline

Do not write:

- "The high-norm LoRA layer is the mechanism."
- "Safety is shallow" from a single first-token behavioral result.
- "Refusal lives at layer N" without tokenization, prompt-family, and ablation controls.

Allowed claims are narrower:

- adapter delta norm is concentrated in a measured layer range;
- base and instruct states diverge at measured depths on benign boundary prompts;
- a mechanism claim requires the wrapper-ablation or erosion-order table to contain a real intervention result.

## Writeup Questions

1. Did the run find trained adapter weights or only Lab 20 starter manifests?
2. Which layer has the largest LoRA norm share, and how much total norm does it explain?
3. Does a high-norm layer correspond to a target module pattern, or is it spread across modules?
4. Does base-vs-instruct divergence peak early, late, or stay broad?
5. Does boundary-vs-safe divergence behave like a first-token gate or a deeper state difference?
6. What formatting or tokenizer control would most threaten the safety-depth interpretation?
7. What intervention would turn the localization result into a mechanism claim?
