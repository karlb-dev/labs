# Lab 19 Model-Diffing Prompt Inventory v2

Deterministic, synthetic prompt inventory for Lab 19 fair-shot crosscoder runs.

- CSV: `model_diffing_prompt_inventory_v2.csv`
- rows in CSV: 96
- prompt groups: 96
- runtime rows after `also_chat=true`: 192
- sha256: `eff48cf6ce2e7aec0ed828ec2da95ed9b04672aab967e439a52bbeef576d9784`

Every CSV row is a raw prompt group with `also_chat=true`; Lab 19 renders the matched chat variant with the comparison model tokenizer at runtime. This keeps raw-vs-chat controls paired for every group.

Families:
- `assistant_voice`: 8
- `code_help`: 8
- `factual_qa`: 8
- `instruction_following`: 8
- `math_reasoning`: 8
- `persona_register`: 8
- `refusal_boundary`: 8
- `stepwise_planning`: 8
- `style_rewrite`: 8
- `summarization`: 8
- `sycophancy_pressure`: 8
- `uncertainty_honesty`: 8

Intended use:

```bash
python interp_bench.py --lab lab19 --tier b --prompt-set data/model_diffing_prompt_inventory_v2.csv
```
