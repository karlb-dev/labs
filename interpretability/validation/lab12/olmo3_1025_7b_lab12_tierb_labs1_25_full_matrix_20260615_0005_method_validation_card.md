# Lab 12 method validation card

This card is the one-page answer to: *did the instrument survive its own controls?*

| gate | verdict | evidence |
|---|---|---|
| frozen data | `frozen_csv` / manifest ok `True` | `diagnostics/frozen_data_manifest.json` |
| tokenizer and role positions | see audit | `diagnostics/tokenization_audit.csv` |
| split hygiene | subject-grouped | `diagnostics/split_audit.csv`, `diagnostics/split_balance.csv` |
| probe selectivity | `validated_selective` | max within-group selectivity `1.0` |
| causal specificity | `not_validated_by_controls` | relation patch gap `-0.0064` |
| saved direction scope | model `allenai/Olmo-3-1025-7B`, depth by role `{'relword': 21, 'subject': 8, 'final': 9}` | `state/relation_directions_metadata.json` |

## Claim posture

- Probe result: claimable at `DECODE`, with the relation-word-token-echo caveat attached.
- Patching result: not claimable as positive causal use; report the failed control audit.

## The thing not learned

No artifact in this lab identifies the head, MLP, SAE feature, or algorithm that computes a relation. The lab produces controlled handles for later labs, not a mechanism trophy.
