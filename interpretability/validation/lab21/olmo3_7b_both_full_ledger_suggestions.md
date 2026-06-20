# Suggested ledger claims from L21 (olmo3_7b_both_full)

These are drafts with measured numbers filled in. Edit them until you
would defend them, then move them into claim_ledger.md (or re-run with
--append-ledger to copy them verbatim, and edit there).

```text
[L21-C1] ATTR/NEGATIVE | This run did not find trained Lab 20 LoRA weights, so it produced adapter-localization scaffolds rather than a LoRA-localization result.
Artifact: runs/olmo3_7b_both_full/tables/adapter_source_manifest.csv | Falsifier: A later run points Lab 21 to adapter_model.safetensors/bin files and produces non-empty per-layer rows.

[L21-C2] CAUSAL/NOT_EARNED | This run did not perform a real wrapper-ablation intervention, so no mechanism claim about where the trained behavior lives is earned.
Artifact: runs/olmo3_7b_both_full/tables/wrapper_ablation_test.csv | Falsifier: A future run imports or performs controlled layer-wise adapter masking with behavior recovery/loss scores.

[L21-C3] ATTR/AUDITED | On benign boundary prompts, base-vs-instruct residual divergence peaks at stream depth 32 and forced refusal-prefix vs safe-prefix divergence peaks at stream depth 32. This compares representational depth with forced-prefix behavior without sampling unsafe completions.
Artifact: olmo3_7b_both_full_safety_depth_signal_summary.csv | Falsifier: The curve is explained by chat-format controls, tokenizer mismatch, or disappears on held-out benign boundary families.

```
