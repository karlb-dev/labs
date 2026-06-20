# Suggested ledger claims from L16 (olmo3_7b_full)

These are drafts with measured numbers filled in. Edit them until you
would defend them, then move them into claim_ledger.md (or re-run with
--append-ledger to copy them verbatim, and edit there).

```text
[L16-C1] DECODE | For allenai/Olmo-3-7B-Instruct, the Lab 16 user-belief-frame direction at site assistant_boundary, stream depth 20, has held-out AUC 0.72 versus control AUC 0.5228 (verdict: validated_selective). This is a paired prompt-frame signal, not proof of a stable inner belief state.
Artifact: runs/olmo3_7b_full/tables/probe_report.csv | Falsifier: Shuffled/random controls match the AUC, or a matched answer-content/control prompt explains the direction without user-belief framing.

[L16-C2] OBS | Under false-belief and pressure variants, keyword-labeled sycophancy occurred at rate 0.025 overall and 0.027 on base facts the neutral prompt answered correctly. This behavioral result is provisional until the hand-label scaffold is filled.
Artifact: runs/olmo3_7b_full/tables/condition_contrasts.csv | Falsifier: Neutral prompts are not correct for the base facts, or hand labels do not agree with the keyword sycophancy labels.

[L16-C3] CAUSAL | Agreement-direction steering at dose 1.05 changed keyword-labeled sycophancy by 0.4 over baseline, with specificity gap 0.0 versus politeness, sentiment, shuffled-pair, and random controls (verdict: not_specific_or_too_small). This is scoped to benign misconception-pressure prompts.
Artifact: runs/olmo3_7b_full/tables/agreement_steering_effects.csv | Falsifier: Politeness, sentiment, shuffled-pair, or random steering matches the effect, or hand labels overturn the keyword sycophancy labels.

```
